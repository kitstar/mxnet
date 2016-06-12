# if defined(MXNET_USE_DIST_CHANA)

# include "kvstore_chana.h"
# include "windows.h"

namespace mxnet
{
    namespace kvstore
    {
        void chana_push_callback(size_t count, uint64_t *keys, void **vals, size_t *val_sizes, void *args)
        {
            auto ctx = reinterpret_cast<ChanaCallbackContext *>(args);
            ctx->cb();
            delete ctx;
        }

        void free_chana_cmd_callback(void *args)
        {
            auto ctx = reinterpret_cast<KVStoreChana::ChanaCmdContxt *>(args);
            delete ctx;
        }

        KVStoreChana::KVStoreChana(const char *machine_list_for_chana, const int _ps_per_machine) : num_servers(0), id_bit(0), ps_per_machine(_ps_per_machine), sync_mode(false)
        {
            bool use_rdma = false;
                        
            std::cout << "pid = " << ::GetCurrentProcessId() << std::endl;            

            auto use_rdma_str = getenv("CHANA_USE_RDMA");
            if (use_rdma_str != nullptr)
            {
                std::string ret(use_rdma_str);                
                std::transform(ret.begin(), ret.end(), ret.begin(), std::tolower);
                use_rdma = (ret == "true");
            }

            auto chana_config = getenv("CHANA_CONFIG_FILE");
            if (chana_config != nullptr)
            {
                std::string config(chana_config);                
                config = "config=" + config;
                const char *argv[] = { config.c_str() };
                chana_initialize(1, argv);
            }
            else
            {
                chana_initialize(0, nullptr);
            }

            CreateParameterServer(machine_list_for_chana, ps_per_machine, use_rdma, mxnet_ps_create_function, nullptr);

            num_servers = GetMachineCount() * ps_per_machine;
            while ((1 << id_bit) < num_servers) ++id_bit;
        }

        /* virtual */ void KVStoreChana::RunServer(const Controller& controller)
        {
            for (size_t i = 0; i < ps_per_machine; ++i)
            {                
                ((KVStoreChanaServer *)ChaNaPSGetInstance(i))->set_controller(controller);
            }
            sync_mode = ((KVStoreChanaServer *)ChaNaPSGetInstance(0))->get_sync_mode();

            if (sync_mode == true)
            {
                LG << "Running Kvstore ChaNa on " << get_group_size() << " at Sync mode.";
            }
            else
            {
                LG << "Running Kvstore ChaNa on " << get_group_size() << " at Async mode.";
            }

            Barrier();
        }

        /* virtual */ void KVStoreChana::set_updater(const Updater& updater)
        {
            // Kit: TODO avoid setting more than once!
            CHECK(updater) << "invalid updater";
            for (size_t i = 0; i < ps_per_machine; ++i)
            {
                ((KVStoreChanaServer *)ChaNaPSGetInstance(i))->set_updater(updater);
            }                
            updater_ = updater;
        }
                
        
        void KVStoreChana::Init(const std::vector<int>& keys,
            const std::vector<NDArray>& values)
        {
            CheckUnique(keys);
            if (get_rank() == 0)
            {
                Push(keys, values, 0);
                // wait until the push is finished
                if (!sync_mode) Wait(keys);
            }
            else
            {
                if (sync_mode) Barrier();
            }

            Barrier();
        }
        
        
        void KVStoreChana::Wait(const std::vector<int>& keys)
        {
            for (int key : keys)
            {
                auto it = merge_buf_.find(key);
                CHECK(it != merge_buf_.end())
                    << "there is no push/pull on key " << key << " before";
                CHECK(!it->second.merged.is_none())
                    << "there is no push/pull on key " << key << " before";
                it->second.merged.WaitToWrite();
            }
            ChaNaPSWait();
        }


        /* virtual */ void KVStoreChana::Push(const std::vector<int>& keys,
            const std::vector<NDArray>& values,
            int priority)
        {
            // first aggregate the values over keys
            std::vector<int> uniq_keys;
            std::vector<std::vector<NDArray> > grouped_vals;
            GroupKVPairs(keys, values, &uniq_keys, &grouped_vals);

            for (size_t i = 0; i < uniq_keys.size(); ++i)
            {
                // merge over devcies
                int key = uniq_keys[i];
                const NDArray &merged = MergePushValue(key, grouped_vals[i], priority);

                // push to servers
                auto push_to_servers = [this, key, merged](RunContext rctx, Engine::CallbackOnComplete cb)
                {
                    // convert to ps keys
                    size_t size = merged.shape().Size();                    

                    // do push
                    auto data = static_cast<real_t *>(merged.data().dptr_);
                    auto &pskv = EncodeKey(key, data, size);

                    auto ctx = new ChanaCallbackContext;
                    ctx->cb = cb;

                    ChaNaPSPush(pskv.keys.size(), pskv.keys.data(), pskv.vals.data(), pskv.lens.data(), chana_push_callback, ctx);
                };

                Engine::Get()->PushAsync(
                    push_to_servers,
                    pinned_ctx_,
                    { merged.var() },
                    {},
                    FnProperty::kNormal, priority);
            }

            if (sync_mode)
            {
                Wait(keys);
                Barrier();
            }
        }


        /* virtual */ void KVStoreChana::Pull(const std::vector<int> &keys,
            const std::vector<NDArray*> &values,
            int priority)
        {
            std::vector<int> uniq_keys;
            std::vector<std::vector<NDArray*> > grouped_vals;
            GroupKVPairs(keys, values, &uniq_keys, &grouped_vals);

            for (size_t i = 0; i < uniq_keys.size(); ++i)
            {
                int key = uniq_keys[i];
                const auto& vals = grouped_vals[i];

                // first pull to a buffer. we reuse the merge buf so that all pushes and
                // pulls on the same key on the local machine are always sequentials
                auto& buf = merge_buf_[key].merged;
                if (buf.is_none()) 
                {
                    buf = NDArray(vals[0]->shape(), pinned_ctx_);
                }
                real_t *data = static_cast<real_t*>(buf.data().dptr_);
                size_t size = buf.shape().Size();

                auto pull_from_servers = [this, key, data, size](
                    RunContext rctx, Engine::CallbackOnComplete cb) 
                {
                    // convert to ps keys                    
                    auto &pskv = EncodeKey(key, data, size);

                    // issue pull
                    auto ctx = new ChanaCallbackContext;
                    ctx->cb = cb;
                    ChaNaPSPull(pskv.keys.size(), pskv.keys.data(), pskv.vals.data(), pskv.lens.data(), ctx);
                };

                CHECK_NOTNULL(Engine::Get())->PushAsync(
                    pull_from_servers,
                    pinned_ctx_,
                    {},
                    { buf.var() },
                    FnProperty::kNormal, priority);

                // copy data from buffer to vals
                for (auto v : vals) {
                    CopyFromTo(buf, v);
                }
            }
        }

        /* virtual */ void KVStoreChana::SendCommandToServers(int cmd_id, const std::string &cmd_body)
        {                        
            auto ctx = new ChanaCmdContxt(cmd_id, cmd_body);                        
            ChaNaPSControl(PS_ROLE_ALL, ctx->data, ctx->size, free_chana_cmd_callback, ctx);
            ChaNaPSWait();
        }
        
        
        void KVStoreChana::CheckUnique(const std::vector<int> &keys)
        {
            auto keys_copy = keys;
            auto last = std::unique(keys_copy.begin(), keys_copy.end());
            CHECK_EQ(static_cast<size_t>(std::distance(keys_copy.begin(), last)),
                static_cast<size_t>(keys.size()));
        }
        

        KVStoreChana::ChanaKV & KVStoreChana::EncodeKey(int key, real_t *val, size_t size)
        {
            mu_.lock();
            auto &pskv = ps_kv_[key];
            mu_.unlock();

            if (!pskv.keys.empty()) 
            {
                CHECK_EQ(static_cast<size_t>(pskv.size), size) << "The value size cannot be changed";
            }
            else
            {
                // a simple heuristic for load balance
                if (size < bigarray_bound_) {
                    // send it to a single random picked server
                    int server = key % num_servers;
                    uint64_t ps_key = get_ps_key(key, server);
                    pskv.keys.push_back(ps_key);
                    pskv.vals.push_back(val);
                    pskv.lens.push_back(size * sizeof(real_t));
                    pskv.size = size;
                }
                else 
                {
                    // parition it to all servers
                    pskv.size = 0;
# if !defined(NDEBUG)
                    uint64_t server_sum = 0;
# endif
                    for (int i = 0; i < num_servers; ++i) 
                    {
                        size_t part_size =
                            static_cast<size_t>(static_cast<double>(size) / num_servers * (i + 1)) -
                            static_cast<size_t>(static_cast<double>(size) / num_servers * i);
                        uint64_t ps_key = get_ps_key(key, i);
# if !defined(NDEBUG)
                        server_sum += ps_key % num_servers;
# endif
                        pskv.keys.push_back(ps_key);
                        pskv.vals.push_back(val + (i * part_size));
                        pskv.lens.push_back(part_size * sizeof(real_t));
                        pskv.size += part_size;
                    }
                    CHECK_EQ(static_cast<size_t>(pskv.size), size);                    
# if !defined(NDEBUG)                    
                    CHECK_LT(abs(server_sum - (num_servers - 1) * num_servers / 2.0), 1e-7);
# endif
                }
            }
            return pskv;
        }
    }
}

# endif