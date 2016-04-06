# include "kvstore_chana_server.h"

namespace mxnet
{
    namespace kvstore
    {
        ChaNaPSBase * mxnet_ps_create_function(void *args)
        {
            return new KVStoreChanaServer();
        }
        
        /* virtual */ void KVStoreChanaServer::control(int cmd_id, void *data, const size_t len)
        {
            switch (cmd_id)
            {
                case (kSetOptimizer) :
                {
                    std::string optimizer((const char *)data, len);
                    exec_.Exec([this, cmd_id, optimizer]() {
                        CHECK(controller_);
                        controller_(cmd_id, optimizer);
                    });
                }
                    break;

                case (kStopServer) :
                    exec_.Stop();
                    break;

                case (kSyncMode) :
                {
                    binary_reader reader(data, len);
                    reader.read(sync_mode_);
                }
                    break;

                default:
                    assert(false);
            }
        }

                
        /* virtual */ void KVStoreChanaServer::server_process_pull(
            uint64_t *keys,
            size_t key_count,
            void **vals,
            size_t *val_sizes,
            val_deallocator_t *dealloc,
            void **args,
            bool *fixed_val_size)
        {
# if defined(KIT_PERFORMANCE_PROFILE)
            auto start_time = std::chrono::system_clock::now();
# endif
            // do some check
            CHECK_EQ(key_count, 1);

            int key = keys[0];                
            auto& stored = store_[key];                                    
            CHECK(!stored.is_none()) << "init " << key << " first";
            size_t len = stored.shape()[0];
                        
            val_sizes[0] = len * sizeof(real_t);
            vals[0] = stored.data().dptr_;
                        
            *fixed_val_size = true;
# if defined(KIT_PERFORMANCE_PROFILE)
            auto end_time = std::chrono::system_clock::now();
            auto elapse_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            pull_time_in_ms += elapse_in_ms.count();
# endif
        }


        /* virtual */ void KVStoreChanaServer::server_process_push(size_t key_count, uint64_t *keys, void **vals, size_t *valsizes)
        {
# if defined(KIT_PERFORMANCE_PROFILE)
            auto start_time = std::chrono::system_clock::now();
# endif
            CHECK_EQ(key_count, 1);            

            uint64_t key = *keys;
            real_t *val = static_cast<real_t *>(*vals);
            size_t len = *valsizes / sizeof(real_t);
            auto &stored = store_[key];

            // there used several WaitToRead, this is because \a recved's memory
            // could be deallocated when this function returns. so we need to make sure
            // the operators with \a NDArray are actually finished

            size_t ds[] = { len };
            TShape dshape(ds, ds + 1);
            TBlob recv_blob(val, dshape, cpu::kDevMask);
            NDArray recved = NDArray(recv_blob, 0);
            if (stored.is_none())
            {
                // initialization
                stored = NDArray(dshape, Context());
                CopyFromTo(recved, &stored, 0);
                stored.WaitToRead();
            }
            else if (sync_mode_)
            {
                // synced push
                auto &merged = merge_buf_[key];
                if (merged.array.is_none()) merged.array = NDArray(dshape, Context());

                if (merged.count == 0) CopyFromTo(recved, &merged.array, 0);
                else merged.array += recved;                

                if (++merged.count == NumWorkers())
                {
                    // let the main thread to execute updater_, which is necessary for python
                    exec_.Exec([this, key, &merged, &stored]()
                    {
                        CHECK(updater_);
                        updater_(key, merged.array, &stored);
                    });

                    merged.count = 0;
                    stored.WaitToRead();
                }
                else 
                {
                    merged.array.WaitToRead();
                }
            }
            else 
            {            
                // async push
                exec_.Exec([this, key, &recved, &stored]()
                {
                    CHECK(updater_);
                    updater_(key, recved, &stored);
                });
                
                stored.WaitToRead();
            }

# if defined(KIT_PERFORMANCE_PROFILE)
            auto end_time = std::chrono::system_clock::now();
            auto elapse_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            push_time_in_ms += elapse_in_ms.count();
# endif
        }
    }
}