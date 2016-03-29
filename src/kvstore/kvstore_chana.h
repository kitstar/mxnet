/**
* Copyright (c) 2016 by Contributors
* @file   kvstore_chana.h
* @brief  distributed implementation based on chana
*/
# pragma once
# include <string>
# include <vector>
# include "./kvstore_local.h"
# include "mxnet/engine.h"
# include "kvstore_chana_server.h"

namespace mxnet
{
    namespace kvstore
    {
        /**
        * \brief distributed kvstore
        *
        * for a worker node, it always guarantees that all push and pull issued from
        * this worker on the same key are serialized. namely push(3) and then pull(3),
        * then the data pulled is always containing the modification from the push(3).
        *
        * it's the server node's job to control the data consistency among all
        * workers. see details on \ref ServerHandle::Start
        */
        
        void chana_push_callback(size_t count, uint64_t *keys, void **vals, size_t *val_sizes, void *args);

        void free_chana_cmd_callback(void *args);

        class KVStoreChana : public KVStoreLocal
        {        
        public:
            class ChanaCmdContxt
            {
            public:
                ChanaCmdContxt(int cmd_id, const std::string &cmd_body)
                {
                    size = sizeof(uint32_t) + sizeof(int) + cmd_body.length();
                    assert(size <= 500 * 1024);

                    data = malloc(size);
                    binary_writer writer(data, size);
                    writer.write(cmd_id);
                    bool ret = writer.write_pod((void *)cmd_body.c_str(), cmd_body.length());
                    assert(ret);
                }

                ~ChanaCmdContxt() { free(data); }

            public:
                void    *data;
                size_t  size;
            };

        public:
            KVStoreChana(const char *machine_list_for_chana, const int ps_per_machine);

            virtual ~KVStoreChana()
            {
                Engine::Get()->WaitForAll();
                if (get_rank() == 0)
                {                
                    // stop the executor at servers
                    SendCommandToServers(kStopServer, "");
                    ChaNaPSWait();
                }                
                Barrier();
            }

            void Init(const std::vector<int>& keys,
                const std::vector<NDArray>& values) override;

            virtual void Push(const std::vector<int>& keys,
                const std::vector<NDArray>& values,
                int priority) override;


            virtual void Pull(const std::vector<int>& keys,
                const std::vector<NDArray*>& values,
                int priority) override;

            virtual void set_updater(const Updater& updater) override;            

            void Barrier() override 
            {
                BarrierEnter();
            }


            virtual void SendCommandToServers(int cmd_id,
                const std::string& cmd_body) override;            

            int get_group_size() const override
            {                
                return GetMachineCount();
            }

            int get_rank() const override
            {
                return GetMyRank();
            }

            virtual void RunServer(const Controller& controller) override;            

        private:
            /**
            * \brief Wait until all pushes and pulls issued on each key have been
            * finished
            *
            * \param keys a list of keys
            */
            void Wait(const std::vector<int>& keys);


            /**
            * \brief check if the keys are all unique
            */
            void CheckUnique(const std::vector<int>& keys);


            /**
            * \brief struct for ps keys and lens
            */
            struct ChanaKV
            {
                std::vector<uint64_t>   keys;
                std::vector<void *>     vals;
                std::vector<size_t>     lens;
                size_t                  size;
            };
    

            /**
            * \brief cache all key partitions
            */
            std::unordered_map<int, ChanaKV> ps_kv_;

            /**
            * \brief serizelize EncodeKey
            */
            std::mutex mu_;

            /**
            * \brief convert to keys in chana
            */
            /* inline */ ChanaKV & EncodeKey(int key, real_t *val, size_t size);

            inline uint64_t get_ps_key(int key, int num_servers)
            {
                return (key << id_bit) | num_servers;
            }

            int num_servers;
            uint64_t id_bit;
            int ps_per_machine;
            bool sync_mode;
        };

    }  // namespace kvstore
}  // namespace mxnet