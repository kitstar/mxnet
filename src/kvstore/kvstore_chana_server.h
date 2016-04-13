/*!
* Copyright (c) 2015 by Contributors
* \file kvstore_chana_server.h
* \brief implement mxnet nodes
*/
# pragma once

#include <queue>
#include <string>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <functional>
#include <future>
#include <vector>
#include "chana_ps.h"
#include "mxnet/kvstore.h"

namespace mxnet 
{
    namespace kvstore 
    {
        static const int kSetOptimizer = 0;
        static const int kStopServer = 1;
        static const int kSyncMode = 2;

        /**
        * \brief executor runs a function using the thread called \ref Start
        */
        class Executor
        {
        public:
            /**
            * \brief start the executor
            */
            void Start()
            {
                exec_time_in_ms = 0;
                std::unique_lock<std::mutex> lk(mu_);
                while (true)
                {
                    cond_.wait(lk, [this] {return !queue_.empty(); });
                    Block blk = std::move(queue_.front());
                    queue_.pop();
                    lk.unlock();

                    auto start_time = std::chrono::system_clock::now();
                    if (blk.f)
                    {
                        blk.f();
                        blk.p->set_value();
                    }
                    else
                    {
                        blk.p->set_value(); 
                        break;
                    }

                    auto end_time = std::chrono::system_clock::now();
                    auto elapse_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                    exec_time_in_ms += elapse_in_ms.count();
                    lk.lock();
                }

                std::cout << "Executor Time: " << exec_time_in_ms / 1000.0 << "s" << std::endl;
            }

            /**
            * \brief function
            */
            typedef std::function<void()> Func;

            /**
            * \brief let the thread called \ref Start to exec a function. threadsafe
            */
            void Exec(const Func& func)
            {
                Block blk(func);
                auto fut = blk.p->get_future();
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    queue_.push(std::move(blk));
                    cond_.notify_one();
                }
                fut.wait();
            }

            /**
            * \brief stop the thread, threadsafe
            */
            void Stop()
            {
                Exec(Func());
            }

        private:
            struct Block
            {
                explicit Block(const Func& func) : f(func), p(std::make_shared<std::promise<void>>()) { }
                Func f;
                std::shared_ptr<std::promise<void>> p;
            };

            std::queue<Block> queue_;
            std::mutex mu_;
            std::condition_variable cond_;

            int64_t exec_time_in_ms;
        };


        class binary_writer
        {
        public:
            binary_writer(size_t buffer_size) : m_own_buf(true)
            {
                reallocate(buffer_size);
            }

            binary_writer(void *data_, size_t size_) : m_own_buf(false)
            {
                m_current = m_head = data_;
                m_tail = reinterpret_cast<size_t>(m_head)+size_;
            }

            ~binary_writer() 
            { if (m_own_buf) free(m_head); }

            template<typename T> bool write(const T &val)
            {
                assert(false);                
                return false;
            }

            inline void set_own_buf(bool _own) { m_own_buf = _own; }

            inline void * data() { return m_head; }

            inline void reset() { m_current = m_head; }

            inline bool empty() { return (m_head == m_current); }

            inline bool is_enough(size_t data_size) { return reserve() >= data_size; }

            inline size_t reserve() { return m_tail - reinterpret_cast<size_t>(m_current); }

            inline size_t size() { return reinterpret_cast<size_t>(m_current)-reinterpret_cast<size_t>(m_head); }

            inline size_t capacity() { return m_tail - reinterpret_cast<size_t>(m_head); }

            // Warning: you need to free old buffer by yourself!!!
            inline bool reallocate(size_t buffer_size)
            {
                m_current = m_head = malloc(buffer_size);
                m_tail = reinterpret_cast<size_t>(m_head)+buffer_size;
                return (m_head != NULL);
            }

            bool write(const uint64_t &data) { return write_pod(data); }
            bool write(const uint32_t &data) { return write_pod(data); }
            bool write(const uint16_t &data) { return write_pod(data); }
            bool write(const uint8_t &data) { return write_pod(data); }
            bool write(const int64_t &data) { return write_pod(data); }
            bool write(const int32_t &data) { return write_pod(data); }
            bool write(const int16_t &data) { return write_pod(data); }
            bool write(const int8_t &data) { return write_pod(data); }
            bool write(const double &data) { return write_pod(data); }
            bool write(const float &data) { return write_pod(data); }            

            // Warning: assume the vector.size() <= uint32_t, and the elements in the vector are POD
            template<typename T> bool write(const std::vector<T> &data)
            {
                assert(std::is_pod<T>::value);
                write(static_cast<uint32_t>(data.size()));  // For Saving the buffer
                return write_pod((char *)(&(data[0])), sizeof(T) * data.size());
            }

            template<typename T> bool write_pod(const T &data)
            {
                assert(is_enough(sizeof(T)));
                memcpy(m_current, &data, sizeof(T));
                m_current = reinterpret_cast<void *>(reinterpret_cast<uint64_t>(m_current)+sizeof(T));
                return true;
            }

            bool write_pod(void *data, size_t size)
            {
                assert(is_enough(size));
                memcpy(m_current, data, size);
                m_current = reinterpret_cast<void *>(reinterpret_cast<uint64_t>(m_current)+size);
                return true;
            }

            bool skip(size_t size)
            {
                assert(is_enough(size));
                m_current = reinterpret_cast<void *>(reinterpret_cast<uint64_t>(m_current)+size);
                return true;
            }

        private:
            void    *m_head;
            void    *m_current;
            size_t  m_tail;
            bool    m_own_buf;
        };


        class binary_reader
        {
        public:
            binary_reader(void *data_, size_t size_)
            {
                m_current = m_head = data_;
                m_tail = reinterpret_cast<size_t>(m_head) + size_;
            }

            inline void reset() { m_current = m_head; }

            inline void * data() { return m_head; }

            inline void * current() { return m_current; }

            inline bool empty() { return (m_head == m_current); }

            inline bool is_enough(size_t data_size) { return reserve() >= data_size; }

            inline size_t reserve() { return m_tail - reinterpret_cast<size_t>(m_current); }

            inline size_t size() { return reinterpret_cast<size_t>(m_current)-reinterpret_cast<size_t>(m_head); }

            inline size_t capacity() { return m_tail - reinterpret_cast<size_t>(m_head); }

            bool read(uint64_t &data) { return read_pod(data); }
            bool read(uint32_t &data) { return read_pod(data); }
            bool read(uint16_t &data) { return read_pod(data); }
            bool read(uint8_t &data) { return read_pod(data); }
            bool read(int64_t &data) { return read_pod(data); }
            bool read(int32_t &data) { return read_pod(data); }
            bool read(int16_t &data) { return read_pod(data); }
            bool read(int8_t &data) { return read_pod(data); }
            bool read(double &data) { return read_pod(data); }
            bool read(bool &data) { return read_pod(data); }

            template<typename T> bool read(const T &data)
            {
                assert(false);
                return false;
            }

            template<typename T> bool read_pod(const T &data)
            {
                if (!is_enough(sizeof(T))) return false;
                memcpy((char*)&data, m_current, sizeof(T));
                m_current = reinterpret_cast<void *>(reinterpret_cast<uint64_t>(m_current) + sizeof(T));
                return true;
            }

            bool read_pod(void *data, size_t size)
            {
                if (!is_enough(size)) return false;
                memcpy(data, m_current, size);
                m_current = reinterpret_cast<void *>(reinterpret_cast<uint64_t>(m_current) + size);
                return true;
            }

        private:
            void        *m_head;
            void        *m_current;
            size_t      m_tail;
        };        


        struct ChanaCallbackContext
        {
            Engine::CallbackOnComplete cb;
        };

        
        ChaNaPSBase * mxnet_ps_create_function(void *args);

        
        class KVStoreChanaServer : public ChaNaPSBase
        {
        private:
            struct MergeBuf
            {
                MergeBuf() : count(0) { }
                int     count;
                NDArray array;
            };


        public:
            KVStoreChanaServer() : sync_mode_(false)
            {
                work_thread_ = std::unique_ptr<std::thread>(new std::thread(&Executor::Start, &exec_));
# if defined(KIT_PERFORMANCE_PROFILE)
                pull_time_in_ms = push_time_in_ms = 0;
                pull_packet_count = push_packet_count = 0;
                pull_packet_total_size_in_byte = push_packet_total_size_in_byte = 0;
# endif
            }

            virtual void control(int cmd_id, void *data, const size_t len);            
            
            virtual void server_process_pull(
                uint64_t *keys,
                size_t key_count,
                void **vals,
                size_t *val_sizes,
                val_deallocator_t *dealloc,
                void **args,
                bool *fixed_val_size);
            
            virtual void server_process_push(size_t key_count, uint64_t *keys, void **vals, size_t *valsizes);

            virtual void worker_apply_pull(void *args)
            {
                auto ctx = reinterpret_cast<ChanaCallbackContext *>(args);
                ctx->cb();
                delete ctx;
            }

            void set_controller(const KVStore::Controller &controller) 
            {
                CHECK(controller);
                controller_ = controller;
            }

            void set_updater(const KVStore::Updater &updater)  
            {
                CHECK(updater);
                updater_ = updater;
            }

            inline bool get_sync_mode() { return sync_mode_; }

        private:
            inline int NumWorkers()
            {                
                return GetMachineCount();
            }
            
            int DecodeKey(uint64_t key) 
            {
                // No implement
                assert(false);
            }

            
        public:
# if defined(KIT_PERFORMANCE_PROFILE)
            int64_t pull_time_in_ms;
            int64_t push_time_in_ms;
            uint64_t pull_packet_count;
            uint64_t push_packet_count;
            size_t pull_packet_total_size_in_byte;
            size_t push_packet_total_size_in_byte;
# endif
        
        private:            
            std::unordered_map<uint64_t, NDArray>   store_; // Model
            std::unordered_map<int, MergeBuf>       merge_buf_; // For sync mode
            bool                                    sync_mode_;
            KVStore::Controller                     controller_;
            KVStore::Updater                        updater_;                                    
            Executor                                exec_;
            std::unique_ptr<std::thread>            work_thread_;
        };        
    }  // namespace kvstore
}  // namespace mxnet