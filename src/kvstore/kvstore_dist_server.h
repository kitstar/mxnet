/*!
 * Copyright (c) 2015 by Contributors
 * \file mxnet_node.h
 * \brief implement mxnet nodes
 */
#ifndef MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
#define MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
#include <queue>
#include <string>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <functional>
#include <future>
#include <vector>
#include "ps/ps.h"
#include "mxnet/kvstore.h"

namespace mxnet {
namespace kvstore {

static const int kStopServer = -1;
static const int kSyncMode = -2;

/**
 * \brief executor runs a function using the thread called \ref Start
 */
class Executor {
 public:
  /**
   * \brief start the executor
   */
  void Start() {
      exec_time_in_ms = 0;
      std::unique_lock<std::mutex> lk(mu_);
    while (true) {
      cond_.wait(lk, [this]{return !queue_.empty();});
      Block blk = std::move(queue_.front());
      queue_.pop();
      lk.unlock();

      auto start_time = std::chrono::system_clock::now();      
      if (blk.f) {
        blk.f(); blk.p->set_value();
      } else {
        blk.p->set_value(); break;
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
  void Exec(const Func& func) {
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
  void Stop() {
    Exec(Func());    
  }

 private:
  struct Block {
  explicit Block(const Func& func) : f(func), p(std::make_shared<std::promise<void>>()) { }
    Func f;
    std::shared_ptr<std::promise<void>> p;
  };
  std::queue<Block> queue_;
  std::mutex mu_;
  std::condition_variable cond_;

  int64_t exec_time_in_ms;
};

class KVStoreDistServer {
 public:
  KVStoreDistServer() {
# if defined(KIT_PERFORMANCE_PROFILE)
      pull_time_in_ms = push_time_in_ms = 0;
      pull_packet_count = push_packet_count = 0;
      pull_packet_total_size_in_byte = push_packet_total_size_in_byte = 0;
# endif

    using namespace std::placeholders;
    ps_server_ = new ps::KVServer<float>(0);
    static_cast<ps::SimpleApp*>(ps_server_)->set_request_handle(
        std::bind(&KVStoreDistServer::CommandHandle, this, _1, _2));
    ps_server_->set_request_handle(
        std::bind(&KVStoreDistServer::DataHandle, this, _1, _2, _3));
    sync_mode_ = false;
  }

  ~KVStoreDistServer() {
      std::cout << "Server Statistics Information" << std::endl;
      std::cout << "\tkey count : [" << store_.size() << "]" << std::endl;
# if defined(KIT_PERFORMANCE_PROFILE)
      std::cout << "\tpull cost time: [" << pull_time_in_ms / 1000.0 << "s]" << std::endl;
      std::cout << "\tpush cost time: [" << push_time_in_ms / 1000.0 << "s]" << std::endl;
      std::cout << "\tpull packet: [" << pull_packet_count << "]" << std::endl;
      std::cout << "\tpush packet: [" << push_packet_count << "]" << std::endl;
      std::cout << "\tpull size: [" << pull_packet_total_size_in_byte / 1024.0 / 1024 << "MB]" << std::endl;
      std::cout << "\tpush size: [" << push_packet_total_size_in_byte / 1024.0 / 1024 << "MB]" << std::endl;
# endif
      
      delete ps_server_;
  }

  void set_controller(const KVStore::Controller& controller) {
    CHECK(controller);
    controller_ = controller;
  }

  void set_updater(const KVStore::Updater& updater)  {
    CHECK(updater);
    updater_ = updater;
  }

  /**
   * \brief blocked until received the command \a kSyncMode
   */
  void Run() {
    exec_.Start();
  }

 private:
  void CommandHandle(const ps::SimpleData& recved, ps::SimpleApp* app) {
    if (recved.head == kStopServer) {
      exec_.Stop();
    } else if (recved.head == kSyncMode) {
      sync_mode_ = true;
    } else {
      // let the main thread to execute ctrl, which is necessary for python
      exec_.Exec([this, recved]() {
          CHECK(controller_);
          controller_(recved.head, recved.body);
        });
    }
    app->Response(recved);
  }

  void DataHandle(const ps::KVMeta& req_meta,
                  const ps::KVPairs<real_t>& req_data,
                  ps::KVServer<real_t>* server) {
# if defined(KIT_PERFORMANCE_PROFILE)
      auto start_time = std::chrono::system_clock::now();
# endif
      
    // do some check
    CHECK_EQ(req_data.keys.size(), (size_t)1);
    if (req_meta.push) {
      CHECK_EQ(req_data.lens.size(), (size_t)1);
      CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
    }

    int key = DecodeKey(req_data.keys[0]);
    auto& stored = store_[key];

    // there used several WaitToRead, this is because \a recved's memory
    // could be deallocated when this function returns. so we need to make sure
    // the operators with \a NDArray are actually finished
    if (req_meta.push) {
      size_t ds[] = {(size_t)req_data.lens[0]};
      TShape dshape(ds, ds + 1);
      TBlob recv_blob((real_t*)req_data.vals.data(), // NOLINT(*)
                      dshape, cpu::kDevMask);
      NDArray recved = NDArray(recv_blob, 0);

# if defined(KIT_PERFORMANCE_PROFILE)
      ++push_packet_count;
      push_packet_total_size_in_byte += dshape.Size() * sizeof(real_t);
# endif

      if (stored.is_none()) {
        // initialization
        stored = NDArray(dshape, Context());
        CopyFromTo(recved, &stored, 0);
        server->Response(req_meta);
        stored.WaitToRead();
      } else if (sync_mode_) {
        // synced push
        auto& merged = merge_buf_[key];
        if (merged.array.is_none()) {
          merged.array = NDArray(dshape, Context());
        }

        if (merged.request.size() == 0) {
          CopyFromTo(recved, &merged.array, 0);
        } else {
          merged.array += recved;
        }

        merged.request.push_back(req_meta);

        if (merged.request.size() == (size_t)ps::NumWorkers()) {
          // let the main thread to execute updater_, which is necessary for
          // python
          exec_.Exec([this, key, &merged, &stored](){
              CHECK(updater_);
              updater_(key, merged.array, &stored);
            });
          for (const auto& req : merged.request) {
            server->Response(req);
          }
          merged.request.clear();
          stored.WaitToRead();
        } else {
          merged.array.WaitToRead();
        }
      } else {
        // async push
        exec_.Exec([this, key, &recved, &stored](){
            CHECK(updater_);
            updater_(key, recved, &stored);
          });
        server->Response(req_meta);
        stored.WaitToRead();
      }

# if defined(KIT_PERFORMANCE_PROFILE)
      auto end_time = std::chrono::system_clock::now();
      auto elapse_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);      
      push_time_in_ms += elapse_in_ms.count();
# endif
    } else {
      // pull
      ps::KVPairs<real_t> response;
      CHECK(!stored.is_none()) << "init " << key << " first";
      int len = stored.shape()[0];
      response.keys = req_data.keys;
      response.lens = {len};
      response.vals.CopyFrom(static_cast<const float*>(stored.data().dptr_), len);
      server->Response(req_meta, response);

# if defined(KIT_PERFORMANCE_PROFILE)
      ++pull_packet_count;
      pull_packet_total_size_in_byte += len * sizeof(real_t);      

      auto end_time = std::chrono::system_clock::now();
      auto elapse_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
      pull_time_in_ms += elapse_in_ms.count();
# endif
    }
  }

  int DecodeKey(ps::Key key) {
    auto kr = ps::Postoffice::Get()->GetServerKeyRanges()[ps::MyRank()];
    return key - kr.begin();
  }

  /**
   * \brief user defined
   */
  bool sync_mode_;
  KVStore::Controller controller_;
  KVStore::Updater updater_;

  std::unordered_map<int, NDArray> store_;

  struct MergeBuf {
    std::vector<ps::KVMeta> request;
    NDArray array;
  };
  std::unordered_map<int, MergeBuf> merge_buf_;

  Executor exec_;

  ps::KVServer<float>* ps_server_;

# if defined(KIT_PERFORMANCE_PROFILE)
  int64_t pull_time_in_ms;
  int64_t push_time_in_ms;
  uint64_t pull_packet_count;
  uint64_t push_packet_count;
  size_t pull_packet_total_size_in_byte;
  size_t push_packet_total_size_in_byte;
# endif
};



}  // namespace kvstore
}  // namespace mxnet

#endif  // MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
