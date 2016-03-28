/*!
 * Copyright (c) 2015 by Contributors
 * \file kvstore.cc
 * \brief implement kv_store
 */
#include <mxnet/kvstore.h>
#include <stdlib.h>
#include <dmlc/logging.h>
#include "./kvstore_local.h"
#include "./kvstore_device.h"
# if MXNET_USE_DIST_CHANA
# include "./kvstore_chana.h"
# elif MXNET_USE_DIST_KVSTORE
#include "./kvstore_dist.h"
#endif  // MXNET_USE_DIST_KVSTORE

namespace mxnet {

KVStore* KVStore::Create(const char *type_name) {
  std::string tname = type_name;
  std::transform(tname.begin(), tname.end(), tname.begin(), ::tolower);
  KVStore* kv = nullptr;
  if (tname == "local" ||
      tname == "local_update_cpu" ||
      tname == "local_allreduce_cpu") {
    kv =  new kvstore::KVStoreLocal();
  } else if (tname == "device" ||
             tname == "local_allreduce_device") {
    tname = "local_allreduce_device";
    kv = new kvstore::KVStoreDevice();
  } else if (tname.substr(0, 10) == "dist_async" ||
             tname.substr(0, 9) == "dist_sync" ||
             tname.substr(0, 4) == "dist") {      
# if MXNET_USE_DIST_CHANA
      {
          auto machine_start = tname.find("#");
          auto ps_start = tname.find('#', machine_start + 1);
          std::string machine_list = tname.substr(machine_start + 1, ps_start - machine_start - 1);
          int ps_per_machine = atoi(tname.substr(ps_start + 1).c_str());
          kv = new kvstore::KVStoreChana(machine_list.c_str(), ps_per_machine);

          if (tname.substr(0, 9) == "dist_sync" && kv->get_rank() == 0)
          {
              kv->SendCommandToServers(kvstore::kSyncMode, std::to_string(true));
          }
      }
# elif MXNET_USE_DIST_KVSTORE
    kv = new kvstore::KVStoreDist();
    if (tname == "dist_sync" &&
        kv->IsWorkerNode() &&
        kv->get_rank() == 0) {
      // configure the server to be the sync mode
      kv->SendCommandToServers(kvstore::kSyncMode, "\1");
    }
# else
    LOG(FATAL) << "compile with USE_DIST_KVSTORE=1 to use " << tname;
    return nullptr;
#endif  // MXNET_USE_DIST_KVSTORE
  } else {
    LOG(FATAL) << "Unknown KVStore type \"" << tname << "\"";
  }
  kv->type_ = tname;
  return kv;
}

}  // namespace mxnet
