/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_FLOWRM_FLOWGW_CLIENT_H_
#define AIR_RUNTIME_HETEROGENEOUS_FLOWRM_FLOWGW_CLIENT_H_

#include <string>
#include "ge/ge_api_error_codes.h"
#include "aicpu/queue_schedule/dgw_client.h"
#include "aicpu/queue_schedule/qs_client.h"
#include "common/mem_grp/memory_group_manager.h"
#include "graph_metadef/common/ge_common/util.h"
#include "common/config/json_parser.h"
#include "common/subprocess/subprocess_manager.h"
#include "proto/deployer.pb.h"

namespace ge {
enum class GrantType {
  kReadOnly = 0,
  kWriteOnly,
  kReadAndWrite,
  kInvalid
};

enum class ExchangeEndpointType : int32_t {
  kEndpointTypeNone = -1,
  kEndpointTypeExternalQueue = 0,
  kEndpointTypeQueue,
  kEndpointTypeTag,
  kEndpointTypeGroup,
  kEndpointTypeRefQueue,
  kEndpointTypeDummyQueue,
};

struct ExchangeEndpoint {
  uint32_t id = UINT32_MAX;
  ExchangeEndpointType type = ExchangeEndpointType::kEndpointTypeExternalQueue;
  uint64_t hcom_handle = UINT64_MAX;
  int32_t device_id = -1;
  int32_t device_type = -1;
  std::string name;
  std::vector<int32_t> endpoint_indices;
  uint32_t tag_id;
  uint32_t peer_tag_id;
  uint32_t rank_id;
  uint32_t peer_rank_id;
  int32_t fusion_offset;
  uint32_t instance_num;
  uint32_t instance_idx;
  uint32_t depth;
  uint32_t model_id;
  // src group use for whether need keep order, is_dynamic_sched=true means no need keep order.
  // dst group use for whether dynamic schedule.
  bool is_dynamic_sched;
  uint32_t root_model_id;
  int32_t index;
  int32_t node_id = -1;
  int32_t peer_node_id = -1;
  int32_t peer_device_id = -1;
  int32_t peer_device_type = -1;
  bool is_del = false;

  std::string GetKey() const {
    std::string key;
    if ((type == ExchangeEndpointType::kEndpointTypeQueue) ||
        (type == ExchangeEndpointType::kEndpointTypeExternalQueue)) {
      key = "queue:" + std::to_string(id) + ", " +
            "device_id:" + std::to_string(device_id) + ", " +
            "device_type:" + std::to_string(device_type);
    } else if (type == ExchangeEndpointType::kEndpointTypeTag) {
      key = "tag:" + std::to_string(id) + ", rank:" + std::to_string(rank_id) + "->" + std::to_string(peer_rank_id);
    } else if (type == ExchangeEndpointType::kEndpointTypeGroup) {
      key = "group:" + std::to_string(id) + ", " +
            "device_id:" + std::to_string(device_id) + ", " +
            "device_type:" + std::to_string(device_type);
    } else {
      // do nothing
    }
    return key;
  }

  std::string DebugString() const {
    std::string debug = "id:" + std::to_string(id) + ", " +
                        "name:" + name + ", " +
                        "node_id:" + std::to_string(node_id) + ", " +
                        "device_id:" + std::to_string(device_id) + ", " +
                        "device_type:" + std::to_string(device_type) + ", " +
                        "is_need_del:" + std::to_string(is_del);
    if ((type == ExchangeEndpointType::kEndpointTypeQueue) ||
        (type == ExchangeEndpointType::kEndpointTypeExternalQueue)) {
      debug += ", type:queue info";
    } else if (type == ExchangeEndpointType::kEndpointTypeTag) {
      debug += (", type:tag info"
                ", handle:" + std::to_string(hcom_handle) +
                ", tag_id:" + std::to_string(tag_id) +
                ", peer_tag_id:" + std::to_string(peer_tag_id) +
                ", rank_id:" + std::to_string(rank_id) +
                ", peer_rank_id:" + std::to_string(peer_rank_id) +
                ", peer_node_id:" + std::to_string(peer_node_id) +
                ", peer_device_id:" + std::to_string(peer_device_id) +
                ", peer_device_type:" + std::to_string(peer_device_type));
    } else if (type == ExchangeEndpointType::kEndpointTypeGroup) {
      debug += ", type:group info"
                ", elements indices:" + ToString(endpoint_indices) +
                ", instance_num:" + std::to_string(instance_num) +
                ", instance_idx:" + std::to_string(instance_idx) +
                ", is_dynamic_sched:" + std::to_string(is_dynamic_sched) +
                ", root_model_id:" + std::to_string(root_model_id);
    } else {
      // do nothing
    }
    return debug;
  }
};

class FlowGwClient {
 public:
  struct ProcessParam {
    uint32_t device_id;
    uint32_t vf_id;
    std::string group_name;
    std::vector<int32_t> res_ids;
    std::vector<int32_t> dev_ids;
  };

  struct RouteDeviceInfo {
    int32_t src_device_id;
    int32_t src_device_type;
    int32_t dst_device_id;
    int32_t dst_device_type;
  };

  struct ExceptionDeviceInfo {
    int32_t node_id;
    int32_t device_type;
    int32_t device_id;
  };

  FlowGwClient(uint32_t device_id, int32_t device_type, const std::vector<int32_t> &res_ids, bool is_proxy);
  virtual ~FlowGwClient() = default;

  virtual Status Initialize();

  Status Finalize();

  Status BindQueues(const std::vector<std::pair<const ExchangeEndpoint *,
                                                const ExchangeEndpoint *>> &queue_routes) const;

  Status UnbindQueues(const std::vector<std::pair<const ExchangeEndpoint *,
                                                  const ExchangeEndpoint *>> &queue_routes) const;

  virtual ProcStatus GetSubProcStat() const;

  void SetExceptionFlag();

  Status DestroyHcomHandle();

  Status GetOrCreateHcomHandle(uint64_t &hcom_handle);

  Status WaitConfigEffect();

  Status CreateFlowGwGroup(const std::vector<const ExchangeEndpoint *> &endpoint_list,
                           int32_t &group_id) const;

  Status DestroyFlowGwGroup(int32_t group_id) const;

  static Status GrantQueue(uint32_t device_id, uint32_t qid, pid_t pid, GrantType grant_type);

  uint32_t GetDeviceId() const {
    return device_id_;
  }

  int32_t GetDeviceType() const {
    return device_type_;
  }

  void SetHcomInfo(const std::string &rank_table, int32_t rank_id) {
    rank_table_ = rank_table;
    rank_id_ = rank_id;
  }

  static void PrintFlowRoute(const std::vector<std::pair<const ExchangeEndpoint *,
                                                         const ExchangeEndpoint *>> &queue_routes,
                             bool print_error = true);

  Status ConfigSchedInfoToDataGw(const uint32_t device_id, const int32_t input_indice, const uint32_t input,
                                 const uint32_t output, const uint32_t root_model_id, const bool is_proxy) const;
  Status UpdateProfiling() const;

  Status ClearFlowgwModelData(const std::set<uint32_t> &model_ids, const int32_t type);

  Status UpdateExceptionRoutes(const std::vector<std::pair<const ExchangeEndpoint *,
                               const ExchangeEndpoint *>> &queue_routes,
                               std::map<int32_t, std::shared_ptr<ExchangeEndpoint>> &endpoints) const;
 protected:
  virtual int32_t KillProcess(pid_t pid, int32_t signal);

 private:
  Status InnerStartFlowGw(const ProcessParam &param);

  Status Shutdown(pid_t pid);

  Status StartFlowGwFromTsd(uint32_t device_id, const std::string &group_name);

  Status StartFlowGw();

  Status InitFlowGwClient() const;

  Status SetHcclProtocol() const;

  static Status ToVisibleDeviceId(const std::vector<int32_t> &logical_device_ids,
                                  std::vector<int32_t> &visible_device_ids);

  Status GrantQueueForRoute(const std::pair<const ExchangeEndpoint *,
                                            const ExchangeEndpoint *> &queue_route) const;

  static std::string FormatListParam(const std::vector<int32_t> &list);

  bqs::Endpoint ToFlowGwEndpoint(const ExchangeEndpoint &endpoint) const;

  void ToFlowGwRoutes(
      const std::vector<std::pair<const ExchangeEndpoint *,
                                const ExchangeEndpoint *>> &queue_routes,
      std::vector<bqs::Route> &flowgw_routes) const;

  void ToFlowGwEndpoints(const std::vector<const ExchangeEndpoint *> &endpoints,
                         std::vector<bqs::Endpoint> &flowgw_endpoints) const;

  static Status FillCommChannelAttrEndpoint(const ExchangeEndpoint &endpoint, bqs::Endpoint &ret);

  ExchangeEndpoint *GetEndpoint(std::map<int32_t, std::shared_ptr<ExchangeEndpoint>> &endpoints,
                                int32_t index) const;

  Status UpdateGroupRoute(const ExchangeEndpoint *group_endpoint,
                          std::map<int32_t, std::shared_ptr<ExchangeEndpoint>> &endpoints) const;

  bool IsExceptionGroup(const ExchangeEndpoint *group_endpoint,
                        std::map<int32_t, std::shared_ptr<ExchangeEndpoint>> &endpoints) const;

  Status CreateHcomHandle();

  pid_t pid_;
  ProcStatus proc_status_;
  uint64_t hcom_handle_;
  uint32_t device_id_;
  int32_t device_type_;
  bool is_proxy_;
  bool is_inited_;
  bool is_exception_;
  std::vector<int32_t> res_ids_;
  std::function<ProcStatus(void)> status_func_;
  std::string rank_table_;
  int32_t rank_id_ = -1;
};
}
#endif // AIR_RUNTIME_HETEROGENEOUS_FLOWRM_FLOWGW_CLIENT_H_
