/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DFLOW_SESSION_INNER_SESSION_H_
#define DFLOW_SESSION_INNER_SESSION_H_

#include <map>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include "framework/common/ge_types.h"
#include "ge/ge_api_types.h"
#include "ge/ge_data_flow_api.h"
#include "dflow/compiler/model/dflow_graph_manager.h"
#include "flow_graph/flow_graph.h"
#include "external/ge/ge_api_v2.h"

namespace ge {
GE_FUNC_VISIBILITY Status DFlowInitializeInner(const std::map<AscendString, AscendString> &options);
GE_FUNC_VISIBILITY void DFlowFinalizeInner();

class DFlowSessionImpl {
 public:
  /**
   * @brief 构造函数
   * @param session_id 会话ID
   * @param options 选项
   * @param options 从inner_session传入的options
   */
  DFlowSessionImpl(uint64_t session_id, const std::map<std::string, std::string> &options);

  ~DFlowSessionImpl();

  Status Initialize(const std::map<std::string, std::string> &options = {});

  Status AddGraph(uint32_t graph_id, const dflow::FlowGraph &graph, const std::map<std::string, std::string> &options);

  Status AddGraph(uint32_t graph_id, const Graph &graph, const std::map<std::string, std::string> &options);

  Status RemoveGraph(uint32_t graph_id);

  Status CompileGraph(uint32_t graph_id, const std::vector<GeTensor> &ge_inputs);

  Status BuildGraph(uint32_t graph_id, const std::vector<Tensor> &inputs);
  Status BuildGraph(uint32_t graph_id, const std::vector<ge::GeTensor> &ge_inputs);
  // used for compatibility_mode
  Status RunGraph(uint32_t graph_id, const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs);

  Status Finalize();

  Status FeedDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes, const std::vector<Tensor> &inputs,
                           const DataFlowInfo &info, int32_t timeout);

  Status FetchDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes, std::vector<Tensor> &outputs,
                            DataFlowInfo &info, int32_t timeout);

  Status FeedDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                           const std::vector<FlowMsgPtr> &inputs, int32_t timeout);

  Status FetchDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                            std::vector<FlowMsgPtr> &outputs, int32_t timeout);

  Status FeedRawData(uint32_t graph_id, const std::vector<RawData> &raw_data_list, const uint32_t index,
                     const DataFlowInfo &info, int32_t timeout);

  uint64_t GetSessionId() const {
    return session_id_;
  }
  FlowModelPtr GetFlowModel(uint32_t graph_id) const;
 private:
  void UpdateThreadContext(const std::map<std::string, std::string> &options) const;
  void UpdateGlobalSessionContext() const;
  void UpdateThreadContext(uint32_t graph_id);
  void SetRtSocVersion() const;
  FlowModelPtr CompileAndLoadGraph(uint32_t graph_id, const std::vector<GeTensor> &inputs);
  // 仅用于防重复初始化，若初始化失败，inner session对象不应被获取到，其成员方法也不会被调用, 因此初始化成功后成员方法内不必再判断
  bool is_initialized_{false};
  uint64_t session_id_;
  std::map<std::string, std::string> options_;
  DflowGraphManager dflow_graph_manager_;
  std::mutex resource_mutex_;  // AddGraph, RemoveGraph and Finalize use
  std::mutex build_run_mutex_;  // BuildGraph and RunGraph use
  std::mutex model_mutex_;
  std::set<uint32_t> loaded_models_;
  std::shared_ptr<GeSession> ge_session_;
};
}  // namespace ge

#endif  // DFLOW_SESSION_INNER_SESSION_H_
