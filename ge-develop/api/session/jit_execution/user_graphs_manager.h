/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef USER_GRAPHS_MANAGER_H
#define USER_GRAPHS_MANAGER_H
#include <cstdint>
#include <map>

#include "graph/graph.h"
#include "exe_graph/runtime/tensor.h"
#include "ge/ge_api_types.h"

#include "user_graph_ctrl.h"
#include "compile_context.h"
namespace ge {
class UserGraphsManager {
 public:
  explicit UserGraphsManager(GraphManager &graph_manager)
      : compile_context_(graph_manager), graph_manager_(graph_manager) {}
  Status AddGraph(uint32_t user_graph_id, const Graph &graph, const std::map<std::string, std::string> &options);
  Status BuildGraph(uint32_t user_graph_id, const std::vector<GeTensor> &inputs, uint64_t session_id) const;
  Status RunGraphAsync(uint32_t user_graph_id, std::vector<gert::Tensor> &&inputs,
      uint64_t session_id, const RunAsyncCallbackV2 &callback);
  Status RemoveGraph(uint32_t user_graph_id);
  bool IsGraphNeedRebuild(uint32_t user_graph_id);
  Status GetCompiledFlag(uint32_t user_graph_id, bool &flag);
  Status SetCompiledFlag(uint32_t user_graph_id, bool flag);
  Status Finalize();
  Status CompileGraph(uint32_t user_graph_id, uint64_t session_id, const vector<ge::Tensor> &inputs);
  Status GetCompiledGraphSummary(uint32_t user_graph_id, CompiledGraphSummaryPtr &summary);
  Status LoadGraph(const uint32_t user_graph_id, const std::map<AscendString, AscendString> &options,
                   void *stream);
  Status ExecuteGraphWithStreamAsync(uint32_t user_graph_id, void *stream, const std::vector<gert::Tensor> &inputs,
                                     std::vector<gert::Tensor> &outputs, uint64_t session_id);
  Status GetOmeContextByGraphId(const GraphId &graph_id, OmeContext &ome_context) const;
 private:
  UserGraphControl* GetUserGraphControl(uint32_t user_graph_id);
 private:
  CompileContext compile_context_;
  GraphManager &graph_manager_;

  std::mutex user_graph_ctrl_mutex_;
  std::map<uint32_t, std::unique_ptr<UserGraphControl>> ids_to_user_graph_ctrl_;
};
using UserGraphsManagerPtr = std::shared_ptr<UserGraphsManager>;
} // ge

#endif // USER_GRAPHS_MANAGER_H
