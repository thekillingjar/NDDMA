/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef USER_HYBRID_GRAPH_MANAGER_H
#define USER_HYBRID_GRAPH_MANAGER_H

#include <cstdint>
#include <map>

#include "graph/graph.h"
#include "exe_graph/runtime/tensor.h"
#include "ge/ge_api_types.h"
#include "jit_execution/user_graphs_manager.h"

namespace ge {
struct HybridDynamicDimsInfo {
  std::vector<std::vector<int64_t>> dynamic_shape_dims;
  std::vector<std::pair<std::string, std::vector<int64_t>>> user_input_dims;
};
// 在动态分档混合模式开启情况下，ge_api传入一张分档图，内部会编译一张分档图和一张动态shape图。
// 执行时候根据输入shape判断，在分档范围走分档图执行，否则走动态shape图执行
// UserHybridGraphManager类就是处理上述逻辑，包含但不仅限于管理内外部graphid，保存分档信息，判断此次执行图等。
// 向上由session的相关api调用，向下调用已有的UserGraphsManager的相关api。
class UserHybridGraphManager {
 public:
  explicit UserHybridGraphManager(UserGraphsManager &user_graph_manager) : user_graph_manager_(user_graph_manager) {}
  Status AddGraph(uint32_t user_graph_id, const Graph &graph, const std::map<std::string, std::string> &options);
  Status BuildGraph(uint32_t user_graph_id, const std::vector<GeTensor> &inputs, uint64_t session_id);
  Status RunGraphAsync(uint32_t user_graph_id, std::vector<gert::Tensor> &&inputs, uint64_t session_id,
      const RunAsyncCallbackV2 &callback);
  Status RemoveGraph(uint32_t user_graph_id);
  bool IsGraphNeedRebuild(uint32_t user_graph_id);
  Status GetCompiledFlag(uint32_t user_graph_id, bool &flag);
  Status SetCompiledFlag(uint32_t user_graph_id, bool flag);
  Status Finalize();
 private:
  bool IsHybridMode(const std::map<std::string, std::string> &options) const;
  void SetDynamicGraphId(const uint32_t user_graph_id);
  bool TryGetDynamicGraphId(const uint32_t user_graph_id, uint32_t &dynamic_gear_graph_id,
                            uint32_t &dynamic_shape_graph_id);
  Status RecordDynamicGearInfo(const uint32_t graph_id);
  void SetDynamicGearInfo(const uint32_t graph_id, const HybridDynamicDimsInfo &dynamic_dims_info);
  Status GetDynamicGearInfo(const uint32_t graph_id, HybridDynamicDimsInfo &dynamic_dims_info);
  Status SelectExecuteGraph(const uint32_t dynamic_gear_graph_id, const uint32_t dynamic_shape_graph_id,
                            const std::vector<gert::Tensor> &inputs, uint32_t &select_graph_id);

 private:
  UserGraphsManager &user_graph_manager_;
  std::mutex dynamic_graph_id_mu_;
  std::map<uint32_t, std::pair<uint32_t, uint32_t>> dynamic_graph_id_map_;
  std::mutex dynamic_dims_info_mu_;
  std::map<uint32_t, HybridDynamicDimsInfo> dynamic_dims_info_map_;
  uint32_t inner_graph_id_cnt_{0U};
};
using UserHybridGraphManagerPtr = std::shared_ptr<UserHybridGraphManager>;
} // ge

#endif  // USER_HYBRID_GRAPH_MANAGER_H
