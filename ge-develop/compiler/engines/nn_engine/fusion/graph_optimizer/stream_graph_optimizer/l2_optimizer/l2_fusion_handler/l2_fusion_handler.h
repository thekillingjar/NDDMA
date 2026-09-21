/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_STREAM_GRAPH_OPTIMIZER_L2_OPTIMIZER_L2_FUSION_HANDLER_L2_FUSION_HANDLER_H_
#define FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_STREAM_GRAPH_OPTIMIZER_L2_OPTIMIZER_L2_FUSION_HANDLER_L2_FUSION_HANDLER_H_

#include "common/l2fusion_struct.h"
#include "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_fusion_comm/l2_fusion_comm.h"

namespace fe {
class L2FusionHandler {
 public:
  static Status GetL2DataAlloc(const uint64_t &mem_base, const ge::ComputeGraph &graph, TaskL2InfoMap &l2_info_map);

 private:
  static Status GetNormalL2DataAlloc(const uint64_t &mem_base, const ge::ComputeGraph &graph,
                                     TaskL2InfoMap &l2_info_map);
  static Status GenerateFinalL2Info(const uint64_t &mem_base, const ge::ComputeGraph &graph,
                                    const OpL2AllocMap &l2_alloc_map, TaskL2InfoMap &l2_info_map);
  static void GenerateL2Data(const uint64_t &mem_base, const TensorL2AllocMap &src_data, L2DataMap &dst_data);
  static void DisplayL2Info(const TaskL2InfoMap &l2_info_map);
  static void DisplayL2DataInfo(const string &title, const L2DataMap &input, rtL2Ctrl_t &l2ctrl, L2DataMap data);
};
}  // namespace fe
#endif  // FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_STREAM_GRAPH_OPTIMIZER_L2_OPTIMIZER_L2_FUSION_HANDLER_L2_FUSION_HANDLER_H_
