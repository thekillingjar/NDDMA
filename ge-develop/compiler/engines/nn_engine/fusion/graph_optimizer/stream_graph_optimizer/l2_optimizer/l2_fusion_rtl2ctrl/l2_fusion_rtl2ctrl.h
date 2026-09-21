/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_STREAM_GRAPH_OPTIMIZER_L2_OPTIMIZER_L2_FUSION_RTL2CTRL_L2_FUSION_RTL2CTRL_H_
#define FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_STREAM_GRAPH_OPTIMIZER_L2_OPTIMIZER_L2_FUSION_RTL2CTRL_L2_FUSION_RTL2CTRL_H_

#include "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_fusion_comm/l2_fusion_comm.h"

namespace fe {
class L2FusionRtl2ctrl {
 public:
  static void InitRtl2ctrl(rtL2Ctrl_t &l2ctrl);
  static Status UpdateRtL2Ctrl(OpL2AllocMap &l2_alloc_map, uint64_t max_page_num, uint32_t max_data_num);

 private:
  static void SetDataL2ctrl(uint32_t max_data_num, TensorL2AllocMap &data, rtL2Ctrl_t &l2ctrl);
  static void SetOutputDataL2ctrl(uint32_t max_data_num, TensorL2AllocMap &standing, TensorL2AllocMap &data,
                                  rtL2Ctrl_t &out_l2ctrl);
  static Status UpdateInputDatal2ctrlId(TensorL2AllocMap &input, TensorL2AllocMap &standing_data);
};
}  // namespace fe
#endif  // FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_STREAM_GRAPH_OPTIMIZER_L2_OPTIMIZER_L2_FUSION_RTL2CTRL_L2_FUSION_RTL2CTRL_H_
