/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_STREAM_GRAPH_OPTIMIZER_L2_OPTIMIZER_L2_FUSION_ALLOCATION_L2_FUSION_ALLOCATION_H_
#define FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_STREAM_GRAPH_OPTIMIZER_L2_OPTIMIZER_L2_FUSION_ALLOCATION_L2_FUSION_ALLOCATION_H_

#include "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_fusion_comm/l2_fusion_comm.h"

namespace fe {
class L2FusionAllocation {
 public:
  static Status AllocateData(const L2BufferInfo &l2_buffer_info,
                             const std::vector<OpL2DataInfo> &op_l2_data_vec,
                             std::map<uint64_t, size_t> &count_map,
                             OpL2AllocMap &op_l2_alloc_map, uint64_t &max_page_num);

 private:
  static Status AllocateStandingData(const int64_t &page_size, const std::map<uint64_t, size_t> &count_map,
                                     const TensorL2DataMap &output_data_map, TensorL2AllocMap &tensor_l2_alloc_map,
                                     uint32_t &data_in_l2_id, int32_t &page_num_left);

  static Status AllocateStandingDataSpecial(const int64_t &page_size, const std::map<uint64_t, size_t> &count_map,
                                            TensorL2AllocMap &tensor_l2_alloc_map, uint32_t &data_in_l2_id,
                                            int32_t &page_num_left);

  static Status AllocateInputData(const TensorL2AllocMap &tensor_l2_alloc_map,
                                  const TensorL2DataMap &input_data_map,
                                  std::map<uint64_t, size_t> &count_map,
                                  TensorL2AllocMap &input_alloc_map);

  static Status AllocateOutputData(const int64_t &page_size, const uint32_t &max_page_num,
                                   const TensorL2DataMap &output_data_map, uint32_t &data_in_l2_id,
                                   int32_t &data_num_left, int32_t &page_num_left,
                                   TensorL2AllocMap &tensor_l2_alloc_map, TensorL2AllocMap &output_alloc_map);

  static Status UpdateStandingDataSizeWithOutputSet(const int64_t &page_size, const TensorL2DataMap &output_data_map,
                                                    TensorL2AllocMap &tensor_l2_alloc_map, int32_t page_num_left);

  static Status AllocateStandingDataByTaskNum(const uint32_t &node_task_num, const uint64_t &page_size,
                                              const std::map<uint64_t, size_t> &count_map,
                                              const TensorL2DataMap &output_data_map,
                                              TensorL2AllocMap &tensor_l2_alloc_map, uint32_t &data_in_l2_id,
                                              int32_t &page_num_left);

  static void EraseStandingDataAllocCountMapZero(const std::map<uint64_t, size_t> &count_map,
                                                 TensorL2AllocMap &tensor_l2_alloc_map);

  static Status InitDataAlloc(const TensorL2DataInfo &l2_data, const uint8_t &priority, const int32_t &page,
                              const uint32_t &max_page_num, const int32_t &page_num_left, const int64_t &page_size,
                              uint32_t &data_in_l2_id, TensorL2AllocInfo &tensor_alloc_info);
};
}  // namespace fe
#endif  // FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_STREAM_GRAPH_OPTIMIZER_L2_OPTIMIZER_L2_FUSION_ALLOCATION_L2_FUSION_ALLOCATION_H_
