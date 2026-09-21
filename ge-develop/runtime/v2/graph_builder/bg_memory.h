/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_MEMORY_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_MEMORY_H_
#include <vector>

#include "exe_graph/lowering/dev_mem_value_holder.h"
#include "exe_graph/lowering/lowering_global_data.h"
#include "exe_graph/runtime/tensor.h"
#include "graph/types.h"
namespace gert {
namespace bg {
struct MemoryKernelsName {
  const char *alloc_name;
  const char *free_name;
  const char *alloc_batch;
  const char *free_batch;
};
std::vector<DevMemValueHolderPtr> AllocOutputMemory(TensorPlacement placement, const ge::NodePtr &node,
                                                    const std::vector<ValueHolderPtr> &output_sizes,
                                                    LoweringGlobalData &global_data);

std::vector<DevMemValueHolderPtr> AllocOutputMemory(TensorPlacement placement, const ge::NodePtr &node,
                                                    const std::vector<ValueHolderPtr> &output_sizes,
                                                    const std::vector<DevMemValueHolderPtr> &input_addrs,
                                                    LoweringGlobalData &global_data);

std::vector<DevMemValueHolderPtr> AllocMemories(TensorPlacement placement, const std::vector<ValueHolderPtr> &sizes,
                                                LoweringGlobalData &global_data, const int64_t stream_id);

std::vector<DevMemValueHolderPtr> AllocMemoriesWithoutGuarder(TensorPlacement placement,
                                                              const std::vector<ValueHolderPtr> &sizes,
                                                              LoweringGlobalData &global_data, const int64_t stream_id);

DevMemValueHolderPtr AllocWorkspaceMem(TensorPlacement placement, const ValueHolderPtr &workspaces_size,
                                       LoweringGlobalData &global_data);

DevMemValueHolderPtr FreeWorkspaceMem(TensorPlacement placement, const DevMemValueHolderPtr &addr);
ValueHolderPtr CalcTensorSizeFromStorage(ge::DataType dt, const ValueHolderPtr &storage_shape);
ValueHolderPtr CalcTensorSizeFromShape(ge::DataType dt, const ValueHolderPtr &shape);
ValueHolderPtr CalcOutTensorSize(const ge::NodePtr &node, int32_t out_index, const ValueHolderPtr &shape);
std::vector<ValueHolderPtr> CalcTensorSize(const ge::NodePtr &node, const std::vector<ValueHolderPtr> &output_shapes);
ge::Status GetNodeRefMap(const ge::NodePtr &node, std::map<size_t, size_t> &ref_map);
DevMemValueHolderPtr AllocMem(TensorPlacement placement, const ValueHolderPtr &size_holder,
                              LoweringGlobalData &global_data, const int64_t stream_id);
DevMemValueHolderPtr AllocMemWithoutGuarder(TensorPlacement placement, const ValueHolderPtr &size_holder,
                                            LoweringGlobalData &global_data, const int64_t stream_id);
DevMemValueHolderPtr AllocMemOnInit(TensorPlacement placement, size_t size, LoweringGlobalData &global_data);
ValueHolderPtr AllocFixedFeatureMemory(const ValueHolderPtr &session_id_holder,
                                       const TensorPlacement placement,
                                       const ValueHolderPtr &size_holder,
                                       LoweringGlobalData &global_data);
// ffts memory
ValueHolderPtr CreateFftsMemAllocator(const ValueHolderPtr &window_size, LoweringGlobalData &global_data);
ValueHolderPtr RecycleFftsMems(const ValueHolderPtr &ffts_mem_allocator);
std::vector<DevMemValueHolderPtr> AllocateFftsMems(const ValueHolderPtr &ffts_mem_allocator, const int64_t stream_id,
                                                   const std::vector<ValueHolderPtr> &block_sizes);
DevMemValueHolderPtr AllocateBatchFftsMems(const ValueHolderPtr &ffts_mem_allocator, const ValueHolderPtr &batch_sizes,
                                           const int64_t stream_id);
const MemoryKernelsName &GetMemoryKernelsName(const TensorPlacement placement);
}  // namespace bg
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_MEMORY_H_
