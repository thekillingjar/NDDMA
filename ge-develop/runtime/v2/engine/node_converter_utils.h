/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_NODE_CONVERTER_NODE_CONVERTER_UTILS_H_
#define AIR_CXX_RUNTIME_V2_NODE_CONVERTER_NODE_CONVERTER_UTILS_H_

#include <vector>
#include "proto/task.pb.h"
#include "graph/op_desc.h"
#include "graph/ge_tensor.h"
#include "graph/compute_graph.h"
#include "exe_graph/lowering/value_holder.h"
#include "exe_graph/lowering/lowering_global_data.h"

namespace gert {
enum class TaskDefType{
    kAtomicClean,
    kAICore,
    kAICpu,
    kMax
};
enum class OpParamType {
  REQUIRED = 0,
  OPTIONAL,
  DYNAMIC,
  RESERVED
};
enum class OpDfxOpt {
  PRINT = 0,
  ASSERT,
  RESERVED
};
constexpr const ge::char_t *kOpDfxAssert = "assert";
constexpr const ge::char_t *kOpDfxPrintf = "printf";

using HistoryNodeMemInfo = std::vector<bg::ValueHolderPtr>;
using HistoryMemInfo = std::vector<HistoryNodeMemInfo>;
using HistoryMemInfoPtr = std::shared_ptr<HistoryMemInfo>;
constexpr size_t kDefaultWorkspaceCap = 16;
const domi::TaskDef *GetTaskDef(const ge::NodePtr &node, const LoweringGlobalData::NodeCompileResult *compile_result,
                                TaskDefType type);

class NodeConverterUtils {
 public:
  static std::vector<bg::ValueHolderPtr> CreateOutputShapes(const ge::OpDescPtr &op_desc);
  static std::vector<bg::ValueHolderPtr> CreateOutputShapes(
      const std::vector<ge::ConstGeTensorDescPtr> &output_tensor_descs);
  static bg::ValueHolderPtr CreateOutputShape(const ge::ConstGeTensorDescPtr &tensor_desc);
  static bg::ValueHolderPtr CreateOutputShape(const ge::ConstGeTensorDescPtr &tensor_desc, StorageShape &shape);
  static Shape GetShapeFromGeShape(const ge::GeShape &ge_shape);
};
constexpr const ge::char_t *kFftsFreeVecHolder = "_ffts_free_vec_holder";
constexpr const ge::char_t *kFftsAllocVecHolder = "_ffts_alloc_vec_holder";
constexpr const ge::char_t *kStaticToDynamicSoftSyncOp = "_static_to_dynamic_softsync_op";
constexpr const ge::char_t *kFtfsMemoryPoolType = "_ffts_memory_pool_type";
constexpr const ge::char_t *kFtfsSubGraphOutputsIndex = "_ffts_subgraph_outputs_index";
constexpr const ge::char_t *kFftsMemAllocVecHolder = "_ffts_mem_alloc_vec_holder";
constexpr const ge::char_t *kOpDfxOptions = "_op_dfx_options";
constexpr const ge::char_t *kOpDfxBufferSize = "_op_dfx_buffer_size";
const std::vector<std::string> CMO_IDX_KEY = {"prefetch_idx", "invalidate_idx", "write back_idx"};

const std::unordered_set<std::string> kDSAStatelessOps = {"DSAStatelessRandomTruncatedNormal",
                                                          "DSAStatelessRandomNormal",
                                                          "DSAStatelessGenBitMask",
                                                          "DSAStatelessRandomUniform"};

std::vector<bg::ValueHolderPtr> CreateOutputShapes(const ge::OpDescPtr &op_desc);
bg::ValueHolderPtr ConvertWorkspaceSize(const ge::NodePtr &node);
std::vector<bg::ValueHolderPtr> CreateInputOutputShapes(bool is_input, bool is_orishape,
                                                        const ge::OpDescPtr &op_desc);
std::vector<bg::ValueHolderPtr> CalcInputOutputTensorSize(bool is_input, bool is_orishape,
                                                          const ge::NodePtr &node,
                                                          const std::vector<bg::ValueHolderPtr> &shapes);
std::vector<bg::ValueHolderPtr> GetOrCreateInputFeeds(LoweringGlobalData *global_data,
    const ge::ComputeGraphPtr &graph);
bool IsDataDumpOpen();
bool IsExceptionDumpOpen();
bool GetDfxOptFlagByType(const ge::NodePtr node, OpDfxOpt opt_type);
} // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_NODE_CONVERTER_NODE_CONVERTER_UTILS_H_
