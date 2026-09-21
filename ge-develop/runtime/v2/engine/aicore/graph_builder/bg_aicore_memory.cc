/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bg_aicore_memory.h"
#include "common/checker.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_builder/value_holder_generator.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "engine/node_converter_utils.h"
#include "engine/aicore/fe_rt2_common.h"

namespace {
const size_t kMaxDimNum = 8;
const size_t kShapeBufferNum = kMaxDimNum + 1U;
}  // namespace
namespace gert {
namespace bg {
DevMemValueHolderPtr AllocShapeBufferMem(TensorPlacement placement, const ge::NodePtr &node,
                                         LoweringGlobalData &global_data) {
  int32_t unknown_shape_type_val = 0;
  (void)ge::AttrUtils::GetInt(node->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);
  // only alloc shape buffer for third class op
  if (static_cast<ge::UnknowShapeOpType>(unknown_shape_type_val) != ge::DEPEND_SHAPE_RANGE) {
    gert::GertTensorData *tensor_data = nullptr;
    return DevMemValueHolder::CreateConst(&tensor_data, sizeof(gert::GertTensorData *),
                                          node->GetOpDesc()->GetStreamId());
  }
  FE_ASSERT_TRUE(placement < kTensorPlacementEnd, "Failed to create AllocMem for workspace, unknown memory type %d",
                 placement);

  auto allocator_holder = global_data.GetOrCreateAllocator({placement, AllocatorUsage::kAllocNodeShapeBuffer});
  const size_t kShapeBufferSize = kShapeBufferNum * sizeof(uint64_t);
  size_t shape_buffer_size = node->GetOpDesc()->GetOutputsSize() * kShapeBufferSize;
  auto shape_buffer_size_holder = ValueHolder::CreateConst(&shape_buffer_size, sizeof(shape_buffer_size));
  auto mem = DevMemValueHolder::CreateSingleDataOutput(GetMemoryKernelsName(placement).alloc_name,
                                                       {allocator_holder, shape_buffer_size_holder},
                                                       node->GetOpDesc()->GetStreamId());
  FE_ASSERT_NOTNULL(mem);
  mem->SetPlacement(placement);
  return mem;
}
ValueHolderPtr FreeShapeBufferMem(TensorPlacement placement, const DevMemValueHolderPtr &addr) {
  if (addr == nullptr) {
    return nullptr;
  }
  if (placement >= kTensorPlacementEnd) {
    return CreateErrorValueHolders(1, "Failed to create free node for workspace, unknown memory type %d", placement)[0];
  }
  return ValueHolder::CreateVoidGuarder(GetMemoryKernelsName(placement).free_name, {addr}, {});
}

DevMemValueHolderPtr AllocAiCoreWorkspaceMem(const ge::NodePtr node, TensorPlacement placement,
                                             const ValueHolderPtr &workspaces_size, LoweringGlobalData &global_data) {
  int64_t buffer_size = 0;
  bg::ValueHolderPtr expand_work = nullptr;
  if (ge::AttrUtils::GetInt(node->GetOpDesc(), gert::kOpDfxBufferSize, buffer_size) && buffer_size > 0) {
    GELOGD("Node[%s] buffer size is %ld.", node->GetNamePtr(), buffer_size);
    size_t buf_size = buffer_size;
    auto buffer_size_holder = bg::ValueHolder::CreateConst(&buf_size, sizeof(buf_size));
    expand_work = bg::ValueHolder::CreateVoid<bg::ValueHolder>("ExpandDfxWorkspaceSize", {workspaces_size,
                                                                                           buffer_size_holder});
  }
  auto work_mem = bg::AllocWorkspaceMem(placement, workspaces_size, global_data);
  if (expand_work != nullptr) {
    FE_ASSERT_HYPER_SUCCESS(bg::ValueHolder::AddDependency(expand_work, work_mem));
  }
  return work_mem;
}
}  // namespace bg
}  // namespace gert
