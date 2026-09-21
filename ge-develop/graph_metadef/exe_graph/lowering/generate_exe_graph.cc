/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/lowering/generate_exe_graph.h"
#include "exe_graph/lowering/lowering_global_data.h"
namespace gert {
namespace bg {
GenerateExeGraph::ExeGraphGenerator GenerateExeGraph::generator_ = {nullptr, nullptr, nullptr};

ValueHolderPtr GenerateExeGraph::MakeSureTensorAtHost(const ge::Node *node, LoweringGlobalData &global_data,
                                                      const ValueHolderPtr &addr, const ValueHolderPtr &size) {
  std::vector<bg::ValueHolderPtr> copy_inputs{global_data.GetStream()};
  copy_inputs.emplace_back(global_data.GetOrCreateAllocator({kOnHost, AllocatorUsage::kAllocNodeWorkspace}));
  copy_inputs.emplace_back(addr);
  copy_inputs.emplace_back(size);
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  return bg::DevMemValueHolder::CreateSingleDataOutput("MakeSureTensorAtHost", copy_inputs, op_desc->GetStreamId());
}

ValueHolderPtr GenerateExeGraph::CalcTensorSizeFromShape(ge::DataType dt, const ValueHolderPtr &shape) {
  auto data_type = ValueHolder::CreateConst(&dt, sizeof(dt));
  return ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromShape", {data_type, shape});
}

ValueHolderPtr GenerateExeGraph::FreeMemoryGuarder(const ValueHolderPtr &resource) {
  return ValueHolder::CreateVoidGuarder("FreeMemory", resource, {});
}
}  // namespace bg
}  // namespace gert
