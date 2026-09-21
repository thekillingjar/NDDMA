/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bg_model_desc.h"
#include "common/checker.h"
#include "exe_graph/lowering/frame_selector.h"
#include "framework/runtime/gert_const_types.h"

namespace gert {
namespace bg {
ValueHolderPtr GetModelDesc(const LoweringGlobalData &global_data) {
  return global_data.GetUniqueValueHolder(GetConstDataTypeStr(ConstDataType::kModelDescription));
}

ValueHolderPtr GetSpaceRegistry(const LoweringGlobalData &global_data, ge::OppImplVersion opp_impl_version) {
  auto outputs = FrameSelector::OnInitRoot([&global_data, &opp_impl_version]() -> std::vector<ValueHolderPtr> {
    auto rt_model_desc = bg::HolderOnInit(bg::GetModelDesc(global_data));
    auto opp_impl_version_addr = ValueHolder::CreateConst(&opp_impl_version, sizeof(opp_impl_version), false);
    return {ValueHolder::CreateSingleDataOutput("GetSpaceRegistry", {rt_model_desc, opp_impl_version_addr})};
  });
  return outputs[0];
}

ValueHolderPtr GetSpaceRegistries(const LoweringGlobalData &global_data) {
  auto outputs = FrameSelector::OnInitRoot([&global_data]() -> std::vector<ValueHolderPtr> {
    auto rt_model_desc = bg::HolderOnInit(bg::GetModelDesc(global_data));
    return {ValueHolder::CreateSingleDataOutput("GetSpaceRegistries", {rt_model_desc})};
  });
  return outputs[0];
}

ValueHolderPtr GetFileConstantWeightDir(const LoweringGlobalData &global_data) {
  auto outputs = FrameSelector::OnInitRoot([&global_data]() -> std::vector<ValueHolderPtr> {
    auto rt_model_desc = bg::HolderOnInit(bg::GetModelDesc(global_data));
    return {ValueHolder::CreateSingleDataOutput("GetFileConstantWeightDir", {rt_model_desc})};
  });
  GE_ASSERT_TRUE(outputs.size() == 1U, "GetFileConstantWeightDir output size[%zu], expect 1.", outputs.size());
  return outputs[0];
}
}  // namespace bg
}  // namespace gert
