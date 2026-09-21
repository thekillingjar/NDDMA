/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bg_core_type.h"
#include "exe_graph/lowering/frame_selector.h"
#include "exe_graph/lowering/lowering_global_data.h"
#include "graph/debug/ge_attr_define.h"

namespace gert {
namespace bg {
namespace {
constexpr char const *kDefaultCoreType = "AiCore";
const std::map<std::string, CoreTypeIndex> kCoreTypeReflection{
    {"AiCore", CoreTypeIndex::kAiCore}, {"VectorCore", CoreTypeIndex::kVectorCore}, {"Mix", CoreTypeIndex::kMix},
    {"AIC", CoreTypeIndex::kAiCore}, {"AIV", CoreTypeIndex::kVectorCore}, {"MIX_AIC", CoreTypeIndex::kMixAic},
    {"MIX_AIV", CoreTypeIndex::kMixAiv}, {"MIX_AICORE", CoreTypeIndex::kMixAiCore},
    {"MIX_VECTOR_CORE", CoreTypeIndex::kMixAiVector}};
const std::string kMixIsAiv = "_mix_is_aiv";
}  // namespace
ValueHolderPtr GetCoreType(const ge::NodePtr &node, LoweringGlobalData *global_data) {
  GE_ASSERT_NOTNULL(global_data);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  std::string core_type = kDefaultCoreType;
  const std::string *core_type_ptr = ge::AttrUtils::GetStr(op_desc, ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE);
  if (core_type_ptr != nullptr) {
    core_type = *core_type_ptr;
    GELOGD("Get attr: %s of op[%s, %s] is %s", ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE.c_str(), op_desc->GetNamePtr(),
           op_desc->GetTypePtr(), core_type.c_str());
  }

  CoreTypeIndex core_type_index = CoreTypeIndex::kAiCore;
  const auto iter = kCoreTypeReflection.find(core_type);
  if (iter != kCoreTypeReflection.end()) {
    core_type_index = iter->second;
  } else {
    bool mix_is_aiv = false;
    (void)ge::AttrUtils::GetBool(op_desc, kMixIsAiv, mix_is_aiv);
    if (mix_is_aiv) {
      core_type_index = CoreTypeIndex::kMixAiv;
      GELOGI("Lowering node MIX_AIV");
    }
  }

  const auto core_type_key = "CoreType_" + core_type;
  GELOGD("Op[%s, %s] core type key: %s", op_desc->GetNamePtr(), op_desc->GetTypePtr(), core_type_key.c_str());
  auto builder = [&core_type_index]() -> std::vector<bg::ValueHolderPtr> {
    return bg::FrameSelector::OnInitRoot([&core_type_index]() -> std::vector<bg::ValueHolderPtr> {
      auto core_type_holder = bg::ValueHolder::CreateConst(&core_type_index, sizeof(CoreTypeIndex));
      return {core_type_holder};
    });
  };
  return global_data->GetOrCreateUniqueValueHolder(core_type_key, builder)[0];
}
}  // namespace bg
}  // namespace gert
