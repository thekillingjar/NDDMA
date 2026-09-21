/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bg_platform.h"
#include "bg_core_type.h"
#include "exe_graph/lowering/frame_selector.h"
#include "exe_graph/lowering/lowering_global_data.h"
#include "graph/debug/ge_attr_define.h"
#include "ge_common/util.h"
#include "register/core_num_utils.h"

namespace gert {
namespace bg {
namespace {
constexpr char const *kPlatformInfo = "PlatformInfos";
constexpr char const *kDefaultCoreType = "AiCore";
constexpr char const *kAiCoreNum = "_op_aicore_num";
constexpr char const *kVectorCoreNum = "_op_vectorcore_num";
}  // namespace
std::vector<ValueHolderPtr> GetPlatformInfo(LoweringGlobalData *global_data) {
  auto builder = []() -> std::vector<bg::ValueHolderPtr> {
    return bg::FrameSelector::OnInitRoot([]() -> std::vector <bg::ValueHolderPtr> {
      return bg::ValueHolder::CreateDataOutput("GetPlatformInfo", {}, 2U);
    });
  };
  return global_data->GetOrCreateUniqueValueHolder(kPlatformInfo, builder);
}

std::vector<ValueHolderPtr> AppendCoreTypeToPlatform(const ge::NodePtr &node, LoweringGlobalData *global_data) {
  GE_ASSERT_NOTNULL(global_data);
  std::string core_type = kDefaultCoreType;
  int32_t op_aicore_num = -1;
  int32_t op_vec_core_num = -1;
  string aicore_num_str;
  string vec_core_num_str;
  const auto op_desc = node->GetOpDescBarePtr();
  if (op_desc != nullptr) {
    const std::string *tmp_core_type = ge::AttrUtils::GetStr(op_desc, ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE);
    if (tmp_core_type != nullptr) {
      core_type = *tmp_core_type;
      GELOGD("Get attr: %s of node: %s is %s.", ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE.c_str(), op_desc->GetNamePtr(), core_type.c_str());
    }
    const std::string *tmp_aicore_num_str = ge::AttrUtils::GetStr(op_desc, kAiCoreNum);
    if ((tmp_aicore_num_str != nullptr) && !tmp_aicore_num_str->empty()) {
      aicore_num_str = *tmp_aicore_num_str;
      GELOGD("Get attr: %s of node: %s is %s.", kAiCoreNum, op_desc->GetNamePtr(), aicore_num_str.c_str());
      GE_ASSERT_SUCCESS(ge::CoreNumUtils::ParseAndValidateCoreNum(kAiCoreNum, aicore_num_str, 0, INT32_MAX, op_aicore_num));
    }

    const std::string *tmp_vec_core_num_str = ge::AttrUtils::GetStr(op_desc, kVectorCoreNum);
    if ((tmp_vec_core_num_str != nullptr) && !tmp_vec_core_num_str->empty()) {
      vec_core_num_str = *tmp_vec_core_num_str;
      GELOGD("Get attr: %s of node: %s is %s.", kVectorCoreNum, op_desc->GetNamePtr(), vec_core_num_str.c_str());
      GE_ASSERT_SUCCESS(ge::CoreNumUtils::ParseAndValidateCoreNum(kVectorCoreNum, vec_core_num_str, 0, INT32_MAX, op_vec_core_num));
    }
  }

  std::string core_type_key = kPlatformInfo;
  core_type_key += ("_Append_CoreType_" + core_type + "_" + aicore_num_str + "_" + vec_core_num_str);
  auto builder = [&global_data, &node, &op_aicore_num, &op_vec_core_num]() -> std::vector<bg::ValueHolderPtr> {
    return bg::FrameSelector::OnInitRoot([&global_data, &node, &op_aicore_num, &op_vec_core_num]() -> std::vector<bg::ValueHolderPtr> {
      auto platform_info_holders = bg::GetPlatformInfo(global_data);

      auto platform_info = HolderOnInit(platform_info_holders[0]);
      GE_ASSERT_NOTNULL(platform_info);

      auto core_type_holder = HolderOnInit(bg::GetCoreType(node, global_data));
      GE_ASSERT_NOTNULL(core_type_holder);

      auto op_ai_core_num_holder = ValueHolder::CreateConst(&op_aicore_num, sizeof(op_aicore_num));
      auto op_vector_core_num_holder = ValueHolder::CreateConst(&op_vec_core_num, sizeof(op_vec_core_num));

      auto core_num_infos_holder = HolderOnInit(platform_info_holders[1]);
      GE_ASSERT_NOTNULL(core_num_infos_holder);

      return bg::ValueHolder::CreateDataOutput(
          "AppendCoreTypeToPlatform",
          {platform_info, core_type_holder, op_ai_core_num_holder, op_vector_core_num_holder, core_num_infos_holder},
          2U);
    });
  };
  return global_data->GetOrCreateUniqueValueHolder(core_type_key, builder);
}
}  // namespace bg
}  // namespace gert
