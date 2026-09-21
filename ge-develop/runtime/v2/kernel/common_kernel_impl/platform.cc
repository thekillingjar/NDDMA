/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "platform.h"
#include "common/checker.h"
#include "framework/common/helper/model_helper.h"
#include "register/kernel_registry.h"
#include "register/core_num_utils.h"
#include "acl/acl_rt.h"

namespace gert {
namespace kernel {
namespace {
constexpr char const *kDefaultCoreType = "AiCore";
const std::map<CoreTypeIndex, std::string> kCoreTypeReflection{
    {CoreTypeIndex::kAiCore, "AiCore"}, {CoreTypeIndex::kVectorCore, "VectorCore"}, {CoreTypeIndex::kMix, "Mix"},
    {CoreTypeIndex::kMixAic, "MIX_AIC"}, {CoreTypeIndex::kMixAiv, "MIX_AIV"},
    {CoreTypeIndex::kMixAiCore, "MIX_AICORE"}, {CoreTypeIndex::kMixAiVector, "MIX_VECTOR_CORE"}};
constexpr char_t const *kAicCntKeyIni = "ai_core_cnt";
constexpr char_t const *kCubeCntKeyIni = "cube_core_cnt";
constexpr char_t const *kAivCntKeyIni = "vector_core_cnt";
constexpr char_t const *kSocInfo = "SoCInfo";
constexpr char_t const *kAiCoreNum = "_op_aicore_num";
constexpr char_t const *kVectorCoreNum = "_op_vectorcore_num";
}  // namespace
ge::graphStatus GetPlatformInfo(KernelContext *context) {
  auto platform_holder = context->GetOutputPointer<fe::PlatFormInfos>(0);
  GE_ASSERT_NOTNULL(platform_holder);
  ge::ModelHelper model_helper;
  fe::PlatformInfo platform_info;
  GE_ASSERT_SUCCESS(model_helper.HandleDeviceInfo(*platform_holder, platform_info));
  auto core_num_infos_holder = context->GetOutputPointer<CoreNumInfos>(1);
  GE_ASSERT_NOTNULL(core_num_infos_holder);
  core_num_infos_holder->soc_aicore_num = static_cast<int32_t>(platform_info.soc_info.ai_core_cnt);
  core_num_infos_holder->soc_vec_core_num = static_cast<int32_t>(platform_info.soc_info.vector_core_cnt);
  core_num_infos_holder->global_aicore_num = static_cast<int32_t>(platform_holder->GetCoreNumByType("AiCore"));
  core_num_infos_holder->global_vec_core_num = static_cast<int32_t>(platform_holder->GetCoreNumByType("VectorCore"));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildPlatformOutputs(const ge::FastNode *node, KernelContext *context) {
  (void)node;

  auto platform_chain = context->GetOutput(0);
  GE_ASSERT_NOTNULL(platform_chain);
  auto platform_info = new (std::nothrow) fe::PlatFormInfos();
  platform_chain->SetWithDefaultDeleter(platform_info);

  auto core_num_infos_chain = context->GetOutput(1);
  GE_ASSERT_NOTNULL(core_num_infos_chain);
  auto core_num_infos = new (std::nothrow) CoreNumInfos();
  core_num_infos_chain->SetWithDefaultDeleter(core_num_infos);

  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(GetPlatformInfo).RunFunc(GetPlatformInfo).OutputsCreator(BuildPlatformOutputs);

ge::graphStatus AppendCoreTypeToPlatform(KernelContext *context) {
  auto platform_holder = context->GetInputValue<fe::PlatFormInfos *>(0);
  GE_ASSERT_NOTNULL(platform_holder);
  auto core_type_holder = context->GetInputPointer<CoreTypeIndex>(1);
  GE_ASSERT_NOTNULL(core_type_holder);
  const auto op_ai_core_num_holder = context->GetInputValue<int32_t>(2);
  const auto op_vector_core_num_holder = context->GetInputValue<int32_t>(3);
  const auto core_num_infos_holder = context->GetInputValue<CoreNumInfos *>(4);

  // 用coreType和算子级核数刷新第一个输出：PlatformInfos
  auto out_platform_holder = context->GetOutputPointer<fe::PlatFormInfos>(0);
  GE_ASSERT_NOTNULL(out_platform_holder);
  *out_platform_holder = *platform_holder;
  std::map<std::string, std::string> res;
  (void)out_platform_holder->GetPlatformResWithLock(kSocInfo, res);
  bool is_op_core_num_set = false;
  if (op_ai_core_num_holder > 0) {
    GE_ASSERT_SUCCESS(ge::CoreNumUtils::UpdateCoreCountWithOpDesc(kAiCoreNum, std::to_string(op_ai_core_num_holder),
                                                    core_num_infos_holder->soc_aicore_num, kAicCntKeyIni, res));
    // .ini文件中ai_core_cnt和cube_core_cnt是相同的，直接赋值
    res[kCubeCntKeyIni] = res[kAicCntKeyIni];
    is_op_core_num_set = true;
  }
  if (op_vector_core_num_holder > 0) {
    GE_ASSERT_SUCCESS(ge::CoreNumUtils::UpdateCoreCountWithOpDesc(kVectorCoreNum, std::to_string(op_vector_core_num_holder),
                                                    core_num_infos_holder->soc_vec_core_num, kAivCntKeyIni, res));
    is_op_core_num_set = true;
  }

  if (is_op_core_num_set) {
    int32_t device_id = -1;
    GE_CHK_RT_RET(aclrtGetDevice(&device_id));
    fe::PlatFormInfos platform_infos_bak;
    GE_ASSERT_TRUE(fe::PlatformInfoManager::GeInstance().GetRuntimePlatformInfosByDevice(
                   static_cast<uint32_t>(device_id), platform_infos_bak, true) == 0,
                   "Get runtime platformInfos by device failed, deviceId = %d", device_id);
    platform_infos_bak.SetPlatformResWithLock(kSocInfo, res);
    *out_platform_holder = platform_infos_bak;
  }

  const auto iter = kCoreTypeReflection.find(*core_type_holder);
  if (iter != kCoreTypeReflection.end()) {
    out_platform_holder->SetCoreNumByCoreType(iter->second);
    GELOGD("Set core type to %s.", iter->second.c_str());
  } else {
    out_platform_holder->SetCoreNumByCoreType(kDefaultCoreType);
    GELOGD("Set core type to %s.", kDefaultCoreType);
  }

  // 用算子级核数填充第二个输出：CoreNumInfos
  auto out_core_num_infos_holder = context->GetOutputPointer<CoreNumInfos>(1);
  GE_ASSERT_NOTNULL(out_core_num_infos_holder);
  *out_core_num_infos_holder = *core_num_infos_holder;
  out_core_num_infos_holder->op_aicore_num = op_ai_core_num_holder;
  out_core_num_infos_holder->op_vec_core_num = op_vector_core_num_holder;

  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(AppendCoreTypeToPlatform).RunFunc(AppendCoreTypeToPlatform).OutputsCreator(BuildPlatformOutputs);
}  // namespace kernel
}  // namespace gert
