/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/platform_utils.h"
#include "common/fe_log.h"
#include "common/string_utils.h"
#include "common/aicore_util_constants.h"
#include "common/fe_type_utils.h"
#include "platform/platform_info.h"
#include "common/fe_report_error.h"
#include "ge/ge_api_types.h"
#include "graph/ge_context.h"
#include "transfer_shape_utils.h"

namespace fe {
namespace {
const int32_t kDecimal = 10;
const size_t kAicVersionSize = 3;
constexpr const char* kShortSocVersionAscend310P = "Ascend310P";
constexpr const char* kSocVersionAscend910 = "Ascend910";
constexpr const char* kSocVersionAscend910A = "Ascend910A";
const std::unordered_set<std::string> kValidCoreTypes = {"AiCore", "VectorCore"};

// maps aic version to ISA arch VERSION
const std::map<int32_t, ISAArchVersion> kAicIsaArchVersionMap {
        {100, ISAArchVersion::EN_ISA_ARCH_V100},
        {200, ISAArchVersion::EN_ISA_ARCH_V200},
        {202, ISAArchVersion::EN_ISA_ARCH_V200},
        {210, ISAArchVersion::EN_ISA_ARCH_V200},
        {220, ISAArchVersion::EN_ISA_ARCH_V220},
        {300, ISAArchVersion::EN_ISA_ARCH_V300},
        {310, ISAArchVersion::EN_ISA_ARCH_V300},
        {350, ISAArchVersion::EN_ISA_ARCH_V350}
};
const std::map<ISAArchVersion, std::string> kIsaArchVersionMapStr {
        {ISAArchVersion::EN_ISA_ARCH_V100, "v100"},
        {ISAArchVersion::EN_ISA_ARCH_V200, "v200"},
        {ISAArchVersion::EN_ISA_ARCH_V220, "v220"},
        {ISAArchVersion::EN_ISA_ARCH_V300, "v300"},
        {ISAArchVersion::EN_ISA_ARCH_V350, "v350"}
};
}

const std::map<PlatformUtils::PlatformInfoItem, PlatformUtils::PmItemParseFunc> PlatformUtils::kPmItemParseFuncMap {
        {PlatformUtils::PlatformInfoItem::IsaArchVersion, &PlatformUtils::ParseIsaArchVersion},
        {PlatformUtils::PlatformInfoItem::SupportFixpipe, &PlatformUtils::ParseSupportFixpipe},
        {PlatformUtils::PlatformInfoItem::FixpipeSupportMultiOutput, &PlatformUtils::ParseFixpipeSupportMultiOutput},
        {PlatformUtils::PlatformInfoItem::ContextSwitch, &PlatformUtils::ParseContextSwitch},
        {PlatformUtils::PlatformInfoItem::DsaWorkspaceSize, &PlatformUtils::ParseDsaWorkspaceSize},
        {PlatformUtils::PlatformInfoItem::FftsMode, &PlatformUtils::ParseFftsMode},
        {PlatformUtils::PlatformInfoItem::CubeVecState, &PlatformUtils::ParseCubeVecState},
        {PlatformUtils::PlatformInfoItem::AllowHf32, &PlatformUtils::ParseAllowHf32},
        {PlatformUtils::PlatformInfoItem::CubeHighPrecison, &PlatformUtils::ParseCubeHighPrecison},
        {PlatformUtils::PlatformInfoItem::L2Type, &PlatformUtils::ParseL2Type},
        {PlatformUtils::PlatformInfoItem::L2CacheMode, &PlatformUtils::ParseL2CacheMode},
        {PlatformUtils::PlatformInfoItem::SupportVectorEngine, &PlatformUtils::ParseSupportVectorEngine},
        {PlatformUtils::PlatformInfoItem::SpecifiedMemBase, &PlatformUtils::ParseSpecifiedMemBase},
        {PlatformUtils::PlatformInfoItem::HardwareCoreSync, &PlatformUtils::ParseHardwareCoreSync}
};

PlatformUtils::PlatformUtils() : is_init_(false), ai_core_num_(0) {}

PlatformUtils::~PlatformUtils() {}

PlatformUtils& PlatformUtils::Instance() {
  static PlatformUtils platform_utils;
  return platform_utils;
}

Status PlatformUtils::Initialize(const std::map<string, string> &options) {
  if (is_init_) {
    return SUCCESS;
  }

  FE_LOGD("Begin to init platform utils info of soc version.");
  const std::map<string, string>::const_iterator iter = options.find(ge::SOC_VERSION);
  if (iter != options.end() && !iter->second.empty()) {
    FE_LOGI("The value of [%s] from ge options is [%s].", ge::SOC_VERSION.c_str(), iter->second.c_str());
    soc_version_ = iter->second;
    if (soc_version_ == fe::kSocVersionAscend910) {
      FE_LOGI("The soc_version is Ascend910, it will be replaced by Ascend910A.");
      soc_version_ =  fe::kSocVersionAscend910A;
    }
  } else {
    ErrorMessageDetail err_msg(EM_INPUT_OPTION_INVALID,
        {soc_version_, ge::SOC_VERSION, "soc_version is not configured or is empty"});
    ReportErrorMessage(err_msg);
    FE_LOGE("Parameter ge.soc_version is not present in options or has an empty value.");
    return FAILED;
  }

  FE_LOGI("Current soc version is [%s].", soc_version_.c_str());
  PlatformInfoManager &platform_inst = PlatformInfoManager::Instance();
  if (InitPlatformInfo(platform_inst, options) != SUCCESS) {
    return FAILED;
  }
  // init ge_instance for transfer shape utils
  PlatformInfoManager &ge_instance = PlatformInfoManager::GeInstance();
  if (InitPlatformInfo(ge_instance, options) != SUCCESS) {
    return FAILED;
  }

  if (InitPlatformInfoParameters() != SUCCESS) {
    return FAILED;
  }
  (void)transformer::TransferShapeUtils::InitPlatformInfo();
  is_init_ = true;
  FE_LOGD("Platform utils has been initialized successfully.");
  return SUCCESS;
}

Status PlatformUtils::InitPlatformInfo(PlatformInfoManager &platform_inst,
                                       const std::map<std::string, std::string> &options) const {
  if (platform_inst.InitializePlatformInfo() != SUCCESS) {
    REPORT_FE_ERROR("[FusionMngr][InitPlatCfg] Initialize platform_info failed.");
    return FAILED;
  }

  PlatformInfo platform_info;
  OptionalInfo optional_info;
  if (platform_inst.GetPlatformInfo(soc_version_, platform_info, optional_info) != SUCCESS) {
    FE_LOGE("[FusionMngr][InitPlatCfg] Cannot find platform config file by soc version[%s].", soc_version_.c_str());
    ErrorMessageDetail err_msg(EM_INPUT_OPTION_INVALID,
        {soc_version_, ge::SOC_VERSION, "The corresponding platform configuration file cannot be found based on the configured soc_version"});
    ReportErrorMessage(err_msg);
    return FAILED;
  }

  PlatFormInfos platform_infos;
  OptionalInfos optional_infos;
  if (platform_inst.GetPlatformInfos(soc_version_, platform_infos, optional_infos) != SUCCESS) {
    FE_LOGE("[PlatformUtils][Init] Cannot find platform config file by soc version[%s].", soc_version_.c_str());
    ErrorMessageDetail err_msg(EM_INPUT_OPTION_INVALID,
        {soc_version_, ge::SOC_VERSION, "The corresponding platform configuration file cannot be found based on the configured soc_version"});
    ReportErrorMessage(err_msg);
    return FAILED;
  }

  if (InitOptionalInfo(options, platform_infos, optional_info, optional_infos) != SUCCESS) {
    return FAILED;
  }
  // set param to optional info
  platform_inst.SetOptionalCompilationInfo(optional_info);
  platform_inst.SetOptionalCompilationInfo(optional_infos);
  return SUCCESS;
}

Status PlatformUtils::InitOptionalInfo(const std::map<std::string, std::string> &options,
                                       PlatFormInfos &platform_infos,
                                       OptionalInfo &optional_info, OptionalInfos &optional_infos) const {
  optional_info.soc_version = soc_version_;
  optional_infos.SetSocVersion(soc_version_);
  FE_LOGD("Soc version for platform's optional information is [%s].", soc_version_.c_str());

  // core type
  auto iter = options.find(ge::CORE_TYPE);
  std::string core_type = "AiCore";
  if (iter != options.cend() && !iter->second.empty()) {
    FE_LOGD("Param[%s] of options is [%s].", ge::CORE_TYPE.c_str(), iter->second.c_str());
    if (kValidCoreTypes.count(iter->second) == 0) {
      ErrorMessageDetail err_msg(EM_INPUT_OPTION_INVALID,
          {iter->second, ge::CORE_TYPE, "Only AI Core or Vector Core is supported"});
      ReportErrorMessage(err_msg);
      FE_LOGE("[PlatformUtils][Init] Invalid param value [%s] for [%s].",
              iter->second.c_str(), ge::CORE_TYPE.c_str());
      return FAILED;
    }
    core_type = iter->second;
  }
  FE_LOGD("Core type for platform's optional info is set to [%s].", core_type.c_str());
  optional_info.core_type = core_type;
  optional_infos.SetCoreType(core_type);

  // l1_fusion_flag
  iter = options.find(ge::BUFFER_OPTIMIZE);
  std::string l1_flag_str = (iter != options.cend() && iter->second == L1_OPTIMIZE) ? kStrTrue : kStrFalse;
  FE_LOGD("L1 fusion flag for platform's optional info is set to [%s].", l1_flag_str.c_str());
  optional_info.l1_fusion_flag = l1_flag_str;
  optional_infos.SetL1FusionFlag(l1_flag_str);

  // aicore cnt from platform
  int32_t ai_core_cnt = 1;
  std::string ai_core_cnt_str;
  if (platform_infos.GetPlatformRes("SoCInfo", "ai_core_cnt", ai_core_cnt_str)) {
    FE_LOGD("Ai core count in SoCInfo from platform is [%s].", ai_core_cnt_str.c_str());
    ai_core_cnt = static_cast<int32_t>(std::atoi(ai_core_cnt_str.c_str()));
    if (ai_core_cnt <= 0) {
      REPORT_FE_ERROR("Ai core count [%d] for soc version [%s] is invalid.", ai_core_cnt, soc_version_.c_str());
      return FAILED;
    }
  }

  // aicore num from options
  int32_t aicore_num = 0;
  iter = options.find(ge::AICORE_NUM);
  if (iter != options.cend() && !iter->second.empty()) {
    FE_LOGD("Param[%s] of options is [%s].", ge::AICORE_NUM.c_str(), iter->second.c_str());
    aicore_num = static_cast<int32_t>(std::atoi(iter->second.c_str()));
    if (aicore_num <= 0 || (std::to_string(aicore_num) != iter->second)) {
      std::string aicore_num_config = ge::GetContext().GetReadableName(ge::AICORE_NUM);
      ErrorMessageDetail err_msg(EM_INPUT_OPTION_INVALID,
          {iter->second, aicore_num_config, "The AI core num should be a positive integer"});
      ReportErrorMessage(err_msg);
      REPORT_FE_ERROR("[FusionMngr][ChkOptiCmpt] The ai_core_num should be positive integer.");
      return FAILED;
    }
  } else {
    aicore_num = ai_core_cnt;
  }

  if (aicore_num > ai_core_cnt) {
    std::string aicore_num_config = ge::GetContext().GetReadableName(ge::AICORE_NUM);
    ErrorMessageDetail err_msg(EM_AICORENUM_OUT_OF_RANGE,
                               {std::to_string(aicore_num), aicore_num_config, std::to_string(ai_core_cnt)});
    ReportErrorMessage(err_msg);
    REPORT_FE_ERROR("[Platform][Init] ai_core_num[%d] is out range of platformInfo ai_core_num (0, %d].", aicore_num,
                    ai_core_cnt);
    return FAILED;
  }
  FE_LOGD("AI core number for platform's optional info is [%d].", aicore_num);
  optional_info.ai_core_num = static_cast<uint32_t>(aicore_num);
  optional_infos.SetAICoreNum(static_cast<uint32_t>(aicore_num));

  return SUCCESS;
}

Status PlatformUtils::InitVirTypeList(PlatFormInfos &platform_infos) {
  std::string vir_type_str;
  std::vector<std::string> vir_type_list;
  (void)platform_infos.GetPlatformResWithLock(kCfgSocInfo, kCfgVirTypeList, vir_type_str);
  FE_LOGD("Parameter[vir_type_list] from platform is %s.", vir_type_str.c_str());
  StringUtils::SplitWithTrim(vir_type_str, ',', vir_type_list);
  for (auto &vir_type : vir_type_list) {
    int core_num;
    try {
      core_num = stoi(vir_type);
    } catch (...) {
      FE_LOGE("Convert %s to int value failed.", vir_type.c_str());
      return FAILED;
    }
    if (core_num <= 0) {
      FE_LOGE("core_num [%d] from the platform is invalid.", core_num);
      return FAILED;
    }
    vir_type_list_.push_back(static_cast<uint32_t>(core_num));
  }
  return SUCCESS;
}

Status PlatformUtils::InitPlatformInfoParameters() {
  PlatFormInfos platform_infos;
  OptionalInfos optional_infos;
  if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_infos, optional_infos) != SUCCESS) {
    FE_LOGE("SOC version [%s] platform not found.", soc_version_.c_str());
    return FAILED;
  }

  // short soc version
  if (!platform_infos.GetPlatformRes("version", "Short_SoC_version", short_soc_version_) ||
      short_soc_version_.empty()) {
    FE_LOGE("Short soc version of [%s] is empty.", soc_version_.c_str());
    return FAILED;
  }
  FE_LOGD("Short soc version of [%s] is [%s].", soc_version_.c_str(), short_soc_version_.c_str());

  // aic num
  ai_core_num_ = optional_infos.GetAICoreNum();
  FE_LOGD("AI core count for [%s] is [%u].", soc_version_.c_str(), ai_core_num_);

  // init cfg item from platform
  for (const auto &item_pair : kPmItemParseFuncMap) {
    pm_item_vec_[static_cast<size_t>(item_pair.first)] = item_pair.second(platform_infos);
  }

  // init vir_type_list
  if (InitVirTypeList(platform_infos) != SUCCESS) {
    FE_LOGE("Failed to init vir_type_list.");
    return FAILED;
  }
  return SUCCESS;
}

int64_t PlatformUtils::ParseIsaArchVersion(PlatFormInfos &platform_infos) {
  ISAArchVersion isa_arch_version = ISAArchVersion::EN_ISA_ARCH_V100;
  // aic version, ISAArchVersion
  std::string aic_version_str;
  if (!platform_infos.GetPlatformRes("version", "AIC_version", aic_version_str) || aic_version_str.empty()) {
    FE_LOGW("Aic version is empty.");
    return static_cast<int64_t>(isa_arch_version);
  }
  FE_LOGD("Aic version is [%s].", aic_version_str.c_str());
  std::vector<string> aic_version_vec = StringUtils::Split(aic_version_str, "-");
  if (aic_version_vec.size() < kAicVersionSize) {
    FE_LOGW("The AIC version [%s] is invalid.", aic_version_str.c_str());
    return static_cast<int64_t>(isa_arch_version);
  }
  int32_t aic_version = atoi(aic_version_vec[2].c_str());
  auto iter_aic = kAicIsaArchVersionMap.find(aic_version);
  if (iter_aic != kAicIsaArchVersionMap.end()) {
    isa_arch_version = iter_aic->second;
  }
  FE_LOGI("ISA arch version is [%s].", GetIsaArchVersionStr(isa_arch_version).c_str());
  return static_cast<int64_t>(isa_arch_version);
}

int64_t PlatformUtils::ParseSupportFixpipe(PlatFormInfos &platform_infos) {
  bool is_support_fixpipe = false;
  // is support fixpipe
  std::string is_support_fixpipe_str;
  if (platform_infos.GetPlatformRes("AICoreSpec", "support_fixpipe", is_support_fixpipe_str)) {
    FE_LOGD("Parameter [support_fixpipe] of AICoreSpec from the platform is [%s].", is_support_fixpipe_str.c_str());
    is_support_fixpipe = static_cast<bool>(std::atoi(is_support_fixpipe_str.c_str()));
  }
  FE_LOGI("Is support fixpipe: [%d]", is_support_fixpipe);
  return static_cast<int64_t>(is_support_fixpipe);
}

int64_t PlatformUtils::ParseFixpipeSupportMultiOutput(PlatFormInfos &platform_infos) {
  bool is_fixpipe_support_multi_output = true;
  std::string is_fixpipe_support_multi_output_str;
  if (platform_infos.GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_fix_pipe_support_multiple_output",
      is_fixpipe_support_multi_output_str) && !is_fixpipe_support_multi_output_str.empty()) {
    FE_LOGD("Parameter[fixpipe_support_multi_output] of AICoreintrinsicDtypeMap from platform is [%s].",
            is_fixpipe_support_multi_output_str.c_str());
    StringUtils::ToLowerString(is_fixpipe_support_multi_output_str);
    if (is_fixpipe_support_multi_output_str == "false") {
      is_fixpipe_support_multi_output = false;
    }
  }
  FE_LOGI("The flag for is_fixpipe_support_multi_output is set to [%d]", is_fixpipe_support_multi_output);
  return static_cast<int64_t>(is_fixpipe_support_multi_output);
}

int64_t PlatformUtils::ParseContextSwitch(PlatFormInfos &platform_infos) {
  bool is_context_switch = false;
  // is support context switch
  std::string context_switch_str;
  if (platform_infos.GetPlatformRes("SoCInfo", "context_switch", context_switch_str)) {
    FE_LOGD("Context switch of AICoreSpec from platform: %s.", context_switch_str.c_str());
    is_context_switch = static_cast<bool>(std::atoi(context_switch_str.c_str()));
  }
  FE_LOGI("Is support context switch: [%d]", is_context_switch);
  return static_cast<int64_t>(is_context_switch);
}

int64_t PlatformUtils::ParseDsaWorkspaceSize(PlatFormInfos &platform_infos) {
  int64_t dsa_workspace_size = 0;
  // dsa workspace size
  std::string dsa_workspace_size_str;
  if (platform_infos.GetPlatformRes("DSARandom", "CounterWorkspaceSize", dsa_workspace_size_str)) {
    FE_LOGD("Parameter [CounterWorkspaceSize] of DSARandom from the platform is [%s].", dsa_workspace_size_str.c_str());
    dsa_workspace_size = std::strtoll(dsa_workspace_size_str.c_str(), nullptr, kDecimal);
  }
  FE_LOGI("Dsa workspace size is [%ld]", dsa_workspace_size);
  return dsa_workspace_size;
}

int64_t PlatformUtils::ParseFftsMode(PlatFormInfos &platform_infos) {
  AICoreMode ffts_mode = AICoreMode::FFTS_MODE_NO_FFTS;
  // init ffts mode
  std::string ffts_mode_str;
  if (platform_infos.GetPlatformRes(kCfgSocInfo, kAICoreMode, ffts_mode_str)) {
    FE_LOGD("AI Core mode is [%s].", ffts_mode_str.c_str());
    if (ffts_mode_str == kFFTSPlusMode) {
      ffts_mode = FFTS_MODE_FFTS_PLUS;
    } else if (ffts_mode_str == kFFTSMode) {
      ffts_mode = FFTS_MODE_FFTS;
    }
  }
  FE_LOGI("Ffts mode is [%ld].", static_cast<int64_t>(ffts_mode));
  return static_cast<int64_t>(ffts_mode);
}

int64_t PlatformUtils::ParseCubeVecState(PlatFormInfos &platform_infos) {
  CubeVecStateNew cube_vec_state = CubeVecStateNew::CUBE_VEC_UNKNOWN;
  // init cube vec state
  std::string cube_vec_state_str;
  if (platform_infos.GetPlatformRes(kCfgSocInfo, kCubeVecCombine, cube_vec_state_str)) {
    FE_LOGD("Parameter [cube_vector_combine] is set to [%s]", cube_vec_state_str.c_str());
    std::vector<std::string> cube_vec_states = StringUtils::Split(cube_vec_state_str, ',');
    if (!cube_vec_states.empty()) {
      if (std::find(cube_vec_states.begin(), cube_vec_states.end(), kCubeVecFuse) != cube_vec_states.end()) {
        cube_vec_state = CubeVecStateNew::CUBE_VEC_FUSE;
      } else if (std::find(cube_vec_states.begin(), cube_vec_states.end(), kCubeVecSplit) != cube_vec_states.end()) {
        cube_vec_state = CubeVecStateNew::CUBE_VEC_SPLIT;
      }
    }
  }
  FE_LOGI("Cube vector state is [%ld].", static_cast<int64_t>(cube_vec_state));
  return static_cast<int64_t>(cube_vec_state);
}

int64_t PlatformUtils::ParseCubeHighPrecison(PlatFormInfos &platform_infos) {
  bool enable_cube_high_precison = false;
  // init enable_cube_c08
  std::map<std::string, std::vector<std::string>> intrinsic_map = platform_infos.GetAICoreIntrinsicDtype();
  auto iter_inc = intrinsic_map.find("Intrinsic_mmad");
  if (iter_inc != intrinsic_map.end()) {
    std::vector<std::string> intrinsic_vec = iter_inc->second;
    // init enable_cube_c08
    auto intrs = std::find(intrinsic_vec.cbegin(), intrinsic_vec.cend(), "f322f32");
    if (intrs != intrinsic_vec.cend()) {
      enable_cube_high_precison = true;
    }
  }
  FE_LOGI("Enable cube high precision is set to [%d].", enable_cube_high_precison);
  return static_cast<int64_t>(enable_cube_high_precison);
}

int64_t PlatformUtils::ParseAllowHf32(PlatFormInfos &platform_infos) {
  bool allow_hf32 = false;
  // init enable_cube_c08
  std::map<std::string, std::vector<std::string>> intrinsic_map = platform_infos.GetAICoreIntrinsicDtype();
  auto iter_inc = intrinsic_map.find("Intrinsic_mmad");
  if (iter_inc != intrinsic_map.end()) {
    std::vector<std::string> intrinsic_vec = iter_inc->second;
    // init allow_hf32
    auto intrs = std::find(intrinsic_vec.cbegin(), intrinsic_vec.cend(), "h322f32");
    if (intrs != intrinsic_vec.cend()) {
      allow_hf32 = true;
    }
  }
  FE_LOGI("Enable allow hf32 is set to [%d].", allow_hf32);
  return static_cast<int64_t>(allow_hf32);
}

int64_t PlatformUtils::ParseL2Type(PlatFormInfos &platform_infos) {
  L2Type l2_type = L2Type::Cache;
  std::string l2_type_str;
  if (platform_infos.GetPlatformRes("SoCInfo", "l2_type", l2_type_str)) {
    FE_LOGD("L2 type from platform: %s.", l2_type_str.c_str());
    if (l2_type_str == "1") {
      l2_type = L2Type::Buff;
    }
  }
  FE_LOGI("L2 type is %ld.", static_cast<int64_t>(l2_type));
  return static_cast<int64_t>(l2_type);
}

int64_t PlatformUtils::ParseL2CacheMode(PlatFormInfos &platform_infos) {
  L2CacheMode l2_cache_mode = L2CacheMode::DEFAULT;
  std::string l2_cache_mode_str;
  if (platform_infos.GetPlatformRes("SoCInfo", "l2_cache_mode", l2_cache_mode_str)) {
    FE_LOGD("L2 cache mode received from platform: [%s].", l2_cache_mode_str.c_str());
    if (l2_cache_mode_str == "cmo") {
      l2_cache_mode = L2CacheMode::CMO;
    } else if (l2_cache_mode_str == "rc") {
      l2_cache_mode = L2CacheMode::RC;
    }
  }
  FE_LOGI("L2 cache mode is %ld.", static_cast<int64_t>(l2_cache_mode));
  return static_cast<int64_t>(l2_cache_mode);
}

int64_t PlatformUtils::ParseSupportVectorEngine(PlatFormInfos &platform_infos) {
  bool is_support_vector_engine = false;
  // is support vector engine
  std::string is_support_vector_engine_str;
  if (platform_infos.GetPlatformRes("SoCInfo", "support_vector_engine", is_support_vector_engine_str)) {
    FE_LOGD("Parameter [support_vector_engine] received from platform: %s.", is_support_vector_engine_str.c_str());
    is_support_vector_engine = static_cast<bool>(std::atoi(is_support_vector_engine_str.c_str()));
  }
  FE_LOGI("Parameter [support_vector_engine] value is [%d]", is_support_vector_engine);
  return static_cast<int64_t>(is_support_vector_engine);
}

int64_t PlatformUtils::ParseSpecifiedMemBase(PlatFormInfos &platform_infos) {
  bool is_specified_mem_base = false;
  // is support specified mem base
  std::string is_specified_mem_base_str;
  if (platform_infos.GetPlatformRes("SoCInfo", "specified_mem_base", is_specified_mem_base_str)) {
    FE_LOGD("Parameter [specified_mem_base] from the platform is [%s].", is_specified_mem_base_str.c_str());
    is_specified_mem_base = static_cast<bool>(std::atoi(is_specified_mem_base_str.c_str()));
  }
  FE_LOGI("Parameter [specified_mem_base] is set to %d", is_specified_mem_base);
  return static_cast<int64_t>(is_specified_mem_base);
}

int64_t PlatformUtils::ParseHardwareCoreSync(PlatFormInfos &platform_infos) {
  bool is_core_sync = false;
  std::string hardware_sync_str;
  if (platform_infos.GetPlatformRes(kCfgSocInfo, kHardwareSyncBetweenCore, hardware_sync_str)) {
    FE_LOGD("Parameter[hardware_sync_betweencore] is [%s]", hardware_sync_str.c_str());
    if (hardware_sync_str == "1") {
      is_core_sync = true;
    }
  }
  FE_LOGI("Hardware support status for core sync is [%d]", is_core_sync);
  return static_cast<int64_t>(is_core_sync);
}

Status PlatformUtils::Finalize() {
  if (!is_init_) {
    return SUCCESS;
  }

  if (PlatformInfoManager::Instance().Finalize() != SUCCESS) {
    FE_LOGE("Failed to finalize platform info.");
    return FAILED;
  }
  is_init_ = false;
  FE_LOGD("PlatformUtils has been finalized.");
  return SUCCESS;
}

const std::string& PlatformUtils::GetSocVersion() const {
  return soc_version_;
}

const std::string& PlatformUtils::GetShortSocVersion() const {
  return short_soc_version_;
}

bool PlatformUtils::IsEnableCubeHighPrecision() const {
  return static_cast<bool>(pm_item_vec_[static_cast<size_t>(PlatformInfoItem::CubeHighPrecison)]);
}

bool PlatformUtils::IsEnableAllowHF32() const {
  return static_cast<bool>(pm_item_vec_[static_cast<size_t>(PlatformInfoItem::AllowHf32)]);
}

bool PlatformUtils::IsSupportFixpipe() const {
  return static_cast<bool>(pm_item_vec_[static_cast<size_t>(PlatformInfoItem::SupportFixpipe)]);
}

bool PlatformUtils::IsFixpipeSupportMultiOutput() const {
  return static_cast<bool>(pm_item_vec_[static_cast<size_t>(PlatformInfoItem::FixpipeSupportMultiOutput)]);
}

bool PlatformUtils::IsSupportContextSwitch() const {
  return static_cast<bool>(pm_item_vec_[static_cast<size_t>(PlatformInfoItem::ContextSwitch)]);
}

int64_t PlatformUtils::GetDsaWorkspaceSize() const {
  return static_cast<int64_t>(pm_item_vec_[static_cast<size_t>(PlatformInfoItem::DsaWorkspaceSize)]);
}

AICoreMode PlatformUtils::GetFftsMode() const {
  return static_cast<AICoreMode>(pm_item_vec_[static_cast<size_t>(PlatformInfoItem::FftsMode)]);
}

CubeVecStateNew PlatformUtils::GetCubeVecState() const {
  return static_cast<CubeVecStateNew>(pm_item_vec_[static_cast<size_t>(PlatformInfoItem::CubeVecState)]);
}

ISAArchVersion PlatformUtils::GetIsaArchVersion() const {
  return static_cast<ISAArchVersion>(pm_item_vec_[static_cast<size_t>(PlatformInfoItem::IsaArchVersion)]);
}

std::string PlatformUtils::GetIsaArchVersionStr(const ISAArchVersion isa_arch_version) {
  std::string isa_version_str;
  auto iter = kIsaArchVersionMapStr.find(isa_arch_version);
  if (iter != kIsaArchVersionMapStr.end()) {
    isa_version_str = iter->second;
  }
  return isa_version_str;
}

std::string PlatformUtils::GetIsaArchVersionStr() const {
  return GetIsaArchVersionStr(GetIsaArchVersion());
}

L2Type PlatformUtils::GetL2Type() const {
  return static_cast<L2Type>(pm_item_vec_[static_cast<size_t>(PlatformInfoItem::L2Type)]);
}

bool PlatformUtils::IsEnableL2CacheCmo() const {
  L2CacheMode l2_cache_mode =
      static_cast<L2CacheMode>(pm_item_vec_[static_cast<size_t>(PlatformInfoItem::L2CacheMode)]);
  return GetL2Type() == L2Type::Cache && l2_cache_mode == L2CacheMode::CMO;
}

bool PlatformUtils::IsEnableL2CacheRc() const {
  L2CacheMode l2_cache_mode =
      static_cast<L2CacheMode>(pm_item_vec_[static_cast<size_t>(PlatformInfoItem::L2CacheMode)]);
  return GetL2Type() == L2Type::Cache && l2_cache_mode == L2CacheMode::RC;
}

bool PlatformUtils::IsSupportVectorEngine() const {
  return static_cast<bool>(pm_item_vec_[static_cast<size_t>(PlatformInfoItem::SupportVectorEngine)]);
}

bool PlatformUtils::IsSpecifiedMemBase() const {
  return static_cast<bool>(pm_item_vec_[static_cast<size_t>(PlatformInfoItem::SpecifiedMemBase)]);
}

bool PlatformUtils::IsHardwareSupportCoreSync() const  {
  return static_cast<bool>(pm_item_vec_[static_cast<size_t>(PlatformInfoItem::HardwareCoreSync)]);
}

bool PlatformUtils::IsDCSoc() const {
  return short_soc_version_ == kShortSocVersionAscend310P;
}

const std::vector<uint32_t> &PlatformUtils::GetVirTypeList() const {
  return vir_type_list_;
}

uint32_t PlatformUtils::GetAICoreNum() const {
  return ai_core_num_;
}
}
