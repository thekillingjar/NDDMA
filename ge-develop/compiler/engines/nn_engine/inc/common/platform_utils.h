/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_PLATFORM_UTILS_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_PLATFORM_UTILS_H_

#include <string>
#include <map>
#include "graph/types.h"
#include "common/aicore_util_types.h"
#include "platform/platform_info.h"
#include "platform/platform_info_def.h"
#include "platform/platform_infos_def.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
class PlatformUtils {
 public:
  PlatformUtils(const PlatformUtils &) = delete;
  PlatformUtils &operator=(const PlatformUtils &) = delete;

  /**
   * Get the Singleton Instance by engine name
   */
  static PlatformUtils &Instance();

  Status Initialize(const std::map<std::string, std::string> &options);

  Status Finalize();

  const std::string& GetSocVersion() const;

  const std::string& GetShortSocVersion() const;

  bool IsEnableCubeHighPrecision() const;

  bool IsEnableAllowHF32() const;

  bool IsSupportFixpipe() const;

  bool IsFixpipeSupportMultiOutput() const;

  bool IsSupportContextSwitch() const;

  int64_t GetDsaWorkspaceSize() const;

  CubeVecStateNew GetCubeVecState() const;

  AICoreMode GetFftsMode() const;

  ISAArchVersion GetIsaArchVersion() const;

  std::string GetIsaArchVersionStr() const;

  L2Type GetL2Type() const;

  bool IsEnableL2CacheCmo() const;

  bool IsEnableL2CacheRc() const;

  bool IsSupportVectorEngine() const;

  bool IsSpecifiedMemBase() const;

  bool IsHardwareSupportCoreSync() const;

  bool IsDCSoc() const;

  const std::vector<uint32_t> &GetVirTypeList() const;

  uint32_t GetAICoreNum() const;

 private:
  PlatformUtils();
  ~PlatformUtils();

  Status InitPlatformInfo(PlatformInfoManager &platform_inst, const std::map<std::string, std::string> &options) const;
  Status InitOptionalInfo(const std::map<std::string, std::string> &options, PlatFormInfos &platform_infos,
                          OptionalInfo &optional_info, OptionalInfos &optional_infos) const;
  Status InitPlatformInfoParameters();
  Status InitVirTypeList(PlatFormInfos &platform_infos);
  static std::string GetIsaArchVersionStr(const ISAArchVersion isa_arch_version);

  static int64_t ParseIsaArchVersion(PlatFormInfos &platform_infos);
  static int64_t ParseCubeHighPrecison(PlatFormInfos &platform_infos);
  static int64_t ParseAllowHf32(PlatFormInfos &platform_infos);
  static int64_t ParseSupportFixpipe(PlatFormInfos &platform_infos);
  static int64_t ParseFixpipeSupportMultiOutput(PlatFormInfos &platform_infos);
  static int64_t ParseContextSwitch(PlatFormInfos &platform_infos);
  static int64_t ParseDsaWorkspaceSize(PlatFormInfos &platform_infos);
  static int64_t ParseFftsMode(PlatFormInfos &platform_infos);
  static int64_t ParseCubeVecState(PlatFormInfos &platform_infos);
  static int64_t ParseL2Type(PlatFormInfos &platform_infos);
  static int64_t ParseL2CacheMode(PlatFormInfos &platform_infos);
  static int64_t ParseSupportVectorEngine(PlatFormInfos &platform_infos);
  static int64_t ParseSpecifiedMemBase(PlatFormInfos &platform_infos);
  static int64_t ParseHardwareCoreSync(PlatFormInfos &platform_infos);

private:
  enum class PlatformInfoItem {
    IsaArchVersion = 0,
    SupportFixpipe,
    FixpipeSupportMultiOutput,
    ContextSwitch,
    DsaWorkspaceSize,
    FftsMode,
    CubeVecState,
    AllowHf32,
    CubeHighPrecison,
    L2Type,
    L2CacheMode,
    SupportVectorEngine,
    SpecifiedMemBase,
    HardwareCoreSync,
    ItemBottom
  };
  bool is_init_;
  std::string soc_version_;
  std::string short_soc_version_;
  uint32_t ai_core_num_;
  std::vector<uint32_t> vir_type_list_;
  std::array<int64_t, static_cast<size_t>(PlatformInfoItem::ItemBottom)> pm_item_vec_;
  using PmItemParseFunc = std::function<int64_t(PlatFormInfos &)>;
  static const std::map<PlatformInfoItem, PmItemParseFunc> kPmItemParseFuncMap;
};
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_PLATFORM_UTILS_H_
