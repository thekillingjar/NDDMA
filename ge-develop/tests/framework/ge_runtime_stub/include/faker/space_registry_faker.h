/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_UT_GE_RUNTIME_V2_SPACE_REGISTRY_FAKER_H_
#define AIR_CXX_TESTS_UT_GE_RUNTIME_V2_SPACE_REGISTRY_FAKER_H_

#include "common/debug/ge_log.h"
#include "common/util/mem_utils.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace gert {
using OpTypesToImplMap = std::map<OpImplRegisterV2::OpType, OpImplKernelRegistry::OpImplFunctionsV2>;
int SuperSystem(const char *cmd, char *retmsg, int msg_len);
std::string GetTempOppBasePath();
void DestroyTempOppPath();
std::vector<std::string> CreateSceneInfo();
void CreateVersionInfo();
void DestroyVersionInfo();
void LoadDefaultSpaceRegistry();
void LoadMainSpaceRegistry();
void UnLoadDefaultSpaceRegistry();
void CreateOpmasterSoEnvInfoFunc(std::string opp_path);
void CreateVendorsOppSo(const std::string &opp_path, const std::string &customize_1 = "",
                        const std::string &customize_2 = "");

void CreateBuiltInSplitAndUpgradedSo(std::vector<std::string> &paths);
void CreateBuiltInSubPkgSo(std::vector<std::string> &paths);
class SpaceRegistryFaker {
 public:
  ~SpaceRegistryFaker();
  OpImplSpaceRegistryV2Ptr Build();
  OpImplSpaceRegistryV2Ptr BuildMainSpace();

  std::shared_ptr<OpImplSpaceRegistryV2Array> BuildRegistryArray();
  std::shared_ptr<OpImplSpaceRegistryV2Array> BuildMainSpaceRegistryArray();
  static OpTypesToImplMap *SavePreRegisteredSpaceRegistry();
  static void RestorePreRegisteredSpaceRegistry();
  static void UpdateOpImplToDefaultSpaceRegistry();
  static void CreateDefaultSpaceRegistry(bool only_main_space = false);
  static void CreateDefaultSpaceRegistryImpl2(bool only_main_space = false);
  static void CreateDefaultSpaceRegistryImpl();

  static void SetefaultSpaceRegistryNull() {
    gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
  }

  static int32_t GetAllOpImplFunctions(const char *so_path, std::shared_ptr<gert::OpImplSpaceRegistryV2> &space_registry);
};
}  // namespace gert
#endif  //AIR_CXX_TESTS_UT_GE_RUNTIME_V2_SPACE_REGISTRY_FAKER_H_
