/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DEPLOY_DEPLOYER_DEPLOYER_AUTHENTICATION_H_
#define AIR_RUNTIME_HETEROGENEOUS_DEPLOY_DEPLOYER_DEPLOYER_AUTHENTICATION_H_

#include <string>
#include "ge/ge_api_error_codes.h"
#include "common/config/configurations.h"

namespace ge {
class DeployerAuthentication {
 public:
  static DeployerAuthentication &GetInstance();

  Status Initialize(const std::string &auth_lib_path, bool need_deploy = false);

  Status AuthSign(const std::string &data, std::string &sign_data) const;

  Status AuthVerify(const std::string &data, const std::string &sign_data) const;

  void Finalize();

 private:
  DeployerAuthentication() = default;
  ~DeployerAuthentication();

  Status AuthDeploy() const;
  Status LoadAuthLib(const std::string &auth_lib_path);
  
  void *handle_ = nullptr;
};
}  // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_DEPLOY_DEPLOYER_DEPLOYER_AUTHENTICATION_H_
