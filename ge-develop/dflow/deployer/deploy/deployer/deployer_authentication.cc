/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/deployer/deployer_authentication.h"
#include "mmpa/mmpa_api.h"
#include "common/debug/log.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
struct SignResult;
using NewSignResult = SignResult *(*)();
using DeleteSignResult = void (*)(SignResult *sign_result);
using GetSignLength = uint32_t (*)(SignResult *sign_result);
using GetSignData = const unsigned char *(*)(SignResult *sign_result);
using DataFlowAuthMasterInit = int32_t(*)();
using DataFlowAuthSign = int32_t(*)(const unsigned char *data, uint32_t data_len, SignResult *sign_result);
using DataFlowAuthVerify = int32_t(*)(const unsigned char *data, uint32_t data_len,
                                      const unsigned char *sign_data, uint32_t sign_data_len);

constexpr const char_t *kFuncNameNewSignResult = "NewSignResult";
constexpr const char_t *kFuncNameDeleteSignResult = "DeleteSignResult";
constexpr const char_t *kFuncNameGetSignLength = "GetSignLength";
constexpr const char_t *kFuncNameGetSignData = "GetSignData";
constexpr const char_t *kFuncNameDataFlowAuthMasterInit = "DataFlowAuthMasterInit";
constexpr const char_t *kFuncNameDataFlowAuthSign = "DataFlowAuthSign";
constexpr const char_t *kFuncNameDataFlowAuthVerify = "DataFlowAuthVerify";
}  // namespace

DeployerAuthentication::~DeployerAuthentication() {
  Finalize();
}

DeployerAuthentication &DeployerAuthentication::GetInstance() {
  static DeployerAuthentication instance;
  return instance;
}

Status DeployerAuthentication::Initialize(const std::string &auth_lib_path, bool need_deploy) {
  if (auth_lib_path.empty()) {
    GELOGI("No need to auth for deployer");
    return SUCCESS;
  }
  GE_CHK_STATUS_RET(LoadAuthLib(auth_lib_path), "Failed to load auth lib.");
  if (need_deploy) {
    GE_CHK_STATUS_RET(AuthDeploy(), "Failed to auth deploy.");
  }
  GEEVENT("Initialize deployer auth successfully.");
  return SUCCESS;
}

Status DeployerAuthentication::LoadAuthLib(const std::string &auth_lib_path) {
  const auto open_flag =
      static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW) | static_cast<uint32_t>(MMPA_RTLD_GLOBAL));
  handle_ = mmDlopen(auth_lib_path.c_str(), open_flag);
  GE_CHK_BOOL_RET_STATUS(handle_ != nullptr, FAILED,
                         "[Dlopen][So] failed, so name = %s, error_msg = %s",
                         auth_lib_path.c_str(), mmDlerror());
  GELOGI("Open %s succeeded", auth_lib_path.c_str());
  return SUCCESS;
}

void DeployerAuthentication::Finalize() {
  if (handle_ != nullptr) {
    (void) mmDlclose(handle_);
    handle_ = nullptr;
  }
  GELOGI("Deployer authentication finalized");
}

Status DeployerAuthentication::AuthDeploy() const {
  GEEVENT("Deployer auth deploy begin.");
  auto proc = reinterpret_cast<DataFlowAuthMasterInit>(mmDlsym(handle_, kFuncNameDataFlowAuthMasterInit));
  GE_CHK_BOOL_RET_STATUS(proc != nullptr, FAILED, "[Dlsym][So] failed, func name = %s, error_msg = %s",
                         kFuncNameDataFlowAuthMasterInit, mmDlerror());
  auto ret = proc();
  GE_CHK_STATUS_RET(ret, "Failed to auth deploy, ret = %d.", ret);
  GEEVENT("Deployer auth deploy successfully.");
  return SUCCESS;
}

Status DeployerAuthentication::AuthSign(const std::string &data, std::string &sign_data) const {
  GEEVENT("Deployer auth sign begin, data = %s.", data.c_str());
  auto new_proc = reinterpret_cast<NewSignResult>(mmDlsym(handle_, kFuncNameNewSignResult));
  GE_CHK_BOOL_RET_STATUS(new_proc != nullptr, FAILED, "[Dlsym][So] failed, func name = %s, error_msg = %s",
                         kFuncNameNewSignResult, mmDlerror());
  auto delete_proc = reinterpret_cast<DeleteSignResult>(mmDlsym(handle_, kFuncNameDeleteSignResult));
  GE_CHK_BOOL_RET_STATUS(delete_proc != nullptr, FAILED, "[Dlsym][So] failed, func name = %s, error_msg = %s",
                         kFuncNameDeleteSignResult, mmDlerror());
  auto sign_proc = reinterpret_cast<DataFlowAuthSign>(mmDlsym(handle_, kFuncNameDataFlowAuthSign));
  GE_CHK_BOOL_RET_STATUS(sign_proc != nullptr, FAILED, "[Dlsym][So] failed, func name = %s, error_msg = %s",
                         kFuncNameDataFlowAuthSign, mmDlerror());
  auto get_sign_data_proc = reinterpret_cast<GetSignData>(mmDlsym(handle_, kFuncNameGetSignData));
  GE_CHK_BOOL_RET_STATUS(get_sign_data_proc != nullptr, FAILED, "[Dlsym][So] failed, func name = %s, error_msg = %s",
                         kFuncNameGetSignData, mmDlerror());
  auto get_sign_len_proc = reinterpret_cast<GetSignLength>(mmDlsym(handle_, kFuncNameGetSignLength));
  GE_CHK_BOOL_RET_STATUS(get_sign_data_proc != nullptr, FAILED, "[Dlsym][So] failed, func name = %s, error_msg = %s",
                         kFuncNameGetSignLength, mmDlerror());
  auto result = new_proc();
  GE_CHECK_NOTNULL(result);
  GE_MAKE_GUARD(result, ([result, delete_proc]() {
    delete_proc(result);
  }));
  auto ret = sign_proc(reinterpret_cast<const unsigned char *>(data.c_str()), data.length(), result);
  GE_CHK_STATUS_RET(ret, "Failed to auth sign, ret = %d.", ret);
  auto sign = get_sign_data_proc(result);
  GE_CHECK_NOTNULL(sign);
  auto sign_len = get_sign_len_proc(result);
  sign_data = std::string(reinterpret_cast<const char *>(sign), sign_len);
  GEEVENT("Deployer auth sign successfully, data = %s.", data.c_str());
  return SUCCESS;
}

Status DeployerAuthentication::AuthVerify(const std::string &data, const std::string &sign_data) const {
  GEEVENT("Deployer auth verify begin.");
  auto proc = reinterpret_cast<DataFlowAuthVerify>(mmDlsym(handle_, kFuncNameDataFlowAuthVerify));
  GE_CHK_BOOL_RET_STATUS(proc != nullptr, FAILED, "[Dlsym][So] failed, func name = %s, error_msg = %s",
                         kFuncNameDataFlowAuthVerify, mmDlerror());
  auto ret = proc(reinterpret_cast<const unsigned char *>(data.c_str()), data.length(),
                  reinterpret_cast<const unsigned char *>(sign_data.c_str()), sign_data.length());
  if (ret != 0) {
    GEEVENT("Failed to auth verify, ret = %d, data = %s.", ret, data.c_str());
    return FAILED;
  }
  GEEVENT("Deployer auth verify successfully.");
  return SUCCESS;
}
}  // namespace ge
