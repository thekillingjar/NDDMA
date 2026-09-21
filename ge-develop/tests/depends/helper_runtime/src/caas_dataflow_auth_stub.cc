/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "caas_dataflow_auth_stub.h"

struct SignResult {
  std::string sign_result;
};

static SignResult g_sign_result;

SignResult *NewSignResult() {
  return &g_sign_result;
}

void DeleteSignResult(SignResult *sign_result) {}

uint32_t GetSignLength(SignResult *sign_result) {
  return g_sign_result.sign_result.length();
}

const unsigned char *GetSignData(SignResult *sign_result) {
  return reinterpret_cast<const unsigned char *>(sign_result->sign_result.c_str());
}

int32_t DataFlowAuthMasterInit() {
  return 0;
}

int32_t DataFlowAuthSign(const unsigned char *data, uint32_t data_len, SignResult *sign_result) {
  if (data != nullptr) {
    g_sign_result.sign_result = std::string(reinterpret_cast<const char *>(data), data_len);
  }
  return 0;
}

int32_t DataFlowAuthVerify(const unsigned char *data, uint32_t data_len,
                           const unsigned char *sign_data, uint32_t sign_data_len) {
  return 0;
}
