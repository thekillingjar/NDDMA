/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __INC_LLT_RUNTIME_STUB_H
#define __INC_LLT_RUNTIME_STUB_H

#include <vector>
#include <memory>
#include <mutex>
#include "mmpa/mmpa_api.h"
#include "runtime/rt.h"
#include "rt_error_codes.h"
namespace ge {
class RuntimeStub {
public:
 virtual ~RuntimeStub() = default;

 static RuntimeStub* GetInstance();

 static void SetInstance(const std::shared_ptr<RuntimeStub> &instance) {
   instance_ = instance;
 }

 static void Install(RuntimeStub*);
 static void UnInstall(RuntimeStub*);

 static void Reset() {
   instance_.reset();
 }

 virtual rtError_t rtGetSocVersion(char *version, const uint32_t maxLen);
 virtual rtError_t rtGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen);

private:
 static std::mutex mutex_;
 static std::shared_ptr<RuntimeStub> instance_;
 static thread_local RuntimeStub *fake_instance_;
};

class RuntimeStubV2Common : public RuntimeStub {
 public:
  rtError_t rtGetSocVersion(char *version, const uint32_t maxLen) override {
    (void)strcpy_s(version, maxLen, "Ascend910_9591");
    return RT_ERROR_NONE;
  }

  rtError_t rtGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen) override {
    (void)label;
    (void)key;
    (void)strcpy_s(val, maxLen, "3510");
    return RT_ERROR_NONE;
  }
};
}  // namespace ge
#endif // __INC_LLT_RUNTIME_STUB_H
