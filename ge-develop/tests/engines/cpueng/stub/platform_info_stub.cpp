/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "platform_info_stub.h"
#include "framework/common/debug/ge_log.h"
#include "graph/def_types.h"

namespace fe {

PlatformInfoManager &PlatformInfoManager::Instance() {
    static PlatformInfoManager platformInfo;
    return platformInfo;
}

PlatformInfoManager::PlatformInfoManager() {}

PlatformInfoManager::~PlatformInfoManager() {}

uint32_t PlatformInfoManager::InitializePlatformInfo() {
    return 0;
}

uint32_t PlatformInfoManager::Finalize() {
    return 0;
}

thread_local std::string socVersion;

uint32_t PlatformInfoManager::GetPlatformInfos(const string SoCVersion,
                                              PlatFormInfos &platformInfo,
                                              OptionalInfos &optiCompilationInfo) {
    socVersion = SoCVersion;
    return 0;
}

bool PlatFormInfos::GetPlatformRes(const std::string &label, const std::string &key, std::string &val)
{
    if (label == "CPUCache" && key == "AICPUSyncBySW") {
        if (socVersion.find("Hi") != string::npos || socVersion.find("SD") != string::npos) {
            val = "1";
        } else {
            val = "0";
        }
        return true;
    }
    return false;
}

}

