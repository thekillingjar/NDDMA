/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"
#include <fstream>

bool LogSwitch::logSwitch = false;

void LogCharBuffer(const char* input, size_t len)
{
    if (!LogSwitch::logSwitch) {
        return;
    }
    std::ostringstream os;
    for (size_t i = 0; i < len; i++) {
        os << static_cast<unsigned char>(input[i]) << " ";
    }
    Log(os.str());
}

void LogHexBuffer(const char* input, size_t len)
{
    if (!LogSwitch::logSwitch) {
        return;
    }
    std::ostringstream os;
    for (size_t i = 0; i < len; i++) {
        os << std::hex << static_cast<unsigned char>(input[i]) << " ";
    }
    Log(os.str());
}

const std::string GetProcessName(const pid_t& pid)
{
    std::string procPath = "/proc/" + std::to_string(pid) + "/comm";
    std::ifstream comm(procPath);
    std::string processName;

    if (!comm.is_open()) {
        return processName;
    }
    comm >> processName;
    comm.close();
    return processName;
}
