/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LOG_H
#define LOG_H
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <unistd.h>
#include <string>

class LogSwitch {
public:
    static bool logSwitch;
};

// strftime format
#define LOGGER_PRETTY_TIME_FORMAT "%Y-%m-%d-%H:%M:%S"
constexpr uint32_t DECIMAL_WIDTH = 3;
constexpr uint32_t SECOND_UNIT = 1000;

// format it in two parts: main part with date and time and part with milliseconds
static std::string PrettyTime()
{
    auto tp = std::chrono::system_clock::now();
    std::time_t currentTime = std::chrono::system_clock::to_time_t(tp);

    // this function use static global pointer. so it is not thread safe solution
    std::tm* timeInfo = std::localtime(&currentTime);

    if (timeInfo == NULL) {
        return "";
    }
    char buffer[128];

    (void)strftime(buffer, sizeof(buffer), LOGGER_PRETTY_TIME_FORMAT, timeInfo);

    std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch());
    std::chrono::microseconds cs = std::chrono::duration_cast<std::chrono::microseconds>(tp.time_since_epoch());
    std::ostringstream ss;
    ss.fill('0');
    ss << buffer << "." << std::setw(DECIMAL_WIDTH) << (ms.count() % SECOND_UNIT) <<
        "." << std::setw(DECIMAL_WIDTH) << (cs.count() % SECOND_UNIT);

    return ss.str();
}

#define Log(content)                           \
do {                                           \
    if (LogSwitch::logSwitch == true) {        \
        pid_t pid = getpid();                  \
        const std::string processName = GetProcessName(pid); \
        std::cout << "[INFO] CMP(" << pid <<   \
        "," << processName << "):" <<          \
        PrettyTime() << " [" << __FILE__ <<    \
        ":"<< __LINE__ << "]" << __func__ <<   \
        content << std::endl;                  \
    }                                          \
} while (0)                                    \

#define LogFatal(content)                      \
do {                                           \
    pid_t pid = getpid();                      \
    const std::string processName = GetProcessName(pid); \
    std::cout << "[WARNING] CMP(" << pid       \
    << "," << processName << "):" <<           \
    PrettyTime() << " [" << __FILE__ <<        \
    ":"<< __LINE__ << "]" << __func__ <<       \
    content << std::endl;                      \
} while (0)                                    \

void LogCharBuffer(const char* input, size_t len);
void LogHexBuffer(const char* input, size_t len);
const std::string GetProcessName(const pid_t& pid);

#endif  // LOG_H
