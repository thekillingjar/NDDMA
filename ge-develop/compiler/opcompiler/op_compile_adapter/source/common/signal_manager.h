/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_SIGNAL_MANAGER_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_SIGNAL_MANAGER_H_

#include <signal.h>
#include <array>
#include <mutex>
#include <set>
#include <string>

namespace te {
namespace fusion {
enum class SignoIdx {
    SIGINT_IDX = 0,
    SIGQUIT_IDX,
    SIGTERM_IDX,
    SIGIDX_BOTTOM
};

struct TESigAction {
    bool teRegistered = false;
    struct sigaction oldAct;
};

void ClearTEResource(int signo);
class SignalManager {
public:
    SignalManager() {};
    ~SignalManager() = default;
    static SignalManager &Instance();
    void Initialize();
    const struct sigaction& GetOldHandleByIdx(size_t idx);
    size_t GetOldHandlesSize() const;
    void SaveKernelTempDir(const std::string &kernelTempDir);
    void RemoveKernelTempDirs() const;
    static void RmKernelTempDir(const char *kernelMetaTempPath);
    bool HasTERegistered(size_t idx) const;
    void UnRegTEHandle(size_t idx);
    void Finalize();

private:
    void RegisterSignalHandle();

    bool init_flag_ = false;
    std::array<TESigAction, static_cast<size_t>(SignoIdx::SIGIDX_BOTTOM)> globalActions_;
    std::set<std::string> kernelTempDirSet_;
    mutable std::mutex kernelMetaMutex_;
};
}
}
#endif