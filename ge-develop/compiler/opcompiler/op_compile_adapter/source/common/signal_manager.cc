/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "signal_manager.h"
#include <map>
#include <vector>
#include <string>
#include "common/te_config_info.h"
#include "file_handle/te_file_handle.h"
#include "common/common_utils.h"
#include "inc/te_fusion_log.h"

namespace te {
namespace fusion {

namespace {
    const std::vector<int> EXCEPT_SIGNUMS = {SIGINT, SIGQUIT, SIGTERM};
    const std::map<int, SignoIdx> SIGNUM_IDX = {
        {SIGINT, SignoIdx::SIGINT_IDX}, {SIGQUIT, SignoIdx::SIGQUIT_IDX}, {SIGTERM, SignoIdx::SIGTERM_IDX}
    };
    const std::string ASCEND_CORE_DUMP_SIG_NONE = "none";
    const size_t CMD_INI_LENGTH = 9;
    const size_t PATH_MAX = 4096;
}

SignalManager& SignalManager::Instance()
{
    static SignalManager sigManager;
    return sigManager;
}


const struct sigaction& SignalManager::GetOldHandleByIdx(size_t idx)
{
    return globalActions_[idx].oldAct;
}

void SignalManager::Initialize()
{
    if (init_flag_) {
        return;
    }
    if (TeConfigInfo::Instance().GetEnvAscendCoreDumpSignal() == ASCEND_CORE_DUMP_SIG_NONE) {
        TE_DBGLOG("No need to register signal handle.");
        return;
    }
    TESigAction actDft;
    sigemptyset(&actDft.oldAct.sa_mask);
    actDft.oldAct.sa_flags = 0;
    actDft.oldAct.sa_handler = nullptr;

    globalActions_.fill(actDft);
    RegisterSignalHandle();
    SaveKernelTempDir(TeConfigInfo::Instance().GetKernelMetaTempDir());
    init_flag_ = true;
}

size_t SignalManager::GetOldHandlesSize() const
{
    return globalActions_.size();
}

bool SignalManager::HasTERegistered(size_t idx) const
{
    return globalActions_[idx].teRegistered;
}

void SignalManager::UnRegTEHandle(size_t idx)
{
    globalActions_[idx].teRegistered = false;
}

void ClearTEResource(int signo)
{
    TE_INFOLOG("Process exception happened, ready to callback for signo[%d].", signo);
    auto iter = SIGNUM_IDX.find(signo);
    if (iter == SIGNUM_IDX.end()) {
        return;
    }
    size_t signoIdx = static_cast<size_t>(iter->second);
    if (signoIdx < SignalManager::Instance().GetOldHandlesSize() &&
        SignalManager::Instance().HasTERegistered(signoIdx)) {
        if (sigaction(signo, &SignalManager::Instance().GetOldHandleByIdx(signoIdx), nullptr) < 0) {
            TE_INFOLOG("Signo[%d] has not register signal handle.", signo);
            return;
        }
        SignalManager::Instance().UnRegTEHandle(signoIdx);
        (void)raise(signo);
    }
    SignalManager::Instance().RemoveKernelTempDirs();
    TE_INFOLOG("Finish call signal handle func for signo[%d].", signo);
}

void SignalManager::RegisterSignalHandle()
{
    struct sigaction teAct;
    teAct.sa_handler = ClearTEResource;
    sigemptyset(&teAct.sa_mask);
    teAct.sa_flags = 0;

    for (size_t i = 0; i < EXCEPT_SIGNUMS.size(); ++i) {
        if (sigaction(EXCEPT_SIGNUMS[i], &teAct, &globalActions_[i].oldAct) < 0) {
            TE_DBGLOG("Register signal handle for signo[%d] unsuccessful.", EXCEPT_SIGNUMS[i]);
            continue;
        }
        globalActions_[i].teRegistered = true;
        TE_DBGLOG("Finish to register single handle func for signo[%d].", EXCEPT_SIGNUMS[i]);
    }
}

void SignalManager::RmKernelTempDir(const char *kernelMetaTempPath)
{
    char cmd[CMD_INI_LENGTH + PATH_MAX] = "rm -rf ";
    if (sprintf_s(cmd, sizeof(cmd), "%s %s", cmd, kernelMetaTempPath) < 0) {
        return;
    }
    int res = system(cmd);
    if (res != 0) {
        TE_DBGLOG("Remove kernel_meta_temp Path unsuccessfully.");
    } else {
        TE_DBGLOG("Remove kernel_meta_temp Path successfully.");
    }
}

void SignalManager::RemoveKernelTempDirs() const
{
    std::lock_guard<std::mutex> lock_guard(kernelMetaMutex_);
    for (const auto &kernelTempDir : kernelTempDirSet_) {
        TE_DBGLOG("Ready to remove kernelTempDir.");
        RmKernelTempDir(kernelTempDir.data());
    }
}

void SignalManager::SaveKernelTempDir(const std::string &kernelTempDir)
{
    std::string kernelTempRealPath = RealPath(kernelTempDir);
    std::lock_guard<std::mutex> lock_guard(kernelMetaMutex_);
    if (!kernelTempRealPath.empty() && kernelTempDirSet_.count(kernelTempRealPath) == 0) {
        TE_DBGLOG("Set kernelTempDir[%s].", kernelTempRealPath.c_str());
        kernelTempDirSet_.emplace(kernelTempRealPath);
    }
}

void SignalManager::Finalize()
{
    for (size_t i = 0; i < EXCEPT_SIGNUMS.size(); ++i) {
        if (!globalActions_[i].teRegistered) {
            continue;
        }
        if (sigaction(EXCEPT_SIGNUMS[i], &globalActions_[i].oldAct, nullptr) < 0) {
            TE_WARNLOG("Unregister signal handle for signo[%d] unsuccessful.", EXCEPT_SIGNUMS[i]);
            continue;
        }
        globalActions_[i].teRegistered = false;
        TE_DBGLOG("Finish to unregister single handle func for signo[%d].", EXCEPT_SIGNUMS[i]);
    }
    init_flag_ = false;
    std::lock_guard<std::mutex> lock_guard(kernelMetaMutex_);
    kernelTempDirSet_.clear();
}
}
}
