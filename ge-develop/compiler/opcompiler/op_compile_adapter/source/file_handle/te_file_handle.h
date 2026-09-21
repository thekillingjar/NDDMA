/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_FILE_HANDLE_TE_FILE_HANDLER_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_FILE_HANDLE_TE_FILE_HANDLER_H_

#include <vector>
#include <string>
#include <cstdint>

namespace te {
namespace fusion {
bool CheckFileEmpty(const std::string &realPath);

bool IsPidActive(uint32_t pid, std::vector<uint32_t> &pids);

void RemovePidTimeIdAndFile(std::vector<std::pair<uint32_t, std::string>> &pidTimeId);

void UpdatePidTimeIdToJson(const std::string &jsonFileDir, const std::string &realPath,
    std::vector<std::pair<uint32_t, std::string>> &pidTimeId);

std::string GetKernelMetaDir();

void GetCurrProcesses(std::vector<uint32_t> &pids);

bool IsPid(const char *dname);

void PushPidTimeId(std::vector<std::pair<uint32_t, std::string>> &pidTimeId,
    uint32_t pid, const std::string &kernelParentDir);

bool ReadPidTimeIdFromJson(const std::string &realPath,
    std::vector<std::pair<uint32_t, std::string>> &pidTimeId);

void MgrPidTimeIdInfo(std::vector<std::pair<uint32_t, std::string>> &delPidTimeId,
    std::vector<std::pair<uint32_t, std::string>> &curPidTimeId,
    std::vector<std::pair<uint32_t, std::string>> &mgrPidTimeId);

void WritePidTimeIdToJson(const std::string &realPath, uint32_t pid, const std::string &kernelParentDir);

void RecordPidTimeIdInfo();

void RemovePidTimeIdInfo();

void RemoveBuildLockFile(const std::string &kernelMetaParentDir);

void RemoveKernelMetaDir(const std::string &kernelMetaParentDir);

void RemoveKernelTempDir(const std::string &kernelMetaTempPath);

void RemoveParentKernelMetaDir(const std::string &parentKernelMetaPath);

void RemoveAllWorkFiles(bool keepProcessKernelMeta = true);

void RemoveProcessKernelMetaDir(const std::string &kernelMetaParentDir);

void CreateKernelMetaTempDir();

std::string GetNpuCollectPath();

void ClearKernelMetaProc(const std::string &kernelMetaPath);
}  // namespace fusion
}  // namespace te
#endif  // ATC_OPCOMPILER_TE_FUSION_SOURCE_FILE_HANDLE_TE_FILE_HANDLER_H_
