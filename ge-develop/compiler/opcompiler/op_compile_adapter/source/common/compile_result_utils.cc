/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/compile_result_utils.h"
#include "inc/te_fusion_log.h"
#include "inc/te_fusion_check.h"
#include "common/common_utils.h"
#include "common/te_file_utils.h"
#include "mmpa/mmpa_api.h"
#include "common/te_config_info.h"

namespace te {
namespace fusion {
namespace {
bool IsEndWith(const std::string &str, const std::string &suffix)
{
    if (str.empty() || suffix.empty()) {
        return false;
    }
    if (str.size() < suffix.size()) {
        return false;
    }
    return str.substr(str.size() - suffix.size()) == suffix;
}
void GetFileName(const std::string &wholeFileName, std::string &fileName)
{
    if (wholeFileName.empty()) {
        return;
    }
    size_t pre_pos = wholeFileName.rfind('/');
    if (pre_pos == std::string::npos) {
        fileName = wholeFileName;
    } else {
        fileName = wholeFileName.substr(pre_pos + 1);
    }
    size_t post_pos = fileName.find('.');
    if (post_pos != std::string::npos) {
        fileName = fileName.substr(0, post_pos);
    }
}
}
CompileResultPtr CompileResultUtils::ParseCompileResult(const std::string &jsonFilePath)
{
    TE_DBGLOG("Parse compile result from json file[%s].", jsonFilePath.c_str());
    if (jsonFilePath.empty()) {
        return nullptr;
    }
    CompileResultPtr compileRetPtr = nullptr;
    TE_FUSION_MAKE_SHARED(compileRetPtr = std::make_shared<CompileResult>(), return nullptr);
    compileRetPtr->jsonPath = RealPath(jsonFilePath);
    // read json info from json file
    TE_DBGLOG("Read json info from json file[%s].", compileRetPtr->jsonPath.c_str());
    nlohmann::json jsonInfo;
    if (!TeFileUtils::GetJsonValueFromJsonFile(compileRetPtr->jsonPath, jsonInfo)) {
        TE_INFOLOG("Can not Get jsonValue from jsonFile[%s].", jsonFilePath.c_str());
        return nullptr;
    }
    TE_FUSION_MAKE_SHARED(compileRetPtr->jsonInfo = std::make_shared<nlohmann::json>(jsonInfo), return nullptr);

    // set bin file path
    if (!SetBinFilePath(compileRetPtr)) {
        TE_INFOLOG("Can not set bin file name from json file[%s].", compileRetPtr->jsonPath.c_str());
        return nullptr;
    }

    // if bin file is .o file, read bin file and set kernel bin
    if (IsEndWith(compileRetPtr->binPath, ".o")) {
        if (!SetKernelBin(compileRetPtr)) {
            TE_INFOLOG("Can not read bin file for json file[%s].", compileRetPtr->jsonPath.c_str());
            return nullptr;
        }
    } else if (IsEndWith(compileRetPtr->binPath, ".so")) {
        if (!SetKernelBinBySo(compileRetPtr)) {
            TE_INFOLOG("Can not read bin file for json file[%s].", compileRetPtr->jsonPath.c_str());
            return nullptr;
        }
    }

    // set header file path
    SetHeaderFilePath(compileRetPtr);
    TE_DBGLOG("Finish parsing compile result from json file[%s].", jsonFilePath.c_str());
    return compileRetPtr;
}

bool CompileResultUtils::GetCompileResultKernelName(const CompileResultPtr &compileRetPtr, std::string &kernelName)
{
    if (compileRetPtr == nullptr) {
        TE_INFOLOG("Compile result is null.");
        return false;
    }
    GetFileName(compileRetPtr->jsonPath, kernelName);
    if (kernelName.empty()) {
        TE_INFOLOG("Kernel name of json file[%s] is empty.", compileRetPtr->jsonPath.c_str());
        return false;
    }
    std::string binKernelName;
    GetFileName(compileRetPtr->binPath, binKernelName);
    if (binKernelName.empty()) {
        TE_INFOLOG("Kernel name of bin file[%s] is empty.", compileRetPtr->binPath.c_str());
        return false;
    }
    if (compileRetPtr->jsonInfo != nullptr) {
        if (!compileRetPtr->jsonInfo->contains("binFileName")) {
            TE_INFOLOG("Json file[%s] does not contain binFileName.", compileRetPtr->jsonPath.c_str());
            return false;
        }
        std::string binFileName = compileRetPtr->jsonInfo->at("binFileName");
        if (binFileName != binKernelName) {
            TE_INFOLOG("Bin file name[%s] of json and bin file name[%s] is not same.",
                       binFileName.c_str(), binKernelName.c_str());
            return false;
        }
    }
    return true;
}

bool CompileResultUtils::SetBinFilePath(const CompileResultPtr &compileRetPtr)
{
    if (compileRetPtr == nullptr || compileRetPtr->jsonInfo == nullptr) {
        return false;
    }
    if (!compileRetPtr->jsonInfo->contains("binFileName") || !compileRetPtr->jsonInfo->contains("binFileSuffix")) {
        TE_INFOLOG("binFileName, binFileSuffix or sha256 is not found in json file.");
        return false;
    }
    std::string binFileName = compileRetPtr->jsonInfo->at("binFileName");
    std::string binFileSuffix = compileRetPtr->jsonInfo->at("binFileSuffix");
    TE_DBGLOG("Get bin file name[%s] and bin file suffix[%s] from json.", binFileName.c_str(), binFileSuffix.c_str());
    // check is bin file existed
    std::string binFileDirPath;
    size_t pos = compileRetPtr->jsonPath.find_last_of("\\/");
    if (pos != std::string::npos) {
        binFileDirPath = compileRetPtr->jsonPath.substr(0, pos);
    } else {
        binFileDirPath = ".";
    }
    TE_DBGLOG("Bin file dir path is [%s].", binFileDirPath.c_str());
    compileRetPtr->binPath = RealPath(binFileDirPath + "/" + binFileName + binFileSuffix);
    TE_DBGLOG("Bin file path is [%s].", compileRetPtr->binPath.c_str());
    return !compileRetPtr->binPath.empty();
}

bool CompileResultUtils::SetKernelBin(const CompileResultPtr &compileRetPtr)
{
    if (compileRetPtr->binPath.empty()) {
        return false;
    }
    if (!compileRetPtr->jsonInfo->contains("kernelName")) {
        TE_INFOLOG("KernelName is not found in json file[%s].", compileRetPtr->jsonPath.c_str());
        return false;
    }
    std::string kernelName = compileRetPtr->jsonInfo->at("kernelName").get<std::string>();
    TE_DBGLOG("Kernel name for kernel bin is [%s].", kernelName.c_str());

    std::vector<char> buffer;
    // read bin file
    if (!TeFileUtils::GetBufferFromBinFile(compileRetPtr->binPath, buffer)) {
        TE_INFOLOG("Read buffer from bin file[%s] not successfully, need to compile.", compileRetPtr->binPath.c_str());
        return false;
    }

    // make kernel bin
    TE_FUSION_MAKE_SHARED(compileRetPtr->kernelBin = std::make_shared<ge::OpKernelBin>(kernelName, std::move(buffer)),
                          return false);
    return true;
}

bool CompileResultUtils::SetKernelBinBySo(const CompileResultPtr &compileRetPtr)
{
    std::vector<char> buffer;
    // read bin file
    const std::string& libPath = compileRetPtr->binPath;
    TE_INFOLOG("To invoke dlopen on lib: %s", libPath.c_str());
    constexpr uint32_t openFlag = static_cast<uint32_t>(MMPA_RTLD_NOW) | static_cast<uint32_t>(RTLD_LOCAL);
    auto handle = mmDlopen(libPath.c_str(), static_cast<int32_t>(openFlag));
    if (handle == nullptr) {
        const char_t *error = mmDlerror();
        TE_ERRLOG("[Invoke][DlOpen] failed. path = %s, error = %s", libPath.c_str(), error != nullptr ? error : "");
        return false;
    }
    // void GetKernelBin(std::vector<int8_t> &kernel)
    const auto get_kernel_bin = reinterpret_cast<void (*)(std::vector<char> &)>(mmDlsym(handle, "GetKernelBin"));
    if (get_kernel_bin != nullptr) {
        TE_INFOLOG("Invoke function get_kernel_bin in lib: %s", libPath.c_str());
        get_kernel_bin(buffer);
    }
    if (mmDlclose(handle) != 0) {
        const char_t *error = mmDlerror();
        TE_ERRLOG("[Close][Handle]Failed, handle of %s: %s", libPath.c_str(), error != nullptr ? error : "");
    }
    TE_FUSION_CHECK((buffer.empty()), {
        TE_ERRLOG("Failed to get kernelbin %s.", libPath.c_str());
        return false;
    });

    // kernel name
    if (!compileRetPtr->jsonInfo->contains("kernelName")) {
        TE_INFOLOG("KernelName is not found in json file[%s].", compileRetPtr->jsonPath.c_str());
        return false;
    }
    std::string kernelName = compileRetPtr->jsonInfo->at("kernelName").get<std::string>();
    TE_DBGLOG("Kernel name for kernel bin is [%s].", kernelName.c_str());
    // make kernel bin
    TE_FUSION_MAKE_SHARED(compileRetPtr->kernelBin = std::make_shared<ge::OpKernelBin>(kernelName, std::move(buffer)),
                          return false);
    return true;
}

void CompileResultUtils::SetHeaderFilePath(const CompileResultPtr &compileRetPtr)
{
    if (compileRetPtr == nullptr || compileRetPtr->jsonInfo == nullptr) {
        return;
    }
    if (!compileRetPtr->jsonInfo->contains("headerFileSuffix")) {
        return;
    }
    // set head file path if existed
    size_t pos = compileRetPtr->binPath.find_last_of(".");
    if (pos == std::string::npos) {
        return;
    }
    std::string headerFilePath =
        compileRetPtr->binPath.substr(0, pos) + compileRetPtr->jsonInfo->at("headerFileSuffix").get<std::string>();
    compileRetPtr->headerPath = RealPath(headerFilePath);
    TE_DBGLOG("Set header file path [%s].", compileRetPtr->headerPath.c_str());
}
}
}
