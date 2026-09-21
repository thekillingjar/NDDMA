/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_COMPILE_RESULT_UTILS_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_COMPILE_RESULT_UTILS_H_

#include <string>
#include "inc/te_fusion_types.h"

namespace te {
namespace fusion {
class CompileResultUtils {
public:
    static CompileResultPtr ParseCompileResult(const std::string &jsonFilePath);
    static bool GetCompileResultKernelName(const CompileResultPtr &compileRetPtr, std::string &kernelName);
private:
    static bool SetBinFilePath(const CompileResultPtr &compileRetPtr);
    static bool SetKernelBin(const CompileResultPtr &compileRetPtr);
    static bool SetKernelBinBySo(const CompileResultPtr &compileRetPtr);
    static void SetHeaderFilePath(const CompileResultPtr &compileRetPtr);
};
}
}
#endif  // ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_COMPILE_RESULT_UTILS_H_