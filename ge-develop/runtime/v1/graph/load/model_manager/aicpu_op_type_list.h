/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_OP_TYPE_LIST_H_
#define AICPU_OP_TYPE_LIST_H_

extern "C" {
enum OpKernelType {
    TF_KERNEL,
    CPU_KERNEL
};

enum ReturnCode {
    OP_TYPE_NOT_SUPPORT,
    FORMAT_NOT_SUPPORT,
    DTYPE_NOT_SUPPORT
};

#pragma pack(push, 1)
// One byte alignment
struct SysOpInfo {
    uint64_t opLen;
    uint64_t opType;
    OpKernelType kernelsType;
};

struct SysOpCheckInfo {
    uint64_t opListNum;
    uint64_t offSetLen;
    uint64_t sysOpInfoList;
    uint64_t opParamInfoList;
};

struct SysOpCheckResp {
    uint64_t opListNum;
    bool isWithoutJson;
    uint64_t returnCodeList;
    uint64_t sysOpInfoList;
    uint64_t opParamInfoList;
};
#pragma pack(pop)
}

#endif  // AICPU_OP_TYPE_LIST_H_
