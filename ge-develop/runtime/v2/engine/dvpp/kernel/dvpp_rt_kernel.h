/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DVPP_RT_KERNEL_H
#define DVPP_RT_KERNEL_H

#ifdef DVPP_LLT
#include "stub_fast_node.h"
#include "stub_dvpp_context.h"
#include "stub_ge_error_codes.h"
#else
#include "dvpp_context.h"
#include "graph/ge_error_codes.h"
#endif

struct DvppSqeInfo {
    uint8_t* sqePtr{nullptr};   // 入参, SQE指针，由GenerateDvppSqe申请，最大申请640字节，即最大10个SQE
    uint32_t sqeSize{0};  // 入参，sqe_ptr的size
    uint32_t sqeCnt{0};   // 出参，sqe_cnt为生成的sqe个数
    void* argsPtr{nullptr};
    uint32_t argSize{0};
};

namespace gert {
namespace kernel {
ge::graphStatus CreateGenerateDvppSqeOutputs(const ge::FastNode* node, KernelContext* context);
ge::graphStatus GenerateSqeAndLaunchTask(KernelContext* context);
ge::graphStatus CreateCalcWorkspaceSizeOutputs(const ge::FastNode* node, KernelContext* context);
ge::graphStatus CalcOpWorkSpaceSize(KernelContext *context);

int32_t BuildDvppCmdlistV2(gert::DvppContext* context, std::vector<void*>& ioAddrs, DvppSqeInfo& sqeInfo);
int32_t CalcOpWorkspaceMemSize(gert::DvppContext* context, size_t& memSize);
int32_t StarsBatchTaskLaunch(void* taskSqe, const uint32_t sqeCount, void* stream);
int32_t StarsMultipleTaskLaunch(void* taskSqe, const uint32_t sqeCount, void* stream);
void FreeSqeReadOnlyMem(void* sqePtr);

} // namespace kernel
} // namespace gert
#endif // DVPP_RT_KERNEL_H