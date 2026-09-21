/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dlfcn.h>
#include <mutex>
#include "securec.h"
#include "acl/acl_rt.h"
#ifdef DVPP_LLT
#include "stub_op_impl_registry.h"
#include "stub_kernel_registry.h"
#include "stub_ge_log.h"
#else
#include "register/kernel_registry.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "debug/ge_log.h"
#endif
#include "dvpp_rt_kernel.h"

namespace gert {
namespace kernel {

using FreeSqeFunPtr = void(*)(void*);   
constexpr uint32_t DVPP_SQE_LEN = 64;
constexpr uint32_t DVPP_SQE_MAX_CNT = 10;
constexpr uint32_t ARGS_SIZE = 20480; // 20*1024 考虑拆分模式（拆9块）需要较大内存
constexpr uint32_t ARGS_MAX_CNT = 10;
constexpr uint32_t SQE_FIELD_LEN = 160;
constexpr size_t FIRST_OUTPUT = 0;
constexpr size_t MAX_WORKSPACE_COUNT = 2UL; // max 2 workspace

static std::once_flag loadFlag;
static void* dvppHandle = nullptr;

static int32_t (*dvppBuildDvppCmdlistV2FunPtr)(gert::DvppContext*, std::vector<void*>&, DvppSqeInfo&) = nullptr;
static int32_t (*dvppCalcOpWorkspaceMemSizeFunPtr)(gert::DvppContext*, size_t&) = nullptr;
static int32_t (*dvppStarsBatchTaskLaunchFunPtr)(void *, const uint32_t, void *) = nullptr;
static int32_t (*dvppStarsMultipleTaskLaunchFunPtr)(void *, const uint32_t, void *) = nullptr;
static void (*dvppFreeSqeReadOnlyMemFunPtr)(void *) = nullptr;

template <class T>
static inline void DvppRtKernelLoadFunc(void * const handle, T& funPtr, const std::string& funName)
{
    funPtr = reinterpret_cast<T>(dlsym(handle, funName.c_str()));
    if (funPtr == nullptr) {
        GELOGE(ge::FAILED, "rtKernel function %s not load.", funName.c_str());
    }

    return;
}

static void LoadDvppRtKernelFunc()
{
    dvppHandle = dlopen("libdvpp_rtkernel.so", RTLD_NOW);
    if (dvppHandle == nullptr) {
        const auto errMsg = dlerror();
        GELOGE(ge::FAILED, "dlopen libdvpp_rtkernel.so fail [%s].", (errMsg != nullptr) ? errMsg : "unknown");
        return;
    }

    DvppRtKernelLoadFunc(dvppHandle, dvppBuildDvppCmdlistV2FunPtr, "BuildDvppCmdlistV2");
    DvppRtKernelLoadFunc(dvppHandle, dvppCalcOpWorkspaceMemSizeFunPtr, "CalcOpWorkspaceMemSize");
    DvppRtKernelLoadFunc(dvppHandle, dvppStarsBatchTaskLaunchFunPtr, "StarsBatchTaskLaunch");
    DvppRtKernelLoadFunc(dvppHandle, dvppStarsMultipleTaskLaunchFunPtr, "StarsMultipleTaskLaunch");
    DvppRtKernelLoadFunc(dvppHandle, dvppFreeSqeReadOnlyMemFunPtr, "FreeSqeReadOnlyMem");
}

int32_t BuildDvppCmdlistV2(gert::DvppContext* context, std::vector<void*>& ioAddrs, DvppSqeInfo& sqeInfo)
{
    std::call_once(loadFlag, LoadDvppRtKernelFunc);
    if (dvppBuildDvppCmdlistV2FunPtr == nullptr) {
        GELOGE(ge::FAILED, "dvppBuildDvppCmdlistV2FunPtr_ is null");
        return ge::FAILED;
    }

    return dvppBuildDvppCmdlistV2FunPtr(context, ioAddrs, sqeInfo);
}

int32_t CalcOpWorkspaceMemSize(gert::DvppContext* context, size_t& memSize)
{
    std::call_once(loadFlag, LoadDvppRtKernelFunc);
    if (dvppCalcOpWorkspaceMemSizeFunPtr == nullptr) {
        GELOGE(ge::FAILED, "dvppCalcOpWorkspaceMemSizeFunPtr is null");
        return ge::FAILED;
    }

    return dvppCalcOpWorkspaceMemSizeFunPtr(context, memSize);
}

int32_t StarsBatchTaskLaunch(void* taskSqe, const uint32_t sqeCount, void* stream)
{
    std::call_once(loadFlag, LoadDvppRtKernelFunc);
    if (dvppStarsBatchTaskLaunchFunPtr == nullptr) {
        GELOGE(ge::FAILED, "dvppStarsBatchTaskLaunchFunPtr is null");
        return ge::FAILED;
    }

    return dvppStarsBatchTaskLaunchFunPtr(taskSqe, sqeCount, stream);
}

int32_t StarsMultipleTaskLaunch(void* taskSqe, const uint32_t sqeCount, void* stream)
{
    std::call_once(loadFlag, LoadDvppRtKernelFunc);
    if (dvppStarsMultipleTaskLaunchFunPtr == nullptr) {
        GELOGE(ge::FAILED, "dvppStarsMultipleTaskLaunchFunPtr is null");
        return ge::FAILED;
    }

    return dvppStarsMultipleTaskLaunchFunPtr(taskSqe, sqeCount, stream);
}

void FreeSqeReadOnlyMem(void* sqePtr)
{
    std::call_once(loadFlag, LoadDvppRtKernelFunc);
    if (dvppFreeSqeReadOnlyMemFunPtr == nullptr) {
        GELOGE(ge::FAILED, "dvppFreeSqeReadOnlyMemFunPtr is null");
        return;
    }

    dvppFreeSqeReadOnlyMemFunPtr(sqePtr);
}

struct Sqe {
    uint32_t field[SQE_FIELD_LEN];
    void* args{nullptr};
    uint32_t argsSize{0};
};

void FreeSqeBuf(void *ptr)
{
    if (ptr == nullptr) {
        return;
    }
    Sqe* dvppSqe = static_cast<Sqe*>(ptr);
    if (dvppSqe->args != nullptr) {
        free(dvppSqe->args);
        dvppSqe->args = nullptr;
    }
    delete dvppSqe;
    ptr = nullptr;

    return;
}

ge::graphStatus CreateGenerateDvppSqeOutputs(const ge::FastNode* node, KernelContext* context)
{
    (void)node;
    if (context == nullptr) {
        GELOGE(ge::FAILED, "param context is null");
        return ge::FAILED;
    }
    
    auto dvppSqe = new (std::nothrow) Sqe;
    if (dvppSqe == nullptr) {
        GELOGE(ge::FAILED, "dvppSqe is null");
        return ge::FAILED;
    }
    
    dvppSqe->argsSize = ARGS_SIZE * ARGS_MAX_CNT;
    dvppSqe->args = malloc(static_cast<size_t>(dvppSqe->argsSize));
    if (dvppSqe->args == nullptr) {
        GELOGE(ge::FAILED, "malloc dvppSqe args failed.");
        delete dvppSqe;
        return ge::FAILED;
    }

    FreeSqeFunPtr freeSqeFunPtr = FreeSqeBuf;
    context->GetOutput(0)->Set(dvppSqe, freeSqeFunPtr); // 由ge负责释放
    return ge::SUCCESS;
}

static int32_t FillDvppSqeInfo(KernelContext* context, DvppSqeInfo &sqeInfo, Sqe** sqe)
{
    Sqe* dvppSqe = *sqe;
    auto output0 = context->GetOutputPointer<uint32_t*>(FIRST_OUTPUT); // sqe内存保存在FIRST_OUTPUT
    if (output0 == nullptr) {
        GELOGE(ge::FAILED, "output0 is null");
        return ge::FAILED;
    }

    dvppSqe = reinterpret_cast<Sqe*>(*output0);

    // 对SQE进行清零，方便后边只读内存释放
    (void)memset_s(dvppSqe, (sizeof(uint32_t) * SQE_FIELD_LEN), 0, (sizeof(uint32_t) * SQE_FIELD_LEN));

    sqeInfo.sqePtr = reinterpret_cast<uint8_t*>(dvppSqe);
    sqeInfo.sqeSize = DVPP_SQE_LEN * DVPP_SQE_MAX_CNT;
    sqeInfo.sqeCnt = 1U;
    sqeInfo.argsPtr = dvppSqe->args;
    sqeInfo.argSize = dvppSqe->argsSize;

    return ge::SUCCESS;
}

static int32_t StarTaskLaunch(KernelContext* context, DvppSqeInfo &sqeInfo, const std::string &opType)
{
    int32_t ret = 0;
    // stream 排布在kernel输入的最后一位
    const auto stream = context->GetInputValue<aclrtStream>(context->GetInputNum() - 1UL);
    
    if ((opType == "DecodeJpeg") || (opType == "DecodeAndCropJpeg")) {
        ret = StarsMultipleTaskLaunch(sqeInfo.sqePtr, sqeInfo.sqeCnt, stream);
    } else {
        ret = StarsBatchTaskLaunch(sqeInfo.sqePtr, sqeInfo.sqeCnt, stream);
    }

    if (ret != 0) {
        GELOGE(ge::FAILED, "stars batch task launch failed.");
        return ge::FAILED;
    }

    return ret;
}

ge::graphStatus GenerateSqeAndLaunchTask(KernelContext* context)
{
    GELOGD("enter dvpp runtime kernel.");
    if (context == nullptr) {
        GELOGE(ge::FAILED, "param context is null");
        return ge::FAILED;
    }
    const size_t ioNum = reinterpret_cast<DvppContext*>(context)->GetComputeNodeInputNum() +
                         reinterpret_cast<DvppContext*>(context)->GetComputeNodeOutputNum();

    const size_t ioAddrStart = ioNum;
    auto workSpace = context->GetInputPointer<ContinuousVector>(ioAddrStart + ioNum);
    if (workSpace == nullptr) {
        GELOGE(ge::FAILED, "workSpace is null");
        return ge::FAILED;
    }
    const size_t workSpaceNums = workSpace->GetSize();
    std::vector<void*> ioAddrs(ioNum + workSpaceNums);
    for (size_t i = 0UL; i < ioNum; i++) {
        const auto inputTensorData = context->GetInputValue<gert::TensorData*>(ioAddrStart + i);
        if (inputTensorData == nullptr) {
            GELOGE(ge::FAILED, "inputTensorData is null");
            return ge::FAILED;
        }
        ioAddrs[i] = inputTensorData->GetAddr();
    }
    const auto workSpaceVector = reinterpret_cast<gert::GertTensorData *const *>(workSpace->GetData());
    const auto dvppKernelContext = reinterpret_cast<DvppContext *>(context);
    const std::string opType = dvppKernelContext->GetNodeType();
    if (workSpaceNums != 0UL) {
        GELOGD("op_type : %s workspace_addr size: %zu", opType.c_str(), workSpaceNums);
        for (size_t j = 0UL; j < workSpaceNums; ++j) {
            ioAddrs[ioNum + j] = workSpaceVector[j]->GetAddr();
        }
    }

    Sqe* dvppSqe = nullptr;
    DvppSqeInfo sqeInfo{};
    int32_t ret = FillDvppSqeInfo(context, sqeInfo, &dvppSqe);
    if (ret != ge::SUCCESS) {
        return ge::FAILED;
    }

    ret = BuildDvppCmdlistV2(reinterpret_cast<DvppContext*>(context), ioAddrs, sqeInfo);
    if (ret != 0) {
        GELOGE(ge::FAILED, "build dvpp cmdlistV2 failed, ret=%d.", ret);
        FreeSqeReadOnlyMem(sqeInfo.sqePtr);
        return ge::FAILED;
    }

    ret = StarTaskLaunch(context, sqeInfo, opType);
    if (ret != 0) {
        return ge::FAILED;
    }

    return ge::SUCCESS;
}

ge::graphStatus CreateCalcWorkspaceSizeOutputs(const ge::FastNode* node, KernelContext* context)
{
    (void)node;
    auto workspaceSize = ContinuousVector::Create<size_t>(MAX_WORKSPACE_COUNT);
    if (workspaceSize == nullptr) {
        return ge::FAILED;
    }
    context->GetOutput(0)->SetWithDefaultDeleter<uint8_t[]>(workspaceSize.release());
    return ge::SUCCESS;
}

ge::graphStatus CalcOpWorkSpaceSize(KernelContext *context)
{
    const auto dvppKernelContext = reinterpret_cast<DvppContext *>(context);
    const std::string opType = dvppKernelContext->GetNodeType();
    auto workspaceSizes = context->GetOutputPointer<gert::ContinuousVector>(0);
    if (workspaceSizes == nullptr) {
        GELOGE(ge::FAILED, "workspaceSizes is null");
        return ge::FAILED;
    }
    GELOGD("CalcOpWorkSpaceSize op type %s", opType.c_str());

    size_t memSize = 0;
    const int32_t ret = CalcOpWorkspaceMemSize(reinterpret_cast<DvppContext*>(context), memSize);
    if (ret != 0) {
        GELOGE(ge::FAILED, "Calc dvpp workspace size failed, ret=%d.", ret);
        return ge::FAILED;
    }

    GELOGD("The needed workspace mem size is %zu", memSize);

    if (memSize == 0UL) {
        (void)workspaceSizes->SetSize(0);
    } else {
        auto workspaceSizeVec = reinterpret_cast<size_t*>(workspaceSizes->MutableData());
        // 当前默认2个中间内存大小相同
        workspaceSizeVec[0] = memSize;
        workspaceSizeVec[1] = memSize;
        (void)workspaceSizes->SetSize(MAX_WORKSPACE_COUNT);
    }
    return ge::SUCCESS;
}

REGISTER_KERNEL(GenerateSqeAndLaunchTask).RunFunc(GenerateSqeAndLaunchTask)
    .OutputsCreator(CreateGenerateDvppSqeOutputs);
REGISTER_KERNEL(CalcDvppWorkSpaceSize).RunFunc(CalcOpWorkSpaceSize).OutputsCreator(CreateCalcWorkspaceSizeOutputs);
}  // namespace kernel
}  // namespace gert
