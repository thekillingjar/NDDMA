/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STUB_KERNEL_REGISTRY_H
#define STUB_KERNEL_REGISTRY_H
#include <functional>
#include <string>
#include <memory>
#include "stub_dvpp_context.h"
#include "stub_fast_node.h"

namespace gert {
class KernelRegister {
public:
    KernelRegister(const char* kernelType);
    ~KernelRegister();
    typedef uint32_t (*KernelFunc)(gert::KernelContext *context);
    typedef uint32_t (*OutputCreator)(const ge::FastNode *, KernelContext *);

    KernelRegister& RunFunc(KernelRegister::KernelFunc func);
    KernelRegister& OutputsCreator(KernelRegister::OutputCreator func);
};

KernelRegister::KernelRegister(const char* kernelType) {}
KernelRegister::~KernelRegister() {}

KernelRegister& KernelRegister::RunFunc(KernelRegister::KernelFunc func)
{
    return *this;
}

KernelRegister& KernelRegister::OutputsCreator(KernelRegister::OutputCreator func)
{
    return *this;
}

}
#define REGISTER_KERNEL_COUNTER2(type, counter) static auto g_register_kernel_##counter = gert::KernelRegister(#type)
#define REGISTER_KERNEL_COUNTER(type, counter) REGISTER_KERNEL_COUNTER2(type, counter)
#define REGISTER_KERNEL(type) REGISTER_KERNEL_COUNTER(type, __COUNTER__)
#endif // #ifndef STUB_KERNEL_REGISTRY_H
