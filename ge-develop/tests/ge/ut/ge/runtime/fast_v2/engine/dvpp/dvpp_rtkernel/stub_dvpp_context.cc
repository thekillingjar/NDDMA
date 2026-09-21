/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include "stub_dvpp_context.h"
#include <stdio.h>

namespace ge {
Node::Node() {}
Node::~Node() {}
}

namespace gert {

Chain::Chain()
{
    data_ = nullptr;
}
Chain::~Chain()
{
    if (data_ != nullptr) {
        //delete data_;
        data_ = nullptr;
    }
    FreeResource();
}

void Chain::SetWithDefaultDeleter(void* data)
{
    data_ = reinterpret_cast<uint8_t*>(data);
}

KernelContext::KernelContext()
{
    for (int i = 0; i < 4; i++) {
        chainArray_[i] = new Chain();
    }
}
KernelContext::~KernelContext()
{
    for (int i = 0; i < 4; i++) {
        if (chainArray_[i] != nullptr) {
            delete chainArray_[i];
            chainArray_[i] = nullptr;
        }
    }

}

Chain* KernelContext::GetOutput(size_t i)
{
    if (i < 4) {
        return chainArray_[i];
    }
    return nullptr;
}

#if 0
template<>
gert::ContinuousVector* KernelContext::GetOutputPointer<gert::ContinuousVector>(size_t i) {
    return reinterpret_cast<gert::ContinuousVector*>(0);
}
#endif
}