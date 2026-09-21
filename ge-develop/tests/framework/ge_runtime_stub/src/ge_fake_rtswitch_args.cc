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
#include <iostream>
#include "stub/ge_fake_rtswitch_args.h"
using namespace std;

namespace ge {
GetAllSwitchArgs::GetAllSwitchArgs(void *ptr, void *value_ptr, std::unique_ptr<std::string> tag) :
    handle_(ptr), tag_name_(std::move(tag)) { 
        size_t data_size = 2 * sizeof(void *);
        args_holder_ = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[data_size]());
        uintptr_t *ptr_s = reinterpret_cast<uintptr_t *>(args_holder_.get());
        uintptr_t *value_ptr_s = reinterpret_cast<uintptr_t *>(args_holder_.get() + sizeof(void *));
        *ptr_s = reinterpret_cast<uint64_t>(ptr);
        *value_ptr_s = reinterpret_cast<uint64_t>(value_ptr); 

        switch_addr_ = (uint64_t)(ptr_s);
    }
}
