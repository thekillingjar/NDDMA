/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_FAKE_RTSWITCH_ARGS_H
#define GE_FAKE_RTSWITCH_ARGS_H
#include <cstdlib>
#include "runtime_stub.h"

namespace ge {
struct GetAllSwitchArgs {
    GetAllSwitchArgs(void *ptr, void *value_ptr, std::unique_ptr<std::string> tag);

    const std::string *GetTag() const {
        return tag_name_.get();
    }

private:
  uint64_t switch_addr_{0};
  std::unique_ptr<std::string> tag_name_;
  const void *handle_;
  std::unique_ptr<uint8_t[]> args_holder_;
};
}

#endif
