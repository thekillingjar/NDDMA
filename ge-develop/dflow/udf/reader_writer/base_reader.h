/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HICAID_BASE_READER_H
#define HICAID_BASE_READER_H

#include "flow_func/flow_func_defines.h"
#include "common/inner_error_codes.h"

namespace FlowFunc {

struct FLOW_FUNC_VISIBILITY ReadInitOptions {};

class FLOW_FUNC_VISIBILITY BaseReader {
public:
    BaseReader() = default;

    virtual ~BaseReader() = default;

    virtual int32_t Init(const ReadInitOptions &) {
        return HICAID_SUCCESS;
    }

    virtual void ReadMessage() = 0;

protected:
    virtual int32_t Enable() {
        return HICAID_SUCCESS;
    }

    virtual int32_t Disable() {
        return HICAID_SUCCESS;
    }
};
}  // namespace FlowFunc

#endif