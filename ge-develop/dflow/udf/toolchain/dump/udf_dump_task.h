/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UDF_TOOLCHAIN_DUMP_UDF_DUMP_TASK_H
#define UDF_TOOLCHAIN_DUMP_UDF_DUMP_TASK_H

#include <cstdint>
#include <string>
#include <vector>
#include "flow_func/mbuf_flow_msg.h"

namespace FlowFunc {
class UdfDumpTask {
public:
    UdfDumpTask() = default;

    virtual ~UdfDumpTask() = default;

    virtual int32_t DumpOpInfo(const std::string &dump_file_path, const uint32_t step_id) = 0;

    virtual int32_t PreProcessOutput(uint32_t index, const MbufTensor *tensor) = 0;

    virtual int32_t PreProcessInput(const std::vector<MbufTensor *> &tensors) = 0;
};
}

#endif // UDF_TOOLCHAIN_DUMP_UDF_DUMP_TASK_H