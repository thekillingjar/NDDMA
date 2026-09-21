/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_DUMPER_H
#define FLOW_FUNC_DUMPER_H

#include "mbuf_flow_msg.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY FlowFuncDumper {
public:
    FlowFuncDumper() = default;

    virtual ~FlowFuncDumper() = default;

    virtual int32_t DumpInput(const std::string &flow_func_info, const std::vector<MbufTensor *> &tensors,
                              const uint32_t step_id) = 0;

    virtual int32_t DumpOutput(const std::string &flow_func_info, uint32_t index, const MbufTensor *tensor,
                               const uint32_t step_id) = 0;

    virtual bool IsInDumpStep(const uint32_t step_id) = 0;
};

class FLOW_FUNC_VISIBILITY FlowFuncDumpManager {
public:
    static void SetDumper(std::shared_ptr<FlowFuncDumper> dumper);

    static int32_t DumpInput(const std::string &flow_func_info, const std::vector<MbufTensor *> &tensors,
                             const uint32_t step_id);

    static int32_t DumpOutput(const std::string &flow_func_info, uint32_t index, const MbufTensor *tensor,
                              const uint32_t step_id);

    static bool IsInDumpStep(const uint32_t step_id);

    static bool IsEnable();
private:
    static std::shared_ptr<FlowFuncDumper> dumper_impl_;
};
}

#endif // FLOW_FUNC_DUMPER_H
