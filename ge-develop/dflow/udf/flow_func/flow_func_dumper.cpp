/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_func_dumper.h"

namespace FlowFunc {
std::shared_ptr<FlowFuncDumper> FlowFuncDumpManager::dumper_impl_ = nullptr;

void FlowFuncDumpManager::SetDumper(std::shared_ptr<FlowFuncDumper> dumper) {
    dumper_impl_ = dumper;
}

bool FlowFuncDumpManager::IsEnable() {
    return dumper_impl_ != nullptr;
}

int32_t FlowFuncDumpManager::DumpInput(const std::string &flow_func_info, const std::vector<MbufTensor *> &tensors,
                                       const uint32_t step_id) {
    if (!IsEnable()) {
        // no need dump
        return FLOW_FUNC_SUCCESS;
    }
    return dumper_impl_->DumpInput(flow_func_info, tensors, step_id);
}

int32_t FlowFuncDumpManager::DumpOutput(const std::string &flow_func_info, uint32_t index, const MbufTensor *tensor,
                                        const uint32_t step_id) {
    if (!IsEnable()) {
        // no need dump
        return FLOW_FUNC_SUCCESS;
    }
    return dumper_impl_->DumpOutput(flow_func_info, index, tensor, step_id);
}

bool FlowFuncDumpManager::IsInDumpStep(const uint32_t step_id) {
    if (!IsEnable()) {
        // no need dump
        return false;
    }
    return dumper_impl_->IsInDumpStep(step_id);
}
}