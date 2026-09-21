/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NPU_SCHED_PROCESSOR_H
#define NPU_SCHED_PROCESSOR_H

#include "common/common_define.h"
#include "model/flow_func_model.h"
#include "data_flow/deploy/npu_sched_model.h"

namespace FlowFunc {
class NpuSchedProcessor {
public:
    NpuSchedProcessor() = default;

    ~NpuSchedProcessor();

    int32_t Initialize(int32_t device_id);

    void Finalize();

    int32_t LoadNpuSchedModel(const FlowFuncModel &model, ModelQueueInfos &model_queue_infos);

private:

    void UnloadAndDestroyAllModelHandler();

    void GenerateNpuSchedLoadParam(const FlowFuncModel &flow_func_model, NpuSchedLoadParam &npu_sched_load_param) const;

    void FinalizeNpuSched() const;

    template<typename FUNC>
    FUNC GetFunc(const char *func_name) const;

    void *so_handle_ = nullptr;
    std::vector<NpuSchedModelHandler> model_handler_list_;
    // default no device
    int32_t device_id_ = -1;
};
}

#endif // NPU_SCHED_PROCESSOR_H
