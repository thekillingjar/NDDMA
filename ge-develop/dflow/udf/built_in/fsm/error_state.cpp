/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fsm/error_state.h"
#include "entity/llm_comm_entity_mgr.h"

namespace FlowFunc {
FsmStatus ErrorState::Preprocess(LlmCommEntity &entity) {
    UDF_LOG_DEBUG("Enter error state, entity:%s.", entity.GetDesc().c_str());
    entity.ClearResource();
    return Process(entity);
}

FsmStatus ErrorState::Process(LlmCommEntity &entity) {
    entity.Dump();
    return Postprocess(entity);
}

FsmStatus ErrorState::Postprocess(LlmCommEntity &entity) {
    (void) entity;
    return FsmStatus::kFsmSuccess;
}
}  // namespace FlowFunc
