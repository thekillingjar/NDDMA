/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fsm/receive_check_state.h"
#include <mutex>
#include "entity/llm_comm_entity_mgr.h"
#include "llm_common/cache_manager.h"
#include "llm_common/hccl_proxy.h"

namespace FlowFunc {

FsmStatus ReceiveCheckState::Preprocess(LlmCommEntity &entity) {
    UDF_LOG_INFO("Enter receive check state, entity:%s.", entity.GetDesc().c_str());
    std::vector<HcclMessage> &probe_msgs = entity.GetProbeMsgs();
    HcclRequest request;
    HcclResult ret = HcclRawImrecv(&entity.GetCheckLinkRecvData(), sizeof(uint8_t), HCCL_DATA_TYPE_INT8,
                                   &probe_msgs.front(), &request);
    if (ret != HCCL_SUCCESS) {
        UDF_LOG_ERROR("Fail to call HcclRawImrecv for check status, entity:%s.", entity.GetDesc().c_str());
        return FsmStatus::kFsmHcclFailed;
    }
    entity.GetReceiveRequests().emplace_back(request);  // only one receive request
    return Process(entity);
}

FsmStatus ReceiveCheckState::Process(LlmCommEntity &entity) {
    std::vector<HcclRequest> &receive_requests = entity.GetReceiveRequests();
    const int32_t test_count = 1;
    int32_t comp_count = 0;
    auto ret = entity.TestCompleteAsync(receive_requests.data(), test_count, comp_count);
    if (ret != FsmStatus::kFsmSuccess) {
        return FsmStatus::kFsmHcclFailed;
    }
    if (comp_count == 0) {
        UDF_LOG_INFO("Test some when receive transfer kv request ret complete count is zero.");
        return FsmStatus::kFsmKeepState;
    }
    return Postprocess(entity);
}

FsmStatus ReceiveCheckState::Postprocess(LlmCommEntity &entity) {
    UDF_LOG_INFO("Finish receive check state, entity:%s.", entity.GetDesc().c_str());
    return entity.ChangeState(FsmState::kFsmIdleState);
}
}  // namespace FlowFunc