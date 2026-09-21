/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef BUILT_IN_COUNT_BATCH_FLOW_FUNC_H
#define BUILT_IN_COUNT_BATCH_FLOW_FUNC_H

#include <mutex>
#include <vector>
#include <deque>
#include "flow_func/meta_flow_func.h"

namespace FlowFunc {
class CountBatchFlowFunc : public MetaFlowFunc {
public:
    CountBatchFlowFunc() = default;
    ~CountBatchFlowFunc() override;
    int32_t Init() override;
    int32_t Proc(const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) override;

private:
    int32_t GetBatchAttr();
    int32_t PaddingToBatchSize(int64_t padding_cnt);
    int32_t PaddingInputCache(size_t cache_size);
    int32_t ConstructOutputTensor(std::deque<std::pair<std::shared_ptr<FlowMsg>, bool>> &batch_tensor,
        std::shared_ptr<FlowMsg> &output_flow_msg) const;
    int32_t CheckTensorInfo(const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) const;
    int32_t CheckInput(const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) const;
    void AbnormalProc(int32_t error_code);
    /*
     * bool : true : this flowMsg is user's input flowMsg,
     *       false : this flowMsg is padding with 0.
     */
    std::vector<std::deque<std::pair<std::shared_ptr<FlowMsg>, bool>>> batch_flow_msg_;  // cache msg;
    int64_t batch_size_ = 0L;
    int64_t timeout_ = 0L;
    int64_t slide_stride_ = 0L;
    std::mutex mutex_;
    uint32_t published_output_num_ = 0U;
    uint32_t total_output_num_ = 0U;
    bool timer_flag_ = false;
    bool padding_ = false;
    void *timer_handle_ = nullptr;
    uint64_t start_time_ = 0UL;
};
}  // namespace FlowFunc

#endif  // BUILT_IN_COUNT_BATCH_FLOW_FUNC_H
