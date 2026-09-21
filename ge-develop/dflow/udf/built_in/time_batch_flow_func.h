/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef BUILT_IN_TIME_BATCH_FLOW_FUNC_H
#define BUILT_IN_TIME_BATCH_FLOW_FUNC_H

#include "flow_func/meta_flow_func.h"

namespace FlowFunc {
class TimeBatchFlowFunc : public MetaFlowFunc {
public:
    int32_t Init() override;
    int32_t Proc(const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) override;

private:
    bool IsEqualFlowInfo(const std::shared_ptr<FlowMsg> &msg0,
                         const std::shared_ptr<FlowMsg> &msg1) const;
    void ResetState();
    int32_t CheckFlowInfo(const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) const;
    int32_t IsOkShapeForTimeBatch(const std::vector<int64_t> &base_shape,
                                  const std::vector<int64_t> &shape) const;
    int32_t CheckTensorInfo(const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) const;
    int32_t CheckInput(const std::vector<std::shared_ptr<FlowMsg>> &input_msgs);
    int32_t UpdateState(const std::vector<std::shared_ptr<FlowMsg>> &input_msgs);
    void CalcCopyParams(const std::vector<std::shared_ptr<FlowMsg>> &input, std::vector<int64_t> &input_copy_sizes,
                        std::vector<int64_t> &output_shape, int64_t &output_flat_dim0, uint32_t &max_step) const;
    int32_t TimeBatchAll();
    int32_t TimeBatch(const std::vector<std::shared_ptr<FlowMsg>> &input, const uint32_t out_index);
    int32_t PublishErrorOut(const int32_t error_code) const;
    int32_t PublishEmptyEosOut();

    int64_t window_ = -1;
    int64_t batch_dim_ = 0;
    bool drop_remainder_ = false;
    std::vector<std::vector<std::shared_ptr<FlowMsg>>> input_cache_;
    uint64_t start_time_ = 0U;
    uint64_t end_time_ = 0U;
    bool is_time_batch_ok_ = false;
    size_t published_out_num_ = 0U;
    size_t output_num_ = 0U;
    bool is_empty_msgs_ = false;
    bool is_eos_ = false;
};
}

#endif // BUILT_IN_TIME_BATCH_FLOW_FUNC_H