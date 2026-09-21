/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_META_RUN_CONTEXT_H
#define FLOW_FUNC_META_RUN_CONTEXT_H

#include <vector>
#include <memory>
#include "flow_func_defines.h"
#include "attr_value.h"
#include "flow_msg.h"
#include "meta_params.h"
#include "out_options.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY MetaRunContext {
 public:
  MetaRunContext() = default;

  ~MetaRunContext() = default;

  /**
   * @brief alloc tensor msg by shape and data type.
   * @param shape tensor shape
   * @param data_type data type
   * @return tensor
   */
  virtual std::shared_ptr<FlowMsg> AllocTensorMsg(const std::vector<int64_t> &shape, TensorDataType data_type) {
    auto flow_msg = std::make_shared<FlowMsg>(shape, data_type);
    return flow_msg;
  }

  /**
   * @brief alloc empty data msg.
   * @param msg_type msg type which msg will be alloc
   * @return empty data FlowMsg
   */
  virtual std::shared_ptr<FlowMsg> AllocEmptyDataMsg(MsgType msg_type) {
    return AllocTensorMsg({0}, TensorDataType::DT_FLOAT);
  }

  virtual std::shared_ptr<FlowMsg> AllocRawDataMsg(int64_t size, uint32_t align = 512) {
    auto flow_msg = std::make_shared<FlowMsg>(size);
    return flow_msg;
  }

  virtual std::shared_ptr<FlowMsg> ToFlowMsg(std::shared_ptr<Tensor> tensor) {
    auto flow_msg = std::make_shared<FlowMsg>(tensor);
    return flow_msg;
  }

  /**
   * @brief set output tensor.
   * @param out_idx output index, start from 0.
   * @param out_msg output msg.
   * @return 0:success, other failed.
   */
  virtual int32_t SetOutput(uint32_t out_idx, std::shared_ptr<FlowMsg> out_msg) {
    if (out_idx >= out_msgs_.size()) {
      out_msgs_.resize(out_idx + 1UL);
    }
    out_msgs_[out_idx] = out_msg;
    return FLOW_FUNC_SUCCESS;
  }

  /**
   * @brief run flow model.
   * @param model_key invoked flow model key.
   * @param input_msgs flow model input message.
   * @param output_msgs flow model output message.
   * @param timeout timeout(ms), -1 means never timeout.
   * @return 0:success, other failed.
   */
  virtual int32_t RunFlowModel(const char *model_key, const std::vector<std::shared_ptr<FlowMsg>> &input_msgs,
                               std::vector<std::shared_ptr<FlowMsg>> &output_msgs, int32_t timeout) {
    (void)model_key;
    (void)input_msgs;
    (void)output_msgs;
    (void)timeout;
    return FLOW_FUNC_SUCCESS;
  }

  /**
   * @brief get user data, max data size is 64.
   * @param data user data point, output.
   * @param size user data size, need in (0, 64].
   * @param offset user data offset, need in [0, 64), size + offset <= 64.
   * @return success:FLOW_FUNC_SUCCESS, failed:OTHERS.
   */
  virtual int32_t GetUserData(void *data, size_t size, size_t offset = 0U) const;

  /**
   * @brief set output tensor.
   * @param out_idx output index, start from 0.
   * @param out_msg output msg.
   * @param options output options.
   * @return 0:success, other failed.
   */
  virtual int32_t SetOutput(uint32_t out_idx, std::shared_ptr<FlowMsg> out_msg, const OutOptions &options) {
    if (out_idx >= out_msgs_.size()) {
      out_msgs_.resize(out_idx + 1UL);
    }
    out_msgs_[out_idx] = out_msg;
    return FLOW_FUNC_SUCCESS;
  }

  /**
   * @brief set output tensor.
   * @param out_idx output index, start from 0.
   * @param out_msgs output msgs.
   * @param options output options.
   * @return 0:success, other failed.
   */
  virtual int32_t SetMultiOutputs(uint32_t out_idx, const std::vector<std::shared_ptr<FlowMsg>> &out_msgs,
                                  const OutOptions &options) {
    return FLOW_FUNC_SUCCESS;
  }

  /**
   * @brief alloc tensor msg by shape and data type.
   * @param shape tensor shape
   * @param data_type data type
   * @param align: align of tensor, must be[32, 1014] and must can be divisible by 1024.
   * @return tensor
   */
  virtual std::shared_ptr<FlowMsg> AllocTensorMsgWithAlign(const std::vector<int64_t> &shape, TensorDataType data_type,
                                                           uint32_t align) {
    auto flow_msg = std::make_shared<FlowMsg>(shape, data_type);
    return flow_msg;
  }

  void RaiseException(int32_t exp_code, uint64_t user_context_id);
  bool GetException(int32_t &exp_code, uint64_t &user_context_id);

 private:
  MetaParams meta_params_;
  std::vector<std::shared_ptr<FlowMsg>> out_msgs_;
};
}  // namespace FlowFunc

#endif  // FLOW_FUNC_META_RUN_CONTEXT_H
