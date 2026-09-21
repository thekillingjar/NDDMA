/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_FLOW_MSG_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_FLOW_MSG_H_
#include <memory>
#include "ge/ge_api_error_codes.h"
#include "ge/ge_data_flow_api.h"
#include "dflow/base/deploy/exchange_service.h"
#include "framework/common/runtime_tensor_desc.h"

namespace ge {
class FlowMsgBase : public FlowMsg {
 public:
  FlowMsgBase();

  ~FlowMsgBase() override;

  /**
   * @brief get msg type.
   * @return msg type.
   */
  MsgType GetMsgType() const override;

  /**
   * @brief set msg type.
   * @param msg type.
   */
  void SetMsgType(MsgType msg_type) override;

  /**
   * @brief get tensor.
   * when MsgType=TENSOR_DATA_TYPE, can get tensor.
   * @return tensor.
   * attention:
   * Tensor life cycle is same as FlowMsg, Caller should not keep it alone.
   */
  Tensor *GetTensor() const override;

  /**
   * @brief get raw data.
   * @param data ptr.
   * @param data size.
   * @return success:0, failed:others.
   */
  Status GetRawData(void *&data_ptr, uint64_t &data_size) const override;

  /**
   * @brief get ret code.
   * @return ret code.
   */
  int32_t GetRetCode() const override;

  /**
   * @brief set ret code.
   * @param ret code.
   */
  void SetRetCode(int32_t ret_code) override;

  /**
   * @brief set start time.
   * @param start time.
   */
  void SetStartTime(uint64_t start_time) override;

  /**
   * @brief get start time.
   * @return start time.
   */
  uint64_t GetStartTime() const override;

  /**
   * @brief set end time.
   * @param end time.
   */
  void SetEndTime(uint64_t end_time) override;

  /**
   * @brief get end time.
   * @return end time.
   */
  uint64_t GetEndTime() const override;

  /**
   * @brief set flow flags.
   * @param flags flags is FlowFlag set, multi FlowFlags can use bit or operation.
   */
  void SetFlowFlags(uint32_t flags) override;

  /**
   * @brief get flow flags.
   * @return FlowFlag set, can use bit and operation check if FlowFlag is set.
   */
  uint32_t GetFlowFlags() const override;

  /**
   * @brief get transaction id.
   * @return transaction id.
   */
  uint64_t GetTransactionId() const override;

  /**
   * @brief set transaction id.
   * @param transaction id.
   */
  void SetTransactionId(uint64_t transaction_id) override;

  /**
   * @brief set user data, max data size is 64.
   * @param data user data point, input.
   * @param size user data size, need in (0, 64].
   * @param offset user data offset, need in [0, 64), size + offset <= 64.
   * @return success:0, failed:others.
   */
  Status SetUserData(const void *data, size_t size, size_t offset = 0U) override;

  /**
   * @brief get user data, max data size is 64.
   * @param data user data point, output.
   * @param size user data size, need in (0, 64]. The value must be the same as that of SetUserData.
   * @param offset user data offset, need in [0, 64), size + offset <= 64. The value must be the same as that of
   * SetUserData.
   * @return success:0, failed:others.
   */
  Status GetUserData(void *data, size_t size, size_t offset = 0U) const override;

  rtMbufPtr_t MbufCopyRef() const;

  bool IsEndOfSequence() const;

  static Status GetMsgType(rtMbufPtr_t mbuf, MsgType &msg_type, bool &is_null_data);

 protected:
  void SetNullData() const;

  Status BuildFlowMsg(rtMbufPtr_t mbuf);

  Status MbufAlloc(size_t size);

  Status MbufGetBuffer(void *&buffer) const;

 private:
  Status CheckParamsForUserData(const void *data, size_t size, size_t offset) const;

  ExchangeService::MsgInfo* msg_info_ = nullptr;
  int8_t *user_data_ = nullptr;
  size_t user_data_size_ = 0U;
  rtMbufPtr_t alloced_mbuf_ = nullptr;
  bool is_end_of_sequence_ = false;
};

using FlowMsgBasePtr = std::shared_ptr<FlowMsgBase>;

class TensorFlowMsg : public FlowMsgBase {
 public:
  TensorFlowMsg();

  ~TensorFlowMsg() override;

  Tensor *GetTensor() const override;

  Status AllocTensor(const TensorDesc &tensor_desc);

  // take over ownership
  Status BuildTensor(rtMbufPtr_t mbuf, const GeTensorDesc &output_desc);

 private:
  static Status UpdateTensorDesc(const RuntimeTensorDesc &runtime_desc, GeTensorDesc &tensor_desc);

  std::shared_ptr<Tensor> tensor_;
};

class RawDataFlowMsg : public FlowMsgBase {
 public:
  RawDataFlowMsg();

  ~RawDataFlowMsg() override;

  Status GetRawData(void *&data_ptr, uint64_t &data_size) const override;

  Status AllocRawData(size_t size);

  // take over ownership
  Status BuildRawData(rtMbufPtr_t mbuf);

 private:
  void *addr_ = nullptr;
  size_t len_ = 0U;
};

class EmptyDataFlowMsg : public FlowMsgBase {
 public:
  EmptyDataFlowMsg();

  ~EmptyDataFlowMsg() override;

  Tensor *GetTensor() const override;

  Status AllocEmptyData(MsgType type);

  // take over ownership
  Status BuildNullData(rtMbufPtr_t mbuf);
};
}  // namespace ge

#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_FLOW_MSG_H_
