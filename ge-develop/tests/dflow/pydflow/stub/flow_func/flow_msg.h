/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_FLOW_MSG_H
#define FLOW_FUNC_FLOW_MSG_H

#include <cstdint>
#include <vector>
#include "flow_func_defines.h"
#include "tensor_data_type.h"

namespace FlowFunc {
enum class MsgType : uint16_t {
  MSG_TYPE_TENSOR_DATA = 0,  // tensor data msg type
  MSG_TYPE_RAW_MSG           // raw data msg type
};

enum class FlowFlag : uint32_t {
  FLOW_FLAG_EOS = (1U << 0U),  // data flag end
  FLOW_FLAG_SEG = (1U << 1U)   // segment flag for discontinuous
};

class FLOW_FUNC_VISIBILITY Tensor {
 public:
  Tensor();

  Tensor(const std::vector<int64_t> &shape, const TensorDataType &data_type);

  ~Tensor();

  virtual const std::vector<int64_t> &GetShape() const {
    return stub_shape_;
  }

  virtual TensorDataType GetDataType() const {
    return stub_data_type_;
  }

  virtual void *GetData() const {
    return data_;
  }

  virtual uint64_t GetDataSize() const {
    return data_size_;
  }

  /**
   * @brief get tensor element count.
   * @return element count. if return -1, it means unknown or overflow.
   */
  virtual int64_t GetElementCnt() const;

  virtual int32_t Reshape(const std::vector<int64_t> &shape);

 private:
  void *data_ = nullptr;
  uint64_t data_size_ = 0UL;
  std::vector<int64_t> stub_shape_ = {2, 5};
  TensorDataType stub_data_type_ = TensorDataType::DT_UINT32;
};

struct MbufHeadMsg {
  uint64_t trans_id;
  uint16_t version;
  uint16_t msg_type;
  int32_t ret_code;
  uint64_t start_time;
  uint64_t end_time;
  uint32_t flags;
  uint8_t data_flag;
  uint8_t rsv[23];
  uint32_t route_label;  // will be delete when new design given.

  std::string DebugString() const;
};

class FLOW_FUNC_VISIBILITY FlowBufferFactory{public : static std::shared_ptr<Tensor> AllocTensor(
    const std::vector<int64_t> &shape, TensorDataType data_type, uint32_t align = 512U){(void)align;
return std::make_shared<Tensor>(shape, data_type);
}  // namespace FlowFunc
}
;

class FLOW_FUNC_VISIBILITY FlowMsg {
 public:
  FlowMsg() {
    flowFuncTensor_ = std::make_shared<Tensor>();
    raw_data_.resize(2024);
  }

  FlowMsg(const std::vector<int64_t> &shape, const TensorDataType &data_type) {
    flowFuncTensor_ = std::make_shared<Tensor>(shape, data_type);
    raw_data_.resize(2024);
  }

  explicit FlowMsg(int64_t size) {
    raw_data_.resize(size);
  }

  explicit FlowMsg(std::shared_ptr<Tensor> tensor) : flowFuncTensor_(tensor) {}

  ~FlowMsg() = default;

  virtual MsgType GetMsgType() const {
    return MsgType::MSG_TYPE_TENSOR_DATA;
  }

  virtual void SetMsgType(MsgType msg_type) {
    head_msg_.msg_type = static_cast<uint16_t>(msg_type);
  }

  virtual Tensor *GetTensor() const {
    return flowFuncTensor_.get();
  }

  virtual int32_t GetRetCode() const {
    return head_msg_.ret_code;
  }

  virtual void SetRetCode(int32_t ret_code) {
    head_msg_.ret_code = ret_code;
  }

  virtual void SetStartTime(uint64_t start_time) {
    head_msg_.start_time = start_time;
  }

  virtual uint64_t GetStartTime() const {
    return head_msg_.start_time;
  }

  virtual void SetEndTime(uint64_t end_time) {
    head_msg_.end_time = end_time;
  }

  virtual uint64_t GetEndTime() const {
    return head_msg_.end_time;
  }

  /**
   * @brief set flow flags.
   * @param flags flags is FlowFlag set, multi FlowFlags can use bit or operation.
   */
  virtual void SetFlowFlags(uint32_t flags) {
    head_msg_.flags = flags;
  }

  /**
   * @brief get flow flags.
   * @return FlowFlag set, can use bit and operation check if FlowFlag is set.
   */
  virtual uint32_t GetFlowFlags() const {
    return head_msg_.flags;
  }

  /**
   * @brief set route label.
   * @param route_label route label, value 0 means not used.
   */
  virtual void SetRouteLabel(uint32_t route_label) {
    head_msg_.route_label = route_label;
  }

  virtual uint64_t GetTransactionId() const {
    return UINT64_MAX;
  }

  virtual void SetTransactionId(uint64_t transaction_id) {}

  virtual int32_t GetRawData(void *&data_ptr, uint64_t &data_size) const {
    data_ptr = static_cast<void *>(&raw_data_[0]);
    data_size = raw_data_.size();
    return 0;
  }

 private:
  mutable std::shared_ptr<Tensor> flowFuncTensor_;
  MbufHeadMsg head_msg_ = {};
  mutable std::vector<uint8_t> raw_data_;
};
}

#endif  // FLOW_FUNC_FLOW_MSG_H