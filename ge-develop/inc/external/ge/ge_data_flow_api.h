/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_GE_GE_DATA_FLOW_API_H
#define INC_EXTERNAL_GE_GE_DATA_FLOW_API_H
#include <memory>
#include "ge_api_error_codes.h"
#include "graph/tensor.h"

namespace ge {
enum class DataFlowFlag : uint32_t {
  DATA_FLOW_FLAG_EOS = (1U << 0U),  // data flag end
  DATA_FLOW_FLAG_SEG = (1U << 1U)   // segment flag for discontinuous
};

struct RawData {
  const void *addr;
  size_t len;
};

class DataFlowInfoImpl;
class GE_FUNC_VISIBILITY DataFlowInfo {
 public:
  DataFlowInfo();
  ~DataFlowInfo();

  DataFlowInfo(const DataFlowInfo &context) = delete;
  DataFlowInfo(const DataFlowInfo &&context) = delete;
  DataFlowInfo &operator=(const DataFlowInfo &context)& = delete;
  DataFlowInfo &operator=(const DataFlowInfo &&context)& = delete;

  void SetStartTime(const uint64_t start_time);
  uint64_t GetStartTime() const;

  void SetEndTime(const uint64_t end_time);
  uint64_t GetEndTime() const;

  /**
   * @brief Set the Flow Flags object.
   * @param flow_flags can use operate | to merge multi DataFlowFla to flags.
   */
  void SetFlowFlags(const uint32_t flow_flags);
  /**
   * @brief Get the Flow Flags object.
   * @return uint32_t flow flags, can use operate & with DataFlowFlag to check which bit is set.
   */
  uint32_t GetFlowFlags() const;

  /**
   * @brief set user data, max data size is 64.
   * @param data user data point, input.
   * @param size user data size, need in (0, 64].
   * @param offset user data offset, need in [0, 64), size + offset <= 64.
   * @return success:0, failed:others.
   */
  Status SetUserData(const void *data, size_t size, size_t offset = 0U);

  /**
   * @brief get user data, max data size is 64.
   * @param data user data point, output.
   * @param size user data size, need in (0, 64]. The value must be the same as that of SetUserData.
   * @param offset user data offset, need in [0, 64), size + offset <= 64. The value must be the same as that of
   * SetUserData.
   * @return success:0, failed:others.
   */
  Status GetUserData(void *data, size_t size, size_t offset = 0U) const;

  /**
   * get transaction id.
   * only set n-mapping attr value true it can take fetch data transaction id out.
   * @return transaction id
   */
  uint64_t GetTransactionId() const;

  /**
   * set transaction id.
   * Enabled only when set n-mapping attr value true.
   * @param transaction_id transaction id
   */
  void SetTransactionId(uint64_t transaction_id);

 private:
  std::shared_ptr<DataFlowInfoImpl> impl_;
  friend class DataFlowInfoUtils;
};

enum class MsgType : uint16_t {
  MSG_TYPE_TENSOR_DATA = 0,              // tensor data msg type
  MSG_TYPE_RAW_MSG = 1,                  // raw data msg type
  MSG_TYPE_USER_DEFINE_START = 1024
};


class GE_FUNC_VISIBILITY FlowMsg {
 public:
  FlowMsg() = default;

  virtual ~FlowMsg() = default;

  /**
   * @brief get msg type.
   * @return msg type.
   */
  virtual MsgType GetMsgType() const = 0;

  /**
   * @brief set msg type.
   * @param msg type.
   */
  virtual void SetMsgType(MsgType msg_type) = 0;

  /**
   * @brief get tensor.
   * when MsgType=TENSOR_DATA_TYPE, can get tensor.
   * @return tensor.
   * attention:
   * Tensor life cycle is same as FlowMsg, Caller should not keep it alone.
   */
  virtual Tensor *GetTensor() const = 0;

  /**
   * @brief get raw data.
   * @param data ptr.
   * @param data size.
   * @return success:0, failed:others.
   */
  virtual Status GetRawData(void *&data_ptr, uint64_t &data_size) const = 0;

  /**
   * @brief get ret code.
   * @return ret code.
   */
  virtual int32_t GetRetCode() const = 0;

  /**
   * @brief set ret code.
   * @param ret code.
   */
  virtual void SetRetCode(int32_t ret_code) = 0;

  /**
   * @brief set start time.
   * @param start time.
   */
  virtual void SetStartTime(uint64_t start_time) = 0;

  /**
   * @brief get start time.
   * @return start time.
   */
  virtual uint64_t GetStartTime() const = 0;

  /**
   * @brief set end time.
   * @param end time.
   */
  virtual void SetEndTime(uint64_t end_time) = 0;

  /**
   * @brief get end time.
   * @return end time.
   */
  virtual uint64_t GetEndTime() const = 0;

  /**
   * @brief set flow flags.
   * @param flags flags is FlowFlag set, multi FlowFlags can use bit or operation.
   */
  virtual void SetFlowFlags(uint32_t flags) = 0;

  /**
   * @brief get flow flags.
   * @return FlowFlag set, can use bit and operation check if FlowFlag is set.
   */
  virtual uint32_t GetFlowFlags() const = 0;

  /**
   * @brief get transaction id.
   * @return transaction id.
   */
  virtual uint64_t GetTransactionId() const = 0;

  /**
   * @brief set transaction id.
   * @param transaction id.
   */
  virtual void SetTransactionId(uint64_t transaction_id) = 0;

  /**
   * @brief set user data, max data size is 64.
   * @param data user data point, input.
   * @param size user data size, need in (0, 64].
   * @param offset user data offset, need in [0, 64), size + offset <= 64.
   * @return success:0, failed:others.
   */
  virtual Status SetUserData(const void *data, size_t size, size_t offset = 0U) = 0;

  /**
   * @brief get user data, max data size is 64.
   * @param data user data point, output.
   * @param size user data size, need in (0, 64]. The value must be the same as that of SetUserData.
   * @param offset user data offset, need in [0, 64), size + offset <= 64. The value must be the same as that of
   * SetUserData.
   * @return success:0, failed:others.
   */
  virtual Status GetUserData(void *data, size_t size, size_t offset = 0U) const = 0;
};

using FlowMsgPtr = std::shared_ptr<FlowMsg>;

class GE_FUNC_VISIBILITY FlowBufferFactory {
 public:
  static std::shared_ptr<Tensor> AllocTensor(const std::vector<int64_t> &shape,
                                             DataType data_type,
                                             uint32_t align = 512U);
  static FlowMsgPtr AllocTensorMsg(const std::vector<int64_t> &shape, DataType data_type, uint32_t align = 512U);
  static FlowMsgPtr AllocRawDataMsg(size_t size, uint32_t align = 512U);
  static FlowMsgPtr AllocEmptyDataMsg(MsgType type);
  static FlowMsgPtr ToFlowMsg(const Tensor &tensor);
  static FlowMsgPtr ToFlowMsg(const RawData &raw_data);
};
}  // namespace ge
#endif  // INC_EXTERNAL_GE_GE_DATA_FLOW_API_H
