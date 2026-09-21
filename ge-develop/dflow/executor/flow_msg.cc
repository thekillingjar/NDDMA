/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_msg.h"
#include "common/checker.h"
#include "common/util/mem_utils.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "securec.h"
#include "graph/utils/math_util.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/tensor_utils.h"
#include "data_flow_executor_utils.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
constexpr uint8_t kEndOfSequenceFlag = 0x5A;
constexpr uint32_t kMbufHeadEndOfSequencePos = 128U;
}

FlowMsgBase::FlowMsgBase() {}

FlowMsgBase::~FlowMsgBase() {
  if (alloced_mbuf_!= nullptr) {
    GE_CHK_RT(rtMbufFree(alloced_mbuf_));
    alloced_mbuf_ = nullptr;
  }
}

MsgType FlowMsgBase::GetMsgType() const {
  return static_cast<MsgType>(msg_info_->msg_type);
}

void FlowMsgBase::SetMsgType(MsgType msg_type) {
  if (msg_info_ != nullptr) {
    msg_info_->msg_type = static_cast<uint16_t>(msg_type);
  }
}

Tensor *FlowMsgBase::GetTensor() const {
  GELOGE(UNSUPPORTED, "Not supported, please implement the method.");
  return nullptr;
}

Status FlowMsgBase::GetRawData(void *&data_ptr, uint64_t &data_size) const {
  GELOGE(UNSUPPORTED, "Not supported, please implement the method.");
  (void)data_ptr;
  (void)data_size;
  return UNSUPPORTED;
}

int32_t FlowMsgBase::GetRetCode() const {
  int32_t ret_code = 0;
  if (msg_info_ != nullptr) {
    ret_code = msg_info_->ret_code;
  }
  return ret_code;
}

void FlowMsgBase::SetRetCode(int32_t ret_code) {
  if (msg_info_ != nullptr) {
    msg_info_->ret_code = ret_code;
  }
}

void FlowMsgBase::SetStartTime(uint64_t start_time) {
  if (msg_info_ != nullptr) {
    msg_info_->start_time = start_time;
  }
}

uint64_t FlowMsgBase::GetStartTime() const {
  uint64_t start_time = 0UL;
  if (msg_info_ != nullptr) {
    start_time = msg_info_->start_time;
  }
  return start_time;
}

void FlowMsgBase::SetEndTime(uint64_t end_time) {
  if (msg_info_ != nullptr) {
    msg_info_->end_time = end_time;
  }
}

uint64_t FlowMsgBase::GetEndTime() const {
  uint64_t end_time = 0UL;
  if (msg_info_ != nullptr) {
    end_time = msg_info_->end_time;
  }
  return end_time;
}

void FlowMsgBase::SetFlowFlags(uint32_t flags) {
  if (msg_info_ != nullptr) {
    msg_info_->flags = flags;
  }
}

uint32_t FlowMsgBase::GetFlowFlags() const {
  uint32_t flags = 0U;
  if (msg_info_ != nullptr) {
    flags = msg_info_->flags;
  }
  return flags;
}

uint64_t FlowMsgBase::GetTransactionId() const {
  uint64_t trans_id = 0UL;
  if (msg_info_ != nullptr) {
    trans_id = msg_info_->trans_id;
  }
  return trans_id;
}

void FlowMsgBase::SetTransactionId(uint64_t transaction_id) {
  if (msg_info_ != nullptr) {
    if (transaction_id == 0UL) {
      // clear custom trans id flag
      msg_info_->data_flag &= (~kCustomTransIdFlagBit);
    } else {
      msg_info_->data_flag |= kCustomTransIdFlagBit;
    }
    msg_info_->trans_id = transaction_id;
  }
}

void FlowMsgBase::SetNullData() const {
  if (msg_info_ != nullptr) {
    msg_info_->data_flag |= kNullDataFlagBit;
  }
}

bool FlowMsgBase::IsEndOfSequence() const {
  return is_end_of_sequence_;
}

Status FlowMsgBase::GetMsgType(rtMbufPtr_t mbuf, MsgType &msg_type, bool &is_null_data) {
  void *head_buf = nullptr;
  uint64_t head_size = 0U;
  GE_CHK_RT_RET(rtMbufGetPrivInfo(mbuf, &head_buf, &head_size));
  GE_CHECK_NOTNULL(head_buf);
  GE_CHK_BOOL_RET_STATUS(head_size >= sizeof(ExchangeService::MsgInfo), FAILED,
                         "Invalid head, size must >=%zu", sizeof(ExchangeService::MsgInfo));
  ExchangeService::MsgInfo *msg_info = reinterpret_cast<ExchangeService::MsgInfo *>(
      static_cast<char_t *>(head_buf) + head_size - sizeof(ExchangeService::MsgInfo));
  msg_type = static_cast<MsgType>(msg_info->msg_type);
  is_null_data = ((msg_info->data_flag & kNullDataFlagBit) != 0U);
  return SUCCESS;
}

Status FlowMsgBase::BuildFlowMsg(rtMbufPtr_t mbuf) {
  GE_CHK_BOOL_RET_STATUS(alloced_mbuf_ == nullptr, FAILED, "Failed to set mbuf, already set or alloced.");
  alloced_mbuf_ = mbuf;
  void *head_buf = nullptr;
  uint64_t head_size = 0U;
  GE_CHK_RT_RET(rtMbufGetPrivInfo(mbuf, &head_buf, &head_size));
  GE_CHECK_NOTNULL(head_buf);

  if (head_size > kMbufHeadEndOfSequencePos) {
    uint8_t end_of_sequence = *(static_cast<uint8_t *>(head_buf) + kMbufHeadEndOfSequencePos);
    if (end_of_sequence == kEndOfSequenceFlag) {
      is_end_of_sequence_ = true;
    }
  }
  GE_CHK_BOOL_RET_STATUS(head_size >= kMaxUserDataSize, FAILED,
                         "Failed to get user data, the mbuf head size[%" PRIu64 "] < user data size[%zu].",
                        head_size, kMaxUserDataSize);
  user_data_ = static_cast<int8_t *>(head_buf);
  user_data_size_ = kMaxUserDataSize;
  if (head_size >= sizeof(ExchangeService::MsgInfo)) {
    ExchangeService::MsgInfo *msg_info = reinterpret_cast<ExchangeService::MsgInfo *>(
        static_cast<char_t *>(head_buf) + head_size - sizeof(ExchangeService::MsgInfo));
    msg_info_ = msg_info;
  }
  return SUCCESS;
}

Status FlowMsgBase::MbufAlloc(size_t size) {
  // free in ~FlowMsgBase
  GE_CHK_RT_RET(rtMbufAlloc(&alloced_mbuf_, size));
  GE_CHK_RT_RET(rtMbufSetDataLen(alloced_mbuf_, size));
  void *head_buf = nullptr;
  uint64_t head_size = 0U;
  GE_CHK_RT_RET(rtMbufGetPrivInfo(alloced_mbuf_, &head_buf, &head_size));
  if (head_size > kMbufHeadEndOfSequencePos) {
    *(static_cast<uint8_t *>(head_buf) + kMbufHeadEndOfSequencePos) = 0U;
  }
  GE_CHECK_NOTNULL(head_buf);
  if (head_size >= sizeof(ExchangeService::MsgInfo)) {
    msg_info_ = reinterpret_cast<ExchangeService::MsgInfo *>(
        static_cast<char_t *>(head_buf) + head_size - sizeof(ExchangeService::MsgInfo));
    *msg_info_ = {};
  }
  GE_CHK_BOOL_RET_STATUS(head_size >= kMaxUserDataSize, FAILED,
                         "Failed to check head size, the mbuf head size[%" PRIu64 "] < user data size[%zu].",
                        head_size, kMaxUserDataSize);
  user_data_ = static_cast<int8_t *>(head_buf);
  user_data_size_ = kMaxUserDataSize;
  return SUCCESS;
}

Status FlowMsgBase::MbufGetBuffer(void *&buffer) const {
  GE_CHK_RT_RET(rtMbufGetBuffAddr(alloced_mbuf_, &buffer));
  return SUCCESS;
}

rtMbufPtr_t FlowMsgBase::MbufCopyRef() const {
  rtMbufPtr_t tensor_mbuf = nullptr;
  GE_CHK_RT(rtMbufCopyBufRef(alloced_mbuf_, &tensor_mbuf));
  return tensor_mbuf;
}

Status FlowMsgBase::SetUserData(const void *data, size_t size, size_t offset) {
  GE_CHK_STATUS_RET(CheckParamsForUserData(data, size, offset), "Failed to set user data, the params is invalid.");
  const auto cpy_ret = memcpy_s((user_data_ + offset), (user_data_size_ - offset), data, size);
  GE_ASSERT_EOK(cpy_ret, "Failed to set user data, memcpy_s error, size[%zu], offset[%zu], ret[%d].", size, offset,
                cpy_ret);
  GELOGD("Success to set user data, size[%zu], offset[%zu].", size, offset);
  return SUCCESS;
}

Status FlowMsgBase::GetUserData(void *data, size_t size, size_t offset) const {
  GE_CHK_STATUS_RET(CheckParamsForUserData(data, size, offset), "Failed to get user data, the params is invalid.");
  const auto cpy_ret = memcpy_s(data, size, (user_data_ + offset), size);
  GE_ASSERT_EOK(cpy_ret, "Failed to get user data, memcpy_s error, size[%zu], offset[%zu], ret[%d].", size, offset,
                cpy_ret);
  GELOGD("Success to get user data, size[%zu], offset[%zu].", size, offset);
  return SUCCESS;
}


Status FlowMsgBase::CheckParamsForUserData(const void *data, size_t size, size_t offset) const {
  GE_CHK_BOOL_RET_STATUS(data != nullptr, ACL_ERROR_GE_PARAM_INVALID, "The data is nullptr.");
  GE_CHK_BOOL_RET_STATUS(size != 0U, ACL_ERROR_GE_PARAM_INVALID, "The size is 0, should in (0, 64].");
  GE_CHK_BOOL_RET_STATUS((offset < user_data_size_ && (user_data_size_ - offset) >= size),
                          ACL_ERROR_GE_PARAM_INVALID,
                          "The size + offset need <= %zu, but size = %zu, offset = %zu.",
                          user_data_size_, size, offset);
  return SUCCESS;
}

TensorFlowMsg::TensorFlowMsg() : FlowMsgBase() {}

TensorFlowMsg::~TensorFlowMsg() {}

Tensor *TensorFlowMsg::GetTensor() const {
  return tensor_.get();
}

Status TensorFlowMsg::AllocTensor(const TensorDesc &tensor_desc) {
  const GeTensorDesc ge_tensor_desc = TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc);
  int64_t tensor_raw_size = -1;
  const auto ret = TensorUtils::CalcTensorMemSize(ge_tensor_desc.GetShape(),
                                                  ge_tensor_desc.GetFormat(),
                                                  ge_tensor_desc.GetDataType(),
                                                  tensor_raw_size);
  GE_CHK_BOOL_RET_STATUS(ret == ge::GRAPH_SUCCESS, FAILED, "Failed to calc tensor raw size");
  GE_CHK_BOOL_RET_STATUS(tensor_raw_size >= 0, FAILED,
                         "Failed to check tensor size[%" PRId64 "], must >= 0.", tensor_raw_size);
  RuntimeTensorDesc runtime_tensor_desc{};
  GE_CHK_STATUS_RET(DataFlowExecutorUtils::FillRuntimeTensorDesc(ge_tensor_desc, runtime_tensor_desc),
                    "Failed to fill runtime tensor desc");
  size_t flow_msg_size = static_cast<size_t>(tensor_raw_size) + sizeof(runtime_tensor_desc);
  void *msg_buffer = nullptr;
  GE_CHK_STATUS_RET(MbufAlloc(flow_msg_size), "Failed to alloc mbuf, size = %zu", flow_msg_size);
  GE_CHK_STATUS_RET(MbufGetBuffer(msg_buffer), "Failed to get mbuf buffer");
  SetMsgType(MsgType::MSG_TYPE_TENSOR_DATA);
  const auto cpy_ret = memcpy_s(msg_buffer, flow_msg_size, &runtime_tensor_desc, sizeof(runtime_tensor_desc));
  GE_ASSERT_EOK(cpy_ret, "Failed to cpy runtime tensor desc, memcpy_s error, dst size[%zu], src size[%zu], ret[%d].",
                flow_msg_size, sizeof(runtime_tensor_desc), cpy_ret);

  tensor_ = MakeShared<Tensor>(tensor_desc);
  GE_CHECK_NOTNULL(tensor_);

  uint8_t *tensor_data = static_cast<uint8_t *>(msg_buffer) + sizeof(runtime_tensor_desc);
  rtMbufPtr_t tensor_mbuf = MbufCopyRef();
  GE_CHECK_NOTNULL(tensor_mbuf);
  GE_DISMISSABLE_GUARD(tensor_mbuf, ([tensor_mbuf]() { GE_CHK_RT(rtMbufFree(tensor_mbuf)); }));
  const AlignedPtr::Deleter deleter = [tensor_mbuf](const uint8_t *const ptr){
    (void)ptr;
    GE_CHK_RT(rtMbufFree(tensor_mbuf));
  };
  const auto graph_ret = tensor_->SetData(tensor_data, static_cast<size_t>(tensor_raw_size), deleter);
  GE_CHK_BOOL_RET_STATUS(graph_ret == ge::GRAPH_SUCCESS, FAILED, "Failed to set tensor data.");
  GE_DISMISS_GUARD(tensor_mbuf);
  GELOGD("Success to alloc tensor, size[%" PRId64 "].", tensor_raw_size);
  return SUCCESS;
}

Status TensorFlowMsg::UpdateTensorDesc(const RuntimeTensorDesc &runtime_desc,
                                       GeTensorDesc &tensor_desc) {
  auto num_dims = runtime_desc.shape[0];
  auto num_ori_dims = runtime_desc.original_shape[0];
  GE_CHK_BOOL_RET_STATUS(num_dims <= kMaxDimSize,
                         UNSUPPORTED,
                         "shape dim number out of range, num_dims = %" PRId64 ", max = %" PRId64 "",
                         num_dims, kMaxDimSize);
  GE_CHK_BOOL_RET_STATUS(num_ori_dims <= kMaxDimSize,
                         UNSUPPORTED,
                         "original shape dim number out of range, num_dims = %" PRId64 ", max = %" PRId64 "",
                         num_ori_dims, kMaxDimSize);
  GeShape shape(std::vector<int64_t>(&runtime_desc.shape[1], &runtime_desc.shape[1 + num_dims]));
  GeShape ori_shape(std::vector<int64_t>(&runtime_desc.shape[1], &runtime_desc.shape[1 + num_dims]));
  tensor_desc.MutableShape() = std::move(shape);
  tensor_desc.MutableShape() = std::move(ori_shape);
  tensor_desc.SetDataType(static_cast<DataType>(runtime_desc.dtype));
  return SUCCESS;
}

Status TensorFlowMsg::BuildTensor(rtMbufPtr_t mbuf, const GeTensorDesc &output_desc) {
  GE_CHK_STATUS_RET(BuildFlowMsg(mbuf), "Failed to build flow msg from mbuf");
  SetMsgType(MsgType::MSG_TYPE_TENSOR_DATA);
  void *buffer = nullptr;
  GE_CHK_RT_RET(rtMbufGetBuffAddr(mbuf, &buffer));
  uint64_t buffer_size = 0;
  GE_CHK_RT_RET(rtMbufGetBuffSize(mbuf, &buffer_size));
  GE_CHK_BOOL_RET_STATUS(buffer_size >= sizeof(RuntimeTensorDesc), FAILED,
                         "Failed to check buffer size[%" PRIu64 "], tensor buff must > %zu.",
                         buffer_size, sizeof(RuntimeTensorDesc));
  
  RuntimeTensorDesc *const runtime_tensor_desc = PtrToPtr<void, RuntimeTensorDesc>(buffer);
  GeTensorDesc ge_tensor_desc = output_desc;
  GE_CHK_STATUS_RET(UpdateTensorDesc(*runtime_tensor_desc, ge_tensor_desc), "Failed to update output tensor desc");

  int64_t tensor_raw_size = -1;
  GE_CHK_GRAPH_STATUS_RET(TensorUtils::CalcTensorMemSize(ge_tensor_desc.GetShape(),
                                                         ge_tensor_desc.GetFormat(),
                                                         ge_tensor_desc.GetDataType(),
                                                         tensor_raw_size),
                          "Failed to calc tensor mem size");
  size_t tensor_data_size = buffer_size - sizeof(RuntimeTensorDesc);
  GE_CHK_BOOL_RET_STATUS(tensor_raw_size >= 0 && tensor_data_size >= static_cast<size_t>(tensor_raw_size),
                         FAILED,
                         "Failed to check tensor data size[%zu], must >= %" PRId64 ".",
                         tensor_data_size, tensor_raw_size);
  auto tensor_desc = TensorAdapter::GeTensorDesc2TensorDesc(ge_tensor_desc);
  tensor_ = MakeShared<Tensor>(tensor_desc);
  GE_CHECK_NOTNULL(tensor_);

  uint8_t *tensor_data = static_cast<uint8_t *>(buffer) + sizeof(RuntimeTensorDesc);
  rtMbufPtr_t tensor_mbuf = nullptr;
  GE_CHK_RT_RET(rtMbufCopyBufRef(mbuf, &tensor_mbuf));
  GE_DISMISSABLE_GUARD(tensor_mbuf, ([tensor_mbuf]() { GE_CHK_RT(rtMbufFree(tensor_mbuf)); }));
  const AlignedPtr::Deleter deleter = [tensor_mbuf](const uint8_t *const ptr){
    (void)ptr;
    GE_CHK_RT(rtMbufFree(tensor_mbuf));
  };
  const auto graph_ret = tensor_->SetData(tensor_data, static_cast<size_t>(tensor_raw_size), deleter);
  GE_CHK_BOOL_RET_STATUS(graph_ret == ge::GRAPH_SUCCESS, FAILED, "Failed to set tensor data.");
  GE_DISMISS_GUARD(tensor_mbuf);
  GELOGD("Success to build tensor, tensor size[%" PRId64 "], buffer size[%" PRIu64 "].", tensor_raw_size, buffer_size);
  return SUCCESS;
}

RawDataFlowMsg::RawDataFlowMsg() : FlowMsgBase() {}

RawDataFlowMsg::~RawDataFlowMsg() {}

Status RawDataFlowMsg::GetRawData(void *&data_ptr, uint64_t &data_size) const {
  data_ptr = addr_;
  data_size = len_;
  if (data_ptr == nullptr) {
    return FAILED;
  }
  return SUCCESS;
}

Status RawDataFlowMsg::AllocRawData(size_t size) {
  void *msg_buffer = nullptr;
  GE_CHK_STATUS_RET(MbufAlloc(size), "Failed to alloc mbuf, size = %zu", size);
  GE_CHK_STATUS_RET(MbufGetBuffer(msg_buffer), "Failed to get mbuf buffer");
  SetMsgType(MsgType::MSG_TYPE_RAW_MSG);
  addr_ = msg_buffer;
  len_ = size;
  GELOGD("Success to alloc raw data, size[%zu].", size);
  return SUCCESS;
}

Status RawDataFlowMsg::BuildRawData(rtMbufPtr_t mbuf) {
  GE_CHK_STATUS_RET(BuildFlowMsg(mbuf), "Failed to build flow msg from mbuf");
  void *buffer = nullptr;
  GE_CHK_RT_RET(rtMbufGetBuffAddr(mbuf, &buffer));
  uint64_t buffer_size = 0;
  GE_CHK_RT_RET(rtMbufGetBuffSize(mbuf, &buffer_size));
  addr_ = buffer;
  len_ = buffer_size;
  GELOGD("Success to build raw data, size[%" PRIu64 "].", buffer_size);
  return SUCCESS;
}

EmptyDataFlowMsg::EmptyDataFlowMsg() : FlowMsgBase() {}

EmptyDataFlowMsg::~EmptyDataFlowMsg() {}

Tensor *EmptyDataFlowMsg::GetTensor() const {
  return nullptr;
}

Status EmptyDataFlowMsg::AllocEmptyData(MsgType type) {
  (void)type;
  const auto ge_tensor_desc = GeTensorDesc(GeShape({0}));
  RuntimeTensorDesc runtime_tensor_desc{};
  GE_CHK_STATUS_RET(DataFlowExecutorUtils::FillRuntimeTensorDesc(ge_tensor_desc, runtime_tensor_desc),
                    "Failed to fill runtime tensor desc");
  size_t flow_msg_size = sizeof(runtime_tensor_desc);
  void *msg_buffer = nullptr;
  GE_CHK_STATUS_RET(MbufAlloc(flow_msg_size), "Failed to alloc mbuf, size = %zu", flow_msg_size);
  GE_CHK_STATUS_RET(MbufGetBuffer(msg_buffer), "Failed to get mbuf buffer");
  SetMsgType(MsgType::MSG_TYPE_TENSOR_DATA);
  const auto cpy_ret = memcpy_s(msg_buffer, flow_msg_size, &runtime_tensor_desc, sizeof(runtime_tensor_desc));
  GE_ASSERT_EOK(cpy_ret, "Failed to cpy runtime tensor desc, memcpy_s error, dst size[%zu], src size[%zu], ret[%d].",
                flow_msg_size, sizeof(runtime_tensor_desc), cpy_ret);
  SetNullData();
  return SUCCESS;
}

Status EmptyDataFlowMsg::BuildNullData(rtMbufPtr_t mbuf) {
  GE_CHK_STATUS_RET(BuildFlowMsg(mbuf), "Failed to build flow msg from mbuf");
  SetMsgType(MsgType::MSG_TYPE_TENSOR_DATA);
  GELOGD("Success to null data.");
  return SUCCESS;
}

std::shared_ptr<Tensor> FlowBufferFactory::AllocTensor(const std::vector<int64_t> &shape,
                                                       DataType data_type,
                                                       uint32_t align) {
  (void)align;
  auto flow_msg = MakeShared<TensorFlowMsg>();
  if (flow_msg != nullptr) {
    TensorDesc tensor_desc(Shape(shape), FORMAT_ND, data_type);
    auto ret = flow_msg->AllocTensor(tensor_desc);
    if (ret == SUCCESS) {
      return MakeShared<Tensor>(*(flow_msg->GetTensor()));
    }
  }
  return std::shared_ptr<Tensor>(nullptr);
}

FlowMsgPtr FlowBufferFactory::AllocTensorMsg(const std::vector<int64_t> &shape, DataType data_type, uint32_t align) {
  (void)align;
  FlowMsgPtr ret_msg = nullptr;
  GE_DISMISSABLE_GUARD(failed, ([]() { GELOGE(FAILED, "Failed to alloc tensor msg"); }));
  auto flow_msg = MakeShared<TensorFlowMsg>();
  if (flow_msg != nullptr) {
    TensorDesc tensor_desc(Shape(shape), FORMAT_ND, data_type);
    auto ret = flow_msg->AllocTensor(tensor_desc);
    if (ret == SUCCESS) {
      ret_msg = flow_msg;
      GE_DISMISS_GUARD(failed);
    }
  }
  return ret_msg;
}

FlowMsgPtr FlowBufferFactory::AllocRawDataMsg(size_t size, uint32_t align) {
  (void)align;
  FlowMsgPtr ret_msg = nullptr;
  GE_DISMISSABLE_GUARD(failed, ([]() { GELOGE(FAILED, "Failed to alloc raw data msg"); }));
  auto flow_msg = MakeShared<RawDataFlowMsg>();
  if (flow_msg != nullptr) {
    auto ret = flow_msg->AllocRawData(size);
    if (ret == SUCCESS) {
      ret_msg = flow_msg;
      GE_DISMISS_GUARD(failed);
    }
  }
  return ret_msg;
}

FlowMsgPtr FlowBufferFactory::AllocEmptyDataMsg(MsgType type) {
  FlowMsgPtr ret_msg = nullptr;
  GE_DISMISSABLE_GUARD(failed, ([]() { GELOGE(FAILED, "Failed to alloc empty data msg"); }));
  auto flow_msg = MakeShared<EmptyDataFlowMsg>();
  if (flow_msg != nullptr) {
    auto ret = flow_msg->AllocEmptyData(type);
    if (ret == SUCCESS) {
      ret_msg = flow_msg;
      GE_DISMISS_GUARD(failed);
    }
  }
  return ret_msg;
}

FlowMsgPtr FlowBufferFactory::ToFlowMsg(const Tensor &tensor) {
  auto flow_msg = MakeShared<TensorFlowMsg>();
  if (flow_msg != nullptr) {
    auto tensor_desc = tensor.GetTensorDesc();
    auto ret = flow_msg->AllocTensor(tensor_desc);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "Failed to alloc tensor");
      return FlowMsgPtr(nullptr);
    }
    auto data = tensor.GetData();
    auto size = tensor.GetSize();
    auto flow_tensor = flow_msg->GetTensor();
    if (flow_tensor->SetData(data, size) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Failed to set tensor data");
      return FlowMsgPtr(nullptr);
    }
  }
  return flow_msg;
}

FlowMsgPtr FlowBufferFactory::ToFlowMsg(const RawData &raw_data) {
  auto flow_msg = MakeShared<RawDataFlowMsg>();
  if (flow_msg != nullptr) {
    auto ret = flow_msg->AllocRawData(raw_data.len);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "Failed to alloc raw data");
      return FlowMsgPtr(nullptr);
    }
    void *data_ptr = nullptr;
    uint64_t data_size = 0UL;
    ret = flow_msg->GetRawData(data_ptr, data_size);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "Failed to get raw data");
      return FlowMsgPtr(nullptr);
    }
    ret = GeMemcpy(reinterpret_cast<uint8_t *>(data_ptr), data_size,
                   reinterpret_cast<const uint8_t *>(raw_data.addr), raw_data.len);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "Failed to copy raw data to mbuf, memcpy_s error, size[%zu], ret[%d].",
             raw_data.len, ret);
      return FlowMsgPtr(nullptr);
    }
  }
  return flow_msg;
}
}  // namespace ge
