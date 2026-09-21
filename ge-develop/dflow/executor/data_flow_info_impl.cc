/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge/ge_data_flow_api.h"
#include "common/checker.h"
#include "common/util/mem_utils.h"
#include "framework/common/debug/ge_log.h"
#include "securec.h"
#include "data_flow_info_utils.h"

namespace ge {
class DataFlowInfoImpl {
 public:
  DataFlowInfoImpl() = default;
  ~DataFlowInfoImpl() = default;

  void SetStartTime(const uint64_t start_time) {
    start_time_ = start_time;
  }
  uint64_t GetStartTime() const {
    return start_time_;
  }

  void SetEndTime(const uint64_t end_time) {
    end_time_ = end_time;
  }
  uint64_t GetEndTime() const {
    return end_time_;
  }

  void SetFlowFlags(const uint32_t flags) {
    flow_flags_ = flags;
  }
  uint32_t GetFlowFlags() const {
    return flow_flags_;
  }

  Status SetUserData(const void *data, size_t size, size_t offset) {
    if (CheckParamsForUserData(data, size, offset) != SUCCESS) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Failed to set user data, the params is invalid.");
      return ACL_ERROR_GE_PARAM_INVALID;
    }
    const auto cpy_ret = memcpy_s((user_data_ + offset), (kMaxUserDataSize - offset), data, size);
    GE_ASSERT_EOK(cpy_ret, "Failed to set user data, memcpy_s error, size[%zu], offset[%zu], ret[%d].", size, offset,
                  cpy_ret);
    GELOGD("Success to set user data, size[%zu], offset[%zu].", size, offset);
    return SUCCESS;
  }

  Status GetUserData(void *data, size_t size, size_t offset) const {
    if (CheckParamsForUserData(data, size, offset) != SUCCESS) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Failed to get user data, the params is invalid.");
      return ACL_ERROR_GE_PARAM_INVALID;
    }
    const auto cpy_ret = memcpy_s(data, size, (user_data_ + offset), size);
    GE_ASSERT_EOK(cpy_ret, "Failed to get user data, memcpy_s error, size[%zu], offset[%zu], ret[%d].", size, offset,
                  cpy_ret);
    GELOGD("Success to get user data, size[%zu], offset[%zu].", size, offset);
    return SUCCESS;
  }

  uint64_t GetTransactionId() const {
    return transaction_id_;
  }

  void SetTransactionId(uint64_t transaction_id) {
    transaction_id_ = transaction_id;
    has_custom_transaction_id_ = transaction_id_ > 0;
  }

  bool HasCustomTransactionId() const {
    return has_custom_transaction_id_ && (transaction_id_ > 0);
  }

 private:
  friend DataFlowInfoUtils;
  uint64_t start_time_ = 0UL;
  uint64_t end_time_ = 0UL;
  uint64_t transaction_id_ = 0UL;
  uint32_t flow_flags_ = 0U;
  bool has_custom_transaction_id_ = false;
  int8_t user_data_[kMaxUserDataSize] = {};

  Status CheckParamsForUserData(const void *data, size_t size, size_t offset) const {
    if (data == nullptr) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "The data is nullptr.");
      return ACL_ERROR_GE_PARAM_INVALID;
    }
    if (size == 0U) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "The size is 0, should in (0, 64].");
      return ACL_ERROR_GE_PARAM_INVALID;
    }
    if ((offset >= kMaxUserDataSize) || (kMaxUserDataSize - offset) < size) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "The size + offset need <= %zu, but size = %zu, offset = %zu.",
             kMaxUserDataSize, size, offset);
      return ACL_ERROR_GE_PARAM_INVALID;
    }
    return SUCCESS;
  }
};

DataFlowInfo::DataFlowInfo() {
  impl_ = MakeShared<DataFlowInfoImpl>();
}
DataFlowInfo::~DataFlowInfo() {}
void DataFlowInfo::SetStartTime(const uint64_t start_time) {
  if (impl_ != nullptr) {
    impl_->SetStartTime(start_time);
  }
}
uint64_t DataFlowInfo::GetStartTime() const {
  if (impl_ != nullptr) {
    return impl_->GetStartTime();
  }
  return 0UL;
}

void DataFlowInfo::SetEndTime(const uint64_t end_time) {
  if (impl_ != nullptr) {
    impl_->SetEndTime(end_time);
  }
}
uint64_t DataFlowInfo::GetEndTime() const {
  if (impl_ != nullptr) {
    return impl_->GetEndTime();
  }
  return 0UL;
}

void DataFlowInfo::SetFlowFlags(const uint32_t flow_flags) {
  if (impl_ != nullptr) {
    impl_->SetFlowFlags(flow_flags);
  }
}
uint32_t DataFlowInfo::GetFlowFlags() const {
  if (impl_ != nullptr) {
    return impl_->GetFlowFlags();
  }
  return 0U;
}

Status DataFlowInfo::SetUserData(const void *data, size_t size, size_t offset) {
  if (impl_ != nullptr) {
    return impl_->SetUserData(data, size, offset);
  }
  return ACL_ERROR_GE_INTERNAL_ERROR;
}

Status DataFlowInfo::GetUserData(void *data, size_t size, size_t offset) const {
  if (impl_ != nullptr) {
    return impl_->GetUserData(data, size, offset);
  }
  return ACL_ERROR_GE_INTERNAL_ERROR;
}

uint64_t DataFlowInfo::GetTransactionId() const {
  if (impl_ != nullptr) {
    return impl_->GetTransactionId();
  }
  return 0UL;
}

void DataFlowInfo::SetTransactionId(uint64_t transaction_id) {
  if (impl_ != nullptr) {
    impl_->SetTransactionId(transaction_id);
  }
}

bool DataFlowInfoUtils::HasCustomTransactionId(const DataFlowInfo &info) {
  return (info.impl_ != nullptr) && info.impl_->HasCustomTransactionId();
}

void DataFlowInfoUtils::InitMsgInfoByDataFlowInfo(ExchangeService::MsgInfo &msg_info, const DataFlowInfo &info,
                                                    bool contains_n_mapping_node) {
  msg_info.start_time = info.GetStartTime();
  msg_info.end_time = info.GetEndTime();
  msg_info.flags = info.GetFlowFlags();
  uint64_t user_assign_trans_id = info.GetTransactionId();
  if ((user_assign_trans_id > 0) && DataFlowInfoUtils::HasCustomTransactionId(info)) {
    if (contains_n_mapping_node) {
      msg_info.trans_id = user_assign_trans_id;
      msg_info.data_flag |= kCustomTransIdFlagBit;
    } else {
      GELOGW("cannot assign transaction id=%" PRIu64 " as no contains_n-mapping node or exception_catch not set, "
        "ignore it", user_assign_trans_id);
    }
  }
}

void DataFlowInfoUtils::InitDataFlowInfoByMsgInfo(DataFlowInfo &info, const ExchangeService::MsgInfo &msg_info) {
  info.SetStartTime(msg_info.start_time);
  info.SetEndTime(msg_info.end_time);
  info.SetFlowFlags(msg_info.flags);
  info.SetTransactionId(msg_info.trans_id);
  if (info.impl_ != nullptr) {
    bool has_custom_trans_id = (msg_info.data_flag & kCustomTransIdFlagBit) != 0;
    info.impl_->has_custom_transaction_id_ = has_custom_trans_id;
  }
}

}  // namespace ge
