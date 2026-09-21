/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_DUMP_ADUMP_OPINFO_BUILDER_H_
#define GE_COMMON_DUMP_ADUMP_OPINFO_BUILDER_H_

#include <string>
#include <vector>

#include "adump_pub.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
class AdumpOpInfoBuilder {
 public:
  explicit AdumpOpInfoBuilder(const std::string &op_name, const std::string &op_type, bool is_dynamic) {
    info_.opName = op_name;
    info_.opType = op_type;
    info_.agingFlag = is_dynamic;
  }

  ~AdumpOpInfoBuilder() = default;

  AdumpOpInfoBuilder &Task(uint32_t device_id, uint32_t stream_id, uint32_t task_id, uint32_t context_id = UINT32_MAX) {
    info_.deviceId = device_id;
    info_.streamId = stream_id;
    info_.taskId = task_id;
    info_.contextId = context_id;
    return *this;
  }

  AdumpOpInfoBuilder &TersorInfo(std::vector<Adx::TensorInfoV2> &infos) {
    for (auto &iter : infos) {
      (void)info_.tensorInfos.emplace_back(iter);
    }
    return *this;
  }

  AdumpOpInfoBuilder &AdditionInfo(const std::string &key, const std::string &value) {
    info_.additionalInfo[key] = value;
    return *this;
  }

  AdumpOpInfoBuilder &DeviceInfo(const std::string &name, void *addr, uint64_t length) {
    if ((addr != nullptr) && (length > 0U)) {
      Adx::DeviceInfo device_info;
      device_info.name = name;
      device_info.addr = addr;
      device_info.length = length;
      (void)info_.deviceInfos.emplace_back(device_info);
    }
    return *this;
  }

  const Adx::OperatorInfoV2 &Build() const {
    return info_;
  }

 private:
  Adx::OperatorInfoV2 info_{};
};
}  // namespace ge
#endif