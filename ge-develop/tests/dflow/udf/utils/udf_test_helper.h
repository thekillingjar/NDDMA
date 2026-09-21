/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_UDF_TEST_HELPER_H
#define ASCEND_UDF_TEST_HELPER_H
#include <vector>
#include "common/common_define.h"

namespace FlowFunc {
class UdfTestHelper {
 public:
  static QueueDevInfo CreateQueueDevInfo(uint32_t qid, int32_t device_type = Common::kDeviceTypeCpu,
                                         int32_t device_id = 0, bool is_proxy_queue = false, uint32_t logic_qid = 0U) {
    return QueueDevInfo{qid, device_type, device_id, device_id, logic_qid, is_proxy_queue};
  }

  static std::vector<QueueDevInfo> CreateQueueDevInfos(const std::vector<uint32_t> &qids,
                                                       int32_t device_type = Common::kDeviceTypeCpu,
                                                       int32_t device_id = 0, bool is_proxy_queue = false,
                                                       const std::vector<uint32_t> &logic_qids = {}) {
    std::vector<QueueDevInfo> queue_dev_infos;
    queue_dev_infos.reserve(qids.size());
    int32_t index = 0;
    for (const uint32_t qid : qids) {
      if (logic_qids.empty()) {
        queue_dev_infos.emplace_back(CreateQueueDevInfo(qid, device_type, device_id, is_proxy_queue));
      } else {
        queue_dev_infos.emplace_back(
            CreateQueueDevInfo(qid, device_type, device_id, is_proxy_queue, logic_qids[index]));
      }
      index++;
    }
    return queue_dev_infos;
  }
};
}  // namespace FlowFunc
#endif  // ASCEND_UDF_TEST_HELPER_H
