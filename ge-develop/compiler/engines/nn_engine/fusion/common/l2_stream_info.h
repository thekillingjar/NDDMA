/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_COMMON_L2_STREAM_INFO_H_
#define FUSION_ENGINE_FUSION_COMMON_L2_STREAM_INFO_H_

#include <map>
#include <string>
#include <mutex>
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"
#include "runtime/base.h"
#include "common/l2fusion_struct.h"

namespace fe {
class StreamL2Info {
 public:
  StreamL2Info(const StreamL2Info &) = delete;
  StreamL2Info &operator=(const StreamL2Info &) = delete;
  static StreamL2Info& Instance();

  Status GetStreamL2Info(const int64_t &stream_id, const std::string &node_name, TaskL2Info *&l2_data,
                         const std::string &batch_label);
  Status SetStreamL2Info(const int64_t &stream_id, TaskL2InfoMap &l2_alloc_res, const std::string &batch_label);

 private:
  StreamL2Info();
  ~StreamL2Info();
  mutable std::mutex stream_l2_mutex_;
  std::map<std::string, TaskL2InfoMap> stream_l2_map_;
};
}  // namespace fe

#endif  // FUSION_ENGINE_FUSION_COMMON_L2_STREAM_INFO_H_
