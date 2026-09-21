/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_COMMON_FUSION_STATISTIC_BUFFER_FUSION_INFO_COLLECTER_H_
#define FUSION_ENGINE_FUSION_COMMON_FUSION_STATISTIC_BUFFER_FUSION_INFO_COLLECTER_H_

#include <set>
#include "common/aicore_util_attr_define.h"
#include "common/fe_inner_error_codes.h"
#include "common/fe_log.h"
#include "common/graph_comm.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"

namespace fe {

using PassNameIdMap = std::map<std::string, std::set<int64_t>>;
using PassNameIdPair = std::pair<std::string, std::set<int64_t>>;

class BufferFusionInfoCollecter {
 public:
  explicit BufferFusionInfoCollecter();

  virtual ~BufferFusionInfoCollecter();

  BufferFusionInfoCollecter(const BufferFusionInfoCollecter &) = delete;

  BufferFusionInfoCollecter &operator=(const BufferFusionInfoCollecter &) = delete;

  Status CountBufferFusionTimes(ge::ComputeGraph &graph) const;

 private:
  Status GetPassNameOfScopeId(ge::ComputeGraph &graph, PassNameIdMap &pass_name_scope_id_map) const;

  Status GetPassNameOfFailedId(ge::ComputeGraph &graph, PassNameIdMap &pass_name_fusion_failed_id_map) const;

  void SetPassName(const ge::ComputeGraph &graph, std::set<std::string> &pass_name_set) const;
};
}

#endif  // FUSION_ENGINE_FUSION_COMMON_FUSION_STATISTIC_BUFFER_FUSION_INFO_COLLECTER_H_
