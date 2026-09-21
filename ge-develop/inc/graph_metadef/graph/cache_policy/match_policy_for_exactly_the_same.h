/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_GRAPH_CACHE_POLICY_MATCH_POLICY_FOR_EXACTLY_THE_SAME_H
#define METADEF_CXX_GRAPH_CACHE_POLICY_MATCH_POLICY_FOR_EXACTLY_THE_SAME_H
#include "graph/cache_policy/match_policy.h"
#include "graph/cache_policy/policy_register.h"

namespace ge {
class MatchPolicyForExactlyTheSame : public MatchPolicy {
 public:
  MatchPolicyForExactlyTheSame() = default;
  ~MatchPolicyForExactlyTheSame() override = default;

  CacheItemId GetCacheItemId(const CCStatType &cc_state, const CacheDescPtr &cache_desc) const override;
};
REGISTER_MATCH_POLICY_CREATOR(MatchPolicyType::MATCH_POLICY_FOR_EXACTLY_THE_SAME,
                              []() { return std::make_shared<MatchPolicyForExactlyTheSame>(); });
}  // namespace ge
#endif  // METADEF_CXX_GRAPH_CACHE_POLICY_MATCH_POLICY_FOR_EXACTLY_THE_SAME_H
