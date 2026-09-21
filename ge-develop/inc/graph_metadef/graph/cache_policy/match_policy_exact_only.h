/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRAPH_CACHE_POLICY_POLICY_MANAGEMENT_MATCH_POLICY_EXACT_ONLY_H_
#define GRAPH_CACHE_POLICY_POLICY_MANAGEMENT_MATCH_POLICY_EXACT_ONLY_H_
#include <memory>
#include "graph/cache_policy/match_policy.h"
#include "graph/cache_policy/policy_register.h"

namespace ge {
class MatchPolicyExactOnly : public MatchPolicy {
public:
  CacheItemId GetCacheItemId(const CCStatType &cc_state, const CacheDescPtr &desc) const override;
  ~MatchPolicyExactOnly() override = default;
};

REGISTER_MATCH_POLICY_CREATOR(MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
                              []() {
                                return std::make_shared<MatchPolicyExactOnly>();
                              });
}  // namespace ge
#endif

