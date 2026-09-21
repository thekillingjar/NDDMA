/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/cache_policy/policy_register.h"
namespace ge {
PolicyRegister &PolicyRegister::GetInstance() {
  static PolicyRegister instance;
  return instance;
}

MatchPolicyRegister::MatchPolicyRegister(const MatchPolicyType match_policy_type, const MatchPolicyCreator &creator) {
  PolicyRegister::GetInstance().RegisterMatchPolicy(match_policy_type, creator);
}

AgingPolicyRegister::AgingPolicyRegister(const AgingPolicyType aging_policy_type, const AgingPolicyCreator &creator) {
  PolicyRegister::GetInstance().RegisterAgingPolicy(aging_policy_type, creator);
}
}  // namespace ge
