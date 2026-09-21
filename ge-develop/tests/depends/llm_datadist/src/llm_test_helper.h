/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_TESTS_DEPENDS_LLM_ENGINE_SRC_LLM_TEST_HELPER_H_
#define AIR_TESTS_DEPENDS_LLM_ENGINE_SRC_LLM_TEST_HELPER_H_

#include <vector>
#include <cstdlib>
#include <iostream>
#include <regex>
#include "mmpa/mmpa_api.h"
#include "llm_datadist/llm_engine_types.h"
#include "common/llm_inner_types.h"
#include "llm_datadist/llm_datadist.h"
#include "graph/tensor.h"

namespace llm {
const std::string PNE_ID_NPU = "NPU";
// deleted file dflow/llm_datadist/v1/priority_schedule/priority_schedule_manager.h, which defined RoleType
enum class RoleType : uint32_t { PROMPT = 0, DECODER, ROLE_TYPE_END };

class LLMTestUtils {
 public:
  static void BuildInput(ge::Tensor &input_tensor, const std::vector<int64_t> &shape, const size_t input_size);
};
} // namespace llm

#endif  // AIR_TESTS_DEPENDS_LLM_ENGINE_SRC_LLM_TEST_HELPER_H_
