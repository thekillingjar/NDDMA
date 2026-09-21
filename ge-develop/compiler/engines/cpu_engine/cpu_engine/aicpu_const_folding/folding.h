/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GE_GE_LOCAL_ENGINE_ENGINE_AICPU_FOLDING_FOLDING_H_
#define GE_GE_LOCAL_ENGINE_ENGINE_AICPU_FOLDING_FOLDING_H_

#include <cstdint>

#include <map>
#include <string>

#include "graph/utils/op_desc_utils.h"
#include "graph_metadef/register/register.h"

extern "C" {
__attribute__((visibility("default"))) int32_t InitCpuConstantFoldingNew(
    ge::HostCpuOp *(*create_fn)());

__attribute__((visibility("default"))) int32_t CpuConstantFoldingComputeNew(
    const ge::Operator &op, const std::map<std::string, const ge::Tensor> &inputs,
    std::map<std::string, ge::Tensor> outputs);
}
#endif  // GE_GE_LOCAL_ENGINE_ENGINE_AICPU_FOLDING_FOLDING_H_
