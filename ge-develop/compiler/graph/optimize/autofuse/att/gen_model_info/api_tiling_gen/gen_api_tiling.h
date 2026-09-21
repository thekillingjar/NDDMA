/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEN_API_TILING_H_
#define GEN_API_TILING_H_

#include <string>
#include <map>
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "base/base_types.h"
#include "gen_autofuse_api_tiling.h"

namespace att {
struct ApiTilingParams {
  ge::AscGraph graph;
  std::string tiling_data_type;
};

ge::Status GetApiTilingInfo(const uint32_t tiling_case_id, const ApiTilingParams &params,
                            std::map<std::string, NodeApiTilingCode> &node_name_to_api_code);
}  // namespace att
#endif  // GEN_API_TILING_H_
