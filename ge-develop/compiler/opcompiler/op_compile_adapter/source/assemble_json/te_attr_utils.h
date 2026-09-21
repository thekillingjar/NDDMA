/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_ASSEMBLE_JSON_TE_ATTR_UTILS_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_ASSEMBLE_JSON_TE_ATTR_UTILS_H_

#include <nlohmann/json.hpp>
#include "tensor_engine/tbe_attr_value.h"
#include "inc/te_fusion_types.h"

namespace te {
namespace fusion {
void SetAttrDtype2Json(const TbeAttrValue &cattr, nlohmann::json &jsonData);
void GetAttrValueToJson(const TbeAttrValue &attrValue, nlohmann::json &attrValueJson, NullDesc &nullDesc);
void GetAttrValueToJson(const TbeAttrValue &attrValue, nlohmann::json &attrValueJson);
}
}
#endif  // ATC_OPCOMPILER_TE_FUSION_SOURCE_ASSEMBLE_JSON_TE_ATTR_UTILS_H_