/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_UTILS_ATTR_UTILS_H_
#define ACL_UTILS_ATTR_UTILS_H_

#include <string>
#include <sstream>
#include <map>

#include "acl/acl_base.h"
#include "graph/op_desc.h"
#include "utils/string_utils.h"
#include "types/op_attr.h"
#include "types/acl_op_inner.h"
#include "types/op_model.h"

namespace acl {
namespace attr_utils {
ACL_FUNC_VISIBILITY std::string GeAttrValueToString(const ge::GeAttrValue &val);

ACL_FUNC_VISIBILITY std::string AttrMapToString(const std::map<std::string, ge::GeAttrValue> &attrMap);

ACL_FUNC_VISIBILITY size_t AttrMapToDigest(const std::map<std::string, ge::GeAttrValue> &attrMap);

ACL_FUNC_VISIBILITY bool AttrValueEquals(const ge::GeAttrValue &lhs, const ge::GeAttrValue &rhs);

ACL_FUNC_VISIBILITY bool OpAttrEquals(const aclopAttr *const lhs, const aclopAttr *const rhs);

ACL_FUNC_VISIBILITY uint64_t GetCurrentTimestamp();

bool ValueRangeCheck(const std::map<AttrRangeType, ge::GeAttrValue> &valueRange,
                     const aclDataBuffer *const dataBuffer, const aclDataType dataType);

bool IsSameValueRange(const std::map<AttrRangeType, ge::GeAttrValue> &valueRange1,
                      const std::map<AttrRangeType, ge::GeAttrValue> &valueRange2);

bool SaveConstToAttr(OpModelDef& modelDef);

bool SaveConstToAttr(const AclOp &opDesc, aclopAttr *const opAttr);

bool IsListFloatEquals(const std::vector<float32_t> &lhsValue, const std::vector<float32_t> &rhsValue);
} // namespace attr_utils
} // acl

#endif // ACL_UTILS_ATTR_UTILS_H_
