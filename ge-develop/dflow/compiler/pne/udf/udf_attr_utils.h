/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDF_UDF_ATTR_UTILS_H
#define UDF_UDF_ATTR_UTILS_H

#include "framework/common/ge_inner_error_codes.h"
#include "graph/any_value.h"
#include "proto/udf_attr.pb.h"

namespace ge {
class UdfAttrUtils {
 public:
  static Status SetStringAttr(const AnyValue &value, udf::AttrValue &attr);

  static Status SetIntAttr(const AnyValue &value, udf::AttrValue &attr);

  static Status SetFloatAttr(const AnyValue &value, udf::AttrValue &attr);

  static Status SetBoolAttr(const AnyValue &value, udf::AttrValue &attr);

  static Status SetDataTypeAttr(const AnyValue &value, udf::AttrValue &attr);

  static Status SetListStringAttr(const AnyValue &value, udf::AttrValue &attr);

  static Status SetListIntAttr(const AnyValue &value, udf::AttrValue &attr);

  static Status SetListFloatAttr(const AnyValue &value, udf::AttrValue &attr);

  static Status SetListBoolAttr(const AnyValue &value, udf::AttrValue &attr);

  static Status SetListDataTypeAttr(const AnyValue &value, udf::AttrValue &attr);

  static Status SetListListIntAttr(const AnyValue &value, udf::AttrValue &attr);

  using SetAttrFunc = Status (*)(const AnyValue &value, udf::AttrValue &attr);

  static const std::map<AnyValue::ValueType, SetAttrFunc> set_attr_funcs_;
};
}  // namespace ge

#endif  // UDF_UDF_ATTR_UTILS_H
