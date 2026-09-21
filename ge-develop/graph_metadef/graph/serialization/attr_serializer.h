/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_ATTR_SERIALIZER_H
#define METADEF_CXX_ATTR_SERIALIZER_H

#include "proto/ge_ir.pb.h"

#include "graph/any_value.h"

namespace ge {
/**
 * 所有的serializer都应该是无状态的、可并发调用的，全局仅构造一份，后续多线程并发调用
 */
class GeIrAttrSerializer {
 public:
  virtual graphStatus Serialize(const AnyValue &av, proto::AttrDef &def) = 0;
  virtual graphStatus Deserialize(const proto::AttrDef &def, AnyValue &av) = 0;
  virtual ~GeIrAttrSerializer() = default;
  GeIrAttrSerializer() = default;
  GeIrAttrSerializer(const GeIrAttrSerializer &) = delete;
  GeIrAttrSerializer &operator=(const GeIrAttrSerializer &) = delete;
  GeIrAttrSerializer(GeIrAttrSerializer &&) = delete;
  GeIrAttrSerializer &operator=(GeIrAttrSerializer &&) = delete;
};
}  // namespace ge

#endif  // METADEF_CXX_ATTR_SERIALIZER_H
