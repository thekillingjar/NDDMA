/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_GRAPH_SERIALIZATION_STRING_SERIALIZER_H_
#define METADEF_GRAPH_SERIALIZATION_STRING_SERIALIZER_H_

#include "attr_serializer.h"
#include "attr_serializer_registry.h"
namespace ge {
class StringSerializer : public GeIrAttrSerializer {
 public:
  StringSerializer() = default;
  graphStatus Serialize(const AnyValue &av, proto::AttrDef &def) override;
  graphStatus Deserialize(const proto::AttrDef &def, AnyValue &av) override;
};
}  // namespace ge

#endif // METADEF_GRAPH_SERIALIZATION_STRING_SERIALIZER_H_
