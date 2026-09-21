/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_VALUE_HOLDER_UTILS_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_VALUE_HOLDER_UTILS_H_

#include "value_holder.h"
#include "graph/op_desc.h"

namespace gert {
namespace bg {
class ValueHolderUtils {
public:
  static bool IsNodeValid(const ValueHolderPtr &holder);

  static bool IsNodeEqual(const ValueHolderPtr &src, const ValueHolderPtr &dst);

  static std::string GetNodeName(const ValueHolderPtr &holder);
  static const char *GetNodeNameBarePtr(const ValueHolderPtr &holder);

  static std::string GetNodeType(const ValueHolderPtr &holder);
  static const char *GetNodeTypeBarePtr(const ValueHolderPtr &holder);

  static ge::OpDescPtr GetNodeOpDesc(const ValueHolderPtr &holder);
  static ge::OpDesc *GetNodeOpDescBarePtr(const ValueHolderPtr &holder);

  static bool IsDirectlyControlled(const bg::ValueHolderPtr &src, const bg::ValueHolderPtr &dst);
};
} // bg
} // gert

#endif  // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_VALUE_HOLDER_UTILS_H_
