/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_SYMBOLIC_SYMBOLIC_KERNEL_SHAPE_ENV_GUARDER_H_
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_SYMBOLIC_SYMBOLIC_KERNEL_SHAPE_ENV_GUARDER_H_
#include "graph/attribute_group/attr_group_shape_env.h"

namespace ge {
class ShapeEnvGuarder {
 public:
  explicit ShapeEnvGuarder(ShapeEnvAttr *shape_env_context);
  ~ShapeEnvGuarder();
  ShapeEnvGuarder(const ShapeEnvGuarder &) = delete;
  ShapeEnvGuarder(const ShapeEnvGuarder &&) = delete;
  ShapeEnvGuarder &operator=(const ShapeEnvGuarder &) = delete;
  ShapeEnvGuarder &&operator=(const ShapeEnvGuarder &&) = delete;
 private:
  ShapeEnvAttr *origin_context_;
};
}
#endif // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_SYMBOLIC_SYMBOLIC_KERNEL_SHAPE_ENV_GUARDER_H_
