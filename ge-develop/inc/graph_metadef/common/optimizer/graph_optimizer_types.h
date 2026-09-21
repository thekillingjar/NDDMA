/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_COMMON_OPTIMIZER_GRAPH_OPTIMIZER_TYPES_H_
#define INC_COMMON_OPTIMIZER_GRAPH_OPTIMIZER_TYPES_H_

#include <string>

namespace ge {
enum OPTIMIZER_SCOPE {
  UNIT = 0,
  ENGINE,
};

struct GraphOptimizerAttribute {
  std::string engineName;
  OPTIMIZER_SCOPE scope;
};
}  // namespace ge

#endif  // INC_COMMON_OPTIMIZER_GRAPH_OPTIMIZER_TYPES_H_
