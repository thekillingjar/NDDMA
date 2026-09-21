/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_UNKNOWN_SHAPE_UTILS_H
#define INC_REGISTER_GRAPH_OPTIMIZER_UNKNOWN_SHAPE_UTILS_H
#include "graph/utils/graph_utils.h"
namespace fe {
class UnknownShapeUtils {
public:
  /*
   *  @ingroup fe
   *  @brief   check whether the node is unknown shape.
   *  @param   [in]  input or output tensor.
   *  @return  true: unknown; false: known
   */
  static bool IsUnknownShapeOp(const ge::OpDesc &op_desc);

  /*
   *  @ingroup fe
   *  @brief   check whether the input or output shape contains -2.
   *  @param   op_desc input or output desc.
   *  @return  true: contains; false: not contains
   */
  static bool IsContainUnknownDimNum(const ge::OpDesc &op_desc);

  /*
   *  @brief   check whether the value is -1 or -2
   *  @param   input or ourput shape dim
   *  @return  true: contains; false: not contains
   */
  static bool IsUnknownShapeValue(const int64_t &value);
private:
  static bool IsUnKnownShapeTensor(const ge::OpDesc &op_desc);
};
}  // namespace fe

#endif
