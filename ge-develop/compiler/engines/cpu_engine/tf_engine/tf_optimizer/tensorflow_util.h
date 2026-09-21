/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_TENSORFLOW_UTIL_H_
#define AICPU_TENSORFLOW_UTIL_H_

#include <string>
#include "proto/tensorflow/node_def.pb.h"

namespace aicpu {
class TensorFlowUtil {
 public:
  /**
   * Add tf node attr
   * @param attr_name Attr name
   * @param attr_value Attr value
   * @param node_def Node def
   * @return true: success  false: failed
  */
  static bool AddNodeAttr(const std::string &attr_name,
                          const domi::tensorflow::AttrValue &attr_value,
                          domi::tensorflow::NodeDef *node_def);

  /**
   * Find attr from node def
   * @param node_def Node def
   * @param attr_name Attr name
   * @param attr_value Attr value
   * @return true: success  false: failed
  */
  static bool FindAttrValue(const domi::tensorflow::NodeDef *node_def,
                            const std::string attr_name,
                            domi::tensorflow::AttrValue &attr_value);
};
} // namespace aicpu
#endif // AICPU_TENSORFLOW_UTIL_H_

