/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DVPP_ENGINE_COMMON_DVPP_OPS_CHECKER_H_
#define DVPP_ENGINE_COMMON_DVPP_OPS_CHECKER_H_

#include "common/dvpp_ops.h"
#include "graph/node.h"

namespace dvpp {
class DvppOpsChecker {
 public:
  /**
   * @brief check if dvpp support the op
   * @param node node pointer
   * @param unsupported_reason if not support, need to write reason
   * @return whether support the op
   */
  static bool CheckSupported(
      const ge::NodePtr& node, std::string& unsupported_reason);

  /**
   * @brief check if Sub and Mul can fused into NormalizeV2
   * @param mul_node node pointer
   * @return whether support fused
   */
  static bool CheckSubAndMulFusedIntoNormalizeV2(const ge::NodePtr& mul_node);

 private:
  /**
   * @brief check if dvpp support the op parameter
   * @param dvpp_op_info op info in dvpp ops lib
   * @param node node pointer
   * @param unsupported_reason if not support, need to write reason
   * @return whether support the op
   */
  static bool CheckParameterSupported(const DvppOpInfo& dvpp_op_info,
      const ge::NodePtr& node, std::string& unsupported_reason);

  /**
   * @brief check if dvpp support the op input
   * @param dvpp_op_info op info in dvpp ops lib
   * @param op_desc_ptr OpDesc pointer
   * @param unsupported_reason if not support, need to write reason
   * @return whether support the op
   */
  static bool CheckInputSupported(const DvppOpInfo& dvpp_op_info,
      const ge::OpDescPtr& op_desc_ptr, std::string& unsupported_reason);

  /**
   * @brief check if dvpp support the op output
   * @param dvpp_op_info op info in dvpp ops lib
   * @param op_desc_ptr OpDesc pointer
   * @param unsupported_reason if not support, need to write reason
   * @return whether support the op
   */
  static bool CheckOutputSupported(const DvppOpInfo& dvpp_op_info,
      const ge::OpDescPtr& op_desc_ptr, std::string& unsupported_reason);

  /**
   * @brief check if dvpp support the op attr
   * @param dvpp_op_info op info in dvpp ops lib
   * @param op_desc_ptr OpDesc pointer
   * @param unsupported_reason if not support, need to write reason
   * @return whether support the op
   */
  static bool CheckAttrSupported(const DvppOpInfo& dvpp_op_info,
      const ge::OpDescPtr& op_desc_ptr, std::string& unsupported_reason);

  /**
   * @brief check if c-axix in the same position and value is 3
   * @param node node pointer
   * @return whether c-axis is ok
   */
  static bool CheckAxisC(const ge::NodePtr& node);
}; // class DvppOpsChecker
} // namespace dvpp

#endif // DVPP_ENGINE_COMMON_DVPP_OPS_CHECKER_H_
