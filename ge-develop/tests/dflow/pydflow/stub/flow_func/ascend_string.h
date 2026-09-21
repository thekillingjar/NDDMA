/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_ASCEND_STRING_H
#define FLOW_FUNC_ASCEND_STRING_H

#include <string>
#include <memory>
#include "flow_func_defines.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY AscendString {
 public:
  AscendString() = default;

  ~AscendString() = default;

  AscendString(const char *const name);

  AscendString(const char *const name, size_t length);

  /**
   * @brief get string info.
   * @return string
   */
  const char *GetString() const;

  size_t GetLength() const;

  /**
   * @brief check if current string is less than the target string.
   * @param the target string to be compared
   * @return true if src string is less than the target string, otherwise return false.
   */
  bool operator<(const AscendString &d) const;

  /**
   * @brief check if current string is larger than the target string.
   * @param the target string to be compared
   * @return true if src string is larger than the target string, otherwise return false.
   */
  bool operator>(const AscendString &d) const;

  /**
   * @brief check if current string is less than or equal to the target string.
   * @param the target string to be compared
   * @return true if src string is less than or equal to target string, otherwise return false.
   */
  bool operator<=(const AscendString &d) const;

  /**
   * @brief check if current string is larger than or equal to the target string.
   * @param the target string to be compared
   * @return true if src string is larger than or equal to the target string, otherwise return false.
   */
  bool operator>=(const AscendString &d) const;

  /**
   * @brief check if current string is equal to the target string.
   * @param the target string to be compared
   * @return true if src string is equal to the target string, otherwise return false.
   */
  bool operator==(const AscendString &d) const;

  /**
   * @brief check if current string is not equal to the target string.
   * @param the target string to be compared
   * @return true if src string is not equal to the target string, otherwise return false.
   */
  bool operator!=(const AscendString &d) const;

 private:
  std::shared_ptr<std::string> name_;
};
}  // namespace FlowFunc

#endif  // FLOW_FUNC_ASCEND_STRING_H