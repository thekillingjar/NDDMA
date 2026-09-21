/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARSER_COMMON_PASS_H_
#define PARSER_COMMON_PASS_H_

#include <memory>

namespace ge {
///
/// @ingroup domi_omg
/// @brief pass
/// @author
///
template <typename T>
class Pass {
 public:
  virtual ~Pass() {}

  /// run pass
  /// @author
  virtual Status Run(std::shared_ptr<T>) = 0;
};
}  // namespace ge

#endif  // PARSER_COMMON_PASS_H_
