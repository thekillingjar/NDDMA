/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DVPP_ENGINE_UTIL_DVPP_DEFINE_H_
#define DVPP_ENGINE_UTIL_DVPP_DEFINE_H_

namespace dvpp {
// 用于简化异常分支代码 降低圈复杂度 降低llt成本
#define DVPP_CHECK_IF_THEN_DO(check_expr, do_expr) \
  {                                                \
    if (check_expr) {                              \
      do_expr;                                     \
    }                                              \
  }

// 用于简化try-catch场景 降低圈复杂度 降低llt成本
#define DVPP_TRY_CATCH(try_expr, catch_expr) \
  try {                                      \
    try_expr;                                \
  } catch (...) {                            \
    catch_expr;                              \
  }
} // namespace dvpp

#endif // DVPP_ENGINE_UTIL_DVPP_DEFINE_H_
