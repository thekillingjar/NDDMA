/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_CHECKER_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_CHECKER_H_
#include "value_holder_generator.h"
namespace gert {
namespace bg {
#define CHECK_HOLDERS_ALL_OK(holders, expect_num)                       \
  do {                                                                  \
    if ((holders).size() != (expect_num)) {                               \
      return CreateErrorValueHolders(1, "Failed");                      \
    }                                                                   \
    for (const auto &holder : (holders)) {                              \
      if (holder == nullptr) {                                          \
        return CreateErrorValueHolders(1, "Failed, found null holder"); \
      }                                                                 \
      if (!holder->IsOk()) {                                            \
        return {holder};                                                \
      }                                                                 \
    }                                                                   \
  } while (0)

#define CHECK_HOLDER_OK(holder)                                       \
  do {                                                                \
    const auto &ret = (holder);                                       \
    if (ret == nullptr) {                                             \
      return CreateErrorValueHolders(1, "Failed, found null holder"); \
    }                                                                 \
    if (!ret->IsOk()) {                                               \
      return {ret};                                                   \
    }                                                                 \
  } while (0)

#define CHECK_DEVMEMHOLDER_OK(holder)                                                                     \
  do {                                                                                                    \
    const auto &ret = (holder);                                                                           \
    if (ret == nullptr) {                                                                                 \
      return CreateDevMemErrorValueHolders(0, 1, "Failed, found null holder");                            \
    }                                                                                                     \
    if (!ret->IsOk()) {                                                                                   \
      return CreateDevMemErrorValueHolders(0, 1, "Failed, found invalid holder");                         \
    }                                                                                                     \
  } while (0)
}  // namespace bg
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_CHECKER_H_
