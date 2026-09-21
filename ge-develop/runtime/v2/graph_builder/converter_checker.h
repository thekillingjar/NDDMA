/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_NODE_CONVERTER_CONVERTER_CHECKER_H_
#define AIR_CXX_RUNTIME_V2_NODE_CONVERTER_CONVERTER_CHECKER_H_
#include <vector>
#include <algorithm>
#include "register/node_converter_registry.h"
#include "register/ffts_node_converter_registry.h"
#include "common/debug/ge_log.h"
#include "common/checker.h"
#include "exe_graph/lowering/dev_mem_value_holder.h"
#include "base/err_msg.h"
#include "common/hyper_status.h"
#include "exe_graph/lowering/value_holder.h"

namespace gert {
LowerResult CreateErrorLowerResult(const char *error_msg, ...);
bool IsValidHolder(const bg::ValueHolderPtr &holder);
bool IsValidHolder(const std::vector<bg::ValueHolderPtr> &holders);
bool IsValidHolder(const bg::DevMemValueHolderPtr &holder);
bool IsValidHolder(const std::vector<bg::DevMemValueHolderPtr> &holders);
HyperStatus CheckLowerInput(const LowerInput &lower_input);
HyperStatus CheckFFTSLowerInput(const FFTSLowerInput &lower_input);
HyperStatus CheckFFTSStaLowerInput(const FFTSLowerInput &lower_input);
#define CONVERTER_CHECK_HOLDERS_ALL_OK(holders, expect_num)         \
  do {                                                              \
    if ((holders).size() != (expect_num)) {                         \
      return CreateErrorLowerResult("Failed");                      \
    }                                                               \
    for (const auto &holder : (holders)) {                          \
      if (holder == nullptr) {                                      \
        return CreateErrorLowerResult("Failed, found null holder"); \
      }                                                             \
      if (!holder->IsOk()) {                                        \
        return CreateErrorLowerResult("Failed");                    \
      }                                                             \
    }                                                               \
  } while (0)

#define LOWER_REQUIRE(exp, ...)                                  \
  do {                                                           \
    if (!(exp)) {                                                \
      auto msg = CreateErrorMsg(__VA_ARGS__);                    \
      if (msg.empty()) {                                         \
        REPORT_INNER_ERR_MSG("E19999", "Assert %s failed", #exp);  \
        GELOGE(ge::FAILED, "Assert %s failed", #exp);            \
        return CreateErrorLowerResult("Assert %s failed", #exp); \
      }                                                          \
      REPORT_INNER_ERR_MSG("E19999", "%s", msg.data());            \
      GELOGE(ge::FAILED, "%s", msg.data());                      \
      return CreateErrorLowerResult(msg.data());                 \
    }                                                            \
  } while (false)

#define LOWER_REQUIRE_NOTNULL(v, ...) LOWER_REQUIRE(((v) != nullptr), __VA_ARGS__)
#define LOWER_REQUIRE_SUCCESS(v, ...) LOWER_REQUIRE(((v) == ge::SUCCESS), __VA_ARGS__)
#define LOWER_REQUIRE_GRAPH_SUCCESS(v, ...) LOWER_REQUIRE(((v) == ge::GRAPH_SUCCESS), __VA_ARGS__)
#define LOWER_REQUIRE_VALID_HOLDER(v, ...) LOWER_REQUIRE((IsValidHolder(v)), __VA_ARGS__)
#define LOWER_REQUIRE_HYPER_SUCCESS(v) \
  do {                                 \
    auto status = (v);                 \
    if (!(status).IsSuccess()) {       \
      return {status, {}, {}, {}};     \
    }                                  \
  } while (false)

#define LOWER_RETURN_IF_ERROR(ctx)    \
  do {                                \
    auto status = (ctx);              \
    if (!status.result.IsSuccess()) { \
      return status;                  \
    }                                 \
  } while (false)

#define RET_ERR_RET_IF(ctx, msg)        \
  do {                                \
    if (ctx) {                        \
      return {HyperStatus::ErrorStatus(msg), {}, {}, {}};    \
    }                                 \
  } while (false)
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_NODE_CONVERTER_CONVERTER_CHECKER_H_
