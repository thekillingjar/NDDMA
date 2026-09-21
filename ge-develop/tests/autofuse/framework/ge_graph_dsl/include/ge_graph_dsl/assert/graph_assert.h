/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AD954C4ADF5B44F5B1CC8BCD72EE9ED6
#define AD954C4ADF5B44F5B1CC8BCD72EE9ED6

#include "ge_graph_dsl/ge.h"
#include "ge_graph_dsl/assert/check_utils.h"
#include "ge_graph_dsl/assert/assert_error.h"
#include "ge_graph_dsl/assert/filter_scope_guard.h"

GE_NS_BEGIN

#ifdef GTEST_MESSAGE_AT_
#define GRAPH_CHECK_MESSAGE(file, line, message) \
  GTEST_MESSAGE_AT_(file, line, message, ::testing::TestPartResult::kFatalFailure)
#elif
#define GRAPH_CHECK_MESSAGE(file, line, message) throw AssertError(file, line, message)
#endif

namespace detail {
struct GraphAssert {
  GraphAssert(const char *file, unsigned int line, const std::string &phase_id)
      : file_(file), line_(line), phase_id_(phase_id) {}

  void operator|(const ::GE_NS::GraphCheckFun &check_fun) {
    bool ret = ::GE_NS::CheckUtils::CheckGraph(phase_id_, check_fun);
    if (!ret) {
      auto message = "expect dump graph in phase: [" + phase_id_ + "], while not find the dump graph! ";
      GRAPH_CHECK_MESSAGE(file_, line_, message.c_str());
    }
  }

 private:
  const char *file_;
  unsigned int line_;
  const std::string phase_id_;
};
}  // namespace detail

#define DUMP_GRAPH_WHEN(...) ::GE_NS::FilterScopeGuard guard__COUNTER__({__VA_ARGS__});
#define CHECK_GRAPH(phase_id) \
  ::GE_NS::detail::GraphAssert(__FILE__, __LINE__, #phase_id) | [&](const ::GE_NS::ComputeGraphPtr &graph)

GE_NS_END

#endif
