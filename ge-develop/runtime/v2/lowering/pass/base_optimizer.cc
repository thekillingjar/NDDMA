/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "base_optimizer.h"

#include <sstream>

#include "common/checker.h"
#include "common/compile_profiling/ge_call_wrapper.h"

namespace gert {
namespace bg {
namespace {
void PrintChangedPasses(
    const std::vector<std::vector<std::reference_wrapper<const std::string>>> &turn_indexes_to_changed_pass_names) {
  if (!IsLogEnable(GE_MODULE_NAME, DLOG_WARN)) {
    return;
  }
  std::stringstream ss;
  ss << "Changed pass names every optimize turn: ";
  for (size_t i = 0U; i < turn_indexes_to_changed_pass_names.size(); ++i) {
    ss << "Turn " << i << ": ";
    for (size_t j = 0U; j < turn_indexes_to_changed_pass_names[i].size(); ++j) {
      if (j > 0U) {
        ss << ", ";
      }
      ss << turn_indexes_to_changed_pass_names[i][j].get();
    }
    ss << ". ";
  }
  GELOGW("%s", ss.str().c_str());
}
}  // namespace
BaseOptimizer &BaseOptimizer::AddPass(std::string name, std::unique_ptr<Pass> pass) {
  if (pass == nullptr) {
    GELOGW("Add nullptr pass %s", name.c_str());
    return *this;
  }
  passes_.emplace_back(std::move(name), std::move(pass));
  return *this;
}

BaseOptimizer &BaseOptimizer::AddOncePass(std::string name, std::unique_ptr<Pass> pass) {
  if (pass == nullptr) {
    GELOGW("Add nullptr once pass %s", name.c_str());
    return *this;
  }
  once_passes_.emplace_back(std::move(name), std::move(pass));
  return *this;
}
ge::graphStatus BaseOptimizer::Run(ge::ExecuteGraph *const graph, bool &changed) {
  GELOGD("start to run all passes");
  std::vector<std::vector<std::reference_wrapper<const std::string>>> turn_indexes_to_changed_pass_names;
  bool changed_this_turn;
  int32_t turn_count = 0;
  constexpr int32_t kMaxTurnCount = 10;
  do {
    turn_indexes_to_changed_pass_names.emplace_back();
    auto &changed_pass_names = *turn_indexes_to_changed_pass_names.rbegin();
    changed_this_turn = false;
    for (auto &name_and_pass : passes_) {
      bool pass_changed = false;
      GE_TIMESTAMP_START(RunPass);
      GE_ASSERT_SUCCESS(name_and_pass.second->Run(graph, pass_changed), "Failed to run pass %s",
                        name_and_pass.first.c_str());
      if (pass_changed) {
        changed_pass_names.emplace_back(name_and_pass.first);
        changed_this_turn = true;
        changed = true;
        GE_TIMESTAMP_EVENT_END(RunPass, name_and_pass.first.c_str());
      }
    }
  } while (changed_this_turn && turn_count++ < kMaxTurnCount);

  if (turn_count >= kMaxTurnCount) {
    GELOGW("Run all optimization passes on execute graph more than %d times (%d), jump out the re-pass process",
           kMaxTurnCount, turn_count);
    PrintChangedPasses(turn_indexes_to_changed_pass_names);
  }

  for (auto &name_and_pass : once_passes_) {
    bool pass_changed = false;
    GE_TIMESTAMP_START(RunOncePass);
    GE_ASSERT_SUCCESS(name_and_pass.second->Run(graph, pass_changed), "Failed to run once pass %s",
                      name_and_pass.first.c_str());
    GE_TIMESTAMP_EVENT_END(RunOncePass, name_and_pass.first.c_str());
  }

  return ge::GRAPH_SUCCESS;
}
}  // namespace bg
}  // namespace gert
