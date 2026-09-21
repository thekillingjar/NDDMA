/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <dlfcn.h>
#include <memory>

#include "common/plugin/ge_make_unique_util.h"
#include "graph/utils/graph_utils.h"
#include "util/mem_utils.h"
#include "ge_common/ge_api_error_codes.h"
#include "graph_metadef/common/ge_common/util.h"
#include "guarded_execution_point.h"
#include "execution_point.h"

#include "common/checker.h"
#include "mmpa/mmpa_api.h"
#include "common/compile_profiling/ge_call_wrapper.h"

namespace ge {
constexpr uint32_t kMaxFileNameLen = 128U;
constexpr uint64_t kMaxStringSize = 1024U;
const std::string kGuardCheckSoName = "libguard_check.so";
constexpr char_t const *kGuardCheckSoDataResult = "_guard_check_so_data";

bool GuardedExecutionPoint::Match(const std::vector<gert::Tensor> &inputs) const {
  return matcher_.Match(inputs);
}

bool GuardCheckFuncCaller::Match(const vector<gert::Tensor> &inputs) const {
  if (func_ == nullptr) {
    return false;
  }

  size_t num_tensors = inputs.size();
  // todo remove this transform later
  // CheckGuard() use std::vector<gert::Tensor *> while ExecuteGraphWithStreamAsync() use std::vector<gert::Tensor>
  std::vector<gert::Tensor *> rt_inputs;
  for (size_t i = 0U; i < num_tensors; i++) {
    rt_inputs.emplace_back(const_cast<gert::Tensor*>(&inputs[i]));
  }

  char_t reason[kMaxStringSize];
  GE_TIMESTAMP_START(GuardMatch);
  bool match_result = func_(rt_inputs.data(), num_tensors, reason, kMaxStringSize);
  GE_TIMESTAMP_END(GuardMatch, "GuardMatch");
  if (!match_result) {
    GELOGI("GuardMiss reason: %s", reason);
  }
  return match_result;
}

Status GuardCheckFuncCaller::LoadGuardCheckFunc(ComputeGraphPtr computeGraphPtr) {
  GELOGD("Start load guard check func");
  const std::string *buffer = ge::AttrUtils::GetStr(computeGraphPtr, kGuardCheckSoDataResult);
  if ((buffer == nullptr) || buffer->empty()) {
    GELOGE(ge::FAILED, "LoadGuardCheckFunc GetStr fail %s", kGuardCheckSoDataResult);
    return ge::FAILED;
  }

  file_handle_ = static_cast<int32_t>(syscall(__NR_memfd_create, kGuardCheckSoName.c_str(), 0));
  const auto write_count = mmWrite(file_handle_, const_cast<char_t *>(buffer->c_str()), buffer->size());
  GE_ASSERT_TRUE(((write_count != EN_INVALID_PARAM) && (write_count != EN_ERROR)), "Write data failed, errno: %lld",
                 write_count);
  (void)lseek(static_cast<int32_t>(file_handle_), 0, SEEK_SET);
  std::string so_path = "/proc/self/fd/" + std::to_string(file_handle_);
  GELOGI("LoadGuardCheckFunc so_path:%s", so_path.c_str());
  // 通过/proc访问文件描述符对应的"文件"
  so_handle_ = mmDlopen(so_path.c_str(), static_cast<int32_t>(MMPA_RTLD_NOW));
  GE_ASSERT_NOTNULL(so_handle_);
  func_ = reinterpret_cast<GuardCheckFunc>(mmDlsym(so_handle_, "GuardCheckFunc"));
  GE_ASSERT_NOTNULL(func_);
  return ge::SUCCESS;
}

Status GuardCheckFuncCaller::UnloadGraphCheckFunc() const{
  if (so_handle_) {
    mmDlclose(so_handle_);
  }
  if (file_handle_ != -1) {
    close(file_handle_);
  }
  return ge::SUCCESS;
}

uint32_t GuardedExecutionPoint::GetPriority() const{
  return priority_;
}

void GuardedExecutionPoint::SetPriority(uint32_t userPriority) {
  priority_ = userPriority;
}

Status GuardedExecutionPoint::RemoveItem() {
  return matcher_.UnloadGraphCheckFunc();
}

bool GuardedExecutionPoint::SetCompiled(uint32_t compiled_graph_id, ComputeGraphPtr graph) {
  auto ret = matcher_.LoadGuardCheckFunc(graph);
  if (ret != ge::SUCCESS) {
    int64_t ep_id = -1;
    if (owner_point_) {
      ep_id = owner_point_->GetId();
    }
    GELOGE(GRAPH_FAILED, "[SetCompiled][LoadGuardCheckFunc] EP[%ld] GEP(%lu) failed", ep_id, compiled_graph_id);
    return false;
  }

  compiled_graph_id_ = compiled_graph_id;
  compiled_ = true;
  return true;
}

ComputeGraphPtr GuardedExecutionPoint::GetGraph() const {
  return compiled_graph_;
}

ComputeGraphPtr GuardedExecutionPoint::GetSlicedGraph() const {
  if (!owner_point_) {
    return nullptr;
  }
  return owner_point_->GetSlicedGraph();
}

Status GuardedExecutionPoint::CopySlicedGraph() {
  GELOGD("Copy compute graph begin.");
  if (!owner_point_) {
    GELOGD("Waring: owner_point_ is null.");
    return ge::SUCCESS;
  }
  const auto &sliced_graph = GetSlicedGraph();

  const std::string new_graph_name = sliced_graph->GetName();
  ComputeGraphPtr new_graph = MakeShared<ComputeGraph>(new_graph_name);
  GE_ASSERT_NOTNULL(new_graph);
  GE_ASSERT_SUCCESS(GraphUtils::CopyComputeGraph(sliced_graph, new_graph));

  compiled_graph_ = new_graph;

  GELOGI("Copy compute graph [%s] success.", new_graph_name.c_str());
  return ge::SUCCESS;
}
} // ge