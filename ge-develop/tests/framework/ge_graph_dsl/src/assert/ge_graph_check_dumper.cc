/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge_graph_check_dumper.h"
#include "graph/model.h"
#include "graph/buffer.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "ge_graph_default_checker.h"

GE_NS_BEGIN

GeGraphCheckDumper::GeGraphCheckDumper() { Reset(); }

bool GeGraphCheckDumper::IsNeedDump(const std::string &suffix) const {
  auto iter = std::find(suffixes_.begin(), suffixes_.end(), suffix);
  return (iter != suffixes_.end());
}

void GeGraphCheckDumper::Dump(const ge::ComputeGraphPtr &graph, const std::string &suffix) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (!IsNeedDump(suffix)) {
    return;
  }
  auto iter = buffers_.find(suffix);
  if (iter != buffers_.end()) {
    DumpGraph(graph, iter->second);
  } else {
    buffers_[suffix] = Buffer();
    DumpGraph(graph, buffers_.at(suffix));
  }
}

bool GeGraphCheckDumper::CheckFor(const GeGraphChecker &checker) {
  const std::lock_guard<std::mutex> lock(mutex_);
  auto iter = buffers_.find(checker.PhaseId());
  if (iter == buffers_.end()) {
    return false;
  }
  DoCheck(checker, iter->second);
  return true;
}

void GeGraphCheckDumper::DoCheck(const GeGraphChecker &checker, ::GE_NS::Buffer &buffer) {
  Model model("", "");
  Model::Load(buffer.GetData(), buffer.GetSize(), model);
  auto load_graph = model.GetGraph();
  checker.Check(load_graph);
}

void GeGraphCheckDumper::DumpGraph(const ge::ComputeGraphPtr &graph, ::GE_NS::Buffer &buffer) {
  Model model("", "");
  buffer.clear();
  model.SetGraph(graph);
  model.Save(buffer, true);
}

void GeGraphCheckDumper::Update(const std::vector<std::string> &new_suffixes_) {
  const std::lock_guard<std::mutex> lock(mutex_);
  suffixes_ = new_suffixes_;
  buffers_.clear();
}

void GeGraphCheckDumper::Reset() {
  const std::lock_guard<std::mutex> lock(mutex_);
  static std::vector<std::string> default_suffixes_{"PreRunAfterBuild"};
  suffixes_ = default_suffixes_;
  buffers_.clear();
}

GE_NS_END
