/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_8EFED0015C27464897BF64531355C810
#define INC_8EFED0015C27464897BF64531355C810

#include <string>
#include <mutex>
#include "ge_graph_dsl/ge.h"
#include "graph/utils/dumper/ge_graph_dumper.h"
#include "ge_dump_filter.h"

GE_NS_BEGIN

struct GeGraphChecker;

struct GeGraphCheckDumper : GeGraphDumper, GeDumpFilter {
  GeGraphCheckDumper();
  void Dump(const ge::ComputeGraphPtr &graph, const std::string &suffix) override;
  bool CheckFor(const GeGraphChecker &checker);

 private:
  void DoCheck(const GeGraphChecker &checker, ::GE_NS::Buffer &buffer);
  void DumpGraph(const ge::ComputeGraphPtr &graph, ::GE_NS::Buffer &buffer);

 private:
  void Update(const std::vector<std::string> &) override;
  void Reset() override;
  bool IsNeedDump(const std::string &suffix) const;

 private:
  mutable std::mutex mutex_;
  std::map<std::string, ::GE_NS::Buffer> buffers_;
  std::vector<std::string> suffixes_;
};

GE_NS_END

#endif
