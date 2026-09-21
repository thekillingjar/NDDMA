/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRAPH_UTILS_DUMPER_GE_GRAPH_DUMPER_H_
#define GRAPH_UTILS_DUMPER_GE_GRAPH_DUMPER_H_

#include "graph/compute_graph.h"

namespace ge {
struct GeGraphDumper {
  GeGraphDumper() = default;
  GeGraphDumper(const GeGraphDumper &) = delete;
  GeGraphDumper &operator=(const GeGraphDumper &) = delete;
  GeGraphDumper(GeGraphDumper &&) = delete;
  GeGraphDumper &operator=(GeGraphDumper &&) = delete;
  virtual void Dump(const ge::ComputeGraphPtr &graph, const std::string &suffix) = 0;
  virtual ~GeGraphDumper() = default;
};

struct GraphDumperRegistry {
  static GeGraphDumper &GetDumper();
  static void Register(GeGraphDumper &dumper);
  static void Unregister();
};

}  // namespace ge

#endif
