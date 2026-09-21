/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_COMPILE_NODES_PASS_H_
#define GE_GRAPH_PASSES_COMPILE_NODES_PASS_H_

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "graph/passes/graph_pass.h"
#include "api/gelib/gelib.h"

namespace ge {
///
/// compile nodes
///
class CompileNodesPass : public GraphPass {
 public:
  CompileNodesPass() {}
  virtual ~CompileNodesPass() {}

  graphStatus Run(ComputeGraphPtr graph) override;

 private:
  graphStatus GetSupportedKernel(const NodePtr &node,
                                 const std::shared_ptr<GELib> instance,
                                 std::string &kernel_lib_name) const;
  bool CheckAccuracySupport(const OpsKernelInfoStorePtr &kernel_info, const std::shared_ptr<GELib> instance,
                            const NodePtr &node, std::string& unsupported_reason) const;
  graphStatus CompileNodes(const std::shared_ptr<GELib> instance,
                           std::unordered_map<std::string, std::vector<NodePtr>> &kernel_to_compile_nodes) const;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_COMPILE_NODES_PASS_H_
