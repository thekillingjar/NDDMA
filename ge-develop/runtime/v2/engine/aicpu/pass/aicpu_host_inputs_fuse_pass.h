/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_ENGINE_AICPU_PASS_AICPU_HOST_INPUTS_FUSE_PASS_H_
#define AIR_CXX_RUNTIME_V2_ENGINE_AICPU_PASS_AICPU_HOST_INPUTS_FUSE_PASS_H_

#include "lowering/pass/pass.h"
#include "graph/utils/fast_node_utils.h"
#include "graph/utils/execute_graph_utils.h"

namespace gert {
namespace bg {
class AicpuHostInputsFusePass : public Pass {
 public:
  ge::graphStatus Run(ge::ExecuteGraph *const graph, bool &changed) override;

 private:
  ge::graphStatus SetHostInputsNodesAndIndex(ge::FastNode *const node);
  ge::graphStatus ReplaceFuseHostInputsNode(ge::FastNode *const fuse_node);
  ge::graphStatus FuseHostInputsProcNodes(ge::FastNode *const node, bool &changed);

  ge::graphStatus AddFuseOpOutputDesc(const ge::OpDescPtr &fuse_desc) const;
  ge::graphStatus AddFuseOpInputDesc(const ge::OpDescPtr &fuse_desc, const ge::OpDescPtr &dst_desc) const;
  ge::graphStatus AddFuseIndexesInput(ge::FastNode *const fuse_node) const;
  ge::FastNode *CreateFuseHostInputsNode(ge::FastNode *const node) const;

  std::vector<ge::FastNode *> host_inputs_proc_nodes_;
  std::vector<int32_t> host_inputs_addr_index_;
};
}  // namespace bg
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_ENGINE_AICPU_PASS_AICPU_HOST_INPUTS_FUSE_PASS_H_
