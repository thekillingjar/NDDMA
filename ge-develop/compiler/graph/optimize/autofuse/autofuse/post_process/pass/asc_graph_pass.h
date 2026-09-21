/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_POST_PROCESS_PASS_ASC_GRAPH_PASS_H_
#define AUTOFUSE_POST_PROCESS_PASS_ASC_GRAPH_PASS_H_
#include "ge_common/ge_api_types.h"
#include "graph/compute_graph.h"
#include "cse_pass.h"
#include "cyclic_external_lift_pass.h"
#include "broadcast_backward_pass.h"
#include "cube_fixpip_pass.h"

namespace ge {
class AscGraphPass {
 public:
  Status Run(const ComputeGraphPtr &graph) const;

 private:
  CsePass cse_pass_;
  CyclicExternalLiftPass cyclic_external_lift_pass_;
  BroadcastBackwardPass broadcast_backward_pass_;
  CubeFixpipPass cube_fixpip_pass_;
};
}  // namespace ge

#endif  // AUTOFUSE_POST_PROCESS_PASS_ASC_GRAPH_PASS_H_
