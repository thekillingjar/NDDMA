/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_VARIABLE_REF_DELETE_OP_PASS_H_
#define GE_GRAPH_PASSES_VARIABLE_REF_DELETE_OP_PASS_H_

#include <string>
#include "framework/common/ge_inner_error_codes.h"
#include "graph/passes/graph_pass.h"

namespace ge {
class VariableRefDeleteOpPass : public GraphPass {
 public:
  Status Run(ge::ComputeGraphPtr graph);

 private:
  Status DealVariableRef(ge::ComputeGraphPtr &graph,
                         ge::NodePtr &variable_ref,
                         const std::string &ref_var_src_var_name) const;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_VARIABLE_REF_DELETE_OP_PASS_H_
