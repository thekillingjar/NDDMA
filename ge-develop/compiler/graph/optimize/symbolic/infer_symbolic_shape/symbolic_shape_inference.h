/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_SYMBOLIC_INFER_SYMBOLIC_SHAPE_SYMBOLIC_SHAPE_INFERENCE_H_
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_SYMBOLIC_INFER_SYMBOLIC_SHAPE_SYMBOLIC_SHAPE_INFERENCE_H_

#include "ge_common/ge_api_types.h"
#include "graph/ge_tensor.h"
#include "graph/gnode.h"
#include "graph/node.h"
#include "register/op_impl_kernel_registry.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"
#include "symbolic_shape_symbolizer.h"

namespace ge {
class SymbolicShapeInference {
 public:
  Status Infer(const ComputeGraphPtr &graph) const;
 private:
  graphStatus CheckOutputSymbolDimNumValid(const OpDescPtr &op_desc) const;
  graphStatus PrintSymbolShapeInfo(const OpDescPtr &op_desc, const std::string &stage) const;
  Status SimplifyTensorSymbol(const GeTensorDescPtr &ge_tensor_desc) const;
  Status Simplify(const ComputeGraphPtr &graph) const;
  Status InferOneNode(NodePtr &node) const;
  Status UpdateSymbolShapeAndDtypeToPeerInputs(const NodePtr &node) const;
  Status UseStaticShapeIfWeCan(OpDescPtr &op_desc) const;
  Status DoComputeAndUpdate(const NodePtr &node, const OpDescPtr &op_desc,
                            const InferSymbolComputeKernelFunc &kernel_func) const;
  Status DoInferAndUpdate(const NodePtr &node, const OpDescPtr &op_desc,
                          const gert::OpImplKernelRegistry::OpImplFunctionsV2 *func) const;
};
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_SYMBOLIC_INFER_SYMBOLIC_SHAPE_SYMBOLIC_SHAPE_INFERENCE_H_
