/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_INFER_BASE_PASS_H_
#define GE_GRAPH_PASSES_INFER_BASE_PASS_H_

#include "graph/passes/base_pass.h"

namespace ge {
class InferBasePass : public BaseNodePass {
 public:
  Status Run(NodePtr &node) override;
  graphStatus InferAndUpdate(NodePtr &node, bool before_subgraph, OrderedNodeSet &changed_nodes);

 protected:
  virtual std::string SerialTensorInfo(const GeTensorDescPtr &tensor_desc) const = 0;
  virtual bool NeedInfer(const NodePtr &node) const;
  virtual graphStatus Infer(NodePtr &node) = 0;

  /**
   * Update the output TensorDesc by src TensorDesc. This will be called when updating peer node input desc.
   * @param src, input TensorDesc
   * @param dst, output TensorDesc to be updated
   * @return
   */
  virtual graphStatus UpdateTensorDesc(const GeTensorDescPtr &src, GeTensorDescPtr &dst, bool &changed) = 0;

  /**
   * Update the output TensorDesc for nodes which contain subgraphs.
   * In dynamic multi-dims/batch/images size scene, the update process maybe different,
   * in which case, the `InferBasePass` will call method `UpdateOutputFromSubgraphsForMultiDims` instead.
   * @param src, input TensorDesc from NetOutput nodes in all subgraphs
   * @param dst, output TensorDesc to be updated
   * @return
   */
  virtual graphStatus UpdateOutputFromSubgraphs(const std::vector<GeTensorDescPtr> &src,
                                                const GeTensorDescPtr &dst) = 0;
  virtual graphStatus UpdateOutputFromSubgraphsForMultiDims(const std::vector<GeTensorDescPtr> &src,
                                                            const GeTensorDescPtr &dst) = 0;
  virtual graphStatus UpdateOutputFromSubgraphsForSubgraphMultiDims(const std::vector<GeTensorDescPtr> &src,
                                                                    const GeTensorDescPtr &dst) = 0;

 private:
  void AddChangedNodesImmediateRepass(const OrderedNodeSet &changed_nodes);
  bool ContainsSubgraph(const NodePtr &node) const;
  std::vector<ComputeGraphPtr> GetCurNodeSubgraphs(const NodePtr &node) const;
  bool CheckSkipUpdateForMultiDims(const OpDescPtr &op_desc) const;
  graphStatus UpdateTensorDescToSubgraphData(const NodePtr &node);
  graphStatus UpdateTensorDescToParentNodeOutput(const NodePtr &node);
  graphStatus UpdateParentNodeContainsSubgraphs(const NodePtr &node,
                                                const std::vector<std::vector<GeTensorDescPtr>> &ref_out_tensors);
  graphStatus UpdateTensorDescToPeerInputs(const NodePtr &node, OrderedNodeSet &changed_nodes);
  void PrintInOutTensors(const NodePtr &node, const std::string &phase) const;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_INFER_BASE_PASS_H_
