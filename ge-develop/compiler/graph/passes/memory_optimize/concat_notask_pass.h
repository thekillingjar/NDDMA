/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_CONCAT_NOTASK_PASS_H
#define GE_GRAPH_PASSES_CONCAT_NOTASK_PASS_H
#include "graph/graph.h"
#include "graph/passes/graph_pass.h"
#include "exe_graph/runtime/shape.h"
#include "transfer_shape_utils.h"

namespace ge {
class ConcatNotaskPass : public GraphPass {
public:
  Status Run(ComputeGraphPtr graph);
private:
  // concat dim轴校验
  bool CheckConcatDim(const ge::OpDescPtr &op_desc) const;
  bool CheckRealConcatDim(const gert::Shape &align_shape, const gert::Shape &src_shape,
    const transformer::AxisIndexMapping &axis_index_mapping, const int64_t &concat_dim,
    const ge::GeTensorDesc &input_tensor) const;
  void PrintTransferDims(const std::string name, const std::vector<std::vector<int32_t>> &transfer_dims) const;
  void PrintShape(const std::string name, const gert::Shape &shape) const;
  bool CheckConcatDimAlignment(const ge::OpDescPtr &op_desc, const gert::Shape &align_shape,
  const int64_t concat_dim, const ge::GeShape &ori_shape) const;
  bool GetTransferDims(const ge::OpDescPtr &op_desc, const gert::Shape &src_shape, const int64_t &reshape_type_mask,
    const ge::GeTensorDesc &input_tensor, transformer::AxisIndexMapping &axis_index_mapping) const;
  bool GetAlignedShape(const ge::OpDescPtr &op_desc, const gert::Shape &src_shape,
    const int64_t &reshape_type_mask,
    const ge::GeTensorDesc &input_tensor, gert::Shape &align_shape) const;
  bool CheckSplitAxis(const std::vector<int32_t> &src_axes, const int64_t &axis_idx,
    const int32_t &from_axis, const gert::Shape &align_shape, const gert::Shape &src_shape) const;
  bool IsFromAxisOne(const int64_t &axis_idx,
    const transformer::AxisIndexMapping &axis_index_mapping,
    const gert::Shape &align_shape, const gert::Shape &src_shape, const int32_t &from_axis) const;
  bool IsMergedAxisAllOnes(const int64_t &axis_idx, const std::vector<int64_t> &shape) const;
  bool IsFrontDimsAllOnesInMergedAxis(const gert::Shape &align_shape, const gert::Shape &src_shape,
  const transformer::AxisIndexMapping &axis_index_mapping, const int64_t &real_concat_dim,
   const int64_t &concat_dim) const;
  bool IsFrontDimsAllOnes(const transformer::AxisIndexMapping &axis_index_mapping,
    const std::vector<int64_t> &shape, const int64_t &real_concat_dim) const;

  // 输入校验
  bool InputCheck(const ge::NodePtr &node);
  bool CheckTensorAlign(const ge::NodePtr &node, const size_t input_index) const;
  bool HasSameSourceAnchor(const ge::InDataAnchorPtr &in_anchor,
    std::set<ge::OutDataAnchorPtr> &src_anchors) const;
  bool IsPreNodeTypeValid(const ge::InDataAnchorPtr &in_anchor);
  bool IsPreNodeWithSubgraph(const ge::InDataAnchorPtr &in_anchor) const;
  bool IsPreOutAnchorCanReuseForConcatOptimize(const ge::OutDataAnchorPtr out_anchor) const;
  bool IsPreOutAnchorValidMultiRef(const ge::OutDataAnchorPtr out_anchor) const;
  bool IsPreNodeAttrValid(const ge::OpDescPtr &pre_op_desc);
  void GetFirstOutAnchorNotInRefNode(const ge::InDataAnchorPtr &input_anchor,
    ge::OutDataAnchorPtr &src_anchor, int32_t current_deep) const;
  void GetFirstNotRefNode(const ge::InDataAnchorPtr &input_anchor, ge::NodePtr &node) const;
  bool IsSameInputMemType(const ge::OpDescPtr &pre_op_desc, const size_t outupt_idx,
    std::set<int64_t> &mem_types) const;
  bool IsScalarInput(const ge::NodePtr &node, const size_t input_index) const;

  // 输出校验
  bool OutputCheck(const ge::NodePtr &concat_node) const;

  // 是否是unknown shape校验
  bool IsOwnerGraphUnknown(const ge::NodePtr &concat_node) const;
  bool IsUnknownShapeOp(const ge::OpDescPtr &op_desc) const;

  // LxFusion校验
  bool LxFusionCheck(const ge::NodePtr &node) const;
  bool IsLxFusionMem(const ge::OpDescPtr &op_desc) const;
  bool IsLxFusionOp(const ge::NodePtr &node) const;

  // 设置notask等属性
  void SetAttrForConcatNotask(const ge::NodePtr &node) const;

  std::string cur_pro_node_name_;
};
}  // namespace ge
#endif