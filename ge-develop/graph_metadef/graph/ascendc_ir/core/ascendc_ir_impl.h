/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRAPH_ASCENDC_IR_IMPL_H
#define GRAPH_ASCENDC_IR_IMPL_H

#include <string>
#include "attr_store.h"
#include "graph/compute_graph.h"
#include "graph/node.h"
#include "graph/anchor.h"
#include "graph/utils/op_desc_utils.h"
#include "common/ge_common/debug/ge_log.h"
#include "graph/operator.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir_def.h"
#include "graph/symbolizer/symbolic.h"

namespace ge {
namespace ascir {
namespace cg {
class CodeGenUtils;
}
}
class AscGraphImpl {
  friend class AscGraph;
  friend class AscGraphUtils;
  friend class ascir::cg::CodeGenUtils;
 public:
  explicit AscGraphImpl(const char *name);

  Axis *FindAxis(const int64_t axis_id) const;

  void SetTilingKey(const uint32_t tiling_key);

  int64_t GetTilingKey() const;

  void SetGraphType(const AscGraphType type);

  AscGraphType GetGraphType() const;

  AscNodePtr AddNode(ge::Operator &op);

  Expression CreateSizeVar(const int64_t value);

  AxisPtr CreateAxis(const std::string &name, Axis::Type type, const ge::Expression &size,
                     const std::vector<int64_t> &from, const int64_t split_peer = 0UL);

  graphStatus CreateSizeVar(const Expression &expression);

  Expression CreateSizeVar(const std::string &name);

  std::pair<AxisPtr, AxisPtr> BlockSplit(const int64_t axis_id, const std::string &outer_axis_name,
                                         const std::string &inner_axis_name);

  std::pair<AxisPtr, AxisPtr> TileSplit(const int64_t axis_id, const std::string &outer_axis_name,
                                        const std::string &inner_axis_name);

  AxisPtr MergeAxis(const std::vector<int64_t> &axis_ids, const std::string &merge_axis_name);

  bool BindBlock(const int64_t outter_id, const int64_t inner_id);

  bool ApplySplit(const AscNodePtr &node, const int64_t outter_id, const int64_t inner_id);

  bool ApplyMerge(const AscNodePtr &node, const int64_t merged_axis_id);

  static bool ApplyTensorAxisMerge(const AscNodePtr &node,
                                   const int64_t merged_axis_id,
                                   const std::vector<int64_t> &original);

  bool ApplyTensorAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id);

  static bool ApplySchedAxisMerge(const AscNodePtr &node,
                                  const int64_t merged_axis_id,
                                  const std::vector<int64_t> &original);

  bool ApplySchedAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id);

  static bool ApplyReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis);

  static bool ApplySchedAxisReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis);

  static bool ApplyTensorAxisReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis);

  static bool TryApplyAxisReplace(const AscNodePtr &node, const Axis &src, const Axis &dst);

  AscNodePtr FindNode(const char *name) const;

  std::vector<AxisPtr> GetAllAxis() const;

  std::vector<SizeVarPtr> GetAllSizeVar() const;

  TransInfoRoadOfGraph GetAllAxisTransInfo() const;

  AscNodeVisitor GetAllNodes() const;

  AscNodeVisitor GetInputNodes() const;

  std::string GetName() const;

  AscOpOutput CreateContiguousData(const char *name,
                                   const ge::DataType &dt,
                                   const std::vector<ge::Axis> &axes,
                                   const ge::Format &format);

  AscOpOutput CreateContiguousOut(const char *name,
                                  const ge::DataType &dt,
                                  const std::vector<ge::Axis> &axes,
                                  const ge::Format &format);

  void SortByExecOrder();

  const ComputeGraphPtr GetComputeGraph() const;

  static bool CopyFrom(const ge::AscGraph &src_graph, ge::AscGraph &dst_graph);
 private:
  std::pair<AxisPtr, AxisPtr> DoSplit(const int64_t axis_id, const std::string &outer_axis_name,
                                      const std::string &inner_axis_name, const bool is_tile_split);

  bool DoApplySplit(const AscNodePtr &node, const int64_t outter_id, const int64_t inner_id, const int64_t original_id);

  static bool DoApplyMerge(const AscNodePtr &node, const int64_t merged_axis_id, const std::vector<int64_t> &original);

  static bool DoApplyTensorAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id,
                                     const std::vector<int64_t> &original);

  bool DoApplyTensorAxisSplit(const AscNodePtr &node, const int64_t outter_id, const int64_t inner_id,
                              const int64_t original_id);

  static bool DoApplySchedAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id,
                                    const std::vector<int64_t> &original);

  static bool DoApplySchedAxisSplit(const AscNodePtr &node, const int64_t outter_id, const int64_t inner_id,
                                    const int64_t original_id);

  static bool DoApplySchedAxisReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis);

  static bool DoApplyTensorAxisReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis);

  static bool DoCopyAscGraphAttr(const AscGraph &src_asc_graph, AscGraph &dst_asc_graph);
  static bool DoCopyAscGraphAttrImpl(const ComputeGraphPtr &src_compute_graph,
                                     const ComputeGraphPtr &dst_compute_graph);

  static bool DoCopyAscNodeAndRelink(const AscGraph &src_asc_graph, AscGraph &dst_asc_graph);

  static bool DoCopyAscNodeTensorAttr(const AscNodePtr &src_node, AscNodePtr &dst_node);

  AscGraphAttr *GetOrCreateGraphAttrsGroup();
  AscGraphAttr *GetGraphAttrsGroup() const;
  static bool CheckContinuous(const AscNodePtr &node,
                              const uint32_t tensor_index,
                              const std::vector<int64_t> &original);
  Status AddSubGraph(const ComputeGraphPtr &sub_graph) const;
  Status FindSubGraph(const std::string &name, std::shared_ptr<AscGraphImpl> &graph_impl) const;
 private:
  ComputeGraphPtr compute_graph_;
};
using AscGraphImplPtr = std::shared_ptr<AscGraphImpl>;
}  // namespace ge

#endif  // GRAPH_ASCENDC_IR_IMPL_H
