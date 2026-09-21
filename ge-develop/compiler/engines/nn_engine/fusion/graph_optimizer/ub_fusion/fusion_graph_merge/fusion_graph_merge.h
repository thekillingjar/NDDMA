/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_UB_FUSION_FUSION_GRAPH_MERGE_H_
#define FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_UB_FUSION_FUSION_GRAPH_MERGE_H_

#include <map>
#include <string>
#include <vector>
#include "common/aicore_util_types.h"
#include "common/graph_comm.h"
#include "common/util/op_info_util.h"
#include "common/sgt_slice_type.h"

namespace fe {
class FusionGraphMerge;
using FusionGraphMergePtr = std::shared_ptr<FusionGraphMerge>;

Status CopyOutputSliceInfo(const ffts::ThreadSliceMap &a, size_t output_index, const ffts::ThreadSliceMapPtr &b,
                           bool &is_consistent);
void CopyBasicInfo(const ffts::ThreadSliceMap &a, const ffts::ThreadSliceMapPtr &b);
Status CopyInputSliceInfo(const ffts::ThreadSliceMap &a, size_t input_index, const ffts::ThreadSliceMapPtr &b);

class FusionGraphMerge {
 public:
  FusionGraphMerge(const std::string &scope_attr, const GraphCommPtr &graph_comm_ptr);
  virtual ~FusionGraphMerge();
  FusionGraphMerge(const FusionGraphMerge &in) = delete;
  FusionGraphMerge &operator=(const FusionGraphMerge &in) = delete;

  Status MergeFusionGraph(ge::ComputeGraph &fusion_graph);

  const std::string& GetScopeAttr() const;

 private:
  Status MergeFusionNodes(ge::ComputeGraph &fusion_graph);
  Status AddRelatedThreadNode(ScopeNodeMap &fusion_scope_map) const;
  Status GetScopeNodeMap(const ge::ComputeGraph &fusion_graph, ScopeNodeMap &fusion_scope_map) const;
  Status MergeEachFusionNode(ge::ComputeGraph &fusion_graph, std::vector<ge::NodePtr> &fus_nodelist);
  Status MergeFusionNodeL2Info(const ge::ComputeGraph &fusion_graph);

  virtual Status AfterMergeFusionGraph(ge::ComputeGraph &graph) {
    (void) graph;
    return SUCCESS;
  }

  void SetFusionOpOutputInplaceAbility(ge::OpDescPtr fus_op_desc,
                                       const vector<vector<ge::ConstGeTensorDescPtr>> &output_place);

  void SetAtomicFlagAndOutputIndex(const ge::NodePtr &first_node, const ge::NodePtr &fus_node) const;
  Status SetL2TaskInfoToFusionOp(ge::NodePtr fus_node) const;

  Status UpdateL2Info(const int64_t &origin_index, const int64_t &fusion_index, const L2FusionInfoPtr &originl2_info,
                      const L2FusionInfoPtr &fusion_l2_info) const;

  Status SetL2NameAndIndex(const L2FusionInfoPtr &originl2_info, L2FusionInfoPtr &fusion_l2_info) const;

  Status CreateFusionOpNodeGraph(vector<FusionDataFlow> &fus_input_edge_list,
                                 vector<FusionDataFlow> &fus_output_edge_list, vector<ge::NodePtr> &fus_nodelist,
                                 ge::OpDescPtr fusion_op_desc, ge::ComputeGraph &orig_graph);

  Status AddFusionNodeOpDesc(ge::OpDescPtr &fus_op, vector<FusionDataFlow> &fus_input_edge_list,
                             vector<FusionDataFlow> &fus_output_edge_list,
                             vector<pair<uint32_t, ge::NodePtr>> &src_op_out_index_in_fus_op);

  Status AddFusionNodeOutputDesc(ge::OpDescPtr fus_op, std::vector<FusionDataFlow> &fus_output_edge_list,
                                 vector<pair<uint32_t, ge::NodePtr>> &src_op_out_index_in_fus_op);

  Status AddFusionNodeInputDesc(ge::OpDescPtr fus_op, std::vector<FusionDataFlow> &fus_input_edge_list);

  void UpdateOutputRefPortIndex(ge::OpDescPtr &in_edge_dst_op_desc_ptr, const ge::OpDescPtr &fus_op,
                                const uint32_t dst_anchor_index) const;

  void SetMultiKernelOutPutOffsets(const ge::OpDescPtr &src_op, size_t src_out_idx, const ge::OpDescPtr &fus_op,
                                   std::vector<int64_t> &save_pre_output_offset) const;

  void UpdateOutputSgtSliceInfo(const ge::OpDescPtr &src_op, size_t src_out_idx, ge::OpDescPtr &fus_op,
                                std::vector<int64_t> &save_pre_output_offset) const;

  void CopySameAtomic(const ffts::ThreadSliceMapPtr &ori_slice_info,
                      const ffts::ThreadSliceMapPtr &fused_slice_info) const;

  void UpdateL1Attr(ge::OpDescPtr op_desc_ptr, const string &attr_key, const uint32_t &anchor_index,
                    const uint32_t &tensor_desc_index, vector<int64_t> &target_vec) const;

  Status RefreshFusionNodeDataFlow(ge::NodePtr fus_node, const ge::ComputeGraph &fusion_graph);

  void AddBuffFusionNodeInputDesc(vector<int> &in_mem_type_old_node,
                                  ge::OpDescPtr &in_edge_dst_op_desc_ptr,
                                  const ge::DataAnchorPtr &in_edge_dst_data_anchor_ptr,
                                  vector<int64_t> &FusNodeInputOffset,
                                  vector<int> &in_mem_type_fus_node) const;

  Status SetDataOutPutMapingAttr(
      std::map<ge::NodePtr, std::map<ge::AnchorPtr, ge::AnchorPtr>> fusion_op_anchors_map) const;

  void SetDataDumpRef(ge::NodePtr fus_node, const ge::ComputeGraph &fusion_graph) const;

  void SetDataDumpRefForInDataAnchors(ge::NodePtr fus_node) const;

  void SetDataDumpRefForOutputDataAnchors(ge::NodePtr fus_node) const;

  Status SetL2NameAndIndexForUnfusNode(L2FusionInfoPtr &originl2_info) const;

  Status GetFusionAnchorInfo(const std::string &origin_name, std::map<std::int64_t, std::int64_t> &out_index_map,
                             ge::NodePtr &fusion_node) const;

  void CreateOriginalFusionOpGraph(ge::NodePtr &fus_node_ptr, vector<ge::NodePtr> &fus_nodelist) const;

  void DelAttrsInOriginalFusionOpGraph(ge::NodePtr &node) const;

  void CalcPassStridedOutSize(const ge::NodePtr &fus_node_ptr, vector<ge::NodePtr> &fus_nodelist,
                              vector<pair<uint32_t, ge::NodePtr>> &src_op_out_index_in_fus_op) const;

  void CalcSingleOpStridedOutSize(const ge::NodePtr &node) const;

  void CalcSingleOpStridedSize(ge::ComputeGraph &graph) const;

  Status CalcStridedWriteOutSize(const ge::NodePtr &fus_node_ptr, vector<ge::NodePtr> &fus_nodelist,
                                 vector<pair<uint32_t, ge::NodePtr>> &src_op_out_index_in_fus_op) const;

  Status CalcStridedReadInSize(const ge::NodePtr &fus_node_ptr, vector<ge::NodePtr> &fus_nodelist) const;

  void UpdateFusionOpRefPortIndex(const ge::OpDescPtr &fus_op) const;

  Status ConvertSgtToJsonAttr(ge::OpDescPtr &fus_op) const;
  std::map<std::string, std::map<std::int64_t, ge::NodePtr>> fusion_op_name_map_all_;

  std::string scope_attr_;
 protected:
  GraphCommPtr graph_comm_ptr_;
  std::map<ge::ConstGeTensorDescPtr, int64_t> fusion_op_input_idx_map_;
  std::map<ge::ConstGeTensorDescPtr, int64_t> fusion_op_output_idx_map_;
};

}  // namespace fe
#endif  // FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_UB_FUSION_FUSION_GRAPH_MERGE_H_
