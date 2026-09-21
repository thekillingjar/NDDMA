/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PREPROCESS_INSERT_OP_GE_AIPP_OP_H_
#define GE_GRAPH_PREPROCESS_INSERT_OP_GE_AIPP_OP_H_

#include <utility>
#include <vector>
#include "graph/preprocess/insert_op/base_insert_op.h"
#include "proto/insert_op.pb.h"
#include "nlohmann/json.hpp"

namespace ge {
class AippOp : public BaseInsertOp {
 public:
  AippOp() {}
  Status Init(domi::AippOpParams *aipp_params);

  ~AippOp() override;

  /// @ingroup domi_omg
  /// @brief Set Default Params
  Status SetDefaultParams() override;

  /// @ingroup domi_omg
  /// @brief Validate Params
  Status ValidateParams() override;

 protected:
  /// @ingroup domi_omg
  /// @brief Generate Op Desc
  Status GenerateOpDesc(OpDescPtr op_desc) override;

  /// @ingroup domi_omg
  /// @brief Get Target Position
  /// @param [in] graph graph
  /// @param [in|out] target_input target input
  /// @param [in|out] target_edges target edges
  Status GetTargetPosition(ComputeGraphPtr graph, NodePtr &target_input,
                           std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> &target_edges) override;

  Status InsertAippToGraph(ComputeGraphPtr &graph,
                           std::string &aippConfigPath,
                           const uint32_t index) override ;

  domi::AippOpParams::AippMode GetAippMode() override;

 private:
  AippOp &operator=(const AippOp &aipp_op);
  AippOp(const AippOp &aipp_op);
  std::string ConvertParamToJson() const;
  void SetVarReciChn(nlohmann::json& cfg_json) const;
  void SetInputFormat(nlohmann::json &cfg_json) const;
  void SetAippMode(nlohmann::json &cfg_json) const;
  void SetChnData(nlohmann::json& cfg_json) const;
  void SetPaddingData(nlohmann::json& cfg_json) const;
  void SetResizeData(nlohmann::json& cfg_json) const;
  void SetCropData(nlohmann::json& cfg_json) const;
  void SetOtherStaticData(nlohmann::json& cfg_json) const;
  void SetMatrix(nlohmann::json &cfg_json) const;
  void SetInputBias(nlohmann::json& cfg_json) const;
  void SetOutputBias(nlohmann::json& cfg_json) const;
  void ConvertParamToAttr(NamedAttrs &aipp_attrs);
  void SetCscDefaultValue();
  void SetDtcDefaultValue();
  NodePtr FindDataByIndex(const ComputeGraphPtr &graph, int32_t rank) const;
  Status FindDataNodeInSubgraph(const ComputeGraphPtr &graph, uint32_t in_tensor_idx,
                                std::vector<std::shared_ptr<Anchor>> &target_in_data_anchors,
                                uint32_t recursion_depth) const;
  Status FindInSubGraph(const std::shared_ptr<Node> &node,
                        std::vector<std::shared_ptr<Anchor>> &target_in_data_anchors) const;
  bool InDataAnchorHasSubGraphs(const NodePtr &node) const;
  Status GetAndCheckTarget(const ComputeGraphPtr &graph, int32_t rank, NodePtr &target,
                           std::set<uint32_t> &edge_indexes);
  Status GetStaticTargetNode(NodePtr &data_node, NodePtr &target);
  NodePtr CreateAipp(const OutDataAnchorPtr &out_anchor, const uint32_t index,
                     const InDataAnchorPtr &in_anchor);
  Status CreateAippData(const NodePtr &aipp_node);
  Status AddNodeToGraph(const NodePtr &aipp_node, int64_t max_dynamic_aipp_size);
  Status AddAippAttrbutes(const OpDescPtr &op_desc, const uint32_t &index);
  Status AddAttrToAippData(const OpDescPtr &aipp_data_op_desc);
  Status ConvertRelatedInputNameToRank();

  domi::AippOpParams *aipp_params_ = nullptr;
  NodePtr aipp_node_ = nullptr;
  NodePtr data_node_linked_aipp = nullptr;
  std::string cfg_json_;
};
}  // namespace ge

#endif  // GE_GRAPH_PREPROCESS_INSERT_OP_GE_AIPP_OP_H_

