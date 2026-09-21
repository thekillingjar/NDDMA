/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_STREAM_GRAPH_OPTIMIZER_L2_OPTIMIZER_L2_FUSION_PARSER_L2_FUSION_PARSER_H_
#define FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_STREAM_GRAPH_OPTIMIZER_L2_OPTIMIZER_L2_FUSION_PARSER_L2_FUSION_PARSER_H_

#include "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_fusion_comm/l2_fusion_comm.h"

namespace fe {
class L2FusionParser {
 public:
  static Status GetDataDependentCountMap(const ge::ComputeGraph &graph,
                                         std::map<uint64_t, size_t> &data_dependent_count_map);
  static Status GetDataFromGraph(const ge::ComputeGraph &graph,
                                 const OpReferTensorIndexMap &refer_tensor_index_map,
                                 std::vector<OpL2DataInfo> &op_l2_data_vec);

  static void GetOpReferTensorIndexMap(const ge::ComputeGraph &graph, OpReferTensorIndexMap &refer_tensor_index_map);

 private:
  static bool HasAtomicNode(const ge::NodePtr &nodePtr);
  static Status GetDataFromNode(const ge::NodePtr &node, const OpReferTensorIndexMap &refer_tensor_index_map,
                                OpL2DataInfo &op_l2_data, std::vector<ge::NodePtr> &end_node_vec);
  static Status GetInputDataFromNode(const ge::NodePtr &node, TensorL2DataMap &input_tensor_l2_map);
  static Status GetOutputDataFromNode(const ge::NodePtr &node, const OpReferTensorIndexMap &refer_tensor_index_map,
                                      TensorL2DataMap &output_tensor_l2_map);
  static void ClearReferDataInfo(const ge::NodePtr &node, const OpReferTensorIndexMap &refer_tensor_index_map,
                                 std::vector<OpL2DataInfo> &op_l2_data_vec);
  static void ClearReferTensorData(const ge::InDataAnchorPtr &in_data_anchor,
                                   const OpReferTensorIndexMap &refer_tensor_index_map,
                                   std::vector<OpL2DataInfo> &op_l2_data_vec);

  static uint64_t GetDataUnitDataId(const ge::NodePtr &node, const size_t &tensor_index, const uint32_t &data_type,
                                    const ge::GeTensorDesc &tensor);

  static bool IsNotSupportOp(const ge::NodePtr &node);
  static bool NoNeedAllocOpL2Info(const ge::NodePtr &node);
  static bool NoNeedAllocOutputs(const ge::NodePtr &node);
  static bool NoNeedAllocOutput(const ge::GeTensorDesc &tensor_desc);
};
}  // namespace fe
#endif  // FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_STREAM_GRAPH_OPTIMIZER_L2_OPTIMIZER_L2_FUSION_PARSER_L2_FUSION_PARSER_H_
