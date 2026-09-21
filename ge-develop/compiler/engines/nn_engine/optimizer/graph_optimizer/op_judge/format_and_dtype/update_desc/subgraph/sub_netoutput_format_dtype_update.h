/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_UPDATE_DESC_SUBGRAPH_SUB_NETOUTPUT_FORMAT_DTYPE_UPDATE_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_UPDATE_DESC_SUBGRAPH_SUB_NETOUTPUT_FORMAT_DTYPE_UPDATE_H_

#include "graph_optimizer/op_judge/format_and_dtype/update_desc/subgraph/sub_graph_format_dtype_update.h"
namespace fe {

/** @brief update the input and the output descs of the netoutput node in the
* subgraph */
class SubNetOutputFormatDtypeUpdate : public SubGraphFormatDtypeUpdate {
 public:
  explicit SubNetOutputFormatDtypeUpdate(RefRelationsPtr reflection_builder_ptr)
      : SubGraphFormatDtypeUpdate(reflection_builder_ptr){};
  ~SubNetOutputFormatDtypeUpdate() override {};

  Status UpdateTensorDesc(ge::NodePtr node_ptr) override;

 private:
  Status UpdateDtype(ge::NodePtr node_ptr, const ge::InDataAnchorPtr &in_data_anchor_ptr);
};
}  // namespace fe

#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_UPDATE_DESC_SUBGRAPH_SUB_NETOUTPUT_FORMAT_DTYPE_UPDATE_H_
