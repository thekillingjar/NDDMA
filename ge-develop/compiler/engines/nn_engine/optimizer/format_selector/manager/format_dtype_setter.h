/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_MANAGER_FORMAT_DTYPE_SETTER_H_
#define FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_MANAGER_FORMAT_DTYPE_SETTER_H_

#include "common/graph/fe_graph_utils.h"
#include "common/fe_op_info_common.h"
#include "format_selector/manager/format_dtype_manager_base.h"
#include "graph/ge_local_context.h"

namespace fe {
class FormatDtypeSetter : public FormatDtypeManagerBase {
 public:
  explicit FormatDtypeSetter(const std::string& engine_name);
  ~FormatDtypeSetter() override;
  Status SetSupportFormatDtype(const ge::ComputeGraph& graph) const;
  Status SetSupportFormatDtypeByNode(ge::NodePtr node_ptr, const HeavyFormatInfo& heavy_format_info) const;
  void JudgeFirstLayerConv(const ge::NodePtr& node, const ge::OpDescPtr& op_desc) const;
  static Status MultiThreadSetSupportFormatDtypeOneNode(ge::NodePtr &node_ptr,
      FormatDtypeSetter *format_dtype_setter_ptr, const ge::GEThreadLocalContext &ge_context);
  Status MultiThreadSetSupportFormatDtype(const ge::ComputeGraph &graph);
 private:
  Status SetSupportFormatDtypeByNode(ge::NodePtr node_ptr) const;
  void SetTensorDtype(const ge::OpDescPtr &op_desc, const bool &is_input,
                      const std::vector<ge::DataType> &tensor_dtypes) const;
  void JudgeFirstLayerConvForInfer(const ge::NodePtr& node, const ge::OpDescPtr& op_desc) const;
  void JudgeFirstLayerConvForTrain(const ge::NodePtr& node, const ge::OpDescPtr& op_desc) const;
  void SetNodeDtypeByCustomDtypes(const ge::NodePtr &node_ptr) const;
  bool GetFirstLayerConv2DInputData(const ge::NodePtr &node, ge::NodePtr &data, uint32_t depth) const;
  bool GetFirstLayerConv2DWeight(const ge::NodePtr &node, ge::NodePtr &weight) const;
  void GetFirstLayerConv2DDW(const ge::NodePtr &data, ge::NodePtr &conv2d_dw,
                             const ge::NodePtr &conv2d_node, uint32_t depth, bool &find_flag) const;
};
}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_MANAGER_FORMAT_DTYPE_SETTER_H_
