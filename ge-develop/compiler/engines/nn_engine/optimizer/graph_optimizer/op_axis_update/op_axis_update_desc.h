/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_AXIS_UPDATE_OP_AXIS_UPDATE_DESC_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_AXIS_UPDATE_OP_AXIS_UPDATE_DESC_H_

#include "common/fe_inner_error_codes.h"
#include "common/fe_utils.h"
#include "common/format/axis_util.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/node.h"
#include "graph/op_desc.h"

namespace fe {
class OpAxisUpdateDesc {

 public:
  explicit OpAxisUpdateDesc(const std::string& engine_name);
  ~OpAxisUpdateDesc();

  Status UpdateAxis(ge::ComputeGraph &graph);

 private:
  /**
   * set axis value for new format
   * @param op_desc : op desc info
   * @param origin_format : original format of current op
   * @param current_format : current format of op
   * @param origin_shape : original shape of current op
   * @return SUCCESS/FAILED
   */
  Status SetAxisAttributeValue(ge::OpDesc &op_desc, ge::Format &origin_format, ge::Format &current_format,
                               ge::GeShape &origin_shape);

  /**
   * Reset shape when format is fractal_z and update axis value.
   * @param input_or_output_tensor_desc : input or output desc info
   * @param op_imply_type : value of imply_type
   * @return SUCCESS/FAILED
   */
  Status ReshapeFz3DAndUpdateAxis(ge::OpDesc::Vistor<ge::GeTensorDescPtr> &input_or_output_tensor_desc,
                                  const bool &update_axis_flag, ge::OpDesc &op_desc);

  ge::GeShape GetFractalZNewShape(const ge::GeShape &origin_shape, const ge::Format &origin_format,
                                  const ge::Format primary_format, const int32_t sub_format,
                                  const ge::DataType &current_data_type) const;

  Status GetNewAxisForNz(const ge::OpDesc &op_desc, const ge::GeShape &origin_shape, vector<int64_t> &axis_index_vec);
  std::string engine_name_;
};
}  // namespace fe

#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_AXIS_UPDATE_OP_AXIS_UPDATE_DESC_H_
