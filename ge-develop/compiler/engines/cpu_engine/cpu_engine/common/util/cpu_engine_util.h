/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_ENGINE_UTIL_H_
#define AICPU_ENGINE_UTIL_H_

#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

#include "error_code/error_code.h"
#include "ge/ge_api_error_codes.h"
#include "graph/compute_graph.h"
#include "proto/aicpu/cpu_node_def.pb.h"
#include "aicpu_ops_kernel_info_store/op_struct.h"
#include "common/aicpu_ops_kernel_builder/kernel_builder.h"
#include "common/sgt_slice_type.h"

namespace aicpu {

ge::Status BuildAicpuNodeDef(const ge::OpDescPtr &op_desc_ptr,
                             aicpuops::NodeDef &node_def);
ge::Status GetAttrValueFromGe(const ge::OpDescPtr &op_desc_ptr,
                              const std::string &attr_name,
                              const ge::GeAttrValue::ValueType ge_value_type,
                              aicpuops::AttrValue &attr_value);

ge::Status FillAttrOfAicpuNodeDef(const ge::OpDescPtr &op_desc_ptr, aicpuops::NodeDef &node_def);

ge::Status GetAicpuFftsPlusInfo(const ge::OpDescPtr &op_desc_ptr,
                                FftsPlusInfo &ffts_info);

ge::Status InsertAicpuNodeDefAttrToOp(const ge::OpDescPtr &op_desc_ptr,
                                      aicpuops::NodeDef &node_def,
                                      const std::string &attr_name);

ge::Status BuildFftsPlusAicpuNodeShapeInfo(const ge::OpDescPtr &op_desc_ptr,
                                           aicpuops::NodeDef &node_def,
                                           const FftsPlusInfo &ffts_info);

ge::Status BuildFftsPlusAicpuNodeDef(const ge::OpDescPtr &op_desc_ptr,
                                     FftsPlusInfo &ffts_info);

ge::Status GenerateTransposeBeforeNode(ge::ComputeGraph &graph, const ge::NodePtr &node);

ge::Status GenerateTransposeAfterNode(ge::ComputeGraph &graph, const ge::NodePtr &node);

ge::NodePtr CreateTransposeNode(ge::ComputeGraph &graph, const std::string &name,
    ge::GeTensorDesc tensor_desc, const int insert_position);

ge::OpDescPtr CreateConstNode(const int insert_position);

ge::Status InsertTransposeNode(ge::OpDescPtr &op_desc_ptr, ge::ComputeGraph &graph, const ge::NodePtr &node);
                                     
/**
 * Set shape type
 * @param op_desc_ptr op desc
 * @param all_op_info op info
 * @return status whether this operation success
 */
ge::Status CheckAndSetUnknowType(ge::OpDescPtr &op_desc_ptr, const map<string, OpFullInfo> &all_op_info);
}  // namespace aicpu

#endif
