/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_INC_COMMON_LXFUSION_JSON_UTIL_H_
#define FUSION_ENGINE_INC_COMMON_LXFUSION_JSON_UTIL_H_

#include <nlohmann/json.hpp>
#include "common/aicore_util_types.h"
#include "common/aicore_util_attr_define.h"
#include "graph_optimizer/fusion_common/op_slice_info.h"
#include "graph_optimizer/graph_optimize_register_error_codes.h"
#include "graph/compute_graph.h"

void from_json(const nlohmann::json& json_value, rtSmData_t& rtSmData);
void from_json(const nlohmann::json& json_value, rtSmDesc_t& rtSmDesc);
void to_json(nlohmann::json& json_value, const rtSmData_t& rtSmData);
void to_json(nlohmann::json& json_value, const rtSmDesc_t& rtSmDesc);

namespace fe {
using ToOpStructPtr = std::shared_ptr<fe::ToOpStruct_t>;
using L2FusionInfoPtr = std::shared_ptr<fe::TaskL2FusionInfo_t>;
extern const std::string L1_FUSION_TO_OP_STRUCT;
extern const std::string L2_FUSION_TO_OP_STRUCT;

Status GetL1InfoFromJson(ge::OpDescPtr op_desc_ptr);

Status GetL2InfoFromJson(ge::OpDescPtr op_desc_ptr);

Status GetTaskL2FusionInfoFromJson(ge::OpDescPtr op_desc_ptr);

Status ReadGraphInfoFromJson(ge::ComputeGraph &graph);

Status WriteGraphInfoToJson(ge::ComputeGraph &graph);

Status ReadOpSliceInfoFromJson(ge::ComputeGraph &graph);

Status WriteOpSliceInfoToJson(ge::ComputeGraph &graph);

void SetOpSliceInfoToJson(fe::OpCalcInfo &op_calc_info, std::string &op_calc_info_str);

void SetFusionOpSliceInfoToJson(fe::OpCalcInfo &op_calc_info, std::string &op_calc_info_str);

void GetOpSliceInfoFromJson(fe::OpCalcInfo &op_calc_info, std::string &op_calc_info_str);

void GetFusionOpSliceInfoFromJson(fe::OpCalcInfo &op_calc_info, std::string &op_calc_info_str);

void GetL2ToOpStructFromJson(ge::OpDescPtr &op_desc_ptr, ToOpStructPtr &l2_info_ptr);

void GetL1ToOpStructFromJson(ge::OpDescPtr &op_desc_ptr, ToOpStructPtr &l1_info_ptr);

void GetStridedToOpStructFromJson(ge::OpDescPtr& op_desc_ptr, ToOpStructPtr& strided_info_ptr,
                                  const string& pointer_key, const string& json_key);

L2FusionInfoPtr GetL2FusionInfoFromJson(ge::OpDescPtr &op_desc_ptr);

void SetL2FusionInfoToNode(ge::OpDescPtr &op_desc_ptr, L2FusionInfoPtr &l2_fusion_info_ptr);

void SetStridedInfoToNode(ge::OpDescPtr& op_desc_ptr, ToOpStructPtr& strided_info_ptr,
                          const string& pointer_key, const string& json_key);

void to_json(nlohmann::json& json_value, const InputReduceInfo& input_reduce_info);

void to_json(nlohmann::json& json_value, const OutputReduceInfo& output_reduce_info);

void to_json(nlohmann::json& json_value, const fe_sm_desc_t& fe_sm_desc);

void to_json(nlohmann::json& json_value, const InputSplitInfo& input_split_info);

void to_json(nlohmann::json& json_value, const OutputSplitInfo& output_split_info);

void to_json(nlohmann::json& json_value, const ToOpStruct_t& op_struct);

void to_json(nlohmann::json& json_value, const AxisSplitMap& axis_split_map);

void to_json(nlohmann::json& json_value, const AxisReduceMap& axis_reduce_map);

void to_json(nlohmann::json& json_value, const OpCalcInfo& op_calc_info);

void to_json(nlohmann::json& json_value, const TaskL2FusionInfo_t& task_l2_fusion_info);

void to_json(nlohmann::json& json_value, const L2FusionData_t& l2_fusion_data);

void from_json(const nlohmann::json& json_value, L2FusionData_t& l2_fusion_data);

void from_json(const nlohmann::json& json_value, AxisReduceMap& axis_reduce_map);

void from_json(const nlohmann::json& json_value, InputReduceInfo& input_reduce_info);

void from_json(const nlohmann::json& json_value, ToOpStruct_t& op_struct);

void from_json(const nlohmann::json& json_value, OpCalcInfo& op_calc_info);

void from_json(const nlohmann::json& json_value, OutputReduceInfo& output_reduce_info);

void from_json(const nlohmann::json& json_value, fe_sm_desc_t& fe_sm_desc);

void from_json(const nlohmann::json& json_value, AxisSplitMap& axis_split_map);

void from_json(const nlohmann::json& json_value, InputSplitInfo& input_split_info);

void from_json(const nlohmann::json& json_value, TaskL2FusionInfo_t& task_l2_fusion_info);

void from_json(const nlohmann::json& json_value, OutputSplitInfo& output_split_info);
} // namespace fe
#endif  // FUSION_ENGINE_INC_COMMON_LXFUSION_JSON_UTIL_H_
