/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <nlohmann/json.hpp>
#include "common/sgt_slice_type.h"
#include "common/fe_log.h"
#include "common/thread_slice_info_utils.h"
namespace ffts {
void to_json(nlohmann::json &json_value, const DimRange &struct_value) {
  json_value = nlohmann::json{{"lower", struct_value.lower}, {"higher", struct_value.higher}};
}

void to_json(nlohmann::json &j, const OpCut &d) {
  j = nlohmann::json{{"splitCutIndex", d.split_cut_idx}, {"reduceCutIndex", d.reduce_cut_idx}, {"cutId", d.cut_id}};
}

void to_json(nlohmann::json &json_value, const ThreadSliceMap &struct_value) {
  json_value = nlohmann::json{{"thread_scopeId", struct_value.thread_scope_id},
                              {"is_first_node_in_topo_order", struct_value.is_first_node_in_topo_order},
                              {"threadMode", struct_value.GetThreadMode()},
                              {"node_num_in_thread_scope", struct_value.node_num_in_thread_scope},
                              {"slice_instance_num", struct_value.slice_instance_num},
                              {"parallel_window_size", struct_value.parallel_window_size},
                              {"thread_id", struct_value.thread_id},
                              {"dependencies", struct_value.dependencies},
                              {"input_axis", struct_value.input_axis},
                              {"output_axis", struct_value.output_axis},
                              {"input_tensor_indexes", struct_value.input_tensor_indexes},
                              {"output_tensor_indexes", struct_value.output_tensor_indexes},
                              {"input_tensor_slice", struct_value.input_tensor_slice},
                              {"output_tensor_slice", struct_value.output_tensor_slice},
                              {"ori_input_tensor_slice", struct_value.ori_input_tensor_slice},
                              {"ori_output_tensor_slice", struct_value.ori_output_tensor_slice},
                              {"atomic_types", struct_value.atomic_types},
                              {"same_atomic_clean_nodes", struct_value.same_atomic_clean_nodes},
                              {"cutType", struct_value.cut_type},
                              {"is_input_node_of_thread_scope", struct_value.is_input_node_of_thread_scope},
                              {"is_output_node_of_thread_scope", struct_value.is_output_node_of_thread_scope},
                              {"oriInputTensorShape", struct_value.ori_input_tensor_shape},
                              {"oriOutputTensorShape", struct_value.ori_output_tensor_shape},
                              {"original_node", struct_value.original_node},
                              {"inputCutList", struct_value.input_cut_list},
                              {"outputCutList", struct_value.output_cut_list}};
}

void from_json(const nlohmann::json &j, OpCut &d)
{
  j.at("splitCutIndex").get_to(d.split_cut_idx);
  j.at("reduceCutIndex").get_to(d.reduce_cut_idx);
  j.at("cutId").get_to(d.cut_id);
}

void from_json(const nlohmann::json &json_value, DimRange &struct_value)
{
  json_value.at("lower").get_to(struct_value.lower);
  json_value.at("higher").get_to(struct_value.higher);
}

void from_json(const nlohmann::json &json_value, ThreadSliceMap &struct_value)
{
  json_value.at("is_input_node_of_thread_scope").get_to(struct_value.is_input_node_of_thread_scope);
  json_value.at("is_output_node_of_thread_scope").get_to(struct_value.is_output_node_of_thread_scope);
  json_value.at("oriInputTensorShape").get_to(struct_value.ori_input_tensor_shape);
  json_value.at("oriOutputTensorShape").get_to(struct_value.ori_output_tensor_shape);
  json_value.at("original_node").get_to(struct_value.original_node);
  json_value.at("threadMode").get_to(struct_value.thread_mode);
  json_value.at("thread_scopeId").get_to(struct_value.thread_scope_id);
  json_value.at("node_num_in_thread_scope").get_to(struct_value.node_num_in_thread_scope);
  json_value.at("slice_instance_num").get_to(struct_value.slice_instance_num);
  json_value.at("parallel_window_size").get_to(struct_value.parallel_window_size);
  json_value.at("thread_id").get_to(struct_value.thread_id);
  json_value.at("dependencies").get_to(struct_value.dependencies);
  json_value.at("cutType").get_to(struct_value.cut_type);
  json_value.at("atomic_types").get_to(struct_value.atomic_types);
  json_value.at("same_atomic_clean_nodes").get_to(struct_value.same_atomic_clean_nodes);
  json_value.at("input_tensor_slice").get_to(struct_value.input_tensor_slice);
  json_value.at("output_tensor_slice").get_to(struct_value.output_tensor_slice);
  json_value.at("input_axis").get_to(struct_value.input_axis);
  json_value.at("output_axis").get_to(struct_value.output_axis);
  json_value.at("input_tensor_indexes").get_to(struct_value.input_tensor_indexes);
  json_value.at("output_tensor_indexes").get_to(struct_value.output_tensor_indexes);
  json_value.at("inputCutList").get_to(struct_value.input_cut_list);
  json_value.at("outputCutList").get_to(struct_value.output_cut_list);
}

void GetSliceJsonInfoStr(const ThreadSliceMap &struct_value, std::string &string_str) {
  nlohmann::json json = struct_value;
  string_str = json.dump();
}

void GetSliceInfoFromJson(ThreadSliceMap &value, const std::string &json_str) {
  try {
    nlohmann::json slice_info_json = nlohmann::json::parse(json_str);
    if (slice_info_json.is_null()) {
      FE_LOGW("Get l2 info: %s is empty.", json_str.c_str());
    } else {
      slice_info_json.get_to(value);
    }
  } catch (const nlohmann::json::exception &e) {
    FE_LOGW("Parse JSON string failed: %s", e.what());
    return;
  }
}
}  // namespace ffts
