/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "user_hybrid_graph_manager.h"
#include "common/checker.h"
#include "common/memory/tensor_trans_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/ge_context.h"
#include "api/aclgrph/option_utils.h"

namespace ge {
namespace {
  const std::string kEnableExternalWeight = "1";
  const std::string kEnableAllowMultiParaCompile = "1";
  void SetDefaultExternalWeightAndMultiParaCompile(const uint32_t user_graph_id, const std::map<std::string, std::string> &origin_options,
                                std::map<std::string, std::string> &options_out) {
    std::string external_weight_val;
    (void)GetContext().GetOption(EXTERNAL_WEIGHT, external_weight_val);
    GELOGI("get external weight %s in context", external_weight_val.c_str());
    options_out = origin_options;
    auto it = options_out.find(EXTERNAL_WEIGHT);
    const bool has_set_external_weight =
        !external_weight_val.empty() || (it != options_out.end() && !it->second.empty());
    if (!has_set_external_weight) {
      GELOGI("set default external weight in hybrid mode in graph %u", user_graph_id);
      options_out[EXTERNAL_WEIGHT] = kEnableExternalWeight;
    }
    // 需要多线程并发编译多张图，ge.AllowMultiGraphParallelCompile 必须为 true
    options_out[OPTION_ALLOW_MULTI_GRAPH_PARALLEL_COMPILE] = kEnableAllowMultiParaCompile;
  }

void SetDynamicShapeOptions(const std::map<std::string, std::string> &origin_options,
                            std::map<std::string, std::string> &options_out) {
  options_out = origin_options;
  // need to erase some dynamic dims options
  options_out.erase(INPUT_SHAPE);
  options_out.erase(kDynamicDims);
  options_out.erase(DYNAMIC_NODE_TYPE);
  const auto it = options_out.find(OPTION_GRAPH_KEY);
  if (it != options_out.end()) {
    // compile cache key should be different from dynamic gear graph
    it->second = it->second + "_dynamic_shape";
  }
}
}
void UserHybridGraphManager::SetDynamicGraphId(const uint32_t user_graph_id) {
  const std::lock_guard<std::mutex> locker(dynamic_graph_id_mu_);
  const auto it = dynamic_graph_id_map_.find(user_graph_id);
  const uint32_t dynamic_gear_graph_id = inner_graph_id_cnt_++;
  const uint32_t dynamic_shape_graph_id = inner_graph_id_cnt_++;
  if (it == dynamic_graph_id_map_.end()) {
    dynamic_graph_id_map_[user_graph_id] = std::make_pair(dynamic_gear_graph_id, dynamic_shape_graph_id);
    GELOGI("set user graph id %u map, inner dynamic gear graph id %u, dynamic shape graph id %u",
           user_graph_id, dynamic_gear_graph_id, dynamic_shape_graph_id);
  }
}

bool UserHybridGraphManager::TryGetDynamicGraphId(const uint32_t user_graph_id, uint32_t &dynamic_gear_graph_id,
                                                  uint32_t &dynamic_shape_graph_id) {
  const std::lock_guard<std::mutex> locker(dynamic_graph_id_mu_);
  const auto it = dynamic_graph_id_map_.find(user_graph_id);
  if (it != dynamic_graph_id_map_.end()) {
    dynamic_gear_graph_id = it->second.first;
    dynamic_shape_graph_id = it->second.second;
  }
  return (it != dynamic_graph_id_map_.end());
}

bool UserHybridGraphManager::IsHybridMode(const std::map<std::string, std::string> &options) const {
  auto it = options.find(INPUT_SHAPE);
  const bool has_input_shape = (it != options.end() && !it->second.empty());
  it = options.find(kDynamicDims);
  const bool has_dynamic_dims = (it != options.end() && !it->second.empty());
  it = options.find(DYNAMIC_NODE_TYPE);
  const bool has_dynamic_node_type = (it != options.end() && it->second == kEnableExternalWeight);
  it = options.find("ge.compileHybridMode");
  const bool enable_hybrid_mode = (it != options.end() && it->second == kEnableExternalWeight);
  return (has_input_shape && has_dynamic_dims && has_dynamic_node_type && enable_hybrid_mode);
}

Status UserHybridGraphManager::AddGraph(uint32_t user_graph_id, const Graph &graph,
                                        const std::map<std::string, std::string> &options) {
  if (!IsHybridMode(options)) {
    return user_graph_manager_.AddGraph(user_graph_id, graph, options);
  }
  if (EnableSliceSchedule()) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10003", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>({"experimental_enable_jit_executor_v2", "true",
            "experimental_enable_jit_executor_v2 and compile_hybrid_mode feature cannot be enabled at the same time."}));
    GELOGE(PARAM_INVALID, "experimental_enable_jit_executor_v2 and compile_hybrid_mode feature cannot be enabled at the same time.");
    return PARAM_INVALID;
  }
  AscendString graph_name;
  GE_ASSERT_SUCCESS(graph.GetName(graph_name));
  GELOGI("find graph id %u[%s] is hybrid mode", user_graph_id, graph_name.GetString());
  std::string dynamic_graph_name = std::string(graph_name.GetString()) + "_dynamic";
  Graph graph_clone(dynamic_graph_name.c_str());
  // need to clone first, because added graph will be set attr ATTR_NAME_GRAPH_HAS_BEEN_ADDED
  // which will be verified in AddGraph process
  GE_ASSERT_GRAPH_SUCCESS(graph_clone.CopyFrom(graph));
  SetDynamicGraphId(user_graph_id);
  uint32_t dynamic_gear_graph_id = 0;
  uint32_t dynamic_shape_graph_id = 0;
  GE_ASSERT_TRUE(TryGetDynamicGraphId(user_graph_id, dynamic_gear_graph_id, dynamic_shape_graph_id));
  std::map<std::string, std::string> dynamic_gear_options;
  SetDefaultExternalWeightAndMultiParaCompile(user_graph_id, options, dynamic_gear_options);
  GE_ASSERT_SUCCESS(user_graph_manager_.AddGraph(dynamic_gear_graph_id, graph, dynamic_gear_options));
  std::map<std::string, std::string> dynamic_shape_options;
  SetDynamicShapeOptions(dynamic_gear_options, dynamic_shape_options);
  GE_ASSERT_SUCCESS(user_graph_manager_.AddGraph(dynamic_shape_graph_id, graph_clone, dynamic_shape_options),
                    "Add dynamic graph id %u failed", dynamic_shape_graph_id);
  GELOGI("Add dynamic graph id %u successfully", dynamic_shape_graph_id);
  return SUCCESS;
}

Status UserHybridGraphManager::RecordDynamicGearInfo(const uint32_t graph_id) {
  GraphNodePtr graph_node = nullptr;
  OmeContext ome_ctx;
  GE_ASSERT_SUCCESS(user_graph_manager_.GetOmeContextByGraphId(graph_id, ome_ctx));
  HybridDynamicDimsInfo dynamic_dims_info;
  dynamic_dims_info.user_input_dims = ome_ctx.user_input_dims;
  for (const auto &ele : ome_ctx.dynamic_shape_dims) {
    std::vector<int64_t> tmp_dims;
    tmp_dims.insert(tmp_dims.end(), ele.begin(), ele.end());
    dynamic_dims_info.dynamic_shape_dims.emplace_back(std::move(tmp_dims));
  }
  for (size_t i = 0U; i < dynamic_dims_info.dynamic_shape_dims.size(); ++i) {
    GELOGI("index %zu dynamic dims is %s", i, ToString(dynamic_dims_info.dynamic_shape_dims[i]).c_str());
  }
  for (size_t i = 0U; i < dynamic_dims_info.user_input_dims.size(); ++i) {
    const auto &input_dims_map = dynamic_dims_info.user_input_dims[i];
    GELOGI("input name %s shape is %s", input_dims_map.first.c_str(), ToString(input_dims_map.second).c_str());
  }
  SetDynamicGearInfo(graph_id, dynamic_dims_info);
  return SUCCESS;
}

void UserHybridGraphManager::SetDynamicGearInfo(const uint32_t graph_id, const HybridDynamicDimsInfo &dynamic_dims_info) {
  const std::lock_guard<std::mutex> locker(dynamic_dims_info_mu_);
  const auto it = dynamic_dims_info_map_.find(graph_id);
  if (it == dynamic_dims_info_map_.end()) {
    dynamic_dims_info_map_[graph_id] = dynamic_dims_info;
  }
}

Status UserHybridGraphManager::BuildGraph(uint32_t user_graph_id, const std::vector<GeTensor> &inputs,
    uint64_t session_id) {
  uint32_t dynamic_gear_graph_id = 0;
  uint32_t dynamic_shape_graph_id = 0;
  if (!TryGetDynamicGraphId(user_graph_id, dynamic_gear_graph_id, dynamic_shape_graph_id)) {
    return user_graph_manager_.BuildGraph(user_graph_id, inputs, session_id);
  }
  const uint32_t kHybridThreadNum = 2;
  ThreadPool thread_pool("ge_hybrid_compile_mode", kHybridThreadNum);
  auto fut1 = thread_pool.commit([dynamic_gear_graph_id, this, &inputs, session_id]() -> Status {
    GE_ASSERT_SUCCESS(user_graph_manager_.BuildGraph(dynamic_gear_graph_id, inputs, session_id));
    GE_ASSERT_SUCCESS(RecordDynamicGearInfo(dynamic_gear_graph_id));
    return SUCCESS;
  });
  GE_ASSERT_TRUE(fut1.valid(), "create multi hybrid compile dynamic gear graph thread fail,graph_id = %u",
                 user_graph_id);
  auto fut2 = thread_pool.commit([dynamic_shape_graph_id, this, &inputs, session_id]() -> Status {
    GE_ASSERT_SUCCESS(user_graph_manager_.BuildGraph(dynamic_shape_graph_id, inputs, session_id));
    return SUCCESS;
  });
  GE_ASSERT_TRUE(fut2.valid(), "create multi hybrid compile dynamic shape graph thread fail,graph_id = %u",
                 user_graph_id);
  Status ret = SUCCESS;
  ret = fut1.get();
  GE_ASSERT_SUCCESS(ret, "Failed to compile dynamic gear graph,graph_id = %u", user_graph_id);
  ret = fut2.get();
  GE_ASSERT_SUCCESS(ret, "Failed to compile dynamic shape graph,graph_id = %u", user_graph_id);
  return ret;
}

Status UserHybridGraphManager::GetDynamicGearInfo(const uint32_t graph_id, HybridDynamicDimsInfo &dynamic_dims_info) {
  const std::lock_guard<std::mutex> locker(dynamic_dims_info_mu_);
  const auto it = dynamic_dims_info_map_.find(graph_id);
  GE_ASSERT_TRUE(it != dynamic_dims_info_map_.end(), "can not find graph %u dynamic dims info", graph_id);
  dynamic_dims_info = it->second;
  return SUCCESS;
}

Status UserHybridGraphManager::SelectExecuteGraph(const uint32_t dynamic_gear_graph_id,
    const uint32_t dynamic_shape_graph_id, const std::vector<gert::Tensor> &inputs, uint32_t &select_graph_id) {
  select_graph_id = dynamic_shape_graph_id;
  HybridDynamicDimsInfo dynamic_dims_info;
  GE_ASSERT_SUCCESS(GetDynamicGearInfo(dynamic_gear_graph_id, dynamic_dims_info),
    "can not get %u dynamic dims", dynamic_gear_graph_id);
  std::vector<std::vector<int64_t>> cur_shapes;
  for (size_t i = 0U; i < inputs.size(); ++i) {
    const auto &input_shape = inputs[i].GetStorageShape();
    auto cur_shape = TensorTransUtils::GetDimsFromGertShape(input_shape);
    GELOGI("get graph id %u current index %zu, input shape is %s", dynamic_gear_graph_id, i,
      ToString(cur_shape).c_str());
    cur_shapes.emplace_back(std::move(cur_shape));
  }

  GE_ASSERT_TRUE(cur_shapes.size() == dynamic_dims_info.user_input_dims.size(),
      "current shape %zu is not equal dynamic dims %zu", cur_shapes.size(), dynamic_dims_info.user_input_dims.size());
  std::vector<int64_t> cur_dynamic_dims;
  const auto &user_input_dims = dynamic_dims_info.user_input_dims;
  for (size_t i = 0UL; i < user_input_dims.size(); ++i) {
    const auto &user_input_dim = user_input_dims.at(i).second;
    for (size_t j = 0U; j < user_input_dim.size(); ++j) {
      if (user_input_dim.at(j) < 0) {
        GE_ASSERT_TRUE(cur_shapes[i].size() == user_input_dim.size(),
                       "[Check][Param] The shape size:%zu of dynamic input:%s should equal to input shape:%zu",
                       cur_shapes[i].size(), user_input_dims[i].first.c_str(), user_input_dim.size());
        cur_dynamic_dims.emplace_back(cur_shapes[i][j]);
      }
    }
  }
  for (const auto &dynamic_dim : dynamic_dims_info.dynamic_shape_dims) {
    if (dynamic_dim == cur_dynamic_dims) {
      select_graph_id = dynamic_gear_graph_id;
      break;
    }
  }
  if (select_graph_id == dynamic_gear_graph_id) {
    GELOGI("find dynamic dims %s matched, execute dynamic gear graph %u", ToString(cur_dynamic_dims).c_str(), dynamic_gear_graph_id);
  } else {
    GELOGI("find dynamic dims %s not matched, execute dynamic shape graph %u", ToString(cur_dynamic_dims).c_str(), dynamic_shape_graph_id);
  }
  return SUCCESS;
}

Status UserHybridGraphManager::RunGraphAsync(uint32_t user_graph_id, std::vector<gert::Tensor> &&inputs,
    uint64_t session_id, const RunAsyncCallbackV2 &callback) {
  uint32_t dynamic_gear_graph_id = 0;
  uint32_t dynamic_shape_graph_id = 0;
  if (!TryGetDynamicGraphId(user_graph_id, dynamic_gear_graph_id, dynamic_shape_graph_id)) {
    return user_graph_manager_.RunGraphAsync(user_graph_id, std::move(inputs), session_id, callback);
  }
  uint32_t selected_graph_id = 0U;
  GE_ASSERT_SUCCESS(SelectExecuteGraph(dynamic_gear_graph_id, dynamic_shape_graph_id, inputs, selected_graph_id));
  GE_ASSERT_SUCCESS(user_graph_manager_.RunGraphAsync(selected_graph_id, std::move(inputs), session_id, callback));
  return SUCCESS;
}

Status UserHybridGraphManager::RemoveGraph(uint32_t user_graph_id) {
  uint32_t dynamic_gear_graph_id = 0;
  uint32_t dynamic_shape_graph_id = 0;
  if (!TryGetDynamicGraphId(user_graph_id, dynamic_gear_graph_id, dynamic_shape_graph_id)) {
    return user_graph_manager_.RemoveGraph(user_graph_id);
  }
  GE_ASSERT_SUCCESS(user_graph_manager_.RemoveGraph(dynamic_gear_graph_id));
  GE_ASSERT_SUCCESS(user_graph_manager_.RemoveGraph(dynamic_shape_graph_id));
  {
    const std::lock_guard<std::mutex> locker(dynamic_graph_id_mu_);
    dynamic_graph_id_map_.erase(user_graph_id);
  }
  {
    const std::lock_guard<std::mutex> locker(dynamic_dims_info_mu_);
    dynamic_dims_info_map_.erase(user_graph_id);
  }

  return SUCCESS;
}

bool UserHybridGraphManager::IsGraphNeedRebuild(uint32_t user_graph_id) {
  uint32_t dynamic_gear_graph_id = 0;
  uint32_t dynamic_shape_graph_id = 0;
  if (!TryGetDynamicGraphId(user_graph_id, dynamic_gear_graph_id, dynamic_shape_graph_id)) {
    return user_graph_manager_.IsGraphNeedRebuild(user_graph_id);
  }

  return user_graph_manager_.IsGraphNeedRebuild(dynamic_gear_graph_id) ||
         user_graph_manager_.IsGraphNeedRebuild(dynamic_shape_graph_id);
}

Status UserHybridGraphManager::GetCompiledFlag(uint32_t user_graph_id, bool &flag) {
  uint32_t dynamic_gear_graph_id = 0;
  uint32_t dynamic_shape_graph_id = 0;
  if (!TryGetDynamicGraphId(user_graph_id, dynamic_gear_graph_id, dynamic_shape_graph_id)) {
    return user_graph_manager_.GetCompiledFlag(user_graph_id, flag);
  }
  return user_graph_manager_.GetCompiledFlag(dynamic_gear_graph_id, flag);
}

Status UserHybridGraphManager::SetCompiledFlag(uint32_t user_graph_id, bool flag) {
  uint32_t dynamic_gear_graph_id = 0;
  uint32_t dynamic_shape_graph_id = 0;
  if (!TryGetDynamicGraphId(user_graph_id, dynamic_gear_graph_id, dynamic_shape_graph_id)) {
    return user_graph_manager_.SetCompiledFlag(user_graph_id, flag);
  }
  GE_ASSERT_SUCCESS(user_graph_manager_.SetCompiledFlag(dynamic_gear_graph_id, flag));
  GE_ASSERT_SUCCESS(user_graph_manager_.SetCompiledFlag(dynamic_shape_graph_id, flag));
  return SUCCESS;
}

Status UserHybridGraphManager::Finalize() {
  {
    const std::lock_guard<std::mutex> locker(dynamic_dims_info_mu_);
    dynamic_dims_info_map_.clear();
  }
  {
    const std::lock_guard<std::mutex> locker(dynamic_graph_id_mu_);
    dynamic_graph_id_map_.clear();
  }
  return SUCCESS;
}
} // namespace ge