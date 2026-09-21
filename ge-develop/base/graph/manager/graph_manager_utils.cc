/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/manager/graph_manager_utils.h"
#include <set>
#include "common/checker.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "ge/ge_api_types.h"
#include "base/err_msg.h"
#include "common/memory/tensor_trans_utils.h"

namespace ge {
namespace {
const char_t *kFrozenInputIndexes = "ge.exec.frozenInputIndexes";

constexpr size_t kIndexOfFrozenDataIndex = 0UL;
constexpr size_t kIndexOfFrozenDataAddr = 1UL;
constexpr size_t kIndexOfFrozenDataLen = 2UL;

Status IsDigitStrings(const std::vector<std::string> &input_vec) {
  for (const auto &input : input_vec) {
    for (const char ch : input) {
      if (ch != ' ' && (static_cast<bool>(isdigit(static_cast<unsigned char>(ch))) == false)) {
        (void)REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char_t *>({"parameter", "value", "reason"}),
                           std::vector<const char_t *>({kFrozenInputIndexes, input.c_str(),
                           "The frozen input index is not a digit."}));
        return GE_GRAPH_OPTIONS_INVALID;
      }
    }
  }
  return SUCCESS;
}
}
const char_t *GetRunGraphModeStr(RunGraphMode mode) {
  static constexpr const char_t *run_graph_mode_str[] = {"RunGraph", "RunGraphAsync", "RunGraphWithStreamAsync",
    "InitRunGraphMode"};

  static_assert(sizeof(run_graph_mode_str) / sizeof(run_graph_mode_str[0]) ==
    (static_cast<size_t>(RunGraphMode::kRunGraphModeEnd) + 1U), "RunGraphMode enum and string array size mismatch");

  const auto index = static_cast<size_t>(mode);
  if (index >= static_cast<size_t>(RunGraphMode::kRunGraphModeEnd)) {
    return "UnknownRunGraphMode";
  }
  return run_graph_mode_str[index];
}

GraphNode::GraphNode(const GraphId graph_id) : graph_id_(graph_id), sem_(1U), const_mem_(), feature_mem_() {
  std::string opt;
  (void)GetContext().GetOption(GRAPH_MAX_PARALLEL_MODEL_NUM, opt);
  int32_t max_num = 0;
  try {
    max_num = std::stoi(opt);
    if (max_num > 0) {
      max_load_record_ = max_num;
    } else {
      GELOGW("Option %s param failed:%s", GRAPH_MAX_PARALLEL_MODEL_NUM.c_str(), opt.c_str());
    }
  } catch (...) {
    GELOGW("Option %s param failed:%s", GRAPH_MAX_PARALLEL_MODEL_NUM.c_str(), opt.c_str());
  }
  GELOGD("[GraphManager] graphMaxParallelModelNum is %u", max_load_record_);
}

GraphNode::~GraphNode() = default;

void GraphNode::Lock() {
  (void)sem_.Push(0U);
}

void GraphNode::Unlock() {
  uint8_t unused;
  (void)sem_.Pop(unused);
  (void)unused;
}

void GraphNode::IncreaseLoadCount() {
  const std::unique_lock<std::mutex> lock(load_count_mu_);
  if (load_record_ == max_load_record_) {
    GELOGW("Reach the maximum of load_count:%u", kMaxLoadNum);
    return;
  }
  ++load_count_;
}

Status GraphNode::ParseFrozenInputIndex() {
  std::string frozen_input;
  (void)ge::GetContext().GetOption(kFrozenInputIndexes, frozen_input);
  if (frozen_input.empty()) {
    return SUCCESS;
  }
  // frozen input: ids(0;1;2) id,addr,len(0,100,4;1,200,4)
  std::vector<std::string> frozen_input_vec = StringUtils::Split(frozen_input, ';');
  for (auto &frozen_info : frozen_input_vec) {
    std::vector<std::string> frozen_info_vec = StringUtils::Split(frozen_info, ',');
    GE_ASSERT_TRUE((frozen_info_vec.size() == 1UL) || (frozen_info_vec.size() == 3UL),
        "frozen info vector size should be 1 or 3 parsed by [%s]", frozen_info.c_str());
    GE_ASSERT_SUCCESS(IsDigitStrings(frozen_info_vec),
        "There are some invalid characters in frozen option value[%s].", frozen_input.c_str());
    int32_t frozen_input_index = -1;
    GE_ASSERT_SUCCESS(ConvertToInt32(frozen_info_vec[kIndexOfFrozenDataIndex], frozen_input_index));
    GE_ASSERT_TRUE((frozen_input_index >= 0), "Frozen_input_index must be greater than zero: %u", frozen_input_index);
    (void)frozen_input_indexes_.insert(static_cast<uint32_t>(frozen_input_index));
    if (frozen_info_vec.size() == 1UL) {
      GELOGD("Parse frozen input index[%d] success.", frozen_input_index);
      continue;
    }
    uint64_t addr = 0UL;
    GE_ASSERT_SUCCESS(ConvertToUint64(frozen_info_vec[kIndexOfFrozenDataAddr], addr));
    GE_ASSERT_TRUE(addr != 0UL, "Frozen input addr can not be nullptr.");
    uint64_t len = 0UL;
    GE_ASSERT_SUCCESS(ConvertToUint64(frozen_info_vec[kIndexOfFrozenDataLen], len));
    GE_ASSERT_TRUE(len != 0UL, "Frozen input length can not be zero.");
    frozen_index_to_node_info_[static_cast<uint32_t>(frozen_input_index)] = std::make_pair(addr, len);
    GELOGI("Parse and set frozen addr[%lx] length[%lu] for input index[%d] success.", addr, len, frozen_input_index);
  }
  return SUCCESS;
}

void GraphNode::SetLoaded() {
  --load_count_;
  ++load_record_;
  load_flag_ = true;
}
// 只fork编译时信息
std::shared_ptr<GraphNode> GraphNode::Fork(uint32_t forked_graph_id) {
  GraphNodePtr graph_node = MakeShared<GraphNode>(forked_graph_id);
  GE_ASSERT_NOTNULL(graph_node, "[Fork][GraphNode] fail, graph_id:%u", forked_graph_id);
  graph_node->origin_graph_id_ = this->graph_id_;
  graph_node->options_ = this->options_;
  graph_node->context_ = this->context_;
  graph_node->graph_ = this->graph_;
  graph_node->compute_graph_ = this->compute_graph_;
  graph_node->compiled_flag_ = this->compiled_flag_;
  graph_node->build_flag_ = this->GetBuildFlag();
  if (this->GetBuildFlag()) {
    // 当前状态机管理不好，build flag意味着编译加载成功
    // 因此fork时若原始graph node已经build成功，则fork的node也认为build成功
    graph_node->compiled_flag_ = true;
  }
  graph_node->async_ = this->async_;
  graph_node->is_specific_stream_ = this->is_specific_stream_;
  // todo GeModelPtr ge_model_;
  GE_ASSERT_NOTNULL(ge_root_model_);
  auto forked_root_model = ge_root_model_->Fork();
  GE_ASSERT_NOTNULL(forked_root_model);
  graph_node->ge_root_model_ = forked_root_model;
  graph_node->is_feature_base_refreshable_ = this->is_feature_base_refreshable_;
  // 如下三个数据结构跨越了编译与加载， first是地址属于加载阶段数据，second是size属于编译阶段数据（虽然是在loadgraph接口中生成的）
  // 在当前的实现中， loadgraph接口生成compile summary，并将如下三个数据结构中的size字段设置好。在加载时分配内存的时候，直接读数据结构中的size字段
  // 并且loadgraph接口判断若已存在compile summary，便不再生成summary，也不再设置size
  // 这样意味着，如下三个数据结构中的size字段和compile summary数据存在冗余，且数据是不同步的。坏味道：编译、加载阶段数据没有解耦
  // 对于fork接口，需要全盘接收编译时信息。如下三个size字段也算。
  // 临时方案：fork这里先复制
  // 正式方案：将如上的数据解耦
  graph_node->const_mem_ = std::make_pair(nullptr, this->const_mem_.second);
  graph_node->feature_mem_ = std::make_pair(nullptr, this->feature_mem_.second);
  graph_node->refreshable_feature_mem_ = std::make_pair(nullptr, this->refreshable_feature_mem_.second);

  graph_node->compiled_summary_ = this->compiled_summary_;
  graph_node->net_output_node_ = this->net_output_node_;
  graph_node->tensor_sizes_ = this->tensor_sizes_;
  graph_node->is_saved_net_output_tensor_info_flag_ = this->is_saved_net_output_tensor_info_flag_;
  graph_node->ge_tensor_descs_ = this->ge_tensor_descs_;
  graph_node->group_2_communication_nodes_ = this->group_2_communication_nodes_;
  return graph_node;
}

SubGraphInfo::SubGraphInfo() : subgraph_ptr_(nullptr), ge_model_ptr_(nullptr) {}

SubGraphInfo::~SubGraphInfo() = default;

GraphModelListener::GraphModelListener() : ModelListener() {}

Status GraphModelListener::OnComputeDone(const uint32_t model_id, const uint32_t data_index, const uint32_t result_code,
                                         std::vector<gert::Tensor> &outputs) {
  (void)outputs;
  GELOGI(
      "[GraphManager] graph compute call back, model_id:%u, task_id:%u, "
      "resultCode:%u.",
      model_id, data_index, result_code);

  const std::lock_guard<std::mutex> lock(mutex_);
  result_code_ = result_code;
  is_finished_ = true;
  condition_.notify_all();

  return SUCCESS;
}

uint32_t GraphModelListener::GetResultCode() {
  // Pending until async execute graph complete
  std::unique_lock<std::mutex> lock(mutex_);
  if (!is_finished_) {
    GELOGI("[GetResultCode] wait model execute finished.");
    condition_.wait(lock);
  }

  if (!is_finished_) {
    REPORT_INNER_ERR_MSG("E19999", "Model not run finish");
    GELOGE(INTERNAL_ERROR, "[Check][Param] model not run finish.");
    return INTERNAL_ERROR;
  }
  return result_code_;
}

Status GraphModelListener::ResetResult() {
  const std::lock_guard<std::mutex> lock(mutex_);
  result_code_ = 0U;
  is_finished_ = false;

  return SUCCESS;
}

void RunAsyncListener::SetCallback(const RunAsyncCallbackV2 &callback) {
  (void)sem_v2_.Push(callback);
}

Status RunAsyncListener::OnComputeDone(const uint32_t model_id, const uint32_t data_index, const uint32_t result_code,
                                       std::vector<gert::Tensor> &outputs) {
  GELOGI("[GraphManager] run graph async call back, modelId:%u, taskId:%u, resultCode:%u.",
         model_id, data_index, result_code);
  RunAsyncCallbackV2 callback;
  (void)sem_v2_.Pop(callback, 0U); // pop with no wait
  GE_CHECK_NOTNULL(callback);
  callback(result_code, outputs);
  return SUCCESS;
}
}  // namespace ge
