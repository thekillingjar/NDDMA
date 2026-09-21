/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "model_args_layout_planner.h"
#include "utils/extern_math_util.h"
#include "checker.h"
#include "task_args_refresh_type_classifier.h"

namespace ge {
constexpr uint32_t kTaskArgsLayoutResultSize = 2;
constexpr uint32_t kPartitionAlignLen = 64U;
constexpr size_t kUBAlignedLen = 32U;

namespace {
constexpr std::array<const char *, static_cast<size_t>(AddrUseFor::kEnd) + 1U> g_addr_use_for_str = {
    "Addr use for args",     // kAddrUseForArgs
    "Addr use for persistent workspace",  // kAddrUseForPersistentWorkspace
    "Unknown"};

const char_t *GetAddrUseForStr(AddrUseFor addr_use_for) {
  size_t i = static_cast<size_t>(addr_use_for);
  if (i > static_cast<size_t>(AddrUseFor::kEnd)) {
    i = static_cast<size_t>(AddrUseFor::kEnd);
  }
  return g_addr_use_for_str[i];
}
}

ModelArgsLayoutPlanner::ModelArgsLayoutPlanner(
    const vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> &task_indexes_to_refresh_type,
    const vector<TaskRunParam> &task_indexes_to_param, uint64_t host_input_size)
    : need_debug_log_(IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)),
      task_indexes_to_refresh_type_(task_indexes_to_refresh_type),
      task_indexes_to_param_(task_indexes_to_param),
      host_input_size_(host_input_size) {}
Status ModelArgsLayoutPlanner::Plan(ModelArgsLayoutPlannedResult &ret, const AddrUseFor &addr_use_for) const {
  GE_ASSERT_SUCCESS(PlanPartitions(ret.placements_to_partitions_to_len, addr_use_for));
  if (need_debug_log_) {
    GE_ASSERT_SUCCESS(LogPartitionLengths(ret.placements_to_partitions_to_len, GetAddrUseForStr(addr_use_for)));
  }

  MergePolicy merge_policy;
  GE_ASSERT_SUCCESS(MergePlacements(ret.placements_to_partitions_to_len, merge_policy));
  if (need_debug_log_) {
    GE_ASSERT_SUCCESS(LogPartitionLengths(ret.placements_to_partitions_to_len, "After merge partitions, "));
  }

  GE_ASSERT_SUCCESS(AlignPartitions(ret.placements_to_partitions_to_len, ret.placements_to_partitions_to_align_offset));
  if (need_debug_log_) {
    GE_ASSERT_SUCCESS(LogPartitionLengths(ret.placements_to_partitions_to_len, "After align partitions, "));
  }

  GE_ASSERT_SUCCESS(PlanTasks(ret.placements_to_partitions_to_len, ret.placements_to_partitions_to_align_offset,
                              merge_policy, ret.task_indexes_to_arg_results, addr_use_for));
  return SUCCESS;
}

UpdateTriggerType ModelArgsLayoutPlanner::GetTriggerTypeByTaskIndex(size_t index) const {
  if (static_cast<bool>(task_indexes_to_refresh_type_[index].task_refresh_type &
                        TaskArgsRefreshTypeClassifier::kRefreshByModelIo)) {
    return UpdateTriggerType::kTriggerByFmAndIo;
  }
  if (static_cast<bool>(task_indexes_to_refresh_type_[index].task_refresh_type &
                        TaskArgsRefreshTypeClassifier::kRefreshByFm)) {
    return UpdateTriggerType::kTriggerByFm;
  }
  return UpdateTriggerType::kNoNeedUpdate;
}

Status ModelArgsLayoutPlanner::PlanPartitions(PlacementsToPartitionsToLenType &placements_to_partitions_to_len,
    const AddrUseFor &addr_use_for) const {
  placements_to_partitions_to_len = {};
  for (size_t i = 0U; i < task_indexes_to_param_.size(); ++i) {
    auto &args_descs_to_plan = (addr_use_for == AddrUseFor::kAddrUseForArgs) ?
        task_indexes_to_param_[i].args_descs : task_indexes_to_param_[i].persistent_workspace_descs;
    for (const auto &args_desc : args_descs_to_plan) {
      const size_t pi = static_cast<size_t>(args_desc.placement);
      const size_t tti = static_cast<size_t>(GetTriggerTypeByTaskIndex(i));
      GE_ASSERT_TRUE(!AddOverflow(placements_to_partitions_to_len[pi][tti], args_desc.args_len,
                                  placements_to_partitions_to_len[pi][tti]));
    }
  }

  // 添加随路拷贝的partition
  if (host_input_size_ > 0) {
    placements_to_partitions_to_len[static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)][static_cast<size_t>(UpdateTriggerType::KTriggerByHostInput)] =
      host_input_size_;
  }
  return SUCCESS;
}

Status ModelArgsLayoutPlanner::MergePlacements(PlacementsToPartitionsToLenType &placements_to_partitions_to_len,
                                               MergePolicy &merge_policy) const {
  merge_policy = {
      ArgsPlacement::kEnd,              // hbm不做合并
      ArgsPlacement::kEnd,              // ts不做合并
      ArgsPlacement::kArgsPlacementHbm, // sqe合并到hbm中
      ArgsPlacement::kEnd               // host_svm不做合并
  };

  for (size_t src_pi = 0U; src_pi < static_cast<size_t>(ArgsPlacement::kEnd); ++src_pi) {
    if (merge_policy[src_pi] == ArgsPlacement::kEnd) {
      continue;
    }
    const size_t dst_pi = static_cast<size_t>(merge_policy[src_pi]);
    for (size_t i = 0U; i < static_cast<size_t>(UpdateTriggerType::kEnd); ++i) {
      GE_ASSERT_TRUE(!AddOverflow(placements_to_partitions_to_len[dst_pi][i],
                                  placements_to_partitions_to_len[src_pi][i],
                                  placements_to_partitions_to_len[dst_pi][i]));
      placements_to_partitions_to_len[src_pi][i] = 0;
    }
  }

  return SUCCESS;
}

Status ModelArgsLayoutPlanner::AlignPartitions(PlacementsToPartitionsToLenType &placements_to_partitions_to_len,
    PlacementsToPartitionsToLenType &placements_to_partitions_to_align_offset) const {
  placements_to_partitions_to_align_offset = {};
  for (size_t pi = 0U; pi < static_cast<size_t>(ArgsPlacement::kEnd); ++pi) {
    for (size_t tti = 0U; tti < static_cast<size_t>(UpdateTriggerType::kEnd); ++tti) {
        if (placements_to_partitions_to_len[pi][tti] == 0) {
          placements_to_partitions_to_align_offset[pi][tti] = 0;
          continue;
        }

        /* partition长度需要64字节对齐，因为memcpy_async的地址需要64字节对齐
         * 芯片约束，从GM搬运数据到UB时，最低拷贝数据量为32字节
         * 当partition尾部task的args table实际可访问的device内存长度小于32时，GM搬运该task的args table到UB时会产生读越界错误
         * 修改方案为每个partion都补上32字节后再做64字节对齐
         */
        const int64_t align_size =
          ge::MemSizeAlign(static_cast<size_t>(placements_to_partitions_to_len[pi][tti]) + kUBAlignedLen,
                           kPartitionAlignLen);
        placements_to_partitions_to_align_offset[pi][tti] = align_size - placements_to_partitions_to_len[pi][tti];
        placements_to_partitions_to_len[pi][tti] = align_size;
    }
  }

  return SUCCESS;
}

Status ModelArgsLayoutPlanner::PlanTasks(
    const PlacementsToPartitionsToLenType &placements_to_partitions_to_len,
    const PlacementsToPartitionsToLenType &placements_to_partitions_to_align_offset,
    const MergePolicy &merge_policy,
    std::vector<SmallVector<TaskArgsLayoutResult, kTaskArgsLayoutResultSize>> &task_indexes_to_arg_results,
    const AddrUseFor &addr_use_for) const {
  PlacementsToPartitionsToLenType placements_to_partitions_to_offset;
  for (size_t i = 0U; i < static_cast<size_t>(ArgsPlacement::kEnd); ++i) {
    placements_to_partitions_to_offset[i][0U] = 0;
    for (size_t j = 1U; j < static_cast<size_t>(UpdateTriggerType::kEnd); ++j) {
      GE_ASSERT_TRUE(!AddOverflow(placements_to_partitions_to_offset[i][j - 1U],
                                  placements_to_partitions_to_len[i][j - 1U],
                                  placements_to_partitions_to_offset[i][j]));
    }
  }

  task_indexes_to_arg_results.resize(task_indexes_to_param_.size());
  for (size_t i = 0U; i < task_indexes_to_param_.size(); ++i) {
    auto &args_descs_to_plan = (addr_use_for == AddrUseFor::kAddrUseForArgs) ?
        task_indexes_to_param_[i].args_descs : task_indexes_to_param_[i].persistent_workspace_descs;
    for (const auto &args_desc : args_descs_to_plan) {
      // args_desc.args_len == 0 是允许的，自管理args的task会返回一个长度为0的args_desc
      // 对于args_len == 0的args，只有trigger type是有效的，其他字段随心生成就可以了
      auto pi = static_cast<size_t>(args_desc.placement);
      if (merge_policy[pi] != ArgsPlacement::kEnd) {
        pi = static_cast<size_t>(merge_policy[pi]);
      }
      const auto trigger_type = GetTriggerTypeByTaskIndex(i);
      auto &offset = placements_to_partitions_to_offset[pi][static_cast<size_t>(trigger_type)];
      task_indexes_to_arg_results[i].emplace_back(
          TaskArgsLayoutResult{static_cast<ArgsPlacement>(pi), trigger_type, offset});

      if (need_debug_log_) {
        DebugLogPlanResult(i, args_desc, pi, trigger_type, offset);
      }

      GE_ASSERT_TRUE(!AddOverflow(offset, args_desc.args_len, offset));
    }
  }

  // offset 增加随路拷贝的长度
  if (host_input_size_ > 0) {
    auto &offset =
      placements_to_partitions_to_offset[static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)][static_cast<size_t>(UpdateTriggerType::KTriggerByHostInput)];
    GE_ASSERT_TRUE(!AddOverflow(offset, host_input_size_, offset));
  }

  // 校验，确保没有意外的踩踏
  for (size_t i = 0U; i < static_cast<size_t>(ArgsPlacement::kEnd); ++i) {
    GE_ASSERT_EQ(placements_to_partitions_to_offset[i][0U] + placements_to_partitions_to_align_offset[i][0U],
                 placements_to_partitions_to_len[i][0U]);
    for (size_t j = 1U; j < static_cast<size_t>(UpdateTriggerType::kEnd); ++j) {
      GE_ASSERT_EQ(placements_to_partitions_to_offset[i][j] + placements_to_partitions_to_align_offset[i][j] - (
                   placements_to_partitions_to_offset[i][j - 1U] + placements_to_partitions_to_align_offset[i][j - 1U]),
                   placements_to_partitions_to_len[i][j]);
    }
  }

  return SUCCESS;
}
Status ModelArgsLayoutPlanner::LogPartitionLengths(
    const PlacementsToPartitionsToLenType &placements_to_partitions_to_len, const char_t *desc) {
  for (size_t i = 0u; i < static_cast<size_t>(ArgsPlacement::kEnd); ++i) {
    std::stringstream ss;
    ss << desc << "Model args placement " << GetArgsPlacementStr(static_cast<ArgsPlacement>(i));
    int64_t total_len = 0;
    for (size_t j = 0u; j < static_cast<size_t>(UpdateTriggerType::kEnd); ++j) {
      GE_ASSERT_TRUE(!AddOverflow(total_len, placements_to_partitions_to_len[i][j], total_len));
    }
    ss << " total length " << total_len << ", partition lengths ";
    for (size_t j = 0u; j < static_cast<size_t>(UpdateTriggerType::kEnd); ++j) {
      ss << placements_to_partitions_to_len[i][j] << '/';
    }
    GELOGD("%s", ss.str().c_str());
  }
  return SUCCESS;
}
void ModelArgsLayoutPlanner::DebugLogPlanResult(size_t task_index, const TaskArgsDesc &args_desc,
                                                size_t placement_index, UpdateTriggerType trigger_type,
                                                int64_t offset) {
  if (placement_index == static_cast<size_t>(args_desc.placement)) {
    GELOGD("Task index %zu, arg placement %s, trigger type %d, offset %lld, len %lld", task_index,
           GetArgsPlacementStr(args_desc.placement), static_cast<int32_t>(trigger_type), offset, args_desc.args_len);
  } else {
    GELOGD("Task index %zu, arg placement %s -> %s, trigger type %d, offset %lld, len %lld", task_index,
           GetArgsPlacementStr(args_desc.placement), GetArgsPlacementStr(static_cast<ArgsPlacement>(placement_index)),
           static_cast<int32_t>(trigger_type), offset, args_desc.args_len);
  }
}
}  // namespace ge
