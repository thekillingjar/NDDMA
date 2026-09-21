/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_MODEL_ARGS_LAYOUT_PLANNER_H_
#define AIR_CXX_EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_MODEL_ARGS_LAYOUT_PLANNER_H_
#include <vector>
#include <cstdint>
#include "task_info/task_info.h"
#include "task_args_refresh_type_classifier.h"
#include "graph/small_vector.h"

namespace ge {
enum class UpdateTriggerType : int32_t {
  kNoNeedUpdate,      // 不需要被刷新
  kTriggerByFm,       // fm地址变化时，需要被刷新
  kTriggerByFmAndIo,  // fm、输入输出变化时，需要被刷新
  KTriggerByHostInput, // 存在host输入随路拷贝时，需要被刷新
  kEnd
};

enum class AddrUseFor : int32_t {
  kAddrUseForArgs = 0,
  kAddrUseForPersistentWorkspace = 1,
  kEnd = 2
};

struct TaskArgsLayoutResult {
  ArgsPlacement placement;
  UpdateTriggerType trigger_type;
  int64_t offset;  // offset独立在partition外，为相对本placement基地址的offset
};

/*
 * 按照placement分类，每个placement一段连续的args内存，在每一段args内，
 * 内存排布按照trigger type做分区，分区排布次序为：
 * | partition 0    | partition 1         | partition 2                 |
 * | 不需要刷新的args | fm变化触发刷新的args   | fm、modelio变化触发刷新的args  |
 *
 * 当一块partition的长度为0时，该partition可以不存在。
 * * 只要fm地址变化，刷新1和2
 * * 当fm地址不变，仅model io地址变化时，刷新2
 *
 * todo 当前只要model io触发的args都放在partition 2中，没有做细分排布，
 *      理论上可以将受到model io和fm同时触发的args放在p2的前段，仅被model io触发的args放在p2的后段
 *      这样可以在仅fm变化时，减少args的更新量。不过这个优化复杂度高，收益不大，因此暂时不做。
 */
static constexpr uint32_t kGeneralMaxArgsNumInOneTask = 2U;
using OneTaskArgsLayoutResult = SmallVector<TaskArgsLayoutResult, kGeneralMaxArgsNumInOneTask>;
using PlacementsToPartitionsToLenType = std::array<std::array<int64_t, static_cast<size_t>(UpdateTriggerType::kEnd)>,
                                                   static_cast<size_t>(ArgsPlacement::kEnd)>;
struct ModelArgsLayoutPlannedResult {
  PlacementsToPartitionsToLenType placements_to_partitions_to_len;
  PlacementsToPartitionsToLenType placements_to_partitions_to_align_offset;
  // todo 这里逻辑不太对，trigger type是task级别，不是args级别，因此应该把trigger type提到上一层，与args并列
  std::vector<OneTaskArgsLayoutResult> task_indexes_to_arg_results;
};

class ModelArgsLayoutPlanner {
 public:
  ModelArgsLayoutPlanner(
      const std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> &task_indexes_to_refresh_type,
      const std::vector<TaskRunParam> &task_indexes_to_param,
      uint64_t host_input_size = 0UL);
  Status Plan(ModelArgsLayoutPlannedResult &ret, const AddrUseFor &addr_use_for = AddrUseFor::kAddrUseForArgs) const;

 private:
  using MergePolicy = std::array<ArgsPlacement, static_cast<size_t>(ArgsPlacement::kEnd)>;
  // Placements Partitions Lengths

 private:
  UpdateTriggerType GetTriggerTypeByTaskIndex(size_t index) const;
  Status PlanPartitions(PlacementsToPartitionsToLenType &placements_to_partitions_to_len,
                        const AddrUseFor &addr_use_for = AddrUseFor::kAddrUseForArgs) const;

  /**
   * 合并分区
   * @param [IO] placements_to_partitions_to_len 原始计算出来的各placement在各trigger
   * type下的长度，函数中会将其修改为merge后的长度
   * @param [OUT] merge_policy merge_policy[i] < ArgsPlacement::kEnd时，代表i的placement要被合并到merge_policy[i]中
   * @return
   */
  Status MergePlacements(PlacementsToPartitionsToLenType &placements_to_partitions_to_len,
                         MergePolicy &merge_policy) const;

  Status AlignPartitions(PlacementsToPartitionsToLenType &placements_to_partitions_to_len,
                         PlacementsToPartitionsToLenType &placements_to_partitions_to_align_offset) const;

  Status PlanTasks(const PlacementsToPartitionsToLenType &placements_to_partitions_to_len,
                   const PlacementsToPartitionsToLenType &placements_to_partitions_to_align_offset,
                   const MergePolicy &merge_policy,
                   std::vector<OneTaskArgsLayoutResult> &task_indexes_to_arg_results,
                   const AddrUseFor &addr_use_for = AddrUseFor::kAddrUseForArgs) const;

  static Status LogPartitionLengths(const PlacementsToPartitionsToLenType &placements_to_partitions_to_len,
                                    const char_t *desc);
  static void DebugLogPlanResult(size_t task_index, const TaskArgsDesc &args_desc, size_t placement_index,
                                 UpdateTriggerType trigger_type, int64_t offset);

 private:
  bool need_debug_log_;
  const std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> &task_indexes_to_refresh_type_;
  const std::vector<TaskRunParam> &task_indexes_to_param_;
  uint64_t host_input_size_{0U};
};
}  // namespace ge

#endif  // AIR_CXX_EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_MODEL_ARGS_LAYOUT_PLANNER_H_
