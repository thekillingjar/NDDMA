/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/model_args_layout_planner.h"
#include <gtest/gtest.h>
#include "faker/task_run_param_faker.h"
#include "stub/gert_runtime_stub.h"
namespace ge {
namespace {
struct ArgOffsetAndLen {
  size_t task_index;
  size_t args_index;
  int64_t offset;
  int64_t len;
  UpdateTriggerType partition_type;
};
bool operator<(const ArgOffsetAndLen &lhs, const ArgOffsetAndLen &rhs) {
  return lhs.offset < rhs.offset;
}
class PlanRetChecker {
 public:
  PlanRetChecker(const vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> &task_indexes_to_rt,
                 const vector<TaskRunParam> &task_indexes_to_param)
      : task_indexes_to_rt_(task_indexes_to_rt), task_indexes_to_param_(task_indexes_to_param) {}
  PlanRetChecker &PartitionLen(ArgsPlacement placement, UpdateTriggerType partition_type,
                               int64_t len, int64_t align_offset) {
    expect_partition_[static_cast<int32_t>(placement)][static_cast<int32_t>(partition_type)] = len;
    expect_partition_align_offset[static_cast<int32_t>(placement)][static_cast<int32_t>(partition_type)] = align_offset;
    return *this;
  }
  std::string Check(const ModelArgsLayoutPlannedResult &ret) const {
    std::stringstream ss;
    bool success = true;

    success &= CheckPartition(ret, ss);
    success &= CheckPartitionAlignOffset(ret, ss);
    success &= CheckTaskIsPlanned(ret, ss);
    success &= CheckTaskPlacementAndPartition(ret, ss);
    success &= CheckTaskOffset(ret, ss);

    if (success) {
      return "success";
    } else {
      return ss.str();
    }
  }

  bool CheckPartition(const ModelArgsLayoutPlannedResult &ret, std::stringstream &ss) const {
    bool success = true;
    for (int32_t pli = 0; pli < static_cast<int32_t>(ArgsPlacement::kEnd); ++pli) {
      for (int32_t pai = 0; pai < static_cast<int32_t>(UpdateTriggerType::kEnd); ++pai) {
        if (ret.placements_to_partitions_to_len[pli][pai] != expect_partition_[pli][pai]) {
          ss << '[' << pli << ']' << '[' << pai << ']' << " partition len "
             << ret.placements_to_partitions_to_len[pli][pai] << ", expect len " << expect_partition_[pli][pai]
             << std::endl;
          success = false;
        }
      }
    }
    return success;
  }

  bool CheckPartitionAlignOffset(const ModelArgsLayoutPlannedResult &ret, std::stringstream &ss) const {
    bool success = true;
    for (int32_t pli = 0; pli < static_cast<int32_t>(ArgsPlacement::kEnd); ++pli) {
      for (int32_t pai = 0; pai < static_cast<int32_t>(UpdateTriggerType::kEnd); ++pai) {
        if (ret.placements_to_partitions_to_align_offset[pli][pai] != expect_partition_align_offset[pli][pai]) {
          ss << '[' << pli << ']' << '[' << pai << ']' << " partition align offset "
             << ret.placements_to_partitions_to_align_offset[pli][pai] << ", expect len "
             << expect_partition_align_offset[pli][pai] << std::endl;
          success = false;
        }
      }
    }
    return success;
  }

  bool CheckTaskIsPlanned(const ModelArgsLayoutPlannedResult &ret, std::stringstream &ss) const {
    for (size_t i = 0UL; i < task_indexes_to_param_.size(); ++i) {
      for (size_t j = 0UL; j < task_indexes_to_param_[i].args_descs.size(); ++j) {
        auto &args_desc = task_indexes_to_param_[i].args_descs[j];
        auto expect_placement = args_desc.placement == ArgsPlacement::kArgsPlacementSqe
                                    ? ArgsPlacement::kArgsPlacementHbm
                                    : args_desc.placement;
        if (ret.task_indexes_to_arg_results[i][j].placement != expect_placement) {
          ss << "Task index " << i << " " << j << "th args expect " << GetArgsPlacementStr(expect_placement)
             << " but get " << GetArgsPlacementStr(ret.task_indexes_to_arg_results[i][j].placement) << std::endl;
          return false;
        }
        if (ret.task_indexes_to_arg_results[i][j].trigger_type == UpdateTriggerType::kEnd) {
          ss << "Invalid trigger type, task index " << i << " " << j << "th args" << std::endl;
          return false;
        }
      }
    }

    return true;
  }

  bool CheckTaskPlacementAndPartition(const ModelArgsLayoutPlannedResult &ret, std::stringstream &ss) const {
    bool success = true;
    for (size_t i = 0UL; i < task_indexes_to_param_.size(); ++i) {
      const auto &param = task_indexes_to_param_[i];
      for (size_t j = 0UL; j < param.args_descs.size(); ++j) {
        const auto &args_desc = param.args_descs[j];
        if (args_desc.args_len == 0) {
          continue;
        }

        // check placement
        ArgsPlacement expect_placement = args_desc.placement;
        if (expect_placement == ArgsPlacement::kArgsPlacementSqe) {
          expect_placement = ArgsPlacement::kArgsPlacementHbm;
        }
        const auto &tret = ret.task_indexes_to_arg_results[i][j];
        if (expect_placement != tret.placement) {
          success = false;
          ss << "Task index " << i << "args " << j << " placement " << static_cast<int32_t>(tret.placement)
             << ", expect " << static_cast<int32_t>(expect_placement) << std::endl;
        }

        // checkpartition
        UpdateTriggerType expect_partition = UpdateTriggerType::kNoNeedUpdate;
        const auto &rt = task_indexes_to_rt_[i];
        if (rt.task_refresh_type & TaskArgsRefreshTypeClassifier::kRefreshByModelIo) {
          expect_partition = UpdateTriggerType::kTriggerByFmAndIo;
        } else if (rt.task_refresh_type & TaskArgsRefreshTypeClassifier::kRefreshByFm) {
          expect_partition = UpdateTriggerType::kTriggerByFm;
        }
        if (expect_partition != tret.trigger_type) {
          success = false;
          ss << "Task index " << i << "args " << j << " partition " << static_cast<int32_t>(tret.trigger_type)
             << ", expect " << static_cast<int32_t>(expect_partition) << std::endl;
        }
      }
    }
    return success;
  }

  bool CheckTaskOffset(const ModelArgsLayoutPlannedResult &ret, std::stringstream &ss) const {
    bool success = true;
    std::set<ArgOffsetAndLen> pis_to_offsets[static_cast<int32_t>(ArgsPlacement::kEnd) + 1];
    for (size_t i = 0UL; i < ret.task_indexes_to_arg_results.size(); ++i) {
      const auto &task_rets = ret.task_indexes_to_arg_results[i];
      for (size_t j = 0UL; j < task_rets.size(); ++j) {
        auto pi = static_cast<int32_t>(task_rets[j].placement);
        pis_to_offsets[pi].insert(
            {i, j, task_rets[j].offset, task_indexes_to_param_[i].args_descs[j].args_len, task_rets[j].trigger_type});
      }
    }

    for (int32_t pli = 0; pli < static_cast<int32_t>(ArgsPlacement::kEnd); ++pli) {
      int64_t expect_offset = 0;
      int64_t pais_to_task_len[static_cast<int32_t>(UpdateTriggerType::kEnd)] = {0};
      UpdateTriggerType last_partition = UpdateTriggerType::kNoNeedUpdate;
      for (const auto &ol : pis_to_offsets[pli]) {
        if (ol.partition_type < last_partition) {
          success = false;
          ss << "The order of partitions in placement " << pli << " are invalid. Task index " << ol.task_index
             << " partition index " << static_cast<int32_t>(ol.partition_type) << ", last "
             << static_cast<int32_t>(last_partition) << std::endl;
          break;
        }
        // partion 发生变化时，需要加上前一个partion的align offset
        if (ol.partition_type != last_partition) {
          expect_offset += ret.placements_to_partitions_to_align_offset[pli][static_cast<int32_t>(last_partition)];
          last_partition = ol.partition_type;
        }

        if (expect_offset != ol.offset) {
          success = false;
          ss << "Task index " << ol.task_index << " offset " << ol.offset << ", expect " << expect_offset << std::endl;
          break;
        }
        expect_offset += ol.len;
        pais_to_task_len[static_cast<int32_t>(ol.partition_type)] += ol.len;
      }
      // 添加最后一个partion的align offset
      expect_offset += ret.placements_to_partitions_to_align_offset[pli][static_cast<int32_t>(last_partition)];

      for (int32_t pai = 0; pai < static_cast<int32_t>(UpdateTriggerType::kEnd); ++pai) {
        if (pais_to_task_len[pai] + ret.placements_to_partitions_to_align_offset[pli][pai] !=
            ret.placements_to_partitions_to_len[pli][pai]) {
          ss << " placement [" << pli << "] partition " << pai << " len "
             << ret.placements_to_partitions_to_len[pli][pai] << ", not match all task args len "
             << pais_to_task_len[pai] << ", align offset "
             << ret.placements_to_partitions_to_align_offset[pli][pai] << std::endl;
          success = false;
        }
      }
      auto total_len = ret.placements_to_partitions_to_len[pli][0] + ret.placements_to_partitions_to_len[pli][1] +
                       ret.placements_to_partitions_to_len[pli][2];
      if (expect_offset != total_len) {
        ss << " placement [" << pli << "] len " << total_len << ", not match all task args len " << expect_offset
           << std::endl;
        success = false;
      }
    }
    return success;
  }

 private:
  int64_t expect_partition_[static_cast<int32_t>(ArgsPlacement::kEnd)][static_cast<int32_t>(UpdateTriggerType::kEnd)] =
      {{}, {}};
  int64_t expect_partition_align_offset[static_cast<int32_t>(ArgsPlacement::kEnd)][static_cast<int32_t>(UpdateTriggerType::kEnd)] =
      {{}, {}};
  const std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> &task_indexes_to_rt_;
  const std::vector<TaskRunParam> &task_indexes_to_param_;
};
}  // namespace
class ModelArgsLayoutPlannerUT : public testing::Test {
 public:
  static constexpr uint64_t kNothing = 0UL;
  static constexpr uint64_t kFmAndModelIo =
      TaskArgsRefreshTypeClassifier::kRefreshByFm | TaskArgsRefreshTypeClassifier::kRefreshByModelIo;
  static constexpr uint64_t kModelIo = TaskArgsRefreshTypeClassifier::kRefreshByModelIo;
  static constexpr uint64_t kFm = TaskArgsRefreshTypeClassifier::kRefreshByFm;
};
TEST_F(ModelArgsLayoutPlannerUT, Plan_Ok_OneTask) {
  std::vector<TaskRunParam> params;
  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts;
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x200)
                          .RefreshHbmWorkspace(0x300)
                          .HbmTaskArgsLen(32)
                          .Build());
  rts.push_back({kNothing, {kNothing}, {kNothing}, {kNothing}});

  ModelArgsLayoutPlannedResult result;
  ASSERT_EQ(ModelArgsLayoutPlanner(rts, params).Plan(result), SUCCESS);
  EXPECT_EQ(PlanRetChecker(rts, params)
                .PartitionLen(ArgsPlacement::kArgsPlacementHbm, UpdateTriggerType::kNoNeedUpdate, 64, 32)
                .Check(result),
            "success");
}
TEST_F(ModelArgsLayoutPlannerUT, Plan_Ok_OneTaskTwoPlacements) {
  std::vector<TaskRunParam> params;
  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts;
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x200)
                          .RefreshHbmWorkspace(0x300)
                          .HbmTaskArgsLen(32)
                          .SqeTaskArgsLen(8)
                          .Build());
  rts.push_back({kNothing, {kNothing}, {kNothing}, {kNothing}});

  ModelArgsLayoutPlannedResult result;
  ASSERT_EQ(ModelArgsLayoutPlanner(rts, params).Plan(result), SUCCESS);
  EXPECT_EQ(PlanRetChecker(rts, params)
                .PartitionLen(ArgsPlacement::kArgsPlacementHbm, UpdateTriggerType::kNoNeedUpdate, 128, 88)
                .Check(result),
            "success");
}
TEST_F(ModelArgsLayoutPlannerUT, Plan_Ok_MultiTasksOnePlacement) {
  std::vector<TaskRunParam> params;
  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts;
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x200)
                          .RefreshHbmWorkspace(0x300)
                          .HbmTaskArgsLen(32)
                          .Build());
  rts.push_back({kNothing, {kNothing}, {kNothing}, {kNothing}});
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x400)
                          .RefreshHbmWorkspace(0x300)
                          .HbmTaskArgsLen(32)
                          .Build());
  rts.push_back({kNothing, {kNothing}, {kNothing}, {kNothing}});
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x500)
                          .RefreshHbmWorkspace(0x300)
                          .HbmTaskArgsLen(16)
                          .Build());
  rts.push_back({kNothing, {kNothing}, {kNothing}, {kNothing}});

  ModelArgsLayoutPlannedResult result;
  ASSERT_EQ(ModelArgsLayoutPlanner(rts, params).Plan(result), SUCCESS);
  EXPECT_EQ(PlanRetChecker(rts, params)
                .PartitionLen(ArgsPlacement::kArgsPlacementHbm, UpdateTriggerType::kNoNeedUpdate, 128, 48)
                .Check(result),
            "success");
}
TEST_F(ModelArgsLayoutPlannerUT, Plan_Ok_MultiTasksOnePlacement1) {
  std::vector<TaskRunParam> params;
  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts;
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x200)
                          .RefreshHbmWorkspace(0x300)
                          .HbmTaskArgsLen(32)
                          .Build());
  rts.push_back({kFmAndModelIo, {kFmAndModelIo}, {kFmAndModelIo}, {kFmAndModelIo}});
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x400)
                          .RefreshHbmWorkspace(0x300)
                          .HbmTaskArgsLen(32)
                          .Build());
  rts.push_back({kFmAndModelIo, {kFmAndModelIo}, {kFmAndModelIo}, {kFmAndModelIo}});
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x500)
                          .RefreshHbmWorkspace(0x300)
                          .HbmTaskArgsLen(16)
                          .Build());
  rts.push_back({kFmAndModelIo, {kFmAndModelIo}, {kFmAndModelIo}, {kFmAndModelIo}});

  ModelArgsLayoutPlannedResult result;
  ASSERT_EQ(ModelArgsLayoutPlanner(rts, params).Plan(result), SUCCESS);
  EXPECT_EQ(PlanRetChecker(rts, params)
                .PartitionLen(ArgsPlacement::kArgsPlacementHbm, UpdateTriggerType::kTriggerByFmAndIo, 128, 48)
                .Check(result),
            "success");
}
TEST_F(ModelArgsLayoutPlannerUT, Plan_Ok_MultiTasksOnePlacementOnePartition) {
  std::vector<TaskRunParam> params;
  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts;
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x200)
                          .RefreshHbmWorkspace(0x300)
                          .HbmTaskArgsLen(64)
                          .Build());
  rts.push_back({kNothing, {kNothing}, {kNothing}, {kNothing}});
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x400)
                          .RefreshHbmWorkspace(0x300)
                          .HbmTaskArgsLen(32)
                          .Build());
  rts.push_back({kNothing, {kNothing}, {kNothing}, {kNothing}});
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x400)
                          .RefreshHbmWorkspace(0x300)
                          .HbmTaskArgsLen(16)
                          .Build());
  rts.push_back({kNothing, {kNothing}, {kNothing}, {kNothing}});
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x400)
                          .RefreshHbmWorkspace(0x300)
                          .HbmTaskArgsLen(8)
                          .Build());
  rts.push_back({kNothing, {kNothing}, {kNothing}, {kNothing}});

  ModelArgsLayoutPlannedResult result;
  ASSERT_EQ(ModelArgsLayoutPlanner(rts, params).Plan(result), SUCCESS);
  EXPECT_EQ(PlanRetChecker(rts, params)
                .PartitionLen(ArgsPlacement::kArgsPlacementHbm, UpdateTriggerType::kNoNeedUpdate, 192, 72)
                .Check(result),
            "success");
}
TEST_F(ModelArgsLayoutPlannerUT, Plan_Ok_MultiTasksMultiPlacements) {
  std::vector<TaskRunParam> params;
  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts;
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x200)
                          .RefreshHbmWorkspace(0x300)
                          .HbmTaskArgsLen(32)
                          .Build());
  rts.push_back({kFmAndModelIo, {kFmAndModelIo}, {kFmAndModelIo}, {kFmAndModelIo}});
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x400)
                          .RefreshHbmWorkspace(0x300)
                          .HbmTaskArgsLen(32)
                          .Build());
  rts.push_back({kFmAndModelIo, {kFmAndModelIo}, {kFmAndModelIo}, {kFmAndModelIo}});
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x500)
                          .RefreshHbmWorkspace(0x300)
                          .TsTaskArgsLen(16)
                          .Build());
  rts.push_back({kFmAndModelIo, {kFmAndModelIo}, {kFmAndModelIo}, {kFmAndModelIo}});
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x600)
                          .RefreshHbmWorkspace(0x300)
                          .TsTaskArgsLen(16)
                          .Build());
  rts.push_back({kFmAndModelIo, {kFmAndModelIo}, {kFmAndModelIo}, {kFmAndModelIo}});
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x700)
                          .RefreshHbmWorkspace(0x300)
                          .TsTaskArgsLen(32)
                          .Build());
  rts.push_back({kFmAndModelIo, {kFmAndModelIo}, {kFmAndModelIo}, {kFmAndModelIo}});

  ModelArgsLayoutPlannedResult result;
  ASSERT_EQ(ModelArgsLayoutPlanner(rts, params).Plan(result), SUCCESS);
  EXPECT_EQ(PlanRetChecker(rts, params)
                .PartitionLen(ArgsPlacement::kArgsPlacementHbm, UpdateTriggerType::kTriggerByFmAndIo, 128, 64)
                .PartitionLen(ArgsPlacement::kArgsPlacementTs, UpdateTriggerType::kTriggerByFmAndIo, 128, 64)
                .Check(result),
            "success");
}
TEST_F(ModelArgsLayoutPlannerUT, Plan_Ok_MultiTasksMultiPlacementsPartitions) {
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevel(DLOG_DEBUG);
  std::vector<TaskRunParam> params;
  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts;
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x200)
                          .RefreshHbmWorkspace(0x300)
                          .HbmTaskArgsLen(32)
                          .Build());
  rts.push_back({kFmAndModelIo, {kFmAndModelIo}, {kFmAndModelIo}, {kFmAndModelIo}});
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x400)
                          .RefreshHbmWorkspace(0x300)
                          .HbmTaskArgsLen(32)
                          .Build());
  rts.push_back({kFm, {kFm}, {kFm}, {kFm}});
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x500)
                          .RefreshHbmWorkspace(0x300)
                          .TsTaskArgsLen(16)
                          .Build());
  rts.push_back({kFmAndModelIo, {kFmAndModelIo}, {kFmAndModelIo}, {kFmAndModelIo}});
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x600)
                          .RefreshHbmWorkspace(0x300)
                          .TsTaskArgsLen(16)
                          .Build());
  rts.push_back({kFmAndModelIo, {kFmAndModelIo}, {kFmAndModelIo}, {kFmAndModelIo}});
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x700)
                          .RefreshHbmWorkspace(0x300)
                          .TsTaskArgsLen(32)
                          .Build());
  rts.push_back({kFmAndModelIo, {kFmAndModelIo}, {kFmAndModelIo}, {kFmAndModelIo}});

  ModelArgsLayoutPlannedResult result;
  ASSERT_EQ(ModelArgsLayoutPlanner(rts, params).Plan(result), SUCCESS);
  EXPECT_EQ(PlanRetChecker(rts, params)
                .PartitionLen(ArgsPlacement::kArgsPlacementHbm, UpdateTriggerType::kTriggerByFmAndIo, 64, 32)
                .PartitionLen(ArgsPlacement::kArgsPlacementHbm, UpdateTriggerType::kTriggerByFm, 64, 32)
                .PartitionLen(ArgsPlacement::kArgsPlacementTs, UpdateTriggerType::kTriggerByFmAndIo, 128, 64)
                .Check(result),
            "success");
  for (const auto &log : runtime_stub.GetSlogStub().GetLogs(DLOG_DEBUG)) {
    std::cout << log.content << std::endl;
  }
}
TEST_F(ModelArgsLayoutPlannerUT, Plan_SqeMergedToHbm_HasSqePlacement) {
  std::vector<TaskRunParam> params;
  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts;
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x200)
                          .RefreshHbmWorkspace(0x300)
                          .HbmTaskArgsLen(32)
                          .Build());
  rts.push_back({kNothing, {kNothing}, {kNothing}, {kNothing}});
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x400)
                          .RefreshHbmWorkspace(0x300)
                          .SqeTaskArgsLen(32)
                          .Build());
  rts.push_back({kFmAndModelIo, {kFmAndModelIo}, {kFmAndModelIo}, {kFmAndModelIo}});
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x500)
                          .RefreshHbmWorkspace(0x300)
                          .HbmTaskArgsLen(16)
                          .Build());
  rts.push_back({kFmAndModelIo, {kFmAndModelIo}, {kFmAndModelIo}, {kFmAndModelIo}});

  ModelArgsLayoutPlannedResult result;
  ASSERT_EQ(ModelArgsLayoutPlanner(rts, params).Plan(result), SUCCESS);
  EXPECT_EQ(PlanRetChecker(rts, params)
                .PartitionLen(ArgsPlacement::kArgsPlacementHbm, UpdateTriggerType::kNoNeedUpdate, 64, 32)
                .PartitionLen(ArgsPlacement::kArgsPlacementHbm, UpdateTriggerType::kTriggerByFmAndIo, 128, 80)
                .Check(result),
            "success");
}
TEST_F(ModelArgsLayoutPlannerUT, Plan_NoArgsGen_TaskHasNoArgReq) {
  std::vector<TaskRunParam> params;
  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts;
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x200)
                          .RefreshHbmWorkspace(0x300)
                          .HbmTaskArgsLen(32)
                          .Build());
  rts.push_back({kNothing, {kNothing}, {kNothing}, {kNothing}});
  params.emplace_back(
      TaskRunParamFaker().RefreshHbmInput(0x100).RefreshHbmOutput(0x400).RefreshHbmWorkspace(0x300).Build());
  rts.push_back({kFmAndModelIo, {kFmAndModelIo}, {kFmAndModelIo}, {kFmAndModelIo}});
  params.emplace_back(TaskRunParamFaker()
                          .RefreshHbmInput(0x100)
                          .RefreshHbmOutput(0x500)
                          .RefreshHbmWorkspace(0x300)
                          .TsTaskArgsLen(16)
                          .Build());
  rts.push_back({kFmAndModelIo, {kFmAndModelIo}, {kFmAndModelIo}, {kFmAndModelIo}});

  ModelArgsLayoutPlannedResult result;
  ASSERT_EQ(ModelArgsLayoutPlanner(rts, params).Plan(result), SUCCESS);
  EXPECT_EQ(PlanRetChecker(rts, params)
                .PartitionLen(ArgsPlacement::kArgsPlacementHbm, UpdateTriggerType::kNoNeedUpdate, 64, 32)
                .PartitionLen(ArgsPlacement::kArgsPlacementTs, UpdateTriggerType::kTriggerByFmAndIo, 64, 48)
                .Check(result),
            "success");
}
}  // namespace ge
