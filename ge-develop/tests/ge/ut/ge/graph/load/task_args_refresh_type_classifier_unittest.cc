/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_args_refresh_type_classifier.h"
#include <gtest/gtest.h>

#include "faker/task_run_param_faker.h"
#include "faker/model_task_run_params_faker.h"
#include "stub/gert_runtime_stub.h"
#include "common/share_graph.h"
#include "task_args_refresh_type_classifier_common_header.h"
namespace ge {
namespace {
class FakedAddrs {
 public:
  /*
   * 0x0 ~ 0x10000000-1:        weight
   * 0x10000000 ~ 0x20000000-1: feature map
   * 0x20000000 ~ 0x30000000-1: model io
   */
  FakedAddrs() {
    for (uint64_t i = 0; i < 10; ++i) {
      addrs_to_mat_[std::pair<uint64_t, uint64_t>(kWeightMemType, W(i))] = MemoryAppType::kMemoryTypeFix;
      addrs_to_mat_[std::pair<uint64_t, uint64_t>(RT_MEMORY_HBM, Fm(i))] = MemoryAppType::kMemoryTypeFeatureMap;
      addrs_to_mat_[std::pair<uint64_t, uint64_t>(RT_MEMORY_HBM, Io(i))] = MemoryAppType::kMemoryTypeModelIo;
    }
  }

  const std::map<std::pair<uint64_t, uint64_t>, MemoryAppType> &Get() const {
    return addrs_to_mat_;
  }

  static uint64_t W(uint64_t i) {
    return i * 0x100;
  }

  static uint64_t Fm(uint64_t i) {
    return i * 0x100 + 0x10000000;
  }
  static uint64_t Io(uint64_t i) {
    return i * 0x100 + 0x20000000;
  }

 private:
  std::map<std::pair<uint64_t, uint64_t>, MemoryAppType> addrs_to_mat_;
};
}  // namespace
class TaskArgsRefreshTypeClassifierUT : public testing::Test {
 public:
  static constexpr uint64_t kNothing = 0UL;
  static constexpr uint64_t kFmAndModelIo =
      TaskArgsRefreshTypeClassifier::kRefreshByFm | TaskArgsRefreshTypeClassifier::kRefreshByModelIo;
  static constexpr uint64_t kModelIo = TaskArgsRefreshTypeClassifier::kRefreshByModelIo;
  static constexpr uint64_t kFm = TaskArgsRefreshTypeClassifier::kRefreshByFm;
};
TEST_F(TaskArgsRefreshTypeClassifierUT, ClassifyMultiTasks_Refresh_IoFmAndSupportRefresh) {
  FakedAddrs fa;
  std::vector<TaskRunParam> params;
  params.emplace_back(TaskRunParamFaker().RefreshHbmInput(fa.Io(0)).RefreshHbmOutput(fa.Fm(0)).Build());
  auto tn_map = TaskNodeMapFaker(params).RangeOneOneMap(0, 1).Build();

  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts;
  TaskArgsRefreshTypeClassifier::FixedAddrs fixed_addrs;
  ASSERT_EQ(TaskArgsRefreshTypeClassifier(tn_map, fa.Get(), true).ClassifyMultiTasks(params, rts, fixed_addrs),
            SUCCESS);

  ASSERT_EQ(rts.size(), 1U);
  ASSERT_TRUE(fixed_addrs.empty());

  EXPECT_EQ(rts[0].task_refresh_type, kFmAndModelIo);
  EXPECT_EQ(rts[0].input_refresh_types, std::vector<uint64_t>{kModelIo});
  EXPECT_EQ(rts[0].output_refresh_types, std::vector<uint64_t>{kFm});
  EXPECT_EQ(rts[0].workspace_refresh_types, std::vector<uint64_t>{});
}
TEST_F(TaskArgsRefreshTypeClassifierUT, ClassifyMultiTasks_Refresh_IoAndSupportRefresh) {
  FakedAddrs fa;
  std::vector<TaskRunParam> params;
  params.emplace_back(TaskRunParamFaker().RefreshHbmInput(fa.Io(0)).RefreshHbmOutput(fa.Io(1)).Build());
  auto tn_map = TaskNodeMapFaker(params).RangeOneOneMap(0, 1).Build();

  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts;
  TaskArgsRefreshTypeClassifier::FixedAddrs fixed_addrs;
  ASSERT_EQ(TaskArgsRefreshTypeClassifier(tn_map, fa.Get(), true).ClassifyMultiTasks(params, rts, fixed_addrs),
            SUCCESS);

  ASSERT_EQ(rts.size(), 1U);
  ASSERT_TRUE(fixed_addrs.empty());

  EXPECT_EQ(rts[0].task_refresh_type, kModelIo);
  EXPECT_EQ(rts[0].input_refresh_types, std::vector<uint64_t>{kModelIo});
  EXPECT_EQ(rts[0].output_refresh_types, std::vector<uint64_t>{kModelIo});
  EXPECT_EQ(rts[0].workspace_refresh_types, std::vector<uint64_t>{});
}
TEST_F(TaskArgsRefreshTypeClassifierUT, ClassifyMultiTasks_Refresh_FmAndSupportRefresh) {
  FakedAddrs fa;
  std::vector<TaskRunParam> params;
  params.emplace_back(TaskRunParamFaker().RefreshHbmInput(fa.Fm(0)).RefreshHbmOutput(fa.Fm(1)).Build());
  auto tn_map = TaskNodeMapFaker(params).RangeOneOneMap(0, 1).Build();

  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts;
  TaskArgsRefreshTypeClassifier::FixedAddrs fixed_addrs;
  ASSERT_EQ(TaskArgsRefreshTypeClassifier(tn_map, fa.Get(), true).ClassifyMultiTasks(params, rts, fixed_addrs),
            SUCCESS);

  ASSERT_EQ(rts.size(), 1U);
  ASSERT_TRUE(fixed_addrs.empty());

  EXPECT_EQ(rts[0].task_refresh_type, kFm);
  EXPECT_EQ(rts[0].input_refresh_types, std::vector<uint64_t>{kFm});
  EXPECT_EQ(rts[0].output_refresh_types, std::vector<uint64_t>{kFm});
  EXPECT_EQ(rts[0].workspace_refresh_types, std::vector<uint64_t>{});
}
TEST_F(TaskArgsRefreshTypeClassifierUT, ClassifyMultiTasks_NoRefresh_NoNeedAndNoSupportRefresh) {
  FakedAddrs fa;
  std::vector<TaskRunParam> params;
  params.emplace_back(
      TaskRunParamFaker().Input(fa.W(1), kWeightMemType, false).Output(fa.Fm(0), RT_MEMORY_HBM, false).Build());
  auto tn_map = TaskNodeMapFaker(params).RangeOneOneMap(0, 1).Build();

  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts;
  TaskArgsRefreshTypeClassifier::FixedAddrs fixed_addrs;
  ASSERT_EQ(TaskArgsRefreshTypeClassifier(tn_map, fa.Get(), false).ClassifyMultiTasks(params, rts, fixed_addrs),
            SUCCESS);

  ASSERT_EQ(rts.size(), 1U);
  ASSERT_TRUE(fixed_addrs.empty());

  EXPECT_EQ(rts[0].task_refresh_type, kNothing);
  EXPECT_EQ(rts[0].input_refresh_types, std::vector<uint64_t>{kNothing});
  EXPECT_EQ(rts[0].output_refresh_types, std::vector<uint64_t>{kNothing});
  EXPECT_EQ(rts[0].workspace_refresh_types, std::vector<uint64_t>{});
}
TEST_F(TaskArgsRefreshTypeClassifierUT, ClassifyMultiTasks_NoRefresh_NoNeedAndSupportRefresh) {
  FakedAddrs fa;
  std::vector<TaskRunParam> params;
  params.emplace_back(
      TaskRunParamFaker().Input(fa.W(1), kWeightMemType, true).Output(fa.W(0), kWeightMemType, true).Build());
  auto tn_map = TaskNodeMapFaker(params).RangeOneOneMap(0, 1).Build();

  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts;
  TaskArgsRefreshTypeClassifier::FixedAddrs fixed_addrs;
  ASSERT_EQ(TaskArgsRefreshTypeClassifier(tn_map, fa.Get(), true).ClassifyMultiTasks(params, rts, fixed_addrs),
            SUCCESS);

  ASSERT_EQ(rts.size(), 1U);
  ASSERT_TRUE(fixed_addrs.empty());

  ASSERT_EQ(rts[0].task_refresh_type, kNothing);
  EXPECT_EQ(rts[0].input_refresh_types, std::vector<uint64_t>{kNothing});
  EXPECT_EQ(rts[0].output_refresh_types, std::vector<uint64_t>{kNothing});
  EXPECT_EQ(rts[0].workspace_refresh_types, std::vector<uint64_t>{});
}
/*
 * 校验点：
 * 1. dsa节点被推导为fixed
 * 2. 与dsa节点直接相连的identity被推导为fixed，并且地址正确
 * 3. 与dsa直接相连的identity重刷task级的refresh type，并且刷新正确
 * 4.
 */
TEST_F(TaskArgsRefreshTypeClassifierUT, ClassifyMultiTasks_FixedAddrAndInferOk_NeedAndNoSupportRefresh) {
  auto graph = gert::ShareGraph::FixedAddrNodeGraph();
  FakedAddrs fa;
  auto params_and_tn_map =
      ModelTaskRunParamsFaker(graph)
          .Param("id1", std::move(TaskRunParamFaker()  //
                                      .RefreshWeightInput(fa.W(0))
                                      .RefreshHbmWorkspace(fa.Fm(0))
                                      .RefreshHbmOutput(fa.Fm(1))))
          .Param("id2", std::move(TaskRunParamFaker().RefreshWeightInput(fa.W(0)).RefreshHbmOutput(fa.Fm(3))))
          .Param("DsaNode", std::move(TaskRunParamFaker()
                                          .Input(fa.Fm(1), RT_MEMORY_HBM, false)
                                          .Input(fa.Fm(3), RT_MEMORY_HBM, false)
                                          .Output(fa.Fm(4), RT_MEMORY_HBM, false)
                                          .Output(fa.Fm(5), RT_MEMORY_HBM, false)
                                          .Workspace(fa.Fm(6), RT_MEMORY_HBM, false)
                                          .Workspace(fa.Fm(7), RT_MEMORY_HBM, false)))
          .Param("id3", std::move(TaskRunParamFaker().RefreshHbmInput(fa.Fm(4)).RefreshHbmOutput(fa.Io(0))))
          .Param("id4", std::move(TaskRunParamFaker()  //
                                      .RefreshHbmInput(fa.Fm(5))
                                      .RefreshHbmWorkspace(fa.Fm(0))
                                      .RefreshHbmOutput(fa.Io(1))))
          .Build();
  auto &params = params_and_tn_map.params;
  auto &tn_map = params_and_tn_map.tnm;

  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts;
  TaskArgsRefreshTypeClassifier::FixedAddrs fixed_addrs;
  ASSERT_EQ(TaskArgsRefreshTypeClassifier(tn_map, fa.Get(), true).ClassifyMultiTasks(params, rts, fixed_addrs),
            SUCCESS);

  ASSERT_EQ(rts.size(), 5U);
  ASSERT_EQ(fixed_addrs.size(), 6U);

  auto dsa_ti = params_and_tn_map.node_names_to_task_index.at("DsaNode");
  auto &dsa_rts = rts.at(dsa_ti);
  ASSERT_EQ(dsa_rts.task_refresh_type, kNothing);
  ASSERT_EQ(dsa_rts.input_refresh_types, (std::vector<uint64_t>{kNothing, kNothing}));
  ASSERT_EQ(dsa_rts.output_refresh_types, (std::vector<uint64_t>{kNothing, kNothing}));
  ASSERT_EQ(dsa_rts.workspace_refresh_types, (std::vector<uint64_t>{kNothing, kNothing}));

  auto id1_ti = params_and_tn_map.node_names_to_task_index.at("id1");
  auto &id1_rts = rts.at(id1_ti);
  ASSERT_EQ(id1_rts.task_refresh_type, kFm);
  ASSERT_EQ(id1_rts.input_refresh_types, (std::vector<uint64_t>{kNothing}));
  ASSERT_EQ(id1_rts.output_refresh_types, (std::vector<uint64_t>{kNothing}));
  ASSERT_EQ(id1_rts.workspace_refresh_types, (std::vector<uint64_t>{kFm}));

  auto id2_ti = params_and_tn_map.node_names_to_task_index.at("id2");
  auto &id2_rts = rts.at(id2_ti);
  ASSERT_EQ(id2_rts.task_refresh_type, kNothing);
  ASSERT_EQ(id2_rts.input_refresh_types, (std::vector<uint64_t>{kNothing}));
  ASSERT_EQ(id2_rts.output_refresh_types, (std::vector<uint64_t>{kNothing}));
  ASSERT_EQ(id2_rts.workspace_refresh_types, (std::vector<uint64_t>{}));

  auto id3_ti = params_and_tn_map.node_names_to_task_index.at("id3");
  auto &id3_rts = rts.at(id3_ti);
  EXPECT_EQ(id3_rts.task_refresh_type, kModelIo);
  EXPECT_EQ(id3_rts.input_refresh_types, (std::vector<uint64_t>{kNothing}));
  EXPECT_EQ(id3_rts.output_refresh_types, (std::vector<uint64_t>{kModelIo}));
  EXPECT_EQ(id3_rts.workspace_refresh_types, (std::vector<uint64_t>{}));

  auto id4_ti = params_and_tn_map.node_names_to_task_index.at("id4");
  auto &id4_rts = rts.at(id4_ti);
  ASSERT_EQ(id4_rts.task_refresh_type, kFmAndModelIo);
  ASSERT_EQ(id4_rts.input_refresh_types, (std::vector<uint64_t>{kNothing}));
  ASSERT_EQ(id4_rts.output_refresh_types, (std::vector<uint64_t>{kModelIo}));
  ASSERT_EQ(id4_rts.workspace_refresh_types, (std::vector<uint64_t>{kFm}));

  std::set<SmallVector<TaskArgsRefreshTypeClassifier::TaskFixedAddr, 2>> tfas;
  tfas.insert(
      {{dsa_ti, 0, TaskArgsRefreshTypeClassifier::kInput}, {id1_ti, 0, TaskArgsRefreshTypeClassifier::kOutput}});
  tfas.insert(
      {{dsa_ti, 1, TaskArgsRefreshTypeClassifier::kInput}, {id2_ti, 0, TaskArgsRefreshTypeClassifier::kOutput}});

  tfas.insert(
      {{dsa_ti, 0, TaskArgsRefreshTypeClassifier::kOutput}, {id3_ti, 0, TaskArgsRefreshTypeClassifier::kInput}});
  tfas.insert(
      {{dsa_ti, 1, TaskArgsRefreshTypeClassifier::kOutput}, {id4_ti, 0, TaskArgsRefreshTypeClassifier::kInput}});

  tfas.insert({{dsa_ti, 0, TaskArgsRefreshTypeClassifier::kWorkspace}});
  tfas.insert({{dsa_ti, 1, TaskArgsRefreshTypeClassifier::kWorkspace}});

  for (const auto &same_fixed_addrs : fixed_addrs) {
    ASSERT_EQ(tfas.count(same_fixed_addrs), 1U);
  }
}
TEST_F(TaskArgsRefreshTypeClassifierUT, ClassifyMultiTasks_NoRefresh_NoNeedFmAndSupportRefresh) {
  FakedAddrs fa;
  std::vector<TaskRunParam> params;
  params.emplace_back(TaskRunParamFaker()
                          .Input(fa.Fm(0), RT_MEMORY_HBM, false)
                          .Output(fa.Fm(1), RT_MEMORY_HBM, false)
                          .Workspace(fa.Fm(2), RT_MEMORY_HBM, false)
                          .Build());
  auto tn_map = TaskNodeMapFaker(params).RangeOneOneMap(0, params.size()).Build();

  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts;
  TaskArgsRefreshTypeClassifier::FixedAddrs fixed_addrs;
  ASSERT_EQ(TaskArgsRefreshTypeClassifier(tn_map, fa.Get(), false).ClassifyMultiTasks(params, rts, fixed_addrs),
            SUCCESS);

  ASSERT_EQ(rts.size(), 1U);
  ASSERT_TRUE(fixed_addrs.empty());

  ASSERT_EQ(rts[0].task_refresh_type, kNothing);
  EXPECT_EQ(rts[0].input_refresh_types, std::vector<uint64_t>{kNothing});
  EXPECT_EQ(rts[0].output_refresh_types, std::vector<uint64_t>{kNothing});
  EXPECT_EQ(rts[0].workspace_refresh_types, std::vector<uint64_t>{kNothing});
}
TEST_F(TaskArgsRefreshTypeClassifierUT, ClassifyMultiTasks_PriorityRefresh_MixNeedsAndSupportRefresh) {
  FakedAddrs fa;
  std::vector<TaskRunParam> params;
  params.emplace_back(TaskRunParamFaker()
                          .Input(fa.Fm(0), RT_MEMORY_HBM, false)
                          .Output(fa.Io(1), RT_MEMORY_HBM, true)
                          .Workspace(fa.Fm(2), RT_MEMORY_HBM, true)
                          .Build());
  auto tn_map = TaskNodeMapFaker(params).RangeOneOneMap(0, params.size()).Build();

  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts;
  TaskArgsRefreshTypeClassifier::FixedAddrs fixed_addrs;
  ASSERT_EQ(TaskArgsRefreshTypeClassifier(tn_map, fa.Get(), false).ClassifyMultiTasks(params, rts, fixed_addrs),
            SUCCESS);

  ASSERT_EQ(rts.size(), 1U);
  ASSERT_TRUE(fixed_addrs.empty());

  ASSERT_EQ(rts[0].task_refresh_type, kModelIo);
  EXPECT_EQ(rts[0].input_refresh_types, std::vector<uint64_t>{kNothing});
  EXPECT_EQ(rts[0].output_refresh_types, std::vector<uint64_t>{kModelIo});
  EXPECT_EQ(rts[0].workspace_refresh_types, std::vector<uint64_t>{kNothing});
}
TEST_F(TaskArgsRefreshTypeClassifierUT, ClassifyMultiTasks_Ok_MultipleParams) {
  auto graph = gert::ShareGraph::FixedAddrNodeGraph1();
  FakedAddrs fa;
  auto params_and_tn_map = ModelTaskRunParamsFaker(graph)
                               .Param("id1", std::move(TaskRunParamFaker()
                                                           .Input(fa.Io(0), RT_MEMORY_HBM, true)
                                                           .Output(fa.Fm(0), RT_MEMORY_HBM, false)
                                                           .Workspace(fa.Io(4), RT_MEMORY_HBM, false)))
                               .Param("DsaNode", std::move(TaskRunParamFaker()
                                                               .Input(fa.Fm(0), RT_MEMORY_HBM, false)
                                                               .Input(fa.W(0), kWeightMemType, false)
                                                               .Output(fa.Io(1), RT_MEMORY_HBM, true)
                                                               .Workspace(fa.Fm(2), RT_MEMORY_HBM, true)))
                               .Build();
  auto &params = params_and_tn_map.params;
  auto &tn_map = params_and_tn_map.tnm;

  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts;
  TaskArgsRefreshTypeClassifier::FixedAddrs fixed_addrs;
  ASSERT_EQ(TaskArgsRefreshTypeClassifier(tn_map, fa.Get(), false).ClassifyMultiTasks(params, rts, fixed_addrs),
            SUCCESS);

  ASSERT_EQ(rts.size(), 2U);
  ASSERT_EQ(fixed_addrs.size(), 1UL);

  auto id1_ti = params_and_tn_map.node_names_to_task_index.at("id1");
  ASSERT_EQ(rts[id1_ti].task_refresh_type, kModelIo);
  EXPECT_EQ(rts[id1_ti].input_refresh_types, std::vector<uint64_t>{kModelIo});
  EXPECT_EQ(rts[id1_ti].output_refresh_types, std::vector<uint64_t>{kNothing});
  EXPECT_EQ(rts[id1_ti].workspace_refresh_types, std::vector<uint64_t>{kNothing});

  auto dsa_ti = params_and_tn_map.node_names_to_task_index.at("DsaNode");
  ASSERT_EQ(rts[dsa_ti].task_refresh_type, kModelIo);
  EXPECT_EQ(rts[dsa_ti].input_refresh_types, (std::vector<uint64_t>{kNothing, kNothing}));
  EXPECT_EQ(rts[dsa_ti].output_refresh_types, std::vector<uint64_t>{kModelIo});
  EXPECT_EQ(rts[dsa_ti].workspace_refresh_types, std::vector<uint64_t>{kNothing});

  ASSERT_EQ(fixed_addrs.at(0).size(), 1UL);
  EXPECT_EQ(fixed_addrs.at(0).at(0).task_index, id1_ti);
  EXPECT_EQ(fixed_addrs.at(0).at(0).iow_index, 0);
  EXPECT_EQ(fixed_addrs.at(0).at(0).iow_index_type, TaskArgsRefreshTypeClassifier::kWorkspace);
}
TEST_F(TaskArgsRefreshTypeClassifierUT, ClassifyMultiTasks_NotSupportYet_FixedAddrHasPhonyConcatPeer) {
  auto graph = gert::ShareGraph::FixedAddrConnectToPhonyConcat();
  FakedAddrs fa;
  auto pam =  // params and map
      ModelTaskRunParamsFaker(graph)
          .Param("id1", std::move(TaskRunParamFaker().RefreshHbmInput(fa.Io(0)).RefreshHbmOutput(fa.Fm(0))))
          .Param("id2", std::move(TaskRunParamFaker().RefreshHbmInput(fa.Io(0)).RefreshHbmOutput(fa.Fm(1))))
          .Param("DsaNode",
                 std::move(
                     TaskRunParamFaker().Input(fa.Fm(0), RT_MEMORY_HBM, false).Output(fa.Fm(2), RT_MEMORY_HBM, false)))
          .Build();
  auto &params = pam.params;
  auto &tn_map = pam.tnm;

  gert::GertRuntimeStub runtime_stub;
  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts;
  TaskArgsRefreshTypeClassifier::FixedAddrs fixed_addrs;
  ASSERT_NE(TaskArgsRefreshTypeClassifier(tn_map, fa.Get(), true).ClassifyMultiTasks(params, rts, fixed_addrs),
            SUCCESS);
  EXPECT_TRUE(runtime_stub.GetSlogStub().FindErrorLogEndsWith(
                  "Failed to find peers by graph for node DsaNode input index 0, only support peer node type Identity, "
                  "but get PhonyConcat(pc1)") >= 0);
}
TEST_F(TaskArgsRefreshTypeClassifierUT, ClassifyMultiTasks_NotSupportYet_FixedAddrHasMultiPeers) {
  auto graph = gert::ShareGraph::FixedAddrConnectToMultiPeers();
  FakedAddrs fa;
  auto pam =  // params and map
      ModelTaskRunParamsFaker(graph)
          .Param("id1", std::move(TaskRunParamFaker().RefreshHbmInput(fa.Io(0)).RefreshHbmOutput(fa.Fm(0))))
          .Param("id2", std::move(TaskRunParamFaker().RefreshHbmInput(fa.Fm(1)).RefreshHbmOutput(fa.Io(1))))
          .Param("id3", std::move(TaskRunParamFaker().RefreshHbmInput(fa.Fm(1)).RefreshHbmOutput(fa.Io(2))))
          .Param("DsaNode",
                 std::move(
                     TaskRunParamFaker().Input(fa.Fm(0), RT_MEMORY_HBM, false).Output(fa.Fm(1), RT_MEMORY_HBM, false)))
          .Build();
  auto &params = pam.params;
  auto &tn_map = pam.tnm;

  gert::GertRuntimeStub runtime_stub;
  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> rts;
  TaskArgsRefreshTypeClassifier::FixedAddrs fixed_addrs;
  EXPECT_EQ(TaskArgsRefreshTypeClassifier(tn_map, fa.Get(), true).ClassifyMultiTasks(params, rts, fixed_addrs),
            SUCCESS);
}
}  // namespace ge
