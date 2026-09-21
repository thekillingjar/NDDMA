/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "lowering/model_converter.h"
#include "common/share_graph.h"
#include "faker/ge_model_builder.h"
#include "graph/utils/graph_utils.h"
#include "framework/common/helper/model_helper.h"
#include "faker/aicore_taskdef_faker.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "runtime/gert_api.h"
#include "common/pretty_table.h"
#include "faker/global_data_faker.h"
#include "lowering/graph_converter.h"
#include "common/helper/model_parser_base.h"
#include "common/tbe_handle_store/cust_aicpu_kernel_store.h"
#include "faker/aicpu_taskdef_faker.h"
#include "graph/utils/graph_dump_utils.h"
#include "graph/utils/execute_graph_utils.h"
#include "graph_metadef/common/ge_common/util.h"

namespace gert {
class LowerExeGraphFromModelUT : public testing::Test {
 protected:
  void TearDown() override {
    Test::TearDown();
    while (bg::ValueHolder::PopGraphFrame() != nullptr) {
    };
  }
};

const std::string output_file = "lstmp_graph";

std::map<std::string, size_t> GetNodeTypesToNum(const ge::ExecuteGraph *const graph) {
  std::map<std::string, size_t> node_types_to_num;
  for (const auto &node : graph->GetAllNodes()) {
    node_types_to_num[node->GetType()]++;
  }
  return node_types_to_num;
}

std::string PrintNodesDetail(const std::string &reason, const std::map<std::string, size_t> &actual_types_to_num,
                             const std::map<std::string, size_t> &expect_types_to_num) {
  PrettyTable pt;
  pt.SetHeader({"Actual Type", "Actual Num", "Expect Type", "Expect Num"});
  auto actual_iter = actual_types_to_num.begin();
  auto expect_iter = expect_types_to_num.begin();
  while (actual_iter != actual_types_to_num.end() || expect_iter != expect_types_to_num.end()) {
    std::string actual_type = "-";
    std::string actual_num = "-";
    std::string expect_type = "-";
    std::string expect_num = "-";
    bool same_row = false;
    if (actual_iter != actual_types_to_num.end() && expect_iter != expect_types_to_num.end()) {
      if (actual_iter->first == expect_iter->first) {
        actual_type = actual_iter->first;
        actual_num = std::to_string(actual_iter->second);
        expect_type = expect_iter->first;
        expect_num = std::to_string(expect_iter->second);
        same_row = (*actual_iter == *expect_iter);
        ++actual_iter, ++expect_iter;
      } else {
        if (actual_iter->first < expect_iter->first) {
          actual_type = actual_iter->first;
          actual_num = std::to_string(actual_iter->second);
          ++actual_iter;
        } else {
          expect_type = expect_iter->first;
          expect_num = std::to_string(expect_iter->second);
          ++expect_iter;
        }
      }
    } else if (actual_iter == actual_types_to_num.end()) {
      expect_type = expect_iter->first;
      expect_num = std::to_string(expect_iter->second);
      ++expect_iter;
    } else if (expect_iter == expect_types_to_num.end()) {
      actual_type = actual_iter->first;
      actual_num = std::to_string(actual_iter->second);
      ++actual_iter;
    } else {
      // 两个都到end了
      throw std::exception();
    }
    if (same_row) {
      pt.AddRow({actual_type, actual_num, expect_type, expect_num});
    } else {
      pt.AddColorRow({actual_type, actual_num, expect_type, expect_num});
    }
  }

  std::stringstream ss;
  if (!reason.empty()) {
    ss << reason << std::endl;
  }
  pt.Print(ss);
  return ss.str();
}

TEST_F(LowerExeGraphFromModelUT, LowerFromGeModel) {
  auto graph = ShareGraph::LstmpGraph();
  graph->TopologicalSorting();
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  GeModelBuilder builder(graph);
  SetGraphOutShapeRange(graph);
  auto ge_root_model = builder.AddTaskDefForAll(AiCoreTaskDefFaker("stub_func")).BuildGeRootModel();

  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "LstmpLoweringExeGraph");

  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(exe_graph->GetDirectNodesSize(), 3);

  auto init_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Init"), 0);
  auto main_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main"), 0);
  auto de_init_graph = ge::FastNodeUtils::GetSubgraphFromNode(
      ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "DeInit"), 0);

  std::map<std::string, size_t> expect_init_graph_nodes{{"AllocLaunchArg", 7},
                                                        {"AssignWeightMemory", 3},
                                                        {"BuildTensor", 2},
                                                        {"Const", 227},
                                                        {"ConstData", 3},
                                                        {"Data", 4},
                                                        {"CreateL1Allocator", 1},
                                                        {"CreateMemAssigner", 1},
                                                        {"GetOrCreateWeightMem", 1},
                                                        {"SinkWeightData", 3},
                                                        {"FindInferShapeFunc", 3},
                                                        {"FindTilingFunc", 2},
                                                        {"GetSpaceRegistry", 5},
                                                        {"InnerNetOutput", 1},
                                                        {"SelectL1Allocator", 1},
                                                        {"SinkNodeBinWithoutHandle", 7},
                                                        {"SplitRtStreams", 1},
                                                        {"SplitConstTensor", 5},
                                                        {"AppendCoreTypeToPlatform", 1},
                                                        {"GetPlatformInfo", 1},
                                                        {"TilingParse", 7},
                                                        {"NoOp", 1},
                                                        {"CreateInitL2Allocator", 1},
                                                        {"CreateL2Allocators", 1},
                                                        {"PrepareCacheableTilingFwkData", 7}};
  EXPECT_EQ(GetNodeTypesToNum(init_graph), expect_init_graph_nodes)
            << PrintNodesDetail("Check init graph failed: ", GetNodeTypesToNum(init_graph), expect_init_graph_nodes);

  std::map<std::string, size_t> expect_main_graph_nodes{{"AllocBatchHbm", 7},
                                                        {"AllocMemHbm", 11},
                                                        {"AllocModelOutTensor", 3},
                                                        {"EnsureTensorAtOutMemory", 3},
                                                        {"CalcTensorSizeFromStorage", 31},
                                                        {"Data", 7},
                                                        {"FreeBatchHbm", 7},
                                                        {"FreeMemory", 17},
                                                        {"InferShape", 9},
                                                        {"InnerData", 151},
                                                        {"LaunchKernelWithFlag", 7},
                                                        {"CopyFlowLaunch", 3},
                                                        {"OutputData", 1},
                                                        {"SplitRtStreams", 1},
                                                        {"NetOutput", 1},
                                                        {"SelectL1Allocator", 1},
                                                        {"SelectL2Allocator", 1},
                                                        {"SplitDataTensor", 3},
                                                        {"CacheableTiling", 7},
                                                        {"TilingAppendWorkspace", 7},
                                                        {"TilingAppendDfxInfo", 7}};
  EXPECT_EQ(GetNodeTypesToNum(main_graph), expect_main_graph_nodes)
            << PrintNodesDetail("Check main graph failed: ", GetNodeTypesToNum(main_graph), expect_main_graph_nodes);

  std::map<std::string, size_t> expect_de_init_graph_nodes{};
  EXPECT_EQ(GetNodeTypesToNum(de_init_graph), expect_de_init_graph_nodes)
            << PrintNodesDetail("Check de-init graph failed: ", GetNodeTypesToNum(de_init_graph), expect_de_init_graph_nodes);

  memory::CachingMemAllocator::GetAllocator()->Finalize();
}

TEST_F(LowerExeGraphFromModelUT, LowerFromGeModelTwice) {
  auto graph = ShareGraph::LstmpGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDefForAll(AiCoreTaskDefFaker("stub_func")).BuildGeRootModel();

  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "LstmpLoweringExeGraph");

  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(exe_graph->GetDirectNodesSize(), 3);

  // lower same model twice
  auto exe_graph2 = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph2, nullptr);
  memory::CachingMemAllocator::GetAllocator()->Finalize();
}

TEST_F(LowerExeGraphFromModelUT, LowerStaticGeModel) {
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto graph = ShareGraph::AicoreStaticGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDefForAll(AiCoreTaskDefFaker("stub_func")).BuildGeRootModel();
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph.get(), "StaticAiCoreExe");

  ASSERT_EQ(exe_graph->GetDirectNodesSize(), 3);

  auto init_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Init"), 0);
  auto main_graph =
      ge::FastNodeUtils::GetSubgraphFromNode(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main"), 0);
  auto de_init_graph = ge::FastNodeUtils::GetSubgraphFromNode(
      ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "DeInit"), 0);

  std::map<std::string, size_t> expect_init_graph_nodes{{"AllocLaunchArg", 1},
                                                        {"CalcTensorSizeFromStorage", 2},
                                                        {"Const", 32},
                                                        {"ConstData", 3},
                                                        {"Data", 4},
                                                        {"FindInferShapeFunc", 1},
                                                        {"GetSpaceRegistry", 1},
                                                        {"CreateInitL2Allocator", 1},
                                                        {"CreateL1Allocator", 2},
                                                        {"CreateL2Allocators", 1},
                                                        {"InnerNetOutput", 1},
                                                        {"SelectL1Allocator", 2},
                                                        {"SinkNodeBinWithoutHandle", 1},
                                                        {"SplitRtStreams", 1},
                                                        {"NoOp", 1},
                                                        {"CreateHostL2Allocator", 1}};

  EXPECT_EQ(GetNodeTypesToNum(init_graph), expect_init_graph_nodes)
            << PrintNodesDetail("Check init graph failed: ", GetNodeTypesToNum(init_graph), expect_init_graph_nodes);

  std::map<std::string, size_t> expect_main_graph_nodes{{"AllocMemHbm", 1},
                                                        {"AllocBatchHbm", 1},
                                                        {"BuildTensor", 1},
                                                        {"InferShape", 1},
                                                        {"CalcTensorSizeFromStorage", 2},
                                                        {"CopyD2H", 1},
                                                        {"Data", 6},
                                                        {"EnsureTensorAtOutMemory", 1},
                                                        {"FreeBatchHbm", 1},
                                                        {"FreeMemory", 4},
                                                        {"FreeTensorMemory", 1},
                                                        {"InnerData", 28}, // LoweringStaticAicoreNode, args直连增加了InnerData
                                                        {"SplitRtStreams", 1},
                                                        {"LaunchKernelWithFlag", 1},
                                                        {"CopyFlowLaunch", 1},
                                                        {"SplitDataTensor", 2},
                                                        {"SelectL1Allocator", 2},
                                                        {"SelectL2Allocator", 1},
                                                        {"OutputData", 1},
                                                        {"NetOutput", 1},
                                                        {"SyncStream", 1},
                                                        {"CreateHostL2Allocator", 1}};

  EXPECT_EQ(GetNodeTypesToNum(main_graph), expect_main_graph_nodes)
            << PrintNodesDetail("Check main graph failed: ", GetNodeTypesToNum(main_graph), expect_main_graph_nodes);

  std::map<std::string, size_t> expect_de_init_graph_nodes{};
  EXPECT_EQ(GetNodeTypesToNum(de_init_graph), expect_de_init_graph_nodes)
            << PrintNodesDetail("Check de-init graph failed: ", GetNodeTypesToNum(de_init_graph), expect_de_init_graph_nodes);

  memory::CachingMemAllocator::GetAllocator()->Finalize();
}

const std::string save_om_file_name = "file_name_prefix.om";
TEST_F(LowerExeGraphFromModelUT, lower_from_static_model_file) {
  GeModelBuilder builder(ShareGraph::LstmpGraph());
  auto ge_root_model = builder.AddTaskDefForAll(AiCoreTaskDefFaker("stub_func")).AddWeight().BuildGeRootModel();
  ge::ModelBufferData model_buffer;
  ge::ModelHelper model_helper;
  model_helper.SetSaveMode(true);  // Save to buffer.
  ASSERT_EQ(model_helper.SaveToOmRootModel(ge_root_model, save_om_file_name, model_buffer, true),
            ge::SUCCESS);

  ge::graphStatus error_code;
  auto executor_model = gert::LoadExecutorFromFile(save_om_file_name.c_str(), error_code);
  ASSERT_NE(error_code, ge::GRAPH_SUCCESS);
  ASSERT_EQ(executor_model, nullptr);
  memory::CachingMemAllocator::GetAllocator()->Finalize();
}

TEST_F(LowerExeGraphFromModelUT, check_is_dynamic_model) {
  GeModelBuilder builder(ShareGraph::LstmpGraph());
  auto ge_root_model = builder.AddTaskDefForAll(AiCoreTaskDefFaker("stub_func")).AddWeight().BuildGeRootModel();
  ge::ModelBufferData model_buffer;
  ge::ModelHelper model_helper;
  model_helper.SetSaveMode(true);  // Save to buffer.
  ASSERT_EQ(model_helper.SaveToOmRootModel(ge_root_model, save_om_file_name, model_buffer, true),
            ge::SUCCESS);
  bool is_dynamic_model = true;
  ASSERT_EQ(IsDynamicModel(save_om_file_name.c_str(), is_dynamic_model), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(is_dynamic_model == true);
}

TEST_F(LowerExeGraphFromModelUT, load_data_from_file_fail) {
  std::string invalid_model = "temp.om";
  ge::ModelData model_data;
  auto error_code = LoadDataFromFile(invalid_model.c_str(), model_data);
  ASSERT_NE(error_code, ge::GRAPH_SUCCESS);
}

/*
 * ConvertGeModelToExecuteGraph老接口UT适配，后续需要配套代码删除
 * */
TEST_F(LowerExeGraphFromModelUT, LowerExeGraphSucc) {
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto graph = ShareGraph::LstmpGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDefForAll(AiCoreTaskDefFaker("stub_func")).BuildGeRootModel();
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
}

TEST_F(LowerExeGraphFromModelUT, check_is_dynamic_model_with_model)
{
  GeModelBuilder builder(ShareGraph::LstmpGraph());
  auto ge_root_model = builder.AddTaskDefForAll(AiCoreTaskDefFaker("stub_func")).AddWeight().BuildGeRootModel();
  ge::ModelBufferData model_buffer;
  ge::ModelHelper model_helper;
  model_helper.SetSaveMode(true);  // Save to buffer.
  ASSERT_EQ(model_helper.SaveToOmRootModel(ge_root_model, save_om_file_name, model_buffer, true),
            ge::SUCCESS);
  ge::ModelData model_data;
  ge::graphStatus error_code = ge::ModelParserBase::LoadFromFile(save_om_file_name.c_str(), -1, model_data);
  ASSERT_EQ(error_code, ge::SUCCESS);
  GE_MAKE_GUARD(release_model_data, [&model_data] {
    if (model_data.model_data != nullptr) {
      delete[] static_cast<char *>(model_data.model_data);
      model_data.model_data = nullptr;
    }
  });
  bool is_dynamic_model = false;
  ASSERT_EQ(IsDynamicModel(model_data.model_data, model_data.model_len, is_dynamic_model), ge::GRAPH_SUCCESS);
  EXPECT_EQ(is_dynamic_model, true);
  ASSERT_EQ(IsDynamicModel(model_data.model_data, 0UL, is_dynamic_model), ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);
}

TEST_F(LowerExeGraphFromModelUT, check_cust_aicpu_kernel_bin_ok)
{
  auto graph = ShareGraph::ThirdAicpuOpGraph();
  auto add_op_desc = graph->FindNode("add1")->GetOpDesc();
  graph->TopologicalSorting();

  GeModelBuilder builder(graph);
  AiCpuTfTaskDefFaker aicpu_task_def_faker;
  auto ge_root_model = builder.AddTaskDef("Add", aicpu_task_def_faker.SetNeedMemcpy(true))
      .AddTaskDef("NonZero", aicpu_task_def_faker.SetNeedMemcpy(true))
      .BuildGeRootModel();

  builder.AddCustAicpuKernelStore("add1");
  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);

  const ge::OpKernelBinPtr bin_ptr = add_op_desc->TryGetExtAttr<ge::OpKernelBinPtr>(ge::OP_EXTATTR_CUSTAICPU_KERNEL, nullptr);
  ASSERT_NE(bin_ptr, nullptr);
}

TEST_F(LowerExeGraphFromModelUT, LowerFromGeModelWithNullSpaceRegistry) {
  auto graph = ShareGraph::LstmpGraph();
  graph->TopologicalSorting();
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDefForAll(AiCoreTaskDefFaker("stub_func")).BuildGeRootModel();
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
}

TEST_F(LowerExeGraphFromModelUT, LowerFromGeModelWithNullTaskDef) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();
  ge_root_model->SetFlattenGraph(nullptr);
  auto name_to_model = ge_root_model->GetSubgraphInstanceNameToModel();
  name_to_model["test"]->SetModelTaskDef(nullptr);
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_EQ(exe_graph, nullptr);
}
}  // namespace gert
