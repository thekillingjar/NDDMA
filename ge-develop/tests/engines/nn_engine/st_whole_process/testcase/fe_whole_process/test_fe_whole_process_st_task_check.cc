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
#include <nlohmann/json.hpp>
#define protected public
#define private public
#include <graph/tensor.h>
#include "framework/fe_running_env/include/fe_running_env.h"
#include "graph_optimizer/fe_graph_optimizer.h"

#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "graph_constructor/graph_constructor.h"
#include "register/ops_kernel_builder_registry.h"
#include "st_whole_process/stub/aicore_ops_kernel_builder_stub.h"
#include "google/protobuf/text_format.h"
#undef protected
#undef private
#include "stub/te_fusion_api.h"
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;
using CreateFn = ge::OpsKernelBuilder *(*)();
using OpsKernelBuilderPtr = std::shared_ptr<ge::OpsKernelBuilder>;
namespace fe {
class STestFeOmConsistencyCheck: public testing::Test
{
 protected:
  static void SetUpTestCase() {
    const char *soc_version = "Ascend910A";
    rtSetSocVersion(soc_version);
    auto &fe_env = fe_env::FeRunningEnv::Instance();
    std::map<string, string> new_options;
    new_options.emplace(ge::SOC_VERSION, soc_version);
    fe_env.InitRuningEnv(new_options);
  }

  static void TearDownTestCase() {
    te::TeStub::GetInstance().SetImpl(std::make_shared<te::TeStubApi>());
    auto &fe_env = fe_env::FeRunningEnv::Instance();
    fe_env.FinalizeEnv();
  }
};

// TEST_F(STestFeOmConsistencyCheck, test_tasks_check2) {
//   /* This case uses real compilation environment. */
//   te::TeStub::GetInstance().SetImpl(std::make_shared<te::RealTimeCompilation>());
//   auto &fe_env = fe_env::FeRunningEnv::Instance();
//   ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("fe_st");
//   std::map<string, string> session_options;
//   session_options[ge::PRECISION_MODE] = fe::FORCE_FP16;

//   string network_path = fe_env::FeRunningEnv::GetNetworkPath("inceptionv3_aipp_int8_16batch.txt");
//   string task_context_path = fe_env::FeRunningEnv::GetNetworkPath("task_context.txt");
//   bool state = ge::GraphUtils::LoadGEGraph(network_path.c_str(), graph);
//   ASSERT_EQ(state, true);
//   ASSERT_EQ(graph->GetName(), "inceptionv3_framework_caffe_aipp_1_batch_16_input_int8_output_FP32");
//   EXPECT_EQ(graph->GetDirectNode().size(), 661);

//   const auto &ops_kernel_builders_const = ge::OpsKernelBuilderRegistry::GetInstance().GetAll();
//   std::map<std::string, OpsKernelBuilderPtr> &ops_kernel_builders = const_cast<std::map<std::string, OpsKernelBuilderPtr>&>(ops_kernel_builders_const);

//   for (auto &ops_kernel_builder: ops_kernel_builders) {
//     if (ops_kernel_builder.first == AI_CORE_NAME) {
//       AICoreOpsKernelSubBuilder test;
//       CreateFn fn = []()->::ge::OpsKernelBuilder* {return new (std::nothrow) (AICoreOpsKernelSubBuilder)();};
//       ops_kernel_builder.second.reset(fn());
//       std::cout << "inject AICoreOpsKernelSubBuilder success!" << std::endl;
//     }
//   }

//   auto ret = fe_env.Run(graph, session_options);
//   EXPECT_EQ(ret, fe::SUCCESS);
//   std::string task_def_str;
//   for (auto task_iter : fe_env::FeRunningEnv::tasks_map) {
//     for (auto task : task_iter.second) {
//       std::string task_def_str_tmp;
//       google::protobuf::TextFormat::PrintToString(task, &task_def_str_tmp);
//       //std::cout << task_def_str_tmp << std::endl;
//       task_def_str.append(task_def_str_tmp);
//     }
//   }

//   /*校验om一致性*/
//   //std::cout << "=========Stat to check task context=======" << std::endl;
//   //std::cout << task_def_str << std::endl;
//   //std::cout << "=========End to check task context=======" << std::endl;
//   ifstream ifs;
//   ifs.open("./task_context.txt", ios::in);
//   if (!ifs.is_open()) {
//     std::cout << "open task_context filed!" << std::endl;
//   }
//   std::string task_context;
//   std::ostringstream oss;
//   oss << ifs.rdbuf();
//   task_context = oss.str();
//   ifs.close();
//   bool same_size = task_def_str.size() == task_context.size();
//   //EXPECT_EQ(same_size, true);
//   //EXPECT_EQ(task_context.compare(task_def_str), 0);

//   /* 在执行后可以对特定阶段的图进行校验 */
//   CHECK_GRAPH(PreRunAfterBuild) {
//     std::cout << "PreRunAfterBuild GetDirectNodesSize = " <<  graph->GetDirectNodesSize() << std::endl;
//     size_t cp_info_size = 0;
//     size_t kernel_buffer_size = 0;

//     for (auto &node : graph->GetDirectNode()) {
//       if (ge::AttrUtils::HasAttr(node->GetOpDesc(), COMPILE_INFO_JSON)) {
//         cp_info_size++;
//       }
//       if (ge::AttrUtils::HasAttr(node->GetOpDesc(), ge::ATTR_NAME_TBE_KERNEL_BUFFER)) {
//         kernel_buffer_size++;
//       }
//     }
//     EXPECT_EQ(cp_info_size, 0);
//     EXPECT_EQ(kernel_buffer_size, 0);
//   };
// }
}
