/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "jit_execution/user_graph_ctrl.h"
#include "hybrid/model/hybrid_model_builder.h"
#include <stdio.h>
#include <gtest/gtest.h>
#include "mmpa/mmpa_api.h"
#include "macro_utils/dt_public_scope.h"
#include "framework/common/helper/model_helper.h"
#include "hybrid/node_executor/aicore/aicore_op_task.h"
#include "graph/utils/graph_utils_ex.h"
#include "stub/gert_runtime_stub.h"
#include "common/share_graph.h"
#include "proto/task.pb.h"
#include "faker/ge_model_builder.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "common/env_path.h"

namespace ge {
const char *const kEnvName = "ASCEND_OPP_PATH";
const char *const kEnvNameCustom = "ASCEND_CUSTOM_OPP_PATH";
const string kOpsProto = "libopsproto_rt2.0.so";
const string kOpMaster = "libopmaster_rt2.0.so";
const string kInner = "built-in";
const string kOpsProtoPath = "/op_proto/lib/linux/x86_64/";
const string kOpMasterPath = "/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/";
#if 0
ge-dev失败，ge仓正常，两个仓合并时，再打开
static GeRootModelPtr ConstructGeRootModel(bool is_dynamic_shape = true,
                                           ModelTaskType task_type = ModelTaskType::MODEL_TASK_KERNEL,
                                           ccKernelType kernel_type = ccKernelType::TE,
                                           std::string so_name = "") {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  AttrUtils::SetBool(graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic_shape);
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_root_model->subgraph_instance_name_to_model_["graph"] = ge_model;
  ge_model->SetGraph(graph);
  (void)ge::AttrUtils::SetStr(ge_model, ATTR_MODEL_HOST_ENV_OS, "linux");
  (void)ge::AttrUtils::SetStr(ge_model, ATTR_MODEL_HOST_ENV_CPU, "x86_64");

  GeModelPtr ge_model1 = std::make_shared<GeModel>();
  ge::ComputeGraphPtr graph1 = std::make_shared<ge::ComputeGraph>("graph1");
  ge_model1->SetGraph(graph1);
  ge_root_model->subgraph_instance_name_to_model_["graph1"] = ge_model1;

  {
    domi::ModelTaskDef model_task_def;
    std::shared_ptr<domi::ModelTaskDef> model_task_def_ptr = make_shared<domi::ModelTaskDef>(model_task_def);
    domi::TaskDef *task_def = model_task_def_ptr->add_task();
    ge_model->SetModelTaskDef(model_task_def_ptr);
    ge_model1->SetModelTaskDef(model_task_def_ptr);

    auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());
    task_def->set_type(static_cast<uint32_t>(task_type));

    domi::KernelDef *kernel_def = task_def->mutable_kernel();
    kernel_def->set_block_dim(32);
    kernel_def->set_args_size(64);
    kernel_def->set_so_name(so_name);
    string args(64, '1');
    uint16_t args_offset[9] = {0};
    kernel_def->set_args(args.data(), 64);
    domi::KernelContext *context = kernel_def->mutable_context();
    context->set_kernel_type(static_cast<uint32_t>(kernel_type));
    context->set_op_index(1);
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }
  return ge_root_model;
}
#endif 
std::string GetAutofuseSoPath() {
  std::string cmake_binary_path = CMAKE_BINARY_DIR;
  return cmake_binary_path + "/tests/depends/op_stub/libautofuse_stub.so";
}

ge::ComputeGraphPtr BuildAutofuseGraph() {
    auto graph = gert::ShareGraph::AutoFuseNodeGraph();
    (void)ge::AttrUtils::SetInt(graph, "_all_symbol_num", 8);
    auto fused_graph_node = graph->FindNode("fused_graph");
    auto fused_graph_node1 = graph->FindNode("fused_graph1");

    auto autofuse_stub_so = GetAutofuseSoPath();
    std::cout << "bin path: " << autofuse_stub_so << std::endl;
    (void)ge::AttrUtils::SetStr(fused_graph_node->GetOpDesc(), "bin_file_path", autofuse_stub_so);
    (void)ge::AttrUtils::SetStr(fused_graph_node1->GetOpDesc(), "bin_file_path", autofuse_stub_so);

    return graph;
}

class AutofuseOfflineSt : public testing::Test {};
TEST_F(AutofuseOfflineSt, CheckSaveAutofuseSo) {
  std::string output_file{"split_opp.om"};
  auto graph = BuildAutofuseGraph();
  auto ge_root_model = gert::GeModelBuilder(graph).BuildGeRootModel();

  ModelHelper model_helper;
  ModelBufferData model_buff;
  Status ret = model_helper.SaveToOmRootModel(ge_root_model, output_file, model_buff, true);
  EXPECT_EQ(ret, SUCCESS);

  OmFileLoadHelper load_helper;
  load_helper.is_inited_ = true;
  OmFileContext cur_ctx;
  ModelPartition so_patition;
  so_patition.type = ModelPartitionType::SO_BINS;
  auto so_data = std::unique_ptr<char[]>(new(std::nothrow) char[20]);
  so_patition.data = reinterpret_cast<uint8_t*>(so_data.get());
  so_patition.size = 20;
  cur_ctx.partition_datas_.push_back(so_patition);
  load_helper.model_contexts_.push_back(cur_ctx);
  EXPECT_EQ(model_helper.LoadOpSoBin(load_helper, ge_root_model), SUCCESS);
}

// ge-dev失败，ge仓正常，两个仓合并时，再打开。
// TEST_F(AutofuseOfflineSt, CheckLoadAutofuseSo) {
//   std::string ut_run_base = EnvPath().GetOrCreateCaseTmpPath("autofuse");
//   auto opp_path = ut_run_base + "/opp/";
//   mmSetEnv(kEnvName, opp_path.c_str(), 1);
//
//   std::string inner_op_master = opp_path + "built_in/op_master_device/lib/";
//   system(("mkdir -p " + inner_op_master).c_str());
//   inner_op_master += "Ascend-V7.6-libopmaster.so";
//   system(("touch " + inner_op_master).c_str());
//   system(("echo 'Ascend-V7.6-libopmaster' > " + inner_op_master).c_str());
//
//   std::string inner_proto_path = opp_path + kInner + kOpsProtoPath;
//   system(("mkdir -p " + inner_proto_path).c_str());
//   inner_proto_path += kOpsProto;
//   system(("touch " + inner_proto_path).c_str());
//   system(("echo 'ops proto' > " + inner_proto_path).c_str());
//
//   std::string inner_tiling_path = opp_path + kInner + kOpMasterPath;
//   system(("mkdir -p " + inner_tiling_path).c_str());
//   inner_tiling_path += kOpMaster;
//   system(("touch " + inner_tiling_path).c_str());
//   system(("echo 'op tiling ' > " + inner_tiling_path).c_str());
//
//   ModelBufferData model;
//   ModelHelper model_helper;
//   model_helper.SetSaveMode(true);
//   GeRootModelPtr ge_root_model =
//       ConstructGeRootModel(true, ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL, ccKernelType::AI_CPU, inner_op_master);
//   EXPECT_EQ(ge_root_model->CheckAndSetNeedSoInOM(), SUCCESS);
//   EXPECT_EQ(ge_root_model->GetSoInOmFlag(), 0xc000);
//
//   auto ge_root_graph = ge_root_model->GetRootGraph();
//
//   OpDescBuilder op_desc_builder("test", "AscBackend");
//   const auto &op_desc = op_desc_builder.Build();
//   auto node = ge_root_graph->AddNode(op_desc);
//   node->SetOwnerComputeGraph(ge_root_graph);
//   auto autofuse_stub_so = inner_op_master;
//   std::cout << "bin path: " << autofuse_stub_so << std::endl;
//   auto nodes = ge_root_graph->GetAllNodesPtr();
//   for (auto n : nodes) {
//     (void)ge::AttrUtils::SetStr(n->GetOpDesc(), "bin_file_path", autofuse_stub_so);
//   }
//
//   EXPECT_EQ(ge_root_model->CheckAndSetNeedSoInOM(), SUCCESS);
//
//   std::map<std::string, std::string> options_map;
//   options_map["ge.host_env_os"] = "linux";
//   options_map["ge.host_env_cpu"] = "x86_64";
//   GetThreadLocalContext().SetGlobalOption(options_map);
//   std::string output_file = opp_path + "/output.om";
//   EXPECT_EQ(model_helper.SaveToOmRootModel(ge_root_model, output_file, model, false), SUCCESS);
//
//   ge::ModelParserBase base;
//   ge::ModelData model_data;
//   EXPECT_EQ(base.LoadFromFile((opp_path + "output_linux_x86_64.om").c_str(), -1, model_data), SUCCESS);
//   EXPECT_EQ(model_helper.LoadRootModel(model_data), SUCCESS);
//   if (model_data.model_data != nullptr) {
//     delete[] reinterpret_cast<char_t *>(model_data.model_data);
//   }
//
//   const auto &root_model = model_helper.GetGeRootModel();
//   const auto &so_list = root_model->GetAllSoBin();
//   EXPECT_EQ(so_list.size(), 4UL);
//   unordered_map<SoBinType, uint32_t> res;
//   res.emplace(SoBinType::kSpaceRegistry, 0U);
//   res.emplace(SoBinType::kOpMasterDevice, 0U);
//   std::for_each(so_list.begin(), so_list.end(), [&res](const OpSoBinPtr &so_bin) { res[so_bin->GetSoBinType()]++; });
//   EXPECT_EQ(res[SoBinType::kSpaceRegistry], 2U);
//   EXPECT_EQ(res[SoBinType::kOpMasterDevice], 1U);
//   system(("rm -rf " + opp_path).c_str());
// }
}
