/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "autofuse_config/auto_fuse_config.h"
#define private public
#include "codegen_tiling.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "schedule_result.h"
#include "runtime_stub.h"
#include "platform_context.h"
#include "ascgraph_info_complete.h"
#include "optimize/optimize.h"
#include <fstream>
#include <filesystem>

namespace {
std::pair<int, std::string> execute_command(const std::string& command) {
  std::array<char, 128> buffer;
  std::string output;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);

  if (!pipe) {
    throw std::runtime_error("Failed to open pipe");
  }

  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    output += buffer.data();
  }

  return {WEXITSTATUS(pclose(pipe.release())), output};
}

bool CompileCode(const std::string &code){
  std::string cmake_dir = CMAKE_BINARY_DIR;
  // 临时目录
  std::string temp_dir = cmake_dir + "/tests/autofuse/ut/temp_compile_codegen_tiling";
  std::filesystem::remove_all(temp_dir);
  std::filesystem::create_directories(temp_dir);
  // 生成的 C++ 文件路径
  std::string source_file = temp_dir + "/temp_codegen_infershape.cpp";
  // 生成 C++ 代码
  std::ofstream source_stream(source_file);
  source_stream << code << R"(
        int main() {
            return 0;
        }
    )";
  source_stream.close();
  // 头文件路径
  std::string ascend_install_path = ASCEND_INSTALL_PATH;
  std::string include_path = "-I" + ascend_install_path + "/include/ ";
  std::string link_path = "-L" + ascend_install_path + "/lib64";
  // 编译代码
  std::string compile_command = "g++ -std=c++17 " + include_path + " " + link_path + " " + source_file + " -lc_sec";
  auto [compile_exit_code, compile_output] = execute_command(compile_command);
  // 清理临时目录
  std::filesystem::remove_all(temp_dir);
  return compile_exit_code == 0;
}
}

class TestCodegenTiling : public testing::Test, public codegen::TilingLib {
  public:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_DEBUG, 0);
  }
  void TearDown() override {
  }
  std::map<std::string, std::string> GenTilingCode(const std::vector<ge::Expression> &origin_vars = {},
                                                   const std::map<std::string, std::string> &shape_info = {}) {
    ge::AscGraph graph("test_graph");
    auto s0 = graph.CreateSizeVar("s0");
    auto z0 = graph.CreateAxis("z0", ge::ops::Zero);

    ge::ascir_op::Data x_op("x", graph);
    x_op.ir_attr.SetIndex(0);
    ge::ascir_op::Load load_op("load");
    ge::ascir_op::Store store_op("store");
    ge::ascir_op::Output y_op("y");
    y_op.ir_attr.SetIndex(0);
    graph.AddNode(load_op);
    graph.AddNode(store_op);
    graph.AddNode(y_op);

    load_op.x = x_op.y;
    load_op.y.dtype = ge::DT_FLOAT16;
    store_op.x = load_op.y;
    y_op.x = store_op.y;

    auto x = graph.FindNode("x");
    auto load = graph.FindNode("load");
    auto store = graph.FindNode("store");
    auto y = graph.FindNode("y");

    x->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeGlobal;
    x->outputs[0].attr.mem.tensor_id = 0;
    x->attr.api.unit = ge::ComputeUnit::kUnitNone;
    y->attr.api.unit = ge::ComputeUnit::kUnitNone;

    load->outputs[0].attr.axis = {z0.id};
    load->outputs[0].attr.vectorized_axis = {z0.id};
    load->outputs[0].attr.vectorized_strides = {ge::ops::One};
    load->outputs[0].attr.repeats = {z0.size};
    load->outputs[0].attr.strides = {ge::ops::One};
    load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
    load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
    load->outputs[0].attr.mem.tensor_id = 1;
    load->outputs[0].attr.que.id = 0;
    load->outputs[0].attr.mem.reuse_id = 0;
    load->outputs[0].attr.que.depth = 2;
    load->outputs[0].attr.que.buf_num = 2;
    load->outputs[0].attr.opt.merge_scope = ge::kIdNone;

    store->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeGlobal;
    store->outputs[0].attr.mem.tensor_id = 2;
    store->outputs[0].attr.axis = {z0.id};
    store->outputs[0].attr.vectorized_axis = {z0.id};
    store->outputs[0].attr.vectorized_strides = {ge::ops::One};
    store->outputs[0].attr.repeats = {z0.size};
    store->outputs[0].attr.strides = {ge::ops::One};

    ::ascir::ScheduledResult schedule_result;
    schedule_result.schedule_groups.resize(1);
    for (auto &schedule_group : schedule_result.schedule_groups) {
      schedule_group.impl_graphs.emplace_back(graph);
    }
    std::vector<ascir::ScheduledResult> schedule_results;
    schedule_results.push_back(schedule_result);
    schedule_results.push_back(schedule_result);

    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.fused_graph_name = ge::AscendString(graph.GetName().c_str());
    fused_schedule_result.input_nodes.push_back(x);
    fused_schedule_result.output_nodes.push_back(y);
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    fused_schedule_result.origin_vars = origin_vars;
    return this->Generate(fused_schedule_result, shape_info, "", "0");
  }

  protected:
  TestCodegenTiling() : codegen::TilingLib("test", "test") {}
};

TEST_F(TestCodegenTiling, NoWorkspaceTest) {
  ascir::ImplGraph graph0("test_graph0");
  graph0.CreateSizeVar("s0");
  graph0.CreateSizeVar("s1");

  std::vector<ascir::ImplGraph> impl_graphs;
  impl_graphs.push_back(graph0);

  std::vector<ascir::ScheduledResult> schedule_results;
  ascir::ScheduledResult schedule_result;
  ascir::ScheduleGroup schedule_group;
  schedule_group.impl_graphs = impl_graphs;
  schedule_result.schedule_groups.push_back(schedule_group);
  schedule_results.push_back(schedule_result);
  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);

  EXPECT_EQ(this->GenGetWorkspaceSizeFunc("AutofuseTilingData", fused_schedule_result), std::string{
    "uint32_t GetWorkspaceSize(const AutofuseTilingData &t) {\n"
    "  using namespace optiling;\n"
    "  uint32_t ws_size = 0;\n"
    "    if (t.tiling_key == 0) {\n"
    "      ws_size += 0;\n"
    "    }\n"
    "\n"
    "  ws_size = (ws_size + 512 - 1) / 512 * 512;\n"
    "  return ws_size;\n"
    "}\n"});
}

TEST_F(TestCodegenTiling, SingleGroupWorkspaceSymbolTest) {
  ascir::ImplGraph graph0("test_graph0");
  auto s0 = graph0.CreateSizeVar("s0");
  auto s1 = graph0.CreateSizeVar("s1");

  auto z0 = graph0.CreateAxis("z0", s0);
  auto z1 = graph0.CreateAxis("z1", s1);

  ge::ascir_op::Workspace workspace("workspace");
  graph0.AddNode(workspace);
  workspace.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Load load("load");
  graph0.AddNode(load);
  load.x = workspace.y;
  load.attr.sched.axis = {z0.id, z1.id};
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, ge::ops::One};

  auto load_node = graph0.FindNode("load");
  auto workspace_node = graph0.FindNode("workspace");

  workspace_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  workspace_node->outputs[0].attr.mem.tensor_id = 0;

  load_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load_node->outputs[0].attr.mem.tensor_id = 1;

  std::vector<ascir::ImplGraph> impl_graphs;
  impl_graphs.push_back(graph0);

  std::vector<ascir::ScheduledResult> schedule_results;
  ascir::ScheduledResult schedule_result;
  ascir::ScheduleGroup schedule_group;
  schedule_group.impl_graphs = impl_graphs;
  schedule_result.schedule_groups.push_back(schedule_group);
  schedule_results.push_back(schedule_result);

  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.workspace_nodes.push_back(workspace_node);
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  EXPECT_EQ(this->GenGetWorkspaceSizeFunc("AutofuseTilingData", fused_schedule_result), std::string{
    "uint32_t GetWorkspaceSize(const AutofuseTilingData &t) {\n"
    "  using namespace optiling;\n"
    "  uint32_t ws_size = 0;\n"
    "    if (t.tiling_key == 0) {\n"
    "      ws_size += Max(0, (2 * Max(Max(1, t.s1), (t.s0 * t.s1))));\n"
    "    }\n"
    "\n"
    "  ws_size = (ws_size + 512 - 1) / 512 * 512;\n"
    "  return ws_size;\n"
    "}\n"
  });
}

TEST_F(TestCodegenTiling, SingleGroupWorkspaceValueTest) {
  ascir::ImplGraph graph0("test_graph0");
  auto s0 = graph0.CreateSizeVar(150);
  auto s1 = graph0.CreateSizeVar(2);

  auto z0 = graph0.CreateAxis("z0", s0);
  auto z1 = graph0.CreateAxis("z1", s1);

  ge::ascir_op::Workspace workspace("workspace");
  graph0.AddNode(workspace);
  workspace.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Load load("load");
  graph0.AddNode(load);
  load.x = workspace.y;
  load.attr.sched.axis = {z0.id, z1.id};
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, ge::ops::One};

  auto load_node = graph0.FindNode("load");
  auto workspace_node = graph0.FindNode("workspace");

  workspace_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  workspace_node->outputs[0].attr.mem.tensor_id = 0;

  load_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load_node->outputs[0].attr.mem.tensor_id = 1;

  std::vector<ascir::ImplGraph> impl_graphs;
  impl_graphs.push_back(graph0);

  std::vector<ascir::ScheduledResult> schedule_results;
  ascir::ScheduledResult schedule_result;
  ascir::ScheduleGroup schedule_group;
  schedule_group.impl_graphs = impl_graphs;
  schedule_result.schedule_groups.push_back(schedule_group);
  schedule_results.push_back(schedule_result);

  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.workspace_nodes.push_back(workspace_node);
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  EXPECT_EQ(this->GenGetWorkspaceSizeFunc("AutofuseTilingData", fused_schedule_result), std::string{
    "uint32_t GetWorkspaceSize(const AutofuseTilingData &t) {\n"
    "  using namespace optiling;\n"
    "  uint32_t ws_size = 0;\n"
    "    if (t.tiling_key == 0) {\n"
    "      ws_size += 600;\n"
    "    }\n"
    "\n"
    "  ws_size = (ws_size + 512 - 1) / 512 * 512;\n"
    "  return ws_size;\n"
    "}\n"});
}

TEST_F(TestCodegenTiling, MultiGroupWorkspaceSymbolTest) {
  ascir::ImplGraph graph0("test_graph0");
  auto s0 = graph0.CreateSizeVar("s0");
  auto s1 = graph0.CreateSizeVar("s1");

  auto z0 = graph0.CreateAxis("z0", s0);
  auto z1 = graph0.CreateAxis("z1", s1);

  ge::ascir_op::Workspace workspace("workspace");
  graph0.AddNode(workspace);
  workspace.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Load load("load");
  graph0.AddNode(load);
  load.x = workspace.y;
  load.attr.sched.axis = {z0.id, z1.id};
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, ge::ops::One};

  auto load_node = graph0.FindNode("load");
  auto workspace_node = graph0.FindNode("workspace");

  workspace_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  workspace_node->outputs[0].attr.mem.tensor_id = 0;

  load_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load_node->outputs[0].attr.mem.tensor_id = 2;

  ascir::ImplGraph graph1("test_graph1");
  s0 = graph1.CreateSizeVar("s0");
  s1 = graph1.CreateSizeVar("s1");

  z0 = graph1.CreateAxis("z0", s0);
  z1 = graph1.CreateAxis("z1", s1);

  ge::ascir_op::Workspace workspace1("workspace1");
  graph1.AddNode(workspace1);
  workspace1.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Load load1("load1");
  graph1.AddNode(load1);
  load1.x = workspace1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, ge::ops::One};

  auto load1_node = graph1.FindNode("load1");
  auto workspace1_node = graph1.FindNode("workspace1");

  workspace1_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  workspace1_node->outputs[0].attr.mem.tensor_id = 1;

  load1_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load1_node->outputs[0].attr.mem.tensor_id = 3;

  std::vector<ascir::ScheduledResult> schedule_results;
  ascir::ScheduledResult schedule_result;
  ascir::ScheduleGroup sch_groups0;
  ascir::ScheduleGroup sch_groups1;
  sch_groups0.impl_graphs = {graph0};
  sch_groups1.impl_graphs = {graph1};
  schedule_result.schedule_groups.push_back(sch_groups0);
  schedule_result.schedule_groups.push_back(sch_groups1);
  schedule_results.push_back(schedule_result);

  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.workspace_nodes.push_back(workspace_node);
  fused_schedule_result.workspace_nodes.push_back(workspace1_node);
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  EXPECT_EQ(this->GenGetWorkspaceSizeFunc("AutofuseTilingData", fused_schedule_result), std::string{
    "uint32_t GetWorkspaceSize(const AutofuseTilingData &t) {\n"
    "  using namespace optiling;\n"
    "  uint32_t ws_size = 0;\n"
    "  if (t.graph0_tiling_key == 0) {\n"
    "    if (t.graph0_result0_g0_tiling_data.tiling_key == 0) {\n"
    "      ws_size += Max(0, (2 * Max(Max(1, t.graph0_result0_g0_tiling_data.s1), (t.graph0_result0_g0_tiling_data.s0 * t.graph0_result0_g0_tiling_data.s1))));\n"
    "    }\n"
    "    if (t.graph0_result0_g1_tiling_data.tiling_key == 0) {\n"
    "      ws_size += Max(0, (2 * Max(Max(1, t.graph0_result0_g1_tiling_data.s1), (t.graph0_result0_g1_tiling_data.s0 * t.graph0_result0_g1_tiling_data.s1))));\n"
    "    }\n"
    "  }\n"
    "  ws_size = (ws_size + 512 - 1) / 512 * 512;\n"
    "  return ws_size;\n"
    "}\n"
  });
}

TEST_F(TestCodegenTiling, MultiGroupWorkspaceValueTest) {
  ascir::ImplGraph graph0("test_graph0");
  auto s0 = graph0.CreateSizeVar(16);
  auto s1 = graph0.CreateSizeVar(32);

  auto z0 = graph0.CreateAxis("z0", s0);
  auto z1 = graph0.CreateAxis("z1", s1);

  ge::ascir_op::Workspace workspace("workspace");
  graph0.AddNode(workspace);
  workspace.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Load load("load");
  graph0.AddNode(load);
  load.x = workspace.y;
  load.attr.sched.axis = {z0.id, z1.id};
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, ge::ops::One};

  auto load_node = graph0.FindNode("load");
  auto workspace_node = graph0.FindNode("workspace");

  workspace_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  workspace_node->outputs[0].attr.mem.tensor_id = 0;

  load_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load_node->outputs[0].attr.mem.tensor_id = 2;

  ascir::ImplGraph graph1("test_graph1");
  s0 = graph1.CreateSizeVar(5);
  s1 = graph1.CreateSizeVar(100);

  z0 = graph1.CreateAxis("z0", s0);
  z1 = graph1.CreateAxis("z1", s1);

  ge::ascir_op::Workspace workspace1("workspace1");
  graph1.AddNode(workspace1);
  workspace1.y.dtype = ge::DT_FLOAT16;

  ge::ascir_op::Load load1("load1");
  graph1.AddNode(load1);
  load1.x = workspace1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, ge::ops::One};

  auto load1_node = graph1.FindNode("load1");
  auto workspace1_node = graph1.FindNode("workspace1");

  workspace1_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  workspace1_node->outputs[0].attr.mem.tensor_id = 1;

  load1_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load1_node->outputs[0].attr.mem.tensor_id = 3;

  std::vector<ascir::ScheduledResult> schedule_results;
  ascir::ScheduledResult schedule_result;
  ascir::ScheduleGroup sch_groups0;
  ascir::ScheduleGroup sch_groups1;
  sch_groups0.impl_graphs = {graph0};
  sch_groups1.impl_graphs = {graph1};
  schedule_result.schedule_groups.push_back(sch_groups0);
  schedule_result.schedule_groups.push_back(sch_groups1);
  schedule_results.push_back(schedule_result);

  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.workspace_nodes.push_back(workspace_node);
  fused_schedule_result.workspace_nodes.push_back(workspace1_node);
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  EXPECT_EQ(this->GenGetWorkspaceSizeFunc("AutofuseTilingData", fused_schedule_result), std::string{
    "uint32_t GetWorkspaceSize(const AutofuseTilingData &t) {\n"
    "  using namespace optiling;\n"
    "  uint32_t ws_size = 0;\n"
    "  if (t.graph0_tiling_key == 0) {\n"
    "    if (t.graph0_result0_g0_tiling_data.tiling_key == 0) {\n"
    "      ws_size += 1024;\n"
    "    }\n"
    "    if (t.graph0_result0_g1_tiling_data.tiling_key == 0) {\n"
    "      ws_size += 1000;\n"
    "    }\n"
    "  }\n"
    "  ws_size = (ws_size + 512 - 1) / 512 * 512;\n"
    "  return ws_size;\n"
    "}\n"});
}

TEST_F(TestCodegenTiling, EmptyTensorKernel) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", ge::ops::Zero);

  ge::ascir_op::Data x_op("x", graph);
  x_op.ir_attr.SetIndex(0);
  ge::ascir_op::Load load_op("load");
  ge::ascir_op::Store store_op("store");
  ge::ascir_op::Output y_op("y");
  y_op.ir_attr.SetIndex(0);
  graph.AddNode(load_op);
  graph.AddNode(store_op);
  graph.AddNode(y_op);

  load_op.x = x_op.y;
  load_op.y.dtype = ge::DT_FLOAT16;
  store_op.x = load_op.y;
  y_op.x = store_op.y;

  auto x = graph.FindNode("x");
  auto load = graph.FindNode("load");
  auto store = graph.FindNode("store");
  auto y = graph.FindNode("y");

  x->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->attr.api.unit = ge::ComputeUnit::kUnitNone;
  y->attr.api.unit = ge::ComputeUnit::kUnitNone;

  load->outputs[0].attr.axis = {z0.id};
  load->outputs[0].attr.vectorized_axis = {z0.id};
  load->outputs[0].attr.vectorized_strides = {ge::ops::One};
  load->outputs[0].attr.repeats = {z0.size};
  load->outputs[0].attr.strides = {ge::ops::One};
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.mem.tensor_id = 1;
  load->outputs[0].attr.que.id = 0;
  load->outputs[0].attr.mem.reuse_id = 0;
  load->outputs[0].attr.que.depth = 2;
  load->outputs[0].attr.que.buf_num = 2;
  load->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  store->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.axis = {z0.id};
  store->outputs[0].attr.vectorized_axis = {z0.id};
  store->outputs[0].attr.vectorized_strides = {ge::ops::One};
  store->outputs[0].attr.repeats = {z0.size};
  store->outputs[0].attr.strides = {ge::ops::One};

  ::ascir::ScheduledResult schedule_result;
  schedule_result.schedule_groups.resize(1);
  for (auto &schedule_group : schedule_result.schedule_groups) {
    schedule_group.impl_graphs.emplace_back(graph);
  }
  std::vector<ascir::ScheduledResult> schedule_results;
  schedule_results.push_back(schedule_result);
  schedule_results.push_back(schedule_result);

  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = ge::AscendString(graph.GetName().c_str());
  fused_schedule_result.input_nodes.push_back(x);
  fused_schedule_result.output_nodes.push_back(y);
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  const std::map<std::string, std::string> shape_info;
  auto res = this->Generate(fused_schedule_result, shape_info, "", "0");
  std::string tiling_func_declare {"TilingFunc(gert::TilingSymbolEvalContext *context)\n{\n"};
  auto pos = res["tiling_def_and_tiling_const"].find(tiling_func_declare) + tiling_func_declare.size();
  std::string expect_str {"  context->SetBlockDim(1);\n  *context->GetWorkspaceSizes(1) = 0;\n  return ge::GRAPH_SUCCESS;\n"};
  std::string tiling_func_content = res["tiling_def_and_tiling_const"].substr(pos, expect_str.size());
  EXPECT_EQ(expect_str, tiling_func_content);
}

TEST_F(TestCodegenTiling, TestGenDfxInputSymbolInfo) {
  std::map<std::string, std::string> shape_info;
  shape_info["s0"] = R"([&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(0);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(0);
  }())";
  shape_info["s1"] = R"([&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(0);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(1);
  }())";
  shape_info["s2"] = R"([&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(1);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(0);
  }())";

  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ge::Expression> origin_vars{ge::Symbol("s0"), ge::Symbol("s1"), ge::Symbol("s2")};
  fused_schedule_result.origin_vars = origin_vars;
  auto gen_func = this->GenDfxInputSymbolInfo(fused_schedule_result, shape_info);
  auto expect_func = R"(extern "C" ge::graphStatus DfxInputSymbolInfo(gert::TilingSymbolEvalContext *context, char *out_symbol_info, size_t size)
{
  if (out_symbol_info == nullptr || size == 0) {
    return ge::GRAPH_SUCCESS;
  }
  std::string symbol_info;
  auto s0 = [&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(0);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(0);
  }();
  symbol_info += ("s0: " + std::to_string(s0));

  auto s1 = [&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(0);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(1);
  }();
  symbol_info += (", s1: " + std::to_string(s1));

  auto s2 = [&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(1);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(0);
  }();
  symbol_info += (", s2: " + std::to_string(s2));


  if (symbol_info.empty()) {
    out_symbol_info[0] = '\0';
    return ge::GRAPH_SUCCESS;
  }
  symbol_info += ".";
  if (strncpy_s(out_symbol_info, size, symbol_info.c_str(), std::min(symbol_info.size(), size - 1)) != 0) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}
)";
  EXPECT_EQ(gen_func, expect_func);
}

TEST_F(TestCodegenTiling, TestCompileSuccess) {
  std::stringstream ss;

  ss << "#include <stdexcept>" << std::endl;
  ss << "#include <sstream>" << std::endl;
  ss << "#include <cmath>" << std::endl;
  ss << "#ifndef __CCE_KT_TEST__" << std::endl;
  ss << "#include \"register/op_def_registry.h\"" << std::endl;
  ss << "#include \"exe_graph/runtime/infer_shape_context.h\"" << std::endl;
  ss << "#include \"exe_graph/runtime/kernel_context.h\"" << std::endl;
  ss << "#include \"exe_graph/runtime/continuous_vector.h\"" << std::endl;
  ss << "#endif" << std::endl;

  ss << "#define Max(a, b) ((double)(a) > (double)(b) ? (a) : (b))" << std::endl;
  ss << "#define Min(a, b) ((double)(a) < (double)(b) ? (a) : (b))" << std::endl;
  ss << "#define Log(a) (log((double)(a)))" << std::endl;
  ss << "#define Pow(a, b) pow(a, b)" << std::endl;
  ss << "#define Rational(a, b) ((double)(a) / (double)(b))" << std::endl;

  std::string tiling_context = R"(
namespace gert {
  class TilingSymbolEvalContext : public TilingContext {
    public:
      const gert::Tensor *GetGraphInputTensor(size_t data_index) const {
        auto *tensor = GetInputPointer<gert::Tensor>(data_index + 1);
        if (tensor == nullptr) {
          return nullptr;
        }
        return tensor;
      }
  };
})";

  ss << tiling_context << std::endl;

  std::map<std::string, std::string> shape_info;
  shape_info["s0"] = R"([&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(0);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(0);
  }())";
  shape_info["s1"] = R"([&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(0);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(1);
  }())";
  shape_info["s2"] = R"([&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(1);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(0);
  }())";
  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ge::Expression> origin_vars{ge::Symbol("s0"), ge::Symbol("s1"), ge::Symbol("s2")};
  fused_schedule_result.origin_vars = origin_vars;
  auto dfx_func = this->GenDfxInputSymbolInfo(fused_schedule_result, shape_info);
  ss << dfx_func << std::endl;
  ASSERT_TRUE(CompileCode(ss.str()));
}

/*
 * Codegen FindBestTilingKey测试
 * 1、单graph，单result单group
 * 2、多graph，仅在inductor场景下有，本轮暂不支持
 * 3、单graph，多result组合场景
 *    result1：单group，单graph
 *    result2：单group，多graph
 *    result3：多group场景组合
 *             group1：单graph
 *             group2：多graph
 * 4、enable_group_parallel场景， 不支持生成
 */

TEST_F(TestCodegenTiling, TestGenFindBestTilingKeyFuncFor1Group) {
  ge::AscGraph graph1("graph1");
  ge::ascir_op::Workspace workspace("workspace");
  graph1.AddNode(workspace);

  ge::AscGraph graph2("graph2");
  ge::AscGraph graph3("graph3");

  ascir::ScheduleGroup schedule_group;
  schedule_group.impl_graphs.push_back(graph1);
  schedule_group.impl_graphs.push_back(graph2);
  schedule_group.impl_graphs.push_back(graph3);

  ascir::ScheduledResult schedule_result;
  schedule_result.schedule_groups.push_back(schedule_group);

  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ascir::ScheduledResult> graph0_results = {schedule_result};
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(std::move(graph0_results));

  const std::map<std::string, std::string> shape_info;
  auto res = this->Generate(fused_schedule_result, shape_info, "", "0");

  std::string expect = R"(extern "C" int64_t FindBestTilingKey(AutofuseTilingData &t)
{
  if (t.tiling_key == 0) {
    return 0;
  } else if (t.tiling_key == 1) {
    return 1;
  } else if (t.tiling_key == 2) {
    return 2;
  }
  return -1;
}
)";
  auto pos = res["tiling_def_and_tiling_const"].find("extern \"C\" int64_t FindBestTilingKey(AutofuseTilingData &t)");
  auto func = res["tiling_def_and_tiling_const"].substr(pos, expect.size());
  ASSERT_EQ(func, expect);
}

TEST_F(TestCodegenTiling, TestGenFindBestTilingKeyFuncForMultiResult) {
  ge::AscGraph graph1("graph1");
  ge::AscGraph graph2("graph2");

  ascir::ScheduleGroup schedule_group1;
  schedule_group1.impl_graphs.push_back(graph1);

  ascir::ScheduleGroup schedule_group2;
  schedule_group2.impl_graphs.push_back(graph1);
  schedule_group2.impl_graphs.push_back(graph2);


  // 单group，单graph
  ascir::ScheduledResult schedule_result1;
  schedule_result1.schedule_groups.push_back(schedule_group1);

  // 单group，多graph
  ascir::ScheduledResult schedule_result2;
  schedule_result2.schedule_groups.push_back(schedule_group2);

  // 多group
  // group1单graph
  // group2多graph
  ascir::ScheduledResult schedule_result3;
  schedule_result3.schedule_groups.push_back(schedule_group1);
  schedule_result3.schedule_groups.push_back(schedule_group2);

  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ascir::ScheduledResult> graph0_results = {schedule_result1, schedule_result2, schedule_result3};
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(std::move(graph0_results));

  const std::map<std::string, std::string> shape_info;
  auto res = this->Generate(fused_schedule_result, shape_info, "", "0");

  std::string expect = R"(extern "C" int64_t FindBestTilingKey(AutofuseTilingData &t)
{
  if (t.graph0_tiling_key == 0) {
    if (t.graph0_result0_g0_tiling_data.tiling_key == 0) {
      return 0;
    }
  }  else if (t.graph0_tiling_key == 1) {
    if (t.graph0_result1_g0_tiling_data.tiling_key == 0) {
      return 1;
    } else if (t.graph0_result1_g0_tiling_data.tiling_key == 1) {
      return 2;
    }
  }  else if (t.graph0_tiling_key == 2) {
    if (t.graph0_result2_g0_tiling_data.tiling_key == 0 && t.graph0_result2_g1_tiling_data.tiling_key == 0) {
      return 3;
    } else if (t.graph0_result2_g0_tiling_data.tiling_key == 0 && t.graph0_result2_g1_tiling_data.tiling_key == 1) {
      return 4;
    }
  }
  return -1;
}
)";
  auto pos = res["tiling_def_and_tiling_const"].find("extern \"C\" int64_t FindBestTilingKey(AutofuseTilingData &t)");
  auto func = res["tiling_def_and_tiling_const"].substr(pos, expect.size());
  ASSERT_EQ(func, expect);
}

TEST_F(TestCodegenTiling, TestGenFindBestTilingKeyFuncForEnableParallel) {
  ge::AscGraph graph1("graph1");
  ge::AscGraph graph2("graph2");

  ascir::ScheduleGroup schedule_group1;
  schedule_group1.impl_graphs.push_back(graph1);

  ascir::ScheduleGroup schedule_group2;
  schedule_group2.impl_graphs.push_back(graph1);
  schedule_group2.impl_graphs.push_back(graph2);


  // 单group，单graph
  ascir::ScheduledResult schedule_result1;
  schedule_result1.schedule_groups.push_back(schedule_group1);

  // 单group，多graph
  ascir::ScheduledResult schedule_result2;
  schedule_result2.schedule_groups.push_back(schedule_group2);

  // 多group
  // group1单graph
  // group2多graph
  ascir::ScheduledResult schedule_result3;
  schedule_result3.enable_group_parallel = true;
  schedule_result3.schedule_groups.push_back(schedule_group1);
  schedule_result3.schedule_groups.push_back(schedule_group2);

  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ascir::ScheduledResult> graph0_results = {schedule_result1, schedule_result2, schedule_result3};
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(std::move(graph0_results));

  const std::map<std::string, std::string> shape_info;
  auto res = this->Generate(fused_schedule_result, shape_info, "", "0");

  auto pos = res["tiling_def_and_tiling_const"].find("extern \"C\" int64_t FindBestTilingKey(AutofuseTilingData &t)");
  ASSERT_EQ(pos, std::string::npos);
}

TEST_F(TestCodegenTiling, TestGenExternTilingFunc) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<ge::RuntimeStubV2Common>();
  ge::RuntimeStub::SetInstance(stub_v2);
  ge::AscGraph graph1("graph1");
  ge::AscGraph graph2("graph2");

  ascir::ScheduleGroup schedule_group1;
  schedule_group1.impl_graphs.push_back(graph1);

  ascir::ScheduleGroup schedule_group2;
  schedule_group2.impl_graphs.push_back(graph1);
  schedule_group2.impl_graphs.push_back(graph2);


  // 单group，单graph
  ascir::ScheduledResult schedule_result1;
  schedule_result1.schedule_groups.push_back(schedule_group1);

  // 单group，多graph
  ascir::ScheduledResult schedule_result2;
  schedule_result2.schedule_groups.push_back(schedule_group2);

  // 多group
  // group1单graph
  // group2多graph
  ascir::ScheduledResult schedule_result3;
  schedule_result3.enable_group_parallel = true;
  schedule_result3.schedule_groups.push_back(schedule_group1);
  schedule_result3.schedule_groups.push_back(schedule_group2);

  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ascir::ScheduledResult> graph0_results = {schedule_result1, schedule_result2, schedule_result3};
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(std::move(graph0_results));

  const std::map<std::string, std::string> shape_info;
  auto res = this->Generate(fused_schedule_result, shape_info, "", "0");

  auto pos = res["tiling_def_and_tiling_const"].find("extern \"C\" int64_t FindBestTilingKey(AutofuseTilingData &t)");
  ASSERT_EQ(pos, std::string::npos);
  ge::RuntimeStub::Reset();
  ge::PlatformContext::GetInstance().Reset();
}

TEST_F(TestCodegenTiling, TestPGOSearchTensorMallocDef) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", ge::ops::One);

  ge::ascir_op::Data x_op("x", graph);
  x_op.ir_attr.SetIndex(0);
  ge::ascir_op::Load load_op("load");
  ge::ascir_op::Store store_op("store");
  ge::ascir_op::Output y_op("y");
  y_op.ir_attr.SetIndex(0);
  graph.AddNode(load_op);
  graph.AddNode(store_op);
  graph.AddNode(y_op);

  load_op.x = x_op.y;
  load_op.y.dtype = ge::DT_FLOAT16;
  store_op.x = load_op.y;
  y_op.x = store_op.y;

  auto x = graph.FindNode("x");
  auto load = graph.FindNode("load");
  auto store = graph.FindNode("store");
  auto y = graph.FindNode("y");

  x->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->attr.api.unit = ge::ComputeUnit::kUnitNone;
  y->attr.api.unit = ge::ComputeUnit::kUnitNone;

  load->outputs[0].attr.axis = {z0.id};
  load->outputs[0].attr.vectorized_axis = {z0.id};
  load->outputs[0].attr.vectorized_strides = {ge::ops::One};
  load->outputs[0].attr.repeats = {z0.size};
  load->outputs[0].attr.strides = {ge::ops::One};
  load->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.mem.tensor_id = 1;
  load->outputs[0].attr.que.id = 0;
  load->outputs[0].attr.mem.reuse_id = 0;
  load->outputs[0].attr.que.depth = 2;
  load->outputs[0].attr.que.buf_num = 2;
  load->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  store->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.axis = {z0.id};
  store->outputs[0].attr.vectorized_axis = {z0.id};
  store->outputs[0].attr.vectorized_strides = {ge::ops::One};
  store->outputs[0].attr.repeats = {z0.size};
  store->outputs[0].attr.strides = {ge::ops::One};

  y->inputs[0].attr.repeats = {z0.size};
  y->inputs[0].attr.strides = {ge::ops::One};
  y->inputs[0].attr.dtype = ge::DT_FLOAT16;

  ::ascir::ScheduledResult schedule_result;
  schedule_result.schedule_groups.resize(1);
  for (auto &schedule_group : schedule_result.schedule_groups) {
    schedule_group.impl_graphs.emplace_back(graph);
  }
  std::vector<ascir::ScheduledResult> schedule_results;
  schedule_results.push_back(schedule_result);
  schedule_results.push_back(schedule_result);

  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = ge::AscendString(graph.GetName().c_str());
  fused_schedule_result.input_nodes.push_back(x);
  fused_schedule_result.output_nodes.push_back(y);
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);

  std::string mallocdef = this->PGOSearchTensorMallocDef(fused_schedule_result);
  const std::string expect = R"(  size_t input0_size = 2;
  ret = aclrtMalloc(&input0, input0_size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    DLOGE("aclrtMalloc input0 failed. ERROR: %d", ret);
    return FAILED;
  }
  size_t output0_size = 2;
  ret = aclrtMalloc(&output0, output0_size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    DLOGE("aclrtMalloc output0 failed. ERROR: %d", ret);
    return FAILED;
  }
)";
  ASSERT_EQ(mallocdef, expect);
}

TEST_F(TestCodegenTiling, TestCalculateTensorMemorySizeStrWithNoRepeatsOrStrides) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", ge::ops::One);

  ge::ascir_op::Data x_op("x", graph);
  x_op.ir_attr.SetIndex(0);

  auto x = graph.FindNode("x");

  x->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->attr.api.unit = ge::ComputeUnit::kUnitNone;
  x->outputs[0].attr.repeats = {};
  x->outputs[0].attr.strides = {};

  std::string memory_size = this->CalculateTensorMemorySizeStr(x->outputs[0]);
  const std::string expect = "0";
  ASSERT_EQ(memory_size, expect);
}

TEST_F(TestCodegenTiling, TestCalculateTensorMemorySizeStrWithZeroFirstStride) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data x_op("x", graph);
  x_op.ir_attr.SetIndex(0);

  auto x = graph.FindNode("x");

  x->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->attr.api.unit = ge::ComputeUnit::kUnitNone;
  x->outputs[0].attr.dtype = ge::DT_FLOAT16;
  x->outputs[0].attr.axis = {z0.id, z1.id};
  x->outputs[0].attr.repeats = {s0, s1};
  x->outputs[0].attr.strides = {ge::ops::Zero, ge::ops::One};

  std::string memory_size = this->CalculateTensorMemorySizeStr(x->outputs[0]);
  const std::string expect = std::string(ge::sym::Mul(s1, ge::Expression::Parse("2")).Simplify().Str().get());
  ASSERT_EQ(memory_size, expect);
}

TEST_F(TestCodegenTiling, TestCalculateTensorMemorySizeStrWithOnlyZeroStride) {
  ge::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  ge::ascir_op::Data x_op("x", graph);
  x_op.ir_attr.SetIndex(0);

  auto x = graph.FindNode("x");

  x->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->attr.api.unit = ge::ComputeUnit::kUnitNone;
  x->outputs[0].attr.dtype = ge::DT_FLOAT16;
  x->outputs[0].attr.axis = {z0.id};
  x->outputs[0].attr.repeats = {s0};
  x->outputs[0].attr.strides = {ge::ops::Zero};

  std::string memory_size = this->CalculateTensorMemorySizeStr(x->outputs[0]);
  const std::string expect = std::string(ge::sym::Mul(ge::ops::One, ge::Expression::Parse("2")).Simplify().Str().get());
  ASSERT_EQ(memory_size, expect);
}

void CreateMatmulGraph(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(31);
  auto s1 = graph.CreateSizeVar(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  ge::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  ge::ascir_op::Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  *data1.y.repeats = {ge::ops::One, ge::ops::One};
  *data1.y.strides = {ge::ops::Zero, ge::ops::Zero};
  data1.ir_attr.SetIndex(1);

  ge::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *load1.y.repeats = {ge::ops::One, ge::ops::One};

  ge::ascir_op::BatchMatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, ge::ops::One};
  matmul.ir_attr.SetAdj_x1(1);
  matmul.ir_attr.SetAdj_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  ge::ascir_op::Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = matmul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1 ,ge::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(ge::ops::One);

  ge::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
}

TEST_F(TestCodegenTiling, TestMatmulElemwiseFuse) {
  ge::AscGraph graph("matmul_elemwise_pro");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  ge::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  *data0.y.strides = {s1 ,ge::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  ge::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1 ,ge::ops::One};
  *load0.y.repeats = {s0, s1};

  ge::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load0.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, ge::ops::One};
  abs.attr.api.compute_type = ge::ComputeType::kComputeElewise;

  ge::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.attr.sched.axis = {z0.id, z1.id};
  scalar0.ir_attr.SetValue("0");
  scalar0.y.dtype = ge::DT_FLOAT;
  *scalar0.y.axis = {z0.id, z1.id};
  *scalar0.y.repeats = {ge::ops::One, ge::ops::One};
  *scalar0.y.strides = {ge::ops::Zero, ge::ops::Zero};

  ge::ascir_op::Broadcast broadcast0("broadcast0");
  broadcast0.x = scalar0.y;
  broadcast0.attr.sched.axis = {z0.id, z1.id};
  *broadcast0.y.axis = {z0.id, z1.id};
  broadcast0.y.dtype = ge::DT_FLOAT;
  *broadcast0.y.repeats = {ge::ops::One, s1};
  *broadcast0.y.strides = {ge::ops::Zero, ge::ops::One};

  ge::ascir_op::Broadcast broadcast1("broadcast1");
  broadcast1.x = broadcast0.y;
  broadcast1.attr.sched.axis = {z0.id, z1.id};
  *broadcast1.y.axis = {z0.id, z1.id};
  broadcast1.y.dtype = ge::DT_FLOAT;
  *broadcast1.y.repeats = {s0, s1};
  *broadcast1.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id};
  add_op.x1 = abs.y;
  add_op.x2 = broadcast1.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id};
  *add_op.y.repeats = {s0, s1};
  *add_op.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  *data1.y.repeats = {ge::ops::One, ge::ops::One};
  *data1.y.strides = {ge::ops::Zero, ge::ops::Zero};
  data1.ir_attr.SetIndex(1);

  ge::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {ge::ops::Zero, ge::ops::Zero};
  *load1.y.repeats = {ge::ops::One, ge::ops::One};

  ge::ascir_op::Broadcast broadcast2("broadcast2");
  broadcast2.x = load1.y;
  broadcast2.attr.sched.axis = {z0.id, z1.id};
  *broadcast2.y.axis = {z0.id, z1.id};
  broadcast2.y.dtype = ge::DT_FLOAT;
  *broadcast2.y.repeats = {ge::ops::One, s1};
  *broadcast2.y.strides = {ge::ops::Zero, ge::ops::One};

  ge::ascir_op::Broadcast broadcast3("broadcast3");
  broadcast3.x = broadcast2.y;
  broadcast3.attr.sched.axis = {z0.id, z1.id};
  *broadcast3.y.axis = {z0.id, z1.id};
  broadcast3.y.dtype = ge::DT_FLOAT;
  *broadcast3.y.repeats = {s0, s1};
  *broadcast3.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Mul mul("mul");
  mul.attr.sched.axis = {z0.id, z1.id};
  mul.x1 = add_op.y;
  mul.x2 = broadcast3.y;
  mul.y.dtype = ge::DT_FLOAT;
  *mul.y.axis = {z0.id, z1.id};
  *mul.y.repeats = {s0, s1};
  *mul.y.strides = {s1, ge::ops::One};

  ge::ascir_op::Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = mul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1 ,ge::ops::One};
  *store_op.y.repeats = {s0, s1};

  ge::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto x1Local = graph.FindNode("data0");
  x1Local->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  x1Local->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  x1Local->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;

  ge::AscGraph mm_graph("mutmul");
  CreateMatmulGraph(mm_graph);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);

  ascir::ScheduleGroup schedule_group2;
  schedule_group2.impl_graphs.push_back(mm_graph);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.push_back(schedule_group2);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].cube_type = ascir::CubeTemplateType::kUBFuse;;
  const std::map<std::string, std::string> shape_info;
  auto res = this->Generate(fused_schedule_result, shape_info, "", "0");

  std::fstream tiling_func("Mutmul_fuse_tiling_func.cpp", std::ios::out);
  tiling_func << res["tiling_def_and_tiling_const"];

  auto pos = res["tiling_def_and_tiling_const"].find("extern \"C\" int64_t FindBestTilingKey(AutofuseTilingData &t)");
  ASSERT_NE(pos, std::string::npos);
}
