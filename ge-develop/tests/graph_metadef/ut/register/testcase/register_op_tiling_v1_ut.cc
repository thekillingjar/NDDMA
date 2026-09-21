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
#include "graph/ge_tensor.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/op_desc.h"
#include "graph/operator.h"
#include "graph/compute_graph.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "register/op_tiling_registry.h"
#include "op_tiling/op_tiling_utils.h"
#include "op_tiling/op_tiling_constants.h"
#include "op_tiling.h"

using namespace std;
using namespace ge;

namespace optiling {
class RegisterOpTilingV1UT : public testing::Test {
protected:
  void SetUp() {}

  void TearDown() {}
};
bool op_tiling_stub_v1(const TeOpParas &op_paras, const OpCompileInfo &compile_info, OpRunInfo &run_info) {
  return true;
}

static string parse_int(const std::stringstream& tiling_data) {
  auto data = tiling_data.str();
  string result;
  int32_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }

  return result;
}

static string parse_int(void *const addr_base, const uint64_t size) {
  char *data = reinterpret_cast<char *>(addr_base);
  string result;
  int32_t tmp = 0;
  for (size_t i = 0; i < size; i += sizeof(int32_t)) {
    memcpy(&tmp, data + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }

  return result;
}

REGISTER_OP_TILING(ReluV1, op_tiling_stub_v1);
//REGISTER_OP_TILING(DynamicAtomicAddrClean, op_tiling_stub_v1);

TEST_F(RegisterOpTilingV1UT, op_para_calculate_v1_1) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV1");
  GeShape shape({4,3,14,14});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);
  string compile_info_key = "compile_info_key";
  string compile_info_json = "compile_info_jsoncompile_info_jsoncompile_info_jsoncompile_info_jsoncompile_info_json";
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, compile_info_json);

  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);
  EXPECT_EQ(op_desc->GetTilingFuncInfo(), nullptr);
  auto op = OpDescUtils::CreateOperatorFromNode(node);
  utils::OpRunInfo run_info;
  graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  auto &op_func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  auto iter = op_func_map.find("ReluV1");
  EXPECT_NE(iter, op_func_map.end());
  EXPECT_NE(&(iter->second), nullptr);
  EXPECT_EQ(op_desc->GetTilingFuncInfo(), reinterpret_cast<void *>(&(iter->second)));
  EXPECT_EQ(OpParaCalculateV2(op, run_info), GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingV1UT, op_para_calculate_v1_2) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluVVV");
  GeShape shape({4,3,14,14});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);
  string compile_info_key = "compile_info_key";
  string compile_info_json = "compile_info_jsoncompile_info_jsoncompile_info_jsoncompile_info_jsoncompile_info_json";
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, compile_info_json);

  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);
  utils::OpRunInfo run_info;
  auto op = OpDescUtils::CreateOperatorFromNode(node);
  graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, GRAPH_FAILED);

  OpTilingFuncInfo op_func_info(OP_TYPE_AUTO_TILING);
  op_func_info.tiling_func_ = op_tiling_stub_v1;
  std::unordered_map<std::string, OpTilingFuncInfo> &tiling_func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  tiling_func_map.emplace(OP_TYPE_AUTO_TILING, op_func_info);
  EXPECT_EQ(op_desc->GetTilingFuncInfo(), nullptr);
  ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  auto &op_func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  auto iter = op_func_map.find(OP_TYPE_AUTO_TILING);
  EXPECT_NE(iter, op_func_map.end());
  EXPECT_NE(&(iter->second), nullptr);
  EXPECT_EQ(op_desc->GetTilingFuncInfo(), reinterpret_cast<void *>(&(iter->second)));
  tiling_func_map.erase(OP_TYPE_AUTO_TILING);
}

TEST_F(RegisterOpTilingV1UT, op_para_calculate_v1_3) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV1");
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);
  string compile_info_key = "compile_info_key";
  string compile_info_json = "compile_info_jsoncompile_info_jsoncompile_info_jsoncompile_info_jsoncompile_info_json";
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, compile_info_json);

  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);
  utils::OpRunInfo run_info;
  auto op = OpDescUtils::CreateOperatorFromNode(node);
  graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingV1UT, op_para_calculate_v1_4) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV1");
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);
  string compile_info_key = "compile_info_key";
  string compile_info_json = "compile_info_jsoncompile_info_jsoncompile_info_jsoncompile_info_jsoncompile_info_json";
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, compile_info_json);

  vector<string> depend_names = {"x"};
  AttrUtils::SetListStr(op_desc, "_op_infer_depends", depend_names);

  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);
  utils::OpRunInfo run_info;
  auto op = OpDescUtils::CreateOperatorFromNode(node);
  graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingV1UT, op_atomic_calculate_v1_1) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginShape(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);
  string compile_info_key = "compile_info_key";
  string compile_info_json = "{\"_workspace_size_list\":[]}";
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_JSON, compile_info_json);
  std::vector<int64_t> atomic_output_indices = {0};
  (void) ge::AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);

  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);

  std::unordered_map<std::string, OpTilingFuncInfo> &tiling_func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  OpTilingFuncInfo op_func_info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  op_func_info.tiling_func_ = op_tiling_stub_v1;
  tiling_func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, op_func_info);
  utils::OpRunInfo run_info;
  EXPECT_EQ(op_desc->GetAtomicTilingFuncInfo(), nullptr);
  graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  auto &op_func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  auto iter = op_func_map.find(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  EXPECT_NE(iter, op_func_map.end());
  EXPECT_NE(&(iter->second), nullptr);
  EXPECT_EQ(op_desc->GetAtomicTilingFuncInfo(), reinterpret_cast<void *>(&(iter->second)));
  tiling_func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingV1UT, op_atomic_calculate_v1_2) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);
  string compile_info_key = "compile_info_key";
  string compile_info_json = "{\"_workspace_size_list\":[]}";
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_JSON, compile_info_json);

  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);

  std::unordered_map<std::string, OpTilingFuncInfo> &tiling_func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  OpTilingFuncInfo op_func_info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  op_func_info.tiling_func_ = op_tiling_stub_v1;
  tiling_func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, op_func_info);
  utils::OpRunInfo run_info;
  graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, GRAPH_FAILED);
  tiling_func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingV1UT, op_atomic_calculate_v1_3) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);
  string compile_info_key = "compile_info_key";
  string compile_info_json = "{\"_workspace_size_list\":[]}";
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_JSON, compile_info_json);
  std::vector<int64_t> atomic_output_indices = {1};
  (void) ge::AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);

  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);

  std::unordered_map<std::string, OpTilingFuncInfo> &tiling_func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  OpTilingFuncInfo op_func_info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  op_func_info.tiling_func_ = op_tiling_stub_v1;
  tiling_func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, op_func_info);
  utils::OpRunInfo run_info;
  graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, GRAPH_FAILED);
  tiling_func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingV1UT, add_tiling_data) {
  utils::OpRunInfo run_info;

  int32_t data1 = 123;
  int32_t data2 = 456;
  run_info.AddTilingData(data1);
  run_info.AddTilingData(data2);
  run_info.SetClearAtomic(false);
  run_info.SetTilingKey(100);
  run_info.SetTilingCond(-1);
  std::vector<int64_t> workspace{1, 2, 3};
  run_info.SetWorkspaces(workspace);
  std::string get_data = parse_int(run_info.GetAllTilingData());
  EXPECT_EQ(get_data, "123 456 ");
  EXPECT_EQ(run_info.GetClearAtomic(), false);
  EXPECT_EQ(run_info.GetTilingKey(), 100);
  EXPECT_EQ(run_info.GetWorkspaceNum(), 3);
  EXPECT_EQ(run_info.GetAllWorkspaces(), workspace);
  EXPECT_EQ(run_info.GetTilingCond(), -1);

  run_info.GetAllTilingData().str("");
  run_info.AddTilingData(reinterpret_cast<const char *>(&data2), sizeof(int));
  get_data = parse_int(run_info.GetAllTilingData());
  EXPECT_EQ(get_data, "456 ");

  char arg_base[20] = {0};
  run_info.ResetWorkspace();
  run_info.ResetAddrBase(arg_base, sizeof(arg_base));
  data1 = 78;
  data2 = 90;
  run_info.AddTilingData(data1);
  run_info.AddTilingData(data2);
  run_info.AddWorkspace(4);
  run_info.AddWorkspace(5);
  get_data = parse_int(arg_base, sizeof(int) * 2);
  EXPECT_EQ(get_data, "78 90 ");
  workspace = std::vector<int64_t>{4, 5};
  EXPECT_EQ(run_info.GetAllWorkspaces(), workspace);

  run_info.ResetWorkspace();
  run_info.ResetAddrBase(arg_base, sizeof(arg_base));
  run_info.AddTilingData(reinterpret_cast<const char *>(&data2), sizeof(int));
  run_info.AddWorkspace(6);
  get_data = parse_int(arg_base, sizeof(int));
  EXPECT_EQ(get_data, "90 ");
  workspace = std::vector<int64_t>{6};
  EXPECT_EQ(run_info.GetAllWorkspaces(), workspace);
}
}
