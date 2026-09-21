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
#include <vector>
#include "framework/runtime/model_rt_var_manager.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "exec_runtime/execution_runtime_utils.h"

using namespace std;
using namespace gert;

namespace ge {
static uint64_t default_session_id = 0xff66ff;
class UtestRtVarManager : public testing::Test {
 protected:
  void SetUp() {
    VarManagerPool::Instance().Destory();
  }
  void TearDown() {
    VarManagerPool::Instance().Destory();
  }
};

TEST(UtestRtVarManager, init_success) {
  RtVarManagerPool::Instance().session_id_to_var_manager_.clear();
  EXPECT_EQ(ModelRtVarManager::Instance(default_session_id)->Init(0, 0, 20480, nullptr, 0), SUCCESS);
  EXPECT_EQ(ModelRtVarManager::Instance(default_session_id)->Init(0, 0, 20480, nullptr, 0), SUCCESS);
  EXPECT_EQ(RtVarManagerPool::Instance().session_id_to_var_manager_.size(), 1UL);
  RtVarManagerPool::Instance().RemoveRtVarManager(default_session_id);
  EXPECT_EQ(RtVarManagerPool::Instance().session_id_to_var_manager_.size(), 0UL);
}

TEST(UtestRtVarManager, restore_varibles) {
  auto rt_var_manager = ModelRtVarManager::Instance(default_session_id);
  ASSERT_NE(rt_var_manager, nullptr);
  EXPECT_EQ(rt_var_manager->Init(0, 0, 204800, nullptr, 0), SUCCESS);

  auto var_manager = VarManager::Instance(default_session_id);
  ASSERT_NE(var_manager, nullptr);
  std::vector<NodePtr> var_nodes;
  for (int i = 0; i < 10; ++i) {
    ge::OpDescPtr op_desc_i = std::make_shared<OpDesc>("var" + std::to_string(i), "Variable");
    GeTensorDesc desc(GeShape({1, 1, 4, 4}));
    ge::TensorUtils::SetSize(desc, 64);
    op_desc_i->AddOutputDesc(desc);
    op_desc_i->SetOutputOffset({static_cast<int64_t>(var_manager->GetVarMemLogicBase() + 2048 * i)});
    var_nodes.push_back(NodeUtils::CreatNodeWithoutGraph(op_desc_i));
  }

  ge::OpDescPtr op_desc_i = std::make_shared<OpDesc>("constant", "CONSTANTOP");
  GeTensorDesc desc(GeShape({1, 1, 4, 4}));
  op_desc_i->AddOutputDesc(desc);
  op_desc_i->SetOutputOffset({static_cast<int64_t>(var_manager->GetVarMemLogicBase() + 20480)});
  var_nodes.push_back(NodeUtils::CreatNodeWithoutGraph(op_desc_i));

  EXPECT_EQ(rt_var_manager->RestoreDeviceVariables(var_nodes, 0, 0), SUCCESS);
  gert::StorageShape rt_shape;
  gert::TensorData rt_tensor_data;
  EXPECT_EQ(rt_var_manager->GetVarShapeAndMemory("var0", rt_shape, rt_tensor_data), SUCCESS);
  EXPECT_NE(rt_var_manager->GetVarShapeAndMemory("constant", rt_shape, rt_tensor_data), SUCCESS);
  rt_var_manager->name_to_var_info_.clear();
  VarManager::Instance(default_session_id)->Destory();
}

TEST(UtestRtVarManager, restore_varibles_auto_malloc) {
  auto rt_var_manager = ModelRtVarManager::Instance(default_session_id);
  ASSERT_NE(rt_var_manager, nullptr);
  EXPECT_EQ(rt_var_manager->Init(0, 0, 204800, nullptr, 0), SUCCESS);

  auto var_manager = VarManager::Instance(default_session_id);
  ASSERT_NE(var_manager, nullptr);
  ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
  ModelRtVarManager::VarInfo tmp;
  rt_var_manager->name_to_var_info_["var0"] = tmp;
  std::vector<NodePtr> var_nodes;
  for (int i = 0; i < 10; ++i) {
    ge::OpDescPtr op_desc_i = std::make_shared<OpDesc>("var" + std::to_string(i), "Variable");
    GeTensorDesc desc(GeShape({1, 1, 4, 4}));
    ge::TensorUtils::SetSize(desc, 64);
    op_desc_i->AddOutputDesc(desc);
    op_desc_i->SetOutputOffset({static_cast<int64_t>(var_manager->GetVarMemLogicBase() + 2048 * i)});
    var_nodes.push_back(NodeUtils::CreatNodeWithoutGraph(op_desc_i));
  }

  ge::OpDescPtr op_desc_i = std::make_shared<OpDesc>("constant", "CONSTANTOP");
  GeTensorDesc desc(GeShape({1, 1, 4, 4}));
  op_desc_i->AddOutputDesc(desc);
  op_desc_i->SetOutputOffset({static_cast<int64_t>(var_manager->GetVarMemLogicBase() + 20480)});
  var_nodes.push_back(NodeUtils::CreatNodeWithoutGraph(op_desc_i));

  EXPECT_EQ(rt_var_manager->RestoreDeviceVariables(var_nodes, 0, 0), SUCCESS);
  gert::StorageShape rt_shape;
  gert::TensorData rt_tensor_data;
  EXPECT_EQ(rt_var_manager->GetVarShapeAndMemory("var0", rt_shape, rt_tensor_data), SUCCESS);
  EXPECT_NE(rt_var_manager->GetVarShapeAndMemory("constant", rt_shape, rt_tensor_data), SUCCESS);
  VarManager::Instance(default_session_id)->Destory();
}

TEST(UtestRtVarManager, restore_varibles_auto_malloc_helper_enable) {
  auto rt_var_manager = ModelRtVarManager::Instance(default_session_id);
  ASSERT_NE(rt_var_manager, nullptr);
  EXPECT_EQ(rt_var_manager->Init(0, 0, 204800, nullptr, 0), SUCCESS);

  auto var_manager = VarManager::Instance(default_session_id);
  ASSERT_NE(var_manager, nullptr);
  ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
  ModelRtVarManager::VarInfo tmp;
  rt_var_manager->name_to_var_info_["var0"] = tmp;
  std::vector<NodePtr> var_nodes;
  for (int i = 0; i < 10; ++i) {
    ge::OpDescPtr op_desc_i = std::make_shared<OpDesc>("var" + std::to_string(i), "Variable");
    GeTensorDesc desc(GeShape({1, 1, 4, 4}));
    ge::TensorUtils::SetSize(desc, 64);
    op_desc_i->AddOutputDesc(desc);
    op_desc_i->SetOutputOffset({static_cast<int64_t>(var_manager->GetVarMemLogicBase() + 2048 * i)});
    var_nodes.push_back(NodeUtils::CreatNodeWithoutGraph(op_desc_i));
  }

  ge::OpDescPtr op_desc_i = std::make_shared<OpDesc>("constant", "CONSTANTOP");
  GeTensorDesc desc(GeShape({1, 1, 4, 4}));
  op_desc_i->AddOutputDesc(desc);
  op_desc_i->SetOutputOffset({static_cast<int64_t>(var_manager->GetVarMemLogicBase() + 20480)});
  var_nodes.push_back(NodeUtils::CreatNodeWithoutGraph(op_desc_i));

  EXPECT_EQ(rt_var_manager->RestoreDeviceVariables(var_nodes, 0, 0), SUCCESS);
  gert::StorageShape rt_shape;
  gert::TensorData rt_tensor_data;
  EXPECT_EQ(rt_var_manager->GetVarShapeAndMemory("var0", rt_shape, rt_tensor_data), SUCCESS);
  EXPECT_NE(rt_var_manager->GetVarShapeAndMemory("constant", rt_shape, rt_tensor_data), SUCCESS);
  VarManager::Instance(default_session_id)->Destory();
}

TEST(UtestRtVarManager, restore_varibles_external_var) {
  auto rt_var_manager = ModelRtVarManager::Instance(default_session_id);
  ASSERT_NE(rt_var_manager, nullptr);
  void *var_addr = nullptr;
  rtMalloc(&var_addr, 204800, RT_MEMORY_DEFAULT,GE_MODULE_NAME_U16);
  EXPECT_EQ(rt_var_manager->Init(0, 0, 204800, var_addr, 204800), SUCCESS);

  auto var_manager = VarManager::Instance(default_session_id);
  ASSERT_NE(var_manager, nullptr);
  ge::ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
  ModelRtVarManager::VarInfo tmp;
  rt_var_manager->name_to_var_info_["var0"] = tmp;
  std::vector<NodePtr> var_nodes;
  for (int i = 0; i < 10; ++i) {
    ge::OpDescPtr op_desc_i = std::make_shared<OpDesc>("var" + std::to_string(i), "Variable");
    GeTensorDesc desc(GeShape({1, 1, 4, 4}));
    ge::TensorUtils::SetSize(desc, 64);
    op_desc_i->AddOutputDesc(desc);
    op_desc_i->SetOutputOffset({static_cast<int64_t>(var_manager->GetVarMemLogicBase() + 2048 * i)});
    var_nodes.push_back(NodeUtils::CreatNodeWithoutGraph(op_desc_i));
  }

  ge::OpDescPtr op_desc_i = std::make_shared<OpDesc>("constant", "CONSTANTOP");
  GeTensorDesc desc(GeShape({1, 1, 4, 4}));
  op_desc_i->AddOutputDesc(desc);
  op_desc_i->SetOutputOffset({static_cast<int64_t>(var_manager->GetVarMemLogicBase() + 20480)});
  var_nodes.push_back(NodeUtils::CreatNodeWithoutGraph(op_desc_i));

  EXPECT_EQ(rt_var_manager->RestoreDeviceVariables(var_nodes, 0, 0), SUCCESS);
  gert::StorageShape rt_shape;
  gert::TensorData rt_tensor_data;
  EXPECT_EQ(rt_var_manager->GetVarShapeAndMemory("var0", rt_shape, rt_tensor_data), SUCCESS);
  EXPECT_NE(rt_var_manager->GetVarShapeAndMemory("constant", rt_shape, rt_tensor_data), SUCCESS);
  rtFree(var_addr);
  VarManager::Instance(default_session_id)->Destory();
}
}  // namespace ge
