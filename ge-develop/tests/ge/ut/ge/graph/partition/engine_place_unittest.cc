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

#include "compute_graph.h"
#include "graph/partition/engine_place.h"
#include "engines/manager/engine_manager/dnnengine_manager.h"
#include "graph/debug/ge_attr_define.h"


namespace ge {

class UtestEnginePlace : public testing::Test {
  protected:
    void SetUp() {}

    void TearDown() {}
};

TEST_F(UtestEnginePlace, select_engine_by_attr_when_opdesc_empty) {
  std::string engine_name = "engine_name";
  std::string kernel_name = "kernel_name";
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  auto op_desc = std::make_shared<OpDesc>("mock_op_name", "mock_op_type");
  AttrUtils::SetStr(op_desc, ATTR_NAME_ENGINE_NAME_FOR_LX, engine_name);
  AttrUtils::SetStr(op_desc, ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX, kernel_name);
  auto node_ptr = graph->AddNode(op_desc);

  EnginePlacer engine_place(graph);
  bool is_check_support_success = true;
  std::set<std::string> exclude_engines;
  DNNEngineManager::GetExcludeEngines(exclude_engines);
  OpInfo op_info;
  ASSERT_EQ(engine_place.SelectEngine(node_ptr, exclude_engines, is_check_support_success, op_info), SUCCESS);

  ASSERT_EQ(op_desc->GetOpEngineName(), engine_name);
  ASSERT_EQ(op_info.opKernelLib, kernel_name);
}

TEST_F(UtestEnginePlace, select_engine_by_when_opdesc_and_attr_empty) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  auto op_desc = std::make_shared<OpDesc>("mock_op_name", "mock_op_type");
  auto node_ptr = graph->AddNode(op_desc);

  EnginePlacer engine_place(graph);
  bool is_check_support_success = true;
  std::set<std::string> exclude_engines;
  DNNEngineManager::GetExcludeEngines(exclude_engines);
  OpInfo op_info;
  ASSERT_EQ(engine_place.SelectEngine(node_ptr, exclude_engines, is_check_support_success, op_info), FAILED);

  ASSERT_TRUE(op_desc->GetOpEngineName().empty());
}

TEST_F(UtestEnginePlace, select_engine_when_opdesc_not_empty) {
  std::string engine_name = "engine_name";
  std::string kernel_name = "kernel_name";
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  auto op_desc = std::make_shared<OpDesc>("mock_op_name", "mock_op_type");
  op_desc->SetOpEngineName(engine_name);
  op_desc->SetOpKernelLibName(kernel_name);
  auto node_ptr = graph->AddNode(op_desc);

  EnginePlacer engine_place(graph);
  bool is_check_support_success = true;
  std::set<std::string> exclude_engines;
  DNNEngineManager::GetExcludeEngines(exclude_engines);
  OpInfo op_info;
  ASSERT_EQ(engine_place.SelectEngine(node_ptr, exclude_engines, is_check_support_success, op_info), SUCCESS);
  DNNEngineManager::UpdateOpDescWithOpInfo(node_ptr->GetOpDesc(), op_info);

  std::string attr_engine_name;
  AttrUtils::GetStr(op_desc, ATTR_NAME_ENGINE_NAME_FOR_LX, attr_engine_name);
  ASSERT_EQ(attr_engine_name, engine_name);
  std::string attr_kernel_name;
  AttrUtils::GetStr(op_desc, ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX, attr_kernel_name);
  ASSERT_EQ(attr_kernel_name, kernel_name);
}

TEST_F(UtestEnginePlace, select_engine_when_opdesc_confilct_with_attr) {
  std::string op_engine_name = "op_engine_name";
  std::string op_kernel_name = "op_kernel_name";
  std::string attr_engine_name = "attr_engine_name";
  std::string attr_kernel_name = "attr_kernel_name";
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  auto op_desc = std::make_shared<OpDesc>("mock_op_name", "mock_op_type");
  op_desc->SetOpEngineName(op_engine_name);
  op_desc->SetOpKernelLibName(op_kernel_name);
  AttrUtils::SetStr(op_desc, ATTR_NAME_ENGINE_NAME_FOR_LX, attr_engine_name);
  AttrUtils::SetStr(op_desc, ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX, attr_kernel_name);
  auto node_ptr = graph->AddNode(op_desc);

  EnginePlacer engine_place(graph);
  bool is_check_support_success = true;
  std::set<std::string> exclude_engines;
  DNNEngineManager::GetExcludeEngines(exclude_engines);
  OpInfo op_info;
  ASSERT_EQ(engine_place.SelectEngine(node_ptr, exclude_engines, is_check_support_success, op_info), SUCCESS);

  std::string fetched_attr_engine_name;
  AttrUtils::GetStr(op_desc, ATTR_NAME_ENGINE_NAME_FOR_LX, fetched_attr_engine_name);
  ASSERT_EQ(fetched_attr_engine_name, attr_engine_name);
  std::string fetched_attr_kernel_name;
  AttrUtils::GetStr(op_desc, ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX, fetched_attr_kernel_name);
  ASSERT_EQ(fetched_attr_kernel_name, attr_kernel_name);

  ASSERT_EQ(op_desc->GetOpEngineName(), op_engine_name);
  ASSERT_EQ(op_desc->GetOpKernelLibName(), op_kernel_name);
}

} // namespace ge
