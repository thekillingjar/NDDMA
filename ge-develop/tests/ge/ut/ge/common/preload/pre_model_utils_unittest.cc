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

#include "macro_utils/dt_public_scope.h"
#include "common/preload/model/pre_model_utils.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "framework/common/types.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;

namespace ge {
class UtestPreModelUtils : public testing::Test {};

TEST_F(UtestPreModelUtils, GetWorkspaceDataAddrs_tensor_mem_type) {
  PreRuntimeParam runtime_param;
  uint8_t mem_base_addr = 0;
  runtime_param.mem_size = 0x20002000u;
  runtime_param.logic_mem_base = reinterpret_cast<uintptr_t>(&mem_base_addr);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr add_node = CreateNode(*graph, "add", ADD, 2, 1);
  auto op_desc = add_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);

  op_desc->SetWorkspace({0x10001000, 0x10002000, 0, 0x10004000, 0x10008000, 0x1000c000});
  op_desc->SetWorkspaceBytes({0x1000, 0x2000, 0, 0x1000, 0x3000, 0x5000});

  vector<int64_t> v_memory_type{RT_MEMORY_L1};
  AttrUtils::SetListInt(op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE, v_memory_type);
  vector<uint64_t> args_offset_values;
  vector<KernelArgsParam> args_param;
  EXPECT_EQ(PreModelUtils::GetWorkspaceDataAddrOffset(runtime_param, op_desc, args_param, args_offset_values).size(), 0);
  v_memory_type.assign({RT_MEMORY_L1, RT_MEMORY_P2P_DDR, RT_MEMORY_HBM, RT_MEMORY_HBM, RT_MEMORY_HBM, kRtMemoryUB});
  AttrUtils::SetListInt(op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE, v_memory_type);
  AttrUtils::SetListInt(op_desc, ATTR_NAME_WORKSPACE_TYPE_LIST, v_memory_type);
  const vector<int32_t> workspace_no_reuse_scope{1, 1, 1, 1};
  AttrUtils::SetListInt(op_desc, ATTR_NAME_WORKSPACE_MEMORY_NO_REUSE_SCOPE, workspace_no_reuse_scope);
  uint8_t p2p_mem_base_addr = 0;
  { // RT_MEMORY_P2P_DDR
    PreMemInfo &mem_info = runtime_param.memory_infos[RT_MEMORY_P2P_DDR];
    mem_info.memory_size = 0x20002000u;
    mem_info.logic_memory_base = 0u;
    mem_info.memory_base = &p2p_mem_base_addr;
    mem_info.memory_type = RT_MEMORY_P2P_DDR;
  }

  uint8_t session_scope_mem_base_addr = 0;
  { // kSessionScopeMemory
    PreMemInfo &mem_info = runtime_param.memory_infos[0x100000000u | RT_MEMORY_HBM];
    mem_info.memory_size = 0x20002000u;
    mem_info.logic_memory_base = 0u;
    mem_info.memory_base = &session_scope_mem_base_addr;
  }
  auto workspace_addr = PreModelUtils::GetWorkspaceDataAddrOffset(runtime_param, op_desc, args_param, args_offset_values);
  EXPECT_EQ(workspace_addr.size(), 6);
}

TEST_F(UtestPreModelUtils, GetWeightSize) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr add_node = CreateNode(*graph, "add", ADD, 2, 1);
  auto op_desc = add_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);
  op_desc->SetType(CONSTANTOP);
  auto value_tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> float_value(4 * 8 * sizeof(int32_t));
  auto *int_value = reinterpret_cast<int32_t *>(float_value.data());
  for (int32_t i = 0; i < 32; ++i) {
    int_value[i] = i;
  }
  value_tensor->SetData(float_value);
  AttrUtils::SetTensor(op_desc, ATTR_NAME_WEIGHTS, value_tensor);

  auto weight_size = PreModelUtils::GetWeightSize(op_desc);
  EXPECT_EQ(weight_size.size(), 1);
}

TEST_F(UtestPreModelUtils, GetWorkspaceSize) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr add_node = CreateNode(*graph, "add", ADD, 2, 1);
  auto op_desc = add_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);
  op_desc->SetWorkspace({0x10001000, 0x10002000, 0, 0x10004000, 0x10008000});
  op_desc->SetWorkspaceBytes({0x1000, 0x2000, 0, 0x1000, 0x3000});

  auto workspace_size = PreModelUtils::GetWorkspaceSize(op_desc);
  EXPECT_EQ(workspace_size.size(), 5);

  op_desc->SetWorkspace({0x10001000, 0x10002000, 0, 0x10004000, 0x10008000});
  op_desc->SetWorkspaceBytes({0x1000, 0x2000, 0, 0x1000});
  workspace_size = PreModelUtils::GetWorkspaceSize(op_desc);
  EXPECT_EQ(workspace_size.size(), 0);
}

TEST_F(UtestPreModelUtils, ValidateMemRange) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr add_node = CreateNode(*graph, "add", ADD, 2, 1);
  auto op_desc = add_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);

  auto result = PreModelUtils::ValidateMemRange(op_desc, 1, std::numeric_limits<int64_t>::max(), 64);
  EXPECT_EQ(result, false);
}

TEST_F(UtestPreModelUtils, GetInputDataAddrOffset) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr add_node = CreateNode(*graph, "add", ADD, 2, 1);
  auto op_desc = add_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);

  vector<bool> is_input_const = {true, false};
  op_desc->SetIsInputConst(is_input_const);
  {
    auto tensor_desc = op_desc->MutableInputDesc(0);
    EXPECT_NE(tensor_desc, nullptr);
    TensorUtils::SetSize(*tensor_desc, 64);
    tensor_desc->SetShape(GeShape({1, 1}));
    tensor_desc->SetOriginShape(GeShape({1, 1}));
    TensorUtils::SetDataOffset(*tensor_desc, 0);
    AttrUtils::SetInt(tensor_desc, ATTR_NAME_TENSOR_MEM_TYPE, RT_MEMORY_P2P_DDR);
  }
  {
    auto tensor_desc = op_desc->MutableInputDesc(1);
    EXPECT_NE(tensor_desc, nullptr);
    TensorUtils::SetSize(*tensor_desc, 32);
    tensor_desc->SetShape(GeShape({1, 0}));
    tensor_desc->SetOriginShape(GeShape({1, 0}));
    TensorUtils::SetDataOffset(*tensor_desc, 64);
  }

  PreRuntimeParam runtime_param;
  uint8_t mem_base_addr = 0;
  runtime_param.mem_size = 0x20002000u;
  runtime_param.logic_mem_base = reinterpret_cast<uintptr_t>(&mem_base_addr);
  runtime_param.weight_size = 8;
  vector<KernelArgsParam> args_param;
  vector<uint64_t> args_offset_values;
  auto result = PreModelUtils::GetInputDataAddrOffset(runtime_param, op_desc, args_param, args_offset_values);
  //EXPECT_EQ(result, false);
}

TEST_F(UtestPreModelUtils, GetInputDataAddrOffset_tensor_mem_type) {
  PreRuntimeParam runtime_param;
  uint8_t mem_base_addr = 0;
  runtime_param.mem_size = 0x20002000u;
  runtime_param.weight_size = 0x10u;
  runtime_param.logic_mem_base = reinterpret_cast<uintptr_t>(&mem_base_addr);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr add_node = CreateNode(*graph, "add", ADD, 2, 1);
  auto op_desc = add_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);

  op_desc->SetIsInputConst({true});
  auto input_desc = op_desc->MutableInputDesc(0);
  int64_t input_size = 8;
  AttrUtils::SetInt(input_desc, ATTR_NAME_INPUT_ORIGIN_SIZE, input_size);
  input_desc->SetShape(GeShape());
  input_desc->SetDataType(DT_INT64);
  TensorUtils::SetSize(*input_desc, 64);

  vector<int64_t> v_memory_type{RT_MEMORY_L1};
  AttrUtils::SetListInt(op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE, v_memory_type);
  vector<uint64_t> args_offset_values;
  vector<KernelArgsParam> args_param;
  EXPECT_EQ(PreModelUtils::GetInputDataAddrOffset(runtime_param, op_desc, args_param, args_offset_values).size(), 2);

  AttrUtils::SetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, {RT_MEMORY_HBM, RT_MEMORY_L1});
  EXPECT_EQ(PreModelUtils::GetInputDataAddrOffset(runtime_param, op_desc, args_param, args_offset_values).size(), 2);

  auto tensor_desc = op_desc->MutableInputDesc(1);
  EXPECT_NE(tensor_desc, nullptr);
  TensorUtils::SetSize(*tensor_desc, 32);
  TensorUtils::SetDataOffset(*tensor_desc, 0x10004000);
  AttrUtils::SetInt(tensor_desc, ATTR_NAME_TENSOR_MEM_TYPE, RT_MEMORY_RESERVED);
  EXPECT_EQ(PreModelUtils::GetInputDataAddrOffset(runtime_param, op_desc, args_param, args_offset_values).size(), 0);
}

TEST_F(UtestPreModelUtils, GetOutputDataAddrOffset_tensor_mem_type) {
  PreRuntimeParam runtime_param;
  uint8_t mem_base_addr = 0;
  runtime_param.mem_size = 0x20002000u;
  runtime_param.weight_size = 0x10u;
  runtime_param.logic_mem_base = reinterpret_cast<uintptr_t>(&mem_base_addr);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr add_node = CreateNode(*graph, "add", ADD, 2, 1);
  auto op_desc = add_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);

  auto output_desc = op_desc->MutableOutputDesc(0);
  EXPECT_TRUE(ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, {RT_MEMORY_HBM, RT_MEMORY_L1}));

  vector<uint64_t> args_offset_values;
  vector<KernelArgsParam> args_param;
  EXPECT_EQ(PreModelUtils::GetOutputDataAddrOffset(runtime_param, op_desc, args_param, args_offset_values).size(), 0);

  EXPECT_TRUE(ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, {RT_MEMORY_TS}));
  auto tensor_desc = op_desc->MutableOutputDesc(0);
  EXPECT_NE(tensor_desc, nullptr);
  TensorUtils::SetSize(*tensor_desc, 32);
  TensorUtils::SetDataOffset(*tensor_desc, 0x10004000);
  AttrUtils::SetInt(tensor_desc, ATTR_NAME_TENSOR_MEM_TYPE, RT_MEMORY_TS);
  EXPECT_EQ(PreModelUtils::GetOutputDataAddrOffset(runtime_param, op_desc, args_param, args_offset_values).size(), 1);

  AttrUtils::SetInt(tensor_desc, ATTR_NAME_MEMORY_SIZE_CALC_TYPE,
                    static_cast<int32_t>(MemorySizeCalcType::ALWAYS_EMPTY));
  EXPECT_EQ(PreModelUtils::GetOutputDataAddrOffset(runtime_param, op_desc, args_param, args_offset_values).size(), 0);
}

TEST_F(UtestPreModelUtils, GetWeightSize_constant) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr add_node = CreateNode(*graph, "add", ADD, 2, 1);
  auto op_desc = add_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);

  op_desc->SetType(CONSTANT);
  const uint64_t const_value = 101;
  auto weight = std::make_shared<GeTensor>(op_desc->GetOutputDesc(0), (uint8_t *)&const_value, sizeof(uint64_t));
  AttrUtils::SetTensor(op_desc, ge::ATTR_NAME_WEIGHTS, weight);

  EXPECT_EQ(PreModelUtils::GetWeightSize(op_desc).size(), 1);
}
}  // namespace ge
