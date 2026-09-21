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
#include "graph/load/model_manager/model_utils.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "framework/common/types.h"
#include "graph/manager/mem_manager.h"
#include "runtime_stub.h"

using namespace std;

namespace ge {
class UtestModelUtils : public testing::Test {
 protected:
  void SetUp() {
    RTS_STUB_SETUP();
  }
  void TearDown() {
    RTS_STUB_TEARDOWN();
  }
};

// test ModelUtils::GetVarAddr

TEST_F(UtestModelUtils, get_var_addr_hbm) {
  uint8_t test = 2;
  uint8_t *pf = &test;
  RuntimeParam runtime_param;
  runtime_param.session_id = 0;
  runtime_param.logic_var_base = 0;
  runtime_param.var_base = reinterpret_cast<uintptr_t>(pf);
  runtime_param.var_size = 16;

  int64_t offset = 8;
  EXPECT_EQ(VarManager::Instance(runtime_param.session_id)->Init(0, 0, 0, 0), SUCCESS);
  EXPECT_NE(VarManager::Instance(runtime_param.session_id)->var_resource_, nullptr);
  VarManager::Instance(runtime_param.session_id)->var_resource_->var_offset_map_[offset] = RT_MEMORY_HBM;
  void *var_addr = nullptr;
  EXPECT_NE(ModelUtils::GetVarAddr(runtime_param, offset, var_addr), SUCCESS);
  EXPECT_NE(ModelUtils::GetRtAddress(runtime_param, 0, pf), SUCCESS);
  VarManager::Instance(runtime_param.session_id)->FreeVarMemory();
  VarManager::Instance(runtime_param.session_id)->Destory();
}

TEST_F(UtestModelUtils, get_rt_address_and_mem_type) {
  uint8_t test = 2;
  uint8_t *pf = &test;

  std::shared_ptr<ge::GeModel> ge_model = std::make_shared<ge::GeModel>();
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 0x10000);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, 0x5000);
  (void)AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0x0);
  (void)AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0x15000);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_TASK_GEN_VAR_ADDR, 0x20000);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0x800000000);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0);
  (void)AttrUtils::SetInt(ge_model, MODEL_ATTR_HOST_MEMORY_SIZE, 0x10000);
  (void)AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_HOST_BASE_ADDR, 0x900000000);
  (void)AttrUtils::SetInt(ge_model, MODEL_ATTR_HOST_SVM_SIZE, 0x10000);
  (void)AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_HOST_SVM_BASE_ADDR, 0x900010000);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_P2P_MEMORY_SIZE, 0x5000);

  (void)AttrUtils::SetInt(ge_model, MODEL_ATTR_SESSION_ID, 0);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 0);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_NOTIFY_NUM, 0);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 0);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_BATCH_NUM, 0);

  RuntimeParam runtime_param;
  EXPECT_EQ(ModelUtils::InitRuntimeParams(ge_model, runtime_param, 0U), SUCCESS);
  runtime_param.var_base = reinterpret_cast<uintptr_t>(pf);
  runtime_param.fileconstant_addr_mapping[0x20002] = 0x20002;

  EXPECT_EQ(VarManager::Instance(runtime_param.session_id)->Init(0, 0, 0, 0), SUCCESS);

  uint64_t mem_type = 0ULL;
  EXPECT_EQ(ModelUtils::GetRtAddress(runtime_param, 0x20002, pf, mem_type), SUCCESS);
  EXPECT_EQ(mem_type, kConstantMemType);
  EXPECT_EQ(ModelUtils::GetRtAddress(runtime_param, 0x900000001, pf, mem_type), SUCCESS);
  EXPECT_EQ(mem_type, RT_MEMORY_HOST);
  EXPECT_EQ(ModelUtils::GetRtAddress(runtime_param, 0x900010001, pf, mem_type), SUCCESS);
  EXPECT_EQ(mem_type, RT_MEMORY_HOST_SVM);
  EXPECT_EQ(ModelUtils::GetRtAddress(runtime_param, 0x14000, pf, mem_type), SUCCESS);
  EXPECT_EQ(mem_type, RT_MEMORY_P2P_DDR);
  VarManager::Instance(runtime_param.session_id)->FreeVarMemory();
  VarManager::Instance(runtime_param.session_id)->Destory();
}

TEST_F(UtestModelUtils, get_var_addr_rdma_hbm) {
  uint8_t test = 2;
  uint8_t *pf = &test;
  RuntimeParam runtime_param;
  runtime_param.session_id = 0;
  runtime_param.logic_var_base = 0;
  runtime_param.var_base = reinterpret_cast<uintptr_t>(pf);

  int64_t offset = 8;
  EXPECT_EQ(VarManager::Instance(runtime_param.session_id)->Init(0, 0, 0, 0), SUCCESS);

  EXPECT_NE(VarManager::Instance(runtime_param.session_id)->var_resource_, nullptr);
  VarManager::Instance(runtime_param.session_id)->var_resource_->var_offset_map_[offset] = RT_MEMORY_RDMA_HBM;

  void *var_addr = nullptr;
  EXPECT_EQ(ModelUtils::GetVarAddr(runtime_param, offset, var_addr), SUCCESS);
  EXPECT_EQ(reinterpret_cast<uint8_t *>(offset), var_addr);
}

TEST_F(UtestModelUtils, get_var_addr_rdma_hbm_negative_offset) {
  uint8_t test = 2;
  uint8_t *pf = &test;
  RuntimeParam runtime_param;
  runtime_param.session_id = 0;
  runtime_param.logic_var_base = 0;
  runtime_param.var_base = reinterpret_cast<uintptr_t>(pf);

  int64_t offset = -1;
  EXPECT_EQ(VarManager::Instance(runtime_param.session_id)->Init(0, 0, 0, 0), SUCCESS);

  EXPECT_NE(VarManager::Instance(runtime_param.session_id)->var_resource_, nullptr);
  VarManager::Instance(runtime_param.session_id)->var_resource_->var_offset_map_[offset] = RT_MEMORY_RDMA_HBM;
  void *var_addr = nullptr;
  EXPECT_NE(ModelUtils::GetVarAddr(runtime_param, offset, var_addr), SUCCESS);
  VarManager::Instance(runtime_param.session_id)->Destory();
}

TEST_F(UtestModelUtils, get_var_addr_with_invalid_mem_type) {
  uint8_t test = 2;
  uint8_t *pf = &test;
  RuntimeParam runtime_param;
  runtime_param.session_id = 0;
  runtime_param.logic_var_base = 0;
  runtime_param.var_base = reinterpret_cast<uintptr_t>(pf);

  int64_t offset = -1;
  EXPECT_EQ(VarManager::Instance(runtime_param.session_id)->Init(0, 0, 0, 0), SUCCESS);

  EXPECT_NE(VarManager::Instance(runtime_param.session_id)->var_resource_, nullptr);
  VarManager::Instance(runtime_param.session_id)->var_resource_->var_offset_map_[offset] = INT32_MAX;
  void *var_addr = nullptr;
  EXPECT_NE(ModelUtils::GetVarAddr(runtime_param, offset, var_addr), SUCCESS);
  VarManager::Instance(runtime_param.session_id)->Destory();
}

TEST_F(UtestModelUtils, test_GetInputDataAddrs_input_const) {
  RuntimeParam runtime_param;
  uint8_t weight_base_addr = 0;
  runtime_param.session_id = 0;
  runtime_param.weight_base = reinterpret_cast<uintptr_t>(&weight_base_addr);
  runtime_param.weight_size = 64;

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr add_node = CreateNode(*graph, "add", ADD, 2, 1);
  auto op_desc = add_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);

  vector<bool> is_input_const = {true, true};
  op_desc->SetIsInputConst(is_input_const);
  {
    auto tensor_desc = op_desc->MutableInputDesc(0);
    EXPECT_NE(tensor_desc, nullptr);
    TensorUtils::SetSize(*tensor_desc, 64);
    tensor_desc->SetShape(GeShape({1, 1}));
    tensor_desc->SetOriginShape(GeShape({1, 1}));
    TensorUtils::SetDataOffset(*tensor_desc, 0);
  }
  {
    auto tensor_desc = op_desc->MutableInputDesc(1);
    EXPECT_NE(tensor_desc, nullptr);
    TensorUtils::SetSize(*tensor_desc, 32);
    tensor_desc->SetShape(GeShape({1, 0}));
    tensor_desc->SetOriginShape(GeShape({1, 0}));
    TensorUtils::SetDataOffset(*tensor_desc, 64);
  }
  vector<void *> input_data_addr = ModelUtils::GetInputAddrs(runtime_param, op_desc);
  EXPECT_EQ(input_data_addr.size(), 2);
  EXPECT_EQ(input_data_addr.at(0), static_cast<void *>(&weight_base_addr + 0));
  EXPECT_EQ(input_data_addr.at(1), static_cast<void *>(&weight_base_addr + 64));
}

TEST_F(UtestModelUtils, GetInputDataAddrs_tensor_mem_type) {
  RuntimeParam runtime_param;
  runtime_param.session_id = 0;
  runtime_param.mem_size = 0x20002000u;

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr add_node = CreateNode(*graph, "add", ADD, 3, 1);
  auto op_desc = add_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);

  const vector<int64_t> v_memory_type{RT_MEMORY_L1, RT_MEMORY_TS, RT_MEMORY_P2P_DDR};
  AttrUtils::SetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, v_memory_type);
  op_desc->SetInputOffset({0x10001000, 0x10002000, 0x10004000});

  { // RT_MEMORY_L1
    auto tensor_desc = op_desc->MutableInputDesc(0);
    EXPECT_NE(tensor_desc, nullptr);
    TensorUtils::SetSize(*tensor_desc, 64);
    TensorUtils::SetDataOffset(*tensor_desc, 0x10001000);
  }

  uint8_t ts_mem_base_addr = 0;
  { // RT_MEMORY_TS
    auto tensor_desc = op_desc->MutableInputDesc(1);
    EXPECT_NE(tensor_desc, nullptr);
    TensorUtils::SetSize(*tensor_desc, 64);
    TensorUtils::SetDataOffset(*tensor_desc, 0x10002000);

    TsMemMall &ts_mem_mall = *runtime_param.ts_mem_mall;
    ts_mem_mall.mem_store_size_[0x10002000] = &ts_mem_base_addr;   // offset is 0
    ts_mem_mall.mem_store_addr_[&ts_mem_base_addr] = 0x10002000;
  }

  uint8_t p2p_mem_base_addr = 0;
  { // RT_MEMORY_P2P_DDR
    auto tensor_desc = op_desc->MutableInputDesc(2);
    EXPECT_NE(tensor_desc, nullptr);
    TensorUtils::SetSize(*tensor_desc, 32);
    TensorUtils::SetDataOffset(*tensor_desc, 0x10004000);
    AttrUtils::SetInt(tensor_desc, ATTR_NAME_TENSOR_MEM_TYPE, RT_MEMORY_P2P_DDR);

    MemInfo &mem_info = runtime_param.memory_infos[RT_MEMORY_P2P_DDR];
    mem_info.memory_size = 0x20002000u;
    mem_info.logic_memory_base = 0u;
    mem_info.memory_base = &p2p_mem_base_addr;
    mem_info.memory_type = RT_MEMORY_P2P_DDR;
  }

  const vector<void *> input_data_addr = ModelUtils::GetInputDataAddrs(runtime_param, op_desc);
  EXPECT_EQ(input_data_addr.size(), 3);
  EXPECT_EQ(input_data_addr.at(0), reinterpret_cast<void *>(0x10001000));
  EXPECT_EQ(input_data_addr.at(1), static_cast<void *>(&ts_mem_base_addr));
  EXPECT_EQ(input_data_addr.at(2), static_cast<void *>(&p2p_mem_base_addr + 0x10004000));

  runtime_param.ts_mem_mall->mem_store_size_.clear();
  runtime_param.ts_mem_mall->mem_store_addr_.clear();
}

TEST_F(UtestModelUtils, GetInputDataAddrs_tensor_mem_type_with_variable) {
  const std::vector<rtMemType_t> memory_types({RT_MEMORY_HBM, RT_MEMORY_P2P_DDR});
  RuntimeParam runtime_param;
  runtime_param.session_id = 100;
  runtime_param.mem_size = 0x20002000u;
  runtime_param.var_size = 1024;

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr add_node = CreateNode(*graph, "add", ADD, 2, 1);
  auto op_desc = add_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);
  EXPECT_EQ(VarManager::Instance(runtime_param.session_id)->Init(0, runtime_param.session_id, 0, 0), SUCCESS);
  EXPECT_EQ(MemManager::Instance().Initialize(memory_types), SUCCESS);
  VarManager::Instance(runtime_param.session_id)->SetMemManager(&MemManager::Instance());

  GeShape shape({16, 16});
  GeTensorDesc tensor_desc(shape, FORMAT_ND, DT_FLOAT16);
  TensorUtils::SetSize(tensor_desc, 256);
  op_desc->UpdateInputDesc(0, tensor_desc);
  op_desc->UpdateInputDesc(1, tensor_desc);
  op_desc->UpdateOutputDesc(0, tensor_desc);
  op_desc->SetInputOffset({0x800000000, 0x800000600});
  op_desc->SetOutputOffset({0x800000c00});
  VarManager::Instance(runtime_param.session_id)->AssignVarMem("add_in_0", op_desc, op_desc->GetInputDesc(0), RT_MEMORY_HBM);
  VarManager::Instance(runtime_param.session_id)->AssignVarMem("add_in_1", op_desc, op_desc->GetInputDesc(1), RT_MEMORY_HBM);
  VarManager::Instance(runtime_param.session_id)->AssignVarMem("add_out_0", op_desc, op_desc->GetOutputDesc(0), RT_MEMORY_HBM);
  const vector<void *> input_data_addr = ModelUtils::GetInputDataAddrs(runtime_param, op_desc);
  EXPECT_EQ(input_data_addr.size(), 2);

  const vector<void *> output_data_addr = ModelUtils::GetOutputDataAddrs(runtime_param, op_desc);
  EXPECT_EQ(output_data_addr.size(), 1);
  VarManager::Instance(runtime_param.session_id)->FreeVarMemory();
  VarManager::Instance(runtime_param.session_id)->Destory();
  runtime_param.ts_mem_mall->mem_store_size_.clear();
  runtime_param.ts_mem_mall->mem_store_addr_.clear();
}

TEST_F(UtestModelUtils, GetOutputDataAddrs_tensor_mem_type) {
  RuntimeParam runtime_param;
  runtime_param.session_id = 0;
  runtime_param.mem_size = 0x20002000u;

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr add_node = CreateNode(*graph, "add", ADD, 1, 5);
  auto op_desc = add_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);

  const vector<int64_t> v_memory_type{RT_MEMORY_L1, RT_MEMORY_TS, RT_MEMORY_P2P_DDR, RT_MEMORY_HBM, RT_MEMORY_L2};
  AttrUtils::SetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_memory_type);
  op_desc->SetOutputOffset({0x10001000, 0x10002000, 0x10004000, 0x10008000, 0x1000c000});

  {  // RT_MEMORY_L1
    auto tensor_desc = op_desc->MutableOutputDesc(0);
    EXPECT_NE(tensor_desc, nullptr);
    TensorUtils::SetSize(*tensor_desc, 64);
    TensorUtils::SetDataOffset(*tensor_desc, 0x10001000);
  }

  uint8_t ts_mem_base_addr = 0;
  {  // RT_MEMORY_TS
    auto tensor_desc = op_desc->MutableOutputDesc(1);
    EXPECT_NE(tensor_desc, nullptr);
    TensorUtils::SetSize(*tensor_desc, 64);
    TensorUtils::SetDataOffset(*tensor_desc, 0x10002000);

    TsMemMall &ts_mem_mall = *runtime_param.ts_mem_mall;
    ts_mem_mall.mem_store_size_[0x10002000] = &ts_mem_base_addr;  // offset is 0
    ts_mem_mall.mem_store_addr_[&ts_mem_base_addr] = 0x10002000;
  }

  uint8_t p2p_mem_base_addr = 0;
  {  // RT_MEMORY_P2P_DDR
    auto tensor_desc = op_desc->MutableOutputDesc(2);
    EXPECT_NE(tensor_desc, nullptr);
    TensorUtils::SetSize(*tensor_desc, 32);
    TensorUtils::SetDataOffset(*tensor_desc, 0x10004000);
    AttrUtils::SetInt(tensor_desc, ATTR_NAME_TENSOR_MEM_TYPE, RT_MEMORY_P2P_DDR);

    MemInfo &mem_info = runtime_param.memory_infos[RT_MEMORY_P2P_DDR];
    mem_info.memory_size = 0x20002000u;
    mem_info.logic_memory_base = 0u;
    mem_info.memory_base = &p2p_mem_base_addr;
    mem_info.memory_type = RT_MEMORY_P2P_DDR;
  }
  uint8_t hbm_mem_base_addr = 0;
  {  // RT_MEMORY_HBM_DDR
    auto tensor_desc = op_desc->MutableOutputDesc(3);
    EXPECT_NE(tensor_desc, nullptr);
    TensorUtils::SetSize(*tensor_desc, 32);
    runtime_param.mem_base = reinterpret_cast<uintptr_t>(&hbm_mem_base_addr);
  }

  {  // RT_MEMORY_L2
    auto tensor_desc = op_desc->MutableOutputDesc(4);
    EXPECT_NE(tensor_desc, nullptr);
    TensorUtils::SetSize(*tensor_desc, 32);
    runtime_param.mem_base = reinterpret_cast<uintptr_t>(&hbm_mem_base_addr);
  }
  const vector<void *> output_data_addr = ModelUtils::GetOutputDataAddrs(runtime_param, op_desc);
  EXPECT_EQ(output_data_addr.size(), 5);
  EXPECT_EQ(output_data_addr.at(0), reinterpret_cast<void *>(0x10001000));
  EXPECT_EQ(output_data_addr.at(1), static_cast<void *>(&ts_mem_base_addr));
  EXPECT_EQ(output_data_addr.at(2), static_cast<void *>(&p2p_mem_base_addr + 0x10004000));
  EXPECT_EQ(output_data_addr.at(3), static_cast<void *>(&hbm_mem_base_addr + 0x10008000));
  EXPECT_EQ(output_data_addr.at(4), static_cast<void *>(&hbm_mem_base_addr + 0x1000c000));

  runtime_param.ts_mem_mall->mem_store_size_.clear();
  runtime_param.ts_mem_mall->mem_store_addr_.clear();
}

TEST_F(UtestModelUtils, GetOutputDataAddrs_with_optional_output) {
  RuntimeParam runtime_param;
  runtime_param.session_id = 0;
  runtime_param.mem_size = 0x20002000u;

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr add_node = CreateNode(*graph, "add", ADD, 1, 5);
  auto op_desc = add_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);

  const vector<int64_t> v_memory_type{RT_MEMORY_L1, RT_MEMORY_TS, RT_MEMORY_P2P_DDR, RT_MEMORY_HBM, RT_MEMORY_L2};
  AttrUtils::SetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_memory_type);
  op_desc->SetOutputOffset({0x10001000, 0x10002000, 0x10004000, 0x10008000, 0x1000c000});

  {  // RT_MEMORY_L1
    auto tensor_desc = op_desc->MutableOutputDesc(0);
    EXPECT_NE(tensor_desc, nullptr);
    TensorUtils::SetSize(*tensor_desc, 64);
    TensorUtils::SetDataOffset(*tensor_desc, 0x10001000);
  }

  uint8_t ts_mem_base_addr = 0;
  {  // RT_MEMORY_TS
    auto tensor_desc = op_desc->MutableOutputDesc(1);
    EXPECT_NE(tensor_desc, nullptr);
    TensorUtils::SetSize(*tensor_desc, 64);
    TensorUtils::SetDataOffset(*tensor_desc, 0x10002000);

    TsMemMall &ts_mem_mall = *runtime_param.ts_mem_mall;
    ts_mem_mall.mem_store_size_[0x10002000] = &ts_mem_base_addr;  // offset is 0
    ts_mem_mall.mem_store_addr_[&ts_mem_base_addr] = 0x10002000;
  }

  uint8_t p2p_mem_base_addr = 0;
  {  // RT_MEMORY_P2P_DDR
    auto tensor_desc = op_desc->MutableOutputDesc(2);
    EXPECT_NE(tensor_desc, nullptr);
    TensorUtils::SetSize(*tensor_desc, 32);
    TensorUtils::SetDataOffset(*tensor_desc, 0x10004000);
    AttrUtils::SetInt(tensor_desc, ATTR_NAME_TENSOR_MEM_TYPE, RT_MEMORY_P2P_DDR);

    MemInfo &mem_info = runtime_param.memory_infos[RT_MEMORY_P2P_DDR];
    mem_info.memory_size = 0x20002000u;
    mem_info.logic_memory_base = 0u;
    mem_info.memory_base = &p2p_mem_base_addr;
    mem_info.memory_type = RT_MEMORY_P2P_DDR;
  }
  uint8_t hbm_mem_base_addr = 0;
  {  // RT_MEMORY_HBM_DDR
    auto tensor_desc = op_desc->MutableOutputDesc(3);
    EXPECT_NE(tensor_desc, nullptr);
    AttrUtils::SetInt(tensor_desc, ATTR_NAME_MEMORY_SIZE_CALC_TYPE,
                      static_cast<int32_t>(MemorySizeCalcType::ALWAYS_EMPTY));
  }

  {  // RT_MEMORY_L2
    auto tensor_desc = op_desc->MutableOutputDesc(4);
    EXPECT_NE(tensor_desc, nullptr);
    TensorUtils::SetSize(*tensor_desc, 32);
    runtime_param.mem_base = reinterpret_cast<uintptr_t>(&hbm_mem_base_addr);
  }

  std::vector<uint64_t> mem_type;
  vector<uint64_t > output_data_addr = ModelUtils::GetOutputAddrsValue(runtime_param, op_desc, mem_type, true);
  EXPECT_EQ(output_data_addr.size(), 5);
  EXPECT_EQ(output_data_addr.at(0), 0x10001000);
  EXPECT_EQ(output_data_addr.at(1), PtrToValue(&ts_mem_base_addr));
  EXPECT_EQ(output_data_addr.at(2), PtrToValue(&p2p_mem_base_addr) + 0x10004000);
  EXPECT_EQ(output_data_addr.at(3), 0);
  EXPECT_EQ(output_data_addr.at(4), PtrToValue(&hbm_mem_base_addr) + 0x1000c000);

  EXPECT_EQ(mem_type.size(), 5);
  EXPECT_EQ(mem_type.at(3), kFixMemType);

  mem_type.clear();
  output_data_addr.clear();
  output_data_addr = ModelUtils::GetOutputAddrsValue(runtime_param, op_desc, mem_type, false);
  EXPECT_EQ(output_data_addr.size(), 4);
  EXPECT_EQ(output_data_addr.at(0), 0x10001000);
  EXPECT_EQ(output_data_addr.at(1), PtrToValue(&ts_mem_base_addr));
  EXPECT_EQ(output_data_addr.at(2), PtrToValue(&p2p_mem_base_addr) + 0x10004000);
  EXPECT_EQ(output_data_addr.at(3), PtrToValue(&hbm_mem_base_addr) + 0x1000c000);
  EXPECT_EQ(mem_type.size(), 4);

  mem_type.clear();
  output_data_addr.clear();
  output_data_addr = ModelUtils::GetOutputAddrsValue(runtime_param, op_desc, mem_type);
  EXPECT_EQ(output_data_addr.size(), 4);
  EXPECT_EQ(output_data_addr.at(0), 0x10001000);
  EXPECT_EQ(output_data_addr.at(1), PtrToValue(&ts_mem_base_addr));
  EXPECT_EQ(output_data_addr.at(2), PtrToValue(&p2p_mem_base_addr) + 0x10004000);
  EXPECT_EQ(output_data_addr.at(3), PtrToValue(&hbm_mem_base_addr) + 0x1000c000);
  EXPECT_EQ(mem_type.size(), 4);

  runtime_param.ts_mem_mall->mem_store_size_.clear();
  runtime_param.ts_mem_mall->mem_store_addr_.clear();
}

TEST_F(UtestModelUtils, GetWorkspaceDataAddrs_tensor_mem_type) {
  RuntimeParam runtime_param;
  uint8_t mem_base_addr = 0;
  runtime_param.session_id = 0;
  runtime_param.mem_size = 0x20002000u;
  runtime_param.mem_base = reinterpret_cast<uintptr_t>(&mem_base_addr);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr add_node = CreateNode(*graph, "add", ADD, 2, 1);
  auto op_desc = add_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);

  op_desc->SetWorkspace({0x10001000, 0x10002000, 0, 0x10004000, 0x10008000, 0x1000c000});
  op_desc->SetWorkspaceBytes({0x1000, 0x2000, 0, 0x1000, 0x3000, 0x5000});

  vector<int64_t> v_memory_type{RT_MEMORY_L1};
  AttrUtils::SetListInt(op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE, v_memory_type);
  // type number != workspace number.
  EXPECT_EQ(ModelUtils::GetWorkspaceDataAddrs(runtime_param, op_desc).size(), 0);

  v_memory_type.assign({RT_MEMORY_L1, RT_MEMORY_P2P_DDR, RT_MEMORY_HBM, RT_MEMORY_HBM, RT_MEMORY_HBM, kRtMemoryUB});
  AttrUtils::SetListInt(op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE, v_memory_type);
  AttrUtils::SetListInt(op_desc, ATTR_NAME_WORKSPACE_TYPE_LIST, v_memory_type);

  constexpr int32_t kSessionNoReuse = 1;
  const vector<int32_t> workspace_no_reuse_scope{kSessionNoReuse, kSessionNoReuse, kSessionNoReuse, kSessionNoReuse};
  AttrUtils::SetListInt(op_desc, ATTR_NAME_WORKSPACE_MEMORY_NO_REUSE_SCOPE, workspace_no_reuse_scope);
  uint8_t p2p_mem_base_addr = 0;
  { // RT_MEMORY_P2P_DDR
    MemInfo &mem_info = runtime_param.memory_infos[RT_MEMORY_P2P_DDR];
    mem_info.memory_size = 0x20002000u;
    mem_info.logic_memory_base = 0u;
    mem_info.memory_base = &p2p_mem_base_addr;
    mem_info.memory_type = RT_MEMORY_P2P_DDR;
  }

  uint8_t session_scope_mem_base_addr = 0;
  { // kSessionScopeMemory
    MemInfo &mem_info = runtime_param.memory_infos[0x100000000u | RT_MEMORY_HBM];
    mem_info.memory_size = 0x20002000u;
    mem_info.logic_memory_base = 0u;
    mem_info.memory_base = &session_scope_mem_base_addr;
  }
  const vector<void *> workspace_addr = ModelUtils::GetWorkspaceDataAddrs(runtime_param, op_desc);
  EXPECT_EQ(workspace_addr.size(), 6);
  EXPECT_EQ(workspace_addr.at(0), reinterpret_cast<void *>(0x10001000));
  EXPECT_EQ(workspace_addr.at(1), static_cast<void *>(&p2p_mem_base_addr + 0x10002000));
  EXPECT_EQ(workspace_addr.at(2), nullptr);
  EXPECT_EQ(workspace_addr.at(3), static_cast<void *>(&session_scope_mem_base_addr + 0x10004000));
  EXPECT_EQ(workspace_addr.at(4), static_cast<void *>(&mem_base_addr + 0x10008000));
}

TEST_F(UtestModelUtils, test_get_input_descs_with_large_tensor_dim) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_input");
  NodePtr add_node = CreateNode(*graph, "add", ADD, 2, 2);
  auto op_desc = add_node->GetOpDesc();
  GeShape InputShape;
  InputShape.SetDimNum(1);
  InputShape.SetDim(0, 1L + INT32_MAX);
  op_desc->MutableInputDesc(0)->SetShape(InputShape);
  EXPECT_NE(op_desc, nullptr);

  const auto input_descs = ModelUtils::GetInputDescs(op_desc);
  EXPECT_EQ(input_descs.size(), 2);

  const auto output_descs = ModelUtils::GetOutputDescs(op_desc);
  EXPECT_EQ(input_descs.size(), 2);
}

TEST_F(UtestModelUtils, test_get_output_descs_with_large_tensor_dim) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_output");
  NodePtr add_node = CreateNode(*graph, "add", ADD, 2, 2);
  auto op_desc = add_node->GetOpDesc();
  GeShape OutputShape;
  OutputShape.SetDimNum(1);
  OutputShape.SetDim(0, 1L + INT32_MAX);
  op_desc->MutableOutputDesc(0)->SetShape(OutputShape);
  EXPECT_NE(op_desc, nullptr);

  const auto input_descs = ModelUtils::GetInputDescs(op_desc);
  EXPECT_EQ(input_descs.size(), 2);

  const auto output_descs = ModelUtils::GetOutputDescs(op_desc);
  EXPECT_EQ(input_descs.size(), 2);
}

TEST_F(UtestModelUtils, test_refresh_address_by_mem_type) {
  RuntimeParam runtime_param;
  uint64_t mem_type = 0;
  ConstOpDescPtr op_desc = nullptr;
  size_t index = 0;
  std::string io_type = "";
  int64_t size = 0;
  int64_t logical_offset = 0;
  ModelUtils::NodeMemInfo node_mem_info = {mem_type, op_desc, index, io_type, size, logical_offset};
  node_mem_info.mem_type_ = UINT64_MAX;
  void *mem_addr = nullptr;
  Status ret = ModelUtils::RefreshAddressByMemType(runtime_param, node_mem_info, mem_addr);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestModelUtils, test_get_weight_size) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr add_node = CreateNode(*graph, "add", ADD, 2, 2);
  auto op_desc = add_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);

  const auto weight_size = ModelUtils::GetWeightSize(op_desc);
  EXPECT_EQ(weight_size.size(), 0);

  const auto weight_tensor = ModelUtils::GetWeights(op_desc);
  EXPECT_EQ(weight_tensor.size(), 0);

  const auto input_descs = ModelUtils::GetInputDescs(op_desc);
  EXPECT_EQ(input_descs.size(), 2);

  const auto output_descs = ModelUtils::GetOutputDescs(op_desc);
  EXPECT_EQ(input_descs.size(), 2);
}

TEST_F(UtestModelUtils, test_get_string_output_size) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr add_node = CreateNode(*graph, "add", ADD, 2, 1);
  auto op_desc = add_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);

  auto tensor_desc = op_desc->MutableOutputDesc(0);
  EXPECT_NE(tensor_desc, nullptr);
  tensor_desc->SetDataType(DT_STRING);

  int64_t max_size = 100000;
  AttrUtils::SetInt(op_desc, "_op_max_size", max_size);

  std::vector<int64_t> v_max_size({max_size});
  EXPECT_EQ(ModelUtils::GetOutputSize(op_desc), v_max_size);
}

TEST_F(UtestModelUtils, IsGeUseExtendSizeStaticMemory_Set1_True) {
  // set and check option
  std::map<std::string, std::string> options_map;
  options_map[STATIC_MEMORY_POLICY] = "1";
  GetThreadLocalContext().SetGraphOption(options_map);
  EXPECT_TRUE(ModelUtils::IsGeUseExtendSizeMemory());
  EXPECT_TRUE(!ModelUtils::IsGeUseExtendSizeMemory(true));
}

TEST_F(UtestModelUtils, IsGeUseExtendSizeStaticMemory_Set2_True) {
  // set and check option
  std::map<std::string, std::string> options_map;
  options_map[STATIC_MEMORY_POLICY] = "2";
  GetThreadLocalContext().SetGraphOption(options_map);
  EXPECT_TRUE(ModelUtils::IsGeUseExtendSizeMemory());
  EXPECT_TRUE(!ModelUtils::IsGeUseExtendSizeMemory(true));
}

TEST_F(UtestModelUtils, IsGeUseExtendSizeMemory_Set1_True) {
  // set and check option
  std::map<std::string, std::string> options_map;
  options_map[STATIC_MEMORY_POLICY] = "3";
  GetThreadLocalContext().SetGraphOption(options_map);
  EXPECT_TRUE(!ModelUtils::IsGeUseExtendSizeMemory());
  EXPECT_TRUE(ModelUtils::IsGeUseExtendSizeMemory(true));
}

TEST_F(UtestModelUtils, IsGeUseExtendSizeMemory_Set2_True) {
  // set and check option
  std::map<std::string, std::string> options_map;
  options_map[STATIC_MEMORY_POLICY] = "4";
  GetThreadLocalContext().SetGraphOption(options_map);
  EXPECT_TRUE(ModelUtils::IsGeUseExtendSizeMemory());
  EXPECT_TRUE(ModelUtils::IsGeUseExtendSizeMemory(true));
}

TEST_F(UtestModelUtils, IsStaticMemory_Set1_False) {
  // set and check option
  std::map<std::string, std::string> options_map;
  options_map[STATIC_MEMORY_POLICY] = "0";
  GetThreadLocalContext().SetGraphOption(options_map);
  EXPECT_TRUE(!ModelUtils::IsGeUseExtendSizeMemory());
  EXPECT_TRUE(!ModelUtils::IsGeUseExtendSizeMemory(true));
}

TEST_F(UtestModelUtils, MallocFreeP2pMem_Success_WhenUserSetFixedP2pMem) {
  RuntimeParam runtime_param;
  runtime_param.p2p_fixed_mem_base = 0x123;
  runtime_param.p2p_fixed_mem_size = 1024;
  MemInfo mem_info;
  mem_info.memory_type = RT_MEMORY_P2P_DDR;
  mem_info.memory_size = 1024U;
  runtime_param.memory_infos.insert({RT_MEMORY_P2P_DDR, mem_info});

  ASSERT_EQ(ModelUtils::MallocExMem(0U, runtime_param), SUCCESS);
  EXPECT_EQ(PtrToValue(runtime_param.memory_infos[RT_MEMORY_P2P_DDR].memory_base), 0x123);

  ModelUtils::FreeExMem(0U, runtime_param, 0);
  EXPECT_EQ(PtrToValue(runtime_param.memory_infos[RT_MEMORY_P2P_DDR].memory_base), 0x123);
}

TEST_F(UtestModelUtils, MallocFreeP2pMem_Success_WhenUserNotSetFixedP2pMem) {
  RuntimeParam runtime_param;
  runtime_param.p2p_fixed_mem_base = 0U;
  runtime_param.p2p_fixed_mem_size = 1024;
  MemInfo mem_info;
  mem_info.memory_type = RT_MEMORY_P2P_DDR;
  mem_info.memory_size = 1024U;
  runtime_param.memory_infos.insert({RT_MEMORY_P2P_DDR, mem_info});

  ASSERT_EQ(ModelUtils::MallocExMem(0U, runtime_param), SUCCESS);
  EXPECT_NE(runtime_param.memory_infos[RT_MEMORY_P2P_DDR].memory_base, nullptr);

  ModelUtils::FreeExMem(0U, runtime_param, 0);
  EXPECT_EQ(runtime_param.memory_infos[RT_MEMORY_P2P_DDR].memory_base, nullptr);
}
}  // namespace ge
