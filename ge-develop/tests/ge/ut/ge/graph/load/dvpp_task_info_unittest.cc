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
#include "graph/load/model_manager/task_info/dvpp/dvpp_task_info.h"
#include "graph/load/model_manager/davinci_model.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "compiler/engines/manager/opskernel_manager/ops_kernel_manager.h"

namespace ge {
class UtestDvppTask : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

  struct FakeOpsKernelInfoStore : OpsKernelInfoStore {
    FakeOpsKernelInfoStore() = default;

   private:
    Status Initialize(const std::map<std::string, std::string> &options) override {
      return SUCCESS;
    };
    Status Finalize() override {
      return SUCCESS;
    };
    bool CheckSupported(const OpDescPtr &op_desc, std::string &reason) const override {
      return true;
    };
    void GetAllOpsKernelInfo(std::map<std::string, ge::OpInfo> &infos) const override{};
  };
};

// test Init_DvppTaskInfo
TEST_F(UtestDvppTask, init_dvpp_task_info) {
  domi::TaskDef task_def;
  DvppTaskInfo task_info;
  EXPECT_EQ(task_info.Init(task_def, nullptr), PARAM_INVALID);

  DavinciModel model(0, nullptr);
  task_def.set_stream_id(0);
  EXPECT_EQ(task_info.Init(task_def, &model), FAILED);

  model.stream_list_.push_back((void *)0x12345);
  model.runtime_param_.mem_size = 10240;
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(new uint8_t[model.runtime_param_.mem_size]);
  domi::DvppTaskDef *dvpp_task = task_def.mutable_dvpp_task();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr add_node = CreateNode(*graph, "add", ADD, 3, 1);
  auto op_desc = add_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);

  const vector<int64_t> v_memory_type{RT_MEMORY_L1};
  AttrUtils::SetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_memory_type);
  op_desc->SetOutputOffset({2048});
  {  // RT_MEMORY_L1
    auto tensor_desc = op_desc->MutableOutputDesc(0);
    EXPECT_NE(tensor_desc, nullptr);
    TensorUtils::SetSize(*tensor_desc, 64);
    TensorUtils::SetDataOffset(*tensor_desc, 2048);
  }

  const vector<int64_t> v_memory_type1{RT_MEMORY_L1, RT_MEMORY_L1, RT_MEMORY_L1};
  AttrUtils::SetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, v_memory_type1);
  op_desc->SetInputOffset({2048, 2048, 2048});

  {  // RT_MEMORY_L1
    for (int i = 0; i < 3; ++i) {
      auto tensor_desc = op_desc->MutableOutputDesc(0);
      EXPECT_NE(tensor_desc, nullptr);
      TensorUtils::SetSize(*tensor_desc, 64);
      TensorUtils::SetDataOffset(*tensor_desc, 2048);
    }
  }

  const vector<int64_t> v_memory_type2{RT_MEMORY_HBM, RT_MEMORY_HBM};
  op_desc->SetWorkspace({1308, 1458});
  op_desc->SetWorkspaceBytes({150, 150});
  AttrUtils::SetListInt(op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE, v_memory_type2);
  AttrUtils::SetListInt(op_desc, ATTR_NAME_WORKSPACE_TYPE_LIST, v_memory_type2);

  OpsKernelInfoStorePtr ops_kernel_info_ptr = MakeShared<FakeOpsKernelInfoStore>();
  op_desc->SetExtAttr<OpsKernelInfoStore *> ("OpsKernelInfoStorePtr", ops_kernel_info_ptr.get());
  model.op_list_[0] = op_desc;

  dvpp_task->set_op_index(0);

  EXPECT_EQ(task_info.Init(task_def, &model), SUCCESS);

  delete [] reinterpret_cast<uint8_t *>(model.runtime_param_.mem_base);
  model.runtime_param_.mem_base = 0U;
  model.stream_list_.clear();
}

TEST_F(UtestDvppTask, testDistribute) {
  DvppTaskInfo task_info;

  domi::TaskDef task_def;
  task_def.set_stream_id(0);

  DavinciModel model(0, nullptr);
  model.stream_list_.push_back((void *)0x12345);
  task_info.Init(task_def, &model);

  domi::GetContext().is_online_model = true;
  auto ret = task_info.Distribute();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_TRUE(task_info.IsSupportReDistribute());
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
  model.stream_list_.clear();
  domi::GetContext().is_online_model = false;
}
}  // namespace ge
