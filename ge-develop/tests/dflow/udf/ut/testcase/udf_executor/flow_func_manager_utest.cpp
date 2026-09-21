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
#include "flow_func/flow_func_manager.h"
#include "flow_func/flow_func_params.h"
#include "flow_func/flow_func_run_context.h"
#include "flow_func/flow_msg_queue.h"

namespace FlowFunc {
namespace {
class FlowFuncStub : public MetaFlowFunc {
 public:
  FlowFuncStub() = default;

  ~FlowFuncStub() override = default;

  int32_t Init() override {
    return FLOW_FUNC_SUCCESS;
  }

  int32_t Proc(const std::vector<std::shared_ptr<FlowMsg>> &input_tensors) override {
    return FLOW_FUNC_SUCCESS;
  }
};

class MultiFlowFuncStub : public MetaMultiFunc {
 public:
  MultiFlowFuncStub() = default;
  ~MultiFlowFuncStub() override = default;

  int32_t Proc1(const std::shared_ptr<MetaRunContext> &run_context,
                const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) {
    auto output = run_context->AllocTensorMsg({1}, TensorDataType::DT_INT64);
    run_context->SetOutput(0, output);
    return FLOW_FUNC_SUCCESS;
  }

  int32_t Proc2(const std::shared_ptr<MetaRunContext> &run_context,
                const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) {
    auto output = run_context->AllocTensorMsg({1}, TensorDataType::DT_INT64);
    run_context->SetOutput(0, output);
    return FLOW_FUNC_SUCCESS;
  }

  int32_t Proc3(const std::shared_ptr<MetaRunContext> &run_context,
                const std::vector<std::shared_ptr<FlowMsgQueue>> &input_msg_queues) {
    auto output = run_context->AllocTensorMsg({1}, TensorDataType::DT_INT64);
    run_context->SetOutput(0, output);
    return FLOW_FUNC_SUCCESS;
  }
};

REGISTER_FLOW_FUNC("Register_success_stub", FlowFuncStub);
FLOW_FUNC_REGISTRAR(MultiFlowFuncStub)
    .RegProcFunc("Proc1", &MultiFlowFuncStub::Proc1)
    .RegProcFunc("Proc2", &MultiFlowFuncStub::Proc2)
    .RegProcFunc("Proc3", &MultiFlowFuncStub::Proc3);
}  // namespace
class FlowFuncManagerUTest : public testing::Test {
 protected:
  virtual void SetUp() {}

  virtual void TearDown() {}
};

TEST_F(FlowFuncManagerUTest, register_func_name_null) {
  auto func = []() -> std::shared_ptr<MetaFlowFunc> { return nullptr; };
  auto ret = RegisterFlowFunc(nullptr, func);
  EXPECT_FALSE(ret);
}

TEST_F(FlowFuncManagerUTest, register_func_null) {
  auto ret = RegisterFlowFunc("test_func_null", nullptr);
  EXPECT_FALSE(ret);
}

TEST_F(FlowFuncManagerUTest, not_register) {
  FlowFuncManager &instance = FlowFuncManager::Instance();
  std::string instance_name = "pp0";
  auto flow_func_wrapper = instance.GetFlowFuncWrapper("not_exit_flow_func", instance_name);
  EXPECT_EQ(flow_func_wrapper, nullptr);
}

TEST_F(FlowFuncManagerUTest, Register_success) {
  FlowFuncManager &instance = FlowFuncManager::Instance();
  std::string instance_name = "pp0";
  auto flow_func_wrapper = instance.GetFlowFuncWrapper("Register_success_stub", instance_name);
  EXPECT_NE(flow_func_wrapper, nullptr);
}

TEST_F(FlowFuncManagerUTest, single_func_register_failed) {
  FlowFuncManager &instance = FlowFuncManager::Instance();
  std::string instance_name = "pp0";
  auto single_func_wrapper = instance.GetFlowFuncWrapper("non_exist_success_stub", instance_name);
  EXPECT_EQ(single_func_wrapper, nullptr);
}

TEST_F(FlowFuncManagerUTest, single_func_init_failed) {
  FlowFuncManager &instance = FlowFuncManager::Instance();
  std::string instance_name = "pp0";
  auto single_func_wrapper = instance.GetFlowFuncWrapper("Register_success_stub", instance_name);
  EXPECT_NE(single_func_wrapper, nullptr);
  std::shared_ptr<FlowFuncParams> null_flow_func_param;
  std::shared_ptr<FlowFuncRunContext> null_run_context;
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams(instance_name, 1, 1, 0, 0));
  std::shared_ptr<FlowFuncRunContext> run_context(new (std::nothrow) FlowFuncRunContext(0, flow_func_param, nullptr));
  int32_t ret = single_func_wrapper->Init(null_flow_func_param, run_context);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
  ret = single_func_wrapper->Init(flow_func_param, null_run_context);
  EXPECT_NE(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(FlowFuncManagerUTest, single_func_register_success) {
  FlowFuncManager &instance = FlowFuncManager::Instance();
  std::string instance_name = "pp0";
  auto single_func_wrapper = instance.GetFlowFuncWrapper("Register_success_stub", instance_name);
  EXPECT_NE(single_func_wrapper, nullptr);
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams(instance_name, 1, 1, 0, 0));
  std::shared_ptr<FlowFuncRunContext> run_context(new (std::nothrow) FlowFuncRunContext(0, flow_func_param, nullptr));
  int32_t ret = single_func_wrapper->Init(flow_func_param, run_context);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  const std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  ret = single_func_wrapper->Proc({input_msgs});
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  const std::vector<std::shared_ptr<FlowMsgQueue>> input_msg_queues;
  ret = single_func_wrapper->Proc({input_msg_queues});  // single func not support stream input
  EXPECT_EQ(ret, FLOW_FUNC_ERR_NOT_SUPPORT);
}

TEST_F(FlowFuncManagerUTest, MultiFuncRegister_success) {
  FlowFuncManager &instance = FlowFuncManager::Instance();
  std::string instance_name = "pp0";
  auto multi_func_wrapper = instance.GetFlowFuncWrapper("Proc1", instance_name);
  EXPECT_NE(multi_func_wrapper, nullptr);
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams(instance_name, 1, 2, 0, 0));
  std::shared_ptr<FlowFuncRunContext> run_context(new (std::nothrow) FlowFuncRunContext(0, flow_func_param, nullptr));
  int32_t ret = multi_func_wrapper->Init(flow_func_param, run_context);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  const std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  ret = multi_func_wrapper->Proc({input_msgs});
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  const std::vector<std::shared_ptr<FlowMsgQueue>> input_msg_queues;
  ret = multi_func_wrapper->Proc({input_msg_queues});
  EXPECT_EQ(ret, FLOW_FUNC_FAILED);
}

TEST_F(FlowFuncManagerUTest, MultiFuncResetFuncStateNotSupport) {
  FlowFuncManager &instance = FlowFuncManager::Instance();
  std::string instance_name = "pp0";
  auto multi_func_wrapper = instance.GetFlowFuncWrapper("Proc1", instance_name);
  EXPECT_NE(multi_func_wrapper, nullptr);
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams(instance_name, 1, 2, 0, 0));
  std::shared_ptr<FlowFuncRunContext> run_context(new (std::nothrow) FlowFuncRunContext(0, flow_func_param, nullptr));
  int32_t ret = multi_func_wrapper->Init(flow_func_param, run_context);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = multi_func_wrapper->ResetFlowFuncState();
  EXPECT_EQ(ret, FLOW_FUNC_ERR_NOT_SUPPORT);
}

TEST_F(FlowFuncManagerUTest, MultiFuncWithQueueRegister_success) {
  FlowFuncManager &instance = FlowFuncManager::Instance();
  std::string instance_name = "pp0";
  auto multi_func_wrapper = instance.GetFlowFuncWrapper("Proc3", instance_name);
  EXPECT_NE(multi_func_wrapper, nullptr);
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams(instance_name, 1, 2, 0, 0));
  std::shared_ptr<FlowFuncRunContext> run_context(new (std::nothrow) FlowFuncRunContext(0, flow_func_param, nullptr));
  int32_t ret = multi_func_wrapper->Init(flow_func_param, run_context);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  const std::vector<std::shared_ptr<FlowMsgQueue>> input_msg_queues;
  ret = multi_func_wrapper->Proc({input_msg_queues});
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  const std::vector<std::shared_ptr<FlowMsg>> input_msgs;
  ret = multi_func_wrapper->Proc({input_msgs});
  EXPECT_EQ(ret, FLOW_FUNC_FAILED);
}
}  // namespace FlowFunc