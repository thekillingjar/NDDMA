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
#include "model/attr_value_impl.h"
#include "utils/udf_utils.h"

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
    return FLOW_FUNC_SUCCESS;
  }

  int32_t Proc2(const std::shared_ptr<MetaRunContext> &run_context,
                const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) {
    return FLOW_FUNC_SUCCESS;
  }
};

REGISTER_FLOW_FUNC("Register_success_stub", FlowFuncStub);
FLOW_FUNC_REGISTRAR(MultiFlowFuncStub)
    .RegProcFunc("Proc1", &MultiFlowFuncStub::Proc1)
    .RegProcFunc("Proc2", &MultiFlowFuncStub::Proc2);
}  // namespace
class FlowFuncManagerSTest : public testing::Test {
 protected:
  virtual void SetUp() {}

  virtual void TearDown() {}
};

TEST_F(FlowFuncManagerSTest, register_func_name_null) {
  auto func = []() -> std::shared_ptr<MetaFlowFunc> { return nullptr; };
  auto ret = RegisterFlowFunc(nullptr, func);
  EXPECT_FALSE(ret);
}

TEST_F(FlowFuncManagerSTest, register_func_null) {
  auto ret = RegisterFlowFunc("test_func_null", nullptr);
  EXPECT_FALSE(ret);
}

TEST_F(FlowFuncManagerSTest, not_register) {
  FlowFuncManager &instance = FlowFuncManager::Instance();
  std::string instance_name = "pp0";
  auto flow_func_wrapper = instance.GetFlowFuncWrapper("not_exit_flow_func", instance_name);
  EXPECT_EQ(flow_func_wrapper, nullptr);
}

TEST_F(FlowFuncManagerSTest, Register_success) {
  FlowFuncManager &instance = FlowFuncManager::Instance();
  std::string instance_name = "pp0";
  auto flow_func_wrapper = instance.GetFlowFuncWrapper("Register_success_stub", instance_name);
  EXPECT_NE(flow_func_wrapper, nullptr);
}

TEST_F(FlowFuncManagerSTest, single_func_register_failed) {
  FlowFuncManager &instance = FlowFuncManager::Instance();
  std::string instance_name = "pp0";
  auto single_func_wrapper = instance.GetFlowFuncWrapper("non_exist_success_stub", instance_name);
  EXPECT_EQ(single_func_wrapper, nullptr);
}

TEST_F(FlowFuncManagerSTest, single_func_register_success) {
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
}

TEST_F(FlowFuncManagerSTest, MultiFuncRegister_success) {
  FlowFuncManager &instance = FlowFuncManager::Instance();
  std::string instance_name = "pp0";
  auto multi_func_wrapper = instance.GetFlowFuncWrapper("Proc1", instance_name);
  EXPECT_NE(multi_func_wrapper, nullptr);
  std::map<std::string, std::shared_ptr<const AttrValue>> attrs;
  ff::udf::AttrValue data_type;
  data_type.set_type((uint32_t)TensorDataType::DT_UINT32);
  auto data_type_attr = std::make_shared<AttrValueImpl>(data_type);
  attrs["out_type"] = data_type_attr;
  std::shared_ptr<FlowFuncParams> flow_func_param(new (std::nothrow) FlowFuncParams(instance_name, 1, 1, 0, 0));
  flow_func_param->SetAttrMap(attrs);
  std::shared_ptr<FlowFuncRunContext> run_context(new (std::nothrow) FlowFuncRunContext(0, flow_func_param, nullptr));
  int32_t ret = multi_func_wrapper->Init(flow_func_param, run_context);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
}
}  // namespace FlowFunc