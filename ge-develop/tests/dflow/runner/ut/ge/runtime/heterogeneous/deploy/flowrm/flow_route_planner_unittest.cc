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
#include <memory>

#include "macro_utils/dt_public_scope.h"
#include "deploy/flowrm/flow_route_planner.h"
#include "deploy/resource/resource_manager.h"
#include "macro_utils/dt_public_unscope.h"

#include "framework/common/ge_inner_error_codes.h"
#include "proto/deployer.pb.h"
#include "stub_models.h"
#include "stub/heterogeneous_stub_env.h"
#include "depends/mmpa/src/mmpa_stub.h"

using namespace std;
using namespace ::testing;

namespace ge {
namespace {
class MockMmpaRealPath : public ge::MmpaStubApiGe {
public:
  int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) override {
    (void)strncpy_s(realPath, realPathLen, path, strlen(path));
    return 0;
  }
};
}
class FlowRoutePlannerTest : public testing::Test {
 protected:
  void SetUp() override {
    MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaRealPath>());
    HeterogeneousStubEnv::SetupAIServerEnv();
  }

  void TearDown() override {
    HeterogeneousStubEnv::ClearEnv();
    MmpaStub::GetInstance().Reset();
  }

  static size_t CountEndpoints(const deployer::FlowRoutePlan &route_plan, int32_t type) {
    return std::count_if(route_plan.endpoints().begin(), route_plan.endpoints().end(),
                         [type](const deployer::EndpointDesc &queue_desc) -> bool {
                           return queue_desc.type() == type;
                         });
  }

  static size_t CountValidEndpoints(const deployer::FlowRoutePlan &route_plan) {
    return std::count_if(route_plan.endpoints().begin(), route_plan.endpoints().end(),
                         [](const deployer::EndpointDesc &queue_desc) -> bool {
                           return queue_desc.type() >= 0;
                         });
  }
};

TEST_F(FlowRoutePlannerTest, ResolveFlowRoutePlan_IrrelevantNode) {
  auto deploy_plan = StubModels::BuildSingleModelDeployPlan();
  deployer::FlowRoutePlan exchange_plan;
  ASSERT_EQ(FlowRoutePlanner::ResolveFlowRoutePlan(deploy_plan,
                                                   2,
                                                   exchange_plan),
            SUCCESS);
  ASSERT_EQ(CountValidEndpoints(exchange_plan), 0);
  ASSERT_EQ(exchange_plan.bindings_size(), 0);
}

TEST_F(FlowRoutePlannerTest, ResolveFlowRoutePlan_LocalDevice) {
  auto deploy_plan = StubModels::BuildSingleModelDeployPlan();
  deployer::FlowRoutePlan exchange_plan;
  ASSERT_EQ(FlowRoutePlanner::ResolveFlowRoutePlan(deploy_plan,
                                                   0,
                                                   exchange_plan),
            SUCCESS);
  ASSERT_EQ(exchange_plan.bindings_before_load_size(), 2);
}

TEST_F(FlowRoutePlannerTest, ResolveFlowRoutePlan_LocalDeviceWithDummyQ) {
  auto deploy_plan = StubModels::BuildSingleModelDeployPlanWithDummyQ();
  deployer::FlowRoutePlan exchange_plan;
  ASSERT_EQ(FlowRoutePlanner::ResolveFlowRoutePlan(deploy_plan,
                                                   0,
                                                   exchange_plan),
            SUCCESS);
  ASSERT_EQ(exchange_plan.bindings_before_load_size(), 2);
}

TEST_F(FlowRoutePlannerTest, TestResolveProxyPlan) {
  HeterogeneousStubEnv::ClearEnv();
  HeterogeneousStubEnv::SetupAIServerEnv();
  auto deploy_plan = StubModels::BuildSingleModelDeployPlanWithProxy(0);
  DeployState deploy_state;
  deploy_state.SetDeployPlan(std::move(deploy_plan));
  ASSERT_EQ(FlowRoutePlanner::ResolveFlowRoutePlans(deploy_state), SUCCESS);
}

///    NetOutput (4)       8->4(input_group)
///        |
///      PC_2 (2)/(3)      3->7(output_group)
///        |
///      PC_1 (1)/(2)     6(input_group) ->1
///        |
///      data1(0)  0->5(output_group)
TEST_F(FlowRoutePlannerTest, ResolveFlowRoutePlan_RemoteDevice) {
  auto deploy_plan = StubModels::BuildSingleModelDeployPlan();
  deployer::FlowRoutePlan exchange_plan;
  ASSERT_EQ(FlowRoutePlanner::ResolveFlowRoutePlan(deploy_plan,
                                                   1,
                                                   exchange_plan),
            SUCCESS);
  auto num_queues = CountEndpoints(exchange_plan, static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue));
  auto num_tags = CountEndpoints(exchange_plan, static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeTag));
  ASSERT_EQ(num_queues, 2);
  ASSERT_EQ(num_tags, 2);
  ASSERT_EQ(exchange_plan.bindings_before_load_size(), 2);
}
}  // namespace ge
