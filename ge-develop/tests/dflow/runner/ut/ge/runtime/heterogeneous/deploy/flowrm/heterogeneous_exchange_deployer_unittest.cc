/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "macro_utils/dt_public_scope.h"
#include "deploy/flowrm/heterogeneous_exchange_deployer.h"
#include "macro_utils/dt_public_unscope.h"

#include "framework/common/ge_inner_error_codes.h"
#include "proto/deployer.pb.h"
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
  int32_t Sleep(UINT32 microSecond) override {
    return 0;
  }
};
}
class HeterogeneousExchangeDeployerTest : public testing::Test {
 protected:
  void SetUp() override {
    MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaRealPath>());
    HeterogeneousStubEnv::SetupAIServerEnv();
    EXPECT_CALL(mock_exchange_service_, DoCreateQueue).WillRepeatedly(Return(SUCCESS));
    EXPECT_CALL(mock_exchange_service_, DestroyQueue).WillRepeatedly(Return(SUCCESS));
    client_manager_.GetOrCreateClient(0, 0, {0}, false);
    client_manager_.GetOrCreateClient(1, 0, {0}, false);
  }
  void TearDown() override {
    client_manager_.Finalize();
    HeterogeneousStubEnv::ClearEnv();
    MmpaStub::GetInstance().Reset();
  }

  class MockHeterogeneousExchangeDeployer : public HeterogeneousExchangeDeployer {
   public:
    MockHeterogeneousExchangeDeployer(ExchangeService &exchange_service,
                               const deployer::FlowRoutePlan &deploy_plan,
                               FlowGwClientManager &client_manager)
      : HeterogeneousExchangeDeployer(exchange_service, deploy_plan, client_manager) {}

    MOCK_METHOD1(BindRoute, Status(const std::vector<std::pair<const ExchangeEndpoint *,
                                                               const ExchangeEndpoint *>> &));
    MOCK_CONST_METHOD1(CreateTag, Status(ExchangeEndpoint & ));
  };

  stub::MockExchangeService mock_exchange_service_;
  FlowGwClientManager client_manager_;
};

namespace {
void AddEndpoint(deployer::FlowRoutePlan &exchange_plan, int32_t type, const std::string &name,
                 int32_t device_id = 0) {
  auto endpoint = exchange_plan.add_endpoints();
  endpoint->set_type(type);
  endpoint->set_name(name);
  endpoint->set_node_id(0);
  endpoint->set_device_id(device_id);
  endpoint->set_device_type(0);
}

void AddEndpointBinding(deployer::FlowRoutePlan &exchange_plan, int32_t src_index, int32_t dst_index) {
  auto binding = exchange_plan.add_bindings();
  binding->set_src_index(src_index);
  binding->set_dst_index(dst_index);
}

void AddEndpointGroup(deployer::FlowRoutePlan &exchange_plan,
                      const std::string &name,
                      const std::vector<int32_t> &group) {
  auto endpoint = exchange_plan.add_endpoints();
  endpoint->set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeGroup));
  endpoint->set_name(name);
  endpoint->mutable_group_desc()->mutable_endpoint_indices()->Add(group.begin(), group.end());
}

/**
 *    NetOutput
 *        |
 *      PC_2
 *        |
 *      PC_1
 *        |
 *      data1
 */
deployer::FlowRoutePlan BuildExchangePlanSimple() {
  deployer::FlowRoutePlan exchange_plan;
  // Model input
  AddEndpoint(exchange_plan, 1, "input-0");
  // PartitionedCall_1 input
  AddEndpoint(exchange_plan, 2, "input-0");
  // PartitionedCall_1 output
  AddEndpoint(exchange_plan, 1, "internal-queue-0");
  // PartitionedCall_2 output
  AddEndpoint(exchange_plan, 2, "output-0");
  // Model output
  AddEndpoint(exchange_plan, 1, "output-0");
  AddEndpoint(exchange_plan, 5, "output-1");

  AddEndpointBinding(exchange_plan, 0, 1);
  AddEndpointBinding(exchange_plan, 3, 4);
  return exchange_plan;
}

deployer::FlowRoutePlan BuildExchangePlanWithDummyQ() {
  deployer::FlowRoutePlan exchange_plan;
  AddEndpoint(exchange_plan, 1, "input-0");
  AddEndpoint(exchange_plan, 1, "output-0");
  AddEndpoint(exchange_plan, 5, "output-1");
  return exchange_plan;
}

/**
 *    NetOutput
 *        |
 *      PC_2
 *        |  \
 *      PC_1 PC_1
 *        |  /
 *      data1
 */
deployer::FlowRoutePlan BuildExchangePlanSimple2PG() {
  deployer::FlowRoutePlan exchange_plan;
  // Model input, index:0
  AddEndpoint(exchange_plan, 1, "input");
  // PartitionedCall_1@0_0_0, index:1
  AddEndpoint(exchange_plan, 2, "data1_output_-1_0_to_pc1_input0_0_0");
  // PartitionedCall_1@0_0_0 output, index:2
  AddEndpoint(exchange_plan, 1, "pc1_output0_0_0_to_pc2_input0_0_0");
  // PartitionedCall_1@0_0_1 input, index:3
  AddEndpoint(exchange_plan, 2, "data1_output_-1_0_to_pc1_input0_0_1");
  // PartitionedCall_1@0_0_1 output, index:4
  AddEndpoint(exchange_plan, 1, "pc1_output0_0_0_to_pc2_input0_0_1");
  // PartitionedCall_2 input, index:5
  AddEndpoint(exchange_plan, 1, "pc2_input");
  // PartitionedCall_2 output, index:6
  AddEndpoint(exchange_plan, 2, "pc2_output");
  // Model output, index:7
  AddEndpoint(exchange_plan, 1, "output");

  // pc1 input group, index:8
  AddEndpointGroup(exchange_plan, "pc1_input", {1, 3});
  // pc1 input@0_0_0 group num,  index:9
  AddEndpointGroup(exchange_plan, "data1_output_-1_0_to_pc1_input0_0_0", {0});
  // pc1 input@0_0_1 group num,  index:10
  AddEndpointGroup(exchange_plan, "data1_output_-1_0_to_pc1_input0_0_1", {0});

  // pc1 output group, index:11
  AddEndpointGroup(exchange_plan, "pc1_output", {2, 4});
  // pc1 output@0_0_0 group num, index:12
  AddEndpointGroup(exchange_plan, "pc1_output0_0_0_to_pc2_input0_0_0", {5});
  // pc1 output@0_0_1 group num, index:13
  AddEndpointGroup(exchange_plan, "pc1_output0_0_0_to_pc2_input0_0_1", {5});

  // device 1, index:14
  AddEndpoint(exchange_plan, 1, "pc2_output");
  // device 1, index:15
  AddEndpoint(exchange_plan, 1, "pc1_output");
  // device 1, index:16
  AddEndpoint(exchange_plan, 1, "pc2_output_1", 1);
  // device 1, index:17
  AddEndpoint(exchange_plan, 1, "pc1_output_1", 1);
  // exception group, index:18
  AddEndpointGroup(exchange_plan, "exception1", {2, 16});
  // exception group, index:19
  AddEndpointGroup(exchange_plan, "exception2", {4, 17});
  // exception group, index:20
  AddEndpointGroup(exchange_plan, "exception3", {14, 15});
  // exception group, index:21
  AddEndpointGroup(exchange_plan, "exception4", {2, 4});

  AddEndpointBinding(exchange_plan, 0, 8);
  AddEndpointBinding(exchange_plan, 0, 1);
  AddEndpointBinding(exchange_plan, 0, 3);
  AddEndpointBinding(exchange_plan, 9, 1);
  AddEndpointBinding(exchange_plan, 10, 3);
  AddEndpointBinding(exchange_plan, 2, 12);
  AddEndpointBinding(exchange_plan, 2, 5);
  AddEndpointBinding(exchange_plan, 4, 13);
  AddEndpointBinding(exchange_plan, 4, 5);
  AddEndpointBinding(exchange_plan, 11, 5);
  AddEndpointBinding(exchange_plan, 6, 7);
  AddEndpointBinding(exchange_plan, 14, 15);
  AddEndpointBinding(exchange_plan, 18, 19);
  AddEndpointBinding(exchange_plan, 16, 20);
  AddEndpointBinding(exchange_plan, 21, 17);
  return exchange_plan;
}
}  // namespace

TEST_F(HeterogeneousExchangeDeployerTest, BindRoute) {
  deployer::FlowRoutePlan flow_route_plan;
  HeterogeneousExchangeDeployer exchange_deployer(mock_exchange_service_, flow_route_plan, client_manager_);
  std::vector<std::pair<const ExchangeEndpoint *,
                        const ExchangeEndpoint *>> queue_routes;
  EXPECT_EQ(exchange_deployer.BindRoute(queue_routes), SUCCESS);
}

TEST_F(HeterogeneousExchangeDeployerTest, UnbindRoute) {
  std::vector<std::pair<const ExchangeEndpoint *,
                        const ExchangeEndpoint *>> queue_routes;
  EXPECT_EQ(HeterogeneousExchangeDeployer::UnbindRoute(queue_routes, client_manager_), SUCCESS);
}

TEST_F(HeterogeneousExchangeDeployerTest, CreateGroupFailed_EntryTypeIsGroup) {
  deployer::FlowRoutePlan flow_route_plan;
  AddEndpointGroup(flow_route_plan, "input_group", {2, 2});
  AddEndpoint(flow_route_plan, static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue), "input_queue");
  AddEndpointGroup(flow_route_plan, "p0_output_queue", {2, 2});
  AddEndpointBinding(flow_route_plan, 0, 1);
  MockHeterogeneousExchangeDeployer exchange_deployer(mock_exchange_service_, flow_route_plan, client_manager_);
  EXPECT_CALL(exchange_deployer, CreateTag).WillRepeatedly(Return(SUCCESS));
  EXPECT_EQ(exchange_deployer.PreDeploy(), INTERNAL_ERROR);
}

TEST_F(HeterogeneousExchangeDeployerTest, CreateExchangeEndpoints) {
  auto exchange_plan = BuildExchangePlanSimple();
  EXPECT_CALL(mock_exchange_service_, DoCreateQueue).Times(3);
  MockHeterogeneousExchangeDeployer exchange_deployer(mock_exchange_service_, exchange_plan, client_manager_);
  EXPECT_CALL(exchange_deployer, CreateTag).WillRepeatedly(Return(SUCCESS));

  ExchangeRoute exchange_route;
  EXPECT_EQ(exchange_deployer.CreateExchangeEndpoints(), SUCCESS);
}

TEST_F(HeterogeneousExchangeDeployerTest, CreateExchangeEndpointsWithDummQ) {
  auto exchange_plan = BuildExchangePlanWithDummyQ();
  EXPECT_CALL(mock_exchange_service_, DoCreateQueue).Times(2);
  MockHeterogeneousExchangeDeployer exchange_deployer(mock_exchange_service_, exchange_plan, client_manager_);
  ExchangeRoute exchange_route;
  EXPECT_EQ(exchange_deployer.CreateExchangeEndpoints(), SUCCESS);
}

TEST_F(HeterogeneousExchangeDeployerTest, CreateExchangeEndpoints_ExternalQueue) {
  deployer::FlowRoutePlan flow_route_plan;
  AddEndpoint(flow_route_plan, static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeExternalQueue), "input_queue");
  EXPECT_CALL(mock_exchange_service_, DoCreateQueue).Times(0);
  MockHeterogeneousExchangeDeployer exchange_deployer(mock_exchange_service_, flow_route_plan, client_manager_);
  EXPECT_EQ(exchange_deployer.CreateExchangeEndpoints(), SUCCESS);
}

TEST_F(HeterogeneousExchangeDeployerTest, CreateGroupFailed_SingleEntry) {
  deployer::FlowRoutePlan flow_route_plan;
  AddEndpointGroup(flow_route_plan, "input_group", {2});
  AddEndpoint(flow_route_plan, static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue), "input_queue");
  AddEndpoint(flow_route_plan, static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeTag), "p0_output_queue");
  AddEndpointBinding(flow_route_plan, 0, 1);
  MockHeterogeneousExchangeDeployer exchange_deployer(mock_exchange_service_, flow_route_plan, client_manager_);
  EXPECT_CALL(exchange_deployer, CreateTag).WillRepeatedly(Return(SUCCESS));
  EXPECT_EQ(exchange_deployer.PreDeploy(), SUCCESS);
  EXPECT_EQ(exchange_deployer.GetRoute()->groups_.size(), 0);
}

TEST_F(HeterogeneousExchangeDeployerTest, CreateGroupFailed_SrcGroupEntryIndexOutOfRange) {
  deployer::FlowRoutePlan flow_route_plan;
  AddEndpointGroup(flow_route_plan, "input_group", {2, 3});  // 3 is invalid
  AddEndpoint(flow_route_plan, static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue), "input_queue");
  AddEndpoint(flow_route_plan, static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeTag), "p0_output_queue");
  AddEndpointBinding(flow_route_plan, 0, 1);
  MockHeterogeneousExchangeDeployer exchange_deployer(mock_exchange_service_, flow_route_plan, client_manager_);
  EXPECT_CALL(exchange_deployer, CreateTag).WillRepeatedly(Return(SUCCESS));
  EXPECT_EQ(exchange_deployer.PreDeploy(), PARAM_INVALID);
}

TEST_F(HeterogeneousExchangeDeployerTest, CreateGroupFailed_DstGroupEntryIndexOutOfRange) {
  deployer::FlowRoutePlan flow_route_plan;
  AddEndpoint(flow_route_plan, static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue), "output_queue");
  AddEndpointGroup(flow_route_plan, "input_group", {2, 3});  // 3 is invalid
  AddEndpoint(flow_route_plan, static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeTag), "p1_input_queue");
  AddEndpointBinding(flow_route_plan, 0, 1);
  HeterogeneousExchangeDeployer exchange_deployer(mock_exchange_service_, flow_route_plan, client_manager_);
  EXPECT_EQ(exchange_deployer.PreDeploy(), PARAM_INVALID);
}

TEST_F(HeterogeneousExchangeDeployerTest, TestDeployModel) {
  auto exchange_plan = BuildExchangePlanSimple();
  MockHeterogeneousExchangeDeployer exchange_deployer(mock_exchange_service_, exchange_plan, client_manager_);
  EXPECT_CALL(exchange_deployer, CreateTag).WillRepeatedly(Return(SUCCESS));
  EXPECT_CALL(exchange_deployer, BindRoute).WillRepeatedly(Return(SUCCESS));

  ExchangeRoute exchange_route;
  ASSERT_EQ(exchange_deployer.Deploy(exchange_route), SUCCESS);

  std::vector<uint32_t> queue_ids;
  // out of range
  exchange_route.GetQueueIds({99, 100}, queue_ids);
  ASSERT_EQ(queue_ids.size(), 0);
  // endpoint is not a queue
  exchange_route.GetQueueIds({1}, queue_ids);
  ASSERT_EQ(queue_ids.size(), 0);
  exchange_route.GetQueueIds({3}, queue_ids);
  ASSERT_EQ(queue_ids.size(), 0);
  exchange_route.GetQueueIds({0, 2, 4}, queue_ids);
  ASSERT_EQ(queue_ids.size(), 3);
  // endpoint is not exist
  ASSERT_EQ(exchange_route.IsProxyQueue(10000), false);
  ASSERT_EQ(MockHeterogeneousExchangeDeployer::Undeploy(mock_exchange_service_, exchange_route, client_manager_), SUCCESS);
}

TEST_F(HeterogeneousExchangeDeployerTest, TestDeployModel2PG) {
  auto exchange_plan = BuildExchangePlanSimple2PG();
  MockHeterogeneousExchangeDeployer exchange_deployer(mock_exchange_service_, exchange_plan, client_manager_);
  EXPECT_CALL(exchange_deployer, CreateTag).WillRepeatedly(Return(SUCCESS));
  EXPECT_CALL(exchange_deployer, BindRoute).WillRepeatedly(Return(SUCCESS));

  ExchangeRoute exchange_route;
  ASSERT_EQ(exchange_deployer.Deploy(exchange_route), SUCCESS);
  ASSERT_EQ(MockHeterogeneousExchangeDeployer::Undeploy(mock_exchange_service_, exchange_route, client_manager_), SUCCESS);
}

TEST_F(HeterogeneousExchangeDeployerTest, UpdateExceptionRoutes) {
  auto exchange_plan = BuildExchangePlanSimple2PG();
  MockHeterogeneousExchangeDeployer exchange_deployer(mock_exchange_service_, exchange_plan, client_manager_);
  EXPECT_CALL(exchange_deployer, CreateTag).WillRepeatedly(Return(SUCCESS));
  EXPECT_CALL(exchange_deployer, BindRoute).WillRepeatedly(Return(SUCCESS));

  ExchangeRoute exchange_route;
  ASSERT_EQ(exchange_deployer.Deploy(exchange_route), SUCCESS);
  FlowGwClient::ExceptionDeviceInfo device;
  device.node_id = 0;
  device.device_id = 0;
  device.device_type = 0;
  std::vector<FlowGwClient::ExceptionDeviceInfo> devices;
  devices.emplace_back(device);
  ASSERT_EQ(MockHeterogeneousExchangeDeployer::UpdateExceptionRoutes(exchange_route,
      client_manager_, devices), SUCCESS);
}

TEST_F(HeterogeneousExchangeDeployerTest, DeleteExceptionGroup) {
  auto exchange_plan = BuildExchangePlanSimple2PG();
  MockHeterogeneousExchangeDeployer exchange_deployer(mock_exchange_service_, exchange_plan, client_manager_);
  EXPECT_CALL(exchange_deployer, CreateTag).WillRepeatedly(Return(SUCCESS));
  EXPECT_CALL(exchange_deployer, BindRoute).WillRepeatedly(Return(SUCCESS));

  ExchangeRoute exchange_route;
  ASSERT_EQ(exchange_deployer.Deploy(exchange_route), SUCCESS);
  FlowGwClient::ExceptionDeviceInfo device;
  device.node_id = 1;
  device.device_id = 1;
  device.device_type = 0;
  std::vector<FlowGwClient::ExceptionDeviceInfo> devices;
  devices.emplace_back(device);
  MockHeterogeneousExchangeDeployer::DeleteExceptionGroup(exchange_route, devices);
}
}  // namespace ge
