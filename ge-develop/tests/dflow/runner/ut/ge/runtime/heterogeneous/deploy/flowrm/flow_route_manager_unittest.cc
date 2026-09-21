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
#include "deploy/flowrm/flow_route_manager.h"
#include "macro_utils/dt_public_unscope.h"

#include "framework/common/ge_inner_error_codes.h"
#include "proto/deployer.pb.h"

using namespace std;
using namespace ::testing;

namespace ge {
class FlowRouteManagerTest : public testing::Test {
};

TEST_F(FlowRouteManagerTest, GenerateOuteId) {
  int64_t route_id = -1;
  FlowRouteManager flow_route_manager;
  EXPECT_EQ(flow_route_manager.GenerateRouteId(route_id), SUCCESS);
  EXPECT_EQ(route_id, 0);
  EXPECT_EQ(flow_route_manager.GenerateRouteId(route_id), SUCCESS);
  EXPECT_EQ(route_id, 1);
  flow_route_manager.route_id_gen_ = INT64_MAX;
  EXPECT_EQ(flow_route_manager.GenerateRouteId(route_id), SUCCESS);
  EXPECT_EQ(route_id, INT64_MAX);
  EXPECT_EQ(flow_route_manager.GenerateRouteId(route_id), FAILED);
}

TEST_F(FlowRouteManagerTest, QueryRoute_RouteNotFound) {
  FlowRouteManager flow_route_manager;
  int64_t route_id = -1;
  EXPECT_TRUE(flow_route_manager.QueryRoute(FlowRouteType::kModelFlowRoute, route_id) == nullptr);
}

TEST_F(FlowRouteManagerTest, QueryRoute_RouteIdNotFound) {
  FlowRouteManager flow_route_manager;
  int64_t route_id = 1;
  flow_route_manager.exchange_routes_[FlowRouteType::kModelFlowRoute][route_id] = ExchangeRoute{};
  EXPECT_TRUE(flow_route_manager.QueryRoute(FlowRouteType::kModelFlowRoute, 2) == nullptr);
}

TEST_F(FlowRouteManagerTest, QueryRoute_Success) {
  FlowRouteManager flow_route_manager;
  int64_t route_id = 666;
  flow_route_manager.exchange_routes_[FlowRouteType::kModelFlowRoute][route_id] = ExchangeRoute{};
  EXPECT_EQ(flow_route_manager.QueryRoute(FlowRouteType::kModelFlowRoute, route_id),
            &flow_route_manager.exchange_routes_[FlowRouteType::kModelFlowRoute][route_id]);
}

TEST_F(FlowRouteManagerTest, UpdateExceptionRoutes_Success) {
  FlowRouteManager flow_route_manager;
  int64_t route_id = 666;
  FlowGwClientManager client_manager;
  std::vector<FlowGwClient::ExceptionDeviceInfo> devices;
  flow_route_manager.exchange_routes_[FlowRouteType::kModelFlowRoute][route_id] = ExchangeRoute{};
  EXPECT_NO_THROW(flow_route_manager.UpdateExceptionRoutes(FlowRouteType::kModelFlowRoute, route_id, client_manager, devices));
}
}  // namespace ge
