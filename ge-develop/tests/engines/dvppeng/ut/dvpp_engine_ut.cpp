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
#include <map>
#include <string>

#define private public
#define protected public
#include "dvpp_engine.h"
#undef protected
#undef private

#include "util/dvpp_constexpr.h"

TEST(DvppEngine, Initialize)
{
  std::map<std::string, std::string> options;
  options[ge::SOC_VERSION] = dvpp::kSocVersionAscend910B1;
  ge::Status status = Initialize(options);
  ASSERT_EQ(status, ge::SUCCESS);
}

TEST(DvppEngine, GetOpsKernelInfoStores)
{
  std::map<std::string, OpsKernelInfoStorePtr> ops_kernel_info_stores;
  GetOpsKernelInfoStores(ops_kernel_info_stores);
  ASSERT_NE(ops_kernel_info_stores[dvpp::kDvppOpsKernel], nullptr);

  // Initialize
  std::map<std::string, std::string> options;
  options[ge::SOC_VERSION] = dvpp::kSocVersionAscend910B1;
  ge::Status status = ops_kernel_info_stores[dvpp::kDvppOpsKernel]->Initialize(options);
  ASSERT_EQ(status, ge::SUCCESS);
}

TEST(DvppEngine, GetGraphOptimizerObjs)
{
  std::map<std::string, GraphOptimizerPtr> graph_optimizers;
  GetGraphOptimizerObjs(graph_optimizers);
  ASSERT_NE(graph_optimizers[dvpp::kDvppGraphOptimizer], nullptr);

  // Initialize
  std::map<std::string, std::string> options;
  options[ge::SOC_VERSION] = dvpp::kSocVersionAscend910B1;
  ge::Status status = graph_optimizers[dvpp::kDvppGraphOptimizer]->Initialize(options, nullptr);
  ASSERT_EQ(status, ge::SUCCESS);
}

TEST(DvppEngine, GetOpsKernelBuilderObjs)
{
  std::map<std::string, OpsKernelBuilderPtr> ops_kernel_builders;
  GetOpsKernelBuilderObjs(ops_kernel_builders);
  ASSERT_NE(ops_kernel_builders[dvpp::kDvppOpsKernel], nullptr);

  // Initialize
  std::map<std::string, std::string> options;
  options[ge::SOC_VERSION] = dvpp::kSocVersionAscend910B1;
  ge::Status status = ops_kernel_builders[dvpp::kDvppOpsKernel]->Initialize(options);
  ASSERT_EQ(status, ge::SUCCESS);
}

TEST(DvppEngine, Finalize)
{
  ge::Status status = Finalize();
  ASSERT_EQ(status, ge::SUCCESS);
}