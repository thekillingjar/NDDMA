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
#include "dvpp_ops_kernel_builder.h"
#undef protected
#undef private

#include "stub/stub.h"
#include "util/dvpp_constexpr.h"
#include "graph/compute_graph.h"

TEST(DvppOpsKernelBuilder, AllTest_910B)
{
  dvpp::DvppOpsKernelBuilder dvppOpsKernelBuilder;

  // Initialize
  std::map<std::string, std::string> options;
  options[ge::SOC_VERSION] = dvpp::kSocVersionAscend910B1;
  ge::Status status = dvppOpsKernelBuilder.Initialize(options);
  ASSERT_EQ(status, ge::SUCCESS);

  // CalcOpRunningParam
  ge::OpDescPtr op_desc_ptr = std::make_shared<ge::OpDesc>("Resize", "ResizeBilinearV2");
  ge::GeTensorDesc tensorX(ge::GeShape(std::vector<int64_t>{1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_UINT8);
  op_desc_ptr->AddInputDesc("x", tensorX);
  ge::GeTensorDesc tensorSize(ge::GeShape(std::vector<int64_t>{1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_INT32);
  op_desc_ptr->AddInputDesc("size", tensorSize);
  ge::GeTensorDesc tensorY(ge::GeShape(std::vector<int64_t>{1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_UINT8);
  op_desc_ptr->AddOutputDesc("y", tensorY);
  ge::AttrUtils::SetBool(op_desc_ptr, "align_corners", false);
  ge::AttrUtils::SetBool(op_desc_ptr, "half_pixel_centers", false);
  ge::AttrUtils::SetStr(op_desc_ptr, "dtype", "DT_UINT8");
  std::shared_ptr<ge::ComputeGraph> graphPtr = std::make_shared<ge::ComputeGraph>("test_graph");
  status = dvppOpsKernelBuilder.CalcOpRunningParam(*(graphPtr->AddNode(op_desc_ptr)));
  ASSERT_EQ(status, ge::SUCCESS);

  // GenerateTask
  ge::RunContext context = CreateContext();
  std::vector<domi::TaskDef> tasks;
  status = dvppOpsKernelBuilder.GenerateTask(*(graphPtr->AddNode(op_desc_ptr)), context, tasks);
  ASSERT_EQ(status, ge::SUCCESS);
  DestroyContext(context);

  // Finalize
  status = dvppOpsKernelBuilder.Finalize();
  ASSERT_EQ(status, ge::SUCCESS);
}

TEST(DvppOpsKernelBuilder, Initialize)
{
  dvpp::DvppOpsKernelBuilder dvppOpsKernelBuilder;
  std::map<std::string, std::string> options;
  options[ge::SOC_VERSION] = "Ascend";
  ge::Status status = dvppOpsKernelBuilder.Initialize(options);
  ASSERT_EQ(status, ge::SUCCESS);
}