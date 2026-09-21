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
#include <nlohmann/json.hpp>

#include <memory>

#define protected public
#define private public
#include "common/configuration.h"
#include "fusion_manager/fusion_manager.h"
#include "graph_optimizer/spacesize_calculator/spacesize_calculator.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#undef private
#undef protected

#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common/util/op_info_util.h"
#include "graph/utils/tensor_utils.h"

#include "fusion_manager/fusion_manager.h"

using namespace std;
using namespace ge;
using namespace fe;



class spacesize_calculator_st : public testing::Test
{
protected:
  std::shared_ptr<SpaceSizeCalculator> space_size_calculator_ptr_;
  void SetUp()
  {
    space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();
  }

  void TearDown()
  {

  }

  static void CreateTensorSizeGraph(ComputeGraphPtr graph,
    GeTensorDesc &in_tensor_desc, GeTensorDesc &out_tensor_desc, int impl_type) {
    OpDescPtr op_desc = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
    op_desc->AddInputDesc("x", in_tensor_desc);
    op_desc->AddOutputDesc("y", out_tensor_desc);
    ge::AttrUtils::SetInt(op_desc, FE_IMPLY_TYPE, impl_type);
    NodePtr node = graph->AddNode(op_desc);
  }
};

TEST_F(spacesize_calculator_st, cce_nchw_success)
{
  GeShape in_shape({1,7,7,1});
  GeTensorDesc in_tensor_desc(in_shape);
  in_tensor_desc.SetOriginFormat(FORMAT_NCHW);
  in_tensor_desc.SetFormat(FORMAT_NCHW);
  in_tensor_desc.SetDataType(DT_FLOAT16);
  GeShape out_shape({4,16,7,7});
  GeTensorDesc out_tensor_desc(out_shape);
  out_tensor_desc.SetOriginFormat(FORMAT_NCHW);
  out_tensor_desc.SetFormat(FORMAT_NCHW);
  out_tensor_desc.SetDataType(DT_FLOAT);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateTensorSizeGraph(graph, in_tensor_desc, out_tensor_desc, 6);
  Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  for (auto node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    int64_t insize = 0;
    ge::TensorUtils::GetSize(op_desc_ptr->GetInputDesc(0), insize);
    EXPECT_EQ(insize, 160);

    int64_t outsize = 0;
    ge::TensorUtils::GetSize(op_desc_ptr->GetOutputDesc(0), outsize);
    EXPECT_EQ(outsize, 12576);
  }
}

TEST_F(spacesize_calculator_st, cce_nhwc_success)
{
  GeShape in_shape({1,7,7,1});
  GeTensorDesc in_tensor_desc(in_shape);
  in_tensor_desc.SetOriginFormat(FORMAT_NCHW);
  in_tensor_desc.SetFormat(FORMAT_NHWC);
  in_tensor_desc.SetDataType(DT_FLOAT16);
  GeShape out_shape({4,16,7,7});
  GeTensorDesc out_tensor_desc(out_shape);
  out_tensor_desc.SetOriginFormat(FORMAT_NCHW);
  out_tensor_desc.SetFormat(FORMAT_NHWC);
  out_tensor_desc.SetDataType(DT_FLOAT);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateTensorSizeGraph(graph, in_tensor_desc, out_tensor_desc, 6);
  Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  for (auto node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    int64_t insize = 0;
    ge::TensorUtils::GetSize(op_desc_ptr->GetInputDesc(0), insize);
    EXPECT_EQ(insize, 160);

    int64_t outsize = 0;
    ge::TensorUtils::GetSize(op_desc_ptr->GetOutputDesc(0), outsize);
    EXPECT_EQ(outsize, 12576);
  }
}

TEST_F(spacesize_calculator_st, cce_hwcn_success)
{
  GeShape in_shape({1,7,7,1});
  GeTensorDesc in_tensor_desc(in_shape);
  in_tensor_desc.SetOriginFormat(FORMAT_NCHW);
  in_tensor_desc.SetFormat(FORMAT_HWCN);
  in_tensor_desc.SetDataType(DT_DOUBLE);
  GeShape out_shape({4,3,7,7});
  GeTensorDesc out_tensor_desc(out_shape);
  out_tensor_desc.SetOriginFormat(FORMAT_NCHW);
  out_tensor_desc.SetFormat(FORMAT_HWCN);
  out_tensor_desc.SetDataType(DT_BOOL);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateTensorSizeGraph(graph, in_tensor_desc, out_tensor_desc, 6);
  Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  for (auto node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    int64_t insize = 0;
    ge::TensorUtils::GetSize(op_desc_ptr->GetInputDesc(0), insize);
    EXPECT_EQ(insize, 448);

    int64_t outsize = 0;
    ge::TensorUtils::GetSize(op_desc_ptr->GetOutputDesc(0), outsize);
    EXPECT_EQ(outsize, 640);
  }
}

TEST_F(spacesize_calculator_st, cce_nd_success)
{
  std::vector<int64_t> dims = {};
  GeShape in_shape(dims);
  GeTensorDesc in_tensor_desc(in_shape);
  in_tensor_desc.SetOriginFormat(FORMAT_ND);
  in_tensor_desc.SetFormat(FORMAT_ND);
  in_tensor_desc.SetDataType(DT_INT32);
  GeShape out_shape({4,3,7,7});
  GeTensorDesc out_tensor_desc(out_shape);
  out_tensor_desc.SetOriginFormat(FORMAT_ND);
  out_tensor_desc.SetFormat(FORMAT_ND);
  out_tensor_desc.SetDataType(DT_INT16);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateTensorSizeGraph(graph, in_tensor_desc, out_tensor_desc, 6);
  Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  for (auto node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    int64_t insize = 0;
    ge::TensorUtils::GetSize(op_desc_ptr->GetInputDesc(0), insize);
    EXPECT_EQ(insize, 64);

    int64_t outsize = 0;
    ge::TensorUtils::GetSize(op_desc_ptr->GetOutputDesc(0), outsize);
    EXPECT_EQ(outsize, 1216);
  }
}
//TEST_F(spacesize_calculator_st, cce_nchw_nc1hwc0_success)
//{
//  GeShape in_shape({4,20,7,7});
//  GeTensorDesc in_tensor_desc(in_shape);
//  in_tensor_desc.SetOriginFormat(FORMAT_NCHW);
//  in_tensor_desc.SetFormat(FORMAT_NC1HWC0);
//  in_tensor_desc.SetDataType(DT_FLOAT16);
//  GeShape out_shape({4,3,7,7});
//  GeTensorDesc out_tensor_desc(out_shape);
//  out_tensor_desc.SetOriginFormat(FORMAT_NCHW);
//  out_tensor_desc.SetFormat(FORMAT_NC1HWC0);
//  out_tensor_desc.SetDataType(DT_FLOAT16);
//  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//  CreateTensorSizeGraph(graph, in_tensor_desc, out_tensor_desc, 6);
//  Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
//  EXPECT_EQ(fe::SUCCESS, status);
//
//  for (auto node : graph->GetDirectNode()) {
//    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
//    int64_t insize = 0;
//    ge::TensorUtils::GetSize(op_desc_ptr->GetInputDesc(0), insize);
//    EXPECT_EQ(insize, 12576);
//
//    int64_t outsize = 0;
//    ge::TensorUtils::GetSize(op_desc_ptr->GetOutputDesc(0), outsize);
//    EXPECT_EQ(outsize, 6304);
//  }
//}
//
//TEST_F(spacesize_calculator_st, cce_nhwc_nc1hwc0_success)
//{
//  GeShape in_shape({4,7,7,20});
//  GeTensorDesc in_tensor_desc(in_shape);
//  in_tensor_desc.SetOriginFormat(FORMAT_NHWC);
//  in_tensor_desc.SetFormat(FORMAT_NC1HWC0);
//  in_tensor_desc.SetDataType(DT_FLOAT16);
//  GeShape out_shape({4,7,7,3});
//  GeTensorDesc out_tensor_desc(out_shape);
//  out_tensor_desc.SetOriginFormat(FORMAT_NHWC);
//  out_tensor_desc.SetFormat(FORMAT_NC1HWC0);
//  out_tensor_desc.SetDataType(DT_FLOAT16);
//  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//  CreateTensorSizeGraph(graph, in_tensor_desc, out_tensor_desc, 6);
//  Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
//  EXPECT_EQ(fe::SUCCESS, status);
//
//  for (auto node : graph->GetDirectNode()) {
//    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
//    int64_t insize = 0;
//    ge::TensorUtils::GetSize(op_desc_ptr->GetInputDesc(0), insize);
//    EXPECT_EQ(insize, 12576);
//
//    int64_t outsize = 0;
//    ge::TensorUtils::GetSize(op_desc_ptr->GetOutputDesc(0), outsize);
//    EXPECT_EQ(outsize, 6304);
//  }
//}
//
//TEST_F(spacesize_calculator_st, cce_hwcn_nc1hwc0_success)
//{
//  GeShape in_shape({7,7,20,4});
//  GeTensorDesc in_tensor_desc(in_shape);
//  in_tensor_desc.SetOriginFormat(FORMAT_HWCN);
//  in_tensor_desc.SetFormat(FORMAT_NC1HWC0);
//  in_tensor_desc.SetDataType(DT_FLOAT16);
//  GeShape out_shape({7,7,3,4});
//  GeTensorDesc out_tensor_desc(out_shape);
//  out_tensor_desc.SetOriginFormat(FORMAT_HWCN);
//  out_tensor_desc.SetFormat(FORMAT_NC1HWC0);
//  out_tensor_desc.SetDataType(DT_FLOAT16);
//  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//  CreateTensorSizeGraph(graph, in_tensor_desc, out_tensor_desc, 6);
//  Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
//  EXPECT_EQ(fe::SUCCESS, status);
//
//  for (auto node : graph->GetDirectNode()) {
//    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
//    int64_t insize = 0;
//    ge::TensorUtils::GetSize(op_desc_ptr->GetInputDesc(0), insize);
//    EXPECT_EQ(insize, 12576);
//
//    int64_t outsize = 0;
//    ge::TensorUtils::GetSize(op_desc_ptr->GetOutputDesc(0), outsize);
//    EXPECT_EQ(outsize, 6304);
//  }
//}
//
//TEST_F(spacesize_calculator_st, cce_nchw_fraz_success)
//{
//  GeShape in_shape({4,20,7,7});
//  GeTensorDesc in_tensor_desc(in_shape);
//  in_tensor_desc.SetOriginFormat(FORMAT_NCHW);
//  in_tensor_desc.SetFormat(FORMAT_FRACTAL_Z);
//  in_tensor_desc.SetDataType(DT_FLOAT16);
//  GeShape out_shape({20,10,7,7});
//  GeTensorDesc out_tensor_desc(out_shape);
//  out_tensor_desc.SetOriginFormat(FORMAT_NCHW);
//  out_tensor_desc.SetFormat(FORMAT_FRACTAL_Z);
//  out_tensor_desc.SetDataType(DT_FLOAT16);
//  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//  CreateTensorSizeGraph(graph, in_tensor_desc, out_tensor_desc, 6);
//  Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
//  EXPECT_EQ(fe::SUCCESS, status);
//
//  for (auto node : graph->GetDirectNode()) {
//    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
//    int64_t insize = 0;
//    ge::TensorUtils::GetSize(op_desc_ptr->GetInputDesc(0), insize);
//    EXPECT_EQ(insize, 50208);
//
//    int64_t outsize = 0;
//    ge::TensorUtils::GetSize(op_desc_ptr->GetOutputDesc(0), outsize);
//    EXPECT_EQ(outsize, 50208);
//  }
//}
//
//TEST_F(spacesize_calculator_st, cce_nhwc_fraz_success)
//{
//  GeShape in_shape({4,7,7,20});
//  GeTensorDesc in_tensor_desc(in_shape);
//  in_tensor_desc.SetOriginFormat(FORMAT_NHWC);
//  in_tensor_desc.SetFormat(FORMAT_FRACTAL_Z);
//  in_tensor_desc.SetDataType(DT_FLOAT16);
//  GeShape out_shape({20,7,7,10});
//  GeTensorDesc out_tensor_desc(out_shape);
//  out_tensor_desc.SetOriginFormat(FORMAT_NHWC);
//  out_tensor_desc.SetFormat(FORMAT_FRACTAL_Z);
//  out_tensor_desc.SetDataType(DT_FLOAT16);
//  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//  CreateTensorSizeGraph(graph, in_tensor_desc, out_tensor_desc, 6);
//  Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
//  EXPECT_EQ(fe::SUCCESS, status);
//
//  for (auto node : graph->GetDirectNode()) {
//    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
//    int64_t insize = 0;
//    ge::TensorUtils::GetSize(op_desc_ptr->GetInputDesc(0), insize);
//    EXPECT_EQ(insize, 50208);
//
//    int64_t outsize = 0;
//    ge::TensorUtils::GetSize(op_desc_ptr->GetOutputDesc(0), outsize);
//    EXPECT_EQ(outsize, 50208);
//  }
//}
//
//TEST_F(spacesize_calculator_st, cce_hwcn_fraz_success)
//{
//  GeShape in_shape({7,7,20,4});
//  GeTensorDesc in_tensor_desc(in_shape);
//  in_tensor_desc.SetOriginFormat(FORMAT_HWCN);
//  in_tensor_desc.SetFormat(FORMAT_FRACTAL_Z);
//  in_tensor_desc.SetDataType(DT_FLOAT16);
//  GeShape out_shape({7,7,10,20});
//  GeTensorDesc out_tensor_desc(out_shape);
//  out_tensor_desc.SetOriginFormat(FORMAT_HWCN);
//  out_tensor_desc.SetFormat(FORMAT_FRACTAL_Z);
//  out_tensor_desc.SetDataType(DT_FLOAT16);
//  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//  CreateTensorSizeGraph(graph, in_tensor_desc, out_tensor_desc, 6);
//  Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
//  EXPECT_EQ(fe::SUCCESS, status);
//
//  for (auto node : graph->GetDirectNode()) {
//    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
//    int64_t insize = 0;
//    ge::TensorUtils::GetSize(op_desc_ptr->GetInputDesc(0), insize);
//    EXPECT_EQ(insize, 50208);
//
//    int64_t outsize = 0;
//    ge::TensorUtils::GetSize(op_desc_ptr->GetOutputDesc(0), outsize);
//    EXPECT_EQ(outsize, 50208);
//  }
//}
//
//TEST_F(spacesize_calculator_st, cce_nchw_frazc04_success)
//{
//  GeShape in_shape({4,3,7,7});
//  GeTensorDesc in_tensor_desc(in_shape);
//  in_tensor_desc.SetOriginFormat(FORMAT_NCHW);
//  in_tensor_desc.SetFormat(FORMAT_FRACTAL_Z_C04);
//  in_tensor_desc.SetDataType(DT_FLOAT16);
//  GeShape out_shape({20,4,7,7});
//  GeTensorDesc out_tensor_desc(out_shape);
//  out_tensor_desc.SetOriginFormat(FORMAT_NCHW);
//  out_tensor_desc.SetFormat(FORMAT_FRACTAL_Z_C04);
//  out_tensor_desc.SetDataType(DT_FLOAT16);
//  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//  CreateTensorSizeGraph(graph, in_tensor_desc, out_tensor_desc, 6);
//  Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
//  EXPECT_EQ(fe::SUCCESS, status);
//
//  for (auto node : graph->GetDirectNode()) {
//    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
//    int64_t insize = 0;
//    ge::TensorUtils::GetSize(op_desc_ptr->GetInputDesc(0), insize);
//    EXPECT_EQ(insize, 6688);
//
//    int64_t outsize = 0;
//    ge::TensorUtils::GetSize(op_desc_ptr->GetOutputDesc(0), outsize);
//    EXPECT_EQ(outsize, 13344);
//  }
//}
//
//TEST_F(spacesize_calculator_st, cce_nhwc_frazc04_success)
//{
//  GeShape in_shape({4,7,7,3});
//  GeTensorDesc in_tensor_desc(in_shape);
//  in_tensor_desc.SetOriginFormat(FORMAT_NHWC);
//  in_tensor_desc.SetFormat(FORMAT_FRACTAL_Z_C04);
//  in_tensor_desc.SetDataType(DT_FLOAT16);
//  GeShape out_shape({20,7,7,4});
//  GeTensorDesc out_tensor_desc(out_shape);
//  out_tensor_desc.SetOriginFormat(FORMAT_NHWC);
//  out_tensor_desc.SetFormat(FORMAT_FRACTAL_Z_C04);
//  out_tensor_desc.SetDataType(DT_FLOAT16);
//  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//  CreateTensorSizeGraph(graph, in_tensor_desc, out_tensor_desc, 6);
//  Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
//  EXPECT_EQ(fe::SUCCESS, status);
//
//  for (auto node : graph->GetDirectNode()) {
//    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
//    int64_t insize = 0;
//    ge::TensorUtils::GetSize(op_desc_ptr->GetInputDesc(0), insize);
//    EXPECT_EQ(insize, 6688);
//
//    int64_t outsize = 0;
//    ge::TensorUtils::GetSize(op_desc_ptr->GetOutputDesc(0), outsize);
//    EXPECT_EQ(outsize, 13344);
//  }
//}
//
//TEST_F(spacesize_calculator_st, cce_hwcn_frazc04_success)
//{
//  GeShape in_shape({7,7,3,4});
//  GeTensorDesc in_tensor_desc(in_shape);
//  in_tensor_desc.SetOriginFormat(FORMAT_HWCN);
//  in_tensor_desc.SetFormat(FORMAT_FRACTAL_Z_C04);
//  in_tensor_desc.SetDataType(DT_FLOAT16);
//  GeShape out_shape({7,7,4,20});
//  GeTensorDesc out_tensor_desc(out_shape);
//  out_tensor_desc.SetOriginFormat(FORMAT_HWCN);
//  out_tensor_desc.SetFormat(FORMAT_FRACTAL_Z_C04);
//  out_tensor_desc.SetDataType(DT_FLOAT16);
//  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//  CreateTensorSizeGraph(graph, in_tensor_desc, out_tensor_desc, 6);
//  Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
//  EXPECT_EQ(fe::SUCCESS, status);
//
//  for (auto node : graph->GetDirectNode()) {
//    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
//    int64_t insize = 0;
//    ge::TensorUtils::GetSize(op_desc_ptr->GetInputDesc(0), insize);
//    EXPECT_EQ(insize, 6688);
//
//    int64_t outsize = 0;
//    ge::TensorUtils::GetSize(op_desc_ptr->GetOutputDesc(0), outsize);
//    EXPECT_EQ(outsize, 13344);
//  }
//}

// TBE format NCHW,NHWC,CHWN,HWCN,NDHWC?,ND,NC1HWC0,NC1HWC0_C04,C1HWNCoC0,FRACTAL_Z,FRACTAL_Z_C04,FRACTAL_NZ
TEST_F(spacesize_calculator_st, tbe_nchw_success)
{
  GeShape in_shape({4,3,7,7});
  GeTensorDesc in_tensor_desc(in_shape);
  in_tensor_desc.SetFormat(FORMAT_NCHW);
  in_tensor_desc.SetDataType(DT_INT8);
  GeShape out_shape({6,4,7,7});
  GeTensorDesc out_tensor_desc(out_shape);
  out_tensor_desc.SetFormat(FORMAT_NCHW);
  out_tensor_desc.SetDataType(DT_UINT8);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateTensorSizeGraph(graph, in_tensor_desc, out_tensor_desc, 6);
  Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  for (auto node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    int64_t insize = 0;
    ge::TensorUtils::GetSize(op_desc_ptr->GetInputDesc(0), insize);
    EXPECT_EQ(insize, 640);

    int64_t outsize = 0;
    ge::TensorUtils::GetSize(op_desc_ptr->GetOutputDesc(0), outsize);
    EXPECT_EQ(outsize, 1216);
  }
}

TEST_F(spacesize_calculator_st, tbe_chwn_success)
{
  GeShape in_shape({9,3,5,5});
  GeTensorDesc in_tensor_desc(in_shape);
  in_tensor_desc.SetFormat(FORMAT_CHWN);
  in_tensor_desc.SetDataType(DT_INT32);
  GeShape out_shape({1,16,16,1});
  GeTensorDesc out_tensor_desc(out_shape);
  out_tensor_desc.SetFormat(FORMAT_CHWN);
  out_tensor_desc.SetDataType(DT_UINT32);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateTensorSizeGraph(graph, in_tensor_desc, out_tensor_desc, 6);
  Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  for (auto node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    int64_t insize = 0;
    ge::TensorUtils::GetSize(op_desc_ptr->GetInputDesc(0), insize);
    EXPECT_EQ(insize, 2752);

    int64_t outsize = 0;
    ge::TensorUtils::GetSize(op_desc_ptr->GetOutputDesc(0), outsize);
    EXPECT_EQ(outsize, 1056);
  }
}

TEST_F(spacesize_calculator_st, tbe_nd_success)
{
  GeShape in_shape({20,4,7,7});
  GeTensorDesc in_tensor_desc(in_shape);
  in_tensor_desc.SetFormat(FORMAT_CHWN);
  in_tensor_desc.SetDataType(DT_FLOAT16);
  GeShape out_shape({200,100,1001,100});
  GeTensorDesc out_tensor_desc(out_shape);
  out_tensor_desc.SetFormat(FORMAT_CHWN);
  out_tensor_desc.SetDataType(DT_FLOAT16);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateTensorSizeGraph(graph, in_tensor_desc, out_tensor_desc, 6);
  Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(spacesize_calculator_st, tbe_nc1hwc0_success)
{
  GeShape in_shape({4,3,7,7,16});
  GeTensorDesc in_tensor_desc(in_shape);
  in_tensor_desc.SetFormat(FORMAT_NC1HWC0);
  in_tensor_desc.SetDataType(DT_FLOAT);
  GeShape out_shape({20,4,7,7,4});
  GeTensorDesc out_tensor_desc(out_shape);
  out_tensor_desc.SetFormat(FORMAT_NC1HWC0_C04);
  out_tensor_desc.SetDataType(DT_FLOAT);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateTensorSizeGraph(graph, in_tensor_desc, out_tensor_desc, 6);
  Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  for (auto node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    int64_t insize = 0;
    ge::TensorUtils::GetSize(op_desc_ptr->GetInputDesc(0), insize);
    EXPECT_EQ(insize, 37664);

    int64_t outsize = 0;
    ge::TensorUtils::GetSize(op_desc_ptr->GetOutputDesc(0), outsize);
    EXPECT_EQ(outsize, 62752);
  }
}

TEST_F(spacesize_calculator_st, tbe_fraz_success)
{
  GeShape in_shape({4,3,16,16});
  GeTensorDesc in_tensor_desc(in_shape);
  in_tensor_desc.SetFormat(FORMAT_FRACTAL_Z);
  in_tensor_desc.SetDataType(DT_FLOAT16);
  GeShape out_shape({13,2,16,16});
  GeTensorDesc out_tensor_desc(out_shape);
  out_tensor_desc.SetFormat(FORMAT_FRACTAL_Z_C04);
  out_tensor_desc.SetDataType(DT_FLOAT16);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateTensorSizeGraph(graph, in_tensor_desc, out_tensor_desc, 6);
  Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  for (auto node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    int64_t insize = 0;
    ge::TensorUtils::GetSize(op_desc_ptr->GetInputDesc(0), insize);
    EXPECT_EQ(insize, 6176);

    int64_t outsize = 0;
    ge::TensorUtils::GetSize(op_desc_ptr->GetOutputDesc(0), outsize);
    EXPECT_EQ(outsize, 13344);
  }
}

TEST_F(spacesize_calculator_st, tbe_other_success)
{
  GeShape in_shape({4,3,16,16,5,2});
  GeTensorDesc in_tensor_desc(in_shape);
  in_tensor_desc.SetFormat(FORMAT_C1HWNCoC0);
  in_tensor_desc.SetDataType(DT_FLOAT16);
  GeShape out_shape({16,16});
  GeTensorDesc out_tensor_desc(out_shape);
  out_tensor_desc.SetFormat(FORMAT_FRACTAL_NZ);
  out_tensor_desc.SetDataType(DT_FLOAT16);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateTensorSizeGraph(graph, in_tensor_desc, out_tensor_desc, 6);
  Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
  EXPECT_EQ(fe::SUCCESS, status);

  for (auto node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    int64_t insize = 0;
    ge::TensorUtils::GetSize(op_desc_ptr->GetInputDesc(0), insize);
    EXPECT_EQ(insize, 61472);

    int64_t outsize = 0;
    ge::TensorUtils::GetSize(op_desc_ptr->GetOutputDesc(0), outsize);
    EXPECT_EQ(outsize, 544);
  }
}

//TEST_F(spacesize_calculator_st, calculate_reshape_workspace_1)
//{
//    OpDescPtr reshape_ptr = std::make_shared<OpDesc>("reshape", "Reshape");
//    OpDescPtr transpose_ptr = std::make_shared<OpDesc>("transpose", TRANSPOSE);
//    vector<int64_t> dims = {20, 10, 7, 7};
//    GeShape shape(dims);
//    GeTensorDesc in_desc1(shape);
//    in_desc1.SetOriginFormat(FORMAT_NCHW);
//    in_desc1.SetFormat(FORMAT_NC1HWC0);
//    in_desc1.SetDataType(DT_FLOAT16);
//    reshape_ptr->AddInputDesc("x", in_desc1);
//    transpose_ptr->AddInputDesc("x", in_desc1);
//
//    GeTensorDesc out_desc1(shape);
//    out_desc1.SetOriginFormat(FORMAT_NCHW);
//    out_desc1.SetFormat(FORMAT_NC1HWC0);
//    out_desc1.SetDataType(DT_FLOAT16);
//    reshape_ptr->AddOutputDesc("y", out_desc1);
//    transpose_ptr->AddOutputDesc("y", out_desc1);
//
//    ge::AttrUtils::SetInt(reshape_ptr, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
//    ge::AttrUtils::SetInt(transpose_ptr, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
//
//    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//    NodePtr reshape_node = graph->AddNode(reshape_ptr);
//    NodePtr transpose_node = graph->AddNode(transpose_ptr);
//    GraphUtils::AddEdge(reshape_node->GetOutDataAnchor(0), transpose_node->GetInDataAnchor(0));

//    std::shared_ptr<SpaceSizeCalculator> space_size_calculator_ptr = std::make_shared<SpaceSizeCalculator>(op_store_adapter_manager_);
//    Status ret = space_size_calculator_ptr->CalculateRunningParams(*(graph.get()));
//    EXPECT_EQ(fe::SUCCESS, ret);
//    for (auto node : graph->GetDirectNode()) {
//        string op_type = node->GetType();
//        if (op_type == OP_TYPE_PLACE_HOLDER ||
//            op_type == OP_TYPE_END) {
//            continue;
//        }
//        ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
//        if (op_desc_ptr->GetType() == "Reshape") {
//            vector<int64_t> vec1 = {62816};
//            EXPECT_EQ(op_desc_ptr->GetWorkspaceBytes(), vec1);
//        }
//        if (op_desc_ptr->GetType() == TRANSPOSE) {
//            vector<int64_t> vec2 = {0};
//            EXPECT_EQ(op_desc_ptr->GetWorkspaceBytes(), vec2);
//        }
//    }
//}
//
//TEST_F(spacesize_calculator_st, calculate_reshape_workspace_2)
//{
//    OpDescPtr reshape_ptr = std::make_shared<OpDesc>("reshape", "Reshape");
//    OpDescPtr transpose_ptr = std::make_shared<OpDesc>("transpose", TRANSPOSE);
//    vector<int64_t> dims = {1, 2, 3, 4};
//    GeShape shape(dims);
//    GeTensorDesc in_desc1(shape);
//    in_desc1.SetOriginFormat(FORMAT_NCHW);
//    in_desc1.SetFormat(FORMAT_NCHW);
//    in_desc1.SetDataType(DT_FLOAT16);
//    reshape_ptr->AddInputDesc("x", in_desc1);
//    transpose_ptr->AddInputDesc("x", in_desc1);
//
//    GeTensorDesc out_desc1(shape);
//    out_desc1.SetOriginFormat(FORMAT_NCHW);
//    out_desc1.SetFormat(FORMAT_NCHW);
//    out_desc1.SetDataType(DT_FLOAT16);
//    reshape_ptr->AddOutputDesc("y", out_desc1);
//    transpose_ptr->AddOutputDesc("y", out_desc1);
//
//    ge::AttrUtils::SetInt(reshape_ptr, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
//    ge::AttrUtils::SetInt(transpose_ptr, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
//
//    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//    NodePtr reshape_node = graph->AddNode(reshape_ptr);
//    NodePtr transpose_node = graph->AddNode(transpose_ptr);
//    GraphUtils::AddEdge(reshape_node->GetOutDataAnchor(0), transpose_node->GetInDataAnchor(0));

//    std::shared_ptr<SpaceSizeCalculator> space_size_calculator_ptr = std::make_shared<SpaceSizeCalculator>(op_store_adapter_manager_);
//    Status ret = space_size_calculator_ptr->CalculateRunningParams(*(graph.get()));
//    EXPECT_EQ(fe::SUCCESS, ret);
//    for (auto node : graph->GetDirectNode()) {
//        string op_type = node->GetType();
//        if (op_type == OP_TYPE_PLACE_HOLDER ||
//            op_type == OP_TYPE_END) {
//            continue;
//        }
//        ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
//        if (op_desc_ptr->GetType() == "Reshape") {
//            vector<int64_t> vec1 = {0};
//            EXPECT_EQ(op_desc_ptr->GetWorkspaceBytes(), vec1);
//        }
//        if (op_desc_ptr->GetType() == TRANSPOSE) {
//            vector<int64_t> vec2 = {192};
//            EXPECT_EQ(op_desc_ptr->GetWorkspaceBytes(), vec2);
//        }
//    }
//}
