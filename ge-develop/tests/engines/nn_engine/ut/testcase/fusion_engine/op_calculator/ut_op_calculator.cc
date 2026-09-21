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


static const std::vector<ge::DataType> TENSOR_SIZE_DTYPE_VEC {
        ge::DT_BOOL,             // bool type
        ge::DT_INT8,             // int8 type
        ge::DT_UINT8,            // uint8 type
        ge::DT_INT16,            // int16 type
        ge::DT_UINT16,           // uint16 type
        ge::DT_INT32,            //
        ge::DT_UINT32,           // unsigned int32
        ge::DT_INT64,            // int64 type
        ge::DT_UINT64,           // unsigned int64
        ge::DT_FLOAT16,          // fp16 type
        ge::DT_FLOAT,            // float type
        ge::DT_DOUBLE           // double type
};

class spacesize_calculator_ut : public testing::Test
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

    static void CreateTensorSizeGraphWithAllDataType(ComputeGraphPtr graph,
                    GeTensorDesc &in_tensor_desc, GeTensorDesc &out_tensor_desc, int impl_type) {
      OpDescPtr op_desc = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
      for (uint32_t i = 0; i < TENSOR_SIZE_DTYPE_VEC.size(); i++) {
          GeTensorDesc in_desc = in_tensor_desc;
          in_desc.SetDataType(TENSOR_SIZE_DTYPE_VEC[i]);
          std::string input = "input" +std::to_string(i);
          op_desc->AddInputDesc(input, in_desc);

          GeTensorDesc out_desc = out_tensor_desc;
          out_desc.SetDataType(TENSOR_SIZE_DTYPE_VEC[i]);
          std::string output = "output" + std::to_string(i);
          op_desc->AddOutputDesc(output, out_desc);
      }

      ge::AttrUtils::SetInt(op_desc, FE_IMPLY_TYPE, impl_type);
      NodePtr node = graph->AddNode(op_desc);
    }

    static void VerifyTensorSize(ge::OpDescPtr op_desc_ptr,
                               std::vector<uint32_t> &input_size_vec, std::vector<uint32_t> &output_size_vec) {
      if (op_desc_ptr->GetInputsSize() == 0 || op_desc_ptr->GetOutputsSize() == 0
          || op_desc_ptr->GetInputsSize() != input_size_vec.size() || op_desc_ptr->GetOutputsSize()!= output_size_vec.size()) {
          EXPECT_EQ(true, false);
          return;
      }
      int64_t tenosr_size = 0;
      for (uint32_t i = 0; i < op_desc_ptr->GetInputsSize(); i++) {
          ge::TensorUtils::GetSize(op_desc_ptr->GetInputDesc(i), tenosr_size);
          EXPECT_EQ(tenosr_size, input_size_vec[i]);
      }
      for (uint32_t i = 0; i < op_desc_ptr->GetOutputsSize(); i++) {
          ge::TensorUtils::GetSize(op_desc_ptr->GetOutputDesc(i), tenosr_size);
          EXPECT_EQ(tenosr_size, output_size_vec[i]);
      }
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

//TEST_F(spacesize_calculator_ut, cce_nchw_success)
//{
//    GeShape in_shape({1,7,7,1});
//    GeTensorDesc in_tensor_desc(in_shape);
//    in_tensor_desc.SetOriginFormat(FORMAT_NCHW);
//    in_tensor_desc.SetFormat(FORMAT_NCHW);
//    in_tensor_desc.SetDataType(DT_FLOAT16);
//    GeShape out_shape({4,16,7,7});
//    GeTensorDesc out_tensor_desc(out_shape);
//    out_tensor_desc.SetOriginFormat(FORMAT_NCHW);
//    out_tensor_desc.SetFormat(FORMAT_NCHW);
//    out_tensor_desc.SetDataType(DT_FLOAT);
//    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//    CreateTensorSizeGraphWithAllDataType(graph, in_tensor_desc, out_tensor_desc, 4);
//    Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
//    EXPECT_EQ(fe::SUCCESS, status);
//
//    for (auto node : graph->GetDirectNode()) {
//      ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
//      std::vector<uint32_t> input_size_vec = {96,96,96,160,160,256,256,448,448,160,256,448};
//      std::vector<uint32_t> output_size_vec = {3168,3168,3168,6304,6304,12576,12576,25120,25120,6304,12576,25120};
//      VerifyTensorSize(op_desc_ptr, input_size_vec, output_size_vec);
//    }
//}
//
//TEST_F(spacesize_calculator_ut, cce_nhwc_success)
//{
//    GeShape in_shape({1,7,7,1});
//    GeTensorDesc in_tensor_desc(in_shape);
//    in_tensor_desc.SetOriginFormat(FORMAT_NCHW);
//    in_tensor_desc.SetFormat(FORMAT_NHWC);
//    in_tensor_desc.SetDataType(DT_FLOAT16);
//    GeShape out_shape({4,16,7,7});
//    GeTensorDesc out_tensor_desc(out_shape);
//    out_tensor_desc.SetOriginFormat(FORMAT_NCHW);
//    out_tensor_desc.SetFormat(FORMAT_NHWC);
//    out_tensor_desc.SetDataType(DT_FLOAT);
//    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//    CreateTensorSizeGraphWithAllDataType(graph, in_tensor_desc, out_tensor_desc, 4);
//    Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
//    EXPECT_EQ(fe::SUCCESS, status);
//
//    for (auto node : graph->GetDirectNode()) {
//      ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
//      std::vector<uint32_t> input_size_vec = {96,96,96,160,160,256,256,448,448,160,256,448};
//      std::vector<uint32_t> output_size_vec = {3168,3168,3168,6304,6304,12576,12576,25120,25120,6304,12576,25120};
//      VerifyTensorSize(op_desc_ptr, input_size_vec, output_size_vec);
//    }
//}
//
//TEST_F(spacesize_calculator_ut, cce_hwcn_success)
//{
//    GeShape in_shape({1,7,7,1});
//    GeTensorDesc in_tensor_desc(in_shape);
//    in_tensor_desc.SetOriginFormat(FORMAT_NCHW);
//    in_tensor_desc.SetFormat(FORMAT_HWCN);
//    in_tensor_desc.SetDataType(DT_DOUBLE);
//    GeShape out_shape({4,3,7,7});
//    GeTensorDesc out_tensor_desc(out_shape);
//    out_tensor_desc.SetOriginFormat(FORMAT_NCHW);
//    out_tensor_desc.SetFormat(FORMAT_HWCN);
//    out_tensor_desc.SetDataType(DT_BOOL);
//    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//    CreateTensorSizeGraphWithAllDataType(graph, in_tensor_desc, out_tensor_desc, 4);
//    Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
//    EXPECT_EQ(fe::SUCCESS, status);
//
//    for (auto node : graph->GetDirectNode()) {
//      ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
//      std::vector<uint32_t> input_size_vec = {96,96,96,160,160,256,256,448,448,160,256,448};
//      std::vector<uint32_t> output_size_vec = {640,640,640,1216,1216,2400,2400,4736,4736,1216,2400,4736};
//      VerifyTensorSize(op_desc_ptr, input_size_vec, output_size_vec);
//    }
//}
//
//TEST_F(spacesize_calculator_ut, cce_nd_success)
//{
//    std::vector<int64_t> dims = {};
//    GeShape in_shape(dims);
//    GeTensorDesc in_tensor_desc(in_shape);
//    in_tensor_desc.SetOriginFormat(FORMAT_ND);
//    in_tensor_desc.SetFormat(FORMAT_ND);
//    in_tensor_desc.SetDataType(DT_INT32);
//    GeShape out_shape({4,3,2,2});
//    GeTensorDesc out_tensor_desc(out_shape);
//    out_tensor_desc.SetOriginFormat(FORMAT_ND);
//    out_tensor_desc.SetFormat(FORMAT_ND);
//    out_tensor_desc.SetDataType(DT_INT16);
//    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//    CreateTensorSizeGraphWithAllDataType(graph, in_tensor_desc, out_tensor_desc, 4);
//    Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
//    EXPECT_EQ(fe::SUCCESS, status);
//
//    for (auto node : graph->GetDirectNode()) {
//        ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
//        std::vector<uint32_t> input_size_vec = {64,64,64,64,64,64,64,64,64,64,64,64};
//        std::vector<uint32_t> output_size_vec = {96,96,96,128,128,224,224,416,416,128,224,416};
//        VerifyTensorSize(op_desc_ptr, input_size_vec, output_size_vec);
//    }
//}
//
//TEST_F(spacesize_calculator_ut, cce_nchw_nc1hwc0_success)
//{
//    GeShape in_shape({4,20,7,7});
//    GeTensorDesc in_tensor_desc(in_shape);
//    in_tensor_desc.SetOriginFormat(FORMAT_NCHW);
//    in_tensor_desc.SetFormat(FORMAT_NC1HWC0);
//    in_tensor_desc.SetDataType(DT_FLOAT16);
//    GeShape out_shape({4,3,7,7});
//    GeTensorDesc out_tensor_desc(out_shape);
//    out_tensor_desc.SetOriginFormat(FORMAT_NCHW);
//    out_tensor_desc.SetFormat(FORMAT_NC1HWC0);
//    out_tensor_desc.SetDataType(DT_FLOAT16);
//    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//    CreateTensorSizeGraphWithAllDataType(graph, in_tensor_desc, out_tensor_desc, 4);
//    Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
//    EXPECT_EQ(fe::SUCCESS, status);
//
//    for (auto node : graph->GetDirectNode()) {
//      ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
//      std::vector<uint32_t> input_size_vec = {6304,6304,6304,12576,12576,25120,25120,50208,50208,12576,25120,50208};
//      std::vector<uint32_t> output_size_vec = {3168,6304,6304,6304,6304,12576,12576,25120,25120,6304,12576,25120};
//      VerifyTensorSize(op_desc_ptr, input_size_vec, output_size_vec);
//    }
//}
//
//TEST_F(spacesize_calculator_ut, cce_nhwc_nc1hwc0_success)
//{
//    GeShape in_shape({4,7,7,20});
//    GeTensorDesc in_tensor_desc(in_shape);
//    in_tensor_desc.SetOriginFormat(FORMAT_NHWC);
//    in_tensor_desc.SetFormat(FORMAT_NC1HWC0);
//    in_tensor_desc.SetDataType(DT_FLOAT16);
//    GeShape out_shape({4,7,7,3});
//    GeTensorDesc out_tensor_desc(out_shape);
//    out_tensor_desc.SetOriginFormat(FORMAT_NHWC);
//    out_tensor_desc.SetFormat(FORMAT_NC1HWC0);
//    out_tensor_desc.SetDataType(DT_FLOAT16);
//    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//    CreateTensorSizeGraphWithAllDataType(graph, in_tensor_desc, out_tensor_desc, 4);
//    Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
//    EXPECT_EQ(fe::SUCCESS, status);
//
//    for (auto node : graph->GetDirectNode()) {
//      ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
//      std::vector<uint32_t> input_size_vec = {6304,6304,6304,12576,12576,25120,25120,50208,50208,12576,25120,50208};
//      std::vector<uint32_t> output_size_vec = {3168,6304,6304,6304,6304,12576,12576,25120,25120,6304,12576,25120};
//      VerifyTensorSize(op_desc_ptr, input_size_vec, output_size_vec);
//    }
//}
//
//TEST_F(spacesize_calculator_ut, cce_hwcn_nc1hwc0_success)
//{
//    GeShape in_shape({7,7,20,4});
//    GeTensorDesc in_tensor_desc(in_shape);
//    in_tensor_desc.SetOriginFormat(FORMAT_HWCN);
//    in_tensor_desc.SetFormat(FORMAT_NC1HWC0);
//    in_tensor_desc.SetDataType(DT_FLOAT16);
//    GeShape out_shape({7,7,3,4});
//    GeTensorDesc out_tensor_desc(out_shape);
//    out_tensor_desc.SetOriginFormat(FORMAT_HWCN);
//    out_tensor_desc.SetFormat(FORMAT_NC1HWC0);
//    out_tensor_desc.SetDataType(DT_FLOAT16);
//    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//    CreateTensorSizeGraphWithAllDataType(graph, in_tensor_desc, out_tensor_desc, 4);
//    Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
//    EXPECT_EQ(fe::SUCCESS, status);
//
//    for (auto node : graph->GetDirectNode()) {
//      ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
//      std::vector<uint32_t> input_size_vec = {6304,6304,6304,12576,12576,25120,25120,50208,50208,12576,25120,50208};
//      std::vector<uint32_t> output_size_vec = {3168,6304,6304,6304,6304,12576,12576,25120,25120,6304,12576,25120};
//      VerifyTensorSize(op_desc_ptr, input_size_vec, output_size_vec);
//    }
//}
//
//TEST_F(spacesize_calculator_ut, cce_nchw_fraz_success)
//{
//    GeShape in_shape({4,20,7,7});
//    GeTensorDesc in_tensor_desc(in_shape);
//    in_tensor_desc.SetOriginFormat(FORMAT_NCHW);
//    in_tensor_desc.SetFormat(FORMAT_FRACTAL_Z);
//    in_tensor_desc.SetDataType(DT_FLOAT16);
//    GeShape out_shape({20,10,7,7});
//    GeTensorDesc out_tensor_desc(out_shape);
//    out_tensor_desc.SetOriginFormat(FORMAT_NCHW);
//    out_tensor_desc.SetFormat(FORMAT_FRACTAL_Z);
//    out_tensor_desc.SetDataType(DT_FLOAT16);
//    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//    CreateTensorSizeGraphWithAllDataType(graph, in_tensor_desc, out_tensor_desc, 4);
//    Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
//    EXPECT_EQ(fe::SUCCESS, status);
//
//    for (auto node : graph->GetDirectNode()) {
//      ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
//      std::vector<uint32_t> input_size_vec = {25120,25120,25120,50208,50208,100384,100384,200736,200736,50208,100384,200736};
//      std::vector<uint32_t> output_size_vec = {25120,50208,50208,50208,50208,100384,100384,200736,200736,50208,100384,200736};
//      VerifyTensorSize(op_desc_ptr, input_size_vec, output_size_vec);
//    }
//}
//
//TEST_F(spacesize_calculator_ut, cce_nhwc_fraz_success)
//{
//    GeShape in_shape({4,7,7,20});
//    GeTensorDesc in_tensor_desc(in_shape);
//    in_tensor_desc.SetOriginFormat(FORMAT_NHWC);
//    in_tensor_desc.SetFormat(FORMAT_FRACTAL_Z);
//    in_tensor_desc.SetDataType(DT_FLOAT16);
//    GeShape out_shape({20,7,7,10});
//    GeTensorDesc out_tensor_desc(out_shape);
//    out_tensor_desc.SetOriginFormat(FORMAT_NHWC);
//    out_tensor_desc.SetFormat(FORMAT_FRACTAL_Z);
//    out_tensor_desc.SetDataType(DT_FLOAT16);
//    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//    CreateTensorSizeGraphWithAllDataType(graph, in_tensor_desc, out_tensor_desc, 4);
//    Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
//    EXPECT_EQ(fe::SUCCESS, status);
//
//    for (auto node : graph->GetDirectNode()) {
//      ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
//      std::vector<uint32_t> input_size_vec = {25120,25120,25120,50208,50208,100384,100384,200736,200736,50208,100384,200736};
//      std::vector<uint32_t> output_size_vec = {25120,50208,50208,50208,50208,100384,100384,200736,200736,50208,100384,200736};
//      VerifyTensorSize(op_desc_ptr, input_size_vec, output_size_vec);
//    }
//}
//
//TEST_F(spacesize_calculator_ut, cce_hwcn_fraz_success)
//{
//    GeShape in_shape({7,7,20,4});
//    GeTensorDesc in_tensor_desc(in_shape);
//    in_tensor_desc.SetOriginFormat(FORMAT_HWCN);
//    in_tensor_desc.SetFormat(FORMAT_FRACTAL_Z);
//    in_tensor_desc.SetDataType(DT_FLOAT16);
//    GeShape out_shape({7,7,10,20});
//    GeTensorDesc out_tensor_desc(out_shape);
//    out_tensor_desc.SetOriginFormat(FORMAT_HWCN);
//    out_tensor_desc.SetFormat(FORMAT_FRACTAL_Z);
//    out_tensor_desc.SetDataType(DT_FLOAT16);
//    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//    CreateTensorSizeGraphWithAllDataType(graph, in_tensor_desc, out_tensor_desc, 4);
//    Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
//    EXPECT_EQ(fe::SUCCESS, status);
//
//    for (auto node : graph->GetDirectNode()) {
//      ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
//      std::vector<uint32_t> input_size_vec = {25120,25120,25120,50208,50208,100384,100384,200736,200736,50208,100384,200736};
//
//
//      std::vector<uint32_t> output_size_vec = {25120,50208,50208,50208,50208,100384,100384,200736,200736,50208,100384,200736};
//      VerifyTensorSize(op_desc_ptr, input_size_vec, output_size_vec);
//    }
//}
//
//TEST_F(spacesize_calculator_ut, cce_nchw_frazc04_success)
//{
//    GeShape in_shape({4,3,7,7});
//    GeTensorDesc in_tensor_desc(in_shape);
//    in_tensor_desc.SetOriginFormat(FORMAT_NCHW);
//    in_tensor_desc.SetFormat(FORMAT_FRACTAL_Z_C04);
//    in_tensor_desc.SetDataType(DT_FLOAT16);
//    GeShape out_shape({20,4,7,7});
//    GeTensorDesc out_tensor_desc(out_shape);
//    out_tensor_desc.SetOriginFormat(FORMAT_NCHW);
//    out_tensor_desc.SetFormat(FORMAT_FRACTAL_Z_C04);
//    out_tensor_desc.SetDataType(DT_FLOAT16);
//    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//    CreateTensorSizeGraphWithAllDataType(graph, in_tensor_desc, out_tensor_desc, 4);
//    Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
//    EXPECT_EQ(fe::SUCCESS, status);
//
//    for (auto node : graph->GetDirectNode()) {
//      ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
//      std::vector<uint32_t> input_size_vec = {3360,3360,3360,6688,6688,13344,13344,26656,26656,6688,13344,26656};
//      std::vector<uint32_t> output_size_vec = {6688,6688,6688,13344,13344,26656,26656,53280,53280,13344,26656,53280};
//      VerifyTensorSize(op_desc_ptr, input_size_vec, output_size_vec);
//    }
//}
//
//TEST_F(spacesize_calculator_ut, cce_nhwc_frazc04_success)
//{
//    GeShape in_shape({4,7,7,3});
//    GeTensorDesc in_tensor_desc(in_shape);
//    in_tensor_desc.SetOriginFormat(FORMAT_NHWC);
//    in_tensor_desc.SetFormat(FORMAT_FRACTAL_Z_C04);
//    in_tensor_desc.SetDataType(DT_FLOAT16);
//    GeShape out_shape({20,7,7,4});
//    GeTensorDesc out_tensor_desc(out_shape);
//    out_tensor_desc.SetOriginFormat(FORMAT_NHWC);
//    out_tensor_desc.SetFormat(FORMAT_FRACTAL_Z_C04);
//    out_tensor_desc.SetDataType(DT_FLOAT16);
//    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//    CreateTensorSizeGraphWithAllDataType(graph, in_tensor_desc, out_tensor_desc, 4);
//    Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
//    EXPECT_EQ(fe::SUCCESS, status);
//
//    for (auto node : graph->GetDirectNode()) {
//      ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
//      std::vector<uint32_t> input_size_vec = {3360,3360,3360,6688,6688,13344,13344,26656,26656,6688,13344,26656};
//      std::vector<uint32_t> output_size_vec = {6688,6688,6688,13344,13344,26656,26656,53280,53280,13344,26656,53280};
//      VerifyTensorSize(op_desc_ptr, input_size_vec, output_size_vec);
//    }
//}
//
//TEST_F(spacesize_calculator_ut, cce_hwcn_frazc04_success)
//{
//    GeShape in_shape({7,7,3,4});
//    GeTensorDesc in_tensor_desc(in_shape);
//    in_tensor_desc.SetOriginFormat(FORMAT_HWCN);
//    in_tensor_desc.SetFormat(FORMAT_FRACTAL_Z_C04);
//    in_tensor_desc.SetDataType(DT_FLOAT16);
//    GeShape out_shape({7,7,4,20});
//    GeTensorDesc out_tensor_desc(out_shape);
//    out_tensor_desc.SetOriginFormat(FORMAT_HWCN);
//    out_tensor_desc.SetFormat(FORMAT_FRACTAL_Z_C04);
//    out_tensor_desc.SetDataType(DT_FLOAT16);
//    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//    CreateTensorSizeGraphWithAllDataType(graph, in_tensor_desc, out_tensor_desc, 4);
//    Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
//    EXPECT_EQ(fe::SUCCESS, status);
//
//    for (auto node : graph->GetDirectNode()) {
//      ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
//      std::vector<uint32_t> input_size_vec = {3360,3360,3360,6688,6688,13344,13344,26656,26656,6688,13344,26656};
//      std::vector<uint32_t> output_size_vec = {6688,6688,6688,13344,13344,26656,26656,53280,53280,13344,26656,53280};
//      VerifyTensorSize(op_desc_ptr, input_size_vec, output_size_vec);
//    }
//}

// TBE format NCHW,NHWC,CHWN,HWCN,NDHWC?,ND,NC1HWC0,NC1HWC0_C04,C1HWNCoC0,FRACTAL_Z,FRACTAL_Z_C04,FRACTAL_NZ
TEST_F(spacesize_calculator_ut, tbe_nchw_success)
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
    CreateTensorSizeGraphWithAllDataType(graph, in_tensor_desc, out_tensor_desc, 6);
    Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
    EXPECT_EQ(fe::SUCCESS, status);

    for (auto node : graph->GetDirectNode()) {
      ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
      std::vector<uint32_t> input_size_vec = {640,640,640,1216,1216,2400,2400,4736,4736,1216,2400,4736};
      std::vector<uint32_t> output_size_vec = {1216,1216,1216,2400,2400,4736,4736,9440,9440,2400,4736,9440};
      VerifyTensorSize(op_desc_ptr, input_size_vec, output_size_vec);
    }
}

TEST_F(spacesize_calculator_ut, tbe_chwn_success)
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
    CreateTensorSizeGraphWithAllDataType(graph, in_tensor_desc, out_tensor_desc, 6);
    Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
    EXPECT_EQ(fe::SUCCESS, status);

    for (auto node : graph->GetDirectNode()) {
      ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
      std::vector<uint32_t> input_size_vec = {736,736,736,1408,1408,2752,2752,5440,5440,1408,2752,5440};
      std::vector<uint32_t> output_size_vec = {288,288,288,544,544,1056,1056,2080,2080,544,1056,2080};
      VerifyTensorSize(op_desc_ptr, input_size_vec, output_size_vec);
    }
}

TEST_F(spacesize_calculator_ut, tbe_nd_success)
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

TEST_F(spacesize_calculator_ut, tbe_nc1hwc0_success)
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
    CreateTensorSizeGraphWithAllDataType(graph, in_tensor_desc, out_tensor_desc, 6);
    Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
    EXPECT_EQ(fe::SUCCESS, status);

    for (auto node : graph->GetDirectNode()) {
      ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
      std::vector<uint32_t> input_size_vec = {9440,9440,9440,18848,18848,37664,37664,75296,75296,18848,37664,75296};
      std::vector<uint32_t> output_size_vec = {15712,15712,15712,31392,31392,62752,62752,125472,125472,31392,62752,125472};
      VerifyTensorSize(op_desc_ptr, input_size_vec, output_size_vec);
    }
}

TEST_F(spacesize_calculator_ut, tbe_fraz_success)
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
    CreateTensorSizeGraphWithAllDataType(graph, in_tensor_desc, out_tensor_desc, 6);
    Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
    EXPECT_EQ(fe::SUCCESS, status);

    for (auto node : graph->GetDirectNode()) {
      ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
      std::vector<uint32_t> input_size_vec = {3104,3104,3104,6176,6176,12320,12320,24608,24608,6176,12320,24608};
      std::vector<uint32_t> output_size_vec = {6688,6688,6688,13344,13344,26656,26656,53280,53280,13344,26656,53280};
      VerifyTensorSize(op_desc_ptr, input_size_vec, output_size_vec);
    }
}

TEST_F(spacesize_calculator_ut, tbe_other_success)
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
    CreateTensorSizeGraphWithAllDataType(graph, in_tensor_desc, out_tensor_desc, 6);
    Status status = space_size_calculator_ptr_->CalculateRunningParams(*(graph.get()));
    EXPECT_EQ(fe::SUCCESS, status);

    for (auto node : graph->GetDirectNode()) {
      ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
      std::vector<uint32_t> input_size_vec = {30752,30752,30752,61472,61472,122912,122912,245792,245792,61472,122912,245792};
      std::vector<uint32_t> output_size_vec = {288,288,288,544,544,1056,1056,2080,2080,544,1056,2080};
      VerifyTensorSize(op_desc_ptr, input_size_vec, output_size_vec);
    }
}

//TEST_F(spacesize_calculator_ut, calculate_reshape_workspace_1)
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

//TEST_F(spacesize_calculator_ut, calculate_reshape_workspace_2)
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
