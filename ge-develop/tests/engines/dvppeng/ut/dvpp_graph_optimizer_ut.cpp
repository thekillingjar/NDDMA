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

#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/debug/ge_attr_define.h"

#define private public
#define protected public
#include "dvpp_graph_optimizer.h"
#include "dvpp_ops_checker.h"
#include "dvpp_optimizer.h"
#include "dvpp_ops_kernel_info_store.h"
#undef protected
#undef private

#include "util/dvpp_constexpr.h"

TEST(DvppGraphOptimizer, AllTest_910B)
{
  dvpp::DvppGraphOptimizer dvppGraphOptimizer;

  // Initialize
  std::map<std::string, std::string> options;
  options[ge::SOC_VERSION] = dvpp::kSocVersionAscend910B1;
  ge::Status status = dvppGraphOptimizer.Initialize(options, nullptr);
  ASSERT_EQ(status, ge::SUCCESS);

  // OptimizeOriginalGraph
  // set op
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

  // set ATTR_NAME_PERFORMANCE_PRIOR
  ge::AttrUtils::SetBool(op_desc_ptr, dvpp::ATTR_NAME_PERFORMANCE_PRIOR, true);

  // set graph
  std::shared_ptr<ge::ComputeGraph> graphPtr = std::make_shared<ge::ComputeGraph>("test_graph");
  graphPtr->AddNode(op_desc_ptr);

  // OptimizeOriginalGraph
  status = dvppGraphOptimizer.OptimizeOriginalGraph(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // OptimizeOriginalGraphJudgeInsert
  status = dvppGraphOptimizer.OptimizeOriginalGraphJudgeInsert(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // OptimizeFusedGraph
  status = dvppGraphOptimizer.OptimizeFusedGraph(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // OptimizeWholeGraph
  status = dvppGraphOptimizer.OptimizeWholeGraph(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // GetAttributes
  ge::GraphOptimizerAttribute attrs;
  status = dvppGraphOptimizer.GetAttributes(attrs);
  ASSERT_EQ(status, ge::SUCCESS);

  // Finalize
  status = dvppGraphOptimizer.Finalize();
  ASSERT_EQ(status, ge::SUCCESS);
}

TEST(DvppGraphOptimizer, AllTest_910B_ResizeBilinearV2)
{
  dvpp::DvppGraphOptimizer dvppGraphOptimizer;

  // Initialize
  std::map<std::string, std::string> options;
  options[ge::SOC_VERSION] = dvpp::kSocVersionAscend910B1;
  ge::Status status = dvppGraphOptimizer.Initialize(options, nullptr);
  ASSERT_EQ(status, ge::SUCCESS);

  // OptimizeOriginalGraph
  // set op
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

  // set ATTR_NAME_PERFORMANCE_PRIOR
  ge::AttrUtils::SetBool(op_desc_ptr, dvpp::ATTR_NAME_PERFORMANCE_PRIOR, true);

  // set graph
  std::shared_ptr<ge::ComputeGraph> graphPtr = std::make_shared<ge::ComputeGraph>("test_graph");
  graphPtr->AddNode(op_desc_ptr);
  ge::AttrUtils::SetBool(graphPtr, "_single_op_scene", true);

  // OptimizeGraphPrepare
  status = dvppGraphOptimizer.OptimizeGraphPrepare(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

 // OptimizeOriginalGraph
  status = dvppGraphOptimizer.OptimizeOriginalGraph(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // OptimizeOriginalGraphJudgeInsert
  status = dvppGraphOptimizer.OptimizeOriginalGraphJudgeInsert(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // OptimizeFusedGraph
  status = dvppGraphOptimizer.OptimizeFusedGraph(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // OptimizeWholeGraph
  status = dvppGraphOptimizer.OptimizeWholeGraph(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // GetAttributes
  ge::GraphOptimizerAttribute attrs;
  status = dvppGraphOptimizer.GetAttributes(attrs);
  ASSERT_EQ(status, ge::SUCCESS);

  // Finalize
  status = dvppGraphOptimizer.Finalize();
  ASSERT_EQ(status, ge::SUCCESS);
}

TEST(DvppGraphOptimizer, AllTest_910B_Resize)
{
  dvpp::DvppGraphOptimizer dvppGraphOptimizer;

  // Initialize
  std::map<std::string, std::string> options;
  options[ge::SOC_VERSION] = dvpp::kSocVersionAscend910B1;
  ge::Status status = dvppGraphOptimizer.Initialize(options, nullptr);
  ASSERT_EQ(status, ge::SUCCESS);

  // OptimizeOriginalGraph
  // set op
  ge::OpDescPtr op_desc_ptr = std::make_shared<ge::OpDesc>("Resize", "Resize");
  ge::GeTensorDesc tensorX(ge::GeShape(std::vector<int64_t>{1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_UINT8);
  op_desc_ptr->AddInputDesc("x", tensorX);
  ge::GeTensorDesc tensorRoi(ge::GeShape(std::vector<int64_t>{1, 1, 1, 1}), ge::FORMAT_ND, ge::DT_FLOAT);
  op_desc_ptr->AddInputDesc("roi", tensorRoi);
  ge::GeTensorDesc tensorScales(ge::GeShape(std::vector<int64_t>{1, 1, 1, 1}), ge::FORMAT_ND, ge::DT_FLOAT);
  op_desc_ptr->AddInputDesc("scales", tensorScales);
  ge::GeTensorDesc tensorSizes(ge::GeShape(std::vector<int64_t>{1, 1280, 720, 3}), ge::FORMAT_ND, ge::DT_INT32);
  op_desc_ptr->AddInputDesc("sizes", tensorSizes);
  ge::GeTensorDesc tensorY(ge::GeShape(std::vector<int64_t>{1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_UINT8);
  op_desc_ptr->AddOutputDesc("y", tensorY);
  ge::AttrUtils::SetBool(op_desc_ptr, "align_corners", false);
  ge::AttrUtils::SetBool(op_desc_ptr, "half_pixel_centers", false);
  ge::AttrUtils::SetStr(op_desc_ptr, "dtype", "DT_UINT8");

  // set ATTR_NAME_PERFORMANCE_PRIOR
  ge::AttrUtils::SetBool(op_desc_ptr, dvpp::ATTR_NAME_PERFORMANCE_PRIOR, true);
  ge::OpDescUtilsEx::SetType(op_desc_ptr, "Resize");

  // set graph
  std::shared_ptr<ge::ComputeGraph> graphPtr = std::make_shared<ge::ComputeGraph>("test_graph");
  graphPtr->AddNode(op_desc_ptr);
  ge::AttrUtils::SetBool(graphPtr, "_single_op_scene", true);

  // OptimizeGraphPrepare
  status = dvppGraphOptimizer.OptimizeGraphPrepare(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

 // OptimizeOriginalGraph
  status = dvppGraphOptimizer.OptimizeOriginalGraph(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // OptimizeOriginalGraphJudgeInsert
  status = dvppGraphOptimizer.OptimizeOriginalGraphJudgeInsert(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // OptimizeFusedGraph
  status = dvppGraphOptimizer.OptimizeFusedGraph(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // OptimizeWholeGraph
  status = dvppGraphOptimizer.OptimizeWholeGraph(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // GetAttributes
  ge::GraphOptimizerAttribute attrs;
  status = dvppGraphOptimizer.GetAttributes(attrs);
  ASSERT_EQ(status, ge::SUCCESS);

  // Finalize
  status = dvppGraphOptimizer.Finalize();
  ASSERT_EQ(status, ge::SUCCESS);
}

TEST(DvppGraphOptimizer, Test_910B_DecodeJpeg_01)
{
  dvpp::DvppGraphOptimizer dvppGraphOptimizer;

  // Initialize
  std::map<std::string, std::string> options;
  options[ge::SOC_VERSION] = dvpp::kSocVersionAscend910B1;
  ge::Status status = dvppGraphOptimizer.Initialize(options, nullptr);
  ASSERT_EQ(status, ge::SUCCESS);

  ge::OpDescPtr op_desc_ptr = std::make_shared<ge::OpDesc>("DecodeJpeg", "DecodeJpeg");
  ge::GeTensorDesc tensor_x(ge::GeShape(std::vector<int64_t>{1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_STRING);
  op_desc_ptr->AddInputDesc("contents", tensor_x);
  ge::GeTensorDesc tensor_y(ge::GeShape(std::vector<int64_t>{1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_UINT8);
  op_desc_ptr->AddOutputDesc("image", tensor_y);

  ge::AttrUtils::SetInt(op_desc_ptr, "channels", 3);
  ge::AttrUtils::SetInt(op_desc_ptr, "ratio", 1);
  ge::AttrUtils::SetBool(op_desc_ptr, "fancy_upscaling", true);
  ge::AttrUtils::SetBool(op_desc_ptr, "try_recover_truncated", false);
  ge::AttrUtils::SetFloat(op_desc_ptr, "acceptable_fraction", 1.0);
  ge::AttrUtils::SetStr(op_desc_ptr, "dct_method", "");
  ge::AttrUtils::SetStr(op_desc_ptr, "dst_img_format", "HWC");
  // set ATTR_NAME_PERFORMANCE_PRIOR
  ge::AttrUtils::SetBool(op_desc_ptr, dvpp::ATTR_NAME_PERFORMANCE_PRIOR, true);

  ge::OpDescPtr data_op_desc_ptr = std::make_shared<ge::OpDesc>("Data", "Data");
  ge::GeTensorDesc data_tensor_x(ge::GeShape({-2}), ge::FORMAT_ND, ge::DT_STRING);
  data_op_desc_ptr->AddInputDesc("x", data_tensor_x);
  ge::GeTensorDesc data_tensor_y(ge::GeShape({-2}), ge::FORMAT_ND, ge::DT_STRING);
  data_op_desc_ptr->AddOutputDesc("y", data_tensor_y);

  ge::OpDescPtr data_int_op_desc_ptr = std::make_shared<ge::OpDesc>("DataInt", "Data");
  ge::GeTensorDesc data_int_tensor_x(ge::GeShape(std::vector<int64_t>{4}), ge::FORMAT_ND, ge::DT_INT32);
  data_op_desc_ptr->AddInputDesc("x", data_int_tensor_x);
  ge::GeTensorDesc data_int_tensor_y(ge::GeShape(std::vector<int64_t>{4}), ge::FORMAT_ND, ge::DT_INT32);
  data_op_desc_ptr->AddOutputDesc("y", data_int_tensor_y);


  ge::OpDescPtr output_op_desc_ptr = std::make_shared<ge::OpDesc>("NetOutput", "NetOutput");
  ge::GeTensorDesc output_tensor_x(ge::GeShape({-2}), ge::FORMAT_ND, ge::DT_UINT8);
  output_op_desc_ptr->AddInputDesc(0, output_tensor_x);

  // set graph
  std::shared_ptr<ge::ComputeGraph> graphPtr = std::make_shared<ge::ComputeGraph>("test_graph");
  ge::AttrUtils::SetStr(graphPtr, ge::ATTR_NAME_SESSION_GRAPH_ID, "_0_1");

  ge::NodePtr decode_node = graphPtr->AddNode(op_desc_ptr);
  ge::NodePtr data_node = graphPtr->AddNode(data_op_desc_ptr);
  ge::NodePtr data_int_node = graphPtr->AddNode(data_int_op_desc_ptr);
  ge::NodePtr output_node = graphPtr->AddNode(output_op_desc_ptr);

  auto out_anchor = data_node->GetOutDataAnchor(0);
  auto peer_in_anchor = out_anchor->GetPeerInDataAnchors();
  std::vector<ge::InDataAnchorPtr> in_anchors(peer_in_anchor.begin(), peer_in_anchor.end());
  auto error = ge::GraphUtils::InsertNodeAfter(out_anchor, in_anchors, decode_node);
  ASSERT_EQ(error, ge::GRAPH_SUCCESS);

  auto decode_out_anchor = decode_node->GetOutDataAnchor(0);
  auto decode_peer_in_anchor = decode_out_anchor->GetPeerInDataAnchors();
  std::vector<ge::InDataAnchorPtr> output_in_anchors(decode_peer_in_anchor.begin(), decode_peer_in_anchor.end());
  error = ge::GraphUtils::InsertNodeAfter(decode_out_anchor, output_in_anchors, output_node);
  ASSERT_EQ(error, ge::GRAPH_SUCCESS);

  status = dvppGraphOptimizer.OptimizeOriginalGraph(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // Finalize
  status = dvppGraphOptimizer.Finalize();
  ASSERT_EQ(status, ge::SUCCESS);
}

TEST(DvppGraphOptimizer, Test_910B_DecodeJpeg_02)
{
  dvpp::DvppGraphOptimizer dvppGraphOptimizer;

  // Initialize
  std::map<std::string, std::string> options;
  options[ge::SOC_VERSION] = dvpp::kSocVersionAscend910B1;
  ge::Status status = dvppGraphOptimizer.Initialize(options, nullptr);
  ASSERT_EQ(status, ge::SUCCESS);

  // OptimizeOriginalGraph
  // set op
  ge::OpDescPtr op_desc_ptr = std::make_shared<ge::OpDesc>("DecodeJpeg_dvpp", "DecodeJpeg");
  ge::GeTensorDesc tensor_x(ge::GeShape(std::vector<int64_t>{1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_STRING);
  op_desc_ptr->AddInputDesc("contents", tensor_x);
  ge::GeTensorDesc tensor_y(ge::GeShape(std::vector<int64_t>{1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_UINT8);
  op_desc_ptr->AddOutputDesc("image", tensor_y);

  ge::AttrUtils::SetInt(op_desc_ptr, "channels", 3);
  ge::AttrUtils::SetInt(op_desc_ptr, "ratio", 1);
  ge::AttrUtils::SetBool(op_desc_ptr, "fancy_upscaling", true);
  ge::AttrUtils::SetBool(op_desc_ptr, "try_recover_truncated", false);
  ge::AttrUtils::SetFloat(op_desc_ptr, "acceptable_fraction", 1.0);
  ge::AttrUtils::SetStr(op_desc_ptr, "dct_method", "");
  ge::AttrUtils::SetStr(op_desc_ptr, "dst_img_format", "HWC");
  // set ATTR_NAME_PERFORMANCE_PRIOR
  ge::AttrUtils::SetBool(op_desc_ptr, dvpp::ATTR_NAME_PERFORMANCE_PRIOR, true);

  // set graph
  std::shared_ptr<ge::ComputeGraph> graphPtr = std::make_shared<ge::ComputeGraph>("test_graph");
  graphPtr->AddNode(op_desc_ptr);

  status = dvppGraphOptimizer.OptimizeGraphPrepare(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  status = dvppGraphOptimizer.OptimizeOriginalGraph(*graphPtr);
  ASSERT_EQ(status, ge::FAILED);

  // Finalize
  status = dvppGraphOptimizer.Finalize();
  ASSERT_EQ(status, ge::SUCCESS);
}

TEST(DvppGraphOptimizer, Test_910B_GaussianBlur_01)
{
  dvpp::DvppGraphOptimizer dvppGraphOptimizer;

  // Initialize
  std::map<std::string, std::string> options;
  options[ge::SOC_VERSION] = dvpp::kSocVersionAscend910B1;
  ge::Status status = dvppGraphOptimizer.Initialize(options, nullptr);
  ASSERT_EQ(status, ge::SUCCESS);

  dvpp::DvppOpsKernelInfoStore dvppOpsKernelInfoStore;
  status = dvppOpsKernelInfoStore.Initialize(options);
  ASSERT_EQ(status, ge::SUCCESS);

  ge::OpDescPtr op_desc_ptr = std::make_shared<ge::OpDesc>("GaussianBlur_dvpp", "GaussianBlur");
  ge::GeTensorDesc tensor_x(ge::GeShape(std::vector<int64_t>{1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_UINT8);
  op_desc_ptr->AddInputDesc("x", tensor_x);
  ge::GeTensorDesc tensor_y(ge::GeShape(std::vector<int64_t>{1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_UINT8);
  op_desc_ptr->AddOutputDesc("y", tensor_y);

  std::vector<int32_t> kernel_size;
  kernel_size.push_back(5);
  kernel_size.push_back(5);
  ge::AttrUtils::SetListInt(op_desc_ptr, "kernel_size", kernel_size);
  std::vector<float32_t> sigma;
  sigma.push_back(0);
  sigma.push_back(0);
  ge::AttrUtils::SetListFloat(op_desc_ptr, "sigma", sigma);
  ge::AttrUtils::SetStr(op_desc_ptr, "padding_mode", "constant");
  // set ATTR_NAME_PERFORMANCE_PRIOR
  ge::AttrUtils::SetBool(op_desc_ptr, dvpp::ATTR_NAME_PERFORMANCE_PRIOR, true);

  ge::OpDescPtr data_op_desc_ptr = std::make_shared<ge::OpDesc>("Data", "Data");
  ge::GeTensorDesc data_tensor_x(ge::GeShape({-2}), ge::FORMAT_ND, ge::DT_UINT8);
  data_op_desc_ptr->AddInputDesc("x", data_tensor_x);
  ge::GeTensorDesc data_tensor_y(ge::GeShape({-2}), ge::FORMAT_ND, ge::DT_UINT8);
  data_op_desc_ptr->AddOutputDesc("y", data_tensor_y);

  ge::OpDescPtr output_op_desc_ptr = std::make_shared<ge::OpDesc>("NetOutput", "NetOutput");
  ge::GeTensorDesc output_tensor_x(ge::GeShape({-2}), ge::FORMAT_ND, ge::DT_UINT8);
  output_op_desc_ptr->AddInputDesc(0, output_tensor_x);

  // set graph
  std::shared_ptr<ge::ComputeGraph> graphPtr = std::make_shared<ge::ComputeGraph>("test_graph");
  ge::AttrUtils::SetStr(graphPtr, ge::ATTR_NAME_SESSION_GRAPH_ID, "_0_1");
  ge::AttrUtils::SetBool(graphPtr, "_single_op_scene", true);

  ge::NodePtr blur_node = graphPtr->AddNode(op_desc_ptr);
  ge::NodePtr data_node = graphPtr->AddNode(data_op_desc_ptr);
  ge::NodePtr output_node = graphPtr->AddNode(output_op_desc_ptr);

  auto out_anchor = data_node->GetOutDataAnchor(0);
  auto peer_in_anchor = out_anchor->GetPeerInDataAnchors();
  std::vector<ge::InDataAnchorPtr> in_anchors(peer_in_anchor.begin(), peer_in_anchor.end());
  auto error = ge::GraphUtils::InsertNodeAfter(out_anchor, in_anchors, blur_node);
  ASSERT_EQ(error, ge::GRAPH_SUCCESS);

  auto decode_out_anchor = blur_node->GetOutDataAnchor(0);
  auto decode_peer_in_anchor = decode_out_anchor->GetPeerInDataAnchors();
  std::vector<ge::InDataAnchorPtr> output_in_anchors(decode_peer_in_anchor.begin(), decode_peer_in_anchor.end());
  error = ge::GraphUtils::InsertNodeAfter(decode_out_anchor, output_in_anchors, output_node);
  ASSERT_EQ(error, ge::GRAPH_SUCCESS);

  status = dvppGraphOptimizer.OptimizeOriginalGraph(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // Finalize
  status = dvppGraphOptimizer.Finalize();
  ASSERT_EQ(status, ge::SUCCESS);
}

TEST(DvppGraphOptimizer, Test_910B_Rotate_01)
{
  dvpp::DvppGraphOptimizer dvppGraphOptimizer;

  // Initialize
  std::map<std::string, std::string> options;
  options[ge::SOC_VERSION] = dvpp::kSocVersionAscend910B1;
  ge::Status status = dvppGraphOptimizer.Initialize(options, nullptr);
  ASSERT_EQ(status, ge::SUCCESS);

  dvpp::DvppOpsKernelInfoStore dvppOpsKernelInfoStore;
  status = dvppOpsKernelInfoStore.Initialize(options);
  ASSERT_EQ(status, ge::SUCCESS);

  ge::OpDescPtr op_desc_ptr = std::make_shared<ge::OpDesc>("Rotate_dvpp", "Rotate");
  ge::GeTensorDesc tensor_x(ge::GeShape(std::vector<int64_t>{1280, 720, 3}), ge::FORMAT_ND, ge::DT_UINT8);
  op_desc_ptr->AddInputDesc("x", tensor_x);
  ge::GeTensorDesc tensor_y(ge::GeShape(std::vector<int64_t>{1280, 720, 3}), ge::FORMAT_ND, ge::DT_UINT8);
  op_desc_ptr->AddOutputDesc("y", tensor_y);

  std::vector<int32_t> center;
  center.push_back(640);
  center.push_back(360);
  ge::AttrUtils::SetListInt(op_desc_ptr, "center", center);
  ge::AttrUtils::SetFloat(op_desc_ptr, "angle", 180.0);
  ge::AttrUtils::SetBool(op_desc_ptr, "expand", true);
  ge::AttrUtils::SetStr(op_desc_ptr, "interpolation", "bilinear");
  ge::AttrUtils::SetStr(op_desc_ptr, "padding_mode", "constant");
  ge::AttrUtils::SetFloat(op_desc_ptr, "padding_value", 0.0);
  ge::AttrUtils::SetStr(op_desc_ptr, "data_format", "HWC");
  // set ATTR_NAME_PERFORMANCE_PRIOR
  ge::AttrUtils::SetBool(op_desc_ptr, dvpp::ATTR_NAME_PERFORMANCE_PRIOR, true);
  ge::OpDescUtilsEx::SetType(op_desc_ptr, "Rotate");

  ge::OpDescPtr data_op_desc_ptr = std::make_shared<ge::OpDesc>("Data", "Data");
  ge::GeTensorDesc data_tensor_x(ge::GeShape({1280, 720, 3}), ge::FORMAT_ND, ge::DT_UINT8);
  data_op_desc_ptr->AddInputDesc("x", data_tensor_x);
  ge::GeTensorDesc data_tensor_y(ge::GeShape({1280, 720, 3}), ge::FORMAT_ND, ge::DT_UINT8);
  data_op_desc_ptr->AddOutputDesc("y", data_tensor_y);

  ge::OpDescPtr output_op_desc_ptr = std::make_shared<ge::OpDesc>("NetOutput", "NetOutput");
  ge::GeTensorDesc output_tensor_x(ge::GeShape({1280, 720, 3}), ge::FORMAT_ND, ge::DT_UINT8);
  output_op_desc_ptr->AddInputDesc(0, output_tensor_x);

  // set graph
  std::shared_ptr<ge::ComputeGraph> graphPtr = std::make_shared<ge::ComputeGraph>("test_graph");
  ge::AttrUtils::SetStr(graphPtr, ge::ATTR_NAME_SESSION_GRAPH_ID, "_0_1");
  ge::AttrUtils::SetBool(graphPtr, "_single_op_scene", true);

  ge::NodePtr blur_node = graphPtr->AddNode(op_desc_ptr);
  ge::NodePtr data_node = graphPtr->AddNode(data_op_desc_ptr);
  ge::NodePtr output_node = graphPtr->AddNode(output_op_desc_ptr);

  auto out_anchor = data_node->GetOutDataAnchor(0);
  auto peer_in_anchor = out_anchor->GetPeerInDataAnchors();
  std::vector<ge::InDataAnchorPtr> in_anchors(peer_in_anchor.begin(), peer_in_anchor.end());
  auto error = ge::GraphUtils::InsertNodeAfter(out_anchor, in_anchors, blur_node);
  ASSERT_EQ(error, ge::GRAPH_SUCCESS);

  auto decode_out_anchor = blur_node->GetOutDataAnchor(0);
  auto decode_peer_in_anchor = decode_out_anchor->GetPeerInDataAnchors();
  std::vector<ge::InDataAnchorPtr> output_in_anchors(decode_peer_in_anchor.begin(), decode_peer_in_anchor.end());
  error = ge::GraphUtils::InsertNodeAfter(decode_out_anchor, output_in_anchors, output_node);
  ASSERT_EQ(error, ge::GRAPH_SUCCESS);

  status = dvppGraphOptimizer.OptimizeGraphPrepare(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  status = dvppGraphOptimizer.OptimizeOriginalGraph(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // Finalize
  status = dvppGraphOptimizer.Finalize();
  ASSERT_EQ(status, ge::SUCCESS);
}

TEST(DvppGraphOptimizer, Test_910B_RgbToGrayscale_01)
{
  dvpp::DvppGraphOptimizer dvppGraphOptimizer;

  // Initialize
  std::map<std::string, std::string> options;
  options[ge::SOC_VERSION] = dvpp::kSocVersionAscend910B1;
  ge::Status status = dvppGraphOptimizer.Initialize(options, nullptr);
  ASSERT_EQ(status, ge::SUCCESS);

  dvpp::DvppOpsKernelInfoStore dvppOpsKernelInfoStore;
  status = dvppOpsKernelInfoStore.Initialize(options);
  ASSERT_EQ(status, ge::SUCCESS);

  ge::OpDescPtr op_desc_ptr = std::make_shared<ge::OpDesc>("RgbToGrayscale_dvpp", "RgbToGrayscale");
  ge::GeTensorDesc tensor_x(ge::GeShape(std::vector<int64_t>{1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_UINT8);
  op_desc_ptr->AddInputDesc("image", tensor_x);
  ge::GeTensorDesc tensor_y(ge::GeShape(std::vector<int64_t>{1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_UINT8);
  op_desc_ptr->AddOutputDesc("y", tensor_y);

  ge::AttrUtils::SetInt(op_desc_ptr, "output_channels", 3);
  ge::AttrUtils::SetStr(op_desc_ptr, "data_format", "HWC");
  // set ATTR_NAME_PERFORMANCE_PRIOR
  ge::AttrUtils::SetBool(op_desc_ptr, dvpp::ATTR_NAME_PERFORMANCE_PRIOR, true);
  ge::OpDescUtilsEx::SetType(op_desc_ptr, "RgbToGrayscale");

  ge::OpDescPtr data_op_desc_ptr = std::make_shared<ge::OpDesc>("Data", "Data");
  ge::GeTensorDesc data_tensor_x(ge::GeShape({1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_UINT8);
  data_op_desc_ptr->AddInputDesc("x", data_tensor_x);
  ge::GeTensorDesc data_tensor_y(ge::GeShape({1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_UINT8);
  data_op_desc_ptr->AddOutputDesc("y", data_tensor_y);

  ge::OpDescPtr output_op_desc_ptr = std::make_shared<ge::OpDesc>("NetOutput", "NetOutput");
  ge::GeTensorDesc output_tensor_x(ge::GeShape({1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_UINT8);
  output_op_desc_ptr->AddInputDesc(0, output_tensor_x);

  // set graph
  std::shared_ptr<ge::ComputeGraph> graphPtr = std::make_shared<ge::ComputeGraph>("test_graph");
  ge::AttrUtils::SetStr(graphPtr, ge::ATTR_NAME_SESSION_GRAPH_ID, "_0_1");
  ge::AttrUtils::SetBool(graphPtr, "_single_op_scene", true);

  ge::NodePtr blur_node = graphPtr->AddNode(op_desc_ptr);
  ge::NodePtr data_node = graphPtr->AddNode(data_op_desc_ptr);
  ge::NodePtr output_node = graphPtr->AddNode(output_op_desc_ptr);

  auto out_anchor = data_node->GetOutDataAnchor(0);
  auto peer_in_anchor = out_anchor->GetPeerInDataAnchors();
  std::vector<ge::InDataAnchorPtr> in_anchors(peer_in_anchor.begin(), peer_in_anchor.end());
  auto error = ge::GraphUtils::InsertNodeAfter(out_anchor, in_anchors, blur_node);
  ASSERT_EQ(error, ge::GRAPH_SUCCESS);

  auto decode_out_anchor = blur_node->GetOutDataAnchor(0);
  auto decode_peer_in_anchor = decode_out_anchor->GetPeerInDataAnchors();
  std::vector<ge::InDataAnchorPtr> output_in_anchors(decode_peer_in_anchor.begin(), decode_peer_in_anchor.end());
  error = ge::GraphUtils::InsertNodeAfter(decode_out_anchor, output_in_anchors, output_node);
  ASSERT_EQ(error, ge::GRAPH_SUCCESS);

  status = dvppGraphOptimizer.OptimizeGraphPrepare(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  status = dvppGraphOptimizer.OptimizeOriginalGraph(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // Finalize
  status = dvppGraphOptimizer.Finalize();
  ASSERT_EQ(status, ge::SUCCESS);
}

TEST(DvppGraphOptimizer, Test_910B_Rotate_02)
{
  dvpp::DvppGraphOptimizer dvppGraphOptimizer;

  // Initialize
  std::map<std::string, std::string> options;
  options[ge::SOC_VERSION] = dvpp::kSocVersionAscend910B1;
  ge::Status status = dvppGraphOptimizer.Initialize(options, nullptr);
  ASSERT_EQ(status, ge::SUCCESS);

  dvpp::DvppOpsKernelInfoStore dvppOpsKernelInfoStore;
  status = dvppOpsKernelInfoStore.Initialize(options);
  ASSERT_EQ(status, ge::SUCCESS);

  ge::OpDescPtr op_desc_ptr = std::make_shared<ge::OpDesc>("Rotate_dvpp", "Rotate");
  ge::GeTensorDesc tensor_x(ge::GeShape(std::vector<int64_t>{3, 1280, 720}), ge::FORMAT_ND, ge::DT_UINT8);
  op_desc_ptr->AddInputDesc("x", tensor_x);
  ge::GeTensorDesc tensor_y(ge::GeShape(std::vector<int64_t>{3, 1280, 720}), ge::FORMAT_ND, ge::DT_UINT8);
  op_desc_ptr->AddOutputDesc("y", tensor_y);

  std::vector<int32_t> center;
  center.push_back(640);
  center.push_back(360);
  ge::AttrUtils::SetListInt(op_desc_ptr, "center", center);
  ge::AttrUtils::SetFloat(op_desc_ptr, "angle", 180.0);
  ge::AttrUtils::SetBool(op_desc_ptr, "expand", true);
  ge::AttrUtils::SetStr(op_desc_ptr, "interpolation", "bilinear");
  ge::AttrUtils::SetStr(op_desc_ptr, "padding_mode", "constant");
  ge::AttrUtils::SetFloat(op_desc_ptr, "padding_value", 0.0);
  ge::AttrUtils::SetStr(op_desc_ptr, "data_format", "CHW");
  // set ATTR_NAME_PERFORMANCE_PRIOR
  ge::AttrUtils::SetBool(op_desc_ptr, dvpp::ATTR_NAME_PERFORMANCE_PRIOR, true);
  ge::OpDescUtilsEx::SetType(op_desc_ptr, "Rotate");

  ge::OpDescPtr data_op_desc_ptr = std::make_shared<ge::OpDesc>("Data", "Data");
  ge::GeTensorDesc data_tensor_x(ge::GeShape({3, 1280, 720}), ge::FORMAT_ND, ge::DT_UINT8);
  data_op_desc_ptr->AddInputDesc("x", data_tensor_x);
  ge::GeTensorDesc data_tensor_y(ge::GeShape({3, 1280, 720}), ge::FORMAT_ND, ge::DT_UINT8);
  data_op_desc_ptr->AddOutputDesc("y", data_tensor_y);

  ge::OpDescPtr output_op_desc_ptr = std::make_shared<ge::OpDesc>("NetOutput", "NetOutput");
  ge::GeTensorDesc output_tensor_x(ge::GeShape({3, 1280, 720}), ge::FORMAT_ND, ge::DT_UINT8);
  output_op_desc_ptr->AddInputDesc(0, output_tensor_x);

  // set graph
  std::shared_ptr<ge::ComputeGraph> graphPtr = std::make_shared<ge::ComputeGraph>("test_graph");
  ge::AttrUtils::SetStr(graphPtr, ge::ATTR_NAME_SESSION_GRAPH_ID, "_0_1");
  ge::AttrUtils::SetBool(graphPtr, "_single_op_scene", true);

  ge::NodePtr blur_node = graphPtr->AddNode(op_desc_ptr);
  ge::NodePtr data_node = graphPtr->AddNode(data_op_desc_ptr);
  ge::NodePtr output_node = graphPtr->AddNode(output_op_desc_ptr);

  auto out_anchor = data_node->GetOutDataAnchor(0);
  auto peer_in_anchor = out_anchor->GetPeerInDataAnchors();
  std::vector<ge::InDataAnchorPtr> in_anchors(peer_in_anchor.begin(), peer_in_anchor.end());
  auto error = ge::GraphUtils::InsertNodeAfter(out_anchor, in_anchors, blur_node);
  ASSERT_EQ(error, ge::GRAPH_SUCCESS);

  auto decode_out_anchor = blur_node->GetOutDataAnchor(0);
  auto decode_peer_in_anchor = decode_out_anchor->GetPeerInDataAnchors();
  std::vector<ge::InDataAnchorPtr> output_in_anchors(decode_peer_in_anchor.begin(), decode_peer_in_anchor.end());
  error = ge::GraphUtils::InsertNodeAfter(decode_out_anchor, output_in_anchors, output_node);
  ASSERT_EQ(error, ge::GRAPH_SUCCESS);

  status = dvppGraphOptimizer.OptimizeGraphPrepare(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  status = dvppGraphOptimizer.OptimizeOriginalGraph(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // Finalize
  status = dvppGraphOptimizer.Finalize();
  ASSERT_EQ(status, ge::SUCCESS);
}

TEST(DvppGraphOptimizer, Test_910B_WarpAffineV2_01)
{
  dvpp::DvppGraphOptimizer dvppGraphOptimizer;

  // Initialize
  std::map<std::string, std::string> options;
  options[ge::SOC_VERSION] = dvpp::kSocVersionAscend910B1;
  ge::Status status = dvppGraphOptimizer.Initialize(options, nullptr);
  ASSERT_EQ(status, ge::SUCCESS);

  dvpp::DvppOpsKernelInfoStore dvppOpsKernelInfoStore;
  status = dvppOpsKernelInfoStore.Initialize(options);
  ASSERT_EQ(status, ge::SUCCESS);

  ge::OpDescPtr op_desc_ptr = std::make_shared<ge::OpDesc>("WarpAffineV2_dvpp", "WarpAffineV2");
  ge::GeTensorDesc tensor_x(ge::GeShape(std::vector<int64_t>{1280, 720, 3}), ge::FORMAT_ND, ge::DT_UINT8);
  op_desc_ptr->AddInputDesc("x", tensor_x);
  ge::GeTensorDesc tensor_matrix(ge::GeShape(std::vector<int64_t>{2, 3}), ge::FORMAT_ND, ge::DT_FLOAT);
  op_desc_ptr->AddInputDesc("matrix", tensor_matrix);
  ge::GeTensorDesc tensor_dst_size(ge::GeShape(std::vector<int64_t>{1,2}), ge::FORMAT_ND, ge::DT_INT32);
  op_desc_ptr->AddInputDesc("dst_size", tensor_dst_size);
  ge::GeTensorDesc tensor_y(ge::GeShape(std::vector<int64_t>{1280, 720, 3}), ge::FORMAT_ND, ge::DT_UINT8);
  op_desc_ptr->AddOutputDesc("y", tensor_y);

  ge::AttrUtils::SetStr(op_desc_ptr, "interpolation", "bilinear");
  ge::AttrUtils::SetStr(op_desc_ptr, "padding_mode", "constant");
  ge::AttrUtils::SetFloat(op_desc_ptr, "padding_value", 0.0);
  ge::AttrUtils::SetStr(op_desc_ptr, "data_format", "HWC");
  // set ATTR_NAME_PERFORMANCE_PRIOR
  ge::AttrUtils::SetBool(op_desc_ptr, dvpp::ATTR_NAME_PERFORMANCE_PRIOR, true);
  ge::OpDescUtilsEx::SetType(op_desc_ptr, "WarpAffineV2");

  ge::OpDescPtr data_op_desc_ptr = std::make_shared<ge::OpDesc>("Data", "Data");
  ge::GeTensorDesc data_tensor_x(ge::GeShape({3, 1280, 720}), ge::FORMAT_ND, ge::DT_UINT8);
  data_op_desc_ptr->AddInputDesc("x", data_tensor_x);
  ge::GeTensorDesc data_tensor_y(ge::GeShape({3, 1280, 720}), ge::FORMAT_ND, ge::DT_UINT8);
  data_op_desc_ptr->AddOutputDesc("y", data_tensor_y);

  ge::OpDescPtr output_op_desc_ptr = std::make_shared<ge::OpDesc>("NetOutput", "NetOutput");
  ge::GeTensorDesc output_tensor_x(ge::GeShape({3, 1280, 720}), ge::FORMAT_ND, ge::DT_UINT8);
  output_op_desc_ptr->AddInputDesc(0, output_tensor_x);

  // set graph
  std::shared_ptr<ge::ComputeGraph> graphPtr = std::make_shared<ge::ComputeGraph>("test_graph");
  ge::AttrUtils::SetStr(graphPtr, ge::ATTR_NAME_SESSION_GRAPH_ID, "_0_1");
  ge::AttrUtils::SetBool(graphPtr, "_single_op_scene", true);

  ge::NodePtr warp_affine_node = graphPtr->AddNode(op_desc_ptr);
  ge::NodePtr data_node = graphPtr->AddNode(data_op_desc_ptr);
  ge::NodePtr output_node = graphPtr->AddNode(output_op_desc_ptr);

  auto out_anchor = data_node->GetOutDataAnchor(0);
  auto peer_in_anchor = out_anchor->GetPeerInDataAnchors();
  std::vector<ge::InDataAnchorPtr> in_anchors(peer_in_anchor.begin(), peer_in_anchor.end());
  auto error = ge::GraphUtils::InsertNodeAfter(out_anchor, in_anchors, warp_affine_node);
  ASSERT_EQ(error, ge::GRAPH_SUCCESS);

  auto warp_affine_out_anchor = warp_affine_node->GetOutDataAnchor(0);
  auto warp_affine_peer_in_anchor = warp_affine_out_anchor->GetPeerInDataAnchors();
  std::vector<ge::InDataAnchorPtr> output_in_anchors(warp_affine_peer_in_anchor.begin(),
      warp_affine_peer_in_anchor.end());
  error = ge::GraphUtils::InsertNodeAfter(warp_affine_out_anchor, output_in_anchors, output_node);
  ASSERT_EQ(error, ge::GRAPH_SUCCESS);

  status = dvppGraphOptimizer.OptimizeGraphPrepare(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  status = dvppGraphOptimizer.OptimizeOriginalGraph(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // Finalize
  status = dvppGraphOptimizer.Finalize();
  ASSERT_EQ(status, ge::SUCCESS);
}

TEST(DvppGraphOptimizer, Initialize)
{
  dvpp::DvppGraphOptimizer dvppGraphOptimizer;
  std::map<std::string, std::string> options;
  options[ge::SOC_VERSION] = "Ascend";
  ge::Status status = dvppGraphOptimizer.Initialize(options, nullptr);
  ASSERT_EQ(status, ge::FAILED);
}

TEST(DvppGraphOptimizer, AllTest_910B_WarpPerspective)
{
  dvpp::DvppGraphOptimizer dvppGraphOptimizer;

  // Initialize
  std::map<std::string, std::string> options;
  options[ge::SOC_VERSION] = dvpp::kSocVersionAscend910B1;
  ge::Status status = dvppGraphOptimizer.Initialize(options, nullptr);
  ASSERT_EQ(status, ge::SUCCESS);

  // OptimizeOriginalGraph
  ge::OpDescPtr op_desc_ptr = std::make_shared<ge::OpDesc>("WarpPerspective", "WarpPerspective");
  ge::GeTensorDesc tensorX(ge::GeShape(std::vector<int64_t>{1, 224, 224, 3}), ge::FORMAT_NHWC, ge::DT_UINT8);
  op_desc_ptr->AddInputDesc("x", tensorX);
  ge::GeTensorDesc tensorMatrix(ge::GeShape(std::vector<int64_t>(0.5, 0.5)), ge::FORMAT_ND, ge::DT_DOUBLE);
  op_desc_ptr->AddInputDesc("matrix", tensorMatrix);
  ge::GeTensorDesc tensorY(ge::GeShape(std::vector<int64_t>{1, 224, 224, 3}), ge::FORMAT_NHWC, ge::DT_UINT8);
  op_desc_ptr->AddOutputDesc("y", tensorY);
  ge::AttrUtils::SetBool(op_desc_ptr, "align_corners", false);
  ge::AttrUtils::SetBool(op_desc_ptr, "half_pixel_centers", false);
  ge::AttrUtils::SetInt(op_desc_ptr, "out_height", 224);
  ge::AttrUtils::SetInt(op_desc_ptr, "out_width", 224);
  ge::AttrUtils::SetStr(op_desc_ptr, "interpolation_mode", "bilinear");
  ge::AttrUtils::SetStr(op_desc_ptr, "border_type", "BORDER_CONSTANT");
  ge::AttrUtils::SetFloat(op_desc_ptr, "constant", 0.0);
  ge::AttrUtils::SetStr(op_desc_ptr, "data_format", "CHW");

  // set ATTR_NAME_PERFORMANCE_PRIOR
  ge::AttrUtils::SetBool(op_desc_ptr, dvpp::ATTR_NAME_PERFORMANCE_PRIOR, true);
  ge::OpDescUtilsEx::SetType(op_desc_ptr, "WarpPerspective");

  // set graph
  std::shared_ptr<ge::ComputeGraph> graphPtr = std::make_shared<ge::ComputeGraph>("test_graph");
  graphPtr->AddNode(op_desc_ptr);
  ge::AttrUtils::SetBool(graphPtr, "_single_op_scene", true);

  // OptimizeGraphPrepare
  status = dvppGraphOptimizer.OptimizeGraphPrepare(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

 // OptimizeOriginalGraph
  status = dvppGraphOptimizer.OptimizeOriginalGraph(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // OptimizeOriginalGraphJudgeInsert
  status = dvppGraphOptimizer.OptimizeOriginalGraphJudgeInsert(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // OptimizeFusedGraph
  status = dvppGraphOptimizer.OptimizeFusedGraph(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // OptimizeWholeGraph
  status = dvppGraphOptimizer.OptimizeWholeGraph(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // GetAttributes
  ge::GraphOptimizerAttribute attrs;
  status = dvppGraphOptimizer.GetAttributes(attrs);
  ASSERT_EQ(status, ge::SUCCESS);

  // Finalize
  status = dvppGraphOptimizer.Finalize();
  ASSERT_EQ(status, ge::SUCCESS);
}

TEST(DvppGraphOptimizer, AllTest_910B_ResizeV2)
{
  dvpp::DvppGraphOptimizer dvppGraphOptimizer;

  // Initialize
  std::map<std::string, std::string> options;
  options[ge::SOC_VERSION] = dvpp::kSocVersionAscend910B1;
  ge::Status status = dvppGraphOptimizer.Initialize(options, nullptr);
  ASSERT_EQ(status, ge::SUCCESS);

  // OptimizeOriginalGraph
  ge::OpDescPtr op_desc_ptr = std::make_shared<ge::OpDesc>("ResizeV2", "ResizeV2");
  ge::GeTensorDesc tensorX(ge::GeShape(std::vector<int64_t>{1, 3, 224, 224}), ge::FORMAT_NCHW, ge::DT_UINT8);
  op_desc_ptr->AddInputDesc("x", tensorX);
  ge::GeTensorDesc tensorDstSize(ge::GeShape(std::vector<int64_t>(224, 224)), ge::FORMAT_ND, ge::DT_INT64);
  op_desc_ptr->AddInputDesc("dst_size", tensorDstSize);
  ge::GeTensorDesc tensorY(ge::GeShape(std::vector<int64_t>{1, 3, 224, 224}), ge::FORMAT_NCHW, ge::DT_UINT8);
  op_desc_ptr->AddOutputDesc("y", tensorY);
  ge::AttrUtils::SetBool(op_desc_ptr, "align_corners", false);
  ge::AttrUtils::SetBool(op_desc_ptr, "half_pixel_centers", false);
  ge::AttrUtils::SetStr(op_desc_ptr, "interpolation", "bilinear");
  ge::AttrUtils::SetStr(op_desc_ptr, "data_format", "CHW");

  // set ATTR_NAME_PERFORMANCE_PRIOR
  ge::AttrUtils::SetBool(op_desc_ptr, dvpp::ATTR_NAME_PERFORMANCE_PRIOR, true);
  ge::OpDescUtilsEx::SetType(op_desc_ptr, "ResizeV2");

  // set graph
  std::shared_ptr<ge::ComputeGraph> graphPtr = std::make_shared<ge::ComputeGraph>("test_graph");
  graphPtr->AddNode(op_desc_ptr);
  ge::AttrUtils::SetBool(graphPtr, "_single_op_scene", true);

  // OptimizeGraphPrepare
  status = dvppGraphOptimizer.OptimizeGraphPrepare(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

 // OptimizeOriginalGraph
  status = dvppGraphOptimizer.OptimizeOriginalGraph(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // OptimizeOriginalGraphJudgeInsert
  status = dvppGraphOptimizer.OptimizeOriginalGraphJudgeInsert(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // OptimizeFusedGraph
  status = dvppGraphOptimizer.OptimizeFusedGraph(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // OptimizeWholeGraph
  status = dvppGraphOptimizer.OptimizeWholeGraph(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // GetAttributes
  ge::GraphOptimizerAttribute attrs;
  status = dvppGraphOptimizer.GetAttributes(attrs);
  ASSERT_EQ(status, ge::SUCCESS);

  // Finalize
  status = dvppGraphOptimizer.Finalize();
  ASSERT_EQ(status, ge::SUCCESS);
}

TEST(DvppGraphOptimizer, Test_910B_AdjustContrast_01)
{
  dvpp::DvppGraphOptimizer dvppGraphOptimizer;

  // Initialize
  std::map<std::string, std::string> options;
  options[ge::SOC_VERSION] = dvpp::kSocVersionAscend910B1;
  ge::Status status = dvppGraphOptimizer.Initialize(options, nullptr);
  ASSERT_EQ(status, ge::SUCCESS);

  dvpp::DvppOpsKernelInfoStore dvppOpsKernelInfoStore;
  status = dvppOpsKernelInfoStore.Initialize(options);
  ASSERT_EQ(status, ge::SUCCESS);

  ge::OpDescPtr op_desc_ptr = std::make_shared<ge::OpDesc>("AdjustContrast_dvpp", "AdjustContrast");
  ge::GeTensorDesc tensor_x(ge::GeShape(std::vector<int64_t>{1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_UINT8);
  op_desc_ptr->AddInputDesc("image", tensor_x);
  ge::GeTensorDesc tensor_factor(ge::GeShape(std::vector<int64_t>{1}), ge::FORMAT_ND, ge::DT_FLOAT);
  op_desc_ptr->AddInputDesc("constrast_factor", tensor_factor);
  ge::GeTensorDesc tensor_y(ge::GeShape(std::vector<int64_t>{1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_UINT8);
  op_desc_ptr->AddOutputDesc("y", tensor_y);

  ge::AttrUtils::SetStr(op_desc_ptr, "mean_mode", "chn_y");
  ge::AttrUtils::SetStr(op_desc_ptr, "data_format", "HWC");
  // set ATTR_NAME_PERFORMANCE_PRIOR
  ge::AttrUtils::SetBool(op_desc_ptr, dvpp::ATTR_NAME_PERFORMANCE_PRIOR, true);
  ge::OpDescUtilsEx::SetType(op_desc_ptr, "AdjustContrast");

  // set graph
  std::shared_ptr<ge::ComputeGraph> graphPtr = std::make_shared<ge::ComputeGraph>("test_graph");
  ge::AttrUtils::SetStr(graphPtr, ge::ATTR_NAME_SESSION_GRAPH_ID, "_0_1");
  ge::AttrUtils::SetBool(graphPtr, "_single_op_scene", true);

  graphPtr->AddNode(op_desc_ptr);

  status = dvppGraphOptimizer.OptimizeGraphPrepare(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  status = dvppGraphOptimizer.OptimizeOriginalGraph(*graphPtr);
  ASSERT_EQ(status, ge::FAILED);

  // Finalize
  status = dvppGraphOptimizer.Finalize();
  ASSERT_EQ(status, ge::SUCCESS);
}

TEST(DvppGraphOptimizer, Test_910B_AdjustContrast_02)
{
  dvpp::DvppGraphOptimizer dvppGraphOptimizer;
  // Initialize
  std::map<std::string, std::string> options;
  options[ge::SOC_VERSION] = dvpp::kSocVersionAscend910B1;
  ge::Status status = dvppGraphOptimizer.Initialize(options, nullptr);
  ASSERT_EQ(status, ge::SUCCESS);

  ge::OpDescPtr op_desc_ptr = std::make_shared<ge::OpDesc>("AdjustContrast_dvpp", "AdjustContrast");
  ge::GeTensorDesc tensor_x(ge::GeShape(std::vector<int64_t>{1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_UINT8);
  op_desc_ptr->AddInputDesc("image", tensor_x);
  ge::GeTensorDesc tensor_factor(ge::GeShape(std::vector<int64_t>{1}), ge::FORMAT_ND, ge::DT_FLOAT);
  op_desc_ptr->AddInputDesc("constrast_factor", tensor_factor);
  ge::GeTensorDesc tensor_y(ge::GeShape(std::vector<int64_t>{1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_UINT8);
  op_desc_ptr->AddOutputDesc("y", tensor_y);

  ge::AttrUtils::SetStr(op_desc_ptr, "mean_mode", "chn_y");
  ge::AttrUtils::SetStr(op_desc_ptr, "data_format", "HWC");
  // set ATTR_NAME_PERFORMANCE_PRIOR
  ge::AttrUtils::SetBool(op_desc_ptr, dvpp::ATTR_NAME_PERFORMANCE_PRIOR, true);
  ge::OpDescUtilsEx::SetType(op_desc_ptr, "AdjustContrast");

  ge::OpDescPtr data_op_desc_ptr = std::make_shared<ge::OpDesc>("Data", "Data");
  ge::GeTensorDesc data_tensor_x(ge::GeShape({1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_UINT8);
  data_op_desc_ptr->AddInputDesc("x", data_tensor_x);
  ge::GeTensorDesc data_tensor_y(ge::GeShape({1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_UINT8);
  data_op_desc_ptr->AddOutputDesc("y", data_tensor_y);

  ge::OpDescPtr data_factor_op_desc_ptr = std::make_shared<ge::OpDesc>("DataFactor", "Data");
  ge::GeTensorDesc factor_tensor_x(ge::GeShape(std::vector<int64_t>{1}), ge::FORMAT_ND, ge::DT_FLOAT);
  data_factor_op_desc_ptr->AddInputDesc("x", factor_tensor_x);
  ge::GeTensorDesc factor_tensor_y(ge::GeShape(std::vector<int64_t>{1}), ge::FORMAT_ND, ge::DT_FLOAT);
  data_factor_op_desc_ptr->AddOutputDesc("y", factor_tensor_y);


  ge::OpDescPtr output_op_desc_ptr = std::make_shared<ge::OpDesc>("NetOutput", "NetOutput");
  ge::GeTensorDesc output_tensor_x(ge::GeShape({-1, -1, -1, -1}), ge::FORMAT_ND, ge::DT_UINT8);
  output_op_desc_ptr->AddInputDesc(0, output_tensor_x);

  // set graph
  std::shared_ptr<ge::ComputeGraph> graphPtr = std::make_shared<ge::ComputeGraph>("test_graph");
  ge::AttrUtils::SetStr(graphPtr, ge::ATTR_NAME_SESSION_GRAPH_ID, "_0_1");

  ge::NodePtr contrast_node = graphPtr->AddNode(op_desc_ptr);
  ge::NodePtr data_node = graphPtr->AddNode(data_op_desc_ptr);
  ge::NodePtr data_factor_node = graphPtr->AddNode(data_factor_op_desc_ptr);
  ge::NodePtr output_node = graphPtr->AddNode(output_op_desc_ptr);

  auto out_anchor = data_node->GetOutDataAnchor(0);
  auto contrast_in0_anchor = contrast_node->GetInDataAnchor(0);
  std::vector<ge::InDataAnchorPtr> in0_anchors;
  in0_anchors.push_back(contrast_in0_anchor);
  auto error = ge::GraphUtils::InsertNodeAfter(out_anchor, in0_anchors, contrast_node);
  ASSERT_EQ(error, ge::GRAPH_SUCCESS);

  auto factor_out_anchor = data_factor_node->GetOutAnchor(0);
  auto contrast_in1_anchor = contrast_node->GetInAnchor(1);
  error = ge::GraphUtils::AddEdge(factor_out_anchor, contrast_in1_anchor);
  ASSERT_EQ(error, ge::GRAPH_SUCCESS);

  auto contrast_out_anchor = contrast_node->GetOutAnchor(0);
  auto output_in0_anchor = output_node->GetInAnchor(0);
  error = ge::GraphUtils::AddEdge(contrast_out_anchor, output_in0_anchor);
  ASSERT_EQ(error, ge::GRAPH_SUCCESS);

  status = dvppGraphOptimizer.OptimizeOriginalGraph(*graphPtr);
  ASSERT_EQ(status, ge::SUCCESS);

  // Finalize
  status = dvppGraphOptimizer.Finalize();
  ASSERT_EQ(status, ge::SUCCESS);
}

TEST(DvppGraphOptimizer, Test_CheckSubAndMulFusedIntoNormalizeV2_01)
{
  // set graph
  std::shared_ptr<ge::ComputeGraph> graphPtr = std::make_shared<ge::ComputeGraph>("test_graph");
  ge::AttrUtils::SetStr(graphPtr, ge::ATTR_NAME_SESSION_GRAPH_ID, "_0_1");

  ge::OpDescPtr op_desc_ptr = std::make_shared<ge::OpDesc>("AdjustContrast_dvpp", "AdjustContrast");
  ge::GeTensorDesc tensor_x(ge::GeShape(std::vector<int64_t>{1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_UINT8);
  op_desc_ptr->AddInputDesc("image", tensor_x);
  ge::GeTensorDesc tensor_factor(ge::GeShape(std::vector<int64_t>{1}), ge::FORMAT_ND, ge::DT_FLOAT);
  op_desc_ptr->AddInputDesc("constrast_factor", tensor_factor);
  ge::GeTensorDesc tensor_y(ge::GeShape(std::vector<int64_t>{1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_UINT8);
  op_desc_ptr->AddOutputDesc("y", tensor_y);
  op_desc_ptr->SetType("Mul");

  ge::NodePtr node = graphPtr->AddNode(op_desc_ptr);
  bool status = dvpp::DvppOpsChecker::CheckSubAndMulFusedIntoNormalizeV2(node);
  ASSERT_EQ(status, false);
}

TEST(DvppGraphOptimizer, Test_CheckSubAndMulFusedIntoNormalizeV2_02)
{
  // set graph
  std::shared_ptr<ge::ComputeGraph> graphPtr = std::make_shared<ge::ComputeGraph>("test_graph");
  ge::AttrUtils::SetStr(graphPtr, ge::ATTR_NAME_SESSION_GRAPH_ID, "_0_1");

  ge::OpDescPtr op_desc_ptr = std::make_shared<ge::OpDesc>("AdjustContrast_dvpp", "AdjustContrast");
  ge::GeTensorDesc tensor_x(ge::GeShape(std::vector<int64_t>{1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_UINT8);
  op_desc_ptr->AddInputDesc("image", tensor_x);
  ge::GeTensorDesc tensor_factor(ge::GeShape(std::vector<int64_t>{1}), ge::FORMAT_ND, ge::DT_FLOAT);
  op_desc_ptr->AddInputDesc("constrast_factor", tensor_factor);
  ge::GeTensorDesc tensor_y(ge::GeShape(std::vector<int64_t>{1, 1280, 720, 3}), ge::FORMAT_NHWC, ge::DT_UINT8);
  op_desc_ptr->AddOutputDesc("y", tensor_y);
  op_desc_ptr->SetType("Mul");
  ge::NodePtr node = graphPtr->AddNode(op_desc_ptr);

  ge::OpDescPtr sub_op_desc_ptr = std::make_shared<ge::OpDesc>("AdjustContrast_dvpp", "AdjustContrast");
  sub_op_desc_ptr->AddInputDesc("i", tensor_x);
  sub_op_desc_ptr->AddInputDesc("j", tensor_factor);
  sub_op_desc_ptr->AddOutputDesc("y", tensor_y);
  sub_op_desc_ptr->SetType("Sub");
  ge::NodePtr sub_node = graphPtr->AddNode(sub_op_desc_ptr);
  ge::NodePtr sub_node_2 = graphPtr->AddNode(sub_op_desc_ptr);

  auto in0_anchor = node->GetInDataAnchor(0);
  auto error = ge::GraphUtils::AddEdge(sub_node->GetOutDataAnchor(0), in0_anchor);
  auto in1_anchor = node->GetInDataAnchor(1);
  error = ge::GraphUtils::AddEdge(sub_node_2->GetOutDataAnchor(0), in1_anchor);
  ASSERT_EQ(error, ge::GRAPH_SUCCESS);

  dvpp::DvppOpsChecker::CheckSubAndMulFusedIntoNormalizeV2(node);
}