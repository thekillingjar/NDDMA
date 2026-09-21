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
#include <iostream>

#include <list>
#define protected public
#define private public
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "common/graph_comm.h"
#include "common/fusion_op_comm.h"
#include "graph/utils/graph_utils.h"
#include "common/configuration.h"
#include "common/fe_inner_attr_define.h"
#include "common/util/constants.h"
#include "graph/op_kernel_bin.h"
#include "common/graph/fe_graph_utils.h"

#undef private
#undef protected

using namespace std;
using namespace fe;
using namespace ge;

class graph_comm_ut: public testing::Test
{
public:

protected:

  void SetUp() {
  }

  void TearDown()
  {

  }

  static ComputeGraphPtr CreateEmptyOriginGraph() {
      ComputeGraphPtr graph = std::make_shared<ComputeGraph>("empty_graph");
      return graph;
  }

  static ComputeGraphPtr CreateOriginGraph() {
      ComputeGraphPtr graph = std::make_shared<ComputeGraph>("origin_graph");
      std::string session_graph_id = "123456";
      //session_graph_id
      ge::AttrUtils::SetStr(graph, "session_graph_id", session_graph_id);
      // Node
      OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
      OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Activation");

      // add descriptor
      vector<int64_t> dims = {1,2,3,4};
      GeShape shape(dims);

      GeTensorDesc in_desc1(shape);
      in_desc1.SetOriginFormat(FORMAT_NCHW);
      in_desc1.SetFormat(FORMAT_NCHW);
      in_desc1.SetDataType(DT_FLOAT16);
      relu_op->AddInputDesc("x", in_desc1);

      GeTensorDesc out_desc1(shape);
      out_desc1.SetOriginFormat(FORMAT_HWCN);
      out_desc1.SetFormat(FORMAT_HWCN);
      out_desc1.SetDataType(DT_FLOAT16);
      relu_op->AddOutputDesc("y", out_desc1);


      GeTensorDesc in_desc2(shape);
      in_desc2.SetOriginFormat(FORMAT_FRACTAL_Z);
      in_desc2.SetFormat(FORMAT_FRACTAL_Z);
      in_desc2.SetDataType(DT_FLOAT16);
      bn_op->AddInputDesc("x", in_desc2);

      GeTensorDesc out_desc2(shape);
      out_desc2.SetOriginFormat(FORMAT_NHWC);
      out_desc2.SetFormat(FORMAT_NHWC);
      out_desc2.SetDataType(DT_FLOAT16);
      bn_op->AddOutputDesc("y", out_desc2);

      NodePtr bn_node = graph->AddNode(bn_op);
      bn_node->AddSendEventId(123);
      bn_node->AddSendEventId(234);
      bn_node->AddSendEventId(345);
      NodePtr relu_node = graph->AddNode(relu_op);
      relu_node->AddRecvEventId(234);
      relu_node->AddRecvEventId(345);
      ge::GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
      return graph;
  }

  static ComputeGraphPtr CreateOriginGraph2() {
      ComputeGraphPtr graph = std::make_shared<ComputeGraph>("origin_graph");
      std::string session_graph_id = "123456";
      //session_graph_id
      ge::AttrUtils::SetStr(graph, "session_graph_id", session_graph_id);
      // Node
      OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
      OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Activation");
      OpDescPtr netout_op = std::make_shared<OpDesc>("netoutput", "NetOutput");

      // add descriptor
      vector<int64_t> dims = {1,2,3,4};
      GeShape shape(dims);

      GeTensorDesc in_desc1(shape);
      in_desc1.SetOriginFormat(FORMAT_NCHW);
      in_desc1.SetFormat(FORMAT_NCHW);
      in_desc1.SetDataType(DT_FLOAT16);
      relu_op->AddInputDesc("x", in_desc1);

      GeTensorDesc out_desc1(shape);
      out_desc1.SetOriginFormat(FORMAT_HWCN);
      out_desc1.SetFormat(FORMAT_HWCN);
      out_desc1.SetDataType(DT_FLOAT16);
      relu_op->AddOutputDesc("y", out_desc1);


      GeTensorDesc in_desc2(shape);
      in_desc2.SetOriginFormat(FORMAT_FRACTAL_Z);
      in_desc2.SetFormat(FORMAT_FRACTAL_Z);
      in_desc2.SetDataType(DT_FLOAT16);
      bn_op->AddInputDesc("x", in_desc2);

      GeTensorDesc out_desc2(shape);
      out_desc2.SetOriginFormat(FORMAT_NHWC);
      out_desc2.SetFormat(FORMAT_NHWC);
      out_desc2.SetDataType(DT_FLOAT16);
      bn_op->AddOutputDesc("y", out_desc2);

      GeTensorDesc in_desc3(shape);
      in_desc3.SetOriginFormat(FORMAT_FRACTAL_Z);
      in_desc3.SetFormat(FORMAT_FRACTAL_Z);
      in_desc3.SetDataType(DT_FLOAT16);
      netout_op->AddInputDesc("x", in_desc3);

      GeTensorDesc out_desc3(shape);
      out_desc3.SetOriginFormat(FORMAT_NHWC);
      out_desc3.SetFormat(FORMAT_NHWC);
      out_desc3.SetDataType(DT_FLOAT16);
      netout_op->AddOutputDesc("y", out_desc3);

      NodePtr bn_node = graph->AddNode(bn_op);
      NodePtr relu_node = graph->AddNode(relu_op);
      NodePtr netout_node = graph->AddNode(netout_op);
      ge::GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
      ge::GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), netout_node->GetInDataAnchor(0));
      return graph;
  }
// AUTO GEN PLEASE DO NOT MODIFY IT
};

TEST_F(graph_comm_ut, set_Batch_bind_only) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr conv_desc = std::make_shared<OpDesc>("conv", "Conv2D");
  ge::AttrUtils::SetInt(conv_desc, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
  AttrUtils::SetBool(conv_desc, "_is_n_batch_split", true);
  NodePtr conv_node = graph->AddNode(conv_desc);
  const char tbe_bin[] = "tbe_bin";
  vector<char> buffer(tbe_bin, tbe_bin+strlen(tbe_bin));
  ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
          conv_node->GetName(), std::move(buffer));
  conv_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);
  auto fusion_op_comm = std::make_shared<FusionOpComm>();
  vector<ge::NodePtr> fus_nodelist;
  fus_nodelist.push_back(conv_node);
  OpDescPtr new_conv_desc = std::shared_ptr<OpDesc>(new (std::nothrow) OpDesc());
  fusion_op_comm->SetTBEFusionOp(fus_nodelist, new_conv_desc, "engineName", "");
  bool is_batch = false;
  AttrUtils::GetBool(conv_desc, "_is_n_batch_split", is_batch);
  EXPECT_EQ(is_batch, true);
}

TEST_F(graph_comm_ut, convarage) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr conv_desc = std::make_shared<OpDesc>("conv", "Conv2D");
  ge::AttrUtils::SetInt(conv_desc, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
  string cur_type = "AiCore";
  (void)AttrUtils::SetStr(conv_desc, "_sgt_cube_vector_core_type", cur_type);
  NodePtr conv_node = graph->AddNode(conv_desc);
  const char tbe_bin[] = "tbe_bin";
  vector<char> buffer(tbe_bin, tbe_bin+strlen(tbe_bin));
  ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
          conv_node->GetName(), std::move(buffer));
  conv_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);
  auto fusion_op_comm = std::make_shared<FusionOpComm>();
  vector<ge::NodePtr> fus_nodelist;
  fus_nodelist.push_back(conv_node);
  EXPECT_EQ(fus_nodelist.size(), 1);
  OpDescPtr new_conv_desc = std::shared_ptr<OpDesc>(new (std::nothrow) OpDesc());
  fusion_op_comm->SetTBEFusionOp(fus_nodelist, new_conv_desc, "engineName", "");
}

TEST_F(graph_comm_ut, set_buffer_fusion_info) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr conv_desc = std::make_shared<OpDesc>("conv", "Conv2D");
  ge::AttrUtils::SetInt(conv_desc, ge::ATTR_NAME_IMPLY_TYPE,static_cast<int64_t>(domi::ImplyType::TVM));
  AttrUtils::SetBool(conv_desc, "is_nbatch_spilt", true);
  NodePtr conv_node = graph->AddNode(conv_desc);
  const char tbe_bin[] = "tbe_bin";
  vector<char> buffer(tbe_bin, tbe_bin+strlen(tbe_bin));
  ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
          conv_node->GetName(), std::move(buffer));
  conv_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);
  ge::AttrUtils::SetInt(conv_desc, "l2_fusion_group_id", 1);
  ge::AttrUtils::SetBool(conv_desc, NEED_RE_PRECOMPILE, true);
  ge::AttrUtils::SetStr(conv_desc, ge::ATTR_NAME_FUSION_GROUP_TYPE, "ut_test");
  ge::AttrUtils::SetStr(conv_desc, "continuous_stream_label","L2StreamLabel");
  std::vector<int32_t>  l2_output_offset;
  l2_output_offset.push_back(1);
  ge::AttrUtils::SetListInt(conv_desc, "output_offset_for_buffer_fusion", l2_output_offset);
  ge::AttrUtils::SetInt(conv_desc, ge::ATTR_NAME_GET_TENSOR_ACTUAL_SIZE, 1);
  ge::AttrUtils::SetBool(conv_desc, ge::ATTR_NO_TASK_AND_DUMP_NEEDED, true);
  auto fusion_op_comm = std::make_shared<FusionOpComm>();
  vector<ge::NodePtr> fus_nodelist;
  fus_nodelist.push_back(conv_node);
  OpDescPtr new_conv_desc = std::shared_ptr<OpDesc>(new (std::nothrow) OpDesc());
  fusion_op_comm->SetTBEFusionOp(fus_nodelist, new_conv_desc, "engineName", "");
  bool l2_optimize = false;
  AttrUtils::GetBool(new_conv_desc, NEED_RE_PRECOMPILE, l2_optimize);
  EXPECT_EQ(l2_optimize, true);
  int32_t output_real_calc_flag = 0;
  AttrUtils::GetInt(new_conv_desc, ge::ATTR_NAME_GET_TENSOR_ACTUAL_SIZE, output_real_calc_flag);
  EXPECT_EQ(output_real_calc_flag, 1);
}

TEST_F(graph_comm_ut, set_l1_fusion_info) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr conv_desc = std::make_shared<OpDesc>("conv", "Conv2D");
  ge::AttrUtils::SetInt(conv_desc, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
  AttrUtils::SetBool(conv_desc, "is_nbatch_spilt", true);
  NodePtr conv_node = graph->AddNode(conv_desc);
  const char tbe_bin[] = "tbe_bin";
  vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
  ge::OpKernelBinPtr tbe_kernel_ptr =
      std::make_shared<ge::OpKernelBin>(conv_node->GetName(), std::move(buffer));
  conv_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);
  ge::AttrUtils::SetInt(conv_desc, "_l1_fusion_group_id", 1);
  ge::AttrUtils::SetBool(conv_desc, NEED_RE_PRECOMPILE, true);
  ge::AttrUtils::SetStr(conv_desc, ge::ATTR_NAME_FUSION_GROUP_TYPE, "ut_test");
  ge::AttrUtils::SetStr(conv_desc, "continuous_stream_label", "L2StreamLabel");
  std::vector<int32_t> l1_output_offset;
  l1_output_offset.push_back(1);
  ge::AttrUtils::SetListInt(conv_desc, "output_offset_for_buffer_fusion",
                            l1_output_offset);
  ge::AttrUtils::SetInt(conv_desc, ge::ATTR_NAME_GET_TENSOR_ACTUAL_SIZE, 1);
  ge::AttrUtils::SetStr(conv_desc, "_L1_fusion_sub_graph_no", "1");
  auto fusion_op_comm = std::make_shared<FusionOpComm>();
  vector<ge::NodePtr> fus_nodelist;
  fus_nodelist.push_back(conv_node);
  OpDescPtr new_conv_desc = std::shared_ptr<OpDesc>(new (std::nothrow) OpDesc());
  Configuration::Instance(fe::AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
          static_cast<int64_t>(EN_L1_OPTIMIZE);
  fusion_op_comm->SetTBEFusionOp(fus_nodelist, new_conv_desc, "engineName", "");
  bool l1_optimize = false;
  AttrUtils::GetBool(new_conv_desc, NEED_RE_PRECOMPILE, l1_optimize);
  EXPECT_EQ(l1_optimize, true);
  int32_t output_real_calc_flag = 0;
  AttrUtils::GetInt(new_conv_desc, ge::ATTR_NAME_GET_TENSOR_ACTUAL_SIZE,
                    output_real_calc_flag);
  EXPECT_EQ(output_real_calc_flag, 1);
}
