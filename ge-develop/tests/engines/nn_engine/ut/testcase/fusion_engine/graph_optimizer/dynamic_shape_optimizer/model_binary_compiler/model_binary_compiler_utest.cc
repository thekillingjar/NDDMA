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

#define protected public
#define private public
#include "graph_optimizer/fe_graph_optimizer.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "common/configuration.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/compute_graph.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "common/util/op_info_util.h"
#include "all_ops_stub.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include <graph_optimizer/fe_graph_optimizer.h>
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "ops_store/sub_op_info_store.h"
#include "ops_store/ops_kernel_manager.h"
#include <fusion_rule_manager/fusion_rule_data/fusion_rule_pattern.h>
#include "graph_optimizer/graph_fusion/graph_replace.h"
#include "./ge_context.h"
#include "./ge_local_context.h"

#include "graph_optimizer/heavy_format_propagation/heavy_format_propagation.h"
#include "ge/ge_api_types.h"
#include "common/lxfusion_json_util.h"
#include "adapter/common/op_store_adapter_manager.h"

#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "common/graph/fe_graph_utils.h"
#include "common/configuration.h"
#include <vector>
#include "common/fe_inner_attr_define.h"
#include "ge/ge_api_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/tuning_utils.h"
#include "graph/node.h"
#include "graph_optimizer/dynamic_shape_optimizer/model_binary_compiler/model_binary_compiler.h"
#include "graph_constructor.h"

#undef protected
#undef private

using namespace testing;
using namespace ge;
using namespace fe;

class UTEST_fusion_engine_model_binary_compiler : public testing::Test {
public:
  static std::map<std::string, ge::OpKernelBinPtr> OpKernelBinMap_;

protected:
  void SetUp()
  {

  }

  void TearDown() {

  }

  static void CreateOmSubGraph(ComputeGraphPtr &sub_graph, ge::NodePtr &return_node, bool flag) {
    std::string graph_name = sub_graph->GetName();
    OpDescPtr op1 = std::make_shared<OpDesc>("op1_" + graph_name, "op1");
    OpDescPtr op2 = std::make_shared<OpDesc>("op2_" + graph_name, "op2");
    OpDescPtr op3 = std::make_shared<OpDesc>("op3_" + graph_name, "Matmul");
    OpDescPtr data0 = std::make_shared<OpDesc>("DATA0_" + graph_name, fe::DATA);
    OpDescPtr data1 = std::make_shared<OpDesc>("DATA1_" + graph_name, fe::DATA);
    OpDescPtr const_op = std::make_shared<OpDesc>("const_" + graph_name, fe::CONSTANT);
    OpDescPtr net_output_op = std::make_shared<OpDesc>("net_output_" + graph_name, fe::NETOUTPUT);

    // add descriptor
    vector<int64_t> dims = {1, 2, 3, 32};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_ND);
    in_desc1.SetOriginFormat(FORMAT_ND);
    in_desc1.SetDataType(DT_FLOAT16);
    in_desc1.SetOriginShape(shape);
    op1->AddInputDesc("x1", in_desc1);
    op2->AddInputDesc("x2", in_desc1);
    op2->AddInputDesc("x3", in_desc1);
    op3->AddInputDesc("x4", in_desc1);
    op3->AddInputDesc("x5", in_desc1);
    data0->AddOutputDesc("x1", in_desc1);
    data0->AddInputDesc(in_desc1);
    data1->AddOutputDesc("x2", in_desc1);
    data1->AddInputDesc(in_desc1);
    const_op->AddOutputDesc("x3", in_desc1);
    const_op->AddInputDesc(in_desc1);
    net_output_op->AddInputDesc("y1", in_desc1);
    net_output_op->AddInputDesc("y2", in_desc1);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_NHWC);
    out_desc2.SetOriginFormat(FORMAT_NHWC);
    out_desc2.SetDataType(DT_FLOAT16);
    op1->AddOutputDesc("y1", out_desc2);
    op2->AddOutputDesc("y2", out_desc2);
    op3->AddOutputDesc("y3", out_desc2);
    op3->AddOutputDesc("y4", out_desc2);

    std::vector<std::string> variable_attr = {"attr1", "attr2"};
    (void)ge::AttrUtils::SetListStr(op2, "variable_attr", variable_attr);
    (void)ge::AttrUtils::SetBool(const_op, "_binary_attr_required", true);
    if (flag) {
      (void)ge::AttrUtils::SetInt(data0, ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
      (void)ge::AttrUtils::SetInt(data1, ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
    } else {
      (void)ge::AttrUtils::SetInt(data0, ge::ATTR_NAME_PARENT_NODE_INDEX, -1);
      (void)ge::AttrUtils::SetInt(data1, ge::ATTR_NAME_PARENT_NODE_INDEX, -1);
    }
    (void)ge::AttrUtils::SetBool(op3, "transpose_x1", false);
    (void)ge::AttrUtils::SetBool(op3, "transpose_x2", false);
    (void)ge::AttrUtils::SetInt(op3, "offset_x", 0);

    NodePtr op1_node = sub_graph->AddNode(op1);
    NodePtr op2_node = sub_graph->AddNode(op2);
    return_node = sub_graph->AddNode(op3);
    NodePtr data0_node = sub_graph->AddNode(data0);
    NodePtr data1_node = sub_graph->AddNode(data1);
    NodePtr const_node = sub_graph->AddNode(const_op);
    NodePtr net_output_node = sub_graph->AddNode(net_output_op);

    uint8_t* data = nullptr;
    size_t data_len = 8;
    vector<ge::GeTensorPtr> weigths;
    weigths.push_back(std::make_shared<ge::GeTensor>(in_desc1, data, data_len));
    ge::OpDescUtils::SetWeights(const_node, weigths);

    GraphUtils::AddEdge(data0_node->GetOutDataAnchor(0), op1_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), op2_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), op2_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(op1_node->GetOutDataAnchor(0), return_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(op2_node->GetOutDataAnchor(0), return_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(return_node->GetOutDataAnchor(0), net_output_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(return_node->GetOutDataAnchor(1), net_output_node->GetInDataAnchor(1));

    const auto in_tensor_desc0 = net_output_op->MutableInputDesc(0);
    const auto in_tensor_desc1 = net_output_op->MutableInputDesc(1);
    if (flag) {
      (void)ge::AttrUtils::SetInt(in_tensor_desc0, ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
      (void)ge::AttrUtils::SetInt(in_tensor_desc1, ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
    } else {
      (void)ge::AttrUtils::SetInt(in_tensor_desc0, ge::ATTR_NAME_PARENT_NODE_INDEX, -1);
      (void)ge::AttrUtils::SetInt(in_tensor_desc1, ge::ATTR_NAME_PARENT_NODE_INDEX, -1);
    }

    std::vector<std::string> kernel_prefix_list = {"_mix_aic", "_mix_aiv"};
    (void)ge::AttrUtils::SetListStr(op3, kKernelNamesPrefix, kernel_prefix_list);
    (void)ge::AttrUtils::SetStr(op1, fe::kKernelName, op1->GetName() + "_kernelName");
    (void)ge::AttrUtils::SetStr(op2, fe::kKernelName, op2->GetName() + "_kernelName");
    (void)ge::AttrUtils::SetStr(op2, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_OM");
    (void)ge::AttrUtils::SetStr(op3, "_mix_aic" + ge::ATTR_NAME_TBE_KERNEL_NAME, "_mix_aic" + op3->GetName() + "_kernelName");
    (void)ge::AttrUtils::SetStr(op3, "_mix_aiv" + ge::ATTR_NAME_TBE_KERNEL_NAME, "_mix_aiv" + op3->GetName() + "_kernelName");

    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr1 = std::make_shared<ge::OpKernelBin>(op1_node->GetName(), std::move(buffer));
    ge::OpKernelBinPtr tbe_kernel_ptr2 = std::make_shared<ge::OpKernelBin>(op2_node->GetName(), std::move(buffer));
    ge::OpKernelBinPtr tbe_kernel_ptr3 = std::make_shared<ge::OpKernelBin>(return_node->GetName(), std::move(buffer));
    OpKernelBinMap_[op1->GetName() + "_kernelName"] = tbe_kernel_ptr1;
    OpKernelBinMap_[op2->GetName() + "_kernelName"] = tbe_kernel_ptr2;
    OpKernelBinMap_["_mix_aic" + op3->GetName() + "_kernelName"] = tbe_kernel_ptr3;
    OpKernelBinMap_["_mix_aiv" + op3->GetName() + "_kernelName"] = tbe_kernel_ptr3;
  }

  static void SetSubgraphInfo(ComputeGraphPtr &parent_graph, ge::NodePtr &parent_node, ComputeGraphPtr &sub_graph) {
    sub_graph->SetParentGraph(parent_graph);
    sub_graph->SetParentNode(parent_node);
    parent_node->GetOpDesc()->AddSubgraphName(sub_graph->GetName());
    parent_node->GetOpDesc()->SetSubgraphInstanceName(0, sub_graph->GetName());
  }

  static void CreateTensorSliceInfo(ffts::ThreadSliceMapPtr &slice_info_ptr) {
    slice_info_ptr->thread_mode = 0;
    ffts::DimRange dim_rang1;
    dim_rang1.higher = 1;
    dim_rang1.lower = 0;
    ffts::DimRange dim_rang2;
    dim_rang2.higher = 20;
    dim_rang2.lower = 10;
    slice_info_ptr->ori_input_tensor_shape = {{{{1,2,3,4}}, {{5,6,7,8}}}, {{{1,3,5,7}}, {{2,4,6,8}}}};
    slice_info_ptr->ori_output_tensor_shape = {{{{10,20,30,40}}, {{50,60,70,80}}}, {{{10,30,50,70}}, {{20,40,60,80}}}};
    slice_info_ptr->input_tensor_slice = {{{{dim_rang1, dim_rang1}}, {{dim_rang2, dim_rang2}}},
                                          {{{dim_rang1, dim_rang1}}, {{dim_rang2, dim_rang2}}}};
    slice_info_ptr->output_tensor_slice = {{{{dim_rang1, dim_rang1}}, {{dim_rang2, dim_rang2}}},
                                           {{{dim_rang1, dim_rang1}}, {{dim_rang2, dim_rang2}}}};
    slice_info_ptr->ori_input_tensor_slice = {{{{dim_rang1, dim_rang1}}, {{dim_rang2, dim_rang2}}},
                                              {{{dim_rang1, dim_rang1}}, {{dim_rang2, dim_rang2}}}};
    slice_info_ptr->ori_output_tensor_slice = {{{{dim_rang1, dim_rang1}}, {{dim_rang2, dim_rang2}}},
                                               {{{dim_rang1, dim_rang1}}, {{dim_rang2, dim_rang2}}}};
    slice_info_ptr->input_axis = {3,5};
    slice_info_ptr->output_axis = {4,6};
    slice_info_ptr->input_tensor_indexes = {0,1};
    slice_info_ptr->output_tensor_indexes = {0,1};
  }

  static void CreateWholeGraph(ComputeGraphPtr &root_graph, ComputeGraphPtr &om_graph, ComputeGraphPtr &om_sub_graph, bool flag) {
    ge::NodePtr root_func_node;
    CreateOmSubGraph(root_graph, root_func_node, flag);
    ge::NodePtr om_func_node;
    CreateOmSubGraph(om_graph, om_func_node, flag);
    ge::NodePtr om_sub_func_node;
    CreateOmSubGraph(om_sub_graph, om_sub_func_node, flag);

    auto op_desc = om_func_node->GetOpDesc();
    ge::OpDescUtilsEx::SetType(op_desc, "PartitionedCall");
    op_desc->SetName("Function_op1");
    SetSubgraphInfo(root_graph, root_func_node, om_graph);
    root_graph->AddSubgraph(om_graph->GetName(), om_graph);
    SetSubgraphInfo(om_graph, om_func_node, om_sub_graph);
    root_graph->AddSubgraph(om_sub_graph->GetName(), om_sub_graph);

    (void)ge::AttrUtils::SetInt(root_func_node->GetOpDesc(), "attr1", 100);
    (void)ge::AttrUtils::SetBool(root_func_node->GetOpDesc(), "attr2", true);
    (void)ge::AttrUtils::SetStr(root_func_node->GetOpDesc(), ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_OM");
    (void)ge::AttrUtils::SetStr(om_func_node->GetOpDesc(), ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_OM");
    (void)ge::AttrUtils::SetStr(om_sub_func_node->GetOpDesc(), ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_OM");

    ffts::ThreadSliceMapPtr slice_info_ptr = make_shared<ffts::ThreadSliceMap>();
    CreateTensorSliceInfo(slice_info_ptr);
    (void)root_func_node->GetOpDesc()->SetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  }

  static ge::OpKernelBinPtr GetOpKernelBinByKernelName(const std::string &kernel_name) {
    auto it = OpKernelBinMap_.find(kernel_name);
    if (it != OpKernelBinMap_.end()) {
      return it->second;
    }
    return nullptr;
  }
};

std::map<std::string, ge::OpKernelBinPtr> UTEST_fusion_engine_model_binary_compiler::OpKernelBinMap_;

std::vector<std::vector<std::vector<std::vector<int64_t>>>> GetVectorDimRange(std::vector<std::vector<std::vector<ffts::DimRange>>> tensor_slice) {
  std::vector<std::vector<std::vector<std::vector<int64_t>>>> vvvv;
  for (size_t i = 0; i < tensor_slice.size(); ++i) {
    std::vector<std::vector<std::vector<int64_t>>> vvv;
    for (size_t j = 0; j < tensor_slice[i].size(); ++j) {
      std::vector<std::vector<int64_t>> vv;
      for (size_t k = 0; k < tensor_slice[i][j].size(); ++k) {
        std::vector<int64_t> v;
        v.emplace_back(tensor_slice[i][j][k].higher);
        v.emplace_back(tensor_slice[i][j][k].lower);
        vv.emplace_back(v);
      }
      vvv.emplace_back(vv);
    }
    vvvv.emplace_back(vvv);
  }
  return vvvv;
}

std::string TensorSliceInfoToStr(ffts::ThreadSliceMapPtr &slice_info_ptr) {
  nlohmann::json tensor_slice_info;
  tensor_slice_info["ori_input_tensor_shape"] = slice_info_ptr->ori_input_tensor_shape;
  tensor_slice_info["ori_output_tensor_shape"] = slice_info_ptr->ori_output_tensor_shape;
  tensor_slice_info["input_axis"] = slice_info_ptr->input_axis;
  tensor_slice_info["output_axis"] = slice_info_ptr->output_axis;
  tensor_slice_info["input_tensor_indexes"] = slice_info_ptr->input_tensor_indexes;
  tensor_slice_info["output_tensor_indexes"] = slice_info_ptr->output_tensor_indexes;
  tensor_slice_info["input_tensor_slice"] = GetVectorDimRange(slice_info_ptr->input_tensor_slice);
  tensor_slice_info["output_tensor_slice"] = GetVectorDimRange(slice_info_ptr->output_tensor_slice);
  tensor_slice_info["ori_input_tensor_slice"] = GetVectorDimRange(slice_info_ptr->ori_input_tensor_slice);
  tensor_slice_info["ori_output_tensor_slice"] = GetVectorDimRange(slice_info_ptr->ori_output_tensor_slice);
  return tensor_slice_info.dump();
}

TEST_F(UTEST_fusion_engine_model_binary_compiler, update_node_info_in_omSubGraph_success) {
  ComputeGraphPtr root_graph = std::make_shared<ComputeGraph>("root_graph");
  ComputeGraphPtr om_graph = std::make_shared<ComputeGraph>("om_graph");
  ComputeGraphPtr om_sub_graph = std::make_shared<ComputeGraph>("om_sub_graph");
  CreateWholeGraph(root_graph, om_graph, om_sub_graph, true);

  ModelBinaryCompilerPtr model_binary_compiler_ptr = std::make_shared<ModelBinaryCompiler>();
  Status ret = model_binary_compiler_ptr->UpdateNodeInfoInOmSubGraph(*(om_graph.get()), GetOpKernelBinByKernelName);
  EXPECT_EQ(ret, fe::SUCCESS);

  for (const auto &node : om_graph->GetAllNodes()) {
    if (node->GetType() == CONSTANT) {
      const std::vector<ge::GeTensorPtr> weights = ge::OpDescUtils::MutableWeights(node);
      ge::GeTensorPtr const_tensor_ptr = weights[0];
      uint8_t *data = const_cast<uint8_t *>(const_tensor_ptr->GetData().GetData());
      size_t data_size = const_tensor_ptr->GetData().GetSize() / sizeof(uint8_t);
      for (size_t i = 0; i < data_size; i++) {
        std::cout << "i:" << i << " = " << data[i] <<std::endl;
      }
    }

    if (node->GetType() == NETOUTPUT) {
      const auto input_tensor_desc = node->GetOpDesc()->MutableInputDesc(0);
      EXPECT_EQ(input_tensor_desc->GetFormat(), FORMAT_NHWC);
      EXPECT_EQ(input_tensor_desc->GetOriginFormat(), FORMAT_NHWC);
    }

    if (node->GetName() == "op2_om_graph") {
      int32_t attr1 = -1;
      (void)ge::AttrUtils::GetInt(node->GetOpDesc(), "attr1", attr1);
      EXPECT_EQ(attr1, 100);
      bool attr2 = false;
      (void)ge::AttrUtils::GetBool(node->GetOpDesc(), "attr2", attr2);
      EXPECT_EQ(attr2, true);
      ffts::DimRange dim_rang2;
      dim_rang2.higher = 20;
      dim_rang2.lower = 10;
      ffts::ThreadSliceMapPtr ori_slice_info_ptr = nullptr;
      FE_MAKE_SHARED(ori_slice_info_ptr = std::make_shared<ffts::ThreadSliceMap>(), return);
      ori_slice_info_ptr->ori_input_tensor_shape = {{{{5,6,7,8}}}, {{{2,4,6,8}}}};
      ori_slice_info_ptr->input_tensor_slice = {{{{dim_rang2, dim_rang2}}}, {{{dim_rang2, dim_rang2}}}};
      ori_slice_info_ptr->ori_input_tensor_slice = {{{{dim_rang2, dim_rang2}}}, {{{dim_rang2, dim_rang2}}}};
      ori_slice_info_ptr->input_axis = {5};
      ori_slice_info_ptr->input_tensor_indexes = {1};
      ffts::ThreadSliceMapPtr slice_info_ptr = nullptr;
      slice_info_ptr = node->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
      EXPECT_NE(slice_info_ptr, nullptr);

      bool res = (TensorSliceInfoToStr(ori_slice_info_ptr) == TensorSliceInfoToStr(slice_info_ptr));
      EXPECT_EQ(res, true);
    }

    if (node->GetName() == "op1_om_graph") {
      ffts::DimRange dim_rang1;
      dim_rang1.higher = 1;
      dim_rang1.lower = 0;
      ffts::ThreadSliceMapPtr ori_slice_info_ptr = nullptr;
      FE_MAKE_SHARED(ori_slice_info_ptr = std::make_shared<ffts::ThreadSliceMap>(), return);
      ori_slice_info_ptr->ori_input_tensor_shape = {{{{1,2,3,4}}}, {{{1,3,5,7}}}};
      ori_slice_info_ptr->input_tensor_slice = {{{{dim_rang1, dim_rang1}}}, {{{dim_rang1, dim_rang1}}}};
      ori_slice_info_ptr->ori_input_tensor_slice = {{{{dim_rang1, dim_rang1}}}, {{{dim_rang1, dim_rang1}}}};
      ori_slice_info_ptr->input_axis = {3};
      ori_slice_info_ptr->input_tensor_indexes = {0};
      ffts::ThreadSliceMapPtr slice_info_ptr = nullptr;
      slice_info_ptr = node->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
      EXPECT_NE(slice_info_ptr, nullptr);

      bool res = (TensorSliceInfoToStr(ori_slice_info_ptr) == TensorSliceInfoToStr(slice_info_ptr));
      EXPECT_EQ(res, true);
    }

    if (node->GetName() == "op3_om_graph") {
      ffts::DimRange dim_rang1;
      dim_rang1.higher = 1;
      dim_rang1.lower = 0;
      ffts::DimRange dim_rang2;
      dim_rang2.higher = 20;
      dim_rang2.lower = 10;
      ffts::ThreadSliceMapPtr ori_slice_info_ptr = nullptr;
      FE_MAKE_SHARED(ori_slice_info_ptr = std::make_shared<ffts::ThreadSliceMap>(), return);
      ori_slice_info_ptr->ori_output_tensor_shape = {{{{10,20,30,40}}, {{50,60,70,80}}}, {{{10,30,50,70}}, {{20,40,60,80}}}};
      ori_slice_info_ptr->output_tensor_slice = {{{{dim_rang1, dim_rang1}}, {{dim_rang2, dim_rang2}}},
                                                 {{{dim_rang1, dim_rang1}}, {{dim_rang2, dim_rang2}}}};
      ori_slice_info_ptr->ori_output_tensor_slice = {{{{dim_rang1, dim_rang1}}, {{dim_rang2, dim_rang2}}},
                                                     {{{dim_rang1, dim_rang1}}, {{dim_rang2, dim_rang2}}}};
      ori_slice_info_ptr->output_axis = {4,6};
      ori_slice_info_ptr->output_tensor_indexes = {0,1};
      ffts::ThreadSliceMapPtr slice_info_ptr = nullptr;
      slice_info_ptr = node->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
      EXPECT_NE(slice_info_ptr, nullptr);

      bool res = (TensorSliceInfoToStr(ori_slice_info_ptr) == TensorSliceInfoToStr(slice_info_ptr));
      EXPECT_EQ(res, true);
    }
  }
}

TEST_F(UTEST_fusion_engine_model_binary_compiler, update_axis_and_tensor_index) {
  ModelBinaryCompilerPtr model_binary_compiler_ptr = std::make_shared<ModelBinaryCompiler>();
  std::set<uint32_t> index;
  std::vector<uint32_t> tensor_indexes = {1, 2};
  std::vector<uint32_t> axis = {1};
  std::vector<uint32_t> new_tensor_indexes;
  std::vector<uint32_t> new_axis;
  Status ret = model_binary_compiler_ptr->UpdateAxisAndTensorIndex(index, tensor_indexes, axis, new_tensor_indexes, new_axis);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(UTEST_fusion_engine_model_binary_compiler, update_node_info_in_omSubGraph_fail) {
  ComputeGraphPtr root_graph = std::make_shared<ComputeGraph>("root_graph");
  ComputeGraphPtr om_graph = std::make_shared<ComputeGraph>("om_graph");
  ComputeGraphPtr om_sub_graph = std::make_shared<ComputeGraph>("om_sub_graph");
  CreateWholeGraph(root_graph, om_graph, om_sub_graph, false);
  ModelBinaryCompilerPtr model_binary_compiler_ptr = std::make_shared<ModelBinaryCompiler>();
  Status ret = model_binary_compiler_ptr->UpdateNodeInfoInOmSubGraph(*(om_graph.get()), GetOpKernelBinByKernelName);
  EXPECT_EQ(ret, fe::FAILED);
}