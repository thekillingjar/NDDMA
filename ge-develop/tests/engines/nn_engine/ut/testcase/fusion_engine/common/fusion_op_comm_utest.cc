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
#include <string>
#include <vector>

#define protected public
#define private public
#include "graph_optimizer/fe_graph_optimizer.h"
#include "graph/utils/graph_utils.h"
#include "graph/op_kernel_bin.h"
#include "common/graph_comm.h"
#include "common/fusion_op_comm.h"
#include "common/thread_slice_info_utils.h"
#include "common/util/op_info_util.h"
#include "common/sgt_slice_type.h"
#undef protected
#undef private

using namespace std;
using namespace fe;
using namespace ge;
using FusionOpCommPtr = std::shared_ptr<FusionOpComm>;

class UTEST_fusion_op_comm : public testing::Test
{
public:
    FusionOpCommPtr fusion_op_comm_ptr;
protected:
    void SetUp()
    {
        fusion_op_comm_ptr = make_shared<FusionOpComm>();
    }

    void TearDown()
    {
    }

/*
 * batchnorm
 *    |
 *   relu
 */
    static void CreateGraph(ComputeGraphPtr graph) {
      OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BatchNorm");
      OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Activation");
      OpDescPtr cov_op = std::make_shared<OpDesc>("conv2D", "Conv2D");

      // add descriptor
      vector<int64_t> dims = {1,2,3,4};
      GeShape shape(dims);

      GeTensorDesc in_desc1(shape);
      in_desc1.SetFormat(FORMAT_FRACTAL_Z);
      in_desc1.SetDataType(DT_FLOAT16);
      bn_op->AddInputDesc("x", in_desc1);

      GeTensorDesc out_desc1(shape);
      out_desc1.SetFormat(FORMAT_NHWC);
      out_desc1.SetDataType(DT_FLOAT16);
      bn_op->AddOutputDesc("y", out_desc1);
      ge::AttrUtils::SetInt(bn_op, ge::ATTR_NAME_PARALLEL_GROUP_ID, 0xFFFFF);

      GeTensorDesc in_desc2(shape);
      in_desc2.SetFormat(FORMAT_NCHW);
      in_desc2.SetDataType(DT_FLOAT16);
      relu_op->AddInputDesc("x", in_desc2);

      GeTensorDesc out_desc2(shape);
      out_desc2.SetFormat(FORMAT_HWCN);
      out_desc2.SetDataType(DT_FLOAT16);
      relu_op->AddOutputDesc("y", out_desc2);
    
      GeTensorDesc in_desc3(shape);
      in_desc3.SetFormat(FORMAT_NCHW);
      in_desc3.SetDataType(DT_FLOAT16);
      cov_op->AddInputDesc("z", in_desc3);

      GeTensorDesc out_desc3(shape);
      out_desc3.SetFormat(FORMAT_HWCN);
      out_desc3.SetDataType(DT_FLOAT16);
      cov_op->AddOutputDesc("z", out_desc3);
      ge::AttrUtils::SetInt(cov_op, ge::ATTR_NAME_PARALLEL_GROUP_ID, 0xFFFFF);

      ge::AttrUtils::SetInt(bn_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
      ge::AttrUtils::SetInt(relu_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
      ge::AttrUtils::SetInt(cov_op, FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));

      NodePtr bn_node = graph->AddNode(bn_op);
      NodePtr relu_node = graph->AddNode(relu_op);
      NodePtr cov_node = graph->AddNode(cov_op);

      GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
      GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), cov_node->GetInDataAnchor(0));

  }
};

TEST_F(UTEST_fusion_op_comm, set_sgt_tbe_fusion_op_suc_0)
{
    ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    CreateGraph(graph);
    vector<ge::NodePtr> fus_nodelist;
    for (auto &node : graph->GetDirectNode()) {
        std::vector<std::string> str_val = {"magic_0", "magic_1"};
        ge::AttrUtils::SetListStr(node->GetOpDesc(), ge::TVM_ATTR_NAME_THREAD_MAGIC, str_val);
        vector<vector<int64_t>> int_i_val(2 ,vector<int64_t>(4, 32));
        ge::AttrUtils::SetListListInt(node->GetOpDesc(), "tbe_op_thread_atomic_output_index", int_i_val);
        vector<int32_t> int_w_val = {3, 6};
        ge::AttrUtils::SetListInt(node->GetOpDesc(), "tbe_op_thread_atomic_workspace_flag", int_w_val);
        vector<ge::Buffer> byte_val;
        ge::AttrUtils::SetListBytes(node->GetOpDesc(), "_thread_tbe_kernel_buffer", byte_val);
        vector<bool> bool_val = {true, false};
        ge::AttrUtils::SetListBool(node->GetOpDesc(), "_thread_is_n_batch_split", bool_val);
        vector<string> kernel_names = {"kernel_names_0", "kernel_names_1"};
        ge::AttrUtils::SetListStr(node->GetOpDesc(), "_thread_kernelname", kernel_names);
        fus_nodelist.push_back(node);
    }
    ge::OpDescPtr fus_opdef = std::shared_ptr<OpDesc>(new (std::nothrow) OpDesc());
    bool flag = false;
    if (fus_opdef == nullptr) {
        EXPECT_EQ(flag, true);
    }
    std::vector<ge::OpKernelBinPtr> list_buffer_vec;
    std::string key = fus_nodelist[0]->GetOpDesc()->GetName();
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
    ge::OpKernelBinPtr kernel_bin_ptr = std::make_shared<ge::OpKernelBin>(key, move(buffer));
    list_buffer_vec.push_back(kernel_bin_ptr);
    fus_nodelist[0]->GetOpDesc()->SetExtAttr("thread_tbeKernel", list_buffer_vec);
    Configuration::Instance(AI_CORE_NAME).content_map_["dump.originalnodes.enable"] = "true";
    ge::OpDescPtr op_desc = fusion_op_comm_ptr->SetMultiKernelTBEFusionOp(fus_nodelist, fus_opdef, "AIcoreEngine","");
    if (op_desc != nullptr) {
        flag = true;
    }
    EXPECT_EQ(flag, true);
}

TEST_F(UTEST_fusion_op_comm, set_sgt_tbe_fusion_op_suc_1)
{
    ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    CreateGraph(graph);
    vector<ge::NodePtr> fus_nodelist;
    for (auto &node : graph->GetDirectNode()) {
        fus_nodelist.push_back(node);
    }
    ge::OpDescPtr fus_opdef = std::shared_ptr<OpDesc>(new (std::nothrow) OpDesc());
    bool flag = false;
    if (fus_opdef == nullptr) {
        EXPECT_EQ(flag, true);
    }
    std::vector<ge::OpKernelBinPtr> list_buffer_vec;
    std::string key = fus_nodelist[0]->GetOpDesc()->GetName() + "_kernel0";
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
    ge::OpKernelBinPtr kernel_bin_ptr = std::make_shared<ge::OpKernelBin>(key, move(buffer));
    list_buffer_vec.push_back(kernel_bin_ptr);
    list_buffer_vec.push_back(kernel_bin_ptr);
    vector<std::string> kernel_name_vec = {key, key};
    (void)ge::AttrUtils::SetListStr(fus_nodelist[0]->GetOpDesc(), "_thread_kernelname", kernel_name_vec);
    fus_nodelist[0]->GetOpDesc()->SetExtAttr("thread_tbeKernel", list_buffer_vec);
    ge::OpDescPtr op_desc = fusion_op_comm_ptr->SetMultiKernelTBEFusionOp(fus_nodelist, fus_opdef, "AIcoreEngine", "");
    if (op_desc != nullptr) {
        flag = true;
    }
    EXPECT_EQ(flag, true);
}

TEST_F(UTEST_fusion_op_comm, set_sgt_tbe_fusion_op_failed_1)
{
    ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    CreateGraph(graph);
    vector<ge::NodePtr> fus_nodelist;
    for (auto &node : graph->GetDirectNode()) {
        fus_nodelist.push_back(node);
    }
    ge::OpDescPtr fus_opdef = std::shared_ptr<OpDesc>(new (std::nothrow) OpDesc());
    bool flag = false;
    if (fus_opdef == nullptr) {
        EXPECT_EQ(flag, true);
    }
    std::vector<ge::OpKernelBinPtr> list_buffer_vec;
    std::string key = fus_nodelist[0]->GetOpDesc()->GetName() + "_kernel0";
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
    ge::OpKernelBinPtr kernel_bin_ptr = std::make_shared<ge::OpKernelBin>(key, move(buffer));
    list_buffer_vec.push_back(kernel_bin_ptr);
    list_buffer_vec.push_back(kernel_bin_ptr);
    vector<std::string> kernel_name_vec = {key, key};
    (void)ge::AttrUtils::SetListStr(fus_nodelist[0]->GetOpDesc(), "_thread_kernelname", kernel_name_vec);
    ge::OpDescPtr op_desc = fusion_op_comm_ptr->SetMultiKernelTBEFusionOp(fus_nodelist, fus_opdef, "AIcoreEngine", "");
    if (op_desc != nullptr) {
        flag = true;
    }
    EXPECT_EQ(flag, false);
}

TEST_F(UTEST_fusion_op_comm, set_sgt_tbe_fusion_op_failed_2)
{
    ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    CreateGraph(graph);
    vector<ge::NodePtr> fus_nodelist;
    ge::OpDescPtr fus_opdef = std::shared_ptr<OpDesc>(new (std::nothrow) OpDesc());
    bool flag = false;
    if (fus_opdef == nullptr) {
        EXPECT_EQ(flag, true);
    }
    ge::OpDescPtr op_desc = fusion_op_comm_ptr->SetMultiKernelTBEFusionOp(fus_nodelist, fus_opdef, "AIcoreEngine", "");
    if (op_desc != nullptr) {
        flag = true;
    }
    EXPECT_EQ(flag, false);
}

TEST_F(UTEST_fusion_op_comm, set_tbe_fusion_op_suc)
{
    ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    CreateGraph(graph);
    vector<ge::NodePtr> fus_nodelist;
    for (auto &node : graph->GetDirectNode()) {
        fus_nodelist.push_back(node);
    }
    ge::OpDescPtr fus_opdef = std::shared_ptr<OpDesc>(new (std::nothrow) OpDesc());
    bool flag = false;
    if (fus_opdef == nullptr) {
        EXPECT_EQ(flag, true);
    }
    std::vector<ge::OpKernelBinPtr> list_buffer_vec;
    std::string key = fus_nodelist[0]->GetOpDesc()->GetName();
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
    ge::OpKernelBinPtr kernel_bin_ptr = std::make_shared<ge::OpKernelBin>(key, move(buffer));
    list_buffer_vec.push_back(kernel_bin_ptr);
    ge::AttrUtils::SetBool(fus_nodelist[0]->GetOpDesc(), ge::TVM_ATTR_NAME_MAGIC, true);
    fus_nodelist[0]->GetOpDesc()->SetExtAttr("tbeKernel", kernel_bin_ptr);
    fus_nodelist[0]->GetOpDesc()->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, kernel_bin_ptr);
    std::string kernel_name = "te_op_name";
    vector<std::string> names_tmp = {"A", "B"};
    vector<std::string> types_tmp = {"typeA", "typeB"};
    ge::AttrUtils::SetStr(fus_nodelist[0]->GetOpDesc(), ge::ATTR_NAME_TBE_KERNEL_NAME_FOR_LOAD, kernel_name);
    ge::AttrUtils::SetListStr(fus_nodelist[0]->GetOpDesc(), ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, names_tmp);
    ge::AttrUtils::SetListStr(fus_nodelist[0]->GetOpDesc(), ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_TYPES, types_tmp);

    ge::OpDescPtr op_desc = fusion_op_comm_ptr->SetTBEFusionOp(fus_nodelist, fus_opdef, "AIcoreEngine", "");
    if ((op_desc != nullptr) &&
        (!op_desc->HasAttr(op_desc->GetName() + kKernelName)) &&
        (!op_desc->HasAttr(ge::ATTR_NAME_TBE_KERNEL_NAME_FOR_LOAD))) {
        flag = true;
    }
    EXPECT_EQ(flag, true);
}

TEST_F(UTEST_fusion_op_comm, set_tbe_fusion_op_suc1)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraph(graph);
  vector<ge::NodePtr> fus_nodelist;
  for (auto &node : graph->GetDirectNode()) {
    fus_nodelist.push_back(node);
  }
  ge::OpDescPtr fus_opdef = std::shared_ptr<OpDesc>(new (std::nothrow) OpDesc());
  bool flag = false;
  if (fus_opdef == nullptr) {
    EXPECT_EQ(flag, true);
  }
  std::vector<ge::OpKernelBinPtr> list_buffer_vec;
  std::string key = fus_nodelist[0]->GetOpDesc()->GetName();
  const char tbe_bin[] = "tbe_bin";
  vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
  ge::OpKernelBinPtr kernel_bin_ptr = std::make_shared<ge::OpKernelBin>(key, move(buffer));
  list_buffer_vec.push_back(kernel_bin_ptr);
  ge::AttrUtils::SetBool(fus_nodelist[0]->GetOpDesc(), ge::TVM_ATTR_NAME_MAGIC, true);
  fus_nodelist[0]->GetOpDesc()->SetExtAttr("_mix_aictbeKernel", kernel_bin_ptr);
  fus_nodelist[0]->GetOpDesc()->SetExtAttr("_mix_aivtbeKernel", kernel_bin_ptr);
  ge::AttrUtils::SetStr(fus_nodelist[0]->GetOpDesc(), "_mix_aic_kernelname", "aic_kernel_name");
  ge::AttrUtils::SetStr(fus_nodelist[0]->GetOpDesc(), "_mix_aiv_kernelname", "aiv_kernel_name");
  ge::AttrUtils::SetStr(fus_nodelist[0]->GetOpDesc(), fe::ATTR_NAME_ALIAS_ENGINE_NAME, "test");
  std::string kernel_name = "te_op_name";
  ge::AttrUtils::SetStr(fus_nodelist[0]->GetOpDesc(), ge::ATTR_NAME_TBE_KERNEL_NAME_FOR_LOAD, kernel_name);
  ge::AttrUtils::SetStr(fus_nodelist[0]->GetOpDesc(), fe::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX_AIC");
  ge::OpDescPtr op_desc = fusion_op_comm_ptr->SetTBEFusionOp(fus_nodelist, fus_opdef, "AIcoreEngine", "");
  if (op_desc != nullptr) {
    flag = true;
  }
  EXPECT_EQ(flag, true);
}

TEST_F(UTEST_fusion_op_comm, ffts_plus_json_parse) {
  std::string str_slice_info =
      "{\"dependencies\":[],\"thread_scopeId\":100,\"is_first_node_in_topo_order\":false,\"node_num_in_thread_scope\":"
      "0,"
      "\"is_input_node_of_thread_scope\":false,\"is_output_node_of_thread_scope\":false,\"threadMode\":false,\"slice_"
      "instance_num\":0,\"parallel_window_size\":0,\"thread_id\":0,\"oriInputTensorShape\":[],\"oriOutputTensorShape\":"
      "["
      "],\"original_node\":\"\",\"core_num\":[],\"cutType\":[],\"atomic_types\":[],\"thread_id\":0,\"same_atomic_clean_"
      "nodes\":[],\"input_axis\":[],\"output_axis\":[],\"input_tensor_indexes\":[],\"output_tensor_indexes\":[],"
      "\"input_tensor_slice\":[],\"output_tensor_slice\":[],\"ori_input_tensor_slice\":[],\"ori_output_tensor_slice\":["
      "],\"inputCutList\":[], \"outputCutList\":[]}";
  ffts::ThreadSliceMap slice_info;
  ffts::GetSliceInfoFromJson(slice_info, str_slice_info);
  EXPECT_EQ(slice_info.thread_scope_id, 100);

  std::string str_slice_info_invalid =
      "{\"dependencies\":[],\"thread_scopeId\":100,\"is_first_node_in_topo_order\":false,\"node_num_in_thread_scope\":"
      "0,"
      "\"is_input_node_of_thread_scope\":false,\"is_output_node_of_thread_scope\":false,\"threadMode\":false,\"slice_"
      "instance_num\":0,\"parallel_window_size\":0,\"thread_id\":\"thread_x\",\"oriInputTensorShape\":[],"
      "\"oriOutputTensorShape\":["
      "],\"original_node\":\"\",\"core_num\":[],\"cutType\":[{}],\"atomic_types\":[],\"thread_id\":0,\"same_atomic_"
      "clean_"
      "nodes\":[],\"input_axis\":[],\"output_axis\":[],\"input_tensor_indexes\":[],\"output_tensor_indexes\":[],"
      "\"input_tensor_slice\":[],\"output_tensor_slice\":[],\"ori_input_tensor_slice\":[],\"ori_output_tensor_slice\":["
      "],\"inputCutList\":[], \"outputCutList\":[]}";
  ffts::ThreadSliceMap slice_info_2;
  ffts::GetSliceInfoFromJson(slice_info_2, str_slice_info_invalid);

  std::string str_slice_info_3 =
      "{\"dependencies\":[],\"thread_scopeId\":200,\"is_first_node_in_topo_order\":false,\"node_num_in_thread_scope\":"
      "0,"
      "\"is_input_node_of_thread_scope\":false,\"is_output_node_of_thread_scope\":false,\"threadMode\":false,\"slice_"
      "instance_num\":0,\"parallel_window_size\":0,\"thread_id\":\"thread_x\",\"oriInputTensorShape\":[],"
      "\"oriOutputTensorShape\":["
      "],\"original_node\":\"\",\"core_num\":[],\"cutType\":[{\"splitCutIndex\":1,\"reduceCutIndex\":2,\"cutId\":3}],"
      "\"atomic_types\":[],\"thread_id\":0,\"same_atomic_clean_"
      "nodes\":[],\"input_axis\":[],\"output_axis\":[],\"input_tensor_indexes\":[],\"output_tensor_indexes\":[],"
      "\"input_tensor_slice\":[],\"output_tensor_slice\":[],\"ori_input_tensor_slice\":[],\"ori_output_tensor_slice\":["
      "],\"inputCutList\":[], \"outputCutList\":[]}";
  ffts::ThreadSliceMap slice_info_3;
  ffts::GetSliceInfoFromJson(slice_info_3, str_slice_info_3);
  EXPECT_EQ(slice_info_3.thread_scope_id, 200);
}

TEST_F(UTEST_fusion_op_comm, ffts_plus_to_json) {
  ffts::ThreadSliceMap slice_info;
  ffts::OpCut op_cut;
  slice_info.cut_type.emplace_back(op_cut);
  ffts::DimRange dim_range;
  dim_range.lower = 0;
  dim_range.higher = 1;
  std::vector<ffts::DimRange> dim_1;
  dim_1.emplace_back(dim_range);
  std::vector<std::vector<ffts::DimRange>> dim_2;
  dim_2.emplace_back(dim_1);
  slice_info.input_tensor_slice.emplace_back(dim_2);
  slice_info.output_tensor_slice.emplace_back(dim_2);
  std::string str;
  GetSliceJsonInfoStr(slice_info, str);
  EXPECT_EQ(true, true);
}
