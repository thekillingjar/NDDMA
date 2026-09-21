/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <map>
#include <stdlib.h>
#include <stdio.h>
#include <memory>
#include <string>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "fe_llt_utils.h"
#define protected public
#define private public
#include "fusion_stub.hpp"
#include "common/scope_allocator.h"
#include "graph/utils/attr_utils.h"
#include "framework/common/ge_types.h"
#include "graph/ge_local_context.h"
#include "common/fusion_op_comm.h"
#include "common/graph_comm.h"
#include "graph_optimizer/op_compiler/op_compiler.h"
#include "graph_optimizer/ub_fusion/buffer_fusion.h"
#include "graph/op_kernel_bin.h"
#include "common/platform_utils.h"
#include "common/configuration.h"
#include "graph_optimizer/ub_fusion/automatic_buffer_fusion.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "ops_store/ops_kernel_manager.h"
#include "../../../graph_constructor/graph_constructor.h"

using namespace std;
using namespace ge;
using namespace fe;

using OpCompilerPtr = std::shared_ptr<OpCompiler>;
class UBFUSION_ST : public testing::Test {
 protected:
  static void SetUpTestCase()
  {
    std::cout << "fusion ST SetUp" << std::endl;
  }
  static void TearDownTestCase()
  {
    std::cout << "fusion ST TearDown" << std::endl;
  }

  virtual void SetUp() {
  }

  virtual void TearDown()
  {

  }
};
//namespace UB_FUSION {
const int LOOPTIMES = 10;
const string type_list[] = {"ElemWise", "CommReduce", "Opaque", "ElemWise", "ElemWise", "ElemWise"};
const string type_list1[] = {"ElemWise", "CommReduce", "Opaque", "Segment"};

void set_pattern(OpDescPtr OpDesc, string s)
{
  int i = rand() % (sizeof(type_list) / sizeof(type_list[0]));
  SetPattern(OpDesc, type_list[i]);
}

void set_pattern1(OpDescPtr OpDesc, string s)
{
  int i = rand() % (sizeof(type_list1) / sizeof(type_list1[0]));
  SetPattern(OpDesc, type_list1[i]);
}

void BuildGraphElemwiseCast(ComputeGraphPtr graph) {
  OpDescPtr data = std::make_shared<OpDesc>("DATA0", "Data");
  OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", "Data");
  OpDescPtr data2 = std::make_shared<OpDesc>("DATA2", "Data");
  OpDescPtr conv = std::make_shared<OpDesc>("conv", "Convolution");
  OpDescPtr eltwise = std::make_shared<OpDesc>("eltwise", "Eltwise");
  OpDescPtr relu = std::make_shared<OpDesc>("relu", "Relu");
  OpDescPtr quant = std::make_shared<OpDesc>("quant", "Quant");

  SetPattern(conv, "Convolution");
  SetPattern(eltwise, "ElemWise");
  SetPattern(relu, "ElemWise");
  SetPattern(quant, "quant");
  SetTvmType(conv);
  SetTvmType(eltwise);
  SetTvmType(relu);
  SetTvmType(quant);
  // add descriptor
  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);

  data->AddOutputDesc(out_desc);
  data1->AddOutputDesc(out_desc);
  data2->AddOutputDesc(out_desc);
  conv->AddInputDesc(out_desc);
  conv->AddInputDesc(out_desc);
  conv->AddOutputDesc(out_desc);
  eltwise->AddInputDesc(out_desc);
  eltwise->AddInputDesc(out_desc);
  eltwise->AddOutputDesc(out_desc);
  relu->AddInputDesc(out_desc);
  relu->AddOutputDesc(out_desc);
  quant->AddInputDesc(out_desc);
  quant->AddOutputDesc(out_desc);
  AttrUtils::SetInt(conv, FE_IMPLY_TYPE, 6);
  AttrUtils::SetInt(eltwise, FE_IMPLY_TYPE, 6);
  AttrUtils::SetInt(relu, FE_IMPLY_TYPE, 6);
  AttrUtils::SetInt(quant, FE_IMPLY_TYPE, 6);

  NodePtr data_node = graph->AddNode(data);
  NodePtr data1_node = graph->AddNode(data1);
  NodePtr data2_node = graph->AddNode(data2);
  NodePtr conv_node = graph->AddNode(conv);
  NodePtr eltwise_node = graph->AddNode(eltwise);
  NodePtr relu_node = graph->AddNode(relu);
  NodePtr quant_node = graph->AddNode(quant);
  const char tbe_bin[] = "tbe_bin";
  vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
  ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
      conv_node->GetName(), std::move(buffer));
  conv_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                      conv_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0),
                      conv_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                      eltwise_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0),
                      eltwise_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(eltwise_node->GetOutDataAnchor(0),
                      relu_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0),
                      quant_node->GetInDataAnchor(0));
}

void RunUbFusion(ComputeGraphPtr model_graph, string engine_name = fe::AI_CORE_NAME) {
  //  TEUBFUSION START
  std::shared_ptr<GraphComm> graph_comm_ptr = std::make_shared<GraphComm>("engineName");
  graph_comm_ptr->Initialize();
  std::shared_ptr<FusionPriorityManager> fusion_priority_mgr_ptr =
      std::make_shared<FusionPriorityManager>("engineName", nullptr);
  std::vector<BufferFusionInfo> sorted_buffer_fusion_vec = SortedBufferFusionFun();
  fusion_priority_mgr_ptr->sorted_buffer_fusion_map_[FusionPriorityManager::GetCurrentHashedKey()] = sorted_buffer_fusion_vec;
  shared_ptr <BufferFusion> graph_builder(new BufferFusion(graph_comm_ptr,
                                                           fusion_priority_mgr_ptr, nullptr));
  graph_builder->SetEngineName(engine_name);

  graph_builder->MatchFusionPattern(*model_graph);
  auto auto_buffer_fusion_ptr = std::make_shared<AutomaticBufferFusion>(nullptr);
  auto_buffer_fusion_ptr->Run(*model_graph);
}

void check_result(ComputeGraphPtr model_graph, int i, set<set<string>>result)
{
  cout << "=======================graph=============================" << endl;
  PrintGraph(model_graph);
  cout << "=======================before fusion=============================" << endl;
  int64_t scope_id1 = 0;
  uint32_t id1 = 0;
  cout << "before fusion" << endl;
  for (NodePtr node: model_graph->GetAllNodes()) {
    cout << "name: " << node->GetName().c_str() << endl;
    if (ScopeAllocator::GetScopeAttr(node->GetOpDesc(), scope_id1)) {
      cout << "scopeID: " << scope_id1 << endl;
    } else {
      cout << "no scope ID" << endl;
    }
  }
  cout << "=======================debug info=============================" << endl;
  //  TEUBFUSION START

  RunUbFusion(model_graph);
  cout << "=======================after fusion " << i << " =============================" << endl;
  map<uint32_t, set<string>> res1;
  set<set<string>> res;
  for (NodePtr node: model_graph->GetAllNodes()) {
    string pattern = "";
    GetPattern(node->GetOpDesc(), pattern);
    if (ScopeAllocator::GetScopeAttr(node->GetOpDesc(), scope_id1)) {
      cout << "name: " << node->GetName().c_str() << "====scope_id: " << scope_id1 << "========type: "
           << pattern << endl;
      if (res1.find(scope_id1) != res1.end()) {
        ((res1.find(scope_id1))->second).insert(node->GetName());
      } else {
        set <string> s1 = {node->GetName()};
        res1[scope_id1] = s1;
      }
    } else {
      cout << "name: " << node->GetName().c_str() << "=====no scope ID" << "=======type: " << pattern
           << endl;
    }
  }
  for (auto const &x : res1) {
    res.insert(x.second);
  }
  if (res.size() == 0) {
    res = {{}};
  }
  if (res != result) {
    cout << res.size() << " size error " << i << endl;
  }
  EXPECT_EQ(true, res == result);
}
//}

/************************
*nodec1-->noder-->nodee-->nodeu-->nodec3
*                   ^       |
*                   |       V
*                 nodec2   nodeu1
*************************/
TEST_F(UBFUSION_ST, for_coverage)
{
  // initial node
  ComputeGraphPtr model_graph = std::make_shared<ComputeGraph>("modelgraph");

  map<string, vector<desc_info>> src_map;
  map<string, vector<desc_info>> dst_map;
  map<string, vector<desc_info>>::iterator it;
  desc_info dsc_info;
  vector<desc_info> desc_tmp;
  desc_tmp.clear();
  src_map.insert(pair<string, vector<desc_info>> ("nodec1", desc_tmp));
  src_map.insert(pair<string, vector<desc_info>> ("nodec2", desc_tmp));
  src_map.insert(pair<string, vector<desc_info>> ("nodec3", desc_tmp));
  src_map.insert(pair<string, vector<desc_info>> ("nodec4", desc_tmp));
  src_map.insert(pair<string, vector<desc_info>> ("nodec5", desc_tmp));
  src_map.insert(pair<string, vector<desc_info>> ("noder", desc_tmp));
  src_map.insert(pair<string, vector<desc_info>> ("nodee", desc_tmp));
  src_map.insert(pair<string, vector<desc_info>> ("multiOutput", desc_tmp));

  dst_map.insert(pair<string, vector<desc_info>> ("nodec1", desc_tmp));
  dst_map.insert(pair<string, vector<desc_info>> ("nodec2", desc_tmp));
  dst_map.insert(pair<string, vector<desc_info>> ("nodec3", desc_tmp));
  dst_map.insert(pair<string, vector<desc_info>> ("nodec4", desc_tmp));
  dst_map.insert(pair<string, vector<desc_info>> ("nodec5", desc_tmp));
  dst_map.insert(pair<string, vector<desc_info>> ("noder", desc_tmp));
  dst_map.insert(pair<string, vector<desc_info>> ("nodee", desc_tmp));
  dst_map.insert(pair<string, vector<desc_info>> ("multiOutput", desc_tmp));

  OpDescPtr OpDesc;
  vector<OpDescPtr> op_list;
  vector<string> src_name_list;
  vector<string> dst_name_list;
  GeTensorDesc input_desc;
  GeTensorDesc input_desc1;
  GeTensorDesc input_desc2;
  GeTensorDesc output_desc;
  GeTensorDesc output_desc1;
  GeTensorDesc output_desc2;
  vector<GeTensorDesc> input_desc_list;
  vector<GeTensorDesc> output_desc_list;

  // nodec1
  filltensordesc(input_desc, 1, 16, 200, 200, DT_FLOAT16, FORMAT_NCHW);
  filltensordesc(output_desc, 1, 16, 160, 160, DT_FLOAT16, FORMAT_NCHW);
  src_name_list.clear();
  src_name_list.push_back("nodec5");
  dst_name_list.clear();
  dst_name_list.push_back("noder");
  input_desc_list.clear();
  input_desc_list.push_back(input_desc);
  output_desc_list.clear();
  output_desc_list.push_back(output_desc);
  OpDesc = CreateOpDefUbFusion("nodec1", "Data", src_name_list, dst_name_list, input_desc_list, output_desc_list);
  ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
  ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);
  op_list.push_back(OpDesc);

  it = dst_map.find("nodec1");
  vector<desc_info> &vec1 = it->second;
  dsc_info.targetname = "noder";
  dsc_info.index = 0;
  vec1.push_back(dsc_info);
  SetTvmType(OpDesc);
  SetAICoreOp(OpDesc);
  SetPattern(OpDesc, "ElemWise");

  // nodec5
  filltensordesc(input_desc, 1, 16, 500, 500, DT_FLOAT16, FORMAT_NCHW);
  filltensordesc(output_desc, 1, 16, 170, 170, DT_FLOAT16, FORMAT_NCHW);
  src_name_list.clear();
  dst_name_list.clear();
  dst_name_list.push_back("nodec1");
  input_desc_list.clear();
  input_desc_list.push_back(input_desc);
  output_desc_list.clear();
  output_desc_list.push_back(output_desc);
  OpDesc = CreateOpDefUbFusion("nodec5", "Data", src_name_list, dst_name_list, input_desc_list, output_desc_list);
  ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
  op_list.push_back(OpDesc);

  it = dst_map.find("nodec5");
  vector<desc_info> &vec2 = it->second;
  dsc_info.targetname = "nodec1";
  dsc_info.index = 0;
  vec2.push_back(dsc_info);
  SetTvmType(OpDesc);
  SetAICoreOp(OpDesc);
  SetPattern(OpDesc, "ElemWise");

  // noder
  filltensordesc(input_desc, 1, 16, 160, 160, DT_FLOAT16, FORMAT_NCHW);
  filltensordesc(output_desc, 1, 16, 180, 180, DT_FLOAT16, FORMAT_NCHW);
  src_name_list.clear();
  src_name_list.push_back("nodec1");
  dst_name_list.clear();
  dst_name_list.push_back("nodee");
  input_desc_list.clear();
  input_desc_list.push_back(input_desc);
  output_desc_list.clear();
  output_desc_list.push_back(output_desc);
  OpDesc = CreateOpDefUbFusion("noder", "noder", src_name_list, dst_name_list, input_desc_list, output_desc_list);
  ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
  ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);
  SetTvmType(OpDesc);
  op_list.push_back(OpDesc);

  it = dst_map.find("noder");
  vector<desc_info> &vec3 = it->second;
  dsc_info.targetname = "nodee";
  dsc_info.index = 0;
  vec3.push_back(dsc_info);

  it = src_map.find("noder");
  vector<desc_info> &vec4 = it->second;
  dsc_info.targetname = "nodec1";
  dsc_info.index = 0;
  vec4.push_back(dsc_info);
  SetTvmType(OpDesc);
  SetAICoreOp(OpDesc);
  SetPattern(OpDesc, "ElemWise");

  // nodee
  filltensordesc(input_desc, 1, 16, 160, 160, DT_FLOAT16, FORMAT_NCHW);
  filltensordesc(input_desc1, 1, 16, 170, 170, DT_FLOAT16, FORMAT_NCHW);
  filltensordesc(input_desc2, 1, 16, 180, 180, DT_FLOAT16, FORMAT_NCHW);
  filltensordesc(output_desc, 1, 16, 190, 190, DT_FLOAT16, FORMAT_NCHW);
  src_name_list.clear();
  src_name_list.push_back("noder");
  dst_name_list.clear();
  dst_name_list.push_back("multiOutput");
  input_desc_list.clear();
  input_desc_list.push_back(input_desc);
  input_desc_list.push_back(input_desc1);
  input_desc_list.push_back(input_desc2);
  output_desc_list.clear();
  output_desc_list.push_back(output_desc);
  OpDesc = CreateOpDefUbFusion("nodee", "nodee", src_name_list, dst_name_list, input_desc_list, output_desc_list);
  ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
  ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);

  SetTvmType(OpDesc);
  op_list.push_back(OpDesc);

  it = src_map.find("nodee");
  vector<desc_info> &vec5 = it->second;
  dsc_info.targetname = "noder";
  dsc_info.index = 2;
  vec5.push_back(dsc_info);

  it = dst_map.find("nodee");
  vector<desc_info> &vec6 = it->second;
  dsc_info.targetname = "multiOutput";
  dsc_info.index = 0;
  vec6.push_back(dsc_info);
  SetTvmType(OpDesc);
  SetAICoreOp(OpDesc);
  SetPattern(OpDesc, "ElemWise");

  // multi_output
  filltensordesc(input_desc, 1, 16, 190, 190, DT_FLOAT16, FORMAT_NCHW);
  filltensordesc(output_desc, 1, 16, 200, 200, DT_FLOAT16, FORMAT_NCHW);
  filltensordesc(output_desc1, 1, 16, 201, 201, DT_FLOAT16, FORMAT_NCHW);
  src_name_list.clear();
  src_name_list.push_back("nodee");
  dst_name_list.clear();
  dst_name_list.push_back("nodec2");
  dst_name_list.push_back("nodec4");
  input_desc_list.clear();
  input_desc_list.push_back(input_desc);
  output_desc_list.clear();
  output_desc_list.push_back(output_desc);
  output_desc_list.push_back(output_desc1);
  OpDesc = CreateOpDefUbFusion("multiOutput", "multiOutput", src_name_list, dst_name_list, input_desc_list, output_desc_list);
  ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
  ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);

  SetTvmType(OpDesc);
  op_list.push_back(OpDesc);

  it = dst_map.find("multiOutput");
  vector<desc_info> &vec7 = it->second;
  dsc_info.targetname = "nodec2";
  dsc_info.index = 4;
  vec7.push_back(dsc_info);

  dsc_info.targetname = "nodec4";
  dsc_info.index = 0;
  vec7.push_back(dsc_info);

  it = src_map.find("multiOutput");
  vector<desc_info> &vec8 = it->second;
  dsc_info.targetname = "nodee";
  dsc_info.index = 1;
  vec8.push_back(dsc_info);

  SetTvmType(OpDesc);
  SetAICoreOp(OpDesc);
  set_pattern(OpDesc, "ElemWise");

  // nodec2
  filltensordesc(input_desc, 1, 16, 200, 200, DT_FLOAT16, FORMAT_NCHW);
  filltensordesc(output_desc, 1, 16, 207, 207, DT_FLOAT16, FORMAT_NCHW);
  src_name_list.clear();
  src_name_list.push_back("multiOutput");
  dst_name_list.clear();
  dst_name_list.push_back("nodec3");
  input_desc_list.clear();
  input_desc_list.push_back(input_desc);
  output_desc_list.clear();
  output_desc_list.push_back(output_desc);
  OpDesc = CreateOpDefUbFusion("nodec2", "nodec2", src_name_list, dst_name_list, input_desc_list, output_desc_list);
  ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
  ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);
  op_list.push_back(OpDesc);

  it = src_map.find("nodec2");
  vector<desc_info> &vec9 = it->second;
  dsc_info.targetname = "multiOutput";
  dsc_info.index = 0;
  vec9.push_back(dsc_info);

  dsc_info.targetname = "nodec3";
  dsc_info.index = 1;
  vec7.push_back(dsc_info);


  SetTvmType(OpDesc);
  SetAICoreOp(OpDesc);
  set_pattern(OpDesc, "ElemWise");

  // nodec3
  filltensordesc(input_desc, 1, 16, 201, 201, DT_FLOAT16, FORMAT_NCHW);
  filltensordesc(output_desc, 1, 16, 208, 208, DT_FLOAT16, FORMAT_NCHW);
  src_name_list.clear();
  src_name_list.push_back("nodec2");
  dst_name_list.clear();
  input_desc_list.clear();
  input_desc_list.push_back(input_desc);
  output_desc_list.clear();
  output_desc_list.push_back(output_desc);
  OpDesc = CreateOpDefUbFusion("nodec3", "nodec", src_name_list, dst_name_list, input_desc_list, output_desc_list);
  ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);
  op_list.push_back(OpDesc);

  it = src_map.find("nodec3");
  vector<desc_info> &vec10 = it->second;
  dsc_info.targetname = "nodec2";
  dsc_info.index = 0;
  vec10.push_back(dsc_info);
  SetTvmType(OpDesc);
  SetAICoreOp(OpDesc);

  // nodec4
  filltensordesc(input_desc, 1, 16, 201, 201, DT_FLOAT16, FORMAT_NCHW);
  filltensordesc(output_desc, 1, 16, 209, 209, DT_FLOAT16, FORMAT_NCHW);
  src_name_list.clear();
  src_name_list.push_back("multiOutput");
  dst_name_list.clear();
  input_desc_list.clear();
  input_desc_list.push_back(input_desc);
  output_desc_list.clear();
  output_desc_list.push_back(output_desc);
  OpDesc = CreateOpDefUbFusion("nodec4", "nodec", src_name_list, dst_name_list, input_desc_list, output_desc_list);
  ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);
  op_list.push_back(OpDesc);

  it = src_map.find("nodec4");
  vector<desc_info> &vec11 = it->second;
  dsc_info.targetname = "multiOutput";
  dsc_info.index = 0;
  vec11.push_back(dsc_info);
  SetTvmType(OpDesc);
  SetAICoreOp(OpDesc);
  set_pattern(OpDesc, "Elem");

  cout << "=======================test pattern=============================" << endl;

  CreateModelGraph(model_graph, op_list, src_map, dst_map);
  std::shared_ptr<GraphComm> graph_comm_ptr = std::make_shared<GraphComm>("engineName");
  graph_comm_ptr->Initialize();
  std::shared_ptr<FusionPriorityManager> fusion_priority_mgr_ptr =
      std::make_shared<FusionPriorityManager>("engineName", nullptr);
  shared_ptr <BufferFusion> graph_builder(new BufferFusion(graph_comm_ptr,
                                                           fusion_priority_mgr_ptr, nullptr));
  graph_builder->SetEngineName(fe::AI_CORE_NAME);
  EXPECT_EQ(fe::SUCCESS, graph_builder->MatchFusionPattern(*model_graph));
  cout << "=======================graph=============================" << endl;
  PrintGraph(model_graph);
  int64_t scope_id1 = 0;
  for (NodePtr node: model_graph->GetAllNodes()) {
    string pattern = "";
    GetPattern(node->GetOpDesc(), pattern);
    if (ScopeAllocator::GetScopeAttr(node->GetOpDesc(), scope_id1)) {
      cout << "name: " << node->GetName().c_str() << "====scope_id: " << scope_id1 << "========type: "
           << pattern << endl;
    } else {
      cout << "name: " << node->GetName().c_str() << "=====no scope ID" << "=======type: " << pattern
           << endl;
    }
  }
}

/************************
*            nodec5 -> nodec4
*             ^         |
*             |         V
*  noder-->nodee-->multi_output-->nodec2
*   ^                   |
*   |                   V
*  nodec1            nodec3
*************************/
TEST_F(UBFUSION_ST, fusion_test_tefusion_close_circle)
{
  srand(1);
  vector<set<set<string>>> res{
      {{"nodec1", "noder", "nodee", "nodec5", "nodec4", "multiOutput", "nodec2", "nodec3"}}, // 0
  };
  for (int i = 0; i < 1; i++) {
    // initial node
    ComputeGraphPtr model_graph = std::make_shared<ComputeGraph>("modelgraph");

    map<string, vector<desc_info>> src_map;
    map<string, vector<desc_info>> dst_map;
    map<string, vector<desc_info>> ::iterator it;
    desc_info dsc_info;
    vector<desc_info> desc_tmp;
    desc_tmp.clear();
    src_map.insert(pair<string, vector<desc_info>> ("nodec1", desc_tmp));
    src_map.insert(pair<string, vector<desc_info>> ("nodec2", desc_tmp));
    src_map.insert(pair<string, vector<desc_info>> ("nodec3", desc_tmp));
    src_map.insert(pair<string, vector<desc_info>> ("nodec4", desc_tmp));
    src_map.insert(pair<string, vector<desc_info>> ("nodec5", desc_tmp));
    src_map.insert(pair<string, vector<desc_info>> ("noder", desc_tmp));
    src_map.insert(pair<string, vector<desc_info>> ("nodee", desc_tmp));
    src_map.insert(pair<string, vector<desc_info>> ("multiOutput", desc_tmp));

    dst_map.insert(pair<string, vector<desc_info>> ("nodec1", desc_tmp));
    dst_map.insert(pair<string, vector<desc_info>> ("nodec2", desc_tmp));
    dst_map.insert(pair<string, vector<desc_info>> ("nodec3", desc_tmp));
    dst_map.insert(pair<string, vector<desc_info>> ("nodec4", desc_tmp));
    dst_map.insert(pair<string, vector<desc_info>> ("nodec5", desc_tmp));
    dst_map.insert(pair<string, vector<desc_info>> ("noder", desc_tmp));
    dst_map.insert(pair<string, vector<desc_info>> ("nodee", desc_tmp));
    dst_map.insert(pair<string, vector<desc_info>> ("multiOutput", desc_tmp));

    OpDescPtr OpDesc;
    vector<OpDescPtr> op_list;
    vector<string> src_name_list;
    vector<string> dst_name_list;
    GeTensorDesc input_desc;
    GeTensorDesc input_desc1;
    GeTensorDesc input_desc2;
    GeTensorDesc output_desc;
    GeTensorDesc output_desc1;
    GeTensorDesc output_desc2;
    vector<GeTensorDesc> input_desc_list;
    vector<GeTensorDesc> output_desc_list;

    // nodec1
    filltensordesc(input_desc, 1, 16, 200, 200, DT_FLOAT16, FORMAT_NCHW);
    filltensordesc(output_desc, 1, 16, 160, 160, DT_FLOAT16, FORMAT_NCHW);
    src_name_list.clear();
    dst_name_list.clear();
    dst_name_list.push_back("noder");
    input_desc_list.clear();
    input_desc_list.push_back(input_desc);
    output_desc_list.clear();
    output_desc_list.push_back(output_desc);
    OpDesc = CreateOpDefUbFusion("nodec1", "Data", src_name_list, dst_name_list, input_desc_list, output_desc_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
    op_list.push_back(OpDesc);

    it = dst_map.find("nodec1");
    vector<desc_info> &vec1 = it->second;
    dsc_info.targetname = "noder";
    dsc_info.index = 0;
    vec1.push_back(dsc_info);

    SetTvmType(OpDesc);
    SetAICoreOp(OpDesc);
    SetPattern(OpDesc, "ElemWise");

    // nodec5
    filltensordesc(input_desc, 1, 16, 500, 500, DT_FLOAT16, FORMAT_NCHW);
    filltensordesc(output_desc, 1, 16, 170, 170, DT_FLOAT16, FORMAT_NCHW);
    src_name_list.clear();
    src_name_list.push_back("nodee");
    dst_name_list.clear();
    dst_name_list.push_back("nodec4");
    input_desc_list.clear();
    input_desc_list.push_back(input_desc);
    output_desc_list.clear();
    output_desc_list.push_back(output_desc);
    OpDesc = CreateOpDefUbFusion("nodec5", "Data", src_name_list, dst_name_list, input_desc_list, output_desc_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);
    op_list.push_back(OpDesc);

    it = dst_map.find("nodec5");
    vector<desc_info> &vec2 = it->second;
    dsc_info.targetname = "nodec4";
    dsc_info.index = 0;
    vec2.push_back(dsc_info);

    it = src_map.find("nodec5");
    vector<desc_info> &vec4e = it->second;
    dsc_info.targetname = "nodee";
    dsc_info.index = 0;
    vec4e.push_back(dsc_info);
    SetTvmType(OpDesc);
    SetAICoreOp(OpDesc);
    SetPattern(OpDesc, "ElemWise");

    // noder
    filltensordesc(input_desc, 1, 16, 160, 160, DT_FLOAT16, FORMAT_NCHW);
    filltensordesc(output_desc, 1, 16, 180, 180, DT_FLOAT16, FORMAT_NCHW);
    src_name_list.clear();
    src_name_list.push_back("nodec1");
    dst_name_list.clear();
    dst_name_list.push_back("nodee");
    input_desc_list.clear();
    input_desc_list.push_back(input_desc);
    output_desc_list.clear();
    output_desc_list.push_back(output_desc);
    OpDesc = CreateOpDefUbFusion("noder", "noder", src_name_list, dst_name_list, input_desc_list, output_desc_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);
    op_list.push_back(OpDesc);

    it = dst_map.find("noder");
    vector<desc_info> &vec3 = it->second;
    dsc_info.targetname = "nodee";
    dsc_info.index = 0;
    vec3.push_back(dsc_info);

    it = src_map.find("noder");
    vector<desc_info> &vec4 = it->second;
    dsc_info.targetname = "nodec1";
    dsc_info.index = 0;
    vec4.push_back(dsc_info);

    SetTvmType(OpDesc);
    SetAICoreOp(OpDesc);
    SetPattern(OpDesc, "ElemWise");

    // nodee
    filltensordesc(input_desc, 1, 16, 160, 160, DT_FLOAT16, FORMAT_NCHW);
    filltensordesc(input_desc1, 1, 16, 170, 170, DT_FLOAT16, FORMAT_NCHW);
    filltensordesc(input_desc2, 1, 16, 180, 180, DT_FLOAT16, FORMAT_NCHW);
    filltensordesc(output_desc, 1, 16, 190, 190, DT_FLOAT16, FORMAT_NCHW);
    src_name_list.clear();
    src_name_list.push_back("noder");
    dst_name_list.clear();
    dst_name_list.push_back("multiOutput");
    dst_name_list.push_back("nodec5");
    input_desc_list.clear();
    input_desc_list.push_back(input_desc);
    output_desc_list.clear();
    output_desc_list.push_back(output_desc);
    OpDesc = CreateOpDefUbFusion("nodee", "nodee", src_name_list, dst_name_list, input_desc_list, output_desc_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);

    op_list.push_back(OpDesc);

    it = src_map.find("nodee");
    vector<desc_info> &vec5 = it->second;
    dsc_info.targetname = "noder";
    dsc_info.index = 0;
    vec5.push_back(dsc_info);

    it = dst_map.find("nodee");
    vector<desc_info> &vec6 = it->second;
    dsc_info.targetname = "multiOutput";
    dsc_info.index = 0;
    vec6.push_back(dsc_info);

    dsc_info.targetname = "nodec5";
    dsc_info.index = 0;
    vec6.push_back(dsc_info);

    SetTvmType(OpDesc);
    SetAICoreOp(OpDesc);
    SetPattern(OpDesc, "ElemWise");

    // multi_output
    filltensordesc(input_desc, 1, 16, 190, 190, DT_FLOAT16, FORMAT_NCHW);
    filltensordesc(input_desc1, 1, 16, 190, 190, DT_FLOAT16, FORMAT_NCHW);
    filltensordesc(output_desc, 1, 16, 200, 200, DT_FLOAT16, FORMAT_NCHW);
    src_name_list.clear();
    src_name_list.push_back("nodee");
    src_name_list.push_back("nodec4");
    dst_name_list.clear();
    dst_name_list.push_back("nodec2");
    dst_name_list.push_back("nodec3");
    input_desc_list.clear();
    input_desc_list.push_back(input_desc);
    input_desc_list.push_back(input_desc1);
    output_desc_list.clear();
    output_desc_list.push_back(output_desc);
    OpDesc = CreateOpDefUbFusion("multiOutput", "multiOutput", src_name_list, dst_name_list, input_desc_list, output_desc_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);
    op_list.push_back(OpDesc);

    it = dst_map.find("multiOutput");
    vector<desc_info> &vec7 = it->second;
    dsc_info.targetname = "nodec2";
    dsc_info.index = 0;
    vec7.push_back(dsc_info);

    dsc_info.targetname = "nodec3";
    dsc_info.index = 0;
    vec7.push_back(dsc_info);

    it = src_map.find("multiOutput");
    vector<desc_info> &vec8 = it->second;
    dsc_info.targetname = "nodee";
    dsc_info.index = 0;
    vec8.push_back(dsc_info);

    dsc_info.targetname = "nodec4";
    dsc_info.index = 1;
    vec8.push_back(dsc_info);

    SetTvmType(OpDesc);
    SetAICoreOp(OpDesc);
    SetPattern(OpDesc, "ElemWise");

    // nodec2
    filltensordesc(input_desc, 1, 16, 200, 200, DT_FLOAT16, FORMAT_NCHW);
    filltensordesc(output_desc, 1, 16, 207, 207, DT_FLOAT16, FORMAT_NCHW);
    src_name_list.clear();
    src_name_list.push_back("multiOutput");
    dst_name_list.clear();
    input_desc_list.clear();
    input_desc_list.push_back(input_desc);
    output_desc_list.clear();
    output_desc_list.push_back(output_desc);
    OpDesc = CreateOpDefUbFusion("nodec2", "nodec2", src_name_list, dst_name_list, input_desc_list, output_desc_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);
    op_list.push_back(OpDesc);

    it = src_map.find("nodec2");
    vector<desc_info> &vec9 = it->second;
    dsc_info.targetname = "multiOutput";
    dsc_info.index = 0;
    vec9.push_back(dsc_info);

    SetTvmType(OpDesc);
    SetAICoreOp(OpDesc);
    SetPattern(OpDesc, "ElemWise");

    // nodec3
    filltensordesc(input_desc, 1, 16, 201, 201, DT_FLOAT16, FORMAT_NCHW);
    filltensordesc(output_desc, 1, 16, 208, 208, DT_FLOAT16, FORMAT_NCHW);
    src_name_list.clear();
    src_name_list.push_back("multiOutput");
    dst_name_list.clear();
    input_desc_list.clear();
    input_desc_list.push_back(input_desc);
    output_desc_list.clear();
    output_desc_list.push_back(output_desc);
    OpDesc = CreateOpDefUbFusion("nodec3", "nodec", src_name_list, dst_name_list, input_desc_list, output_desc_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);
    op_list.push_back(OpDesc);

    it = src_map.find("nodec3");
    vector<desc_info> &vec10 = it->second;
    dsc_info.targetname = "multiOutput";
    dsc_info.index = 0;
    vec10.push_back(dsc_info);

    SetTvmType(OpDesc);
    SetAICoreOp(OpDesc);
    SetPattern(OpDesc, "ElemWise");

    // nodec4
    filltensordesc(input_desc, 1, 16, 201, 201, DT_FLOAT16, FORMAT_NCHW);
    filltensordesc(output_desc, 1, 16, 209, 209, DT_FLOAT16, FORMAT_NCHW);
    src_name_list.clear();
    src_name_list.push_back("nodec5");
    dst_name_list.clear();
    dst_name_list.push_back("multiOutput");
    input_desc_list.clear();
    input_desc_list.push_back(input_desc);
    output_desc_list.clear();
    output_desc_list.push_back(output_desc);
    OpDesc = CreateOpDefUbFusion("nodec4", "nodec", src_name_list, dst_name_list, input_desc_list, output_desc_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);
    op_list.push_back(OpDesc);

    it = src_map.find("nodec4");
    vector<desc_info> &vec11 = it->second;
    dsc_info.targetname = "nodec5";
    dsc_info.index = 0;
    vec11.push_back(dsc_info);

    it = dst_map.find("nodec4");
    vector<desc_info> &vec1c = it->second;
    dsc_info.targetname = "multiOutput";
    dsc_info.index = 0;
    vec1c.push_back(dsc_info);


    SetTvmType(OpDesc);
    SetAICoreOp(OpDesc);
    SetPattern(OpDesc, "ElemWise");

    CreateModelGraph(model_graph, op_list, src_map, dst_map);
    check_result(model_graph, i, res[i]);
  }
}

/***************************************************************************
 *                                 nodec11------->--nodec12
 *                                   ^                |
 *                                   |                V
 *            nodec5     nodec4--->nodec8->nodec9->nodec10---->nodec6
 *             |          ^                                       |
 *             V          |                                       V
 *  noder-->nodee-->multi_output->nodec7------------------------->nodec2--->nodec3
 *   ^
 *   |
 *  nodec1
*****************************************************************************/
TEST_F(UBFUSION_ST, fusion_test_tefusion_bifurcated_with_circle_nesting2)
{

  srand(1);
  vector<set<set<string>>> res{
      {{"nodec1", "noder"}, {"nodee",   "multiOutput"}, {"nodec4", "nodec8"}, {"nodec2",  "nodec3"}, {"nodec11", "nodec12"}},// 0
      {{"nodec1", "noder"}, {"nodee",   "multiOutput"}, {"nodec4", "nodec8"}, {"nodec10",  "nodec6"}, {"nodec11", "nodec12"}},// 1
      {{"nodec1", "noder"}, {"nodee",   "multiOutput"}, {"nodec4", "nodec8"}, {"nodec10",  "nodec6"}, {"nodec11", "nodec12"}},// 2
      {{"nodec1", "noder"}, {"nodec10",  "nodec6"}, {"nodec11", "nodec12"}, {"nodec2",  "nodec3"}},// 3
      {{"nodec1", "noder"}, {"nodee",   "multiOutput"}, {"nodec4", "nodec8"}, {"nodec2",  "nodec3"}, {"nodec11", "nodec12"}},// 4
      {{"nodec1", "noder"}, {"nodee",   "multiOutput"}, {"nodec4", "nodec8"}, {"nodec2",  "nodec3"}, {"nodec11", "nodec12"}, {"nodec10",  "nodec6"}},// 5
      {{"nodec1", "noder"}, {"nodec12", "nodec11"},     {"nodec4", "nodec8"}},// 6
      {{"nodec1", "noder"}, {"nodee",   "multiOutput"}, {"nodec10",  "nodec6"}, {"nodec2",  "nodec3"}, {"nodec11", "nodec12"}},// 7
      {{"nodec1", "noder"}, {"nodec3",  "nodec2"},      {"nodec4", "nodec8"}, {"nodec11", "nodec12"}},// 8
      {{"nodec1", "noder"}, {"nodee",   "multiOutput"}, {"nodec10",  "nodec6"}, {"nodec2",  "nodec3"}, {"nodec11", "nodec12"}},// 9
  };

  for (int i = 0; i < 1; i++) {
    // initial node
    ComputeGraphPtr model_graph = std::make_shared<ComputeGraph>("modelgraph");

    map<string, vector<desc_info>> src_map;
    map<string, vector<desc_info>> dst_map;
    map<string, vector<desc_info>>::iterator it;
    desc_info dsc_info;
    vector<desc_info> desc_tmp;
    desc_tmp.clear();

    src_map.insert(pair<string, vector<desc_info>> ("nodec1", desc_tmp));
    src_map.insert(pair<string, vector<desc_info>> ("nodec2", desc_tmp));
    src_map.insert(pair<string, vector<desc_info>> ("nodec3", desc_tmp));
    src_map.insert(pair<string, vector<desc_info>> ("nodec4", desc_tmp));
    src_map.insert(pair<string, vector<desc_info>> ("nodec5", desc_tmp));
    src_map.insert(pair<string, vector<desc_info>> ("nodec6", desc_tmp));
    src_map.insert(pair<string, vector<desc_info>> ("nodec7", desc_tmp));
    src_map.insert(pair<string, vector<desc_info>> ("nodec8", desc_tmp));
    src_map.insert(pair<string, vector<desc_info>> ("nodec9", desc_tmp));
    src_map.insert(pair<string, vector<desc_info>> ("nodec10", desc_tmp));
    src_map.insert(pair<string, vector<desc_info>> ("nodec11", desc_tmp));
    src_map.insert(pair<string, vector<desc_info>> ("nodec12", desc_tmp));
    src_map.insert(pair<string, vector<desc_info>> ("noder", desc_tmp));
    src_map.insert(pair<string, vector<desc_info>> ("nodee", desc_tmp));
    src_map.insert(pair<string, vector<desc_info>> ("multiOutput", desc_tmp));

    dst_map.insert(pair<string, vector<desc_info>> ("nodec1", desc_tmp));
    dst_map.insert(pair<string, vector<desc_info>> ("nodec2", desc_tmp));
    dst_map.insert(pair<string, vector<desc_info>> ("nodec3", desc_tmp));
    dst_map.insert(pair<string, vector<desc_info>> ("nodec4", desc_tmp));
    dst_map.insert(pair<string, vector<desc_info>> ("nodec5", desc_tmp));
    dst_map.insert(pair<string, vector<desc_info>> ("nodec6", desc_tmp));
    dst_map.insert(pair<string, vector<desc_info>> ("nodec7", desc_tmp));
    dst_map.insert(pair<string, vector<desc_info>> ("nodec8", desc_tmp));
    dst_map.insert(pair<string, vector<desc_info>> ("nodec9", desc_tmp));
    dst_map.insert(pair<string, vector<desc_info>> ("nodec10", desc_tmp));
    dst_map.insert(pair<string, vector<desc_info>> ("nodec11", desc_tmp));
    dst_map.insert(pair<string, vector<desc_info>> ("nodec12", desc_tmp));
    dst_map.insert(pair<string, vector<desc_info>> ("noder", desc_tmp));
    dst_map.insert(pair<string, vector<desc_info>> ("nodee", desc_tmp));
    dst_map.insert(pair<string, vector<desc_info>> ("multiOutput", desc_tmp));

    OpDescPtr OpDesc;
    vector<OpDescPtr> op_list;
    vector<string> src_name_list;
    vector<string> dst_name_list;
    GeTensorDesc input_desc;
    GeTensorDesc input_desc1;
    GeTensorDesc input_desc2;
    GeTensorDesc output_desc;
    GeTensorDesc output_desc1;
    GeTensorDesc output_desc2;
    vector<GeTensorDesc> input_desc_list;
    vector<GeTensorDesc> output_desc_list;

    // nodec1
    filltensordesc(input_desc, 1, 16, 200, 200, DT_FLOAT16,
                   FORMAT_NCHW);
    filltensordesc(output_desc, 1, 16, 160, 160, DT_FLOAT16,
                   FORMAT_NCHW);
    src_name_list.clear();
    dst_name_list.clear();
    dst_name_list.push_back("noder");
    input_desc_list.clear();
    input_desc_list.push_back(input_desc);
    output_desc_list.clear();
    output_desc_list.push_back(output_desc);
    OpDesc = CreateOpDefUbFusion("nodec1", "Data", src_name_list, dst_name_list,
                                 input_desc_list, output_desc_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
    op_list.push_back(OpDesc);

    it = dst_map.find("nodec1");
    vector<desc_info> &vec1 = it->second;
    dsc_info.targetname = "noder";
    dsc_info.index = 0;
    vec1.push_back(dsc_info);


    SetTvmType(OpDesc);
    SetAICoreOp(OpDesc);
    SetPattern(OpDesc, "ElemWise");

    // nodec5
    filltensordesc(input_desc, 1, 16, 500, 500, DT_FLOAT16,
                   FORMAT_NCHW);
    filltensordesc(output_desc, 1, 16, 170, 170, DT_FLOAT16,
                   FORMAT_NCHW);
    src_name_list.clear();
    dst_name_list.clear();
    dst_name_list.push_back("nodee");
    input_desc_list.clear();
    input_desc_list.push_back(input_desc);
    output_desc_list.clear();
    output_desc_list.push_back(output_desc);
    OpDesc = CreateOpDefUbFusion("nodec5", "Data", src_name_list, dst_name_list,
                                 input_desc_list, output_desc_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
    op_list.push_back(OpDesc);

    it = dst_map.find("nodec5");
    vector<desc_info> &vec2 = it->second;
    dsc_info.targetname = "nodee";
    dsc_info.index = 0;
    vec2.push_back(dsc_info);
    SetTvmType(OpDesc);
    SetAICoreOp(OpDesc);
    SetPattern(OpDesc, "ElemWise");

    // noder
    filltensordesc(input_desc, 1, 16, 160, 160, DT_FLOAT16,
                   FORMAT_NCHW);
    filltensordesc(output_desc, 1, 16, 180, 180, DT_FLOAT16,
                   FORMAT_NCHW);
    src_name_list.clear();
    src_name_list.push_back("nodec1");
    dst_name_list.clear();
    dst_name_list.push_back("nodee");
    input_desc_list.clear();
    input_desc_list.push_back(input_desc);
    output_desc_list.clear();
    output_desc_list.push_back(output_desc);
    OpDesc = CreateOpDefUbFusion("noder", "cov", src_name_list, dst_name_list,
                                 input_desc_list, output_desc_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);
    op_list.push_back(OpDesc);


    it = dst_map.find("noder");
    vector<desc_info> &vec3 = it->second;
    dsc_info.targetname = "nodee";
    dsc_info.index = 0;
    vec3.push_back(dsc_info);

    it = src_map.find("noder");
    vector<desc_info> &vec4 = it->second;
    dsc_info.targetname = "nodec1";
    dsc_info.index = 0;
    vec4.push_back(dsc_info);
    SetTvmType(OpDesc);
    SetAICoreOp(OpDesc);
    SetPattern(OpDesc, "ElemWise");

    // nodee
    filltensordesc(input_desc, 1, 16, 160, 160, DT_FLOAT16,
                   FORMAT_NCHW);
    filltensordesc(input_desc1, 1, 16, 170, 170, DT_FLOAT16,
                   FORMAT_NCHW);
    filltensordesc(output_desc, 1, 16, 190, 190, DT_FLOAT16,
                   FORMAT_NCHW);
    src_name_list.clear();
    src_name_list.push_back("nodec5");
    src_name_list.push_back("noder");
    dst_name_list.clear();
    dst_name_list.push_back("multiOutput");
    input_desc_list.clear();
    input_desc_list.push_back(input_desc);
    input_desc_list.push_back(input_desc1);
    output_desc_list.clear();
    output_desc_list.push_back(output_desc);
    OpDesc = CreateOpDefUbFusion("nodee", "nodee", src_name_list, dst_name_list,
                                 input_desc_list, output_desc_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);

    op_list.push_back(OpDesc);

    it = src_map.find("nodee");
    vector<desc_info> &vec5 = it->second;
    dsc_info.targetname = "nodec5";
    dsc_info.index = 0;
    vec5.push_back(dsc_info);

    dsc_info.targetname = "noder";
    dsc_info.index = 1;
    vec5.push_back(dsc_info);

    it = dst_map.find("nodee");
    vector<desc_info> &vec6 = it->second;
    dsc_info.targetname = "multiOutput";
    dsc_info.index = 0;
    vec6.push_back(dsc_info);
    SetTvmType(OpDesc);
    SetAICoreOp(OpDesc);
    SetPattern(OpDesc, "ElemWise");

    // multi_output
    filltensordesc(input_desc, 1, 16, 190, 190, DT_FLOAT16, FORMAT_NCHW);
    filltensordesc(output_desc, 1, 16, 200, 200, DT_FLOAT16, FORMAT_NCHW);
    src_name_list.clear();
    src_name_list.push_back("nodee");
    dst_name_list.clear();
    dst_name_list.push_back("nodec7");
    dst_name_list.push_back("nodec4");
    input_desc_list.clear();
    input_desc_list.push_back(input_desc);
    output_desc_list.clear();
    output_desc_list.push_back(output_desc);
    OpDesc = CreateOpDefUbFusion("multiOutput", "multiOutput", src_name_list,
                                 dst_name_list, input_desc_list, output_desc_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);

    op_list.push_back(OpDesc);

    it = dst_map.find("multiOutput");
    vector<desc_info> &vec7 = it->second;
    dsc_info.targetname = "nodec7";
    dsc_info.index = 0;
    vec7.push_back(dsc_info);
    dsc_info.targetname = "nodec4";
    dsc_info.index = 0;
    vec7.push_back(dsc_info);

    it = src_map.find("multiOutput");
    vector<desc_info> &vec8 = it->second;
    dsc_info.targetname = "nodee";
    dsc_info.index = 0;
    vec8.push_back(dsc_info);

    SetTvmType(OpDesc);
    SetAICoreOp(OpDesc);
    set_pattern1(OpDesc, "ElemWise");

    // nodec2
    filltensordesc(input_desc, 1, 16, 200, 200, DT_FLOAT16,
                   FORMAT_NCHW);
    filltensordesc(input_desc1, 1, 16, 200, 200, DT_FLOAT16,
                   FORMAT_NCHW);
    filltensordesc(output_desc, 1, 16, 207, 207, DT_FLOAT16,
                   FORMAT_NCHW);
    src_name_list.clear();
    src_name_list.push_back("nodec6");
    src_name_list.push_back("nodec7");
    dst_name_list.clear();
    dst_name_list.push_back("nodec3");
    input_desc_list.clear();
    input_desc_list.push_back(input_desc);
    input_desc_list.push_back(input_desc1);
    output_desc_list.clear();
    output_desc_list.push_back(output_desc);
    OpDesc = CreateOpDefUbFusion("nodec2", "nodec2", src_name_list, dst_name_list,
                                 input_desc_list, output_desc_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);
    op_list.push_back(OpDesc);

    it = src_map.find("nodec2");
    vector<desc_info> &vec9 = it->second;
    dsc_info.targetname = "nodec6";
    dsc_info.index = 0;
    vec9.push_back(dsc_info);
    dsc_info.targetname = "nodec7";
    dsc_info.index = 1;
    vec9.push_back(dsc_info);

    it = dst_map.find("nodec2");
    vector<desc_info> &vec19 = it->second;
    dsc_info.targetname = "nodec3";
    dsc_info.index = 0;
    vec19.push_back(dsc_info);


    SetTvmType(OpDesc);
    SetAICoreOp(OpDesc);
    SetPattern(OpDesc, "ElemWise");

    // nodec10
    filltensordesc(input_desc, 1, 16, 200, 200, DT_FLOAT16,
                   FORMAT_NCHW);
    filltensordesc(input_desc1, 1, 16, 200, 200, DT_FLOAT16,
                   FORMAT_NCHW);
    filltensordesc(output_desc, 1, 16, 207, 207, DT_FLOAT16,
                   FORMAT_NCHW);
    src_name_list.clear();
    src_name_list.push_back("nodec9");
    src_name_list.push_back("nodec12");
    dst_name_list.clear();
    dst_name_list.push_back("nodec6");
    input_desc_list.clear();
    input_desc_list.push_back(input_desc);
    input_desc_list.push_back(input_desc1);
    output_desc_list.clear();
    output_desc_list.push_back(output_desc);
    OpDesc = CreateOpDefUbFusion("nodec10", "nodec2", src_name_list, dst_name_list,
                                 input_desc_list, output_desc_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);
    op_list.push_back(OpDesc);

    it = src_map.find("nodec10");
    vector<desc_info> &vec910 = it->second;
    dsc_info.targetname = "nodec9";
    dsc_info.index = 0;
    vec910.push_back(dsc_info);
    dsc_info.targetname = "nodec12";
    dsc_info.index = 1;
    vec910.push_back(dsc_info);

    it = dst_map.find("nodec10");
    vector<desc_info> &vec1990 = it->second;
    dsc_info.targetname = "nodec6";
    dsc_info.index = 0;
    vec1990.push_back(dsc_info);

    SetTvmType(OpDesc);
    SetAICoreOp(OpDesc);
    set_pattern1(OpDesc, "ElemWise");

    // nodec6
    filltensordesc(input_desc, 1, 16, 200, 200, DT_FLOAT16,
                   FORMAT_NCHW);
    filltensordesc(output_desc, 1, 16, 207, 207, DT_FLOAT16,
                   FORMAT_NCHW);
    src_name_list.clear();
    src_name_list.push_back("nodec10");
    dst_name_list.clear();
    dst_name_list.push_back("nodec2");
    input_desc_list.clear();
    input_desc_list.push_back(input_desc);
    output_desc_list.clear();
    output_desc_list.push_back(output_desc);
    OpDesc = CreateOpDefUbFusion("nodec6", "nodec6", src_name_list, dst_name_list,
                                 input_desc_list, output_desc_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);
    op_list.push_back(OpDesc);

    it = src_map.find("nodec6");
    vector<desc_info> &vec96 = it->second;
    dsc_info.targetname = "nodec10";
    dsc_info.index = 0;
    vec96.push_back(dsc_info);

    it = dst_map.find("nodec6");
    vector<desc_info> &vec199 = it->second;
    dsc_info.targetname = "nodec2";
    dsc_info.index = 0;
    vec199.push_back(dsc_info);


    SetTvmType(OpDesc);
    SetAICoreOp(OpDesc);
    SetPattern(OpDesc, "ElemWise");

    // nodec3
    filltensordesc(input_desc, 1, 16, 201, 201, DT_FLOAT16,
                   FORMAT_NCHW);
    filltensordesc(output_desc, 1, 16, 208, 208, DT_FLOAT16,
                   FORMAT_NCHW);
    src_name_list.clear();
    src_name_list.push_back("nodec2");
    dst_name_list.clear();
    input_desc_list.clear();
    input_desc_list.push_back(input_desc);
    output_desc_list.clear();
    output_desc_list.push_back(output_desc);
    OpDesc = CreateOpDefUbFusion("nodec3", "nodec", src_name_list, dst_name_list,
                                 input_desc_list, output_desc_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);
    op_list.push_back(OpDesc);

    it = src_map.find("nodec3");
    vector<desc_info> &vec103 = it->second;
    dsc_info.targetname = "nodec2";
    dsc_info.index = 0;
    vec103.push_back(dsc_info);

    SetTvmType(OpDesc);
    SetAICoreOp(OpDesc);

    set_pattern1(OpDesc, "ElemWise");

    // nodec11
    filltensordesc(input_desc, 1, 16, 201, 201, DT_FLOAT16,
                   FORMAT_NCHW);
    filltensordesc(output_desc, 1, 16, 208, 208, DT_FLOAT16,
                   FORMAT_NCHW);
    src_name_list.clear();
    src_name_list.push_back("nodec8");
    dst_name_list.clear();
    dst_name_list.push_back("nodec12");
    input_desc_list.clear();
    input_desc_list.push_back(input_desc);
    output_desc_list.clear();
    output_desc_list.push_back(output_desc);
    OpDesc = CreateOpDefUbFusion("nodec11", "nodec", src_name_list, dst_name_list,
                                 input_desc_list, output_desc_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);
    op_list.push_back(OpDesc);

    it = src_map.find("nodec11");
    vector<desc_info> &vec1011 = it->second;
    dsc_info.targetname = "nodec8";
    dsc_info.index = 0;
    vec1011.push_back(dsc_info);

    it = dst_map.find("nodec11");
    vector<desc_info> &vec1911 = it->second;
    dsc_info.targetname = "nodec12";
    dsc_info.index = 0;
    vec1911.push_back(dsc_info);

    SetTvmType(OpDesc);
    SetAICoreOp(OpDesc);
    SetPattern(OpDesc, "ElemWise");

    // nodec12
    filltensordesc(input_desc, 1, 16, 201, 201, DT_FLOAT16,
                   FORMAT_NCHW);
    filltensordesc(output_desc, 1, 16, 208, 208, DT_FLOAT16,
                   FORMAT_NCHW);
    src_name_list.clear();
    src_name_list.push_back("nodec11");
    dst_name_list.clear();
    dst_name_list.push_back("nodec10");
    input_desc_list.clear();
    input_desc_list.push_back(input_desc);
    output_desc_list.clear();
    output_desc_list.push_back(output_desc);
    OpDesc = CreateOpDefUbFusion("nodec12", "nodec", src_name_list, dst_name_list,
                                 input_desc_list, output_desc_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);
    op_list.push_back(OpDesc);

    it = src_map.find("nodec12");
    vector<desc_info> &vec1012 = it->second;
    dsc_info.targetname = "nodec11";
    dsc_info.index = 0;
    vec1012.push_back(dsc_info);

    it = dst_map.find("nodec12");
    vector<desc_info> &vec1912 = it->second;
    dsc_info.targetname = "nodec10";
    dsc_info.index = 0;
    vec1912.push_back(dsc_info);

    SetTvmType(OpDesc);
    SetAICoreOp(OpDesc);
    SetPattern(OpDesc, "ElemWise");

    // nodec9
    filltensordesc(input_desc, 1, 16, 201, 201, DT_FLOAT16,
                   FORMAT_NCHW);
    filltensordesc(output_desc, 1, 16, 208, 208, DT_FLOAT16,
                   FORMAT_NCHW);
    src_name_list.clear();
    src_name_list.push_back("nodec8");
    dst_name_list.clear();
    dst_name_list.push_back("nodec10");
    input_desc_list.clear();
    input_desc_list.push_back(input_desc);
    output_desc_list.clear();
    output_desc_list.push_back(output_desc);
    OpDesc = CreateOpDefUbFusion("nodec9", "nodec", src_name_list, dst_name_list,
                                 input_desc_list, output_desc_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);
    op_list.push_back(OpDesc);

    it = src_map.find("nodec9");
    vector<desc_info> &vec1909 = it->second;
    dsc_info.targetname = "nodec8";
    dsc_info.index = 0;
    vec1909.push_back(dsc_info);

    it = dst_map.find("nodec9");
    vector<desc_info> &vec1999 = it->second;
    dsc_info.targetname = "nodec10";
    dsc_info.index = 0;
    vec1999.push_back(dsc_info);

    SetTvmType(OpDesc);
    SetAICoreOp(OpDesc);
    SetPattern(OpDesc, "ElemWise");

    // nodec8
    filltensordesc(input_desc, 1, 16, 201, 201, DT_FLOAT16, FORMAT_NCHW);
    filltensordesc(output_desc, 1, 16, 208, 208, DT_FLOAT16, FORMAT_NCHW);
    src_name_list.clear();
    src_name_list.push_back("nodec4");
    dst_name_list.clear();
    dst_name_list.push_back("nodec9");
    dst_name_list.push_back("nodec11");
    input_desc_list.clear();
    input_desc_list.push_back(input_desc);
    output_desc_list.clear();
    output_desc_list.push_back(output_desc);
    OpDesc = CreateOpDefUbFusion("nodec8", "nodec", src_name_list, dst_name_list,
                                 input_desc_list, output_desc_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);
    op_list.push_back(OpDesc);

    it = src_map.find("nodec8");
    vector<desc_info> &vec108 = it->second;
    dsc_info.targetname = "nodec4";
    dsc_info.index = 0;
    vec108.push_back(dsc_info);

    it = dst_map.find("nodec8");
    vector<desc_info> &vec1938 = it->second;
    dsc_info.targetname = "nodec9";
    dsc_info.index = 0;
    vec1938.push_back(dsc_info);

    dsc_info.targetname = "nodec11";
    dsc_info.index = 0;
    vec1938.push_back(dsc_info);

    SetTvmType(OpDesc);
    SetAICoreOp(OpDesc);
    SetPattern(OpDesc, "ElemWise");

    // nodec4
    filltensordesc(input_desc, 1, 16, 201, 201, DT_FLOAT16,
                   FORMAT_NCHW);
    filltensordesc(output_desc, 1, 16, 209, 209, DT_FLOAT16,
                   FORMAT_NCHW);
    src_name_list.clear();
    src_name_list.push_back("multiOutput");
    dst_name_list.clear();
    dst_name_list.push_back("nodec8");
    input_desc_list.clear();
    input_desc_list.push_back(input_desc);
    output_desc_list.clear();
    output_desc_list.push_back(output_desc);
    OpDesc = CreateOpDefUbFusion("nodec4", "nodec", src_name_list, dst_name_list,
                                 input_desc_list, output_desc_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);
    op_list.push_back(OpDesc);

    it = src_map.find("nodec4");
    vector<desc_info> &vec11 = it->second;
    dsc_info.targetname = "multiOutput";
    dsc_info.index = 0;
    vec11.push_back(dsc_info);

    it = dst_map.find("nodec4");
    vector<desc_info> &vec100 = it->second;
    dsc_info.targetname = "nodec8";
    dsc_info.index = 0;
    vec100.push_back(dsc_info);

    SetTvmType(OpDesc);
    SetAICoreOp(OpDesc);
    set_pattern1(OpDesc, "ElemWise");

    // nodec7
    filltensordesc(input_desc, 1, 16, 201, 201, DT_FLOAT16,
                   FORMAT_NCHW);
    filltensordesc(output_desc, 1, 16, 209, 209, DT_FLOAT16,
                   FORMAT_NCHW);
    src_name_list.clear();
    src_name_list.push_back("multiOutput");
    dst_name_list.clear();
    dst_name_list.push_back("nodec2");
    input_desc_list.clear();
    input_desc_list.push_back(input_desc);
    output_desc_list.clear();
    output_desc_list.push_back(output_desc);
    OpDesc = CreateOpDefUbFusion("nodec7", "nodec", src_name_list, dst_name_list,
                                 input_desc_list, output_desc_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_DST_NAME, dst_name_list);
    ge::AttrUtils::SetListStr(OpDesc, OPDESC_SRC_NAME, src_name_list);
    op_list.push_back(OpDesc);

    it = src_map.find("nodec7");
    vector<desc_info> &vec17 = it->second;
    dsc_info.targetname = "multiOutput";
    dsc_info.index = 0;
    vec17.push_back(dsc_info);

    it = dst_map.find("nodec7");
    vector<desc_info> &vec101 = it->second;
    dsc_info.targetname = "nodec2";
    dsc_info.index = 0;
    vec101.push_back(dsc_info);

    SetTvmType(OpDesc);
    SetAICoreOp(OpDesc);
    SetPattern(OpDesc, "ElemWise");

    CreateModelGraph(model_graph, op_list, src_map, dst_map);

    std::shared_ptr<GraphComm> graph_comm_ptr = std::make_shared<GraphComm>("engineName");
    graph_comm_ptr->Initialize();
    std::shared_ptr<FusionPriorityManager> fusion_priority_mgr_ptr =
        std::make_shared<FusionPriorityManager>("engineName", nullptr);
    std::shared_ptr<BufferFusion> sub_graph_optimizer_ptr =
        std::make_shared<BufferFusion>(graph_comm_ptr, fusion_priority_mgr_ptr, nullptr);

    // find sub-graphs that match UB fusion pattern
    //ComputeGraphPtr origin_graph_ptr = std::make_shared<ComputeGraph>(*model_graph);
    EXPECT_EQ(sub_graph_optimizer_ptr->MatchFusionPatternFromGraph(*model_graph), fe::SUCCESS);

    // create fused Graph, and merge matched sub-graphs into fusion ops
    EXPECT_EQ(sub_graph_optimizer_ptr->BuildFusionGraph(*model_graph), fe::SUCCESS);

    uint32_t id = 0;

    cerr << endl;
    cerr << "UB fusion befre" << endl;
    for (auto node : model_graph->GetDirectNode())
    {
      cerr << " id:" << id << endl;
      uint32_t scope_id = 0;
      cerr << "opdef : " << node->GetType() << endl;
      if (AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id))
      {
        cerr << "scope id : " << scope_id << endl;
      }
      id++;

    }
    id = 0;
    cerr << endl;
    cerr << "UB fusion result" << endl;
    for (auto node : model_graph->GetDirectNode())
    {
      cerr << " id:" << id << endl;
      uint32_t scope_id = 0;
      cerr << "opdef : " << node->GetType() << endl;
      if (AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id))
      {
        cerr << "scope id : " << scope_id << endl;
      }
      id++;
    }
  }
//}
}

namespace st {
class BuiltBufferFusionPassTest : public BufferFusionPassBase {
 public:
  std::vector<BufferFusionPattern *> DefinePatterns() override {
    return vector<BufferFusionPattern *>{};
  }
};
}

TEST_F(UBFUSION_ST, register_buitin_buffer_fusion_pass)
{
  mmSetEnv("ASCEND_HOME_PATH", "/home/jenkins/Ascend/ascend-toolkit/latest", 0);
  BufferFusionPassType type = BUFFER_FUSION_PASS_TYPE_RESERVED;
  REGISTER_BUFFER_FUSION_PASS("BuiltBufferFusionPassTest", type, st::BuiltBufferFusionPassTest);
  size_t size = BufferFusionPassRegistry::GetInstance().GetCreateFnByType(type).size();
  EXPECT_EQ(0, size);
  type = BUILT_IN_AI_CORE_BUFFER_FUSION_PASS;
  size_t size_before = BufferFusionPassRegistry::GetInstance().GetCreateFnByType(type).size();
  REGISTER_BUFFER_FUSION_PASS("", type, st::BuiltBufferFusionPassTest);
  size = BufferFusionPassRegistry::GetInstance().GetCreateFnByType(type).size();
  EXPECT_EQ(size, size_before);
  REGISTER_BUFFER_FUSION_PASS("test", type, st::BuiltBufferFusionPassTest);
  std::shared_ptr<GraphComm> graph_comm_ptr = std::make_shared<GraphComm>("engineName");
  graph_comm_ptr->Initialize();
  std::shared_ptr<FusionPriorityManager> fusion_priority_mgr_ptr =
      std::make_shared<FusionPriorityManager>("engineName", nullptr);
  fusion_priority_mgr_ptr->Initialize();
  std::shared_ptr<BufferFusion> sub_graph_optimizer_ptr =
      std::make_shared<BufferFusion>(graph_comm_ptr, fusion_priority_mgr_ptr, nullptr);
  auto graph = std::make_shared<ComputeGraph>("test");
  Status status = sub_graph_optimizer_ptr->RunRegisterBufferFusionPass(*graph, type);
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(UBFUSION_ST, create_original_fusion_op_graph) {
  vector<int64_t> dim = {4, -1, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc(shape);
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");
  relu_op->AddInputDesc(tenosr_desc);
  relu_op->AddOutputDesc(tenosr_desc);
  relu_op->AddOutputDesc(tenosr_desc);
  ge::AttrUtils::SetBool(relu_op, ATTR_NAME_UNKNOWN_SHAPE, true);

  OpDescPtr relu_op1 = std::make_shared<OpDesc>("relu1", "Relu");
  relu_op1->AddInputDesc(tenosr_desc);
  relu_op1->AddOutputDesc(tenosr_desc);
  relu_op1->AddOutputDesc(tenosr_desc);
  ge::AttrUtils::SetBool(relu_op1, ATTR_NAME_UNKNOWN_SHAPE, true);

  ComputeGraphPtr graph_out = std::make_shared<ComputeGraph>("test");
  ge::NodePtr relu_node1 = graph_out->AddNode(relu_op1);
  ge::NodePtr relu_node = graph_out->AddNode(relu_op);

  std::map<ge::AnchorPtr, ge::AnchorPtr> out_anchor_map;
  out_anchor_map.emplace(relu_node1->GetOutDataAnchor(0), relu_node->GetOutDataAnchor(0));
  out_anchor_map.emplace(relu_node1->GetOutDataAnchor(1), relu_node->GetOutDataAnchor(1));
  std::map<ge::NodePtr, std::map<ge::AnchorPtr, ge::AnchorPtr>> fusion_op_output_anchors_map;
  fusion_op_output_anchors_map.emplace(relu_node1, out_anchor_map);
  relu_op->SetExtAttr("OutEdgeAnchorMap", fusion_op_output_anchors_map);

  vector<ge::NodePtr> fus_nodelist;
  fus_nodelist.push_back(relu_node1);
  EXPECT_EQ(fus_nodelist.empty(), false);
  std::shared_ptr<GraphComm> graph_comm_ptr = std::make_shared<GraphComm>(AI_CORE_NAME);
  std::unique_ptr<FusionGraphMerge> ub_fusion_graph_merge_ptr_ =
          std::unique_ptr<FusionGraphMerge>(new (std::nothrow) UBFusionGraphMerge(SCOPE_ID_ATTR, graph_comm_ptr));
  ub_fusion_graph_merge_ptr_->CreateOriginalFusionOpGraph(relu_node, fus_nodelist);
}

TEST_F(UBFUSION_ST, create_original_fusion_op_graph1) {
  string compile_info_dummy = "compile_info_json,compile_info_key";
  const ge::Buffer kernel_buffer(20U, 0x11U);
  vector<int64_t> dim = {4, -1, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc(shape);
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");
  relu_op->AddInputDesc(tenosr_desc);
  relu_op->AddOutputDesc(tenosr_desc);
  relu_op->AddOutputDesc(tenosr_desc);
  ge::AttrUtils::SetBool(relu_op, ATTR_NAME_UNKNOWN_SHAPE, true);

  OpDescPtr relu_op1 = std::make_shared<OpDesc>("relu1", "Relu");
  relu_op1->AddInputDesc(tenosr_desc);
  relu_op1->AddOutputDesc(tenosr_desc);
  relu_op1->AddOutputDesc(tenosr_desc);
  ge::AttrUtils::SetBool(relu_op1, ATTR_NAME_UNKNOWN_SHAPE, true);
  AttrUtils::SetStr(relu_op1, COMPILE_INFO_JSON, compile_info_dummy);
  AttrUtils::SetBytes(relu_op1, ge::ATTR_NAME_TBE_KERNEL_BUFFER, kernel_buffer);

  ComputeGraphPtr graph_out = std::make_shared<ComputeGraph>("test");
  ge::NodePtr relu_node1 = graph_out->AddNode(relu_op1);
  ge::NodePtr relu_node = graph_out->AddNode(relu_op);

  std::map<ge::AnchorPtr, ge::AnchorPtr> out_anchor_map;
  out_anchor_map.emplace(relu_node1->GetOutDataAnchor(0), relu_node->GetOutDataAnchor(0));
  out_anchor_map.emplace(relu_node1->GetOutDataAnchor(1), relu_node->GetOutDataAnchor(1));
  std::map<ge::NodePtr, std::map<ge::AnchorPtr, ge::AnchorPtr>> fusion_op_output_anchors_map;
  fusion_op_output_anchors_map.emplace(relu_node1, out_anchor_map);
  relu_op->SetExtAttr("OutEdgeAnchorMap", fusion_op_output_anchors_map);

  vector<ge::NodePtr> fus_nodelist;
  fus_nodelist.push_back(relu_node1);

  std::shared_ptr<GraphComm> graph_comm_ptr = std::make_shared<GraphComm>(AI_CORE_NAME);
  std::unique_ptr<FusionGraphMerge> ub_fusion_graph_merge_ptr_ =
          std::unique_ptr<FusionGraphMerge>(new (std::nothrow) UBFusionGraphMerge(SCOPE_ID_ATTR, graph_comm_ptr));
  ub_fusion_graph_merge_ptr_->CreateOriginalFusionOpGraph(relu_node, fus_nodelist);
  ge::ComputeGraphPtr graph_ptr = nullptr;
  ge::OpDescPtr op_desc = relu_node1->GetOpDesc();
  bool has_attr = false;
  if (op_desc->HasAttr(ge::ATTR_NAME_TBE_KERNEL_BUFFER) || op_desc->HasAttr(COMPILE_INFO_JSON)) {
    has_attr = true;
  }
  EXPECT_EQ(has_attr, false);
}

using BufferFusionFn = BufferFusionPassBase *(*)();

class TestBufferFusionPass : public BufferFusionPassBase {
 protected:

  vector<BufferFusionPattern *> DefinePatterns() override {
    return {};
  };

};

fe::BufferFusionPassBase *BufferFunc() {
  return new(std::nothrow) TestBufferFusionPass();
}

void RegisterBufferFusionFunc(BufferFusionFn create_fn) {
  BufferFusionPassRegistry::GetInstance().RegisterPass(CUSTOM_AI_CORE_BUFFER_FUSION_PASS, "CUSTOM_PASS1", create_fn, 0);
  BufferFusionPassRegistry::GetInstance().RegisterPass(CUSTOM_AI_CORE_BUFFER_FUSION_PASS, "CUSTOM_PASS2", create_fn, 0);
  BufferFusionPassRegistry::GetInstance().RegisterPass(CUSTOM_AI_CORE_BUFFER_FUSION_PASS, "CUSTOM_PASS3", create_fn, 0);

  BufferFusionPassRegistry::GetInstance().RegisterPass(
      CUSTOM_VECTOR_CORE_BUFFER_FUSION_PASS, "CUSTOM_PASS4", create_fn, 0);
  BufferFusionPassRegistry::GetInstance().RegisterPass(
      CUSTOM_VECTOR_CORE_BUFFER_FUSION_PASS, "CUSTOM_PASS5", create_fn, 0);

  BufferFusionPassRegistry::GetInstance().RegisterPass(
      BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, "BUILT_IN_PASS1", create_fn, 0);
  BufferFusionPassRegistry::GetInstance().RegisterPass(
      BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, "BUILT_IN_PASS2", create_fn, 0);

  BufferFusionPassRegistry::GetInstance().RegisterPass(
      BUILT_IN_VECTOR_CORE_BUFFER_FUSION_PASS, "BUILT_IN_PASS3", create_fn, 0);
  BufferFusionPassRegistry::GetInstance().RegisterPass(
      BUILT_IN_VECTOR_CORE_BUFFER_FUSION_PASS, "BUILT_IN_PASS4", create_fn, 0);


}

TEST_F(UBFUSION_ST, converage_09) {
  mmSetEnv("ASCEND_HOME_PATH", "/home/jenkins/Ascend/ascend-toolkit/latest", 0);
  std::map<std::string, std::string> options;
  auto fe_ops_kernel_info_store = make_shared<fe::FEOpsKernelInfoStore>();
  FEOpsStoreInfo heavy_op_info{
      6,
      "tbe-builtin",
      EN_IMPL_HW_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/fusion_rule_manager",
      "",
      false,
      false,
      false};

  vector<FEOpsStoreInfo> store_info;
  store_info.emplace_back(heavy_op_info);
  Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
  OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

  string file_path =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_rule_parser/cycle_detection.json";
  fe_ops_kernel_info_store->Initialize(options);
  auto fusion_rule_mgr = std::make_shared<FusionRuleManager>(fe_ops_kernel_info_store);

  auto fusion_priority_mgr =
      std::make_shared<FusionPriorityManager>(AI_CORE_NAME, fusion_rule_mgr);
  auto fusion_priority_mgr_vec =
      std::make_shared<FusionPriorityManager>(VECTOR_CORE_NAME, fusion_rule_mgr);

  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_config_manager/custom_config/fusion_config3.json";
  fusion_priority_mgr->fusion_config_parser_ptr_ = std::make_unique<FusionConfigParser>(fe::AI_CORE_NAME);
  fusion_priority_mgr->fusion_config_parser_ptr_->ParseFusionConfigFile();

  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_config_manager/custom_config/fusion_config3.json";
  fusion_priority_mgr_vec->fusion_config_parser_ptr_ = std::make_unique<FusionConfigParser>(fe::VECTOR_CORE_NAME);
  fusion_priority_mgr_vec->fusion_config_parser_ptr_->ParseFusionConfigFile();

  RegisterBufferFusionFunc(BufferFunc);
  EXPECT_EQ(fe::SUCCESS, fusion_priority_mgr->SortBufferFusion());
  EXPECT_EQ(fe::SUCCESS, fusion_priority_mgr->SortBufferFusion());
  EXPECT_EQ(fe::SUCCESS, fusion_priority_mgr_vec->SortBufferFusion());
  BufferFusionPassRegistry instance;
  BufferFusionPassRegistry::GetInstance().impl_.swap(instance.impl_);
  OpsKernelManager::Instance(AI_CORE_NAME).Finalize();
}

TEST_F(UBFUSION_ST, change_scope_id_test) {
  int64_t old_scope_id = 0;
  int64_t new_scope_id = 2;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr op = std::make_shared<ge::OpDesc>("test", "Test");
  vector<int64_t> dims = {1, 2, 3};
  ge::GeShape shape = ge::GeShape(dims);
  ge::GeTensorDesc tensor(shape);
  op->AddInputDesc(tensor);
  op->AddOutputDesc(tensor);
  ge::NodePtr node1 = graph->AddNode(op);

  ScopeInfo inner;
  inner.nodes.emplace(std::make_pair(0, node1));
  inner.nodes.emplace(std::make_pair(1, node1));
  inner.nodes.emplace(std::make_pair(2, node1));
  inner.nodes.emplace(std::make_pair(3, node1));
  inner.nodes.emplace(std::make_pair(4, node1));
  inner.nodes.emplace(std::make_pair(5, node1));
  inner.nodes.emplace(std::make_pair(6, node1));
  inner.nodes.emplace(std::make_pair(7, node1));
  inner.nodes.emplace(std::make_pair(8, node1));
  inner.nodes.emplace(std::make_pair(9, node1));
  inner.nodes.emplace(std::make_pair(10, node1));
  inner.nodes.emplace(std::make_pair(11, node1));
  inner.nodes.emplace(std::make_pair(12, node1));
  inner.nodes.emplace(std::make_pair(13, node1));
  inner.nodes.emplace(std::make_pair(14, node1));
  inner.nodes.emplace(std::make_pair(15, node1));
  std::shared_ptr<AutomaticBufferFusion> auto_buffer_fusion_ptr_ = std::make_shared<AutomaticBufferFusion>(nullptr);
  auto_buffer_fusion_ptr_->scope_id_nodes_map_.emplace(std::make_pair(0, inner));
  auto_buffer_fusion_ptr_->scope_id_nodes_map_.emplace(std::make_pair(2, inner));
  Status ret = auto_buffer_fusion_ptr_->ChangeScopeId(old_scope_id, new_scope_id);
  EXPECT_EQ(ret, GRAPH_OPTIMIZER_NOT_FUSE_TWO_SCOPE);
}

TEST_F(UBFUSION_ST, test_vector_core_buffer_fusion) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  RunUbFusion(graph, fe::VECTOR_CORE_NAME);
  EXPECT_EQ(graph->GetDirectNode().size(), 0);
}

namespace fe {
extern void CopyBasicInfo(const ffts::ThreadSliceMap &a, const ffts::ThreadSliceMapPtr &b);
extern Status CopyOutputSliceInfo(const ffts::ThreadSliceMap &a, size_t output_index, const ffts::ThreadSliceMapPtr &b,
                                  bool &is_consistent);
extern Status CopyInputSliceInfo(const ffts::ThreadSliceMap &a, size_t input_index, const ffts::ThreadSliceMapPtr &b);
}
TEST_F(UBFUSION_ST, fusion_slice_info_test) {
  ffts::ThreadSliceMapPtr slice_b_ptr = std::make_shared<ffts::ThreadSliceMap>();
  ffts::ThreadSliceMap slice_a;
  slice_b_ptr->thread_mode = 1;
  slice_a.thread_mode = 1;
  bool is_consistent;
  EXPECT_EQ(fe::SUCCESS, CopyOutputSliceInfo(slice_a, 0, slice_b_ptr, is_consistent));
  CopyBasicInfo(slice_a, slice_b_ptr);
  slice_b_ptr->thread_mode = 1;
  EXPECT_EQ(fe::SUCCESS, CopyInputSliceInfo(slice_a, 0, slice_b_ptr));
}

TEST_F(UBFUSION_ST, merge_mixl2_fusion_graph) {
  OpDescPtr data1 = std::make_shared<OpDesc>("Data1", "Data");
  OpDescPtr data2 = std::make_shared<OpDesc>("Data2", "Data");
  OpDescPtr conv_op= std::make_shared<OpDesc>("conv1", "Conv2D");
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr out_op = std::make_shared<OpDesc>("output", "NetOut");

  vector<int64_t> dim = {1, 1, 4, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc(shape);
  
  data1->AddOutputDesc(tenosr_desc);
  data2->AddOutputDesc(tenosr_desc);
  conv_op->AddInputDesc(tenosr_desc);
  conv_op->AddInputDesc(tenosr_desc);
  conv_op->AddOutputDesc(tenosr_desc);
  relu_op->AddInputDesc(tenosr_desc);
  relu_op->AddOutputDesc(tenosr_desc);
  out_op->AddInputDesc(tenosr_desc);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::NodePtr data_node1 = graph->AddNode(data1);
  ge::NodePtr data_node2 = graph->AddNode(data2);
  ge::NodePtr conv_node = graph->AddNode(conv_op);
  ge::NodePtr relu_node = graph->AddNode(relu_op);
  ge::NodePtr out_node = graph->AddNode(out_op);
  GraphUtils::AddEdge(data_node1->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(data_node2->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));

  ge::AttrUtils::SetInt(conv_node->GetOpDesc(), SCOPE_ID_ATTR, 1);
  ge::AttrUtils::SetInt(relu_node->GetOpDesc(), SCOPE_ID_ATTR, 1);
  ge::AttrUtils::SetInt(conv_node->GetOpDesc(), FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetInt(relu_node->GetOpDesc(), FE_IMPLY_TYPE, 6);
  ge::AttrUtils::SetStr(conv_node->GetOpDesc(), TVM_ATTR_NAME_MAGIC, "FFTS_BINARY_MAGIC_ELF_MIX_AIC");
  ge::AttrUtils::SetStr(conv_node->GetOpDesc(), ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX_AIC");
  ge::AttrUtils::SetInt(conv_node->GetOpDesc(), kModeInArgsFirstField, 1);
  std::vector<int64_t> tvm_workspace_sizes = {1};
  conv_node->GetOpDesc()->SetWorkspaceBytes(tvm_workspace_sizes);
  std::vector<int64_t> parameters_index = {0,0,1,0,0};
  ge::AttrUtils::SetListInt(conv_node->GetOpDesc(), "ub_atomic_params", parameters_index);
  vector<char> buffer;
  conv_node->GetOpDesc()->SetExtAttr(string("_mix_aic") + ge::OP_EXTATTR_NAME_TBE_KERNEL, std::make_shared<OpKernelBin>("tbe", std::move(buffer)));
  conv_node->GetOpDesc()->SetExtAttr(string("_mix_aiv") + ge::OP_EXTATTR_NAME_TBE_KERNEL, std::make_shared<OpKernelBin>("tbe", std::move(buffer)));

  std::vector<uint32_t> input_tensor_indexes = {1};
  std::vector<uint32_t> output_tensor_indexes = {0};
  ffts::ThreadSliceMapPtr conv_slice_info = std::make_shared<ffts::ThreadSliceMap>();
  conv_slice_info->thread_mode = 1;
  conv_slice_info->same_atomic_clean_nodes = {"conv1"};
  conv_node->GetOpDesc()->SetExtAttr(ffts::kAttrSgtStructInfo, conv_slice_info);
  (void)ge::AttrUtils::SetListInt(conv_node->GetOpDesc(), ge::kInputTensorIndexs, input_tensor_indexes);
  (void)ge::AttrUtils::SetListInt(conv_node->GetOpDesc(), ge::kOutputTensorIndexs, output_tensor_indexes);

  std::vector<uint32_t> input_tensor_indexes_1 = {0};
  std::vector<uint32_t> output_tensor_indexes_1 = {0};
  (void)ge::AttrUtils::SetListInt(relu_node->GetOpDesc(), ge::kInputTensorIndexs, input_tensor_indexes_1);
  (void)ge::AttrUtils::SetListInt(relu_node->GetOpDesc(), ge::kOutputTensorIndexs, output_tensor_indexes_1);
  relu_node->GetOpDesc()->SetExtAttr(ffts::kAttrSgtStructInfo, conv_slice_info);

  std::shared_ptr<GraphComm> graph_comm_ptr = std::make_shared<GraphComm>(AI_CORE_NAME);
  std::unique_ptr<FusionGraphMerge> ub_fusion_graph_merge_ptr_ =
        std::unique_ptr<FusionGraphMerge>(new (std::nothrow) UBFusionGraphMerge(SCOPE_ID_ATTR, graph_comm_ptr));
  ub_fusion_graph_merge_ptr_->MergeFusionGraph(*graph);

  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "conv1relu1") {
      std::vector<int64_t> output_index;
      std::vector<int64_t> exp_output_index = {1};
      ge::AttrUtils::GetListInt(node->GetOpDesc(), TBE_OP_ATOMIC_OUTPUT_INDEX, output_index);
      EXPECT_EQ(output_index, exp_output_index);
      std::vector<uint32_t> input_tensor_indexes;
      std::vector<uint32_t> output_tensor_indexes;
      (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), ge::kInputTensorIndexs, input_tensor_indexes);
      (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), ge::kOutputTensorIndexs, output_tensor_indexes);
      EXPECT_EQ(input_tensor_indexes.size(), 1);
      EXPECT_EQ(output_tensor_indexes.size(), 1);

      ffts::ThreadSliceMapPtr slice_info = nullptr;
      slice_info = node->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info);
      vector<string> result = {"conv1"};
      EXPECT_EQ(slice_info->same_atomic_clean_nodes, result);
    }
  }
}

TEST_F(UBFUSION_ST, outputInplaceAbilityx0) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
  OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", fe::DATA);
  OpDescPtr data2 = std::make_shared<OpDesc>("DATA2", fe::DATA);
  OpDescPtr data3 = std::make_shared<OpDesc>("DATA3", fe::DATA);
  OpDescPtr conv = std::make_shared<OpDesc>("conv", "Convolution");
  OpDescPtr dequant = std::make_shared<OpDesc>("dequant", "Dequant");
  OpDescPtr dequant1 = std::make_shared<OpDesc>("dequant1", "Dequant");
  OpDescPtr output = std::make_shared<OpDesc>("output", "output");
  SetPattern(conv, "Convolution");
  SetPattern(dequant, "dequant");
  SetPattern(dequant1, "dequant"); 
  SetPattern(output, "output");
  // add descriptor
  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  data->AddOutputDesc(out_desc);
  data1->AddOutputDesc(out_desc);
  data2->AddOutputDesc(out_desc);
  data3->AddOutputDesc(out_desc);
  conv->AddInputDesc(out_desc);
  conv->AddInputDesc(out_desc);
  conv->AddOutputDesc(out_desc);
  dequant->AddInputDesc(out_desc);
  dequant->AddInputDesc(out_desc);
  dequant->AddOutputDesc(out_desc);
  dequant1->AddInputDesc(out_desc);
  dequant1->AddInputDesc(out_desc);
  dequant1->AddOutputDesc(out_desc);
  output->AddInputDesc(out_desc);
  AttrUtils::SetInt(conv, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
  std::vector<int64_t> params = {0, 0, 0, 0, 0, 1, 0, 1};
  AttrUtils::SetListInt(conv, "ub_atomic_params", params);
  conv->SetWorkspaceBytes({0});
  AttrUtils::SetInt(dequant, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
  AttrUtils::SetInt(dequant1, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
  NodePtr data_node = graph->AddNode(data);
  NodePtr data1_node = graph->AddNode(data1);
  NodePtr data2_node = graph->AddNode(data2);
  NodePtr data3_node = graph->AddNode(data3);
  NodePtr conv_node = graph->AddNode(conv);
  NodePtr dequant_node = graph->AddNode(dequant);
  NodePtr dequant1_node = graph->AddNode(dequant1);
  NodePtr output_node = graph->AddNode(output);
  const char tbe_bin[] = "tbe_bin";
  vector<char> buffer(tbe_bin, tbe_bin+strlen(tbe_bin));
  ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
      conv_node->GetName(), std::move(buffer));
  conv_node->GetOpDesc()->SetExtAttr(
      OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
      conv_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0),
      conv_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
      dequant_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0),
      dequant_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(dequant_node->GetOutDataAnchor(0),
      dequant1_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(data3_node->GetOutDataAnchor(0),
      dequant1_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(dequant1_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
  ge::AttrUtils::SetListListInt(output, kAttrOutputInplaceAbility, {{0,0}});
  ge::AttrUtils::SetListListInt(dequant1, kAttrOutputInplaceAbility, {{0,0}, {0, 1}, {0,2}});
  ge::AttrUtils::SetListListInt(dequant, kAttrOutputInplaceAbility, {{0,0}, {0, 1}, {0,2}});
  ge::AttrUtils::SetListListInt(conv, kAttrOutputInplaceAbility, {{0,1}});
  int64_t scope_id = ScopeAllocator::Instance().AllocateScopeId();
  ScopeAllocator::SetScopeAttr(conv, scope_id);
  ScopeAllocator::SetScopeAttr(dequant, scope_id);
  ScopeAllocator::SetScopeAttr(dequant1, scope_id);
  vector<ge::NodePtr> fus_nodelist = {conv_node, dequant_node, dequant1_node};
  auto graph_common = std::make_shared<GraphComm>("engineName");
  graph_common->Initialize();
  auto fusion_graph_merge_ptr =
      std::make_shared<FusionGraphMerge>("fusion_scope", graph_common);
  EXPECT_EQ(fusion_graph_merge_ptr->MergeEachFusionNode(*graph, fus_nodelist), fe::SUCCESS);
}

TEST_F(UBFUSION_ST, outputInplaceAbilityx01) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
  OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", fe::DATA);
  OpDescPtr data2 = std::make_shared<OpDesc>("DATA2", fe::DATA);
  OpDescPtr data3 = std::make_shared<OpDesc>("DATA3", fe::DATA);
  OpDescPtr conv = std::make_shared<OpDesc>("conv", "Convolution");
  OpDescPtr dequant = std::make_shared<OpDesc>("dequant", "Dequant");
  OpDescPtr dequant1 = std::make_shared<OpDesc>("dequant1", "Dequant");
  OpDescPtr output = std::make_shared<OpDesc>("output", "output");
  OpDescPtr A = std::make_shared<OpDesc>("A", "A");
  OpDescPtr B = std::make_shared<OpDesc>("B", "B");
  SetPattern(conv, "Convolution");
  SetPattern(dequant, "dequant");
  SetPattern(dequant1, "dequant"); 
  SetPattern(output, "output");
  SetPattern(A, "A"); 
  SetPattern(B, "B");
  // add descriptor
  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  data->AddOutputDesc(out_desc);
  data1->AddOutputDesc(out_desc);
  data2->AddOutputDesc(out_desc);
  data3->AddOutputDesc(out_desc);
  conv->AddInputDesc(out_desc);
  conv->AddInputDesc(out_desc);
  conv->AddOutputDesc(out_desc);
  dequant->AddInputDesc(out_desc);
  dequant->AddInputDesc(out_desc);
  dequant->AddOutputDesc(out_desc);
  dequant1->AddInputDesc(out_desc);
  dequant1->AddInputDesc(out_desc);
  dequant1->AddOutputDesc(out_desc);
  output->AddInputDesc(out_desc);
  output->AddOutputDesc(out_desc);
  output->AddOutputDesc(out_desc);
  A->AddInputDesc(out_desc);
  A->AddOutputDesc(out_desc);
  B->AddInputDesc(out_desc);
  B->AddOutputDesc(out_desc);
  AttrUtils::SetInt(conv, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
  std::vector<int64_t> params = {0, 0, 0, 0, 0, 1, 0, 1};
  AttrUtils::SetListInt(conv, "ub_atomic_params", params);
  conv->SetWorkspaceBytes({0});
  AttrUtils::SetInt(dequant, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
  AttrUtils::SetInt(dequant1, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
  NodePtr data_node = graph->AddNode(data);
  NodePtr data1_node = graph->AddNode(data1);
  NodePtr data2_node = graph->AddNode(data2);
  NodePtr data3_node = graph->AddNode(data3);
  NodePtr conv_node = graph->AddNode(conv);
  NodePtr dequant_node = graph->AddNode(dequant);
  NodePtr dequant1_node = graph->AddNode(dequant1);
  NodePtr output_node = graph->AddNode(output);
  NodePtr A_node = graph->AddNode(A);
  NodePtr B_node = graph->AddNode(B);
  const char tbe_bin[] = "tbe_bin";
  vector<char> buffer(tbe_bin, tbe_bin+strlen(tbe_bin));
  ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
      conv_node->GetName(), std::move(buffer));
  conv_node->GetOpDesc()->SetExtAttr(
      OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
      conv_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0),
      conv_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
      dequant_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0),
      dequant_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(dequant_node->GetOutDataAnchor(0),
      dequant1_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(data3_node->GetOutDataAnchor(0),
      dequant1_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(dequant1_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(output_node->GetOutDataAnchor(0), A_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(output_node->GetOutDataAnchor(1), B_node->GetInDataAnchor(0));
  ge::AttrUtils::SetListListInt(output, kAttrOutputInplaceAbility, {{0,0}, {1, 0}});
  ge::AttrUtils::SetListListInt(dequant1, kAttrOutputInplaceAbility, {{0,0}, {0, 1}});
  ge::AttrUtils::SetListListInt(dequant, kAttrOutputInplaceAbility, {{0,0}, {0, 1}});
  ge::AttrUtils::SetListListInt(conv, kAttrOutputInplaceAbility, {{0,1}});
  int64_t scope_id = ScopeAllocator::Instance().AllocateScopeId();
  ScopeAllocator::SetScopeAttr(conv, scope_id);
  ScopeAllocator::SetScopeAttr(dequant, scope_id);
  ScopeAllocator::SetScopeAttr(dequant1, scope_id);
  ScopeAllocator::SetScopeAttr(output, scope_id);
  vector<ge::NodePtr> fus_nodelist = {conv_node, dequant_node, output_node, dequant1_node};
  auto graph_common = std::make_shared<GraphComm>("engineName");
  graph_common->Initialize();
  auto fusion_graph_merge_ptr =
      std::make_shared<FusionGraphMerge>("fusion_scope", graph_common);
  EXPECT_EQ(fusion_graph_merge_ptr->MergeEachFusionNode(*graph, fus_nodelist), fe::SUCCESS);
}


TEST_F(UBFUSION_ST, test_mark_duplcated_nodes) {
  auto graph = std::make_shared<ge::ComputeGraph>("test");
  GraphConstructor test(graph, "test", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                        GeShape({288, 8, 24, 33}));
  test.AddOpDesc(EN_IMPL_HW_TBE, "A", "conv1", "Conv2D", 1, 1)
      .SetInputs({"Data_1:0"})
      .AddOpDesc(EN_IMPL_HW_TBE, "B", "relu1", "Relu", 1, 1)
      .SetInputs({"conv1:0"})
      .AddOpDesc(EN_IMPL_HW_TBE, "A", "conv2", "Conv2D", 1, 1)
      .SetInputs({"Data_1:0"})
      .AddOpDesc(EN_IMPL_HW_TBE, "B", "relu2", "Relu", 1, 1)
      .SetInputs({"conv2:0"})
      .AddOpDesc("out", "NetOutput", 2, 1)
      .SetInputs({"relu1:0", "relu2:0"});

  ge::NodePtr conv1 = nullptr;
  test.GetNodeByName("conv1", conv1);
  ge::NodePtr conv2 = nullptr;
  test.GetNodeByName("conv2", conv2);
  ge::NodePtr relu1 = nullptr;
  test.GetNodeByName("relu1", relu1);
  ge::NodePtr relu2 = nullptr;
  test.GetNodeByName("relu2", relu2);

  ge::AttrUtils::SetInt(conv1->GetOpDesc(), SCOPE_ID_ATTR, 2);
  ge::AttrUtils::SetInt(relu1->GetOpDesc(), SCOPE_ID_ATTR, 2);

  auto conv_nodes = std::vector<ge::NodePtr>({conv1, conv2});
  for (auto &node : conv_nodes) {
    ge::AttrUtils::SetStr(node->GetOpDesc(), TVM_ATTR_NAME_MAGIC, "FFTS_BINARY_MAGIC_ELF_MIX_AIC");
    ge::AttrUtils::SetStr(node->GetOpDesc(), ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX_AIC");
    ge::AttrUtils::SetInt(node->GetOpDesc(), kModeInArgsFirstField, 1);
    std::vector<int64_t> parameters_index = {0,0,0,1};
    ge::AttrUtils::SetListInt(node->GetOpDesc(), "ub_atomic_params", parameters_index);
    vector<char> buffer;
    node->GetOpDesc()->SetExtAttr(string("_mix_aic") + ge::OP_EXTATTR_NAME_TBE_KERNEL,
                                  std::make_shared<OpKernelBin>("tbe", std::move(buffer)));
    node->GetOpDesc()->SetExtAttr(string("_mix_aiv") + ge::OP_EXTATTR_NAME_TBE_KERNEL,
                                  std::make_shared<OpKernelBin>("tbe", std::move(buffer)));

  }

  ffts::ThreadSliceMapPtr conv1_slice_info = std::make_shared<ffts::ThreadSliceMap>();
  conv1_slice_info->thread_mode = 0;
  conv1_slice_info->slice_instance_num = 4;
  conv1_slice_info->original_node = "conv";
  conv1_slice_info->thread_id = 1;
  conv1->GetOpDesc()->SetExtAttr(ffts::kAttrSgtStructInfo, conv1_slice_info);
  ge::AttrUtils::SetInt(conv1->GetOpDesc(), kThreadMode, 0);

  ffts::ThreadSliceMapPtr conv2_slice_info = std::make_shared<ffts::ThreadSliceMap>();
  conv2_slice_info->thread_mode = 0;
  conv2_slice_info->slice_instance_num = 4;
  conv2_slice_info->original_node = "conv";
  conv2_slice_info->thread_id = 2;
  conv2->GetOpDesc()->SetExtAttr(ffts::kAttrSgtStructInfo, conv2_slice_info);
  ge::AttrUtils::SetInt(conv2->GetOpDesc(), kThreadMode, 0);

  ffts::ThreadSliceMapPtr relu1_slice_info = std::make_shared<ffts::ThreadSliceMap>();
  relu1_slice_info->thread_mode = 0;
  relu1_slice_info->slice_instance_num = 4;
  relu1_slice_info->original_node = "relu";
  relu1_slice_info->thread_id = 1;
  relu1->GetOpDesc()->SetExtAttr(ffts::kAttrSgtStructInfo, relu1_slice_info);
  ge::AttrUtils::SetInt(relu1->GetOpDesc(), kThreadMode, 0);

  ffts::ThreadSliceMapPtr relu2_slice_info = std::make_shared<ffts::ThreadSliceMap>();
  relu2_slice_info->thread_mode = 0;
  relu2_slice_info->slice_instance_num = 4;
  relu2_slice_info->original_node = "relu";
  relu2_slice_info->thread_id = 2;
  relu2->GetOpDesc()->SetExtAttr(ffts::kAttrSgtStructInfo, relu2_slice_info);
  ge::AttrUtils::SetInt(relu2->GetOpDesc(), kThreadMode, 0);

  std::shared_ptr<GraphComm> graph_comm_ptr = std::make_shared<GraphComm>(AI_CORE_NAME);
  std::unique_ptr<FusionGraphMerge> ub_fusion_graph_merge_ptr_ =
      std::unique_ptr<FusionGraphMerge>(new (std::nothrow) UBFusionGraphMerge(SCOPE_ID_ATTR, graph_comm_ptr));
  OpCompilerPtr op_compiler_ptr = make_shared<OpCompiler>("compiler_name", AI_CORE_NAME, nullptr);
  op_compiler_ptr->MarkDuplicatedNode(*graph);

  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "conv2") {
      ffts::ThreadSliceMapPtr slice_info = nullptr;
      slice_info = node->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info);
      EXPECT_NE(slice_info, nullptr);

      ge::NodePtr thread0_node = nullptr;
      thread0_node = node->GetOpDesc()->TryGetExtAttr(kAttrThread1Node, thread0_node);
      EXPECT_NE(thread0_node, nullptr);

      std::shared_ptr<std::vector<ge::NodePtr>> related_nodes = nullptr;
      related_nodes = node->GetOpDesc()->TryGetExtAttr(kAttrRelatedThreadsNodes, related_nodes);
      EXPECT_EQ(related_nodes, nullptr);
    }

    if (node->GetName() == "conv1") {
      ffts::ThreadSliceMapPtr slice_info = nullptr;
      slice_info = node->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info);
      EXPECT_NE(slice_info, nullptr);

      ge::NodePtr thread0_node = nullptr;
      thread0_node = node->GetOpDesc()->TryGetExtAttr(kAttrThread1Node, thread0_node);
      EXPECT_EQ(thread0_node, nullptr);

      std::shared_ptr<std::vector<ge::NodePtr>> related_nodes = nullptr;
      related_nodes = node->GetOpDesc()->TryGetExtAttr(kAttrRelatedThreadsNodes, related_nodes);
      EXPECT_NE(related_nodes, nullptr);
      EXPECT_EQ(related_nodes->size(), 1);
    }
  }

  ub_fusion_graph_merge_ptr_->MergeFusionGraph(*graph);
  int32_t count_fusion_nodes = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "conv1relu1" || node->GetName() == "conv2relu2") {
      ++count_fusion_nodes;
    }
  }
  EXPECT_EQ(count_fusion_nodes, 2);
}
