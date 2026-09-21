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
#include <algorithm>
#include <list>
#include "ffts_llt_utils.h"
#define private public
#include "base/registry/op_impl_space_registry_v2.h"
#undef private
#define private public
#define protected public
#include "common/aicore_util_attr_define.h"
#include "common/sgt_slice_type.h"
#include "common/fe_gentask_utils.h"
#include "inc/ffts_log.h"
#include "inc/ffts_error_codes.h"
#include "inc/ffts_type.h"
#include "engine/engine_manager.h"
#include "task_builder/fftsplus_ops_kernel_builder.h"
#include "task_builder/mode/auto/auto_thread_task_builder.h"
#include "task_builder/mode/manual/manual_thread_task_builder.h"
#include "register/op_ext_gentask_registry.h"
#include "graph/node.h"
#include "graph_builder_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/anchor_utils.h"
#include "graph/anchor.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "runtime/context.h"
#include "runtime/stream.h"
#include "runtime/rt_model.h"
#include "runtime/kernel.h"
#include "runtime/mem.h"
#include "../test_utils.h"
#include "inc/ffts_configuration.h"
using namespace std;
using namespace ffts;
using namespace ge;
using FFTSPlusOpsKernelBuilderPtr = shared_ptr<FFTSPlusOpsKernelBuilder>;
const std::string kTaskRadio = "_task_ratio";

#define SET_SIZE 10000

Status ScheculePolicyPassStub1(domi::TaskDef &, std::vector<ffts::FftsPlusContextPath> &) {
  return ffts::FAILED;
}

Status ScheculePolicyPassStub2(domi::TaskDef &, std::vector<ffts::FftsPlusContextPath> &) {
  return ffts::SUCCESS;
}

class FFTSPlusOpsKernelBuilderUTest : public testing::Test{
 protected:
  static void SetUpTestCase() {
    cout << "FFTSPlusOpsKernelBuilder SetUpTestCase" << endl;
  }
  static void TearDownTestCase() {
    cout << "FFTSPlusOpsKernelBuilder TearDownTestCase" << endl;
  }

  // Some expensive resource shared by all tests.
  virtual void SetUp(){
    ffts_plus_ops_kernel_builder_ptr = make_shared<FFTSPlusOpsKernelBuilder>();
    ffts_plus_manual_mode_ptr = make_shared<ManualTheadTaskBuilder>();
    ffts_plus_auto_mode_ptr = make_shared<AutoTheadTaskBuilder>();
    std::map<std::string, std::string> options;
    ffts_plus_ops_kernel_builder_ptr->Initialize(options);
    context_ = CreateContext();
    auto space_registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
    gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);
  }
  virtual void TearDown(){
    cout << "a test Tear Down" << endl;
    ffts_plus_ops_kernel_builder_ptr->Finalize();
  }

  static RunContext CreateContext()
  {
    RunContext context;
    context.dataMemSize = 101;
    context.dataMemBase = (uint8_t *) (intptr_t) 1000;
    context.weightMemSize = 200;
    context.weightMemBase = (uint8_t *) (intptr_t) 1101;
    context.weightsBuffer = Buffer(20);

    return context;
  }
 public:
  FFTSPlusOpsKernelBuilderPtr ffts_plus_ops_kernel_builder_ptr;
  ManualTheadTaskBuilderPtr ffts_plus_manual_mode_ptr;
  AutoTheadTaskBuilderPtr ffts_plus_auto_mode_ptr;
  RunContext context_;
};

void SetOpDecSize(NodePtr& node) {
  OpDesc::Vistor<GeTensorDesc> tensors = node->GetOpDesc()->GetAllInputsDesc();
  for (int i = 0; i < node->GetOpDesc()->GetAllInputsDesc().size(); i++) {
    ge::GeTensorDesc tensor = node->GetOpDesc()->GetAllInputsDesc().at(i);
    ge::TensorUtils::SetSize(tensor, SET_SIZE);
    node->GetOpDesc()->UpdateInputDesc(i, tensor);
  }
  OpDesc::Vistor<GeTensorDesc> tensorsOutput = node->GetOpDesc()->GetAllOutputsDesc();
  for (int i = 0; i < tensorsOutput.size(); i++) {
    ge::GeTensorDesc tensorOutput = tensorsOutput.at(i);
    ge::TensorUtils::SetSize(tensorOutput, SET_SIZE);
    node->GetOpDesc()->UpdateOutputDesc(i, tensorOutput);
  }
}

static Status GenManAicAivCtxDef(NodePtr node) {
  std::shared_ptr<domi::FftsPlusCtxDef> ffts_plus_def_ptr = make_shared<domi::FftsPlusCtxDef>();
  domi::FftsPlusAicAivCtxDef *aic_aiv_ctx_def = ffts_plus_def_ptr->mutable_aic_aiv_ctx();
  FFTS_CHECK_NOTNULL(aic_aiv_ctx_def);

  aic_aiv_ctx_def->set_prefetch_once_bitmap(0);
  aic_aiv_ctx_def->set_prefetch_enable_bitmap(0);
  aic_aiv_ctx_def->set_aten(ffts::kManualMode);
  aic_aiv_ctx_def->set_atm(ffts::kManualMode);
  aic_aiv_ctx_def->set_thread_dim(4);

  int32_t block_dim = 1;
  aic_aiv_ctx_def->set_tail_block_dim(static_cast<uint32_t>(block_dim));
  aic_aiv_ctx_def->set_non_tail_block_dim(static_cast<uint32_t>(block_dim));

  //input
  aic_aiv_ctx_def->add_task_addr(111);
  aic_aiv_ctx_def->add_task_addr(222);

  //output
  aic_aiv_ctx_def->add_task_addr(333);

  //workspace
  aic_aiv_ctx_def->add_task_addr(444);

  auto op_desc = node->GetOpDesc();
  string attr_key_kernel_name = kKernelName;
  string attr_kernel_name;
  (void) ge::AttrUtils::GetStr(op_desc, attr_key_kernel_name, attr_kernel_name);
  aic_aiv_ctx_def->add_kernel_name(attr_kernel_name);
  (void)ge::AttrUtils::SetInt(op_desc, ffts::kAttrAICoreCtxType,
                              static_cast<int64_t>(ffts::TaskBuilderType::EN_TASK_TYPE_AIC_AIV));
  std::string core_type = "AIC";
  (void)ge::AttrUtils::SetStr(op_desc, ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type);
  (void)op_desc->SetExtAttr(ffts::kAttrAICoreCtxDef, ffts_plus_def_ptr);
  return ffts::SUCCESS;
}

/*
 * Data -cast - netoutput
 */
ComputeGraphPtr BuildGraph_Readonly_Subgraph(const string &subraph_name, const bool &thread_mode){
  auto sub_builder = ut::ComputeGraphBuilder(subraph_name);
  auto data1 = sub_builder.AddNode("sdma1", "HcomAllReduce", 0,1);
  auto cast = sub_builder.AddNode("sdma2", "HcomAllReduce", 1,1);
  auto netoutput = sub_builder.AddNode("sdma3","HcomAllReduce", 1,1);
  AttrUtils::SetInt(data1->GetOpDesc(),ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(netoutput->GetOpDesc(),ATTR_NAME_PARENT_NODE_INDEX,0);

  ThreadSliceMap thread_slice_map;
  thread_slice_map.thread_mode = thread_mode;
  ThreadSliceMapPtr thread_slice_map_ptr = std::make_shared<ThreadSliceMap>(thread_slice_map);
  data1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, thread_slice_map_ptr);
  cast->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, thread_slice_map_ptr);
  netoutput->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, thread_slice_map_ptr);

  sub_builder.AddDataEdge(data1,0,cast,0);
  sub_builder.AddDataEdge(cast,0,netoutput,0);
  return sub_builder.GetGraph();
}

/*
 *      const - allreduce
 *            \ if
 *         insert identity
 */
ComputeGraphPtr BuildGraph_Readonly_ScopeWrite() {
  auto builder = ut::ComputeGraphBuilder("test");
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto ctrl_const = builder.AddNode("ctrl_const", CONSTANT, 0, 1);
  auto allreduce = builder.AddNode("allreduce", "allreduce", 1, 1);
  auto if_node = builder.AddNode("if", "if", 1,0);

  builder.AddDataEdge(const1, 0, allreduce, 0);
  builder.AddDataEdge(const1, 0, if_node, 0);
  builder.AddControlEdge(ctrl_const, allreduce);

  auto root_graph = builder.GetGraph();
  string subgraph_name = "then_branch";
  ComputeGraphPtr then_branch_graph = BuildGraph_Readonly_Subgraph(subgraph_name, false);
  then_branch_graph->SetParentNode(if_node);
  then_branch_graph->SetParentGraph(root_graph);
  if_node->GetOpDesc()->AddSubgraphName(subgraph_name);
  if_node->GetOpDesc()->SetSubgraphInstanceName(0,subgraph_name);
  root_graph->AddSubgraph(subgraph_name, then_branch_graph);
  return root_graph;
}

static Status GenManualMixAICAIVCtxDef(NodePtr node) {
  auto op_desc = node->GetOpDesc();
  std::shared_ptr<domi::FftsPlusCtxDef> ffts_plus_def_ptr = make_shared<domi::FftsPlusCtxDef>();
  domi::FftsPlusMixAicAivCtxDef *mix_aic_aiv_ctx_def = ffts_plus_def_ptr->mutable_mix_aic_aiv_ctx();
  FFTS_CHECK_NOTNULL(mix_aic_aiv_ctx_def);

  uint32_t task_ratio;
  (void)ge::AttrUtils::GetInt(op_desc, kTaskRadio, task_ratio);

  mix_aic_aiv_ctx_def->set_prefetch_once_bitmap(0);
  mix_aic_aiv_ctx_def->set_prefetch_enable_bitmap(0);
  mix_aic_aiv_ctx_def->set_ns(1);
  mix_aic_aiv_ctx_def->set_atm(ffts::kManualMode);
  mix_aic_aiv_ctx_def->set_thread_dim(1);

  mix_aic_aiv_ctx_def->set_tail_block_ratio_n(task_ratio);
  mix_aic_aiv_ctx_def->set_non_tail_block_ratio_n(task_ratio);

  int32_t block_dim = 0;
  (void)ge::AttrUtils::GetInt(op_desc, ge::TVM_ATTR_NAME_BLOCKDIM, block_dim);
  mix_aic_aiv_ctx_def->set_tail_block_dim(static_cast<uint32_t>(block_dim));
  mix_aic_aiv_ctx_def->set_non_tail_block_dim(static_cast<uint32_t>(block_dim));

  // modeInArgsFirstField
  uint32_t mode = 0;
  (void)ge::AttrUtils::GetInt(op_desc, kModeInArgsFirstField, mode);
  // mode == 1 indicates we need reserve 8 Bytes for the args beginning
  if (mode == 1) {
    uint64_t modeArgs = 0;
    mix_aic_aiv_ctx_def->add_task_addr(modeArgs);
  }

  //input
  mix_aic_aiv_ctx_def->add_task_addr(222);
  //output
  mix_aic_aiv_ctx_def->add_task_addr(333);

  for (auto &prefix : kMixPrefixs) {
    string attr_key_kernel_name = prefix + kKernelName;
    string attr_kernel_name;
    (void)ge::AttrUtils::GetStr(op_desc, attr_key_kernel_name, attr_kernel_name);
    mix_aic_aiv_ctx_def->add_kernel_name(attr_kernel_name);
  }
  (void)ge::AttrUtils::SetInt(op_desc, ffts::kAttrAICoreCtxType, static_cast<int64_t>(ffts::TaskBuilderType::EN_TASK_TYPE_MIX_AIC_AIV));
  (void)op_desc->SetExtAttr(ffts::kAttrAICoreCtxDef, ffts_plus_def_ptr);
  return ffts::SUCCESS;
}


ComputeGraphPtr BuildGraph_Mix_Subgraph(const string &subraph_name) {
  auto builder = ut::ComputeGraphBuilder(subraph_name);
  auto lstm = builder.AddNode("LSTM", "LSTM", 2, 1, ge::FORMAT_NCHW, ge::DT_FLOAT, {2, 4, 4, 4});
  ge::AttrUtils::SetInt(lstm->GetOpDesc(), ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
  ge::AttrUtils::SetInt(lstm->GetOpDesc(), kModeInArgsFirstField, 1);
  ge::AttrUtils::SetInt(lstm->GetOpDesc(), kTaskRadio, 2);
  ge::AttrUtils::SetStr(lstm->GetOpDesc(), "_cube_vector_core_type", "MIX_AIC");
  (void)ge::AttrUtils::SetInt(lstm->GetOpDesc(), kAttrAICoreCtxType,
                              static_cast<int64_t>(ffts::TaskBuilderType::EN_TASK_TYPE_MIX_AIC_AIV));
  ge::AttrUtils::SetStr(lstm->GetOpDesc(), ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_MIX_AIC");
  ge::AttrUtils::SetInt(lstm->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  (void)GenManualMixAICAIVCtxDef(lstm);
  ThreadSliceMapPtr threadSliceMap = std::make_shared<ThreadSliceMap>();
  lstm->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  SetOpDecSize(lstm);

  auto sub_graph = builder.GetGraph();
  return sub_graph;
}

/*
 *
 *            \
 *         insert identity
 */
ComputeGraphPtr BuildGraph_Mix_ScopeWrite() {
  auto builder = ut::ComputeGraphBuilder("test");
  auto sub_node = builder.AddNode("sub_node", "sub_node", 1, 0);
  auto root_graph = builder.GetGraph();
  string subgraph_name = "then_branch";
  ComputeGraphPtr then_branch_graph = BuildGraph_Mix_Subgraph(subgraph_name);
  then_branch_graph->SetParentNode(sub_node);
  then_branch_graph->SetParentGraph(root_graph);
  sub_node->GetOpDesc()->AddSubgraphName(subgraph_name);
  sub_node->GetOpDesc()->SetSubgraphInstanceName(0, subgraph_name);
  root_graph->AddSubgraph(subgraph_name, then_branch_graph);
  for (auto &node : then_branch_graph->GetDirectNode()) {
    (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kThreadId, 0);
  }
  for (auto &node : root_graph->GetDirectNode()) {
    (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kThreadId, 0);
  }
  return root_graph;
}


static Status GenAutoAicAivCtxDef(NodePtr node) {
  std::shared_ptr<domi::FftsPlusCtxDef> ffts_plus_def_ptr = make_shared<domi::FftsPlusCtxDef>();
  domi::FftsPlusAicAivCtxDef *aic_aiv_ctx_def = ffts_plus_def_ptr->mutable_aic_aiv_ctx();
  FFTS_CHECK_NOTNULL(aic_aiv_ctx_def);

  aic_aiv_ctx_def->set_prefetch_once_bitmap(0);
  aic_aiv_ctx_def->set_prefetch_enable_bitmap(0);
  aic_aiv_ctx_def->set_aten(ffts::kAutoMode);
  aic_aiv_ctx_def->set_atm(ffts::kAutoMode);
  aic_aiv_ctx_def->set_thread_dim(4);

  int32_t block_dim = 1;
  aic_aiv_ctx_def->set_tail_block_dim(static_cast<uint32_t>(block_dim));
  aic_aiv_ctx_def->set_non_tail_block_dim(static_cast<uint32_t>(block_dim));

  //input
  aic_aiv_ctx_def->add_task_addr(111);
  aic_aiv_ctx_def->add_task_addr(222);

  //output
  aic_aiv_ctx_def->add_task_addr(333);

  //workspace
  aic_aiv_ctx_def->add_task_addr(444);

  auto op_desc = node->GetOpDesc();
  string attr_key_kernel_name = kKernelName;
  string attr_kernel_name;
  (void) ge::AttrUtils::GetStr(op_desc, attr_key_kernel_name, attr_kernel_name);
  aic_aiv_ctx_def->add_kernel_name(attr_kernel_name);
  (void)ge::AttrUtils::SetInt(op_desc, ffts::kAttrAICoreCtxType,
                              static_cast<int64_t>(ffts::TaskBuilderType::EN_TASK_TYPE_AIC_AIV_AUTO));
  (void)op_desc->SetExtAttr(ffts::kAttrAICoreCtxDef, ffts_plus_def_ptr);
  return ffts::SUCCESS;
}

/*
 *     data1     data2
 *       |        |
 *      add - - - -
 *       |
 *     relu
 *       |
 *    netoutput
 */
ComputeGraphPtr BuilGraph_Sgt_Subgraph_1(const string &subgraph_name, const uint32_t &thread_mode,
                                        const uint32_t &window_size, uint32_t slice_num = 4) {
  auto builder = ut::ComputeGraphBuilder(subgraph_name);
  auto data1 = builder.AddNode("data1", DATA, 0, 1);
  auto data2 = builder.AddNode("data2", DATA, 0, 1);
  auto add = builder.AddNode("add", "Add", 2, 1, FORMAT_NCHW, DT_FLOAT, {4, 4, 4, 4});
  auto relu = builder.AddNode("relu", RELU, 1, 1, FORMAT_NCHW, DT_FLOAT, {4, 4, 4, 4});
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 1);

  for (auto anchor : add->GetAllInDataAnchors()) {
    ge::AnchorUtils::SetStatus(anchor, ge::ANCHOR_DATA);
  }
  for (auto anchor : relu->GetAllInDataAnchors()) {
    ge::AnchorUtils::SetStatus(anchor, ge::ANCHOR_DATA);
  }
  GenAutoAicAivCtxDef(add);
  GenAutoAicAivCtxDef(relu);

  AttrUtils::SetInt(data1->GetOpDesc(),ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(data2->GetOpDesc(),ATTR_NAME_PARENT_NODE_INDEX, 1);

  ThreadSliceMap thread_slice_map;
  thread_slice_map.thread_mode = thread_mode;
  thread_slice_map.parallel_window_size = window_size;
  thread_slice_map.slice_instance_num = slice_num;
  DimRange dim_rang;
  dim_rang.higher = 1;
  dim_rang.lower = 0;


  vector<vector<DimRange>> input_tensor_slice_vv;
  vector<DimRange> input_tensor_slice_v;
  input_tensor_slice_v.push_back(dim_rang);
  dim_rang.higher = 0;
  dim_rang.lower = 0;
  input_tensor_slice_v.push_back(dim_rang);
  input_tensor_slice_v.push_back(dim_rang);
  input_tensor_slice_v.push_back(dim_rang);
  input_tensor_slice_vv.push_back(input_tensor_slice_v);

  vector<vector<vector<DimRange>>> output_tensor_slice_vvv = {input_tensor_slice_vv, input_tensor_slice_vv,
                                                              input_tensor_slice_vv, input_tensor_slice_vv};
  input_tensor_slice_v.clear();
  dim_rang.higher = 1;
  dim_rang.lower = 1;
  input_tensor_slice_v.push_back(dim_rang);
  dim_rang.higher = 3;
  dim_rang.lower = 0;
  input_tensor_slice_v.push_back(dim_rang);
  input_tensor_slice_v.push_back(dim_rang);
  input_tensor_slice_v.push_back(dim_rang);
  input_tensor_slice_vv.push_back(input_tensor_slice_v);

  vector<vector<vector<DimRange>>> input_tensor_slice_vvv = {input_tensor_slice_vv, input_tensor_slice_vv,
                                                             input_tensor_slice_vv, input_tensor_slice_vv};
  thread_slice_map.input_tensor_slice = input_tensor_slice_vvv;
  thread_slice_map.output_tensor_slice = output_tensor_slice_vvv;
  ThreadSliceMapPtr thread_slice_map_ptr = std::make_shared<ThreadSliceMap>(thread_slice_map);
  thread_slice_map.input_tensor_slice = output_tensor_slice_vvv;
  ThreadSliceMapPtr thread_slice_map_ptr_relu = std::make_shared<ThreadSliceMap>(thread_slice_map);
  add->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, thread_slice_map_ptr);
  relu->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, thread_slice_map_ptr_relu);
  ge::AttrUtils::SetInt(add->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  ge::AttrUtils::SetInt(relu->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);

  int64_t ge_impl_type = static_cast<int>(domi::ImplyType::BUILDIN);
  ge::AttrUtils::SetInt(add->GetOpDesc(), ge::ATTR_NAME_IMPLY_TYPE, ge_impl_type);
  ge::AttrUtils::SetInt(relu->GetOpDesc(), ge::ATTR_NAME_IMPLY_TYPE, ge_impl_type);
  if (thread_mode == 1) {
    vector<string> thread_core_type = {"AIC", "AIC"};
    (void)ge::AttrUtils::SetListStr(add->GetOpDesc(), "_thread_cube_vector_core_type", thread_core_type);
    (void)ge::AttrUtils::SetListStr(relu->GetOpDesc(), "_thread_cube_vector_core_type", thread_core_type);
  } else {
    ge::AttrUtils::SetStr(add->GetOpDesc(), "_cube_vector_core_type", "AIC");
    ge::AttrUtils::SetStr(relu->GetOpDesc(), "_cube_vector_core_type", "AIC");
  }

  auto op_desc = netoutput->GetOpDesc();
  for (auto &tensor : op_desc->GetAllInputsDescPtr()) {
    ge::AttrUtils::SetInt(tensor, ATTR_NAME_PARENT_NODE_INDEX, 0);
  }

  builder.AddDataEdge(data1, 0, add, 0);
  builder.AddDataEdge(data2, 0, add, 1);
  builder.AddDataEdge(add, 0, relu, 0);
  builder.AddDataEdge(relu, 0, netoutput, 0);
  auto sub_graph = builder.GetGraph();
  return sub_graph;
}

/*
 *         data1     data2
 *           |        |
 *          add <- - -
 *           |  \
 *         relu  \
 *       /        \
 * netoutput2   netoutput1
 */
ComputeGraphPtr BuilGraph_Sgt_Subgraph_2(const string &subgraph_name, const bool &thread_mode,
                                       const uint32_t &window_size) {
  auto builder = ut::ComputeGraphBuilder(subgraph_name);
  auto data1 = builder.AddNode("data1", DATA, 0, 1);
  auto data2 = builder.AddNode("data2", DATA, 0, 1);
  auto add = builder.AddNode("add", "Add", 2, 1);
  auto relu = builder.AddNode("relu", RELU, 1, 1);
  auto netoutput1 = builder.AddNode("netoutput1", NETOUTPUT, 1, 1);
  auto netoutput2 = builder.AddNode("netoutput2", NETOUTPUT, 1, 1);

  AttrUtils::SetInt(data1->GetOpDesc(),ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(data2->GetOpDesc(),ATTR_NAME_PARENT_NODE_INDEX, 1);

  ThreadSliceMap thread_slice_map;
  thread_slice_map.thread_mode = thread_mode;
  thread_slice_map.parallel_window_size = window_size;
  ThreadSliceMapPtr thread_slice_map_ptr = std::make_shared<ThreadSliceMap>(thread_slice_map);
  add->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, thread_slice_map_ptr);
  relu->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, thread_slice_map_ptr);

  auto op_desc1 = netoutput1->GetOpDesc();
  for (auto &tensor1 : op_desc1->GetAllInputsDescPtr()) {
    ge::AttrUtils::SetInt(tensor1, ATTR_NAME_PARENT_NODE_INDEX, 0);
  }
  auto op_desc2 = netoutput2->GetOpDesc();
  for (auto &tensor2 : op_desc2->GetAllInputsDescPtr()) {
    ge::AttrUtils::SetInt(tensor2, ATTR_NAME_PARENT_NODE_INDEX, 0);
  }

  builder.AddDataEdge(data1, 0, add, 0);
  builder.AddDataEdge(data2, 0, add, 1);
  builder.AddDataEdge(add, 0, relu, 0);
  builder.AddDataEdge(add, 0, netoutput1, 0);
  builder.AddDataEdge(relu, 0, netoutput2, 0);
  auto sub_graph = builder.GetGraph();
  return sub_graph;
}

//graph with atomic node
ComputeGraphPtr BuilGraph_Sgt_Subgraph_3(const string &subgraph_name, const bool &thread_mode,
                                         const uint32_t &window_size) {
  auto builder = ut::ComputeGraphBuilder(subgraph_name);
  auto data1 = builder.AddNode("data1", DATA, 0, 1);
  auto data2 = builder.AddNode("data2", DATA, 0, 1);
  auto data3 = builder.AddNode("data3", DATA, 0, 1);
  auto data4 = builder.AddNode("data4", DATA, 0, 1);
  auto add1 = builder.AddNode("add1", "Add", 2, 1);
  auto add2 = builder.AddNode("add2", "Add", 2, 1);
  auto add3 = builder.AddNode("add3", "Add", 2, 1);
  auto netoutput1 = builder.AddNode("netoutput1", NETOUTPUT, 1, 1);

  ComputeGraphPtr temp_graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr atomic_op = std::make_shared<OpDesc>("MemSet1", "MemSet");
  auto atomic_node = temp_graph->AddNode(atomic_op);
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  ge::AttrUtils::SetStr(atomic_node->GetOpDesc(), "compile_info_json", json_str);
  (void)add1->GetOpDesc()->SetExtAttr("memset_node_ptr", atomic_node);
  (void)add2->GetOpDesc()->SetExtAttr("memset_node_ptr", atomic_node);
  AttrUtils::SetInt(data1->GetOpDesc(),ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(data2->GetOpDesc(),ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(data3->GetOpDesc(),ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(data4->GetOpDesc(),ATTR_NAME_PARENT_NODE_INDEX, 1);

  ThreadSliceMap thread_slice_map;
  thread_slice_map.thread_mode = thread_mode;
  thread_slice_map.parallel_window_size = window_size;
  ThreadSliceMapPtr thread_slice_map_ptr = std::make_shared<ThreadSliceMap>(thread_slice_map);
  thread_slice_map_ptr->same_atomic_clean_nodes = {"add1", "add2"};
  add1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, thread_slice_map_ptr);
  add2->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, thread_slice_map_ptr);

  auto op_desc1 = netoutput1->GetOpDesc();
  for (auto &tensor1 : op_desc1->GetAllInputsDescPtr()) {
    ge::AttrUtils::SetInt(tensor1, ATTR_NAME_PARENT_NODE_INDEX, 0);
  }
  GenManAicAivCtxDef(add1);
  GenManAicAivCtxDef(add2);
  GenManAicAivCtxDef(add3);
  builder.AddDataEdge(data1, 0, add1, 0);
  builder.AddDataEdge(data2, 0, add1, 1);
  builder.AddDataEdge(data3, 0, add2, 0);
  builder.AddDataEdge(data4, 0, add2, 1);
  builder.AddDataEdge(add1, 0, add3, 0);
  builder.AddDataEdge(add2, 0, add3, 1);
  builder.AddDataEdge(add3, 0, netoutput1, 0);
  auto sub_graph = builder.GetGraph();
  return sub_graph;
}

/*
 *         data1     data2
 *           |        |
 *          add <- - -
 *           |  \
 *         relu  prelu
 *       /        \
 *  netoutput1   netoutput2
 */
ComputeGraphPtr BuilGraph_Sgt_Subgraph_4(const string &subgraph_name, const bool &thread_mode,
                                         const uint32_t &window_size, const bool &with_sub_stream_id) {
  auto builder = ut::ComputeGraphBuilder(subgraph_name);
  auto data1 = builder.AddNode("data1", DATA, 0, 1);
  auto data2 = builder.AddNode("data2", DATA, 0, 1);
  auto add1 = builder.AddNode("add1", "Add", 2, 1);
  auto relu1 = builder.AddNode("relu1", "Relu", 2, 1);
  auto prelu1 = builder.AddNode("prelu1", "PRelu", 2, 1);
  auto netoutput1 = builder.AddNode("netoutput1", NETOUTPUT, 1, 1);
  auto netoutput2 = builder.AddNode("netoutput2", NETOUTPUT, 1, 1);

  AttrUtils::SetInt(data1->GetOpDesc(),ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(data2->GetOpDesc(),ATTR_NAME_PARENT_NODE_INDEX, 1);

  ThreadSliceMap thread_slice_map;
  thread_slice_map.thread_mode = thread_mode;
  thread_slice_map.parallel_window_size = window_size;
  ThreadSliceMapPtr thread_slice_map_ptr = std::make_shared<ThreadSliceMap>(thread_slice_map);
  add1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, thread_slice_map_ptr);
  relu1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, thread_slice_map_ptr);
  prelu1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, thread_slice_map_ptr);
  (void)ge::AttrUtils::SetBool(add1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetBool(relu1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetBool(prelu1->GetOpDesc(), kTypeFFTSPlus, true);
  if (with_sub_stream_id) {
    ge::AttrUtils::SetInt(add1->GetOpDesc(), ge::ATTR_NAME_SUB_STREAM_ID, 1);
    ge::AttrUtils::SetInt(relu1->GetOpDesc(), ge::ATTR_NAME_SUB_STREAM_ID, 1);
    ge::AttrUtils::SetInt(prelu1->GetOpDesc(), ge::ATTR_NAME_SUB_STREAM_ID, 1);
  }

  auto op_desc1 = netoutput1->GetOpDesc();
  for (auto &tensor1 : op_desc1->GetAllInputsDescPtr()) {
    ge::AttrUtils::SetInt(tensor1, ATTR_NAME_PARENT_NODE_INDEX, 0);
  }

  auto op_desc2 = netoutput2->GetOpDesc();
  for (auto &tensor2 : op_desc2->GetAllInputsDescPtr()) {
    ge::AttrUtils::SetInt(tensor2, ATTR_NAME_PARENT_NODE_INDEX, 0);
  }
  GenManAicAivCtxDef(add1);
  GenManAicAivCtxDef(relu1);
  GenManAicAivCtxDef(prelu1);
  builder.AddDataEdge(data1, 0, add1, 0);
  builder.AddDataEdge(data2, 0, add1, 1);
  builder.AddDataEdge(add1, 0, relu1, 0);
  builder.AddDataEdge(add1, 0, prelu1, 1);
  builder.AddDataEdge(relu1, 0, netoutput1, 0);
  builder.AddDataEdge(prelu1, 0, netoutput2, 1);
  auto sub_graph = builder.GetGraph();
  return sub_graph;
}

/*
 *     data1     data2
 *       |        |
 *    ge_local    |
 *       |        |
 *      add - - - -
 *       |
 *     relu
 *       |
 *    ge_local
 *       |
 *    netoutput
 */
ComputeGraphPtr BuilGraph_Sgt_Subgraph_5(const string &subgraph_name, const uint32_t &thread_mode,
                                         const uint32_t &window_size, uint32_t slice_num = 4) {
  auto builder = ut::ComputeGraphBuilder(subgraph_name);
  auto data1 = builder.AddNode("data1", DATA, 0, 1);
  auto data2 = builder.AddNode("data2", DATA, 0, 1);
  auto reshape_1 = builder.AddNode("reshape_1", "reshape", 1, 1, FORMAT_NCHW, DT_FLOAT, {4, 4, 4, 4});
  auto add = builder.AddNode("add", "Add", 2, 1, FORMAT_NCHW, DT_FLOAT, {4, 4, 4, 4});
  auto relu = builder.AddNode("relu", RELU, 1, 1, FORMAT_NCHW, DT_FLOAT, {4, 4, 4, 4});
  auto reshape_2 = builder.AddNode("reshape_2", "reshape", 1, 1, FORMAT_NCHW, DT_FLOAT, {4, 4, 4, 4});
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 1);

  for (auto anchor : add->GetAllInDataAnchors()) {
    ge::AnchorUtils::SetStatus(anchor, ge::ANCHOR_DATA);
  }
  for (auto anchor : relu->GetAllInDataAnchors()) {
    ge::AnchorUtils::SetStatus(anchor, ge::ANCHOR_DATA);
  }
  GenAutoAicAivCtxDef(add);
  GenAutoAicAivCtxDef(relu);

  AttrUtils::SetInt(data1->GetOpDesc(),ATTR_NAME_PARENT_NODE_INDEX, 1);
  AttrUtils::SetInt(data2->GetOpDesc(),ATTR_NAME_PARENT_NODE_INDEX, 1);

  ThreadSliceMap thread_slice_map;
  thread_slice_map.thread_mode = thread_mode;
  thread_slice_map.parallel_window_size = window_size;
  thread_slice_map.slice_instance_num = slice_num;
  DimRange dim_rang;
  dim_rang.higher = 1;
  dim_rang.lower = 0;


  vector<vector<DimRange>> input_tensor_slice_vv;
  vector<DimRange> input_tensor_slice_v;
  input_tensor_slice_v.push_back(dim_rang);
  dim_rang.higher = 0;
  dim_rang.lower = 0;
  input_tensor_slice_v.push_back(dim_rang);
  input_tensor_slice_v.push_back(dim_rang);
  input_tensor_slice_v.push_back(dim_rang);
  input_tensor_slice_vv.push_back(input_tensor_slice_v);

  vector<vector<vector<DimRange>>> output_tensor_slice_vvv = {input_tensor_slice_vv, input_tensor_slice_vv,
                                                              input_tensor_slice_vv, input_tensor_slice_vv};
  input_tensor_slice_v.clear();
  dim_rang.higher = 1;
  dim_rang.lower = 1;
  input_tensor_slice_v.push_back(dim_rang);
  dim_rang.higher = 3;
  dim_rang.lower = 0;
  input_tensor_slice_v.push_back(dim_rang);
  input_tensor_slice_v.push_back(dim_rang);
  input_tensor_slice_v.push_back(dim_rang);
  input_tensor_slice_vv.push_back(input_tensor_slice_v);

  vector<vector<vector<DimRange>>> input_tensor_slice_vvv = {input_tensor_slice_vv, input_tensor_slice_vv,
                                                             input_tensor_slice_vv, input_tensor_slice_vv};
  thread_slice_map.input_tensor_slice = input_tensor_slice_vvv;
  thread_slice_map.output_tensor_slice = output_tensor_slice_vvv;
  ThreadSliceMapPtr thread_slice_map_ptr = std::make_shared<ThreadSliceMap>(thread_slice_map);
  thread_slice_map.input_tensor_slice = output_tensor_slice_vvv;
  ThreadSliceMapPtr thread_slice_map_ptr_relu = std::make_shared<ThreadSliceMap>(thread_slice_map);
  add->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, thread_slice_map_ptr);
  relu->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, thread_slice_map_ptr_relu);
  ge::AttrUtils::SetInt(add->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  ge::AttrUtils::SetInt(relu->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);

  ge::AttrUtils::SetBool(reshape_1->GetOpDesc(), ge::ATTR_NAME_NOTASK, true);
  ge::AttrUtils::SetBool(reshape_2->GetOpDesc(), ge::ATTR_NAME_NOTASK, true);

  int64_t ge_impl_type = static_cast<int>(domi::ImplyType::BUILDIN);
  ge::AttrUtils::SetInt(add->GetOpDesc(), ge::ATTR_NAME_IMPLY_TYPE, ge_impl_type);
  ge::AttrUtils::SetInt(relu->GetOpDesc(), ge::ATTR_NAME_IMPLY_TYPE, ge_impl_type);
  if (thread_mode == 1) {
    vector<string> thread_core_type = {"AIC", "AIC"};
    (void)ge::AttrUtils::SetListStr(add->GetOpDesc(), "_thread_cube_vector_core_type", thread_core_type);
    (void)ge::AttrUtils::SetListStr(relu->GetOpDesc(), "_thread_cube_vector_core_type", thread_core_type);
  } else {
    ge::AttrUtils::SetStr(add->GetOpDesc(), "_cube_vector_core_type", "AIC");
    ge::AttrUtils::SetStr(relu->GetOpDesc(), "_cube_vector_core_type", "AIC");
  }

  auto op_desc = netoutput->GetOpDesc();
  for (auto &tensor : op_desc->GetAllInputsDescPtr()) {
    ge::AttrUtils::SetInt(tensor, ATTR_NAME_PARENT_NODE_INDEX, 0);
  }

  builder.AddDataEdge(data1, 0, reshape_1, 0);
  builder.AddDataEdge(reshape_1, 0, add, 0);
  builder.AddDataEdge(data2, 0, add, 1);
  builder.AddDataEdge(add, 0, relu, 0);
  builder.AddDataEdge(relu, 0, reshape_2, 0);
  builder.AddDataEdge(reshape_2, 0, netoutput, 0);
  auto sub_graph = builder.GetGraph();
  return sub_graph;
}

static Status GenManualAICAIVCtxDef(NodePtr node) {
  std::shared_ptr<domi::FftsPlusCtxDef> ffts_plus_def_ptr = make_shared<domi::FftsPlusCtxDef>();
  domi::FftsPlusAicAivCtxDef *aic_aiv_ctx_def = ffts_plus_def_ptr->mutable_aic_aiv_ctx();
  FFTS_CHECK_NOTNULL(aic_aiv_ctx_def);

  // cache managemet will do at GenerateDataTaskDef()
  aic_aiv_ctx_def->set_prefetch_once_bitmap(0);
  aic_aiv_ctx_def->set_prefetch_enable_bitmap(0);
  aic_aiv_ctx_def->set_atm(ffts::kManualMode);
  aic_aiv_ctx_def->set_thread_dim(1);

  int32_t block_dim = 1;
  //(void) ge::AttrUtils::GetInt(op_desc, ge::TVM_ATTR_NAME_BLOCKDIM, block_dim);
  aic_aiv_ctx_def->set_tail_block_dim(static_cast<uint32_t>(block_dim));
  aic_aiv_ctx_def->set_non_tail_block_dim(static_cast<uint32_t>(block_dim));

  //input
  aic_aiv_ctx_def->add_task_addr(111);
  aic_aiv_ctx_def->add_task_addr(222);

  //output
  aic_aiv_ctx_def->add_task_addr(333);

  //workspace
  aic_aiv_ctx_def->add_task_addr(444);

  auto op_desc = node->GetOpDesc();
  string attr_key_kernel_name = kKernelName;
  string attr_kernel_name;
  (void) ge::AttrUtils::GetStr(op_desc, attr_key_kernel_name, attr_kernel_name);
  aic_aiv_ctx_def->add_kernel_name(attr_kernel_name);
  (void)ge::AttrUtils::SetInt(op_desc, ffts::kAttrAICoreCtxType,
                              static_cast<int64_t>(ffts::TaskBuilderType::EN_TASK_TYPE_AIC_AIV));
  (void)op_desc->SetExtAttr(ffts::kAttrAICoreCtxDef, ffts_plus_def_ptr);
  return ffts::SUCCESS;
}

ComputeGraphPtr BuildGraph_SubGraph_Greater26(const string &subraph_name) {
  auto builder = ut::ComputeGraphBuilder(subraph_name);

  ThreadSliceMap thread_slice_map;
  thread_slice_map.thread_mode = 0;
  ThreadSliceMapPtr thread_slice_map_ptr = std::make_shared<ThreadSliceMap>(thread_slice_map);

  auto input = builder.AddNode("input", "input", 0, 30, ge::FORMAT_NCHW, ge::DT_FLOAT, {2, 4, 4, 4});
  (void)ge::AttrUtils::SetInt(input->GetOpDesc(), ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
  (void)ge::AttrUtils::SetStr(input->GetOpDesc(), "_cube_vector_core_type", "AIC");
  (void)ge::AttrUtils::SetInt(input->GetOpDesc(), kAttrAICoreCtxType,
                              static_cast<int64_t>(ffts::TaskBuilderType::EN_TASK_TYPE_AIC_AIV));
  (void)ge::AttrUtils::SetStr(input->GetOpDesc(), ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AICUBE");
  (void)ge::AttrUtils::SetBool(input->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(input->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
  input->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, thread_slice_map_ptr);
  GenManualAICAIVCtxDef(input);
  SetOpDecSize(input);

  auto output = builder.AddNode("output", "output", 30, 1, ge::FORMAT_NCHW, ge::DT_FLOAT, {2, 4, 4, 4});
  (void)ge::AttrUtils::SetInt(output->GetOpDesc(), ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
  (void)ge::AttrUtils::SetStr(output->GetOpDesc(), "_cube_vector_core_type", "AIC");
  (void)ge::AttrUtils::SetInt(output->GetOpDesc(), kAttrAICoreCtxType,
                              static_cast<int64_t>(ffts::TaskBuilderType::EN_TASK_TYPE_AIC_AIV));
  (void)ge::AttrUtils::SetStr(output->GetOpDesc(), ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AICUBE");
  (void)ge::AttrUtils::SetBool(output->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(output->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
  output->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, thread_slice_map_ptr);
  GenManualAICAIVCtxDef(output);
  SetOpDecSize(output);
  for (auto i = 0; i < 30; i++) {
    string node_name = "conv2d";
    node_name += to_string(i);
    auto conv2d = builder.AddNode(node_name, "conv2d", 1, 1, ge::FORMAT_NCHW, ge::DT_FLOAT, {2, 4, 4, 4});
    (void)ge::AttrUtils::SetInt(conv2d->GetOpDesc(), ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
    (void)ge::AttrUtils::SetStr(conv2d->GetOpDesc(), "_cube_vector_core_type", "AIC");
    (void)ge::AttrUtils::SetInt(conv2d->GetOpDesc(), kAttrAICoreCtxType,
                                static_cast<int64_t>(ffts::TaskBuilderType::EN_TASK_TYPE_AIC_AIV));
    (void)ge::AttrUtils::SetStr(conv2d->GetOpDesc(), ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AICUBE");
    (void)ge::AttrUtils::SetBool(conv2d->GetOpDesc(), kTypeFFTSPlus, true);
    (void)ge::AttrUtils::SetInt(conv2d->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    conv2d->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, thread_slice_map_ptr);
    SetOpDecSize(conv2d);
    GenManualAICAIVCtxDef(conv2d);
    builder.AddDataEdge(input, i, conv2d, 0);
    builder.AddDataEdge(conv2d, 0, output, i);
  }

  auto sub_graph = builder.GetGraph();
  return sub_graph;
}

ComputeGraphPtr BuildGraph_SubGraph_Greater60(const string &subraph_name) {
  auto builder = ut::ComputeGraphBuilder(subraph_name);

  ThreadSliceMap thread_slice_map;
  thread_slice_map.thread_mode = 0;
  ThreadSliceMapPtr thread_slice_map_ptr = std::make_shared<ThreadSliceMap>(thread_slice_map);

  auto input = builder.AddNode("input", "input", 0, 60, ge::FORMAT_NCHW, ge::DT_FLOAT, {2, 4, 4, 4});
  (void)ge::AttrUtils::SetInt(input->GetOpDesc(), ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
  (void)ge::AttrUtils::SetStr(input->GetOpDesc(), "_cube_vector_core_type", "AIV");
  (void)ge::AttrUtils::SetInt(input->GetOpDesc(), kAttrAICoreCtxType,
                              static_cast<int64_t>(ffts::TaskBuilderType::EN_TASK_TYPE_AIC_AIV));
  (void)ge::AttrUtils::SetStr(input->GetOpDesc(), ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
  (void)ge::AttrUtils::SetBool(input->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(input->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
  input->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, thread_slice_map_ptr);
  GenManualAICAIVCtxDef(input);
  SetOpDecSize(input);

  auto output = builder.AddNode("output", kTypePhonyConcat, 60, 1, ge::FORMAT_NCHW, ge::DT_FLOAT, {2, 4, 4, 4});
  (void)ge::AttrUtils::SetInt(output->GetOpDesc(), ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
  (void)ge::AttrUtils::SetStr(output->GetOpDesc(), "_cube_vector_core_type", "AIV");
  (void)ge::AttrUtils::SetInt(output->GetOpDesc(), kAttrAICoreCtxType,
                              static_cast<int64_t>(ffts::TaskBuilderType::EN_TASK_TYPE_AIC_AIV));
  (void)ge::AttrUtils::SetStr(output->GetOpDesc(), ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
  (void)ge::AttrUtils::SetBool(output->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(output->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
  output->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, thread_slice_map_ptr);
  GenManualAICAIVCtxDef(output);
  SetOpDecSize(output);

  auto output1 = builder.AddNode("output1", "output1", 1, 1, ge::FORMAT_NCHW, ge::DT_FLOAT, {2, 4, 4, 4});
  (void)ge::AttrUtils::SetInt(output1->GetOpDesc(), ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
  (void)ge::AttrUtils::SetStr(output1->GetOpDesc(), "_cube_vector_core_type", "AIV");
  (void)ge::AttrUtils::SetInt(output1->GetOpDesc(), kAttrAICoreCtxType,
                              static_cast<int64_t>(ffts::TaskBuilderType::EN_TASK_TYPE_AIC_AIV));
  (void)ge::AttrUtils::SetStr(output1->GetOpDesc(), ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
  (void)ge::AttrUtils::SetBool(output1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(output1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 2);
  output1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, thread_slice_map_ptr);
  GenManualAICAIVCtxDef(output1);
  SetOpDecSize(output1);

  for (auto i = 0; i < 60; i++) {
    string node_name = "conv2d";
    node_name += to_string(i);
    auto conv2d = builder.AddNode(node_name, "conv2d", 1, 1, ge::FORMAT_NCHW, ge::DT_FLOAT, {2, 4, 4, 4});
    (void)ge::AttrUtils::SetInt(conv2d->GetOpDesc(), ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
    (void)ge::AttrUtils::SetStr(conv2d->GetOpDesc(), "_cube_vector_core_type", "AIV");
    (void)ge::AttrUtils::SetInt(conv2d->GetOpDesc(), kAttrAICoreCtxType,
                                static_cast<int64_t>(ffts::TaskBuilderType::EN_TASK_TYPE_AIC_AIV));
    (void)ge::AttrUtils::SetStr(conv2d->GetOpDesc(), ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
    (void)ge::AttrUtils::SetBool(conv2d->GetOpDesc(), kTypeFFTSPlus, true);
    (void)ge::AttrUtils::SetInt(conv2d->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    conv2d->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, thread_slice_map_ptr);
    SetOpDecSize(conv2d);
    GenManualAICAIVCtxDef(conv2d);
    builder.AddDataEdge(input, i, conv2d, 0);
    builder.AddDataEdge(conv2d, 0, output, i);
  }
  builder.AddDataEdge(output, 0, output1, 0);
  auto sub_graph = builder.GetGraph();
  return sub_graph;
}

ComputeGraphPtr BuildGraph_Greater26() {
  auto builder = ut::ComputeGraphBuilder("test");
  auto sub_node = builder.AddNode("sub_node", "sub_node", 1, 0);
  auto root_graph = builder.GetGraph();
  string subgraph_name = "then_branch";
  ComputeGraphPtr then_branch_graph = BuildGraph_SubGraph_Greater26(subgraph_name);
  then_branch_graph->SetParentNode(sub_node);
  then_branch_graph->SetParentGraph(root_graph);
  sub_node->GetOpDesc()->AddSubgraphName(subgraph_name);
  sub_node->GetOpDesc()->SetSubgraphInstanceName(0, subgraph_name);
  root_graph->AddSubgraph(subgraph_name, then_branch_graph);
  for (auto &node : then_branch_graph->GetDirectNode()) {
    (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kThreadId, 0);
  }
  for (auto &node : root_graph->GetDirectNode()) {
    (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kThreadId, 0);
  }
  return root_graph;
}

ComputeGraphPtr BuildGraph_Greater60() {
  auto builder = ut::ComputeGraphBuilder("test");
  auto sub_node = builder.AddNode("sub_node", "sub_node", 1, 0);
  auto root_graph = builder.GetGraph();
  string subgraph_name = "then_branch";
  ComputeGraphPtr then_branch_graph = BuildGraph_SubGraph_Greater60(subgraph_name);
  then_branch_graph->SetParentNode(sub_node);
  then_branch_graph->SetParentGraph(root_graph);
  sub_node->GetOpDesc()->AddSubgraphName(subgraph_name);
  sub_node->GetOpDesc()->SetSubgraphInstanceName(0, subgraph_name);
  root_graph->AddSubgraph(subgraph_name, then_branch_graph);
  return root_graph;
}

ComputeGraphPtr BuildGraph_AICPU_Subgraph(const string &subraph_name) {
  auto builder = ut::ComputeGraphBuilder(subraph_name);
  auto lstm = builder.AddNode("add", "Add", 2, 1);
  FftsPlusCtxDefPtr ctxDefPtr = std::make_shared<domi::FftsPlusCtxDef>();
  lstm->GetOpDesc()->SetExtAttr("_ffts_plus_aicpu_ctx_def", ctxDefPtr);
  ThreadSliceMapPtr threadSliceMap = std::make_shared<ThreadSliceMap>();
  lstm->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(lstm->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);

  auto sub_graph = builder.GetGraph();
  return sub_graph;
}

ComputeGraphPtr BuildGraph_HCCL_Subgraph(const string &subraph_name) {
  auto builder = ut::ComputeGraphBuilder(subraph_name);
  auto hccl_mode = builder.AddNode("HcomAllReduce", "HcomAllReduce", 1, 1);
  auto up_mode = builder.AddNode("upmode", "reduce", 1, 1, ge::FORMAT_NCHW, ge::DT_FLOAT, {2, 4, 4, 4});
  builder.AddDataEdge(up_mode, 0, hccl_mode, 0);
  std::vector<domi::FftsPlusCtxDef> hccl_sub_tasks;
  domi::FftsPlusCtxDef hccl_sub_task;
  hccl_sub_task.set_context_type(RT_CTX_TYPE_SDMA);
  hccl_sub_tasks.push_back(hccl_sub_task);
  hccl_sub_task.set_context_type(RT_CTX_TYPE_NOTIFY_WAIT);
  hccl_sub_tasks.push_back(hccl_sub_task);
  hccl_sub_task.set_context_type(RT_CTX_TYPE_WRITE_VALUE);
  hccl_sub_tasks.push_back(hccl_sub_task);

  hccl_mode->GetOpDesc()->SetExtAttr(kHcclSubTasks, hccl_sub_tasks);
  ThreadSliceMapPtr threadSliceMap = std::make_shared<ThreadSliceMap>();
  hccl_mode->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  up_mode->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(hccl_mode->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 2);
  ge::AttrUtils::SetInt(up_mode->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 2);
  ge::AttrUtils::SetInt(up_mode->GetOpDesc(), ge::ATTR_NAME_IMPLY_TYPE, static_cast<int>(domi::ImplyType::TVM));
  AttrUtils::SetStr(up_mode->GetOpDesc(), "_cube_vector_core_type", "AIV");
  (void)ge::AttrUtils::SetInt(up_mode->GetOpDesc(), ffts::kAttrAICoreCtxType,
                              static_cast<int64_t>(ffts::TaskBuilderType::EN_TASK_TYPE_AIC_AIV));
  (void)GenManualAICAIVCtxDef(up_mode);
  (void)ge::AttrUtils::SetStr(up_mode->GetOpDesc(), ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
  SetOpDecSize(up_mode);
  std::vector<std::vector<int64_t>> adjacency_list = {{1, 2}, {3}, {}};
  (void)ge::AttrUtils::SetListListInt(hccl_mode->GetOpDesc(), kAdjacencyList, adjacency_list);

  auto sub_graph = builder.GetGraph();
  return sub_graph;
}

ComputeGraphPtr BuildGraph_AICPU_ScopeWrite() {
  auto builder = ut::ComputeGraphBuilder("test");
  auto sub_node = builder.AddNode("sub_node", "sub_node", 1, 0);
  auto root_graph = builder.GetGraph();
  string subgraph_name = "then_branch";
  ComputeGraphPtr then_branch_graph = BuildGraph_AICPU_Subgraph(subgraph_name);
  then_branch_graph->SetParentNode(sub_node);
  then_branch_graph->SetParentGraph(root_graph);
  sub_node->GetOpDesc()->AddSubgraphName(subgraph_name);
  sub_node->GetOpDesc()->SetSubgraphInstanceName(0, subgraph_name);
  root_graph->AddSubgraph(subgraph_name, then_branch_graph);
  return root_graph;
}

ComputeGraphPtr BuildGraph_HCCL_Graph() {
  auto builder = ut::ComputeGraphBuilder("test");
  auto sub_node = builder.AddNode("sub_node", "sub_node", 1, 0);
  auto root_graph = builder.GetGraph();
  string subgraph_name = "then_branch";
  ComputeGraphPtr then_branch_graph = BuildGraph_HCCL_Subgraph(subgraph_name);
  then_branch_graph->SetParentNode(sub_node);
  then_branch_graph->SetParentGraph(root_graph);
  sub_node->GetOpDesc()->AddSubgraphName(subgraph_name);
  sub_node->GetOpDesc()->SetSubgraphInstanceName(0, subgraph_name);
  root_graph->AddSubgraph(subgraph_name, then_branch_graph);
  return root_graph;
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, Initialize_SUCCESS) {
  map<string, string> options;
  string pre_soc_version = EngineManager::Instance(kFFTSPlusCoreName).GetSocVersion();
  EngineManager::Instance(kFFTSPlusCoreName).soc_version_ = "ascend910b2";

  std::string path = GetCodeDir() + "/tests/engines/ffts_engine/config/data/platform_config";
  std::string real_path = RealPath(path);
  fe::PlatformInfoManager::Instance().platform_info_map_.clear();
  fe::PlatformInfoManager::Instance().platform_infos_map_.clear();
  uint32_t init_ret = fe::PlatformInfoManager::Instance().LoadConfigFile(real_path);

  std::map<std::string, std::vector<std::string>> a;
  std::map<std::string, std::vector<std::string>> b;
  a = b;
  fe::PlatFormInfos platform_infos;
  fe::OptionalInfos opti_compilation_infos;
  std::map<std::string, fe::PlatFormInfos> platform_infos_map;
  platform_infos.Init();
  std::map<std::string, std::string> res;
  res[kFFTSMode] = kFFTSPlus;
  platform_infos.SetPlatformRes(kSocInfo, res);
  platform_infos_map["ascend910b2"] = platform_infos;
  fe::PlatformInfoManager::Instance().platform_infos_map_ = platform_infos_map;

  ffts_plus_ops_kernel_builder_ptr->init_flag_ = true;
  Status ret = ffts_plus_ops_kernel_builder_ptr->Initialize(options);
  EngineManager::Instance(kFFTSPlusCoreName).soc_version_ = pre_soc_version;
  fe::PlatformInfoManager::Instance().platform_infos_map_.clear();
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, Initialize_FAILED) {
  map<string, string> options;
  string pre_soc_version = EngineManager::Instance(kFFTSPlusCoreName).GetSocVersion();
  EngineManager::Instance(kFFTSPlusCoreName).soc_version_ = "ascend910b2";

  std::string path = GetCodeDir() + "/tests/engines/ffts_engine/config/data/platform_config";
  std::string real_path = RealPath(path);
  fe::PlatformInfoManager::Instance().platform_info_map_.clear();
  fe::PlatformInfoManager::Instance().platform_infos_map_.clear();
  uint32_t init_ret = fe::PlatformInfoManager::Instance().LoadConfigFile(real_path);

  std::map<std::string, std::vector<std::string>> a;
  std::map<std::string, std::vector<std::string>> b;
  a = b;
  fe::PlatFormInfos platform_infos;
  fe::OptionalInfos opti_compilation_infos;
  std::map<std::string, fe::PlatFormInfos> platform_infos_map;
  platform_infos.Init();
  std::map<std::string, std::string> res;
  res[kFFTSMode] = kFFTSPlus;
  platform_infos.SetPlatformRes(kSocInfo, res);
  platform_infos_map["ascend910b2"] = platform_infos;
  fe::PlatformInfoManager::Instance().platform_infos_map_ = platform_infos_map;

  // have no libascend_sch_policy_pass.so, initialize fail
  Status ret = ffts_plus_ops_kernel_builder_ptr->Initialize(options);
  EngineManager::Instance(kFFTSPlusCoreName).soc_version_ = pre_soc_version;
  fe::PlatformInfoManager::Instance().platform_infos_map_.clear();
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, Finalize_SUCCESS) {
  ffts_plus_ops_kernel_builder_ptr->sch_policy_pass_plugin_ = std::make_shared<PluginManager>("plugin_path");
  ffts_plus_ops_kernel_builder_ptr->init_flag_ = true;
  Status ret = ffts_plus_ops_kernel_builder_ptr->Finalize();
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, GenerateTask_SUCCESS)
{
  ComputeGraphPtr graph = BuildGraph_Readonly_ScopeWrite();
  auto ifnode = graph->FindNode("if");

  vector<domi::TaskDef> tasks;
  ffts_plus_ops_kernel_builder_ptr->schecule_policy_pass_ = ScheculePolicyPassStub2;
  Status ret = ffts_plus_ops_kernel_builder_ptr->GenerateTask(*ifnode, context_, tasks);

  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, GenerateTask_Greater26_SUCCESS)
{
  ComputeGraphPtr graph = BuildGraph_Greater26();
  auto sub_node = graph->FindNode("sub_node");
  for (auto &node : graph->GetDirectNode()) {
    (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kThreadId, 0);
  }
  vector<domi::TaskDef> tasks;
  ffts_plus_ops_kernel_builder_ptr->schecule_policy_pass_ = ScheculePolicyPassStub2;
  Status ret = ffts_plus_ops_kernel_builder_ptr->GenerateTask(*sub_node, context_, tasks);
  EXPECT_EQ(ffts::SUCCESS, ret);
  domi::FftsPlusTaskDef *ffts_plus_task_def = tasks[0].mutable_ffts_plus_task();
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 33);

  domi::FftsPlusCtxDef* label_ctx_def =
          ffts_plus_task_def->mutable_ffts_plus_ctx(static_cast<int>(32));
  domi::FftsPlusLabelCtxDef* label_ctx = label_ctx_def->mutable_label_ctx();
  EXPECT_EQ(label_ctx_def->context_type(), RT_CTX_TYPE_LABEL);
  EXPECT_EQ(label_ctx->pred_cnt(), 1);
  EXPECT_EQ(label_ctx->successor_num(), 5);
  EXPECT_EQ(label_ctx->successor_list(0), 5);

  domi::FftsPlusCtxDef* second_ctx_first =
          ffts_plus_task_def->mutable_ffts_plus_ctx(static_cast<int>(RT_CTX_SUCCESSOR_NUM - 1));
  EXPECT_NE(second_ctx_first->context_type(), RT_CTX_TYPE_LABEL);
}

Status ScheculePolicyPassStub3(domi::TaskDef &task_def, std::vector<ffts::FftsPlusContextPath> &ctx_path_vector) {
  // construct a sub-graph and set precnt
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->mutable_ffts_plus_ctx(0);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_MIX_AIC);
  ffts_plus_ctx_def = ffts_plus_task_def->mutable_ffts_plus_ctx(1);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AICPU);
  auto aicpu = ffts_plus_ctx_def->mutable_aicpu_ctx();
  aicpu->set_pred_cnt(2);
  aicpu->set_pred_cnt_init(2);
  ffts_plus_ctx_def = ffts_plus_task_def->mutable_ffts_plus_ctx(2);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_SDMA);
  auto sdma = ffts_plus_ctx_def->mutable_sdma_ctx();
  sdma->set_pred_cnt(2);
  sdma->set_pred_cnt_init(2);
  ffts_plus_ctx_def = ffts_plus_task_def->mutable_ffts_plus_ctx(3);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AICORE);
  auto aicaiv = ffts_plus_ctx_def->mutable_aic_aiv_ctx();
  aicaiv->set_pred_cnt(3);
  aicaiv->set_pred_cnt_init(3);
  ffts_plus_ctx_def = ffts_plus_task_def->mutable_ffts_plus_ctx(27);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_WRITE_VALUE);
  ffts_plus_ctx_def = ffts_plus_task_def->mutable_ffts_plus_ctx(28);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_CASE_SWITCH);
  auto case_switch = ffts_plus_ctx_def->mutable_case_switch_ctx();
  case_switch->set_pred_cnt(2);
  case_switch->set_pred_cnt_init(2);
  ffts_plus_ctx_def = ffts_plus_task_def->mutable_ffts_plus_ctx(29);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_INVALIDATE_DATA);
  auto nofity = ffts_plus_ctx_def->mutable_data_ctx();
  nofity->set_cnt(2);
  nofity->set_cnt_init(2);
  ffts_plus_ctx_def = ffts_plus_task_def->mutable_ffts_plus_ctx(30);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_DSA);
  auto dsa = ffts_plus_ctx_def->mutable_dsa_ctx();
  dsa->set_pred_cnt(2);
  dsa->set_pred_cnt_init(2);

  // 0--> 1,1
  // 1--> 2,2,3,4,5,62(label)
  // 2--> 3,3
  // 3--> 4,5,6,...,34,35,63(label)
  // 27=>28=>29=>30
  // construct ctx_path_vector ctx_id pre_cnt policy_pri max_pre_index cmo_list label_list pre_list succ_list
  FftsPlusContextPath ctx_path0 = {0, 0, 55, 0, {50}, {}, {}, {1,1}};
  FftsPlusContextPath ctx_path1 = {1, 2, 54, 1, {}, {62}, {0,0}, {2,2,3,4,5}};
  FftsPlusContextPath ctx_path2 = {2, 2, 53, 2, {}, {}, {1,1}, {3,3}};
  FftsPlusContextPath ctx_path3 = {3, 3, 52, 3, {}, {63}, {1,2,2}, {4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,
                                                                    23,24,25,26,27,28,29,30,31,32,33,34,35}};
  FftsPlusContextPath ctx_path4 = {4, 2, 51, 4, {}, {}, {1,3}, {}};
  FftsPlusContextPath ctx_path5 = {5, 2, 51, 4, {}, {}, {1,3}, {}};
  ctx_path_vector.emplace_back(ctx_path0);
  ctx_path_vector.emplace_back(ctx_path1);
  ctx_path_vector.emplace_back(ctx_path2);
  ctx_path_vector.emplace_back(ctx_path3);
  ctx_path_vector.emplace_back(ctx_path4);
  ctx_path_vector.emplace_back(ctx_path5);
  for (uint32_t i = 6; i < 27; i++) {
    FftsPlusContextPath ctx_pathi = {i, 1, 51, 4, {}, {}, {3}, {}};
    ctx_path_vector.emplace_back(ctx_pathi);
  }
  FftsPlusContextPath ctx_path27 = {27, 1, 51, 4, {}, {}, {3}, {28,28}};
  FftsPlusContextPath ctx_path28 = {28, 3, 50, 5, {29}, {}, {27,27,3}, {29,29}};
  FftsPlusContextPath ctx_path29 = {29, 3, 49, 6, {40}, {}, {28,28,3}, {30,30}};
  FftsPlusContextPath ctx_path30 = {30, 3, 48, 7, {}, {}, {3,29,29}, {}};
  ctx_path_vector.emplace_back(ctx_path27);
  ctx_path_vector.emplace_back(ctx_path28);
  ctx_path_vector.emplace_back(ctx_path29);
  ctx_path_vector.emplace_back(ctx_path30);

  ffts_plus_ctx_def = ffts_plus_task_def->mutable_ffts_plus_ctx(40);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_INVALIDATE_DATA);
  auto data = ffts_plus_ctx_def->mutable_data_ctx();

  for (uint32_t i = 31; i < 36; i++) {
    FftsPlusContextPath ctx_pathi = {i, 1, 50, 4, {}, {}, {3}, {}};
    ctx_path_vector.emplace_back(ctx_pathi);
  }
  return ffts::SUCCESS;
}

Status CheckVectorSame(std::vector<uint32_t> &vector1, std::vector<uint32_t> &vector2) {
  if (vector1.size() != vector2.size()) {
    return ffts::FAILED;
  }

  sort(vector1.begin(), vector1.end());
  sort(vector2.begin(), vector2.end());

  for (size_t i = 0; i < vector1.size(); ++i) {
    if (vector1[i] != vector2[i]) {
      return ffts::FAILED;
    }
  }
  return ffts::SUCCESS;
}

Status CheckScheculePolicyPass(domi::FftsPlusTaskDef *ffts_plus_task_def) {
  // origin graph:
  // 0--> 1,1
  // 1--> 2,2,3,4,5,62(label)
  // 2--> 3,3
  // 3--> 4,5,6,...,34,35,62(label)
  // 27=>28=>29=>30
  // result should be:
  // 0--> 1
  // 1--> 2
  // 2--> 3
  // 3--> 4,5,6,...27,31,63(label:32,33,34,35)
  // 27-->28-->29-->30
  // check ctx 0 succlist 1, 50
  domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->mutable_ffts_plus_ctx(0);
  auto mixaicaiv = ffts_plus_ctx_def->mutable_mix_aic_aiv_ctx();
  if (mixaicaiv->successor_num() != 2) {
    cout << "check ctx 0 successor_num failed" << endl;
    return ffts::FAILED;
  }
  std::vector<uint32_t> succlist;
  for (auto succid : mixaicaiv->successor_list()) {
    succlist.emplace_back(succid);
  }
  std::vector<uint32_t> target_succlist0 = {1, 50};
  if (CheckVectorSame(succlist, target_succlist0) != ffts::SUCCESS) {
    cout << "check ctx 0 successlist failed" << endl;
    return ffts::FAILED;
  }

  // check ctx 1 succlist 2
  succlist.clear();
  ffts_plus_ctx_def = ffts_plus_task_def->mutable_ffts_plus_ctx(1);
  auto aicpu = ffts_plus_ctx_def->mutable_aicpu_ctx();
  if (aicpu->successor_num() != 1 || aicpu->pred_cnt() != 1) {
    cout << "check ctx 1 successor_num or precnt failed" << endl;
    return ffts::FAILED;
  }
  for (auto succid : aicpu->successor_list()) {
    succlist.emplace_back(succid);
  }
  std::vector<uint32_t> target_succlist1 = {2};
  if (CheckVectorSame(succlist, target_succlist1) != ffts::SUCCESS) {
    cout << "check ctx 1 successlist failed" << endl;
    return ffts::FAILED;
  }
  // check ctx 3 succlist  4,5,6,...27,31,62(label:32,33,34,35)
  succlist.clear();
  ffts_plus_ctx_def = ffts_plus_task_def->mutable_ffts_plus_ctx(3);
  auto aicaiv = ffts_plus_ctx_def->mutable_aic_aiv_ctx();
  if (aicaiv->successor_num() != RT_CTX_SUCCESSOR_NUM || aicaiv->pred_cnt() != 1) {
    cout << "check ctx 3 successor_num or precnt failed" << endl;
    return ffts::FAILED;
  }
  for (auto succid : aicaiv->successor_list()) {
    succlist.emplace_back(succid);
  }
  std::vector<uint32_t> target_succlist3 = {4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,31,62};
  if (CheckVectorSame(succlist, target_succlist3) != ffts::SUCCESS) {
    cout << "check ctx 3 successlist failed" << endl;
    return ffts::FAILED;
  }
  // label 63:32,33,34,35
  succlist.clear();
  ffts_plus_ctx_def = ffts_plus_task_def->mutable_ffts_plus_ctx(62);
  auto label = ffts_plus_ctx_def->mutable_label_ctx();
  if (label->successor_num() != 4) {
    cout << "check label 62 successor_num failed" << endl;
    return ffts::FAILED;
  }
  for (auto succid : label->successor_list()) {
    succlist.emplace_back(succid);
  }
  std::vector<uint32_t> target_succlist4 = {32,33,34,35};
  if (CheckVectorSame(succlist, target_succlist4) != ffts::SUCCESS) {
    cout << "check label 62 successlist failed" << endl;
    return ffts::FAILED;
  }
  return ffts::SUCCESS;
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, GenerateTask_Greater60_Schecule_SUCCESS)
{
  ComputeGraphPtr graph = BuildGraph_Greater60();
  auto sub_node = graph->FindNode("sub_node");
  vector<domi::TaskDef> tasks;
  ffts_plus_ops_kernel_builder_ptr->schecule_policy_pass_ = ScheculePolicyPassStub3;
  Status ret = ffts_plus_ops_kernel_builder_ptr->GenerateTask(*sub_node, context_, tasks);
  EXPECT_EQ(ffts::SUCCESS, ret);
  if (tasks.empty()) {
    EXPECT_EQ(ffts::SUCCESS, ffts::FAILED);
  }
  domi::FftsPlusTaskDef *ffts_plus_task_def = tasks[0].mutable_ffts_plus_task();
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 63);
  // check if ctx is ok
  ASSERT_EQ(ffts::SUCCESS, CheckScheculePolicyPass(ffts_plus_task_def));
}

Status ScheculePolicyPassStub4(domi::TaskDef &task_def, std::vector<ffts::FftsPlusContextPath> &ctx_path_vector) {
  // construct a sub-graph and set precnt
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->mutable_ffts_plus_ctx(0);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_MIX_AIC);
  ffts_plus_ctx_def = ffts_plus_task_def->mutable_ffts_plus_ctx(1);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AICPU);
  auto aicpu = ffts_plus_ctx_def->mutable_aicpu_ctx();
  ffts_plus_ctx_def = ffts_plus_task_def->mutable_ffts_plus_ctx(2);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_SDMA);
  auto sdma = ffts_plus_ctx_def->mutable_sdma_ctx();
  // construct ctx_path_vector ctx_id pre_cnt policy_pri max_pre_index cmo_list label_list pre_list succ_list
  FftsPlusContextPath ctx_path0 = {0, 0, 55, 0, {50}, {}, {}, {1,1}};
  FftsPlusContextPath ctx_path1 = {1, 2, 54, 1, {}, {62}, {}, {2,2,3,4,5}};
  FftsPlusContextPath ctx_path2 = {2, 2, 53, 2, {}, {}, {1,1}, {3,3}};
 
  ctx_path_vector.emplace_back(ctx_path0);
  ctx_path_vector.emplace_back(ctx_path1);
  ctx_path_vector.emplace_back(ctx_path2);

  return ffts::SUCCESS;
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, GenerateTask_Greater60_Schecule_READYNUM_FAIL)
{
  ComputeGraphPtr graph = BuildGraph_Greater60();
  auto sub_node = graph->FindNode("sub_node");
  vector<domi::TaskDef> tasks;
  ffts_plus_ops_kernel_builder_ptr->schecule_policy_pass_ = ScheculePolicyPassStub4;
  Status ret = ffts_plus_ops_kernel_builder_ptr->GenerateTask(*sub_node, context_, tasks);
  EXPECT_EQ(ffts::FAILED, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, GenerateTask_Greater60_SUCCESS)
{
  ComputeGraphPtr graph = BuildGraph_Greater60();
  auto sub_node = graph->FindNode("sub_node");
  vector<domi::TaskDef> tasks;
  ffts_plus_ops_kernel_builder_ptr->schecule_policy_pass_ = ScheculePolicyPassStub2;
  Status ret = ffts_plus_ops_kernel_builder_ptr->GenerateTask(*sub_node, context_, tasks);
  EXPECT_EQ(ffts::SUCCESS, ret);
  domi::FftsPlusTaskDef *ffts_plus_task_def = tasks[0].mutable_ffts_plus_task();
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 64);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, Mix_GenerateTask_SUCCESS)
{
  ComputeGraphPtr graph = BuildGraph_Mix_ScopeWrite();
  cout << "========================MIX AIC/AIV GENTASK BEGIN========================" << endl;
  auto sub_node = graph->FindNode("sub_node");
  if (sub_node == nullptr) {
    cout << "[ERROR] FE:sub node is nullptr";
  }
  vector<domi::TaskDef> tasks;
  ffts_plus_ops_kernel_builder_ptr->schecule_policy_pass_ = ScheculePolicyPassStub2;
  Status ret = ffts_plus_ops_kernel_builder_ptr->GenerateTask(*sub_node, context_, tasks);

  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, Case_Gen_AutoThread_Context_Id)
{
  std::vector<ge::NodePtr> sub_graph_nodes;
  uint32_t thread_mode = true;
  uint32_t window_size = 4;
  ComputeGraphPtr sgt_graph = BuilGraph_Sgt_Subgraph_1("sgt_graph", thread_mode, window_size);
  uint64_t ready_context_num = 0;
  uint64_t total_context_number = 0;
  std::vector<ge::NodePtr> atomic_nodes;
  Status ret = ffts_plus_auto_mode_ptr->GenFftsPlusContextId(*sgt_graph, sub_graph_nodes, ready_context_num,
                                                                      total_context_number, atomic_nodes);
  // check result
  int total_count = 0;
  int flag_attr_count = 0;
  for (const auto &node : sub_graph_nodes) {
    total_count++;
    if (node->GetName() == "add") {
      uint32_t in_label_ctx_id = 1;
      vector<uint32_t> at_start_ctx_id_list;
      vector<uint32_t> context_id_list;
      if (ge::AttrUtils::GetInt(node->GetOpDesc(), kAutoInlabelCtxId, in_label_ctx_id) &&
          ge::AttrUtils::GetListInt(node->GetOpDesc(), kAutoAtStartCtxIdList, at_start_ctx_id_list) &&
          ge::AttrUtils::GetListInt(node->GetOpDesc(), kAutoCtxIdList, context_id_list)) {
        flag_attr_count++;
      }
    } else if (node->GetName() == "relu") {
      uint32_t at_end_pre_cnt = 0;
      uint32_t out_label_ctx_id = 0;
      vector<uint32_t> at_end_ctx_id_list;
      vector<uint32_t> context_id_list;
      if (ge::AttrUtils::GetInt(node->GetOpDesc(), kAutoOutlabelCtxId, out_label_ctx_id) &&
          ge::AttrUtils::GetListInt(node->GetOpDesc(), kAutoAtEndCtxIdList, at_end_ctx_id_list) &&
          ge::AttrUtils::GetInt(node->GetOpDesc(), kAutoAtEndPreCnt, at_end_pre_cnt) &&
          ge::AttrUtils::GetListInt(node->GetOpDesc(), kAutoCtxIdList, context_id_list)) {
        flag_attr_count++;
      }
    }
  }
  EXPECT_EQ(2, total_count);
  EXPECT_EQ(2, flag_attr_count);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, Case_Gen_AutoThread_AtEnd_PreCnt)
{
  std::vector<ge::NodePtr> sub_graph_nodes;
  bool thread_mode = 1;
  uint32_t window_size = 4;
  ComputeGraphPtr sgt_graph = BuilGraph_Sgt_Subgraph_2("sgt_graph", thread_mode, window_size);
  uint64_t ready_context_num = 0;
  uint64_t total_context_number = 0;
  std::vector<ge::NodePtr> atomic_nodes;
  Status ret = ffts_plus_auto_mode_ptr->GenFftsPlusContextId(*sgt_graph, sub_graph_nodes, ready_context_num,
                                                                      total_context_number, atomic_nodes);
  // check result
  bool status_pre_cnt = false;
  for (const auto &node : sub_graph_nodes) {
    if (node->GetName() == "relu") {
      uint32_t at_end_pre_cnt = 0;
      (void)ge::AttrUtils::GetInt(node->GetOpDesc(), kAutoAtEndPreCnt, at_end_pre_cnt);
      if (at_end_pre_cnt == 2) {
        status_pre_cnt = true;
      }
    }
  }
  EXPECT_EQ(true, status_pre_cnt);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, Case_Gen_Atomic_Context_Id)
{
  std::vector<ge::NodePtr> sub_graph_nodes;
  uint32_t thread_mode = true;
  uint32_t window_size = 4;
  ComputeGraphPtr sgt_graph = BuilGraph_Sgt_Subgraph_1("sgt_graph", thread_mode, window_size);
  uint64_t ready_context_num = 0;
  uint64_t total_context_number = 0;
  std::vector<ge::NodePtr> atomic_nodes;
  ffts_plus_auto_mode_ptr->mode_type_ = ModeType::AUTO_MODE_TYPE;
  auto relu = sgt_graph->FindNode("relu");
  EXPECT_NE(relu, nullptr);

  ge::ComputeGraphPtr tmp_graph = std::make_shared<ge::ComputeGraph>("OpCompileGraph");
  ge::OpDescPtr memset_op_desc_ptr =  make_shared<ge::OpDesc>("memset_node", fe::MEMSET_OP_TYPE);
  ge::NodePtr memset_node = tmp_graph->AddNode(memset_op_desc_ptr, relu->GetOpDesc()->GetId());
  relu->GetOpDesc()->SetExtAttr(fe::ATTR_NAME_MEMSET_NODE, memset_node);
  ffts_plus_auto_mode_ptr->Initialize();
  Status ret = ffts_plus_auto_mode_ptr->GenFftsPlusContextId(*sgt_graph, sub_graph_nodes, ready_context_num,
                                                             total_context_number, atomic_nodes);
  vector<uint32_t> context_id_list;
  ge::AttrUtils::GetListInt(memset_node->GetOpDesc(), kAutoCtxIdList, context_id_list);
  EXPECT_EQ(4, context_id_list.size());

  domi::TaskDef task_def;
  ret = ffts_plus_auto_mode_ptr->GenSubGraphTaskDef(atomic_nodes, sub_graph_nodes, task_def);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  uint64_t gen_ctx_num = ffts_plus_task_def->ffts_plus_ctx_size();
  EXPECT_EQ(22, gen_ctx_num);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, Case_Gen_AtStart_Atomic_Context_Id)
{
  std::vector<ge::NodePtr> sub_graph_nodes;
  uint32_t thread_mode = true;
  uint32_t window_size = 4;
  ComputeGraphPtr sgt_graph = BuilGraph_Sgt_Subgraph_1("sgt_graph", thread_mode, window_size);
  uint64_t ready_context_num = 0;
  uint64_t total_context_number = 0;
  std::vector<ge::NodePtr> atomic_nodes;
  ffts_plus_auto_mode_ptr->mode_type_ = ModeType::AUTO_MODE_TYPE;
  auto add = sgt_graph->FindNode("add");
  EXPECT_NE(add, nullptr);

  ge::ComputeGraphPtr tmp_graph = std::make_shared<ge::ComputeGraph>("OpCompileGraph");
  ge::OpDescPtr memset_op_desc_ptr =  make_shared<ge::OpDesc>("memset_node", fe::MEMSET_OP_TYPE);
  ge::NodePtr memset_node = tmp_graph->AddNode(memset_op_desc_ptr, add->GetOpDesc()->GetId());
  add->GetOpDesc()->SetExtAttr(fe::ATTR_NAME_MEMSET_NODE, memset_node);
  ffts_plus_auto_mode_ptr->Initialize();
  Status ret = ffts_plus_auto_mode_ptr->GenFftsPlusContextId(*sgt_graph, sub_graph_nodes, ready_context_num,
                                                             total_context_number, atomic_nodes);
  vector<uint32_t> context_id_list;
  ge::AttrUtils::GetListInt(memset_node->GetOpDesc(), kAutoCtxIdList, context_id_list);
  EXPECT_EQ(4, context_id_list.size());

  domi::TaskDef task_def;
  ret = ffts_plus_auto_mode_ptr->GenSubGraphTaskDef(atomic_nodes, sub_graph_nodes, task_def);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  uint64_t gen_ctx_num = ffts_plus_task_def->ffts_plus_ctx_size();
  EXPECT_EQ(22, gen_ctx_num);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, Case_Gen_Sub_Graph_Task_Def)
{
  std::vector<ge::NodePtr> sub_graph_nodes;
  uint32_t thread_mode = 1;
  uint32_t window_size = 4;
  ComputeGraphPtr sgt_graph = BuilGraph_Sgt_Subgraph_1("sgt_graph", thread_mode, window_size);
  uint64_t ready_context_num = 0;
  uint64_t total_context_number = 0;
  ffts_plus_auto_mode_ptr->Initialize();
  std::vector<ge::NodePtr> atomic_nodes;
  Status ret = ffts_plus_auto_mode_ptr->GenFftsPlusContextId(*sgt_graph, sub_graph_nodes, ready_context_num,
                                                             total_context_number, atomic_nodes);
  domi::TaskDef task_def;
  ret = ffts_plus_auto_mode_ptr->GenSubGraphTaskDef(atomic_nodes, sub_graph_nodes, task_def);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  uint64_t gen_ctx_num = ffts_plus_task_def->ffts_plus_ctx_size();

  EXPECT_EQ(18, gen_ctx_num);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

// test include ge local op
TEST_F(FFTSPlusOpsKernelBuilderUTest, Case_Gen_Sub_Graph_Task_Def_1)
{
  std::vector<ge::NodePtr> sub_graph_nodes;
  uint32_t thread_mode = 1;
  uint32_t window_size = 4;
  ComputeGraphPtr sgt_graph = BuilGraph_Sgt_Subgraph_5("sgt_graph", thread_mode, window_size);
  uint64_t ready_context_num = 0;
  uint64_t total_context_number = 0;
  ffts_plus_auto_mode_ptr->Initialize();
  std::vector<ge::NodePtr> atomic_nodes;
  Status ret = ffts_plus_auto_mode_ptr->GenFftsPlusContextId(*sgt_graph, sub_graph_nodes, ready_context_num,
                                                             total_context_number, atomic_nodes);
  domi::TaskDef task_def;
  ret = ffts_plus_auto_mode_ptr->GenSubGraphTaskDef(atomic_nodes, sub_graph_nodes, task_def);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  uint64_t gen_ctx_num = ffts_plus_task_def->ffts_plus_ctx_size();
  EXPECT_EQ(18, gen_ctx_num);
  EXPECT_EQ(ffts::SUCCESS, ret);
}
/*
 *         data1
 *           |
 *       labelswitch
 *           |
 *         labelset
 *           |
 *        identity
 *           |
 *        output
 */
ComputeGraphPtr BuildGraph_RuntimeOp_Subgraph(const string &subraph_name, bool is_auto) {
  auto builder = ut::ComputeGraphBuilder(subraph_name);
  auto data = builder.AddNode("data", DATA, 0, 1);
  auto labelswitch = builder.AddNode("labelswitch", "LabelSwitchByIndex", 1, 1);
  auto labelset = builder.AddNode("labelset", "LabelSet", 1, 1);
  auto identity = builder.AddNode("identity", "Identity", 1, 1);
  auto output = builder.AddNode("output", NETOUTPUT, 1, 1);
  FftsPlusCtxDefPtr ctxDefPtr = std::make_shared<domi::FftsPlusCtxDef>();
  labelswitch->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  labelset->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  labelswitch->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  identity->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  ThreadSliceMapPtr threadSliceMap = std::make_shared<ThreadSliceMap>();
  if (is_auto) {
    threadSliceMap->parallel_window_size = 4;
    threadSliceMap->thread_mode = kAutoMode;
    (void)ge::AttrUtils::SetInt(labelswitch->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto anchor = output->GetOpDesc()->MutableInputDesc(0);
    ge::AttrUtils::SetInt(anchor, ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  }
  labelswitch->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labelswitch->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  labelset->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labelset->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  identity->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(identity->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);

  builder.AddDataEdge(data, 0, labelswitch, 0);
  builder.AddDataEdge(labelswitch, 0, labelset, 0);
  builder.AddDataEdge(labelset, 0, identity, 0);
  builder.AddDataEdge(identity, 0, output, 0);

  for (auto anchor : labelswitch->GetAllInDataAnchors()) {
    ge::AnchorUtils::SetStatus(anchor, ge::ANCHOR_DATA);
  }
  for (auto anchor : labelset->GetAllInDataAnchors()) {
    ge::AnchorUtils::SetStatus(anchor, ge::ANCHOR_DATA);
  }
  for (auto anchor : identity->GetAllInDataAnchors()) {
    ge::AnchorUtils::SetStatus(anchor, ge::ANCHOR_DATA);
  }
  auto sub_graph = builder.GetGraph();
  builder.SetConstantInputOffset();
  return sub_graph;
}

ComputeGraphPtr BuildGraph_RuntimeOp_ScopeWrite(ComputeGraphPtr &func_op_branch_graph, bool is_auto) {
  auto builder = ut::ComputeGraphBuilder("rts_root_graph");
  auto sub_node = builder.AddNode("sub_node", "sub_node", 1, 0);
  auto root_graph = builder.GetGraph();
  string subgraph_name = "func_op_branch";
  func_op_branch_graph = BuildGraph_RuntimeOp_Subgraph(subgraph_name, is_auto);
  func_op_branch_graph->SetParentNode(sub_node);
  func_op_branch_graph->SetParentGraph(root_graph);
  sub_node->GetOpDesc()->AddSubgraphName(subgraph_name);
  sub_node->GetOpDesc()->SetSubgraphInstanceName(0, subgraph_name);
  root_graph->AddSubgraph(subgraph_name, func_op_branch_graph);
  (void)AttrUtils::SetStr(func_op_branch_graph, "_parent_graph_name", "rts_root_graph");
  (void)root_graph->SetExtAttr("part_src_graph", func_op_branch_graph);
  return root_graph;
}

//if

ComputeGraphPtr BuildGraph_RuntimeOp_TestCase1_If_0_else_sub_graph(const string &subraph_name) {
  auto builder = ut::ComputeGraphBuilder(subraph_name);
  auto labeset_1 = builder.AddNode("labeset_1_1", "LabelSet", 0, 0);
  ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), "_label_switch_index", 1);
  auto data1 = builder.AddNode("data_1_1", "Data", 0, 1);
  auto streamactive1 = builder.AddNode("streamactive_1_1", "StreamActive", 0, 0);
  auto square1 = builder.AddNode("square_1_1", "Square", 1, 1);
  auto netoutput0 = builder.AddNode("netoutput_0_1", "NetOutput", 1, 1);
  auto labelset_2 = builder.AddNode("labeset_1_2", "LabelSet", 0, 0);

  (void)ge::AttrUtils::SetBool(labeset_1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(data1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(data1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(streamactive1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(streamactive1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(square1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(square1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(netoutput0->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(netoutput0->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(labelset_2->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(labelset_2->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);


  ge::AttrUtils::SetInt(labelset_2->GetOpDesc(), "_label_switch_index", 2);
  FftsPlusCtxDefPtr ctxDefPtr = std::make_shared<domi::FftsPlusCtxDef>();
  labeset_1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  labelset_2->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);

  ThreadSliceMapPtr threadSliceMap = std::make_shared<ThreadSliceMap>();
  labeset_1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  labelset_2->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labelset_2->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  square1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(square1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  GenManAicAivCtxDef(square1);
  builder.AddControlEdge(labeset_1, streamactive1);
  builder.AddControlEdge(streamactive1, square1);
  builder.AddDataEdge(data1, 0, square1, 0);
  builder.AddDataEdge(square1, 0, netoutput0, 0);
  builder.AddControlEdge(netoutput0, labelset_2);
  builder.SetConstantInputOffset();
  auto sub_graph = builder.GetGraph();

  return sub_graph;
}


ComputeGraphPtr BuildGraph_RuntimeOp_TestCase1_If_0_thensub_graph(const string &subraph_name) {
  auto builder = ut::ComputeGraphBuilder(subraph_name);

  auto labelswitchbyindex1 = builder.AddNode("labelswitch_2_1", "LabelSwitchByIndex", 1, 0);
  ge::GeTensorDescPtr tensor_desc = labelswitchbyindex1->GetOpDesc()->MutableInputDesc(0);
  ge::GeTensorPtr tensor_node = nullptr;
  FE_MAKE_SHARED(tensor_node = std::make_shared<ge::GeTensor>(), return nullptr);
  tensor_node->SetTensorDesc(*tensor_desc);
  ge::GeShape shape{};
  uint32_t index = 0;
  tensor_desc->SetDataType(ge::DT_UINT32);
  tensor_desc->SetOriginDataType(ge::DT_UINT32);
  tensor_desc->SetShape(shape);
  tensor_desc->SetOriginShape(shape);
  tensor_desc->SetFormat(ge::FORMAT_ND);
  tensor_desc->SetOriginFormat(ge::FORMAT_ND);
  (void)tensor_node->SetData(reinterpret_cast<uint8_t *>(&index), 4);
  ge::OpDescPtr newdatadesc = nullptr;
  FE_MAKE_SHARED(newdatadesc = std::make_shared<ge::OpDesc>("data_2_1","Constant"), return nullptr);
  (void)ge::AttrUtils::SetBool(newdatadesc, "_is_single_op", true);
  (void)ge::AttrUtils::SetBool(newdatadesc, ge::ATTR_NAME_IS_ORIGINAL_INPUT, true);
  if (newdatadesc->AddOutputDesc(*tensor_desc) != ge::GRAPH_SUCCESS) {
    FE_LOGD("AddOutputDesc Failed");
    return nullptr;
  }
  if (!ge::AttrUtils::SetTensor(newdatadesc, ge::ATTR_NAME_WEIGHTS, tensor_node)) {
    FE_LOGD("SetTensor = null");
    return nullptr;
  }
  auto sub_graph = builder.GetGraph();
  auto data1 = builder.AddNode("data_2_1", "Data", 0, 1);
  (void)ge::AttrUtils::SetInt(data1->GetOpDesc(), "_parent_node_index", 0);

  std::vector<uint32_t> index_list;
  index_list.push_back(0);
  index_list.push_back(1);
  ge::AttrUtils::SetListInt(labelswitchbyindex1->GetOpDesc(), "_label_switch_list", index_list);
  auto labeset_1 = builder.AddNode("labelset_2_1", "LabelSet", 0, 0);
  ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), "_label_switch_index", 0);
  auto data2 = builder.AddNode("data_2_2", "Data", 0, 1);
  auto data3 = builder.AddNode("data_2_3", "Data", 0, 1);
  auto streamactive1 = builder.AddNode("streamactive_2_1", "StreamActive", 0, 0);
  auto mul1 = builder.AddNode("mul_2_1", "Mul", 2, 1);
  auto netoutput1 = builder.AddNode("netoutput_2_1", "NetOutput", 1, 0);
  auto labegotoex1 = builder.AddNode("labelgoto_2_1", "LabelGotoEx", 0, 0);

  (void)ge::AttrUtils::SetBool(data1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(data1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(labelswitchbyindex1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(labelswitchbyindex1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(labeset_1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
  (void)ge::AttrUtils::SetBool(data2->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(data2->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(data3->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(data3->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(streamactive1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(streamactive1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(mul1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(mul1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(netoutput1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(netoutput1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(labegotoex1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(labegotoex1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);


  ge::AttrUtils::SetInt(labegotoex1->GetOpDesc(), "_label_switch_index", 2);
  FftsPlusCtxDefPtr ctxDefPtr = std::make_shared<domi::FftsPlusCtxDef>();
  labelswitchbyindex1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  labeset_1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  labegotoex1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  GenManAicAivCtxDef(mul1);
  ThreadSliceMapPtr threadSliceMap = std::make_shared<ThreadSliceMap>();
  labelswitchbyindex1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labelswitchbyindex1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  labeset_1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  mul1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(mul1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  labegotoex1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labegotoex1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);


  builder.AddDataEdge(data1, 0, labelswitchbyindex1, 0);
  auto dataconstanchor = labelswitchbyindex1->GetAllInDataAnchors().at(0);
  (void)ge::AnchorUtils::SetStatus(dataconstanchor, ge::ANCHOR_CONST);
  builder.AddControlEdge(labelswitchbyindex1, labeset_1);
  builder.AddControlEdge(labeset_1, streamactive1);
  builder.AddControlEdge(streamactive1, mul1);
  builder.AddDataEdge(data2, 0, mul1, 0);
  builder.AddDataEdge(data3, 0, mul1, 1);
  builder.AddDataEdge(mul1, 0, netoutput1, 0);
  builder.AddControlEdge(netoutput1, labegotoex1);
  builder.SetConstantInputOffset();
  return sub_graph;
}

ComputeGraphPtr BuildGraph_RuntimeOp_TestCase1_If_0_root_graph(const string &subraph_name) {
  auto builder = ut::ComputeGraphBuilder(subraph_name);
  auto data1 = builder.AddNode("data_0_1", "Data", 0, 1);
  auto data2 = builder.AddNode("data_0_2", "Data", 0, 1);
  auto data3 = builder.AddNode("data_0_3", "Data", 0, 1);
  auto square1 = builder.AddNode("square_0_1", "Square", 2, 1);
  auto if_0_node = builder.AddNode("If", "If", 4, 1);
  auto add = builder.AddNode("add_0_1", "Add", 2, 1);
  auto variable = builder.AddNode("variable_0_1", "Variable", 0, 0);
  auto netoutput0 = builder.AddNode("netoutput_0_1", "NetOutput", 1, 0);


  (void)ge::AttrUtils::SetBool(data1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(data1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(data2->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(data2->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(data3->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(data3->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(square1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(square1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(if_0_node->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(if_0_node->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(add->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(add->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(variable->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(variable->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(netoutput0->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(netoutput0->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);


  FftsPlusCtxDefPtr ctxDefPtr = std::make_shared<domi::FftsPlusCtxDef>();
  if_0_node->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  variable->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);

  ThreadSliceMapPtr threadSliceMap = std::make_shared<ThreadSliceMap>();
  if_0_node->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(if_0_node->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  square1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(square1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  add->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(add->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  variable->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(variable->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  GenManAicAivCtxDef(square1);
  builder.AddDataEdge(data1, 0, if_0_node, 2);
  builder.AddDataEdge(data1, 0, if_0_node, 1);
  builder.AddDataEdge(square1, 0, if_0_node, 0);
  builder.AddDataEdge(data2, 0, if_0_node, 3);
  builder.AddDataEdge(data3, 0, square1, 0);
  builder.AddDataEdge(data3, 0, square1, 1);
  builder.AddDataEdge(if_0_node, 0, add, 0);
  builder.AddDataEdge(data2, 0, add, 1);
  builder.AddDataEdge(add, 0, netoutput0, 0);
  builder.AddControlEdge(variable, netoutput0);
  GenManAicAivCtxDef(add);
  auto sub_if_graph = builder.GetGraph();
  builder.SetConstantInputOffset();
  return sub_if_graph;
}

ComputeGraphPtr BuildGraph_RuntimeOp_If_ScopeWrite() {
  auto builder = ut::ComputeGraphBuilder("rts_root_graph");
  auto sub_node = builder.AddNode("sub_node", "sub_node", 1, 0);
  auto root_graph = builder.GetGraph();
  string subgraph_name = "func_op_branch";
  ComputeGraphPtr func_op_branch_graph = BuildGraph_RuntimeOp_TestCase1_If_0_root_graph(subgraph_name);
  func_op_branch_graph->SetParentNode(sub_node);
  func_op_branch_graph->SetParentGraph(root_graph);
  sub_node->GetOpDesc()->AddSubgraphName(subgraph_name);
  sub_node->GetOpDesc()->SetSubgraphInstanceName(0, subgraph_name);
  root_graph->AddSubgraph(subgraph_name, func_op_branch_graph);

  auto if_0_node = func_op_branch_graph->FindNode("If");
  if (if_0_node == nullptr) {
    cout << "[ERROR] FE:sub node is nullptr";
    return root_graph;
  }
  string then_subgraph_name = "then_sub_graph";
  ComputeGraphPtr then_branch_graph = BuildGraph_RuntimeOp_TestCase1_If_0_thensub_graph(then_subgraph_name);
  then_branch_graph->SetParentNode(if_0_node);
  then_branch_graph->SetParentGraph(func_op_branch_graph);
  if_0_node->GetOpDesc()->AddSubgraphName(then_subgraph_name);
  if_0_node->GetOpDesc()->SetSubgraphInstanceName(0, then_subgraph_name);
  root_graph->AddSubgraph(then_subgraph_name, then_branch_graph);

  string else_subgraph_name = "else_sub_graph";
  ComputeGraphPtr else_branch_graph = BuildGraph_RuntimeOp_TestCase1_If_0_else_sub_graph(else_subgraph_name);
  else_branch_graph->SetParentNode(if_0_node);
  else_branch_graph->SetParentGraph(func_op_branch_graph);
  if_0_node->GetOpDesc()->AddSubgraphName(else_subgraph_name);
  if_0_node->GetOpDesc()->SetSubgraphInstanceName(1, else_subgraph_name);
  root_graph->AddSubgraph(else_subgraph_name, else_branch_graph);
  builder.SetConstantInputOffset();

  return root_graph;
}


//while
ComputeGraphPtr BuildGraph_RuntimeOp_TestCase1_while_0_condi_sub_graph(const string &subraph_name) {
  auto builder = ut::ComputeGraphBuilder(subraph_name);
  auto labeset_1 = builder.AddNode("labelset_1_1", "LabelSet", 0, 0);
  auto streamactive1 = builder.AddNode("streamactive_2_1", "StreamActive", 0, 0);
  ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), "_label_switch_index", 0);
  auto data2 = builder.AddNode("data_1_2", "Data", 0, 1);
  auto data3 = builder.AddNode("data_1_3", "Constant", 0, 1);
  auto less = builder.AddNode("less_1_1", "Less", 2, 1);
  auto cast = builder.AddNode("cast_1_1", "Cast", 2, 1);
  auto netoutput1 = builder.AddNode("netoutput_1_1", "NetOutput", 1, 0);
  auto labelswitchbyindex1 = builder.AddNode("labelswitch_1_1", "LabelSwitchByIndex", 2, 0);
  ge::GeTensorDescPtr tensor_desc = labelswitchbyindex1->GetOpDesc()->MutableInputDesc(0);
  ge::GeTensorPtr tensor_node = nullptr;
  FE_MAKE_SHARED(tensor_node = std::make_shared<ge::GeTensor>(), return nullptr);
  tensor_node->SetTensorDesc(*tensor_desc);
  ge::GeShape shape{};
  uint32_t index = 0;
  tensor_desc->SetDataType(ge::DT_UINT32);
  tensor_desc->SetOriginDataType(ge::DT_UINT32);
  tensor_desc->SetShape(shape);
  tensor_desc->SetOriginShape(shape);
  tensor_desc->SetFormat(ge::FORMAT_ND);
  tensor_desc->SetOriginFormat(ge::FORMAT_ND);
  (void)tensor_node->SetData(reinterpret_cast<uint8_t *>(&index), 4);
  ge::OpDescPtr newdatadesc = nullptr;
  FE_MAKE_SHARED(newdatadesc = std::make_shared<ge::OpDesc>("data_2_1","Constant"), return nullptr);
  (void)ge::AttrUtils::SetBool(newdatadesc, "_is_single_op", true);
  (void)ge::AttrUtils::SetBool(newdatadesc, ge::ATTR_NAME_IS_ORIGINAL_INPUT, true);
  if (newdatadesc->AddOutputDesc(*tensor_desc) != ge::GRAPH_SUCCESS) {
    FE_LOGD("AddOutputDesc Failed");
    return nullptr;
  }
  if (!ge::AttrUtils::SetTensor(newdatadesc, ge::ATTR_NAME_WEIGHTS, tensor_node)) {
    FE_LOGD("SetTensor = null");
    return nullptr;
  }
  auto data1 = builder.AddNode("data_2_1", "Data", 0, 1);
  builder.AddDataEdge(data1, 0, labelswitchbyindex1, 0);
  std::vector<uint32_t> index_list;
  index_list.push_back(2);
  index_list.push_back(1);
  ge::AttrUtils::SetListInt(labelswitchbyindex1->GetOpDesc(), "_label_switch_list", index_list);
  (void)ge::AttrUtils::SetBool(labeset_1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);


  (void)ge::AttrUtils::SetBool(streamactive1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(streamactive1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(data2->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(data2->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(data3->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(data3->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(less->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(less->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(cast->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(cast->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);


  (void)ge::AttrUtils::SetBool(netoutput1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(netoutput1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(labelswitchbyindex1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(labelswitchbyindex1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  //GenManAicAivCtxDef(less);
  GenManAicAivCtxDef(cast);

  FftsPlusCtxDefPtr ctxDefPtr = std::make_shared<domi::FftsPlusCtxDef>();
  labelswitchbyindex1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  labeset_1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  less->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  ThreadSliceMapPtr threadSliceMap = std::make_shared<ThreadSliceMap>();
  labelswitchbyindex1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labelswitchbyindex1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  labeset_1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  less->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(less->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
   cast->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(cast->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  netoutput1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(netoutput1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);

  builder.AddControlEdge(labeset_1, streamactive1);
  builder.AddControlEdge(streamactive1, less);
  builder.AddDataEdge(data2, 0, less, 0);
  builder.AddDataEdge(data3, 0, less, 1);
  builder.AddDataEdge(less, 0, cast, 0);
  builder.AddDataEdge(cast, 0, netoutput1, 0);
  builder.AddDataEdge(cast, 0, labelswitchbyindex1, 1);
  auto dataconstanchor = labelswitchbyindex1->GetAllInDataAnchors().at(0);
  (void)ge::AnchorUtils::SetStatus(dataconstanchor, ge::ANCHOR_CONST);
  builder.AddControlEdge(netoutput1, labelswitchbyindex1);
  auto sub_graph = builder.GetGraph();
  builder.SetConstantInputOffset();
  return sub_graph;
}


ComputeGraphPtr BuildGraph_RuntimeOp_TestCase1_while_0_then_sub_graph(const string &subraph_name) {
  auto builder = ut::ComputeGraphBuilder(subraph_name);

  auto labeset_1 = builder.AddNode("labelset_2_1", "LabelSet", 0, 0);
  ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), "_label_switch_index", 1);
  auto data1 = builder.AddNode("data_2_1", "Data", 0, 1);
  auto data2 = builder.AddNode("data_2_2", "Data", 0, 1);
  auto data3 = builder.AddNode("data_2_3", "Data", 0, 1);
  auto streamactive1 = builder.AddNode("streamactive_2_1", "StreamActive", 0, 0);
  auto identity1 = builder.AddNode("identity_2_1", "Identity", 3, 3);
  auto constan1 = builder.AddNode("constant_2_1", "Constant", 0, 1);
  auto add1 = builder.AddNode("add_2_1", "Add", 2, 1);
  auto add2 = builder.AddNode("add_2_2", "Add", 2, 1);
  auto identity2 = builder.AddNode("identity_2_2", "Identity", 3, 3);
  auto netoutput1 = builder.AddNode("netoutput_2_1", "NetOutput", 3, 0);
  auto labegotoex1 = builder.AddNode("labelgoto_2_1", "LabelGotoEx", 0, 0);
  ge::AttrUtils::SetInt(labegotoex1->GetOpDesc(), "_label_switch_index", 0);
  auto labelset_2 = builder.AddNode("labelset_2_2", "LabelSet", 0, 0);
  ge::AttrUtils::SetInt(labelset_2->GetOpDesc(), "_label_switch_index", 2);
  FftsPlusCtxDefPtr ctxDefPtr = std::make_shared<domi::FftsPlusCtxDef>();
  labeset_1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  identity1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  identity2->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  labegotoex1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  labelset_2->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);


  (void)ge::AttrUtils::SetBool(labeset_1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(data1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(data1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(data2->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(data2->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(data3->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(data3->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(streamactive1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(streamactive1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(identity1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(identity1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(constan1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(constan1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(add1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(add1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(add2->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(add2->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);


  (void)ge::AttrUtils::SetBool(identity2->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(identity2->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(netoutput1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(netoutput1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(labegotoex1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(labegotoex1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(labelset_2->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(labelset_2->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);


  GenManAicAivCtxDef(add1);
  GenManAicAivCtxDef(add2);
  ThreadSliceMapPtr threadSliceMap = std::make_shared<ThreadSliceMap>();
  labeset_1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  identity1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(identity1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  add1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(add1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  add2->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(add2->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  identity2->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(identity2->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  labegotoex1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labegotoex1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  labelset_2->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labelset_2->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);

  builder.AddControlEdge(labeset_1, streamactive1);
  builder.AddControlEdge(streamactive1, identity1);
  builder.AddDataEdge(data1, 0, identity1, 0);
  builder.AddDataEdge(data2, 0, identity1, 1);
  builder.AddDataEdge(data3, 0, identity1, 2);
  builder.AddDataEdge(identity1, 1, add1, 0);
  builder.AddDataEdge(identity1, 2, add1, 1);
  builder.AddDataEdge(identity1, 0, add2, 1);
  builder.AddDataEdge(constan1, 0, add2, 0);
  builder.AddDataEdge(identity1, 2, identity2, 2);
  builder.AddDataEdge(add1, 0, identity2, 1);
  builder.AddDataEdge(add2, 0, identity2, 0);
  builder.AddDataEdge(identity2, 0, netoutput1, 0);
  builder.AddDataEdge(identity2, 1, netoutput1, 1);
  builder.AddDataEdge(identity2, 2, netoutput1, 2);
  builder.AddControlEdge(netoutput1, labegotoex1);
  builder.AddControlEdge(labegotoex1, labelset_2);
  auto sub_graph = builder.GetGraph();
  builder.SetConstantInputOffset();
  return sub_graph;
}

ComputeGraphPtr BuildGraph_RuntimeOp_TestCase1_While_0_root_graph(const string &subraph_name) {
  auto builder = ut::ComputeGraphBuilder(subraph_name);

  auto data1 = builder.AddNode("data_0_1", "Data", 0, 1);
  auto data2 = builder.AddNode("data_0_2", "Data", 0, 1);
  auto constant1 = builder.AddNode("constant_0_1", "Constant", 0, 1);
  auto data3 = builder.AddNode("data_0_3", "Data", 0, 1);
  auto identity = builder.AddNode("identity_0_1", "Identity", 1, 1);
  auto mu1 = builder.AddNode("mul_0_1", "Mul", 2, 1);
  auto while_0_node = builder.AddNode("While", "While", 3, 3);
  auto cast = builder.AddNode("cast_0_1", "Cast", 1, 1);
  auto pack = builder.AddNode("pack_0_1", "Pack", 3, 1);
  auto add = builder.AddNode("add_0_1", "Add", 2, 1);
  auto variabel = builder.AddNode("variable_1", "Variable", 0, 0);
  auto netoutput0 = builder.AddNode("netoutput_0_1", "NetOutput", 1, 0);


  (void)ge::AttrUtils::SetBool(data1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(data1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(data2->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(data2->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(constant1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(constant1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(data3->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(data3->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(identity->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(identity->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(mu1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(mu1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(while_0_node->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(while_0_node->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(cast->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(cast->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(pack->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(pack->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);


  (void)ge::AttrUtils::SetBool(add->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(add->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(variabel->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(variabel->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(netoutput0->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(netoutput0->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);


  FftsPlusCtxDefPtr ctxDefPtr = std::make_shared<domi::FftsPlusCtxDef>();
  identity->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  while_0_node->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);

  ThreadSliceMapPtr threadSliceMap = std::make_shared<ThreadSliceMap>();
  identity->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(identity->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  mu1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(mu1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  while_0_node->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(while_0_node->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  cast->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(cast->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  pack->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(pack->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  add->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  GenManAicAivCtxDef(add);
  GenManAicAivCtxDef(mu1);
  GenManAicAivCtxDef(cast);
  GenManAicAivCtxDef(pack);
  builder.AddDataEdge(data1, 0, identity, 0);
  builder.AddDataEdge(data1, 0, add, 1);
  builder.AddDataEdge(data2, 0, mu1, 1);
  builder.AddDataEdge(constant1, 0, mu1, 0);
  builder.AddDataEdge(data3, 0, while_0_node, 0);
  builder.AddDataEdge(identity, 0, while_0_node, 2);
  builder.AddDataEdge(mu1, 0, while_0_node, 1);
  builder.AddDataEdge(while_0_node, 0, cast, 0);
  builder.AddDataEdge(cast, 0, pack, 0);
  builder.AddDataEdge(while_0_node, 1, pack, 1);
  builder.AddDataEdge(while_0_node, 2, pack, 2);
  builder.AddDataEdge(pack, 0, add, 0);
  builder.AddDataEdge(add, 0, netoutput0, 0);
  builder.AddControlEdge(variabel, netoutput0);

  auto sub_while_graph = builder.GetGraph();
  builder.SetConstantInputOffset();
  return sub_while_graph;
}

ComputeGraphPtr BuildGraph_RuntimeOp_While_ScopeWrite() {
  auto builder = ut::ComputeGraphBuilder("rts_root_graph");
  auto sub_node = builder.AddNode("sub_node", "sub_node", 1, 0);
  auto root_graph = builder.GetGraph();
  string subgraph_name = "func_op_branch";
  ComputeGraphPtr func_op_branch_graph = BuildGraph_RuntimeOp_TestCase1_While_0_root_graph(subgraph_name);
  func_op_branch_graph->SetParentNode(sub_node);
  func_op_branch_graph->SetParentGraph(root_graph);
  sub_node->GetOpDesc()->AddSubgraphName(subgraph_name);
  sub_node->GetOpDesc()->SetSubgraphInstanceName(0, subgraph_name);
  root_graph->AddSubgraph(subgraph_name, func_op_branch_graph);

  auto while_0_node = func_op_branch_graph->FindNode("While");
  if (while_0_node == nullptr) {
    cout << "[ERROR] FE:sub node is nullptr";
    return root_graph;
  }
  string else_subgraph_name = "condi_sub_graph";
  ComputeGraphPtr else_branch_graph = BuildGraph_RuntimeOp_TestCase1_while_0_condi_sub_graph(else_subgraph_name);
  else_branch_graph->SetParentNode(while_0_node);
  else_branch_graph->SetParentGraph(func_op_branch_graph);
  while_0_node->GetOpDesc()->AddSubgraphName(else_subgraph_name);
  while_0_node->GetOpDesc()->SetSubgraphInstanceName(0, else_subgraph_name);
  root_graph->AddSubgraph(else_subgraph_name, else_branch_graph);
  string then_subgraph_name = "then_sub_graph";
  ComputeGraphPtr then_branch_graph = BuildGraph_RuntimeOp_TestCase1_while_0_then_sub_graph(then_subgraph_name);
  then_branch_graph->SetParentNode(while_0_node);
  then_branch_graph->SetParentGraph(func_op_branch_graph);
  while_0_node->GetOpDesc()->AddSubgraphName(then_subgraph_name);
  while_0_node->GetOpDesc()->SetSubgraphInstanceName(1, then_subgraph_name);
  root_graph->AddSubgraph(then_subgraph_name, then_branch_graph);
  builder.SetConstantInputOffset();
  return root_graph;



}

//case
ComputeGraphPtr BuildGraph_RuntimeOp_TestCase1_case_0_condi_sub_graph1(const string &subraph_name) {
  auto builder = ut::ComputeGraphBuilder(subraph_name);

  auto labelswitchbyindex1 = builder.AddNode("labelswitch_2_1", "LabelSwitchByIndex", 1, 0);
  ge::GeTensorDescPtr tensor_desc = labelswitchbyindex1->GetOpDesc()->MutableInputDesc(0);
  ge::GeTensorPtr tensor_node = nullptr;
  FE_MAKE_SHARED(tensor_node = std::make_shared<ge::GeTensor>(), return nullptr);
  tensor_node->SetTensorDesc(*tensor_desc);
  ge::GeShape shape{};
  uint32_t index = 0;
  tensor_desc->SetDataType(ge::DT_UINT32);
  tensor_desc->SetOriginDataType(ge::DT_UINT32);
  tensor_desc->SetShape(shape);
  tensor_desc->SetOriginShape(shape);
  tensor_desc->SetFormat(ge::FORMAT_ND);
  tensor_desc->SetOriginFormat(ge::FORMAT_ND);
  (void)tensor_node->SetData(reinterpret_cast<uint8_t *>(&index), 4);
  ge::OpDescPtr newdatadesc = nullptr;
  FE_MAKE_SHARED(newdatadesc = std::make_shared<ge::OpDesc>("data_1_1","Constant"), return nullptr);
  (void)ge::AttrUtils::SetBool(newdatadesc, "_is_single_op", true);
  (void)ge::AttrUtils::SetBool(newdatadesc, ge::ATTR_NAME_IS_ORIGINAL_INPUT, true);
  if (newdatadesc->AddOutputDesc(*tensor_desc) != ge::GRAPH_SUCCESS) {
    FE_LOGD("AddOutputDesc Failed");
    return nullptr;
  }
  if (!ge::AttrUtils::SetTensor(newdatadesc, ge::ATTR_NAME_WEIGHTS, tensor_node)) {
    FE_LOGD("SetTensor = null");
    return nullptr;
  }
  auto sub_graph = builder.GetGraph();
  auto data1 = builder.AddNode("data_2_1", "Data", 0, 1);
  (void)ge::AttrUtils::SetInt(data1->GetOpDesc(), "_parent_node_index", 0);


  std::vector<uint32_t> index_list;
  index_list.push_back(0);
  index_list.push_back(1);
  index_list.push_back(2);
  ge::AttrUtils::SetListInt(labelswitchbyindex1->GetOpDesc(), "_label_switch_list", index_list);
  auto labeset_1 = builder.AddNode("labeset_2_1", "LabelSet", 0, 0);
  ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), "_label_switch_index", 1);


  auto streamactive1 = builder.AddNode("streamactive_2_1", "StreamActive", 0, 0);
  auto partitioncall = builder.AddNode("constant_2_1", "Constant", 0, 1);
  auto netoutput1 = builder.AddNode("netoutput_2_1", "NetOutput", 1, 0);
  auto labegotoex1 = builder.AddNode("labelgoto_2_1", "LabelGotoEx", 0, 0);

  (void)ge::AttrUtils::SetBool(data1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(data1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(labeset_1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(streamactive1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(streamactive1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(partitioncall->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(partitioncall->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(netoutput1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(netoutput1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(labegotoex1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(labegotoex1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  ge::AttrUtils::SetInt(labegotoex1->GetOpDesc(), "_label_switch_index", 0);
  FftsPlusCtxDefPtr ctxDefPtr = std::make_shared<domi::FftsPlusCtxDef>();

  labelswitchbyindex1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  labeset_1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  labegotoex1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  ThreadSliceMapPtr threadSliceMap = std::make_shared<ThreadSliceMap>();
  labelswitchbyindex1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labelswitchbyindex1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  labeset_1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  labegotoex1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labegotoex1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);



  builder.AddDataEdge(data1, 0, labelswitchbyindex1, 0);
  auto dataconstanchor = labelswitchbyindex1->GetAllInDataAnchors().at(0);
  if (dataconstanchor == nullptr) {
    std::cout << "[ERROR] FE:dataconstanchor is nullptr" << std::endl;
  }
  (void)ge::AnchorUtils::SetStatus(dataconstanchor, ge::ANCHOR_CONST);
  std::cout << "ff e status = " << static_cast<uint32_t>(ge::AnchorUtils::GetStatus(dataconstanchor)) << std::endl;
  builder.AddControlEdge(labelswitchbyindex1, labeset_1);
  builder.AddControlEdge(labeset_1, streamactive1);
  builder.AddControlEdge(streamactive1, partitioncall);
  builder.AddControlEdge(partitioncall, netoutput1);
  builder.AddControlEdge(netoutput1, labegotoex1);
  builder.SetConstantInputOffset();
  return sub_graph;
}

ComputeGraphPtr BuildGraph_RuntimeOp_TestCase1_case_0_condi_sub_graph2(const string &subraph_name) {
  auto builder = ut::ComputeGraphBuilder(subraph_name);

  auto labeset_1 = builder.AddNode("labeset_1_1", "LabelSet", 0, 0);
  ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), "_label_switch_index", 2);
  auto streamactive1 = builder.AddNode("streamactive_1_1", "StreamActive", 0, 0);
  auto partitioncall = builder.AddNode("constant_1_1", "Constant", 0, 1);
  auto netoutput1 = builder.AddNode("netoutput_1_1", "NetOutput", 1, 0);
  auto labegotoex1 = builder.AddNode("labelgoto_1_1", "LabelGotoEx", 0, 0);
  ge::AttrUtils::SetInt(labegotoex1->GetOpDesc(), "_label_switch_index", 0);

  FftsPlusCtxDefPtr ctxDefPtr = std::make_shared<domi::FftsPlusCtxDef>();


  labeset_1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  labegotoex1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  ThreadSliceMapPtr threadSliceMap = std::make_shared<ThreadSliceMap>();
  labeset_1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  labegotoex1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labegotoex1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);

  (void)ge::AttrUtils::SetBool(labeset_1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(streamactive1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(streamactive1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(partitioncall->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(partitioncall->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(netoutput1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(netoutput1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(labegotoex1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(labegotoex1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  builder.AddControlEdge(labeset_1, streamactive1);
  builder.AddControlEdge(streamactive1, partitioncall);
  builder.AddControlEdge(partitioncall, netoutput1);
  builder.AddControlEdge(netoutput1, labegotoex1);
  auto sub_graph = builder.GetGraph();
  builder.SetConstantInputOffset();
  return sub_graph;
}

ComputeGraphPtr BuildGraph_RuntimeOp_TestCase1_case_0_condi_sub_graph3(const string &subraph_name) {
  auto builder = ut::ComputeGraphBuilder(subraph_name);

  auto labeset_1 = builder.AddNode("labelset_3_1", "LabelSet", 0, 0);
  ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), "_label_switch_index", 3);
  auto streamactive1 = builder.AddNode("streamactive_3_1", "StreamActive", 0, 0);
  auto partitioncall = builder.AddNode("constant_3_1", "Constant", 0, 1);
  auto netoutput1 = builder.AddNode("netoutput_3_1", "NetOutput", 1, 0);
  auto labelset_2 = builder.AddNode("labelset_3_2", "LabelSet", 0, 0);
  ge::AttrUtils::SetInt(labelset_2->GetOpDesc(), "_label_switch_index",  0);
  FftsPlusCtxDefPtr ctxDefPtr = std::make_shared<domi::FftsPlusCtxDef>();


  (void)ge::AttrUtils::SetBool(labeset_1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(streamactive1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(streamactive1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(partitioncall->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(partitioncall->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(netoutput1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(netoutput1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(labelset_2->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(labelset_2->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  labeset_1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  labelset_2->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  ThreadSliceMapPtr threadSliceMap = std::make_shared<ThreadSliceMap>();
  labeset_1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  labelset_2->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labelset_2->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);

  builder.AddControlEdge(labeset_1, streamactive1);
  builder.AddControlEdge(streamactive1, partitioncall);
  builder.AddControlEdge(partitioncall, netoutput1);
  builder.AddControlEdge(netoutput1, labelset_2);
  auto sub_graph = builder.GetGraph();
  builder.SetConstantInputOffset();
  return sub_graph;
}


ComputeGraphPtr BuildGraph_RuntimeOp_TestCase1_Case_0_root_graph(const string &subraph_name) {
  auto builder = ut::ComputeGraphBuilder(subraph_name);

  auto data1 = builder.AddNode("data_0_1", "Data", 0, 1);
  auto case1 = builder.AddNode("Case", "Case", 1, 1);
  auto variabel = builder.AddNode("variable_1", "Variable", 0, 0);
  auto netoutput0 = builder.AddNode("netoutput_0_1", "NetOutput", 1, 0);


  (void)ge::AttrUtils::SetBool(data1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(data1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(case1->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(case1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(variabel->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(variabel->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  (void)ge::AttrUtils::SetBool(netoutput0->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(netoutput0->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);

  FftsPlusCtxDefPtr ctxDefPtr = std::make_shared<domi::FftsPlusCtxDef>();

  case1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);

  ThreadSliceMapPtr threadSliceMap = std::make_shared<ThreadSliceMap>();
  case1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(case1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);

  builder.AddDataEdge(data1, 0, case1, 0);
  builder.AddDataEdge(case1, 0, netoutput0, 1);
  builder.AddControlEdge(variabel, netoutput0);
  GenManAicAivCtxDef(case1);
  auto sub_case_graph = builder.GetGraph();
  builder.SetConstantInputOffset();
  return sub_case_graph;
}

ComputeGraphPtr BuildGraph_RuntimeOp_Case_ScopeWrite() {
  auto builder = ut::ComputeGraphBuilder("rts_root_graph");
  auto sub_node = builder.AddNode("sub_node", "sub_node", 1, 0);
  auto root_graph = builder.GetGraph();
  string subgraph_name = "func_op_branch";
  ComputeGraphPtr func_op_branch_graph = BuildGraph_RuntimeOp_TestCase1_Case_0_root_graph(subgraph_name);
  func_op_branch_graph->SetParentNode(sub_node);
  func_op_branch_graph->SetParentGraph(root_graph);
  sub_node->GetOpDesc()->AddSubgraphName(subgraph_name);
  sub_node->GetOpDesc()->SetSubgraphInstanceName(0, subgraph_name);
  root_graph->AddSubgraph(subgraph_name, func_op_branch_graph);
   auto case1 = func_op_branch_graph->FindNode("Case");
  if (case1 == nullptr) {
    cout << "[ERROR] FE:sub node is nullptr";
    return root_graph;
  }
  string case_swtich_subgraph_name = "case_sub_graph1";
  ComputeGraphPtr case_branch_graph = BuildGraph_RuntimeOp_TestCase1_case_0_condi_sub_graph1(case_swtich_subgraph_name);
  case_branch_graph->SetParentNode(case1);
  case_branch_graph->SetParentGraph(func_op_branch_graph);
  case1->GetOpDesc()->AddSubgraphName(case_swtich_subgraph_name);
  case1->GetOpDesc()->SetSubgraphInstanceName(0, case_swtich_subgraph_name);
  root_graph->AddSubgraph(case_swtich_subgraph_name, case_branch_graph);
  string case1_subgraph_name = "case_sub_graph2";
  ComputeGraphPtr case1_branch_graph = BuildGraph_RuntimeOp_TestCase1_case_0_condi_sub_graph2(case1_subgraph_name);
  case1_branch_graph->SetParentNode(case1);
  case1_branch_graph->SetParentGraph(func_op_branch_graph);
  case1->GetOpDesc()->AddSubgraphName(case1_subgraph_name);
  case1->GetOpDesc()->SetSubgraphInstanceName(1, case1_subgraph_name);
  root_graph->AddSubgraph(case1_subgraph_name, case1_branch_graph);
  string case2_subgraph_name = "case_sub_graph3";
  ComputeGraphPtr case2_branch_graph = BuildGraph_RuntimeOp_TestCase1_case_0_condi_sub_graph3(case2_subgraph_name);
  case2_branch_graph->SetParentNode(case1);
  case2_branch_graph->SetParentGraph(func_op_branch_graph);
  case1->GetOpDesc()->AddSubgraphName(case2_subgraph_name);
  case1->GetOpDesc()->SetSubgraphInstanceName(2, case2_subgraph_name);
  root_graph->AddSubgraph(case2_subgraph_name, case2_branch_graph);
  builder.SetConstantInputOffset();
  return root_graph;
}

//if_if
ComputeGraphPtr BuildGraph_RuntimeOp_TestCase1_If_If_1_if_else_sub_graph(const string &subraph_name) {
  auto builder = ut::ComputeGraphBuilder(subraph_name);
  auto labeset_1 = builder.AddNode("labeset_1_1_1", "LabelSet", 0, 0);
  ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), "_label_switch_index", 1);
  auto data1 = builder.AddNode("data_1_1_1", "Data", 0, 1);
  auto data2 = builder.AddNode("data_1_1_2", "Data", 0, 1);

  auto streamactive1 = builder.AddNode("streamactive_1_1_1", "StreamActive", 0, 0);
  auto constant1 = builder.AddNode("constant_1_1_1", "Constant", 0, 1);
  auto realdiv1 = builder.AddNode("realdiv_1_1_1", "RealDiv", 2, 1);
  auto identity1 = builder.AddNode("identity_1_1_1", "Identity", 1, 1);
  auto netoutput0 = builder.AddNode("netoutput_1_1_1", "NetOutput", 2, 0);
  auto labelset_2 = builder.AddNode("labeset_1_1_2", "LabelSet", 0, 0);
  ge::AttrUtils::SetInt(labelset_2->GetOpDesc(), "_label_switch_index", 2);
  FftsPlusCtxDefPtr ctxDefPtr = std::make_shared<domi::FftsPlusCtxDef>();
  labeset_1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  identity1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  labelset_2->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  GenManAicAivCtxDef(realdiv1);
  ThreadSliceMapPtr threadSliceMap = std::make_shared<ThreadSliceMap>();
  labeset_1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  realdiv1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(realdiv1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  labelset_2->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labelset_2->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  identity1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(identity1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);

  builder.AddControlEdge(labeset_1, streamactive1);
  builder.AddControlEdge(streamactive1, realdiv1);
  builder.AddDataEdge(data1, 0, realdiv1, 0);
  builder.AddDataEdge(data2, 0, realdiv1, 1);
  builder.AddControlEdge(streamactive1, identity1);
  builder.AddDataEdge(constant1, 0, identity1, 0);
  builder.AddDataEdge(identity1, 0, netoutput0, 0);
  builder.AddDataEdge(realdiv1, 0, netoutput0, 1);
  builder.AddControlEdge(netoutput0, labelset_2);
  auto sub_graph = builder.GetGraph();
  builder.SetConstantInputOffset();
  return sub_graph;
}


ComputeGraphPtr BuildGraph_RuntimeOp_TestCase1_If_If_1_if_thensub_graph(const string &subraph_name) {
  auto builder = ut::ComputeGraphBuilder(subraph_name);


  auto labelswitchbyindex1 = builder.AddNode("labelswitch_2_1", "LabelSwitchByIndex", 1, 0);
  ge::GeTensorDescPtr tensor_desc = labelswitchbyindex1->GetOpDesc()->MutableInputDesc(0);
  ge::GeTensorPtr tensor_node = nullptr;
  FE_MAKE_SHARED(tensor_node = std::make_shared<ge::GeTensor>(), return nullptr);
  tensor_node->SetTensorDesc(*tensor_desc);
  ge::GeShape shape{};
  uint32_t index = 0;
  tensor_desc->SetDataType(ge::DT_UINT32);
  tensor_desc->SetOriginDataType(ge::DT_UINT32);
  tensor_desc->SetShape(shape);
  tensor_desc->SetOriginShape(shape);
  tensor_desc->SetFormat(ge::FORMAT_ND);
  tensor_desc->SetOriginFormat(ge::FORMAT_ND);
  (void)tensor_node->SetData(reinterpret_cast<uint8_t *>(&index), 4);
  ge::OpDescPtr newdatadesc = nullptr;
  FE_MAKE_SHARED(newdatadesc = std::make_shared<ge::OpDesc>("data_2_1","Constant"), return nullptr);
  (void)ge::AttrUtils::SetBool(newdatadesc, "_is_single_op", true);
  (void)ge::AttrUtils::SetBool(newdatadesc, ge::ATTR_NAME_IS_ORIGINAL_INPUT, true);
  if (newdatadesc->AddOutputDesc(*tensor_desc) != ge::GRAPH_SUCCESS) {
    FE_LOGD("AddOutputDesc Failed");
    return nullptr;
  }
  if (!ge::AttrUtils::SetTensor(newdatadesc, ge::ATTR_NAME_WEIGHTS, tensor_node)) {
    FE_LOGD("SetTensor = null");
    return nullptr;
  }
  auto sub_graph = builder.GetGraph();
  auto data1 = builder.AddNode("data_2_1_1", "Data", 0, 1);
  (void)ge::AttrUtils::SetInt(data1->GetOpDesc(), "_parent_node_index", 0);


  std::vector<uint32_t> index_list;
  index_list.push_back(0);
  index_list.push_back(1);
  ge::AttrUtils::SetListInt(labelswitchbyindex1->GetOpDesc(), "_label_switch_list", index_list);

  auto labeset_1 = builder.AddNode("labelset_2_1_1", "LabelSet", 0, 0);
  ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), "_label_switch_index", 0);
  auto data2 = builder.AddNode("data_2_1_2", "Data", 0, 1);
  auto data3 = builder.AddNode("data_2_1_3", "Data", 0, 1);
  auto streamactive1 = builder.AddNode("streamactive_2_1_1", "StreamActive", 0, 0);
  auto add1 = builder.AddNode("add_2_1_1", "Add", 2, 1);
  auto identity1 = builder.AddNode("identity_2_1_1", "Identity", 1, 1);
  auto netoutput1 = builder.AddNode("netoutput_2_1_1", "NetOutput", 1, 1);
  auto labegotoex1 = builder.AddNode("labelgoto_2_1_1", "LabelGotoEx", 1, 0);
  ge::AttrUtils::SetInt(labegotoex1->GetOpDesc(), "_label_switch_index", 2);
  FftsPlusCtxDefPtr ctxDefPtr = std::make_shared<domi::FftsPlusCtxDef>();
  labelswitchbyindex1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  labeset_1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  identity1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  labegotoex1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  GenManAicAivCtxDef(identity1);
   GenManAicAivCtxDef(add1);
  ThreadSliceMapPtr threadSliceMap = std::make_shared<ThreadSliceMap>();
  labelswitchbyindex1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labelswitchbyindex1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  labeset_1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  add1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(add1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  identity1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(identity1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  labegotoex1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labegotoex1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);


  builder.AddDataEdge(data1, 0, labelswitchbyindex1, 0);
    auto dataconstanchor = labelswitchbyindex1->GetAllInDataAnchors().at(0);
  (void)ge::AnchorUtils::SetStatus(dataconstanchor, ge::ANCHOR_CONST);
  builder.AddControlEdge(labelswitchbyindex1, labeset_1);
  builder.AddControlEdge(labeset_1, streamactive1);
  builder.AddDataEdge(data2, 0, add1, 0);
  builder.AddDataEdge(data3, 0, add1, 1);
  builder.AddControlEdge(streamactive1, add1);
  builder.AddControlEdge(streamactive1, identity1);
  builder.AddDataEdge(data2, 0, identity1, 0);
  builder.AddDataEdge(add1, 0, netoutput1, 0);
  builder.AddDataEdge(identity1, 0, netoutput1, 0);
  builder.AddDataEdge(identity1, 0, netoutput1, 1);
  builder.AddControlEdge(netoutput1, labegotoex1);
  builder.SetConstantInputOffset();
  return sub_graph;
}

ComputeGraphPtr BuildGraph_RuntimeOp_TestCase1_If_IF_0_else_sub_graph(const string &subraph_name) {
  auto builder = ut::ComputeGraphBuilder(subraph_name);
  auto labeset_1 = builder.AddNode("labeset_1_1", "LabelSet", 0, 0);
  ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), "_label_switch_index", 4);
  auto labelset_2 = builder.AddNode("labeset_1_2", "LabelSet", 1, 0);
  ge::AttrUtils::SetInt(labelset_2->GetOpDesc(), "_label_switch_index", 5);
  auto data1 = builder.AddNode("data_1_1", "Data", 0, 1);
  auto streamactive1 = builder.AddNode("streamactive_1_1", "StreamActive", 0, 0);
  auto square1 = builder.AddNode("square_1_1", "Square", 1, 1);
  auto netoutput0 = builder.AddNode("netoutput_0_1", "NetOutput", 1, 1);


  FftsPlusCtxDefPtr ctxDefPtr = std::make_shared<domi::FftsPlusCtxDef>();
  labeset_1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  labelset_2->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);

  ThreadSliceMapPtr threadSliceMap = std::make_shared<ThreadSliceMap>();
  labeset_1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  labelset_2->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labelset_2->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  square1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(square1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);

  GenManAicAivCtxDef(square1);
  builder.AddControlEdge(labeset_1, streamactive1);
  builder.AddControlEdge(streamactive1, square1);
  builder.AddDataEdge(data1, 0, square1, 0);
  builder.AddDataEdge(square1, 0, netoutput0, 0);
  builder.AddControlEdge(netoutput0, labelset_2);
  auto sub_graph = builder.GetGraph();
  builder.SetConstantInputOffset();
  return sub_graph;
}

ComputeGraphPtr BuildGraph_RuntimeOp_TestCase1_If_IF_0_thensub_graph(const string &subraph_name) {
  auto builder = ut::ComputeGraphBuilder(subraph_name);


  auto labelswitchbyindex1 = builder.AddNode("labelswitch_2_1", "LabelSwitchByIndex", 1, 0);
  ge::GeTensorDescPtr tensor_desc = labelswitchbyindex1->GetOpDesc()->MutableInputDesc(0);
  ge::GeTensorPtr tensor_node = nullptr;
  FE_MAKE_SHARED(tensor_node = std::make_shared<ge::GeTensor>(), return nullptr);
  tensor_node->SetTensorDesc(*tensor_desc);
  ge::GeShape shape{};
  uint32_t index = 0;
  tensor_desc->SetDataType(ge::DT_UINT32);
  tensor_desc->SetOriginDataType(ge::DT_UINT32);
  tensor_desc->SetShape(shape);
  tensor_desc->SetOriginShape(shape);
  tensor_desc->SetFormat(ge::FORMAT_ND);
  tensor_desc->SetOriginFormat(ge::FORMAT_ND);
  (void)tensor_node->SetData(reinterpret_cast<uint8_t *>(&index), 4);
  ge::OpDescPtr newdatadesc = nullptr;
  FE_MAKE_SHARED(newdatadesc = std::make_shared<ge::OpDesc>("data_2_1","Constant"), return nullptr);
  (void)ge::AttrUtils::SetBool(newdatadesc, "_is_single_op", true);
  (void)ge::AttrUtils::SetBool(newdatadesc, ge::ATTR_NAME_IS_ORIGINAL_INPUT, true);
  if (newdatadesc->AddOutputDesc(*tensor_desc) != ge::GRAPH_SUCCESS) {
    FE_LOGD("AddOutputDesc Failed");
    return nullptr;
  }
  if (!ge::AttrUtils::SetTensor(newdatadesc, ge::ATTR_NAME_WEIGHTS, tensor_node)) {
    FE_LOGD("SetTensor = null");
    return nullptr;
  }
  auto sub_graph = builder.GetGraph();
  auto data1 = builder.AddNode("data_2_3", "Data", 0, 1);
  (void)ge::AttrUtils::SetInt(data1->GetOpDesc(), "_parent_node_index", 0);


  std::vector<uint32_t> index_list;
  index_list.push_back(4);
  index_list.push_back(3);
  ge::AttrUtils::SetListInt(labelswitchbyindex1->GetOpDesc(), "_label_switch_list", index_list);

  auto labeset_1 = builder.AddNode("labelset_2_1", "LabelSet", 0, 0);
  ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), "_label_switch_index", 3);
  auto data2 = builder.AddNode("data_2_2", "Data", 0, 1);

  auto streamactive1 = builder.AddNode("streamactive_2_1", "StreamActive", 0, 0);
  auto data3 = builder.AddNode("data_2_3", "Data", 0, 1);
  auto data4 = builder.AddNode("data_2_4", "Data", 0, 1);
  auto constant1 = builder.AddNode("constant_2_1", "Constant", 0, 1);
  auto add1 = builder.AddNode("add_2_1", "Add", 2, 1);
  auto mul1 = builder.AddNode("mul_2_1", "Mul", 2, 1);
  auto square1 = builder.AddNode("square_2_1", "Square", 2, 1);
  auto if_1_node = builder.AddNode("If_1_0", "If", 4, 0);
  auto netoutput1 = builder.AddNode("netoutput_2_1", "NetOutput", 1, 1);
  auto labegotoex1 = builder.AddNode("labelgoto_2_1", "LabelGotoEx", 1, 0);
  ge::AttrUtils::SetInt(labegotoex1->GetOpDesc(), "_label_switch_index", 3);
  FftsPlusCtxDefPtr ctxDefPtr = std::make_shared<domi::FftsPlusCtxDef>();
  labelswitchbyindex1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  labeset_1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  if_1_node->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  labegotoex1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);


  ThreadSliceMapPtr threadSliceMap = std::make_shared<ThreadSliceMap>();
  labelswitchbyindex1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labelswitchbyindex1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  labeset_1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  mul1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(mul1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  add1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(add1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  if_1_node->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(if_1_node->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  square1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(square1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  labegotoex1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labegotoex1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  GenManAicAivCtxDef(square1);
  GenManAicAivCtxDef(add1);
  GenManAicAivCtxDef(mul1);
  builder.AddDataEdge(data1, 0, labelswitchbyindex1, 0);
    auto dataconstanchor = labelswitchbyindex1->GetAllInDataAnchors().at(0);
  (void)ge::AnchorUtils::SetStatus(dataconstanchor, ge::ANCHOR_CONST);
  builder.AddControlEdge(labelswitchbyindex1, labeset_1);
  builder.AddControlEdge(labeset_1, streamactive1);
  builder.AddControlEdge(streamactive1, mul1);
  builder.AddDataEdge(data3, 0, mul1, 0);
  builder.AddDataEdge(data4, 0, mul1, 1);
  builder.AddControlEdge(streamactive1, mul1);
  builder.AddDataEdge(constant1, 0, add1, 0);
  builder.AddDataEdge(constant1, 0, add1, 1);
  builder.AddControlEdge(streamactive1, add1);
  builder.AddDataEdge(data3, 0, square1, 0);
  builder.AddDataEdge(data4, 0, square1, 1);
  builder.AddControlEdge(streamactive1, square1);

  builder.AddDataEdge(add1, 0, if_1_node, 0);
  builder.AddDataEdge(data2, 0, if_1_node, 1);
  builder.AddDataEdge(square1, 0, if_1_node, 2);
  builder.AddDataEdge(data4, 0, if_1_node, 3);
  builder.AddDataEdge(mul1, 0, netoutput1, 0);
  builder.AddControlEdge(if_1_node, netoutput1);
  builder.AddControlEdge(netoutput1, labegotoex1);
  return sub_graph;
}


ComputeGraphPtr BuildGraph_RuntimeOp_TestCase1_If_If_0_root_graph(const string &subraph_name) {
  auto builder = ut::ComputeGraphBuilder(subraph_name);
  auto if_0_node = builder.AddNode("If", "If", 4, 1);
  auto data1 = builder.AddNode("data_0_1", "Data", 0, 1);
  auto data2 = builder.AddNode("data_0_2", "Data", 0, 3);
  auto data3 = builder.AddNode("data_0_3", "Data", 0, 1);
  auto square1 = builder.AddNode("square_0_1", "Square", 2, 1);
  auto add = builder.AddNode("add_0_1", "Add", 2, 1);
  auto variable = builder.AddNode("variable_0_1", "Variable", 0, 1);
  auto netoutput0 = builder.AddNode("netoutput_0_1", "NetOutput", 2, 0);


  FftsPlusCtxDefPtr ctxDefPtr = std::make_shared<domi::FftsPlusCtxDef>();
  if_0_node->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);

  ThreadSliceMapPtr threadSliceMap = std::make_shared<ThreadSliceMap>();
  if_0_node->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(if_0_node->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  square1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(square1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  add->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(add->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  variable->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(variable->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  GenManAicAivCtxDef(square1);
  builder.AddDataEdge(data1, 0, if_0_node, 0);
  builder.AddDataEdge(data1, 0, if_0_node, 1);
  builder.AddDataEdge(square1, 0, if_0_node, 2);
  builder.AddDataEdge(data2, 0, if_0_node, 3);
  builder.AddDataEdge(data3, 0, square1, 0);
  builder.AddDataEdge(data3, 0, square1, 0);
  builder.AddDataEdge(if_0_node, 0, add, 0);
  builder.AddDataEdge(data2, 0, add, 1);
  builder.AddDataEdge(add, 0, netoutput0, 0);
  builder.AddControlEdge(variable, netoutput0);
   GenManAicAivCtxDef(add);
  auto sub_if_graph = builder.GetGraph();
  builder.SetConstantInputOffset();
  return sub_if_graph;
}

ComputeGraphPtr BuildGraph_RuntimeOp_If_If_ScopeWrite() {
  auto builder = ut::ComputeGraphBuilder("rts_root_graph");
  auto sub_node = builder.AddNode("sub_node", "sub_node", 1, 0);
  auto root_graph = builder.GetGraph();
  string subgraph_name = "func_op_branch";
  ComputeGraphPtr func_op_branch_graph = BuildGraph_RuntimeOp_TestCase1_If_If_0_root_graph(subgraph_name);
  func_op_branch_graph->SetParentNode(sub_node);
  func_op_branch_graph->SetParentGraph(root_graph);
  sub_node->GetOpDesc()->AddSubgraphName(subgraph_name);
  sub_node->GetOpDesc()->SetSubgraphInstanceName(0, subgraph_name);
  root_graph->AddSubgraph(subgraph_name, func_op_branch_graph);
  builder.SetConstantInputOffset();

  auto if_0_node = func_op_branch_graph->FindNode("If");
  if (if_0_node == nullptr) {
    cout << "[ERROR] FE:sub node is nullptr";
    return root_graph;
  }
  string then_subgraph_name = "then_sub_graph";
  ComputeGraphPtr then_branch_graph = BuildGraph_RuntimeOp_TestCase1_If_IF_0_thensub_graph(then_subgraph_name);
  then_branch_graph->SetParentNode(if_0_node);
  then_branch_graph->SetParentGraph(func_op_branch_graph);
  if_0_node->GetOpDesc()->AddSubgraphName(then_subgraph_name);
  if_0_node->GetOpDesc()->SetSubgraphInstanceName(0, then_subgraph_name);
  root_graph->AddSubgraph(then_subgraph_name, then_branch_graph);

  string else_subgraph_name = "else_sub_graph";
  ComputeGraphPtr else_branch_graph = BuildGraph_RuntimeOp_TestCase1_If_IF_0_else_sub_graph(else_subgraph_name);
  else_branch_graph->SetParentNode(if_0_node);
  else_branch_graph->SetParentGraph(func_op_branch_graph);
  if_0_node->GetOpDesc()->AddSubgraphName(else_subgraph_name);
  if_0_node->GetOpDesc()->SetSubgraphInstanceName(1, else_subgraph_name);
  root_graph->AddSubgraph(else_subgraph_name, else_branch_graph);
  auto if_1_node = then_branch_graph->FindNode("If_1_0");
  if (if_1_node == nullptr) {
    cout << "[ERROR] FE:sub sub node is nullptr";
    return root_graph;
  }
  string then_then_subgraph_name = "then_then_sub_graph";
  ComputeGraphPtr then_then_branch_graph = BuildGraph_RuntimeOp_TestCase1_If_If_1_if_thensub_graph(then_then_subgraph_name);
  then_then_branch_graph->SetParentNode(if_1_node);
  then_then_branch_graph->SetParentGraph(then_branch_graph);
  if_1_node->GetOpDesc()->AddSubgraphName(then_then_subgraph_name);
  if_1_node->GetOpDesc()->SetSubgraphInstanceName(0, then_then_subgraph_name);
  root_graph->AddSubgraph(then_then_subgraph_name, then_then_branch_graph);

  string then_else_subgraph_name = "then_else_sub_graph";
  ComputeGraphPtr then_else_branch_graph = BuildGraph_RuntimeOp_TestCase1_If_If_1_if_else_sub_graph(then_else_subgraph_name);
  then_else_branch_graph->SetParentNode(if_1_node);
  then_else_branch_graph->SetParentGraph(then_branch_graph);
  if_1_node->GetOpDesc()->AddSubgraphName(then_else_subgraph_name);
  if_1_node->GetOpDesc()->SetSubgraphInstanceName(1, then_else_subgraph_name);
  root_graph->AddSubgraph(then_else_subgraph_name, then_else_branch_graph);
  return root_graph;
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, RTSOP_GenerateTask_SUCCESS)
{
  ComputeGraphPtr func_op_branch_graph = nullptr;
  ComputeGraphPtr graph = BuildGraph_RuntimeOp_ScopeWrite(func_op_branch_graph, false);
  cout << "========================RTSOP GENTASK BEGIN========================" << endl;
  auto sub_node = graph->FindNode("sub_node");

  if (sub_node == nullptr) {
    cout << "[ERROR] FE:sub node is nullptr";
  }
  RunContext context = CreateContext();
  vector<domi::TaskDef> tasks;
  ffts_plus_ops_kernel_builder_ptr->schecule_policy_pass_ = ScheculePolicyPassStub2;
  Status ret = ffts_plus_ops_kernel_builder_ptr->GenerateTask(*sub_node, context, tasks);
  EXPECT_EQ(ffts::FAILED, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, Auto_RTSOP_GenerateTask_SUCCESS)
{
  ComputeGraphPtr func_op_branch_graph = nullptr;
  ComputeGraphPtr graph = BuildGraph_RuntimeOp_ScopeWrite(func_op_branch_graph, true);
  cout << "========================AUTO RTSOP GENTASK BEGIN========================" << endl;
  auto sub_node = graph->FindNode("sub_node");
  EXPECT_NE(sub_node, nullptr);
  auto cpy_node = func_op_branch_graph->FindNode("identity");
  EXPECT_NE(cpy_node, nullptr);
  const auto &tensor_desc = cpy_node->GetOpDesc()->MutableInputDesc(0U);
  EXPECT_NE(tensor_desc, nullptr);
  tensor_desc->SetShape(GeShape({-1, -1}));

  RunContext context = CreateContext();
  RunContextPtr contxt_ptr = std::make_shared<ge::RunContext>(context);
  for (auto node : func_op_branch_graph->GetAllNodes()) {
    (void)node->GetOpDesc()->SetExtAttr(kRuntimeContentx, contxt_ptr);
  }
  vector<domi::TaskDef> tasks;
  ffts_plus_ops_kernel_builder_ptr->schecule_policy_pass_ = ScheculePolicyPassStub2;
  Status ret = ffts_plus_ops_kernel_builder_ptr->GenerateTask(*sub_node, context, tasks);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, AICPU_GenerateTask_Schecule_Failed)
{
  ComputeGraphPtr graph = BuildGraph_AICPU_ScopeWrite();
  cout << "========================aicpu GENTASK BEGIN========================" << endl;
  auto sub_node = graph->FindNode("sub_node");

  if (sub_node == nullptr) {
    cout << "[ERROR] FE:sub node is nullptr";
  }
  RunContext context = CreateContext();
  vector<domi::TaskDef> tasks;
  ffts_plus_ops_kernel_builder_ptr->schecule_policy_pass_ = ScheculePolicyPassStub1;
  Status ret = ffts_plus_ops_kernel_builder_ptr->GenerateTask(*sub_node, context, tasks);

  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, RTSOP_UNFOLDCALLONLYONEDEPTH)
{
  ComputeGraphPtr func_op_branch_graph = nullptr;
  ComputeGraphPtr graph = BuildGraph_RuntimeOp_ScopeWrite(func_op_branch_graph, false);
  cout << "=======================RTSOP_UNFOLDCALLONLYONEDEPTH========================" << endl;
  auto sub_node = graph->FindNode("sub_node");

  if (sub_node == nullptr) {
    cout << "[ERROR] FE:sub node is nullptr";
  }
  RunContext context = CreateContext();
  vector<domi::TaskDef> tasks;
  Status ret = UnfoldPartionCallOnlyOneDepth(*(graph.get()), "sub_node");

  EXPECT_EQ(ffts::FAILED, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, RTSOP_GenerateTask_IF_SETIF)
{
  ComputeGraphPtr graph = BuildGraph_RuntimeOp_If_ScopeWrite();
  cout << "\n========================IF START ========================" << endl;
  auto sub_node = graph->FindNode("sub_node");

  if (sub_node == nullptr) {
    cout << "[ERROR] FE:sub node is nullptr";
  }
  RunContext context = CreateContext();
  vector<domi::TaskDef> tasks;
  ffts_plus_ops_kernel_builder_ptr->schecule_policy_pass_ = ScheculePolicyPassStub2;
  Status ret = ffts_plus_ops_kernel_builder_ptr->GenerateTask(*sub_node, context, tasks);
  EXPECT_EQ(ffts::SUCCESS, ret);
}


TEST_F(FFTSPlusOpsKernelBuilderUTest, RTSOP_GenerateTask_CASE_SETIF)
{
  ComputeGraphPtr graph = BuildGraph_RuntimeOp_Case_ScopeWrite();
  cout << "========================CASE START ========================" << endl;
  auto sub_node = graph->FindNode("sub_node");

  if (sub_node == nullptr) {
    cout << "[ERROR] FE:sub node is nullptr";
  }
  RunContext context = CreateContext();
  vector<domi::TaskDef> tasks;
  ffts_plus_ops_kernel_builder_ptr->schecule_policy_pass_ = ScheculePolicyPassStub2;
  Status ret = ffts_plus_ops_kernel_builder_ptr->GenerateTask(*sub_node, context, tasks);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, RTSOP_GenerateTask_While_SETIF)
{
  ComputeGraphPtr graph = BuildGraph_RuntimeOp_While_ScopeWrite();
  cout << "========================WHILE START ========================" << endl;
  auto sub_node = graph->FindNode("sub_node");

  if (sub_node == nullptr) {
    cout << "[ERROR] FE:sub node is nullptr";
  }
  RunContext context = CreateContext();
  vector<domi::TaskDef> tasks;
  ffts_plus_ops_kernel_builder_ptr->schecule_policy_pass_ = ScheculePolicyPassStub2;
  Status ret = ffts_plus_ops_kernel_builder_ptr->GenerateTask(*sub_node, context, tasks);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, RTSOP_GenerateTask_If_If_SETIF)
{
  ComputeGraphPtr graph = BuildGraph_RuntimeOp_If_If_ScopeWrite();
  cout << "========================If_If START ========================" << endl;
  auto sub_node = graph->FindNode("sub_node");

  if (sub_node == nullptr) {
    cout << "[ERROR] FE:sub node is nullptr";
  }
  RunContext context = CreateContext();
  vector<domi::TaskDef> tasks;
  ffts_plus_ops_kernel_builder_ptr->schecule_policy_pass_ = ScheculePolicyPassStub2;
  Status ret = ffts_plus_ops_kernel_builder_ptr->GenerateTask(*sub_node, context, tasks);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, AICPU_GenerateTask_SUCCESS)
{
  ComputeGraphPtr graph = BuildGraph_AICPU_ScopeWrite();
  cout << "========================aicpu GENTASK BEGIN========================" << endl;
  auto sub_node = graph->FindNode("sub_node");

  if (sub_node == nullptr) {
    cout << "[ERROR] FE:sub node is nullptr";
  }
  RunContext context = CreateContext();
  vector<domi::TaskDef> tasks;
  ffts_plus_ops_kernel_builder_ptr->schecule_policy_pass_ = ScheculePolicyPassStub2;
  Status ret = ffts_plus_ops_kernel_builder_ptr->GenerateTask(*sub_node, context, tasks);

  EXPECT_EQ(ffts::SUCCESS, ret);
}

#if 0
TEST_F(FFTSPlusOpsKernelBuilderUTest, DOWITHMANALLABELX)
{
  auto builder = ut::ComputeGraphBuilder("DOWITHMANALLABELX");

  auto labelswitchbyindex1 = builder.AddNode("labelswitch_2_1", "LabelSwitchByIndex", 1, 0);
  ge::GeTensorDescPtr tensor_desc = labelswitchbyindex1->GetOpDesc()->MutableInputDesc(0);
  ge::GeTensorPtr tensor_node = nullptr;
  FE_MAKE_SHARED(tensor_node = std::make_shared<ge::GeTensor>(), return);
  tensor_node->SetTensorDesc(*tensor_desc);
  ge::GeShape shape{};
  uint32_t index = 0;
  tensor_desc->SetDataType(ge::DT_UINT32);
  tensor_desc->SetOriginDataType(ge::DT_UINT32);
  tensor_desc->SetShape(shape);
  tensor_desc->SetOriginShape(shape);
  tensor_desc->SetFormat(ge::FORMAT_ND);
  tensor_desc->SetOriginFormat(ge::FORMAT_ND);
  (void)tensor_node->SetData(reinterpret_cast<uint8_t *>(&index), 4);
  ge::OpDescPtr newdatadesc = nullptr;
  FE_MAKE_SHARED(newdatadesc = std::make_shared<ge::OpDesc>("data_2_1","Data"), return);
  (void)ge::AttrUtils::SetBool(newdatadesc, "_is_single_op", true);
  (void)ge::AttrUtils::SetBool(newdatadesc, ge::ATTR_NAME_IS_ORIGINAL_INPUT, true);
  if (newdatadesc->AddOutputDesc(*tensor_desc) != ge::GRAPH_SUCCESS) {
    FE_LOGD("AddOutputDesc Failed");
  }
  if (!ge::AttrUtils::SetTensor(newdatadesc, ge::ATTR_NAME_WEIGHTS, tensor_node)) {
    FE_LOGD("SetTensor = null");

  }
  auto sub_graph = builder.GetGraph();
  ge::NodePtr data1 = sub_graph->AddNode(newdatadesc);
  (void)ge::AttrUtils::SetInt(data1->GetOpDesc(), "_parent_node_index", 0);

  std::vector<uint32_t> index_list;
  index_list.push_back(0);
  index_list.push_back(1);
  auto control_node = builder.AddNode("If", "If", 1, 1);
  auto if_input_node = builder.AddNode("Data_p_1", "Data", 0, 1);
  auto if_output_node = builder.AddNode("Data_p_1", "Data", 1, 0);

  ge::AttrUtils::SetListInt(labelswitchbyindex1->GetOpDesc(), "_label_switch_list", index_list);
  auto labeset_1 = builder.AddNode("labelset_2_1", "LabelSet", 0, 0);
  ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), "_label_switch_index", 0);
  auto labeset_2 = builder.AddNode("labelset_2_2", "LabelSet", 0, 0);
  ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), "_label_switch_index", 1);
  auto data2 = builder.AddNode("data_2_2", "Data", 0, 1);
  auto data3 = builder.AddNode("data_2_3", "Data", 0, 1);
  auto streamactive1 = builder.AddNode("streamactive_2_1", "StreamActive", 0, 0);
  auto mul1 = builder.AddNode("mul_2_1", "Mul", 2, 1);
  auto netoutput1 = builder.AddNode("netoutput_2_1", "NetOutput", 1, 0);
  auto labegotoex1 = builder.AddNode("labelgoto_2_1", "LabelGotoEx", 0, 0);
  ge::AttrUtils::SetInt(labegotoex1->GetOpDesc(), "_label_switch_index", 1);
  FftsPlusCtxDefPtr ctxDefPtr = std::make_shared<domi::FftsPlusCtxDef>();
  labelswitchbyindex1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  labeset_1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  labegotoex1->GetOpDesc()->SetExtAttr("FFTS_PLUS_TASK_DEF", ctxDefPtr);
  GenManAicAivCtxDef(mul1);
  ThreadSliceMapPtr threadSliceMap = std::make_shared<ThreadSliceMap>();
  labelswitchbyindex1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labelswitchbyindex1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  labeset_1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labeset_1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  mul1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(mul1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  labegotoex1->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, threadSliceMap);
  ge::AttrUtils::SetInt(labegotoex1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);


  builder.AddDataEdge(data1, 0, labelswitchbyindex1, 0);
  builder.AddControlEdge(labelswitchbyindex1, labeset_1);
  builder.AddControlEdge(labeset_1, streamactive1);
  builder.AddControlEdge(streamactive1, mul1);
  builder.AddDataEdge(data2, 0, mul1, 0);
  builder.AddDataEdge(data3, 0, mul1, 1);
  builder.AddDataEdge(mul1, 0, netoutput1, 0);

  builder.AddDataEdge(if_input_node, 0, control_node, 0);
  builder.AddDataEdge(control_node, 0, if_output_node, 0);
  builder.AddControlEdge(netoutput1, labegotoex1);
  builder.AddControlEdge(labegotoex1, labeset_2);
  std::map<ge::NodePtr, std::vector<std::pair<ge::ComputeGraph, \
  std::map<std::string, std::vector<ge::NodePtr>>>>> node_pair;
  std::vector<ge::NodePtr> labelsets;
  labelsets.push_back(labeset_1);
  labelsets.push_back(labeset_2);
  std::map<std::string, std::vector<ge::NodePtr>> nodemap;
  nodemap.emplace(std::make_pair("LabelSet", labelsets));
  std::vector<std::pair<ge::ComputeGraph, std::map<std::string, std::vector<ge::NodePtr>>>> nodemapvector;
  nodemapvector.push_back(std::make_pair(*sub_graph , nodemap));
  node_pair.emplace(std::make_pair(control_node, nodemapvector));
  (void)ffts_plus_manual_mode_ptr->DoWithLabelSwitchByIndex(control_node, *sub_graph,labelswitchbyindex1, node_pair);
  (void)ffts_plus_manual_mode_ptr->DoWithLabelGoto(control_node, *sub_graph, labegotoex1, node_pair);
  std::vector<ge::NodePtr> labelswitches;
  labelswitches.push_back(labelswitchbyindex1);
  (void)ffts_plus_manual_mode_ptr->DoWithLabelSwitchs(control_node, *sub_graph, labelswitches, node_pair);
  (void)ffts_plus_manual_mode_ptr->RepalceDataWithReal(control_node, *sub_graph, labelswitchbyindex1, node_pair);
  Status ret = ffts::SUCCESS;
  EXPECT_EQ(ffts::SUCCESS, ret);
}
#endif
TEST_F(FFTSPlusOpsKernelBuilderUTest, HCCL_GenerateTask_SUCCESS)
{
  ComputeGraphPtr graph = BuildGraph_HCCL_Graph();
  cout << "========================hccl GENTASK BEGIN========================" << endl;
  auto sub_node = graph->FindNode("sub_node");
  if (sub_node == nullptr) {
    cout << "[ERROR] FE:sub node is nullptr";
  }
  RunContext context = CreateContext();
  vector<domi::TaskDef> tasks;
  ffts_plus_ops_kernel_builder_ptr->schecule_policy_pass_ = ScheculePolicyPassStub2;
  Status ret = ffts_plus_ops_kernel_builder_ptr->GenerateTask(*sub_node, context, tasks);

  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, AutoFillContextSuccList)
{
  auto builder = ut::ComputeGraphBuilder("test");
  auto input = builder.AddNode("test", "test", 0, 1);
  auto out_node = builder.AddNode("test1", "NetOutput", 1, 0);
  builder.AddDataEdge(input, 0, out_node, 0);
  vector<uint32_t> context_id_list = {3, 4};
  ge::AttrUtils::SetListInt(out_node->GetOpDesc(), kAutoCtxIdList, context_id_list);
  domi::FftsPlusTaskDef taskdef;
  domi::FftsPlusCtxDef *ffts_plus_ctx_def = taskdef.add_ffts_plus_ctx();
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_NOTIFY_RECORD);
  ffts_plus_ctx_def->set_context_id(0);
  ffts_plus_ctx_def = taskdef.add_ffts_plus_ctx();
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_SDMA);
  ffts_plus_ctx_def->set_context_id(1);
  ffts_plus_ctx_def = taskdef.add_ffts_plus_ctx();
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_NOTIFY_WAIT);
  ffts_plus_ctx_def->set_context_id(2);
  ffts_plus_ctx_def = taskdef.add_ffts_plus_ctx();
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_WRITE_VALUE);
  ffts_plus_ctx_def->set_context_id(3);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AIV);
  ffts_plus_ctx_def->set_context_id(4);
  ffts_plus_ctx_def = taskdef.add_ffts_plus_ctx();
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_MIX_AIV);
  ffts_plus_ctx_def->set_context_id(5);
  ffts_plus_ctx_def = taskdef.add_ffts_plus_ctx();
  FFTSPlusTaskBuilderPtr task_builder = nullptr;
  FFTS_MAKE_SHARED(task_builder = std::make_shared<FFTSPlusTaskBuilder>(), return);
  vector<uint32_t> context_list;
  context_list.push_back(0);
  context_list.push_back(1);
  context_list.push_back(2);
  context_list.push_back(3);
  context_list.push_back(4);
  context_list.push_back(5);
  ffts_plus_auto_mode_ptr->ffts_plus_task_def_ = &taskdef;
  bool netoutput_flag = false;
  Status ret = ffts_plus_auto_mode_ptr->FillContextSuccList(input, task_builder, context_list,
                                                            context_list, netoutput_flag);
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, AutoFillSerialDependency_SUCCESS)
{
  auto builder = ut::ComputeGraphBuilder("test_serial_dependency");
  auto input = builder.AddNode("data1", "Data", 0, 1);
  auto node1 = builder.AddNode("square_1_1", "Square", 1, 1);
  auto node2 = builder.AddNode("relu", "Relu", 1, 1);
  auto netoutput0 = builder.AddNode("netoutput_0_1", "NetOutput", 1, 1);
  ge::AttrUtils::SetListInt(node1->GetOpDesc(), kAutoCtxIdList, {3, 4});
  ge::AttrUtils::SetListInt(node2->GetOpDesc(), kAutoCtxIdList, {5, 6});

  std::shared_ptr<std::vector<ge::NodePtr>> suc_nodes = nullptr;
  FFTS_MAKE_SHARED(suc_nodes = std::make_shared<std::vector<ge::NodePtr>>(), return);
  suc_nodes->emplace_back(node2);
  node1->GetOpDesc()->SetExtAttr(kNonEdgeSuccList, suc_nodes);

  domi::FftsPlusTaskDef taskdef;
  domi::FftsPlusCtxDef *ffts_plus_ctx_def = taskdef.add_ffts_plus_ctx();
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_LABEL);
  ffts_plus_ctx_def->set_context_id(0);
  ffts_plus_ctx_def = taskdef.add_ffts_plus_ctx();
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AT_START);
  ffts_plus_ctx_def->set_context_id(1);
  ffts_plus_ctx_def = taskdef.add_ffts_plus_ctx();
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AT_START);
  ffts_plus_ctx_def->set_context_id(2);
  ffts_plus_ctx_def = taskdef.add_ffts_plus_ctx();
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AIV);
  ffts_plus_ctx_def->set_context_id(3);
  ffts_plus_ctx_def = taskdef.add_ffts_plus_ctx();
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AIV);
  ffts_plus_ctx_def->set_context_id(4);
  ffts_plus_ctx_def = taskdef.add_ffts_plus_ctx();
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AIV);
  ffts_plus_ctx_def->set_context_id(5);
  ffts_plus_ctx_def = taskdef.add_ffts_plus_ctx();
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AIV);
  ffts_plus_ctx_def->set_context_id(6);
  ffts_plus_ctx_def = taskdef.add_ffts_plus_ctx();
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AT_END);
  ffts_plus_ctx_def->set_context_id(7);
  ffts_plus_ctx_def = taskdef.add_ffts_plus_ctx();
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AT_END);
  ffts_plus_ctx_def->set_context_id(8);
  ffts_plus_ctx_def = taskdef.add_ffts_plus_ctx();
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_LABEL);
  ffts_plus_ctx_def->set_context_id(9);
  ffts_plus_auto_mode_ptr->ffts_plus_task_def_ = &taskdef;
  FFTSPlusTaskBuilderPtr task_builder = nullptr;
  FFTS_MAKE_SHARED(task_builder = std::make_shared<FFTSPlusTaskBuilder>(), return);
  Status ret = ffts_plus_auto_mode_ptr->FillSerialDependency(node1, ffts_plus_auto_mode_ptr->ffts_plus_task_def_,
                                                             task_builder, {3, 4});
  EXPECT_EQ(fe::SUCCESS, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, AutoFillSerialDependency_FAILED1) {
  auto builder = ut::ComputeGraphBuilder("test_serial_dependency");
  auto input = builder.AddNode("data1", "Data", 0, 1);
  auto node1 = builder.AddNode("square_1_1", "Square", 1, 1);
  auto node2 = builder.AddNode("relu", "Relu", 1, 1);
  auto netoutput0 = builder.AddNode("netoutput_0_1", "NetOutput", 1, 1);
  ge::AttrUtils::SetListInt(node1->GetOpDesc(), kAutoCtxIdList, {3, 4});

  std::shared_ptr<std::vector<ge::NodePtr>> suc_nodes = nullptr;
  FFTS_MAKE_SHARED(suc_nodes = std::make_shared<std::vector<ge::NodePtr>>(), return);
  suc_nodes->emplace_back(node2);
  node1->GetOpDesc()->SetExtAttr(kNonEdgeSuccList, suc_nodes);

  FFTSPlusTaskBuilderPtr task_builder = nullptr;
  FFTS_MAKE_SHARED(task_builder = std::make_shared<FFTSPlusTaskBuilder>(), return);
  Status ret = ffts_plus_auto_mode_ptr->FillSerialDependency(node1, ffts_plus_auto_mode_ptr->ffts_plus_task_def_,
                                                             task_builder, {3, 4});
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, AutoFillSerialDependency_FAILED2) {
  auto builder = ut::ComputeGraphBuilder("test_serial_dependency");
  auto input = builder.AddNode("data1", "Data", 0, 1);
  auto node1 = builder.AddNode("square_1_1", "Square", 1, 1);
  auto node2 = builder.AddNode("relu", "Relu", 1, 1);
  auto netoutput0 = builder.AddNode("netoutput_0_1", "NetOutput", 1, 1);
  ge::AttrUtils::SetListInt(node1->GetOpDesc(), kAutoCtxIdList, {3, 4});
  ge::AttrUtils::SetListInt(node2->GetOpDesc(), kAutoCtxIdList, {5, 6});

  std::shared_ptr<std::vector<ge::NodePtr>> suc_nodes = nullptr;
  FFTS_MAKE_SHARED(suc_nodes = std::make_shared<std::vector<ge::NodePtr>>(), return);
  suc_nodes->emplace_back(node2);
  node1->GetOpDesc()->SetExtAttr(kNonEdgeSuccList, suc_nodes);

  FFTSPlusTaskBuilderPtr task_builder = nullptr;
  FFTS_MAKE_SHARED(task_builder = std::make_shared<FFTSPlusTaskBuilder>(), return);
  Status ret = ffts_plus_auto_mode_ptr->FillSerialDependency(node1, ffts_plus_auto_mode_ptr->ffts_plus_task_def_,
                                                             task_builder, {{3, 4, 5}});
  EXPECT_EQ(fe::FAILED, ret);
}

ge::Status TestOpExtGenTask(const ge::Node &node, ge::RunContext &context, std::vector<domi::TaskDef> &tasks) {
  domi::TaskDef tmp_task;
  tasks.emplace_back(tmp_task);
  return ge::SUCCESS;
}

ge::Status TestOpExtGenTaskFail(const ge::Node &node, ge::RunContext &context, std::vector<domi::TaskDef> &tasks) {
  return ge::FAILED;
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, MIX_L2_GenerateTask_SUCCESS)
{
  RunContext context = CreateContext();
  vector<domi::TaskDef> tasks;
  auto builder = ut::ComputeGraphBuilder("test");
  auto sub_node = builder.AddNode("mix_node", "mix_node", 1, 0);
  (void)ge::AttrUtils::SetStr(sub_node->GetOpDesc(), ffts::ATTR_NAME_ALIAS_ENGINE_NAME, "ffts_plus");
  FftsPlusCtxDefPtr ffts_plus_ctx_def = nullptr;
  ffts_plus_ctx_def = std::make_shared<domi::FftsPlusCtxDef>();
  domi::FftsPlusMixAicAivCtxDef *mix_l2_ctx = ffts_plus_ctx_def->mutable_mix_aic_aiv_ctx();
  mix_l2_ctx->set_ns(1);
  mix_l2_ctx->set_atm(1);
  mix_l2_ctx->set_thread_dim(1);
  mix_l2_ctx->set_tail_block_ratio_n(2);
  mix_l2_ctx->set_non_tail_block_ratio_n(2);
  mix_l2_ctx->add_task_addr(0x1114);
  mix_l2_ctx->add_task_addr(0x1111);
  mix_l2_ctx->add_task_addr(0x1112);
  mix_l2_ctx->add_kernel_name("mix1");
  mix_l2_ctx->add_kernel_name("mix2");
  (void)sub_node->GetOpDesc()->SetExtAttr(kAttrAICoreCtxDef, ffts_plus_ctx_def);
  ge::ComputeGraphPtr tmp_graph = std::make_shared<ge::ComputeGraph>("OpCompileGraph");
  ge::OpDescPtr memset_op_desc_ptr =  make_shared<ge::OpDesc>("memset_node", fe::MEMSET_OP_TYPE);
  ge::NodePtr memset_node = tmp_graph->AddNode(memset_op_desc_ptr, sub_node->GetOpDesc()->GetId());
  sub_node->GetOpDesc()->SetExtAttr(fe::ATTR_NAME_MEMSET_NODE, memset_node);
  (void) ge::AttrUtils::SetInt(sub_node->GetOpDesc(), kModeInArgsFirstField, 1);
  ffts_plus_ops_kernel_builder_ptr->schecule_policy_pass_ = ScheculePolicyPassStub2;
  (void)ge::AttrUtils::SetBool(sub_node->GetOpDesc(), kFFTSPlusInDynamic, true);

  REGISTER_NODE_EXT_GENTASK("mix_node", TestOpExtGenTask);

  Status ret = ffts_plus_ops_kernel_builder_ptr->GenerateTask(*sub_node, context, tasks);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(tasks.size(), 2);
  domi::TaskDef &task_def = tasks.at(0U);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  uint64_t gen_ctx_num = ffts_plus_task_def->ffts_plus_ctx_size();
  EXPECT_EQ(gen_ctx_num, 2);

  tasks.clear();
  REGISTER_NODE_EXT_GENTASK("mix_node", TestOpExtGenTaskFail);
  ret = ffts_plus_ops_kernel_builder_ptr->GenerateTask(*sub_node, context, tasks);
  EXPECT_EQ(fe::FAILED, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, tiling_sink_gentask_for_ffts)
{
  RunContext context = CreateContext();
  vector<domi::TaskDef> tasks;
  auto builder = ut::ComputeGraphBuilder("test");
  auto sub_node = builder.AddNode("mix_node", "mix_node", 1, 0);
  (void)ge::AttrUtils::SetStr(sub_node->GetOpDesc(), ffts::ATTR_NAME_ALIAS_ENGINE_NAME, "ffts_plus");
  FftsPlusCtxDefPtr ffts_plus_ctx_def = nullptr;
  ffts_plus_ctx_def = std::make_shared<domi::FftsPlusCtxDef>();
  domi::FftsPlusMixAicAivCtxDef *mix_l2_ctx = ffts_plus_ctx_def->mutable_mix_aic_aiv_ctx();
  mix_l2_ctx->set_ns(1);
  mix_l2_ctx->set_atm(1);
  mix_l2_ctx->set_thread_dim(1);
  mix_l2_ctx->set_tail_block_ratio_n(2);
  mix_l2_ctx->set_non_tail_block_ratio_n(2);
  mix_l2_ctx->add_task_addr(0x1114);
  mix_l2_ctx->add_task_addr(0x1111);
  mix_l2_ctx->add_task_addr(0x1112);
  mix_l2_ctx->add_kernel_name("mix1");
  mix_l2_ctx->add_kernel_name("mix2");
  (void)sub_node->GetOpDesc()->SetExtAttr(kAttrAICoreCtxDef, ffts_plus_ctx_def);
  ge::ComputeGraphPtr tmp_graph = std::make_shared<ge::ComputeGraph>("OpCompileGraph");
  ge::OpDescPtr memset_op_desc_ptr =  make_shared<ge::OpDesc>("memset_node", fe::MEMSET_OP_TYPE);
  ge::NodePtr memset_node = tmp_graph->AddNode(memset_op_desc_ptr, sub_node->GetOpDesc()->GetId());
  sub_node->GetOpDesc()->SetExtAttr(fe::ATTR_NAME_MEMSET_NODE, memset_node);
  (void) ge::AttrUtils::SetInt(sub_node->GetOpDesc(), kModeInArgsFirstField, 1);
  ffts_plus_ops_kernel_builder_ptr->schecule_policy_pass_ = ScheculePolicyPassStub2;
  (void)ge::AttrUtils::SetBool(sub_node->GetOpDesc(), kFFTSPlusInDynamic, true);

  REGISTER_NODE_EXT_GENTASK("mix_node", TestOpExtGenTask);

  Status ret = ffts_plus_ops_kernel_builder_ptr->GenerateTask(*sub_node, context, tasks);
  EXPECT_EQ(fe::SUCCESS, ret);
  EXPECT_EQ(tasks.size(), 2);
  domi::TaskDef &task_def = tasks.at(0U);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  uint64_t gen_ctx_num = ffts_plus_task_def->ffts_plus_ctx_size();
  EXPECT_EQ(gen_ctx_num, 2);

  task_def.set_type(RT_MODEL_TASK_FFTS_PLUS_TASK);
  domi::KernelDefWithHandle *kernel_def = task_def.mutable_kernel_with_handle();
  kernel_def->set_args_size(66);
  string args(66, '1');
  kernel_def->set_args(args.data(), 66);
  domi::KernelContext *kernel_context = kernel_def->mutable_context();
  kernel_context->set_op_index(1);
  kernel_context->set_args_format("{i0}{i_instance0}{i_desc1}{o0}{o_instance0}{ws0}{t_ffts.non_tail}");
  gert::ExeResGenerationCtxBuilder exe_ctx_builder;
  auto res_ptr_holder = exe_ctx_builder.CreateOpExeContext(*sub_node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_ptr_holder->context_);
  fe::ParamDef param;
  ret = fe::GenerateTaskForSinkOp(op_exe_res_ctx, param, tasks);
}

ge::graphStatus GenTaskKernelFuncFail(const gert::ExeResGenerationContext *context,
                                  std::vector<std::vector<uint8_t>> &tasks) {
  return ge::GRAPH_FAILED;
}

ge::graphStatus GenTaskKernelFunc(const gert::ExeResGenerationContext *context,
                                  std::vector<std::vector<uint8_t>> &tasks) {
  const_cast<gert::ExeResGenerationContext *>(context)->SetWorkspaceBytes({333, 444});
  return ge::GRAPH_SUCCESS;
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, Case_Gen_Sub_Graph_Task_WithMemSet)
{
  std::vector<ge::NodePtr> sub_graph_nodes;
  uint32_t thread_mode = 0;
  uint32_t window_size = 4;
  ComputeGraphPtr sgt_graph = BuilGraph_Sgt_Subgraph_3("sgt_graph", thread_mode, window_size);
  uint64_t ready_context_num = 0;
  uint64_t total_context_number = 0;
  ffts_plus_manual_mode_ptr->Initialize();
  std::vector<ge::NodePtr> atomic_nodes;
  Status ret = ffts_plus_manual_mode_ptr->GenFftsPlusContextId(*sgt_graph, sub_graph_nodes, ready_context_num,
                                                             total_context_number, atomic_nodes);
  EXPECT_EQ(ready_context_num, 1);
  EXPECT_EQ(total_context_number, 4);
  EXPECT_EQ(atomic_nodes.size(), 1);
  domi::TaskDef task_def;
  ret = ffts_plus_manual_mode_ptr->GenSubGraphTaskDef(atomic_nodes, sub_graph_nodes, task_def);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  uint64_t gen_ctx_num = ffts_plus_task_def->ffts_plus_ctx_size();

  EXPECT_EQ(gen_ctx_num, 4);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, Dynamic_Mode_SUCCESS)
{
  auto builder = ut::ComputeGraphBuilder("test");
  auto sub_node = builder.AddNode("dy_node", "dy_node", 1, 0);
  ge::GeTensorDescPtr tensor_desc = sub_node->GetOpDesc()->MutableInputDesc(0);
  ge::GeShape shape({ -1, -1, -1});
  tensor_desc->SetDataType(ge::DT_UINT32);
  tensor_desc->SetShape(shape);
  tensor_desc->SetOriginShape(shape);
  tensor_desc->SetFormat(ge::FORMAT_ND);
  tensor_desc->SetOriginFormat(ge::FORMAT_ND);
  auto sub_graph = builder.GetGraph();
  TheadTaskBuilderPtr base_mode_ptr = nullptr;
  base_mode_ptr = ffts_plus_ops_kernel_builder_ptr->GetFftsPlusMode(*(sub_node.get()), *(sub_graph.get()));
  EXPECT_NE(nullptr, base_mode_ptr);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, WITHOUT_SUB_STREAM_ID_SUCCESS)
{
  std::vector<ge::NodePtr> sub_graph_nodes;
  uint32_t thread_mode = 0;
  uint32_t window_size = 4;
  ComputeGraphPtr sgt_graph = BuilGraph_Sgt_Subgraph_4("sgt_graph", thread_mode, window_size, false);
  uint64_t ready_context_num = 0;
  uint64_t total_context_number = 0;
  ffts_plus_manual_mode_ptr->Initialize();
  ffts_plus_ops_kernel_builder_ptr->GenSerialDependency(sgt_graph);
  std::shared_ptr<std::vector<ge::NodePtr>> succlist;
  std::shared_ptr<std::vector<ge::NodePtr>> pre_node;
  for (auto &node : sgt_graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Add") {
      succlist = node->GetOpDesc()->TryGetExtAttr(kNonEdgeSuccList, succlist);
      EXPECT_EQ(succlist, nullptr);
    } else if (node->GetOpDesc()->GetType() == "PRelu") {
      pre_node = node->GetOpDesc()->TryGetExtAttr(kNonEdgePreNodes, pre_node);
      EXPECT_EQ(succlist, nullptr);
    } else if (node->GetOpDesc()->GetType() == "Relu") {
      pre_node = node->GetOpDesc()->TryGetExtAttr(kNonEdgePreNodes, pre_node);
      EXPECT_EQ(succlist, nullptr);
      succlist = node->GetOpDesc()->TryGetExtAttr(kNonEdgeSuccList, succlist);
      EXPECT_EQ(succlist, nullptr);
    }
  }

  std::vector<ge::NodePtr> atomic_nodes;
  Status ret = ffts_plus_manual_mode_ptr->GenFftsPlusContextId(*sgt_graph, sub_graph_nodes, ready_context_num,
                                                               total_context_number, atomic_nodes);
  EXPECT_EQ(ready_context_num, 1); // add
  EXPECT_EQ(total_context_number, 3);
  domi::TaskDef task_def;
  ret = ffts_plus_manual_mode_ptr->GenSubGraphTaskDef(atomic_nodes, sub_graph_nodes, task_def);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  uint64_t gen_ctx_num = ffts_plus_task_def->ffts_plus_ctx_size();

  EXPECT_EQ(gen_ctx_num, 3);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, SAME_SUB_STREAM_ID_SUCCESS)
{
  std::vector<ge::NodePtr> sub_graph_nodes;
  uint32_t thread_mode = 0;
  uint32_t window_size = 4;
  ComputeGraphPtr sgt_graph = BuilGraph_Sgt_Subgraph_4("sgt_graph", thread_mode, window_size, true);
  uint64_t ready_context_num = 0;
  uint64_t total_context_number = 0;
  ffts_plus_manual_mode_ptr->Initialize();
  ffts_plus_ops_kernel_builder_ptr->GenSerialDependency(sgt_graph);

  for (auto &node : sgt_graph->GetDirectNode()) {
    std::shared_ptr<std::vector<ge::NodePtr>> succlist;
    std::shared_ptr<std::vector<ge::NodePtr>> pre_node;
    if (node->GetOpDesc()->GetType() == "Add") {
      succlist = node->GetOpDesc()->TryGetExtAttr(kNonEdgeSuccList, succlist);
      EXPECT_EQ(succlist, nullptr);
    } else if (node->GetOpDesc()->GetType() == "PRelu") {
      pre_node = node->GetOpDesc()->TryGetExtAttr(kNonEdgePreNodes, pre_node);
      EXPECT_EQ(pre_node, nullptr);
      succlist = node->GetOpDesc()->TryGetExtAttr(kNonEdgeSuccList, succlist);
      EXPECT_EQ((*succlist).size(), 1);
    } else if (node->GetOpDesc()->GetType() == "Relu") {
      pre_node = node->GetOpDesc()->TryGetExtAttr(kNonEdgePreNodes, pre_node);
      EXPECT_EQ((*pre_node).size(), 1);
      succlist = node->GetOpDesc()->TryGetExtAttr(kNonEdgeSuccList, succlist);
      EXPECT_EQ(succlist, nullptr);
    }
  }

  std::vector<ge::NodePtr> atomic_nodes;
  Status ret = ffts_plus_manual_mode_ptr->GenFftsPlusContextId(*sgt_graph, sub_graph_nodes, ready_context_num,
                                                               total_context_number, atomic_nodes);
  EXPECT_EQ(ready_context_num, 1);
  EXPECT_EQ(total_context_number, 3);
  domi::TaskDef task_def;
  ret = ffts_plus_manual_mode_ptr->GenSubGraphTaskDef(atomic_nodes, sub_graph_nodes, task_def);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  uint64_t gen_ctx_num = ffts_plus_task_def->ffts_plus_ctx_size();

  EXPECT_EQ(gen_ctx_num, 3);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, SAME_SUB_STREAM_ID_GEN_TASK_SUCCESS)
{
  std::vector<ge::NodePtr> sub_graph_nodes;
  uint32_t thread_mode = 0;
  uint32_t window_size = 4;
  ComputeGraphPtr sgt_graph = BuilGraph_Sgt_Subgraph_4("sgt_graph", thread_mode, window_size, true);
  uint64_t ready_context_num = 0;
  uint64_t total_context_number = 0;
  ffts_plus_manual_mode_ptr->Initialize();
  ffts_plus_ops_kernel_builder_ptr->GenSerialDependency(sgt_graph);

  std::vector<ge::NodePtr> atomic_nodes;
  Status ret = ffts_plus_manual_mode_ptr->GenFftsPlusContextId(*sgt_graph, sub_graph_nodes, ready_context_num,
                                                               total_context_number, atomic_nodes);
  EXPECT_EQ(ready_context_num, 1);
  EXPECT_EQ(total_context_number, 3);
  domi::TaskDef task_def;
  ret = ffts_plus_manual_mode_ptr->GenSubGraphTaskDef(atomic_nodes, sub_graph_nodes, task_def);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  uint64_t gen_ctx_num = ffts_plus_task_def->ffts_plus_ctx_size();

  EXPECT_EQ(gen_ctx_num, 3);
  EXPECT_EQ(ffts::SUCCESS, ret);
  // add1
  domi::FftsPlusCtxDef ffts_plus_ctx_def = ffts_plus_task_def->_impl_.ffts_plus_ctx_[0];
  domi::FftsPlusAicAivCtxDef *aic_aiv_ctx_def = ffts_plus_ctx_def.mutable_aic_aiv_ctx();
  EXPECT_EQ(aic_aiv_ctx_def->successor_list_size(), 2);
  EXPECT_EQ(aic_aiv_ctx_def->successor_list(0), 2);
  EXPECT_EQ(aic_aiv_ctx_def->successor_list(1), 1);

  // prelu1
  ffts_plus_ctx_def = ffts_plus_task_def->_impl_.ffts_plus_ctx_[1];
  aic_aiv_ctx_def = ffts_plus_ctx_def.mutable_aic_aiv_ctx();
  EXPECT_EQ(aic_aiv_ctx_def->successor_list_size(), 1);
  EXPECT_EQ(aic_aiv_ctx_def->successor_list(0), 2);
  EXPECT_EQ(aic_aiv_ctx_def->_impl_.pred_cnt_, 1);

  // relu1
  ffts_plus_ctx_def = ffts_plus_task_def->_impl_.ffts_plus_ctx_[2];
  aic_aiv_ctx_def = ffts_plus_ctx_def.mutable_aic_aiv_ctx();
  EXPECT_EQ(aic_aiv_ctx_def->successor_list_size(), 0);
  EXPECT_EQ(aic_aiv_ctx_def->_impl_.pred_cnt_, 2);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, GenInLabelAtStartCtxDefFail)
{
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto builder = ut::ComputeGraphBuilder("test");
  auto sub_node = builder.AddNode("mix_node", "mix_node", 1, 0);
  ge::ComputeGraphPtr tmp_graph = std::make_shared<ge::ComputeGraph>("OpCompileGraph");
  ge::OpDescPtr memset_op_desc_ptr =  make_shared<ge::OpDesc>("memset_node", fe::MEMSET_OP_TYPE);
  ge::NodePtr memset_node = tmp_graph->AddNode(memset_op_desc_ptr, sub_node->GetOpDesc()->GetId());
  ThreadSliceMap thread_slice_map;
  const uint32_t kAutoMode = 1U;
  ThreadSliceMapPtr thread_slice_map_ptr = std::make_shared<ThreadSliceMap>(thread_slice_map);
  thread_slice_map_ptr->thread_mode = kAutoMode;
  memset_node->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, thread_slice_map_ptr);
  Status ret = ffts_plus_auto_mode_ptr->GenInLabelAtStartCtxDef(memset_node, ffts_plus_task_def);
  EXPECT_EQ(ffts::FAILED, ret);
}

TEST_F(FFTSPlusOpsKernelBuilderUTest, UpdateContextForRemoveDuplicate)
{
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  vector<FftsPlusContextPath> context_paths;
  map<uint32_t, vector<uint32_t>> real_ctx_succ_list;
  /*
  *           0
  * 1 2 3 4 5........256
  *          257
  */
  for (uint32_t i = 0; i <= 257; ++i) {
    auto ctx = ffts_plus_task_def->add_ffts_plus_ctx();
    ctx->set_context_type(RT_CTX_TYPE_AICORE);
    FftsPlusContextPath context_path;
    context_path.ctx_id = i;
    context_path.policy_pri = 0;
    if (i == 0) {
      context_path.pre_cnt = 0;
      for (uint32_t j = 1; j <= 256; ++j) {
        context_path.succ_list.emplace_back(j);
      }
      real_ctx_succ_list[i] = context_path.succ_list;
      // for duplicate
      context_path.succ_list.emplace_back(1);
    } else if (i == 257) {
      context_path.pre_cnt = 256;
      for (uint32_t j = 1; j <= 256; ++j) {
        context_path.pre_list.emplace_back(j);
      }
      real_ctx_succ_list[i] = context_path.succ_list;
    } else {
      context_path.pre_cnt = 1;
      if (i == 1) {
        // for duplicate
        context_path.pre_cnt = 2;
      }
      context_path.pre_list.emplace_back(0);
      context_path.succ_list.emplace_back(257);
      real_ctx_succ_list[i] = context_path.succ_list;
    }
    auto label = ctx->mutable_aic_aiv_ctx();
    label->set_pred_cnt(context_path.pre_cnt);
    label->set_pred_cnt_init(context_path.pre_cnt);
    context_paths.emplace_back(context_path);
  }
  for (uint32_t i = 258; i < 268; ++i) {
    //256 outputs need 10 lable
    auto ctx = ffts_plus_task_def->add_ffts_plus_ctx();
    ctx->set_context_type(RT_CTX_TYPE_LABEL);
    auto &context_path = context_paths[0];
    context_path.label_list.emplace_back(i);
  }
  TimeLineOptimizerContext context;
  context.ctx_path_vector_ = context_paths;
  Status ret = ffts_plus_ops_kernel_builder_ptr->UpdateContexts(ffts_plus_task_def, real_ctx_succ_list, context);
  ret = ffts_plus_ops_kernel_builder_ptr->UpdateContextsPreList(task_def, context.ctx_path_vector_, context.cmo_id_map_);
  EXPECT_EQ(ret, ffts::SUCCESS);
  EXPECT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 269); // add one label for precnt
  auto ctx_new = ffts_plus_task_def->mutable_ffts_plus_ctx(268);
  auto label_new = ctx_new->mutable_label_ctx();
  EXPECT_EQ(label_new->pred_cnt(), 255);
  EXPECT_EQ(label_new->successor_num(), 1);
  auto ctx_258 = ffts_plus_task_def->mutable_ffts_plus_ctx(257);
  auto label_258 = ctx_258->mutable_aic_aiv_ctx();
  EXPECT_EQ(label_258->pred_cnt(), 2);
}

/*
 *  old ctxid list is 0,1,2,3,4,5
 *  label null ctxid is 0 2 4
 *  old ctxlist         new ctxlist
 *    0          x
 *    1                    0
 *    2          x
 *    3                    1
 *    4          x
 *    5                    2
 *    so the relation map will be {(0, 1),(1, 3),(2, 5)}
 *    dependency is   1
 *                   / \
 *                  /   5
 *                  |  /
 *                   3
 *   after rebuild,dependency should be
 *                    0
 *                   / \
 *                  /   2
 *                  |  /
 *                   1
 */
TEST_F(FFTSPlusOpsKernelBuilderUTest, ReBuildCtxIdsRelationSucc)
{
  domi::TaskDef task_def;
  domi::TaskDef task_def_new;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto ctx = ffts_plus_task_def->add_ffts_plus_ctx();
  ctx->set_context_type(RT_CTX_TYPE_LABEL);
  ctx = ffts_plus_task_def->add_ffts_plus_ctx();
  ctx->set_context_type(RT_CTX_TYPE_AICORE);
  auto aic_ctx = ctx->mutable_aic_aiv_ctx();
  aic_ctx->set_successor_num(2);
  aic_ctx->add_successor_list(3);
  aic_ctx->add_successor_list(5);

  ctx = ffts_plus_task_def->add_ffts_plus_ctx();
  ctx->set_context_type(RT_CTX_TYPE_LABEL);
  ctx = ffts_plus_task_def->add_ffts_plus_ctx();
  ctx->set_context_type(RT_CTX_TYPE_PERSISTENT_CACHE);
  aic_ctx = ctx->mutable_aic_aiv_ctx();
  aic_ctx->set_pred_cnt(2);
  aic_ctx->set_pred_cnt_init(2);
  ctx = ffts_plus_task_def->add_ffts_plus_ctx();
  ctx->set_context_type(RT_CTX_TYPE_LABEL);
  ctx = ffts_plus_task_def->add_ffts_plus_ctx();
  ctx->set_context_type(RT_CTX_TYPE_AICORE);
  aic_ctx = ctx->mutable_aic_aiv_ctx();
  aic_ctx->set_pred_cnt(1);
  aic_ctx->set_pred_cnt_init(1);
  aic_ctx->set_successor_num(1);
  aic_ctx->add_successor_list(3);

  std::unordered_map<uint32_t, uint32_t> new_old_map;
  std::unordered_map<uint32_t, uint32_t> old_new_map;
  Status ret = ffts_plus_ops_kernel_builder_ptr->ReBuildCtxIdsRelation(task_def, new_old_map, old_new_map);
  EXPECT_EQ(new_old_map[0], 1);
  EXPECT_EQ(new_old_map[1], 3);
  EXPECT_EQ(new_old_map[2], 5);

  EXPECT_EQ(old_new_map[1], 0);
  EXPECT_EQ(old_new_map[3], 1);
  EXPECT_EQ(old_new_map[5], 2);

  auto builder = ut::ComputeGraphBuilder("test");
  auto node = builder.AddNode("test", DATA, 0, 1);

  ffts_plus_ops_kernel_builder_ptr->GenNewSubGraphTaskDef(*node.get(), task_def, task_def_new,
      new_old_map, old_new_map);
  domi::FftsPlusTaskDef *ffts_plus_task_def_new = task_def_new.mutable_ffts_plus_task();
  EXPECT_EQ(ffts_plus_task_def_new->ffts_plus_ctx_size(), 3);
  auto ctx_new = ffts_plus_task_def_new->mutable_ffts_plus_ctx(0);
  EXPECT_EQ(ctx_new->context_type(), RT_CTX_TYPE_AICORE);
  aic_ctx = ctx_new->mutable_aic_aiv_ctx();
  EXPECT_EQ(aic_ctx->successor_num(), 2);
  EXPECT_EQ(aic_ctx->successor_list(0), 1);
  EXPECT_EQ(aic_ctx->successor_list(1), 2);
  EXPECT_EQ(aic_ctx->pred_cnt(), 0);

  ctx_new = ffts_plus_task_def_new->mutable_ffts_plus_ctx(1);
  EXPECT_EQ(ctx_new->context_type(), RT_CTX_TYPE_PERSISTENT_CACHE);
  aic_ctx = ctx_new->mutable_aic_aiv_ctx();
  EXPECT_EQ(aic_ctx->successor_num(), 0);
  EXPECT_EQ(aic_ctx->pred_cnt(), 2);

  ctx_new = ffts_plus_task_def_new->mutable_ffts_plus_ctx(2);
  EXPECT_EQ(ctx_new->context_type(), RT_CTX_TYPE_AICORE);
  aic_ctx = ctx_new->mutable_aic_aiv_ctx();
  EXPECT_EQ(aic_ctx->successor_num(), 1);
  EXPECT_EQ(aic_ctx->successor_list(0), 1);
  EXPECT_EQ(aic_ctx->pred_cnt(), 1);
}