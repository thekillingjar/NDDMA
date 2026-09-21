/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <map>
#include <memory>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "proto/om.pb.h"
#include "fe_llt_utils.h"
#define protected public
#define private public
#include "common/graph_comm.h"
#include "pass_manager.h"
#include "common/util/op_info_util.h"
#include "common/configuration.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/op_kernel_bin.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/ub_fusion/buffer_fusion.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "graph_optimizer/op_setter/op_setter.h"
#include "ops_store/ops_kernel_manager.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "../fusion_stub.hpp"
#undef protected
#undef private
using namespace std;
using namespace domi;
using namespace fe;
using namespace ge;

using OpSetterPtr = std::shared_ptr<OpSetter>;
class TBE_READ_SELECT_ELTWISE_FUSION_SLICE_INFO_UNITTEST : public testing::Test {
public:
  using AttrDefMap = ::google::protobuf::Map<::std::string, AttrDef>;

protected:
  static void SetUpTestCase() { std::cout << "UB fusion SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "UB fusion TearDown" << std::endl; }

  void SetUp()
  {
    std::map<std::string, std::string> options;
    fe_ops_kernel_info_store_ptr_ = make_shared<fe::FEOpsKernelInfoStore>();
    FEOpsStoreInfo tbe_custom {
            6,
            "tbe-custom",
            EN_IMPL_HW_TBE,
            GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_slice_op_info/slice_success",
            ""};
    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(tbe_custom);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();
    fe_ops_kernel_info_store_ptr_->Initialize(options);
  }

  virtual void TearDown() {}

  void SetPattern(ge::OpDescPtr opdef, string optype) {
    auto key_pattern = "_pattern";
    ge::AttrUtils::SetStr(opdef, key_pattern, optype);
  }

  void SetTvmType(ge::OpDescPtr opdef) {
    ge::AttrUtils::SetInt(opdef, ge::ATTR_NAME_IMPLY_TYPE,static_cast<int64_t>(domi::ImplyType::TVM));
  }

  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr_;

  /************************
   ReadSelect-->ElemWise
  *************************/
  // {"_fusion_op_slice_info":{"l1FusionEnable":1,"minTbeL1Space":0,"reduceMaps":[],"splitMaps":[{
  //   "inputList":[{"axis":[1],"headOverLap":[-1],"idx":0,"tailOverLap":[-1]}],"outputList":[{"axis":[1],"idx":0}]},
  //   {"inputList":[{"axis":[0],"headOverLap":[-1],"idx":0,"tailOverLap":[-1]}],"outputList":[{"axis":[0],"idx":0}]}]}}
  void BuildGraphForTbeReadSelecEltwisetFusionPassOKCase1(ComputeGraphPtr &graph) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
    OpDescPtr read_select = std::make_shared<OpDesc>("readselect", "ReadSelect");
    OpDescPtr eltwise = std::make_shared<OpDesc>("eltwise", "Eltwise");
    OpDescPtr end = std::make_shared<OpDesc>("end", "End");
    SetPattern(read_select, "read_select");
    SetPattern(eltwise, "ElemWise");
    SetTvmType(read_select);
    SetTvmType(eltwise);
    ge::AttrUtils::SetInt(read_select, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);
    ge::AttrUtils::SetInt(eltwise, FE_IMPLY_TYPE, EN_IMPL_HW_TBE);

    // add descriptor
    GeTensorDesc tensor_desc(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NC1HWC0,ge::DT_FLOAT16);
    tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);

    data->AddOutputDesc(tensor_desc);
    read_select->AddInputDesc(tensor_desc);
    read_select->AddOutputDesc(tensor_desc);
    eltwise->AddInputDesc(tensor_desc);
    eltwise->AddOutputDesc(tensor_desc);
    end->AddInputDesc(tensor_desc);
    end->AddOutputDesc(tensor_desc);

    NodePtr data_node = graph->AddNode(data);
    NodePtr read_select_node = graph->AddNode(read_select);
    NodePtr eltwise_node = graph->AddNode(eltwise);
    NodePtr end_node = graph->AddNode(end);

    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin+strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(read_select_node->GetName(), std::move(buffer));
    read_select_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), read_select_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(read_select_node->GetOutDataAnchor(0), eltwise_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(eltwise_node->GetOutDataAnchor(0), end_node->GetInDataAnchor(0));
  }
};

TEST_F(TBE_READ_SELECT_ELTWISE_FUSION_SLICE_INFO_UNITTEST, pass_case1) {
  ComputeGraphPtr graph_out = std::make_shared<ComputeGraph>("test");
  BuildGraphForTbeReadSelecEltwisetFusionPassOKCase1(graph_out);
  graph_out->TopologicalSorting();
  graph_out->Dump();
  // set op slice info for each op
  OpSetterPtr op_setter_ptr = std::make_shared<OpSetter>(AI_CORE_NAME);
  Status ret = op_setter_ptr->SetOpInfo(*(graph_out.get()));
  EXPECT_EQ(fe::SUCCESS, ret);
  std::shared_ptr<GraphComm> graph_comm_ptr = std::make_shared<GraphComm>("engineName");
  graph_comm_ptr->Initialize();
  std::shared_ptr<FusionPriorityManager> fusion_priority_mgr_ptr =
          std::make_shared<FusionPriorityManager>("engineName", nullptr);
  std::vector<BufferFusionInfo> sorted_buffer_fusion_vec = SortedBufferFusionFun();
  fusion_priority_mgr_ptr->sorted_buffer_fusion_map_[FusionPriorityManager::GetCurrentHashedKey()] = sorted_buffer_fusion_vec;
  std::shared_ptr<BufferFusion> sub_graph_optimizer_ptr =
          std::make_shared<BufferFusion>(graph_comm_ptr, fusion_priority_mgr_ptr, nullptr);
  uint32_t id = 0;
  cerr << endl;
  cerr << "TBE_READ_SELECT_ELTWISE_FUSION_SLICE_INFO_UNITTEST::pass_case1 UB fusion before" << endl;
  for (auto &node : graph_out->GetDirectNode()) {
    cerr << "id:" << id << endl;
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType() << endl;
    if (AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    }
    id++;
  }
  sub_graph_optimizer_ptr->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  sub_graph_optimizer_ptr->MatchFusionPatternFromGraph(*graph_out);
  // create fused Graph, and merge matched sub-graphs into fusion ops
  sub_graph_optimizer_ptr->BuildFusionGraph(*graph_out);

  string fusion_op_slice_info;
  string op_slice_info;
  int find = 0;
  cerr << endl;
  cerr << "TBE_READ_SELECT_ELTWISE_FUSION_SLICE_INFO_UNITTEST::pass_case1 UB fusion result" << endl;
  for (auto &node : graph_out->GetDirectNode()) {
    cerr  << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType() << endl;
    if (node->GetOpDesc()->GetType() == "ReadSelect" && node->GetName() == "readselecteltwise") {
      AttrUtils::GetStr(node->GetOpDesc(), OP_SLICE_INFO, op_slice_info);
      cerr << "op slice info is :   " << endl << op_slice_info << endl;
      AttrUtils::GetStr(node->GetOpDesc(), FUSION_OP_SLICE_INFO, fusion_op_slice_info);
      cerr << "fusion op slice info is :   " << endl << fusion_op_slice_info << endl;
      find = 1;
    }
  }
  EXPECT_EQ(find, 1);
}
