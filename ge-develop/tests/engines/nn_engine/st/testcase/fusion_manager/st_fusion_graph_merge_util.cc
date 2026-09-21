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
#include "common/lxfusion_json_util.h"
#include "common/configuration.h"
#include "common/fe_inner_attr_define.h"
#include "common/util/constants.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/op_kernel_bin.h"
#include "graph/ge_local_context.h"
#include "graph/utils/graph_utils.h"
#include "graph_optimizer/ub_fusion/fusion_graph_merge/fusion_graph_merge.h"
#include "graph_optimizer/ub_fusion/automatic_buffer_fusion.h"
#include "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_optimizer.h"
#include "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_fusion_comm/l2_fusion_comm.h"
#include "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_fusion_parser/l2_fusion_parser.h"
#include "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_fusion_allocation/l2_fusion_allocation.h"

#undef private
#undef protected

using namespace std;
using namespace fe;
using namespace ge;

class fusion_graph_merge_util_st : public testing::Test {
public:
protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(fusion_graph_merge_util_st, set_l2_task_info_to_fusion_op) {
  auto graph_common = std::make_shared<GraphComm>("engineName");
  graph_common->Initialize();
  auto fusion_graph_merge_ptr =
      std::make_shared<FusionGraphMerge>("fusion_scope", graph_common);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::OpDescPtr node1_op = std::make_shared<ge::OpDesc>("conv0","conv");
  ge::NodePtr node1 = std::make_shared<ge::Node>(node1_op, graph);
  Status ret = fusion_graph_merge_ptr->SetL2TaskInfoToFusionOp(node1);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(fusion_graph_merge_util_st, fuse_two_nodes) {
  std::shared_ptr<AutomaticBufferFusion> auto_buffer_fusion_ptr = std::make_shared<AutomaticBufferFusion>(nullptr);
  ComputeGraphPtr owner_graph = std::make_shared<ComputeGraph>("test");
  ge::OpDescPtr node1_op = std::make_shared<ge::OpDesc>("conv0","conv");
  ge::OpDescPtr node2_op = std::make_shared<ge::OpDesc>("conv1","conv");
  ge::NodePtr node1 = std::make_shared<ge::Node>(node1_op, owner_graph);
  ge::NodePtr node2 = std::make_shared<ge::Node>(node1_op, owner_graph);
  int64_t producer_scope_id = 0;
  int64_t consumer_scope_id = 0;
  Status ret = auto_buffer_fusion_ptr->FuseTwoNodes(node1, node2, producer_scope_id, consumer_scope_id);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(fusion_graph_merge_util_st, test_allocdata_standing_data_special)
{
    L2FusionAllocation l2fusionAlloc;
    L2BufferInfo l2;
    std::map<uint64_t, std::size_t> count_map;
    std::vector<OpL2DataInfo> datas_map;
    TensorL2AllocMap standing_alloc_data;
    TensorL2DataMap converge_data;
    OpL2AllocMap alloc_map;
    uint64_t max_page = 63;
    uint32_t data_in_l2_id = 1;
    int64_t page_size = 1;
    int32_t page_num_left = 0;
    Status status = l2fusionAlloc.AllocateStandingDataSpecial(page_size, count_map, standing_alloc_data,
                                                              data_in_l2_id,  page_num_left);
    EXPECT_EQ(status, fe::SUCCESS);
}

TEST_F(fusion_graph_merge_util_st, set_multi_kernel_output_offsets) {
  std::vector<int64_t> buffer_fusion_output_offset;
  std::vector<int64_t> save_pre_output_offset;
  buffer_fusion_output_offset.push_back(1);

  auto graph_common = std::make_shared<GraphComm>("engineName");
  graph_common->Initialize();
  auto fusion_graph_merge_ptr =
      std::make_shared<FusionGraphMerge>(SCOPE_ID_ATTR, graph_common);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
  std::int64_t fusion_anchor_idx = 0;
  ge::OpDescPtr op_desc_src = make_shared<OpDesc>("sourceNode", "Conv");
  ge::OpDescPtr op_desc_fus = make_shared<OpDesc>("fusionNode", "Conv2D");
  ge::GeTensorDesc tensor_desc =
  ge::GeTensorDesc(ge::GeShape({4, 4}), FORMAT_ND, DT_FLOAT16);
  ge::AttrUtils::SetInt(tensor_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX,0);
  ge::AttrUtils::SetListInt(op_desc_fus, ge::ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, buffer_fusion_output_offset);
  op_desc_fus->AddOutputDesc(tensor_desc);
  op_desc_src->AddOutputDesc(tensor_desc);
  fusion_graph_merge_ptr->SetMultiKernelOutPutOffsets(op_desc_src, 2,
                                                      op_desc_fus, save_pre_output_offset);
  ge::AttrUtils::GetListInt(op_desc_fus, ge::ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, buffer_fusion_output_offset);
  EXPECT_EQ(buffer_fusion_output_offset[1], 0);
}

TEST_F(fusion_graph_merge_util_st, update_output_ref_port_index) {
  std::vector<int32_t> ref_port_index = {1};

  auto graph_common = std::make_shared<GraphComm>("engineName");
  graph_common->Initialize();
  auto fusion_graph_merge_ptr =
      std::make_shared<FusionGraphMerge>(SCOPE_ID_ATTR, graph_common);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
  std::int64_t fusion_anchor_idx = 0;
  ge::OpDescPtr op_desc_src = make_shared<OpDesc>("sourceNode", "Conv");
  ge::OpDescPtr op_desc_fus = make_shared<OpDesc>("fusionNode", "Conv2D");
  ge::GeTensorDesc tensor_desc_fus =
  ge::GeTensorDesc(ge::GeShape({4, 4}), FORMAT_ND, DT_FLOAT16);
  ge::GeTensorDesc tensor_desc_src =
  ge::GeTensorDesc(ge::GeShape({4, 4}), FORMAT_ND, DT_FLOAT16);
  ge::AttrUtils::SetListInt(tensor_desc_src, "ref_port_index", ref_port_index);
  op_desc_fus->AddInputDesc(tensor_desc_fus);
  op_desc_fus->AddInputDesc(tensor_desc_fus);
  op_desc_fus->AddInputDesc(tensor_desc_fus);
  ge::AttrUtils::SetBool(op_desc_fus, ge::ATTR_NAME_REFERENCE, true);

  op_desc_src->AddOutputDesc(tensor_desc_src);
  fusion_graph_merge_ptr->UpdateOutputRefPortIndex(op_desc_src, op_desc_fus, 1);
  op_desc_fus->AddOutputDesc(op_desc_src->GetOutputDesc(0));
  fusion_graph_merge_ptr->UpdateFusionOpRefPortIndex(op_desc_fus);

  ge::AttrUtils::GetListInt(op_desc_fus->MutableOutputDesc(0), "ref_port_index", ref_port_index);
  std::cout << "ref_port_index: " << ref_port_index[0] << std::endl;
  EXPECT_EQ(ref_port_index[0], 2);
}

TEST_F(fusion_graph_merge_util_st, calc_pass_strided_outsize) {
  auto graph_common = std::make_shared<GraphComm>("engineName");
  graph_common->Initialize();
  auto fusion_graph_merge_ptr =
          std::make_shared<FusionGraphMerge>(SCOPE_ID_ATTR, graph_common);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
  vector<pair<uint32_t, ge::NodePtr>> src_op_out_index_in_fus_op;
  ge::OpDescPtr op_desc_src = make_shared<OpDesc>("sourceNode", "Conv");
  ge::GeTensorDesc tensor_desc_src = ge::GeTensorDesc(ge::GeShape({1, 32, 17, 17}), FORMAT_NCHW, DT_FLOAT16);
  op_desc_src->AddOutputDesc(tensor_desc_src);
  ge::NodePtr src_node = graph->AddNode(op_desc_src);
  ToOpStructPtr optimize_info = std::make_shared<ToOpStruct_t>();
  optimize_info->slice_output_shape = {{1, 32, 17, 17}};
  SetStridedInfoToNode(op_desc_src, optimize_info, kConcatCOptimizeInfoPtr, kConcatCOptimizeInfoStr);
  ge::OpDescPtr op_desc_fus = make_shared<OpDesc>("fusionNode", "Conv2D");
  op_desc_fus->AddOutputDesc(tensor_desc_src);
  ge::NodePtr dst_node = graph->AddNode(op_desc_fus);
  std::vector<ge::NodePtr> nodevec = {src_node};
  src_op_out_index_in_fus_op.push_back(std::make_pair(0, src_node));
  fusion_graph_merge_ptr->CalcPassStridedOutSize(dst_node, nodevec, src_op_out_index_in_fus_op);
  int64_t tensor_size = 0;
  (void)ge::AttrUtils::GetInt(dst_node->GetOpDesc()->MutableOutputDesc(0), "_special_output_size", tensor_size);
  EXPECT_EQ(tensor_size, 18496);
}

TEST_F(fusion_graph_merge_util_st, calc_pass_strided_outsize2) {
  auto graph_common = std::make_shared<GraphComm>("engineName");
  graph_common->Initialize();
  auto fusion_graph_merge_ptr =
          std::make_shared<FusionGraphMerge>(SCOPE_ID_ATTR, graph_common);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
  vector<pair<uint32_t, ge::NodePtr>> src_op_out_index_in_fus_op;
  ge::OpDescPtr op_desc_src = make_shared<OpDesc>("sourceNode", "Conv");
  ge::GeTensorDesc tensor_desc_src = ge::GeTensorDesc(ge::GeShape({1, 32, 17, 17}), FORMAT_NCHW, DT_FLOAT16);
  ge::GeTensorDesc tensor_desc_src2 = ge::GeTensorDesc(ge::GeShape({1, 32, 17, 17}), FORMAT_NCHW, DT_INT8);

  op_desc_src->AddOutputDesc(tensor_desc_src);
  op_desc_src->AddOutputDesc(tensor_desc_src2);
  ge::NodePtr src_node = graph->AddNode(op_desc_src);
  ToOpStructPtr optimize_info = std::make_shared<ToOpStruct_t>();
  optimize_info->slice_output_shape = {{1, 32, 17, 17}};
  SetStridedInfoToNode(op_desc_src, optimize_info, kConcatCOptimizeInfoPtr, kConcatCOptimizeInfoStr);
  ge::OpDescPtr op_desc_fus = make_shared<OpDesc>("fusionNode", "Conv2D");
  op_desc_fus->AddOutputDesc(tensor_desc_src);
  op_desc_fus->AddOutputDesc(tensor_desc_src2);
  ge::NodePtr dst_node = graph->AddNode(op_desc_fus);
  std::vector<ge::NodePtr> nodevec = {src_node};
  src_op_out_index_in_fus_op.push_back(std::make_pair(0, src_node));
  src_op_out_index_in_fus_op.push_back(std::make_pair(1, src_node));
  fusion_graph_merge_ptr->CalcPassStridedOutSize(dst_node, nodevec, src_op_out_index_in_fus_op);
  int64_t tensor_size = 0;
  int64_t tensor_size2 = 0;
  (void)ge::AttrUtils::GetInt(dst_node->GetOpDesc()->MutableOutputDesc(0), "_special_output_size", tensor_size);
  (void)ge::AttrUtils::GetInt(dst_node->GetOpDesc()->MutableOutputDesc(1), "_special_output_size", tensor_size2);
  EXPECT_EQ(tensor_size, 18496);
  EXPECT_EQ(tensor_size2, 9248);
}

TEST_F(fusion_graph_merge_util_st, calc_single_op_pass_strided_outsize) {
  auto graph_common = std::make_shared<GraphComm>("engineName");
  graph_common->Initialize();
  auto fusion_graph_merge_ptr =
          std::make_shared<FusionGraphMerge>(SCOPE_ID_ATTR, graph_common);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");
  ge::OpDescPtr op_desc_src = make_shared<OpDesc>("sourceNode", "Conv");
  ge::GeTensorDesc tensor_desc_src = ge::GeTensorDesc(ge::GeShape({1, 32, 17, 17}), FORMAT_NCHW, DT_FLOAT16);
  op_desc_src->AddOutputDesc(tensor_desc_src);
  ge::NodePtr src_node = graph->AddNode(op_desc_src);
  ToOpStructPtr optimize_info = std::make_shared<ToOpStruct_t>();
  optimize_info->slice_output_shape = {{1, 32, 17, 17}};
  SetStridedInfoToNode(op_desc_src, optimize_info, kConcatCOptimizeInfoPtr, kConcatCOptimizeInfoStr);
  ge::AttrUtils::SetInt(op_desc_src, SCOPE_ID_ATTR, -1);
  std::vector<ge::NodePtr> nodevec = {src_node};
  std::map<std::string, std::string> options;
  options.insert(std::pair<std::string, std::string>("ge.buildMode", "tuning"));
  options.insert(std::pair<std::string, std::string>("ge.buildStep", "after_ub_match"));
  ge::GetThreadLocalContext().SetGlobalOption(options);
  fusion_graph_merge_ptr->CalcSingleOpStridedSize(*graph.get());
  int64_t tensor_size = 0;
  (void)ge::AttrUtils::GetInt(src_node->GetOpDesc()->MutableOutputDesc(0), "_special_output_size", tensor_size);
  EXPECT_EQ(tensor_size, 18496);
}
