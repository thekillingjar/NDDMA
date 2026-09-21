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

#define private public
#define protected public
#include "common/sgt_slice_type.h"
#include "inc/ffts_log.h"
#include "inc/ffts_error_codes.h"
#include "inc/ffts_type.h"
#include "task_builder/fftsplus_task_builder.h"
#include "task_builder/thread_ctx/aic_aiv_manual_task_builder.h"
#include "task_builder/thread_ctx/mix_aic_aiv_manual_task_builder.h"
#include "graph/node.h"
#include "graph_builder_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"

using namespace std;
using namespace ffts;
using namespace ge;
using AICAIVTaskBuilderPtr = std::shared_ptr<AICAIVTaskBuilder>;
using MixAICAIVTaskBuilderPtr = std::shared_ptr<MixAICAIVTaskBuilder>;

class FFTSPlusTaskBuilderTest : public testing::Test {
 protected:
  void SetUp(){
    aic_aiv_task_builder_ptr = make_shared<AICAIVTaskBuilder>();
    mix_aic_aiv_task_builder_ptr = make_shared<MixAICAIVTaskBuilder>();
    slice_info_ptr = make_shared<ThreadSliceMap>();
    slice_info_ptr->thread_mode = 0;
  }
  void TearDown(){
  }
 public:
  AICAIVTaskBuilderPtr aic_aiv_task_builder_ptr = nullptr;
  MixAICAIVTaskBuilderPtr mix_aic_aiv_task_builder_ptr = nullptr;
  ThreadSliceMapPtr slice_info_ptr  = nullptr;
};

TEST_F(FFTSPlusTaskBuilderTest, FFTSPlusTaskBuilder_failed)
{
  domi::FftsPlusTaskDef *ffts_plus_task_def;
  domi::FftsPlusLabelCtxDef *pred_label_ctx;
  domi::FftsPlusLabelCtxDef **avl_label_context;
  uint32_t recursion_count = 10000;
  Status ret = aic_aiv_task_builder_ptr->GetFirstAvailableLabel(ffts_plus_task_def, pred_label_ctx, avl_label_context,
                                                                recursion_count);
  EXPECT_EQ(ffts::FAILED, ret);
}

TEST_F(FFTSPlusTaskBuilderTest, UpdateSuccList_succ_1)
{
  uint32_t succ_id = 1;
  uint32_t curr_id = 0;
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  size_t succ_num = 2;
  for (size_t i = 0; i < succ_num; i++) {
    domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
    ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_MIX_AIC);
    auto aic_aiv = ffts_plus_ctx_def->mutable_mix_aic_aiv_ctx();
    aic_aiv->set_successor_num(i + 1);
    aic_aiv->add_successor_list(i + 1);
  }

  Status ret = mix_aic_aiv_task_builder_ptr->UpdateSuccList(succ_id, curr_id, ffts_plus_task_def);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusTaskBuilderTest, UpdateSuccList_failed)
{
  uint32_t succ_id = 1;
  uint32_t curr_id = 0;
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  size_t succ_num = 2;
  for (size_t i = 0; i < succ_num; i++) {
    domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
    ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_LABEL);
    auto aic_aiv = ffts_plus_ctx_def->mutable_aic_aiv_ctx();
    aic_aiv->set_successor_num(i + 1);
    aic_aiv->add_successor_list(i + 1);
  }

  Status ret = mix_aic_aiv_task_builder_ptr->UpdateSuccList(succ_id, curr_id, ffts_plus_task_def);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusTaskBuilderTest, Update_Pre_Cnt)
{
  uint32_t curr_id = 0;
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AICORE);
  Status ret = mix_aic_aiv_task_builder_ptr->UpdatePreCnt(curr_id, ffts_plus_task_def, 1);
  EXPECT_EQ(ffts::SUCCESS, ret);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_NOTIFY_WAIT);
  ret = mix_aic_aiv_task_builder_ptr->UpdatePreCnt(curr_id, ffts_plus_task_def, 1);
  EXPECT_EQ(ffts::SUCCESS, ret);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_WRITE_VALUE);
  ret = mix_aic_aiv_task_builder_ptr->UpdatePreCnt(curr_id, ffts_plus_task_def, 1);
  EXPECT_EQ(ffts::SUCCESS, ret);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_SDMA);
  ret = mix_aic_aiv_task_builder_ptr->UpdatePreCnt(curr_id, ffts_plus_task_def, 1);
  EXPECT_EQ(ffts::SUCCESS, ret);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_INVALIDATE_DATA);
  ret = mix_aic_aiv_task_builder_ptr->UpdatePreCnt(curr_id, ffts_plus_task_def, 1);
  EXPECT_EQ(ffts::SUCCESS, ret);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_COND_SWITCH);
  ret = mix_aic_aiv_task_builder_ptr->UpdatePreCnt(curr_id, ffts_plus_task_def, 1);
  EXPECT_EQ(ffts::SUCCESS, ret);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_CASE_SWITCH);
  ret = mix_aic_aiv_task_builder_ptr->UpdatePreCnt(curr_id, ffts_plus_task_def, 1);
  EXPECT_EQ(ffts::SUCCESS, ret);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_WRITE_VALUE);
  ret = mix_aic_aiv_task_builder_ptr->UpdatePreCnt(curr_id, ffts_plus_task_def, 1);
  EXPECT_EQ(ffts::SUCCESS, ret);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AT_END);
  ret = mix_aic_aiv_task_builder_ptr->UpdatePreCnt(curr_id, ffts_plus_task_def, 1);
  EXPECT_EQ(ffts::SUCCESS, ret);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AT_START);
  ret = mix_aic_aiv_task_builder_ptr->UpdatePreCnt(curr_id, ffts_plus_task_def, 1);
  EXPECT_EQ(ffts::SUCCESS, ret);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_LABEL);
  ret = mix_aic_aiv_task_builder_ptr->UpdatePreCnt(curr_id, ffts_plus_task_def, 1);
  EXPECT_EQ(ffts::SUCCESS, ret);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_MIX_AIC);
  ret = mix_aic_aiv_task_builder_ptr->UpdatePreCnt(curr_id, ffts_plus_task_def, 1);
  EXPECT_EQ(ffts::SUCCESS, ret);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AICPU);
  ret = mix_aic_aiv_task_builder_ptr->UpdatePreCnt(curr_id, ffts_plus_task_def, 1);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusTaskBuilderTest, ReplaceSuccList)
{
  uint32_t curr_id = 0;
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();

  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AICORE);
  ffts_plus_ctx_def->mutable_aic_aiv_ctx()->add_successor_list(1);
  ffts_plus_ctx_def->mutable_aic_aiv_ctx()->set_successor_num(1);
  Status ret = mix_aic_aiv_task_builder_ptr->ReplaceSuccList(1, 2, curr_id, ffts_plus_task_def);
  EXPECT_EQ(ffts_plus_ctx_def->mutable_aic_aiv_ctx()->successor_list(0), 2);

  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_NOTIFY_WAIT);
  ffts_plus_ctx_def->mutable_notify_ctx()->add_successor_list(1);
  ffts_plus_ctx_def->mutable_notify_ctx()->set_successor_num(1);
  ret = mix_aic_aiv_task_builder_ptr->ReplaceSuccList(1, 2, curr_id, ffts_plus_task_def);
  EXPECT_EQ(ffts_plus_ctx_def->mutable_notify_ctx()->successor_list(0), 2);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_SDMA);
  ffts_plus_ctx_def->mutable_sdma_ctx()->add_successor_list(1);
  ffts_plus_ctx_def->mutable_sdma_ctx()->set_successor_num(1);
  ret = mix_aic_aiv_task_builder_ptr->ReplaceSuccList(1, 2, curr_id, ffts_plus_task_def);
  EXPECT_EQ(ffts_plus_ctx_def->mutable_sdma_ctx()->successor_list(0), 2);

  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_INVALIDATE_DATA);
  ffts_plus_ctx_def->mutable_data_ctx()->add_successor_list(1);
  ffts_plus_ctx_def->mutable_data_ctx()->set_successor_num(1);
  ret = mix_aic_aiv_task_builder_ptr->ReplaceSuccList(1, 2, curr_id, ffts_plus_task_def);
  EXPECT_EQ(ffts_plus_ctx_def->mutable_data_ctx()->successor_list(0), 2);

  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_WRITE_VALUE);
  ffts_plus_ctx_def->mutable_write_value_ctx()->add_successor_list(1);
  ffts_plus_ctx_def->mutable_write_value_ctx()->set_successor_num(1);
  ret = mix_aic_aiv_task_builder_ptr->ReplaceSuccList(1, 2, curr_id, ffts_plus_task_def);
  EXPECT_EQ(ffts_plus_ctx_def->mutable_write_value_ctx()->successor_list(0), 2);

  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AT_START);
  ffts_plus_ctx_def->mutable_at_start_ctx()->add_successor_list(1);
  ffts_plus_ctx_def->mutable_at_start_ctx()->set_successor_num(1);
  ret = mix_aic_aiv_task_builder_ptr->ReplaceSuccList(1, 2, curr_id, ffts_plus_task_def);
  EXPECT_EQ(ffts_plus_ctx_def->mutable_at_start_ctx()->successor_list(0), 2);

  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_LABEL);
  ffts_plus_ctx_def->mutable_label_ctx()->add_successor_list(1);
  ffts_plus_ctx_def->mutable_label_ctx()->set_successor_num(1);
  ret = mix_aic_aiv_task_builder_ptr->ReplaceSuccList(1, 2, curr_id, ffts_plus_task_def);
  EXPECT_EQ(ffts_plus_ctx_def->mutable_label_ctx()->successor_list(0), 2);

  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_MIX_AIC);
  ffts_plus_ctx_def->mutable_mix_aic_aiv_ctx()->add_successor_list(1);
  ffts_plus_ctx_def->mutable_mix_aic_aiv_ctx()->set_successor_num(1);
  ret = mix_aic_aiv_task_builder_ptr->ReplaceSuccList(1, 2, curr_id, ffts_plus_task_def);
  EXPECT_EQ(ffts_plus_ctx_def->mutable_mix_aic_aiv_ctx()->successor_list(0), 2);

  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_CASE_SWITCH);
  ffts_plus_ctx_def->mutable_case_switch_ctx()->add_successor_list(1);
  ffts_plus_ctx_def->mutable_case_switch_ctx()->set_successor_num(1);
  ret = mix_aic_aiv_task_builder_ptr->ReplaceSuccList(1, 2, curr_id, ffts_plus_task_def);
  EXPECT_EQ(ffts_plus_ctx_def->mutable_case_switch_ctx()->successor_list(0), 2);

  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_DSA);
  ffts_plus_ctx_def->mutable_dsa_ctx()->add_successor_list(1);
  ffts_plus_ctx_def->mutable_dsa_ctx()->set_successor_num(1);
  ret = mix_aic_aiv_task_builder_ptr->ReplaceSuccList(1, 2, curr_id, ffts_plus_task_def);
  EXPECT_EQ(ffts_plus_ctx_def->mutable_dsa_ctx()->successor_list(0), 2);
  // fail
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AICPU);
  ffts_plus_ctx_def->mutable_aicpu_ctx()->add_successor_list(3);
  ffts_plus_ctx_def->mutable_aicpu_ctx()->set_successor_num(1);
  ret = mix_aic_aiv_task_builder_ptr->ReplaceSuccList(1, 2, curr_id, ffts_plus_task_def);
  EXPECT_EQ(ffts_plus_ctx_def->mutable_aicpu_ctx()->successor_list(0), 3);
}

TEST_F(FFTSPlusTaskBuilderTest, FillManualCustomersInfo)
{
  FftsPlusComCtx_t sub_ffts_plus_context_elem;

  ComputeGraphPtr fused_graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_0 = std::make_shared<OpDesc>("add", "Add");
  OpDescPtr op_desc_1 = std::make_shared<OpDesc>("PhonyConcat", "PhonyConcat");
  OpDescPtr op_desc_2 = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  op_desc_0->AddInputDesc(out_desc);
  op_desc_0->AddOutputDesc(out_desc);
  op_desc_1->AddInputDesc(out_desc);
  op_desc_1->AddOutputDesc(out_desc);
  op_desc_2->AddInputDesc(out_desc);
  op_desc_2->AddOutputDesc(out_desc);
  NodePtr node_add = fused_graph->AddNode(op_desc_0);
  NodePtr node_phony = fused_graph->AddNode(op_desc_1);
  NodePtr node_relu = fused_graph->AddNode(op_desc_2);
  GraphUtils::AddEdge(node_add->GetOutDataAnchor(0), node_phony->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_phony->GetOutDataAnchor(0), node_relu->GetInDataAnchor(0));
  (void)ge::AttrUtils::SetBool(node_phony->GetOpDesc(), kTypeFFTSPlus, true);
  (void)ge::AttrUtils::SetInt(node_relu->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 0);
  (void)ge::AttrUtils::SetInt(node_relu->GetOpDesc(), kContextId, 1);

  aic_aiv_task_builder_ptr->FillManualCustomersInfo(node_add, sub_ffts_plus_context_elem);
  bool ret = sub_ffts_plus_context_elem.succ_list.size() != 0;
  EXPECT_EQ(true, ret);
}

TEST_F(FFTSPlusTaskBuilderTest, FillCustomersInfo_1)
{
  FftsPlusComCtx_t sub_ffts_plus_context = {0};
  auto builder = ut::ComputeGraphBuilder("test");
  auto input = builder.AddNode("test", "test", 0, 4);
  for (int i = 0; i < 4; i++) {
    string node_name = "conv2d";
    node_name += to_string(i);
    auto conv2d = builder.AddNode(node_name, "conv2d", 1, 0);
    ge::AttrUtils::SetInt(conv2d->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    (void)ge::AttrUtils::SetInt(conv2d->GetOpDesc(), kContextId, i);
    builder.AddDataEdge(input, i, conv2d, 0);
  }
  vector<FftsPlusComCtx_t> context_vec;
  ge::AttrUtils::SetInt(input->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
  (void)input->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, slice_info_ptr);
  Status ret = aic_aiv_task_builder_ptr->FillCustomersInfo(input, sub_ffts_plus_context, context_vec);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusTaskBuilderTest, FillCustomersInfo_2)
{
  FftsPlusComCtx_t sub_ffts_plus_context = {0};
  auto builder = ut::ComputeGraphBuilder("test");
  auto input = builder.AddNode("test", "test", 0, 2);
  auto concat = builder.AddNode(kTypePhonyConcat, kTypePhonyConcat, 1, 3);
  for (int i = 0; i < 3; i++) {
    string node_name = "test";
    node_name += to_string(i);
    auto test_node = builder.AddNode(node_name, "test", 1, 0);
    ge::AttrUtils::SetInt(test_node->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, i);
    if (i == 2) {
      (void)ge::AttrUtils::SetInt(test_node->GetOpDesc(), kContextId, i);
    }
    builder.AddDataEdge(concat, i, test_node, 0);
  }
  auto conv2d = builder.AddNode("conv2d", "conv2d", 1, 0);
  ge::AttrUtils::SetInt(conv2d->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 4);
  (void)ge::AttrUtils::SetInt(conv2d->GetOpDesc(), kContextId, 8);
  builder.AddDataEdge(input, 0, concat, 0);
  builder.AddDataEdge(input, 1, conv2d, 0);
  ge::AttrUtils::SetInt(input->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
  vector<FftsPlusComCtx_t> context_vec;
  (void)input->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, slice_info_ptr);
  Status ret = aic_aiv_task_builder_ptr->FillCustomersInfo(input, sub_ffts_plus_context, context_vec);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusTaskBuilderTest, FillCustomersInfo_3)
{
  FftsPlusComCtx_t sub_ffts_plus_context = {0};
  auto builder = ut::ComputeGraphBuilder("test");
  auto input = builder.AddNode("test", "test", 0, 2);
  auto concat = builder.AddNode(kTypePhonyConcat, kTypePhonyConcat, 1, 3);
  for (int i = 0; i < 3; i++) {
    string node_name = "test";
    node_name += to_string(i);
    auto test_node = builder.AddNode(node_name, "test", 1, 0);
    ge::AttrUtils::SetInt(test_node->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    if (i == 2) {
      (void)ge::AttrUtils::SetInt(test_node->GetOpDesc(), kContextId, i);
    }
    builder.AddDataEdge(concat, i, test_node, 0);
  }
  auto conv2d = builder.AddNode("conv2d", "conv2d", 1, 0);
  ge::AttrUtils::SetInt(conv2d->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 4);
  (void)ge::AttrUtils::SetInt(conv2d->GetOpDesc(), kContextId, 8);
  builder.AddDataEdge(input, 0, concat, 0);
  builder.AddDataEdge(input, 1, conv2d, 0);
  ge::AttrUtils::SetInt(input->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
  ge::AttrUtils::SetInt(concat->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
  vector<FftsPlusComCtx_t> context_vec;
  (void)input->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, slice_info_ptr);
  Status ret = aic_aiv_task_builder_ptr->FillCustomersInfo(input, sub_ffts_plus_context, context_vec);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusTaskBuilderTest, FillCustomersInfo_4)
{
  FftsPlusComCtx_t sub_ffts_plus_context = {0};
  auto builder = ut::ComputeGraphBuilder("test");
  auto input = builder.AddNode("test", "test", 0, 2);
  auto concat = builder.AddNode(kTypePhonyConcat, kTypePhonyConcat, 1, 3);
  for (int i = 0; i < 3; i++) {
    string node_name = "test";
    node_name += to_string(i);
    auto test_node = builder.AddNode(node_name, "test", 1, 0);
    ge::AttrUtils::SetInt(test_node->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    if (i == 2) {
      (void)ge::AttrUtils::SetInt(test_node->GetOpDesc(), kContextId, i);
    }
    builder.AddDataEdge(concat, i, test_node, 0);
  }
  auto conv2d = builder.AddNode("conv2d", "conv2d", 1, 0);
  ge::AttrUtils::SetInt(conv2d->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 4);
  (void)ge::AttrUtils::SetInt(conv2d->GetOpDesc(), kContextId, 8);
  builder.AddDataEdge(input, 0, concat, 0);
  builder.AddDataEdge(input, 1, conv2d, 0);
  (void)ge::AttrUtils::SetBool(input->GetOpDesc(), kTypeFFTSPlus, true);
  ge::AttrUtils::SetInt(concat->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
  vector<FftsPlusComCtx_t> context_vec;
  (void)input->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, slice_info_ptr);
  Status ret = aic_aiv_task_builder_ptr->FillCustomersInfo(input, sub_ffts_plus_context, context_vec);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusTaskBuilderTest, FillProducersInfo1)
{
  FftsPlusComCtx_t sub_ffts_plus_context = {0};
  auto builder = ut::ComputeGraphBuilder("test");
  auto input = builder.AddNode("test", "test", 2, 0);
  auto concat = builder.AddNode(kTypePhonyConcat, kTypePhonyConcat, 3, 1);
  for (int i = 0; i < 3; i++) {
    string node_name = "test";
    node_name += to_string(i);
    auto test_node = builder.AddNode(node_name, "test", 0, 1);
    if (i == 2) {
      (void)ge::AttrUtils::SetInt(test_node->GetOpDesc(), kContextId, i);
    }
    builder.AddDataEdge(test_node, 0, concat, i);
  }
  auto conv2d = builder.AddNode("conv2d", "conv2d", 0, 1);
  (void)ge::AttrUtils::SetInt(conv2d->GetOpDesc(), kContextId, 8);
  builder.AddDataEdge(concat, 0, input, 0);
  builder.AddDataEdge(conv2d, 0, input, 1);
  (void)ge::AttrUtils::SetBool(input->GetOpDesc(), kTypeFFTSPlus, true);
  (void)input->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, slice_info_ptr);
  Status ret = aic_aiv_task_builder_ptr->FillProducersInfo(input, sub_ffts_plus_context);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusTaskBuilderTest, FillProducersInfo2)
{
  FftsPlusComCtx_t sub_ffts_plus_context = {0};
  auto builder = ut::ComputeGraphBuilder("test");
  auto input = builder.AddNode("test", "test", 2, 0);
  auto concat = builder.AddNode(kTypePhonyConcat, kTypePhonyConcat, 3, 1);
  for (int i = 0; i < 3; i++) {
    string node_name = "test";
    node_name += to_string(i);
    auto test_node = builder.AddNode(node_name, "test", 0, 1);
    (void)ge::AttrUtils::SetBool(input->GetOpDesc(), kTypeFFTSPlus, true);
    if (i == 2) {
      (void)ge::AttrUtils::SetInt(test_node->GetOpDesc(), kContextId, i);
    }
    builder.AddDataEdge(test_node, 0, concat, i);
  }
  auto conv2d = builder.AddNode("conv2d", "conv2d", 0, 1);
  (void)ge::AttrUtils::SetInt(conv2d->GetOpDesc(), kContextId, 8);
  (void)ge::AttrUtils::SetBool(conv2d->GetOpDesc(), kTypeFFTSPlus, true);
  builder.AddDataEdge(concat, 0, input, 0);
  builder.AddDataEdge(conv2d, 0, input, 1);
  (void)ge::AttrUtils::SetBool(input->GetOpDesc(), kTypeFFTSPlus, true);
  (void)input->GetOpDesc()->SetExtAttr(kAttrSgtStructInfo, slice_info_ptr);
  Status ret = aic_aiv_task_builder_ptr->FillProducersInfo(input, sub_ffts_plus_context);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FFTSPlusTaskBuilderTest, add_at_end_to_write_back_succ_list_suc)
{
  uint32_t at_end_ctx_id = 0;
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  size_t succ_num = 2;
  for (size_t i = 0; i < succ_num; i++) {
    domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
    if (i == 0) {
      ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AICORE);
      auto aic_aiv = ffts_plus_ctx_def->mutable_aic_aiv_ctx();
      aic_aiv->set_successor_num(1);
      aic_aiv->add_successor_list(1);
    } else {
      ffts_plus_ctx_def->set_context_type(RT_HW_CTX_TYPE_WRITEBACK_DATA);
      auto data_ctx = ffts_plus_ctx_def->mutable_data_ctx();
    }
  }
  domi::FftsPlusCtxDef* ffts_plus_ctx_end = ffts_plus_task_def->mutable_ffts_plus_ctx(0);
  bool already_add = FFTSPlusTaskBuilder::add_at_end_to_write_back_succ_list(at_end_ctx_id,
                                                                             ffts_plus_ctx_end->mutable_aic_aiv_ctx(),
                                                                             ffts_plus_task_def);
  EXPECT_EQ(true, already_add);
}

TEST_F(FFTSPlusTaskBuilderTest, FillContextData_failed)
{
  domi::FftsPlusAicAivCtxDef *aicore_ctx_def = nullptr;
  domi::FftsPlusAicAivCtxDef *aic_aiv_ctx_def = nullptr;
  domi::FftsPlusMixAicAivCtxDef *mix_aicore_ctx_def = nullptr;
  domi::FftsPlusMixAicAivCtxDef *mix_aic_aiv_ctx_def = nullptr;
  bool ret = true;
  aic_aiv_task_builder_ptr->FillContextData(aicore_ctx_def, aic_aiv_ctx_def);
  mix_aic_aiv_task_builder_ptr->FillContextData(mix_aicore_ctx_def, mix_aic_aiv_ctx_def);
  EXPECT_EQ(true, ret);
}
