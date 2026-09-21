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
#include "task_builder/fftsplus_ops_kernel_builder.h"
#undef private
#undef protected

using namespace std;
using namespace ffts;
using namespace ge;

using DataTaskBuilderPtr = shared_ptr<DataTaskBuilder>;
using OutTaskBuilderPtr = shared_ptr<OutTaskBuilder>;
class DataTaskBUildeUTEST : public testing::Test
{
protected:
  void SetUp()
  {
    data_task_builder_ptr_ = make_shared<DataTaskBuilder>();
    out_manual_task_builder_ptr_ = make_shared<OutTaskBuilder>();
    ffts_plus_def_ptr_ = make_shared<domi::FftsPlusCtxDef>();
  }
  void TearDown() {
  }
public:
  DataTaskBuilderPtr data_task_builder_ptr_;
  OutTaskBuilderPtr out_manual_task_builder_ptr_;
  std::shared_ptr<domi::FftsPlusCtxDef> ffts_plus_def_ptr_;
  NodePtr node_{ nullptr };
};

TEST_F(DataTaskBUildeUTEST, UpdateSrcSlotAndPfBm_suc) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_0 = std::make_shared<OpDesc>("add", "Add");

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
  uint32_t context_id = 0;
  Status ret = data_task_builder_ptr_->UpdateSrcSlotAndPfBm(ffts_plus_task_def, context_id);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(DataTaskBUildeUTEST, UpdateSrcSlotAndPfBm_failed) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_0 = std::make_shared<OpDesc>("add", "Add");

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
  uint32_t context_id = 0;
  Status ret = data_task_builder_ptr_->UpdateSrcSlotAndPfBm(ffts_plus_task_def, context_id);
  EXPECT_EQ(ffts::FAILED, ret);
}

TEST_F(DataTaskBUildeUTEST, GetSuccessorContextId_failed) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_0 = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  op_desc_0->AddInputDesc(out_desc);
  op_desc_0->AddOutputDesc(out_desc);
  NodePtr node_add = graph->AddNode(op_desc_0);
  uint32_t out_anchor_index;
  std::vector<uint32_t> succ_list;
  uint32_t cons_cnt;
  Status ret = data_task_builder_ptr_->GetSuccessorContextId(out_anchor_index, node_add, succ_list, cons_cnt);
  EXPECT_EQ(ffts::FAILED, ret);
}

TEST_F(DataTaskBUildeUTEST, GetSuccessorContextId_success1) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_0 = std::make_shared<OpDesc>("add", "Add");
  OpDescPtr op_desc_1 = std::make_shared<OpDesc>("relu", "relu");
  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  op_desc_0->AddInputDesc(out_desc);
  op_desc_0->AddOutputDesc(out_desc);
  op_desc_1->AddInputDesc(out_desc);
  op_desc_1->AddOutputDesc(out_desc);
  NodePtr node_add = graph->AddNode(op_desc_0);
  NodePtr node_relu = graph->AddNode(op_desc_1);
  GraphUtils::AddEdge(node_add->GetOutDataAnchor(0), node_relu->GetInDataAnchor(0));
  uint32_t out_anchor_index = 0;
  std::vector<uint32_t> succ_list;
  uint32_t cons_cnt = 0;
  Status ret = data_task_builder_ptr_->GetSuccessorContextId(out_anchor_index, node_add, succ_list, cons_cnt);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(DataTaskBUildeUTEST, GetSuccessorContextId_success2) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_0 = std::make_shared<OpDesc>("add", "Add");
  OpDescPtr op_desc_1 = std::make_shared<OpDesc>("phony_concat", "PhonyConcat");
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
  NodePtr node_add = graph->AddNode(op_desc_0);
  NodePtr node_phony_concat = graph->AddNode(op_desc_1);
  NodePtr node_relu = graph->AddNode(op_desc_2);
  GraphUtils::AddEdge(node_add->GetOutDataAnchor(0), node_phony_concat->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_phony_concat->GetOutDataAnchor(0), node_relu->GetInDataAnchor(0));
  uint32_t out_anchor_index = 0;
  std::vector<uint32_t> succ_list;
  uint32_t cons_cnt = 0;
  Status ret = data_task_builder_ptr_->GetSuccessorContextId(out_anchor_index, node_add, succ_list, cons_cnt);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(DataTaskBUildeUTEST, GetSuccessorContextId_success3) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_0 = std::make_shared<OpDesc>("add", "Add");
  OpDescPtr op_desc_1 = std::make_shared<OpDesc>("phony_concat", "PhonyConcat");
  OpDescPtr op_desc_2 = std::make_shared<OpDesc>("phony_concat2", "PhonyConcat");
  OpDescPtr op_desc_3 = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  op_desc_0->AddInputDesc(out_desc);
  op_desc_0->AddOutputDesc(out_desc);
  op_desc_1->AddInputDesc(out_desc);
  op_desc_1->AddOutputDesc(out_desc);
  op_desc_2->AddInputDesc(out_desc);
  op_desc_2->AddOutputDesc(out_desc);
  op_desc_3->AddInputDesc(out_desc);
  op_desc_3->AddOutputDesc(out_desc);
  NodePtr node_add = graph->AddNode(op_desc_0);
  NodePtr node_phony_concat = graph->AddNode(op_desc_1);
  NodePtr node_phony_concat2 = graph->AddNode(op_desc_2);
  NodePtr node_relu = graph->AddNode(op_desc_3);
  GraphUtils::AddEdge(node_add->GetOutDataAnchor(0), node_phony_concat->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_phony_concat->GetOutDataAnchor(0), node_phony_concat2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_phony_concat2->GetOutDataAnchor(0), node_relu->GetInDataAnchor(0));
  uint32_t out_anchor_index = 0;
  std::vector<uint32_t> succ_list;
  uint32_t cons_cnt = 0;
  Status ret = out_manual_task_builder_ptr_->GetSuccessorContextId(out_anchor_index, node_add, succ_list, cons_cnt);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(DataTaskBUildeUTEST, GetSuccessorContextId_success4) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_0 = std::make_shared<OpDesc>("add", "Add");
  OpDescPtr op_desc_1 = std::make_shared<OpDesc>("phony_concat", "PhonyConcat");
  OpDescPtr op_desc_2 = std::make_shared<OpDesc>("phony_concat2", "PhonyConcat");
  OpDescPtr op_desc_3 = std::make_shared<OpDesc>("relu", "Relu");
  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  op_desc_0->AddInputDesc(out_desc);
  op_desc_0->AddOutputDesc(out_desc);
  op_desc_1->AddInputDesc(out_desc);
  op_desc_1->AddOutputDesc(out_desc);
  op_desc_2->AddInputDesc(out_desc);
  op_desc_2->AddOutputDesc(out_desc);
  op_desc_3->AddInputDesc(out_desc);
  op_desc_3->AddOutputDesc(out_desc);
  (void)ge::AttrUtils::SetInt(op_desc_3, kContextId, 1);
  NodePtr node_add = graph->AddNode(op_desc_0);
  NodePtr node_phony_concat = graph->AddNode(op_desc_1);
  NodePtr node_phony_concat2 = graph->AddNode(op_desc_2);
  NodePtr node_relu = graph->AddNode(op_desc_3);
  GraphUtils::AddEdge(node_add->GetOutDataAnchor(0), node_phony_concat->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_phony_concat->GetOutDataAnchor(0), node_phony_concat2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_phony_concat2->GetOutDataAnchor(0), node_relu->GetInDataAnchor(0));
  uint32_t out_anchor_index = 0;
  std::vector<uint32_t> succ_list;
  uint32_t cons_cnt = 0;
  Status ret = out_manual_task_builder_ptr_->GetSuccessorContextId(out_anchor_index, node_add, succ_list, cons_cnt);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(DataTaskBUildeUTEST, GetIndices_scalar_test) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_0 = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim;
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  op_desc_0->AddInputDesc(out_desc);
  op_desc_0->AddOutputDesc(out_desc);
  NodePtr node_add = graph->AddNode(op_desc_0);
  data_task_builder_ptr_->operation_ = CACHE_OPERATION::PREFETCH;
  auto ret = data_task_builder_ptr_->GetIndices(node_add);
  EXPECT_EQ(ret.size(), 0);
  data_task_builder_ptr_->operation_ = CACHE_OPERATION::INVALIDATE;
  ret = data_task_builder_ptr_->GetIndices(node_add);
  EXPECT_EQ(ret.size(), 0);
}
