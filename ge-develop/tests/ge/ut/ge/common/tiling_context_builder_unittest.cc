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
#include "graph/utils/math_util.h"
#include "exe_graph/lowering/tiling_context_builder.h"
#include "exe_graph/lowering/device_tiling_context_builder.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "exe_graph/runtime/atomic_clean_tiling_context.h"
#include "common/share_graph.h"
#include "platform/platform_infos_def.h"
#include "framework/common/util.h"

namespace gert {
class TilingContextBuilderUT : public testing::Test {};
namespace {
struct DummyTilingParams {
  int64_t x;
  int64_t y;
};
struct DummyCompileInfo {
  int64_t a;
  int64_t b;
  std::vector<int64_t> c;
};
ge::Status AddDataNodeForAtomic(ge::ComputeGraphPtr &graph, ge::NodePtr &clean_node, size_t output_size) {
  // add data node for workspace
  auto workspace_data_op_desc = std::make_shared<ge::OpDesc>(clean_node->GetName() + "_Data_0", "Data");
  GE_CHECK_NOTNULL(workspace_data_op_desc);
  if (workspace_data_op_desc->AddOutputDesc(ge::GeTensorDesc()) != ge::SUCCESS) {
    GELOGE(ge::FAILED, "workspace_data_op_desc add output desc failed");
    return ge::FAILED;
  }
  auto workspace_data_node = graph->AddNode(workspace_data_op_desc);
  GE_CHECK_NOTNULL(workspace_data_node);
  auto ret = ge::GraphUtils::AddEdge(workspace_data_node->GetOutDataAnchor(0), clean_node->GetInDataAnchor(0));
  if (ret != ge::SUCCESS) {
    GELOGE(ge::FAILED, "add edge between [%s] and [%s] failed", workspace_data_node->GetName().c_str(),
           clean_node->GetName().c_str());
    return ge::FAILED;
  }

  // add data node for output
  for (size_t i = 0U; i < output_size; ++i) {
    auto data_op_desc = std::make_shared<ge::OpDesc>(clean_node->GetName() + "_Data_" + std::to_string(i + 1), "Data");
    GE_CHECK_NOTNULL(data_op_desc);
    if (data_op_desc->AddOutputDesc(ge::GeTensorDesc()) != ge::SUCCESS) {
      GELOGE(ge::FAILED, "data_op_desc add output desc failed, i = %zu", i);
      return ge::FAILED;
    }
    auto data_node = graph->AddNode(data_op_desc);
    GE_CHECK_NOTNULL(data_node);
    ret = ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), clean_node->GetInDataAnchor(i + 1));
    if (ret != ge::SUCCESS) {
      GELOGE(ge::FAILED, "add edge between [%s] and [%s] failed", data_node->GetName().c_str(),
             clean_node->GetName().c_str());
      return ge::FAILED;
    }
  }
  return ge::SUCCESS;
}

ge::NodePtr BuildAtomicNode(ge::ComputeGraphPtr &graph) {
  std::vector<int64_t> workspace_indexes = {1,2};
  std::vector<int64_t> outputs_indexes = {0,2};

  auto atomic_op_desc = std::make_shared<ge::OpDesc>("AtomicClean", "DynamicAtomicAddrClean");

  atomic_op_desc->AppendIrInput("workspace", ge::kIrInputRequired);
  atomic_op_desc->AppendIrInput("output", ge::kIrInputDynamic);

  atomic_op_desc->AddInputDesc("workspace", ge::GeTensorDesc());
  for (size_t i = 0U; i < outputs_indexes.size(); ++i) {
    atomic_op_desc->AddInputDesc("output" + std::to_string(i + 1), ge::GeTensorDesc());
  }
  if (!ge::AttrUtils::SetListInt(atomic_op_desc, "WorkspaceIndexes", workspace_indexes)) {
    return nullptr;
  }
  auto clean_node = graph->AddNode(atomic_op_desc);
  if (clean_node == nullptr) {
    GELOGE(ge::FAILED, "add node failed");
    return nullptr;
  }
  if (AddDataNodeForAtomic(graph, clean_node, outputs_indexes.size()) != ge::SUCCESS) {
    GELOGE(ge::FAILED, "add data node for atomic clean node failed, outputs_indexes size = %zu",
           outputs_indexes.size());
    return nullptr;
  }
  return clean_node;
}
} // namespace

// 校验 input/output shape，compile info, tiling相关的输出可以正常地设置和读取
TEST_F(TilingContextBuilderUT, BuildTilingContextHolderSuccess) {
  auto graph = ShareGraph::ConcatV2ConstDependencyGraph();
  // set origin shape not equal with storage shape
  auto concatv2_node = graph->FindNode("concatv2");
  std::vector<int64_t> origin_shape_dims = {1, 3, 4, 5};
  concatv2_node->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(ge::GeShape(origin_shape_dims));
  auto tiling_data = gert::TilingData::CreateCap(1024);
  auto workspace_size = gert::ContinuousVector::Create<size_t>(16);
  DummyCompileInfo compile_info_holder = {10, 200, {10, 20, 30}};
  fe::PlatFormInfos platform_infos;
  auto op = ge::OpDescUtils::CreateOperatorFromNode(graph->FindNode("concatv2"));
  auto builder = TilingContextBuilder();
  auto space_registry_v2 = DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(
      static_cast<gert::OppImplVersionTag>(concatv2_node->GetOpDesc()->GetOppImplVersion()));
  auto tiling_context_holder = builder
      .CompileInfo(&compile_info_holder)
      .PlatformInfo(reinterpret_cast<void *>(&platform_infos))
      .TilingData(tiling_data.get())
      .Workspace(reinterpret_cast<gert::ContinuousVector *>(workspace_size.get()))
      .SetSpaceRegistryV2(space_registry_v2, static_cast<gert::OppImplVersionTag>(concatv2_node->GetOpDesc()->GetOppImplVersion()))
      .Build(op);

  auto tiling_context = reinterpret_cast<TilingContext *>(tiling_context_holder.context_);
  // check content in context
  // 1.check input shape and tensor
  auto input_shape0 = tiling_context->GetDynamicInputShape(0, 0);
  EXPECT_NE(input_shape0, nullptr);
  auto input_shape_storage0 = input_shape0->GetStorageShape();
  EXPECT_EQ(input_shape_storage0.GetDimNum(), 4);
  EXPECT_EQ(input_shape_storage0.GetDim(0), 2);
  EXPECT_EQ(input_shape_storage0.GetDim(1), 3);
  EXPECT_EQ(input_shape_storage0.GetDim(2), 4);
  EXPECT_EQ(input_shape_storage0.GetDim(3), 5);

  auto input_shape_origin0 = input_shape0->GetOriginShape();
  EXPECT_EQ(input_shape_origin0.GetDimNum(), origin_shape_dims.size());
  for (size_t i = 0U; i < origin_shape_dims.size(); ++i) {
    EXPECT_EQ(input_shape_origin0.GetDim(i), origin_shape_dims[i]);
  }

  auto input_shape1 = tiling_context->GetDynamicInputShape(0,1);
  EXPECT_NE(input_shape1, nullptr);
  EXPECT_EQ((*input_shape1).GetStorageShape(), (*input_shape0).GetStorageShape());
  auto input_tensor1 = tiling_context->GetDynamicInputTensor(0, 1);
  EXPECT_NE(input_tensor1, nullptr);
  EXPECT_EQ(input_tensor1->GetAddr(),nullptr);

  // 2.check output shape
  auto output_shape = tiling_context->GetOutputShape(0);
  auto out_shape_storage = output_shape->GetStorageShape();
  EXPECT_EQ(out_shape_storage.GetDimNum(), 4);
  EXPECT_EQ(out_shape_storage.GetDim(0), 2);
  EXPECT_EQ(out_shape_storage.GetDim(1), 6);
  EXPECT_EQ(out_shape_storage.GetDim(2), 4);
  EXPECT_EQ(out_shape_storage.GetDim(3), 5);

  // 3.check compile info input
  auto compile_info = reinterpret_cast<const DummyCompileInfo *>(tiling_context->GetCompileInfo());
  ASSERT_NE(compile_info, nullptr);
  EXPECT_EQ(compile_info->a, 10);
  EXPECT_EQ(compile_info->b, 200);
  EXPECT_EQ(compile_info->c, compile_info_holder.c);

  // simulate op write tiling info
  tiling_context->SetTilingKey(123);
  tiling_context->SetBlockDim(456);
  tiling_context->SetNeedAtomic(true);
  // write tiling data
  DummyTilingParams *tiling_data_ptr = tiling_context->GetTilingData<DummyTilingParams>();
  EXPECT_NE(tiling_data_ptr, nullptr);
  tiling_data_ptr->x = 1;
  tiling_data_ptr->y = 2;

  // write workspace {0,1,2}
  for (size_t i = 0U; i < 3; ++i) {
    size_t *workspace_size = tiling_context->GetWorkspaceSizes(i + 1);
    *(workspace_size + i) = i;
  }
  tiling_context->SetTilingCond(789);

  // get from context
  EXPECT_EQ(tiling_context->GetTilingKey(), 123);
  EXPECT_EQ(tiling_context->GetBlockDim(), 456);
  EXPECT_EQ(tiling_context->NeedAtomic(), true);
  EXPECT_EQ(tiling_context->GetTilingCond(), 789);

  DummyTilingParams *output_tiling_data = tiling_context->GetTilingData<DummyTilingParams>();
  EXPECT_NE(output_tiling_data, nullptr);
  EXPECT_EQ(output_tiling_data->x, 1);
  EXPECT_EQ(output_tiling_data->y, 2);
}

// 校验 單算子單輸出多引用場景，input/output shape，compile info, tiling相关的输出可以正常地设置和读取
/*
 *    data1 data2   const
 *       \    |     /
 *         concatv2
 *          /   \
 *         /   size
 *         \    /
 *        netoutput
 *
 */
TEST_F(TilingContextBuilderUT, BuildTilingContextHolderWithMultiOutputSuccess) {
  auto graph = ShareGraph::ConcatV2MultiOutNodesGraph();
  auto tiling_data = gert::TilingData::CreateCap(1024);
  auto workspace_size = gert::ContinuousVector::Create<size_t>(16);
  DummyCompileInfo compile_info_holder = {10, 200, {10, 20, 30}};
  fe::PlatFormInfos platform_infos;
  auto concatv2_node = graph->FindNode("concatv2");
  auto op = ge::OpDescUtils::CreateOperatorFromNode(concatv2_node);
  auto space_registry =
      DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(
          static_cast<gert::OppImplVersionTag>(concatv2_node->GetOpDesc()->GetOppImplVersion()));
  auto builder = TilingContextBuilder();
  auto tiling_context_holder = builder
      .CompileInfo(&compile_info_holder)
      .PlatformInfo(reinterpret_cast<void *>(&platform_infos))
      .TilingData(tiling_data.get())
      .Workspace(reinterpret_cast<gert::ContinuousVector *>(workspace_size.get()))
      .SetSpaceRegistryV2(space_registry, static_cast<gert::OppImplVersionTag>(concatv2_node->GetOpDesc()->GetOppImplVersion()))
      .Build(op);

  auto tiling_context = reinterpret_cast<TilingContext *>(tiling_context_holder.context_);
  // check content in context
  // 1.check input shape and tensor
  auto input_shape0 = tiling_context->GetDynamicInputShape(0, 0);
  EXPECT_NE(input_shape0, nullptr);
  auto input_shape_storage0 = input_shape0->GetStorageShape();
  EXPECT_EQ(input_shape_storage0.GetDimNum(), 4);
  EXPECT_EQ(input_shape_storage0.GetDim(0), 2);
  EXPECT_EQ(input_shape_storage0.GetDim(1), 3);
  EXPECT_EQ(input_shape_storage0.GetDim(2), 4);
  EXPECT_EQ(input_shape_storage0.GetDim(3), 5);

  auto input_shape_origin0 = input_shape0->GetOriginShape();
  EXPECT_EQ(input_shape_origin0, input_shape_storage0);

  auto input_shape1 = tiling_context->GetDynamicInputShape(0,1);
  EXPECT_NE(input_shape1, nullptr);
  EXPECT_EQ(*input_shape1, *input_shape0);
  auto input_tensor1 = tiling_context->GetDynamicInputTensor(0, 1);
  EXPECT_NE(input_tensor1, nullptr);
  EXPECT_EQ(input_tensor1->GetAddr(),nullptr);

  // 2.check output shape
  auto output_shape = tiling_context->GetOutputShape(0);
  auto out_shape_storage = output_shape->GetStorageShape();
  EXPECT_EQ(out_shape_storage.GetDimNum(), 4);
  EXPECT_EQ(out_shape_storage.GetDim(0), 2);
  EXPECT_EQ(out_shape_storage.GetDim(1), 6);
  EXPECT_EQ(out_shape_storage.GetDim(2), 4);
  EXPECT_EQ(out_shape_storage.GetDim(3), 5);
  auto out_shape_origin = output_shape->GetOriginShape();
  EXPECT_EQ(out_shape_origin.GetDimNum(), 4);
  EXPECT_EQ(out_shape_origin.GetDim(0), 2);
  EXPECT_EQ(out_shape_origin.GetDim(1), 3);
  EXPECT_EQ(out_shape_origin.GetDim(2), 4);
  EXPECT_EQ(out_shape_origin.GetDim(3), 5);

  // 3.check compile info input
  auto compile_info = reinterpret_cast<const DummyCompileInfo *>(tiling_context->GetCompileInfo());
  ASSERT_NE(compile_info, nullptr);
  EXPECT_EQ(compile_info->a, 10);
  EXPECT_EQ(compile_info->b, 200);
  EXPECT_EQ(compile_info->c, compile_info_holder.c);
}

// 值依赖场景，输入数据来自const
TEST_F(TilingContextBuilderUT, BuildWithInputConstSuccess) {
  auto graph = ShareGraph::ConcatV2ConstDependencyGraph();
  auto tiling_data = gert::TilingData::CreateCap(1024);
  auto workspace_size = gert::ContinuousVector::Create<size_t>(16);
  std::string op_compile_info_json = "{}";
  fe::PlatFormInfos platform_infos;
  auto builder = TilingContextBuilder();
  auto concatv2_node = graph->FindNode("concatv2");
  auto op = ge::OpDescUtils::CreateOperatorFromNode(concatv2_node);
  auto space_registry =
      DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(
          static_cast<gert::OppImplVersionTag>(concatv2_node->GetOpDesc()->GetOppImplVersion()));
  auto tiling_context_holder = builder
                               .CompileInfo(const_cast<char *>(op_compile_info_json.c_str()))
                               .PlatformInfo(reinterpret_cast<void *>(&platform_infos))
                               .TilingData(tiling_data.get())
                               .Workspace(reinterpret_cast<gert::ContinuousVector *>(workspace_size.get()))
                               .SetSpaceRegistryV2(space_registry, static_cast<gert::OppImplVersionTag>(concatv2_node->GetOpDesc()->GetOppImplVersion()))
                               .Build(op);

  auto tiling_context = reinterpret_cast<TilingContext *>(tiling_context_holder.context_);
  // check content in context
  // 1.check input shape and tensor
  auto input_tensor2 = tiling_context->GetInputTensor(2);
  EXPECT_NE(input_tensor2, nullptr);
  EXPECT_EQ(input_tensor2->GetDataType(), ge::DT_INT32);
  const uint8_t * concat_dim_data_ptr = input_tensor2->GetData<uint8_t>();
  EXPECT_EQ(concat_dim_data_ptr[0], 1);
}

// 值依赖场景，输入数据来自上游算子输出
TEST_F(TilingContextBuilderUT, BuildWithInputDataDependencySuccess) {
  auto graph = ShareGraph::ConcatV2ConstDependencyGraph();
  auto tiling_data = gert::TilingData::CreateCap(1024);
  auto workspace_size = gert::ContinuousVector::Create<size_t>(16);
  std::string op_compile_info_json = "{}";
  fe::PlatFormInfos platform_infos;
  auto concatv2_node = graph->FindNode("concatv2");
  auto op = ge::OpDescUtils::CreateOperatorFromNode(concatv2_node);
  auto space_registry = DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(
      static_cast<gert::OppImplVersionTag>(concatv2_node->GetOpDesc()->GetOppImplVersion()));
  auto builder = TilingContextBuilder();
  auto tiling_context_holder = builder
      .CompileInfo(const_cast<char *>(op_compile_info_json.c_str()))
      .PlatformInfo(reinterpret_cast<void *>(&platform_infos))
      .TilingData(tiling_data.get())
      .Workspace(reinterpret_cast<gert::ContinuousVector *>(workspace_size.get()))
      .SetSpaceRegistryV2(space_registry, static_cast<gert::OppImplVersionTag>(concatv2_node->GetOpDesc()->GetOppImplVersion()))
      .Build(op);

  auto context = reinterpret_cast<TilingContext *>(tiling_context_holder.context_);
  // check content in context
  // 1.check input shape and tensor
  auto input_tensor2 = context->GetInputTensor(2);
  EXPECT_NE(input_tensor2, nullptr);
  EXPECT_EQ(input_tensor2->GetDataType(), ge::DT_INT32);
  const uint8_t *concat_dim_data_ptr = input_tensor2->GetData<uint8_t>();
  EXPECT_EQ(concat_dim_data_ptr[0], 1);
}

TEST_F(TilingContextBuilderUT, BuildAtomicTilingContextSuccess) {
  // build atomic clean node
  std::vector<int64_t> output_clean_sizes = {44, 55};
  auto tmp_graph = std::make_shared<ge::ComputeGraph>("tmp-graph");
  BuildAtomicNode(tmp_graph);

  auto tiling_data = gert::TilingData::CreateCap(1024);
  auto workspace_size = gert::ContinuousVector::Create<size_t>(16);

  std::string op_compile_info_json = "{}";
  auto clean_workspace_size = gert::ContinuousVector::Create<size_t>(16);
  auto clean_workspace_ptr = reinterpret_cast<gert::TypedContinuousVector<size_t> *>(clean_workspace_size.get());
  clean_workspace_ptr->SetSize(2);
  *(clean_workspace_ptr->MutableData()) = 22;
  *(clean_workspace_ptr->MutableData() + 1) = 33;

  auto op = ge::OpDescUtils::CreateOperatorFromNode(tmp_graph->FindNode("AtomicClean"));
  auto builder = AtomicTilingContextBuilder();
  auto tiling_context_holder = builder
      .CompileInfo(const_cast<char *>(op_compile_info_json.c_str()))
      .CleanWorkspaceSizes(reinterpret_cast<gert::ContinuousVector *>(clean_workspace_size.get()))
      .CleanOutputSizes(output_clean_sizes)
      .TilingData(tiling_data.get())
      .Workspace(reinterpret_cast<gert::ContinuousVector *>(workspace_size.get()))
      .Build(op);

  auto context = reinterpret_cast<AtomicCleanTilingContext *>(tiling_context_holder.context_);
  // check content in context
  auto clean_workspace_size_in_context = context->GetCleanWorkspaceSizes();
  EXPECT_EQ(clean_workspace_size_in_context->GetSize(), 2);
  auto ws_size_data = reinterpret_cast<const uint64_t*>(clean_workspace_size_in_context->GetData());
  EXPECT_EQ(ws_size_data[0], 22);
  EXPECT_EQ(ws_size_data[1], 33);

  EXPECT_EQ(context->GetCleanOutputSize(0), 44);
  EXPECT_EQ(context->GetCleanOutputSize(1), 55);
}
}  // namespace gert
