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
#define protected public
#include "base/registry/op_impl_space_registry_v2.h"
#include "graph/utils/math_util.h"
#include "exe_graph/lowering/tiling_context_builder.h"
#include "exe_graph/lowering/device_tiling_context_builder.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "exe_graph/runtime/atomic_clean_tiling_context.h"
#include "exe_graph/lowering/value_holder.h"
#include "platform/platform_infos_def.h"
#include "graph_metadef/common/ge_common/util.h"
#include "register/op_impl_registry.h"
#include "faker/node_faker.h"
#include "faker/space_registry_faker.h"
#include "graph/debug/ge_attr_define.h"
#include "common/checker.h"
#undef protected

namespace gert {
class TilingContextBuilderUT : public testing::Test {};

namespace {
IMPL_OP(DDIT02).InputsDataDependency({0, 2});

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
  std::vector<int64_t> workspace_indexes = {1, 2};
  std::vector<int64_t> outputs_indexes = {0, 2};

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
}  // namespace

TEST_F(TilingContextBuilderUT, CompileInfoNullptr) {
  fe::PlatFormInfos platform_infos;
  auto builder = TilingContextBuilder();
  auto node = ComputeNodeFaker().NameAndType("Test", "DDIT02").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ASSERT_NE(node, nullptr);
  node->GetOpDesc()->SetOpInferDepends({"x", "z"});
  auto op = ge::OpDescUtils::CreateOperatorFromNode(node->shared_from_this());

  ge::graphStatus ret;
  auto tiling_context_holder =
      builder.CompileInfo(nullptr).PlatformInfo(reinterpret_cast<void *>(&platform_infos)).Build(op, ret);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_context_holder.context_, nullptr);
}

TEST_F(TilingContextBuilderUT, PlatformInfoNullptr) {
  fe::PlatFormInfos platform_infos;
  auto builder = TilingContextBuilder();
  std::string op_compile_info_json = "{}";

  auto node = ComputeNodeFaker().NameAndType("Test", "DDIT02").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ASSERT_NE(node, nullptr);
  node->GetOpDesc()->SetOpInferDepends({"x", "z"});
  auto op = ge::OpDescUtils::CreateOperatorFromNode(node->shared_from_this());

  ge::graphStatus ret;
  auto tiling_context_holder = builder.CompileInfo(&op_compile_info_json).PlatformInfo(nullptr).Build(op, ret);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_context_holder.context_, nullptr);
}

TEST_F(TilingContextBuilderUT, BuildRTInputTensorsFailed) {
  auto node = ComputeNodeFaker().NameAndType("UbNode", "DDIT02").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  node->GetOpDesc()->SetOpInferDepends({"x", "z"});

  auto graph = std::make_shared<ge::ComputeGraph>("ub_graph");
  auto data0 = ComputeNodeFaker(graph)
                   .NameAndType("Data0", "Data")
                   .Attr<int64_t>(ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), 0)
                   .IoNum(0, 1)
                   .Build();
  auto data1 = ComputeNodeFaker(graph)
                   .NameAndType("Data1", "Data")
                   .Attr<int64_t>(ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), 1)
                   .IoNum(0, 1)
                   .Build();
  auto data2 = ComputeNodeFaker(graph)
                   .NameAndType("Data2", "Data")
                   .Attr<int64_t>(ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), 2)
                   .IoNum(0, 1)
                   .Build();
  auto node2 = ComputeNodeFaker().NameAndType("UbNode2", "DDIT02").IoNum(1, 1).InputNames({"d"}).Build();
  ge::GraphUtils::AddEdge(data0->GetOutDataAnchor(0), node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data1->GetOutDataAnchor(0), node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(data2->GetOutDataAnchor(0), node->GetInDataAnchor(2));
  graph->SetParentNode(node2);

  // construct op impl registry
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto tiling_data = gert::TilingData::CreateCap(1024);
  auto workspace_size = gert::ContinuousVector::Create<size_t>(16);
  std::string op_compile_info_json = "{}";
  fe::PlatFormInfos platform_infos;
  auto builder = TilingContextBuilder();
  auto op = ge::OpDescUtils::CreateOperatorFromNode(node->shared_from_this());
  ge::graphStatus ret;
  auto tiling_context_holder = builder.CompileInfo(const_cast<char *>(op_compile_info_json.c_str()))
                                   .PlatformInfo(reinterpret_cast<void *>(&platform_infos))
                                   .TilingData(tiling_data.get())
                                   .Workspace(reinterpret_cast<gert::ContinuousVector *>(workspace_size.get()))
                                   .SetSpaceRegistryV2(space_registry, gert::OppImplVersionTag::kOpp)
                                   .Build(op, ret);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_NE(tiling_context_holder.context_, nullptr);
  DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
}

TEST_F(TilingContextBuilderUT, BuildRTInputTensorsFailed_UseRegistryV2) {
  auto node = ComputeNodeFaker().NameAndType("UbNode", "DDIT02").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  node->GetOpDesc()->SetOpInferDepends({"x", "z"});

  auto graph = std::make_shared<ge::ComputeGraph>("ub_graph");
  auto data0 = ComputeNodeFaker(graph)
                   .NameAndType("Data0", "Data")
                   .Attr<int64_t>(ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), 0)
                   .IoNum(0, 1)
                   .Build();
  auto data1 = ComputeNodeFaker(graph)
                   .NameAndType("Data1", "Data")
                   .Attr<int64_t>(ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), 1)
                   .IoNum(0, 1)
                   .Build();
  auto data2 = ComputeNodeFaker(graph)
                   .NameAndType("Data2", "Data")
                   .Attr<int64_t>(ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), 2)
                   .IoNum(0, 1)
                   .Build();
  auto node2 = ComputeNodeFaker().NameAndType("UbNode2", "DDIT02").IoNum(1, 1).InputNames({"d"}).Build();
  ge::GraphUtils::AddEdge(data0->GetOutDataAnchor(0), node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data1->GetOutDataAnchor(0), node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(data2->GetOutDataAnchor(0), node->GetInDataAnchor(2));
  graph->SetParentNode(node2);

  // construct op impl registry
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);

  auto tiling_data = gert::TilingData::CreateCap(1024);
  auto workspace_size = gert::ContinuousVector::Create<size_t>(16);
  std::string op_compile_info_json = "{}";
  fe::PlatFormInfos platform_infos;
  auto builder = TilingContextBuilder();
  auto op = ge::OpDescUtils::CreateOperatorFromNode(node->shared_from_this());
  ge::graphStatus ret;
  auto tiling_context_holder = builder.CompileInfo(const_cast<char *>(op_compile_info_json.c_str()))
                                   .PlatformInfo(reinterpret_cast<void *>(&platform_infos))
                                   .TilingData(tiling_data.get())
                                   .Workspace(reinterpret_cast<gert::ContinuousVector *>(workspace_size.get()))
                                   .SetSpaceRegistryV2(space_registry, OppImplVersionTag::kOpp)
                                   .Build(op, ret);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_NE(tiling_context_holder.context_, nullptr);

  auto builder_failed = TilingContextBuilder();
  auto tiling_context_holder_fail = builder_failed.CompileInfo(const_cast<char *>(op_compile_info_json.c_str()))
                                        .PlatformInfo(reinterpret_cast<void *>(&platform_infos))
                                        .TilingData(tiling_data.get())
                                        .Workspace(reinterpret_cast<gert::ContinuousVector *>(workspace_size.get()))
                                        .SetSpaceRegistryV2(space_registry, OppImplVersionTag::kVersionEnd)
                                        .Build(op, ret);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto ctx = tiling_context_holder_fail.context_;
  EXPECT_NE(ctx, nullptr);
  DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
}

// 值依赖场景，输入数据来自const
TEST_F(TilingContextBuilderUT, BuildWithInputConstSuccess) {
  auto tiling_data = gert::TilingData::CreateCap(1024);
  auto workspace_size = gert::ContinuousVector::Create<size_t>(16);
  std::string op_compile_info_json = "{}";
  fe::PlatFormInfos platform_infos;
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto builder = TilingContextBuilder();

  auto foo_node = ComputeNodeFaker().NameAndType("foo", "Foo").IoNum(1, 2).InputNames({"x"}).Build();
  auto bar_node = ComputeNodeFaker().NameAndType("bar", "Bar").IoNum(2, 1).InputNames({"x", "y"}).Build();
  ge::GraphUtils::AddEdge(foo_node->GetOutDataAnchor(0), bar_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(foo_node->GetOutDataAnchor(1), bar_node->GetInDataAnchor(1));
  const size_t k_input_anchor = 2U;
  ge::NodeUtils::AppendInputAnchor(bar_node, k_input_anchor);
  EXPECT_EQ(bar_node->GetAllInDataAnchorsSize(), k_input_anchor);
  ge::OpDescPtr op_desc = bar_node->GetOpDesc();
  ge::GeTensorDesc tensor_desc(ge::GeShape({1}));
  op_desc->AddOutputDesc("z", tensor_desc);
  op_desc->MutableInputDesc(1)->SetDataType(ge::DT_INT32);
  op_desc->MutableInputDesc(1)->SetShape(ge::GeShape({1}));
  op_desc->MutableInputDesc(1)->SetOriginShape(ge::GeShape({1}));
  auto op = ge::OpDescUtils::CreateOperatorFromNode(bar_node->shared_from_this());
  ge::graphStatus ret;
  auto tiling_context_holder = builder.CompileInfo(const_cast<char *>(op_compile_info_json.c_str()))
                                   .PlatformInfo(reinterpret_cast<void *>(&platform_infos))
                                   .TilingData(tiling_data.get())
                                   .Deterministic(1)
                                   .DeterministicLevel(1)
                                   .Workspace(reinterpret_cast<gert::ContinuousVector *>(workspace_size.get()))
                                   .SetSpaceRegistryV2(space_registry, gert::OppImplVersionTag::kOpp)
                                   .Build(op, ret);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto tiling_context = reinterpret_cast<TilingContext *>(tiling_context_holder.context_);
  // check content in context
  // 1.check input shape and tensor
  auto input_tensor1 = tiling_context->GetInputTensor(1);
  EXPECT_NE(input_tensor1, nullptr);
  EXPECT_EQ(input_tensor1->GetDataType(), ge::DT_INT32);
  EXPECT_EQ(input_tensor1->GetOriginShape().GetDim(0), 1);
  //  强一致性计算紧急需求上库，ge暂时不能依赖metadef，已于BBIT及本地验证DT通过，后续补上
  //  auto deterministic_level = tiling_context->GetDeterministicLevel();
  //  EXPECT_EQ(deterministic_level, 1);

  // deprecated later
  builder.CompileInfo(const_cast<char *>(op_compile_info_json.c_str()))
      .PlatformInfo(reinterpret_cast<void *>(&platform_infos))
      .TilingData(tiling_data.get())
      .Deterministic(1)
      .Workspace(reinterpret_cast<gert::ContinuousVector *>(workspace_size.get()))
      .SetSpaceRegistryV2(space_registry, gert::OppImplVersionTag::kOpp)
      .Build(op);

  bg::ValueHolder::PopGraphFrame();
  DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(nullptr);
}

TEST_F(TilingContextBuilderUT, BuildAtomicCompileInfoNullptr) {
  // build atomic clean node
  auto tmp_graph = std::make_shared<ge::ComputeGraph>("tmp-graph");
  BuildAtomicNode(tmp_graph);
  auto op = ge::OpDescUtils::CreateOperatorFromNode(tmp_graph->FindNode("AtomicClean"));
  auto builder = AtomicTilingContextBuilder();
  ge::graphStatus ret;
  auto tiling_context_holder = builder.CompileInfo(nullptr).Build(op, ret);
  auto context = reinterpret_cast<AtomicCleanTilingContext *>(tiling_context_holder.context_);
  EXPECT_EQ(context, nullptr);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  // deprecated later
  builder.CompileInfo(nullptr).Build(op);
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
  ge::graphStatus ret;
  auto tiling_context_holder =
      builder.CompileInfo(const_cast<char *>(op_compile_info_json.c_str()))
          .CleanWorkspaceSizes(reinterpret_cast<gert::ContinuousVector *>(clean_workspace_size.get()))
          .CleanOutputSizes(output_clean_sizes)
          .TilingData(tiling_data.get())
          .Workspace(reinterpret_cast<gert::ContinuousVector *>(workspace_size.get()))
          .Build(op, ret);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto context = reinterpret_cast<AtomicCleanTilingContext *>(tiling_context_holder.context_);
  // check content in context
  auto clean_workspace_size_in_context = context->GetCleanWorkspaceSizes();
  EXPECT_EQ(clean_workspace_size_in_context->GetSize(), 2);
  auto ws_size_data = reinterpret_cast<const uint64_t *>(clean_workspace_size_in_context->GetData());
  EXPECT_EQ(ws_size_data[0], 22);
  EXPECT_EQ(ws_size_data[1], 33);

  EXPECT_EQ(context->GetCleanOutputSize(0), 44);
  EXPECT_EQ(context->GetCleanOutputSize(1), 55);
}

static ge::ComputeGraphPtr ConcatV2ConstDependencyGraph() {
  // root
  auto root_graph = std::make_shared<ge::ComputeGraph>("root_graph");
  auto op_desc = std::make_shared<ge::OpDesc>("ifa", "IncreFlashAttention");

  // set ifa ir
  op_desc->AppendIrInput("query", ge::kIrInputRequired);
  op_desc->AppendIrInput("key", ge::kIrInputDynamic);
  op_desc->AppendIrInput("value", ge::kIrInputDynamic);
  op_desc->AppendIrInput("pse_shift", ge::kIrInputOptional);
  op_desc->AppendIrInput("atten_mask", ge::kIrInputOptional);
  op_desc->AppendIrInput("actual_seq_lengths", ge::kIrInputOptional);
  op_desc->MutableAllInputName() = {{"query", 0}, {"key0", 1}, {"value0", 2}, {"actual_seq_lengths", 5}};
  op_desc->AppendIrOutput("attention_out", ge::kIrOutputRequired);
  op_desc->MutableAllOutputName() = {{"attention_out", 0}};

  std::vector<int64_t> in_shape = {1, 4, 1, 1024};
  ge::GeShape shape(in_shape);
  ge::GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetShape(shape);

  ge::GeTensorDesc invalid_desc;
  invalid_desc.SetDataType(ge::DT_UNDEFINED);
  invalid_desc.SetFormat(ge::FORMAT_RESERVED);

  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(invalid_desc);
  op_desc->AddInputDesc(invalid_desc);
  op_desc->AddInputDesc(tensor_desc);

  op_desc->AddOutputDesc(tensor_desc);

  const auto node_id = op_desc->GetId();
  auto ifa_node = root_graph->AddNode(op_desc);
  for (size_t i = 0UL; i < op_desc->GetAllInputsSize(); ++i) {
    const auto input_desc = op_desc->GetInputDesc(i);
    if (input_desc.IsValid() != ge::GRAPH_SUCCESS) {
      GELOGD("Node: %s, input: %zu, is invalid, skip add edge.", op_desc->GetNamePtr(), i);
      continue;
    }
    auto op_data = ge::OpDescBuilder(std::to_string(i), "Data").AddInput("x").AddOutput("y").Build();
    auto data_node = root_graph->AddNode(op_data);
    ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), ifa_node->GetInDataAnchor(i));
  }

  for (size_t i = 0UL; i < op_desc->GetOutputsSize(); ++i) {
    const auto input_desc = op_desc->GetOutputDesc(i);
    auto out_op = ge::OpDescBuilder(std::to_string(i), "Data").AddInput("x").AddOutput("y").Build();
    auto out_node = root_graph->AddNode(out_op);
    ge::GraphUtils::AddEdge(ifa_node->GetOutDataAnchor(i), out_node->GetInDataAnchor(0));
  }

  // AddNode operation may change node id to 0, which need to be recovered
  op_desc->SetId(node_id);

  return root_graph;
}

TEST_F(TilingContextBuilderUT, BuildDeviceTilingContextSuccess) {
  auto graph = ConcatV2ConstDependencyGraph();
  fe::PlatFormInfos platform_infos;
  auto node = graph->FindNode("ifa");
  EXPECT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);

  size_t total_plain_size{0UL};

  size_t extend_info_size{0UL};
  bg::BufferPool buffer_pool;
  auto compute_node_extend_holder = bg::CreateComputeNodeInfo(node, buffer_pool, extend_info_size);
  EXPECT_NE(compute_node_extend_holder, nullptr);

  const size_t device_tiling_size = gert::DeviceTilingContextBuilder::CalcTotalTiledSize(op_desc);
  size_t aligned_tiling_context_size = ge::RoundUp(static_cast<uint64_t>(device_tiling_size), sizeof(uintptr_t));
  aligned_tiling_context_size += extend_info_size;
  total_plain_size += aligned_tiling_context_size;

  // tiling
  const auto aligned_tiling_size = 1024UL;
  total_plain_size += aligned_tiling_size;

  const size_t workspace_addr_size = 16 * sizeof(gert::ContinuousVector);
  total_plain_size += workspace_addr_size;

  auto host_pointer = std::unique_ptr<uint8_t[]>(new uint8_t[total_plain_size]);
  EXPECT_NE(host_pointer, nullptr);
  auto device_addr = std::unique_ptr<uint8_t[]>(new uint8_t[total_plain_size]);
  EXPECT_NE(device_addr, nullptr);

  // copy tiling_data
  uint8_t *context_host_begin = &host_pointer[aligned_tiling_size + workspace_addr_size];
  uint64_t context_dev_begin = ge::PtrToValue(device_addr.get()) + aligned_tiling_size + workspace_addr_size;

  auto space_registry = DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  gert::Tensor host_tensor;
  gert::Tensor device_tensor;
  host_tensor.MutableTensorData().SetAddr(reinterpret_cast<void *>(0x120000), nullptr);
  device_tensor.MutableTensorData().SetAddr(reinterpret_cast<void *>(0x120000), nullptr);
  std::map<size_t, gert::AddrRefreshedTensor> index_to_tensor;
  index_to_tensor[3] = {&host_tensor, ge::PtrToValue(&device_tensor)};

  gert::TiledKernelContextHolder tiling_context_holder;
  tiling_context_holder.compute_node_info_size_ = extend_info_size;
  tiling_context_holder.host_compute_node_info_ = compute_node_extend_holder.get();

  auto context_builder = gert::DeviceTilingContextBuilder();
  ge::Status ret = context_builder.PlatformInfo(reinterpret_cast<void *>(&platform_infos))
                       .TilingData(device_addr.get())
                       .Deterministic(0)
                       .DeterministicLevel(1)
                       .CompileInfo(nullptr)
                       .Workspace(ge::ValueToPtr(ge::PtrToValue(device_addr.get()) + aligned_tiling_size))
                       .AddrRefreshedInputTensor(index_to_tensor)
                       .TiledHolder(context_host_begin, context_dev_begin,
                                    total_plain_size - aligned_tiling_size - workspace_addr_size)
                       .Build(node, tiling_context_holder);
  // mock h2d
  EXPECT_EQ(memcpy_s(device_addr.get(), total_plain_size, host_pointer.get(), total_plain_size), 0);

  auto host_context = reinterpret_cast<TilingContext *>(tiling_context_holder.host_context_);
  EXPECT_NE(host_context, nullptr);
  auto device_context = reinterpret_cast<TilingContext *>(tiling_context_holder.dev_context_addr_);
  EXPECT_NE(device_context, nullptr);

  // checkout input chains
  // input0 shape
  EXPECT_NE(device_context->GetInputShape(0), nullptr);
  EXPECT_EQ(device_context->GetInputShape(0)->GetStorageShape().GetDimNum(),
            op_desc->GetInputDesc(0).GetShape().GetDimNum());
  for (size_t i = 0UL; i < device_context->GetInputShape(0)->GetStorageShape().GetDimNum(); ++i) {
    EXPECT_EQ(device_context->GetInputShape(0)->GetStorageShape().GetDim(i),
              op_desc->GetInputDesc(0).GetShape().GetDim(i));
  }

  // tiling depends tensor addr
  const gert::Tensor *value_tensor = device_context->GetInputTensor(3);
  EXPECT_EQ(value_tensor, &device_tensor);
  EXPECT_EQ(value_tensor->GetTensorData().GetAddr(), reinterpret_cast<void *>(0x120000));

  // checkout output chains
  // tiling_data addr
  const auto tiling_data_ptr = device_context->GetOutputPointer<TilingData>(TilingContext::kOutputTilingData);
  EXPECT_EQ(ge::PtrToValue(tiling_data_ptr), ge::PtrToValue(device_addr.get()));

  // tiling_key_addr
  const auto tiling_key_ptr = device_context->GetOutputPointer<uint64_t>(TilingContext::kOutputTilingKey);
  EXPECT_EQ(reinterpret_cast<uint64_t>(tiling_key_ptr) % 128U, 0U);
  EXPECT_EQ(ge::PtrToValue(tiling_key_ptr),
            tiling_context_holder.output_addrs_[gert::TilingContext::TilingOutputIndex::kOutputTilingKey]);
  device_context->SetTilingKey(0x123UL);
  EXPECT_EQ(*tiling_key_ptr, 0x123UL);

  // block_dim addr
  const auto block_dim_ptr = device_context->GetOutputPointer<uint32_t>(TilingContext::kOutputBlockDim);
  EXPECT_EQ(reinterpret_cast<uint64_t>(block_dim_ptr) % 128U, 0U);
  EXPECT_EQ(ge::PtrToValue(block_dim_ptr),
            tiling_context_holder.output_addrs_[gert::TilingContext::TilingOutputIndex::kOutputBlockDim]);
  device_context->SetBlockDim(40U);
  EXPECT_EQ(*block_dim_ptr, 40U);

  // op type
  char *op_type = reinterpret_cast<char *>(tiling_context_holder.dev_op_type_addr_);
  EXPECT_EQ(op_type, op_desc->GetType());

  // deterministic level
  //  强一致性计算紧急需求上库，ge暂时不能依赖metadef，已于BBIT及本地验证DT通过，后续补上
  //  auto deterministic_level = device_context->GetDeterministicLevel();
  //  EXPECT_EQ(deterministic_level, 1);

  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(TilingContextBuilderUT, GetDependInputTensorAddr_Data_Input_Success) {
  auto node = ComputeNodeFaker().NameAndType("Test", "Test").IoNum(1, 1).InputNames({"x"}).Build();
  auto builder = TilingContextBuilder();
  TensorAddress address = nullptr;
  const auto ret = builder.GetDependInputTensorAddr(ge::OpDescUtils::CreateOperatorFromNode(node), 0, address);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_TRUE(address == nullptr);
}

TEST_F(TilingContextBuilderUT, GetDependInputTensorAddr_Const_Input_Success) {
  auto node = ComputeNodeFaker().NameAndType("Test", "Test").IoNum(1, 1).InputNames({"x"}).Build();
  auto const1 = ComputeNodeFaker().NameAndType("Const", "Const").IoNum(1, 1).InputNames({"x"}).Build();
  ge::GraphUtils::AddEdge(const1->GetOutDataAnchor(0), node->GetInDataAnchor(0));
  int32_t weight[1] = {1};
  ge::GeTensorDesc weight_desc(ge::GeShape({1}), ge::FORMAT_NHWC, ge::DT_INT32);
  ge::GeTensorPtr tensor0 = std::make_shared<ge::GeTensor>(weight_desc, (uint8_t *) weight, sizeof(weight));
  ge::OpDescUtils::SetWeights(const1, {tensor0});
  auto builder = TilingContextBuilder();
  TensorAddress address = 0x0;
  const auto ret = builder.GetDependInputTensorAddr(ge::OpDescUtils::CreateOperatorFromNode(node), 0, address);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_NE(address, nullptr);
}
}  // namespace gert
