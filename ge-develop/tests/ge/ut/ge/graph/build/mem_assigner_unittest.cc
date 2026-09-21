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
#include <memory>

#include "graph/anchor.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "omg/omg_inner_types.h"
#include "../passes/graph_builder_utils.h"

#include "macro_utils/dt_public_scope.h"
#include "graph/build/memory/binary_block_mem_assigner.h"
#include "framework/memory/memory_assigner.h"
#include "graph/build/memory/hybrid_mem_assigner.h"
#include "graph/build/memory/max_block_mem_assigner.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/ge_local_context.h"
#include "common/share_graph.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/build/model_builder.h"
#include "stub/gert_runtime_stub.h"
#include "framework/common/runtime_tensor_desc.h"
#include "common/mem_conflict_share_graph.h"
#include "macro_utils/dt_public_unscope.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "test_memory_shared_graph.h"

using namespace std;
using namespace testing;
using namespace ge;
using domi::GetContext;

namespace {
struct ReuseNode {
  ReuseNode(std::string name, OpMemoryType mem_type, uint32_t index, int64_t offset = 0, int64_t size = 0)
      : name(std::move(name)), mem_type(mem_type), index(index), offset(offset), size(size) {}

  std::string name;
  OpMemoryType mem_type;
  uint32_t index;
  int64_t offset;
  int64_t size;

  std::string ToString() {
    std::stringstream ss;
    ss << "Node:" << name <<" type:" << mem_type << " index:" << index << " offset:" << offset <<" size:"<<size;
    return ss.str();
  }
};
}

class UtestMemoryAssignerTest : public testing::Test {
 public:
  ge::OpDescPtr CreateOpWithWsSize(const string &name, int64_t wsByte, const string &type = "some",
                                   int64_t size = 1024, std::vector<int64_t> shape = {1, 1, 16, 8},
                                   Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT) {
    ge::OpDescPtr op_def = std::make_shared<ge::OpDesc>(name, type);
    auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
    desc_temp_ptr->SetShape(GeShape(shape));
    desc_temp_ptr->SetFormat(format);
    desc_temp_ptr->SetDataType(data_type);
    desc_temp_ptr->SetOriginFormat(format);
    desc_temp_ptr->SetOriginShape(GeShape(shape));
    desc_temp_ptr->SetOriginDataType(data_type);

    auto desc_temp = *desc_temp_ptr;

    TensorUtils::SetSize(desc_temp, size);
    op_def->AddInputDesc(desc_temp);
    op_def->AddOutputDesc(desc_temp);
    if (wsByte != 0) {
      std::vector<int64_t> workspace_bytes;
      workspace_bytes.push_back(wsByte);
      op_def->SetWorkspaceBytes(workspace_bytes);
    }
    return op_def;
  }
  ge::OpDescPtr CreateRefOpWithWsSize(const string &name, int64_t wsByte, const string &type = "some") {
    ge::OpDescPtr op_def = std::make_shared<ge::OpDesc>(name, type);
    auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
    auto desc_temp = *desc_temp_ptr;

    TensorUtils::SetSize(desc_temp, 1024);
    op_def->AddInputDesc(desc_temp);

    auto desc_output_ptr = std::make_shared<ge::GeTensorDesc>();
    auto desc_output = *desc_output_ptr;
    TensorUtils::SetSize(desc_output, 6500);
    ge::TensorUtils::SetReuseInput(desc_output, true);
    ge::TensorUtils::SetReuseInputIndex(desc_output, 0);
    op_def->AddOutputDesc(desc_output);

    std::vector<int64_t> workspace_bytes;
    workspace_bytes.push_back(wsByte);
    op_def->SetWorkspaceBytes(workspace_bytes);
    return op_def;
  }

  void SetStreamId(ge::ComputeGraphPtr &graph, const string &node, int64_t stream_id) {
    const auto call_node = graph->FindNode(node);
    EXPECT_NE(call_node, nullptr);
    const auto call_desc = call_node->GetOpDesc();
    EXPECT_NE(call_desc, nullptr);
    call_desc->SetStreamId(stream_id);
    if (call_desc->GetType() == DATA) {
      call_desc->SetOpKernelLibName(kEngineNameGeLocal);
    }
  }

  int64_t GetStreamId(ge::ComputeGraphPtr &graph, const string &node) {
    const auto call_node = graph->FindNode(node);
    EXPECT_NE(call_node, nullptr);
    const auto call_desc = call_node->GetOpDesc();
    EXPECT_NE(call_desc, nullptr);
    return MemReuseUtils::GetStreamId(call_desc.get());
  }

  void SetTensorSize(const GeShape &shape,
                     const Format format,
                     const DataType data_type,
                     GeTensorDesc &tensor_desc) {
    int64_t tensor_size = 0;
    TensorUtils::CalcTensorMemSize(shape, format, data_type, tensor_size);
    TensorUtils::SetSize(tensor_desc, tensor_size);
  }

  void UpdateGraphTensorSize(ComputeGraphPtr &graph) {
    for (auto &node : graph->GetAllNodes()) {
      for (auto &input_name : node->GetOpDesc()->GetAllInputNames()) {
        auto input = node->GetOpDesc()->MutableInputDesc(input_name);
        SetTensorSize(input->GetShape(), input->GetFormat(), input->GetDataType(), *input);
      }
      auto out_size = node->GetOpDesc()->GetAllOutputsDescSize();
      for (int32_t id = 0; id < static_cast<int32_t>(out_size); id++) {
        auto output = node->GetOpDesc()->MutableOutputDesc(id);
        SetTensorSize(output->GetShape(), output->GetFormat(), output->GetDataType(), *output);
      }
    }
  }

  void MakeGraph(ge::ComputeGraphPtr &graph, const string &type = "some") {
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 6000, type);
    op_def_a->SetStreamId(0);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 120000);
    op_def_b->SetStreamId(0);
    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 16000);
    op_def_c->SetStreamId(1);
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 24000);
    op_def_d->SetStreamId(2);
    ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 24000);
    op_def_e->SetStreamId(3);
    ge::OpDescPtr op_def_f = CreateOpWithWsSize("F", 30000);
    op_def_f->SetStreamId(2);
    ge::OpDescPtr op_def_g = CreateOpWithWsSize("G", 32000);
    op_def_g->SetStreamId(3);
    ge::OpDescPtr op_def_h = CreateOpWithWsSize("H", 48000);
    op_def_h->SetStreamId(2);
    ge::OpDescPtr op_def_i = CreateOpWithWsSize("I", 60000);
    op_def_i->SetStreamId(2);
    ge::OpDescPtr op_def_j = CreateOpWithWsSize("J", 256000, NETOUTPUT);
    op_def_j->SetStreamId(3);

    // add node
    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);
    ge::NodePtr node_e = graph->AddNode(op_def_e);
    ge::NodePtr node_f = graph->AddNode(op_def_f);
    ge::NodePtr node_g = graph->AddNode(op_def_g);
    ge::NodePtr node_h = graph->AddNode(op_def_h);
    ge::NodePtr node_i = graph->AddNode(op_def_i);
    ge::NodePtr node_j = graph->AddNode(op_def_j);

    // add edge
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_g->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_f->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_g->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_f->GetOutDataAnchor(0), node_h->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_g->GetOutDataAnchor(0), node_j->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_h->GetOutDataAnchor(0), node_i->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_i->GetOutDataAnchor(0), node_j->GetInDataAnchor(1));

    domi::GetContext().out_nodes_map["H"] = {0};
    domi::GetContext().out_nodes_map["I"] = {0};
    domi::GetContext().out_nodes_map["J"] = {0};
    graph->TopologicalSorting();
  }

  void MakeStreamGraph(ge::ComputeGraphPtr &graph, const string &type = "some") {
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 6000, type);
    op_def_a->SetStreamId(39);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 120000);
    op_def_b->SetStreamId(39);
    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 16000, "some", 20480);
    op_def_c->SetStreamId(44);
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 24000, "some", 40960);
    op_def_d->SetStreamId(44);
    ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 24000);
    op_def_e->SetStreamId(17);
    ge::OpDescPtr op_def_f = CreateOpWithWsSize("F", 30000);
    op_def_f->SetStreamId(6);
    ge::OpDescPtr op_def_g = CreateOpWithWsSize("G", 32000);
    op_def_g->SetStreamId(6);
    ge::OpDescPtr op_def_h = CreateOpWithWsSize("H", 48000);
    op_def_h->SetStreamId(42);

    auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
    auto desc_temp = *desc_temp_ptr;
    TensorUtils::SetSize(desc_temp, 1024);
    op_def_e->AddInputDesc(desc_temp);

    // add node
    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);
    ge::NodePtr node_e = graph->AddNode(op_def_e);
    ge::NodePtr node_f = graph->AddNode(op_def_f);
    ge::NodePtr node_g = graph->AddNode(op_def_g);
    ge::NodePtr node_h = graph->AddNode(op_def_h);

    // a:39 ----b:39--h:42
    //   \       |
    //   c:44   f:6
    //     \     |
    //     d:44 g:6
    //        \  |
    //         e:17
    // add edge
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_h->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_f->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_f->GetOutDataAnchor(0), node_g->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_g->GetOutDataAnchor(0), node_e->GetInDataAnchor(1));
    graph->TopologicalSorting();
  }

  void MakeOutputNotZeroCopyGraph(ge::ComputeGraphPtr &graph, const string &type = "some") {
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 1024, type, 2048);
    op_def_a->SetStreamId(1);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 1024, "HcomAllGather", 2048);
    op_def_b->SetStreamId(1);
    op_def_b->SetOpKernelLibName(ge::kEngineNameHccl);
    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 1024, type, 2048);
    op_def_c->SetStreamId(1);
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 1024, "NetOutput", 2048);
    op_def_d->SetStreamId(1);

    auto output_tensordesc = op_def_c->MutableOutputDesc(0);
    ge::TensorUtils::SetReuseInput(*output_tensordesc, true);
    uint32_t reuse_input_index = 0;
    ge::TensorUtils::SetReuseInputIndex(*output_tensordesc, reuse_input_index);

    // add node
    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);

    /**
     *       a:1
     *        |
     *       b:1   hccl算子
     *        |
     *       c:1  输出ref输入
     *        |
     *       d:1
     */

    // add edge
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    graph->TopologicalSorting();
  }

  void MakeDiffStreamGraph(ge::ComputeGraphPtr &graph, const string &type = "some") {
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 1024, type, 2048);
    op_def_a->SetStreamId(1);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 1024, type, 2048);
    op_def_b->SetStreamId(0);
    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 1024, type, 2048);
    op_def_c->SetStreamId(1);
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 1024, type, 2048);
    op_def_d->SetStreamId(1);

    ge::AttrUtils::SetBool(op_def_d, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
    (void)ge::AttrUtils::SetBool(op_def_d, ATTR_NAME_OUTPUT_REUSE_INPUT, true);
    (void)ge::AttrUtils::SetInt(op_def_d, ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);
    auto output_tensordesc = op_def_d->MutableOutputDesc(0);
    ge::TensorUtils::SetReuseInput(*output_tensordesc, true);
    uint32_t reuse_input_index = 0;
    ge::TensorUtils::SetReuseInputIndex(*output_tensordesc, reuse_input_index);

    ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 1024, type, 2048);
    op_def_e->SetStreamId(1);

    auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
    auto desc_temp = *desc_temp_ptr;
    TensorUtils::SetSize(desc_temp, 1024);
    op_def_d->AddInputDesc(desc_temp);

    ge::OpDescPtr op_def_f = CreateOpWithWsSize("F", 1024, type, 2048);
    op_def_f->SetStreamId(1);

    // add node
    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);
    ge::NodePtr node_e = graph->AddNode(op_def_e);
    ge::NodePtr node_f = graph->AddNode(op_def_f);

    /**       a:1
     *    |     \
     *   b:0    c:1
     *     \    /
     *      d:1
     *        |
     *      e:1
     *        |
     *      f:1
     */
    // add edge
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_d->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_f->GetInDataAnchor(0));
    graph->TopologicalSorting();
  }

  /**     a:1
   *    |     \
   *   b:0    c:1
   *     \    /\
   *      d:2  e:1
   *            |
   *           f:1
   *  b and f可能并发执行，f不能复用b，而b和c使用统一块内存，所以f不能复用c
   */
  void MakeDiffStreamSameOutGraphWithNoPaddingContinousInput(ge::ComputeGraphPtr &graph, const string &type = "some", bool d_before_f = false) {
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 0, type, 2048);
    op_def_a->SetStreamId(1);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 0, type, 2048);
    op_def_b->SetStreamId(0);
    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 0, type, 2048);
    op_def_c->SetStreamId(1);
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 0, type, 2048);
    op_def_d->SetStreamId(2);

    ge::AttrUtils::SetBool(op_def_d, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
    ge::AttrUtils::SetBool(op_def_d, ATTR_NAME_OUTPUT_REUSE_INPUT, true);
    (void)ge::AttrUtils::SetInt(op_def_d, ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);
    auto output_tensordesc = op_def_d->MutableOutputDesc(0);
    ge::TensorUtils::SetReuseInput(*output_tensordesc, true);
    uint32_t reuse_input_index = 0;
    ge::TensorUtils::SetReuseInputIndex(*output_tensordesc, reuse_input_index);

    ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 0, type, 2048);
    op_def_e->SetStreamId(1);

    auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
    auto desc_temp = *desc_temp_ptr;
    TensorUtils::SetSize(desc_temp, 1024);
    op_def_d->AddInputDesc(desc_temp);

    ge::OpDescPtr op_def_f = CreateOpWithWsSize("F", 0, type, 2048);
    op_def_f->SetStreamId(1);

    // add node
    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);
    ge::NodePtr node_e = graph->AddNode(op_def_e);
    ge::NodePtr node_f = graph->AddNode(op_def_f);

    // add edge
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_d->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_f->GetInDataAnchor(0));
    if (d_before_f) {
      ge::GraphUtils::AddEdge(node_d->GetOutControlAnchor(), node_f->GetInControlAnchor());
    }
    graph->TopologicalSorting();
  }

  //       data
  //        |
  //        A:1(s:0) (two out data anchor, but only has one output node)
  //        |
  //        B:2(s:0)  C:3(s:1)
  //        \         /
  //          D:4(s:0) (nopadding continuous input)
  //
  void MakeDiffStreamSameOutGraphWithNoPaddingContinousInput2(ge::ComputeGraphPtr &graph, const string &type = "some") {
    ge::OpDescPtr op_def_data = CreateOpWithWsSize("data", 0, DATA, 2048);
    op_def_data->AddOutputDesc(op_def_data->GetOutputDesc(0));
    op_def_data->SetStreamId(0);
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 0, type, 2048);
    op_def_a->AddOutputDesc(op_def_a->GetOutputDesc(0));
    op_def_a->SetStreamId(0);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 0, type, 2048);
    op_def_b->SetStreamId(0);
    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 0, type, 2048);
    op_def_c->SetStreamId(1);
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 0, type, 4096);
    op_def_d->SetStreamId(0);

    ge::AttrUtils::SetBool(op_def_d, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
    (void)ge::AttrUtils::SetInt(op_def_d, ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);
    ge::AttrUtils::SetBool(op_def_d, ATTR_NAME_OUTPUT_REUSE_INPUT, true);
    auto output_tensordesc = op_def_d->MutableOutputDesc(0);
    ge::TensorUtils::SetReuseInput(*output_tensordesc, true);
    uint32_t reuse_input_index = 0;
    ge::TensorUtils::SetReuseInputIndex(*output_tensordesc, reuse_input_index);

    auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
    auto desc_temp = *desc_temp_ptr;
    TensorUtils::SetSize(desc_temp, 1024);
    op_def_d->AddInputDesc(desc_temp);

    // add node
    ge::NodePtr node_data = graph->AddNode(op_def_data);
    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);

    // add edge
    ge::GraphUtils::AddEdge(node_data->GetOutDataAnchor(0), node_a->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_d->GetInDataAnchor(1));

    graph->TopologicalSorting();
  }

  //
  //        E
  //        |
  //        A:1(s:0)--+ (two out data anchor, but only has one output node)
  //        |         |(ctrl)
  //        B:2(s:0)  C:3(s:1)
  //        \         /
  //          D:4(s:0) (nopadding continuous input)
  //
  void MakeDiffStreamSameOutGraphWithNoPaddingContinousInput3(ge::ComputeGraphPtr &graph, const string &type = "some") {
    ge::OpDescPtr op_def_data = CreateOpWithWsSize("E", 0, type, 2048);
    op_def_data->AddOutputDesc(op_def_data->GetOutputDesc(0));
    op_def_data->SetStreamId(0);
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 0, type, 2048);
    op_def_a->AddOutputDesc(op_def_a->GetOutputDesc(0));
    op_def_a->SetStreamId(0);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 0, type, 2048);
    op_def_b->SetStreamId(0);
    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 0, type, 2048);
    op_def_c->SetStreamId(1);
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 0, type, 4096);
    op_def_d->SetStreamId(0);

    ge::AttrUtils::SetBool(op_def_d, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
    (void)ge::AttrUtils::SetInt(op_def_d, ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);
    ge::AttrUtils::SetBool(op_def_d, ATTR_NAME_OUTPUT_REUSE_INPUT, true);
    auto output_tensordesc = op_def_d->MutableOutputDesc(0);
    ge::TensorUtils::SetReuseInput(*output_tensordesc, true);
    uint32_t reuse_input_index = 0;
    ge::TensorUtils::SetReuseInputIndex(*output_tensordesc, reuse_input_index);

    auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
    auto desc_temp = *desc_temp_ptr;
    TensorUtils::SetSize(desc_temp, 1024);
    op_def_d->AddInputDesc(desc_temp);

    // add node
    ge::NodePtr node_data = graph->AddNode(op_def_data);
    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);

    // add edge
    ge::GraphUtils::AddEdge(node_data->GetOutDataAnchor(0), node_a->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_d->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_a->GetOutControlAnchor(), node_c->GetInControlAnchor() );

    graph->TopologicalSorting();
  }

  //        G:0(s:0)
  //        |
  //        A:1(s:0)--+ (two out data anchor, but only has one output node)
  //        |         |(ctrl)
  //        |         B:2(s:1)
  //        |         |
  //        C:3(s:0)  D:4(s:1)
  //        \         /
  //          E:5(s:0) (nopadding continuous input)
  //           |
  //           F:5(s:1)
  //
  // C的block是流0，因此life begin得是流0上的id。由于D是流1，因此找in_stream_edge stream1<-stream0
  // 一共有life time[2<-1] and [5<-4]，由于2小于D(4)，因此使用边[2<-1]， 确定life_begin_为1
  //
  void MakeDiffStreamSameOutGraphWithNoPaddingContinousInput4(ge::ComputeGraphPtr &graph, const string &type = "some") {
    ge::OpDescPtr op_def_data = CreateOpWithWsSize("G", 0, type, 3072);
    op_def_data->AddOutputDesc(op_def_data->GetOutputDesc(0));
    op_def_data->SetStreamId(0);
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 0, type, 2048);
    op_def_a->SetStreamId(0);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 0, type, 2048);
    op_def_b->AddOutputDesc(op_def_b->GetOutputDesc(0));
    op_def_b->SetStreamId(1);
    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 0, type, 2048);
    op_def_c->SetStreamId(0);
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 0, type, 2048);
    op_def_d->SetStreamId(1);
    ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 0, type, 4096);
    op_def_e->SetStreamId(0);
    ge::OpDescPtr op_def_f = CreateOpWithWsSize("F", 0, type, 1024);
    op_def_f->SetStreamId(1);

    ge::AttrUtils::SetBool(op_def_e, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
    (void)ge::AttrUtils::SetInt(op_def_e, ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);
    ge::AttrUtils::SetBool(op_def_e, ATTR_NAME_OUTPUT_REUSE_INPUT, true);
    auto output_tensordesc = op_def_e->MutableOutputDesc(0);
    ge::TensorUtils::SetReuseInput(*output_tensordesc, true);
    uint32_t reuse_input_index = 0;
    ge::TensorUtils::SetReuseInputIndex(*output_tensordesc, reuse_input_index);

    auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
    auto desc_temp = *desc_temp_ptr;
    TensorUtils::SetSize(desc_temp, 1024);
    op_def_e->AddInputDesc(desc_temp);

    // add node
    ge::NodePtr node_data = graph->AddNode(op_def_data);
    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);
    ge::NodePtr node_e = graph->AddNode(op_def_e);
    ge::NodePtr node_f = graph->AddNode(op_def_f);

    // add edge
    ge::GraphUtils::AddEdge(node_data->GetOutDataAnchor(0), node_a->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_e->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_f->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutControlAnchor(), node_b->GetInControlAnchor());

    graph->TopologicalSorting();
  }


  /**     a:1
   *    |       \
   *   b:0       c:1
   *     \     /     \
   *   StreamMerge:2  e:1
   *                  |
   *                 f:1
   *  b and f可能并发执行，f不能复用b，而b和c使用统一块内存，所以f不能复用c
   */
  void MakeDiffStreamSameOutGraphWithStreamMerge(ge::ComputeGraphPtr &graph, const string &type = "some") {
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 0, type, 2048);
    op_def_a->SetStreamId(1);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 0, type, 2048);
    op_def_b->SetStreamId(0);
    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 0, type, 2048);
    op_def_c->SetStreamId(1);
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("StreamMerge", 0, "StreamMerge", 2048);
    op_def_d->SetStreamId(2);

    ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 0, type, 2048);
    op_def_e->SetStreamId(1);

    auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
    auto desc_temp = *desc_temp_ptr;
    TensorUtils::SetSize(desc_temp, 1024);
    op_def_d->AddInputDesc(desc_temp);

    ge::OpDescPtr op_def_f = CreateOpWithWsSize("F", 0, type, 2048);
    op_def_f->SetStreamId(1);

    // add node
    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);
    ge::NodePtr node_e = graph->AddNode(op_def_e);
    ge::NodePtr node_f = graph->AddNode(op_def_f);

    // add edge
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_d->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_f->GetInDataAnchor(0));

    graph->TopologicalSorting();
  }
  void MakeDiffStreamGraphForLifeTime(ge::ComputeGraphPtr &graph, const string &type = "some") {
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 0, "Data", 2048);

    // b
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 1024, type, 2048);
    op_def_b->SetStreamId(0);

    // d
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 1024, type, 2048);
    op_def_d->SetStreamId(1);
    ge::AttrUtils::SetBool(op_def_d, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
    (void)ge::AttrUtils::SetBool(op_def_d, ATTR_NAME_OUTPUT_REUSE_INPUT, true);
    (void)ge::AttrUtils::SetInt(op_def_d, ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);
    auto output_tensordesc = op_def_d->MutableOutputDesc(0);
    ge::TensorUtils::SetReuseInput(*output_tensordesc, true);
    uint32_t reuse_input_index = 0;
    ge::TensorUtils::SetReuseInputIndex(*output_tensordesc, reuse_input_index);

    // e
    ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 1024, type, 2048);
    op_def_e->SetStreamId(1);

    // f
    ge::OpDescPtr op_def_f = CreateOpWithWsSize("F", 1024, type, 2048);
    op_def_f->SetStreamId(1);

    // add node
    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_d = graph->AddNode(op_def_d);
    ge::NodePtr node_e = graph->AddNode(op_def_e);
    ge::NodePtr node_f = graph->AddNode(op_def_f);

    /**
     *      b:0         // b: stream 0;
     *       |
     *       d:1      // d: stream 1 reuse input 0
     *        |
     *       e:1      // e: stream 1
     *        |
     *       f:1      // f: stream 1
     */
    // add edge
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_f->GetInDataAnchor(0));
    graph->TopologicalSorting();
  }

  void MakeMultiDiffStreamGraph(ge::ComputeGraphPtr &graph, const string &type = "some") {
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 1024, type, 2048);
    op_def_a->SetStreamId(1);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 1024, type, 4096);
    op_def_b->SetStreamId(0);
    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 1024, type, 2048);
    op_def_c->SetStreamId(2);
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 1024, type, 2048);
    op_def_d->SetStreamId(1);
    auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
    auto desc_temp = *desc_temp_ptr;
    TensorUtils::SetSize(desc_temp, 2048);
    op_def_d->AddInputDesc(desc_temp);

    ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 1024, type, 2048);
    op_def_e->SetStreamId(1);
    op_def_e->AddOutputDesc(desc_temp);
    op_def_e->AddOutputDesc(desc_temp);
    op_def_e->AddOutputDesc(desc_temp);

    ge::OpDescPtr op_def_f = CreateOpWithWsSize("F", 1024, type, 2048);
    op_def_f->SetStreamId(1);
    op_def_f->AddInputDesc(desc_temp);
    op_def_f->AddInputDesc(desc_temp);
    op_def_f->AddInputDesc(desc_temp);

    // add node
    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);
    ge::NodePtr node_e = graph->AddNode(op_def_e);
    ge::NodePtr node_f = graph->AddNode(op_def_f);

    /**
     *      a:1
     *    |     \
     *   b:0    c:2
     *     \    /
     *      d:1
     *        |
     *      e:1
     *      /|\
     *      f:1
     */
    // add edge
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_d->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_f->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(1), node_f->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(2), node_f->GetInDataAnchor(2));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(3), node_f->GetInDataAnchor(3));
    graph->TopologicalSorting();
  }

  void MakeMultiDiffStreamGraph1(ge::ComputeGraphPtr &graph, const string &type = "some") {
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 1024, type, 2048);
    op_def_a->SetStreamId(1);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 1024, type, 4096);
    op_def_b->SetStreamId(0);
    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 1024, type, 2048);
    op_def_c->SetStreamId(2);
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 1024, type, 2048);
    op_def_d->SetStreamId(2);
    auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
    auto desc_temp = *desc_temp_ptr;
    TensorUtils::SetSize(desc_temp, 2048);
    op_def_d->AddInputDesc(desc_temp);

    ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 1024, type, 2048);
    op_def_e->SetStreamId(2);
    op_def_e->AddOutputDesc(desc_temp);
    op_def_e->AddOutputDesc(desc_temp);
    op_def_e->AddOutputDesc(desc_temp);

    ge::OpDescPtr op_def_f = CreateOpWithWsSize("F", 1024, type, 2048);
    op_def_f->SetStreamId(2);
    op_def_f->AddInputDesc(desc_temp);
    op_def_f->AddInputDesc(desc_temp);
    op_def_f->AddInputDesc(desc_temp);

    // add node
    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);
    ge::NodePtr node_e = graph->AddNode(op_def_e);
    ge::NodePtr node_f = graph->AddNode(op_def_f);

    /**
     *      a:1
     *    |     \
     *   b:1    c:2
     *     \    /
     *      d:2
     *        |
     *      e:2
     *      /|\
     *      f:2
     */
    // add edge
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_d->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_f->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(1), node_f->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(2), node_f->GetInDataAnchor(2));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(3), node_f->GetInDataAnchor(3));
    graph->TopologicalSorting();
  }

  void MakeDiffStreamDependenceGraph(ge::ComputeGraphPtr &graph, const string &type = "some") {
    auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
    auto desc_temp = *desc_temp_ptr;
    TensorUtils::SetSize(desc_temp, 2048);

    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 1024, type, 2048);
    op_def_a->SetStreamId(0);
    op_def_a->AddOutputDesc(desc_temp);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 1024, type, 4608);
    op_def_b->SetStreamId(0);
    op_def_b->AddOutputDesc(desc_temp);
    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 1024, type, 2048);
    op_def_c->SetStreamId(0);
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 1024, type, 2048);
    op_def_d->SetStreamId(0);
    ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 1024, type, 2048);
    op_def_e->SetStreamId(0);

    ge::OpDescPtr op_def_f = CreateOpWithWsSize("F", 1024, type, 2048);
    op_def_f->SetStreamId(1);
    ge::OpDescPtr op_def_g = CreateOpWithWsSize("G", 1024, type, 4096);
    op_def_g->SetStreamId(1);
    op_def_g->AddInputDesc(desc_temp);
    ge::OpDescPtr op_def_h = CreateOpWithWsSize("H", 1024, type, 2048);
    op_def_h->SetStreamId(1);
    op_def_h->AddInputDesc(desc_temp);
    ge::OpDescPtr op_def_i = CreateOpWithWsSize("I", 1024, type, 4608);
    op_def_i->SetStreamId(1);
    ge::OpDescPtr op_def_j = CreateOpWithWsSize("J", 1024, type, 2048);
    op_def_j->SetStreamId(1);


    // add node
    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);
    ge::NodePtr node_e = graph->AddNode(op_def_e);
    ge::NodePtr node_f = graph->AddNode(op_def_f);
    ge::NodePtr node_g = graph->AddNode(op_def_g);
    ge::NodePtr node_h = graph->AddNode(op_def_h);
    ge::NodePtr node_i = graph->AddNode(op_def_i);
    ge::NodePtr node_j = graph->AddNode(op_def_j);

    /**
     *      a:0    j:1
     *      | \    ⬆
     *     b:0 \   i:1
     *      |  \\  ⬆
     *     c:   -\-h:1
     *      |     \ ⬆
     *     d:0     g:1
     *      |      ⬆
     *     e:0---f:1
     */
    // add edge
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_f->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_f->GetOutDataAnchor(0), node_g->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(1), node_g->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_g->GetOutDataAnchor(0), node_h->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(1), node_h->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_h->GetOutDataAnchor(0), node_i->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_i->GetOutDataAnchor(0), node_j->GetInDataAnchor(0));
    graph->TopologicalSorting();
  }

  void MakeRefContinuousInputGraph(ge::ComputeGraphPtr &graph, const string &type = "some") {
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 1024, type, 2048);
    op_def_a->SetStreamId(1);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 1024, type, 2048);
    op_def_b->SetStreamId(1);
    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 1024, type, 2048);
    op_def_c->SetStreamId(1);
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 1024, type, 2048);
    op_def_d->SetStreamId(1);
    ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 1024, type, 2048);
    op_def_e->SetStreamId(1);
    ge::OpDescPtr op_def_f = CreateOpWithWsSize("F", 1024, type, 2048);
    op_def_f->SetStreamId(1);
    ge::OpDescPtr op_def_g = CreateOpWithWsSize("G", 1024, type, 10240);
    op_def_g->SetStreamId(1);
    ge::OpDescPtr op_def_h = CreateOpWithWsSize("H", 1024, type);
    op_def_h->SetStreamId(1);

    auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
    auto desc_temp = *desc_temp_ptr;
    TensorUtils::SetSize(desc_temp, 1024);
    op_def_h->AddInputDesc(desc_temp);
    op_def_h->AddInputDesc(desc_temp);
    auto output_tensordesc = op_def_h->MutableOutputDesc(0);
    ge::TensorUtils::SetReuseInput(*output_tensordesc, true);
    uint32_t reuse_input_index = 1;
    ge::TensorUtils::SetReuseInputIndex(*output_tensordesc, reuse_input_index);

    ge::OpDescPtr op_def_i = CreateOpWithWsSize("I", 1024, type, 2048);
    op_def_i->SetStreamId(1);

    ge::OpDescPtr op_def_j = CreateOpWithWsSize("J", 1024, type, 2048);
    op_def_j->SetStreamId(1);
    ge::AttrUtils::SetBool(op_def_j, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
    (void)ge::AttrUtils::SetBool(op_def_j, ATTR_NAME_OUTPUT_REUSE_INPUT, true);
    (void)ge::AttrUtils::SetInt(op_def_j, ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);
    output_tensordesc = op_def_j->MutableOutputDesc(0);
    ge::TensorUtils::SetReuseInput(*output_tensordesc, true);
    reuse_input_index = 0;
    ge::TensorUtils::SetReuseInputIndex(*output_tensordesc, reuse_input_index);
    desc_temp = *desc_temp_ptr;
    TensorUtils::SetSize(desc_temp, 1024);
    op_def_j->AddInputDesc(desc_temp);
    op_def_j->AddInputDesc(desc_temp);

    // add node
    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);
    ge::NodePtr node_e = graph->AddNode(op_def_e);
    ge::NodePtr node_f = graph->AddNode(op_def_f);
    ge::NodePtr node_g = graph->AddNode(op_def_g);
    ge::NodePtr node_h = graph->AddNode(op_def_h);
    ge::NodePtr node_i = graph->AddNode(op_def_i);
    ge::NodePtr node_j = graph->AddNode(op_def_j);

    /**
     *
     *           a    b   c
     *           |    |   |
     *           d    e   f
     *           |___|___|
     *               |
     *          g    h   i
     *          |___|___|
     *              |
     *              j
     */
    // h and j are nopading continuous input, no need to alloc a's memory

    // add edge
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_f->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_h->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_h->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_f->GetOutDataAnchor(0), node_h->GetInDataAnchor(2));
    ge::GraphUtils::AddEdge(node_g->GetOutDataAnchor(0), node_j->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_h->GetOutDataAnchor(0), node_j->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_i->GetOutDataAnchor(0), node_j->GetInDataAnchor(2));
    graph->TopologicalSorting();
  }

  void MakeCascadeContinuousGraph(ge::ComputeGraphPtr &graph, const string &type = "some") {
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 1024, type, 2048);
    op_def_a->SetStreamId(1);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 1024, type, 2048);
    op_def_b->SetStreamId(1);
    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 1024, type, 2048);
    op_def_c->SetStreamId(1);
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 1024, type, 2048);
    op_def_d->SetStreamId(1);
    ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 1024, type, 2048);
    op_def_e->SetStreamId(1);
    ge::OpDescPtr op_def_f = CreateOpWithWsSize("F", 1024, type, 2048);
    op_def_f->SetStreamId(1);
    ge::OpDescPtr op_def_g = CreateOpWithWsSize("G", 1024, type, 10240);
    op_def_g->SetStreamId(1);
    ge::OpDescPtr op_def_h = CreateOpWithWsSize("H", 1024, type, 2048);
    op_def_h->SetStreamId(1);
    ge::AttrUtils::SetBool(op_def_h, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
    (void)ge::AttrUtils::SetBool(op_def_h, ATTR_NAME_OUTPUT_REUSE_INPUT, true);
    (void)ge::AttrUtils::SetInt(op_def_h, ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);
    auto output_tensordesc = op_def_h->MutableOutputDesc(0);
    ge::TensorUtils::SetReuseInput(*output_tensordesc, true);
    uint32_t reuse_input_index = 0;
    ge::TensorUtils::SetReuseInputIndex(*output_tensordesc, reuse_input_index);

    auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
    auto desc_temp = *desc_temp_ptr;
    TensorUtils::SetSize(desc_temp, 1024);
    op_def_h->AddInputDesc(desc_temp);
    op_def_h->AddInputDesc(desc_temp);

    ge::OpDescPtr op_def_i = CreateOpWithWsSize("I", 1024, type, 2048);
    op_def_i->SetStreamId(1);

    ge::OpDescPtr op_def_j = CreateOpWithWsSize("J", 1024, type, 2048);
    op_def_j->SetStreamId(1);
    ge::AttrUtils::SetBool(op_def_j, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
    (void)ge::AttrUtils::SetBool(op_def_j, ATTR_NAME_OUTPUT_REUSE_INPUT, true);
    (void)ge::AttrUtils::SetInt(op_def_j, ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);
    output_tensordesc = op_def_j->MutableOutputDesc(0);
    ge::TensorUtils::SetReuseInput(*output_tensordesc, true);
    reuse_input_index = 0;
    ge::TensorUtils::SetReuseInputIndex(*output_tensordesc, reuse_input_index);
    desc_temp = *desc_temp_ptr;
    TensorUtils::SetSize(desc_temp, 1024);
    op_def_j->AddInputDesc(desc_temp);
    op_def_j->AddInputDesc(desc_temp);

    // add node
    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);
    ge::NodePtr node_e = graph->AddNode(op_def_e);
    ge::NodePtr node_f = graph->AddNode(op_def_f);
    ge::NodePtr node_g = graph->AddNode(op_def_g);
    ge::NodePtr node_h = graph->AddNode(op_def_h);
    ge::NodePtr node_i = graph->AddNode(op_def_i);
    ge::NodePtr node_j = graph->AddNode(op_def_j);

    /**
     *
     *           a    b   c
     *           |    |   |
     *           d    e   f
     *           |___|___|
     *               |
     *          g    h   i
     *          |___|___|
     *              |
     *              j
     */
    // h and j are nopading continuous input, no need to alloc a's memory

    // add edge
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_f->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_h->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_h->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_f->GetOutDataAnchor(0), node_h->GetInDataAnchor(2));
    ge::GraphUtils::AddEdge(node_g->GetOutDataAnchor(0), node_j->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_h->GetOutDataAnchor(0), node_j->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_i->GetOutDataAnchor(0), node_j->GetInDataAnchor(2));
    graph->TopologicalSorting();
  }

  void MakeDiffStreamRefMemGraph(ge::ComputeGraphPtr &graph, const string &type = "some", bool diff_stream=true) {
    auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
    auto desc_temp = *desc_temp_ptr;
    TensorUtils::SetSize(desc_temp, 2048);

    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 1024, type, 2048);
    op_def_a->SetStreamId(1);
    op_def_a->AddOutputDesc(desc_temp);

    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 1024, type, 2048);
    op_def_b->SetStreamId(0);
    ge::AttrUtils::SetBool(op_def_b, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);

    (void)ge::AttrUtils::SetBool(op_def_b, ATTR_NAME_OUTPUT_REUSE_INPUT, true);
    (void)ge::AttrUtils::SetInt(op_def_b, ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);
    auto output_tensordesc = op_def_b->MutableOutputDesc(0);
    ge::TensorUtils::SetReuseInput(*output_tensordesc, true);
    uint32_t reuse_input_index = 0;
    ge::TensorUtils::SetReuseInputIndex(*output_tensordesc, reuse_input_index);

    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 1024, type, 2048);
    op_def_c->SetStreamId(0);
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 1024, type, 2048);
    op_def_d->SetStreamId(1);
    op_def_d->AddOutputDesc(desc_temp);

    ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 1024, type, 2048);
    op_def_e->SetStreamId(0);
    ge::AttrUtils::SetBool(op_def_e, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
    (void)ge::AttrUtils::SetBool(op_def_e, ATTR_NAME_OUTPUT_REUSE_INPUT, true);
    (void)ge::AttrUtils::SetInt(op_def_e, ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);
    auto output_tensordesc_e = op_def_e->MutableOutputDesc(0);
    ge::TensorUtils::SetReuseInput(*output_tensordesc_e, true);
    uint32_t reuse_input_index_e = 0;
    ge::TensorUtils::SetReuseInputIndex(*output_tensordesc_e, reuse_input_index_e);

    ge::OpDescPtr op_def_f = CreateOpWithWsSize("F", 1024, type, 2048);
    op_def_f->SetStreamId(0);
    ge::OpDescPtr op_def_g = CreateOpWithWsSize("G", 1024, type, 2048);
    int64_t stream_id = diff_stream ? 1 : 0;
    op_def_g->SetStreamId(stream_id);
    op_def_g->AddInputDesc(desc_temp);

    ge::OpDescPtr op_def_h = CreateOpWithWsSize("H", 1024, type, 2048);
    op_def_h->SetStreamId(0);
    ge::OpDescPtr op_def_i = CreateOpWithWsSize("I", 1024, type, 2048);
    op_def_i->SetStreamId(0);

    // add node
    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);
    ge::NodePtr node_e = graph->AddNode(op_def_e);
    ge::NodePtr node_f = graph->AddNode(op_def_f);
    ge::NodePtr node_g = graph->AddNode(op_def_g);
    ge::NodePtr node_h = graph->AddNode(op_def_h);
    ge::NodePtr node_i = graph->AddNode(op_def_i);

    /**
     *  a ....>  d
     *  | \     / |
     *  b  h   i  e
     *  |        |
     *  c        f
     *   \      /
     *      g
     */
    // b and e output ref input
    // add edge
    ge::GraphUtils::AddEdge(node_a->GetOutControlAnchor(), node_d->GetInControlAnchor());
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(1), node_h->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_g->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_i->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_f->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_f->GetOutDataAnchor(0), node_g->GetInDataAnchor(1));
    graph->TopologicalSorting();
  }

  void MakeFirstNotMaxLifeContinuousGraph(ge::ComputeGraphPtr &graph, const string &type = "some") {
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 1024, type, 2048);
    op_def_a->SetStreamId(1);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 1024, type, 2048);
    op_def_b->SetStreamId(1);
    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 1024, type, 2048);
    op_def_c->SetStreamId(1);
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 1024, type, 2048);
    op_def_d->SetStreamId(1);
    ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 1024, type, 2048);
    op_def_e->SetStreamId(1);
    ge::OpDescPtr op_def_f = CreateOpWithWsSize("F", 1024, type, 2048);
    op_def_f->SetStreamId(1);
    ge::OpDescPtr op_def_h = CreateOpWithWsSize("H", 1024, type, 6144);
    op_def_h->SetStreamId(1);
    ge::AttrUtils::SetBool(op_def_h, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
    (void)ge::AttrUtils::SetBool(op_def_h, ATTR_NAME_OUTPUT_REUSE_INPUT, true);
    (void)ge::AttrUtils::SetInt(op_def_h, ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);
    auto output_tensordesc = op_def_h->MutableOutputDesc(0);
    ge::TensorUtils::SetReuseInput(*output_tensordesc, true);
    uint32_t reuse_input_index = 0;
    ge::TensorUtils::SetReuseInputIndex(*output_tensordesc, reuse_input_index);

    auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
    auto desc_temp = *desc_temp_ptr;
    TensorUtils::SetSize(desc_temp, 1024);
    op_def_h->AddInputDesc(desc_temp);
    op_def_h->AddInputDesc(desc_temp);

    ge::OpDescPtr op_def_i = CreateOpWithWsSize("I", 1024, type, 2048);
    op_def_i->SetStreamId(1);

    ge::OpDescPtr op_def_j = CreateOpWithWsSize("J", 1024, type, 2048);
    op_def_j->SetStreamId(1);
    op_def_j->AddInputDesc(desc_temp);

    // add node
    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);
    ge::NodePtr node_e = graph->AddNode(op_def_e);
    ge::NodePtr node_f = graph->AddNode(op_def_f);
    ge::NodePtr node_h = graph->AddNode(op_def_h);
    ge::NodePtr node_i = graph->AddNode(op_def_i);
    ge::NodePtr node_j = graph->AddNode(op_def_j);

    /**
     *           a   b   c
     *           |   |   |
     *           d   e   f
     *           |___|___|
     *               |   |
     *               h   |
     *               |   |
     *               i   |
     *               |   |
     *               j___|
     */
    // d's max life is not h

    // add edge
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_f->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_h->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_h->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_f->GetOutDataAnchor(0), node_h->GetInDataAnchor(2));
    ge::GraphUtils::AddEdge(node_f->GetOutDataAnchor(0), node_j->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_h->GetOutDataAnchor(0), node_i->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_i->GetOutDataAnchor(0), node_j->GetInDataAnchor(0));
    graph->TopologicalSorting();
  }

  void MakeNotFistContinuousInputAsSymbolGraph(ge::ComputeGraphPtr &graph, const string &type = "some",
                                               bool muti_ref = false, bool diff_stream = false,
                                               bool is_first_input_muti_ref = true, bool is_second_input_ref = false,
                                               bool is_no_padding = true) {
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 1024, type, 2048);
    op_def_a->SetStreamId(1);

    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 1024, type, 2048);
    op_def_b->SetStreamId(1);
    (void)ge::AttrUtils::SetInt(op_def_b, ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);
    auto b_output_tensordesc = op_def_b->MutableOutputDesc(0);
    ge::TensorUtils::SetReuseInput(*b_output_tensordesc, true);
    uint32_t reuse_input_index = 0;
    ge::TensorUtils::SetReuseInputIndex(*b_output_tensordesc, reuse_input_index);

    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 1024, type, 2048);
    op_def_c->SetStreamId(1);
    //
    if (is_second_input_ref) {
      (void)ge::AttrUtils::SetInt(op_def_c, ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);
      auto c_output_tensordesc = op_def_c->MutableOutputDesc(0);
      ge::TensorUtils::SetReuseInput(*c_output_tensordesc, true);
      ge::TensorUtils::SetReuseInputIndex(*c_output_tensordesc, reuse_input_index);
    }

    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 1024, type, 2048);
    op_def_d->SetStreamId(1);
    if (is_no_padding) {
      ge::AttrUtils::SetBool(op_def_d, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
      (void)ge::AttrUtils::SetBool(op_def_d, ATTR_NAME_OUTPUT_REUSE_INPUT, true);
      (void)ge::AttrUtils::SetInt(op_def_d, ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);
      auto d_output_tensordesc = op_def_d->MutableOutputDesc(0);
      ge::TensorUtils::SetReuseInput(*d_output_tensordesc, true);
      ge::TensorUtils::SetReuseInputIndex(*d_output_tensordesc, reuse_input_index);
    } else {
      ge::AttrUtils::SetBool(op_def_d, ATTR_NAME_CONTINUOUS_INPUT, true);
      ge::AttrUtils::SetBool(op_def_d, "need_gentask_atomic", true);
      std::vector<int32_t> input_indexes = {-1};
      (void)ge::AttrUtils::SetListInt(op_def_d, ATOMIC_ATTR_INPUT_INDEX, input_indexes);
    }


    ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 1024, type, 2048);
    int64_t stream_id = diff_stream ? 0 : 1;
    op_def_e->SetStreamId(stream_id);
    ge::OpDescPtr op_def_f = CreateOpWithWsSize("F", 1024, type, 2048);
    op_def_f->SetStreamId(stream_id);
    ge::OpDescPtr op_def_g = CreateOpWithWsSize("G", 1024, type, 2048);
    op_def_g->SetStreamId(stream_id);
    ge::OpDescPtr op_def_h = CreateOpWithWsSize("H", 1024, type, 2048);
    op_def_h->SetStreamId(stream_id);

    auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
    auto desc_temp = *desc_temp_ptr;
    TensorUtils::SetSize(desc_temp, 1024);
    op_def_d->AddInputDesc(desc_temp);
    if (muti_ref) {
      op_def_f->AddInputDesc(desc_temp);
    }

    // add node
    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);
    ge::NodePtr node_e = graph->AddNode(op_def_e);
    ge::NodePtr node_f = graph->AddNode(op_def_f);
    ge::NodePtr node_g = graph->AddNode(op_def_g);
    ge::NodePtr node_h = graph->AddNode(op_def_h);

    // add edge
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));
    if (muti_ref){
      if (!is_first_input_muti_ref) {
        ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_f->GetInDataAnchor(1));
      } else {
        ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_f->GetInDataAnchor(1));
      }
    }
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_d->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_f->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_f->GetOutDataAnchor(0), node_g->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_g->GetOutDataAnchor(0), node_h->GetInDataAnchor(0));
    graph->TopologicalSorting();
  }

  void MakeReuseGraph(ge::ComputeGraphPtr graph) {
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 6000);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 120000);
    ge::OpDescPtr op_def_c = CreateRefOpWithWsSize("C", 120000);
    ge::OpDescPtr op_def_d = std::make_shared<ge::OpDesc>("D", "CONSTANT");

    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);

    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    domi::GetContext().out_nodes_map["B"] = {0};
    domi::GetContext().out_nodes_map["C"] = {0};
    graph->TopologicalSorting();
  }

  /*
   *   A
   *   |
   *   B
   *   |
   *   C (REF)
   *   |
   *   D (REF)
   *   |
   *   E
   *   |
   *   F
   *
   */
  void MakeRefReuseGraph(ge::ComputeGraphPtr graph) {
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 1024, "some", 6656);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 1024);
    ge::OpDescPtr op_def_c = CreateRefOpWithWsSize("C", 1024);
    ge::OpDescPtr op_def_d = CreateRefOpWithWsSize("D", 1024);
    ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 1024);
    ge::OpDescPtr op_def_f = CreateOpWithWsSize("F", 1024);

    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);
    ge::NodePtr node_e = graph->AddNode(op_def_e);
    ge::NodePtr node_f = graph->AddNode(op_def_f);

    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_f->GetInDataAnchor(0));
    graph->TopologicalSorting();
  }

  ComputeGraphPtr MakeCascadeContinuousMemoryGraph() {
    ge::ut::GraphBuilder builder("graph");
    auto data = builder.AddNode("data", "Data", 1, 1);
    auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);
    auto addn2 = builder.AddNode("addn2", "AddN", 1, 1);
    auto addn3 = builder.AddNode("addn3", "AddN", 1, 1);
    auto concat1 = builder.AddNode("concat1", "Concat", 2, 1);
    auto concat2 = builder.AddNode("concat2", "Concat", 2, 1);
    auto netoutput = builder.AddNode("netoutput", "NetOutput", 2, 0);

    ge::AttrUtils::SetBool(concat1->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
    (void)ge::AttrUtils::SetBool(concat1->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
    ge::AttrUtils::SetBool(concat1->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);

    ge::AttrUtils::SetBool(concat2->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
    (void)ge::AttrUtils::SetBool(concat2->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
    ge::AttrUtils::SetBool(concat2->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);

    addn1->GetOpDesc()->SetOutputOffset({100});
    addn2->GetOpDesc()->SetOutputOffset({200});
    concat1->GetOpDesc()->SetOutputOffset({100});
    addn3->GetOpDesc()->SetOutputOffset({700});
    concat2->GetOpDesc()->SetOutputOffset({500});

    ge::AttrUtils::SetListInt(addn1->GetOpDesc(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, {100});
    ge::AttrUtils::SetListInt(addn2->GetOpDesc(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, {100});
    ge::AttrUtils::SetListInt(addn3->GetOpDesc(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, {100});
    ge::AttrUtils::SetListInt(concat1->GetOpDesc(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, {200});
    ge::AttrUtils::SetListInt(concat2->GetOpDesc(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, {300});


    builder.AddDataEdge(data, 0, addn1, 0);
    builder.AddDataEdge(data, 0, addn2, 0);
    builder.AddDataEdge(addn1, 0, concat1, 0);
    builder.AddDataEdge(addn2, 0, concat1, 1);
    builder.AddDataEdge(concat1, 0, concat2, 0);
    builder.AddDataEdge(addn3, 0, concat2, 1);

    return builder.GetGraph();
  }

  ComputeGraphPtr MakeRefNodeGraph() {
    ge::ut::GraphBuilder builder("graph");
    auto var_input = builder.AddNode("var", "Variable", 1, 1);
    auto const_input = builder.AddNode("const", "Const", 1, 1);
    auto assign = builder.AddNode("assgin", "Assign", 2, 1);
     // add link
    builder.AddDataEdge(var_input, 0, assign, 0);
    builder.AddDataEdge(const_input, 0, assign, 1);
    // set offset
    assign->GetOpDesc()->SetInputOffset({100, 0});
    assign->GetOpDesc()->SetOutputOffset({10000});
    var_input->GetOpDesc()->SetOutputOffset({10000});
    const_input->GetOpDesc()->SetOutputOffset({1000});
    // set mem type
    ge::AttrUtils::SetListInt(assign->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, {RT_MEMORY_HBM, RT_MEMORY_L1});
    // set ref
    auto output_tensordesc = assign->GetOpDesc()->MutableOutputDesc(0);
    ge::TensorUtils::SetReuseInput(*output_tensordesc, true);
    uint32_t reuse_input_index = 0;
    ge::TensorUtils::SetReuseInputIndex(*output_tensordesc, reuse_input_index);

    return builder.GetGraph();
  }

  ComputeGraphPtr MakeAtomicGraph() {
    ge::ut::GraphBuilder builder("test_graph");
    auto data = builder.AddNode("data", "Data", 1, 1, FORMAT_NCHW, DT_FLOAT, {8, 3, 224, 224});
    auto addn1 = builder.AddNode("addn1", "AddN", 1, 1, FORMAT_NCHW, DT_FLOAT, {1, 3, 224, 224});
    auto addn2 = builder.AddNode("addn2", "AddN", 1, 1, FORMAT_NCHW, DT_FLOAT, {8, 3, 224, 224});
    auto merge = builder.AddNode("merge", "Merge", 2, 1);
    auto netouput = builder.AddNode("netoutput", "NetOutput", 1, 0);
    vector<int64_t> output_list;
    output_list.emplace_back(0);
    auto addn1_output_desc = addn1->GetOpDesc()->GetOutputDesc(0);
    // 1*3*224*224*3
    int64_t addn1_output_size = 602112;
    ge::TensorUtils::SetSize(addn1_output_desc, addn1_output_size);
    addn1->GetOpDesc()->UpdateOutputDesc(0, addn1_output_desc);
    addn1->GetOpDesc()->SetOutputOffset(output_list);

    auto addn2_output_desc = addn2->GetOpDesc()->GetOutputDesc(0);
    //8*3*224*224*3
    int64_t addn2_output_size = 4816896;
    ge::TensorUtils::SetSize(addn2_output_desc, addn2_output_size);
    addn2->GetOpDesc()->UpdateOutputDesc(0, addn2_output_desc);
    addn2->GetOpDesc()->SetOutputOffset(output_list);
    merge->GetOpDesc()->SetOutputOffset(output_list);
    builder.AddDataEdge(data, 0, addn1, 0);
    builder.AddDataEdge(data, 0, addn2, 0);
    builder.AddDataEdge(addn1, 0, merge, 0);
    builder.AddDataEdge(addn2, 0, merge, 1);
    builder.AddDataEdge(merge, 0, netouput, 0);

    vector<int> atomic_output_index = {0};
    (void) ge::AttrUtils::SetBool(addn1->GetOpDesc(), ATOMIC_ATTR_IS_ATOMIC_NODE, true);
    (void) ge::AttrUtils::SetListInt(addn1->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);
    auto atomic_clean = builder.AddNode("atomic_clean", "AtomicAddrClean", 0, 0);
    builder.AddControlEdge(atomic_clean, addn1);
    auto graph = builder.GetGraph();
    graph->SetInputSize(1);
    graph->SetInputsOrder({"data"});
    graph->AddInputNode(data);
    return graph;
  }

  ComputeGraphPtr MakeAtomicAndRefGraph() {
    ge::ut::GraphBuilder builder("test_graph");
    auto data = builder.AddNode("data", "Data", 1, 1, FORMAT_NCHW, DT_FLOAT, {8, 3, 224, 224});
    auto addn1 = builder.AddNode("addn1", "AddN", 1, 1, FORMAT_NCHW, DT_FLOAT, {1, 3, 224, 224});
    auto addn2 = builder.AddNode("addn2", "AddN", 1, 1, FORMAT_NCHW, DT_FLOAT, {8, 3, 224, 224});
    auto merge = builder.AddNode("merge", "Merge", 2, 1);
    auto netouput = builder.AddNode("netoutput", "NetOutput", 1, 0);
    vector<int64_t> output_list;
    output_list.emplace_back(0);
    auto addn1_output_desc = addn1->GetOpDesc()->GetOutputDesc(0);
    // 1*3*224*224*3
    int64_t addn1_output_size = 602112;
    ge::TensorUtils::SetSize(addn1_output_desc, addn1_output_size);
    addn1->GetOpDesc()->UpdateOutputDesc(0, addn1_output_desc);
    addn1->GetOpDesc()->SetOutputOffset(output_list);

    auto addn2_output_desc = addn2->GetOpDesc()->GetOutputDesc(0);
    //8*3*224*224*3
    int64_t addn2_output_size = 4816896;
    ge::TensorUtils::SetSize(addn2_output_desc, addn2_output_size);
    addn2->GetOpDesc()->UpdateOutputDesc(0, addn2_output_desc);
    addn2->GetOpDesc()->SetOutputOffset(output_list);
    merge->GetOpDesc()->SetOutputOffset(output_list);
    builder.AddDataEdge(data, 0, addn1, 0);
    builder.AddDataEdge(data, 0, addn2, 0);
    builder.AddDataEdge(addn1, 0, merge, 0);
    builder.AddDataEdge(addn2, 0, merge, 1);
    builder.AddDataEdge(merge, 0, netouput, 0);

    vector<int> atomic_output_index = {0};
    (void) ge::AttrUtils::SetBool(addn1->GetOpDesc(), ATOMIC_ATTR_IS_ATOMIC_NODE, true);
    (void) ge::AttrUtils::SetBool(addn1->GetOpDesc(), ATTR_NAME_REFERENCE, true);
    (void) ge::AttrUtils::SetListInt(addn1->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);
    auto atomic_clean = builder.AddNode("atomic_clean", "AtomicAddrClean", 0, 0);
    builder.AddControlEdge(atomic_clean, addn1);
    auto graph = builder.GetGraph();
    graph->SetInputSize(1);
    graph->SetInputsOrder({"data"});
    graph->AddInputNode(data);
    return graph;
  }

  void MakeSessionScopeReuseGraph(ge::ComputeGraphPtr graph) {
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 512);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 0);
    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 512);
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 512);
    ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 1024);
    ge::OpDescPtr op_def_f = CreateOpWithWsSize("F", 512, "some", 2048UL);
    ge::OpDescPtr op_def_g = CreateOpWithWsSize("G", 0);

    std::vector<int64_t> workspace_bytes;
    workspace_bytes.push_back(1024);
    workspace_bytes.push_back(512);
    op_def_c->SetWorkspaceBytes(workspace_bytes);
    vector<int32_t> workspace_no_reuse_scope = { 0 , 1 };
    (void)ge::AttrUtils::SetListInt(op_def_c, ATTR_NAME_WORKSPACE_MEMORY_NO_REUSE_SCOPE, workspace_no_reuse_scope);

    vector<int32_t> workspace_no_reuse_scope_e = { 1 };
    (void)ge::AttrUtils::SetListInt(op_def_e, ATTR_NAME_WORKSPACE_MEMORY_NO_REUSE_SCOPE, workspace_no_reuse_scope_e);

    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);
    ge::NodePtr node_e = graph->AddNode(op_def_e);
    ge::NodePtr node_f = graph->AddNode(op_def_f);
    ge::NodePtr node_g = graph->AddNode(op_def_g);

    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_f->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_f->GetOutDataAnchor(0), node_g->GetInDataAnchor(0));
    graph->TopologicalSorting();
  }

 void MakeContinuousReuseGraph(ge::ComputeGraphPtr graph, bool nopading = false) {
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 512);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 0);
    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 512);
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 512);
    ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 1024);
    ge::OpDescPtr op_def_f = CreateOpWithWsSize("F", 512, "some", 2048UL);
    ge::OpDescPtr op_def_g = CreateOpWithWsSize("G", 0);

    if (nopading) {
      (void)ge::AttrUtils::SetBool(op_def_d, ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
      (void)ge::AttrUtils::SetBool(op_def_d, ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
      (void)ge::AttrUtils::SetBool(op_def_d, ATTR_NAME_OUTPUT_REUSE_INPUT, true);
      (void)ge::AttrUtils::SetInt(op_def_d, ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);
    } else {
      (void)ge::AttrUtils::SetBool(op_def_d, ATTR_NAME_CONTINUOUS_INPUT, true);
      (void)ge::AttrUtils::SetBool(op_def_d, ATTR_NAME_CONTINUOUS_OUTPUT, true);
    }

    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);
    ge::NodePtr node_e = graph->AddNode(op_def_e);
    ge::NodePtr node_f = graph->AddNode(op_def_f);
    ge::NodePtr node_g = graph->AddNode(op_def_g);
    /*
     *    A B C
     *    |
     *    D
     *   /|\
     *  E F G
     */
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_f->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_g->GetInDataAnchor(0));
    graph->TopologicalSorting();
  }

void MakeContinuousReuseGraphDiffStream(ge::ComputeGraphPtr graph) {
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 512);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 0);
    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 512);
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 512);
    ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 1024);
    ge::OpDescPtr op_def_f = CreateOpWithWsSize("F", 512, "some", 2048UL);
    ge::OpDescPtr op_def_g = CreateOpWithWsSize("G", 0);

    (void)ge::AttrUtils::SetBool(op_def_d, ATTR_NAME_CONTINUOUS_INPUT, true);
    (void)ge::AttrUtils::SetBool(op_def_d, ATTR_NAME_CONTINUOUS_OUTPUT, true);
    ge::TensorUtils::SetReuseInput(*(op_def_e->MutableOutputDesc(0)), true);
    op_def_d->SetStreamId(2);
    op_def_g->SetStreamId(2);

    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);
    ge::NodePtr node_e = graph->AddNode(op_def_e);
    ge::NodePtr node_f = graph->AddNode(op_def_f);
    ge::NodePtr node_g = graph->AddNode(op_def_g);

    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_f->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_g->GetInDataAnchor(0));
    graph->TopologicalSorting();
  }

  void MakeMultiBatchReuseGraph(ge::ComputeGraphPtr graph, bool continuous = false) {
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 512);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 0);
    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 512);
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 512);
    ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 1024);
    ge::OpDescPtr op_def_f = CreateOpWithWsSize("F", 512, HCOMALLREDUCE, 2048);
    ge::OpDescPtr op_def_b1 = CreateOpWithWsSize("B1", 0, "some", 2048);
    ge::OpDescPtr op_def_c1 = CreateOpWithWsSize("C1", 512, "some", 2048);
    ge::OpDescPtr op_def_d1 = CreateOpWithWsSize("D1", 512, "some", 2048);
    ge::OpDescPtr op_def_e1 = CreateOpWithWsSize("E1", 1024, "some", 2048);
    ge::OpDescPtr op_def_f1 = CreateOpWithWsSize("F1", 512, HCOMALLREDUCE, 3072);
    ge::OpDescPtr op_def_b2 = CreateOpWithWsSize("B2", 0, "some", 3072);
    ge::OpDescPtr op_def_c2 = CreateOpWithWsSize("C2", 512, "some", 3072);
    ge::OpDescPtr op_def_d2 = CreateOpWithWsSize("D2", 512, "some", 3072);
    ge::OpDescPtr op_def_e2 = CreateOpWithWsSize("E2", 1024, "some", 3072);
    ge::OpDescPtr op_def_f2 = CreateOpWithWsSize("F2", 512, HCOMALLREDUCE, 4096);
    ge::OpDescPtr op_def_g = CreateOpWithWsSize("G", 0);

    auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
    auto desc_temp = *desc_temp_ptr;
    TensorUtils::SetSize(desc_temp, 1024);
    op_def_f->AddInputDesc(desc_temp);
    op_def_f1->AddInputDesc(desc_temp);
    op_def_f2->AddInputDesc(desc_temp);

    (void) ge::AttrUtils::SetBool(op_def_f, ATTR_NAME_CONTINUOUS_OUTPUT, true);
    (void) ge::AttrUtils::SetBool(op_def_f1, ATTR_NAME_CONTINUOUS_OUTPUT, true);
    (void) ge::AttrUtils::SetBool(op_def_f2, ATTR_NAME_CONTINUOUS_OUTPUT, true);
    (void) ge::AttrUtils::SetListInt(op_def_f, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, {RT_MEMORY_P2P_DDR});
    (void) ge::AttrUtils::SetListInt(op_def_f1, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, {RT_MEMORY_P2P_DDR});
    (void) ge::AttrUtils::SetListInt(op_def_f2, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, {RT_MEMORY_P2P_DDR});
    if (continuous) {
      (void) ge::AttrUtils::SetBool(op_def_f, ATTR_NAME_CONTINUOUS_INPUT, true);
      (void) ge::AttrUtils::SetBool(op_def_f1, ATTR_NAME_CONTINUOUS_INPUT, true);
      (void) ge::AttrUtils::SetBool(op_def_f2, ATTR_NAME_CONTINUOUS_INPUT, true);
    }

   (void)ge::AttrUtils::SetStr(op_def_b, ATTR_NAME_BATCH_LABEL, "Batch_0");
   (void)ge::AttrUtils::SetStr(op_def_c, ATTR_NAME_BATCH_LABEL, "Batch_0");
   (void)ge::AttrUtils::SetStr(op_def_d, ATTR_NAME_BATCH_LABEL, "Batch_0");
   (void)ge::AttrUtils::SetStr(op_def_e, ATTR_NAME_BATCH_LABEL, "Batch_0");
   (void)ge::AttrUtils::SetStr(op_def_f, ATTR_NAME_BATCH_LABEL, "Batch_0");
   (void)ge::AttrUtils::SetStr(op_def_b1, ATTR_NAME_BATCH_LABEL, "Batch_1");
   (void)ge::AttrUtils::SetStr(op_def_c1, ATTR_NAME_BATCH_LABEL, "Batch_1");
   (void)ge::AttrUtils::SetStr(op_def_d1, ATTR_NAME_BATCH_LABEL, "Batch_1");
   (void)ge::AttrUtils::SetStr(op_def_e1, ATTR_NAME_BATCH_LABEL, "Batch_1");
   (void)ge::AttrUtils::SetStr(op_def_f1, ATTR_NAME_BATCH_LABEL, "Batch_1");
   (void)ge::AttrUtils::SetStr(op_def_b2, ATTR_NAME_BATCH_LABEL, "Batch_2");
   (void)ge::AttrUtils::SetStr(op_def_c2, ATTR_NAME_BATCH_LABEL, "Batch_2");
   (void)ge::AttrUtils::SetStr(op_def_d2, ATTR_NAME_BATCH_LABEL, "Batch_2");
   (void)ge::AttrUtils::SetStr(op_def_e2, ATTR_NAME_BATCH_LABEL, "Batch_2");
   (void)ge::AttrUtils::SetStr(op_def_f2, ATTR_NAME_BATCH_LABEL, "Batch_2");
   vector<int32_t> workspace_no_reuse_scope = { 1 };
   (void)ge::AttrUtils::SetListInt(op_def_c, ATTR_NAME_WORKSPACE_MEMORY_NO_REUSE_SCOPE, workspace_no_reuse_scope);
   (void)ge::AttrUtils::SetListInt(op_def_c1, ATTR_NAME_WORKSPACE_MEMORY_NO_REUSE_SCOPE, workspace_no_reuse_scope);
   (void)ge::AttrUtils::SetListInt(op_def_c2, ATTR_NAME_WORKSPACE_MEMORY_NO_REUSE_SCOPE, workspace_no_reuse_scope);

    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);
    ge::NodePtr node_e = graph->AddNode(op_def_e);
    ge::NodePtr node_f = graph->AddNode(op_def_f);
    ge::NodePtr node_b1 = graph->AddNode(op_def_b1);
    ge::NodePtr node_c1 = graph->AddNode(op_def_c1);
    ge::NodePtr node_d1 = graph->AddNode(op_def_d1);
    ge::NodePtr node_e1 = graph->AddNode(op_def_e1);
    ge::NodePtr node_f1 = graph->AddNode(op_def_f1);
    ge::NodePtr node_b2 = graph->AddNode(op_def_b2);
    ge::NodePtr node_c2 = graph->AddNode(op_def_c2);
    ge::NodePtr node_d2 = graph->AddNode(op_def_d2);
    ge::NodePtr node_e2 = graph->AddNode(op_def_e2);
    ge::NodePtr node_f2 = graph->AddNode(op_def_f2);
    ge::NodePtr node_g = graph->AddNode(op_def_g);

    //        --b--c--f------
    //       |   d--e/       |
    //    a--|--b1--c1--f1---|--g
    //       |   d1--e1/     |
    //        --b2--c2--f2---
    //            d2--e2/
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b1->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b2->GetInDataAnchor(0));

    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_f->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_f->GetInDataAnchor(1));

    ge::GraphUtils::AddEdge(node_b1->GetOutDataAnchor(0), node_c1->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d1->GetOutDataAnchor(0), node_e1->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_c1->GetOutDataAnchor(0), node_f1->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e1->GetOutDataAnchor(0), node_f1->GetInDataAnchor(1));

    ge::GraphUtils::AddEdge(node_b2->GetOutDataAnchor(0), node_c2->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d2->GetOutDataAnchor(0), node_e2->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_c2->GetOutDataAnchor(0), node_f2->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e2->GetOutDataAnchor(0), node_f2->GetInDataAnchor(1));

    ge::GraphUtils::AddEdge(node_f->GetOutDataAnchor(0), node_g->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_f1->GetOutDataAnchor(0), node_g->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_f2->GetOutDataAnchor(0), node_g->GetInDataAnchor(0));
    graph->TopologicalSorting();
  }

  void MakeContinuousOutputGraph(ge::ComputeGraphPtr &graph, const string &type = "some") {
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 1024, type, 2048);
    op_def_a->SetStreamId(1);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 1024, type, 2048);
    op_def_b->SetStreamId(1);
    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 1024, type, 2048);
    op_def_c->SetStreamId(1);
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 1024, type, 6144);
    ge::AttrUtils::SetBool(op_def_d, ATTR_NAME_CONTINUOUS_INPUT, true);
    ge::AttrUtils::SetBool(op_def_d, ATTR_NAME_CONTINUOUS_OUTPUT, true);
    auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
    auto desc_temp = *desc_temp_ptr;
    TensorUtils::SetSize(desc_temp, 2048);
    op_def_d->AddInputDesc(desc_temp);
    op_def_d->AddInputDesc(desc_temp);
    op_def_d->AddOutputDesc(desc_temp);
    op_def_d->AddOutputDesc(desc_temp);
    op_def_d->SetStreamId(2);
    ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 2048, type, 2048);
    op_def_e->SetStreamId(1);
    ge::OpDescPtr op_def_f = CreateOpWithWsSize("F", 2048, type, 2048);
    op_def_f->SetStreamId(1);
    ge::OpDescPtr op_def_g = CreateOpWithWsSize("G", 2048, type, 2048);
    op_def_g->SetStreamId(1);
    ge::OpDescPtr op_def_h = CreateOpWithWsSize("H", 1024, type, 2048);
    op_def_h->SetStreamId(1);
    ge::OpDescPtr op_def_i = CreateOpWithWsSize("I", 1024, type, 2048);
    op_def_i->SetStreamId(1);
    ge::OpDescPtr op_def_j = CreateOpWithWsSize("J", 1024, type, 2048);
    op_def_j->SetStreamId(1);
    ge::OpDescPtr op_def_k = CreateOpWithWsSize("K", 1024, type, 6144);
    ge::AttrUtils::SetBool(op_def_k, ATTR_NAME_CONTINUOUS_INPUT, true);
    ge::AttrUtils::SetBool(op_def_k, ATTR_NAME_CONTINUOUS_OUTPUT, true);
    op_def_k->AddInputDesc(desc_temp);
    op_def_k->AddInputDesc(desc_temp);
    op_def_k->AddOutputDesc(desc_temp);
    op_def_k->AddOutputDesc(desc_temp);
    op_def_k->SetStreamId(2);
    ge::OpDescPtr op_def_l = CreateOpWithWsSize("L", 1024, type, 2048);
    op_def_l->SetStreamId(1);
    ge::OpDescPtr op_def_m = CreateOpWithWsSize("M", 1024, type, 2048);
    op_def_m->SetStreamId(1);
    ge::OpDescPtr op_def_n = CreateOpWithWsSize("N", 1024, type, 2048);
    op_def_n->SetStreamId(1);
    ge::OpDescPtr op_def_o = CreateOpWithWsSize("O", 1024, type, 6144);
    op_def_o->SetStreamId(2);
    op_def_o->AddInputDesc(desc_temp);
    op_def_o->AddInputDesc(desc_temp);


    // add node
    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);
    ge::NodePtr node_e = graph->AddNode(op_def_e);
    ge::NodePtr node_f = graph->AddNode(op_def_f);
    ge::NodePtr node_g = graph->AddNode(op_def_g);
    ge::NodePtr node_h = graph->AddNode(op_def_h);
    ge::NodePtr node_i = graph->AddNode(op_def_i);
    ge::NodePtr node_j = graph->AddNode(op_def_j);
    ge::NodePtr node_k = graph->AddNode(op_def_k);
    ge::NodePtr node_l = graph->AddNode(op_def_l);
    ge::NodePtr node_m = graph->AddNode(op_def_m);
    ge::NodePtr node_n = graph->AddNode(op_def_n);
    ge::NodePtr node_o = graph->AddNode(op_def_o);

    /**
     *           a:1  b:1  c:1
     *            |___|___|
     *                |
     *               d:2
     *             ___|___
     *           |   |   |
     *         e:1  f:1  g:1
     *           |   |   |
     *         h:1  i:1  j:1
     *           |___|___|
     *               |
     *              k:2
     *            ___|___
     *           |   |   |
     *          l:1  m:1 n:1
     *           |___|___|
     *              o:1
     */

    // add edge
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_d->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_d->GetInDataAnchor(2));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(1), node_f->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(2), node_g->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_h->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_f->GetOutDataAnchor(0), node_i->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_g->GetOutDataAnchor(0), node_j->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_h->GetOutDataAnchor(0), node_k->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_i->GetOutDataAnchor(0), node_k->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_j->GetOutDataAnchor(0), node_k->GetInDataAnchor(2));
    ge::GraphUtils::AddEdge(node_k->GetOutDataAnchor(0), node_l->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_k->GetOutDataAnchor(1), node_m->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_k->GetOutDataAnchor(2), node_n->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_l->GetOutDataAnchor(0), node_o->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_m->GetOutDataAnchor(0), node_o->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_n->GetOutDataAnchor(0), node_o->GetInDataAnchor(2));
    graph->TopologicalSorting();
  }

  ComputeGraphPtr MakeRefVariableGraph() {
    ge::ut::GraphBuilder builder("graph_ref_variable");
    auto var_input = builder.AddNode("var", "Variable", 1, 1);
    int64_t output_size = 602112;
    auto output_tensordesc = var_input->GetOpDesc()->MutableOutputDesc(0);
    ge::TensorUtils::SetSize(*output_tensordesc, output_size);
    var_input->GetOpDesc()->UpdateOutputDesc(0, *output_tensordesc);

    auto const_input = builder.AddNode("const", "Const", 1, 1);
    auto add = builder.AddNode("add", "Add", 0, 1);
    auto add1 = builder.AddNode("add1", "Add", 1, 1);
    // add link
    builder.AddControlEdge(var_input, add);
    builder.AddDataEdge(const_input, 0, add, 0);
    builder.AddDataEdge(add, 0, add1, 0);

    // set ref
    output_tensordesc = add->GetOpDesc()->MutableOutputDesc(0);
    ge::AttrUtils::SetStr(output_tensordesc, ASSIGN_VAR_NAME, var_input->GetName());
    ge::TensorUtils::SetSize(*output_tensordesc, output_size);
    add->GetOpDesc()->UpdateOutputDesc(0, *output_tensordesc);
    return builder.GetGraph();
  }

  void MakeSortTestGraph(ge::ComputeGraphPtr &graph, const string &type = "some") {
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 1024, type, 2048);
    op_def_a->SetStreamId(1);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 1024, type, 2048);
    op_def_b->SetStreamId(1);
    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 1024, type, 2048);
    op_def_c->SetStreamId(1);
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 1024, type, 2048);
    op_def_d->SetStreamId(1);

    ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 1024, type, 2048);
    op_def_e->SetStreamId(1);

    auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
    auto desc_temp = *desc_temp_ptr;
    TensorUtils::SetSize(desc_temp, 1024);
    op_def_d->AddInputDesc(desc_temp);
    op_def_d->AddInputDesc(desc_temp);

    // add node
    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);
    ge::NodePtr node_e = graph->AddNode(op_def_e);

    /**
     *    b:0  a:1  c:1
     *      \  |  /
     *       d:1
     *         |
     *       e:1
     */

    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_d->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_d->GetInDataAnchor(2));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
    graph->TopologicalSorting();
  }
  void MakePaddingInputContinuousReuseGraph(ge::ComputeGraphPtr &graph) {
    /**
     *           a:0  b:1
     *            |___|
     *              |
     *             c:2
     *              |
     *             d:3   e:4
     *              |___|
     *               |
     *               f:5
     */
    // node d and g is padding input continuous
    const string &type = "some";
    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 1024, type, 2048);
    op_def_a->SetStreamId(1);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 1024, type, 2048);
    op_def_b->SetStreamId(1);
    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 1024, type, 4096);
    ge::AttrUtils::SetBool(op_def_c, ATTR_NAME_CONTINUOUS_INPUT, true);
    ge::AttrUtils::SetBool(op_def_c, "need_gentask_atomic", true);
    std::vector<int32_t> input_indexes = {-1};
    (void)ge::AttrUtils::SetListInt(op_def_c, ATOMIC_ATTR_INPUT_INDEX, input_indexes);
    op_def_c->SetStreamId(1);
    auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
    auto desc_temp = *desc_temp_ptr;
    TensorUtils::SetSize(desc_temp, 2048);
    op_def_c->AddInputDesc(desc_temp);
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 1024, type, 2048);
    op_def_d->SetStreamId(1);
    ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 1024, type, 2048);
    op_def_e->SetStreamId(1);
    ge::OpDescPtr op_def_f = CreateOpWithWsSize("F", 1024, type, 4096);
    ge::AttrUtils::SetBool(op_def_f, ATTR_NAME_CONTINUOUS_INPUT, true);
    ge::AttrUtils::SetBool(op_def_f, "need_gentask_atomic", true);
    (void)ge::AttrUtils::SetListInt(op_def_f, ATOMIC_ATTR_INPUT_INDEX, input_indexes);
    op_def_f->SetStreamId(1);
    TensorUtils::SetSize(desc_temp, 2048);
    op_def_f->AddInputDesc(desc_temp);
    // add node
    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);
    ge::NodePtr node_e = graph->AddNode(op_def_e);
    ge::NodePtr node_f = graph->AddNode(op_def_f);

    // add edge
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_c->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_f->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_f->GetInDataAnchor(1));
    graph->TopologicalSorting();
  }

  void MakePaddingInputContinuousReuseGraphMultiRef(ge::ComputeGraphPtr &graph, bool diff_stream = false) {
    /**
     *          a:0  b:1        a:1 f:5
     *           |___|          |___|
     *             |              |
     *            c:2            g:6
     *             |             |
     *            d:3   e:4     h:7
     *             |___|_______|
     *                  |      |
     *                 i:8     |
     *                 |       |
     *                 j:9     |
     *                 |_______|
     *                     |
     *                     k:10
     */
    // node c g i is padding input continuous
    const string &type = "some";
    std::vector<int32_t> input_indexes = {-1};
    auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
    auto desc_temp = *desc_temp_ptr;
    TensorUtils::SetSize(desc_temp, 2048);

    ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 1024, type, 2048);
    op_def_a->SetStreamId(1);
    ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 1024, type, 2048);
    op_def_b->SetStreamId(1);
    ge::OpDescPtr op_def_f = CreateOpWithWsSize("F", 1024, type, 4096);
    op_def_f->SetStreamId(1);

    ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 1024, type, 4096);
    ge::AttrUtils::SetBool(op_def_c, ATTR_NAME_CONTINUOUS_INPUT, true);
    ge::AttrUtils::SetBool(op_def_c, "need_gentask_atomic", true);
    (void)ge::AttrUtils::SetListInt(op_def_c, ATOMIC_ATTR_INPUT_INDEX, input_indexes);
    op_def_c->SetStreamId(1);
    op_def_c->AddInputDesc(desc_temp);

    ge::OpDescPtr op_def_g = CreateOpWithWsSize("G", 1024, type, 4096);
    ge::AttrUtils::SetBool(op_def_g, ATTR_NAME_CONTINUOUS_INPUT, true);
    ge::AttrUtils::SetBool(op_def_g, "need_gentask_atomic", true);
    (void)ge::AttrUtils::SetListInt(op_def_g, ATOMIC_ATTR_INPUT_INDEX, input_indexes);
    int64_t stream_id = diff_stream ? 0 : 1;
    op_def_g->SetStreamId(stream_id);
    op_def_g->AddInputDesc(desc_temp);

    ge::OpDescPtr op_def_h = CreateOpWithWsSize("H", 1024, type, 2048);
    op_def_h->SetStreamId(1);
    ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 1024, type, 2048);
    op_def_d->SetStreamId(1);
    ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 1024, type, 2048);
    op_def_e->SetStreamId(1);
    ge::OpDescPtr op_def_i = CreateOpWithWsSize("I", 1024, type, 4096);
    ge::AttrUtils::SetBool(op_def_i, ATTR_NAME_CONTINUOUS_INPUT, true);
    ge::AttrUtils::SetBool(op_def_i, "need_gentask_atomic", true);
    (void)ge::AttrUtils::SetListInt(op_def_i, ATOMIC_ATTR_INPUT_INDEX, input_indexes);
    op_def_i->SetStreamId(1);
    op_def_i->AddInputDesc(desc_temp);
    op_def_i->AddInputDesc(desc_temp);

    ge::OpDescPtr op_def_j = CreateOpWithWsSize("J", 1024, type, 2048);
    op_def_j->SetStreamId(1);
    ge::OpDescPtr op_def_k = CreateOpWithWsSize("K", 1024, type, 2048);
    op_def_k->SetStreamId(stream_id);
    op_def_k->AddInputDesc(desc_temp);

    // add node
    ge::NodePtr node_a = graph->AddNode(op_def_a);
    ge::NodePtr node_b = graph->AddNode(op_def_b);
    ge::NodePtr node_c = graph->AddNode(op_def_c);
    ge::NodePtr node_d = graph->AddNode(op_def_d);
    ge::NodePtr node_e = graph->AddNode(op_def_e);
    ge::NodePtr node_f = graph->AddNode(op_def_f);
    ge::NodePtr node_g = graph->AddNode(op_def_g);
    ge::NodePtr node_h = graph->AddNode(op_def_h);
    ge::NodePtr node_i = graph->AddNode(op_def_i);
    ge::NodePtr node_j = graph->AddNode(op_def_j);
    ge::NodePtr node_k = graph->AddNode(op_def_k);

    // add edge
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_c->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_g->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_f->GetOutDataAnchor(0), node_g->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_g->GetOutDataAnchor(0), node_h->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_e->GetOutDataAnchor(0), node_i->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_i->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(node_h->GetOutDataAnchor(0), node_i->GetInDataAnchor(2));
    ge::GraphUtils::AddEdge(node_i->GetOutDataAnchor(0), node_j->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_j->GetOutDataAnchor(0), node_k->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(node_h->GetOutDataAnchor(0), node_k->GetInDataAnchor(1));
    graph->TopologicalSorting();
  }

  // offset从小到大排，offset相同时，内存从大到小排序
  static bool CmpOffset(const ReuseNode &l, const ReuseNode &r) {
    if (l.offset == r.offset) {
      return l.size > r.size;
    }
    return l.offset < r.offset;
  }

  void AlignMemOffset(int64_t &mem_align_size) {
    if (mem_align_size <= 0) {
      return;
    }
    mem_align_size = (mem_align_size + MEM_ALIGN_SIZE - 1) / MEM_ALIGN_SIZE * MEM_ALIGN_SIZE;
  }

  bool Match(const MemoryBlock *const block, const ReuseNode &reuse_node) {
    if (block != nullptr) {
      for (const auto &node : block->node_type_index_list_) {
        if ((reuse_node.name == node.node_->GetName()) && (node.mem_type_ == reuse_node.mem_type)
            && (reuse_node.index == node.index_)) {
           return true;
        }
      }
    }
    return false;
  }

  Status GetReuseBlock(const BlockMemAssignerPtr &assigner, ge::ComputeGraphPtr &graph,
                       ReuseNode reuse_node, MemoryBlock **reuse_block) {
    EXPECT_TRUE(assigner != nullptr);
    const auto blocks = assigner->GetMemoryBlocks();
    EXPECT_TRUE(blocks.size() > 1U);
    for (const auto block : blocks) {
      if (Match(block, reuse_node)) {
        GELOGI("Match block is %s.", reuse_node.ToString().c_str());
        *reuse_block = block;
        break;
      }
    }
    return SUCCESS;
  }

  Status CheckReuse(const BlockMemAssignerPtr &assigner, ge::ComputeGraphPtr &graph,
      std::vector<ReuseNode> check_reuse_nodes, bool reuse = true) {
    EXPECT_TRUE(assigner != nullptr);
    const auto blocks = assigner->GetMemoryBlocks();
    EXPECT_TRUE(blocks.size() > 1U);
    std::vector<ReuseNode> offsets;
    for (const auto &reuse_node : check_reuse_nodes) {
      bool match = false;
      // get from memory block
      for (const auto block : blocks) {
        if (Match(block, reuse_node)) {
          offsets.emplace_back(reuse_node.name, reuse_node.mem_type, reuse_node.index, block->HeadOffset(),
              block->Size());
          GELOGI("Match block is %s.", offsets.back().ToString().c_str());
          match = true;
        }
      }

      if (match) {
        continue;
      }

      // 连续内存非首节点内存复用阶段未分内存，memory block获取不到，get from node
      auto node = graph->FindNode(reuse_node.name);
      int64_t size = 0;
      int64_t offset = 0;
      if (reuse_node.mem_type == kOutput) {
        ge::TensorUtils::GetSize(*(node->GetOpDesc()->MutableOutputDesc(reuse_node.index)), size);
        offset = node->GetOpDesc()->GetOutputOffset()[reuse_node.index];
      } else if (reuse_node.mem_type == kWorkspace) {
        offset = node->GetOpDesc()->GetWorkspace()[reuse_node.index];
        size = node->GetOpDesc()->GetWorkspaceBytes()[reuse_node.index];
      } else {
        size = sizeof(RuntimeTensorDesc);
        (void)AttrUtils::GetInt(node->GetOpDesc()->MutableOutputDesc(reuse_node.index),
            ATTR_NAME_TENSOR_DESC_MEM_OFFSET, offset);
      }
      AlignMemOffset(size);
      offsets.emplace_back(reuse_node.name, reuse_node.mem_type, reuse_node.index, offset, size);
      GELOGI("Match node is %s.", offsets.back().ToString().c_str());
    }

    std::sort(offsets.begin(), offsets.end(), CmpOffset);
    EXPECT_TRUE(offsets.size() == check_reuse_nodes.size());
    for (size_t i = 1U; i < offsets.size(); ++i) {
      bool check_ok = (offsets[i].offset + offsets[i].size) <= (offsets[0].offset + offsets[0].size);
      if (reuse) {
        if (!check_ok) {
          GELOGE(FAILED, "Max block is %s, %s is out of ranged.", offsets[0].ToString().c_str(),
                 offsets[i].ToString().c_str());
        }
        EXPECT_TRUE(check_ok);
      } else {
        EXPECT_TRUE(!check_ok);
      }
    }
    return SUCCESS;
  }

  bool GetOffsetAndSize(const std::map<std::string, const ge::Node *> &name_to_node_map, const std::string node_name,
      uint32_t out_index, int64_t &offset, int64_t &size) {
    auto node_iter = name_to_node_map.find(node_name);
    ge::Node *node = nullptr;
    if (node_iter == name_to_node_map.end()) {
      std::cerr << "can not find node " << node_name << std::endl;
      return false;
    }
    node = const_cast<ge::Node *>(node_iter->second);
    auto a_offsets = node->GetOpDescBarePtr()->GetOutputOffset();
    TensorUtils::GetSize(node->GetOpDescBarePtr()->GetOutputDesc(out_index), size);
    if (out_index >= a_offsets.size()) {
      std::cerr <<  node->GetName() << " output index[" << out_index << "] >= output offsets size[" << a_offsets.size()
                << "]" << std::endl;
      return false;
    }
    offset = a_offsets[out_index];
    return true;
  }

  bool CheckNotReuse(const ge::ComputeGraphPtr &graph, const std::string &a, uint32_t a_out_index,
                     const std::string &b, uint32_t b_out_index) {
    int64_t a_offset;
    int64_t a_size;
    std::map<std::string, const ge::Node *> name_to_node_map;
    for (auto node : graph->GetAllNodesPtr()) {
      name_to_node_map[node->GetName()] = node;
    }
    if (!GetOffsetAndSize(name_to_node_map, a, a_out_index, a_offset, a_size)) {
      std::cerr << "get " << a << " offset and size failed" << std::endl;
      return false;
    }
    int64_t b_offset;
    int64_t b_size;
    if (!GetOffsetAndSize(name_to_node_map, b, b_out_index, b_offset, b_size)) {
      std::cerr << "get " << b << " offset and size failed" << std::endl;
      return false;
    }
    if (a_offset < b_offset) {
      if (a_size + a_offset > b_offset) {
        std::cerr << "CheckNotReuse failed" << std::endl;
        std::cerr << a << ", out " << a_out_index << ", offset: " << a_offset << ", size: " << a_size << std::endl;
        std::cerr << b << ", out " << b_out_index << ", offset: " << b_offset << ", size: " << b_size << std::endl;
        return false;
      }
    } else {
      if (b_offset + b_size > a_offset) {
        std::cerr << "CheckNotReuse failed" << std::endl;
        std::cerr << a << ", out " << a_out_index << ", offset: " << a_offset << ", size: " << a_size << std::endl;
        std::cerr << b << ", out " << b_out_index << ", offset: " << b_offset << ", size: " << b_size << std::endl;
        return false;
      }
    }
    return true;
  }

 protected:
  void SetUp() {}

  void TearDown() { domi::GetContext().out_nodes_map.clear(); }
  ReuseStrategy reuse_strategy_{};
};

namespace ge {

class MockBlockMemAssigner : public BlockMemAssigner {
 public:
  explicit MockBlockMemAssigner(const MemAssistInfo &mem_assist_info) : BlockMemAssigner(mem_assist_info) {};

  virtual ~MockBlockMemAssigner(){};

  Status GetMemoryRanges(std::vector<int64_t> &ranges) override { return FAILED; }
};
}  // namespace ge

TEST_F(UtestMemoryAssignerTest, graph_mark_life_time_for_ref_mem_in_diff_stream) {
  auto options_backup = GetThreadLocalContext().GetAllGraphOptions();
  const map<string, string> options{{MEMORY_OPTIMIZATION_POLICY, kMemoryPriority}};
  GetThreadLocalContext().SetGraphOption(options);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeDiffStreamGraphForLifeTime(graph);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }

  EXPECT_EQ(offset, 10240);
  GetThreadLocalContext().SetGraphOption(options_backup);
}

/*
 *                     sub_graph
 *   partitioned_call  +---------------+
 *                     |    a          |
 *                     |    |          |
 *                     |    b          |
 *                     |    |          |
 *                     |  sub_netoutput|
 *                     +---------------+
 * partitioned_call output suspended
 */
TEST_F(UtestMemoryAssignerTest, PartitionedCallSuspendOut_Reuse_Ok) {
  auto graph = block_mem_ut::BuildPartitionedCallSuspendOut();
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  EXPECT_TRUE(CheckNotReuse(graph, "partitioned_call", 0, "a", 0));
}

/*
 *                     sub_graph
 *   partitioned_call  +---------------+
 *                     |    a          |
 *                     |    |          |
 *                     |    b          |
 *                     |    |          |
 *                     |  sub_netoutput|
 *                     +---------------+
 * partitioned_call output suspended
 */
TEST_F(UtestMemoryAssignerTest, MemorySizeCalctypeAttr_And_MemoryTypeL1_SkipCheck) {
  auto graph = block_mem_ut::BuildNodeWithMemoSizeCalcTypeAndMemoryTypeL1();
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
}

TEST_F(UtestMemoryAssignerTest, graph_memory_assign_continuous_input) {
  ge::ComputeGraphPtr graph = MakeCascadeContinuousMemoryGraph();
  auto addn1 = graph->FindNode("addn1");
  auto addn2 = graph->FindNode("addn2");
  EXPECT_EQ(addn1->GetOpDesc()->GetOutputOffset()[0], 100);
  EXPECT_EQ(addn2->GetOpDesc()->GetOutputOffset()[0], 200);
  GraphMemoryAssigner memoryAssigner(graph);
  MemoryOffset memory_offset(RT_MEMORY_HBM, 1024);
  memoryAssigner.memory_offset_.emplace(RT_MEMORY_HBM, memory_offset);
  EXPECT_EQ(memoryAssigner.ReAssignContinuousMemory(), GRAPH_SUCCESS);
  EXPECT_EQ(addn1->GetOpDesc()->GetOutputOffset()[0], 500);
  EXPECT_EQ(addn2->GetOpDesc()->GetOutputOffset()[0], 600);
}

TEST_F(UtestMemoryAssignerTest, block_memory_assign_nopading_continuous_memory) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeContinuousReuseGraph(graph, true);
  HybridMemAssigner hybridMemAssigner(graph);
  ge::Status ret = hybridMemAssigner.Assign();
  size_t offset = 0;
  auto it = hybridMemAssigner.GetMemOffsets().find(RT_MEMORY_HBM);
  if (it != hybridMemAssigner.GetMemOffsets().end()) {
    offset = it->second;
  }

  EXPECT_EQ(offset, 3584);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestMemoryAssignerTest, block_memory_assign_nopading_continuous_memory_fail) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeContinuousReuseGraph(graph, true);
  auto op_d = graph->FindNode("D");
  (void)ge::AttrUtils::SetInt(op_d->GetOpDesc(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 3);
  GraphMemoryAssigner memoryAssigner(graph);
  MemoryOffset memory_offset(RT_MEMORY_HBM, 0);
  memoryAssigner.memory_offset_.emplace(RT_MEMORY_HBM, memory_offset);
  auto ret = memoryAssigner.ReAssignContinuousMemory();
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(UtestMemoryAssignerTest, block_memory_assign_continuous_memory) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeContinuousReuseGraph(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_copy_mem_size = 0;
  MemoryAssigner memoryAssigner(graph);
  ge::Status ret = memoryAssigner.AssignMemory(mem_offset, zero_copy_mem_size);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }

  EXPECT_EQ(offset, 3584);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestMemoryAssignerTest, block_memory_assign_continuous_memory_out_diff_stream) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeContinuousReuseGraphDiffStream(graph);
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  auto ret = GraphUtils::GetRefMapping(graph, mem_assist_info.symbol_to_anchors,
                                       mem_assist_info.anchor_to_symbol);
  EXPECT_EQ(ret, SUCCESS);
  std::vector<int64_t> ranges;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  assigner.SetReuseStrategy(ReuseStrategy{false, true, false, true});
  assigner.GetMemoryRanges(ranges);
  assigner.AssignMemoryWithReuse(ranges);
  const auto blocks = assigner.GetMemoryBlocks();
  bool has_checked = false;
  assigner.SetOpMemOffset(false);
  for (const auto block : blocks) {
    for (const auto &node : block->node_type_index_list_) {
      if (node.mem_type_ != kOutput) {
        continue;
      }
      if (node.node_->GetOpDesc()->GetName() == "E") {
        EXPECT_EQ(block->GetLifeEnd(2), 4);
        has_checked = true;
      }
    }
  }
  EXPECT_TRUE(has_checked);
}

TEST_F(UtestMemoryAssignerTest, block_memory_assign_with_release_first_reuse_first_strategy) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeGraph(graph);
  (void)AttrUtils::SetBool(graph, "_mem_release_first_reuse_first", true);
  map<uint64_t, size_t> mem_offset;
  size_t zero_copy_mem_size = 0;
  MemoryAssigner memoryAssigner(graph);
  EXPECT_EQ(memoryAssigner.AssignMemory(mem_offset, zero_copy_mem_size), SUCCESS);
}

TEST_F(UtestMemoryAssignerTest, graph_memory_set_last_used_attr) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeGraph(graph);
  auto node_f = graph->FindNode("F");
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);

  bool flag = 0;
  (void) ge::AttrUtils::GetBool(node_f->GetOpDesc()->GetInputDesc(0), ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE, flag);
  EXPECT_EQ(flag, true);
}

TEST_F(UtestMemoryAssignerTest, graph_memory_assign_ref_var) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeGraph(graph, VARIABLE);
  auto node_a = graph->FindNode("A");
  auto node_b = graph->FindNode("B");
  std::string value = "A";
  (void) ge::AttrUtils::SetStr(node_b->GetOpDesc()->MutableOutputDesc(0), REF_VAR_SRC_VAR_NAME, value);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);

  EXPECT_EQ(node_b->GetOpDesc()->GetOutputOffset()[0], node_a->GetOpDesc()->GetOutputOffset()[0]);
}

TEST_F(UtestMemoryAssignerTest, graph_memory_assign_ref_var_not_found) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeGraph(graph, VARIABLE);

  ge::ComputeGraphPtr sub_graph = std::make_shared<ge::ComputeGraph>("");
  MakeReuseGraph(sub_graph);
  graph->AddSubGraph(sub_graph);

  auto node_a = graph->FindNode("A");
  auto node_b = graph->FindNode("B");
  std::string value = "M";
  (void) ge::AttrUtils::SetStr(node_b->GetOpDesc()->MutableOutputDesc(0), REF_VAR_SRC_VAR_NAME, value);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  EXPECT_NE(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
}

TEST_F(UtestMemoryAssignerTest, graph_memory_assign_src_const) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeGraph(graph, CONSTANTOP);

  auto node_a = graph->FindNode("A");
  (void) ge::AttrUtils::SetStr(node_a->GetOpDesc(), ATTR_NAME_SRC_CONST_NAME, "src_const");
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
}

TEST_F(UtestMemoryAssignerTest, graph_memory_assign_set_input_offset) {
  ge::ComputeGraphPtr graph = MakeRefNodeGraph();
  auto assgin = graph->FindNode("assgin");
  EXPECT_EQ(assgin->GetOpDesc()->GetOutputOffset()[0], 10000);
  EXPECT_EQ(assgin->GetOpDesc()->GetInputOffset()[0], 100);
  EXPECT_EQ(assgin->GetOpDesc()->GetInputOffset()[1], 0);
  GraphMemoryAssigner memory_assigner(graph);
  memory_assigner.mem_assigner_.reset(new(std::nothrow) HybridMemAssigner(graph));
  MemoryOffset memory_offset(RT_MEMORY_HBM, 0);
  memory_assigner.memory_offset_.emplace(RT_MEMORY_HBM, memory_offset);
  EXPECT_EQ(memory_assigner.SetInputOffset(), GRAPH_SUCCESS);
  EXPECT_EQ(assgin->GetOpDesc()->GetOutputOffset()[0], 10100);
  EXPECT_EQ(assgin->GetOpDesc()->GetInputOffset()[0], 10100);
  EXPECT_EQ(assgin->GetOpDesc()->GetInputOffset()[1], 0);
  EXPECT_EQ(memory_assigner.CheckOffset(), GRAPH_SUCCESS);
}

TEST_F(UtestMemoryAssignerTest, graph_memory_assign_check_inner_offset) {
  ge::ComputeGraphPtr graph = MakeRefNodeGraph();
  auto assign = graph->FindNode("assgin");
  auto op_desc = assign->GetOpDesc();
  int64_t inner_offset = 0;
  EXPECT_EQ(ge::AttrUtils::GetInt(op_desc->MutableInputDesc(0), ATTR_NAME_INNER_OFFSET, inner_offset), false);
  EXPECT_EQ(ge::AttrUtils::GetInt(op_desc->MutableInputDesc(1), ATTR_NAME_INNER_OFFSET, inner_offset), false);
  GraphMemoryAssigner memoryAssigner(graph);
  MemoryOffset memory_offset(RT_MEMORY_HBM, 0);
  memoryAssigner.memory_offset_.emplace(RT_MEMORY_HBM, memory_offset);
  EXPECT_EQ(memoryAssigner.SetInputOffset(), GRAPH_SUCCESS);
  EXPECT_EQ(ge::AttrUtils::GetInt(op_desc->MutableInputDesc(0), ATTR_NAME_INNER_OFFSET, inner_offset), true);
  EXPECT_EQ(inner_offset, 100);
  EXPECT_EQ(ge::AttrUtils::GetInt(op_desc->MutableOutputDesc(0), ATTR_NAME_INNER_OFFSET, inner_offset), true);
  EXPECT_EQ(inner_offset, 100);
  EXPECT_EQ(ge::AttrUtils::GetInt(op_desc->MutableInputDesc(1), ATTR_NAME_INNER_OFFSET, inner_offset), false);
}

TEST_F(UtestMemoryAssignerTest, graph_memory_assign_update_ref_op_offset_reverse) {
    ge::ut::GraphBuilder builder("graph");
    auto data_input = builder.AddNode("data", "Data", 1, 1);
    auto const_input = builder.AddNode("const", "Const", 1, 1);
    auto add = builder.AddNode("add", "Add", 2, 1);
     // add link
    builder.AddDataEdge(data_input, 0, add, 0);
    builder.AddDataEdge(const_input, 0, add, 1);
    // set ref
    uint32_t reuse_input_index = 0;
    auto output_tensordesc = data_input->GetOpDesc()->MutableOutputDesc(0);
    ge::TensorUtils::SetReuseInput(*output_tensordesc, true);
    ge::TensorUtils::SetReuseInputIndex(*output_tensordesc, reuse_input_index);
    auto output_tensordesc1 = add->GetOpDesc()->MutableOutputDesc(0);
    ge::TensorUtils::SetReuseInput(*output_tensordesc1, true);
    ge::TensorUtils::SetReuseInputIndex(*output_tensordesc1, reuse_input_index);
    ge::ComputeGraphPtr graph = builder.GetGraph();

    GraphMemoryAssigner memoryAssigner(graph);
    EXPECT_EQ(memoryAssigner.UpdateRefOpOffsetReverse(add), SUCCESS);
}

TEST_F(UtestMemoryAssignerTest, graph_memory_assign_var_input_ref_cascade_false) {
  ge::ut::GraphBuilder builder("graph");
  auto var = builder.AddNode("var", VARIABLE, 1, 1);
  auto broadcast = builder.AddNode("broadcast", HCOMBROADCAST, 1, 1);
  auto assign = builder.AddNode("assign", "Assign", 2, 1);
  // add link
  builder.AddDataEdge(var, 0, assign, 0);
  builder.AddDataEdge(var, 0, broadcast, 0);
  builder.AddDataEdge(broadcast, 0, assign, 1);

  int reuse_input_index = 0;
  auto broadcast_desc = broadcast->GetOpDesc()->MutableOutputDesc(0);
  ge::TensorUtils::SetReuseInput(*broadcast_desc, true);
  ge::TensorUtils::SetReuseInputIndex(*broadcast_desc, reuse_input_index);

  ge::ComputeGraphPtr graph = builder.GetGraph();

  GraphMemoryAssigner memory_assigner(graph);
  bool ref_cascade = memory_assigner.IsRefFromInputOpCascade(broadcast);
  EXPECT_EQ(ref_cascade, false);
  ref_cascade = memory_assigner.IsRefFromInputOpCascade(assign);
  EXPECT_EQ(ref_cascade, false);
  auto ret = memory_assigner.UpdateRefOpOffsetReverse(broadcast);
  EXPECT_EQ(ret, SUCCESS);
  ret = memory_assigner.UpdateRefOpOffsetReverse(assign);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestMemoryAssignerTest, graph_memory_assign_atomic_output_and_workspace) {
  ge::ut::GraphBuilder builder("graph");
  auto data_input = builder.AddNode("data", "Data", 1, 1);
  auto const_input = builder.AddNode("const", "Const", 1, 1);
  auto add = builder.AddNode("add", "Add", 2, 1);
  // add link
  builder.AddDataEdge(data_input, 0, add, 0);
  builder.AddDataEdge(const_input, 0, add, 1);
  ge::ComputeGraphPtr graph = builder.GetGraph();

  auto node = graph->FindNode("add");
  EXPECT_NE(node, nullptr);
  auto output_tensor_desc = node->GetOpDesc()->MutableOutputDesc(0);
  ge::TensorUtils::SetSize(*output_tensor_desc, 100);
  vector<int64_t> output_list = {0};
  node->GetOpDesc()->SetOutputOffset(output_list);
  vector<int64_t> workspace_list = {0};
  node->GetOpDesc()->SetWorkspace(workspace_list);
  vector<int64_t> atomic_output_index = {0};
  bool set_attr = ge::AttrUtils::SetListInt(node->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);
  EXPECT_EQ(set_attr, true);

  map<string, map<int64_t, int64_t>> workspace_info;
  workspace_info["add"][0] = 100;
  set_attr = node->GetOpDesc()->SetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspace_info);
  EXPECT_EQ(set_attr, true);

  {
    bool is_fusion_node = false;
    set_attr = ge::AttrUtils::SetBool(node->GetOpDesc(), ATOMIC_ATTR_IS_FUSION_NODE, is_fusion_node);
    EXPECT_EQ(set_attr, true);

    GraphMemoryAssigner graph_memory_assigner(graph);
    graph_memory_assigner.mem_assigner_.reset(new(std::nothrow) HybridMemAssigner(graph));
    graph_memory_assigner.memory_offset_.insert({RT_MEMORY_HBM, MemoryOffset(RT_MEMORY_HBM, 0)});
    std::map<int64_t, std::vector<int64_t>> mem_type_to_offset_end;
    std::map<int64_t, std::vector<int64_t>> mem_type_to_real_atomic_sizes;
    Status ret = graph_memory_assigner.AssignAtomicOutputAndWorkspaceMemory(node, mem_type_to_offset_end,
                                                                            mem_type_to_real_atomic_sizes);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(mem_type_to_offset_end[RT_MEMORY_HBM].size(), 2);
    MemoryOffset mem_offset = graph_memory_assigner.memory_offset_.at(RT_MEMORY_HBM);
    EXPECT_EQ(mem_offset.mem_offset_, 1024);
  }

  {
    bool is_fusion_node = true;
    set_attr = ge::AttrUtils::SetBool(node->GetOpDesc(), ATOMIC_ATTR_IS_FUSION_NODE, is_fusion_node);
    EXPECT_EQ(set_attr, true);

    GraphMemoryAssigner graph_memory_assigner(graph);
    graph_memory_assigner.mem_assigner_.reset(new(std::nothrow) HybridMemAssigner(graph));
    graph_memory_assigner.memory_offset_.insert({RT_MEMORY_HBM, MemoryOffset(RT_MEMORY_HBM, 0)});
    std::map<int64_t, std::vector<int64_t>> mem_type_to_offset_end;
    std::map<int64_t, std::vector<int64_t>> mem_type_to_real_atomic_sizes;
    Status ret = graph_memory_assigner.AssignAtomicOutputAndWorkspaceMemory(node, mem_type_to_offset_end,
                                                                            mem_type_to_real_atomic_sizes);
    EXPECT_EQ(ret, SUCCESS);
    EXPECT_EQ(mem_type_to_offset_end[RT_MEMORY_HBM].size(), 2);
    MemoryOffset mem_offset = graph_memory_assigner.memory_offset_.at(RT_MEMORY_HBM);
    EXPECT_EQ(mem_offset.mem_offset_, 1024);
  }
}

TEST_F(UtestMemoryAssignerTest, one_session_scope_op) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeSessionScopeReuseGraph(graph);
  HybridMemAssigner hybridMemAssigner(graph);
  ge::Status ret = hybridMemAssigner.Assign();
  size_t offset = 0;
  auto it = hybridMemAssigner.GetMemOffsets().find(RT_MEMORY_HBM);
  if (it != hybridMemAssigner.GetMemOffsets().end()) {
    offset = it->second;
  }

  auto mem_type_session_scope = (kSessionScopeMemory | RT_MEMORY_HBM);
  size_t session_scope_offset = 0;
  it = hybridMemAssigner.GetMemOffsets().find(mem_type_session_scope);
  if (it != hybridMemAssigner.GetMemOffsets().end()) {
    session_scope_offset = it->second;
  }
  EXPECT_EQ(offset, 3584);
  EXPECT_EQ(session_scope_offset, 1536);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestMemoryAssignerTest, multi_batch_reuse) {
  std::map<std::string, std::string> graph_options;
  graph_options[STATIC_MEMORY_POLICY] = "2";
  GetThreadLocalContext().SetGraphOption(graph_options);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeMultiBatchReuseGraph(graph);
  HybridMemAssigner hybridMemAssigner(graph);
  ge::Status ret = hybridMemAssigner.Assign();
  size_t offset = 0;
  auto it = hybridMemAssigner.GetMemOffsets().find(RT_MEMORY_HBM);
  if (it != hybridMemAssigner.GetMemOffsets().end()) {
    offset = it->second;
  }

  auto mem_type_session_scope = (kSessionScopeMemory | RT_MEMORY_HBM);
  size_t session_scope_offset = 0;
  it = hybridMemAssigner.GetMemOffsets().find(mem_type_session_scope);
  if (it != hybridMemAssigner.GetMemOffsets().end()) {
    session_scope_offset = it->second;
  }
  size_t p2p_offset = 0U;
  it = hybridMemAssigner.GetMemOffsets().find(RT_MEMORY_P2P_DDR);
  if (it != hybridMemAssigner.GetMemOffsets().end()) {
    p2p_offset = it->second;
  }
  EXPECT_EQ(offset, 11776);
  EXPECT_EQ(session_scope_offset, 1536);
  EXPECT_EQ(p2p_offset, 4096);
  EXPECT_EQ(ret, SUCCESS);

  GraphMemSplitter graph_mem_splitter(hybridMemAssigner.GetPriorityAssinger()->GetMemoryBlocks(), 512);
  graph_mem_splitter.Split(hybridMemAssigner.GetMemOffsets());
  auto split_offsets = graph_mem_splitter.GetSubMemOffsets();
  EXPECT_EQ(split_offsets.size(), 3);
  graph_options[STATIC_MEMORY_POLICY] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestMemoryAssignerTest, multi_batch_reuse_continuous) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeMultiBatchReuseGraph(graph, true);
  HybridMemAssigner hybridMemAssigner(graph);
  ge::Status ret = hybridMemAssigner.Assign();
  size_t offset = 0;
  auto it = hybridMemAssigner.GetMemOffsets().find(RT_MEMORY_HBM);
  if (it != hybridMemAssigner.GetMemOffsets().end()) {
    offset = it->second;
  }

  auto mem_type_session_scope = (kSessionScopeMemory | RT_MEMORY_HBM);
  size_t session_scope_offset = 0;
  it = hybridMemAssigner.GetMemOffsets().find(mem_type_session_scope);
  if (it != hybridMemAssigner.GetMemOffsets().end()) {
    session_scope_offset = it->second;
  }
  EXPECT_EQ(offset, 11776);
  EXPECT_EQ(session_scope_offset, 1536);
  EXPECT_EQ(ret, SUCCESS);

  GraphMemSplitter graph_mem_splitter(hybridMemAssigner.GetPriorityAssinger()->GetMemoryBlocks(), 512);
  graph_mem_splitter.Split(hybridMemAssigner.GetMemOffsets());
  auto split_offsets = graph_mem_splitter.GetSubMemOffsets();
  EXPECT_EQ(split_offsets.size(), 4);
}

TEST_F(UtestMemoryAssignerTest, graph_memory_assign_atomic_node_assign_success) {
  ge::ComputeGraphPtr graph = MakeAtomicGraph();
  GraphMemoryAssigner memoryAssigner(graph);
  MemoryOffset memory_offset(RT_MEMORY_HBM, 0);
  memoryAssigner.memory_offset_.emplace(RT_MEMORY_HBM, memory_offset);
  map<string, map<NodePtr, vector<NodePtr>>> normal_atomic_nodes_map;
  EXPECT_EQ(memoryAssigner.FilterAtomicNodes(normal_atomic_nodes_map),
            SUCCESS);
}

TEST_F(UtestMemoryAssignerTest, graph_memory_assign_atomic_ref_node_assign_success) {
  ge::ComputeGraphPtr graph = MakeAtomicAndRefGraph();
  GraphMemoryAssigner memoryAssigner(graph);
  MemoryOffset memory_offset(RT_MEMORY_HBM, 0);
  memoryAssigner.memory_offset_.emplace(RT_MEMORY_HBM, memory_offset);
  map<string, map<NodePtr, vector<NodePtr>>> normal_atomic_nodes_map;
  EXPECT_EQ(memoryAssigner.FilterAtomicNodes(normal_atomic_nodes_map),
            SUCCESS);
}

TEST_F(UtestMemoryAssignerTest, graph_memory_assign_atomic_ref_node_assign_failed) {
  ge::ComputeGraphPtr graph = MakeAtomicAndRefGraph();
  auto addn1 = graph->FindNode("addn1");
  vector<int> atomic_out_index = {0, 1};
  ge::AttrUtils::SetListInt(addn1->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, atomic_out_index);
  GraphMemoryAssigner memoryAssigner(graph);
  MemoryOffset memory_offset(RT_MEMORY_HBM, 0);
  memoryAssigner.memory_offset_.emplace(RT_MEMORY_HBM, memory_offset);
  map<string, map<NodePtr, vector<NodePtr>>> normal_atomic_nodes_map;
  EXPECT_EQ(memoryAssigner.FilterAtomicNodes(normal_atomic_nodes_map),
            PARAM_INVALID);
}

/**
 *      data  data
 *        \   /
 *         add
 *          |
 *      NETOUTPUT
 */
TEST_F(UtestMemoryAssignerTest, AtomicClean_Success_CleanOutAndWorkspace_FusionNode) {
  std::vector<int64_t> atomic_output_index = {0};
  auto netout = OP_CFG(NETOUTPUT);
  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto add = OP_CFG(ADD).Attr(ATOMIC_ATTR_IS_ATOMIC_NODE, true);

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("add_1", add));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("add_1", add));
    CHAIN(NODE("add_1", data2)->EDGE(0, 0)->NODE("out_1", netout));
    CHAIN(NODE("memset", MEMSET)->Ctrl()->NODE("add_1", add));

  };

  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  auto node = compute_graph->FindNode("add_1");
  UpdateGraphTensorSize(compute_graph);
  auto op_desc = node->GetOpDesc();
  AttrUtils::SetBool(op_desc, ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  (void)ge::AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);
  (void)ge::AttrUtils::SetBool(op_desc, ATOMIC_ATTR_IS_FUSION_NODE, true);

  map<string, map<int64_t, int64_t>> workspace_info;
  workspace_info["add_1"][0] = 100;
  (void)op_desc->SetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspace_info);

  MemoryAssigner mem_assigner(compute_graph);
  std::map<uint64_t, size_t> mem_offsets;
  size_t zero_copy_mem_size;
  EXPECT_EQ(mem_assigner.AssignMemory(mem_offsets, zero_copy_mem_size), SUCCESS);
}

TEST_F(UtestMemoryAssignerTest, test_no_tiling_mem_assign) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  MockBlockMemAssigner block_mem_assigner(mem_assist_info);

  OpDescPtr op_desc = std::make_shared<OpDesc>("where", "Where");
  GeTensorDesc tensor(GeShape(), FORMAT_ND, DT_INT64);
  op_desc->AddOutputDesc(tensor);
  NodePtr node = graph->AddNode(op_desc);
  vector<int64_t> ranges;
  auto mem_block = block_mem_assigner.ApplyOutDescMemory(node, 0, ranges);
  EXPECT_EQ(mem_block != nullptr, true);
}

TEST_F(UtestMemoryAssignerTest, test_no_tiling_mem_assign_ref) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");

  OpDescPtr op_desc = std::make_shared<OpDesc>("where", "Where");
  GeTensorDesc tensor(GeShape(), FORMAT_ND, DT_INT64);
  op_desc->AddOutputDesc(tensor);
  NodePtr node = graph->AddNode(op_desc);
  vector<int64_t> ranges;

  NodeIndexIO node_index_io(node, 0, kOut);
  std::string symbol = "where";
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  mem_assist_info.anchor_to_symbol = {std::make_pair(node_index_io.ToString(), symbol)};
  MockBlockMemAssigner block_mem_assigner(mem_assist_info);

  MemoryBlock *block = new MemoryBlock(reuse_strategy_, 512);
  block_mem_assigner.symbol_desc_blocks_[symbol] = block;
  auto mem_block = block_mem_assigner.ApplyOutDescMemory(node, 0, ranges);
  EXPECT_EQ(mem_block != nullptr, true);
  delete block;
}

TEST_F(UtestMemoryAssignerTest, test_input_desc_offset_update) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  GraphMemoryAssigner graph_mem_assigner(graph);

  OpDescPtr where_desc = std::make_shared<OpDesc>("where", WHERE);
  GeTensorDescPtr where_tensor = std::make_shared<GeTensorDesc>();
  ge::AttrUtils::SetInt(where_tensor, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 1024);
  where_desc->AddOutputDesc(where_tensor->Clone());
  NodePtr where = graph->AddNode(where_desc);

  OpDescPtr where1_desc = std::make_shared<OpDesc>("where", WHERE);
  GeTensorDescPtr where1_tensor = std::make_shared<GeTensorDesc>();
  ge::AttrUtils::SetBool(where1_tensor, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
  where1_desc->AddInputDesc(where1_tensor->Clone());
  NodePtr where1 = graph->AddNode(where1_desc);

  GraphUtils::AddEdge(where->GetOutDataAnchor(0), where1->GetInDataAnchor(0));

  EXPECT_EQ(graph_mem_assigner.UpdateOpInputDescOffset(where1), SUCCESS);

  uint32_t offset = 0;
  auto tensor = where1_desc->GetInputDescPtr(0);
  ge::AttrUtils::GetInt(tensor, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, offset);
  EXPECT_EQ(offset, 1024);
}

TEST_F(UtestMemoryAssignerTest, test_data_input_desc_offset_update) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  ComputeGraphPtr sub_graph = std::make_shared<ComputeGraph>("sub");
  sub_graph->SetParentGraph(graph);
  GraphMemoryAssigner graph_mem_assigner(graph);

  OpDescPtr p_desc = std::make_shared<OpDesc>("partitioned_call", PARTITIONEDCALL);
  GeTensorDescPtr p_tensor = std::make_shared<GeTensorDesc>();
  ge::AttrUtils::SetInt(p_tensor, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 1024);
  p_desc->AddInputDesc(p_tensor->Clone());
  NodePtr partitioned_call = graph->AddNode(p_desc);
  sub_graph->SetParentNode(partitioned_call);

  OpDescPtr sub_data_desc = std::make_shared<OpDesc>("sub_data", DATA);
  ge::AttrUtils::SetInt(sub_data_desc, ATTR_NAME_PARENT_NODE_INDEX, 0);
  GeTensorDescPtr sub_data_tensor = std::make_shared<GeTensorDesc>();
  ge::AttrUtils::SetBool(sub_data_tensor, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
  sub_data_desc->AddOutputDesc(sub_data_tensor->Clone());
  NodePtr sub_data = sub_graph->AddNode(sub_data_desc);

  EXPECT_EQ(graph_mem_assigner.UpdateOpInputDescOffset(sub_data), SUCCESS);

  uint32_t offset = 0;
  auto tensor = sub_data_desc->GetOutputDescPtr(0);
  ge::AttrUtils::GetInt(tensor, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, offset);
  EXPECT_EQ(offset, 1024);
}

TEST_F(UtestMemoryAssignerTest, test_get_node_memory_type) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  GraphMemoryAssigner graph_mem_assigner(graph);

  OpDescPtr hcom_desc = std::make_shared<OpDesc>("broadcast", HCOMBROADCAST);
  GeTensorDescPtr hcom_tensor = std::make_shared<GeTensorDesc>();
  hcom_desc->AddOutputDesc(hcom_tensor->Clone());
  NodePtr hcom = graph->AddNode(hcom_desc);
  int64_t memory_type = RT_MEMORY_HBM;
  MemoryOffset offset(RT_MEMORY_HBM, 0);
  graph_mem_assigner.memory_offset_.emplace(RT_MEMORY_HBM, offset);
  std::vector<int64_t> mem_type_list = {RT_MEMORY_HBM};
  (void)ge::AttrUtils::SetListInt(hcom_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, mem_type_list);
  EXPECT_EQ(graph_mem_assigner.GetNodeMemoryType(hcom, memory_type, "input"), SUCCESS);
  EXPECT_EQ(graph_mem_assigner.GetNodeMemoryType(hcom, memory_type, "output"), SUCCESS);

  (void)ge::AttrUtils::SetListInt(hcom_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, mem_type_list);
  EXPECT_EQ(graph_mem_assigner.GetNodeMemoryType(hcom, memory_type, "input"), FAILED);
}

TEST_F(UtestMemoryAssignerTest, graph_memory_stream_reuse) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeStreamGraph(graph);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
}

TEST_F(UtestMemoryAssignerTest, graph_ref_memory_reuse) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeRefReuseGraph(graph);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }

  EXPECT_EQ(offset, 14336);
}

TEST_F(UtestMemoryAssignerTest, MemoryBlockTest1) {
  MemoryBlock *block = new MemoryBlock(reuse_strategy_, 512);
  MemoryBlock *block2 = new MemoryBlock(reuse_strategy_, 1024);
  block->no_align_size_list_ = {1,2,3};
  block2->no_align_size_list_ = {1,2,3};
  EXPECT_NO_THROW(block->AddContinuousLifeReuseBlock(*block2));
  block->real_size_list_ = {1,2,3};
  block2->real_size_list_ = {1,2,3};
  block2->is_zero_copy_ = true;
  block->is_zero_copy_ = true;
  EXPECT_NO_THROW(block->AddZeroCopyLifeReuseBlock(*block2));
  block2->is_zero_copy_ = false;
  block->is_zero_copy_ = false;
  block->AddZeroCopyLifeReuseBlock(*block2);
  EXPECT_FALSE(block->is_zero_copy_);

  block2->real_size_list_ = {};
  block->AddZeroCopyLifeReuseBlock(*block2);
  delete block;
  delete block2;
}

TEST_F(UtestMemoryAssignerTest, MemoryBlockTest2) {
  MemoryBlock *block = new MemoryBlock(reuse_strategy_, 512);
  MemoryBlock *block2 = new MemoryBlock(reuse_strategy_, 1024);
  block->no_align_size_list_ = {1,2,3};
  block2->no_align_size_list_ = {1,2,3};
  block2->continuous_block_ = true;
  block->AddContinuousLifeReuseBlock(*block2);
  EXPECT_EQ(block->child_block_, true);

  block->child_block_ = false;
  block->real_size_list_ = {0};
  block2->real_size_list_ = {0};
  block->max_real_size_ = 0;
  block2->max_real_size_ = 0;
  block2->is_zero_copy_ = true;
  block->is_zero_copy_ = false;
  block->AddZeroCopyLifeReuseBlock(*block2);
  EXPECT_EQ(block->child_block_, true);

  block->child_block_ = false;
  block2->is_zero_copy_ = false;
  block->is_zero_copy_ = true;
  block->max_real_size_ = 0;
  block2->max_real_size_ = 0;
  block->AddZeroCopyLifeReuseBlock(*block2);
  EXPECT_EQ(block2->child_block_, true);

  delete block;
  delete block2;
}

TEST_F(UtestMemoryAssignerTest, AddContinuousLifeReuseBlock_HasChildCannotBeChild) {
  MemoryBlock *block = new MemoryBlock(reuse_strategy_, 512);
  MemoryBlock *block2 = new MemoryBlock(reuse_strategy_, 1024);
  MemoryBlock *block3 = new MemoryBlock(reuse_strategy_, 512);
  block->child_blocks_.emplace_back(block3);
  block->no_align_size_list_ = {1,2,3};
  block2->no_align_size_list_ = {1,2,3};
  block2->continuous_block_ = true;
  block->AddContinuousLifeReuseBlock(*block2);
  EXPECT_FALSE(block->child_block_);

  block->child_block_ = false;
  block->real_size_list_ = {0};
  block2->real_size_list_ = {0};
  block->max_real_size_ = 0;
  block2->max_real_size_ = 0;
  block2->is_zero_copy_ = true;
  block->is_zero_copy_ = false;
  block->AddZeroCopyLifeReuseBlock(*block2);
  EXPECT_FALSE(block->child_block_);

  delete block;
  delete block2;
  delete block3;
}

TEST_F(UtestMemoryAssignerTest, graph_ref_not_zero_copy) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeOutputNotZeroCopyGraph(graph);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }

  EXPECT_EQ(offset, 5120);
}

TEST_F(UtestMemoryAssignerTest, graph_hccl_ws_no_reuse_io) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeOutputNotZeroCopyGraph(graph);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);

  MemoryBlock *reuse_block = nullptr;
  EXPECT_EQ(GetReuseBlock(memory_assigner.GetGraphMemoryAssigner()->GetMemAssignerPtr(), graph,
    {"B", kWorkspace, 0}, &reuse_block), GRAPH_SUCCESS);

  EXPECT_NE(reuse_block, nullptr);
  EXPECT_EQ(reuse_block->is_reuse_zero_copy_, false);
}

//
//     A:0    B:1
//      \  /
//     stream_merge1:2(s:0) (two out data anchor, but only has one output node)
//        |
//        C:3(s:0)  D:4(s:1)
//        \         /
//     stream_merge2:5(s:2) (C D output use same symbol)
//          |
//       netoutput:6
//
// 输出使用同一个memory_block的当成一组，在进行复用时，要么同时复用某个节点，要么同时不复用某个节点，共同进退
// C, D输出同符号，C不能复用stream_merge1的第2个输出，因为stream_merge1和D可能并发执行
//
// 与DiffMergeInputNodesNotReuse场景的差异，nopadding continuous input的输入没有将符号merge在一起，
// 在二级CanIntervalLifeReuse中判断same_stream时存在差异。
//
TEST_F(UtestMemoryAssignerTest, DiffMergeInputNodesNotReuse) {
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &node_a = root_builder.AddNode("A", ADD, 0, 1, 1, {1, 1, 44, 448});
  const auto &node_b = root_builder.AddNode("B", ADD, 0, 1, 1, {1, 1, 44, 448});
  const auto &node_stream_merge1 = root_builder.AddNode("stream_merge1", STREAMMERGE, 2, 2, 1, {1, 1, 44, 448});
  const auto &node_c = root_builder.AddNode("C", ADD, 1, 1, 1, {1, 1, 44, 448});
  const auto &node_d = root_builder.AddNode("D", ADD, 0, 1, 1, {1, 1, 44, 448});
  const auto &node_stream_merge2 = root_builder.AddNode("stream_merge2", STREAMMERGE, 2, 2, 1, {1, 1, 44, 448});
  const auto &netout = root_builder.AddNode("NETOUTPUT", NETOUTPUT, 1, 0, 1, {1, 1, 44, 448});

  root_builder.AddDataEdge(node_a, 0, node_stream_merge1, 0);
  root_builder.AddDataEdge(node_b, 0, node_stream_merge1, 1);
  root_builder.AddDataEdge(node_stream_merge1, 0, node_c, 0);
  root_builder.AddDataEdge(node_d, 0, node_stream_merge2, 1);
  root_builder.AddDataEdge(node_c, 0, node_stream_merge2, 0);
  root_builder.AddDataEdge(node_stream_merge2, 0, netout, 0);
  const auto &root_graph = root_builder.GetGraph();

  root_graph->TopologicalSorting();
  node_d->GetOpDescBarePtr()->SetStreamId(1);
  node_stream_merge2->GetOpDescBarePtr()->SetStreamId(2);
  MemoryAssigner memory_assigner(root_graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  EXPECT_TRUE(CheckNotReuse(root_graph, "stream_merge1", 1, "C", 0));
}

//
//       data:0
//        |
//        A:1(s:0) (two out data anchor, but only has one output node)
//        |
//        B:2(s:0)  C:3(s:1)
//        \         /
//          D:4(s:0) (nopadding continuous input)
//
// 输出使用同一个memory_block的当成一组，在进行复用时，要么同时复用某个节点，要么同时不复用某个节点，共同进退
// B C输出同符号，B的block是流0，因此life begin得是流0上的id。由于c是流1，因此找in_stream_edge，由于没有stream1<-stream0的边，因此life_begin_为默认最小值1
//
TEST_F(UtestMemoryAssignerTest, DiffStreamNoPaddingContinuousInputNotReuse) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeDiffStreamSameOutGraphWithNoPaddingContinousInput2(graph);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  EXPECT_TRUE(CheckNotReuse(graph, "A", 1, "B", 0));
}

//        E(two out data anchor, but only has one output node)
//        |
//        A:1(s:0)--+ (two out data anchor, but only has one output node)
//        |         |(ctrl)
//        B:2(s:0)  C:3(s:1)
//        \         /
//          D:4(s:0) (nopadding continuous input)
//
// 输出使用同一个memory_block的当成一组，在进行复用时，要么同时复用某个节点，要么同时不复用某个节点，共同进退
// B C输出同符号，B的block是流0，因此life begin得是流0上的id。由于c是流1，因此找in_stream_edge，找到了A. 因此life_begin_为A(1)
//
TEST_F(UtestMemoryAssignerTest, DiffStreamNoPaddingContinuousInputHasEdgeCanReuse) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeDiffStreamSameOutGraphWithNoPaddingContinousInput3(graph);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  EXPECT_EQ(CheckReuse(memory_assigner.GetGraphMemoryAssigner()->GetMemAssignerPtr(), graph,
                       {{"E", kOutput, 1},
                        {"B", kOutput, 0}}), GRAPH_SUCCESS);
}

//        G:0(s:0)(two out data anchor, but only has one output node)
//        |
//        A:1(s:0)--+
//        |         |(ctrl)
//        |         B:2(s:1)
//        |         |
//        C:3(s:0)  D:4(s:1)
//        \         /
//          E:5(s:0) (nopadding continuous input)
//           |
//           F:5(s:1)
//
// C的block是流0，因此life begin得是流0上的id。由于D是流1，因此找in_stream_edge stream1<-stream0
// 一共有life time[2<-1] and [5<-4]，由于2小于D(4)，因此使用边[2<-1]， 确定life_begin_为1
//
TEST_F(UtestMemoryAssignerTest, DiffStreamNoPaddingContinuousInput_CanNotReuse_HasEdgeButLifeTimeTooBig) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeDiffStreamSameOutGraphWithNoPaddingContinousInput4(graph);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  EXPECT_EQ(CheckReuse(memory_assigner.GetGraphMemoryAssigner()->GetMemAssignerPtr(), graph,
                       {{"G", kOutput, 1},
                        {"C", kOutput, 0}}), GRAPH_SUCCESS);
}

/**     a:1
 *    |     \
 *   b:0    c:1
 *     \    /
 *      d:1
 *        |
 *      e:1
 *        |
 *      f:1
 */
TEST_F(UtestMemoryAssignerTest, graph_diff_stream_memory_reuse) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeDiffStreamGraph(graph);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  EXPECT_EQ(CheckReuse(memory_assigner.GetGraphMemoryAssigner()->GetMemAssignerPtr(), graph,
                       {{"C", kOutput, 0},
                        {"F", kOutput, 0}}), GRAPH_SUCCESS);
}

/**     a:1
 *    |     \
 *   b:0    c:1
 *     \    /\
 *      d:2  e:1
 *            |
 *           f:1
 *  b and f可能并发执行，f不能复用b，而b和c使用统一块内存，所以f不能复用c
 */
TEST_F(UtestMemoryAssignerTest, DiffStreamSameOut_NoPaddingContinousInput_NotReuse) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeDiffStreamSameOutGraphWithNoPaddingContinousInput(graph);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  EXPECT_TRUE(CheckNotReuse(graph, "B", 0, "F", 0));
}

/**     a:1
 *    |     \
 *   b:0      c:1
 *     \     /     \
 *   StreamMerge:2  e:1
 *                   |
 *                  f:1
 *  b and f可能并发执行，f不能复用b，而b和c使用统一块内存，所以f不能复用c
 */
TEST_F(UtestMemoryAssignerTest, DiffStreamSameOut_StreamMerge_NotReuse) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeDiffStreamSameOutGraphWithStreamMerge(graph);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  EXPECT_TRUE(CheckNotReuse(graph, "B", 0, "F", 0));
}

/**     a:1
 *    |     \
 *   b:0    c:1
 *     \    /\
 *      d:1  e:1
 *   (ctrl)\    |
 *           f:1
 *  理论上f可以复用c，当前实现b的life end为无穷大，所以没有复用。所以当前不校验
 */
TEST_F(UtestMemoryAssignerTest, DiffStreamSameOut_NoPaddingContinousInput_Reuse) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeDiffStreamSameOutGraphWithNoPaddingContinousInput(graph, "some", true);

  // set d stream to 1
  auto d = graph->FindNode("D");
  d->GetOpDescBarePtr()->SetStreamId(1);

  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
}

TEST_F(UtestMemoryAssignerTest, graph_cascade_continuous_memory_reuse) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph_cascade_continuous");
  MakeCascadeContinuousGraph(graph);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }

  EXPECT_EQ(offset, 13312);
}
/**
 *
 *           a(0)    b(2)   c(4)
 *           |       |      |
 *           d(1)    e(3)  f(5)
 *           |______|______|
 *                  |
 *          g(7)    h(6)   i(8)  (h output reuse input 1)
 *          |______|______|
 *                 |
 *                 j(9)
 */
TEST_F(UtestMemoryAssignerTest, graph_ref_continuous_input_memory_reuse) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph_cascade_continuous");
  MakeRefContinuousInputGraph(graph);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }

  EXPECT_EQ(offset, 17408);
}

TEST_F(UtestMemoryAssignerTest, graph_continuous_output_memory_reuse) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph_continuous_output");
  MakeContinuousOutputGraph(graph);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }

  EXPECT_EQ(offset, 40960);
}

TEST_F(UtestMemoryAssignerTest, graph_multi_stream_memory_reuse) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph_multi_stream_memory_reuse");
  MakeMultiDiffStreamGraph(graph);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }

  EXPECT_EQ(offset, 19456);
}

TEST_F(UtestMemoryAssignerTest, graph1_multi_stream_memory_reuse) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph1_multi_stream_memory_reuse");
  MakeMultiDiffStreamGraph1(graph);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }

  EXPECT_EQ(offset, 19456);
}

TEST_F(UtestMemoryAssignerTest, graph_multi_stream_dependence_memory_reuse) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph_multi_stream_dependence_memory_reuse");
  MakeDiffStreamDependenceGraph(graph);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }

  EXPECT_EQ(offset, 17408);
}

TEST_F(UtestMemoryAssignerTest, graph_diff_stream_ref_reuse) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph_diff_stream_ref_reuse");
  MakeDiffStreamRefMemGraph(graph);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }

  EXPECT_EQ(offset, 14336);
}

TEST_F(UtestMemoryAssignerTest, graph_diff_stream_ref_no_return_reuse) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph_diff_stream_ref_no_return_reuse");
  MakeDiffStreamRefMemGraph(graph, "some", false);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }

  EXPECT_EQ(offset, 16384);
}

TEST_F(UtestMemoryAssignerTest, graph_first_not_max_life_continuous_reuse) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph_first_not_max_life_continuous_reuse");
  MakeFirstNotMaxLifeContinuousGraph(graph);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }

  EXPECT_EQ(offset, 11264);
}

/**
 *            a
 *            |  \
 *            b    c
 *            |____|
 *            |  |
 *            |  d
 *            |  |
 *            |  e
 *            |  |
 *            |_ f
 *               |
 *               g
 *               |
 *               h
 */
// d need continuous input
// b and c output ref input
TEST_F(UtestMemoryAssignerTest, graph_not_first_continuous_input_as_symbol_reuse) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph_not_first_continuous_input_as_symbol_reuse");
  MakeNotFistContinuousInputAsSymbolGraph(graph, "some", true, false);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }

  EXPECT_EQ(offset, 7168);
}

/**
 *            a
 *            |  \
 *            b    c
 *            |____|
 *               |
 *               d
 *               |
 *               e
 *               |
 *               f
 *               |
 *               g
 *               |
 *               h
 */
// d need continuous input
// b and c output ref input
TEST_F(UtestMemoryAssignerTest, graph_not_first_continuous_input_as_symbol_reuse1) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph_not_first_continuous_input_as_symbol_reuse1");
  MakeNotFistContinuousInputAsSymbolGraph(graph);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }

  EXPECT_EQ(offset, 5120);
}

/**
 *            a
 *            |  \
 *            b    c
 *            |____|
 *               |
 *               d 1
 *               |
 *               e 0
 *               |
 *               f 0
 *               |
 *               g 0
 *               |
 *               h 0
 */
// d need continuous input
// b and c output ref input
TEST_F(UtestMemoryAssignerTest, graph_not_first_continuous_input_as_symbol_reuse2) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph_not_first_continuous_input_as_symbol_reuse2");
  MakeNotFistContinuousInputAsSymbolGraph(graph, "some", false, true);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }

  EXPECT_EQ(offset, 8192);
}

/**
 *            a
 *            |  \
 *            b    c
 *            |____|
 *               | |
 *               d |
 *               | |
 *               e |
 *               | |
 *               f_|
 *               |
 *               g
 *               |
 *               h
 */
// d need continuous input
// b and c output ref input
TEST_F(UtestMemoryAssignerTest, graph_not_first_continuous_input_as_symbol_reuse3) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph_not_first_continuous_input_as_symbol_reuse3");
  MakeNotFistContinuousInputAsSymbolGraph(graph, "some", true, false, false);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }

  EXPECT_EQ(offset, 7168);
}

/**
 *            a
 *            |  \
 *            b    c
 *            |____|
 *               | |
 *               d | 1
 *               | |
 *               e | 0
 *               | |
 *               f_| 0
 *               |
 *               g   0
 *               |
 *               h   0
 */
// d need continuous input
// b and c output ref input
TEST_F(UtestMemoryAssignerTest, graph_not_first_continuous_input_as_symbol_reuse4) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph_not_first_continuous_input_as_symbol_reuse4");
  MakeNotFistContinuousInputAsSymbolGraph(graph, "some", true, true, false);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }

  EXPECT_EQ(offset, 8192);
}

/**
 *            a
 *            |  \
 *            b    c
 *            |____|
 *               | |
 *               d |
 *               | |
 *               e |
 *               | |
 *               f_|
 *               |
 *               g
 *               |
 *               h
 */
// d need continuous input
// b and c output ref input
TEST_F(UtestMemoryAssignerTest, graph_not_first_continuous_input_as_symbol_reuse5) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph_not_first_continuous_input_as_symbol_reuse5");
  MakeNotFistContinuousInputAsSymbolGraph(graph, "some", true, false, false, false);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }

  EXPECT_EQ(offset, 7168);
}
/**
 *            a
 *            |   \
 *            b    c
 *            |____|
 *               | |
 *               d | 1
 *               | |
 *               e | 0
 *               | |
 *               f_| 0
 *               |
 *               g   0
 *               |
 *               h   0
 */
// d need continuous input
// b and c output ref input
TEST_F(UtestMemoryAssignerTest, graph_not_first_continuous_input_as_symbol_reuse6) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph_not_first_continuous_input_as_symbol_reuse6");
  MakeNotFistContinuousInputAsSymbolGraph(graph, "some", true, true, false, false);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }

  EXPECT_EQ(offset, 8192);
}

/**
 *            a
 *            |  \
 *            b    c
 *            |____|
 *               | |
 *               d |
 *               | |
 *               e |
 *               | |
 *               f_|
 *               |
 *               g
 *               |
 *               h
 */
// d need continuous input and seperate clean input
// b and c output ref input
TEST_F(UtestMemoryAssignerTest, graph_not_first_continuous_input_as_symbol_reuse7) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph_not_first_continuous_input_as_symbol_reuse7");
  MakeNotFistContinuousInputAsSymbolGraph(graph, "some", true, false, false, false, false);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }

  EXPECT_EQ(offset, 9216);
}

TEST_F(UtestMemoryAssignerTest, graph_ref_variable) {
  VarManager::Instance(0)->Destory();
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  ge::ComputeGraphPtr graph = MakeRefVariableGraph();
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }
  EXPECT_EQ(offset, 0);
  auto node = graph->FindNode("add");
  EXPECT_NE(node, nullptr);
  // not assign memory, use variable's memory
  vector<int64_t> out_list = node->GetOpDesc()->GetOutputOffset();
  EXPECT_EQ(out_list[0], VarManager::Instance(0)->GetVarMemLogicBase());
  node = graph->FindNode("add1");
  EXPECT_NE(node, nullptr);
  vector<int64_t> in_list = node->GetOpDesc()->GetInputOffset();
  EXPECT_EQ(in_list[0], VarManager::Instance(0)->GetVarMemLogicBase());
}

TEST_F(UtestMemoryAssignerTest, graph_sort_test) {
  VarManager::Instance(0)->Destory();
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph1_sort_test");
  MakeSortTestGraph(graph);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }
  EXPECT_EQ(offset, 9216);
}

// 单独清零场景，两块输入连续的内存之间可以复用
// C输入为A，B，F输入为D，E，C和F都是要求输入连续，D的输出和A的输出可以复用
TEST_F(UtestMemoryAssignerTest, graph_continuous_input_memory_reuse) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph_continuous_input_reuse");
  MakePaddingInputContinuousReuseGraph(graph);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }
  EXPECT_EQ(CheckReuse(memory_assigner.GetGraphMemoryAssigner()->GetMemAssignerPtr(), graph,
      {{"A", kOutput, 0},
       {"D", kOutput, 0}
      }), GRAPH_SUCCESS);
  EXPECT_EQ(offset, 9216);
}

// 单独清零场景，两块输入连续的内存之间可以复用
// C输入为A，B，G输入为A，F，C和G都是要求输入连续，A,B,F,I复用
TEST_F(UtestMemoryAssignerTest, graph_continuous_input_memory_reuse_multi_ref) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph_continuous_input_reuse_multi_ref");
  MakePaddingInputContinuousReuseGraphMultiRef(graph);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }
  EXPECT_EQ(CheckReuse(memory_assigner.GetGraphMemoryAssigner()->GetMemAssignerPtr(), graph,
      {{"A", kOutput, 0},
       {"B", kOutput, 0},
       {"F", kOutput, 0},
       {"I", kOutput, 0}
       }), GRAPH_SUCCESS);
  EXPECT_EQ(offset, 17408);
}

TEST_F(UtestMemoryAssignerTest, graph_continuous_input_memory_reuse_multi_ref_diff_stream) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph_continuous_input_reuse_multi_ref_diff_stream");
  MakePaddingInputContinuousReuseGraphMultiRef(graph, true);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  // 打印日志才能覆盖到新增代码
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  EXPECT_EQ(offset, 20480);
}

TEST_F(UtestMemoryAssignerTest, test_l2_memory_type) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeGraph(graph);
  auto op_b = graph->FindNode("B");
  (void)ge::AttrUtils::SetListInt(op_b->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, {RT_MEMORY_L2});
  map<uint64_t, size_t> mem_offset;
  size_t zero_copy_mem_size = 0;
  MemoryAssigner memoryAssigner(graph);
  EXPECT_EQ(memoryAssigner.AssignMemory(mem_offset, zero_copy_mem_size), SUCCESS);
  EXPECT_EQ(mem_offset.size(), 2U);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }

  EXPECT_EQ(offset, 459264);
}

TEST_F(UtestMemoryAssignerTest, test_host_memory_type) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeGraph(graph);
  auto op_b = graph->FindNode("B");
  (void)ge::AttrUtils::SetListInt(op_b->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, {RT_MEMORY_HOST});
  map<uint64_t, size_t> mem_offset;
  size_t zero_copy_mem_size = 0;
  MemoryAssigner memoryAssigner(graph);
  EXPECT_EQ(memoryAssigner.AssignMemory(mem_offset, zero_copy_mem_size), SUCCESS);
  EXPECT_EQ(mem_offset.size(), 3U);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }
  EXPECT_EQ(offset, 458240);

  it = mem_offset.find(RT_MEMORY_HOST);
  if (it != mem_offset.end()) {
    offset = it->second;
  }
  EXPECT_EQ(offset, 68719477760);
}

TEST_F(UtestMemoryAssignerTest, test_p2p_memory_type) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MakeGraph(graph);
  auto op_b = graph->FindNode("B");
  (void)ge::AttrUtils::SetListInt(op_b->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, {RT_MEMORY_P2P_DDR});
  map<uint64_t, size_t> mem_offset;
  size_t zero_copy_mem_size = 0;
  MemoryAssigner memoryAssigner(graph);
  EXPECT_EQ(memoryAssigner.AssignMemory(mem_offset, zero_copy_mem_size), SUCCESS);
  EXPECT_EQ(mem_offset.size(), 2U);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }
  EXPECT_EQ(offset, 458240);

  it = mem_offset.find(RT_MEMORY_P2P_DDR);
  if (it != mem_offset.end()) {
   offset = it->second;
  }
  EXPECT_EQ(offset, 1024);
}

TEST_F(UtestMemoryAssignerTest, graph_option_test) {
  VarManager::Instance(0)->Destory();
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  OptionSetter option({{OPTION_EXEC_DISABLE_REUSED_MEMORY, "1"}});
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph_option_test");
  MakeSortTestGraph(graph);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }
  // no memory reuse
  EXPECT_EQ(offset, 15360);
}

TEST_F(UtestMemoryAssignerTest, InputOutputNodeReuseMemFlagTest) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  MemAssistInfo mem_assist_info {.compute_graph = graph};
  MockBlockMemAssigner block_mem_assigner(mem_assist_info);

  block_mem_assigner.output_index_to_reuse_mem_flag_ = {true, false, true};
  bool flag = block_mem_assigner.GetOutputNodeReuseMemFlagByIndex(0);
  EXPECT_EQ(flag, true);
  flag = block_mem_assigner.GetOutputNodeReuseMemFlagByIndex(1);
  EXPECT_EQ(flag, false);
  flag = block_mem_assigner.GetOutputNodeReuseMemFlagByIndex(3);
  EXPECT_EQ(flag, false);

  OpDescPtr data_desc1 = std::make_shared<OpDesc>("data1", DATA);
  ge::AttrUtils::SetInt(data_desc1, ATTR_NAME_PARENT_NODE_INDEX, 0);
  GeTensorDescPtr data_tensor = std::make_shared<GeTensorDesc>();
  ge::AttrUtils::SetBool(data_tensor, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
  data_desc1->AddOutputDesc(data_tensor->Clone());
  NodePtr data1 = graph->AddNode(data_desc1);
  block_mem_assigner.input_index_to_reuse_mem_flag_ = {true, true, false};
  flag = block_mem_assigner.GetInputNodeReuseMemFlag(data1);
  EXPECT_EQ(flag, false);

  OpDescPtr data_desc2 = std::make_shared<OpDesc>("data2", DATA);
  ge::AttrUtils::SetInt(data_desc2, ATTR_NAME_INDEX, 0);
  data_desc2->AddOutputDesc(data_tensor->Clone());
  NodePtr data2 = graph->AddNode(data_desc2);
  flag = block_mem_assigner.GetInputNodeReuseMemFlag(data2);
  EXPECT_EQ(flag, true);
}

TEST_F(UtestMemoryAssignerTest, ParseIoReuseMemOptionTest) {
  auto graph1 = gert::ShareGraph::BuildSwitchMergeGraph();
  auto compute_graph1 = GraphUtilsEx::GetComputeGraph(graph1);

  MemAssistInfo mem_assist_info {.compute_graph = compute_graph1};
  MockBlockMemAssigner block_mem_assigner(mem_assist_info);

  std::map<std::string, std::string> options;
  options.emplace(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES, "0,1");
  options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "0,1,2");
  GetThreadLocalContext().SetGraphOption(options);

  block_mem_assigner.ParseIoReuseMemOption();
  EXPECT_EQ((block_mem_assigner.input_index_to_reuse_mem_flag_.size() > 0), true);
  EXPECT_EQ((block_mem_assigner.output_index_to_reuse_mem_flag_.size() > 0), true);
}

TEST_F(UtestMemoryAssignerTest, graph_continuous_input_memory_reuse_fail) {
  OptionSetter option({{"ge.hardwareInfo", "memory_size:1024"}});
  VarManager::Instance(0)->SetMemoryMallocSize({{}}, 1024UL);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph_continuous_input_reuse");
  MakePaddingInputContinuousReuseGraph(graph);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0U;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), ge::FAILED);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }

  EXPECT_EQ(offset, 9216);
}

TEST_F(UtestMemoryAssignerTest, ReAssignContinuousMemoryFail) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  OpDescPtr op_desc_two = std::make_shared<OpDesc>("node_two", "type");
  NodePtr node_two = graph->AddNode(op_desc_two);
  std::vector<int64_t> mem_type_list;
  mem_type_list.emplace_back(66);
  ge::AttrUtils::SetListInt(node_two->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, mem_type_list);
  ge::AttrUtils::SetBool(node_two->GetOpDesc(), ATTR_NAME_CONTINUOUS_OUTPUT, true);

  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0U;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), FAILED);
}

TEST_F(UtestMemoryAssignerTest, SingleOutDiffStreamOptimize) {
  map<string, string> options{{"ge.hardwareInfo", "memory_size:102400"},
                              {MEMORY_OPTIMIZATION_POLICY, kMemoryPriority}};

  options.emplace(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES, "0,1");
  options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "0,1,2");
  GetThreadLocalContext().SetGraphOption({});
  OptionSetter temp_option(options);
  auto b = OP_CFG("some").InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 32, 32});;
  auto e = OP_CFG("some").InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 512, 512});
  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", DATA)
              ->NODE("A", "some")
              ->NODE("B", b)
              ->NODE("C", "some")
              ->NODE("D", "some")
              ->NODE("E", e)
              ->NODE("F", "some")
              ->NODE("Node_Output", NETOUTPUT));
  };
  ComputeGraphPtr graph = ToComputeGraph(g1);
  UpdateGraphTensorSize(graph);
  graph->TopologicalSorting();
  SetStreamId(graph, "A", 0);
  SetStreamId(graph, "B", 1);
  SetStreamId(graph, "C", 2);
  SetStreamId(graph, "D", 0);
  SetStreamId(graph, "E", 0);
  SetStreamId(graph, "F", 0);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0U;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }
  EXPECT_EQ(GetStreamId(graph, "B"), GetStreamId(graph, "A"));
  EXPECT_EQ(GetStreamId(graph, "C"), GetStreamId(graph, "A"));

  // A,C,E可以复用
  EXPECT_EQ(CheckReuse(memory_assigner.GetGraphMemoryAssigner()->GetMemAssignerPtr(), graph,
                       {{"A", kOutput, 0},
                        {"C", kOutput, 0},
                        {"E", kOutput, 0}
                       }), GRAPH_SUCCESS);
  EXPECT_EQ(offset, 1250304);
}
/*
 *        _arg_0
 *          |
 * (stream0)A
 *          |
 * (stream1)B
 *          |
 * (stream2)C-ctr->D(stream0)
 *          |    /
 *          |  /
 * (stream0)E
 *          |
 * (stream0)F
 *          |
 * (stream0)G
 *          |
 *       Node_Output
 *
 * stream 0->1->2->0,优先流间复用
 * 校验重点：虽然A是流0上的block，在流0上的结束点是4(D)。但是A和C(3)也是串行关系，也是可以复用的。
 * 在CanBlockLifeReuse中开启memory_priority_mode_下，判断出来的a和c可以复用
 * 为什么要放在内存优先option控制下？是因为优先流间复用无法保证最优。
 * 更优方案是：为B设置sub stream id，设置为0，使得流间关系更近。
 *
 * 与SingleOutDiffStreamOptimize用例的区别在与，C-D-E都是数据边，所以MemReuseStrategy::OptimizeDiffStream中会都标记成stream 0
 * 而本用例中，E有两个输入，C和D 是控制边，所以MemReuseStrategy::OptimizeDiffStream函数没有处理。
 * 期望有一个更为通用的算法，将符合条件的串行关系，都标记上sub_stream id.
 */
TEST_F(UtestMemoryAssignerTest, SingleOutDiffStream) {
  map<string, string> options{{"ge.hardwareInfo", "memory_size:102400"},
                              {MEMORY_OPTIMIZATION_POLICY, kMemoryPriority}};

  options.emplace(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES, "0,1");
  options.emplace(ge::OPTION_INPUT_REUSE_MEM_INDEXES, "0,1,2");
  GetThreadLocalContext().SetGraphOption({});
  OptionSetter temp_option(options);
  auto b = OP_CFG("some").InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 32, 32});;
  auto e = OP_CFG("some").InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 512, 512});
  auto f = OP_CFG("some").InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 2, 512, 512});
  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", DATA)
              ->NODE("A", "some")
              ->NODE("B", b)
              ->NODE("C", "some")
              ->NODE("E", e)
              ->NODE("F", f)
              ->NODE("G", "some")
              ->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("D", "some")->NODE("E", e));
    CHAIN(NODE("C", "some")->CTRL_EDGE()->NODE("D", "some"));
  };
  ComputeGraphPtr graph = ToComputeGraph(g1);
  UpdateGraphTensorSize(graph);
  graph->TopologicalSorting();
  SetStreamId(graph, "A", 0);
  SetStreamId(graph, "B", 1);
  SetStreamId(graph, "C", 2);
  SetStreamId(graph, "D", 0);
  SetStreamId(graph, "E", 0);
  SetStreamId(graph, "F", 0);
  SetStreamId(graph, "G", 0);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0U;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }

  EXPECT_EQ(GetStreamId(graph, "B"), 1);
  EXPECT_EQ(GetStreamId(graph, "C"), 2);

  // A,C,F可以复用
  EXPECT_EQ(CheckReuse(memory_assigner.GetGraphMemoryAssigner()->GetMemAssignerPtr(), graph,
                       {{"A", kOutput, 0},
                           {"C", kOutput, 0},
                           {"F", kOutput, 0},
                       }), GRAPH_SUCCESS);
  EXPECT_EQ(offset, 3346944);
}
// stream 0->1->2->0 ,遇到STREAMMERGE，不能复用
TEST_F(UtestMemoryAssignerTest, SingleOutDiffStreamWithMerge) {
  DEF_GRAPH(g1) {
      CHAIN(NODE("_arg_0", DATA)
      ->NODE("A", "some")
      ->NODE("B", "some")
      ->NODE("C", STREAMMERGE)
      ->NODE("D", "some")
      ->NODE("E", "some")
      ->NODE("Node_Output", NETOUTPUT));
  };
  ComputeGraphPtr graph = ToComputeGraph(g1);
  UpdateGraphTensorSize(graph);
  graph->TopologicalSorting();
  SetStreamId(graph, "A", 0);
  SetStreamId(graph, "B", 1);
  SetStreamId(graph, "C", 2);
  SetStreamId(graph, "D", 0);
  SetStreamId(graph, "E", 0);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0U;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }
  const auto blocks = memory_assigner.GetGraphMemoryAssigner()->GetMemAssignerPtr()->GetMemoryBlocks();
  bool has_checked = false;
  for (const auto block : blocks) {
    for (const auto &node : block->node_type_index_list_) {
      if (node.mem_type_ != kOutput) {
        continue;
      }
      if (node.node_->GetOpDesc()->GetName() == "A") {
        EXPECT_EQ(block->GetLifeEnd(2), kMaxLifeTime);
        EXPECT_EQ(block->GetLifeEnd(1), 2);
        has_checked = true;
      }
    }
  }
  EXPECT_TRUE(has_checked);
  EXPECT_EQ(offset, 203264);
}

//     const
//      |
//      a
//      |                +-----------+
//  partitioncall0-------| data      |
//      |                |  |        |
//      d                |  b        |
//      |                |  |        |
//  netoutput            |  c        |
//                       |  |        |
//                       | hcom(输入清零)|
//                       |  |        |
//                       |netoutput1 |
//                       +-----------+
//
TEST_F(UtestMemoryAssignerTest, AtomicCleanNotReuseWithSubGraphData) {
  auto graph = block_mem_ut::BuildAtomicCleanNotReuseSubGraphData();
  UpdateGraphTensorSize(graph);
  graph->TopologicalSorting();

  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0U;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  // a c 不复用
  EXPECT_EQ(CheckReuse(memory_assigner.GetGraphMemoryAssigner()->GetMemAssignerPtr(), graph,
                       {{"a", kOutput, 0},
                        {"c", kOutput, 0}
                       }, false), GRAPH_SUCCESS);
}

//   Data
//     |----------
//     |          |
//  D stream 0   E stream 1
//  Data不是实际执行节点，产生stream 1->stream 0的依赖会导致错误结果
TEST_F(UtestMemoryAssignerTest, DiffStreamEdgeExcludeData) {
  auto b = OP_CFG("some").InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 32, 32});;
  auto e = OP_CFG("some").InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 512, 512});
  DEF_GRAPH(g1) {
      CHAIN(NODE("_arg_0", DATA)
          ->NODE("A", "some")
          ->NODE("B", b)
          ->NODE("C", "some")
          ->NODE("Node_Output", NETOUTPUT));
      CHAIN(NODE("_arg_1", DATA)
          ->NODE("D", "some")
          ->NODE("Node_Output", NETOUTPUT));
      CHAIN(NODE("E", e)
          ->NODE("F", "some")
          ->NODE("Node_Output", NETOUTPUT));
      CHAIN(NODE("C", "some")->CTRL_EDGE()->NODE("_arg_1", DATA));
      CHAIN(NODE("_arg_1", DATA)->CTRL_EDGE()->NODE("E", e));
  };
  ComputeGraphPtr graph = ToComputeGraph(g1);
  UpdateGraphTensorSize(graph);
  graph->TopologicalSorting();
  SetStreamId(graph, "_arg_0", -1);
  SetStreamId(graph, "_arg_1", -1);
  SetStreamId(graph, "A", 0);
  SetStreamId(graph, "B", 0);
  SetStreamId(graph, "C", 0);
  SetStreamId(graph, "D", 0);
  SetStreamId(graph, "E", 1);
  SetStreamId(graph, "F", 1);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0U;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  size_t offset = 0;
  auto it = mem_offset.find(RT_MEMORY_HBM);
  if (it != mem_offset.end()) {
    offset = it->second;
  }
  // A,E不复用
  EXPECT_EQ(CheckReuse(memory_assigner.GetGraphMemoryAssigner()->GetMemAssignerPtr(), graph,
                     {{"A", kOutput, 0},
                      {"E", kOutput, 0}
                     }, false), GRAPH_SUCCESS);

  EXPECT_EQ(offset, 1456128);
}

/*
 *     data
 *      |
 *      a
 *      |
 *  partitioned_call    +----------------------+
 *      |               | inner_data   memset  |
 *      |               |     |       /(ctl)   |
 *      |               | atomic_node          |
 *      |               |     |                |
 *      |               |  reshape             |
 *      |               |     |                |
 *      b               | netoutput2           |
 *      |               +----------------------+
 *    netoutput1
 */
TEST_F(UtestMemoryAssignerTest, PartitionedCallWithAtomicNode_CheckOffsetSuccess) {
  auto graph = block_mem_ut::BuildPartitionedCallWithAtomicNode();
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0U;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);

  auto partitioned_call = graph->FindNode("partitioned_call");
  NodePtr atomic_node = nullptr;
  for (auto node : graph->GetAllNodes()) {
    if (node->GetName() == "atomic_node") {
      atomic_node = node;
    }
  }
  ASSERT_NE(partitioned_call, nullptr);
  ASSERT_NE(atomic_node, nullptr);
  auto partitioned_call_offsets = partitioned_call->GetOpDesc()->GetOutputOffset();
  auto atomic_node_offsets = atomic_node->GetOpDesc()->GetOutputOffset();
  ASSERT_FALSE(partitioned_call_offsets.empty());
  ASSERT_FALSE(atomic_node_offsets.empty());
  EXPECT_EQ(partitioned_call_offsets.at(0), atomic_node_offsets.at(0));
}

TEST_F(UtestMemoryAssignerTest, InvalidWorkSpace) {
  DEF_GRAPH(g1) {
      CHAIN(NODE("_arg_0", DATA)
                ->NODE("A", "some")
                ->NODE("B", "some")
                ->NODE("C", "some")
                ->NODE("Node_Output", NETOUTPUT));
  };
  ComputeGraphPtr graph = ToComputeGraph(g1);
  const auto call_node = graph->FindNode("B");
  EXPECT_NE(call_node, nullptr);
  const auto call_desc = call_node->GetOpDesc();
  EXPECT_NE(call_desc, nullptr);
  std::vector<int64_t> workspace_bytes;
  workspace_bytes.push_back(-2);
  call_desc->SetWorkspaceBytes(workspace_bytes);
  UpdateGraphTensorSize(graph);
  graph->TopologicalSorting();
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0U;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_FAILED);
}

/*
 *     hcom1   b
 *    /  | |  /
 *   a   hcom2
 *    \ /
 *     c
 *     |
 *     d
 *     |
 *   netoutput
 */
TEST_F(UtestMemoryAssignerTest, ContinuousOutIn_SUCCESS_OutAsHeader) {
  auto graph = block_mem_ut::BuildContinuousOutInWithOutAsHeader();
  std::map<uint64_t, size_t> mem_type_to_mem_offset;
  size_t zero_copy_mem_size;
  VarManager::Instance(graph->GetSessionID())->Init(0, graph->GetSessionID(), 0, 0);
  MemoryAssigner mem_assigner(graph);
  EXPECT_EQ(mem_assigner.AssignMemory(mem_type_to_mem_offset, zero_copy_mem_size), SUCCESS);
}

/*
 *     hcom1   b
 *    /  | |  /
 *   a   hcom2
 *    \ /
 *     c
 *     |
 *     d
 *     |
 *   netoutput
 */
TEST_F(UtestMemoryAssignerTest, ContinuousOutIn_SUCCESS_OutAsHeader_SeparateClean) {
  OptionSetter option({{ATOMIC_CLEAN_POLICY, "1"}});
  auto graph = block_mem_ut::BuildContinuousOutInWithOutAsHeader();
  std::map<uint64_t, size_t> mem_type_to_mem_offset;
  size_t zero_copy_mem_size;
  VarManager::Instance(graph->GetSessionID())->Init(0, graph->GetSessionID(), 0, 0);
  MemoryAssigner mem_assigner(graph);
  EXPECT_EQ(mem_assigner.AssignMemory(mem_type_to_mem_offset, zero_copy_mem_size), SUCCESS);
  EXPECT_EQ(CheckReuse(mem_assigner.GetGraphMemoryAssigner()->GetMemAssignerPtr(), graph,
                       {{"hcom1", kOutput, 0},
                        {"c", kOutput, 0}
                       }, true), GRAPH_SUCCESS);
}

/*
 *     hcom1
 *    / | |  \
 *   a  hcom2  b
 *   |         ^
 *   c---------|(ctrl edge)
 *   |
 *   d
 *   |
 *   netoutput
 */
TEST_F(UtestMemoryAssignerTest, ContinuousOutIn_SUCCESS_OutAsHeaderAndTail_SeparateClean) {
  OptionSetter option({{ATOMIC_CLEAN_POLICY, "1"}});
  auto graph = block_mem_ut::BuildContinuousOutInWithOutAsHeaderAndTail();
  std::map<uint64_t, size_t> mem_type_to_mem_offset;
  size_t zero_copy_mem_size;
  VarManager::Instance(graph->GetSessionID())->Init(0, graph->GetSessionID(), 0, 0);
  MemoryAssigner mem_assigner(graph);
  EXPECT_EQ(mem_assigner.AssignMemory(mem_type_to_mem_offset, zero_copy_mem_size), SUCCESS);
  EXPECT_EQ(CheckReuse(mem_assigner.GetGraphMemoryAssigner()->GetMemAssignerPtr(), graph,
                       {{"hcom1", kOutput, 0},
                        {"c", kOutput, 0}
                       }, false), GRAPH_SUCCESS);
}

/*
 *  a  hcom1
 *   \  | | \
 *     hcom2 b
 *       |   ^
 *       c---| (ctrl edge)
 *       |
 *       d
 *       |
 *    netoutput
 */
TEST_F(UtestMemoryAssignerTest, ContinuousOutIn_SUCCESS_InAsHeader_SeparateClean) {
  OptionSetter option({{ATOMIC_CLEAN_POLICY, "1"}});
  auto graph = block_mem_ut::BuildContinuousOutInWithInAsHeader();
  std::map<uint64_t, size_t> mem_type_to_mem_offset;
  size_t zero_copy_mem_size;
  VarManager::Instance(graph->GetSessionID())->Init(0, graph->GetSessionID(), 0, 0);
  MemoryAssigner mem_assigner(graph);
  EXPECT_EQ(mem_assigner.AssignMemory(mem_type_to_mem_offset, zero_copy_mem_size), SUCCESS);
  EXPECT_EQ(CheckReuse(mem_assigner.GetGraphMemoryAssigner()->GetMemAssignerPtr(), graph,
                       {{"hcom1", kOutput, 0},
                        {"c", kOutput, 0}
                       }, false), GRAPH_SUCCESS);
  EXPECT_EQ(CheckReuse(mem_assigner.GetGraphMemoryAssigner()->GetMemAssignerPtr(), graph,
                       {{"a", kOutput, 0},
                        {"c", kOutput, 0}
                       }, false), GRAPH_SUCCESS);
}

/*
 *  a  hcom1 b
 *   \  | | /
 *     hcom2
 */
TEST_F(UtestMemoryAssignerTest, ContinuousOutIn_SUCCESS_InAsHeaderAndTail) {
  auto graph = block_mem_ut::BuildContinuousOutInWithInAsHeaderAndTail();
  std::map<uint64_t, size_t> mem_type_to_mem_offset;
  size_t zero_copy_mem_size;
  VarManager::Instance(graph->GetSessionID())->Init(0, graph->GetSessionID(), 0, 0);
  MemoryAssigner mem_assigner(graph);
  EXPECT_EQ(mem_assigner.AssignMemory(mem_type_to_mem_offset, zero_copy_mem_size), SUCCESS);
}

/*
 *           c
 *           |
 *           d
 *           |
 *  a  hcom1 b
 *   \  | | /
 *     hcom2
 *
 *  设置a为stream1,其他都stream0, 校验c和a无法复用
 *  topo序： "c", "d", "b", "hcom1", "a", "hcom2"
 */
TEST_F(UtestMemoryAssignerTest, ContinuousOutIn_SUCCESS_InAsHeaderAndTail_SeparateClean) {
  OptionSetter option({{ATOMIC_CLEAN_POLICY, "1"}});
  auto graph = block_mem_ut::BuildContinuousOutInWithInAsHeaderAndTail();
  std::map<uint64_t, size_t> mem_type_to_mem_offset;
  size_t zero_copy_mem_size;
  VarManager::Instance(graph->GetSessionID())->Init(0, graph->GetSessionID(), 0, 0);
  MemoryAssigner mem_assigner(graph);
  EXPECT_EQ(mem_assigner.AssignMemory(mem_type_to_mem_offset, zero_copy_mem_size), SUCCESS);
  EXPECT_EQ(CheckReuse(mem_assigner.GetGraphMemoryAssigner()->GetMemAssignerPtr(), graph,
                       {{"c", kOutput, 0},
                        {"b", kOutput, 0}
                       }, false), GRAPH_SUCCESS);
  EXPECT_EQ(CheckReuse(mem_assigner.GetGraphMemoryAssigner()->GetMemAssignerPtr(), graph,
                       {{"c", kOutput, 0},
                        {"hcom1", kOutput, 0}
                       }, false), GRAPH_SUCCESS);
  EXPECT_EQ(CheckReuse(mem_assigner.GetGraphMemoryAssigner()->GetMemAssignerPtr(), graph,
                       {{"c", kOutput, 0},
                        {"a", kOutput, 0}
                       }, false), GRAPH_SUCCESS);
}

/*
 *      hcom1
 *   / ||   ||  \
 *  a hcom2 hcom3 b
 */
TEST_F(UtestMemoryAssignerTest, ContinuousOutIn_SUCCESS_TwoContinuousInNode) {
  auto graph = block_mem_ut::BuildContinuousOutInWithTwoContinuousInNode();
  std::map<uint64_t, size_t> mem_type_to_mem_offset;
  size_t zero_copy_mem_size;
  VarManager::Instance(graph->GetSessionID())->Init(0, graph->GetSessionID(), 0, 0);
  MemoryAssigner mem_assigner(graph);
  EXPECT_EQ(mem_assigner.AssignMemory(mem_type_to_mem_offset, zero_copy_mem_size), SUCCESS);
}

/*
 *      hcom1
 *   / ||   ||  \
 *  a hcom2 hcom3 b
 *                |
 *                c
 *                |
 *                d
 *  a stream1, others: stream0
 *  topo order: hcom1, a, hcom2, hcom3
 *  校验hcom1和c不复用
 */
TEST_F(UtestMemoryAssignerTest, ContinuousOutIn_SUCCESS_TwoContinuousInNode_SeparateClean) {
  OptionSetter option({{ATOMIC_CLEAN_POLICY, "1"}});
  auto graph = block_mem_ut::BuildContinuousOutInWithTwoContinuousInNode();
  std::map<uint64_t, size_t> mem_type_to_mem_offset;
  size_t zero_copy_mem_size;
  VarManager::Instance(graph->GetSessionID())->Init(0, graph->GetSessionID(), 0, 0);
  MemoryAssigner mem_assigner(graph);
  EXPECT_EQ(mem_assigner.AssignMemory(mem_type_to_mem_offset, zero_copy_mem_size), SUCCESS);
  EXPECT_EQ(CheckReuse(mem_assigner.GetGraphMemoryAssigner()->GetMemAssignerPtr(), graph,
                       {{"c", kOutput, 0},
                        {"hcom1", kOutput, 0}
                       }, false), GRAPH_SUCCESS);
}

/*
 *                     e
 *                     |
 *                     f
 *                     |
 *      hcom1        c d
 *   / ||   ||  \  \ | |
 *  a hcom2 hcom3 b hcom4
 *
 *  topo order: "hcom1", "e", "f", "c", "d
 */
TEST_F(UtestMemoryAssignerTest, ContinuousOutIn_SUCCESS_TwoContinuousInNodeAndRefNode_SeparateClean) {
  OptionSetter option({{ATOMIC_CLEAN_POLICY, "1"}});
  auto graph = block_mem_ut::BuildContinuousOutInWithTwoContinuousInNodeAndRefNode();
  std::map<uint64_t, size_t> mem_type_to_mem_offset;
  size_t zero_copy_mem_size;
  VarManager::Instance(graph->GetSessionID())->Init(0, graph->GetSessionID(), 0, 0);
  MemoryAssigner mem_assigner(graph);
  InfoLog log; // 打印info日志才能覆盖到一些分支
  EXPECT_EQ(mem_assigner.AssignMemory(mem_type_to_mem_offset, zero_copy_mem_size), SUCCESS);
  EXPECT_EQ(CheckReuse(mem_assigner.GetGraphMemoryAssigner()->GetMemAssignerPtr(), graph,
                       {{"e", kOutput, 0},
                        {"hcom1", kOutput, 0}
                       }, false), GRAPH_SUCCESS);
}
/*
 *    a hcom1 b hcom2 c
 *    \  ||   |  ||   |
 *          hcom3
 */
TEST_F(UtestMemoryAssignerTest, ContinuousOutIn_SUCCESS_TwoContinuousOutNodeAndRefNode) {
  auto graph = block_mem_ut::BuildContinuousOutInWithTwoContinuousOutNodeAndRefNode();
  std::map<uint64_t, size_t> mem_type_to_mem_offset;
  size_t zero_copy_mem_size;
  VarManager::Instance(graph->GetSessionID())->Init(0, graph->GetSessionID(), 0, 0);
  MemoryAssigner mem_assigner(graph);
  EXPECT_EQ(mem_assigner.AssignMemory(mem_type_to_mem_offset, zero_copy_mem_size), SUCCESS);
}
/*
 *    hcom1
 *   / |   \
 *  a hcom2  c
 */
TEST_F(UtestMemoryAssignerTest, ContinuousOutIn_SUCCESS_OneInput) {
  auto graph = block_mem_ut::BuildContinuousOutInWithOneInput();
  std::map<uint64_t, size_t> mem_type_to_mem_offset;
  size_t zero_copy_mem_size;
  VarManager::Instance(graph->GetSessionID())->Init(0, graph->GetSessionID(), 0, 0);
  MemoryAssigner mem_assigner(graph);
  EXPECT_EQ(mem_assigner.AssignMemory(mem_type_to_mem_offset, zero_copy_mem_size), SUCCESS);
}
/*
 *    hcom1
 *   / |   \
 *  a hcom2  c
 */
TEST_F(UtestMemoryAssignerTest, ContinuousOutIn_SUCCESS_OneOutput) {
  auto graph = block_mem_ut::BuildContinuousOutInWithOneOutput();
  std::map<uint64_t, size_t> mem_type_to_mem_offset;
  size_t zero_copy_mem_size;
  VarManager::Instance(graph->GetSessionID())->Init(0, graph->GetSessionID(), 0, 0);
  MemoryAssigner mem_assigner(graph);
  EXPECT_EQ(mem_assigner.AssignMemory(mem_type_to_mem_offset, zero_copy_mem_size), SUCCESS);
}
/*
 *  a  hcom1 hcom2 hcom3
 *  | |    | |   | |   |
 *  hcom4 hcom5  hcom6 b
 */
TEST_F(UtestMemoryAssignerTest, ContinuousOutIn_SUCCESS_ThreeNode) {
  auto graph = block_mem_ut::BuildContinuousOutInWithThreeNode();
  std::map<uint64_t, size_t> mem_type_to_mem_offset;
  size_t zero_copy_mem_size;
  VarManager::Instance(graph->GetSessionID())->Init(0, graph->GetSessionID(), 0, 0);
  MemoryAssigner mem_assigner(graph);
  EXPECT_EQ(mem_assigner.AssignMemory(mem_type_to_mem_offset, zero_copy_mem_size), SUCCESS);
}

TEST_F(UtestMemoryAssignerTest, AtomicCleanInput_Success_NonContinuousInput) {
  DEF_GRAPH(graph) {
    CHAIN(NODE("a", RELU)->NODE("atomic_node", "HcomNode")->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("b", RELU)->NODE("atomic_node", "HcomNode"));
    CTRL_CHAIN(NODE("memset", MEMSET)->Ctrl()->NODE("atomic_node", "HcomNode"));
  };
  auto root_graph = ToComputeGraph(graph);
  root_graph->TopologicalSorting();
  UpdateGraphTensorSize(root_graph);
  auto atomic_node = root_graph->FindFirstNodeMatchType("HcomNode");

  std::vector<int32_t> mem_type_list = {RT_MEMORY_P2P_DDR, RT_MEMORY_P2P_DDR};
  ge::AttrUtils::SetListInt(atomic_node->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, mem_type_list);

  AttrUtils::SetBool(atomic_node->GetOpDesc(), ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node->GetOpDesc(), ATOMIC_ATTR_INPUT_INDEX, {-1}), true);

  std::map<uint64_t, size_t> mem_type_to_mem_offset;
  size_t zero_copy_mem_size;
  MemoryAssigner mem_assigner(root_graph);
  EXPECT_EQ(mem_assigner.AssignMemory(mem_type_to_mem_offset, zero_copy_mem_size), SUCCESS);
}

TEST_F(UtestMemoryAssignerTest, AtomicCleanMerge_Sucess_ConsiderSpltMemory) {
  std::string memory_optimization_policy = "MemoryPriority";
  const auto old_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
  auto new_options = old_options;
  new_options.insert(std::make_pair(MEMORY_OPTIMIZATION_POLICY, memory_optimization_policy));
  ge::GetThreadLocalContext().SetGlobalOption(new_options);
  auto hcom1 = OP_CFG(HCOMALLREDUCE).InCnt(3).OutCnt(2).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 256, 1024, 32})
                   .Attr(ATTR_NAME_CONTINUOUS_INPUT, true);
  auto cast3 = OP_CFG(CAST).InCnt(3).OutCnt(2).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 256, 1024, 32});
  auto hcom3 = OP_CFG(HCOMALLREDUCE).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 256, 1024, 32});

  std::vector<uint32_t> value{0};
  auto cast0 = OP_CFG(CAST).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 256, 1024, 32})
                   .Attr(ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  auto cast1 = OP_CFG(CAST).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 256, 1024, 32})
                   .Attr(ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  auto data1 = OP_CFG(RELU).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 256, 1024, 32});
  auto data6 = OP_CFG(RELU).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 256, 1024, 32});
  auto data7 = OP_CFG(RELU).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 256, 1024, 32});

  auto data5 = OP_CFG(RELU).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 256, 1024, 32});

  auto atomic_memset = OP_CFG(MEMSET);
  auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(0).TensorDesc(FORMAT_ND, DT_INT32, {-1});
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", data1)->EDGE(0, 0)->NODE("hcom_1", hcom1));
    CHAIN(NODE("data6", data6)->EDGE(0, 1)->NODE("hcom_1", hcom1));
    CHAIN(NODE("data7", data7)->EDGE(0, 2)->NODE("hcom_1", hcom1));

    CHAIN(NODE("hcom_1", hcom1)->EDGE(1, 0)->NODE("hcom_3", hcom3)->NODE("netoutput", net_output));
    CHAIN(NODE("cast0", cast0)->EDGE(0, 0)->NODE("cast3", cast3)->NODE("netoutput", net_output));
    CHAIN(NODE("cast1", cast1)->EDGE(0, 1)->NODE("cast3", cast3)->NODE("netoutput", net_output));
    CHAIN(NODE("data_5", data5)->EDGE(0, 2)->NODE("cast3", cast3));
    CHAIN(NODE("atomic_memset", atomic_memset)->Ctrl()->NODE("data_1", data1));
    CHAIN(NODE("atomic_memset", atomic_memset)->Ctrl()->NODE("data_5", data5));
    CHAIN(NODE("atomic_memset", atomic_memset)->Ctrl()->NODE("cast_0", cast0));
    CHAIN(NODE("atomic_memset", atomic_memset)->Ctrl()->NODE("cast_1", cast1));
    CHAIN(NODE("atomic_memset", atomic_memset)->Ctrl()->NODE("hcom_1", hcom1));
  };

  auto graph = ToComputeGraph(g1);
  auto cast0_node = graph->FindNode("cast_0");
  auto cast1_node = graph->FindNode("cast_1");
  auto hcom1_node = graph->FindNode("hcom_1");
  AttrUtils::SetBool(cast0_node->GetOpDesc(), ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  AttrUtils::SetBool(cast1_node->GetOpDesc(), ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  EXPECT_EQ(AttrUtils::SetListInt(cast0_node->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, value), true);
  EXPECT_EQ(AttrUtils::SetListInt(cast1_node->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, value), true);
  EXPECT_EQ(AttrUtils::SetListInt(hcom1_node->GetOpDesc(), ATOMIC_ATTR_INPUT_INDEX, {-1}), true);

  UpdateGraphTensorSize(graph);
  graph->TopologicalSorting();
  bool dynamic_shape_partition = true;
  (void) ge::AttrUtils::SetBool(graph, ge::ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, dynamic_shape_partition);

  std::map<std::string, std::string> option;
  Graph2SubGraphInfoList subgraphs;
  std::map<std::string, int> stream_max_parallel_num;
  ge::ModelBuilder builder(0, graph, subgraphs, stream_max_parallel_num, false);
  MemoryAssigner mem_assigner(graph);

  ASSERT_EQ(mem_assigner.AssignMemory(builder.mem_type_to_mem_offset_, builder.zero_copy_mem_size_), SUCCESS);
  builder.sub_mem_offsets_ = mem_assigner.GetSubMemOffsets();
  ge::Model model;
  ASSERT_EQ(builder.BuildModelDef(model), SUCCESS);
  std::vector<std::vector<int64_t>> sub_mem_offsets;
  ge::AttrUtils::GetListListInt(&model, ATTR_MODEL_SUB_MEMORY_INFO, sub_mem_offsets);
  EXPECT_FALSE(sub_mem_offsets.empty());
  ge::GetThreadLocalContext().SetGlobalOption(old_options);

  auto memset = graph->FindNode("atomic_memset");
  ASSERT_NE(memset, nullptr);
  auto workspaces = memset->GetOpDesc()->GetWorkspace();
  auto workspace_sizes = memset->GetOpDesc()->GetWorkspaceBytes();
  ASSERT_TRUE(workspace_sizes.size() == workspaces.size());
  for (size_t i = 0U; i < workspaces.size(); ++i) {
    auto offset = workspaces[i];
    auto size = workspace_sizes[i];
    for (auto &sub_mem_offset : sub_mem_offsets) {
      if (sub_mem_offset[1] <= offset && offset < sub_mem_offset[1] + sub_mem_offset[2]) {
        EXPECT_TRUE(offset + size <= sub_mem_offset[1] + sub_mem_offset[2]);
      }
    }
  }
}

/*
 *         a
 *         |
 * d   c   b
 *  \ / \ /
 *  pc1 pc2
 *   |  |
 *   e  f
 *   \  /
 *  netoutput
 *
 *   topo order: b < d < c
 */
TEST_F(UtestMemoryAssignerTest, OneNodeConnectTwoPhonyConcat) {
  auto graph = block_mem_ut::BuildOneNodeConnectTwoPhonyConcat();
  std::map<uint64_t, size_t> mem_type_to_mem_offset;
  size_t zero_copy_mem_size;
  MemoryAssigner mem_assigner(graph);
  EXPECT_EQ(mem_assigner.AssignMemory(mem_type_to_mem_offset, zero_copy_mem_size), SUCCESS);
}

TEST_F(UtestMemoryAssignerTest, ValidWorkSpaceNeg1) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", DATA)
              ->NODE("A", "some")
              ->NODE("B", "some")
              ->NODE("C", "some")
              ->NODE("Node_Output", NETOUTPUT));
  };
  ComputeGraphPtr graph = ToComputeGraph(g1);
  const auto call_node = graph->FindNode("B");
  EXPECT_NE(call_node, nullptr);
  const auto call_desc = call_node->GetOpDesc();
  EXPECT_NE(call_desc, nullptr);
  std::vector<int64_t> workspace_bytes;
  workspace_bytes.push_back(-1);
  call_desc->SetWorkspaceBytes(workspace_bytes);
  UpdateGraphTensorSize(graph);
  graph->TopologicalSorting();
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0U;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
}
