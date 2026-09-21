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
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "graph/ascendc_ir/core/ascendc_ir_impl.h"
#include "testcase/ascendc_ir_dump_test/stub_graph.h"
#include "utils/graph_utils.h"
#include "mmpa/mmpa_api.h"
#include "graph/ge_context.h"
#include <iostream>
using namespace ge;
class UtestAscirGraphUtils : public testing::Test {
 protected:
  void SetUp() {
    dlog_setlevel(0, 3, 0);
  }

  void TearDown() {}
};
REG_OP(Constant)
    .INPUT(x, TensorType::ALL())
        .OUTPUT(y, TensorType::ALL())
        .OP_END_FACTORY_REG(Constant);

REG_OP(Abs)
    .INPUT(x, TensorType::ALL())
        .OUTPUT(y, TensorType::ALL())
        .OP_END_FACTORY_REG(Abs);
namespace {
std::stringstream GetFilePathWhenDumpPathSet(const string &ascend_work_path) {
  std::stringstream dump_file_path;
  dump_file_path << ascend_work_path << "/pid_" << mmGetPid() << "_deviceid_" << GetContext().DeviceId() << "/";
  return dump_file_path;
}
struct AscNodeInfo {
  std::string name;
  std::string type;
  size_t input_num;
  size_t output_num;
  std::vector<int64_t> axis_ids;
};
NodePtr BuildNode(const ComputeGraphPtr &asc_graph,
                  const AscNodeInfo &node, bool make_asc_node_directly = true) {
  OpDescBuilder op_desc_builder(node.name, node.type);
  for (size_t input_index = 0U; input_index < node.input_num; ++input_index) {
    op_desc_builder.AddInput("x_" + std::to_string(input_index));
  }
  for (size_t output_index = 0U; output_index < node.output_num; ++output_index) {
    op_desc_builder.AddOutput("y_" + std::to_string(output_index));
  }
  const auto &op_desc = op_desc_builder.Build();
  auto node_attr_group = op_desc->GetOrCreateAttrsGroup<AscNodeAttr>();

  node_attr_group->sched.exec_condition = ExecuteCondition::kNoCache;
  node_attr_group->sched.axis = node.axis_ids;
  if (!node_attr_group->sched.axis.empty()) {
    node_attr_group->sched.loop_axis = node_attr_group->sched.axis.back();
  }
  if (make_asc_node_directly) {
    const auto &asc_node = ComGraphMakeShared<AscNode>(op_desc, asc_graph);
    asc_node->Init();
    return asc_graph->AddNode(asc_node);
  }
  return asc_graph->AddNode(op_desc);
}
std::string GetSpecificFilePath(const std::string &file_path, const string &suffix) {
  DIR *dir;
  struct dirent *ent;
  dir = opendir(file_path.c_str());
  if (dir == nullptr) {
    return "";
  }
  while ((ent = readdir(dir)) != nullptr) {
    if (strstr(ent->d_name, suffix.c_str()) != nullptr) {
      std::string d_name(ent->d_name);
      closedir(dir);
      return file_path + "/" + d_name;
    }
  }
  closedir(dir);
  return "";
}
}
TEST_F(UtestAscirGraphUtils, SampleGraphSerializeDeserializeReadableSuccess) {
  AscGraph g("graph");
  auto op = ascir_op::Constant("abc");
  auto node = g.AddNode(op);
  node->inputs();
  node->outputs();
  g.SetTilingKey(0x5a5a);
  std::string output;
  EXPECT_EQ(AscGraphUtils::SerializeToReadable(g, output), GRAPH_SUCCESS);
  AscGraph out_asc_graph("");
  EXPECT_EQ(AscGraphUtils::DeserializeFromReadable(output, out_asc_graph), GRAPH_SUCCESS);
  EXPECT_EQ(out_asc_graph.GetName(), "graph");
  EXPECT_EQ(out_asc_graph.GetTilingKey(), 0x5a5a);
}

TEST_F(UtestAscirGraphUtils, SampleGraphSerializeDeserializeBinarySuccess) {
  AscGraph g("graph");
  auto op = ascir_op::Constant("abc");
  auto node = g.AddNode(op);
  node->inputs();
  node->outputs();
  g.SetTilingKey(0x5a5a);
  std::string output;
  EXPECT_EQ(AscGraphUtils::SerializeToBinary(g, output), GRAPH_SUCCESS);
  AscGraph out_asc_graph("");
  EXPECT_EQ(AscGraphUtils::DeserializeFromBinary(output, out_asc_graph), GRAPH_SUCCESS);
  EXPECT_EQ(out_asc_graph.GetName(), "graph");
  EXPECT_EQ(out_asc_graph.GetTilingKey(), 0x5a5a);
}

// constant->cast1
// expr->cast2
TEST_F(UtestAscirGraphUtils, ConstantCastIndexExprGraphSerializeDeserializeTestSuccess) {
  AscGraph g("graph");
  ascir_op::Constant constant("constant", g);
  constant.ir_attr.SetValue(0);
  ascir_op::IndexExpr expr("expr", g);
  expr.ir_attr.SetExpr(0x5a);
  ascir_op::Cast cast1("cast1");
  cast1.x = constant.y;
  ascir_op::Cast cast2("cast2");
  cast2.x = expr.y;
  g.SetTilingKey(0x5a5a);
  std::string output;
  EXPECT_EQ(AscGraphUtils::SerializeToReadable(g, output), GRAPH_SUCCESS);
  AscGraph out_asc_graph("");
  EXPECT_EQ(AscGraphUtils::DeserializeFromReadable(output, out_asc_graph), GRAPH_SUCCESS);
  EXPECT_EQ(out_asc_graph.GetName(), "graph");
  EXPECT_EQ(out_asc_graph.GetTilingKey(), 0x5a5a);
}

TEST_F(UtestAscirGraphUtils, FaGraphSerializeDeserializeBinarySuccess) {
  std::string graph_name("test_graph");
  AscGraph graph(graph_name.c_str());
  FaBeforeAutoFuse(graph);
  FaAfterScheduler(graph);
  FaAfterQueBufAlloc(graph);
  std::string output;
  ASSERT_EQ(AscGraphUtils::SerializeToBinary(graph, output), GRAPH_SUCCESS);
  AscGraph out_asc_graph("");
  ASSERT_EQ(AscGraphUtils::DeserializeFromBinary(output, out_asc_graph), GRAPH_SUCCESS);
  AscGraphAttr *out_asc_graph_attr = out_asc_graph.impl_->GetOrCreateGraphAttrsGroup();
  ASSERT_NE(out_asc_graph_attr, nullptr);
  AscGraphAttr *graph_attr = graph.impl_->GetOrCreateGraphAttrsGroup();
  ASSERT_NE(out_asc_graph_attr, nullptr);
  ASSERT_NE(graph_attr, nullptr);
  EXPECT_EQ(out_asc_graph_attr->tiling_key, graph_attr->tiling_key);
  ASSERT_EQ(out_asc_graph_attr->axis.size(), graph_attr->axis.size());
  for (size_t id = 0UL; id < out_asc_graph_attr->axis.size(); id++) {
    EXPECT_EQ(out_asc_graph_attr->axis[id]->name, graph_attr->axis[id]->name);
    EXPECT_EQ(out_asc_graph_attr->axis[id]->type, graph_attr->axis[id]->type);
    EXPECT_EQ(out_asc_graph_attr->axis[id]->allow_unaligned_tail, graph_attr->axis[id]->allow_unaligned_tail);
    EXPECT_EQ(out_asc_graph_attr->axis[id]->allow_oversize_axis, graph_attr->axis[id]->allow_oversize_axis);
    EXPECT_EQ(out_asc_graph_attr->axis[id]->bind_block, graph_attr->axis[id]->bind_block);
    EXPECT_EQ(out_asc_graph_attr->axis[id]->align, graph_attr->axis[id]->align);
    EXPECT_EQ(out_asc_graph_attr->axis[id]->split_pair_other_id, graph_attr->axis[id]->split_pair_other_id);
    EXPECT_EQ(out_asc_graph_attr->axis[id]->from, graph_attr->axis[id]->from);
    EXPECT_EQ(string(out_asc_graph_attr->axis[id]->size.Str().get()), string(graph_attr->axis[id]->size.Str().get()));
  }
  std::vector<AscNodePtr> asc_nodes;
  for (const auto &n : graph.GetAllNodes()) {
    asc_nodes.emplace_back(n);
  }
  EXPECT_EQ(graph_name, out_asc_graph.GetName());
  std::vector<AscNodePtr> out_asc_nodes;
  const auto out_graph = out_asc_graph.impl_->compute_graph_;
  for (const auto &n : out_asc_graph.GetAllNodes()) {
    out_asc_nodes.emplace_back(n);
  }
  ASSERT_EQ(asc_nodes.size(), out_asc_nodes.size());
  for (size_t i = 0UL; i < asc_nodes.size(); i++) {
    // check ascend node
    auto &asc_node = asc_nodes[i];
    auto &out_asc_node = out_asc_nodes[i];
    ASSERT_EQ(asc_node->GetName(), out_asc_node->GetName()) << " node name = " << asc_node->GetName();
    if (asc_node->GetType() == "Data") {
      int64_t index_src{0};
      int64_t index_dst{-1};
      auto src_ir_attr = asc_node->attr.ir_attr.get();
      auto dst_ir_attr = out_asc_node->attr.ir_attr.get();
      if (src_ir_attr->GetAttrValue("index", index_src) == GRAPH_SUCCESS) {
        EXPECT_EQ(dst_ir_attr->GetAttrValue("index", index_dst), GRAPH_SUCCESS)
                  << " node name = " << asc_node->GetName();
        EXPECT_EQ(index_src, index_dst);
      }
    }
    EXPECT_EQ(asc_node->attr.type, out_asc_node->attr.type) << " node name = " << asc_node->GetName();
    EXPECT_EQ(asc_node->attr.name, out_asc_node->attr.name) << " node name = " << asc_node->GetName();
    EXPECT_EQ(asc_node->attr.api.type, out_asc_node->attr.api.type) << " node name = " << asc_node->GetName();
    EXPECT_EQ(asc_node->attr.api.unit, out_asc_node->attr.api.unit) << " node name = " << asc_node->GetName();
    EXPECT_EQ(asc_node->attr.api.compute_type, out_asc_node->attr.api.compute_type)
        << " node name = " << asc_node->GetName();
    EXPECT_EQ(asc_node->attr.sched.exec_order, out_asc_node->attr.sched.exec_order)
        << " node name = " << asc_node->GetName();
    EXPECT_EQ(asc_node->attr.sched.loop_axis, out_asc_node->attr.sched.loop_axis)
        << " node name = " << asc_node->GetName();
    EXPECT_EQ(asc_node->attr.sched.exec_condition, ExecuteCondition::kNoCache)
        << " node name = " << asc_node->GetName();
    EXPECT_EQ(asc_node->attr.sched.axis, out_asc_node->attr.sched.axis) << " node name = " << asc_node->GetName();
    // check input tensor
    auto inputs = asc_node->inputs.tensors_;
    auto new_inputs = out_asc_node->inputs.tensors_;
    for (size_t id = 0UL; id < inputs.size(); id++) {
      // check anchor ref
      ASSERT_NE(inputs[id].anchor.GetOwnerNode(), nullptr);
      ASSERT_NE(new_inputs[id].anchor.GetOwnerNode(), nullptr);
      EXPECT_EQ(new_inputs[id].anchor.GetOwnerNode()->GetName(), inputs[id].anchor.GetOwnerNode()->GetName());
      EXPECT_EQ(new_inputs[id].anchor.GetIdx(), inputs[id].anchor.GetIdx());
      // check attr
      const auto owner_node = ge::ascir::AscTensorUtils::GetOwner(inputs[id]);
      EXPECT_EQ(inputs[id].attr.axis, new_inputs[id].attr.axis);
      EXPECT_EQ(inputs[id].attr.dtype, new_inputs[id].attr.dtype)
          << "node name=" << asc_node->GetNamePtr() << ", in id=" << id
          << ", from=" << ((owner_node != nullptr) ? "null" : owner_node->GetName());
      // mem attr
      EXPECT_EQ(inputs[id].attr.mem.tensor_id, new_inputs[id].attr.mem.tensor_id);
      EXPECT_EQ(inputs[id].attr.mem.name, new_inputs[id].attr.mem.name);
      EXPECT_EQ(inputs[id].attr.mem.buf_ids, new_inputs[id].attr.mem.buf_ids);
      EXPECT_EQ(inputs[id].attr.mem.position, new_inputs[id].attr.mem.position);
      EXPECT_EQ(inputs[id].attr.mem.hardware, new_inputs[id].attr.mem.hardware);
      EXPECT_EQ(inputs[id].attr.mem.alloc_type, new_inputs[id].attr.mem.alloc_type);
      // que/buf attr
      EXPECT_EQ(inputs[id].attr.que.id, new_inputs[id].attr.que.id);
      EXPECT_EQ(inputs[id].attr.que.buf_num, new_inputs[id].attr.que.buf_num);
      EXPECT_EQ(inputs[id].attr.que.depth, new_inputs[id].attr.que.depth);
      EXPECT_EQ(inputs[id].attr.buf.id, new_inputs[id].attr.buf.id);
      // opt attr
      EXPECT_EQ(inputs[id].attr.opt.reuse_id, new_inputs[id].attr.opt.reuse_id);
      EXPECT_EQ(inputs[id].attr.opt.ref_tensor, new_inputs[id].attr.opt.ref_tensor);
      EXPECT_EQ(inputs[id].attr.opt.merge_scope, new_inputs[id].attr.opt.merge_scope);
      ASSERT_EQ(inputs[id].attr.strides.size(), new_inputs[id].attr.strides.size());
      for (size_t stride_id = 0UL; stride_id < inputs[id].attr.strides.size(); stride_id++) {
        EXPECT_EQ(std::string(inputs[id].attr.strides[stride_id].Str().get()),
                  std::string(new_inputs[id].attr.strides[stride_id].Str().get()));
      }
      ASSERT_EQ(inputs[id].attr.repeats.size(), new_inputs[id].attr.repeats.size());
      for (size_t repeat_id = 0UL; repeat_id < inputs[id].attr.repeats.size(); repeat_id++) {
        EXPECT_EQ(std::string(inputs[id].attr.repeats[repeat_id].Str().get()),
                  std::string(new_inputs[id].attr.repeats[repeat_id].Str().get()));
      }
      EXPECT_EQ(inputs[id].attr.vectorized_axis, new_inputs[id].attr.vectorized_axis);
      ASSERT_EQ(inputs[id].attr.vectorized_strides.size(), new_inputs[id].attr.vectorized_strides.size());
      for (size_t vectorized_stride_id = 0UL; vectorized_stride_id < inputs[id].attr.vectorized_strides.size();
           vectorized_stride_id++) {
        EXPECT_EQ(std::string(inputs[id].attr.vectorized_strides[vectorized_stride_id].Str().get()),
                  std::string(new_inputs[id].attr.vectorized_strides[vectorized_stride_id].Str().get()))
            << " node name=" << asc_node->GetName() << " input index=" << id;
      }
    }

    auto outputs = asc_node->outputs.tensors_;
    auto new_outputs = out_asc_node->outputs.tensors_;
    for (size_t id = 0UL; id < outputs.size(); id++) {
      // check anchor ref
      ASSERT_NE(outputs[id].anchor.GetOwnerNode(), nullptr);
      ASSERT_NE(new_outputs[id].anchor.GetOwnerNode(), nullptr);
      EXPECT_EQ(new_outputs[id].anchor.GetOwnerNode()->GetName(), outputs[id].anchor.GetOwnerNode()->GetName());
      EXPECT_EQ(new_outputs[id].anchor.GetIdx(), outputs[id].anchor.GetIdx());
      // check attr
      EXPECT_EQ(outputs[id].attr.axis, new_outputs[id].attr.axis);
      EXPECT_EQ(outputs[id].attr.dtype, new_outputs[id].attr.dtype)
          << "node name=" << asc_node->GetNamePtr() << ",out id=" << id;
      // mem attr
      EXPECT_EQ(outputs[id].attr.mem.tensor_id, new_outputs[id].attr.mem.tensor_id);
      EXPECT_EQ(outputs[id].attr.mem.name, new_outputs[id].attr.mem.name);
      EXPECT_EQ(outputs[id].attr.mem.buf_ids, new_outputs[id].attr.mem.buf_ids);
      EXPECT_EQ(outputs[id].attr.mem.position, new_outputs[id].attr.mem.position);
      EXPECT_EQ(outputs[id].attr.mem.hardware, new_outputs[id].attr.mem.hardware);
      EXPECT_EQ(outputs[id].attr.mem.alloc_type, new_outputs[id].attr.mem.alloc_type);
      // que/buf attr
      EXPECT_EQ(outputs[id].attr.que.id, new_outputs[id].attr.que.id);
      EXPECT_EQ(outputs[id].attr.que.buf_num, new_outputs[id].attr.que.buf_num);
      EXPECT_EQ(outputs[id].attr.que.depth, new_outputs[id].attr.que.depth);
      EXPECT_EQ(outputs[id].attr.buf.id, new_outputs[id].attr.buf.id);
      // opt attr
      EXPECT_EQ(outputs[id].attr.opt.reuse_id, new_outputs[id].attr.opt.reuse_id);
      EXPECT_EQ(outputs[id].attr.opt.ref_tensor, new_outputs[id].attr.opt.ref_tensor);
      EXPECT_EQ(outputs[id].attr.opt.merge_scope, new_outputs[id].attr.opt.merge_scope);
      ASSERT_EQ(outputs[id].attr.strides.size(), new_outputs[id].attr.strides.size());
      for (size_t stride_id = 0UL; stride_id < outputs[id].attr.strides.size(); stride_id++) {
        EXPECT_EQ(std::string(outputs[id].attr.strides[stride_id].Str().get()),
                  std::string(new_outputs[id].attr.strides[stride_id].Str().get()));
      }
      ASSERT_EQ(outputs[id].attr.repeats.size(), new_outputs[id].attr.repeats.size());
      for (size_t repeat_id = 0UL; repeat_id < outputs[id].attr.repeats.size(); repeat_id++) {
        EXPECT_EQ(std::string(outputs[id].attr.repeats[repeat_id].Str().get()),
                  std::string(new_outputs[id].attr.repeats[repeat_id].Str().get()));
      }
      EXPECT_EQ(outputs[id].attr.vectorized_axis, new_outputs[id].attr.vectorized_axis);
      ASSERT_EQ(outputs[id].attr.vectorized_strides.size(), new_outputs[id].attr.vectorized_strides.size());
      for (size_t vectorized_stride_id = 0UL; vectorized_stride_id < outputs[id].attr.vectorized_strides.size();
           vectorized_stride_id++) {
        EXPECT_EQ(std::string(outputs[id].attr.vectorized_strides[vectorized_stride_id].Str().get()),
                  std::string(new_outputs[id].attr.vectorized_strides[vectorized_stride_id].Str().get()))
            << " node name=" << asc_node->GetName() << ", output index=" << id;
      }
    }
  }
}

TEST_F(UtestAscirGraphUtils, FaGraphSerializeDeserializeReadableSuccess) {
  std::string graph_name("test_graph");
  AscGraph graph(graph_name.c_str());
  FaBeforeAutoFuse(graph);
  FaAfterScheduler(graph);
  FaAfterQueBufAlloc(graph);
  std::string output;
  ASSERT_EQ(AscGraphUtils::SerializeToReadable(graph, output), GRAPH_SUCCESS);
  AscGraph out_asc_graph("");
  ASSERT_EQ(AscGraphUtils::DeserializeFromReadable(output, out_asc_graph), GRAPH_SUCCESS);
  AscGraphAttr *out_asc_graph_attr = out_asc_graph.impl_->GetOrCreateGraphAttrsGroup();
  ASSERT_NE(out_asc_graph_attr, nullptr);
  AscGraphAttr *graph_attr = graph.impl_->GetOrCreateGraphAttrsGroup();
  ASSERT_NE(out_asc_graph_attr, nullptr);
  ASSERT_NE(graph_attr, nullptr);
  EXPECT_EQ(out_asc_graph_attr->tiling_key, graph_attr->tiling_key);
  ASSERT_EQ(out_asc_graph_attr->axis.size(), graph_attr->axis.size());
  for (size_t id = 0UL; id < out_asc_graph_attr->axis.size(); id++) {
    EXPECT_EQ(out_asc_graph_attr->axis[id]->name, graph_attr->axis[id]->name);
    EXPECT_EQ(out_asc_graph_attr->axis[id]->type, graph_attr->axis[id]->type);
    EXPECT_EQ(out_asc_graph_attr->axis[id]->allow_unaligned_tail, graph_attr->axis[id]->allow_unaligned_tail);
    EXPECT_EQ(out_asc_graph_attr->axis[id]->allow_oversize_axis, graph_attr->axis[id]->allow_oversize_axis);
    EXPECT_EQ(out_asc_graph_attr->axis[id]->bind_block, graph_attr->axis[id]->bind_block);
    EXPECT_EQ(out_asc_graph_attr->axis[id]->align, graph_attr->axis[id]->align);
    EXPECT_EQ(out_asc_graph_attr->axis[id]->split_pair_other_id, graph_attr->axis[id]->split_pair_other_id);
    EXPECT_EQ(out_asc_graph_attr->axis[id]->from, graph_attr->axis[id]->from);
    EXPECT_EQ(string(out_asc_graph_attr->axis[id]->size.Str().get()), string(graph_attr->axis[id]->size.Str().get()));
  }
  std::vector<AscNodePtr> asc_nodes;
  for (const auto &n : graph.GetAllNodes()) {
    asc_nodes.emplace_back(n);
  }
  EXPECT_EQ(graph_name, out_asc_graph.GetName());
  std::vector<AscNodePtr> out_asc_nodes;
  const auto out_graph = out_asc_graph.impl_->compute_graph_;
  for (const auto &n : out_asc_graph.GetAllNodes()) {
    out_asc_nodes.emplace_back(n);
  }
  ASSERT_EQ(asc_nodes.size(), out_asc_nodes.size());
  for (size_t i = 0UL; i < asc_nodes.size(); i++) {
    // check ascend node
    auto &asc_node = asc_nodes[i];
    auto &out_asc_node = out_asc_nodes[i];
    ASSERT_EQ(asc_node->GetName(), out_asc_node->GetName()) << " node name = " << asc_node->GetName();
    if (asc_node->GetType() == "Data") {
      int64_t index_src{0};
      int64_t index_dst{-1};
      auto src_ir_attr = asc_node->attr.ir_attr.get();
      auto dst_ir_attr = out_asc_node->attr.ir_attr.get();
      if (src_ir_attr->GetAttrValue("index", index_src) == GRAPH_SUCCESS) {
        EXPECT_EQ(dst_ir_attr->GetAttrValue("index", index_dst), GRAPH_SUCCESS)
                  << " node name = " << asc_node->GetName();
        EXPECT_EQ(index_src, index_dst);
      }
    }
    EXPECT_EQ(asc_node->attr.type, out_asc_node->attr.type) << " node name = " << asc_node->GetName();
    EXPECT_EQ(asc_node->attr.name, out_asc_node->attr.name) << " node name = " << asc_node->GetName();
    EXPECT_EQ(asc_node->attr.api.type, out_asc_node->attr.api.type) << " node name = " << asc_node->GetName();
    EXPECT_EQ(asc_node->attr.api.unit, out_asc_node->attr.api.unit) << " node name = " << asc_node->GetName();
    EXPECT_EQ(asc_node->attr.api.compute_type, out_asc_node->attr.api.compute_type)
        << " node name = " << asc_node->GetName();
    EXPECT_EQ(asc_node->attr.sched.exec_order, out_asc_node->attr.sched.exec_order)
        << " node name = " << asc_node->GetName();
    EXPECT_EQ(asc_node->attr.sched.loop_axis, out_asc_node->attr.sched.loop_axis)
        << " node name = " << asc_node->GetName();
    EXPECT_EQ(asc_node->attr.sched.exec_condition, ExecuteCondition::kNoCache)
        << " node name = " << asc_node->GetName();
    EXPECT_EQ(asc_node->attr.sched.axis, out_asc_node->attr.sched.axis) << " node name = " << asc_node->GetName();
    // check input tensor
    auto inputs = asc_node->inputs.tensors_;
    auto new_inputs = out_asc_node->inputs.tensors_;
    for (size_t id = 0UL; id < inputs.size(); id++) {
      // check anchor ref
      ASSERT_NE(inputs[id].anchor.GetOwnerNode(), nullptr);
      ASSERT_NE(new_inputs[id].anchor.GetOwnerNode(), nullptr);
      EXPECT_EQ(new_inputs[id].anchor.GetOwnerNode()->GetName(), inputs[id].anchor.GetOwnerNode()->GetName());
      EXPECT_EQ(new_inputs[id].anchor.GetIdx(), inputs[id].anchor.GetIdx());
      // check attr
      const auto owner_node = ge::ascir::AscTensorUtils::GetOwner(inputs[id]);
      EXPECT_EQ(inputs[id].attr.axis, new_inputs[id].attr.axis);
      EXPECT_EQ(inputs[id].attr.dtype, new_inputs[id].attr.dtype)
          << "node name=" << asc_node->GetNamePtr() << ", in id=" << id
          << ", from=" << ((owner_node != nullptr) ? "null" : owner_node->GetName());
      // mem attr
      EXPECT_EQ(inputs[id].attr.mem.tensor_id, new_inputs[id].attr.mem.tensor_id);
      EXPECT_EQ(inputs[id].attr.mem.name, new_inputs[id].attr.mem.name);
      EXPECT_EQ(inputs[id].attr.mem.buf_ids, new_inputs[id].attr.mem.buf_ids);
      EXPECT_EQ(inputs[id].attr.mem.position, new_inputs[id].attr.mem.position);
      EXPECT_EQ(inputs[id].attr.mem.hardware, new_inputs[id].attr.mem.hardware);
      EXPECT_EQ(inputs[id].attr.mem.alloc_type, new_inputs[id].attr.mem.alloc_type);
      // que/buf attr
      EXPECT_EQ(inputs[id].attr.que.id, new_inputs[id].attr.que.id);
      EXPECT_EQ(inputs[id].attr.que.buf_num, new_inputs[id].attr.que.buf_num);
      EXPECT_EQ(inputs[id].attr.que.depth, new_inputs[id].attr.que.depth);
      EXPECT_EQ(inputs[id].attr.buf.id, new_inputs[id].attr.buf.id);
      // opt attr
      EXPECT_EQ(inputs[id].attr.opt.reuse_id, new_inputs[id].attr.opt.reuse_id);
      EXPECT_EQ(inputs[id].attr.opt.ref_tensor, new_inputs[id].attr.opt.ref_tensor);
      EXPECT_EQ(inputs[id].attr.opt.merge_scope, new_inputs[id].attr.opt.merge_scope);
      ASSERT_EQ(inputs[id].attr.strides.size(), new_inputs[id].attr.strides.size());
      for (size_t stride_id = 0UL; stride_id < inputs[id].attr.strides.size(); stride_id++) {
        EXPECT_EQ(std::string(inputs[id].attr.strides[stride_id].Str().get()),
                  std::string(new_inputs[id].attr.strides[stride_id].Str().get()));
      }
      ASSERT_EQ(inputs[id].attr.repeats.size(), new_inputs[id].attr.repeats.size());
      for (size_t repeat_id = 0UL; repeat_id < inputs[id].attr.repeats.size(); repeat_id++) {
        EXPECT_EQ(std::string(inputs[id].attr.repeats[repeat_id].Str().get()),
                  std::string(new_inputs[id].attr.repeats[repeat_id].Str().get()));
      }
      EXPECT_EQ(inputs[id].attr.vectorized_axis, new_inputs[id].attr.vectorized_axis);
      ASSERT_EQ(inputs[id].attr.vectorized_strides.size(), new_inputs[id].attr.vectorized_strides.size());
      for (size_t vectorized_stride_id = 0UL; vectorized_stride_id < inputs[id].attr.vectorized_strides.size();
           vectorized_stride_id++) {
        EXPECT_EQ(std::string(inputs[id].attr.vectorized_strides[vectorized_stride_id].Str().get()),
                  std::string(new_inputs[id].attr.vectorized_strides[vectorized_stride_id].Str().get()))
            << " node name=" << asc_node->GetName() << " input index=" << id;
      }
    }

    auto outputs = asc_node->outputs.tensors_;
    auto new_outputs = out_asc_node->outputs.tensors_;
    for (size_t id = 0UL; id < outputs.size(); id++) {
      // check anchor ref
      ASSERT_NE(outputs[id].anchor.GetOwnerNode(), nullptr);
      ASSERT_NE(new_outputs[id].anchor.GetOwnerNode(), nullptr);
      EXPECT_EQ(new_outputs[id].anchor.GetOwnerNode()->GetName(), outputs[id].anchor.GetOwnerNode()->GetName());
      EXPECT_EQ(new_outputs[id].anchor.GetIdx(), outputs[id].anchor.GetIdx());
      // check attr
      EXPECT_EQ(outputs[id].attr.axis, new_outputs[id].attr.axis);
      EXPECT_EQ(outputs[id].attr.dtype, new_outputs[id].attr.dtype)
          << "node name=" << asc_node->GetNamePtr() << ",out id=" << id;
      // mem attr
      EXPECT_EQ(outputs[id].attr.mem.tensor_id, new_outputs[id].attr.mem.tensor_id);
      EXPECT_EQ(outputs[id].attr.mem.name, new_outputs[id].attr.mem.name);
      EXPECT_EQ(outputs[id].attr.mem.buf_ids, new_outputs[id].attr.mem.buf_ids);
      EXPECT_EQ(outputs[id].attr.mem.position, new_outputs[id].attr.mem.position);
      EXPECT_EQ(outputs[id].attr.mem.hardware, new_outputs[id].attr.mem.hardware);
      EXPECT_EQ(outputs[id].attr.mem.alloc_type, new_outputs[id].attr.mem.alloc_type);
      // que/buf attr
      EXPECT_EQ(outputs[id].attr.que.id, new_outputs[id].attr.que.id);
      EXPECT_EQ(outputs[id].attr.que.buf_num, new_outputs[id].attr.que.buf_num);
      EXPECT_EQ(outputs[id].attr.que.depth, new_outputs[id].attr.que.depth);
      EXPECT_EQ(outputs[id].attr.buf.id, new_outputs[id].attr.buf.id);
      // opt attr
      EXPECT_EQ(outputs[id].attr.opt.reuse_id, new_outputs[id].attr.opt.reuse_id);
      EXPECT_EQ(outputs[id].attr.opt.ref_tensor, new_outputs[id].attr.opt.ref_tensor);
      EXPECT_EQ(outputs[id].attr.opt.merge_scope, new_outputs[id].attr.opt.merge_scope);
      ASSERT_EQ(outputs[id].attr.strides.size(), new_outputs[id].attr.strides.size());
      for (size_t stride_id = 0UL; stride_id < outputs[id].attr.strides.size(); stride_id++) {
        EXPECT_EQ(std::string(outputs[id].attr.strides[stride_id].Str().get()),
                  std::string(new_outputs[id].attr.strides[stride_id].Str().get()));
      }
      ASSERT_EQ(outputs[id].attr.repeats.size(), new_outputs[id].attr.repeats.size());
      for (size_t repeat_id = 0UL; repeat_id < outputs[id].attr.repeats.size(); repeat_id++) {
        EXPECT_EQ(std::string(outputs[id].attr.repeats[repeat_id].Str().get()),
                  std::string(new_outputs[id].attr.repeats[repeat_id].Str().get()));
      }
      EXPECT_EQ(outputs[id].attr.vectorized_axis, new_outputs[id].attr.vectorized_axis);
      ASSERT_EQ(outputs[id].attr.vectorized_strides.size(), new_outputs[id].attr.vectorized_strides.size());
      for (size_t vectorized_stride_id = 0UL; vectorized_stride_id < outputs[id].attr.vectorized_strides.size();
           vectorized_stride_id++) {
        EXPECT_EQ(std::string(outputs[id].attr.vectorized_strides[vectorized_stride_id].Str().get()),
                  std::string(new_outputs[id].attr.vectorized_strides[vectorized_stride_id].Str().get()))
            << " node name=" << asc_node->GetName() << ", output index=" << id;
      }
    }
  }
}

TEST_F(UtestAscirGraphUtils, ConvertComputeGraphToAscGraph_Success) {
  auto compute_graph = ComGraphMakeShared<ComputeGraph>("test");
  const auto graph_attr_group_ptr = compute_graph->GetOrCreateAttrsGroup<AscGraphAttr>();
  EXPECT_TRUE(graph_attr_group_ptr != nullptr);
  std::vector<int64_t> axes{1, 2, 3};
  std::vector<AxisPtr> axis_ptrs;
  auto axis_ptr = ComGraphMakeShared<Axis>();
  axis_ptr->name = "axis1";
  axis_ptr->size = sym::kSymbolOne;
  axis_ptr->id = 0;
  axis_ptrs.push_back(axis_ptr);
  graph_attr_group_ptr->axis = axis_ptrs;
  auto data = BuildNode(compute_graph, {"data0", "Data", 0, 1, axes}, false);
  auto load = BuildNode(compute_graph, {"load0", "Load", 1, 1, axes}, false);
  GraphUtils::AddEdge(data->GetOutDataAnchor(0), load->GetInDataAnchor(0));
  EXPECT_TRUE(dynamic_cast<AscNode *>(data.get()) == nullptr);
  AscGraph asc_graph("");
  EXPECT_EQ(AscGraphUtils::ConvertComputeGraphToAscGraph(compute_graph, asc_graph), GRAPH_SUCCESS);
  EXPECT_EQ(asc_graph.GetName(), "test");

  auto asc_data = asc_graph.FindNode("data0");
  EXPECT_TRUE(asc_data != nullptr);
  EXPECT_EQ(asc_data->attr.sched.axis, axes);
  EXPECT_EQ(asc_data->attr.sched.loop_axis, axes.back());
  EXPECT_EQ(asc_data->attr.sched.exec_condition, ExecuteCondition::kNoCache);
  EXPECT_EQ(asc_data->outputs().size(), 1U);
  auto &tensor_attr = asc_data->outputs[0U];
  tensor_attr.attr.axis = axes;
  EXPECT_EQ(asc_data->GetOpDesc()->GetOutputDescPtr(0U)->GetAttrsGroup<AscTensorAttr>()->axis, axes);
  auto asc_load = asc_graph.FindNode("load0");
  EXPECT_EQ(asc_load->inputs.Size(), 1U);
  EXPECT_EQ(asc_load->GetInDataNodesSize(), 1U);
  EXPECT_EQ(asc_load->GetInNodesPtr()[0U], asc_data.get());
  auto axis_to_find = asc_graph.FindAxis(0);
  EXPECT_TRUE(axis_to_find != nullptr);
  EXPECT_EQ(axis_to_find->name, "axis1");
  EXPECT_EQ(axis_to_find->id, 0);
  EXPECT_EQ(axis_to_find->size, Symbol(1));
}

TEST_F(UtestAscirGraphUtils, ConcatGraphSerializeDeserializeReadableSuccess) {
  std::string graph_name("concat_graph");
  AscGraph graph(graph_name.c_str());
  CreatConcatAscGraph(graph);
  std::string output;
  ASSERT_EQ(AscGraphUtils::SerializeToReadable(graph, output), GRAPH_SUCCESS);
  AscGraph out_asc_graph("");
  ASSERT_EQ(AscGraphUtils::DeserializeFromReadable(output, out_asc_graph), GRAPH_SUCCESS);
  AscGraphAttr *out_asc_graph_attr = out_asc_graph.impl_->GetOrCreateGraphAttrsGroup();
  ASSERT_NE(out_asc_graph_attr, nullptr);
  AscGraphAttr *graph_attr = graph.impl_->GetOrCreateGraphAttrsGroup();
  ASSERT_NE(out_asc_graph_attr, nullptr);
  ASSERT_NE(graph_attr, nullptr);
  EXPECT_EQ(out_asc_graph_attr->tiling_key, graph_attr->tiling_key);
  EXPECT_EQ(out_asc_graph_attr->tiling_key, graph_attr->tiling_key);
  ASSERT_EQ(out_asc_graph_attr->axis.size(), graph_attr->axis.size());
  for (size_t id = 0UL; id < out_asc_graph_attr->axis.size(); id++) {
    EXPECT_EQ(out_asc_graph_attr->axis[id]->name, graph_attr->axis[id]->name);
    EXPECT_EQ(out_asc_graph_attr->axis[id]->type, graph_attr->axis[id]->type);
    EXPECT_EQ(out_asc_graph_attr->axis[id]->allow_unaligned_tail, graph_attr->axis[id]->allow_unaligned_tail);
    EXPECT_EQ(out_asc_graph_attr->axis[id]->allow_oversize_axis, graph_attr->axis[id]->allow_oversize_axis);
    EXPECT_EQ(out_asc_graph_attr->axis[id]->bind_block, graph_attr->axis[id]->bind_block);
    EXPECT_EQ(out_asc_graph_attr->axis[id]->align, graph_attr->axis[id]->align);
    EXPECT_EQ(out_asc_graph_attr->axis[id]->split_pair_other_id, graph_attr->axis[id]->split_pair_other_id);
    EXPECT_EQ(out_asc_graph_attr->axis[id]->from, graph_attr->axis[id]->from);
    EXPECT_EQ(string(out_asc_graph_attr->axis[id]->size.Str().get()), string(graph_attr->axis[id]->size.Str().get()));
  }
}

TEST_F(UtestAscirGraphUtils, FaGraphDumpWithAttrGroupSuccess) {
  std::string graph_name("test_graph");
  AscGraph graph(graph_name.c_str());
  FaBeforeAutoFuse(graph);
  FaAfterScheduler(graph);
  FaAfterQueBufAlloc(graph);
  std::string ascend_work_path = "./test_ge_graph_path";
  setenv("DUMP_GRAPH_PATH", ascend_work_path.c_str(), 1);
  EXPECT_NO_THROW(GraphUtils::DumpGEGraph(AscGraphUtils::GetComputeGraph(graph), "attr_group_test", true););
  EXPECT_NO_THROW(GraphUtils::DumpGEGraphToOnnx(*AscGraphUtils::GetComputeGraph(graph), "attr_group_test", true));
  std::stringstream dump_file_path = GetFilePathWhenDumpPathSet(ascend_work_path);
  std::string dump_graph_path = ge::RealPath(dump_file_path.str().c_str());
  std::string
      dump_txt_graph_path = GetSpecificFilePath(ge::RealPath(dump_file_path.str().c_str()), "attr_group_test.txt");
  ComputeGraphPtr com_graph = std::make_shared<ComputeGraph>("load_test_graph");
  // 测试反序列化之后的图
  auto state = GraphUtils::LoadGEGraph(dump_txt_graph_path.c_str(), *com_graph);
  ASSERT_EQ(state, true);
  auto data = com_graph->FindNode("query");
  EXPECT_NE(data, nullptr);
  auto data_op = data->GetOpDesc();
  EXPECT_NE(data_op, nullptr);
  auto data_attr_group = data_op->GetAttrsGroup<AscNodeAttr>();
  EXPECT_NE(data_attr_group, nullptr);
  ascendc_ir::proto::AscNodeAttrGroupsDef asc_node_group;
  EXPECT_EQ(data_attr_group->SerializeAttr(asc_node_group), GRAPH_SUCCESS);
  EXPECT_EQ(asc_node_group.DebugString(), R"PROTO(name: "query"
type: "Data"
sched {
  axis: 10
  axis: 11
  axis: 12
  axis: 8
  axis: 13
  axis: 5
  axis: 6
  loop_axis: 12
}
api {
  type: 2
  compute_type: 11
  unit: 7
}
ir_attr_def {
  attr {
    key: "index"
    value {
      i: 0
    }
  }
}
)PROTO");
  auto data_desc_attr_group = data_op->GetOutputDescPtr(0U)->GetAttrsGroup<AscTensorAttr>();
  EXPECT_NE(data_desc_attr_group, nullptr);
  EXPECT_EQ(data_desc_attr_group->dtype, DT_FLOAT16);
  unsetenv("DUMP_GRAPH_PATH");
  system(("rm -rf " + ascend_work_path).c_str());
}
