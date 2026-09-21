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
#include "graph/op_desc.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_adapter.h"
#include "graph/detail/model_serialize_imp.h"
#include "proto/ge_ir.pb.h"
#include "es_graph_builder.h"
#include "compliant_node_builder.h"

using namespace ge;
using namespace ge::es;
using namespace testing;
namespace {
auto Normalize = [](const std::string &code) {
  std::string str = code;
  str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
  return str;
};

class OpDescChecker {
 public:
  explicit OpDescChecker(const GNode &node) {
    auto node_ptr = ge::NodeAdapter::GNode2Node(node);
    op_desc_ = node_ptr->GetOpDesc();
    ge::proto::OpDef def;
    ModelSerializeImp imp;
    imp.SerializeOpDesc(op_desc_, &def);
    string_desc_ = def.DebugString();
  }
  void PrintDebugString() {
    std::cout << string_desc_ << std::endl;
  }
  OpDescChecker CheckAll(const std::string &golden) {
    EXPECT_EQ(Normalize(string_desc_), Normalize(golden)) << "Actual code:\n"
                                                          << string_desc_ << "\nExpected:\n"
                                                          << golden;
    return *this;
  }
  OpDescChecker CheckSubgraph(const std::vector<std::string> &subgraphs) {
    EXPECT_TRUE(op_desc_->GetSubgraphInstanceNames() == subgraphs);
    return *this;
  }
  OpDescChecker CheckOutDataTypes(const std::vector<ge::DataType> &data_types) {
    EXPECT_TRUE(op_desc_->GetAllOutputsDesc().size() == data_types.size());
    for (size_t i = 0; i < data_types.size(); ++i) {
      EXPECT_EQ(op_desc_->GetOutputDescPtr(i)->GetDataType(), data_types[i]);
    }
    return *this;
  }
  OpDescChecker CheckOutFormats(const std::vector<ge::Format> &formats) {
    EXPECT_TRUE(op_desc_->GetAllOutputsDesc().size() == formats.size());
    for (size_t i = 0; i < formats.size(); ++i) {
      EXPECT_EQ(op_desc_->GetOutputDescPtr(i)->GetFormat(), formats[i]);
    }
    return *this;
  }
  OpDescChecker CheckInputIrInfo(const std::map<size_t, std::pair<size_t, size_t>> &ir_ranges_golden) {
    std::map<size_t, std::pair<size_t, size_t>> ir_ranges;  // Idx, Start, Num
    EXPECT_EQ(OpDescUtils::GetIrInputRawDescRange(op_desc_, ir_ranges), GRAPH_SUCCESS);
    EXPECT_EQ(ir_ranges, ir_ranges_golden);
    return *this;
  }

  OpDescChecker CheckOutputIrInfo(const std::map<size_t, std::pair<size_t, size_t>> &ir_ranges_golden) {
    std::map<size_t, std::pair<size_t, size_t>> ir_ranges;  // Idx, Start, Num
    EXPECT_EQ(OpDescUtils::GetIrOutputDescRange(op_desc_, ir_ranges), GRAPH_SUCCESS);
    EXPECT_EQ(ir_ranges, ir_ranges_golden);
    return *this;
  }
  OpDescChecker CheckId(const size_t id) {
    EXPECT_EQ(op_desc_->GetId(), id);
    return *this;
  }
  OpDescChecker CheckName(const std::string &name) {
    EXPECT_EQ(op_desc_->GetName(), name);
    return *this;
  }
  OpDescChecker CheckType(const std::string &type) {
    EXPECT_EQ(op_desc_->GetType(), type);
    return *this;
  }
  OpDescChecker CheckAttr(const std::vector<std::string> &attrs) {
    for (const auto &attr : attrs) {
      EXPECT_TRUE(op_desc_->HasAttr(attr));
    }
    return *this;
  }

 private:
  ge::OpDescPtr op_desc_;
  std::string string_desc_;
};
}  // namespace

class CompliantNodeBuilderLLT : public ::testing::Test {
 protected:
  void SetUp() override {
    // 创建测试用的Graph
    test_graph_ = test_graph_builder_.BuildAndReset();
    // 初始化测试数据
    input_defs_ = {{"input1", CompliantNodeBuilder::kEsIrInputRequired, "x"},
                   {"input2", CompliantNodeBuilder::kEsIrInputOptional, "y"},
                   {"input3", CompliantNodeBuilder::kEsIrInputDynamic, "z"}};

    output_defs_ = {{"output1", CompliantNodeBuilder::kEsIrOutputRequired, "out1"},
                    {"output2", CompliantNodeBuilder::kEsIrOutputDynamic, "out2"}};

    // 创建AttrValue对象
    auto attr1_value = AttrValue();
    attr1_value.SetAttrValue(static_cast<int64_t>(1));
    auto attr2_value = AttrValue();

    attr2_value.SetAttrValue(static_cast<float>(2.0f));

    attr_defs_ = {{"attr1", CompliantNodeBuilder::kEsAttrRequired, "Int", attr1_value},
                  {"attr2", CompliantNodeBuilder::kEsAttrOptional, "Float", attr2_value}};

    shape_ = {1, 2, 3, 4};
    subgraphs_ = {"subgraph_inst1", "subgraph_inst2"};
  }

  void TearDown() override {}
  es::EsGraphBuilder test_graph_builder_{"test_graph"};
  std::unique_ptr<Graph> test_graph_{nullptr};
  std::vector<CompliantNodeBuilder::IrInputDef> input_defs_;
  std::vector<CompliantNodeBuilder::IrOutputDef> output_defs_;
  std::vector<CompliantNodeBuilder::IrAttrDef> attr_defs_;
  std::vector<int64_t> shape_;
  std::vector<std::string> subgraphs_;
};

// 测试完整的Build流程 - 基础情况
TEST_F(CompliantNodeBuilderLLT, BuildBasic) {
  CompliantNodeBuilder builder(test_graph_.get());
  builder.OpType("TestOp").Name("TestName").IrDefInputs(input_defs_).IrDefOutputs(output_defs_).IrDefAttrs(attr_defs_);

  auto node = builder.Build();
  OpDescChecker checker(node);
  std::string golden = R"PROTO(name: "TestName"
                               type: "TestOp"
                               attr {
                                 key: "__dynamic_input_input3_cnt"
                                 value { i: 0 }
                               }
                               attr {
                                 key: "__dynamic_output_output2_cnt"
                                 value { i: 0 }
                               }
                               attr {
                                 key: "_input_name_key"
                                 value { list { s: "input1" s: "input2" } }
                               }
                               attr {
                                 key: "_input_name_value"
                                 value { list { i: 0 i: 1 } }
                               }
                               attr {
                                 key: "_ir_attr_names"
                                 value { list { s: "attr1" s: "attr2" } }
                               }
                               attr {
                                 key: "_ir_inputs_key"
                                 value { list { s: "input1" s: "input2" s: "input3" } }
                               }
                               attr {
                                 key: "_ir_inputs_value"
                                 value { list { i: 0 i: 1 i: 2 } }
                               }
                               attr {
                                 key: "_ir_outputs_key"
                                 value { list { s: "output1" s: "output2" } }
                               }
                               attr {
                                 key: "_ir_outputs_value"
                                 value { list { i: 0 i: 1 } }
                               }
                               attr {
                                 key: "_opt_input"
                                 value { list { s: "input2" } }
                               }
                               attr {
                                 key: "_output_name_key"
                                 value { list { s: "output1" } }
                               }
                               attr {
                                 key: "_output_name_value"
                                 value { list { i: 0 } }
                               }
                               attr {
                                 key: "attr1"
                                 value { i: 1 }
                               }
                               attr {
                                 key: "attr2"
                                 value { f: 2 }
                               }
                               has_out_attr: true
                               id: 1
                               input_desc {
                                 dtype: DT_FLOAT
                                 shape {}
                                 layout: "ND"
                                 attr {
                                   key: "format_for_int"
                                   value { i: 2 }
                                 }
                                 attr {
                                   key: "origin_format"
                                   value { s: "ND" }
                                 }
                                 attr {
                                   key: "origin_format_for_int"
                                   value { i: 2 }
                                 }
                                 attr {
                                   key: "origin_shape_initialized"
                                   value { b: false }
                                 }
                                 device_type: "NPU"
                               }
                               input_desc {
                                 shape {}
                                 layout: "FORMAT_RESERVED"
                                 attr {
                                   key: "format_for_int"
                                   value { i: 40 }
                                 }
                                 attr {
                                   key: "origin_format"
                                   value { s: "ND" }
                                 }
                                 attr {
                                   key: "origin_format_for_int"
                                   value { i: 2 }
                                 }
                                 attr {
                                   key: "origin_shape_initialized"
                                   value { b: false }
                                 }
                                 device_type: "NPU"
                               }
                               output_desc {
                                 dtype: DT_FLOAT
                                 shape {}
                                 layout: "ND"
                                 attr {
                                   key: "format_for_int"
                                   value { i: 2 }
                                 }
                                 attr {
                                   key: "origin_format"
                                   value { s: "ND" }
                                 }
                                 attr {
                                   key: "origin_format_for_int"
                                   value { i: 2 }
                                 }
                                 attr {
                                   key: "origin_shape_initialized"
                                   value { b: false }
                                 }
                                 device_type: "NPU"
                               })PROTO";
  checker.CheckAll(golden);
}

// 测试Build流程 - 包含输出数据类型
TEST_F(CompliantNodeBuilderLLT, BuildWithOutputDesc) {
  CompliantNodeBuilder builder(test_graph_.get());
  builder.OpType("TestOp")
      .Name("TestName")
      .IrDefOutputs({{"output1", CompliantNodeBuilder::kEsIrOutputRequired, "out1"}})
      .InstanceOutputShape("output1", shape_)
      .InstanceOutputDataType("output1", DT_INT32)
      .InstanceOutputFormat("output1", FORMAT_NCHW);

  auto node = builder.Build();
  OpDescChecker checker(node);
  checker.CheckOutFormats({FORMAT_NCHW}).CheckOutDataTypes({DT_INT32});
  EXPECT_NE(ge::NodeAdapter::GNode2Node(node), nullptr);
}

// 测试Build流程 - 动静态算子逻辑去除了

// 测试Build流程 - 包含所有组件
TEST_F(CompliantNodeBuilderLLT, BuildComplete) {
  CompliantNodeBuilder builder(test_graph_.get());
  builder.OpType("TestOp")
      .Name("TestName")
      .IrDefInputs(input_defs_)
      .IrDefOutputs(output_defs_)
      .IrDefAttrs(attr_defs_)
      .InstanceDynamicInputNum("input3", 5)
      .InstanceDynamicOutputNum("output2", 3)
      .InstanceOutputShape("output1", shape_)
      .InstanceOutputDataType("output1", DT_INT64)
      .InstanceOutputFormat("output1", FORMAT_NCHW);

  auto node = builder.Build();
  OpDescChecker checker(node);
  checker.CheckId(1)
      .CheckName("TestName")
      .CheckType("TestOp")
      .CheckInputIrInfo({{0, {0, 1}}, {1, {1, 0}}, {2, {2, 5}}})
      .CheckOutputIrInfo({{0, {0, 1}}, {1, {1, 3}}});
  auto ge_node = ge::NodeAdapter::GNode2Node(node);
  EXPECT_NE(ge_node, nullptr);
}

// 测试边界情况 - 空输入
TEST_F(CompliantNodeBuilderLLT, BuildWithEmptyInputs) {
  CompliantNodeBuilder builder(test_graph_.get());
  builder.OpType("TestOp").Name("TestName").IrDefInputs({}).IrDefOutputs(output_defs_);

  auto node = builder.Build();
  EXPECT_NE(ge::NodeAdapter::GNode2Node(node), nullptr);
}

// 测试边界情况 - 空输出
TEST_F(CompliantNodeBuilderLLT, BuildWithEmptyOutputs) {
  CompliantNodeBuilder builder(test_graph_.get());
  builder.OpType("TestOp").Name("TestName").IrDefInputs(input_defs_).IrDefOutputs({});

  auto node = builder.Build();
  EXPECT_NE(ge::NodeAdapter::GNode2Node(node), nullptr);
}

// 测试边界情况 - 空属性
TEST_F(CompliantNodeBuilderLLT, BuildWithEmptyAttrs) {
  CompliantNodeBuilder builder(test_graph_.get());
  builder.OpType("TestOp").Name("TestName").IrDefInputs(input_defs_).IrDefOutputs(output_defs_).IrDefAttrs({});

  auto node = builder.Build();
  EXPECT_NE(ge::NodeAdapter::GNode2Node(node), nullptr);
}

// 测试不同输入类型
TEST_F(CompliantNodeBuilderLLT, DifferentInputTypes) {
  CompliantNodeBuilder builder(test_graph_.get());
  std::vector<CompliantNodeBuilder::IrInputDef> mixed_inputs = {
      {"required_input", CompliantNodeBuilder::kEsIrInputRequired, "req"},
      {"optional_input", CompliantNodeBuilder::kEsIrInputOptional, "opt"},
      {"dynamic_input", CompliantNodeBuilder::kEsIrInputDynamic, "dyn"}};

  builder.OpType("TestOp")
      .Name("TestName")
      .IrDefInputs(mixed_inputs)
      .IrDefOutputs(output_defs_)
      .InstanceDynamicInputNum("dynamic_input", 3);

  auto node = builder.Build();
  EXPECT_NE(ge::NodeAdapter::GNode2Node(node), nullptr);
}

// 测试不同输出类型
TEST_F(CompliantNodeBuilderLLT, DifferentOutputTypes) {
  CompliantNodeBuilder builder(test_graph_.get());
  std::vector<CompliantNodeBuilder::IrOutputDef> mixed_outputs = {
      {"required_output", CompliantNodeBuilder::kEsIrOutputRequired, "req_out"},
      {"dynamic_output", CompliantNodeBuilder::kEsIrOutputDynamic, "dyn_out"}};

  builder.OpType("TestOp")
      .Name("TestName")
      .IrDefInputs(input_defs_)
      .IrDefOutputs(mixed_outputs)
      .InstanceDynamicOutputNum("dynamic_output", 2);

  auto node = builder.Build();
  EXPECT_NE(ge::NodeAdapter::GNode2Node(node), nullptr);
}

// 测试不同属性类型
TEST_F(CompliantNodeBuilderLLT, DifferentAttrTypes) {
  CompliantNodeBuilder builder(test_graph_.get());

  auto required_attr = AttrValue();
  required_attr.SetAttrValue(static_cast<int64_t>(42));
  auto optional_attr = AttrValue();
  optional_attr.SetAttrValue(static_cast<float>(3.14f));
  auto string_attr = AttrValue();
  string_attr.SetAttrValue(AscendString("test"));

  std::vector<CompliantNodeBuilder::IrAttrDef> mixed_attrs = {
      {"required_attr", CompliantNodeBuilder::kEsAttrRequired, "Int", required_attr},
      {"optional_attr", CompliantNodeBuilder::kEsAttrOptional, "Float", optional_attr},
      {"string_attr", CompliantNodeBuilder::kEsAttrOptional, "String", string_attr}};

  builder.OpType("TestOp").Name("TestName").IrDefInputs(input_defs_).IrDefOutputs(output_defs_).IrDefAttrs(mixed_attrs);

  auto node = builder.Build();
  auto ge_node = ge::NodeAdapter::GNode2Node(node);
  EXPECT_TRUE(ge_node != nullptr);
  OpDescChecker(node).CheckAttr({"required_attr", "optional_attr", "string_attr"});
}

// 测试多个动态输入输出
TEST_F(CompliantNodeBuilderLLT, MultipleDynamicIO) {
  CompliantNodeBuilder builder(test_graph_.get());
  std::vector<CompliantNodeBuilder::IrInputDef> multiple_dynamic_inputs = {
      {"input1", CompliantNodeBuilder::kEsIrInputDynamic, "x"},
      {"input2", CompliantNodeBuilder::kEsIrInputDynamic, "y"},
      {"input3", CompliantNodeBuilder::kEsIrInputDynamic, "z"}};

  std::vector<CompliantNodeBuilder::IrOutputDef> multiple_dynamic_outputs = {
      {"output1", CompliantNodeBuilder::kEsIrOutputDynamic, "out1"},
      {"output2", CompliantNodeBuilder::kEsIrOutputDynamic, "out2"}};

  builder.OpType("TestOp")
      .Name("TestName")
      .IrDefInputs(multiple_dynamic_inputs)
      .IrDefOutputs(multiple_dynamic_outputs)
      .InstanceDynamicInputNum("input1", 2)
      .InstanceDynamicInputNum("input2", 3)
      .InstanceDynamicInputNum("input3", 4)
      .InstanceDynamicOutputNum("output1", 2)
      .InstanceDynamicOutputNum("output2", 3);

  auto node = builder.Build();
  OpDescChecker(node)
      .CheckInputIrInfo({{0, {0, 2}}, {1, {2, 3}}, {2, {5, 4}}})
      .CheckSubgraph({})
      .CheckOutputIrInfo({{0, {0, 2}}, {1, {2, 3}}});
  EXPECT_NE(ge::NodeAdapter::GNode2Node(node), nullptr);
}
