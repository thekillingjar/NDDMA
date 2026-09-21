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
#include "flow_graph/data_flow.h"
#include "flow_graph/flow_graph_utils.h"
#include "proto/dflow.pb.h"
#include "graph/serialization/attr_serializer_registry.h"

using namespace ge::dflow;

namespace ge {
class ProcessPointUTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(ProcessPointUTest, GraphPpNormalTest) {
  ge::Graph graph("user_graph");
  GraphBuilder graph_build = [graph]() { return graph; };
  auto pp1 = GraphPp("pp1", graph_build).SetCompileConfig("./pp1.json");

  ASSERT_EQ(strcmp(pp1.GetProcessPointName(), "pp1"), 0);
  ASSERT_EQ(pp1.GetProcessPointType(), ProcessPointType::GRAPH);
  ASSERT_EQ(strcmp(pp1.GetCompileConfig(), "./pp1.json"), 0);
  auto builder = pp1.GetGraphBuilder();
  auto sub_graph = builder();
  ASSERT_EQ(sub_graph.GetName(), "user_graph");
}

TEST_F(ProcessPointUTest, FlowGraphPpNormalTest) {
  FlowGraph flow_graph("flow_graph");
  FlowGraphBuilder graph_build = [flow_graph]() { return flow_graph; };
  auto pp1 = FlowGraphPp("pp1", graph_build).SetCompileConfig("./pp1.json");
  ASSERT_EQ(strcmp(pp1.GetProcessPointName(), "pp1"), 0);
  ASSERT_EQ(pp1.GetProcessPointType(), ProcessPointType::FLOW_GRAPH);
  ASSERT_EQ(strcmp(pp1.GetCompileConfig(), "./pp1.json"), 0);
  ge::AscendString str;
  pp1.Serialize(str);
  auto builder = pp1.GetGraphBuilder();
  auto sub_graph = builder();
  ASSERT_EQ(sub_graph.GetName(), "flow_graph");
}

TEST_F(ProcessPointUTest, PpNullBuilder) {
  auto pp1 = GraphPp("pp1", nullptr).SetCompileConfig("./pp1.json");
  auto pp2 = GraphPp("pp2", nullptr).SetCompileConfig(nullptr);
  ASSERT_EQ(strcmp(pp1.GetProcessPointName(), "pp1"), 0);
  ASSERT_EQ(pp1.GetProcessPointType(), ProcessPointType::GRAPH);
  ASSERT_EQ(strcmp(pp1.GetCompileConfig(), ""), 0);
  ASSERT_EQ(pp1.GetGraphBuilder(), nullptr);
  ASSERT_EQ(strcmp(pp2.GetCompileConfig(), ""), 0);

  auto pp3 = FlowGraphPp("flowpp1", nullptr).SetCompileConfig("./flowpp1.json");
  FlowGraph flow_graph("flow_graph");
  FlowGraphBuilder graph_build = [flow_graph]() { return flow_graph; };
  auto pp4 = FlowGraphPp("flowpp2", graph_build).SetCompileConfig(nullptr);
  ASSERT_EQ(strcmp(pp3.GetProcessPointName(), "flowpp1"), 0);
  ASSERT_EQ(pp3.GetProcessPointType(), ProcessPointType::FLOW_GRAPH);
  ASSERT_EQ(strcmp(pp3.GetCompileConfig(), ""), 0);
  ASSERT_EQ(pp3.GetGraphBuilder(), nullptr);
  ASSERT_EQ(strcmp(pp4.GetCompileConfig(), ""), 0);
}

TEST_F(ProcessPointUTest, PpNullPpName) {
  auto pp1 = GraphPp(nullptr, nullptr).SetCompileConfig("./pp1.json");
  ASSERT_EQ(pp1.GetProcessPointName(), nullptr);
  ASSERT_EQ(pp1.GetProcessPointType(), ProcessPointType::INVALID);
  ASSERT_EQ(pp1.GetCompileConfig(), nullptr);
  ASSERT_EQ(pp1.GetGraphBuilder(), nullptr);

  auto pp2 = FlowGraphPp(nullptr, nullptr).SetCompileConfig("./pp1.json");
  ASSERT_EQ(pp2.GetProcessPointName(), nullptr);
  ASSERT_EQ(pp2.GetProcessPointType(), ProcessPointType::INVALID);
  ASSERT_EQ(pp2.GetCompileConfig(), nullptr);
  ASSERT_EQ(pp2.GetGraphBuilder(), nullptr);
}

TEST_F(ProcessPointUTest, FunctionPpNormalTest) {
  int64_t batch_value = 124;
  std::vector<int64_t> list_int = {10, 20, 30};
  std::vector<std::vector<int64_t>> list_list_int = {{0, 1}, {-1, 1}};
  bool bool_attr = true;
  std::vector<bool> list_bool = {true, false, true};
  float float_attr = 2.1;
  std::vector<float> list_float = {1.1, 2.2, 3.3};
  ge::DataType dt = DT_DOUBLE;
  std::vector<ge::DataType> list_dt = {DT_BOOL, DT_DOUBLE, DT_INT64};
  ge::AscendString str_value("str_value");
  std::vector<ge::AscendString> list_str_value;
  list_str_value.emplace_back(str_value);
  list_str_value.emplace_back(str_value);

  auto pp1 = FunctionPp("pp1")
                 .SetCompileConfig("./pp1.json")
                 .SetInitParam("_batchsize", batch_value)
                 .SetInitParam("_list_int", list_int)
                 .SetInitParam("_list_list_int", list_list_int)
                 .SetInitParam("_bool_attr", bool_attr)
                 .SetInitParam("_list_bool", list_bool)
                 .SetInitParam("_float_attr", float_attr)
                 .SetInitParam("_list_float", list_float)
                 .SetInitParam("_data_type_attr", dt)
                 .SetInitParam("_list_dt", list_dt)
                 .SetInitParam("_c_str_value", "c_str_value")
                 .SetInitParam("_c_str_null_value", nullptr)
                 .SetInitParam("_str_value", str_value)
                 .SetInitParam("_list_str_value", list_str_value);

  ge::AscendString str;
  pp1.Serialize(str);
  std::string tar_str(str.GetString(), str.GetLength());
  ASSERT_EQ(strcmp(pp1.GetProcessPointName(), "pp1"), 0);
  ASSERT_EQ(pp1.GetProcessPointType(), ProcessPointType::FUNCTION);
  ASSERT_EQ(strcmp(pp1.GetCompileConfig(), "./pp1.json"), 0);

  auto process_point = dataflow::ProcessPoint();
  auto flag = process_point.ParseFromString(tar_str);
  ASSERT_TRUE(flag);
  ASSERT_EQ(process_point.name(), "pp1");
  ASSERT_EQ(process_point.type(), dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  ASSERT_EQ(process_point.compile_cfg_file(), "./pp1.json");

  auto int_attr = process_point.attrs();
  // int check
  auto *int_deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kI);
  ASSERT_NE(int_deserializer, nullptr);
  AnyValue value;
  int_deserializer->Deserialize(int_attr["_batchsize"], value);
  int64_t get_batch_value;
  value.GetValue(get_batch_value);
  ASSERT_EQ(get_batch_value, batch_value);
  value.Clear();
  // list int check
  auto *list_int_deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kList);
  ASSERT_NE(list_int_deserializer, nullptr);
  list_int_deserializer->Deserialize(int_attr["_list_int"], value);
  std::vector<int64_t> get_list_int_value;
  value.GetValue(get_list_int_value);
  ASSERT_EQ(get_list_int_value, list_int);
  value.Clear();
  // bool check
  auto *bool_deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kB);
  ASSERT_NE(bool_deserializer, nullptr);
  bool_deserializer->Deserialize(int_attr["_bool_attr"], value);
  bool get_bool_value;
  value.GetValue(get_bool_value);
  ASSERT_EQ(get_bool_value, bool_attr);
  value.Clear();
  // list bool check
  auto *list_bool_deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kList);
  ASSERT_NE(list_bool_deserializer, nullptr);
  list_bool_deserializer->Deserialize(int_attr["_list_bool"], value);
  std::vector<bool> get_list_bool_value;
  value.GetValue(get_list_bool_value);
  ASSERT_EQ(get_list_bool_value, list_bool);
  value.Clear();
  // float check
  auto *float_deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kF);
  ASSERT_NE(float_deserializer, nullptr);
  float_deserializer->Deserialize(int_attr["_float_attr"], value);
  float get_float_value;
  value.GetValue(get_float_value);
  ASSERT_EQ(get_float_value, float_attr);
  value.Clear();
  // list float check
  auto *list_float_deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kList);
  ASSERT_NE(list_float_deserializer, nullptr);
  list_float_deserializer->Deserialize(int_attr["_list_float"], value);
  std::vector<float> get_list_float_value;
  value.GetValue(get_list_float_value);
  ASSERT_EQ(get_list_float_value, list_float);
  value.Clear();
  // data type check
  auto *dt_deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kDt);
  ASSERT_NE(dt_deserializer, nullptr);
  dt_deserializer->Deserialize(int_attr["_data_type_attr"], value);
  ge::DataType get_dt_value;
  value.GetValue(get_dt_value);
  ASSERT_EQ(get_dt_value, dt);
  value.Clear();
  // list data check
  auto *list_dt_deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kList);
  ASSERT_NE(list_dt_deserializer, nullptr);
  list_dt_deserializer->Deserialize(int_attr["_list_dt"], value);
  std::vector<ge::DataType> get_list_dt_value;
  value.GetValue(get_list_dt_value);
  ASSERT_EQ(get_list_dt_value, list_dt);
  value.Clear();
  // c-string check
  auto *c_str_deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kS);
  ASSERT_NE(c_str_deserializer, nullptr);
  c_str_deserializer->Deserialize(int_attr["_c_str_value"], value);
  std::string get_c_str_value;
  value.GetValue(get_c_str_value);
  ASSERT_EQ(get_c_str_value, "c_str_value");
  value.Clear();
  // c-string null check
  auto *c_str_null_deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kS);
  ASSERT_NE(c_str_null_deserializer, nullptr);
  c_str_null_deserializer->Deserialize(int_attr["_c_str_null_value"], value);
  std::string get_c_str_null_value;
  value.GetValue(get_c_str_null_value);
  ASSERT_EQ(get_c_str_null_value, "");
  value.Clear();
  // string check
  auto *str_deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kS);
  ASSERT_NE(str_deserializer, nullptr);
  str_deserializer->Deserialize(int_attr["_str_value"], value);
  std::string get_str_value;
  value.GetValue(get_str_value);
  ASSERT_EQ(get_str_value, str_value.GetString());
  value.Clear();
  // list string check
  auto *list_str_deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kList);
  ASSERT_NE(list_str_deserializer, nullptr);
  list_str_deserializer->Deserialize(int_attr["_list_str_value"], value);
  std::vector<std::string> get_list_str_value;
  value.GetValue(get_list_str_value);
  ASSERT_EQ(get_list_str_value[0], str_value.GetString());
  ASSERT_EQ(get_list_str_value[1], str_value.GetString());
  value.Clear();
}

TEST_F(ProcessPointUTest, FunctionPpNullPpName) {
  int64_t batch_value = 124;
  std::vector<int64_t> list_int = {10, 20, 30};
  std::vector<std::vector<int64_t>> list_list_int = {{0, 1}, {-1, 1}};
  bool bool_attr = true;
  std::vector<bool> list_bool = {true, false, true};
  float float_attr = 2.1;
  std::vector<float> list_float = {1.1, 2.2, 3.3};
  ge::DataType dt = DT_DOUBLE;
  std::vector<ge::DataType> list_dt = {DT_BOOL, DT_DOUBLE, DT_INT64};
  ge::AscendString str_value("str_value");
  std::vector<ge::AscendString> list_str_value;
  list_str_value.emplace_back(str_value);
  list_str_value.emplace_back(str_value);

  auto pp1 = FunctionPp(nullptr).SetCompileConfig("./pp1.json")
                                .SetInitParam("_batchsize", batch_value)
                                .SetInitParam("_list_int", list_int)
                                .SetInitParam("_list_list_int", list_list_int)
                                .SetInitParam("_bool_attr", bool_attr)
                                .SetInitParam("_list_bool", list_bool)
                                .SetInitParam("_float_attr", float_attr)
                                .SetInitParam("_list_float", list_float)
                                .SetInitParam("_data_type_attr", dt)
                                .SetInitParam("_list_dt", list_dt)
                                .SetInitParam("_str_value", str_value)
                                .SetInitParam("_list_str_value", list_str_value);
  ASSERT_EQ(pp1.GetProcessPointName(), nullptr);
  ASSERT_EQ(pp1.GetProcessPointType(), ProcessPointType::INVALID);
  ASSERT_EQ(pp1.GetCompileConfig(), nullptr);
}

TEST_F(ProcessPointUTest, FunctionPpInvokedPp) {
  ge::Graph ge_graph("ge_graph");
  GraphBuilder graph_build = [ge_graph]() { return ge_graph; };
  GraphBuilder graph_build2 = []() { return ge::Graph("ge_graph2"); };
  auto graphPp1 = GraphPp("graphPp_1", graph_build).SetCompileConfig("./graph.json");
  auto graphPp2 = GraphPp("graphPp_2", graph_build2).SetCompileConfig("./graph2.json");
  auto pp1 = FunctionPp("pp1").AddInvokedClosure("graph1", graphPp1)
                              .AddInvokedClosure("graph1", graphPp2);
  auto invoked_pp = FlowGraphUtils::GetInvokedClosures(dynamic_cast<const FunctionPp *>(&pp1));
  ASSERT_EQ(invoked_pp.size(), 1);

  auto pp2 = FunctionPp("pp2").AddInvokedClosure("graph1", graphPp1)
                              .AddInvokedClosure("graph2", graphPp2);
  auto invoked_pp2 = FlowGraphUtils::GetInvokedClosures(dynamic_cast<const FunctionPp *>(&pp2));
  ASSERT_EQ(invoked_pp2.size(), 2);

  ge::AscendString str;
  pp2.Serialize(str);
  std::string tar_str(str.GetString(), str.GetLength());
  auto process_point = dataflow::ProcessPoint();
  (void)process_point.ParseFromString(tar_str);
  ASSERT_EQ(strcmp(pp2.GetProcessPointName(), "pp2"), 0);
  ASSERT_EQ(pp1.GetProcessPointType(), ProcessPointType::FUNCTION);

  auto invoke_pps = process_point.invoke_pps();
  auto invoke_pp0 = invoke_pps["graph1"];
  ASSERT_EQ(invoke_pp0.name(), "graphPp_1");
  ASSERT_EQ(invoke_pp0.type(), dataflow::ProcessPoint_ProcessPointType_GRAPH);
  ASSERT_EQ(invoke_pp0.compile_cfg_file(), "./graph.json");
  ASSERT_EQ(invoke_pp0.graphs(0), "graphPp_1");

  auto invoke_pp1 = invoke_pps["graph2"];
  ASSERT_EQ(invoke_pp1.name(), "graphPp_2");
  ASSERT_EQ(invoke_pp1.type(), dataflow::ProcessPoint_ProcessPointType_GRAPH);
  ASSERT_EQ(invoke_pp1.compile_cfg_file(), "./graph2.json");
  ASSERT_EQ(invoke_pp1.graphs(0), "graphPp_2");
}

TEST_F(ProcessPointUTest, SetInitParamFailed) {
  const int64_t value = 1;
  auto pp1 = FunctionPp("func_pp").SetInitParam(nullptr, value);
  AscendString str;
  pp1.SetInitParam(nullptr, str);
  std::string s;
  pp1.SetInitParam(nullptr, s.c_str());
  std::vector<AscendString> strs;
  pp1.SetInitParam(nullptr, strs);
  std::vector<int64_t> ints;
  pp1.SetInitParam(nullptr, ints);
  std::vector<std::vector<int64_t>> intss;
  pp1.SetInitParam(nullptr, intss);
  float f = 0;
  pp1.SetInitParam(nullptr, f);
  std::vector<float> fs;
  pp1.SetInitParam(nullptr, fs);
  bool b = false;
  pp1.SetInitParam(nullptr, b);
  std::vector<bool> bs;
  pp1.SetInitParam(nullptr, bs);
  DataType dt = DT_UNDEFINED;
  pp1.SetInitParam(nullptr, dt);
  std::vector<DataType> dts;
  pp1.SetInitParam(nullptr, dts);
  ASSERT_EQ(strcmp(pp1.GetProcessPointName(), "func_pp"), 0);
}

namespace {
class StubProcessPoint : public ProcessPoint {
 public:
  StubProcessPoint(const char_t *name, ProcessPointType type) : ProcessPoint(name, type) {}
  void Serialize(ge::AscendString &str) const override {
    dataflow::ProcessPoint process_point;
    process_point.set_name(this->GetProcessPointName());
    process_point.set_type(dataflow::ProcessPoint_ProcessPointType_INNER);
    std::string target_str;
    process_point.SerializeToString(&target_str);
    str = ge::AscendString(target_str.c_str(), target_str.length());
  }
  StubProcessPoint &SetCompileConfig(const char_t *json_file_path) {
    ProcessPoint::SetCompileConfigFile(json_file_path);
    return *this;
  }
};
class StubProcessPointSerializeFailed : public ProcessPoint {
 public:
  StubProcessPointSerializeFailed(const char_t *name, ProcessPointType type) : ProcessPoint(name, type) {}
  void Serialize(ge::AscendString &str) const override {
    str = ge::AscendString("invalid Serialize content");
  }
  StubProcessPointSerializeFailed &SetCompileConfig(const char_t *json_file_path) {
    ProcessPoint::SetCompileConfigFile(json_file_path);
    return *this;
  }
};
}  // namespace

TEST_F(ProcessPointUTest, SetCompileConfigFileFailed) {
  auto pp = FunctionPp(nullptr).SetCompileConfig("./config.json");
  ASSERT_EQ(pp.GetProcessPointName(), nullptr);
  auto funcpp = FunctionPp("funcpp").SetCompileConfig(nullptr);
  ASSERT_EQ(strcmp(funcpp.GetCompileConfig(), ""), 0);
  auto graphpp = GraphPp("graphpp", []() { return Graph(); }).SetCompileConfig(nullptr);
  ASSERT_EQ(strcmp(funcpp.GetCompileConfig(), ""), 0);
  auto stubpp = StubProcessPoint(nullptr, ProcessPointType::GRAPH).SetCompileConfig("./config");
  ASSERT_EQ(stubpp.GetCompileConfig(), nullptr);
  stubpp = StubProcessPoint("stubpp", ProcessPointType::GRAPH).SetCompileConfig(nullptr);
  ASSERT_EQ(strcmp(stubpp.GetCompileConfig(), ""), 0);
}

TEST_F(ProcessPointUTest, AddInvokedClosureUnsupport) {
  auto funcpp = FunctionPp("funcpp").SetCompileConfig("./config");
  StubProcessPoint stubpp("test stub pp", ProcessPointType::INVALID);
  funcpp.AddInvokedClosure("invoke_key", stubpp);
  ge::AscendString str;
  funcpp.Serialize(str);
  dataflow::ProcessPoint process_point;
  ASSERT_TRUE(process_point.ParseFromArray(str.GetString(), str.GetLength()));
  ASSERT_EQ(process_point.invoke_pps().size(), 0);
}

TEST_F(ProcessPointUTest, AddInvokedClosureMismatch) {
  auto funcpp = FunctionPp("funcpp").SetCompileConfig("./config");
  StubProcessPoint stubpp("test stub pp", ProcessPointType::GRAPH);
  funcpp.AddInvokedClosure("invoke_key", stubpp);
  ge::AscendString str;
  funcpp.Serialize(str);
  dataflow::ProcessPoint process_point;
  ASSERT_TRUE(process_point.ParseFromArray(str.GetString(), str.GetLength()));
  ASSERT_EQ(process_point.invoke_pps().size(), 0);
  StubProcessPoint stubpp2("test stub pp", ProcessPointType::FLOW_GRAPH);
  funcpp.AddInvokedClosure("flow_invoke_key", stubpp2);
  funcpp.Serialize(str);
  ASSERT_TRUE(process_point.ParseFromArray(str.GetString(), str.GetLength()));
  ASSERT_EQ(process_point.invoke_pps().size(), 0);
}

TEST_F(ProcessPointUTest, AddInvokedClosureInner) {
  auto funcpp = FunctionPp("funcpp").SetCompileConfig("./config");
  StubProcessPoint stubpp("test stub pp", ProcessPointType::INNER);
  funcpp.AddInvokedClosure("invoke_key", stubpp);
  ge::AscendString str;
  funcpp.Serialize(str);
  dataflow::ProcessPoint process_point;
  ASSERT_TRUE(process_point.ParseFromArray(str.GetString(), str.GetLength()));
  ASSERT_EQ(process_point.invoke_pps().size(), 1);
  EXPECT_EQ(process_point.invoke_pps().begin()->first, "invoke_key");
  ASSERT_EQ(process_point.invoke_pps().begin()->second.name(), "test stub pp");
}

TEST_F(ProcessPointUTest, AddInvokedClosureSerializeFailed) {
  auto funcpp = FunctionPp("funcpp").SetCompileConfig("./config");
  StubProcessPointSerializeFailed stubpp("test stub pp", ProcessPointType::INNER);
  funcpp.AddInvokedClosure("invoke_key", stubpp);
  ge::AscendString str;
  funcpp.Serialize(str);
  dataflow::ProcessPoint process_point;
  ASSERT_TRUE(process_point.ParseFromArray(str.GetString(), str.GetLength()));
  ASSERT_EQ(process_point.invoke_pps().size(), 0);
}

TEST_F(ProcessPointUTest, FunctionPpInvokedPpByBase) {
  ge::Graph ge_graph("ge_graph");
  GraphBuilder graph_build = [ge_graph]() { return ge_graph; };
  auto graph_pp = GraphPp("graphPp_1", graph_build).SetCompileConfig("./graph.json");
  const ProcessPoint &base_pp = graph_pp;
  auto pp1 = FunctionPp("pp1").AddInvokedClosure("key1", base_pp);

  auto invoked_pp = FlowGraphUtils::GetInvokedClosures(dynamic_cast<const FunctionPp *>(&pp1));
  ASSERT_EQ(invoked_pp.size(), 1);
}

TEST_F(ProcessPointUTest, FunctionPpInvokedPpRepeat) {
  ge::Graph ge_graph("ge_graph");
  GraphBuilder graph_build = [ge_graph]() { return ge_graph; };
  auto graph_pp = GraphPp("graphPp_1", graph_build).SetCompileConfig("./graph.json");
  FunctionPp funcpp("funcpp");
  auto pp1 = funcpp.AddInvokedClosure("repeat_key1", graph_pp);

  StubProcessPoint stubpp1("test stub pp", ProcessPointType::INNER);
  funcpp.AddInvokedClosure("invoke_key", stubpp1);

  StubProcessPoint stubpp2("test stub pp", ProcessPointType::INNER);
  funcpp.AddInvokedClosure("repeat_key1", stubpp2);

  auto invoked_pp = FlowGraphUtils::GetInvokedClosures(dynamic_cast<const FunctionPp *>(&pp1));
  ASSERT_EQ(invoked_pp.size(), 1);
  ge::AscendString str;
  funcpp.Serialize(str);
  dataflow::ProcessPoint process_point;
  ASSERT_TRUE(process_point.ParseFromArray(str.GetString(), str.GetLength()));
  ASSERT_EQ(process_point.invoke_pps().size(), 2);
}

TEST_F(ProcessPointUTest, FunctionPpInvokedPpRepeatInnerFirst) {
  FunctionPp funcpp("funcpp");

  StubProcessPoint stubpp1("test stub pp", ProcessPointType::INNER);
  funcpp.AddInvokedClosure("repeat_key1", stubpp1);

  ge::Graph ge_graph("ge_graph");
  GraphBuilder graph_build = [ge_graph]() { return ge_graph; };
  auto graph_pp = GraphPp("graphPp_1", graph_build).SetCompileConfig("./graph.json");
  auto pp1 = funcpp.AddInvokedClosure("repeat_key1", graph_pp);

  StubProcessPoint stubpp2("test stub pp", ProcessPointType::INNER);
  funcpp.AddInvokedClosure("invoke_key", stubpp2);

  auto invoked_pp = FlowGraphUtils::GetInvokedClosures(dynamic_cast<const FunctionPp *>(&pp1));
  ASSERT_EQ(invoked_pp.size(), 0);
  ge::AscendString str;
  funcpp.Serialize(str);
  dataflow::ProcessPoint process_point;
  ASSERT_TRUE(process_point.ParseFromArray(str.GetString(), str.GetLength()));
  ASSERT_EQ(process_point.invoke_pps().size(), 2);
}

TEST_F(ProcessPointUTest, InvalidPp) {
  auto graphpp = GraphPp("graphpp", nullptr);
  AscendString str;
  graphpp.Serialize(str);
  ASSERT_EQ(str.GetLength(), 0);
  auto funcpp = FunctionPp(nullptr).AddInvokedClosure("graph", graphpp).SetInitParam("attr", "value");
  funcpp.Serialize(str);
  ASSERT_EQ(str.GetLength(), 0);
  ASSERT_TRUE(FlowGraphUtils::GetInvokedClosures(dynamic_cast<const FunctionPp *>(&funcpp)).empty());
  auto flowgraphpp = FlowGraphPp("flowgraphpp", nullptr);
  flowgraphpp.Serialize(str);
  ASSERT_EQ(str.GetLength(), 0);
  auto flowfuncpp = FunctionPp(nullptr).AddInvokedClosure("flowgraph", flowgraphpp);
  flowfuncpp.Serialize(str);
  ASSERT_EQ(str.GetLength(), 0);
  ASSERT_TRUE(FlowGraphUtils::GetInvokedFlowClosures(dynamic_cast<const FunctionPp *>(&flowfuncpp)).empty());
}

TEST_F(ProcessPointUTest, AddInvokedClosureFailed) {
  auto graphpp = GraphPp("graphpp", nullptr);
  auto funcpp = FunctionPp("funcpp").AddInvokedClosure(nullptr, graphpp).AddInvokedClosure("graph", graphpp);
  ASSERT_EQ(FlowGraphUtils::GetInvokedClosures(dynamic_cast<const FunctionPp *>(&funcpp)).size(), 0);
  auto flowgraphpp = FlowGraphPp("flowgraphpp", nullptr);
  auto flowfuncpp = FunctionPp("flowfuncpp").AddInvokedClosure(nullptr, flowgraphpp).AddInvokedClosure("graph", flowgraphpp);
  ASSERT_EQ(FlowGraphUtils::GetInvokedFlowClosures(dynamic_cast<const FunctionPp *>(&flowfuncpp)).size(), 0);
}

TEST_F(ProcessPointUTest, FunctionPpInvokedFlowGraphPp) {
  ge::Graph ge_graph("ge_graph");
  GraphBuilder graph_build = [ge_graph]() { return ge_graph; };
  FlowGraph flow_graph("flow_graph");
  FlowGraphBuilder flow_graph_build = [flow_graph]() { return flow_graph; };
  
  auto graphPp1 = GraphPp("graphPp_1", graph_build).SetCompileConfig("./graph.json");
  auto flowGraphPp2 = FlowGraphPp("graphPp_2", flow_graph_build).SetCompileConfig("./flowgraph.json");
  auto pp1 = FunctionPp("pp1").AddInvokedClosure("graph2", flowGraphPp2)
                              .AddInvokedClosure("graph2", flowGraphPp2);
  auto invoked_pp = FlowGraphUtils::GetInvokedClosures(dynamic_cast<const FunctionPp *>(&pp1));
  auto invoked_flow_pp = FlowGraphUtils::GetInvokedFlowClosures(dynamic_cast<const FunctionPp *>(&pp1));
  ASSERT_EQ(invoked_pp.size(), 0);
  ASSERT_EQ(invoked_flow_pp.size(), 1);

  auto pp2 = FunctionPp("pp2").AddInvokedClosure("graph1", graphPp1)
                              .AddInvokedClosure("graph2", flowGraphPp2);
  auto invoked_pp2 = FlowGraphUtils::GetInvokedClosures(dynamic_cast<const FunctionPp *>(&pp2));
  auto invoked_flow_pp2 = FlowGraphUtils::GetInvokedFlowClosures(dynamic_cast<const FunctionPp *>(&pp2));
  ASSERT_EQ(invoked_pp2.size(), 1);
  ASSERT_EQ(invoked_flow_pp2.size(), 1);

  ge::AscendString str;
  pp2.Serialize(str);
  std::string tar_str(str.GetString(), str.GetLength());
  auto process_point = dataflow::ProcessPoint();
  (void)process_point.ParseFromString(tar_str);
  ASSERT_EQ(strcmp(pp2.GetProcessPointName(), "pp2"), 0);
  ASSERT_EQ(pp1.GetProcessPointType(), ProcessPointType::FUNCTION);

  auto invoke_pps = process_point.invoke_pps();
  auto invoke_pp0 = invoke_pps["graph1"];
  ASSERT_EQ(invoke_pp0.name(), "graphPp_1");
  ASSERT_EQ(invoke_pp0.type(), dataflow::ProcessPoint_ProcessPointType_GRAPH);
  ASSERT_EQ(invoke_pp0.compile_cfg_file(), "./graph.json");
  ASSERT_EQ(invoke_pp0.graphs(0), "graphPp_1");

  auto invoke_pp1 = invoke_pps["graph2"];
  ASSERT_EQ(invoke_pp1.name(), "graphPp_2");
  ASSERT_EQ(invoke_pp1.type(), dataflow::ProcessPoint_ProcessPointType_FLOW_GRAPH);
  ASSERT_EQ(invoke_pp1.compile_cfg_file(), "./flowgraph.json");
  ASSERT_EQ(invoke_pp1.graphs(0), "graphPp_2");
}

} // namespace ge
