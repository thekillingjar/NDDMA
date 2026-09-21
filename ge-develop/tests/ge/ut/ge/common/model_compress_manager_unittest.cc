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

#include "macro_utils/dt_public_scope.h"
#include "common/model/model_compress_manager.h"
#include "macro_utils/dt_public_unscope.h"

#include "ge_graph_dsl/graph_dsl.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/types.h"
#include "common/summary_checker.h"
#include "common/share_graph.h"

using namespace std;
using namespace testing;

namespace ge {
constexpr int64_t kNeedCompressVersion = 0x1;
const string kKernelName = "_kernelname";
const string kAtomicKernelName = "_atomic_kernelname";

class UtestModelCompressManager : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(UtestModelCompressManager, Compress_Decompress) {
  auto add = OP_CFG(ADD).Attr(ATTR_NAME_OP_NO_TILING, true)
                        .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF")
                        .Attr("_l2fusion_ToOpStruct", "opL1WorkspaceFlag");
  auto data = OP_CFG(DATA).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto output = OP_CFG(NETOUTPUT).Attr(ATTR_NAME_OP_NO_TILING, true);
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", data)->EDGE(0, 0)->NODE("add", add));
    CHAIN(NODE("add", add)->EDGE(0, 0)->NODE("output", output));
  };
  auto graph = ToComputeGraph(g1);
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  auto ret = ModelCompressManager::Compress(ge_model);
  EXPECT_EQ(ret, SUCCESS);
  auto after_compress_graph = ge_model->GetGraph();
  ASSERT_EQ(gert::SummaryChecker(after_compress_graph)
            .StrictAllNodeTypes({{"Data", 1}, {"Add", 1}, {"NetOutput", 1}}), "success");

  GeAttrValue attr_value;
  int64_t om_compress_version;
  vector<string> enum_attr_names;
  vector<string> enum_attr_values;
  vector<bool> name_use_string_values;

  (void)ge_model->GetAttr(ATTR_MODEL_OM_COMPRESS_VERSION, attr_value);
  attr_value.GetValue(om_compress_version);
  ASSERT_EQ(om_compress_version, kNeedCompressVersion);

  (void)ge_model->GetAttr(ATTR_MODEL_ATTR_NAME_ENUM, attr_value);
  attr_value.GetValue(enum_attr_names);
  ASSERT_EQ(enum_attr_names.size(), 2);
  ASSERT_EQ(enum_attr_names[0], "_op_no_tiling");
  ASSERT_EQ(enum_attr_names[1], "tvm_magic");

  (void)ge_model->GetAttr(ATTR_MODEL_ATTRS_USE_STRING_VALUE, attr_value);
  attr_value.GetValue(name_use_string_values);
  ASSERT_EQ(name_use_string_values.size(), 2);
  ASSERT_EQ(name_use_string_values[0], false);
  ASSERT_EQ(name_use_string_values[1], true);

  (void)ge_model->GetAttr(ATTR_MODEL_ATTR_VALUE_ENUM, attr_value);
  attr_value.GetValue(enum_attr_values);
  ASSERT_EQ(enum_attr_values.size(), 1);
  ASSERT_EQ(enum_attr_values[0], "RT_DEV_BINARY_MAGIC_ELF");

  ret = ModelCompressManager::Decompress(ge_model);
  EXPECT_EQ(ret, SUCCESS);
  auto after_decompress_graph = ge_model->GetGraph();
  ASSERT_EQ(gert::SummaryChecker(after_decompress_graph)
            .StrictAllNodeTypes({{"Data", 1}, {"Add", 1}, {"NetOutput", 1}}), "success");
}

TEST_F(UtestModelCompressManager, Compress_Decompress_With_Tensor_Attrs) {
  const auto graph = gert::ShareGraph::Aicpu4thGraph();
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(graph);
  auto ret = ModelCompressManager::Compress(ge_model);
  EXPECT_EQ(ret, SUCCESS);
  auto after_compress_graph = ge_model->GetGraph();
  GeAttrValue attr_value;
  int64_t om_compress_version;
  vector<string> enum_attr_names;
  vector<string> enum_attr_values;

  (void)ge_model->GetAttr(ATTR_MODEL_OM_COMPRESS_VERSION, attr_value);
  attr_value.GetValue(om_compress_version);
  ASSERT_EQ(om_compress_version, kNeedCompressVersion);

  (void)ge_model->GetAttr(ATTR_MODEL_ATTR_NAME_ENUM, attr_value);
  attr_value.GetValue(enum_attr_names);
  ASSERT_EQ(enum_attr_names.size(), 10);

  (void)ge_model->GetAttr(ATTR_MODEL_ATTR_VALUE_ENUM, attr_value);
  attr_value.GetValue(enum_attr_values);
  ASSERT_EQ(enum_attr_values.size(), 5);

  ret = ModelCompressManager::Decompress(ge_model);
  EXPECT_EQ(ret, SUCCESS);

  GeModelPtr dst_ge_model = std::make_shared<GeModel>();
  ret = ModelCompressManager::CpyModelAttrs2Dst(ge_model, dst_ge_model);
  EXPECT_EQ(ret, SUCCESS);

  (void)dst_ge_model->GetAttr(ATTR_MODEL_ATTR_NAME_ENUM, attr_value);
  attr_value.GetValue(enum_attr_names);
  ASSERT_EQ(enum_attr_names.size(), 10);

  ret = ModelCompressManager::Compress(dst_ge_model);
  EXPECT_EQ(ret, SUCCESS);

  GeAttrValue del_attr_value;
  vector<string> del_enum_attr_names;
  ModelCompressManager::DeleteModelAttrs(dst_ge_model);
  (void)dst_ge_model->GetAttr(ATTR_MODEL_ATTR_NAME_ENUM, del_attr_value);
  del_attr_value.GetValue(del_enum_attr_names);
  ASSERT_EQ(del_enum_attr_names.size(), 0);
}

TEST_F(UtestModelCompressManager, Compress_Decompress_With_Duplicate_Attrs) {
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  vector<int64_t> test_data = {};
  test_data.emplace_back(0);
  auto add = OP_CFG(ADD).Attr("OUTPUT_IS_VAR", test_data)
                        .Attr("OwnerGraphIsUnknown", 0)
                        .Attr("_composite_engine_kernel_lib_name", "ffts_plus")
                        .Attr("_composite_engine_name", "ffts_plus")
                        .Attr("_ge_attr_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                        .Attr("_is_connected_to_netoutput", test_data)
                        .Attr("_lxfusion_engine_name", "DNN_VM_GE_LOCAL")
                        .Attr("_lxfusion_op_kernel_lib_name", "DNN_VM_GE_LOCAL_OP_STORE")
                        .Attr("_no_mem_reuse", 1)
                        .Attr("_no_stream_split", 1)
                        .Attr("_op_no_tiling", 0)
                        .Attr("_sub_stream_id", 0)
                        .Attr("_thread_scope_id", 1)
                        .Attr("data_type", 9)
                        .Attr("index", 0)
                        .Attr("is_first_node", 1);
  auto data = OP_CFG(DATA).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto output = OP_CFG(NETOUTPUT).Attr(ATTR_NAME_OP_NO_TILING, true);
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", data)->EDGE(0, 0)->NODE("add", add));
    CHAIN(NODE("add", add)->EDGE(0, 0)->NODE("output", output));
  };
  auto graph = ToComputeGraph(g1);
  GeModelPtr ge_model = std::make_shared<GeModel>();

  for (const auto &node : graph->GetAllNodes()) {
    const auto &op_desc = node->GetOpDesc();
    for (size_t i = 0U; i < op_desc->GetInputsSize(); i++) {
      const auto &input_tensor = op_desc->MutableInputDesc(i);
      (void)input_tensor->SetAttr("origin_format_is_set", GeAttrValue::CreateFrom<bool>(true));
      (void)input_tensor->SetAttr("origin_shape_range", GeAttrValue::CreateFrom<vector<int64_t>>(test_data));
      (void)input_tensor->SetAttr("placement", GeAttrValue::CreateFrom<vector<int64_t>>(test_data));
      (void)input_tensor->SetAttr("shape_range", GeAttrValue::CreateFrom<vector<int64_t>>(test_data));
    }

    for (size_t i = 0U; i < op_desc->GetOutputsSize(); i++) {
      const auto &output_tensor = op_desc->MutableOutputDesc(i);
      (void)output_tensor->SetAttr("_is_zero_copy_block", GeAttrValue::CreateFrom<bool>(true));
      (void)output_tensor->SetAttr("origin_format_is_set", GeAttrValue::CreateFrom<bool>(true));
      (void)output_tensor->SetAttr("origin_shape_range", GeAttrValue::CreateFrom<vector<int64_t>>(test_data));
      (void)output_tensor->SetAttr("placement", GeAttrValue::CreateFrom<vector<int64_t>>(test_data));
      (void)output_tensor->SetAttr("shape_range", GeAttrValue::CreateFrom<vector<int64_t>>(test_data));
    }
  }

  ge_model->SetGraph(graph);
  auto ret = ModelCompressManager::Compress(ge_model);
  EXPECT_EQ(ret, SUCCESS);

  GeAttrValue attr_value;
  int64_t om_compress_version;
  vector<string> enum_attr_names;
  vector<string> enum_attr_values;
  vector<bool> name_use_string_values;

  (void)ge_model->GetAttr(ATTR_MODEL_OM_COMPRESS_VERSION, attr_value);
  attr_value.GetValue(om_compress_version);
  ASSERT_EQ(om_compress_version, kNeedCompressVersion);

  (void)ge_model->GetAttr(ATTR_MODEL_ATTR_NAME_ENUM, attr_value);
  attr_value.GetValue(enum_attr_names);
  ASSERT_EQ(enum_attr_names.size(), 21);
  ASSERT_EQ(enum_attr_names[0], "OUTPUT_IS_VAR");
  ASSERT_EQ(enum_attr_names[10], "_op_no_tiling");
  ASSERT_EQ(enum_attr_names[19], "shape_range");
  ASSERT_EQ(enum_attr_names[20], "_is_zero_copy_block");

  (void)ge_model->GetAttr(ATTR_MODEL_ATTRS_USE_STRING_VALUE, attr_value);
  attr_value.GetValue(name_use_string_values);
  ASSERT_EQ(name_use_string_values.size(), 21);
  ASSERT_EQ(name_use_string_values[0], false);
  ASSERT_EQ(name_use_string_values[2], true);
  ASSERT_EQ(name_use_string_values[20], false);

  (void)ge_model->GetAttr(ATTR_MODEL_ATTR_VALUE_ENUM, attr_value);
  attr_value.GetValue(enum_attr_values);
  ASSERT_EQ(enum_attr_values.size(), 3);
  ASSERT_EQ(enum_attr_values[0], "ffts_plus");
  ASSERT_EQ(enum_attr_values[1], "DNN_VM_GE_LOCAL_OP_STORE");
  ASSERT_EQ(enum_attr_values[2], "DNN_VM_GE_LOCAL");

  ret = ModelCompressManager::Decompress(ge_model);
  EXPECT_EQ(ret, SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

TEST_F(UtestModelCompressManager, Compress_Decompress_With_Kernelname) {
  vector<int64_t> test_data = {};
  test_data.emplace_back(0);
  auto add = OP_CFG(ADD).Attr("OUTPUT_IS_VAR", test_data);
  auto data = OP_CFG(DATA).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto output = OP_CFG(NETOUTPUT).Attr(ATTR_NAME_OP_NO_TILING, true);
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", data)->EDGE(0, 0)->NODE("add", add));
    CHAIN(NODE("add", add)->EDGE(0, 0)->NODE("output", output));
  };
  auto graph = ToComputeGraph(g1);
  GeModelPtr ge_model = std::make_shared<GeModel>();

  for (const auto &node : graph->GetAllNodes()) {
    const auto &op_desc = node->GetOpDesc();
    AttrUtils::SetStr(op_desc, kKernelName, "KernelName");
    AttrUtils::SetStr(op_desc, kAtomicKernelName, "AtomicKernelName");
  }

  ge_model->SetGraph(graph);
  auto ret = ModelCompressManager::Compress(ge_model);
  EXPECT_EQ(ret, SUCCESS);

  GeAttrValue attr_value;
  int64_t om_compress_version;
  vector<string> enum_attr_names;
  vector<string> enum_attr_values;
  vector<bool> name_use_string_values;

  (void)ge_model->GetAttr(ATTR_MODEL_OM_COMPRESS_VERSION, attr_value);
  attr_value.GetValue(om_compress_version);
  ASSERT_EQ(om_compress_version, kNeedCompressVersion);

  (void)ge_model->GetAttr(ATTR_MODEL_ATTR_NAME_ENUM, attr_value);
  attr_value.GetValue(enum_attr_names);
  ASSERT_EQ(enum_attr_names.size(), 4);
  ASSERT_EQ(enum_attr_names[0], "OUTPUT_IS_VAR");
  ASSERT_EQ(enum_attr_names[1], kAtomicKernelName);
  ASSERT_EQ(enum_attr_names[2], kKernelName);
  ASSERT_EQ(enum_attr_names[3], ATTR_NAME_OP_NO_TILING);

  (void)ge_model->GetAttr(ATTR_MODEL_ATTRS_USE_STRING_VALUE, attr_value);
  attr_value.GetValue(name_use_string_values);
  ASSERT_EQ(name_use_string_values.size(), 4);
  ASSERT_EQ(name_use_string_values[0], false);
  ASSERT_EQ(name_use_string_values[1], true);
  ASSERT_EQ(name_use_string_values[2], true);
  ASSERT_EQ(name_use_string_values[3], false);

  (void)ge_model->GetAttr(ATTR_MODEL_ATTR_VALUE_ENUM, attr_value);
  attr_value.GetValue(enum_attr_values);
  ASSERT_EQ(enum_attr_values.size(), 2);
  ASSERT_EQ(enum_attr_values[0], "AtomicKernelName");
  ASSERT_EQ(enum_attr_values[1], "KernelName");

  ret = ModelCompressManager::Decompress(ge_model);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestModelCompressManager, Compress_Decompress_With_Opname_Kernelname) {
  vector<int64_t> test_data = {};
  test_data.emplace_back(0);
  auto add = OP_CFG(ADD).Attr("OUTPUT_IS_VAR", test_data);
  auto data = OP_CFG(DATA).Attr(ATTR_NAME_OP_NO_TILING, true);
  auto output = OP_CFG(NETOUTPUT).Attr(ATTR_NAME_OP_NO_TILING, true);
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", data)->EDGE(0, 0)->NODE("add", add));
    CHAIN(NODE("add", add)->EDGE(0, 0)->NODE("output", output));
  };
  auto graph = ToComputeGraph(g1);
  GeModelPtr ge_model = std::make_shared<GeModel>();

  for (const auto &node : graph->GetAllNodes()) {
  const auto &op_desc = node->GetOpDesc();
    AttrUtils::SetStr(op_desc, op_desc->GetName() + kKernelName, "KernelName");
    AttrUtils::SetStr(op_desc, op_desc->GetName() + kAtomicKernelName, "AtomicKernelName");
  }

  ge_model->SetGraph(graph);
  auto ret = ModelCompressManager::Compress(ge_model);
  EXPECT_EQ(ret, SUCCESS);

  GeAttrValue attr_value;
  int64_t om_compress_version;
  vector<string> enum_attr_names;
  vector<string> enum_attr_values;
  vector<bool> name_use_string_values;

  (void)ge_model->GetAttr(ATTR_MODEL_OM_COMPRESS_VERSION, attr_value);
  attr_value.GetValue(om_compress_version);
  ASSERT_EQ(om_compress_version, kNeedCompressVersion);

  (void)ge_model->GetAttr(ATTR_MODEL_ATTR_NAME_ENUM, attr_value);
  attr_value.GetValue(enum_attr_names);
  ASSERT_EQ(enum_attr_names.size(), 4);
  ASSERT_EQ(enum_attr_names[0], "OUTPUT_IS_VAR");
  ASSERT_EQ(enum_attr_names[1], kAtomicKernelName);
  ASSERT_EQ(enum_attr_names[2], kKernelName);
  ASSERT_EQ(enum_attr_names[3], ATTR_NAME_OP_NO_TILING);

  (void)ge_model->GetAttr(ATTR_MODEL_ATTRS_USE_STRING_VALUE, attr_value);
  attr_value.GetValue(name_use_string_values);
  ASSERT_EQ(name_use_string_values.size(), 4);
  ASSERT_EQ(name_use_string_values[0], false);
  ASSERT_EQ(name_use_string_values[1], true);
  ASSERT_EQ(name_use_string_values[2], true);
  ASSERT_EQ(name_use_string_values[3], false);

  (void)ge_model->GetAttr(ATTR_MODEL_ATTR_VALUE_ENUM, attr_value);
  attr_value.GetValue(enum_attr_values);
  ASSERT_EQ(enum_attr_values.size(), 2);
  ASSERT_EQ(enum_attr_values[0], "AtomicKernelName");
  ASSERT_EQ(enum_attr_values[1], "KernelName");

  ret = ModelCompressManager::Decompress(ge_model);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestModelCompressManager, common_graph) {
  vector<ComputeGraphPtr> all_graph = {};
  all_graph.emplace_back(gert::ShareGraph::AicoreGraph());
  all_graph.emplace_back(gert::ShareGraph::AtcNanoGraph());
  all_graph.emplace_back(gert::ShareGraph::LstmpGraph());
  all_graph.emplace_back(gert::ShareGraph::IfGraph());
  all_graph.emplace_back(gert::ShareGraph::BuildWithAllConstKnownSubgraph());
  all_graph.emplace_back(gert::ShareGraph::BuildZeroInputAicoreGraph());
  all_graph.emplace_back(gert::ShareGraph::ConcatV2ValueDependencyGraph());
  all_graph.emplace_back(gert::ShareGraph::TensorListGraph());
  all_graph.emplace_back(gert::ShareGraph::BuildDsaRandomNormalKnownGraph());
  for (const auto &graph : all_graph) {
    GeModelPtr ge_model = std::make_shared<GeModel>();
    ge_model->SetGraph(graph);
    auto ret = ModelCompressManager::Compress(ge_model);
    EXPECT_EQ(ret, SUCCESS);

    ret = ModelCompressManager::Decompress(ge_model);
    EXPECT_EQ(ret, SUCCESS);
  }
}
}  // namespace ge
