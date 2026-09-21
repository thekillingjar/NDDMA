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
#include "ge/ge_api.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_adapter.h"
#include "framework/common/types.h"
#include "graph/utils/op_desc_utils.h"

#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "mmpa/mmpa_api.h"

#include "common/share_graph.h"
#include "faker/space_registry_faker.h"
#include "register/op_tiling_info.h"
#include "register/op_compile_info_base.h"
#include "graph/utils/tensor_utils.h"
#include "exe_graph/runtime/atomic_clean_tiling_context.h"
#include "platform/platform_infos_def.h"
#include "common/op_tiling/op_tiling_rt2.h"
#include "register/op_impl_registry.h"

using namespace std;
using namespace ge;
namespace {
/** data a = 2;
 *  for(int i =0; i<5; ++i){
 *    a=a * 2;
 * }
 *  return a;
 *                     ----------------------------------------------|
 *                    |  const(5)             exit  const(1)         |
 *                    |     \                  /       \             |
 *   data(i)--Enter--merge--less--loopcond--switch-----add-----nextiteration
 *                       \________________\___/
 *                                   ------\------------------------|
 *                                  |       \        const(2)       |
 *                                  |        \         \            |
 *                 data(a)--Enter--merge--switch------mul-----nextiteration
 *                                            \
 *                                           exit
 *                                             \
 *                                           netoutput
 *
 **/
Graph BuildV1ControlFlowGraph() {
  int64_t dims_size = 1;
  vector<int64_t> data_vec = {5};
  for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
  vector<int32_t> data_value_vec(dims_size, 1);
  GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, DT_INT32);
  GeTensorPtr data_tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                  data_value_vec.size() * sizeof(int32_t));

  auto enter = OP_CFG(ENTER).Attr(ENTER_ATTR_FRAME_NAME, "1");
  auto const_op = OP_CFG(CONSTANT).Weight(data_tensor);

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_i", DATA)
              ->NODE("enter_i", enter)
              ->EDGE(0, 0)
              ->NODE("merge_i", MERGE)
              ->NODE("less", LESS)
              ->NODE("loopcond", LOOPCOND));
    CHAIN(NODE("const_1", const_op)
              ->EDGE(0, 1)
              ->NODE("add", ADD)
              ->NODE("iteration_i", NEXTITERATION)
              ->EDGE(0, 1)
              ->NODE("merge_i"));
    CHAIN(NODE("const_5", const_op)->EDGE(0, 1)->NODE("less"));
    CHAIN(NODE("loopcond")
              ->EDGE(0, 1)
              ->NODE("switch_i", SWITCH)
              ->EDGE(0, 0)
              ->NODE("exit_i", EXIT)
              ->EDGE(0, 1)
              ->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("merge_i")->EDGE(0, 0)->NODE("switch_i")->EDGE(1, 0)->NODE("add"));
    CHAIN(NODE("data_a", DATA)
              ->NODE("enter_a", enter)
              ->NODE("merge_a", MERGE)
              ->NODE("switch_a", SWITCH)
              ->NODE("exit_a", EXIT)
              ->EDGE(0, 0)
              ->NODE("netoutput"));
    CHAIN(NODE("iteration_a", NEXTITERATION)->EDGE(0, 1)->NODE("merge_a"));
    CHAIN(NODE("loopcond")->EDGE(0, 1)->NODE("switch_a")->EDGE(1, 0)->NODE("mul", MUL));
    CHAIN(NODE("const_2", const_op)->EDGE(0, 1)->NODE("mul")->EDGE(0, 0)->NODE("iteration_a"));
  };
  // set mul as atomic op
  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto mul = compute_graph->FindNode("mul");
  mul->GetOpDesc()->SetAttr("atomic_output_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({0}));
  return graph;
}
struct DynamicAtomicAddrCleanCompileInfo : public optiling::CompileInfoBase {
  uint32_t workspace_num = 0;
  uint32_t core_num = 0;
  uint32_t ub_size = 0;
  std::vector<int64_t> _workspace_index_list;
};
struct DummyCompileInfo {
  int64_t tiling_key;
  int64_t block_dim;
  bool is_need_atomic;
  int64_t tiling_cond;
  std::vector<int64_t> c;
};
ge::graphStatus DummyTiling(gert::TilingContext *tiling_context) {
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus DummyTilingParse(gert::TilingParseContext *tiling_parse_context) {
  return ge::GRAPH_SUCCESS;
}

UINT32 MemsetTilingParse(gert::KernelContext *kernel_context) {
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(ConcatV2).TilingParse<DummyCompileInfo>(DummyTilingParse).Tiling(DummyTiling);
// 把输入的compile info写入out workspace，用于外部校验vn
ge::graphStatus DummyTilingForDynamicAtomicAddrClean(gert::TilingContext *context) {
  auto compute_node_info = context->GetComputeNodeInfo();
  if (compute_node_info == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto compile_info = reinterpret_cast<const DynamicAtomicAddrCleanCompileInfo *>(context->GetCompileInfo());
  if (compile_info == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const std::vector<int64_t> &workspace_idx = compile_info->_workspace_index_list;

  size_t clean_tensor_num = compute_node_info->GetInputsNum() - 1;

  gert::TilingData *tiling_data = context->GetRawTilingData();
  if (tiling_data == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "op: tiling_data nullptr!");
    return ge::GRAPH_FAILED;
  }

  auto atomic_clean_context = reinterpret_cast<gert::AtomicCleanTilingContext *>(context);
  // write workspace
  size_t *workspace_size = context->GetWorkspaceSizes(6);
  *workspace_size = compile_info->core_num;
  *(workspace_size + 1) = compile_info->ub_size;
  *(workspace_size + 2) = compile_info->workspace_num;
  *(workspace_size + 3) = workspace_idx[0];

  size_t idx = 0U;
  for (; idx < clean_tensor_num; ++idx) {
    auto tensor_size = atomic_clean_context->GetCleanOutputSize(idx);
    *(workspace_size + 4 + idx) = tensor_size;
  }

  if (!workspace_idx.empty()) {
    auto ws_sizes = atomic_clean_context->GetCleanWorkspaceSizes();
    if (ws_sizes == nullptr) {
      GELOGE(ge::GRAPH_FAILED, "op: ws_size nullptr!");
      return ge::GRAPH_FAILED;
    }
    if (ws_sizes->GetSize() == 0U) {
      GELOGE(ge::GRAPH_FAILED, "Clean workspace size is 0!");
      return ge::GRAPH_FAILED;
    }
    auto ws_size_data = reinterpret_cast<const uint64_t *>(ws_sizes->GetData());
    for (size_t i = 0U; i < workspace_idx.size(); ++i, ++idx) {
      auto tensor_size = ws_size_data[workspace_idx[i]];
      *(workspace_size + 4 + idx) = tensor_size;
    }
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace
class FrameworkTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {
    unsetenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM");
  }
};

/**
 *      data   data
 *        \    /
 *         add
 */
TEST_F(FrameworkTest, test_framework_add) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->NODE("add", ADD));
    CHAIN(NODE("data2", DATA)->NODE("add"));
  };

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, ToGeGraph(g1), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    ASSERT_EQ(graph->GetAllNodesSize(), 4);
  };
}

/** data a = 2;
 *  for(int i =0; i<5; ++i){
 *    a=a * 2;
 * }
 *  return a;
 *                     ----------------------------------------------|
 *                    |  const(5)             exit  const(1)         |
 *                    |     \                  /       \             |
 *   data(i)--Enter--merge--less--loopcond--switch-----add-----nextiteration
 *                       \________________\___/
 *                                   ------\------------------------|
 *                                  |       \        const(2)       |
 *                                  |        \         \            |
 *                 data(a)--Enter--merge--switch------mul-----nextiteration
 *                                            \
 *                                           exit
 *                                             \
 *                                           netoutput
 *
 **/
TEST_F(FrameworkTest, test_framework_v1_control_flow) {
  // build graph
  Graph graph = BuildV1ControlFlowGraph();
  // new session & add graph
  map<AscendString, AscendString> options;
  Session session(options);
  auto ret = session.AddGraph(2, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(2, inputs);
  EXPECT_EQ(ret, SUCCESS); // todo fix ret fail because atomic clean
  // check result
}

/**
 *      data  data
 *        \   / |
 *         add  |
 *           \  |
 *            add
 */
TEST_F(FrameworkTest, test_framework_add_fail) {
  setenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM", "1", 0);
  auto add1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1})
                         .Attr(ATTR_NAME_CONTROL_FLOW_GROUP, 0);
  auto add2 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1})
                         .Attr(ATTR_NAME_CONTROL_FLOW_GROUP, 0);
  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", DATA)->EDGE(0, 0)->NODE("add_1", add1));
    CHAIN(NODE("data_2", DATA)->EDGE(0, 1)->NODE("add_1", add1));
    CHAIN(NODE("add_1", add1)->EDGE(0, 0)->NODE("add_2", add2));
    CHAIN(NODE("data_2", DATA)->EDGE(0, 1)->NODE("add_2", add2));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto add_1 = compute_graph->FindNode("add_1");
  add_1->GetOpDesc()->SetAttr("atomic_output_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({0}));
  OpDescPtr op_desc = add_1->GetOpDesc();
  ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_NOTASK, true);
  
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(3, graph, options);

  std::vector<InputTensorInfo> inputs;
  (void)session.BuildGraph(3, inputs);
  CHECK_GRAPH(PreRunAfterBuild) {
    EXPECT_EQ(graph->GetAllNodesSize(), 5);
    EXPECT_TRUE(graph->FindFirstNodeMatchType(MEMSET) == nullptr);
  };
  unsetenv("ENABLE_DYNAMIC_SHAPE_MULTI_STREAM");
}

// This interface use ut is enough
// 校验compile info json null, no cache
TEST_F(FrameworkTest, AicoreParseAndTilingSuccessTwiceWithEmptyTiling) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry(true);
  auto graph = gert::ShareGraph::ConcatV2ConstDependencyGraph();
  auto concatv2_node = graph->FindNode("concatv2");
  (void)ge::AttrUtils::SetStr(concatv2_node->GetOpDesc(), "compile_info_json", "");
  optiling::utils::OpRunInfo run_info;
  auto op = ge::OpDescUtils::CreateOperatorFromNode(concatv2_node);
  fe::PlatFormInfos platform_infos;
  EXPECT_EQ(optiling::AicoreRtParseAndTiling(op, platform_infos, run_info), ge::GRAPH_SUCCESS);
}

TEST_F(FrameworkTest, AicoreParseAndTilingSuccessWithCoreNum) {
  gert::SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  auto graph = gert::ShareGraph::ConcatV2ConstDependencyGraph();
  auto concatv2_node = graph->FindNode("concatv2");
  (void)ge::AttrUtils::SetStr(concatv2_node->GetOpDesc(), "compile_info_json", "");
  (void)ge::AttrUtils::SetStr(concatv2_node->GetOpDesc(), "_op_aicore_num", "2");
  (void)ge::AttrUtils::SetStr(concatv2_node->GetOpDesc(), "_op_vectorcore_num", "5");
  optiling::utils::OpRunInfo run_info;
  auto op = ge::OpDescUtils::CreateOperatorFromNode(concatv2_node);
  fe::PlatFormInfos platform_infos;
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 1);
  EXPECT_EQ(optiling::AicoreRtParseAndTiling(op, platform_infos, run_info), ge::GRAPH_SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

TEST_F(FrameworkTest, AicoreWithAtomicParseAndTilingSuccess) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry();
  typedef void* (*CreateCompileInfo)();
  typedef void (*DeleteCompileInfo)(void *obj);
  CreateCompileInfo create_compile_info = []() -> void *{
    auto info = new DynamicAtomicAddrCleanCompileInfo();
    info->core_num = 8;
    info->ub_size = 131072;
    info->workspace_num = 1;
    info->_workspace_index_list.emplace_back(0);
    return info;
  };

  DeleteCompileInfo delete_compile_info = [](void *obj) -> void {
    if (obj != nullptr) {
      delete (DynamicAtomicAddrCleanCompileInfo *)obj;
      obj = nullptr;
    }
  };

  // mock dynamic atomic clean tiling func
  auto op_impl_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl("MemSet");
  op_impl_func->tiling = DummyTilingForDynamicAtomicAddrClean;
  op_impl_func->tiling_parse = MemsetTilingParse;
  op_impl_func->compile_info_creator = create_compile_info;
  op_impl_func->compile_info_deleter = delete_compile_info;

  auto graph = gert::ShareGraph::BuildAtomicAicoreGraph();
  auto trans1_node = graph->FindNode("trans1");
  trans1_node->GetOpDesc()->SetWorkspaceBytes({256,1,2}); // simulate trans1 node finished tiling
  (void)ge::AttrUtils::SetListInt(trans1_node->GetOpDesc(), "tbe_op_atomic_dtypes", {0});
  ge::TensorUtils::SetSize(*trans1_node->GetOpDesc()->MutableOutputDesc(0), 128);
  std::map<int64_t, int64_t> index_2_workspace_size = {{0,5}};
  std::map<string, std::map<int64_t, int64_t>> atomic_workspace_info = {{"trans1", index_2_workspace_size}};
  trans1_node->GetOpDesc()->SetExtAttr(ge::EXT_ATTR_ATOMIC_WORKSPACE_INFO, atomic_workspace_info);

  optiling::utils::OpRunInfo run_info;
  fe::PlatFormInfos platform_infos;
  auto op = ge::OpDescUtils::CreateOperatorFromNode(trans1_node);
  graphStatus ret = optiling::AtomicRtParseAndTiling(op, platform_infos, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  // expect value
  std::vector<std::pair<std::string, int64_t>> expect_value = {{"core_num", 8},         {"ub_size", 131072},
                                                               {"workspace_num", 1},    {"workspace_index", 0},
                                                               {"clean_out_size", 128}, {"clean_workspace_size", 256}};
  auto workspace = run_info.GetAllWorkspaces();
  for (size_t i = 0; i < run_info.GetWorkspaceNum(); ++i) {
    EXPECT_EQ(workspace[i], expect_value[i].second);
  }
}
