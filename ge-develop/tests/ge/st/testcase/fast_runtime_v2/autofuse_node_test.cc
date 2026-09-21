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
#include "faker/fake_value.h"
#include "faker/space_registry_faker.h"
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "faker/model_data_faker.h"
#include "runtime/dev.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "stub/gert_runtime_stub.h"
#include "op_impl/less_important_op_impl.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_metadef/common/plugin/plugin_manager.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "engine/aicore/converter/autofuse_node_converter.h"
#include "engine/gelocal/inputs_converter.h"
#include "framework/ge_runtime_stub/include/common/bg_test.h"
#include "graph_metadef/graph/utils/file_utils.h"

using namespace ge;
namespace gert {
namespace {
std::string GetAutofuseSoPath() {
  std::string cmake_binary_path = CMAKE_BINARY_DIR;
  return cmake_binary_path + "/tests/depends/op_stub/libautofuse_stub.so";
}

ge::ComputeGraphPtr BuildAutofuseGraph() {
  auto graph = ShareGraph::AutoFuseNodeGraph();
  (void)ge::AttrUtils::SetInt(graph, "_all_symbol_num", 8);
  auto fused_graph_node = graph->FindNode("fused_graph");
  auto fused_graph_node1 = graph->FindNode("fused_graph1");

  auto autofuse_stub_so = GetAutofuseSoPath();
  std::cout << "bin path: " << autofuse_stub_so << std::endl;
  (void)ge::AttrUtils::SetStr(fused_graph_node->GetOpDesc(), "bin_file_path", autofuse_stub_so);
  (void)ge::AttrUtils::SetStr(fused_graph_node1->GetOpDesc(), "bin_file_path", autofuse_stub_so);

  (void)ge::AttrUtils::SetStr(fused_graph_node->GetOpDesc(), "_symbol_infer_shape_cache_key", "xxxxx");
  (void)ge::AttrUtils::SetStr(fused_graph_node1->GetOpDesc(), "_symbol_infer_shape_cache_key", "xxxxx");

  return graph;
}

class AutofuseNodeST : public bg::BgTestAutoCreate3StageFrame  {
 protected:
  void SetUp() override {
   MmpaStub::GetInstance().Reset();
    BgTestAutoCreate3StageFrame::SetUp();
   // creates a temporary opp directory
   gert::CreateVersionInfo();
  }

  void TearDown() override {
   gert::DestroyVersionInfo();
    BgTestAutoCreate3StageFrame::TearDown();
   // remove the temporary opp directory
   gert::DestroyTempOppPath();
  }
};

TEST_F(AutofuseNodeST, CheckAutofuseSoFromBuffer) {
  auto graph = BuildAutofuseGraph();
  auto fused_graph_node = graph->FindNode("fused_graph");
  auto op_desc = fused_graph_node->GetOpDesc();

  auto autofuse_stub_so = GetAutofuseSoPath();
  std::cout << "bin path: " << autofuse_stub_so << std::endl;

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("AscBackend", false).Build();
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kInit);
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kMain);
  // bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto compile_result = global_data.FindCompiledResult(fused_graph_node);
  ASSERT_NE(compile_result, nullptr);

  // 读取so并转换成二进制
  uint32_t bin_len = 0U;
  auto bin = ge::GetBinDataFromFile(autofuse_stub_so, bin_len);
  const auto &pos = autofuse_stub_so.find_last_of("/");
  ASSERT_TRUE(pos != std::string::npos);
  const auto &so_name = autofuse_stub_so.substr(pos + 1UL);
  const auto &vendor_name = autofuse_stub_so.substr(0, pos);
  std::unique_ptr<char[]> so_bin = std::unique_ptr<char[]>(new(std::nothrow) char[bin_len]);
  std::string so_bin_str(bin.get(), bin_len);
  (void) memcpy_s(so_bin.get(), bin_len, so_bin_str.c_str(), bin_len);
  ge::OpSoBinPtr so_bin_ptr = ge::MakeShared<ge::OpSoBin>(so_name, vendor_name, std::move(so_bin), bin_len);

  // 修改路径为新创建的so
  std::string so_path_for_test = vendor_name + "/1.so";
  (void)ge::AttrUtils::SetStr(fused_graph_node->GetOpDesc(), "bin_file_path", so_path_for_test);
  system(("rm -f " + so_path_for_test).c_str());

  std::map<std::string, ge::OpSoBinPtr> bin_file_buffer_map;
  bin_file_buffer_map[so_path_for_test] = so_bin_ptr;
  // 创建bin_file_buffer
  graph->SetExtAttr<std::map<string, OpSoBinPtr>>("bin_file_buffer", bin_file_buffer_map);
  
  auto data0_ret = LoweringDataNode(graph->FindNode("data0"), data_input);
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  auto data3_ret = LoweringDataNode(graph->FindNode("data3"), data_input);
  ASSERT_TRUE(data0_ret.result.IsSuccess());
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());
  ASSERT_TRUE(data3_ret.result.IsSuccess());

  LowerInput add_input = {{data0_ret.out_shapes[0], data1_ret.out_shapes[0],
                              data2_ret.out_shapes[0], data3_ret.out_shapes[0]},
                          {data0_ret.out_addrs[0], data1_ret.out_addrs[0],
                              data2_ret.out_addrs[0], data3_ret.out_addrs[0]},
                          &global_data};

  gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kExceptionDump}));
  auto autofuse_ret = LoweringAutofuseNode(fused_graph_node, add_input);
  ASSERT_TRUE(autofuse_ret.result.IsSuccess());

  gert::GlobalDumper::GetInstance()->SetEnableFlags(0UL);
  (void)ge::AttrUtils::SetStr(fused_graph_node->GetOpDesc(), "bin_file_path", autofuse_stub_so);
  graph->DelExtAttr("bin_file_buffer");
}

TEST_F(AutofuseNodeST, CheckAutofuseSoNoBinFilePath) {
 auto graph = ShareGraph::AutoFuseNodeGraph();
 (void)ge::AttrUtils::SetInt(graph, "_all_symbol_num", 8);
 auto fused_graph_node = graph->FindNode("fused_graph");
 auto op_desc = fused_graph_node->GetOpDesc();

 auto root_model = GeModelBuilder(graph).BuildGeRootModel();
 auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("AscBackend", false).Build();
 global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kInit);
 global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kMain);
 LowerInput data_input = {{}, {}, &global_data};
 auto compile_result = global_data.FindCompiledResult(fused_graph_node);
 ASSERT_NE(compile_result, nullptr);

 auto data0_ret = LoweringDataNode(graph->FindNode("data0"), data_input);
 auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
 auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
 auto data3_ret = LoweringDataNode(graph->FindNode("data3"), data_input);
 ASSERT_TRUE(data0_ret.result.IsSuccess());
 ASSERT_TRUE(data1_ret.result.IsSuccess());
 ASSERT_TRUE(data2_ret.result.IsSuccess());
 ASSERT_TRUE(data3_ret.result.IsSuccess());

 LowerInput add_input = {{data0_ret.out_shapes[0], data1_ret.out_shapes[0],
                             data2_ret.out_shapes[0], data3_ret.out_shapes[0]},
                         {data0_ret.out_addrs[0], data1_ret.out_addrs[0],
                             data2_ret.out_addrs[0], data3_ret.out_addrs[0]},
                         &global_data};

 auto autofuse_ret = LoweringAutofuseNode(fused_graph_node, add_input);
 ASSERT_FALSE(autofuse_ret.result.IsSuccess());

 graph->DelExtAttr("bin_file_buffer");
}

TEST_F(AutofuseNodeST, CheckAutofuseSoNoBinFileBuffer) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry(true);
  auto graph = BuildAutofuseGraph();
  auto fused_graph_node = graph->FindNode("fused_graph");
  auto op_desc = fused_graph_node->GetOpDesc();

  auto autofuse_stub_so = GetAutofuseSoPath();
  std::cout << "bin path: " << autofuse_stub_so << std::endl;

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("AscBackend", false).Build();
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kInit);
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kMain);
  LowerInput data_input = {{}, {}, &global_data};
  auto compile_result = global_data.FindCompiledResult(fused_graph_node);
  ASSERT_NE(compile_result, nullptr);

  // 读取so并转换成二进制
  uint32_t bin_len = 0U;
  auto bin = ge::GetBinDataFromFile(autofuse_stub_so, bin_len);
  const auto &pos = autofuse_stub_so.find_last_of("/");
  ASSERT_TRUE(pos != std::string::npos);
  const auto &so_name = autofuse_stub_so.substr(pos + 1UL);
  const auto &vendor_name = autofuse_stub_so.substr(0, pos);
  std::unique_ptr<char[]> so_bin = std::unique_ptr<char[]>(new(std::nothrow) char[bin_len]);
  std::string so_bin_str(bin.get(), bin_len);
  (void) memcpy_s(so_bin.get(), bin_len, so_bin_str.c_str(), bin_len);
  ge::OpSoBinPtr so_bin_ptr = ge::MakeShared<ge::OpSoBin>(so_name, vendor_name, std::move(so_bin), bin_len);

  // 修改路径为新创建的so
  std::string so_path_for_test = vendor_name + "/1.so";
  system(("rm -f " + so_path_for_test).c_str());

  std::map<std::string, ge::OpSoBinPtr> bin_file_buffer_map;
  bin_file_buffer_map[so_path_for_test] = so_bin_ptr;
  // 创建bin_file_buffer
  graph->SetExtAttr<std::map<string, OpSoBinPtr>>("bin_file_buffer", bin_file_buffer_map);

  auto data0_ret = LoweringDataNode(graph->FindNode("data0"), data_input);
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  auto data3_ret = LoweringDataNode(graph->FindNode("data3"), data_input);
  ASSERT_TRUE(data0_ret.result.IsSuccess());
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());
  ASSERT_TRUE(data3_ret.result.IsSuccess());

  LowerInput add_input = {{data0_ret.out_shapes[0], data1_ret.out_shapes[0],
                              data2_ret.out_shapes[0], data3_ret.out_shapes[0]},
                          {data0_ret.out_addrs[0], data1_ret.out_addrs[0],
                              data2_ret.out_addrs[0], data3_ret.out_addrs[0]},
                          &global_data};

  auto autofuse_ret = LoweringAutofuseNode(fused_graph_node, add_input);
  ASSERT_FALSE(autofuse_ret.result.IsSuccess());

  graph->DelExtAttr("bin_file_buffer");
}
}
}