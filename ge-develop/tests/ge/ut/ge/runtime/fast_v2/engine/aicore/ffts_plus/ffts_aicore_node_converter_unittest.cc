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
#include "graph/normal_graph/op_desc_impl.h"
#include "graph/op_desc.h"
#include "macro_utils/dt_public_unscope.h"

#include "engine/aicore/converter/aicore_ffts_node_converter.h"
#include "register/ffts_node_converter_registry.h"
#include "register/ffts_node_calculater_registry.h"
#include "engine/aicore/converter/aicore_compile_results.h"
#include "engine/ffts_plus/converter/ffts_plus_common.h"
#include "exe_graph/lowering/lowering_definitions.h"
#include "graph_builder/bg_memory.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "engine/gelocal/inputs_converter.h"
#include "engine/node_converter_utils.h"
#include "engine/aicore/fe_rt2_common.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph_builder/exe_graph_comparer.h"
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "common/topo_checker.h"
#include "common/sgt_slice_type.h"
#include "op_tiling/op_tiling_constants.h"
#include "graph_builder/bg_tiling.h"
#include "register/op_impl_registry.h"
#include "aicore/converter/bg_kernel_launch.h"
#include "common/const_data_helper.h"
#include "graph_tuner/rt2_src/graph_tunner_rt2_stub.h"
#include "runtime/subscriber/global_dumper.h"
#include "graph/utils/execute_graph_utils.h"
#include "graph/utils/graph_dump_utils.h"

using namespace ge;
using namespace gert::bg;

namespace gert {
namespace {
ge::graphStatus InferShapeStub(InferShapeContext *context) { return SUCCESS;}
IMPL_OP(Conv2d).InferShape(InferShapeStub);
IMPL_OP(Relu).InferShape(InferShapeStub);

/*
                            g1

┌────────┐  (0,0)   ┌──────────────┐  (0,0)   ┌───────────┐
│ data_a │ ───────> │ conv2d_fused │ ───────> │ netoutput │
└────────┘          └──────────────┘          └───────────┘
                          |
                          |_origin_fuse_graph属性
                          |
                          |
-----------------------------------------------------------------------------
|                            fuse_origin_graph                              |
|                                                                           |
|┌───────┐  (0,0)   ┌────────┐  (0,0)   ┌──────┐  (0,0)   ┌───────────────┐ |
|│ data1 │ ───────> │ conv2d │ ───────> │ relu │ ───────> │ netoutput_sub │ |
|└───────┘          └────────┘          └──────┘          └───────────────┘ |
-----------------------------------------------------------------------------
*/
} // namespace

class FFTSAicoreNodeConverterUT : public BgTestAutoCreate3StageFrame {
 public:
  bg::ValueHolderPtr node_para_;
  std::unique_ptr<uint8_t[]> host_holder_;

  void CreateNodeParam(ge::NodePtr &node, LoweringGlobalData &global_data) {
    auto add_desc = node->GetOpDesc();
    (void)ge::AttrUtils::SetStr(add_desc, ge::kAttrCalcArgsSizeFunc, ge::kEngineNameAiCore);
    size_t size = 0;
    size_t pre_size = 0;
    // todo get node calculate size interface
    std::unique_ptr<uint8_t[]> pre_args_data = nullptr;
    auto cal_func = GetNodeCalculater(node);
    ASSERT_NE(cal_func, nullptr);
    cal_func(node, &global_data, size, pre_size, pre_args_data);
    ASSERT_NE(pre_args_data, nullptr);
    NodeMemPara node_para;
    node_para.size = size;
    host_holder_ = ge::MakeUnique<uint8_t[]>(size);
    ASSERT_NE(host_holder_, nullptr);
    memcpy(host_holder_.get(), pre_args_data.get(), pre_size);
    node_para.dev_addr = host_holder_.get();
    node_para.host_addr = host_holder_.get();
    node_para_ = bg::ValueHolder::CreateConst(&node_para, sizeof(node_para));
  }
};

// test sink ffts static bin
TEST_F(FFTSAicoreNodeConverterUT, SinkBinForFFTSAicoreTest) {
  auto graph = ShareGraph::AicoreNoCompileResultGraph();
  auto add_node = graph->FindNode("add1");
  auto add_desc = add_node->GetOpDesc();
  string compile_info_key = "compile_info_key";
  string compile_info_json = "{\"_workspace_size_list\":[]}";
  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("sgt/test", std::move(test_bin));
  (void)ge::AttrUtils::SetStr(add_desc, optiling::COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(add_desc, optiling::COMPILE_INFO_JSON, compile_info_json);
  add_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);
  (void)ge::AttrUtils::SetStr(add_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AICUBE");
  std::vector<bg::ValueHolderPtr> tiling_ret;
  auto ret = SinkBinForFFTSAicore(add_node, tiling_ret);
  ASSERT_NE(ret, nullptr);
}
// test sink ffts auto static bin
TEST_F(FFTSAicoreNodeConverterUT, SinkBinForFFTSAutoAicoreTest) {
  auto graph = ShareGraph::AicoreNoCompileResultGraph();
  auto add_node = graph->FindNode("add1");
  auto add_desc = add_node->GetOpDesc();
  std::vector<std::string> c_magic_vec = {"RT_DEV_BINARY_MAGIC_ELF_AICUBE", "RT_DEV_BINARY_MAGIC_ELF_AIVEC"};
  (void)ge::AttrUtils::SetListStr(add_desc, ge::TVM_ATTR_NAME_THREAD_MAGIC, c_magic_vec);
  std::vector<char> test_bin(64, '\0');
  ge::OpKernelBinPtr test_kernel = MakeShared<ge::OpKernelBin>("sgt/test", std::move(test_bin));
  std::vector<ge::OpKernelBinPtr> kernel_bin_ptr_vec = {test_kernel, test_kernel};
  add_desc->SetExtAttr(ge::OP_EXTATTR_NAME_THREAD_TBE_KERNEL, kernel_bin_ptr_vec);
  std::vector<std::string> tbe_kernel_name_vec = {"kernel", "kernel"};
  std::vector<std::string> meta_data_vec = {"magic", "magic"};
  (void)ge::AttrUtils::SetListStr(add_desc, ge::ATTR_NAME_THREAD_TBE_KERNEL_NAME, tbe_kernel_name_vec);
  (void)ge::AttrUtils::SetListStr(add_desc, ge::TVM_ATTR_NAME_THREAD_METADATA, meta_data_vec);
  auto ret = SinkFFTSStaAutoNodeBin(add_node);
  ASSERT_NE(ret, nullptr);

  std::vector<std::string> tmp_magic_vec = {"RT_DEV_BINARY_MAGIC_ELF_AICUBE"};
  (void)ge::AttrUtils::SetListStr(add_desc, ge::TVM_ATTR_NAME_THREAD_MAGIC, tmp_magic_vec);
  ASSERT_EQ(SinkFFTSStaAutoNodeBin(add_node), nullptr);
  (void)ge::AttrUtils::SetListStr(add_desc, ge::TVM_ATTR_NAME_THREAD_MAGIC, c_magic_vec);

   tmp_magic_vec = {"RT_DEV_BINARY_MAGIC_ELF_AICUBE", "NON_MAGIC"};
  (void)ge::AttrUtils::SetListStr(add_desc, ge::TVM_ATTR_NAME_THREAD_MAGIC, tmp_magic_vec);
  ASSERT_EQ(SinkFFTSStaAutoNodeBin(add_node), nullptr);
  (void)ge::AttrUtils::SetListStr(add_desc, ge::TVM_ATTR_NAME_THREAD_MAGIC, c_magic_vec);

  std::vector<ge::OpKernelBinPtr> tmp_kernel_vec = {nullptr, test_kernel};
  add_desc->SetExtAttr(ge::OP_EXTATTR_NAME_THREAD_TBE_KERNEL, tmp_kernel_vec);
  ASSERT_EQ(SinkFFTSStaAutoNodeBin(add_node), nullptr);
  add_desc->SetExtAttr(ge::OP_EXTATTR_NAME_THREAD_TBE_KERNEL, kernel_bin_ptr_vec);

  std::vector<std::string> tmp_kernel_name_vec = {"kernel"};
  (void)ge::AttrUtils::SetListStr(add_desc, ge::ATTR_NAME_THREAD_TBE_KERNEL_NAME, tmp_kernel_name_vec);
  ASSERT_EQ(SinkFFTSStaAutoNodeBin(add_node), nullptr);
  (void)ge::AttrUtils::SetListStr(add_desc, ge::ATTR_NAME_THREAD_TBE_KERNEL_NAME, tbe_kernel_name_vec);
}

// test sink ffts atomic bin
TEST_F(FFTSAicoreNodeConverterUT, SinkBinForFFTSAtomicTest) {
  auto graph = ShareGraph::AicoreNoCompileResultGraph();
  auto add_node = graph->FindNode("add1");
  auto add_desc = add_node->GetOpDesc();
  string compile_info_key = "compile_info_key";
  string compile_info_json = "{\"_workspace_size_list\":[]}";
  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("sgt/test", std::move(test_bin));
  (void)ge::AttrUtils::SetStr(add_desc, optiling::COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(add_desc, optiling::COMPILE_INFO_JSON, compile_info_json);
  add_desc->SetExtAttr(EXT_ATTR_ATOMIC_TBE_KERNEL, test_kernel);
  (void)ge::AttrUtils::SetStr(add_desc, ge::ATOMIC_ATTR_TVM_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
  std::vector<bg::ValueHolderPtr> tiling_ret;
  auto ret = SinkFFTSAtomicBin(add_node);
  ASSERT_NE(ret, nullptr);
}

TEST_F(FFTSAicoreNodeConverterUT, LoweringGraphPostProcTest) {
  const LowerResult *graph_result = nullptr;
  std::vector<bg::ValueHolderPtr> task_ret;
  task_ret.resize(static_cast<size_t>(TaskProcKey::kNUM) - 1);
  const std::vector<bg::ValueHolderPtr> free_vec;
  const std::vector<bg::ValueHolderPtr> alloc_vec;
  ge::Status result = LoweringGraphPostProc(graph_result, task_ret, free_vec, alloc_vec);
  EXPECT_EQ(ge::FAILED, result);
}
}  // namespace gert
