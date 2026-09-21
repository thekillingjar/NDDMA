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
#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"
#include "common/share_graph.h"
#include "common/plugin/ge_make_unique_util.h"
#include <kernel/memory/mem_block.h>
#include "engine/aicore/kernel/mixl2_update_kernel.h"
#include "engine/aicore/kernel/aicore_update_kernel.h"
#include "kernel/common_kernel_impl/sink_node_bin.h"
#include "kernel/memory/single_stream_l2_allocator.h"
#include "stub/gert_runtime_stub.h"
#include "exe_graph/runtime/tensor.h"
#include "engine/aicore/kernel/rt_ffts_plus_launch_args.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "macro_utils/dt_public_scope.h"
#include "common/dump/exception_dumper.h"
#include "subscriber/dumper/executor_dumper.h"
#include "subscriber/profiler/cann_profiler_v2.h"
#include "macro_utils/dt_public_unscope.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "register/op_tiling/op_tiling_constants.h"
#include "depends/profiler/src/dump_stub.h"
#include "register/op_tiling_registry.h"
namespace gert {
namespace kernel {
}
using namespace kernel;
using namespace ge;

class MixL2KernelTestUT : public testing::Test {
 public:
  KernelRegistryImpl &registry = KernelRegistryImpl::GetInstance();
};


ComputeGraphPtr BuildMixl2NodeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("add1", "Add")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("add1", "Add")->EDGE(2, 2)->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1", "Add"));
    CHAIN(NODE("data3", "Data")->EDGE(0, 2)->NODE("add1", "Add"));
    CHAIN(NODE("data4", "Data")->EDGE(0, 3)->NODE("add1", "Add"));
  };
  auto graph = ToComputeGraph(g1);
  auto op_desc = graph->FindNode("add1")->GetOpDesc();

  for (size_t i = 0; i < op_desc->GetAllInputsDesc().size(); ++i) {
    op_desc->MutableInputDesc(i)->SetShape(GeShape({4, -1, -1, 4}));
  }

  for (size_t i = 0; i < op_desc->GetAllOutputsDesc().size(); ++i) {
    op_desc->MutableOutputDesc(i)->SetShape(GeShape({4, -1, -1, 4}));
  }
  return graph;
}

TEST_F(MixL2KernelTestUT, test_mixl2_update_args) {
  GlobalDumper::GetInstance()->SetEnableFlags(
      BuiltInSubscriberUtil::BuildEnableFlags<DumpType>({DumpType::kLiteExceptionDump}));
  ge::DumpStub::GetInstance().Clear();
  // update args
  constexpr uint64_t stub_mem_hbm_addr = 0x22;
  memory::CachingMemAllocator stub_allocator{0, 1};
  ge::MemBlock mem_block(stub_allocator, reinterpret_cast<void *>(stub_mem_hbm_addr), 3);
  memory::SingleStreamL2Allocator single_stream_l2_allocator{&stub_allocator};
  memory::MultiStreamMemBlock ms_block;
  ms_block.ReInit(&single_stream_l2_allocator, &mem_block, memory::BlockAllocType(memory::BlockAllocType::kNorm, 0U));
  GertTensorData workspace_gtd = GertTensorData(ms_block.GetAddr(), ms_block.GetSize(), TensorPlacement::kOnDeviceHbm, 0);
  auto work_space = ContinuousVector::Create<GertTensorData *>(3);
  auto work_space_vector = reinterpret_cast<ContinuousVector *>(work_space.get());
  work_space_vector->SetSize(3);
  auto work_space_ptr = reinterpret_cast<GertTensorData **>(work_space_vector->MutableData());
  for (size_t i = 0; i < work_space_vector->GetSize(); i++) {
    work_space_ptr[i] = &workspace_gtd;
  }

  auto io_addr = ContinuousVector::Create<uintptr_t>(3);
  uint32_t need_mode_addr = 1;
  uint32_t io_addr_num = 3;
  AICoreSinkRet sink_ret;
  auto device_data = std::vector<int8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  GertTensorData tensor_data = {(uint8_t *)device_data.data(), 0U, kTensorPlacementEnd, -1};
  auto run_context = BuildKernelRunContext(static_cast<size_t>(MixL2ArgsInKey::kNUM) + 5, static_cast<size_t>(MixL2ArgsOutKey::kNUM));
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::WORKSPACE)].Set(work_space_vector, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::NEED_MODE_ADDR)].Set(reinterpret_cast<void *>(need_mode_addr), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::SINK_RET)].Set(&sink_ret, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::SHAPE_BUFFER)].Set(&tensor_data, nullptr);
  uint32_t need_assert = 1;
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::HAS_ASSERT)].Set(reinterpret_cast<void *>(need_assert), nullptr);

  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_ADDR_NUM)].Set(reinterpret_cast<void *>(io_addr_num), nullptr);

  gert::StorageShape in_shape = {{5,2,3,4}, {5, 1, 1, 1, 1}};
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START)].Set(&in_shape, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 1].Set(&in_shape, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 2].Set(&in_shape, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 3].Set(&tensor_data, nullptr);
  auto device_data1 = std::vector<int8_t>{10};
  GertTensorData in_data = {(uint8_t *)device_data1.data(), 0U, kTensorPlacementEnd, -1};
  GertTensorData out_data = {(uint8_t *)device_data1.data(), 0U, kTensorPlacementEnd, -1};
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 4].Set(&in_data, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 5].Set(&out_data, nullptr);

  RtFFTSKernelLaunchArgs::ComputeNodeDesc node_desc{};
  node_desc.max_tiling_data = 64;
  node_desc.max_tail_tiling_data = 63;
  node_desc.input_num = 2;
  node_desc.output_num = 1;
  node_desc.addr_num = 3;
  node_desc.workspace_cap = 8;
  node_desc.thread_num_max = 1;
  size_t size = 0;
  auto graph = BuildMixl2NodeGraph();
  auto tmp_node = graph->FindNode("add1");
  auto rt_arg = RtFFTSKernelLaunchArgs::Create(tmp_node, node_desc, size);

  NodeMemPara node_para;
  node_para.size = size;
  char *mem_1 = new char[size];
  char *mem_2 = new char[size];
  node_para.host_addr = (void*)mem_1;
  node_para.dev_addr = (void*)mem_2;
  memcpy_s(node_para.host_addr, node_para.size, rt_arg.get(), node_para.size);

  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 7].Set(&node_para, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("FFTSUpdateMixL2Args")->outputs_creator(nullptr, run_context),
  ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("FFTSUpdateMixL2Args")->run_func(run_context),
      ge::GRAPH_SUCCESS);
  EXPECT_FALSE(registry.FindKernelFuncs("FFTSUpdateMixL2Args")->trace_printer(run_context).empty());
  EXPECT_EQ(DumpStub::GetInstance().GetUnits().size(), 12);
  EXPECT_EQ(DumpStub::GetInstance().GetUnits()[11][0], 12); // 5 +  input size(2) + output size(2) + ws size(3)
  EXPECT_EQ(DumpStub::GetInstance().GetUnits()[11][1], 10);
  delete[] mem_1;
  delete[] mem_2;
}

TEST_F(MixL2KernelTestUT, test_mixl2_dyn_update_args) {
  // update args
  constexpr uint64_t stub_mem_hbm_addr = 0x22;
  memory::CachingMemAllocator stub_allocator{0, 1};
  ge::MemBlock mem_block(stub_allocator, reinterpret_cast<void *>(stub_mem_hbm_addr), 3);
  memory::SingleStreamL2Allocator single_stream_l2_allocator{&stub_allocator};
  memory::MultiStreamMemBlock ms_block;
  ms_block.ReInit(&single_stream_l2_allocator, &mem_block, memory::BlockAllocType(memory::BlockAllocType::kNorm, 0U));
  GertTensorData workspace_gtd = GertTensorData(ms_block.GetAddr(), ms_block.GetSize(), TensorPlacement::kOnDeviceHbm, 0);
  auto work_space = ContinuousVector::Create<GertTensorData *>(3);
  auto work_space_vector = reinterpret_cast<ContinuousVector *>(work_space.get());
  work_space_vector->SetSize(3);
  auto work_space_ptr = reinterpret_cast<GertTensorData **>(work_space_vector->MutableData());
  for (size_t i = 0; i < work_space_vector->GetSize(); i++) {
    work_space_ptr[i] = &workspace_gtd;
  }

  auto io_addr = ContinuousVector::Create<uintptr_t>(3);
  uint32_t need_mode_addr = 1;
  uint32_t io_addr_num = 6;
  AICoreSinkRet sink_ret;
  auto device_data = std::vector<int8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  GertTensorData tensor_data = {(uint8_t *)device_data.data(), 0U, kTensorPlacementEnd, -1};

  auto run_context = BuildKernelRunContext(static_cast<size_t>(MixL2ArgsInKey::kNUM) + 11, static_cast<size_t>(MixL2ArgsOutKey::kNUM));
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::WORKSPACE)].Set(work_space_vector, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::NEED_MODE_ADDR)].Set(reinterpret_cast<void *>(need_mode_addr), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::SINK_RET)].Set(&sink_ret, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::SHAPE_BUFFER)].Set(&tensor_data, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_ADDR_NUM)].Set(reinterpret_cast<void *>(io_addr_num), nullptr);

  gert::StorageShape in_shape = {{5,2,3,4}, {5, 1, 1, 1, 1}};
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START)].Set(&in_shape, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 1].Set(&in_shape, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 2].Set(&in_shape, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 3].Set(&in_shape, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 4].Set(&in_shape, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 5].Set(&in_shape, nullptr);
  auto device_data1 = std::vector<int8_t>{10};
  GertTensorData tensor_data1 = {(uint8_t *)device_data1.data(), 0U, kTensorPlacementEnd, -1};
  GertTensorData tensor_data2 = {(uint8_t *)device_data1.data(), 0U, kTensorPlacementEnd, -1};
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 6].Set(&tensor_data, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 7].Set(&tensor_data1, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 8].Set(&tensor_data2, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 9].Set(&tensor_data, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 10].Set(&tensor_data1, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 11].Set(&tensor_data2, nullptr);

  RtFFTSKernelLaunchArgs::ComputeNodeDesc node_desc{};
  node_desc.max_tiling_data = 64;
  node_desc.max_tail_tiling_data = 63;
  node_desc.input_num = 4;
  node_desc.output_num = 2;
  node_desc.addr_num = 3;
  node_desc.workspace_cap = 8;
  node_desc.thread_num_max = 1;
  node_desc.dynamic_folded = true;
  size_t size = 0;
  auto graph = BuildMixl2NodeGraph();
  auto tmp_node = graph->FindNode("add1");
  std::vector<std::vector<int64_t>> dyn_in_vv = {{1,2}};
  (void)ge::AttrUtils::SetListListInt(tmp_node->GetOpDesc(), kDynamicInputsIndexes, dyn_in_vv);
  std::vector<std::vector<int64_t>> dyn_out_vv = {{1}};
  (void)ge::AttrUtils::SetListListInt(tmp_node->GetOpDesc(), kDynamicOutputsIndexes, dyn_out_vv);
  auto rt_arg = RtFFTSKernelLaunchArgs::Create(tmp_node, node_desc, size);

  NodeMemPara node_para;
  node_para.size = size;
  char *mem_1 = new char[size];
  char *mem_2 = new char[size];
  node_para.host_addr = (void*)mem_1;
  node_para.dev_addr = (void*)mem_2;
  memcpy_s(node_para.host_addr, node_para.size, rt_arg.get(), node_para.size);

  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 13].Set(&node_para, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("FFTSUpdateMixL2Args")->outputs_creator(nullptr, run_context),
  ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("FFTSUpdateMixL2Args")->run_func(run_context),
      ge::GRAPH_SUCCESS);
  EXPECT_FALSE(registry.FindKernelFuncs("FFTSUpdateMixL2Args")->trace_printer(run_context).empty());
  auto rt_args = reinterpret_cast<RtFFTSKernelLaunchArgs *>(mem_1);
  size_t dev_pos = rt_args->GetArgsPointer<uint8_t>(RtFFTSKernelLaunchArgs::kDyInputsHostAddr) - reinterpret_cast<uint8_t*>(rt_args);
  auto dyn_dev_ptr = static_cast<void*>(&(mem_2[dev_pos]));
  auto src_addr = rt_args->GetArgsPointer<uintptr_t>(RtFFTSKernelLaunchArgs::kArgsHostAddr)[2];
  ASSERT_EQ(src_addr, reinterpret_cast<uintptr_t>(dyn_dev_ptr));
  size_t offset = static_cast<char*>(dyn_dev_ptr) -  mem_2;
  ASSERT_EQ(offset % sizeof(uint64_t), 0);
  delete[] mem_1;
  delete[] mem_2;
}

TEST_F(MixL2KernelTestUT, test_mixl2_dyn_update_args_align) {
  // update args
  constexpr uint64_t stub_mem_hbm_addr = 0x22;
  memory::CachingMemAllocator stub_allocator{0, 1};
  ge::MemBlock mem_block(stub_allocator, reinterpret_cast<void *>(stub_mem_hbm_addr), 3);
  memory::SingleStreamL2Allocator single_stream_l2_allocator{&stub_allocator};
  memory::MultiStreamMemBlock ms_block;
  ms_block.ReInit(&single_stream_l2_allocator, &mem_block, memory::BlockAllocType(memory::BlockAllocType::kNorm, 0U));
  GertTensorData workspace_gtd = GertTensorData(ms_block.GetAddr(), ms_block.GetSize(), TensorPlacement::kOnDeviceHbm, 0);
  auto work_space = ContinuousVector::Create<GertTensorData *>(3);
  auto work_space_vector = reinterpret_cast<ContinuousVector *>(work_space.get());
  work_space_vector->SetSize(3);
  auto work_space_ptr = reinterpret_cast<GertTensorData **>(work_space_vector->MutableData());
  for (size_t i = 0; i < work_space_vector->GetSize(); i++) {
  work_space_ptr[i] = &workspace_gtd;
  }

  auto io_addr = ContinuousVector::Create<uintptr_t>(3);
  uint32_t need_mode_addr = 1;
  uint32_t io_addr_num = 6;
  AICoreSinkRet sink_ret;
  auto device_data = std::vector<int8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  GertTensorData tensor_data = {(uint8_t *)device_data.data(), 0U, kTensorPlacementEnd, -1};

  auto run_context = BuildKernelRunContext(static_cast<size_t>(MixL2ArgsInKey::kNUM) + 11, static_cast<size_t>(MixL2ArgsOutKey::kNUM));
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::WORKSPACE)].Set(work_space_vector, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::NEED_MODE_ADDR)].Set(reinterpret_cast<void *>(need_mode_addr), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::SINK_RET)].Set(&sink_ret, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::SHAPE_BUFFER)].Set(&tensor_data, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_ADDR_NUM)].Set(reinterpret_cast<void *>(io_addr_num), nullptr);

  gert::StorageShape in_shape = {{5,2,3,4}, {5, 1, 1, 1, 1}};
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START)].Set(&in_shape, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 1].Set(&in_shape, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 2].Set(&in_shape, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 3].Set(&in_shape, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 4].Set(&in_shape, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 5].Set(&in_shape, nullptr);
  auto device_data1 = std::vector<int8_t>{10};
  GertTensorData tensor_data1 = {(uint8_t *)device_data1.data(), 0U, kTensorPlacementEnd, -1};
  GertTensorData tensor_data2 = {(uint8_t *)device_data1.data(), 0U, kTensorPlacementEnd, -1};
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 6].Set(&tensor_data, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 7].Set(&tensor_data1, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 8].Set(&tensor_data2, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 9].Set(&tensor_data, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 10].Set(&tensor_data1, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 11].Set(&tensor_data2, nullptr);

  RtFFTSKernelLaunchArgs::ComputeNodeDesc node_desc{};
  node_desc.max_tiling_data = 64;
  node_desc.max_tail_tiling_data = 63;
  node_desc.input_num = 4;
  node_desc.output_num = 2;
  node_desc.addr_num = 3;
  node_desc.workspace_cap = 8;
  node_desc.thread_num_max = 1;
  node_desc.dynamic_folded = true;
  size_t size = 0;
  auto graph = BuildMixl2NodeGraph();
  auto tmp_node = graph->FindNode("add1");
  std::vector<std::vector<int64_t>> dyn_in_vv = {{1}};
  (void)ge::AttrUtils::SetListListInt(tmp_node->GetOpDesc(), kDynamicInputsIndexes, dyn_in_vv);
  std::vector<std::vector<int64_t>> dyn_out_vv = {{1}};
  (void)ge::AttrUtils::SetListListInt(tmp_node->GetOpDesc(), kDynamicOutputsIndexes, dyn_out_vv);
  auto rt_arg = RtFFTSKernelLaunchArgs::Create(tmp_node, node_desc, size);

  NodeMemPara node_para;
  node_para.size = size;
  char *mem_1 = new char[size];
  char *mem_2 = new char[size];
  node_para.host_addr = (void*)mem_1;
  node_para.dev_addr = (void*)mem_2;
  memcpy_s(node_para.host_addr, node_para.size, rt_arg.get(), node_para.size);

  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 13].Set(&node_para, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("FFTSUpdateMixL2Args")->outputs_creator(nullptr, run_context),
  ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("FFTSUpdateMixL2Args")->run_func(run_context),
      ge::GRAPH_SUCCESS);
  EXPECT_FALSE(registry.FindKernelFuncs("FFTSUpdateMixL2Args")->trace_printer(run_context).empty());
  auto rt_args = reinterpret_cast<RtFFTSKernelLaunchArgs *>(mem_1);
  size_t dev_pos = rt_args->GetArgsPointer<uint8_t>(RtFFTSKernelLaunchArgs::kDyInputsHostAddr) - reinterpret_cast<uint8_t*>(rt_args);
  auto dyn_dev_ptr = static_cast<void*>(&(mem_2[dev_pos]));
  auto src_addr = rt_args->GetArgsPointer<uintptr_t>(RtFFTSKernelLaunchArgs::kArgsHostAddr)[2];
  ASSERT_EQ(src_addr, reinterpret_cast<uintptr_t>(dyn_dev_ptr));
  size_t offset = static_cast<char*>(dyn_dev_ptr) -  mem_2;
  ASSERT_EQ(offset % sizeof(uint64_t), 0);

  delete[] mem_1;
  delete[] mem_2;
}
TEST_F(MixL2KernelTestUT, test_aicore_update_context) {

  AICoreSubTaskFlush flush_data{};
  flush_data.input_addr_vv[0][0] = 0x1000;
  flush_data.input_addr_vv[0][1] = 0x1001;
  flush_data.output_addr_vv[0][0] = 0x2000;
  flush_data.output_addr_vv[0][1] = 0x2001;
  auto run_context = KernelRunContextFaker().KernelIONum(static_cast<size_t>(MixL2UpdateKey::RESERVED), 1).
                     NodeIoNum(2,2).IrInputNum(2)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                         .Build();
  uint32_t blk_dim = 4;
  uint32_t schem = 1;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::FLUSH_DATA)].Set(&flush_data, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::BLOCK_DIM)]
      .Set(reinterpret_cast<void *>(blk_dim), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::SCHEDULE_MODE)]
      .Set(reinterpret_cast<void *>(schem), nullptr);
  uint32_t ctx_id = 0;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CTX_ID)].Set(reinterpret_cast<void *>(ctx_id), nullptr);
  bool is_mix_ratio = false;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::MIX_RATIO)]
      .Set(reinterpret_cast<void *>(is_mix_ratio), nullptr);
  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(1);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t) ;
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  TransTaskInfo *task_info_ptr = reinterpret_cast<TransTaskInfo*>(holder.get());
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  task_info_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  task_info_ptr->rt_task_info.descBufLen = descBufLen;
  auto *buff_ptr = &task_info_ptr->args[buf_offset];
  auto context = reinterpret_cast<rtFftsPlusMixAicAivCtx_t*>(buff_ptr);
  context->contextType = RT_CTX_TYPE_MIX_AIC;
  context->nonTailBlockdim = 4;
  context->nonTailBlockRatioN = 2;
  rtFftsPlusTaskInfo_t task_inf;
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info_ptr->args);
  task_inf.fftsPlusSqe = ffts_plus_sqe;
  task_inf.descBuf = &task_info_ptr->args[buf_offset];
  ffts_plus_sqe->totalContextNum = 1;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TASK_INFO)].Set(&task_inf, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("MixL2UpdateContext")->run_func(run_context), ge::GRAPH_SUCCESS);
  ctx_id = 1;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CTX_ID)].Set(reinterpret_cast<void *>(ctx_id), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("MixL2UpdateContext")->run_func(run_context), ge::GRAPH_SUCCESS);

  // profiling
  ctx_id = 0;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CTX_ID)].Set(reinterpret_cast<void *>(ctx_id), nullptr);
  ProfNodeAdditionInfo info;
  CannProfilingInfoWrapper prof_info(&info);
  ASSERT_EQ(registry.FindKernelFuncs("MixL2UpdateContext")->profiling_info_filler(run_context, prof_info),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(prof_info.add_infos_[0]->node_basic_info.data.nodeBasicInfo.blockDim, 131076);  // 0x0204
}

TEST_F(MixL2KernelTestUT, test_mix_ratio_update_context) {
  AICoreSubTaskFlush flush_data{};
  flush_data.input_addr_vv[0][0] = 0x1000;
  flush_data.input_addr_vv[0][1] = 0x1001;
  flush_data.output_addr_vv[0][0] = 0x2000;
  flush_data.output_addr_vv[0][1] = 0x2001;
  auto run_context = KernelRunContextFaker().KernelIONum(static_cast<size_t>(MixL2UpdateKey::RESERVED), 1).
                     NodeIoNum(2,2).IrInputNum(2)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                         .Build();
  uint32_t blk_dim = 4;
  uint32_t schem = 1;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::FLUSH_DATA)].Set(&flush_data, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::BLOCK_DIM)]
      .Set(reinterpret_cast<void *>(blk_dim), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::SCHEDULE_MODE)]
      .Set(reinterpret_cast<void *>(schem), nullptr);
  uint32_t ctx_id = 0;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CTX_ID)]
      .Set(reinterpret_cast<void *>(ctx_id), nullptr);
  uint16_t tiling_key = 1;
  auto tilings = ContinuousVector::Create<uint64_t>(1);
  auto tiling_vec = reinterpret_cast<ContinuousVector *>(tilings.get());
  tiling_vec->SetSize(1);
  auto tilings_ptr = reinterpret_cast<uint64_t *>(tiling_vec->MutableData());
  tilings_ptr[0] = 1;
  auto crations = ContinuousVector::Create<int32_t>(1);
  auto cration_vec = reinterpret_cast<ContinuousVector *>(crations.get());
  cration_vec->SetSize(1);
  auto cration_ptr = reinterpret_cast<int32_t *>(cration_vec->MutableData());
  cration_ptr[0] = 2;
  auto vrations = ContinuousVector::Create<int32_t>(1);
  auto vration_vec = reinterpret_cast<ContinuousVector *>(vrations.get());
  vration_vec->SetSize(1);
  auto vration_ptr = reinterpret_cast<int32_t *>(vration_vec->MutableData());
  vration_ptr[0] = 1;
  bool is_mix_ratio = true;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::MIX_RATIO)]
      .Set(reinterpret_cast<void *>(is_mix_ratio), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TILING_KEY)]
      .Set(reinterpret_cast<void *>(tiling_key), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TILING_VEC)].Set(tiling_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CRATIO_VEC)].Set(cration_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::VRATIO_VEC)].Set(vration_vec, nullptr);
  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(1);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t) ;
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  TransTaskInfo *task_info_ptr = reinterpret_cast<TransTaskInfo*>(holder.get());
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  task_info_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  task_info_ptr->rt_task_info.descBufLen = descBufLen;
  auto *buff_ptr = &task_info_ptr->args[buf_offset];
  auto context = reinterpret_cast<rtFftsPlusMixAicAivCtx_t*>(buff_ptr);
  context->contextType = RT_CTX_TYPE_MIX_AIC;
  context->nonTailBlockdim = 4;
  context->nonTailBlockRatioN = 2;
  rtFftsPlusTaskInfo_t task_inf;
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info_ptr->args);
  task_inf.fftsPlusSqe = ffts_plus_sqe;
  task_inf.descBuf = &task_info_ptr->args[buf_offset];
  ffts_plus_sqe->totalContextNum = 1;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TASK_INFO)].Set(&task_inf, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("MixL2UpdateContext")->run_func(run_context), ge::GRAPH_SUCCESS);

  cration_ptr[0] = 1;
  vration_ptr[0] = 2;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CRATIO_VEC)].Set(cration_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::VRATIO_VEC)].Set(vration_vec, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("MixL2UpdateContext")->run_func(run_context), ge::GRAPH_SUCCESS);

  cration_ptr[0] = 0;
  vration_ptr[0] = 1;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CRATIO_VEC)].Set(cration_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::VRATIO_VEC)].Set(vration_vec, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("MixL2UpdateContext")->run_func(run_context), ge::GRAPH_SUCCESS);

  cration_ptr[0] = 1;
  vration_ptr[0] = 0;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CRATIO_VEC)].Set(cration_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::VRATIO_VEC)].Set(vration_vec, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("MixL2UpdateContext")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(MixL2KernelTestUT, test_mix_ratio_update_context_2) {
  AICoreSubTaskFlush flush_data{};
  flush_data.input_addr_vv[0][0] = 0x1000;
  flush_data.input_addr_vv[0][1] = 0x1001;
  flush_data.output_addr_vv[0][0] = 0x2000;
  flush_data.output_addr_vv[0][1] = 0x2001;
  auto run_context = KernelRunContextFaker().KernelIONum(static_cast<size_t>(MixL2UpdateKey::RESERVED), 1).
                     NodeIoNum(2,2).IrInputNum(2)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                         .Build();
  uint32_t blk_dim = 4;
  uint32_t schem = 1;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::FLUSH_DATA)].Set(&flush_data, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::BLOCK_DIM)]
      .Set(reinterpret_cast<void *>(blk_dim), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::SCHEDULE_MODE)]
      .Set(reinterpret_cast<void *>(schem), nullptr);
  uint32_t ctx_id = 0;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CTX_ID)]
      .Set(reinterpret_cast<void *>(ctx_id), nullptr);
  uint16_t tiling_key = 1;
  auto tilings = ContinuousVector::Create<uint64_t>(1);
  auto tiling_vec = reinterpret_cast<ContinuousVector *>(tilings.get());
  tiling_vec->SetSize(1);
  auto tilings_ptr = reinterpret_cast<uint64_t *>(tiling_vec->MutableData());
  tilings_ptr[0] = 1;
  auto crations = ContinuousVector::Create<int32_t>(1);
  auto cration_vec = reinterpret_cast<ContinuousVector *>(crations.get());
  cration_vec->SetSize(1);
  auto cration_ptr = reinterpret_cast<int32_t *>(cration_vec->MutableData());
  cration_ptr[0] = 0;
  auto vrations = ContinuousVector::Create<int32_t>(1);
  auto vration_vec = reinterpret_cast<ContinuousVector *>(vrations.get());
  vration_vec->SetSize(1);
  auto vration_ptr = reinterpret_cast<int32_t *>(vration_vec->MutableData());
  vration_ptr[0] = 1;
  bool is_mix_ratio = true;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::MIX_RATIO)]
      .Set(reinterpret_cast<void *>(is_mix_ratio), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TILING_KEY)]
      .Set(reinterpret_cast<void *>(tiling_key), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TILING_VEC)].Set(tiling_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CRATIO_VEC)].Set(cration_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::VRATIO_VEC)].Set(vration_vec, nullptr);
  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(1);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t) ;
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  TransTaskInfo *task_info_ptr = reinterpret_cast<TransTaskInfo*>(holder.get());
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  task_info_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  task_info_ptr->rt_task_info.descBufLen = descBufLen;
  auto *buff_ptr = &task_info_ptr->args[buf_offset];
  auto context = reinterpret_cast<rtFftsPlusMixAicAivCtx_t*>(buff_ptr);
  context->contextType = RT_CTX_TYPE_MIX_AIC;
  context->nonTailBlockdim = 4;
  context->nonTailBlockRatioN = 2;
  rtFftsPlusTaskInfo_t task_inf;
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info_ptr->args);
  task_inf.fftsPlusSqe = ffts_plus_sqe;
  task_inf.descBuf = &task_info_ptr->args[buf_offset];
  ffts_plus_sqe->totalContextNum = 1;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TASK_INFO)].Set(&task_inf, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("MixL2UpdateContext")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(MixL2KernelTestUT, test_mix_ratio_update_context_3) {
  AICoreSubTaskFlush flush_data{};
  flush_data.input_addr_vv[0][0] = 0x1000;
  flush_data.input_addr_vv[0][1] = 0x1001;
  flush_data.output_addr_vv[0][0] = 0x2000;
  flush_data.output_addr_vv[0][1] = 0x2001;
  auto run_context = KernelRunContextFaker().KernelIONum(static_cast<size_t>(MixL2UpdateKey::RESERVED), 1).
                     NodeIoNum(2,2).IrInputNum(2)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                         .Build();
  uint32_t blk_dim = 4;
  uint32_t schem = 1;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::FLUSH_DATA)].Set(&flush_data, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::BLOCK_DIM)]
      .Set(reinterpret_cast<void *>(blk_dim), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::SCHEDULE_MODE)]
      .Set(reinterpret_cast<void *>(schem), nullptr);
  uint32_t ctx_id = 0;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CTX_ID)]
      .Set(reinterpret_cast<void *>(ctx_id), nullptr);
  uint16_t tiling_key = 1;
  auto tilings = ContinuousVector::Create<uint64_t>(1);
  auto tiling_vec = reinterpret_cast<ContinuousVector *>(tilings.get());
  tiling_vec->SetSize(1);
  auto tilings_ptr = reinterpret_cast<uint64_t *>(tiling_vec->MutableData());
  tilings_ptr[0] = 1;
  auto crations = ContinuousVector::Create<int32_t>(1);
  auto cration_vec = reinterpret_cast<ContinuousVector *>(crations.get());
  cration_vec->SetSize(1);
  auto cration_ptr = reinterpret_cast<int32_t *>(cration_vec->MutableData());
  cration_ptr[0] = 1;
  auto vrations = ContinuousVector::Create<int32_t>(1);
  auto vration_vec = reinterpret_cast<ContinuousVector *>(vrations.get());
  vration_vec->SetSize(1);
  auto vration_ptr = reinterpret_cast<int32_t *>(vration_vec->MutableData());
  vration_ptr[0] = 0;
  bool is_mix_ratio = true;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::MIX_RATIO)]
      .Set(reinterpret_cast<void *>(is_mix_ratio), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TILING_KEY)]
      .Set(reinterpret_cast<void *>(tiling_key), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TILING_VEC)].Set(tiling_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CRATIO_VEC)].Set(cration_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::VRATIO_VEC)].Set(vration_vec, nullptr);
  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(1);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t) ;
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  TransTaskInfo *task_info_ptr = reinterpret_cast<TransTaskInfo*>(holder.get());
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  task_info_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  task_info_ptr->rt_task_info.descBufLen = descBufLen;
  auto *buff_ptr = &task_info_ptr->args[buf_offset];
  auto context = reinterpret_cast<rtFftsPlusMixAicAivCtx_t*>(buff_ptr);
  context->contextType = RT_CTX_TYPE_MIX_AIC;
  context->nonTailBlockdim = 4;
  context->nonTailBlockRatioN = 2;
  rtFftsPlusTaskInfo_t task_inf;
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info_ptr->args);
  task_inf.fftsPlusSqe = ffts_plus_sqe;
  task_inf.descBuf = &task_info_ptr->args[buf_offset];
  ffts_plus_sqe->totalContextNum = 1;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TASK_INFO)].Set(&task_inf, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("MixL2UpdateContext")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(MixL2KernelTestUT, test_mix_ratio_update_context_failed) {
  AICoreSubTaskFlush flush_data{};
  flush_data.input_addr_vv[0][0] = 0x1000;
  flush_data.input_addr_vv[0][1] = 0x1001;
  flush_data.output_addr_vv[0][0] = 0x2000;
  flush_data.output_addr_vv[0][1] = 0x2001;
  auto run_context = KernelRunContextFaker().KernelIONum(static_cast<size_t>(MixL2UpdateKey::RESERVED), 1).
                     NodeIoNum(2,2).IrInputNum(2)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                         .Build();
  uint32_t blk_dim = 4;
  uint32_t schem = 1;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::FLUSH_DATA)].Set(&flush_data, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::BLOCK_DIM)]
      .Set(reinterpret_cast<void *>(blk_dim), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::SCHEDULE_MODE)]
      .Set(reinterpret_cast<void *>(schem), nullptr);
  uint32_t ctx_id = 0;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CTX_ID)]
      .Set(reinterpret_cast<void *>(ctx_id), nullptr);
  uint16_t tiling_key = 0;
  auto tilings = ContinuousVector::Create<uint64_t>(1);
  auto tiling_vec = reinterpret_cast<ContinuousVector *>(tilings.get());
  tiling_vec->SetSize(1);
  auto tilings_ptr = reinterpret_cast<uint64_t *>(tiling_vec->MutableData());
  tilings_ptr[0] = 1;
  auto crations = ContinuousVector::Create<int32_t>(1);
  auto cration_vec = reinterpret_cast<ContinuousVector *>(crations.get());
  cration_vec->SetSize(1);
  auto cration_ptr = reinterpret_cast<int32_t *>(cration_vec->MutableData());
  cration_ptr[0] = 2;
  auto vrations = ContinuousVector::Create<int32_t>(1);
  auto vration_vec = reinterpret_cast<ContinuousVector *>(vrations.get());
  vration_vec->SetSize(1);
  auto vration_ptr = reinterpret_cast<int32_t *>(vration_vec->MutableData());
  vration_ptr[0] = 1;
  bool is_mix_ratio = true;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::MIX_RATIO)]
      .Set(reinterpret_cast<void *>(is_mix_ratio), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TILING_KEY)]
      .Set(reinterpret_cast<void *>(tiling_key), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TILING_VEC)].Set(tiling_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CRATIO_VEC)].Set(cration_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::VRATIO_VEC)].Set(vration_vec, nullptr);
  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(1);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t) ;
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  TransTaskInfo *task_info_ptr = reinterpret_cast<TransTaskInfo*>(holder.get());
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  task_info_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  task_info_ptr->rt_task_info.descBufLen = descBufLen;
  auto *buff_ptr = &task_info_ptr->args[buf_offset];
  auto context = reinterpret_cast<rtFftsPlusMixAicAivCtx_t*>(buff_ptr);
  context->contextType = RT_CTX_TYPE_MIX_AIC;
  context->nonTailBlockdim = 4;
  context->nonTailBlockRatioN = 2;
  rtFftsPlusTaskInfo_t task_inf;
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info_ptr->args);
  task_inf.fftsPlusSqe = ffts_plus_sqe;
  task_inf.descBuf = &task_info_ptr->args[buf_offset];
  ffts_plus_sqe->totalContextNum = 1;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TASK_INFO)].Set(&task_inf, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("MixL2UpdateContext")->run_func(run_context), ge::GRAPH_FAILED);
}

TEST_F(MixL2KernelTestUT, test_mix_ratio_update_context_failed_02) {
  AICoreSubTaskFlush flush_data{};
  flush_data.input_addr_vv[0][0] = 0x1000;
  flush_data.input_addr_vv[0][1] = 0x1001;
  flush_data.output_addr_vv[0][0] = 0x2000;
  flush_data.output_addr_vv[0][1] = 0x2001;
  auto run_context = KernelRunContextFaker().KernelIONum(static_cast<size_t>(MixL2UpdateKey::RESERVED), 1).
                     NodeIoNum(2,2).IrInputNum(2)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                         .Build();
  uint32_t blk_dim = 4;
  uint32_t schem = 1;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::FLUSH_DATA)].Set(&flush_data, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::BLOCK_DIM)]
      .Set(reinterpret_cast<void *>(blk_dim), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::SCHEDULE_MODE)]
      .Set(reinterpret_cast<void *>(schem), nullptr);
  uint32_t ctx_id = 0;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CTX_ID)]
      .Set(reinterpret_cast<void *>(ctx_id), nullptr);
  uint16_t tiling_key = 0;
  auto tilings = ContinuousVector::Create<uint64_t>(1);
  auto tiling_vec = reinterpret_cast<ContinuousVector *>(tilings.get());
  tiling_vec->SetSize(0);
  auto crations = ContinuousVector::Create<int32_t>(1);
  auto cration_vec = reinterpret_cast<ContinuousVector *>(crations.get());
  cration_vec->SetSize(0);
  auto vrations = ContinuousVector::Create<int32_t>(1);
  auto vration_vec = reinterpret_cast<ContinuousVector *>(vrations.get());
  vration_vec->SetSize(0);
  bool is_mix_ratio = true;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::MIX_RATIO)]
      .Set(reinterpret_cast<void *>(is_mix_ratio), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TILING_KEY)]
      .Set(reinterpret_cast<void *>(tiling_key), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TILING_VEC)].Set(tiling_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CRATIO_VEC)].Set(cration_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::VRATIO_VEC)].Set(vration_vec, nullptr);
  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(1);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t) ;
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  TransTaskInfo *task_info_ptr = reinterpret_cast<TransTaskInfo*>(holder.get());
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  task_info_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  task_info_ptr->rt_task_info.descBufLen = descBufLen;
  auto *buff_ptr = &task_info_ptr->args[buf_offset];
  auto context = reinterpret_cast<rtFftsPlusMixAicAivCtx_t*>(buff_ptr);
  context->contextType = RT_CTX_TYPE_MIX_AIC;
  context->nonTailBlockdim = 4;
  context->nonTailBlockRatioN = 2;
  rtFftsPlusTaskInfo_t task_inf;
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info_ptr->args);
  task_inf.fftsPlusSqe = ffts_plus_sqe;
  task_inf.descBuf = &task_info_ptr->args[buf_offset];
  ffts_plus_sqe->totalContextNum = 1;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TASK_INFO)].Set(&task_inf, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("MixL2UpdateContext")->run_func(run_context), ge::GRAPH_FAILED);
}

TEST_F(MixL2KernelTestUT, test_MixL2UpdateGeDataDumpInfo) {
  auto context = BuildKernelRunContext(static_cast<size_t>(MixL2DataDumpKey::RESERVED) + 3, 0);
  uint32_t ctxid = 1;
  uint32_t in_num = 1;
  uint32_t out_num = 1;
  context.value_holder[static_cast<size_t>(MixL2DataDumpKey::CONTEXT_ID)].Set(reinterpret_cast<void *>(ctxid), nullptr);
  context.value_holder[static_cast<size_t>(MixL2DataDumpKey::IN_NUM)].Set(reinterpret_cast<void *>(in_num), nullptr);
  context.value_holder[static_cast<size_t>(MixL2DataDumpKey::OUT_NUM)].Set(reinterpret_cast<void *>(out_num), nullptr);

  auto device_data1 = std::vector<int8_t>{10};
  GertTensorData in_data = {(uint8_t *)device_data1.data(), 0U, kTensorPlacementEnd, -1};
  GertTensorData out_data = {(uint8_t *)device_data1.data(), 0U, kTensorPlacementEnd, -1};
  context.value_holder[static_cast<size_t>(MixL2DataDumpKey::IO_START)].Set(&in_data, nullptr);
  context.value_holder[static_cast<size_t>(MixL2DataDumpKey::IO_START) + 1].Set(&out_data, nullptr);

  ASSERT_NE(registry.FindKernelFuncs("MixL2UpdateDataDumpInfo"), nullptr);
  NodeDumpUnit dump_unit;
  gert::ExecutorDataDumpInfoWrapper wrapper(&dump_unit);
  auto ret = registry.FindKernelFuncs("MixL2UpdateDataDumpInfo")->data_dump_info_filler(context, wrapper);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("MixL2UpdateDataDumpInfo")->run_func(context),
      ge::GRAPH_SUCCESS);
}

TEST_F(MixL2KernelTestUT, test_MixL2UpdateGeExceptionDumpInfo) {
  auto context = BuildKernelRunContext(static_cast<size_t>(MixL2ExceptionDumpKey::RESERVED), 0);
  size_t ws_size = 2;
  auto work_space = ContinuousVector::Create<GertTensorData *>(ws_size);
  auto work_space_vector = reinterpret_cast<ContinuousVector *>(work_space.get());
  work_space_vector->SetSize(ws_size);
  auto work_space_ptr = reinterpret_cast<GertTensorData **>(work_space_vector->MutableData());
  constexpr uint64_t stub_mem_hbm_addr = 0x22;
  memory::CachingMemAllocator stub_allocator{0, 1};
  ge::MemBlock mem_block(stub_allocator, reinterpret_cast<void *>(stub_mem_hbm_addr), 100);
  memory::SingleStreamL2Allocator single_stream_l2_allocator{&stub_allocator};
  memory::MultiStreamMemBlock ms_block;
  ms_block.ReInit(&single_stream_l2_allocator, &mem_block, memory::BlockAllocType(memory::BlockAllocType::kNorm, 0U));
  GertTensorData workspace_gtd = GertTensorData(ms_block.GetAddr(), ms_block.GetSize(), TensorPlacement::kOnDeviceHbm, 0);
  for (size_t i = 0; i < work_space_vector->GetSize(); i++) {
    work_space_ptr[i] = &workspace_gtd;
  }
  context.value_holder[static_cast<size_t>(MixL2ExceptionDumpKey::WORKSPACE)].Set(work_space_vector, nullptr);

  RtFFTSKernelLaunchArgs::ComputeNodeDesc node_desc;
  node_desc.max_tiling_data = 63;
  node_desc.max_tail_tiling_data = 0;
  node_desc.addr_num = 0;
  node_desc.input_num = 2;
  node_desc.output_num = 1;
  node_desc.workspace_cap = 8;
  node_desc.thread_num_max = 1;
  size_t size = 0;
  auto graph = BuildMixl2NodeGraph();
  auto tmp_node = graph->FindNode("add1");
  auto rt_arg = RtFFTSKernelLaunchArgs::Create(tmp_node, node_desc, size);

  NodeMemPara node_para;
  char *mem_1 = new char[size];
  char *mem_2 = new char[size];
  node_para.host_addr = (void*)mem_1;
  node_para.dev_addr = (void*)mem_2;
  (void)memcpy_s(node_para.host_addr, node_para.size, rt_arg.get(), node_para.size);
  context.value_holder[static_cast<size_t>(MixL2ExceptionDumpKey::ARGS_PARA)].Set(&node_para, nullptr);

  ExceptionDumpUint dump_unit;
  ExecutorExceptionDumpInfoWrapper wrapper(&dump_unit);
  ASSERT_NE(registry.FindKernelFuncs("MixL2UpdateExceptionDumpInfo"), nullptr);
  auto ret = registry.FindKernelFuncs("MixL2UpdateExceptionDumpInfo")->exception_dump_info_filler(context, wrapper);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("MixL2UpdateExceptionDumpInfo")->run_func(context),
      ge::GRAPH_SUCCESS);
  delete[] mem_1;
  delete[] mem_2;
}

TEST_F(MixL2KernelTestUT, test_CopyAtomicTilingdata) {
  auto context = BuildKernelRunContext(3, 0);
  RtFFTSKernelLaunchArgs::ComputeNodeDesc node_desc;
  node_desc.max_tiling_data = 63;
  node_desc.max_tail_tiling_data = 0;
  node_desc.addr_num = 0;
  node_desc.input_num = 2;
  node_desc.output_num = 1;
  node_desc.workspace_cap = 8;
  node_desc.thread_num_max = 1;
  node_desc.max_atom_tiling_data = 63;
  size_t size = 0;
  auto graph = BuildMixl2NodeGraph();
  auto tmp_node = graph->FindNode("add1");
  auto rt_arg = RtFFTSKernelLaunchArgs::Create(tmp_node, node_desc, size);

  NodeMemPara node_para;
  node_para.size = size;
  char *mem_1 = new char[size];
  char *mem_2 = new char[size];
  node_para.host_addr = (void*)mem_1;
  node_para.dev_addr = (void*)mem_2;
  (void)memcpy_s(node_para.host_addr, node_para.size, rt_arg.get(), node_para.size);
  auto rt_args = reinterpret_cast<RtFFTSKernelLaunchArgs *>(node_para.host_addr);
  auto &tilingData = rt_args->GetAtomTilingData();
  tilingData.Init(node_desc.max_atom_tiling_data, rt_args->GetArgsPointer<void>(RtFFTSKernelLaunchArgs::FFTSArgsType::kAtomTilingData));
  context.value_holder[0].Set(&node_para, nullptr);

  std::vector<uint8_t> tiling_data = {1,2,3,0,5,0};
  optiling::utils::OpRunInfo run_info;
  run_info.AddTilingData(reinterpret_cast<ge::char_t *>(tiling_data.data()), tiling_data.size());
  std::string tilingdata_str = run_info.GetAllTilingData().str();
  size = tilingdata_str.size();
  context.value_holder[1].Set(const_cast<char *>(tilingdata_str.c_str()), nullptr);
  context.value_holder[2].Set(reinterpret_cast<void *>(size), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("CopyAtomicTilingdata")->run_func(context), ge::GRAPH_SUCCESS);
  rt_args = reinterpret_cast<RtFFTSKernelLaunchArgs *>(node_para.host_addr);
  ASSERT_EQ(rt_args->GetAtomTilingData().GetDataSize(), size);
  auto data_p = static_cast<uint8_t*>(rt_args->GetAtomTilingData().GetData());
  for (size_t i = 0; i < tiling_data.size(); ++i) {
    EXPECT_EQ(data_p[i], tiling_data[i]);
  }
  delete[] mem_1;
  delete[] mem_2;
}

TEST_F(MixL2KernelTestUT, test_CopyTilingdata) {
  auto context = BuildKernelRunContext(3, 0);
  RtFFTSKernelLaunchArgs::ComputeNodeDesc node_desc;
  node_desc.max_tiling_data = 63;
  node_desc.max_tail_tiling_data = 0;
  node_desc.addr_num = 0;
  node_desc.input_num = 2;
  node_desc.output_num = 1;
  node_desc.workspace_cap = 8;
  node_desc.thread_num_max = 1;
  size_t size = 0;
  auto graph = BuildMixl2NodeGraph();
  auto tmp_node = graph->FindNode("add1");
  auto rt_arg = RtFFTSKernelLaunchArgs::Create(tmp_node, node_desc, size);

  NodeMemPara node_para;
  node_para.size = size;
  char *mem_1 = new char[size];
  char *mem_2 = new char[size];
  node_para.host_addr = (void*)mem_1;
  node_para.dev_addr = (void*)mem_2;
  (void)memcpy_s(node_para.host_addr, node_para.size, rt_arg.get(), node_para.size);
  auto rt_args = reinterpret_cast<RtFFTSKernelLaunchArgs *>(node_para.host_addr);
  auto &tilingData = rt_args->GetTilingData();
  tilingData.Init(node_desc.max_tiling_data, rt_args->GetArgsPointer<void>(RtFFTSKernelLaunchArgs::FFTSArgsType::kTilingData));
  context.value_holder[0].Set(&node_para, nullptr);

  std::vector<uint8_t> tiling_data = {1,2,3,0,5,0};
  optiling::utils::OpRunInfo run_info;
  run_info.AddTilingData(reinterpret_cast<ge::char_t *>(tiling_data.data()), tiling_data.size());
  std::string tilingdata_str = run_info.GetAllTilingData().str();
  size = tilingdata_str.size();
  context.value_holder[1].Set(const_cast<char *>(tilingdata_str.c_str()), nullptr);
  context.value_holder[2].Set(reinterpret_cast<void *>(size), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("CopyTilingdata")->run_func(context), ge::GRAPH_SUCCESS);
  rt_args = reinterpret_cast<RtFFTSKernelLaunchArgs *>(node_para.host_addr);
  ASSERT_EQ(rt_args->GetTilingData().GetDataSize(), size);
  auto data_p = static_cast<uint8_t*>(rt_args->GetTilingData().GetData());
  for (size_t i = 0; i < tiling_data.size(); ++i) {
    EXPECT_EQ(data_p[i], tiling_data[i]);
  }
  delete[] mem_1;
  delete[] mem_2;
}
}  // namespace gert
