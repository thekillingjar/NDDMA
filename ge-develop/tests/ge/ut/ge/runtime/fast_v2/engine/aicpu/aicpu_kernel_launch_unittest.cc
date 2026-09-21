/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <gtest/gtest.h>
#include <kernel/memory/mem_block.h>
#include "stub/gert_runtime_stub.h"
#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"
#include "aicpu_engine_struct.h"
#include "engine/aicpu/kernel/aicpu_args_handler.h"
#include "engine/aicpu/kernel/aicpu_ext_info_handle.h"
#include "engine/aicpu/kernel/aicpu_resource_manager.h"
#include "graph/op_desc.h"
#include "graph/compute_graph.h"
#include "graph/op_kernel_bin.h"
#include "aicpu_task_struct.h"
#include "faker/aicpu_ext_info_faker.h"
#include "kernel/memory/host_mem_allocator.h"
#include "core/utils/tensor_utils.h"
#include "graph/load/model_manager/model_manager.h"
#include "subscriber/dumper/executor_dumper.h"

namespace gert {
using AicpuExtInfo = aicpu::FWKAdapter::ExtInfo;
using AicpuShapeAndType = aicpu::FWKAdapter::ShapeAndType;
using CustAICPUKernelPtr = std::shared_ptr<ge::OpKernelBin>;

namespace {
struct AicpuTaskStruct {
  aicpu::AicpuParamHead head;
  uint64_t io_addrp[6];
}__attribute__((packed));
} // namespace

class AicpuKernelLaunchUT : public testing::Test {
 public:
  memory::HostMemAllocator host_mem_allocator_;
  memory::HostGertMemAllocator host_gert_mem_allocator_{&host_mem_allocator_};
  KernelRegistry &registry = KernelRegistry::GetInstance();
};

TEST_F(AicpuKernelLaunchUT, test_run_launch_tf_kernel) {
  auto run_context = BuildKernelRunContext(2, 0);
  AicpuTfArgsHandler args_handler("temp", 1, false);
  std::string arg_data = "111";
  std::string task_info = "222";
  size_t ext_info_size = 20;
  args_handler.SetAsyncTimeout(0);
  args_handler.SetBlockOp(true);
  rtEvent_t rtEvent = reinterpret_cast<void *>(1);
  args_handler.SetRtEvent(rtEvent);
  args_handler.BuildTfArgs(arg_data , task_info, ext_info_size, 0, 0);
  run_context.value_holder[0].Set(&args_handler, nullptr);

  auto temp_addr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[20]);
  GertTensorData temp_buffer = {temp_addr.get(), 20, kOnHost, -1};
  run_context.value_holder[1].Set(&temp_buffer, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("UpdateAicpuIoAddr")->run_func(run_context), ge::GRAPH_SUCCESS);
  uint64_t *io_addrs = reinterpret_cast<uint64_t *>(args_handler.GetIoAddr());
  EXPECT_EQ(io_addrs[0], 0);
  temp_buffer.SetPlacement(kTensorPlacementEnd);
  ASSERT_EQ(registry.FindKernelFuncs("UpdateAicpuIoAddr")->run_func(run_context), ge::GRAPH_SUCCESS);
  EXPECT_NE(io_addrs[0], 0);

  auto run_context2 = BuildKernelRunContext(4, 0);
  run_context2.value_holder[0].Set(&args_handler, nullptr);
  rtStream_t stream = nullptr;
  run_context2.value_holder[1].Set(stream, nullptr);
  rtFuncHandle bin_handle = nullptr;
  run_context2.value_holder[2].Set(&bin_handle, nullptr);
  std::string node_type = "test_aicpu";
  run_context2.value_holder[3].Set(const_cast<char *>(node_type.c_str()), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("AicpuLaunchTfKernel")->run_func(run_context2), ge::GRAPH_SUCCESS);

  auto ret = registry.FindKernelFuncs("AicpuLaunchTfKernel")->trace_printer(run_context2);
  EXPECT_FALSE(ret.empty());
}

TEST_F(AicpuKernelLaunchUT, test_FillLaunchArgs) {
  std::string args(100, '0');
  AicpuTaskStruct aicpu_args;
  aicpu_args.head.length = args.size();
  aicpu_args.head.ioAddrNum = 1;
  ASSERT_EQ(memcpy_s(args.data(), args.size(), &aicpu_args, sizeof(AicpuTaskStruct)), EOK);
  AicpuCCArgsHandler args_handler("temp", 1, false);
  args_handler.BuildHostCCArgs(args, 1);

  auto run_context = BuildKernelRunContext(1, 0);
  run_context.value_holder[0].Set(&args_handler, nullptr);

  ExceptionDumpUint dump_unit;
  ExecutorExceptionDumpInfoWrapper wrapper(&dump_unit);
  ASSERT_EQ(registry.FindKernelFuncs("BuildHostCCArgs")->exception_dump_info_filler(run_context, wrapper), ge::GRAPH_SUCCESS);
  EXPECT_EQ(dump_unit.is_host_args, true);
}

TEST_F(AicpuKernelLaunchUT, test_create_and_destroy_session) {
  auto run_context = BuildKernelRunContext(1, 1);
  uint64_t session_id  = 1;
  run_context.value_holder[0].Set(reinterpret_cast<void *>(session_id), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("EnsureCreateTfSession")->run_func(run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("EnsureCreateTfSession")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(AicpuKernelLaunchUT, test_calc_block_dim) {
  auto run_context = BuildKernelRunContext(2, 1);
  int32_t block_dim_index = 0;
  Shape shape({1});
  uint32_t block_dim = 0;
  run_context.value_holder[0].Set(reinterpret_cast<void *>(block_dim_index), nullptr);
  run_context.value_holder[1].Set(&shape, nullptr);
  run_context.value_holder[2].Set(reinterpret_cast<void *>(block_dim), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("CalcBlockDim")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(AicpuKernelLaunchUT, test_build_aicpu_args) {
  auto alloc_handle_context = BuildKernelRunContext(11, 1);
  auto ext_data = GetFakeExtInfo();
  auto ext_len = ext_data.size();
  std::string node_name = "Add";
  auto input_num = 2U;
  auto output_num = 1U;
  auto unknown_shape_type = ge::DEPEND_COMPUTE;
  auto host_args_addr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[500]);
  auto device_args_addr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[500]);
  uint64_t session_id = 0U;
  bool is_block_op = true;
  uint32_t event_id = 0;
  rtEvent_t rt_event = reinterpret_cast<void *>(0x05000);
  alloc_handle_context.value_holder[0].Set(const_cast<char *>(ext_data.c_str()), nullptr);
  alloc_handle_context.value_holder[1].Set(reinterpret_cast<void *>(ext_len), nullptr);
  alloc_handle_context.value_holder[2].Set(const_cast<char *>(node_name.c_str()), nullptr);
  alloc_handle_context.value_holder[3].Set(reinterpret_cast<void *>(input_num), nullptr);
  alloc_handle_context.value_holder[4].Set(reinterpret_cast<void *>(output_num), nullptr);
  alloc_handle_context.value_holder[5].Set(reinterpret_cast<void *>(unknown_shape_type), nullptr);
  alloc_handle_context.value_holder[6].Set(host_args_addr.get(), nullptr);
  alloc_handle_context.value_holder[7].Set(device_args_addr.get(), nullptr);
  alloc_handle_context.value_holder[8].Set(reinterpret_cast<void *>(session_id), nullptr);
  alloc_handle_context.value_holder[9].Set(reinterpret_cast<void *>(is_block_op), nullptr);
  alloc_handle_context.value_holder[10].Set(reinterpret_cast<void *>(event_id), nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("BuildExtInfoHandle")->outputs_creator(nullptr, alloc_handle_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("BuildExtInfoHandle")->run_func(alloc_handle_context), ge::GRAPH_SUCCESS);

  auto ext_handle = alloc_handle_context.GetContext<KernelContext>()->GetOutputPointer<AicpuExtInfoHandler>(0U);
  ASSERT_NE(ext_handle, nullptr);

  auto tf_run_context = BuildKernelRunContext(13, 1);
  static std::shared_ptr<STR_FWK_OP_KERNEL> str_fwkop_kernel_ptr;
  str_fwkop_kernel_ptr = std::make_shared<STR_FWK_OP_KERNEL>();
  str_fwkop_kernel_ptr->fwkKernelType = 0;// FMK_KERNEL_TYPE_TF;
  aicpu::FWKAdapter::FWKOperateParam *str_tf_kernel = &(str_fwkop_kernel_ptr->fwkKernelBase.fwk_kernel);
  str_tf_kernel->opType = aicpu::FWKAdapter::FWK_ADPT_KERNEL_RUN;
  str_tf_kernel->sessionID = 2;
  str_tf_kernel->kernelID = 1;
  auto tf_args_len = sizeof(STR_FWK_OP_KERNEL) + 500;
  auto tf_args_addr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[tf_args_len]);
  size_t arg_size = sizeof(STR_FWK_OP_KERNEL);
  std::string task_info = "task_info";
  size_t task_size = task_info.size();
  auto ext_size = 20;
  auto io_num = 1;
  std::string op_name = "temp";
  bool need_device_ext = false;
  int64_t step_id = 0U;
  uint32_t async_timeout = 0;

  tf_run_context.value_holder[0].Set(reinterpret_cast<void *>(io_num), nullptr);
  tf_run_context.value_holder[1].Set(op_name.data(), nullptr);
  tf_run_context.value_holder[2].Set(reinterpret_cast<void *>(need_device_ext), nullptr);
  tf_run_context.value_holder[3].Set(str_fwkop_kernel_ptr.get(), nullptr);
  tf_run_context.value_holder[4].Set(reinterpret_cast<void *>(arg_size), nullptr);
  tf_run_context.value_holder[5].Set(reinterpret_cast<void *>(ext_size), nullptr);
  tf_run_context.value_holder[6].Set(reinterpret_cast<void *>(is_block_op), nullptr);
  tf_run_context.value_holder[7].Set(rt_event, nullptr);
  tf_run_context.value_holder[8].Set(reinterpret_cast<void *>(async_timeout), nullptr);
  tf_run_context.value_holder[9].Set(task_info.data(), nullptr);
  tf_run_context.value_holder[10].Set(reinterpret_cast<void *>(task_size), nullptr);
  tf_run_context.value_holder[11].Set(reinterpret_cast<void *>(session_id), nullptr);
  tf_run_context.value_holder[12].Set(reinterpret_cast<void *>(step_id), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("BuildTfArgs")->outputs_creator(nullptr, tf_run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("BuildTfArgs")->run_func(tf_run_context), ge::GRAPH_SUCCESS);

  auto cc_run_context = BuildKernelRunContext(13, 1);
  std::string so_name = "libcpu_kernels.so";
  std::string kernel_name = "RunCpuKernel";
  const auto kernel_name_size = kernel_name.size();
  const auto so_name_size = so_name.size();

  std::string args(100, '0');
  AicpuTaskStruct aicpu_args;
  aicpu_args.head.length = args.size();
  aicpu_args.head.ioAddrNum = 1;
  ASSERT_EQ(memcpy_s(args.data(), args.size(), &aicpu_args, sizeof(AicpuTaskStruct)), EOK);

  cc_run_context.value_holder[0].Set(reinterpret_cast<void *>(io_num), nullptr);
  cc_run_context.value_holder[1].Set(op_name.data(), nullptr);
  cc_run_context.value_holder[2].Set(reinterpret_cast<void *>(need_device_ext), nullptr);
  cc_run_context.value_holder[3].Set(args.data(), nullptr);
  cc_run_context.value_holder[4].Set(reinterpret_cast<void *>(args.size()), nullptr);
  cc_run_context.value_holder[5].Set(reinterpret_cast<void *>(ext_size), nullptr);
  cc_run_context.value_holder[6].Set(reinterpret_cast<void *>(is_block_op), nullptr);
  cc_run_context.value_holder[7].Set(rt_event, nullptr);
  tf_run_context.value_holder[8].Set(reinterpret_cast<void *>(async_timeout), nullptr);
  cc_run_context.value_holder[9].Set(const_cast<char *>(kernel_name.c_str()), nullptr);
  cc_run_context.value_holder[10].Set(reinterpret_cast<void *>(kernel_name_size), nullptr);
  cc_run_context.value_holder[11].Set(const_cast<char *>(so_name.c_str()), nullptr);
  cc_run_context.value_holder[12].Set(reinterpret_cast<void *>(so_name_size), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("BuildCCArgs")->outputs_creator(nullptr, cc_run_context),  ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("BuildCCArgs")->run_func(cc_run_context), ge::GRAPH_SUCCESS);

  auto host_run_context = BuildKernelRunContext(6, 1);
  host_run_context.value_holder[0].Set(reinterpret_cast<void *>(io_num), nullptr);
  host_run_context.value_holder[1].Set(op_name.data(), nullptr);
  host_run_context.value_holder[2].Set(reinterpret_cast<void *>(need_device_ext), nullptr);
  host_run_context.value_holder[3].Set(args.data(), nullptr);
  host_run_context.value_holder[4].Set(reinterpret_cast<void *>(args.size()), nullptr);
  host_run_context.value_holder[5].Set(reinterpret_cast<void *>(ext_size), nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("BuildHostCCArgs")->outputs_creator(nullptr, host_run_context),  ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("BuildHostCCArgs")->run_func(host_run_context), ge::GRAPH_SUCCESS);
}

TEST_F(AicpuKernelLaunchUT, test_run_launch_cc_kernel_with_block) {
  rtFuncHandle bin_handle = nullptr;
  AicpuCCArgsHandler args_handler("temp", 1, false);
  std::string arg_data = "111";
  std::string so_name = "libcpu_kernels.so";
  std::string kernel_name = "RunCpuKernel";
  size_t ext_info_size = 200;
  args_handler.BuildCCArgs(arg_data , kernel_name, so_name, ext_info_size);
  args_handler.SetBlockOp(true);
  rtEvent_t rt_event = reinterpret_cast<void *>(0x01);
  args_handler.SetRtEvent(rt_event);

  rtStream_t stream = nullptr;
  uint32_t kernel_type = 6;

  auto run_context2 = BuildKernelRunContext(7, 0);
  run_context2.value_holder[0].Set(&args_handler, nullptr);
  run_context2.value_holder[1].Set(stream, nullptr);
  run_context2.value_holder[2].Set(reinterpret_cast<void *>(1), nullptr);
  run_context2.value_holder[3].Set(reinterpret_cast<void *>(kernel_type), nullptr);
  run_context2.value_holder[5].Set(&bin_handle, nullptr);
  std::string node_type = "test_aicpu";
  run_context2.value_holder[6].Set(const_cast<char *>(node_type.c_str()), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("AicpuLaunchCCKernel")->run_func(run_context2), ge::GRAPH_SUCCESS);
  auto ret = registry.FindKernelFuncs("AicpuLaunchCCKernel")->trace_printer(run_context2);
  EXPECT_FALSE(ret.empty());
}

TEST_F(AicpuKernelLaunchUT, test_alloc_hostcpu_output_memory) {
  size_t size = 1;
  auto run_context = BuildKernelRunContext(2, 1);
  run_context.value_holder[0].Set((void *)size, nullptr);
  run_context.value_holder[1].Set((void *)(&host_gert_mem_allocator_), nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("AllocHostCpuOutputMemory")->outputs_creator(nullptr, run_context),  ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("AllocHostCpuOutputMemory")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(AicpuKernelLaunchUT, test_launch_cust_aicpu_kernel) {
  auto run_context = BuildKernelRunContext(2, 1);
  std::string key = "cust_aicpu";
  const char cust_bin[] = "aicpu_cust_bin";
  std::vector<char> buffer(cust_bin, cust_bin + strlen(cust_bin));
  CustAICPUKernelPtr aicpu_kernel_ptr = std::make_shared<ge::OpKernelBin>(key, move(buffer));
  std::string cust_so_name = "libcust_aicpu_kernels.so";
  run_context.value_holder[0].Set(aicpu_kernel_ptr.get(), nullptr);
  run_context.value_holder[1].Set(cust_so_name.data(), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("LaunchAicpuCustKernel")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(AicpuKernelLaunchUT, test_creat_event) {
  auto tmp_graph = std::make_shared<ge::ComputeGraph>("tmp_graph");
  auto tmp_op_desc = std::make_shared<ge::OpDesc>("tmp_opdesc", "Dequeue");
  ASSERT_NE(tmp_graph, nullptr);
  ASSERT_NE(tmp_op_desc, nullptr);
  auto node = tmp_graph->AddNode(tmp_op_desc);
  ASSERT_NE(node, nullptr);
  auto run_context = BuildKernelRunContext(0, 2);
  ASSERT_EQ(registry.FindKernelFuncs("CreateEvent")->run_func(run_context), ge::GRAPH_SUCCESS);

  auto event = run_context.GetContext<KernelContext>()->GetOutputPointer<rtEvent_t>(0U);
  (void)run_context.GetContext<KernelContext>()->GetOutputPointer<uint32_t>(1U);
  ASSERT_NE(event, nullptr);
  auto run_context2 = BuildKernelRunContext(1, 0);
  run_context2.value_holder[0].Set(*event, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("DestroyEvent")->run_func(run_context2), ge::GRAPH_SUCCESS);
}

TEST_F(AicpuKernelLaunchUT, test_run_new_launch_cc_kernel) {
  rtFuncHandle tmp_function_handle = nullptr;
  rtFuncHandle bin_handle = &tmp_function_handle;
  AicpuCCArgsHandler args_handler("temp", 1, false);
  std::string arg_data = "111";
  std::string so_name = "libcpu_kernels.so";
  std::string kernel_name = "RunCpuKernel";
  size_t ext_info_size = 200;
  args_handler.BuildCCArgs(arg_data , kernel_name, so_name, ext_info_size);
  args_handler.SetBlockOp(true);
  rtEvent_t rt_event = reinterpret_cast<void *>(0x01);
  args_handler.SetRtEvent(rt_event);

  rtStream_t stream = nullptr;
  uint32_t kernel_type = 6;

  auto run_context2 = BuildKernelRunContext(7, 0);
  run_context2.value_holder[0].Set(&args_handler, nullptr);
  run_context2.value_holder[1].Set(stream, nullptr);
  run_context2.value_holder[2].Set(reinterpret_cast<void *>(1), nullptr);
  run_context2.value_holder[3].Set(reinterpret_cast<void *>(kernel_type), nullptr);
  run_context2.value_holder[5].Set(&bin_handle, nullptr);
  std::string node_type = "test_aicpu";
  run_context2.value_holder[6].Set(const_cast<char *>(node_type.c_str()), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("AicpuLaunchCCKernel")->run_func(run_context2), ge::GRAPH_SUCCESS);
  auto ret = registry.FindKernelFuncs("AicpuLaunchCCKernel")->trace_printer(run_context2);
  EXPECT_FALSE(ret.empty());
}

TEST_F(AicpuKernelLaunchUT, test_run_new_launch_tf_kernel) {
  auto run_context = BuildKernelRunContext(2, 0);
  AicpuTfArgsHandler args_handler("temp", 1, false);
  std::string arg_data = "111";
  std::string task_info = "222";
  size_t ext_info_size = 20;
  args_handler.SetAsyncTimeout(0);
  args_handler.SetBlockOp(true);
  rtEvent_t rtEvent = reinterpret_cast<void *>(1);
  args_handler.SetRtEvent(rtEvent);
  args_handler.BuildTfArgs(arg_data , task_info, ext_info_size, 0, 0);
  run_context.value_holder[0].Set(&args_handler, nullptr);

  auto temp_addr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[20]);
  GertTensorData temp_buffer = {temp_addr.get(), 20, kOnHost, -1};
  run_context.value_holder[1].Set(&temp_buffer, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("UpdateAicpuIoAddr")->run_func(run_context), ge::GRAPH_SUCCESS);
  uint64_t *io_addrs = reinterpret_cast<uint64_t *>(args_handler.GetIoAddr());
  EXPECT_EQ(io_addrs[0], 0);
  temp_buffer.SetPlacement(kTensorPlacementEnd);
  ASSERT_EQ(registry.FindKernelFuncs("UpdateAicpuIoAddr")->run_func(run_context), ge::GRAPH_SUCCESS);
  EXPECT_NE(io_addrs[0], 0);

  auto run_context2 = BuildKernelRunContext(4, 0);
  run_context2.value_holder[0].Set(&args_handler, nullptr);
  rtStream_t stream = nullptr;
  run_context2.value_holder[1].Set(stream, nullptr);
  rtFuncHandle tmp_function_handle = nullptr;
  rtFuncHandle bin_handle = &tmp_function_handle;
  run_context2.value_holder[2].Set(&bin_handle, nullptr);
  std::string node_type = "test_aicpu";
  run_context2.value_holder[3].Set(const_cast<char *>(node_type.c_str()), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("AicpuLaunchTfKernel")->run_func(run_context2), ge::GRAPH_SUCCESS);

  auto ret = registry.FindKernelFuncs("AicpuLaunchTfKernel")->trace_printer(run_context2);
  EXPECT_FALSE(ret.empty());
}

TEST_F(AicpuKernelLaunchUT, test_build_cc_function_handle) {
  std::string node_type = "test_cc_aicpu";
  auto run_context = BuildKernelRunContext(2, 1);
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>(node_type, node_type);
  (void)ge::AttrUtils::SetBool(op_desc, "_cust_aicpu_flag", false);
  (void)ge::AttrUtils::SetStr(op_desc, "ops_json_path", "/test.json");
  run_context.value_holder[0].Set(op_desc.get(), nullptr);
  run_context.value_holder[1].Set(const_cast<char *>(node_type.c_str()), nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("BuildCCArgsBinHandle")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(AicpuKernelLaunchUT, test_build_cust_cc_function_handle) {
  std::string node_type = "test_cc_aicpu";
  vector<char> buffer(1, 'a');
  ge::OpKernelBinPtr bin = std::make_shared<ge::OpKernelBin>("op", std::move(buffer));
  auto run_context = BuildKernelRunContext(2, 1);
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>(node_type, node_type);
  (void)ge::AttrUtils::SetBool(op_desc, "_cust_aicpu_flag", true);
  (void)ge::AttrUtils::SetStr(op_desc, "funcName", "RunCpuKernel");
  (void)ge::AttrUtils::SetStr(op_desc, "kernelSo", "libtest_cust.so");
  (void)ge::AttrUtils::SetStr(op_desc, "kernelSo", "libtest_cust.so");
  op_desc->SetExtAttr(ge::OP_EXTATTR_CUSTAICPU_KERNEL, bin);
  run_context.value_holder[0].Set(op_desc.get(), nullptr);
  run_context.value_holder[1].Set(const_cast<char *>(node_type.c_str()), nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("BuildCCArgsBinHandle")->run_func(run_context), ge::GRAPH_SUCCESS);
}
}  // namespace gert
