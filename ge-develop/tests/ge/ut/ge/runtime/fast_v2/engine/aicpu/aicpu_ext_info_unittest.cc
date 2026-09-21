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
#include <kernel/memory/mem_block.h>
#include "stub/gert_runtime_stub.h"
#include "faker/kernel_run_context_facker.h"
#include "faker/aicpu_ext_info_faker.h"
#include "register/kernel_registry.h"
#include "aicpu_engine_struct.h"
#include "engine/aicpu/kernel/aicpu_ext_info_handle.h"
#include "graph/op_desc.h"
#include "graph/def_types.h"
#include "graph/compute_graph.h"
#include "subscriber/dumper/executor_dumper.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "core/utils/tensor_utils.h"

namespace gert {
class AicpuExtInfoUT : public testing::Test {
 public:
  KernelRegistry &registry = KernelRegistry::GetInstance();
};

TEST_F(AicpuExtInfoUT, test_parse_and_update) {
  auto alloc_handle_context = BuildKernelRunContext(9, 1);
  auto ext_data = GetFakeExtInfo();
  auto ext_len = ext_data.size();
  std::string node_name = "Add";
  auto input_num = 2U;
  auto output_num = 1U;
  auto unknown_shape_type = ge::DEPEND_COMPUTE;
  auto args_len = 500;
  auto host_args_addr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[args_len]);
  auto device_args_addr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[args_len]);
  uint64_t session_id = 0U;
  alloc_handle_context.value_holder[0].Set(const_cast<char *>(ext_data.c_str()), nullptr);
  alloc_handle_context.value_holder[1].Set(reinterpret_cast<void *>(ext_len), nullptr);
  alloc_handle_context.value_holder[2].Set(const_cast<char *>(node_name.c_str()), nullptr);
  alloc_handle_context.value_holder[3].Set(reinterpret_cast<void *>(input_num), nullptr);
  alloc_handle_context.value_holder[4].Set(reinterpret_cast<void *>(output_num), nullptr);
  alloc_handle_context.value_holder[5].Set(reinterpret_cast<void *>(unknown_shape_type), nullptr);
  alloc_handle_context.value_holder[6].Set(host_args_addr.get(), nullptr);
  alloc_handle_context.value_holder[7].Set(device_args_addr.get(), nullptr);
  alloc_handle_context.value_holder[8].Set(reinterpret_cast<void *>(session_id), nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("BuildExtInfoHandle")->outputs_creator(nullptr, alloc_handle_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("BuildExtInfoHandle")->run_func(alloc_handle_context), ge::GRAPH_SUCCESS);

  auto ext_handle = alloc_handle_context.GetContext<KernelContext>()->GetOutputPointer<AicpuExtInfoHandler>(0U);
  ASSERT_NE(ext_handle, nullptr);

  StorageShape shape({1}, {1});
  auto update_shape_context = BuildKernelRunContext(7, 1);
  rtStream_t stream = nullptr;
  update_shape_context.value_holder[0].Set(ext_handle, nullptr);
  update_shape_context.value_holder[1].Set(reinterpret_cast<void *>(input_num), nullptr);
  update_shape_context.value_holder[2].Set(reinterpret_cast<void *>(output_num), nullptr);
  update_shape_context.value_holder[3].Set(stream, nullptr);
  update_shape_context.value_holder[4].Set(&shape, nullptr);
  update_shape_context.value_holder[5].Set(&shape, nullptr);
  update_shape_context.value_holder[6].Set(&shape, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("UpdateExtShape")->run_func(update_shape_context), ge::GRAPH_SUCCESS);
  auto ret = registry.FindKernelFuncs("UpdateExtShape")->trace_printer(update_shape_context);
  EXPECT_FALSE(ret.empty());

  ASSERT_EQ(ext_handle->UpdateSessionInfoId(1), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ext_handle->UpdateSessionInfo(1, 2, false), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ext_handle->UpdateExecuteMode(0), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ext_handle->UpdateExecuteMode(1), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ext_handle->UpdateEventId(0), ge::GRAPH_SUCCESS);
  Shape shape0;
  ge::DataType datatype;
  ASSERT_EQ(ext_handle->GetOutputShapeAndType(0, shape0, datatype), ge::GRAPH_SUCCESS);

  auto get_ext_output_shapes_ctx = KernelRunContextFaker().KernelIONum(3, 1).NodeIoNum(2, 1).IrInputNum(2).Build();
  std::string engine_name = "aicpu_tf_kernel";
  get_ext_output_shapes_ctx.value_holder[0].Set(ext_handle, nullptr);
  get_ext_output_shapes_ctx.value_holder[1].Set(const_cast<char_t *>(engine_name.c_str()), nullptr);
  get_ext_output_shapes_ctx.value_holder[2].Set(reinterpret_cast<void *>(output_num), nullptr);
  StorageShape storage_shape;
  get_ext_output_shapes_ctx.value_holder[3].Set(&storage_shape, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("GetExtOutputShapes")->outputs_creator(nullptr, get_ext_output_shapes_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("GetExtOutputShapes")->run_func(get_ext_output_shapes_ctx), ge::GRAPH_SUCCESS);
  ret = registry.FindKernelFuncs("GetExtOutputShapes")->trace_printer(get_ext_output_shapes_ctx);
  EXPECT_FALSE(ret.empty());
  alloc_handle_context.FreeValue(8);
}

TEST_F(AicpuExtInfoUT, test_parse_and_update_worksapce_info) {
  auto alloc_handle_context = BuildKernelRunContext(9, 1);
  auto ext_data = GetFakeExtInfo();
  auto ext_len = ext_data.size();
  std::string node_name = "Add";
  auto input_num = 2U;
  auto output_num = 1U;
  auto unknown_shape_type = ge::DEPEND_COMPUTE;
  auto args_len = 500;
  auto host_args_addr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[args_len]);
  auto device_args_addr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[args_len]);
  uint64_t session_id = 0U;
  alloc_handle_context.value_holder[0].Set(const_cast<char *>(ext_data.c_str()), nullptr);
  alloc_handle_context.value_holder[1].Set(reinterpret_cast<void *>(ext_len), nullptr);
  alloc_handle_context.value_holder[2].Set(const_cast<char *>(node_name.c_str()), nullptr);
  alloc_handle_context.value_holder[3].Set(reinterpret_cast<void *>(input_num), nullptr);
  alloc_handle_context.value_holder[4].Set(reinterpret_cast<void *>(output_num), nullptr);
  alloc_handle_context.value_holder[5].Set(reinterpret_cast<void *>(unknown_shape_type), nullptr);
  alloc_handle_context.value_holder[6].Set(host_args_addr.get(), nullptr);
  alloc_handle_context.value_holder[7].Set(device_args_addr.get(), nullptr);
  alloc_handle_context.value_holder[8].Set(reinterpret_cast<void *>(session_id), nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("BuildExtInfoHandle")->outputs_creator(nullptr, alloc_handle_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("BuildExtInfoHandle")->run_func(alloc_handle_context), ge::GRAPH_SUCCESS);

  auto ext_handle = alloc_handle_context.GetContext<KernelContext>()->GetOutputPointer<AicpuExtInfoHandler>(0U);
  ASSERT_NE(ext_handle, nullptr);

  auto update_workspace_info_ctx = BuildKernelRunContext(3, 1);
  uint64_t workspace_size = 1024UL;
  auto temp_addr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[1024]);
  GertTensorData temp_buffer = {temp_addr.get(), 1024, kOnDeviceHbm, -1};
  update_workspace_info_ctx.value_holder[0].Set(reinterpret_cast<void *>(workspace_size), nullptr);
  update_workspace_info_ctx.value_holder[1].Set(&temp_buffer, nullptr);
  update_workspace_info_ctx.value_holder[2].Set(ext_handle, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("UpdateExtWorkSpaceInfo")->run_func(update_workspace_info_ctx), ge::GRAPH_SUCCESS);
  alloc_handle_context.FreeValue(8);
}

TEST_F(AicpuExtInfoUT, update_and_get_workspaceInfo_nullptr) {
  const std::string node_name = "test";
  uint32_t input_num = 1U;
  uint32_t output_num = 1U;
  ge::UnknowShapeOpType unknown_type = ge::DEPEND_IN_SHAPE;
  AicpuExtInfoHandler ext_handle(node_name, input_num, output_num, unknown_type);
  uint64_t workspace_size = 1024UL;
  uint64_t workspace_addr = 1024UL;
  ext_handle.UpdateWorkSpaceInfo(workspace_size, workspace_addr);
  auto workspace_info = ext_handle.GetWorkSpaceInfo();
  ASSERT_EQ(workspace_info, nullptr);
}

TEST_F(AicpuExtInfoUT, test_cpy_host_cpu_output_shape) {
  auto alloc_handle_context = BuildKernelRunContext(9, 1);
  auto ext_data = GetFakeExtInfo();
  auto ext_len = ext_data.size();
  std::string node_name = "Add";
  auto input_num = 2U;
  auto output_num = 1U;
  auto unknown_shape_type = ge::DEPEND_SHAPE_RANGE;
  auto args_len = 500;
  auto host_args_addr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[args_len]);
  auto device_args_addr = nullptr;
  uint64_t session_id = 0U;
  alloc_handle_context.value_holder[0].Set(const_cast<char *>(ext_data.c_str()), nullptr);
  alloc_handle_context.value_holder[1].Set(reinterpret_cast<void *>(ext_len), nullptr);
  alloc_handle_context.value_holder[2].Set(const_cast<char *>(node_name.c_str()), nullptr);
  alloc_handle_context.value_holder[3].Set(reinterpret_cast<void *>(input_num), nullptr);
  alloc_handle_context.value_holder[4].Set(reinterpret_cast<void *>(output_num), nullptr);
  alloc_handle_context.value_holder[5].Set(reinterpret_cast<void *>(unknown_shape_type), nullptr);
  alloc_handle_context.value_holder[6].Set(host_args_addr.get(), nullptr);
  alloc_handle_context.value_holder[7].Set(device_args_addr, nullptr);
  alloc_handle_context.value_holder[8].Set(reinterpret_cast<void *>(session_id), nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("BuildExtInfoHandle")->outputs_creator(nullptr, alloc_handle_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("BuildExtInfoHandle")->run_func(alloc_handle_context), ge::GRAPH_SUCCESS);

  auto ext_handle = alloc_handle_context.GetContext<KernelContext>()->GetOutputPointer<AicpuExtInfoHandler>(0U);
  ASSERT_NE(ext_handle, nullptr);

  auto get_ext_output_shapes_ctx = KernelRunContextFaker().KernelIONum(3, 1).NodeIoNum(2, 1).IrInputNum(2).Build();
  get_ext_output_shapes_ctx.value_holder[0].Set(ext_handle, nullptr);
  std::string engine_name = "DNN_VM_HOST_CPU_OP_STORE";
  get_ext_output_shapes_ctx.value_holder[1].Set(const_cast<char_t *>(engine_name.c_str()), nullptr);
  get_ext_output_shapes_ctx.value_holder[2].Set(reinterpret_cast<void *>(output_num), nullptr);
  StorageShape storage_shape;
  get_ext_output_shapes_ctx.value_holder[3].Set(&storage_shape, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("GetExtOutputShapes")->outputs_creator(nullptr, get_ext_output_shapes_ctx), ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("GetExtOutputShapes")->run_func(get_ext_output_shapes_ctx), ge::GRAPH_SUCCESS);
  alloc_handle_context.FreeValue(8);
}

TEST_F(AicpuExtInfoUT, test_FillWorkSpace) {
  auto alloc_handle_context = BuildKernelRunContext(9, 1);
  auto ext_data = GetFakeExtInfo();
  auto ext_len = ext_data.size();
  std::string node_name = "Add";
  auto input_num = 2U;
  auto output_num = 1U;
  auto unknown_shape_type = ge::DEPEND_COMPUTE;
  auto args_len = 500;
  auto host_args_addr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[args_len]);
  auto device_args_addr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[args_len]);
  uint64_t session_id = 0U;
  alloc_handle_context.value_holder[0].Set(const_cast<char *>(ext_data.c_str()), nullptr);
  alloc_handle_context.value_holder[1].Set(reinterpret_cast<void *>(ext_len), nullptr);
  alloc_handle_context.value_holder[2].Set(const_cast<char *>(node_name.c_str()), nullptr);
  alloc_handle_context.value_holder[3].Set(reinterpret_cast<void *>(input_num), nullptr);
  alloc_handle_context.value_holder[4].Set(reinterpret_cast<void *>(output_num), nullptr);
  alloc_handle_context.value_holder[5].Set(reinterpret_cast<void *>(unknown_shape_type), nullptr);
  alloc_handle_context.value_holder[6].Set(host_args_addr.get(), nullptr);
  alloc_handle_context.value_holder[7].Set(device_args_addr.get(), nullptr);
  alloc_handle_context.value_holder[8].Set(reinterpret_cast<void *>(session_id), nullptr);
  auto ext_handle = alloc_handle_context.GetContext<KernelContext>()->GetOutputPointer<AicpuExtInfoHandler>(0U);

  auto update_workspace_info_ctx = BuildKernelRunContext(3, 1);
  uint64_t workspace_size = 1024UL;

  auto temp_addr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[1024]);
  GertTensorData temp_buffer = {temp_addr.get(), 1024, kOnDeviceHbm, -1};
  update_workspace_info_ctx.value_holder[0].Set(reinterpret_cast<void *>(workspace_size), nullptr);
  update_workspace_info_ctx.value_holder[2].Set(ext_handle, nullptr);

  NodeDumpUnit dump_unit;
  gert::ExecutorDataDumpInfoWrapper wrapper(&dump_unit);
  ASSERT_NE(registry.FindKernelFuncs("UpdateExtWorkSpaceInfo")->data_dump_info_filler(update_workspace_info_ctx, wrapper), ge::GRAPH_SUCCESS);

  ExceptionDumpUint exception_dump_unit;
  ExecutorExceptionDumpInfoWrapper exception_wrapper(&exception_dump_unit);
  ASSERT_NE(registry.FindKernelFuncs("UpdateExtWorkSpaceInfo")->exception_dump_info_filler(update_workspace_info_ctx, exception_wrapper), ge::GRAPH_SUCCESS);

  update_workspace_info_ctx.value_holder[1].Set(&temp_buffer, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("UpdateExtWorkSpaceInfo")->data_dump_info_filler(update_workspace_info_ctx, wrapper), ge::GRAPH_SUCCESS);

  ASSERT_EQ(registry.FindKernelFuncs("UpdateExtWorkSpaceInfo")->exception_dump_info_filler(update_workspace_info_ctx, exception_wrapper), ge::GRAPH_SUCCESS);
}

}  // namespace gert
