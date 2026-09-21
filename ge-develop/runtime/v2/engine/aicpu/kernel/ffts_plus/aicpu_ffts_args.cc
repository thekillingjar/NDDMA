/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_ffts_args.h"
#include <cstddef>
#include <iomanip>
#include <cinttypes>
#include "register/kernel_registry.h"
#include "register/ffts_node_calculater_registry.h"
#include "kernel/memory/ffts_mem_allocator.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "framework/common/debug/log.h"
#include "exe_graph/runtime/tensor.h"
#include "common/dump/kernel_tracing_utils.h"
#include "runtime/mem.h"
#include "common/checker.h"
#include "common/math/math_util.h"
#include "engine/aicpu/kernel/aicpu_ext_info_handle.h"
#include "aicpu_task_struct.h"
#include "engine/aicpu/kernel/ffts_plus/aicpu_update_kernel.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"

namespace {
  constexpr int32_t kSoNameAddrLIndex = 0;
  constexpr int32_t kSoNameAddrHIndex = 1;
  constexpr int32_t kKernelNameAddrLIndex = 4;
  constexpr int32_t kKernelNameAddrHIndex = 5;
  constexpr uint32_t k32BitsMask = 0xFFFFFFFFU;
}

namespace gert {
namespace kernel {
ge::graphStatus CalFlushDataBasicParam(const size_t ext_len, const size_t arg_len, AICpuSubTaskFlush *flush_data,
                                       const NodeMemPara *args_addr, const size_t arg_size) {
  GE_ASSERT_NOTNULL(args_addr->dev_addr);
  GE_ASSERT_NOTNULL(args_addr->host_addr);
  flush_data->task_base_addr_dev = args_addr->dev_addr;
  flush_data->args_base_addr_dev = ge::ValueToPtr(ge::PtrToValue(flush_data->task_base_addr_dev) + ext_len);
  flush_data->task_base_addr_host = args_addr->host_addr;
  flush_data->args_base_addr_host = ge::ValueToPtr(ge::PtrToValue(flush_data->task_base_addr_host) + ext_len);
  flush_data->task_size = ext_len + arg_len;
  flush_data->arg_size = arg_size;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AicpuUpdateExtInfo(gert::KernelContext *context, const AICpuThreadParam *thread_param,
                                   AicpuExtInfoHandler &ext_handler, const bool last_flag, uint64_t &kernel_id) {
  auto session_id = context->GetInputPointer<uint64_t>(static_cast<size_t>(FFTSAicpuArgss::kSessionId));
  auto not_last_input_shapes =
      context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(FFTSAicpuArgss::kNotLastInSlice));
  auto last_input_shapes =
      context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(FFTSAicpuArgss::kLastInSlice));
  auto not_last_output_shapes =
      context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(FFTSAicpuArgss::kNotLastOutSlice));
  auto last_output_shapes =
      context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(FFTSAicpuArgss::kLastOutSlice));
  GE_ASSERT_NOTNULL(session_id);
  GE_ASSERT_NOTNULL(not_last_input_shapes);
  GE_ASSERT_NOTNULL(last_input_shapes);
  GE_ASSERT_NOTNULL(not_last_output_shapes);
  GE_ASSERT_NOTNULL(last_output_shapes);

  size_t input_num = thread_param->input_num;
  size_t output_num = thread_param->output_num;

  GE_ASSERT_EQ(not_last_input_shapes->GetSize(), last_input_shapes->GetSize());
  GE_ASSERT_EQ(not_last_output_shapes->GetSize(), last_output_shapes->GetSize());
  if (input_num > not_last_input_shapes->GetSize() || output_num > not_last_output_shapes->GetSize()) {
    GELOGE(ge::FAILED, "Invalid thread dims, input[num:%zu, dims:%zu], output[num:%zu, dims:%zu]", input_num,
           not_last_input_shapes->GetSize(), output_num, not_last_output_shapes->GetSize());
    return ge::GRAPH_FAILED;
  }

  kernel_id = ::ge::hybrid::AicpuExtInfoHandler::GenerateKernelId();
  GE_CHK_STATUS_RET_NOLOG(ext_handler.UpdateSessionInfo(*session_id, kernel_id, false));
  GE_CHK_STATUS_RET_NOLOG(ext_handler.UpdateExecuteMode(false));

  auto last_input_vec = reinterpret_cast<const Shape*>(last_input_shapes->GetData());
  auto not_last_input_vec = reinterpret_cast<const Shape*>(not_last_input_shapes->GetData());
  auto input_shapes = last_flag ? last_input_vec : not_last_input_vec;
  for (size_t i = 0; i < input_num; ++i) {
    GE_CHK_STATUS_RET_NOLOG(ext_handler.UpdateInputShape(i, input_shapes[i]));
  }
  auto last_output_vec = reinterpret_cast<const Shape*>(last_output_shapes->GetData());
  auto not_last_output_vec = reinterpret_cast<const Shape*>(not_last_output_shapes->GetData());
  auto output_shapes = last_flag ? last_output_vec : not_last_output_vec;
  for (size_t i = 0; i < output_num; ++i) {
    GE_CHK_STATUS_RET_NOLOG(ext_handler.UpdateOutputShape(i, output_shapes[i]));
  }

  return ge::GRAPH_SUCCESS;
}

/*
 * memory addr format:
 * extinfo_to| exitinfo_t1 | 32BytePadding | Args_t0 | Args_t1
 *                                         ^
 *                                         |args_addr_base
 */
void InitAicpuFftsIoAddrs(const AICpuThreadParam *thread_para, memory::FftsMemBlock *ffts_mem,
                          AICpuSubTaskFlush *flush_data, const size_t ctx_idx, const size_t task_addr_offset) {
  size_t thread_dim = static_cast<size_t>(thread_para->thread_dim);
  uint8_t *const arg_data_base =
      ge::PtrToPtr<void, uint8_t>(flush_data->args_base_addr_host);
  for (size_t i = 0UL; i < thread_dim; ++i) {
    const size_t ctx_io_pos = (i * flush_data->arg_size) + task_addr_offset;
    GELOGD("thread_dim: %zu, ctx_idx: %zu, thread index: %zu, ctx_io_pos: %zu", thread_dim, ctx_idx, i, ctx_io_pos);
    *ge::PtrToPtr<uint8_t, uint64_t>(arg_data_base + ctx_io_pos) = ge::PtrToValue(ffts_mem->Addr(i));
  }
}

void InitAicpuFftsIoAddrs(const AICpuThreadParam *thread_para, const void *addr, AICpuSubTaskFlush *flush_data,
                          const size_t ctx_idx, const size_t task_addr_offset) {
  const uintptr_t data_base = ge::PtrToValue(addr);
  size_t thread_dim = static_cast<size_t>(thread_para->thread_dim);
  uint8_t *const arg_data_base =
      ge::PtrToPtr<void, uint8_t>(flush_data->args_base_addr_host);
  for (size_t i = 0UL; i < thread_dim; ++i) {
    const size_t ctx_io_pos = (i * flush_data->arg_size) + task_addr_offset;
    GELOGD(
        "thread_dim: %zu, ctx_idx: %zu, thread index: %zu, thread addr offset: 0x%lx, ctx_io_pos: "
        "%zu",
        thread_dim, ctx_idx, i, thread_para->task_addr_offset[ctx_idx], ctx_io_pos);
    *ge::PtrToPtr<uint8_t, uint64_t>(arg_data_base + ctx_io_pos) =
        data_base + (thread_para->task_addr_offset[ctx_idx] * i);
  }
}

ge::graphStatus UpdateAicpuIoAddrs(KernelContext *context, AICpuSubTaskFlush *flush_data,
                                   const AICpuThreadParam *thread_para, const size_t offset) {
  auto in_mem_type = context->GetInputPointer<ContinuousVector>(static_cast<size_t>(FFTSAicpuArgss::kInMemType));
  GE_ASSERT_NOTNULL(in_mem_type);
  auto out_mem_type = context->GetInputPointer<ContinuousVector>(static_cast<size_t>(FFTSAicpuArgss::kOutMemType));
  GE_ASSERT_NOTNULL(out_mem_type);
  GE_ASSERT_TRUE(thread_para->input_output_num <= MAX_OFFSET_NUM,
                 "input_output_num[%zu] can not be large than MAX_OFFSET_NUM[%zu]", thread_para->input_output_num,
                 MAX_OFFSET_NUM);
  if ((in_mem_type->GetSize() != thread_para->input_num) || (out_mem_type->GetSize() != thread_para->output_num)) {
    GELOGE(ge::FAILED, "In/Out mem type size:%zu/%zu is invalid.", in_mem_type->GetSize(), out_mem_type->GetSize());
    return ge::GRAPH_FAILED;
  }
  auto in_mem_type_vec = reinterpret_cast<const uint32_t*>(in_mem_type->GetData());
  auto out_mem_type_vec = reinterpret_cast<const uint32_t*>(out_mem_type->GetData());
  for (size_t i = 0UL; i < thread_para->input_output_num; ++i) {
    const size_t task_addr_offset = offset + (i * sizeof(uintptr_t));
    uint32_t mem_pool_type =
        i < thread_para->input_num ? in_mem_type_vec[i] : out_mem_type_vec[i - thread_para->input_num];
    // When the input mem type is 1, select the secondary memory pool
    if (mem_pool_type == 1U) {
      auto ffts_mem = context->GetInputValue<memory::FftsMemBlock *>(static_cast<size_t>(FFTSAicpuArgss::kIoStart) + i);
      GE_ASSERT_NOTNULL(ffts_mem);
      InitAicpuFftsIoAddrs(thread_para, ffts_mem, flush_data, i, task_addr_offset);
    } else {
      auto tensor = context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(FFTSAicpuArgss::kIoStart) + i);
      GE_ASSERT_NOTNULL(tensor);
      GELOGD("input idx:[%zu], size:[%zu].", i, tensor->GetSize());
      InitAicpuFftsIoAddrs(thread_para, tensor->GetAddr(), flush_data, i, task_addr_offset);
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFTSUpdateAICpuCCArgs(KernelContext *context) {
  GELOGI("begin FFTSUpdateAICpuCCArgs.");
  auto ext_data = context->GetInputValue<char *>(static_cast<size_t>(FFTSAicpuArgss::kExtInfo));
  auto ext_size = context->GetInputValue<size_t>(static_cast<size_t>(FFTSAicpuArgss::kExtLen));
  auto arg_data = context->GetInputValue<uint8_t *>(static_cast<size_t>(FFTSAicpuArgss::kArgInfo));
  auto arg_size = context->GetInputValue<size_t>(static_cast<size_t>(FFTSAicpuArgss::kArgLen));
  auto node_name = context->GetInputValue<char *>(static_cast<size_t>(FFTSAicpuArgss::kNodeName));
  auto unknown_shape_type_val = context->GetInputValue<int32_t>(static_cast<size_t>(FFTSAicpuArgss::kUnknownShapeType));
  auto thread_param = context->GetInputPointer<AICpuThreadParam>(static_cast<size_t>(FFTSAicpuArgss::kThreadParam));
  auto flush_data = context->GetOutputPointer<AICpuSubTaskFlush>(static_cast<size_t>(FFTSAicpuArgsOutKey::kFlushData));
  auto args_addr = context->GetOutputPointer<NodeMemPara>(static_cast<size_t>(FFTSAicpuArgsOutKey::kArgAddr));

  GE_ASSERT_NOTNULL(ext_data);
  GE_ASSERT_NOTNULL(arg_data);
  GE_ASSERT_NOTNULL(args_addr);
  GE_ASSERT_NOTNULL(node_name);
  GE_ASSERT_NOTNULL(thread_param);
  GE_ASSERT_NOTNULL(flush_data);

  auto thread_dim = static_cast<size_t>(thread_param->thread_dim);
  size_t input_num = thread_param->input_num;
  size_t output_num = thread_param->output_num;
  const size_t task_ext_info_len = ge::MemSizeAlign(ext_size * thread_dim);
  const size_t task_arg_data_len = arg_size * thread_dim;
  if (CalFlushDataBasicParam(task_ext_info_len, task_arg_data_len, flush_data, args_addr, arg_size) !=
      ge::GRAPH_SUCCESS) {
    GELOGE(ge::FAILED, "aicpu calculate flush data failed!");
    return ge::GRAPH_FAILED;
  }

  uint8_t *const ext_info_base_host = ge::PtrToPtr<void, uint8_t>(flush_data->task_base_addr_host);
  uint8_t *const ext_info_base_dev = ge::PtrToPtr<void, uint8_t>(flush_data->task_base_addr_dev);
  uint8_t *const arg_data_base =
      ge::PtrToPtr<void, uint8_t>(flush_data->args_base_addr_host);
  GELOGD("[%s] task args size %zu, ext info len %zu, arg data len %zu, thread dim %zu",
         node_name, flush_data->task_size, ext_size, arg_size, thread_dim);

  // ext_handle
  const auto unknown_type = static_cast<ge::UnknowShapeOpType>(unknown_shape_type_val);
  AicpuExtInfoHandler ext_handle(node_name, input_num, output_num, unknown_type);
  auto ext_str = std::string(ext_data, ext_size);

  const uintptr_t ext_info_addr = ge::PtrToValue(flush_data->task_base_addr_dev);
  uint64_t kernel_id = 0;
  for (size_t i = 0UL; i < thread_dim; ++i) {
    const bool last_flag = (i == thread_dim - 1) ? true : false;
    uint8_t *const thrd_ext_info_addr_host = ge::PtrAdd(ext_info_base_host, task_ext_info_len, i * ext_size);
    uint8_t *const thrd_ext_info_addr_dev = ge::PtrAdd(ext_info_base_dev, task_ext_info_len, i * ext_size);
    ext_handle.Parse(ext_str, thrd_ext_info_addr_host, thrd_ext_info_addr_dev);
    GE_CHK_STATUS_RET_NOLOG(AicpuUpdateExtInfo(context, thread_param, ext_handle, last_flag, kernel_id));
    if (memcpy_s(thrd_ext_info_addr_host, ext_size, ext_handle.GetExtInfo(), ext_size) != EOK) {
      GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[%s] Copy AiCpu ext info failed, data size %zu", node_name, ext_size);
      return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
    }

    uint8_t *const thrd_arg_data_addr = ge::PtrAdd(arg_data_base, task_arg_data_len, i * arg_size);
    if (memcpy_s(thrd_arg_data_addr, arg_size, arg_data, arg_size) != EOK) {
      GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[%s] Copy AiCpu arg data failed, data size %zu", node_name, arg_size);
      return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
    }

    auto &cpu_param_head = *ge::PtrToPtr<uint8_t, aicpu::AicpuParamHead>(thrd_arg_data_addr);
    cpu_param_head.extInfoAddr = ext_info_addr + (i * ext_size);  // device memory addr.
    cpu_param_head.extInfoLength = static_cast<uint32_t>(ext_size);
    GELOGI("[%s] Param length %d, ext info addr 0x%lx, ext info len %u", node_name, cpu_param_head.length,
           cpu_param_head.extInfoAddr, cpu_param_head.extInfoLength);
  }
  const size_t offset = sizeof(aicpu::AicpuParamHead);
  GE_ASSERT_GRAPH_SUCCESS(UpdateAicpuIoAddrs(context, flush_data, thread_param, offset));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFTSUpdateAICpuTfArgs(KernelContext *context) {
  auto ext_data = context->GetInputValue<char *>(static_cast<size_t>(FFTSAicpuArgss::kExtInfo));
  GE_ASSERT_NOTNULL(ext_data);
  auto ext_size = context->GetInputValue<size_t>(static_cast<size_t>(FFTSAicpuArgss::kExtLen));
  auto arg_data = context->GetInputValue<uint8_t *>(static_cast<size_t>(FFTSAicpuArgss::kArgInfo));
  GE_ASSERT_NOTNULL(arg_data);
  auto arg_size = context->GetInputValue<size_t>(static_cast<size_t>(FFTSAicpuArgss::kArgLen));
  auto node_name = context->GetInputValue<char *>(static_cast<size_t>(FFTSAicpuArgss::kNodeName));
  GE_ASSERT_NOTNULL(node_name);
  auto unknown_shape_type_val = context->GetInputValue<int32_t>(static_cast<size_t>(FFTSAicpuArgss::kUnknownShapeType));
  auto thread_param = context->GetInputPointer<AICpuThreadParam>(static_cast<size_t>(FFTSAicpuArgss::kThreadParam));
  GE_ASSERT_NOTNULL(thread_param);
  auto session_id = context->GetInputPointer<uint64_t>(static_cast<size_t>(FFTSAicpuArgss::kSessionId));
  GE_ASSERT_NOTNULL(session_id);
  auto step_id =
      context->GetInputValue<void *>((static_cast<size_t>(FFTSAicpuArgss::kIoStart) + thread_param->input_output_num));
  GE_ASSERT_NOTNULL(step_id);
  auto flush_data = context->GetOutputPointer<AICpuSubTaskFlush>(static_cast<size_t>(FFTSAicpuArgsOutKey::kFlushData));
  GE_ASSERT_NOTNULL(flush_data);
  auto args_addr = context->GetOutputPointer<NodeMemPara>(static_cast<size_t>(FFTSAicpuArgsOutKey::kArgAddr));
  GE_ASSERT_NOTNULL(args_addr);
  auto in_mem_type = context->GetInputPointer<ContinuousVector>(static_cast<size_t>(FFTSAicpuArgss::kInMemType));
  GE_ASSERT_NOTNULL(in_mem_type);
  auto out_mem_type = context->GetInputPointer<ContinuousVector>(static_cast<size_t>(FFTSAicpuArgss::kOutMemType));
  GE_ASSERT_NOTNULL(out_mem_type);
  GE_ASSERT_TRUE(thread_param->input_output_num <= MAX_OFFSET_NUM,
                 "input_output_num[%zu] can not be large than MAX_OFFSET_NUM[%zu]", thread_param->input_output_num,
                 MAX_OFFSET_NUM);
  if ((in_mem_type->GetSize() != thread_param->input_num) || (out_mem_type->GetSize() != thread_param->output_num)) {
    GELOGE(ge::FAILED, "In/Out mem type size:%zu/%zu is invalid.", in_mem_type->GetSize(), out_mem_type->GetSize());
    return ge::GRAPH_FAILED;
  }
  auto in_mem_type_vec = reinterpret_cast<const uint32_t *>(in_mem_type->GetData());
  auto out_mem_type_vec = reinterpret_cast<const uint32_t *>(out_mem_type->GetData());

  auto thread_dim = static_cast<size_t>(thread_param->thread_dim);
  const size_t task_ext_info_len = ge::MemSizeAlign(ext_size * thread_dim);
  const size_t task_arg_data_len = arg_size * thread_dim;
  const size_t task_arg_data_align = ge::MemSizeAlign(arg_size * thread_dim);
  const size_t io_total_bytes = thread_param->input_output_num * sizeof(uintptr_t) * thread_dim;
  if (CalFlushDataBasicParam(task_ext_info_len, task_arg_data_len, flush_data, args_addr, arg_size) !=
      ge::GRAPH_SUCCESS) {
    GELOGE(ge::FAILED, "aicpu calculate flush data failed!");
    return ge::GRAPH_FAILED;
  }

  uint8_t *const ext_info_base_host = ge::PtrToPtr<void, uint8_t>(flush_data->task_base_addr_host);
  uint8_t *const ext_info_base_dev = ge::PtrToPtr<void, uint8_t>(flush_data->task_base_addr_dev);
  uint8_t *const arg_data_base = ge::PtrToPtr<void, uint8_t>(flush_data->args_base_addr_host);
  uint8_t *const arg_data_base_dev = ge::PtrToPtr<void, uint8_t>(flush_data->args_base_addr_dev);
  uint8_t *io_base_dev =
      ge::PtrToPtr<void, uint8_t>(ge::ValueToPtr(ge::PtrToValue(flush_data->args_base_addr_dev) + task_arg_data_align));
  uint8_t *io_base_host = ge::PtrToPtr<void, uint8_t>(
      ge::ValueToPtr(ge::PtrToValue(flush_data->args_base_addr_host) + task_arg_data_align));

  // ext_handle
  const auto unknown_type = static_cast<ge::UnknowShapeOpType>(unknown_shape_type_val);
  AicpuExtInfoHandler ext_handle(node_name, thread_param->input_num, thread_param->output_num, unknown_type);
  auto ext_str = std::string(ext_data, ext_size);

  const uintptr_t ext_info_addr = ge::PtrToValue(flush_data->task_base_addr_dev);
  uint64_t kernel_id = 0;
  for (size_t i = 0UL; i < thread_dim; ++i) {
    const bool last_flag = (i == thread_dim - 1) ? true : false;
    uint8_t *const thrd_ext_info_addr_host = ge::PtrAdd(ext_info_base_host, task_ext_info_len, i * ext_size);
    uint8_t *const thrd_ext_info_addr_dev = ge::PtrAdd(ext_info_base_dev, task_ext_info_len, i * ext_size);
    ext_handle.Parse(ext_str, thrd_ext_info_addr_host, thrd_ext_info_addr_dev);
    GE_CHK_STATUS_RET_NOLOG(AicpuUpdateExtInfo(context, thread_param, ext_handle, last_flag, kernel_id));
    if (memcpy_s(thrd_ext_info_addr_host, ext_size, ext_handle.GetExtInfo(), ext_size) != EOK) {
      GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[%s] Copy AiCpu ext info failed, data size %zu", node_name,
             ext_handle.GetExtInfoLen());
      return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
    }

    uint8_t *const thrd_arg_data_addr = ge::PtrAdd(arg_data_base, task_arg_data_len, i * arg_size);
    if (memcpy_s(thrd_arg_data_addr, arg_size, arg_data, arg_size) != EOK) {
      GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[%s] Copy AiCpu arg data failed, data size %zu", node_name, arg_size);
      return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
    }
    uint8_t *const thrd_arg_data_addr_dev = ge::PtrAdd(arg_data_base_dev, task_arg_data_len, i * arg_size);
    auto &fwk_op_kernel = *ge::PtrToPtr<uint8_t, STR_FWK_OP_KERNEL>(thrd_arg_data_addr);
    fwk_op_kernel.fwkKernelBase.fwk_kernel.sessionID = *session_id;
    fwk_op_kernel.fwkKernelBase.fwk_kernel.kernelID = kernel_id;
    fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoLen = static_cast<uint32_t>(ext_size);
    fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoAddr = ext_info_addr + (i * ext_size);  // device memory addr.
    fwk_op_kernel.fwkKernelBase.fwk_kernel.stepIDAddr = ge::PtrToValue(step_id);
    fwk_op_kernel.fwkKernelBase.fwk_kernel.workspaceBaseAddr =
        ge::PtrToValue(thrd_arg_data_addr_dev) + sizeof(STR_FWK_OP_KERNEL);
    for (size_t io_index = 0UL; io_index < thread_param->input_output_num; ++io_index) {
      uint32_t mem_pool_type = io_index < thread_param->input_num
                                   ? in_mem_type_vec[io_index]
                                   : out_mem_type_vec[io_index - thread_param->input_num];
      const size_t ctx_io_pos = io_index * sizeof(uintptr_t) + i * thread_param->input_output_num * sizeof(uintptr_t);
      // When the input mem type is 1, select the secondary memory pool
      if (mem_pool_type == 1U) {
        auto ffts_mem =
            context->GetInputValue<memory::FftsMemBlock *>(static_cast<size_t>(FFTSAicpuArgss::kIoStart) + io_index);
        GE_ASSERT_NOTNULL(ffts_mem);
        *ge::PtrToPtr<uint8_t, uint64_t>(io_base_host + ctx_io_pos) = ge::PtrToValue(ffts_mem->Addr(i));
      } else {
        auto tensor =
            context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(FFTSAicpuArgss::kIoStart) + io_index);
        GE_ASSERT_NOTNULL(tensor);
        const uintptr_t tensor_data_base = ge::PtrToValue(tensor->GetAddr());
        *ge::PtrToPtr<uint8_t, uint64_t>(io_base_host + ctx_io_pos) =
            tensor_data_base + (thread_param->task_addr_offset[io_index] * i);
        GELOGD("input idx:[%zu], size:[%zu].", i, tensor->GetSize());
      }
    }
    uint8_t *const thrd_io_addr_dev =
        ge::PtrAdd(io_base_dev, io_total_bytes, i * thread_param->input_output_num * sizeof(uintptr_t));
    fwk_op_kernel.fwkKernelBase.fwk_kernel.inputOutputAddr = ge::PtrToValue(thrd_io_addr_dev);
    GELOGI("[%s] sessionID[%lu], kernelID[%lu], extInfoLen[%lu], fwkKernelType[%u]", node_name, *session_id,
           fwk_op_kernel.fwkKernelBase.fwk_kernel.kernelID, fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoLen,
           fwk_op_kernel.fwkKernelType);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AiCpuCreateFlushData(const ge::FastNode *node, KernelContext *context) {
  (void) node;
  auto sub_task_flush = new (std::nothrow) AICpuSubTaskFlush();
  GE_ASSERT_NOTNULL(sub_task_flush);
  context->GetOutput(0)->SetWithDefaultDeleter(sub_task_flush);
  return ge::GRAPH_SUCCESS;
}

std::string PrintAIcpuArgsDetail(const KernelContext *context) {
  auto args_addr = context->GetOutputPointer<NodeMemPara>(static_cast<size_t>(FFTSAicpuArgsOutKey::kArgAddr));
  if (args_addr == nullptr) {
    return "Aicpu launch args pointer is null.";
  }
  std::stringstream ss;
  ss << "All args: ";
  auto args_host_addr = ge::PtrToPtr<void, uintptr_t>(args_addr->host_addr);
  PrintHex(args_host_addr, args_addr->size / sizeof(TensorAddress), ss);
  return ss.str();
}

std::vector<std::string> PrintAICpuArgs(const KernelContext *context) {
  std::stringstream ss;
  ss << "FFTS Plus launch AICpu Args: ";
  std::vector<std::string> msgs;
  msgs.emplace_back(ss.str());
  msgs.emplace_back(PrintAIcpuArgsDetail(context));
  return msgs;
}

ge::graphStatus FillDataDumpCtxInfo(const KernelContext *context, gert::DataDumpInfoWrapper &wrapper) {
  GELOGI("Begin FillDataDumpCtxInfo.");
  auto ctx_ids = context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(FFTSAicpuArgss::kCtxIds));
  GE_ASSERT_NOTNULL(ctx_ids);
  auto ctx_id_vec = reinterpret_cast<const int32_t *>(ctx_ids->GetData());
  const size_t ctx_num = ctx_ids->GetSize();

  auto thread_param = context->GetInputPointer<AICpuThreadParam>(static_cast<size_t>(FFTSAicpuArgss::kThreadParam));
  GE_ASSERT_NOTNULL(thread_param);
  size_t thread_dim = static_cast<size_t>(thread_param->thread_dim);

  for (size_t thread_id = 0; thread_id < thread_dim; ++thread_id) {
    auto ctxid = ctx_id_vec[thread_id % ctx_num];
    if (wrapper.CreateFftsCtxInfo(static_cast<uint32_t>(thread_id),
                                  static_cast<uint32_t>(ctxid)) != ge::GRAPH_SUCCESS) {
      GELOGE(ge::FAILED, "CreateFftsCtxInfo failed, thread_id:%u, ctx_id:%u.", thread_id, ctxid);
      return ge::GRAPH_FAILED;
    }
  }

  GELOGI("End FillDataDumpCtxInfo.");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetAddrSize(const KernelContext *context, size_t param_id, bool input_flag, bool last_flag,
                            uint64_t &addr_size) {
  GELOGI("Begin GetAddrSize.");
  auto extend_context = reinterpret_cast<const ExtendedKernelContext *>(context);
  GE_ASSERT_NOTNULL(extend_context);
  auto compute_node_info = extend_context->GetComputeNodeInfo();
  GE_ASSERT_NOTNULL(compute_node_info);

  auto tensor = input_flag ?
                compute_node_info->GetInputTdInfo(param_id):
                compute_node_info->GetOutputTdInfo(param_id);

  int64_t batch_size = 1;
  if (last_flag) {
    auto tail_slice =
        input_flag ?
        context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(FFTSAicpuArgss::kLastInSlice)):
        context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(FFTSAicpuArgss::kLastOutSlice));
    auto tail_slice_vec = reinterpret_cast<const Shape *>(tail_slice->GetData());

    const size_t slice_dim_num = tail_slice_vec[param_id].GetDimNum();
    for (size_t j = 0; j < slice_dim_num; ++j) {
      if (ge::MulOverflow(batch_size, tail_slice_vec[param_id][j], batch_size)) {
        GELOGE(ge::FAILED, "tail_size[%ld] calculate err with [%lu].", batch_size, tail_slice_vec[param_id][j]);
        return ge::GRAPH_FAILED;
      }
    }
  } else {
    auto not_tail_slice =
        input_flag ?
        context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(FFTSAicpuArgss::kNotLastInSlice)):
        context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(FFTSAicpuArgss::kNotLastOutSlice));
    auto not_tail_slice_vec = reinterpret_cast<const Shape *>(not_tail_slice->GetData());

    const size_t slice_dim_num = not_tail_slice_vec[param_id].GetDimNum();
    for (size_t j = 0; j < slice_dim_num; ++j) {
      if (ge::MulOverflow(batch_size, not_tail_slice_vec[param_id][j], batch_size)) {
        GELOGE(ge::FAILED, "tail_size[%ld] calculate err with [%lu].", batch_size, not_tail_slice_vec[param_id][j]);
        return ge::GRAPH_FAILED;
      }
    }
  }
  addr_size = ge::GetSizeInBytes(batch_size, tensor->GetDataType());
  GELOGI("End GetAddrSize.");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetFftsMemBlockInfo(const KernelContext *context, size_t index, uint32_t thread_dim, size_t param_id,
                                    bool input_flag, gert::DataDumpInfoWrapper &wrapper) {
  GELOGI("Begin GetFftsMemBlockInfo.");
  auto ffts_mem = context->GetInputValue<memory::FftsMemBlock *>(static_cast<size_t>(FFTSAicpuArgss::kIoStart) + index);
  GE_ASSERT_NOTNULL(ffts_mem);
  for (uint32_t j = 0U; j < thread_dim; ++j) {
    uint64_t addr_size = 0UL;
    if (GetAddrSize(context, param_id, input_flag, j == (thread_dim - 1), addr_size) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    auto addr = ge::PtrToValue(ffts_mem->Addr(j));
    if (wrapper.AddFftsCtxAddr(j, input_flag, addr, addr_size) != ge::GRAPH_SUCCESS) {
      GELOGE(ge::FAILED, "AddFftsCtxAddr failed, thread_id:%u, input_addr:%lu, input_size:%lu.", j, addr, addr_size);
      return ge::GRAPH_FAILED;
    }
  }
  GELOGI("End GetFftsMemBlockInfo.");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetNormalInfo(const KernelContext *context, size_t index, uint32_t thread_dim, size_t param_id,
                              bool input_flag, const AICpuThreadParam *thread_param,
                              gert::DataDumpInfoWrapper &wrapper) {
  GELOGI("Begin GetNormalInfo.");
  auto tensor = context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(FFTSAicpuArgss::kIoStart) + index);
  GE_ASSERT_NOTNULL(tensor);
  const uintptr_t data_base = ge::PtrToValue(tensor->GetAddr());
  for (uint32_t j = 0U; j < thread_dim; ++j) {
    uint64_t addr_size = 0UL;
    if (GetAddrSize(context, param_id, input_flag, j == (thread_dim - 1), addr_size) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    auto addr = data_base + (thread_param->task_addr_offset[index] * j);
    if (wrapper.AddFftsCtxAddr(j, input_flag, addr, addr_size) != ge::GRAPH_SUCCESS) {
      GELOGE(ge::FAILED, "AddFftsCtxAddr failed, thread_id:%u, input_addr:%lu, input_size:%lu.", j, addr, addr_size);
      return ge::GRAPH_FAILED;
    }
  }
  GELOGI("End GetNormalInfo.");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillAICpuDataDumpInfo(const KernelContext *context, gert::DataDumpInfoWrapper &wrapper) {
  GELOGI("Begin FillAICpuDataDumpInfo.");
  auto thread_param = context->GetInputPointer<AICpuThreadParam>(static_cast<size_t>(FFTSAicpuArgss::kThreadParam));
  GE_ASSERT_NOTNULL(thread_param);

  auto in_mem_type = context->GetInputPointer<ContinuousVector>(static_cast<size_t>(FFTSAicpuArgss::kInMemType));
  GE_ASSERT_NOTNULL(in_mem_type);
  auto out_mem_type = context->GetInputPointer<ContinuousVector>(static_cast<size_t>(FFTSAicpuArgss::kOutMemType));
  GE_ASSERT_NOTNULL(out_mem_type);
  auto in_mem_type_vec = reinterpret_cast<const uint32_t *>(in_mem_type->GetData());
  auto out_mem_type_vec = reinterpret_cast<const uint32_t *>(out_mem_type->GetData());

  if (FillDataDumpCtxInfo(context, wrapper) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  uint32_t thread_dim = thread_param->thread_dim;
  for (size_t i = 0UL; i < thread_param->input_output_num; ++i) {
    const bool input_flag = i < thread_param->input_num;
    const auto param_id = input_flag ? i : (i - thread_param->input_num);
    uint32_t mem_pool_type =
        input_flag ? in_mem_type_vec[param_id] : out_mem_type_vec[param_id];

    if (mem_pool_type == 1U) {
      if (GetFftsMemBlockInfo(context, i, thread_dim, param_id, input_flag, wrapper) != ge::GRAPH_SUCCESS) {
          return ge::GRAPH_FAILED;
      }
    } else {
      if (GetNormalInfo(context, i, thread_dim, param_id, input_flag, thread_param, wrapper) !=
          ge::GRAPH_SUCCESS) {
          return ge::GRAPH_FAILED;
      }
    }
  }
  GELOGI("End FillAICpuDataDumpInfo.");
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(FFTSUpdateAICpuCCArgs)
    .RunFunc(FFTSUpdateAICpuCCArgs)
    .OutputsCreator(AiCpuCreateFlushData)
    .DataDumpInfoFiller(FillAICpuDataDumpInfo)
    .TracePrinter(PrintAICpuArgs);

REGISTER_KERNEL(FFTSUpdateAICpuTfArgs)
    .RunFunc(FFTSUpdateAICpuTfArgs)
    .OutputsCreator(AiCpuCreateFlushData)
    .DataDumpInfoFiller(FillAICpuDataDumpInfo)
    .TracePrinter(PrintAICpuArgs);

ge::graphStatus FFTSInitAicpuCtxUserData(KernelContext *context) {
  auto ctx_ids = context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(FFTSAicpuSoAndKernel::kCtxIds));
  auto index = context->GetInputValue<size_t>(static_cast<size_t>(FFTSAicpuSoAndKernel::kIndex));
  auto so_name_dev =
      context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(FFTSAicpuSoAndKernel::kSoNameDev));
  auto so_name_size = context->GetInputValue<size_t>(static_cast<size_t>(FFTSAicpuSoAndKernel::kSoNameSize));
  auto so_name_data = context->GetInputValue<char *>(static_cast<size_t>(FFTSAicpuSoAndKernel::kSoNameData));
  auto kernel_name_dev =
      context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(FFTSAicpuSoAndKernel::kKernelNameDev));
  auto kernel_name_size = context->GetInputValue<size_t>(static_cast<size_t>(FFTSAicpuSoAndKernel::kKernelNameSize));
  auto kernel_name_data = context->GetInputValue<char *>(static_cast<size_t>(FFTSAicpuSoAndKernel::kKernelNameData));
  auto stream = context->GetInputValue<rtStream_t>(static_cast<size_t>(FFTSAicpuSoAndKernel::kStream));
  auto task_info = context->GetOutputPointer<rtFftsPlusTaskInfo_t>(static_cast<int32_t>(0));

  GE_ASSERT_NOTNULL(ctx_ids);
  GE_ASSERT_NOTNULL(so_name_dev);
  GE_ASSERT_NOTNULL(so_name_data);
  GE_ASSERT_NOTNULL(kernel_name_dev);
  GE_ASSERT_NOTNULL(kernel_name_data);
  GE_ASSERT_NOTNULL(task_info);
  GE_ASSERT_NOTNULL(task_info->descBuf);
  GE_ASSERT_NOTNULL(task_info->fftsPlusSqe);

  rtFftsPlusAiCpuCtx_t *ctx_head = reinterpret_cast<rtFftsPlusAiCpuCtx_t *>(const_cast<void *>(task_info->descBuf));
  auto ctx_id_vec = reinterpret_cast<const int32_t *>(ctx_ids->GetData());
  const uint64_t address_offset = 32UL;
  uint16_t total_num = task_info->fftsPlusSqe->totalContextNum;

  if (ctx_id_vec[index] > total_num) {
    GELOGE(ge::FAILED, "Context Id(%d) over flow.", ctx_id_vec[index]);
    return ge::GRAPH_FAILED;
  }
  auto ctx = reinterpret_cast<rtFftsPlusAiCpuCtx_t *>(ctx_head + ctx_id_vec[index]);

  // copy so name
  const std::string so_name = std::string(so_name_data, so_name_size);
  GE_CHK_RT_RET(rtMemcpyAsync(so_name_dev->GetAddr(), so_name_size, so_name.data(), so_name_size,
                              RT_MEMCPY_HOST_TO_DEVICE_EX, stream));
  ctx->usrData[kSoNameAddrLIndex] = static_cast<uint32_t>(ge::PtrToValue(so_name_dev->GetAddr()) & k32BitsMask);
  ctx->usrData[kSoNameAddrHIndex] = static_cast<uint32_t>(ge::PtrToValue(so_name_dev->GetAddr()) >> address_offset);
  // copy kernel name
  const std::string kernel_name = std::string(kernel_name_data, kernel_name_size);
  GE_CHK_RT_RET(rtMemcpyAsync(kernel_name_dev->GetAddr(), kernel_name_size, kernel_name.data(), kernel_name_size,
                              RT_MEMCPY_HOST_TO_DEVICE_EX, stream));
  ctx->usrData[kKernelNameAddrLIndex] = static_cast<uint32_t>(ge::PtrToValue(kernel_name_dev->GetAddr()) & k32BitsMask);
  ctx->usrData[kKernelNameAddrHIndex] =
      static_cast<uint32_t>(ge::PtrToValue(kernel_name_dev->GetAddr()) >> address_offset);

  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(FFTSInitAicpuCtxUserData).RunFunc(FFTSInitAicpuCtxUserData);
}  // namespace kernel
}  // namespace gert
