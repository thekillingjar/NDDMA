/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ffts_update_kernel.h"
#include "register/ffts_plus_task_update.h"
#include "register/ffts_node_calculater_registry.h"
#include "kernel/kernel_log.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "register/op_tiling.h"
#include "graph/ge_error_codes.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "register/kernel_registry_impl.h"
#include "common/checker.h"
#include "common/sgt_slice_type.h"
#include "engine/aicore/fe_rt2_common.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "engine/ffts_plus/converter/ffts_plus_common.h"
namespace gert {
namespace kernel {
namespace {
ge::graphStatus FftsTaskInfoPreProc(KernelContext *context) {
  auto task_data = context->GetInputValue<uint8_t*>(static_cast<size_t>(0));
  FE_ASSERT_NOTNULL(task_data);
  auto task_info_para = context->GetInputValue<NodeMemPara*>(static_cast<size_t>(1));
  FE_ASSERT_NOTNULL(task_info_para);
  const auto task_info_data = reinterpret_cast<const TransTaskInfo*>(task_data);
  auto task_info = reinterpret_cast<TransTaskInfo*>(task_info_para->host_addr);
  size_t head_len = sizeof(TransTaskInfo) + sizeof(rtFftsPlusSqe_t);
  if (memcpy_s(task_info, task_info_para->size, task_info_data, head_len) != EOK) {
    KLOGE("Copy task info data[%zu] to dst[%zu] failed.", head_len, task_info_para->size);
    return ge::GRAPH_FAILED;
  }
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info->args);
  task_info->rt_task_info.fftsPlusSqe = ffts_plus_sqe;

  // descBuf ptr addr need 128byte align
  size_t buf_offset = task_info_data->offsets[static_cast<size_t>(InfoStType::kDescBuf)];
  auto dev_task_info = reinterpret_cast<TransTaskInfo*>(task_info_para->dev_addr);
  GELOGD("Base add host[%lx], dev[%lx].", task_info_para->host_addr, task_info_para->dev_addr);
  GE_ASSERT_TRUE(buf_offset < (task_info_para->size - sizeof(TransTaskInfo)));
  uintptr_t dev_buf_base = reinterpret_cast<uintptr_t>(&dev_task_info->args[buf_offset]);
  uintptr_t align_base = AddrAlignBy128(dev_buf_base);
  size_t new_offset = reinterpret_cast<uint8_t*>(align_base) - dev_task_info->args;
  GE_ASSERT_TRUE(new_offset < (task_info_para->size - sizeof(TransTaskInfo)));
  task_info->rt_task_info.descBuf = &task_info->args[new_offset];
  const void* pre_data_base = &task_info_data->args[buf_offset];
  size_t buf_len = task_info->rt_task_info.descBufLen;
  size_t left_len = task_info_para->size - sizeof(TransTaskInfo) - new_offset;
  GELOGD("Mem base:%lx, align_base:%lx, offset:%zu, new offset:%zu, buf_len[%zu], left len[%zu].",
         dev_buf_base, align_base, buf_offset, new_offset, buf_len, left_len);
  if (memcpy_s(const_cast<void*>(task_info->rt_task_info.descBuf), left_len, pre_data_base, buf_len) != EOK) {
    KLOGE("Failed to copy op_desc buffer.");
    return ge::GRAPH_FAILED;
  }
  task_info->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = new_offset;
  auto out_ret = context->GetOutput(0U);
  FE_ASSERT_NOTNULL(out_ret);
  out_ret->Set(&task_info->rt_task_info, nullptr);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(FftsTaskInfoPreProc).RunFunc(FftsTaskInfoPreProc);

ge::graphStatus NodeMemParaAssign(KernelContext *context) {
  auto node_para = context->GetOutputPointer<NodeMemPara>(static_cast<size_t>(MemParaOutKey::NODE_PARA));
  FE_ASSERT_NOTNULL(node_para);
  auto mem_guard = context->GetOutputPointer<gert::ContinuousVector>(static_cast<size_t>(MemParaOutKey::MEM_GUARD));
  FE_ASSERT_NOTNULL(mem_guard);
  auto pre_para = context->GetInputPointer<MemPrePara>(static_cast<size_t>(MemParaInKey::PRE_PARA));
  FE_ASSERT_NOTNULL(pre_para);
  auto pre_data = context->GetInputValue<uint8_t*>(static_cast<size_t>(MemParaInKey::PRE_DATA));
  auto dev_addr = context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(MemParaInKey::DEV_ADDR));
  FE_ASSERT_NOTNULL(dev_addr);
  auto host_addr = context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(MemParaInKey::HOST_ADDR));
  FE_ASSERT_NOTNULL(host_addr);
  void* dev_addr_base = dev_addr->GetAddr();
  void* host_addr_base = host_addr->GetAddr();
  node_para->dev_addr = static_cast<uint8_t*>(dev_addr_base) + pre_para->offset;
  node_para->host_addr = static_cast<uint8_t*>(host_addr_base) + pre_para->offset;
  node_para->size = pre_para->size;
  auto guard_vec = reinterpret_cast<MemGuard*>(mem_guard->MutableData());
  size_t cur_idx = mem_guard->GetSize();
  cur_idx = (cur_idx == mem_guard->GetCapacity()) ? 0 : cur_idx;
  if (mem_guard->SetSize(cur_idx + 1) != ge::GRAPH_SUCCESS) {
    KLOGE("Resize mem guard size failed.");
    return ge::GRAPH_FAILED;
  }
  guard_vec[cur_idx].guard_ptr = static_cast<uint8_t*>(host_addr_base) + pre_para->offset + pre_para->size;
  guard_vec[cur_idx].guard_val = rand();
  *reinterpret_cast<int64_t*>(guard_vec[cur_idx].guard_ptr) = guard_vec[cur_idx].guard_val;
  GELOGD("Base addr:%lx, node addr base:%lx, size:%zu, offset:%zu, guard[%ld].", host_addr_base,
         node_para->host_addr, node_para->size, pre_para->offset, guard_vec[cur_idx].guard_val);
  if (pre_data && (pre_para->pre_size > 0)) {
    if (memcpy_s(node_para->host_addr, node_para->size, pre_data, pre_para->pre_size) != EOK) {
      KLOGE("Failed to copy pre data.");
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus CreateNodeMemPara(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto mem_para = context->GetOutput(0);
  FE_ASSERT_NOTNULL(mem_para);
  mem_para->SetWithDefaultDeleter(new (std::nothrow) NodeMemPara());
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(NodeMemParaAssign).RunFunc(NodeMemParaAssign).OutputsCreator(CreateNodeMemPara);

ge::graphStatus FFTSTaskAndArgsCopy(KernelContext *context) {
  auto need_launch = context->GetInputValue<uint32_t>(static_cast<size_t>(H2DInKey::LAUNCH_FLAG));
  if (need_launch == 0U) {
    GELOGD("No need to transfer arguments to the device.");
    return ge::GRAPH_SUCCESS;
  }
  auto stream = context->GetInputValue<rtStream_t>(static_cast<size_t>(H2DInKey::STREAM_ID));
  auto dev_addr = context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(H2DInKey::DEV_ADDR));
  FE_ASSERT_NOTNULL(dev_addr);
  auto host_addr = context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(H2DInKey::HOST_ADDR));
  FE_ASSERT_NOTNULL(host_addr);
  void* dev_addr_base = dev_addr->GetAddr();
  void* host_addr_base = host_addr->GetAddr();
  GELOGD("H2D{%lx}{%lx}{%zu}{%zu}.", dev_addr_base, host_addr_base, dev_addr->GetSize(), host_addr->GetSize());
  GE_CHK_RT_RET(rtMemcpyAsync(dev_addr_base, dev_addr->GetSize(), host_addr_base,
                              host_addr->GetSize(), RT_MEMCPY_HOST_TO_DEVICE_EX, stream));
  return ge::GRAPH_SUCCESS;
}
std::vector<std::string> CheckMemGuard(const KernelContext *context) {
  std::stringstream ss;
  std::vector<std::string> msgs;
  auto mem_guard = context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(H2DInKey::MEM_GUARD));
  if (mem_guard == nullptr) {
    GELOGW("No memory guard set up.");
    return msgs;
  }
  auto guard_vec = reinterpret_cast<const MemGuard*>(mem_guard->GetData());
  for (size_t i = 0; i < mem_guard->GetSize(); ++i) {
    auto guard = guard_vec[i];
    auto cur_val = *static_cast<int64_t*>(guard.guard_ptr);
    if (cur_val != guard.guard_val) {
      GELOGW("Mem guard[%zu] value[%ld] does not match the actual value[%ld].", i, guard.guard_val, cur_val);
      ss << "FFTS memory may has over write with block index: " << i;
      msgs.emplace_back(ss.str());
      return msgs;
    }
  }
  ss << "FFTS memory check ok.";
  msgs.emplace_back(ss.str());
  return msgs;
}
REGISTER_KERNEL(FFTSTaskAndArgsCopy).RunFunc(FFTSTaskAndArgsCopy).TracePrinter(CheckMemGuard);
}  // namespace
}  // namespace kernel
}  // namespace gert
