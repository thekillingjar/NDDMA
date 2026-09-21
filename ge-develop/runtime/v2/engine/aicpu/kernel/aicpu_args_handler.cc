/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_args_handler.h"
#include <cstddef>
#include <iomanip>
#include <cinttypes>
#include "aicpu_ext_info_handle.h"
#include "graph/op_kernel_bin.h"
#include "graph/ge_error_codes.h"
#include "graph/def_types.h"
#include "register/kernel_registry.h"
#include "framework/common/debug/log.h"
#include "exe_graph/runtime/tensor.h"
#include "kernel/memory/mem_block.h"
#include "aicpu_engine_struct.h"
#include "runtime/mem.h"
#include "common/checker.h"
#include "aicpu_task_struct.h"
#include "framework/common/taskdown_common.h"
#include "proto/task.pb.h"
#include "common/plugin/ge_make_unique_util.h"
#include "core/debug/kernel_tracing.h"
#include "hybrid/node_executor/aicpu/aicpu_ext_info_handler.h"
#include "graph/types.h"
#include "aicpu_bin_handler.h"

using namespace ge;

namespace gert {
namespace {
constexpr size_t kMaxTotalHostLen = 128U;

enum class CommonBinArgsInfo {
 kOpDesc,
 kNodeType
};

enum class CommonArgsInfo {
  kIoNum,
  kNodeName,
  kNeedDeviceExt,
  kArgData,
  kArgSize,
  kExtSize,
  kIsBlockOp,
  kRtEvent,
  kAsyncTimeout,
  kNum
};

enum class CCArgsInfo {
  kKernelName = static_cast<int32_t>(CommonArgsInfo::kNum),
  kKernelNameSize,
  kSoName,
  kSoNameSize,
};

enum class TfArgsInfo {
  kTaskData = static_cast<int32_t>(CommonArgsInfo::kNum),
  kTaskSize,
  kSessionId,
  kStepId
};

const std::string kAttrJsonPath = "ops_json_path";
const std::string kAttrCustAicpuFlag = "_cust_aicpu_flag";
const std::string kAttrFuncName = "funcName";
const std::string kAttrSoName = "kernelSo";
} // namespace

AicpuArgsHandler::~AicpuArgsHandler() {
  if (ext_info_device_buffer_ != nullptr) {
    (void) rtFree(ext_info_device_buffer_);
    ext_info_device_buffer_ = nullptr;
  }
}

ge::graphStatus AicpuArgsHandler::MallocMem() {
  host_buffer_ = ge::MakeUnique<uint8_t[]>(buffer_size_);
  GE_ASSERT_NOTNULL(host_buffer_);
  if (need_device_ext_) {
    GE_ASSERT_RT_OK(rtMalloc(&ext_info_device_buffer_, ext_info_size_, RT_MEMORY_HBM, GE_MODULE_NAME_U16));
  }
  args_.args = host_buffer_.get();
  args_.argsSize = buffer_size_;
  args_.hostInputInfoNum = 0U;
  args_.soNameAddrOffset = so_name_offset_;
  args_.kernelNameAddrOffset = kernel_name_offset_;
  return ge::GRAPH_SUCCESS;
}

void AicpuArgsHandler::ResetHostInputInfo() {
  host_input_size_ = 0U;
  host_input_info_.clear();
  args_.hostInputInfoNum = 0U;
  args_.hostInputInfoPtr = host_input_info_.data();
}

ge::graphStatus AicpuArgsHandler::AddHostInput(const size_t idx, void *data, const size_t src_size,
                                               const size_t align_size) {
  GE_ASSERT_TRUE(host_input_size_ + align_size <= kMaxTotalHostLen);
  const size_t remain_size = kMaxTotalHostLen - host_input_size_;
  auto host_data_offset = host_mem_offset_ + host_input_size_;
  auto host_data_addr = host_buffer_.get() + host_data_offset;
  if (src_size > 0U) {
    GE_ASSERT_EOK(memcpy_s(host_data_addr, remain_size, data, src_size));
  }

  args_.hostInputInfoNum += 1U;
  auto host_addr_offset = io_addr_offset_ + idx * sizeof(void *);
  host_input_info_.emplace_back(rtHostInputInfo_t({static_cast<uint16_t>(host_addr_offset),
                                                   static_cast<uint16_t>(host_data_offset)}));
  args_.hostInputInfoPtr = host_input_info_.data();
  GE_ASSERT_TRUE(args_.hostInputInfoNum == host_input_info_.size());

  host_input_size_ += align_size;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetExtInfoAddrFromArgs(KernelContext *context) {
  auto args_handler = context->GetInputPointer<AicpuArgsHandler>(0U);
  auto host_ext_addr = context->GetOutputPointer<uint8_t *>(0U);
  auto device_ext_addr = context->GetOutputPointer<void *>(1U);
  GE_ASSERT_NOTNULL(args_handler);
  GE_ASSERT_NOTNULL(host_ext_addr);
  GE_ASSERT_NOTNULL(device_ext_addr);

  *host_ext_addr = args_handler->GetExtInfoAddr();
  *device_ext_addr = args_handler->GetExtInfoDeivceAddr();
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(GetExtInfoAddrFromArgs).RunFunc(GetExtInfoAddrFromArgs);

ge::graphStatus CreateOutputsForAicpuArgs(const ge::FastNode *node, KernelContext *context) {
  (void) node;
  auto io_num = context->GetInputValue<size_t>(static_cast<size_t>(CommonArgsInfo::kIoNum));
  auto node_name = context->GetInputValue<char_t *>(static_cast<size_t>(CommonArgsInfo::kNodeName));
  auto need_device_ext = context->GetInputValue<bool>(static_cast<size_t>(CommonArgsInfo::kNeedDeviceExt));
  GE_ASSERT_NOTNULL(node_name);

  auto args_handler = ge::MakeUnique<AicpuArgsHandler>(node_name, io_num, need_device_ext);
  auto av_holder = context->GetOutput(0U);
  GE_ASSERT_NOTNULL(args_handler);
  GE_ASSERT_NOTNULL(av_holder);
  av_holder->SetWithDefaultDeleter<AicpuArgsHandler>(args_handler.release());
  return ge::GRAPH_SUCCESS;
}

// for CC task
// ext_info_addr may host or device
ge::graphStatus AicpuCCArgsHandler::SetHostArgs(const std::string &arg_data, const size_t ext_info_size,
                                                const uint64_t ext_info_addr) {
  GE_ASSERT_EOK(memcpy_s(host_buffer_.get(), buffer_size_, arg_data.data(), arg_data.size()));
  auto aicpu_param_head = PtrToPtr<uint8_t, aicpu::AicpuParamHead>(host_buffer_.get());

  GE_ASSERT_TRUE(aicpu_param_head->ioAddrNum == io_num_);
  const auto mini_len = sizeof(aicpu::AicpuParamHead) + io_num_ * sizeof(void *);
  GE_ASSERT_TRUE(aicpu_param_head->length >= mini_len);
  io_addr_offset_ = sizeof(aicpu::AicpuParamHead);

  aicpu_param_head->extInfoLength = ext_info_size;
  aicpu_param_head->extInfoAddr = ext_info_addr;

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AicpuCCArgsHandler::BuildHostCCArgs(const std::string &arg_data, const size_t ext_info_size) {
  const size_t arg_size = arg_data.size();
  ext_info_offset_ = arg_size + io_num_ * sizeof(void *);
  buffer_size_ = ext_info_offset_ + ext_info_size;

  host_buffer_ = ge::MakeUnique<uint8_t[]>(buffer_size_);
  GE_ASSERT_NOTNULL(host_buffer_);
  auto ext_info_addr = PtrToValue(PtrToPtr<uint8_t, void>(host_buffer_.get())) + ext_info_offset_;
  GE_ASSERT_SUCCESS(SetHostArgs(arg_data, ext_info_size, ext_info_addr));

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildHostCCArgs(KernelContext *context) {
  auto arg_data = context->GetInputValue<char_t *>(static_cast<size_t>(CommonArgsInfo::kArgData));
  auto arg_size = context->GetInputValue<size_t>(static_cast<size_t>(CommonArgsInfo::kArgSize));
  auto ext_size = context->GetInputValue<size_t>(static_cast<size_t>(CommonArgsInfo::kExtSize));
  GE_ASSERT_NOTNULL(arg_data);
  const std::string arg_info = std::string(arg_data, arg_size);

  auto args_handler = context->GetOutputPointer<AicpuCCArgsHandler>(0U);
  GE_ASSERT_NOTNULL(args_handler);
  return args_handler->BuildHostCCArgs(arg_info, ext_size);
}

ge::graphStatus FillLaunchArgs(const KernelContext *context, gert::ExceptionDumpInfoWrapper &wrapper) {
  auto args_handler = context->GetInputPointer<AicpuArgsHandler>(0U);
  GE_ASSERT_NOTNULL(args_handler);

  auto &arg_ex = args_handler->GetArgsEx();
  size_t args_size = static_cast<size_t>(arg_ex.argsSize);

  uintptr_t args = static_cast<uintptr_t>(ge::PtrToValue(args_handler->GetArgs()));
  wrapper.SetHostArgs(args, args_size);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(BuildHostCCArgs).RunFunc(BuildHostCCArgs).OutputsCreator(CreateOutputsForAicpuArgs)
    .ExceptionDumpInfoFiller(FillLaunchArgs);

ge::graphStatus AicpuCCArgsHandler::SetOffsetArgs() {
  aicpu::AicpuParamHead param_head;
  args_.kernelOffsetInfoNum = 0U;
  if (!need_device_ext_) {
    auto ext_info_addr_offset = PtrToValue(&(param_head.extInfoAddr)) - PtrToValue(&param_head);
    args_.kernelOffsetInfoNum += 1U;
    kernel_offset_info_.emplace_back(rtHostInputInfo_t({static_cast<uint32_t>(ext_info_addr_offset),
                                                        static_cast<uint32_t>(ext_info_offset_)}));
  }
  args_.kernelOffsetInfoPtr = kernel_offset_info_.data();
  args_.soNameAddrOffset = so_name_offset_;
  args_.kernelNameAddrOffset = kernel_name_offset_;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AicpuCCArgsHandler::BuildCCArgs(const std::string &arg_data, const std::string &kernel_name,
                                                const std::string &so_name, const size_t ext_info_size) {
  const size_t arg_size = arg_data.size();
  const size_t kernel_name_size = kernel_name.size();
  const size_t so_name_size = so_name.size();

  ext_info_size_ = ext_info_size;
  kernel_name_offset_ = arg_size + io_num_ * sizeof(void *);
  so_name_offset_ = kernel_name_offset_ + kernel_name_size;
  ext_info_offset_ = so_name_offset_ + so_name_size;
  host_mem_offset_ = ext_info_offset_ + ext_info_size;
  buffer_size_ = host_mem_offset_ + kMaxTotalHostLen;

  GE_ASSERT_SUCCESS(MallocMem());
  // ext_info_device_buffer_ only valid in dynamic 3th op
  GE_ASSERT_SUCCESS(SetHostArgs(arg_data, ext_info_size, PtrToValue(ext_info_device_buffer_)));
  GE_ASSERT_SUCCESS(SetOffsetArgs());

  GE_ASSERT_EOK(memcpy_s(host_buffer_.get() + kernel_name_offset_, kernel_name_size, kernel_name.data(),
                         kernel_name_size));
  GE_ASSERT_EOK(memcpy_s(host_buffer_.get() + so_name_offset_, so_name_size, so_name.data(), so_name_size));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildCCArgs(KernelContext *context) {
  auto arg_data = context->GetInputValue<char_t *>(static_cast<size_t>(CommonArgsInfo::kArgData));
  auto arg_size = context->GetInputValue<size_t>(static_cast<size_t>(CommonArgsInfo::kArgSize));
  auto ext_info_size = context->GetInputValue<size_t>(static_cast<size_t>(CommonArgsInfo::kExtSize));
  auto is_block_op = context->GetInputValue<bool>(static_cast<size_t>(CommonArgsInfo::kIsBlockOp));
  auto rt_event = context->GetInputValue<rtEvent_t>(static_cast<size_t>(CommonArgsInfo::kRtEvent));
  auto async_timeout = context->GetInputValue<uint32_t>(static_cast<size_t>(CommonArgsInfo::kAsyncTimeout));
  auto kernel_name_data = context->GetInputValue<char_t *>(static_cast<size_t>(CCArgsInfo::kKernelName));
  auto kernel_name_size = context->GetInputValue<size_t>(static_cast<size_t>(CCArgsInfo::kKernelNameSize));
  auto so_name_data = context->GetInputValue<char_t *>(static_cast<size_t>(CCArgsInfo::kSoName));
  auto so_name_size = context->GetInputValue<size_t>(static_cast<size_t>(CCArgsInfo::kSoNameSize));
  GE_ASSERT_NOTNULL(arg_data);
  GE_ASSERT_NOTNULL(kernel_name_data);
  GE_ASSERT_NOTNULL(so_name_data);
  const std::string arg_info = std::string(arg_data, arg_size);
  const std::string kernel_name = std::string(kernel_name_data, kernel_name_size);
  const std::string so_name = std::string(so_name_data, so_name_size);

  auto args_handler = context->GetOutputPointer<AicpuCCArgsHandler>(0U);
  GE_ASSERT_NOTNULL(args_handler);
  args_handler->SetBlockOp(is_block_op);
  args_handler->SetRtEvent(rt_event);
  args_handler->SetAsyncTimeout(async_timeout);
  return args_handler->BuildCCArgs(arg_info, kernel_name, so_name, ext_info_size);
}
REGISTER_KERNEL(BuildCCArgs).RunFunc(BuildCCArgs).OutputsCreator(CreateOutputsForAicpuArgs);

// for Tf task
ge::graphStatus AicpuTfArgsHandler::SetOffsetArgs() {
  STR_FWK_OP_KERNEL fwk_op_kernel = {};
  auto workspace_base_addr_offset = PtrToValue(&(fwk_op_kernel.fwkKernelBase.fwk_kernel.workspaceBaseAddr)) -
                                    PtrToValue(&fwk_op_kernel);
  auto io_addr_offset = PtrToValue(&(fwk_op_kernel.fwkKernelBase.fwk_kernel.inputOutputAddr)) -
                        PtrToValue(&fwk_op_kernel);

  args_.kernelOffsetInfoNum = 2U; // workspace & io_addr
  kernel_offset_info_.emplace_back(rtHostInputInfo_t({static_cast<uint32_t>(workspace_base_addr_offset),
                                                      static_cast<uint32_t>(task_info_offset_)}));
  kernel_offset_info_.emplace_back(rtHostInputInfo_t({static_cast<uint32_t>(io_addr_offset),
                                                      static_cast<uint32_t>(io_addr_offset_)}));

  if (!need_device_ext_) {
    auto ext_info_addr_offset = PtrToValue(&(fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoAddr)) -
                                PtrToValue(&fwk_op_kernel);
    args_.kernelOffsetInfoNum += 1U;
    kernel_offset_info_.emplace_back(rtHostInputInfo_t({static_cast<uint32_t>(ext_info_addr_offset),
                                                        static_cast<uint32_t>(ext_info_offset_)}));
  }
  args_.kernelOffsetInfoPtr = kernel_offset_info_.data();
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AicpuTfArgsHandler::BuildTfArgs(const std::string &arg_data, const std::string &task_info,
                                                const size_t ext_info_size, const uint64_t session_id,
                                                const void *step_id) {
  const size_t arg_size = arg_data.size();
  const size_t task_info_size = task_info.size();
  const size_t io_addr_size = io_num_ * sizeof(void *);

  ext_info_size_ = ext_info_size;
  task_info_offset_ = arg_size;
  io_addr_offset_ = task_info_offset_ + task_info_size;
  ext_info_offset_ = io_addr_offset_ + io_addr_size;
  host_mem_offset_ = ext_info_offset_ + ext_info_size;
  so_name_offset_ = host_mem_offset_ + kMaxTotalHostLen;
  kernel_name_offset_ = so_name_offset_ + sizeof(void *);
  buffer_size_ = kernel_name_offset_ + sizeof(void *);

  // tf input addr must align 64 bytes
  align_bytes_ = 64U;

  GE_ASSERT_SUCCESS(MallocMem());
  GE_ASSERT_SUCCESS(SetOffsetArgs());
  // update ext_info, kernel_id
  STR_FWK_OP_KERNEL fwk_op_kernel = {};
  GE_ASSERT_EOK(memcpy_s(&fwk_op_kernel, sizeof(STR_FWK_OP_KERNEL), arg_data.data(), arg_size));
  fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoLen = ext_info_size;
  // ext_info_device_buffer_ only valid in dynamic 3th op
  fwk_op_kernel.fwkKernelBase.fwk_kernel.extInfoAddr = PtrToValue(ext_info_device_buffer_);
  fwk_op_kernel.fwkKernelBase.fwk_kernel.kernelID = ::ge::hybrid::AicpuExtInfoHandler::GenerateKernelId();
  fwk_op_kernel.fwkKernelBase.fwk_kernel.sessionID = session_id;
  fwk_op_kernel.fwkKernelBase.fwk_kernel.stepIDAddr = ge::PtrToValue(step_id);

  GE_ASSERT_EOK(memcpy_s(host_buffer_.get(), sizeof(STR_FWK_OP_KERNEL), &fwk_op_kernel, sizeof(STR_FWK_OP_KERNEL)));
  GE_ASSERT_EOK(memcpy_s(host_buffer_.get() + task_info_offset_, task_info_size, task_info.data(), task_info_size));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildTfArgs(KernelContext *context) {
  auto arg_data = context->GetInputValue<char_t *>(static_cast<size_t>(CommonArgsInfo::kArgData));
  auto arg_size = context->GetInputValue<size_t>(static_cast<size_t>(CommonArgsInfo::kArgSize));
  auto ext_size = context->GetInputValue<size_t>(static_cast<size_t>(CommonArgsInfo::kExtSize));
  auto is_block_op = context->GetInputValue<bool>(static_cast<size_t>(CommonArgsInfo::kIsBlockOp));
  auto rt_event = context->GetInputValue<rtEvent_t>(static_cast<size_t>(CommonArgsInfo::kRtEvent));
  auto async_timeout = context->GetInputValue<uint32_t>(static_cast<size_t>(CommonArgsInfo::kAsyncTimeout));
  auto task_data = context->GetInputValue<char_t *>(static_cast<size_t>(TfArgsInfo::kTaskData));
  auto task_size = context->GetInputValue<size_t>(static_cast<size_t>(TfArgsInfo::kTaskSize));
  auto session_id = context->GetInputPointer<uint64_t>(static_cast<size_t>(TfArgsInfo::kSessionId));
  auto step_id = context->GetInputValue<void *>(static_cast<size_t>(TfArgsInfo::kStepId));
  GE_ASSERT_NOTNULL(arg_data);
  GE_ASSERT_NOTNULL(task_data);
  GE_ASSERT_NOTNULL(session_id);
  const std::string arg_info = std::string(arg_data, arg_size);
  const std::string task_info = std::string(task_data, task_size);

  auto args_handler = context->GetOutputPointer<AicpuTfArgsHandler>(0U);
  GE_ASSERT_NOTNULL(args_handler);

  args_handler->SetBlockOp(is_block_op);
  args_handler->SetRtEvent(rt_event);
  args_handler->SetAsyncTimeout(async_timeout);
  return args_handler->BuildTfArgs(arg_info, task_info, ext_size, *session_id, step_id);
}
REGISTER_KERNEL(BuildTfArgs).RunFunc(BuildTfArgs).OutputsCreator(CreateOutputsForAicpuArgs);

ge::graphStatus BuildCCArgsBinHandle(KernelContext *context) {
  auto op_desc = context->GetInputValue<ge::OpDesc *>(static_cast<size_t>(CommonBinArgsInfo::kOpDesc));
  auto node_type = context->GetInputValue<char_t *>(static_cast<size_t>(CommonBinArgsInfo::kNodeType));
  auto bin_handle = context->GetOutputPointer<rtBinHandle>(0U);
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_NOTNULL(node_type);
  GE_ASSERT_NOTNULL(bin_handle);
  bool custom_flag = false;
  (void)ge::AttrUtils::GetBool(op_desc, kAttrCustAicpuFlag, custom_flag);
  if (!custom_flag) {
    std::string json_path;
    (void)ge::AttrUtils::GetStr(op_desc, kAttrJsonPath, json_path);
    if (json_path == "") {
      *bin_handle = nullptr;
      GELOGI("Node[%s] attr ops_json_path is empty, can not get bin handle.", node_type);
      return ge::GRAPH_SUCCESS;
    }
    GELOGI("Building aicpu bin handle, node_type=%s, json_path=%s", node_type, json_path.c_str());
    GE_ASSERT_SUCCESS(AicpuJsonBinHandler::Instance().LoadBinary(json_path));
    *bin_handle = AicpuJsonBinHandler::Instance().GetBinHandle();
    GE_ASSERT_NOTNULL(*bin_handle);
  }
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(BuildCCArgsBinHandle).RunFunc(BuildCCArgsBinHandle);

ge::graphStatus BuildTfArgsBinHandle(KernelContext *context) {
  auto op_desc = context->GetInputValue<ge::OpDesc *>(static_cast<size_t>(CommonBinArgsInfo::kOpDesc));
  auto node_type = context->GetInputValue<char_t *>(static_cast<size_t>(CommonBinArgsInfo::kNodeType));
  auto bin_handle = context->GetOutputPointer<rtBinHandle>(0U);
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_NOTNULL(node_type);
  GE_ASSERT_NOTNULL(bin_handle);
  std::string json_path;
  (void)ge::AttrUtils::GetStr(op_desc, kAttrJsonPath, json_path);
  if (json_path == "") {
    *bin_handle = nullptr;
    GELOGI("Node[%s] attr ops_json_path is empty, can not get bin handle.", node_type);
    return ge::GRAPH_SUCCESS;
  }
  GELOGI("Building aicpu tf bin handle, node_type=%s, json_path=%s", node_type, json_path.c_str());
  GE_ASSERT_SUCCESS(TfJsonBinHandler::Instance().LoadBinary(json_path));
  *bin_handle = TfJsonBinHandler::Instance().GetBinHandle();
  GE_ASSERT_NOTNULL(*bin_handle);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(BuildTfArgsBinHandle).RunFunc(BuildTfArgsBinHandle);

}  // namespace gert
