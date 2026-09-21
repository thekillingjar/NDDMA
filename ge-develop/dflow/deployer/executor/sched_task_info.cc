/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "executor/sched_task_info.h"
#include "acl/acl.h"
#include "runtime/rt_external.h"
#include "graph/def_types.h"
#include "framework/common/util.h"
#include "common/df_chk.h"

namespace ge {
namespace {
constexpr uint32_t kCoreDim = 1U;
constexpr int32_t kInvalidStreamId = -1;
constexpr uint32_t kDynamicFlag = 1U;
constexpr uint32_t kDynamicFlagV2 = 2U;
constexpr const char *kSchedTaskModelDequeue = "modelDequeue";
constexpr const char *kSchedTaskModelEnqueue = "modelEnqueue";
constexpr const char *kSchedTaskModelBatchDequeue = "modelBatchDequeue";
constexpr const char *kSchedTaskPrepareDynamicInputOutput = "prepareDynamicInputOutput";
constexpr const char *kSchedTaskPrepareDynamicInputOutputV2 = "prepareDynamicInputOutputV2";
constexpr const char *kSchedTaskPostprecessDynamicOutput = "postprocessDynamicOutput";
constexpr const char *kSchedTaskPostprecessDynamicOutputV2 = "postprocessDynamicOutputV2";
constexpr const char *kSchedTaskModelBatchEnqueue = "modelBatchEnqueue";
constexpr const char *kSchedTaskNotifyWait = "waitNotify";
constexpr const char *kSchedTaskNotifyRecord = "recordNotify";
constexpr const char *kSchedTaskZeroCopy = "cpuZeroCpy";
constexpr const char *kSchedTaskStreamRepeat = "streamRepeat";
constexpr const char *kSchedTaskMarkStep = "markStep";
constexpr const char *kSchedTaskModelGatherDequeue = "gatherDequeue";
}

SchedTaskInfo::SchedTaskInfo(aclrtStream const stream) {
  stream_ = stream;
}

Status SchedTaskInfo::Release() {
  DF_FREE_ACL_RT_LOG(args_);
  args_size_ = 0UL;
  return SUCCESS;
}
Status SchedTaskInfo::LaunchCpuKernel(const char *kernel_name) const {
  rtArgsEx_t args_info = {};
  args_info.args = args_;
  args_info.argsSize = static_cast<uint32_t>(args_size_);
  args_info.isNoNeedH2DCopy = 1U;
  GE_CHK_RT_RET(rtCpuKernelLaunchWithFlag(nullptr, kernel_name, kCoreDim,
      &args_info, nullptr, stream_, RT_KERNEL_DEFAULT));
  return SUCCESS;
}

Status SchedTaskModelDequeue::Init(const uint32_t queue_id, uint64_t &mbuf_addr) {
  GE_CHK_BOOL_RET_STATUS((args_ == nullptr) && (args_size_ == 0UL), FAILED,
                         "Task:%s has already initialized, size:%lu.", kSchedTaskModelDequeue, args_size_);
  args_size_ = sizeof(QueueInfoKernelArgs) + sizeof(uint64_t);
  DF_CHK_ACL_RET(aclrtMalloc(&args_, args_size_, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  GE_PRINT_DYNAMIC_MEMORY(aclrtMalloc, "args data.", args_size_);
  mbuf_addr = PtrToValue(args_) + sizeof(QueueInfoKernelArgs);
  QueueInfoKernelArgs queue_info;
  queue_info.queue_id = queue_id;
  queue_info.mbuf_addr = mbuf_addr;
  DF_CHK_ACL_RET(aclrtMemcpy(args_, args_size_, &queue_info, sizeof(queue_info), ACL_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status SchedTaskModelDequeue::Distribute() {
  GE_CHK_BOOL_RET_STATUS((args_ != nullptr) && (args_size_ != 0UL) && (stream_ != nullptr), FAILED,
                         "Task:%s not initialized, distribute failed, size:%lu.", kSchedTaskModelDequeue, args_size_);
  GE_CHK_RT_RET(LaunchCpuKernel(kSchedTaskModelDequeue));
  GELOGI("Success to launch task:%s.", kSchedTaskModelDequeue);
  return SUCCESS;
}

Status SchedTaskModelEnqueue::Init(const uint32_t queue_id, const uint64_t mbuf_addr) {
  GE_CHK_BOOL_RET_STATUS((args_ == nullptr) && (args_size_ == 0UL), FAILED,
                         "Task:%s has already initialized, size:%lu.", kSchedTaskModelEnqueue, args_size_);
  args_size_ = sizeof(QueueInfoKernelArgs);
  DF_CHK_ACL_RET(aclrtMalloc(&args_, args_size_, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  GE_PRINT_DYNAMIC_MEMORY(aclrtMalloc, "args data.", args_size_);
  QueueInfoKernelArgs queue_info{};
  queue_info.queue_id = queue_id;
  queue_info.mbuf_addr = mbuf_addr;
  DF_CHK_ACL_RET(aclrtMemcpy(args_, args_size_, &queue_info, sizeof(queue_info), ACL_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status SchedTaskModelEnqueue::Distribute() {
  GE_CHK_BOOL_RET_STATUS((args_ != nullptr) && (args_size_ != 0UL) && (stream_ != nullptr), FAILED,
                         "Task:%s not initialized, distribute failed, size:%lu", kSchedTaskModelEnqueue, args_size_);
  GE_CHK_RT_RET(LaunchCpuKernel(kSchedTaskModelEnqueue));
  GELOGI("Success to launch task:%s.", kSchedTaskModelEnqueue);
  return SUCCESS;
}


Status SchedTaskModelBatchDequeue::Init(const std::vector<uint32_t> &queue_ids,
                                        const uint32_t align_interval,
                                        const std::vector<uint32_t> &align_offsets,
                                        std::vector<uint64_t> &mbuf_addrs) {
  GE_CHK_BOOL_RET_STATUS((args_ == nullptr) && (args_size_ == 0UL), FAILED,
                         "Task:%s has already initialized, size:%lu.", kSchedTaskModelBatchDequeue, args_size_);
  GE_CHK_BOOL_RET_STATUS(align_offsets.empty() || (queue_ids.size() == align_offsets.size()), FAILED,
                         "Size of align_offsets[%zu] mismatch size of queue ids[%zu].",
                         align_offsets.size(), queue_ids.size());
  const size_t queue_num = queue_ids.size();
  // memory layout: kernel_args | mbuf_addrs_buffer | mbufs_buffer | queue_ids_buffer | align_offsets_buffer
  BatchQueueInfoKernelArgs kernel_args{};
  kernel_args.queue_num = static_cast<uint32_t>(queue_num);
  kernel_args.align_interval = align_interval;
  args_size_ = sizeof(kernel_args);
  const uint64_t mbuf_addrs_offset = args_size_;
  const size_t mbuf_addrs_size = sizeof(uint64_t) * queue_num;
  args_size_ += mbuf_addrs_size;
  const uint64_t mbufs_offset = args_size_;
  const uint64_t mbufs_size = sizeof(uint64_t) * queue_num;
  args_size_ += mbufs_size;
  const uint64_t queue_ids_offset = args_size_;
  const size_t queue_ids_size = sizeof(uint32_t) * queue_num;
  args_size_ += queue_ids_size;
  const uint64_t align_offsets_offset = align_offsets.empty() ? 0UL : args_size_;
  const size_t align_offsets_size = align_offsets.empty() ? 0UL : sizeof(uint32_t) * queue_num;
  args_size_ += align_offsets_size;
  DF_CHK_ACL_RET(aclrtMalloc(&args_, args_size_, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  GE_PRINT_DYNAMIC_MEMORY(aclrtMalloc, "args data.", args_size_);
  kernel_args.align_offsets_addr = align_offsets.empty() ? 0UL : PtrToValue(args_) + align_offsets_offset;
  kernel_args.queue_ids_addr = PtrToValue(args_) + queue_ids_offset;
  kernel_args.mbuf_addrs_addr = PtrToValue(args_) + mbuf_addrs_offset;
  const uint64_t mbufs_addr = PtrToValue(args_) + mbufs_offset;
  for (size_t i = 0UL; i < queue_num; ++i) {
    mbuf_addrs.emplace_back(mbufs_addr + sizeof(uint64_t) * i);
  }
  DF_CHK_ACL_RET(aclrtMemcpy(args_, args_size_,
                         &kernel_args, sizeof(kernel_args), ACL_MEMCPY_HOST_TO_DEVICE));
  DF_CHK_ACL_RET(aclrtMemcpy(ValueToPtr(kernel_args.queue_ids_addr), queue_ids_size,
                         queue_ids.data(), queue_ids_size, ACL_MEMCPY_HOST_TO_DEVICE));
  if (!align_offsets.empty()) {
    DF_CHK_ACL_RET(aclrtMemcpy(ValueToPtr(kernel_args.align_offsets_addr), align_offsets_size,
                           align_offsets.data(), align_offsets_size, ACL_MEMCPY_HOST_TO_DEVICE));
  }
  DF_CHK_ACL_RET(aclrtMemcpy(ValueToPtr(kernel_args.mbuf_addrs_addr), mbuf_addrs_size,
                         mbuf_addrs.data(), mbuf_addrs.size() * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status SchedTaskModelBatchDequeue::Distribute() {
  GE_CHK_BOOL_RET_STATUS((args_ != nullptr) && (args_size_ != 0UL) && (stream_ != nullptr), FAILED,
                         "Task:%s not initialized, distribute failed, size:%lu.",
                         kSchedTaskModelBatchDequeue, args_size_);
  GE_CHK_RT_RET(LaunchCpuKernel(kSchedTaskModelBatchDequeue));
  GELOGI("Success to launch task:%s.", kSchedTaskModelBatchDequeue);
  return SUCCESS;
}

Status SchedTaskModelGatherDequeue::Init(const std::vector<QueueAttrs> &queues,
                                         const InputAlignAttrs &input_align_attrs,
                                         std::vector<uint64_t> &mbuf_addrs) {
  GE_CHK_BOOL_RET_STATUS((args_ == nullptr) && (args_size_ == 0UL), FAILED,
                         "Task:%s has already initialized, size:%lu.", kSchedTaskModelGatherDequeue, args_size_);
  const size_t queue_num = queues.size();
  // memory layout: kernel_args | queue_ids_addr | mbufs_buffer_ptr_addr |
  //                queue_device_buff | queue_device_type_buff | mbuff_buffs_ptrs
  GatherDequeueParam kernel_args{};
  kernel_args.input_nums = static_cast<uint32_t>(queue_num);
  kernel_args.inputs_align_timeout = input_align_attrs.align_timeout;
  kernel_args.inputs_align_max_cache_num = input_align_attrs.align_max_cache_num;
  kernel_args.inputs_align_drop_out = static_cast<int32_t>(input_align_attrs.drop_when_not_align);
  args_size_ = sizeof(GatherDequeueParam);

  const uint64_t queue_ids_offset = args_size_;
  const size_t queue_id_addrs_size = sizeof(uint32_t) * queue_num;
  args_size_ += queue_id_addrs_size;

  const uint64_t mbuf_addrs_offset = args_size_;
  const uint64_t mbuf_addrs_size = sizeof(uint64_t) * queue_num;
  args_size_ += mbuf_addrs_size;

  const uint64_t device_ids_offset = args_size_;
  const size_t device_ids_size = sizeof(uint32_t) * queue_num;
  args_size_ += device_ids_size;

  const uint64_t device_type_offset = args_size_;
  const size_t device_type_size = sizeof(uint32_t) * queue_num;
  args_size_ += device_type_size;

  const uint64_t mbufs_offset = args_size_;
  const size_t mbuff_size = sizeof(uint64_t) * queue_num;
  args_size_ += mbuff_size;

  DF_CHK_ACL_RET(aclrtMalloc(&args_, args_size_, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  GE_PRINT_DYNAMIC_MEMORY(aclrtMalloc, "args data.", args_size_);
  kernel_args.queue_ids_addr = PtrToValue(args_) + queue_ids_offset;
  kernel_args.mbuf_addrs_addr = PtrToValue(args_) + mbuf_addrs_offset;
  kernel_args.queue_device_ids_addr = PtrToValue(args_) + device_ids_offset;
  kernel_args.queue_device_type_addr = PtrToValue(args_) + device_type_offset;
  const uint64_t mbufs_addr = PtrToValue(args_) + mbufs_offset;
  std::vector<uint32_t> queue_ids;
  std::vector<uint32_t> device_ids;
  std::vector<uint32_t> device_types;
  
  for (size_t i = 0UL; i < queue_num; ++i) {
    mbuf_addrs.emplace_back(mbufs_addr + sizeof(uint64_t) * i);
    queue_ids.emplace_back(queues[i].queue_id);
    device_ids.emplace_back(queues[i].device_id);
    device_types.emplace_back(queues[i].device_type);
  }
  DF_CHK_ACL_RET(aclrtMemcpy(args_, args_size_, &kernel_args, sizeof(kernel_args), ACL_MEMCPY_HOST_TO_DEVICE));
  DF_CHK_ACL_RET(aclrtMemcpy(ValueToPtr(kernel_args.queue_ids_addr), queue_id_addrs_size,
                         queue_ids.data(), queue_id_addrs_size, ACL_MEMCPY_HOST_TO_DEVICE));
  DF_CHK_ACL_RET(aclrtMemcpy(ValueToPtr(kernel_args.mbuf_addrs_addr), mbuf_addrs_size,
                         mbuf_addrs.data(), mbuf_addrs.size() * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));
  DF_CHK_ACL_RET(aclrtMemcpy(ValueToPtr(kernel_args.queue_device_ids_addr), device_ids_size,
                         device_ids.data(), device_ids_size, ACL_MEMCPY_HOST_TO_DEVICE));
  DF_CHK_ACL_RET(aclrtMemcpy(ValueToPtr(kernel_args.queue_device_type_addr), device_type_size,
                         device_types.data(), device_type_size, ACL_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status SchedTaskModelGatherDequeue::Distribute() {
  GE_CHK_BOOL_RET_STATUS((args_ != nullptr) && (args_size_ != 0UL) && (stream_ != nullptr), FAILED,
                         "Task:%s not initialized, distribute failed, size:%lu.",
                         kSchedTaskModelGatherDequeue, args_size_);
  GE_CHK_RT_RET(LaunchCpuKernel(kSchedTaskModelGatherDequeue));
  GELOGI("Success to launch task:%s.", kSchedTaskModelGatherDequeue);
  return SUCCESS;
}

Status SchedTaskPrepareDynamicInputOutput::Init(const std::vector<uint32_t> &input_dynamic_flags,
                                                const std::vector<uint64_t> &input_mbuf_addrs,
                                                const std::vector<int32_t> &input_fusion_offsets,
                                                const std::vector<int64_t> &output_tensor_sizes,
                                                std::vector<uint64_t> &output_mbuf_addrs,
                                                uint64_t &req_msg_mbuf_addr,const bool enable_v2) {
  enable_v2_ = enable_v2;
  GE_CHK_BOOL_RET_STATUS((args_ == nullptr) && (args_size_ == 0UL), FAILED,
                         "Task:%s has already initialized, size:%lu.",
                         kSchedTaskPrepareDynamicInputOutput, args_size_);
  GE_CHK_BOOL_RET_STATUS(input_dynamic_flags.size() == input_mbuf_addrs.size(), FAILED,
                         "Size of input_dynamic_flags[%zu] mismatch size of input_mbuf_addrs[%zu].",
                         input_dynamic_flags.size(), input_mbuf_addrs.size());
  GE_CHK_BOOL_RET_STATUS(input_fusion_offsets.size() == input_mbuf_addrs.size(), FAILED,
                         "Size of input_fusion_offsets[%zu] mismatch size of input_mbuf_addrs[%zu].",
                         input_fusion_offsets.size(), input_mbuf_addrs.size());
  const size_t inputs_num = input_mbuf_addrs.size();
  const size_t outputs_num = output_tensor_sizes.size();
  // memory layout: kernel_args | input_dynamic_flags buffer | output_tensor_sizes buffer | input_mbuf_addrs buffer |
  //                output_mbuf_addrs buffer | output_mbufs buffer | input_fusion_offsets buffer | req_msg_mbuf buffer
  PrepareDynamicInputOutputKernelArgs kernel_args{};
  kernel_args.inputs_num = static_cast<uint32_t>(inputs_num);
  kernel_args.outputs_num = static_cast<uint32_t>(outputs_num);
  args_size_ = sizeof(kernel_args);
  const uint64_t input_dynamic_flags_offset = args_size_;
  const size_t input_dynamic_flags_size = sizeof(uint32_t) * inputs_num;
  args_size_ += input_dynamic_flags_size;
  const uint64_t output_tensor_sizes_offset = args_size_;
  const size_t output_tensor_sizes_size = sizeof(int64_t) * outputs_num;
  args_size_ += output_tensor_sizes_size;
  const uint64_t input_mbuf_addrs_offset = args_size_;
  const size_t input_mbuf_addrs_size = sizeof(uint64_t) * inputs_num;
  args_size_ += input_mbuf_addrs_size;
  const uint64_t output_mbuf_addrs_offset = args_size_;
  const size_t output_mbuf_addrs_size = sizeof(uint64_t) * outputs_num;
  args_size_ += output_mbuf_addrs_size;
  const uint64_t output_mbufs_offset = args_size_;
  const size_t output_mbufs_size = sizeof(uint64_t) * outputs_num;
  args_size_ += output_mbufs_size;
  const uint64_t input_fusion_offsets_offset = args_size_;
  const size_t input_fusion_offsets_size = sizeof(int32_t) * inputs_num;
  args_size_ += input_fusion_offsets_size;
  const uint64_t req_msg_mbuf_offset = args_size_;
  const size_t req_msg_mbuf_size = sizeof(uint64_t);
  args_size_ += req_msg_mbuf_size;
  DF_CHK_ACL_RET(aclrtMalloc(&args_, args_size_, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  GE_PRINT_DYNAMIC_MEMORY(aclrtMalloc, "args data.", args_size_);
  kernel_args.input_dynamic_flags_addr = PtrToValue(args_) + input_dynamic_flags_offset;
  kernel_args.output_tensor_sizes_addr = PtrToValue(args_) + output_tensor_sizes_offset;
  kernel_args.input_mbuf_addrs_addr = PtrToValue(args_) + input_mbuf_addrs_offset;
  kernel_args.output_mbuf_addrs_addr = PtrToValue(args_) + output_mbuf_addrs_offset;
  kernel_args.input_fusion_offsets_addr = PtrToValue(args_) + input_fusion_offsets_offset;
  kernel_args.req_msg_mbuf_addr = PtrToValue(args_) + req_msg_mbuf_offset;
  const uint64_t output_mbufs_addr = PtrToValue(args_) + output_mbufs_offset;
  for (size_t i = 0UL; i < outputs_num; ++i) {
    output_mbuf_addrs.emplace_back(output_mbufs_addr + sizeof(uint64_t) * i);
  }
  req_msg_mbuf_addr = kernel_args.req_msg_mbuf_addr;
  DF_CHK_ACL_RET(aclrtMemcpy(args_, args_size_,
                         &kernel_args, sizeof(kernel_args), ACL_MEMCPY_HOST_TO_DEVICE));
  DF_CHK_ACL_RET(aclrtMemcpy(ValueToPtr(kernel_args.input_dynamic_flags_addr), input_dynamic_flags_size,
                         input_dynamic_flags.data(), input_dynamic_flags_size, ACL_MEMCPY_HOST_TO_DEVICE));
  DF_CHK_ACL_RET(aclrtMemcpy(ValueToPtr(kernel_args.output_tensor_sizes_addr), output_tensor_sizes_size,
                         output_tensor_sizes.data(), output_tensor_sizes_size, ACL_MEMCPY_HOST_TO_DEVICE));
  DF_CHK_ACL_RET(aclrtMemcpy(ValueToPtr(kernel_args.input_mbuf_addrs_addr), input_mbuf_addrs_size,
                         input_mbuf_addrs.data(), input_mbuf_addrs_size, ACL_MEMCPY_HOST_TO_DEVICE));
  DF_CHK_ACL_RET(aclrtMemcpy(ValueToPtr(kernel_args.input_fusion_offsets_addr), input_fusion_offsets_size,
                         input_fusion_offsets.data(), input_fusion_offsets_size, ACL_MEMCPY_HOST_TO_DEVICE));
  DF_CHK_ACL_RET(aclrtMemcpy(ValueToPtr(kernel_args.output_mbuf_addrs_addr), output_mbufs_size,
                         output_mbuf_addrs.data(), output_mbuf_addrs.size() * sizeof(uint64_t),
                         ACL_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status SchedTaskPrepareDynamicInputOutput::Distribute() {
  const std::string kernel_name = enable_v2_ ? kSchedTaskPrepareDynamicInputOutputV2 :
                                               kSchedTaskPrepareDynamicInputOutput;
  GE_CHK_BOOL_RET_STATUS((args_ != nullptr) && (args_size_ != 0UL) && (stream_ != nullptr), FAILED,
                         "Task:%s not initialized, distribute failed, size:%lu.",
                         kernel_name.c_str(), args_size_);
  GE_CHK_RT_RET(LaunchCpuKernel(kernel_name.c_str()));
  GELOGI("Success to launch task:%s.", kernel_name.c_str());
  return SUCCESS;
}

Status SchedTaskModelBatchEnqueue::Init(const std::vector<uint32_t> &queue_ids,
                                        const std::vector<uint64_t> &mbuf_addrs) {
  GE_CHK_BOOL_RET_STATUS((args_ == nullptr) && (args_size_ == 0UL), FAILED,
                         "Task:%s has already initialized, size:%lu.", kSchedTaskModelBatchEnqueue, args_size_);
  GE_CHK_BOOL_RET_STATUS(queue_ids.size() == mbuf_addrs.size(), FAILED,
                         "Size of queue_ids[%zu] mismatch size of mbuf_addrs[%zu].",
                         queue_ids.size(), mbuf_addrs.size());
  const size_t outputs_num = queue_ids.size();
  // memory layout: kernel_args | queue_ids buffer | mbuf_addrs buffer
  BatchQueueInfoKernelArgs kernel_args{};
  kernel_args.queue_num = static_cast<uint32_t>(outputs_num);
  args_size_ = sizeof(kernel_args);
  const uint64_t queue_ids_offset = args_size_;
  const size_t queue_ids_size = sizeof(uint32_t) * outputs_num;
  args_size_ += queue_ids_size;
  const uint64_t mbuf_addrs_offset = args_size_;
  const size_t mbuf_addrs_size = sizeof(uint64_t) * outputs_num;
  args_size_ += mbuf_addrs_size;
  DF_CHK_ACL_RET(aclrtMalloc(&args_, args_size_, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  GE_PRINT_DYNAMIC_MEMORY(aclrtMalloc, "args data.", args_size_);
  kernel_args.queue_ids_addr = PtrToValue(args_) + queue_ids_offset;
  kernel_args.mbuf_addrs_addr = PtrToValue(args_) + mbuf_addrs_offset;
  DF_CHK_ACL_RET(aclrtMemcpy(args_, args_size_,
                         &kernel_args, sizeof(kernel_args), ACL_MEMCPY_HOST_TO_DEVICE));
  DF_CHK_ACL_RET(aclrtMemcpy(ValueToPtr(kernel_args.queue_ids_addr), queue_ids_size,
                         queue_ids.data(), queue_ids_size, ACL_MEMCPY_HOST_TO_DEVICE));
  DF_CHK_ACL_RET(aclrtMemcpy(ValueToPtr(kernel_args.mbuf_addrs_addr), mbuf_addrs_size,
                         mbuf_addrs.data(), mbuf_addrs_size, ACL_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status SchedTaskModelBatchEnqueue::Distribute() {
  GE_CHK_BOOL_RET_STATUS((args_ != nullptr) && (args_size_ != 0UL) && (stream_ != nullptr), FAILED,
                         "Task:%s not initialized, distribute failed, size:%lu.",
                         kSchedTaskModelBatchEnqueue, args_size_);
  GE_CHK_RT_RET(LaunchCpuKernel(kSchedTaskModelBatchEnqueue));
  GELOGI("Success to launch task:%s.", kSchedTaskModelBatchEnqueue);
  return SUCCESS;
}

Status SchedTaskPostprocessDynamicOutput::Init(const uint64_t resp_msg_mbuf_addr,
                                               const std::vector<uint64_t> &input_mbuf_addrs,
                                               const std::vector<uint64_t> &output_mbuf_addrs,
                                               const std::vector<uint32_t> &output_dynamic_flags,
                                               const RuntimeTensorDesc *output_static_tensor_descs,
                                               const size_t output_static_tensor_num,
                                               const bool enable_v2) {
  enable_v2_ = enable_v2;
  GE_CHK_BOOL_RET_STATUS((args_ == nullptr) && (args_size_ == 0UL), FAILED,
                         "Task:%s has already initialized, size:%lu.", kSchedTaskPostprecessDynamicOutput, args_size_);
  GE_CHK_BOOL_RET_STATUS(output_mbuf_addrs.size() == output_dynamic_flags.size(), FAILED,
                         "Size of output_mbuf_addrs[%zu] mismatch size of output_dynamic_flag[%zu].",
                         output_mbuf_addrs.size(), output_dynamic_flags.size());
  const size_t dynamic_outputs_num = std::count_if(output_dynamic_flags.begin(), output_dynamic_flags.end(),
                                                   [](uint32_t flag) { return (flag == kDynamicFlag) ||
                                                                               (flag == kDynamicFlagV2);});
  GE_CHK_BOOL_RET_STATUS(dynamic_outputs_num + output_static_tensor_num == output_mbuf_addrs.size(), FAILED,
                         "Size of output_static_tensor_descs[%zu] + dynamic_outputs_num[%zu] mismatch "
                         "size of output_mbuf_addrs[%zu].",
                         output_static_tensor_num, dynamic_outputs_num, output_mbuf_addrs.size());
  if (output_static_tensor_num != 0UL) {
    GE_CHECK_NOTNULL(output_static_tensor_descs);
  }
  const size_t outputs_num = output_mbuf_addrs.size();
  const size_t inputs_num = input_mbuf_addrs.size();
  // memory layout: kernel args | input_mbuf_addrs buffer | output_mbuf_addrs buffer | output_dynamic_flag buffer |
  //                output_static_tensor_descs buffer
  PostprocessDynamicOutputKernelArgs kernel_args{};
  kernel_args.outputs_num = outputs_num;
  kernel_args.inputs_num = inputs_num;
  kernel_args.resp_msg_mbuf_addr = resp_msg_mbuf_addr;
  args_size_ = sizeof(kernel_args);
  const uint64_t input_mbuf_addrs_offset = args_size_;
  const size_t input_mbuf_addrs_size = sizeof(uint64_t) * inputs_num;
  args_size_ += input_mbuf_addrs_size;
  const uint64_t output_mbuf_addrs_offset = args_size_;
  const size_t output_mbuf_addrs_size = sizeof(uint64_t) * outputs_num;
  args_size_ += output_mbuf_addrs_size;
  const uint64_t output_dynamic_flags_offset = args_size_;
  const size_t output_dynamic_flags_size = sizeof(uint32_t) * outputs_num;
  args_size_ += output_dynamic_flags_size;
  const uint64_t output_static_tensor_descs_offset = args_size_;
  const size_t output_static_tensor_descs_size = sizeof(RuntimeTensorDesc) * output_static_tensor_num;
  args_size_ += output_static_tensor_descs_size;
  DF_CHK_ACL_RET(aclrtMalloc(&args_, args_size_, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  GE_PRINT_DYNAMIC_MEMORY(aclrtMalloc, "args data.", args_size_);
  kernel_args.input_mbuf_addrs_addr = PtrToValue(args_) + input_mbuf_addrs_offset;
  kernel_args.output_mbuf_addrs_addr = PtrToValue(args_) + output_mbuf_addrs_offset;
  kernel_args.output_dynamic_flags_addr = PtrToValue(args_) + output_dynamic_flags_offset;
  kernel_args.output_static_tensor_desc_addr = PtrToValue(args_) + output_static_tensor_descs_offset;
  DF_CHK_ACL_RET(aclrtMemcpy(args_, args_size_,
                         &kernel_args, sizeof(kernel_args), ACL_MEMCPY_HOST_TO_DEVICE));
  DF_CHK_ACL_RET(aclrtMemcpy(ValueToPtr(kernel_args.input_mbuf_addrs_addr), input_mbuf_addrs_size,
                         input_mbuf_addrs.data(), input_mbuf_addrs_size, ACL_MEMCPY_HOST_TO_DEVICE));
  DF_CHK_ACL_RET(aclrtMemcpy(ValueToPtr(kernel_args.output_mbuf_addrs_addr), output_mbuf_addrs_size,
                         output_mbuf_addrs.data(), output_mbuf_addrs_size, ACL_MEMCPY_HOST_TO_DEVICE));
  DF_CHK_ACL_RET(aclrtMemcpy(ValueToPtr(kernel_args.output_dynamic_flags_addr), output_dynamic_flags_size,
                         output_dynamic_flags.data(), output_dynamic_flags_size, ACL_MEMCPY_HOST_TO_DEVICE));
  if (output_static_tensor_num != 0) {
    DF_CHK_ACL_RET(aclrtMemcpy(ValueToPtr(kernel_args.output_static_tensor_desc_addr), output_static_tensor_descs_size,
                           output_static_tensor_descs, output_static_tensor_descs_size, ACL_MEMCPY_HOST_TO_DEVICE));
  }
  return SUCCESS;
}

Status SchedTaskPostprocessDynamicOutput::Distribute() {
  const std::string kernel_name = enable_v2_ ? kSchedTaskPostprecessDynamicOutputV2 :
                                               kSchedTaskPostprecessDynamicOutput;
  GE_CHK_BOOL_RET_STATUS((args_ != nullptr) && (args_size_ != 0UL) && (stream_ != nullptr), FAILED,
                         "Task:%s not initialized, distribute failed, size:%lu.",
                         kernel_name.c_str(), args_size_);
  GE_CHK_RT_RET(LaunchCpuKernel(kernel_name.c_str()));
  GELOGI("Success to launch task:%s.", kernel_name.c_str());
  return SUCCESS;
}

Status SchedTaskNotifyWait::Init(const uint32_t notify_id) {
  GE_CHK_BOOL_RET_STATUS((args_ == nullptr) && (args_size_ == 0UL), FAILED,
                         "Task:%s has already initialized, size:%lu.", kSchedTaskNotifyWait, args_size_);
  args_size_ = sizeof(AicpuNotifyKernelArgs);
  DF_CHK_ACL_RET(aclrtMalloc(&args_, args_size_, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  GE_PRINT_DYNAMIC_MEMORY(aclrtMalloc, "args data.", args_size_);
  AicpuNotifyKernelArgs notify_param{};
  notify_param.notify_id = notify_id;
  GE_CHK_RT_RET(aclrtMemcpy(args_, args_size_, &notify_param, sizeof(notify_param), ACL_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status SchedTaskNotifyWait::Distribute() {
  GE_CHK_BOOL_RET_STATUS((args_ != nullptr) && (args_size_ != 0UL) && (stream_ != nullptr), FAILED,
                         "Task:%s not initialized, distribute failed, size:%lu.", kSchedTaskNotifyWait, args_size_);
  GE_CHK_RT_RET(LaunchCpuKernel(kSchedTaskNotifyWait));
  GELOGI("Success to launch task:%s.", kSchedTaskNotifyWait);
  return SUCCESS;
}

Status SchedTaskNotifyRecord::Distribute() {
  GE_CHK_BOOL_RET_STATUS((args_ != nullptr) && (args_size_ != 0UL) && (stream_ != nullptr), FAILED,
                         "Task:%s not initialized, distribute failed, size:%lu.",
                         kSchedTaskNotifyRecord, args_size_);
  GE_CHK_RT_RET(LaunchCpuKernel(kSchedTaskNotifyRecord));
  GELOGI("Success to launch task:%s.", kSchedTaskNotifyRecord);
  return SUCCESS;
}

Status SchedTaskZeroCopy::Init(const std::vector<uint64_t> &src_addrs, std::vector<uint64_t> &dst_addrs) {
  GE_CHK_BOOL_RET_STATUS((args_ == nullptr) && (args_size_ == 0UL), FAILED,
                         "Task:%s has already initialized, size:%lu.", kSchedTaskZeroCopy, args_size_);
  const size_t addrs_num = src_addrs.size();
  // memory layout: kernel args | src_addrs buffer | dst_addrs buffer | dsts buffer
  AddrMapInfoKernelArgs kernel_args{};
  kernel_args.addr_num = addrs_num;
  args_size_ = sizeof(AddrMapInfoKernelArgs);
  const uint64_t src_addrs_offset = args_size_;
  const size_t src_addrs_size = sizeof(uint64_t) * addrs_num;
  args_size_ += src_addrs_size;
  const uint64_t dst_addrs_offset = args_size_;
  const size_t dst_addrs_size = sizeof(uint64_t) * addrs_num;
  args_size_ += dst_addrs_size;
  const uint64_t dsts_offset = args_size_;
  const size_t dsts_size = sizeof(uint64_t) * addrs_num;
  args_size_ += dsts_size;
  DF_CHK_ACL_RET(aclrtMalloc(&args_, args_size_, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  GE_PRINT_DYNAMIC_MEMORY(aclrtMalloc, "args data.", args_size_);
  kernel_args.src_addr_list = PtrToValue(args_) + src_addrs_offset;
  kernel_args.dst_addr_list = PtrToValue(args_) + dst_addrs_offset;
  const uint64_t dsts_addr = PtrToValue(args_) + dsts_offset;
  for (size_t i = 0UL; i < addrs_num; ++i) {
    dst_addrs.emplace_back(dsts_addr + sizeof(uint64_t) * i);
  }
  DF_CHK_ACL_RET(aclrtMemcpy(args_, args_size_, &kernel_args, sizeof(kernel_args), ACL_MEMCPY_HOST_TO_DEVICE));
  DF_CHK_ACL_RET(aclrtMemcpy(ValueToPtr(kernel_args.src_addr_list), src_addrs_size,
                         src_addrs.data(), src_addrs_size, ACL_MEMCPY_HOST_TO_DEVICE));
  DF_CHK_ACL_RET(aclrtMemcpy(ValueToPtr(kernel_args.dst_addr_list), dst_addrs_size,
                         dst_addrs.data(), dst_addrs.size() * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status SchedTaskZeroCopy::Distribute() {
  GE_CHK_BOOL_RET_STATUS((args_ != nullptr) && (args_size_ != 0UL) && (stream_ != nullptr), FAILED,
                         "Task:%s not initialized, distribute failed, size:%lu.", kSchedTaskZeroCopy, args_size_);
  GE_CHK_RT_RET(LaunchCpuKernel(kSchedTaskZeroCopy));
  GELOGI("Success to launch task:%s.", kSchedTaskZeroCopy);
  return SUCCESS;
}

Status SchedTaskStreamRepeat::Init(const uint32_t model_id) {
  GE_CHK_BOOL_RET_STATUS((args_ == nullptr) && (args_size_ == 0UL), FAILED,
                         "Task:%s has already initialized, size:%lu.", kSchedTaskStreamRepeat, args_size_);
  int32_t stream_id = kInvalidStreamId;
  DF_CHK_ACL_RET(aclrtStreamGetId(stream_, &stream_id));
  StreamRepeatKernelArgs kernel_args = {
      .model_id = model_id,
      .stream_id = static_cast<uint32_t>(stream_id),
  };
  args_size_ = sizeof(kernel_args);
  DF_CHK_ACL_RET(aclrtMalloc(&args_, args_size_, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  GE_PRINT_DYNAMIC_MEMORY(aclrtMalloc, "args data.", args_size_);
  DF_CHK_ACL_RET(aclrtMemcpy(args_, args_size_, &kernel_args, sizeof(kernel_args), ACL_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status SchedTaskStreamRepeat::Distribute() {
  GE_CHK_BOOL_RET_STATUS((args_ != nullptr) && (args_size_ != 0UL) && (stream_ != nullptr), FAILED,
                         "Task:%s not initialized, distribute failed, size:%lu.",
                         kSchedTaskStreamRepeat, args_size_);
  GE_CHK_RT_RET(LaunchCpuKernel(kSchedTaskStreamRepeat));
  GELOGI("Success to launch task:%s.", kSchedTaskStreamRepeat);
  return SUCCESS;
}

Status SchedTaskMarkStep::Init(const uint32_t group_total_count, const uint32_t group_index,
                               const uint32_t group_policy, const std::string &dump_step,
                               const uint64_t step_id_addr) {
  GE_CHK_BOOL_RET_STATUS((args_ == nullptr) && (args_size_ == 0UL), FAILED,
                         "Task:%s has already initialized, size:%lu.", kSchedTaskMarkStep, args_size_);
  args_size_ = sizeof(MarkStepKernelArgs);
  DF_CHK_ACL_RET(aclrtMalloc(&args_, args_size_, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  GE_PRINT_DYNAMIC_MEMORY(aclrtMalloc, "args data.", args_size_);
  MarkStepKernelArgs kernel_args{};
  kernel_args.group_total_count = group_total_count;
  kernel_args.group_index = group_index;
  kernel_args.group_policy = group_policy;
  kernel_args.step_id_addr = step_id_addr;
  const auto ret = strcpy_s(kernel_args.dump_step, sizeof(kernel_args.dump_step), dump_step.c_str());
  if (ret != EOK) {
    GELOGE(FAILED, "Fail to call strcpy_s, result: %d, dump_step: %s.", ret, dump_step.c_str());
    return FAILED;
  }
  DF_CHK_ACL_RET(aclrtMemcpy(args_, args_size_, &kernel_args, sizeof(MarkStepKernelArgs), ACL_MEMCPY_HOST_TO_DEVICE));
  GELOGI("Mark step, group_total_count:%u, group_index:%u, group_policy:%u, step_id_addr:0x%lx, dump_step:%s.",
         group_total_count, group_index, group_policy, step_id_addr, dump_step.c_str());
  return SUCCESS;
}

Status SchedTaskMarkStep::Distribute() {
  GE_CHK_BOOL_RET_STATUS((args_ != nullptr) && (args_size_ != 0UL) && (stream_ != nullptr), FAILED,
                         "Task:%s not initialized, distribute failed, size:%lu.",
                         kSchedTaskMarkStep, args_size_);
  GE_CHK_RT_RET(LaunchCpuKernel(kSchedTaskMarkStep));
  GELOGI("Success to launch task:%s.", kSchedTaskMarkStep);
  return SUCCESS;
}
}
