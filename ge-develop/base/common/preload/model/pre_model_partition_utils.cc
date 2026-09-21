/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/math/math_util.h"
#include "common/preload/model/pre_model_partition_utils.h"

namespace ge {
void PreModelPartitionUtils::Reset() {
  pre_task_desc_info_.clear();
  kernel_args_info_.clear();
  partition_kernel_args_param_.clear();
  kernel_args_info_size_ = 0UL;
  task_build_buf_ = nullptr;
  nano_partition_type_to_buf_.clear();
  total_tlv_len_ = 0U;
}

void PreModelPartitionUtils::AddPreTaskDescInfo(const std::vector<PreTaskDescInfo> &pre_task_desc_infos) {
  (void)pre_task_desc_info_.insert(pre_task_desc_info_.cend(), pre_task_desc_infos.cbegin(),
                                   pre_task_desc_infos.cend());
}

void PreModelPartitionUtils::AddNanoHostFuncParamData(const std::shared_ptr<uint8_t> &nano_hostfunc_param_data) {
  (void)nano_hostfunc_param_data_.push_back(nano_hostfunc_param_data);
}

Status PreModelPartitionUtils::PreparePartitionData(const EngineType type) {
  GELOGI("begin prepare partition data, pre_task_desc_info_ size:%zu", pre_task_desc_info_.size());
  GE_ASSERT_TRUE((pre_task_desc_info_.size() > 0), "EngineType : %u. Partition data is empty, nano unsupport.",
                 static_cast<uint32_t>(type));
  std::vector<PreTaskDescInfo> zero_copy_task;
  for (auto task_desc : pre_task_desc_info_) {
    if (task_desc.kernel_args_desc_info.zero_copy_data.size() == 0U) {
      if (type == EngineType::kDefaultEngine) {
        GE_CHK_STATUS_RET(PrepareDefaultPartitionData(task_desc), "[Call][PreparePartitionData] failed.");
      } else {
        GE_CHK_STATUS_RET(PrepareNanoPartitionData(task_desc), "[Call][PreparePartitionData] failed.");
      }
    } else {
      GELOGD("need to do zero copy.");
      (void)zero_copy_task.emplace_back(task_desc);
    }
  }
  // zero copy task need to move to last position
  for (auto task_desc : zero_copy_task) {
    GE_CHK_STATUS_RET(PrepareDefaultPartitionData(task_desc), "[Call][PreparePartitionData] failed.");
  }
  GELOGI("success prepare partition data.");
  return SUCCESS;
}

Status PreModelPartitionUtils::InitTaskBuildMem(const uint32_t huge_stream_size, const uint32_t stream_size) {
  GELOGI("begin init task build mem.");
  constexpr uint32_t huge_buf_len = 0U;
  constexpr uint32_t default_buf_len = 1U;
  GELOGI("Init mem info huge_stream_size:%u, huge_buf_len:%u, stream_size:%u, default_buf_len:%u.", huge_stream_size,
         huge_buf_len, stream_size, default_buf_len);
  uint32_t orgi_size = (huge_stream_size * huge_buf_len) + ((stream_size - huge_stream_size) * default_buf_len);
  GE_ASSERT_TRUE(!(orgi_size == 0U), "[Call] orgi_size is 0.");
  GE_ASSERT_SUCCESS(GenTaskBuildBuf(task_build_buf_, orgi_size));
  GELOGI("success init task build mem.");
  return SUCCESS;
}

Status PreModelPartitionUtils::CheckNanoPartitionType(const uint8_t type) const {
  if ((type == static_cast<uint8_t>(HWTS_STATIC_TASK_DESC)) || (type == static_cast<uint8_t>(HWTS_DYNAMIC_TASK_DESC)) ||
      (type == static_cast<uint8_t>(PARAM_TASK_INFO_DESC))) {
    return SUCCESS;
  }
  GELOGE(FAILED, "unsupport taskBuffType : %u", static_cast<uint32_t>(type));
  return FAILED;
}

Status PreModelPartitionUtils::InitTaskBuildMem(const uint32_t task_num) {
  GELOGI("begin init task build mem.");
  for (const PreTaskDescInfo &task_desc : pre_task_desc_info_) {
    if (task_desc.seq_info.taskType == RT_TASK_TYPE_KERNEL_NANO_AICPU_HOSTFUNC) {
      total_tlv_len_ += task_desc.seq_info.u.nanoHostFuncTask.u.paramBufDesc.bufSize;
    }
  }
  GELOGI("total_tlv_len = %u", total_tlv_len_);

  GE_CHK_STATUS_RET(InitTaskBuildMem(HWTS_STATIC_TASK_DESC, task_num),
                    "[Call][InitTaskBuildMem] HWTS_STATIC_TASK_DESC init failed.");
  GE_CHK_STATUS_RET(InitTaskBuildMem(HWTS_DYNAMIC_TASK_DESC, task_num),
                    "[Call][InitTaskBuildMem] HWTS_DYNAMIC_TASK_DESC init failed.");
  GE_CHK_STATUS_RET(InitTaskBuildMem(PARAM_TASK_INFO_DESC, task_num),
                    "[Call][InitTaskBuildMem] PARAM_TASK_INFO_DESC init failed.");
  GELOGI("success init task build mem.");
  return SUCCESS;
}

Status PreModelPartitionUtils::InitTaskBuildMem(const tagRtTaskBuffType type, const uint32_t task_num) {
  GELOGI("begin init task build mem.");
  uint32_t default_buf_len = 0U;
  uint32_t orgi_size = 0U;
  GE_CHK_RT_RET(rtGetTaskBufferLen(type, &default_buf_len));
  GELOGI("Init mem info task_num:%u, default_buf_len:%u.", task_num, default_buf_len);
  GE_ASSERT_SUCCESS(CheckUint32MulOverflow(task_num, default_buf_len),
                    "orgi_size overflow! task_num[%u], default_buf_len[%u],", task_num, default_buf_len);
  orgi_size = task_num * default_buf_len;
  orgi_size += ((type == PARAM_TASK_INFO_DESC) ? total_tlv_len_ : 0U);
  GE_ASSERT_TRUE(!(orgi_size == 0U), "[Call] orgi_size is 0.");
  GE_ASSERT_SUCCESS(GenTaskBuildBuf(nano_partition_type_to_buf_[type], orgi_size));
  GELOGI("success init task build mem.");
  return SUCCESS;
}

Status PreModelPartitionUtils::GenModelPartitionBuf(const uint8_t type, const uint32_t orgi_size) {
  GELOGI("begin init nano model partition buf, type[%u] size[%u].", static_cast<uint32_t>(type), orgi_size);

  std::shared_ptr<TaskBuildBuf> model_partition_buf = std::make_shared<TaskBuildBuf>();
  GE_ASSERT_NOTNULL(model_partition_buf);
  model_partition_buf->orgi_size = orgi_size;
  model_partition_buf->used_size = 0U;
  model_partition_buf->buf.reset(new (std::nothrow) uint8_t[orgi_size], std::default_delete<uint8_t[]>());
  GE_ASSERT_NOTNULL(model_partition_buf->buf, "[Create][GenModelPartitionBuf]reset buf failed, it is nullptr");

  nano_model_partition_type_to_buf_[type] = model_partition_buf;
  GELOGD("success generate model partition buf.");
  return SUCCESS;
}

Status PreModelPartitionUtils::SaveNanoModelPartitionData(const uint8_t type, const void *const src,
                                                          const uint32_t len) {
  std::shared_ptr<TaskBuildBuf> model_buf = nano_model_partition_type_to_buf_[type];
  GE_ASSERT_NOTNULL(model_buf);
  const uint32_t orgi_size = model_buf->orgi_size;
  const uint32_t used_size = model_buf->used_size;
  GE_ASSERT_TRUE((model_buf->buf != nullptr) && (orgi_size >= used_size + len),
                 "ModelPartitionType: %u, buff error. orgi_size: %u, used_size: %u, len: %u",
                 static_cast<uint32_t>(type), model_buf->orgi_size, model_buf->used_size, len);
  uint8_t *des_addr =
      PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(model_buf->buf.get()) + static_cast<size_t>(used_size)));
  const auto ret = memcpy_s(des_addr, static_cast<size_t>(orgi_size - used_size), src, static_cast<size_t>(len));
  GE_ASSERT_EOK(ret, "memcpy_s failed, return: %d", ret);
  model_buf->used_size += len;

  return SUCCESS;
}

void PreModelPartitionUtils::GetNanoModelPartitionData(const uint8_t type, uint8_t **data,
                                                       uint32_t &len) {
  std::shared_ptr<TaskBuildBuf> model_buf = nano_model_partition_type_to_buf_[type];
  if (model_buf->buf == nullptr) {
    GELOGD("model buf is nullptr, type %u.", static_cast<uint32_t>(type));
    return;
  }
  *data = model_buf->buf.get();
  len = model_buf->used_size;
}

Status PreModelPartitionUtils::GenTaskBuildBuf(std::shared_ptr<TaskBuildBuf> &build_buf,
                                               const uint32_t orgi_size) const {
  GELOGD("Init generate task build buf, orgi_size:%u.", orgi_size);
  std::shared_ptr<TaskBuildBuf> task_build_buf(new (std::nothrow) TaskBuildBuf());
  GE_ASSERT_NOTNULL(task_build_buf);
  task_build_buf->orgi_size = orgi_size;
  task_build_buf->used_size = 0U;
  task_build_buf->buf.reset(new (std::nothrow) uint8_t[orgi_size], std::default_delete<uint8_t[]>());
  GE_ASSERT_NOTNULL(task_build_buf->buf, "[Create][GenTaskBuildBuf]reset buf failed, it is nullptr");
  build_buf = task_build_buf;
  GELOGD("success generate task build buf.");
  return SUCCESS;
}

Status PreModelPartitionUtils::PrepareDefaultPartitionData(const PreTaskDescInfo &task_desc) {
  // 1. update sqe info
  GE_CHK_STATUS_RET(UpdateSqeInfo(task_desc, task_build_buf_), "[Call][UpdateSqeInfo] update failed.");

  // 2. update KernelArgsInfo
  (void)kernel_args_info_.emplace_back(task_desc.kernel_args_info);

  // 3. update KernelArgsDescInfo
  PartitionKernelArgsParam partition_kernel_arg;
  for (const auto kernel_arg : task_desc.kernel_args_desc_info.kernel_args_desc_data) {
    partition_kernel_arg.type = kernel_arg.type;
    partition_kernel_arg.offset =
        kernel_arg.offset.need_refresh ? (kernel_args_info_size_ + kernel_arg.offset.offset) : kernel_arg.offset.offset;
    partition_kernel_arg.para = kernel_arg.para;
    (void)partition_kernel_args_param_.emplace_back(partition_kernel_arg);
  }

  // 4. refresh to next task offset
  kernel_args_info_size_ += task_desc.kernel_args_info.kernel_args_data_size;
  return SUCCESS;
}

Status PreModelPartitionUtils::PrepareNanoPartitionData(const PreTaskDescInfo &task_desc) {
  const auto type = task_desc.seq_info.u.nanoAicoreTask.type;
  GE_ASSERT_SUCCESS(CheckNanoPartitionType(static_cast<uint8_t>(type)));
  // 1. update sqe info
  GE_CHK_STATUS_RET(UpdateSqeInfo(task_desc, nano_partition_type_to_buf_[static_cast<uint8_t>(type)]),
                    "[Call][UpdateSqeInfo] update failed.");

  // 2. refresh to next task offset
  if (type == PARAM_TASK_INFO_DESC) {
    kernel_args_info_size_ = nano_partition_type_to_buf_[static_cast<uint8_t>(type)]->used_size;
  }
  return SUCCESS;
}

Status PreModelPartitionUtils::UpdateSqeInfo(const PreTaskDescInfo &task_desc,
                                             std::shared_ptr<TaskBuildBuf> task_build_buf) const {
  rtTaskInput_t task_input;
  GE_ASSERT_NOTNULL(task_build_buf);
  GELOGD("begin to update sqe info, task_build_buf used size %u.", task_build_buf->used_size);
  task_input.compilerInfo = task_desc.seq_info;
  task_input.argOffset = kernel_args_info_size_;
  task_input.dataBuffer = PtrToPtr<void, uint8_t>(
      ValueToPtr(PtrToValue(task_build_buf->buf.get()) + static_cast<uint64_t>(task_build_buf->used_size)));
  task_input.bufferLen = task_build_buf->orgi_size - task_build_buf->used_size;
  uint32_t task_len = 0U;
  GE_CHK_RT_RET(rtTaskBuild(&task_input, &task_len));
  GE_ASSERT_TRUE((task_build_buf->used_size + task_len <= task_build_buf->orgi_size),
                 "task_len overflow! used size is %d, task_len is %d, orgin size is %d", task_build_buf->used_size,
                 task_len, task_build_buf->orgi_size);
  task_build_buf->used_size += task_len;
  GELOGD("success update sqe info, task_build_buf used size %u.", task_build_buf->used_size);
  return SUCCESS;
}
}  // namespace ge