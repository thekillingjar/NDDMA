/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GE_COMMON_PRELOAD_PRE_MODEL_PARTITION_UTILS_H_
#define GE_COMMON_PRELOAD_PRE_MODEL_PARTITION_UTILS_H_
#include "common/preload/model/pre_model_utils.h"
#include "common/preload/model/pre_model_types.h"

namespace ge {
struct PartitionKernelArgsParam {
  uint8_t type;
  uint64_t offset;
  uint64_t para;
};

struct TaskBuildBuf {
  std::shared_ptr<uint8_t> buf;
  uint32_t orgi_size;
  uint32_t used_size;
};

class PreModelPartitionUtils {
 public:
  PreModelPartitionUtils() = default;
  ~PreModelPartitionUtils() = default;
  static PreModelPartitionUtils &GetInstance() {
    static PreModelPartitionUtils instance;
    return instance;
  }
  void Reset();
  void AddPreTaskDescInfo(const std::vector<PreTaskDescInfo> &pre_task_desc_infos);
  Status InitTaskBuildMem(const uint32_t huge_stream_size, const uint32_t stream_size);
  Status InitTaskBuildMem(const uint32_t task_num);
  Status PreparePartitionData(const EngineType type);
  Status CheckNanoPartitionType(const uint8_t type) const;
  Status GenModelPartitionBuf(const uint8_t type, const uint32_t orgi_size);
  Status SaveNanoModelPartitionData(const uint8_t type, const void *const src, const uint32_t len);
  void GetNanoModelPartitionData(const uint8_t type, uint8_t **data, uint32_t &len);
  std::shared_ptr<TaskBuildBuf> &GetTaskBuildBuf() { return task_build_buf_; }
  std::shared_ptr<TaskBuildBuf> &GetNanoTaskBuildBuf(const uint8_t type) {
    return nano_partition_type_to_buf_[type];
  }
  void AddNanoHostFuncParamData(const std::shared_ptr<uint8_t> &nano_hostfunc_param_data);
  const std::unordered_map<int64_t, uint32_t> GetZeroCopyTable() {
    return zero_copy_offset_to_ids_;
  }

  void SetZeroCopyTable(const int64_t key, const uint32_t value) {
    zero_copy_offset_to_ids_[key] = value;
  }

  void AddGlobalDataTensorSize(const uint64_t size) {
    (void)global_data_tensor_size_.push_back(size);
  }

  const std::vector<uint64_t> &GetGlobalDataTensorSizes() const {
    return global_data_tensor_size_;
  }

 private:
  Status PrepareDefaultPartitionData(const PreTaskDescInfo &task_desc);
  Status PrepareNanoPartitionData(const PreTaskDescInfo &task_desc);
  Status UpdateSqeInfo(const PreTaskDescInfo &task_desc, std::shared_ptr<TaskBuildBuf> task_build_buf) const;
  Status InitTaskBuildMem(const tagRtTaskBuffType type, const uint32_t task_num);
  Status GenTaskBuildBuf(std::shared_ptr<TaskBuildBuf> &build_buf, const uint32_t orgi_size) const;
  std::vector<PreTaskDescInfo> pre_task_desc_info_;
  std::vector<KernelArgsInfo> kernel_args_info_;
  std::vector<PartitionKernelArgsParam> partition_kernel_args_param_;
  uint64_t kernel_args_info_size_ = 0UL;
  std::shared_ptr<TaskBuildBuf> task_build_buf_ = nullptr;
  std::unordered_map<uint8_t, std::shared_ptr<TaskBuildBuf>> nano_partition_type_to_buf_;
  std::unordered_map<uint8_t, std::shared_ptr<TaskBuildBuf>> nano_model_partition_type_to_buf_;
  uint32_t total_tlv_len_ = 0U;
  std::vector<std::shared_ptr<uint8_t>> nano_hostfunc_param_data_;
  std::unordered_map<int64_t, uint32_t> zero_copy_offset_to_ids_;
  std::vector<uint64_t> global_data_tensor_size_;
};
}  // namespace ge
#endif  // GE_COMMON_PRELOAD_PRE_MODEL_PARTITION_UTILS_H_