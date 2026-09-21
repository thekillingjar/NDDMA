/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_REGISTER_KERNEL_LAUNCH_INFO_IMPL_H
#define METADEF_REGISTER_KERNEL_LAUNCH_INFO_IMPL_H

#include "graph/kernel_launch_info.h"
#include "proto/task.pb.h"

namespace ge {
constexpr uint32_t TASK_CCU_KERNEL_SQE_NUM = 4;
class KernelLaunchInfoImpl {
 public:
  ~KernelLaunchInfoImpl() = default;
  KernelLaunchInfoImpl() = default;
  static KernelLaunchInfoImplPtr LoadFromData(const gert::ExeResGenerationContext *context,
      const std::vector<uint8_t> &data);
  static KernelLaunchInfoImplPtr CreateAicpuKfcTask(const gert::ExeResGenerationContext *context,
      const char *so_name, const char *kernel_name);
  static KernelLaunchInfoImplPtr CreateHcomRecordTask(const gert::ExeResGenerationContext *context,
      const char *group_name);
  static KernelLaunchInfoImplPtr CreateHcomWaitTask(const gert::ExeResGenerationContext *context,
      const char *group_name);
  static KernelLaunchInfoImplPtr CreateCcuTask(const gert::ExeResGenerationContext *context, const std::vector<std::string>& groups);
  static KernelLaunchInfoImplPtr CreateFusionTask(const gert::ExeResGenerationContext *context,
        const std::vector<const KernelLaunchInfoImpl*>& sub_tasks);
  std::vector<uint8_t> Serialize() const;
  uint32_t GetStreamId() const;
  void SetStreamId(uint32_t stream_id);
  uint32_t GetBlockDim() const;
  graphStatus SetBlockDim(uint32_t block_dim);
  const char *GetArgsFormat() const;
  graphStatus SetArgsFormat(const char *args_format);
  const char *GetSoName() const;
  const char *GetKernelName() const;
 private:
  static bool ProcessFusionTask(const gert::ExeResGenerationContext* context, const std::vector<const KernelLaunchInfoImpl*>& sub_tasks,
      KernelLaunchInfoImpl* impl_ptr);
  static bool ProcessFusionSubTask(const KernelLaunchInfoImpl* sub_task, domi::FusionTaskDef* fusion_task_def);
  static bool ProcessAicoreFusionSubTask(const KernelLaunchInfoImpl* sub_task, domi::FusionSubTaskInfo* sub_task_info);
  static bool ProcessCcuFusionSubTask(const KernelLaunchInfoImpl* sub_task, domi::FusionSubTaskInfo* sub_task_info);
  static void SetBlockDimForAicoreFusionSubTask(const KernelLaunchInfoImpl* sub_task, domi::FusionSubTaskInfo* sub_task_info);

 private:
  domi::TaskDef task_def_;
  gert::ExeResGenerationContext *context_;
};
}
#endif // METADEF_REGISTER_KERNEL_LAUNCH_INFO_IMPL_H