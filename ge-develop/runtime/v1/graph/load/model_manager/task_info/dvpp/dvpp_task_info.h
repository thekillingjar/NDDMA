/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_DVPP_TASK_INFO_H_
#define GE_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_DVPP_TASK_INFO_H_

#include "graph/load/model_manager/task_info/task_info.h"
#include "common/opskernel/ops_kernel_info_store.h"
#include "common/opskernel/ge_task_info.h"

namespace ge {
class DvppTaskInfo : public TaskInfo {
 public:
  DvppTaskInfo() = default;

  ~DvppTaskInfo() override = default;

  Status Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
              const PisToArgs &args = {}, const PisToPersistentWorkspace &persistent_workspace = {},
              const IowAddrs &iow_addrs = {{}, {}, {}}) override;

  Status Distribute() override;

 private:
  OpsKernelInfoStore *ops_kernel_store_{nullptr};
  std::vector<void *> io_addrs_;
  DvppInfo dvpp_info_;
  GETaskInfo ge_task_;
};
}  // namespace ge

#endif  // GE_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_DVPP_TASK_INFO_H_
