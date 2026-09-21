/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_generator_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/debug/ge_log.h"
namespace ge {
bool NoNeedGenTask(const OpDescPtr &op_desc) {
  bool no_need_gen_task = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_NOTASK, no_need_gen_task);
  if (no_need_gen_task) {
    GELOGD("Node[name:%s, type:%s] does not need to generate task.", op_desc->GetName().c_str(),
           op_desc->GetType().c_str());
    return true;
  }
  std::string op_kernel_lib_name = op_desc->GetOpKernelLibName();
  if (op_kernel_lib_name.empty()) {
    GELOGW("Node[name:%s, type:%s] does not need to generate task.", op_desc->GetName().c_str(),
           op_desc->GetType().c_str());
    return true;
  }
  return false;
}

void RefreshTaskDefStreamId(bool has_attached_stream, int64_t logical_stream_id, int64_t real_stream_id,
                            std::vector<domi::TaskDef> &task_defs) {
  if (has_attached_stream) {
    for (auto &task_def : task_defs) {
      auto task_def_origin_stream_id = static_cast<int64_t>(task_def.stream_id());
      if ((task_def_origin_stream_id == logical_stream_id) && (task_def_origin_stream_id != real_stream_id)) {
        task_def.set_stream_id(static_cast<uint32_t>(real_stream_id));
      }
    }
  } else {
    for (auto &task_def : task_defs) {
      task_def.set_stream_id(static_cast<uint32_t>(real_stream_id));
    }
  }
}
}