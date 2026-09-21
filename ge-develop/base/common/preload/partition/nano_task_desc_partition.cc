/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/preload/partition/nano_task_desc_partition.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
Status NanoTaskDescPartition::Init(const GeModelPtr &ge_model, const uint8_t type) {
  (void)ge_model;
  GELOGD("begin to generate task desc, type:%u.", static_cast<uint32_t>(type));
  const auto &task_build_buf = PreModelPartitionUtils::GetInstance().GetNanoTaskBuildBuf(type);
  GE_ASSERT_NOTNULL(task_build_buf);
  const auto task_desc = task_build_buf->buf;
  GE_CHECK_GE(task_build_buf->orgi_size, task_build_buf->used_size);
  GE_CHECK_NOTNULL(task_desc);
  buff_size_ = task_build_buf->used_size;
  GE_CHECK_SIZE(buff_size_);
  buff_.reset(new (std::nothrow) uint8_t[buff_size_], std::default_delete<uint8_t[]>());
  GE_CHECK_NOTNULL(buff_);

  const auto ret =
      memcpy_s(buff_.get(), static_cast<uint64_t>(buff_size_), task_desc.get(), static_cast<uint64_t>(buff_size_));
  GE_ASSERT_EOK(ret, "[Operate][Init]Failed at offset %u, error-code %d", buff_size_, ret);

  GELOGD("end generate dynamic task desc, type:%u, buff_size_:%u", static_cast<uint32_t>(type), buff_size_);
  return SUCCESS;
}
}  // namespace ge
