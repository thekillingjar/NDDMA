/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_metadef/common/ge_common/util.h"
#include "common/preload/partition/model_kernel_args_partition.h"

namespace ge {
Status ModelKernelArgsPartition::Init(const GeModelPtr &ge_model, const uint8_t type) {
  (void)ge_model;
  (void)type;
  GELOGD("begin to generate kernel args.");
  const std::string str = "this is model kernel args partition";
  buff_size_ = static_cast<uint32_t>(str.length()) + 1U;
  buff_.reset(new (std::nothrow) uint8_t[buff_size_], std::default_delete<uint8_t[]>());
  GE_CHECK_NOTNULL(buff_);

  const auto ret =
      memcpy_s(buff_.get(), static_cast<uint64_t>(buff_size_), str.c_str(), static_cast<uint64_t>(buff_size_));
  GE_ASSERT_EOK(ret, "[Operate][Init]Failed at offset %u, error-code %d", buff_size_, ret);

  GELOGD("end generate kernel args info, buff_size_:%u", buff_size_);
  return SUCCESS;
}
}  // namespace ge
