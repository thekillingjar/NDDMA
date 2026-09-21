/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_DUMP_DUMP_UTILS_H_
#define GE_COMMON_DUMP_DUMP_UTILS_H_
#include "graph/op_desc.h"
#include "ge/ge_api_types.h"

namespace ge {
constexpr uint64_t kAssertWorkFlag = 4U;
constexpr uint16_t kDumpTypeBitNum = 56U;

// 值为uint64_max表示占位，8bit为0x1或者0x2表示二级指针，否则表示args中的相对位置
bool OpNeedAssert(const OpDescPtr &op_desc);
Status ReportL0ExceptionDumpInfo(const OpDescPtr &op_desc, const std::vector<uint64_t> &l0_size_list);
Status SetL0ExceptionSizeInfo(const OpDescPtr &op_desc, const std::vector<uint64_t> &l0_size_list);
Status UpdateL0ExceptionDumpInfoSize(const OpDescPtr &op_desc, std::vector<uint64_t> &l0_size_list,
                                     const size_t &index);
}  // namespace ge

#endif  // GE_COMMON_DUMP_DUMP_UTILS_H_
