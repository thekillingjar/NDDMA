/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * 版权所有 (c) 华为技术有限公司 2024
 *
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/
#include "graph/utils/mem_utils.h"
namespace ge {
std::atomic<int64_t> MemUtils::gen_container_id_(0L);
std::atomic<int64_t> MemUtils::scope_id_(0L);
TQueConfig::TQueConfig(const int64_t id, const ge::Position pos, const int64_t depth, const int64_t buf_num)
    : queue_attr_({id, depth, buf_num, ""}), pos_(pos) {}

TBufConfig::TBufConfig(const int64_t id, const ge::Position pos) : buf_attr_({id, ""}), pos_(pos) {}

TQueConfig MemUtils::CreateTQueConfig(const ge::Position pos, const int64_t depth, const int64_t buf_num) {
  GE_ASSERT_TRUE(pos == Position::kPositionVecIn || (pos == Position::kPositionVecOut));
  return TQueConfig(gen_container_id_++, pos, depth, buf_num);
}

TBufConfig MemUtils::CreateTBufConfig(const ge::Position pos) {
  GE_ASSERT_TRUE(pos == Position::kPositionVecIn || (pos == Position::kPositionVecOut));
  return TBufConfig(gen_container_id_++, pos);
}
}