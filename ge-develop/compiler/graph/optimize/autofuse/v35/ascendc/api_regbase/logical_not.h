/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_LOGICAL_NOT_H
#define __ASCENDC_API_REGBASE_LOGICAL_NOT_H

template <typename T>
inline __aicore__ void LogicalNotExtend(const AscendC::LocalTensor<uint8_t> &dst, const AscendC::LocalTensor<T> &src,
                                        const uint32_t count) {
  auto dst_bool = dst.template ReinterpretCast<bool>();
  AscendC::LogicalNot(dst_bool, src, count);
}
#endif  // __ASCENDC_API_REGBASE_LOGICAL_NOT_H
