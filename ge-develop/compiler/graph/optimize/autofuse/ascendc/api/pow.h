/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_POW_H__
#define __ASCENDC_API_POW_H__

template <typename T>
inline __aicore__ void Pow(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src1,
                           const AscendC::LocalTensor<T> &src2, const uint32_t calCount,
                           AscendC::LocalTensor<uint8_t> &tmp_buf) {
  Power(dst, src1, src2, tmp_buf, calCount);
}

template <typename T>
inline __aicore__ void Pow(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src1, const T &src2,
                           const uint32_t calCount, AscendC::LocalTensor<uint8_t> &tmp_buf) {
  Power(dst, src1, src2, tmp_buf, calCount);
}

template <typename T>
inline __aicore__ void Pow(const AscendC::LocalTensor<T> &dst, const T &src1, const AscendC::LocalTensor<T> &src2,
                           const uint32_t calCount, AscendC::LocalTensor<uint8_t> &tmp_buf) {
  Power(dst, src1, src2, tmp_buf, calCount);
}

template <typename T>
inline __aicore__ void Pow(const AscendC::LocalTensor<T> &dst, const T &src1, const T &src2, const uint32_t calCount,
                           AscendC::LocalTensor<uint8_t> &tmp_buf) {
  auto block_cnt = KernelUtils::BlkSize<T>();
  // 将src1扩充为一个blockSize的tensor
  LocalTensor<T> src1_buf = tmp_buf.template ReinterpretCast<T>();
  src1_buf.SetSize(block_cnt);
  LocalTensor<T> dst_buf = tmp_buf[ONE_BLK_SIZE].template ReinterpretCast<T>();
  dst_buf.SetSize(block_cnt);

  LocalTensor<uint8_t> left_tmp_buf = tmp_buf[2*ONE_BLK_SIZE].template ReinterpretCast<uint8_t>();
  Duplicate(src1_buf, src1, block_cnt);
  // 调用Power基础API：tensor(block size) + scalar
  Power(dst_buf, src1_buf, src2, left_tmp_buf, block_cnt);
  // 取block tensor中的一个scalar元素，扩充为dst size长度的tensor
  Duplicate(dst, dst_buf.GetValue(0), dst.GetSize());
}
#endif  // __ASCENDC_API_POW_H__
