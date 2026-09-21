/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DATACOPY_NDDMA_H
#define DATACOPY_NDDMA_H

template <typename T, uint8_t dim>
inline __aicore__ void DataCopyNddma(const AscendC::LocalTensor<T> &dst, const AscendC::GlobalTensor<T> &src,
                                     const int64_t (&output_dims)[dim], const int64_t (&output_stride)[dim],
                                     const int64_t (&input_stride)[dim]) {
 AscendC::MultiCopyLoopInfo<dim> loop_info;
 for (uint32_t i = 0; i < dim; i++) {
  loop_info.loopSize[dim - 1 - i] = output_dims[i];
  loop_info.loopSrcStride[dim - 1 - i] = input_stride[i];
  loop_info.loopDstStride[dim - 1 - i] = output_stride[i];
 }
 AscendC::NdDmaDci();
 T const_value = 0;
 AscendC::MultiCopyParams<T, dim> copyParams = {loop_info, const_value};
 static constexpr AscendC::MultiCopyConfig config = {false, 0, 0, false};
 AscendC::DataCopy<T, dim, config>(dst, src, copyParams);
}

#endif //DATACOPY_NDDMA_H
