/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AUTOFUSE_REG_API_CALL_UTILS_H__
#define __AUTOFUSE_REG_API_CALL_UTILS_H__

#include "codegen_kernel.h"
#include "api_call/utils/api_call_utils.h"

namespace codegen {
/*
for循环组织形式如下:
for (i, 0, loop2Size) {
for (j, 0, loop1Size) {
  dst[i * loop2DstStride + j * loop1DrcStride] = src[i * loop2SrcStride + j * loop1SrcStride]
}
}
*/

struct LoopModeParams {
  // loop1Size, loop2Size
  std::vector<std::string> loop_size = {"1", "1"};
  // loop1SrcStride, loop2SrcStride 单位: 数字个数，在拼接参数时，会转换成字节数
  std::vector<std::string> loop_src_stride = {"0", "0"};
  // loop1DrcStride, loop2DstStride单位: 数字个数，在拼接参数时，会转换成字节数
  std::vector<std::string> loop_dst_stride = {"0", "0"};
};

struct NddmaParams {
  std::stringstream ss_output_dims;
  std::stringstream ss_output_stride;
  std::stringstream ss_input_stride;
};

void CreateEnhanceDmaCall(const TPipe &tpipe, const Tensor &input, const Tensor &output, const string &gm_offset,
                          const DataCopyParams &data_copy_param, const ascir::SizeExpr &offset, std::stringstream &ss,
                          bool copy_in);
void CreateNddmaCall(const TPipe &tpipe, const Tensor &input, const Tensor &output, const string &gm_offset,
                     const DataCopyParams &data_copy_param, const ascir::SizeExpr &offset, std::stringstream &ss);
void SetNddmaParams(const TPipe &tpipe, const DataCopyParams &data_copy_param, NddmaParams &nddma_param,
                    const int64_t &tensor_id, std::stringstream &ss);

}  // namespace codegen

#endif