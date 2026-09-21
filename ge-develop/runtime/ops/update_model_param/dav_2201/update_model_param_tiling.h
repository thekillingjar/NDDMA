/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file update_model_param_tiling.h
 * \brief
 */
#ifndef UPDATE_MODEL_PARAM_TILING_H
#define UPDATE_MODEL_PARAM_TILING_H
#include <cstdint>
 
struct UpdateModelParamTilingData {
    uint32_t totalActiveBaseTblCnt; // activeBase表包含的地址数量，以uint32为单位，必须32Byte对齐
    uint32_t blockCnt;              // 每个block处理的数据量，以uint32为单位，必须8Byte对齐
    uint32_t tileCnt;               // 每个tile处理的数据量，以uint32为单位，必须256Byte对齐
    uint32_t tailCnt;               // 最后一个tile处理的数据量，以uint32为单位，必须256Byte对齐
    uint32_t lastTailCnt;           // 最后一个block最后一个tile处理的数据量，以uint32为单位，必须256Byte对齐
    uint16_t tileNum;               // 每个block的循环次数
    uint16_t lastTileNum;           // 最后一个block的循环次数
    uint32_t lastTailCntOri;
    uint32_t reserve1;
};
 
__aicore__ inline void GetTilingData(UpdateModelParamTilingData &tiling_data, const __gm__ uint8_t *tiling_arg) {
    const __gm__ uint8_t *src = tiling_arg;
    uint8_t *dst = reinterpret_cast<uint8_t*>(&tiling_data);
    for (size_t i = 0; i < sizeof(UpdateModelParamTilingData); i++) {
        dst[i] = src[i];
    }
}
 
#endif