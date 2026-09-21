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
 * \file update_model_param.h
 * \brief
 */
#ifndef __UPDATE_MODEL_PARAM_H__
#define __UPDATE_MODEL_PARAM_H__
 
#include "kernel_operator.h"
#include "update_model_param_tiling.h"
 
__gm__ struct OpSystemRunCfg g_opSystemRunCfg = {0};
 
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t adds_scalar = 0x80000000;
constexpr uint16_t shift_scalar = 1;
const half one_scalar = half(1.0);
 
class KernelUpdateModelParam {
public:
    __aicore__ inline KernelUpdateModelParam() {}
    __aicore__ inline void Init(GM_ADDR active_base, GM_ADDR index, GM_ADDR offset, GM_ADDR model_args,
                                UpdateModelParamTilingData &tiling, uint32_t blockId)
    {
        uint64_t blockAddr = tiling.blockCnt * blockId;
        activeBaseGm.SetGlobalBuffer((__gm__ int32_t *)active_base);
        indexGm.SetGlobalBuffer((__gm__ uint32_t *)index + blockAddr);
        offsetGm.SetGlobalBuffer((__gm__ int32_t *)offset + blockAddr);
        modelArgsGm.SetGlobalBuffer((__gm__ int32_t *)model_args + blockAddr);
 
        pipe.InitBuffer(inQueueActiveBase, BUFFER_NUM, tiling.totalActiveBaseTblCnt * sizeof(int32_t));
        pipe.InitBuffer(inQueueIndex, BUFFER_NUM, tiling.tileCnt * sizeof(uint32_t));
        pipe.InitBuffer(inQueueOffset, BUFFER_NUM, tiling.tileCnt * sizeof(int32_t));
        pipe.InitBuffer(outQueueModelArgs, BUFFER_NUM, tiling.tileCnt * sizeof(int32_t));
 
        pipe.InitBuffer(tempCompQueue, tiling.tileCnt * sizeof(uint8_t));
        pipe.InitBuffer(tempSelectQueue, tiling.tileCnt * sizeof(half));
        pipe.InitBuffer(tempOrQueue, tiling.tileCnt * sizeof(uint16_t));
        pipe.InitBuffer(tempZeroQueue, tiling.tileCnt * sizeof(half));
 
        tempOrLocal = tempOrQueue.Get<uint16_t>();
        tempZeroLocal = tempZeroQueue.Get<half>();
        AscendC::Duplicate<uint16_t>(tempOrLocal, 0x5555, tiling.tileCnt);
        AscendC::Duplicate<half>(tempZeroLocal, 0x0, tiling.tileCnt);
    }
    __aicore__ inline void Process(UpdateModelParamTilingData &tiling, uint32_t blockId, uint32_t blockNum)
    {
        AscendC::LocalTensor<int32_t> activeBaseLocal_ = inQueueActiveBase.AllocTensor<int32_t>();
        AscendC::DataCopy(activeBaseLocal_, activeBaseGm[0], tiling.totalActiveBaseTblCnt);
        uint16_t tileNum = (blockId == (blockNum - 1)) ? tiling.lastTileNum : tiling.tileNum;
        for (uint16_t i = 0; i < tileNum; i++) {
            uint32_t calCnt = tiling.tileCnt;
            uint32_t gatherCnt = tiling.tileCnt;
            if (i == (tileNum - 1)) {
                calCnt = (blockId == (blockNum - 1)) ? tiling.lastTailCnt : tiling.tailCnt;
                gatherCnt = (blockId == (blockNum - 1)) ? tiling.lastTailCntOri : tiling.tailCnt;
            }
            uint32_t halfCnt = gatherCnt / 0x2;
            // CopyIn
            AscendC::LocalTensor<uint32_t> indexLocal_ = inQueueIndex.AllocTensor<uint32_t>();
            AscendC::LocalTensor<int32_t> offsetLocal_ = inQueueOffset.AllocTensor<int32_t>();
            AscendC::DataCopy(indexLocal_, indexGm[i * tiling.tileCnt], calCnt);
            AscendC::DataCopy(offsetLocal_, offsetGm[i * tiling.tileCnt], calCnt);
            inQueueActiveBase.EnQue(activeBaseLocal_);
            inQueueIndex.EnQue(indexLocal_);
            inQueueOffset.EnQue(offsetLocal_);
            // Compute
            AscendC::LocalTensor<int32_t> activeBaseLocal = inQueueActiveBase.DeQue<int32_t>();
            AscendC::LocalTensor<uint32_t> indexLocal = inQueueIndex.DeQue<uint32_t>();
            AscendC::LocalTensor<int32_t> offsetLocal = inQueueOffset.DeQue<int32_t>();
            AscendC::LocalTensor<int32_t> modelArgsLocal = outQueueModelArgs.AllocTensor<int32_t>();
 
            AscendC::Gather(modelArgsLocal, activeBaseLocal, indexLocal, 0, gatherCnt);
            inQueueIndex.EnQue(indexLocal);
            AscendC::LocalTensor<int32_t> indexLocal_tmp = inQueueIndex.DeQue<int32_t>();
            AscendC::Add(modelArgsLocal, modelArgsLocal, offsetLocal, gatherCnt);
 
            AscendC::Adds(indexLocal_tmp, modelArgsLocal, adds_scalar, gatherCnt);
            AscendC::Adds(offsetLocal, offsetLocal, adds_scalar, gatherCnt);
            AscendC::Max(offsetLocal, indexLocal_tmp, offsetLocal, gatherCnt);
 
            AscendC::LocalTensor<uint8_t> tempCompLocal = tempCompQueue.Get<uint8_t>();
            AscendC::Compare(tempCompLocal, indexLocal_tmp, offsetLocal, AscendC::CMPMODE::EQ, calCnt);
 
            AscendC::LocalTensor<uint16_t> tempShiftLocal = tempCompQueue.Get<uint16_t>();
            AscendC::ShiftLeft(tempShiftLocal, tempShiftLocal, shift_scalar, halfCnt);
            AscendC::Or(tempShiftLocal, tempShiftLocal, tempOrLocal, halfCnt);
 
            AscendC::LocalTensor<half> tempSelectLocal = tempSelectQueue.Get<half>();
            AscendC::Select(tempSelectLocal, tempShiftLocal, tempZeroLocal, one_scalar,
                AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, gatherCnt);
 
            AscendC::Cast(indexLocal_tmp, tempSelectLocal, AscendC::RoundMode::CAST_RINT, gatherCnt);
            AscendC::Add(modelArgsLocal, modelArgsLocal, indexLocal_tmp, gatherCnt);
            inQueueIndex.FreeTensor(indexLocal_);
            inQueueOffset.FreeTensor(offsetLocal_);
            outQueueModelArgs.EnQue<int32_t>(modelArgsLocal);
            // CopyOut
            AscendC::LocalTensor<int32_t> modelArgsLocal_ = outQueueModelArgs.DeQue<int32_t>();
            AscendC::DataCopy(modelArgsGm[i * tiling.tileCnt], modelArgsLocal_, gatherCnt);
            outQueueModelArgs.FreeTensor(modelArgsLocal_);
        }
        inQueueActiveBase.FreeTensor(activeBaseLocal_);
    }
 
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueActiveBase, inQueueIndex, inQueueOffset;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueModelArgs;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempCompQueue, tempOrQueue, tempZeroQueue, tempSelectQueue;
    AscendC::LocalTensor<uint16_t> tempOrLocal;
    AscendC::LocalTensor<half> tempZeroLocal;
    AscendC::GlobalTensor<int32_t> activeBaseGm;
    AscendC::GlobalTensor<uint32_t> indexGm;
    AscendC::GlobalTensor<int32_t> offsetGm;
    AscendC::GlobalTensor<int32_t> modelArgsGm;
};
 
extern "C" __global__ __aicore__ void update_model_param(GM_ADDR offset, GM_ADDR index, GM_ADDR active_base,
                                                         GM_ADDR model_args, GM_ADDR workspace, GM_ADDR tiling)
{
    UpdateModelParamTilingData tiling_data;
    GetTilingData(tiling_data, tiling);
    KernelUpdateModelParam op;
    op.Init(active_base, index, offset, model_args, tiling_data, AscendC::GetBlockIdx());
    op.Process(tiling_data, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
}
 
struct AscendCInfoMetaDFX {
    BaseTlv head;
    uint8_t value[112];
};
 
static const struct FunLevelKType UpdateModelParam_kernel_type_section __attribute__
    ((used, section (".ascend.meta.UpdateModelParam"))) = { {{F_TYPE_KTYPE, sizeof(unsigned int)}, K_TYPE_AIV} };
static const struct AscendCInfoMetaDFX UpdateModelParam_dfx_section __attribute__
    ((used, section (".ascend.meta.UpdateModelParam"))) = {{4, 112}, {0, 5, 0, 2, 0, 0, 0, 0, 0, 1, 0, 2, 255, 255, 255,
    255, 255, 255, 255, 255, 0, 5, 0, 2, 0, 0, 0, 0, 0, 1, 0, 2, 255, 255, 255, 255, 255, 255, 255, 255, 0, 5, 0, 2, 0,
    0, 0, 0, 0, 1, 0, 2, 255, 255, 255, 255, 255, 255, 255, 255, 0, 5, 0, 2, 0, 0, 0, 0, 0, 1, 0, 3, 255, 255, 255, 255,
    255, 255, 255, 255, 0, 5, 0, 1, 0, 0, 0, 0, 0, 1, 0, 4, 0, 5, 0, 2, 0, 0, 0, 0, 0, 1, 0, 7, 0, 0, 0, 0, 0, 0, 0,
    40}};
#endif