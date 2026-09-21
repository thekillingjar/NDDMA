/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma pack(1)
struct Packgraph_normalTilingData {
  uint32_t nbo_size = 0;
  uint32_t nio_size = 0;
  uint32_t sbo_size = 0;
  uint32_t sio_size = 0;
  uint32_t wbo_size = 0;
  uint32_t wio_size = 0;
  uint32_t block_dim = 0;
  uint32_t ub_size = 0;
  uint32_t workspaceSize = 0;
  uint32_t A = 0;
  uint32_t BL = 0;
  uint32_t R = 0;
  uint32_t additional_output = 0;
  uint32_t A_aligned_size = 0;
  uint32_t R_aligned_size = 0;
  uint32_t nio_tail_size = 0;
  uint32_t nio_loop_num = 0;
  uint32_t nbo_tail_tile_nio_tail_size = 0;
  uint32_t nbo_tail_tile_nio_loop_num = 0;
  uint32_t nbo_tail_size = 0;
  uint32_t nbo_loop_num = 0;
  uint32_t sio_tail_size = 0;
  uint32_t sio_loop_num = 0;
  uint32_t sbo_tail_size = 0;
  uint32_t sbo_loop_num = 0;
  uint32_t wio_tail_size = 0;
  uint32_t wio_loop_num = 0;
  uint32_t wbo_tail_size = 0;
  uint32_t wbo_loop_num = 0;
  uint32_t output3_total_size = 0;
  uint32_t output2_single_core_size = 0;
  uint32_t output2_total_size = 0;
  uint32_t output1_total_size = 0;
  uint32_t Q0 = 0;
  uint32_t output1_single_core_size = 0;
  uint32_t gm_size = 0;
  uint32_t Q1 = 0;
  uint32_t Q3 = 0;
  uint32_t output3_single_core_size = 0;
  uint32_t Q8 = 0;
  uint32_t Q2 = 0;
  uint32_t Q4 = 0;
  uint32_t Q5 = 0;
  uint32_t Q9 = 0;
  uint32_t Q6 = 0;
  uint32_t output0_total_size = 0;
  uint32_t Q7 = 0;
  uint32_t output0_single_core_size = 0;
  uint32_t tiling_key = 0;
};
#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  tilingStruct *tilingDataPointer = reinterpret_cast<tilingStruct *>((uint8_t *)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
  CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                    \
  Packgraph_normalTilingData tilingData;                                      \
  INIT_TILING_DATA(Packgraph_normalTilingData, tilingDataPointer, tilingPointer);    \
  (tilingData).nbo_size = tilingDataPointer->nbo_size;                   \
  (tilingData).nio_size = tilingDataPointer->nio_size;                   \
  (tilingData).sbo_size = tilingDataPointer->sbo_size;                   \
  (tilingData).sio_size = tilingDataPointer->sio_size;                   \
  (tilingData).wbo_size = tilingDataPointer->wbo_size;                   \
  (tilingData).wio_size = tilingDataPointer->wio_size;                   \
  (tilingData).block_dim = tilingDataPointer->block_dim;                   \
  (tilingData).ub_size = tilingDataPointer->ub_size;                   \
  (tilingData).workspaceSize = tilingDataPointer->workspaceSize;                   \
  (tilingData).A = tilingDataPointer->A;                   \
  (tilingData).BL = tilingDataPointer->BL;                   \
  (tilingData).R = tilingDataPointer->R;                   \
  (tilingData).additional_output = tilingDataPointer->additional_output;                   \
  (tilingData).A_aligned_size = tilingDataPointer->A_aligned_size;                   \
  (tilingData).R_aligned_size = tilingDataPointer->R_aligned_size;                   \
  (tilingData).nio_tail_size = tilingDataPointer->nio_tail_size;                   \
  (tilingData).nio_loop_num = tilingDataPointer->nio_loop_num;                   \
  (tilingData).nbo_tail_tile_nio_tail_size = tilingDataPointer->nbo_tail_tile_nio_tail_size;                   \
  (tilingData).nbo_tail_tile_nio_loop_num = tilingDataPointer->nbo_tail_tile_nio_loop_num;                   \
  (tilingData).nbo_tail_size = tilingDataPointer->nbo_tail_size;                   \
  (tilingData).nbo_loop_num = tilingDataPointer->nbo_loop_num;                   \
  (tilingData).sio_tail_size = tilingDataPointer->sio_tail_size;                   \
  (tilingData).sio_loop_num = tilingDataPointer->sio_loop_num;                   \
  (tilingData).sbo_tail_size = tilingDataPointer->sbo_tail_size;                   \
  (tilingData).sbo_loop_num = tilingDataPointer->sbo_loop_num;                   \
  (tilingData).wio_tail_size = tilingDataPointer->wio_tail_size;                   \
  (tilingData).wio_loop_num = tilingDataPointer->wio_loop_num;                   \
  (tilingData).wbo_tail_size = tilingDataPointer->wbo_tail_size;                   \
  (tilingData).wbo_loop_num = tilingDataPointer->wbo_loop_num;                   \
  (tilingData).output3_total_size = tilingDataPointer->output3_total_size;                   \
  (tilingData).output2_single_core_size = tilingDataPointer->output2_single_core_size;                   \
  (tilingData).output2_total_size = tilingDataPointer->output2_total_size;                   \
  (tilingData).output1_total_size = tilingDataPointer->output1_total_size;                   \
  (tilingData).Q0 = tilingDataPointer->Q0;                   \
  (tilingData).output1_single_core_size = tilingDataPointer->output1_single_core_size;                   \
  (tilingData).gm_size = tilingDataPointer->gm_size;                   \
  (tilingData).Q1 = tilingDataPointer->Q1;                   \
  (tilingData).Q3 = tilingDataPointer->Q3;                   \
  (tilingData).output3_single_core_size = tilingDataPointer->output3_single_core_size;                   \
  (tilingData).Q8 = tilingDataPointer->Q8;                   \
  (tilingData).Q2 = tilingDataPointer->Q2;                   \
  (tilingData).Q4 = tilingDataPointer->Q4;                   \
  (tilingData).Q5 = tilingDataPointer->Q5;                   \
  (tilingData).Q9 = tilingDataPointer->Q9;                   \
  (tilingData).Q6 = tilingDataPointer->Q6;                   \
  (tilingData).output0_total_size = tilingDataPointer->output0_total_size;                   \
  (tilingData).Q7 = tilingDataPointer->Q7;                   \
  (tilingData).output0_single_core_size = tilingDataPointer->output0_single_core_size;                   \
  (tilingData).tiling_key = tilingDataPointer->tiling_key;
void PrintTilingData(Packgraph_normalTilingData& tilingData) {
  std::cout << "=======================================" << std::endl;
  std::cout << " nbo_size: " << tilingData.nbo_size << std::endl;
  std::cout << " nio_size: " << tilingData.nio_size << std::endl;
  std::cout << " sbo_size: " << tilingData.sbo_size << std::endl;
  std::cout << " sio_size: " << tilingData.sio_size << std::endl;
  std::cout << " wbo_size: " << tilingData.wbo_size << std::endl;
  std::cout << " wio_size: " << tilingData.wio_size << std::endl;
  std::cout << " block_dim: " << tilingData.block_dim << std::endl;
  std::cout << " ub_size: " << tilingData.ub_size << std::endl;
  std::cout << " workspaceSize: " << tilingData.workspaceSize << std::endl;
  std::cout << " A: " << tilingData.A << std::endl;
  std::cout << " BL: " << tilingData.BL << std::endl;
  std::cout << " R: " << tilingData.R << std::endl;
  std::cout << " additional_output: " << tilingData.additional_output << std::endl;
  std::cout << " A_aligned_size: " << tilingData.A_aligned_size << std::endl;
  std::cout << " R_aligned_size: " << tilingData.R_aligned_size << std::endl;
  std::cout << " nio_tail_size: " << tilingData.nio_tail_size << std::endl;
  std::cout << " nio_loop_num: " << tilingData.nio_loop_num << std::endl;
  std::cout << " nbo_tail_tile_nio_tail_size: " << tilingData.nbo_tail_tile_nio_tail_size << std::endl;
  std::cout << " nbo_tail_tile_nio_loop_num: " << tilingData.nbo_tail_tile_nio_loop_num << std::endl;
  std::cout << " nbo_tail_size: " << tilingData.nbo_tail_size << std::endl;
  std::cout << " nbo_loop_num: " << tilingData.nbo_loop_num << std::endl;
  std::cout << " sio_tail_size: " << tilingData.sio_tail_size << std::endl;
  std::cout << " sio_loop_num: " << tilingData.sio_loop_num << std::endl;
  std::cout << " sbo_tail_size: " << tilingData.sbo_tail_size << std::endl;
  std::cout << " sbo_loop_num: " << tilingData.sbo_loop_num << std::endl;
  std::cout << " wio_tail_size: " << tilingData.wio_tail_size << std::endl;
  std::cout << " wio_loop_num: " << tilingData.wio_loop_num << std::endl;
  std::cout << " wbo_tail_size: " << tilingData.wbo_tail_size << std::endl;
  std::cout << " wbo_loop_num: " << tilingData.wbo_loop_num << std::endl;
  std::cout << " output3_total_size: " << tilingData.output3_total_size << std::endl;
  std::cout << " output2_single_core_size: " << tilingData.output2_single_core_size << std::endl;
  std::cout << " output2_total_size: " << tilingData.output2_total_size << std::endl;
  std::cout << " output1_total_size: " << tilingData.output1_total_size << std::endl;
  std::cout << " Q0: " << tilingData.Q0 << std::endl;
  std::cout << " output1_single_core_size: " << tilingData.output1_single_core_size << std::endl;
  std::cout << " gm_size: " << tilingData.gm_size << std::endl;
  std::cout << " Q1: " << tilingData.Q1 << std::endl;
  std::cout << " Q3: " << tilingData.Q3 << std::endl;
  std::cout << " output3_single_core_size: " << tilingData.output3_single_core_size << std::endl;
  std::cout << " Q8: " << tilingData.Q8 << std::endl;
  std::cout << " Q2: " << tilingData.Q2 << std::endl;
  std::cout << " Q4: " << tilingData.Q4 << std::endl;
  std::cout << " Q5: " << tilingData.Q5 << std::endl;
  std::cout << " Q9: " << tilingData.Q9 << std::endl;
  std::cout << " Q6: " << tilingData.Q6 << std::endl;
  std::cout << " output0_total_size: " << tilingData.output0_total_size << std::endl;
  std::cout << " Q7: " << tilingData.Q7 << std::endl;
  std::cout << " output0_single_core_size: " << tilingData.output0_single_core_size << std::endl;
  std::cout << " tiling_key: " << tilingData.tiling_key << std::endl;
  std::cout << "=======================================" << std::endl;
}

