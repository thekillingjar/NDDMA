/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_FFTS_PLUS_AICORE_UPDATE_KERNEL_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_FFTS_PLUS_AICORE_UPDATE_KERNEL_H_
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "runtime/rt_ffts_plus_define.h"
#include "runtime/rt_ffts_plus.h"
#include "runtime/mem.h"
#include "kernel/kernel_log.h"
#include "kernel/memory/mem_block.h"
#include "kernel/memory/ffts_mem_allocator.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "engine/aicore/fe_rt2_common.h"
namespace gert {
namespace kernel {
constexpr int kMaxCmoType = 3;
constexpr size_t kMaxIndexNum = 64;
constexpr size_t kFFTSMaxThreadNum = 32;
constexpr size_t kMaxOffsetNum = 128;
constexpr size_t kSgtSlicePartNum = 2;

struct AICoreSubTaskFlush {
  int32_t device_id{0};
  bool need_mode_addr{false};
  void *args_base{nullptr};
  uintptr_t aic_non_tail_task_start_pc{0U};
  uintptr_t aic_tail_task_start_pc{0U};
  uint32_t aic_icache_prefetch_cnt{0U};
  uintptr_t aiv_non_tail_task_start_pc{0U};
  uintptr_t aiv_tail_task_start_pc{0U};
  uint32_t aiv_icache_prefetch_cnt{0U};
  // Task I/O Addrs.
  uintptr_t input_addr_vv[kMaxIndexNum][kFFTSMaxThreadNum];
  uint64_t input_offset[kMaxIndexNum];
  uintptr_t output_addr_vv[kMaxIndexNum][kFFTSMaxThreadNum];
  uint64_t output_offset[kMaxIndexNum];
  uint32_t thread_dim{0U};
  uint32_t window_size{0U};
  uint32_t atom_proc_type{0U};
  uint16_t blk_dim{0U};
  uint16_t tail_blk_dim{0U};
  uint16_t param_ptr_offset{0U};
  uint16_t schedule_mode{0U};
};

struct AICoreSinkRet {
  uintptr_t aic_non_tail_task_start_pc{0U};
  uintptr_t aic_tail_task_start_pc{0U};
  uint32_t aic_icache_prefetch_cnt{0U};
  uintptr_t aiv_non_tail_task_start_pc{0U};
  uintptr_t aiv_tail_task_start_pc{0U};
  uint32_t aiv_icache_prefetch_cnt{0U};
};

struct DataContextParam {
  uint32_t addressOffset{0U};
  uint16_t nonTailNumOutter{0U};
  uint16_t nonTailNumInner{0U};
  uint32_t nonTailLengthInner{0U};
  uint32_t nonTailStrideOutter{0U};
  uint32_t nonTailStrideInner{0U};
  uint16_t tailNumOutter{0U};
  uint16_t tailNumInner{0U};
  uint32_t tailLengthInner{0U};
  uint32_t tailStrideOutter{0U};
  uint32_t tailStrideInner{0U};
  uint32_t index{0U};
};

struct CalcFftsMemoryReuseInfo {
  size_t workspace_addrs_size{0U};
  size_t output_addrs_index{0U};
  size_t output_addrs_size{0U};
  size_t old_output_index{0U};
  size_t old_output_size{0U};
};

struct CmoFlushData {
  DataContextParam *data_params{nullptr};
  uint32_t slice_num{0U};
  uint32_t ori_window_size{0U};
  uint32_t window_size{0U};
  uint32_t real_size{0U};
  const Shape* no_tail_slice_in{nullptr};
  size_t no_tail_size_in{0U};
  const Shape* tail_slice_in{nullptr};
  size_t tail_size_in{0U};
  const Shape* no_tail_slice_out{nullptr};
  size_t no_tail_size_out{0U};
  const Shape* tail_slice_out{nullptr};
  size_t tail_size_out{0U};
  const AICoreSubTaskFlush *flush_data{nullptr};
  const ComputeNodeInfo *compute_node_info{nullptr};
  uint32_t out_start_pos{0U};
};

/**************** FOR DYNAMIC GRAPH BEGIN ***************/
enum class CalOffsetKey {
  THREAD_DIM = 0,
  WORKSPACE = 1,
  INPUT_INDEXES = 2,
  INPUT_SLICE = 3,
  OUTPUT_INDEXES = 4,
  OUTPUT_SLICE = 5,
  BLOCK_DIM = 6,
  TAIL_BLOCK_DIM = 7,
  SCHEDULE_MODE = 8,
  RESERVED
};

enum class UpdateKey {
  FLUSH_DATA = 0,
  AICORE_CTX,
  FIRST_IN_SLICE,
  LAST_IN_SLICE,
  FIRST_OUT_SLICE,
  LAST_OUT_SLICE = 5,
  PREFETCH_IDX,
  WRITEBACK_IDX,
  INVALID_IDX,
  PREFETCH_CTX,
  WRITEBACK_CTX = 10,
  INVALID_CTX,
  ORI_WINDOW_SIZE,
  DATA_PARAMS,
  TASK_INFO,
  REUSE_CTX = 15,
  RESERVED
};

// data dump
enum class DataDumpKey {
  THREAD_DIM = 0,
  AICORE_CTX,
  INPUT_SLICE_INDEXES,
  OUTPUT_SLICE_INDEXES,
  IN_SHAPES,
  LAST_IN_SHAPES = 5,
  OUT_SHAPES,
  LAST_OUT_SHAPES,
  THREAD_PARAM,
  IN_MEM_TYPE,
  OUT_MEM_TYPE = 10,
  IO_START,
  RESERVED
};

// Exception dump
enum class ExceptionDumpKey {
  THREAD_DIM = 0,
  WORKSPACE,
  ARGS_PARA,
  RESERVED
};

struct AICoreThreadParam {
  uint64_t offset_vec[kMaxOffsetNum]; // input + output + workspace
  uint32_t input_num{0U};  // input
  uint32_t input_output_num{0U};  // input + output
  uint16_t blk_dim;
  uint16_t tail_blk_dim;
  uint16_t schedule_mode;
};

enum class ArgsInKey {
  CHECK_RES = 0,
  WORKSPACE,
  NEED_MODE_ADDR,
  THREAD_PARAM,
  IN_MEM_TYPE,
  OUT_MEM_TYPE = 5,
  THREAD_DIM,
  WINDOW_SIZE,
  SINK_RET,
  ARGS_PARA,
  IO_START,
  kNUM
};
enum class ArgsOutKey {
  FLUSH_DATA = 0,
  kNUM
};
enum class ThreadOutKey {
  IN_SHAPES = 0,
  LAST_IN_SHAPES = 1,
  ORI_IN_SHAPES = 2,
  LAST_ORI_IN_SHAPES = 3,
  OUT_SHAPES = 4,
  LAST_OUT_SHAPES = 5,
  ORI_OUT_SHAPES = 6,
  LAST_ORI_OUT_SHAPES = 7,
  kNUM
};

enum class UpTilingKey {
  DEPEND_VEC = 0,
  SLICE_SHAPE = 1,
  SLICE_ORI_SHAPE = 2,
  IN_SHAPE_START = 3,
  kNUM
};

enum class SkipInKey : uint32_t {
  DISCARD_HOLDER = 0,
  CTX_TYPE,
  CTX_ID,
  ATOMIC_CTX_ID,
  PREFETCH_CTX_ID,
  INVALID_CTX_ID,
  WRITEBACK_CTX_ID,
  kNUM
};
/**************** FOR DYNAMIC GRAPH END ***************/

/**************** FOR STATIC GRAPH BEGIN ***************/
enum class StaArgsInKey {
  WORKSPACE = 0,
  SINK_RET,
  ARGS_PARA,
  IN_NUM,
  OUT_NUM,
  IO_START = 5,
  kNUM
};
enum class StaArgsOutKey {
  FLUSH_DATA = 0,
  kNUM
};

enum class StaUpdateKey {
  FLUSH_DATA = 0,
  TASK_INFO,
  AICORE_CTX,
  PREFETCH_IDX,
  PREFETCH_CTX,
  WRITEBACK_IDX = 5,
  WRITEBACK_CTX,
  INVALID_IDX,
  INVALID_CTX,
  RESERVED
};

// Static Manual data dump
enum class ManualDataDumpKey {
  THREAD_ID = 0,
  CONTEXT_ID,
  IN_NUM,
  OUT_NUM,
  /**
  * IO_START memory layer:
  *     1. INPUT_SHAPES_SIZE: each input shape size, continous vector, size IN_NUM
  *     2. OUTPUT_SHAPES_SIZE: each output shape size, continous vector, size OUT_NUM
  *     3. IO_ADDRS: input_addrs and output_addrs tensors
  */
  IO_START,
  RESERVED
};

// Static Manual exception dump
enum class ManualExceptionDumpKey {
  WORKSPACE = 0,
  ARGS_PARA,
  RESERVED
};

enum class AutoArgsInKey {
  WORKSPACE = 0,
  SINK_RET,
  ARGS_PARA,
  THREAD_DIM,
  WINDOW_SIZE,
  THREAD_OFFSET = 5,
  IN_NUM,
  OUT_NUM,
  IO_START,
  kNUM
};

enum class AutoUpdateKey {
  FLUSH_DATA = 0,
  TASK_INFO,
  BLOCK_DIM,
  AICORE_CTX,
  PREFETCH_IDX,
  PREFETCH_CTX = 5,
  WRITEBACK_IDX,
  WRITEBACK_CTX,
  INVALID_IDX,
  INVALID_CTX,
  RESERVED
};

// Static Auto data dump
enum class AutoDataDumpKey {
  THREAD_DIM = 0,
  WINDOW_SIZE,
  AICORE_CTX,
  THREAD_ADDR_OFFSET,
  IN_NUM,
  OUT_NUM = 5,
  /**
  * IO_START memory layer:
  *     1. INPUT_ORI_SHAPES_SIZE: each input ori shape size, continous vector, size IN_NUM
  *     2. OUTPUT_ORI_SHAPES_SIZE: each output ori shape size, continous vector, size OUT_NUM
  *     3. IO_ADDRS: input_addrs and output_addrs tensors
  */
  IO_START,
  RESERVED
};

// Static Auto exception dump
enum class AutoExceptionDumpKey {
  THREAD_DIM = 0,
  WORKSPACE,
  ARGS_PARA,
  RESERVED
};

/**************** FOR STATIC GRAPH END ***************/

/**************** FOR ATOMIC NODE BEGIN ***************/
enum class AtomArgsInKey {
  THREAD_DIM,
  WINDOW_SIZE,
  PROC_TYPE,
  SINK_RET,
  ARGS_PARA,
  WORK_CLEAR_IDX,
  OUT_CLEAR_IDX,
  WORK_ADDR,
  OUT_MEM_TYPE,
  THREAD_PARAM,
  IO_START,
  kNUM
};

enum class AtomUpdateKey {
  FLUSH_DATA = 0,
  AICORE_CTX,
  BLOCK_DIM,
  TAIL_BLOCK_DIM,
  TASK_INFO,
  RESERVED
};

enum class AtomProcType {
  DY_SLICE_OP = 0,
  DY_OP,
  STATIC_OP,
  RESERVED
};
/**************** FOR ATOMIC NODE END ***************/

#define FFTS_CTX_BIT_NUM_32 32
inline void SetLow32FromSrc(uint32_t &dst, const uint64_t &src) {
  dst = static_cast<uint32_t>(src);
  return;
}
inline void SetHigh32FromSrc(uint32_t &dst, const uint64_t &src) {
  dst = static_cast<uint32_t>(src >> FFTS_CTX_BIT_NUM_32);
  return;
}
inline void SetHigh16FromSrc(uint16_t &dst, const uint64_t &src) {
  dst = static_cast<uint16_t>(src >> FFTS_CTX_BIT_NUM_32);
  return;
}
bool InitCtxIoAddrs(const size_t index, const memory::FftsMemBlock *ffts_mem, const AICoreThreadParam *task_param,
                    AICoreSubTaskFlush *flush_data, uintptr_t *args_host_data);
bool InitCtxIoAddrs(size_t index, const gert::GertTensorData *tensor_data, const AICoreThreadParam *task_param,
                    AICoreSubTaskFlush *flush_data, uintptr_t *args_host_data);
void InitL1WorkAddrs(const size_t index, const void* addr, AICoreSubTaskFlush *flush_data, uintptr_t *args_host_data);
void InitOpTiling(const size_t index, const AICoreSubTaskFlush *flush_data,
                  uintptr_t *args_host_data, size_t tiling_offset, const void *args_addr);

inline void UpdateDyAicAivCtx(rtFftsPlusAicAivCtx_t *ctx, const AICoreSubTaskFlush *flush_data, uint64_t paraBase) {
  ctx->nonTailBlockdim = flush_data->blk_dim;
  ctx->tailBlockdim = flush_data->tail_blk_dim;
  ctx->threadDim = flush_data->thread_dim;
  ctx->schem = flush_data->schedule_mode;
  SetLow32FromSrc(ctx->taskParamPtrBaseL, paraBase);
  SetHigh16FromSrc(ctx->taskParamPtrBaseH, paraBase);
  SetLow32FromSrc(ctx->nonTailTaskStartPcL, flush_data->aic_non_tail_task_start_pc);
  SetHigh16FromSrc(ctx->nonTailTaskStartPcH, flush_data->aic_non_tail_task_start_pc);
  SetLow32FromSrc(ctx->tailTaskStartPcL, flush_data->aic_tail_task_start_pc);
  SetHigh16FromSrc(ctx->tailTaskStartPcH, flush_data->aic_tail_task_start_pc);
  ctx->icachePrefetchCnt = flush_data->aic_icache_prefetch_cnt;
  ctx->taskParamPtrOffset = flush_data->param_ptr_offset * sizeof(uintptr_t);
  return;
}

inline void UpdateStaAicAivCtx(rtFftsPlusAicAivCtx_t *ctx, const AICoreSubTaskFlush *flush_data, uint64_t paraBase) {
  SetLow32FromSrc(ctx->taskParamPtrBaseL, paraBase);
  SetHigh16FromSrc(ctx->taskParamPtrBaseH, paraBase);
  SetLow32FromSrc(ctx->tailTaskStartPcL, flush_data->aic_tail_task_start_pc);
  SetHigh16FromSrc(ctx->tailTaskStartPcH, flush_data->aic_tail_task_start_pc);
  SetLow32FromSrc(ctx->nonTailTaskStartPcL, flush_data->aic_non_tail_task_start_pc);
  SetHigh16FromSrc(ctx->nonTailTaskStartPcH, flush_data->aic_non_tail_task_start_pc);
  ctx->icachePrefetchCnt = flush_data->aic_icache_prefetch_cnt;
  ctx->taskParamPtrOffset = flush_data->param_ptr_offset * sizeof(uintptr_t);
  return;
}
}
}
#endif  // AIR_CXX_RUNTIME_V2_KERNEL_FFTS_PLUS_AICORE_UPDATE_KERNEL_H_
