/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string.h>
#include "macro_utils/dt_public_scope.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "macro_utils/dt_public_unscope.h"


using namespace std;
using namespace testing;

namespace gert {
class UtestRt2ScatteredCollection : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};
TEST_F(UtestRt2ScatteredCollection, InitAutoMixAicAivCtx_Test) {
  std::vector<uintptr_t> io_addrs;
  std::vector<void *> ext_args;
  std::vector<size_t> mode_addr_idx;
  const ge::RuntimeParam runtime_param;
  FftsPlusProtoTransfer ffpt(0U, io_addrs, runtime_param, ext_args, mode_addr_idx);

  domi::TaskDef task_def;
  task_def.set_stream_id(0);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  mixaicaivctx->set_op_index(0);
  mixaicaivctx->set_context_id(0);
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));
  domi::FftsPlusMixAicAivCtxDef &mixctxdef = *mixaicaivctx->mutable_mix_aic_aiv_ctx();

  rtFftsPlusMixAicAivCtx_t ctx;
  uint32_t start_idx = 0;

  // Test: ctx_def.kernel_name_size() == 0
  mixctxdef.set_save_task_addr(1);
  mixctxdef.set_thread_dim(2);
  ctx.threadDim = mixctxdef.thread_dim();
  EXPECT_EQ(ffpt.InitAutoMixAicAivCtx(mixctxdef, ctx, start_idx), ge::FAILED);

  start_idx = 7;
  EXPECT_EQ(ffpt.InitAutoMixAicAivCtx(mixctxdef, ctx, start_idx), ge::FAILED);
}
}
