/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/fe_running_env/include/mock_fftsplus_ops_kernel_builder.h"
#include <fstream>
#include "framework/fe_running_env/include/fe_env_utils.h"
#include "common/fe_log.h"

namespace ffts {
MockFFTSPlusOpsKernelBuilder::MockFFTSPlusOpsKernelBuilder() {
  fftsplus_ops_kernel_build_ptr_ = std::make_shared<FFTSPlusOpsKernelBuilder>();
}

MockFFTSPlusOpsKernelBuilder::~MockFFTSPlusOpsKernelBuilder() {

}

Status MockFFTSPlusOpsKernelBuilder::Initialize(const std::map<std::string, std::string> &options) {
  return fftsplus_ops_kernel_build_ptr_->Initialize(options);
}

Status MockFFTSPlusOpsKernelBuilder::Finalize() {
  return fftsplus_ops_kernel_build_ptr_->Finalize();
}

Status MockFFTSPlusOpsKernelBuilder::CalcOpRunningParam(ge::Node &node) {
  return fftsplus_ops_kernel_build_ptr_->CalcOpRunningParam(node);
}



Status MockFFTSPlusOpsKernelBuilder::GenerateTask(const ge::Node &node,
                                                  ge::RunContext &context, std::vector<domi::TaskDef> &task_defs) {
  FE_LOGD("MockFFTSPlusOpsKernelBuilder.GenerateTask begin");
  Status ret = fftsplus_ops_kernel_build_ptr_->GenerateTask(node, context, task_defs);
  if (task_defs.empty()) {
    FE_LOGD("FFTSPlusOpsKernelBuilder GenerateTask failed");
    return ret;
  }
  fe_env::FeEnvUtils::SaveTaskDefContextToFile(task_defs);

  domi::TaskDef &task_def = task_defs.at(task_defs.size() - 1);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  int32_t gen_ctx_size = ffts_plus_task_def->ffts_plus_ctx_size();
  int32_t invalid_count = 0;
  int32_t writeback_count = 0;
  int32_t prefetch_count = 0;
  for (int32_t i = 0; i < gen_ctx_size; ++i) {
    domi::FftsPlusCtxDef *ffts_plus_ctx = ffts_plus_task_def->mutable_ffts_plus_ctx(i);
    uint32_t type = ffts_plus_ctx->context_type();
    switch (type) {
    case RT_CTX_TYPE_FLUSH_DATA:
        ++prefetch_count;
        break;
    case RT_CTX_TYPE_INVALIDATE_DATA:
        ++invalid_count;
        break;
    case RT_CTX_TYPE_WRITEBACK_DATA:
        ++writeback_count;
        break;
    default:
        break;
    }
  }
  domi::FftsPlusSqeDef *ffts_plus_sqe = ffts_plus_task_def->mutable_ffts_plus_sqe();

  // write file, in case check the file, then remove the file
  std::ofstream outfile;
  std::string file_name = fe_env::FeEnvUtils::GetFFTSLogFile();
  outfile.open(file_name.c_str());
  if (outfile.is_open()) {
    outfile << "ready_context_num" << task_defs.size() - 1 << "=" << ffts_plus_sqe->ready_context_num() << "\n";
    outfile << "total_context_num" << task_defs.size() - 1 << "=" << ffts_plus_sqe->total_context_num() << "\n";
    outfile << "prefetch_count" << task_defs.size() - 1 << "=" << prefetch_count << "\n";
    outfile << "invalid_count" << task_defs.size() - 1 << "=" << invalid_count << "\n";
    outfile << "writeback_count" << task_defs.size() - 1 << "=" << writeback_count << "\n";
    outfile.close();
  }
  return ret;
}

} // namespace
