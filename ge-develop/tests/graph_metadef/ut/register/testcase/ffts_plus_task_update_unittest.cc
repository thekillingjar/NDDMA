/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "register/ffts_plus_update_manager.h"
#include "graph_metadef/common/plugin/plugin_manager.h"

namespace ge {
class FFTSPlusTaskUpdateStub : public FFTSPlusTaskUpdate {
 public:
  Status GetAutoThreadParam(const NodePtr &node, const std::vector<optiling::utils::OpRunInfo> &op_run_info,
                            AutoThreadParam &auto_thread_param) override {
    return SUCCESS;
  }

  Status UpdateSubTaskAndCache(const NodePtr &node, const AutoThreadSubTaskFlush &sub_task_flush,
                               rtFftsPlusTaskInfo_t &ffts_plus_task_info) override {
    return SUCCESS;
  }

  Status UpdateCommonCtx(const ComputeGraphPtr &sgt_graph, rtFftsPlusTaskInfo_t &task_info) override {
    return SUCCESS;
  }
};

class UtestFftsPlusUpdate : public testing::Test {
 protected:
  void SetUp() {
    const std::string kCoreTypeTest = "FFTS_TEST";     // FftsPlusUpdateManager::FftsPlusUpdateRegistrar
    REGISTER_FFTS_PLUS_CTX_UPDATER(kCoreTypeTest, FFTSPlusTaskUpdateStub);
  }

  void TearDown() {
    FftsPlusUpdateManager::Instance().creators_.clear();
    FftsPlusUpdateManager::Instance().plugin_manager_.reset();
  }
};

TEST_F(UtestFftsPlusUpdate, GetUpdater) {
  EXPECT_EQ(FftsPlusUpdateManager::Instance().GetUpdater("AIC_AIV"), nullptr);
  EXPECT_NE(FftsPlusUpdateManager::Instance().GetUpdater("FFTS_TEST"), nullptr);
}
}
