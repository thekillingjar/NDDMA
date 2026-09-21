/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "utils/graph_factory.h"
#include <gtest/gtest.h>
#include <map>
#include "ge/ge_api.h"
#include "utils/synchronizer.h"
#include "ge_running_env/tensor_utils.h"
#include "utils/tensor_adapter.h"
#include "graph/utils/graph_utils.h"
#include "mmpa/mmpa_api.h"
#include "init_ge.h"

namespace ge {
class V1LoopSt : public testing::Test {
 protected:
  void SetUp() override {
    char_t npu_collect_path[MMPA_MAX_PATH] = "valid_path";
    mmSetEnv("NPU_COLLECT_PATH", &npu_collect_path[0U], MMPA_MAX_PATH);
    ReInitGe();
  }
  void TearDown() override {
    unsetenv("DUMP_GE_GRAPH");
    remove("valid_path");
  }
};

TEST_F(V1LoopSt, BasicV1LoopSuccess) {
  std::map<AscendString, AscendString> options;
  Session sess(options);

  auto g1 = GraphFactory::BuildV1LoopGraph1();
  EXPECT_EQ(sess.AddGraph(1, g1), SUCCESS);

  std::vector<ge::Tensor> g1_inputs;
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor(DT_INT32, {})));
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({8,3,24,1})));
  Synchronizer sync;
  Status run_ret;
  auto ret = sess.RunGraphAsync(1, g1_inputs, [&run_ret, &sync](Status result, std::vector<ge::Tensor> &) {
    run_ret = result;
    sync.OnDone();
  });
  EXPECT_EQ(ret, SUCCESS);
  sync.WaitFor(10);
  EXPECT_EQ(run_ret, SUCCESS);
}

TEST_F(V1LoopSt, V1LoopSuccess_CtrlEnterIn) {
  std::map<AscendString, AscendString> options;
  Session sess(options);

  auto g1 = GraphFactory::BuildV1LoopGraph2_CtrlEnterIn();

  EXPECT_EQ(sess.AddGraph(1, g1), SUCCESS);

  std::vector<ge::Tensor> g1_inputs;
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor(DT_INT32, {})));
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({8,3,24,1})));
  Synchronizer sync;
  Status run_ret;
  auto ret = sess.RunGraphAsync(1, g1_inputs, [&run_ret, &sync](Status result, std::vector<ge::Tensor> &) {
    run_ret = result;
    sync.OnDone();
  });
  EXPECT_EQ(ret, SUCCESS);
  sync.WaitFor(10);
  EXPECT_EQ(run_ret, SUCCESS);
}

TEST_F(V1LoopSt, V1LoopSuccess_DataEnterIn_BypassMerge) {
  std::map<AscendString, AscendString> options;
  Session sess(options);

  auto g1 = GraphFactory::BuildV1LoopGraph4_DataEnterInByPassMerge();

  EXPECT_EQ(sess.AddGraph(1, g1), SUCCESS);

  std::vector<ge::Tensor> g1_inputs;
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor(DT_INT32, {})));
  Synchronizer sync;
  Status run_ret;
  auto ret = sess.RunGraphAsync(1, g1_inputs, [&run_ret, &sync](Status result, std::vector<ge::Tensor> &) {
    run_ret = result;
    sync.OnDone();
  });
  EXPECT_EQ(ret, SUCCESS);
  sync.WaitFor(10);
  EXPECT_EQ(run_ret, SUCCESS);
}

TEST_F(V1LoopSt, V1LoopSuccess_EnterCtrlConst) {
  std::map<AscendString, AscendString> options;
  Session sess(options);

  auto g1 = GraphFactory::BuildV1LoopGraph3_CtrlEnterIn2();

  EXPECT_EQ(sess.AddGraph(1, g1), SUCCESS);

  std::vector<ge::Tensor> g1_inputs;
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor(DT_INT32, {})));
  Synchronizer sync;
  Status run_ret;
  auto ret = sess.RunGraphAsync(1, g1_inputs, [&run_ret, &sync](Status result, std::vector<ge::Tensor> &) {
    run_ret = result;
    sync.OnDone();
  });
  EXPECT_EQ(ret, SUCCESS);
  sync.WaitFor(10);
  EXPECT_EQ(run_ret, SUCCESS);
}
}
