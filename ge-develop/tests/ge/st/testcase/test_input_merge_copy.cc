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
#include "stub/gert_runtime_stub.h"
#include <rt_error_codes.h>

namespace ge {
class InputMergeCopy : public testing::Test {
 protected:
  void SetUp() override {
    char_t npu_collect_path[MMPA_MAX_PATH] = "valid_path";
    mmSetEnv("NPU_COLLECT_PATH", &npu_collect_path[0U], MMPA_MAX_PATH);
    ReInitGe();
    RuntimeStub::GetInstance()->input_mem_copy_batch_count_ = 0;
    RTS_STUB_SETUP();
  }
  void TearDown() override {
    unsetenv("DUMP_GE_GRAPH");
    remove("valid_path");
    RuntimeStub::GetInstance()->input_mem_copy_batch_count_ = 0;
    RTS_STUB_SETUP();
  }
};

TEST_F(InputMergeCopy, InputMergeCopy) {
  std::map<AscendString, AscendString> options;
  options["ge.exec.input_fusion_size"] = "1024";
  Session sess(options);

  auto g1 = GraphFactory::BuildInputMergeCopyGraph();
  EXPECT_EQ(sess.AddGraph(1, g1), SUCCESS);

  // input fusion size: 1024, data1, refdata1 merged copy(h2h2d), data0, data2 non-merge-copy(h2d)
  std::vector<ge::Tensor> g1_inputs;
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({8, 3, 16, 16})));  // data0, 24608
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({2, 2, 3, 2})));    // data1, 128
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({8, 3, 16, 16})));  // data2, 24608
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({2, 2, 3, 2})));    // refdata1, 128

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

TEST_F(InputMergeCopy, InputMergeAndBatchCopy) {
  std::map<AscendString, AscendString> options;
  options["ge.inputBatchCpy"] = "1";
  options["ge.exec.input_fusion_size"] = "200";
  Session sess(options);

  auto g1 = GraphFactory::BuildInputMergeCopyGraph();
  EXPECT_EQ(sess.AddGraph(1, g1), SUCCESS);

  // input fusion size: 1024, data1, refdata1 merged copy(h2h2d), data0, data2 non-merge-copy(h2d)
  std::vector<ge::Tensor> g1_inputs;
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({8, 3, 16, 16})));  // data0, 24608
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({2, 2, 3, 2})));    // data1, 128
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({8, 3, 16, 16})));  // data2, 24608
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({2, 2, 3, 2})));    // refdata1, 128

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

TEST_F(InputMergeCopy, InputMergeAndBatchCopyButOnlyOneForBatch) {
  std::map<AscendString, AscendString> options;
  options["ge.inputBatchCpy"] = "1";
  options["ge.exec.input_fusion_size"] = "200";
  Session sess(options);

  auto g1 = GraphFactory::BuildInputMergeCopyGraph();
  EXPECT_EQ(sess.AddGraph(1, g1), SUCCESS);

  // input fusion size: 1024, data1, refdata1 merged copy(h2h2d), data0, data2 non-merge-copy(h2d)
  std::vector<ge::Tensor> g1_inputs;
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({2, 2, 3, 2})));  // data0, 128
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({2, 2, 3, 2})));    // data1, 128
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({8, 3, 16, 16})));  // data2, 24608
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({2, 2, 3, 2})));    // refdata1, 128

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

TEST_F(InputMergeCopy, InputMergeAndBatchCopyBatchNotSupported) {
  std::map<AscendString, AscendString> options;
  options["ge.inputBatchCpy"] = "1";
  options["ge.exec.input_fusion_size"] = "200";
  Session sess(options);

  auto g1 = GraphFactory::BuildInputMergeCopyGraph();
  EXPECT_EQ(sess.AddGraph(1, g1), SUCCESS);

  // input fusion size: 1024, data1, refdata1 merged copy(h2h2d), data0, data2 non-merge-copy(h2d)
  std::vector<ge::Tensor> g1_inputs;
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({8, 3, 16, 16})));  // data0, 24608
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({2, 2, 3, 2})));    // data1, 128
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({8, 3, 16, 16})));  // data2, 24608
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({2, 2, 3, 2})));    // refdata1, 128

  Synchronizer sync;
  Status run_ret;
  RTS_STUB_RETURN_VALUE(rtsMemcpyBatch, rtError_t, ACL_ERROR_RT_FEATURE_NOT_SUPPORT);
  auto ret = sess.RunGraphAsync(1, g1_inputs, [&run_ret, &sync](Status result, std::vector<ge::Tensor> &) {
    run_ret = result;
    sync.OnDone();
  });
  EXPECT_EQ(ret, SUCCESS);
  sync.WaitFor(10);
  EXPECT_EQ(run_ret, SUCCESS);
}

TEST_F(InputMergeCopy, InputBatchCopy) {
  std::map<AscendString, AscendString> options;
  options["ge.inputBatchCpy"] = "1";
  options["ge.exec.input_fusion_size"] = "0";
  Session sess(options);

  auto g1 = GraphFactory::BuildInputMergeCopyGraph();
  EXPECT_EQ(sess.AddGraph(1, g1), SUCCESS);

  // input fusion size: 1024, data1, refdata1 merged copy(h2h2d), data0, data2 non-merge-copy(h2d)
  std::vector<ge::Tensor> g1_inputs;
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({8, 3, 16, 16})));  // data0, 24608
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({2, 2, 3, 2})));    // data1, 128
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({8, 3, 16, 16})));  // data2, 24608
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({2, 2, 3, 2})));    // refdata1, 128

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

TEST_F(InputMergeCopy, InputBatchCopyBatchNotSupportedThenFallback) {
  std::map<AscendString, AscendString> options;
  options["ge.inputBatchCpy"] = "1";
  options["ge.exec.input_fusion_size"] = "0";
  Session sess(options);

  auto g1 = GraphFactory::BuildInputMergeCopyGraph();
  EXPECT_EQ(sess.AddGraph(1, g1), SUCCESS);

  // input fusion size: 1024, data1, refdata1 merged copy(h2h2d), data0, data2 non-merge-copy(h2d)
  std::vector<ge::Tensor> g1_inputs;
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({8, 3, 16, 16})));  // data0, 24608
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({2, 2, 3, 2})));    // data1, 128
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({8, 3, 16, 16})));  // data2, 24608
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({2, 2, 3, 2})));    // refdata1, 128

  Synchronizer sync;
  Status run_ret;
  RTS_STUB_RETURN_VALUE(rtsMemcpyBatch, rtError_t, ACL_ERROR_RT_FEATURE_NOT_SUPPORT);
  auto ret = sess.RunGraphAsync(1, g1_inputs, [&run_ret, &sync](Status result, std::vector<ge::Tensor> &) {
    run_ret = result;
    sync.OnDone();
  });
  EXPECT_EQ(ret, SUCCESS);
  sync.WaitFor(10);
  EXPECT_EQ(run_ret, SUCCESS);
}

TEST_F(InputMergeCopy, InputBatchCopyBatchFailed) {
  std::map<AscendString, AscendString> options;
  options["ge.inputBatchCpy"] = "1";
  options["ge.exec.input_fusion_size"] = "0";
  Session sess(options);

  auto g1 = GraphFactory::BuildInputMergeCopyGraph();
  EXPECT_EQ(sess.AddGraph(1, g1), SUCCESS);

  // input fusion size: 1024, data1, refdata1 merged copy(h2h2d), data0, data2 non-merge-copy(h2d)
  std::vector<ge::Tensor> g1_inputs;
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({8, 3, 16, 16})));  // data0, 24608
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({2, 2, 3, 2})));    // data1, 128
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({8, 3, 16, 16})));  // data2, 24608
  g1_inputs.emplace_back(TensorAdapter::AsTensor(*GenerateTensor({2, 2, 3, 2})));    // refdata1, 128

  Synchronizer sync;
  Status run_ret;
  RTS_STUB_RETURN_VALUE(rtsMemcpyBatch, rtError_t, -1);
  auto ret = sess.RunGraphAsync(1, g1_inputs, [&run_ret, &sync](Status result, std::vector<ge::Tensor> &) {
    run_ret = result;
    sync.OnDone();
  });
  EXPECT_EQ(ret, SUCCESS);
  sync.WaitFor(10);
  EXPECT_NE(run_ret, SUCCESS);
}
}
