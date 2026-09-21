/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MODELINFERENCE_H
#define MODELINFERENCE_H

#include <acl_base.h>
#include <acl_rt.h>
#include <condition_variable>
#include "ge/ge_api.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <queue>
#include <thread>
#include <map>
#include <atomic>

namespace gerec {
struct InferenceParams {
  std::string model_path_;
  std::string model_type_;
  int32_t multi_instance_num_{1};
  std::string ai_core_num_;
  std::map<ge::AscendString, ge::AscendString> parser_params_;
  bool enable_input_batch_cpy_{false};
  size_t global_max_queue_size_{50000};
};

class ModelInference {
public:
  class Builder;
  explicit ModelInference(const InferenceParams &inference_params);
  ~ModelInference();

  ge::Status Init();
  using Callback = std::function<void(
      std::shared_ptr<std::vector<gert::Tensor>> outputs,
      std::shared_ptr<std::vector<gert::Tensor>> inputs,
      bool status,
      long long exec_us // 执行时延（微秒）
      )>;

  ge::Status RunGraphAsync(const std::shared_ptr<std::vector<gert::Tensor>> &host_inputs,
                           const std::shared_ptr<std::vector<gert::Tensor>> &host_outputs,
                           const Callback &callback);

  class Builder {
  public:
    explicit Builder(const std::string& modelPath, const std::string& modelType);
    Builder &AiCoreNum(const std::string &aiCoreNum);
    Builder &InputBatchCopy(bool enableBatchH2D);
    Builder &MultiInstanceNum(int32_t multiInstanceNum);
    Builder &MaxQueueSize(size_t maxSize);
    Builder &GraphParserParams(const std::map<ge::AscendString, ge::AscendString> &parserParams);
    std::unique_ptr<ModelInference> Build();

  private:
    InferenceParams infer_params_;
  };

private:
  struct GraphWorker {
    int32_t deviceId;
    uint32_t graphId;
    std::unique_ptr<ge::Graph> graph;
    std::thread worker;
    std::queue<std::function<void()>> tasks;
    std::mutex mtx;
    std::condition_variable cv;
    aclrtStream stream;
    bool batchCopy{false};
    bool stop{false};

    size_t max_queue_size{50000};

    void Run(const std::shared_ptr<ge::Session> &session);

    bool Enqueue(std::function<void()> task);
  };

  struct GraphTask {
    std::shared_ptr<ge::Session> session;
    GraphWorker *worker;
    std::shared_ptr<std::vector<gert::Tensor>> host_inputs;
    std::shared_ptr<std::vector<gert::Tensor>> host_outputs;
    Callback callback;
    void operator()();
  };

  std::string model_path_;
  std::string model_type_;
  int32_t multi_instance_num_{1};
  std::string ai_core_num_;
  std::shared_ptr<ge::Session> session_;
  std::map<ge::AscendString, ge::AscendString> parser_params_;
  bool enable_input_batch_cpy_{false};

  std::vector<std::unique_ptr<GraphWorker>> workers_;
  // 用于轮询选择 worker
  std::atomic<size_t> next_worker_index_{0};

  size_t global_max_queue_size_{50000};
};

struct MemcpyBatchParam {
  std::vector<void *> dsts;
  std::vector<size_t> destMaxs;
  std::vector<void *> srcs;
  std::vector<size_t> sizes;
  size_t numBatches;
  std::vector<aclrtMemcpyBatchAttr> attrs;
  std::vector<size_t> attrsIndexes;
  size_t numAttrs;
};
} // namespace gerec

#endif  // MODELINFERENCE_H