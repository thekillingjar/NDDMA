/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_EXECUTOR_RT_V2_STAGE_STATE_H_
#define GE_HYBRID_EXECUTOR_RT_V2_STAGE_STATE_H_

#include "rt_v2_executor_interface.h"
#include "rt_v2_simple_executor.h"
#include "rt_v2_utils.h"

#include <thread>

namespace gert {
struct StageTask {
  enum class Signal { RUN, STOP };
  // Task type
  Signal signal = Signal::RUN;
  // Steps to run
  size_t num_steps = 1U;
  // Model input args
  gert::Tensor **inputs = nullptr;
  size_t num_input = 0U;
  gert::Tensor **outputs = nullptr;
  size_t num_output = 0U;
};
struct StageNotification {
  StageTask GetTask() {
    // We use lock here for thread able to sleeping when no task
    std::unique_lock<std::mutex> lk(lk_);
    cv_.wait(lk, [this]() { return !q_.empty(); });
    auto signal = q_.front();
    q_.pop();
    return signal;
  }

  void Notify(const StageTask &args) {
    done_ = false;
    std::unique_lock<std::mutex> lk(lk_);
    q_.push(args);
    cv_.notify_one();
  }

  void Done() {
    done_ = true;
  }

  void Wait() {  // Wait until Done() was called
    while (!done_) {
    }
  }

 private:
  std::queue<StageTask> q_;
  std::mutex lk_;
  std::condition_variable cv_;
  std::atomic_bool done_{false};
};
class StageState {
 public:
  static std::unique_ptr<StageState> Create(const ge::GeRootModelPtr &model, RtSession *session);

  ge::Status Load(const gert::ModelExecuteArg &arg, const gert::ModelLoadArg &load_arg, bool daemon = false);

  using CtxInitializer = std::function<void()>;
  ge::Status LoadAsDaemon(const gert::ModelExecuteArg &arg, const gert::ModelLoadArg &load_arg,
                          CtxInitializer ctx_initializer, std::shared_ptr<StageNotification> &notification);

  ge::Status Run(const StageTask &args);

  void Reset();

  ge::Status Stop();

  bool IsErrorStatus() const;

  const std::string &Id() const;

  static ge::Status MappingStageIO(const std::map<ge::NodePtr, StageState *> &named_stages);

  ge::Status GetConsumedModelInputDesc(std::map<size_t, ModelIoDesc> &descs) const;

  ge::Status GetProducedModelOutputDesc(std::map<size_t, ModelIoDesc> &descs) const;

  void AddSubscriber(SubscriberExtendInfo extend_info) const {
    executor_->AddSubscriber(extend_info);
  }

  ~StageState();

 private:
  explicit StageState(std::string id, std::unique_ptr<gert::RtV2ExecutorInterface> &&executor);

  ge::Status ConsumeInputs();

  ge::Status RunTask();

  ge::Status RunInternal(const StageTask &args);

  ge::Status ProduceOutputs();

  ge::Status GetOutput(size_t index, gert::Tensor &tensor);

  static void ModelDescToTensorSpec(const gert::ModelIoDesc *desc, size_t num, std::vector<gert::Tensor> &tensors);

  ge::Status AssembleModelInputs(gert::Tensor **inputs, size_t input_num);

  void FreeInterimOutputs();

  // Basic meta
  std::string id_;
  std::atomic_size_t running_step_{0};   // Stage current running steps
  std::atomic_size_t done_steps_{0};     // Stage done steps
  std::vector<uint8_t> outputs_buffer_;  // Buffer for store stage outputs

  std::atomic_bool error_status_{false};

  // Input/output dependencies
  size_t num_inputs_{0U};
  size_t num_outputs_{0U};

  // {s : {out,in}} means input stage 's' output 'out' flow to current stage input 'in'
  std::map<StageState *, std::map<size_t, std::set<size_t>>> stage_input_infos_;  // Ordered
  std::set<StageState *> output_stages_;

  // map[i] = j means feed input 'i' flow to current stage input 'j'
  std::map<size_t, std::set<size_t>> feed_2_stage_inputs_;
  // map[i] = j means stage output 'j' flow to model fetch output 'i'
  std::map<size_t, size_t> fetch_2_stage_outputs_;

  // Runtime param
  std::thread worker_;
  std::shared_ptr<StageNotification> notification_ = nullptr;
  std::unique_ptr<gert::RtV2ExecutorInterface> executor_;
  gert::ModelExecuteArg execute_arg_;
  std::vector<gert::Tensor *> executor_inputs_;
  std::vector<gert::Tensor> executor_inputs_holder_;
  std::vector<gert::Tensor *> executor_outputs_;
  std::vector<gert::Tensor> executor_outputs_holder_;

  // Stage outputs
  std::vector<gert::Tensor> stage_outputs_;
};
}  // namespace gert

#endif  // GE_HYBRID_EXECUTOR_RT_V2_STAGE_STATE_H_
