/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_DEPLOY_DATAFLOW_EXCEPTION_HANDLER_H_
#define EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_DEPLOY_DATAFLOW_EXCEPTION_HANDLER_H_

#include <set>
#include <map>
#include <mutex>
#include <thread>
#include "ge/ge_data_flow_api.h"
#include "dflow/base/deploy/model_deployer.h"
#include "common/blocking_queue.h"
#include "proto/task.pb.h"
#include "inner_process_msg_forwarding.h"

namespace ge {
constexpr uint32_t kExceptionTypeOccured = 0;
constexpr uint32_t kExceptionTypeExpired = 1;
class DataFlowExceptionHandler {
 public:
  explicit DataFlowExceptionHandler(std::function<void(const UserExceptionNotify &)> exception_notify_callback);
  ~DataFlowExceptionHandler();

  Status Initialize(InnerProcessMsgForwarding &process_forwarding);
  void Finalize();

  bool TakeWaitModelIoException(DataFlowInfo &info);

  bool IsModelIoIgnoreTransId(uint64_t trans_id);
  using ModelIoExpTransIdCallbackFunc = std::function<void(uint64_t trans_id, uint32_t type)>;
  void RegisterModelIoExpTransIdCallback(const ModelIoExpTransIdCallbackFunc &callback);

 private:
  void Process();
  Status NotifyException(const domi::DataFlowException &data_flow_exception);
  void ProcessException(const domi::DataFlowException &data_flow_exception);
  void NotifyModelIO(const domi::DataFlowException &data_flow_exception, uint32_t type);
  void NotifyExecutor(const domi::DataFlowException &data_flow_exception, uint32_t type);

  std::function<void(const UserExceptionNotify &)> exception_notify_callback_;
  BlockingQueue<domi::DataFlowException> process_queue_;
  std::thread process_thread_;
  // guard all_exceptions_
  std::mutex mt_for_all_exception_;
  // key <trans_id, scope>
  std::map<std::pair<uint64_t, std::string>, domi::DataFlowException> all_exceptions_;

  // guard model_io_exception_all_ and model_io_exception_wait_report_
  std::mutex mt_for_model_io_;

  // model io just focus scope empty trans id.
  std::set<uint64_t> model_io_exception_all_;
  // wait to report to user.
  std::set<uint64_t> model_io_exception_wait_report_;
  std::vector<ModelIoExpTransIdCallbackFunc> model_io_callback_list_;
};
}  // namespace ge
#endif  // EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_DEPLOY_DATAFLOW_EXCEPTION_HANDLER_H_
