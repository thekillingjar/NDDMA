/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_DEPLOY_EXECUTOR_EVENT_HANDLER_H_
#define AIR_RUNTIME_DEPLOY_EXECUTOR_EVENT_HANDLER_H_

#include <string>
#include <vector>
#include <memory>
#include "executor/executor_context.h"
#include "proto/deployer.pb.h"

namespace ge {
class EventHandler {
 public:
  Status Initialize();
  void Finalize();
  void HandleEvent(deployer::ExecutorRequest &request, deployer::ExecutorResponse &response);
  void SetBaseDir(const std::string &base_dir);

 private:
  void HandleSyncVarManagerRequest(deployer::ExecutorRequest &request, deployer::ExecutorResponse &response);
  void HandleBatchLoadRequest(deployer::ExecutorRequest &request, deployer::ExecutorResponse &response);
  void HandleUnloadRequest(deployer::ExecutorRequest &request, deployer::ExecutorResponse &response) const;
  void HandleClearModelRequest(deployer::ExecutorRequest &request, deployer::ExecutorResponse &response);
  void HandleDataFlowExceptionNotifyRequest(deployer::ExecutorRequest &request, deployer::ExecutorResponse &response);
  void DoClearModel(const std::vector<uint32_t> &davinci_model_runtime_ids,
                    const std::vector<ExecutorContext::ModelHandle *> &dynamic_model_handles,
                    const int32_t clear_msg_type,
                    deployer::ExecutorResponse &response) const;
  void DoClearModelPara(const std::vector<uint32_t> &davinci_model_runtime_ids,
                        const std::vector<ExecutorContext::ModelHandle *> &dynamic_model_handles,
                        const int32_t clear_msg_type,
                        deployer::ExecutorResponse &response) const;
  Status DoDataFlowExceptionNotify(const std::vector<uint32_t> &davinci_model_runtime_ids,
                                   const std::vector<ExecutorContext::ModelHandle *> &dynamic_model_handles,
                                   const deployer::DataFlowExceptionNotify &exception_notify) const;
  Status BatchLoadModels(deployer::ExecutorRequest &request);
  Status BatchParseAndLoadModels(const deployer::ExecutorRequest_BatchLoadModelMessage &model_messages);
  void HandleProfInfo(deployer::ExecutorRequest &request, deployer::ExecutorResponse &response);

  std::unique_ptr<ExecutorContext> context_;
  bool once_inited_ = false;
  bool rank_table_deployed_ = false;
  std::string base_dir_;
};
}  // namespace ge

#endif  // AIR_RUNTIME_DEPLOY_EXECUTOR_EVENT_HANDLER_H_
