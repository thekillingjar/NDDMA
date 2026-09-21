/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DEPLOY_EXECFWK_UDF_PROXY_CLIENT_H_
#define AIR_RUNTIME_HETEROGENEOUS_DEPLOY_EXECFWK_UDF_PROXY_CLIENT_H_

#include "deploy/execfwk/udf_executor_client.h"

namespace ge {
class UdfProxyClient : public UdfExecutorClient {
 public:
  explicit UdfProxyClient(int32_t device_id);
  ~UdfProxyClient() override = default;
  Status Initialize() override;
  ProcStatus GetSubProcStat() override;
  Status LoadModel(deployer::ExecutorRequest_BatchLoadModelMessage load_model_desc) override;
  Status PreProcess(const std::vector<deployer::SubmodelDesc> &model_descs,
                    const std::string &base_dir) override {
    (void) model_descs;
    (void) base_dir;
    return SUCCESS;
  }
 protected:
  Status DoMemGrpAddProc(const std::string &group_name, const pid_t child_pid) override;
 private:
  void NotifySubprocessShutdown(pid_t pid) const override;
  Status ShutdownSubprocess(pid_t pid, uint32_t wait_time_in_sec = 3U) const override;
  Status LoadProcess(const deployer::ExecutorRequest_BatchLoadModelMessage &load_model_desc,
                     const std::string &msg_file_path, const std::string &group_name) override;
  Status ForkChildProcess(const deployer::ExecutorRequest_LoadModelRequest &model_req, const std::string &file_path,
                          const std::string &group_name, const SubProcessParams &params, pid_t &child_pid) override;
  Status PreLoadProcess(const std::vector<deployer::ExecutorRequest_BatchLoadModelMessage> &load_model_descs,
                        const std::vector<std::string> &load_model_msg_paths) const;
  Status GenerateCmdStrSupportUnpack(const std::vector<std::string> &load_model_msg_paths,
      const std::vector<std::string> &saved_model_paths, const std::vector<std::string> &local_builtin_model_paths,
      std::string &path_list_str) const;
  bool is_support_builtin_udf_launch_ = false;
};
}  // namespace ge
#endif
