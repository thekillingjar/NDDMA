/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DEPLOY_EXECFWK_UDF_EXECUTOR_CLIENT_H_
#define AIR_RUNTIME_HETEROGENEOUS_DEPLOY_EXECFWK_UDF_EXECUTOR_CLIENT_H_

#include <atomic>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <mutex>
#include "pne_executor_client.h"
#include "common/message_handle/message_client.h"

namespace ge {
class UdfExecutorClient : public PneExecutorClient {
 public:
  struct SubProcessParams {
    bool is_builtin = false;
    uint32_t requeset_queue_id = UINT32_MAX;
    uint32_t response_queue_id = UINT32_MAX;
    std::string npu_sched_model;
  };

  explicit UdfExecutorClient(int32_t device_id) : PneExecutorClient(device_id){};
  ~UdfExecutorClient() override = default;
  Status Initialize() override;
  Status Finalize() override;
  static Status PreprocessUdfTarPackage(const std::set<std::string> &local_udf_saved_path,
                                         const std::string &local_udf_load_path);
  Status SyncVarManager(deployer::ExecutorRequest_SyncVarManageRequest sync_var_manage_desc) override;
  bool SupportSyncVarManager() override {
    return false;
  };
  Status PreProcess(const std::vector<deployer::SubmodelDesc> &model_descs,
                    const std::string &base_dir) override;
  Status LoadModel(deployer::ExecutorRequest_BatchLoadModelMessage load_model_desc) override;
  Status UpdateProfilingFromExecutor(deployer::ExecutorRequest_UpdateProfRequest &prof_message) override {
    (void) prof_message;
    return SUCCESS;
  }
  
  Status UnloadModel(uint32_t model_id) override;
  ProcStatus GetSubProcStat() override;
  void GetAbnormalModelInsName(std::map<uint32_t, std::vector<std::string>> &abnormal_model_instances_name) override;
  void UpdateModelInsNameByPid(pid_t pid);
  Status ClearModelRunningData(uint32_t model_id, int32_t type, const std::set<int32_t> &device_ids) override;
  Status DataFlowExceptionNotify(const deployer::DataFlowExceptionNotifyRequest &req_body) override;

 protected:
  virtual void NotifySubprocessShutdown(pid_t pid) const;
  virtual Status ShutdownSubprocess(pid_t pid, uint32_t wait_time_in_sec = 3U) const;
  Status StartUdfProcess(const std::vector<deployer::ExecutorRequest_BatchLoadModelMessage> &load_model_descs,
                         const std::vector<std::string> &msg_file_paths);
  virtual Status LoadProcess(const deployer::ExecutorRequest_BatchLoadModelMessage &load_model_desc,
                             const std::string &msg_file_path,
                             const std::string &group_name);

  virtual Status ForkChildProcess(const deployer::ExecutorRequest_LoadModelRequest &model_req,
                                  const std::string &file_path, const std::string &group_name,
                                  const SubProcessParams &params, pid_t &child_pid);
  virtual Status DoMemGrpAddProc(const std::string &group_name, const pid_t child_pid);
  Status DoGrantQueues(int32_t pid, const std::vector<DeployQueueAttr> &queue_attrs) override;
  Status DoBindHostPid(const int32_t pid) override;
  Status GrantQueuesForUdf(const deployer::ExecutorRequest_BatchLoadModelMessage &load_model_desc, const pid_t pid,
                           int32_t process_device_type);
  Status GrantDynamicSchedQueuesForUdf(const deployer::ExecutorRequest_BatchLoadModelMessage &load_model_desc,
                                       const pid_t pid, int32_t process_device_type);
  void AddPidToModelInstanceNameRelation(pid_t child_pid,
      const deployer::ExecutorRequest_BatchLoadModelMessage& model_desc);
  Status ForkAndInit(const deployer::ExecutorRequest_BatchLoadModelMessage& model_desc,
                     const std::string &group_name, const std::string &message_path,
                     pid_t &child_pid);
  Status DistributeAndSerializeModelDescs(deployer::ExecutorRequest_BatchLoadModelMessage load_model_desc,
      std::vector<deployer::ExecutorRequest_BatchLoadModelMessage> &load_model_descs,
      std::vector<std::string> &msg_file_paths);
  ProcStatus GetSubProcStatStartByPm();
  ProcStatus PostProcSubProcessStatus(const std::map<pid_t, ProcStatus> &stats);
  std::mutex mutex_;  // lock model_id_to_pids_ and load_model_message_paths_
  std::map<uint32_t, std::vector<pid_t>> model_id_to_pids_;
  std::map<pid_t, uint32_t> pid_to_model_id_;
  // key: pid, value pair<request queue id, queue id>
  std::map<pid_t, std::shared_ptr<ExecutorMessageClient>> pid_to_message_client_;
  bool is_proxy_ = false;
 private:
  Status SerializeLoadModelMessageToFile(const deployer::ExecutorRequest_BatchLoadModelMessage &load_model_desc,
                                         const std::string &base_dir, std::string &file_path);

  Status GenLoadModelFile(const std::string &model_path, const std::string &base_dir,
                          const std::string &load_model_message, std::string &file_path);

  static Status DistributeModel(const deployer::ExecutorRequest_BatchLoadModelMessage &load_model_desc,
                                std::vector<deployer::ExecutorRequest_BatchLoadModelMessage> &load_model_descs);

  Status CheckDevPidStatus(const int32_t device_id, const pid_t &pid);
  void ShutdownByRelatedDeviceIds(const std::set<int32_t> &device_ids);
  void RecordPidWithNpuDeviceId(int32_t queue_owner_pid,
      deployer::ExecutorRequest_ModelQueuesAttrs model_queues_attrs);
  std::string GetPidModelInstanceName(pid_t child_pid);
  Status GetModelMessageClients(uint32_t root_model_id,
                                std::map<pid_t, std::shared_ptr<ExecutorMessageClient>> &pid_and_message_clients);
  static Status GrantAndGetUdfAicpuPid(int32_t phy_device_id, pid_t udf_pid, pid_t &aicpu_pid);
  static Status NotifyUdfContinue(const std::shared_ptr<ExecutorMessageClient> &message_client, pid_t udf_pid,
                                  pid_t aicpu_pid);

 private:
  static std::atomic<uint64_t> load_model_message_id_;
  std::vector<std::string> load_model_message_paths_;
  std::mutex stat_mutex_;
  std::map<pid_t, ProcStatus> sub_proc_stat_flag_;
  std::mutex pid_to_model_mutex_;
  std::map<pid_t, std::vector<std::string>> pid_to_model_instances_name_;
  std::mutex abnormal_model_mutex_;
  std::map<uint32_t, std::vector<std::string>> abnormal_model_instances_name_;
  std::mutex related_mutex_;
  std::map<int32_t, std::set<int32_t>> npu_device_id_related_pids_;
  // guard bind pids
  std::mutex pid_mutex_;
  std::set<int32_t> bind_pids_;
  std::atomic<uint32_t> get_proc_status_failed_times_{0U};
};
}  // namespace ge
#endif  // AIR_RUNTIME_HETEROGENEOUS_DEPLOY_EXECFWK_UDF_EXECUTOR_CLIENT_H_
