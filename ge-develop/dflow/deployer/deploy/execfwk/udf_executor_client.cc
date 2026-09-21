/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "udf_executor_client.h"

#include <regex>
#include <sys/file.h>
#include <signal.h>
#include "common/checker.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "common/config/configurations.h"
#include "common/mem_grp/memory_group_manager.h"
#include "common/subprocess/subprocess_manager.h"
#include "dflow/base/utils/process_utils.h"
#include "common/data_flow/event/proxy_event_manager.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "mmpa/mmpa_api.h"
#include "dflow/inc/data_flow/model/pne_model.h"
#include "proto/udf_def.pb.h"
#include "deploy/flowrm/tsd_client.h"
#include "deploy/flowrm/flowgw_client.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "common/utils/rts_api_utils.h"

namespace ge {
namespace {
const std::string kProcessName = "udf_executor";

// udf release pkg unpack limit size
constexpr int32_t kUdfReleasePkgUnpackLimitSize = 200 * 1024 * 1024;

constexpr uint32_t kBindAllDevice = 0xffffffff;

constexpr uint32_t kUdfExecutorShutdownWaitTimeInSec = 10U;
constexpr uint32_t kRspWaitTimeInSec = 1200U; // 20min same as default grpc timeout

constexpr size_t kMaxThreadNum = 12UL;

constexpr size_t kMaxThreadNumToSendMsg = 32UL;

const std::string kUdfResourceSubDir = "udf_resource";
PneExecutorClientCreatorRegistrar<UdfExecutorClient> __attribute__((unused)) udf_client_reg(PNE_ID_UDF);
}  // namespace

std::atomic<uint64_t> UdfExecutorClient::load_model_message_id_{0U};

Status UdfExecutorClient::Initialize() {
  return SUCCESS;
}

Status UdfExecutorClient::PreProcess(const std::vector<deployer::SubmodelDesc> &model_descs,
                                     const std::string &base_dir) {
  std::set<std::string> local_udf_saved_path;
  std::string local_udf_load_path;
  for (const auto &submodel_desc : model_descs) {
    // 用户自定义udf且在当前节点部署，获取原始tar包位置和目标解压位置
    local_udf_load_path = submodel_desc.model_path();
    if (!submodel_desc.is_builtin_udf() && submodel_desc.replica_idx_on_node() == 0U) {
      const auto saved_model_path = submodel_desc.is_remote_model() ?
                                    base_dir + submodel_desc.saved_model_file_path():
                                    submodel_desc.saved_model_file_path();
      (void)local_udf_saved_path.insert(saved_model_path);
      GELOGD("Get saved model path[%s].", saved_model_path.c_str());
    }
  }

  // path is empty if there is no udf in model
  if (!local_udf_load_path.empty()) {
    local_udf_load_path = base_dir + local_udf_load_path;
  }
  GELOGI("Get local udf size[%zu], untar to path[%s].", local_udf_saved_path.size(), local_udf_load_path.c_str());
  GE_TIMESTAMP_START(HostUdfUntarProcess);
  GE_CHK_STATUS_RET(UdfExecutorClient::PreprocessUdfTarPackage(local_udf_saved_path, local_udf_load_path),
                    "[PreLoad][Model] Failed to pre process host udf.");
  GE_TIMESTAMP_EVENT_END(HostUdfUntarProcess, "host udf do untar process");
  return SUCCESS;
}

Status UdfExecutorClient::DoGrantQueues(int32_t pid, const std::vector<DeployQueueAttr> &queue_attrs) {
  // each model starts a process, no need to protected by lock
  for (const auto &queue_attr : queue_attrs) {
    GE_CHK_STATUS_RET(FlowGwClient::GrantQueue(static_cast<uint32_t>(queue_attr.device_id), queue_attr.queue_id,
                                               pid, GrantType::kReadAndWrite),
                      "Grant queue failed, device id=%d, queue id=%u, pid = %d",
                      queue_attr.device_id, queue_attr.queue_id, pid);
  }
  return SUCCESS;
}

Status UdfExecutorClient::DoBindHostPid(const int32_t pid) {
  std::lock_guard<std::mutex> guard(pid_mutex_);
  const auto &it = bind_pids_.find(pid);
  if (it == bind_pids_.cend()) {
    GE_CHK_STATUS_RET(BindHostPid(pid), "Failed to bind host pid");
    bind_pids_.emplace(pid);
  }
  return SUCCESS;
}

Status UdfExecutorClient::DoMemGrpAddProc(const std::string &group_name, const pid_t child_pid) {
  GE_CHK_STATUS_RET(MemoryGroupManager::GetInstance().MemGrpAddProc(group_name, child_pid, false, true),
                    "Failed to add group[%s] to process[%d].", group_name.c_str(), child_pid);
  return SUCCESS;
}

void UdfExecutorClient::NotifySubprocessShutdown(pid_t pid) const {
  SubprocessManager::GetInstance().NotifySubprocessShutdown(pid);
}

Status UdfExecutorClient::ShutdownSubprocess(pid_t pid, uint32_t wait_time_in_sec) const {
  return SubprocessManager::GetInstance().ShutdownSubprocess(pid, wait_time_in_sec);
}

Status UdfExecutorClient::Finalize() {
  std::unique_lock<std::mutex> guard(mutex_);
  for (const auto &load_model_message_path : load_model_message_paths_) {
    (void)remove(load_model_message_path.c_str());
  }

  for (const auto &mode_pids : model_id_to_pids_) {
    uint32_t wait_time_in_sec = kUdfExecutorShutdownWaitTimeInSec;
    for (const auto &pid : mode_pids.second) {
      if (ShutdownSubprocess(pid, wait_time_in_sec) != SUCCESS) {
        wait_time_in_sec = 0U;
      }
    }
  }
  model_id_to_pids_.clear();
  pid_to_message_client_.clear();
  return SUCCESS;
}

Status UdfExecutorClient::SyncVarManager(deployer::ExecutorRequest_SyncVarManageRequest sync_var_manage_desc) {
  (void)sync_var_manage_desc;
  return SUCCESS;
}

Status UdfExecutorClient::DistributeModel(
    const deployer::ExecutorRequest_BatchLoadModelMessage &load_model_desc,
    std::vector<deployer::ExecutorRequest_BatchLoadModelMessage> &load_model_descs) {
  load_model_descs.reserve(load_model_desc.models_size());
  for (int32_t i = 0; i < load_model_desc.models_size(); ++i) {
    deployer::ExecutorRequest_BatchLoadModelMessage load_model_desc_buff;
    load_model_desc_buff.set_root_model_id(load_model_desc.root_model_id());
    auto model_desc = load_model_desc_buff.add_models();
    GE_CHECK_NOTNULL(model_desc);
    *model_desc = load_model_desc.models(i);
    load_model_descs.emplace_back(std::move(load_model_desc_buff));
  }
  return SUCCESS;
}

Status UdfExecutorClient::StartUdfProcess(
    const std::vector<deployer::ExecutorRequest_BatchLoadModelMessage> &load_model_descs,
    const std::vector<std::string> &msg_file_paths) {
  GELOGI("Load model size %zu.", load_model_descs.size());
  const auto &group_name = MemoryGroupManager::GetInstance().GetQsMemGroupName();
  const size_t final_thread_num = (load_model_descs.size() > kMaxThreadNum) ?
                                   kMaxThreadNum : load_model_descs.size();
  ThreadPool thread_pool("ge_udf_load", final_thread_num, false);
  std::vector<std::future<Status>> process_futures;
  process_futures.reserve(load_model_descs.size());
  for (size_t i = 0UL; i < load_model_descs.size(); ++i) {
    const auto &load_model_desc = load_model_descs[i];
    const auto &msg_file_path = msg_file_paths[i];
    auto fut = thread_pool.commit([this, load_model_desc, msg_file_path, &group_name]() {
      GE_CHK_STATUS_RET(LoadProcess(load_model_desc, msg_file_path, group_name),
                        "Failed to load model.");
      return SUCCESS;
    });
    process_futures.emplace_back(std::move(fut));
  }
  Status process_ret = SUCCESS;
  for (auto &fut : process_futures) {
    auto ret = fut.get();
    if (ret != SUCCESS) {
      process_ret = ret;
    }
  }
    return process_ret;
}

Status UdfExecutorClient::CheckDevPidStatus(const int32_t device_id, const pid_t &pid) {
  if (is_proxy_) {
    ProcStatus stat = ProcStatus::NORMAL;
    GE_CHK_STATUS_RET_NOLOG(TsdClient::GetInstance().GetProcStatus(device_id, pid, stat, PNE_ID_UDF));
    return (stat == ProcStatus::EXITED) ? FAILED : SUCCESS;
  }
  std::unique_lock<std::mutex> guard(stat_mutex_);
  // sub_proc_stat_flag_ is set only when wait_pid not equal to 0
  const auto iter = sub_proc_stat_flag_.find(pid);
  if (iter != sub_proc_stat_flag_.cend()) {
    return (iter->second == ProcStatus::EXITED) ? FAILED : SUCCESS;
  }
  return SUCCESS;
}

void UdfExecutorClient::AddPidToModelInstanceNameRelation(pid_t child_pid,
    const deployer::ExecutorRequest_BatchLoadModelMessage& model_desc) {
  GELOGI("add pid to model instance name relation, models size is %zu", model_desc.models_size());
  std::lock_guard<std::mutex> guard(pid_to_model_mutex_);
  for (int32_t i = 0; i < model_desc.models_size(); ++i) {
    const auto &model = model_desc.models(i);
    pid_to_model_instances_name_[child_pid].push_back(model.model_instance_name());
    GELOGI("add pid to model instance name relation, pid[%d], model_instance_name[%s]", child_pid,
        model.model_instance_name().c_str());
  }
  return;
}

std::string UdfExecutorClient::GetPidModelInstanceName(pid_t child_pid) {
  std::lock_guard<std::mutex> guard(pid_to_model_mutex_);
  auto iter = pid_to_model_instances_name_.find(child_pid);
  if ((iter == pid_to_model_instances_name_.cend()) || (iter->second.empty())) {
    return "Unknown";
  }
  return ToString(iter->second);
}

Status UdfExecutorClient::ForkAndInit(const deployer::ExecutorRequest_BatchLoadModelMessage &model_desc,
                                      const std::string &group_name,
                                      const std::string &message_path, pid_t &child_pid) {
  GE_CHK_BOOL_RET_STATUS((model_desc.models_size() != 0), FAILED, "No model in BatchLoadModelMessage.");
  // 1. create message queue
  int32_t device_id = GetDeviceId();
  const std::string name_suffix = "udf_executor_" + std::to_string(device_id) + std::to_string(model_desc.graph_id());
  uint32_t req_msg_queue_id = UINT32_MAX;
  uint32_t rsp_msg_queue_id = UINT32_MAX;
  const auto message_client = MakeShared<ExecutorMessageClient>(device_id);
  GE_CHECK_NOTNULL(message_client);
  GE_CHK_STATUS_RET(message_client->CreateMessageQueue(name_suffix, req_msg_queue_id, rsp_msg_queue_id, is_proxy_),
                    "[Create][Message] queues for udf executor failed.");

  // 2. start UDF process and add pid to model_instances_name relation
  SubProcessParams params;
  const auto &model = model_desc.models(0);
  const auto &model_path = GetContext().base_dir + model.model_path();
  params.is_builtin = model.is_builtin_udf();
  params.requeset_queue_id = req_msg_queue_id;
  params.response_queue_id = rsp_msg_queue_id;

  const auto &model_attrs = model.attrs();
  const auto find_ret = model_attrs.find("_npu_sched_model");
  if (find_ret != model_attrs.end()) {
    GELOGD("model_instance_name[%s], npu_sched_model=[%s].", model.model_instance_name().c_str(), find_ret->second.c_str());
    params.npu_sched_model = find_ret->second;
  }

  bool need_start_aicpu = (params.npu_sched_model == "1");

  GE_CHK_STATUS_RET(ForkChildProcess(model, message_path, group_name, params, child_pid),
                    "Failed to fork child process.");
  GE_CHK_STATUS_RET(DoMemGrpAddProc(group_name, child_pid), "Failed to add group[%s] to process[%d].",
                    group_name.c_str(), child_pid);
  {
    std::unique_lock<std::mutex> guard(mutex_);
    model_id_to_pids_[model_desc.root_model_id()].emplace_back(child_pid);
    pid_to_model_id_[child_pid] = model_desc.root_model_id();
    pid_to_message_client_[child_pid] = message_client;
  }
  AddPidToModelInstanceNameRelation(child_pid, model_desc);

  // 3. initialize message client
  const auto get_stat_func = [this, device_id, child_pid]() -> Status {
    return CheckDevPidStatus(device_id, child_pid);
  };
  GE_CHK_STATUS_RET(message_client->Initialize(child_pid, get_stat_func, need_start_aicpu),
                    "Failed to initialize message client");

  pid_t io_pid = child_pid;
  if (need_start_aicpu) {
    GE_CHK_STATUS_RET(GrantAndGetUdfAicpuPid(model.phy_device_id(), child_pid, io_pid),
                      "Grant and get udf aicpu pid failed, device_id=%d, udf_pid=%d.", model.phy_device_id(),
                      child_pid);
  }
  int32_t process_device_type = need_start_aicpu ? static_cast<int32_t>(NPU) : GetContext().device_type;
  // 4. grant queues (message queue, io queue, dynamic sched queue)
  GE_CHK_STATUS_RET(GrantQueuesForUdf(model_desc, io_pid, process_device_type), "Failed to grant queues for %s[%d].",
                    kProcessName.c_str(), io_pid);
  GE_CHK_STATUS_RET(GrantDynamicSchedQueuesForUdf(model_desc, io_pid, process_device_type),
                    "Failed to grant queues for %s[%d].", kProcessName.c_str(), io_pid);
  if (need_start_aicpu) {
    GE_CHK_STATUS_RET(NotifyUdfContinue(message_client, child_pid, io_pid),
                      "Failed to notify udf continue, udf_pid=%d.", child_pid);
  }
  return SUCCESS;
}

Status UdfExecutorClient::NotifyUdfContinue(const std::shared_ptr<ExecutorMessageClient> &message_client, pid_t udf_pid,
                                          pid_t aicpu_pid) {
  deployer::ExecutorRequest notify_request;
  notify_request.set_type(deployer::ExecutorRequestType::kNotify);
  Status ret = message_client->SendRequestWithoutResponse(notify_request);
  GELOGI("grant aicpu[%d] finished, notify udf continue. udf_pid=%d, ret=%u", aicpu_pid, udf_pid, ret);
  return ret;
}

Status UdfExecutorClient::GrantAndGetUdfAicpuPid(int32_t phy_device_id, pid_t udf_pid, pid_t &aicpu_pid) {
  GE_CHK_STATUS_RET(RtsApiUtils::GetAicpuSchedulePid(phy_device_id, udf_pid, aicpu_pid),
                    "Query aicpu schedule failed, device_id=%d, udf_pid=%d.", phy_device_id, udf_pid);
  GELOGI("io will take by aicpu schedule, device_id=%d, udf_pid=%d, aicpu_pid=%d.", phy_device_id, udf_pid, aicpu_pid);

  const auto &remote_group_name = MemoryGroupManager::GetInstance().GetRemoteMemGroupName(phy_device_id);
  GE_CHK_STATUS_RET(
      MemoryGroupManager::GetInstance().RemoteMemGrpAddProc(phy_device_id, remote_group_name, aicpu_pid, false, true),
      "Failed to add group for aicpu, pid=%d, remote_group_name=%s", aicpu_pid, remote_group_name.c_str());
  return SUCCESS;
}

Status UdfExecutorClient::LoadProcess(const deployer::ExecutorRequest_BatchLoadModelMessage &load_model_desc,
                                      const std::string &msg_file_path,
                                      const std::string &group_name) {
  pid_t child_pid = -1;
  GEEVENT("Fork udf process to load model on executor start.");
  GE_CHK_STATUS_RET(ForkAndInit(load_model_desc, group_name, msg_file_path, child_pid),
                    "[Fork][Init] udf executor failed.");
  GEEVENT("Fork udf process to load model on executor success. "
          "model_type = %s, pid = %d, graph_id = %u, deployer pid = %d, device_id = %d.",
          PNE_ID_UDF.c_str(), child_pid, load_model_desc.graph_id(), GetDeployerPid(), GetDeviceId());
  return SUCCESS;
}

Status UdfExecutorClient::LoadModel(deployer::ExecutorRequest_BatchLoadModelMessage load_model_desc) {
  GEEVENT("[Load][Model] begin, model size = %d.", load_model_desc.models_size());
  std::vector<deployer::ExecutorRequest_BatchLoadModelMessage> load_model_descs;
  std::vector<std::string> msg_file_paths;
  GE_CHK_STATUS_RET(DistributeAndSerializeModelDescs(load_model_desc, load_model_descs, msg_file_paths),
                    "Distribute and serialize model desc failed.");
  GE_TIMESTAMP_START(StartUdfProcess);
  GE_CHK_STATUS_RET(StartUdfProcess(load_model_descs, msg_file_paths),
                    "Pre process udf model and startup udf procedure failed.");
  GE_TIMESTAMP_EVENT_END(StartUdfProcess, "starting udf and loading models in deploying");
  GEEVENT("[Load][Model] success.");
  return SUCCESS;
}

Status UdfExecutorClient::DistributeAndSerializeModelDescs(
    deployer::ExecutorRequest_BatchLoadModelMessage load_model_desc,
    std::vector<deployer::ExecutorRequest_BatchLoadModelMessage> &load_model_descs,
    std::vector<std::string> &msg_file_paths) {
  GE_CHK_BOOL_RET_STATUS((load_model_desc.models_size() > 0), FAILED, "No model in BatchLoadModelMessage.");
  GE_CHK_STATUS_RET(DistributeModel(load_model_desc, load_model_descs), "Failed to distribute model.");
  const auto &context = GetContext();
  msg_file_paths.reserve(load_model_descs.size());
  for (const auto &model_desc : load_model_descs) {
    GE_CHK_BOOL_RET_STATUS((model_desc.models_size() != 0), FAILED, "No model in BatchLoadModelMessage.");
    if (load_model_desc.models(0).saved_model_file_path().empty()) {
      GELOGE(FAILED, "Udf saved model file path shold not be empty which maybe result of udf cache is old version.");
      return FAILED;
    }
    std::string load_model_message_path;
    GE_CHK_STATUS_RET(SerializeLoadModelMessageToFile(model_desc, context.base_dir, load_model_message_path),
                      "Failed to serialize load model message to file.");
    msg_file_paths.emplace_back(std::move(load_model_message_path));
  }
  return SUCCESS;
}

void UdfExecutorClient::ShutdownByRelatedDeviceIds(const std::set<int32_t> &device_ids) {
  std::set<int32_t> abnormal_related_pids;
  {
    std::unique_lock<std::mutex> guard(related_mutex_);
    for (const auto &device_id : device_ids) {
      const auto &relead_pids = npu_device_id_related_pids_[device_id];
      (void)abnormal_related_pids.insert(relead_pids.begin(), relead_pids.end());
      GELOGI("Get related abnormal device id:%d.", device_id);
    }
  }
  for (const auto &pid : abnormal_related_pids) {
    GELOGI("Notify shutdown pid:%d result of io related device is abnormal.", pid);
    NotifySubprocessShutdown(pid);
  }

  std::lock_guard<std::mutex> pids_lock(mutex_);
  for (auto &model_pids : model_id_to_pids_) {
    auto iter = std::remove_if(model_pids.second.begin(), model_pids.second.end(),
        [&abnormal_related_pids](pid_t pid) { return abnormal_related_pids.count(pid) != 0UL; });
    model_pids.second.erase(iter, model_pids.second.end());
  }
}

Status UdfExecutorClient::ClearModelRunningData(uint32_t model_id, int32_t type, const std::set<int32_t> &device_ids) {
  GELOGI("Begin to send control message for model id %u.", model_id);
  ShutdownByRelatedDeviceIds(device_ids);

  std::map<pid_t, std::shared_ptr<ExecutorMessageClient>> model_pid_and_message_clients;
  GE_CHK_STATUS_RET(GetModelMessageClients(model_id, model_pid_and_message_clients),
                    "Failed to get message clients for model %u", model_id);

  deployer::ExecutorRequest executor_request;
  auto control_message = executor_request.mutable_clear_model_message();
  GE_CHECK_NOTNULL(control_message);
  control_message->set_clear_msg_type(static_cast<int32_t>(type));
  control_message->set_model_id(model_id);

  const uint32_t parallel_num = (model_pid_and_message_clients.size() > kMaxThreadNumToSendMsg)
                                    ? kMaxThreadNumToSendMsg
                                    : model_pid_and_message_clients.size();
  ThreadPool pool("ge_udf_clr_", parallel_num, false);
  std::vector<std::future<Status>> fut_rets;
  // send control message
  for (const auto &pid_and_client : model_pid_and_message_clients) {
    pid_t pid = pid_and_client.first;
    auto client = pid_and_client.second;
    auto fut = pool.commit([pid, client, &executor_request, model_id, type, this]() -> Status {
      deployer::ExecutorResponse executor_response;
      GE_CHK_STATUS_RET(client->SendRequest(executor_request, executor_response, kRspWaitTimeInSec),
                        "[Send][ControlMessage] failed to executor, pid = %d", pid);
      if (executor_response.error_code() != SUCCESS) {
        GELOGE(FAILED, "[Control][Message] get from pid = %d is failed, error_message = %s",
               pid, executor_response.error_message().c_str());
        return FAILED;
      }
      return SUCCESS;
    });
    fut_rets.emplace_back(std::move(fut));
  }
  auto ret = SUCCESS;
  for (auto &fut : fut_rets) {
    if (fut.get() != SUCCESS) {
      GELOGE(FAILED, "Send or get response failed.");
      ret = FAILED;
    }
  }
  GELOGI("Get response from all udf ret = %d.", ret);
  return ret;
}

Status UdfExecutorClient::GetModelMessageClients(
    uint32_t root_model_id, std::map<pid_t, std::shared_ptr<ExecutorMessageClient>> &pid_and_message_clients) {
  std::unique_lock<std::mutex> guard(mutex_);
  const auto pid_iter = model_id_to_pids_.find(root_model_id);
  if (pid_iter == model_id_to_pids_.cend()) {
    GELOGE(FAILED, "Can not find udf execute process for model %u.", root_model_id);
    return FAILED;
  }
  for (const auto &pid : pid_iter->second) {
    const auto handle_iter = pid_to_message_client_.find(pid);
    if (handle_iter == pid_to_message_client_.cend()) {
      GELOGE(FAILED, "Cant find executor handle for pid %d", pid);
      return FAILED;
    }
    pid_and_message_clients[pid] = handle_iter->second;
  }
  return SUCCESS;
}

Status UdfExecutorClient::DataFlowExceptionNotify(const deployer::DataFlowExceptionNotifyRequest &req_body) {
  uint32_t root_model_id = req_body.root_model_id();
  std::map<pid_t, std::shared_ptr<ExecutorMessageClient>> notify_pid_and_message_clients;
  GE_CHK_STATUS_RET(GetModelMessageClients(root_model_id, notify_pid_and_message_clients),
                    "Failed to get message clients for model %u", root_model_id);
  deployer::ExecutorRequest executor_request;
  executor_request.set_type(deployer::ExecutorRequestType::kExecutorExceptionNotify);
  auto exception_notify_req_body = executor_request.mutable_exception_notify_request();
  GE_CHECK_NOTNULL(exception_notify_req_body);
  *exception_notify_req_body = req_body;

  const uint32_t parallel_num = (notify_pid_and_message_clients.size() > kMaxThreadNumToSendMsg)
                                    ? kMaxThreadNumToSendMsg
                                    : notify_pid_and_message_clients.size();
  std::vector<std::future<Status>> fut_rets;
  fut_rets.reserve(notify_pid_and_message_clients.size());
  ThreadPool pool("ge_udf_ntf_", parallel_num, false);
  for (const auto &pid_and_client : notify_pid_and_message_clients) {
    pid_t pid = pid_and_client.first;
    auto client = pid_and_client.second;
    auto fut = pool.commit([pid, client, &executor_request, &req_body, this]() -> Status {
      deployer::ExecutorResponse executor_response;
      GE_CHK_STATUS_RET(client->SendRequest(executor_request, executor_response),
                        "[Send][DataFlowExceptionNotify] failed to executor, pid = %d", pid);
      if (executor_response.error_code() != SUCCESS) {
        GELOGE(FAILED, "[Send][DataFlowExceptionNotify] failed, request = %s, error_code=%u, error_message = %s",
               executor_request.DebugString().c_str(), executor_response.error_code(),
               executor_response.error_message().c_str());
        return FAILED;
      }
      GELOGI("[Send][DataFlowExceptionNotify] to executor end, device_id = %d, pid = %d, trans_id = %lu, type=%u",
             GetDeviceId(), pid, req_body.exception_notify().trans_id(), req_body.exception_notify().type());
      return SUCCESS;
    });
    fut_rets.emplace_back(std::move(fut));
  }
  auto ret = SUCCESS;
  for (auto &fut : fut_rets) {
    auto task_ret = fut.get();
    if (task_ret != SUCCESS) {
      GELOGE(FAILED, "Send or get response failed.");
      ret = task_ret;
    }
  }
  return ret;
}

Status UdfExecutorClient::UnloadModel(uint32_t model_id) {
  std::vector<pid_t> pids;
  {
    std::unique_lock<std::mutex> guard(mutex_);
    const auto iter = model_id_to_pids_.find(model_id);
    if (iter != model_id_to_pids_.end()) {
      pids = iter->second;
      (void)model_id_to_pids_.erase(iter);
    }
  }
  // can not merge to ShutdownSubprocess loop, as it will be shutdown one by one.
  for (const auto &pid : pids) {
    NotifySubprocessShutdown(pid);
    const auto iter_handle = pid_to_message_client_.find(pid);
    if (iter_handle != pid_to_message_client_.cend()) {
      iter_handle->second->Stop();
    }
  }

  uint32_t wait_time_in_sec = kUdfExecutorShutdownWaitTimeInSec;
  for (const auto &pid : pids) {
    if (ShutdownSubprocess(pid, wait_time_in_sec) != SUCCESS) {
      wait_time_in_sec = 0U;
    }
    (void)pid_to_message_client_.erase(pid);
  }
  return SUCCESS;
}

ProcStatus UdfExecutorClient::GetSubProcStat() {
  std::unique_lock<std::mutex> guard(stat_mutex_);
  return PostProcSubProcessStatus(sub_proc_stat_flag_);
}

ProcStatus UdfExecutorClient::GetSubProcStatStartByPm() {
  ProcStatus stat = ProcStatus::NORMAL;
  std::vector<pid_t> pids;
  {
    std::lock_guard<std::mutex> pids_lock(mutex_);
    for (const auto &model_pids : model_id_to_pids_) {
      pids.insert(pids.end(), model_pids.second.begin(), model_pids.second.end());
    }
  }
  if (pids.empty()) {
    return stat;
  }

  std::map<pid_t, ProcStatus> stats;
  if (TsdClient::GetInstance().GetProcStatus(GetDeviceId(), pids, stats, PNE_ID_UDF) != SUCCESS) {
    ++get_proc_status_failed_times_;
    GELOGE(FAILED, "Failed to get udf process status, device_id = %d, pid = %s, failed times=%u.", GetDeviceId(),
           ToString(pids).c_str(), get_proc_status_failed_times_.load());
    constexpr uint32_t kMaxAllowGetFailedTimes = 3U;
    if (get_proc_status_failed_times_.load() < kMaxAllowGetFailedTimes) {
      return ProcStatus::EXITED;
    }
    for (auto pid : pids) {
      stats[pid] = ProcStatus::EXITED;
    }
  } else {
    get_proc_status_failed_times_.store(0U);
  }
  return PostProcSubProcessStatus(stats);
}

ProcStatus UdfExecutorClient::PostProcSubProcessStatus(const std::map<pid_t, ProcStatus> &stats) {
  std::set<pid_t> exited_pids;
  ProcStatus status = ProcStatus::NORMAL;
  for (const auto &pid_stat : stats) {
    // maybe not only one pid exit.
    if (pid_stat.second != ProcStatus::NORMAL) {
      (void)exited_pids.insert(pid_stat.first);
      if (pid_stat.second == ProcStatus::EXITED) {
        UpdateModelInsNameByPid(pid_stat.first);
        status = ProcStatus::EXITED;
        GELOGE(FAILED, "udf process[%d] exited, device_id = %d, model_instance_name=%s", pid_stat.first, GetDeviceId(),
               GetPidModelInstanceName(pid_stat.first).c_str());
      }
    }
  }
  if (exited_pids.empty()) {
    return status;
  }
  std::lock_guard<std::mutex> pids_lock(mutex_);
  for (auto &model_pids : model_id_to_pids_) {
    auto iter = std::remove_if(model_pids.second.begin(), model_pids.second.end(),
        [&exited_pids](pid_t pid) { return exited_pids.count(pid) != 0UL; });
    model_pids.second.erase(iter, model_pids.second.end());
  }
  return status;
}

void UdfExecutorClient::UpdateModelInsNameByPid(pid_t pid) {
  uint32_t root_model_id = UINT32_MAX;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    const auto iter = pid_to_model_id_.find(pid);
    if (iter == pid_to_model_id_.cend()) {
      return;
    }
    root_model_id = iter->second;
  }
  std::lock_guard<std::mutex> lk(pid_to_model_mutex_);
  const auto &pid_to_model_instances_name = pid_to_model_instances_name_.find(pid);
  if (pid_to_model_instances_name != pid_to_model_instances_name_.end()) {
    GELOGI("AbnormalStatus, update model ins name by pid[%d]", pid);
    std::lock_guard<std::mutex> abnormal_lk(abnormal_model_mutex_);
    abnormal_model_instances_name_[root_model_id].insert(abnormal_model_instances_name_[root_model_id].end(),
                                                         pid_to_model_instances_name->second.begin(),
                                                         pid_to_model_instances_name->second.end());
  }
}

void UdfExecutorClient::GetAbnormalModelInsName(std::map<uint32_t,
    std::vector<std::string>> &abnormal_model_instances_name) {
  std::lock_guard<std::mutex> lk(abnormal_model_mutex_);
  for (const auto &iter : abnormal_model_instances_name_) {
    abnormal_model_instances_name[iter.first].insert(abnormal_model_instances_name[iter.first].end(),
        iter.second.begin(), iter.second.end());
  }
}

Status UdfExecutorClient::GenLoadModelFile(const std::string &model_path, const std::string &base_dir,
                                           const std::string &load_model_message, std::string &file_path) {
  std::string dir;
  auto pos = model_path.rfind('/');
  if (pos != std::string::npos) {
    dir = model_path.substr(0, pos);
  }
  // exclude udf_resource dir if existed
  if (dir.find(kUdfResourceSubDir) != std::string::npos) {
    pos = dir.rfind('/');
    dir = dir.substr(0, pos);
  }
  dir += "/";
  const std::string full_dir = base_dir + dir;
  file_path = dir + "load_model_message_" + std::to_string(load_model_message_id_++);
  const mmMode_t kAccess = static_cast<mmMode_t>(M_UMASK_USRREAD | M_UMASK_USRWRITE | M_UMASK_GRPREAD);
  const std::string full_file_path = base_dir + file_path;
  const int32_t fd = mmOpen2(full_file_path.c_str(), (M_WRONLY | M_CREAT | O_TRUNC), kAccess);
  GE_CHK_BOOL_RET_STATUS((fd >= 0), FAILED, "Failed to open file, path = %s", full_file_path.c_str());
  ScopeGuard file_guard([fd]() { (void)mmClose(fd); });

  graphStatus write_status = WriteBinToFile(fd, load_model_message.c_str(), load_model_message.size());
  if (write_status != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "Write data to file: %s failed, write_status=%u.", full_file_path.c_str(), write_status);
    return FAILED;
  }
  {
    std::unique_lock<std::mutex> guard(mutex_);
    load_model_message_paths_.emplace_back(full_file_path);
  }
  return SUCCESS;
}

Status UdfExecutorClient::SerializeLoadModelMessageToFile(
    const deployer::ExecutorRequest_BatchLoadModelMessage &load_model_desc, const std::string &base_dir,
    std::string &file_path) {
  auto &model = load_model_desc.models(0);
  const std::string &model_path = model.model_path();
  std::string load_model_message;
  GE_CHK_BOOL_RET_STATUS(load_model_desc.SerializeToString(&load_model_message), FAILED,
                         "ExecutorRequest_BatchLoadModelMessage serialize to string failed.");
  GE_CHK_STATUS_RET(GenLoadModelFile(model_path, base_dir, load_model_message, file_path),
                    "Failed to gen serialize file path from model path: %s", model_path.c_str());
  return SUCCESS;
}

Status UdfExecutorClient::PreprocessUdfTarPackage(const std::set<std::string> &local_udf_saved_path,
                                                  const std::string &local_udf_load_path) {
  if (local_udf_load_path.empty()) {
    return SUCCESS;
  }
  std::string model_dir;
  const auto pos = local_udf_load_path.rfind(kUdfResourceSubDir);
  if (pos != std::string::npos) {
    // user define udf model path include udf_resource dir
    model_dir = local_udf_load_path.substr(0, pos);
  } else {
    // builtin udf is a file path xxx/xxx/xxx/xxx.om exclude udf_resource dir
    model_dir = local_udf_load_path.substr(0, local_udf_load_path.rfind("/") + 1);
  }
  GE_CHK_STATUS_RET(ProcessUtils::IsValidPath(model_dir), "The path[%s] is not valid.", model_dir.c_str());
  GEEVENT("Number [%zu] of local udfs will untar to file path[%s].", local_udf_saved_path.size(), model_dir.c_str());
    GE_ASSERT_TRUE((ge::CreateDir(model_dir) == EOK), "Create direct failed, path: %s.", model_dir.c_str());
  if (local_udf_saved_path.empty()) {
    GELOGI("There are no local udf need to untar package.");
    return SUCCESS;
  }
  std::string source_tar_list;
  for (const auto &saved_file : local_udf_saved_path) {
    std::regex dir_pattern(R"([A-Za-z0-9./+\-_]+)");
    std::smatch match_result;
    GE_CHK_BOOL_RET_STATUS(std::regex_match(saved_file, match_result, dir_pattern), PARAM_INVALID,
                           "Invalid source tar file path: %s", saved_file.c_str());
    source_tar_list += saved_file + " ";
  }
  source_tar_list = "(" + source_tar_list.substr(0, source_tar_list.length() - 1UL) + ")";
  GELOGD("Get local udf load path is [%s]", local_udf_load_path.c_str());

  std::string untar_cmd = R"(
tar_list=)";
  untar_cmd.append(source_tar_list);

  std::string tar_num_var = R"(
tar_num=)";
  untar_cmd.append(tar_num_var);
  untar_cmd.append(std::to_string(local_udf_saved_path.size()));

  std::string max_size_var = R"(
max_size=)";
  untar_cmd.append(max_size_var);
  untar_cmd.append(std::to_string(kUdfReleasePkgUnpackLimitSize));

  std::string target_path_var = R"(
target_path=")";
  untar_cmd.append(target_path_var);
  untar_cmd.append(model_dir);

  untar_cmd.append(R"("
function untar() {
  tar_file=$1
  dst_path=$2
  size=`tar -tvf $tar_file | awk '{sum += $3} END {print sum}'`
  if [ $size -le 0 ]; then
    return 100
  elif [ $size -gt $max_size ]; then
    return 101
  else
    tar -zxf $tar_file -C $dst_path > /dev/null 2>&1
    if [ $? -ne 0 ]; then
      return 103
    else
      return 0
    fi
  fi
}
pid_idx=0
mkdir -p $target_path
for ((i=0; i<"$tar_num"; i++))
do
  untar ${tar_list[i]} $target_path &
  pids[pid_idx]=$!
  ((pid_idx++))
done
for ((i=0; i<pid_idx; i++))
do
  wait ${pids[$i]}
  ret=$?
  if [ $ret -ne 0 ];then
    exit $ret
  fi
done
chmod -R 750 "$target_path"
)");

  GE_CHK_STATUS_RET(ProcessUtils::System(untar_cmd), "Failed to execute cmd[%s].", untar_cmd.c_str());
  return SUCCESS;
}

Status UdfExecutorClient::ForkChildProcess(const deployer::ExecutorRequest_LoadModelRequest &model_req,
                                           const std::string &file_path, const std::string &group_name,
                                           const SubProcessParams &params, pid_t &child_pid) {
  int32_t phy_device_id = model_req.phy_device_id();
  const auto &context = GetContext();
  const std::string om_dir = params.is_builtin ? "" : context.base_dir + model_req.model_path() + "_dir";
  SubprocessManager::SubprocessConfig config{};
  config.death_signal = SIGKILL;
  config.args = {kProcessName};
  config.kv_args = {{"--load_path", file_path},
                    {"--group_name", group_name},
                    {"--device_id", std::to_string(GetDeviceId())},
                    {"--phy_device_id", std::to_string(phy_device_id)},
                    {"--req_queue_id", std::to_string(params.requeset_queue_id)},
                    {"--rsp_queue_id", std::to_string(params.response_queue_id)},
                    {"--base_dir", context.base_dir}};
  if (!params.npu_sched_model.empty()) {
    config.kv_args.emplace("--npu_sched", params.npu_sched_model);
  }
  const auto process_cfg = GetDevMaintenanceCfg();
  if (process_cfg != nullptr) {
    std::map<std::string, std::string> args_option;
    GE_CHK_STATUS_RET(process_cfg->DecodeConfig(config.envs, args_option), "Decode config failed.");
    config.kv_args.insert(args_option.begin(), args_option.end());
  }

  config.process_type = PNE_ID_UDF;
  const char_t *ld_library_path_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_LD_LIBRARY_PATH, ld_library_path_env);
  std::string new_ld_library_path((ld_library_path_env != nullptr) ? ld_library_path_env : "");
  if (!params.is_builtin) {
    new_ld_library_path = new_ld_library_path + ":" + om_dir;
  }
  // only LD_LIBRARY_PATH need set, other env can be inherited by subprocess
  config.envs.emplace("LD_LIBRARY_PATH", new_ld_library_path);
  GELOGD("LD_LIBRARY_PATH is been set to %s", new_ld_library_path.c_str());
  config.unset_envs = Configurations::GetHeterogeneousEnvs();

  GE_CHK_STATUS_RET(SubprocessManager::GetInstance().ForkSubprocess(config, child_pid), "Failed to fork %s.",
                    kProcessName.c_str());
  const auto &instance_name = model_req.model_instance_name();
  std::function<void(const ProcStatus &)> excpt_handle_callback = [this, child_pid,
                                                                   instance_name](const ProcStatus &proc_status) {
    GEEVENT("%s[%d] status is %d, model_instance_name=%s.", kProcessName.c_str(), child_pid,
            static_cast<int32_t>(proc_status), instance_name.c_str());
    std::unique_lock<std::mutex> guard(stat_mutex_);
    sub_proc_stat_flag_[child_pid] = proc_status;
  };
  SubprocessManager::GetInstance().RegExcptHandleCallback(child_pid, excpt_handle_callback);
  return SUCCESS;
}

Status UdfExecutorClient::GrantQueuesForUdf(const deployer::ExecutorRequest_BatchLoadModelMessage &load_model_desc,
                                            const pid_t pid, int32_t process_device_type) {
  for (const auto &model : load_model_desc.models()) {
    GE_CHK_STATUS_RET(GrantQueuesForProcess(pid, process_device_type, model.model_queues_attrs()),
                      "Failed to grant queues for model[%u], pid[%d].", model.model_id(), pid);
    for (const auto &invoke_model : model.invoked_model_queues_attrs()) {
      GE_CHK_STATUS_RET(GrantQueuesForProcess(pid, process_device_type, invoke_model.second),
                        "Failed to grant queues for model[%u], pid[%d].", model.model_id(), pid);
    }
    RecordPidWithNpuDeviceId(pid, model.model_queues_attrs());
  }
  return SUCCESS;
}

void UdfExecutorClient::RecordPidWithNpuDeviceId(int32_t queue_owner_pid,
    deployer::ExecutorRequest_ModelQueuesAttrs model_queues_attrs) {
  std::unique_lock<std::mutex> guard(related_mutex_);
  for (const auto &input_queue : model_queues_attrs.input_queues_attrs()) {
    if (input_queue.device_type() == NPU) {
      (void)npu_device_id_related_pids_[input_queue.device_id()].insert(queue_owner_pid);
    }
  }
  for (const auto &output_queue : model_queues_attrs.output_queues_attrs()) {
    if (output_queue.device_type() == NPU) {
      (void)npu_device_id_related_pids_[output_queue.device_id()].insert(queue_owner_pid);
    }
  }
}

Status UdfExecutorClient::GrantDynamicSchedQueuesForUdf(
    const deployer::ExecutorRequest_BatchLoadModelMessage &load_model_desc, const pid_t pid,
    int32_t process_device_type) {
  for (const auto &model : load_model_desc.models()) {
    for (auto input_queue : model.status_queues().input_queues_attrs()) {
      GELOGI("DynamicSched Grant queues, status input queue id=%u, pid=%d", input_queue.queue_id(), pid);
    }
    for (auto output_queue : model.status_queues().output_queues_attrs()) {
      GELOGI("DynamicSched Grant queues, status output queue id=%u, pid=%d", output_queue.queue_id(), pid);
    }

    GE_CHK_STATUS_RET(GrantQueuesForProcess(pid, process_device_type, model.status_queues()),
                      "Failed to grant queues for model[%u], pid[%d].", model.model_id(), pid);
  }
  return SUCCESS;
}
}  // namespace ge
