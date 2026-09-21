/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/execfwk/udf_proxy_client.h"
#include "common/debug/log.h"
#include "common/checker.h"
#include "deploy/flowrm/tsd_client.h"
#include "dflow/base/utils/process_utils.h"
#include "dflow/inc/data_flow/model/pne_model.h"
#include "common/mem_grp/memory_group_manager.h"
#include "toolchain/prof_api.h"
#include "prof_common.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "graph_metadef/graph/utils/file_utils.h"

namespace ge {
namespace {
const std::string kProfilingExecuteOn = "1";
}  // namespace

PneExecutorClientCreatorRegistrar<UdfProxyClient> __attribute__((unused)) udf_proxy_reg(PNE_ID_UDF, true);
UdfProxyClient::UdfProxyClient(int32_t device_id) : UdfExecutorClient(device_id) {
  is_proxy_ = true;
}

Status UdfProxyClient::Initialize() {
  // 1971 helper host config set
  MsprofConfigParam param = {};
  param.deviceId = static_cast<uint32_t>(GetDeviceId());
  param.type = MsprofConfigParamType::HELPER_HOST_SERVER;
  param.value = 1;
  (void) MsprofSetConfig(0, reinterpret_cast<const char *>(&param), sizeof(param));
  return SUCCESS;
}

ProcStatus UdfProxyClient::GetSubProcStat() {
  return GetSubProcStatStartByPm();
}

Status UdfProxyClient::LoadModel(deployer::ExecutorRequest_BatchLoadModelMessage load_model_desc) {
  GEEVENT("[Load][Model] begin, model size = %d.", load_model_desc.models_size());
  std::vector<deployer::ExecutorRequest_BatchLoadModelMessage> load_model_descs;
  std::vector<std::string> msg_file_paths;
  GE_CHK_STATUS_RET(DistributeAndSerializeModelDescs(load_model_desc, load_model_descs, msg_file_paths),
                    "Distribute and serialize model desc failed.");
  GE_TIMESTAMP_START(PreLoadProcess);
  GE_CHK_STATUS_RET(PreLoadProcess(load_model_descs, msg_file_paths));
  GE_TIMESTAMP_EVENT_END(PreLoadProcess, "preloading proxy udf models during deploying");
  GE_TIMESTAMP_START(LoadProcess);
  const auto device_id = GetDeviceId();
  const auto &group_name = MemoryGroupManager::GetInstance().GetRemoteMemGroupName(device_id);
  GE_CHK_STATUS_RET(TsdClient::GetInstance().CheckSupportBuiltinUdfLaunch(device_id, is_support_builtin_udf_launch_),
                    "Failed to check tsd support builtin udf launch, device_id=%d.", device_id);
  for (size_t i = 0U; i < load_model_descs.size(); ++i) {
    const auto &model_desc = load_model_descs[i];
    const auto &msg_file_path = msg_file_paths[i];
    GE_CHK_STATUS_RET(LoadProcess(model_desc, msg_file_path, group_name),
                      "Failed to load model.");
  }
  GE_TIMESTAMP_EVENT_END(LoadProcess, "starting poxy udf and loading models during deploying");
  GEEVENT("[Load][Model] success.");
  return SUCCESS;
}

void UdfProxyClient::NotifySubprocessShutdown(pid_t pid) const {
  (void)pid;
  // do nothing
}

Status UdfProxyClient::ShutdownSubprocess(pid_t pid, uint32_t wait_time_in_sec) const {
  (void)wait_time_in_sec;
  return TsdClient::GetInstance().ShutdownSubprocess(GetDeviceId(), pid, PNE_ID_UDF);
}

Status UdfProxyClient::PreLoadProcess(
    const std::vector<deployer::ExecutorRequest_BatchLoadModelMessage> &load_model_descs,
    const std::vector<std::string> &load_model_msg_paths) const {
  const auto &context = GetContext();
  bool is_tsd_support_unpack = false;
  GE_CHK_STATUS_RET(TsdClient::GetInstance().CheckSupportInnerPackUnpack(context.device_id, is_tsd_support_unpack),
                    "Failed to check tsd support inner pack unpack, device_id=%d.", context.device_id);

  GE_ASSERT_TRUE(!load_model_msg_paths.empty(), "Load model message should not be empty.");
  auto model_path = load_model_msg_paths[0].substr(0, load_model_msg_paths[0].rfind('/') + 1);
  const auto &base_path = context.base_dir;
  auto tar_path = base_path + model_path;
  GE_CHK_STATUS_RET(ProcessUtils::IsValidPath(tar_path), "The path[%s] is not valid.", tar_path.c_str());

  // if tsd support unpack, udf client no need unpack
  std::vector<std::string> saved_model_paths;
  std::vector<std::string> local_builtin_model_paths;
  std::string link_cmd;
  for (size_t i = 0UL; i < load_model_descs.size(); ++i) {
    const auto &model_file_path = base_path + load_model_descs[i].models(0).model_path();
    const auto &saved_model_path = load_model_descs[i].models(0).saved_model_file_path();
    if (!load_model_descs[i].models(0).is_builtin_udf()) {
      const std::string linked_model_name = "Ascend_udf_submodel_" + std::to_string(GetDeviceId()) +
                                            "_" + std::to_string(i) + ".tar.gz";
      link_cmd += "ln -sf " + saved_model_path + " " + tar_path + linked_model_name + ";";
      saved_model_paths.emplace_back(model_path + linked_model_name);
      GELOGD("Linked model model in [%s].", (tar_path + linked_model_name).c_str());
    } else {
      local_builtin_model_paths.emplace_back(load_model_descs[i].models(0).model_path());
      GELOGD("Saved builtin model in [%s]", model_path.c_str());
    }
  }
  auto tar_name = std::to_string(TsdClient::GetLoadFileCount()) + "_Ascend-flow_model.tar.gz";
  std::string tar_cmd = link_cmd + (is_tsd_support_unpack ? "tar cf " : "tar czf ") + tar_path + tar_name;

  std::string path_list_str;

  GE_CHK_STATUS_RET(GenerateCmdStrSupportUnpack(load_model_msg_paths, saved_model_paths,
      local_builtin_model_paths, path_list_str), "Generate cmd str for tsd support unpack failed.");

  tar_cmd += path_list_str;
  GE_TIMESTAMP_START(TarCmdExecute);
  GE_CHK_STATUS_RET(ProcessUtils::System(tar_cmd), "Failed to tar om files, cmd = %s.", tar_cmd.c_str());
  GELOGI("Process tar[%s] to dir[%s] success.", tar_name.c_str(), tar_path.c_str());
  GE_TIMESTAMP_EVENT_END(TarCmdExecute, "Link and tar cmd");

  GE_TIMESTAMP_START(TransTar);
  GE_CHK_STATUS_RET(TsdClient::GetInstance().LoadFile(context.device_id, tar_path, tar_name),
                    "Failed to transfer om files, device_id = %d, tar path = %s, tar name = %s.", context.device_id,
                    tar_path.c_str(), tar_name.c_str());
  GELOGI("Send tar[%s] to device[%d] success.", tar_name.c_str(), context.device_id);
  GE_TIMESTAMP_EVENT_END(TransTar, "Tsd transport tar");
  return SUCCESS;
}

Status UdfProxyClient::GenerateCmdStrSupportUnpack(const std::vector<std::string> &load_model_msg_paths,
    const std::vector<std::string> &saved_model_paths, const std::vector<std::string> &local_builtin_model_paths,
    std::string &path_list_str) const {
  const auto &base_path = GetContext().base_dir;
  path_list_str = " -C " + base_path + " ";
  for (const auto &msg_path : load_model_msg_paths) {
    path_list_str += msg_path + " ";
  }
  for (const auto &builtin_path : local_builtin_model_paths) {
    path_list_str += builtin_path + " ";
  }
  path_list_str += " -h ";
  for (const auto &tar_path : saved_model_paths) {
    path_list_str += tar_path + " ";
  }
  return SUCCESS;
}

Status UdfProxyClient::LoadProcess(const deployer::ExecutorRequest_BatchLoadModelMessage &load_model_desc,
                                   const std::string &msg_file_path,
                                   const std::string &group_name) {
  GEEVENT("Fork udf process to load model on executor start.");
  pid_t child_pid = -1;
  GE_CHK_STATUS_RET(ForkAndInit(load_model_desc, group_name, msg_file_path, child_pid),
                    "Failed to fork child process.");
  const int32_t device_id = GetDeviceId();
  GEEVENT("Fork udf process to load model on executor success. "
          "model_type = %s, pid = %d, graph_id = %u, deployer pid = %d, device_id = %d.",
          PNE_ID_UDF.c_str(), child_pid, load_model_desc.graph_id(), GetDeployerPid(), device_id);
  return SUCCESS;
}

Status UdfProxyClient::ForkChildProcess(const deployer::ExecutorRequest_LoadModelRequest &model_req,
                                        const std::string &file_path, const std::string &group_name,
                                        const SubProcessParams &params, pid_t &child_pid) {
  const std::string &model_path = model_req.model_path();
  int32_t phy_device_id = model_req.phy_device_id();
  SubprocessManager::SubprocessConfig config{};
  std::string ld_library_path;
  if (params.is_builtin && is_support_builtin_udf_launch_) {
    config.process_type = "BUILTIN_UDF";
  } else {
    config.process_type = PNE_ID_UDF;
    ld_library_path = model_path + "_dir";
    GELOGD("LD_LIBRARY_PATH is been set to %s", ld_library_path.c_str());
  }
  config.args = {"udf_executor"};
  config.kv_args = {
      {"--load_path", file_path}, {"--group_name", group_name},
      {"--phy_device_id", std::to_string(phy_device_id)},
      {"--req_queue_id", std::to_string(params.requeset_queue_id)},
      {"--rsp_queue_id", std::to_string(params.response_queue_id)}};

  config.kv_args.emplace("PROFILING_RENEW_PATH", "0");

  const auto process_cfg = GetDevMaintenanceCfg();
  if (process_cfg != nullptr) {
    GE_CHK_STATUS_RET(process_cfg->DecodeConfig(config.envs, config.kv_args), "Decode config failed.");
  }

  auto step = config.kv_args.find(OPTION_EXEC_DUMP_STEP);
  if (step != config.kv_args.cend()) {
    std::string dump_step = step->second;
    replace(dump_step.begin(), dump_step.end(), '|', '_');
    step->second = dump_step;
    GELOGI("ForkChildProcess dump step[%s] trans success.", dump_step.c_str());
  }

  GE_CHK_STATUS_RET(TsdClient::GetInstance().ForkSubprocess(GetDeviceId(), config, ld_library_path, child_pid),
                    "Failed to fork udf_executor on device[%d], model_path=%s, is_builtin=%d.", GetDeviceId(),
                    model_path.c_str(), static_cast<int32_t>(params.is_builtin));
  GELOGI("for child process success, model_path=%s, is_builtin=%d", model_path.c_str(),
         static_cast<int32_t>(params.is_builtin));
  return SUCCESS;
}

Status UdfProxyClient::DoMemGrpAddProc(const std::string &group_name, const pid_t child_pid) {
  const int32_t device_id = GetDeviceId();
  GE_CHK_STATUS_RET(
      MemoryGroupManager::GetInstance().RemoteMemGrpAddProc(device_id, group_name, child_pid, false, true),
      "Failed to add group[%s] to process[%d], device id[%d].", group_name.c_str(), child_pid, device_id);
  return SUCCESS;
}
}  // namespace ge
