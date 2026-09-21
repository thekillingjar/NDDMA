/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/abnormal_status_handler/abnormal_status_handler.h"
#include <algorithm>
#include <future>
#include <sys/inotify.h>
#include "mmpa/mmpa_api.h"
#include "framework/common/util.h"
#include "deploy/deployer/heterogeneous_model_deployer.h"
#include "deploy/resource/resource_manager.h"
#include "securec.h"
#include "common/compile_profiling/ge_call_wrapper.h"

namespace ge {
namespace {
constexpr const char_t *const kRedeployFileName = "redeploy";
constexpr const char_t *const kRedeployDoneFileName = "redeploy.done";
constexpr const char_t *const kRedeployErrorFileName = "redeploy.error";
constexpr size_t kMaxPathLen = 1024UL;
constexpr int32_t kMaxReadMonitorFileStrLen = 1024;
constexpr int32_t kFileMonitorInterval = 500;
constexpr int32_t kCheckReployFileInterval = 50;
constexpr int32_t kCheckReployFileTimes = 10;
constexpr int32_t kNotSupportDefault = 0;
constexpr int32_t kNotSupportDynamicSched = 1;
constexpr int32_t kNotSupportRedeploy = 2;
}  // namespace

void AbnormalStatusHandler::FindOldDevice(DeployPlan::DeviceStateList &device_state_list,
    NodeConfig node_new, NodeConfig node_old) const {
  for (auto& iter_device_old : node_old.device_list) { // 新devices里面找老的device，找不到说明老的device损坏
    bool find_old_in_new = false;
    for (auto& iter_device_new : node_new.device_list) {
      if (iter_device_old.device_type == iter_device_new.device_type &&
          iter_device_old.device_id == iter_device_new.device_id) {
        DeployPlan::DeviceInfo device_info =
            DeployPlan::DeviceInfo(iter_device_old.device_type, node_old.node_id, iter_device_old.device_id);
        device_state_list.emplace(device_info, true);
        GELOGI("AbnormalStatusMonitor, device is normal, node_id=%d, device_id=%u, device_type=%u",
          device_info.GetNodeId(), device_info.GetDeviceId(), device_info.GetType());
        find_old_in_new = true;
        break;
      }
    }
    if (!find_old_in_new) {
      DeployPlan::DeviceInfo device_info =
          DeployPlan::DeviceInfo(iter_device_old.device_type, node_old.node_id, iter_device_old.device_id);
      device_state_list.emplace(device_info, false);
      GEEVENT("AbnormalStatusMonitor, device is abnormal, node_id=%d, device_id=%u, device_type=%u",
          device_info.GetNodeId(), device_info.GetDeviceId(), device_info.GetType());
    }
  }
}

Status AbnormalStatusHandler::FindAbnormalDeviceOnServer(DeployPlan::DeviceStateList &device_state_list,
    DeployerConfig information_new, DeployerConfig information_old) const {
  // 把所有node_config信息汇总到node_config_list
  std::vector<NodeConfig> node_config_list_new;
  node_config_list_new.assign(information_new.remote_node_config_list.begin(),
      information_new.remote_node_config_list.end());
  node_config_list_new.insert(node_config_list_new.begin(), information_new.node_config);
  std::vector<NodeConfig> node_config_list_old;
  node_config_list_old.assign(information_old.remote_node_config_list.begin(),
      information_old.remote_node_config_list.end());
  node_config_list_old.insert(node_config_list_old.begin(), information_old.node_config);

  for (auto& iter_old : node_config_list_old) {
    bool is_node_abnormal = true;
    for (auto& iter_new : node_config_list_new) {
      if (iter_old.ipaddr.compare(iter_new.ipaddr) != 0) {
        continue;
      }
      is_node_abnormal = false;
      FindOldDevice(device_state_list, iter_new, iter_old);
    }
    if (is_node_abnormal) {
      for (auto& iter_device_old : iter_old.device_list) {
        DeployPlan::DeviceInfo device_info =
            DeployPlan::DeviceInfo(iter_device_old.device_type, iter_old.node_id, iter_device_old.device_id);
        device_state_list.emplace(device_info, false);
        GEEVENT("AbnormalStatusMonitor, node is abnormal, node_id=%d, device_id=%u, device_type=%u",
            device_info.GetNodeId(), device_info.GetDeviceId(), device_info.GetType());
      }
    }
  }
  return SUCCESS;
}

Status AbnormalStatusHandler::FindAbnormalDevice(DeployPlan::DeviceStateList &device_state_list,
    DeployerConfig information_new, DeployerConfig information_old) const {
  auto& node_config_old = information_old.node_config;
  auto& node_config_new = information_new.node_config;
  if (node_config_old.ipaddr.compare(node_config_new.ipaddr) == 0) { // host对比
    DeployPlan::DeviceInfo device_info =
        DeployPlan::DeviceInfo(CPU, node_config_old.node_id, 0);
    device_state_list.emplace(device_info, true);
    GELOGI("AbnormalStatusMonitor, device is normal, node_id=%d, device_id=%u, device_type=%u, ipaddr=%s",
        device_info.GetNodeId(), device_info.GetDeviceId(), device_info.GetType(), node_config_old.ipaddr.c_str());
  } else {
    DeployPlan::DeviceInfo device_info = DeployPlan::DeviceInfo(CPU, node_config_old.node_id, 0);
    device_state_list.emplace(device_info, false);
    GEEVENT("AbnormalStatusMonitor, device is abnormal, node_id=%d, device_id=%u, device_type=%u, ipaddr=%s",
        device_info.GetNodeId(), device_info.GetDeviceId(), device_info.GetType(),
        information_old.node_config.ipaddr.c_str());
  }
  for (auto& iter_old : information_old.remote_node_config_list) { // device对比
    bool find_old_in_new = false;
    for (auto& iter_new : information_new.remote_node_config_list) {
      // 新devices里面找老的device(51上device信息存在node中)，找不到说明老的device损坏
      if (iter_old.ipaddr.compare(iter_new.ipaddr) == 0) {
        DeployPlan::DeviceInfo device_info0 =
            DeployPlan::DeviceInfo(NPU, iter_old.node_id, 0);
        DeployPlan::DeviceInfo device_info1 =
            DeployPlan::DeviceInfo(NPU, iter_old.node_id, 1);
        device_state_list.emplace(device_info0, true);
        device_state_list.emplace(device_info1, true);
        GELOGI("AbnormalStatusMonitor, device is normal ipaddr=%s, node_id=%d, device_id=%u, device_type=%u,"
            " node_id=%d, device_id=%u, device_type=%u", iter_old.ipaddr.c_str(),
            device_info0.GetNodeId(), device_info0.GetDeviceId(), device_info0.GetType(),
            device_info1.GetNodeId(), device_info1.GetDeviceId(), device_info1.GetType());
        find_old_in_new = true;
      }
    }
    if (!find_old_in_new) { // 51上面一个device损坏两个卡都异常
      DeployPlan::DeviceInfo device_info0 =
          DeployPlan::DeviceInfo(NPU, iter_old.node_id, 0);
      DeployPlan::DeviceInfo device_info1 =
          DeployPlan::DeviceInfo(NPU, iter_old.node_id, 1);
      device_state_list.emplace(device_info0, false);
      device_state_list.emplace(device_info1, false);
      GELOGI("AbnormalStatusMonitor, device is abnormal ipaddr=%s, node_id=%d, device_id=%u, device_type=%u,"
          " node_id=%d, device_id=%u, device_type=%u", iter_old.ipaddr.c_str(),
          device_info0.GetNodeId(), device_info0.GetDeviceId(), device_info0.GetType(),
          device_info1.GetNodeId(), device_info1.GetDeviceId(), device_info1.GetType());
    }
  }
  return SUCCESS;
}

Status AbnormalStatusHandler::ParseDeviceStateList(const std::string &file_path,
    DeployPlan::DeviceStateList &device_state_list) {
  // 解析异常设备信息
  auto information_old = Configurations::GetInstance().GetHostInformation();
  GELOGI("AbnormalStatusMonitor, show old node info");
  ShowNodeInfo(information_old);
  DeployerConfig information_new;
  GE_CHK_STATUS_RET_NOLOG(ConfigParser::ParseServerInfo(file_path, information_new));
  GELOGI("AbnormalStatusMonitor, show new node info on server");
  ShowNodeInfo(information_new);
  GE_CHK_STATUS_RET(FindAbnormalDeviceOnServer(device_state_list, information_new, information_old),
    "AbnormalStatusMonitor, failed to do FindAbnormalDevice if is on server");
  return SUCCESS;
}

bool AbnormalStatusHandler::IsHeartbeatNormal() const {
  auto &deploy_context = DeployContext::LocalContext();
  std::lock_guard<std::mutex> lk(deploy_context.GetAbnormalHeartbeatInfoMu());
  return deploy_context.GetAbnormalSubmodelInstanceName().empty() && deploy_context.GetAbnormalNodeConfig().empty()
      && deploy_context.GetAbnormalDeviceInfo().empty();
}

void AbnormalStatusHandler::ParseAbnormalNodeConfig(DeployPlan::DeviceStateList &device_state_list) const {
  auto &deploy_context = DeployContext::LocalContext();
  for (auto &iter : deploy_context.GetAbnormalNodeConfig()) {
    for (auto& iter_device : iter.first.device_list) {
      DeployPlan::DeviceInfo device_info =
          DeployPlan::DeviceInfo(iter_device.device_type, iter.first.node_id, iter_device.device_id);
      device_state_list.emplace(device_info, false);
      GELOGI("AbnormalStatusMonitor, node abnormal(process OnServer), node_id=%d, device_id=%d, device_type=%d",
          device_info.GetNodeId(), device_info.GetDeviceId(), device_info.GetType());
    }
  }
}

void AbnormalStatusHandler::ParseAbnormalDeviceInfo(DeployPlan::DeviceStateList &device_state_list) const {
  auto &deploy_context = DeployContext::LocalContext();
  for (auto &iter : deploy_context.GetAbnormalDeviceInfo()) {
    GELOGI("AbnormalStatusMonitor, ParseAbnormalDeviceInfo: node_id=%d, device_id=%d, device_type=%d",
        iter.first.GetNodeId(), iter.first.GetDeviceId(), iter.first.GetType());
    device_state_list.emplace(iter.first, false);
  }
}

void AbnormalStatusHandler::ParseAbnormalModelInstances(bool &is_new_abnormal) {
  auto &deploy_context = DeployContext::LocalContext();
  for (auto &iter : deploy_context.GetAbnormalSubmodelInstanceName()) {
    for (auto &submodel_instance_name : iter.second) {
      auto instances_model_iter = abnormal_submodel_instances_name_[iter.first].find(submodel_instance_name.first);
      if (instances_model_iter == abnormal_submodel_instances_name_[iter.first].end()) {
        abnormal_submodel_instances_name_[iter.first][submodel_instance_name.first] = true;
        GELOGI("AbnormalStatusMonitor, ParseAbnormalModelInstances: root model id=%u,"
            " submodel[%s] is on abnormal process",
            iter.first, submodel_instance_name.first.c_str());
        is_new_abnormal = true;
      }
    }
  }
}

void AbnormalStatusHandler::ParseHeartbeatAbnormalInfo(bool &is_new_abnormal,
    DeployPlan::DeviceStateList &device_state_list) {
  GELOGI("AbnormalStatusMonitor, ParseHeartbeatAbnormalInfo start");
  auto information_old = Configurations::GetInstance().GetHostInformation();
  ShowNodeInfo(information_old);
  auto &deploy_context = DeployContext::LocalContext();
  std::lock_guard<std::mutex> lk(deploy_context.GetAbnormalHeartbeatInfoMu());
  if (!deploy_context.GetAbnormalNodeConfig().empty()) { // tsd进程异常或心跳监测失败异常
    GELOGI("AbnormalStatusMonitor, tsd process abnormal, abnormal node info size=%zu",
        deploy_context.GetAbnormalNodeConfig().size());
    ParseAbnormalNodeConfig(device_state_list);
    AbnormalDevices2ModelInstances(device_state_list, is_new_abnormal); // node异常转为model instance异常
    deploy_context.ClearAbnormalNodeConfig();
  }
  if (!deploy_context.GetAbnormalDeviceInfo().empty()) { // flowgw进程异常
    GELOGI("AbnormalStatusMonitor, flowgw process abnormal, abnormal device info size=%zu",
        deploy_context.GetAbnormalDeviceInfo().size());
    ParseAbnormalDeviceInfo(device_state_list);
    AbnormalDevices2ModelInstances(device_state_list, is_new_abnormal); // device异常转为model instance异常
    deploy_context.ClearAbnormalDeviceInfo();
  }
  if (!deploy_context.GetAbnormalSubmodelInstanceName().empty()) { // 执行进程异常
    GELOGI("AbnormalStatusMonitor, executor process abnormal, abnormal submodel instances size=%zu",
        deploy_context.GetAbnormalSubmodelInstanceName().size());
    ParseAbnormalModelInstances(is_new_abnormal);
    deploy_context.ClearAbnormalSubmodelInstanceName();
  }
  GELOGI("AbnormalStatusMonitor, ParseHeartbeatAbnormalInfo end");
}

bool AbnormalStatusHandler::IsModelMulInstace(std::map<const std::string, bool> &abnormal_submodel_instances_name,
    DeployPlan::ModelDeployInfo model_deploy_infos) const {
    for (auto model_deploy_info = model_deploy_infos.begin();
        model_deploy_info != model_deploy_infos.end(); model_deploy_info++) {
      bool is_model_mul_model_instance = false;
      for (auto& model_instance_info : model_deploy_info->second) {
        auto abnormal_submodel_instances = abnormal_submodel_instances_name.find(model_instance_info.first);
        if (abnormal_submodel_instances == abnormal_submodel_instances_name.end()) {
          GELOGI("AbnormalStatusMonitor, the model[%s]'s instance[%s] deployed on normal process",
              model_deploy_info->first.c_str(), model_instance_info.first.c_str());
          is_model_mul_model_instance = true;
        }
      }
      if (!is_model_mul_model_instance) {
        GEEVENT("AbnormalStatusMonitor, it doesn't support dynamic sched,"
            "the model[%s] only deployed on abnormal process", model_deploy_info->first.c_str());
        return false;
      }
    }
  return true;
}

bool AbnormalStatusHandler::IsSupportDynamicSchedRecover(const uint32_t &root_model_id) {
  if (!is_dynamic_sched_) {
    GELOGI("AbnormalStatusMonitor, is_dynamic_sched_ is unenable");
    return false;
  }

  // 获取root_model_id对应的submodel_instance_name 的映射
  DeployPlan::ModelDeployInfo model_deploy_info;
  {
    std::lock_guard<std::mutex> lk(mu_);
    model_deploy_info = deployed_models_[root_model_id].model_deploy_infos;
  }

  // 检查root_model_id对应的异常模型是否多实例
  if (!IsModelMulInstace(abnormal_submodel_instances_name_[root_model_id], model_deploy_info)) {
    return false;
  }
  return true;
}

Status AbnormalStatusHandler::GenerateFile(const std::string &file_path, const char_t *const file_name) const {
  auto pos = file_path.find_last_of('/');
  GE_CHK_BOOL_RET_STATUS(pos != std::string::npos, FAILED,
    "AbnormalStatusMonitor, failed to handle path[%s]", file_path.c_str());
  std::string new_file_path = file_path.substr(0, pos + 1) + file_name;
  std::ofstream file(new_file_path);
  GE_CHK_BOOL_RET_STATUS(file.is_open(), FAILED, "AbnormalStatusMonitor, failed generate path[%s]",
      new_file_path.c_str());
  file.close();
  GEEVENT("AbnormalStatusMonitor, the path[%s] has generated", new_file_path.c_str());
  return SUCCESS;
}

Status AbnormalStatusHandler::AfterHandleAbnormalInfo(const std::string &file_path, const char_t *const file_name) {
  GE_CHK_STATUS_RET(GenerateFile(file_path, file_name),
      "AbnormalStatusMonitor, failed to do GenerateFile, file_name[%s]", file_name);
  return SUCCESS;
}

Status AbnormalStatusHandler::FailedHandleAbnormal(uint32_t root_model_id) {
  GELOGI("AbnormalStatusMonitor, abnormal status doesn't support redeploy, callback exec, model_id=%u", root_model_id);
  std::lock_guard<std::mutex> lk(abnormal_status_callback_info_.mu);
  if (abnormal_status_callback_info_.callback_list[root_model_id] != nullptr) {
    GE_CHK_STATUS_RET(abnormal_status_callback_info_.callback_list[root_model_id](kCallbackFailedRedeploy,
        abnormal_submodel_instances_name_),
        "AbnormalStatusMonitor, callback exec failed, root_model_id=%u", root_model_id);
  }
  return SUCCESS;
}

void AbnormalStatusHandler::GetDeviceListDiff(const DeployPlan::DeviceStateList &device_state_list_new,
    DeployPlan::DeviceStateList &device_state_list_old, DeployPlan::DeviceStateList &device_state_list_diff) const {
  for (auto &iter_new : device_state_list_new) {
    if (iter_new.second) {
      continue;
    }
    DeployPlan::DeviceInfo device_info = DeployPlan::DeviceInfo(iter_new.first.GetType(),
        iter_new.first.GetNodeId(), iter_new.first.GetDeviceId());
    auto iter_old = device_state_list_old.find(device_info);
    if (iter_old != device_state_list_old.end() && !(iter_old->second)) {
      continue;
    } else {
      GELOGI("AbnormalStatusMonitor, GetDeviceListDiff: node_id=%d, device_id=%u, device_type=%u",
          device_info.GetNodeId(), device_info.GetDeviceId(), device_info.GetType());
      device_state_list_diff[device_info] = false;
      device_state_list_old[device_info] = false;
    }
  }
}

bool AbnormalStatusHandler::IsInDeviceList(std::set<DeployPlan::DeviceInfo> &instance_device_infos,
    DeployPlan::DeviceStateList &device_state_list_diff) const {
  for (const auto &instance_device_info : instance_device_infos) {
    DeployPlan::DeviceInfo device_info = DeployPlan::DeviceInfo(instance_device_info.GetType(),
        instance_device_info.GetNodeId(), instance_device_info.GetDeviceId());
    auto new_abnormal_device = device_state_list_diff.find(device_info);
    if (new_abnormal_device != device_state_list_diff.end() && !(new_abnormal_device->second)) {
      return true;
    }
  }
  return false;
}

bool AbnormalStatusHandler::IsInModelInstanceList(uint32_t root_model_id,
    const std::string &model_instance_name,
    RootModelId2SubmodelName &abnormal_submodel_instances_name) const {
  auto iter = abnormal_submodel_instances_name[root_model_id].find(model_instance_name);
  if (iter != abnormal_submodel_instances_name[root_model_id].end() && !(iter->second)) {
    return true;
  }
  return false;
}

void AbnormalStatusHandler::Add2ModelInstanceList(uint32_t root_model_id, const std::string &model_instance_name,
    RootModelId2SubmodelName &abnormal_submodel_instances_name) const {
  abnormal_submodel_instances_name[root_model_id][model_instance_name] = false;
}

void AbnormalStatusHandler::AbnormalDiffDevices2ModelInstances(uint32_t root_model_id,
    std::map<std::string, std::set<DeployPlan::DeviceInfo>> &model_deploy_info,
    DeployPlan::DeviceStateList &device_state_list_diff,
    bool &is_new_abnormal) {
  for (auto &model_instance_info : model_deploy_info) {
    if (!IsInDeviceList(model_instance_info.second, device_state_list_diff)) {
      GELOGI("AbnormalStatusMonitor, model instance[%s] is not on abnormal device list",
          model_instance_info.first.c_str());
      continue;
    }
    if (IsInModelInstanceList(root_model_id, model_instance_info.first, abnormal_submodel_instances_name_)) {
      GELOGI("AbnormalStatusMonitor, model instance[%s] is already in abnormal model instance list",
          model_instance_info.first.c_str());
      continue;
    } else {
      GELOGI("AbnormalStatusMonitor, model instance[%s] is add to abnormal list",
          model_instance_info.first.c_str());
      Add2ModelInstanceList(root_model_id, model_instance_info.first, abnormal_submodel_instances_name_);
      is_new_abnormal =  true;
    }
  }
}

void AbnormalStatusHandler::AbnormalDevices2ModelInstances(DeployPlan::DeviceStateList &device_state_list,
    bool &is_new_abnormal) {
  DeployPlan::DeviceStateList device_state_list_diff;
  GetDeviceListDiff(device_state_list, device_state_list_, device_state_list_diff);
  if (device_state_list_diff.empty()) {
    GELOGI("AbnormalStatusMonitor, device_state_list has no new abnormal devices");
    return;
  }
  std::lock_guard<std::mutex> lk(mu_);
  for (auto &deployed_model : deployed_models_) {
    for (auto &model_deploy_info : deployed_model.second.model_deploy_infos) {
      AbnormalDiffDevices2ModelInstances(deployed_model.first, model_deploy_info.second,
          device_state_list_diff, is_new_abnormal);
    }
  }
}

Status AbnormalStatusHandler::RedeployStart(const uint32_t &root_model_id) {
  GELOGI("AbnormalStatusMonitor, redeploy start: pre callback exec, root_model_id=%u", root_model_id);
  std::lock_guard<std::mutex> lk(abnormal_status_callback_info_.mu);
  if (abnormal_status_callback_info_.callback_list[root_model_id] != nullptr) {
    GE_CHK_STATUS_RET(abnormal_status_callback_info_.callback_list[root_model_id](kCallbackStartRedeploy,
        abnormal_submodel_instances_name_),
        "AbnormalStatusMonitor, failed to do exec callback, root_model_id=%u", root_model_id);
  }
  return SUCCESS;
}

void AbnormalStatusHandler::PreHandleAbnormalInfo() {
  {
    std::lock_guard<std::mutex> lk(abnormal_status_callback_info_.mu);
    GELOGI("AbnormalStatusMonitor, wait callback init, callback size=%zu, deployed models size=%zu",
        abnormal_status_callback_info_.callback_list.size(),
        deployed_models_.size());
  }
  while (!IsDeployingRootModel() && !IsAllCallbackInitFinished()) { // 无模型正在部署并且所有执行器把callbacks初始化后可开始重部署
    std::this_thread::sleep_for(std::chrono::milliseconds(kFileMonitorInterval)); // 等待0.5秒进行下一次检查
  }
  GELOGI("AbnormalStatusMonitor, callback init success");
  return;
}

void AbnormalStatusHandler::ShowNodeInfo(DeployerConfig &information) const {
  GELOGI("AbnormalStatusMonitor, ShowNodeInfo start");
  std::vector<NodeConfig> node_config_list;
  node_config_list.assign(information.remote_node_config_list.begin(),
      information.remote_node_config_list.end());
  node_config_list.insert(node_config_list.begin(), information.node_config);
  for (auto& iter : node_config_list) {
    GELOGI("AbnormalStatusMonitor, device info: node_id=%d", iter.node_id);
    for (auto& iter_device : iter.device_list) {
      GELOGI("AbnormalStatusMonitor, device info: device_type=%u, node_id=%d, device_id=%u",
          iter_device.device_type, iter.node_id, iter_device.device_id);
    }
  }
  GELOGI("AbnormalStatusMonitor, ShowNodeInfo end");
}

Status AbnormalStatusHandler::ParallelAbnormalStatusHandle(uint32_t check_devices_flag) {
  auto root_model_num = abnormal_submodel_instances_name_.size();
  if (root_model_num > 1) {
    ThreadPool pool("ge_dpl_rd", static_cast<uint32_t>(root_model_num), false);
    std::vector<std::future<Status>> fut_rets;
    for (const auto &it : abnormal_submodel_instances_name_) {
      const auto &root_model_id = it.first;
      auto fut = pool.commit([this, &root_model_id, &check_devices_flag]() -> Status {
        GE_CHK_STATUS_RET(RedeployProc(root_model_id, check_devices_flag),
        "AbnormalStatusMonitor, failed to do RedeployProc, root_model_id=%u", root_model_id);
        return SUCCESS;
      });
      fut_rets.emplace_back(std::move(fut));
    }
    bool has_root_model_redeploy_failed = false;
    for (auto &fut : fut_rets) {
      if (fut.get() != SUCCESS) { // fut.get()是同步操作
        has_root_model_redeploy_failed = true;
      }
    }
    if (has_root_model_redeploy_failed) {
      GELOGE(FAILED, "AbnormalStatusMonitor, there is a model redeployed failed");
      return FAILED;
    }
    return SUCCESS;
  }
  for (const auto &it : abnormal_submodel_instances_name_) {
    const auto &root_model_id = it.first;
    GE_CHK_STATUS_RET(RedeployProc(root_model_id, check_devices_flag),
      "AbnormalStatusMonitor, failed to do RedeployProc, root_model_id=%u", root_model_id);
  }
  return SUCCESS;
}

Status AbnormalStatusHandler::FileMonitorProc(const std::string &file_path) {
  GELOGI("AbnormalStatusMonitor, FileMonitorProc start, file_path=%s", file_path.c_str());
  DeployPlan::DeviceStateList device_state_list;
  GE_CHK_STATUS_RET(ParseDeviceStateList(file_path, device_state_list),
      "AbnormalStatusMonitor, failed to do ParseDeviceStateList");
  bool is_new_abnormal = false;
  AbnormalDevices2ModelInstances(device_state_list, is_new_abnormal); // device异常转换为model_instance异常(即执行进程异常)
  if (is_new_abnormal) {
    PreHandleAbnormalInfo();
    auto check_devices_flag = CheckAbnormalDevices(device_state_list);
    if (check_devices_flag == kNotSupportRedeploy) {
      // host异常，无法恢复业务, 写redeploy.error文件
      GE_CHK_STATUS_RET(AfterHandleAbnormalInfo(file_path, kRedeployErrorFileName),
        "AbnormalStatusMonitor, failed to do AfterHandleAbnormalInfo, kRedeployErrorFileName");
      GELOGE(FAILED, "AbnormalStatusMonitor, it(cause by abnormal device) can't recover by redeploying");
    }
    if (ParallelAbnormalStatusHandle(check_devices_flag) == SUCCESS) {
      GE_CHK_STATUS_RET(AfterHandleAbnormalInfo(file_path, kRedeployDoneFileName),
        "AbnormalStatusMonitor, failed to do AfterHandleAbnormalInfo, kRedeployDoneFileName");
      return SUCCESS;
    } else {
      GE_CHK_STATUS_RET(AfterHandleAbnormalInfo(file_path, kRedeployErrorFileName),
        "AbnormalStatusMonitor, failed to do AfterHandleAbnormalInfo, kRedeployErrorFileName");
      return FAILED;
    }
  }
  GELOGI("AbnormalStatusMonitor, FileMonitorProc end, no new abnormal model instance"); // 无新异常或新异常上无模型实例部署
  return SUCCESS;
}

Status AbnormalStatusHandler::WaitReployFileGenerate(const std::string &file_path) {
  uint32_t times = 0;
  while (times++ < kCheckReployFileTimes) {
    GELOGI("AbnormalStatusMonitor, Wait for redeploy file generated, times=%u", times);
    if (IsReployFileGeneratedThenRemove(file_path)) {
      return SUCCESS;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kCheckReployFileInterval)); // 0.05秒检查一次，检查10次
  }
  return FAILED;
}

bool AbnormalStatusHandler::IsReployFileGeneratedThenRemove(const std::string &file_path) const {
  auto pos = file_path.find_last_of('/');
  if (pos == std::string::npos) {
    GELOGE(FAILED, "AbnormalStatusMonitor, file path[%s] is illegal", file_path.c_str());
    return false;
  }
  std::string redeploy_file_path = file_path.substr(0, pos + 1) + kRedeployFileName;
  std::ifstream ifs(redeploy_file_path);
  if (!ifs.is_open()) {
    GELOGI("AbnormalStatusMonitor, The path[%s] hasn't generated", redeploy_file_path.c_str());
    return false;
  }
  ifs.close();
  (void) std::remove(redeploy_file_path.c_str());
  GELOGI("AbnormalStatusMonitor, The path[%s] has generated, now remove it", redeploy_file_path.c_str());
  return true;
}

Status AbnormalStatusHandler::GetFilePath(std::string &config_dir,
                                      const char_t *const path_env) const {
  char_t file_path[kMaxPathLen]{};
  const int32_t ret = mmGetEnv(path_env, file_path, kMaxPathLen);
  if (ret == EN_OK) {
    const std::string real_path = RealPath(file_path);
    if (!real_path.empty()) {
      config_dir = file_path;
      GELOGI("AbnormalStatusMonitor, Get file_path[%s] success from env", file_path);
      return SUCCESS;
    }
  }
  return FAILED;
}

Status AbnormalStatusHandler::GetMonitorFilePath(std::string &file_path) {
  GELOGI("AbnormalStatusMonitor, try get resource config path, on server");
  GE_CHK_STATUS_RET(Configurations::GetResourceConfigPath(file_path), "Failed to get resource file");
  GELOGI("AbnormalStatusMonitor, get resource config path[%s] success", file_path.c_str());
  return SUCCESS;
}

void AbnormalStatusHandler::MonitorFileAndHeartbeatProc(const std::string &file_path,
    const int32_t &fd) {
  char_t buf[kMaxReadMonitorFileStrLen];
  ssize_t len = 0;
  GELOGI("AbnormalStatusMonitor, monitoring path[%s]", file_path.c_str());
  while (file_monitor_flag_.load()) {
    len = read(fd, buf, sizeof(buf));
    if (len <= 0) { // 优先文件监测无异常，再心跳监测
      if (IsHeartbeatNormal()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(kFileMonitorInterval)); // 等待0.5秒进行下一次检查
      } else if (HeartbeatMonitorProc() != SUCCESS) {
        GELOGE(FAILED, "AbnormalStatusMonitor, failed to do HeartbeatMonitorProc");
        break;
      }
      continue;
    }
    GELOGI("AbnormalStatusMonitor, The path[%s] is different, parser the different type", file_path.c_str());
    char_t *event_buf = buf;
    bool resource_config_modify = false;
    while (event_buf < buf + len) {
      struct inotify_event *event = static_cast<struct inotify_event*>(static_cast<void*>(event_buf));
      if ((event->mask & IN_MODIFY) != 0) {
        resource_config_modify = true;
        break;
      }
      event_buf += sizeof(struct inotify_event*) + event->len;
    }
    if (resource_config_modify) {
      GELOGI("AbnormalStatusMonitor, The path[%s] has modify", file_path.c_str());
      if (WaitReployFileGenerate(file_path) == SUCCESS &&
          FileMonitorProc(file_path) != SUCCESS) {
        GELOGE(FAILED, "AbnormalStatusMonitor, failed to FileMonitorProc");
        break;
      }
    }
  }
}

void AbnormalStatusHandler::AbnormalStatusMonitorRun() {
  GEEVENT("AbnormalStatusMonitor, abnormal status monitor thread start");
  std::string file_path;
  if (GetMonitorFilePath(file_path) != SUCCESS) {
    GELOGW("AbnormalStatusMonitor, get no monitor file path");
    return;
  }
  int32_t fd = inotify_init1(IN_NONBLOCK); // 设置非阻塞模式，inotify_init没有非阻塞模式
  if (fd < 0) {
    GELOGW("AbnormalStatusMonitor, inotify1 init get fd = %d", fd);
    return;
  }
  int32_t wd = inotify_add_watch(fd, file_path.c_str(), IN_MODIFY);
  if (wd < 0) {
    GELOGW("AbnormalStatusMonitor, inotify and watch get wd = %d", wd);
    close(fd);
    return;
  }
  MonitorFileAndHeartbeatProc(file_path, fd);
  inotify_rm_watch(fd, wd);
  close(fd);
  return;
}

void AbnormalStatusHandler::Initialize() {
  GELOGI("AbnormalStatusMonitor, Initialize");
  file_monitor_flag_.store(true);
  run_context_ = GetThreadLocalContext();
  file_monitor_thread_ = std::thread([this]() {
    SET_THREAD_NAME(pthread_self(), "ge_dpl_fmon");
    GetThreadLocalContext() = run_context_;
    AbnormalStatusMonitorRun();
  });
  return;
}

void AbnormalStatusHandler::Finalize() {
  GELOGI("AbnormalStatusMonitor, Finalize");
  file_monitor_flag_.store(false);
  if (file_monitor_thread_.joinable()) {
    file_monitor_thread_.join();
  }
  deployed_models_.clear();
  {
    std::lock_guard<std::mutex> lk(abnormal_status_callback_info_.mu);
    abnormal_status_callback_info_.callback_list.clear();
  }
  device_state_list_.clear();
  abnormal_submodel_instances_name_.clear();
  return;
}

Status AbnormalStatusHandler::DynamicSchedRecoverProc(uint32_t root_model_id) {
  std::vector<DeployPlan::DeviceInfo> device_infos;
  for (auto &iter : device_state_list_) {
    if (!iter.second) {
      device_infos.push_back(iter.first);
    }
  }
  GE_CHK_STATUS_RET(ClearModelExceptionData(root_model_id, device_infos), // root_model_id用于数据清理，device_infos用于删除路由
      "AbnormalStatusMonitor, failed to do ClearModelExceptionData");
  GELOGI("AbnormalStatusMonitor, dynamic sched callback exec, model_id=%u", root_model_id);
  std::lock_guard<std::mutex> lk(abnormal_status_callback_info_.mu);
  if (abnormal_status_callback_info_.callback_list[root_model_id] != nullptr) {
    GE_CHK_STATUS_RET(abnormal_status_callback_info_.callback_list[root_model_id](kCallbackDynamicSched,
        abnormal_submodel_instances_name_),
        "AbnormalStatusMonitor, callback exec failed, root_model_id=%u", root_model_id);
  }
  return SUCCESS;
}

uint32_t AbnormalStatusHandler::CheckAbnormalDevices(DeployPlan::DeviceStateList &device_state_list) const {
  // 异常的device_info为主server所在的host(数据清理需要通过host），或者主server所在的device0异常（调度请求接收在device0）则不支持动态调度
  for (auto iter = device_state_list.begin(); iter != device_state_list.end(); iter++) {
    if (!iter->second && (iter->first.GetNodeId() == 0)) { // 主server异常
      if (iter->first.GetType() == static_cast<int32_t>(CPU)) { // host异常
        GELOGE(FAILED, "AbnormalStatusMonitor, it doesn't support redeploy, host is abnormal");
        return kNotSupportRedeploy;
      }
      if (is_dynamic_sched_ && (iter->first.GetDeviceId() == 0)) { // 动态调度场景device0异常
        GELOGI("AbnormalStatusMonitor, it doesn't support dynamic sched redeploy, device0 is abnormal");
        return kNotSupportDynamicSched;
      }
    }
  }
  return kNotSupportDefault;
}

Status AbnormalStatusHandler::RedeployProc(uint32_t root_model_id, uint32_t check_devices_flag) {
  GELOGI("AbnormalStatusMonitor, RedeployProc start, root_model_id=%u", root_model_id);
  GE_DISMISSABLE_GUARD(guard, ([this, &root_model_id]() {
    if (FailedHandleAbnormal(root_model_id) != SUCCESS) { // 无法恢复业务, 做相应处理:ModelIO调用返回失败
      GELOGE(FAILED, "AbnormalStatusMonitor, failed to do FailedHandleAbnormal");
    }
  }));
  GE_CHK_STATUS_RET(RedeployStart(root_model_id),
      "AbnormalStatusMonitor, failed to do RedeployStart");
  if (check_devices_flag != kNotSupportRedeploy && IsSupportDynamicSchedRecover(root_model_id)) { // 1、动态调度降级服务
    GE_CHK_STATUS_RET(DynamicSchedRecoverProc(root_model_id),
        "AbnormalStatusMonitor, failed to do dynamic sched recover proc");
    GELOGI("AbnormalStatusMonitor, redeploy success");
    GE_DISMISS_GUARD(guard);
    return SUCCESS;
  }
  // 2、重部署恢复业务
  // 3、无法恢复业务, 做相应处理:ModelIO调用返回失败:见FailedHandleAbnormal
  GELOGE(FAILED, "AbnormalStatusMonitor, root_model_id=%u can't recover by redeploying",
      root_model_id);
  return FAILED;
}

Status AbnormalStatusHandler::HeartbeatMonitorProc() {
  GELOGI("AbnormalStatusMonitor, HeartbeatMonitorProc start");
  bool is_new_abnormal = false;
  DeployPlan::DeviceStateList device_state_list;
  ParseHeartbeatAbnormalInfo(is_new_abnormal, device_state_list);
  if (is_new_abnormal) {
    PreHandleAbnormalInfo();
    auto check_devices_flag = CheckAbnormalDevices(device_state_list);
    if (check_devices_flag == kNotSupportRedeploy) {
      GELOGE(FAILED, "AbnormalStatusMonitor, it(cause by abnormal process) can't recover by redeploying");
    }
    GE_CHK_STATUS_RET(ParallelAbnormalStatusHandle(check_devices_flag),
      "AbnormalStatusMonitor, failed to do ParallelAbnormalStatusHandle");
    return SUCCESS;
  }
  GELOGI("AbnormalStatusMonitor, HeartbeatMonitorProc end, no new abnormal process");
  return SUCCESS;
}

AbnormalStatusHandler::AbnormalStatusHandler(std::mutex &mu)
  : mu_(mu) {
  GELOGI("AbnormalStatusHandler, start.");
}

AbnormalStatusHandler::~AbnormalStatusHandler() {
  if (file_monitor_flag_) {
    GELOGW("AbnormalStatusHandler, file_monitor_thread_ is not stopped");
    Finalize();
  }
}

void AbnormalStatusHandler::IncDeployingRootModelNum() {
  deploying_root_model_cnt_++;
}

void AbnormalStatusHandler::DecreaseDeployingRootModelNum() {
  deploying_root_model_cnt_--;
}

bool AbnormalStatusHandler::IsDeployingRootModel() {
  return deploying_root_model_cnt_.load() != 0U;
}

bool AbnormalStatusHandler::IsAllCallbackInitFinished() {
  std::lock_guard<std::mutex> lk(abnormal_status_callback_info_.mu);
  return abnormal_status_callback_info_.callback_list.size() ==
      deployed_models_.size();
}

void AbnormalStatusHandler::DelCallback(const uint32_t root_model_id) {
  std::lock_guard<std::mutex> lk(abnormal_status_callback_info_.mu);
  abnormal_status_callback_info_.callback_list.erase(root_model_id);
}

DeployPlan::AbnormalStatusCallbackInfo* AbnormalStatusHandler::GetAbnormalStatusCallbackInfo() {
  return &abnormal_status_callback_info_;
}

void AbnormalStatusHandler::SetDynamicSchedFlag(bool flag) {
  is_dynamic_sched_ = flag;
}

void AbnormalStatusHandler::AddDeployedModelInfo(uint32_t model_id,
    const DeployPlan::ModelDeployInfo &model_deploy_infos,
    const std::set<int32_t> &deployed_remote_nodes) {
  deployed_models_[model_id].model_id = model_id;
  deployed_models_[model_id].model_deploy_infos = model_deploy_infos;
  deployed_models_[model_id].deployed_remote_nodes = deployed_remote_nodes;
}

void AbnormalStatusHandler::DelDeployedModelInfo(uint32_t model_id) {
  deployed_models_.erase(model_id);
}

Status AbnormalStatusHandler::ParallelClearData(const std::pair<uint32_t, std::set<uint32_t>> &need_clear_root_models,
                                                const std::vector<DeployPlan::DeviceInfo> &device_infos,
                                                const int32_t type) const {
  auto clear_num = need_clear_root_models.second.size();
  if (clear_num > 1U) {
    ThreadPool pool("ge_dpl_rdc", static_cast<uint32_t>(clear_num), false);
    std::vector<std::future<Status>> fut_rets;
    for (const auto &node_id : need_clear_root_models.second) {
      const auto &model_id = need_clear_root_models.first;
      auto fut = pool.commit([&node_id, &model_id, &device_infos, type]() -> Status {
        GE_CHK_STATUS_RET_NOLOG(HeterogeneousModelDeployer::ClearNodelExceptionData(node_id, model_id,
                                device_infos, type));
        return SUCCESS;
      });
      fut_rets.emplace_back(std::move(fut));
    }
    for (auto &fut : fut_rets) {
      GE_CHK_STATUS_RET_NOLOG(fut.get());
    }
  } else {
    for (const auto &node_id : need_clear_root_models.second) {
      GE_CHK_STATUS_RET_NOLOG(HeterogeneousModelDeployer::ClearNodelExceptionData(node_id,
                              need_clear_root_models.first, device_infos, type));
    }
  }
  return SUCCESS;
}

// 如果node整个坏了或者host坏了就不要执行清理动作了
Status AbnormalStatusHandler::ClearModelExceptionData(uint32_t root_model_id,
    const std::vector<DeployPlan::DeviceInfo> &device_infos) {
  for (const auto &device_info : device_infos) {
    GELOGI("AbnormalStatusMonitor, Exception root_model_id=%u, device info: device = %s.",
      root_model_id, device_info.GetDesc().c_str());
  }
  std::pair<uint32_t, std::set<uint32_t>> need_clear_root_models; // model_id, node_ids
  need_clear_root_models.first = root_model_id;
  const int32_t local_node_id = ResourceManager::GetInstance().GetLocalNodeId(); // local node也要一同清理
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto &model_info_iter = deployed_models_[root_model_id];
    auto clear_nodes = model_info_iter.deployed_remote_nodes;
    clear_nodes.emplace(local_node_id);  // local也要算上
    for (auto clear_node : clear_nodes) {
      need_clear_root_models.second.emplace(clear_node);
    }
  }
  for (auto &node_id : need_clear_root_models.second) {
    GELOGI("AbnormalStatusMonitor, Exception model info: root_model_id = %u, node_id = %u.",
        need_clear_root_models.first, node_id);
  }

  GE_CHK_STATUS_RET_NOLOG(ParallelClearData(need_clear_root_models, device_infos, EXCEPTION_HANDLE_STOP));
  GE_CHK_STATUS_RET_NOLOG(ParallelClearData(need_clear_root_models, device_infos, EXCEPTION_HANDLE_CLEAR));
  return SUCCESS;
}
}  // namespace ge
