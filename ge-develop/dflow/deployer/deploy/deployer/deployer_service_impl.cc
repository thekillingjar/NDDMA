/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/deployer/deployer_service_impl.h"

#include <string>
#include <vector>
#include "deploy/deployer/heterogeneous_model_deployer.h"
#include "ge/ge_api_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "proto/deployer.pb.h"
#include "securec.h"

namespace ge {
namespace {
constexpr int32_t kIpIndex = 1;
constexpr int32_t kPortIndex = 2;
}  // namespace
class ServiceRegister {
 public:
  ServiceRegister(const deployer::DeployerRequestType type, const DeployerServiceImpl::ProcessFunc &fn) {
    DeployerServiceImpl::GetInstance().RegisterReqProcessor(type, fn);
  }

  ~ServiceRegister() = default;
};

#define REGISTER_REQUEST_PROCESSOR(type, fn) \
REGISTER_REQUEST_PROCESSOR_UNIQ_DEPLOYER(__COUNTER__, type, fn)

#define REGISTER_REQUEST_PROCESSOR_UNIQ_DEPLOYER(ctr, type, fn) \
REGISTER_REQUEST_PROCESSOR_UNIQ(ctr, type, fn)

#define REGISTER_REQUEST_PROCESSOR_UNIQ(ctr, type, fn) \
static ::ge::ServiceRegister register_request_processor##ctr \
__attribute__((unused)) = \
::ge::ServiceRegister(type, fn)

REGISTER_REQUEST_PROCESSOR(deployer::kUpdateDeployPlan, DeployerServiceImpl::UpdateDeployPlanProcess);
REGISTER_REQUEST_PROCESSOR(deployer::kAddFlowRoutePlan, DeployerServiceImpl::FlowRoutePlanProcess);
REGISTER_REQUEST_PROCESSOR(deployer::kDownloadConf, DeployerServiceImpl::DownloadDevMaintenanceCfgProcess);
REGISTER_REQUEST_PROCESSOR(deployer::kLoadModel, DeployerServiceImpl::LoadModelProcess);
REGISTER_REQUEST_PROCESSOR(deployer::kUnloadModel, DeployerServiceImpl::UnloadModelProcess);
REGISTER_REQUEST_PROCESSOR(deployer::kDownloadVarManager, DeployerServiceImpl::MultiVarManagerInfoProcess);
REGISTER_REQUEST_PROCESSOR(deployer::kDownloadSharedContent, DeployerServiceImpl::SharedContentProcess);
REGISTER_REQUEST_PROCESSOR(deployer::kTransferFile, DeployerServiceImpl::TransferFileProcess);
REGISTER_REQUEST_PROCESSOR(deployer::kInitProcessResource, DeployerServiceImpl::InitProcessResourceProcess);
REGISTER_REQUEST_PROCESSOR(deployer::kHeartbeat, DeployerServiceImpl::HeartbeatProcess);
REGISTER_REQUEST_PROCESSOR(deployer::kDatagwSchedInfo, DeployerServiceImpl::DataGwSchedInfo);
REGISTER_REQUEST_PROCESSOR(deployer::kClearModelData, DeployerServiceImpl::ClearModelRunningData);
REGISTER_REQUEST_PROCESSOR(deployer::kDataFlowExceptionNotify, DeployerServiceImpl::DataFlowExceptionNotifyProcess);
REGISTER_REQUEST_PROCESSOR(deployer::kUpdateProfilingInfo, DeployerServiceImpl::UpdateProfilingInfoProcess);

DeployerServiceImpl &DeployerServiceImpl::GetInstance() {
  static DeployerServiceImpl instance;
  return instance;
}

void DeployerServiceImpl::RegisterReqProcessor(deployer::DeployerRequestType type,
                                               const DeployerServiceImpl::ProcessFunc &fn) {
  process_fns_.emplace(type, fn);
}

void DeployerServiceImpl::DownloadDevMaintenanceCfgProcess(DeployContext &context,
                                                           const deployer::DeployerRequest &request,
                                                           deployer::DeployerResponse &response) {
  context.DownloadDevMaintenanceCfg(request, response);
}

void DeployerServiceImpl::LoadModelProcess(DeployContext &context,
                                           const deployer::DeployerRequest &request,
                                           deployer::DeployerResponse &response) {
  const auto &req_body = request.load_model_request();
  uint32_t root_model_id = req_body.root_model_id();
  DeployState *deploy_state = nullptr;
  if (context.flow_model_receiver_.GetDeployState(root_model_id, deploy_state) != SUCCESS) {
    GELOGE(FAILED, "[Load][Model] model deploy plan not found, id = %u", root_model_id);
    response.set_error_code(FAILED);
    response.set_error_message("deploy plan not found");
    return;
  }

  if (HeterogeneousModelDeployer::DeployModel(context, *deploy_state) != SUCCESS) {
    context.flow_model_receiver_.DestroyDeployState(root_model_id);
    GELOGE(FAILED, "[Load][Model] failed, id = %u", root_model_id);
    response.set_error_code(FAILED);
    response.set_error_message("Failed to load model");
    return;
  }
  GEEVENT("Deploy model ended, model_id = %u, result = %u", root_model_id, response.error_code());
  context.flow_model_receiver_.DestroyDeployState(root_model_id);
}

void DeployerServiceImpl::UnloadModelProcess(DeployContext &context,
                                             const deployer::DeployerRequest &request,
                                             deployer::DeployerResponse &response) {
  const auto &req_body = request.unload_model_request();
  auto ret = context.UnloadSubmodels(req_body.model_id());
  if (ret != SUCCESS) {
    response.set_error_code(FAILED);
    response.set_error_message("Failed to unload model");
  }
}

void DeployerServiceImpl::MultiVarManagerInfoProcess(DeployContext &context,
                                                     const deployer::DeployerRequest &request,
                                                     deployer::DeployerResponse &response) {
  auto ret = context.ProcessMultiVarManager(request.multi_var_manager_request());
  if (ret != SUCCESS) {
    response.set_error_code(FAILED);
    response.set_error_message("Failed to initialize VarManager");
  }
}

void DeployerServiceImpl::SharedContentProcess(DeployContext &context,
                                               const deployer::DeployerRequest &request,
                                               deployer::DeployerResponse &response) {
  auto ret = context.ProcessSharedContent(request.shared_content_desc_request(), response);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Failed to process shared content.");
    response.set_error_code(FAILED);
    response.set_error_message("Failed to process shared content.");
  }
}

void DeployerServiceImpl::InitProcessResourceProcess(DeployContext &context, const deployer::DeployerRequest &request,
                                                     deployer::DeployerResponse &response) {
  context.InitProcessResource(request.init_process_resource_request(), response);
}

void DeployerServiceImpl::HeartbeatProcess(DeployContext &context,
                                           const deployer::DeployerRequest &request,
                                           deployer::DeployerResponse &response) {
  context.ProcessHeartbeat(request, response);
}

Status DeployerServiceImpl::Process(DeployContext &context,
                                    const deployer::DeployerRequest &request,
                                    deployer::DeployerResponse &response) {
  auto type = request.type();
  GELOGI("[Process][Request] start, client = %s, type = %s",
         context.GetName().c_str(), deployer::DeployerRequestType_Name(type).c_str());
  auto it = process_fns_.find(type);
  if (it == process_fns_.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Find api type[%d]  failed.", type);
    GELOGE(FAILED, "[Find][Api] Find api type[%d] failed.", type);
    response.set_error_code(FAILED);
    response.set_error_message("Api does not exist");
    return SUCCESS;
  }

  auto process_fn = it->second;
  if (process_fn == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Function pointer does not exist.");
    GELOGE(FAILED, "[Find][Function] Find pointer does not exist.");
    response.set_error_code(FAILED);
    response.set_error_message("Find pointer does not exist.");
    return SUCCESS;
  }

  process_fn(context, request, response);
  GE_CHK_STATUS(response.error_code(), "[Process][Request] failed, client = %s, type = %s, ret = %u",
                context.GetName().c_str(), deployer::DeployerRequestType_Name(type).c_str(), response.error_code());
  GELOGI("[Process][Request] end, client = %s, type = %s, ret = %u",
         context.GetName().c_str(), deployer::DeployerRequestType_Name(type).c_str(), response.error_code());
  return SUCCESS;
}

void DeployerServiceImpl::FlowRoutePlanProcess(DeployContext &context,
                                               const deployer::DeployerRequest &request,
                                               deployer::DeployerResponse &response) {
  if (!request.has_add_flow_route_plan_request()) {
    GELOGE(PARAM_INVALID, "request body is null");
    response.set_error_code(PARAM_INVALID);
    response.set_error_message("invalid request");
    return;
  }

  const auto &request_body = request.add_flow_route_plan_request();
  auto ret = context.flow_model_receiver_.AddFlowRoutePlan(request_body);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Failed to update flow route plan.");
    response.set_error_code(FAILED);
    response.set_error_message("Failed to update flow route plan.");
    return;
  }
  GEEVENT("AddFlowRoutePlan success, node_id = %d, model_id = %u", request_body.node_id(),
          request_body.root_model_id());
}

void DeployerServiceImpl::UpdateDeployPlanProcess(DeployContext &context,
                                                  const deployer::DeployerRequest &request,
                                                  deployer::DeployerResponse &response) {
  if (!request.has_update_deploy_plan_request()) {
    GELOGE(PARAM_INVALID, "request body is null");
    response.set_error_code(PARAM_INVALID);
    response.set_error_message("invalid request");
    return;
  }

  const auto &request_body = request.update_deploy_plan_request();
  auto ret = context.flow_model_receiver_.UpdateDeployPlan(request_body);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Failed to update deploy plan.");
    response.set_error_code(FAILED);
    response.set_error_message("Failed to update deploy plan.");
    return;
  }
  GEEVENT("UpdateDeployPlan success, device_id = %d, model_id = %u", request_body.device_id(),
          request_body.root_model_id());
}

void DeployerServiceImpl::TransferFileProcess(DeployContext &context,
                                              const deployer::DeployerRequest &request,
                                              deployer::DeployerResponse &response) {
  if (!request.has_transfer_file_request()) {
    GELOGE(PARAM_INVALID, "request body is null");
    response.set_error_code(PARAM_INVALID);
    response.set_error_message("invalid request");
    return;
  }

  const auto &req_body = request.transfer_file_request();
  auto ret = context.flow_model_receiver_.AppendToFile(context.GetBaseDir() + req_body.path(),
                                                       req_body.content().data(),
                                                       req_body.content().size(),
                                                       req_body.eof());
  if (ret != SUCCESS) {
    const std::string err = strerror(errno);
    std::string err_msg = "Failed to write to file[" + context.GetBaseDir() + req_body.path() + "]" +
                          (errno == 0 ? "." : ", err_msg[" + err + "].");
    response.set_error_code(FAILED);
    response.set_error_message(err_msg);
  }
}

void DeployerServiceImpl::DataGwSchedInfo(DeployContext &context,
                                          const deployer::DeployerRequest &request,
                                          deployer::DeployerResponse &response) {
  if (!request.has_datagw_sched_info()) {
    GELOGE(PARAM_INVALID, "DynamicSched request body is null");
    response.set_error_code(PARAM_INVALID);
    response.set_error_message("invalid request");
    return;
  }

  const auto &request_body = request.datagw_sched_info();
  auto ret = context.flow_model_receiver_.AddDataGwSchedInfos(request_body);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "DynamicSched Failed to update flow route plan.");
    response.set_error_code(FAILED);
    response.set_error_message("Failed to update flow route plan.");
    return;
  }
  GEEVENT("DynamicSched DataGwSchedInfo success, model_id = %u", request_body.root_model_id());
}

void DeployerServiceImpl::ClearModelRunningData(DeployContext &context,
                                                const deployer::DeployerRequest &request,
                                                deployer::DeployerResponse &response) {
  const auto &req_body = request.model_data_clear();
  auto ret = context.ClearModelRunningData(req_body);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Failed to clear data.");
    response.set_error_code(FAILED);
    response.set_error_message("Failed to clear data");
  }
}
void DeployerServiceImpl::DataFlowExceptionNotifyProcess(DeployContext &context,
                                                         const deployer::DeployerRequest &request,
                                                         deployer::DeployerResponse &response) {
  const auto &req_body = request.exception_notify_request();
  auto ret = context.DataFlowExceptionNotifyProcess(req_body);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Failed to process data flow exception notify data.");
    response.set_error_code(FAILED);
    response.set_error_message("Failed to process data flow exception notify data");
  }
}

void DeployerServiceImpl::UpdateProfilingInfoProcess(DeployContext &context,
                                                     const deployer::DeployerRequest &request,
                                                     deployer::DeployerResponse &response) {
  const auto &req_body = request.prof_info();
  auto ret = context.UpdateProfilingInfoProcess(req_body);
  if (ret != SUCCESS) {
    response.set_error_code(FAILED);
    response.set_error_message("Failed to update profiling info");
  }
}
}  // namespace ge
