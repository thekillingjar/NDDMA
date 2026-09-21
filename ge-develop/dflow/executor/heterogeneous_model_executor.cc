/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "heterogeneous_model_executor.h"
#include <numeric>
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/math_util.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/ge_global_options.h"
#include "framework/common/types.h"
#include "framework/common/runtime_tensor_desc.h"
#include "common/model/ge_root_model.h"
#include "graph/ge_context.h"
#include "graph/utils/op_type_utils.h"
#include "proto/task.pb.h"
#include "data_flow_info_utils.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "dflow/inc/data_flow/model/graph_model.h"
#include "data_flow_executor_utils.h"
#include "acl/acl.h"

namespace ge {
namespace {
constexpr size_t kDataOutputIndexVal0 = 0U;
constexpr size_t kAlignmentVal64 = 64U;
constexpr int32_t kRetryInterval = 1000; // 1000 ms
constexpr int32_t kQueueNeverTimeout = -1;
constexpr int32_t kQueueDefaultTimeout = 10 * 60 * 1000; // ms
constexpr int32_t kTimeoutValuePerSecond = 1000;
// limit max cache dynamic schedule route trans id num.
constexpr size_t kCacheMaxTransIdNum = 1024;
constexpr int32_t kDequeueMbufWaitTime = 3000;
constexpr int32_t kMicrosecondToNanosecond = 1000;
constexpr int32_t kDynamicSchedDuration = 100 * kMicrosecondToNanosecond; // 100us
constexpr int32_t kAttachTimeOut = 3000;
constexpr int32_t kDequeueSize = 1U;
constexpr uint16_t kRawDataMsg = 1U;

ge::PneModelPtr GetSubmodel(const PneModelPtr &root_model, const std::string &submodel_name) {
  auto submodel = root_model->GetSubmodel(submodel_name);
  if (submodel == nullptr) {
    for (const auto &pne_model_pair : root_model->GetSubmodels()) {
      submodel = pne_model_pair.second->GetSubmodel(submodel_name);
      if (submodel != nullptr) {
        break;
      }
    }
  }
  return submodel;
}
}  // namespace

thread_local uint64_t HeterogeneousModelExecutor::duration_total_ = 0ULL;
thread_local uint64_t HeterogeneousModelExecutor::cnt_total_ = 0ULL;
thread_local uint64_t HeterogeneousModelExecutor::duration_max_ = 0ULL;
thread_local uint64_t HeterogeneousModelExecutor::duration_size_ = 0ULL;
thread_local uint64_t HeterogeneousModelExecutor::call_ = 0ULL;

HeterogeneousModelExecutor::HeterogeneousModelExecutor(const FlowModelPtr &flow_model,
                                                       const DeployResult &deploy_result)
    : flow_model_(flow_model),
      deployed_model_id_(deploy_result.model_id),
      input_queue_attrs_(deploy_result.input_queue_attrs),
      control_input_queue_attrs_(deploy_result.control_input_queue_attrs),
      output_queue_attrs_(deploy_result.output_queue_attrs),
      control_output_queue_attrs_(deploy_result.control_output_queue_attrs),
      dev_abnormal_callback_(deploy_result.dev_abnormal_callback),
      io_helper_(deploy_result.input_queue_attrs, deploy_result.broadcast_input_queue_attrs),
      exception_handler_(deploy_result.exception_notify_callback),
      replica_num_(deploy_result.replica_num),
      input_model_name_(deploy_result.input_model_name),
      status_input_queue_attrs_(deploy_result.status_output_queue_attrs),
      sched_input_queue_attrs_(deploy_result.sched_input_queue_attrs),
      sched_output_queue_attrs_(deploy_result.sched_output_queue_attrs),
      model_index_info_(deploy_result.model_index_info),
      datagw_request_bindings_(deploy_result.datagw_request_bindings),
      is_dynamic_sched_(deploy_result.is_dynamic_sched),
      is_exception_catch_(deploy_result.is_exception_catch),
      contains_n_mapping_node_(deploy_result.contains_n_mapping_node),
      abnormal_status_callback_info_(deploy_result.abnormal_status_callback_info),
      model_trimming_edges_model_instances_(deploy_result.model_trimming_edges_model_instances),
      align_attrs_(deploy_result.input_align_attrs) {
}

HeterogeneousModelExecutor::~HeterogeneousModelExecutor() {
  if (run_flag_) {
    GELOGW("Run thread is not stopped");
    (void)ModelRunStop();
  }
}

void HeterogeneousModelExecutor::UpdateAbnormalInstanceList(
    RootModelId2SubmodelName &abnormal_submodel_instances_name) {
  const std::lock_guard<std::mutex> lk(abnormal_name_mu_);
  for (auto &iter : abnormal_submodel_instances_name) {
    for (auto &iter_name : iter.second) {
      abnormal_submodel_instances_name_[iter.first].emplace(iter_name.first, iter_name.second);
      if ((!model_trimming_edges_model_instances_.empty()) && iter_name.second) {
        GELOGD("Start update abnormal instance list for trimming scenario.");
        UpdateAbnormalInstanceForTrimmingModel(iter.first, iter_name.first);
      }
    }
  }
}

void HeterogeneousModelExecutor::UpdateAbnormalInstanceForTrimmingModel(const uint32_t root_model_id,
                                                                        const std::string &abnormal_name) {
  for (const auto &trimming_models : model_trimming_edges_model_instances_) {
    if (trimming_models.count(abnormal_name) != 0UL) {
      GELOGI("There are num[%zu] trimming relation models for Abnormal model[%s].",
             trimming_models.size(), abnormal_name.c_str());
      for (const auto &trimming_model : trimming_models) {
        abnormal_submodel_instances_name_[root_model_id].emplace(trimming_model, true);
        GELOGI("Record model instance[%s] as abnormal result of trimming model relation.", trimming_model.c_str());
      }
    }
  }
}

void HeterogeneousModelExecutor::AbnormalStatusCallbackInit() {
  if (abnormal_status_callback_info_ != nullptr) {
    DeployPlan::AbnormalStatusCallback abnormal_status_clear_callback = [this](
        uint32_t operation_type,
        RootModelId2SubmodelName &abnormal_submodel_instances_names) -> Status {
      if (IsRedeployStart(operation_type)) {
        deploy_state_.store(kCallbackStartRedeploy);
        GELOGI("abnormal status callback: deploy status is kCallbackStartRedeploy");
        UpdateAbnormalInstanceList(abnormal_submodel_instances_names);
        return SUCCESS;
      } else if (IsDynamicSched(operation_type)) {
        DynamicSchedClear();
        ModelIndexInfoUpdate();
      } else if (IsRedeployFailed(operation_type)) {
        deploy_state_.store(kCallbackFailedRedeploy);
        GELOGI("abnormal status callback: deploy status is kCallbackFailedRedeploy");
        return SUCCESS;
      }
      ClearFeedData();
      ClearFetchData();
      deploy_state_.store(kCallbackRedeployDone);
      return SUCCESS;
    };
    std::lock_guard<std::mutex> lk((*abnormal_status_callback_info_).mu);
    auto root_model_id = GetDeployedModelId();
    (*abnormal_status_callback_info_).callback_list[root_model_id] = abnormal_status_clear_callback;
    GELOGI("abnormal status callback init success, model_id=%u", root_model_id);
  }
}

Status HeterogeneousModelExecutor::Initialize() {
  GE_CHECK_NOTNULL(flow_model_);
  GE_CHK_BOOL_RET_STATUS(!flow_model_->GetSubmodels().empty(), INTERNAL_ERROR, "Sub models in flow model is empty.");
  GE_CHK_STATUS_RET_NOLOG(WrapSingleModel());
  GE_CHECK_NOTNULL(model_relation_);

  std::vector<uint32_t> input_queue_ids;
  std::vector<uint32_t> control_input_queue_ids;
  std::vector<uint32_t> output_queue_ids;
  std::vector<uint32_t> control_output_queue_ids;
  for (size_t i = 0; i < input_queue_attrs_.size(); ++i) {
    input_queue_ids.emplace_back(input_queue_attrs_[i].queue_id);
    GEEVENT("[IO info], flow model input info = [index:%zu], queue info = [%s].",
            i, input_queue_attrs_[i].DebugString().c_str());
  }
  for (size_t i = 0; i < output_queue_attrs_.size(); ++i) {
    output_queue_ids.emplace_back(output_queue_attrs_[i].queue_id);
    GEEVENT("[IO info], flow model output info = [index:%zu], queue info = [%s].",
            i, output_queue_attrs_[i].DebugString().c_str());
  }

  GEEVENT("[IO info], flow model input_queues = %s, control_input_queues = %s, "
          "output_queues = %s, control_output_queues = %s",
          ToString(input_queue_ids).c_str(),
          ToString(control_input_queue_ids).c_str(),
          ToString(output_queue_ids).c_str(),
          ToString(control_output_queue_ids).c_str());

  GE_CHK_STATUS_RET(ParseInputTensorInfo(), "Failed to parse input tensor info, model name = %s",
                    flow_model_->GetModelName().c_str());
  GE_CHK_STATUS_RET(ParseOutputTensorInfo(), "Failed to parse output tensor info, model name = %s",
                    flow_model_->GetModelName().c_str());
  const auto execution_runtime = ExecutionRuntime::GetInstance();
  GE_CHECK_NOTNULL(execution_runtime);
  exchange_service_ = &execution_runtime->GetExchangeService();
  GE_CHECK_NOTNULL(exchange_service_);
  GE_CHK_STATUS_RET(io_helper_.Initialize(), "Failed to init model io helper.");

  if (is_exception_catch_ || is_dynamic_sched_) {
    GE_CHK_STATUS_RET(process_forwarding_.Initialize(status_input_queue_attrs_),
                      "Inner process message forwarding module initialize failed.");
    GE_CHK_STATUS_RET(DynamicSchedQueueInitialize(is_dynamic_sched_),
                      "DynamicSched, failed to initialize dynamic sched queue");
    if (is_exception_catch_) {
      GE_CHK_STATUS_RET(exception_handler_.Initialize(process_forwarding_), "init data flow exception handler failed");
    }
  }
  AbnormalStatusCallbackInit();
  return SUCCESS;
}

Status HeterogeneousModelExecutor::DynamicSchedQueueInitialize(const bool is_dynamic_sched) {
  for (const auto &id : status_input_queue_attrs_) {
    auto ret = rtMemQueueAttach(id.device_id, id.queue_id, kAttachTimeOut);
    GE_CHK_BOOL_RET_STATUS(ret == RT_ERROR_NONE, FAILED,
        "DynamicSched, mem queue attach failed, device_id=%d, status_input_queue_ids queue_id=%u, ret=%d",
        id.device_id, id.queue_id, ret);
  }
  if (!is_dynamic_sched) {
    return SUCCESS;
  }
  // app 请求相应的逻辑queue id和物理queue id
  for (uint32_t i = 0U; i < sched_input_queue_attrs_.size(); i++) {
    auto iter = datagw_request_bindings_.find(sched_input_queue_attrs_[i].global_logic_id);
    GE_CHK_BOOL_RET_STATUS(iter != datagw_request_bindings_.end(), FAILED,
        "DynamicSched, can't find sched app output indices=%u record.", sched_input_queue_attrs_[i].global_logic_id);
    datagw_rqt_to_rsp_[iter->second] = sched_input_queue_attrs_[i]; // 获得datagw逻辑input queue到app物理queueid映射
    GELOGI("DynamicSched, scheding request, find datagw input indices=%d to sched app output queueid=%u.",
        iter->second, sched_input_queue_attrs_[i].queue_id);
  }

  for (auto id : sched_input_queue_attrs_) {
    auto drv_ret = rtMemQueueAttach(id.device_id, id.queue_id, kAttachTimeOut);
    GE_CHK_BOOL_RET_STATUS(drv_ret == RT_ERROR_NONE, FAILED,
        "DynamicSched, mem queue attach failed, device_id=%d, sched_input_queue_attrs queue_id=%u, ret=%d",
        id.device_id, id.queue_id, drv_ret);
  }

  for (auto id : sched_output_queue_attrs_) {
    auto drv_ret = rtMemQueueAttach(id.device_id, id.queue_id, kAttachTimeOut);
    GE_CHK_BOOL_RET_STATUS(drv_ret == RT_ERROR_NONE, FAILED,
        "DynamicSched, mem queue attach failed, device_id=%d, sched_output_queue_attrs queue_id=%u, ret=%d",
        id.device_id, id.queue_id, drv_ret);
  }
  return SUCCESS;
}

Status HeterogeneousModelExecutor::WrapSingleModel() {
  model_relation_ = flow_model_->GetModelRelation();
  if (model_relation_ != nullptr) {
    return SUCCESS;
  }
  for (const auto &pne_model_pair : flow_model_->GetSubmodels()) {
    const PneModelPtr &single_model = pne_model_pair.second;
    GE_CHECK_NOTNULL(single_model);
    if (!single_model->GetSubmodels().empty()) {
      model_relation_ = single_model->GetModelRelation();
      break;
    }

    auto model_relation = ge::MakeShared<ModelRelation>();
    GE_CHECK_NOTNULL(model_relation);
    GE_CHK_STATUS_RET(ModelRelationBuilder().BuildForSingleModel(*single_model->GetRootGraph(),
                                                                 *model_relation),
                      "Failed to build model relation for single model");
    model_relation_ = std::move(model_relation);
  }
  return SUCCESS;
}

Status HeterogeneousModelExecutor::ParseInputTensorInfo() {
  // 1. build queue name to tensor desc mapping
  std::map<std::string, GeTensorDescPtr> input_tensor_desc_map;
  GE_CHK_STATUS_RET_NOLOG(BuildInputTensorDescMapping(input_tensor_desc_map));
  GELOGD("Start to set tensor info for inputs, size = %zu",
         model_relation_->root_model_endpoint_info.input_endpoint_names.size());
  GE_CHK_STATUS_RET_NOLOG(SetTensorInfo(input_tensor_desc_map,
                                        model_relation_->root_model_endpoint_info.input_endpoint_names,
                                        true));
  GE_CHK_STATUS_RET_NOLOG(BuildFusionInputTensorMapping());
  return SUCCESS;
}

Status HeterogeneousModelExecutor::BuildFusionInputTensorMapping() {
  is_fusion_input_.resize(input_queue_attrs_.size());
  for (size_t i = 0; i < input_queue_attrs_.size(); ++i) {
    fusion_input_queue_attrs_[input_queue_attrs_[i]].emplace_back(i);
  }

  for (const auto &it : fusion_input_queue_attrs_) {
    const auto &input_indices = it.second;
    auto is_fusion_input = input_indices.size() > 1;
    for (auto index : input_indices) {
      is_fusion_input_[index] = is_fusion_input;
      GELOGI("Build fusion input mapping success, input index = %zu, is fusion input = %d",
             index, static_cast<int32_t>(is_fusion_input));
    }
  }
  return SUCCESS;
}

Status HeterogeneousModelExecutor::GetIndicesToTensorDesc(const ComputeGraphPtr &compute_graph,
                                                          std::map<int64_t, GeTensorDescPtr> &indices_to_tensor_descs) {
  GE_CHECK_NOTNULL(compute_graph);
  for (const auto &node : compute_graph->GetDirectNode()) {
    if (!OpTypeUtils::IsDataNode(NodeUtils::GetNodeType(node))) {
      continue;
    }
    int64_t index = -1;
    if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_INDEX, index)) {
      GELOGE(PARAM_INVALID, "Failed to get data index, node name = %s", node->GetName().c_str());
      return PARAM_INVALID;
    }
    GELOGD("Data index of node [%s] = [%" PRId64 "]", node->GetName().c_str(), index);
    auto input_desc = node->GetOpDesc()->MutableOutputDesc(kDataOutputIndexVal0);
    GE_CHECK_NOTNULL(input_desc);
    (void)indices_to_tensor_descs.emplace(index, input_desc);
  }
  return SUCCESS;
}

Status HeterogeneousModelExecutor::BuildInputTensorDescMapping(std::map<std::string, GeTensorDescPtr> &mapping) {
  for (const auto &it : model_relation_->submodel_endpoint_infos) {
    const auto &submodel_name = it.first;
    const auto &input_endpoint_names = it.second.input_endpoint_names;
    const auto submodel = GetSubmodel(flow_model_, submodel_name);
    GE_CHECK_NOTNULL(submodel);
    std::map<int64_t, GeTensorDescPtr> indices_to_tensor_descs;
    GE_CHK_STATUS_RET_NOLOG(GetIndicesToTensorDesc(submodel->GetRootGraph(), indices_to_tensor_descs));
    if (indices_to_tensor_descs.size() != input_endpoint_names.size()) {
      GELOGE(PARAM_INVALID, "Number of inputs(%zu) mismatches that of input queues(%zu), sub model name = %s",
             indices_to_tensor_descs.size(), input_endpoint_names.size(), submodel_name.c_str());
      return PARAM_INVALID;
    }

    for (size_t i = 0U; i < input_endpoint_names.size(); ++i) {
      const auto &queue_name = input_endpoint_names[i];
      mapping[queue_name] = indices_to_tensor_descs[static_cast<int64_t>(i)];
    }
  }
  std::set<std::string> input_dummy_queue_names;
  for (const auto &endpoint : model_relation_->endpoints) {
    if (endpoint.GetEndpointType() == EndpointType::kDummyQueue) {
      input_dummy_queue_names.emplace(endpoint.GetName());
    }
  }
  if (!input_dummy_queue_names.empty()) {
    std::map<int64_t, GeTensorDescPtr> indices_to_tensor_descs;
    GE_CHK_STATUS_RET_NOLOG(GetIndicesToTensorDesc(flow_model_->GetRootGraph(), indices_to_tensor_descs));
    for (size_t i = 0U; i < model_relation_->root_model_endpoint_info.input_endpoint_names.size(); ++i) {
      const auto &queue_name = model_relation_->root_model_endpoint_info.input_endpoint_names[i];
      if (input_dummy_queue_names.count(queue_name) != 0U) {
        mapping[queue_name] = indices_to_tensor_descs[static_cast<int64_t>(i)];
        GELOGI("Root model input %zu is linked to netoutput by control link, queue name = %s", i, queue_name.c_str());
      }
    }
  }

  return SUCCESS;
}

Status HeterogeneousModelExecutor::BuildOutputTensorDescMapping(std::map<std::string, GeTensorDescPtr> &mapping) {
  GE_CHECK_NOTNULL(flow_model_);
  for (const auto &it : model_relation_->submodel_endpoint_infos) {
    const auto &submodel_name = it.first;
    const auto &output_endpoint_names = it.second.output_endpoint_names;
    const auto submodel = GetSubmodel(flow_model_, submodel_name);
    GE_CHECK_NOTNULL(submodel);
    const auto &subgraph = submodel->GetRootGraph();
    const auto net_output = subgraph->FindFirstNodeMatchType(NETOUTPUT);
    GE_CHECK_NOTNULL(net_output);
    const auto op_desc = net_output->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    auto output_desc_list = op_desc->GetAllInputsDescPtr();
    if (output_desc_list.size() != output_endpoint_names.size()) {
      GELOGE(PARAM_INVALID, "Number of outputs(%zu) mismatches that of output queues(%zu), sub model name = %s",
             output_desc_list.size(), output_endpoint_names.size(), submodel_name.c_str());
      return PARAM_INVALID;
    }
    for (size_t i = 0U; i < output_desc_list.size(); ++i) {
      const auto &queue_name = output_endpoint_names[i];
      mapping[queue_name] = std::move(output_desc_list.at(i));
    }
  }
  return SUCCESS;
}

Status HeterogeneousModelExecutor::ParseOutputTensorInfo() {
  std::map<std::string, GeTensorDescPtr> output_tensor_desc_map;
  GE_CHK_STATUS_RET_NOLOG(BuildOutputTensorDescMapping(output_tensor_desc_map));
  GELOGD("Start to set tensor info for outputs, size = %zu",
         model_relation_->root_model_endpoint_info.output_endpoint_names.size());
  GE_CHK_STATUS_RET_NOLOG(SetTensorInfo(
      output_tensor_desc_map, model_relation_->root_model_endpoint_info.output_endpoint_names, false));
  return SUCCESS;
}

Status HeterogeneousModelExecutor::SetTensorInfo(std::map<std::string, GeTensorDescPtr> &mapping,
                                                 const std::vector<std::string> &queue_names,
                                                 const bool is_input) {
  for (size_t i = 0U; i < queue_names.size(); ++i) {
    const auto &queue_name = queue_names[i];
    auto tensor_desc = mapping[queue_name];
    // actually this will not happen if the ModelRelation passes DeployPlanner
    if (tensor_desc == nullptr) {
      GELOGE(PARAM_INVALID, "queue name not found in submodels, name = %s", queue_name.c_str());
      return PARAM_INVALID;
    }

    bool is_no_tiling = false;
    (void)AttrUtils::GetBool(tensor_desc, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, is_no_tiling);
    GELOGD("%s tensor:%zu marks no-tiling is %d", is_input ? "Input" : "Output", i, static_cast<int32_t>(is_no_tiling));
    is_no_tiling = is_no_tiling || tensor_desc->MutableShape().IsUnknownShape();

    int64_t tensor_size = -1;
    (void)TensorUtils::GetSize(*tensor_desc, tensor_size);
    int64_t tensor_raw_size = -1;
    GE_CHK_GRAPH_STATUS_RET(TensorUtils::CalcTensorMemSize(tensor_desc->GetShape(),
                                                           tensor_desc->GetFormat(),
                                                           tensor_desc->GetDataType(),
                                                           tensor_raw_size),
                            "Failed to calc tensor raw size, queue name = %s", queue_name.c_str());
    GELOGD("index = %zu, queue name = %s, shape = [%s], tensor raw size = %" PRId64 ", "
           "padded size = %" PRId64 ", notiling = %d",
           i,
           queue_name.c_str(),
           tensor_desc->GetShape().ToString().c_str(),
           tensor_raw_size,
           tensor_size,
           static_cast<int32_t>(is_no_tiling));
    if ((!is_no_tiling) && ((tensor_raw_size < 0) || (tensor_size < 0))) {
      GELOGE(UNSUPPORTED, "Dynamic shape is not supported yet, raw/padded size = %" PRId64 "/%" PRId64 ", "
             "shape of queue[%s] = [%s]",
             tensor_raw_size,
             tensor_size,
             queue_name.c_str(),
             tensor_desc->GetShape().ToString().c_str());
      return UNSUPPORTED;
    }
    if (is_input) {
      input_tensor_sizes_.emplace_back(tensor_size);
      input_tensor_raw_sizes_.emplace_back(tensor_raw_size);
      input_tensor_desc_.emplace_back(std::move(tensor_desc));
      input_is_no_tiling_.push_back(is_no_tiling);
    } else {
      output_tensor_sizes_.emplace_back(tensor_size);
      output_tensor_raw_sizes_.emplace_back(tensor_raw_size);
      output_tensor_desc_.emplace_back(std::move(tensor_desc));
      output_is_no_tiling_.push_back(is_no_tiling);
    }
  }
  return SUCCESS;
}

uint32_t HeterogeneousModelExecutor::GetDeployedModelId() const {
  return deployed_model_id_;
}

Status HeterogeneousModelExecutor::Execute(const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs) {
  GELOGD("Start to execute model, model id = %u", model_id_);
  GE_CHK_STATUS_RET(EnqueueInputTensors(inputs, replica_num_), "Failed to enqueue input tensors with replica");
  GE_CHK_STATUS_RET(DequeueOutputTensors(outputs), "Failed to dequeue output tensors");
  GELOGD("Execute model successfully, model id = %u", model_id_);
  return SUCCESS;
}

Status HeterogeneousModelExecutor::GetTimeoutFromOption(int32_t &timeout) {
  std::string option_timeout;
  if ((GetContext().GetOption(OPTION_EXEC_GRAPH_EXEC_TIMEOUT, option_timeout) == GRAPH_SUCCESS) &&
      (!option_timeout.empty())) {
    try {
      timeout = std::stoi(option_timeout);
    } catch (std::out_of_range &) {
      GELOGE(FAILED, "[Queue] Option timeout:%s is out of range.", option_timeout.c_str());
      return FAILED;
    } catch (std::invalid_argument &) {
      GELOGE(FAILED, "[Queue] Option timeout:%s is invalid.", option_timeout.c_str());
      return FAILED;
    }

    if ((timeout < kQueueNeverTimeout) || (timeout == 0) || ((timeout % kTimeoutValuePerSecond) != 0)) {
      GELOGE(FAILED, "Timeout value:%d is incorrect.", timeout);
      return FAILED;
    }
  }
  return SUCCESS;
}

Status HeterogeneousModelExecutor::EnqueueInputTensors(const std::vector<GeTensor> &inputs, const size_t replica_num) {
  GELOGD("Start to enqueue inputs with replica, size = %zu, replica_num = %zu.", inputs.size(), replica_num);
  for (size_t i = 0U; i < replica_num; ++i) {
    GE_CHK_STATUS_RET(EnqueueInputTensors(inputs), "Failed to enqueue input tensors.");
  }
  return SUCCESS;
}

Status HeterogeneousModelExecutor::FillFusionInput(const std::vector<GeTensor> &fusion_inputs,
                                                   void *const buffer,
                                                   const size_t size) const {
  auto fusion_buffer = PtrToPtr<void, uint8_t>(buffer);
  auto fusion_size = size;
  for (size_t i = 0U; i < fusion_inputs.size(); ++i) {
    const auto &tensor_desc = fusion_inputs[i].GetTensorDesc();
    const auto &tensor_data = fusion_inputs[i].GetData().GetData();
    const auto &tensor_size = fusion_inputs[i].GetData().GetSize();
    RuntimeTensorDesc *const runtime_desc = PtrToPtr<void, RuntimeTensorDesc>(fusion_buffer);
    GE_CHK_STATUS_RET(DataFlowExecutorUtils::FillRuntimeTensorDesc(tensor_desc, *runtime_desc));
    auto *const mbuf_tensor_data = PtrAdd<uint8_t>(fusion_buffer, fusion_size, sizeof(RuntimeTensorDesc));
    fusion_size -= sizeof(RuntimeTensorDesc);
    const size_t dst_size = (fusion_size > SECUREC_MEM_MAX_LEN) ? SECUREC_MEM_MAX_LEN : fusion_size;
    if (tensor_size > 0 && memcpy_s(mbuf_tensor_data, dst_size, tensor_data, tensor_size) != EOK) {
      GELOGE(FAILED, "Failed to copy mbuf data, mbuf size:%zu, tensor size:%zu", fusion_size, tensor_size);
      return FAILED;
    }
    runtime_desc->data_addr = PtrToValue(mbuf_tensor_data);
    fusion_buffer = PtrAdd<uint8_t>(mbuf_tensor_data, fusion_size, tensor_size);
    fusion_size -= tensor_size;
  }
  return SUCCESS;
}

Status HeterogeneousModelExecutor::EnqueueFusionInputs(const std::map<DeployQueueAttr,
                                                                      std::vector<GeTensor>> &fusion_inputs,
                                                       ExchangeService::ControlInfo &control_info) const {
  for (const auto &it : fusion_inputs) {
    const auto &queue_attr = it.first;
    const auto &tensors = it.second;
    uint64_t total_size = 0U;
    for (const auto &input : tensors) {
      const auto tensor_size = input.GetData().GetSize();
      GE_CHECK_LE(total_size, UINT64_MAX - tensor_size);
      total_size += tensor_size;
      GE_CHECK_LE(total_size, UINT64_MAX - sizeof(RuntimeTensorDesc));
      total_size += sizeof(RuntimeTensorDesc);
    }
    const ExchangeService::FillFunc fill_func = [this, &tensors](void *const buffer, const size_t size) {
      return FillFusionInput(tensors, buffer, size);
    };
    GE_CHK_STATUS_RET(exchange_service_->Enqueue(static_cast<int32_t>(queue_attr.device_id),
                                                 queue_attr.queue_id,
                                                 total_size,
                                                 fill_func,
                                                 control_info),
                      "Failed to enqueue input, model id = %u, queue id = %u", model_id_, queue_attr.queue_id);
    GELOGD("Enqueue input successfully, model id = %u, queue id = %u, size = %" PRIu64,
           model_id_, queue_attr.queue_id, total_size);
  }
  return SUCCESS;
}

Status HeterogeneousModelExecutor::EnqueueInputTensors(const std::vector<GeTensor> &inputs) {
  GELOGD("Start to enqueue inputs, size = %zu", inputs.size());
  GE_CHK_STATUS_RET_NOLOG(ValidateInputTensors(inputs));
  int32_t timeout = kQueueDefaultTimeout;
  GE_CHK_STATUS_RET(GetTimeoutFromOption(timeout), "Get graph timeout failed.");

  ExchangeService::ControlInfo control_info{};
  control_info.timeout = timeout;
  std::map<DeployQueueAttr, std::vector<GeTensor>> fusion_inputs;
  for (size_t i = 0U; i < inputs.size(); ++i) {
    const auto &input = inputs[i];
    const auto &queue_attr = input_queue_attrs_[i];
    if (is_fusion_input_[i]) {
      fusion_inputs[queue_attr].emplace_back(input);
      continue;
    }

    const auto tensor_data = input.GetData().GetData();
    const auto tensor_size = input.GetData().GetSize();
    // Enqueue tensor
    const auto &tensor_desc = input.GetTensorDesc();
    RuntimeTensorDesc runtime_tensor_desc = {};
    GE_CHK_STATUS_RET(DataFlowExecutorUtils::FillRuntimeTensorDesc(tensor_desc, runtime_tensor_desc));
    const ExchangeService::BuffInfo tensor_desc_buff = {.addr = &runtime_tensor_desc,
                                                        .len = sizeof(RuntimeTensorDesc)};
    const ExchangeService::BuffInfo data_buff = {.addr = ValueToPtr(PtrToValue(tensor_data)), .len = tensor_size};
    const std::vector<ExchangeService::BuffInfo> buffs = {tensor_desc_buff, data_buff};
    GELOGI("Enqueue buffs size is %zu, tensor_size is %zu.", buffs.size(), tensor_size);
    GE_CHK_STATUS_RET(exchange_service_->Enqueue(queue_attr.device_id, queue_attr.queue_id, buffs, control_info),
                      "Failed to enqueue input[%zu], model id=%u, queue id=%u", i, model_id_, queue_attr.queue_id);

    GELOGD("Enqueue input[%zu] successfully, model id = %u, queue id = %u, size = %zu, notiling = %d",
           i, model_id_, queue_attr.queue_id, tensor_size, static_cast<int32_t>(input_is_no_tiling_[i]));
  }

  GE_CHK_STATUS_RET(EnqueueFusionInputs(fusion_inputs, control_info), "Failed to enqueue fusion inputs.");

  for (size_t i = 0U; i < control_input_queue_attrs_.size(); ++i) {
    const auto &queue_attr = control_input_queue_attrs_[i];
    const int32_t control_value = 0;
    GE_CHK_STATUS_RET(
        exchange_service_->Enqueue(queue_attr.device_id,
                                   queue_attr.queue_id,
                                   &control_value,
                                   sizeof(control_value),
                                   control_info),
        "Failed to enqueue control input[%zu], model id = %u, queue id = %u", i, model_id_, queue_attr.queue_id);
  }
  return SUCCESS;
}

Status HeterogeneousModelExecutor::ValidateInputTensors(const std::vector<GeTensor> &inputs) {
  if (inputs.size() != input_queue_attrs_.size()) {
    GELOGE(PARAM_INVALID,
           "Number of inputs (%zu) mismatches that of model inputs (%zu).", inputs.size(),
           input_queue_attrs_.size());
    return PARAM_INVALID;
  }

  for (size_t i = 0U; i < inputs.size(); ++i) {
    if (!input_is_no_tiling_[i]) {
      const auto &input = inputs[i];
      const auto tensor_size = input.GetData().GetSize();
      REQUIRE_COMPAT_INT64(tensor_size);
      GE_CHK_BOOL_RET_STATUS(static_cast<int64_t>(tensor_size) <= input_tensor_sizes_[i],
                             PARAM_INVALID,
                             "Model[%u] validate input tensor[%zu] failed, expect at most = %" PRId64 ", "
                             "but given = %zu", model_id_, i, input_tensor_sizes_[i], tensor_size);
    }
  }
  return SUCCESS;
}

Status HeterogeneousModelExecutor::DequeueControlOutputs(const size_t replica_num, const int32_t timeout) {
  GELOGD("Start to dequeue control outputs with replica, size = %zu, replica_num = %zu",
         control_output_queue_attrs_.size(), replica_num);
  for (size_t i = 0U; i < replica_num; ++i) {
    for (size_t j = 0U; j < control_output_queue_attrs_.size(); ++j) {
      const auto &queue_attr = control_output_queue_attrs_[j];
      uint8_t data[kAlignmentVal64] = {};
      ExchangeService::ControlInfo control_info{};
      control_info.end_of_sequence_flag = false;
      control_info.timeout = timeout;
      GE_CHK_STATUS_RET(
          exchange_service_->Dequeue(queue_attr.device_id, queue_attr.queue_id, &data[0], 1U, control_info),
          "Failed to dequeue control output[%zu], model id = %u, queue id = %u", j, model_id_, queue_attr.queue_id);
    }
  }
  GELOGD("Dequeue control outputs successfully, model id = %u.", model_id_);
  return SUCCESS;
}

Status HeterogeneousModelExecutor::DequeueOutputTensors(std::vector<GeTensor> &outputs) {
  size_t end_of_sequence_num = 0UL;
  Status status = SUCCESS;
  ExchangeService::ControlInfo control_info{};
  control_info.timeout = kQueueDefaultTimeout;
  GE_CHK_STATUS_RET(GetTimeoutFromOption(control_info.timeout), "Get graph timeout failed.");
  for (size_t i = 0U; i < output_queue_attrs_.size(); ++i) {
    // Dequeue tensor
    const auto &queue_attr = output_queue_attrs_[i];
    auto output_tensor_raw_size = output_tensor_raw_sizes_[i];
    const auto &output_tensor_desc = output_tensor_desc_[i];
    GeTensor output_tensor(*output_tensor_desc);
    control_info.end_of_sequence_flag = false;
    std::shared_ptr<AlignedPtr> aligned_ptr = nullptr;
    if (!output_is_no_tiling_[i]) {
      GELOGD("MakeShared start size=%" PRId64, output_tensor_raw_size);
      aligned_ptr = MakeShared<AlignedPtr>(output_tensor_raw_size, kAlignmentVal64);
      GELOGD("MakeShared end size=%" PRId64, output_tensor_raw_size);
      GE_CHECK_NOTNULL(aligned_ptr);
      control_info.skip_size = sizeof(RuntimeTensorDesc);
    }

    const auto deq_ret = DoDequeue(output_tensor, aligned_ptr, control_info, i);
    if (deq_ret != SUCCESS && deq_ret != ACL_ERROR_GE_REDEPLOYING) {
      status = deq_ret;
      continue;
    }
    GELOGD(
        "Dequeue output[%zu] successfully, model id = %u, queue id = %u, size = %" PRId64 ", notiling = %d",
        i, model_id_, queue_attr.queue_id, output_tensor_raw_size, static_cast<int32_t>(output_is_no_tiling_[i]));
    if (control_info.end_of_sequence_flag) {
      end_of_sequence_num++;
      continue;
    }
    outputs.emplace_back(std::move(output_tensor));
  }

  GE_CHK_STATUS_RET(DequeueControlOutputs(replica_num_, control_info.timeout), "Dequeue control outputs failed.");
  if (end_of_sequence_num > 0U) {
    GELOGI("return end of sequence.");
    return END_OF_SEQUENCE;
  }
  return status;
}

Status HeterogeneousModelExecutor::DoDequeue(GeTensor &output_tensor,
                                             std::shared_ptr<AlignedPtr> &aligned_ptr,
                                             ExchangeService::ControlInfo &control_info,
                                             const size_t output_index) {
  Status status = SUCCESS;
  const int32_t total_timeout = control_info.timeout;
  int32_t left_wait_time = total_timeout;
  GELOGD("Do dequeue output[%zu] begin, timeout=%d", output_index, total_timeout);
  GE_DISMISSABLE_GUARD(timeout, ([&control_info, total_timeout]() { control_info.timeout = total_timeout; }));
  while (true) {
    // -1 means always wait when queue is empty.
    const int32_t wait_time =
        ((total_timeout == -1) || (left_wait_time >= kRetryInterval)) ? kRetryInterval : left_wait_time;
    control_info.timeout = wait_time;
    status = DoDequeueOnce(output_tensor, aligned_ptr, control_info, output_index);
    if (status == SUCCESS) {
      break;
    }
    if (status == RT_ERROR_TO_GE_STATUS(ACL_ERROR_RT_QUEUE_EMPTY)) {
      if (left_wait_time >= wait_time) {
        left_wait_time -= wait_time;
      }
      if ((total_timeout == -1) || (left_wait_time > 0)) {
        continue;
      }
      GE_DISMISS_GUARD(timeout);  // change timeout to 1s, make other output dequeue soon
      GELOGE(FAILED, "Dequeue output[%zu] timeout, model id=%u, queue id=%u, total_timeout = %d ms", output_index,
             model_id_, output_queue_attrs_[output_index].queue_id, total_timeout);
    }
    GELOGE(FAILED, "Failed to dequeue output[%zu], model id = %u, queue id = %u, ret = %u", output_index, model_id_,
           output_queue_attrs_[output_index].queue_id, status);
    return status;
  }
  GELOGD("Do dequeue output[%zu] end.", output_index);
  return status;
}

Status HeterogeneousModelExecutor::DoDequeue(FlowMsgBasePtr &flow_msg,
                                             ExchangeService::ControlInfo &control_info,
                                             size_t output_index) {
  Status ret = SUCCESS;
  const int32_t total_timeout = control_info.timeout;
  int32_t left_wait_time = total_timeout;
  GELOGD("Do dequeue output[%zu] begin, timeout=%d", output_index, total_timeout);
  GE_DISMISSABLE_GUARD(timeout, ([&control_info, total_timeout]() { control_info.timeout = total_timeout; }));
  while (true) {
    // -1 means always wait when queue is empty.
    const int32_t wait_time =
        ((total_timeout == -1) || (left_wait_time >= kRetryInterval)) ? kRetryInterval : left_wait_time;
    control_info.timeout = wait_time;
    ret = DoDequeueOnce(flow_msg, control_info, output_index);
    if (ret == SUCCESS) {
      break;
    }
    if (ret == RT_ERROR_TO_GE_STATUS(ACL_ERROR_RT_QUEUE_EMPTY)) {
      if (left_wait_time >= wait_time) {
        left_wait_time -= wait_time;
      }
      if ((total_timeout == -1) || (left_wait_time > 0)) {
        continue;
      }
      GE_DISMISS_GUARD(timeout);  // change timeout to 1s, make other output dequeue soon
      GELOGE(FAILED, "Dequeue output[%zu] flow msg timeout, model id=%u, queue id=%u, total_timeout = %d ms",
             output_index, model_id_, output_queue_attrs_[output_index].queue_id, total_timeout);
    }
    GELOGE(FAILED, "Failed to dequeue output[%zu] flow msg, model id = %u, queue id = %u, ret = %u",
           output_index, model_id_, output_queue_attrs_[output_index].queue_id, ret);
    return ret;
  }
  GELOGD("Do dequeue output[%zu] end.", output_index);
  return ret;
}

Status HeterogeneousModelExecutor::DoDequeueOnce(GeTensor &output_tensor, std::shared_ptr<AlignedPtr> &aligned_ptr,
                                                 ExchangeService::ControlInfo &control_info,
                                                 const size_t output_index) {
  GE_CHK_STATUS_RET_NOLOG(GetRedeployStatus());
  const auto &queue_attr = output_queue_attrs_[output_index];
  const auto device = queue_attr.device_id;
  const auto queue_id = queue_attr.queue_id;
  const auto output_tensor_raw_size = output_tensor_raw_sizes_[output_index];
  Status status = SUCCESS;
  if (aligned_ptr == nullptr) {
    status = exchange_service_->DequeueTensor(device, queue_id, output_tensor, control_info);
  } else {
    status = exchange_service_->Dequeue(device, queue_id, aligned_ptr->MutableGet(),
                                        static_cast<size_t>(output_tensor_raw_size), control_info);
  }

  if (status == SUCCESS) {
    if (aligned_ptr != nullptr) {
      output_tensor.SetData(aligned_ptr, static_cast<uint64_t>(output_tensor_raw_size));
    }
    return SUCCESS;
  }
  if (status == RT_ERROR_TO_GE_STATUS(ACL_ERROR_RT_QUEUE_EMPTY)) {
    if (dev_abnormal_callback_ != nullptr) {
      Status dev_abnormal_code = dev_abnormal_callback_();
      GE_CHK_STATUS_RET(dev_abnormal_code, "device is abnormal, error code=%u", dev_abnormal_code);
    }
  }
  return status;
}

Status HeterogeneousModelExecutor::DoDequeueOnce(FlowMsgBasePtr &flow_msg,
                                                 ExchangeService::ControlInfo &control_info,
                                                 size_t output_index) {
  GE_CHK_STATUS_RET_NOLOG(GetRedeployStatus());
  const auto &queue_attr = output_queue_attrs_[output_index];
  Status status = io_helper_.FetchFlowMsg(queue_attr, control_info, output_tensor_desc_[output_index], flow_msg);
  if (status == RT_ERROR_TO_GE_STATUS(ACL_ERROR_RT_QUEUE_EMPTY)) {
    if (dev_abnormal_callback_ != nullptr) {
      Status dev_abnormal_code = dev_abnormal_callback_();
      GE_CHK_STATUS_RET(dev_abnormal_code, "device is abnormal, error code=%u", dev_abnormal_code);
    }
  }
  return status;
}

Status HeterogeneousModelExecutor::ModelRunStart() {
  GELOGI("model id = %u, deployed model id = %u", model_id_, deployed_model_id_);
  const std::lock_guard<std::mutex> lk(mu_);
  run_flag_ = true;
  run_context_ = GetThreadLocalContext();

  if (is_dynamic_sched_) {
    uint32_t threadNum = sched_output_queue_attrs_.size();
    for (auto iter : datagw_rqt_to_rsp_) {
      sched_input_cnt_[iter.first] = 0;
    }
    for (uint32_t i = 0U; i < threadNum; i++) {
      std::thread sched_thread = std::thread([this, i]() {
        std::string threadName = "ge_dpl_iosc_" + std::to_string(i);
        SET_THREAD_NAME(pthread_self(), threadName.c_str());
        GetThreadLocalContext() = run_context_;
        GE_CHK_STATUS(SchedRun(i), "DynamicSched, SchedRun failed");
      });
      sched_threads_.emplace_back(std::move(sched_thread));
    }
  }
  if (is_dynamic_sched_ || is_exception_catch_) {
    status_run_thread_ = std::thread([this]() {
      SET_THREAD_NAME(pthread_self(), "ge_dpl_iostat");
      GetThreadLocalContext() = run_context_;
      GE_CHK_STATUS(StatusRun(), "DynamicSched, StatusRun failed");
    });
    auto func = [this](const domi::SubmodelStatus& request) {
      (void)status_messages_queue_.Push(std::move(request));
      return SUCCESS;
    };
    GE_CHK_STATUS_RET(process_forwarding_.RegisterCallBackFunc(StatusQueueMsgType::STATUS, func),
                      "Failed to register status callback func");
    process_forwarding_.Start();
  }
  return SUCCESS;
}

Status HeterogeneousModelExecutor::FindValidGroupEntry(uint32_t uuid, int32_t logic_queue_id, int32_t logic_group_id,
    DeployPlan::DstGroupInfo **group_entry_ptr) {
  auto model_iter = model_index_info_.find(uuid);
  GE_CHK_BOOL_RET_SPECIAL_STATUS(model_iter == model_index_info_.end(),
                         PARAM_INVALID,
                         "DynamicSched can't find valid queue status data, uuid=%u", uuid);
  auto group_iters = model_iter->second.find(logic_queue_id); // 模型输出逻辑id
  GE_CHK_BOOL_RET_SPECIAL_STATUS(group_iters == model_iter->second.end() || !(group_iters->second.first.is_normal),
                         PARAM_INVALID,
                         "DynamicSched can't find valid queue status data, uuid=%u,"
                         " logic_queue_id=%d", uuid, logic_queue_id);
  auto group_iter = group_iters->second.second.find(logic_group_id);
  GE_CHK_BOOL_RET_SPECIAL_STATUS(group_iter == group_iters->second.second.end(),
                         PARAM_INVALID,
                         "DynamicSched can't find valid queue status data, uuid=%u,"
                         " logic_queue_id=%d, logic_group_id=%d", uuid, logic_queue_id, logic_group_id);
  GE_CHK_BOOL_RET_SPECIAL_STATUS(group_iter->second.routes.size() == 0,
                                 PARAM_INVALID,
                                 "DynamicSched group size is 0!");
  *group_entry_ptr = &(group_iter->second);
  return SUCCESS;
}

void HeterogeneousModelExecutor::ProcAfterFindGroupEntry(DeployPlan::DstGroupInfo &group_info,
                                                         int32_t group_entry_index) {
  // group_entry_index为上一层函数从group_info遍历获取，无越界风险
  uint32_t dst_uuid = queue_status_info_[group_info.routes[group_entry_index].endpoint_index].first.model_uuid;
  queue_status_info_[group_info.routes[group_entry_index].endpoint_index].second = Now();
  // 路径被选中后队列深度加1，下次选路优先级降低（做出的选择若直接体现在结果上，可能导致不同routelabel做出相同路径选择）
  queue_status_info_[group_info.routes[group_entry_index].endpoint_index].first.queue_depth++;
  GELOGI("DynamicSched found group_entry_index=%d, dst_uuid=%u, group_num=%u, mul_submodel_id=%u.",
      group_entry_index, dst_uuid, group_info.routes.size(), group_info.model_id);
}

bool HeterogeneousModelExecutor::FindGroupEntryIndexInSingleInstance(DeployPlan::DstGroupInfo &group_info,
  std::pair<int32_t, std::string> &group_entry_index_and_name) {
  std::pair<int32_t, std::string> group_main_entry_index_and_name;
  std::pair<int32_t, std::string> group_redundant_entry_index_and_name;
  uint32_t normal_instance_num = 0U;
  for (uint32_t i = 0U; i < group_info.routes.size(); i++) {
    if (!group_info.routes[i].extended_info.is_normal) {
      GELOGI("DynamicSched, group index=%u is bind to abnormal device", i);
      continue;
    }
    if (group_info.routes[i].is_redundant) {
      group_redundant_entry_index_and_name.first = i;
      group_redundant_entry_index_and_name.second = group_info.routes[i].extended_info.submodel_instance_name;
      GELOGI("DynamicSched, group index=%u is bind to redundant instance", i);
      continue;
    }
    group_main_entry_index_and_name.first = i;
    group_main_entry_index_and_name.second = group_info.routes[i].extended_info.submodel_instance_name;
    normal_instance_num++;
  }
  if (normal_instance_num > 1U) {
    return false;
  }
  group_entry_index_and_name =
    normal_instance_num == 1U ? group_main_entry_index_and_name : group_redundant_entry_index_and_name;

  return true;
}

bool HeterogeneousModelExecutor::FindGroupEntryIndexFromCache(DeployPlan::DstGroupInfo &group_info,
  std::pair<int32_t, std::string> &group_entry_index_and_name,
  uint64_t trans_id,
  uint32_t route_label) {
  const std::lock_guard<std::mutex> lk(cache_mu_);
  auto cache_iter = routelabel_cache_info_.find(std::make_pair(trans_id, route_label));
  if (cache_iter != routelabel_cache_info_.end()) {
    auto groups_iter = cache_iter->second.find(group_info.routes.size());
    if (groups_iter != cache_iter->second.end()) {
      group_entry_index_and_name = groups_iter->second;
      GELOGI("DynamicSched, found routelabel cache info, trans_id=%" PRIu64 ", route_label=%u,"
          " group_entry_index=%d", trans_id, route_label, group_entry_index_and_name.first);
      const std::lock_guard<std::mutex> lk2(queue_status_mu_);
      ProcAfterFindGroupEntry(group_info, group_entry_index_and_name.first);
      return true;
    }
  }
  return false;
}

void HeterogeneousModelExecutor::UpdateQueueDefaultStatus(const DeployPlan::DstGroupInfo &group_info, int32_t device_id,
                                                          int32_t device_type, uint32_t i) {
  QueueStatus status{};
  status.queue_depth = 0;
  status.device_id = device_id;
  status.device_type = device_type;
  GELOGI("DynamicSched doesn't find status info, set default depth=0, dst endpoint index=%d.",
         group_info.routes[i].endpoint_index);
  queue_status_info_[group_info.routes[i].endpoint_index].first = status;
  queue_status_info_[group_info.routes[i].endpoint_index].second = 0;
}

void HeterogeneousModelExecutor::FindGroupEntryIndexBySchedule(
    DeployPlan::DstGroupInfo &group_info, int32_t device_id, int32_t device_type,
    std::pair<int32_t, std::string> &group_entry_index_and_name) {
  auto depth = INT32_MAX;
  auto cur_depth = 0;
  uint64_t duration = 0;
  const std::lock_guard<std::mutex> lk(queue_status_mu_);
  for (uint32_t i = 0U; i < group_info.routes.size(); i++) {
    if (!group_info.routes[i].extended_info.is_normal) {
      GELOGI("DynamicSched, group index=%u is bind to abnormal device", i);
      continue;
    }
    if (group_info.routes[i].is_redundant) {
      GELOGI("DynamicSched, group index=%u is bind to redundant instance", i);
      continue;
    }
    auto iter = queue_status_info_.find(group_info.routes[i].endpoint_index);
    if (iter == queue_status_info_.end()) {
      // 异常流程，设置默认状态
      UpdateQueueDefaultStatus(group_info, device_id, device_type, i);
      iter = queue_status_info_.find(group_info.routes[i].endpoint_index);
      GE_CHK_BOOL_EXEC(iter != queue_status_info_.end(), return,
                       "DynamicSched doesn't find status info, dst endpoint index=%d",
                       group_info.routes[i].endpoint_index);
    } else {
      GELOGI(
          "DynamicSched search group entry info: entry logic id=%d, entry index=%d dst endpoint index=%d,"
          " queue_depth=%d, device_id=%d, uuid=%u, device_type=%d, pre time=%llu.",
          group_info.routes[i].entry_index, i, group_info.routes[i].endpoint_index, iter->second.first.queue_depth,
          iter->second.first.device_id, iter->second.first.model_uuid, iter->second.first.device_type,
          iter->second.second);
      cur_depth = iter->second.first.queue_depth;
    }
    if (cur_depth < depth) {
      depth = cur_depth;
      group_entry_index_and_name.first = i;
      group_entry_index_and_name.second = group_info.routes[i].extended_info.submodel_instance_name;
      duration = Now() - iter->second.second;
    } else if (cur_depth == depth) {  // 如果队列深度一样，挑选上一次发送时间距今最久的路径
      uint64_t cur_duration = Now() - iter->second.second;
      if (cur_duration > duration) {
        duration = cur_duration;
        group_entry_index_and_name.first = i;
        group_entry_index_and_name.second = group_info.routes[i].extended_info.submodel_instance_name;
      }
    }
  }
  ProcAfterFindGroupEntry(group_info, group_entry_index_and_name.first);
  return;
}

void HeterogeneousModelExecutor::DeleteInvalidCache() {
  while (cached_trans_ids_.size() > kCacheMaxTransIdNum) {
    auto head_iter = cached_trans_ids_.cbegin();
    uint64_t trans_id = head_iter->first;
    const auto &route_labels = head_iter->second;
    for (uint32_t route_label : route_labels) {
      routelabel_cache_info_.erase({trans_id, route_label});
      GELOGD("DynamicSched, delete route label cache info, trans_id=%" PRIu64 ", route_label=%u.",
        trans_id, route_label);
    }
    (void)cached_trans_ids_.erase(head_iter);
  }
}

template<typename T>
Status HeterogeneousModelExecutor::GetQueueInfoByDequeueMbuf(const int32_t device_id,
    const uint32_t queue_id, T &info, const int32_t time_out) const {
  rtMbufPtr_t m_buf = nullptr;
  auto ret = exchange_service_->DequeueMbuf(device_id, queue_id, &m_buf, time_out);
  if (ret != SUCCESS) {
    GELOGW("DynamicSched DequeueMbuf failed, device_id=%u, queue_id = %u", device_id, queue_id);
    return FAILED;
  }

  GELOGD("DynamicSched [Dequeue][Message] success");
  GE_MAKE_GUARD(m_buf, [m_buf]() { (void)rtMbufFree(m_buf); });
  void *buffer_addr = nullptr;
  uint64_t buffer_size = 0;
  GE_CHK_RT_RET(rtMbufGetBuffAddr(m_buf, &buffer_addr));
  GE_CHK_RT_RET(rtMbufGetBuffSize(m_buf, &buffer_size));
  google::protobuf::io::ArrayInputStream stream(buffer_addr, static_cast<int32_t>(buffer_size));
  GE_CHK_BOOL_RET_STATUS(info.ParseFromZeroCopyStream(&stream), FAILED,
      "DynamicSched info parse from zero copy stream failed, queue_id=%u", queue_id);
  return SUCCESS;
}

Status HeterogeneousModelExecutor::DynamicSchedProc(const domi::FlowgwRequest &flowgw_request,
                                                    int32_t queue_infos_index,
                                                    domi::FlowgwResponse &flowgw_response) {
  int32_t node_id = flowgw_request.node_id();
  int32_t datagw_input_index = flowgw_request.input_index();
  const auto &queue_infos = flowgw_request.queue_infos(queue_infos_index);
  int32_t device_id = queue_infos.queue_attrs().device_id();
  int32_t device_type = queue_infos.queue_attrs().device_type();
  int32_t queue_id = queue_infos.queue_attrs().queue_id();
  int32_t logic_id = queue_infos.queue_attrs().logic_id();
  int32_t logic_group_id = queue_infos.logic_group_id();
  uint32_t uuid = queue_infos.model_uuid();
  uint64_t trans_id = queue_infos.trans_id();
  uint32_t route_label = queue_infos.route_label();
  if (trans_id == 0) {
    trans_id = static_cast<uint64_t>(queue_infos.trans_id_old());
    route_label = static_cast<uint32_t>(queue_infos.route_label_old());
  }
  uint32_t root_model_id = queue_infos.root_model_id();
  GELOGI("DynamicSched FlowgwRequest info: node_id=%d, input_index=%d, queue_id=%d, device_type=%d, device_id=%d,"
      " logic_id=%d, logic_group_id=%d, model_uuid=%u, trans_id=%" PRIu64 ", route_label=%u, root_model_id=%u,"
      " queue_infos_index=%d", node_id, datagw_input_index, queue_id, device_type, device_id, logic_id,
      logic_group_id, uuid, trans_id, route_label, root_model_id, queue_infos_index);

  DeployPlan::DstGroupInfo *group_entry_ptr = nullptr;
  // group_entry: logic_group_id绑定的group entry数组<endpoint index, 对应的对端逻辑queueid(input)>
  GE_CHK_STATUS_RET_NOLOG(FindValidGroupEntry(uuid, logic_id, logic_group_id, &group_entry_ptr));
  auto group_entry = *group_entry_ptr;
  std::pair<int32_t, std::string> group_entry_index_and_name;
  group_entry_index_and_name.first = INT32_MAX;
  bool need_flowgw_cache = false;
  if (FindGroupEntryIndexInSingleInstance(group_entry, group_entry_index_and_name)) {
    need_flowgw_cache = true;
  } else if (!FindGroupEntryIndexFromCache(group_entry, group_entry_index_and_name, trans_id, route_label)) {
    FindGroupEntryIndexBySchedule(group_entry, device_id, device_type, group_entry_index_and_name);
    const std::lock_guard<std::mutex> lk(cache_mu_);
    // routelabel_cache_info_: key: trans_id; data: group_entry
    routelabel_cache_info_[std::make_pair(trans_id, route_label)][group_entry.routes.size()] =
      group_entry_index_and_name;
    (void)cached_trans_ids_[trans_id].emplace(route_label);
    DeleteInvalidCache();
    GELOGD("DynamicSched add sched cache info: uuid=%u, trans_id=%" PRIu64 ", route_label=%u,"
        " logic_group_id=%d, group_entry_index=%d, cached_trans_ids size=%zu.",
        uuid, trans_id, route_label, logic_group_id, group_entry_index_and_name.first, cached_trans_ids_.size());
  }

  auto queue_infos_rsp = flowgw_response.add_queue_infos();
  domi::QueueAttrs *queue_attrs = queue_infos_rsp->mutable_queue_attrs();
  queue_attrs->set_queue_id(queue_id);
  queue_attrs->set_device_id(device_id);
  queue_attrs->set_device_type(device_type);
  queue_attrs->set_logic_id(logic_id);
  queue_infos_rsp->set_trans_id(trans_id);
  queue_infos_rsp->set_route_label(route_label);
  // compatible for old version
  queue_infos_rsp->set_trans_id_old(static_cast<int32_t>(trans_id));
  queue_infos_rsp->set_route_label_old(static_cast<int32_t>(route_label));
  queue_infos_rsp->set_choose_logic_id(group_entry_index_and_name.first); // 直接把group的index给QS
  queue_infos_rsp->set_logic_group_id(logic_group_id);
  queue_infos_rsp->set_root_model_id(root_model_id);
  queue_infos_rsp->set_need_cache(need_flowgw_cache);
  GELOGI("DynamicSched FlowgwResponse info: queue_id=%d, device_type=%d, device_id=%d, logic_id=%d, logic_group_id=%d,"
      " model_uuid=%u, trans_id=%" PRIu64 ", route_label=%u, choose_logic_id=%d, root_model_id=%u, queue_infos_index=%d,"
      " entry_logic_id=%d, group_num=%u, mul_submodel_id=%u, need_flowgw_cache=%d, choose_instance_name=%s.",
      queue_id, device_type, device_id,
      logic_id, logic_group_id, uuid, trans_id, route_label, group_entry_index_and_name.first, root_model_id,
      queue_infos_index, group_entry.routes[group_entry_index_and_name.first].entry_index, group_entry.routes.size(),
      group_entry.model_id, static_cast<int32_t>(need_flowgw_cache), group_entry_index_and_name.second.c_str());
  return SUCCESS;
}

Status HeterogeneousModelExecutor::FlowgwResponseEnqueue(int32_t device_id, int32_t datagw_input_index,
                                                         domi::FlowgwResponse &flowgw_response) {
  auto iter = datagw_rqt_to_rsp_.find(datagw_input_index);
  GE_CHK_BOOL_RET_STATUS(iter != datagw_rqt_to_rsp_.end(),
      FAILED, "DynamicSched can't find datagw request input indices id=%d", datagw_input_index);
  int32_t out_queue_id = iter->second.queue_id;
  GELOGI("DynamicSched FlowgwResponse Enqueue info: datagw_input_index=%d, out_queue_id=%d.",
      datagw_input_index, out_queue_id);
  auto rsp_size = flowgw_response.ByteSizeLong();
  ExchangeService::ControlInfo control_info{};
  ExchangeService::FillFunc fill_func = [&flowgw_response] (void *buffer, size_t size) {
    GE_CHK_BOOL_RET_STATUS(flowgw_response.SerializeToArray(buffer, static_cast<int32_t>(size)),
        FAILED, "DynamicSched flowgw_response serialize to array failed.");
    return SUCCESS;
  };
  GE_CHK_STATUS_RET(exchange_service_->Enqueue(device_id,
                                               out_queue_id,
                                               rsp_size,
                                               fill_func,
                                               control_info),
                    "DynamicSched Failed to enqueue flowgw_response");
  GELOGI("DynamicSched, sent scheding response, datagw_input_index=%d, datagw_input_cnt=%d.", datagw_input_index,
      sched_input_cnt_[datagw_input_index]++);
  return SUCCESS;
}

void HeterogeneousModelExecutor::DynamicSchedDurationStart() {
  call_ = Now();
  return;
}

void HeterogeneousModelExecutor::DynamicSchedDurationEnd() {
  uint64_t duration = Now() - call_;
  duration_total_ += duration;
  if (duration > kDynamicSchedDuration) {
    duration_size_++;
  }
  if (duration > duration_max_) {
    duration_max_ = duration;
  }
  cnt_total_++;
  call_ = 0ULL;
  return;
}

Status HeterogeneousModelExecutor::SchedRun(uint32_t index) {
  GELOGD("DynamicSched datagw request process thread started, model id=%u, deployed model id=%u",
      model_id_, deployed_model_id_);
  while (run_flag_) {
    GELOGD("DynamicSched datagw request process input queue info, size=%zu", sched_output_queue_attrs_.size());
    domi::FlowgwRequest flowgw_request;
    GE_CHK_BOOL_RET_STATUS(sched_output_queue_attrs_.size() != 0, FAILED,
        "DynamicSched, sched input queue size is 0!");
    const uint32_t queue_id = sched_output_queue_attrs_[index].queue_id;
    const int32_t device_id = sched_output_queue_attrs_[index].device_id;
    if (GetQueueInfoByDequeueMbuf(device_id, queue_id, flowgw_request, kDequeueMbufWaitTime) != SUCCESS) {
      continue;
    }

    DynamicSchedDurationStart();
    int32_t datagw_input_index = flowgw_request.input_index();
    GELOGI("DynamicSched receive scheding request, datagw_input_index=%d, datagw_input_cnt=%d.",
        datagw_input_index, sched_input_cnt_[datagw_input_index]);
    domi::FlowgwResponse flowgw_response;
    for (int32_t i = 0; i < flowgw_request.queue_infos_size(); ++i) {
      if (DynamicSchedProc(flowgw_request, i, flowgw_response) != SUCCESS) {
        GELOGW("DynamicSched, queue_index=%u is bind to abnormal device", i);
        continue;
      }
    }
    if (flowgw_response.queue_infos_size() > 0) {
      GE_CHK_STATUS_RET(FlowgwResponseEnqueue(device_id, datagw_input_index, flowgw_response),
          "DynamicSched proc enqueue flowgw response failed.");
    }
    DynamicSchedDurationEnd();
    GELOGD("DynamicSched Dequeue datagw request successfully, model_id=%u.", model_id_);
  }
  DynamicSchedInfoClear();
  GELOGD("DynamicSched datagw request process thread ended, model id=%u, deployed model id=%u.",
      model_id_, deployed_model_id_);
  return SUCCESS;
}

void HeterogeneousModelExecutor::UpdateQueueStatusInfo(const domi::SubmodelStatus &submodel_status,
                                                       int32_t queue_status_index) {
  const std::lock_guard<std::mutex> lk(queue_status_mu_);
  const auto &queue_status = submodel_status.queue_statuses(queue_status_index);
  int32_t input_consume_num = queue_status.input_consume_num();
  int32_t logic_queue_id = queue_status.queue_attrs().logic_id();
  queue_status_info_[logic_queue_id].first.queue_depth -= input_consume_num;
  return;
}

Status HeterogeneousModelExecutor::StatusRun() {
  GELOGD("DynamicSched Status Run thread started, model id=%u, deployed model id=%u",
      model_id_, deployed_model_id_);
  while (run_flag_) {
    GE_DISMISSABLE_GUARD(report_error, []() { REPORT_INNER_ERR_MSG("E19999", "Status run failed."); });
    GELOGD("DynamicSched start to dequeue status info, size=%zu", status_input_queue_attrs_.size());
    GE_CHK_BOOL_RET_STATUS(status_input_queue_attrs_.size() != 0, FAILED,
        "DynamicSched, status input queue size is 0!");
    domi::SubmodelStatus submodel_status;
    if ((!status_messages_queue_.Pop(submodel_status))) {
      GELOGI("Status message blocking queue is stopped.");
      break;
    }
    for (int32_t i = 0; i < submodel_status.queue_statuses_size(); ++i) {
      UpdateQueueStatusInfo(submodel_status, i);
    }
    GELOGD("DynamicSched Dequeue status outputs successfully, model_id=%u", model_id_);
  }
  GELOGD("DynamicSched Status Run thread ended, model id=%u, deployed model id=%u.", model_id_, deployed_model_id_);
  return SUCCESS;
}

bool HeterogeneousModelExecutor::IsModelInstanceAbnormal(const std::string &submodel_instance_name) {
  const auto iter = abnormal_submodel_instances_name_[deployed_model_id_].find(submodel_instance_name);
  if (iter != abnormal_submodel_instances_name_[deployed_model_id_].cend()) {
    GELOGI("ModelIndexInfoUpdate, excutor process is abnormal, submodel instance[%s] is on this process",
        submodel_instance_name.c_str());
    return true;
  }
  GELOGI("ModelIndexInfoUpdate, submodel instance[%s] is normals", submodel_instance_name.c_str());
  return false;
}

void HeterogeneousModelExecutor::DynamicSchedClear() {
  // 1、调度决策队列清空
  GELOGI("AbnormalDataClear, SchedRun input queue size=%zu", sched_output_queue_attrs_.size());
  domi::FlowgwRequest flowgw_request;
  for (auto id : sched_output_queue_attrs_) {
    const uint32_t queue_id = id.queue_id;
    const int32_t device_id = id.device_id;
    uint32_t times = 0U;
    Status status = SUCCESS;
    while (status == SUCCESS) {
      status = GetQueueInfoByDequeueMbuf(device_id,
                                          queue_id,
                                          flowgw_request);
      times++;
      GELOGI("AbnormalDataClear, SchedRun Dequeue clear, times=%u, device id=%d, queue id=%u, ret=%u",
          times, device_id, queue_id, status);
    }
  }
  // 2、状态管理队列清空
  GELOGI("AbnormalDataClear, StatusRun input queue size=%zu", status_input_queue_attrs_.size());
  domi::SubmodelStatus submodel_status;
  for (auto id : status_input_queue_attrs_) {
    const uint32_t queue_id = id.queue_id;
    const int32_t device_id = id.device_id;
    uint32_t times = 0U;
    Status status = SUCCESS;
    while (status == SUCCESS) {
      status = GetQueueInfoByDequeueMbuf(device_id,
                                          queue_id,
                                          submodel_status);
      times++;
      GELOGI("AbnormalDataClear, StatusRun Dequeue clear, times=%u, device id =%d, queue id =%u, ret=%u",
          times, device_id, queue_id, status);
    }
  }
  // 3、决策数据结构清空
  GELOGI("AbnormalDataClear, clear queue status(%zu), routelabel cache(%zu), cached transids(%zu)",
         queue_status_info_.size(), routelabel_cache_info_.size(), cached_trans_ids_.size());
  queue_status_info_.clear();
  cached_trans_ids_.clear();
  routelabel_cache_info_.clear();
  std::lock_guard<std::mutex> lk(data_aligner_mu_);
  for (auto &aligner : data_aligner_map_) {
    aligner.second->ClearCache();
  }
  GELOGI("AbnormalDataClear, clear success");
  return;
}

void HeterogeneousModelExecutor::ClearFeedData() {
  GELOGI("AbnormalDataClear, start to clear transId, input size = %zu, control input size = %zu",
      input_queue_attrs_.size(), control_input_queue_attrs_.size());
  for (size_t i = 0U; i < input_queue_attrs_.size(); ++i) {
    const auto &queue_attr = input_queue_attrs_[i];
    exchange_service_->ResetQueueInfo(queue_attr.device_id, queue_attr.queue_id);
    GELOGI("AbnormalDataClear, input[%zu], model id = %u, queue id = %u", i, model_id_, queue_attr.queue_id);
  }
  for (size_t i = 0U; i < control_input_queue_attrs_.size(); ++i) {
    const auto &queue_attr = control_input_queue_attrs_[i];
    exchange_service_->ResetQueueInfo(queue_attr.device_id, queue_attr.queue_id);
    GELOGI("AbnormalDataClear, control input[%zu], model id = %u, queue id = %u", i, model_id_, queue_attr.queue_id);
  }
  GELOGI("AbnormalDataClear, end clear transId.");
  return;
}

void HeterogeneousModelExecutor::ClearFetchData() {
  GELOGI("AbnormalDataClear, start to clear fetch data, output size = %zu,"
         " control output size = %zu, replica_num = %zu",
         output_queue_attrs_.size(), control_output_queue_attrs_.size(), replica_num_);
  for (uint32_t i = 0U; i < output_queue_attrs_.size(); i++) {
    const auto &queue_attr = output_queue_attrs_[i];
    ExchangeService::ControlInfo control_info{};
    control_info.end_of_sequence_flag = false;
    control_info.timeout = 0;
    Status status = SUCCESS;
    uint32_t times = 0U;
    while (status != ACL_ERROR_RT_QUEUE_EMPTY) {
      times++;
      status = exchange_service_->Dequeue(queue_attr.device_id, queue_attr.queue_id, nullptr, 0U, control_info);
      GELOGI("AbnormalDataClear, output queue[%zu], times=%u, ret=%u, device id =%d, queue id =%u",
          i, times, status, queue_attr.device_id, queue_attr.queue_id);
    }
  }

  for (size_t i = 0U; i < replica_num_; ++i) {
    for (size_t j = 0U; j < control_output_queue_attrs_.size(); ++j) {
      const auto &queue_attr = control_output_queue_attrs_[j];
      ExchangeService::ControlInfo control_info{};
      control_info.end_of_sequence_flag = false;
      control_info.timeout = 0;
      uint8_t data[kAlignmentVal64] = {};
      Status status = SUCCESS;
      uint32_t times = 0U;
      while (status != ACL_ERROR_RT_QUEUE_EMPTY) {
        times++;
        status = exchange_service_->Dequeue(queue_attr.device_id,
                                            queue_attr.queue_id,
                                            &data[0],
                                            kDequeueSize,
                                            control_info);
        GELOGI("AbnormalDataClear, dequeue control output[%zu], model id=%u, times=%u, ret=%u, device id =%d,"
            " queue id=%u,",
            j, model_id_, times, status, queue_attr.device_id, queue_attr.queue_id);
      }
    }
  }
  GELOGI("AbnormalDataClear, end clear fetch data.");
  return;
}

void HeterogeneousModelExecutor::ModelIndexGroupInfoUpdate(DeployPlan::DstGroupInfo &group_info) {
  for (auto group_entry_iter = group_info.routes.begin(); group_entry_iter != group_info.routes.end();
      group_entry_iter++) { // group_entry_index
    // 判断dst映射的模型实例是否异常，防止往异常的模型实例发数据
    if (IsModelInstanceAbnormal(group_entry_iter->extended_info.submodel_instance_name)) {
      group_entry_iter->extended_info.is_normal = false;
      GELOGI("ModelIndexInfoUpdate: group entry index[%u] is on abnormal process",
          group_entry_iter - group_info.routes.begin());
    }
  }
  return;
}

void HeterogeneousModelExecutor::ModelIndexInfoUpdate() {
  GELOGI("ModelIndexInfoUpdate: start to update model index info");
  for (auto model_iter = model_index_info_.begin(); model_iter != model_index_info_.end(); model_iter++) {
    for (auto group_iters = model_iter->second.begin(); group_iters != model_iter->second.end(); group_iters++) {
      // 判断src映射的submodel instance是否异常，防止从异常的submodel instance收到脏数据
      if (IsModelInstanceAbnormal(group_iters->second.first.submodel_instance_name)) {
        group_iters->second.first.is_normal = false;
        GELOGI("ModelIndexInfoUpdate: model id[%d], src endpoint index[%d] is on abnormal process",
            model_iter->first, group_iters->first);
        continue;
      }
      GELOGI("ModelIndexInfoUpdate: model id[%d], src endpoint index[%d] is on normal process",
          model_iter->first, group_iters->first);
      for (auto group_iter = group_iters->second.second.begin(); group_iter != group_iters->second.second.end();
          group_iter++) { // logic_group_id
        ModelIndexGroupInfoUpdate(group_iter->second);
      }
    }
  }
  GELOGI("ModelIndexInfoUpdate: end update model index info");
  return;
}

bool HeterogeneousModelExecutor::IsRedeployStart(const uint32_t abnormal_status_operation_type) const {
  return abnormal_status_operation_type == kCallbackStartRedeploy;
}

bool HeterogeneousModelExecutor::IsDynamicSched(const uint32_t abnormal_status_operation_type) const {
  return abnormal_status_operation_type == kCallbackDynamicSched;
}

bool HeterogeneousModelExecutor::IsRedeployFailed(const uint32_t abnormal_status_operation_type) const {
  return abnormal_status_operation_type == kCallbackFailedRedeploy;
}

Status HeterogeneousModelExecutor::ModelRunStop() {
  GELOGI("model id = %u, deployed model id = %u", model_id_, deployed_model_id_);
  const std::lock_guard<std::mutex> lk(mu_);
  run_flag_ = false;
  if (is_exception_catch_ || is_dynamic_sched_) {
    status_messages_queue_.Stop();
    process_forwarding_.Finalize();
    if (status_run_thread_.joinable()) {
      status_run_thread_.join();
    }
  }
  if (is_dynamic_sched_) {
    for (auto &sched_thread : sched_threads_) {
      if (sched_thread.joinable()) {
        sched_thread.join();
      }
    }
  }
  GELOGI("Model run stopped, model id = %u, deployed model id = %u", model_id_, deployed_model_id_);
  return SUCCESS;
}

void HeterogeneousModelExecutor::SetModelId(const uint32_t model_id) {
  model_id_ = model_id;
}

Status HeterogeneousModelExecutor::FeedEmptyEosData(ExchangeService::ControlInfo &control_info) const {
  if ((control_info.msg_info->flags & static_cast<uint32_t>(DataFlowFlag::DATA_FLOW_FLAG_EOS)) != 0U) {
    control_info.msg_info->data_flag |= kNullDataFlagBit;
    const auto empty_tensor_desc = GeTensorDesc(GeShape({0}));
    const auto empty_tensor = GeTensor(empty_tensor_desc, nullptr, 0U);
    for (size_t i = 0U; i < input_queue_attrs_.size(); ++i) {
      const auto &queue_attr = input_queue_attrs_[i];
      RuntimeTensorDesc runtime_tensor_desc = {};
      GE_CHK_STATUS_RET(DataFlowExecutorUtils::FillRuntimeTensorDesc(empty_tensor_desc, runtime_tensor_desc));
      const ExchangeService::BuffInfo tensor_desc_buff = {.addr = &runtime_tensor_desc,
                                                          .len = sizeof(RuntimeTensorDesc)};
      const std::vector<ExchangeService::BuffInfo> buffs = {tensor_desc_buff};
      GE_CHK_STATUS_RET(exchange_service_->Enqueue(queue_attr.device_id,
                                                   queue_attr.queue_id,
                                                   buffs,
                                                   control_info),
                        "Failed to enqueue input[%zu], model id=%u, queue id=%u",
                        i, model_id_, queue_attr.queue_id);
      GELOGD("Enqueue empty eos input[%zu] successfully, model id=%u, queue id=%u.",
             i, model_id_, queue_attr.queue_id);
    }
  } else {
    GELOGE(FAILED, "Inputs not eos data.");
    return FAILED;
  }
  return SUCCESS;
}

Status HeterogeneousModelExecutor::GetRedeployStatus() {
  if (deploy_state_.load() == kCallbackStartRedeploy) {
    GELOGW("AbnormalStatusMonitor, it is redeploying, please wait");
    return ACL_ERROR_GE_REDEPLOYING;
  }
  if (deploy_state_.load() == kCallbackFailedRedeploy) {
    GELOGE(FAILED, "AbnormalStatusMonitor, abnormal status can't recovery by redeploying");
    return FAILED;
  }
  return SUCCESS;
}

Status HeterogeneousModelExecutor::FeedRawData(const std::vector<RawData> &raw_data_list,
                                               uint32_t index, const DataFlowInfo &info,
                                               int32_t timeout) {
  GE_CHK_STATUS_RET_NOLOG(GetRedeployStatus());
  GE_CHK_BOOL_RET_STATUS(timeout >= kQueueNeverTimeout, FAILED, "timeout is invalid, must be >=%d, timeout=%d",
                         kQueueNeverTimeout, timeout);
  GELOGD("Start to feed raw data for index:%u, raw list size:%zu, timeout:%d", index, raw_data_list.size(), timeout);
  ExchangeService::MsgInfo msg_info{};
  msg_info.msg_type = kRawDataMsg;
  DataFlowInfoUtils::InitMsgInfoByDataFlowInfo(msg_info, info, contains_n_mapping_node_);
  ExchangeService::ControlInfo control_info{};
  control_info.timeout = timeout;
  control_info.msg_info = &msg_info;
  GE_CHK_STATUS_RET(info.GetUserData(control_info.user_data, kMaxUserDataSize), "Failed to get user data.");
  if (raw_data_list.empty()) {
    GE_CHK_STATUS_RET_NOLOG(FeedEmptyEosData(control_info));
  } else {
    GE_CHK_STATUS_RET(io_helper_.FeedRawData(raw_data_list, index, control_info),
                      "Failed to raw data for index %u failed.", index);
  }
  for (size_t i = 0UL; i < control_input_queue_attrs_.size(); ++i) {
    const int32_t control_value = 0;
    GE_CHK_STATUS_RET(exchange_service_->Enqueue(control_input_queue_attrs_[i].device_id,
        control_input_queue_attrs_[i].queue_id, &control_value, sizeof(control_value), control_info),
        "Failed to enqueue control input[%zu], model id=%u, queue id=%u",
        i, model_id_, control_input_queue_attrs_[i].queue_id);
  }
  return deploy_state_.load() == kCallbackRedeployDone ? ACL_ERROR_GE_SUBHEALTHY : SUCCESS;
}

Status HeterogeneousModelExecutor::FeedData(const std::vector<uint32_t> &indexes, const std::vector<GeTensor> &inputs,
                                            const DataFlowInfo &info, int32_t timeout) {
  GE_CHK_STATUS_RET_NOLOG(GetRedeployStatus());
  GE_CHK_BOOL_RET_STATUS(timeout >= kQueueNeverTimeout, FAILED, "timeout is invalid, must be >=%d, timeout=%d",
                         kQueueNeverTimeout, timeout);
  GELOGD("Start to feed data, size:%zu, timeout:%d", inputs.size(), timeout);
  ExchangeService::MsgInfo msg_info{};
  DataFlowInfoUtils::InitMsgInfoByDataFlowInfo(msg_info, info, contains_n_mapping_node_);
  ExchangeService::ControlInfo control_info{};
  control_info.timeout = timeout;
  control_info.msg_info = &msg_info;
  GE_CHK_STATUS_RET(info.GetUserData(control_info.user_data, kMaxUserDataSize), "Failed to get user data.");
  if (inputs.empty() && (!input_queue_attrs_.empty())) {
    GE_CHK_STATUS_RET_NOLOG(FeedEmptyEosData(control_info));
  } else {
    std::vector<uint32_t> feed_indexes(indexes);
    if (feed_indexes.empty()) {
      feed_indexes.resize(input_queue_attrs_.size());
      // generate feed indexes from 0
      std::iota(feed_indexes.begin(), feed_indexes.end(), 0U);
    }
    // empty input is used for feed eos data
    GE_ASSERT_TRUE(feed_indexes.size() == inputs.size(), "feed indexes size:%zu != inputs size:%zu.",
                   feed_indexes.size(), inputs.size());
    std::map<DeployQueueAttr, std::vector<GeTensor>> fusion_inputs;
    std::map<size_t, size_t> enqueue_indexes;
    for (size_t idx = 0U; idx < feed_indexes.size(); ++idx) {
      const auto &input = inputs[idx];
      const uint32_t i = feed_indexes[idx];
      GE_CHK_BOOL_RET_STATUS((i < input_queue_attrs_.size()), FAILED,
          "idx must be less than input num, idx=%zu, input num=%zu.", idx, input_queue_attrs_.size());
      const auto &queue_attr = input_queue_attrs_[i];
      if (queue_attr.queue_id == UINT32_MAX) {
        // dummy queue for input
        GELOGD("Input %u is not used after prune dataflow graph.", i);
        continue;
      }
      if (is_fusion_input_[i]) {
        fusion_inputs[queue_attr].emplace_back(input);
        continue;
      }
      enqueue_indexes[idx] = static_cast<size_t>(i);
    }
    GE_CHK_STATUS_RET(io_helper_.Feed(enqueue_indexes, inputs, control_info), "Failed to enqueue inputs.");
    GE_CHK_STATUS_RET(EnqueueFusionInputs(fusion_inputs, control_info), "Failed to enqueue fusion inputs.");
  }
  for (size_t i = 0UL; i < control_input_queue_attrs_.size(); ++i) {
    const int32_t control_value = 0;
    GE_CHK_STATUS_RET(exchange_service_->Enqueue(control_input_queue_attrs_[i].device_id,
        control_input_queue_attrs_[i].queue_id, &control_value, sizeof(control_value), control_info),
        "Failed to enqueue control input[%zu], model id=%u, queue id=%u",
        i, model_id_, control_input_queue_attrs_[i].queue_id);
  }

  return deploy_state_.load() == kCallbackRedeployDone ? ACL_ERROR_GE_SUBHEALTHY : SUCCESS;
}

Status HeterogeneousModelExecutor::FeedFlowMsg(const std::vector<uint32_t> &indexes,
                                               const std::vector<FlowMsgPtr> &inputs,
                                               int32_t timeout) {
  GE_CHK_STATUS_RET_NOLOG(GetRedeployStatus());
  GE_CHK_BOOL_RET_STATUS(timeout >= kQueueNeverTimeout, FAILED, "timeout is invalid, must be >=%d, timeout=%d",
                         kQueueNeverTimeout, timeout);
  GELOGD("Start to feed data, size:%zu, timeout:%d", inputs.size(), timeout);
  std::vector<uint32_t> feed_indexes(indexes);
  if (feed_indexes.empty()) {
    feed_indexes.resize(input_queue_attrs_.size());
    // generate feed indexes from 0
    std::iota(feed_indexes.begin(), feed_indexes.end(), 0U);
  }

  GE_ASSERT_TRUE(feed_indexes.size() == inputs.size(), "feed indexes size:%zu != inputs size:%zu.",
                  feed_indexes.size(), inputs.size());
  std::map<size_t, size_t> enqueue_indexes;
  std::vector<FlowMsgBasePtr> feed_inputs;
  for (size_t idx = 0U; idx < feed_indexes.size(); ++idx) {
    const uint32_t i = feed_indexes[idx];
    GE_CHK_BOOL_RET_STATUS((i < input_queue_attrs_.size()), FAILED,
        "idx must be less than input num, idx=%zu, input num=%zu.", idx, input_queue_attrs_.size());
    auto feed_input = std::dynamic_pointer_cast<FlowMsgBase>(inputs[idx]);
    GE_CHECK_NOTNULL(feed_input);
    feed_inputs.emplace_back(feed_input);
    const auto &queue_attr = input_queue_attrs_[i];
    if (queue_attr.queue_id == UINT32_MAX) {
      // dummy queue for input
      GELOGD("Input %u is not used after prune dataflow graph.", i);
      continue;
    }
    enqueue_indexes[idx] = static_cast<size_t>(i);
  }
  ExchangeService::ControlInfo control_info{};
  control_info.timeout = timeout;
  GE_CHK_STATUS_RET(io_helper_.FeedFlowMsg(enqueue_indexes, feed_inputs, control_info), "Failed to feed flow msg.");

  for (size_t i = 0UL; i < control_input_queue_attrs_.size(); ++i) {
    const int32_t control_value = 0;
    GE_CHK_STATUS_RET(exchange_service_->Enqueue(control_input_queue_attrs_[i].device_id,
        control_input_queue_attrs_[i].queue_id, &control_value, sizeof(control_value), control_info),
        "Failed to enqueue control input[%zu], model id=%u, queue id=%u",
        i, model_id_, control_input_queue_attrs_[i].queue_id);
  }
  return deploy_state_.load() == kCallbackRedeployDone ? ACL_ERROR_GE_SUBHEALTHY : SUCCESS;
}

void HeterogeneousModelExecutor::DynamicSchedInfoClear() {
  const std::lock_guard<std::mutex> lk(queue_status_mu_);
  queue_status_info_.clear();
  GEEVENT("DynamicSched, scheding data: Total(us)=%" PRIu64 ", Cnt=%" PRIu64 ", Per duration(ns)=%" PRIu64 ", "
      "Max duration(ns)=%" PRIu64 ", Greater 100us cnt=%" PRIu64, duration_total_ / kMicrosecondToNanosecond, cnt_total_,
      (duration_total_ / (cnt_total_ != 0ULL ? cnt_total_ : 1ULL)), duration_max_, duration_size_);
  duration_total_ = 0ULL;
  cnt_total_ = 0ULL;
  duration_max_ = 0ULL;
  duration_size_ = 0ULL;
  call_ = 0ULL;
  return;
}

Status HeterogeneousModelExecutor::FetchFlowMsg(const std::vector<uint32_t> &indexes,
                                                std::vector<FlowMsgPtr> &outputs,
                                                int32_t timeout) {
  GE_CHK_STATUS_RET_NOLOG(GetRedeployStatus());
  GE_CHK_BOOL_RET_STATUS(timeout >= kQueueNeverTimeout, FAILED, "timeout:%d is invalid, must be >=%d",
                         timeout, kQueueNeverTimeout);
  std::vector<uint32_t> msg_indexs(indexes);
  if (msg_indexs.empty()) {
    msg_indexs.resize(output_queue_attrs_.size());
    // generate fetch indexes from 0
    std::iota(msg_indexs.begin(), msg_indexs.end(), 0U);
  } else {
    std::set<uint32_t> index_set(indexes.cbegin(), indexes.cend());
    GE_ASSERT_TRUE(index_set.size() == indexes.size(), "Fetch indexes cannot have same element, indexes=%s.",
                   ToString(indexes).c_str());
  }
  GELOGD("Start to fetch flow msg, timeout=%dms, fetch index=%s.", timeout, ToString(msg_indexs).c_str());
  size_t end_of_sequence_num = 0UL;
  Status status = SUCCESS;
  ExchangeService::ControlInfo control_info{};
  control_info.timeout = timeout;
  for (const uint32_t idx : msg_indexs) {
    GE_CHK_BOOL_RET_STATUS((idx < output_queue_attrs_.size()), FAILED,
        "idx must be less than output num, idx=%u, output num=%zu.", idx, output_queue_attrs_.size());
    // Dequeue tensor
    const auto queue_id = output_queue_attrs_[idx].queue_id;
    FlowMsgBasePtr flow_msg = nullptr;
    const auto deq_ret = DoDequeue(flow_msg, control_info, idx);
    if (deq_ret != SUCCESS && deq_ret != ACL_ERROR_GE_REDEPLOYING) {
      status = deq_ret;
      GELOGE(status, "Dequeue output[%u] Failed, model id=%u, queue id=%u", idx, model_id_, queue_id);
      continue;
    }
    if (flow_msg != nullptr) {
      GELOGD("Dequeue output[%u] successfully, model id=%u, queue id=%u", idx, model_id_, queue_id);
      if (flow_msg->IsEndOfSequence()) {
        end_of_sequence_num++;
        GELOGD("Output[%u] is end of sequence, model id=%u, queue id=%u", idx, model_id_, queue_id);
      }
      outputs.emplace_back(flow_msg);
      }
  }

  GE_CHK_STATUS_RET(DequeueControlOutputs(replica_num_, control_info.timeout), "Dequeue control outputs failed.");
  GE_CHK_BOOL_RET_SPECIAL_STATUS(end_of_sequence_num > 0U, END_OF_SEQUENCE, "return end of sequence.");

  return (status == SUCCESS && deploy_state_.load() == kCallbackRedeployDone) ? ACL_ERROR_GE_SUBHEALTHY : status;
}

Status HeterogeneousModelExecutor::FetchData(const std::vector<uint32_t> &indexes, std::vector<GeTensor> &outputs,
                                             DataFlowInfo &info, int32_t timeout) {
  GE_CHK_STATUS_RET_NOLOG(GetRedeployStatus());
  GE_CHK_BOOL_RET_STATUS(timeout >= kQueueNeverTimeout, FAILED, "timeout:%d is invalid, must be >=%d",
                         timeout, kQueueNeverTimeout);
  std::vector<uint32_t> fetch_indexs(indexes);
  if (fetch_indexs.empty()) {
    fetch_indexs.resize(output_queue_attrs_.size());
    // generate fetch indexes from 0
    std::iota(fetch_indexs.begin(), fetch_indexs.end(), 0U);
  } else {
    std::set<uint32_t> index_set(indexes.cbegin(), indexes.cend());
    GE_ASSERT_TRUE(index_set.size() == indexes.size(), "Fetch indexes cannot have same element, indexes=%s.",
                   ToString(indexes).c_str());
  }
  GELOGD("Start to fetch data, timeout=%dms, fetch index=%s.", timeout, ToString(fetch_indexs).c_str());
  if (align_attrs_.align_max_cache_num > 0) {
    return AlignFetchData(fetch_indexs, outputs, info, timeout);
  }
  size_t end_of_sequence_num = 0UL;
  Status status = SUCCESS;
  ExchangeService::MsgInfo msg_info{};
  bool set_data_flow_info = false;
  ExchangeService::ControlInfo control_info{};
  control_info.timeout = timeout;
  control_info.msg_info = &msg_info;
  for (const uint32_t idx : fetch_indexs) {
    GE_CHK_BOOL_RET_STATUS((idx < output_queue_attrs_.size()), FAILED,
        "idx must be less than output num, idx=%u, output num=%zu.", idx, output_queue_attrs_.size());
    // Dequeue tensor
    const auto queue_id = output_queue_attrs_[idx].queue_id;
    GeTensor output_tensor(*(output_tensor_desc_[idx]));
    control_info.end_of_sequence_flag = false;
    std::shared_ptr<AlignedPtr> aligned_ptr = nullptr;
    const auto deq_ret = DoDequeue(output_tensor, aligned_ptr, control_info, idx);
    if (deq_ret != SUCCESS && deq_ret != ACL_ERROR_GE_REDEPLOYING) {
      status = deq_ret;
      GELOGE(status, "Dequeue output[%u] Failed, model id=%u, queue id=%u", idx, model_id_, queue_id);
      continue;
    }
    status = ((deq_ret == ACL_ERROR_GE_REDEPLOYING) ? deq_ret : status);
    GELOGD("Dequeue output[%u] successfully, model id=%u, queue id=%u", idx, model_id_, queue_id);
    if (control_info.end_of_sequence_flag) {
      end_of_sequence_num++;
      GELOGD("Output[%u] is end of sequence, model id=%u, queue id=%u", idx, model_id_, queue_id);
      continue;
    }
    if ((control_info.msg_info->data_flag & kNullDataFlagBit) == 0U) {
      // not null data
      outputs.emplace_back(std::move(output_tensor));
    }
    if (!set_data_flow_info) {
      GE_CHK_STATUS_RET(info.SetUserData(control_info.user_data, kMaxUserDataSize), "Failed to set user data.");
      DataFlowInfoUtils::InitDataFlowInfoByMsgInfo(info, msg_info);
      set_data_flow_info = true;
    }
  }

  GE_CHK_STATUS_RET(DequeueControlOutputs(replica_num_, control_info.timeout), "Dequeue control outputs failed.");
  GE_CHK_BOOL_RET_SPECIAL_STATUS(end_of_sequence_num > 0U, END_OF_SEQUENCE, "return end of sequence.");

  return (status == SUCCESS && deploy_state_.load() == kCallbackRedeployDone) ? ACL_ERROR_GE_SUBHEALTHY : status;
}

Status HeterogeneousModelExecutor::AlignFetchData(const std::vector<uint32_t> &fetch_indexes,
                                                  std::vector<GeTensor> &outputs, DataFlowInfo &info,
                                                  const int32_t timeout) {
  std::shared_ptr<DataFlowDataAligner> data_aligner;
  GE_CHK_STATUS_RET(GetOrCreateDataAligner(fetch_indexes, data_aligner),
                    "Failed to get or create data aligner for fetch indexes=%s", ToString(fetch_indexes).c_str());

  ExchangeService::MsgInfo msg_info{};
  ExchangeService::ControlInfo control_info{};
  control_info.timeout = kRetryInterval;
  control_info.msg_info = &msg_info;
  auto start_time = std::chrono::steady_clock::now();
  Status status = SUCCESS;
  while (true) {
    if (exception_handler_.TakeWaitModelIoException(info)) {
      GELOGW("raise exception by user");
      return ACL_ERROR_GE_USER_RAISE_EXCEPTION;
    }
    bool has_output = false;
    GE_CHK_STATUS_RET(data_aligner->TryTakeExpiredOrOverLimitData(outputs, info, has_output),
                      "take expire or over limit data failed");
    if (has_output) {
      GELOGW("aligner has expired or over limit data");
      return ACL_ERROR_GE_DATA_NOT_ALIGNED;
    }
    uint32_t queue_idx = data_aligner->SelectNextQueueIdx();
    const auto queue_id = output_queue_attrs_[queue_idx].queue_id;
    GELOGD("next fetch queue index=%u, queue id=%u", queue_idx, queue_id);
    GeTensor output_tensor(*(output_tensor_desc_[queue_idx]));
    std::shared_ptr<AlignedPtr> aligned_ptr = nullptr;
    status = DoDequeueOnce(output_tensor, aligned_ptr, control_info, queue_idx);
    if (status == RT_ERROR_TO_GE_STATUS(ACL_ERROR_RT_QUEUE_EMPTY)) {
      if (timeout == -1) {
        continue;
      }
      auto current_time = std::chrono::steady_clock::now();
      int64_t elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
      if (elapsed_time >= timeout) {
        GELOGE(status, "Dequeue output[%u] timeout, model id=%u, queue id=%u, "
               "timeout=%d ms, elapsed_time=%" PRId64 " ms",
               queue_idx, model_id_, queue_id, timeout, elapsed_time);
        break;
      }
      continue;
    }

    if (status == ACL_ERROR_GE_REDEPLOYING) {
      break;
    }

    // ret code error save to cache.
    if ((status != SUCCESS) && (status != static_cast<uint32_t>(msg_info.ret_code))) {
      GELOGE(status, "Dequeue output[%u] Failed, model id=%u, queue id=%u", queue_idx, model_id_, queue_id);
      break;
    }
    GELOGD("Dequeue output[%u] end, model id=%u, queue id=%u", queue_idx, model_id_, queue_id);

    TensorWithHeader tensor_with_header{};
    tensor_with_header.tensor = std::move(output_tensor);
    tensor_with_header.msg_info = *control_info.msg_info;
    auto mem_ret = memcpy_s(tensor_with_header.user_data, kMaxUserDataSize, control_info.user_data, kMaxUserDataSize);
    GE_ASSERT_EOK(mem_ret, "copy user data failed, copy size=%zu", kMaxUserDataSize);
    bool is_aligned = false;
    GE_CHK_STATUS_RET(
        data_aligner->PushAndAlignData(queue_idx, std::move(tensor_with_header), outputs, info, is_aligned),
        "push and align data return failed, queue_idx=%u.", queue_idx);
    // align return output or null data.
    if (is_aligned) {
      break;
    }
    // clear ret code
    msg_info.ret_code = 0;
  }
  return (status == SUCCESS && deploy_state_.load() == kCallbackRedeployDone) ? ACL_ERROR_GE_SUBHEALTHY : status;
}

Status HeterogeneousModelExecutor::GetOrCreateDataAligner(const std::vector<uint32_t> &fetch_indexes,
                                                          std::shared_ptr<DataFlowDataAligner> &data_aligner) {
  std::lock_guard<std::mutex> lk(data_aligner_mu_);
  auto iter = data_aligner_map_.find(fetch_indexes);
  if (iter != data_aligner_map_.end()) {
    data_aligner = iter->second;
    return SUCCESS;
  }
  for (uint32_t fetch_idx : fetch_indexes) {
    GE_CHK_BOOL_RET_STATUS((fetch_idx < output_queue_attrs_.size()), FAILED,
                           "idx must be less than output num, idx=%u, output num=%zu.", fetch_idx,
                           output_queue_attrs_.size());
    const auto tmp = input_to_data_aligner_map_.find(fetch_idx);
    GE_CHK_BOOL_RET_STATUS(tmp == input_to_data_aligner_map_.end(), FAILED,
                           "fetch idx[%s] is conflicted with other aligner[%s].", ToString(fetch_indexes).c_str(),
                           ToString(tmp->second->GetQueueIdxes()).c_str());
  }

  data_aligner = MakeShared<DataFlowDataAligner>(fetch_indexes, align_attrs_, [this](uint64_t trans_id) -> bool {
    return exception_handler_.IsModelIoIgnoreTransId(trans_id);
  });
  GE_ASSERT_NOTNULL(data_aligner);
  data_aligner_map_[fetch_indexes] = data_aligner;
  for (uint32_t fetch_idx : fetch_indexes) {
    input_to_data_aligner_map_[fetch_idx] = data_aligner;
  }
  exception_handler_.RegisterModelIoExpTransIdCallback([data_aligner](uint64_t trans_id, uint32_t type) {
    if (type == kExceptionTypeOccured) {
      data_aligner->ClearCacheByTransId(trans_id);
    }
  });
  return SUCCESS;
}
}  // namespace ge
