/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "heterogeneous_model_io_helper.h"
#include "framework/common/ge_types.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "dflow/base/deploy/exchange_service.h"
#include "graph/utils/tensor_adapter.h"
#include "data_flow_executor_utils.h"
#include "acl/acl.h"
#include "common/df_chk.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
constexpr size_t kDefaultThreadNum = 12U;
}

HeterogeneousModelIoHelper::HeterogeneousModelIoHelper(
    const std::vector<DeployQueueAttr> &input_queue_attrs,
    const std::vector<std::vector<DeployQueueAttr>> &broadcast_input_queue_attrs)
    : input_queue_attrs_(input_queue_attrs),
      broadcast_input_queue_attrs_(broadcast_input_queue_attrs) {}

Status HeterogeneousModelIoHelper::Initialize() {
  size_t thread_num = 0U;
  for (size_t i = 0; i < input_queue_attrs_.size(); ++i) {
    if ((broadcast_input_queue_attrs_.size() >= i + 1U) && broadcast_input_queue_attrs_[i].size() > 0U) {
      thread_num += broadcast_input_queue_attrs_[i].size();
    }
  }

  if (thread_num > 1U) {
    thread_num = thread_num > kDefaultThreadNum ? kDefaultThreadNum : thread_num;
    pool_ = MakeUnique<ThreadPool>("ge_dpl_ioeq", thread_num, false);
    GE_CHECK_NOTNULL(pool_);
    GELOGI("Create thread pool success, thread num = %zu.", thread_num);
  }

  const auto execution_runtime = ExecutionRuntime::GetInstance();
  GE_CHECK_NOTNULL(execution_runtime);
  exchange_service_ = &execution_runtime->GetExchangeService();
  GE_CHECK_NOTNULL(exchange_service_);
  return SUCCESS;
}

Status HeterogeneousModelIoHelper::ExecuteEnqueueTask(const EnqueueTask &enqueue_task,
                                                      const DeployQueueAttr &queue_attr,
                                                      std::vector<std::future<Status>> &fut_rets,
                                                      bool execute_parallel) {
  if (pool_ != nullptr && execute_parallel) {
    auto fut = pool_->commit(enqueue_task, queue_attr);
    fut_rets.emplace_back(std::move(fut));
  } else {
    return enqueue_task(queue_attr);
  }
  return SUCCESS;
}

Status HeterogeneousModelIoHelper::FillBuffInfos(const GeTensor &tensor,
                                                 RuntimeTensorDesc &tensor_desc,
                                                 std::vector<ExchangeService::BuffInfo> &buffs) {
  ExchangeService::BuffInfo desc = {};
  GE_CHK_STATUS_RET(DataFlowExecutorUtils::FillRuntimeTensorDesc(tensor.GetTensorDesc(), tensor_desc, false),
                    "Failed to fill runtime tensor desc");
  tensor_desc.data_size = static_cast<uint64_t>(tensor.GetData().GetSize());
  desc.len = sizeof(tensor_desc);
  desc.addr = &tensor_desc;
  buffs.emplace_back(desc);
  ExchangeService::BuffInfo data = {};
  data.addr = ValueToPtr(PtrToValue(tensor.GetData().GetData()));
  data.len = tensor.GetData().GetSize();
  buffs.emplace_back(data);
  return SUCCESS;
}

Status HeterogeneousModelIoHelper::Feed(const std::map<size_t, size_t> &indexes,
                                        const std::vector<GeTensor> &inputs,
                                        const ExchangeService::ControlInfo &control_info) {
  std::map<size_t, std::vector<size_t>> input_idx_to_tensor_list_idx;
  Status ret = SUCCESS;
  {
    std::vector<std::future<Status>> fut_rets;
    // make sure all thread execute end
    GE_MAKE_GUARD(future_ret, ([&fut_rets, &ret]() {
      for (auto &fut : fut_rets) {
        auto fut_ret = fut.get();
        if (fut_ret != SUCCESS) {
          ret = fut_ret;
        }
      }
    }));

    // input_idx: input index; tensor_list_idx:feed tensor index
    for (const auto &it : indexes) {
      input_idx_to_tensor_list_idx[it.second].emplace_back(it.first);
    }
    for (const auto &it : input_idx_to_tensor_list_idx) {
      const auto &tensor_list_idx = it.second;
      const size_t i = it.first; // input index
      GE_CHK_BOOL_RET_STATUS((i < input_queue_attrs_.size()),
                            FAILED,
                            "idx must be less than input num, idx=%zu, "
                            "input queue attr size = %zu, broadcast input attr size = %zu.",
                            i, input_queue_attrs_.size(), broadcast_input_queue_attrs_.size());
      EnqueueTask enqueue_task = [this, &tensor_list_idx, &inputs, &control_info, i](
          const DeployQueueAttr &queue_attr) -> Status {
        DF_CHK_ACL_RET(aclrtSetDevice(queue_attr.device_id));
        for (const size_t tensor_index : tensor_list_idx) {
          const auto &input = inputs[tensor_index];
          std::vector<ExchangeService::BuffInfo> buffs; // tensor desc and data
          RuntimeTensorDesc runtime_tensor_desc{};
          GE_CHK_STATUS_RET(FillBuffInfos(input, runtime_tensor_desc, buffs),
                            "Failed to fill buff infos from tensor");
          GE_CHK_STATUS_RET(exchange_service_->Enqueue(queue_attr.device_id, queue_attr.queue_id, buffs, control_info),
                            "Failed to enqueue input, queue id=%u", queue_attr.queue_id);
          GELOGI("Enqueue input[%zu] successfully, queue attr=[%s]", i, queue_attr.DebugString().c_str());
        }
        return SUCCESS;
      };
      if ((broadcast_input_queue_attrs_.size() >= i + 1U) && (!broadcast_input_queue_attrs_[i].empty())) {
        for (const auto &broadcast_input : broadcast_input_queue_attrs_[i]) {
          GE_CHK_STATUS_RET(ExecuteEnqueueTask(enqueue_task, broadcast_input, fut_rets, true),
                            "Failed to execute enqueue task, input index = %zu", i);
        }
      } else {
        GE_CHK_STATUS_RET(ExecuteEnqueueTask(enqueue_task, input_queue_attrs_[i], fut_rets),
                          "Failed to execute enqueue task, input index = %zu", i);
      }
    }
  }
  GE_CHK_STATUS(ret, "Failed to execute multi-thread enqueue task.");
  return ret;
}

Status HeterogeneousModelIoHelper::FeedRawData(const std::vector<RawData> &raw_data_list, const uint32_t index,
                                               const ExchangeService::ControlInfo &control_info) {
  std::map<size_t, std::vector<size_t>> input_idx_to_tensor_list_idx;
  Status ret = SUCCESS;
  {
    std::vector<std::future<Status>> fut_rets;
    // make sure all thread execute end
    GE_MAKE_GUARD(future_ret, ([&fut_rets, &ret]() {
      for (auto &fut : fut_rets) {
        auto fut_ret = fut.get();
        if (fut_ret != SUCCESS) {
          ret = fut_ret;
        }
      }
    }));
    GE_CHK_BOOL_RET_STATUS((index < input_queue_attrs_.size()), FAILED,
                           "idx must be less than input num, idx=%u, "
                           "input queue attr size = %zu, broadcast input attr size = %zu.",
                           index, input_queue_attrs_.size(), broadcast_input_queue_attrs_.size());
      EnqueueTask enqueue_task = [this, &control_info, &raw_data_list, index](
          const DeployQueueAttr &queue_attr) -> Status {
        std::vector<ExchangeService::BuffInfo> fusion_buffs;
        for (const auto raw_data : raw_data_list) {
          ExchangeService::BuffInfo buff_info = {.addr = const_cast<void *>(raw_data.addr),
                                                 .len = raw_data.len};
          fusion_buffs.push_back(buff_info);
        }
        GE_CHK_STATUS_RET(exchange_service_->Enqueue(queue_attr.device_id, queue_attr.queue_id,
            fusion_buffs, control_info), "Failed to enqueue input, queue id=%u", queue_attr.queue_id);
        GELOGI("Enqueue input[%u] successfully, queue id=%u, size=%zu",
               index, queue_attr.queue_id, fusion_buffs.size());
        return SUCCESS;
      };
      if ((index < broadcast_input_queue_attrs_.size()) && (!broadcast_input_queue_attrs_[index].empty())) {
        for (const auto &broadcast_input : broadcast_input_queue_attrs_[index]) {
          GE_CHK_STATUS_RET(ExecuteEnqueueTask(enqueue_task, broadcast_input, fut_rets, true),
                            "Failed to execute enqueue task, input index = %zu", index);
        }
      } else {
        GE_CHK_STATUS_RET(ExecuteEnqueueTask(enqueue_task, input_queue_attrs_[index], fut_rets),
                          "Failed to execute enqueue task, input index = %zu", index);
      }
  }
  GE_CHK_STATUS(ret, "Failed to execute multi-thread enqueue task.");
  return ret;
}

Status HeterogeneousModelIoHelper::EnqueueFlowMsg(const FlowMsgBasePtr &flow_msg,
                                                  const DeployQueueAttr &queue_attr,
                                                  const ExchangeService::ControlInfo &control_info) const {
  auto mbuf = flow_msg->MbufCopyRef();
  GE_DISMISSABLE_GUARD(mbuf, ([mbuf]() { GE_CHK_RT(rtMbufFree(mbuf)); }));
  GE_CHK_STATUS_RET(exchange_service_->EnqueueMbuf(queue_attr.device_id, queue_attr.queue_id,
                                                   mbuf, control_info.timeout),
                    "Failed to enqueue mbuf flow msg, device_id = %u, queue_id = %u",
                    queue_attr.device_id, queue_attr.queue_id);
  GELOGD("Enqueue flow msg successfully, queue attr=[%s], msg_type = %d",
         queue_attr.DebugString().c_str(), static_cast<int32_t>(flow_msg->GetMsgType()));
  GE_DISMISS_GUARD(mbuf);
  return SUCCESS;
}

Status HeterogeneousModelIoHelper::FeedFlowMsg(const std::map<size_t, size_t> &indexes,
                                               const std::vector<FlowMsgBasePtr> &inputs,
                                               const ExchangeService::ControlInfo &control_info) {
  std::map<size_t, std::vector<size_t>> input_idx_to_msg_list_idx;
  Status ret = SUCCESS;
  {
    std::vector<std::future<Status>> fut_rets;
    // make sure all thread execute end
    GE_MAKE_GUARD(future, ([&fut_rets, &ret]() {
      for (auto &fut : fut_rets) {
        auto fut_ret = fut.get();
        if (fut_ret != SUCCESS) {
          ret = fut_ret;
        }
      }
    }));

    // input_idx: input index; flow_msg_list_idx:feed tensor index
    for (const auto &it : indexes) {
      input_idx_to_msg_list_idx[it.second].emplace_back(it.first);
    }
    for (const auto &it : input_idx_to_msg_list_idx) {
      const auto &msg_list_idx = it.second;
      const size_t i = it.first; // input index
      GE_CHK_BOOL_RET_STATUS((i < input_queue_attrs_.size()),
                            FAILED,
                            "idx must be less than input num, idx=%zu, "
                            "input queue attr size = %zu, broadcast input attr size = %zu.",
                            i, input_queue_attrs_.size(), broadcast_input_queue_attrs_.size());
      EnqueueTask enqueue_task = [this, &msg_list_idx, &inputs, &control_info, i](
          const DeployQueueAttr &queue_attr) -> Status {
        for (const size_t msg_index : msg_list_idx) {
          GE_CHK_STATUS_RET(EnqueueFlowMsg(inputs[msg_index], queue_attr, control_info),
                            "Failed to enqueue input[%zu] flow msg, device_id = %u, queue id=%u",
                            i, queue_attr.device_id, queue_attr.queue_id);
          GELOGI("Enqueue input[%zu] successfully, device_id = %u, queue id=%u",
                 i, queue_attr.device_id, queue_attr.queue_id);
        }
        return SUCCESS;
      };
      if ((broadcast_input_queue_attrs_.size() >= i + 1U) && (!broadcast_input_queue_attrs_[i].empty())) {
        for (const auto &broadcast_input : broadcast_input_queue_attrs_[i]) {
          GE_CHK_STATUS_RET(ExecuteEnqueueTask(enqueue_task, broadcast_input, fut_rets, true),
                            "Failed to execute enqueue flow msg task, input index = %zu", i);
        }
      } else {
        GE_CHK_STATUS_RET(ExecuteEnqueueTask(enqueue_task, input_queue_attrs_[i], fut_rets),
                          "Failed to execute enqueue flow msg task, input index = %zu", i);
      }
    }
  }
  GE_CHK_STATUS(ret, "Failed to execute multi-thread enqueue task.");
  return ret;
}

Status HeterogeneousModelIoHelper::FetchFlowMsg(const DeployQueueAttr &queue_attr,
                                                const ExchangeService::ControlInfo &control_info,
                                                const GeTensorDescPtr &output_desc,
                                                FlowMsgBasePtr &flow_msg) const {
  rtMbufPtr_t mbuf = nullptr;
  // queue empty is normal, can not print error
  GE_CHK_STATUS_RET_NOLOG(exchange_service_->DequeueMbuf(queue_attr.device_id, queue_attr.queue_id,
                                                         &mbuf, control_info.timeout));
  GE_DISMISSABLE_GUARD(mbuf, ([mbuf]() { GE_CHK_RT(rtMbufFree(mbuf)); }));
  MsgType msg_type;
  bool is_null_data = false;
  GE_CHK_STATUS_RET(FlowMsgBase::GetMsgType(mbuf, msg_type, is_null_data), "Failed to get mbuf msg type");
  if (msg_type == MsgType::MSG_TYPE_TENSOR_DATA) {
    if (is_null_data) {
      auto null_flow_msg = MakeShared<EmptyDataFlowMsg>();
      GE_CHECK_NOTNULL(null_flow_msg);
      GE_CHK_STATUS_RET(null_flow_msg->BuildNullData(mbuf), "Failed to build null data");
      flow_msg = null_flow_msg;
    } else {
      auto tensor_flow_msg = MakeShared<TensorFlowMsg>();
      GE_CHECK_NOTNULL(tensor_flow_msg);
      GE_CHK_STATUS_RET(tensor_flow_msg->BuildTensor(mbuf, *output_desc), "Failed to build tensor");
      flow_msg = tensor_flow_msg;
    }
  } else {
    auto raw_data_flow_msg = MakeShared<RawDataFlowMsg>();
    GE_CHECK_NOTNULL(raw_data_flow_msg);
    GE_CHK_STATUS_RET(raw_data_flow_msg->BuildRawData(mbuf), "Failed to build raw data");
    flow_msg = raw_data_flow_msg;
  }
  GELOGD("Fetch flow msg successfully, queue attr=[%s], msg_type = %d",
         queue_attr.DebugString().c_str(), static_cast<int32_t>(flow_msg->GetMsgType()));
  // mbuf will be freed by consumer
  GE_DISMISS_GUARD(mbuf);
  return SUCCESS;
}
}  // namespace ge
