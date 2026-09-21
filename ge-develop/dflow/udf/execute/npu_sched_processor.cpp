/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "npu_sched_processor.h"
#include "mmpa/mmpa_api.h"
#include "common/udf_log.h"
#include "config/global_config.h"

namespace FlowFunc {
namespace {
constexpr const char *kSoNameNpuSchedModelLoader = "libnpu_sched_model_loader.so";
constexpr const char *kFuncNameInitializeNpuSched = "InitializeNpuSched";
constexpr const char *kFuncNameCreateNpuSchedModelHandler = "CreateNpuSchedModelHandler";
constexpr const char *kFuncNameLoadNpuSchedModel = "LoadNpuSchedModel";
constexpr const char *kFuncNameUnloadNpuSchedModel = "UnloadNpuSchedModel";
constexpr const char *kFuncNameDestroyNpuSchedModelHandler = "DestroyNpuSchedModelHandler";
constexpr const char *kFuncNameFinalizeNpuSched = "FinalizeNpuSched";

using FuncInitializeNpuSched = uint32_t(*)(int32_t device_id);
using FuncCreateNpuSchedModelHandler = NpuSchedModelHandler(*)();
using FuncLoadNpuSchedModel = uint32_t(*)(NpuSchedModelHandler handler, NpuSchedLoadParam *load_param);
using FuncUnloadNpuSchedModel = uint32_t(*)(NpuSchedModelHandler handler);
using FuncDestroyNpuSchedModelHandler = uint32_t(*)(NpuSchedModelHandler handler);
using FuncFinalizeNpuSched = uint32_t(*)(int32_t device_id);

void ConvertToQueueAttrs(const QueueDevInfo &queue_dev_info, ge::QueueAttrs &queue_attr) {
    queue_attr.queue_id = queue_dev_info.queue_id;
    queue_attr.device_type = queue_dev_info.device_type;
    // just run on host
    queue_attr.device_id = queue_dev_info.device_id;
    queue_attr.logic_id = queue_dev_info.logic_queue_id;
}

void ConvertInputAlignAttrs(const InputAlignAttrs &input_align_attrs, ge::InputAlignAttrs &ge_input_align_attrs) {
    ge_input_align_attrs.align_max_cache_num = input_align_attrs.align_max_cache_num;
    ge_input_align_attrs.align_timeout = input_align_attrs.align_timeout;
    ge_input_align_attrs.drop_when_not_align = input_align_attrs.drop_when_not_align;
}
}

NpuSchedProcessor::~NpuSchedProcessor() {
    Finalize();
}

template<typename FUNC>
FUNC NpuSchedProcessor::GetFunc(const char *func_name) const {
    auto func = reinterpret_cast<FUNC>(mmDlsym(so_handle_, func_name));
    if (func == nullptr) {
        const char *error = mmDlerror();
        error = (error == nullptr) ? "" : error;
        UDF_LOG_ERROR("dlsym failed, lib=%s, func_name=%s, error msg=%s.", kSoNameNpuSchedModelLoader,
            func_name, error);
        return nullptr;
    }
    return func;
}

int32_t NpuSchedProcessor::Initialize(int32_t device_id) {
    constexpr auto mode = static_cast<int32_t>(MMPA_RTLD_NOW);
    so_handle_ = mmDlopen(kSoNameNpuSchedModelLoader, mode);
    if (so_handle_ == nullptr) {
        const char *error = mmDlerror();
        error = (error == nullptr) ? "" : error;
        UDF_LOG_ERROR("load lib failed, lib=%s, error msg=%s.", kSoNameNpuSchedModelLoader, error);
        return FLOW_FUNC_FAILED;
    }
    auto func_init = GetFunc<FuncInitializeNpuSched>(kFuncNameInitializeNpuSched);
    if (func_init == nullptr) {
        UDF_LOG_ERROR("Get func %s failed.", kFuncNameInitializeNpuSched);
        return FLOW_FUNC_FAILED;
    }
    uint32_t init_ret = func_init(device_id);
    if (init_ret != ge::SUCCESS) {
        UDF_LOG_ERROR("call %s failed, ret=%u.", kFuncNameInitializeNpuSched, init_ret);
        return FLOW_FUNC_FAILED;
    }
    device_id_ = device_id;
    auto phy_device_id = GlobalConfig::Instance().GetPhyDeviceId();
    UDF_LOG_INFO("npu sched processor init success, running device id=%d,  phy_device_id=%d.", device_id, phy_device_id);
    return FLOW_FUNC_SUCCESS;
}

void NpuSchedProcessor::Finalize() {
    UnloadAndDestroyAllModelHandler();
    if (device_id_ != -1) {
        FinalizeNpuSched();
    }

    if (so_handle_ != nullptr) {
        (void)mmDlclose(so_handle_);
        so_handle_ = nullptr;
        UDF_LOG_INFO("close so handle end.");
    }
    device_id_ = -1;
}

void NpuSchedProcessor::UnloadAndDestroyAllModelHandler() {
    if (model_handler_list_.empty()) {
        return;
    }
    auto unload_func = GetFunc<FuncUnloadNpuSchedModel>(kFuncNameUnloadNpuSchedModel);
    auto destroy_func = GetFunc<FuncDestroyNpuSchedModelHandler>(kFuncNameDestroyNpuSchedModelHandler);
    for (const auto &model_handler : model_handler_list_) {
        if (unload_func != nullptr) {
            (void)unload_func(model_handler);
        }
        if (destroy_func != nullptr) {
            auto ret = destroy_func(model_handler);
            if (ret != ge::SUCCESS) {
                UDF_LOG_ERROR("call %s failed, ret=%u.", kFuncNameFinalizeNpuSched, ret);
                // no need return
            }
        }
    }
    UDF_LOG_INFO("Unload and destroy all ModelHandler end, size=%zu.", model_handler_list_.size());
    model_handler_list_.clear();
}

void NpuSchedProcessor::FinalizeNpuSched() const {
    auto func = GetFunc<FuncFinalizeNpuSched>(kFuncNameFinalizeNpuSched);
    if (func == nullptr) {
        return;
    }
    auto ret = func(device_id_);
    if (ret != ge::SUCCESS) {
        UDF_LOG_ERROR("call %s failed, ret=%u.", kFuncNameFinalizeNpuSched, ret);
    } else {
        UDF_LOG_INFO("call %s success.", kFuncNameFinalizeNpuSched);
    }
}

void NpuSchedProcessor::GenerateNpuSchedLoadParam(const FlowFuncModel &flow_func_model,
    NpuSchedLoadParam &npu_sched_load_param) const {
    const auto &flow_func_model_param = flow_func_model.GetFlowFuncModelParam();
    npu_sched_load_param.model_id = flow_func_model_param.model_uuid;
    npu_sched_load_param.device_id = device_id_;

    auto &model_queue_param = npu_sched_load_param.model_queue_param;

    const auto &input_queues = flow_func_model_param.input_queues;
    auto &input_queues_attrs = model_queue_param.input_queues_attrs;
    input_queues_attrs.reserve(input_queues.size());
    for (const auto &input_queue : input_queues) {
        ge::QueueAttrs queue_attr{};
        ConvertToQueueAttrs(input_queue, queue_attr);
        input_queues_attrs.emplace_back(std::move(queue_attr));
        model_queue_param.input_queues.emplace_back(input_queue.queue_id);
    }

    const auto &output_queues = flow_func_model_param.output_queues;
    auto &output_queues_attrs = model_queue_param.output_queues_attrs;
    output_queues_attrs.reserve(output_queues.size());
    for (const auto &output_queue : output_queues) {
        ge::QueueAttrs queue_attr{};
        ConvertToQueueAttrs(output_queue, queue_attr);
        output_queues_attrs.emplace_back(std::move(queue_attr));
        model_queue_param.output_queues.emplace_back(output_queue.queue_id);
    }

    ConvertToQueueAttrs(flow_func_model_param.status_output_queue, model_queue_param.status_output_queue);
    model_queue_param.model_uuid = flow_func_model.GetModelUuid();
    model_queue_param.is_dynamic_sched = flow_func_model_param.is_dynamic_sched;
    model_queue_param.need_report_status = flow_func_model_param.need_report_status;
    model_queue_param.is_head = flow_func_model_param.is_head;
    ConvertInputAlignAttrs(flow_func_model.GetInputAlignAttrs(), model_queue_param.input_align_attrs);
}

int32_t NpuSchedProcessor::LoadNpuSchedModel(const FlowFuncModel &model, ModelQueueInfos &model_queue_infos) {
    auto create_func = GetFunc<FuncCreateNpuSchedModelHandler>(kFuncNameCreateNpuSchedModelHandler);
    if (create_func == nullptr) {
        UDF_LOG_ERROR("Get func %s failed.", kFuncNameCreateNpuSchedModelHandler);
        return FLOW_FUNC_FAILED;
    }
    auto load_func = GetFunc<FuncLoadNpuSchedModel>(kFuncNameLoadNpuSchedModel);
    if (load_func == nullptr) {
        UDF_LOG_ERROR("Get func %s failed.", kFuncNameLoadNpuSchedModel);
        return FLOW_FUNC_FAILED;
    }
    auto model_handler = create_func();
    if (model_handler == nullptr) {
        UDF_LOG_ERROR("call func %s failed.", kFuncNameCreateNpuSchedModelHandler);
        return FLOW_FUNC_FAILED;
    }
    model_handler_list_.emplace_back(model_handler);
    NpuSchedLoadParam load_param{};
    GenerateNpuSchedLoadParam(model, load_param);
    uint32_t load_ret = load_func(model_handler, &load_param);
    if (load_ret != ge::SUCCESS) {
        UDF_LOG_ERROR("call func %s failed.", kFuncNameLoadNpuSchedModel);
        return FLOW_FUNC_FAILED;
    }
    uint32_t req_msg_queue_id = load_param.req_msg_queue_id;
    uint32_t resp_msg_queue_id = load_param.resp_msg_queue_id;
    // runtime can only use running device id, udf use all device id.
    auto phy_device_id = GlobalConfig::Instance().GetPhyDeviceId();
    QueueDevInfo req_msg_queue_info{};
    req_msg_queue_info.queue_id = req_msg_queue_id;
    req_msg_queue_info.device_type = Common::kDeviceTypeNpu;
    req_msg_queue_info.device_id = phy_device_id;
    req_msg_queue_info.deploy_device_id = phy_device_id;
    req_msg_queue_info.is_proxy_queue = true;
    model_queue_infos.input_queues.emplace_back(req_msg_queue_info);
    QueueDevInfo resp_msg_queue_info{};
    resp_msg_queue_info.queue_id = resp_msg_queue_id;
    resp_msg_queue_info.device_type = Common::kDeviceTypeNpu;
    resp_msg_queue_info.device_id = phy_device_id;
    resp_msg_queue_info.deploy_device_id = phy_device_id;
    resp_msg_queue_info.is_proxy_queue = true;
    model_queue_infos.output_queues.emplace_back(resp_msg_queue_info);
    UDF_LOG_INFO("Load npu sched model end, device_id=%d, req_msg_queue_id=%u, resp_msg_queue_id=%u", phy_device_id,
        req_msg_queue_id, resp_msg_queue_id);
    return FLOW_FUNC_SUCCESS;
}
}
