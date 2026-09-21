/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_func_model.h"
#include <fstream>
#include <regex>
#include <set>
#include "common/util.h"
#include "common/udf_log.h"
#include "common/scope_guard.h"
#include "mmpa/mmpa_api.h"
#include "attr_value_impl.h"
#include "common/data_utils.h"
#include "config/global_config.h"

namespace {
constexpr const char *kUdfBuiltInBinName = "libbuilt_in_flowfunc.so";
constexpr const char *kModelEschedProcessPriorityAttrName = "_eschedProcessPriority";
constexpr const char *kModelEschedEventPriorityAttrName = "_eschedEventPriority";
constexpr const char *kModelCpuNumAttrName = "__cpu_num";
constexpr const char *kPageTypeNormal = "normal";
constexpr const char *kPageTypeHuge = "huge";
constexpr const char *kAttrNameDataFlowVisibleDeviceEnable = "_dflow_visible_device_enable";
constexpr const char *kEnvNameAscendRtVisibleDevices  = "ASCEND_RT_VISIBLE_DEVICES";
constexpr const char *kAttrNameDataFlowDataFlowScope = "_dflow_data_flow_scope";
constexpr const char *kAttrNameDataFlowDataFlowInvokeScopes = "_dflow_data_flow_invoked_scopes";
constexpr const int32_t kMaxBufCfgNum = 64;
constexpr const uint32_t kMaxBlkSize = 2 * 1024 * 1024; // 2M
constexpr const uint32_t kNormalPageAtomicValue = 4 * 1024; // 4K
constexpr const uint32_t kHugePageAtomicValue = 2 * 1024 * 1024; // 2M
}
namespace FlowFunc {
FlowFuncModel::FlowFuncModel(FlowFuncModelParam flow_func_model_param)
    : model_param_(std::move(flow_func_model_param)) {}

bool FlowFuncModel::IsFileNameValid(const std::string &file_name) {
    std::regex pattern(R"([A-Za-z0-9.\-_]+)");
    std::smatch match_result;
    return std::regex_match(file_name, match_result, pattern);
}

int32_t FlowFuncModel::LoadUdfModel(ff::udf::UdfModelDef &udf_model_def, std::string &real_model_path) {
    char real_model_path_array[MMPA_MAX_PATH] = {};
    INT32 mm_ret = mmRealPath(model_param_.model_path.c_str(), real_model_path_array, MMPA_MAX_PATH);
    if (mm_ret != EN_OK) {
        UDF_LOG_ERROR("mmRealPath failed, model_path=%s, mm_ret=%d.", model_param_.model_path.c_str(), mm_ret);
        return FLOW_FUNC_FAILED;
    }
    std::ifstream ifs(real_model_path_array, std::ifstream::in | std::ifstream::binary);
    if (!ifs.is_open()) {
        int32_t error_no = errno;
        UDF_LOG_ERROR("open udf def failed, model_path=%s, errno=%d, strerror=%s.", model_param_.model_path.c_str(),
            error_no, GetErrorNoStr(error_no).c_str());
        return FLOW_FUNC_FAILED;
    }
    bool parse_ret = udf_model_def.ParseFromIstream(&ifs);
    if (!parse_ret) {
        UDF_LOG_ERROR("parse udf def failed, model_path=%s.", model_param_.model_path.c_str());
        return FLOW_FUNC_FAILED;
    }
    real_model_path = real_model_path_array;
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncModel::CheckBufCfgItem(const BufferConfigItem& item) {
    if ((item.total_size == 0U) || (item.blk_size == 0U) || (item.max_buf_size == 0U)) {
        UDF_LOG_ERROR("Any of the following configuration items equal to 0 is not allowed. "
            "total size[%u] blk size[%u] or max buf size[%u].", item.total_size, item.blk_size, item.max_buf_size);
        return FLOW_FUNC_FAILED;
    }
    if ((item.total_size <= item.max_buf_size) || (item.max_buf_size < item.blk_size)) {
        UDF_LOG_ERROR("The following three configuration parameters must meet the requirements: "
            "Total size[%u] > max_buf_size[%u] >= blk_size[%u].", item.total_size, item.max_buf_size, item.blk_size);
        return FLOW_FUNC_FAILED;
    }
    if (item.page_type == kPageTypeNormal) {
        if (item.total_size % kNormalPageAtomicValue != 0) {
            UDF_LOG_ERROR("Total size[%u] is expected as an integer multiple of 4 * 1024(4K) for normal page type.",
                          item.total_size);
            return FLOW_FUNC_FAILED;
        }
    } else if(item.page_type == kPageTypeHuge) {
        if (item.total_size % kHugePageAtomicValue != 0) {
            UDF_LOG_ERROR("Total size[%u] is expected as an integer multiple of 2 * 1024 * 1024(2M) "
                          "for huge page type.", item.total_size);
            return FLOW_FUNC_FAILED;
        }
    } else {
        UDF_LOG_ERROR("Page type[%s] is invalid parsed from buf cfg proto.", item.page_type.c_str());
        return FLOW_FUNC_FAILED;
    }
    if ((item.blk_size & (item.blk_size - 1U)) != 0U) {
        UDF_LOG_ERROR("Block size[%lu] is expected to be power of 2.", item.blk_size);
        return FLOW_FUNC_FAILED;
    }
    if (item.blk_size > kMaxBlkSize) {
        UDF_LOG_ERROR("Block size[%u] is expected less than 2M.", item.blk_size);
        return FLOW_FUNC_FAILED;
    }
    if (item.total_size % item.blk_size != 0U) {
        UDF_LOG_ERROR("Total size[%u] is expected as an integer multiple of blk size[%u]",
                      item.total_size, item.blk_size);
        return FLOW_FUNC_FAILED;
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncModel::CheckAndSetBufCfg(const ff::udf::UdfDef &udf_def) {
    if (udf_def.user_buf_cfg_size() == 0) {
        UDF_LOG_INFO("Buf config is not set.");
        return FLOW_FUNC_SUCCESS;
    }
    if (udf_def.user_buf_cfg_size() > kMaxBufCfgNum) {
        UDF_LOG_ERROR("Buf cfg num[%d] is out of limit num[%d].", udf_def.user_buf_cfg_size(), kMaxBufCfgNum);
        return FLOW_FUNC_FAILED;
    }
    uint32_t last_normal_max_buf_size= 0U;
    uint32_t las_huge_max_buf_size = 0U;
    std::vector<BufferConfigItem> buf_cfg;
    buf_cfg.reserve(udf_def.user_buf_cfg_size());
    for (int32_t i = 0; i < udf_def.user_buf_cfg_size(); ++i) {
        const auto &cfg = udf_def.user_buf_cfg(i);
        BufferConfigItem item = {.total_size = cfg.total_size(), .blk_size = cfg.blk_size(),
                                 .max_buf_size = cfg.max_buf_size(), .page_type = cfg.page_type()};
        if(CheckBufCfgItem(item) != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("Buf Cfg is invalid.");
            return FLOW_FUNC_FAILED;
        }
        // check buf cfg item is sorted from small to large by total_size
        if (item.page_type == kPageTypeNormal) {
            if (item.max_buf_size <= last_normal_max_buf_size) {
                UDF_LOG_ERROR("Normal page type buf cfg should be sorted. But current size[%u] is "
                    "less or equal to previous gear size[%u].", item.max_buf_size, last_normal_max_buf_size);
                return FLOW_FUNC_FAILED;
            }
            last_normal_max_buf_size= item.max_buf_size;
        }
        if(item.page_type == kPageTypeHuge) {
            if (item.max_buf_size <= las_huge_max_buf_size) {
                UDF_LOG_ERROR("Huge page type buf cfg should be sorted. But current size[%u] is "
                    "less or equal to previous gear size[%u].", item.max_buf_size, las_huge_max_buf_size);
                return FLOW_FUNC_FAILED;
            }
            las_huge_max_buf_size = item.max_buf_size;
        }
        buf_cfg.emplace_back(item);
    }
    GlobalConfig::Instance().SetBufCfg(buf_cfg);
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncModel::ParseUdfBasicInfos(const ff::udf::UdfDef &udf_def, const std::string &real_model_path) {
    flow_func_name_ = udf_def.func_name();
    name_ = udf_def.name();
    if (model_param_.model_instance_name.empty()) {
        model_param_.model_instance_name = name_;
    }

    const std::string &bin_name = udf_def.bin_name();
    if (!FlowFuncModel::IsFileNameValid(bin_name)) {
        UDF_LOG_ERROR("bin_name[%s] is invalid.", bin_name.c_str());
        return FLOW_FUNC_FAILED;
    }
    if (bin_name == kUdfBuiltInBinName) {
        // reset work path
        work_path_ = "";
        lib_path_ = bin_name;
        UDF_LOG_INFO("%s is built in lib, funcName=%s, name=%s.", bin_name.c_str(),
            flow_func_name_.c_str(), name_.c_str());
    } else {
        if (GlobalConfig::Instance().IsLimitRunBuiltinUdf()) {
            UDF_LOG_ERROR("Current udf only can run builtin udf, funcName=%s, model_path=%s, bin_name=%s.",
                flow_func_name_.c_str(), model_param_.model_path.c_str(), bin_name.c_str());
            return FLOW_FUNC_FAILED;
        }
        work_path_ = real_model_path + "_dir";
        lib_path_ = work_path_ + "/" + bin_name;
        UDF_LOG_INFO("lib path=%s, funcName=%s, name=%s, instanceName=%s.", lib_path_.c_str(), flow_func_name_.c_str(),
            name_.c_str(), model_param_.model_instance_name.c_str());
    }
    return CheckAndSetBufCfg(udf_def);
}

int32_t FlowFuncModel::Init() {
    ff::udf::UdfModelDef udf_model_def;
    std::string real_model_path;
    if(LoadUdfModel(udf_model_def, real_model_path) != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Convert UDF from model_path=%s to udf_model_def failed.", model_param_.model_path.c_str());
        return FLOW_FUNC_FAILED;
    }
    auto udf_def_size = udf_model_def.udf_def_size();
    if (udf_def_size != 1) {
        UDF_LOG_ERROR("udf_model_def must have only one udf def, model_path=%s, udf_def_size=%d.",
            model_param_.model_path.c_str(), udf_def_size);
        return FLOW_FUNC_FAILED;
    }
    const ff::udf::UdfDef &udf_def = udf_model_def.udf_def(0);
    if (ParseUdfBasicInfos(udf_def, real_model_path) != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Parse udf basic info[name, bin_name, BufCfg] failed.");
        return FLOW_FUNC_FAILED;
    }

    auto &proto_attrs = udf_def.attrs();
    for (auto &proto_attr : proto_attrs) {
        try {
            auto attr_impl = std::make_shared<AttrValueImpl>(proto_attr.second);
            attr_map_.emplace(proto_attr.first, attr_impl);
        } catch (std::exception &e) {
            UDF_LOG_ERROR("new AttrValueImpl failed, funcName=%s, model_path=%s, error=%s.",
                flow_func_name_.c_str(), model_param_.model_path.c_str(), e.what());
            return FLOW_FUNC_FAILED;
        }
    }

    // multi func input maps
    auto &input_maps = udf_def.func_inputs_map();
    for (auto &input_map : input_maps) {
        auto &array = input_map.second;
        std::vector<uint32_t> index_maps;
        for (auto &index : array.num()) {
            index_maps.emplace_back(index);
        }
        multi_func_input_maps_.emplace(input_map.first, index_maps);
    }
    // multi func output maps
    auto &output_maps = udf_def.func_outputs_map();
    for (auto &output_map : output_maps) {
        auto &array = output_map.second;
        std::vector<uint32_t> index_maps;
        for (auto &index : array.num()) {
            index_maps.emplace_back(index);
        }
        multi_func_output_maps_.emplace(output_map.first, index_maps);
    }

    auto &stream_input_func_names = udf_def.stream_input_func_name();
    for (auto &stream_input_func_name : stream_input_func_names) {
        stream_input_func_names_.insert(stream_input_func_name);
    }

    if (EnableVisibleDeviceId() != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Enable visible device id failed.");
        return FLOW_FUNC_FAILED;
    }

    if (SetDataFlowScope() != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Set data flow scope failed.");
        return FLOW_FUNC_FAILED;
    }

    if (SetDataFlowInvokeScopes() != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Set data flow scope failed.");
        return FLOW_FUNC_FAILED;
    }

    UDF_RUN_LOG_INFO("parse name=%s end, flowFuncName=%s, instanceName=%s.", name_.c_str(), flow_func_name_.c_str(),
        model_param_.model_instance_name.c_str());
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncModel::ParseModelAttrs(const ff::deployer::ExecutorRequest_LoadModelRequest &model_request) {
    auto attrs = model_request.attrs();
    if (attrs.contains(kModelEschedProcessPriorityAttrName)) {
        if (!ConvertToInt32(attrs[kModelEschedProcessPriorityAttrName], model_esched_process_priority_)) {
            UDF_LOG_ERROR("Failed to parse %s.", kModelEschedProcessPriorityAttrName);
            return FLOW_FUNC_FAILED;
        }
        UDF_LOG_DEBUG("Parse attr:%s, value:%d", kModelEschedProcessPriorityAttrName,
            model_esched_process_priority_);
    }
    if (attrs.contains(kModelEschedEventPriorityAttrName)) {
        if (!ConvertToInt32(attrs[kModelEschedEventPriorityAttrName], model_esched_event_priority_)) {
            UDF_LOG_ERROR("Failed to parse %s.", kModelEschedEventPriorityAttrName);
            return FLOW_FUNC_FAILED;
        }
        UDF_LOG_DEBUG("Parse attr:%s, value:%d", kModelEschedEventPriorityAttrName,
            model_esched_event_priority_);
    }
    if (model_request.has_input_align_attrs()) {
        const auto &input_align_attrs = model_request.input_align_attrs();
        input_align_attrs_.align_max_cache_num = input_align_attrs.align_max_cache_num();
        input_align_attrs_.align_timeout = input_align_attrs.align_timeout();
        input_align_attrs_.drop_when_not_align = input_align_attrs.drop_when_not_align();
        // max cache num is 1024
        if (input_align_attrs_.align_max_cache_num > 1024) {
            UDF_LOG_ERROR("align_max_cache_num=%u is out of range [0, 1024].", input_align_attrs_.align_max_cache_num);
            return FLOW_FUNC_FAILED;
        }

        // max value is 600 * 1000(ms)=10 minutes
        constexpr int32_t align_timeout_max_value = 600 * 1000;
        // align_timeout -1 means never timeout
        constexpr int32_t align_never_timeout = -1;
        if ((input_align_attrs_.align_timeout != align_never_timeout) &&
            !((input_align_attrs_.align_timeout > 0) && (input_align_attrs_.align_timeout <= align_timeout_max_value))) {
            UDF_LOG_ERROR("align_timeout=%d is invalid, must be -1 or in range(0, %d]ms.",
                input_align_attrs_.align_timeout, align_timeout_max_value);
            return FLOW_FUNC_FAILED;
        }
        UDF_LOG_INFO("model input align attrs value:align_max_cache_num=%u, align_timeout=%d ms, drop_when_not_align=%d",
            input_align_attrs_.align_max_cache_num, input_align_attrs_.align_timeout,
            static_cast<int32_t>(input_align_attrs_.drop_when_not_align));
    }

    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncModel::ConstructModelParam(const ff::deployer::ExecutorRequest_LoadModelRequest
    &model, FlowFuncModelParam &flow_func_model_param) {
    flow_func_model_param.model_path = GlobalConfig::Instance().GetBaseDir() + model.model_path();
    flow_func_model_param.model_instance_name = model.model_instance_name();
    int32_t convert_ret = CheckAndConvert(model.model_queues_attrs(), flow_func_model_param.input_queues,
        flow_func_model_param.output_queues);
    if (convert_ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("CheckAndConvert model queue attr failed.");
        return FLOW_FUNC_FAILED;
    }

    flow_func_model_param.scope = model.scope();
    flow_func_model_param.is_head = model.is_head();
    flow_func_model_param.exception_catch = model.enable_exception_catch();
    flow_func_model_param.is_dynamic_sched = model.is_dynamic_sched();
    flow_func_model_param.need_report_status = (model.is_dynamic_sched() && model.need_report_status());
    if (flow_func_model_param.need_report_status || (model.enable_exception_catch())) {
        UDF_LOG_INFO("Need to report status or exception, root model id[%u], model id[%u].", model.root_model_id(),
            model.model_id());
        // status ouput queues size must be 1 when need to report status
        if (model.status_queues().output_queues_attrs_size() != 1U) {
            UDF_LOG_ERROR("status ouput queues size[%zu] is not equal to 1.",
                model.status_queues().output_queues_attrs_size());
            return FLOW_FUNC_FAILED;
        }
        const auto check_ret = CheckAndConvert(model.status_queues().output_queues_attrs(0),
            flow_func_model_param.status_output_queue);
        if (check_ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("CheckAndConvert output queue attr failed.");
            return FLOW_FUNC_FAILED;
        }
        flow_func_model_param.model_uuid = model.model_uuid();
    }
    return FLOW_FUNC_SUCCESS;
}

std::unique_ptr<FlowFuncModel> FlowFuncModel::ParseSingleModel(const ff::deployer::ExecutorRequest_LoadModelRequest
    &model) {
    FlowFuncModelParam flow_func_model_param = {};
    if (ConstructModelParam(model, flow_func_model_param) != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Construct flow func model param failed.");
        return nullptr;
    }
    std::unique_ptr<FlowFuncModel> flow_func_model(new(std::nothrow)FlowFuncModel(flow_func_model_param));
    if (flow_func_model == nullptr) {
        UDF_LOG_ERROR("malloc FlowFunc failed.");
        return nullptr;
    }
    if (flow_func_model->ParseModelAttrs(model) != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("flow func parse model attrs failed.");
        return nullptr;
    }
    auto ret = flow_func_model->Init();
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("flow func init failed.");
        return nullptr;
    }

    flow_func_model->replica_idx_= static_cast<int32_t>(model.replica_idx());
    flow_func_model->replica_num_= static_cast<int32_t>(model.replica_num());
    for (const auto &invoked_model_queue_attr : model.invoked_model_queues_attrs()) {
        std::vector<QueueDevInfo> feed_queue_infos;
        std::vector<QueueDevInfo> fetch_queue_infos;
        int32_t check_ret = CheckAndConvert(invoked_model_queue_attr.second, feed_queue_infos, fetch_queue_infos);
        if (check_ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("CheckAndConvert model queue attr failed, invoke model key=%s.",
                invoked_model_queue_attr.first.c_str());
            return nullptr;
        }

        std::vector<QueueDevInfo> real_feed_queue_infos;
        std::set<QueueDevInfo> queue_infos;
        for (const auto &queue_info : feed_queue_infos) {
            if (queue_infos.count(queue_info) == 0UL) {
                (void)queue_infos.insert(queue_info);
                real_feed_queue_infos.emplace_back(queue_info);
            }
        }
        UDF_RUN_LOG_INFO("Invoke queue size[%zu] after deduplicate is [%zu].",
                         feed_queue_infos.size(), real_feed_queue_infos.size());
        flow_func_model->AddInvokedModelQueue(invoked_model_queue_attr.first, std::move(real_feed_queue_infos),
            std::move(fetch_queue_infos));
    }
    return flow_func_model;
}

std::vector<std::unique_ptr<FlowFuncModel>> FlowFuncModel::ParseModels(const std::string &batch_model_path) {
    std::ifstream ifs(batch_model_path.c_str(), std::ifstream::in | std::ifstream::binary);
    if (!ifs.is_open()) {
        int32_t error_no = errno;
        UDF_LOG_ERROR("open batch model file failed, file=%s, errno=%d, strerror=%s.", batch_model_path.c_str(), error_no,
            GetErrorNoStr(error_no).c_str());
        return {};
    }

    ff::deployer::ExecutorRequest_BatchLoadModelMessage batch_load_model_req;
    bool parse_ret = batch_load_model_req.ParseFromIstream(&ifs);
    if (!parse_ret) {
        UDF_LOG_ERROR("parse BatchLoadModelMessage failed, file=%s.", batch_model_path.c_str());
        return {};
    }

    int32_t model_num = batch_load_model_req.models_size();
    if (model_num == 0) {
        UDF_LOG_ERROR("models is not exits, file=%s.", batch_model_path.c_str());
        return {};
    }

    std::vector<std::unique_ptr<FlowFuncModel>> flow_func_models;
    flow_func_models.reserve(model_num);
    for (int32_t idx = 0; idx < model_num; ++idx) {
        const auto &model = batch_load_model_req.models(idx);
        std::unique_ptr<FlowFuncModel> flow_func_model(ParseSingleModel(model));
        if (flow_func_model == nullptr) {
            UDF_LOG_ERROR("Parse model idx[%d] failed.", idx);
            return {};
        }
        flow_func_models.emplace_back(std::move(flow_func_model));
    }
    UDF_LOG_INFO("parse models end, file=%s, model num=%zu.", batch_model_path.c_str(), flow_func_models.size());
    return flow_func_models;
}

int32_t FlowFuncModel::CheckAndConvert(const ff::deployer::ExecutorRequest_ModelQueuesAttrs &model_queue_attrs,
    std::vector<QueueDevInfo> &input_queue_infos, std::vector<QueueDevInfo> &output_queue_infos) {
    for (const auto &input_queue_attr : model_queue_attrs.input_queues_attrs()) {
        QueueDevInfo queue_info{};
        int32_t check_ret = CheckAndConvert(input_queue_attr, queue_info);
        if (check_ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("CheckAndConvert input queue attr failed.");
            return check_ret;
        }
        input_queue_infos.emplace_back(queue_info);
    }
    for (const auto &output_queue_attr : model_queue_attrs.output_queues_attrs()) {
        QueueDevInfo queue_info{};
        int32_t check_ret = CheckAndConvert(output_queue_attr, queue_info);
        if (check_ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("CheckAndConvert output queue attr failed.");
            return check_ret;
        }
        output_queue_infos.emplace_back(queue_info);
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncModel::CheckAndConvert(const ff::deployer::ExecutorRequest_QueueAttrs &queue_attr,
    QueueDevInfo &queue_info) {
    queue_info.queue_id = queue_attr.queue_id();
    queue_info.device_type = queue_attr.device_type();
    queue_info.device_id = queue_attr.device_id();
    queue_info.deploy_device_id = queue_attr.device_id();
    queue_info.logic_queue_id = queue_attr.global_logic_id();
    if (GlobalConfig::Instance().IsOnDevice()) {
        // device just support npu queue
        if (queue_info.device_type != Common::kDeviceTypeNpu) {
            UDF_LOG_ERROR("udf is run on device, but device_type[%d] is not npu, queue_id=%u, device_id=%d.",
                queue_info.device_type, queue_info.queue_id, queue_info.device_id);
            return FLOW_FUNC_ERR_PARAM_INVALID;
        }
        queue_info.device_id = static_cast<int32_t>(GlobalConfig::Instance().GetDeviceId());
        queue_info.is_proxy_queue = false;
    } else {
        // run on host and queue on npu means queue is proxy.
        queue_info.is_proxy_queue = (queue_attr.device_type() == Common::kDeviceTypeNpu);
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncModel::GetCpuNumFromAttr(uint32_t &cpu_num, bool &is_attr_get) const {
    const auto attr_iter = attr_map_.find(kModelCpuNumAttrName);
    if (attr_iter == attr_map_.cend() || attr_iter->second == nullptr) {
        UDF_LOG_INFO("no attr found, attrName=%s, funcName=%s, name=%s.",
            kModelCpuNumAttrName, flow_func_name_.c_str(), name_.c_str());
        is_attr_get= false;
        return FLOW_FUNC_SUCCESS;
    }
    int64_t attr_cpu_num = 0L;
    const int32_t ret = attr_iter->second->GetVal(attr_cpu_num);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("get attr val fail, attrName=%s, funcName=%s, name=%s.",
            kModelCpuNumAttrName, flow_func_name_.c_str(), name_.c_str());
        return ret;
    }
    if (attr_cpu_num <= 0L || attr_cpu_num >= static_cast<int64_t>(UINT32_MAX)) {
        UDF_LOG_ERROR("attr %s val %ld is invalid, funcName=%s, name=%s.", kModelCpuNumAttrName,
            attr_cpu_num, flow_func_name_.c_str(), name_.c_str());
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    is_attr_get= true;
    cpu_num = static_cast<uint32_t>(attr_cpu_num);
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncModel::GetVisibleDeviceEnableFromAttr(bool &visible_device_enable) const
{
    const auto attr_iter = attr_map_.find(kAttrNameDataFlowVisibleDeviceEnable);
    if (attr_iter != attr_map_.cend() && attr_iter->second != nullptr) {
        const int32_t ret = attr_iter->second->GetVal(visible_device_enable);
        if (ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("get attr val fail, attrName=%s, funcName=%s, name=%s.",
                kAttrNameDataFlowVisibleDeviceEnable, flow_func_name_.c_str(), name_.c_str());
            return ret;
        }
        if (visible_device_enable && GlobalConfig::Instance().GetPhyDeviceId() < 0) {
            UDF_LOG_ERROR("attr %s not support enable when PhyDeviceId < 0, funcName=%s, name=%s.",
                kAttrNameDataFlowVisibleDeviceEnable, flow_func_name_.c_str(), name_.c_str());
            return FLOW_FUNC_ERR_PARAM_INVALID;
        }
        UDF_LOG_INFO("attr found, attrName=%s, val=%d, funcName=%s, name=%s.",
            kAttrNameDataFlowVisibleDeviceEnable, static_cast<int32_t>(visible_device_enable),
            flow_func_name_.c_str(), name_.c_str());
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncModel::EnableVisibleDeviceId() const {
    bool visible_device_enable = false;
    const auto ret = GetVisibleDeviceEnableFromAttr(visible_device_enable);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("get visible device enable attr fail, funcName=%s, name=%s.",
            flow_func_name_.c_str(), name_.c_str());
        return ret;
    }

    if (visible_device_enable) {
        auto phy_device_id = GlobalConfig::Instance().GetPhyDeviceId();
        (void)mmSetEnv(kEnvNameAscendRtVisibleDevices, std::to_string(phy_device_id).c_str(), 1);
        GlobalConfig::Instance().SetRunningDeviceId(0);
        UDF_RUN_LOG_INFO("enable visible device id success, envName=%s, val=%d, funcName=%s, name=%s.",
            kEnvNameAscendRtVisibleDevices, phy_device_id, flow_func_name_.c_str(), name_.c_str());
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncModel::SetDataFlowScope() {
    AscendString data_flow_scope;
    const auto attr_iter = attr_map_.find(kAttrNameDataFlowDataFlowScope);
    if (attr_iter != attr_map_.cend() && attr_iter->second != nullptr) {
        const int32_t ret = attr_iter->second->GetVal(data_flow_scope);
        if (ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("get attr val fail, attrName=%s, funcName=%s, name=%s.",
                kAttrNameDataFlowDataFlowScope, flow_func_name_.c_str(), name_.c_str());
            return ret;
        }
        data_flow_scope_ = data_flow_scope.GetString();
        UDF_LOG_INFO("attr found, attrName=%s, val=%s, funcName=%s, name=%s.",
            kAttrNameDataFlowDataFlowScope, data_flow_scope_.c_str(),
            flow_func_name_.c_str(), name_.c_str());
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncModel::SetDataFlowInvokeScopes() {
    std::vector<AscendString> invoked_scopes;
    const auto attr_iter = attr_map_.find(kAttrNameDataFlowDataFlowInvokeScopes);
    if (attr_iter != attr_map_.cend() && attr_iter->second != nullptr) {
        const int32_t ret = attr_iter->second->GetVal(invoked_scopes);
        if (ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("get attr val fail, attrName=%s, funcName=%s, name=%s.",
                kAttrNameDataFlowDataFlowInvokeScopes, flow_func_name_.c_str(), name_.c_str());
            return ret;
        }
        for (size_t i = 0; i < invoked_scopes.size(); i++) {
            invoked_scopes_.emplace_back(std::string(invoked_scopes[i].GetString()));
        }
        UDF_LOG_INFO("attr found, attrName=%s, size=%zu, funcName=%s, name=%s.",
            kAttrNameDataFlowDataFlowInvokeScopes, invoked_scopes_.size(),
            flow_func_name_.c_str(), name_.c_str());
    }
    return FLOW_FUNC_SUCCESS;
}
}
