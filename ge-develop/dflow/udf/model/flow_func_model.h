/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_MODEL_H
#define FLOW_FUNC_MODEL_H

#include <string>
#include <vector>
#include <unordered_set>
#include <memory>
#include <map>
#include "flow_func/attr_value.h"
#include "common/common_define.h"
#include "ff_deployer.pb.h"
#include "ff_udf_def.pb.h"

namespace FlowFunc {
constexpr int32_t kUserUnsetESchedPriority = -1;
struct ModelQueue {
    std::vector<QueueDevInfo> feed_queue_infos;
    std::vector<QueueDevInfo> fetch_queue_infos;
};
struct FlowFuncModelParam {
    std::string model_path;
    std::string model_instance_name;
    std::vector<QueueDevInfo> input_queues;
    std::vector<QueueDevInfo> output_queues;
    QueueDevInfo status_output_queue;
    uint32_t model_uuid;
    bool is_dynamic_sched;
    bool need_report_status;
    bool is_head;
    bool exception_catch;
    std::string scope;
};
struct InputAlignAttrs {
    uint32_t align_max_cache_num; // 0 means align not enable
    int32_t align_timeout;  // -1 means never timeout
    bool drop_when_not_align;
    uint8_t res[3];
};
static_assert(std::is_pod<InputAlignAttrs>::value, "The class InputAlignAttrs must be a POD");

class FlowFuncModel {
public:
    explicit FlowFuncModel(FlowFuncModelParam flow_func_model_param);

    ~FlowFuncModel() = default;

    int32_t Init();

    /**
     * @brief parse from batch model path.
     * @param batch_model_path batch model path
     * @return FlowFunc model list
     */
    static std::vector<std::unique_ptr<FlowFuncModel>> ParseModels(const std::string &batch_model_path);

    const std::string &GetName() const {
        return name_;
    }

    const std::string &GetFlowFuncName() const {
        return flow_func_name_;
    }

    const std::string &GetLibPath() const {
        return lib_path_;
    }

    /**
     * @brief get node attr.
     * @return convert node proto attr to FlowFunc attr value.
     */
    const std::map<std::string, std::shared_ptr<const AttrValue>> &GetNodeAttrMap() const {
        return attr_map_;
    }

    const std::map<std::string, ModelQueue> &GetInvokedModelQueueInfos() const {
        return invoked_model_queue_infos_;
    }

    void AddInvokedModelQueue(const std::string &model_key, std::vector<QueueDevInfo> &&feed_queue_ids,
        std::vector<QueueDevInfo> &&fetch_queue_ids) {
        invoked_model_queue_infos_[model_key] = {feed_queue_ids, fetch_queue_ids};
    }

    const std::vector<QueueDevInfo> &GetInputQueues() const {
        return model_param_.input_queues;
    }

    const std::vector<QueueDevInfo> &GetOutputQueues() const {
        return model_param_.output_queues;
    }

    const QueueDevInfo &GetStatusOutputQueue() const {
        return model_param_.status_output_queue;
    }

    uint32_t GetModelUuid() const {
        return model_param_.model_uuid;
    }

    bool NeedReportStatus() const {
        return model_param_.need_report_status;
    }

    const std::string &GetWorkPath() const {
        return work_path_;
    }

    int32_t GetModelEschedProcessPriority() const {
        return model_esched_process_priority_;
    }

    int32_t GetModelEschedEventPriority() const {
        return model_esched_event_priority_;
    }

    const std::map<std::string, std::vector<uint32_t>> &GetMultiFuncInputMaps() const {
        return multi_func_input_maps_;
    }

    const std::map<std::string, std::vector<uint32_t>> &GetMultiFuncOutputMaps() const {
        return multi_func_output_maps_;
    }

    const std::unordered_set<std::string> &GetStreamInputFuncNames() const {
        return stream_input_func_names_;
    }

    int32_t GetCpuNumFromAttr(uint32_t &cpu_num, bool &is_attr_get) const;

    const FlowFuncModelParam &GetFlowFuncModelParam() const {
        return model_param_;
    }
    bool IsHead() const {
        return model_param_.is_head;
    }

    const InputAlignAttrs &GetInputAlignAttrs() const {
        return input_align_attrs_;
    }

    const std::string &GetModelInstanceName() const {
        return model_param_.model_instance_name;
    }

    const std::string &GetScope() const {
        return model_param_.scope;
    }

    const std::string &GetDataFlowScope() const {
        return data_flow_scope_;
    }

    const std::vector<std::string> &GetInvokedScopes() const {
        return invoked_scopes_;
    }

    bool GetEnableRaiseException() const {
        return model_param_.exception_catch;
    }

    int32_t GetReplicaIdx() const {
        return replica_idx_;
    }

    int32_t GetReplicaNum() const {
        return replica_num_;
    }

private:

    int32_t ParseModelAttrs(const ff::deployer::ExecutorRequest_LoadModelRequest &model_request);
    int32_t ParseUdfBasicInfos(const ff::udf::UdfDef &udf_def, const std::string &real_model_path);
    int32_t LoadUdfModel(ff::udf::UdfModelDef &udf_model_def, std::string &real_model_path);

    static int32_t ConstructModelParam(const ff::deployer::ExecutorRequest_LoadModelRequest &model,
                                       FlowFuncModelParam &flow_func_model_param);
    static std::unique_ptr<FlowFuncModel> ParseSingleModel(const ff::deployer::ExecutorRequest_LoadModelRequest &model);
    static int32_t CheckAndConvert(const ff::deployer::ExecutorRequest_QueueAttrs &queue_attr, QueueDevInfo &queue_info);

    static int32_t CheckAndConvert(const ff::deployer::ExecutorRequest_ModelQueuesAttrs &model_queue_attrs,
        std::vector<QueueDevInfo> &input_queue_infos, std::vector<QueueDevInfo> &output_queue_infos);

    static bool IsFileNameValid(const std::string &file_name);

    static int32_t CheckBufCfgItem(const BufferConfigItem& item);

    static int32_t CheckAndSetBufCfg(const ff::udf::UdfDef &udf_def);

    int32_t EnableVisibleDeviceId() const;
    int32_t GetVisibleDeviceEnableFromAttr(bool &visible_device_enable) const;
    int32_t SetDataFlowScope();
    int32_t SetDataFlowInvokeScopes();

    FlowFuncModelParam model_param_{};
    std::string lib_path_;
    std::string work_path_;
    std::string flow_func_name_;
    std::string name_;
    std::string data_flow_scope_;
    std::vector<std::string> invoked_scopes_;

    InputAlignAttrs input_align_attrs_{};

    std::map<std::string, ModelQueue> invoked_model_queue_infos_;
    // multi func input maps
    std::map<std::string, std::vector<uint32_t>> multi_func_input_maps_;
    // multi func output maps
    std::map<std::string, std::vector<uint32_t>> multi_func_output_maps_;
    std::unordered_set<std::string> stream_input_func_names_;

    // attr
    std::map<std::string, std::shared_ptr<const AttrValue>> attr_map_;

    // model esched priority
    int32_t model_esched_process_priority_ = kUserUnsetESchedPriority;
    int32_t model_esched_event_priority_ = kUserUnsetESchedPriority;

    int32_t replica_idx_ = 0;
    int32_t replica_num_ = 1;
};
}

#endif // FLOW_FUNC_MODEL_H
