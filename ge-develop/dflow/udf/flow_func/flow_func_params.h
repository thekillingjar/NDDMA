/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_PARAMS_H
#define FLOW_FUNC_PARAMS_H

#include <map>
#include <mutex>
#include <vector>
#include <unordered_set>
#include <memory>
#include "common/common_define.h"
#include "flow_func/meta_params.h"
#include "flow_func/meta_flow_func.h"
#include "flow_func/flow_model.h"
#include "flow_func/attr_value.h"
#include "mbuf_flow_msg.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY FlowFuncParams : public MetaParams {
public:
    FlowFuncParams(const std::string &name, size_t input_queue_num, size_t output_queue_num,
        int32_t running_device_id, int32_t device_id);

    ~FlowFuncParams() override;

    const char *GetName() const override {
        return name_.c_str();
    }

    /*
     * get attr.
     * @return AttrValue *: not null->success, null->failed
     */
    std::shared_ptr<const AttrValue> GetAttr(const char *attr_name) const override;

    size_t GetInputNum() const override {
        return input_queue_num_;
    }

    size_t GetOutputNum() const override {
        // now all output must with queues, pls change here if support part output with queues.
        return output_queue_num_;
    }

    void SetLibPath(const std::string &lib_path) {
        lib_path_ = lib_path;
    }

    const std::string &GetLibPath() const {
        return lib_path_;
    }

    void SetAttrMap(const std::map<std::string, std::shared_ptr<const AttrValue>> &attr_map) {
        attr_map_ = attr_map;
    }

    void SetWorkPath(const std::string &work_path) {
        work_path_ = work_path;
    }

    const char *GetWorkPath() const override {
        return work_path_.c_str();
    }

    void AddFlowModel(const std::string &model_key, std::unique_ptr<FlowModel> &&flow_model);

    FlowModel *GetFlowModels(const std::string &model_key) const;

    bool HasInvokedModel(const std::string &scope, std::string &key) const;

    bool HandleInvokedException(const std::string &scope, uint64_t trans_id, bool is_add_exception) const;

    std::mutex &GetOutputQueueLocks(uint32_t output_idx);

    int32_t Init();

    int32_t GetRunningDeviceId() const override {
        return running_device_id_;
    }

    uint32_t GetDeviceId() const {
        return device_id_;
    }

    bool IsBalanceScatter() const {
        return is_balance_scatter_;
    }

    bool IsBalanceGather() const {
        return is_balance_gather_;
    }

    void SetStatusOutputQueue(const QueueDevInfo &status_output_queue) {
        status_output_queue_ = status_output_queue;
    }

    const QueueDevInfo &GetStatusOutputQueue() const {
        return status_output_queue_;
    }

    void SetModelUuid(uint32_t model_uuid) {
        model_uuid_ = model_uuid;
    }

    uint32_t GetModelUuid() const {
        return model_uuid_;
    }

    void SetNeedReportStatusFlag(bool need_report_status) {
        need_report_status_ = need_report_status;
    }

    bool GetNeedReportStatusFlag() const {
        return need_report_status_;
    }

    std::mutex &GetReportStatusMutexRef() {
        return report_status_lock_;
    }

    void SetHeadInfo(bool is_head) {
        is_head_ = is_head;
    }

    bool IsHead() const {
        return is_head_;
    }

    const char *GetInstanceName() const {
        return instance_name_.c_str();
    }

    void SetInstanceName(const std::string &instance_name) {
        instance_name_ = instance_name;
    }

    const std::string &GetScope() const {
        return scope_;
    }

    void SetScope(const std::string &scope) {
        scope_ = scope;
    }

    bool GetEnableRaiseException() const {
        return exception_catch_;
    }

    void SetEnableRaiseException(const bool enable_exp) {
        exception_catch_ = enable_exp;
    }

    void SetStreamInputFuncNames(const std::unordered_set<std::string> &func_names) {
        stream_input_func_names_ = func_names;
    }

    const std::unordered_set<std::string> &GetStreamInputFuncNames() const {
        return stream_input_func_names_;
    }

    int32_t GetRunningInstanceId() const override {
        return running_instance_id_;
    }

    void SetRunnningInstanceId(int32_t instance_id) {
        running_instance_id_ = instance_id;
    }

    int32_t GetRunningInstanceNum() const override {
        return running_instance_num_;
    }

    void SetRunnningInstanceNum(int32_t instance_num) {
        running_instance_num_ = instance_num;
    }

    void SetInvokedScopes(const std::vector<std::string> &invoked_scopes) {
        invoked_scopes_.assign(invoked_scopes.cbegin(), invoked_scopes.cend());
    }

private:
    int32_t InitBalanceAttr();

    std::map<std::string, std::unique_ptr<FlowModel>> flow_models_;
    std::string name_;
    std::string instance_name_;
    std::vector<std::mutex> output_queued_locks_;
    std::string lib_path_;
    std::map<std::string, std::shared_ptr<const AttrValue>> attr_map_;
    size_t input_queue_num_;
    size_t output_queue_num_;
    std::string work_path_;
    volatile bool is_inited_ = false;
    bool is_balance_scatter_ = false;
    bool is_balance_gather_ = false;
    int32_t running_device_id_;
    uint32_t device_id_;
    QueueDevInfo status_output_queue_ = {};
    uint32_t model_uuid_ = 0U;
    bool need_report_status_ = false;
    std::mutex report_status_lock_;
    bool is_head_ = false;
    std::string scope_;
    bool exception_catch_ = false;
    std::unordered_set<std::string> stream_input_func_names_;
    int32_t running_instance_id_ = 0;
    int32_t running_instance_num_ = 1;
    std::vector<std::string> invoked_scopes_;
};
}  // namespace FlowFunc

#endif  // FLOW_FUNC_PARAMS_H
