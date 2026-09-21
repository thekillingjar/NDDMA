/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_func_drv_manager.h"
#include "securec.h"
#include "config/global_config.h"
#include "common/common_define.h"
#include "common/udf_log.h"
#include "ascend_hal.h"

namespace FlowFunc {
namespace {
constexpr const char *kPageTypeNormal = "normal";
}
FlowFuncDrvManager &FlowFuncDrvManager::Instance() {
    static FlowFuncDrvManager instance;
    return instance;
}

int32_t FlowFuncDrvManager::Init() const {
    UDF_RUN_LOG_INFO("ready to init drv");
    if (!GlobalConfig::Instance().IsOnDevice()) {
        struct process_sign sign = {};
        drvError_t drv_get_ret = drvGetProcessSign(&sign);
        UDF_RUN_LOG_INFO("drvGetProcessSign ret=%d, tgid=%d", static_cast<int32_t>(drv_get_ret), sign.tgid);
    }
    int32_t init_mem_grp_ret = InitMemGroup();
    if (init_mem_grp_ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("InitMemGroup failed, ret=%d", init_mem_grp_ret);
        return init_mem_grp_ret;
    }
    uint32_t device_id = GlobalConfig::Instance().GetDeviceId();
    drvError_t drv_ret = halEschedAttachDevice(device_id);
    if (drv_ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("Failed to attach device[%u] for eSched, drv_ret=%d.", device_id,
            static_cast<int32_t>(drv_ret));
        return FLOW_FUNC_ERR_DRV_ERROR;
    }
    int32_t init_buf_ret = InitBuff();
    if (init_buf_ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Init mbuf failed, ret=%d.", init_buf_ret);
        return init_buf_ret;
    }

    QueueSetInputPara input{};
    drv_ret = halQueueSet(device_id, QUEUE_ENABLE_LOCAL_QUEUE, &input);
    if (drv_ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("halQueueSet error, device_id=%u, drv_ret=%d", device_id, static_cast<int32_t>(drv_ret));
        return FLOW_FUNC_ERR_QUEUE_ERROR;
    }

    int32_t create_ret = CreateSchedGroup();
    if (create_ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("create sched group failed, ret=%d.", create_ret);
        return create_ret;
    }

    return FLOW_FUNC_SUCCESS;
}

void FlowFuncDrvManager::Finalize() const {
    uint32_t device_id = GlobalConfig::Instance().GetDeviceId();
    (void)halEschedDettachDevice(device_id);
}

int32_t FlowFuncDrvManager::InitMemGroup() const {
    const std::string &mem_group_name = GlobalConfig::Instance().GetMemGroupName();
    int32_t drv_ret = halGrpAttach(mem_group_name.c_str(), Common::kAttachMemGrpWaitTimeout);
    if (drv_ret != static_cast<int32_t>(DRV_ERROR_NONE)) {
        UDF_LOG_ERROR("halGrpAttach group[%s] failed. drv_ret=%d, timeout=%d(ms)", mem_group_name.c_str(), drv_ret,
            Common::kAttachMemGrpWaitTimeout);
        return FLOW_FUNC_ERR_DRV_ERROR;
    }
    UDF_RUN_LOG_INFO("halGrpAttach group[%s] success.", mem_group_name.c_str());
    GroupQueryInput query_input{};
    int32_t cpy_ret = memcpy_s(query_input.grpQueryGroupId.grpName, BUFF_GRP_NAME_LEN, mem_group_name.c_str(),
        mem_group_name.size());
    if (cpy_ret != EOK) {
        UDF_LOG_ERROR("copy group name failed, dst len=%d, src len=%zu, group name=%s",
            BUFF_GRP_NAME_LEN, mem_group_name.size(), mem_group_name.c_str());
        return FLOW_FUNC_FAILED;
    }
    std::unique_ptr<GroupQueryOutput> query_output(new(std::nothrow) GroupQueryOutput());
    if (query_output == nullptr) {
        UDF_LOG_ERROR("alloc GroupQueryOutput failed, alloc size=%zu.", sizeof(GroupQueryOutput));
        return FLOW_FUNC_FAILED;
    }
    uint32_t out_buf_len = sizeof(GroupQueryOutput);
    (void)memset_s(query_output.get(), out_buf_len, '\0', out_buf_len);
    drv_ret = halGrpQuery(GRP_QUERY_GROUP_ID, &query_input, sizeof(query_input), query_output.get(), &out_buf_len);
    if (drv_ret != static_cast<int32_t>(DRV_ERROR_NONE)) {
        UDF_LOG_ERROR("halGrpQuery group[%s] failed. drv_ret=%d, cmd_type=%d", mem_group_name.c_str(), drv_ret,
            static_cast<int32_t>(GRP_QUERY_GROUP_ID));
        return FLOW_FUNC_FAILED;
    }
    GlobalConfig::Instance().SetMemGroupId(query_output->grpQueryGroupIdInfo.groupId);
    UDF_LOG_INFO("query group[%s] id success, group id=%d.", mem_group_name.c_str(),
        query_output->grpQueryGroupIdInfo.groupId);
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncDrvManager::InitBuff() const {
    BuffCfg default_cfg = {};
    const auto &buf_cfg = GlobalConfig::Instance().GetBufCfg();
    if (buf_cfg.size() > BUFF_MAX_CFG_NUM) {
        UDF_LOG_ERROR("Buf cfg number[%zu] can not be larger than max limit[%d].", buf_cfg.size(), BUFF_MAX_CFG_NUM);
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    for (size_t i = 0UL; i < buf_cfg.size(); ++i) {
        const auto &cfg = buf_cfg[i];
        uint32_t page_type = (cfg.page_type == kPageTypeNormal) ? BUFF_SP_NORMAL : BUFF_SP_HUGEPAGE_ONLY;
        default_cfg.cfg[i] = {.cfg_id = static_cast<uint32_t>(i), .total_size = cfg.total_size,
            .blk_size = cfg.blk_size, .max_buf_size = cfg.max_buf_size,
            .page_type = (page_type | BUFF_SP_DVPP), 1, 0, 0, 0, 6, {}};
        UDF_LOG_INFO("Prepare to set buf config{ id[%lu] total_size[%u] blk_size[%u] max_buf_size[%u] "
                     "page_type[%s|dvpp] }.", i, cfg.total_size, cfg.blk_size, cfg.max_buf_size, cfg.page_type.c_str());
    }
    UDF_RUN_LOG_INFO("Prepare to init buf using buffer config number[%zu]", buf_cfg.size());
    int32_t dev_int_ret = halBuffInit(&default_cfg);
    if ((dev_int_ret != static_cast<int32_t>(DRV_ERROR_NONE)) &&
        (dev_int_ret != static_cast<int32_t>(DRV_ERROR_REPEATED_INIT))) {
        UDF_LOG_ERROR("halBuffInit init failed, drv_ret=%d", dev_int_ret);
        return FLOW_FUNC_ERR_MEM_BUF_ERROR;
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncDrvManager::CreateSchedGroup() const {
    uint32_t device_id = GlobalConfig::Instance().GetDeviceId();
    // run on device, use GRP_TYPE_BIND_DP_CPU, host use GRP_TYPE_BIND_CP_CPU
    const GROUP_TYPE group_type = GlobalConfig::Instance().IsRunOnAiCpu() ? GRP_TYPE_BIND_DP_CPU : GRP_TYPE_BIND_CP_CPU;
    constexpr uint32_t sched_max_thread_num = 256;
    esched_grp_para main_sched_group_para = {group_type, sched_max_thread_num, "udf_main", {}};
    uint32_t main_sched_group_id = 0;
    drvError_t drv_ret = halEschedCreateGrpEx(device_id, &main_sched_group_para, &main_sched_group_id);
    if (drv_ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("Failed to create main sched group[%s], group_type=%d, threadNum=%u, drv_ret=%d.",
            main_sched_group_para.grp_name, static_cast<int32_t>(group_type), sched_max_thread_num,
            static_cast<int32_t>(drv_ret));
        return FLOW_FUNC_ERR_DRV_ERROR;
    }
    uint32_t worker_sched_group_id = main_sched_group_id;
    if (!GlobalConfig::Instance().IsOnDevice()) {
        esched_grp_para worker_sched_group_para = {group_type, sched_max_thread_num, "udf_worker", {}};
        drv_ret = halEschedCreateGrpEx(device_id, &worker_sched_group_para, &worker_sched_group_id);
        if (drv_ret != DRV_ERROR_NONE) {
            UDF_LOG_ERROR("Failed to create worker sched group[%s], group_type=%d, threadNum=%u, drv_ret=%d.",
                worker_sched_group_para.grp_name, static_cast<int32_t>(group_type), sched_max_thread_num,
                static_cast<int32_t>(drv_ret));
            return FLOW_FUNC_ERR_DRV_ERROR;
        }
    }
    esched_grp_para invoke_model_sched_group_para = {group_type, sched_max_thread_num, "udf_invoke", {}};
    uint32_t invoke_model_sched_group_id = 0;
    drv_ret = halEschedCreateGrpEx(device_id, &invoke_model_sched_group_para, &invoke_model_sched_group_id);
    if (drv_ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("Failed to create invoked model sched group[%s], group_type=%d, threadNum=%u, drv_ret=%d.",
            invoke_model_sched_group_para.grp_name, static_cast<int32_t>(group_type), sched_max_thread_num,
            static_cast<int32_t>(drv_ret));
        return FLOW_FUNC_ERR_DRV_ERROR;
    }

    esched_grp_para flow_msg_queue_sched_group_para = {group_type, sched_max_thread_num, "udf_dequeue", {}};
    uint32_t flow_msg_queue_sched_group_id = 0;
    drv_ret = halEschedCreateGrpEx(device_id, &flow_msg_queue_sched_group_para, &flow_msg_queue_sched_group_id);
    if (drv_ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("Failed to create flow msg queue sched group[%s], group_type=%d, threadNum=%u, drv_ret=%d.",
            flow_msg_queue_sched_group_para.grp_name, static_cast<int32_t>(group_type), sched_max_thread_num,
            static_cast<int32_t>(drv_ret));
        return FLOW_FUNC_ERR_DRV_ERROR;
    }

    GlobalConfig::Instance().SetMainSchedGroupId(main_sched_group_id);
    GlobalConfig::Instance().SetWorkerSchedGroupId(worker_sched_group_id);
    GlobalConfig::Instance().SetInvokeModelSchedGroupId(invoke_model_sched_group_id);
    GlobalConfig::Instance().SetFlowMsgQueueSchedGroupId(flow_msg_queue_sched_group_id);
    UDF_LOG_INFO("create sched group success, main_sched_group_id=%u, worker_sched_group_id=%u, invoke_model_sched_group_id=%u.",
        main_sched_group_id, worker_sched_group_id, invoke_model_sched_group_id);
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncDrvManager::WaitBindHostPid(uint32_t wait_timeout) const {
    int32_t udf_pid = getpid();
    uint32_t chip_id = 0;
    uint32_t vf_id = 0;
    uint32_t host_pid = 0;
    uint32_t cp_type = 0;

    // retry interval is 50ms
    constexpr uint32_t retry_interval = 50;
    uint32_t retry_time = 0;
    do {
        // need drvQueryProcessHostPid to wait bind host pid success.
        const drvError_t drv_query_ret = drvQueryProcessHostPid(udf_pid, &chip_id, &vf_id, &host_pid, &cp_type);
        if ((drv_query_ret == DRV_ERROR_NONE)) {
            UDF_LOG_INFO("drvQueryProcessHostPid success, udf_pid=%d, chip_id=%u, vf_id=%u, host_pid=%u, cp_type=%u",
                udf_pid, chip_id, vf_id, host_pid, cp_type);
            return FLOW_FUNC_SUCCESS;
        }
        if (drv_query_ret != DRV_ERROR_NO_PROCESS) {
            UDF_LOG_WARN("drvQueryProcessHostPid failed, drv_query_ret=%d, udf_pid=%u",
                static_cast<int32_t>(drv_query_ret), udf_pid);
            return FLOW_FUNC_ERR_DRV_ERROR;
        }
        if (retry_time >= wait_timeout) {
            UDF_LOG_WARN("drvQueryProcessHostPid timeout, udf_pid=%u, retry_time=%u ms", udf_pid, retry_time);
            break;
        }
        UDF_LOG_DEBUG("drvQueryProcessHostPid need retry later, udf_pid=%u", udf_pid);
        std::this_thread::sleep_for(std::chrono::milliseconds(retry_interval));
        retry_time += retry_interval;
    } while (true);
    return FLOW_FUNC_FAILED;
}
}