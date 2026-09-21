/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_proxy.h"
#include "common/udf_log.h"
#include "llm_common/hccl_so_manager.h"
#include "llm_common/statistic_manager.h"

namespace FlowFunc {
HcclResult HcclRawAccept(HcclConn listen_conn, HcclAddr *accept_addr, HcclConn *accept_conn) {
    const uint64_t start_tick = FlowFunc::StatisticManager::GetInstance().GetCpuTick();
    void *const func = FlowFunc::HcclSoManager::GetInstance()->GetFunc(FlowFunc::kHcclRawAcceptFuncName);
    if (func == nullptr) {
        UDF_LOG_ERROR("Fail to get function:%s from hccl so.", FlowFunc::kHcclRawAcceptFuncName);
        return HCCL_E_NOT_SUPPORT;
    }
    auto ret = (reinterpret_cast<FlowFunc::HcclRawAcceptFunc>(func))(listen_conn, accept_addr, accept_conn);
    FlowFunc::StatisticManager::GetInstance().AddRawAcceptCost(
        FlowFunc::StatisticManager::GetInstance().GetCpuTick() - start_tick,
        (ret == HCCL_SUCCESS) && (accept_conn != nullptr));
    return ret;
}

HcclResult HcclRawIsend(const void *buf, int count, HcclDataType data_type, HcclConn conn, HcclRequest *request) {
    const uint64_t start_tick = FlowFunc::StatisticManager::GetInstance().GetCpuTick();
    void *const func = FlowFunc::HcclSoManager::GetInstance()->GetFunc(FlowFunc::kHcclRawIsendFuncName);
    if (func == nullptr) {
        UDF_LOG_ERROR("Fail to get function:%s from hccl so.", FlowFunc::kHcclRawIsendFuncName);
        return HCCL_E_NOT_SUPPORT;
    }
    auto ret = (reinterpret_cast<FlowFunc::HcclRawIsendFunc>(func))(buf, count, data_type, conn, request);
    FlowFunc::StatisticManager::GetInstance().AddIsendCost(
        FlowFunc::StatisticManager::GetInstance().GetCpuTick() - start_tick);
    return ret;
}

HcclResult HcclRawImprobe(HcclConn conn, int *flag, HcclMessage *msg, HcclStatus *status) {
    const uint64_t start_tick = FlowFunc::StatisticManager::GetInstance().GetCpuTick();
    void *const func = FlowFunc::HcclSoManager::GetInstance()->GetFunc(FlowFunc::kHcclRawImprobeFuncName);
    if (func == nullptr) {
        UDF_LOG_ERROR("Fail to get function:%s from hccl so.", FlowFunc::kHcclRawImprobeFuncName);
        return HCCL_E_NOT_SUPPORT;
    }
    auto ret = (reinterpret_cast<FlowFunc::HcclRawImprobeFunc>(func))(conn, flag, msg, status);
    FlowFunc::StatisticManager::GetInstance().AddImprobeCost(
        FlowFunc::StatisticManager::GetInstance().GetCpuTick() - start_tick);
    return ret;
}

HcclResult HcclRawGetCount(const HcclStatus *status, HcclDataType data_type, int *count) {
    const uint64_t start_tick = FlowFunc::StatisticManager::GetInstance().GetCpuTick();
    void *const func = FlowFunc::HcclSoManager::GetInstance()->GetFunc(FlowFunc::kHcclRawGetCountFuncName);
    if (func == nullptr) {
        UDF_LOG_ERROR("Fail to get function:%s from hccl so.", FlowFunc::kHcclRawGetCountFuncName);
        return HCCL_E_NOT_SUPPORT;
    }
    auto ret = (reinterpret_cast<FlowFunc::HcclRawGetCountFunc>(func))(status, data_type, count);
    FlowFunc::StatisticManager::GetInstance().AddGetCountCost(
        FlowFunc::StatisticManager::GetInstance().GetCpuTick() - start_tick);
    return ret;
}

HcclResult HcclRawImrecv(void *buf, int count, HcclDataType data_type, HcclMessage *msg, HcclRequest *request) {
    const uint64_t start_tick = FlowFunc::StatisticManager::GetInstance().GetCpuTick();
    void *const func = FlowFunc::HcclSoManager::GetInstance()->GetFunc(FlowFunc::kHcclRawImRecvFuncName);
    if (func == nullptr) {
        UDF_LOG_ERROR("Fail to get function:%s from hccl so.", FlowFunc::kHcclRawImRecvFuncName);
        return HCCL_E_NOT_SUPPORT;
    }
    auto ret = (reinterpret_cast<FlowFunc::HcclRawImrecvFunc>(func))(buf, count, data_type, msg, request);
    FlowFunc::StatisticManager::GetInstance().AddImrecvCost(
        FlowFunc::StatisticManager::GetInstance().GetCpuTick() - start_tick);
    return ret;
}

HcclResult HcclRawImrecvScatter(void *buf[], int count[], int buf_count, HcclDataType data_type, HcclMessage *msg,
                                HcclRequest *request) {
    const uint64_t start_tick = FlowFunc::StatisticManager::GetInstance().GetCpuTick();
    void *const func = FlowFunc::HcclSoManager::GetInstance()->GetFunc(FlowFunc::kHcclRawImRecvScatterFuncName);
    if (func == nullptr) {
        UDF_LOG_ERROR("Fail to get function:%s from hccl so.", FlowFunc::kHcclRawImRecvScatterFuncName);
        return HCCL_E_NOT_SUPPORT;
    }
    auto ret =
        (reinterpret_cast<FlowFunc::HcclRawImrecvScatterFunc>(func))(buf, count, buf_count, data_type, msg, request);
    FlowFunc::StatisticManager::GetInstance().AddImrecvScatterCost(
        FlowFunc::StatisticManager::GetInstance().GetCpuTick() - start_tick);
    return ret;
}

HcclResult HcclRawTestSome(int count, HcclRequest request_array[], int *comp_count, int comp_indices[],
                           HcclStatus comp_status[]) {
    const uint64_t start_tick = FlowFunc::StatisticManager::GetInstance().GetCpuTick();
    void *const func = FlowFunc::HcclSoManager::GetInstance()->GetFunc(FlowFunc::kHcclRawTestSomeFuncName);
    if (func == nullptr) {
        UDF_LOG_ERROR("Fail to get function:%s from hccl so.", FlowFunc::kHcclRawTestSomeFuncName);
        return HCCL_E_NOT_SUPPORT;
    }
    auto ret = (reinterpret_cast<FlowFunc::HcclRawTestSomeFunc>(func))(count, request_array, comp_count, comp_indices,
                                                                       comp_status);
    FlowFunc::StatisticManager::GetInstance().AddTestSomeCost(
        FlowFunc::StatisticManager::GetInstance().GetCpuTick() - start_tick);
    return ret;
}

HcclResult HcclRawConnect(HcclConn conn, HcclAddr *connect_addr) {
    const uint64_t start_tick = FlowFunc::StatisticManager::GetInstance().GetCpuTick();
    void *const func = FlowFunc::HcclSoManager::GetInstance()->GetFunc(FlowFunc::kHcclRawConnectFuncName);
    if (func == nullptr) {
        UDF_LOG_ERROR("Fail to get function:%s from hccl so.", FlowFunc::kHcclRawConnectFuncName);
        return HCCL_E_NOT_SUPPORT;
    }
    auto ret = (reinterpret_cast<FlowFunc::HcclRawConnectFunc>(func))(conn, connect_addr);
    FlowFunc::StatisticManager::GetInstance().AddRawConnectCost(
        FlowFunc::StatisticManager::GetInstance().GetCpuTick() - start_tick, (ret == HCCL_SUCCESS));
    return ret;
}

HcclResult HcclRawBind(HcclConn conn, HcclAddr *bind_addr) {
    void *const func = FlowFunc::HcclSoManager::GetInstance()->GetFunc(FlowFunc::kHcclRawBindFuncName);
    if (func == nullptr) {
        UDF_LOG_ERROR("Fail to get function:%s from hccl so.", FlowFunc::kHcclRawBindFuncName);
        return HCCL_E_NOT_SUPPORT;
    }
    return (reinterpret_cast<FlowFunc::HcclRawBindFunc>(func))(conn, bind_addr);
}

HcclResult HcclRawListen(HcclConn conn, int back_log) {
    void *const func = FlowFunc::HcclSoManager::GetInstance()->GetFunc(FlowFunc::kHcclRawListenFuncName);
    if (func == nullptr) {
        UDF_LOG_ERROR("Fail to get function:%s from hccl so.", FlowFunc::kHcclRawListenFuncName);
        return HCCL_E_NOT_SUPPORT;
    }
    return (reinterpret_cast<FlowFunc::HcclRawListenFunc>(func))(conn, back_log);
}

HcclResult HcclRawOpen(HcclConn *conn) {
    const uint64_t start_tick = FlowFunc::StatisticManager::GetInstance().GetCpuTick();
    void *const func = FlowFunc::HcclSoManager::GetInstance()->GetFunc(FlowFunc::kHcclRawOpenFuncName);
    if (func == nullptr) {
        UDF_LOG_ERROR("Fail to get function:%s from hccl so.", FlowFunc::kHcclRawOpenFuncName);
        return HCCL_E_NOT_SUPPORT;
    }
    auto ret = (reinterpret_cast<FlowFunc::HcclRawOpenFunc>(func))(conn);
    FlowFunc::StatisticManager::GetInstance().AddRawOpenCost(
        FlowFunc::StatisticManager::GetInstance().GetCpuTick() - start_tick);
    return ret;
}

HcclResult HcclRawClose(HcclConn conn) {
    const uint64_t start_tick = FlowFunc::StatisticManager::GetInstance().GetCpuTick();
    void *const func = FlowFunc::HcclSoManager::GetInstance()->GetFunc(FlowFunc::kHcclRawCloseFuncName);
    if (func == nullptr) {
        UDF_LOG_ERROR("Fail to get function:%s from hccl so.", FlowFunc::kHcclRawCloseFuncName);
        return HCCL_E_NOT_SUPPORT;
    }
    auto ret = (reinterpret_cast<FlowFunc::HcclRawCloseFunc>(func))(conn);
    FlowFunc::StatisticManager::GetInstance().AddRawCloseCost(
        FlowFunc::StatisticManager::GetInstance().GetCpuTick() - start_tick);
    return ret;
}

HcclResult HcclRawForceClose(HcclConn conn) {
    UDF_LOG_INFO("Call HcclRawForceClose");
    const uint64_t start_tick = FlowFunc::StatisticManager::GetInstance().GetCpuTick();
    void *const func = FlowFunc::HcclSoManager::GetInstance()->GetFunc(FlowFunc::kHcclRawForceCloseFuncName);
    auto ret = HCCL_E_NOT_SUPPORT;
    if (func != nullptr) {
        ret = (reinterpret_cast<FlowFunc::HcclRawForceCloseFunc>(func))(conn);
        FlowFunc::StatisticManager::GetInstance().AddRawCloseCost(
                FlowFunc::StatisticManager::GetInstance().GetCpuTick() - start_tick);
    }
    return ret;
}

HcclResult HcclRegisterGlobalMemory(void *addr, u64 size) {
    void *const func = FlowFunc::HcclSoManager::GetInstance()->GetFunc(FlowFunc::kHcclRegisterGlobalMemoryFuncName);
    if (func == nullptr) {
        UDF_LOG_ERROR("Fail to get function:%s from hccl so.", FlowFunc::kHcclRegisterGlobalMemoryFuncName);
        return HCCL_E_NOT_SUPPORT;
    }
    return (reinterpret_cast<FlowFunc::HcclRegisterGlobalMemoryFunc>(func))(addr, size);
}

HcclResult HcclUnregisterGlobalMemory(void *addr) {
    void *const func =
        FlowFunc::HcclSoManager::GetInstance()->GetFunc(FlowFunc::kHcclUnregisterGlobalMemoryFuncName);
    if (func == nullptr) {
        UDF_LOG_ERROR("Fail to get function:%s from hccl so.", FlowFunc::kHcclUnregisterGlobalMemoryFuncName);
        return HCCL_E_NOT_SUPPORT;
    }
    return (reinterpret_cast<FlowFunc::HcclUnregisterGlobalMemoryFunc>(func))(addr);
}
}