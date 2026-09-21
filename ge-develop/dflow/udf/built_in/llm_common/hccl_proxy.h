/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BUILT_IN_LLM_COMMON_HCCL_PROXY_H
#define BUILT_IN_LLM_COMMON_HCCL_PROXY_H

#include "hccl/base.h"

namespace FlowFunc {
HcclResult HcclRawAccept(HcclConn listen_conn, HcclAddr *accept_addr, HcclConn *accept_conn);

HcclResult HcclRawIsend(const void *buf, int count, HcclDataType data_type, HcclConn conn, HcclRequest *request);

HcclResult HcclRawImprobe(HcclConn conn, int *flag, HcclMessage *msg, HcclStatus *status);

HcclResult HcclRawGetCount(const HcclStatus *status, HcclDataType data_type, int *count);

HcclResult HcclRawImrecv(void *buf, int count, HcclDataType data_type, HcclMessage *msg, HcclRequest *request);

HcclResult HcclRawImrecvScatter(void *buf[], int count[], int buf_count, HcclDataType data_type, HcclMessage *msg,
    HcclRequest *request);

HcclResult HcclRawTestSome(int count, HcclRequest request_array[], int *comp_count, int comp_indices[],
    HcclStatus comp_status[]);

HcclResult HcclRawConnect(HcclConn conn, HcclAddr *connect_addr);

HcclResult HcclRawBind(HcclConn conn, HcclAddr *bind_addr);

HcclResult HcclRawListen(HcclConn conn, int back_log);

HcclResult HcclRawOpen(HcclConn *conn);

HcclResult HcclRawClose(HcclConn conn);

HcclResult HcclRawForceClose(HcclConn conn);

HcclResult HcclRegisterGlobalMemory(void *addr, u64 size);

HcclResult HcclUnregisterGlobalMemory(void *addr);
}
#endif  // BUILT_IN_LLM_COMMON_HCCL_PROXY_H