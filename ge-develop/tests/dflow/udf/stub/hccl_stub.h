/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDF_HCCL_STUB_H
#define UDF_HCCL_STUB_H

#include "hccl/base.h"

HcclResult HcclRawAcceptStub(HcclConn listen_conn, HcclAddr *accept_addr, HcclConn *accept_conn);

HcclResult HcclRawIsendStub(const void *buf, int32_t count, HcclDataType data_type, HcclConn conn,
                            HcclRequest *request);

HcclResult HcclRawImprobeStub(HcclConn conn, int32_t *flag, HcclMessage *msg, HcclStatus *status);

HcclResult HcclRawGetCountStub(const HcclStatus *status, HcclDataType data_type, int *count);

HcclResult HcclRawImrecvStub(void *buf, int32_t count, HcclDataType data_type, HcclMessage *msg, HcclRequest *request);

HcclResult HcclRawImrecvScatterStub(void *buf[], int counts[], int buf_count, HcclDataType data_type, HcclMessage *msg,
                                    HcclRequest *request);

HcclResult HcclRawTestSomeStub(int32_t count, HcclRequest request_array[], int32_t *comp_count, int32_t comp_indices[],
                               HcclStatus comp_status[]);

HcclResult HcclRawConnectStub(HcclConn conn, HcclAddr *connect_addr);

HcclResult HcclRawBindStub(HcclConn conn, HcclAddr *bind_addr);

HcclResult HcclRawListenStub(HcclConn conn, int back_log);

HcclResult HcclRawOpenStub(HcclConn *conn);

HcclResult HcclRawCloseStub(HcclConn conn);

HcclResult HcclRawForceCloseStub(HcclConn conn);

HcclResult HcclRegisterGlobalMemoryStub(void *addr, u64 size);

HcclResult HcclUnregisterGlobalMemoryStub(void *addr);

#endif  // UDF_HCCL_STUB_H
