/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOM_EXECUTOR_H
#define HCOM_EXECUTOR_H

#include "hccl/base.h"
#include "hccl/hccl_types.h"
#include <functional>
#include <vector>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/**
 * @brief Initialize hcom executor.
 *
 * @param void
 * @return HcclResult
 */
HcclResult HcomExecInitialize();

/**
 * @brief Finalize hcom executor.
 *
 * @param void
 * @return HcclResult
 */
HcclResult HcomExecFinalize();

/**
 * @brief Put collective communication operation into hcom executor.
 *
 * @param opInfo information about collective communication operation.
 * @param callback callback after collective communication operation.
 * @return HcclResult
 */
HcclResult HcomExecEnqueueOperation(HcomOperation opInfo, std::function<void(HcclResult status)> callback);

/**
 * @brief Put collective communication remote operation into hcom executor.
 *
 * @param opInfo information about collective communication remote operation.
 * @param callback callback after collective communication remote operation.
 * @return HcclResult
 */
HcclResult HcomExecEnqueueRemoteOperation(HcomRemoteOperation opInfo, std::function<void(HcclResult status)> callback);

/**
 * @brief Put remote access operation into hcom executor.
 *
 * @param remoteAccessType operation type (read or write).
 * @param addrInfos address information about collective communication operation.
 * @param callback callback after collective communication operation.
 * @return HcclResult
 */
HcclResult HcomExecEnqueueRemoteAccess(const std::string &remoteAccessType,
                                       const std::vector<HcomRemoteAccessAddrInfo> &addrInfos,
                                       std::function<void(HcclResult status)> callback);

/**
 * @brief Put alltoallv communication operation into hcom executor.
 *
 * @param params information about alltoallv communication operation.
 * @param callback callback after collective communication operation.
 * @return HcclResult
 */
HcclResult HcomExecEnqueueAllToAllV(HcomAllToAllVParams params, std::function<void(HcclResult status)> callback);

/**
 * @brief Put alltoallvc communication operation into hcom executor.
 *
 * @param params information about alltoallvc communication operation.
 * @param callback callback after collective communication operation.
 * @return HcclResult
 */
HcclResult HcomExecEnqueueAllToAllVC(HcomAllToAllVCParams params, std::function<void(HcclResult status)> callback);

/**
 * @brief Put agther alltoallv communication operation into hcom executor.
 *
 * @param params information about agther alltoallv communication operation.
 * @param callback callback after collective communication operation.
 * @return HcclResult
 */
HcclResult HcomExecEnqueueGatherAllToAllV(HcomGatherAllToAllVParams params,
                                          std::function<void(HcclResult status)> callback);

/**
 * @brief Register memories and init resources for remote access.
 *
 * @param addrList memory addresses for remote access.
 * @param count number of remote memory addresses.
 * @return HcclResult
 */
HcclResult HcomRegRemoteAccessMem(const MemRegisterAddr *addrList, u32 count);

HcclResult HcomGetActualRankSize(const char *group, u32 *rankSize);

HcclResult HcomSetRankTable(const char *rankTable);
#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // HCOM_H
