/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __HCCL_STUB_H__
#define __HCCL_STUB_H__

#include <string>
#include <map>
#include <atomic>
#include <mutex>
#include <list>
#include <vector>

using std::string;
using std::list;

#include "acl/acl_rt.h"
// #include "hccl/hccl.h"
#include "hcom.h"
#include "runtime/base.h"
#include "rts/rts_mem.h"
#include "ge/ge_error_codes.h"
#include "hccl/hccl_types.h"
#include "hccl/base.h"
#include "exe_graph/lowering/dev_mem_value_holder.h"
#include "hccl/hcom.h"
#include "llt_hccl_stub_sal_pub.h"
#include "llt_hccl_stub_gert.h"

#include "graph/compute_graph.h"

#include "graph/gnode.h"

#include "exe_graph/lowering/shape_utils.h"

#include "graph/utils/attr_utils.h"

#include "llt_hccl_stub.h"

typedef enum tagDrvError {
    DRV_ERROR_NONE = 0,                /**< success */
    DRV_ERROR_NO_DEVICE = 1,           /**< no valid device */
    DRV_ERROR_INVALID_DEVICE = 2,      /**< invalid device */
    DRV_ERROR_INVALID_VALUE = 3,       /**< invalid value */
    DRV_ERROR_INVALID_HANDLE = 4,      /**< invalid handle */
    DRV_ERROR_INVALID_MALLOC_TYPE = 5, /**< invalid malloc type */
    DRV_ERROR_OUT_OF_MEMORY = 6,       /**< out of memory */
    DRV_ERROR_INNER_ERR = 7,           /**< driver inside error */
    DRV_ERROR_PARA_ERROR = 8,          /**< driver wrong parameter */
    DRV_ERROR_UNINIT = 9,              /**< driver uninit */
    DRV_ERROR_REPEATED_INIT = 10,          /**< driver repeated init */
    DRV_ERROR_NOT_EXIST = 11,        /**< there is resource*/
    DRV_ERROR_REPEATED_USERD = 12,
    DRV_ERROR_BUSY = 13,                /**< task already running */
    DRV_ERROR_NO_RESOURCES = 14,        /**< driver short of resouces */
    DRV_ERROR_OUT_OF_CMD_SLOT = 15,
    DRV_ERROR_WAIT_TIMEOUT = 16,       /**< driver wait timeout*/
    DRV_ERROR_IOCRL_FAIL = 17,         /**< driver ioctl fail*/

    DRV_ERROR_SOCKET_CREATE = 18,      /**< driver create socket error*/
    DRV_ERROR_SOCKET_CONNECT = 19,     /**< driver connect socket error*/
    DRV_ERROR_SOCKET_BIND = 20,        /**< driver bind socket error*/
    DRV_ERROR_SOCKET_LISTEN = 21,      /**< driver listen socket error*/
    DRV_ERROR_SOCKET_ACCEPT = 22,      /**< driver accept socket error*/
    DRV_ERROR_CLIENT_BUSY = 23,        /**< driver client busy error*/
    DRV_ERROR_SOCKET_SET = 24,         /**< driver socket set error*/
    DRV_ERROR_SOCKET_CLOSE = 25,       /**< driver socket close error*/
    DRV_ERROR_RECV_MESG = 26,          /**< driver recv message error*/
    DRV_ERROR_SEND_MESG = 27,          /**< driver send message error*/
    DRV_ERROR_SERVER_BUSY = 28,
    DRV_ERROR_CONFIG_READ_FAIL = 29,
    DRV_ERROR_STATUS_FAIL = 30,
    DRV_ERROR_SERVER_CREATE_FAIL = 31,
    DRV_ERROR_WAIT_INTERRUPT = 32,
    DRV_ERROR_BUS_DOWN = 33,
    DRV_ERROR_DEVICE_NOT_READY = 34,
    DRV_ERROR_REMOTE_NOT_LISTEN = 35,
    DRV_ERROR_NON_BLOCK = 36,

    DRV_ERROR_OVER_LIMIT = 37,
    DRV_ERROR_FILE_OPS = 38,
    DRV_ERROR_MBIND_FAIL = 39,
    DRV_ERROR_MALLOC_FAIL = 40,

    DRV_ERROR_REPEATED_SUBSCRIBED = 41,
    DRV_ERROR_PROCESS_EXIT = 42,
    DRV_ERROR_DEV_PROCESS_HANG = 43,

    DRV_ERROR_REMOTE_NO_SESSION = 44,

    DRV_ERROR_HOT_RESET_IN_PROGRESS = 45,

    DRV_ERROR_OPER_NOT_PERMITTED = 46,

    DRV_ERROR_NO_EVENT_RESOURCES = 47,
    DRV_ERROR_NO_STREAM_RESOURCES = 48,
    DRV_ERROR_NO_NOTIFY_RESOURCES = 49,
    DRV_ERROR_NO_MODEL_RESOURCES = 50,
    DRV_ERROR_TRY_AGAIN = 51,

    DRV_ERROR_DST_PATH_ILLEGAL = 52,                    /**< send file dst path illegal*/
    DRV_ERROR_OPEN_FAILED = 53,                         /**< send file open failed */
    DRV_ERROR_NO_FREE_SPACE = 54,                       /**< send file no free space */
    DRV_ERROR_LOCAL_ABNORMAL_FILE = 55,                 /**< send file local file abnormal*/
    DRV_ERROR_DST_PERMISSION_DENIED = 56,               /**< send file dst Permission denied*/
    DRV_ERROR_DST_NO_SUCH_FILE = 57,                    /**< pull file no such file or directory*/

    DRV_ERROR_MEMORY_OPT_FAIL = 58,
    DRV_ERROR_RUNTIME_ON_OTHER_PLAT = 59,
    DRV_ERROR_SQID_FULL = 60,                           /**< driver SQ   is full */

    DRV_ERROR_SERVER_HAS_BEEN_CREATED = 61,
    DRV_ERROR_NO_PROCESS = 62,
    DRV_ERROR_NO_SUBSCRIBE_THREAD = 63,
    DRV_ERROR_NON_SCHED_GRP_MUL_THREAD = 64,
    DRV_ERROR_NO_GROUP = 65,
    DRV_ERROR_GROUP_EXIST = 66,
    DRV_ERROR_THREAD_EXCEEDS_SPEC = 67,
    DRV_ERROR_THREAD_NOT_RUNNIG = 68,
    DRV_ERROR_PROCESS_NOT_MATCH = 69,
    DRV_ERROR_EVENT_NOT_MATCH = 70,
    DRV_ERROR_PROCESS_REPEAT_ADD = 71,
    DRV_ERROR_GROUP_NON_SCHED = 72,
    DRV_ERROR_NO_EVENT = 73,
    DRV_ERROR_COPY_USER_FAIL = 74,
    DRV_ERROR_QUEUE_EMPTY = 75,
    DRV_ERROR_QUEUE_FULL = 76,
    DRV_ERROR_RUN_IN_ILLEGAL_CPU = 77,
    DRV_ERROR_SUBSCRIBE_THREAD_TIMEOUT = 78,
    DRV_ERROR_BAD_ADDRESS = 79,
    DRV_ERROR_DST_FILE_IS_BEING_WRITTEN = 80,           /**< send file The dts file is being written */
    DRV_ERROR_EPOLL_CLOSE = 81,                         /**< epoll close */
    DRV_ERROR_CDQ_ABNORMAL = 82,
    DRV_ERROR_CDQ_NOT_EXIST = 83,
    DRV_ERROR_NO_CDQ_RESOURCES = 84,
    DRV_ERROR_CDQ_QUIT = 85,
    DRV_ERROR_PARTITION_NOT_RIGHT = 86,
    DRV_ERROR_RESOURCE_OCCUPIED = 87,
    DRV_ERROR_PERMISSION = 88,
    DRV_ERROR_RESUME = 89,
    DEV_ERROR_UNAUTHORIZED_ACCESS_DEVICE = 90,
    DEV_ERROR_BIST_HW_ERR = 91,
    DEV_ERROR_BIST_SW_ERR = 92,
    DEV_ERROR_DUP_CONFIG = 93,
    DRV_ERROR_POWER_OP_FAIL = 94,
    DRV_ERROR_NET_UNREACH = 95,         /**< net error or device failure*/
    DRV_ERROR_TRANS_LINK_ABNORMAL = 96,
    DRV_ERROR_CALL_NO_RETRY,            /**< Function call error，no need to retry */   
    DRV_ERROR_NOT_SUPPORT = 0xfffe,
    DRV_ERROR_RESERVED
} drvError_t;//lint !e116 !e17

typedef drvError_t DVresult;

int set_board_id(unsigned int board_id);


#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#define  ACL_ERROR_RT_PARAM_INVALID              107000 // param invalid

HcclResult CommGetInstSizeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t *rankNum);

HcclResult CommGetInstTopoTypeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t *topoType);

HcclResult GetGroupNameByOpBaseHcom(s64 opBaseHcom, char **groupname);

HcclResult HcomGetRankId(const char *group, u32 *rankId);

HcclResult HcomInitByString(const char *rankTableM, const char *identify,
    WorkMode commWorkMode, HcomInitConfig *initConfig);

bool HcomFindGroup(const char *group);

HcclResult HcomCreateGroup(const char *group, u32 rankNum, u32 *rankIds);

HcclResult HcomSetWorkspaceResource(const char *tag, const char *group, rtStream_t *stream,
    s32 len, void *memPtr, u64 maxSize);

HcclResult HcomGetInCCLbuffer(const char *group, void** buffer, u64 *size);

HcclResult HcomGetOutCCLbuffer(const char *group, void** buffer, u64 *size);

HcclResult HcomCreateCommCCLbuffer(const char *group);

HcclResult HcomGetInCclBuf(const int64_t &hcomComm, const char *sGroup, void* &commInputPtr, u64 &commInputSize);

HcclResult HcomGetOutCclBuf(const int64_t &hcomComm, const char *sGroup, void* &commOutputPtr, u64 &commOutputSize);

HcclResult HcomCreateCommCclBuf(const int64_t &hcomComm, const char *group);

HcclResult HcomGetIndirectInCclBuf(const int64_t &hcomComm, const char *sGroup, void* &indirectInCCLbufPtr, u64 &indirectCommInputSize);

HcclResult HcomGetIndirectOutCclBuf(const int64_t &hcomComm, const char *sGroup, void* &indirectOutCCLbufPtr, u64 &indirectCommOutputSize);



aclError aclrtGetMemcpyDescSize(aclrtMemcpyKind kind, size_t *descSize);

aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy);

aclError aclrtSetMemcpyDesc(void *desc, aclrtMemcpyKind kind, void *srcAddr, void *dstAddr, size_t count, void *config);


aclError aclrtMemcpyAsyncWithDesc(void *desc, aclrtMemcpyKind kind, aclrtStream stream);



HcclResult HcomGetMemType(const char *group, const char *socVersion, bool isMalloc, u32 *memType, bool *isTsMem,
    bool withoutImplCompile, bool level2Address);


HcclResult HcomGetSplitStrategy(const char *group, const struct model_feature *feature,
    u32 **segmentIdxPtr, u32 *len, bool *configured, GradSplitForceMode force,
    OriginalGraphShapeType shapeType);

HcclResult HcomGetRankSize(const char *group, u32 *rankSize);

HcclWorkflowMode HcomGetWorkflowMode();

HcclResult HcomSetWorkflowMode(HcclWorkflowMode mode);

HcclResult HcomCalcOpResOffline(HcomOpParam *hcomOpParam, HcomResResponse *hcomResResponse);

HcclResult HcomCalcOpOnline(HcomOpParam *hcomOpParam, HcomResResponse *hcomResResponse);

bool HcomGetSecAddrCopyFlag(const char *socVersion);

HcclResult GenerateCclOpTag(const std::string &opType, const int64_t &hcomComm,
    std::string& group, std::string &sTag);

HcclResult HcomGetInitStatus(bool *initiated);

HcclResult HcomGetCommHandleByGroup(const char *group, HcclComm *commHandle);

HcclResult HcomAllToAll(const void *sendBuf, u64 sendCount, HcclDataType sendType,
                        const void *recvBuf, u64 recvCount, HcclDataType recvType,
                        const char *group, rtStream_t stream, const char *tag);

HcclResult HcomAlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
    const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
    const char *group, rtStream_t stream, const char *tag);

HcclResult HcomAlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
    const void *recvBuf, HcclDataType recvType, const char *group, rtStream_t stream, const char *tag);

DevType HcomGetDeviceType();

HcclResult HcomTbeMemClean(int64_t addrList[], int64_t sizeList[], uint32_t count,
    rtStream_t stream, int32_t deviceLogicId);

HcclResult HcomAllReduce(const char *tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, const char *group, rtStream_t stream);

HcclResult HcomAllGather(const char *tag, void *inputPtr, void *outputPtr, u64 inputCount,
    HcclDataType dataType, const char *group, rtStream_t stream);

HcclResult HcomAllGatherV(const char *tag, const void *sendBuf, u64 sendCount, const void *recvBuf,
    const void *recvCounts, const void *rdispls, HcclDataType dataType, const char *group, rtStream_t stream);

HcclResult HcomReduceScatter(const char *tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, const char *group, rtStream_t stream);

HcclResult HcomReduceScatterV(const char *tag, void *sendBuf, const void *sendCounts, const void *sdispls,
    void *recvBuf, u64 recvCount, HcclDataType dataType, HcclReduceOp op, const char *group, rtStream_t stream);

HcclResult HcomBroadcast(const char *tag, void *ptr, u64 count, HcclDataType dataType, u32 root,
    const char *group, rtStream_t stream);

HcclResult HcomReduce(const char *tag, void *inputPtr, void *outputPtr, u64 count, HcclDataType dataType,
    HcclReduceOp op, u32 root, const char *group, rtStream_t stream);

HcclResult HcomSend(const char *tag, void *inputPtr, u64 count, HcclDataType dataType,
    u32 destRank, u32 srTag, const char *group, rtStream_t stream);

HcclResult HcomReceive(const char *tag, void *outputPtr, u64 count, HcclDataType dataType,
    u32 srcRank, u32 srTag, const char *group, rtStream_t stream);

HcclResult HcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType,
                               const void *recvBuf, uint64_t recvCount, HcclDataType recvType,
                               HcclComm comm, aclrtStream stream);

HcclResult HcclAlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
                         const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
                         HcclComm comm, aclrtStream stream);

HcclResult HcclAlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType, 
                            const void *recvBuf, HcclDataType recvType, HcclComm comm, aclrtStream stream);

HcclResult HcomSupportDeterministicOptim(const char *group, bool *isDeterministicOptim);



void HcomSetDumpDebugMode(const bool dumpDebug);

HcclResult HcomClearAivSyncBuf(const char *group, bool aivClearEnable);

HcclResult HcceGetandClearOverFlowTasks(const char *group, hccl::HcclDumpInfo **hcclDumpInfo, s32 *len);

HcclResult HcomUnloadTask(const char *group, const char *tag);

HcclResult HcomReleaseSubComms();

HcclResult HcomMc2AiCpuStreamAllocAndGet(const char *group, u32 streamMode, rtStream_t *aiCpuStream);

HcclResult HcomSetAivCoreLimit(const char *group, u32 aivCoreLimit);

HcclResult HcclAllocComResourceByTiling(HcclComm comm, void* stream, void* Mc2Tiling, void** commContext);

HcclResult HcomCreateComResourceByComm(HcclComm comm, u32 streamMode, bool isOpbaseMode,
    void** commContext, bool isMC2);

HcclResult HcomGetAicpuOpStreamNotify(const char *group, HcclRtStream *opStream, u8 aicpuNotifyNum, void** aicpuNotify);

HcclResult HcomGenerateCclOpTag(const char *opType, s64 hcomComm, const char *group, char *sTag);

HcclResult HcomGetAlgExecParam(const char *tag, const char *group, u64 count, void *inputPtr, void *outputPtr,
    HcclCMDType opType, bool clearEnable, HcclDataType dataType, HcclReduceOp op, 
    void **commContext, u64 *len, u32 aivCoreLimit);

void HcomSetLaunchKernelMode(bool state);


void HcomSetAutoTuneMode(bool autoTuneMode);

HcclResult HcomGetServerNumAndDeviceNumPerServer(u32 *serverNum, u32 *deviceNumPerServer, u32 *deviceNumPerAggregation);

HcclResult HcomGetAlgorithm(u32 level, char** algo);

HcclResult HcomGetBandWidthPerNPU(u32 level, float *bandWidth);

void HcomTopoInfoRegCallback(HcclResult (*p1)(const char *, uint32_t), void (*p2)(const char *));

HcclResult HcomSetProfilingMode(HcomProfilingMode profilingMode, const char *profilingOption);

HcclResult HcomInitByMasterInfo(const char *masterIp, const char *masterPort,
    const char *masterDeviceId, const char *rankSize, const char *rankIp, HcomInitConfig *initConfig);

HcclResult HcomDestroy(void);

HcclResult HcomSetGlobalWorkSpace(const char *group, void **globalWorkSpaceAddr, u32 len);

HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType,
    HcclComm comm, aclrtStream stream);

HcclResult HcclAllGatherV(void *sendBuf, uint64_t sendCount, void *recvBuf,
    const void *recvCounts, const void *recvDispls, HcclDataType dataType, HcclComm comm, aclrtStream stream);

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream);

HcclResult HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm,
    aclrtStream stream);

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream);

HcclResult HcclReduceScatterV(void *sendBuf, const void *sendCounts, const void *sendDispls,
    void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream);

HcclResult HcclSend(void* sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank,
                           HcclComm comm, aclrtStream stream);

HcclResult HcclReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
                             HcclReduceOp op, uint32_t root, HcclComm comm, aclrtStream stream);

HcclResult HcclRecv(void* recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank,
                           HcclComm comm, aclrtStream stream);

HcclResult HcomSetAttachedStream(const char *group, u32 graphId, const rtStream_t *stream, s32 len);
HcclResult HcomSelectAlg(s64 comm, const char *group, u64 count, void* counts,
    HcclDataType dataType, HcclReduceOp op, HcclCMDType opType, int32_t aivCoreLimit,
    bool &ifAiv, char *algName);
HcclResult HcomCalcAivCoreNum(const char *group, HcclCMDType opType, u64 count, void* counts, HcclDataType dataType, int32_t aivCoreLimit,
        char *algName, u32 *blockDim);
HcclResult HcomGetCommCCLBufferSize(const char *group, uint64_t &size);

HcclResult HcclGetInstTopoTypeByNetLayer(HcclComm comm, uint32_t netLayer, CommTopo *topoType);
HcclResult HcclGetInstSizeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t *rankNum);
HcclResult HcclGetNetLayers(HcclComm comm, uint32_t **netLayers, uint32_t *netLayerNum);
#ifdef __cplusplus
}
#endif // __cplusplus

#endif