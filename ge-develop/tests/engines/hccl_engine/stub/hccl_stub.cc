/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_stub.h"

using namespace gert;
using namespace bg;


#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

HcclResult CommGetInstSizeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t *rankNum)
{
    return HCCL_SUCCESS;
}

HcclResult CommGetInstTopoTypeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t *topoType)
{
    return HCCL_SUCCESS;
}

HcclResult GetGroupNameByOpBaseHcom(s64 opBaseHcom, char **groupname)
{
    return HCCL_SUCCESS;
}

HcclResult HcomGetRankId(const char *group, u32 *rankId)
{
    return HCCL_SUCCESS;
}

HcclResult HcomInitByString(const char *rankTableM, const char *identify,
    WorkMode commWorkMode, HcomInitConfig *initConfig)
{
    return HCCL_SUCCESS;
}

bool HcomFindGroup(const char *group)
{
    return true;
}

HcclResult HcomCreateGroup(const char *group, u32 rankNum, u32 *rankIds)
{
    return HCCL_SUCCESS;
}

HcclResult HcomSetWorkspaceResource(const char *tag, const char *group, rtStream_t *stream,
    s32 len, void *memPtr, u64 maxSize)
{
    return HCCL_SUCCESS;
}

HcclResult HcomGetInCCLbuffer(const char *group, void** buffer, u64 *size)
{
    return HCCL_SUCCESS;
}

HcclResult HcomGetOutCCLbuffer(const char *group, void** buffer, u64 *size)
{
    return HCCL_SUCCESS;
}

HcclResult HcomCreateCommCCLbuffer(const char *group)
{
    return HCCL_SUCCESS;
}

HcclResult HcomGetInCclBuf(const int64_t &hcomComm, const char *sGroup, void* &commInputPtr, u64 &commInputSize)
{
    return HCCL_SUCCESS;
}

HcclResult HcomGetOutCclBuf(const int64_t &hcomComm, const char *sGroup, void* &commOutputPtr, u64 &commOutputSize)
{
    return HCCL_SUCCESS;
}

HcclResult HcomCreateCommCclBuf(const int64_t &hcomComm, const char *group)
{
    return HCCL_SUCCESS;
}

HcclResult HcomGetIndirectInCclBuf(const int64_t &hcomComm, const char *sGroup, void* &indirectInCCLbufPtr, u64 &indirectCommInputSize)
{
    return HCCL_SUCCESS;
}

HcclResult HcomGetIndirectOutCclBuf(const int64_t &hcomComm, const char *sGroup, void* &indirectOutCCLbufPtr, u64 &indirectCommOutputSize)
{
    return HCCL_SUCCESS;
}



aclError aclrtGetMemcpyDescSize(aclrtMemcpyKind kind, size_t *descSize)
{
    return ACL_SUCCESS;
}

aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    return ACL_SUCCESS;
}
aclError aclrtSetMemcpyDesc(void *desc, aclrtMemcpyKind kind, void *srcAddr, void *dstAddr, size_t count, void *config)
{
    return ACL_SUCCESS;
}


aclError aclrtMemcpyAsyncWithDesc(void *desc, aclrtMemcpyKind kind, aclrtStream stream)
{
    return ACL_SUCCESS;
}



HcclResult HcomGetMemType(const char *group, const char *socVersion, bool isMalloc, u32 *memType, bool *isTsMem,
    bool withoutImplCompile, bool level2Address)
{
    return HCCL_SUCCESS;
}


HcclResult HcomGetSplitStrategy(const char *group, const struct model_feature *feature,
    u32 **segmentIdxPtr, u32 *len, bool *configured, GradSplitForceMode force,
    OriginalGraphShapeType shapeType)
{
    return HCCL_SUCCESS;
}

HcclResult HcomGetRankSize(const char *group, u32 *rankSize)
{
    *rankSize = 1 ;
    return HCCL_SUCCESS;
}

HcclWorkflowMode HcomGetWorkflowMode()
{
    return HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED;
}

HcclResult HcomSetWorkflowMode(HcclWorkflowMode mode)
{
    return HCCL_SUCCESS;
}

HcclResult HcomCalcOpResOffline(HcomOpParam *hcomOpParam, HcomResResponse *hcomResResponse)
{
    return HCCL_SUCCESS;
}

HcclResult HcomCalcOpOnline(HcomOpParam *hcomOpParam, HcomResResponse *hcomResResponse)
{
    return HCCL_SUCCESS;
}

bool HcomGetSecAddrCopyFlag(const char *socVersion)
{
    return true;
}

HcclResult GenerateCclOpTag(const std::string &opType, const int64_t &hcomComm,
    std::string& group, std::string &sTag)
{
    return HCCL_SUCCESS;
}

HcclResult HcomGetInitStatus(bool *initiated)
{
    return HCCL_SUCCESS;
}

HcclResult HcomGetCommHandleByGroup(const char *group, HcclComm *commHandle)
{   
    static char dummy;
    *commHandle = &dummy;
    return HCCL_SUCCESS;
}

HcclResult HcomAllToAll(const void *sendBuf, u64 sendCount, HcclDataType sendType,
                        const void *recvBuf, u64 recvCount, HcclDataType recvType,
                        const char *group, rtStream_t stream, const char *tag)
{
    return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
    const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
    const char *group, rtStream_t stream, const char *tag)
{
    return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
    const void *recvBuf, HcclDataType recvType, const char *group, rtStream_t stream, const char *tag)
{
    return HCCL_SUCCESS;
}

DevType HcomGetDeviceType()
{
    return DevType::DEV_TYPE_910;
}

HcclResult HcomTbeMemClean(int64_t addrList[], int64_t sizeList[], uint32_t count,
    rtStream_t stream, int32_t deviceLogicId)
{
    return HCCL_SUCCESS;
}

HcclResult HcomAllReduce(const char *tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, const char *group, rtStream_t stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcomAllGather(const char *tag, void *inputPtr, void *outputPtr, u64 inputCount,
    HcclDataType dataType, const char *group, rtStream_t stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcomAllGatherV(const char *tag, const void *sendBuf, u64 sendCount, const void *recvBuf,
    const void *recvCounts, const void *rdispls, HcclDataType dataType, const char *group, rtStream_t stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcomReduceScatter(const char *tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, const char *group, rtStream_t stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcomReduceScatterV(const char *tag, void *sendBuf, const void *sendCounts, const void *sdispls,
    void *recvBuf, u64 recvCount, HcclDataType dataType, HcclReduceOp op, const char *group, rtStream_t stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcomBroadcast(const char *tag, void *ptr, u64 count, HcclDataType dataType, u32 root,
    const char *group, rtStream_t stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcomReduce(const char *tag, void *inputPtr, void *outputPtr, u64 count, HcclDataType dataType,
    HcclReduceOp op, u32 root, const char *group, rtStream_t stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcomSend(const char *tag, void *inputPtr, u64 count, HcclDataType dataType,
    u32 destRank, u32 srTag, const char *group, rtStream_t stream)
{
    return HCCL_SUCCESS;
}
HcclResult HcomReceive(const char *tag, void *outputPtr, u64 count, HcclDataType dataType,
    u32 srcRank, u32 srTag, const char *group, rtStream_t stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType,
                               const void *recvBuf, uint64_t recvCount, HcclDataType recvType,
                               HcclComm comm, aclrtStream stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclAlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
                         const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
                         HcclComm comm, aclrtStream stream)
{
    return HCCL_SUCCESS;
}


HcclResult HcclAlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType, 
                            const void *recvBuf, HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcomSupportDeterministicOptim(const char *group, bool *isDeterministicOptim)
{
    return HCCL_SUCCESS;
}



void HcomSetDumpDebugMode(const bool dumpDebug){}

HcclResult HcomClearAivSyncBuf(const char *group, bool aivClearEnable)
{
    return HCCL_SUCCESS;
}

HcclResult HcceGetandClearOverFlowTasks(const char *group, hccl::HcclDumpInfo **hcclDumpInfo, s32 *len)
{
    return HCCL_SUCCESS;
}

HcclResult HcomUnloadTask(const char *group, const char *tag)
{
    return HCCL_SUCCESS;
}

HcclResult HcomReleaseSubComms()
{
    return HCCL_SUCCESS;
}

HcclResult HcomMc2AiCpuStreamAllocAndGet(const char *group, u32 streamMode, rtStream_t *aiCpuStream)
{
    return HCCL_SUCCESS;
}

HcclResult HcomSetAivCoreLimit(const char *group, u32 aivCoreLimit)
{
    return HCCL_SUCCESS;
}

HcclResult HcclAllocComResourceByTiling(HcclComm comm, void* stream, void* Mc2Tiling, void** commContext)
{
    return HCCL_SUCCESS;
}

HcclResult HcomCreateComResourceByComm(HcclComm comm, u32 streamMode, bool isOpbaseMode,
    void** commContext, bool isMC2)
{
    return HCCL_SUCCESS;
}

HcclResult HcomGetAicpuOpStreamNotify(const char *group, HcclRtStream *opStream, u8 aicpuNotifyNum, void** aicpuNotify)
{
    return HCCL_SUCCESS;
}

HcclResult HcomGenerateCclOpTag(const char *opType, s64 hcomComm, const char *group, char *sTag)
{
    return HCCL_SUCCESS;
}

HcclResult HcomGetAlgExecParam(const char *tag, const char *group, u64 count, void *inputPtr, void *outputPtr,
    HcclCMDType opType, bool clearEnable, HcclDataType dataType, HcclReduceOp op, 
    void **commContext, u64 *len, u32 aivCoreLimit)
{
    return HCCL_SUCCESS;
}

GraphFrame * ValueHolder::GetCurrentFrame()
{
    return nullptr;
}


void HcomSetLaunchKernelMode(bool state){}


void HcomSetAutoTuneMode(bool autoTuneMode){}

HcclResult HcomGetServerNumAndDeviceNumPerServer(u32 *serverNum, u32 *deviceNumPerServer, u32 *deviceNumPerAggregation)
{
    return HCCL_SUCCESS;
}

HcclResult HcomGetAlgorithm(u32 level, char** algo)
{
    static std::string str ="strange";
    *algo = const_cast<char *>(str.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcomGetBandWidthPerNPU(u32 level, float *bandWidth)
{
    return HCCL_SUCCESS;
}

void HcomTopoInfoRegCallback(HcclResult (*p1)(const char *, uint32_t), void (*p2)(const char *)){}

HcclResult HcomSetProfilingMode(HcomProfilingMode profilingMode, const char *profilingOption)
{
    return HCCL_SUCCESS;
}

HcclResult HcomInitByMasterInfo(const char *masterIp, const char *masterPort,
    const char *masterDeviceId, const char *rankSize, const char *rankIp, HcomInitConfig *initConfig)
{
    return HCCL_SUCCESS;
}

HcclResult HcomDestroy(void)
{
    return HCCL_SUCCESS;
}

aclError aclrtMemcpyAsync(
    void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind, aclrtStream stream)
{
    return ACL_SUCCESS;
}

aclError aclrtMallocForTaskScheduler(void **devPtr, size_t size, aclrtMemMallocPolicy policy, aclrtMallocConfig *cfg)
{
    return aclrtMallocWithCfg(devPtr, size, policy, cfg);
}

aclError aclrtMemcpyAsyncWithOffset(void **dst, size_t destMax, size_t dstDataOffset, const void **src,
    size_t count, size_t srcDataOffset, aclrtMemcpyKind kind, aclrtStream stream)
{
    return ACL_SUCCESS;
}

rtError_t rtMemcpyAsyncWithOffset(void **dst, uint64_t dstMax, uint64_t dstDataOffset, const void **src,
    uint64_t cnt, uint64_t srcDataOffset, rtMemcpyKind kind, rtStream_t stm)
{
    return RT_ERROR_NONE;
}

rtError_t rtFftsPlusTaskLaunchWithFlag(const rtFftsPlusType_t *const fftsPlusTaskInfo, const void *const stm,
                                       const uint32_t flag) {
  return ACL_SUCCESS;
}

aclError aclrtSynchronizeStream(aclrtStream stream)
{
    return ACL_SUCCESS;
}

aclError aclrtCreateStream(aclrtStream *stream)
{
    rtStreamCreate(stream, 0);
    return ACL_SUCCESS;
}

aclError aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag)
{
    return aclrtCreateStream(stream);
}


HcclResult HcomSetGlobalWorkSpace(const char *group, void **globalWorkSpaceAddr, u32 len)
{
    return HCCL_SUCCESS;
}

HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType,
    HcclComm comm, aclrtStream stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclAllGatherV(void *sendBuf, uint64_t sendCount, void *recvBuf,
    const void *recvCounts, const void *recvDispls, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm,
    aclrtStream stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclReduceScatterV(void *sendBuf, const void *sendCounts, const void *sendDispls,
    void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclSend(void* sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank,
                           HcclComm comm, aclrtStream stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
                             HcclReduceOp op, uint32_t root, HcclComm comm, aclrtStream stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclRecv(void* recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank,
                           HcclComm comm, aclrtStream stream)
{
    return HCCL_SUCCESS;
}

INT32 mmDladdr(VOID *addr, mmDlInfo *info) {
  int ret = dladdr(addr, (Dl_info *)info);
  if (ret != -1){
    return 0;
  }
  return -1;
}

HcclResult HcomSetAttachedStream(const char *group, u32 graphId, const rtStream_t *stream, s32 len)
{
    return HCCL_SUCCESS;
}
HcclResult HcomSelectAlg(s64 comm, const char *group, u64 count, void* counts,
    HcclDataType dataType, HcclReduceOp op, HcclCMDType opType, int32_t aivCoreLimit,
    bool &ifAiv, char *algName)
{
    return HCCL_SUCCESS;
}
HcclResult HcomCalcAivCoreNum(const char *group, HcclCMDType opType, u64 count, void* counts, HcclDataType dataType, int32_t aivCoreLimit,
        char *algName, u32 *blockDim)
{
    return HCCL_SUCCESS;
}
HcclResult HcomGetCommCCLBufferSize(const char *group, uint64_t &size)
{
    return HCCL_SUCCESS;
}
HcclResult HcclGetInstTopoTypeByNetLayer(HcclComm comm, uint32_t netLayer, CommTopo *topoType)
{
    return HCCL_SUCCESS;
}
HcclResult HcclGetInstSizeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t *rankNum)
{
    return HCCL_SUCCESS;
}
HcclResult HcclGetNetLayers(HcclComm comm, uint32_t **netLayers, uint32_t *netLayerNum)
{
    return HCCL_SUCCESS;
}

aclError aclsysGetVersionStr(CHAR *pkgName, char *versionStr)
{
 	return RT_ERROR_NONE;
}

#ifdef __cplusplus
}
#endif // __cplusplus


