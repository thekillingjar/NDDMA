/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <runtime/rt.h>
#include <hccl/base.h>
#include <hccl/hccl_types.h>
#include <securec.h>
#include <unistd.h>
#include <signal.h>
#include <syscall.h>
#include <sys/prctl.h>
#include <syslog.h>
#include <sys/mman.h>
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */
#include <arpa/inet.h>
#include <time.h>
#include <runtime/base.h>
#include <runtime/dev.h>
#include <map>
#include <stdlib.h>
#include <set>
#include <mutex>
#include <utility>
#include "llt_hccl_stub.h"
#include "hccl_stub.h"
#include "llt_hccl_stub_fp16.h"
#include "hcom_acl_adapter.h"
#include <runtime/kernel.h>
#include "mmpa_api.h"
#include "acl/acl_rt.h"

/*----------------------------------------------*
 * 外部变量说明                                 *
 *----------------------------------------------*/

/*----------------------------------------------*
 * 外部函数原型说明                             *
 *----------------------------------------------*/

/*----------------------------------------------*
 * 内部函数原型说明                             *
 *----------------------------------------------*/
/*----------------------------------------------*
 * 全局变量                                     *
 *----------------------------------------------*/
/*记录当前操作的设备*/
__thread int32_t current_dev;
__thread int32_t current_die;
__thread rtDeviceMode current_devMode;

/*记录当前操作的设备*/
u32 gBoardId;
uint32_t gDevPhyId;
u32 gIsVM = 0;  //当前是否为虚拟机
static s32 chip_type_stub[256] = {0}; /*最大为16，下面不再做判断*/

/*----------------------------------------------*
 * 模块级变量                                   *
 *----------------------------------------------*/

/*----------------------------------------------*
 * 常量定义                                     *
 *----------------------------------------------*/

/*----------------------------------------------*
 * 宏定义                                       *
 *----------------------------------------------*/

static s32 stub_log_level = DLOG_ERROR;
static u32 FailureDeviceId = 0xFFFFFFFF;
static tasktype_e FailureTaskType = TASK_TYPE_RESERVED;
static std::mutex taskFailCallbackMapMutex;
static std::mutex taskAbortCallbackMapMutex;
static std::mutex isExecutedMutex;
static std::map<string, rtTaskFailCallback> taskFailCallbackMap;
static std::map<string, rtTaskAbortCallBack> taskAbortCallbackMap;
s32 log_level_get_stub(){
    return stub_log_level;
}
void log_level_set_stub(s32 log_level){
	stub_log_level =log_level;
}
string get_log_str_from_type_stub(s32          type)
{
    string str = "";
    switch (type) {
        case DLOG_DEBUG:
            str = "[DEBUG]";
            break;
        case DLOG_INFO:
            str = "[INFO]";
            break;
        case DLOG_WARN:
            str = "[WARNING]";
            break;
        case DLOG_ERROR:
            str = "[ERROR]";
            break;
        // case DLOG_EVENT:
        //     str = "[EVENT]";
        //     break;
        default:
            break;
    }
    return str;
}
using DevicePlaneInfo_t = struct DevicePlaneInfo {
    s32 devicePhyId;                    // 服务器内device唯一标识
    s32 planeId;
    bool operator == (const s32 &devicePhyId){
        return (this->devicePhyId == devicePhyId);
    }
};
std::vector<DevicePlaneInfo_t> DevicePlaneList;
// 记录日志时获取当前时间字符串
HcclResult sal_get_current_time(char *timeStr, u32 len)
{
    struct timeval tv;
    time_t tmpt;
    struct tm *now;

    if (timeStr == NULL) {
        return HCCL_E_PARA;
    }

    if (0 > gettimeofday(&tv, NULL)) {
        return HCCL_E_INTERNAL;
    }

    tmpt = (time_t)tv.tv_sec;
    now = localtime(&tmpt);
    if (now == NULL) {
        return HCCL_E_INTERNAL;
    }

    int iLen = snprintf_s(timeStr, len, len, "%04d-%02d-%02d %02d:%0d:%02d.%06u",\
        now->tm_year + TIME_FROM_1900,
        now->tm_mon + 1, now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec, (u32)tv.tv_usec);
    if (iLen == -1) {
        HCCL_WARNING("Print time failed[%d]." \
            "params: time[%s], len[%u], time_format:%04d-%02d-%02d %02d:%02d:%02d.%06u",\
            iLen, timeStr, len, now->tm_year + TIME_FROM_1900, now->tm_mon + 1, now->tm_mday,
            now->tm_hour, now->tm_min, now->tm_sec, (u32)tv.tv_usec);
    }

    return HCCL_SUCCESS;
}

int CheckLogLevel(int moduleId, int level)
{
    return 1;
}

int dlog_getlevel(int moduleId, int *enableEvent)
{
    return 0;
}

int dlog_setlevel(int moduleId, int level, int enableEvent)
{
    return 0;
}

/**
 * @ingroup dvrt_stream
 * @brief create stream instance
 * @param [in|out] stream   created stream
 * @param [in] priority   stream priority
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input stream handle
 * @return ACL_ERROR_RT_PARAM_INVALID for error input priority
 */
rtError_t rtStreamCreate(rtStream_t* stream, int32_t priority)
{
    // Mod for optimize runtime Stub by l on 2018-01-11 Below
    stream_class* rtstream;
    s32 device_id;
    aclrtGetDevice(&device_id);
    if ((rtstream = new(std::nothrow) stream_class(device_id)) == nullptr) {
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }
    // 记录当前设备信息
    rtstream->current_dev = current_dev;

    *stream = (rtStream_t)rtstream;
    // Mod for optimize runtime Stub by l on 2018-01-11 Above
    return RT_ERROR_NONE;
}
/**
 * @ingroup dvrt_base
 * @brief register callback for fail task
 * @param [in] uniName unique register name, can't be null
 * @param [in] callback fail task callback function
 * @param [out] NA
 * @return RT_ERROR_NONE for ok
 */
aclError aclrtSetExceptionInfoCallback(aclrtExceptionInfoCallback callback)
{
    string tmpStr = "ASCENDCL";
    std::unique_lock<std::mutex> lock(taskFailCallbackMapMutex);
    taskFailCallbackMap.clear();
    taskFailCallbackMap.insert({tmpStr, callback});
    return ACL_SUCCESS;
}
rtError_t rtSetTaskAbortCallBack(const char *moduleName, rtTaskAbortCallBack callback, void *args)
{
    string tmpStr(moduleName);
    void *p = nullptr;
    std::unique_lock<std::mutex> lock(taskAbortCallbackMapMutex);
    taskAbortCallbackMap.clear();
    taskAbortCallbackMap.insert({tmpStr, callback});
    return RT_ERROR_NONE;
}
rtError_t rtResourceClean(int32_t devId, rtIdType_t type)
{
    return RT_ERROR_NONE;
}
rtError_t TaskFailCallbackClean()
{
    taskFailCallbackMap.clear();
    return RT_ERROR_NONE;
}
/**
 * @ingroup dvrt_stream
 * @brief inquire max stream count and max task count per stream
 * @param [in] streamType   Stream Type
 * @param [in] MaxStrCount   Max stream count
 * @param [in] MaxTaskCount   max task count per stream
 * @return RT_ERROR_NONE for complete
 * @return RT_ERROR_INVALID_VALUE for error input
 */
rtError_t rtGetMaxStreamAndTask(uint32_t streamType, uint32_t *maxStrCount, uint32_t *maxTaskCount)
{
    *maxStrCount = 1024;
    *maxTaskCount = 1024;
    return RT_ERROR_NONE;
}
/**
 * @ingroup dvrt_stream
 * @brief create stream instance
 * @param [in|out] stream   created stream
 * @param [in] priority   stream priority
 * @param [in] flags  stream op flags
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input stream handle
 * @return ACL_ERROR_RT_PARAM_INVALID for error input priority
 */
rtError_t rtStreamCreateWithFlags(rtStream_t * stream, int32_t priority, uint32_t flags)
{
    return rtStreamCreate(stream, priority);
}

aclError aclrtSetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType, aclrtStreamAttrValue *value)
{
    return ACL_SUCCESS;
}

aclError aclrtGetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType, aclrtStreamAttrValue *value)
{
    return ACL_SUCCESS;
}

std::mutex g_tidStreamMapMutex;
std::map<uint64_t, std::vector<void *>> g_tidStreamMap;

/**
 * @ingroup dvrt_stream
 * @brief destroy stream instance.
 * @param [in] stream   the stream to destroy
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input stream handle
 */
rtError_t rtStreamDestroy(rtStream_t stream)
{
    // Mod for optimize runtime Stub by l on 2018-01-11 Below
    stream_class* rtstream = NULL;
    rtstream = (stream_class*)stream;
    if (NULL != rtstream)
    {
        delete rtstream;
        rtstream = NULL;
    }
    std::unique_lock<std::mutex> lock(g_tidStreamMapMutex);
    g_tidStreamMap.clear();
    lock.unlock();

    // Mod for optimize runtime Stub by l on 2018-01-11 Above

    return RT_ERROR_NONE;
}

/*add ipc stub optimize*/
std::map<void*, u32> g_ipcBasePtrMap;
std::mutex g_ipcMtx;
void rtIpcBasePtrAdd(void *ptr,u32 size)
{
    std::lock_guard<std::mutex> lock(g_ipcMtx);
    g_ipcBasePtrMap[ptr] = size;
    return ;
}
void rtIpcBasePtrErase(void *ptr)
{
    std::lock_guard<std::mutex> lock(g_ipcMtx);
    g_ipcBasePtrMap.erase(ptr);
    return ;
}

void* rtIpcBasePtrLookup(const void *ptr)
{
    std::lock_guard<std::mutex> lock(g_ipcMtx);
    for (const auto& pair : g_ipcBasePtrMap) {
        if ((ptr >=pair.first) && (static_cast<const u8*>(ptr) < static_cast<u8*>(pair.first) + pair.second)) {
            return pair.first;
        } 
    }
    return nullptr;
}
std::map<void*, u32> g_ipcOpenBasePtrMap;
std::mutex g_openIpcMtx;
void rtIpcOpenBasePtrAdd(void *ptr,u32 size)
{
    std::lock_guard<std::mutex> lock(g_openIpcMtx);
    g_ipcOpenBasePtrMap[ptr] = size;
    return ;
}
void rtIpcOpenBasePtrErase(void *ptr)
{
    std::lock_guard<std::mutex> lock(g_openIpcMtx);
    g_ipcOpenBasePtrMap.erase(ptr);
    return ;
}

void* rtIpcOpenBasePtrLookup(const void *ptr)
{
    std::lock_guard<std::mutex> lock(g_openIpcMtx);
    for (const auto& pair : g_ipcOpenBasePtrMap) {
        if ((ptr >=pair.first) && (static_cast<const u8*>(ptr) < static_cast<u8*>(pair.first) + pair.second)) {
            return pair.first;
        } 
    }
    return nullptr;
}

aclError aclrtDestroyStreamForce(aclrtStream stream)
{
    return rtStreamDestroy(stream);
}

/**
 * @ingroup dvrt_stream
 * @brief get stream id from a stream handle
 * @param [in] stream   stream hadle
 * @param [in] streamId   stream id
 * @return RT_ERROR_NONE for complete
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input stream handle
 */
aclError aclrtStreamGetId(aclrtStream stream, int32_t *streamId)
{
    static int32_t stream_id_counter = 0;
    static std::map<rtStream_t, int32_t> streamMap;
    static std::mutex mapMutex;

    if (streamId == nullptr || stream == nullptr) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    stream_class* rtstream = (stream_class*)stream;
    *streamId = rtstream->get_stream_id();

    return ACL_SUCCESS;
}

/* 记录task_info*/
thread_local task_info_t task_info;

/**
 * @ingroup dvrt_base
 * @brief get current thread last stream id and task id
 * @param [out] stream id and task id
 * @param [in] null
 * @return RT_ERROR_NONE for ok
 * @return ACL_ERROR_RT_PARAM_INVALID for input null ptr
 */
rtError_t rtGetTaskIdAndStreamID(uint32_t *taskId, uint32_t *streamId)
{
    *streamId = task_info.streamId;
    *taskId = task_info.taskId;
    return RT_ERROR_NONE;
}


//Add for optimize runtime Stub by l on 2018-01-11 Below
/**
 * @ingroup dvrt_stream
 * @brief wait stream to be complete
 * @param [in] stream   stream to wait
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input stream or event handle
 */
rtError_t rtStreamSynchronize(rtStream_t stream)
{
    stream_class* rtstream = NULL;
    rtstream = (stream_class*)stream;
    rtstream->stream_synchronize();
    return RT_ERROR_NONE;
}

rtError_t rtStreamSynchronizeWithTimeout(rtStream_t stream, int32_t timeout) {
    return rtStreamSynchronize(stream);
}

static u32 modelCallBackStub = 0; // 0:host网卡 1：HVD
aclError aclrtActiveStream(aclrtStream activeStream, aclrtStream stream)
{
    return ACL_SUCCESS;
}

int rtModelFake = 0;
rtError_t rtStreamGetCaptureInfo(rtStream_t stm, rtStreamCaptureStatus *status,
                                 rtModel_t *captureMdl)
{   
    *captureMdl = &rtModelFake;
    return RT_ERROR_NONE;
}

rtError_t rtModelGetId(rtModel_t mdl, uint32_t *modelId){
    return RT_ERROR_NONE;
}

rtError_t rtStreamAddToModel(rtStream_t stm, rtModel_t captureMdl)
{
    captureMdl = &rtModelFake;
    return RT_ERROR_NONE;
}

void SetRtCallbackModleStub(u32 model)
{
    modelCallBackStub = model;
}

// std::map<uint64_t, std::vector<void *>> g_tidStreamMap;

aclError aclrtSubscribeReport(uint64_t threadId, aclrtStream stream)
{
    if(modelCallBackStub != 0) {
        return ACL_SUCCESS;
    }
    std::unique_lock<std::mutex> lock(g_tidStreamMapMutex);
    g_tidStreamMap[threadId].push_back(stream);
    lock.unlock();
    return ACL_SUCCESS;
}

aclError aclrtProcessReport(int32_t timeout)
{
    if(modelCallBackStub != 0) {
        return ACL_SUCCESS;
    }
    uint64_t threadId = pthread_self();
    auto iter = g_tidStreamMap.find(threadId);
    if (iter == g_tidStreamMap.end()) {
        HCCL_ERROR("this thread id[%llu] has not been registered", threadId);
        return 1;
    }
    for (auto stream : g_tidStreamMap[threadId]) {
        stream_class* rtstream = (stream_class*)stream;
        if (rtstream->stream_task_list.front().task_type == TASK_TYPE_CALLBACK_FUNC &&
            rtstream->stream_task_list.front().stream_para.callbackTask.isExecuted == 0) {
            rtstream->ExecuteCallbackFunc();
        } else {
            SaluSleep(timeout*10);
        }
    }
    return ACL_SUCCESS;
}

aclError aclrtUnSubscribeReport(uint64_t threadId, aclrtStream stream)
{
    size_t tmpVectorSize = g_tidStreamMap[threadId].size();
    for (size_t i = 0; i < tmpVectorSize; ++i) {
        if (g_tidStreamMap[threadId][i] == stream) {
            std::unique_lock<std::mutex> lock(g_tidStreamMapMutex);
            g_tidStreamMap[threadId].erase(g_tidStreamMap[threadId].begin() + i);
            lock.unlock();
            HCCL_INFO("llt rt UnSubscribe Report success thread[%llu], stream[%p]", threadId, stream);
            return ACL_SUCCESS;
        }
    }
    HCCL_INFO("llt rt UnSubscribe Report success thread[%llu] stream[%p] tmpVectorSize[%d]", threadId, stream, tmpVectorSize);
    return ACL_SUCCESS;
}

aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event)
{
    return ACL_SUCCESS;
}

aclError aclrtResetEvent(aclrtEvent event, aclrtStream stream)
{
    return ACL_SUCCESS;
}

rtError_t rtEventCreateWithFlag(rtEvent_t* event, uint32_t flag)
{
    return aclrtCreateEvent(event);
}
/**
 * @ingroup dvrt_stream
 * @brief switch to the corresponding stream according to the contents of the ptr
 * @param [in] ptr  Determine the address where the value of the true and false branches is located
 * @param [in] condition switch condition
 * @param [in] value  switch value
 * @param [in] true_stream  Stream that needs to be activated when the value is non-zero
 * @param [in] stream input stream to init task
 * @return RT_ERROR_NONE for complete
 * @return ACL_ERROR_RT_PARAM_INVALID for error input
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for invalid resource handle
 * @return RT_ERROR_INVALID_DEVICE for invalid device handle
 * @return ERROR_RECYCLE for switching task init failed or submit failed
 */
rtError_t rtStreamSwitchEx(void *ptr,  rtCondition_t condition, void *value_ptr,
                           rtStream_t true_stream, rtStream_t stream, rtSwitchDataType_t dataType)
{
    stream_class* rtstream = NULL;
    rtstream = (stream_class*)true_stream;
    stream_class* rtstreamMain = (stream_class*)stream;
    s64 mem_value;
    s64 mem_value2;
    if (dataType == RT_SWITCH_INT32) {
        mem_value = *static_cast<s32*>(ptr);
        mem_value2 = *static_cast<s32*>(value_ptr);
    } else {
        mem_value = *static_cast<s64*>(ptr);
        mem_value2 = *static_cast<s64*>(value_ptr);
    }
    switch (condition) {
        case RT_EQUAL :
            if (mem_value == mem_value2) {
                rtstream->set_stream_enabled(true);
                rtstreamMain->set_stream_enabled(false);
            }
            else {
                rtstream->set_stream_enabled(false);
                rtstreamMain->set_stream_enabled(true);
            }
            break;
        case RT_NOT_EQUAL:
            if (mem_value != mem_value2) {
                rtstream->set_stream_enabled(true);
                rtstreamMain->set_stream_enabled(false);
            }
            else {
                rtstream->set_stream_enabled(false);
                rtstreamMain->set_stream_enabled(true);
            }
            break;
        default:
            break;
    }
    return RT_ERROR_NONE;
}


rtError_t rtModelBindStream(rtModel_t model, rtStream_t stream, uint32_t flag)
{
    return RT_ERROR_NONE;
}


rtError_t rtModelUnbindStream(rtModel_t model, rtStream_t stream)
{
    return RT_ERROR_NONE;
}

rtError_t rtModelExecute(rtModel_t model, rtStream_t stream, uint32_t flag)
{
    return RT_ERROR_NONE;
}

rtError_t rtModelCreate(rtModel_t *model, uint32_t flag)
{
    *model = (rtModel_t)1;
    return RT_ERROR_NONE;
}
rtError_t rtGetDeviceCount(int32_t* count);


/*****************************************************************************
 函 数 名  : rtGetDeviceCount
 功能描述  : 获取设备数量
 输入参数  : int32_t *count
 输出参数  : 无
 返 回 值  : rtError_t
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2017年12月9日
    作    者   : p00335137
    修改内容   : 新生成函数

*****************************************************************************/
rtError_t rtGetDeviceCount( int32_t* count )
{
    /*打桩函数先默认设备上芯片数量为8*/
    *count = 8;

    return RT_ERROR_NONE;
}

rtError_t rtSetDeviceV2(int32_t device, rtDeviceMode deviceMode)
{
    current_dev = device;
    current_devMode = deviceMode;
    return RT_ERROR_NONE;
}

/*****************************************************************************
 函 数 名  : aclrtSetDevice
 功能描述  : 设置当前操作的设备
 输入参数  : int32_t device
 输出参数  : 无
 返 回 值  : aclError
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2017年12月7日
    作    者   : p00335137
    修改内容   : 新生成函数

*****************************************************************************/
aclError aclrtSetDevice( int32_t device )
{
    current_dev = device;
    return ACL_SUCCESS;
}

rtError_t rtSetDie( int32_t die )
{
    current_die = die;

    return RT_ERROR_NONE;
}

/*****************************************************************************
 函 数 名  : aclrtResetDevice
 功能描述  : 设置当前操作的设备
 输入参数  : int32_t deviceId
 输出参数  : 无
 返 回 值  : aclError
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2019年05月09日
    作    者   : w00500539
    修改内容   : 新生成函数

*****************************************************************************/
aclError aclrtResetDevice(int32_t deviceId)
{
    current_dev = deviceId;

    return ACL_SUCCESS;
}

/*****************************************************************************
 函 数 名  : aclrtGetDevice
 功能描述  : 查询当前操作的设备
 输入参数  : int32_t *device
 输出参数  : 无
 返 回 值  : aclError
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2017年12月7日
    作    者   : p00335137
    修改内容   : 新生成函数

*****************************************************************************/
aclError aclrtGetDevice( int32_t* device )
{
    *device = current_dev;
    return ACL_SUCCESS;
}

rtError_t rtGetDeviceMode(rtDeviceMode *deviceMode)
{
    *deviceMode = current_devMode;
    return RT_ERROR_NONE;
}

aclError aclrtGetCurrentContext(aclrtContext *ctx)
{
    void* tmp;
    *ctx = tmp;
    return ACL_SUCCESS;
}

aclError aclrtSetCurrentContext(aclrtContext ctx)
{
    return ACL_SUCCESS;
}

rtError_t rtGetPairDevicesInfo(uint32_t devId, uint32_t otherDevId, int32_t infoType, int64_t *value)
{
    if (chip_type_stub[0] == static_cast<s32>(DevType::DEV_TYPE_910B)) {

        if ((devId / 8)  != (otherDevId / 8)) {
            *value = 1; // PXI
        } else {
            *value = 0; // HCCS
        }
    }
    if (chip_type_stub[0] == static_cast<s32>(DevType::DEV_TYPE_910_93))
    {
        // 0-1 2-3 4-5 6-7
        if ((abs(static_cast<int32_t>(devId - otherDevId)) == 1) && ((devId + otherDevId) % 4 == 1)) {
            *value = 5;     // SIO
        } else {
            *value = 6;     // HCCS_SW
        }
        return RT_ERROR_NONE;
    }

    if ( (gBoardId == 0x1E ||  gIsVM == 1 ) || (devId / 4)  != (otherDevId / 4)) // 若当前为标卡/虚拟机/非同一clustor
    {
        *value = 1; // PXI
    } else {
        *value = 0; // HCCS
    }

    return RT_ERROR_NONE;
}

rtError_t rtGetPairPhyDevicesInfo(uint32_t devId, uint32_t otherDevId, int32_t infoType, int64_t *value)
{
    if (chip_type_stub[0] == static_cast<s32>(DevType::DEV_TYPE_910B)) {

        if ((devId / 8)  != (otherDevId / 8)) {
            *value = 1; // PXI
        } else {
            *value = 0; // HCCS
        }
    }
    if (chip_type_stub[0] == static_cast<s32>(DevType::DEV_TYPE_910_93))
    {
        // 0-1 2-3 4-5 6-7
        if ((abs(static_cast<int32_t>(devId - otherDevId)) == 1) && ((devId + otherDevId) % 4 == 1)) {
            *value = 5;     // SIO
        } else {
            *value = 6;     // HCCS_SW
        }
        return RT_ERROR_NONE;
    }

    if ( (gBoardId == 0x1E ||  gIsVM == 1 ) || (devId / 4)  != (otherDevId / 4)) // 若当前为标卡/虚拟机/非同一clustor
    {
        *value = 1; // PXI
    } else {
        *value = 0; // HCCS
    }

    return RT_ERROR_NONE;
}

rtError_t rtGetDeviceSatMode(rtFloatOverflowMode_t * floatOverflowMode)
{
    *floatOverflowMode = RT_OVERFLOW_MODE_INFNAN;
    return RT_ERROR_NONE;
}

aclError aclrtSetDeviceSatMode(aclrtFloatOverflowMode mode)
{
    return ACL_SUCCESS;
}

/*****************************************************************************
 函 数 名  : rtMemcpyAsync
 功能描述  : 异步内存拷贝
 输入参数  : void *dst
             void *src
             uint64_t count
             rtMemcpyKind_t kind
             rtStream_t stream
 输出参数  : 无
 返 回 值  : rtError_t
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2017年12月7日
    作    者   : p00335137
    修改内容   : 新生成函数

*****************************************************************************/
rtError_t rtMemcpyAsync( void* dst, uint64_t destMax,const void* src, uint64_t count, rtMemcpyKind_t kind, rtStream_t stream )
{
    /*拷贝类型检测*/
    if ((kind < RT_MEMCPY_HOST_TO_HOST) || (kind > RT_MEMCPY_DEVICE_TO_DEVICE))
    {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    if ((!dst) || (!src))
    {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    /*桩函数中不管kind，都统一使用memcpy完成*/
    // Mod for optimize runtime Stub by l on 2018-01-11 Below
    // 将asynchronous Memcpy任务，压入任务队列
    stream_class* rtstream = nullptr;
    rtstream = (stream_class*)stream;

    stream_task_t stream_task;
    stream_task.task_type = TASK_TYPE_MEMCPY;
    stream_task.stream_para.memcpystruct.dst = dst;
    stream_task.stream_para.memcpystruct.src = (void*)src;
    stream_task.stream_para.memcpystruct.count = count;

    rtstream->push_task(&stream_task);
    // Mod for optimize runtime Stub by l on 2018-01-11 Above

    return RT_ERROR_NONE;
}

rtError_t rtMemcpyAsyncWithoutCheckKind( void* dst, uint64_t destMax,const void* src, uint64_t count, rtMemcpyKind_t kind, rtStream_t stream )
{
    /*拷贝类型检测*/
    if ((kind < RT_MEMCPY_HOST_TO_HOST) || (kind > RT_MEMCPY_DEVICE_TO_DEVICE))
    {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    if ((!dst) || (!src))
    {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    /*桩函数中不管kind，都统一使用memcpy完成*/
    // Mod for optimize runtime Stub by l on 2018-01-11 Below
    // 将asynchronous Memcpy任务，压入任务队列
    stream_class* rtstream = nullptr;
    rtstream = (stream_class*)stream;

    stream_task_t stream_task;
    stream_task.task_type = TASK_TYPE_MEMCPY;
    stream_task.stream_para.memcpystruct.dst = dst;
    stream_task.stream_para.memcpystruct.src = (void*)src;
    stream_task.stream_para.memcpystruct.count = count;

    rtstream->push_task(&stream_task);
    // Mod for optimize runtime Stub by l on 2018-01-11 Above

    return RT_ERROR_NONE;
}

rtError_t rtMemcpyD2DAddrAsync(void *dst, uint64_t destMax, uint64_t destOffset, const void *src, uint64_t count,
                            uint64_t srcOffset, rtStream_t stream)
{

    std::cout << "rtMemcpyD2DAddrAsync rtStream_t: " << stream << std::endl; // TODO
    if ((!dst) || (!src))
    {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    stream_class* rtstream = nullptr;
    rtstream = (stream_class*)stream;

    std::cout << "rtMemcpyD2DAddrAsync stream_class*: " << rtstream << std::endl; // TODO

    stream_task_t stream_task;
    stream_task.task_type = TASK_TYPE_MEMCPY;
    stream_task.stream_para.memcpystruct.dst = dst;
    stream_task.stream_para.memcpystruct.src = (void*)src;
    stream_task.stream_para.memcpystruct.count = count;

    rtstream->push_task(&stream_task);

    return RT_ERROR_NONE;
}


rtError_t rtNotifyCreate(int32_t device, rtNotify_t *notify)
{
    return RT_ERROR_NONE;
}


/*****************************************************************************
 函 数 名  : aclrtCreateEvent
 功能描述  : 创建事件
 输入参数  : aclrtEvent *event
 输出参数  : 无
 返 回 值  : aclError
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2017年12月7日
    作    者   : p00335137
    修改内容   : 新生成函数

*****************************************************************************/
aclError aclrtCreateEvent(aclrtEvent *event)
{
    /*
    1、原来的event机制需要使用者确保先下发record然后再下发wait。新的event机制没有这种限制，与notify流程类似。故打桩直接使用notify机制
    2、event为软件资源，与device无关。为了复用notify已有机制，不关注device id默认填0。*/
    return rtNotifyCreate(0, (rtNotify_t *)event);
}

aclError aclrtGetEventId(aclrtEvent event, uint32_t *eventId)
{
    *eventId = 0;
    return ACL_SUCCESS;
}

/*****************************************************************************
 函 数 名  : aclrtDestroyEvent
 功能描述  : 销毁事件
 输入参数  : rtEvent_t event
 输出参数  : 无
 返 回 值  : rtError_t
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2017年12月7日
    作    者   : p00335137
    修改内容   : 新生成函数

*****************************************************************************/
rtError_t aclrtDestroyEvent( rtEvent_t event )
{
    return aclrtDestroyNotify((rtNotify_t)event);
}

aclError aclrtDestroyNotify(aclrtNotify notify)
{
    return RT_ERROR_NONE;
}
/*****************************************************************************
 函 数 名  : aclrtRecordEvent
 功能描述  : 记录事件
 输入参数  : aclrtEvent event
             aclrtStream stream
 输出参数  : 无
 返 回 值  : aclError
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2017年12月7日
    作    者   : p00335137
    修改内容   : 新生成函数

*****************************************************************************/
aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream)
{
    /*异步下发record后返回*/
    return aclrtRecordNotify((aclrtNotify)event, stream);
}

aclError aclrtRecordNotify(aclrtNotify notify, aclrtStream stream)
{
    return ACL_SUCCESS;
}

/*阻塞查询event是否被record*/
aclError aclrtQueryEventStatus(aclrtEvent event, aclrtEventRecordedStatus *status)
{
    rt_notify_t* ipc_notify = (rt_notify_t*)event;
    rt_shm_notify_t* notify_shm = (rt_shm_notify_t*)ipc_notify->ipc_notify_shm;
    if (nullptr == notify_shm) {
        HCCL_ERROR("parameter error : notify_shm[%p]", notify_shm);
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    HCCL_DEBUG("wait : device[%d], notify_id[%llu]", notify_shm->device_id, ipc_notify->notify_id);

    s32 timeout_cnt = NOTIFY_TIMEOUT_CNT;//20s
    while (!__sync_bool_compare_and_swap(&(notify_shm->record_cnt[ipc_notify->notify_id]), 1, 0)) {
        SaluSleep(1000);

        timeout_cnt--;
        if (timeout_cnt <= 0) {
            HCCL_ERROR("wait timeout : record_cnt[%d], device_id[%d], notify_id[%llu]",
                notify_shm->record_cnt[ipc_notify->notify_id],
                notify_shm->device_id,
                ipc_notify->notify_id);
            return ACL_ERROR_RT_PARAM_INVALID;
        }
    }

    *status = ACL_EVENT_RECORDED_STATUS_COMPLETE;
    return  ACL_SUCCESS;
}

aclError aclrtNotifyBatchReset(aclrtNotify *notifies, size_t num)
{
    return ACL_SUCCESS;
}

rtError_t rtStreamClear(rtStream_t stm, rtClearStep_t step)
{
    return  RT_ERROR_NONE;
}

s32 sal_close_name_map(void* name_map)
{
    /* 通过共享内存名称在本进程关闭name_map共享设备内存 */
    (void)sal_share_memory_destroy((void*)name_map);

    return SAL_OK;
}

s32 sal_create_name_map(const char* name, rt_name_map_stub_t** name_map)
{
    s32 ret;

    /* 名字长度不能超标 */
    if (SalStrLen(name) > SAL_DMEM_NAME_MAX_BYTES)
    {
        HCCL_ERROR("length of name is [%d], out of range", SalStrLen(name));
        return SAL_E_PARA;
    }

    /* 将名字添加前缀，防止该名字在别的共享内存已经使用了 */
    char mapped_name[SAL_DMEM_UNIQUE_ID_BYTES] = {0};
    ret = snprintf_s(mapped_name, SAL_DMEM_UNIQUE_ID_BYTES, SAL_DMEM_UNIQUE_ID_BYTES - 1,
                            "%s%s", SAL_DMEM_UNIQUE_ID_PREFIX, name);
    if (-1 == ret)
    {
        HCCL_ERROR("snprintf_s failed[%d]", ret);
        return SAL_E_MEMORY;
    }

    // 创建或打开name_map映射表
    rt_name_map_stub_t* name_map_ptr =
        (rt_name_map_stub_t*)sal_share_memory_create(mapped_name, (sizeof(rt_name_map_stub_t)));

    if ( NULL == name_map_ptr )
    {
        HCCL_ERROR("create share memory %s failed", mapped_name);
        return SAL_E_PARA;
    }

    *name_map = name_map_ptr;

    return SAL_OK;
}

/*****************************************************************************
 函 数 名  : rtMalloc
 功能描述  : 设备内存申请
 输入参数  : void **devPtr
             uint64_t size
             rtMemType_t type
             uint16_t moudleId
 输出参数  : 无
 返 回 值  : rtError_t
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2017年12月7日
    作    者   : p00335137
    修改内容   : 新生成函数
  2.日    期   : 2018年6月12日
    作    者   : liubanglan
    修改内容   : 桩函数中rtMalloc申请共享内存，以便rtIpcSetMemoryName等跨进程操作

*****************************************************************************/
rtError_t rtMalloc( void** devPtr, uint64_t size, rtMemType_t type , uint16_t moudleId)
{
    char my_unique_id[SAL_UNIQUE_ID_BYTES];
    char mem_name[SAL_DMEM_UNIQUE_ID_BYTES];

    if (!devPtr)
    {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    if (RT_MEMORY_RESERVED <= type)
    {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    if (moudleId != HCCL)
    {
        return ACL_ERROR_RT_PARAM_INVALID;
    }
    /* 通过uniqueid的方式，申请桩函数设备内存的名字 */
    s32 ret = SalGetUniqueId(my_unique_id);

    if (ret != SAL_OK)
    {
        HCCL_ERROR("Generate sal_unique_id failed[%d]", ret);
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    /* UniqueID添加HCCL-mem-的前缀 */
    ret = snprintf_s(mem_name, SAL_DMEM_UNIQUE_ID_BYTES, SAL_DMEM_UNIQUE_ID_BYTES - 1,
                            "%s%s", SAL_DMEM_UNIQUE_ID_PREFIX, my_unique_id);

    if (-1 == ret)
    {
        HCCL_ERROR("snprintf_s failed[%d]", ret);
        return ACL_ERROR_RT_CONTEXT_NULL;
    }

    /* 封装的共享内存结构自带了管理信息,返回值是管理信息后的有效存储空间 */
    void* shm_buf = (void*)sal_share_memory_create(mem_name, size);

    if (NULL == shm_buf)
    {
        HCCL_ERROR("sal_share_memory_create failed, device_memory_share_info is NULL");
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }

    HCCL_DEBUG("rtMalloc sal_share_memory_create[%s] OK", mem_name);

    *devPtr = shm_buf;

    rtIpcBasePtrAdd(shm_buf, size);
    return RT_ERROR_NONE;
}


/*****************************************************************************
 函 数 名  : aclrtFree
 功能描述  : 设备内存释放
 输入参数  : void *devPtr
 输出参数  : 无
 返 回 值  : rtError_t
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2017年12月7日
    作    者   : p00335137
    修改内容   : 新生成函数
  2.日    期   : 2018年6月26日
    作    者   : liubanglan
    修改内容   : 桩函数中rtFree释放共享内存，以便rtIpcSetMemoryName等跨进程操作

*****************************************************************************/
aclError aclrtFree( void* devPtr )
{
    if (!devPtr)
    {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    /* 通过共享内存管理信息中的关联指针找到name map */
    share_mem_t* shm_head = (share_mem_t*)((char*)devPtr - offsetof(share_mem_t, user_data));
    for (int i =0 ;i<shm_head->relate_ptr_cnt; i++) 
    {
        rt_name_map_stub_t* name_map_ptr = (rt_name_map_stub_t*)shm_head->relate_ptr[i] ; // name map的地址作为本块共享内存的关联指针
        /* 关闭该段内存的name map */
        if (nullptr != name_map_ptr)
        {
            HCCL_DEBUG("rtfree devPtr[%p] name_map_ptr[%p], ", devPtr, name_map_ptr);
            (void)sal_close_name_map(name_map_ptr);
            shm_head->relate_ptr[i] = nullptr;
        }
    }
    shm_head->relate_ptr_cnt = 0;   
    /* destroy devPtr指向的共享内存 */
    (void)sal_share_memory_destroy(devPtr);

    rtIpcBasePtrErase(devPtr);

    return ACL_SUCCESS;
}

/*
        creat和destroy share memory中的映射表
process A                                            process B
rtIpcSetMemoryName   ----    create map
                             create map       ----       rtIpcOpenMemory
                             destroy map     ----       rtIpcOpenMemory
aclrtFree               ----    destroy map

*/

/*****************************************************************************
 函 数 名  : aclrtMallocHostWithCfg
 功能描述  :
 输入参数  : void** hostPtr
            uint64_t size
            aclrtMallocConfig cfg
 输出参数  : 无
 返 回 值  :
 调用函数  :
 被调函数  :

 修改历史  :
      1. 2018年7月18日      创建

*****************************************************************************/
aclError aclrtMallocHostWithCfg(void **hostPtr, uint64_t size, aclrtMallocConfig *cfg)
{
    void* buf = NULL;

    if (!hostPtr) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    if (cfg == nullptr || cfg->attrs == nullptr) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    uint16_t moduleId = 0;
    for (uint32_t i = 0U; i < cfg->numAttrs; i++) {
        aclrtMallocAttribute* attr = &(cfg->attrs[i]);
        if (attr->attr == ACL_RT_MEM_ATTR_MODULE_ID) {
            moduleId = attr->value.moduleId;
            break;
        }
    }
    if (moduleId != HCCL) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    buf = malloc(size);

    if (buf == nullptr) {
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }

    *hostPtr = buf;

    return RT_ERROR_NONE;
}

/*****************************************************************************
 函 数 名  : aclrtFreeHost
 功能描述  :
 输入参数  : void* hostPtr
 输出参数  : 无
 返 回 值  :
 调用函数  :
 被调函数  :

 修改历史  :
      1. 2018年7月18日      创建

*****************************************************************************/
rtError_t aclrtFreeHost( void* hostPtr )
{
    if (!hostPtr)
    {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    free(hostPtr);

    return RT_ERROR_NONE;
}

bool Adx::AdumpIsDumpEnable(Adx::DumpType type)
{
    if (type == Adx::DumpType::OP_OVERFLOW) {
        return false;
    } else {
        return true;
    }
}

void Adx::AdumpPrintWorkSpace(const void *workSpaceAddr, const size_t dumpWorkSpaceSize,
                                 rtStream_t stream, const char *opType, bool enableSync)
{
    return;
}

/*****************************************************************************
 函 数 名  : rtMemAllocManaged
 功能描述  : alloc managed memory
 输入参数  : void **ptr
             uint64_t size
             uint32_t flag
 输出参数  : void **ptr
 返 回 值  : rtError_t
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2018年06月26日
    作    者   :
    修改内容   : 新生成函数

*****************************************************************************/
rtError_t rtMemAllocManaged(void** ptr, uint64_t size, uint32_t flag)
{
    void* buf = NULL;

    if (!ptr)
    {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    buf = malloc(size);

    if (NULL == buf)
    {
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }

    *ptr = buf;

    return RT_ERROR_NONE;
}


/*****************************************************************************
 函 数 名  : rtMemFreeManaged
 功能描述  : 设备内存释放
 输入参数  : void **ptr
 输出参数  : 无
 返 回 值  : rtError_t
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2018年06月26日
    作    者   :
    修改内容   : 新生成函数

*****************************************************************************/
rtError_t rtMemFreeManaged(void* ptr)
{
    if (!ptr)
    {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    free(ptr);

    return RT_ERROR_NONE;
}

/*****************************************************************************
 函 数 名  : set_chip_peer_stub
 功能描述  : 设置芯片peer类型
 输入参数  :  handler
 输出参数  : 无
 返 回 值  : -
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2018年8月7日
    作    者   : l00442453
    修改内容   : 新生成函数

*****************************************************************************/
static bool chip_peer_stub[16][16] = {0}; /*最大为16，下面不再做判断*/
void set_chip_peer_stub(s32 devId1, s32 devId2, bool can_peer)
{
    if ((devId1 >= 0) && (devId2 >= 0) && (devId1 < 16) && (devId2 < 16))
    {
        chip_peer_stub[devId1][devId2] = can_peer;
    }
    else
    {
        HCCL_ERROR("device id is illegal");
    }

}

#if 0
pid_t drvDeviceGetBarePid(void)
{
    return getpid();
}
#endif
bool g_isCommonPidMode = false;

void rtSetCommonPidMode(bool state)
{
    g_isCommonPidMode = state;
}

rtError_t rtDeviceGetBareTgid(u32 *pid)
{
    if (g_isCommonPidMode == false) {
        *pid = syscall(SYS_gettid);
    } else {
        *pid = getpid();
    }

    return RT_ERROR_NONE;
}

aclError aclrtReserveMemAddress(void **virPtr, size_t size, size_t alignment, void *expectPtr, uint64_t flags)
{
    (void)virPtr;
    (void)size;
    (void)alignment;
    (void)expectPtr;
    (void)flags;
    return ACL_SUCCESS;;
}

aclError aclrtReleaseMemAddress(void *virPtr)
{
    (void)virPtr;
    return ACL_SUCCESS;
}

rtError_t rtMallocPhysical(void **handle, size_t size, rtDrvMemProp_t *prop, uint64_t flags)
{
    // void
    (void)handle;
    (void)size;
    (void)prop;
    (void)flags;
    return RT_ERROR_NONE;
}

aclError aclrtFreePhysical(aclrtDrvMemHandle handle)
{
    (void)handle;
    return ACL_SUCCESS;
}

aclError aclrtMapMem(void *virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags)
{
    (void)virPtr;
    (void)size;
    (void)offset;
    (void)handle;
    return ACL_SUCCESS;
}

aclError aclrtUnmapMem(void *devPtr)
{
    (void)devPtr;
    return ACL_SUCCESS;
}

aclError aclrtMemExportToShareableHandle(aclrtDrvMemHandle handle, aclrtMemHandleType handleType, uint64_t flags,
    uint64_t *shareableHandle)
{
    (void)handle;
    (void)handleType;
    (void)flags;
    (void)shareableHandle;
    return ACL_SUCCESS;
}

aclError aclrtMemImportFromShareableHandle(uint64_t shareableHandle, int32_t deviceId, aclrtDrvMemHandle *handle)
{
    (void)shareableHandle;
    (void)deviceId;
    (void)handle;
    return ACL_SUCCESS;
}

aclError aclrtMemSetPidToShareableHandle(uint64_t shareableHandle, int32_t *pid, size_t pidNum)
{
    (void)shareableHandle;
    (void)pid;
    (void)pidNum;
    return ACL_SUCCESS;
}

/*****************************************************************************
 函 数 名  : drvRegisterExitHandler
 功能描述  : 注册异常退出处理函数
 输入参数  :  handler
 输出参数  : 无
 返 回 值  : drvError_t
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2018年8月7日
    作    者   : l00442456
    修改内容   : 新生成函数

*****************************************************************************/
drvError_t drvRegisterExitHandler(void (*handler)(int signum))
{
    /* 进程异常退出时的处理回调函数,接管系统信号*/
    struct sigaction sa_exit;   /*包含信号处理动作的结构体*/\
    sa_exit.sa_handler = handler; /*指定信号处理函数*/
    sigemptyset(&sa_exit.sa_mask);
    sigaction(SIGINT, &sa_exit, NULL);   /* 注册SIGINT信号, 对应 Ctrl+C */
    sigaction(SIGTERM, &sa_exit, NULL);   /* 注册SIGINT信号, 对应 普通kill */
    return DRV_ERROR_NONE;
}


int setDevPhyId(uint32_t devIndex)
{
    gDevPhyId = devIndex;
    return DRV_ERROR_NONE;
}

rtError_t rtGetDevicePhyIdByIndex(uint32_t devIndex, uint32_t *phyId)
{
    if (gBoardId == 0x2000) {
        *phyId = devIndex*2;
        return RT_ERROR_NONE;
    }

    *phyId = devIndex;
    if (gDevPhyId) {
        *phyId = gDevPhyId;
    }
    return RT_ERROR_NONE;
}

rtError_t rtGetDeviceIndexByPhyId(u32 phyId, u32* devIndex)
{
    if (gBoardId == 0x2000) {
        *devIndex = phyId/2;
        return RT_ERROR_NONE;
    }
    *devIndex = phyId;
    if (gDevPhyId) {
        *devIndex = gDevPhyId;
    }
    return RT_ERROR_NONE;
}

rtError_t rtGetVisibleDeviceIdByLogicDeviceId(const int32_t logicDeviceId, int32_t *visibleDeviceId)
{
    *visibleDeviceId = logicDeviceId;
    return RT_ERROR_NONE;
}

rtError_t rtGetSocVersion(char *chipVer, u32 maxLen)
{
    if (chipVer == NULL) { return ACL_ERROR_RT_PARAM_INVALID; }
    sal_memcpy(chipVer, sizeof("Ascend910"), "Ascend910", sizeof("Ascend910"));

    if (chip_type_stub[0] == static_cast<s32>(DevType::DEV_TYPE_910B)) {
        sal_memcpy(chipVer, sizeof("Ascend910B1"), "Ascend910B1", sizeof("Ascend910B1"));
    } else if(chip_type_stub[0] == static_cast<s32>(DevType::DEV_TYPE_910_93)) {
        sal_memcpy(chipVer, sizeof("Ascend910_9391"), "Ascend910_9391", sizeof("Ascend910_9391"));
    } else if(gBoardId == 0x0000) {
        sal_memcpy(chipVer, sizeof("Ascend910"), "Ascend910", sizeof("Ascend910"));
    } else if (gBoardId == 0x2000) {  // 临时定义的 board id
        sal_memcpy(chipVer, sizeof("Ascend310P3"), "Ascend310P3", sizeof("Ascend310P3"));
    }
    return RT_ERROR_NONE;
}

int set_board_id(unsigned int board_id)
{
    gBoardId = board_id;
    return DRV_ERROR_NONE;
}
int set_VM(unsigned int VMModel)
{
    gIsVM = VMModel;
    return DRV_ERROR_NONE;
}
int dsmi_get_board_id(int device_id, unsigned int *board_id)
{
    if (board_id == nullptr) { return DRV_ERROR_INVALID_VALUE; }
    *board_id = 0;
    if(gBoardId)
    {
        *board_id = gBoardId;
    }
    return DRV_ERROR_NONE;
}

/*****************************************************************************
 函 数 名  : set_chip_type_stub
 功能描述  : 设置芯片类型
 输入参数  :  handler
 输出参数  : 无
 返 回 值  : drvError_t
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2018年8月7日
    作    者   : l00442453
    修改内容   : 新生成函数

*****************************************************************************/

void set_chip_type_stub(s32 devId, s32 chipType)
{
    if ((devId >= 0) && (chipType < static_cast<s32>(DevType::DEV_TYPE_COUNT))&& (devId < 256))
    {
        chip_type_stub[devId] = chipType;
    }
    else
    {
        HCCL_ERROR("device id is illegal");
    }
}

/*****************************************************************************
 函 数 名  : dsmi_get_chip_info
 功能描述  : 获取芯片信息
 输入参数  :  handler
 输出参数  : 无
 返 回 值  : drvError_t
 调用函数  :git log
 被调函数  :

 修改历史      :
  1.日    期   : 2018年8月7日
    作    者   : l00442453
    修改内容   : 新生成函数

*****************************************************************************/
namespace cce
{
    /*****************************************************************************
     函 数 名  : ccVectorReduce
     功能描述  : 规约操作
     输入参数  : const void *src1
                 const void *src2
                 uint32_t count
                 const ccDataType_t datatype
                 const HcclReduceOp op
                 rtStream_t stream
                 void *dst
     输出参数  : 无
     返 回 值  : ccStatus_t
     调用函数  :
     被调函数  :

     修改历史      :
      1.日    期   : 2017年12月7日
        作    者   : p00335137
        修改内容   : 新生成函数

    *****************************************************************************/
    ccStatus_t  ccVectorReduce( const void* src1, const void* src2, uint64_t count, const ccDataType_t datatype,
        const ccReduceOp_t op, rtStream_t stream, const void* dst )
    {
        if (!src1 || !src2 || !dst)
        {
            return cce::CC_STATUS_RESERVED;
        }

        if (datatype >= cce::CC_DATA_RESERVED)
        {
            return cce::CC_STATUS_RESERVED;
        }

        if (op >= CCE_RED_OP_RESERVED)
        {
            return cce::CC_STATUS_RESERVED;
        }

        // Mod for optimize runtime Stub by l on 2018-01-11 Below
        stream_class* rtstream = NULL;
        rtstream = (stream_class*)stream;

        // 将vector reduce任务，压入任务队列
        stream_task_t stream_task;
        stream_task.task_type = TASK_TYPE_REDUCE;
        stream_task.stream_para.reducestruct.src1 = (void*)src1;
        stream_task.stream_para.reducestruct.src2 = (void*)src2;
        stream_task.stream_para.reducestruct.count_reduce = count;
        stream_task.stream_para.reducestruct.datatype = datatype;
        stream_task.stream_para.reducestruct.op = op;
        stream_task.stream_para.reducestruct.dst_reduce = (void*)dst;

        rtstream->push_task(&stream_task);
        // Mod for optimize runtime Stub by l on 2018-01-11 Above

        return CC_STATUS_SUCCESS;
    }

/*****************************************************************************
     函 数 名  : cceSysInit
     功能描述  : cce init
     输入参数  : 无
     输出参数  : 无
     返 回 值  : void
     调用函数  :
     被调函数  :

     修改历史      :
      1.日    期   : 2019年05月16日
        作    者   : w00500539
        修改内容   : 新生成函数

    *****************************************************************************/
    void cceSysInit()
    {
        return;
    }


}

// Add for optimize runtime Stub by l on 2018-01-11 Below
/*
 *****************************************************************************
 * 函 数 名  : threadfun
 * 功能描述  : 线程函数实现
 * 输入参数  : p
 * 输出参数  : 无
 * 返 回 值  : void
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
void* threadfun(void* p)
{
    s32 iRet = SAL_OK;
    thread_class* pcthread =  (thread_class*)p;      // 管理本任务的对象.

    iRet = pcthread->update_thread_state(THREAD_STATE_WORKING);

    if (iRet)
    {
        HCCL_ERROR("Thread Update State To WORKING failed[%d]", iRet);
        return NULL;

    }

    iRet = pcthread->thread_handler();

    if (iRet)
    {
        HCCL_WARNING("[STUB] Thread Handler Return Faile[%d]", iRet);
    }

    iRet = pcthread->update_thread_state(THREAD_STATE_STOPED);

    if (iRet)
    {
        HCCL_WARNING("[STUB] Thread Update State To STOP failed[%d]", iRet);
    }

    return NULL;
}

/*
 *****************************************************************************
 * 函 数 名  : thread_class.thread_class
 * 功能描述  : 构造函数, 填充默认值.
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
thread_class::thread_class()
{
    //HCCL_INFO("Creat a new ThreadClass.");
    uithread_update_interval = THREAD_DEFAULT_UPDATE_INTERVAL; // 100ms
    uithreadstate = THREAD_STATE_STOPED;
    threadfd = NULL;
}

/*
 *****************************************************************************
 * 函 数 名  : thread_class.~thread_class
 * 功能描述  : 析构函数, 释放必要资源.
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
thread_class::~thread_class()
{
    try
    {
        (void)stop_thread();           // 停止任务
    }
    catch (...)
    {
        HCCL_ERROR("exception.");
    }
}

/*
 *****************************************************************************
 * 函 数 名  : thread_class.set_new_interval
 * 功能描述  : 设定任务间隔
 * 输入参数  : uiNewInterval
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
s32 thread_class::set_new_interval(u32 uinewinterval)
{
    s32 iRet = SAL_OK;

    uithread_update_interval = uinewinterval;

    return iRet;
}

/*
 *****************************************************************************
 * 函 数 名  : thread_class.get_current_interval
 * 功能描述  : 查询当前任务间隔
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  : u32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
u32 thread_class::get_current_interval()
{
    return uithread_update_interval;
}

/*
 *****************************************************************************
 * 函 数 名  : thread_class.start_thread
 * 功能描述  : 启动任务(带线程名).
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
s32 thread_class::start_thread(string thread_name)
{
    s32 iRet = SAL_OK;

    if (threadfd)   // 先停止任务.
    {
        (void)stop_thread();
    }

    iRet = update_thread_state(THREAD_STATE_INITALING); // 恢复任务默认状态

    if (iRet)
    {
        HCCL_ERROR("Thread Update State failed[%d]", iRet);
        threadfd = 0;
        return iRet;
    }

    iRet = pre_start_handler();

    if (iRet)
    {
        HCCL_ERROR("Pre Start Handler failed[%d]", iRet);
        threadfd = 0;
        return SAL_E_ERROR;
    }

    // 启动任务
    threadfd = sal_thread_create(thread_name, threadfun, this);

    if (NULL == threadfd)   // 任务启动失败
    {
        HCCL_ERROR("Create Thread failed");
        threadfd = NULL;
        return SAL_E_ERROR;
    }

    return iRet;
}

/*
 *****************************************************************************
 * 函 数 名  : thread_class.start_thread
 * 功能描述  : 启动任务(使用默认线程名).
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
s32 thread_class::start_thread()
{
    return start_thread("ThreadClass");
}

/*
 *****************************************************************************
 * 函 数 名  : thread_class.stop_thread
 * 功能描述  : 停止任务.
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
s32 thread_class::stop_thread()
{
    u32 uiInterval = get_current_interval() / 10 + THREAD_UPDATE_MIN;  // 计算检查周期
    u32 uiStopCounter = THREAD_STOP_COUNTER;
    s32 iRet = SAL_OK;

    if (threadfd)
    {
        iRet = pre_stop_handler();// 通知Handler主动退出.

        if (iRet)
        {
            HCCL_WARNING("Pre Stop Handler failed[%d]", iRet);
        }

        // 等待任务主动停止
        for (uiStopCounter = THREAD_STOP_COUNTER;
             (THREAD_STATE_STOPED != uithreadstate) && uiStopCounter;
             uiStopCounter --)
        {
            SaluSleep(uiInterval);
        }

        if (THREAD_STATE_STOPED != uithreadstate)  // 任务没有主动停止,尝试强制退出.
        {
            HCCL_WARNING("Stop Thread failed after [%d]us, Force Stop...", uiInterval * THREAD_STOP_COUNTER);

            iRet = sal_thread_destroy(threadfd);

            if (SAL_OK != iRet)
            {
                HCCL_ERROR("Force Stop Thread failed[%d]", iRet);

                iRet = SAL_E_ERROR;
            }

            iRet = update_thread_state(THREAD_STATE_STOPED); // 恢复任务默认状态

            if (iRet)
            {
                HCCL_ERROR("Thread Update State failed[%d]", iRet);

                threadfd = 0;
            }

            HCCL_WARNING("Force Stop Thread Sucess");
        }
    }

    threadfd = 0;
    return iRet;
}

/*
 *****************************************************************************
 * 函 数 名  : thread_class.update_thread_state
 * 功能描述  : 刷新任务状态
 * 输入参数  : uiNewState
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
s32 thread_class::update_thread_state(u32 uinewstate)
{
    if (THREAD_STATE_MAX <= uinewstate)
    {
        HCCL_ERROR("Thread update to State [%d] failed", uinewstate);

        return SAL_E_PARA;
    }

    uithreadstate = uinewstate;

    return SAL_OK;
}

/*
 *****************************************************************************
 * 函 数 名  : thread_class.thread_handler
 * 功能描述  : Thread回调函数.
               pre_stop_handler()执行后, thread_handler()需要在
GetCurrentInterval() us内主动退出.
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
s32 thread_class::thread_handler()
{
    while (get_current_interval())
    {
        HCCL_WARNING("Thread Handler use default func");

        SaluSleep(get_current_interval());
    }

    return SAL_OK;
}

/*
 *****************************************************************************
 * 函 数 名  : thread_class.pre_start_handler
 * 功能描述  : Thread即将启动, 通知ThreadHandler做好准备.
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
s32 thread_class::pre_start_handler()
{
    HCCL_WARNING("Pre Start Handler use default func");
    return SAL_OK;
}

/*
 *****************************************************************************
 * 函 数 名  : thread_class.pre_stop_handler
 * 功能描述  : Thread即将停止, 通知ThreadHandler主动退出.
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
s32 thread_class::pre_stop_handler()
{
    HCCL_WARNING("Pre Stop Handler use default func");
    (void)set_new_interval(0);
    return SAL_OK;
}

std::atomic<s32> stream_class::streamIdCounter_(0);
std::atomic<u32> stream_class::taskIdCounter_(0);

std::map<rtStream_t, int32_t> stream_class::streamMap_;
std::mutex stream_class::mapMutex_;
std::map<s32, atomic_ptr_t> stream_class::refCountMap_;
std::array<std::string, 8> stream_class::lineFeed_ = {"", "", "", "", "", "", "", ""};

/*
 *****************************************************************************
 * 函 数 名  : stream_class.stream_class
 * 功能描述  : 构造函数, 填充默认值.
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
stream_class::stream_class(s32 device_id) : deviceId_(device_id), streamId_(-1), stream_enabled_(true)
{
    // 生成stream id, 并添加到容器
    streamId_= streamIdCounter_.fetch_add(1);
    task_info.streamId = streamId_;
    {
        std::unique_lock<std::mutex> lock(mapMutex_);
        streamMap_[(rtStream_t)this] = streamId_; // 构造函数中使用this当Key是可行的
    }

    // 初始化reportData_
    std::string postFixRuntime("runtime-");
    postFixRuntime += std::to_string(deviceId_);
    memcpy(dataRuntime_.tag, postFixRuntime.c_str(), postFixRuntime.size() + 1);
    dataRuntime_.deviceId = deviceId_;

    std::string postFixHWTS("HWTS-");
    postFixHWTS += std::to_string(deviceId_);
    memcpy(dataHWTS_.tag, postFixHWTS.c_str(), postFixHWTS.size() + 1);
    dataHWTS_.deviceId = deviceId_;

    // device id增加stream的计数
    {
        std::unique_lock<std::mutex> lock(mapMutex_);

        if (refCountMap_.find(device_id) == refCountMap_.end()) {
            refCountMap_[device_id] = atomic_ptr_t(new std::atomic<u32>(0));

            HCCL_DEBUG("new device found[%d]", device_id);
            std::string file_content("{\r\"traceEvents\": [\r");
            dataHWTS_.data = (unsigned char*)(const_cast<char*>(file_content.c_str()));
            dataHWTS_.dataLen = file_content.size();
        }
        u32 streamCount = refCountMap_[device_id]->fetch_add(1);
        HCCL_DEBUG("streamCount = %u, deviceId_[%d]", streamCount, deviceId_);
    }

    stream_task_lock = sal_mutex_create("stream_task_lock");
    stream_task_list.clear(); // 清空任务队列

    thread_trigger = sal_sem_create("stream_thread", SAL_FALSE, 0);
    stream_task_done  = sal_sem_create("stream_task_done", SAL_FALSE, 0);
    (void)start_thread("StreamThread");
}

/*
 *****************************************************************************
 * 函 数 名  : stream_class.~stream_class
 * 功能描述  : 析构函数, 释放必要资源.
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
stream_class::~stream_class()
{
    HCCL_INFO("HCCL TEST stream_class xigou1");
    try
    {
        // 退出信息记录, 方便问题定位
        HCCL_DEBUG("Stream[%p] Destroy Now", this);
        // 停止任务
        (void)stop_thread();

        if (thread_trigger)
        { sal_sem_destroy(thread_trigger); }

        if (stream_task_done)
        { sal_sem_destroy(stream_task_done); }


        // device id增加stream的计数, 最后一个stream销毁时将文件Flush
        {
            std::unique_lock<std::mutex> lock(mapMutex_);
            u32 streamCount = refCountMap_[deviceId_]->fetch_sub(1);
            HCCL_DEBUG("streamCount = %u, deviceId_[%d]", streamCount, deviceId_);
            if (streamCount == 1)
            {
                // HWTS的task信息文件Flush
                std::string file_content;
                file_content += "\r]\r}";
                dataHWTS_.data = (unsigned char*)(const_cast<char*>(file_content.c_str()));
                dataHWTS_.dataLen = file_content.size();

                // Map里删除对应的deviceId
                refCountMap_.erase(deviceId_);
            }

            streamMap_.erase((rtStream_t)this);
        }

        // 清空软表
        stream_task_list.clear();

        // 销毁互斥锁
        if (stream_task_lock)
        { sal_mutex_destroy(stream_task_lock); }

        stream_task_lock = NULL;

    }
    catch (...)
    {
        HCCL_ERROR("exception.");
    }
}

void stream_class::HWTSLog(const stream_task_t& task, u64 ts_start, u64 duration)
{
    std::string content;
    content += "{ \"pid\":";
    content += std::to_string(streamId_);
    content += ", \"ts\":";
    content += std::to_string((double)ts_start / 1000.0);
    content += ", \"dur\":";
    content += std::to_string((double)duration / 1000.0);
    content += ", \"name\":";
    content += std::to_string(task.task_id);
    content += ", \"args\":{ ";
    content += "\"us\":";
    content += std::to_string((double)duration / 1000.0);
    content += " } }";

    {
        std::unique_lock<std::mutex> lock(mapMutex_);

        std::string file_content("");
        file_content += lineFeed_[deviceId_];
        file_content += content;
        dataHWTS_.data = (unsigned char*)(const_cast<char*>(file_content.c_str()));
        dataHWTS_.dataLen = file_content.size();
        lineFeed_[deviceId_] = ",\r";
    }
}

u64 stream_class::TimestampNanosecond()
{
    // 此时间戳获取方式需要与runtime保持一致

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (u64)(ts.tv_sec * 1000000000 + ts.tv_nsec);
}

/*
*****************************************************************************
* 函 数 名  : stream_class.push_task
* 功能描述  : stream list pushback API
* 输入参数  : module
*             eLogType
*             fmt
*             args
* 输出参数  : 无
* 返 回 值  : s32
* 其它说明  :
*
* 修改历史      :
*  1.日    期   : 2018年1月11日
*    作    者   : lifuning
*    修改内容   : 新生成函数
*
*****************************************************************************
*/
s32 stream_class::push_task(stream_task_t* stream_task)
{
    if (stream_task == nullptr) {
        HCCL_ERROR("stream_task is NULL");
        return SAL_E_ERROR;
    }

    // add task_id for this task
    stream_task->task_id = taskIdCounter_.fetch_add(1);
    task_info.taskId = stream_task->task_id;
    task_info.streamId = streamId_;
    // 将task信息写入到文件, runtime profiling桩函数
    rtProfTaskTrack_t taskTrackData;
    taskTrackData.head.rserved = 0x1020304; //hccl_perf_tool用于防错
    taskTrackData.timeStamp = TimestampNanosecond();
    taskTrackData.streamId = streamId_;
    taskTrackData.taskType = (u16)stream_task->task_type;
    taskTrackData.taskId = (u16)stream_task->task_id;
    taskTrackData.deviceId = deviceId_;
    dataRuntime_.data = (unsigned char*)&taskTrackData;
    dataRuntime_.dataLen = sizeof(rtProfTaskTrack_t);

    {
        std::unique_lock<std::mutex> lock(mapMutex_);
    }

    MSG_LOCK();
    stream_task_list.push_back(*stream_task);
    MSG_UNLOCK();

    trigger_thread();

    return SAL_OK;
}

s32 stream_class::get_stream_id() const
{
    return streamId_;
}

s32 stream_class::get_device_id() const
{
    return deviceId_;
}

void stream_class::trigger_thread()
{
    if (thread_trigger)
    {
        (void)sal_sem_give(thread_trigger);
    }
    else
    {
        HCCL_ERROR("thread_trigger is NULL");
    }

    return;
}

void stream_class::set_stream_enabled(bool enabled)
{
    stream_enabled_ = enabled;
}

bool IsFailureTask(u32 deviceId, tasktype_e task_type)
{
    return (FailureTaskType!=TASK_TYPE_RESERVED && FailureDeviceId ==  deviceId && FailureTaskType == task_type);
}
void ClearFailureTask()
{
    FailureDeviceId = 0xFFFFFFFF;
    FailureTaskType = TASK_TYPE_RESERVED;
}
/*
 *****************************************************************************
 * 函 数 名  : stream_class.thread_handler
 * 功能描述  : 任务队列主处理函数
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
s32 stream_class::thread_handler()
{
    u32 streamId = get_stream_id();
    u32 deviceId = get_device_id();
    while (get_current_interval())
    {
        // 释放 CPU, 周期性唤醒任务
        (void)sal_sem_take(thread_trigger, get_current_interval());
        // 检查 stream_task_list 是否为空
        if (!stream_task_list.empty())
        {
            // 处理stream_task_list
            do
            {
                list<stream_task_t>::iterator iter;

                MSG_LOCK();

                for (iter = stream_task_list.begin(); iter != stream_task_list.end();)
                {
                    MSG_UNLOCK();
                    if(IsFailureTask(deviceId,  iter->task_type)) {
                        for(auto it : taskFailCallbackMap) {
                            rtExceptionInfo tmpInfo;
                            tmpInfo.taskid = iter->task_id;
                            tmpInfo.streamid = streamId;
                            tmpInfo.tid = 0;
                            tmpInfo.deviceid = deviceId;
                            tmpInfo.retcode = 1;
                            if(it.second != nullptr) {
                                it.second(&tmpInfo);
                            }

                        }
                        ClearFailureTask();
                    }

                    switch (iter->task_type)
                    {
                        case TASK_TYPE_MEMCPY:
                        {
                            u64 ts_start = TimestampNanosecond();
                            rtError_t ret =
                                memcpy_async(iter->stream_para.memcpystruct.dst,
                                             iter->stream_para.memcpystruct.src, iter->stream_para.memcpystruct.count);
                            u64 duration = TimestampNanosecond() - ts_start;
                             HWTSLog(*iter, ts_start, duration);

                            if (RT_ERROR_NONE != ret)
                            {
                                HCCL_ERROR("Task MemcpyAsync failed, ret[%d], para:dst[%d] src[%d] count[%d].",
                                           ret, iter->stream_para.memcpystruct.dst,
                                           iter->stream_para.memcpystruct.src, iter->stream_para.memcpystruct.count);
                            }

                            break;
                        }

                        case TASK_TYPE_RECORD:
                        {
                            rtError_t ret = event_record(&(iter->stream_para.event));

                            if (RT_ERROR_NONE != ret)
                            {
                                HCCL_ERROR("Task event record failed, ret[%d], para:event[%p],sem[%p].",
                                           ret, iter->stream_para.event.event_handler, iter->stream_para.event.sem);
                            }

                            break;
                        }

                        case TASK_TYPE_MULTIDEV_RECORD:
                        {
                            rtError_t ret = event_multidev_record(&(iter->stream_para.event));

                            if (RT_ERROR_NONE != ret)
                            {
                                HCCL_ERROR("Task event multidev record failed, ret[%d], para:event[%p],sem[%p].",
                                           ret, iter->stream_para.event.event_handler, iter->stream_para.event.sem);
                            }

                            break;
                        }

                        case TASK_TYPE_WAIT:
                        {
                            rtError_t ret = event_wait(&(iter->stream_para.event));

                            if (RT_ERROR_NONE != ret)
                            {
                                HCCL_ERROR("Task event wait failed, ret[%d], para:event[%p],sem[%p].",
                                           ret, iter->stream_para.event.event_handler, iter->stream_para.event.sem);
                            }

                            break;
                        }

                        case TASK_TYPE_REDUCE:
                        {
                            u64 ts_start = TimestampNanosecond();
                            cce::ccStatus_t ret =
                                vector_reduce(iter->stream_para.reducestruct.src1,
                                              iter->stream_para.reducestruct.src2,
                                              iter->stream_para.reducestruct.count_reduce,
                                              iter->stream_para.reducestruct.datatype,
                                              iter->stream_para.reducestruct.op,
                                              iter->stream_para.reducestruct.dst_reduce);
                            u64 duration = TimestampNanosecond() - ts_start;
                            HWTSLog(*iter, ts_start, duration);

                            if (cce::CC_STATUS_SUCCESS != ret)
                            {
                                HCCL_ERROR("Task vector reduce failed, ret[%d], para:src1[%d] src2[%d] count_reduce[%d] datatype[%d] op[%d] dst_reduce[%d].",
                                           ret,
                                           iter->stream_para.reducestruct.src1,
                                           iter->stream_para.reducestruct.src2,
                                           iter->stream_para.reducestruct.count_reduce,
                                           iter->stream_para.reducestruct.datatype,
                                           iter->stream_para.reducestruct.op,
                                           iter->stream_para.reducestruct.dst_reduce);
                            }

                            break;
                        }

                        case TASK_TYPE_USLEEP:
                        {
                            rtError_t ret = stream_usleep(iter->stream_para.usec);

                            if (RT_ERROR_NONE != ret)
                            {
                                HCCL_ERROR("Task MemcpyAsync failed, ret[%d], para:usec[%d].",
                                           ret, iter->stream_para.usec);
                            }

                            break;
                        }

                        case TASK_TYPE_NOTIFY_RECORD:
                        {
                            u64 ts_start = TimestampNanosecond();

                            rtError_t ret = notify_record(iter->stream_para.notify);
                            u64 duration = TimestampNanosecond() - ts_start;
                            HWTSLog(*iter, ts_start, duration);

                            if (RT_ERROR_NONE != ret)
                            {
                                HCCL_ERROR("Task NotyfiRecordAsync failed, ret[%d]", ret);
                            }

                            break;
                        }

                        case TASK_TYPE_NOTIFY_WAIT:
                        {
                            u64 ts_start = TimestampNanosecond();

                            rtError_t ret = notify_wait(iter->stream_para.notify);
                            u64 duration = TimestampNanosecond() - ts_start;
                            HWTSLog(*iter, ts_start, duration);

                            if (RT_ERROR_NONE != ret)
                            {
                                HCCL_ERROR("Task NotyfiWaitAsync failed, ret[%d]", ret);
                            }

                            break;
                        }

                        case TASK_TYPE_RDMA_SEND:
                        {
                            u64 ts_start = TimestampNanosecond();
                            rtError_t ret = rdma_send(iter->stream_para.rdmasend.wqe_index, iter->stream_para.rdmasend.cn);
                            u64 duration = TimestampNanosecond() - ts_start;
                            HWTSLog(*iter, ts_start, duration);

                            if (RT_ERROR_NONE != ret)
                            {
                                HCCL_ERROR("Task RdmaSend Async failed, ret[%d], wqe_index[%d]", ret, iter->stream_para.rdmasend.wqe_index);
                            }

                            break;
                        }

                        case TASK_TYPE_CALLBACK_FUNC:
                        {
                            while (iter->stream_para.callbackTask.isBlock) {
                                if (iter->stream_para.callbackTask.isExecuted) {
                                    break;
                                }
                            }
                            break;
                        }

                        default:
                        {
                            HCCL_DEBUG("Not support the task type [%d].", iter->task_type);
                            break;
                        }

                    }

                    MSG_LOCK();
                    iter = stream_task_list.erase(iter);
                    MSG_UNLOCK();

                    (void)sal_sem_give(stream_task_done);
                    MSG_LOCK();
                }

                MSG_UNLOCK();

            }
            while (!stream_task_list.empty());
        }
    }

    return SAL_OK;
}

rtError_t stream_class::rdma_send(u32 wqe_index, void* cnn)
{
    return RT_ERROR_NONE;
}
/*
 *****************************************************************************
 * 函 数 名  : stream_class.pre_stop_handler
 * 功能描述  : StopThread触发, 通知ThreadHandler主动退出.
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
s32 stream_class::pre_stop_handler()
{
    s32 ret = SAL_OK;
    // 唤醒日志线程, 让日志线程将所有日志写入文件.
    trigger_thread();
    ret = set_new_interval(0);
    // 唤醒日志线程, 让其快速主动退出
    trigger_thread();
    return ret;
}

/*
 *****************************************************************************
 * 函 数 名  : stream_class.pre_start_handler
 * 功能描述  : StartThread触发, 通知ThreadHandler即将被调用.
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
s32 stream_class::pre_start_handler()
{
    u32 new_interval = THREAD_UPDATE_MIN;

    return set_new_interval(new_interval);
}

rtError_t stream_class::stream_synchronize()
{
    if (!stream_task_list.empty())
    {
        do
        {
            (void)sal_sem_take(stream_task_done, get_current_interval());

        }
        while (!stream_task_list.empty());
    }
    return RT_ERROR_NONE;
}

rtError_t stream_class::stream_usleep(u32 usec)
{
    s32 iRet = 0;
    iRet = usleep(usec);

    if (iRet)
    {
        HCCL_ERROR("Sleep: usleep failed[%d]: %s [%d]", iRet, strerror(errno), errno);
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    return RT_ERROR_NONE;

}

rtError_t stream_class::event_record(rtEvent_t event)
{
    s32 err = -1;

    if (!event)
    {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    rt_event_stub_t* rtevent = NULL;
    rtevent = (rt_event_stub_t*)event;

    if (!rtevent->sem)
    {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    err = sal_sem_give(rtevent->sem);

    if (err)
    {
        return ACL_ERROR_RT_INTERNAL_ERROR;
    }

    return RT_ERROR_NONE;
}

rtError_t stream_class::event_multidev_record(rtEvent_t event)
{
    s32 err = -1;

    if (!event)
    {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    rt_event_stub_t* rtevent = NULL;
    rtevent = (rt_event_stub_t*)event;

    if (!rtevent->sem)
    {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    err = sal_sem_give(rtevent->sem);

    if (err)
    {
        return ACL_ERROR_RT_INTERNAL_ERROR;
    }

    return RT_ERROR_NONE;
}

rtError_t stream_class::event_wait(rtEvent_t event)
{
    s32 err = -1;
    s32 usec = SAL_SEM_FOREVER;               // 定义10ms等不到event，认为出错，避免挂死

    if (!event)
    {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    rt_event_stub_t* rtevent = NULL;
    rtevent = (rt_event_stub_t*)event;

    if (!rtevent->sem)
    {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    err = sal_sem_take(rtevent->sem, usec);

    if (err)
    {
        return ACL_ERROR_RT_INTERNAL_ERROR;
    }

    if (rtevent->sem)
    {
        HCCL_DEBUG(" ++ event[%p] wait sem[%p] destroy!", rtevent->event_handler, rtevent->sem);
        sal_sem_destroy(rtevent->sem);
        rtevent->sem = NULL;
    }

    return RT_ERROR_NONE;
}

rtError_t stream_class::memcpy_async(void* dst, void* src, uint64_t count)
{
    if (stream_enabled_ == false) {
        return RT_ERROR_NONE;
    }

    rtError_t ret;

    ret = (rtError_t)memcpy_s(dst, count, src, count);

    if (ret)
    {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    return RT_ERROR_NONE;
}

rtError_t stream_class::notify_record(rt_notify_t* notify)
{
    if (stream_enabled_ == false) {
        return RT_ERROR_NONE;
    }
    rt_notify_t* ipc_notify = notify;
    rt_shm_notify_t* notify_shm = (rt_shm_notify_t*)ipc_notify->ipc_notify_shm;
    if (nullptr == notify_shm) {
        HCCL_ERROR("parameter error : notify_shm[%p]", notify_shm);
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    HCCL_DEBUG("record : device[%d], notify_id[%llu]", notify_shm->device_id, ipc_notify->notify_id);

    // wait notify, 通过shm的原子操作实现
    s32 timeout_cnt = NOTIFY_TIMEOUT_CNT;
    while (!__sync_bool_compare_and_swap(&(notify_shm->record_cnt[ipc_notify->notify_id]), 0, 1)) {
        SaluSleep(1000);

        timeout_cnt--;
        if  (timeout_cnt <= 0) {
            HCCL_ERROR("record timeout : record_cnt[%d], device_id[%d], notify_id[%llu]",
                notify_shm->record_cnt[ipc_notify->notify_id],
                notify_shm->device_id,
                ipc_notify->notify_id);
            return ACL_ERROR_RT_PARAM_INVALID;
        }
    }

    return  RT_ERROR_NONE;
}

rtError_t stream_class::notify_wait(rt_notify_t* notify)
{
    if (stream_enabled_ == false) {
        return RT_ERROR_NONE;
    }
    rt_notify_t* ipc_notify = notify;
    HCCL_DEBUG("notify_wait: notify[%p]", ipc_notify);
    rt_shm_notify_t* notify_shm = (rt_shm_notify_t*)ipc_notify->ipc_notify_shm;
    if (nullptr == notify_shm) {
        HCCL_ERROR("parameter error : notify_shm[%p]", notify_shm);
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    void* tmpPtr = &(notify_shm->record_cnt[ipc_notify->notify_id]);
    HCCL_INFO("wait : device[%d], notify_id[%llu] tmpPtr[%p]", notify_shm->device_id, ipc_notify->notify_id, tmpPtr);

    s32 timeout_cnt = NOTIFY_TIMEOUT_CNT;
    while (!__sync_bool_compare_and_swap(&(notify_shm->record_cnt[ipc_notify->notify_id]), 1, 0)) {
        SaluSleep(1000);

        timeout_cnt--;
        if (timeout_cnt <= 0) {
            HCCL_ERROR("wait timeout : record_cnt[%d], device_id[%d], notify_id[%llu]",
                notify_shm->record_cnt[ipc_notify->notify_id],
                notify_shm->device_id,
                ipc_notify->notify_id);
            return ACL_ERROR_RT_PARAM_INVALID;
        }
    }

    return  RT_ERROR_NONE;
}

void stream_class::ExecuteCallbackFunc()
{
}

template <typename T>
cce::ccStatus_t stream_class::reduce_op(T src1, T src2, T* dst, const cce::ccReduceOp_t op)
{
    switch (op)
    {
        case cce::CCE_RED_OP_SUM:
            *dst = src1 + src2;
            break;

        case cce::CCE_RED_OP_PROD:
            *dst = src1 * src2;
            break;

        case cce::CCE_RED_OP_MAX:
            *dst = (src1 > src2) ? src1 : src2;
            break;

        case cce::CCE_RED_OP_Min:
            *dst = (src1 < src2) ? src1 : src2;
            break;

        default:
            return cce::CC_STATUS_NOT_SUPPORTED;

    }

    return cce::CC_STATUS_SUCCESS;
}

cce::ccStatus_t stream_class::vector_reduce( const void* src1, const void* src2,
        uint32_t count, const cce::ccDataType_t datatype,
        const cce::ccReduceOp_t op, void* dst )
{
    int32_t loop;
    void* input1 = (void*)src1;
    void* input2 = (void*)src2;
    float* x1, *x2, *x;
    char* y1, *y2, *y;
    u16* z1, *z2, *z;
    s32 *m1, *m2, *m;
    s16 *n1, *n2, *n;
    if (!src1 || !src2 || !dst)
    {
        return cce::CC_STATUS_RESERVED;
    }

    if (datatype >= cce::CC_DATA_RESERVED)
    {
        return cce::CC_STATUS_RESERVED;
    }

    if (op >= cce::CCE_RED_OP_RESERVED)
    {
        return cce::CC_STATUS_RESERVED;
    }

    // float16在桩函数中使用int16代替
    if (datatype == cce::CC_DATA_HALF)
    {
        z1 = (u16*)input1;
        z2 = (u16*)input2;
        z = (u16*)dst;
    }

    if (datatype == cce::CC_DATA_FLOAT)
    {
        x1 = (float*)input1;
        x2 = (float*)input2;
        x = (float*)dst;
    }

    if (datatype == cce::CC_DATA_INT8)
    {
        y1 = (char*)input1;
        y2 = (char*)input2;
        y = (char*)dst;
    }

    if (datatype == cce::CC_DATA_INT16)
    {
        n1 = (s16*)input1;
        n2 = (s16*)input2;
        n = (s16*)dst;
    }

    if (datatype == cce::CC_DATA_INT32)
    {
        m1 = (s32*)input1;
        m2 = (s32*)input2;
        m = (s32*)dst;
    }

    for (loop = 0; loop < count; loop++)
    {
        if (datatype == cce::CC_DATA_HALF)
        {
            if (op == cce::CCE_RED_OP_SUM) {
                float f1 = fp16_ieee_to_fp32_value(*z1);
                float f2 = fp16_ieee_to_fp32_value(*z2);
                float f = (u32)f1 + (u32)f2;
                *z = fp16_ieee_from_fp32_value(f);
            } else {
                (void)reduce_op(*z1, *z2, z, op);
            }
            z1++;
            z2++;
            z++;
        }

        if (datatype == cce::CC_DATA_FLOAT)
        {
            (void)reduce_op(*x1, *x2, x, op);
            x1++;
            x2++;
            x++;
        }

        if (datatype == cce::CC_DATA_INT8)
        {
            (void)reduce_op(*y1, *y2, y, op);
            y1++;
            y2++;
            y++;
        }

        if (datatype == cce::CC_DATA_INT16)
        {
            (void)reduce_op(*n1, *n2, n, op);
            n1++;
            n2++;
            n++;
        }

        if (datatype == cce::CC_DATA_INT32)
        {
            (void)reduce_op(*m1, *m2, m, op);
            m1++;
            m2++;
            m++;
        }
    }

    return cce::CC_STATUS_SUCCESS;
}
// Add for optimize runtime Stub by l on 2018-01-11 Above

#ifdef __cplusplus
extern "C"
{
#endif

void* __WorkSpaceMemAllocStub(std::string tag, u64 size)
{
    void *ptr = NULL;
    HcclResult ret = hrtMalloc(&ptr,size);
    CHK_PRT_RET(ret, HCCL_ERROR("rt_malloc fail, tag[%s], size[%llu], ret[%d]",
        tag.c_str(), size, ret),NULL);
    return ptr;
}

strong_alias(__WorkSpaceMemAllocStub, WorkSpaceMemAlloc);

#ifdef __cplusplus
} // extern "C"
#endif

//获取随机值
int RAND_bytes(char *buf, int num)
{
    int i;
    unsigned seed;  // Random generator seed
    CHK_PTR_NULL(buf);
    seed = time(0);

    srand(seed);

    for(i=0; i< num; i++) {
        buf[i] = (rand() % 25)+65; //产生1~15 的随机数
    }
    return 1;
}

static const s32 size_table[HCCL_DATA_TYPE_RESERVED] = { 1, 2, 4, 2, 4 };  // HCCL_DATA_TYPE_RESERVED
s32 get_data_size(const HcclDataType dataType)
{
    if (dataType < HCCL_DATA_TYPE_RESERVED) {
        return size_table[dataType];
    } else {
        HCCL_ERROR("data type[%d] out of range[%d, %d]", dataType, HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_RESERVED - 1);
        return 0;
    }
}

/* 底层 runtime api 返回值转换成 HcclResult 格式 */

int32_t QueryPartitionMapPsId(uint64_t key, uint32_t *psId){
    *psId = 0;
    return 0;
}

int32_t InitPartitionMap(uint32_t partitionNum, uint32_t psNum, const uint32_t psId[])
{
    return 0;
}

int32_t GetBatchPsIds(uint64_t *keys, uint32_t *psIds[], uint32_t num)
{
    for (int i = 0; i < num; i++) {
        (*psIds)[i] = 0;
    }
    return 0;
}

int32_t MsprofRegisterCallback(uint32_t moduleId, ProfCommandHandle handle);

int32_t MsprofRegTypeInfo(uint16_t level, uint32_t typeId, const char *typeName);

int32_t MsprofReportApi(uint32_t agingFlag, const MsprofApi *api);

int32_t MsprofReportCompactInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length);

int32_t MsprofReportAdditionalInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length);

uint64_t MsprofGetHashId(const char *hashInfo, size_t length);

uint64_t MsprofSysCycleTime();

HcclResult AtraceSubmit(int32_t handle, const void *buffer, uint32_t bufSize)
{
    (void)(handle);
    (void)(buffer);
    (void)(bufSize);
    return HCCL_SUCCESS;
}

void AtraceDestroy(int32_t handle)
{
    (void)(handle);
    return;
}

HcclResult UtraceSubmit(int32_t handle, const void *buffer, uint32_t bufSize)
{
    (void)(handle);
    (void)(buffer);
    (void)(bufSize);
    return HCCL_SUCCESS;
}

void UtraceDestroy(int32_t handle)
{
    (void)(handle);
    return;
}

typedef struct TraceAttr {
    bool exitSave;          // exec save when AtraceDestroy
} TraceAttr;

typedef struct TraceGlobalAttr {
    uint8_t saveMode;   // 0: local save; 1: send to remote and save
    uint8_t deviceId;   // 0: default; 32~63:vf
    uint32_t pid;       // 0: default; if saveMode=1, means host pid
    uint8_t reserve[32];
} TraceGlobalAttr;

int32_t AtraceCreateWithAttr(int32_t tracerType, const char *objName, const TraceAttr *attr)
{
    (void)(tracerType);
    (void)(objName);
    (void)(attr);
    return 0;
}

int32_t UtraceCreateWithAttr(int32_t tracerType, const char *objName, const TraceAttr *attr)
{
    (void)(tracerType);
    (void)(objName);
    (void)(attr);
    return 0;
}

int32_t UtraceSetGlobalAttr(const TraceGlobalAttr *attr)
{
    (void)(attr);
    return 0;
}

typedef enum TracerType {
    TRACER_TYPE_SCHEDULE   = 0,
    TRACER_TYPE_PROGRESS   = 1,
    TRACER_TYPE_STATISTICS = 2,
    TRACER_TYPE_MAX,
} TracerType;

int32_t UtraceSave(TracerType tracerType, bool syncFlag)
{
    (void)(tracerType);
    (void)(syncFlag);
    return 0;
}

int ibv_ext_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
				   struct ibv_send_wr **bad_wr, struct ibv_post_send_ext_attr *ext_attr,
				   struct ibv_post_send_ext_resp *ext_resp) {
    return 0;
}

int stub_ibv_ext_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
				   struct ibv_send_wr **bad_wr, struct ibv_post_send_ext_attr *ext_attr,
				   struct ibv_post_send_ext_resp *ext_resp) {
    return ibv_ext_post_send(qp, wr, bad_wr, ext_attr, ext_resp);
}

int stub_ibv_exp_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr,
    struct wr_exp_rsp *exp_rsp)
{
    return -12;
}

namespace hccl
{
const std::unordered_map<std::string, DevType> SOC_VER_CONVERT{
    {"Ascend310P1", DevType::DEV_TYPE_310P1},
    {"Ascend310P3", DevType::DEV_TYPE_310P3},
    {"Ascend310P5", DevType::DEV_TYPE_310P3},
    {"Ascend310P7", DevType::DEV_TYPE_310P3},
    {"Ascend910", DevType::DEV_TYPE_910},
    {"Ascend910A", DevType::DEV_TYPE_910},
    {"Ascend910B", DevType::DEV_TYPE_910},
    {"Ascend910ProA", DevType::DEV_TYPE_910},
    {"Ascend910ProB", DevType::DEV_TYPE_910},
    {"Ascend910PremiumA", DevType::DEV_TYPE_910},
    {"Ascend910B1", DevType::DEV_TYPE_910B},
    {"Ascend910B2", DevType::DEV_TYPE_910B},
    {"Ascend910B2C", DevType::DEV_TYPE_910B},
    {"Ascend910B3", DevType::DEV_TYPE_910B},
    {"Ascend910B4", DevType::DEV_TYPE_910B},
    {"Ascend910_9391", DevType::DEV_TYPE_910_93},
    {"Ascend910_9381", DevType::DEV_TYPE_910_93},
    {"Ascend910_9372", DevType::DEV_TYPE_910_93},
    {"nosoc", DevType::DEV_TYPE_NOSOC}};

enum callHccl {
    HcclOpKernelAddCounter,
    HcclOpKernelClearCounter,
    HcclOpKernelLogStr,
    HcclOpKernelCmpInt8,
    HcclOpKernelCmpInt32,
    HcclOpKernelCmpFp16,
    HcclOpKernelCmpFp32,
    HcclOpKernelLogInt8,
    HcclOpKernelLogInt32,
    HcclOpKernelLogFp16,
    HcclOpKernelLogFp32,
    HcclOpKernelLogVariable,
};

int32_t mmDladdr(void *addr, mmDlInfo *info)
{
    return 0;
}

#ifdef __cplusplus
}  // extern "C"
#endif
HcclResult SetDevicePlaneId(u32 devicePhyId, u32 planeId)
{
    auto it = std::find(DevicePlaneList.begin(), DevicePlaneList.end(),devicePhyId);
    if(it == DevicePlaneList.end())
    {
        DevicePlaneInfo_t tmpPlaneInfo;
        tmpPlaneInfo.devicePhyId = devicePhyId;
        tmpPlaneInfo.planeId = planeId;
        DevicePlaneList.push_back(tmpPlaneInfo);
        return HCCL_SUCCESS;
    }
    return HCCL_E_PARA;
}
HcclResult GetDevicePlaneId(u32 devicePhyId, u32 &planeId)
{
    auto it = std::find(DevicePlaneList.begin(), DevicePlaneList.end(),devicePhyId);
    if(it != DevicePlaneList.end())
    {
        planeId = it->planeId;
        return HCCL_SUCCESS;
    }
    return HCCL_E_PARA;
}
void ClearDevicePlaneId()
{
    DevicePlaneList.clear();
    return;
}
void FailureInjectStub(u32 deviceId, tasktype_e taskType)
{
    FailureDeviceId = deviceId;
    FailureTaskType = taskType;
}
void FailureClear()
{
    FailureDeviceId = 0xFFFFFFFF;
    FailureTaskType = TASK_TYPE_RESERVED;
}

bool g_isUseRealPortAndName = false;
void UseRealPortAndName(bool isUse)
{
    g_isUseRealPortAndName = isUse;
}
bool IsUseRealPortAndName()
{
    return g_isUseRealPortAndName;
}

aclError aclrtCreateContext(aclrtContext *ctx, int32_t deviceId)
{
    static int rtCtx = 0;
    *ctx = &rtCtx;
    return ACL_SUCCESS;
}

aclError aclrtDestroyContext(aclrtContext ctx)
{
    return ACL_SUCCESS;
}

uint32_t GetCPUNum()
{
    return 1;
}

rtError_t rtStreamGetCqid(const rtStream_t stm, uint32_t *cqId, uint32_t *logicCqId) {
    static uint32_t i = 0U;
    *logicCqId = i++;
    return RT_ERROR_NONE;
}


rtError_t rtCtxGetOverflowAddr(void **overflowAddr) {
  *overflowAddr = (void *)0x1;
  return RT_ERROR_NONE;
}

rtError_t rtKernelGetAddrAndPrefCnt(void *hdl, const uint64_t tilingKey, const void * const stubFunc,
                                            const uint32_t flag, void **addr, uint32_t *prefetchCnt)
{
    return RT_ERROR_NONE;
}
rtError_t rtGetDevArgsAddr(rtStream_t stm, rtArgsEx_t *argsInfo, void **devArgsAddr, void **argsHandle)
{
    return RT_ERROR_NONE;
}



/**
 * @ingroup dvrt_mem
 * @brief set the attribute of shared memory
 * @param [in] name   identification name 
 * @param [in] type   shared memory mapping type 
 * @param [in] attr   shared memory attribute
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 * @return RT_ERROR_DRV_ERR for driver error
*/
rtError_t rtIpcSetMemoryAttr(const char *name, uint32_t type, uint64_t attr)
{
    return RT_ERROR_NONE;
}

int tsDevSendMsgAsync(unsigned int devId, unsigned int tsId, char *msg, unsigned int msgLen, unsigned int handleId)
{
    return DRV_ERROR_NONE;
}

drvError_t halEschedSubmitEvent(uint32_t devId, struct event_summary *event)
{
    return DRV_ERROR_NONE;
}


aclError aclrtBinaryUnLoad(aclrtBinHandle binHandle)
{
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsAppend(aclrtArgsHandle argsHandle, void *param, size_t paramSize,
    aclrtParamHandle *paramHandle)
{
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsAppendPlaceHolder(aclrtArgsHandle argsHandle, aclrtParamHandle *paramHandle)
{
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsGetPlaceHolderBuffer(aclrtArgsHandle argsHandle, aclrtParamHandle paramHandle,
    size_t dataSize, void **bufferAddr)
{
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsFinalize(aclrtArgsHandle argsHandle)
{
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsInit(aclrtFuncHandle funcHandle, aclrtArgsHandle *argsHandle)
{
    static int i = 0;
    *argsHandle = &i;
    return ACL_SUCCESS;
}

aclError aclrtBinaryLoadFromFile(const char* binPath, aclrtBinaryLoadOptions *options,
    aclrtBinHandle *binHandle)
{
    return ACL_SUCCESS;
}

aclError aclrtLaunchKernelWithConfig(aclrtFuncHandle funcHandle, uint32_t blockDim,
    aclrtStream stream, aclrtLaunchKernelCfg *cfg,
    aclrtArgsHandle argsHandle, void *reserve)
{
    return ACL_SUCCESS;
}

aclError aclrtBinaryGetFunction(const aclrtBinHandle binHandle, const char *kernelName,
    aclrtFuncHandle *funcHandle)
{
    static int i = 0;
    *funcHandle = &i;
    return ACL_SUCCESS;
}

aclError aclrtGetDeviceResLimit(int32_t deviceId, aclrtDevResLimitType type, uint32_t* value)
{
    *value = 48;
    return ACL_SUCCESS;
}

aclError aclrtGetResInCurrentThread(aclrtDevResLimitType type, uint32_t *value)
{
    *value = 48;
    return ACL_SUCCESS;
}

aclError aclrtMallocWithCfg(void **devPtr, size_t size, aclrtMemMallocPolicy policy, aclrtMallocConfig *cfg)
{
    char my_unique_id[SAL_UNIQUE_ID_BYTES];
    char mem_name[SAL_DMEM_UNIQUE_ID_BYTES];

    if (!devPtr)
    {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    /* 通过uniqueid的方式，申请桩函数设备内存的名字 */
    s32 ret = SalGetUniqueId(my_unique_id);

    if (ret != SAL_OK)
    {
        HCCL_ERROR("Generate sal_unique_id failed[%d]", ret);
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    /* UniqueID添加HCCL-mem-的前缀 */
    ret = snprintf_s(mem_name, SAL_DMEM_UNIQUE_ID_BYTES, SAL_DMEM_UNIQUE_ID_BYTES - 1,
                            "%s%s", SAL_DMEM_UNIQUE_ID_PREFIX, my_unique_id);

    if (-1 == ret)
    {
        HCCL_ERROR("snprintf_s failed[%d]", ret);
        return ACL_ERROR_RT_CONTEXT_NULL;
    }

    /* 封装的共享内存结构自带了管理信息,返回值是管理信息后的有效存储空间 */
    void* shm_buf = (void*)sal_share_memory_create(mem_name, size);

    if (nullptr == shm_buf)
    {
        HCCL_ERROR("sal_share_memory_create failed, device_memory_share_info is NULL");
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }

    HCCL_DEBUG("aclrtMallocWithCfg sal_share_memory_create[%s] OK", mem_name);

    *devPtr = shm_buf;

    return ACL_SUCCESS;
}

const char *aclrtGetSocName()
{
    if (chip_type_stub[0] == static_cast<s32>(DevType::DEV_TYPE_910B)) {
        return "Ascend910B1";
    } else if(chip_type_stub[0] == static_cast<s32>(DevType::DEV_TYPE_910_93)) {
        return "Ascend910_9391";
    } else if(gBoardId == 0x0000) {
        return "Ascend910";
    } else if (gBoardId == 0x2000) {  // 临时定义的 board id
        return "Ascend310P3";
    }
    return "Ascend910";
}

aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind)
{
    aclError ret;

    ret = (aclError)memcpy_s(dst, count, src, count);
    if (ret)
    {
        return ACL_ERROR_RT_PARAM_INVALID;
    }
    return ACL_SUCCESS;
}

rtError_t rtsLaunchKernelWithHostArgs(rtFuncHandle funcHandle, uint32_t blockDim, rtStream_t stm, rtKernelLaunchCfg_t *cfg,
    void *hostArgs, uint32_t argsSize, rtPlaceHolderInfo_t *placeHolderArray, uint32_t placeHolderNum)
{
    return RT_ERROR_NONE;
}

aclError aclmdlRICaptureThreadExchangeMode(aclmdlRICaptureMode *mode)
{
    return ACL_SUCCESS;
}

aclError aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t *value)
{
    *value = 1;
    return ACL_SUCCESS;
}

aclError aclrtBinaryLoadFromData(const void *data, size_t length,
    const aclrtBinaryLoadOptions *options, aclrtBinHandle *binHandle)
{
    static int i = 0;
    *binHandle = &i;
    return ACL_SUCCESS;
}

aclError aclrtGetFunctionAddr(aclrtFuncHandle funcHandle, void **aicAddr, void **aivAddr)
{
    static int i = 0;
    *aivAddr = &i;
    return ACL_SUCCESS;
}
