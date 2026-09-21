/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "flow_func_thread_pool.h"
#include <thread>
#include <algorithm>
#include <cstring>
#include <bitset>
#include <sys/wait.h>
#ifdef RUN_ON_DEVICE
#include <sys/ioctl.h>
#include <sys/stat.h>
#include "seccomp.h"
#endif
#include "pthread.h"
#include "ascend_hal.h"
#include "flow_func/flow_func_defines.h"
#include "common/udf_log.h"
#include "common/bind_cpu_utils.h"
#include "common/util.h"
#include "config/global_config.h"

namespace FlowFunc {
namespace {
constexpr uint32_t kMinCpuNum = 1U;
}
int32_t DeviceCpuInfo::GetCpuNum(uint32_t device_id, DEV_MODULE_TYPE model_type, uint32_t &num) {
    int64_t os_sched = 0L;
    drvError_t drv_ret = halGetDeviceInfo(device_id, static_cast<int32_t>(model_type),
        static_cast<int32_t>(INFO_TYPE_OS_SCHED), &os_sched);
    if (drv_ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("get device info INFO_TYPE_OS_SCHED failed, ret=%d, model_type=%d, device id=%u",
            static_cast<int32_t>(drv_ret), static_cast<int32_t>(model_type), device_id);
        return FLOW_FUNC_ERR_DRV_ERROR;
    }

    if (os_sched == 0L) {
        num = 0U;
        UDF_LOG_INFO("device[%u] model_type[%d] os sched is 0.", device_id, static_cast<int32_t>(model_type));
        return FLOW_FUNC_SUCCESS;
    }

    int64_t cpu_num = 0L;
    drv_ret = halGetDeviceInfo(device_id, static_cast<int32_t>(model_type), static_cast<int32_t>(INFO_TYPE_CORE_NUM),
        &cpu_num);
    if (drv_ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("get device info INFO_TYPE_CORE_NUM failed, ret=%d, model_type=%d, device id=%u",
            static_cast<int32_t>(drv_ret), static_cast<int32_t>(model_type), device_id);
        return FLOW_FUNC_ERR_DRV_ERROR;
    }
    if (cpu_num < 0L || cpu_num > 1024L) {
        UDF_LOG_ERROR("core num=%ld is not in[0, 1024], model_type=%d, device id=%u",
            cpu_num, static_cast<int32_t>(model_type), device_id);
        return FLOW_FUNC_ERR_DRV_ERROR;
    }
    num = static_cast<uint32_t>(cpu_num);
    return FLOW_FUNC_SUCCESS;
}


int32_t DeviceCpuInfo::Init(uint32_t device_id) {
    uint32_t aicpu_num = 0U;
    auto ret = GetCpuNum(device_id, MODULE_TYPE_AICPU, aicpu_num);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("get aicpu num failed, ret=%d, device id=%u", ret, device_id);
        return ret;
    }
    if (aicpu_num == 0U) {
        UDF_LOG_ERROR("get aicpu num is 0, device id=%u", device_id);
        return FLOW_FUNC_FAILED;
    }

    uint32_t ctrl_cpu_num = 0U;
    ret = GetCpuNum(device_id, MODULE_TYPE_CCPU, ctrl_cpu_num);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("get ctrl cpu num failed, ret=%d, device id=%u", ret, device_id);
        return ret;
    }

    uint32_t data_cpu_num = 0U;
    ret = GetCpuNum(device_id, MODULE_TYPE_DCPU, data_cpu_num);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("get data cpu num failed, ret=%d, device id=%u", ret, device_id);
        return ret;
    }
    uint32_t device_cpu_idx_begin = (aicpu_num + ctrl_cpu_num + data_cpu_num) * device_id;
    int64_t aicpu_bit_map = 0L;
    drvError_t drv_ret = halGetDeviceInfo(device_id, static_cast<int32_t>(MODULE_TYPE_AICPU),
        static_cast<int32_t>(INFO_TYPE_OCCUPY), &aicpu_bit_map);
    if (drv_ret != DRV_ERROR_NONE) {
        UDF_LOG_ERROR("get aicpu INFO_TYPE_OCCUPY failed, ret=%d, device id=%u",
            static_cast<int32_t>(drv_ret), device_id);
        return FLOW_FUNC_ERR_DRV_ERROR;
    }

    // uint64_t bit set num is 64
    std::bitset<64> aicpu_bit_set(static_cast<uint64_t>(aicpu_bit_map));
    if (aicpu_bit_set.count() != aicpu_num) {
        UDF_LOG_ERROR("get aicpu bitmap is not match aicpu num, device id=%u, aicpu_num=%u, aicpu_bit_map=%#lx",
            device_id, aicpu_num, aicpu_bit_map);
        return FLOW_FUNC_FAILED;
    }
    aicpu_num_ = aicpu_num;
    aicpu_indexes_.reserve(aicpu_num_);
    for (size_t idx = 0UL; idx < aicpu_bit_set.size(); ++idx) {
        if (aicpu_bit_set.test(idx)) {
            // con not overflow, as aicpu_bit_set.size max value is 64
            auto aicpuIdx = static_cast<uint32_t>(device_cpu_idx_begin + idx);
            aicpu_indexes_.emplace_back(aicpuIdx);
        }
    }

    UDF_LOG_INFO("device id[%u], aicpu num[%u], aicpuIndexs=%s.", device_id, aicpu_num_, ToString(aicpu_indexes_).c_str());
    return FLOW_FUNC_SUCCESS;
}

FlowFuncThreadPool::~FlowFuncThreadPool() {
    WaitForStop();
}

int32_t FlowFuncThreadPool::Init(const std::function<void(uint32_t)> &thread_func, uint32_t default_thread_num) {
    // device:get aicpu_num, host:use default_thread_num
    uint32_t thread_num = default_thread_num;
    if (GlobalConfig::Instance().IsRunOnAiCpu()) {
        uint32_t device_id = GlobalConfig::Instance().GetDeviceId();
        int32_t ret = device_cpu_info_.Init(device_id);
        if (ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("device cpu info init failed, device id=%u, ret=%d", device_id, ret);
            return ret;
        }
        thread_num = device_cpu_info_.GetAicpuNum();
    } else {
        const uint32_t max_cpu_num = std::thread::hardware_concurrency();
        thread_num = std::min(std::max(kMinCpuNum, max_cpu_num), thread_num);
    }
    wait_thread_ready_latch_.ResetCount(thread_num);
    for (uint32_t thread_idx = 0U; thread_idx < thread_num; ++thread_idx) {
        std::thread th([this, thread_idx, thread_func]() {
            SetThreadName(thread_idx);
            BindAicpu(thread_idx);
            thread_func(thread_idx);
        });
        workers_.emplace_back(std::move(th));
    }
    return FLOW_FUNC_SUCCESS;
}

uint32_t FlowFuncThreadPool::GetThreadNum() const {
    return static_cast<uint32_t>(workers_.size());
}

void FlowFuncThreadPool::WaitForStop() noexcept {
    for (auto &worker : workers_) {
        if (worker.joinable()) {
            try {
                worker.join();
            } catch (std::exception &e) {
                UDF_LOG_ERROR("join thread exception, error=%s.", e.what());
                // no break, need join other thread.
            }
        }
    }
    workers_.clear();
}

void FlowFuncThreadPool::WaitAllThreadReady() {
    wait_thread_ready_latch_.Await();
}

void FlowFuncThreadPool::ThreadReady(uint32_t thread_index) {
    UDF_LOG_INFO("Thread:%u is ready", thread_index);
    wait_thread_ready_latch_.CountDown();
}

void FlowFuncThreadPool::ThreadAbnormal(uint32_t thread_index) {
    UDF_LOG_INFO("Thread:%u is abnormal", thread_index);
    wait_thread_ready_latch_.CountDown();
}

void FlowFuncThreadPool::BindAicpu(uint32_t thread_idx) const {
    if (!GlobalConfig::Instance().IsRunOnAiCpu()) {
        return;
    }
    auto device_id = GlobalConfig::Instance().GetDeviceId();
    auto write_tid = WriteTidForAffinity();
    if (write_tid != FLOW_FUNC_SUCCESS) {
        UDF_LOG_WARN("write tid for affinity failed, ret[%d], thread_idx[%u], device id[%u]",
            write_tid, thread_idx, device_id);
    }
    const uint32_t phys_idx = device_cpu_info_.GetAicpuPhysIndex(thread_idx);
    BindCpuUtils::BindCore(phys_idx);
}

int32_t FlowFuncThreadPool::WriteTidForAffinity() {
    auto tid = GetTid();
    std::string command = "sudo /var/add_aicpu_tid_to_tasks.sh " + std::to_string(tid);
    sighandler_t old_handler = signal(SIGCHLD, SIG_DFL);

    // system() may fail due to  "No child processes".
    // if SIGCHLD is set to SIG_IGN, waitpid() may report ECHILD error because it cannot find the child process.
    // The reason is that the system() relies on a feature of the system, that is,
    // when the kernel initializes the process, the processing mode of SIGCHLD signal is SIG_IGN.
    const int32_t ret = system(command.c_str());
    if (ret != 0) {
        int32_t error_no = errno;
        UDF_LOG_WARN("execute cmd[%s] failed, ret[%d], err=%d[%s]", command.c_str(), ret, error_no,
            GetErrorNoStr(error_no).c_str());
        signal(SIGCHLD, old_handler);
        return FLOW_FUNC_FAILED;
    }
    signal(SIGCHLD, old_handler);
    return FLOW_FUNC_SUCCESS;
}

void FlowFuncThreadPool::SetThreadName(uint32_t thread_idx) const {
    std::string thread_name(thread_name_prefix_);
    (void)thread_name.append(std::to_string(thread_idx));
    int32_t ret = pthread_setname_np(pthread_self(), thread_name.c_str());
    if (ret != 0) {
        UDF_LOG_WARN("pthread_setname_np failed, thread_name=%s, ret=%d", thread_name.c_str(), ret);
    }
}

int32_t FlowFuncThreadPool::ThreadSecureCompute() {
#ifdef RUN_ON_DEVICE
    const int32_t filter_system_call[] = {
        SCMP_SYS(open), SCMP_SYS(close), SCMP_SYS(faccessat), SCMP_SYS(fstat),
        SCMP_SYS(futex), SCMP_SYS(getpid), SCMP_SYS(gettid), SCMP_SYS(ioctl),
        SCMP_SYS(lseek), SCMP_SYS(nanosleep), SCMP_SYS(openat), SCMP_SYS(newfstatat),
        SCMP_SYS(pselect6), SCMP_SYS(read), SCMP_SYS(readlinkat), SCMP_SYS(rt_sigaction),
        SCMP_SYS(mmap), SCMP_SYS(mprotect), SCMP_SYS(exit), SCMP_SYS(exit_group),
        SCMP_SYS(write), SCMP_SYS(getcpu), SCMP_SYS(madvise), SCMP_SYS(munmap),
        SCMP_SYS(sched_getaffinity), SCMP_SYS(set_robust_list), SCMP_SYS(rt_sigprocmask),
        SCMP_SYS(clock_nanosleep), SCMP_SYS(uname), SCMP_SYS(sysinfo),
        SCMP_SYS(tgkill), SCMP_SYS(fchmod), SCMP_SYS(rt_sigreturn), SCMP_SYS(brk)};

    const size_t system_call_size = sizeof(filter_system_call) / sizeof(int32_t);
    // filter enable system calls
    const scmp_filter_ctx ctx = seccomp_init(SCMP_ACT_ERRNO(1U));
    // A flag to specify if the kernel should log all filter actions taken except for the SCMP_ACT_ALLOW action.
    int32_t ret = seccomp_attr_set(ctx, SCMP_FLTATR_CTL_LOG, 1U);
    if (ret != 0) {
        UDF_LOG_ERROR("seccomp_attr_set failed, ret[%d].", ret);
    }
    for (size_t i = 0UL; i < system_call_size; ++i) {
        ret = seccomp_rule_add(ctx, SCMP_ACT_ALLOW, filter_system_call[i], 0U);
        if (ret != 0) {
            UDF_LOG_ERROR("Add the system call failed, ret[%d], filter_system_call[%zu], syscall number[%d].",
                ret, i, filter_system_call[i]);
            return FLOW_FUNC_FAILED;
        }
    }

    ret = seccomp_load(ctx);
    if (ret != 0) {
        UDF_LOG_ERROR("Execute seccomp_load failed, ret[%d].", ret);
        return FLOW_FUNC_FAILED;
    }
    UDF_LOG_INFO("Execute seccomp_load success.");
#endif
    return FLOW_FUNC_SUCCESS;
}
}
