/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <csignal>
#include <pwd.h>
#include "securec.h"
#include "mmpa/mmpa_api.h"
#include "execute/flow_func_executor.h"
#include "model/flow_func_model.h"
#include "common/udf_log.h"
#include "common/bind_cpu_utils.h"
#include "common/data_utils.h"
#include "config/global_config.h"
#include "flow_func/flow_func_config_manager.h"
#include "toolchain/dump/udf_dump_manager.h"
#include "flow_func_drv_manager.h"

extern "C" {
/**
* @brief       : set stackcore file subdirectory to store files in the subdirectory
* @param [in]  : subdir      subdirectory name
* @return      : 0 success; -1 failed
*/
int __attribute__((weak)) StackcoreSetSubdirectory(const char *subdir);
}
namespace FlowFunc {
namespace {
constexpr uint32_t kCpuNumsPerDevice = 8U;
constexpr uint32_t kDeviceIdMaxValue = 1024U;
constexpr const char *kParamDeviceId = "--device_id=";
constexpr const char *kParamLoadPath = "--load_path=";
constexpr const char *kParamGroupName = "--group_name=";
constexpr const char *kParamBaseDir = "--base_dir=";
constexpr const char *kParamPhyDeviceId = "--phy_device_id=";
constexpr const char *kParamReqQueueId = "--req_queue_id=";
constexpr const char *kParamRspQueueId = "--rsp_queue_id=";
constexpr const char *kParamNpuSched = "--npu_sched=";
constexpr const char *kGlobalLogLevelEnvName = "ASCEND_GLOBAL_LOG_LEVEL";
constexpr const char *kModuleLogLevelEnvName = "ASCEND_MODULE_LOG_LEVEL";
constexpr const char *kUdfStackcoreSubDir = "udf";
constexpr const char *kDefaultDumpPath = "/var/log/npu/dump/udf";

FlowFunc::FlowFuncExecutor *g_flow_func_executor = nullptr;
std::map<std::string, std::string> g_args_option;
struct FlowFuncStartParams {
    std::string group_name;
    std::string load_path;
    uint32_t device_id = 0U;
    bool with_device_id = false;
    std::string base_dir;
    int32_t phy_device_id = 0;
    uint32_t request_queue_id = UINT32_MAX;
    uint32_t response_queue_id = UINT32_MAX;
    bool npu_sched = false;
};

/**
 * parse uint32 value from arg.
 * @param  arg   arg string value
 * @param  value uint32 value
 * @return       is parse success
 */
bool ParseUint32Value(const char *arg, uint32_t &value) {
    if (arg == nullptr) {
        return false;
    }
    char extra = '\0';
    int ret = sscanf_s(arg, "%u%c", &value, &extra, sizeof(char));
    // only one value is read and parse successful.
    if (ret == 1) {
        return true;
    }
    UDF_LOG_ERROR("parse failed, ret=%d, arg=%s", ret, arg);
    return false;
}

bool ParseInt32Value(const char *arg, int32_t &value) {
    if (arg == nullptr) {
        return false;
    }
    char extra = '\0';
    int ret = sscanf_s(arg, "%d%c", &value, &extra, sizeof(char));
    // only one value is read and parse successful.
    if (ret == 1) {
        return true;
    }
    UDF_LOG_ERROR("parse failed, ret=%d, arg=%s", ret, arg);
    return false;
}

void TransArray2ArgsOption(const char *arg) {
    const std::string str(arg);
    size_t idx = str.find_first_of('=');
    if (idx == std::string::npos) {
        return;
    }
    std::string key = str.substr(0, idx);
    std::string value = str.substr(idx + 1, str.length());
    if (key.empty() || value.empty()) {
        return;
    }
    // Update data
    g_args_option[key] = value;
}

bool ParseArgs(int32_t argc, char *argv[], FlowFuncStartParams &start_params) {
    UDF_LOG_INFO("Begin to parse args, argc=%d.", argc);
    for (int32_t i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (strncmp(kParamDeviceId, arg, strlen(kParamDeviceId)) == 0) {
            if (!ParseUint32Value(arg + strlen(kParamDeviceId), start_params.device_id)) {
                UDF_LOG_ERROR("Parse device_id from %s failed", arg);
                return false;
            }
            if (start_params.device_id > kDeviceIdMaxValue) {
                UDF_LOG_ERROR("device_id=%u in arg[%s] is over %u.", start_params.device_id, arg, kDeviceIdMaxValue);
                return false;
            }
            start_params.with_device_id = true;
            continue;
        }

        if (strncmp(kParamLoadPath, arg, strlen(kParamLoadPath)) == 0) {
            start_params.load_path.assign(arg + strlen(kParamLoadPath));
            continue;
        }

        if (strncmp(kParamGroupName, arg, strlen(kParamGroupName)) == 0) {
            start_params.group_name.assign(arg + strlen(kParamGroupName));
            continue;
        }

        if (strncmp(kParamBaseDir, arg, strlen(kParamBaseDir)) == 0) {
            start_params.base_dir.assign(arg + strlen(kParamBaseDir));
            continue;
        }

        if (strncmp(kParamPhyDeviceId, arg, strlen(kParamPhyDeviceId)) == 0) {
            if (!ParseInt32Value(arg + strlen(kParamPhyDeviceId), start_params.phy_device_id)) {
                UDF_LOG_ERROR("Parse phy_device_id from %s failed", arg);
                return false;
            }
            // device侧的phyDeviceId应该大于等于0，host侧没有device时phyDeviceId等于-1
            if (start_params.phy_device_id < -1) {
                UDF_LOG_ERROR("phydeviceId=%d in arg[%s] is invalid.", start_params.phy_device_id, arg);
                return false;
            }
            continue;
        }

        if (strncmp(kParamReqQueueId, arg, strlen(kParamReqQueueId)) == 0) {
            if (!ParseUint32Value(arg + strlen(kParamReqQueueId), start_params.request_queue_id)) {
                UDF_LOG_ERROR("Parse request queue id from %s failed", arg);
                return false;
            }
            continue;
        }

        if (strncmp(kParamRspQueueId, arg, strlen(kParamRspQueueId)) == 0) {
            if (!ParseUint32Value(arg + strlen(kParamRspQueueId), start_params.response_queue_id)) {
                UDF_LOG_ERROR("Parse response queue id from %s failed", arg);
                return false;
            }
            continue;
        }

        if (strncmp(kParamNpuSched, arg, strlen(kParamNpuSched)) == 0) {
            int32_t npu_sched = 0;
            if (!ParseInt32Value(arg + strlen(kParamNpuSched), npu_sched)) {
                UDF_LOG_ERROR("Parse npu_sched from %s failed", arg);
                return false;
            }
            if (npu_sched != 0 && npu_sched != 1) {
                UDF_LOG_ERROR("npu_sched=%d in arg[%s] is invalid, only support 0 and 1.", npu_sched, arg);
                return false;
            }
            start_params.npu_sched = (npu_sched == 1);
            continue;
        }
        TransArray2ArgsOption(arg);
    }

    if (!start_params.with_device_id) {
        UDF_LOG_ERROR("Param error. Must specify device_id with %s.", kParamDeviceId);
        return false;
    }

    if (start_params.load_path.empty()) {
        UDF_LOG_ERROR("Param error. Must specify load_path with %s.", kParamLoadPath);
        return false;
    }

    if (start_params.group_name.empty()) {
        UDF_LOG_ERROR("Param error. Must specify group_name with %s.", kParamGroupName);
        return false;
    }

    UDF_LOG_INFO("End to parse args.");
    return true;
}

bool InitDumpFromOption(const std::map<std::string, std::string> &options) {
    auto option_params = options;
    const auto &is_enable_dump = option_params["ge.exec.enableDump"];
    const auto &dump_path = option_params["ge.exec.dumpPath"];
    const auto &dump_step = option_params["ge.exec.dumpStep"];
    const auto &dump_mode = option_params["ge.exec.dumpMode"];
    uint32_t dev_id = GlobalConfig::Instance().IsOnDevice() ?
                     static_cast<unsigned int>(GlobalConfig::Instance().GetPhyDeviceId())
                     : GlobalConfig::Instance().GetDeviceId();

    int32_t pid = getpid();
    if (GlobalConfig::Instance().IsOnDevice()) {
        const char *host_pid = nullptr;
        MM_SYS_GET_ENV(MM_ENV_ASCEND_HOSTPID, host_pid);
        if (host_pid == nullptr) {
            UDF_LOG_WARN("InitDumpFromOption env ASCEND_HOSTPID is not set.");
        } else {
            std::string host_pid_str(host_pid);
            if ((!host_pid_str.empty()) && (!ConvertToInt32(host_pid_str, pid))) {
                UDF_LOG_ERROR("Failed to parse host pid: %s.", host_pid_str.c_str());
            }
        }
    }
    UdfDumpManager::Instance().SetHostPid(pid);
    UDF_LOG_INFO("InitDumpFromOption set host pid: %d.", pid);
    uint32_t logic_dev_id = GlobalConfig::Instance().GetDeviceId();
    UdfDumpManager::Instance().SetLogicDeviceId(logic_dev_id);
    UDF_LOG_INFO("InitDumpFromOption set logic device id: %u.", logic_dev_id);

    if (is_enable_dump.empty() || (is_enable_dump != "1")) {
        UDF_LOG_INFO("Dump is not enabled.");
        return true;
    }

    UdfDumpManager::Instance().EnableDump();
    const std::string default_dump_path(kDefaultDumpPath);
    std::string real_dump_path = dump_path.empty() ? (default_dump_path + "/udf" + std::to_string(getpid()) + "/") :
                                                   dump_path;
    UDF_LOG_INFO("Set dump path is %s.", real_dump_path.c_str());
    UdfDumpManager::Instance().SetDumpPath(real_dump_path);
    UdfDumpManager::Instance().SetDumpStep(dump_step);
    UdfDumpManager::Instance().SetDumpMode(dump_mode);
    UdfDumpManager::Instance().SetDeviceId(dev_id);
    int32_t initRet = UdfDumpManager::Instance().Init();
    if (initRet != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Udf dump init failed, ret=%d", initRet);
        return false;
    }
    UDF_LOG_INFO("InitDumpFromOption, dump_path is %s, dump_mode is %s, dev_id is %u.", real_dump_path.c_str(),
                 dump_mode.c_str(), dev_id);
    return true;
}

bool InitLogMode() {
    if (!GlobalConfig::Instance().IsOnDevice()) {
        UDF_RUN_LOG_INFO("InitLogMode is not run on device.");
        return true;
    }
    const char *env_value = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_HOSTPID, env_value);
    if (env_value == nullptr) {
        UDF_RUN_LOG_INFO("InitLogMode env ASCEND_HOSTPID is not set.");
        return true;
    }
    UDF_RUN_LOG_INFO("InitLogMode env ASCEND_HOSTPID = %s.", env_value);
    env_value = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_LOG_SAVE_MODE, env_value);
    if (env_value == nullptr) {
        UDF_RUN_LOG_INFO("InitLogMode env ASCEND_LOG_SAVE_MODE is not set.");
        return true;
    }
    UDF_RUN_LOG_INFO("InitLogMode env ASCEND_LOG_SAVE_MODE = %s.", env_value);

    // pid 0, log module will get the hostpid form env ASCEND_HOSTPID
    LogAttr log_attr = {};
    log_attr.type = ProcessType::APPLICATION;
    log_attr.pid = 0U;
    log_attr.deviceId = GlobalConfig::Instance().GetDeviceId();
    const int32_t log_ret = DlogSetAttr(log_attr);
    if (log_ret != EN_OK) {
        UDF_LOG_ERROR("Set dlog attr failed, ret is %d.", log_ret);
        return false;
    }
    return true;
}

bool InitializeMaintenanceFromOption(const std::map<std::string, std::string> &options) {
    bool ret = InitLogMode();
    if (!ret) {
        UDF_LOG_ERROR("InitLogMode failed.");
        return false;
    }
    ret = InitDumpFromOption(options);
    if (!ret) {
        UDF_LOG_ERROR("InitDumpFromOption failed.");
        return false;
    }
    return true;
}

/**
 * handle SIGTERM sig.
 * @return  void
 */
void HandleSignal(int32_t sig) {
    (void)sig;
    if (g_flow_func_executor != nullptr) {
        g_flow_func_executor->Stop(true);
    }
}

void PrintLogEnv() {
    const char *env_value = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_GLOBAL_LOG_LEVEL, env_value);
    if (env_value != nullptr) {
        UDF_RUN_LOG_INFO("[PrintLogLevel] env %s = %s.", kGlobalLogLevelEnvName, env_value);
    }
    env_value = nullptr;

    MM_SYS_GET_ENV(MM_ENV_ASCEND_MODULE_LOG_LEVEL, env_value);
    if (env_value != nullptr) {
        UDF_RUN_LOG_INFO("[PrintLogLevel] env %s = %s.", kModuleLogLevelEnvName, env_value);
    }
}

bool IsBuiltinUdfUser() {
    uid_t uid = geteuid();
    // no multi thread here
    struct passwd *pw = getpwuid(uid);
    const std::string build_in_udf_running_user("HwHiAiUser");
    UDF_RUN_LOG_INFO("udf running user=%s", pw->pw_name);
    return build_in_udf_running_user == pw->pw_name;
}
}
}

using namespace FlowFunc;

#ifndef UDF_TEST

int32_t main(int32_t argc, char *argv[])
#else
int32_t FlowFuncTestMain(int32_t argc, char* argv[])
#endif
{
    UDF_RUN_LOG_INFO("Flow func executor start.");
    PrintLogEnv();
    FlowFuncStartParams start_param = {};
    if (!ParseArgs(argc, argv, start_param)) {
        UDF_LOG_ERROR("Parse args failed.");
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }

    FlowFuncConfigManager::SetConfig(
        std::shared_ptr<FlowFuncConfig>(&GlobalConfig::Instance(), [](FlowFuncConfig *) {}));
    GlobalConfig::Instance().SetDeviceId(start_param.device_id);
    GlobalConfig::Instance().SetMemGroupName(start_param.group_name);
    GlobalConfig::Instance().SetBaseDir(start_param.base_dir);
    GlobalConfig::Instance().SetPhyDeviceId(start_param.phy_device_id);
    GlobalConfig::Instance().SetRunningDeviceId(start_param.phy_device_id);
    GlobalConfig::Instance().SetMessageQueueIds(start_param.request_queue_id, start_param.response_queue_id);
    GlobalConfig::Instance().SetNpuSched(start_param.npu_sched);

    bool is_run_on_aicpu = false;
    if (GlobalConfig::Instance().IsOnDevice()) {
        if (StackcoreSetSubdirectory == nullptr) {
            UDF_LOG_WARN("stackcore api StackcoreSetSubdirectory does not exist.");
        } else {
            int32_t stackcore_ret = StackcoreSetSubdirectory(kUdfStackcoreSubDir);
            UDF_RUN_LOG_INFO("set stack core dir[%s] end, result=%d.", kUdfStackcoreSubDir, stackcore_ret);
        }
        // device run on aicpu
        is_run_on_aicpu = true;
    }
    GlobalConfig::Instance().SetRunOnAiCpu(is_run_on_aicpu);

    if (is_run_on_aicpu) {
        // default main thread run on device ctrl cpu
        BindCpuUtils::BindCore(start_param.device_id * kCpuNumsPerDevice);
    }
    if (!InitializeMaintenanceFromOption(g_args_option)) {
        UDF_LOG_ERROR("Failed to initialize maintenance from options.");
        return FLOW_FUNC_FAILED;
    }

    if (GlobalConfig::Instance().IsOnDevice()) {
        bool is_built_in = IsBuiltinUdfUser();
        GlobalConfig::Instance().SetLimitRunBuiltinUdf(is_built_in);
        if (!is_built_in) {
            GlobalConfig::Instance().SetLimitThreadInitFunc([]() { return FlowFuncThreadPool::ThreadSecureCompute(); });
        }
    }

    auto models = FlowFunc::FlowFuncModel::ParseModels(start_param.base_dir + start_param.load_path);
    if (models.empty()) {
        UDF_LOG_ERROR("Failed to parse models failed.");
        return FLOW_FUNC_FAILED;
    }
    int32_t ret = FlowFunc::FlowFuncDrvManager::Instance().Init();
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Failed to init flow func drv manager. ret=%d.", ret);
        return ret;
    }
    FlowFunc::FlowFuncExecutor flow_func_executor;
    ret = flow_func_executor.Init(models);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Failed to init flow func executor. ret=%d.", ret);
        return ret;
    }
    ret = flow_func_executor.Start();
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Failed to start flow func executor. ret=%d.", ret);
    } else {
        g_flow_func_executor = &flow_func_executor;
        std::signal(SIGTERM, HandleSignal);
    }
    // wait for finish.
    flow_func_executor.WaitForStop();
    g_flow_func_executor = nullptr;
    UdfDumpManager::Instance().ClearDumpInfo();
    flow_func_executor.Destroy();
    FlowFunc::FlowFuncDrvManager::Instance().Finalize();
    UDF_RUN_LOG_INFO("Flow func executor exit.");
    return ret;
}
