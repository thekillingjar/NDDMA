/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/subprocess/subprocess_manager.h"
#include <thread>
#include <chrono>
#include <sys/prctl.h>
#include <signal.h>
#include "common/debug/ge_log.h"
#include "dflow/base/utils/process_utils.h"
#include "common/debug/log.h"
#include "dflow/inc/data_flow/model/pne_model.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "mmpa/mmpa_api.h"
#include "common/compile_profiling/ge_call_wrapper.h"

namespace ge {
namespace {
constexpr size_t kMaxArgsSize = 128UL;
constexpr int32_t kWatchSubProcThreadWaitTimeInMs = 200;
}  // namespace
SubprocessManager::SubprocessManager() : run_flag_(false) {}

SubprocessManager &SubprocessManager::GetInstance() {
  static SubprocessManager instance;
  return instance;
}

void SubprocessManager::MonitorSubprocess() {
  while (run_flag_) {
    {
      std::lock_guard<std::mutex> lk(callbacks_mu_);
      for (auto excpt_handle_callback = excpt_handle_callbacks_.begin();
            excpt_handle_callback != excpt_handle_callbacks_.end();) {
        const pid_t pid = excpt_handle_callback->first;
        int32_t wait_status = 0;
        const pid_t wait_pid = ProcessUtils::WaitPid(pid, &wait_status, WNOHANG | WUNTRACED | WCONTINUED);
        if (wait_pid == 0) {
          excpt_handle_callback++;
          continue;
        } else if (wait_pid == -1) {
          GELOGI("Sub process[%d] exit, status[%d].", static_cast<int32_t>(pid), wait_status);
          excpt_handle_callback->second(ProcStatus::EXITED);
          excpt_handle_callbacks_.erase(excpt_handle_callback++);
          planned_shutdown_.erase(pid);
          continue;
        } else if (WIFSTOPPED(wait_status)) {
          GEEVENT("Sub process[%d] stoped.", static_cast<int32_t>(pid));
          excpt_handle_callback->second(ProcStatus::STOPPED);
        } else if (WIFCONTINUED(wait_status)) {
          GEEVENT("Sub process[%d] continued.", static_cast<int32_t>(pid));
          excpt_handle_callback->second(ProcStatus::NORMAL);
        } else if (WIFSIGNALED(wait_status)) {
          const auto signal_num = WTERMSIG(wait_status);
          GEEVENT("Sub process[%d] was terminated by a signal[%d], status[%d]", static_cast<int32_t>(pid),
                  static_cast<int32_t>(signal_num), wait_status);
          GE_LOGE_IF(!planned_shutdown_[pid], "Sub process[%d] was terminated by a signal[%d], status[%d]",
                      static_cast<int32_t>(pid), static_cast<int32_t>(signal_num), wait_status);
        } else {
          // do nothing
        }
        excpt_handle_callback++;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kWatchSubProcThreadWaitTimeInMs));
  }
}

Status SubprocessManager::Initialize() {
  std::string bin_dir;
  GE_CHK_STATUS_RET_NOLOG(GetBinDir(bin_dir));
  executable_paths_.emplace(PNE_ID_CPU, bin_dir + "/host_cpu_executor_main");
  executable_paths_.emplace(PNE_ID_NPU, bin_dir + "/npu_executor_main");
  executable_paths_.emplace(PNE_ID_UDF, bin_dir + "/udf_executor");
  std::string flowgw_bin_dir;
  GE_CHK_STATUS_RET(GetFlowGwBinDir(bin_dir, flowgw_bin_dir), "Failed to get flowgw bin dir");
  if (!flowgw_bin_dir.empty()) {
    executable_paths_.emplace("queue_schedule", flowgw_bin_dir);
  }
  executable_paths_.emplace("deployer_daemon", bin_dir + "/deployer_daemon");
  run_flag_.store(true);
  watch_sub_proc_thread_ = std::thread([this]() {
    SET_THREAD_NAME(pthread_self(), "ge_dpl_watch");
    MonitorSubprocess();
  });
  return SUCCESS;
}

void SubprocessManager::Finalize() {
  run_flag_.store(false);
  if (watch_sub_proc_thread_.joinable()) {
    watch_sub_proc_thread_.join();
  }
  std::lock_guard<std::mutex> lk(callbacks_mu_);
  excpt_handle_callbacks_.clear();
  planned_shutdown_.clear();
  process_status_.clear();
}

void SubprocessManager::NotifySubprocessShutdown(const pid_t &pid) {
  if (pid == -1) {
    return;
  }
  int32_t wait_status = 0;
  // just check pid
  const auto ret = mmWaitPid(pid, &wait_status, M_WAIT_NOHANG);
  if (ret == 0) {
    {
      std::lock_guard<std::mutex> lk(callbacks_mu_);
      if (planned_shutdown_[pid]) {
        return;
      }
      auto callback = excpt_handle_callbacks_.find(pid);
      if (callback != excpt_handle_callbacks_.end()) {
        // shutdown set call back empty.
        callback->second = [](const ProcStatus &proc_status) { (void)proc_status; };
      }
      planned_shutdown_[pid] = true;
    }
    GEEVENT("Notify sub process[%d] to stop.", pid);
    // send term signal.
    (void)kill(pid, SIGTERM);
  }
}

Status SubprocessManager::ShutdownSubprocess(const pid_t &pid, const uint32_t wait_time_in_sec) {
  if (pid == -1) {
    return SUCCESS;
  }
  GEEVENT("Sub process[%d] is stopping.", pid);
  NotifySubprocessShutdown(pid);
  uint32_t times = 0U;
  int32_t wait_status = 0;
  do {
    const auto ret = mmWaitPid(pid, &wait_status, M_WAIT_NOHANG);
    if (ret != 0) {
      GEEVENT("Sub process[%d] stop success, ret[%d], status[%d].", pid, ret, wait_status);
      UnRegExcptHandleCallback(pid);
      return SUCCESS;
    }
    if (wait_time_in_sec != 0U) {
      (void)mmSleep(100U);
    }
    ++times;
  } while (times < wait_time_in_sec * 10U);

  (void)kill(pid, SIGKILL);
  const auto wait_ret = mmWaitPid(pid, &wait_status, M_WAIT_NOHANG);
  GEEVENT("Sub process[%d] stopped timeout and was killed, waitpid ret[%d], waitpid status[%d].", pid, wait_ret,
          wait_status);
  return FAILED;
}

Status SubprocessManager::ForkSubprocess(const SubprocessConfig &subprocess_config, pid_t &pid) {
  // subprocess_config will be pass to child process, so must pass by value.
  auto fut = pool_.commit([this, subprocess_config, &pid]() -> Status {
    // 1. get executable path
    const auto &it = executable_paths_.find(subprocess_config.process_type);
    if (it == executable_paths_.cend()) {
      GELOGE(FAILED, "Unsupported process type: %s", subprocess_config.process_type.c_str());
      return FAILED;
    }
    const auto &path = it->second;
    GELOGI("Executable path for [%s] is [%s]", subprocess_config.process_type.c_str(), path.c_str());

    // 2. format process args
    char_t *var_args[kMaxArgsSize] = {nullptr};
    std::vector<std::string> args_strings = FormatArgs(subprocess_config);
    GE_CHK_STATUS_RET_NOLOG(ToCmdlineArgs(args_strings, var_args));

    // 3. fork and execute
    pid = ProcessUtils::Fork();
    GE_CHK_BOOL_RET_STATUS(pid >= 0, FAILED, "Fork subprocess failed.");
    if (pid == 0) {  // in child process
      const Status ret = Execute(path, subprocess_config, var_args);
      if (ret != SUCCESS) {
        // fork need use _exit.
        _exit(-1);
      }
      return ret;
    }
    return SUCCESS;
  });
  GE_CHK_STATUS_RET_NOLOG(fut.get());
  return SUCCESS;
}

void SubprocessManager::RegExcptHandleCallback(pid_t pid, std::function<void(const ProcStatus &)> callback) {
  std::lock_guard<std::mutex> lk(callbacks_mu_);
  excpt_handle_callbacks_.emplace(pid, callback);
}

void SubprocessManager::UnRegExcptHandleCallback(pid_t pid) {
  std::lock_guard<std::mutex> lk(callbacks_mu_);
  excpt_handle_callbacks_.erase(pid);
}

Status SubprocessManager::Execute(const std::string &path,
                                  const SubprocessManager::SubprocessConfig &subprocess_config,
                                  char_t *const argv[]) {
  if (subprocess_config.death_signal > 0) {
    // swap user will clean PR_SET_PDEATHSIG,so need after swap user.
    ::prctl(PR_SET_PDEATHSIG, subprocess_config.death_signal);
  }

  constexpr int32_t kEnvNoOverwrite = 0;
  // only when AUTO_USE_UC_MEMORY does not exist, set it to 0(close).
  // it can be deleted when mbuf support double page table (all supported drivers).
  int32_t mmRet = 0;
  MM_SYS_SET_ENV(MM_ENV_AUTO_USE_UC_MEMORY, "0", kEnvNoOverwrite, mmRet);
  (void) mmRet;

  // set env variables
  for (const auto &kv : subprocess_config.envs) {
    const std::string &env_name = kv.first;
    const std::string &env_value = kv.second;
    const int32_t is_override = 1;
    (void) mmSetEnv(env_name.c_str(), env_value.c_str(), is_override);
    GELOGI("Set env[%s], value[%s].", env_name.c_str(), env_value.c_str());
  }

  // unset env variables
  for (const auto &env_name : subprocess_config.unset_envs) {
    (void) unsetenv(env_name.c_str());
    GELOGI("unset env[%s] success.", env_name.c_str());
  }

  auto ret = ProcessUtils::Execute(path, argv);
  GE_CHK_BOOL_RET_STATUS(ret == 0, FAILED, "Exec executable failed. path = %s, ret = %d", path.c_str(), ret);
  return SUCCESS;
}

std::vector<std::string> SubprocessManager::FormatArgs(const SubprocessManager::SubprocessConfig &subprocess_config) {
  std::vector<std::string> args_strings = subprocess_config.args; // copy
  FormatKvArgs(subprocess_config.kv_args, args_strings);
  return args_strings;
}

void SubprocessManager::FormatKvArgs(const std::map<std::string, std::string> &kv_args,
                                     std::vector<std::string> &out_args) {
  GELOGI("args option size = %zu.", kv_args.size());
  for (const auto &option_it : kv_args) {
    const auto &key = option_it.first;
    const auto &value = option_it.second;
    out_args.emplace_back(key + "=" += value);
    GELOGI("arg option=%s.", out_args.back().c_str());
  }
}

Status SubprocessManager::ToCmdlineArgs(const std::vector<std::string> &args_strings,
                                        char_t *var_args[]) {
  if ((args_strings.size()) >= kMaxArgsSize) {
    GELOGE(FAILED, "too many args size, var_args_size=%zu, var_args_size can not above %zu.",
           args_strings.size(), kMaxArgsSize);
    return FAILED;
  }
  // variable args
  size_t index = 0U;
  for (const auto &arg : args_strings) {
    var_args[index] = const_cast<char_t *>(arg.c_str());
    GELOGD("Set configure arg[%zu]=%s.", index, var_args[index]);
    ++index;
  }
  return SUCCESS;
}

bool SubprocessManager::FileExist(const std::string &file_path) {
  mmStat_t sb = {};
  if (mmStatGet(file_path.c_str(), &sb) != 0) {
    return false; // 文件不存在或无法访问
  }
  return S_ISREG(sb.st_mode); // 检查是否是常规文件（非目录）
}

Status SubprocessManager::GetFlowGwBinDir(const std::string &bin_dir, std::string &flowgw_bin_dir) {
  if (FileExist(bin_dir + "/host_queue_schedule")) {
    flowgw_bin_dir = RealPath((bin_dir + "/host_queue_schedule").c_str());
  }
  GELOGI("flowgw bin dir = %s.", flowgw_bin_dir.c_str());
  return SUCCESS;
}

Status SubprocessManager::GetBinDir(std::string &bin_dir) {
  mmDlInfo dl_info;
  if (mmDladdr(reinterpret_cast<void *>(&SubprocessManager::GetBinDir), &dl_info) != EN_OK) {
    const char_t *error = mmDlerror();
    GELOGE(FAILED, "Failed to read so path! errmsg:%s", error != nullptr ? error : "");
    return FAILED;
  }

  std::string lib_path = RealPath(dl_info.dli_fname);
  GE_CHK_BOOL_RET_STATUS(!lib_path.empty(), FAILED, "Failed to get lib dir realpath, %s", dl_info.dli_fname);
  std::string lib_dir = lib_path.substr(0U, lib_path.rfind('/') + 1U);
  GELOGD("lib path = %s, lib_dir = %s", lib_path.c_str(), lib_dir.c_str());
  bin_dir = lib_dir + "/../bin";
  bin_dir = RealPath(bin_dir.c_str());
  GE_CHK_BOOL_RET_STATUS(!bin_dir.empty(), FAILED, "Failed to get bin dir from lib dir:%s", lib_dir.c_str());
  GELOGI("bin dir = %s.", bin_dir.c_str());
  return SUCCESS;
}

Status SubprocessManager::HasFlowGw(bool &has_flowgw) {
  std::string bin_dir;
  GE_CHK_STATUS_RET(GetBinDir(bin_dir), "Failed to get bin dir");
  std::string flowgw_bin_dir;
  GE_CHK_STATUS_RET(GetFlowGwBinDir(bin_dir, flowgw_bin_dir), "Failed to get flowgw bin dir");
  has_flowgw = !flowgw_bin_dir.empty();
  return SUCCESS;
}
}  // namespace ge
