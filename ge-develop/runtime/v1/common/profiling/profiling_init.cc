/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "profiling_init.h"

#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "common/profiling/profiling_properties.h"
#include "runtime/base.h"
#include "common/profiling/command_handle.h"
#include "common/profiling/profiling_manager.h"
#include "mmpa/mmpa_api.h"
#include "aprof_pub.h"
#include "base/err_msg.h"

namespace {
using Json = nlohmann::json;

const char *const kTrainingTrace = "training_trace";
const char *const kFpPoint = "fp_point";
const char *const kBpPoint = "bp_point";
}

namespace ge {
ProfilingInit &ProfilingInit::Instance() {
  static ProfilingInit profiling_init;
  return profiling_init;
}

ge::Status ProfilingInit::Init(const std::map<std::string, std::string> &options) {
  GELOGI("Profiling init start.");

  struct MsprofOptions prof_conf = {};
  bool is_execute_profiling = false;
  const Status ret = InitProfOptions(options, prof_conf, is_execute_profiling);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Init][Profiling]Failed, error_code %u", ret);
    return ret;
  }
  ProfRegisterCtrlCallback();
  if (is_execute_profiling) {
    int32_t cb_ret = MsprofInit(static_cast<uint32_t>(MsprofCtrlCallbackType::MSPROF_CTRL_INIT_GE_OPTIONS), &prof_conf,
                                sizeof(MsprofOptions));
    if (cb_ret != 0) {
      GELOGE(FAILED, "[Call][MsprofInit]Failed, type %u, return %d",
             static_cast<uint32_t>(MsprofCtrlCallbackType::MSPROF_CTRL_INIT_GE_OPTIONS), cb_ret);
      REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char_t *>({"parameter", "value", "reason"}),
                                std::vector<const char_t *>({"profiling_options", prof_conf.options,
                                                             "The current value is not within the valid range."}));
      return FAILED;
    }
  } else {
    const auto df_ret = MsprofInit(MsprofCtrlCallbackType::MSPROF_CTRL_INIT_DYNA, nullptr, 0U);
    if (df_ret != 0) {
      GELOGE(FAILED, "[Call][MsprofInit]Failed, type %u, return %d",
             static_cast<uint32_t>(MsprofCtrlCallbackType::MSPROF_CTRL_INIT_DYNA), df_ret);
      return FAILED;
    }
  }

  GELOGI("Profiling init success");
  return SUCCESS;
}

ge::Status ProfilingInit::ProfRegisterCtrlCallback() {;
  rtProfCtrlHandle callback = ProfCtrlHandle;
  const auto ret = MsprofRegisterCallback(GE, callback);
  if (ret != MSPROF_ERROR_NONE) {
    GELOGE(FAILED, "Register CtrlCallBack failed");
    return FAILED;
  }
  return SUCCESS;
}

ge::Status ProfilingInit::InitProfOptions(const std::map<std::string, std::string> &options, MsprofOptions &prof_conf,
                                          bool &is_execute_profiling) {
  std::string prof_mode;
  std::string prof_option;

  auto iter = options.find(ge::OPTION_EXEC_PROFILING_MODE);
  if (iter != options.end()) {
    prof_mode = iter->second;
  }

  iter = options.find(ge::OPTION_EXEC_PROFILING_OPTIONS);
  if (iter != options.end()) {
    prof_option = iter->second;
  }

  if (prof_mode == "1" && !prof_option.empty()) {
    // enable profiling by ge option
    GE_ASSERT_EQ(strncpy_s(prof_conf.options, sizeof(prof_conf.options), prof_option.c_str(),
                           MSPROF_OPTIONS_DEF_LEN_MAX - 1U), EN_OK);
    GELOGI("The profiling in options is %s, %s. origin option: %s", prof_mode.c_str(), prof_conf.options,
           prof_option.c_str());
  } else {
    // enable profiling by env
    const char_t *env_profiling_mode = nullptr;
    MM_SYS_GET_ENV(MM_ENV_PROFILING_MODE, env_profiling_mode);
    // The env is invalid
    if ((env_profiling_mode == nullptr) || (strcmp("true", env_profiling_mode) != 0)) {
      return SUCCESS;
    }
    const char_t *prof_conf_options = nullptr;
    MM_SYS_GET_ENV(MM_ENV_PROFILING_OPTIONS, prof_conf_options);
    if (prof_conf_options != nullptr) {
      GE_ASSERT_EQ(strncpy_s(prof_conf.options, sizeof(prof_conf.options), prof_conf_options,
                             MSPROF_OPTIONS_DEF_LEN_MAX - 1U), EN_OK);
    }
    GELOGI("The profiling in env, prof_conf.options:[%s].", prof_conf.options);
     // set default value
    if (strcmp(prof_conf.options, "\0") == 0) {
      const char* default_options = "{\"output\":\"./\",\"training_trace\":\"on\",\"task_trace\":\"on\","\
        "\"hccl\":\"on\",\"aicpu\":\"on\",\"aic_metrics\":\"PipeUtilization\",\"msproftx\":\"off\"}";
      (void)strncpy_s(prof_conf.options, sizeof(prof_conf.options), default_options, strlen(default_options));
    }
  }

  // Parse json str for bp fp
  const Status ret = ParseOptions(prof_conf.options);
  if (ret != ge::SUCCESS) {
    GELOGE(ge::PARAM_INVALID, "[Parse][Options]Parse training trace param %s failed, error_code %u", prof_conf.options,
           ret);
    REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char_t *>({"parameter", "value", "reason"}),
                              std::vector<const char_t *>({"profiling_options", prof_conf.options,
                                                           "The current value is not within the valid range."}));
    return ge::PARAM_INVALID;
  }

  iter = options.find(ge::OPTION_EXEC_JOB_ID);
  if (iter != options.end()) {
    std::string job_id = iter->second;
    GE_ASSERT_EQ(strncpy_s(prof_conf.jobId, sizeof(prof_conf.jobId), job_id.c_str(), job_id.length()), EN_OK);
    GELOGI("Job id: %s, original job id: %s.", prof_conf.jobId, job_id.c_str());
  }

  is_execute_profiling = true;
  ProfilingProperties::Instance().SetExecuteProfiling(is_execute_profiling);
  return ge::SUCCESS;
}

ge::Status ProfilingInit::ParseOptions(const std::string &options) {
  if (options.empty()) {
    GELOGE(ge::PARAM_INVALID, "[Check][Param]Profiling options is empty");
    REPORT_INNER_ERR_MSG("E19999", "Profiling options is empty");
    return ge::PARAM_INVALID;
  }
  try {
    Json prof_options = Json::parse(options);
    GE_ASSERT_TRUE(prof_options.is_object());
    if (options.find(kTrainingTrace) == std::string::npos) {
      return ge::SUCCESS;
    }
    std::string training_trace;
    if (prof_options.contains(kTrainingTrace)) {
      training_trace = prof_options[kTrainingTrace];
    }
    if (training_trace.empty()) {
      GELOGI("Training trace will not take effect.");
      return ge::SUCCESS;
    }
    GELOGI("GE profiling training trace:%s", training_trace.c_str());
    if (training_trace != "on") {
      GELOGE(ge::PARAM_INVALID, "[Check][Param]Training trace param:%s is invalid.", training_trace.c_str());
      REPORT_INNER_ERR_MSG("E19999", "Training trace param:%s is invalid.", training_trace.c_str());
      return ge::PARAM_INVALID;
    }
    std::string fp_point;
    std::string bp_point;
    if (prof_options.contains(kFpPoint)) {
      fp_point = prof_options[kFpPoint];
    }
    if (prof_options.contains(kBpPoint)) {
      bp_point = prof_options[kBpPoint];
    }
    ProfilingProperties::Instance().SetTrainingTrace(true);
    ProfilingProperties::Instance().SetFpBpPoint(fp_point, bp_point);
  } catch (Json::exception &e) {
    GELOGE(FAILED, "[Check][Param]Json prof_conf options is invalid, catch exception:%s", e.what());
    REPORT_INNER_ERR_MSG("E19999", "Json prof_conf options is invalid, catch exception:%s", e.what());
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

void ProfilingInit::ShutDownProfiling() {
  if (ge::ProfilingProperties::Instance().ProfilingOn()) {
    GELOGI("Begin to shut down profiling. Report uinit to msprof");
    ProfilingProperties::Instance().ClearProperties();
  }

  int32_t cb_ret = MsprofFinalize();
  if (cb_ret != 0) {
    GELOGE(ge::FAILED, "call MsprofFinalize failed, return:%d", cb_ret);
    return;
  }

  GELOGI("Shut Down Profiling success.");
}

Status ProfilingInit::SetDeviceIdByModelId(uint32_t model_id, uint32_t device_id) {
  GELOGD("Set model id:%u and device id:%u to runtime.", model_id, device_id);
  const auto ret = MsprofSetDeviceIdByGeModelIdx(model_id, device_id);
  if (ret != MSPROF_ERROR_NONE) {
    GELOGE(ge::FAILED, "[Set][Device]Set Device id failed");
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

Status ProfilingInit::UnsetDeviceIdByModelId(uint32_t model_id, uint32_t device_id) {
  GELOGD("Unset model id:%u and device id:%u to runtime.", model_id, device_id);
  const auto ret = MsprofUnsetDeviceIdByGeModelIdx(model_id, device_id);
  if (ret != MSPROF_ERROR_NONE) {
    GELOGE(ge::FAILED, "[Set][Device]Set Device id failed");
    return ge::FAILED;
  }
  return ge::SUCCESS;
}
}  // namespace ge
