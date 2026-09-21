/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "inc/ffts_configuration.h"
#include <dlfcn.h>
#include <algorithm>
#include <sstream>
#include "mmpa/mmpa_api.h"
#include "inc/ffts_log.h"
#include "inc/ffts_type.h"
#include "common/fe_type_utils.h"
#include "common/string_utils.h"
#include "common/aicore_util_constants.h"
#include "ge/ge_api_types.h"
#include "graph/ge_context.h"
#include "graph/tuning_utils.h"
#include "graph/ge_local_context.h"

namespace ffts {
const std::string Configuration::config_file_relative_path_ = "ffts_config/ffts.ini";
const std::string Configuration::config_ops_relative_path_ = "../../opp/";
const std::string kConfigFileSuffix = ".ini";
const std::string kRunMode = "runmode";
const uint32_t kDefaultHbmRatio = 50;
const std::string kNoFFTS = "no_ffts";

Configuration::Configuration()
    : is_init_(false),
      content_map_(),
      hbm_ratio_(0),
      inner_cmo_node_distance_threshold_(0),
      inner_write_back_data_size_lower_threshold_(0),
      inner_write_back_data_size_higher_threshold_(0),
      inner_prefetch_data_size_lower_threshold_(0),
      inner_prefetch_data_size_upper_threshold_(0),
      outer_prefetch_data_size_lower_threshold_(0),
      outer_prefetch_data_size_upper_threshold_(0),
      prefetch_consumer_cnt_(0),
      prefetch_ost_num_(0),
      cmaint_ost_num_(0),
      aic_prefetch_lower_(0),
      aic_prefetch_upper_(0),
      aiv_prefetch_lower_(0),
      aiv_prefetch_upper_(0),
      data_split_unit_(0) {}

Configuration::~Configuration() {}

Configuration &Configuration::Instance() {
  static Configuration config;
  return config;
}

Status Configuration::Initialize() {
  if (is_init_) {
    return SUCCESS;
  }

  FFTS_LOGD("FFTS Configuration begin to initialize.");

  Status status = InitLibPath();
  if (status != SUCCESS) {
    FFTS_LOGW("[FFTSConfig][Init] Failed to initialize the real path of libffts.");
    return FAILED;
  }

  status = LoadConfigFile();
  if (status != SUCCESS) {
    FFTS_LOGI("[FFTSConfig][Init] Load the data from configuration file unsuccessful.");
    return status;
  }

  status = InitCMOThreshold();
  if (status != SUCCESS) {
    REPORT_FFTS_ERROR("[FFTSConfig][Init] Failed to load the cmo from configuration file.");
    return status;
  }
  is_init_ = true;
  FFTS_LOGI("FFTS Initialization of Configuration end.");
  return SUCCESS;
}

Status Configuration::Finalize() {
  FFTS_LOGD("Finalizing the Configuration.");
  if (!is_init_) {
    FFTS_LOGD("Configuration has not been initialized, finalize is not allowed.");
    return SUCCESS;
  }

  content_map_.clear();
  is_init_ = false;
  FFTS_LOGI("FFTS Configuration finalize successfully.");
  return SUCCESS;
}

Status Configuration::InitLibPath() {
  Dl_info dl_info;
  Configuration &(*instance_ptr)() = &Configuration::Instance;
  if (dladdr(reinterpret_cast<void *>(instance_ptr), &dl_info) == 0) {
    FFTS_LOGW("[GraphOpt][Init][InitLibPath] Failed to get so file path.");
    return FAILED;
  } else {
    std::string so_path = dl_info.dli_fname;
    FFTS_LOGD("Libffts so file path is %s.", so_path.c_str());

    if (so_path.empty()) {
      FFTS_LOGW("[GraphOpt][Init][InitLibPath] So file path is empty.");
      return FAILED;
    }

    lib_path_ = fe::GetRealPath(so_path);
    int32_t pos = lib_path_.rfind('/');
    if (pos < 0) {
      FFTS_LOGW("[GraphOpt][Init][InitLibPath] The current .so file path does not contain /.");
      return FAILED;
    }

    lib_path_ = lib_path_.substr(0, pos + 1);
  }
  FFTS_LOGD("The real path of libffts is: %s.", lib_path_.c_str());
  return SUCCESS;
}

Status Configuration::LoadConfigFile() {
  FFTS_LOGD("Begin to Load FFTS ConfigFile.");
  std::string config_file_real_path = fe::GetRealPath(lib_path_ + config_file_relative_path_);
  FFTS_LOGI("The real path of ffts.ini is %s.", config_file_real_path.c_str());

  if (config_file_real_path.empty()) {
    FFTS_LOGW("[GraphOpt][InitOpsPath] FFTS configuration file path is invalid.");
    return FAILED;
  }

  std::ifstream ifs(config_file_real_path);
  if (!ifs.is_open()) {
    FFTS_LOGW("[GraphOpt][InitOpsPath] Open configuration file failed. The file does not exist or has been opened.");
    return FAILED;
  }

  content_map_.clear();
  std::string line;
  while (std::getline(ifs, line)) {
    if (line.empty()) {
      continue;
    }
    line = fe::StringUtils::Trim(line);
    if (line.empty() || line.find('#') == 0) {
      continue;
    }

    size_t pos_of_equal = line.find('=');
    if (pos_of_equal == std::string::npos) {
      continue;
    }
    std::string key = line.substr(0, pos_of_equal);
    std::string value = line.substr(pos_of_equal + 1);
    key = fe::StringUtils::Trim(key);
    value = fe::StringUtils::Trim(value);
    if (!key.empty()) {
      content_map_.emplace(key, value);
    }
  }
  ifs.close();
  FFTS_LOGD("End of loadConfigFile.");
  return SUCCESS;
}

Status Configuration::GetStringValue(const std::string &key, std::string &return_value) const {
  auto iter = content_map_.find(key);
  if (iter == content_map_.end()) {
    FFTS_LOGW("Can not find the value of key %s.", key.c_str());
    return FAILED;
  }

  return_value = iter->second;
  return SUCCESS;
}

uint32_t Configuration::GetFftsMergeLimitHBMRatio() {
  if (hbm_ratio_ == 0) {
    hbm_ratio_ = kDefaultHbmRatio;
    const char_t *hbm_ratio = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HBM_RATIO, hbm_ratio);
    if (hbm_ratio != nullptr) {
      if (hbm_ratio[0] != '0') {
        try {
          hbm_ratio_ = std::stoi(hbm_ratio);
        } catch (...) {
          hbm_ratio_ = kDefaultHbmRatio;
          std::string hbm_ratio_str = string(hbm_ratio);
          FFTS_LOGW("The value [%s] for the environment variable [HBM_RATIO] is invalid.", hbm_ratio_str.c_str());
        }
      }
      FFTS_LOGI("GetFftsMergeLimitHBMRatio hbm ratio:%u.", hbm_ratio_);
      return hbm_ratio_;
    }
  } else {
    FFTS_LOGI("GetFftsMergeLimitHBMRatio hbm has value, ratio:%u.", hbm_ratio_);
  }
  return hbm_ratio_;
}


std::string Configuration::GetFftsOptimizeEnable() {
  if (runmode_.empty()) {
    runmode_ = kNoFFTS; // In case of repeated access to this func to get env
    const char_t *enable_ascend_enhance = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_ENHANCE_ENABLE, enable_ascend_enhance);
    if (enable_ascend_enhance != nullptr) {
      FFTS_LOGI("enable_ascend_enhance is set to %c.", enable_ascend_enhance[0]);
      if (enable_ascend_enhance[0] == '1') {
        runmode_ = kFftsFlagAllSubgraph;
      } else if (enable_ascend_enhance[0] == '2') {
        runmode_ = kFftsFlagDefault;
      }
    }
  } else {
    FFTS_LOGD("Runmode switch already has value [%s].", runmode_.c_str());
  }
  if (runmode_ != kNoFFTS) {
    std::string opt_val;
    (void)ge::GetThreadLocalContext().GetOo().GetValue(fe::kComLevelO1Opt, opt_val);
    if (opt_val == fe::kStrFalse) {
      FFTS_LOGI("Ffts plus force close in current compile level.");
      runmode_ = kNoFFTS;
    }
  }

  return runmode_;
}

void Configuration::SetFftsOptimizeEnable() {
  std::lock_guard<std::mutex> lock_guard(mode_lock_);
  runmode_ = "ffts";
}

Status Configuration::InitSingleConfig(const string &key, size_t &value) const {
  string value_str;
  Status status = GetStringValue(key, value_str);
  if (status != SUCCESS || value_str.empty()) {
    return FAILED;
  }
  try {
    value = stoi(value_str);
  } catch (...) {
    FFTS_LOGW("Convert %s to int value failed.", value_str.c_str());
  }
  FFTS_LOGD("Get %s: %zu.", key.c_str(), value);
  return SUCCESS;
}

Status Configuration::InitCMOThreshold() {
  inner_cmo_node_distance_threshold_ = 0;
  inner_write_back_data_size_lower_threshold_ = 0;
  inner_write_back_data_size_higher_threshold_ = 0;
  inner_prefetch_data_size_lower_threshold_ = 0;
  inner_prefetch_data_size_upper_threshold_ = 0;
  outer_prefetch_data_size_lower_threshold_ = 0;
  outer_prefetch_data_size_upper_threshold_ = 0;
  prefetch_consumer_cnt_ = 1;

  (void)InitSingleConfig("cmo.inner_cmo_node_distance.threshold", inner_cmo_node_distance_threshold_);
  (void)InitSingleConfig("cmo.inner_write_back_data_size.threshold.higher_bound",
                         inner_write_back_data_size_higher_threshold_);
  (void)InitSingleConfig("cmo.inner_write_back_data_size.threshold.lower_bound",
                         inner_write_back_data_size_lower_threshold_);
  (void)InitSingleConfig("cmo.inner_prefetch_data_size.threshold.upper_bound",
                         inner_prefetch_data_size_upper_threshold_);
  (void)InitSingleConfig("cmo.inner_prefetch_data_size.threshold.lower_bound",
                         inner_prefetch_data_size_lower_threshold_);
  (void)InitSingleConfig("cmo.outer_prefetch_data_size.threshold.upper_bound",
                         outer_prefetch_data_size_upper_threshold_);
  (void)InitSingleConfig("cmo.outer_prefetch_data_size.threshold.lower_bound",
                         outer_prefetch_data_size_lower_threshold_);

  (void)InitSingleConfig("cmo.prefetch_consumer_cnt", prefetch_consumer_cnt_);

  (void)InitSingleConfig("cmo.prefetch_ost_num", prefetch_ost_num_);
  (void)InitSingleConfig("cmo.cmaint_ost_num", cmaint_ost_num_);

  (void)InitSingleConfig("cmo.aic.prefetch_lower", aic_prefetch_lower_);
  (void)InitSingleConfig("cmo.aic.prefetch_upper", aic_prefetch_upper_);

  (void)InitSingleConfig("cmo.aiv.prefetch_lower", aiv_prefetch_lower_);
  (void)InitSingleConfig("cmo.aiv.prefetch_upper", aiv_prefetch_upper_);

  (void)InitSingleConfig("cmo.data_split_unit", data_split_unit_);
  return SUCCESS;
}


size_t Configuration::GetInnerCmoNodeDistTh() const {
  return inner_cmo_node_distance_threshold_;
}

size_t Configuration::GetWriteBackDataSizeLowerTh() const {
  return inner_write_back_data_size_lower_threshold_;
}

size_t Configuration::GetWriteBackDataSizeHigherTh() const {
  return inner_write_back_data_size_higher_threshold_;
}

size_t Configuration::GetInnerPrefetchDataSizeLowerTh() const {
  return inner_prefetch_data_size_lower_threshold_;
}

size_t Configuration::GetInnerPrefetchDataSizeUpperTh() const {
  return inner_prefetch_data_size_upper_threshold_;
}

size_t Configuration::GetOuterPrefetchDataSizeLowerTh() const {
  return outer_prefetch_data_size_lower_threshold_;
}

size_t Configuration::GetOuterPrefetchDataSizeUpperTh() const {
  return outer_prefetch_data_size_upper_threshold_;
}

size_t Configuration::GetPrefetchConsumerCnt() const {
  return prefetch_consumer_cnt_;
}

size_t Configuration::GetPrefetchOstNum() const {
  return prefetch_ost_num_;
}

size_t Configuration::GetCmaintOstNum() const {
  return cmaint_ost_num_;
}

size_t Configuration::GetAicPrefetchLower() const {
  return aic_prefetch_lower_;
}

size_t Configuration::GetAicPrefetchUpper() const {
  return aic_prefetch_upper_;
}

size_t Configuration::GetAivPrefetchLower() const {
  return aiv_prefetch_lower_;
}

size_t Configuration::GetAivPrefetchUpper() const {
  return aiv_prefetch_upper_;
}

size_t Configuration::GetDataSplitUnit() const {
  return data_split_unit_;
}
}  // namespace ffts
