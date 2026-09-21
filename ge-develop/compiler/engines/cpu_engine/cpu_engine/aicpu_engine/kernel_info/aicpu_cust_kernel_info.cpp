/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_cust_kernel_info.h"

#include <dirent.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include "config/ops_json_file.h"
#include "error_code/error_code.h"
#include "util/constant.h"
#include "util/log.h"
#include "util/util.h"

using namespace std;
namespace aicpu {
KernelInfoPtr AicpuCustKernelInfo::instance_ = nullptr;
const std::string kSplitSeparator = ":";
const char_t *const kCustOppEnvName = "ASCEND_CUSTOM_OPP_PATH";
const char_t *const kOppEnvName = "ASCEND_OPP_PATH";
constexpr int kJsonIndent = 2;

KernelInfoPtr AicpuCustKernelInfo::Instance() {
  static once_flag flag;
  call_once(flag,
      [&]() { instance_.reset(new (std::nothrow) AicpuCustKernelInfo); });
  return instance_;
}

bool AicpuCustKernelInfo::ReadOpInfoFromJsonFile() {
  std::string real_cust_aicpu_ops_file_path;
  std::string custom_path;
  const char* custom_path_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_CUSTOM_OPP_PATH, custom_path_env);
  const char* path_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, path_env);
  std::string env_path;
  if (custom_path_env != nullptr) {
    env_path = std::string(custom_path_env);
    CHECK_RES_BOOL(ReadCustOpInfoFromJsonFile(env_path), false,
                   AICPU_REPORT_INNER_ERR_MSG("Read cust op json file from ASCEND_CUSTOM_OPP_PATH failed."));
  }
  if (path_env != nullptr) {
    env_path = std::string(path_env);
    custom_path = env_path + kAicpuCustPathPrefix;
    // 如果/vendors/config.ini存在且非空认为是新自定义算子包路径，否则读取老路径。
    AICPUE_LOGI("path env set to %s, custom path is %s.", env_path.c_str(), custom_path.c_str());
    if (GetCustJsonFile(custom_path)) {
      return true;
    } else if (GetBuiltInCustJsonFile(env_path)) {
      return true;
    } else {
      real_cust_aicpu_ops_file_path = env_path + kAicpuCustOpsFileBasedOnEnvPathOld;
    }
  } else {
    std::string file_path = GetOpsPath(reinterpret_cast<void*>(&AicpuCustKernelInfo::Instance));
    custom_path = file_path + kAicpuCustPathPrefixOld;
    if (GetCustJsonFile(custom_path)) {
      return true;
    } else {
      real_cust_aicpu_ops_file_path = file_path + kAicpuCustOpsFileRelativePathOld;
    }
  }
  SetJsonPath(real_cust_aicpu_ops_file_path);
  AICPUE_LOGI("AicpuCustKernelInfo real_cust_aicpu_ops_file_path is %s",
              real_cust_aicpu_ops_file_path.c_str());
  std::ifstream ifs(real_cust_aicpu_ops_file_path);
  AICPU_IF_BOOL_EXEC(
      !ifs.is_open(),
      AICPUE_LOGW("Open %s failed, please check whether this file exist",
                  real_cust_aicpu_ops_file_path.c_str());
      op_info_json_file_ = {};
      return true);
  AICPU_CHECK_FALSE_EXEC(
      OpsJsonFile::Instance().ParseUnderPath(
          real_cust_aicpu_ops_file_path, op_info_json_file_).state == ge::SUCCESS,
      AICPU_REPORT_INNER_ERR_MSG("Parse json file[%s] failed.",
          real_cust_aicpu_ops_file_path.c_str());
      return false)
  return true;
}

bool AicpuCustKernelInfo::GetCustJsonFile(const std::string &path) {
  AICPUE_LOGD("Begin to load opp custom file.");
 
  std::string real_path = path + kConfigFile;
  std::string cust_ops_file_path;
  std::vector<string> custom_user;
  // 只加载config.ini文件中的用户算子信息库
  if (ReadConfigFile(real_path, custom_user)) {
    AICPUE_LOGD("AicpuCustKernelInfo custom_user.size() = %zu.", custom_user.size());
    if (custom_user.size() < 1) {
      return false;
    }
    has_cust_op_ = true;
    for (size_t i = 0; i < custom_user.size(); i++) {
      cust_ops_file_path = path + '/'+ custom_user[i] + kAicpuCustOpsFilePath;
      std::string name = path + '/'+ custom_user[i];
      if (!ReadCustJsonFile(name, cust_ops_file_path)) {
        continue;
      }
    }
  } else {
    return false;
  }

  AICPUE_LOGD("custop_info_json_file_ size is %zu .", custop_info_json_file_.size());

  return true;
}

bool AicpuCustKernelInfo::LoadJsonFile(const std::string &file_path, nlohmann::json &out_json) const {
  std::ifstream ifs(file_path);
  if (!ifs.is_open()) {
    return false;
  }
  try {
    ifs >> out_json;
  } catch (const std::exception &e) {
    AICPUE_LOGW("json parse on %s.", file_path.c_str());
    return false;
  }
  return true;
}

bool AicpuCustKernelInfo::SaveJsonFile(const std::string &file_path, const nlohmann::json &content) const {
  if (content.empty()) {
    AICPUE_LOGW("json content is empty skip saving file: %s.", file_path.c_str());
    return false;
  }

  std::ofstream ofs(file_path, std::ios::trunc);
  if (!ofs.is_open()) {
    AICPUE_LOGW("try to write file to: %s.", file_path.c_str());
    return false;
  }

  ofs << content.dump(kJsonIndent);
  AICPUE_LOGI("save json content to %s file successful.", file_path.c_str());
  return true;
}

bool AicpuCustKernelInfo::IsAicpuJsonFile(const std::string &filename) const {
  const std::string prefix = "aicpu_";
  const std::string suffix = ".json";

  if (filename.size() < prefix.size() + suffix.size()) {
    return false;
  }

  const bool has_prefix = filename.compare(0, prefix.size(), prefix) == 0;
  const bool has_suffix = filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) == 0;
  AICPUE_LOGI("check is aicpu json file which has prefix is %d, has suffix is %d.", has_prefix, has_suffix);

  return has_prefix && has_suffix;
}

bool AicpuCustKernelInfo::RecordBuiltInCustConfigFile(const std::string &builtin_custom_config_dir) const {
  DIR *dir = opendir(builtin_custom_config_dir.c_str());
  if (dir == nullptr) {
    return false;
  }

  AICPUE_LOGI("builtin custom config directory is %s", builtin_custom_config_dir.c_str());
  nlohmann::json merged_json;
  struct dirent *entry;

  while ((entry = readdir(dir)) != nullptr) {
    std::string file_name(entry->d_name);

    if (entry->d_type != DT_REG) {
      continue;
    }

    if (!IsAicpuJsonFile(file_name)) {
      continue;
    }

    const std::string file_path = builtin_custom_config_dir + "/" + file_name;
    nlohmann::json current;
    if (!LoadJsonFile(file_path, current)) {
      AICPUE_LOGW("Skip invalid file: %s.", file_path.c_str());
      continue;
    }

    for (auto it = current.begin(); it != current.end(); ++it) {
      merged_json[it.key()] = it.value();
    }

    AICPUE_LOGI("json file merged to: %s.", file_name.c_str());
  }

  if (closedir(dir) != 0) {
    AICPUE_LOGW("close dir failed, config path: [%s].", builtin_custom_config_dir.c_str());
  }

  const std::string output_path = builtin_custom_config_dir + kAicpuBuiltInCustOpsFilePath;
  if (!SaveJsonFile(output_path, merged_json)) {
    AICPUE_LOGW("save json to %s not success.", output_path.c_str());
    return false;
  }

  AICPUE_LOGI("Successfully merged %zu files ops into %s.", merged_json.size(), output_path.c_str());
  return true;
}

bool AicpuCustKernelInfo::GetBuiltInCustJsonFile(const std::string &path) {
  AICPUE_LOGI("Begin to load builtin custom file.");

  const std::string builtin_custom_config = path + kAicpuBuiltInCustConfigFile;
  if (!RecordBuiltInCustConfigFile(builtin_custom_config)) {
    AICPUE_LOGW("don't need to merge builtin custom json files.");
    return false;
  }

  has_cust_op_ = true;

  const std::string cust_ops_file_path = builtin_custom_config + kAicpuBuiltInCustOpsFilePath;
  const std::string name = builtin_custom_config;
  AICPUE_LOGI("cust_ops_file_path=%s, name=%s.", cust_ops_file_path.c_str(), name.c_str());

  if (!ReadBuiltInCustJsonFile(name, cust_ops_file_path)) {
    AICPUE_LOGW("read merged cust json file.");
    return false;
  }

  AICPUE_LOGI("End to load builtin custom file.");
  return true;
}

bool AicpuCustKernelInfo::ReadCustJsonFile(const std::string &user_name, const std::string &cust_ops_file_path) {
  nlohmann::json cust_op_info_file;
  std::ifstream ifs(cust_ops_file_path);
  AICPU_IF_BOOL_EXEC(!ifs.is_open(), AICPUE_LOGW("Open custom op impl %s failed, do next", cust_ops_file_path.c_str());
                     return true);

  AICPU_CHECK_FALSE_EXEC(
      OpsJsonFile::Instance().ParseUnderPath(
          cust_ops_file_path, cust_op_info_file).state == ge::SUCCESS,
      AICPU_REPORT_INNER_ERR_MSG("Parse custom json file[%s] failed.",
          cust_ops_file_path.c_str());
      return true);
  AICPUE_LOGD("user_name is %s, cust_op_info_file = %s", user_name.c_str(), cust_op_info_file.dump().c_str());

  custop_info_json_file_.emplace_back(pair<string, nlohmann::json>(user_name, cust_op_info_file));

  return true;
}

bool AicpuCustKernelInfo::ReadBuiltInCustJsonFile(const std::string &user_name, const std::string &cust_ops_file_path) {
  nlohmann::json cust_op_info_file;
  std::ifstream ifs(cust_ops_file_path);
  AICPU_IF_BOOL_EXEC(!ifs.is_open(), AICPUE_LOGW("Open custom op impl %s failed, do next", cust_ops_file_path.c_str());
                     return true);

  AICPU_CHECK_FALSE_EXEC(
      OpsJsonFile::Instance().ParseUnderPath(
          cust_ops_file_path, cust_op_info_file).state == ge::SUCCESS,
      AICPU_REPORT_INNER_ERR_MSG("Parse custom json file[%s] failed.",
          cust_ops_file_path.c_str());
      return false);

  custop_info_json_file_.emplace_back(pair<string, nlohmann::json>(user_name, cust_op_info_file));
  return true;
}

bool AicpuCustKernelInfo::ReadCustOpInfoFromJsonFile(const std::string &path) {
  std::vector<string> custom_opp_path;
  std::string cust_ops_file_path;
  nlohmann::json cust_op_info_file;
  SplitLine(path, kSplitSeparator, custom_opp_path);
  AICPUE_LOGD("AicpuCustKernelInfo custom_opp_path.size() = %zu.", custom_opp_path.size());
  if (custom_opp_path.size() < 1) {
    return false;
  }
  has_cust_op_ = true;
  for (size_t i = 0; i < custom_opp_path.size(); i++) {
    cust_ops_file_path = custom_opp_path[i] + kAicpuCustOpsFilePath;
    AICPUE_LOGD("cust_ops_file_path is %s .", cust_ops_file_path.c_str());
    if (!ReadCustJsonFile(custom_opp_path[i], cust_ops_file_path)) {
      continue;
    }
  }

  AICPUE_LOGD("custop_info_json_file_ size is %zu .", custop_info_json_file_.size());
  return true;
}

FACTORY_KERNELINFO_CLASS_KEY(AicpuCustKernelInfo, kCustAicpuKernelInfoChoice)
}  // namespace aicpu
