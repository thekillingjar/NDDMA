/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/plugin/tbe_plugin_manager.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "common/plugin/ge_make_unique_util.h"
#include "graph_metadef/common/plugin/plugin_manager.h"
#include "ge/ge_api_types.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/util.h"
#include "framework/engine/dnnengine.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/opsproto_manager.h"
#include "graph/utils/type_utils_inner.h"
#include "mmpa/mmpa_api.h"
#include "register/op_registry.h"

namespace {
const size_t kMaxErrBufStrLen = 128U;
const uint32_t kMaxPointStrLen = 3U;
}  //  namespace

namespace ge {
const int32_t kBaseIntValue = 10;
constexpr const char_t* kLegacySoSuffix = "_legacy.so";

std::map<std::string, std::string> TBEPluginManager::options_ = {};

// Get Singleton Instance
TBEPluginManager &TBEPluginManager::Instance() {
  static TBEPluginManager instance_ptr_;
  return instance_ptr_;
}

Status TBEPluginManager::ClearHandles_() {
  Status ret = SUCCESS;
  for (const auto &handle : handles_vec_) {
    if (mmDlclose(handle) != 0) {
      ret = FAILED;
      const char_t *error = mmDlerror();
      GE_IF_BOOL_EXEC(error == nullptr, error = "");
      GELOGW("Failed to close handle: %s", error);
    }
  }
  handles_vec_.clear();
  return ret;
}

Status TBEPluginManager::Finalize() {
  return ClearHandles_();
}

void TBEPluginManager::ProcessSoFullName(std::vector<std::string> &file_list,
                                         std::string &caffe_parser_path,
                                         const std::string &full_name,
                                         const std::string &caffe_parser_so_suff) {
  if ((full_name.size() >= caffe_parser_so_suff.size()) &&
      (full_name.compare(full_name.size() - caffe_parser_so_suff.size(), caffe_parser_so_suff.size(),
                         caffe_parser_so_suff) == 0)) {
    caffe_parser_path = full_name;
  } else {
    // Save parser so path into file_list vector
    file_list.push_back(full_name);
  }
}

void TBEPluginManager::FindParserUsedSo(const std::string &path, std::vector<std::string> &file_list,
                                        std::string &caffe_parser_path, const uint32_t recursive_depth) {
  static const uint32_t max_recursive_depth = 20U; // For recursive depth protection

  if (recursive_depth >= max_recursive_depth) {
    GELOGW("Recursive depth is become %u, Please check input!", recursive_depth);
    return;
  }

  // Path, change to absolute path
  const std::string real_path = RealPath(path.c_str());
  // Plugin path does not exist
  if (real_path.empty()) {
    GELOGW("RealPath is empty.");
    return;
  }
  const INT32 is_dir = mmIsDir(real_path.c_str());
  // Lib plugin path does not exist
  if (is_dir != EN_OK) {
      char_t err_buf[kMaxErrBufStrLen + 1U] = {};
      const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrBufStrLen);
      GELOGW("%s is not a dir. errmsg:%s", real_path.c_str(), err_msg);
      return;
  }

  mmDirent **entries = nullptr;
  const auto ret = mmScandir(real_path.c_str(), &entries, nullptr, nullptr);
  if (ret < EN_OK) {
      char_t err_buf[kMaxErrBufStrLen + 1U] = {};
      const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrBufStrLen);
      GELOGW("scan dir failed. path = %s, ret = %d, errmsg = %s", real_path.c_str(), ret, err_msg);
      return;
  }
  for (int32_t i = 0; i < ret; ++i) {
      const mmDirent * const dent = entries[i];
      if ((strncmp(dent->d_name, ".", kMaxPointStrLen) == 0) ||
          (strncmp(dent->d_name, "..", kMaxPointStrLen) == 0)) { continue; }
      const std::string name = dent->d_name;
      const std::string full_name = real_path + "/" + name;
      const std::string so_suff = ".so";
      const std::string caffe_parser_so_suff = "lib_caffe_parser.so";
      if ((name.size() >= so_suff.size()) &&
          (name.compare(name.size() - so_suff.size(), so_suff.size(), so_suff) == 0)) {
          ProcessSoFullName(file_list, caffe_parser_path, full_name, caffe_parser_so_suff);
      } else {
          FindParserUsedSo(full_name, file_list, caffe_parser_path, recursive_depth + 1U);
      }
  }
  mmScandirFree(entries, ret);
}

void TBEPluginManager::GetOpPluginSoFileList(const std::string &path,
                                             std::vector<std::string> &file_list,
                                             std::string &caffe_parser_path) {
  // Support to split multiple so directories by ":"
  const std::vector<std::string> v_path = StringUtils::Split(path, ':');
  for (size_t i = 0U; i < v_path.size(); ++i) {
    FindParserUsedSo(v_path[i], file_list, caffe_parser_path);
    GELOGI("CustomOpLib full name = %s", v_path[i].c_str());
  }
}

void TBEPluginManager::GetCustomOpPath(std::string &customop_path) {
  GELOGI("Enter get custom op path schedule");
  std::string fmk_type;
  domi::FrameworkType type = domi::TENSORFLOW;
  const std::map<std::string, std::string>::const_iterator it = options_.find(FRAMEWORK_TYPE);
  if (it != options_.cend()) {
    type = static_cast<domi::FrameworkType>(std::strtol(it->second.c_str(), nullptr, kBaseIntValue /* base int */));
    GELOGI("Framework type is %s.", ge::TypeUtilsInner::FmkTypeToSerialString(type).c_str());
  }
  fmk_type = ge::TypeUtilsInner::FmkTypeToSerialString(type);

  const Status ret = PluginManager::GetCustomOpPath(fmk_type, customop_path);
  if (ret != SUCCESS) {
    GELOGW("Failed to get custom op path!");
  }
}

void TBEPluginManager::LoadCustomOpLib() {
  LoadPluginSo(options_);

  std::string fmk_type = std::to_string(domi::TENSORFLOW);
  const auto it = options_.find(ge::FRAMEWORK_TYPE);
  if (it != options_.end()) {
   fmk_type = it->second;
  }
  const std::vector<OpRegistrationData> registration_datas = domi::OpRegistry::Instance()->registrationDatas;
  GELOGI("The size of registration_datas is: %zu", registration_datas.size());
  for (const OpRegistrationData &reg_data : registration_datas) {
    if (std::to_string(reg_data.GetFrameworkType()) == fmk_type) {
      ge::AscendString om_op_type;
      (void) reg_data.GetOmOptype(om_op_type);
      GELOGD("Begin to register optype: %s, imply_type: %s", om_op_type.GetString(),
             TypeUtilsInner::ImplyTypeToSerialString(reg_data.GetImplyType()).c_str());
      (void)domi::OpRegistry::Instance()->Register(reg_data);
    }
  }
}

void TBEPluginManager::LoadPluginSo(const std::map<std::string, std::string> &options) {
  std::vector<std::string> file_list;
  std::string caffe_parser_path;
  std::string plugin_path;

  options_ = options;
  GetCustomOpPath(plugin_path);

  // Whether there are files in the plugin so path
  GetOpPluginSoFileList(plugin_path, file_list, caffe_parser_path);
  // sort, move "_legacy.so" to the end
  std::stable_partition(file_list.begin(), file_list.end(), [](const std::string &file_path) {
    return file_path.size() < std::strlen(kLegacySoSuffix) ||
           file_path.compare((file_path.size() - std::strlen(kLegacySoSuffix)), std::strlen(kLegacySoSuffix), kLegacySoSuffix) != 0;
  });

  //  No file
  if (file_list.empty()) {
    // Print log
    GELOGW("Can not find any plugin file in plugin_path: %s", plugin_path.c_str());
  }

  GELOGW("The shared library will not be checked. Please ensure that the source of the shared library is trusted.");

  // Load other so files except lib_caffe_parser.so in the plugin so path
  for (auto elem : file_list) {
    (void) StringUtils::Trim(elem);

    void * const handle = mmDlopen(elem.c_str(), static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW) |
                                                                      static_cast<uint32_t>(MMPA_RTLD_GLOBAL) |
                                                                      static_cast<uint32_t>(MMPA_RTLD_NODELETE)));
    if (handle == nullptr) {
      const char_t *error = mmDlerror();
      GE_IF_BOOL_EXEC(error == nullptr, error = "");
      GELOGW("dlopen failed, plugin name:%s. Message(%s).", elem.c_str(), error);
    } else if (find(handles_vec_.begin(), handles_vec_.end(), handle) == handles_vec_.end()) {
      // Close dl when the program exist, not close here
      GELOGI("Plugin load %s success.", elem.c_str());
      handles_vec_.push_back(handle);
    } else {
      GELOGI("Plugin so has already been loaded, no need to load again.");
    }
  }
}

void TBEPluginManager::InitPreparation(const std::map<std::string, std::string> &options) {
  options_.insert(options.begin(), options.end());
  // Load TBE plugin
  TBEPluginManager::Instance().LoadCustomOpLib();
}
}  // namespace ge
