/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_store/sub_op_info_store.h"
#include <dirent.h>
#include <fstream>
#include <algorithm>
#include "common/fe_log.h"
#include "common/aicore_util_types.h"
#include "common/string_utils.h"
#include "common/fe_type_utils.h"
#include "common/configuration.h"
#include "ops_store/ops_kernel_utils.h"
#include "ops_store/ops_kernel_error_codes.h"
#include "ops_store/ops_kernel_constants.h"

namespace fe {
SubOpInfoStore::SubOpInfoStore(const FEOpsStoreInfo &ops_store_info)
    : init_flag_(false), ops_store_info_(ops_store_info) {}
SubOpInfoStore::~SubOpInfoStore() {}

Status SubOpInfoStore::Initialize(const std::string &engine_name) {
  if (init_flag_) {
    return SUCCESS;
  }
  init_flag_ = true;

  if (ops_store_info_.fe_ops_store_name.empty()) {
    FE_LOGW("The name of the operations information library is empty.");
    return OP_STORE_CFG_NAME_EMPTY;
  }
  if (ops_store_info_.cfg_file_path.empty()) {
    FE_LOGW("The configuration file path for the op information library [%s] is empty.", ops_store_info_.fe_ops_store_name.c_str());
    return OP_STORE_CFG_FILE_EMPTY;
  }

  std::string cfg_real_path = GetRealPath(ops_store_info_.cfg_file_path);
  if (cfg_real_path.empty()) {
    FE_LOGW("The configuration file [%s] for the operational information library [%s] does not exist.",
            ops_store_info_.cfg_file_path.c_str(), ops_store_info_.fe_ops_store_name.c_str());
    return OP_STORE_CFG_FILE_NOT_EXIST;
  }

  FE_LOGD("Initialize %s SubOpsStore.", ops_store_info_.fe_ops_store_name.c_str());
  Status ret_value = LoadOpInfo(cfg_real_path);
  if (ret_value != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Init] Failed to load ops store %s. Config file path is %s. Return value is %u.",
                    ops_store_info_.fe_ops_store_name.c_str(), ops_store_info_.cfg_file_path.c_str(), ret_value);
    return OP_STORE_READ_CFG_FILE_FAILED;
  }

  ret_value = ConstructOpKernelInfo(engine_name);
  if (ret_value != SUCCESS) {
    return ret_value;
  }
  FE_LOGI("Size of sub ops store %s is %zu.",
          ops_store_info_.fe_ops_store_name.c_str(), op_kernel_info_map_.size());
  FE_LOGI("Initialize %s SubOpsStore finished.", ops_store_info_.fe_ops_store_name.c_str());
  return SUCCESS;
}

void SubOpInfoStore::SplitOpStoreJson(const std::string& json_file_path, std::string& ops_path_name_prefix) {
  size_t last_under_score = json_file_path.find_last_of('-');
  if (last_under_score == std::string::npos) {
    return;
  }
  size_t dot_json_pos = json_file_path.find(".json", last_under_score);
  if (dot_json_pos == std::string::npos) {
    return;
  }
  ops_path_name_prefix = json_file_path.substr(last_under_score + 1, dot_json_pos - last_under_score - 1);
  if (ops_path_name_prefix == "info") {
    ops_path_name_prefix = "";
  } else {
    ops_path_name_prefix = "ops_" + ops_path_name_prefix;
  }
  FE_LOGD("ops_path_name_prefix is [%s]", ops_path_name_prefix.c_str());
}

Status SubOpInfoStore::LoadOpInfo(const string &real_path) {
  vector<string> ops_cfg_files;
  DIR *dir;
  struct dirent *dirp = nullptr;
  char *file_suffix = const_cast<char *>(".json");
  dir = opendir(real_path.c_str());
  FE_CHECK(dir == nullptr, FE_LOGE("Failed to open directory %s.", real_path.c_str()), return FAILED);
  while ((dirp = readdir(dir)) != nullptr) {
    if (dirp->d_name[0] == '.') {
      continue;
    }
    if (strlen(dirp->d_name) <= strlen(file_suffix)) {
      continue;
    }
    if (strcmp(&(dirp->d_name)[strlen(dirp->d_name) - strlen(file_suffix)], file_suffix) == 0) {
      ops_cfg_files.push_back(real_path + "/" + dirp->d_name);
    }
  }
  closedir(dir);

  if (ops_cfg_files.empty()) {
    FE_LOGI("No JSON file found at path: %s.", real_path.c_str());
    return SUCCESS;
  }

  auto CompareByLength = [](const std::string &a, const std::string &b) {
    // 先读全量，再读分包
    if (a.find("legacy") != std::string::npos) return true; 
    if (b.find("legacy") != std::string::npos) return false; 
    return a.length() < b.length();
  };

  if (ops_cfg_files.size() > 1) {
    FE_LOGD("Begin to sort the files of op info json.");
    std::sort(ops_cfg_files.begin(), ops_cfg_files.end(), CompareByLength);
  }
  for (std::string json_file_path : ops_cfg_files) {
    if (LoadOpJsonFile(json_file_path) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][Init][LoadOpInfo] Failed to load JSON file [%s].", json_file_path.c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status SubOpInfoStore::LoadOpJsonFile(const std::string &json_file_path) {
  vector<string> op_type_vec;
  nlohmann::json op_json_file;
  FE_LOGD("Begin to load json file[%s].", json_file_path.c_str());
  if (ReadJsonObject(json_file_path, op_json_file) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Init][LoadOpJsonFile] Failed to read JSON object in %s.", json_file_path.c_str());
    return FAILED;
  }
  std::string ops_path_name_prefix;
  SplitOpStoreJson(json_file_path, ops_path_name_prefix);
  try {
    if (!op_json_file.is_object()) {
      REPORT_FE_ERROR("[GraphOpt][Init][LoadOpJsonFile] The top level of the JSON file should be an object, but it is actually %s.",
                      GetJsonObjectType(op_json_file).c_str());
      return OP_SUB_STORE_ILLEGAL_JSON;
    }
    for (auto &elem : op_json_file.items()) {
      string op_type_temp = elem.key();
      op_type_vec.push_back(StringUtils::Trim(op_type_temp));
    }
    for (auto &op_type : op_type_vec) {
      OpContent op_content;
      op_content.op_type_ = op_type;
      op_content.ops_path_name_prefix_ = ops_path_name_prefix;
      if (!op_json_file[op_type].is_object()) {
        REPORT_FE_ERROR("[GraphOpt][Init][LoadOpJsonFile] The second level of the JSON file should be an object, but it is actually %s.",
                        GetJsonObjectType(op_json_file[op_type]).c_str());
        return OP_SUB_STORE_ILLEGAL_JSON;
      }
      for (auto &elem_out : op_json_file[op_type].items()) {
        map <string, string> map_temp;
        if (!op_json_file[op_type][elem_out.key()].is_object()) {
          REPORT_FE_ERROR("[GraphOpt][Init][LoadOpJsonFile] The third level of the JSON file should be an object, but it is actually %s.",
                          GetJsonObjectType(op_json_file[op_type][elem_out.key()]).c_str());
          return OP_SUB_STORE_ILLEGAL_JSON;
        }
        for (auto &elem_inner : op_json_file[op_type][elem_out.key()].items()) {
          string key_inner = elem_inner.key();
          string value_inner = elem_inner.value();
          map_temp.emplace(std::make_pair(StringUtils::Trim(key_inner), StringUtils::Trim(value_inner)));
        }
        string key_out = elem_out.key();
        op_content.map_kernel_info_.emplace(std::make_pair(StringUtils::Trim(key_out), map_temp));
      }
      op_content_map_[op_type] = op_content;
    }
  } catch (const std::exception &e) {
    REPORT_FE_ERROR("[GraphOpt][Init][LoadOpJsonFile] Failed to conv file[%s] to Json. Error message is %s",
                    json_file_path.c_str(), e.what());
    return OP_SUB_STORE_ILLEGAL_JSON;
  }
  return SUCCESS;
}

Status SubOpInfoStore::ConstructOpKernelInfo(const std::string &engine_name) {
  OpKernelInfoConstructor op_kernel_info_constructor;
  for (auto &op : op_content_map_) {
    OpKernelInfoPtr op_kernel_info_ptr = nullptr;
    FE_MAKE_SHARED(op_kernel_info_ptr = std::make_shared<OpKernelInfo>(op.first, ops_store_info_.op_impl_type),
                   return FAILED);
    FE_CHECK_NOTNULL(op_kernel_info_ptr);
    /* Here shared ptr and map/vector will destruct automatically, so not necessary
       to run final for those op_kernel_info which fails to initialize. */
    if (op_kernel_info_constructor.InitializeOpKernelInfo(engine_name, op.second, op_kernel_info_ptr) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][Init][ConstructOpKernelInfo] OpKernelInfo %s initialization failed.", op.first.c_str());
      return FAILED;
    }

    op_kernel_info_map_.emplace(std::make_pair(op.first, op_kernel_info_ptr));
  }
  return SUCCESS;
}

Status SubOpInfoStore::Finalize() {
  if (!init_flag_) {
    return SUCCESS;
  }
  OpKernelInfoConstructor op_kernel_info_constructor;
  for (auto &el : this->op_kernel_info_map_) {
    if (el.second == nullptr) {
      FE_LOGW("OpKernelInfo pointer[%s] in op_kernel_info_map_ should not be nullptr.",
              el.first.c_str());
      continue;
    }
    op_kernel_info_constructor.FinalizeOpKernelInfo(el.second);
  }
  this->op_kernel_info_map_.clear();
  init_flag_ = false;
  return SUCCESS;
}

const std::map<std::string, OpKernelInfoPtr>& SubOpInfoStore::GetAllOpKernels() {
  return this->op_kernel_info_map_;
}

OpKernelInfoPtr SubOpInfoStore::GetOpKernelByOpType(const std::string &op_type) {
  std::map<std::string, OpKernelInfoPtr>::const_iterator iter = op_kernel_info_map_.find(op_type);
  FE_LOGD("Size of all kernels is %zu bytes.", op_kernel_info_map_.size());
  if (iter == op_kernel_info_map_.end()) {
    FE_LOGD("Operation type [%s] does not exist in the operation information library [%s].",
            op_type.c_str(), this->ops_store_info_.fe_ops_store_name.c_str());
    return nullptr;
  }
  return iter->second;
}

Status SubOpInfoStore::GetOpContentByOpType(const std::string &op_type, OpContent &op_content) const {
  auto iter = op_content_map_.find(op_type);
  if (iter == op_content_map_.end()) {
    FE_LOGD("Op Type[%s] is not found in op information library[%s].",
            op_type.c_str(), this->ops_store_info_.fe_ops_store_name.c_str());
    return FAILED;
  }
  op_content = iter->second;
  return SUCCESS;
}

Status SubOpInfoStore::SetOpContent(const OpContent &op_content) {
  if (op_content.op_type_.empty()) {
    REPORT_FE_ERROR("[GraphOpt][SetDynamicInfo][SetOpContent] The op type of op_content is empty.");
    return FAILED;
  }
  op_content_map_[op_content.op_type_] = op_content;
  return SUCCESS;
}

const std::string& SubOpInfoStore::GetOpsStoreName() const {
  return this->ops_store_info_.fe_ops_store_name;
}

const OpImplType& SubOpInfoStore::GetOpImplType() const {
  return this->ops_store_info_.op_impl_type;
}

void SubOpInfoStore::UpdateStrToOpContent(OpContent &op_content, const std::string key1, const std::string key2,
                                          const std::string &value) const {
  const auto &key1_pos = op_content.map_kernel_info_.find(key1);
  if (key1_pos == op_content.map_kernel_info_.end()) {
    if (value.empty()) {
      return;
    }
    std::map<std::string, std::string> map_temp;
    map_temp.emplace(std::make_pair(key2, value));
    op_content.map_kernel_info_.emplace(std::make_pair(key1, map_temp));
    return;
  }

  const auto &key2_pos = key1_pos->second.find(key2);
  if (key2_pos == key1_pos->second.end()) {
    if (value.empty()) {
      return;
    }
    key1_pos->second.emplace(std::make_pair(key2, value));
    return;
  }
  key2_pos->second = value;
}
}  // namespace fe
