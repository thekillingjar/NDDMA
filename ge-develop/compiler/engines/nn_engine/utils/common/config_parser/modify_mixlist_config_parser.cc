/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/config_parser/modify_mixlist_config_parser.h"
#include <fstream>
#include "common/fe_log.h"
#include "common/fe_type_utils.h"
#include "common/fe_report_error.h"
#include "ge/ge_api_types.h"
#include "graph/ge_context.h"

namespace fe {
namespace {
constexpr char const *kListTypeGrayList = "gray-list";
constexpr char const *kListTypeWhiteList = "white-list";
constexpr char const *kListTypeBlackList = "black-list";
constexpr char const *kUpdateTypeToAdd = "to-add";
constexpr char const *kUpdateTypeToRemove = "to-remove";
const std::set<std::string> kPrecisionReduceListType = {kListTypeGrayList, kListTypeWhiteList, kListTypeBlackList};
const std::set<std::string> kPrecisionReduceUpdateType = {kUpdateTypeToAdd, kUpdateTypeToRemove};

const std::uint8_t kPrecisionReduceToRemoveShift = 1;
const std::uint8_t kPrecisionReduceBlackShift = 4;
const std::uint8_t kPrecisionReduceWhiteShift = 2;

const std::uint8_t kPrecisionReduceToRemoveBlackList = 1 << (kPrecisionReduceBlackShift +
                                                             kPrecisionReduceToRemoveShift); // 0b00100000
const std::uint8_t kPrecisionReduceToAddBlackList    = 1 << kPrecisionReduceBlackShift;      // 0b00010000
const std::uint8_t kPrecisionReduceToRemoveWhiteList = 1 << (kPrecisionReduceWhiteShift +
                                                             kPrecisionReduceToRemoveShift); // 0b00001000;
const std::uint8_t kPrecisionReduceToAddWhiteList    = 1 << kPrecisionReduceWhiteShift;      // 0b00000100;
const std::uint8_t kPrecisionReduceToRemoveGrayList  = 1 << kPrecisionReduceToRemoveShift;   // 0b00000010;
const std::uint8_t kPrecisionReduceToAddGrayList     = 1;                                    // 0b00000001;
const std::map<uint8_t, std::string> kOperListStrMap = {
    {kPrecisionReduceToRemoveBlackList, "remove black list"},
    {kPrecisionReduceToAddBlackList, "add black list"},
    {kPrecisionReduceToRemoveWhiteList, "remove white list"},
    {kPrecisionReduceToAddWhiteList, "add white list"},
    {kPrecisionReduceToRemoveGrayList, "remove gray list"},
    {kPrecisionReduceToAddGrayList, "add gray list"}
};

const std::set<std::uint8_t> kPrecisionReduceUpdateTemplate = {
        kPrecisionReduceToRemoveBlackList,
        kPrecisionReduceToRemoveWhiteList,
        kPrecisionReduceToAddGrayList,
        kPrecisionReduceToAddGrayList | kPrecisionReduceToRemoveBlackList,
        kPrecisionReduceToAddGrayList | kPrecisionReduceToRemoveWhiteList,

        kPrecisionReduceToAddWhiteList,
        kPrecisionReduceToAddWhiteList | kPrecisionReduceToRemoveGrayList,
        kPrecisionReduceToAddWhiteList | kPrecisionReduceToRemoveBlackList,

        kPrecisionReduceToAddBlackList,
        kPrecisionReduceToAddBlackList | kPrecisionReduceToRemoveGrayList,
        kPrecisionReduceToAddBlackList | kPrecisionReduceToRemoveWhiteList,
};

bool IsFileEmpty(const std::string &file_name) {
  std::ifstream ifs(file_name);
  if (!ifs.is_open()) {
    REPORT_FE_ERROR("[GraphOpt][Init][LoadJson] File [%s] doesn't exist or has been opened.", file_name.c_str());
    return false;
  }
  char ch;
  ifs >> ch;
  if (ifs.eof()) {
    ifs.close();
    return true;
  }
  ifs.close();
  return false;
}
}

ModifyMixlistConfigParser::ModifyMixlistConfigParser() {}
ModifyMixlistConfigParser::~ModifyMixlistConfigParser() {}

Status ModifyMixlistConfigParser::InitializeFromOptions(const std::map<std::string, std::string> &options) {
  const std::map<string, string>::const_iterator iter = options.find(ge::MODIFY_MIXLIST);
  if (iter == options.end() || iter->second.empty()) {
    FE_LOGD("The value of param[modify_mixlist] is not configured in ge.option.");
    return SUCCESS;
  }
  FE_LOGD("The value of param[modify_mixlist] is [%s].", iter->second.c_str());
  if (modify_mixlist_path_ == iter->second) {
    FE_LOGD("The modified mixlist file path is the same as the path from the last time.");
    return SUCCESS;
  }
  modify_mixlist_path_ = iter->second;
  return InitializeModifyMixlist();
}

Status ModifyMixlistConfigParser::InitializeFromContext(const std::string &combined_params_key) {
  if (combined_params_key.empty()) {
    FE_LOGD("The modified mixlist file path is empty.");
    return SUCCESS;
  }
  FE_LOGD("The value of param[modify_mixlist] is [%s].", combined_params_key.c_str());
  if (modify_mixlist_path_ == combined_params_key) {
    FE_LOGD("The modified mixlist file path is the same as the path from the last time.");
    return SUCCESS;
  }
  modify_mixlist_path_ = combined_params_key;
  return InitializeModifyMixlist();
}

Status ModifyMixlistConfigParser::InitializeModifyMixlist() {
  string modify_mixlist_path = GetRealPath(modify_mixlist_path_);
  if (modify_mixlist_path.empty()) {
    ErrorMessageDetail err_msg(EM_OPEN_FILE_FAILED, {modify_mixlist_path_});
    ReportErrorMessage(err_msg);
    return FAILED;
  }
  if (IsFileEmpty(modify_mixlist_path)) {
    REPORT_FE_ERROR("[GraphOpt][InitMixList] The file at path [%s] is empty.", modify_mixlist_path.c_str());
    return FAILED;
  }
  mixlist_map_.clear();
  Status status = LoadModifyMixlistJson(modify_mixlist_path);
  if (status != SUCCESS) {
    ErrorMessageDetail err_msg(EM_READ_FILE_FAILED,
        {modify_mixlist_path, "The configuration file is not in JSON format or its content is invalid"});
    ReportErrorMessage(err_msg);
    FE_LOGE("[GraphOpt][InitMixList] Failed to load JSON from file: %s.", modify_mixlist_path.c_str());
    return status;
  }
  return VerifyMixlist();
}

Status ModifyMixlistConfigParser::LoadModifyMixlistJson(const std::string &modify_mixlist_path) {
  nlohmann::json op_json_file;
  if (ReadMixlistJson(modify_mixlist_path, op_json_file) != SUCCESS) {
    FE_LOGE("[GraphOpt][Init][LoadJson] Failed to read modified mixlist json in [%s].", modify_mixlist_path.c_str());
    return FAILED;
  }

  try {
    if (!op_json_file.is_object()) {
      FE_LOGE("[GraphOpt][Init][LoadJson] The top level of the mixlist JSON is not a JSON object.");
      return FAILED;
    }

    for (auto &elem : op_json_file.items()) {
      auto list_type = elem.key();
      if (kPrecisionReduceListType.find(list_type) == kPrecisionReduceListType.end()) {
        FE_LOGE("[GraphOpt][Init][LoadJson] Top level of json [%s] should be 'gray-list' or 'white-list' or "
                "'black-list'.", list_type.c_str());
        return FAILED;
      }

      if (!op_json_file[list_type].is_object()) {
        FE_LOGE("[GraphOpt][Init][LoadJson] The second level of the JSON is not a JSON object.");
        return FAILED;
      }
      for (auto &precision_reduce_update_type : op_json_file[list_type].items()) {
        auto update_type = precision_reduce_update_type.key();
        if (kPrecisionReduceUpdateType.find(update_type) == kPrecisionReduceUpdateType.end()) {
          FE_LOGE("[GraphOpt][Init][LoadJson] The second level of the JSON [%s] should be either to-add or to-remove.",
                  update_type.c_str());
          return FAILED;
        }

        if (!op_json_file[list_type][update_type].is_array()) {
          FE_LOGE("[GraphOpt][Init][LoadJson] The third level of the JSON should be an array.");
          return FAILED;
        }

        AddMixList(op_json_file, list_type, update_type);
      }
    }
  } catch (const std::exception &e) {
    FE_LOGE("[GraphOpt][Init][LoadJson] Failed to convert file [%s] to JSON, the error message is: [%s].",
            modify_mixlist_path.c_str(), e.what());
    return FAILED;
  }
  return SUCCESS;
}

Status ModifyMixlistConfigParser::ReadMixlistJson(const std::string &file_path, nlohmann::json &json_obj) const {
  std::ifstream ifs(file_path);
  try {
    if (!ifs.is_open()) {
      FE_LOGW("Opening %s failed: file is already open.", file_path.c_str());
      ErrorMessageDetail err_msg(EM_OPEN_FILE_FAILED, {file_path});
      ReportErrorMessage(err_msg);
      return FAILED;
    }
    ifs >> json_obj;
    ifs.close();
  } catch (const std::exception &er) {
    FE_LOGW("Failed to convert file [%s] to JSON. Current message is: %s", file_path.c_str(), er.what());
    ifs.close();
    ErrorMessageDetail err_msg(EM_READ_FILE_FAILED, {file_path, er.what()});
    ReportErrorMessage(err_msg);
    return FAILED;
  }

  return SUCCESS;
}

void ModifyMixlistConfigParser::AddMixList(const nlohmann::json &op_json_file,
                                           const std::string &list_type, const std::string &update_type) {
  for (auto &op_type : op_json_file[list_type][update_type]) {
    uint8_t bitmap = 0;
    if (update_type == kUpdateTypeToRemove) {
      bitmap = kPrecisionReduceToRemoveGrayList;
    } else {
      bitmap = kPrecisionReduceToAddGrayList;
    }

    if (list_type == kListTypeBlackList) {
      bitmap <<= kPrecisionReduceBlackShift;
    } else if (list_type == kListTypeWhiteList) {
      bitmap <<= kPrecisionReduceWhiteShift;
    }

    auto iter = mixlist_map_.find(op_type);
    if (iter == mixlist_map_.end()) {
      mixlist_map_.emplace(std::make_pair(op_type, bitmap));
      continue;
    }
    iter->second |= bitmap;
  }
}

Status ModifyMixlistConfigParser::VerifyMixlist() const {
  if (mixlist_map_.empty()) {
    return SUCCESS;
  }
  for (const std::pair<const string, uint8_t> &item : mixlist_map_) {
    if (kPrecisionReduceUpdateTemplate.find(item.second) == kPrecisionReduceUpdateTemplate.end()) {
      std::string mix_list_desc = GetMixlistDesc(item.second);
      ErrorMessageDetail err_msg(EM_INVALID_CONTENT,
                                 {ge::MODIFY_MIXLIST, modify_mixlist_path_,
                                  "Operation " + mix_list_desc + " for op " + item.first + " is invalid"});
      ReportErrorMessage(err_msg);
      FE_LOGE("The mix list value[%s] of op type[%s] is invalid.", mix_list_desc.c_str(), item.first.c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}
std::string ModifyMixlistConfigParser::GetMixlistDesc(const uint8_t &op_mix_list) {
  std::string mix_list_str;
  bool is_first = true;
  for (const std::pair<uint8_t, std::string> item : kOperListStrMap) {
    if (static_cast<bool>(op_mix_list & item.first)) {
      if (is_first) {
        is_first = false;
      } else {
        mix_list_str += ", ";
      }
      mix_list_str += item.second;
    }
  }
  return mix_list_str;
}
PrecisionPolicy ModifyMixlistConfigParser::GetPrecisionPolicy(const std::string &op_type,
                                                              const PrecisionPolicy &op_kernel_policy) const {
  const std::map<std::string, std::uint8_t>::const_iterator iter = mixlist_map_.find(op_type);
  if (iter == mixlist_map_.cend()) {
    FE_LOGD("The precision policy for operation type [%s] was not found.", op_type.c_str());
    return op_kernel_policy;
  }
  PrecisionPolicy ret_value = op_kernel_policy;
  uint8_t mix_list_val = iter->second;
  if ((((mix_list_val & kPrecisionReduceToRemoveBlackList) != 0) && op_kernel_policy != BLACK) ||
      (((mix_list_val & kPrecisionReduceToRemoveWhiteList) != 0) && op_kernel_policy != WHITE) ||
      (((mix_list_val & kPrecisionReduceToRemoveGrayList) != 0) && op_kernel_policy != GRAY)) {
    FE_LOGW("The remove list for op_type [%s] does not match the precision_reduce [%s], and the mapping is [0x%x].",
            op_type.c_str(), GetPrecisionPolicyString(op_kernel_policy).c_str(), mix_list_val);
  }
  if ((((mix_list_val & kPrecisionReduceToRemoveBlackList) != 0) && op_kernel_policy == BLACK) ||
      (((mix_list_val & kPrecisionReduceToRemoveWhiteList) != 0) && op_kernel_policy == WHITE) ||
      (((mix_list_val & kPrecisionReduceToRemoveGrayList) != 0) && op_kernel_policy == GRAY)) {
    FE_LOGD("The precision policy of op kernel[%s] has been removed. Precision policy will be reset to gray list.",
            op_type.c_str());
    ret_value = GRAY;
  }

  if ((mix_list_val & kPrecisionReduceToAddBlackList) != 0) {
    ret_value = BLACK;
  } else if ((mix_list_val & kPrecisionReduceToAddWhiteList) != 0) {
    ret_value = WHITE;
  }
  return ret_value;
}
}
