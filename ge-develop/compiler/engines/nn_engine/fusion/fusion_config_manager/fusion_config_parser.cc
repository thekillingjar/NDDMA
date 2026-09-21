/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fusion_config_manager/fusion_config_parser.h"
#include <sstream>
#include <unordered_set>
#include "common/configuration.h"
#include "common/fe_context_utils.h"
#include "common/fe_error_code.h"
#include "common/string_utils.h"
#include "common/fusion_pass_util.h"
#include "common/fe_report_error.h"
#include "base/err_msg.h"
#include "common/util/json_util.h"
#include "common/fe_type_utils.h"
#include "common/platform_utils.h"
#include "fusion_rule_manager/fusion_rule_parser/fusion_rule_parser_utils.h"
#include "graph/ge_local_context.h"
#include "register/pass_option_utils.h"
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "register/graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

namespace fe {
namespace {
  const std::string kPriorityType = "Priority";
  const std::string kBuildinType = "build-in";
  const std::string kCustomType = "custom";
  const std::string kPriorityTop = "Top";
  const std::string kPriorityMain = "Main";
  const std::string kPriorityDown = "Down";
  const std::string kSwitchType = "Switch";
  const std::string kExceptionalPass = "ExceptionalPassOfO1Level";
  const std::string kSwitchOn = "on";
  const std::string kSwitchOff = "off";
  const std::string kPassSwitchAll = "ALL";
  const std::string kStatgeChkLstLyrFmt = "[GraphOpt][FusionConfig][ChkLstLyrFmt]";
  const std::string kStatgeJdgPriority = "[GraphOpt][FusionConfig][JdgPriority]";
  const std::string kStatgeChkFusCfgJsLyrFmt = "[GraphOpt][FusionConfig][ChkFusCfgJsLyrFmt]";
  const std::vector<std::string> kFusionTypeVec = {GRAPH_FUSION, UB_FUSION};
}

const int32_t CUSTOM_CFG_TOP_PRIORITY_MIN = 0;
const int32_t CUSTOM_CFG_MAIN_PRIORITY_MIN = 1000;
const int32_t CUSTOM_CFG_DOWN_PRIORITY_MIN = 2000;
const int32_t BUILT_IN_CFG_TOP_PRIORITY_MIN = 3000;
const int32_t BUILT_IN_CFG_MAIN_PRIORITY_MIN = 4000;
const int32_t BUILT_IN_CFG_DOWN_PRIORITY_MIN = 5000;
const int32_t CUSTOM_PASS_PRIORITY_MIN = 6000;
const int32_t CUSTOM_RULE_PRIORITY_MIN = 7000;
const int32_t BUILT_IN_PASS_PRIORITY_MIN = 8000;
const int32_t BUILT_IN_RULE_PRIORITY_MIN = 9000;
const int32_t RESERVED_FOR_DOWN_PRIORITY = 10000;
const uint64_t kBit0Mask = 0x1UL;

FusionConfigParser::FusionConfigParser(std::string engine_name)
  : is_first_init_(true), has_support_fusion_pass_(false), engine_name_(std::move(engine_name)) {}

FusionConfigParser::~FusionConfigParser() = default;

bool FusionConfigParser::IsPassAllowedByLicense(const std::string &pass_name) const {
  bool is_under_control = Configuration::Instance(engine_name_).IsInLicenseControlMap(pass_name);
  FE_LOGD("Whether this pass[%s] is in license control list is [%d].", pass_name.c_str(), is_under_control);
  if (!is_under_control) {
    FE_LOGD("This pass[%s] is allowed for this pass is not in license control list.", pass_name.c_str());
    return true;
  }

  const std::string &license_fusion_cache = Configuration::Instance(engine_name_).GetLicenseFusionStr();
  FE_LOGD("Pass:[%s] license_fusion_cache %s", pass_name.c_str(), license_fusion_cache.c_str());
  if (license_fusion_cache == kPassSwitchAll) {
    FE_LOGD("This pass[%s] is allowed for fusion license cache is ALL.", pass_name.c_str());
    return true;
  }

  const std::set<std::string> &license_detail_info = Configuration::Instance(engine_name_).GetLicenseFusionDetailInfo();
  if (license_detail_info.find(pass_name) != license_detail_info.cend()) {
    FE_LOGD("This pass[%s] is allowed for it is in the license detail list.", pass_name.c_str());
    return true;
  } else {
    FE_LOGD("This pass[%s] is not allowed for it is not in the license detail list.", pass_name.c_str());
    return false;
  }
}

bool FusionConfigParser::GetPassSwitchByExceptionalCfg(const std::string &pass_name) const {
  // 1. check fusion switch map by pass type
  const auto iter = exceptional_pass_.find(pass_name);
  if (iter != exceptional_pass_.cend()) {
    FE_LOGD("The pass[%s] on the exceptional list.", pass_name.c_str());
    return true;
  }

  return false;
}

bool FusionConfigParser::GetPassSwitchByPassName(const std::string &pass_name, const std::string &pass_type,
                                                 bool &pass_switch) const {
  // 1. check fusion switch map by pass type
  const FusionSwitchMap::const_iterator iter = fusion_switch_map_.find(pass_type);
  if (iter == fusion_switch_map_.cend()) {
    return false;
  }

  // 2. check whether is there config item for current pass
  const std::map<std::string, bool>::const_iterator pass_iter = iter->second.find(pass_name);
  if (pass_iter != iter->second.end()) {
    pass_switch = pass_iter->second;
    return true;
  }

  // 3. check whether is there config item for all pass
  auto all_iter = iter->second.find(kPassSwitchAll);
  if (all_iter != iter->second.end()) {
    pass_switch = all_iter->second;
    return true;
  }
  FE_LOGD("The switch of pass[%s] is not configured.", pass_name.c_str());
  return false;
}

bool FusionConfigParser::IsForbiddenClosedPass(const std::string &pass_name, const std::string &pass_type,
                                               const uint64_t &pass_attr) const {
  // 1. FE built-in forbidden closed pass
  if (ForbiddenClosedPass.find(pass_name) != ForbiddenClosedPass.end()) {
    FE_LOGD("Pass:[%s] is forbidden closed, will be on by default.", pass_name.c_str());
    return true;
  }

  // 2. Forbidden close pass by ops resigter
  if (IsPassAttrTypeOn(pass_attr, PassAttrType::FRBDN_CLOSE)) {
    FE_LOGD("Pass:[%s] of [%s : %lu] is set as forbidden closed, will be on by default.",
            pass_name.c_str(), pass_type.c_str(), pass_attr);
    return true;
  }
  return false;
}

bool FusionConfigParser::CheckPassIsWhiteListControl(const std::string &pass_name,
                                                     const GraphFusionPassType &pass_type) const {
  std::string config_name = "fusion config or fusion switch file";
  if (pass_type == CUSTOM_AI_CORE_GRAPH_PASS ||
      pass_type == CUSTOM_VECTOR_CORE_GRAPH_PASS || !has_support_fusion_pass_) {
    FE_LOGD("Pass[%s] is custom_pass or cur_soc don't have white list, can be configured on/off by %s.",
            pass_name.c_str(), config_name.c_str());
    return false;
  }
  if (support_fusion_pass_vec_.empty()) {
    return true;
  }
  auto it = std::find(support_fusion_pass_vec_.begin(), support_fusion_pass_vec_.end(), pass_name);
  if (it != support_fusion_pass_vec_.end()) {
    FE_LOGD("Pass:[%s] is in white list, can be configured on/off by %s.", pass_name.c_str(), config_name.c_str());
    return false;
  } else {
    FE_LOGD("Pass:[%s] isn't in white list, force close.", pass_name.c_str());
    return true;
  }
}

Status FusionConfigParser::GetFusionSwitchByOptimizationOption(const std::string &pass_name, bool &enable_flag,
                                                              const std::string &pass_type,
                                                              const uint64_t pass_attr) {
  if (ge::PassOptionUtils::CheckIsPassEnabledByOption(pass_name, enable_flag) == ge::GRAPH_SUCCESS) {
    FE_LOGI("Pass[%s] switch [%d] is configured by optimization switch.", pass_name.c_str(), enable_flag);
    return SUCCESS;
  }
  // white list (temporary plan)
  if (GetPassSwitchByExceptionalCfg(pass_name)) {
    FE_LOGI("This pass[%s] is configured exceptional of O1 level pass.", pass_name.c_str());
    enable_flag = true;
    return SUCCESS;
  }

  if (ge::PassOptionUtils::CheckIsPassEnabledByOption(kForbiddenPass, enable_flag) == ge::GRAPH_SUCCESS) {
    if(IsForbiddenClosedPass(pass_name, pass_type, pass_attr)) {
        FE_LOGI("Pass:[%s] is forbidden closed pass, pass switch [%d] is configured by optimization_switch option.",
               pass_name.c_str(), enable_flag);
        return SUCCESS;
    }
  }
  // O1 close all pass
  std::string opt_val;
  (void)ge::GetThreadLocalContext().GetOo().GetValue(kComLevelO1Opt, opt_val);
  if (opt_val == kStrFalse) {
    FE_LOGI("The fusion_pass[%s] is not allowed in current O1 level.", pass_name.c_str());
    enable_flag = false;
    return SUCCESS;
  }
  return FAILED;
}

bool FusionConfigParser::GetFusionSwitchByName(const std::string &pass_name, const std::string &pass_type,
                                               const uint64_t pass_attr, const bool can_close_fusion) {
  bool enable_flag = true;
  if (GetFusionSwitchByOptimizationOption(pass_name, enable_flag, pass_type, pass_attr) == SUCCESS) {
    return enable_flag;
  }
  // 1. check whether this pass is forbidden close
  if (IsForbiddenClosedPass(pass_name, pass_type, pass_attr)) {
    FE_LOGD("Pass:[%s] is forbidden closed, will be on by default.", pass_name.c_str());
    return true;
  }

  // 2. check license
  if (!IsPassAllowedByLicense(pass_name)) {
    FE_LOGD("This pass[%s] is not allowed by fusion license.", pass_name.c_str());
    return false;
  }

  // 3. check fusion switch and fusion config
  bool pass_switch = false;
  if (GetPassSwitchByPassName(pass_name, pass_type, pass_switch)) {
    FE_LOGD("This pass[%s] is configured by fusion config or fusion switch file, pass switch is [%d].",
            pass_name.c_str(), pass_switch);
    return pass_switch;
  }

  // 4. if this is single op scene, check pass attr
  if (can_close_fusion) {
    FE_LOGD("The pass attr of pass[%s] is [%lu].", pass_name.c_str(), pass_attr);
    return IsPassAttrTypeOn(pass_attr, PassAttrType::SINGLE_OP_SCENE_MUST_ON);
  }

  // 5. default pass switch is on
  FE_LOGD("The pass switch of [%s] will be on by default.", pass_name.c_str());
  return true;
}

Status FusionConfigParser::GetFusionPriorityByFusionType(const std::string &fusion_type,
                                                         std::map<std::string, int32_t> &fusion_priority_map) const {
  auto iter = fusion_priority_map_.find(fusion_type);
  if (iter != fusion_priority_map_.end()) {
    fusion_priority_map = iter->second;
  } else {
    FE_LOGD("The fusion_priority_map is not include Fusion type [%s].", fusion_type.c_str());
  }
  return SUCCESS;
}

void FusionConfigParser::GetFusionPassNameBySwitch(const std::string &pass_type, const bool &is_switch_on,
                                                   std::vector<std::string> &pass_name_vec) const {
  pass_name_vec.clear();
  const FusionSwitchMap::const_iterator iter = fusion_switch_map_.find(pass_type);
  if (iter == fusion_switch_map_.cend()) {
    return;
  }
  std::map<std::string, bool>::const_iterator switch_iter;
  for (switch_iter = iter->second.begin(); switch_iter != iter->second.end(); switch_iter++) {
    if (is_switch_on == switch_iter->second) {
      pass_name_vec.push_back(switch_iter->first);
    }
  }
}

Status FusionConfigParser::GetKeyAndValueFromJson(const std::string &line, const string &custom_fusion_config_json_file,
                                                  std::string &key, std::string &value) const {
  size_t pos_of_equal = line.find(":");
  if (pos_of_equal == std::string::npos) {
    REPORT_FE_ERROR("[GraphOpt][FusionConfig][GetKeyValFmJs] The content of [%s] delimiter must be colon.",
                    line.c_str());
    ErrorMessageDetail err_msg(EM_INVALID_CONTENT,
                        {line, custom_fusion_config_json_file,
                        "This line " + line + " does not have the delimiter :"});
    ReportErrorMessage(err_msg);
    return FAILED;
  }
  key = line.substr(0, pos_of_equal);
  value = line.substr(pos_of_equal + 1);
  key = StringUtils::Trim(key);
  value = StringUtils::Trim(value);
  if (value != kSwitchOn && value != kSwitchOff) {
    REPORT_FE_ERROR("[GraphOpt][FusionConfig][GetKeyValFmJs] The pass switch value: [%s] is not on or off.",
                    value.c_str());
    std::ostringstream oss;
    oss << "The switch value of pass " << key;
    oss << " must be on or off, instead of " << value << "";
    ErrorMessageDetail err_msg(EM_INVALID_CONTENT, {value, custom_fusion_config_json_file, oss.str()});
    ReportErrorMessage(err_msg);
    return FAILED;
  }
  return SUCCESS;
}

Status FusionConfigParser::LoadOldFormatFusionSwitchFile(const string &custom_fusion_config_json_file,
                                                         std::map<string, bool> &old_fusion_switch_map) const {
  std::ifstream ifs(custom_fusion_config_json_file);
  if (!ifs.is_open()) {
    REPORT_FE_ERROR("[GraphOpt][FusionConfig][LdOldFmtFusSwtFile] The file [%s] does not exist or has been opened.",
                    custom_fusion_config_json_file.c_str());
    ErrorMessageDetail err_msg(EM_OPEN_FILE_FAILED, {custom_fusion_config_json_file});
    ReportErrorMessage(err_msg);
    return FILE_NOT_EXIST;
  }

  std::string line;
  while (std::getline(ifs, line)) {
    if (line.empty()) {
      continue;
    }

    std::string key;
    std::string value;
    Status ret = GetKeyAndValueFromJson(line, custom_fusion_config_json_file, key, value);
    if (ret != SUCCESS) {
      ifs.close();
      return ret;
    }

    if (!key.empty()) {
      if (value != kSwitchOn && value != kSwitchOff) {
        REPORT_FE_ERROR("[GraphOpt][FusionConfig][LdOldFmtFusSwtFile] Pass switch[%s, %s] is not correct.",
                        key.c_str(), value.c_str());
        ifs.close();
        ErrorMessageDetail err_msg(EM_INVALID_CONTENT, {key, custom_fusion_config_json_file,
                                   "pass name " + key + " is duplicate"});
        ReportErrorMessage(err_msg);
        return FAILED;
      }
      const std::map<std::string, bool>::const_iterator iter_switch = old_fusion_switch_map.find(key);
      if (iter_switch != old_fusion_switch_map.end()) {
        REPORT_FE_ERROR("[GraphOpt][FusionConfig][LdOldFmtFusSwtFile] Pass[%s] is repetitive, please check it.",
                        key.c_str());
        ifs.close();
        ErrorMessageDetail err_msg(EM_INVALID_CONTENT, {key, custom_fusion_config_json_file,
                                   "pass name " + key + " is duplicate"});
        ReportErrorMessage(err_msg);
        return FAILED;
      }
      old_fusion_switch_map.emplace(key, value == kSwitchOn);
    }
  }
  ifs.close();
  return SUCCESS;
}

Status FusionConfigParser::VerifyAndParserCustomFile(const string &custom_fusion_config_json_file,
                                                     nlohmann::json &custom_fusion_config_json,
                                                     std::map<string, bool> &old_fusion_switch_map) const {
  // Check custom file(path) is legal or not.
  if (!custom_fusion_config_json_file.empty()) {
    if (GetRealPath(custom_fusion_config_json_file).empty()) {
      REPORT_FE_ERROR("[GraphOpt][Init][VerifyAndParserCustomFile] The file path: [%s] is not legal.",
                      custom_fusion_config_json_file.c_str());
      ErrorMessageDetail err_msg(EM_OPEN_FILE_FAILED, {custom_fusion_config_json_file});
      ReportErrorMessage(err_msg);
      return FAILED;
    }
    if (CheckFileEmpty(custom_fusion_config_json_file)) {
      REPORT_FE_ERROR("[GraphOpt][Init][VerifyAndParserCustomFile] The file [%s] is empty.",
                      custom_fusion_config_json_file.c_str());
      ErrorMessageDetail err_msg(EM_OPEN_FILE_FAILED, {custom_fusion_config_json_file});
      ReportErrorMessage(err_msg);
      return FAILED;
    }
    if (ReadJsonFile(custom_fusion_config_json_file, custom_fusion_config_json) != SUCCESS) {
      FE_LOGD("Read Json content from file:[%s] unsuccess and start to try read this file by old format.",
              custom_fusion_config_json_file.c_str());
      if (LoadOldFormatFusionSwitchFile(custom_fusion_config_json_file, old_fusion_switch_map) != SUCCESS) {
        REPORT_FE_ERROR("[GraphOpt][Init][VerifyAndParserCustomFile] Failed to read this file[%s] by old format.",
                        custom_fusion_config_json_file.c_str());
        return FAILED;
      }
      FE_LOGD("Read this file[%s] by old format successfully.", custom_fusion_config_json_file.c_str());
    }
  }
  return SUCCESS;
}

Status FusionConfigParser::CheckConfigFileFormat(const nlohmann::json &custom_fusion_config_json,
                                                 const nlohmann::json &build_in_fusion_config_json,
                                                 const string &built_in_fusion_config_json_file,
                                                 const string &custom_fusion_config_json_file) const {
  // Judge built-in and custom file(if have) 's top form is or not json object.
  if (custom_fusion_config_json != nullptr && !custom_fusion_config_json.is_object()) {
    REPORT_FE_ERROR(
        "[GraphOpt][Init][CheckCfgFileFormat] Top level of fusion config file is [%s], which should be object.",
        GetJsonType(custom_fusion_config_json).c_str());
    ErrorMessageDetail err_msg(EM_OPEN_FILE_FAILED, {custom_fusion_config_json_file});
    ReportErrorMessage(err_msg);
    return ILLEGAL_JSON;
  }
  if (!build_in_fusion_config_json.is_object()) {
    REPORT_FE_ERROR(
        "[GraphOpt][Init][CheckCfgFileFormat] Top level of built-in fusion config file is [%s],"
        "which should be object.",
        GetJsonType(build_in_fusion_config_json).c_str());
    return ILLEGAL_JSON;
  }

  // Do some judge about form, key and value.
  if (CheckFusionConfigJsonFormat(custom_fusion_config_json, kCustomType) != SUCCESS) {
    REPORT_FE_ERROR(
        "[GraphOpt][Init][CheckCfgFileFormat] Fail to check custom fusion config file format. The file path is %s",
        custom_fusion_config_json_file.c_str());
    ErrorMessageDetail err_msg(EM_READ_FILE_FAILED, {custom_fusion_config_json_file,
                                                     "Failed to check custom fusion config json file format"});
    ReportErrorMessage(err_msg);
    return FAILED;
  }
  if (CheckFusionConfigJsonFormat(build_in_fusion_config_json, kBuildinType) != SUCCESS) {
    REPORT_FE_ERROR(
        "[GraphOpt][Init][CheckCfgFileFormat] Fail to check built-in fusion config file format. The file path is %s",
        built_in_fusion_config_json_file.c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status FusionConfigParser::ParseFusionConfigFile() {
  FE_LOGD("Begin to parse fusion config file for [%s].", engine_name_.c_str());
  FE_TIMECOST_START(ParseFusionConfigFile);

  string built_in_fusion_config_json_file = Configuration::Instance(engine_name_).GetBuiltInFusionConfigFilePath();
  string custom_fusion_config_json_file = FEContextUtils::GetFusionSwitchFilePath();
  if (is_first_init_) {
    is_first_init_ = false;
  } else {
    // if not first init and fusion switch file is same, no need to parser config file again.
    if (custom_fusion_config_json_file == fusion_switch_file_path_) {
      FE_LOGD("FusionConfigParser has already been init by fusion switch file[%s].",
              custom_fusion_config_json_file.c_str());
      return SUCCESS;
    }
  }
  fusion_switch_file_path_ = custom_fusion_config_json_file;
  FE_LOGD("Fusion switch file path is [%s].", custom_fusion_config_json_file.c_str());

  fusion_switch_map_.clear();
  fusion_priority_map_.clear();
  exceptional_pass_.clear();

  nlohmann::json custom_fusion_config_json;
  nlohmann::json build_in_fusion_config_json;
  std::map<string, bool> old_fusion_switch_map;
  if (VerifyAndParserCustomFile(custom_fusion_config_json_file, custom_fusion_config_json,
      old_fusion_switch_map) != SUCCESS) {
    return FAILED;
  }

  // Read fusion config file
  if (ReadJsonFile(built_in_fusion_config_json_file, build_in_fusion_config_json) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Init][ParseFusConfig] Read Json content from file:[%s] failed.",
                    built_in_fusion_config_json_file.c_str());
    return FAILED;
  }
  try {
    Status ret = CheckConfigFileFormat(custom_fusion_config_json, build_in_fusion_config_json,
                                       built_in_fusion_config_json_file, custom_fusion_config_json_file);
    if (ret != SUCCESS) {
      return ret;
    }
    ConstructFusionSwitchMap(custom_fusion_config_json, build_in_fusion_config_json, old_fusion_switch_map);

    if (ConstructFusionPriorityMap(custom_fusion_config_json, build_in_fusion_config_json) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][Init][ParseFusionPrio]Failed to init fusion priority with file [%s] or [%s]."
                      "ErrNo: %u",
                      built_in_fusion_config_json_file.c_str(), custom_fusion_config_json_file.c_str(), ret);
      return FAILED;
    }
  } catch (const std::exception &e) {
    REPORT_FE_ERROR(
        "[GraphOpt][Init][ParseFusConfig] Failed to parse fusion config file[%s] or [%s] to Json. Error message is [%s]",
        custom_fusion_config_json_file.c_str(), built_in_fusion_config_json_file.c_str(), e.what());
    return ILLEGAL_JSON;
  }

  FE_TIMECOST_END(ParseFusionConfigFile, "FusionConfigParser::ParseFusionConfigFile");
  FE_LOGI("[%s] ParseFusionConfigFile success.", engine_name_.c_str());
  return SUCCESS;
}

void FusionConfigParser::ParseSupportFusionPassFile() {
  FE_LOGD("Begin to parse support fusion pass file for [%s].", engine_name_.c_str());
  FE_TIMECOST_START(ParseSupportFusionPassFile);
  string support_fusion_pass_file_path = Configuration::Instance(engine_name_).GetSupportFusionPassFilePath();
  FE_LOGD("Support fusion pass file path is %s.", support_fusion_pass_file_path.c_str());
  nlohmann::json support_fusion_pass_file_json;
  if (ReadJsonFile(support_fusion_pass_file_path, support_fusion_pass_file_json) != SUCCESS) {
    FE_LOGW("[GraphOpt][Init][ParseFusConfig] Read Json content from file:[%s] unsuccessful.",
             support_fusion_pass_file_path.c_str());
    return;
  }

  std::string short_soc_version = PlatformUtils::Instance().GetShortSocVersion();
  auto iter = support_fusion_pass_file_json.find(short_soc_version);
  if (iter != support_fusion_pass_file_json.end()) {
    has_support_fusion_pass_ = true;
    std::string cur_support_fusion_pass = iter.value();
    support_fusion_pass_vec_ = StringUtils::Split(StringUtils::Trim(cur_support_fusion_pass), ',');
    FE_LOGD("Support fusion pass vec is %s with cur_soc_version[%s].",
             StringUtils::StrVecToString(support_fusion_pass_vec_).c_str(), short_soc_version.c_str());
  } else {
    has_support_fusion_pass_ = false;
    FE_LOGI("Not configure support fusion pass with cur soc_version[%s].", short_soc_version.c_str());
  }

  FE_TIMECOST_END(ParseSupportFusionPassFile, "FusionConfigParser::ParseSupportFusionPassFile");
  FE_LOGI("[%s] ParseSupportFusionPassFile success.", engine_name_.c_str());
  return;
}

void FusionConfigParser::ConstructFusionSwitchMap(const nlohmann::json &custom_pass_config_json,
                                                  const nlohmann::json &builtin_pass_config_json,
                                                  const std::map<string, bool> &old_fusion_switch_map) {
  for (const std::string &fusion_type : kFusionTypeVec) {
    // for the old fusion switch version, we can not tell whether it is graph fusion or ub fusion
    std::map<std::string, bool> fusion_switch_map = old_fusion_switch_map;
    // here deal with the custom first, in ModifyFusionSwitchMap function, data will not be coverd
    ModifyFusionSwitchMap(fusion_type, custom_pass_config_json, fusion_switch_map);
    ModifyFusionSwitchMap(fusion_type, builtin_pass_config_json, fusion_switch_map);
    fusion_switch_map_.emplace(fusion_type, fusion_switch_map);
  }

  ConstructExceptionalPassMap(custom_pass_config_json, exceptional_pass_);
  ConstructExceptionalPassMap(builtin_pass_config_json, exceptional_pass_);
}

void FusionConfigParser::ConstructExceptionalPassMap(const nlohmann::json &fusion_pass_config_json,
                                                    std::set<std::string> &exceptional_pass) const {
  if (fusion_pass_config_json == nullptr ||
    fusion_pass_config_json.find(kExceptionalPass) == fusion_pass_config_json.end()) {
    return;
  }
  // after file check, exceptional pass must be array
  if (fusion_pass_config_json[kExceptionalPass].is_array()) {
    for (auto &array_json : fusion_pass_config_json[kExceptionalPass]) {
      std::string json_value = array_json.get<std::string>();
      if (json_value != "") {
        exceptional_pass.emplace(json_value);
      }
    }
  }
  return;
}

void FusionConfigParser::ModifyFusionSwitchMap(const std::string &fusion_type,
                                               const nlohmann::json &fusion_pass_config_json,
                                               std::map<std::string, bool> &fusion_switch_map) const {
  if (fusion_pass_config_json == nullptr ||
      fusion_pass_config_json.find(kSwitchType) == fusion_pass_config_json.end() ||
      fusion_pass_config_json[kSwitchType].find(fusion_type) == fusion_pass_config_json[kSwitchType].end()) {
    return;
  }
  if (!fusion_pass_config_json[kSwitchType][fusion_type].is_object()) {
    FE_LOGW("The third level of json file should be object, actually is %s.",
            GetJsonType(fusion_pass_config_json[kSwitchType][fusion_type]).c_str());
    return;
  }
  for (auto &item_json : fusion_pass_config_json[kSwitchType][fusion_type].items()) {
    string key_inner = item_json.key();
    string value_inner = item_json.value();
    if (!key_inner.empty()) {
      fusion_switch_map.emplace(key_inner, value_inner == kSwitchOn);
    }
  }
}

Status FusionConfigParser::ConstructFusionPriorityMap(const nlohmann::json &custom_pass_config_json,
                                                      const nlohmann::json &builtin_pass_config_json) {
  for (const std::string &fusion_type : kFusionTypeVec) {
    // here deal with the custom first, in ModifyFusionPriorityMap function, data will not be coverd
    std::map<string, int32_t> fusion_priority_map;
    if (ModifyFusionPriorityMap(fusion_type, custom_pass_config_json, fusion_priority_map) != SUCCESS) {
      return FAILED;
    }
    if (ModifyFusionPriorityMap(fusion_type, builtin_pass_config_json, fusion_priority_map) != SUCCESS) {
      return FAILED;
    }
    fusion_priority_map_.emplace(std::make_pair(fusion_type, fusion_priority_map));
  }
  return SUCCESS;
}

Status FusionConfigParser::ModifyFusionPriorityMap(const std::string &fusion_type,
                                                   const nlohmann::json &fusion_pass_config_json,
                                                   std::map<std::string, int32_t> &fusion_priority_map) const {
  if (fusion_pass_config_json == nullptr ||
      fusion_pass_config_json.find(kPriorityType) == fusion_pass_config_json.end() ||
      fusion_pass_config_json[kPriorityType].find(fusion_type) == fusion_pass_config_json[kPriorityType].end()) {
    return SUCCESS;
  }
  if (!fusion_pass_config_json[kPriorityType][fusion_type].is_object()) {
    FE_LOGW("The third level of json file should be object, actually is %s.",
            GetJsonType(fusion_pass_config_json[kPriorityType][fusion_type]).c_str());
    return SUCCESS;
  }

  for (auto &elem_pro_level : fusion_pass_config_json[kPriorityType][fusion_type].items()) {
    string fusion_level = elem_pro_level.key();
    for (auto &elem_inner : fusion_pass_config_json[kPriorityType][fusion_type][fusion_level].items()) {
      string key_inner = elem_inner.key();
      string value_inner = elem_inner.value();
      string key_inner_string = StringUtils::Trim(key_inner);
      string value_inner_string = StringUtils::Trim(value_inner);
      // This is safety, because CheckFusionConfigJsonLastLayerFormat has been checked the value.
      int32_t priority = 0;
      try {
        priority = stoi(value_inner_string);
      } catch (...) {
        REPORT_FE_ERROR("[GraphOpt][FusionConfig][MdfFusPrioMap] Value %s in priority json is not integer.",
                        value_inner_string.c_str());
        return FAILED;
      }
      if (!key_inner.empty()) {
        fusion_priority_map.emplace(std::make_pair(key_inner_string, priority));
      }
    }
  }
  return SUCCESS;
}


Status FusionConfigParser::JudgePriority(string level_elem, int32_t priority_value, const string &owner_type) const {
  const int32_t first_condition =
      (owner_type == kCustomType) ? CUSTOM_CFG_TOP_PRIORITY_MIN : BUILT_IN_CFG_TOP_PRIORITY_MIN;
  const int32_t second_condition =
      (owner_type == kCustomType) ? CUSTOM_CFG_MAIN_PRIORITY_MIN : BUILT_IN_CFG_MAIN_PRIORITY_MIN;
  const int32_t third_condition =
      (owner_type == kCustomType) ? CUSTOM_CFG_DOWN_PRIORITY_MIN : BUILT_IN_CFG_DOWN_PRIORITY_MIN;
  const int32_t fourth_condition =
      (owner_type == kCustomType) ? BUILT_IN_CFG_TOP_PRIORITY_MIN : CUSTOM_PASS_PRIORITY_MIN;

  string level_elem_string = StringUtils::Trim(level_elem);

  if (level_elem_string == kPriorityTop && (priority_value < first_condition || priority_value >= second_condition)) {
    REPORT_FE_ERROR(
        "[%s] The priority value %d of top level in fusion config file[%s] is illegal, which out of range [%d, %d).",
        kStatgeJdgPriority.c_str(), priority_value, owner_type.c_str(), first_condition, second_condition);
    return FAILED;
  }
  if (level_elem_string == kPriorityMain && (priority_value < second_condition || priority_value >= third_condition)) {
    REPORT_FE_ERROR(
        "%s The priority value %d of main level in fusion config file[%s] is illegal, which is out of range [%d, %d).",
        kStatgeJdgPriority.c_str(), priority_value, owner_type.c_str(), second_condition, third_condition);
    return FAILED;
  }
  if (level_elem_string == kPriorityDown && (priority_value < third_condition || priority_value >= fourth_condition)) {
    REPORT_FE_ERROR(
        "%s The priority value %d of down level in fusion config file[%s] is illegal, which is out of range [%d, %d).",
        kStatgeJdgPriority.c_str(), priority_value, owner_type.c_str(), third_condition, fourth_condition);
    return FAILED;
  }

  return SUCCESS;
}

Status FusionConfigParser::CheckLastLayerFormatForPriorityPart(string type_elem, string fusion_elem,
                                                               nlohmann::json pass_switch_file_json,
                                                               const string &owner_type) const {
  unordered_set<string> pass_name_set;
  for (auto &elem3 : pass_switch_file_json[type_elem][fusion_elem].items()) {
    string level_elem = elem3.key();
    string level_elem_string = StringUtils::Trim(level_elem);

    if (level_elem_string != kPriorityTop && level_elem_string != kPriorityMain && level_elem_string != kPriorityDown) {
      REPORT_FE_ERROR(
          "[%s] The third level key [%s] of json file[%s] is error, which should be top/main/down.",
          kStatgeChkLstLyrFmt.c_str(), level_elem.c_str(), owner_type.c_str());
      return FAILED;
    }
    if (!pass_switch_file_json[type_elem][fusion_elem][level_elem].is_object()) {
      REPORT_FE_ERROR(
          "[%s] The fourth level of json file[%s] should be json form, but which actually is [%s].",
          kStatgeChkLstLyrFmt.c_str(), owner_type.c_str(),
          GetJsonType(pass_switch_file_json[type_elem][fusion_elem][level_elem]).c_str());
      return ILLEGAL_JSON;
    }

    // This layer is pass_name:priority
    for (auto &elem4 : pass_switch_file_json[type_elem][fusion_elem][level_elem].items()) {
      string key_inner = elem4.key();
      string value_inner = elem4.value();
      string key_inner_string = StringUtils::Trim(key_inner);
      string value_inner_string = StringUtils::Trim(value_inner);
      // check the same pass
      FE_CHECK(pass_name_set.find(key_inner_string) != pass_name_set.end(),
               REPORT_FE_ERROR("[GraphOpt][FusionConfig][ChkLstLyrFmt] Priority pass [%s] is repetitive.",
                               key_inner_string.c_str()),
               return FAILED);
      pass_name_set.insert(key_inner_string);
      if (!StringUtils::IsInteger(value_inner_string)) {
        REPORT_FE_ERROR(
            "[GraphOpt][FusionConfig][ChkLstLyrFmt] The pass[%s] of priority[%s] value is illegal, please check it.",
            key_inner_string.c_str(), value_inner_string.c_str());
        return FAILED;
      }
      int32_t priority_value = 0;
      try {
        priority_value = stoi(value_inner_string);
      } catch (...) {
        REPORT_FE_ERROR("[GraphOpt][FusionConfig][ChkLstLyrFmt] Failed to convert [%s] to integer.",
                        value_inner_string.c_str());
        return FAILED;
      }
      FE_CHECK(JudgePriority(level_elem, priority_value, owner_type) != SUCCESS,
               REPORT_FE_ERROR("[%s] Failed to check pass[%s] priority of owner_type %s and level %s.",
                               kStatgeChkLstLyrFmt.c_str(), key_inner_string.c_str(), owner_type.c_str(),
                               level_elem.c_str()),
               return FAILED);
    }
  }
  return SUCCESS;
}

Status FusionConfigParser::CheckFusionConfigJsonLastLayerFormat(string type_elem, string fusion_elem,
                                                                nlohmann::json pass_switch_file_json,
                                                                const string &owner_type) const {
  if (StringUtils::Trim(type_elem) == kSwitchType) {
    // This layer is pass_name:on/off
    for (auto &elem3 : pass_switch_file_json[type_elem][fusion_elem].items()) {
      string key_inner = elem3.key();
      string value_inner = elem3.value();
      string key = StringUtils::Trim(key_inner);
      string value = StringUtils::Trim(value_inner);

      if (value != kSwitchOn && value != kSwitchOff) {
        REPORT_FE_ERROR("[GraphOpt][FusionConfig][ChkFusCfgJsLyrFmt] Illegal switch value: [%s]. Must be 'on' or 'off'.",
                        value.c_str());
        return FAILED;
      }
    }
  } else {
    // This layer is top/main/down
    Status ret = CheckLastLayerFormatForPriorityPart(type_elem, fusion_elem, pass_switch_file_json, owner_type);
    if (ret != SUCCESS) {
      return ret;
    }
  }
  return SUCCESS;
}

Status FusionConfigParser::CheckFusionConfigJsonFormat(nlohmann::json pass_switch_file_json,
                                                       const string &owner_type) const {
  FE_LOGD("Start to check [%s] fusion config file.", owner_type.c_str());

  if (pass_switch_file_json == nullptr) {
    FE_LOGD("The [%s] fusion config file does not exist, don't need to check.", owner_type.c_str());
    return SUCCESS;
  }

  // This layer is Switch/Priority.
  for (auto &elem : pass_switch_file_json.items()) {
    string type_elem = elem.key();
    string type_elem_string = StringUtils::Trim(type_elem);
    if (type_elem_string == kExceptionalPass) {
      if (!pass_switch_file_json[type_elem].is_array()) {
        REPORT_FE_ERROR("The second level of [%s] json file is [%s], which should be json form.",
            owner_type.c_str(), GetJsonType(pass_switch_file_json[type_elem]).c_str());
        return ILLEGAL_JSON;
      }
      continue;
    }
    if (type_elem_string != kSwitchType && type_elem_string != kPriorityType) {
      REPORT_FE_ERROR("[%s] The first level key [%s] of [%s] json file is error, which should be Switch or Priority.",
          kStatgeChkFusCfgJsLyrFmt.c_str(), type_elem.c_str(), owner_type.c_str());
      return FAILED;
    }
    if (!pass_switch_file_json[type_elem].is_object()) {
      REPORT_FE_ERROR("[%s] The second level of [%s] json file is [%s], which should be json form.",
          kStatgeChkFusCfgJsLyrFmt.c_str(), owner_type.c_str(), GetJsonType(pass_switch_file_json[type_elem]).c_str());
      return ILLEGAL_JSON;
    }

    // This layer is GraphFusion/UBFusion.
    for (auto &elem2 : pass_switch_file_json[type_elem].items()) {
      string fusion_elem = elem2.key();
      string fusion_elem_string = StringUtils::Trim(fusion_elem);
      if (fusion_elem_string != GRAPH_FUSION && fusion_elem_string != UB_FUSION) {
        REPORT_FE_ERROR(
            "[%s] The second level key [%s] of json file[%s] is error, which should be GraphFusion or UBFusion.",
            kStatgeChkFusCfgJsLyrFmt.c_str(), fusion_elem.c_str(), owner_type.c_str());
        return FAILED;
      }
      if (!pass_switch_file_json[type_elem][fusion_elem].is_object()) {
        REPORT_FE_ERROR("[%s] The third level of [%s] json file is [%s], which should be json form.",
            kStatgeChkFusCfgJsLyrFmt.c_str(), owner_type.c_str(),
            GetJsonType(pass_switch_file_json[type_elem][fusion_elem]).c_str());
        return ILLEGAL_JSON;
      }

      // This layer is on/off or top/main/down
      if (CheckFusionConfigJsonLastLayerFormat(type_elem, fusion_elem, pass_switch_file_json, owner_type) != SUCCESS) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}
}  // namespace fe
