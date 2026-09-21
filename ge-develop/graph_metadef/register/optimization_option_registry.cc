/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/optimization_option_registry.h"
#include "framework/common/debug/ge_log.h"

namespace {
bool IsPassOptionValid(const std::string& pass_name, const std::map<ge::OoHierarchy, std::string> &options) {
  // 一级选项必须有且仅有一个，二级选项有且仅有一个或零个
  if (options.count(ge::OoHierarchy::kH1) == 0UL) {
    GELOGW("The pass [%s] has no primary switch option", pass_name.c_str());
    return false;
  }

  for (const auto &opt : options) {
    if (opt.second.empty()) {
      GELOGW("The pass [%s] has an empty option, pass option is not registered",
             pass_name.c_str());
      return false;
    }
    if (opt.first >= ge::OoHierarchy::kEnd) {
      GELOGW("The hierarchy [%u] of option [%s] is InValid, pass name is [%s]", static_cast<uint32_t>(opt.first),
             opt.second.c_str(), pass_name.c_str());
      return false;
    }
  }
  return true;
}
}
namespace ge {
OptionRegistry &OptionRegistry::GetInstance() {
  static OptionRegistry instance;
  return instance;
}

void OptionRegistry::Register(const OoInfo &option) {
  if (registered_opt_table_.count(option.name) > 0UL) {
    GELOGW("Repeatedly register option [%s]", option.name.c_str());
    return;
  }
  (void) registered_opt_table_.emplace(option.name, option);
  GELOGD("Add optimization option [%s], OoLevel is [%s]", option.name.c_str(),
         OoInfoUtils::GenOoLevelStr(option.levels).c_str());
}

const OoInfo *OptionRegistry::FindOptInfo(const std::string &opt_name) const {
  const auto iter = registered_opt_table_.find(opt_name);
  if (iter != registered_opt_table_.cend()) {
    return &iter->second;
  }
  return nullptr;
}

std::unordered_map<std::string, OoInfo> OptionRegistry::GetVisibleOptions(OoEntryPoint entry_point) const {
  std::unordered_map<std::string, OoInfo> visible_options;
  for (const auto &opt : registered_opt_table_) {
    const auto &option_info = opt.second;
    if (OoInfoUtils::IsBitSet(option_info.visibility, static_cast<uint32_t>(entry_point))) {
      const auto iter = option_info.show_infos.find(entry_point);
      if (iter != option_info.show_infos.end()) {
        (void) visible_options.emplace(iter->second.show_name, option_info);
      }
    }
  }
  return visible_options;
}

PassOptionRegistry &PassOptionRegistry::GetInstance() {
  static PassOptionRegistry instance;
  return instance;
}

void PassOptionRegistry::Register(const std::string &pass_name,
                                  const std::map<OoHierarchy, std::string> &option_names) {
  if (pass_names_to_options_.count(pass_name) > 0UL) {
    GELOGW("Repeatedly register optimization option for Pass [%s]", pass_name.c_str());
    return;
  }
  auto &opt_array = pass_names_to_options_[pass_name];
  for (const auto &opt : option_names) {
    opt_array[static_cast<uint32_t>(opt.first)] = opt.second;
    GELOGD("Add optimization option [%s] for Pass [%s]", opt.second.c_str(), pass_name.c_str());
  }
}

graphStatus PassOptionRegistry::FindOptionNamesByPassName(const std::string &pass_name,
                                                          std::vector<std::string> &option_names) const {
  const auto iter = pass_names_to_options_.find(pass_name);
  if (iter == pass_names_to_options_.end()) {
    return GRAPH_FAILED;
  }
  for (const auto &opt_name : iter->second) {
    if (!opt_name.empty()) {
      (void) option_names.emplace_back(opt_name);
    }
  }
  return GRAPH_SUCCESS;
}

OptionRegister::OptionRegister(const OptionRegister &other) : opt_reg_data_(nullptr) {
  const auto reg_data_ptr = other.opt_reg_data_.get();
  if (reg_data_ptr == nullptr) {
    GELOGW("The opt_reg_data_ is null, failed to register optimization option");
    return;
  }

  if (reg_data_ptr->name.empty()) {
    GELOGW("The option name is empty, failed to register optimization option");
    return;
  }

  if (reg_data_ptr->levels == 0UL) {
    GELOGW("The option level is not set or invalid, failed to register optimization option");
    return;
  }

  if (reg_data_ptr->hierarchy >= OoHierarchy::kEnd) {
    GELOGW("The option hierarchy is not set or invalid, failed to register optimization option");
    return;
  }

  OptionRegistry::GetInstance().Register(*reg_data_ptr);
}

OptionRegister &OptionRegister::SetDefaultValues(const std::map<OoLevel, std::string> &opt_values) {
  if (opt_reg_data_ != nullptr) {
    if (opt_reg_data_->default_values.empty()) {
      opt_reg_data_->default_values = opt_values;
    }
  }
  return *this;
}

OptionRegister &OptionRegister::SetOptLevel(const std::vector<OoLevel> &levels) {
  if (opt_reg_data_ != nullptr) {
    if (opt_reg_data_->levels == 0UL) {
      opt_reg_data_->levels = OoInfoUtils::GenOptLevelBits(levels);
    }
  }
  return *this;
}

OptionRegister &OptionRegister::SetVisibility(const std::vector<OoEntryPoint> &entry_points) {
  if (opt_reg_data_ != nullptr) {
    if (opt_reg_data_->visibility == 0UL) {
      opt_reg_data_->visibility = OoInfoUtils::GenOptVisibilityBits(entry_points);
    }
  }
  return *this;
}

OptionRegister &OptionRegister::SetOptValueChecker(OoInfo::ValueChecker opt_checker) {
  if (opt_reg_data_ != nullptr) {
    if (opt_reg_data_->checker == nullptr) {
      opt_reg_data_->checker = opt_checker;
    }
  }
  return *this;
}

OptionRegister &OptionRegister::SetHelpText(std::string opt_help) {
  if (opt_reg_data_ != nullptr) {
    if (opt_reg_data_->help_text.empty()) {
      opt_reg_data_->help_text = std::move(opt_help);
    }
  }
  return *this;
}

OptionRegister &OptionRegister::SetShowName(OoEntryPoint entry_point, std::string show_name, ge::OoCategory category) {
  if (opt_reg_data_ != nullptr) {
    if (opt_reg_data_->show_infos.count(entry_point) == 0UL) {
      opt_reg_data_->show_infos.emplace(entry_point, OoShowInfo{category, std::move(show_name)});
    }
  }
  return *this;
}

PassOptionRegister::PassOptionRegister(const PassOptionRegister &other) : pass_reg_data_(nullptr) {
  const auto pass_reg_data_ptr = other.pass_reg_data_.get();
  if (pass_reg_data_ptr == nullptr) {
    GELOGW("The pass_reg_data_ is null, failed to bind optimization option to pass");
    return;
  }
  if (pass_reg_data_ptr->pass_name.empty()) {
    GELOGW("The pass name is empty, failed to bind optimization option for pass");
    return;
  }

  if (pass_reg_data_ptr->options.empty()) {
    for (const auto level : pass_reg_data_ptr->levels) {
      if (level >= OoLevel::kEnd) {
        GELOGW("The option level [%u] of pass [%s] is invalid", static_cast<uint32_t>(level),
               pass_reg_data_ptr->pass_name.c_str());
        return;
      }
    }
    // 功能性 Pass 没有对外开放的 Option, 所以使用 PassName 作为 OptionName
    OoInfo default_option{pass_reg_data_ptr->pass_name, OoHierarchy::kH1,
                          OoInfoUtils::GenOptLevelBits(pass_reg_data_ptr->levels)};
    OptionRegistry::GetInstance().Register(default_option);
    PassOptionRegistry::GetInstance().Register(pass_reg_data_ptr->pass_name, {{OoHierarchy::kH1, default_option.name}});
  } else {
    if (IsPassOptionValid(pass_reg_data_ptr->pass_name, pass_reg_data_ptr->options)) {
      PassOptionRegistry::GetInstance().Register(pass_reg_data_ptr->pass_name, pass_reg_data_ptr->options);
    }
  }
}

PassOptionRegister &PassOptionRegister::SetOptLevel(const std::vector<OoLevel> &levels) {
  if (pass_reg_data_ != nullptr) {
    if (pass_reg_data_->levels.empty()) {
      pass_reg_data_->levels = levels;
    }
  }
  return *this;
}

PassOptionRegister &PassOptionRegister::BindSwitchOption(const std::string &opt_name, OoHierarchy hierarchy) {
  if (pass_reg_data_ != nullptr) {
    (void) pass_reg_data_->options.emplace(hierarchy, opt_name);
  }
  return *this;
}
}  // namespace ge