/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/dump/dump_properties.h"

#include <string>
#include <regex>

#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_types.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/utils/attr_utils.h"
#include "mmpa/mmpa_api.h"
#include "common/global_variables/diagnose_switch.h"
#include "base/err_msg.h"

namespace {
const std::string kEnableFlag = "1";
const std::string kDumpStatusOpen = "on";
constexpr uint32_t kAicoreOverflow = 0x1U; // (0x1U << 0U)
constexpr uint32_t kAtomicOverflow = 0x2U; // (0x1U << 1U)
constexpr uint32_t kAllOverflow = kAicoreOverflow | kAtomicOverflow;
constexpr size_t kMaxErrorStrLength = 128U;
}  // namespace
namespace ge {
StepCheckErrorCode DumpProperties::CheckDumpStepInner(const std::string &dump_step) const {
  const auto match_vecs = StringUtils::Split(dump_step, '|');
  // 100 is the max sets of dump steps.
  if (match_vecs.size() > 100U) {
    return StepCheckErrorCode::kTooManySets;
  }
  const auto is_str_digit = [] (const std::string &s) -> bool {
    return (StringUtils::IsSignedInt32(s)) && (isdigit(s.at(0U)) != 0);
  };
  for (const auto &match_vec : match_vecs) {
    const auto vec_after_split = StringUtils::Split(match_vec, '-');
    if (vec_after_split.size() == 1U) {
      if (!is_str_digit(vec_after_split[0])) {
        return StepCheckErrorCode::kIncorrectFormat;
      }
    } else if (vec_after_split.size() == 2U) {
      if ((!is_str_digit(vec_after_split[0])) || (!is_str_digit(vec_after_split[1]))) {
        return StepCheckErrorCode::kIncorrectFormat;
      }
      if (std::strtol(vec_after_split[0U].c_str(), nullptr, 10 /* base int */) >=
          std::strtol(vec_after_split[1U].c_str(), nullptr, 10 /* base int */)) {
        return StepCheckErrorCode::kReverseStep;
      }
    } else {
      return StepCheckErrorCode::kIncorrectFormat;
    }
  }
  return StepCheckErrorCode::kNoError;
}

Status DumpProperties::CheckDumpStep(const std::string &dump_step) const {
  const auto error_code = CheckDumpStepInner(dump_step);
  Status ret;
  switch (error_code) {
    case StepCheckErrorCode::kTooManySets:
      (void)REPORT_PREDEFINED_ERR_MSG(
          "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
          std::vector<const char *>({ge::GetContext().GetReadableName("ge.exec.dumpStep").c_str(), dump_step.c_str(),
                                    "The number of collection ranges cannot exceed 100."}));
      GELOGE(PARAM_INVALID, "[Check][Param] get dump_step value:%s, "
                            "dump_step only support dump <= 100 sets of data.", dump_step.c_str());
      ret = PARAM_INVALID;
      break;
    case StepCheckErrorCode::kIncorrectFormat:
      (void)REPORT_PREDEFINED_ERR_MSG(
          "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
          std::vector<const char *>({ge::GetContext().GetReadableName("ge.exec.dumpStep").c_str(), dump_step.c_str(),
                                    "Incorrect parameter format. Valid format: '0|5|10|50-100'."}));
      GELOGE(PARAM_INVALID, "[Check][Param] get dump_step value:%s, dump_step std::string style is error, "
             "correct example:'0|5|10|50-100'.", dump_step.c_str());
      ret = PARAM_INVALID;
      break;
    case StepCheckErrorCode::kReverseStep:
      (void)REPORT_PREDEFINED_ERR_MSG(
          "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
          std::vector<const char *>({ge::GetContext().GetReadableName("ge.exec.dumpStep").c_str(), dump_step.c_str(),
                                     "In range steps, the first step must be greater than or equal to the second step. "
                                     "Valid format: '0|5|10-20'."}));
      GELOGE(PARAM_INVALID, "[Check][Param] get dump_step value:%s, in range steps, the first step is >= second step, "
             "correct example:'0|5|10-20'.", dump_step.c_str());
      ret = PARAM_INVALID;
      break;
    default:
      ret = SUCCESS;
      break;
  }
  return ret;
}

Status DumpProperties::CheckDumpMode(const std::string &dump_mode) const {
  const std::set<std::string> dump_mode_list = {"input", "output", "all"};
  const auto iter = dump_mode_list.find(dump_mode);
  if (iter == dump_mode_list.end()) {
    (void)REPORT_PREDEFINED_ERR_MSG(
        "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>({ge::GetContext().GetReadableName("ge.exec.dumpMode").c_str(), dump_mode.c_str(),
                                  "The dump mode must be selected from [input, output, all]."}));
    GELOGE(PARAM_INVALID, "[Check][Param] the dump_debug_mode:%s, is is not supported,"
           "should be one of the following:[input, output, all].", dump_mode.c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status DumpProperties::CheckDumpPathValid(const std::string &input) const {
  if (mmIsDir(input.c_str()) != EN_OK) {
    char_t err_buf[kMaxErrorStrLength + 1U] = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStrLength);
    std::string reason =
        "The parameter value is not a directory. [Errno " + std::to_string(mmGetErrorCode()) + "] " + err_msg + ".";
    (void)REPORT_PREDEFINED_ERR_MSG(
        "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>(
            {ge::GetContext().GetReadableName("ge.exec.dumpPath").c_str(), input.c_str(), reason.c_str()}));
    GELOGE(PARAM_INVALID, "[Check][Param] the path:%s, is not directory.", input.c_str());
    return PARAM_INVALID;
  }
  std::array<char_t, MMPA_MAX_PATH> trusted_path = {};
  if (mmRealPath(input.c_str(), trusted_path.data(), MMPA_MAX_PATH) != EN_OK) {
    char_t err_buf[kMaxErrorStrLength + 1U] = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStrLength);
    std::string reason =
        "The parameter value is invalid. [Errno " + std::to_string(mmGetErrorCode()) + "] " + err_msg + ".";
    (void)REPORT_PREDEFINED_ERR_MSG(
        "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>(
            {ge::GetContext().GetReadableName("ge.exec.dumpPath").c_str(), input.c_str(), reason.c_str()}));
    GELOGE(PARAM_INVALID, "[Check][Param] the dumpPath:%s, is invalid.", input.c_str());
    return PARAM_INVALID;
  }
  constexpr int32_t access_flag = static_cast<int32_t>(static_cast<uint32_t>(M_R_OK) | static_cast<uint32_t>(M_W_OK));
  if (mmAccess2(trusted_path.data(), access_flag) != EN_OK) {
    char_t err_buf[kMaxErrorStrLength + 1U] = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStrLength);
    std::string reason =
        "No access permission for the path. [Errno " + std::to_string(mmGetErrorCode()) + "] " + err_msg + ".";
    (void)REPORT_PREDEFINED_ERR_MSG(
        "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>(
            {ge::GetContext().GetReadableName("ge.exec.dumpPath").c_str(), input.c_str(), reason.c_str()}));
    GELOGE(PARAM_INVALID, "[Check][Param] the path:%s, does't have read, write permissions.", input.c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status DumpProperties::CheckEnableDump(const std::string &input) const {
  std::set<std::string> enable_dump_option_list = {"1", "0"};
  const auto it = enable_dump_option_list.find(input);
  if (it == enable_dump_option_list.end()) {
    (void)REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                       std::vector<const char *>({ge::GetContext().GetReadableName("ge.exec.enableDump").c_str(), input.c_str(),
                                                 "The value must be 1 or 0."}));
    GELOGE(PARAM_INVALID, "[Check][Param] Not support ge.exec.enableDump or ge.exec.enableDumpDebug format:%s, "
           "only support 1 or 0.", input.c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

DumpProperties::DumpProperties(const DumpProperties &dump) {
  CopyFrom(dump);
}

DumpProperties &DumpProperties::operator=(const DumpProperties &dump) & {
  CopyFrom(dump);
  return *this;
}

Status DumpProperties::SetDumpOptions() {
  if (enable_dump_ == kEnableFlag) {
    std::string dump_step;
    if ((GetContext().GetOption(OPTION_EXEC_DUMP_STEP, dump_step) == GRAPH_SUCCESS) && (!dump_step.empty())) {
      GE_CHK_STATUS_RET(CheckDumpStep(dump_step), "[Check][dump_step] failed.");
      GELOGI("Get dump step %s successfully", dump_step.c_str());
      SetDumpStep(dump_step);
    }
    std::string dump_mode = "output";
    if (GetContext().GetOption(OPTION_EXEC_DUMP_MODE, dump_mode) == GRAPH_SUCCESS) {
      GELOGI("Get dump mode %s successfully", dump_mode.c_str());
      GE_CHK_STATUS_RET(CheckDumpMode(dump_mode), "[Check][dump_mode] failed.");
    }
    SetDumpMode(dump_mode);
    std::string dump_data = "tensor";
    if (GetContext().GetOption(OPTION_EXEC_DUMP_DATA, dump_data) == GRAPH_SUCCESS) {
      GELOGI("Get dump data %s successfully", dump_data.c_str());
    }
    SetDumpData(dump_data);
    std::string dump_layers;
    if ((GetContext().GetOption(OPTION_EXEC_DUMP_LAYER, dump_layers) == GRAPH_SUCCESS) && (!dump_layers.empty())) {
      GELOGI("Get dump layer %s successfully", dump_layers.c_str());
      SetDumpList(dump_layers);
    } else {
      AddPropertyValue(DUMP_ALL_MODEL, {});
    }
    diagnoseSwitch::EnableDataDump();
  }
  return SUCCESS;
}

Status DumpProperties::InitByOptions() {
  ClearDumpInfo();

  std::string enable_dump;
  (void)GetContext().GetOption(OPTION_EXEC_ENABLE_DUMP, enable_dump);
  if (!enable_dump.empty()) {
    GE_CHK_STATUS_RET(CheckEnableDump(enable_dump), "[Check][enable_dump] failed.");
    enable_dump_ = enable_dump;
  }

  std::string enable_dump_debug;
  (void)GetContext().GetOption(OPTION_EXEC_ENABLE_DUMP_DEBUG, enable_dump_debug);
  if (!enable_dump_debug.empty()) {
    GE_CHK_STATUS_RET(CheckEnableDump(enable_dump_debug), "[Check][enable_dump_debug] failed.");
    enable_dump_debug_ = enable_dump_debug;
  }
  if ((enable_dump_ == kEnableFlag) && (enable_dump_debug_ == kEnableFlag)) {
    (void)REPORT_PREDEFINED_ERR_MSG(
        "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>(
            {(ge::GetContext().GetReadableName("ge.exec.enableDump") + " and " +
              ge::GetContext().GetReadableName("ge.exec.enableDumpDebug")).c_str(),
             (enable_dump_ + ", " + enable_dump_debug_).c_str(),
             "ge.exec.enableDump and ge.exec.enableDumpDebug cannot be set to 1 at the same time."}));
    GELOGE(FAILED, "ge.exec.enableDump and ge.exec.enableDumpDebug cannot be both set to 1 at the same time.");
    return FAILED;
  }
  if (((enable_dump_ == kEnableFlag) || (enable_dump_debug_ == kEnableFlag))) {
    std::string dump_path;
    if (GetContext().GetOption(OPTION_EXEC_DUMP_PATH, dump_path) == GRAPH_SUCCESS) {
      GE_CHK_STATUS_RET(CheckDumpPathValid(dump_path), "Check dump path failed.");
      if ((!dump_path.empty()) && (dump_path[dump_path.size() - 1U] != '/')) {
        dump_path = dump_path + "/";
      }
      dump_path = dump_path + CurrentTimeInStr() + "/";
      GELOGI("Get dump path %s successfully", dump_path.c_str());
      SetDumpPath(dump_path);
    } else {
      (void)REPORT_PREDEFINED_ERR_MSG(
          "E10058", std::vector<const char *>({"parameter"}),
          std::vector<const char *>({ge::GetContext().GetReadableName("ge.exec.dumpPath").c_str()}));
      GELOGE(FAILED, "[Check][dump_path] failed. Dump path is not set.");
      return FAILED;
    }
  }

  GE_CHK_STATUS_RET(SetDumpOptions(), "SetDumpOptions failed.");

  GE_CHK_STATUS_RET(SetDumpDebugOptions(), "SetDumpDebugOptions failed.");

  return SUCCESS;
}

void DumpProperties::SetDumpList(const std::string &layers) {
  std::set<std::string> dump_layers;
  std::istringstream ss(layers);
  std::string layer;
  while (ss >> layer) {
    (void)dump_layers.insert(layer);
  }
  AddPropertyValue(DUMP_LAYER_OP_MODEL, dump_layers);
}

// The following is the new dump scenario of the fusion operator
void DumpProperties::AddPropertyValue(const std::string &model, const std::set<std::string> &layers) {
  for (const std::string &layer : layers) {
    GELOGI("This model %s config to dump layer %s", model.c_str(), layer.c_str());
  }

  GELOGD("Layers are set to model %s.", model.c_str());
  model_dump_properties_map_[model] = layers;
}

void DumpProperties::DeletePropertyValue(const std::string &model) {
  const auto iter = model_dump_properties_map_.find(model);
  if (iter != model_dump_properties_map_.end()) {
    (void)model_dump_properties_map_.erase(iter);
  }
}

void DumpProperties::ClearDumpPropertyValue() {
  model_dump_properties_map_.clear();
}

void DumpProperties::ClearDumpInfo() {
  enable_dump_.clear();
  enable_dump_debug_.clear();
  dump_path_.clear();
  dump_step_.clear();
  dump_mode_.clear();
  dump_op_switch_.clear();
  dump_status_.clear();
  dump_data_.clear();
  model_dump_blacklist_map_.clear();
  model_dump_op_ranges_map_.clear();
  is_train_op_debug_ = false;
  is_infer_op_debug_ = false;
  op_debug_mode_ = 0U;
}

std::set<std::string> DumpProperties::GetAllDumpModel() const {
  std::set<std::string> model_list;
  for (auto &iter : model_dump_properties_map_) {
    (void)model_list.insert(iter.first);
  }

  return model_list;
}

std::set<std::string> DumpProperties::GetPropertyValue(const std::string &model) const {
  const auto iter = model_dump_properties_map_.find(model);
  if (iter != model_dump_properties_map_.end()) {
    return iter->second;
  }
  return {};
}

bool DumpProperties::IsLayerOpOnWatcherMode(const std::string &op_name) const {
  if (model_dump_properties_map_.find(DUMP_WATCHER_MODEL) == model_dump_properties_map_.end()) {
    return false;
  }

  // DUMP_LAYER_OP_MODEL must be set on watcher mode
  const auto layer_op_iter = model_dump_properties_map_.find(DUMP_LAYER_OP_MODEL);
  if (layer_op_iter == model_dump_properties_map_.end()) {
    return false;
  }

  return (layer_op_iter->second.empty() || layer_op_iter->second.find(op_name) != layer_op_iter->second.end());
}

bool DumpProperties::IsWatcherNode(const std::string &op_name) const {
  const auto watcher_node_iter = model_dump_properties_map_.find(DUMP_WATCHER_MODEL);
  if (watcher_node_iter == model_dump_properties_map_.end()) {
    return false;
  }

  return (watcher_node_iter->second.find(op_name) != watcher_node_iter->second.end());
}

bool DumpProperties::IsDumpWatcherModelEnable() const {
  return (model_dump_properties_map_.find(DUMP_WATCHER_MODEL) != model_dump_properties_map_.end());
}

bool DumpProperties::IsNeedDump(const std::string &model, const std::string &om_name,
                                     const std::string &op_name) const {
  // if dump all
  if (model_dump_properties_map_.find(DUMP_ALL_MODEL) != model_dump_properties_map_.end()) {
    return true;
  }

  // if dump layer need dump
  if (model_dump_properties_map_.find(DUMP_LAYER_OP_MODEL) != model_dump_properties_map_.end()) {
    const auto dump_name_iter = model_dump_properties_map_.find(DUMP_LAYER_OP_MODEL);
    if (dump_name_iter->second.empty()) {
      return true;
    }
    return dump_name_iter->second.find(op_name) != dump_name_iter->second.end();
  }

  // if this model need dump
  const auto om_name_iter = model_dump_properties_map_.find(om_name);
  const auto model_name_iter = model_dump_properties_map_.find(model);
  if ((om_name_iter != model_dump_properties_map_.end()) || (model_name_iter != model_dump_properties_map_.end())) {
    // if no dump layer info, dump all layer in this model
    const auto model_iter = (om_name_iter != model_dump_properties_map_.end()) ? om_name_iter : model_name_iter;
    if (model_iter->second.empty() && (GetDumpOpRangeSize(model, om_name) == 0U)) {
      return true;
    }

    return model_iter->second.find(op_name) != model_iter->second.end();
  }

  return false;
}

// wrapper for IsNeedDump with dfx(log)
bool DumpProperties::IsLayerNeedDump(const std::string &model, const std::string &om_name,
                                     const std::string &op_name) const {
  const bool need_dump = IsNeedDump(model, om_name, op_name);
  GELOGI("model[%s] om name[%s] op %s dump flag is %d", model.c_str(), om_name.c_str(), op_name.c_str(),
         static_cast<int32_t>(need_dump));
  return need_dump;
}

bool DumpProperties::IsDumpLayerOpModelEnable() const {
  return (model_dump_properties_map_.find(DUMP_LAYER_OP_MODEL) != model_dump_properties_map_.end());
}

void DumpProperties::SetDumpPath(const std::string &path) {
  dump_path_ = path;
}

const std::string &DumpProperties::GetDumpPath() const {
  return dump_path_;
}

void DumpProperties::SetDumpStep(const std::string &step) {
  dump_step_ = step;
}

const std::string &DumpProperties::GetDumpStep() const {
  return dump_step_;
}

void DumpProperties::SetDumpWorkerId(const std::string &worker_id) {
  dump_worker_id_ = worker_id;
}

const std::string &DumpProperties::GetDumpWorkerId() const {
  return dump_worker_id_;
}

void DumpProperties::SetDumpMode(const std::string &mode) {
  dump_mode_ = mode;
}

const std::string &DumpProperties::GetDumpMode() const {
  return dump_mode_;
}

void DumpProperties::SetDumpData(const std::string &data) {
  dump_data_ = data;
}

const std::string &DumpProperties::GetDumpData() const {
  return dump_data_;
}

const std::map<std::string, ModelOpBlacklist> &DumpProperties::GetModelDumpBlacklistMap() const {
    return model_dump_blacklist_map_;
}

void DumpProperties::SetModelDumpBlacklistMap(const std::map<std::string, ModelOpBlacklist>& new_map) {
    model_dump_blacklist_map_ = new_map;
}

void DumpProperties::SetModelBlacklist(const std::string& model_name,
                                      const ModelOpBlacklist& blacklist) {
    // 如果model_name不存在，直接插入新的blacklist
    if (model_dump_blacklist_map_.find(model_name) == model_dump_blacklist_map_.end()) {
        model_dump_blacklist_map_[model_name] = blacklist;
        GELOGI("Set blacklist for new model[%s].", model_name.c_str());
        return;
    }

    // 如果已存在，合并新的blacklist到现有的
    ModelOpBlacklist& existing_blacklist = model_dump_blacklist_map_[model_name];

    // 统计新增的blacklist数量
    size_t new_optype_count = 0;
    size_t new_opname_count = 0;
    size_t merged_optype_count = 0;
    size_t merged_opname_count = 0;

    // 合并dump_optype_blacklist
    for (auto it = blacklist.dump_optype_blacklist.begin(); it != blacklist.dump_optype_blacklist.end(); ++it) {
        const std::string& op_type = it->first;
        const OpBlacklist& new_op_blacklist = it->second;

        if (existing_blacklist.dump_optype_blacklist.find(op_type) == existing_blacklist.dump_optype_blacklist.end()) {
            existing_blacklist.dump_optype_blacklist[op_type] = new_op_blacklist;
            new_optype_count++;
            GELOGI("Add new op_type blacklist for model[%s]: op_type[%s] with %zu input indices and %zu output indices.",
                   model_name.c_str(), op_type.c_str(),
                   new_op_blacklist.input_indices.size(),
                   new_op_blacklist.output_indices.size());
        } else {
            OpBlacklist& existing_op_blacklist = existing_blacklist.dump_optype_blacklist[op_type];
            const size_t old_input_size = existing_op_blacklist.input_indices.size();
            const size_t old_output_size = existing_op_blacklist.output_indices.size();

            existing_op_blacklist.input_indices.insert(new_op_blacklist.input_indices.begin(), new_op_blacklist.input_indices.end());
            existing_op_blacklist.output_indices.insert(new_op_blacklist.output_indices.begin(), new_op_blacklist.output_indices.end());

            merged_optype_count++;
            GELOGI("Merge op_type blacklist for model[%s]: op_type[%s], input indices: %zu -> %zu, output indices: %zu -> %zu.",
                   model_name.c_str(), op_type.c_str(),
                   old_input_size, existing_op_blacklist.input_indices.size(),
                   old_output_size, existing_op_blacklist.output_indices.size());
        }
    }

    // 合并dump_opname_blacklist
    for (auto it = blacklist.dump_opname_blacklist.begin(); it != blacklist.dump_opname_blacklist.end(); ++it) {
        const std::string& op_name = it->first;
        const OpBlacklist& new_op_blacklist = it->second;

        if (existing_blacklist.dump_opname_blacklist.find(op_name) == existing_blacklist.dump_opname_blacklist.end()) {
            existing_blacklist.dump_opname_blacklist[op_name] = new_op_blacklist;
            new_opname_count++;
            GELOGI("Add new op_name blacklist for model[%s]: op_name[%s] with %zu input indices and %zu output indices.",
                   model_name.c_str(), op_name.c_str(),
                   new_op_blacklist.input_indices.size(),
                   new_op_blacklist.output_indices.size());
        } else {
            OpBlacklist& existing_op_blacklist = existing_blacklist.dump_opname_blacklist[op_name];
            const size_t old_input_size = existing_op_blacklist.input_indices.size();
            const size_t old_output_size = existing_op_blacklist.output_indices.size();

            existing_op_blacklist.input_indices.insert(new_op_blacklist.input_indices.begin(), new_op_blacklist.input_indices.end());
            existing_op_blacklist.output_indices.insert(new_op_blacklist.output_indices.begin(), new_op_blacklist.output_indices.end());

            merged_opname_count++;
            GELOGI("Merge op_name blacklist for model[%s]: op_name[%s], input indices: %zu -> %zu, output indices: %zu -> %zu.",
                   model_name.c_str(), op_name.c_str(),
                   old_input_size, existing_op_blacklist.input_indices.size(),
                   old_output_size, existing_op_blacklist.output_indices.size());
        }
    }

    // 打印总体统计信息
    GELOGI("Set blacklist for model[%s] completed: %zu new op_types, %zu merged op_types, %zu new op_names, %zu merged op_names.",
           model_name.c_str(), new_optype_count, merged_optype_count, new_opname_count, merged_opname_count);
}

bool DumpProperties::IsInputInOpNameBlacklist(const std::string &model_name,
                                            const std::string &op_name,
                                            const uint32_t index) const {
    auto model_it = model_dump_blacklist_map_.find(model_name);
    if (model_it == model_dump_blacklist_map_.end()) {
        return false;
    }

    auto op_input_it = model_it->second.dump_opname_blacklist.find(op_name);
    if (op_input_it == model_it->second.dump_opname_blacklist.end()) {
      return false;
    }

    return op_input_it->second.input_indices.count(index) > 0;
}
bool DumpProperties::IsOutputInOpNameBlacklist(const std::string &model_name,
                                             const std::string &op_name,
                                             const uint32_t index) const {
    auto model_it = model_dump_blacklist_map_.find(model_name);
    if (model_it == model_dump_blacklist_map_.end()) {
      return false;
    }

    auto op_output_it = model_it->second.dump_opname_blacklist.find(op_name);
    if (op_output_it == model_it->second.dump_opname_blacklist.end()) {
      return false;
    }

    return op_output_it->second.output_indices.count(index) > 0;
}

bool DumpProperties::IsInputInOpTypeBlacklist(const std::string &model_name,
                                            const std::string &op_type,
                                            const uint32_t index) const {
    auto model_it = model_dump_blacklist_map_.find(model_name);
    if (model_it == model_dump_blacklist_map_.end()) {
      return false;
    }

    auto op_input_it = model_it->second.dump_optype_blacklist.find(op_type);
    if (op_input_it == model_it->second.dump_optype_blacklist.end()) {
      return false;
    }

    return op_input_it->second.input_indices.count(index) > 0;
}

bool DumpProperties::IsOutputInOpTypeBlacklist(const std::string &model_name,
                                             const std::string &op_type,
                                             const uint32_t index) const {
    auto model_it = model_dump_blacklist_map_.find(model_name);
    if (model_it == model_dump_blacklist_map_.end()) {
      return false;
    }

    auto op_output_it = model_it->second.dump_optype_blacklist.find(op_type);
    if (op_output_it == model_it->second.dump_optype_blacklist.end()) {
      return false;
    }

    return op_output_it->second.output_indices.count(index) > 0;
}

void DumpProperties::SetDumpStatus(const std::string &status) {
  dump_status_ = status;
}

void DumpProperties::InitInferOpDebug() {
  is_infer_op_debug_ = true;
}

void DumpProperties::SetOpDebugMode(const uint32_t op_debug_mode) {
  op_debug_mode_ = op_debug_mode;
}

void DumpProperties::SetDumpOpSwitch(const std::string &dump_op_switch) {
  dump_op_switch_ = dump_op_switch;
}

bool DumpProperties::IsSingleOpNeedDump() const {
  if (dump_op_switch_ == kDumpStatusOpen) {
    return true;
  }
  return false;
}

bool DumpProperties::IsDumpOpen() const {
  if ((enable_dump_ == kEnableFlag) || (dump_status_ == kDumpStatusOpen)) {
    return true;
  }
  return false;
}

void DumpProperties::CopyFrom(const DumpProperties &other) {
  if (&other != this) {
    enable_dump_ = other.enable_dump_;
    enable_dump_debug_ = other.enable_dump_debug_;
    dump_path_ = other.dump_path_;
    dump_step_ = other.dump_step_;
    dump_mode_ = other.dump_mode_;
    dump_status_ = other.dump_status_;
    dump_op_switch_ = other.dump_op_switch_;
    dump_data_ = other.dump_data_;
    model_dump_blacklist_map_ = other.model_dump_blacklist_map_;
    model_dump_op_ranges_map_ = other.model_dump_op_ranges_map_;
    model_dump_properties_map_ = other.model_dump_properties_map_;
    is_train_op_debug_ = other.is_train_op_debug_;
    is_infer_op_debug_ = other.is_infer_op_debug_;
    op_debug_mode_ = other.op_debug_mode_;
  }
}

Status DumpProperties::SetDumpDebugOptions() {
  if (enable_dump_debug_ == kEnableFlag) {
    std::string dump_debug_mode;
    if (GetContext().GetOption(OPTION_EXEC_DUMP_DEBUG_MODE, dump_debug_mode) == GRAPH_SUCCESS) {
      GELOGD("Get ge.exec.dumpDebugMode %s successfully.", dump_debug_mode.c_str());
    } else {
      GELOGW("ge.exec.dumpDebugMode is not set.");
      return SUCCESS;
    }

    if (dump_debug_mode == OP_DEBUG_AICORE) {
      GELOGD("ge.exec.dumpDebugMode=aicore_overflow, op debug is open.");
      is_train_op_debug_ = true;
      op_debug_mode_ = kAicoreOverflow;
    } else if (dump_debug_mode == OP_DEBUG_ATOMIC) {
      GELOGD("ge.exec.dumpDebugMode=atomic_overflow, op debug is open.");
      is_train_op_debug_ = true;
      op_debug_mode_ = kAtomicOverflow;
    } else if (dump_debug_mode == OP_DEBUG_ALL) {
      GELOGD("ge.exec.dumpDebugMode=all, op debug is open.");
      is_train_op_debug_ = true;
      op_debug_mode_ = kAllOverflow;
    } else {
      (void)REPORT_PREDEFINED_ERR_MSG(
          "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
          std::vector<const char *>({ge::GetContext().GetReadableName("ge.exec.dumpDebugMode").c_str(),
                                     dump_debug_mode.c_str(), "The current value is not within the valid range."}));
      GELOGE(PARAM_INVALID, "[Set][DumpDebugOptions] failed, ge.exec.dumpDebugMode is invalid.");
      return PARAM_INVALID;
    }
    diagnoseSwitch::EnableOverflowDump();
  } else {
    GELOGI("ge.exec.enableDumpDebug is false or is not set");
  }
  return SUCCESS;
}

void DumpProperties::SetOpDumpRange(const std::string &model,
                                    const std::vector<std::pair<std::string, std::string>> &op_ranges) {
  if (op_ranges.empty() || model.empty()) {
    return;
  }

  model_dump_op_ranges_map_[model].clear();
  model_dump_op_ranges_map_[model].assign(op_ranges.begin(), op_ranges.end());
  for (const auto &model_range : model_dump_op_ranges_map_) {
    const std::string &model_name = model_range.first;
    const auto &ranges = model_range.second;
    for (const auto &range : ranges) {
      GELOGI("Model name[%s] opname range begin[%s] range end[%s].",
        model_name.c_str(), range.first.c_str(), range.second.c_str());
    }
  }
}

/*
  * |           | kRangeStart | kRangeEnd | kOutofRange
  * |-----------|------------ |-----------|-------------|
  * | kInit     | kStart      | kError    |  kInit      |
  * | kStart    | kStart      | kStop     |  kStart     |
  * | kStop     | kError      | kStop     |  kInit      |
  * | kError    | kError      | kError    |  kError     |
  */
DumpProcState DumpProperties::DumpProcFsmTransition(DumpProcState current, DumpProcEvent evt) {
  static constexpr std::size_t N_STATES = 4U;
  static constexpr std::size_t N_EVENTS = 3U;

  // 行列顺序必须与枚举定义一致
  static constexpr std::array<std::array<DumpProcState, N_EVENTS>, N_STATES> kTable = {{
      /* kInit  */
      {DumpProcState::kStart, DumpProcState::kError, DumpProcState::kInit},
      /* kStart */
      {DumpProcState::kStart, DumpProcState::kStop,  DumpProcState::kStart},
      /* kStop  */
      {DumpProcState::kError, DumpProcState::kStop,  DumpProcState::kInit},
      /* kError */
      {DumpProcState::kError, DumpProcState::kError, DumpProcState::kError}
  }};

  const auto st = static_cast<std::size_t>(current);
  const auto ev = static_cast<std::size_t>(evt);

  if (st >= N_STATES || ev >= N_EVENTS) {
    return DumpProcState::kError;
  }
  return kTable[st][ev];
}

std::string DumpProperties::GetDumpOpRangeModelName(const std::string &model, const std::string &om_name) const {
  std::string model_name;
  const auto model_iter = model_dump_op_ranges_map_.find(model);
  const auto om_name_iter = model_dump_op_ranges_map_.find(om_name);
  if (om_name_iter != model_dump_op_ranges_map_.end()){
    model_name = om_name;
  } else if ((model_iter != model_dump_op_ranges_map_.end())) {
    model_name = model;
  } else {
    // do nothing
  }
  return model_name;
}

size_t DumpProperties::GetDumpOpRangeSize(const std::string &model, const std::string &om_name) const {
  std::string model_name = GetDumpOpRangeModelName(model, om_name);
  if (model_name.empty()) {
    GELOGD("Model[%s] om name[%s] no config opname range.", model.c_str(), om_name.c_str());
    return 0U;
  }

  const auto range_iter = model_dump_op_ranges_map_.find(model_name);
  GELOGD("Model name[%s] model[%s] om name[%s] opname range size[%zu].",
    model_name.c_str(), model.c_str(), om_name.c_str(), range_iter->second.size());
  return range_iter->second.size();
}

DumpProcEvent DumpProperties::GetOpProcEvent(const std::string &op_name,
                                             const std::pair<std::string, std::string> &range) const {
  if (op_name == range.first) {
    return DumpProcEvent::kRangeStart;
  } else if (op_name == range.second) {
    return DumpProcEvent::kRangeEnd;
  } else {
    // do nothing
  }

  return DumpProcEvent::kOutOfRange;
}

Status DumpProperties::SetDumpFsmState(const std::string &model,
                                       const std::string &om_name,
                                       const std::string &op_name,
                                       std::vector<DumpProcState> &dump_fsm_state,
                                       std::unordered_set<std::string> &dump_op_in_range,
                                       const bool is_update_dump_op_range) const {
  std::string model_name = GetDumpOpRangeModelName(model, om_name);
  if (model_name.empty()) {
    return ge::SUCCESS;
  }

  const auto range_iter = model_dump_op_ranges_map_.find(model_name);
  if (range_iter->second.size() != dump_fsm_state.size()) {
    GELOGE(FAILED, "Model name[%s], opname range size[%u], dump fsm state size[%u] is not equal.",
    model_name.c_str(), range_iter->second.size(), dump_fsm_state.size());
    return FAILED;
  }

  for (size_t index = 0U; index < range_iter->second.size(); index++) {
    const DumpProcEvent event = GetOpProcEvent(op_name, range_iter->second[index]);
    const DumpProcState cur_state = dump_fsm_state[index];
    if (cur_state == DumpProcState::kError) { // kError之后need dump op的范围不会再增加
      continue;
    }
    DumpProcState next_state = DumpProcFsmTransition(cur_state, event);
    if (next_state == DumpProcState::kStart || next_state == DumpProcState::kStop) {
      GELOGD("Model name[%s] model[%s] om name[%s] op[%s] cur state[%d] event[%d] next state[%d] "
        "opname range id[%zu] need dump[%d].", model_name.c_str(), model.c_str(), om_name.c_str(), op_name.c_str(),
        static_cast<int32_t>(cur_state), static_cast<int32_t>(event),
        static_cast<int32_t>(next_state), index, is_update_dump_op_range);
      if (is_update_dump_op_range) {
        (void)dump_op_in_range.insert(op_name);
      }
    } else if (next_state == DumpProcState::kError) {
      GELOGW("Model name[%s] opname range[%s, %s] range id[%zu] is invalid.",
        model_name.c_str(), range_iter->second[index].first.c_str(),
        range_iter->second[index].second.c_str(), index);
    } else {
      GELOGD(
        "Model name[%s] model[%s] om name[%s] op[%s] cur state[%d] event[%d] "
        "next state[%d] opname range id[%zu] no need dump.",
        model_name.c_str(), model.c_str(), om_name.c_str(), op_name.c_str(),
        static_cast<int32_t>(cur_state), static_cast<int32_t>(event), static_cast<int32_t>(next_state), index);
    }

    if ((next_state == DumpProcState::kStart) &&
      (range_iter->second[index].first == range_iter->second[index].second)) {
      next_state = DumpProcState::kStop;
      GELOGD(
        "Model name[%s] model[%s] om name[%s] op[%s] next state[%d] opname range id[%zu] range begin[%s] "
        "is same with range end[%s].", model_name.c_str(), model.c_str(), om_name.c_str(), op_name.c_str(),
        static_cast<int32_t>(next_state), index, range_iter->second[index].first.c_str(),
        range_iter->second[index].second.c_str());
    }

    dump_fsm_state[index] = next_state;
  }

  return ge::SUCCESS;
}
}  // namespace ge
