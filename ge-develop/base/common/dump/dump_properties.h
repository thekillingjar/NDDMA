/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_DUMP_DUMP_PROPERTIES_H_
#define GE_COMMON_DUMP_DUMP_PROPERTIES_H_

#include <map>
#include <set>
#include <unordered_set>
#include <string>

#include "ge/ge_api_types.h"

namespace ge {
struct OpBlacklist {
  std::set<uint32_t> input_indices;
  std::set<uint32_t> output_indices;
};

struct ModelOpBlacklist {
  //<op_type, <input_index, output_index>>
  std::map<std::string, OpBlacklist> dump_optype_blacklist;
  //<op_name, <input_index, output_index>>
  std::map<std::string, OpBlacklist> dump_opname_blacklist;
};

enum class StepCheckErrorCode : uint32_t {
  kNoError,
  kTooManySets,
  kIncorrectFormat,
  kReverseStep
};

enum class DumpProcState : int32_t {
  kInit = 0,
  kStart = 1,
  kStop = 2,
  kError = 3,
  kEnd = 4
};

enum class DumpProcEvent : int32_t {
  kRangeStart = 0,
  kRangeEnd = 1,
  kOutOfRange = 2,
  kEnd = 3
};

class DumpProperties {
 public:
  DumpProperties() = default;

  ~DumpProperties() = default;

  DumpProperties(const DumpProperties &dump);

  DumpProperties &operator=(const DumpProperties &dump) & ;

  Status InitByOptions();

  void AddPropertyValue(const std::string &model, const std::set<std::string> &layers);

  void DeletePropertyValue(const std::string &model);

  void ClearDumpPropertyValue();

  void ClearDumpInfo();

  std::set<std::string> GetAllDumpModel() const;

  std::set<std::string> GetPropertyValue(const std::string &model) const;

  bool IsLayerOpOnWatcherMode(const std::string &op_name) const;

  bool IsWatcherNode(const std::string &op_name) const;

  bool IsDumpWatcherModelEnable() const;

  bool IsLayerNeedDump(const std::string &model, const std::string &om_name, const std::string &op_name) const;

  bool IsDumpLayerOpModelEnable() const;

  void SetDumpPath(const std::string &path);

  const std::string &GetDumpPath() const;

  void SetDumpStep(const std::string &step);

  const std::string &GetDumpStep() const;

  void SetDumpWorkerId(const std::string &worker_id);

  const std::string &GetDumpWorkerId() const;

  void SetDumpMode(const std::string &mode);

  const std::string &GetDumpMode() const;

  void SetDumpData(const std::string &data);

  const std::string &GetDumpData() const;

  const std::map<std::string, ModelOpBlacklist> &GetModelDumpBlacklistMap() const;
  void SetModelDumpBlacklistMap(const std::map<std::string, ModelOpBlacklist>& new_map);

  void SetModelBlacklist(const std::string& model_name, const ModelOpBlacklist& blacklist);

  const std::string &GetEnableDump() const {
    return enable_dump_;
  }

  const std::string &GetEnableDumpDebug() const {
    return enable_dump_debug_;
  }

  void SetDumpStatus(const std::string &status);

  void InitInferOpDebug();

  bool IsInferOpDebug() const {
    return is_infer_op_debug_;
  }

  void SetDumpOpSwitch(const std::string &dump_op_switch);

  bool IsOpDebugOpen() const {
    return is_train_op_debug_ || is_infer_op_debug_;
  }

  void ClearOpDebugFlag() {
    is_train_op_debug_ = false;
    is_infer_op_debug_ = false;
  }

  bool IsDumpOpen() const;

  bool IsSingleOpNeedDump() const;

  bool IsInputInOpNameBlacklist(const std::string &model_name, const std::string &op_name, const uint32_t index) const;
  bool IsOutputInOpNameBlacklist(const std::string &model_name, const std::string &op_name, const uint32_t index) const;
  bool IsInputInOpTypeBlacklist(const std::string &model_name, const std::string &op_type, const uint32_t index) const;
  bool IsOutputInOpTypeBlacklist(const std::string &model_name, const std::string &op_type, const uint32_t index) const;

  void SetOpDebugMode(const uint32_t op_debug_mode);

  uint32_t GetOpDebugMode() const { return op_debug_mode_; }

  void SetOpDumpRange(const std::string &model, const std::vector<std::pair<std::string, std::string>> &op_ranges);

  size_t GetDumpOpRangeSize(const std::string &model, const std::string &om_name) const;

  Status SetDumpFsmState(const std::string &model, const std::string &om_name, const std::string &op_name,
                         std::vector<DumpProcState> &dump_fsm_state,
                         std::unordered_set<std::string> &dump_op_in_range,
                         const bool is_update_dump_op_range = true) const;
 private:
  void CopyFrom(const DumpProperties &other);

  void SetDumpList(const std::string &layers);

  Status SetDumpDebugOptions();

  Status SetDumpOptions();

  Status CheckDumpStep(const std::string &dump_step) const;

  StepCheckErrorCode CheckDumpStepInner(const std::string &dump_step) const;

  Status CheckDumpMode(const std::string &dump_mode) const;

  Status CheckDumpPathValid(const std::string &input) const;

  Status CheckEnableDump(const std::string &input) const;

  bool IsNeedDump(const std::string &model, const std::string &om_name, const std::string &op_name) const;

  std::string GetDumpOpRangeModelName(const std::string &model, const std::string &om_name) const;

  static DumpProcState DumpProcFsmTransition(DumpProcState current, DumpProcEvent evt);

  DumpProcEvent GetOpProcEvent(const std::string &op_name, const std::pair<std::string, std::string> &range) const;

  std::string enable_dump_;
  std::string enable_dump_debug_;

  std::string dump_path_;
  std::string dump_step_;
  std::string dump_mode_;
  std::string dump_status_;
  std::string dump_data_;
  std::string dump_op_switch_;
  std::string dump_worker_id_;
  std::map<std::string, ModelOpBlacklist> model_dump_blacklist_map_;

  std::map<std::string, std::set<std::string>> model_dump_properties_map_;

  bool is_train_op_debug_ = false;
  bool is_infer_op_debug_ = false;
  uint32_t op_debug_mode_ = 0U;
  std::map<std::string, std::vector<std::pair<std::string, std::string>>> model_dump_op_ranges_map_;
};
}

#endif // GE_COMMON_DUMP_DUMP_PROPERTIES_H_
