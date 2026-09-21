/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <memory.h>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include "op_log.h"
#include "AddLayerNorm_tiling_data.h"
#include "runtime2_util.h"
#define Max(a, b) ((double)(a) > (double)(b) ? (a) : (b))
#define Min(a, b) ((double)(a) < (double)(b) ? (a) : (b))
#define Log(a) (log((double)(a)))
#define MAX_SOLUTION 50
#define OP_NAME "AddLayerNorm"

namespace {
enum DurationType {
  TILING_FUNC_DURATION_TOTAL = 0,
  TILING_FUNC_DURATION_MAX,
};

struct DurationDef {
  std::string name;
};

DurationDef g_duration_def[TILING_FUNC_DURATION_MAX] = {
  {"TILING_FUNC_DURATION_TOTAL"},
};

class Duration {
 public:
  Duration(const std::string &name): name_(name) {}

  void Begin() {
    call_start_ = Now();
  }

  void End() {
    auto now = Now();
    uint64_t duration = now - call_start_;
    total_count_++;
    total_time_ += duration;
    if (now > max_time_) max_time_ = now;
    if (now < min_time_) min_time_ = now;
  }

  void Print() {
    if (total_count_ == 0ULL) return;
    OP_LOGI(OP_NAME, "Duration record: name[%s], total_count[%lu], total_time[%lu], max_time[%lu], min_time[%lu], average_time[%lu].",
      name_.c_str(), total_count_, total_time_, max_time_, min_time_,
      static_cast<uint64_t>(total_count_ / total_count_));
  } 

  void Clear() {
    total_count_ = 0ULL;
    total_time_ = 0ULL;
    max_time_ = 0ULL;
    min_time_ = UINT64_MAX;
    call_start_ = 0ULL;
  }

private:
  uint64_t Now() {
    static auto zero = std::chrono::system_clock::now();
    auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now() - zero).count();
    return static_cast<uint64_t>(now);
  }

  std::string name_;
  uint64_t total_count_ = 0ULL;
  uint64_t total_time_ = 0ULL;
  uint64_t max_time_ = 0ULL;
  uint64_t min_time_ = UINT64_MAX;
  uint64_t call_start_ = 0ULL;
};

struct DurationInfo {
  std::unique_ptr<Duration> stat;
};

class DurationManager {
public:
  static DurationManager &GetInstance() {
    static DurationManager ins;
    return ins;
  }

  DurationManager() {
    for (uint32_t index = 0U; index < static_cast<uint32_t>(TILING_FUNC_DURATION_MAX); index++) {
      AddDuration(index, g_duration_def[index].name);
    }
  }
  
  void AddDuration(const uint32_t type, const std::string &name) {
    duration_infos_[type].stat = std::unique_ptr<Duration>(new(std::nothrow) Duration(name));
    if (duration_infos_[type].stat == nullptr) {
      OP_LOGW(OP_NAME, "Create Duration failed.");
    }
  }

  void Begin(const DurationType type) {
    const auto &stat = duration_infos_[type].stat;
    if (stat == nullptr) {
      return;
    }
    stat->Begin();
  }

  void End(const DurationType type) {
    const auto &stat = duration_infos_[type].stat;
    if (stat == nullptr) {
      return;
    }
    stat->End();
  }
  void Print() {
    for (int32_t index = 0; index < static_cast<int32_t>(DurationType::TILING_FUNC_DURATION_MAX); index++) {
      const auto &stat = duration_infos_[index].stat;
      if (stat != nullptr) {
        stat->Print();
      }
    }
  }
  void Clear() {
    for (int32_t index = 0; index < static_cast<int32_t>(DurationType::TILING_FUNC_DURATION_MAX); index++) {
      const auto &stat = duration_infos_[index].stat;
      if (stat != nullptr) {
        stat->Clear();
      }
    }
  }
private:
  DurationInfo duration_infos_[TILING_FUNC_DURATION_MAX];
};

static inline void DurationBegin(const DurationType type) {
  DurationManager::GetInstance().Begin(type);
}

static inline void DurationEnd(const DurationType type) {
  DurationManager::GetInstance().End(type);
}

class DurationGuard {
public:
  DurationGuard(const DurationType type) : type_(type)
  {
    DurationBegin(type);
  }

  ~DurationGuard() {
    DurationEnd(type_);
  }
private:
  DurationType type_;
};

#define DURATION_GUARD(type) DurationGuard g_duration##__COUNTER__(type);
} // namespace

namespace optiling {
using namespace std;
inline bool IsEqual(double a, double b)
{
    const double epsilon = 0.001;
    double abs = (a > b) ? (a - b) : (b - a);
    return abs < epsilon;
}
template<typename T>
inline T ceiling(T a)
{
    T value = static_cast<T>(static_cast<int64_t>(a));
    return (IsEqual(value, a)) ? value : (value + 1);
}

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    int64_t res = a / b;
    return (res * b == a) ? res : (res + 1);
}
class TilingCaseImpl {
 public:
  TilingCaseImpl(uint32_t corenum) : corenum_(corenum) {}
  virtual ~TilingCaseImpl() = default;
  bool GetTiling(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    if (CheckContext(context) != true) {
      OP_LOGW(OP_NAME, "Check context failed.");
      return false;
    }
    if (!GetShapeAttrsInfo(tiling_data, context)) {
      OP_LOGW(OP_NAME, "Failed to get shape attrs.");
      return false;
    }
    if (!CheckIsCapable(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to check capable.");
      return false;
    }
    if (!DoTiling(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to do tiling.");
      return false;
    }
    DoApiTiling(tiling_data);
    GetWorkSpaceSize(tiling_data);
    ExtraTilingData(tiling_data, context);
    TilingSummary(tiling_data);
    return true;
  }
  virtual double GetPerf(graph_normalTilingData &tiling_data) { return 0.0; }
 protected:
  bool TilingVarsNumCheck(gert::TilingContext *context) {
    if (context->GetComputeNodeInfo()->GetInputsNum() != input_num_) {
      OP_LOGW(OP_NAME, "Expect input num is [%lu], current value is [%lu], invalid input num.", input_num_, context->GetComputeNodeInfo()->GetInputsNum());
      return false;
    }
    OP_LOGD(OP_NAME, "TilingVarsNumCheck success.");
    return true;
  }
  bool TilingVarsDtypeCheck(gert::TilingContext *context) {
    for (uint32_t i = 0; i < input_num_; i++) {
      if (static_cast<uint32_t>(context->GetInputTensor(i)->GetDataType()) != input_dtype_[i]) {
        OP_LOGW(OP_NAME, "expect input_dtype_[%u] = [%u], current value is [%u], invalid input dtype.", i, input_dtype_[i], context->GetInputTensor(i)->GetDataType());
        return false;
      }
    }
    OP_LOGD(OP_NAME, "TilingVarsDtypeCheck success.");
    return true;
  }
  bool TilingVarsFormatCheck(gert::TilingContext *context) {
    for (uint32_t i = 0; i < input_num_; i++) {
      if (static_cast<uint32_t>(context->GetInputTensor(i)->GetStorageFormat()) != input_format_[i]) {
        OP_LOGW(OP_NAME, "expect input_format_[%u] = [%u], current value is [%u], invalid input format.", i, input_format_[i], context->GetInputTensor(i)->GetStorageFormat());
        return false;
      }
    }
    OP_LOGD(OP_NAME, "TilingVarsFormatCheck success.");
    return true;
  }
  bool TilingVarsShapeDimCheck(gert::TilingContext *context) {
    uint32_t input_size;
    for (uint32_t i = 0; i < input_num_; i++) {
      input_size = context->GetInputTensor(i)->GetStorageShape().GetDimNum();
      if (max_dim_[i] != 0 && input_size > max_dim_[i]) {
        OP_LOGW(OP_NAME, "expect input_size[%u] <= [%u], current value is [%u], invalid input size.", i, max_dim_[i], input_size);
        return false;
      }
      if (min_dim_[i] != 0 && input_size < min_dim_[i]) {
        OP_LOGW(OP_NAME, "expect input_size[%u] >= [%u], current value is [%u], invalid input size.", i, min_dim_[i], input_size);
        return false;
      }
    }
    OP_LOGD(OP_NAME, "TilingVarsShapeDimCheck success.");
    return true;
  }
  bool TilingAttrCheck(gert::TilingContext *context) {
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
      OP_LOGE(OP_NAME, "Pointer context->GetAttrs() is null.");
      return false;
    }
    auto att_ptr_1 = attrs->GetAttrPointer<int32_t>(1U);
    if (att_ptr_1 == nullptr) {
      OP_LOGW(OP_NAME, "Attr for att_ptr is null.");
      return false;
    }
    int64_t att_1 = static_cast<int64_t>(*att_ptr_1);
    if (check_att_[0] && (att_1 < min_att_[0] || att_1 > max_att_[0])) {
      return false;
    }
    OP_LOGD(OP_NAME, "TilingAttrCheck success.");
    return true;
  }
  virtual bool TilingVarsShapeCheck(gert::TilingContext *context) = 0;
  virtual bool TilingVarsCoverCheck(gert::TilingContext *context) {
    return true;
  }
  bool TilingInputVarsCheck(gert::TilingContext *context) {
    if (!TilingVarsNumCheck(context)) {
      OP_LOGW(OP_NAME, "TilingVarsNumCheck failed.");
      return false;
    }
    if (!TilingVarsDtypeCheck(context)) {
      OP_LOGW(OP_NAME, "TilingVarsDtypeCheck failed.");
      return false;
    }
    if (!TilingVarsFormatCheck(context)) {
      OP_LOGW(OP_NAME, "TilingVarsFormatCheck failed.");
      return false;
    }
    if (!TilingVarsShapeDimCheck(context)) {
      OP_LOGW(OP_NAME, "TilingVarsShapeDimCheck failed.");
      return false;
    }
    OP_LOGD(OP_NAME, "TilingInputVarsCheck success.");
    return true;
  }
  bool CheckContext(gert::TilingContext *context) {
    if (!TilingInputVarsCheck(context)) {
      return false;
    }
    if (!TilingAttrCheck(context)) {
      return false;
    }
    return true;
  }
  virtual bool GetShapeAttrsInfo(graph_normalTilingData &tiling_data, gert::TilingContext *context) = 0;
  virtual bool CheckIsCapable(graph_normalTilingData &tiling_data) = 0;
  virtual bool DoTiling(graph_normalTilingData &tiling_data) = 0;
  virtual void DoApiTiling(graph_normalTilingData &tiling_data) {}
  virtual void GetWorkSpaceSize(graph_normalTilingData& tiling_data) {}
  virtual void ExtraTilingData(graph_normalTilingData &tiling_data, gert::TilingContext *context) {}
  virtual void TilingSummary(graph_normalTilingData &tiling_data) = 0;
  uint32_t corenum_;
  uint32_t input_num_;
  uint32_t input_dtype_[5];
  uint32_t input_format_[5];
  uint32_t max_dim_[5];
  uint32_t min_dim_[5];
  bool check_att_[1];
  int64_t max_att_[1];
  int64_t min_att_[1];
  bool* bool_space_{nullptr};
};
using TilingCaseImplPtr = std::shared_ptr<TilingCaseImpl>;

/*
ConstraintType:约束类型
  LOCAL_BUFFER:仅与内存占用相关的约束, 如s1t * s2t < UB
  LB_MIXED:与内存占用相关的约束
  MC_MIXED:纯多核相关的约束
*/
enum class ConstraintType {
  LOCAL_BUFFER = 0,
  LB_MIXED = 1,
  MC_MIXED = 2,
};

struct Variable;
struct TilingVariable;
struct Constraint;
using ConsEvalFuncPtr = int64_t (*)(TilingVariable **rel_tiling_vars, Variable **rel_input_shapes, int64_t rel_hw_spec);
using GetUpperBoundFuncPtr = int64_t (*)(Variable **rel_ori_dims);

struct Variable {
  int64_t value = -1;
};

struct Constraint {
  int64_t rel_hw_spec = 0;
  uint32_t rel_tiling_vars_size = 0u;
  uint32_t rel_in_shapes_size = 0u;
  TilingVariable **rel_tiling_vars = nullptr;
  Variable **rel_in_shapes = nullptr;
  ConsEvalFuncPtr eval = nullptr;
  ConstraintType type;
};

struct TilingVariable : public Variable {
  int64_t align = 1;
  uint32_t rel_cons_size = 0u;
  uint32_t upper_bound_vars_size = 0u;
  bool notail = false;
  TilingVariable *notail_var = nullptr;
  Variable **upper_bound_vars = nullptr;
  Constraint **rel_cons = nullptr;
  GetUpperBoundFuncPtr upper_bound = nullptr;
  bool SetValue(int64_t val) {
    if (val <= 0) {
      return false;
    }
    value = val;
    return true;
  }
};

struct AxesReorderSolverInput {
  uint32_t core_num = 0u;
  uint32_t input_vars_size = 0u;
  uint32_t tiling_vars_size = 0u;
  uint32_t pure_mc_vars_size = 0u;
  uint32_t local_buffer_vars_size = 0u;
  uint32_t all_cons_size = 0u;
  Variable **input_vars = nullptr;
  TilingVariable **tiling_vars = nullptr;
  TilingVariable **pure_mc_vars = nullptr;
  TilingVariable **local_buffer_vars = nullptr;
  Constraint **all_cons = nullptr;
};

class AxesReorderSolver {
public:
  explicit AxesReorderSolver(const AxesReorderSolverInput &input) : input_(input) {}
  ~AxesReorderSolver() = default;
  bool Run();
protected:
  virtual bool CalUsedCoreNum(int32_t &used_core_num, bool &load_balance) = 0;
  AxesReorderSolverInput input_;
private:
  bool TuneNotailVar(TilingVariable *var);
  bool SatisfyCons(ConstraintType cons_type);
  bool SatisfyCons(TilingVariable *var, ConstraintType cons_type);
  bool SatisfyMCCons();
  bool InitLocalBufferVars();
  bool InitMulticoreVars();
  bool MulticoreTiling();
  bool LocalBufTiling();
};

bool AxesReorderSolver::InitLocalBufferVars() {
  for (uint32_t i = 0u; i < input_.local_buffer_vars_size; ++i) {
    auto &var = input_.local_buffer_vars[i];
    if (!var->SetValue(var->align)) {
      OP_LOGW(OP_NAME, "Failed to init local buffer value.");
      return false;
    }
  }
  return true;
}

bool AxesReorderSolver::InitMulticoreVars() {
  for (uint32_t i = input_.pure_mc_vars_size - 1; i >= 0; --i) {
    auto &var = input_.pure_mc_vars[i];
    auto upper_bound_val = var->upper_bound(var->upper_bound_vars);
    if (upper_bound_val == -1) {
      return false;
    }
    upper_bound_val = CeilDiv(upper_bound_val, var->align) * var->align;
    if (!var->SetValue(upper_bound_val)) {
      OP_LOGW(OP_NAME, "Failed to init multicore value.");
      return false;
    }
  }
  return true;
}
bool AxesReorderSolver::SatisfyCons(ConstraintType cons_type) {
  for (uint32_t i = 0u; i < input_.all_cons_size; i++) {
    auto &cons = input_.all_cons[i];
    if (cons->type != cons_type) {
      continue;
    }
    if (cons->eval(cons->rel_tiling_vars, cons->rel_in_shapes, cons->rel_hw_spec) > 0) {
      return false;
    }
  }
  return true;
}

bool AxesReorderSolver::SatisfyCons(TilingVariable *var, ConstraintType cons_type) {
  for (uint32_t j = 0; j < var->rel_cons_size; ++j) {
    auto &cons = var->rel_cons[j];
    if (cons->type != cons_type) {
      continue;
    }
    if (cons->eval(cons->rel_tiling_vars, cons->rel_in_shapes, cons->rel_hw_spec) > 0) {
      return false;
    }
  }
  return true;
}

bool AxesReorderSolver::SatisfyMCCons() {
  int32_t used_core_num = input_.core_num + 1;
  bool load_balance = false;
  CalUsedCoreNum(used_core_num, load_balance);
  return used_core_num <= input_.core_num; 
}

bool AxesReorderSolver::TuneNotailVar(TilingVariable *var) {
  if (!var->notail) {
    return true;
  }
  if (var->notail_var->value % var->value == 0) {
    return true;
  }
  for (; var->value > 0; var->value -= var->align) {
    if (var->notail_var->value % var->value != 0) {
      continue;
    }
    break;
  }
  return var->value != 0;
}

bool AxesReorderSolver::MulticoreTiling() {
  if (!InitMulticoreVars()) {
    OP_LOGE(OP_NAME, "multicore tiling failed");
    return false;
  }
  if (!SatisfyMCCons()) {
    OP_LOGE(OP_NAME, "Multicore Tiling Calculation failed in the first check.");
    return false;
  }
  for (int32_t i=input_.pure_mc_vars_size - 1; i >= 0; --i) {
    auto &var = input_.pure_mc_vars[i];
    int64_t boundary = var->align;
    auto init_val = var->value;
    var->SetValue(boundary);
    if (!SatisfyMCCons()) {
      var->SetValue(init_val);
      int64_t last_boundary = -1;
      int64_t last_val = -1;
      while (!(last_boundary == boundary && last_val == var->value)) {
        last_boundary = boundary;
        last_val = var->value;
        var->value = CeilDiv((boundary + var->value) / 2, var->align) * var->align;
        if (!SatisfyMCCons()) {
          boundary = var->value;
          var->value = last_val;
        }
      }
    }
    while (!SatisfyCons(var, ConstraintType::MC_MIXED) && var->value != init_val) {
      var->value += var->align;
    }
  }
  if (!SatisfyMCCons() || !SatisfyCons(ConstraintType::MC_MIXED)) {
    OP_LOGW(OP_NAME, "Multicore Tiling Calculation failed in the final check.");
    return false;
  }
  return true;
}
bool AxesReorderSolver::LocalBufTiling() {
  if (!InitLocalBufferVars()) {
    OP_LOGE(OP_NAME, "init local buffer failed");
    return false;
  }
  if (!InitMulticoreVars()) {
    OP_LOGE(OP_NAME, "multicore tiling failed");
    return false;
  }
  if (!SatisfyCons(ConstraintType::LOCAL_BUFFER)) {
    OP_LOGE(OP_NAME, "local buffer tiling failed in the initial check");
    return false;
  }
  for (uint32_t i = 0u; i < input_.local_buffer_vars_size; ++i) {
    auto &var = input_.local_buffer_vars[i];
    auto upper_bound = var->upper_bound(var->upper_bound_vars);
    int64_t boundary = CeilDiv(upper_bound, var->align) * var->align;
    int64_t init_val = var->value;
    var->SetValue(boundary);
    if (!SatisfyCons(var, ConstraintType::LOCAL_BUFFER)) {
      var->SetValue(init_val);
      int64_t last_boundary = -1;
      int64_t last_val = -1;
      while (!(last_boundary == boundary && last_val == var->value)) {
        last_boundary = boundary;
        last_val = var->value;
        var->value = CeilDiv((boundary + var->value) / 2, var->align) * var->align;
        if (!SatisfyCons(var, ConstraintType::LOCAL_BUFFER)) {
          boundary = var->value;
          var->value = last_val; 
        }
      }
    }
    if (!TuneNotailVar(var)) {
      return false;
    }
    while (!SatisfyCons(var, ConstraintType::LB_MIXED) && var->value != var->align) {
      var->value -= var->align;
      if (!TuneNotailVar(var)) {
        return false;
      }
    }
  }
  if (!SatisfyCons(ConstraintType::LOCAL_BUFFER) || !SatisfyCons(ConstraintType::LB_MIXED)) {
    OP_LOGW(OP_NAME, "Local Tiling Calculation failed in the final check.");
    return false;
  }
  return true;
}
bool AxesReorderSolver::Run() {
  if (!LocalBufTiling()) {
    OP_LOGE(OP_NAME, "local buffer tiling failed");
    return false;
  }
  OP_LOGI(OP_NAME, "local buffer tiling success");
  if (!MulticoreTiling()) {
    OP_LOGE(OP_NAME, "multicore tiling failed");
    return false;
  }
  OP_LOGI(OP_NAME, "multicore tiling success");
  return true;
}

bool GetPlatformInfo(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
  auto platformInfoPtr = context->GetPlatformInfo();
  if (platformInfoPtr == nullptr) {
    OP_LOGE(OP_NAME, "Pointer platformInfoPtr is null.");
    return false;
  }
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
  auto aivNum = ascendcPlatform.GetCoreNumAiv();
  auto aicNum = ascendcPlatform.GetCoreNumAic();
  uint64_t ub_size;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  if ((aivNum == 0) || (aicNum == 0) || (ub_size == 0)) {
    OP_LOGE(OP_NAME, "Get incorrect platform value.");
    return false;
  } 
  OP_LOGD(OP_NAME, "PlatformInfo is valid.");
  tiling_data.set_block_dim(ascendcPlatform.GetCoreNumAiv());
  OP_LOGD(OP_NAME, "Set block dim to %d.", tiling_data.get_block_dim());
  tiling_data.set_ub_size(ub_size);
  OP_LOGD(OP_NAME, "Set ub_size to %d.", tiling_data.get_ub_size());

  return true;
}

class AxesReorderSolvercase1101 : public AxesReorderSolver {
 public:
  explicit AxesReorderSolvercase1101(const AxesReorderSolverInput input) : AxesReorderSolver(input) {}
  ~AxesReorderSolvercase1101() = default;
  bool CalUsedCoreNum(int32_t &used_core_num, bool &load_balance) override;
};

bool AxesReorderSolvercase1101::CalUsedCoreNum(int32_t &used_core_num, bool &load_balance) {
  double A = static_cast<double>(input_.input_vars[0]->value);
  double nbo_size = static_cast<double>(input_.pure_mc_vars[0]->value);
  double used_core_num_fp = Max(0, ceiling((A / (nbo_size))));
  load_balance = IsEqual(static_cast<double>(static_cast<int64_t>(used_core_num_fp)), used_core_num_fp);
  used_core_num = ceiling(used_core_num_fp);
  return true;
}

class TilingCase1101Impl : public TilingCaseImpl {
 public:
  TilingCase1101Impl(uint32_t corenum) : TilingCaseImpl(corenum) {

   input_num_ = 5;

   input_dtype_[0] = 1;
   input_dtype_[1] = 1;
   input_dtype_[2] = 1;
   input_dtype_[3] = 1;
   input_dtype_[4] = 1;

   input_format_[0] = 2;
   input_format_[1] = 2;
   input_format_[2] = 2;
   input_format_[3] = 2;
   input_format_[4] = 2;

   max_dim_[0] = 0;
   min_dim_[0] = 2;
   max_dim_[1] = 0;
   min_dim_[1] = 2;
   max_dim_[2] = 0;
   min_dim_[2] = 1;
   max_dim_[3] = 0;
   min_dim_[3] = 1;
   max_dim_[4] = 0;
   min_dim_[4] = 2;

   max_att_[0] = 1;
   min_att_[0] = 1;
   check_att_[0] = true;

  }
 protected:
  bool TilingVarsShapeCheck(gert::TilingContext *context) override {
    int64_t cur_size;
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint64_t input1_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    uint64_t input2_size = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    uint64_t input3_size = context->GetInputShape(3)->GetStorageShape().GetDimNum();
    uint64_t input4_size = context->GetInputShape(4)->GetStorageShape().GetDimNum();

    int64_t A_size = 1;
    cur_size = 1;
    for (size_t i = 0; i <= input0_size - 2; i++) {
      cur_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    A_size = cur_size;
    cur_size = 1;
    for (size_t i = 0; i <= input1_size - 2; i++) {
      cur_size *= context->GetInputShape(1)->GetStorageShape().GetDim(i);
    }
    if (A_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for A_size from input0 and input1.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input4_size - 2; i++) {
      cur_size *= context->GetInputShape(4)->GetStorageShape().GetDim(i);
    }
    if (A_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for A_size from input0 and input4.");
      return false;
    }
    int64_t R_size = 1;
    cur_size = context->GetInputShape(0)->GetStorageShape().GetDim(input0_size - 1);
    R_size = cur_size;
    cur_size = context->GetInputShape(1)->GetStorageShape().GetDim(input1_size - 1);
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input1.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input2_size - 1; i++) {
      cur_size *= context->GetInputShape(2)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input2.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input3_size - 1; i++) {
      cur_size *= context->GetInputShape(3)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input3.");
      return false;
    }
    cur_size = context->GetInputShape(4)->GetStorageShape().GetDim(input4_size - 1);
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input4.");
      return false;
    }

    OP_LOGD(OP_NAME, "TilingVarsShapeCheck success.");
    return true;
  }

  bool GetShapeAttrsInfo(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    if (!TilingVarsShapeCheck(context)) {
      OP_LOGW(OP_NAME, "TilingVarsShapeCheck failed.");
      return false;
    }
    OP_LOGD(OP_NAME, "Start setting axis size for 1101.");
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();

    uint32_t A_size = 1;
    for (size_t i = 0; i <= input0_size - 2; i++) {
      A_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    tiling_data.set_A(A_size);
    OP_LOGD(OP_NAME, "Initiate A to %d.", tiling_data.get_A());
    uint32_t R_size = context->GetInputShape(0)->GetStorageShape().GetDim(input0_size - 1);
    tiling_data.set_R(R_size);
    OP_LOGD(OP_NAME, "Initiate R to %d.", tiling_data.get_R());

    OP_LOGD(OP_NAME, "End setting axis size for 1101.");
    return true;
  }

  bool CheckIsCapable(graph_normalTilingData &tiling_data) {
    OP_LOGD(OP_NAME, "CheckIsCapable success.");
    return true;
  }

  bool ExecuteAxesReorderSolver(graph_normalTilingData& tiling_data) {
    Variable A;
    A.value = tiling_data.get_A();
    Variable R;
    R.value = tiling_data.get_R();
    Variable BL;
    BL.value = 8;
    TilingVariable nbo_size;
    TilingVariable nio_size;
    int64_t ub_size = tiling_data.get_ub_size();
    Constraint cons0;
    auto cons0Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t nio_size = rel_tiling_vars[0]->value;
      int64_t R = rel_in_shapes[0]->value;
      int64_t value = ((12 * R * nio_size) + (4 * R) + (8 * nio_size) + Max((4 * R * nio_size), (2 * R * nio_size))) - rel_hw_spec;
      return value;
    };
    cons0.rel_tiling_vars = new TilingVariable*[1];
    cons0.rel_tiling_vars_size = 1u;
    cons0.rel_tiling_vars[0] = &nio_size;
    cons0.rel_in_shapes = new Variable*[1];
    cons0.rel_in_shapes_size = 1u;
    cons0.rel_in_shapes[0] = &R;
    cons0.rel_hw_spec = ub_size;
    cons0.type = ConstraintType::LOCAL_BUFFER;
    cons0.eval = cons0Eval;
    Constraint cons1;
    auto cons1Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t nbo_size = rel_in_shapes[0]->value;
      int64_t A = rel_in_shapes[1]->value;
      int64_t value = (nbo_size - A);
      return value;
    };
    cons1.rel_in_shapes = new Variable*[2];
    cons1.rel_in_shapes_size = 2u;
    cons1.rel_in_shapes[0] = &nbo_size;
    cons1.rel_in_shapes[1] = &A;
    cons1.type = ConstraintType::MC_MIXED;
    cons1.eval = cons1Eval;
    Constraint cons2;
    auto cons2Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t nio_size = rel_tiling_vars[0]->value;
      int64_t nbo_size = rel_in_shapes[0]->value;
      int64_t value = (nio_size - nbo_size);
      return value;
    };
    cons2.rel_tiling_vars = new TilingVariable*[1];
    cons2.rel_tiling_vars_size = 1u;
    cons2.rel_tiling_vars[0] = &nio_size;
    cons2.rel_in_shapes = new Variable*[1];
    cons2.rel_in_shapes_size = 1u;
    cons2.rel_in_shapes[0] = &nbo_size;
    cons2.type = ConstraintType::MC_MIXED;
    cons2.eval = cons2Eval;
    GetUpperBoundFuncPtr nbo_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t A = parent_vars[0]->value;
      if (A == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= A;
      return upper_bound;
    };
    nbo_size.upper_bound = nbo_size_upper_bound;
    nbo_size.upper_bound_vars = new Variable * [1];
    nbo_size.upper_bound_vars_size = 1u;
    nbo_size.upper_bound_vars[0] = &A;
    nbo_size.rel_cons = new Constraint*[2];
    nbo_size.rel_cons_size = 2u;
    nbo_size.rel_cons[0] = &cons1;
    nbo_size.rel_cons[1] = &cons2;
    nio_size.align = 1;
    GetUpperBoundFuncPtr nio_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t nbo_size = parent_vars[0]->value;
      if (nbo_size == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= nbo_size;
      return upper_bound;
    };
    nio_size.upper_bound = nio_size_upper_bound;
    nio_size.upper_bound_vars = new Variable * [1];
    nio_size.upper_bound_vars_size = 1u;
    nio_size.upper_bound_vars[0] = &nbo_size;
    nio_size.rel_cons = new Constraint*[2];
    nio_size.rel_cons_size = 2u;
    nio_size.rel_cons[0] = &cons0;
    nio_size.rel_cons[1] = &cons2;
    AxesReorderSolverInput input;
    input.input_vars = new Variable*[3];
    input.input_vars_size = 3u;
    input.input_vars[0] = &A;
    input.input_vars[1] = &R;
    input.input_vars[2] = &BL;
    input.tiling_vars = new TilingVariable*[2];
    input.tiling_vars_size = 2u;
    input.tiling_vars[0] = &nbo_size;
    input.tiling_vars[1] = &nio_size;
    input.all_cons_size = 3u;
    input.all_cons = new Constraint*[3];
    input.all_cons[0] = &cons0;
    input.all_cons[1] = &cons1;
    input.all_cons[2] = &cons2;
    input.pure_mc_vars_size = 1u;
    input.pure_mc_vars = new TilingVariable*[1];
    input.pure_mc_vars[0] = &nbo_size;
    input.local_buffer_vars_size = 1u;
    input.local_buffer_vars = new TilingVariable*[1];
    input.local_buffer_vars[0] = &nio_size;
    input.core_num = corenum_;
    AxesReorderSolvercase1101* solver = new AxesReorderSolvercase1101(input);
    if (!solver->Run()) {
        return false;
    }
    tiling_data.set_nbo_size(input.pure_mc_vars[0]->value);
    tiling_data.set_nio_size(input.local_buffer_vars[0]->value);
    return true;
  }

  bool DoTiling(graph_normalTilingData &tiling_data) {
    if (!ExecuteAxesReorderSolver(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to execute axes reorder solver for tilingCaseId case1101.");
      return false;
    }
    OP_LOGD(OP_NAME, "Execute axes reorder solver for tilingCaseId case1101 successfully.");

    return true;
  }

  int Getub_size(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double nio_size = tiling_data.get_nio_size();

    return ((12 * R * nio_size) + (4 * R) + (8 * nio_size) + Max((4 * R * nio_size), (2 * R * nio_size)));
  }

  int Getblock_dim(graph_normalTilingData& tiling_data) {
    double A = tiling_data.get_A();
    double nbo_size = tiling_data.get_nbo_size();

    return Max(0, ceiling((A / (nbo_size))));
  }

  void UpdateGeneralTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_block_dim((((tiling_data.get_A() + tiling_data.get_nbo_size()) - 1) / tiling_data.get_nbo_size()));
  }

  void UpdateAxesTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_A_aligned_size((tiling_data.get_A() - 1) / 8 * 8 + 8);
    tiling_data.set_R_aligned_size((tiling_data.get_R() - 1) / 8 * 8 + 8);
    tiling_data.set_nio_loop_num(((tiling_data.get_nbo_size() + tiling_data.get_nio_size()) - 1) / tiling_data.get_nio_size());
    tiling_data.set_nbo_loop_num(((tiling_data.get_A() + tiling_data.get_nbo_size()) - 1) / tiling_data.get_nbo_size());
    tiling_data.set_nio_tail_size((tiling_data.get_nbo_size() % tiling_data.get_nio_size()) == 0 ? tiling_data.get_nio_size() : (tiling_data.get_nbo_size() % tiling_data.get_nio_size()));
    tiling_data.set_nbo_tail_size((tiling_data.get_A() % tiling_data.get_nbo_size()) == 0 ? tiling_data.get_nbo_size() : (tiling_data.get_A() % tiling_data.get_nbo_size()));
    tiling_data.set_nbo_tail_tile_nio_loop_num(((tiling_data.get_nbo_tail_size() + tiling_data.get_nio_size()) - 1) / tiling_data.get_nio_size());
    tiling_data.set_nbo_tail_tile_nio_tail_size((tiling_data.get_nbo_tail_size() % tiling_data.get_nio_size()) == 0 ? tiling_data.get_nio_size() : (tiling_data.get_nbo_tail_size() % tiling_data.get_nio_size()));
  }

  void SetQ0(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q0((2 * R * nio_size));
  }

  void SetQ1(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q1((2 * R * nio_size));
  }

  void SetQ2(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q2((2 * R * nio_size));
  }

  void SetQ3(graph_normalTilingData &tiling_data) {
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q3((4 * nio_size));
  }

  void SetQ4(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q4((2 * R * nio_size));
  }

  void SetQ5(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q5((4 * R * nio_size));
  }

  void SetQ6(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q6(Max((4 * R * nio_size), (2 * R * nio_size)));
  }

  void SetQ7(graph_normalTilingData &tiling_data) {
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q7((4 * nio_size));
  }

  void SetQ8(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    tiling_data.set_Q8((2 * R));
  }

  void SetQ9(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    tiling_data.set_Q9((2 * R));
  }

  void ComputeOptionParam(graph_normalTilingData &tiling_data) {
    SetQ0(tiling_data);
    SetQ1(tiling_data);
    SetQ2(tiling_data);
    SetQ3(tiling_data);
    SetQ4(tiling_data);
    SetQ5(tiling_data);
    SetQ6(tiling_data);
    SetQ7(tiling_data);
    SetQ8(tiling_data);
    SetQ9(tiling_data);

  }
  void AssignAttAndOutputSize(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    OP_LOGD(OP_NAME, "Start assigning attr and output size for tiling case 1101.");
    auto attrs = context->GetAttrs();
    auto additional_output_ptr = attrs->GetAttrPointer<int32_t>(1U);
    int32_t additional_output = *additional_output_ptr;
    tiling_data.set_additional_output(additional_output);
    tiling_data.set_output0_total_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize());
    tiling_data.set_output0_single_core_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output1_total_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize());
    tiling_data.set_output1_single_core_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output2_total_size(context->GetOutputShape(2)->GetStorageShape().GetShapeSize());
    tiling_data.set_output2_single_core_size(context->GetOutputShape(2)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output3_total_size(context->GetOutputShape(3)->GetStorageShape().GetShapeSize());
    tiling_data.set_output3_single_core_size(context->GetOutputShape(3)->GetStorageShape().GetShapeSize() / corenum_);

    OP_LOGD(OP_NAME, "Set additional_output to %u.", tiling_data.get_additional_output());
    OP_LOGD(OP_NAME, "Set output0_single_core_size to %u.", tiling_data.get_output0_single_core_size());
    OP_LOGD(OP_NAME, "Set output0_total_size to %u.", tiling_data.get_output0_total_size());
    OP_LOGD(OP_NAME, "Set output1_single_core_size to %u.", tiling_data.get_output1_single_core_size());
    OP_LOGD(OP_NAME, "Set output1_total_size to %u.", tiling_data.get_output1_total_size());
    OP_LOGD(OP_NAME, "Set output2_single_core_size to %u.", tiling_data.get_output2_single_core_size());
    OP_LOGD(OP_NAME, "Set output2_total_size to %u.", tiling_data.get_output2_total_size());
    OP_LOGD(OP_NAME, "Set output3_single_core_size to %u.", tiling_data.get_output3_single_core_size());
    OP_LOGD(OP_NAME, "Set output3_total_size to %u.", tiling_data.get_output3_total_size());

    OP_LOGD(OP_NAME, "Assigned attr and output size for tiling case 1101 successfully.");
  }

  void ExtraTilingData(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    OP_LOGD(OP_NAME, "Start executing extra tiling for tilingCaseId 1101.");
		UpdateGeneralTilingData(tiling_data);

    ComputeOptionParam(tiling_data);
		UpdateAxesTilingData(tiling_data);

    AssignAttAndOutputSize(tiling_data, context);
    OP_LOGD(OP_NAME, "Execute extra tiling for tilingCaseId 1101 successfully.");
  }

  void GetWorkSpaceSize(graph_normalTilingData& tiling_data) {
    OP_LOGD(OP_NAME, "Start setting workspace for case 1101.");
    tiling_data.set_workspaceSize(static_cast<uint32_t>(0));
    OP_LOGD(OP_NAME, "Setting workspace to %u for case 1101.", tiling_data.get_workspaceSize());
  }

  void TilingSummary(graph_normalTilingData &tiling_data) {
    OP_LOGI(OP_NAME, "Set nbo_size to %u.", tiling_data.get_nbo_size());
    OP_LOGI(OP_NAME, "Set nio_size to %u.", tiling_data.get_nio_size());
    OP_LOGI(OP_NAME, "The value of ub_size is %d.", Getub_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of block_dim is %d.", Getblock_dim(tiling_data));
  }

};

class AxesReorderSolvercase1102 : public AxesReorderSolver {
 public:
  explicit AxesReorderSolvercase1102(const AxesReorderSolverInput input) : AxesReorderSolver(input) {}
  ~AxesReorderSolvercase1102() = default;
  bool CalUsedCoreNum(int32_t &used_core_num, bool &load_balance) override;
};

bool AxesReorderSolvercase1102::CalUsedCoreNum(int32_t &used_core_num, bool &load_balance) {
  double A = static_cast<double>(input_.input_vars[0]->value);
  double nbo_size = static_cast<double>(input_.pure_mc_vars[0]->value);
  double used_core_num_fp = Max(0, ceiling((A / (nbo_size))));
  load_balance = IsEqual(static_cast<double>(static_cast<int64_t>(used_core_num_fp)), used_core_num_fp);
  used_core_num = ceiling(used_core_num_fp);
  return true;
}

class TilingCase1102Impl : public TilingCaseImpl {
 public:
  TilingCase1102Impl(uint32_t corenum) : TilingCaseImpl(corenum) {

   input_num_ = 5;

   input_dtype_[0] = 1;
   input_dtype_[1] = 1;
   input_dtype_[2] = 1;
   input_dtype_[3] = 1;
   input_dtype_[4] = 1;

   input_format_[0] = 2;
   input_format_[1] = 2;
   input_format_[2] = 2;
   input_format_[3] = 2;
   input_format_[4] = 2;

   max_dim_[0] = 0;
   min_dim_[0] = 2;
   max_dim_[1] = 0;
   min_dim_[1] = 2;
   max_dim_[2] = 0;
   min_dim_[2] = 1;
   max_dim_[3] = 0;
   min_dim_[3] = 1;
   max_dim_[4] = 0;
   min_dim_[4] = 1;

   max_att_[0] = 1;
   min_att_[0] = 1;
   check_att_[0] = true;

  }
 protected:
  bool TilingVarsShapeCheck(gert::TilingContext *context) override {
    int64_t cur_size;
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint64_t input1_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    uint64_t input2_size = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    uint64_t input3_size = context->GetInputShape(3)->GetStorageShape().GetDimNum();
    uint64_t input4_size = context->GetInputShape(4)->GetStorageShape().GetDimNum();

    int64_t A_size = 1;
    cur_size = 1;
    for (size_t i = 0; i <= input0_size - 2; i++) {
      cur_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    A_size = cur_size;
    cur_size = 1;
    for (size_t i = 0; i <= input1_size - 2; i++) {
      cur_size *= context->GetInputShape(1)->GetStorageShape().GetDim(i);
    }
    if (A_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for A_size from input0 and input1.");
      return false;
    }
    int64_t R_size = 1;
    cur_size = context->GetInputShape(0)->GetStorageShape().GetDim(input0_size - 1);
    R_size = cur_size;
    cur_size = context->GetInputShape(1)->GetStorageShape().GetDim(input1_size - 1);
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input1.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input2_size - 1; i++) {
      cur_size *= context->GetInputShape(2)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input2.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input3_size - 1; i++) {
      cur_size *= context->GetInputShape(3)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input3.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input4_size - 1; i++) {
      cur_size *= context->GetInputShape(4)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input4.");
      return false;
    }

    OP_LOGD(OP_NAME, "TilingVarsShapeCheck success.");
    return true;
  }

  bool GetShapeAttrsInfo(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    if (!TilingVarsShapeCheck(context)) {
      OP_LOGW(OP_NAME, "TilingVarsShapeCheck failed.");
      return false;
    }
    OP_LOGD(OP_NAME, "Start setting axis size for 1102.");
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();

    uint32_t A_size = 1;
    for (size_t i = 0; i <= input0_size - 2; i++) {
      A_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    tiling_data.set_A(A_size);
    OP_LOGD(OP_NAME, "Initiate A to %d.", tiling_data.get_A());
    uint32_t R_size = context->GetInputShape(0)->GetStorageShape().GetDim(input0_size - 1);
    tiling_data.set_R(R_size);
    OP_LOGD(OP_NAME, "Initiate R to %d.", tiling_data.get_R());

    OP_LOGD(OP_NAME, "End setting axis size for 1102.");
    return true;
  }

  bool CheckIsCapable(graph_normalTilingData &tiling_data) {
    OP_LOGD(OP_NAME, "CheckIsCapable success.");
    return true;
  }

  bool ExecuteAxesReorderSolver(graph_normalTilingData& tiling_data) {
    Variable A;
    A.value = tiling_data.get_A();
    Variable R;
    R.value = tiling_data.get_R();
    Variable BL;
    BL.value = 8;
    TilingVariable nbo_size;
    TilingVariable nio_size;
    int64_t ub_size = tiling_data.get_ub_size();
    Constraint cons0;
    auto cons0Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t nio_size = rel_tiling_vars[0]->value;
      int64_t R = rel_in_shapes[0]->value;
      int64_t value = ((10 * R * nio_size) + (6 * R) + (8 * nio_size) + Max((4 * R * nio_size), (2 * R * nio_size))) - rel_hw_spec;
      return value;
    };
    cons0.rel_tiling_vars = new TilingVariable*[1];
    cons0.rel_tiling_vars_size = 1u;
    cons0.rel_tiling_vars[0] = &nio_size;
    cons0.rel_in_shapes = new Variable*[1];
    cons0.rel_in_shapes_size = 1u;
    cons0.rel_in_shapes[0] = &R;
    cons0.rel_hw_spec = ub_size;
    cons0.type = ConstraintType::LOCAL_BUFFER;
    cons0.eval = cons0Eval;
    Constraint cons1;
    auto cons1Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t nbo_size = rel_in_shapes[0]->value;
      int64_t A = rel_in_shapes[1]->value;
      int64_t value = (nbo_size - A);
      return value;
    };
    cons1.rel_in_shapes = new Variable*[2];
    cons1.rel_in_shapes_size = 2u;
    cons1.rel_in_shapes[0] = &nbo_size;
    cons1.rel_in_shapes[1] = &A;
    cons1.type = ConstraintType::MC_MIXED;
    cons1.eval = cons1Eval;
    Constraint cons2;
    auto cons2Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t nio_size = rel_tiling_vars[0]->value;
      int64_t nbo_size = rel_in_shapes[0]->value;
      int64_t value = (nio_size - nbo_size);
      return value;
    };
    cons2.rel_tiling_vars = new TilingVariable*[1];
    cons2.rel_tiling_vars_size = 1u;
    cons2.rel_tiling_vars[0] = &nio_size;
    cons2.rel_in_shapes = new Variable*[1];
    cons2.rel_in_shapes_size = 1u;
    cons2.rel_in_shapes[0] = &nbo_size;
    cons2.type = ConstraintType::MC_MIXED;
    cons2.eval = cons2Eval;
    GetUpperBoundFuncPtr nbo_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t A = parent_vars[0]->value;
      if (A == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= A;
      return upper_bound;
    };
    nbo_size.upper_bound = nbo_size_upper_bound;
    nbo_size.upper_bound_vars = new Variable * [1];
    nbo_size.upper_bound_vars_size = 1u;
    nbo_size.upper_bound_vars[0] = &A;
    nbo_size.rel_cons = new Constraint*[2];
    nbo_size.rel_cons_size = 2u;
    nbo_size.rel_cons[0] = &cons1;
    nbo_size.rel_cons[1] = &cons2;
    nio_size.align = 1;
    GetUpperBoundFuncPtr nio_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t nbo_size = parent_vars[0]->value;
      if (nbo_size == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= nbo_size;
      return upper_bound;
    };
    nio_size.upper_bound = nio_size_upper_bound;
    nio_size.upper_bound_vars = new Variable * [1];
    nio_size.upper_bound_vars_size = 1u;
    nio_size.upper_bound_vars[0] = &nbo_size;
    nio_size.rel_cons = new Constraint*[2];
    nio_size.rel_cons_size = 2u;
    nio_size.rel_cons[0] = &cons0;
    nio_size.rel_cons[1] = &cons2;
    AxesReorderSolverInput input;
    input.input_vars = new Variable*[3];
    input.input_vars_size = 3u;
    input.input_vars[0] = &A;
    input.input_vars[1] = &R;
    input.input_vars[2] = &BL;
    input.tiling_vars = new TilingVariable*[2];
    input.tiling_vars_size = 2u;
    input.tiling_vars[0] = &nbo_size;
    input.tiling_vars[1] = &nio_size;
    input.all_cons_size = 3u;
    input.all_cons = new Constraint*[3];
    input.all_cons[0] = &cons0;
    input.all_cons[1] = &cons1;
    input.all_cons[2] = &cons2;
    input.pure_mc_vars_size = 1u;
    input.pure_mc_vars = new TilingVariable*[1];
    input.pure_mc_vars[0] = &nbo_size;
    input.local_buffer_vars_size = 1u;
    input.local_buffer_vars = new TilingVariable*[1];
    input.local_buffer_vars[0] = &nio_size;
    input.core_num = corenum_;
    AxesReorderSolvercase1102* solver = new AxesReorderSolvercase1102(input);
    if (!solver->Run()) {
        return false;
    }
    tiling_data.set_nbo_size(input.pure_mc_vars[0]->value);
    tiling_data.set_nio_size(input.local_buffer_vars[0]->value);
    return true;
  }

  bool DoTiling(graph_normalTilingData &tiling_data) {
    if (!ExecuteAxesReorderSolver(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to execute axes reorder solver for tilingCaseId case1102.");
      return false;
    }
    OP_LOGD(OP_NAME, "Execute axes reorder solver for tilingCaseId case1102 successfully.");

    return true;
  }

  int Getub_size(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double nio_size = tiling_data.get_nio_size();

    return ((10 * R * nio_size) + (6 * R) + (8 * nio_size) + Max((4 * R * nio_size), (2 * R * nio_size)));
  }

  int Getblock_dim(graph_normalTilingData& tiling_data) {
    double A = tiling_data.get_A();
    double nbo_size = tiling_data.get_nbo_size();

    return Max(0, ceiling((A / (nbo_size))));
  }

  void UpdateGeneralTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_block_dim((((tiling_data.get_A() + tiling_data.get_nbo_size()) - 1) / tiling_data.get_nbo_size()));
  }

  void UpdateAxesTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_A_aligned_size((tiling_data.get_A() - 1) / 8 * 8 + 8);
    tiling_data.set_R_aligned_size((tiling_data.get_R() - 1) / 8 * 8 + 8);
    tiling_data.set_nio_loop_num(((tiling_data.get_nbo_size() + tiling_data.get_nio_size()) - 1) / tiling_data.get_nio_size());
    tiling_data.set_nbo_loop_num(((tiling_data.get_A() + tiling_data.get_nbo_size()) - 1) / tiling_data.get_nbo_size());
    tiling_data.set_nio_tail_size((tiling_data.get_nbo_size() % tiling_data.get_nio_size()) == 0 ? tiling_data.get_nio_size() : (tiling_data.get_nbo_size() % tiling_data.get_nio_size()));
    tiling_data.set_nbo_tail_size((tiling_data.get_A() % tiling_data.get_nbo_size()) == 0 ? tiling_data.get_nbo_size() : (tiling_data.get_A() % tiling_data.get_nbo_size()));
    tiling_data.set_nbo_tail_tile_nio_loop_num(((tiling_data.get_nbo_tail_size() + tiling_data.get_nio_size()) - 1) / tiling_data.get_nio_size());
    tiling_data.set_nbo_tail_tile_nio_tail_size((tiling_data.get_nbo_tail_size() % tiling_data.get_nio_size()) == 0 ? tiling_data.get_nio_size() : (tiling_data.get_nbo_tail_size() % tiling_data.get_nio_size()));
  }

  void SetQ0(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q0((2 * R * nio_size));
  }

  void SetQ1(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q1((2 * R * nio_size));
  }

  void SetQ2(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    tiling_data.set_Q2((2 * R));
  }

  void SetQ3(graph_normalTilingData &tiling_data) {
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q3((4 * nio_size));
  }

  void SetQ4(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q4((2 * R * nio_size));
  }

  void SetQ5(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q5((4 * R * nio_size));
  }

  void SetQ6(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q6(Max((4 * R * nio_size), (2 * R * nio_size)));
  }

  void SetQ7(graph_normalTilingData &tiling_data) {
    const auto nio_size = tiling_data.get_nio_size();
    tiling_data.set_Q7((4 * nio_size));
  }

  void SetQ8(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    tiling_data.set_Q8((2 * R));
  }

  void SetQ9(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    tiling_data.set_Q9((2 * R));
  }

  void ComputeOptionParam(graph_normalTilingData &tiling_data) {
    SetQ0(tiling_data);
    SetQ1(tiling_data);
    SetQ2(tiling_data);
    SetQ3(tiling_data);
    SetQ4(tiling_data);
    SetQ5(tiling_data);
    SetQ6(tiling_data);
    SetQ7(tiling_data);
    SetQ8(tiling_data);
    SetQ9(tiling_data);

  }
  void AssignAttAndOutputSize(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    OP_LOGD(OP_NAME, "Start assigning attr and output size for tiling case 1102.");
    auto attrs = context->GetAttrs();
    auto additional_output_ptr = attrs->GetAttrPointer<int32_t>(1U);
    int32_t additional_output = *additional_output_ptr;
    tiling_data.set_additional_output(additional_output);
    tiling_data.set_output0_total_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize());
    tiling_data.set_output0_single_core_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output1_total_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize());
    tiling_data.set_output1_single_core_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output2_total_size(context->GetOutputShape(2)->GetStorageShape().GetShapeSize());
    tiling_data.set_output2_single_core_size(context->GetOutputShape(2)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output3_total_size(context->GetOutputShape(3)->GetStorageShape().GetShapeSize());
    tiling_data.set_output3_single_core_size(context->GetOutputShape(3)->GetStorageShape().GetShapeSize() / corenum_);

    OP_LOGD(OP_NAME, "Set additional_output to %u.", tiling_data.get_additional_output());
    OP_LOGD(OP_NAME, "Set output0_single_core_size to %u.", tiling_data.get_output0_single_core_size());
    OP_LOGD(OP_NAME, "Set output0_total_size to %u.", tiling_data.get_output0_total_size());
    OP_LOGD(OP_NAME, "Set output1_single_core_size to %u.", tiling_data.get_output1_single_core_size());
    OP_LOGD(OP_NAME, "Set output1_total_size to %u.", tiling_data.get_output1_total_size());
    OP_LOGD(OP_NAME, "Set output2_single_core_size to %u.", tiling_data.get_output2_single_core_size());
    OP_LOGD(OP_NAME, "Set output2_total_size to %u.", tiling_data.get_output2_total_size());
    OP_LOGD(OP_NAME, "Set output3_single_core_size to %u.", tiling_data.get_output3_single_core_size());
    OP_LOGD(OP_NAME, "Set output3_total_size to %u.", tiling_data.get_output3_total_size());

    OP_LOGD(OP_NAME, "Assigned attr and output size for tiling case 1102 successfully.");
  }

  void ExtraTilingData(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    OP_LOGD(OP_NAME, "Start executing extra tiling for tilingCaseId 1102.");
		UpdateGeneralTilingData(tiling_data);

    ComputeOptionParam(tiling_data);
		UpdateAxesTilingData(tiling_data);

    AssignAttAndOutputSize(tiling_data, context);
    OP_LOGD(OP_NAME, "Execute extra tiling for tilingCaseId 1102 successfully.");
  }

  void GetWorkSpaceSize(graph_normalTilingData& tiling_data) {
    OP_LOGD(OP_NAME, "Start setting workspace for case 1102.");
    tiling_data.set_workspaceSize(static_cast<uint32_t>(0));
    OP_LOGD(OP_NAME, "Setting workspace to %u for case 1102.", tiling_data.get_workspaceSize());
  }

  void TilingSummary(graph_normalTilingData &tiling_data) {
    OP_LOGI(OP_NAME, "Set nbo_size to %u.", tiling_data.get_nbo_size());
    OP_LOGI(OP_NAME, "Set nio_size to %u.", tiling_data.get_nio_size());
    OP_LOGI(OP_NAME, "The value of ub_size is %d.", Getub_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of block_dim is %d.", Getblock_dim(tiling_data));
  }

};

class AxesReorderSolvercase1111 : public AxesReorderSolver {
 public:
  explicit AxesReorderSolvercase1111(const AxesReorderSolverInput input) : AxesReorderSolver(input) {}
  ~AxesReorderSolvercase1111() = default;
  bool CalUsedCoreNum(int32_t &used_core_num, bool &load_balance) override;
};

bool AxesReorderSolvercase1111::CalUsedCoreNum(int32_t &used_core_num, bool &load_balance) {
  double A = static_cast<double>(input_.input_vars[0]->value);
  double sbo_size = static_cast<double>(input_.pure_mc_vars[0]->value);
  double used_core_num_fp = Max(0, ceiling((A / (sbo_size))));
  load_balance = IsEqual(static_cast<double>(static_cast<int64_t>(used_core_num_fp)), used_core_num_fp);
  used_core_num = ceiling(used_core_num_fp);
  return true;
}

class TilingCase1111Impl : public TilingCaseImpl {
 public:
  TilingCase1111Impl(uint32_t corenum) : TilingCaseImpl(corenum) {

   input_num_ = 5;

   input_dtype_[0] = 1;
   input_dtype_[1] = 1;
   input_dtype_[2] = 1;
   input_dtype_[3] = 1;
   input_dtype_[4] = 1;

   input_format_[0] = 2;
   input_format_[1] = 2;
   input_format_[2] = 2;
   input_format_[3] = 2;
   input_format_[4] = 2;

   max_dim_[0] = 0;
   min_dim_[0] = 2;
   max_dim_[1] = 0;
   min_dim_[1] = 2;
   max_dim_[2] = 0;
   min_dim_[2] = 1;
   max_dim_[3] = 0;
   min_dim_[3] = 1;
   max_dim_[4] = 0;
   min_dim_[4] = 2;

   max_att_[0] = 1;
   min_att_[0] = 1;
   check_att_[0] = true;

  }
 protected:
  bool TilingVarsShapeCheck(gert::TilingContext *context) override {
    int64_t cur_size;
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint64_t input1_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    uint64_t input2_size = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    uint64_t input3_size = context->GetInputShape(3)->GetStorageShape().GetDimNum();
    uint64_t input4_size = context->GetInputShape(4)->GetStorageShape().GetDimNum();

    int64_t A_size = 1;
    cur_size = 1;
    for (size_t i = 0; i <= input0_size - 2; i++) {
      cur_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    A_size = cur_size;
    cur_size = 1;
    for (size_t i = 0; i <= input1_size - 2; i++) {
      cur_size *= context->GetInputShape(1)->GetStorageShape().GetDim(i);
    }
    if (A_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for A_size from input0 and input1.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input4_size - 2; i++) {
      cur_size *= context->GetInputShape(4)->GetStorageShape().GetDim(i);
    }
    if (A_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for A_size from input0 and input4.");
      return false;
    }
    int64_t R_size = 1;
    cur_size = context->GetInputShape(0)->GetStorageShape().GetDim(input0_size - 1);
    R_size = cur_size;
    cur_size = context->GetInputShape(1)->GetStorageShape().GetDim(input1_size - 1);
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input1.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input2_size - 1; i++) {
      cur_size *= context->GetInputShape(2)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input2.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input3_size - 1; i++) {
      cur_size *= context->GetInputShape(3)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input3.");
      return false;
    }
    cur_size = context->GetInputShape(4)->GetStorageShape().GetDim(input4_size - 1);
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input4.");
      return false;
    }

    OP_LOGD(OP_NAME, "TilingVarsShapeCheck success.");
    return true;
  }

  bool GetShapeAttrsInfo(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    if (!TilingVarsShapeCheck(context)) {
      OP_LOGW(OP_NAME, "TilingVarsShapeCheck failed.");
      return false;
    }
    OP_LOGD(OP_NAME, "Start setting axis size for 1111.");
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();

    uint32_t A_size = 1;
    for (size_t i = 0; i <= input0_size - 2; i++) {
      A_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    tiling_data.set_A(A_size);
    OP_LOGD(OP_NAME, "Initiate A to %d.", tiling_data.get_A());
    uint32_t R_size = context->GetInputShape(0)->GetStorageShape().GetDim(input0_size - 1);
    tiling_data.set_R(R_size);
    OP_LOGD(OP_NAME, "Initiate R to %d.", tiling_data.get_R());

    OP_LOGD(OP_NAME, "End setting axis size for 1111.");
    return true;
  }

  bool CheckIsCapable(graph_normalTilingData &tiling_data) {
    OP_LOGD(OP_NAME, "CheckIsCapable success.");
    return true;
  }

  bool ExecuteAxesReorderSolver(graph_normalTilingData& tiling_data) {
    Variable A;
    A.value = tiling_data.get_A();
    Variable R;
    R.value = tiling_data.get_R();
    TilingVariable sbo_size;
    TilingVariable sio_size;
    int64_t ub_size = tiling_data.get_ub_size();
    Constraint cons0;
    auto cons0Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t sio_size = rel_tiling_vars[0]->value;
      int64_t R = rel_in_shapes[0]->value;
      int64_t value = ((4 * R) + (8 * sio_size) + 8 + Max((2 * R), (4 * R))) - rel_hw_spec;
      return value;
    };
    cons0.rel_tiling_vars = new TilingVariable*[1];
    cons0.rel_tiling_vars_size = 1u;
    cons0.rel_tiling_vars[0] = &sio_size;
    cons0.rel_in_shapes = new Variable*[1];
    cons0.rel_in_shapes_size = 1u;
    cons0.rel_in_shapes[0] = &R;
    cons0.rel_hw_spec = ub_size;
    cons0.type = ConstraintType::LOCAL_BUFFER;
    cons0.eval = cons0Eval;
    Constraint cons1;
    auto cons1Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t sbo_size = rel_in_shapes[0]->value;
      int64_t A = rel_in_shapes[1]->value;
      int64_t value = (sbo_size - A);
      return value;
    };
    cons1.rel_in_shapes = new Variable*[2];
    cons1.rel_in_shapes_size = 2u;
    cons1.rel_in_shapes[0] = &sbo_size;
    cons1.rel_in_shapes[1] = &A;
    cons1.type = ConstraintType::MC_MIXED;
    cons1.eval = cons1Eval;
    Constraint cons2;
    auto cons2Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t sio_size = rel_tiling_vars[0]->value;
      int64_t R = rel_in_shapes[0]->value;
      int64_t value = (sio_size - R);
      return value;
    };
    cons2.rel_tiling_vars = new TilingVariable*[1];
    cons2.rel_tiling_vars_size = 1u;
    cons2.rel_tiling_vars[0] = &sio_size;
    cons2.rel_in_shapes = new Variable*[1];
    cons2.rel_in_shapes_size = 1u;
    cons2.rel_in_shapes[0] = &R;
    cons2.type = ConstraintType::LB_MIXED;
    cons2.eval = cons2Eval;
    GetUpperBoundFuncPtr sbo_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t A = parent_vars[0]->value;
      if (A == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= A;
      return upper_bound;
    };
    sbo_size.upper_bound = sbo_size_upper_bound;
    sbo_size.upper_bound_vars = new Variable * [1];
    sbo_size.upper_bound_vars_size = 1u;
    sbo_size.upper_bound_vars[0] = &A;
    sbo_size.rel_cons = new Constraint*[1];
    sbo_size.rel_cons_size = 1u;
    sbo_size.rel_cons[0] = &cons1;
    sio_size.align = 16;
    GetUpperBoundFuncPtr sio_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t R = parent_vars[0]->value;
      if (R == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= R;
      return upper_bound;
    };
    sio_size.upper_bound = sio_size_upper_bound;
    sio_size.upper_bound_vars = new Variable * [1];
    sio_size.upper_bound_vars_size = 1u;
    sio_size.upper_bound_vars[0] = &R;
    sio_size.rel_cons = new Constraint*[2];
    sio_size.rel_cons_size = 2u;
    sio_size.rel_cons[0] = &cons0;
    sio_size.rel_cons[1] = &cons2;
    AxesReorderSolverInput input;
    input.input_vars = new Variable*[2];
    input.input_vars_size = 2u;
    input.input_vars[0] = &A;
    input.input_vars[1] = &R;
    input.tiling_vars = new TilingVariable*[2];
    input.tiling_vars_size = 2u;
    input.tiling_vars[0] = &sbo_size;
    input.tiling_vars[1] = &sio_size;
    input.all_cons_size = 3u;
    input.all_cons = new Constraint*[3];
    input.all_cons[0] = &cons0;
    input.all_cons[1] = &cons1;
    input.all_cons[2] = &cons2;
    input.pure_mc_vars_size = 1u;
    input.pure_mc_vars = new TilingVariable*[1];
    input.pure_mc_vars[0] = &sbo_size;
    input.local_buffer_vars_size = 1u;
    input.local_buffer_vars = new TilingVariable*[1];
    input.local_buffer_vars[0] = &sio_size;
    input.core_num = corenum_;
    AxesReorderSolvercase1111* solver = new AxesReorderSolvercase1111(input);
    if (!solver->Run()) {
        return false;
    }
    tiling_data.set_sbo_size(input.pure_mc_vars[0]->value);
    tiling_data.set_sio_size(input.local_buffer_vars[0]->value);
    return true;
  }

  bool DoTiling(graph_normalTilingData &tiling_data) {
    if (!ExecuteAxesReorderSolver(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to execute axes reorder solver for tilingCaseId case1111.");
      return false;
    }
    OP_LOGD(OP_NAME, "Execute axes reorder solver for tilingCaseId case1111 successfully.");

    return true;
  }

  int Getub_size(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double sio_size = tiling_data.get_sio_size();

    return ((4 * R) + (8 * sio_size) + 8 + Max((2 * R), (4 * R)));
  }

  int Getblock_dim(graph_normalTilingData& tiling_data) {
    double A = tiling_data.get_A();
    double sbo_size = tiling_data.get_sbo_size();

    return Max(0, ceiling((A / (sbo_size))));
  }

  void UpdateGeneralTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_block_dim((((tiling_data.get_A() + tiling_data.get_sbo_size()) - 1) / tiling_data.get_sbo_size()));
  }

  void UpdateAxesTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_R_aligned_size((tiling_data.get_R() - 1) / 8 * 8 + 8);
    tiling_data.set_A_aligned_size((tiling_data.get_A() - 1) / 8 * 8 + 8);
    tiling_data.set_sio_loop_num(((tiling_data.get_R() + tiling_data.get_sio_size()) - 1) / tiling_data.get_sio_size());
    tiling_data.set_sbo_loop_num(((tiling_data.get_A() + tiling_data.get_sbo_size()) - 1) / tiling_data.get_sbo_size());
    tiling_data.set_sio_tail_size((tiling_data.get_R() % tiling_data.get_sio_size()) == 0 ? tiling_data.get_sio_size() : (tiling_data.get_R() % tiling_data.get_sio_size()));
    tiling_data.set_sbo_tail_size((tiling_data.get_A() % tiling_data.get_sbo_size()) == 0 ? tiling_data.get_sbo_size() : (tiling_data.get_A() % tiling_data.get_sbo_size()));
  }

  void SetQ0(graph_normalTilingData &tiling_data) {
    const auto sio_size = tiling_data.get_sio_size();
    tiling_data.set_Q0((2 * sio_size));
  }

  void SetQ1(graph_normalTilingData &tiling_data) {
    const auto sio_size = tiling_data.get_sio_size();
    tiling_data.set_Q1((2 * sio_size));
  }

  void SetQ2(graph_normalTilingData &tiling_data) {
    const auto sio_size = tiling_data.get_sio_size();
    tiling_data.set_Q2((2 * sio_size));
  }

  void SetQ3(graph_normalTilingData &tiling_data) {
    tiling_data.set_Q3(4);
  }

  void SetQ4(graph_normalTilingData &tiling_data) {
    const auto sio_size = tiling_data.get_sio_size();
    tiling_data.set_Q4((2 * sio_size));
  }

  void SetQ5(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    tiling_data.set_Q5(Max((2 * R), (4 * R)));
  }

  void SetQ6(graph_normalTilingData &tiling_data) {
    tiling_data.set_Q6(4);
  }

  void SetQ7(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    tiling_data.set_Q7((2 * R));
  }

  void SetQ8(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    tiling_data.set_Q8((2 * R));
  }

  void ComputeOptionParam(graph_normalTilingData &tiling_data) {
    SetQ0(tiling_data);
    SetQ1(tiling_data);
    SetQ2(tiling_data);
    SetQ3(tiling_data);
    SetQ4(tiling_data);
    SetQ5(tiling_data);
    SetQ6(tiling_data);
    SetQ7(tiling_data);
    SetQ8(tiling_data);

  }
  void AssignAttAndOutputSize(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    OP_LOGD(OP_NAME, "Start assigning attr and output size for tiling case 1111.");
    auto attrs = context->GetAttrs();
    auto additional_output_ptr = attrs->GetAttrPointer<int32_t>(1U);
    int32_t additional_output = *additional_output_ptr;
    tiling_data.set_additional_output(additional_output);
    tiling_data.set_output0_total_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize());
    tiling_data.set_output0_single_core_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output1_total_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize());
    tiling_data.set_output1_single_core_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output2_total_size(context->GetOutputShape(2)->GetStorageShape().GetShapeSize());
    tiling_data.set_output2_single_core_size(context->GetOutputShape(2)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output3_total_size(context->GetOutputShape(3)->GetStorageShape().GetShapeSize());
    tiling_data.set_output3_single_core_size(context->GetOutputShape(3)->GetStorageShape().GetShapeSize() / corenum_);

    OP_LOGD(OP_NAME, "Set additional_output to %u.", tiling_data.get_additional_output());
    OP_LOGD(OP_NAME, "Set output0_single_core_size to %u.", tiling_data.get_output0_single_core_size());
    OP_LOGD(OP_NAME, "Set output0_total_size to %u.", tiling_data.get_output0_total_size());
    OP_LOGD(OP_NAME, "Set output1_single_core_size to %u.", tiling_data.get_output1_single_core_size());
    OP_LOGD(OP_NAME, "Set output1_total_size to %u.", tiling_data.get_output1_total_size());
    OP_LOGD(OP_NAME, "Set output2_single_core_size to %u.", tiling_data.get_output2_single_core_size());
    OP_LOGD(OP_NAME, "Set output2_total_size to %u.", tiling_data.get_output2_total_size());
    OP_LOGD(OP_NAME, "Set output3_single_core_size to %u.", tiling_data.get_output3_single_core_size());
    OP_LOGD(OP_NAME, "Set output3_total_size to %u.", tiling_data.get_output3_total_size());

    OP_LOGD(OP_NAME, "Assigned attr and output size for tiling case 1111 successfully.");
  }

  void ExtraTilingData(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    OP_LOGD(OP_NAME, "Start executing extra tiling for tilingCaseId 1111.");
		UpdateGeneralTilingData(tiling_data);

    ComputeOptionParam(tiling_data);
		UpdateAxesTilingData(tiling_data);

    AssignAttAndOutputSize(tiling_data, context);
    OP_LOGD(OP_NAME, "Execute extra tiling for tilingCaseId 1111 successfully.");
  }

  void GetWorkSpaceSize(graph_normalTilingData& tiling_data) {
    OP_LOGD(OP_NAME, "Start setting workspace for case 1111.");
    tiling_data.set_workspaceSize(static_cast<uint32_t>(0));
    OP_LOGD(OP_NAME, "Setting workspace to %u for case 1111.", tiling_data.get_workspaceSize());
  }

  void TilingSummary(graph_normalTilingData &tiling_data) {
    OP_LOGI(OP_NAME, "Set sbo_size to %u.", tiling_data.get_sbo_size());
    OP_LOGI(OP_NAME, "Set sio_size to %u.", tiling_data.get_sio_size());
    OP_LOGI(OP_NAME, "The value of ub_size is %d.", Getub_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of block_dim is %d.", Getblock_dim(tiling_data));
  }

};

class AxesReorderSolvercase1112 : public AxesReorderSolver {
 public:
  explicit AxesReorderSolvercase1112(const AxesReorderSolverInput input) : AxesReorderSolver(input) {}
  ~AxesReorderSolvercase1112() = default;
  bool CalUsedCoreNum(int32_t &used_core_num, bool &load_balance) override;
};

bool AxesReorderSolvercase1112::CalUsedCoreNum(int32_t &used_core_num, bool &load_balance) {
  double A = static_cast<double>(input_.input_vars[0]->value);
  double sbo_size = static_cast<double>(input_.pure_mc_vars[0]->value);
  double used_core_num_fp = Max(0, ceiling((A / (sbo_size))));
  load_balance = IsEqual(static_cast<double>(static_cast<int64_t>(used_core_num_fp)), used_core_num_fp);
  used_core_num = ceiling(used_core_num_fp);
  return true;
}

class TilingCase1112Impl : public TilingCaseImpl {
 public:
  TilingCase1112Impl(uint32_t corenum) : TilingCaseImpl(corenum) {

   input_num_ = 5;

   input_dtype_[0] = 1;
   input_dtype_[1] = 1;
   input_dtype_[2] = 1;
   input_dtype_[3] = 1;
   input_dtype_[4] = 1;

   input_format_[0] = 2;
   input_format_[1] = 2;
   input_format_[2] = 2;
   input_format_[3] = 2;
   input_format_[4] = 2;

   max_dim_[0] = 0;
   min_dim_[0] = 2;
   max_dim_[1] = 0;
   min_dim_[1] = 2;
   max_dim_[2] = 0;
   min_dim_[2] = 1;
   max_dim_[3] = 0;
   min_dim_[3] = 1;
   max_dim_[4] = 0;
   min_dim_[4] = 1;

   max_att_[0] = 1;
   min_att_[0] = 1;
   check_att_[0] = true;

  }
 protected:
  bool TilingVarsShapeCheck(gert::TilingContext *context) override {
    int64_t cur_size;
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint64_t input1_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    uint64_t input2_size = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    uint64_t input3_size = context->GetInputShape(3)->GetStorageShape().GetDimNum();
    uint64_t input4_size = context->GetInputShape(4)->GetStorageShape().GetDimNum();

    int64_t A_size = 1;
    cur_size = 1;
    for (size_t i = 0; i <= input0_size - 2; i++) {
      cur_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    A_size = cur_size;
    cur_size = 1;
    for (size_t i = 0; i <= input1_size - 2; i++) {
      cur_size *= context->GetInputShape(1)->GetStorageShape().GetDim(i);
    }
    if (A_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for A_size from input0 and input1.");
      return false;
    }
    int64_t R_size = 1;
    cur_size = context->GetInputShape(0)->GetStorageShape().GetDim(input0_size - 1);
    R_size = cur_size;
    cur_size = context->GetInputShape(1)->GetStorageShape().GetDim(input1_size - 1);
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input1.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input2_size - 1; i++) {
      cur_size *= context->GetInputShape(2)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input2.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input3_size - 1; i++) {
      cur_size *= context->GetInputShape(3)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input3.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input4_size - 1; i++) {
      cur_size *= context->GetInputShape(4)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input4.");
      return false;
    }

    OP_LOGD(OP_NAME, "TilingVarsShapeCheck success.");
    return true;
  }

  bool GetShapeAttrsInfo(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    if (!TilingVarsShapeCheck(context)) {
      OP_LOGW(OP_NAME, "TilingVarsShapeCheck failed.");
      return false;
    }
    OP_LOGD(OP_NAME, "Start setting axis size for 1112.");
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();

    uint32_t A_size = 1;
    for (size_t i = 0; i <= input0_size - 2; i++) {
      A_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    tiling_data.set_A(A_size);
    OP_LOGD(OP_NAME, "Initiate A to %d.", tiling_data.get_A());
    uint32_t R_size = context->GetInputShape(0)->GetStorageShape().GetDim(input0_size - 1);
    tiling_data.set_R(R_size);
    OP_LOGD(OP_NAME, "Initiate R to %d.", tiling_data.get_R());

    OP_LOGD(OP_NAME, "End setting axis size for 1112.");
    return true;
  }

  bool CheckIsCapable(graph_normalTilingData &tiling_data) {
    OP_LOGD(OP_NAME, "CheckIsCapable success.");
    return true;
  }

  bool ExecuteAxesReorderSolver(graph_normalTilingData& tiling_data) {
    Variable A;
    A.value = tiling_data.get_A();
    Variable R;
    R.value = tiling_data.get_R();
    TilingVariable sbo_size;
    TilingVariable sio_size;
    int64_t ub_size = tiling_data.get_ub_size();
    Constraint cons0;
    auto cons0Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t sio_size = rel_tiling_vars[0]->value;
      int64_t R = rel_in_shapes[0]->value;
      int64_t value = ((4 * R) + (8 * sio_size) + 8 + Max((2 * R), (4 * R))) - rel_hw_spec;
      return value;
    };
    cons0.rel_tiling_vars = new TilingVariable*[1];
    cons0.rel_tiling_vars_size = 1u;
    cons0.rel_tiling_vars[0] = &sio_size;
    cons0.rel_in_shapes = new Variable*[1];
    cons0.rel_in_shapes_size = 1u;
    cons0.rel_in_shapes[0] = &R;
    cons0.rel_hw_spec = ub_size;
    cons0.type = ConstraintType::LOCAL_BUFFER;
    cons0.eval = cons0Eval;
    Constraint cons1;
    auto cons1Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t sbo_size = rel_in_shapes[0]->value;
      int64_t A = rel_in_shapes[1]->value;
      int64_t value = (sbo_size - A);
      return value;
    };
    cons1.rel_in_shapes = new Variable*[2];
    cons1.rel_in_shapes_size = 2u;
    cons1.rel_in_shapes[0] = &sbo_size;
    cons1.rel_in_shapes[1] = &A;
    cons1.type = ConstraintType::MC_MIXED;
    cons1.eval = cons1Eval;
    Constraint cons2;
    auto cons2Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t sio_size = rel_tiling_vars[0]->value;
      int64_t R = rel_in_shapes[0]->value;
      int64_t value = (sio_size - R);
      return value;
    };
    cons2.rel_tiling_vars = new TilingVariable*[1];
    cons2.rel_tiling_vars_size = 1u;
    cons2.rel_tiling_vars[0] = &sio_size;
    cons2.rel_in_shapes = new Variable*[1];
    cons2.rel_in_shapes_size = 1u;
    cons2.rel_in_shapes[0] = &R;
    cons2.type = ConstraintType::LB_MIXED;
    cons2.eval = cons2Eval;
    GetUpperBoundFuncPtr sbo_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t A = parent_vars[0]->value;
      if (A == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= A;
      return upper_bound;
    };
    sbo_size.upper_bound = sbo_size_upper_bound;
    sbo_size.upper_bound_vars = new Variable * [1];
    sbo_size.upper_bound_vars_size = 1u;
    sbo_size.upper_bound_vars[0] = &A;
    sbo_size.rel_cons = new Constraint*[1];
    sbo_size.rel_cons_size = 1u;
    sbo_size.rel_cons[0] = &cons1;
    sio_size.align = 16;
    GetUpperBoundFuncPtr sio_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t R = parent_vars[0]->value;
      if (R == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= R;
      return upper_bound;
    };
    sio_size.upper_bound = sio_size_upper_bound;
    sio_size.upper_bound_vars = new Variable * [1];
    sio_size.upper_bound_vars_size = 1u;
    sio_size.upper_bound_vars[0] = &R;
    sio_size.rel_cons = new Constraint*[2];
    sio_size.rel_cons_size = 2u;
    sio_size.rel_cons[0] = &cons0;
    sio_size.rel_cons[1] = &cons2;
    AxesReorderSolverInput input;
    input.input_vars = new Variable*[2];
    input.input_vars_size = 2u;
    input.input_vars[0] = &A;
    input.input_vars[1] = &R;
    input.tiling_vars = new TilingVariable*[2];
    input.tiling_vars_size = 2u;
    input.tiling_vars[0] = &sbo_size;
    input.tiling_vars[1] = &sio_size;
    input.all_cons_size = 3u;
    input.all_cons = new Constraint*[3];
    input.all_cons[0] = &cons0;
    input.all_cons[1] = &cons1;
    input.all_cons[2] = &cons2;
    input.pure_mc_vars_size = 1u;
    input.pure_mc_vars = new TilingVariable*[1];
    input.pure_mc_vars[0] = &sbo_size;
    input.local_buffer_vars_size = 1u;
    input.local_buffer_vars = new TilingVariable*[1];
    input.local_buffer_vars[0] = &sio_size;
    input.core_num = corenum_;
    AxesReorderSolvercase1112* solver = new AxesReorderSolvercase1112(input);
    if (!solver->Run()) {
        return false;
    }
    tiling_data.set_sbo_size(input.pure_mc_vars[0]->value);
    tiling_data.set_sio_size(input.local_buffer_vars[0]->value);
    return true;
  }

  bool DoTiling(graph_normalTilingData &tiling_data) {
    if (!ExecuteAxesReorderSolver(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to execute axes reorder solver for tilingCaseId case1112.");
      return false;
    }
    OP_LOGD(OP_NAME, "Execute axes reorder solver for tilingCaseId case1112 successfully.");

    return true;
  }

  int Getub_size(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double sio_size = tiling_data.get_sio_size();

    return ((4 * R) + (8 * sio_size) + 8 + Max((2 * R), (4 * R)));
  }

  int Getblock_dim(graph_normalTilingData& tiling_data) {
    double A = tiling_data.get_A();
    double sbo_size = tiling_data.get_sbo_size();

    return Max(0, ceiling((A / (sbo_size))));
  }

  void UpdateGeneralTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_block_dim((((tiling_data.get_A() + tiling_data.get_sbo_size()) - 1) / tiling_data.get_sbo_size()));
  }

  void UpdateAxesTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_R_aligned_size((tiling_data.get_R() - 1) / 8 * 8 + 8);
    tiling_data.set_A_aligned_size((tiling_data.get_A() - 1) / 8 * 8 + 8);
    tiling_data.set_sio_loop_num(((tiling_data.get_R() + tiling_data.get_sio_size()) - 1) / tiling_data.get_sio_size());
    tiling_data.set_sbo_loop_num(((tiling_data.get_A() + tiling_data.get_sbo_size()) - 1) / tiling_data.get_sbo_size());
    tiling_data.set_sio_tail_size((tiling_data.get_R() % tiling_data.get_sio_size()) == 0 ? tiling_data.get_sio_size() : (tiling_data.get_R() % tiling_data.get_sio_size()));
    tiling_data.set_sbo_tail_size((tiling_data.get_A() % tiling_data.get_sbo_size()) == 0 ? tiling_data.get_sbo_size() : (tiling_data.get_A() % tiling_data.get_sbo_size()));
  }

  void SetQ0(graph_normalTilingData &tiling_data) {
    const auto sio_size = tiling_data.get_sio_size();
    tiling_data.set_Q0((2 * sio_size));
  }

  void SetQ1(graph_normalTilingData &tiling_data) {
    const auto sio_size = tiling_data.get_sio_size();
    tiling_data.set_Q1((2 * sio_size));
  }

  void SetQ2(graph_normalTilingData &tiling_data) {
    const auto sio_size = tiling_data.get_sio_size();
    tiling_data.set_Q2((2 * sio_size));
  }

  void SetQ3(graph_normalTilingData &tiling_data) {
    tiling_data.set_Q3(4);
  }

  void SetQ4(graph_normalTilingData &tiling_data) {
    const auto sio_size = tiling_data.get_sio_size();
    tiling_data.set_Q4((2 * sio_size));
  }

  void SetQ5(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    tiling_data.set_Q5(Max((2 * R), (4 * R)));
  }

  void SetQ6(graph_normalTilingData &tiling_data) {
    tiling_data.set_Q6(4);
  }

  void SetQ7(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    tiling_data.set_Q7((2 * R));
  }

  void SetQ8(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    tiling_data.set_Q8((2 * R));
  }

  void ComputeOptionParam(graph_normalTilingData &tiling_data) {
    SetQ0(tiling_data);
    SetQ1(tiling_data);
    SetQ2(tiling_data);
    SetQ3(tiling_data);
    SetQ4(tiling_data);
    SetQ5(tiling_data);
    SetQ6(tiling_data);
    SetQ7(tiling_data);
    SetQ8(tiling_data);

  }
  void AssignAttAndOutputSize(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    OP_LOGD(OP_NAME, "Start assigning attr and output size for tiling case 1112.");
    auto attrs = context->GetAttrs();
    auto additional_output_ptr = attrs->GetAttrPointer<int32_t>(1U);
    int32_t additional_output = *additional_output_ptr;
    tiling_data.set_additional_output(additional_output);
    tiling_data.set_output0_total_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize());
    tiling_data.set_output0_single_core_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output1_total_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize());
    tiling_data.set_output1_single_core_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output2_total_size(context->GetOutputShape(2)->GetStorageShape().GetShapeSize());
    tiling_data.set_output2_single_core_size(context->GetOutputShape(2)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output3_total_size(context->GetOutputShape(3)->GetStorageShape().GetShapeSize());
    tiling_data.set_output3_single_core_size(context->GetOutputShape(3)->GetStorageShape().GetShapeSize() / corenum_);

    OP_LOGD(OP_NAME, "Set additional_output to %u.", tiling_data.get_additional_output());
    OP_LOGD(OP_NAME, "Set output0_single_core_size to %u.", tiling_data.get_output0_single_core_size());
    OP_LOGD(OP_NAME, "Set output0_total_size to %u.", tiling_data.get_output0_total_size());
    OP_LOGD(OP_NAME, "Set output1_single_core_size to %u.", tiling_data.get_output1_single_core_size());
    OP_LOGD(OP_NAME, "Set output1_total_size to %u.", tiling_data.get_output1_total_size());
    OP_LOGD(OP_NAME, "Set output2_single_core_size to %u.", tiling_data.get_output2_single_core_size());
    OP_LOGD(OP_NAME, "Set output2_total_size to %u.", tiling_data.get_output2_total_size());
    OP_LOGD(OP_NAME, "Set output3_single_core_size to %u.", tiling_data.get_output3_single_core_size());
    OP_LOGD(OP_NAME, "Set output3_total_size to %u.", tiling_data.get_output3_total_size());

    OP_LOGD(OP_NAME, "Assigned attr and output size for tiling case 1112 successfully.");
  }

  void ExtraTilingData(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    OP_LOGD(OP_NAME, "Start executing extra tiling for tilingCaseId 1112.");
		UpdateGeneralTilingData(tiling_data);

    ComputeOptionParam(tiling_data);
		UpdateAxesTilingData(tiling_data);

    AssignAttAndOutputSize(tiling_data, context);
    OP_LOGD(OP_NAME, "Execute extra tiling for tilingCaseId 1112 successfully.");
  }

  void GetWorkSpaceSize(graph_normalTilingData& tiling_data) {
    OP_LOGD(OP_NAME, "Start setting workspace for case 1112.");
    tiling_data.set_workspaceSize(static_cast<uint32_t>(0));
    OP_LOGD(OP_NAME, "Setting workspace to %u for case 1112.", tiling_data.get_workspaceSize());
  }

  void TilingSummary(graph_normalTilingData &tiling_data) {
    OP_LOGI(OP_NAME, "Set sbo_size to %u.", tiling_data.get_sbo_size());
    OP_LOGI(OP_NAME, "Set sio_size to %u.", tiling_data.get_sio_size());
    OP_LOGI(OP_NAME, "The value of ub_size is %d.", Getub_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of block_dim is %d.", Getblock_dim(tiling_data));
  }

};

class AxesReorderSolvercase1151 : public AxesReorderSolver {
 public:
  explicit AxesReorderSolvercase1151(const AxesReorderSolverInput input) : AxesReorderSolver(input) {}
  ~AxesReorderSolvercase1151() = default;
  bool CalUsedCoreNum(int32_t &used_core_num, bool &load_balance) override;
};

bool AxesReorderSolvercase1151::CalUsedCoreNum(int32_t &used_core_num, bool &load_balance) {
  double A = static_cast<double>(input_.input_vars[0]->value);
  double wbo_size = static_cast<double>(input_.pure_mc_vars[0]->value);
  double used_core_num_fp = Max(0, ceiling((A / (wbo_size))));
  load_balance = IsEqual(static_cast<double>(static_cast<int64_t>(used_core_num_fp)), used_core_num_fp);
  used_core_num = ceiling(used_core_num_fp);
  return true;
}

class TilingCase1151Impl : public TilingCaseImpl {
 public:
  TilingCase1151Impl(uint32_t corenum) : TilingCaseImpl(corenum) {

   input_num_ = 5;

   input_dtype_[0] = 1;
   input_dtype_[1] = 1;
   input_dtype_[2] = 1;
   input_dtype_[3] = 1;
   input_dtype_[4] = 1;

   input_format_[0] = 2;
   input_format_[1] = 2;
   input_format_[2] = 2;
   input_format_[3] = 2;
   input_format_[4] = 2;

   max_dim_[0] = 0;
   min_dim_[0] = 2;
   max_dim_[1] = 0;
   min_dim_[1] = 2;
   max_dim_[2] = 0;
   min_dim_[2] = 1;
   max_dim_[3] = 0;
   min_dim_[3] = 1;
   max_dim_[4] = 0;
   min_dim_[4] = 2;

   max_att_[0] = 1;
   min_att_[0] = 1;
   check_att_[0] = true;

  }
 protected:
  bool TilingVarsShapeCheck(gert::TilingContext *context) override {
    int64_t cur_size;
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint64_t input1_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    uint64_t input2_size = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    uint64_t input3_size = context->GetInputShape(3)->GetStorageShape().GetDimNum();
    uint64_t input4_size = context->GetInputShape(4)->GetStorageShape().GetDimNum();

    int64_t A_size = 1;
    cur_size = 1;
    for (size_t i = 0; i <= input0_size - 2; i++) {
      cur_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    A_size = cur_size;
    cur_size = 1;
    for (size_t i = 0; i <= input1_size - 2; i++) {
      cur_size *= context->GetInputShape(1)->GetStorageShape().GetDim(i);
    }
    if (A_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for A_size from input0 and input1.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input4_size - 2; i++) {
      cur_size *= context->GetInputShape(4)->GetStorageShape().GetDim(i);
    }
    if (A_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for A_size from input0 and input4.");
      return false;
    }
    int64_t R_size = 1;
    cur_size = context->GetInputShape(0)->GetStorageShape().GetDim(input0_size - 1);
    R_size = cur_size;
    cur_size = context->GetInputShape(1)->GetStorageShape().GetDim(input1_size - 1);
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input1.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input2_size - 1; i++) {
      cur_size *= context->GetInputShape(2)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input2.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input3_size - 1; i++) {
      cur_size *= context->GetInputShape(3)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input3.");
      return false;
    }
    cur_size = context->GetInputShape(4)->GetStorageShape().GetDim(input4_size - 1);
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input4.");
      return false;
    }

    OP_LOGD(OP_NAME, "TilingVarsShapeCheck success.");
    return true;
  }

  bool GetShapeAttrsInfo(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    if (!TilingVarsShapeCheck(context)) {
      OP_LOGW(OP_NAME, "TilingVarsShapeCheck failed.");
      return false;
    }
    OP_LOGD(OP_NAME, "Start setting axis size for 1151.");
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();

    uint32_t A_size = 1;
    for (size_t i = 0; i <= input0_size - 2; i++) {
      A_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    tiling_data.set_A(A_size);
    OP_LOGD(OP_NAME, "Initiate A to %d.", tiling_data.get_A());
    uint32_t R_size = context->GetInputShape(0)->GetStorageShape().GetDim(input0_size - 1);
    tiling_data.set_R(R_size);
    OP_LOGD(OP_NAME, "Initiate R to %d.", tiling_data.get_R());

    OP_LOGD(OP_NAME, "End setting axis size for 1151.");
    return true;
  }

  bool CheckIsCapable(graph_normalTilingData &tiling_data) {
    OP_LOGD(OP_NAME, "CheckIsCapable success.");
    return true;
  }

  bool ExecuteAxesReorderSolver(graph_normalTilingData& tiling_data) {
    Variable A;
    A.value = tiling_data.get_A();
    Variable R;
    R.value = tiling_data.get_R();
    TilingVariable wbo_size;
    TilingVariable wio_size;
    int64_t ub_size = tiling_data.get_ub_size();
    Constraint cons0;
    auto cons0Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t wio_size = rel_tiling_vars[0]->value;
      int64_t value = ((16 * wio_size) + 8 + Max((2 * wio_size), (4 * wio_size))) - rel_hw_spec;
      return value;
    };
    cons0.rel_tiling_vars = new TilingVariable*[1];
    cons0.rel_tiling_vars_size = 1u;
    cons0.rel_tiling_vars[0] = &wio_size;
    cons0.rel_hw_spec = ub_size;
    cons0.type = ConstraintType::LOCAL_BUFFER;
    cons0.eval = cons0Eval;
    Constraint cons1;
    auto cons1Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t wbo_size = rel_in_shapes[0]->value;
      int64_t A = rel_in_shapes[1]->value;
      int64_t value = (wbo_size - A);
      return value;
    };
    cons1.rel_in_shapes = new Variable*[2];
    cons1.rel_in_shapes_size = 2u;
    cons1.rel_in_shapes[0] = &wbo_size;
    cons1.rel_in_shapes[1] = &A;
    cons1.type = ConstraintType::MC_MIXED;
    cons1.eval = cons1Eval;
    Constraint cons2;
    auto cons2Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t wio_size = rel_tiling_vars[0]->value;
      int64_t R = rel_in_shapes[0]->value;
      int64_t value = (wio_size - R);
      return value;
    };
    cons2.rel_tiling_vars = new TilingVariable*[1];
    cons2.rel_tiling_vars_size = 1u;
    cons2.rel_tiling_vars[0] = &wio_size;
    cons2.rel_in_shapes = new Variable*[1];
    cons2.rel_in_shapes_size = 1u;
    cons2.rel_in_shapes[0] = &R;
    cons2.type = ConstraintType::LB_MIXED;
    cons2.eval = cons2Eval;
    GetUpperBoundFuncPtr wbo_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t A = parent_vars[0]->value;
      if (A == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= A;
      return upper_bound;
    };
    wbo_size.upper_bound = wbo_size_upper_bound;
    wbo_size.upper_bound_vars = new Variable * [1];
    wbo_size.upper_bound_vars_size = 1u;
    wbo_size.upper_bound_vars[0] = &A;
    wbo_size.rel_cons = new Constraint*[1];
    wbo_size.rel_cons_size = 1u;
    wbo_size.rel_cons[0] = &cons1;
    wio_size.align = 1;
    GetUpperBoundFuncPtr wio_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t R = parent_vars[0]->value;
      if (R == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= R;
      return upper_bound;
    };
    wio_size.upper_bound = wio_size_upper_bound;
    wio_size.upper_bound_vars = new Variable * [1];
    wio_size.upper_bound_vars_size = 1u;
    wio_size.upper_bound_vars[0] = &R;
    wio_size.rel_cons = new Constraint*[2];
    wio_size.rel_cons_size = 2u;
    wio_size.rel_cons[0] = &cons0;
    wio_size.rel_cons[1] = &cons2;
    AxesReorderSolverInput input;
    input.input_vars = new Variable*[2];
    input.input_vars_size = 2u;
    input.input_vars[0] = &A;
    input.input_vars[1] = &R;
    input.tiling_vars = new TilingVariable*[2];
    input.tiling_vars_size = 2u;
    input.tiling_vars[0] = &wbo_size;
    input.tiling_vars[1] = &wio_size;
    input.all_cons_size = 3u;
    input.all_cons = new Constraint*[3];
    input.all_cons[0] = &cons0;
    input.all_cons[1] = &cons1;
    input.all_cons[2] = &cons2;
    input.pure_mc_vars_size = 1u;
    input.pure_mc_vars = new TilingVariable*[1];
    input.pure_mc_vars[0] = &wbo_size;
    input.local_buffer_vars_size = 1u;
    input.local_buffer_vars = new TilingVariable*[1];
    input.local_buffer_vars[0] = &wio_size;
    input.core_num = corenum_;
    AxesReorderSolvercase1151* solver = new AxesReorderSolvercase1151(input);
    if (!solver->Run()) {
        return false;
    }
    tiling_data.set_wbo_size(input.pure_mc_vars[0]->value);
    tiling_data.set_wio_size(input.local_buffer_vars[0]->value);
    return true;
  }

  bool DoTiling(graph_normalTilingData &tiling_data) {
    if (!ExecuteAxesReorderSolver(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to execute axes reorder solver for tilingCaseId case1151.");
      return false;
    }
    OP_LOGD(OP_NAME, "Execute axes reorder solver for tilingCaseId case1151 successfully.");

    return true;
  }

  int Getub_size(graph_normalTilingData& tiling_data) {
    double wio_size = tiling_data.get_wio_size();

    return ((16 * wio_size) + 8 + Max((2 * wio_size), (4 * wio_size)));
  }

  int Getblock_dim(graph_normalTilingData& tiling_data) {
    double A = tiling_data.get_A();
    double wbo_size = tiling_data.get_wbo_size();

    return Max(0, ceiling((A / (wbo_size))));
  }

  void UpdateGeneralTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_block_dim((((tiling_data.get_A() + tiling_data.get_wbo_size()) - 1) / tiling_data.get_wbo_size()));
  }

  void UpdateAxesTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_R_aligned_size((tiling_data.get_R() - 1) / 8 * 8 + 8);
    tiling_data.set_A_aligned_size((tiling_data.get_A() - 1) / 8 * 8 + 8);
    tiling_data.set_wio_loop_num(((tiling_data.get_R() + tiling_data.get_wio_size()) - 1) / tiling_data.get_wio_size());
    tiling_data.set_wbo_loop_num(((tiling_data.get_A() + tiling_data.get_wbo_size()) - 1) / tiling_data.get_wbo_size());
    tiling_data.set_wio_tail_size((tiling_data.get_R() % tiling_data.get_wio_size()) == 0 ? tiling_data.get_wio_size() : (tiling_data.get_R() % tiling_data.get_wio_size()));
    tiling_data.set_wbo_tail_size((tiling_data.get_A() % tiling_data.get_wbo_size()) == 0 ? tiling_data.get_wbo_size() : (tiling_data.get_A() % tiling_data.get_wbo_size()));
  }

  void SetQ0(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q0((2 * wio_size));
  }

  void SetQ1(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q1((2 * wio_size));
  }

  void SetQ2(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q2((2 * wio_size));
  }

  void SetQ3(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q3((2 * wio_size));
  }

  void SetQ4(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q4((4 * wio_size));
  }

  void SetQ5(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q5(Max((2 * wio_size), (4 * wio_size)));
  }

  void SetQ6(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q6((4 * wio_size));
  }

  void SetQ7(graph_normalTilingData &tiling_data) {
    tiling_data.set_Q7(4);
  }

  void SetQ8(graph_normalTilingData &tiling_data) {
    tiling_data.set_Q8(4);
  }

  void ComputeOptionParam(graph_normalTilingData &tiling_data) {
    SetQ0(tiling_data);
    SetQ1(tiling_data);
    SetQ2(tiling_data);
    SetQ3(tiling_data);
    SetQ4(tiling_data);
    SetQ5(tiling_data);
    SetQ6(tiling_data);
    SetQ7(tiling_data);
    SetQ8(tiling_data);

  }
  void AssignAttAndOutputSize(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    OP_LOGD(OP_NAME, "Start assigning attr and output size for tiling case 1151.");
    auto attrs = context->GetAttrs();
    auto additional_output_ptr = attrs->GetAttrPointer<int32_t>(1U);
    int32_t additional_output = *additional_output_ptr;
    tiling_data.set_additional_output(additional_output);
    tiling_data.set_output0_total_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize());
    tiling_data.set_output0_single_core_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output1_total_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize());
    tiling_data.set_output1_single_core_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output2_total_size(context->GetOutputShape(2)->GetStorageShape().GetShapeSize());
    tiling_data.set_output2_single_core_size(context->GetOutputShape(2)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output3_total_size(context->GetOutputShape(3)->GetStorageShape().GetShapeSize());
    tiling_data.set_output3_single_core_size(context->GetOutputShape(3)->GetStorageShape().GetShapeSize() / corenum_);

    OP_LOGD(OP_NAME, "Set additional_output to %u.", tiling_data.get_additional_output());
    OP_LOGD(OP_NAME, "Set output0_single_core_size to %u.", tiling_data.get_output0_single_core_size());
    OP_LOGD(OP_NAME, "Set output0_total_size to %u.", tiling_data.get_output0_total_size());
    OP_LOGD(OP_NAME, "Set output1_single_core_size to %u.", tiling_data.get_output1_single_core_size());
    OP_LOGD(OP_NAME, "Set output1_total_size to %u.", tiling_data.get_output1_total_size());
    OP_LOGD(OP_NAME, "Set output2_single_core_size to %u.", tiling_data.get_output2_single_core_size());
    OP_LOGD(OP_NAME, "Set output2_total_size to %u.", tiling_data.get_output2_total_size());
    OP_LOGD(OP_NAME, "Set output3_single_core_size to %u.", tiling_data.get_output3_single_core_size());
    OP_LOGD(OP_NAME, "Set output3_total_size to %u.", tiling_data.get_output3_total_size());

    OP_LOGD(OP_NAME, "Assigned attr and output size for tiling case 1151 successfully.");
  }

  void ExtraTilingData(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    OP_LOGD(OP_NAME, "Start executing extra tiling for tilingCaseId 1151.");
		UpdateGeneralTilingData(tiling_data);

    ComputeOptionParam(tiling_data);
		UpdateAxesTilingData(tiling_data);

    AssignAttAndOutputSize(tiling_data, context);
    OP_LOGD(OP_NAME, "Execute extra tiling for tilingCaseId 1151 successfully.");
  }

  void GetWorkSpaceSize(graph_normalTilingData& tiling_data) {
    OP_LOGD(OP_NAME, "Start setting workspace for case 1151.");
    double A = static_cast<double>(tiling_data.get_A());
    double R = static_cast<double>(tiling_data.get_R());
    double wbo_size = static_cast<double>(tiling_data.get_wbo_size());
    tiling_data.set_workspaceSize(static_cast<uint32_t>((4 * Max(0, ceiling((A / (wbo_size)))) * R * wbo_size)));
    OP_LOGD(OP_NAME, "Setting workspace to %u for case 1151.", tiling_data.get_workspaceSize());
  }

  void TilingSummary(graph_normalTilingData &tiling_data) {
    OP_LOGI(OP_NAME, "Set wbo_size to %u.", tiling_data.get_wbo_size());
    OP_LOGI(OP_NAME, "Set wio_size to %u.", tiling_data.get_wio_size());
    OP_LOGI(OP_NAME, "The value of ub_size is %d.", Getub_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of block_dim is %d.", Getblock_dim(tiling_data));
  }

};

class AxesReorderSolvercase1152 : public AxesReorderSolver {
 public:
  explicit AxesReorderSolvercase1152(const AxesReorderSolverInput input) : AxesReorderSolver(input) {}
  ~AxesReorderSolvercase1152() = default;
  bool CalUsedCoreNum(int32_t &used_core_num, bool &load_balance) override;
};

bool AxesReorderSolvercase1152::CalUsedCoreNum(int32_t &used_core_num, bool &load_balance) {
  double A = static_cast<double>(input_.input_vars[0]->value);
  double wbo_size = static_cast<double>(input_.pure_mc_vars[0]->value);
  double used_core_num_fp = Max(0, ceiling((A / (wbo_size))));
  load_balance = IsEqual(static_cast<double>(static_cast<int64_t>(used_core_num_fp)), used_core_num_fp);
  used_core_num = ceiling(used_core_num_fp);
  return true;
}

class TilingCase1152Impl : public TilingCaseImpl {
 public:
  TilingCase1152Impl(uint32_t corenum) : TilingCaseImpl(corenum) {

   input_num_ = 5;

   input_dtype_[0] = 1;
   input_dtype_[1] = 1;
   input_dtype_[2] = 1;
   input_dtype_[3] = 1;
   input_dtype_[4] = 1;

   input_format_[0] = 2;
   input_format_[1] = 2;
   input_format_[2] = 2;
   input_format_[3] = 2;
   input_format_[4] = 2;

   max_dim_[0] = 0;
   min_dim_[0] = 2;
   max_dim_[1] = 0;
   min_dim_[1] = 2;
   max_dim_[2] = 0;
   min_dim_[2] = 1;
   max_dim_[3] = 0;
   min_dim_[3] = 1;
   max_dim_[4] = 0;
   min_dim_[4] = 1;

   max_att_[0] = 1;
   min_att_[0] = 1;
   check_att_[0] = true;

  }
 protected:
  bool TilingVarsShapeCheck(gert::TilingContext *context) override {
    int64_t cur_size;
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint64_t input1_size = context->GetInputShape(1)->GetStorageShape().GetDimNum();
    uint64_t input2_size = context->GetInputShape(2)->GetStorageShape().GetDimNum();
    uint64_t input3_size = context->GetInputShape(3)->GetStorageShape().GetDimNum();
    uint64_t input4_size = context->GetInputShape(4)->GetStorageShape().GetDimNum();

    int64_t A_size = 1;
    cur_size = 1;
    for (size_t i = 0; i <= input0_size - 2; i++) {
      cur_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    A_size = cur_size;
    cur_size = 1;
    for (size_t i = 0; i <= input1_size - 2; i++) {
      cur_size *= context->GetInputShape(1)->GetStorageShape().GetDim(i);
    }
    if (A_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for A_size from input0 and input1.");
      return false;
    }
    int64_t R_size = 1;
    cur_size = context->GetInputShape(0)->GetStorageShape().GetDim(input0_size - 1);
    R_size = cur_size;
    cur_size = context->GetInputShape(1)->GetStorageShape().GetDim(input1_size - 1);
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input1.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input2_size - 1; i++) {
      cur_size *= context->GetInputShape(2)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input2.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input3_size - 1; i++) {
      cur_size *= context->GetInputShape(3)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input3.");
      return false;
    }
    cur_size = 1;
    for (size_t i = 0; i <= input4_size - 1; i++) {
      cur_size *= context->GetInputShape(4)->GetStorageShape().GetDim(i);
    }
    if (R_size != cur_size) {
      OP_LOGW(OP_NAME, "Inconsistent shape for R_size from input0 and input4.");
      return false;
    }

    OP_LOGD(OP_NAME, "TilingVarsShapeCheck success.");
    return true;
  }

  bool GetShapeAttrsInfo(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    if (!TilingVarsShapeCheck(context)) {
      OP_LOGW(OP_NAME, "TilingVarsShapeCheck failed.");
      return false;
    }
    OP_LOGD(OP_NAME, "Start setting axis size for 1152.");
    uint64_t input0_size = context->GetInputShape(0)->GetStorageShape().GetDimNum();

    uint32_t A_size = 1;
    for (size_t i = 0; i <= input0_size - 2; i++) {
      A_size *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    tiling_data.set_A(A_size);
    OP_LOGD(OP_NAME, "Initiate A to %d.", tiling_data.get_A());
    uint32_t R_size = context->GetInputShape(0)->GetStorageShape().GetDim(input0_size - 1);
    tiling_data.set_R(R_size);
    OP_LOGD(OP_NAME, "Initiate R to %d.", tiling_data.get_R());

    OP_LOGD(OP_NAME, "End setting axis size for 1152.");
    return true;
  }

  bool CheckIsCapable(graph_normalTilingData &tiling_data) {
    OP_LOGD(OP_NAME, "CheckIsCapable success.");
    return true;
  }

  bool ExecuteAxesReorderSolver(graph_normalTilingData& tiling_data) {
    Variable A;
    A.value = tiling_data.get_A();
    Variable R;
    R.value = tiling_data.get_R();
    TilingVariable wbo_size;
    TilingVariable wio_size;
    int64_t ub_size = tiling_data.get_ub_size();
    Constraint cons0;
    auto cons0Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t wio_size = rel_tiling_vars[0]->value;
      int64_t value = ((16 * wio_size) + 8 + Max((2 * wio_size), (4 * wio_size))) - rel_hw_spec;
      return value;
    };
    cons0.rel_tiling_vars = new TilingVariable*[1];
    cons0.rel_tiling_vars_size = 1u;
    cons0.rel_tiling_vars[0] = &wio_size;
    cons0.rel_hw_spec = ub_size;
    cons0.type = ConstraintType::LOCAL_BUFFER;
    cons0.eval = cons0Eval;
    Constraint cons1;
    auto cons1Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t wbo_size = rel_in_shapes[0]->value;
      int64_t A = rel_in_shapes[1]->value;
      int64_t value = (wbo_size - A);
      return value;
    };
    cons1.rel_in_shapes = new Variable*[2];
    cons1.rel_in_shapes_size = 2u;
    cons1.rel_in_shapes[0] = &wbo_size;
    cons1.rel_in_shapes[1] = &A;
    cons1.type = ConstraintType::MC_MIXED;
    cons1.eval = cons1Eval;
    Constraint cons2;
    auto cons2Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t wio_size = rel_tiling_vars[0]->value;
      int64_t R = rel_in_shapes[0]->value;
      int64_t value = (wio_size - R);
      return value;
    };
    cons2.rel_tiling_vars = new TilingVariable*[1];
    cons2.rel_tiling_vars_size = 1u;
    cons2.rel_tiling_vars[0] = &wio_size;
    cons2.rel_in_shapes = new Variable*[1];
    cons2.rel_in_shapes_size = 1u;
    cons2.rel_in_shapes[0] = &R;
    cons2.type = ConstraintType::LB_MIXED;
    cons2.eval = cons2Eval;
    GetUpperBoundFuncPtr wbo_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t A = parent_vars[0]->value;
      if (A == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= A;
      return upper_bound;
    };
    wbo_size.upper_bound = wbo_size_upper_bound;
    wbo_size.upper_bound_vars = new Variable * [1];
    wbo_size.upper_bound_vars_size = 1u;
    wbo_size.upper_bound_vars[0] = &A;
    wbo_size.rel_cons = new Constraint*[1];
    wbo_size.rel_cons_size = 1u;
    wbo_size.rel_cons[0] = &cons1;
    wio_size.align = 1;
    GetUpperBoundFuncPtr wio_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t R = parent_vars[0]->value;
      if (R == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= R;
      return upper_bound;
    };
    wio_size.upper_bound = wio_size_upper_bound;
    wio_size.upper_bound_vars = new Variable * [1];
    wio_size.upper_bound_vars_size = 1u;
    wio_size.upper_bound_vars[0] = &R;
    wio_size.rel_cons = new Constraint*[2];
    wio_size.rel_cons_size = 2u;
    wio_size.rel_cons[0] = &cons0;
    wio_size.rel_cons[1] = &cons2;
    AxesReorderSolverInput input;
    input.input_vars = new Variable*[2];
    input.input_vars_size = 2u;
    input.input_vars[0] = &A;
    input.input_vars[1] = &R;
    input.tiling_vars = new TilingVariable*[2];
    input.tiling_vars_size = 2u;
    input.tiling_vars[0] = &wbo_size;
    input.tiling_vars[1] = &wio_size;
    input.all_cons_size = 3u;
    input.all_cons = new Constraint*[3];
    input.all_cons[0] = &cons0;
    input.all_cons[1] = &cons1;
    input.all_cons[2] = &cons2;
    input.pure_mc_vars_size = 1u;
    input.pure_mc_vars = new TilingVariable*[1];
    input.pure_mc_vars[0] = &wbo_size;
    input.local_buffer_vars_size = 1u;
    input.local_buffer_vars = new TilingVariable*[1];
    input.local_buffer_vars[0] = &wio_size;
    input.core_num = corenum_;
    AxesReorderSolvercase1152* solver = new AxesReorderSolvercase1152(input);
    if (!solver->Run()) {
        return false;
    }
    tiling_data.set_wbo_size(input.pure_mc_vars[0]->value);
    tiling_data.set_wio_size(input.local_buffer_vars[0]->value);
    return true;
  }

  bool DoTiling(graph_normalTilingData &tiling_data) {
    if (!ExecuteAxesReorderSolver(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to execute axes reorder solver for tilingCaseId case1152.");
      return false;
    }
    OP_LOGD(OP_NAME, "Execute axes reorder solver for tilingCaseId case1152 successfully.");

    return true;
  }

  int Getub_size(graph_normalTilingData& tiling_data) {
    double wio_size = tiling_data.get_wio_size();

    return ((16 * wio_size) + 8 + Max((2 * wio_size), (4 * wio_size)));
  }

  int Getblock_dim(graph_normalTilingData& tiling_data) {
    double A = tiling_data.get_A();
    double wbo_size = tiling_data.get_wbo_size();

    return Max(0, ceiling((A / (wbo_size))));
  }

  void UpdateGeneralTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_block_dim((((tiling_data.get_A() + tiling_data.get_wbo_size()) - 1) / tiling_data.get_wbo_size()));
  }

  void UpdateAxesTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_R_aligned_size((tiling_data.get_R() - 1) / 8 * 8 + 8);
    tiling_data.set_A_aligned_size((tiling_data.get_A() - 1) / 8 * 8 + 8);
    tiling_data.set_wio_loop_num(((tiling_data.get_R() + tiling_data.get_wio_size()) - 1) / tiling_data.get_wio_size());
    tiling_data.set_wbo_loop_num(((tiling_data.get_A() + tiling_data.get_wbo_size()) - 1) / tiling_data.get_wbo_size());
    tiling_data.set_wio_tail_size((tiling_data.get_R() % tiling_data.get_wio_size()) == 0 ? tiling_data.get_wio_size() : (tiling_data.get_R() % tiling_data.get_wio_size()));
    tiling_data.set_wbo_tail_size((tiling_data.get_A() % tiling_data.get_wbo_size()) == 0 ? tiling_data.get_wbo_size() : (tiling_data.get_A() % tiling_data.get_wbo_size()));
  }

  void SetQ0(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q0((2 * wio_size));
  }

  void SetQ1(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q1((2 * wio_size));
  }

  void SetQ2(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q2((2 * wio_size));
  }

  void SetQ3(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q3((2 * wio_size));
  }

  void SetQ4(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q4((4 * wio_size));
  }

  void SetQ5(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q5(Max((2 * wio_size), (4 * wio_size)));
  }

  void SetQ6(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    tiling_data.set_Q6((4 * wio_size));
  }

  void SetQ7(graph_normalTilingData &tiling_data) {
    tiling_data.set_Q7(4);
  }

  void SetQ8(graph_normalTilingData &tiling_data) {
    tiling_data.set_Q8(4);
  }

  void ComputeOptionParam(graph_normalTilingData &tiling_data) {
    SetQ0(tiling_data);
    SetQ1(tiling_data);
    SetQ2(tiling_data);
    SetQ3(tiling_data);
    SetQ4(tiling_data);
    SetQ5(tiling_data);
    SetQ6(tiling_data);
    SetQ7(tiling_data);
    SetQ8(tiling_data);

  }
  void AssignAttAndOutputSize(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    OP_LOGD(OP_NAME, "Start assigning attr and output size for tiling case 1152.");
    auto attrs = context->GetAttrs();
    auto additional_output_ptr = attrs->GetAttrPointer<int32_t>(1U);
    int32_t additional_output = *additional_output_ptr;
    tiling_data.set_additional_output(additional_output);
    tiling_data.set_output0_total_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize());
    tiling_data.set_output0_single_core_size(context->GetOutputShape(0)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output1_total_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize());
    tiling_data.set_output1_single_core_size(context->GetOutputShape(1)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output2_total_size(context->GetOutputShape(2)->GetStorageShape().GetShapeSize());
    tiling_data.set_output2_single_core_size(context->GetOutputShape(2)->GetStorageShape().GetShapeSize() / corenum_);
    tiling_data.set_output3_total_size(context->GetOutputShape(3)->GetStorageShape().GetShapeSize());
    tiling_data.set_output3_single_core_size(context->GetOutputShape(3)->GetStorageShape().GetShapeSize() / corenum_);

    OP_LOGD(OP_NAME, "Set additional_output to %u.", tiling_data.get_additional_output());
    OP_LOGD(OP_NAME, "Set output0_single_core_size to %u.", tiling_data.get_output0_single_core_size());
    OP_LOGD(OP_NAME, "Set output0_total_size to %u.", tiling_data.get_output0_total_size());
    OP_LOGD(OP_NAME, "Set output1_single_core_size to %u.", tiling_data.get_output1_single_core_size());
    OP_LOGD(OP_NAME, "Set output1_total_size to %u.", tiling_data.get_output1_total_size());
    OP_LOGD(OP_NAME, "Set output2_single_core_size to %u.", tiling_data.get_output2_single_core_size());
    OP_LOGD(OP_NAME, "Set output2_total_size to %u.", tiling_data.get_output2_total_size());
    OP_LOGD(OP_NAME, "Set output3_single_core_size to %u.", tiling_data.get_output3_single_core_size());
    OP_LOGD(OP_NAME, "Set output3_total_size to %u.", tiling_data.get_output3_total_size());

    OP_LOGD(OP_NAME, "Assigned attr and output size for tiling case 1152 successfully.");
  }

  void ExtraTilingData(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
    OP_LOGD(OP_NAME, "Start executing extra tiling for tilingCaseId 1152.");
		UpdateGeneralTilingData(tiling_data);

    ComputeOptionParam(tiling_data);
		UpdateAxesTilingData(tiling_data);

    AssignAttAndOutputSize(tiling_data, context);
    OP_LOGD(OP_NAME, "Execute extra tiling for tilingCaseId 1152 successfully.");
  }

  void GetWorkSpaceSize(graph_normalTilingData& tiling_data) {
    OP_LOGD(OP_NAME, "Start setting workspace for case 1152.");
    double A = static_cast<double>(tiling_data.get_A());
    double R = static_cast<double>(tiling_data.get_R());
    double wbo_size = static_cast<double>(tiling_data.get_wbo_size());
    tiling_data.set_workspaceSize(static_cast<uint32_t>((4 * Max(0, ceiling((A / (wbo_size)))) * R * wbo_size)));
    OP_LOGD(OP_NAME, "Setting workspace to %u for case 1152.", tiling_data.get_workspaceSize());
  }

  void TilingSummary(graph_normalTilingData &tiling_data) {
    OP_LOGI(OP_NAME, "Set wbo_size to %u.", tiling_data.get_wbo_size());
    OP_LOGI(OP_NAME, "Set wio_size to %u.", tiling_data.get_wio_size());
    OP_LOGI(OP_NAME, "The value of ub_size is %d.", Getub_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of block_dim is %d.", Getblock_dim(tiling_data));
  }

};

TilingCaseImplPtr GetTilingImplPtr(uint32_t tilingCaseId, uint32_t corenum) {
  TilingCaseImplPtr tilingCaseImplPtr = nullptr;
  if (tilingCaseId == 1101u) {
    tilingCaseImplPtr = std::make_shared<TilingCase1101Impl>(corenum);
  } else if (tilingCaseId == 1102u) {
    tilingCaseImplPtr = std::make_shared<TilingCase1102Impl>(corenum);
  } else if (tilingCaseId == 1111u) {
    tilingCaseImplPtr = std::make_shared<TilingCase1111Impl>(corenum);
  } else if (tilingCaseId == 1112u) {
    tilingCaseImplPtr = std::make_shared<TilingCase1112Impl>(corenum);
  } else if (tilingCaseId == 1151u) {
    tilingCaseImplPtr = std::make_shared<TilingCase1151Impl>(corenum);
  } else if (tilingCaseId == 1152u) {
    tilingCaseImplPtr = std::make_shared<TilingCase1152Impl>(corenum);
  }
  return tilingCaseImplPtr;
}
bool GetTilingKey(graph_normalTilingData &tiling_data, gert::TilingContext *context, int32_t tilingCaseId = -1) {
  uint32_t corenum = tiling_data.get_block_dim();
  if (tilingCaseId == -1) {
    OP_LOGI(OP_NAME, "The user didn't specify tilingCaseId, iterate all templates.");
    uint32_t tilingKeys[6] = {1101u, 1102u, 1111u, 1112u, 1151u, 1152u};
    for (const auto &tilingKey : tilingKeys) {
      TilingCaseImplPtr tilingCaseImplPtr = GetTilingImplPtr(tilingKey, corenum);
      if (tilingCaseImplPtr == nullptr) {
        OP_LOGE(OP_NAME, "Pointer for tilingCaseId is null.");
        return false;
      }
      if (tilingCaseImplPtr->GetTiling(tiling_data, context)) {
        OP_LOGD(OP_NAME, "Finish calculating the tiling data for tilingCaseId %u.", tilingKey);
        tiling_data.set_tiling_key(tilingKey);
        return true;
      }
    }
    OP_LOGE(OP_NAME, "No solution found in all templates.");
  } else {
    OP_LOGI(OP_NAME, "Calculating the tiling data for tilingCaseId %u.", tilingCaseId);
    TilingCaseImplPtr tilingCaseImplPtr = GetTilingImplPtr(tilingCaseId, corenum);
    if (tilingCaseImplPtr == nullptr) {
      OP_LOGE(OP_NAME, "Pointer for tilingCaseId is null.");
      return false;
    }
    if (tilingCaseImplPtr->GetTiling(tiling_data, context)) {
      tiling_data.set_tiling_key(tilingCaseId);
      return true;
    }
  }
  return false;
}

bool PostTiling(graph_normalTilingData &tiling_data, gert::TilingContext *context) {
// 用户可以自定义函数以修改tiling_data数据与回填的tiling数据
  int64_t workspaceSize = tiling_data.get_workspaceSize();
  int64_t RESERVED_WORKSPACE_SIZE_910B = 16 * 1024 * 1024;
  context->SetTilingKey(tiling_data.get_tiling_key());
  tiling_data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling_data.GetDataSize());
  size_t* currentWorkSpace = context->GetWorkspaceSizes(1);
  size_t sysWorkspaceSize = RESERVED_WORKSPACE_SIZE_910B;
  currentWorkSpace[0] = workspaceSize + sysWorkspaceSize;
  return true;
}

ge::graphStatus GetCtxTiling(gert::TilingContext *context, int32_t tilingCaseId) {
  DurationBegin(TILING_FUNC_DURATION_TOTAL);
  graph_normalTilingData tiling_data;
  OP_LOGI(OP_NAME, "Start context tiling.");
  if (!GetPlatformInfo(tiling_data, context)) {
    OP_LOGE(OP_NAME, "Get platform info Failed.");
    return ge::GRAPH_FAILED;
  }
  OP_LOGI(OP_NAME, "Calculating the tiling data.");
  if (!GetTilingKey(tiling_data, context, tilingCaseId)) {
    OP_LOGE(OP_NAME, "GetTiling Failed.");
    return ge::GRAPH_FAILED;
  }
  OP_LOGI(OP_NAME, "Filing the calculated tiling data in the context.");
  if (PostTiling(tiling_data, context) != true) {
    OP_LOGE(OP_NAME, "PostTiling Failed.");
    return ge::GRAPH_FAILED;
  }
  OP_LOGI(OP_NAME, "End context tiling.");
  DurationEnd(TILING_FUNC_DURATION_TOTAL);
  DurationManager::GetInstance().Print();
  DurationManager::GetInstance().Clear();
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetTiling(gert::TilingContext *context) {
  return GetCtxTiling(context, -1);
}

static ge::graphStatus TilingPrepare4AddLayerNorm(gert::TilingParseContext*) {
  return ge::GRAPH_SUCCESS;
}
struct AddLayerNormCompileInfo {};

IMPL_OP(AddLayerNorm).Tiling(GetTiling).TilingParse<AddLayerNormCompileInfo>(TilingPrepare4AddLayerNorm);
} // namespace optiling

