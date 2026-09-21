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
#include <set>
#include <functional>
#include <chrono>
#include <cstdint>
#include <string>
#include "op_log.h"
#include "AddLayerNorm_tiling_data.h"
#define Max(a, b) ((double)(a) > (double)(b) ? (a) : (b))
#define Min(a, b) ((double)(a) < (double)(b) ? (a) : (b))
#define Log(a) (log((double)(a)))
#define Pow(a, b) pow(a, b)
#define Rational(a, b) ((double)(a) / (double)(b))
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
    if (duration > max_time_) max_time_ = duration;
    if (duration < min_time_) min_time_ = duration;
  }

  void Print() {
    if (total_count_ == 0ULL) return;
    OP_EVENT(OP_NAME, "Duration record: name[%s], total_count[%lu], total_time[%lu], max_time[%lu], min_time[%lu], average_time[%lu].",
      name_.c_str(), total_count_, total_time_, max_time_, min_time_,
      static_cast<uint64_t>(total_time_ / total_count_));
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
    auto now = std::chrono::high_resolution_clock::now();
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());
    return static_cast<uint64_t>(nanoseconds.count());
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

constexpr size_t CASE_ID_LENGTH = 20;
struct IterInfo {
  std::array<char, CASE_ID_LENGTH> case_id;
  int iter_count;
};

class DurationManager {
public:
  static DurationManager &GetInstance() {
    static DurationManager ins;
    return ins;
  }

  DurationManager() {
    const char* env = std::getenv("experimental_enable_att_profiling");
    if (env != nullptr) {
      duration_open_now_ = true;
    } else {
      duration_open_now_ = false;
    }
    for (uint32_t index = 0U; index < static_cast<uint32_t>(TILING_FUNC_DURATION_MAX); index++) {
      AddDuration(index, g_duration_def[index].name);
    }
  }
  
  void AddDuration(const uint32_t type, const std::string &name) {
    if (!duration_open_now_) {
      return;
    }
    duration_infos_[type].stat = std::unique_ptr<Duration>(new(std::nothrow) Duration(name));
    if (duration_infos_[type].stat == nullptr) {
      OP_LOGW(OP_NAME, "Create Duration failed.");
    }
  }

  void AddIterInfo(const char* case_id, uint32_t iter_count) {
    if (!duration_open_now_) {
      return;
    }
    IterInfo info;
    size_t len = Min(strlen(case_id), CASE_ID_LENGTH);
    std::copy(case_id, case_id + len, info.case_id.begin());
    if (len < CASE_ID_LENGTH) {
      info.case_id[len] = '\0';
    }
    info.iter_count = iter_count;
    iter_infos_.push_back(info);
  }

  void AddCaseNumInfo(uint32_t num) {
    if (!duration_open_now_) {
      return;
    }
    case_num_ = num;
  }

  void Begin(const DurationType type) {
    if (!duration_open_now_) {
      return;
    }
    const auto &stat = duration_infos_[type].stat;
    if (stat == nullptr) {
      return;
    }
    stat->Begin();
  }

  void End(const DurationType type) {
    if (!duration_open_now_) {
      return;
    }
    const auto &stat = duration_infos_[type].stat;
    if (stat == nullptr) {
      return;
    }
    stat->End();
  }
  void Print() {
    if (!duration_open_now_) {
      return;
    }
    for (int32_t index = 0; index < static_cast<int32_t>(DurationType::TILING_FUNC_DURATION_MAX); index++) {
      const auto &stat = duration_infos_[index].stat;
      if (stat != nullptr) {
        stat->Print();
      }
    }
    OP_EVENT(OP_NAME, "Case num is %u.", case_num_);
    for (const auto& info : iter_infos_) {
      OP_EVENT(OP_NAME, "%s's iter is %u.", info.case_id.data(), info.iter_count);
    }
  }
  void Clear() {
    if (!duration_open_now_) {
      return;
    }
    for (int32_t index = 0; index < static_cast<int32_t>(DurationType::TILING_FUNC_DURATION_MAX); index++) {
      const auto &stat = duration_infos_[index].stat;
      if (stat != nullptr) {
        stat->Clear();
      }
    }
    iter_infos_.clear();
  }
private:
  bool duration_open_now_ = false;
  DurationInfo duration_infos_[TILING_FUNC_DURATION_MAX];
  std::vector<IterInfo> iter_infos_;
  uint32_t case_num_;
};

static inline void DurationBegin(const DurationType type) {
  DurationManager::GetInstance().Begin(type);
}

static inline void DurationEnd(const DurationType type) {
  DurationManager::GetInstance().End(type);
}

static inline void SaveIterInfo(const char* case_id, uint32_t iter_count) {
  DurationManager::GetInstance().AddIterInfo(case_id, iter_count);
}
static inline void SaveCaseNumInfo(uint32_t num) {
  DurationManager::GetInstance().AddCaseNumInfo(num);
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
    const double epsilon = 1e-8;
    double abs = (a > b) ? (a - b) : (b - a);
    return abs < epsilon;
}
template<typename T>
inline T Ceiling(T a)
{
    T value = static_cast<T>(static_cast<int64_t>(a));
    return (IsEqual(value, a)) ? value : (value + 1);
}
template<typename T>
inline T Floor(T a)
{
    return static_cast<T>(static_cast<int64_t>(a));
}
template<typename T1, typename T2>
inline auto Mod(T1 a, T2 b)->decltype(a % b)
{
    return a % b;
}
template<typename T1, typename T2>
inline auto Mod(T1 a, T2 b)->typename std::enable_if<std::is_floating_point<T1>::value || std::is_floating_point<T2>::value, decltype(std::fmod(a, b))>::type
{
    return std::fmod(a, b);
}

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    int64_t res = a / b;
    return (res * b == a) ? res : (res + 1);
}
struct TilingDataCopy {
  uint32_t A;
  void set_A(uint32_t val) { A = val; }
  inline uint32_t get_A() { return A; }
  uint32_t BL;
  void set_BL(uint32_t val) { BL = val; }
  inline uint32_t get_BL() { return BL; }
  uint32_t KERNEL_INIT_BUFFER;
  void set_KERNEL_INIT_BUFFER(uint32_t val) { KERNEL_INIT_BUFFER = val; }
  inline uint32_t get_KERNEL_INIT_BUFFER() { return KERNEL_INIT_BUFFER; }
  uint32_t Q0;
  void set_Q0(uint32_t val) { Q0 = val; }
  inline uint32_t get_Q0() { return Q0; }
  uint32_t Q1;
  void set_Q1(uint32_t val) { Q1 = val; }
  inline uint32_t get_Q1() { return Q1; }
  uint32_t Q2;
  void set_Q2(uint32_t val) { Q2 = val; }
  inline uint32_t get_Q2() { return Q2; }
  uint32_t Q3;
  void set_Q3(uint32_t val) { Q3 = val; }
  inline uint32_t get_Q3() { return Q3; }
  uint32_t Q4;
  void set_Q4(uint32_t val) { Q4 = val; }
  inline uint32_t get_Q4() { return Q4; }
  uint32_t Q5;
  void set_Q5(uint32_t val) { Q5 = val; }
  inline uint32_t get_Q5() { return Q5; }
  uint32_t Q6;
  void set_Q6(uint32_t val) { Q6 = val; }
  inline uint32_t get_Q6() { return Q6; }
  uint32_t Q7;
  void set_Q7(uint32_t val) { Q7 = val; }
  inline uint32_t get_Q7() { return Q7; }
  uint32_t Q8;
  void set_Q8(uint32_t val) { Q8 = val; }
  inline uint32_t get_Q8() { return Q8; }
  uint32_t Q9;
  void set_Q9(uint32_t val) { Q9 = val; }
  inline uint32_t get_Q9() { return Q9; }
  uint32_t R;
  void set_R(uint32_t val) { R = val; }
  inline uint32_t get_R() { return R; }
  uint32_t block_dim;
  void set_block_dim(uint32_t val) { block_dim = val; }
  inline uint32_t get_block_dim() { return block_dim; }
  uint32_t gm_size;
  void set_gm_size(uint32_t val) { gm_size = val; }
  inline uint32_t get_gm_size() { return gm_size; }
  uint32_t nbo_loop_num;
  void set_nbo_loop_num(uint32_t val) { nbo_loop_num = val; }
  inline uint32_t get_nbo_loop_num() { return nbo_loop_num; }
  uint32_t nbo_size;
  void set_nbo_size(uint32_t val) { nbo_size = val; }
  inline uint32_t get_nbo_size() { return nbo_size; }
  uint32_t nbo_tail_size;
  void set_nbo_tail_size(uint32_t val) { nbo_tail_size = val; }
  inline uint32_t get_nbo_tail_size() { return nbo_tail_size; }
  uint32_t nbo_tail_tile_nio_loop_num;
  void set_nbo_tail_tile_nio_loop_num(uint32_t val) { nbo_tail_tile_nio_loop_num = val; }
  inline uint32_t get_nbo_tail_tile_nio_loop_num() { return nbo_tail_tile_nio_loop_num; }
  uint32_t nbo_tail_tile_nio_tail_size;
  void set_nbo_tail_tile_nio_tail_size(uint32_t val) { nbo_tail_tile_nio_tail_size = val; }
  inline uint32_t get_nbo_tail_tile_nio_tail_size() { return nbo_tail_tile_nio_tail_size; }
  uint32_t nio_loop_num;
  void set_nio_loop_num(uint32_t val) { nio_loop_num = val; }
  inline uint32_t get_nio_loop_num() { return nio_loop_num; }
  uint32_t nio_size;
  void set_nio_size(uint32_t val) { nio_size = val; }
  inline uint32_t get_nio_size() { return nio_size; }
  uint32_t nio_tail_size;
  void set_nio_tail_size(uint32_t val) { nio_tail_size = val; }
  inline uint32_t get_nio_tail_size() { return nio_tail_size; }
  uint32_t output0_single_core_size;
  void set_output0_single_core_size(uint32_t val) { output0_single_core_size = val; }
  inline uint32_t get_output0_single_core_size() { return output0_single_core_size; }
  uint32_t output0_total_size;
  void set_output0_total_size(uint32_t val) { output0_total_size = val; }
  inline uint32_t get_output0_total_size() { return output0_total_size; }
  uint32_t output1_single_core_size;
  void set_output1_single_core_size(uint32_t val) { output1_single_core_size = val; }
  inline uint32_t get_output1_single_core_size() { return output1_single_core_size; }
  uint32_t output1_total_size;
  void set_output1_total_size(uint32_t val) { output1_total_size = val; }
  inline uint32_t get_output1_total_size() { return output1_total_size; }
  uint32_t output2_single_core_size;
  void set_output2_single_core_size(uint32_t val) { output2_single_core_size = val; }
  inline uint32_t get_output2_single_core_size() { return output2_single_core_size; }
  uint32_t output2_total_size;
  void set_output2_total_size(uint32_t val) { output2_total_size = val; }
  inline uint32_t get_output2_total_size() { return output2_total_size; }
  uint32_t output3_single_core_size;
  void set_output3_single_core_size(uint32_t val) { output3_single_core_size = val; }
  inline uint32_t get_output3_single_core_size() { return output3_single_core_size; }
  uint32_t output3_total_size;
  void set_output3_total_size(uint32_t val) { output3_total_size = val; }
  inline uint32_t get_output3_total_size() { return output3_total_size; }
  uint32_t sbo_loop_num;
  void set_sbo_loop_num(uint32_t val) { sbo_loop_num = val; }
  inline uint32_t get_sbo_loop_num() { return sbo_loop_num; }
  uint32_t sbo_size;
  void set_sbo_size(uint32_t val) { sbo_size = val; }
  inline uint32_t get_sbo_size() { return sbo_size; }
  uint32_t sbo_tail_size;
  void set_sbo_tail_size(uint32_t val) { sbo_tail_size = val; }
  inline uint32_t get_sbo_tail_size() { return sbo_tail_size; }
  uint32_t sio_loop_num;
  void set_sio_loop_num(uint32_t val) { sio_loop_num = val; }
  inline uint32_t get_sio_loop_num() { return sio_loop_num; }
  uint32_t sio_size;
  void set_sio_size(uint32_t val) { sio_size = val; }
  inline uint32_t get_sio_size() { return sio_size; }
  uint32_t sio_tail_size;
  void set_sio_tail_size(uint32_t val) { sio_tail_size = val; }
  inline uint32_t get_sio_tail_size() { return sio_tail_size; }
  uint32_t tiling_key;
  void set_tiling_key(uint32_t val) { tiling_key = val; }
  inline uint32_t get_tiling_key() { return tiling_key; }
  uint32_t ub_size;
  void set_ub_size(uint32_t val) { ub_size = val; }
  inline uint32_t get_ub_size() { return ub_size; }
  uint32_t wbo_loop_num;
  void set_wbo_loop_num(uint32_t val) { wbo_loop_num = val; }
  inline uint32_t get_wbo_loop_num() { return wbo_loop_num; }
  uint32_t wbo_size;
  void set_wbo_size(uint32_t val) { wbo_size = val; }
  inline uint32_t get_wbo_size() { return wbo_size; }
  uint32_t wbo_tail_size;
  void set_wbo_tail_size(uint32_t val) { wbo_tail_size = val; }
  inline uint32_t get_wbo_tail_size() { return wbo_tail_size; }
  uint32_t wio_loop_num;
  void set_wio_loop_num(uint32_t val) { wio_loop_num = val; }
  inline uint32_t get_wio_loop_num() { return wio_loop_num; }
  uint32_t wio_size;
  void set_wio_size(uint32_t val) { wio_size = val; }
  inline uint32_t get_wio_size() { return wio_size; }
  uint32_t wio_tail_size;
  void set_wio_tail_size(uint32_t val) { wio_tail_size = val; }
  inline uint32_t get_wio_tail_size() { return wio_tail_size; }
  uint32_t workspaceSize;
  void set_workspaceSize(uint32_t val) { workspaceSize = val; }
  inline uint32_t get_workspaceSize() { return workspaceSize; }
};
struct TilingOption {
  int32_t tiling_case_id{-1};
  int32_t algorithm_index{0};
};
static TilingOption tiling_option_default{};
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
  int64_t prompt_align = 1;
  bool notail = false;
  TilingVariable *notail_var = nullptr;
  uint32_t rel_cons_size = 0u;
  uint32_t upper_bound_vars_size = 0u;
  Variable **upper_bound_vars = nullptr;
  Constraint **rel_cons = nullptr;
  GetUpperBoundFuncPtr upper_bound = nullptr;
  __attribute__((always_inline)) bool SetValue(int64_t val) noexcept{
    return (val > 0) ? (value = val, true) : false;
  }
};

struct AxesReorderSolverInput {
  uint32_t core_num = 0u;
  uint32_t ub_size = 0u;
  uint32_t input_vars_size = 0u;
  uint32_t tiling_vars_size = 0u;
  uint32_t pure_mc_vars_size = 0u;
  uint32_t local_buffer_vars_size = 0u;
  uint32_t all_cons_size = 0u;
  double ub_threshold = 0.2f;
  double corenum_threshold = 0.8f;
  Variable **input_vars = nullptr;
  TilingVariable **tiling_vars = nullptr;
  TilingVariable **pure_mc_vars = nullptr;
  TilingVariable **local_buffer_vars = nullptr;
  Constraint **all_cons = nullptr;
};

class AxesReorderSolver {
public:
  explicit AxesReorderSolver(const AxesReorderSolverInput &input) : input_(input) {}
  ~AxesReorderSolver() {}
  bool Run(const bool is_tuning);
protected:
  virtual bool CalUsedCoreNum(double &used_core_num) = 0;
  virtual bool CalRealUsedCoreNum(int64_t &used_core_num) = 0;
  virtual double GetPerf() = 0;
  virtual bool SatisfyThresholdUBSize() = 0;
  AxesReorderSolverInput input_;
private:
  bool TuneNotailVar(TilingVariable *var);
  bool SatisfyCons(ConstraintType cons_type);
  bool SatisfyCons(TilingVariable *var, ConstraintType cons_type);
  bool SatisfyMCCons();
  bool InitLocalBufferVars();
  bool InitMulticoreVars();
  bool GetMinMulticoreVars();
  bool MulticoreTiling();
  bool NaiveLocalBufTiling();
  bool BinaryLocalBufTiling();
  bool LocalBufTiling(const bool is_tuning);
};

bool AxesReorderSolver::InitLocalBufferVars() {
  auto *vars = input_.local_buffer_vars;
  const auto size = input_.local_buffer_vars_size;
  for (uint32_t i = 0u; i < size; ++i) {
    const uint32_t remain = std::min(4u, size - i);
    for (uint32_t k =0u; k < remain; ++k) {
      if (!vars[i+k]->SetValue(vars[i+k]->align)) {
        OP_LOGW(OP_NAME, "Failed to init local buffer value.");
        return false;
      }
    }
  }
  return true;
}

bool AxesReorderSolver::InitMulticoreVars() {
  uint32_t size = input_.pure_mc_vars_size;
  auto *vars = input_.pure_mc_vars;
  for (uint32_t i = 0u; i < size; i++) {
    auto &var = vars[i];
    auto upper_bound_val = var->upper_bound(var->upper_bound_vars);
    if (upper_bound_val == -1) {
      OP_LOGW(OP_NAME, "Failed to init multicore value.");
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
bool AxesReorderSolver::GetMinMulticoreVars() {
  uint32_t size = input_.pure_mc_vars_size;
  auto *vars = input_.pure_mc_vars;
  for (uint32_t i = 0u; i < size; i++) {
    auto &var = vars[i];
    if (!var->SetValue(var->align)) {
      OP_LOGW(OP_NAME, "Failed to init multicore value.");
      return false;
    }
  }
  return true;
}
bool AxesReorderSolver::SatisfyCons(ConstraintType cons_type) {
  uint32_t size = input_.all_cons_size;
  auto *cons_list = input_.all_cons;
  for (uint32_t i = 0u; i < size; ++i) {
    const uint32_t remain = std::min(4u, size - i);
    for (uint32_t k =0u; k < remain; ++k) {
      auto &cons = cons_list[i+k];
    if (cons->type != cons_type) {
      continue;
    }
    if (cons->eval(cons->rel_tiling_vars, cons->rel_in_shapes, cons->rel_hw_spec) > 0) {
      return false;
    }
    }
  }
  return true;
}
bool AxesReorderSolver::SatisfyCons(TilingVariable *var, ConstraintType cons_type) {
  uint32_t size = var->rel_cons_size;
  auto *cons_list = var->rel_cons;
  for (uint32_t i = 0u; i < size; ++i) {
    const uint32_t remain = std::min(4u, size - i);
    for (uint32_t k =0u; k < remain; ++k) {
      auto &cons = cons_list[i+k];
    if (cons->type != cons_type) {
      continue;
    }
    if (cons->eval(cons->rel_tiling_vars, cons->rel_in_shapes, cons->rel_hw_spec) > 0) {
      return false;
    }
    }
  }
  return true;
}
bool AxesReorderSolver::SatisfyMCCons() {
  int32_t used_core_num = 0;
  CalRealUsedCoreNum(used_core_num);
  return used_core_num <= static_cast<int64_t>(input_.core_num);
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
    OP_LOGW(OP_NAME, "multicore tiling failed");
    return false;
  }
  if (!SatisfyMCCons()) {
    OP_LOGW(OP_NAME, "Multicore Tiling Calculation failed in the first check.");
    return false;
  }
  const int32_t num_vars = input_.pure_mc_vars_size;
  auto *vars = input_.pure_mc_vars;
  for (int32_t i=num_vars - 1; i >= 0; --i) {
    auto &var = vars[i];
    int64_t boundary = var->align;
    auto init_val = var->value;
    int64_t last_boundary = -1;
    int64_t last_val = -1;
    double pre_obj = GetPerf();
    while (!(last_boundary == boundary && last_val == var->value)) {
      last_boundary = boundary;
      last_val = var->value;
      var->value = CeilDiv((boundary + var->value) / 2, var->align) * var->align;
      double cur_obj = GetPerf();
      var->value += 1;
      double next_obj = GetPerf();
      var->value -= 1;
      if (!SatisfyMCCons() || cur_obj > pre_obj || cur_obj > next_obj) {
        boundary = var->value;
        var->value = last_val;
      } else {
        pre_obj = cur_obj;
      }
    }
    if (!SatisfyMCCons()) {
      var->value = init_val;
    }
    while (!SatisfyCons(var, ConstraintType::MC_MIXED) && var->value != init_val) {
      var->value += var->align;
    }
  }
  if (!SatisfyCons(ConstraintType::MC_MIXED)) {
    OP_LOGW(OP_NAME, "Multicore Tiling Calculation failed in the final check.");
    return false;
  }
  return true;
}
bool AxesReorderSolver::NaiveLocalBufTiling() {
  if (!InitLocalBufferVars()) {
    OP_LOGW(OP_NAME, "init local buffer failed");
    return false;
  }
  if (!InitMulticoreVars()) {
    OP_LOGW(OP_NAME, "multicore tiling failed");
    return false;
  }
  if (!SatisfyCons(ConstraintType::LOCAL_BUFFER)) {
    OP_LOGW(OP_NAME, "local buffer tiling failed in the initial check");
    return false;
  }
  uint32_t num_vars = input_.local_buffer_vars_size;
  auto *vars = input_.local_buffer_vars;
  for (uint32_t i = 0u; i < num_vars; ++i) {
    auto &var = vars[i];
    auto upper_bound = var->upper_bound(var->upper_bound_vars);
    int64_t boundary = CeilDiv(upper_bound, var->align) * var->align;
    for (int64_t val=boundary; val >= var->align; val -= var->align) {
      var->SetValue(val);
      if (SatisfyCons(var, ConstraintType::LOCAL_BUFFER)) {
        if (SatisfyThresholdUBSize()) {
          if (!MulticoreTiling()) {
            OP_LOGW(OP_NAME, "Tune multicore tiling failed");
            return false;
          }
          int64_t max_corenum = 0;
          double threshold = input_.corenum_threshold;
          CalRealUsedCoreNum(max_corenum);
          if (max_corenum >= static_cast<int64_t>(threshold * static_cast<double>(input_.core_num))) {
            break;
          }
        } else {
          break;
        }
      }
    }
    if (!TuneNotailVar(var)) {
      OP_LOGW(OP_NAME, "Tune notail var failed");
      return false;
    }
    while (!SatisfyCons(var, ConstraintType::LB_MIXED) && var->value!= var->align) {
      var->value -= var->align;
      if (!TuneNotailVar(var)) {
        OP_LOGW(OP_NAME, "Tune notail var failed");
        return false;
      }
    }
    while ((var->value >= var->prompt_align) && (var->value % var->prompt_align!= 0)) {
      var->value -= var->align;
    }
  }
  if (!SatisfyCons(ConstraintType::LB_MIXED)) {
    OP_LOGW(OP_NAME, "Local Tiling Calculation failed in the final check.");
    return false;
  }
  return true;
}
bool AxesReorderSolver::BinaryLocalBufTiling() {
  if (!InitLocalBufferVars()) {
    OP_LOGW(OP_NAME, "init local buffer failed");
    return false;
  }
  if (!InitMulticoreVars()) {
    OP_LOGW(OP_NAME, "multicore tiling failed");
    return false;
  }
  if (!SatisfyCons(ConstraintType::LOCAL_BUFFER)) {
    OP_LOGW(OP_NAME, "local buffer tiling failed in the initial check");
    return false;
  }
  uint32_t num_vars = input_.local_buffer_vars_size;
  auto *vars = input_.local_buffer_vars;
  for (uint32_t i = 0u; i < num_vars; ++i) {
    auto &var = vars[i];
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
      OP_LOGW(OP_NAME, "Tune notail var failed");
      return false;
    }
    while (!SatisfyCons(var, ConstraintType::LB_MIXED) && var->value != var->align) {
      var->value -= var->align;
      if (!TuneNotailVar(var)) {
        OP_LOGW(OP_NAME, "Tune notail var failed");
        return false;
      }
    }
    while ((var->value >= var->prompt_align) && (var->value % var->prompt_align != 0)) {
      var->value -= var->align;
    }
  }
  if (!SatisfyCons(ConstraintType::LB_MIXED)) {
    OP_LOGW(OP_NAME, "Local Tiling Calculation failed in the final check.");
    return false;
  }
  return true;
}
bool AxesReorderSolver::LocalBufTiling(const bool is_tuning) {
  if (is_tuning) {
    return NaiveLocalBufTiling();
  } else {
    return BinaryLocalBufTiling();
  }
}
bool AxesReorderSolver::Run(const bool is_tuning) {
  if (!LocalBufTiling(is_tuning)) {
    OP_LOGW(OP_NAME, "local buffer tiling failed");
    return false;
  }
  OP_LOGI(OP_NAME, "local buffer tiling success");
  if (!MulticoreTiling()) {
    OP_LOGW(OP_NAME, "multicore tiling failed");
    return false;
  }
  OP_LOGI(OP_NAME, "multicore tiling success");
  return true;
}

class TilingCaseImpl {
 public:
  TilingCaseImpl(uint32_t corenum) : corenum_(corenum) {}
  virtual ~TilingCaseImpl() = default;
  bool GetTiling(graph_normalTilingData &tiling_data, double &cur_ub_ratio) {
    OP_LOGD(OP_NAME, "[PROF]Execute DoTiling.");
    if (!DoTiling(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to do tiling.");
      return false;
    }
    DoApiTiling(tiling_data);
    GeneralTiling(tiling_data);
    GetWorkSpaceSize(tiling_data);
    ExtraTilingData(tiling_data);
    TilingSummary(tiling_data, cur_ub_ratio);
    return true;
  }
  virtual double GetPerf(graph_normalTilingData &tiling_data) { return 0.0; }
  virtual void TilingSummary(graph_normalTilingData &tiling_data, double &cur_ub_ratio) = 0;
  virtual void GetTilingData(TilingDataCopy &from_tiling, graph_normalTilingData &to_tiling) {};
  virtual void SetTilingData(graph_normalTilingData &from_tiling, TilingDataCopy &to_tiling) {};
 protected:
  virtual bool DoTiling(graph_normalTilingData &tiling_data) = 0;
  virtual void DoApiTiling(graph_normalTilingData &tiling_data) {}
  virtual void GeneralTiling(graph_normalTilingData& tiling_data) {}
  virtual void GetWorkSpaceSize(graph_normalTilingData& tiling_data) {}
  virtual void ExtraTilingData(graph_normalTilingData &tiling_data) {}
  uint32_t corenum_;
};
using TilingCaseImplPtr = std::shared_ptr<TilingCaseImpl>;

class AxesReorderSolvercase1101 : public AxesReorderSolver {
 public:
  explicit AxesReorderSolvercase1101(const AxesReorderSolverInput input) : AxesReorderSolver(input) {}
  ~AxesReorderSolvercase1101() = default;
  bool CalUsedCoreNum(double &used_core_num) override;
  bool CalRealUsedCoreNum(int64_t &used_corenum) override;
  bool SatisfyThresholdUBSize() override;
  double GetPerf() override;
};

double AxesReorderSolvercase1101::GetPerf() {
  double R = static_cast<double>(input_.input_vars[0]->value);
  double block_dim = 1;
  CalUsedCoreNum(block_dim);
  double nbo_size = static_cast<double>(input_.pure_mc_vars[0]->value);
  double nio_size = static_cast<double>(input_.local_buffer_vars[0]->value);
  double AIV_MTE2 = ((((2 * R * nio_size / (((24.1000003814697 / (block_dim)) + 8.47000026702881))) + 27.0100002288818) * 3 * ((nbo_size / (nio_size)))) + (((2 * R / (((24.1000003814697 / (block_dim)) + 8.47000026702881))) + 27.0100002288818) * 2 * ((nbo_size / (nio_size)))) + 1305.67004394531);
  double AIV_MTE3 = ((((2 * R * nio_size / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879) * 2 * ((nbo_size / (nio_size)))) + (((4 * nio_size / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879) * 2 * ((nbo_size / (nio_size)))) + 497.359985351562);
  double AIV_VEC = ((((8 * R * nio_size / ((-1 + R))) + 4) * 3 * ((nbo_size / (nio_size)))) + 37.3699989318848);
  return Max(Max(AIV_VEC, AIV_MTE2), AIV_MTE3);
}

  bool AxesReorderSolvercase1101::SatisfyThresholdUBSize() {
    return false;
  }
bool AxesReorderSolvercase1101::CalUsedCoreNum(double &used_core_num) {
  double nbo_size = static_cast<double>(input_.pure_mc_vars[0]->value);
  used_core_num = Max(0, ((128 / (nbo_size))));
  return true;
}
bool AxesReorderSolvercase1101::CalRealUsedCoreNum(int64_t &used_core_num) {
  double nbo_size = static_cast<double>(input_.pure_mc_vars[0]->value);
  used_core_num = Max(0, Ceiling((128 / (nbo_size))));
  return true;
}

/*
  Tensor used for tiling case 1101 is:
  tensor_0:x1Local_output_0
  tensor_1:x2Local_output_0
  tensor_10:mean_output_2
  tensor_2:biasLocal_output_0
  tensor_3:gammaLocal_output_0
  tensor_4:betaLocal_output_0
  tensor_5:mean_output_0
  tensor_6:rstd_output_1
  tensor_7:rstd_output_0
  tensor_8:y_output_0
  tensor_9:mean_output_1
*/
class TilingCase1101Impl : public TilingCaseImpl {
 public:
  TilingCase1101Impl(uint32_t corenum) : TilingCaseImpl(corenum) {

  }
 protected:
  void GetTilingData(TilingDataCopy &from_tiling, graph_normalTilingData &to_tiling) {
    to_tiling.set_R(from_tiling.get_R());
    to_tiling.set_nbo_size(from_tiling.get_nbo_size());
    to_tiling.set_nio_size(from_tiling.get_nio_size());
    to_tiling.set_block_dim(from_tiling.get_block_dim());
    to_tiling.set_KERNEL_INIT_BUFFER(from_tiling.get_KERNEL_INIT_BUFFER());
    to_tiling.set_Q0(from_tiling.get_Q0());
    to_tiling.set_Q1(from_tiling.get_Q1());
    to_tiling.set_Q2(from_tiling.get_Q2());
    to_tiling.set_Q3(from_tiling.get_Q3());
    to_tiling.set_Q4(from_tiling.get_Q4());
    to_tiling.set_Q5(from_tiling.get_Q5());
    to_tiling.set_Q6(from_tiling.get_Q6());
    to_tiling.set_Q7(from_tiling.get_Q7());
    to_tiling.set_Q8(from_tiling.get_Q8());
    to_tiling.set_Q9(from_tiling.get_Q9());
    to_tiling.set_gm_size(from_tiling.get_gm_size());
    to_tiling.set_nbo_loop_num(from_tiling.get_nbo_loop_num());
    to_tiling.set_nbo_tail_size(from_tiling.get_nbo_tail_size());
    to_tiling.set_nbo_tail_tile_nio_loop_num(from_tiling.get_nbo_tail_tile_nio_loop_num());
    to_tiling.set_nbo_tail_tile_nio_tail_size(from_tiling.get_nbo_tail_tile_nio_tail_size());
    to_tiling.set_nio_loop_num(from_tiling.get_nio_loop_num());
    to_tiling.set_nio_tail_size(from_tiling.get_nio_tail_size());
    to_tiling.set_output0_single_core_size(from_tiling.get_output0_single_core_size());
    to_tiling.set_output0_total_size(from_tiling.get_output0_total_size());
    to_tiling.set_output1_single_core_size(from_tiling.get_output1_single_core_size());
    to_tiling.set_output1_total_size(from_tiling.get_output1_total_size());
    to_tiling.set_output2_single_core_size(from_tiling.get_output2_single_core_size());
    to_tiling.set_output2_total_size(from_tiling.get_output2_total_size());
    to_tiling.set_output3_single_core_size(from_tiling.get_output3_single_core_size());
    to_tiling.set_output3_total_size(from_tiling.get_output3_total_size());
    to_tiling.set_tiling_key(from_tiling.get_tiling_key());

  }
  void SetTilingData(graph_normalTilingData &from_tiling, TilingDataCopy &to_tiling) {
    to_tiling.set_R(from_tiling.get_R());
    to_tiling.set_nbo_size(from_tiling.get_nbo_size());
    to_tiling.set_nio_size(from_tiling.get_nio_size());
    to_tiling.set_block_dim(from_tiling.get_block_dim());
    to_tiling.set_KERNEL_INIT_BUFFER(from_tiling.get_KERNEL_INIT_BUFFER());
    to_tiling.set_Q0(from_tiling.get_Q0());
    to_tiling.set_Q1(from_tiling.get_Q1());
    to_tiling.set_Q2(from_tiling.get_Q2());
    to_tiling.set_Q3(from_tiling.get_Q3());
    to_tiling.set_Q4(from_tiling.get_Q4());
    to_tiling.set_Q5(from_tiling.get_Q5());
    to_tiling.set_Q6(from_tiling.get_Q6());
    to_tiling.set_Q7(from_tiling.get_Q7());
    to_tiling.set_Q8(from_tiling.get_Q8());
    to_tiling.set_Q9(from_tiling.get_Q9());
    to_tiling.set_gm_size(from_tiling.get_gm_size());
    to_tiling.set_nbo_loop_num(from_tiling.get_nbo_loop_num());
    to_tiling.set_nbo_tail_size(from_tiling.get_nbo_tail_size());
    to_tiling.set_nbo_tail_tile_nio_loop_num(from_tiling.get_nbo_tail_tile_nio_loop_num());
    to_tiling.set_nbo_tail_tile_nio_tail_size(from_tiling.get_nbo_tail_tile_nio_tail_size());
    to_tiling.set_nio_loop_num(from_tiling.get_nio_loop_num());
    to_tiling.set_nio_tail_size(from_tiling.get_nio_tail_size());
    to_tiling.set_output0_single_core_size(from_tiling.get_output0_single_core_size());
    to_tiling.set_output0_total_size(from_tiling.get_output0_total_size());
    to_tiling.set_output1_single_core_size(from_tiling.get_output1_single_core_size());
    to_tiling.set_output1_total_size(from_tiling.get_output1_total_size());
    to_tiling.set_output2_single_core_size(from_tiling.get_output2_single_core_size());
    to_tiling.set_output2_total_size(from_tiling.get_output2_total_size());
    to_tiling.set_output3_single_core_size(from_tiling.get_output3_single_core_size());
    to_tiling.set_output3_total_size(from_tiling.get_output3_total_size());
    to_tiling.set_tiling_key(from_tiling.get_tiling_key());

  }
  bool ExecuteAxesReorderSolver(graph_normalTilingData& tiling_data) {
    Variable R;
    R.value = tiling_data.get_R();
    TilingVariable nbo_size;
    TilingVariable nio_size;
    int64_t ub_size = tiling_data.get_ub_size();
    Constraint cons0;
    auto cons0Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      double nio_size = rel_tiling_vars[0]->value;
      double R = rel_in_shapes[0]->value;
      double tensor_0 = (2 * R * nio_size);
      double tensor_1 = (2 * R * nio_size);
      double tensor_10 = (4 * R * nio_size);
      double tensor_2 = (2 * R * nio_size);
      double tensor_3 = (2 * R);
      double tensor_4 = (2 * R);
      double tensor_5 = (4 * nio_size);
      double tensor_6 = (4 * nio_size);
      double tensor_7 = (4 * R * nio_size);
      double tensor_8 = (2 * R * nio_size);
      double tensor_9 = (2 * R * nio_size);
      int64_t value = ((32 * Ceiling((Max(tensor_8, tensor_7) * Rational(1 , 32)))) + (32 * Ceiling((Rational(1 , 32) * tensor_0))) + (32 * Ceiling((Rational(1 , 32) * tensor_1))) + (32 * Ceiling((Rational(1 , 32) * tensor_10))) + (32 * Ceiling((Rational(1 , 32) * tensor_2))) + (32 * Ceiling((Rational(1 , 32) * tensor_3))) + (32 * Ceiling((Rational(1 , 32) * tensor_4))) + (32 * Ceiling((Rational(1 , 32) * tensor_5))) + (32 * Ceiling((Rational(1 , 32) * tensor_6))) + (32 * Ceiling((Rational(1 , 32) * tensor_9))) + 8192) - rel_hw_spec;
      return value;
    };
    TilingVariable* cons_0rel_tiling_vars[1] = {&nio_size, };
    cons0.rel_tiling_vars = cons_0rel_tiling_vars;
    cons0.rel_tiling_vars_size = 1u;
    Variable* cons_0rel_in_shapes[1] = {&R, };
    cons0.rel_in_shapes = cons_0rel_in_shapes;
    cons0.rel_in_shapes_size = 1u;
    cons0.rel_hw_spec = ub_size;
    cons0.type = ConstraintType::LOCAL_BUFFER;
    cons0.eval = cons0Eval;
    Constraint cons1;
    auto cons1Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      double nbo_size = rel_in_shapes[0]->value;
      int64_t value = (-128 + nbo_size);
      return value;
    };
    Variable* cons_1rel_in_shapes[1] = {&nbo_size, };
    cons1.rel_in_shapes = cons_1rel_in_shapes;
    cons1.rel_in_shapes_size = 1u;
    cons1.type = ConstraintType::MC_MIXED;
    cons1.eval = cons1Eval;
    Constraint cons2;
    auto cons2Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      double nio_size = rel_tiling_vars[0]->value;
      double nbo_size = rel_in_shapes[0]->value;
      int64_t value = (nio_size - nbo_size);
      return value;
    };
    TilingVariable* cons_2rel_tiling_vars[1] = {&nio_size, };
    cons2.rel_tiling_vars = cons_2rel_tiling_vars;
    cons2.rel_tiling_vars_size = 1u;
    Variable* cons_2rel_in_shapes[1] = {&nbo_size, };
    cons2.rel_in_shapes = cons_2rel_in_shapes;
    cons2.rel_in_shapes_size = 1u;
    cons2.type = ConstraintType::MC_MIXED;
    cons2.eval = cons2Eval;
    GetUpperBoundFuncPtr nbo_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      upper_bound *= 128;
      return upper_bound;
    };
    nbo_size.upper_bound = nbo_size_upper_bound;
    Constraint*nbo_size_rel_cons[2] = {&cons1, &cons2, };
    nbo_size.rel_cons = nbo_size_rel_cons;
    nbo_size.rel_cons_size = 2u;
    nio_size.align = 1;
    GetUpperBoundFuncPtr nio_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      double nbo_size = parent_vars[0]->value;
      if (parent_vars[0]->value == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= nbo_size;
      return upper_bound;
    };
    nio_size.upper_bound = nio_size_upper_bound;
    Variable* nio_size_upper_bound_vars[1] = {&nbo_size, };
    nio_size.upper_bound_vars = nio_size_upper_bound_vars;
    nio_size.upper_bound_vars_size = 1u;
    Constraint*nio_size_rel_cons[1] = {&cons2, };
    nio_size.rel_cons = nio_size_rel_cons;
    nio_size.rel_cons_size = 1u;
    AxesReorderSolverInput input;
    Variable* input_vars[3] = {&R, };
    input.input_vars = input_vars;
    input.input_vars_size = 3u;
    TilingVariable* tiling_vars[2] = {&nbo_size, &nio_size, };
    input.tiling_vars = tiling_vars;
    input.tiling_vars_size = 2u;
    Constraint* all_cons[3] = {&cons0, &cons1, &cons2, };
    input.all_cons_size = 3u;
    input.all_cons = all_cons;
    TilingVariable* pure_mc_vars[1] = {&nbo_size, };
    input.pure_mc_vars_size = 1u;
    input.pure_mc_vars = pure_mc_vars;
    TilingVariable* local_buffer_vars[1] = {&nio_size, };
    input.local_buffer_vars_size = 1u;
    input.local_buffer_vars = local_buffer_vars;
    input.core_num = corenum_;
    input.ub_threshold = 0.200000;
    input.corenum_threshold = 0.800000;
    input.ub_size = tiling_data.get_ub_size();
    AxesReorderSolvercase1101 solver(input);
    if (!solver.Run(false)) {
      return false;
    }
    tiling_data.set_nbo_size(input.pure_mc_vars[0]->value);
    tiling_data.set_nio_size(input.local_buffer_vars[0]->value);
    return true;
  }

  bool DoTiling(graph_normalTilingData &tiling_data) {
    OP_LOGI(OP_NAME, "[PROF]Set input params for tiling case 1101 of ScheduleResult0G0.  R = %u.", tiling_data.get_R());
    OP_LOGI(OP_NAME, "[PROF]Set ub_size for tiling case 1101 of ScheduleResult0G0 to ((32 * Ceiling((Max(tensor_8, tensor_7) * Rational(1 , 32)))) + (32 * Ceiling((Rational(1 , 32) * tensor_0))) + (32 * Ceiling((Rational(1");
    OP_LOGI(OP_NAME, " , 32) * tensor_1))) + (32 * Ceiling((Rational(1 , 32) * tensor_10))) + (32 * Ceiling((Rational(1 , 32) * tensor_2))) + (32 * Ceiling((Rational(1 , 32) * tensor_3))) + (32 * Ceiling((Rational(1 , 32) ");
    OP_LOGI(OP_NAME, "* tensor_4))) + (32 * Ceiling((Rational(1 , 32) * tensor_5))) + (32 * Ceiling((Rational(1 , 32) * tensor_6))) + (32 * Ceiling((Rational(1 , 32) * tensor_9))) + 8192)");
    OP_LOGI(OP_NAME, "[PROF]Set block_dim for tiling case 1101 of ScheduleResult0G0 to Max(0, Ceiling((128 / (nbo_size))))");

    OP_LOGD(OP_NAME, "[PROF]Set hardware params. ub_size = %u. block_dim = %u.", tiling_data.get_ub_size(), tiling_data.get_block_dim());
    uint32_t nbo_size = tiling_data.get_nbo_size();
    uint32_t nio_size = tiling_data.get_nio_size();

    if (!ExecuteAxesReorderSolver(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to execute axes reorder solver for tilingCaseId case1101.");
      return false;
    }
    OP_LOGD(OP_NAME, "Execute axes reorder solver for tilingCaseId case1101 successfully.");

    return true;
  }

void DoApiTiling(graph_normalTilingData &tiling_data) {
}
  void GeneralTiling(graph_normalTilingData &tiling_data) {
    double nbo_size = static_cast<double>(tiling_data.get_nbo_size());
    tiling_data.set_block_dim(Max(0, Ceiling((128 / (nbo_size)))));
  }

  int Getub_size(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double nio_size = tiling_data.get_nio_size();
    double tensor_0 = (2 * R * nio_size);
    double tensor_1 = (2 * R * nio_size);
    double tensor_10 = (4 * R * nio_size);
    double tensor_2 = (2 * R * nio_size);
    double tensor_3 = (2 * R);
    double tensor_4 = (2 * R);
    double tensor_5 = (4 * nio_size);
    double tensor_6 = (4 * nio_size);
    double tensor_7 = (4 * R * nio_size);
    double tensor_8 = (2 * R * nio_size);
    double tensor_9 = (2 * R * nio_size);

    return ((32 * Ceiling((Max(tensor_8, tensor_7) * Rational(1 , 32)))) + (32 * Ceiling((Rational(1 , 32) * tensor_0))) + (32 * Ceiling((Rational(1 , 32) * tensor_1))) + (32 * Ceiling((Rational(1 , 32) * tensor_10))) + (32 * Ceiling((Rational(1 , 32) * tensor_2))) + (32 * Ceiling((Rational(1 , 32) * tensor_3))) + (32 * Ceiling((Rational(1 , 32) * tensor_4))) + (32 * Ceiling((Rational(1 , 32) * tensor_5))) + (32 * Ceiling((Rational(1 , 32) * tensor_6))) + (32 * Ceiling((Rational(1 , 32) * tensor_9))) + 8192);
  }

  int Getblock_dim(graph_normalTilingData& tiling_data) {
    double nbo_size = tiling_data.get_nbo_size();

    return Max(0, Ceiling((128 / (nbo_size))));
  }

  double GetAIV_MTE2(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double block_dim = tiling_data.get_block_dim();
    double nbo_size = tiling_data.get_nbo_size();
    double nio_size = tiling_data.get_nio_size();

    return ((((2 * R * nio_size / (((24.1000003814697 / (block_dim)) + 8.47000026702881))) + 27.0100002288818) * 3 * Ceiling((nbo_size / (nio_size)))) + (((2 * R / (((24.1000003814697 / (block_dim)) + 8.47000026702881))) + 27.0100002288818) * 2 * Ceiling((nbo_size / (nio_size)))) + 1305.67004394531);
  }

  double GetAIV_MTE3(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double block_dim = tiling_data.get_block_dim();
    double nbo_size = tiling_data.get_nbo_size();
    double nio_size = tiling_data.get_nio_size();

    return ((((2 * R * nio_size / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879) * 2 * Ceiling((nbo_size / (nio_size)))) + (((4 * nio_size / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879) * 2 * Ceiling((nbo_size / (nio_size)))) + 497.359985351562);
  }

  double GetAIV_VEC(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double nbo_size = tiling_data.get_nbo_size();
    double nio_size = tiling_data.get_nio_size();

    return ((((8 * R * nio_size / ((-1 + R))) + 4) * 3 * Ceiling((nbo_size / (nio_size)))) + 37.3699989318848);
  }

  double GetPerf(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double block_dim = tiling_data.get_block_dim();
    double nbo_size = tiling_data.get_nbo_size();
    double nio_size = tiling_data.get_nio_size();

    double AIV_MTE2 = ((((2 * R * nio_size / (((24.1000003814697 / (block_dim)) + 8.47000026702881))) + 27.0100002288818) * 3 * Ceiling((nbo_size / (nio_size)))) + (((2 * R / (((24.1000003814697 / (block_dim)) + 8.47000026702881))) + 27.0100002288818) * 2 * Ceiling((nbo_size / (nio_size)))) + 1305.67004394531);
    double AIV_MTE3 = ((((2 * R * nio_size / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879) * 2 * Ceiling((nbo_size / (nio_size)))) + (((4 * nio_size / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879) * 2 * Ceiling((nbo_size / (nio_size)))) + 497.359985351562);
    double AIV_VEC = ((((8 * R * nio_size / ((-1 + R))) + 4) * 3 * Ceiling((nbo_size / (nio_size)))) + 37.3699989318848);

    return Max(Max(AIV_VEC, AIV_MTE2), AIV_MTE3);
  }

  void UpdateAxesTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_nio_loop_num(((tiling_data.get_nbo_size() + tiling_data.get_nio_size()) - 1) / tiling_data.get_nio_size());
    tiling_data.set_nbo_loop_num(((128 + tiling_data.get_nbo_size()) - 1) / tiling_data.get_nbo_size());
    tiling_data.set_nio_tail_size((tiling_data.get_nbo_size() % tiling_data.get_nio_size()) == 0 ? tiling_data.get_nio_size() : (tiling_data.get_nbo_size() % tiling_data.get_nio_size()));
    tiling_data.set_nbo_tail_size((128 % tiling_data.get_nbo_size()) == 0 ? tiling_data.get_nbo_size() : (128 % tiling_data.get_nbo_size()));
    tiling_data.set_nbo_tail_tile_nio_loop_num(((tiling_data.get_nbo_tail_size() + tiling_data.get_nio_size()) - 1) / tiling_data.get_nio_size());
    tiling_data.set_nbo_tail_tile_nio_tail_size((tiling_data.get_nbo_tail_size() % tiling_data.get_nio_size()) == 0 ? tiling_data.get_nio_size() : (tiling_data.get_nbo_tail_size() % tiling_data.get_nio_size()));
  }

  void SetKERNEL_INIT_BUFFER(graph_normalTilingData &tiling_data) {
    tiling_data.set_KERNEL_INIT_BUFFER(8192);
  }

  void SetQ0(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    const auto tensor_0 = (2 * R * nio_size);
    tiling_data.set_Q0((32 * Ceiling((Rational(1 , 32) * tensor_0))));
  }

  void SetQ1(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    const auto tensor_1 = (2 * R * nio_size);
    tiling_data.set_Q1((32 * Ceiling((Rational(1 , 32) * tensor_1))));
  }

  void SetQ2(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    const auto tensor_2 = (2 * R * nio_size);
    tiling_data.set_Q2((32 * Ceiling((Rational(1 , 32) * tensor_2))));
  }

  void SetQ3(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto tensor_3 = (2 * R);
    tiling_data.set_Q3((32 * Ceiling((Rational(1 , 32) * tensor_3))));
  }

  void SetQ4(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto tensor_4 = (2 * R);
    tiling_data.set_Q4((32 * Ceiling((Rational(1 , 32) * tensor_4))));
  }

  void SetQ5(graph_normalTilingData &tiling_data) {
    const auto nio_size = tiling_data.get_nio_size();
    const auto tensor_5 = (4 * nio_size);
    tiling_data.set_Q5((32 * Ceiling((Rational(1 , 32) * tensor_5))));
  }

  void SetQ6(graph_normalTilingData &tiling_data) {
    const auto nio_size = tiling_data.get_nio_size();
    const auto tensor_6 = (4 * nio_size);
    tiling_data.set_Q6((32 * Ceiling((Rational(1 , 32) * tensor_6))));
  }

  void SetQ7(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    const auto tensor_7 = (4 * R * nio_size);
    const auto tensor_8 = (2 * R * nio_size);
    tiling_data.set_Q7((32 * Ceiling((Max(tensor_8, tensor_7) * Rational(1 , 32)))));
  }

  void SetQ8(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    const auto tensor_9 = (2 * R * nio_size);
    tiling_data.set_Q8((32 * Ceiling((Rational(1 , 32) * tensor_9))));
  }

  void SetQ9(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto nio_size = tiling_data.get_nio_size();
    const auto tensor_10 = (4 * R * nio_size);
    tiling_data.set_Q9((32 * Ceiling((Rational(1 , 32) * tensor_10))));
  }

  void ComputeOptionParam(graph_normalTilingData &tiling_data) {
    SetKERNEL_INIT_BUFFER(tiling_data);
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

  void ExtraTilingData(graph_normalTilingData &tiling_data) {
    OP_LOGD(OP_NAME, "Start executing extra tiling for tilingCaseId 1101.");
    ComputeOptionParam(tiling_data);
		UpdateAxesTilingData(tiling_data);

    OP_LOGD(OP_NAME, "Execute extra tiling for tilingCaseId 1101 successfully.");
  }

  void GetWorkSpaceSize(graph_normalTilingData& tiling_data) {
    OP_LOGD(OP_NAME, "Start setting workspace for case 1101.");
    tiling_data.set_workspaceSize(static_cast<uint32_t>(0));
    OP_LOGD(OP_NAME, "Setting workspace to %u for case 1101.", tiling_data.get_workspaceSize());
  }

  void TilingSummary(graph_normalTilingData &tiling_data, double& cur_ub_ratio) {
    OP_LOGI(OP_NAME, "The value of nbo_size is %u in schedule_result0_g0.", tiling_data.get_nbo_size());
    OP_LOGI(OP_NAME, "The value of nio_size is %u in schedule_result0_g0.", tiling_data.get_nio_size());
    OP_LOGI(OP_NAME, "The value of ub_size is %d in schedule_result0_g0.", Getub_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of block_dim is %d in schedule_result0_g0.", Getblock_dim(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIV_MTE2 is %f.", GetAIV_MTE2(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIV_MTE3 is %f.", GetAIV_MTE3(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIV_VEC is %f.", GetAIV_VEC(tiling_data));
    OP_LOGI(OP_NAME, "The objective value of the tiling data is %f.", GetPerf(tiling_data));
    cur_ub_ratio = static_cast<double>(Getub_size(tiling_data)) / tiling_data.get_ub_size();
    if (std::isnan(cur_ub_ratio)) {
      cur_ub_ratio = 1;
      OP_LOGI(OP_NAME, "The ub ratio is NaN, set it to 1.");
    }
  }

};

class AxesReorderSolvercase1111 : public AxesReorderSolver {
 public:
  explicit AxesReorderSolvercase1111(const AxesReorderSolverInput input) : AxesReorderSolver(input) {}
  ~AxesReorderSolvercase1111() = default;
  bool CalUsedCoreNum(double &used_core_num) override;
  bool CalRealUsedCoreNum(int64_t &used_corenum) override;
  bool SatisfyThresholdUBSize() override;
  double GetPerf() override;
};

double AxesReorderSolvercase1111::GetPerf() {
  double A = static_cast<double>(input_.input_vars[0]->value);
  double R = static_cast<double>(input_.input_vars[1]->value);
  double block_dim = 1;
  CalUsedCoreNum(block_dim);
  double sbo_size = static_cast<double>(input_.pure_mc_vars[0]->value);
  double sio_size = static_cast<double>(input_.local_buffer_vars[0]->value);
  double AIV_MTE2 = ((((2 * sio_size / (((24.1000003814697 / (block_dim)) + 8.47000026702881))) + 27.0100002288818) * 5 * ((R / (sio_size))) * sbo_size) + 1305.67004394531);
  double AIV_MTE3 = ((((2 * sio_size / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879) * 2 * ((R / (sio_size))) * sbo_size) + (((4 * sio_size / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879) * 2 * ((R / (sio_size))) * sbo_size) + 497.359985351562);
  double AIV_VEC = ((((8 * sio_size / ((-1 + sio_size))) + 4) * 3 * ((R / (sio_size))) * sbo_size) + 37.3699989318848);
  return Max(Max(AIV_VEC, AIV_MTE2), AIV_MTE3);
}

  bool AxesReorderSolvercase1111::SatisfyThresholdUBSize() {
    return false;
  }
bool AxesReorderSolvercase1111::CalUsedCoreNum(double &used_core_num) {
  double A = static_cast<double>(input_.input_vars[0]->value);
  double sbo_size = static_cast<double>(input_.pure_mc_vars[0]->value);
  used_core_num = Max(0, ((A / (sbo_size))));
  return true;
}
bool AxesReorderSolvercase1111::CalRealUsedCoreNum(int64_t &used_core_num) {
  double A = static_cast<double>(input_.input_vars[0]->value);
  double sbo_size = static_cast<double>(input_.pure_mc_vars[0]->value);
  used_core_num = Max(0, Ceiling((A / (sbo_size))));
  return true;
}

/*
  Tensor used for tiling case 1111 is:
  tensor_0:x1Local_output_0
  tensor_1:x2Local_output_0
  tensor_10:rstd_output_1
  tensor_2:biasLocal_output_0
  tensor_3:mean_output_1
  tensor_4:mean_output_2
  tensor_5:rstd_output_0
  tensor_6:y_output_0
  tensor_7:betaLocal_output_0
  tensor_8:gammaLocal_output_0
  tensor_9:mean_output_0
*/
class TilingCase1111Impl : public TilingCaseImpl {
 public:
  TilingCase1111Impl(uint32_t corenum) : TilingCaseImpl(corenum) {

  }
 protected:
  void GetTilingData(TilingDataCopy &from_tiling, graph_normalTilingData &to_tiling) {
    to_tiling.set_A(from_tiling.get_A());
    to_tiling.set_R(from_tiling.get_R());
    to_tiling.set_sbo_size(from_tiling.get_sbo_size());
    to_tiling.set_sio_size(from_tiling.get_sio_size());
    to_tiling.set_block_dim(from_tiling.get_block_dim());
    to_tiling.set_KERNEL_INIT_BUFFER(from_tiling.get_KERNEL_INIT_BUFFER());
    to_tiling.set_Q0(from_tiling.get_Q0());
    to_tiling.set_Q1(from_tiling.get_Q1());
    to_tiling.set_Q2(from_tiling.get_Q2());
    to_tiling.set_Q3(from_tiling.get_Q3());
    to_tiling.set_Q4(from_tiling.get_Q4());
    to_tiling.set_Q5(from_tiling.get_Q5());
    to_tiling.set_Q6(from_tiling.get_Q6());
    to_tiling.set_Q7(from_tiling.get_Q7());
    to_tiling.set_Q8(from_tiling.get_Q8());
    to_tiling.set_gm_size(from_tiling.get_gm_size());
    to_tiling.set_output0_single_core_size(from_tiling.get_output0_single_core_size());
    to_tiling.set_output0_total_size(from_tiling.get_output0_total_size());
    to_tiling.set_output1_single_core_size(from_tiling.get_output1_single_core_size());
    to_tiling.set_output1_total_size(from_tiling.get_output1_total_size());
    to_tiling.set_output2_single_core_size(from_tiling.get_output2_single_core_size());
    to_tiling.set_output2_total_size(from_tiling.get_output2_total_size());
    to_tiling.set_output3_single_core_size(from_tiling.get_output3_single_core_size());
    to_tiling.set_output3_total_size(from_tiling.get_output3_total_size());
    to_tiling.set_sbo_loop_num(from_tiling.get_sbo_loop_num());
    to_tiling.set_sbo_tail_size(from_tiling.get_sbo_tail_size());
    to_tiling.set_sio_loop_num(from_tiling.get_sio_loop_num());
    to_tiling.set_sio_tail_size(from_tiling.get_sio_tail_size());
    to_tiling.set_tiling_key(from_tiling.get_tiling_key());

  }
  void SetTilingData(graph_normalTilingData &from_tiling, TilingDataCopy &to_tiling) {
    to_tiling.set_A(from_tiling.get_A());
    to_tiling.set_R(from_tiling.get_R());
    to_tiling.set_sbo_size(from_tiling.get_sbo_size());
    to_tiling.set_sio_size(from_tiling.get_sio_size());
    to_tiling.set_block_dim(from_tiling.get_block_dim());
    to_tiling.set_KERNEL_INIT_BUFFER(from_tiling.get_KERNEL_INIT_BUFFER());
    to_tiling.set_Q0(from_tiling.get_Q0());
    to_tiling.set_Q1(from_tiling.get_Q1());
    to_tiling.set_Q2(from_tiling.get_Q2());
    to_tiling.set_Q3(from_tiling.get_Q3());
    to_tiling.set_Q4(from_tiling.get_Q4());
    to_tiling.set_Q5(from_tiling.get_Q5());
    to_tiling.set_Q6(from_tiling.get_Q6());
    to_tiling.set_Q7(from_tiling.get_Q7());
    to_tiling.set_Q8(from_tiling.get_Q8());
    to_tiling.set_gm_size(from_tiling.get_gm_size());
    to_tiling.set_output0_single_core_size(from_tiling.get_output0_single_core_size());
    to_tiling.set_output0_total_size(from_tiling.get_output0_total_size());
    to_tiling.set_output1_single_core_size(from_tiling.get_output1_single_core_size());
    to_tiling.set_output1_total_size(from_tiling.get_output1_total_size());
    to_tiling.set_output2_single_core_size(from_tiling.get_output2_single_core_size());
    to_tiling.set_output2_total_size(from_tiling.get_output2_total_size());
    to_tiling.set_output3_single_core_size(from_tiling.get_output3_single_core_size());
    to_tiling.set_output3_total_size(from_tiling.get_output3_total_size());
    to_tiling.set_sbo_loop_num(from_tiling.get_sbo_loop_num());
    to_tiling.set_sbo_tail_size(from_tiling.get_sbo_tail_size());
    to_tiling.set_sio_loop_num(from_tiling.get_sio_loop_num());
    to_tiling.set_sio_tail_size(from_tiling.get_sio_tail_size());
    to_tiling.set_tiling_key(from_tiling.get_tiling_key());

  }
  bool ExecuteAxesReorderSolver(graph_normalTilingData& tiling_data) {
    Variable A;
    A.value = tiling_data.get_A();
    Variable R;
    R.value = (tiling_data.get_R() + 16 - 1) / 16 * 16;
    TilingVariable sbo_size;
    TilingVariable sio_size;
    int64_t ub_size = tiling_data.get_ub_size();
    Constraint cons0;
    auto cons0Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      double sio_size = rel_tiling_vars[0]->value;
      double R = rel_in_shapes[0]->value;
      double tensor_0 = (2 * sio_size);
      double tensor_1 = (2 * sio_size);
      double tensor_10 = 4;
      double tensor_2 = (2 * sio_size);
      double tensor_3 = (2 * sio_size);
      double tensor_4 = (4 * R);
      double tensor_5 = (4 * R);
      double tensor_6 = (2 * R);
      double tensor_7 = (2 * R);
      double tensor_8 = (2 * R);
      double tensor_9 = 4;
      int64_t value = ((32 * Ceiling((Max(Max(tensor_4, tensor_5), tensor_6) * Rational(1 , 32)))) + (32 * Ceiling((Rational(1 , 32) * tensor_0))) + (32 * Ceiling((Rational(1 , 32) * tensor_1))) + (32 * Ceiling((Rational(1 , 32) * tensor_10))) + (32 * Ceiling((Rational(1 , 32) * tensor_2))) + (32 * Ceiling((Rational(1 , 32) * tensor_3))) + (32 * Ceiling((Rational(1 , 32) * tensor_7))) + (32 * Ceiling((Rational(1 , 32) * tensor_8))) + (32 * Ceiling((Rational(1 , 32) * tensor_9))) + 8192) - rel_hw_spec;
      return value;
    };
    TilingVariable* cons_0rel_tiling_vars[1] = {&sio_size, };
    cons0.rel_tiling_vars = cons_0rel_tiling_vars;
    cons0.rel_tiling_vars_size = 1u;
    Variable* cons_0rel_in_shapes[1] = {&R, };
    cons0.rel_in_shapes = cons_0rel_in_shapes;
    cons0.rel_in_shapes_size = 1u;
    cons0.rel_hw_spec = ub_size;
    cons0.type = ConstraintType::LOCAL_BUFFER;
    cons0.eval = cons0Eval;
    Constraint cons1;
    auto cons1Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      double A = rel_in_shapes[0]->value;
      double sbo_size = rel_in_shapes[1]->value;
      int64_t value = (sbo_size - A);
      return value;
    };
    Variable* cons_1rel_in_shapes[2] = {&A, &sbo_size, };
    cons1.rel_in_shapes = cons_1rel_in_shapes;
    cons1.rel_in_shapes_size = 2u;
    cons1.type = ConstraintType::MC_MIXED;
    cons1.eval = cons1Eval;
    Constraint cons2;
    auto cons2Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      double sio_size = rel_tiling_vars[0]->value;
      double R = rel_in_shapes[0]->value;
      int64_t value = (sio_size - R);
      return value;
    };
    TilingVariable* cons_2rel_tiling_vars[1] = {&sio_size, };
    cons2.rel_tiling_vars = cons_2rel_tiling_vars;
    cons2.rel_tiling_vars_size = 1u;
    Variable* cons_2rel_in_shapes[1] = {&R, };
    cons2.rel_in_shapes = cons_2rel_in_shapes;
    cons2.rel_in_shapes_size = 1u;
    cons2.type = ConstraintType::LB_MIXED;
    cons2.eval = cons2Eval;
    GetUpperBoundFuncPtr sbo_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      double A = parent_vars[0]->value;
      if (parent_vars[0]->value == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= A;
      return upper_bound;
    };
    sbo_size.upper_bound = sbo_size_upper_bound;
    Variable* sbo_size_upper_bound_vars[1] = {&A, };
    sbo_size.upper_bound_vars = sbo_size_upper_bound_vars;
    sbo_size.upper_bound_vars_size = 1u;
    Constraint*sbo_size_rel_cons[1] = {&cons1, };
    sbo_size.rel_cons = sbo_size_rel_cons;
    sbo_size.rel_cons_size = 1u;
    sio_size.align = 16;
    GetUpperBoundFuncPtr sio_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      double R = parent_vars[0]->value;
      if (parent_vars[0]->value == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= R;
      return upper_bound;
    };
    sio_size.upper_bound = sio_size_upper_bound;
    Variable* sio_size_upper_bound_vars[1] = {&R, };
    sio_size.upper_bound_vars = sio_size_upper_bound_vars;
    sio_size.upper_bound_vars_size = 1u;
    Constraint*sio_size_rel_cons[1] = {&cons2, };
    sio_size.rel_cons = sio_size_rel_cons;
    sio_size.rel_cons_size = 1u;
    AxesReorderSolverInput input;
    Variable* input_vars[2] = {&A, &R, };
    input.input_vars = input_vars;
    input.input_vars_size = 2u;
    TilingVariable* tiling_vars[2] = {&sbo_size, &sio_size, };
    input.tiling_vars = tiling_vars;
    input.tiling_vars_size = 2u;
    Constraint* all_cons[3] = {&cons0, &cons1, &cons2, };
    input.all_cons_size = 3u;
    input.all_cons = all_cons;
    TilingVariable* pure_mc_vars[1] = {&sbo_size, };
    input.pure_mc_vars_size = 1u;
    input.pure_mc_vars = pure_mc_vars;
    TilingVariable* local_buffer_vars[1] = {&sio_size, };
    input.local_buffer_vars_size = 1u;
    input.local_buffer_vars = local_buffer_vars;
    input.core_num = corenum_;
    input.ub_threshold = 0.200000;
    input.corenum_threshold = 0.800000;
    input.ub_size = tiling_data.get_ub_size();
    AxesReorderSolvercase1111 solver(input);
    if (!solver.Run(false)) {
      return false;
    }
    tiling_data.set_sbo_size(input.pure_mc_vars[0]->value);
    tiling_data.set_sio_size(input.local_buffer_vars[0]->value);
    return true;
  }

  bool DoTiling(graph_normalTilingData &tiling_data) {
    OP_LOGI(OP_NAME, "[PROF]Set input params for tiling case 1111 of ScheduleResult0G0.  A = %u. R = %u.", tiling_data.get_A(), tiling_data.get_R());
    OP_LOGI(OP_NAME, "[PROF]Set ub_size for tiling case 1111 of ScheduleResult0G0 to ((32 * Ceiling((Max(Max(tensor_4, tensor_5), tensor_6) * Rational(1 , 32)))) + (32 * Ceiling((Rational(1 , 32) * tensor_0))) + (32 * Ceil");
    OP_LOGI(OP_NAME, "ing((Rational(1 , 32) * tensor_1))) + (32 * Ceiling((Rational(1 , 32) * tensor_10))) + (32 * Ceiling((Rational(1 , 32) * tensor_2))) + (32 * Ceiling((Rational(1 , 32) * tensor_3))) + (32 * Ceiling((Ra");
    OP_LOGI(OP_NAME, "tional(1 , 32) * tensor_7))) + (32 * Ceiling((Rational(1 , 32) * tensor_8))) + (32 * Ceiling((Rational(1 , 32) * tensor_9))) + 8192)");
    OP_LOGI(OP_NAME, "[PROF]Set block_dim for tiling case 1111 of ScheduleResult0G0 to Max(0, Ceiling((A / (sbo_size))))");

    OP_LOGD(OP_NAME, "[PROF]Set hardware params. ub_size = %u. block_dim = %u.", tiling_data.get_ub_size(), tiling_data.get_block_dim());
    uint32_t sbo_size = tiling_data.get_sbo_size();
    uint32_t sio_size = tiling_data.get_sio_size();

    if (!ExecuteAxesReorderSolver(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to execute axes reorder solver for tilingCaseId case1111.");
      return false;
    }
    OP_LOGD(OP_NAME, "Execute axes reorder solver for tilingCaseId case1111 successfully.");

    return true;
  }

void DoApiTiling(graph_normalTilingData &tiling_data) {
}
  void GeneralTiling(graph_normalTilingData &tiling_data) {
    double A = static_cast<double>(tiling_data.get_A());
    double sbo_size = static_cast<double>(tiling_data.get_sbo_size());
    tiling_data.set_block_dim(Max(0, Ceiling((A / (sbo_size)))));
  }

  int Getub_size(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double sio_size = tiling_data.get_sio_size();
    double tensor_0 = (2 * sio_size);
    double tensor_1 = (2 * sio_size);
    double tensor_10 = 4;
    double tensor_2 = (2 * sio_size);
    double tensor_3 = (2 * sio_size);
    double tensor_4 = (4 * R);
    double tensor_5 = (4 * R);
    double tensor_6 = (2 * R);
    double tensor_7 = (2 * R);
    double tensor_8 = (2 * R);
    double tensor_9 = 4;

    return ((32 * Ceiling((Max(Max(tensor_4, tensor_5), tensor_6) * Rational(1 , 32)))) + (32 * Ceiling((Rational(1 , 32) * tensor_0))) + (32 * Ceiling((Rational(1 , 32) * tensor_1))) + (32 * Ceiling((Rational(1 , 32) * tensor_10))) + (32 * Ceiling((Rational(1 , 32) * tensor_2))) + (32 * Ceiling((Rational(1 , 32) * tensor_3))) + (32 * Ceiling((Rational(1 , 32) * tensor_7))) + (32 * Ceiling((Rational(1 , 32) * tensor_8))) + (32 * Ceiling((Rational(1 , 32) * tensor_9))) + 8192);
  }

  int Getblock_dim(graph_normalTilingData& tiling_data) {
    double A = tiling_data.get_A();
    double sbo_size = tiling_data.get_sbo_size();

    return Max(0, Ceiling((A / (sbo_size))));
  }

  double GetAIV_MTE2(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double block_dim = tiling_data.get_block_dim();
    double sbo_size = tiling_data.get_sbo_size();
    double sio_size = tiling_data.get_sio_size();

    return ((((2 * sio_size / (((24.1000003814697 / (block_dim)) + 8.47000026702881))) + 27.0100002288818) * 5 * Ceiling((R / (sio_size))) * sbo_size) + 1305.67004394531);
  }

  double GetAIV_MTE3(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double block_dim = tiling_data.get_block_dim();
    double sbo_size = tiling_data.get_sbo_size();
    double sio_size = tiling_data.get_sio_size();

    return ((((2 * sio_size / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879) * 2 * Ceiling((R / (sio_size))) * sbo_size) + (((4 * sio_size / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879) * 2 * Ceiling((R / (sio_size))) * sbo_size) + 497.359985351562);
  }

  double GetAIV_VEC(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double sbo_size = tiling_data.get_sbo_size();
    double sio_size = tiling_data.get_sio_size();

    return ((((8 * sio_size / ((-1 + sio_size))) + 4) * 3 * Ceiling((R / (sio_size))) * sbo_size) + 37.3699989318848);
  }

  double GetPerf(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double block_dim = tiling_data.get_block_dim();
    double sbo_size = tiling_data.get_sbo_size();
    double sio_size = tiling_data.get_sio_size();

    double AIV_MTE2 = ((((2 * sio_size / (((24.1000003814697 / (block_dim)) + 8.47000026702881))) + 27.0100002288818) * 5 * Ceiling((R / (sio_size))) * sbo_size) + 1305.67004394531);
    double AIV_MTE3 = ((((2 * sio_size / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879) * 2 * Ceiling((R / (sio_size))) * sbo_size) + (((4 * sio_size / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879) * 2 * Ceiling((R / (sio_size))) * sbo_size) + 497.359985351562);
    double AIV_VEC = ((((8 * sio_size / ((-1 + sio_size))) + 4) * 3 * Ceiling((R / (sio_size))) * sbo_size) + 37.3699989318848);

    return Max(Max(AIV_VEC, AIV_MTE2), AIV_MTE3);
  }

  void UpdateAxesTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_sio_loop_num(((tiling_data.get_R() + tiling_data.get_sio_size()) - 1) / tiling_data.get_sio_size());
    tiling_data.set_sbo_loop_num(((tiling_data.get_A() + tiling_data.get_sbo_size()) - 1) / tiling_data.get_sbo_size());
    tiling_data.set_sio_tail_size((tiling_data.get_R() % tiling_data.get_sio_size()) == 0 ? tiling_data.get_sio_size() : (tiling_data.get_R() % tiling_data.get_sio_size()));
    tiling_data.set_sbo_tail_size((tiling_data.get_A() % tiling_data.get_sbo_size()) == 0 ? tiling_data.get_sbo_size() : (tiling_data.get_A() % tiling_data.get_sbo_size()));
  }

  void SetKERNEL_INIT_BUFFER(graph_normalTilingData &tiling_data) {
    tiling_data.set_KERNEL_INIT_BUFFER(8192);
  }

  void SetQ0(graph_normalTilingData &tiling_data) {
    const auto sio_size = tiling_data.get_sio_size();
    const auto tensor_0 = (2 * sio_size);
    tiling_data.set_Q0((32 * Ceiling((Rational(1 , 32) * tensor_0))));
  }

  void SetQ1(graph_normalTilingData &tiling_data) {
    const auto sio_size = tiling_data.get_sio_size();
    const auto tensor_1 = (2 * sio_size);
    tiling_data.set_Q1((32 * Ceiling((Rational(1 , 32) * tensor_1))));
  }

  void SetQ2(graph_normalTilingData &tiling_data) {
    const auto sio_size = tiling_data.get_sio_size();
    const auto tensor_2 = (2 * sio_size);
    tiling_data.set_Q2((32 * Ceiling((Rational(1 , 32) * tensor_2))));
  }

  void SetQ3(graph_normalTilingData &tiling_data) {
    const auto sio_size = tiling_data.get_sio_size();
    const auto tensor_3 = (2 * sio_size);
    tiling_data.set_Q3((32 * Ceiling((Rational(1 , 32) * tensor_3))));
  }

  void SetQ4(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto tensor_4 = (4 * R);
    const auto tensor_5 = (4 * R);
    const auto tensor_6 = (2 * R);
    tiling_data.set_Q4((32 * Ceiling((Max(Max(tensor_4, tensor_5), tensor_6) * Rational(1 , 32)))));
  }

  void SetQ5(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto tensor_7 = (2 * R);
    tiling_data.set_Q5((32 * Ceiling((Rational(1 , 32) * tensor_7))));
  }

  void SetQ6(graph_normalTilingData &tiling_data) {
    const auto R = tiling_data.get_R();
    const auto tensor_8 = (2 * R);
    tiling_data.set_Q6((32 * Ceiling((Rational(1 , 32) * tensor_8))));
  }

  void SetQ7(graph_normalTilingData &tiling_data) {
    const auto tensor_9 = 4;
    tiling_data.set_Q7((32 * Ceiling((Rational(1 , 32) * tensor_9))));
  }

  void SetQ8(graph_normalTilingData &tiling_data) {
    const auto tensor_10 = 4;
    tiling_data.set_Q8((32 * Ceiling((Rational(1 , 32) * tensor_10))));
  }

  void ComputeOptionParam(graph_normalTilingData &tiling_data) {
    SetKERNEL_INIT_BUFFER(tiling_data);
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

  void ExtraTilingData(graph_normalTilingData &tiling_data) {
    OP_LOGD(OP_NAME, "Start executing extra tiling for tilingCaseId 1111.");
    ComputeOptionParam(tiling_data);
		UpdateAxesTilingData(tiling_data);

    OP_LOGD(OP_NAME, "Execute extra tiling for tilingCaseId 1111 successfully.");
  }

  void GetWorkSpaceSize(graph_normalTilingData& tiling_data) {
    OP_LOGD(OP_NAME, "Start setting workspace for case 1111.");
    tiling_data.set_workspaceSize(static_cast<uint32_t>(0));
    OP_LOGD(OP_NAME, "Setting workspace to %u for case 1111.", tiling_data.get_workspaceSize());
  }

  void TilingSummary(graph_normalTilingData &tiling_data, double& cur_ub_ratio) {
    OP_LOGI(OP_NAME, "The value of sbo_size is %u in schedule_result0_g0.", tiling_data.get_sbo_size());
    OP_LOGI(OP_NAME, "The value of sio_size is %u in schedule_result0_g0.", tiling_data.get_sio_size());
    OP_LOGI(OP_NAME, "The value of ub_size is %d in schedule_result0_g0.", Getub_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of block_dim is %d in schedule_result0_g0.", Getblock_dim(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIV_MTE2 is %f.", GetAIV_MTE2(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIV_MTE3 is %f.", GetAIV_MTE3(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIV_VEC is %f.", GetAIV_VEC(tiling_data));
    OP_LOGI(OP_NAME, "The objective value of the tiling data is %f.", GetPerf(tiling_data));
    cur_ub_ratio = static_cast<double>(Getub_size(tiling_data)) / tiling_data.get_ub_size();
    if (std::isnan(cur_ub_ratio)) {
      cur_ub_ratio = 1;
      OP_LOGI(OP_NAME, "The ub ratio is NaN, set it to 1.");
    }
  }

};

class AxesReorderSolvercase1151 : public AxesReorderSolver {
 public:
  explicit AxesReorderSolvercase1151(const AxesReorderSolverInput input) : AxesReorderSolver(input) {}
  ~AxesReorderSolvercase1151() = default;
  bool CalUsedCoreNum(double &used_core_num) override;
  bool CalRealUsedCoreNum(int64_t &used_corenum) override;
  bool SatisfyThresholdUBSize() override;
  double GetPerf() override;
};

double AxesReorderSolvercase1151::GetPerf() {
  double A = static_cast<double>(input_.input_vars[0]->value);
  double R = static_cast<double>(input_.input_vars[1]->value);
  double block_dim = 1;
  CalUsedCoreNum(block_dim);
  double wbo_size = static_cast<double>(input_.pure_mc_vars[0]->value);
  double wio_size = static_cast<double>(input_.local_buffer_vars[0]->value);
  double AIV_MTE2 = ((((2 * wio_size / (((24.1000003814697 / (block_dim)) + 8.47000026702881))) + 27.0100002288818) * 5 * ((R / (wio_size))) * wbo_size) + (((4 * wio_size / (((24.1000003814697 / (block_dim)) + 8.47000026702881))) + 27.0100002288818) * ((R / (wio_size))) * wbo_size) + 1305.67004394531);
  double AIV_MTE3 = ((((2 * wio_size / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879) * 2 * ((R / (wio_size))) * wbo_size) + (((4 * wio_size / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879) * 3 * ((R / (wio_size))) * wbo_size) + 497.359985351562);
  double AIV_VEC = ((((4 * wio_size / ((-1 + wio_size))) + 4) * ((R / (wio_size))) * wbo_size) + (((8 * wio_size / ((-1 + wio_size))) + 4) * 2 * ((R / (wio_size))) * wbo_size) + 37.3699989318848);
  return Max(Max(AIV_VEC, AIV_MTE2), AIV_MTE3);
}

  bool AxesReorderSolvercase1151::SatisfyThresholdUBSize() {
    return false;
  }
bool AxesReorderSolvercase1151::CalUsedCoreNum(double &used_core_num) {
  double A = static_cast<double>(input_.input_vars[0]->value);
  double wbo_size = static_cast<double>(input_.pure_mc_vars[0]->value);
  used_core_num = Max(0, ((A / (wbo_size))));
  return true;
}
bool AxesReorderSolvercase1151::CalRealUsedCoreNum(int64_t &used_core_num) {
  double A = static_cast<double>(input_.input_vars[0]->value);
  double wbo_size = static_cast<double>(input_.pure_mc_vars[0]->value);
  used_core_num = Max(0, Ceiling((A / (wbo_size))));
  return true;
}

/*
  Tensor used for tiling case 1151 is:
  tensor_0:x1Local_output_0
  tensor_1:x2Local_output_0
  tensor_10:part1Final_output_0
  tensor_11:part1Final_output_1
  tensor_2:betaLocal_output_0
  tensor_3:biasLocal_output_0
  tensor_4:gammaLocal_output_0
  tensor_5:part1_output_0
  tensor_6:part1_output_2
  tensor_7:y_output_0
  tensor_8:part1_output_1
  tensor_9:x32_output_0
*/
class TilingCase1151Impl : public TilingCaseImpl {
 public:
  TilingCase1151Impl(uint32_t corenum) : TilingCaseImpl(corenum) {

  }
 protected:
  void GetTilingData(TilingDataCopy &from_tiling, graph_normalTilingData &to_tiling) {
    to_tiling.set_A(from_tiling.get_A());
    to_tiling.set_R(from_tiling.get_R());
    to_tiling.set_wbo_size(from_tiling.get_wbo_size());
    to_tiling.set_wio_size(from_tiling.get_wio_size());
    to_tiling.set_block_dim(from_tiling.get_block_dim());
    to_tiling.set_KERNEL_INIT_BUFFER(from_tiling.get_KERNEL_INIT_BUFFER());
    to_tiling.set_Q0(from_tiling.get_Q0());
    to_tiling.set_Q1(from_tiling.get_Q1());
    to_tiling.set_Q2(from_tiling.get_Q2());
    to_tiling.set_Q3(from_tiling.get_Q3());
    to_tiling.set_Q4(from_tiling.get_Q4());
    to_tiling.set_Q5(from_tiling.get_Q5());
    to_tiling.set_Q6(from_tiling.get_Q6());
    to_tiling.set_Q7(from_tiling.get_Q7());
    to_tiling.set_Q8(from_tiling.get_Q8());
    to_tiling.set_gm_size(from_tiling.get_gm_size());
    to_tiling.set_output0_single_core_size(from_tiling.get_output0_single_core_size());
    to_tiling.set_output0_total_size(from_tiling.get_output0_total_size());
    to_tiling.set_output1_single_core_size(from_tiling.get_output1_single_core_size());
    to_tiling.set_output1_total_size(from_tiling.get_output1_total_size());
    to_tiling.set_output2_single_core_size(from_tiling.get_output2_single_core_size());
    to_tiling.set_output2_total_size(from_tiling.get_output2_total_size());
    to_tiling.set_output3_single_core_size(from_tiling.get_output3_single_core_size());
    to_tiling.set_output3_total_size(from_tiling.get_output3_total_size());
    to_tiling.set_wbo_loop_num(from_tiling.get_wbo_loop_num());
    to_tiling.set_wbo_tail_size(from_tiling.get_wbo_tail_size());
    to_tiling.set_wio_loop_num(from_tiling.get_wio_loop_num());
    to_tiling.set_wio_tail_size(from_tiling.get_wio_tail_size());
    to_tiling.set_tiling_key(from_tiling.get_tiling_key());

  }
  void SetTilingData(graph_normalTilingData &from_tiling, TilingDataCopy &to_tiling) {
    to_tiling.set_A(from_tiling.get_A());
    to_tiling.set_R(from_tiling.get_R());
    to_tiling.set_wbo_size(from_tiling.get_wbo_size());
    to_tiling.set_wio_size(from_tiling.get_wio_size());
    to_tiling.set_block_dim(from_tiling.get_block_dim());
    to_tiling.set_KERNEL_INIT_BUFFER(from_tiling.get_KERNEL_INIT_BUFFER());
    to_tiling.set_Q0(from_tiling.get_Q0());
    to_tiling.set_Q1(from_tiling.get_Q1());
    to_tiling.set_Q2(from_tiling.get_Q2());
    to_tiling.set_Q3(from_tiling.get_Q3());
    to_tiling.set_Q4(from_tiling.get_Q4());
    to_tiling.set_Q5(from_tiling.get_Q5());
    to_tiling.set_Q6(from_tiling.get_Q6());
    to_tiling.set_Q7(from_tiling.get_Q7());
    to_tiling.set_Q8(from_tiling.get_Q8());
    to_tiling.set_gm_size(from_tiling.get_gm_size());
    to_tiling.set_output0_single_core_size(from_tiling.get_output0_single_core_size());
    to_tiling.set_output0_total_size(from_tiling.get_output0_total_size());
    to_tiling.set_output1_single_core_size(from_tiling.get_output1_single_core_size());
    to_tiling.set_output1_total_size(from_tiling.get_output1_total_size());
    to_tiling.set_output2_single_core_size(from_tiling.get_output2_single_core_size());
    to_tiling.set_output2_total_size(from_tiling.get_output2_total_size());
    to_tiling.set_output3_single_core_size(from_tiling.get_output3_single_core_size());
    to_tiling.set_output3_total_size(from_tiling.get_output3_total_size());
    to_tiling.set_wbo_loop_num(from_tiling.get_wbo_loop_num());
    to_tiling.set_wbo_tail_size(from_tiling.get_wbo_tail_size());
    to_tiling.set_wio_loop_num(from_tiling.get_wio_loop_num());
    to_tiling.set_wio_tail_size(from_tiling.get_wio_tail_size());
    to_tiling.set_tiling_key(from_tiling.get_tiling_key());

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
      double wio_size = rel_tiling_vars[0]->value;
      double tensor_0 = (2 * wio_size);
      double tensor_1 = (2 * wio_size);
      double tensor_10 = 4;
      double tensor_11 = 4;
      double tensor_2 = (2 * wio_size);
      double tensor_3 = (2 * wio_size);
      double tensor_4 = (2 * wio_size);
      double tensor_5 = (2 * wio_size);
      double tensor_6 = (4 * wio_size);
      double tensor_7 = (2 * wio_size);
      double tensor_8 = (4 * wio_size);
      double tensor_9 = (4 * wio_size);
      int64_t value = ((32 * Ceiling((Max(tensor_2, tensor_1) * Rational(1 , 32)))) + (32 * Ceiling((Max(tensor_4, tensor_3) * Rational(1 , 32)))) + (32 * Ceiling((Max(tensor_6, tensor_7) * Rational(1 , 32)))) + (32 * Ceiling((Rational(1 , 32) * tensor_0))) + (32 * Ceiling((Rational(1 , 32) * tensor_10))) + (32 * Ceiling((Rational(1 , 32) * tensor_11))) + (32 * Ceiling((Rational(1 , 32) * tensor_5))) + (32 * Ceiling((Rational(1 , 32) * tensor_8))) + (32 * Ceiling((Rational(1 , 32) * tensor_9))) + 8192) - rel_hw_spec;
      return value;
    };
    TilingVariable* cons_0rel_tiling_vars[1] = {&wio_size, };
    cons0.rel_tiling_vars = cons_0rel_tiling_vars;
    cons0.rel_tiling_vars_size = 1u;
    cons0.rel_hw_spec = ub_size;
    cons0.type = ConstraintType::LOCAL_BUFFER;
    cons0.eval = cons0Eval;
    Constraint cons1;
    auto cons1Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      double A = rel_in_shapes[0]->value;
      double wbo_size = rel_in_shapes[1]->value;
      int64_t value = (wbo_size - A);
      return value;
    };
    Variable* cons_1rel_in_shapes[2] = {&A, &wbo_size, };
    cons1.rel_in_shapes = cons_1rel_in_shapes;
    cons1.rel_in_shapes_size = 2u;
    cons1.type = ConstraintType::MC_MIXED;
    cons1.eval = cons1Eval;
    Constraint cons2;
    auto cons2Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      double wio_size = rel_tiling_vars[0]->value;
      double R = rel_in_shapes[0]->value;
      int64_t value = (wio_size - R);
      return value;
    };
    TilingVariable* cons_2rel_tiling_vars[1] = {&wio_size, };
    cons2.rel_tiling_vars = cons_2rel_tiling_vars;
    cons2.rel_tiling_vars_size = 1u;
    Variable* cons_2rel_in_shapes[1] = {&R, };
    cons2.rel_in_shapes = cons_2rel_in_shapes;
    cons2.rel_in_shapes_size = 1u;
    cons2.type = ConstraintType::LB_MIXED;
    cons2.eval = cons2Eval;
    GetUpperBoundFuncPtr wbo_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      double A = parent_vars[0]->value;
      if (parent_vars[0]->value == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= A;
      return upper_bound;
    };
    wbo_size.upper_bound = wbo_size_upper_bound;
    Variable* wbo_size_upper_bound_vars[1] = {&A, };
    wbo_size.upper_bound_vars = wbo_size_upper_bound_vars;
    wbo_size.upper_bound_vars_size = 1u;
    Constraint*wbo_size_rel_cons[1] = {&cons1, };
    wbo_size.rel_cons = wbo_size_rel_cons;
    wbo_size.rel_cons_size = 1u;
    wio_size.align = 1;
    GetUpperBoundFuncPtr wio_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      double R = parent_vars[0]->value;
      if (parent_vars[0]->value == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= R;
      return upper_bound;
    };
    wio_size.upper_bound = wio_size_upper_bound;
    Variable* wio_size_upper_bound_vars[1] = {&R, };
    wio_size.upper_bound_vars = wio_size_upper_bound_vars;
    wio_size.upper_bound_vars_size = 1u;
    Constraint*wio_size_rel_cons[1] = {&cons2, };
    wio_size.rel_cons = wio_size_rel_cons;
    wio_size.rel_cons_size = 1u;
    AxesReorderSolverInput input;
    Variable* input_vars[2] = {&A, &R, };
    input.input_vars = input_vars;
    input.input_vars_size = 2u;
    TilingVariable* tiling_vars[2] = {&wbo_size, &wio_size, };
    input.tiling_vars = tiling_vars;
    input.tiling_vars_size = 2u;
    Constraint* all_cons[3] = {&cons0, &cons1, &cons2, };
    input.all_cons_size = 3u;
    input.all_cons = all_cons;
    TilingVariable* pure_mc_vars[1] = {&wbo_size, };
    input.pure_mc_vars_size = 1u;
    input.pure_mc_vars = pure_mc_vars;
    TilingVariable* local_buffer_vars[1] = {&wio_size, };
    input.local_buffer_vars_size = 1u;
    input.local_buffer_vars = local_buffer_vars;
    input.core_num = corenum_;
    input.ub_threshold = 0.200000;
    input.corenum_threshold = 0.800000;
    input.ub_size = tiling_data.get_ub_size();
    AxesReorderSolvercase1151 solver(input);
    if (!solver.Run(false)) {
      return false;
    }
    tiling_data.set_wbo_size(input.pure_mc_vars[0]->value);
    tiling_data.set_wio_size(input.local_buffer_vars[0]->value);
    return true;
  }

  bool DoTiling(graph_normalTilingData &tiling_data) {
    OP_LOGI(OP_NAME, "[PROF]Set input params for tiling case 1151 of ScheduleResult0G0.  A = %u. R = %u.", tiling_data.get_A(), tiling_data.get_R());
    OP_LOGI(OP_NAME, "[PROF]Set ub_size for tiling case 1151 of ScheduleResult0G0 to ((32 * Ceiling((Max(tensor_2, tensor_1) * Rational(1 , 32)))) + (32 * Ceiling((Max(tensor_4, tensor_3) * Rational(1 , 32)))) + (32 * Ceil");
    OP_LOGI(OP_NAME, "ing((Max(tensor_6, tensor_7) * Rational(1 , 32)))) + (32 * Ceiling((Rational(1 , 32) * tensor_0))) + (32 * Ceiling((Rational(1 , 32) * tensor_10))) + (32 * Ceiling((Rational(1 , 32) * tensor_11))) + (");
    OP_LOGI(OP_NAME, "32 * Ceiling((Rational(1 , 32) * tensor_5))) + (32 * Ceiling((Rational(1 , 32) * tensor_8))) + (32 * Ceiling((Rational(1 , 32) * tensor_9))) + 8192)");
    OP_LOGI(OP_NAME, "[PROF]Set block_dim for tiling case 1151 of ScheduleResult0G0 to Max(0, Ceiling((A / (wbo_size))))");

    OP_LOGD(OP_NAME, "[PROF]Set hardware params. ub_size = %u. block_dim = %u.", tiling_data.get_ub_size(), tiling_data.get_block_dim());
    uint32_t wbo_size = tiling_data.get_wbo_size();
    uint32_t wio_size = tiling_data.get_wio_size();

    if (!ExecuteAxesReorderSolver(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to execute axes reorder solver for tilingCaseId case1151.");
      return false;
    }
    OP_LOGD(OP_NAME, "Execute axes reorder solver for tilingCaseId case1151 successfully.");

    return true;
  }

void DoApiTiling(graph_normalTilingData &tiling_data) {
}
  void GeneralTiling(graph_normalTilingData &tiling_data) {
    double A = static_cast<double>(tiling_data.get_A());
    double wbo_size = static_cast<double>(tiling_data.get_wbo_size());
    tiling_data.set_block_dim(Max(0, Ceiling((A / (wbo_size)))));
  }

  int Getub_size(graph_normalTilingData& tiling_data) {
    double wio_size = tiling_data.get_wio_size();
    double tensor_0 = (2 * wio_size);
    double tensor_1 = (2 * wio_size);
    double tensor_10 = 4;
    double tensor_11 = 4;
    double tensor_2 = (2 * wio_size);
    double tensor_3 = (2 * wio_size);
    double tensor_4 = (2 * wio_size);
    double tensor_5 = (2 * wio_size);
    double tensor_6 = (4 * wio_size);
    double tensor_7 = (2 * wio_size);
    double tensor_8 = (4 * wio_size);
    double tensor_9 = (4 * wio_size);

    return ((32 * Ceiling((Max(tensor_2, tensor_1) * Rational(1 , 32)))) + (32 * Ceiling((Max(tensor_4, tensor_3) * Rational(1 , 32)))) + (32 * Ceiling((Max(tensor_6, tensor_7) * Rational(1 , 32)))) + (32 * Ceiling((Rational(1 , 32) * tensor_0))) + (32 * Ceiling((Rational(1 , 32) * tensor_10))) + (32 * Ceiling((Rational(1 , 32) * tensor_11))) + (32 * Ceiling((Rational(1 , 32) * tensor_5))) + (32 * Ceiling((Rational(1 , 32) * tensor_8))) + (32 * Ceiling((Rational(1 , 32) * tensor_9))) + 8192);
  }

  int Getblock_dim(graph_normalTilingData& tiling_data) {
    double A = tiling_data.get_A();
    double wbo_size = tiling_data.get_wbo_size();

    return Max(0, Ceiling((A / (wbo_size))));
  }

  double GetAIV_MTE2(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double block_dim = tiling_data.get_block_dim();
    double wbo_size = tiling_data.get_wbo_size();
    double wio_size = tiling_data.get_wio_size();

    return ((((2 * wio_size / (((24.1000003814697 / (block_dim)) + 8.47000026702881))) + 27.0100002288818) * 5 * Ceiling((R / (wio_size))) * wbo_size) + (((4 * wio_size / (((24.1000003814697 / (block_dim)) + 8.47000026702881))) + 27.0100002288818) * Ceiling((R / (wio_size))) * wbo_size) + 1305.67004394531);
  }

  double GetAIV_MTE3(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double block_dim = tiling_data.get_block_dim();
    double wbo_size = tiling_data.get_wbo_size();
    double wio_size = tiling_data.get_wio_size();

    return ((((2 * wio_size / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879) * 2 * Ceiling((R / (wio_size))) * wbo_size) + (((4 * wio_size / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879) * 3 * Ceiling((R / (wio_size))) * wbo_size) + 497.359985351562);
  }

  double GetAIV_VEC(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double wbo_size = tiling_data.get_wbo_size();
    double wio_size = tiling_data.get_wio_size();

    return ((((4 * wio_size / ((-1 + wio_size))) + 4) * Ceiling((R / (wio_size))) * wbo_size) + (((8 * wio_size / ((-1 + wio_size))) + 4) * 2 * Ceiling((R / (wio_size))) * wbo_size) + 37.3699989318848);
  }

  double GetPerf(graph_normalTilingData& tiling_data) {
    double R = tiling_data.get_R();
    double block_dim = tiling_data.get_block_dim();
    double wbo_size = tiling_data.get_wbo_size();
    double wio_size = tiling_data.get_wio_size();

    double AIV_MTE2 = ((((2 * wio_size / (((24.1000003814697 / (block_dim)) + 8.47000026702881))) + 27.0100002288818) * 5 * Ceiling((R / (wio_size))) * wbo_size) + (((4 * wio_size / (((24.1000003814697 / (block_dim)) + 8.47000026702881))) + 27.0100002288818) * Ceiling((R / (wio_size))) * wbo_size) + 1305.67004394531);
    double AIV_MTE3 = ((((2 * wio_size / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879) * 2 * Ceiling((R / (wio_size))) * wbo_size) + (((4 * wio_size / (((3.78999996185303 / (block_dim)) + 9.96000003814697))) + 12.0900001525879) * 3 * Ceiling((R / (wio_size))) * wbo_size) + 497.359985351562);
    double AIV_VEC = ((((4 * wio_size / ((-1 + wio_size))) + 4) * Ceiling((R / (wio_size))) * wbo_size) + (((8 * wio_size / ((-1 + wio_size))) + 4) * 2 * Ceiling((R / (wio_size))) * wbo_size) + 37.3699989318848);

    return Max(Max(AIV_VEC, AIV_MTE2), AIV_MTE3);
  }

  void UpdateAxesTilingData(graph_normalTilingData& tiling_data) {
    tiling_data.set_wio_loop_num(((tiling_data.get_R() + tiling_data.get_wio_size()) - 1) / tiling_data.get_wio_size());
    tiling_data.set_wbo_loop_num(((tiling_data.get_A() + tiling_data.get_wbo_size()) - 1) / tiling_data.get_wbo_size());
    tiling_data.set_wio_tail_size((tiling_data.get_R() % tiling_data.get_wio_size()) == 0 ? tiling_data.get_wio_size() : (tiling_data.get_R() % tiling_data.get_wio_size()));
    tiling_data.set_wbo_tail_size((tiling_data.get_A() % tiling_data.get_wbo_size()) == 0 ? tiling_data.get_wbo_size() : (tiling_data.get_A() % tiling_data.get_wbo_size()));
  }

  void SetKERNEL_INIT_BUFFER(graph_normalTilingData &tiling_data) {
    tiling_data.set_KERNEL_INIT_BUFFER(8192);
  }

  void SetQ0(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    const auto tensor_0 = (2 * wio_size);
    tiling_data.set_Q0((32 * Ceiling((Rational(1 , 32) * tensor_0))));
  }

  void SetQ1(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    const auto tensor_1 = (2 * wio_size);
    const auto tensor_2 = (2 * wio_size);
    tiling_data.set_Q1((32 * Ceiling((Max(tensor_2, tensor_1) * Rational(1 , 32)))));
  }

  void SetQ2(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    const auto tensor_3 = (2 * wio_size);
    const auto tensor_4 = (2 * wio_size);
    tiling_data.set_Q2((32 * Ceiling((Max(tensor_4, tensor_3) * Rational(1 , 32)))));
  }

  void SetQ3(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    const auto tensor_5 = (2 * wio_size);
    tiling_data.set_Q3((32 * Ceiling((Rational(1 , 32) * tensor_5))));
  }

  void SetQ4(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    const auto tensor_6 = (4 * wio_size);
    const auto tensor_7 = (2 * wio_size);
    tiling_data.set_Q4((32 * Ceiling((Max(tensor_6, tensor_7) * Rational(1 , 32)))));
  }

  void SetQ5(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    const auto tensor_8 = (4 * wio_size);
    tiling_data.set_Q5((32 * Ceiling((Rational(1 , 32) * tensor_8))));
  }

  void SetQ6(graph_normalTilingData &tiling_data) {
    const auto wio_size = tiling_data.get_wio_size();
    const auto tensor_9 = (4 * wio_size);
    tiling_data.set_Q6((32 * Ceiling((Rational(1 , 32) * tensor_9))));
  }

  void SetQ7(graph_normalTilingData &tiling_data) {
    const auto tensor_10 = 4;
    tiling_data.set_Q7((32 * Ceiling((Rational(1 , 32) * tensor_10))));
  }

  void SetQ8(graph_normalTilingData &tiling_data) {
    const auto tensor_11 = 4;
    tiling_data.set_Q8((32 * Ceiling((Rational(1 , 32) * tensor_11))));
  }

  void ComputeOptionParam(graph_normalTilingData &tiling_data) {
    SetKERNEL_INIT_BUFFER(tiling_data);
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

  void ExtraTilingData(graph_normalTilingData &tiling_data) {
    OP_LOGD(OP_NAME, "Start executing extra tiling for tilingCaseId 1151.");
    ComputeOptionParam(tiling_data);
		UpdateAxesTilingData(tiling_data);

    OP_LOGD(OP_NAME, "Execute extra tiling for tilingCaseId 1151 successfully.");
  }

  void GetWorkSpaceSize(graph_normalTilingData& tiling_data) {
    OP_LOGD(OP_NAME, "Start setting workspace for case 1151.");
    tiling_data.set_workspaceSize(static_cast<uint32_t>(0));
    OP_LOGD(OP_NAME, "Setting workspace to %u for case 1151.", tiling_data.get_workspaceSize());
  }

  void TilingSummary(graph_normalTilingData &tiling_data, double& cur_ub_ratio) {
    OP_LOGI(OP_NAME, "The value of wbo_size is %u in schedule_result0_g0.", tiling_data.get_wbo_size());
    OP_LOGI(OP_NAME, "The value of wio_size is %u in schedule_result0_g0.", tiling_data.get_wio_size());
    OP_LOGI(OP_NAME, "The value of ub_size is %d in schedule_result0_g0.", Getub_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of block_dim is %d in schedule_result0_g0.", Getblock_dim(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIV_MTE2 is %f.", GetAIV_MTE2(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIV_MTE3 is %f.", GetAIV_MTE3(tiling_data));
    OP_LOGI(OP_NAME, "The value of AIV_VEC is %f.", GetAIV_VEC(tiling_data));
    OP_LOGI(OP_NAME, "The objective value of the tiling data is %f.", GetPerf(tiling_data));
    cur_ub_ratio = static_cast<double>(Getub_size(tiling_data)) / tiling_data.get_ub_size();
    if (std::isnan(cur_ub_ratio)) {
      cur_ub_ratio = 1;
      OP_LOGI(OP_NAME, "The ub ratio is NaN, set it to 1.");
    }
  }

};

TilingCaseImplPtr GetTilingImplPtr(uint32_t tilingCaseId, uint32_t corenum) {
  TilingCaseImplPtr tilingCaseImplPtr = nullptr;
  if (tilingCaseId == 1101u) {
    tilingCaseImplPtr = std::make_shared<TilingCase1101Impl>(corenum);
  } else if (tilingCaseId == 1111u) {
    tilingCaseImplPtr = std::make_shared<TilingCase1111Impl>(corenum);
  } else if (tilingCaseId == 1151u) {
    tilingCaseImplPtr = std::make_shared<TilingCase1151Impl>(corenum);
  }
  return tilingCaseImplPtr;
}
void UpdateBetterTiling(TilingCaseImpl *tilingCaseImplPtr, TilingDataCopy &tmp_tiling, graph_normalTilingData &tiling_data, uint32_t tilingCaseId) {
  OP_LOGD(OP_NAME, "The solution for tilingCaseId %u is better, updating the tiling data.", tilingCaseId);
  tiling_data.set_tiling_key(tilingCaseId);
  tilingCaseImplPtr->SetTilingData(tiling_data, tmp_tiling);
  OP_LOGD(OP_NAME, "Set the output tiling data.");
  OP_LOGD(OP_NAME, "Updated the best tilingCaseId to %u.", tilingCaseId);
}

bool FindPerfBetterTilingbyCaseId(TilingCaseImpl *tilingCaseImplPtr, double &obj, double &ub_ratio, TilingDataCopy &tmp_tiling, graph_normalTilingData &tiling_data, uint32_t tilingCaseId) {
  double cur_obj;
  double cur_ub_ratio;
  if (tilingCaseImplPtr == nullptr) {
    OP_LOGE(OP_NAME, "Pointer for tilingCaseId is null.");
    return false;
  }
  tilingCaseImplPtr->SetTilingData(tiling_data, tmp_tiling);
  if (tilingCaseImplPtr->GetTiling(tiling_data, cur_ub_ratio)) {
    cur_obj = tilingCaseImplPtr->GetPerf(tiling_data);
    OP_LOGD(OP_NAME, "The ub ratio for tilingCaseId %u is %f.", tilingCaseId, cur_ub_ratio);
    OP_LOGD(OP_NAME, "The optimal objection for tilingCaseId %u is %f.", tilingCaseId, cur_obj);
    if (obj < 0) {
      UpdateBetterTiling(tilingCaseImplPtr, tmp_tiling, tiling_data, tilingCaseId);
      obj = cur_obj;
      ub_ratio = cur_ub_ratio;
      return true;
    }
    if (cur_ub_ratio > 0.800000) {
      if (ub_ratio > 0.800000) {
        if (cur_obj < obj) {
          UpdateBetterTiling(tilingCaseImplPtr, tmp_tiling, tiling_data, tilingCaseId);
          obj = cur_obj;
          ub_ratio = cur_ub_ratio;
        } else {
          tilingCaseImplPtr->GetTilingData(tmp_tiling, tiling_data);
        }
      } else {
        UpdateBetterTiling(tilingCaseImplPtr, tmp_tiling, tiling_data, tilingCaseId);
        obj = cur_obj;
        ub_ratio = cur_ub_ratio;
      }
    } else {
      if (ub_ratio > 0.800000) {
        tilingCaseImplPtr->GetTilingData(tmp_tiling, tiling_data);
      } else {
        if (cur_ub_ratio > ub_ratio) {
          UpdateBetterTiling(tilingCaseImplPtr, tmp_tiling, tiling_data, tilingCaseId);
          obj = cur_obj;
          ub_ratio = cur_ub_ratio;
        } else {
          tilingCaseImplPtr->GetTilingData(tmp_tiling, tiling_data);
        }
      }
    }
    return true;
  } else {
    tilingCaseImplPtr->GetTilingData(tmp_tiling, tiling_data);
  }
  return false;
}

bool GetTilingKey(graph_normalTilingData &tiling_data, int32_t tilingCaseId = -1) {
  bool ret = false;
  double obj = -1;
  double ub_ratio = -1;
  uint32_t corenum = tiling_data.get_block_dim();
  if (tilingCaseId == -1) {
    OP_LOGI(OP_NAME, "The user didn't specify tilingCaseId, iterate all templates.");
    TilingDataCopy tmp_tiling;
    SaveCaseNumInfo(3);
    TilingCaseImpl *tilingCaseImplPtr;
    TilingCase1101Impl case1101(corenum);
    tilingCaseImplPtr = &case1101;
    OP_LOGD(OP_NAME, "[PROF]Calculating the tiling data for tilingCaseId 1101.");
    ret = (FindPerfBetterTilingbyCaseId(tilingCaseImplPtr, obj, ub_ratio, tmp_tiling, tiling_data, 1101u) || ret);
    OP_LOGD(OP_NAME, "[PROF]Finish calculating the tiling data for tilingCaseId 1101.");
    tilingCaseImplPtr->~TilingCaseImpl();
    TilingCase1111Impl case1111(corenum);
    tilingCaseImplPtr = &case1111;
    OP_LOGD(OP_NAME, "[PROF]Calculating the tiling data for tilingCaseId 1111.");
    ret = (FindPerfBetterTilingbyCaseId(tilingCaseImplPtr, obj, ub_ratio, tmp_tiling, tiling_data, 1111u) || ret);
    OP_LOGD(OP_NAME, "[PROF]Finish calculating the tiling data for tilingCaseId 1111.");
    tilingCaseImplPtr->~TilingCaseImpl();
    TilingCase1151Impl case1151(corenum);
    tilingCaseImplPtr = &case1151;
    OP_LOGD(OP_NAME, "[PROF]Calculating the tiling data for tilingCaseId 1151.");
    ret = (FindPerfBetterTilingbyCaseId(tilingCaseImplPtr, obj, ub_ratio, tmp_tiling, tiling_data, 1151u) || ret);
    OP_LOGD(OP_NAME, "[PROF]Finish calculating the tiling data for tilingCaseId 1151.");
    tilingCaseImplPtr->~TilingCaseImpl();
    if (ret) {
      OP_LOGI(OP_NAME, "[PROF]Among the templates, tiling case %u is the best choice.", tiling_data.get_tiling_key());
    }
  } else {
    OP_LOGD(OP_NAME, "[PROF]Calculating the tiling data for tilingCaseId %u.", tilingCaseId);
    SaveCaseNumInfo(1);
    TilingCaseImplPtr tilingCaseImplPtr = GetTilingImplPtr(tilingCaseId, corenum);
    if (tilingCaseImplPtr == nullptr) {
      OP_LOGE(OP_NAME, "Pointer for tilingCaseId is null.");
      return false;
    }
    ret = tilingCaseImplPtr->GetTiling(tiling_data, ub_ratio);
    OP_LOGD(OP_NAME, "[PROF]Finish calculating the tiling data for tilingCaseId %u.", tilingCaseId);
  }
  if (!ret) {
    OP_LOGE(OP_NAME, "[PROF]Set input params for tiling case 1101 of ScheduleResult0G0.  R = %u.", tiling_data.get_R());
    OP_LOGE(OP_NAME, "[PROF]Set input params for tiling case 1111 of ScheduleResult0G0.  A = %u. R = %u.", tiling_data.get_A(), tiling_data.get_R());
    OP_LOGE(OP_NAME, "[PROF]Set input params for tiling case 1151 of ScheduleResult0G0.  A = %u. R = %u.", tiling_data.get_A(), tiling_data.get_R());
    OP_LOGE(OP_NAME, "Failed to execute tiling func.");
  }
  return ret;
}

bool GetTiling(graph_normalTilingData &tiling_data, TilingOption *tiling_option) {
  DurationBegin(TILING_FUNC_DURATION_TOTAL);
  TilingOption *tiling_option_used = nullptr;
  if (tiling_option == nullptr) {
    tiling_option_used = &tiling_option_default;
  } else {
    tiling_option_used = tiling_option;
  }
  OP_LOGI(OP_NAME, "[PROF]Start GetTiling.");
  if (!GetTilingKey(tiling_data, tiling_option_used->tiling_case_id)) {
    OP_LOGE(OP_NAME, "GetTiling Failed.");
    return false;
  }
  OP_LOGI(OP_NAME, "[PROF]End GetTiling.");
  DurationEnd(TILING_FUNC_DURATION_TOTAL);
  DurationManager::GetInstance().Print();
  DurationManager::GetInstance().Clear();
  return true;
}
bool GetTiling(graph_normalTilingData &tiling_data, int32_t tilingCaseId) {
  tiling_option_default.tiling_case_id = tilingCaseId;
  return GetTiling(tiling_data, &tiling_option_default);
}
bool GetTilingOptionRange(const int32_t option_id, int32_t *option_range_size, int32_t *range_type, int32_t *option_range) {
  if (!((option_id >= 0) && (option_id <=1))) {
    OP_LOGE(OP_NAME, "option_id is invalid, valid range is ((option_id >= 0) && (option_id <=1))");
    return false;
  }
  if ((option_range_size != nullptr)) {
    OP_LOGE(OP_NAME, "check failed, option_range_size is nullptr.");
    return false;
  }
  if ((range_type != nullptr)) {
    OP_LOGE(OP_NAME, "check failed, range_type is nullptr.");
    return false;
  }
  if ((option_range != nullptr)) {
    OP_LOGE(OP_NAME, "check failed, option_range is nullptr.");
    return false;
  }
  if (option_id == 1) {
    *option_range_size = 2;
    for (int32_t i = 0; i < 1; i++) {
      *(option_range + 0) = 0;
      *(option_range + 1) = 1;
    }
    return true;
  }
  if (option_id == 1) {
    *option_range_size = 3;
    for (int32_t i = 0; i < 0; i++) {
      *(option_range + 0) = 1101;
      *(option_range + 1) = 1111;
      *(option_range + 2) = 1151;
    }
    return true;
  }
  return true;
}


} // namespace optiling

