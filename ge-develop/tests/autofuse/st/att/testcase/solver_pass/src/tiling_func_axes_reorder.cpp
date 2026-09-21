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
#include "tiling_data.h"
#define Max(a, b) ((double)(a) > (double)(b) ? (a) : (b))
#define Min(a, b) ((double)(a) < (double)(b) ? (a) : (b))
#define Log(a) (log((double)(a)))
#define MAX_SOLUTION 50
#define OP_NAME "OpTest"

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
  bool GetTiling(TilingData &tiling_data) {
    if (!DoTiling(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to do tiling.");
      return false;
    }
    DoApiTiling(tiling_data);
    ExtraTilingData(tiling_data);
    TilingSummary(tiling_data);
    return true;
  }
  virtual void GetTilingData(TilingData &tiling_data, TilingData &to_tiling) {};
  virtual void SetTilingData(TilingData &from_tiling, TilingData &tiling_data) {};
  virtual double GetPerf(TilingData &tiling_data) { return 0.0; }
 protected:
  virtual bool DoTiling(TilingData &tiling_data) = 0;
  virtual void DoApiTiling(TilingData &tiling_data) {}
  virtual void GetWorkSpaceSize(TilingData& tiling_data) {}
  virtual void ExtraTilingData(TilingData &tiling_data) {}
  virtual void TilingSummary(TilingData &tiling_data) = 0;
  uint32_t corenum_;
};
using TilingCaseImplPtr = std::shared_ptr<TilingCaseImpl>;

/*
(可修改变量)用于控制轴排序求解器求解效率与质量的超参数
*/
static const double mc_fast_mode_ratio = 0.5;
static const double accept_core_num_ratio = 0.8;

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
  bool SetNextSmallerValue(TilingVariable *var, bool fast_mode);
  bool SetNextBiggerValue(TilingVariable *var);
  bool SatisfyCons(ConstraintType cons_type);
  TilingVariable *CopyVars(TilingVariable **vars, uint32_t var_size);
  bool MulticoreTiling();
  bool LocalBufTiling();
};

bool AxesReorderSolver::SetNextSmallerValue(TilingVariable *var, bool fast_mode = false) {
  int32_t val = var->value;
  int32_t align = var->align;
  if (val == 1 || val == 0) {
    return false;
  }
  if (fast_mode) {
    val /= 2;
    val = CeilDiv(val, align) * align;
  } else {
    val -= align;
  }
  if (var->notail) {
    for (; val > 0; val -= align) {
      if (var->notail_var->value % val != 0) {
        continue;
      }
      break;
    };
  }
  if (var == 0) {
    return false;
  } else {
    var->value = val;
    return true;
  }
}

bool AxesReorderSolver::SetNextBiggerValue(TilingVariable *var) {
  int32_t val = var->value;
  int32_t align = var->align;
  int32_t upper_bound = var->upper_bound(var->upper_bound_vars);
  val += align;
  if (var->notail) {
    for (; val > 0; val += align) {
      if (var->notail_var->value % val != 0) {
        continue;
      }
      break;
    };
  }
  if (val > upper_bound) {
    return false;
  } else {
    var->value = val;
    return true;
  }
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

TilingVariable *AxesReorderSolver::CopyVars(TilingVariable **vars, uint32_t var_size) {
  TilingVariable *copy_vars = new TilingVariable[var_size];
  for (uint32_t i = 0u; i < var_size; ++i) {
    auto &var = vars[i];
    TilingVariable copy_var = *var;
    copy_vars[i] = copy_var;
  }
  return copy_vars;
}

bool AxesReorderSolver::MulticoreTiling() {
  for (uint32_t i = input_.pure_mc_vars_size - 1; i >= 0; --i) {
    auto &var = input_.pure_mc_vars[i];
    auto upper_bound_val = var->upper_bound(var->upper_bound_vars);
    if (upper_bound_val == -1) {
      return false;
    }
    upper_bound_val = Max(var->align, (upper_bound_val / var->align * var->align));
    if (!var->SetValue(upper_bound_val)) {
      OP_LOGW(OP_NAME, "Failed to set value.");
      return false;
    }
  }
  int32_t max_used_core_num = -1;
  bool max_load_balance = false;
  int32_t used_core_num = 1;
  bool load_balance = false;
  if (!CalUsedCoreNum(used_core_num, load_balance)) {
    return false;
  }
  TilingVariable *optimal_mc_vars;
  if (static_cast<uint32_t>(used_core_num) <= input_.core_num && SatisfyCons(ConstraintType::MC_MIXED)) {
    max_used_core_num = used_core_num;
    max_load_balance = load_balance;
    optimal_mc_vars = CopyVars(input_.pure_mc_vars, input_.pure_mc_vars_size);
  }
  for (uint32_t i = 0u; i < input_.pure_mc_vars_size; ++i) {
    auto &var = input_.pure_mc_vars[i];
    while (SetNextSmallerValue(var)) {
      if (!CalUsedCoreNum(used_core_num, load_balance)) {
        return false;
      }
      if (static_cast<uint32_t>(used_core_num) <= input_.core_num && SatisfyCons(ConstraintType::MC_MIXED)) {
        if (used_core_num > max_used_core_num || (used_core_num == max_used_core_num && !max_load_balance && load_balance)) {
          max_used_core_num = used_core_num;
          max_load_balance = load_balance;
          optimal_mc_vars = CopyVars(input_.pure_mc_vars, input_.pure_mc_vars_size);
        }
      }
      if (static_cast<uint32_t>(used_core_num) > input_.core_num) {
        break;
      }
    }
  }
  if (max_used_core_num == -1) {
    OP_LOGW(OP_NAME, "Multicore tiling calculateion failed.");
    return false;
  }
  for (uint32_t i = 0u; i < input_.pure_mc_vars_size; ++i) {
    *input_.pure_mc_vars[i] = optimal_mc_vars[i];
  }
  return true;
}

bool AxesReorderSolver::LocalBufTiling() {
  for (uint32_t i = 0u; i < input_.local_buffer_vars_size; ++i) {
    auto &var = input_.local_buffer_vars[i];
    var->SetValue(var->align);
  }
  if (!SatisfyCons(ConstraintType::LOCAL_BUFFER)) {
    return false;
  }
  for (uint32_t i = 0u; i < input_.local_buffer_vars_size; ++i) {
    auto &var = input_.local_buffer_vars[i];
    auto tmp = var->value;
    while (SetNextBiggerValue(var)) {
      bool satisfy_all_rel_lb_cons = true;
      for (uint32_t j = 0; j < var->rel_cons_size; ++j) {
        auto &cons = var->rel_cons[j];
        if (cons->type != ConstraintType::LOCAL_BUFFER) {
          continue;
        }
        if (cons->eval(cons->rel_tiling_vars, cons->rel_in_shapes, cons->rel_hw_spec) > 0) {
          satisfy_all_rel_lb_cons = false;
          break;
        }
      }
      if (!satisfy_all_rel_lb_cons) {
        var->value = tmp;
        break;
      }
      bool satisfy_all_rel_mix_cons = true;
      for (uint32_t j = 0; j < var->rel_cons_size; ++j) {
        auto &cons = var->rel_cons[j];
        if (cons->type != ConstraintType::LB_MIXED) {
          continue;
        }
        auto cons_val = cons->eval(cons->rel_tiling_vars, cons->rel_in_shapes, cons->rel_hw_spec);
        ++var->value;
        auto new_cons_val = cons->eval(cons->rel_tiling_vars, cons->rel_in_shapes, cons->rel_hw_spec);
        --var->value;
        if (cons_val > 0 && new_cons_val > cons_val) {
          satisfy_all_rel_mix_cons = false;
          break;
        }
      }
      if (!satisfy_all_rel_mix_cons) {
        var->value = tmp;
        break;
      }
      tmp = var->value;
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
    return false;
  }
  if (!MulticoreTiling()) {
    return false;
  }
  return true;
}


bool GetPlatformInfo(TilingData &tiling_data, gert::TilingContext *context) {
  auto platformInfoPtr = context->GetPlatformInfo();
  if (platformInfoPtr == nullptr) {
    OP_LOGE(OP_NAME, "Pointer platformInfoPtr is null.");
    return false;
  }
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
  auto aivNum = ascendcPlatform.GetCoreNumAiv();
  auto aicNum = ascendcPlatform.GetCoreNumAic();
  uint64_t hbm_size;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::HBM, hbm_size);
  uint64_t ub_size;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  if ((aivNum == 0) || (aicNum == 0) || (hbm_size == 0) || (ub_size == 0)) {
    OP_LOGE(OP_NAME, "Get incorrect platform value.");
    return false;
  } 
  OP_LOGD(OP_NAME, "PlatformInfo is valid.");
  tiling_data.set_block_dim(ascendcPlatform.GetCoreNumAiv());
  OP_LOGD(OP_NAME, "Set block dim to %d.", tiling_data.get_block_dim());
  tiling_data.set_hbm_size(hbm_size);
  OP_LOGD(OP_NAME, "Set hbm_size to %d.", tiling_data.get_hbm_size());
  tiling_data.set_ub_size(ub_size);
  OP_LOGD(OP_NAME, "Set ub_size to %d.", tiling_data.get_ub_size());

  return true;
}

class AxesReorderSolvercase0 : public AxesReorderSolver {
 public:
  explicit AxesReorderSolvercase0(const AxesReorderSolverInput input) : AxesReorderSolver(input) {}
  ~AxesReorderSolvercase0() = default;
  bool CalUsedCoreNum(int32_t &used_core_num, bool &load_balance) override;
};

bool AxesReorderSolvercase0::CalUsedCoreNum(int32_t &used_core_num, bool &load_balance) {
  double B = static_cast<double>(input_.input_vars[0]->value);
  double G = static_cast<double>(input_.input_vars[2]->value);
  double N = static_cast<double>(input_.input_vars[3]->value);
  double S1 = static_cast<double>(input_.input_vars[4]->value);
  double bngs1Tb_size = static_cast<double>(input_.pure_mc_vars[0]->value);
  double s1t_size = static_cast<double>(input_.local_buffer_vars[0]->value);
  double used_core_num_fp = Max(0, ceiling((B * G * N * ceiling((S1 / (s1t_size))) / (bngs1Tb_size))));
  load_balance = (static_cast<double>(static_cast<int64_t>(used_core_num_fp)) == used_core_num_fp);
  used_core_num = ceiling(used_core_num_fp);
  return true;
}

class TilingCase0Impl : public TilingCaseImpl {
 public:
  TilingCase0Impl(uint32_t corenum) : TilingCaseImpl(corenum) {}
 protected:
  bool ExecuteAxesReorderSolver(TilingData &tiling_data) {
    Variable B;
    B.value = tiling_data.get_B();
    Variable D;
    D.value = tiling_data.get_D();
    Variable G;
    G.value = tiling_data.get_G();
    Variable N;
    N.value = tiling_data.get_N();
    Variable S1;
    S1.value = tiling_data.get_S1();
    Variable S2;
    S2.value = tiling_data.get_S2();
    Variable BL;
    BL.value = 8;
    TilingVariable bngs1Tb_size;
    TilingVariable s1t_size;
    TilingVariable s2t_size;
    TilingVariable s1tt_size;
    TilingVariable s1tt2_size;
    int64_t hbm_size = tiling_data.get_hbm_size();
    Constraint cons0;
    auto cons0Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t s1tt2_size = rel_tiling_vars[0]->value;
      int64_t s1t_size = rel_tiling_vars[1]->value;
      int64_t s2t_size = rel_tiling_vars[2]->value;
      int64_t s1tt_size = rel_tiling_vars[3]->value;
      int64_t D = rel_in_shapes[0]->value;
      int64_t S2 = rel_in_shapes[1]->value;
      int64_t value = ((((S2 * s1t_size) + (s1tt_size * s2t_size) - (S2 * s1tt_size)) * 4) + (8 * D * s1t_size) + (8 * D * s1tt2_size) + (8 * s1t_size * s2t_size)) - rel_hw_spec;
      return value;
    };
    cons0.rel_tiling_vars = new TilingVariable*[4];
    cons0.rel_tiling_vars_size = 4u;
    cons0.rel_tiling_vars[0] = &s1tt2_size;
    cons0.rel_tiling_vars[1] = &s1t_size;
    cons0.rel_tiling_vars[2] = &s2t_size;
    cons0.rel_tiling_vars[3] = &s1tt_size;
    cons0.rel_in_shapes = new Variable*[2];
    cons0.rel_in_shapes_size = 2u;
    cons0.rel_in_shapes[0] = &D;
    cons0.rel_in_shapes[1] = &S2;
    cons0.rel_hw_spec = hbm_size;
    cons0.type = ConstraintType::LOCAL_BUFFER;
    cons0.eval = cons0Eval;
    int64_t ub_size = tiling_data.get_ub_size();
    Constraint cons1;
    auto cons1Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t s1t_size = rel_tiling_vars[0]->value;
      int64_t s1tt2_size = rel_tiling_vars[1]->value;
      int64_t s1tt_size = rel_tiling_vars[2]->value;
      int64_t s2t_size = rel_tiling_vars[3]->value;
      int64_t D = rel_in_shapes[0]->value;
      int64_t value = ((12 * s1tt_size * s2t_size) + (160 * s1t_size) + (4 * D * s1tt2_size) + Max((4 * s1tt_size * s2t_size), (4 * D * s1tt2_size))) - rel_hw_spec;
      return value;
    };
    cons1.rel_tiling_vars = new TilingVariable*[4];
    cons1.rel_tiling_vars_size = 4u;
    cons1.rel_tiling_vars[0] = &s1t_size;
    cons1.rel_tiling_vars[1] = &s1tt2_size;
    cons1.rel_tiling_vars[2] = &s1tt_size;
    cons1.rel_tiling_vars[3] = &s2t_size;
    cons1.rel_in_shapes = new Variable*[1];
    cons1.rel_in_shapes_size = 1u;
    cons1.rel_in_shapes[0] = &D;
    cons1.rel_hw_spec = ub_size;
    cons1.type = ConstraintType::LOCAL_BUFFER;
    cons1.eval = cons1Eval;
    Constraint cons2;
    auto cons2Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t s1t_size = rel_tiling_vars[0]->value;
      int64_t S1 = rel_in_shapes[0]->value;
      int64_t value = (s1t_size - S1);
      return value;
    };
    cons2.rel_tiling_vars = new TilingVariable*[1];
    cons2.rel_tiling_vars_size = 1u;
    cons2.rel_tiling_vars[0] = &s1t_size;
    cons2.rel_in_shapes = new Variable*[1];
    cons2.rel_in_shapes_size = 1u;
    cons2.rel_in_shapes[0] = &S1;
    cons2.type = ConstraintType::LB_MIXED;
    cons2.eval = cons2Eval;
    Constraint cons3;
    auto cons3Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t value = 0;
      return value;
    };
    cons3.type = ConstraintType::LB_MIXED;
    cons3.eval = cons3Eval;
    Constraint cons4;
    auto cons4Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t s1t_size = rel_tiling_vars[0]->value;
      int64_t bngs1Tb_size = rel_in_shapes[0]->value;
      int64_t B = rel_in_shapes[1]->value;
      int64_t G = rel_in_shapes[2]->value;
      int64_t N = rel_in_shapes[3]->value;
      int64_t S1 = rel_in_shapes[4]->value;
      int64_t value = (bngs1Tb_size - (B * G * N * ceiling((S1 / (s1t_size)))));
      return value;
    };
    cons4.rel_tiling_vars = new TilingVariable*[1];
    cons4.rel_tiling_vars_size = 1u;
    cons4.rel_tiling_vars[0] = &s1t_size;
    cons4.rel_in_shapes = new Variable*[5];
    cons4.rel_in_shapes_size = 5u;
    cons4.rel_in_shapes[0] = &bngs1Tb_size;
    cons4.rel_in_shapes[1] = &B;
    cons4.rel_in_shapes[2] = &G;
    cons4.rel_in_shapes[3] = &N;
    cons4.rel_in_shapes[4] = &S1;
    cons4.type = ConstraintType::MC_MIXED;
    cons4.eval = cons4Eval;
    Constraint cons5;
    auto cons5Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t s1tt_size = rel_tiling_vars[0]->value;
      int64_t s1t_size = rel_tiling_vars[1]->value;
      int64_t value = (s1tt_size - s1t_size);
      return value;
    };
    cons5.rel_tiling_vars = new TilingVariable*[2];
    cons5.rel_tiling_vars_size = 2u;
    cons5.rel_tiling_vars[0] = &s1tt_size;
    cons5.rel_tiling_vars[1] = &s1t_size;
    cons5.type = ConstraintType::LB_MIXED;
    cons5.eval = cons5Eval;
    Constraint cons6;
    auto cons6Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t s1tt2_size = rel_tiling_vars[0]->value;
      int64_t s1t_size = rel_tiling_vars[1]->value;
      int64_t value = (s1tt2_size - s1t_size);
      return value;
    };
    cons6.rel_tiling_vars = new TilingVariable*[2];
    cons6.rel_tiling_vars_size = 2u;
    cons6.rel_tiling_vars[0] = &s1tt2_size;
    cons6.rel_tiling_vars[1] = &s1t_size;
    cons6.type = ConstraintType::LB_MIXED;
    cons6.eval = cons6Eval;
    GetUpperBoundFuncPtr bngs1Tb_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t B = parent_vars[0]->value;
      int64_t G = parent_vars[1]->value;
      int64_t N = parent_vars[2]->value;
      int64_t S1 = parent_vars[3]->value;
      int64_t s1t_size = parent_vars[4]->value;
      if (B == -1 || G == -1 || N == -1 || S1 == -1 || s1t_size == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= (B * G * N * ceiling((S1 / (s1t_size))));
      return upper_bound;
    };
    bngs1Tb_size.upper_bound = bngs1Tb_size_upper_bound;
    bngs1Tb_size.upper_bound_vars = new Variable*[5];
    bngs1Tb_size.upper_bound_vars_size = 5u;
    bngs1Tb_size.upper_bound_vars[0] = &B;
    bngs1Tb_size.upper_bound_vars[1] = &G;
    bngs1Tb_size.upper_bound_vars[2] = &N;
    bngs1Tb_size.upper_bound_vars[3] = &S1;
    bngs1Tb_size.upper_bound_vars[4] = &s1t_size;
    bngs1Tb_size.rel_cons = new Constraint*[1];
    bngs1Tb_size.rel_cons_size = 1u;
    bngs1Tb_size.rel_cons[0] = &cons4;
    s1t_size.align = 128;
    GetUpperBoundFuncPtr s1t_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t S1 = parent_vars[0]->value;
      if (S1 == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= S1;
      return upper_bound;
    };
    s1t_size.upper_bound = s1t_size_upper_bound;
    s1t_size.upper_bound_vars = new Variable*[1];
    s1t_size.upper_bound_vars_size = 1u;
    s1t_size.upper_bound_vars[0] = &S1;
    s1t_size.rel_cons = new Constraint*[6];
    s1t_size.rel_cons_size = 6u;
    s1t_size.rel_cons[0] = &cons0;
    s1t_size.rel_cons[1] = &cons1;
    s1t_size.rel_cons[2] = &cons2;
    s1t_size.rel_cons[3] = &cons4;
    s1t_size.rel_cons[4] = &cons5;
    s1t_size.rel_cons[5] = &cons6;
    s2t_size.align = 256;
    GetUpperBoundFuncPtr s2t_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t S2 = parent_vars[0]->value;
      if (S2 == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= S2;
      return upper_bound;
    };
    s2t_size.upper_bound = s2t_size_upper_bound;
    s2t_size.upper_bound_vars = new Variable*[1];
    s2t_size.upper_bound_vars_size = 1u;
    s2t_size.upper_bound_vars[0] = &S2;
    s2t_size.rel_cons = new Constraint*[2];
    s2t_size.rel_cons_size = 2u;
    s2t_size.rel_cons[0] = &cons0;
    s2t_size.rel_cons[1] = &cons1;
    s1tt_size.align = 8;
    GetUpperBoundFuncPtr s1tt_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t s1t_size = parent_vars[0]->value;
      if (s1t_size == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= s1t_size;
      return upper_bound;
    };
    s1tt_size.upper_bound = s1tt_size_upper_bound;
    s1tt_size.upper_bound_vars = new Variable*[1];
    s1tt_size.upper_bound_vars_size = 1u;
    s1tt_size.upper_bound_vars[0] = &s1t_size;
    s1tt_size.rel_cons = new Constraint*[3];
    s1tt_size.rel_cons_size = 3u;
    s1tt_size.rel_cons[0] = &cons0;
    s1tt_size.rel_cons[1] = &cons1;
    s1tt_size.rel_cons[2] = &cons5;
    s1tt2_size.align = 8;
    GetUpperBoundFuncPtr s1tt2_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t s1t_size = parent_vars[0]->value;
      if (s1t_size == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= s1t_size;
      return upper_bound;
    };
    s1tt2_size.upper_bound = s1tt2_size_upper_bound;
    s1tt2_size.upper_bound_vars = new Variable*[1];
    s1tt2_size.upper_bound_vars_size = 1u;
    s1tt2_size.upper_bound_vars[0] = &s1t_size;
    s1tt2_size.rel_cons = new Constraint*[3];
    s1tt2_size.rel_cons_size = 3u;
    s1tt2_size.rel_cons[0] = &cons0;
    s1tt2_size.rel_cons[1] = &cons1;
    s1tt2_size.rel_cons[2] = &cons6;
    AxesReorderSolverInput input;
    input.input_vars = new Variable*[7];
    input.input_vars_size = 7u;
    input.input_vars[0] = &B;
    input.input_vars[1] = &D;
    input.input_vars[2] = &G;
    input.input_vars[3] = &N;
    input.input_vars[4] = &S1;
    input.input_vars[5] = &S2;
    input.input_vars[6] = &BL;
    input.tiling_vars = new TilingVariable*[5];
    input.tiling_vars_size = 5u;
    input.tiling_vars[0] = &bngs1Tb_size;
    input.tiling_vars[1] = &s1t_size;
    input.tiling_vars[2] = &s2t_size;
    input.tiling_vars[3] = &s1tt_size;
    input.tiling_vars[4] = &s1tt2_size;
    input.all_cons_size = 7u;
    input.all_cons = new Constraint*[7];
    input.all_cons[0] = &cons0;
    input.all_cons[1] = &cons1;
    input.all_cons[2] = &cons2;
    input.all_cons[3] = &cons3;
    input.all_cons[4] = &cons4;
    input.all_cons[5] = &cons5;
    input.all_cons[6] = &cons6;
    input.pure_mc_vars_size = 1u;
    input.pure_mc_vars = new TilingVariable*[1];
    input.pure_mc_vars[0] = &bngs1Tb_size;
    input.local_buffer_vars_size = 4u;
    input.local_buffer_vars = new TilingVariable*[4];
    input.local_buffer_vars[0] = &s1t_size;
    input.local_buffer_vars[1] = &s2t_size;
    input.local_buffer_vars[2] = &s1tt_size;
    input.local_buffer_vars[3] = &s1tt2_size;
    input.core_num = corenum_;
    AxesReorderSolvercase0 *solver = new AxesReorderSolvercase0(input);
    if (!solver->Run()) {
        return false;
    }
    tiling_data.set_bngs1Tb_size(input.pure_mc_vars[0]->value);
    tiling_data.set_s1t_size(input.local_buffer_vars[0]->value);
    tiling_data.set_s2t_size(input.local_buffer_vars[1]->value);
    tiling_data.set_s1tt_size(input.local_buffer_vars[2]->value);
    tiling_data.set_s1tt2_size(input.local_buffer_vars[3]->value);
    return true;
  }

  bool DoTiling(TilingData &tiling_data) {
    if (!ExecuteAxesReorderSolver(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to execute axes reorder solver for tilingCaseId case0.");
      return false;
    }
    OP_LOGD(OP_NAME, "Execute axes reorder solver for tilingCaseId case0 successfully.");

    return true;
  }

  int Gethbm_size(TilingData& tiling_data) {
    double D = tiling_data.get_D();
    double S2 = tiling_data.get_S2();
    double s1t_size = tiling_data.get_s1t_size();
    double s1tt2_size = tiling_data.get_s1tt2_size();
    double s1tt_size = tiling_data.get_s1tt_size();
    double s2t_size = tiling_data.get_s2t_size();

    return ((((S2 * s1t_size) + (s1tt_size * s2t_size) - (S2 * s1tt_size)) * 4) + (8 * D * s1t_size) + (8 * D * s1tt2_size) + (8 * s1t_size * s2t_size));
  }

  int Getub_size(TilingData& tiling_data) {
    double D = tiling_data.get_D();
    double s1t_size = tiling_data.get_s1t_size();
    double s1tt2_size = tiling_data.get_s1tt2_size();
    double s1tt_size = tiling_data.get_s1tt_size();
    double s2t_size = tiling_data.get_s2t_size();

    return ((12 * s1tt_size * s2t_size) + (160 * s1t_size) + (4 * D * s1tt2_size) + Max((4 * s1tt_size * s2t_size), (4 * D * s1tt2_size)));
  }

  int Getblock_dim(TilingData& tiling_data) {
    double B = tiling_data.get_B();
    double G = tiling_data.get_G();
    double N = tiling_data.get_N();
    double S1 = tiling_data.get_S1();
    double bngs1Tb_size = tiling_data.get_bngs1Tb_size();
    double s1t_size = tiling_data.get_s1t_size();

    return Max(0, ceiling((B * G * N * ceiling((S1 / (s1t_size))) / (bngs1Tb_size))));
  }

  void GetWorkSpaceSize(TilingData& tiling_data) {
    OP_LOGD(OP_NAME, "Start setting workspace for case 0.");
    tiling_data.set_workspaceSize(static_cast<uint32_t>(0));
    OP_LOGD(OP_NAME, "Setting workspace to %u for case 0.", tiling_data.get_workspaceSize());
  }

  void TilingSummary(TilingData &tiling_data) {
    OP_LOGI(OP_NAME, "Set the tiling case id to %u.", tiling_data.get_tiling_key());
    OP_LOGI(OP_NAME, "Set bngs1Tb_size to %u.", tiling_data.get_bngs1Tb_size());
    OP_LOGI(OP_NAME, "Set s1t_size to %u.", tiling_data.get_s1t_size());
    OP_LOGI(OP_NAME, "Set s1tt2_size to %u.", tiling_data.get_s1tt2_size());
    OP_LOGI(OP_NAME, "Set s1tt_size to %u.", tiling_data.get_s1tt_size());
    OP_LOGI(OP_NAME, "Set s2t_size to %u.", tiling_data.get_s2t_size());
    OP_LOGI(OP_NAME, "The value of hbm_size is %d.", Gethbm_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of ub_size is %d.", Getub_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of block_dim is %d.", Getblock_dim(tiling_data));
  }

};

TilingCaseImplPtr GetTilingImplPtr(uint32_t tilingCaseId, uint32_t corenum) {
  TilingCaseImplPtr tilingCaseImplPtr = nullptr;
  if (tilingCaseId == 0u) {
    tilingCaseImplPtr = std::make_shared<TilingCase0Impl>(corenum);
  }
  return tilingCaseImplPtr;
}
bool GetTilingKey(TilingData &tiling_data, int32_t tilingCaseId = -1) {
  uint32_t corenum = tiling_data.get_block_dim();
  if (tilingCaseId == -1) {
    OP_LOGI(OP_NAME, "The user didn't specify tilingCaseId, iterate all templates.");
    uint32_t tilingKeys[1] = {0u};
    for (const auto &tilingKey : tilingKeys) {
      TilingCaseImplPtr tilingCaseImplPtr = GetTilingImplPtr(tilingCaseId, corenum);
      if (tilingCaseImplPtr == nullptr) {
        OP_LOGE(OP_NAME, "Pointer for tilingCaseId is null.");
        return false;
      }
      if (tilingCaseImplPtr->GetTiling(tiling_data)) {
        OP_LOGD(OP_NAME, "Finish calculating the tiling data for tilingCaseId %u.", tilingKey);
        return true;
      }
    }
    OP_LOGE(OP_NAME, "No solution found in all templates.");
    return false;
  } else {
    OP_LOGI(OP_NAME, "Calculating the tiling data for tilingCaseId %u.", tilingCaseId);
    TilingCaseImplPtr tilingCaseImplPtr = GetTilingImplPtr(tilingCaseId, corenum);
    if (tilingCaseImplPtr == nullptr) {
      OP_LOGE(OP_NAME, "Pointer for tilingCaseId is null.");
      return false;
    }
    return tilingCaseImplPtr->GetTiling(tiling_data);
  }
}

bool GetTiling(TilingData &tiling_data, int32_t tilingCaseId) {
  OP_LOGI(OP_NAME, "Start tiling. Calculating the tiling data.");
  if (!GetTilingKey(tiling_data, tilingCaseId)) {
    OP_LOGE(OP_NAME, "GetTiling Failed.");
    return false;
  }
  OP_LOGI(OP_NAME, "Filing the calculated tiling data in the context. End tiling.");
  return true;
}

} // namespace optiling

