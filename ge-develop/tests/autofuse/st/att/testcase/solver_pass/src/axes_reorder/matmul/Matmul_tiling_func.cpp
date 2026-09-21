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
#include "Matmul_tiling_data.h"
#define Max(a, b) ((double)(a) > (double)(b) ? (a) : (b))
#define Min(a, b) ((double)(a) < (double)(b) ? (a) : (b))
#define Log(a) (log((double)(a)))
#define MAX_SOLUTION 50
#define OP_NAME "Matmul"

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
  bool GetTiling(MMTilingData &tiling_data) {
    if (!DoTiling(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to do tiling.");
      return false;
    }
    DoApiTiling(tiling_data);
    ExtraTilingData(tiling_data);
    TilingSummary(tiling_data);
    return true;
  }
  virtual double GetPerf(MMTilingData &tiling_data) { return 0.0; }
 protected:
  virtual bool DoTiling(MMTilingData &tiling_data) = 0;
  virtual void DoApiTiling(MMTilingData &tiling_data) {}
  virtual void GetWorkSpaceSize(MMTilingData& tiling_data) {}
  virtual void ExtraTilingData(MMTilingData &tiling_data) {}
  virtual void TilingSummary(MMTilingData &tiling_data) = 0;
  uint32_t corenum_;
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
  int32_t used_core_num = 0;
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

class AxesReorderSolvercase1 : public AxesReorderSolver {
 public:
  explicit AxesReorderSolvercase1(const AxesReorderSolverInput input) : AxesReorderSolver(input) {}
  ~AxesReorderSolvercase1() = default;
  bool CalUsedCoreNum(int32_t &used_core_num, bool &load_balance) override;
};

bool AxesReorderSolvercase1::CalUsedCoreNum(int32_t &used_core_num, bool &load_balance) {
  return true;
}

class TilingCase1Impl : public TilingCaseImpl {
 public:
  TilingCase1Impl(uint32_t corenum) : TilingCaseImpl(corenum) {

  }
 protected:
  bool ExecuteAxesReorderSolver(MMTilingData& tiling_data) {
    Variable k_size;
    k_size.value = tiling_data.get_k_size();
    Variable m_size;
    m_size.value = tiling_data.get_m_size();
    Variable n_size;
    n_size.value = tiling_data.get_n_size();
    Variable block_dim;
    block_dim.value = 0;
    TilingVariable tilen_size;
    TilingVariable tilem_size;
    TilingVariable stepka_size;
    TilingVariable stepkb_size;
    TilingVariable basek_size;
    TilingVariable basen_size;
    TilingVariable basem_size;
    int64_t l1_size = tiling_data.get_l1_size();
    Constraint cons0;
    auto cons0Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t stepkb_size = rel_tiling_vars[0]->value;
      int64_t basen_size = rel_tiling_vars[1]->value;
      int64_t stepka_size = rel_tiling_vars[2]->value;
      int64_t basem_size = rel_tiling_vars[3]->value;
      int64_t value = ((4 * basem_size * stepka_size) + (4 * basen_size * stepkb_size)) - rel_hw_spec;
      return value;
    };
    cons0.rel_tiling_vars = new TilingVariable*[4];
    cons0.rel_tiling_vars_size = 4u;
    cons0.rel_tiling_vars[0] = &stepkb_size;
    cons0.rel_tiling_vars[1] = &basen_size;
    cons0.rel_tiling_vars[2] = &stepka_size;
    cons0.rel_tiling_vars[3] = &basem_size;
    cons0.rel_hw_spec = l1_size;
    cons0.type = ConstraintType::LOCAL_BUFFER;
    cons0.eval = cons0Eval;
    int64_t l2_size = tiling_data.get_l2_size();
    Constraint cons1;
    auto cons1Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t tilen_size = rel_tiling_vars[0]->value;
      int64_t tilem_size = rel_tiling_vars[1]->value;
      int64_t k_size = rel_in_shapes[0]->value;
      int64_t value = (((tilem_size + tilen_size) * 2 * k_size) + (2 * tilem_size * tilen_size)) - rel_hw_spec;
      return value;
    };
    cons1.rel_tiling_vars = new TilingVariable*[2];
    cons1.rel_tiling_vars_size = 2u;
    cons1.rel_tiling_vars[0] = &tilen_size;
    cons1.rel_tiling_vars[1] = &tilem_size;
    cons1.rel_in_shapes = new Variable*[1];
    cons1.rel_in_shapes_size = 1u;
    cons1.rel_in_shapes[0] = &k_size;
    cons1.rel_hw_spec = l2_size;
    cons1.type = ConstraintType::LOCAL_BUFFER;
    cons1.eval = cons1Eval;
    int64_t l0a_size = tiling_data.get_l0a_size();
    Constraint cons2;
    auto cons2Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t basek_size = rel_tiling_vars[0]->value;
      int64_t basem_size = rel_tiling_vars[1]->value;
      int64_t value = (4 * basek_size * basem_size) - rel_hw_spec;
      return value;
    };
    cons2.rel_tiling_vars = new TilingVariable*[2];
    cons2.rel_tiling_vars_size = 2u;
    cons2.rel_tiling_vars[0] = &basek_size;
    cons2.rel_tiling_vars[1] = &basem_size;
    cons2.rel_hw_spec = l0a_size;
    cons2.type = ConstraintType::LOCAL_BUFFER;
    cons2.eval = cons2Eval;
    int64_t l0b_size = tiling_data.get_l0b_size();
    Constraint cons3;
    auto cons3Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t basek_size = rel_tiling_vars[0]->value;
      int64_t basen_size = rel_tiling_vars[1]->value;
      int64_t value = (4 * basek_size * basen_size) - rel_hw_spec;
      return value;
    };
    cons3.rel_tiling_vars = new TilingVariable*[2];
    cons3.rel_tiling_vars_size = 2u;
    cons3.rel_tiling_vars[0] = &basek_size;
    cons3.rel_tiling_vars[1] = &basen_size;
    cons3.rel_hw_spec = l0b_size;
    cons3.type = ConstraintType::LOCAL_BUFFER;
    cons3.eval = cons3Eval;
    int64_t l0c_size = tiling_data.get_l0c_size();
    Constraint cons4;
    auto cons4Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t basen_size = rel_tiling_vars[0]->value;
      int64_t basem_size = rel_tiling_vars[1]->value;
      int64_t value = (4 * basem_size * basen_size) - rel_hw_spec;
      return value;
    };
    cons4.rel_tiling_vars = new TilingVariable*[2];
    cons4.rel_tiling_vars_size = 2u;
    cons4.rel_tiling_vars[0] = &basen_size;
    cons4.rel_tiling_vars[1] = &basem_size;
    cons4.rel_hw_spec = l0c_size;
    cons4.type = ConstraintType::LOCAL_BUFFER;
    cons4.eval = cons4Eval;
    Constraint cons5;
    auto cons5Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t tilem_size = rel_tiling_vars[0]->value;
      int64_t m_size = rel_in_shapes[0]->value;
      int64_t value = (tilem_size - m_size);
      return value;
    };
    cons5.rel_tiling_vars = new TilingVariable*[1];
    cons5.rel_tiling_vars_size = 1u;
    cons5.rel_tiling_vars[0] = &tilem_size;
    cons5.rel_in_shapes = new Variable*[1];
    cons5.rel_in_shapes_size = 1u;
    cons5.rel_in_shapes[0] = &m_size;
    cons5.type = ConstraintType::LB_MIXED;
    cons5.eval = cons5Eval;
    Constraint cons6;
    auto cons6Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t tilen_size = rel_tiling_vars[0]->value;
      int64_t n_size = rel_in_shapes[0]->value;
      int64_t value = (tilen_size - n_size);
      return value;
    };
    cons6.rel_tiling_vars = new TilingVariable*[1];
    cons6.rel_tiling_vars_size = 1u;
    cons6.rel_tiling_vars[0] = &tilen_size;
    cons6.rel_in_shapes = new Variable*[1];
    cons6.rel_in_shapes_size = 1u;
    cons6.rel_in_shapes[0] = &n_size;
    cons6.type = ConstraintType::LB_MIXED;
    cons6.eval = cons6Eval;
    Constraint cons7;
    auto cons7Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t stepka_size = rel_tiling_vars[0]->value;
      int64_t k_size = rel_in_shapes[0]->value;
      int64_t value = (stepka_size - k_size);
      return value;
    };
    cons7.rel_tiling_vars = new TilingVariable*[1];
    cons7.rel_tiling_vars_size = 1u;
    cons7.rel_tiling_vars[0] = &stepka_size;
    cons7.rel_in_shapes = new Variable*[1];
    cons7.rel_in_shapes_size = 1u;
    cons7.rel_in_shapes[0] = &k_size;
    cons7.type = ConstraintType::LB_MIXED;
    cons7.eval = cons7Eval;
    tilen_size.align = 16;
    GetUpperBoundFuncPtr tilen_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t n_size = parent_vars[0]->value;
      if (n_size == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= n_size;
      return upper_bound;
    };
    tilen_size.upper_bound = tilen_size_upper_bound;
    tilen_size.upper_bound_vars = new Variable * [1];
    tilen_size.upper_bound_vars_size = 1u;
    tilen_size.upper_bound_vars[0] = &n_size;
    tilen_size.rel_cons = new Constraint*[2];
    tilen_size.rel_cons_size = 2u;
    tilen_size.rel_cons[0] = &cons1;
    tilen_size.rel_cons[1] = &cons6;
    tilem_size.align = 16;
    GetUpperBoundFuncPtr tilem_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t m_size = parent_vars[0]->value;
      if (m_size == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= m_size;
      return upper_bound;
    };
    tilem_size.upper_bound = tilem_size_upper_bound;
    tilem_size.upper_bound_vars = new Variable * [1];
    tilem_size.upper_bound_vars_size = 1u;
    tilem_size.upper_bound_vars[0] = &m_size;
    tilem_size.rel_cons = new Constraint*[2];
    tilem_size.rel_cons_size = 2u;
    tilem_size.rel_cons[0] = &cons1;
    tilem_size.rel_cons[1] = &cons5;
    stepka_size.align = 256;
    GetUpperBoundFuncPtr stepka_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t k_size = parent_vars[0]->value;
      if (k_size == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= k_size;
      return upper_bound;
    };
    stepka_size.upper_bound = stepka_size_upper_bound;
    stepka_size.upper_bound_vars = new Variable * [1];
    stepka_size.upper_bound_vars_size = 1u;
    stepka_size.upper_bound_vars[0] = &k_size;
    stepka_size.rel_cons = new Constraint*[2];
    stepka_size.rel_cons_size = 2u;
    stepka_size.rel_cons[0] = &cons0;
    stepka_size.rel_cons[1] = &cons7;
    stepkb_size.align = 16;
    GetUpperBoundFuncPtr stepkb_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t stepka_size = parent_vars[0]->value;
      if (stepka_size == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= stepka_size;
      return upper_bound;
    };
    stepkb_size.upper_bound = stepkb_size_upper_bound;
    stepkb_size.upper_bound_vars = new Variable * [1];
    stepkb_size.upper_bound_vars_size = 1u;
    stepkb_size.upper_bound_vars[0] = &stepka_size;
    stepkb_size.rel_cons = new Constraint*[1];
    stepkb_size.rel_cons_size = 1u;
    stepkb_size.rel_cons[0] = &cons0;
    basek_size.align = 16;
    GetUpperBoundFuncPtr basek_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t stepkb_size = parent_vars[0]->value;
      if (stepkb_size == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= stepkb_size;
      return upper_bound;
    };
    basek_size.upper_bound = basek_size_upper_bound;
    basek_size.upper_bound_vars = new Variable * [1];
    basek_size.upper_bound_vars_size = 1u;
    basek_size.upper_bound_vars[0] = &stepkb_size;
    basek_size.rel_cons = new Constraint*[2];
    basek_size.rel_cons_size = 2u;
    basek_size.rel_cons[0] = &cons2;
    basek_size.rel_cons[1] = &cons3;
    basen_size.align = 16;
    GetUpperBoundFuncPtr basen_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t tilen_size = parent_vars[0]->value;
      if (tilen_size == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= tilen_size;
      return upper_bound;
    };
    basen_size.upper_bound = basen_size_upper_bound;
    basen_size.upper_bound_vars = new Variable * [1];
    basen_size.upper_bound_vars_size = 1u;
    basen_size.upper_bound_vars[0] = &tilen_size;
    basen_size.rel_cons = new Constraint*[3];
    basen_size.rel_cons_size = 3u;
    basen_size.rel_cons[0] = &cons0;
    basen_size.rel_cons[1] = &cons3;
    basen_size.rel_cons[2] = &cons4;
    basem_size.align = 16;
    GetUpperBoundFuncPtr basem_size_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t tilem_size = parent_vars[0]->value;
      if (tilem_size == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= tilem_size;
      return upper_bound;
    };
    basem_size.upper_bound = basem_size_upper_bound;
    basem_size.upper_bound_vars = new Variable * [1];
    basem_size.upper_bound_vars_size = 1u;
    basem_size.upper_bound_vars[0] = &tilem_size;
    basem_size.rel_cons = new Constraint*[3];
    basem_size.rel_cons_size = 3u;
    basem_size.rel_cons[0] = &cons0;
    basem_size.rel_cons[1] = &cons2;
    basem_size.rel_cons[2] = &cons4;
    AxesReorderSolverInput input;
    input.input_vars = new Variable*[4];
    input.input_vars_size = 4u;
    input.input_vars[0] = &k_size;
    input.input_vars[1] = &m_size;
    input.input_vars[2] = &n_size;
    input.input_vars[3] = &block_dim;
    input.tiling_vars = new TilingVariable*[7];
    input.tiling_vars_size = 7u;
    input.tiling_vars[0] = &tilen_size;
    input.tiling_vars[1] = &tilem_size;
    input.tiling_vars[2] = &stepka_size;
    input.tiling_vars[3] = &stepkb_size;
    input.tiling_vars[4] = &basek_size;
    input.tiling_vars[5] = &basen_size;
    input.tiling_vars[6] = &basem_size;
    input.all_cons_size = 8u;
    input.all_cons = new Constraint*[8];
    input.all_cons[0] = &cons0;
    input.all_cons[1] = &cons1;
    input.all_cons[2] = &cons2;
    input.all_cons[3] = &cons3;
    input.all_cons[4] = &cons4;
    input.all_cons[5] = &cons5;
    input.all_cons[6] = &cons6;
    input.all_cons[7] = &cons7;
    input.local_buffer_vars_size = 7u;
    input.local_buffer_vars = new TilingVariable*[7];
    input.local_buffer_vars[0] = &tilen_size;
    input.local_buffer_vars[1] = &tilem_size;
    input.local_buffer_vars[2] = &stepka_size;
    input.local_buffer_vars[3] = &stepkb_size;
    input.local_buffer_vars[4] = &basek_size;
    input.local_buffer_vars[5] = &basen_size;
    input.local_buffer_vars[6] = &basem_size;
    input.core_num = corenum_;
    AxesReorderSolvercase1* solver = new AxesReorderSolvercase1(input);
    if (!solver->Run()) {
        return false;
    }
    tiling_data.set_tilen_size(input.local_buffer_vars[0]->value);
    tiling_data.set_tilem_size(input.local_buffer_vars[1]->value);
    tiling_data.set_stepka_size(input.local_buffer_vars[2]->value);
    tiling_data.set_stepkb_size(input.local_buffer_vars[3]->value);
    tiling_data.set_basek_size(input.local_buffer_vars[4]->value);
    tiling_data.set_basen_size(input.local_buffer_vars[5]->value);
    tiling_data.set_basem_size(input.local_buffer_vars[6]->value);
    return true;
  }

  bool DoTiling(MMTilingData &tiling_data) {
    if (!ExecuteAxesReorderSolver(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to execute axes reorder solver for tilingCaseId case1.");
      return false;
    }
    OP_LOGD(OP_NAME, "Execute axes reorder solver for tilingCaseId case1 successfully.");

    return true;
  }

  int Getl1_size(MMTilingData& tiling_data) {
    double basem_size = tiling_data.get_basem_size();
    double basen_size = tiling_data.get_basen_size();
    double stepka_size = tiling_data.get_stepka_size();
    double stepkb_size = tiling_data.get_stepkb_size();

    return ((4 * basem_size * stepka_size) + (4 * basen_size * stepkb_size));
  }

  int Getl2_size(MMTilingData& tiling_data) {
    double k_size = tiling_data.get_k_size();
    double tilem_size = tiling_data.get_tilem_size();
    double tilen_size = tiling_data.get_tilen_size();

    return (((tilem_size + tilen_size) * 2 * k_size) + (2 * tilem_size * tilen_size));
  }

  int Getl0a_size(MMTilingData& tiling_data) {
    double basek_size = tiling_data.get_basek_size();
    double basem_size = tiling_data.get_basem_size();

    return (4 * basek_size * basem_size);
  }

  int Getl0b_size(MMTilingData& tiling_data) {
    double basek_size = tiling_data.get_basek_size();
    double basen_size = tiling_data.get_basen_size();

    return (4 * basek_size * basen_size);
  }

  int Getl0c_size(MMTilingData& tiling_data) {
    double basem_size = tiling_data.get_basem_size();
    double basen_size = tiling_data.get_basen_size();

    return (4 * basem_size * basen_size);
  }

  void UpdateGeneralTilingData(MMTilingData& tiling_data) {
    tiling_data.set_block_dim(1);
  }

  void UpdateAxesTilingData(MMTilingData& tiling_data) {
    tiling_data.set_tilen_loop_num(((tiling_data.get_n_size() + tiling_data.get_tilen_size()) - 1) / tiling_data.get_tilen_size());
    tiling_data.set_tilem_loop_num(((tiling_data.get_m_size() + tiling_data.get_tilem_size()) - 1) / tiling_data.get_tilem_size());
    tiling_data.set_basek_loop_num(((tiling_data.get_stepkb_size() + tiling_data.get_basek_size()) - 1) / tiling_data.get_basek_size());
    tiling_data.set_basem_loop_num(((tiling_data.get_tilem_size() + tiling_data.get_basem_size()) - 1) / tiling_data.get_basem_size());
    tiling_data.set_stepkb_loop_num(((tiling_data.get_stepka_size() + tiling_data.get_stepkb_size()) - 1) / tiling_data.get_stepkb_size());
    tiling_data.set_stepka_loop_num(((tiling_data.get_k_size() + tiling_data.get_stepka_size()) - 1) / tiling_data.get_stepka_size());
    tiling_data.set_basen_loop_num(((tiling_data.get_tilen_size() + tiling_data.get_basen_size()) - 1) / tiling_data.get_basen_size());
    tiling_data.set_tilem_tail_size((tiling_data.get_m_size() % tiling_data.get_tilem_size()) == 0 ? tiling_data.get_tilem_size() : (tiling_data.get_m_size() % tiling_data.get_tilem_size()));
    tiling_data.set_basek_tail_size((tiling_data.get_stepkb_size() % tiling_data.get_basek_size()) == 0 ? tiling_data.get_basek_size() : (tiling_data.get_stepkb_size() % tiling_data.get_basek_size()));
    tiling_data.set_tilen_tail_size((tiling_data.get_n_size() % tiling_data.get_tilen_size()) == 0 ? tiling_data.get_tilen_size() : (tiling_data.get_n_size() % tiling_data.get_tilen_size()));
    tiling_data.set_stepkb_tail_size((tiling_data.get_stepka_size() % tiling_data.get_stepkb_size()) == 0 ? tiling_data.get_stepkb_size() : (tiling_data.get_stepka_size() % tiling_data.get_stepkb_size()));
    tiling_data.set_basem_tail_size((tiling_data.get_tilem_size() % tiling_data.get_basem_size()) == 0 ? tiling_data.get_basem_size() : (tiling_data.get_tilem_size() % tiling_data.get_basem_size()));
    tiling_data.set_stepka_tail_size((tiling_data.get_k_size() % tiling_data.get_stepka_size()) == 0 ? tiling_data.get_stepka_size() : (tiling_data.get_k_size() % tiling_data.get_stepka_size()));
    tiling_data.set_basen_tail_size((tiling_data.get_tilen_size() % tiling_data.get_basen_size()) == 0 ? tiling_data.get_basen_size() : (tiling_data.get_tilen_size() % tiling_data.get_basen_size()));
    tiling_data.set_tilem_tail_tile_basem_loop_num(((tiling_data.get_tilem_tail_size() + tiling_data.get_basem_size()) - 1) / tiling_data.get_basem_size());
    tiling_data.set_tilen_tail_tile_basen_loop_num(((tiling_data.get_tilen_tail_size() + tiling_data.get_basen_size()) - 1) / tiling_data.get_basen_size());
    tiling_data.set_stepka_tail_tile_stepkb_loop_num(((tiling_data.get_stepka_tail_size() + tiling_data.get_stepkb_size()) - 1) / tiling_data.get_stepkb_size());
    tiling_data.set_stepkb_tail_tile_basek_loop_num(((tiling_data.get_stepkb_tail_size() + tiling_data.get_basek_size()) - 1) / tiling_data.get_basek_size());
    tiling_data.set_tilen_tail_tile_basen_tail_size((tiling_data.get_tilen_tail_size() % tiling_data.get_basen_size()) == 0 ? tiling_data.get_basen_size() : (tiling_data.get_tilen_tail_size() % tiling_data.get_basen_size()));
    tiling_data.set_stepka_tail_tile_stepkb_tail_size((tiling_data.get_stepka_tail_size() % tiling_data.get_stepkb_size()) == 0 ? tiling_data.get_stepkb_size() : (tiling_data.get_stepka_tail_size() % tiling_data.get_stepkb_size()));
    tiling_data.set_tilem_tail_tile_basem_tail_size((tiling_data.get_tilem_tail_size() % tiling_data.get_basem_size()) == 0 ? tiling_data.get_basem_size() : (tiling_data.get_tilem_tail_size() % tiling_data.get_basem_size()));
    tiling_data.set_stepkb_tail_tile_basek_tail_size((tiling_data.get_stepkb_tail_size() % tiling_data.get_basek_size()) == 0 ? tiling_data.get_basek_size() : (tiling_data.get_stepkb_tail_size() % tiling_data.get_basek_size()));
  }

  void SetQ1(MMTilingData &tiling_data) {
    const auto n_size = tiling_data.get_n_size();
    const auto m_size = tiling_data.get_m_size();
    tiling_data.set_Q1((m_size + n_size));
  }

  void SetMATMUL_OUTPUT1(MMTilingData &tiling_data) {
    const auto n_size = tiling_data.get_n_size();
    const auto m_size = tiling_data.get_m_size();
    tiling_data.set_MATMUL_OUTPUT1((m_size + n_size));
  }

  void ComputeOptionParam(MMTilingData &tiling_data) {
    SetQ1(tiling_data);
    SetMATMUL_OUTPUT1(tiling_data);

  }

  void ExtraTilingData(MMTilingData &tiling_data) {
    OP_LOGD(OP_NAME, "Start executing extra tiling for tilingCaseId 1.");
		UpdateGeneralTilingData(tiling_data);

    ComputeOptionParam(tiling_data);
		UpdateAxesTilingData(tiling_data);

    OP_LOGD(OP_NAME, "Execute extra tiling for tilingCaseId 1 successfully.");
  }

  void GetWorkSpaceSize(MMTilingData& tiling_data) {
    OP_LOGD(OP_NAME, "Start setting workspace for case 1.");
    tiling_data.set_workspaceSize(static_cast<uint32_t>(0));
    OP_LOGD(OP_NAME, "Setting workspace to %u for case 1.", tiling_data.get_workspaceSize());
  }

  void TilingSummary(MMTilingData &tiling_data) {
    OP_LOGI(OP_NAME, "Set basek_size to %u.", tiling_data.get_basek_size());
    OP_LOGI(OP_NAME, "Set basem_size to %u.", tiling_data.get_basem_size());
    OP_LOGI(OP_NAME, "Set basen_size to %u.", tiling_data.get_basen_size());
    OP_LOGI(OP_NAME, "Set stepka_size to %u.", tiling_data.get_stepka_size());
    OP_LOGI(OP_NAME, "Set stepkb_size to %u.", tiling_data.get_stepkb_size());
    OP_LOGI(OP_NAME, "Set tilem_size to %u.", tiling_data.get_tilem_size());
    OP_LOGI(OP_NAME, "Set tilen_size to %u.", tiling_data.get_tilen_size());
    OP_LOGI(OP_NAME, "The value of l1_size is %d.", Getl1_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of l2_size is %d.", Getl2_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of l0a_size is %d.", Getl0a_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of l0b_size is %d.", Getl0b_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of l0c_size is %d.", Getl0c_size(tiling_data));
  }

};

TilingCaseImplPtr GetTilingImplPtr(uint32_t tilingCaseId, uint32_t corenum) {
  TilingCaseImplPtr tilingCaseImplPtr = nullptr;
  if (tilingCaseId == 1u) {
    tilingCaseImplPtr = std::make_shared<TilingCase1Impl>(corenum);
  }
  return tilingCaseImplPtr;
}
bool GetTilingKey(MMTilingData &tiling_data, int32_t tilingCaseId = -1) {
  uint32_t corenum = tiling_data.get_block_dim();
  if (tilingCaseId == -1) {
    OP_LOGI(OP_NAME, "The user didn't specify tilingCaseId, iterate all templates.");
    uint32_t tilingKeys[1] = {1u};
    for (const auto &tilingKey : tilingKeys) {
      TilingCaseImplPtr tilingCaseImplPtr = GetTilingImplPtr(tilingKey, corenum);
      if (tilingCaseImplPtr == nullptr) {
        OP_LOGE(OP_NAME, "Pointer for tilingCaseId is null.");
        return false;
      }
      if (tilingCaseImplPtr->GetTiling(tiling_data)) {
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
    if (tilingCaseImplPtr->GetTiling(tiling_data)) {
      tiling_data.set_tiling_key(tilingCaseId);
      return true;
    }
  }
  return false;
}

bool GetTiling(MMTilingData &tiling_data, int32_t tilingCaseId) {
  OP_LOGI(OP_NAME, "Start tiling. Calculating the tiling data.");
  if (!GetTilingKey(tiling_data, tilingCaseId)) {
    OP_LOGE(OP_NAME, "GetTiling Failed.");
    return false;
  }
  OP_LOGI(OP_NAME, "Filing the calculated tiling data in the context. End tiling.");
  return true;
}

} // namespace optiling

