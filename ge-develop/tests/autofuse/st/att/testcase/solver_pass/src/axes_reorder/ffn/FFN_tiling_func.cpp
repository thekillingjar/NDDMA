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
#include "FFN_tiling_data.h"
#define Max(a, b) ((double)(a) > (double)(b) ? (a) : (b))
#define Min(a, b) ((double)(a) < (double)(b) ? (a) : (b))
#define Log(a) (log((double)(a)))
#define MAX_SOLUTION 50
#define OP_NAME "FFN"

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
  bool GetTiling(FFNTilingData &tiling_data) {
    if (!DoTiling(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to do tiling.");
      return false;
    }
    DoApiTiling(tiling_data);
    ExtraTilingData(tiling_data);
    TilingSummary(tiling_data);
    return true;
  }
  virtual double GetPerf(FFNTilingData &tiling_data) { return 0.0; }
 protected:
  virtual bool DoTiling(FFNTilingData &tiling_data) = 0;
  virtual void DoApiTiling(FFNTilingData &tiling_data) {}
  virtual void GetWorkSpaceSize(FFNTilingData& tiling_data) {}
  virtual void ExtraTilingData(FFNTilingData &tiling_data) {}
  virtual void TilingSummary(FFNTilingData &tiling_data) = 0;
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

class AxesReorderSolvercase0 : public AxesReorderSolver {
 public:
  explicit AxesReorderSolvercase0(const AxesReorderSolverInput input) : AxesReorderSolver(input) {}
  ~AxesReorderSolvercase0() = default;
  bool CalUsedCoreNum(int32_t &used_core_num, bool &load_balance) override;
};

bool AxesReorderSolvercase0::CalUsedCoreNum(int32_t &used_core_num, bool &load_balance) {
  return true;
}

class TilingCase0Impl : public TilingCaseImpl {
 public:
  TilingCase0Impl(uint32_t corenum) : TilingCaseImpl(corenum) {

  }
 protected:
  bool ExecuteAxesReorderSolver(FFNTilingData& tiling_data) {
    Variable K1;
    K1.value = tiling_data.get_K1();
    Variable N1;
    N1.value = tiling_data.get_N1();
    Variable N2;
    N2.value = tiling_data.get_N2();
    Variable maxTokens;
    maxTokens.value = tiling_data.get_maxTokens();
    TilingVariable base_n1;
    TilingVariable base_n2;
    TilingVariable base_m1;
    TilingVariable base_m2;
    TilingVariable ub_m;
    int64_t l0c_size = tiling_data.get_l0c_size();
    Constraint cons0;
    auto cons0Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t base_n1 = rel_tiling_vars[0]->value;
      int64_t base_m1 = rel_tiling_vars[1]->value;
      int64_t base_n2 = rel_tiling_vars[2]->value;
      int64_t base_m2 = rel_tiling_vars[3]->value;
      int64_t value = Max((4 * base_m1 * base_n1), (4 * base_m2 * base_n2)) - rel_hw_spec;
      return value;
    };
    cons0.rel_tiling_vars = new TilingVariable*[4];
    cons0.rel_tiling_vars_size = 4u;
    cons0.rel_tiling_vars[0] = &base_n1;
    cons0.rel_tiling_vars[1] = &base_m1;
    cons0.rel_tiling_vars[2] = &base_n2;
    cons0.rel_tiling_vars[3] = &base_m2;
    cons0.rel_hw_spec = l0c_size;
    cons0.type = ConstraintType::LOCAL_BUFFER;
    cons0.eval = cons0Eval;
    int64_t ub_size = tiling_data.get_ub_size();
    Constraint cons1;
    auto cons1Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t ub_m = rel_tiling_vars[0]->value;
      int64_t base_n1 = rel_tiling_vars[1]->value;
      int64_t value = (4 * base_n1 * ub_m) - rel_hw_spec;
      return value;
    };
    cons1.rel_tiling_vars = new TilingVariable*[2];
    cons1.rel_tiling_vars_size = 2u;
    cons1.rel_tiling_vars[0] = &ub_m;
    cons1.rel_tiling_vars[1] = &base_n1;
    cons1.rel_hw_spec = ub_size;
    cons1.type = ConstraintType::LOCAL_BUFFER;
    cons1.eval = cons1Eval;
    int64_t btbuf_size = tiling_data.get_btbuf_size();
    Constraint cons2;
    auto cons2Eval = [](TilingVariable **rel_tiling_vars, Variable **rel_in_shapes, int64_t rel_hw_spec) {
      int64_t base_n1 = rel_tiling_vars[0]->value;
      int64_t base_n2 = rel_tiling_vars[1]->value;
      int64_t value = Max((4 * base_n1), (4 * base_n2)) - rel_hw_spec;
      return value;
    };
    cons2.rel_tiling_vars = new TilingVariable*[2];
    cons2.rel_tiling_vars_size = 2u;
    cons2.rel_tiling_vars[0] = &base_n1;
    cons2.rel_tiling_vars[1] = &base_n2;
    cons2.rel_hw_spec = btbuf_size;
    cons2.type = ConstraintType::LOCAL_BUFFER;
    cons2.eval = cons2Eval;
    base_n1.align = 8;
    GetUpperBoundFuncPtr base_n1_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t N1 = parent_vars[0]->value;
      if (N1 == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= N1;
      return upper_bound;
    };
    base_n1.upper_bound = base_n1_upper_bound;
    base_n1.upper_bound_vars = new Variable * [1];
    base_n1.upper_bound_vars_size = 1u;
    base_n1.upper_bound_vars[0] = &N1;
    base_n1.rel_cons = new Constraint*[3];
    base_n1.rel_cons_size = 3u;
    base_n1.rel_cons[0] = &cons0;
    base_n1.rel_cons[1] = &cons1;
    base_n1.rel_cons[2] = &cons2;
    base_n2.align = 8;
    GetUpperBoundFuncPtr base_n2_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t N2 = parent_vars[0]->value;
      if (N2 == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= N2;
      return upper_bound;
    };
    base_n2.upper_bound = base_n2_upper_bound;
    base_n2.upper_bound_vars = new Variable * [1];
    base_n2.upper_bound_vars_size = 1u;
    base_n2.upper_bound_vars[0] = &N2;
    base_n2.rel_cons = new Constraint*[2];
    base_n2.rel_cons_size = 2u;
    base_n2.rel_cons[0] = &cons0;
    base_n2.rel_cons[1] = &cons2;
    base_m1.align = 8;
    GetUpperBoundFuncPtr base_m1_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t maxTokens = parent_vars[0]->value;
      if (maxTokens == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= maxTokens;
      return upper_bound;
    };
    base_m1.upper_bound = base_m1_upper_bound;
    base_m1.upper_bound_vars = new Variable * [1];
    base_m1.upper_bound_vars_size = 1u;
    base_m1.upper_bound_vars[0] = &maxTokens;
    base_m1.rel_cons = new Constraint*[1];
    base_m1.rel_cons_size = 1u;
    base_m1.rel_cons[0] = &cons0;
    base_m2.align = 8;
    GetUpperBoundFuncPtr base_m2_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t maxTokens = parent_vars[0]->value;
      if (maxTokens == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= maxTokens;
      return upper_bound;
    };
    base_m2.upper_bound = base_m2_upper_bound;
    base_m2.upper_bound_vars = new Variable * [1];
    base_m2.upper_bound_vars_size = 1u;
    base_m2.upper_bound_vars[0] = &maxTokens;
    base_m2.rel_cons = new Constraint*[1];
    base_m2.rel_cons_size = 1u;
    base_m2.rel_cons[0] = &cons0;
    ub_m.align = 8;
    GetUpperBoundFuncPtr ub_m_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      int64_t base_m1 = parent_vars[0]->value;
      if (base_m1 == -1) {
        return static_cast<int64_t>(-1);
      }
      upper_bound *= base_m1;
      return upper_bound;
    };
    ub_m.upper_bound = ub_m_upper_bound;
    ub_m.upper_bound_vars = new Variable * [1];
    ub_m.upper_bound_vars_size = 1u;
    ub_m.upper_bound_vars[0] = &base_m1;
    ub_m.rel_cons = new Constraint*[1];
    ub_m.rel_cons_size = 1u;
    ub_m.rel_cons[0] = &cons1;
    AxesReorderSolverInput input;
    input.input_vars = new Variable*[4];
    input.input_vars_size = 4u;
    input.input_vars[0] = &K1;
    input.input_vars[1] = &N1;
    input.input_vars[2] = &N2;
    input.input_vars[3] = &maxTokens;
    input.tiling_vars = new TilingVariable*[5];
    input.tiling_vars_size = 5u;
    input.tiling_vars[0] = &base_n1;
    input.tiling_vars[1] = &base_n2;
    input.tiling_vars[2] = &base_m1;
    input.tiling_vars[3] = &base_m2;
    input.tiling_vars[4] = &ub_m;
    input.all_cons_size = 3u;
    input.all_cons = new Constraint*[3];
    input.all_cons[0] = &cons0;
    input.all_cons[1] = &cons1;
    input.all_cons[2] = &cons2;
    input.local_buffer_vars_size = 5u;
    input.local_buffer_vars = new TilingVariable*[5];
    input.local_buffer_vars[0] = &base_n1;
    input.local_buffer_vars[1] = &base_n2;
    input.local_buffer_vars[2] = &base_m1;
    input.local_buffer_vars[3] = &base_m2;
    input.local_buffer_vars[4] = &ub_m;
    input.core_num = corenum_;
    AxesReorderSolvercase0* solver = new AxesReorderSolvercase0(input);
    if (!solver->Run()) {
        return false;
    }
    tiling_data.set_base_n1(input.local_buffer_vars[0]->value);
    tiling_data.set_base_n2(input.local_buffer_vars[1]->value);
    tiling_data.set_base_m1(input.local_buffer_vars[2]->value);
    tiling_data.set_base_m2(input.local_buffer_vars[3]->value);
    tiling_data.set_ub_m(input.local_buffer_vars[4]->value);
    return true;
  }

  bool DoTiling(FFNTilingData &tiling_data) {
    if (!ExecuteAxesReorderSolver(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to execute axes reorder solver for tilingCaseId case0.");
      return false;
    }
    OP_LOGD(OP_NAME, "Execute axes reorder solver for tilingCaseId case0 successfully.");

    return true;
  }

  int Getl0c_size(FFNTilingData& tiling_data) {
    double base_m1 = tiling_data.get_base_m1();
    double base_m2 = tiling_data.get_base_m2();
    double base_n1 = tiling_data.get_base_n1();
    double base_n2 = tiling_data.get_base_n2();

    return Max((4 * base_m1 * base_n1), (4 * base_m2 * base_n2));
  }

  int Getub_size(FFNTilingData& tiling_data) {
    double base_n1 = tiling_data.get_base_n1();
    double ub_m = tiling_data.get_ub_m();

    return (4 * base_n1 * ub_m);
  }

  int Getbtbuf_size(FFNTilingData& tiling_data) {
    double base_n1 = tiling_data.get_base_n1();
    double base_n2 = tiling_data.get_base_n2();

    return Max((4 * base_n1), (4 * base_n2));
  }

  void UpdateGeneralTilingData(FFNTilingData& tiling_data) {
    tiling_data.set_block_dim(1);
  }

  void UpdateAxesTilingData(FFNTilingData& tiling_data) {
    tiling_data.set_ub_m_loop_num(((tiling_data.get_base_m1() + tiling_data.get_ub_m()) - 1) / tiling_data.get_ub_m());
    tiling_data.set_base_m1_loop_num(((tiling_data.get_maxTokens() + tiling_data.get_base_m1()) - 1) / tiling_data.get_base_m1());
    tiling_data.set_base_n1_loop_num(((tiling_data.get_N1() + tiling_data.get_base_n1()) - 1) / tiling_data.get_base_n1());
    tiling_data.set_base_n2_loop_num(((tiling_data.get_N2() + tiling_data.get_base_n2()) - 1) / tiling_data.get_base_n2());
    tiling_data.set_base_m2_loop_num(((tiling_data.get_maxTokens() + tiling_data.get_base_m2()) - 1) / tiling_data.get_base_m2());
    tiling_data.set_ub_m_tail_size((tiling_data.get_base_m1() % tiling_data.get_ub_m()) == 0 ? tiling_data.get_ub_m() : (tiling_data.get_base_m1() % tiling_data.get_ub_m()));
    tiling_data.set_base_m1_tail_size((tiling_data.get_maxTokens() % tiling_data.get_base_m1()) == 0 ? tiling_data.get_base_m1() : (tiling_data.get_maxTokens() % tiling_data.get_base_m1()));
    tiling_data.set_base_n1_tail_size((tiling_data.get_N1() % tiling_data.get_base_n1()) == 0 ? tiling_data.get_base_n1() : (tiling_data.get_N1() % tiling_data.get_base_n1()));
    tiling_data.set_base_n2_tail_size((tiling_data.get_N2() % tiling_data.get_base_n2()) == 0 ? tiling_data.get_base_n2() : (tiling_data.get_N2() % tiling_data.get_base_n2()));
    tiling_data.set_base_m2_tail_size((tiling_data.get_maxTokens() % tiling_data.get_base_m2()) == 0 ? tiling_data.get_base_m2() : (tiling_data.get_maxTokens() % tiling_data.get_base_m2()));
    tiling_data.set_base_m1_tail_tile_ub_m_loop_num(((tiling_data.get_base_m1_tail_size() + tiling_data.get_ub_m()) - 1) / tiling_data.get_ub_m());
    tiling_data.set_base_m1_tail_tile_ub_m_tail_size((tiling_data.get_base_m1_tail_size() % tiling_data.get_ub_m()) == 0 ? tiling_data.get_ub_m() : (tiling_data.get_base_m1_tail_size() % tiling_data.get_ub_m()));
  }

  void ComputeOptionParam(FFNTilingData &tiling_data) {

  }

  void ExtraTilingData(FFNTilingData &tiling_data) {
    OP_LOGD(OP_NAME, "Start executing extra tiling for tilingCaseId 0.");
		UpdateGeneralTilingData(tiling_data);

    ComputeOptionParam(tiling_data);
		UpdateAxesTilingData(tiling_data);

    OP_LOGD(OP_NAME, "Execute extra tiling for tilingCaseId 0 successfully.");
  }

  void GetWorkSpaceSize(FFNTilingData& tiling_data) {
    OP_LOGD(OP_NAME, "Start setting workspace for case 0.");
    tiling_data.set_workspaceSize(static_cast<uint32_t>(0));
    OP_LOGD(OP_NAME, "Setting workspace to %u for case 0.", tiling_data.get_workspaceSize());
  }

  void TilingSummary(FFNTilingData &tiling_data) {
    OP_LOGI(OP_NAME, "Set base_m1 to %u.", tiling_data.get_base_m1());
    OP_LOGI(OP_NAME, "Set base_m2 to %u.", tiling_data.get_base_m2());
    OP_LOGI(OP_NAME, "Set base_n1 to %u.", tiling_data.get_base_n1());
    OP_LOGI(OP_NAME, "Set base_n2 to %u.", tiling_data.get_base_n2());
    OP_LOGI(OP_NAME, "Set ub_m to %u.", tiling_data.get_ub_m());
    OP_LOGI(OP_NAME, "The value of l0c_size is %d.", Getl0c_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of ub_size is %d.", Getub_size(tiling_data));
    OP_LOGI(OP_NAME, "The value of btbuf_size is %d.", Getbtbuf_size(tiling_data));
  }

};

TilingCaseImplPtr GetTilingImplPtr(uint32_t tilingCaseId, uint32_t corenum) {
  TilingCaseImplPtr tilingCaseImplPtr = nullptr;
  if (tilingCaseId == 0u) {
    tilingCaseImplPtr = std::make_shared<TilingCase0Impl>(corenum);
  }
  return tilingCaseImplPtr;
}
bool GetTilingKey(FFNTilingData &tiling_data, int32_t tilingCaseId = -1) {
  uint32_t corenum = tiling_data.get_block_dim();
  if (tilingCaseId == -1) {
    OP_LOGI(OP_NAME, "The user didn't specify tilingCaseId, iterate all templates.");
    uint32_t tilingKeys[1] = {0u};
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

bool GetTiling(FFNTilingData &tiling_data, int32_t tilingCaseId) {
  OP_LOGI(OP_NAME, "Start tiling. Calculating the tiling data.");
  if (!GetTilingKey(tiling_data, tilingCaseId)) {
    OP_LOGE(OP_NAME, "GetTiling Failed.");
    return false;
  }
  OP_LOGI(OP_NAME, "Filing the calculated tiling data in the context. End tiling.");
  return true;
}

} // namespace optiling

