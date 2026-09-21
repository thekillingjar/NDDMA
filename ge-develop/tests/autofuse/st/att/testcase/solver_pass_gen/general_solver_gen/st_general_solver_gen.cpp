/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "base/base_types.h"
#include "generator/solver_pass_gen/general_solver/general_solver_gen.h"
#include "test_common_utils.h"

using namespace att;

class ST_GENERAL_SOLVER_GEN : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    Expr x0 = CreateExpr("x0");
    Expr x1 = CreateExpr("x1");
    Expr x2 = CreateExpr("x2");
    Expr x3 = CreateExpr("x3");
    Expr a = CreateExpr("a");

    solver_ = new GeneralSolverGen("Case0", "TilingData");
    solver_->SetSearchArgs({x0, x1, x3});

    ExprExprMap expr_relation;
    ExprExprMap vars_relation;
    vars_relation[x0] = x0;
    vars_relation[x1] = x1;
    solver_->SetExprRelation(expr_relation, vars_relation);

    ExprUintMap const_args;
    const_args[a] = 0;
    solver_->SetConstArgs(const_args);
    solver_->SetSolvedArgs({x2});

    std::map<PipeType, Expr> obj;
    obj[PipeType::AIC_MTE1] = x0 + x1;
    solver_->SetObj(obj);

    std::map<HardwareDef, Expr> buffer_cons;
    buffer_cons[HardwareDef::GM] = x0 + x1;

    solver_->SetBufferCons(buffer_cons);
    solver_->SetCutCons({((x0 + x1) - a)});

    ExprExprMap max_value;
    ExprExprMap min_value;
    max_value[x0] = (CreateExpr(2) * a);
    max_value[x1] = (CreateExpr(2) * a);
    min_value[x0] = ge::sym::kSymbolOne;
    min_value[x1] = ge::sym::kSymbolOne;
    solver_->SetMaxValue(max_value);
    solver_->SetMinValue(min_value);

    solver_->SetInnestDim({x0});
    solver_->FixVar(2, 1);
    solver_->FixRange(0, 1, 5);
  }

  void TearDown() override {
    delete solver_;
  }
  GeneralSolverGen *solver_;
};

TEST_F(ST_GENERAL_SOLVER_GEN, test_gen_solver_impl) {
#include <iostream>
#include "test_common_utils.h"
  std::string codes = solver_->GenSolverClassImpl();
  std::string expect_codes = "";

  expect_codes += "/*\n";
  expect_codes += "用户可以在派生类中重载Run函数,构造自定义的求解算法,即\n";
  expect_codes += "  void bool Run(int32_t &solution_num, uint64_t *solutions) override;\n";
  expect_codes += "其中:\n";
  expect_codes += "  solution_num:int32_t类型的参数,用来输出实际得到的解的个数\n";
  expect_codes += "  solutions:uint64_t类型的数组,指向一块num_var * top_num的内存,算法将可行解放入该空间\n";
  expect_codes += "Run函数可以使用下述函数辅助求解:\n";
  expect_codes += "  bool CheckValid()\n";
  expect_codes += "    用于检测当前解是否为可行解\n";
  expect_codes += "  bool UpdateCurVarVal(uint64_t value, int32_t idx)\n";
  expect_codes += "    将下标为idx的待求解变量改为value,同时更新cons_info_->leqs中的值\n";
  expect_codes += "  bool RecordBestVarVal()\n";
  expect_codes += "    待求解变量的当前值所对应的目标函数寻优\n";
  expect_codes += "Run函数可以使用下述参数辅助求解:\n";
  expect_codes += "  cons_info_->leqs, double类型的数组, 用于记录不等式约束的函数值, 其下标含义如下:\n";
  expect_codes += "    cons_info_->leqs[0] = (x0 + x1 - hbm_size)\n";
  expect_codes += "    cons_info_->leqs[1] = (x0 + x1 - a)\n";
  expect_codes += "  var_info_->cur_vars, uint64_t类型的数组, 用于记录待求解变量的当前值, 其下标含义如下:\n";
  expect_codes += "    var_info_->cur_vars[0] = x3\n";
  expect_codes += "  var_info_->upper_bound, uint64_t类型的数组, 用于记录待求解变量的上界\n";
  expect_codes += "  var_info_->lower_bound, uint64_t类型的数组, 用于记录待求解变量的下界\n";
  expect_codes += "*/\n";
  expect_codes += "class GeneralSolverCase0 : public GeneralSolver<GeneralSolverCase0>\n";
  expect_codes += "{\n";
  expect_codes += "    public:\n";
  expect_codes +=
      "        explicit GeneralSolverCase0(SolverConfig& config, TilingData& tiling_data) {\n";
  expect_codes += "            solver_config_ = config;\n";
  expect_codes += "            hbm_size = tiling_data.get_hbm_size();\n";
  expect_codes += "            x2 = tiling_data.get_x2();\n";
  expect_codes += "        }\n\n";
  expect_codes += "        double GetObj(uint64_t* vars);\n";
  expect_codes += "        double GetSmoothObj(uint64_t* vars);\n";
  expect_codes += "        double GetBuffCost(uint64_t* vars);\n";
  expect_codes += "        bool CheckLocalValid(double* leqs, int32_t idx);\n";
  expect_codes += "        void DisplayVarVal(uint64_t* vars);\n";
  expect_codes += "        void UpdateLeqs(uint64_t* vars, int32_t idx, double* leqs);\n";
  expect_codes += "        double GetBuffDiff(uint64_t* vars, double* weight);\n";
  expect_codes += "        double GetLeqDiff(uint64_t* vars, double* weight);\n";
  expect_codes += "        double Gethbm_sizeCost(uint64_t* vars);\n";
  expect_codes += "        double GetSmoothhbm_sizeCost(uint64_t* vars);\n";
  expect_codes += "        void MapVarVal(uint64_t* vars, TilingData& tiling_data);\n";
  expect_codes += "        void GetResult(int32_t solution_num, uint64_t* solution, TilingData& tiling_data);\n";
  expect_codes += "    private:\n";
  expect_codes += "        const int64_t x0_idx = 0;\n";
  expect_codes += "        const int64_t x1_idx = 1;\n";
  expect_codes += "        uint64_t x3{1};\n";
  expect_codes += "        uint64_t a{0};\n";
  expect_codes += "        uint64_t hbm_size;\n";
  expect_codes += "        uint64_t x2;\n";
  expect_codes += "};\n";

  expect_codes += "/*\n";
  expect_codes += "函数名:Gethbm_sizeCost(重要函数)\n";
  expect_codes += "功能描述:\n";
  expect_codes += "  根据待求解变量值hbm_size缓存占用信息(occupy-buff)\n";
  expect_codes += "输入参数:\n";
  expect_codes += "  vars:一个长度为num_var的数组,对应了待求解变量\n";
  expect_codes += "*/\n";
  expect_codes += "inline double GeneralSolverCase0::Gethbm_sizeCost(uint64_t* vars)\n";
  expect_codes += "{\n";
  expect_codes += "    double x0 = static_cast<double>(vars[x0_idx]);\n";
  expect_codes += "    double x1 = static_cast<double>(vars[x1_idx]);\n";
  expect_codes += "    return (x0 + x1 - hbm_size);\n";
  expect_codes += "}\n";
  expect_codes += "\n";

  expect_codes += "/*\n";
  expect_codes += "函数名:GetSmoothhbm_sizeCost(重要函数)\n";
  expect_codes += "功能描述:\n";
  expect_codes += "  根据待求解变量值hbm_size的平滑化缓存占用信息\n";
  expect_codes += "  与Gethbm_sizeCost函数相比,整除运算被替换为浮点数的除法运算\n";
  expect_codes += "输入参数:\n";
  expect_codes += "  vars:一个长度为num_var的数组,对应了待求解变量\n";
  expect_codes += "*/\n";
  expect_codes += "inline double GeneralSolverCase0::GetSmoothhbm_sizeCost(uint64_t* vars)\n";
  expect_codes += "{\n";
  expect_codes += "    double x0 = static_cast<double>(vars[x0_idx]);\n";
  expect_codes += "    double x1 = static_cast<double>(vars[x1_idx]);\n";
  expect_codes += "    return (x0 + x1 - hbm_size);\n";
  expect_codes += "}\n";
  expect_codes += "\n";

  expect_codes += "/*\n";
  expect_codes += "函数名:GetObj(重要函数)\n";
  expect_codes += "功能描述:\n";
  expect_codes += "  根据待求解变量值输出目标函数\n";
  expect_codes += "输入参数:\n";
  expect_codes += "  vars:一个长度为num_var的数组,对应了待求解变量\n";
  expect_codes += "*/\n";
  expect_codes += "inline double GeneralSolverCase0::GetObj(uint64_t* vars)\n";
  expect_codes += "{\n";
  expect_codes += "    double x0 = static_cast<double>(vars[x0_idx]);\n";
  expect_codes += "    double x1 = static_cast<double>(vars[x1_idx]);\n";
  expect_codes += "    double AIC_MTE1 = (x0 + x1);\n";
  expect_codes += "    OP_LOGD(OP_NAME, \"AIC_MTE1 = %f\", AIC_MTE1);\n";
  expect_codes += "    return AIC_MTE1;\n";
  expect_codes += "}\n";

  expect_codes += "/*\n";
  expect_codes += "函数名:GetSmoothObj(重要函数)\n";
  expect_codes += "功能描述:\n";
  expect_codes += "  根据待求解变量值输出平滑化目标函数\n";
  expect_codes += "  与GetObj函数相比,整除运算被替换为浮点数的除法运算\n";
  expect_codes += "*/\n";
  expect_codes += "inline double GeneralSolverCase0::GetSmoothObj(uint64_t* vars)\n";
  expect_codes += "{\n";
  expect_codes += "    double x0 = static_cast<double>(vars[x0_idx]);\n";
  expect_codes += "    double x1 = static_cast<double>(vars[x1_idx]);\n";
  expect_codes += "    double AIC_MTE1 = (x0 + x1);\n";
  expect_codes += "    return AIC_MTE1;\n";
  expect_codes += "}\n";

  expect_codes += "/*\n";
  expect_codes += "函数名:GetBuffCost(重要函数)\n";
  expect_codes += "功能描述:\n";
  expect_codes += "  根据待求解变量值输出缓存占用信息的罚函数(sigma(min(0, occupy-buff)^2))\n";
  expect_codes += "  该函数用于量化解在缓存占用方面的质量\n";
  expect_codes += "输入参数:\n";
  expect_codes += "  vars:一个长度为num_var的数组,对应了待求解变量\n";
  expect_codes += "*/\n";
  expect_codes += "inline double GeneralSolverCase0::GetBuffCost(uint64_t* vars)\n";
  expect_codes += "{\n";
  expect_codes += "    double hbm_size_cost = Gethbm_sizeCost(vars);\n";
  expect_codes += "    return (Min(0, hbm_size_cost) * Min(0, hbm_size_cost));\n";
  expect_codes += "}\n";

  expect_codes += "/*\n";
  expect_codes += "函数名:GetBuffDiff(重要函数)\n";
  expect_codes += "功能描述:\n";
  expect_codes += "  获取缓冲占用加权差分值,计算平滑缓冲占用的差分\n";
  expect_codes += "  输出的计算公式为sigma_j(delta_{var_i}(g_j(var))) * g_j(var))\n";
  expect_codes += "  其中g_j为第j个缓冲占用不等式,delta_{var_i}(g_j(var))为g_j(var)沿var_i方向更新一个单位后的变化值\n";
  expect_codes += "  该函数用于确定变量沿缓冲占用增大的更新方向\n";
  expect_codes += "输入参数:\n";
  expect_codes += "  vars:一个长度为num_var的数组,对应了待求解变量\n";
  expect_codes += "  weight:一个长度为num_leq的数组,代表了每个缓冲占用的权值\n";
  expect_codes += "*/\n";
  expect_codes += "inline double GeneralSolverCase0::GetBuffDiff(uint64_t* vars, double* weight)\n";
  expect_codes += "{\n";
  expect_codes += "    double hbm_size_cost = GetSmoothhbm_sizeCost(vars);\n";
  expect_codes += "    hbm_size_cost *= weight[0] < 0 ? weight[0] : 0;\n";
  expect_codes += "    return hbm_size_cost;\n";
  expect_codes += "}\n";

  expect_codes += "/*\n";
  expect_codes += "函数名:GetLeqDiff(重要函数)\n";
  expect_codes += "功能描述:\n";
  expect_codes += "  获取不等式约束的加权差分值,计算平滑的不等式函数的差分,权值为实际不等式函数值\n";
  expect_codes += "  输出的计算公式为sigma_j(delta_{var_i}(f_j(var))) * f_j(var))\n";
  expect_codes += "  其中f_j为第j个不等式约束式,delta_{var_i}(f_j(var))为f_j(var)沿var_i方向更新一个单位后的变化值\n";
  expect_codes += "  该函数用于确定变量从可行域外侧沿不等式边界方向移动的更新方向\n";
  expect_codes += "输入参数:\n";
  expect_codes += "  vars:一个长度为num_var的数组,对应了待求解变量\n";
  expect_codes += "  weight:一个长度为num_leq的数组,代表了每个缓冲占用的权值\n";
  expect_codes += "*/\n";
  expect_codes += "inline double GeneralSolverCase0::GetLeqDiff(uint64_t* vars, double* weight)\n";
  expect_codes += "{\n";
  expect_codes += "    double x0 = static_cast<double>(vars[x0_idx]);\n";
  expect_codes += "    double x1 = static_cast<double>(vars[x1_idx]);\n";
  expect_codes += "    double hbm_size_cost = GetSmoothhbm_sizeCost(vars);\n";
  expect_codes += "    hbm_size_cost *= weight[0] > 0 ? weight[0] : 0;\n";
  expect_codes += "    double leq1_cost = (x0 + x1 - a);\n";
  expect_codes += "    leq1_cost *= weight[1] > 0 ? weight[1] : 0;\n";
  expect_codes += "    return hbm_size_cost + leq1_cost;\n";
  expect_codes += "}\n";

  expect_codes += "inline bool GeneralSolverCase0::CheckLocalValid(double* leqs, int32_t idx)\n";
  expect_codes += "{\n";
  expect_codes += "    if (idx == x0_idx) {\n";
  expect_codes += "        return leqs[0] <= 0 && leqs[1] <= 0;\n";
  expect_codes += "    } else if (idx == x1_idx) {\n";
  expect_codes += "        return leqs[0] <= 0 && leqs[1] <= 0;\n";
  expect_codes += "    }\n";
  expect_codes += "    return true;\n";
  expect_codes += "}\n";
  expect_codes += "\n";

  expect_codes += "inline void GeneralSolverCase0::UpdateLeqs(uint64_t* vars, int32_t idx, double* leqs)\n";
  expect_codes += "{\n";
  expect_codes += "    double x0 = static_cast<double>(vars[x0_idx]);\n";
  expect_codes += "    double x1 = static_cast<double>(vars[x1_idx]);\n";
  expect_codes += "    if (idx == x0_idx) {\n";
  expect_codes += "        leqs[0] = (x0 + x1 - hbm_size);\n";
  expect_codes += "        leqs[1] = (x0 + x1 - a);\n";
  expect_codes += "    } else if (idx == x1_idx) {\n";
  expect_codes += "        leqs[0] = (x0 + x1 - hbm_size);\n";
  expect_codes += "        leqs[1] = (x0 + x1 - a);\n";
  expect_codes += "    } else if (idx == -1) {\n";
  expect_codes += "        leqs[0] = (x0 + x1 - hbm_size);\n";
  expect_codes += "        leqs[1] = (x0 + x1 - a);\n";
  expect_codes += "    }\n";
  expect_codes += "}\n";
  expect_codes += "\n";

  expect_codes += "inline void GeneralSolverCase0::DisplayVarVal(uint64_t* vars)\n";
  expect_codes += "{\n";
  expect_codes += "    uint64_t x0 = vars[x0_idx];\n";
  expect_codes += "    uint64_t x1 = vars[x1_idx];\n";
  expect_codes += "    OP_LOGD(OP_NAME, \"x3 = %lu\", static_cast<uint64_t>(1));\n";
  expect_codes += "}\n";
  expect_codes += "\n";

  expect_codes += "inline void GeneralSolverCase0::MapVarVal(uint64_t* vars, TilingData& tiling_data)\n";
  expect_codes += "{\n";
  expect_codes += "    uint64_t x0 = vars[x0_idx];\n";
  expect_codes += "    uint64_t x1 = vars[x1_idx];\n";
  expect_codes += "    OP_LOGD(OP_NAME, \"The output of the solver for tilingCaseId Case0 is:\");\n";
  expect_codes += "    tiling_data.set_x3(static_cast<uint64_t>(1));\n";
  expect_codes += "    OP_LOGD(OP_NAME, \"x3 = %u\", tiling_data.get_x3());\n";
  expect_codes += "}\n";
  expect_codes += "\n";

  expect_codes +=
      "inline void GeneralSolverCase0::GetResult(int32_t solution_num, uint64_t* solution, TilingData& "
      "tiling_data)\n{\n";
  expect_codes += "    if (solution_num > 0) {\n";
  expect_codes += "        OP_LOGD(OP_NAME, \"Filling tilingdata for Case0.\");\n";
  expect_codes += "        OP_LOGD(OP_NAME, \"Estimate the occupy.\");\n";
  expect_codes += "        OP_LOGD(OP_NAME, \"hbm_size = %ld\", static_cast<uint64_t>(Gethbm_sizeCost(solution) + hbm_size));\n";
  expect_codes += "        OP_LOGD(OP_NAME, \"Simulate the cost.\");\n";
  expect_codes += "        OP_LOGD(OP_NAME, \"Objective value for Case0 is %f.\", GetObj(solution));\n";
  expect_codes += "        MapVarVal(solution, tiling_data);\n";
  expect_codes += "    }\n";
  expect_codes += "}\n\n";

  expect_codes += "bool ExecuteCase0GeneralSolver(TilingData& tiling_data)\n";
  expect_codes += "{\n";

  expect_codes += "    SolverConfig cfg;\n";
  expect_codes += "    cfg.top_num = cfg_top_num;\n";
  expect_codes += "    cfg.search_length = cfg_search_length;\n";
  expect_codes += "    cfg.iterations = cfg_iterations;\n";
  expect_codes += "    cfg.simple_ver = cfg_simple_ver;\n";
  expect_codes +=
      "    cfg.momentum_factor = cfg_momentum_factor > 1 ? 1 : (cfg_momentum_factor < 0 ? 0 : cfg_momentum_factor);\n";
  expect_codes += "    OP_LOGD(OP_NAME, \"Record a maximum of %lu solutions.\", cfg.top_num);\n";
  expect_codes += "    OP_LOGD(OP_NAME, \"The searching range covers %lu unit(s).\", cfg.search_length);\n";
  expect_codes += "    OP_LOGD(OP_NAME, \"The maximum number of iterations is %lu.\", cfg.iterations);\n";
  expect_codes += "    if (cfg.simple_ver) {\n";
  expect_codes += "        OP_LOGD(OP_NAME, \"Using high-efficiency version.\");\n";
  expect_codes += "    } else {\n";
  expect_codes += "        OP_LOGD(OP_NAME, \"Using high-performance version.\");\n";
  expect_codes += "    }\n";
  expect_codes += "    OP_LOGD(OP_NAME, \"The momentum factor is %f.\", cfg.momentum_factor);\n";
  expect_codes += "\n";

  expect_codes += "    // 以下参数若未注明是可修改参数,则不建议修改\n";
  expect_codes += "    // 由modelinfo传入的待求解变量个数\n";
  expect_codes += "    int32_t num_var = 2;\n";
  expect_codes += "    // 由modelinfo传入的不等式约束个数\n";
  expect_codes += "    int32_t num_leq = 2;\n";
  expect_codes +=
      "    OP_LOGD(OP_NAME, \"The number of variable is %d(x0, x1), the number of constraints is %d.\", num_var, num_leq);\n";
  expect_codes += "    // (可修改参数) 待求解变量的初始值,算法趋向于求初始值附近的局部最优解\n";
  expect_codes += "    uint64_t init_vars[num_var] = {static_cast<uint64_t>(5), static_cast<uint64_t>((2 * a))};\n";
  expect_codes +=
      "    // (可修改参数) "
      "待求解变量的上界,过大的上界将导致搜索范围与耗时增加,过小的上界更有可能获得较差的局部最优解\n";
  expect_codes += "    uint64_t upper_bound[num_var] = {static_cast<uint64_t>(5), static_cast<uint64_t>((2 * a))};\n";
  expect_codes +=
      "    // (可修改参数) "
      "待求解变量的下界,过小的下界将导致搜索范围与耗时增加,过大的下界更有可能获得较差的局部最优解\n";
  expect_codes += "    uint64_t lower_bound[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(1)};\n";
  expect_codes += "    // (可修改参数) 最后更新的待求解变量,设置为true的对应变量会更接近初始值\n";
  expect_codes += "    bool update_last[num_var] = {true, false};\n";
  expect_codes += "    // 初始化解的个数为0\n";
  expect_codes += "    int32_t solution_num = 0;\n";
  expect_codes += "    // 为求解器的输出分配内存\n";
  expect_codes += "    uint64_t* solution = new(std::nothrow) uint64_t[num_var * cfg.top_num];\n";
  expect_codes += "    if (solution == nullptr)\n";
  expect_codes += "    {\n";
  expect_codes += "        OP_LOGW(OP_NAME, \"Create solution failed.\");\n";
  expect_codes += "        return false;\n";
  expect_codes += "    }\n";
  expect_codes += "    // 通用求解器的输入参数\n";
  expect_codes += "    SolverInput input;\n";
  expect_codes += "    input.var_num = num_var;\n";
  expect_codes += "    input.leq_num = num_leq;\n";
  expect_codes += "    input.cur_vars = init_vars;\n";
  expect_codes += "    input.upper_bound = upper_bound;\n";
  expect_codes += "    input.lower_bound = lower_bound;\n";
  expect_codes += "    input.update_last = update_last;\n";
  expect_codes +=
      "    OP_LOGD(OP_NAME, \"x0->init value: %lu, range: [%lu, %lu].\", init_vars[0], lower_bound[0], "
      "upper_bound[0]);\n";
  expect_codes +=
      "    OP_LOGD(OP_NAME, \"x1->init value: %lu, range: [%lu, %lu].\", init_vars[1], lower_bound[1], "
      "upper_bound[1]);\n";
  expect_codes += "\n";

  expect_codes += "    GeneralSolverCase0* solver = new(std::nothrow) GeneralSolverCase0(cfg, tiling_data);\n";
  expect_codes += "    if (solver != nullptr) {\n";
  expect_codes += "        // 导入通用求解器的输入参数并完成初始化\n";
  expect_codes += "        OP_LOGD(OP_NAME, \"Start initializing the input.\");\n";
  expect_codes += "        if (solver -> Init(input)) {\n";
  expect_codes += "            // 运行通用求解器并获取算法的解\n";
  expect_codes += "            OP_LOGD(OP_NAME, \"Intialization finished, start running the solver.\");\n";
  expect_codes += "            if (solver -> Run(solution_num, solution)) {\n";
  expect_codes += "                solver -> GetResult(solution_num, solution, tiling_data);\n";
  expect_codes += "                delete solver;\n";
  expect_codes += "                delete[] solution;\n";
  expect_codes += "                OP_LOGD(OP_NAME, \"The solver executed successfully.\");\n";
  expect_codes += "                return true;\n";
  expect_codes += "            }\n";
  expect_codes += "            OP_LOGW(OP_NAME, \"Failed to find any solution.\");\n";
  expect_codes += "        }\n";
  expect_codes += "    }\n";
  expect_codes += "    if (solver != nullptr) {\n";
  expect_codes += "        delete solver;\n";
  expect_codes += "    }\n";
  expect_codes += "    if (solution != nullptr) {\n";
  expect_codes += "        delete[] solution;\n";
  expect_codes += "    }\n";

  expect_codes += "    OP_LOGW(OP_NAME, \"The solver executed failed.\");\n";
  expect_codes += "    return false;\n";
  expect_codes += "}\n";
  expect_codes += "\n";

  EXPECT_NE(codes, "");
}

TEST_F(ST_GENERAL_SOLVER_GEN, test_gen_solver_invoke) {
  std::string codes = solver_->GenSolverFuncInvoke();
  std::string expect_codes = "";

  expect_codes += "bool ExecuteCase0GeneralSolver(TilingData& tiling_data)\n";
  expect_codes += "{\n";

  expect_codes += "    SolverConfig cfg;\n";
  expect_codes += "    cfg.top_num = cfg_top_num;\n";
  expect_codes += "    cfg.search_length = cfg_search_length;\n";
  expect_codes += "    cfg.iterations = cfg_iterations;\n";
  expect_codes += "    cfg.simple_ver = cfg_simple_ver;\n";
  expect_codes +=
      "    cfg.momentum_factor = cfg_momentum_factor > 1 ? 1 : (cfg_momentum_factor < 0 ? 0 : cfg_momentum_factor);\n";
  expect_codes += "    OP_LOGD(OP_NAME, \"Record a maximum of %lu solutions.\", cfg.top_num);\n";
  expect_codes += "    OP_LOGD(OP_NAME, \"The searching range covers %lu unit(s).\", cfg.search_length);\n";
  expect_codes += "    OP_LOGD(OP_NAME, \"The maximum number of iterations is %lu.\", cfg.iterations);\n";
  expect_codes += "    if (cfg.simple_ver) {\n";
  expect_codes += "        OP_LOGD(OP_NAME, \"Using high-efficiency version.\");\n";
  expect_codes += "    } else {\n";
  expect_codes += "        OP_LOGD(OP_NAME, \"Using high-performance version.\");\n";
  expect_codes += "    }\n";
  expect_codes += "    OP_LOGD(OP_NAME, \"The momentum factor is %f.\", cfg.momentum_factor);\n";
  expect_codes += "\n";

  expect_codes += "    // 以下参数若未注明是可修改参数,则不建议修改\n";
  expect_codes += "    // 由modelinfo传入的待求解变量个数\n";
  expect_codes += "    int32_t num_var = 2;\n";
  expect_codes += "    // 由modelinfo传入的不等式约束个数\n";
  expect_codes += "    int32_t num_leq = 2;\n";
  expect_codes +=
      "    OP_LOGD(OP_NAME, \"The number of variable is %d(x0, x1), the number of constraints is %d.\", num_var, num_leq);\n";
  expect_codes += "    // (可修改参数) 待求解变量的初始值,算法趋向于求初始值附近的局部最优解\n";
  expect_codes += "    uint64_t init_vars[num_var] = {static_cast<uint64_t>(5), static_cast<uint64_t>((2 * a))};\n";
  expect_codes +=
      "    // (可修改参数) "
      "待求解变量的上界,过大的上界将导致搜索范围与耗时增加,过小的上界更有可能获得较差的局部最优解\n";
  expect_codes += "    uint64_t upper_bound[num_var] = {static_cast<uint64_t>(5), static_cast<uint64_t>((2 * a))};\n";
  expect_codes +=
      "    // (可修改参数) "
      "待求解变量的下界,过小的下界将导致搜索范围与耗时增加,过大的下界更有可能获得较差的局部最优解\n";
  expect_codes += "    uint64_t lower_bound[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(1)};\n";
  expect_codes += "    // (可修改参数) 最后更新的待求解变量,设置为true的对应变量会更接近初始值\n";
  expect_codes += "    bool update_last[num_var] = {true, false};\n";
  expect_codes += "    // 初始化解的个数为0\n";
  expect_codes += "    int32_t solution_num = 0;\n";
  expect_codes += "    // 为求解器的输出分配内存\n";
  expect_codes += "    uint64_t* solution = new(std::nothrow) uint64_t[num_var * cfg.top_num];\n";
  expect_codes += "    if (solution == nullptr)\n";
  expect_codes += "    {\n";
  expect_codes += "        OP_LOGW(OP_NAME, \"Create solution failed.\");\n";
  expect_codes += "        return false;\n";
  expect_codes += "    }\n";
  expect_codes += "    // 通用求解器的输入参数\n";
  expect_codes += "    SolverInput input;\n";
  expect_codes += "    input.var_num = num_var;\n";
  expect_codes += "    input.leq_num = num_leq;\n";
  expect_codes += "    input.cur_vars = init_vars;\n";
  expect_codes += "    input.upper_bound = upper_bound;\n";
  expect_codes += "    input.lower_bound = lower_bound;\n";
  expect_codes += "    input.update_last = update_last;\n";
  expect_codes +=
      "    OP_LOGD(OP_NAME, \"x0->init value: %lu, range: [%lu, %lu].\", init_vars[0], lower_bound[0], "
      "upper_bound[0]);\n";
  expect_codes +=
      "    OP_LOGD(OP_NAME, \"x1->init value: %lu, range: [%lu, %lu].\", init_vars[1], lower_bound[1], "
      "upper_bound[1]);\n";
  expect_codes += "\n";

  expect_codes += "    GeneralSolverCase0* solver = new(std::nothrow) GeneralSolverCase0(cfg, tiling_data);\n";
  expect_codes += "    if (solver != nullptr) {\n";
  expect_codes += "        // 导入通用求解器的输入参数并完成初始化\n";
  expect_codes += "        OP_LOGD(OP_NAME, \"Start initializing the input.\");\n";
  expect_codes += "        if (solver -> Init(input)) {\n";
  expect_codes += "            // 运行通用求解器并获取算法的解\n";
  expect_codes += "            OP_LOGD(OP_NAME, \"Intialization finished, start running the solver.\");\n";
  expect_codes += "            if (solver -> Run(solution_num, solution)) {\n";
  expect_codes += "                solver -> GetResult(solution_num, solution, tiling_data);\n";
  expect_codes += "                delete solver;\n";
  expect_codes += "                delete[] solution;\n";
  expect_codes += "                OP_LOGD(OP_NAME, \"The solver executed successfully.\");\n";
  expect_codes += "                return true;\n";
  expect_codes += "            }\n";
  expect_codes += "            OP_LOGW(OP_NAME, \"Failed to find any solution.\");\n";
  expect_codes += "        }\n";
  expect_codes += "    }\n";
  expect_codes += "    if (solver != nullptr) {\n";
  expect_codes += "        delete solver;\n";
  expect_codes += "    }\n";
  expect_codes += "    if (solution != nullptr) {\n";
  expect_codes += "        delete[] solution;\n";
  expect_codes += "    }\n";

  expect_codes += "    OP_LOGW(OP_NAME, \"The solver executed failed.\");\n";
  expect_codes += "    return false;\n";
  expect_codes += "}\n";
  expect_codes += "\n";

  EXPECT_NE(codes, "");
}
