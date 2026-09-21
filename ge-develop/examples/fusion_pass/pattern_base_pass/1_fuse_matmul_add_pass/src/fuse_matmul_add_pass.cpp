/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "es_all_ops.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"

using namespace ge;
using namespace fusion;
// |o>-----------------------------------
// |o>      a  b
// |o>      \ /                a    b    c
// |o>     MatMul     c   ==>   \   |   /
// |o>        \      /            GEMM
// |o>           Add
// |o>-----------------------------------
// 融合说明：本例识别上图中左边的MatMul+Add结构并通过图修改接口替换为右边的单个GEMM节点
class FuseMatMulAndAddPass : public PatternFusionPass {
 protected:
  std::vector<PatternUniqPtr> Patterns() override {
    std::cout << "Define pattern for FuseMatMulAndAddPass" << std::endl;
    std::vector<PatternUniqPtr> patterns;
    // 此例中在线与离线场景下的原始图不同，需要分别定义模板
    auto graph_builder0 = es::EsGraphBuilder("pattern0");
    auto [a0, b0, c0] = graph_builder0.CreateInputs<3>();
    auto matmul0 = es::MatMul(a0, b0);
    auto add0 = es::Add(matmul0, c0);
    auto graph0 = graph_builder0.BuildAndReset({add0});
    auto pattern0 = std::make_unique<Pattern>(std::move(*graph0));
    patterns.emplace_back(std::move(pattern0));

    auto graph_builder1 = es::EsGraphBuilder("pattern1");
    auto [a1, b1, c1] = graph_builder1.CreateInputs<3>();
    auto matmul1 = es::BatchMatMulV2(a1, b1);
    auto add1 = es::Add(matmul1, c1);
    auto graph1 = graph_builder1.BuildAndReset({add1});
    auto pattern1 = std::make_unique<Pattern>(std::move(*graph1));
    patterns.emplace_back(std::move(pattern1));

    return patterns;
  }
  GraphUniqPtr Replacement(const std::unique_ptr<MatchResult> &match_result) override {
    std::cout << "Define replacement for FuseMatMulAndAddPass" << std::endl;
    auto replace_graph_builder = es::EsGraphBuilder("replacement");
    auto [r_a, r_b, r_c] = replace_graph_builder.CreateInputs<3>();
    auto alpha_const = replace_graph_builder.CreateScalar(1);
    auto beta_const = replace_graph_builder.CreateScalar(1);
    auto gemm = es::GEMM(r_a, r_b, r_c, alpha_const, beta_const);
    return replace_graph_builder.BuildAndReset({gemm});
  }
};
REG_FUSION_PASS(FuseMatMulAndAddPass).Stage(CustomPassStage::kBeforeInferShape);