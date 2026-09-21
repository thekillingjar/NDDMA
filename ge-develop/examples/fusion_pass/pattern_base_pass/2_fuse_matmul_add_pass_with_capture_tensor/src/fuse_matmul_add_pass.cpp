/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
  //在定义 pattern 时，按序capture MatMul 和 Add 的输入 tensor
  const int64_t kMatMulCaptureIdx = 0l;
  const int64_t kAddCaptureIdx = 1l;
  std::vector<PatternUniqPtr> Patterns() override {
    std::cout << "Define pattern for FuseMatMulAndAddPass in capture tensor sample" << std::endl;
    std::vector<PatternUniqPtr> patterns;
    // 此例中在线与离线场景下的原始图不同，需要分别定义模板
    auto graph_builder0 = es::EsGraphBuilder("pattern0");
    auto [a0, b0, c0] = graph_builder0.CreateInputs<3>();
    auto matmul0 = es::MatMul(a0, b0);
    auto add0 = es::Add(matmul0, c0);
    auto graph0 = graph_builder0.BuildAndReset({add0});
    auto pattern0 = std::make_unique<Pattern>(std::move(*graph0));
    pattern0->CaptureTensor({*matmul0.GetProducer(),0})
            .CaptureTensor({*add0.GetProducer(),0});
    patterns.emplace_back(std::move(pattern0));

    auto graph_builder1 = es::EsGraphBuilder("pattern1");
    auto [a1, b1, c1] = graph_builder1.CreateInputs<3>();
    auto matmul1 = es::BatchMatMulV2(a1, b1);
    auto add1 = es::Add(matmul1, c1);
    auto graph1 = graph_builder1.BuildAndReset({add1});
    auto pattern1 = std::make_unique<Pattern>(std::move(*graph1));
    pattern1->CaptureTensor({*matmul1.GetProducer(),0})
            .CaptureTensor({*add1.GetProducer(),0});
    patterns.emplace_back(std::move(pattern1));

    return patterns;
  }
  bool MeetRequirements(const std::unique_ptr<MatchResult> &match_result) override {
    std::cout << "Define MeetRequirements for FuseMatMulAndAddPass in capture tensor sample" << std::endl;
    NodeIo add_node;
    match_result->GetCapturedTensor(kAddCaptureIdx,add_node);

    TensorDesc add_input0_desc;
    add_node.node.GetInputDesc(0, add_input0_desc);
    TensorDesc add_input1_desc;
    add_node.node.GetInputDesc(1, add_input1_desc);
    //check dtype
    if (add_input0_desc.GetDataType() != DT_FLOAT || add_input1_desc.GetDataType() != DT_FLOAT) {
      std::cout << "Only support Add inputs are fp32"<<std::endl;
      return false;
    }
    return true;
  }
  GraphUniqPtr Replacement(const std::unique_ptr<MatchResult> &match_result) override {
    std::cout << "Define replacement for FuseMatMulAndAddPass in capture tensor sample" << std::endl;
    auto replace_graph_builder = es::EsGraphBuilder("replacement");
    auto [r_a, r_b, r_c] = replace_graph_builder.CreateInputs<3>();
    NodeIo matmul_node;
    match_result->GetCapturedTensor(kMatMulCaptureIdx, matmul_node);
    bool transpose_x1;
    matmul_node.node.GetAttr("transpose_x1", transpose_x1);
    bool transpose_x2;
    matmul_node.node.GetAttr("transpose_x2", transpose_x2);

    auto alpha_const = replace_graph_builder.CreateScalar(1);
    auto beta_const = replace_graph_builder.CreateScalar(1);
    auto gemm = es::GEMM(r_a, r_b, r_c, alpha_const, beta_const, transpose_x1, transpose_x2);
    return replace_graph_builder.BuildAndReset({gemm});
  }
};
REG_FUSION_PASS(FuseMatMulAndAddPass).Stage(CustomPassStage::kBeforeInferShape);