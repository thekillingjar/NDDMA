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
// |o>   x   y                x   y  Const
// |o>    \ /                  \  |  /
// |o>   MatMul  Const  ==>     GEMM
// |o>       \    /
// |o>        Add
// |o>-----------------------------------
// 融合说明：本例识别上图中左边的split的属性num_split是否等于1,来判断是否可删除该节点
class MatmulAddFusionPass : public PatternFusionPass {
  //重写构造函数，使能const值匹配能力与ir属性及其值匹配能力
public:
	explicit MatmulAddFusionPass()
	: PatternFusionPass(PatternMatcherConfigBuilder().EnableConstValueMatch().EnableIrAttrMatch().Build()) {}
 protected:
	std::vector<PatternUniqPtr> Patterns() override {
	  std::cout << "Define pattern for MatmulAddFusionPass in matcher config sample" << std::endl;
		std::vector<PatternUniqPtr> patterns;
		auto graph_builder = es::EsGraphBuilder("pattern");
	  auto [x, y] = graph_builder.CreateInputs<2>();
	  auto z = graph_builder.CreateConst(
	    std::vector<float>{0.1f, 0.1f, 0.1f, 0.1f},
	    std::vector<int64_t>{2,2}
	  );
	  auto matmul = es::MatMul(x, y, nullptr, false, false);
	  auto add = es::Add(matmul, z);
	  auto graph = graph_builder.BuildAndReset({add});
	  auto pattern = std::make_unique<Pattern>(std::move(*graph));
		patterns.emplace_back(std::move(pattern));
	  return patterns;
	}
  GraphUniqPtr Replacement(const std::unique_ptr<MatchResult> &match_result) override {
	  std::cout << "Define replacement for MatmulAddFusionPass in matcher config sample" << std::endl;
	  auto replace_graph_builder = es::EsGraphBuilder("replacement");
	  auto [r_a, r_b] = replace_graph_builder.CreateInputs<2>();
	  auto r_c = replace_graph_builder.CreateConst(
      std::vector<float>{0.1f, 0.1f, 0.1f, 0.1f},
      std::vector<int64_t>{2,2}
    );
	  auto alpha_const = replace_graph_builder.CreateScalar(1);
	  auto beta_const = replace_graph_builder.CreateScalar(1);
	  auto gemm = es::GEMM(r_a, r_b, r_c, alpha_const, beta_const);
	  return replace_graph_builder.BuildAndReset({gemm});
	}
};
REG_FUSION_PASS(MatmulAddFusionPass).Stage(CustomPassStage::kBeforeInferShape);