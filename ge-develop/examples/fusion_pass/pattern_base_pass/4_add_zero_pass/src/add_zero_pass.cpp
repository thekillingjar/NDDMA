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
#include <math.h>
#include "es_all_ops.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"

using namespace ge;
using namespace fusion;

/*
|o>-----------------------------------
|o>    a  0
|o>    \ /              a      b
|o>    Add     b   ==>   \    /
|o>      \    /            Add
|o>       Add
|o>-----------------------------------
说明：本例识别上图中左边的Add+0结构并通过图修改接口删除该结构
*/

class AddZeroPass : public PatternFusionPass {
 protected:
  /*
  |o>--------------------
  |o>  Data    Const
  |o>     \     /
  |o>       Add
  |o>--------------------
  上图为Patterns定义的pattern结构
  通过此处定义的pattern在graph中匹配拓扑
  */
  std::vector<PatternUniqPtr> Patterns() override {
    std::cout << "Define pattern for AddZeroPass" << std::endl;
    std::vector<PatternUniqPtr> patterns;
    auto graph_builder = es::EsGraphBuilder("pattern");
    auto a = graph_builder.CreateInput(0);
    auto b = es::Const(graph_builder);
    auto add = es::Add(a, b);
    auto graph = graph_builder.BuildAndReset({add});
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    patterns.emplace_back(std::move(pattern));
    return patterns;
  }

  // 判断符合pattern结构的拓扑中，Const是否为0
  bool MeetRequirements(const std::unique_ptr<MatchResult> &match_result) override {
    std::cout << "Define MeetRequirements for AddZeroPass" << std::endl;
    std::vector<GNode> matched_nodes;
    matched_nodes = match_result->GetMatchedNodes();
    GNode matched_node;
    for (GNode node : matched_nodes) {
      AscendString type;
      node.GetType(type);
      if (type == "Const") {
        Tensor const_tensor;
        node.GetAttr("value", const_tensor);
        if (!IsTensorValueEqualToZero(const_tensor)) {
          return false;
        }
      }
    }
    return true;
  }

  /*
  |o>--------------------
  |o>      Data
  |o>       |
  |o>--------------------
  上图为Replacement定义的replacement结构
  通过此处定义的replacement替换图中MeetRequirements为True的pattern
  */
  GraphUniqPtr Replacement(const std::unique_ptr<MatchResult> &match_result) override {
    std::cout << "Define replacement for AddZeroPass" << std::endl;
    auto replacement_graph_builder = es::EsGraphBuilder("replacement");
    auto r_a = replacement_graph_builder.CreateInput(0);

    return replacement_graph_builder.BuildAndReset({r_a});
  }

 private:
  bool IsTensorValueEqualToZero(const Tensor &tensor) {
    auto tensor_dtype = tensor.GetTensorDesc().GetDataType();
    switch (tensor_dtype) {
      case DT_FLOAT:
        if (std::fabs(*reinterpret_cast<const float *>(tensor.GetData())) < 1e-6) {
          return true;
        }
        return false;
      case DT_DOUBLE:
        if (std::fabs(*reinterpret_cast<const double *>(tensor.GetData())) < 1e-15) {
          return true;
        }
        return false;
      case DT_INT32:
        if (*reinterpret_cast<const int *>(tensor.GetData()) == 0) {
          return true;
        }
        return false;
      // 此处可以增加case支持更多数据类型
      default:
        std::cout << "Unsupported data type" << std::endl;
        return false;
    }
  }
};

REG_FUSION_PASS(AddZeroPass).Stage(CustomPassStage::kBeforeInferShape);