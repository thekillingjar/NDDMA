/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "symbolic_info_post_processor.h"

#include "graph/attribute_group/attr_group_symbolic_desc.h"
#include "graph/attribute_group/attr_group_shape_env.h"
#include "graph/utils/graph_utils.h"
#include "graph/optimize/symbolic/codegen/guard_codegen.h"

namespace ge {
namespace {
const std::string kSymbolInferShapeCacheKey = "_symbol_infer_shape_merge_key";
const std::string kAllSymbolNum = "_all_symbol_num";
constexpr char const *kCastInsertBeforeAutoFuse = "_is_insert_before_autofuse";

/**
 * 仅针对autofuse节点，key相同的autofuse在lowring时的infershape阶段可以merge，使用同一个infershape节点
 * dim之间使用下划线连接，shape使用中括号括起来，例如: [1_2_s1][s2_s3][(s1+s2)_(s3*s4)]
 * @param graph 计算图
 * @return 是否成功
 */
Status MarkInferShapeMergeKey(const ComputeGraphPtr &graph) {
  for (const auto &node : graph->GetAllNodes()) {
    const auto op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    std::string infer_key;
    for (size_t i = 0U; i < op_desc->GetOutputsSize(); i++){
      auto symbol_attr = op_desc->GetOutputDesc(i).GetAttrsGroup<SymbolicDescAttr>();
      if (symbol_attr == nullptr) {
        continue;
      }
      auto output_shape = symbol_attr->symbolic_tensor.GetOriginSymbolShape();
      infer_key += "[";
      for (size_t j = 0U; j < output_shape.GetDimNum(); ++j) {
        infer_key += output_shape.GetDim(j).Str().get();
        if (j < output_shape.GetDimNum() - 1) {
          infer_key += "_";
        }
      }
      infer_key += "]";
    }
    if (!infer_key.empty()) {
      GE_ASSERT_TRUE(AttrUtils::SetStr(op_desc, kSymbolInferShapeCacheKey, infer_key));
      GELOGD("[%s][%s] infer shape merge key: %s.", op_desc->GetNamePtr(), op_desc->GetTypePtr(), infer_key.c_str());
    }
  }
  return GRAPH_SUCCESS;
}

/**
 * 获取图上所有符号的数量，用来生成SymbolTilingCacheKey，host侧生成一个AllSymbolNum大小vector，调用codegen
 * 的GetSymbolTilingCacheKey接口，改接口将使用的符号写到vector里面，CacheableSymbolTilingKernel使用该
 * vector拼接TIling缓存的key。
 *
 * @param graph 融合后的图
 * @return  是否成功
 */
Status MarkAllSymNum(const ComputeGraphPtr &graph) {
  auto root_graph = ge::GraphUtils::FindRootGraph(graph);
  GE_ASSERT_NOTNULL(root_graph);
  auto shape_env_attr = root_graph->GetAttrsGroup<ShapeEnvAttr>();
  GE_ASSERT_NOTNULL(shape_env_attr);
  auto all_sym_num = shape_env_attr->GetAllSym2Src().size();
  GE_ASSERT_TRUE(ge::AttrUtils::SetInt(graph, kAllSymbolNum, all_sym_num));
  GELOGD("[%s] all symbol num is: %zu.", graph->GetName().c_str(), all_sym_num);
  return GRAPH_SUCCESS;
}
} // namespace
Status SymbolicInfoPostProcessor::Run(const ComputeGraphPtr &graph) {
  GE_ASSERT_SUCCESS(MarkAllSymNum(graph));
  GE_ASSERT_SUCCESS(MarkInferShapeMergeKey(graph));

  GuardCodegen guard_codegen{};
  GE_ASSERT_SUCCESS(guard_codegen.GuardFuncCodegenAndCompile(graph));
  return SUCCESS;
}
} // namespace ge