/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ascendc_ir.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace ge {
namespace ascir {

constexpr int32_t kFloatAlignSize = 8;
constexpr int32_t kPerformanceOptimization = 256;
constexpr int32_t kBlockSize = 32;

static ge::AscGraphAttr *GetOrCreateGraphAttrsGroup(const ge::ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL_EXEC(graph, return nullptr;);
  auto attr = graph->GetOrCreateAttrsGroup<ge::AscGraphAttr>();
  GE_CHECK_NOTNULL_EXEC(attr, return nullptr;);
  return attr;
}

static Expression GetAlignSize(Expression in) {
  return sym::Align(in, kFloatAlignSize);
}

static Expression GetByteSize(Expression in) {
  return sym::Mul(in, ge::Symbol(sizeof(float)));
}

static bool IsNeedAccumulation(const ge::AscNode &node) {
  if (node.GetType() == "Sum" || node.GetType() == "Mean" || node.GetType() == "Prod") {
    return true;
  }
  return false;
}

std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcReduceTmpSizeV2(const ge::AscNode &node) {
  std::vector<std::unique_ptr<ge::TmpBufDesc>> tmp_buf_desc;
  AscNodeInputs node_inputs = node.inputs;
  AscNodeOutputs node_outputs = node.outputs;
  if (node_inputs.Size() <= 0) {
    return tmp_buf_desc;
  }

  if (node_outputs[0].attr.vectorized_strides.size() <= 0) {
    return tmp_buf_desc;
  }

  bool is_reuse_source = false;
  auto node_in_anchor = node.GetInDataAnchor(0);
  auto peer_out_anchor = node_in_anchor->GetPeerOutAnchor();
  const auto &in_node = std::dynamic_pointer_cast<ge::AscNode>(peer_out_anchor->GetOwnerNode());
  if (in_node->GetOutAllNodes().size() == 1UL) {
    is_reuse_source = true;
  }

  bool isAr = SymbolicUtils::StaticCheckEq(
                  node_outputs[0].attr.vectorized_strides.back(), sym::kSymbolZero) == TriBool::kTrue;
  auto attr = GetOrCreateGraphAttrsGroup(node.GetOwnerComputeGraph());

  ge::Expression r_in_ub_exp = ge::Symbol(1);
  ge::Expression a_in_ub_exp = ge::Symbol(1);
  for (size_t i = 0; i < node_outputs[0].attr.vectorized_strides.size(); i++) {
    uint64_t vectorized_axis_id = node_outputs[0].attr.vectorized_axis[i];
    ge::Expression tmp_exp = attr->axis[vectorized_axis_id]->size;
    if (i == node_outputs[0].attr.vectorized_strides.size() - 1) {
      tmp_exp = GetAlignSize(tmp_exp);
    }

    if (SymbolicUtils::StaticCheckEq(node_outputs[0].attr.vectorized_strides[i], sym::kSymbolZero) == TriBool::kTrue &&
        SymbolicUtils::StaticCheckEq(node_inputs[0].attr.vectorized_strides[i], sym::kSymbolZero) != TriBool::kTrue) {
      r_in_ub_exp = sym::Mul(r_in_ub_exp, tmp_exp);
    } else {
      a_in_ub_exp = sym::Mul(a_in_ub_exp, tmp_exp);
    }
  }

  ge::Expression rFusedExpression = attr->axis[node.attr.sched.loop_axis]->size;
  if (IsNeedAccumulation(node)) {
    // 高阶API使用  a.UB * r.UB， ar场景需要加一个block，生命周期为-1
    ge::Expression api_size = GetByteSize(sym::Mul(a_in_ub_exp, r_in_ub_exp));
    if (node.GetType() == "Prod") {
      api_size = sym::Add(api_size, ge::Symbol(kPerformanceOptimization));
    }
    if (isAr) {
      api_size = sym::Add(api_size, ge::Symbol(kBlockSize));
    }
    if (!is_reuse_source) {
      ge::TmpBufDesc desc2 = {api_size, -1};
      tmp_buf_desc.emplace_back(std::make_unique<ge::TmpBufDesc>(desc2));
    }

    // UB 间
    if (isAr) {
      a_in_ub_exp = GetAlignSize(a_in_ub_exp);
    }
    ge::Expression a_size = GetByteSize(a_in_ub_exp);
    ge::TmpBufDesc desc3 = {a_size, 0};
    tmp_buf_desc.emplace_back(std::make_unique<ge::TmpBufDesc>(desc3));
  } else {
    // 高阶api部分 先按照最大的申请
    if (!is_reuse_source) {
      ge::Expression api_size = GetByteSize(sym::Mul(a_in_ub_exp, r_in_ub_exp));
      ge::TmpBufDesc desc1 = {api_size, -1};
      tmp_buf_desc.emplace_back(std::make_unique<ge::TmpBufDesc>(desc1));
    }

    // UB 间
    if (isAr) {
      a_in_ub_exp = GetAlignSize(a_in_ub_exp);
    }
    ge::Expression ub_size = GetByteSize(a_in_ub_exp);
    ge::TmpBufDesc desc2 = {ub_size, 0};
    tmp_buf_desc.emplace_back(std::make_unique<ge::TmpBufDesc>(desc2));
  }

  return tmp_buf_desc;
}
}  // namespace ascir
}  // namespace ge