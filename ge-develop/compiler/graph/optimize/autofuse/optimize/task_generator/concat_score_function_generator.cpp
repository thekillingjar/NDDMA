/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "concat_score_function_generator.h"

#include "ascgraph_info_complete.h"
#include "ascir_utils.h"

namespace optimize {
ConcatScoreFunctionGenerator::ConcatScoreFunctionGenerator(const ascir::HintGraph &graph,
                                                           ge::AscNodePtr concat_node,
                                                           uint32_t concat_dim)
    : graph_(&graph), concat_node_(std::move(concat_node)), concat_dim_(concat_dim) {}

Status ConcatScoreFunctionGenerator::Generate(std::string &score_func) {
  ss_ << "int32_t CalcScore(const AutofuseTilingData &tiling_data) {" << std::endl;
  if (::ascir::utils::UseSmallTailConcatApi(*concat_node_)) {
    GenerateReturnValue(1);
    score_func = ss_.str();
    return ASCCOMMON_SUC;
  }
  GE_CHK_STATUS_RET(ParseStride());
  if (const_part_stride_ % kAlignment_ == 0U) {
    GenerateReturnValue(1);
    score_func = ss_.str();
    return ASCCOMMON_SUC;
  }

  int32_t score = 0;
  GE_CHK_STATUS_RET(TryGetScoreByConstExpr(score));
  if (score != 0) {
    GenerateReturnValue(score);
    score_func = ss_.str();
    return ASCCOMMON_SUC;
  }
  GE_CHK_STATUS_RET(GenerateForUnaligned());
  score_func = ss_.str();
  return ASCCOMMON_SUC;
}

ge::Status ConcatScoreFunctionGenerator::ParseStride() {
  num_inputs_ = concat_node_->inputs.Size();
  GE_ASSERT_TRUE(num_inputs_ > 0U);
  const auto &input_attr = concat_node_->inputs[0].attr;
  auto dtype_size = ge::GetSizeByDataType(input_attr.dtype);
  const auto num_dims = static_cast<int32_t>(input_attr.repeats.size());
  stride_ = ge::Symbol(dtype_size);
  ge::Expression const_part_stride = stride_;
  elt_num_after_concat_dim_ = ge::Symbol(1);
  for (int32_t i = num_dims - 1; i > static_cast<int32_t>(concat_dim_); --i) {
    const auto dim_expr = input_attr.repeats[i];
    stride_ = stride_ * dim_expr;
    elt_num_after_concat_dim_ = elt_num_after_concat_dim_ * dim_expr;
    if (dim_expr.IsConstExpr()) {
      const_part_stride = const_part_stride * dim_expr;
    }
  }
  GE_ASSERT_TRUE(const_part_stride.GetConstValue(const_part_stride_),
                 "Failed to get int value, expr = %s",
                 const_part_stride.Str().get());
  return ge::SUCCESS;
}

void ConcatScoreFunctionGenerator::GenerateReturnValue(int32_t score) {
  ss_ << "  return " << score << ";" << std::endl;
  ss_ << "}" << std::endl;
}

Status ConcatScoreFunctionGenerator::TryGetScoreByConstExpr(int32_t &score) {
  auto &output_concat_dim = concat_node_->outputs[0].attr.repeats[concat_dim_];
  if (!output_concat_dim.IsConstExpr()) {
    GELOGI("output concat dim is non-const, cannot resolve score at compile time");
    return ge::SUCCESS;
  }
  int64_t concat_dim_size = -1;
  GE_ASSERT_TRUE(output_concat_dim.GetConstValue(concat_dim_size),
                 "Failed to get int value, expr = %s",
                 output_concat_dim.Str().get());
  int64_t num_aligned = 0U;
  auto align_threshold =
      static_cast<int64_t>(std::ceil(static_cast<double>(concat_dim_size) * (1.0 - kMaxUnalignedRate)));
  for (size_t i = 0U; i < num_inputs_; ++i) {
    const auto &input = concat_node_->inputs[i];
    GE_WARN_ASSERT(input.attr.repeats[concat_dim_].IsConstExpr(), "concat dim of input[%zu] is non-const", i);
    int64_t dim = -1;
    GE_ASSERT_TRUE(input.attr.repeats[concat_dim_].GetConstValue(dim),
                   "Failed to get int value, expr = %s",
                   input.attr.repeats[concat_dim_].Str().get());
    if ((dim * const_part_stride_) % kAlignment_ == 0) {
      GELOGD("input[%zu] is aligned, dim_size = %ld", i, dim);
      num_aligned += dim;
    } else {
      break;
    }
  }
  if (num_aligned >= align_threshold) {
    score = 1;
    GELOGI("aligned = %u/%u, reaches aligned threshold", num_aligned, concat_dim_size);
    return ASCCOMMON_SUC;
  }
  // concat dim之后的dim全为const, 不满足对齐率则为不对齐
  if (stride_.IsConstExpr()) {
    score = -1;
    GELOGI("aligned = %u/%u, did not reach aligned threshold", num_aligned, concat_dim_size);
    return ASCCOMMON_SUC;
  }

  // concat dim之后的dim不全为const, 无法确定是否对齐
  score = 0;
  GELOGI("Cannot resolve score at compile time");
  return ASCCOMMON_SUC;
}

Status ConcatScoreFunctionGenerator::GenerateForUnaligned() {
  // 该模板必然为第0个result的第0个group
  const auto &tiling_data = "tiling_data.graph0_result0_g0_tiling_data";
  std::vector<std::pair<ge::Expression, ge::Expression>> replacements;
  SizeVarSet original_var_set;
  AscGraphInfoComplete::AppendOriginalSizeVar(*graph_, original_var_set);
  for (const auto &size_var : original_var_set) {
    if (!size_var.IsConstExpr()) {
      replacements.emplace_back(size_var, ge::Symbol((std::string("t.") + size_var.Str().get()).c_str()));
    }
  }
  auto &output_concat_dim = concat_node_->outputs[0].attr.repeats[concat_dim_];
  ss_ << "  const auto &t = " << tiling_data << ";" << std::endl;
  ss_ << "  auto stride = static_cast<int64_t>(" << stride_.Replace(replacements).Str().get() << ");" << std::endl;
  ss_ << "  if (stride % 32 == 0) { return 1; }" << std::endl;
  ss_ << "  std::vector<int64_t> concat_dims;" << std::endl;
  ss_ << "  concat_dims.reserve(" << num_inputs_ << ");" << std::endl;
  for (size_t i = 0U; i < num_inputs_; ++i) {
    ss_ << "  concat_dims.emplace_back(static_cast<int64_t>("
        << concat_node_->inputs[i].attr.repeats[concat_dim_].Replace(replacements).Str().get()
        << "));" << std::endl;
  }
  ss_ << "  size_t num_unaligned = 0U;" << std::endl;
  ss_ << "  size_t max_unaligned = static_cast<int64_t>(std::ceil(static_cast<double>(";
  ss_ << output_concat_dim.Replace(replacements).Str().get() << ") * " << kMaxUnalignedRate << "));" << std::endl;
  ss_ << "  bool already_unaligned = false;" << std::endl;
  ss_ << "  for (int32_t i = 0; i < " << num_inputs_ << "; ++i) {" << std::endl;
  ss_ << "    if (already_unaligned || (stride * concat_dims[i] % 32) != 0) {" << std::endl;
  ss_ << "      num_unaligned += concat_dims[i];" << std::endl;
  ss_ << "      already_unaligned = true;" << std::endl;
  ss_ << "      if (num_unaligned >= max_unaligned) { return -1; }" << std::endl;
  ss_ << "    }" << std::endl;
  ss_ << "  }" << std::endl;
  ss_ << "  return 1;" << std::endl;
  ss_ << "}" << std::endl;
  GELOGI("code generated: %s", ss_.str().c_str());
  return ASCCOMMON_SUC;
}

Status ConcatScoreFunctionGenerator::GenerateForCheckSmallTail(std::string &score_func) {
  ss_ << "int32_t CalcScore(const AutofuseTilingData &tiling_data) {" << std::endl;
  GE_CHK_STATUS_RET(ParseStride());
  if (const_part_stride_ % kAlignment_ == 0U) {
    GenerateReturnValue(-1); // 使用default 模板即可
    score_func = ss_.str();
    return ASCCOMMON_SUC;
  }

  // 该模板必然为第1个result的第0个group
  const auto &tiling_data = "tiling_data.graph0_result1_g0_tiling_data";
  std::vector<std::pair<ge::Expression, ge::Expression>> replacements;
  SizeVarSet original_var_set;
  AscGraphInfoComplete::AppendOriginalSizeVar(*graph_, original_var_set);
  for (const auto &size_var : original_var_set) {
    if (!size_var.IsConstExpr()) {
      replacements.emplace_back(size_var, ge::Symbol((std::string("t.") + size_var.Str().get()).c_str()));
    }
  }

  auto dst_cols = concat_node_->outputs[0].attr.repeats[concat_dim_] * elt_num_after_concat_dim_;
  ss_ << "  const auto &t = " << tiling_data << ";" << std::endl;
  ss_ << "  auto stride = static_cast<int64_t>(" << stride_.Replace(replacements).Str().get() << ");" << std::endl;
  ss_ << "  if (stride % 32 == 0) { return -1; }" << std::endl;
  ss_ << "  auto dst_cols = " << dst_cols.Replace(replacements).Str().get() << ";" << std::endl;
  ss_ << "  if (dst_cols > 96) { return -1; }" << std::endl;
  for (uint32_t i = 0U; i < concat_node_->inputs.Size(); ++i) {
    std::string src_col_name = "src_cols_" + std::to_string(i);
    auto src_col_expr = concat_node_->inputs[i].attr.repeats[concat_dim_] * elt_num_after_concat_dim_;
    ss_ << "  auto " << src_col_name << " = " << src_col_expr.Replace(replacements).Str().get() << ";" << std::endl;
    ss_ << "  if (" << src_col_name << " > 64) { return -1; }" << std::endl;
  }
  ss_ << "  return 1;" << std::endl;
  ss_ << "}" << std::endl;
  GELOGI("code generated: %s", ss_.str().c_str());
  score_func = ss_.str();
  return ASCCOMMON_SUC;
}

void ConcatScoreFunctionGenerator::GenerateScoreOne(std::string &score_func) {
  std::stringstream ss;
  ss << "int32_t CalcScore(const AutofuseTilingData &tiling_data) {" << std::endl;
  ss << "  return 1;" << std::endl;
  ss << "}" << std::endl;
  score_func = ss.str();
}
}  // optimize