/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_SYMBOLIC_INFER_SYMBOLIC_SHAPE_SYMBOLIC_SHAPE_SYMBOLIZER_H_
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_SYMBOLIC_INFER_SYMBOLIC_SHAPE_SYMBOLIC_SHAPE_SYMBOLIZER_H_

#include "ge_common/ge_api_types.h"
#include "graph/gnode.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"
#include "attribute_group/attr_group_shape_env.h"

namespace ge {

class InputShapeSource : public ge::Source {
 public:
  InputShapeSource(int32_t input_data_idx, size_t dim_idx) : input_data_idx_(input_data_idx), dim_idx_(dim_idx) {}
  // 兼容上库，待删除
  [[nodiscard]] int32_t GetInputDataIdx() const override {
    return input_data_idx_;
  };
  [[nodiscard]] size_t GetDimIdx() const override {
    return dim_idx_;
  }
  [[nodiscard]] std::string GetSourceStr() const override;

 private:
  int32_t input_data_idx_;  // Data的index，描述symbol来自于graph输入中第几个输入data
  size_t dim_idx_;          // 描述symbol来自于data中对应shape的第几个dim
};

class InputValueSumSource : public ge::Source {
 public:
  InputValueSumSource(int32_t input_data_idx, ge::DataType dtype)
      : input_data_idx_(input_data_idx), dtype_(dtype) {}

  [[nodiscard]] std::string GetSourceStr() const override;

 private:
  int32_t input_data_idx_;  // Data的index，描述symbol来自于graph输入中第几个输入data
  ge::DataType dtype_;      // 描述value的数据类型，用于后续执行时取值
};

class InputRankSource final : public ge::Source {
public:
  explicit InputRankSource(const int32_t input_data_idx)
      : input_data_idx_(input_data_idx) {}

  [[nodiscard]] std::string GetSourceStr() const override;

private:
  int32_t input_data_idx_;  // Data的index，描述symbol来自于graph输入中第几个输入data
};

class SymbolicShapeSymbolizer {
public:
  /**
   * 符号化graph图中的输入shape，根据本次执行的输入 shape、Data 节点的 -1 标注，将未知维度的 shape 符号化，
   * 并将符号化的 shape 保存到 graph 的ShapeEnvAttr属性中。
   * 后续我们会支持多种符号化泛化规则，比如某些二类算子依赖value则泛化Tensorvalue，对于dim=-2泛化Tensorrank等
   *
   * 默认的符号化shape规则为：
   * 1. 若确定某维度为固定值，则创建常量符号
   * 2. 若某维度为动态，则创建变量符号
   * 3. 若两个动态维度对应的`graph_inputs` Tensor 中的实际值相同，先创建不同符号，后续由guard消减符号，减少重编译次数
   *
   * @param graph 待符号化的图
   * @param graph_inputs 本次执行的输入 shape、Data 节点的 -1 标注
   * @return 成功返回 SUCCESS，失败返回对应错误码
   */
  static Status Symbolize(const ComputeGraphPtr &graph, const std::vector<GeTensor> &graph_inputs);
};
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_SYMBOLIC_INFER_SYMBOLIC_SHAPE_SYMBOLIC_SHAPE_SYMBOLIZER_H_
