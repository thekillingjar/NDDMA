/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef METADEF_CXX_INC_GRAPH_UTILS_RECOVER_IR_UTILS_H_
#define METADEF_CXX_INC_GRAPH_UTILS_RECOVER_IR_UTILS_H_
namespace ge {
enum class CompatibilityStrategy {
  kForward,   // 前向兼容：直构图IR版本 > 兼容图IR版本
  kBackward,  // 后向兼容：直构图IR版本 < 兼容图IR版本
  kNone,      // 无差异：直构图IR版本 = 兼容图IR版本
  kFailed     // 推导失败：属性与输入方向不一致
};

class RecoverIrUtils {
 public:
  using InputIrDefs = std::vector<std::pair<std::string, ge::IrInputType>>;
  using OutputIrDefs = std::vector<std::pair<std::string, ge::IrOutputType>>;
  template<typename IrType>
  using IrDefAppender =
  std::function<void(const ge::OpDescPtr &op_desc, const std::string &ir_name, const IrType ir_type)>;

  struct IrDefinition {
    bool inited{false};
    bool has_ir_definition{false};
    std::vector<std::string> attr_names;
    std::map<std::string, ge::AnyValue> attr_value;
    InputIrDefs inputs;
    OutputIrDefs outputs;
    ge::OpDescPtr op_desc{nullptr};
    std::vector<std::uint8_t> is_required_attr;
    CompatibilityStrategy strategy{CompatibilityStrategy::kNone};
  };
  static ge::graphStatus RecoverOpDescIrDefinition(const ge::OpDescPtr &desc,
                                                   const std::string &op_type,
                                                   IrDefinition &ir_def);
  static graphStatus RecoverIrDefinitions(const ge::ComputeGraphPtr &graph, const vector<std::string> &attr_names = {});
  static graphStatus RecoverOpDescIrDefinition(const ge::OpDescPtr &desc, const std::string &op_type = "");
  static void InitIrDefinitionsIfNeed(const std::string &op_type, IrDefinition &ir_def);
  static CompatibilityStrategy DeriveCompatibilityStrategy(const ge::OpDescPtr &desc, const IrDefinition &ir_def);
  static graphStatus RecoverIrAttrDefaultValue(const ge::OpDescPtr &desc, const string &op_type, IrDefinition &ir_def);
  static graphStatus RecoverIrAttrNames(const ge::OpDescPtr &desc, IrDefinition &ir_def);
  static graphStatus RecoverIrInputAndOutput(const ge::OpDescPtr &desc, IrDefinition &ir_def);

  static graphStatus ProcessForwardCompatInputs(const ge::OpDescPtr &desc, const InputIrDefs &ir_inputs_in_node,
                                                const IrDefinition &ir_def);
};
}  // namespace ge

#endif // METADEF_CXX_INC_GRAPH_UTILS_RECOVER_IR_UTILS_H_
