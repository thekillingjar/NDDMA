/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_COMPLIANT_OP_DESC_BUILDER_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_COMPLIANT_OP_DESC_BUILDER_H_
#include <vector>
#include <string>
#include <utility>
#include "graph/op_desc.h"
#include "graph/operator.h"

namespace ge {
enum IrAttrType { kAttrRequired, kAttrOptional };
class CompliantOpDescBuilder {
 public:
  struct IrAttrDef {
    std::string attr_name;
    IrAttrType ir_attr_type;
    std::string attr_data_type;  // see `kAttrTypesMap` in `ge_attr_value.cc`
    AnyValue attr_default_value;
  };
  struct IrInputDef {
    std::string name;
    IrInputType ir_input_type;
    std::string symbol_id;
  };
  struct IrOutputDef {
    std::string name;
    IrOutputType ir_output_type;
    std::string symbol_id;
  };

 public:
  CompliantOpDescBuilder &OpType(const char_t *type);

  CompliantOpDescBuilder &IrDefInputs(std::vector<IrInputDef> input_ir_def);
  CompliantOpDescBuilder &IrDefOutputs(std::vector<IrOutputDef> output_ir_def);
  CompliantOpDescBuilder &IrDefAttrs(std::vector<IrAttrDef> attr_ir_def);
  CompliantOpDescBuilder &IrDefSubgraphs(std::vector<std::string> subgraph_ir_def);

  CompliantOpDescBuilder &Name(const char_t *name);
  CompliantOpDescBuilder &InstanceDynamicInputNum(const char_t *ir_name, int32_t num);
  CompliantOpDescBuilder &InstanceDynamicOutputNum(const char_t *ir_name, int32_t num);
  CompliantOpDescBuilder &InstanceOutputShape(const char_t *input_name, const std::vector<int64_t> &shape);
  CompliantOpDescBuilder &InstanceOutputOriginShape(const char_t *name, const std::vector<int64_t> &shape);
  CompliantOpDescBuilder &InstanceOutputStorageShape(const char_t *name, const std::vector<int64_t> &shape);

  OpDescPtr Build() const;

 private:
  // ir definitions
  std::string type_;
  std::vector<IrInputDef> ir_def_inputs_;
  std::vector<IrOutputDef> ir_def_outputs_;
  std::vector<IrAttrDef> ir_def_attrs_;
  std::vector<std::string> ir_def_subgraphs_;

  // instance info
  std::string name_;
  std::unordered_map<std::string, int32_t> dynamic_input_ir_names_to_inst_num_;
  std::unordered_map<std::string, int32_t> dynamic_output_ir_names_to_inst_num_;
  std::unordered_map<std::string, TensorDesc> input_names_to_td_;
  std::unordered_map<std::string, TensorDesc> output_names_to_td_;
};
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_COMPLIANT_OP_DESC_BUILDER_H_
