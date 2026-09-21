/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "compliant_op_desc_builder.h"
#include "graph/op_desc.h"
#include "graph/utils/op_desc_utils.h"
#include "common/checker.h"
namespace ge {
namespace {
class AnyTypeOperator : public Operator {
 public:
  AnyTypeOperator(const char_t *name, const char_t *type) : Operator(name, type) {}
  using Operator::DynamicInputRegister;
  using Operator::InputRegister;
  using Operator::OptionalInputRegister;

  using Operator::DynamicOutputRegister;
  using Operator::OutputRegister;

  using Operator::AttrRegister;
  using Operator::RequiredAttrWithTypeRegister;
};
}  // namespace
CompliantOpDescBuilder &CompliantOpDescBuilder::OpType(const char_t *type) {
  type_ = type;
  return *this;
}
CompliantOpDescBuilder &CompliantOpDescBuilder::IrDefInputs(std::vector<IrInputDef> input_ir_def) {
  ir_def_inputs_ = std::move(input_ir_def);
  return *this;
}
CompliantOpDescBuilder &CompliantOpDescBuilder::IrDefOutputs(std::vector<IrOutputDef> output_ir_def) {
  ir_def_outputs_ = std::move(output_ir_def);
  return *this;
}
CompliantOpDescBuilder &CompliantOpDescBuilder::IrDefAttrs(std::vector<IrAttrDef> attr_ir_def) {
  ir_def_attrs_ = std::move(attr_ir_def);
  return *this;
}
CompliantOpDescBuilder &CompliantOpDescBuilder::IrDefSubgraphs(std::vector<std::string> subgraph_ir_def) {
  ir_def_subgraphs_ = std::move(subgraph_ir_def);
  return *this;
}
CompliantOpDescBuilder &CompliantOpDescBuilder::Name(const char_t *name) {
  name_ = name;
  return *this;
}
OpDescPtr CompliantOpDescBuilder::Build() const {
  AnyTypeOperator op{name_.c_str(), type_.c_str()};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  for (const auto &input : ir_def_inputs_) {
    if (input.ir_input_type == kIrInputRequired) {
      op.InputRegister(input.name.c_str(), input.symbol_id.c_str());
    } else if (input.ir_input_type == kIrInputOptional) {
      op.OptionalInputRegister(input.name.c_str(), input.symbol_id.c_str());
    } else {
      op.DynamicInputRegister(input.name.c_str(), 0, input.symbol_id.c_str(), true);
      auto iter = dynamic_input_ir_names_to_inst_num_.find(input.name);
      if (iter != dynamic_input_ir_names_to_inst_num_.end()) {
        op.DynamicInputRegister(input.name.c_str(), iter->second, true);
      }
    }
  }

  for (const auto &output : ir_def_outputs_) {
    if (output.ir_output_type == kIrOutputRequired) {
      op.OutputRegister(output.name.c_str(), output.symbol_id.c_str());
    } else {
      op.DynamicOutputRegister(output.name.c_str(), 0, output.symbol_id.c_str(), true);
      auto iter = dynamic_output_ir_names_to_inst_num_.find(output.name);
      if (iter != dynamic_output_ir_names_to_inst_num_.end()) {
        op.DynamicOutputRegister(output.name.c_str(), iter->second, true);
      }
    }
  }
  for (const auto &output_name_and_td : output_names_to_td_) {
    op.UpdateOutputDesc(output_name_and_td.first.c_str(), output_name_and_td.second);
  }

  // todo 这里做得不够好，具体体现为这里构造`op_desc`的时候，没有使用`Operator`的方法。
  //   这会导致，如果`Operator`造`OpDesc`的方式有变化的时候，可能会忘记修改这里的代码，
  //   但是直接使用`Operator`的方法构造属性是较为困难的，因为`Operator`的属性构造没有开放`AnyValue`接口，
  //   而是一堆根据属性类型重载的`AttrRegister`，正式版本中，要解决这个问题
  for (const auto &attr : ir_def_attrs_) {
    if (attr.ir_attr_type == kAttrRequired) {
      op_desc->AddRequiredAttrWithType(attr.attr_name, attr.attr_data_type);
    }
    op_desc->AppendIrAttrName(attr.attr_name);
    GE_ASSERT_GRAPH_SUCCESS(op_desc->SetAttr(attr.attr_name, attr.attr_default_value));
  }

  // todo subgraph

  op.BreakConnect();
  return op_desc;
}
CompliantOpDescBuilder &CompliantOpDescBuilder::InstanceDynamicInputNum(const char_t *ir_name, int32_t num) {
  dynamic_input_ir_names_to_inst_num_[ir_name] = num;
  return *this;
}
CompliantOpDescBuilder &CompliantOpDescBuilder::InstanceDynamicOutputNum(const char_t *ir_name, int32_t num) {
  dynamic_output_ir_names_to_inst_num_[ir_name] = num;
  return *this;
}
CompliantOpDescBuilder &CompliantOpDescBuilder::InstanceOutputShape(const char_t *name, const vector<int64_t> &shape) {
  return InstanceOutputOriginShape(name, shape).InstanceOutputStorageShape(name, shape);
}
CompliantOpDescBuilder &CompliantOpDescBuilder::InstanceOutputOriginShape(const char_t *name,
                                                                          const vector<int64_t> &shape) {
  auto &td = output_names_to_td_[name];
  td.SetOriginShape(Shape{shape});
  return *this;
}
CompliantOpDescBuilder &CompliantOpDescBuilder::InstanceOutputStorageShape(const char_t *name,
                                                                           const vector<int64_t> &shape) {
  auto &td = output_names_to_td_[name];
  td.SetShape(Shape{shape});
  return *this;
}
}  // namespace ge