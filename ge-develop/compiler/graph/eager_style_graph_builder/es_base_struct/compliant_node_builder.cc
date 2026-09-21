/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "compliant_node_builder.h"
#include "graph/graph.h"
#include "common/checker.h"
namespace ge {
namespace es {

using IrInputDef = CompliantNodeBuilder::IrInputDef;
using IrOutputDef = CompliantNodeBuilder::IrOutputDef;
using IrAttrDef = CompliantNodeBuilder::IrAttrDef;
using AnyTypeOperator = CompliantNodeBuilder::AnyTypeOperator;

class CompliantNodeBuilder::CompliantNodeBuilderImpl {
 public:
  explicit CompliantNodeBuilderImpl(ge::Graph *graph) : owner_graph_(graph) {}
  void OpType(const char_t *type) {
    type_ = type;
  }
  void IrDefInputs(std::vector<IrInputDef> input_ir_def) {
    ir_def_inputs_ = std::move(input_ir_def);
  }
  void IrDefOutputs(std::vector<IrOutputDef> output_ir_def) {
    ir_def_outputs_ = std::move(output_ir_def);
  }
  void IrDefAttrs(std::vector<IrAttrDef> attr_ir_def) {
    ir_def_attrs_ = std::move(attr_ir_def);
  }
  void Name(const char_t *name) {
    name_ = name;
  }
  GNode Build() const {
    GE_ASSERT_NOTNULL(owner_graph_);
    AnyTypeOperator op{name_.c_str(), type_.c_str()};
    RegisterInputs(op);
    RegisterOutputs(op);
    GE_ASSERT_SUCCESS(UpdateOutputDescs(op));
    GE_ASSERT_SUCCESS(RegisterAttrs(op));

    op.BreakConnect();
    return owner_graph_->AddNodeByOp(op);
  }

  void RegisterInputs(AnyTypeOperator &op) const {
    for (const auto &input : ir_def_inputs_) {
      if (input.ir_input_type == CompliantNodeBuilder::kEsIrInputRequired) {
        op.InputRegister(input.name.c_str(), input.symbol_id.c_str());
      } else if (input.ir_input_type == CompliantNodeBuilder::kEsIrInputOptional) {
        op.OptionalInputRegister(input.name.c_str(), input.symbol_id.c_str());
      } else {
        op.DynamicInputRegister(input.name.c_str(), 0, input.symbol_id.c_str(), true);
        auto iter = dynamic_input_ir_names_to_inst_num_.find(input.name);
        if (iter != dynamic_input_ir_names_to_inst_num_.end()) {
          op.DynamicInputRegister(input.name.c_str(), iter->second, true);
        }
      }
    }
  }

  void RegisterOutputs(AnyTypeOperator &op) const {
    for (const auto &output : ir_def_outputs_) {
      if (output.ir_output_type == CompliantNodeBuilder::kEsIrOutputRequired) {
        op.OutputRegister(output.name.c_str(), output.symbol_id.c_str());
      } else {
        op.DynamicOutputRegister(output.name.c_str(), 0, output.symbol_id.c_str(), true);
        auto iter = dynamic_output_ir_names_to_inst_num_.find(output.name);
        if (iter != dynamic_output_ir_names_to_inst_num_.end()) {
          op.DynamicOutputRegister(output.name.c_str(), iter->second, true);
        }
      }
    }
  }
  Status UpdateOutputDescs(AnyTypeOperator &op) const {
    for (const auto &output_name_and_td : output_names_to_td_) {
      GE_ASSERT_GRAPH_SUCCESS(op.UpdateOutputDesc(output_name_and_td.first.c_str(), output_name_and_td.second));
    }
    return SUCCESS;
  }

  Status RegisterAttrs(AnyTypeOperator &op) const {
    for (const auto &attr : ir_def_attrs_) {
      if (attr.ir_attr_type == CompliantNodeBuilder::kEsAttrRequired) {
        op.RequiredAttrWithTypeRegister(attr.attr_name.c_str(), attr.attr_data_type.c_str());
        (void)op.SetAttr(attr.attr_name.c_str(), attr.attr_default_value);
      } else {
        op.AttrRegister(attr.attr_name.c_str(), attr.attr_default_value);
      }
    }
    return SUCCESS;
  }

  void InstanceDynamicInputNum(const char_t *ir_name, int32_t num) {
    dynamic_input_ir_names_to_inst_num_[ir_name] = num;
  }
  void InstanceDynamicOutputNum(const char_t *ir_name, int32_t num) {
    dynamic_output_ir_names_to_inst_num_[ir_name] = num;
  }
  void InstanceOutputShape(const char_t *name, const vector<int64_t> &shape) {
    InstanceOutputOriginShape(name, shape);
    InstanceOutputStorageShape(name, shape);
  }
  void InstanceOutputOriginShape(const char_t *name, const vector<int64_t> &shape) {
    auto &td = output_names_to_td_[name];
    td.SetOriginShape(Shape{shape});
  }
  void InstanceOutputStorageShape(const char_t *name, const vector<int64_t> &shape) {
    auto &td = output_names_to_td_[name];
    td.SetShape(Shape{shape});
  }
  void InstanceOutputDataType(const char_t *name, const ge::DataType data_type) {
    auto &td = output_names_to_td_[name];
    td.SetDataType(data_type);
  }
  void InstanceOutputFormat(const char_t *name, ge::Format format) {
    InstanceOutputOriginFormat(name, format);
    InstanceOutputStorageFormat(name, format);
  }
  void InstanceOutputOriginFormat(const char_t *name, ge::Format format) {
    auto &td = output_names_to_td_[name];
    td.SetOriginFormat(format);
  }
  void InstanceOutputStorageFormat(const char_t *name, ge::Format format) {
    auto &td = output_names_to_td_[name];
    td.SetFormat(format);
  }

 private:
  ge::Graph *owner_graph_{nullptr};

  // IR定义相关成员
  std::string type_;
  std::vector<IrInputDef> ir_def_inputs_;
  std::vector<IrOutputDef> ir_def_outputs_;
  std::vector<IrAttrDef> ir_def_attrs_;

  // 实例信息相关成员
  std::string name_;
  std::unordered_map<std::string, int32_t> dynamic_input_ir_names_to_inst_num_;
  std::unordered_map<std::string, int32_t> dynamic_output_ir_names_to_inst_num_;
  std::unordered_map<std::string, TensorDesc> output_names_to_td_;
};

CompliantNodeBuilder::CompliantNodeBuilder(ge::Graph *graph) {
  // new失败说明业务无法正常进行，主动抛出异常
  impl_ = std::make_unique<CompliantNodeBuilderImpl>(graph);
}
CompliantNodeBuilder::~CompliantNodeBuilder() = default;
CompliantNodeBuilder::CompliantNodeBuilder(CompliantNodeBuilder &&) noexcept = default;
CompliantNodeBuilder &CompliantNodeBuilder::operator=(CompliantNodeBuilder &&) noexcept = default;

CompliantNodeBuilder &CompliantNodeBuilder::OpType(const char_t *type) {
  impl_->OpType(type);
  return *this;
}
CompliantNodeBuilder &CompliantNodeBuilder::IrDefInputs(std::vector<IrInputDef> input_ir_def) {
  impl_->IrDefInputs(input_ir_def);
  return *this;
}
CompliantNodeBuilder &CompliantNodeBuilder::IrDefOutputs(std::vector<IrOutputDef> output_ir_def) {
  impl_->IrDefOutputs(output_ir_def);
  return *this;
}
CompliantNodeBuilder &CompliantNodeBuilder::IrDefAttrs(std::vector<IrAttrDef> attr_ir_def) {
  impl_->IrDefAttrs(attr_ir_def);
  return *this;
}
CompliantNodeBuilder &CompliantNodeBuilder::Name(const char_t *name) {
  impl_->Name(name);
  return *this;
}
GNode CompliantNodeBuilder::Build() const {
  return impl_->Build();
}

CompliantNodeBuilder &CompliantNodeBuilder::InstanceDynamicInputNum(const char_t *ir_name, int32_t num) {
  impl_->InstanceDynamicInputNum(ir_name, num);
  return *this;
}
CompliantNodeBuilder &CompliantNodeBuilder::InstanceDynamicOutputNum(const char_t *ir_name, int32_t num) {
  impl_->InstanceDynamicOutputNum(ir_name, num);
  return *this;
}
CompliantNodeBuilder &CompliantNodeBuilder::InstanceOutputShape(const char_t *name, const vector<int64_t> &shape) {
  impl_->InstanceOutputShape(name, shape);
  return *this;
}
CompliantNodeBuilder &CompliantNodeBuilder::InstanceOutputOriginShape(const char_t *name,
                                                                      const vector<int64_t> &shape) {
  impl_->InstanceOutputOriginShape(name, shape);
  return *this;
}
CompliantNodeBuilder &CompliantNodeBuilder::InstanceOutputStorageShape(const char_t *name,
                                                                       const vector<int64_t> &shape) {
  impl_->InstanceOutputStorageShape(name, shape);
  return *this;
}
CompliantNodeBuilder &CompliantNodeBuilder::InstanceOutputDataType(const char_t *name, ge::DataType data_type) {
  impl_->InstanceOutputDataType(name, data_type);
  return *this;
}
CompliantNodeBuilder &CompliantNodeBuilder::InstanceOutputFormat(const char_t *name, ge::Format format) {
  impl_->InstanceOutputFormat(name, format);
  return *this;
}
CompliantNodeBuilder &CompliantNodeBuilder::InstanceOutputOriginFormat(const char_t *name, ge::Format format) {
  impl_->InstanceOutputOriginFormat(name, format);
  return *this;
}
CompliantNodeBuilder &CompliantNodeBuilder::InstanceOutputStorageFormat(const char_t *name, ge::Format format) {
  impl_->InstanceOutputStorageFormat(name, format);
  return *this;
}

ge::graphStatus AddEdgeAndUpdatePeerDesc(Graph &graph, GNode &src_node, int32_t src_port_index, GNode &dst_node,
                                         int32_t dst_port_index) {
  GE_ASSERT_GRAPH_SUCCESS(graph.AddDataEdge(src_node, src_port_index, dst_node, dst_port_index), "Add edge failed");
  TensorDesc dst_tensor_desc;
  if (dst_node.GetInputDesc(dst_port_index, dst_tensor_desc) == GRAPH_FAILED) {
    GE_ASSERT_GRAPH_SUCCESS(dst_node.UpdateInputDesc(dst_port_index, dst_tensor_desc),
                            "Update InputDesc of index %s using default TensorDesc failed", dst_port_index);
  }
  return ge::GRAPH_SUCCESS;
}

}  // namespace es
}  // namespace ge