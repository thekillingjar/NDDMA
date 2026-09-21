/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge_running_env/fake_op.h"
#include "fake_op_repo.h"
#include "ge_running_env/info_store_holder.h"
#include "graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"

FAKE_NS_BEGIN
auto default_infer_fun = [](Operator &op) -> graphStatus { 
  return GRAPH_SUCCESS;
};

FakeOp::FakeOp(const std::string &op_type) : op_type_(op_type), info_fun_(default_infer_fun) {}

FakeOp& FakeOp::Inputs(const std::vector<std::string>& inputs) {
  inputs_ = inputs;
  return *this;
}

FakeOp& FakeOp::Outputs(const std::vector<std::string>& outputs) {
  outputs_ = outputs;
  return *this;
}

FakeOp& FakeOp::InferShape(InferShapeFunc infer_fun) {
  info_fun_ = infer_fun;
  return *this;
}

FakeOp& FakeOp::InfoStoreAndBuilder(const std::string& name) {
  info_store_names_.insert(name);
  return *this;
}

namespace {

void RegistOpToInfoStore(OpsKernelInfoStorePtr& info_store, const std::string& op_type) {
  if (info_store == nullptr) {
    return;
  }
  auto holder = dynamic_cast<InfoStoreHolder*>(info_store.get());
  if (holder != nullptr) {
    holder->RegistOp(op_type);
  }
}

struct FakeOperator : Operator {
  FakeOperator(const std::string& op_type) : Operator(op_type) {}

  FakeOperator& RegistInputs(const std::vector<std::string>& inputs) {
    for (auto& input : inputs) {
      Operator::InputRegister(input);
    }
    return *this;
  }

  FakeOperator& RegistOutputs(const std::vector<std::string>& outputs) {
    for (auto& output : outputs) {
      Operator::OutputRegister(output);
    }
    return *this;
  }

  FakeOperator& AttrRegister(const std::map<std::string, std::variant<int64_t, std::string>>& attrs) {
    for (auto& attr : attrs) {
      std::visit([this, name = attr.first](const auto& value) {
        Operator::AttrRegister(name, value);
      }, attr.second);
    }
    
    return *this;
  }

  FakeOperator& ExtAttrRegister(const std::map<std::string, std::variant<OpKernelBinPtr>>& ext_attrs) {
    for (auto& attr : ext_attrs) {
      std::visit([this, name = attr.first](const auto& value) {
        auto op_desc = OpDescUtils::GetOpDescFromOperator(*this);
        op_desc->SetExtAttr(name, value);
      }, attr.second);
    }
    return *this;
  }
};
}  // namespace

void FakeOp::InstallTo(std::map<string, OpsKernelInfoStorePtr>& info_stores) const {
  std::for_each(info_store_names_.begin(), info_store_names_.end(), [=, &info_stores](auto& info_store_name) {
    auto iter = info_stores.find(info_store_name);
    if (iter != info_stores.end()) {
      RegistOpToInfoStore(iter->second, op_type_);
    }
  });
}

void FakeOp::Install() const {
  FakeOpRepo::Regist(
    op_type_,
    [op_type = this->op_type_, inputs = this->inputs_, outputs = this->outputs_, attrs = this->attrs_, ext_attrs = this->ext_attrs_](const std::string&) -> Operator {
      return FakeOperator(op_type).RegistInputs(inputs).RegistOutputs(outputs).AttrRegister(attrs).ExtAttrRegister(ext_attrs);
    });
  if (info_fun_) {
    FakeOpRepo::Regist(op_type_, info_fun_);
  }
}

FAKE_NS_END
