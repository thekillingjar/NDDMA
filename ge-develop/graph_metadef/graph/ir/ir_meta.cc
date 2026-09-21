/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/ir/ir_meta.h"
#include <algorithm>
#include "common/util/trace_manager/trace_manager.h"
#include "graph/utils/ge_ir_utils.h"

namespace ge {
void IRMetaData::AppendIrAttrName(std::string name) {
  ir_attr_names_.emplace_back(std::move(name));
}
const std::vector<std::string> &IRMetaData::GetIrAttrNames() const {
  return ir_attr_names_;
}
void IRMetaData::RemoveIrAttrName(const std::string &name) {
  ir_attr_names_.erase(std::remove(ir_attr_names_.begin(), ir_attr_names_.end(), name),
                       ir_attr_names_.end());
}
void IRMetaData::AppendIrInput(std::string name, IrInputType input_type) {
  ir_inputs_.AppendIrInput(std::move(name), input_type);
}
const std::vector<std::pair<std::string, IrInputType>> &IRMetaData::GetIrInputs() const {
  return ir_inputs_.ir_inputs;
}
void IRMetaData::RemoveIrInput(const std::string &name) {
  ir_inputs_.RemoveIrInput(name);
}
graphStatus IRMetaData::AddRegisterInputName(const std::string &name) {
  if (register_unique_name_.insert(name).second) {
    register_input_name_.emplace_back(name);
  }
  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "add", TraceManager::GetOutGraphName(),
                   op_name_, "register_input_name", "", "", name);
  return GRAPH_SUCCESS;
}

vector<std::string> IRMetaData::GetRegisterInputName() const {
  return register_input_name_;
}

bool IRMetaData::IsOptionalInput(const std::string &name) const {
  return optional_input_names_.find(name) != optional_input_names_.end();
}

graphStatus IRMetaData::AddRegisterOutputName(const std::string &name) {
  if (register_unique_name_.insert(name).second) {
    register_output_name_.emplace_back(name);
  }

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "add", TraceManager::GetOutGraphName(),
                   op_name_, "register_output_name", "", "", name);
  return GRAPH_SUCCESS;
}

vector<std::string> IRMetaData::GetRegisterOutputName() const {
  return register_output_name_;
}

void IRMetaData::RegisterSubgraphIrName(const std::string &name, const SubgraphType type) {
  subgraph_ir_names_to_type_ordered_.emplace_back(name, type);
  subgraph_ir_names_to_type_[name] = type;
}

const std::map<std::string, SubgraphType> &IRMetaData::GetSubgraphIrNames() const {
  return subgraph_ir_names_to_type_;
}
const std::vector<std::pair<std::string, SubgraphType>> &IRMetaData::GetOrderedSubgraphIrNames() const {
  return subgraph_ir_names_to_type_ordered_;
}

SubgraphType IRMetaData::GetSubgraphTypeByIrName(const std::string &name) const {
  const auto iter = subgraph_ir_names_to_type_.find(name);
  if (iter == subgraph_ir_names_to_type_.end()) {
    return kSubgraphTypeEnd;
  }
  return iter->second;
}

IRDataTypeSymbolStore &IRMetaData::MutableIRDataTypeSymbolStore() {
  return dtype_symbol_store_;
}

const IRDataTypeSymbolStore &IRMetaData::GetIRDataTypeSymbolStore() const {
  return dtype_symbol_store_;
}

graphStatus IRMetaData::AddRegisterOptionalInputName(const string &name) {
  optional_input_names_.insert(name);
  return GRAPH_SUCCESS;
}

bool IRMetaData::operator==(const IRMetaData &other) const {
  return IsEqual(this->optional_input_names_, other.optional_input_names_,
                 "OpDesc.ir_meta.optional_input_names_");
}

std::set<std::string> IRMetaData::GetOptionalInputName() const {
  return optional_input_names_;
}

IrInputType IRMetaData::GetIrInputType(const string &name) const {
  for (const auto &name_2_type : ir_inputs_.ir_inputs) {
    if (name == name_2_type.first) {
      return name_2_type.second;
    }
  }
  return kIrInputTypeEnd;
}

void IRMetaData::AppendIrOutput(std::string name, IrOutputType output_type) {
  ir_outputs_.AppendIrOutput(std::move(name), output_type);
}

const std::vector<std::pair<std::string, IrOutputType>> &IRMetaData::GetIrOutputs() const {
  return ir_outputs_.ir_outputs;
}
} // namespace ge
