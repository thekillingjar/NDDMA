/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_GRAPH_IR_META_H_
#define METADEF_CXX_GRAPH_IR_META_H_

#include <string>
#include <vector>
#include "graph/ascend_limits.h"
#include "graph/small_vector.h"
#include "graph/op_desc.h"
#include "graph/ir/ir_data_type_symbol_store.h"

namespace ge {
/**
 *  IR信息
 */
class IRMetaData {
  struct IrInputs {
    void AppendIrInput(const std::string &name, IrInputType input_type) {
      if (ir_input_names.insert(name).second) {
        ir_inputs.emplace_back(name, input_type);
      }
    }
    void RemoveIrInput(const std::string &name) {
      if (ir_input_names.erase(name) > 0) {
        ir_inputs.erase(std::remove_if(ir_inputs.begin(), ir_inputs.end(),
                                       [&name](const std::pair<std::string, IrInputType> &pair) {
                                         return pair.first == name;
                                       }),
                       ir_inputs.end());
      }
    }
    std::unordered_set<std::string> ir_input_names;
    std::vector<std::pair<std::string, IrInputType>> ir_inputs;
  };
  struct IrOutputs {
    void AppendIrOutput(const std::string &name, IrOutputType output_type) {
      if (ir_output_names.insert(name).second) {
        ir_outputs.emplace_back(name, output_type);
      }
    }
    std::unordered_set<std::string> ir_output_names;
    std::vector<std::pair<std::string, IrOutputType>> ir_outputs;
  };
 public:
  explicit IRMetaData(const std::string &op_name) : op_name_(op_name) {};
  IRMetaData() = default;
  void SetOpName(const std::string &op_name) {
    op_name_ = op_name;
  }
  void AppendIrInput(std::string name, IrInputType input_type);
  const std::vector<std::pair<std::string, IrInputType>> &GetIrInputs() const;
  IrInputType GetIrInputType(const std::string &name) const;
  void RemoveIrInput(const std::string &name);

  void AppendIrOutput(std::string name, IrOutputType output_type);
  const std::vector<std::pair<std::string, IrOutputType>> &GetIrOutputs() const;

  graphStatus AddRegisterInputName(const std::string &name);
  std::vector<std::string> GetRegisterInputName() const;

  graphStatus AddRegisterOptionalInputName(const std::string &name);
  std::set<std::string> GetOptionalInputName() const;
  bool IsOptionalInput(const std::string &name) const;

  graphStatus AddRegisterOutputName(const std::string &name);
  std::vector<std::string> GetRegisterOutputName() const;

  void AppendIrAttrName(std::string name);
  const std::vector<std::string> &GetIrAttrNames() const;
  void RemoveIrAttrName(const std::string &name);

  void RegisterSubgraphIrName(const std::string &name, const SubgraphType type);
  const std::map<std::string, SubgraphType> &GetSubgraphIrNames() const;
  /**
   * @brief Get subgraph names in IR order
   * @return subgraph ir names in IR order
   */
  const std::vector<std::pair<std::string, SubgraphType>> &GetOrderedSubgraphIrNames() const;
  SubgraphType GetSubgraphTypeByIrName(const std::string &name) const;

  IRDataTypeSymbolStore &MutableIRDataTypeSymbolStore();
  const IRDataTypeSymbolStore &GetIRDataTypeSymbolStore() const;

  bool operator==(const IRMetaData &other) const;

 private:
  std::string op_name_;
  IrInputs ir_inputs_;
  IrOutputs ir_outputs_;
  std::vector<std::string> register_input_name_; // todo need to deprecate
  std::set<std::string> optional_input_names_; // todo need to deprecate
  std::vector<std::string> register_output_name_;
  std::vector<std::string> ir_attr_names_;
  // subgraph ir names to type, for a `if` operator:
  // then_branch: static
  // else_branch: static
  // or for a `case` op:
  // branches: dynamic
  std::map<std::string, SubgraphType> subgraph_ir_names_to_type_;
  IRDataTypeSymbolStore dtype_symbol_store_;
  std::set<std::string> register_unique_name_;
  std::vector<std::pair<std::string, SubgraphType>> subgraph_ir_names_to_type_ordered_;
};

class OpMetadata {
 public:
  using SmallIntVector = SmallVector<int64_t, static_cast<size_t>(kDefaultMaxInputNum)>;
  OpMetadata() = default;
  ~OpMetadata() = default;
  OpMetadata(std::string name, std::string type) : name_(std::move(name)), type_(std::move(type)), ir_meta_(name) {}
  int64_t GetId() const {return id_;}
  int64_t GetStreamId() const {return stream_id_;}
  const std::vector<std::string> &GetInputNames() const {return input_names_;}
  const std::vector<std::string> &GetSrcNames() const {return src_names_;}
  const std::vector<int64_t> &GetSrcIndexes() const {return src_indexes_;}
  const std::vector<std::string> &GetDstNames() const {return dst_names_;}
  const std::vector<int64_t> &GetDstIndexes() const {return dst_indexes_;}
  const std::vector<int64_t> &GetInputOffsets() const {return input_offsets_;}
  const std::vector<int64_t> &GetOutputOffsets() const {return output_offsets_;}
  const std::vector<bool> &GetIsInputConsts() const {return is_input_consts_;}
  const std::vector<std::string> &GetSubgraphNames() const {return subgraph_names_;}
  void AddSubGraphName(const std::string &name) {subgraph_names_.push_back(name);}
  void ClearSubgraphNames() { subgraph_names_.clear(); }
  void SetOpName(std::string name) {
    name_ = std::move(name);
    ir_meta_.SetOpName(name);
  }

 private:
  friend class OpDescImpl;
  std::string name_;
  std::string type_;
  std::vector<std::string> inputs_;
  bool has_out_attr_{false};
  int64_t id_{0};
  int64_t stream_id_{0};
  std::vector<std::string> input_names_;
  std::vector<std::string> src_names_;
  std::vector<int64_t> src_indexes_;
  std::vector<std::string> dst_names_;
  std::vector<int64_t> dst_indexes_;
  std::vector<int64_t> input_offsets_;
  std::vector<int64_t> output_offsets_;
  SmallIntVector workspaces;
  SmallIntVector workspace_bytes_list_;
  std::vector<bool> is_input_consts_;
  std::vector<std::string> subgraph_names_;
  IRMetaData ir_meta_;
};
} // namespace ge
#endif // METADEF_CXX_GRAPH_IR_META_H_
