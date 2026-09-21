/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "utils.h"
 #include <cstring>
#include "graph/utils/attr_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/default_attr_utils.h"
namespace ge {
namespace es {
namespace {
const char *kNetOutput = "NETOUTPUT";
const std::string kHeaderGuardNormalPrefix = "AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER";
std::unordered_map<std::string, TypeInfo> kAvToTypesInfo{
    {"VT_INT", {false, "VT_INT", "Int", "int64_t ", "int64_t "}},
    {"VT_FLOAT", {false, "VT_FLOAT", "Float", "float ", "float "}},
    {"VT_STRING", {false, "VT_STRING", "String", "const char *", "const char *"}},
    {"VT_BOOL", {false, "VT_BOOL", "Bool", "bool ", "bool "}},
    {"VT_DATA_TYPE", {false, "VT_DATA_TYPE", "Type", "C_DataType ", "ge::DataType "}},
    {"VT_TENSOR", {false, "VT_TENSOR", "Tensor", "EsCTensor *", "std::unique_ptr<ge::Tensor> "}},

    {"VT_LIST_INT", {true, "VT_LIST_INT", "ListInt", "const int64_t *", "const std::vector<int64_t> &"}},
    {"VT_LIST_FLOAT", {true, "VT_LIST_FLOAT", "ListFloat", "const float *", "const std::vector<float> &"}},
    {"VT_LIST_BOOL", {true, "VT_LIST_BOOL", "ListBool", "const bool *", "const std::vector<uint8_t> &"}},
    {"VT_LIST_DATA_TYPE", {true, "VT_LIST_DATA_TYPE", "ListType", "const C_DataType *", "const std::vector<ge::DataType> &"}},
    {"VT_LIST_LIST_INT", {true, "VT_LIST_LIST_INT", "ListListInt", "const int64_t **", "const std::vector<std::vector<int64_t>> &"}},
    {"VT_LIST_STRING", {true, "VT_LIST_STRING", "ListString", "const char **", "const std::vector<const char *> &"}},
};
template <typename T>
struct IsVectorHelper : std::false_type {};
template <typename T, typename Alloc>
struct IsVectorHelper<std::vector<T, Alloc>> : std::true_type {};

}  // namespace
const TypeInfo &GetTypeInfoByAvType(const std::string &av_type) {
  const auto iter = kAvToTypesInfo.find(av_type);
  if (iter == kAvToTypesInfo.end()) {
    throw NotSupportException("Unsupported attr type: " + av_type);
  }
  return iter->second;
}
std::string GetDefaultValueString(const OpDescPtr &op_desc, const std::string &attr_name, const std::string &av_type) {
  return AttrString::GetDefaultValueString(op_desc, attr_name, av_type);
}
bool IsOpSkip(const string &op_type) {
  if (OpTypeUtils::IsDataNode(op_type)) {
    return true;
  }
  if (op_type == kNetOutput) {
    return true;
  }
  return false;
}
bool IsOpExclude(const string &op_type, vector<std::string> &exlude_ops) {
  return std::find(exlude_ops.begin(), exlude_ops.end(), op_type) != exlude_ops.end();
}
bool IsOpSupport(const OpDescPtr &op, const char **reason) {
  static const std::unordered_set<std::string> unsupported_v1_control_op_names{
    "Switch", "StreamSwitch", "Merge", "StreamMerge", "Enter", "Exit", "LoopCond", "NextIteration"};
  static const std::unordered_set<std::string> already_provided_creation_functions{
    "Variable"};
  if (unsupported_v1_control_op_names.count(op->GetName()) > 0) {
    *reason = "Does not support V1 control Operators";
    return false;
  }
  if (already_provided_creation_functions.count(op->GetName()) > 0) {
    *reason = "Following Operators already provided creation functions";
    return false;
  }

  return true;
}
bool IsOpInputsAllOptional(const std::vector<std::pair<std::string, IrInputType>> &input_infos) {
  if (input_infos.empty()) {
    return false;
  }
  for (const auto &in : input_infos) {
    if (in.second == kIrInputRequired || in.second == kIrInputDynamic) {
      return false;
    }
  }
  return true;
}
void WriteOut(const char *file_path, const std::stringstream &ss) {
  if (file_path == nullptr) {
    std::cout << "File content: " << std::endl;
    std::cout << ss.str() << std::endl;
    std::cout << "=========================================================================" << std::endl;
  } else {
    std::cout << "Write to file: " << file_path << std::endl;
    std::ofstream file(file_path);
    file << ss.str();
    file.close();
  }
}

void GenCommentsIfNeeded(const OpDescPtr &op, std::stringstream &h_stream, bool support_tensor_like) {
  std::vector<std::string> subgraphs_name;
  for (const auto &ir_subgraph : op->GetOrderedSubgraphIrNames()) {
    subgraphs_name.emplace_back(ir_subgraph.first);
  }
  std::vector<std::string> tensor_attr_names;
  for (const auto &ir_attr: GetAllIrAttrsNamesAndTypeInOrder(op)) {
    if (std::strcmp(ir_attr.type_info.c_api_type, "EsCTensor *") == 0) {
      tensor_attr_names.emplace_back(ir_attr.name);
    }
  }
  std::vector<std::string> dynamic_output_names;
  for (const auto &ir_out: op->GetIrOutputs()) {
    if (ir_out.second == kIrOutputDynamic) {
      dynamic_output_names.emplace_back(ir_out.first);
    }
  }

  std::stringstream comment_stream;
  if (support_tensor_like) {
    GenTensorLikeInputComments(op, comment_stream);
  }
  if (!subgraphs_name.empty()) {
    comment_stream << " * @note lifecycles of following subgraph inputs will be transferred to the EsCGraphBuilder:"
                   << std::endl;
    for (const auto &subgraph_name : subgraphs_name) {
      comment_stream << " *   " << SubgraphName(subgraph_name) << std::endl;
    }
  }
  if (!tensor_attr_names.empty()) {
    comment_stream
        << " * @note lifecycles of following tensor attribute inputs will be transferred to the EsCGraphBuilder:"
        << std::endl;
    for (const auto &tensor_attr_name : tensor_attr_names) {
      comment_stream << " *   " << AttrName(tensor_attr_name, op) << std::endl;
    }
  }
  if (!dynamic_output_names.empty()) {
    comment_stream << " * @note user needs to provide following inputs for dynamic output numbers:" << std::endl;
    for (const auto &dynamic_output_name : dynamic_output_names) {
      comment_stream << " *   " << OutName(dynamic_output_name, op) << "_num: " << "dynamic output number of "
                     << dynamic_output_name << std::endl;
    }
  }

  if (!comment_stream.str().empty()) {
    h_stream << "/**" << std::endl;
    h_stream << comment_stream.str();
    h_stream << " */" << std::endl;
  }
}

void GenTensorLikeInputComments(const OpDescPtr &op, std::stringstream &h_stream) {
  for (const auto &in : op->GetIrInputs()) {
    if (in.second == kIrInputDynamic) {
      return;
    }
  }
  if (IsOpInputsAllOptional(op->GetIrInputs())) {
    h_stream << " * @note at least one of the following input arguments should be EsTensorHolder object"
                   << " or owner_builder should be provided:" << std::endl;
  } else {
    h_stream << " * @note at least one of the following input arguments should be EsTensorHolder object:"
                   << std::endl;
  }
  for (const auto &in : op->GetIrInputs()) {
    h_stream << " *   " << InName(in.first) << std::endl;
  }
}

std::string MakeGuardFromString(const string &input, const string &prefix, const string &suffix) {
  std::string guard = input;
  for (auto &ch : guard) {
    if (!(ch >= 'A' && ch <= 'Z') && !(ch >= 'a' && ch <= 'z') && !(ch >= '0' && ch <= '9')) {
      ch = '_';
    }
  }
  return prefix + guard + suffix;
}
std::string MakeGuardFromOp(const string &op_type, const std::string &external_defined_prefix, const string &suffix) {
  return MakeGuardFromString(op_type, kHeaderGuardNormalPrefix + external_defined_prefix + "_OP_", suffix);
}
std::string MakeGuardFromModule(const string &module_name, const std::string &external_defined_prefix,
                                const string &suffix) {
  return MakeGuardFromString(module_name, kHeaderGuardNormalPrefix + external_defined_prefix + "_", suffix);
}
}  // namespace es
}  // namespace ge