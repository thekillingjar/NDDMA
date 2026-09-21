/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_PY_GENERATOR_UTILS_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_PY_GENERATOR_UTILS_H_

#include "utils.h"
#include "cpp_generator.h"
#include "py_gen_constants.h"
#include "py_type_mapper.h"
#include <sstream>
#include <string>
#include <vector>

namespace ge {
namespace es {

// ==================== 名称生成工具 ====================

class NameGenerator {
 public:
  static std::string PyOutputStructName(const OpDescPtr &op) {
    return op->GetType() + PyGenConstants::kOutputSuffix;
  }

  static std::string CtypesStructName(const OpDescPtr &op) {
    return PyGenConstants::kCtypesPrefix + PyOutputStructName(op);
  }

  static std::string MakePerOpFileName(const std::string &op_type) {
    return PyGenConstants::kPerOpFilePrefix + op_type;
  }
};

// ==================== 子图处理工具 ====================

class SubgraphHandler {
 public:
  // ==================== 子图类型判断 ====================

  static bool HasSubgraph(const OpDescPtr &op) {
    return !op->GetOrderedSubgraphIrNames().empty();
  }

  static bool HasStaticSubgraph(const OpDescPtr &op) {
    const auto &subgraph_names = op->GetOrderedSubgraphIrNames();
    return std::any_of(subgraph_names.begin(), subgraph_names.end(),
                       [](const auto &pair) { return pair.second == kStatic; });
  }

  static bool HasDynamicSubgraph(const OpDescPtr &op) {
    const auto &subgraph_names = op->GetOrderedSubgraphIrNames();
    return std::any_of(subgraph_names.begin(), subgraph_names.end(),
                       [](const auto &pair) { return pair.second == kDynamic; });
  }

  // ==================== 子图参数处理 ====================

  /**
   * 为ctypes argtypes添加子图参数
   */
  static void AddSubgraphArgs(const OpDescPtr &op, std::vector<std::string> &arg_types) {
    const auto &subgraph_names = op->GetOrderedSubgraphIrNames();
    for (const auto &subgraph : subgraph_names) {
      if (subgraph.second == kStatic) {
        arg_types.emplace_back(PyGenConstants::kGraphPtrType);  // EsCGraph* 静态子图
      } else if (subgraph.second == kDynamic) {
        arg_types.emplace_back(PyGenConstants::kDynamicGraphPtrType);  // EsCGraph** 动态子图数组
        arg_types.emplace_back(PyGenConstants::kCtypesInt64);          // int64_t 动态子图个数
      }
    }
  }

  /**
   * 为Python函数参数添加子图参数
   */
  static void AddSubgraphParams(const OpDescPtr &op, std::stringstream &ss) {
    const auto &subgraph_names = op->GetOrderedSubgraphIrNames();
    for (const auto &subgraph : subgraph_names) {
      if (subgraph.second == kStatic) {
        ss << ", " << SubgraphName(subgraph.first) << ": " << PyGenConstants::kGraphType;
      } else if (subgraph.second == kDynamic) {
        ss << ", " << SubgraphName(subgraph.first) << ": List[" << PyGenConstants::kGraphType << "]";
      }
    }
  }

  /**
   * 为C函数调用添加子图参数
   */
  static void AddSubgraphCallArgs(const OpDescPtr &op, std::stringstream &ss) {
    const auto &subgraph_names = op->GetOrderedSubgraphIrNames();
    for (const auto &subgraph : subgraph_names) {
      if (subgraph.second == kStatic) {
        ss << ", ctypes.cast(" << SubgraphName(subgraph.first) << "._handle, " << PyGenConstants::kGraphPtrType << ")";
      } else if (subgraph.second == kDynamic) {
        ss << ", " << SubgraphName(subgraph.first) << "_array, len(" << SubgraphName(subgraph.first) << ")";
      }
    }
  }

  /**
   * 生成子图参数转换代码
   */
  static void GenSubgraphConversion(const OpDescPtr &op, std::stringstream &ss) {
    const auto &subgraph_names = op->GetOrderedSubgraphIrNames();
    for (const auto &subgraph : subgraph_names) {
      if (subgraph.second == kStatic) {
        // 静态子图：转移所有权并传递 owner_graph_builder
        ss << "    # 转移静态子图所有权到C++侧，并保持对owner的引用" << std::endl;
        ss << "    " << SubgraphName(subgraph.first) << "._transfer_ownership_when_pass_as_subgraph("
           << PyGenConstants::kOwnerGraphBuilder << ")" << std::endl;
        ss << std::endl;
      } else if (subgraph.second == kDynamic) {
        // 动态子图：先转移所有权（传递 owner），再转换
        ss << "    # 转移动态子图所有权到C++侧，并保持对owner的引用" << std::endl;
        ss << "    for _subgraph in " << SubgraphName(subgraph.first) << ":" << std::endl;
        ss << "        _subgraph._transfer_ownership_when_pass_as_subgraph(" << PyGenConstants::kOwnerGraphBuilder
           << ")" << std::endl;
        ss << std::endl;
        ss << "    # 转换动态子图句柄" << std::endl;
        ss << "    " << SubgraphName(subgraph.first) << "_handles = [ctypes.cast(g._handle, "
           << PyGenConstants::kGraphPtrType << ") for g in " << SubgraphName(subgraph.first) << "]" << std::endl;
        ss << "    " << SubgraphName(subgraph.first) << "_array = (" << PyGenConstants::kGraphPtrType << " * len("
           << SubgraphName(subgraph.first) << "))(*" << SubgraphName(subgraph.first) << "_handles)" << std::endl;
        ss << std::endl;
      }
    }
  }

  /**
   * 生成子图参数文档
   */
  static void GenSubgraphParameterDocs(const OpDescPtr &op, std::stringstream &ss) {
    const auto &subgraph_names = op->GetOrderedSubgraphIrNames();
    for (const auto &subgraph : subgraph_names) {
      if (subgraph.second == kStatic) {
        ss << "        " << SubgraphName(subgraph.first) << ": " << PyGenConstants::kGraphType << " object"
           << std::endl;
      } else if (subgraph.second == kDynamic) {
        ss << "        " << SubgraphName(subgraph.first) << ": List[" << PyGenConstants::kGraphType
           << "] - list of subgraphs" << std::endl;
      }
    }
  }

 private:
  static std::string SubgraphName(const std::string &ir_name) {
    return ir_name;
  }
};

// ==================== 输入处理工具 ====================

class InputHandler {
 public:
  static bool SupportTensorLike(const OpDescPtr &op) {
    return op->GetIrInputs().size() > 1 || IsOpInputsAllOptional(op->GetIrInputs());
  }

  static std::string GetTensorTypeUnion() {
    return "Union[" + std::string(PyGenConstants::kTensorHolderType) + ", " + PyGenConstants::kTensorLikeType + "]";
  }

  static std::string GetTensorLikeConvertExpr(const std::string &var_name) {
    std::ostringstream oss;
    oss << "convert_to_tensor_holder(" << var_name << ", " << PyGenConstants::kOwnerGraphBuilder << ")";
    return oss.str();
  }

  static void GenOwnerGraphBuilder(const std::pair<std::string, IrInputType> &input_info, std::stringstream &ss) {
    const std::string &input_name = InName(input_info.first);
    const IrInputType &input_type = input_info.second;

    if (input_type == kIrInputRequired || input_type == kIrInputOptional) {
      // 必需/可选输入，直接返回
      ss << input_name << ".get_owner_builder()";
      return;
    }
    if (input_type == kIrInputDynamic) {
      // 动态输入，使用列表的第一个元素
      ss << input_name << "[0].get_owner_builder()";
      return;
    }
    throw std::runtime_error("GetOwnerGraphBuilder: input type: " + std::to_string(input_type) + " is not valid");
  }

  static void GenOwnerGraphBuilder(const std::vector<std::pair<std::string, IrInputType>> &input_infos,
                                   std::stringstream &ss) {
    ss << "resolve_builder(";
    bool first = true;

    for (const auto &input : input_infos) {
      if (!first) {
        ss << ", ";
      }
      first = false;

      const std::string &input_name = InName(input.first);
      const IrInputType &input_type = input.second;

      if (input_type == kIrInputRequired || input_type == kIrInputOptional) {
        ss << input_name;
      } else if (input_type == kIrInputDynamic) {
        // 动态输入，如果列表不为空，展开列表；如果为空，展开 [None]
        ss << "*(" << input_name << " if " << input_name << " else [None])";
      } else {
        throw std::runtime_error("GetOwnerGraphBuilder: input type: " + std::to_string(input_type) + " is not valid");
      }
    }
    if (IsOpInputsAllOptional(input_infos)) {
      ss << ", "<< PyGenConstants::kOwnerBuilder;
    }
    ss << ")" << std::endl;
  }
};

// ==================== 动态输出处理工具 ====================

class DynamicOutputHandler {
 public:
  // ==================== 输出类型判断 ====================

  static bool HasDynamicOutput(const OpDescPtr &op) {
    return std::any_of(op->GetIrOutputs().begin(), op->GetIrOutputs().end(),
                       [](const auto &out) { return out.second == kIrOutputDynamic; });
  }

  static bool IsPureDynamicOutput(const OpDescPtr &op) {
    return std::all_of(op->GetIrOutputs().begin(), op->GetIrOutputs().end(),
                       [](const auto &out) { return out.second == kIrOutputDynamic; });
  }

  static bool ShouldReturnList(const OpDescPtr &op) {
    return IsPureDynamicOutput(op) && op->GetIrOutputs().size() == 1U;
  }

  // ==================== 结构体定义生成 ====================

  static std::string GetCtypesStructName(const OpDescPtr &op) {
    return NameGenerator::CtypesStructName(op);
  }

  static std::string GetStructInitParams(const OpDescPtr &op) {
    std::stringstream params;
    const auto &outputs = op->GetIrOutputs();
    bool first = true;

    for (const auto &out : outputs) {
      if (!first) params << ", ";
      first = false;

      if (out.second == kIrOutputDynamic) {
        params << OutName(out.first, op) << ": List[" << PyGenConstants::kTensorHolderType << "]";
      } else {
        params << OutName(out.first, op) << ": " << PyGenConstants::kTensorHolderType;
      }
    }

    return params.str();
  }

  // ==================== 动态输出个数参数处理 ====================

  /**
   * 为ctypes argtypes添加动态输出个数参数
   */
  static void AddDynamicOutputCountArgs(const OpDescPtr &op, std::vector<std::string> &arg_types) {
    for (const auto &out : op->GetIrOutputs()) {
      if (out.second == kIrOutputDynamic) {
        arg_types.emplace_back(PyGenConstants::kCtypesInt64);  // 动态输出个数参数
      }
    }
  }

  /**
   * 为Python函数参数添加动态输出个数参数
   */
  static void AddDynamicOutputCountParams(const OpDescPtr &op, std::stringstream &ss) {
    for (const auto &out : op->GetIrOutputs()) {
      if (out.second == kIrOutputDynamic) {
        ss << ", " << OutName(out.first, op) << "_num: int";
      }
    }
  }

  /**
   * 为C函数调用添加动态输出个数参数
   */
  static void AddDynamicOutputCountCallArgs(const OpDescPtr &op, std::stringstream &ss) {
    for (const auto &out : op->GetIrOutputs()) {
      if (out.second == kIrOutputDynamic) {
        ss << ", " << OutName(out.first, op) << "_num";
      }
    }
  }

  // ==================== 返回处理代码生成 ====================

  static void GenReturnCode(const OpDescPtr &op, std::stringstream &ss) {
    ss << "    # 处理动态输出结果" << std::endl;

    if (ShouldReturnList(op)) {
      GenSingleDynamicOutputList(op, ss);
    } else {
      GenDynamicOutputStruct(op, ss);
    }
  }

  static void GenSingleDynamicOutputList(const OpDescPtr &op, std::stringstream &ss) {
    auto dynamic_out = op->GetIrOutputs().begin();
    ss << "    # 创建动态输出列表" << std::endl;
    GenDynamicOutputListCode(op, dynamic_out->first, ss);
    ss << std::endl;

    GenScopeInfoApplication(op, dynamic_out->first, true, ss);
    ss << "    return " << OutName(dynamic_out->first, op) << "_list" << std::endl;
  }

  static void GenDynamicOutputStruct(const OpDescPtr &op, std::stringstream &ss) {
    GenOutputObjects(op, ss);
    GenScopeInfoApplicationForFirstOutput(op, ss);
    GenDynamicOutputStructReturn(op, ss);
  }

 private:
  // ==================== 辅助函数 ====================

  static void GenDynamicOutputListCode(const OpDescPtr &op, const std::string &output_name, std::stringstream &ss) {
    ss << "    " << OutName(output_name, op) << "_list = [" << PyGenConstants::kTensorHolderType
       << "._create_from(_result." << OutName(output_name, op) << "[i], " << PyGenConstants::kOwnerGraphBuilder
       << ") for i in range(_result." << OutName(output_name, op) << "_num)]";
  }

  static void GenOutputObjects(const OpDescPtr &op, std::stringstream &ss) {
    for (const auto &out : op->GetIrOutputs()) {
      if (out.second == kIrOutputDynamic) {
        ss << "    # 创建动态输出列表: " << out.first << std::endl;
        GenDynamicOutputListCode(op, out.first, ss);
        ss << std::endl;
      } else {
        ss << "    " << OutName(out.first, op) << " = " << PyGenConstants::kTensorHolderType << "._create_from(_result."
           << OutName(out.first, op) << ", " << PyGenConstants::kOwnerGraphBuilder << ")" << std::endl;
      }
    }
    ss << std::endl;
  }

  static void GenScopeInfoApplicationForFirstOutput(const OpDescPtr &op, std::stringstream &ss) {
    if (op->GetIrOutputs().empty()) return;

    auto first_output = op->GetIrOutputs().begin();
    GenScopeInfoApplication(op, first_output->first, first_output->second == kIrOutputDynamic, ss);
  }

  static void GenScopeInfoApplication(const OpDescPtr &op, const std::string &output_name, bool is_dynamic,
                                      std::stringstream &ss) {
    ss << "    # 尝试获取scope内信息设置到节点上" << std::endl;
    if (is_dynamic) {
      ss << "    if " << OutName(output_name, op) << "_list:" << std::endl;
      ss << "        " << PyGenConstants::kOwnerGraphBuilder << "._apply_scope_infos_to_node("
         << OutName(output_name, op) << "_list[0])" << std::endl;
    } else {
      ss << "    " << PyGenConstants::kOwnerGraphBuilder << "._apply_scope_infos_to_node(" << OutName(output_name, op)
         << ")" << std::endl;
    }
    ss << std::endl;
  }

  static void GenDynamicOutputStructReturn(const OpDescPtr &op, std::stringstream &ss) {
    ss << "    return " << NameGenerator::PyOutputStructName(op) << "(" << std::endl;
    for (const auto &out : op->GetIrOutputs()) {
      if (out.second == kIrOutputDynamic) {
        ss << "        " << OutName(out.first, op) << "=" << OutName(out.first, op) << "_list," << std::endl;
      } else {
        ss << "        " << OutName(out.first, op) << "=" << OutName(out.first, op) << "," << std::endl;
      }
    }
    ss << "    )" << std::endl;
  }
};

// ==================== 输出类型处理工具 ====================

class OutputTypeHandler {
 public:
  static bool IsSingleOutput(OutputType output_type) {
    return output_type == OutputType::kNoOutput || output_type == OutputType::kOneOutput;
  }

  static bool IsMultiOutput(OutputType output_type) {
    return output_type == OutputType::kMultiOutput;
  }

  static bool IsDynamicOutput(const OpDescPtr &op) {
    return DynamicOutputHandler::HasDynamicOutput(op);
  }

  static bool IsPureDynamicOutput(const OpDescPtr &op) {
    return DynamicOutputHandler::IsPureDynamicOutput(op);
  }

  static std::string GetCtypesReturnType(const OpDescPtr &op) {
    if (IsDynamicOutput(op)) {
      return DynamicOutputHandler::GetCtypesStructName(op) + "  # " + CGenerator::OutputStructName(op);
    }

    switch (GetOutputType(op)) {
      case OutputType::kNoOutput:
      case OutputType::kOneOutput:
        return PyGenConstants::kTensorHolderPtrType + std::string("  # EsCTensorHolder指针");
      case OutputType::kMultiOutput:
        return NameGenerator::CtypesStructName(op) + "  # " + CGenerator::OutputStructName(op);
      default:
        return "";
    }
  }

  static std::string GetReturnTypeAnnotation(const OpDescPtr &op) {
    if (IsDynamicOutput(op)) {
      if (IsPureDynamicOutput(op)) {
        // 纯动态输出，只有一个动态输出时返回List[TensorHolder]
        const auto &outputs = op->GetIrOutputs();
        if (outputs.size() == 1) {
          return " -> List[" + std::string(PyGenConstants::kTensorHolderType) + "]";
        }
      }
      // 多个动态输出或混合输出，返回自定义结构体
      return " -> " + NameGenerator::PyOutputStructName(op);
    }

    switch (GetOutputType(op)) {
      case OutputType::kNoOutput:
      case OutputType::kOneOutput:
        return " -> " + std::string(PyGenConstants::kTensorHolderType);
      case OutputType::kMultiOutput:
        return " -> " + NameGenerator::PyOutputStructName(op);
      default:
        return "";
    }
  }

  static std::string GetReturnDescription(const OpDescPtr &op) {
    if (IsDynamicOutput(op)) {
      if (IsPureDynamicOutput(op)) {
        // 纯动态输出
        const auto &outputs = op->GetIrOutputs();
        if (outputs.size() == 1) {
          return "        List of TensorHolder objects";
        }
      }
      // 混合输出或复杂动态输出
      return "        " + NameGenerator::PyOutputStructName(op) + " object containing dynamic and/or multiple outputs";
    }

    switch (GetOutputType(op)) {
      case OutputType::kNoOutput:
        return "        New TensorHolder object(does not produce an output, so the returned cannot be used to "
               "connect edges)";
      case OutputType::kOneOutput:
        return "        New TensorHolder object";
      case OutputType::kMultiOutput:
        return "        " + NameGenerator::PyOutputStructName(op) + " object containing multiple outputs";
      default:
        return "";
    }
  }
};

// ==================== 字符串处理工具 ====================

class StringConverter {
 public:
  static std::string ConvertCppDefaultToPython(const std::string &default_val, const std::string &av_type) {
    if (av_type == "VT_STRING") {
      return ConvertStringToPython(default_val);
    } else if (av_type == "VT_BOOL") {
      return (default_val == PyGenConstants::kTrueValue) ? PyGenConstants::kPythonTrue : PyGenConstants::kPythonFalse;
    } else if (av_type == "VT_DATA_TYPE") {
      return PyGenConstants::kDataTypePrefix + default_val.substr(PyGenConstants::kGePrefixLength);
    } else if (av_type == "VT_LIST_INT" || av_type == "VT_LIST_FLOAT") {
      return ConvertCppListToPython(default_val);
    } else if (av_type == "VT_LIST_BOOL") {
      return ConvertCppBoolListToPython(default_val);
    } else if (av_type == "VT_LIST_LIST_INT"){
      return ConvertCppListListIntToPython(default_val);
    } else if (av_type == "VT_LIST_DATA_TYPE"){
      return ConvertCppListDataTypeToPython(default_val);
    } else if (av_type == "VT_LIST_STRING"){
      return ConvertCppListStringToPython(default_val);
    } else if (av_type == "VT_TENSOR"){
      return ConvertCppTensorToPython();
    }
    return default_val;
  }

 private:
   static std::string ConvertCppTensorToPython() {
    return "Tensor()";
  }

  static std::string ConvertCppListListIntToPython(const std::string &default_val) {
    if (default_val.empty() || default_val == "{}") {
      return PyGenConstants::kEmptyList;
    }
    if (default_val.front() == '{' && default_val.back() == '}') {
        std::string content = default_val.substr(1U, default_val.length() - 2U);
        ReplaceString(content, PyGenConstants::kLeftCurlyBracket, PyGenConstants::kLeftBracket);
        ReplaceString(content, PyGenConstants::kRightCurlyBracket, PyGenConstants::kRightBracket);
        return "[" + content + "]";
    }
    return default_val;
  }

  static std::string ConvertCppListDataTypeToPython(const std::string &default_val) {
        if (default_val.empty() || default_val == "{}") {
      return PyGenConstants::kEmptyList;
    }
    if (default_val.front() == '{' && default_val.back() == '}') {
      std::string content = default_val.substr(1U, default_val.length() - 2U);
      ReplaceString(content, PyGenConstants::kGePrefix, PyGenConstants::kDataTypePrefix);
      return "[" + content + "]";
    }
    return default_val;
  }
  
  static std::string ConvertCppListStringToPython(const std::string &default_val) {
    if (default_val.empty() || default_val == "{}") {
      return PyGenConstants::kEmptyList;
    }
    if (default_val.front() == '{' && default_val.back() == '}') {
      std::string content = default_val.substr(1U, default_val.length() - 2U);
      return "[" + content + "]";
    }
    return default_val;
  }

  static std::string ConvertStringToPython(const std::string &default_val) {
    // 如果已经有双引号且引号内字符合法
    if (default_val.length() >= 2U && default_val.front() == '"' && default_val.back() == '"') {
      return default_val;
    }
    // 否则添加双引号
    return "\"" + default_val + "\"";
  }

  static std::string ConvertCppListToPython(const std::string &default_val) {
    if (default_val.empty() || default_val == "{}") {
      return PyGenConstants::kEmptyList;
    }
    if (default_val.front() == '{' && default_val.back() == '}') {
      return "[" + default_val.substr(1U, default_val.length() - 2U) + "]";
    }
    return default_val;
  }

  static std::string ConvertCppBoolListToPython(const std::string &default_val) {
    if (default_val.empty() || default_val == "{}") {
      return PyGenConstants::kEmptyList;
    }
    if (default_val.front() == '{' && default_val.back() == '}') {
      std::string content = default_val.substr(1U, default_val.length() - 2U);
      return "[" + ConvertBoolValuesInString(content) + "]";
    }
    return default_val;
  }

  static std::string ConvertBoolValuesInString(const std::string &content) {
    std::string result;
    size_t pos = 0;
    while (pos < content.length()) {
      // 跳过空格和逗号
      while (pos < content.length() && (content[pos] == ' ' || content[pos] == ',')) {
        if (content[pos] == ',') {
          result += ", ";
        }
        pos++;
      }
      if (pos >= content.length()) break;

      // 检查是否是true或false
      if (content.substr(pos, PyGenConstants::kTrueLength) == PyGenConstants::kTrueValue) {
        result += PyGenConstants::kPythonTrue;
        pos += PyGenConstants::kTrueLength;
      } else if (content.substr(pos, PyGenConstants::kFalseLength) == PyGenConstants::kFalseValue) {
        result += PyGenConstants::kPythonFalse;
        pos += PyGenConstants::kFalseLength;
      } else {
        // 其他字符直接添加
        result += content[pos];
        pos++;
      }
    }
    return result;
  }

  static void ReplaceString(std::string& str, const char *from, const char *to) {
    // 避免空指针或空字符串
    if (from == nullptr || *from == '\0') {
      return;
    }
    std::string from_str(from);
    std::string to_str(to);

    size_t pos = 0;
    while ((pos = str.find(from_str, pos)) != std::string::npos) {
        str.replace(pos, from_str.length(), to_str);
        pos += to_str.length(); // 跳过刚替换的部分，避免死循环
    }
  }
};
}  // namespace es
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_PY_GENERATOR_UTILS_H_
