/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_PY_GENERATOR_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_PY_GENERATOR_H_

#include "utils.h"
#include "generator_interface.h"
#include "py_generator_utils.h"
#include "py_type_mapper.h"
#include <sstream>
#include <unordered_map>
#include <string>
#include <vector>

namespace ge {
namespace es {
/**
 * ES算子的Python代码生成器
 * 使用ctypes调用C实现生成Python包装函数
 */
class PyGenerator : public ICodeGenerator {
 public:
  void GenOp(const OpDescPtr &op) override {
    auto ss = GenOpImpl(op, module_name_);
    per_op_py_sources_[op->GetType()] = std::move(ss);
  }

  void ResetWhenGenFailed(const OpDescPtr &op) override {
    per_op_py_sources_.erase(op->GetType());
  }

  std::stringstream &GetAggregateContentStream() override {
    return aggregatess_;
  }

  std::string GetGeneratorName() const override {
    return PyGenConstants::kPythonGeneratorName;
  }

  std::string GetAggregateFileName() const override {
    return PyGenConstants::kPerOpFilePrefix + module_name_ + "_ops" + PyGenConstants::kPerOpFileSuffix;
  }

  std::string GetCommentStart() const override {
    return PyGenConstants::kCommentStart;
  }

  std::string GetCommentLinePrefix() const override {
    return PyGenConstants::kCommentLinePrefix;
  }

  std::string GetCommentEnd() const override {
    return PyGenConstants::kCommentEnd;
  }

  void GenAggregateHeader() override {
    GenHeader(aggregatess_);
    GenCopyright(aggregatess_, true);
  }

  void GenAggregateIncludes(const std::vector<std::string> &op_types) override {
    for (const auto &op_type : op_types) {
      aggregatess_ << "from ." << NameGenerator::MakePerOpFileName(op_type) << " import *" << std::endl;
    }
  }

  const std::unordered_map<std::string, std::stringstream> &GetPerOpContents() const override {
    return per_op_py_sources_;
  }

  void GenPerOpFiles(const std::string &output_dir, const std::vector<std::string> &op_types) override {
    WritePerOpFiles(output_dir, op_types, GetPerOpContents(),
                    [this](const std::string &op_type) { return GetPerOpFileName(op_type); });
  }

  void SetModuleName(const std::string &module_name) {
    module_name_ = module_name;
  }

  std::string GetPerOpFileName(const std::string &op_type) const override {
    return NameGenerator::MakePerOpFileName(op_type) + PyGenConstants::kPerOpFileSuffix;
  }

 private:
  static std::stringstream GenOpImpl(const OpDescPtr &op, const std::string &module_name) {
    std::stringstream ss;
    GenHeader(ss);
    GenCopyright(ss, true);
    GenImports(ss, module_name);
    GenMultiOutputStructuresIfNeeded(op, ss);
    GenFunctionPrototype(op, ss);
    GenPythonWrapper(op, ss);
    return ss;
  }

  // ==================== 文件结构生成 ====================

  static void GenHeader(std::stringstream &ss) {
    ss << R"(#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
)" << std::endl;
  }

  static void GenImports(std::stringstream &ss, const std::string &module_name) {
    ss << R"(import ctypes
import os
from typing import List, Optional, Union
try:
    from ge.es import GraphBuilder, TensorHolder
    from ge.es.tensor_like import TensorLike, convert_to_tensor_holder, resolve_builder
    from ge.graph.types import DataType, Format
    from ge.graph.graph import Graph
    from ge.graph.tensor import Tensor
    from ge._capi.pyes_graph_builder_wrapper import (
        get_generated_lib,
        EsCTensorHolderPtr,
        EsCGraphBuilderPtr,
        EsCTensorPtr,
        EsCGraph
    )
except ImportError as e:
    raise ImportError(f"Unable to import ge module: {e}")

# 获取生成产物C实现的so句柄
)";
    ss << "try:" << std::endl;
    ss << "    esb_generated_lib = get_generated_lib(\"lib" << PyGenConstants::kPerOpFilePrefix << module_name << ".so\")" << std::endl;
    ss << "except RuntimeError:" << std::endl;
    ss << "    esb_generated_lib = get_generated_lib(\"lib" << module_name << ".so\")" << std::endl;
  }

  // ==================== 多输出结构体生成 ====================

  static void GenMultiOutputStructuresIfNeeded(const OpDescPtr &op, std::stringstream &ss) {
    // 支持多输出和动态输出
    if ((GetOutputType(op) == OutputType::kMultiOutput) || OutputTypeHandler::IsDynamicOutput(op)) {
      GenCtypesStructDefinition(op, ss);
      // 只有一个动态输出，不需要定义py的返回结构体
      if (!DynamicOutputHandler::ShouldReturnList(op)) {
        GenOutputStructDefinition(op, ss);
      }
    }
  }

  static void GenCtypesStructDefinition(const OpDescPtr &op, std::stringstream &ss) {
    ss << std::endl << "# 为 " << op->GetType() << " 定义ctypes结构体" << std::endl;
    ss << "class " << NameGenerator::CtypesStructName(op) << "(ctypes.Structure):" << std::endl;
    ss << "    " << PyGenConstants::kCommentStart << "ctypes structure corresponding to "
       << CGenerator::OutputStructName(op) << PyGenConstants::kCommentEnd << std::endl;
    ss << "    _fields_ = [" << std::endl;

    auto &ir_outputs = op->GetIrOutputs();
    for (const auto &out : ir_outputs) {
      if (out.second == kIrOutputDynamic) {
        ss << "        (\"" << OutName(out.first, op, GenLanType::GenPy) << "\", ctypes.POINTER(" << PyGenConstants::kTensorHolderPtrType
           << ")),  # EsCTensorHolder指针数组" << std::endl;
        ss << "        (\"" << OutName(out.first, op, GenLanType::GenPy) << "_num\", " << PyGenConstants::kCtypesInt64
           << "),  # 动态输出数量" << std::endl;
      } else {
        ss << "        (\"" << OutName(out.first, op, GenLanType::GenPy) << "\", " << PyGenConstants::kTensorHolderPtrType
           << "),  # EsCTensorHolder指针" << std::endl;
      }
    }

    ss << "    ]" << std::endl << std::endl;
  }

  static void GenOutputStructDefinition(const OpDescPtr &op, std::stringstream &ss) {
    ss << "# 为 " << op->GetType() << " 定义输出结构体" << std::endl;
    ss << "class " << NameGenerator::PyOutputStructName(op) << ":" << std::endl;
    ss << "    " << PyGenConstants::kCommentStart << "Output structure for " << op->GetType() << " operator"
       << PyGenConstants::kCommentEnd << std::endl;

    if (OutputTypeHandler::IsDynamicOutput(op)) {
      ss << "    def __init__(self, " << DynamicOutputHandler::GetStructInitParams(op) << "):" << std::endl;
    } else {
      ss << "    def __init__(self, " << GetOutputStructInitParams(op) << "):" << std::endl;
    }

    auto &ir_outputs = op->GetIrOutputs();
    for (const auto &out : ir_outputs) {
      ss << "        self." << OutName(out.first, op, GenLanType::GenPy) << " = " << OutName(out.first, op, GenLanType::GenPy) << std::endl;
    }

    ss << std::endl;
  }

  static std::string GetOutputStructInitParams(const OpDescPtr &op) {
    std::stringstream params;
    auto &ir_outputs = op->GetIrOutputs();
    bool first = true;

    for (const auto &out : ir_outputs) {
      if (!first) params << ", ";
      first = false;
      params << OutName(out.first, op, GenLanType::GenPy) << ": " << PyGenConstants::kTensorHolderType;
    }

    return params.str();
  }

  // ==================== 函数原型生成 ====================

  static void GenFunctionPrototype(const OpDescPtr &op, std::stringstream &ss) {
    ss << std::endl << "# 定义 " << op->GetType() << " 函数原型" << std::endl;
    GenArgTypes(op, ss);
    GenReturnType(op, ss);
  }

  static void GenArgTypes(const OpDescPtr &op, std::stringstream &ss) {
    ss << "esb_generated_lib." << CGenerator::FuncName(op->GetType()) << ".argtypes = [";

    auto arg_types = CollectArgTypes(op);
    for (size_t i = 0; i < arg_types.size(); ++i) {
      if (i > 0) ss << ", ";
      ss << arg_types[i];
    }

    ss << "]" << std::endl;
  }

  static void GenReturnType(const OpDescPtr &op, std::stringstream &ss) {
    ss << "esb_generated_lib." << CGenerator::FuncName(op->GetType()) << ".restype = ";
    ss << OutputTypeHandler::GetCtypesReturnType(op) << std::endl;
  }

  static std::vector<std::string> CollectArgTypes(const OpDescPtr &op) {
    std::vector<std::string> arg_types;

    // 收集输入参数类型
    for (const auto &in : op->GetIrInputs()) {
      if (in.second == kIrInputRequired || in.second == kIrInputOptional) {
        arg_types.emplace_back(PyGenConstants::kTensorHolderPtrType);  // EsCTensorHolder指针
      } else {
        arg_types.emplace_back("ctypes.POINTER(" + std::string(PyGenConstants::kTensorHolderPtrType) + "), " +
                               std::string(PyGenConstants::kCtypesInt64));  // EsCTensorHolder指针数组, int64_t
      }
    }

    // 添加owner_graph_builder
    if (op->GetIrInputs().empty() || IsOpInputsAllOptional(op->GetIrInputs())) {
      arg_types.emplace_back(PyGenConstants::kGraphBuilderPtrType);  // EsCGraphBuilder指针
    }

    // 添加动态输出个数参数
    DynamicOutputHandler::AddDynamicOutputCountArgs(op, arg_types);

    // 添加子图参数
    SubgraphHandler::AddSubgraphArgs(op, arg_types);

    // 收集属性参数类型
    auto ir_and_dts = GetAllIrAttrsNamesAndTypeInOrder(op);
    for (const auto &attr : ir_and_dts) {
      if (attr.type_info.is_list_type) {
        // 列表类型使用POINTER包装
        std::string base_type = PythonTypeMapper::GetCtypesType(attr.type_info);
        if (strcmp(attr.type_info.av_type, "VT_LIST_LIST_INT") == 0) {
          arg_types.push_back("ctypes.POINTER(ctypes.POINTER(" + base_type + "))");
          arg_types.emplace_back(PyGenConstants::kCtypesInt64);  // 列表类型需要额外的计数参数
          arg_types.push_back("ctypes.POINTER(" + base_type + ")");
        } else {
          arg_types.push_back("ctypes.POINTER(" + base_type + ")");
          arg_types.emplace_back(PyGenConstants::kCtypesInt64);  // 列表类型需要额外的计数参数
        }
      } else if(std::strcmp(attr.type_info.c_api_type, "EsCTensor *") == 0) {
        // EsCTensor
        arg_types.emplace_back(PyGenConstants::kEsCTensorPtrType);
      } else {
        arg_types.push_back(PythonTypeMapper::GetCtypesType(attr.type_info));
      }
    }

    return arg_types;
  }

  // ==================== Python包装器生成 ====================

  static void GenPythonWrapper(const OpDescPtr &op, std::stringstream &ss) {
    ss << std::endl << "# 创建Python包装函数" << std::endl;
    GenFunctionSignature(op, ss);
    GenFunctionDocstring(op, ss);
    GenFunctionBody(op, ss);
  }

  static void GenFunctionSignature(const OpDescPtr &op, std::stringstream &ss) {
    ss << "def " << op->GetType() << "(";
    GenFunctionParameters(op, ss);
    ss << ")";
    ss << OutputTypeHandler::GetReturnTypeAnnotation(op);
    ss << ":" << std::endl;
  }

  static void GenFunctionParameters(const OpDescPtr &op, std::stringstream &ss) {
    const auto dv = PythonValueHelper(op);
    bool first = true;
    const std::string type_annotation = InputHandler::SupportTensorLike(op)
                                        ? InputHandler::GetTensorTypeUnion()
                                        : PyGenConstants::kTensorHolderType;

    // 处理输入参数
    for (const auto &in : op->GetIrInputs()) {
      if (!first) ss << ", ";
      first = false;

      if (in.second == kIrInputRequired || in.second == kIrInputOptional) {
        if (dv.IsInputHasDefault(in.first)) {
          ss << InName(in.first, GenLanType::GenPy) << ": Optional[" << type_annotation << "]";
          ss << " = " << PyGenConstants::kNoneValue;
        } else {
          ss << InName(in.first, GenLanType::GenPy) << ": " << type_annotation;
        }
      } else {
        ss << InName(in.first, GenLanType::GenPy) << ": List[" << type_annotation
           << "]";  // 兼容低于3.9版本的python使用typeing.List而非list
      }
    }

    // 如果没有输入则添加owner_graph
    if (first) {
      ss << PyGenConstants::kOwnerGraphBuilder << ": " << PyGenConstants::kGraphBuilderType;
    } else if (IsOpInputsAllOptional(op->GetIrInputs())) {  // 如果都是可选输入则添加owner_builder可选参数
      if (dv.IsAnyInputHasDefault()) {
        ss << ", " << PyGenConstants::kOwnerBuilder << ": Optional[" << PyGenConstants::kGraphBuilderType << "] = "
           << PyGenConstants::kNoneValue;
      } else {
        ss << ", " << PyGenConstants::kOwnerBuilder << ": " << PyGenConstants::kGraphBuilderType;
      }
    }

    // 添加动态输出个数参数
    DynamicOutputHandler::AddDynamicOutputCountParams(op, ss);

    // 添加子图参数
    SubgraphHandler::AddSubgraphParams(op, ss);

    // 在属性参数之前添加 *, 使属性成为关键字参数
    auto ir_and_dts = GetAllIrAttrsNamesAndTypeInOrder(op);
    if (!ir_and_dts.empty()) {
      ss << ", *";
    }

    // 处理属性参数
    for (const auto &attr : ir_and_dts) {
      ss << ", ";
      ss << AttrName(attr.name, op, GenLanType::GenPy) << ": " << PythonTypeMapper::GetTypeAnnotation(attr.type_info);
      if (!attr.is_required) {
        ss << " = " << GetPythonDefaultValue(op, attr.name, attr.type_info.av_type);
      }
    }
  }

  static void GenFunctionDocstring(const OpDescPtr &op, std::stringstream &ss) {
    ss << R"(    """
    Python wrapper for )"
       << CGenerator::FuncName(op->GetType()) << R"( function
    Args:)"
       << std::endl;

    GenInputParameterDocs(op, ss);
    GenDynamicOutputParameterDocs(op, ss);
    GenSubgraphParameterDocs(op, ss);
    GenAttributeParameterDocs(op, ss);
    GenReturnParameterDocs(op, ss);
    GenCommentsIfNeeded(op, ss);

    ss << "    " << PyGenConstants::kCommentEnd;
  }

  static void GenInputParameterDocs(const OpDescPtr &op, std::stringstream &ss) {
    for (const auto &in : op->GetIrInputs()) {
      ss << "        " << InName(in.first, GenLanType::GenPy) << ": " << PyGenConstants::kTensorHolderType;
      if (InputHandler::SupportTensorLike(op)) {
        ss << " or " << PyGenConstants::kTensorLikeType;
      }
      ss << " object";
      if (in.second == kIrInputOptional) {
        ss << " (optional)";
      } else if (in.second == kIrInputDynamic) {
        ss << " list";
      }
      ss << std::endl;
    }

    if (op->GetIrInputs().empty()) {
      ss << "        " << PyGenConstants::kOwnerGraphBuilder << ": " << PyGenConstants::kGraphBuilderType << " object"
         << std::endl;
    } else if (IsOpInputsAllOptional(op->GetIrInputs())) {
      ss << "        " << PyGenConstants::kOwnerBuilder << ": " << PyGenConstants::kGraphBuilderType << " (optional)"
         << std::endl;
    }
  }

  static void GenDynamicOutputParameterDocs(const OpDescPtr &op, std::stringstream &ss) {
    for (const auto &out : op->GetIrOutputs()) {
      if (out.second == kIrOutputDynamic) {
        ss << "        " << OutName(out.first, op, GenLanType::GenPy) << "_num: int - number of dynamic outputs" << std::endl;
      }
    }
  }

  static void GenSubgraphParameterDocs(const OpDescPtr &op, std::stringstream &ss) {
    SubgraphHandler::GenSubgraphParameterDocs(op, ss);
  }

  static void GenAttributeParameterDocs(const OpDescPtr &op, std::stringstream &ss) {
    auto ir_and_dts = GetAllIrAttrsNamesAndTypeInOrder(op);
    for (const auto &attr : ir_and_dts) {
      ss << "        " << AttrName(attr.name, op, GenLanType::GenPy) << ": " << PythonTypeMapper::GetTypeDescription(attr.type_info);
      if (!attr.is_required) {
        ss << " (optional)";
      }
      ss << std::endl;
    }
  }

  static void GenReturnParameterDocs(const OpDescPtr &op, std::stringstream &ss) {
    ss << "    Returns:" << std::endl;
    ss << OutputTypeHandler::GetReturnDescription(op) << std::endl;
  }

  static void GenCommentsIfNeeded(const OpDescPtr &op, std::stringstream &ss) {
    if (!InputHandler::SupportTensorLike(op)) {
      return;
    }

    if (IsOpInputsAllOptional(op->GetIrInputs())) {
      ss << "    Note: at least one of the following input arguments should be TensorHolder object or "
         << PyGenConstants::kOwnerBuilder << " should be provided:" << std::endl;
    } else {
      ss << "    Note: at least one of the following input arguments should be TensorHolder object:" << std::endl;
    }
    for (const auto &input : op->GetIrInputs()) {
      ss << "          " << InName(input.first, GenLanType::GenPy) << std::endl;
    }
  }

  static void GenFunctionBody(const OpDescPtr &op, std::stringstream &ss) {
    ss << std::endl;
    GenOwnerGraphBuilderExtraction(op, ss);
    GenTensorLikeInputNormalization(op, ss);
    GenDynamicInputConversion(op, ss);
    GenSubgraphConversion(op, ss);
    GenTensorAttrConversion(op, ss);
    GenListStringAttrConversion(op, ss);
    GenListListIntAttrConversion(op, ss);
    GenFunctionCall(op, ss);
    GenReturnHandling(op, ss);
  }

  static void GenOwnerGraphBuilderExtraction(const OpDescPtr &op, std::stringstream &ss) {
    const auto &input_infos = op->GetIrInputs();
    if (input_infos.empty()) {
      // 如果没有输入，owner_graph_builder已经作为参数传入，不需要额外处理
      return;
    }
    // 如果有输入，按顺序找到第一个可以获取到GraphBuilder的输入
    ss << "    # 获取owner_graph_builder" << std::endl;
    ss << "    " << PyGenConstants::kOwnerGraphBuilder << " = ";
    if (InputHandler::SupportTensorLike(op)) {
      // 多个入参时，使用 resolve_builder 从所有入参中获取到GraphBuilder
      InputHandler::GenOwnerGraphBuilder(input_infos, ss);
    } else {
      // 单个入参时，直接返回
      InputHandler::GenOwnerGraphBuilder(input_infos[0], ss);
    }
    ss << std::endl;
  }

  static void GenTensorLikeInputNormalization(const OpDescPtr &op, std::stringstream &ss) {
    if (!InputHandler::SupportTensorLike(op)) {
      return;
    }

    ss << "    # 将TensorLike输入转换为TensorHolder" << std::endl;
    for (const auto &input : op->GetIrInputs()) {
      const std::string input_name = InName(input.first, GenLanType::GenPy);
      const IrInputType input_type = input.second;

      if (input_type == kIrInputRequired || input_type == kIrInputOptional) {
        ss << "    " << input_name << " = " << InputHandler::GetTensorLikeConvertExpr(input_name) << std::endl;
      } else {
        ss << "    " << input_name << " = [" << InputHandler::GetTensorLikeConvertExpr("t") << " for t in "
           << input_name << "]" << std::endl;
      }
    }
    ss << std::endl;
  }

  static void GenDynamicInputConversion(const OpDescPtr &op, std::stringstream &ss) {
    for (const auto &in : op->GetIrInputs()) {
      if (in.second == kIrInputDynamic) {
        ss << "    # 转换动态输入" << std::endl;
        ss << "    " << InName(in.first, GenLanType::GenPy) << "_ptrs = [t._handle for t in " << InName(in.first, GenLanType::GenPy) << "]" << std::endl;
        ss << "    " << InName(in.first, GenLanType::GenPy) << "_array = (" << PyGenConstants::kTensorHolderPtrType << " * len("
           << InName(in.first, GenLanType::GenPy) << "))(*" << InName(in.first, GenLanType::GenPy) << "_ptrs)" << std::endl;
        ss << std::endl;
      }
    }
  }

  static void GenSubgraphConversion(const OpDescPtr &op, std::stringstream &ss) {
    SubgraphHandler::GenSubgraphConversion(op, ss);
  }

  static void GenTensorAttrConversion(const OpDescPtr &op, std::stringstream &ss) {
     // 函数内部需要完成转属性Tensor所有权到C++侧owner, 并保持对python侧owner的引用
    for (const auto &ir_attr: GetAllIrAttrsNamesAndTypeInOrder(op)) {
      if (std::strcmp(ir_attr.type_info.c_api_type, "EsCTensor *") == 0) {
          ss << "    # 转移Tensor所有权到C++侧，并保持对owner的引用" << std::endl;
          ss << "    " <<  ir_attr.name << "._transfer_ownership_when_pass_as_attr("
              << PyGenConstants::kOwnerGraphBuilder << ")" << std::endl;
          ss << std::endl;
      }
    }
  }

  static void GenListStringAttrConversion(const OpDescPtr &op, std::stringstream &ss) {
    // 为字符串列表属性生成转换代码
    bool first = true;
    for (const auto &attr : GetAllIrAttrsNamesAndTypeInOrder(op)) {
      if (!attr.type_info.is_list_type || strcmp(attr.type_info.av_type, "VT_LIST_STRING") != 0) {
        continue;
      }

      if (first) ss << "    # 转换字符串列表属性" << std::endl;
      first = false;
      const std::string attr_name = AttrName(attr.name, op, GenLanType::GenPy);
      const std::string bytes_var = "_bytes_" + attr_name;
      const std::string c_var = "_c_" + attr_name;
      // bytes 列表
      ss << "    " << bytes_var << " = [s.encode('utf-8') if isinstance(s, str) else s for s in "
         << attr_name << "]" << std::endl;
      // char** 数组
      ss << "    " << c_var << " = (ctypes.c_char_p * len(" << bytes_var << "))(*" << bytes_var << ")"
         << std::endl;
    }
    if (!first) ss << std::endl;
  }

  static void GenListListIntAttrConversion(const OpDescPtr &op, std::stringstream &ss) {
    // 为二维列表属性生成转换代码
    bool first = true;
    for (const auto &attr : GetAllIrAttrsNamesAndTypeInOrder(op)) {
      if (!attr.type_info.is_list_type || strcmp(attr.type_info.av_type, "VT_LIST_LIST_INT") != 0) {
        continue;
      }

      if (first) ss << "    # 转换二维列表属性" << std::endl;
      first = false;
      const std::string attr_name = AttrName(attr.name, op, GenLanType::GenPy);
      const std::string rows_var  = "_row_arrays_" + attr_name;
      const std::string pp_var    = "_pp_" + attr_name;
      const std::string inner_var = "_inner_num_" + attr_name;
      // 行数组
      ss << "    " << rows_var << " = [(" << PyGenConstants::kCtypesInt64 << " * len(inner_list))"
         << "(*[int(v) for v in inner_list])" << " for inner_list in " << attr_name << "]" << std::endl;
      // 指针数组
      ss << "    " << pp_var << " = (ctypes.POINTER(" << PyGenConstants::kCtypesInt64 << ") * len(" << attr_name << "))"
         << "(*[ctypes.cast(arr, ctypes.POINTER(" << PyGenConstants::kCtypesInt64 << ")) for arr in "
         << rows_var << "])" << std::endl;
      // 每行长度数组
      ss << "    " << inner_var << " = (ctypes.c_int64 * len(" << attr_name << "))"
         << "(*[len(inner_list) for inner_list in " << attr_name << "])" << std::endl;
    }
    if (!first) ss << std::endl;
  }

  static void GenFunctionCall(const OpDescPtr &op, std::stringstream &ss) {
    ss << "    # 调用C函数" << std::endl;
    ss << "    _result = esb_generated_lib." << CGenerator::FuncName(op->GetType()) << "(";
    GenFunctionCallArguments(op, ss);
    ss << ")" << std::endl;
    ss << std::endl;
  }

  static void GenFunctionCallArguments(const OpDescPtr &op, std::stringstream &ss) {
    bool first = true;

    // 处理输入参数
    for (const auto &in : op->GetIrInputs()) {
      if (!first) ss << ", ";
      first = false;

      if (in.second == kIrInputRequired) {
        ss << InName(in.first, GenLanType::GenPy) << "._handle";
      } else if (in.second == kIrInputOptional) {
        ss << "(" << InName(in.first, GenLanType::GenPy) << "._handle if " << InName(in.first, GenLanType::GenPy) << " is not None else None)";
      } else {
        ss << InName(in.first, GenLanType::GenPy) << "_array, len(" << InName(in.first, GenLanType::GenPy) << ")";
      }
    }

    // 如果没有输入则添加owner_graph_builder
    if (first) {
      ss << PyGenConstants::kOwnerGraphBuilder << "._handle";
    } else if (IsOpInputsAllOptional(op->GetIrInputs())) {
      ss << ", " << PyGenConstants::kOwnerBuilder << "._handle if " << PyGenConstants::kOwnerBuilder << " else "
         << PyGenConstants::kNoneValue;
    }

    // 添加动态输出个数参数
    DynamicOutputHandler::AddDynamicOutputCountCallArgs(op, ss);

    // 添加子图参数
    SubgraphHandler::AddSubgraphCallArgs(op, ss);

    // 处理属性参数
    auto ir_and_dts = GetAllIrAttrsNamesAndTypeInOrder(op);
    for (const auto &attr : ir_and_dts) {
      ss << ", ";
      GenAttributeConversion(op, attr, ss);
    }
  }

  static void GenAttributeConversion(const OpDescPtr &op, const IrAttrInfo &attr, std::stringstream &ss) {
    if (attr.type_info.is_list_type) {
      if (strcmp(attr.type_info.av_type, "VT_LIST_LIST_INT") == 0) {
        GenListListIntAttributeConversion(op, attr, ss);
      } else {
        GenListAttributeConversion(op, attr, ss);
      }
    } else {
      GenScalarAttributeConversion(op, attr, ss);
    }
  }

  static void GenListListIntAttributeConversion(const OpDescPtr &op, const IrAttrInfo &attr, std::stringstream &ss) {
    std::string attr_name = AttrName(attr.name, op, GenLanType::GenPy);
    ss << "_pp_" << attr_name << ", len(" << attr_name << "), _inner_num_" << attr_name;
  }

  static void GenListAttributeConversion(const OpDescPtr &op, const IrAttrInfo &attr, std::stringstream &ss) {
    std::string attr_name = AttrName(attr.name, op, GenLanType::GenPy);
    if (strcmp(attr.type_info.av_type, "VT_LIST_STRING") == 0) {
      ss << "_c_" << attr_name << ", len(" << attr_name << ")";
    } else if (strcmp(attr.type_info.av_type, "VT_LIST_DATA_TYPE") == 0) {
      ss << "(" << PythonTypeMapper::GetCtypesType(attr.type_info) << " * len(" << attr_name << "))"
         << "(*[dt.value for dt in " << attr_name << "]), len(" << attr_name << ")";
    } else {
      ss << "(" << PythonTypeMapper::GetCtypesType(attr.type_info) << " * len(" << attr_name << "))(*"
         << attr_name << "), len(" << attr_name << ")";
    }
  }

  static void GenScalarAttributeConversion(const OpDescPtr &op, const IrAttrInfo &attr, std::stringstream &ss) {
    if (strcmp(attr.type_info.av_type, "VT_STRING") == 0) {
      ss << AttrName(attr.name, op, GenLanType::GenPy) << ".encode('" << PyGenConstants::kUtf8Encoding << "')";
    } else if (strcmp(attr.type_info.av_type, "VT_DATA_TYPE") == 0) {
      ss << "ctypes.c_int(" << AttrName(attr.name, op, GenLanType::GenPy) << ".value)";
    } else if (strcmp(attr.type_info.av_type, "VT_TENSOR") == 0) {
      ss << AttrName(attr.name, op, GenLanType::GenPy) << "._handle";
    } else {
      ss << AttrName(attr.name, op, GenLanType::GenPy);
    }
  }

  static void GenReturnHandling(const OpDescPtr &op, std::stringstream &ss) {
    if (OutputTypeHandler::IsDynamicOutput(op)) {
      GenDynamicOutputReturn(op, ss);
    } else if (OutputTypeHandler::IsSingleOutput(GetOutputType(op))) {
      GenSingleOutputReturn(ss);
    } else if (OutputTypeHandler::IsMultiOutput(GetOutputType(op))) {
      GenMultiOutputReturn(op, ss);
    }
  }

  static void GenSingleOutputReturn(std::stringstream &ss) {
    ss << "    # 创建新的Python包装对象" << std::endl;
    ss << "    _tensor_holder = " << PyGenConstants::kTensorHolderType << "._create_from(_result, "
       << PyGenConstants::kOwnerGraphBuilder << ")" << std::endl;
    ss << "    # 尝试获取scope内信息设置到节点上" << std::endl;
    ss << "    " << PyGenConstants::kOwnerGraphBuilder << "._apply_scope_infos_to_node(_tensor_holder)" << std::endl;
    ss << "    return _tensor_holder" << std::endl;
  }

  static void GenMultiOutputReturn(const OpDescPtr &op, std::stringstream &ss) {
    ss << "    # 处理多输出结果" << std::endl;

    // 先创建各个输出tensor
    auto &ir_outputs = op->GetIrOutputs();
    for (const auto &out : ir_outputs) {
      ss << "    " << OutName(out.first, op, GenLanType::GenPy) << " = " << PyGenConstants::kTensorHolderType << "._create_from(_result."
         << OutName(out.first, op, GenLanType::GenPy) << ", " << PyGenConstants::kOwnerGraphBuilder << ")" << std::endl;
    }

    // 对第一个输出添加apply逻辑
    if (!ir_outputs.empty()) {
      auto first_output = ir_outputs.begin();
      ss << "    # 尝试获取scope内信息设置到节点上" << std::endl;
      ss << "    " << PyGenConstants::kOwnerGraphBuilder << "._apply_scope_infos_to_node("
         << OutName(first_output->first, op, GenLanType::GenPy) << ")" << std::endl;
    }

    // 构造并返回输出结构体
    ss << "    return " << NameGenerator::PyOutputStructName(op) << "(" << std::endl;
    for (const auto &out : ir_outputs) {
      ss << "        " << OutName(out.first, op, GenLanType::GenPy) << "=" << OutName(out.first, op, GenLanType::GenPy) << "," << std::endl;
    }
    ss << "    )" << std::endl;
  }

  static void GenDynamicOutputReturn(const OpDescPtr &op, std::stringstream &ss) {
    DynamicOutputHandler::GenReturnCode(op, ss);
  }

  // ==================== 类型转换工具 ====================

  static std::string GetPythonDefaultValue(const OpDescPtr &op, const std::string &attr_name,
                                           const std::string &av_type) {
    try {
      std::string default_val = GetDefaultValueString(op, attr_name, av_type);
      return StringConverter::ConvertCppDefaultToPython(default_val, av_type);
    } catch (...) {
      return PyGenConstants::kNoneValue;
    }
  }

 private:
  std::string module_name_;
  std::stringstream aggregatess_;
  std::unordered_map<std::string, std::stringstream> per_op_py_sources_;
};

}  // namespace es
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_PY_GENERATOR_H_
