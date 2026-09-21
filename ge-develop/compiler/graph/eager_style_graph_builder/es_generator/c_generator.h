/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_C_GENERATOR_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_C_GENERATOR_H_
#include <sstream>
#include <cstring>
#include <string>
#include <unordered_map>
#include "utils.h"
#include "graph/op_desc.h"
#include "generator_interface.h"

namespace {
inline constexpr const char *const kIndentTwo = "  ";       // 4个空格
inline constexpr const char *const kIndentFour = "    ";       // 4个空格
inline constexpr const char *const kIndentSix = "      ";      // 6个空格
inline constexpr const char *const kIndentEight = "        ";  // 8个空格
inline constexpr const char *const kIndentTen = "          ";  // 10个空格
inline constexpr const char *const kIndentTwelve = "            ";  // 12个空格

enum class AssertType {
  Es_NOTNULL = 0,
  Es_SUCCESS,
  Es_GRAPH_SUCCESS,
  Es_RT_OK,
  Es_EOK,
  Es_TRUE,
  Es_HYPER_SUCCESS,
};
}  // namespace

namespace ge {
namespace es {
class CGenerator : public ICodeGenerator {
 public:
  void GenOp(const OpDescPtr &op) override {
    auto ir_and_dts = GetAllIrAttrsNamesAndTypeInOrder(op);
    auto header_stream = GenOpHeaderStream(op, ir_and_dts);
    auto cc_stream = GenOpCCStream(op, ir_and_dts);
    op_type_to_hss_[op->GetType()] = std::move(header_stream);
    op_type_to_css_[op->GetType()] = std::move(cc_stream);
  }

  void ResetWhenGenFailed(const OpDescPtr &op) override {
    op_type_to_hss_.erase(op->GetType());
    op_type_to_css_.erase(op->GetType());
  }

  void GenAggregateHeader() override {
    GenAggregateHHead();
  }

  void GenAggregateIncludes(const std::vector<std::string> &sorted_op_types) override {
    for (const auto &op_type : sorted_op_types) {
      hss_ << "#include \"" << PerOpHeaderFileName(op_type) << "\"\n";
    }
    GenAggregateHTail();
  }

  std::string GetAggregateFileName() const override {
    return "es_" + module_name_ + "_ops_c.h";
  }

  std::stringstream &GetAggregateContentStream() override {
    return hss_;
  }

  std::string GetPerOpFileName(const std::string &op_type) const override {
    return PerOpHeaderFileName(op_type);
  }

  const std::unordered_map<std::string, std::stringstream> &GetPerOpContents() const override {
    return op_type_to_hss_;
  }

  void GenPerOpFiles(const std::string &output_dir, const std::vector<std::string> &op_types) override {
    // 生成 .h 文件
    WritePerOpFiles(output_dir, op_types, GetPerOpContents(),
                    [this](const std::string &op_type) { return GetPerOpFileName(op_type); });

    // 生成 .cpp 文件
    WritePerOpFiles(output_dir, op_types, GetPerOpCCSources(),
                    [](const std::string &op_type) { return PerOpSourceFileName(op_type); });
  }

  std::string GetGeneratorName() const override {
    return "C generator";
  }

  std::string GetCommentStart() const override {
    return "/**";
  }

  std::string GetCommentEnd() const override {
    return " */";
  }

  std::string GetCommentLinePrefix() const override {
    return " * ";
  }

  const std::unordered_map<std::string, std::stringstream> &GetPerOpCCSources() const {
    return op_type_to_css_;
  }
  static std::string PerOpHeaderFileName(const std::string &op_type) {
    return std::string("es_") + op_type + "_c.h";
  }
  static std::string PerOpSourceFileName(const std::string &op_type) {
    return std::string("es_") + op_type + ".cpp";
  }

  // 设置模块名，用于生成保护宏和文件名
  void SetModuleName(const std::string &module_name) {
    module_name_ = module_name;
  }

  // 设置保护宏前缀
  void SetHGuardPrefix(const std::string &h_guard_prefix) {
    h_guard_prefix_ = h_guard_prefix;
  }

  void GenAggregateHHead() {
    GenCopyright(hss_);
    hss_ << "\n#ifndef " << MakeGuardFromModule() << "\n";
    hss_ << "#define " << MakeGuardFromModule() << "\n";
    hss_ << "#ifdef __cplusplus\nextern \"C\" {\n#endif\n";
  }
  void GenAggregateHTail() {
    hss_ << R"(
#ifdef __cplusplus
}
#endif
#endif
)";
  }
  static std::string FuncName(const std::string &op_type) {
    return "Es" + op_type;
  }

  static std::string OutputStructName(const OpDescPtr &op) {
    return FuncName(op->GetType()) + "Output";
  }

 private:
  class IndexCalculator {
  public:
    std::string Index() {
      std::stringstream index;
      index << index_++;
      for (const auto &name : dynamic_num_names_) {
        index << " + " << name;
      }
      return index.str();
    }
    std::string DynamicIndex(const std::string &num_name, const char *relative_index) {
      std::stringstream index;
      index << index_;
      if (dynamic_num_names_set_.insert(num_name).second) {
        dynamic_num_names_.emplace_back(num_name);
      }
      for (const auto &name : dynamic_num_names_) {
        if (name == num_name) {
          continue;
        }
        index << " + " << name;
      }
      if (relative_index != nullptr) {
        index << " + " << relative_index;
      }
      return index.str();
    }
  private:
    int32_t index_ = 0;
    std::unordered_set<std::string> dynamic_num_names_set_;
    std::vector<std::string> dynamic_num_names_;
  };

  std::stringstream GenOpHeaderStream(const OpDescPtr &op, const std::vector<IrAttrInfo> &ir_and_dts) {
    std::stringstream h_stream;

    // 生成头文件头部（版权信息、保护宏、包含文件等）
    GenPerOpHHead(op->GetType(), h_stream);
    
    // 生成输出结构体定义（如果需要）
    GenOutputStructIfNeeded(op, h_stream);

    // 生成函数注释（如果需要）
    GenCommentsIfNeeded(op, h_stream);
    
    // 生成完整函数声明
    auto declare = GenDeclaration(op, ir_and_dts);
    h_stream << declare << ";" << std::endl;

    // 生成头文件尾部
    GenPerOpHTail(h_stream);

    return h_stream;
  }

  static std::stringstream GenOpCCStream(const OpDescPtr &op, const std::vector<IrAttrInfo> &ir_and_dts) {
    std::stringstream cc_stream;

    // 生成CC文件头部（版权信息、包含文件）
    GenPerOpCcHead(op->GetType(), cc_stream);

    // 生成函数体
    GenOpFunctionBody(op, ir_and_dts, cc_stream);

    // 生成CC文件尾部
    GenPerOpCcTail(cc_stream);

    return cc_stream;
  }

  static void GenOutputStructIfNeeded(const OpDescPtr &op, std::stringstream &h_stream) {
    switch (GetOutputType(op)) {
      case OutputType::kDynamicOutput:
      case OutputType::kMultiOutput:
        h_stream << std::endl << GenOutputStructDef(op) << std::endl;
        break;
      case OutputType::kNoOutput:
        h_stream << std::endl << GenNoOutputComment(op) << std::endl;
        break;
      default:
        break;
    }
  }

  static void GenOpFunctionBody(const OpDescPtr &op, const std::vector<IrAttrInfo> &ir_and_dts,
                                std::stringstream &cc_stream) {
    // 生成完整函数声明
    auto declare = GenDeclaration(op, ir_and_dts);
    cc_stream << declare << " {" << std::endl;

    // 生成输入检查代码
    GenInputValidationCode(op, cc_stream);

    // 生成Tensor属性存储代码
    GenAttrTensorPreNodeIfNeeded(op, ir_and_dts, cc_stream);

    // 生成节点构建代码
    GenNodeConstruction(op, ir_and_dts, cc_stream);

    // 生成边连接代码
    GenLinkEdges(op, cc_stream);

    // 生成子图添加代码
    GenAddSubgraphIfNeeded(op, cc_stream);

    // 生成返回值代码
    GenReturn(op, cc_stream);

    cc_stream << "}" << std::endl;
  }

  static void GenInputValidationCode(const OpDescPtr &op, std::stringstream &cc_stream) {
    if (op->GetIrInputs().empty()) {
      GenInputsCheckForNoIrInputs(cc_stream);
    } else {
      GenInputsCheck(op, cc_stream);
    }

    // 子图检查与转移
    GenSubgraphPreNodeIfNeed(op, cc_stream);
  }

  static std::string GetAttrTensorStoringFun(const OpDescPtr &op, const std::string &attr_tensor_name) {
    return "builder.AddResource(std::unique_ptr<ge::Tensor>(static_cast<ge::Tensor *>(static_cast<void *>(" +
           AttrName(attr_tensor_name, op) + "))));";
  }

  static std::string GetStoredAttrTensorName(const OpDescPtr &op, const std::string &attr_tensor_name) {
    return AttrName(attr_tensor_name, op) + "_stored";
  }

  static void GenAttrTensorPreNodeIfNeeded(const OpDescPtr &op, const std::vector<IrAttrInfo> &ir_and_dts,
                                           std::stringstream &cc_stream) {
    for (auto const &attr : ir_and_dts) {
      if (strcmp(attr.type_info.c_api_type, "EsCTensor *") == 0) {
        cc_stream << "  auto " << GetStoredAttrTensorName(op, attr.name) << " = "
                  << GetAttrTensorStoringFun(op, attr.name) << std::endl;
        GenAssertCheck(cc_stream, GetStoredAttrTensorName(op, attr.name), AssertType::Es_NOTNULL);
      }
    }
  }

  std::string MakeGuardFromModule() const {
    return ge::es::MakeGuardFromModule(module_name_, h_guard_prefix_);
  }

  std::string MakeGuardFromOp(const std::string &op_type) {
    return ge::es::MakeGuardFromOp(op_type, h_guard_prefix_);
  }

  void GenPerOpHHead(const std::string &op_type, std::stringstream &ss) {
    GenCopyright(ss);
    ss << "\n#ifndef " << MakeGuardFromOp(op_type) << "\n";
    ss << "#define " << MakeGuardFromOp(op_type) << "\n";
    ss << "#include \"esb_funcs.h\"\n";
    ss << "#include <stdint.h>\n";
    ss << "#include \"graph/types.h\"\n";
    ss << "#ifdef __cplusplus\nextern \"C\" {\n#endif\n";
  }
  static void GenPerOpHTail(std::stringstream &ss) {
    ss << "#ifdef __cplusplus\n}\n#endif\n";
    ss << "#endif\n";
  }
  static void GenPerOpCcHead(const std::string &op_type, std::stringstream &ss) {
    GenCopyright(ss);
    ss << "\n#include \"" << PerOpHeaderFileName(op_type) << "\"\n";
    ss << "#include \"es_c_graph_builder.h\"\n";
    ss << "#include \"compliant_node_builder.h\"\n";
    ss << "#include \"utils/extern_math_util.h\"\n";
    ss << "#include \"es_log.h\"\n";
    ss << "#include \"es_tensor_like.h\"\n";
    ss << "#include <algorithm>\n";
    ss << "#include <vector>\n";
    ss << "#include <functional>\n";
    ss << "#include <cmath>\n";
    ss << "\n";
    ss << "#ifdef __cplusplus\nextern \"C\" {\n#endif\n";
  }
  static void GenPerOpCcTail(std::stringstream &ss) {
    ss << "#ifdef __cplusplus\n}\n#endif\n";
  }
  static void GenAssertCheck(std::stringstream &css, const std::string exp,
                             const AssertType assert_type, const std::string &indent = kIndentTwo) {
    switch (assert_type) {
      case AssertType::Es_NOTNULL:
        css << indent << "ES_ASSERT_NOTNULL(" << exp << ");" << std::endl;
        break;
      case AssertType::Es_SUCCESS:
        css << indent << "ES_ASSERT_SUCCESS(" << exp << ");" << std::endl;
        break;
      case AssertType::Es_GRAPH_SUCCESS:
        css << indent << "ES_ASSERT_GRAPH_SUCCESS(" << exp << ");" << std::endl;
        break;
      case AssertType::Es_RT_OK:
        css << indent << "ES_ASSERT_RT_OK(" << exp << ");" << std::endl;
        break;
      case AssertType::Es_EOK:
        css << indent << "ES_ASSERT_EOK(" << exp << ");" << std::endl;
        break;
      case AssertType::Es_TRUE:
        css << indent << "ES_ASSERT_TRUE(" << exp << ");" << std::endl;
        break;
      case AssertType::Es_HYPER_SUCCESS:
        css << indent << "ES_ASSERT_HYPER_SUCCESS(" << exp << ");" << std::endl;
        break;
      default:
        break;
    }
  }

  static std::vector<std::string> GenDynamicOutputSizeParam(const OpDescPtr &op) {
    std::vector<std::string> output_size_str;
    for (const auto &ir_out : op->GetIrOutputs()) {
      if (ir_out.second == kIrOutputDynamic) {
        output_size_str.emplace_back("int64_t " + OutName(ir_out.first, op) + "_num");
      }
    }
    return output_size_str;
  }
  static std::string GetSubgraphDeclaration(const OpDescPtr &op) {
    std::stringstream ss;
    for (const auto& ir_subgraph: op->GetOrderedSubgraphIrNames()) {
      ss << ", ";
      if (ir_subgraph.second == kStatic) {
        ss << "EsCGraph *" << SubgraphName(ir_subgraph.first);
      } else {
        ss << "EsCGraph **" << SubgraphName(ir_subgraph.first);
        ss << ", ";
        ss << "int64_t " << SubgraphName(ir_subgraph.first) << "_num";
      }
    }
    return ss.str();
  }

  /**
   * 函数定义：EsTensor *Es<OpType>(<all_inputs>, <all_required_attrs>, <all_optional_attrs>)
   * @param op
   * @param ir_and_dts
   * @return
   */
  static std::string GenDeclaration(const OpDescPtr &op, const std::vector<IrAttrInfo> &ir_and_dts) {
    std::stringstream ss;
    switch (GetOutputType(op)) {
      case OutputType::kNoOutput:
      case OutputType::kOneOutput:
        ss << "EsCTensorHolder *";
        break;
      case OutputType::kDynamicOutput: // 与Multioutput逻辑一致
      case OutputType::kMultiOutput:
        ss << OutputStructName(op) << ' ';
        break;
      default:
        break;
    }
    ss << FuncName(op->GetType()) + "(";

    bool first = true;
    for (const auto &in : op->GetIrInputs()) {
      if (first) {
        first = false;
      } else {
        ss << ", ";
      }
      if (in.second == kIrInputRequired || in.second == kIrInputOptional) {
        ss << "EsCTensorHolder *" << InName(in.first);
      } else {
        ss << "EsCTensorHolder **" << InName(in.first) << ", int64_t " << InName(in.first) << "_num";
      }
    }
    if (first) {  // no inputs
      ss << "EsCGraphBuilder *owner_graph_builder";
    } else if (IsOpInputsAllOptional(op->GetIrInputs())) {  // all optional
      ss << ", EsCGraphBuilder *owner_graph_builder";
    }
    for (const auto& dynamic_output_size: GenDynamicOutputSizeParam(op)) {
      ss << ", " << dynamic_output_size;
    }

    ss << GetSubgraphDeclaration(op);

    for (const auto &attr : ir_and_dts) {
      ss << ", ";
      ss << GetAttrDeclaration(op, attr);
    }
    ss << ")";
    return ss.str();
  }

  static bool IsListListIntType(const IrAttrInfo &attr) {
    return strcmp(attr.type_info.c_api_type, "const int64_t **") == 0;
  }

  static std::string GetAttrDeclaration(const OpDescPtr &op, const IrAttrInfo &attr) {
    if (attr.type_info.is_list_type) {
      if (IsListListIntType(attr)) {
        return attr.type_info.c_api_type + AttrName(attr.name, op) + ", int64_t " +
               AttrName(attr.name, op) + "_num" + ", const int64_t *" +
               AttrName(attr.name, op) + "_inner_num";
      } else {
        return attr.type_info.c_api_type + AttrName(attr.name, op) + ", int64_t " +
               AttrName(attr.name, op) + "_num";
      }
    } else {
      return attr.type_info.c_api_type + AttrName(attr.name, op);
    }
  }

  static std::string GenOutputStructDef(const OpDescPtr &op) {
    std::stringstream ss;
    ss << "typedef struct {" << std::endl;
    for (const auto &ir_out : op->GetIrOutputs()) {
      if (ir_out.second == kIrOutputDynamic) {
        ss << "  EsCTensorHolder **" << OutName(ir_out.first, op) << ";" << std::endl;
        ss << "  int64_t " << OutName(ir_out.first, op) << "_num;" << std::endl;
      } else {
        ss << "  EsCTensorHolder *" << OutName(ir_out.first, op) << ";" << std::endl;
      }
    }
    ss << "} " << OutputStructName(op) << ";";
    return ss.str();
  }
  static void GenInputsCheckForNoIrInputs(std::stringstream &css) {
    GenAssertCheck(css, "owner_graph_builder", AssertType::Es_NOTNULL);
    css << "  auto &builder = *owner_graph_builder;" << std::endl;
    css << "  auto ge_graph = builder.GetGraph();" << std::endl;
    css << std::endl;
  }
  static std::string GenNoOutputComment(const OpDescPtr &op) {
    return "// " + op->GetType() +
           " does not produce an output, so the returned EsTensor cannot be used to connect edges.";
  }

  static void GenInputsCheck(const OpDescPtr &op, std::stringstream &css) {
    for (const auto &in : op->GetIrInputs()) {
      if (in.second == kIrInputRequired) {
        GenAssertCheck(css, InName(in.first), AssertType::Es_NOTNULL);
      } else if (in.second == kIrInputDynamic) {
        GenAssertCheck(css, "ge::IntegerChecker<int32_t>::Compat(" + InName(in.first) + "_num)", AssertType::Es_TRUE);
      }
    }

    // 使用 ResolveBuilder 获取 owner_builder
    css << "  auto *owner_builder = ge::es::ResolveBuilder(";
    bool first = true;
    for (const auto &in : op->GetIrInputs()) {
      if (first) {
        first = false;
      } else {
        css << ", ";
      }
      if (in.second == kIrInputRequired || in.second == kIrInputOptional) {
        css << InName(in.first);
      } else if (in.second == kIrInputDynamic) {
        css << "std::vector<EsCTensorHolder *>(" << InName(in.first) << ", "
            << InName(in.first) << " + " << InName(in.first) << "_num)";
      }
    }
    if (IsOpInputsAllOptional(op->GetIrInputs())) {
      css << ", owner_graph_builder";
    }
    css << ");" << std::endl;
    GenAssertCheck(css, "owner_builder, "
                        "\"Failed to resolve owner builder: please ensure at least one input tensor "
                        "or an explicit owner_graph_builder is provided when supported.\"", AssertType::Es_NOTNULL);
    css << "  auto &builder = *owner_builder;" << std::endl;
    css << "  auto ge_graph = builder.GetGraph();" << std::endl;
    css << std::endl;
  }
  static const char *IrInputTypeToString(IrInputType type) {
    switch (type) {
      case kIrInputRequired:
        return "ge::es::CompliantNodeBuilder::kEsIrInputRequired";
      case kIrInputOptional:
        return "ge::es::CompliantNodeBuilder::kEsIrInputOptional";
      case kIrInputDynamic:
        return "ge::es::CompliantNodeBuilder::kEsIrInputDynamic";
      default:
        throw std::invalid_argument("Invalid Ir input type");
    }
  }
  static const char *IrOutputTypeToString(IrOutputType type) {
    switch (type) {
      case kIrOutputRequired:
        return "ge::es::CompliantNodeBuilder::kEsIrOutputRequired";
      case kIrOutputDynamic:
        return "ge::es::CompliantNodeBuilder::kEsIrOutputDynamic";
      default:
        throw std::invalid_argument("Invalid Ir output type");
    }
  }
  static std::string GenAttrValueInit(const OpDescPtr &op, const IrAttrInfo &attr) {
    std::stringstream ss;
    ss << "ge::es::CreateFrom(";
    if (strcmp(attr.type_info.c_api_type, "int64_t ") == 0) {
      ss << "static_cast<int64_t>(" << AttrName(attr.name, op) << ")";
    } else if (strcmp(attr.type_info.c_api_type, "float ") == 0) {
      ss << "static_cast<float>(" << AttrName(attr.name, op) << ")";
    } else if (strcmp(attr.type_info.c_api_type, "const char *") == 0) {
      ss << "ge::AscendString(" << AttrName(attr.name, op) << ")";
    } else if (strcmp(attr.type_info.c_api_type, "bool ") == 0) {
      ss << "static_cast<bool>(" << AttrName(attr.name, op) << ")";
    } else if (strcmp(attr.type_info.c_api_type, "C_DataType ") == 0) {
      ss << "static_cast<ge::DataType>(" << AttrName(attr.name, op) << ")";
    } else if (strcmp(attr.type_info.c_api_type, "EsCTensor *") == 0) {
      ss << "*" << GetStoredAttrTensorName(op, attr.name); // 需要解引用
    } else if (strcmp(attr.type_info.c_api_type, "const int64_t *") == 0) {
      ss << "std::vector<int64_t>(" << AttrName(attr.name, op) << ", " << AttrName(attr.name, op) << " + "
         << AttrName(attr.name, op) << "_num)";
    } else if (strcmp(attr.type_info.c_api_type, "const float *") == 0) {
      ss << "std::vector<float>(" << AttrName(attr.name, op) << ", " << AttrName(attr.name, op) << " + " << AttrName(attr.name, op)
         << "_num)";
    } else if (strcmp(attr.type_info.c_api_type, "const bool *") == 0) {
      ss << "std::vector<bool>(" << AttrName(attr.name, op) << ", " << AttrName(attr.name, op) << " + " << AttrName(attr.name, op)
         << "_num)";
    } else if (strcmp(attr.type_info.c_api_type, "const C_DataType *") == 0) {
      ss << "std::vector<ge::DataType>([](const C_DataType *es_type_list, int64_t es_type_list_size) {" << " ";
      ss << "std::vector<ge::DataType> ge_type_list(es_type_list_size);" << " ";
      ss << "std::transform(es_type_list, es_type_list + es_type_list_size, ge_type_list.begin(), [](C_DataType es_type) {" << " ";
      ss << "return static_cast<ge::DataType>(es_type);});" << " ";
      ss << "return ge_type_list;" << " ";
      ss << "}(" << AttrName(attr.name, op) << ", " << AttrName(attr.name, op) << "_num))";
    } else if (IsListListIntType(attr)) {
      ss << "std::vector<std::vector<int64_t>>([](const int64_t** data, int64_t size, const int64_t *inner_sizes) {" << " ";
      ss << "std::vector<std::vector<int64_t>> ret;" << " ";
      ss << "for (int64_t i = 0; i < size; i++) ret.emplace_back(data[i], data[i] + inner_sizes[i]);" << " ";
      ss << "return ret;" << " ";
      ss << "} (" << AttrName(attr.name, op) << ", " << AttrName(attr.name, op) << "_num, " << AttrName(attr.name, op) << "_inner_num))";
    } else if (strcmp(attr.type_info.c_api_type, "const char **") == 0) {
      ss << "std::vector<ge::AscendString>(" << AttrName(attr.name, op) << ", " << AttrName(attr.name, op) << " + " << AttrName(attr.name, op)
         << "_num)";
    } else {
      // 这里已经在代码生成的过程中了，出现不支持，只能中止当前op的代码生成
      // 在前面的流程中，应该已经完成了对不支持的类型的检查，抛出了NotSupportException
      throw std::runtime_error(std::string("Unsupported attr type: ") + attr.type_info.av_type + ", " +
                               attr.type_info.c_api_type);
    }
    ss << ")";
    return ss.str();
  }

  // 为 ListDataType 生成转换 lambda 表达式
  static std::string GenListDataTypeConversionLambda(const std::string &param_name) {
    return "[](const C_DataType *es_type_list, int64_t es_type_list_size) { "
           "std::vector<ge::DataType> ge_type_list(es_type_list_size); "
           "std::transform(es_type_list, es_type_list + es_type_list_size, ge_type_list.begin(), "
           "[](C_DataType es_type) { return static_cast<ge::DataType>(es_type);}); "
           "return ge_type_list; }(" +
           param_name + ", " + param_name + "_num)";
  }

  // 为 ListListInt 生成转换 lambda 表达式
  static std::string GenListListIntConversionLambda(const std::string &param_name) {
    return "[](const int64_t** data, int64_t size, const int64_t *inner_sizes) { "
           "std::vector<std::vector<int64_t>> ret; "
           "for (int64_t i = 0; i < size; i++) ret.emplace_back(data[i], data[i] + inner_sizes[i]); "
           "return ret; } (" +
           param_name + ", " + param_name + "_num, " + param_name + "_inner_num)";
  }

  // 为可选属性生成表达式：如果传入值等于默认值，则不设置该属性
  static std::string GenOptionalAttrValueInit(const OpDescPtr &op, const IrAttrInfo &attr) {
    const std::string param_name = AttrName(attr.name, op);
    const std::string default_value_str = GetDefaultValueString(op, attr.name, attr.type_info.av_type);
    const char *c_api_type = attr.type_info.c_api_type;
    std::stringstream ss;

    // 标量类型
    if (strcmp(c_api_type, "int64_t ") == 0) {
      ss << "ge::es::CreateFromIfNotEqual(" << param_name << ", static_cast<int64_t>(" << default_value_str << "))";
    } else if (strcmp(c_api_type, "float ") == 0) {
      ss << "ge::es::CreateFromIfNotEqual(" << param_name << ", static_cast<float>(" << default_value_str << "))";
    } else if (strcmp(c_api_type, "bool ") == 0) {
      ss << "ge::es::CreateFromIfNotEqual(" << param_name << ", static_cast<bool>(" << default_value_str << "))";
    } else if (strcmp(c_api_type, "C_DataType ") == 0) {
      ss << "ge::es::CreateFromIfNotEqual(static_cast<ge::DataType>(" << param_name << "), " << default_value_str << ")";
    } else if (strcmp(c_api_type, "const char *") == 0) {
      ss << "ge::es::CreateFromIfNotEqual(ge::AscendString(" << param_name << "), ge::AscendString(" << default_value_str << "))";
    }
    // 简单列表类型
    else if (strcmp(c_api_type, "const int64_t *") == 0) {
      ss << "ge::es::CreateFromIfNotEqual(std::vector<int64_t>(" << param_name << ", " << param_name << " + "
         << param_name << "_num), std::vector<int64_t>" << default_value_str << ")";
    } else if (strcmp(c_api_type, "const float *") == 0) {
      ss << "ge::es::CreateFromIfNotEqual(std::vector<float>(" << param_name << ", " << param_name << " + "
         << param_name << "_num), std::vector<float>" << default_value_str << ")";
    } else if (strcmp(c_api_type, "const bool *") == 0) {
      ss << "ge::es::CreateFromIfNotEqual(std::vector<bool>(" << param_name << ", " << param_name << " + "
         << param_name << "_num), std::vector<bool>" << default_value_str << ")";
    } else if (strcmp(c_api_type, "const char **") == 0) {
      ss << "ge::es::CreateFromIfNotEqual(std::vector<ge::AscendString>(" << param_name << ", " << param_name << " + "
         << param_name << "_num), std::vector<ge::AscendString>" << default_value_str << ")";
    }
    // 特殊类型
    else if (strcmp(c_api_type, "const C_DataType *") == 0) {
      std::string conversion_lambda = GenListDataTypeConversionLambda(param_name);
      ss << "ge::es::CreateFromIfNotEqual(std::vector<ge::DataType>(" << conversion_lambda
         << "), std::vector<ge::DataType>" << default_value_str << ")";
    } else if (IsListListIntType(attr)) {
      std::string conversion_lambda = GenListListIntConversionLambda(param_name);
      ss << "ge::es::CreateFromIfNotEqual(std::vector<std::vector<int64_t>>(" << conversion_lambda
         << "), std::vector<std::vector<int64_t>>" << default_value_str << ")";
    } else if (strcmp(c_api_type, "EsCTensor *") == 0) {
      // Tensor 类型无法判等，因此一直设置
      ss << "ge::es::CreateFrom(*" << GetStoredAttrTensorName(op, attr.name) <<")";
    } else {
      throw std::runtime_error(std::string("Unsupported optional attr type for default value comparison: ") +
                               attr.type_info.av_type + ", " + attr.type_info.c_api_type);
    }
    return ss.str();
  }

  static void GenSubgraphCheckAndStore(std::stringstream &css, const std::string& subgraph_name,
                                       const std::string &input_num, const std::string &output_num,
                                       const bool subgraph_compare_output_num = true,
                                       const std::string &indent = kIndentTwo) {
    css << indent << "auto " << subgraph_name << "_node_type = ge::AscendString();" << std::endl;
    css << indent << "int net_output_num_of_" << subgraph_name << " = 0;" << std::endl;
    css << indent << "std::unordered_set<int64_t> " << subgraph_name << "_data_indexes;" << std::endl;
    css << indent << "" << subgraph_name << "_data_indexes.reserve("<< input_num <<");" << std::endl;
    css << indent << "for (auto &sub_graph_node : input_subgraph->GetDirectNode()) {" << std::endl;
    GenAssertCheck(css, "sub_graph_node.GetType(" + subgraph_name + "_node_type)", AssertType::Es_GRAPH_SUCCESS, indent + kIndentTwo);
    css << indent << "  if (" << subgraph_name << "_node_type != \"Data\") {" << std::endl;
    css << indent << "    if (" << subgraph_name << "_node_type == \"NetOutput\") {" << std::endl;
    GenAssertCheck(css, "net_output_num_of_" + subgraph_name + " <= 1", AssertType::Es_TRUE, indent + kIndentSix);
    css << indent << "      net_output_num_of_" << subgraph_name <<"++;" << std::endl;
    GenAssertCheck(css, "ge::IntegerChecker<int32_t>::Compat(" + output_num + ")", AssertType::Es_TRUE, indent + kIndentSix);
    if (subgraph_compare_output_num) {
      css << indent << "      auto subgraph_output_cnt = sub_graph_node.GetInputsSize();" << std::endl;
      GenAssertCheck(css, "static_cast<int64_t>(subgraph_output_cnt)  == " + output_num, AssertType::Es_TRUE, indent + kIndentEight);
    }
    css << indent << "    }" << std::endl;
    css << indent << "    continue;" << std::endl;
    css << indent << "  }" << std::endl;
    css << indent << "  int64_t index_value;" << std::endl;
    GenAssertCheck(css, "sub_graph_node.GetAttr(\"index\", index_value)", AssertType::Es_GRAPH_SUCCESS, indent + kIndentTwo);
    GenAssertCheck(css, subgraph_name + "_data_indexes.insert(index_value).second", AssertType::Es_TRUE, indent + kIndentTwo);
    GenAssertCheck(css, "index_value < " + input_num, AssertType::Es_TRUE, indent + kIndentTwo);
    css << indent << "}" << std::endl;
    GenAssertCheck(css, "net_output_num_of_" + subgraph_name + " == 1", AssertType::Es_TRUE, indent);
  }

  static void GenSubgraphPreNodeIfNeed(const OpDescPtr &op, std::stringstream &css) {
    std::string input_num;
    std::string output_num;

    for (const auto &ir_in: op->GetIrInputs()) {
      if (ir_in.second == kIrInputDynamic) { // 获取第一个动态输入的个数
        input_num = InName(ir_in.first) + "_num";
        break;
      }
    }

    for (const auto &ir_out: op->GetIrOutputs()) {
      if (ir_out.second == kIrOutputDynamic) { // 获取第一个动态输出的个数
        output_num = OutName(ir_out.first, op) + "_num";
        break;
      }
    }

    bool has_subgraph = false;
    bool subgraph_compare_output_num = true;
    for (const auto &ir_subgraph : op->GetOrderedSubgraphIrNames()) {
      if (!has_subgraph) {
        has_subgraph = true;
        css << "  ge::Graph* input_subgraph;" << std::endl;
      }
      if (op->GetName() == "While" && ir_subgraph.first == "cond") {  // 剔除While算子cond子图特例
        subgraph_compare_output_num = false;
      } else {
        subgraph_compare_output_num = true;
      }
      if (ir_subgraph.second == kStatic) {
        css << "  input_subgraph = static_cast<ge::Graph *>(static_cast<void *>(" << SubgraphName(ir_subgraph.first) << "));" << std::endl;
        GenSubgraphCheckAndStore(css, SubgraphName(ir_subgraph.first), input_num, output_num,
                                 subgraph_compare_output_num, kIndentTwo);
        css << "  auto " << SubgraphName(ir_subgraph.first) << "_ptr = builder.AddResource(std::unique_ptr<ge::Graph>(input_subgraph));"
            << std::endl;
        css << std::endl;
      } else {
        css << "  std::vector<ge::Graph> " << DynamicSubgraphVectorName(ir_subgraph.first) << ";" << std::endl;
        css << "  for (int64_t subgraph_idx = 0; subgraph_idx < " << SubgraphName(ir_subgraph.first)
            << "_num; subgraph_idx++) {" << std::endl;
        css << "    const auto subgraph_instance = " << SubgraphName(ir_subgraph.first) << "[subgraph_idx];"
            << std::endl;
        css << "    input_subgraph = static_cast<ge::Graph *>(static_cast<void *>(subgraph_instance));" << std::endl;
        GenSubgraphCheckAndStore(css, SubgraphName(ir_subgraph.first), input_num, output_num,
                                 subgraph_compare_output_num, kIndentFour);
        css << "    " << DynamicSubgraphVectorName(ir_subgraph.first)
            << ".emplace_back(*builder.AddResource(std::unique_ptr<ge::Graph>(input_subgraph)));" << std::endl;
        css << "  }" << std::endl;
        css << std::endl;
      }
    }
  }
  static void GenNodeAttrByIrOrder(const OpDescPtr &op, const std::vector<IrAttrInfo> &ir_and_dts,
                                   std::stringstream &css) {
    std::unordered_map<std::string, const IrAttrInfo *> attr_map;
    for (const auto &attr : ir_and_dts) {
      attr_map[attr.name] = &attr;
    }

    css << "      .IrDefAttrs({" << std::endl;
    // 按照原始的 ir 顺序构造 Node
    for (const auto &attr_name : op->GetIrAttrNames()) {
      auto it = attr_map.find(attr_name);
      if (it != attr_map.end()) {
        const auto &attr = *(it->second);
        css << "          {" << std::endl;
        css << "              \"" << attr.name << "\"," << std::endl;
        css << "              " << attr.GetRequiredString() << "," << std::endl;
        css << "              \"" << attr.type_info.ir_type << "\"," << std::endl;
        if (attr.is_required) {
          css << "              " << GenAttrValueInit(op, attr) << std::endl;
        } else {
          css << "              " << GenOptionalAttrValueInit(op, attr) << std::endl;
        }
        css << "          }," << std::endl;
      }
    }

    css << "      })" << std::endl;
  }
  static void GenNodeConstruction(const OpDescPtr &op, const std::vector<IrAttrInfo> &ir_and_dts,
                                  std::stringstream &css) {
    css << "  auto node = ge::es::CompliantNodeBuilder(ge_graph).OpType(\"" << op->GetType() << "\")" << std::endl;
    css << "      .Name( builder.GenerateNodeName(\"" << op->GetType() << "\").GetString())" << std::endl;
    css << "      .IrDefInputs({" << std::endl;
    for (const auto &in : op->GetIrInputs()) {
      css << "          {\"" << in.first << "\", " << IrInputTypeToString(in.second) << ", \"\"}," << std::endl;
    }
    css << "      })" << std::endl;
    css << "      .IrDefOutputs({" << std::endl;
    for (const auto &out : op->GetIrOutputs()) {
      css << "          {\"" << out.first << "\", " << IrOutputTypeToString(out.second) << ", \"\"}," << std::endl;
    }
    css << "      })" << std::endl;
    GenNodeAttrByIrOrder(op, ir_and_dts, css);
    for (const auto &in : op->GetIrInputs()) {
      if (in.second == kIrInputDynamic) {
        css << "      .InstanceDynamicInputNum(\"" << in.first << "\", static_cast<int32_t>(" << InName(in.first)
            << "_num))" << std::endl;
      }
    }
    for (const auto &out : op->GetIrOutputs()) {
      if (out.second == kIrOutputDynamic) {
        css << "      .InstanceDynamicOutputNum(\"" << out.first << "\", static_cast<int32_t>("
            << OutName(out.first, op) << "_num))" << std::endl;
      }
    }
    css << "      .Build();" << std::endl;
    css << std::endl;
  }
  static void GenLinkEdges(const OpDescPtr &op, std::stringstream &css) {
    IndexCalculator input_index;
    for (size_t i = 0U; i < op->GetIrInputsSize(); ++i) {
      const auto &in = op->GetIrInputs()[i];
      if (in.second == kIrInputOptional) {
        css << "  if (" << InName(in.first) << " != nullptr) {" << std::endl;
        GenAssertCheck(css,
                       "ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, " + InName(in.first) + "->GetProducer(), " +
                           InName(in.first) + "->GetOutIndex(), node, " + input_index.Index() + ")",
                       AssertType::Es_GRAPH_SUCCESS, kIndentFour);
        css << "  }" << std::endl;
      } else if (in.second == kIrInputDynamic) {
        std::string one_name = "one_" + in.first;
        std::string num_name = InName(in.first) + "_num";
        css << "  if ((" << InName(in.first) << " != nullptr) && (" << num_name << " > 0)) {" << std::endl;
        css << "    for (int64_t i = 0; i < " << num_name << "; ++i) {" << std::endl;
        css << "      auto " << one_name << " = " << InName(in.first) << "[i];" << std::endl;
        GenAssertCheck(css, one_name, AssertType::Es_NOTNULL, kIndentSix);
        GenAssertCheck(css,
                       "ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, " + one_name + "->GetProducer(), " + one_name +
                           "->GetOutIndex(), node, " + input_index.DynamicIndex(num_name, "i") + ")",
                       AssertType::Es_GRAPH_SUCCESS, kIndentSix);
        css << "    }" << std::endl;
        css << "  }" << std::endl;
      } else {
        GenAssertCheck(css,
                       "ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, " + InName(in.first) + "->GetProducer(), " +
                           InName(in.first) + "->GetOutIndex(), node, " + input_index.Index() + ")",
                       AssertType::Es_GRAPH_SUCCESS);
      }
    }
  }
  static void GenAddSubgraphIfNeeded(const OpDescPtr &op, std::stringstream &css) {
    bool has_subgraph = false;
    for (const auto& ir_subgraph: op->GetOrderedSubgraphIrNames()) {
      if (!has_subgraph) {
        has_subgraph = true;
        css << std::endl;
      }
      if (ir_subgraph.second == kStatic) {
        css << "  node.SetSubgraph(\"" << SubgraphName(ir_subgraph.first) << "\", *"<< SubgraphName(ir_subgraph.first) <<"_ptr);" << std::endl;
      } else {
        css << "  node.SetSubgraphs(\"" << SubgraphName(ir_subgraph.first) << "\", "<< DynamicSubgraphVectorName(ir_subgraph.first) <<");" << std::endl;
      }
    }
    if (has_subgraph) {
      css << std::endl;
    }
  }
  static void GenDynamicOutput(const OpDescPtr &op, std::stringstream &css) {
    IndexCalculator input_index;

    std::stringstream dyn_ret_ss;
    std::string num_name;
    dyn_ret_ss << "  return " << OutputStructName(op) << "{" << std::endl;
    for (const auto &ir_out : op->GetIrOutputs()) {
      if (ir_out.second == kIrOutputDynamic) {
        num_name = OutName(ir_out.first, op) + "_num";
        css << "  auto " << OutName(ir_out.first, op) << "_holders = builder.CreateDynamicTensorHolderFromNode(node, "
            << input_index.DynamicIndex(num_name, nullptr) << ", " << num_name << ");" << std::endl;

        dyn_ret_ss << "      " << OutName(ir_out.first, op) << "_holders->data()," << std::endl;
        dyn_ret_ss << "      static_cast<int64_t>(" << OutName(ir_out.first, op) << "_holders->size())," << std::endl;
      } else {
        dyn_ret_ss << "      builder.GetTensorHolderFromNode(node, " << input_index.Index() << ")," << std::endl;
      }
    }
    dyn_ret_ss << "  };" << std::endl;
    css << dyn_ret_ss.str();
  }
  static void GenReturn(const OpDescPtr &op, std::stringstream &css) {
    switch (GetOutputType(op)) {
      case OutputType::kNoOutput:
        css << "  return builder.GetTensorHolderFromNode(std::move(node), -1);" << std::endl;
        break;
      case OutputType::kOneOutput:
        css << "  return builder.GetTensorHolderFromNode(std::move(node), 0);" << std::endl;
        break;
      case OutputType::kMultiOutput: {
        css << "  return " << OutputStructName(op) << "{" << std::endl;
        auto &ir_outputs = op->GetIrOutputs();
        for (size_t i = 0U; i < ir_outputs.size(); ++i) {
          css << "      builder.GetTensorHolderFromNode(node, " << i << ")," << std::endl;
        }
        css << "};" << std::endl;
        break;
      }
      case OutputType::kDynamicOutput:
        GenDynamicOutput(op, css);
        break;
      default:
        break;
      }
  }

 private:
  std::stringstream hss_;
  std::unordered_map<std::string, std::stringstream> op_type_to_hss_;
  std::unordered_map<std::string, std::stringstream> op_type_to_css_;
  std::unordered_map<std::string, std::unordered_set<std::string>> op_to_input_names_;
  std::string module_name_{};
  std::string h_guard_prefix_{};
};
}  // namespace es
}  // namespace ge
#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_C_GENERATOR_H_
