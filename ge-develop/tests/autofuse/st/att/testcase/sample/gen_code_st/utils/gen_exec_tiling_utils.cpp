/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gen_exec_tiling_utils.h"
#include <map>
#include "code_printer.h"
#include "user_input_parser.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/utils/type_utils.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "base/base_types.h"
#include "common/checker.h"
namespace att {
namespace {
constexpr char kTilingDataType[] = "TilingData";
constexpr char kTilingFuncStName[] = "tiling_func_main.cpp";
bool g_need_hbm = false;
constexpr int32_t kInvalidIndex = -1;
inline std::string GetTilingDataType(const std::string &graph_name) {
  return "optiling::" + graph_name + kTilingDataType;
}
inline std::string SettingFuncName(const std::string &graph_name) {
  return graph_name + kTilingDataType + "SettingFunc";
}
inline std::string GetTilingDataSettingFunc(const std::string &graph_name) {
  return "Get" + SettingFuncName(graph_name);
}

void GenLicense(CodePrinter &code_printer) {
  code_printer.AddLine("/**");
  code_printer.AddLine(" * Copyright (c) Huawei Technologies Co., Ltd. 2024 All rights reserved.");
  code_printer.AddLine(" *");
  code_printer.AddLine(" * Licensed under the Apache License, Version 2.0 (the \"License\");");
  code_printer.AddLine(" * you may not use this file except in compliance with the License.");
  code_printer.AddLine(" * You may obtain a copy of the License at");
  code_printer.AddLine(" *");
  code_printer.AddLine(" * http://www.apache.org/licenses/LICENSE-2.0");
  code_printer.AddLine(" *");
  code_printer.AddLine(" * Unless required by applicable law or agreed to in writing, software");
  code_printer.AddLine(" * distributed under the License is distributed on an \"AS IS\" BASIS,");
  code_printer.AddLine(" * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.");
  code_printer.AddLine(" * See the License for the specific language governing permissions and");
  code_printer.AddLine(" * limitations under the License.");
  code_printer.AddLine(" */");
}

void GenSTHead(const std::string &op_name,
               const std::map<uint32_t, std::map<std::string, std::vector<std::string>>> &keys_to_axes_names,
               CodePrinter &code_printer) {
  GenLicense(code_printer);
  code_printer.AddInclude("functional");
  code_printer.AddInclude("map");
  code_printer.AddInclude("common/checker.h");
  code_printer.AddInclude(op_name + "_tiling_data.h");
  code_printer.AddInclude("user_input_parser.h");
  code_printer.AddInclude("struct_info.h");
  for (const auto &iter1 : keys_to_axes_names) {
    for (const auto &iter2 : iter1.second) {
      code_printer.AddLine("using TilingFunction = std::function<void(const uint32_t, optiling::" + iter2.first +
          std::string(kTilingDataType) + " &)>;");
    }
  }
  code_printer.AddInclude("kernel_context_holder_builder.h");
  code_printer.AddLine("namespace optiling {");
  code_printer.AddLine("  extern ge::graphStatus GetCtxTiling(gert::TilingContext *context, int32_t tilingCaseId);");
  code_printer.AddLine("}");
  code_printer.AddLine("using namespace att;");
}

void GenGetSettingFunc(const std::map<uint32_t, std::map<std::string, std::vector<std::string>>> &keys_to_axes_names,
                       CodePrinter &code_printer) {
  for (const auto &iter1 : keys_to_axes_names) {
    const auto &key = iter1.first;
    for (const auto &iter2 : iter1.second) {
      const auto &graph_name = iter2.first;
      code_printer.AddLine("TilingFunction " + GetTilingDataSettingFunc(graph_name) +
          "(const uint32_t key, const std::string &axis_name) {");
      code_printer.AddLine("  const auto iter = g_" + graph_name +
          "_tiling_data_setting_func.find(std::to_string(key) + axis_name);");
      code_printer.AddLine("  if (iter == g_" + graph_name + "_tiling_data_setting_func.cend()) {");
      code_printer.AddLine("    return nullptr;");
      code_printer.AddLine("  }");
      code_printer.AddLine("  return iter->second;");
      code_printer.AddLine("}");
    }
  }
}

void GenMainEntrance(CodePrinter &code_printer) {
  code_printer.AddLine("int main(int32_t argc, char *argv[]) {");
  code_printer.AddLine("  GE_ASSERT_TRUE(argc == 2, \"Get argc[%d] is invalid, valid range is 2.\", argc);");
  code_printer.AddLine("  // get json name");
  code_printer.AddLine("  const std::string input_file = argv[1];");
  code_printer.AddLine("  GE_ASSERT_TRUE(UserInputParser::CheckInputValid(input_file), \"Input file is invalid.\");");
  code_printer.AddLine("  const auto &file_context = UserInputParser::ReadFileToString(input_file);");
  code_printer.AddLine("  UserInput user_input;");
  code_printer.AddLine("  GE_ASSERT_SUCCESS(UserInputParser::FromJson(file_context, user_input),");
  code_printer.AddLine(
      "                     \"Parser user input failed, input_file[%s], file_context[%s].\", input_file.c_str(),"
      "                                file_context.c_str());");
}

void GenHardwareInfo(CodePrinter &code_printer) {
  code_printer.AddLine("      // TODO 用户可根据实际情况设置硬件的参数");
  code_printer.AddLine("      tiling_data.set_block_dim(64);");
  code_printer.AddLine("      tiling_data.set_ub_size(240 * 1024);");
  if (g_need_hbm) {
    code_printer.AddLine("      tiling_data.set_hbm_size(180 * 1024 * 1024);");
  }
}

void GenSetTilingData(const std::string &graph_name, CodePrinter &code_printer) {
  code_printer.AddLine("      for (const auto &axis : input_shape.axes) {");
  code_printer.AddLine("        const auto &axis_name = axis.first;");
  code_printer.AddLine("        const uint32_t axis_size = axis.second;");
  code_printer.AddLine("        const auto set_tiling_data = " + GetTilingDataSettingFunc(graph_name) +
      "(input_shape.tiling_key, axis_name);");
  code_printer.AddLine("        if (set_tiling_data != nullptr) {");
  code_printer.AddLine("          set_tiling_data(axis_size, tiling_data);");
  code_printer.AddLine("        } else {");
  code_printer.AddLine(
      "          GELOGE(ge::FAILED, \"Error:failed to get tiling data setting func, axis name[%s]\", axis_name.c_str());");
  code_printer.AddLine("          return -1;");
  code_printer.AddLine("        }");
  code_printer.AddLine("      }");
}

void GenGetTiling(CodePrinter &code_printer) {
  code_printer.AddLine("      if (optiling::GetTiling(tiling_data, input_shape.tiling_key)) {");
  code_printer.AddLine(
      "        GELOGI(\"Get tiling info successfully, you can print some data what you want to see.\");");
  code_printer.AddLine("        // TODO 用户可根据期望获取的数据，打印不同输出的值");
  code_printer.AddLine("        GELOGI(\"Get tiling data info of tiling case id [%u]\", input_shape.tiling_key);");
  code_printer.AddLine("      } else {");
  code_printer.AddLine(
      "        GELOGE(ge::FAILED, \"Error:failed to get tiling, tiling_case_id [%u]!\", input_shape.tiling_key);");
  code_printer.AddLine(
      "        return -1;");
  code_printer.AddLine("      }");
}

void GenMainFunc(const std::map<uint32_t, std::map<std::string, std::vector<std::string>>> &keys_to_axes_names,
                 CodePrinter &code_printer) {
  GenMainEntrance(code_printer);
  code_printer.AddLine("  for (const auto &input_shape : user_input.input_shapes) {");
  size_t tiling_id = keys_to_axes_names.size();
  for (const auto &iter1 : keys_to_axes_names) {
    const uint32_t key = iter1.first;
    if (tiling_id == keys_to_axes_names.size()) {
      code_printer.AddLine("    if (input_shape.tiling_key == " + std::to_string(key) + ") {");
    } else {
      code_printer.AddLine("    } else if (input_shape.tiling_key == " + std::to_string(key) + ") {");
    }
    for (const auto &iter2 : iter1.second) {
      const auto &graph_name = iter2.first;
      code_printer.AddLine("      " + GetTilingDataType(graph_name) + " tiling_data;");
      GenHardwareInfo(code_printer);
      GenSetTilingData(graph_name, code_printer);
      GenGetTiling(code_printer);
    }
    tiling_id--;
    if (tiling_id == 0UL) {
      code_printer.AddLine("    } else {");
      code_printer.AddLine("      GELOGE(ge::FAILED, \"Error:failed to get tiling case id[%u]!\", input_shape.tiling_key);");
      code_printer.AddLine("      return -1;");
      code_printer.AddLine("    }");
    }
  }
  code_printer.AddLine("  }");
  code_printer.AddLine("  return 0;");
  code_printer.AddLine("}");
}

void GenSetTilingAxisFunc(const std::map<uint32_t, std::map<std::string, std::vector<std::string>>> &keys_to_axes_names,
                          CodePrinter &code_printer) {
  int32_t count = 0;
  for (const auto &key_to_axes_names : keys_to_axes_names) {
    const auto key = key_to_axes_names.first;
    for (const auto &graph_name_to_axes_names : key_to_axes_names.second) {
      const auto &graph_name = graph_name_to_axes_names.first;
      for (const auto &axis_name : graph_name_to_axes_names.second) {
        const auto uniq_name = std::to_string(key) + axis_name;
        std::string func_define("void " + SettingFuncName(graph_name));
        func_define.append(uniq_name).append("(const uint32_t axis_size, " + GetTilingDataType(graph_name) +
            " &tiling_data) {");
        code_printer.AddLine(func_define);
        std::string func_impl("  tiling_data.set_" + axis_name + "(axis_size);");
        code_printer.AddLine(func_impl);
        code_printer.AddLine("}");
        count++;
      }
    }
  }
  for (const auto &key_to_axes_names : keys_to_axes_names) {
    const auto key = key_to_axes_names.first;
    for (const auto &graph_name_to_axes_names : key_to_axes_names.second) {
      const auto &graph_name = graph_name_to_axes_names.first;
      code_printer.AddLine("static std::map<std::string, TilingFunction> g_" + graph_name +
          "_tiling_data_setting_func = {");
      for (const auto &axis_name : graph_name_to_axes_names.second) {
        std::string uniq_name = std::to_string(key) + axis_name;
        std::string key_value_define("  {\"" + uniq_name + "\", " + SettingFuncName(graph_name) + uniq_name + "}");
        count--;
        if (count != 0) {
          key_value_define.append(",");
        }
        code_printer.AddLine(key_value_define);
      }
    }
  }
  code_printer.AddLine("};");
}

std::map<int64_t, std::string> GetIdsToVarNames(const ge::AscGraph &graph) {
  std::map<int64_t, std::string> id_to_var_name;
  for (const auto &var : graph.GetAllAxis()) {
    if ((var->type == ge::Axis::kAxisTypeOriginal)) {
      id_to_var_name[var->id] = att::GetSymbolName(var->size);
    }
  }
  return id_to_var_name;
}

std::map<uint32_t, std::map<std::string, std::vector<std::string>>> GetKeysToOriginalAxisNames(
    std::vector<ge::AscGraph> &graphs) {
  std::map<uint32_t, std::map<std::string, std::vector<std::string>>> keys_to_axes_names;
  for (auto &graph : graphs) {
    const auto tiling_key = graph.GetTilingKey();
    const std::map<int64_t, std::string> id_to_var_name = GetIdsToVarNames(graph);
    for (auto &axis_info : graph.GetAllAxis()) {
      if (axis_info->type == static_cast<int32_t>(ge::Axis::kAxisTypeOriginal)) {
        const auto &iter = id_to_var_name.find(axis_info->id);
        if (iter != id_to_var_name.cend()) {
          keys_to_axes_names[tiling_key][graph.GetName()].emplace_back(iter->second);
        }
      }
    }
  }
  return keys_to_axes_names;
}

bool NeedHbm(const ge::AscGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetType() == "Store") {
      auto output_size = node->GetOpDescBarePtr()->GetOutputsSize();
      auto *n = static_cast<ge::AscNode *>(&(*node));
      for (size_t i = 0UL; i < output_size; i++) {
        if (n->outputs[i].attr.mem.alloc_type == ge::AllocType::kAllocTypeBuffer ||
            n->outputs[i].attr.mem.alloc_type == ge::AllocType::kAllocTypeQueue) {
          return true;
        }
      }
    }
  }
  for (const auto &node : graph.GetInputNodes()) {
    const auto &output_tensor = node->outputs[0U];
    auto ir_attr = node->attr.ir_attr.get();
    int64_t index_value = kInvalidIndex;
    if (ir_attr->GetAttrValue("index", index_value) != ge::GRAPH_SUCCESS) {
      GELOGW("GetAttrValue index expr failed, node[%s] index_value[%ld]", node->GetName().c_str(), index_value);
    }
    if (static_cast<int32_t>(index_value) == kInvalidIndex) {
      continue;
    }
    auto output_size = node->GetOpDescBarePtr()->GetOutputsSize();
    auto *n = static_cast<ge::AscNode *>(&(*node));
    for (size_t i = 0UL; i < output_size; i++) {
      if (n->outputs[i].attr.mem.alloc_type == ge::AllocType::kAllocTypeBuffer ||
          n->outputs[i].attr.mem.alloc_type == ge::AllocType::kAllocTypeQueue) {
        return true;
      }
    }
  }
  return false;
}
}  // namespace

ge::Status GenExecTilingUtils::GenInputJson(std::vector<ge::AscGraph> &graphs, const std::string &output_path) {
  constexpr int32_t kAxisDefaultSize = 1;
  UserInput user_input;
  for (auto &graph : graphs) {
    InputShape input_shape;
    input_shape.tiling_key = graph.GetTilingKey();
    const std::map<int64_t, std::string> id_to_var_name = GetIdsToVarNames(graph);
    for (auto &axis_info : graph.GetAllAxis()) {
      if (axis_info->type == static_cast<int32_t>(ge::Axis::kAxisTypeOriginal)) {
        const auto &iter = id_to_var_name.find(axis_info->id);
        if (iter != id_to_var_name.cend()) {
          input_shape.axes.emplace_back(iter->second, kAxisDefaultSize);
        }
      }
    }
    user_input.input_shapes.emplace_back(input_shape);
  }
  const auto output_json = UserInputParser::ToJson(user_input);
  CodePrinter code_printer;
  code_printer.AddLine(output_json);
  constexpr char kInputJsonName[] = "input_shapes.json";
  code_printer.SaveToFile(output_path + kInputJsonName);
  return ge::SUCCESS;
}

ge::Status GenExecTilingUtils::GenExecFunc(std::vector<ge::AscGraph> &graphs,
                                       const std::string &op_name,
                                       const std::string &output_path) {
  CodePrinter code_printer;
  // TTODO only support one graph
  g_need_hbm = NeedHbm(graphs[0]);
  const auto &keys_to_axes_names = GetKeysToOriginalAxisNames(graphs);
  GenSTHead(op_name, keys_to_axes_names, code_printer);
  GenSetTilingAxisFunc(keys_to_axes_names, code_printer);
  GenGetSettingFunc(keys_to_axes_names, code_printer);
  GenMainFunc(keys_to_axes_names, code_printer);
  code_printer.SaveToFile(output_path + kTilingFuncStName);
  return ge::SUCCESS;
}

std::map<int32_t, std::pair<ge::DataType, ge::Format>> GetInputs(const ge::AscGraph &graph) {
  int32_t input_num = 0;
  std::map<int32_t, std::pair<ge::DataType, ge::Format>> input_dtype_and_format;
  for (const auto &node : graph.GetInputNodes()) {
    const auto &output_tensor = node->outputs[0U];
    auto ir_attr = node->attr.ir_attr.get();
    int64_t index_value = kInvalidIndex;
    if (ir_attr->GetAttrValue("index", index_value) != ge::GRAPH_SUCCESS) {
      GELOGW("GetAttrValue index expr failed, node[%s] index_value[%ld]", node->GetName().c_str(), index_value);
    }
    if (static_cast<int32_t>(index_value) == kInvalidIndex) {
      continue;
    }
    GELOGD("Got data node[%s] index_value[%ld]", node->GetName().c_str(), index_value);
    // fake type
    input_dtype_and_format[input_num] = {ge::DT_FLOAT16, ge::FORMAT_ND};
    input_num++;
  }
  return input_dtype_and_format;
}

int32_t GetOutputsNum(const ge::AscGraph &graph) {
  int32_t output_num = 0;
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetType() == "Output") {
      output_num++;
    }
  }
  return output_num;
}

std::string GetPrivateAttrs(const int32_t id) {
  std::string private_attr("      .AddPrivateAtt({\"");
  static std::map<std::string, std::string> data_type_to_any_value_type = {
      {"int8_t" , "int64_t" },
      {"int32_t" , "int64_t" },
      {"uint8_t" , "int64_t" },
      {"uint32_t" , "int64_t" },
      {"uint64_t" , "int64_t" }
  };
  std::string option_attr_name;
  std::string val("0");
  std::string data_type("int64_t");
  std::string attr_name("test" + std::to_string(id));
  private_attr.append(attr_name)
      .append("\"")
      .append(", ge::AnyValue::CreateFrom<")
      .append(data_type)
      .append(">(")
      .append(val)
      .append(")})");
  return private_attr;
}

void GenMainWithCtx(const ge::AscGraph &graph, CodePrinter &code_printer) {
  g_need_hbm = NeedHbm(graph);
  auto input_dtype_and_format = GetInputs(graph);
  int32_t output_num = GetOutputsNum(graph);
  code_printer.AddLine("int main(int32_t argc, char *argv[]) {");
  code_printer.AddLine("  KernelContextHolderBuilder builder;");
  code_printer.AddLine("  auto holder = builder");
  code_printer.AddLine("  // TODO 用户可以根据实际情况，修改输入的shape");
  for (const auto &input : input_dtype_and_format) {
    std::string input_str("      .AddInput(InOutput(ge::GeShape({1, 2, 3, 4, 5, 6, 7, 8}), ");
    input_str.append("ge::Format::FORMAT_" + ge::TypeUtils::FormatToSerialString(input.second.second)).append(", ");
    input_str.append("ge::DataType::" + ge::TypeUtils::DataTypeToSerialString(input.second.first)).append("))");
    code_printer.AddLine(input_str);
  }
  for (int32_t id = 0; id < output_num; id++) {
    std::string output_str("      .AddOutput(InOutput(ge::GeShape({1, 2, 3, 4, 5, 6, 7, 8}), ");
    output_str.append("ge::Format::FORMAT_ND, ");
    output_str.append("ge::DataType::DT_FLOAT16))");
    code_printer.AddLine(output_str);
  }
  code_printer.AddLine("      .SetTilingData(10240)");
  code_printer.AddLine("      .SetWorkSpace(1600)");
  code_printer.AddLine("      .SetCompileInfo(2)");
  code_printer.AddLine("      .SetPlatformInfo()");
  uint32_t max_id = 0U;
  for (uint32_t id = 0U; id < max_id; id++) {
    code_printer.AddLine(GetPrivateAttrs(id));
  }
  code_printer.AddLine("      .Build();");
  code_printer.AddLine("  auto *ctx = reinterpret_cast<gert::TilingContext *>(holder.context_);");
  code_printer.AddLine("  if (optiling::GetCtxTiling(ctx, -1) == ge::GRAPH_SUCCESS) {");
  code_printer.AddLine("    GELOGI(\"Get tiling info successfully, you can print some data what you want to see.\");");
  code_printer.AddLine("    // TODO 用户可根据期望获取的数据，打印不同输出的值");
  code_printer.AddLine("    GET_TILING_DATA(tmpTiling, ctx->GetRawTilingData()->GetData());");
  code_printer.AddLine("    PrintTilingData(tmpTiling);");
  code_printer.AddLine(
      R"(    std::cout << "Tiling func execute success, tiling_key=" << ctx->GetTilingKey() << std::endl;)");
  code_printer.AddLine("  } else {");
  code_printer.AddLine("    GELOGE(ge::FAILED, \"Error:failed to get tiling!\");");
  code_printer.AddLine("    return -1;");
  code_printer.AddLine("  }");
  code_printer.AddLine("  return 0;");
  code_printer.AddLine("}");
}

ge::Status GenExecTilingUtils::GenContextEntranceExecFunc(std::vector<ge::AscGraph> &graphs,
                                                      const std::string &op_name,
                                                      const std::string &output_path) {
  CodePrinter code_printer;
  GenSTHead(op_name, {}, code_printer);
  GE_ASSERT_TRUE(graphs.size() <= 1,
                 "Gen execute tiling func with tiling context only support one AscendGraph, graphs[%zu].",
                 graphs.size());
  GenMainWithCtx(graphs.front(), code_printer);
  code_printer.SaveToFile(output_path + kTilingFuncStName);
  return ge::SUCCESS;
}

ge::Status GenExecTilingUtils::GenExecFunc(std::vector<ge::AscGraph> &graphs, const InputArgs &input_args) {
  if (input_args.tiling_entrance == kTilingDataEntrance) {
    GE_ASSERT_SUCCESS(GenExecTilingUtils::GenInputJson(graphs, input_args.output_path),
                      "Gen input json failed, output_path[%s].", input_args.output_path.c_str());
    GE_ASSERT_SUCCESS(GenExecTilingUtils::GenExecFunc(graphs, input_args.op_name, input_args.output_path),
                      "Gen tiling st failed, op_name[%s], output_path[%s].", input_args.op_name.c_str(),
                      input_args.output_path.c_str());
  } else {
    GE_ASSERT_SUCCESS(
        GenExecTilingUtils::GenContextEntranceExecFunc(graphs, input_args.op_name, input_args.output_path),
        "Gen execute func with tiling context failed, output_path[%s].", input_args.output_path.c_str());
  }
  return ge::SUCCESS;
}
}  // namespace att
