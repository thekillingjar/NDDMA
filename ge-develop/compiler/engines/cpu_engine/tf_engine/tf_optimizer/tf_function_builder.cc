/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tf_function_builder.h"

#include <map>
#include <atomic>
#include "graph/utils/graph_utils.h"
#include "util/constant.h"
#include "util/log.h"
#include "util/tf_util.h"
#include "error_code/error_code.h"
#include "tensorflow_util.h"
#include "ir2tf/ir2tf_parser_factory.h"

using domi::tensorflow::OpDef;
using domi::tensorflow::NodeDef;
using domi::tensorflow::FunctionDef;
using domi::tensorflow::FunctionDefLibrary;

namespace {
const std::string kArgOp = "Arg";
const std::string kRetOp = "Retval";
}

namespace aicpu {
TfFunctionBuilder& TfFunctionBuilder::GetInstance() {
  static TfFunctionBuilder instance;
  return instance;
}

ge::Status TfFunctionBuilder::BuildFunctionDef(ge::ComputeGraphPtr &graph,
                                               const std::string &func_name,
                                               FunctionDefLibrary *func_lib,
                                               NodeDef *fused_node_def,
                                               std::vector<ge::InDataAnchorPtr> &in_anchors,
                                               std::vector<ge::OutDataAnchorPtr> &out_anchors) const {
  std::string current_time = CurrentTimeInStr();
  static std::atomic<int32_t> index(0);
  std::string name;
  name.append(func_name).append(current_time).append("_").append(Stringcat(index.fetch_add(1)));

  // set node def
  AICPU_CHECK_NOTNULL(fused_node_def);
  fused_node_def->set_op(name);
  fused_node_def->set_name(name);

  // add func attr
  domi::tensorflow::AttrValue value;
  domi::tensorflow::NameAttrList *function = value.mutable_func();
  function->set_name(name);
  *function->mutable_attr() = fused_node_def->attr();
  AICPU_CHECK_FALSE_EXEC(TensorFlowUtil::AddNodeAttr("function", value, fused_node_def),
      AICPU_REPORT_INNER_ERR_MSG("Add attr[function] failed, op[%s].",
          name.c_str());
      return ErrorCode::ADD_ATTR_FAILED)

  // set input for node def
  SetInputForNodeDef(fused_node_def, in_anchors);

  // create input and output node for graph
  std::vector<TFDataType> arg_data_types;
  std::vector<TFDataType> res_data_types;
  AICPU_CHECK_RES_WITH_LOG(CreateInputNodes(graph, in_anchors, arg_data_types),
      "Call TfFunctionBuilder::CreateInputNodes function failed, tf function[%s].",
      func_name.c_str())
  AICPU_CHECK_RES_WITH_LOG(CreateOutputNodes(graph, out_anchors, res_data_types),
      "Call TfFunctionBuilder::CreateOutputNodes function failed, tf function[%s].",
      func_name.c_str())

  // add tin attr
  if (!arg_data_types.empty()) {
    domi::tensorflow::AttrValue t_in_value;
    for (const TFDataType &type : arg_data_types) {
      t_in_value.mutable_list()->add_type(type);
    }
    AICPU_CHECK_FALSE_EXEC(TensorFlowUtil::AddNodeAttr("Tin", t_in_value, fused_node_def),
        AICPU_REPORT_INNER_ERR_MSG("Add attr[Tin] failed, op[%s] .",
            name.c_str());
        return ErrorCode::ADD_ATTR_FAILED)
  }

  // add tout attr
  if (!res_data_types.empty()) {
    domi::tensorflow::AttrValue t_out_value;
    for (const TFDataType &type : res_data_types) {
      t_out_value.mutable_list()->add_type(type);
    }
    AICPU_CHECK_FALSE_EXEC(TensorFlowUtil::AddNodeAttr("Tout", t_out_value, fused_node_def),
        AICPU_REPORT_INNER_ERR_MSG("Add attr[Tout] failed, op[%s].",
            name.c_str());
        return ErrorCode::ADD_ATTR_FAILED)
  }

  // trans ge graph to tf function def
  FunctionDef *func_def = func_lib->add_function();
  AICPU_CHECK_RES_WITH_LOG(TransGraphToFunctionDef(graph, name, func_def),
      "Call TfFunctionBuilder::TransGraphToFunctionDef. tf function[%s].",
      func_name.c_str())

  return ge::SUCCESS;
}

void TfFunctionBuilder::SetInputForNodeDef(NodeDef *node_def, std::vector<ge::InDataAnchorPtr> &in_anchors) const {
  for (const ge::InDataAnchorPtr &anchor : in_anchors) {
    ge::OutDataAnchorPtr peer_out_anchor = anchor->GetPeerOutAnchor();
    if (peer_out_anchor != nullptr) {
      std::string input_str;
      input_str.append(peer_out_anchor->GetOwnerNode()->GetName()).append("_")
               .append(Stringcat(peer_out_anchor->GetIdx()));
      node_def->add_input(input_str);
    }
  }
}

ge::Status TfFunctionBuilder::CreateInputNodes(ge::ComputeGraphPtr &graph,
                                               const std::vector<ge::InDataAnchorPtr> &in_anchors,
                                               std::vector<TFDataType> &arg_data_types) const {
  AICPU_CHECK_NOTNULL(graph);
  int32_t index = 0;
  for (const ge::InDataAnchorPtr &anchor : in_anchors) {
    // generate op name
    std::string op_name;
    op_name.append(anchor->GetPeerOutAnchor()->GetOwnerNode()->GetName()).append("_")
           .append(Stringcat(anchor->GetPeerOutAnchor()->GetIdx()))
           .append("_index").append(Stringcat(index)).append("_arg");

    // create op desc
    ge::OpDescPtr op_desc = nullptr;
    AICPU_MAKE_SHARED(op_desc = std::make_shared<ge::OpDesc>(op_name, kArgOp),
        AICPU_REPORT_INNER_ERR_MSG("Create ge::OpDesc object failed, op[%s].",
            op_name.c_str());
        return ErrorCode::MEMORY_ALLOC_FAILED);
    AICPU_IF_BOOL_EXEC(op_desc == nullptr,
        AICPU_REPORT_INNER_ERR_MSG("Create ge::OpDesc object failed, op[%s].",
            op_name.c_str());
        return ErrorCode::MEMORY_ALLOC_FAILED)

    // get and convert tf data type
    ge::ConstGeTensorDescPtr tensor_desc_ptr = anchor->GetOwnerNode()->GetOpDesc()->GetInputDescPtr(anchor->GetIdx());
    AICPU_CHECK_NOTNULL(tensor_desc_ptr)
    ge::DataType data_type = tensor_desc_ptr->GetDataType();
    TFDataType tf_data_type = ConvertGeDataType2TfDataType(tensor_desc_ptr->GetDataType());
    AICPU_IF_BOOL_EXEC(tf_data_type == TFDataType::DT_INVALID,
        AICPU_REPORT_INNER_ERR_MSG("Ge data type[%d] not supported, op[%s]",
            data_type, op_name.c_str());
        return ErrorCode::DATA_TYPE_UNDEFILED)

    // add default output desc to op
    AICPU_CHECK_FALSE_EXEC(op_desc->AddOutputDesc(ge::GeTensorDesc()) == ge::GRAPH_SUCCESS,
        AICPU_REPORT_INNER_ERR_MSG("Call ge::OpDesc::AddOutputDesc failed, op[%s].",
            op_name.c_str());
        return ErrorCode::ADD_TF_DATADESC_FAILED)
    // set T and arg_index attribute to op
    AICPU_CHECK_FALSE_EXEC(ge::AttrUtils::SetInt(op_desc, "T", static_cast<int32_t>(tf_data_type)),
        AICPU_REPORT_INNER_ERR_MSG(
            "Call ge::AttrUtils::SetInt failed to set attr[T], op[%s].",
            op_name.c_str());
        return ErrorCode::ADD_ATTR_FAILED)
    AICPU_CHECK_FALSE_EXEC(ge::AttrUtils::SetInt(op_desc, "arg_index", static_cast<int32_t>(index)),
        AICPU_REPORT_INNER_ERR_MSG(
            "Call ge::AttrUtils::SetInt failed to set attr[arg_index], op[%s].",
            op_name.c_str());
        return ErrorCode::ADD_ATTR_FAILED)

    // create ge node
    ge::NodePtr arg_node = graph->AddNode(op_desc);
    AICPU_CHECK_NOTNULL(arg_node)

    // remove edge and create new edge.
    for (const ge::NodePtr &node : graph->GetAllNodes()) {
      if (node->GetName() == anchor->GetOwnerNode()->GetName()) {
        ge::OutDataAnchorPtr new_peer_out_anchor = arg_node->GetOutDataAnchor(0);
        // in_anchor is not anchor above. anchor above belong to old node in original graph
        // but here anchor should belong to new node in new graph
        ge::InDataAnchorPtr in_anchor = node->GetInDataAnchor(anchor->GetIdx());
        // anchor of new node has no relation with anchor of old node
        AICPU_CHECK_FALSE_EXEC(ge::GraphUtils::AddEdge(new_peer_out_anchor, in_anchor) == ge::GRAPH_SUCCESS,
            AICPU_REPORT_INNER_ERR_MSG(
                "Call ge::GraphUtils::AddEdge failed, op[%s], peer op[%s].",
                node->GetName().c_str(),
                new_peer_out_anchor->GetOwnerNode()->GetName().c_str());
            return ErrorCode::ADD_EDGE_FAILED)
        break;
      }
    }
    // record input data type
    arg_data_types.emplace_back(tf_data_type);
    index++;
  }
  return ge::SUCCESS;
}


ge::Status TfFunctionBuilder::CreateOutputNodes(ge::ComputeGraphPtr &graph,
                                                const std::vector<ge::OutDataAnchorPtr> &out_anchors,
                                                std::vector<TFDataType> &res_data_types) const {
  AICPU_CHECK_NOTNULL(graph);
  int32_t index = 0;
  for (const ge::OutDataAnchorPtr &anchor : out_anchors) {
    // generate op desc
    std::string op_name;
    op_name.append(anchor->GetOwnerNode()->GetName()).append("_")
           .append(Stringcat(anchor->GetIdx())).append("_retval");

    // create op desc
    ge::OpDescPtr op_desc = nullptr;
    AICPU_MAKE_SHARED(op_desc = std::make_shared<ge::OpDesc>(op_name, kRetOp),
        AICPU_REPORT_INNER_ERR_MSG("Create ge::OpDesc object failed, op[%s]",
            op_name.c_str());
        return ErrorCode::MEMORY_ALLOC_FAILED)
    AICPU_IF_BOOL_EXEC(op_desc == nullptr,
        AICPU_REPORT_INNER_ERR_MSG("Create ge::OpDesc object failed, op[%s]",
            op_name.c_str());
        return ErrorCode::MEMORY_ALLOC_FAILED)

    // get and convert tf data type
    auto tensor_desc_ptr = anchor->GetOwnerNode()->GetOpDesc()->GetOutputDescPtr(anchor->GetIdx());
    AICPU_CHECK_NOTNULL(tensor_desc_ptr)
    ge::DataType data_type = tensor_desc_ptr->GetDataType();
    TFDataType tf_data_type = ConvertGeDataType2TfDataType(tensor_desc_ptr->GetDataType());
    if (tf_data_type == TFDataType::DT_INVALID) {
      AICPU_REPORT_INNER_ERR_MSG("Ge data type[%d] not supported, op[%s]",
          data_type, op_name.c_str());
      return ErrorCode::DATA_TYPE_UNDEFILED;
    }

    // add default input and output desc to op
    AICPU_CHECK_FALSE_EXEC(op_desc->AddInputDesc(ge::GeTensorDesc()) == ge::GRAPH_SUCCESS,
        AICPU_REPORT_INNER_ERR_MSG("Call ge::OpDesc::AddInputDesc failed, op[%s].",
            op_name.c_str());
        return ErrorCode::ADD_TF_DATADESC_FAILED)
    AICPU_CHECK_FALSE_EXEC(op_desc->AddOutputDesc(ge::GeTensorDesc()) == ge::GRAPH_SUCCESS,
        AICPU_REPORT_INNER_ERR_MSG("Call ge::OpDesc::AddOutputDesc failed, op[%s].",
            op_name.c_str());
        return ErrorCode::ADD_TF_DATADESC_FAILED)
    // set T and arg_index attribute to op
    AICPU_CHECK_FALSE_EXEC(ge::AttrUtils::SetInt(op_desc, "T", static_cast<int32_t>(tf_data_type)),
        AICPU_REPORT_INNER_ERR_MSG(
            "Call ge::AttrUtils::SetInt failed to set attr[T], op[%s].",
            op_name.c_str());
        return ErrorCode::ADD_ATTR_FAILED)
    AICPU_CHECK_FALSE_EXEC(ge::AttrUtils::SetInt(op_desc, "retval_index", static_cast<int32_t>(index)),
        AICPU_REPORT_INNER_ERR_MSG(
            "Call ge::AttrUtils::SetInt failed to set attr[retval_index], op[%s].",
            op_name.c_str());
        return ErrorCode::ADD_ATTR_FAILED)

    // create ge node
    ge::NodePtr res_node = graph->AddNode(op_desc);
    AICPU_CHECK_NOTNULL(res_node)

    // remove edge and create new edge
    for (const ge::NodePtr &node : graph->GetAllNodes()) {
      if (node->GetName() == anchor->GetOwnerNode()->GetName()) {
        // out_anchor is not the variable anchor
        ge::OutDataAnchorPtr out_anchor = node->GetOutDataAnchor(anchor->GetIdx());
        ge::InDataAnchorPtr peer_in_anchor = res_node->GetInDataAnchor(0);

        AICPU_CHECK_FALSE_EXEC(ge::GraphUtils::AddEdge(out_anchor, peer_in_anchor) == ge::GRAPH_SUCCESS,
            AICPU_REPORT_INNER_ERR_MSG(
                "Call ge::GraphUtils::AddEdge failed, op[%s], peer op[%s].",
                node->GetName().c_str(), res_node->GetName().c_str());
            return ErrorCode::RMV_EDGE_FAILED)
        break;
      }
    }
    res_data_types.emplace_back(tf_data_type);
    index++;
  }
  return ge::SUCCESS;
}

ge::Status TfFunctionBuilder::TransGraphToFunctionDef(ge::ComputeGraphPtr &graph, const std::string &func_name,
                                                      FunctionDef *func_def) const {
  AICPU_CHECK_NOTNULL(graph)
  AICPU_CHECK_NOTNULL(func_def)
  func_def->mutable_signature()->set_name(func_name);

  std::unordered_map<std::string, std::string> tensor_renaming_map;
  std::unordered_map<std::string, std::string> return_value_map;

  for (const ge::NodePtr &node : graph->GetAllNodes()) {
    AICPU_CHECK_NOTNULL(node)
    AICPU_CHECK_NOTNULL(node->GetOpDesc())
    std::string node_name = node->GetName();
    if (node->GetOpDesc()->GetType() == kArgOp) {
      int64_t index = 0;
      int64_t type = 1;
      AICPU_CHECK_FALSE_EXEC(ge::AttrUtils::GetInt(node->GetOpDesc(), "T", type),
          AICPU_REPORT_INNER_ERR_MSG("Call ge::AttrUtils::GetInt failed to get attr[T], op[%s].", node_name.c_str());
          return ErrorCode::GET_ATTR_FAILED)
      AICPU_CHECK_FALSE_EXEC(ge::AttrUtils::GetInt(node->GetOpDesc(), "arg_index", index),
          AICPU_REPORT_INNER_ERR_MSG("Call ge::AttrUtils::GetInt failed to get attr[arg_index], op[%s].",
              node_name.c_str()); return ErrorCode::GET_ATTR_FAILED)
      while (func_def->signature().input_arg_size() <= index) {
        func_def->mutable_signature()->add_input_arg();
      }
      OpDef::ArgDef* argdef = func_def->mutable_signature()->mutable_input_arg(index);
      argdef->clear_type_attr();
      argdef->clear_number_attr();
      argdef->clear_type_list_attr();
      argdef->set_type(TFDataType(type));
      argdef->set_name(node_name);
      tensor_renaming_map[node_name + ":0"] = node_name;
      continue;
    }

    if (node->GetOpDesc()->GetType() == kRetOp) {
      int64_t index = 0;
      int64_t type = 1;
      AICPU_CHECK_FALSE_EXEC(ge::AttrUtils::GetInt(node->GetOpDesc(), "T", type),
          AICPU_REPORT_INNER_ERR_MSG("Call ge::AttrUtils::GetInt failed to get attr[T], op[%s].", node_name.c_str());
          return ErrorCode::GET_ATTR_FAILED)
      AICPU_CHECK_FALSE_EXEC(ge::AttrUtils::GetInt(node->GetOpDesc(), "retval_index", index),
          AICPU_REPORT_INNER_ERR_MSG("Call ge::AttrUtils::GetInt get attr[retval_index], op[%s].", node_name.c_str());
          return ErrorCode::GET_ATTR_FAILED)
      while (func_def->signature().output_arg_size() <= index) {
        func_def->mutable_signature()->add_output_arg();
      }
      OpDef::ArgDef* argdef = func_def->mutable_signature()->mutable_output_arg(index);
      argdef->set_type(TFDataType(type));
      argdef->set_name(node_name);

      ge::OutDataAnchorPtr out_anchor = node->GetAllInDataAnchors().at(0)->GetPeerOutAnchor();
      std::string output_name;
      output_name.append(out_anchor->GetOwnerNode()->GetName()).append(":").append(Stringcat(out_anchor->GetIdx()));
      return_value_map[node_name] = output_name;
      continue;
    }

    AICPUE_LOGI("Current op[%s], op type[%s].", node->GetName().c_str(), node->GetType().c_str());
    // get tf node def
    bool node_def_from_ir_mapping = false;
    NodeDef original_node_def;
    AICPU_CHECK_RES_WITH_LOG(GetTfNodeDef(node, original_node_def, node_def_from_ir_mapping),
        "call TfFunctionBuilder::GetTfNodeDef function failed. op[%s]", node->GetName().c_str())

    // add node def to function
    NodeDef* node_def = func_def->add_node_def();
    *node_def = original_node_def;

    node_def->mutable_attr()->erase(kTfOpDef);
    node_def->mutable_attr()->erase(kInputTensorDesc);
    node_def->mutable_attr()->erase(kOutputTensorDesc);
    node_def->clear_device();
    node_def->set_name(node_name);

    // Reset input names based on graph rather than the NodeDef.
    node_def->clear_input();

    // Anchors, indexed by index.
    std::vector<ge::InDataAnchorPtr> in_data_anchors;
    for (const ge::InDataAnchorPtr &anchor : node->GetAllInDataAnchors()) {
      if (in_data_anchors.size() <= static_cast<size_t>(anchor->GetIdx())) {
        in_data_anchors.resize(anchor->GetIdx() + 1);
      }
      in_data_anchors[anchor->GetIdx()] = anchor;
    }

    // Add regular inputs
    for (const ge::InDataAnchorPtr &anchor : in_data_anchors) {
      if (anchor->GetPeerOutAnchor() != nullptr) {
        std::string tensor_name;
        tensor_name.append(anchor->GetPeerOutAnchor()->GetOwnerNode()->GetName()).append(":")
                   .append(Stringcat(anchor->GetPeerOutAnchor()->GetIdx()));
        node_def->add_input(tensor_name);
      }
    }

    // Add control inputs
    for (const ge::OutControlAnchorPtr &anchor : node->GetInControlAnchor()->GetPeerOutControlAnchors()) {
      node_def->add_input("^" + anchor->GetOwnerNode()->GetName());
    }

    // Populate tensor_renaming.
    NameRangeMap output_ranges;
    AICPU_CHECK_RES_WITH_LOG(GetTfOutputNameRangeMap(node, original_node_def, output_ranges, node_def_from_ir_mapping),
        "Call TfFunctionBuilder::GetTfOutputNameRangeMap function failed,"
        " op[%s], op type[%s].", node_name.c_str(), node->GetType().c_str())
    for (const auto &output : output_ranges) {
      for (int index = output.second.first; index < output.second.second; ++index) {
        std::string tensor_name;
        tensor_name.append(node_name).append(":").append(output.first).append(":")
                   .append(Stringcat(index - output.second.first));
        std::string ge_tensor_name;
        ge_tensor_name.append(node_name).append(":").append(Stringcat(index));
        tensor_renaming_map[ge_tensor_name] = tensor_name;
      }
    }
  }

  // remap FunctionDef
  AICPU_CHECK_RES_WITH_LOG(RemapFunctionDef(func_def, func_name, tensor_renaming_map, return_value_map),
      "Call TfFunctionBuilder::RemapFunctionDef function failed, tf function[%s].", func_name.c_str())
  return ge::SUCCESS;
}

ge::Status TfFunctionBuilder::NameRangesForNode(const NodeDef &node_def, const OpDef &op_def,
                                                NameRangeMap &outputs) const {
  int start = 0;
  int num = 0;
  for (const auto &arg : op_def.output_arg()) {
    ComputeArgRange(node_def, arg, &num);
    outputs[arg.name()] = std::make_pair(start, start + num);
    start += num;
  }
  return ge::SUCCESS;
}

ge::Status TfFunctionBuilder::ComputeArgRange(const NodeDef &node_def, const OpDef::ArgDef &arg_def, int *num) const {
  AICPU_CHECK_NOTNULL(num);
  if (!arg_def.number_attr().empty()) {
    domi::tensorflow::AttrValue attr_value;
    if (!TensorFlowUtil::FindAttrValue(&node_def, arg_def.number_attr(), attr_value)) {
      AICPU_REPORT_INNER_ERR_MSG("Attr[%s] not exist in node def, op[%s].",
          arg_def.number_attr().c_str(), node_def.name().c_str());
      return ErrorCode::GET_ATTR_FAILED;
    }
    *num = attr_value.i();
  } else if (!arg_def.type_list_attr().empty()) {
    domi::tensorflow::AttrValue attr_value;
    if (!TensorFlowUtil::FindAttrValue(&node_def, arg_def.type_list_attr(), attr_value)) {
      AICPU_REPORT_INNER_ERR_MSG("Attr[%s] not exist in node def, op[%s].",
          arg_def.type_list_attr().c_str(), node_def.name().c_str());
      return ErrorCode::GET_ATTR_FAILED;
    }
    *num = attr_value.list().type_size();
  } else if ((!arg_def.type_attr().empty()) || (arg_def.type() != TFDataType::DT_INVALID)) {
    *num = 1;
  } else {
    AICPU_REPORT_INNER_ERR_MSG("Attr[%s] not exist in node def, op[%s].",
        arg_def.type_list_attr().c_str(), node_def.name().c_str());
    return ErrorCode::PARSE_NODE_DEF_FAILED;
  }
  return ge::SUCCESS;
}

ge::Status TfFunctionBuilder::RemapFunctionDef(FunctionDef *func_def,
                                               const std::string &func_name,
                                               std::unordered_map<std::string, std::string> &tensor_renaming_map,
                                               std::unordered_map<std::string, std::string> &return_value_map) const {
  // Detect missing function inputs.
  for (int index = 0; index < func_def->signature().input_arg_size(); ++index) {
    const std::string &input_name = func_def->signature().input_arg(index).name();
    if (input_name.empty()) {
      AICPU_REPORT_INNER_ERR_MSG("%dth input name is empty in func_def[%s].",
          index, func_def->signature().name().c_str());
      return ErrorCode::REMAP_FUNCTIONDEF_FAILED;
    }
  }

  // Remap input names.  We do this as a second pass to allow the nodes to be in any order.
  for (int index = 0; index < func_def->node_def_size(); ++index) {
    NodeDef* node_def = func_def->mutable_node_def(index);
    for (int i = 0; i < node_def->input_size(); ++i) {
      if (node_def->input(i).find("^") != std::string::npos) {
        // Control input
        const std::string input_ctrl_name = node_def->input(i).substr(1);
        if (input_ctrl_name.empty()) {
          AICPU_REPORT_INNER_ERR_MSG(
              "Could not remap control input[%s] of node[%s] in function[%s].",
              node_def->input(i).c_str(), node_def->name().c_str(), func_name.c_str());
          return ErrorCode::REMAP_FUNCTIONDEF_FAILED;
        }
        *node_def->mutable_input(i) = "^" + input_ctrl_name;
      } else {
        const auto iter = tensor_renaming_map.find(node_def->input(i));
        if (iter == tensor_renaming_map.end()) {
          AICPU_REPORT_INNER_ERR_MSG(
              "Could not remap input[%s] of node[%s] in function[%s].",
              node_def->input(i).c_str(),
              node_def->name().c_str(), func_name.c_str());
          return ErrorCode::REMAP_FUNCTIONDEF_FAILED;
        }
        *node_def->mutable_input(i) = iter->second;
      }
    }
  }

  // Remap return values.
  for (int index = 0; index < func_def->signature().output_arg_size(); ++index) {
    const std::string &ret_name = func_def->signature().output_arg(index).name();
    if (ret_name.empty()) {
      AICPU_REPORT_INNER_ERR_MSG("Missing output[%d] to function[%s].",
          index, func_name.c_str());
      return ErrorCode::REMAP_FUNCTIONDEF_FAILED;
    }
    const std::string &return_value = return_value_map[ret_name];
    if (return_value.empty()) {
      AICPU_REPORT_INNER_ERR_MSG(
          "Could not remap return value[%d], [%s] of [%s] in function[%s].",
          index, ret_name.c_str(), return_value.c_str(), func_name.c_str());
      return ErrorCode::REMAP_FUNCTIONDEF_FAILED;
    }

    const auto iter = tensor_renaming_map.find(return_value);
    if (iter == tensor_renaming_map.end()) {
      AICPU_REPORT_INNER_ERR_MSG(
          "Could not find value[%s] in tensorRenaming map, function[%s].",
          return_value.c_str(), func_name.c_str());
      return ErrorCode::REMAP_FUNCTIONDEF_FAILED;
    }
    (*func_def->mutable_ret())[ret_name] = iter->second;
  }
  return ge::SUCCESS;
}

ge::Status TfFunctionBuilder::GetTfNodeDef(const ge::NodePtr &node, NodeDef &tf_node_def, bool &from_ir_mapping) const {
  from_ir_mapping = false;
  std::string node_name = node->GetName();
  ge::Buffer node_def_bytes;
  // TF_NODE_DEF has exist.
  if (ge::AttrUtils::GetBytes(node->GetOpDesc(), kTfNodeDef, node_def_bytes)) {
    AICPU_IF_BOOL_EXEC(node_def_bytes.GetSize() == 0,
        AICPU_REPORT_INNER_ERR_MSG("Size of [%s] is out of range.",
            node->GetName().c_str());
        return ErrorCode::PARSE_NODE_DEF_FAILED)
    AICPU_CHECK_FALSE_EXEC(tf_node_def.ParseFromArray(node_def_bytes.GetData(), node_def_bytes.GetSize()),
        AICPU_REPORT_INNER_ERR_MSG("Parse node def failed, op[%s], op type[%s].",
            node_name.c_str(), node->GetType().c_str());
        return ErrorCode::PARSE_NODE_DEF_FAILED)
    return ge::SUCCESS;
  }

  AICPUE_LOGI("Tf node def attr is not exist in op[%s], op type[%s].", node_name.c_str(), node->GetType().c_str());
  // IR->TF
  from_ir_mapping = true;
  std::shared_ptr<Ir2tfBaseParser> parser = Ir2tfParserFactory::Instance().CreateIRParser(node->GetOpDesc()->GetType());
  if (parser == nullptr) {
    AICPU_REPORT_INNER_ERR_MSG("Create ir parser failed, op[%s], op type[%s].",
        node_name.c_str(), node->GetOpDesc()->GetType().c_str());
    return ErrorCode::GET_IR2TF_PARSER_FAILED;
  }
  AICPU_CHECK_RES_WITH_LOG(parser->ParseNodeDef(*node, &tf_node_def),
      "Call Ir2tfBaseParser::ParseNodeDef function failed, op[%s], op type[%s].",
      node_name.c_str(), node->GetType().c_str());
  return ge::SUCCESS;
}

ge::Status TfFunctionBuilder::GetTfOutputNameRangeMap(const ge::NodePtr &node, NodeDef &tf_node_def,
                                                      NameRangeMap &outputs, bool &node_def_from_ir_mapping) const {
  const std::string node_name = node->GetName();
  const std::string *op_def_str = ge::AttrUtils::GetStr(node->GetOpDesc(), kTfOpDef);
  // only opdef and nodedef both from tensorflow
  if ((op_def_str != nullptr) && (!node_def_from_ir_mapping)) {
    OpDef op_def;
    if (!op_def.ParseFromString(*op_def_str)) {
      AICPU_REPORT_INNER_ERR_MSG(
          "Parse op def from string failed. op[%s], op type[%s].",
          node_name.c_str(), node->GetType().c_str());
      return ErrorCode::PARSE_NODE_DEF_FAILED;
    }
    AICPU_CHECK_RES_WITH_LOG(NameRangesForNode(tf_node_def, op_def, outputs),
        "Call TfFunctionBuilder::NameRangesForNode function failed, "
        "op[%s], op type[%s].", node_name.c_str(), node->GetType().c_str())
    return ge::SUCCESS;
  }
  AICPUE_LOGI("Tf op def attr is not exist in op[%s], op type[%s].", node_name.c_str(), node->GetType().c_str());

  // Function op
  ge::OpDescPtr op_desc = node->GetOpDesc();
  if (op_desc->GetType() == kFunctionOp) {
    AICPU_CHECK_RES_WITH_LOG(NameRangesForFunc(node, tf_node_def, outputs),
        "Call TfFunctionBuilder::NameRangesForFunc function failed,"
        " op[%s], op type[%s].", node_name.c_str(), node->GetType().c_str())
    return ge::SUCCESS;
  }

  // IR->TF
  std::shared_ptr<Ir2tfBaseParser> parser = Ir2tfParserFactory::Instance().CreateIRParser(op_desc->GetType());
  if (parser == nullptr) {
    AICPU_REPORT_INNER_ERR_MSG("Create ir parser failed. op[%s].",
        node_name.c_str());
    return ErrorCode::GET_IR2TF_PARSER_FAILED;
  }

  AICPU_CHECK_RES_WITH_LOG(parser->ParseOutputArgs(node, op_desc->GetType(), outputs),
      "Call Ir2tfBaseParser::ParseOutputArgs function failed, op[%s], op type[%s].",
      node_name.c_str(), node->GetType().c_str())
  return ge::SUCCESS;
}

ge::Status TfFunctionBuilder::NameRangesForFunc(const ge::NodePtr &node, NodeDef &tf_node_def,
                                                NameRangeMap &outputs) const {
  std::string node_name = node->GetName();
  ge::Buffer func_def_bytes;
  AICPU_CHECK_FALSE_EXEC(ge::AttrUtils::GetBytes(node->GetOpDesc(), kTfFuncDef, func_def_bytes),
                         AICPUE_LOGI("function def attr is not exist in ge op[%s], op type[%s].",
                         node_name.c_str(), node->GetType().c_str());
                         return ErrorCode::FUNC_DEF_NOT_EXIST)
  AICPU_IF_BOOL_EXEC(func_def_bytes.GetSize() == 0,
      AICPU_REPORT_INNER_ERR_MSG("Size of [%s] is out of range.",
          node->GetName().c_str());
      return ErrorCode::PARSE_FUNC_DEF_FAILED)

  FunctionDefLibrary func_def_lib;
  AICPU_CHECK_FALSE_EXEC(func_def_lib.ParseFromArray(func_def_bytes.GetData(), func_def_bytes.GetSize()),
      AICPU_REPORT_INNER_ERR_MSG(
          "Parse function def library failed from array, op[%s], op type[%s].",
          node_name.c_str(), node->GetType().c_str());
      return ErrorCode::PARSE_FUNC_DEF_FAILED)

  int start = 0;
  constexpr int num = 1;
  for (const auto &func : func_def_lib.function()) {
    if (func.signature().name() != tf_node_def.name()) {
      continue;
    }
    // List is not supported now, num will get from function later.
    for (const auto &arg : func.signature().output_arg()) {
      outputs[arg.name()] = std::make_pair(start, start + num);
      start += num;
    }
    break;
  }
  return ge::SUCCESS;
}
} // namespace aicpu
