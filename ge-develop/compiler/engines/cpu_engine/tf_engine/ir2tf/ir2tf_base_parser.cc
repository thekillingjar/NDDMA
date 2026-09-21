/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir2tf_base_parser.h"
#include <string>
#include <map>
#include <mutex>
#include <cctype>
#include <nlohmann/json.hpp>
#include "graph/node.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/operator.h"
#include "util/tf_util.h"
#include "common/util/log.h"
#include "common/util/constant.h"
#include "common/error_code/error_code.h"
#include "config/ir2tf_json_file.h"
#include "base/err_msg.h"

namespace {
using AttrValueMap = google::protobuf::Map<std::string, domi::tensorflow::AttrValue>;
/**
 * filter attrs which in attr_whitelist.
 */
const std::set<std::string> kConstAttrBlacklist = {
    "workspace_memsize",
    "workspace_alignment",
    "output_memsize_vector",
    "output_alignment_vector",
    "original_type",
    "FrameworkOp",
    "node_def",
    "framework_type",
    "t_in_datatype",
    "t_out_datatype",
    "is_input_const",
    "format",
    "inferred_format",
    "net_output_datatype",
    "net_output_format",
    "isCheckSupported",
    "op_def",
    "original_op_names",
    "imply_type",
    "_coretype",
    "_ge_attr_op_kernel_lib_name",
    "_is_unknown_shape",
    "_aicpu_unknown_shape",
    "_lxfusion_engine_name",
    "_lxfusion_op_kernel_lib_name",
    "_unknown_shape",
    "OwnerGraphIsUnknown",
    aicpu::kAicpuPrivate,
    "_datadump_original_op_names",
    "_datadump_original_op_types",
    "needCheckTf",
    "reference",
    "attr_name_engine_async_flag_",
    "async_flag",
    "INPUT_IS_VAR",
    "OUTPUT_IS_VAR",
    "_tbe_kernel_buffer",
    "ub_atomic_params",
    "_l2fusion_ToOpStruct",
    "_composite_engine_name",
    "_composite_engine_kernel_lib_name",
    "LxFusionOptimized",
    "is_first_node",
    "is_last_node",
    "_ffts_plus",
    "_op_slice_info",
    "_thread_scope_id",
    "_thread_mode",
    "_sgt_json_info",
    "cut_type"};

// json config:represent input/output data type are list
const std::string kInOutAttrDataTypeList = "list";

// function name need output index, default is 1
const int kDefaultOutputIndexRange = 1;

// error output index
const int kErrorOutputIndex = -1;

// attrs_map_desc config, represent shape type
const std::string kTfShapeType = "tf_shape";

const std::string kTfEnumType = "tf_enum";
} // namespace

namespace aicpu {
std::shared_ptr<Ir2tfBaseParser> Ir2tfBaseParser::Instance() {
  static std::shared_ptr<Ir2tfBaseParser> instance;
  static std::once_flag flag;
  std::call_once(flag, [&]() {
    instance.reset(new(std::nothrow) Ir2tfBaseParser);
  });
  return instance;
}

ge::Status Ir2tfBaseParser::ParseNodeDef(const ge::Node &node, domi::tensorflow::NodeDef *node_def) {
  AICPU_CHECK_NOTNULL_ERRCODE(node_def, ErrorCode::INPUT_PARAM_NULL)
  node_def->set_name(node.GetName());
  AICPU_CHECK_RES_WITH_LOG(ParseAttr(node, node_def),
      "Call Ir2tfBaseParser::ParseAttr function failed, op[%s], op type[%s].",
      node.GetName().c_str(), node.GetType().c_str())
  AICPU_CHECK_RES_WITH_LOG(ParseInput(node, node_def),
      "Call Ir2tfBaseParser::ParseInput function failed, op[%s], op type[%s].",
      node.GetName().c_str(), node.GetType().c_str())
  return ge::SUCCESS;
}

ge::Status Ir2tfBaseParser::LoadMappingConfig() {
  if (is_loaded_) {
    return ge::SUCCESS;
  }
  std::string real_config_file_path = GetConfigFilePath() + kIr2TfFileRelativePath;
  // get ir_mapping config from prototxt
  nlohmann::json json_read;
  aicpu::State state = ReadJsonFile(real_config_file_path, json_read);
  if (state.state != ge::SUCCESS) {
    map<std::string, std::string> err_map;
    err_map["filename"] = real_config_file_path;
    err_map["reason"] = state.msg;
    REPORT_PREDEFINED_ERR_MSG(GetViewErrorCodeStr(ViewErrorCode::LOAD_IR_CONFIG_ERR).c_str(),
                              std::vector<const char *>({"filename", "reason"}),
                              std::vector<const char *>({real_config_file_path.c_str(), state.msg.c_str()}));
    AICPUE_LOGE("Parse ir2tf config file[%s] failed, %s.",
        real_config_file_path.c_str(), state.msg.c_str());
    return ErrorCode::IR2TF_FILE_PARSE_FAILED;
  }

  AICPU_IF_BOOL_EXEC(json_read.find(kIrMappingConfigIr2Tf) == json_read.end(),
      AICPU_REPORT_INNER_ERR_MSG("json file[%s] does not have IR2TF.",
          real_config_file_path.c_str());
      return ErrorCode::IR2TF_CONFIG_INVALID)
  aicpu::IRFMKOpMapLib ir_mapping_lib = json_read;

  AICPU_CHECK_RES(ValidateOpMappingConfig(ir_mapping_lib.ir2tf))
  for (const auto &op_map_info : ir_mapping_lib.ir2tf) {
    ir2tf_map_.insert(map<std::string, OpMapInfo>::value_type(op_map_info.src_op_type, op_map_info));
  }

  // select op which input or output has ref data type
  for (const auto &item : ir2tf_map_) {
    const OpMapInfo &op_map_info = item.second;
    const std::vector<RefTransDesc> &input_ref_trans_desc = op_map_info.input_ref_map_desc;
    std::set<std::string> input_name;
    for (const auto &desc : input_ref_trans_desc) {
      if (desc.is_ref) {
        input_name.insert(desc.src_inout_name);
      }
    }
    if (!input_name.empty()) {
      ref_input_map_[item.second.src_op_type] = input_name;
    }

    const std::vector<RefTransDesc> &output_ref_trans_desc = op_map_info.output_ref_desc;
    std::set<std::string> output_name;
    for (const auto &desc : output_ref_trans_desc) {
      if (desc.is_ref) {
        output_name.insert(desc.src_inout_name);
      }
    }
    if (!output_name.empty()) {
      ref_output_map_[item.second.src_op_type] = output_name;
    }
  }
  is_loaded_ = true;
  return ge::SUCCESS;
}

ge::Status Ir2tfBaseParser::ParseAttr(const ge::Node &node, domi::tensorflow::NodeDef *node_def) {
  auto iter = ir2tf_map_.find(node.GetType());
  if (iter != ir2tf_map_.end()) {
    AICPU_CHECK_RES_WITH_LOG(ParseConfigAttr(node, node_def, &(iter->second)),
        "Call Ir2tfBaseParser::ParseConfigAttr function failed, op[%s], op type[%s].",
        node.GetName().c_str(), node.GetType().c_str())
    AICPU_CHECK_RES_WITH_LOG(ParseBaseAttr(node, node_def),
        "Call Ir2tfBaseParser::ParseBaseAttr function failed, op[%s], op type[%s].",
        node.GetName().c_str(), node.GetType().c_str())
    AICPU_CHECK_RES_WITH_LOG(ParseExtendAttr(node, node_def, &(iter->second)),
        "Call Ir2tfBaseParser::ParseExtendAttr function failed, op[%s], op type[%s].",
        node.GetName().c_str(), node.GetType().c_str())
    return ge::SUCCESS;
  }
  AICPUE_LOGW("Op[%s] is not exist in ir2tf json config.", node.GetType().c_str());
  node_def->set_op(node.GetType());
  AICPU_CHECK_RES_WITH_LOG(ParseBaseAttr(node, node_def),
      "Call Ir2tfBaseParser::ParseBaseAttr function failed, op[%s], op type[%s].",
      node.GetName().c_str(), node.GetType().c_str());
  return ge::SUCCESS;
}

ge::Status Ir2tfBaseParser::ParseBaseAttr(const ge::Node &node, domi::tensorflow::NodeDef *node_def) {
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL(op_desc_ptr)
  auto map = op_desc_ptr->GetAllAttrs();
  std::set<std::string> attr_blacklist;
  GetOpBlacklist(node.GetType(), attr_blacklist);
  // loop all attrs on NodePtr AND insert to attrs map on NodeDef
  for (auto iter = map.begin(); iter != map.end(); ++iter) {
    const std::string &ge_attr_name = iter->first;
    auto res_iter = kConstAttrBlacklist.find(ge_attr_name);
    if (res_iter != kConstAttrBlacklist.end()) {
      // if attr in white list , no need to transfer.
      continue;
    }

    res_iter = attr_blacklist.find(ge_attr_name);
    if (res_iter != attr_blacklist.end()) {
      continue;
    }

    domi::tensorflow::AttrValue attr_value;
    const ge::GeAttrValue::ValueType ge_value_type = (iter->second).GetValueType();
    AICPUE_LOGD("Get attr: [%s] value from op[%s], ge_value_type is [%d].",
                ge_attr_name.c_str(), node.GetType().c_str(), ge_value_type);

    // If get attr value failed, no need to insert the attr. Only print log
    if (GetAttrValueFromGe(node, ge_attr_name, ge_value_type, attr_value) != ge::SUCCESS) {
      AICPU_REPORT_INNER_ERR_MSG(
          "Call Ir2tfBaseParser::GetAttrValueFromGe failed, op[%s], op type[%s].",
          node.GetName().c_str(), node.GetType().c_str());
      continue;
    }

    AICPU_CHECK_NOTNULL(node_def->mutable_attr())
    auto pair = node_def->mutable_attr()->insert(AttrValueMap::value_type(ge_attr_name, attr_value));
    if (!pair.second) {
      AICPUE_LOGW("Op[%s] insert attr[%s] to node_def failed, op type[%s].",
                  node.GetName().c_str(), ge_attr_name.c_str(), node.GetType().c_str());
    }
  }
  return ge::SUCCESS;
}

ge::Status Ir2tfBaseParser::ParseConfigAttr(const ge::Node &node,
                                            domi::tensorflow::NodeDef *node_def,
                                            const OpMapInfo *ir2tf) {
  AICPU_CHECK_NOTNULL_ERRCODE(ir2tf, ErrorCode::INPUT_PARAM_NULL)
  AICPU_CHECK_NOTNULL_ERRCODE(node_def, ErrorCode::INPUT_PARAM_NULL)
  node_def->set_op(ir2tf->dst_op_type);

  for (const ParserExpDesc &attr_mapping : ir2tf->attrs_map_desc) {
    AICPU_CHECK_RES_WITH_LOG(HandleAttrMapping(node, node_def, attr_mapping),
        "Call Ir2tfBaseParser::HandleAttrMapping function failed, op[%s], op type[%s].",
        node.GetName().c_str(), node.GetType().c_str())
  }

  // parse input_attr_map_desc and output_attr_map_desc config
  AICPU_CHECK_RES_WITH_LOG(HandleInputAttrMapping(node, node_def, ir2tf),
      "Call Ir2tfBaseParser::HandleInputAttrMapping function failed, op[%s], op type[%s].",
      node.GetName().c_str(), node.GetType().c_str())

  AICPU_CHECK_RES_WITH_LOG(HandleOutputAttrMapping(node, node_def, ir2tf),
      "Call Ir2tfBaseParser::HandleOutputAttrMapping function failed, op[%s], op type[%s].",
      node.GetName().c_str(), node.GetType().c_str())
  return ge::SUCCESS;
}

ge::Status Ir2tfBaseParser::HandleAttrMapping(const ge::Node &node,
                                              domi::tensorflow::NodeDef *node_def,
                                              const ParserExpDesc &attr_mapping) {
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL(op_desc_ptr);
  domi::tensorflow::AttrValue attr_value;
  if ((!attr_mapping.src_field_name.empty()) && attr_mapping.parser_express.empty()) {
    // mapping attr ,value from srcAttr
    auto attr_map = op_desc_ptr->GetAllAttrs();
    // loop all attrs on Node AND insert to attrs map on NodeDef
    const auto iter = attr_map.find(attr_mapping.src_field_name);
    if (iter != attr_map.end()) {
      const ge::GeAttrValue::ValueType ge_value_type = (iter->second).GetValueType();
      AICPU_CHECK_RES_WITH_LOG(GetAttrValueFromGe(node, iter->first, ge_value_type, attr_value),
          "Call Ir2tfBaseParser::GetAttrValueFromGe function failed,"
          " attr[%s], op[%s], op type[%s]",
          (iter->first).c_str(), node.GetName().c_str(), node.GetType().c_str())
    } else {
      AICPU_REPORT_INNER_ERR_MSG("Can not find attr[%s], op[%s], op type[%s].",
          attr_mapping.src_field_name.c_str(), node.GetName().c_str(),
          node.GetType().c_str());
      return ErrorCode::IR2TF_SRC_ATTR_MISSING;
    }

    auto pair = node_def->mutable_attr()->insert(AttrValueMap::value_type(attr_mapping.dst_field_name, attr_value));
    if (!pair.second) {
      AICPUE_LOGW("Op[%s] insert attr[%s] failed, op type[%s].",
                  node.GetName().c_str(),
                  attr_mapping.dst_field_name.c_str(), node.GetType().c_str());
    }
    return ge::SUCCESS;
  }

  std::string expression = attr_mapping.parser_express;
  // mapping shape attr, value from ge attr
  if (expression == kTfShapeType) {
    return SetTfAttrShape(op_desc_ptr, node_def, attr_mapping);
  }

  if (expression == kTfEnumType) {
    return SetTfAttrEnum(op_desc_ptr, node_def, attr_mapping);
  }
  // mapping T attr, value from ge op desc
  return SetTfAttrType(op_desc_ptr, node_def, attr_mapping);
}

ge::Status Ir2tfBaseParser::SetTfAttrType(const ge::OpDescPtr &op_desc_ptr,
                                          domi::tensorflow::NodeDef *node_def,
                                          const ParserExpDesc &attr_mapping) {
  // The configuration is like input_xx, proctected by internal ,no need to validate
  std::string expression = attr_mapping.parser_express;
  size_t first_underline_pos = expression.find_first_of('_');
  CHECK_RES_BOOL(first_underline_pos != std::string::npos,
      ErrorCode::IR2TF_CONFIG_PARSE_FAILED,
      AICPU_REPORT_INNER_ERR_MSG(
          "Invalid parser express[%s] shoule be contain '-'. op[%s], op type[%s].",
          expression.c_str(), op_desc_ptr->GetName().c_str(),
          op_desc_ptr->GetType().c_str()))

  std::string src_anchor = expression.substr(0, first_underline_pos);
  std::string anchor_index = expression.substr(first_underline_pos + 1, expression.length());
  int index = 0;
  aicpu::State state = StringToNum(anchor_index, index);
  if (state.state != ge::SUCCESS) {
      AICPU_REPORT_INNER_ERR_MSG(
          "Convert %s to int failed, %s, when paser expression[%s], op[%s]",
          anchor_index.c_str(), state.msg.c_str(), expression.c_str(),
          op_desc_ptr->GetName().c_str());
      return ErrorCode::IR2TF_CONFIG_INVALID;
  }

  if (index < 0) {
    AICPU_REPORT_INNER_ERR_MSG("invalid index of expression[%s], op[%s], op type[%s].",
        expression.c_str(), op_desc_ptr->GetName().c_str(),
        op_desc_ptr->GetType().c_str());
    return ErrorCode::IR2TF_CONFIG_INVALID;
  }

  std::string op_type = op_desc_ptr->GetType();
  auto ir2tf = ir2tf_map_[op_type];
  auto dynamic_attrs = ir2tf.attrs_dynamic_desc;
  ge::DataType ge_data_type = ge::DT_UNDEFINED;
  int start_index = 0;
  for (auto dynamic_attr : dynamic_attrs) {
    int index_dynamic = 0;
    state = StringToNum(dynamic_attr.index, index_dynamic);
    if (state.state != ge::SUCCESS) {
        AICPU_REPORT_INNER_ERR_MSG(
            "Convert %s to int failed, %s, when paser[attrsDynamicDesc] index,"
            " op[%s].", dynamic_attr.index.c_str(), state.msg.c_str(),
            op_desc_ptr->GetName().c_str());
        return ErrorCode::IR2TF_CONFIG_INVALID;
    }

    if (index_dynamic < 0) {
      AICPU_REPORT_INNER_ERR_MSG("invalid dynamic index[%s], op[%s], op type[%s].",
          dynamic_attr.index.c_str(), op_desc_ptr->GetName().c_str(),
          op_type.c_str());
      return ErrorCode::IR2TF_CONFIG_INVALID;
    }
    if (dynamic_attr.type == src_anchor) {
      if (src_anchor == "output" && index_dynamic < index) {
        int temp = 0;
        ge::AttrUtils::GetInt(op_desc_ptr, DYNAMIC_OUTPUT_TD_NUM(dynamic_attr.name), temp);
        start_index += (temp - 1);
      }
      if (src_anchor == "input" && index_dynamic < index) {
        int temp = 0;
        ge::AttrUtils::GetInt(op_desc_ptr, DYNAMIC_INPUT_TD_NUM(dynamic_attr.name), temp);
        start_index += (temp - 1);
      }
    }
  }
  start_index += index;
  if (src_anchor == "output") {
    ge_data_type = op_desc_ptr->GetOutputDesc(start_index).GetDataType();
  } else {
    ge_data_type = op_desc_ptr->GetInputDesc(start_index).GetDataType();
  }
  domi::tensorflow::DataType data_type = ConvertGeDataType2TfDataType(ge_data_type);

  domi::tensorflow::AttrValue attr_value;
  attr_value.set_type(data_type);
  auto pair = node_def->mutable_attr()->insert(AttrValueMap::value_type(attr_mapping.dst_field_name, attr_value));
  if (!pair.second) {
    AICPUE_LOGW("Op[%s] insert attr[%s] failed, op type[%s].",
                op_desc_ptr->GetName().c_str(), attr_mapping.dst_field_name.c_str(), op_type.c_str());
  }
  return ge::SUCCESS;
}

ge::Status Ir2tfBaseParser::SetTfAttrShape(const ge::OpDescPtr &op_desc_ptr,
                                           domi::tensorflow::NodeDef *node_def,
                                           const ParserExpDesc &attr_mapping) const {
  std::vector<int32_t> shape_value;
  CHECK_RES_BOOL(ge::AttrUtils::GetListInt(op_desc_ptr, attr_mapping.src_field_name, shape_value),
      ErrorCode::DATA_TYPE_UNDEFILED,
      AICPU_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::GetListInt failed to get attr[%s], op[%s]",
          attr_mapping.src_field_name.c_str(), op_desc_ptr->GetName().c_str()))

  domi::tensorflow::AttrValue attr_value;
  domi::tensorflow::TensorShapeProto *shape_tensor = attr_value.mutable_shape();
  for (const int32_t &dim : shape_value) {
    if (dim <= 0) {
      shape_tensor->set_unknown_rank(true);
      // If unknown_rank true, dim.size() must be 0.
      shape_tensor->clear_dim();
      break;
    }
    shape_tensor->add_dim()->set_size(dim);
  }

  auto pair = node_def->mutable_attr()->insert(AttrValueMap::value_type(attr_mapping.dst_field_name, attr_value));
  if (!pair.second) {
    AICPUE_LOGW("Op[%s] insert attr[%s] failed, op type[%s].",
                op_desc_ptr->GetName().c_str(), attr_mapping.dst_field_name.c_str(), op_desc_ptr->GetType().c_str());
  }
  return ge::SUCCESS;
}
ge::Status Ir2tfBaseParser::SetTfAttrEnum(const ge::OpDescPtr &op_desc_ptr,
                                          domi::tensorflow::NodeDef *node_def,
                                          const ParserExpDesc &attr_mapping) const {
  ge::DataType ge_data_type = ge::DT_UNDEFINED;
  CHECK_RES_BOOL(ge::AttrUtils::GetDataType(op_desc_ptr, attr_mapping.src_field_name, ge_data_type),
      ErrorCode::DATA_TYPE_UNDEFILED,
      AICPU_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::GetDataType failed to get attr[%s], op[%s]",
          attr_mapping.src_field_name.c_str(), op_desc_ptr->GetName().c_str()))
  AICPUE_LOGD("SetTfAttrEnum ge_data_type = %ld.", static_cast<int64_t>(ge_data_type));
  domi::tensorflow::DataType data_type = ConvertGeDataType2TfDataType(ge_data_type);
  int64_t type_enmu = (ge_data_type == ge::DT_UINT1) ?
                      (static_cast<int64_t>(ge::DT_UINT1)) : (static_cast<int64_t>(data_type));
  AICPUE_LOGD("SetTfAttrEnum type_enmu = %ld.", type_enmu);
  domi::tensorflow::AttrValue attr_value;
  attr_value.set_i(type_enmu);

  auto pair = node_def->mutable_attr()->insert(AttrValueMap::value_type(attr_mapping.dst_field_name, attr_value));
  if (!pair.second) {
    AICPUE_LOGW("Op[%s] insert attr[%s] failed, op type[%s].",
                op_desc_ptr->GetName().c_str(), attr_mapping.dst_field_name.c_str(), op_desc_ptr->GetType().c_str());
  }
  return ge::SUCCESS;
}

ge::Status Ir2tfBaseParser::ValidateOpMappingConfig(const std::vector<OpMapInfo> &op_map_info_list) const {
  if (op_map_info_list.empty()) {
      return ge::SUCCESS;
  }

  for (const OpMapInfo &op_map_info : op_map_info_list) {
    AICPU_IF_BOOL_EXEC(op_map_info.src_op_type.empty(),
        AICPU_REPORT_INNER_ERR_MSG("IR2TF config srcOpType should not be empty.");
        return ErrorCode::IR2TF_CONFIG_INVALID)

    AICPU_IF_BOOL_EXEC(op_map_info.dst_op_type.empty(),
        AICPU_REPORT_INNER_ERR_MSG("IR2TF config dstOpType should not be empty.");
        return ErrorCode::IR2TF_CONFIG_INVALID)
    // validate attr mapping
    AICPU_CHECK_RES(ValidateAttrMapConfig(op_map_info))

    // validate input attr mapping
    AICPU_CHECK_RES(ValidateInputAttrMapConfig(op_map_info))

    // validate output attr mapping
    AICPU_CHECK_RES(ValidateOutputAttrMapConfig(op_map_info))

    // validate output_ref_desc attr mapping
    AICPU_CHECK_RES(ValidateRefTransDesc(op_map_info))
  }
  return ge::SUCCESS;
}

ge::Status Ir2tfBaseParser::ValidateAttrMapConfig(const OpMapInfo &op_map_info) const {
  const std::vector<ParserExpDesc> &attr_mapping_list = op_map_info.attrs_map_desc;
  AICPU_IF_BOOL_EXEC(attr_mapping_list.empty(), return ge::SUCCESS)

  for (const ParserExpDesc &exp_desc : attr_mapping_list) {
    AICPU_IF_BOOL_EXEC(exp_desc.dst_field_name.empty(),
        AICPU_REPORT_INNER_ERR_MSG("dstFieldName of attrsMapDesc in IR config "
            "can not be empty, op type[%s]", op_map_info.src_op_type.c_str());
        return ErrorCode::IR2TF_CONFIG_INVALID)

    AICPU_IF_BOOL_EXEC(exp_desc.parser_express.empty() && exp_desc.src_field_name.empty(),
        AICPU_REPORT_INNER_ERR_MSG("parserExpress and srcFieldName of attrsMapDesc"
            " in IR config can not be empty at the same time, op type[%s].",
            op_map_info.src_op_type.c_str());
        return ErrorCode::IR2TF_CONFIG_INVALID)
  }
  return ge::SUCCESS;
}

ge::Status Ir2tfBaseParser::ValidateOutputAttrMapConfig(const OpMapInfo &op_map_info) const {
  const std::vector<ParserExpDesc> &output_attr_map_desc_list = op_map_info.output_attr_map_desc;
  AICPU_IF_BOOL_EXEC(output_attr_map_desc_list.empty(), return ge::SUCCESS)

  for (const ParserExpDesc &exp_desc : output_attr_map_desc_list) {
    AICPU_IF_BOOL_EXEC(exp_desc.src_field_name.empty(),
        AICPU_REPORT_INNER_ERR_MSG("srcFieldName of outputAttrMapDesc in IR config"
            " can not be empty, op type[%s].", op_map_info.src_op_type.c_str());
        return ErrorCode::IR2TF_CONFIG_INVALID)

    AICPU_IF_BOOL_EXEC(exp_desc.parser_express.empty(),
        AICPU_REPORT_INNER_ERR_MSG("parserExpress of outputAttrMapDesc in IR config"
            " can not be empty, op type[%s].", op_map_info.src_op_type.c_str());
        return ErrorCode::IR2TF_CONFIG_INVALID)
  }
  return ge::SUCCESS;
}

ge::Status Ir2tfBaseParser::ValidateInputAttrMapConfig(const OpMapInfo &op_map_info) const {
  const std::vector<ParserExpDesc> &input_attr_map_desc_list = op_map_info.input_attr_map_desc;
  AICPU_IF_BOOL_EXEC(input_attr_map_desc_list.empty(), return ge::SUCCESS)

  for (const ParserExpDesc &exp_desc : input_attr_map_desc_list) {
    AICPU_IF_BOOL_EXEC(exp_desc.src_field_name.empty(),
        AICPU_REPORT_INNER_ERR_MSG("srcFieldName of inputAttrMapDesc in IR config"
            " is empty, op type[%s].", op_map_info.src_op_type.c_str());
        return ErrorCode::IR2TF_CONFIG_INVALID)

    AICPU_IF_BOOL_EXEC(exp_desc.parser_express.empty(),
        AICPU_REPORT_INNER_ERR_MSG(" parserExpress of inputAttrMapDesc in IR "
            "config is empty, op type[%s].", op_map_info.src_op_type.c_str());
        return ErrorCode::IR2TF_CONFIG_INVALID)

    AICPU_IF_BOOL_EXEC(exp_desc.dst_field_name.empty(),
        AICPU_REPORT_INNER_ERR_MSG("dstFieldName of inputAttrMapDesc in IR config"
            " is empty, op type[%s].", op_map_info.src_op_type.c_str());
        return ErrorCode::IR2TF_CONFIG_INVALID)
  }
  return ge::SUCCESS;
}

std::string Ir2tfBaseParser::GetConfigFilePath() const {
  std::shared_ptr<Ir2tfBaseParser> (*instance_ptr)() = &Ir2tfBaseParser::Instance;
  return GetSoPath(reinterpret_cast<void *>(instance_ptr));
}

ge::Status Ir2tfBaseParser::ParseInput(const ge::Node &node, domi::tensorflow::NodeDef *node_def) const {
  for (const auto &in_anchor : node.GetAllInDataAnchorsPtr()) {
    AICPU_CHECK_NOTNULL(in_anchor->GetOwnerNode())
    std::string name = in_anchor->GetOwnerNode()->GetName();
    std::string src_output = Stringcat(in_anchor->GetIdx());
    node_def->add_input(name + ":" + src_output);
  }
  return ge::SUCCESS;
}

ge::Status Ir2tfBaseParser::GetAttrValueFromGe(const ge::Node &node,
                                               const std::string &attr_name,
                                               const ge::GeAttrValue::ValueType ge_value_type,
                                               domi::tensorflow::AttrValue &attr_value) {
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL(op_desc_ptr);
  switch (ge_value_type) {
    case ge::GeAttrValue::ValueType::VT_STRING: {
      const std::string *string_value = ge::AttrUtils::GetStr(op_desc_ptr, attr_name);
      CHECK_RES_BOOL((string_value != nullptr),
          ErrorCode::DATA_TYPE_UNDEFILED,
          AICPU_REPORT_INNER_ERR_MSG(
              "Call ge::AttrUtils::GetStr failed to get attr[%s], op[%s].",
              attr_name.c_str(), node.GetName().c_str()))
      attr_value.set_s(*string_value);
      break;
    }
    case ge::GeAttrValue::ValueType::VT_FLOAT: {
      float float_value = 0.0;
      CHECK_RES_BOOL(ge::AttrUtils::GetFloat(op_desc_ptr, attr_name, float_value),
          ErrorCode::DATA_TYPE_UNDEFILED,
          AICPU_REPORT_INNER_ERR_MSG(
              "Call ge::AttrUtils::GetFloat failed to get attr[%s], op[%s].",
              attr_name.c_str(), node.GetName().c_str()))
      attr_value.set_f(float_value);
      break;
    }
    case ge::GeAttrValue::ValueType::VT_BOOL: {
      bool bool_value = false;
      CHECK_RES_BOOL(ge::AttrUtils::GetBool(op_desc_ptr, attr_name, bool_value),
          ErrorCode::DATA_TYPE_UNDEFILED,
          AICPU_REPORT_INNER_ERR_MSG(
              "Call ge::AttrUtils::GetBool failed to get attr[%s], op[%s].",
              attr_name.c_str(), node.GetName().c_str()))
      attr_value.set_b(bool_value);
      break;
    }
    case ge::GeAttrValue::ValueType::VT_INT: {
      int64_t int_value = 0;
      CHECK_RES_BOOL(ge::AttrUtils::GetInt(op_desc_ptr, attr_name, int_value),
          ErrorCode::DATA_TYPE_UNDEFILED,
          AICPU_REPORT_INNER_ERR_MSG(
              "Call ge::AttrUtils::GetInt failed to get attr[%s], op[%s].",
              attr_name.c_str(), node.GetName().c_str()))
      attr_value.set_i(int_value);
      break;
    }
    case ge::GeAttrValue::ValueType::VT_TENSOR: {
      AICPU_CHECK_RES(SetTfTensorValue(op_desc_ptr, attr_name, attr_value))
      break;
    }
    case ge::GeAttrValue::ValueType::VT_LIST_FLOAT: {
      std::vector<float> float_list;
      CHECK_RES_BOOL(ge::AttrUtils::GetListFloat(op_desc_ptr, attr_name, float_list),
          ErrorCode::DATA_TYPE_UNDEFILED,
          AICPU_REPORT_INNER_ERR_MSG(
              "Call ge::AttrUtils::GetListFloat failed to get attr[%s], op[%s].",
              attr_name.c_str(), node.GetName().c_str()))

      for (const float &float_element : float_list) {
        attr_value.mutable_list()->add_f(float_element);
      }
      break;
    }
    case ge::GeAttrValue::ValueType::VT_LIST_LIST_INT: {
      AICPU_CHECK_RES(SetTfListShape(op_desc_ptr, attr_name, attr_value))
      break;
    }
    case ge::GeAttrValue::ValueType::VT_LIST_DATA_TYPE: {
      AICPU_CHECK_RES(SetTfListType(op_desc_ptr, attr_name, attr_value))
      break;
    }
    case ge::GeAttrValue::ValueType::VT_LIST_INT: {
      AICPU_CHECK_RES(SetTfListInt(node, attr_name, attr_value))
      break;
    }
    case ge::GeAttrValue::ValueType::VT_DATA_TYPE: {
      AICPU_CHECK_RES(SetTfType(op_desc_ptr, attr_name, attr_value))
      break;
    }
    case ge::GeAttrValue::ValueType::VT_LIST_STRING: {
      AICPU_CHECK_RES(SetTfListString(op_desc_ptr, attr_name, attr_value))
      break;
    }
    default: {
      AICPU_REPORT_INNER_ERR_MSG("Currently ir2tf can not support attr type[%d].",
          ge_value_type);
      return ErrorCode::UNKNOWN_ATTR_TYPE;
    }
  }
  return ge::SUCCESS;
}

ge::Status Ir2tfBaseParser::SetTfTensorValue(const ge::OpDescPtr &op_desc_ptr,
                                             const std::string &attr_name,
                                             domi::tensorflow::AttrValue &attr_value) const {
  ge::ConstGeTensorPtr ge_tensor;
  CHECK_RES_BOOL(ge::AttrUtils::GetTensor(op_desc_ptr, attr_name, ge_tensor),
      ErrorCode::DATA_TYPE_UNDEFILED,
      AICPU_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::GetTensor failed to get attr[%s], op[%s].",
          attr_name.c_str(), op_desc_ptr->GetName().c_str()))

  // 1.Get tensor of attr_value
  ge::DataType ge_data_type = ge_tensor->GetTensorDesc().GetDataType();
  domi::tensorflow::TensorProto *tf_tensor = attr_value.mutable_tensor();
  tf_tensor->set_dtype(ConvertGeDataType2TfDataType(ge_data_type));

  domi::tensorflow::TensorShapeProto *tf_shape = tf_tensor->mutable_tensor_shape();
  for (const auto &dim : ge_tensor->GetTensorDesc().GetShape().GetDims()) {
    domi::tensorflow::TensorShapeProto_Dim *tf_dims = tf_shape->add_dim();
    tf_dims->set_size(dim);
  }

  int32_t data_size = GetDataTypeSize(ge_data_type);
  AICPU_CHECK_GREATER_THAN_ZERO(data_size, ErrorCode::DATA_TYPE_UNDEFILED,
      "Invalid data type[%s], op[%s].",
      ge::TypeUtils::DataTypeToSerialString(ge_data_type).c_str(),
      op_desc_ptr->GetName().c_str())
  int32_t data_count = ge_tensor->GetData().size() / data_size;
  return SetTfTensorValue(ge_tensor, tf_tensor, data_count);
}

ge::Status Ir2tfBaseParser::SetTfListShape(const ge::OpDescPtr &op_desc_ptr,
                                           const std::string &attr_name,
                                           domi::tensorflow::AttrValue &attr_value) const {
  std::vector<std::vector<int64_t>> shape_value;
  CHECK_RES_BOOL(ge::AttrUtils::GetListListInt(op_desc_ptr, attr_name, shape_value),
      ErrorCode::DATA_TYPE_UNDEFILED,
      AICPU_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::GetListListInt failed to get attr[%s], op[%s].",
          attr_name.c_str(), op_desc_ptr->GetName().c_str()))

  auto attr_list = attr_value.mutable_list();
  for (const std::vector<int64_t> &shape : shape_value) {
    domi::tensorflow::TensorShapeProto *shape_tensor = attr_list->add_shape();
    for (const int64_t &dim : shape) {
      if (dim <= 0) {
        shape_tensor->set_unknown_rank(true);
        // If unknown_rank true, dim.size() must be 0.
        shape_tensor->clear_dim();
        break;
      }
      shape_tensor->add_dim()->set_size(dim);
    }
  }
  return ge::SUCCESS;
}

ge::Status Ir2tfBaseParser::SetTfListType(const ge::OpDescPtr &op_desc_ptr,
                                          const std::string &attr_name,
                                          domi::tensorflow::AttrValue &attr_value) const {
  AICPUE_LOGD("Get op type[%s] attr name[%s] data type list.",
              op_desc_ptr->GetName().c_str(), attr_name.c_str());
  std::vector<ge::DataType> ge_type_list;
  CHECK_RES_BOOL(ge::AttrUtils::GetListDataType(op_desc_ptr, attr_name, ge_type_list),
      ErrorCode::DATA_TYPE_UNDEFILED,
      AICPU_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::GetListDataType failed to get attr[%s], op[%s].",
          attr_name.c_str(), op_desc_ptr->GetName().c_str()))

  AICPU_CHECK_NOTNULL(attr_value.mutable_list());
  for (ge::DataType &ge_type : ge_type_list) {
    AICPUE_LOGD("Attr name[%s] ge type[%d].", attr_name.c_str(), ge_type);
    domi::tensorflow::DataType data_type = ConvertGeDataType2TfDataType(ge_type);
    attr_value.mutable_list()->add_type(data_type);
  }
  return ge::SUCCESS;
}

ge::Status Ir2tfBaseParser::SetTfType(const ge::OpDescPtr &op_desc_ptr,
                                      const std::string &attr_name,
                                      domi::tensorflow::AttrValue &attr_value) const {
  AICPUE_LOGD("Attr name[%s] ge op[%s].", attr_name.c_str(), op_desc_ptr->GetName().c_str());
  auto iter = ir2tf_map_.find(op_desc_ptr->GetType());
  if (iter != ir2tf_map_.end()) {
    // foreach attrs_map_desc, find attr_name from src_field_name
    // if find, parser_express is tf_enum, the attr is dtype type
    for (const ParserExpDesc &attr_mapping : iter->second.attrs_map_desc) {
      if ((attr_mapping.src_field_name == attr_name) &&
          (attr_mapping.parser_express == kTfEnumType)) {
        AICPUE_LOGD("Op type[%s] attr name[%s] is in the attrs_map_desc config, represent is enum type.",
                    op_desc_ptr->GetType().c_str(), attr_name.c_str());
        return ge::SUCCESS;
      }
    }
  }

  ge::DataType ge_type;
  CHECK_RES_BOOL(ge::AttrUtils::GetDataType(op_desc_ptr, attr_name, ge_type),
      ErrorCode::DATA_TYPE_UNDEFILED,
      AICPU_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::GetDataType failed to get attr[%s], op[%s].",
          attr_name.c_str(), op_desc_ptr->GetName().c_str()))
  domi::tensorflow::DataType data_type = ConvertGeDataType2TfDataType(ge_type);
  AICPUE_LOGD("ge::AttrUtils:: attr[%s] op[%s] value[%ld]", attr_name.c_str(), op_desc_ptr->GetName().c_str(),
              static_cast<int64_t>(ge_type));
  attr_value.set_type(data_type);
  return ge::SUCCESS;
}

ge::Status Ir2tfBaseParser::SetTfListInt(const ge::Node &node,
                                         const std::string &attr_name,
                                         domi::tensorflow::AttrValue &attr_value) {
  auto iter = ir2tf_map_.find(node.GetType());
  if (iter != ir2tf_map_.end()) {
    // foreach attrs_map_desc, find attr_name from src_field_name
    // if find, parser_express is tf_shape, the attr is shape type
    for (const ParserExpDesc &attr_mapping : iter->second.attrs_map_desc) {
      if ((attr_mapping.src_field_name == attr_name) &&
          (attr_mapping.parser_express == kTfShapeType)) {
        AICPUE_LOGI("Op type[%s] attr name[%s] is in the attrs_map_desc config, represent is shape type.",
                    node.GetType().c_str(), attr_name.c_str());
        return ge::SUCCESS;
      }
    }
  }

  // if not find in json file, the attr is truely list_int
  std::vector<int32_t> list_int_value;
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  CHECK_RES_BOOL(ge::AttrUtils::GetListInt(op_desc_ptr, attr_name, list_int_value),
      ErrorCode::DATA_TYPE_UNDEFILED,
      AICPU_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::GetListInt failed to get attr[%s], op[%s].",
          attr_name.c_str(), op_desc_ptr->GetName().c_str()))

  for (int32_t value : list_int_value) {
    attr_value.mutable_list()->add_i(value);
  }
  return ge::SUCCESS;
}

ge::Status Ir2tfBaseParser::SetTfListString(const ge::OpDescPtr &op_desc_ptr,
                                            const std::string &attr_name,
                                            domi::tensorflow::AttrValue &attr_value) const {
  std::vector<std::string> list_string_value;
  CHECK_RES_BOOL(ge::AttrUtils::GetListStr(op_desc_ptr, attr_name, list_string_value),
      ErrorCode::DATA_TYPE_UNDEFILED,
      AICPU_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::GetListStr failed to get attr[%s], op[%s].",
          attr_name.c_str(), op_desc_ptr->GetName().c_str()))

  for (std::string value : list_string_value) {
    attr_value.mutable_list()->add_s(value);
  }
  return ge::SUCCESS;
}

ge::Status Ir2tfBaseParser::SetTfTensorValue(const ge::ConstGeTensorPtr &ge_tensor,
                                             domi::tensorflow::TensorProto *tf_tensor,
                                             int32_t data_count) const {
  CHECK_RES_BOOL((ge_tensor->GetData().data() != nullptr),
      ErrorCode::INPUT_PARAM_NULL,
      AICPU_REPORT_INNER_ERR_MSG("Tensor data is null."));

  if (data_count > 1) {
    tf_tensor->set_tensor_content(reinterpret_cast<const void *>(ge_tensor->GetData().data()),
                                  ge_tensor->GetData().size());
    return ge::SUCCESS;
  }
  auto ge_dtype = ge_tensor->GetTensorDesc().GetDataType();
  switch (ge_dtype) {
    case ge::DT_FLOAT: {
      const float float_value = *(reinterpret_cast<const float *>(ge_tensor->GetData().data()));
      tf_tensor->add_float_val(float_value);
      break;
    }
    case ge::DT_INT32: {
      const int32_t int_value = *(reinterpret_cast<const int32_t *>(ge_tensor->GetData().data()));
      tf_tensor->add_int_val(int_value);
      break;
    }
    case ge::DT_BOOL: {
      const bool bool_value = *(reinterpret_cast<const bool *>(ge_tensor->GetData().data()));
      tf_tensor->add_bool_val(bool_value);
      break;
    }
    case ge::DT_INT64: {
      const int64_t int_value = *(reinterpret_cast<const int64_t *>(ge_tensor->GetData().data()));
      tf_tensor->add_int64_val(int_value);
      break;
    }
    case ge::DT_FLOAT16: {
      const int32_t float_value = *(reinterpret_cast<const int32_t *>(ge_tensor->GetData().data()));
      tf_tensor->add_half_val(float_value);
      break;
    }
    default: {
      AICPU_REPORT_INNER_ERR_MSG("data type[%s] not supported.",
          ge::TypeUtils::DataTypeToSerialString(ge_dtype).c_str());
    }
  }
  return ge::SUCCESS;
}

ge::Status Ir2tfBaseParser::ParseOutputArgs(const ge::NodePtr &node, const std::string &op_type,
                                            NameRangeMap &outputs) {
  AICPU_CHECK_NOTNULL(node);
  AICPUE_LOGI("Get op[%s] op type[%s] index range from ir2tf config file.",
              op_type.c_str(), node->GetType().c_str());

  auto iter = ir2tf_map_.find(op_type);
  if (iter == ir2tf_map_.end()) {
    AICPU_REPORT_INNER_ERR_MSG("Op type[%s] is not exist in ir2tf config file. op[%s].",
        op_type.c_str(), node->GetName().c_str());
    return ErrorCode::OP_NOT_EXIST_IN_IR2TF_CONFIG_FILE;
  }

  const OpMapInfo &op_map_info = iter->second;
  const std::vector<ParserExpDesc> &output_attr_list = op_map_info.output_attr_map_desc;
  const std::vector<RefTransDesc> &ref_trans_desc = op_map_info.output_ref_desc;

  if (output_attr_list.empty()) {
    return GetSingleOutputNameRange(ref_trans_desc, outputs, op_map_info, node);
  }

  return GetListOutputNameRange(output_attr_list, ref_trans_desc, node, outputs);
}

ge::Status Ir2tfBaseParser::GetSingleOutputNameRange(const std::vector<RefTransDesc> &ref_trans_list,
                                                     NameRangeMap &outputs,
                                                     const OpMapInfo &op_map_info,
                                                     const ge::NodePtr &node) const {
  int start = 0;
  const std::vector<DynamicExpDesc> &dynamic_desc_list = op_map_info.attrs_dynamic_desc;
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  AICPU_CHECK_NOTNULL(op_desc_ptr)
  for (const RefTransDesc &output_arg : ref_trans_list) {
    int set_num = kDefaultOutputIndexRange;
    for (const DynamicExpDesc &dynamic_desc : dynamic_desc_list) {
      if (dynamic_desc.type == "output" && dynamic_desc.name == output_arg.src_inout_name) {
        ge::AttrUtils::GetInt(op_desc_ptr, DYNAMIC_OUTPUT_TD_NUM(dynamic_desc.name), set_num);
      }
    }
    outputs[output_arg.dst_inout_name] = std::make_pair(start, start + set_num);
    start += set_num;
  }
  return ge::SUCCESS;
}

ge::Status Ir2tfBaseParser::ValidateRefTransDesc(const OpMapInfo &op_map_info) const {
  const std::vector<RefTransDesc> &ref_trans_list = op_map_info.output_ref_desc;
  AICPU_IF_BOOL_EXEC(ref_trans_list.empty(), return ge::SUCCESS)

  for (const RefTransDesc &ref_trans_desc : ref_trans_list) {
    AICPU_IF_BOOL_EXEC(ref_trans_desc.src_inout_name.empty(),
        AICPU_REPORT_INNER_ERR_MSG(
            "srcInOutputName of outputRefDesc in IR config can not be empty.");
        return ErrorCode::IR2TF_CONFIG_INVALID)

    AICPU_IF_BOOL_EXEC(ref_trans_desc.dst_inout_name.empty(),
        AICPU_REPORT_INNER_ERR_MSG(
            "dstInOutputName of outputRefDesc in IR config can not be empty.");
        return ErrorCode::IR2TF_CONFIG_INVALID)
  }
  return ge::SUCCESS;
}

ge::Status Ir2tfBaseParser::GetListOutputNameRange(const std::vector<ParserExpDesc> &output_attr_list,
                                                   const std::vector<RefTransDesc> &ref_trans_list,
                                                   const ge::NodePtr &node,
                                                   NameRangeMap &outputs) const {
  std::unordered_map<std::string, ParserExpDesc> output_attr_map;
  for (const ParserExpDesc &attr_mapping : output_attr_list) {
    output_attr_map[attr_mapping.src_field_name] = attr_mapping;
  }

  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  AICPU_CHECK_NOTNULL(op_desc_ptr)

  int start = 0;
  std::unordered_map<std::string, int> output_list_num_map;
  for (const RefTransDesc &output_arg : ref_trans_list) {
    std::string ge_output_name = output_arg.src_inout_name;
    std::string tf_output_name = output_arg.dst_inout_name;

    int index = kDefaultOutputIndexRange;
    auto iter = output_attr_map.find(ge_output_name);
    if (iter != output_attr_map.end()) {
      std::string expression = iter->second.parser_express;

      if (expression == kInOutAttrDataTypeList) {
        index = GetOutputListSize(node, ge_output_name);
      } else {
        AICPUE_LOGI("Get op[%s] output[%s] index range from attr[%s].",
                    node->GetType().c_str(), ge_output_name.c_str(), expression.c_str());
        CHECK_RES_BOOL(ge::AttrUtils::GetInt(op_desc_ptr, expression, index),
            ErrorCode::DATA_TYPE_UNDEFILED,
            AICPU_REPORT_INNER_ERR_MSG(
                "Call ge::AttrUtils::GetInt failed to get attr[%s], op[%s].",
                expression.c_str(), node->GetName().c_str()))
      }

      AICPU_IF_BOOL_EXEC(index < 0,
          AICPU_REPORT_INNER_ERR_MSG(
              "Get list output index range failed, op[%s], op type[%s].",
              node->GetName().c_str(), node->GetType().c_str());
          return IR2TF_CONFIG_PARSE_FAILED)
    }

    int end = start + index;
    outputs[tf_output_name] = std::make_pair(start, end);
    AICPUE_LOGI("Output[%s] is list type, index range is [%d - %d].", ge_output_name.c_str(), start, end);
    start = end;
  }
  return ge::SUCCESS;
}

int Ir2tfBaseParser::GetOutputListSize(const ge::NodePtr &node, const std::string &dest_name) const {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  AICPU_CHECK_NOTNULL(op_desc_ptr)
  AICPUE_LOGI("Get op[%s] output[%s] index range from output name.",
              node->GetType().c_str(), dest_name.c_str());

  int max_index = -1;
  auto anchor_vistor = node->GetAllOutDataAnchors();
  for (const auto &out_anchor : anchor_vistor) {
    std::string name_idx = op_desc_ptr->GetOutputNameByIndex(out_anchor->GetIdx());
    if (!IsListType(name_idx, dest_name)) {
      continue;
    }

    int output_index = GetListIndex(name_idx, dest_name.length(), name_idx.length());
    if (output_index == kErrorOutputIndex) {
        return output_index;
    }
    max_index = max_index < output_index ? output_index : max_index;
  }

  // dynamic input and dynamic output index is begin from 0, input/output number need add 1
  max_index += 1;
  return max_index;
}

ge::Status Ir2tfBaseParser::HandleInputAttrMapping(const ge::Node &node,
                                                   domi::tensorflow::NodeDef *node_def,
                                                   const OpMapInfo *ir2tf) const {
  if (ir2tf->input_attr_map_desc.empty()) {
    return ge::SUCCESS;
  }

  // get input_attr_map_desc src_field_name and dst_field_name
  std::unordered_map<std::string, std::string> input_src_dst_name_map;
  GetSrcAndDstFieldName(ir2tf->input_attr_map_desc, input_src_dst_name_map);
  if (input_src_dst_name_map.empty()) {
    return ge::SUCCESS;
  }

  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL(op_desc_ptr)
  std::unordered_map<std::string, std::vector<domi::tensorflow::DataType>> input_data_type_map;

  // if input is list, input name is name+index, index starts at 1
  auto anchor_vistor = node.GetAllInDataAnchorsPtr();
  for (const auto &in_anchor : anchor_vistor) {
    int index = in_anchor->GetIdx();
    std::string name_idx = op_desc_ptr->GetInputNameByIndex(index);
    std::string dst_field_name;
    if (!IsListType(input_src_dst_name_map, name_idx, dst_field_name)) {
      continue;
    }

    ge::DataType ge_data_type = op_desc_ptr->GetInputDesc(index).GetDataType();
    AICPU_CHECK_RES(GetDataTypeList(ge_data_type, dst_field_name, input_data_type_map))
  }

  AICPU_CHECK_RES(SetAttrTypeList(node_def, input_data_type_map))
  return ge::SUCCESS;
}

ge::Status Ir2tfBaseParser::HandleOutputAttrMapping(const ge::Node &node,
                                                    domi::tensorflow::NodeDef *node_def,
                                                    const OpMapInfo *ir2tf) const {
  if (ir2tf->output_attr_map_desc.empty()) {
      return ge::SUCCESS;
  }

  // get output_attr_map_desc src_field_name and dst_field_name
  std::unordered_map<std::string, std::string> output_src_dst_name_map;
  GetSrcAndDstFieldName(ir2tf->output_attr_map_desc, output_src_dst_name_map);
  if (output_src_dst_name_map.empty()) {
    return ge::SUCCESS;
  }

  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL(op_desc_ptr)
  std::unordered_map<std::string, std::vector<domi::tensorflow::DataType>> output_data_type_map;

  // if output is list, output name is name+index, index starts at 1
  auto anchor_vistor = node.GetAllOutDataAnchors();
  for (const auto &out_anchor : anchor_vistor) {
    int index = out_anchor->GetIdx();
    std::string name_idx = op_desc_ptr->GetOutputNameByIndex(index);
    std::string dst_field_name;
    if (!IsListType(output_src_dst_name_map, name_idx, dst_field_name)) {
      continue;
    }

    // dst_field_name is empty, do not need set tensorflow attr
    if (dst_field_name.empty()) {
      continue;
    }

    ge::DataType ge_data_type = op_desc_ptr->GetOutputDesc(index).GetDataType();
    AICPU_CHECK_RES_WITH_LOG(GetDataTypeList(ge_data_type, dst_field_name, output_data_type_map),
        "Call Ir2tfBaseParser::GetDataTypeList function failed, op[%s], op type[%s]",
        node.GetName().c_str(), node.GetType().c_str())
  }

  AICPU_CHECK_RES_WITH_LOG(SetAttrTypeList(node_def, output_data_type_map),
      "Call Ir2tfBaseParser::SetAttrTypeList function failed, op[%s], op type[%s]",
      node.GetName().c_str(), node.GetType().c_str())
  return ge::SUCCESS;
}

void Ir2tfBaseParser::GetSrcAndDstFieldName(const std::vector<ParserExpDesc> &exp_desc_list,
                                            std::unordered_map<std::string, std::string> &src_dst_name_map) const {
  for (const ParserExpDesc &attr_mapping : exp_desc_list) {
    std::string expression = attr_mapping.parser_express;
    if (expression != kInOutAttrDataTypeList) {
      continue;
    }

    if (attr_mapping.dst_field_name.empty()) {
      continue;
    }
    src_dst_name_map[attr_mapping.src_field_name] = attr_mapping.dst_field_name;
  }
}

ge::Status Ir2tfBaseParser::GetDataTypeList(const ge::DataType ge_data_type,
                                            const std::string &dst_attr_name,
                                            std::unordered_map<std::string,
                                            std::vector<domi::tensorflow::DataType>> &data_type_map) const {
  domi::tensorflow::DataType data_type = ConvertGeDataType2TfDataType(ge_data_type);
  AICPUE_LOGI("Input/Output attr[%s], datatype [%d].", dst_attr_name.c_str(), data_type);

  auto type_iter = data_type_map.find(dst_attr_name);
  if (type_iter == data_type_map.end()) {
    std::vector<domi::tensorflow::DataType> type_list;
    type_list.push_back(data_type);
    data_type_map[dst_attr_name] = type_list;
  } else {
    type_iter->second.push_back(data_type);
  }
  return ge::SUCCESS;
}

bool Ir2tfBaseParser::IsListType(const std::unordered_map<std::string, std::string> &src_dst_field_name_map,
                                 const std::string &name_idx, std::string &dst_field_name) const {
  for (auto iter = src_dst_field_name_map.begin(); iter != src_dst_field_name_map.end(); ++iter) {
    std::string src_field_name = iter->first;
    if (!IsListType(name_idx, src_field_name)) {
      continue;
    }

    dst_field_name = iter->second;

    AICPUE_LOGI("Input/Output[%s] is list type.", name_idx.c_str());
    return true;
  }
  return false;
}

bool Ir2tfBaseParser::IsListType(const std::string &name_idx, const std::string &config_name) const {
  AICPUE_LOGD("Check Input/Output is list or not, name_idx: [%s], config name: [%s].",
              name_idx.c_str(), config_name.c_str());

  int config_len = config_name.length();
  int name_len = name_idx.length();
  if (name_len <= config_len) {
    return false;
  }

  std::string sub_name = name_idx.substr(0, config_len);
  if (config_name != sub_name) {
    return false;
  }

  for (int i = config_len; i < name_len; i++) {
    if (!isdigit(name_idx[i])) {
      return false;
    }
  }
  return true;
}

int Ir2tfBaseParser::GetListIndex(const std::string &name_idx, int start, int end) const {
  std::string anchor_index = name_idx.substr(start, end);
  int index = -1;
  (void) StringToNum(anchor_index, index);
  if (index < 0) {
    AICPU_REPORT_INNER_ERR_MSG("invalid output name[%s].", name_idx.c_str());
    return kErrorOutputIndex;
  }

  AICPUE_LOGI("Output[%s] is list, index is [%d].", name_idx.c_str(), index);
  return index;
}

ge::Status Ir2tfBaseParser::SetAttrTypeList(domi::tensorflow::NodeDef *node_def,
                                            std::unordered_map<std::string,
                                            std::vector<domi::tensorflow::DataType>> &data_type_map) const {
  if (data_type_map.empty()) {
    return ge::SUCCESS;
  }

  for (auto iter = data_type_map.begin(); iter != data_type_map.end(); ++iter) {
    domi::tensorflow::AttrValue attr_value;
    for (domi::tensorflow::DataType type : iter->second) {
      attr_value.mutable_list()->add_type(type);
    }
    auto pair = node_def->mutable_attr()->insert(AttrValueMap::value_type(iter->first, attr_value));
    if (!pair.second) {
      AICPUE_LOGW("NodeDef [%s] insert attr[%s] failed.", node_def->name().c_str(), iter->first.c_str());
    }
  }
  return ge::SUCCESS;
}

void Ir2tfBaseParser::GetOpBlacklist(std::string op_type, std::set<std::string> &attr_blacklist) {
  auto iter = ir2tf_map_.find(op_type);
  if (iter != ir2tf_map_.end()) {
    const OpMapInfo &op_map_info = iter->second;
    const std::string &backlist = op_map_info.attrs_blacklist;
    if (backlist.empty()) {
      return ;
    }

    SplitSequence(backlist, kConfigItemSeparator, attr_blacklist);
  }
}

void Ir2tfBaseParser::GetRefInputSet(const std::string &op_type, std::set<std::string> &ref_set) {
  auto iter = ref_input_map_.find(op_type);
  if (iter != ref_input_map_.end()) {
    ref_set = iter->second;
  }
}

void Ir2tfBaseParser::GetRefOutputSet(const std::string &op_type, std::set<std::string> &ref_set) {
  auto iter = ref_output_map_.find(op_type);
  if (iter != ref_output_map_.end()) {
    ref_set = iter->second;
  }
}

bool Ir2tfBaseParser::InputHaveRef(const std::string &op_type) {
  auto iter = ref_input_map_.find(op_type);
  return (iter != ref_input_map_.end());
}

ge::Status Ir2tfBaseParser::ParseExtendAttr(const ge::Node &node,
                                            domi::tensorflow::NodeDef *node_def,
                                            const OpMapInfo *ir2tf) const {
  AICPU_CHECK_NOTNULL_ERRCODE(ir2tf, ErrorCode::INPUT_PARAM_NULL)
  AICPU_CHECK_NOTNULL_ERRCODE(node_def, ErrorCode::INPUT_PARAM_NULL)

  // mapping default attr=>attr
  for (const ExtendFieldDesc &attr_default_value : ir2tf->attr_ext_desc) {
    AttrValueMap *attr_map = node_def->mutable_attr();
    const std::string &field_name = attr_default_value.field_name;
    if (attr_map->find(field_name) != attr_map->end()) {
      domi::tensorflow::AttrValue src_value = (*attr_map)[field_name];
      if ((src_value.value_case() != domi::tensorflow::AttrValue::kS) || (src_value.s() != "")) {
        return ge::SUCCESS;
      } else {
        attr_map->erase(field_name);
      }
    }

    AICPU_CHECK_RES_WITH_LOG(
        HandleDefaultAttrType(node_def, attr_default_value, node.GetName()),
        "Call Ir2tfBaseParser::HandleDefaultAttrType function failed, op[%s].",
        node.GetName().c_str())
  }
  return ge::SUCCESS;
}
void Ir2tfBaseParser::SetDefaultString(domi::tensorflow::AttrValue &attr_value,
                                       const std::string &default_value_str,
                                       const std::string &node_name) const {
  AICPUE_LOGI("default_value is [%s].", default_value_str.c_str());
  if (default_value_str == "resource_name") {
    attr_value.set_s(node_name);
  } else {
    attr_value.set_s(default_value_str);
  }
}

ge::Status Ir2tfBaseParser::HandleDefaultAttrType(domi::tensorflow::NodeDef *node_def,
                                                  const ExtendFieldDesc &attr_default_value,
                                                  const std::string &node_name) const {
  domi::tensorflow::AttrValue attr_value;
  const std::string default_value_str = attr_default_value.default_value;
  const std::string field_name = attr_default_value.field_name;

  DefaultType index = DefaultType::DEFAULT_UNDEFIN;
  static const map<std::string, DefaultType> DataTypeMap = {{"string", DefaultType::DEFAULT_STRING},
                                                              {"int", DefaultType::DEFAULT_INT},
                                                              {"float", DefaultType::DEFAULT_FLOAT},
                                                              {"bool", DefaultType::DEFAULT_BOOL},
                                                              {"type", DefaultType::DEFAULT_TYPE},
                                                              {"list", DefaultType::DEFAULT_LIST}};
  auto default_data_type = DataTypeMap.find(attr_default_value.data_type);
  if (default_data_type != DataTypeMap.end()) {
    index = default_data_type->second;
  }

  switch (index) {
    case DefaultType::DEFAULT_STRING: {
      SetDefaultString(attr_value, default_value_str, node_name);
      break;
    }
    case DefaultType::DEFAULT_INT: {
      int64_t int_value = 0;
      if (!default_value_str.empty()) {
        aicpu::State ret = StringToNum(default_value_str, int_value);
        AICPU_IF_BOOL_EXEC(ret.state != ge::SUCCESS,
            AICPU_REPORT_INNER_ERR_MSG("invalid default value[%s] of [%s], %s.",
                default_value_str.c_str(), field_name.c_str(), ret.msg.c_str());
            return ErrorCode::IR2TF_CONFIG_INVALID)
      }
      AICPUE_LOGI("default_value is [%ld].", int_value);
      attr_value.set_i(int_value);
      break;
    }
    case DefaultType::DEFAULT_FLOAT: {
      float float_value = 0;
      if (!default_value_str.empty()) {
        aicpu::State ret = StringToNum(default_value_str, float_value);
        AICPU_IF_BOOL_EXEC(ret.state != ge::SUCCESS,
            AICPU_REPORT_INNER_ERR_MSG("invalid default value[%s] of [%s], %s.",
                default_value_str.c_str(), field_name.c_str(), ret.msg.c_str());
            return ErrorCode::IR2TF_CONFIG_INVALID)
      }
      AICPUE_LOGI("default_value is [%f].", float_value);
      attr_value.set_f(float_value);
      break;
    }
    case DefaultType::DEFAULT_BOOL: {
      bool bool_value = false;
      if (!default_value_str.empty()) {
        aicpu::State ret = StringToBool(default_value_str, bool_value);
        AICPU_IF_BOOL_EXEC(ret.state != ge::SUCCESS,
            AICPU_REPORT_INNER_ERR_MSG("invalid default value[%s] of [%s], %s.",
                default_value_str.c_str(), field_name.c_str(), ret.msg.c_str());
            return ErrorCode::IR2TF_CONFIG_INVALID)
      }
      AICPUE_LOGI("default_value is [%d].", bool_value);
      attr_value.set_b(bool_value);
      break;
    }
    case DefaultType::DEFAULT_TYPE: {
      domi::tensorflow::DataType data_type = TFDataType::DT_FLOAT;
      if (!default_value_str.empty()) {
        bool ret = ConvertString2TfDataType(default_value_str, data_type);
        AICPU_IF_BOOL_EXEC(!(ret),
            AICPU_REPORT_INNER_ERR_MSG("invalid default value[%s] of [%s].",
                default_value_str.c_str(), field_name.c_str());
            return ErrorCode::IR2TF_CONFIG_INVALID)
      }
      AICPUE_LOGI("default_value is [%d].", data_type);
      attr_value.set_type(data_type);
      break;
    }
    case DefaultType::DEFAULT_LIST: {
      attr_value.mutable_list()->Clear();
      break;
    }
    default: {
      AICPU_REPORT_INNER_ERR_MSG(
          "Invalid attr type[%s] in IR config, field name[%s].",
          attr_default_value.data_type.c_str(), field_name.c_str());
      return ErrorCode::IR2TF_CONFIG_INVALID;
    }
  }

  auto pair = node_def->mutable_attr()->insert(AttrValueMap::value_type(attr_default_value.field_name, attr_value));
  if (!pair.second) {
    AICPUE_LOGW("Node insert attr[%s] failed.", attr_default_value.field_name.c_str());
  }
  else {
    AICPUE_LOGI("Node insert attr[%s] success.", attr_default_value.field_name.c_str());
  }
  return ge::SUCCESS;
}
}  // namespace aicpu
