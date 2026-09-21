/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tf_util.h"
#include "util/constant.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "tf_kernel_info/tf_kernel_info.h"
#include "ir2tf/ir2tf_parser_factory.h"
#include "graph/utils/attr_utils.h"

using domi::tensorflow::NodeDef;

namespace aicpu {
static uint64_t g_kernel_id = 0;
const std::string kPlaceholderOpType = "PlaceHolder";
const std::string kEndOpType = "End";
const std::string kConstOpType = "Const";

uint64_t GenerateUniqueId() {
  if (g_kernel_id == ULLONG_MAX) {
    g_kernel_id = 0;
  }
  return g_kernel_id++;
}

TFDataType ConvertGeDataType2TfDataType(ge::DataType data_type, bool is_ref) {
  static const std::map<ge::DataType, TFDataType> DataTypeMap = {
      {ge::DataType::DT_FLOAT16, TFDataType::DT_HALF},
      {ge::DataType::DT_FLOAT, TFDataType::DT_FLOAT},
      {ge::DataType::DT_DOUBLE, TFDataType::DT_DOUBLE},
      {ge::DataType::DT_INT8, TFDataType::DT_INT8},
      {ge::DataType::DT_UINT8, TFDataType::DT_UINT8},
      {ge::DataType::DT_INT16, TFDataType::DT_INT16},
      {ge::DataType::DT_UINT16, TFDataType::DT_UINT16},
      {ge::DataType::DT_INT32, TFDataType::DT_INT32},
      {ge::DataType::DT_UINT32, TFDataType::DT_UINT32},
      {ge::DataType::DT_INT64, TFDataType::DT_INT64},
      {ge::DataType::DT_UINT64, TFDataType::DT_UINT64},
      {ge::DataType::DT_BOOL, TFDataType::DT_BOOL},
      {ge::DataType::DT_RESOURCE, TFDataType::DT_RESOURCE},
      {ge::DataType::DT_STRING, TFDataType::DT_STRING},
      {ge::DataType::DT_STRING_REF, TFDataType::DT_STRING_REF},
      {ge::DataType::DT_COMPLEX64, TFDataType::DT_COMPLEX64},
      {ge::DataType::DT_COMPLEX128, TFDataType::DT_COMPLEX128},
      {ge::DataType::DT_QINT8, TFDataType::DT_QINT8},
      {ge::DataType::DT_QUINT8, TFDataType::DT_QUINT8},
      {ge::DataType::DT_QINT16, TFDataType::DT_QINT16},
      {ge::DataType::DT_QUINT16, TFDataType::DT_QUINT16},
      {ge::DataType::DT_QINT32, TFDataType::DT_QINT32},
      {ge::DataType::DT_VARIANT, TFDataType::DT_VARIANT},
      {ge::DataType::DT_BF16, TFDataType::DT_BFLOAT16},
      {ge::DataType::DT_UINT1, TFDataType::DT_BOOL}};

  const auto diff = TFDataType::DT_FLOAT_REF - TFDataType::DT_FLOAT;
  std::map<ge::DataType, TFDataType>::const_iterator iter = DataTypeMap.find(data_type);
  if (iter != DataTypeMap.end()) {
    if (is_ref) {
      TFDataType tf_data_type = iter->second;
      // Avoid mishandling DT_STRING_REF
      if (tf_data_type < diff) {
        tf_data_type = static_cast<TFDataType>(tf_data_type + diff);
      }
      return tf_data_type;
    } else {
      return iter->second;
    }
  } else {
    return TFDataType::DT_INVALID;
  }
}

bool ConvertString2TfDataType(const string &elem, TFDataType &data_type) {
  static const map<string, TFDataType> DataTypeMap = {
      {"DT_INT8", TFDataType::DT_INT8},
      {"DT_UINT8", TFDataType::DT_UINT8},
      {"DT_INT16", TFDataType::DT_INT16},
      {"DT_UINT16", TFDataType::DT_UINT16},
      {"DT_INT32", TFDataType::DT_INT32},
      {"DT_INT64", TFDataType::DT_INT64},
      {"DT_UINT32", TFDataType::DT_UINT32},
      {"DT_UINT64", TFDataType::DT_UINT64},
      {"DT_FLOAT16", TFDataType::DT_HALF},
      {"DT_FLOAT", TFDataType::DT_FLOAT},
      {"DT_DOUBLE", TFDataType::DT_DOUBLE},
      {"DT_BOOL", TFDataType::DT_BOOL},
      {"DT_COMPLEX64", TFDataType::DT_COMPLEX64},
      {"DT_COMPLEX128", TFDataType::DT_COMPLEX128},
      {"DT_STRING", TFDataType::DT_STRING},
      {"DT_STRING_REF", TFDataType::DT_STRING_REF},
      {"DT_RESOURCE", TFDataType::DT_RESOURCE},
      {"DT_QINT8", TFDataType::DT_QINT8},
      {"DT_QUINT8", TFDataType::DT_QUINT8},
      {"DT_QINT16", TFDataType::DT_QINT16},
      {"DT_QUINT16", TFDataType::DT_QUINT16},
      {"DT_QINT32", TFDataType::DT_QINT32},
      {"DT_VARIANT", TFDataType::DT_VARIANT},
      {"DT_BF16", TFDataType::DT_BFLOAT16},
      {"DT_UINT1", TFDataType::DT_BOOL}};

  auto it = DataTypeMap.find(elem);
  // only data_type in configure file exists in DataTypeMap, save it and return
  if (it != DataTypeMap.end()) {
      data_type = it->second;
      return true;
  }
  return false;
}

bool IsSpecialDtString(const ge::OpDescPtr &op_desc_ptr) {
  if (ge::AttrUtils::HasAttr(op_desc_ptr, kAttrNameInputOutputDtString)) {
    return true;
  }
  return false;
}

/**
 * Identify and set ShapeType attr for ge node
 * @param node ge node
 * @return status whether this operation success
 */
ge::Status CheckAndSetUnknowType(ge::NodePtr &node) {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  AICPU_CHECK_NOTNULL(op_desc_ptr);
  std::string op_type = op_desc_ptr->GetType();
  std::string node_name = node->GetName();
  bool shape_is_static = false;
  if (IsUnknowShape(op_desc_ptr) || IsSpecialDtString(op_desc_ptr)) {
    AICPU_CHECK_FALSE_EXEC(
        ge::AttrUtils::SetBool(op_desc_ptr, kAttrNameUnknownShape, true),
        AICPU_REPORT_INNER_ERR_MSG("Call ge::AttrUtils::SetBool to set attr[%s] failed, op[%s], op type[%s].",
                                kAttrNameUnknownShape.c_str(), node_name.c_str(), op_type.c_str());
        return ErrorCode::ADD_ATTR_FAILED)
  } else {
    shape_is_static = true;
  }

  int32_t shape_type = 0;
  std::string max_shape_str;
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, ErrorCode::INPUT_PARAM_NULL);
  if (op_type == kFunctionOp) {
    // get unknow shape type from output desc
    ge::Status status = GetUnKnowTypeByOutDesc(node, shape_type);
    AICPU_CHECK_FALSE_EXEC(status == ge::GRAPH_SUCCESS,
                           AICPU_REPORT_INNER_ERR_MSG("Get shape type with output shape failed, op[%s], op type[%s].",
                                                    node_name.c_str(), op_type.c_str());
                           return status;)
    if (IsSpecialDtString(op_desc_ptr)) {
      shape_type = ge::DEPEND_COMPUTE;
    }
    // set unknow shape type
    AICPU_CHECK_FALSE_EXEC(
        ge::AttrUtils::SetInt(op_desc_ptr, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, shape_type),
        AICPU_REPORT_INNER_ERR_MSG("Call ge::AttrUtils::SetInt failed to set attr[%s], op[%s], op type[%s].",
                                ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE.c_str(), node_name.c_str(), op_type.c_str());
        return ErrorCode::ADD_ATTR_FAILED)
    AICPUE_LOGI("Set unknow shape type[%d] for op[%s], op type[%s].", shape_type, node_name.c_str(), op_type.c_str());
    return ge::SUCCESS;
  }
  if (op_type == kFrameworkOp) {
    AICPUE_LOGW("Invalid op type, op[%s] is FrameworkOp when set shape type, op type[%s].",
                node_name.c_str(), op_type.c_str());
    return ErrorCode::PARAM_INVALID;
  }
  //  set unknow shape type with kernel info
  KernelInfoPtr kernel_info_ptr = TfKernelInfo::Instance();
  AICPU_CHECK_NOTNULL(kernel_info_ptr);
  OpFullInfo op_full_info;
  ge::Status status = kernel_info_ptr->GetOpInfo(op_type, op_full_info);
  AICPU_CHECK_FALSE_EXEC(status == ge::SUCCESS,
      AICPU_REPORT_INNER_ERR_MSG("op type[%s] not support, op[%s].",
          op_type.c_str(), node_name.c_str());
      return status;)
  if (shape_is_static && (op_full_info.shapeType == 4)) {
    shape_type = 1;
  } else if (ge::AttrUtils::GetStr(op_desc_ptr, ge::ATTR_NAME_OP_MAX_SHAPE, max_shape_str)) {
    shape_type = ge::DEPEND_SHAPE_RANGE;
  }else {
    shape_type = op_full_info.shapeType;
  }
  if (IsSpecialDtString(op_desc_ptr)) {
    shape_type = ge::DEPEND_COMPUTE;
  }
  AICPU_CHECK_FALSE_EXEC(ge::AttrUtils::SetInt(op_desc_ptr, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, shape_type),
      AICPU_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::SetInt failed to set attr[%s], op[%s], op type[%s].",
          ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE.c_str(), node_name.c_str(),
          op_type.c_str());
      return ErrorCode::ADD_ATTR_FAILED)
  AICPUE_LOGI("Set unknown shape type[%d] for op[%s], op type[%s].",
              shape_type, node_name.c_str(), op_type.c_str());
  // set topic type
  FWKAdapter::FWKExtTopicType topic_type = op_full_info.topicType;
  AICPU_CHECK_FALSE_EXEC(ge::AttrUtils::SetInt(op_desc_ptr, kTopicType, topic_type),
    AICPU_REPORT_INNER_ERR_MSG(
       "Call ge::AttrUtils::SetInt failed to set attr[%s], op[%d], op type[%s].",
       kTopicType.c_str(), topic_type, op_type.c_str());
  return ErrorCode::ADD_ATTR_FAILED)
  AICPUE_LOGI("Set topic type[%d] for op[%s], op type[%s].",
              topic_type, node_name.c_str(), op_type.c_str());

  return ge::SUCCESS;
}

// get unknown type with outputs shape desc
ge::Status GetUnKnowTypeByOutDesc(const ge::NodePtr &node, int32_t &shape_type) {
  shape_type = 0;
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, ErrorCode::INPUT_PARAM_NULL);
  for (const auto &desc : op_desc_ptr->GetAllOutputsDescPtr()) {
    auto ge_shape = desc->GetShape();
    for (const auto &dim : ge_shape.GetDims()) {
      if (dim == ge::UNKNOWN_DIM || dim == ge::UNKNOWN_DIM_NUM) {
        shape_type = ge::DEPEND_COMPUTE;
        return ge::SUCCESS;
      }
    }
    std::vector<std::pair<int64_t, int64_t>> un_use;
    if (desc->GetShapeRange(un_use) == ge::GRAPH_SUCCESS) {
      // output shape is range
      shape_type = ge::DEPEND_SHAPE_RANGE;
    }
  }

  if ((shape_type != ge::DEPEND_COMPUTE) && (shape_type != ge::DEPEND_SHAPE_RANGE)) {
    shape_type = ge::DEPEND_IN_SHAPE;
    AICPUE_LOGI("Function_Op has known shape type.");
    return ge::SUCCESS;
  } else {
    return ge::SUCCESS;
  }
}

ge::Status UpdataOpInfo(const ge::Node &node, const map<string, OpFullInfo> &all_op_info)
{
  ge::OpDescPtr op_desc = node.GetOpDesc();
  AICPU_CHECK_NOTNULL(op_desc);
  std::string op_type = op_desc->GetType();
  if ((op_type == kPlaceholderOpType) || (op_type == kEndOpType) ||
      (op_type == kConstOpType)) {
    return ge::SUCCESS;
  }

  std::string kernel_lib_name = GetKernelLibNameByOpType(op_type, all_op_info);
  auto iter = all_op_info.find(op_type);
  if (iter == all_op_info.end()) {
    AICPU_REPORT_INNER_ERR_MSG("Can't find op type[%s] in KernelInfo, op[%s].",
                             op_type.c_str(), node.GetName().c_str());
    return ErrorCode::CREATE_NODEDEF_FAILED;
  }

  OpFullInfo op_full_info = iter->second;
  std::string resource = op_full_info.resource;
  AICPUE_LOGI("UpdataOpInfo:: resource[%s]", resource.c_str());
  (void)ge::AttrUtils::SetStr(op_desc, kResource, resource);
  return ge::SUCCESS;
}

/**
 * Create node def for ge node
 * @param  node Ge node
 * @return status whether operation successful
 */
ge::Status CreateNodeDef(const ge::Node &node) {
  std::string node_name = node.GetName();
  ge::OpDescPtr op_desc = node.GetOpDesc();
  AICPU_CHECK_NOTNULL(op_desc)
  std::string op_type = op_desc->GetType();
  // check function op
  if (op_type == kFunctionOp) {
    std::string err_msg = Stringcat("Can not create node def for function op[",
        node_name, "] in graph compile phase, op type[", op_type, "].");
    AICPU_REPORT_INNER_ERR_MSG("%s.", err_msg.c_str());
    return ErrorCode::NODE_DEF_NOT_EXIST;
  }
  if (op_type == kFrameworkOp) {
    const std::string *original_type = ge::AttrUtils::GetStr(op_desc, kOriginalType);
    CHECK_RES_BOOL((original_type != nullptr),
        ErrorCode::GET_ORIGINAL_TYPE_FAILED,
        AICPU_REPORT_INNER_ERR_MSG(
            "Call ge::AttrUtils::GetStr failed to get attr[%s], op[%s].",
            kOriginalType.c_str(), node_name.c_str()))
    ge::OpDescUtilsEx::SetType(op_desc, *original_type);
    op_type = *original_type;
  }

  // IR -> tf
  NodeDef node_def;
  std::shared_ptr<Ir2tfBaseParser> parser = Ir2tfParserFactory::Instance().CreateIRParser(op_type);
  if (parser == nullptr) {
    AICPU_REPORT_INNER_ERR_MSG("Create ir parser failed, op[%s], op type[%s].",
        node_name.c_str(), op_type.c_str());
    return ErrorCode::GET_IR2TF_PARSER_FAILED;
  }
  AICPU_CHECK_RES_WITH_LOG(parser->ParseNodeDef(node, &node_def),
      "Call ParseNodeDef function failed, op[%s], op type[%s].",
      node_name.c_str(), op_type.c_str())
  AICPUE_LOGI("Create node_def for ge op[%s] success, op type[%s].",
              node_name.c_str(), op_type.c_str());

  // set tf node_def for ge node
  auto state = InsertTfNodeDefToOp(op_desc, node_def, op_type, kTfNodeDef);
  if (state != ge::SUCCESS) {
    return state;
  }
  return ge::SUCCESS;
}

ge::Status InsertTfNodeDefToOp(const ge::OpDescPtr &op_desc_ptr, NodeDef &node_def, const std::string &op_type,
                               const std::string &attr_name) {
  ge::Buffer buffer(node_def.ByteSizeLong());
  google::protobuf::io::ArrayOutputStream array_stream(buffer.GetData(), static_cast<int32_t>(buffer.GetSize()));
  google::protobuf::io::CodedOutputStream output_stream(&array_stream);
  output_stream.SetSerializationDeterministic(true);
  if (!(node_def.SerializeToCodedStream(&output_stream))) {
    AICPU_REPORT_INNER_ERR_MSG("The serialization from nodedef probuf to str failed, op[%s], op type[%s].",
                             op_desc_ptr->GetName().c_str(), op_type.c_str());
    return ErrorCode::CREATE_NODEDEF_FAILED;
  }

  if (!(ge::AttrUtils::SetZeroCopyBytes(op_desc_ptr, attr_name, std::move(buffer)))) {
    AICPU_REPORT_INNER_ERR_MSG("Failed to call SetZeroCopyBytes to set node def, op[%s], op type[%s]..",
                            op_desc_ptr->GetName().c_str(), op_type.c_str());
    return ErrorCode::CREATE_NODEDEF_FAILED;
  }
  return ge::SUCCESS;
}

ge::Status SerializeKernelRunParamToBuffer(const FWKAdapter::KernelRunParam &kernel_run_param, const string &op_name,
                                           ge::Buffer &buffer) {
  google::protobuf::io::ArrayOutputStream array_stream(buffer.GetData(), static_cast<int32_t>(buffer.GetSize()));
  google::protobuf::io::CodedOutputStream output_stream(&array_stream);
  output_stream.SetSerializationDeterministic(true);
  if (!kernel_run_param.SerializeToCodedStream(&output_stream)) {
    AICPU_REPORT_INNER_ERR_MSG("Serialize kernel run param to string failed, op[%s].", op_name.c_str());
    return ErrorCode::CREATE_NODEDEF_FAILED;
  }
  return ge::SUCCESS;
}

/**
 * Calculation workspace size for node
 * @param  node Ge node
 * @param  workspace_size workspace_size
 * @return status whether operation successful
 */
ge::Status CalcWorkspaceSize(const ge::Node &node, int64_t &workspace_size) {
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL(op_desc_ptr)
  // Step1 : Calc the InputOuputBuf's size
  FWKAdapter::KernelRunParam kernel_run_param;
  AICPU_CHECK_RES_WITH_LOG(BuildKernelRunParam(*op_desc_ptr, kernel_run_param),
      "Call TfKernelBuilder::BuildKernelRunParam function failed, op[%s].",
      node.GetName().c_str())

  int64_t kernel_run_param_size = static_cast<int64_t>(kernel_run_param.ByteSizeLong());
  AICPUE_LOGI("The kernel_run_param_size size of op type[%s] is [%ld]",
              node.GetType().c_str(), kernel_run_param_size);
  workspace_size = kernel_run_param_size;

  // Step2 : Get the tf's node_def and func_def's definition and size
  ge::Buffer node_def_bytes;
  ge::Buffer func_def_lib_bytes;
  int64_t node_def_size = 0;
  int64_t func_def_lib_size = 0;
  AICPU_CHECK_RES_WITH_LOG(
      ParseNodeDefAndFuncDef(node, node_def_bytes, func_def_lib_bytes,
                             node_def_size, func_def_lib_size),
      "Call TfKernelBuilder::ParseNodeDefAndFuncDef function failed, op[%s].",
      node.GetName().c_str())
  // check overflow
  CHECK_INT64_ADD_OVERFLOW(node_def_size, func_def_lib_size,
      ErrorCode::DATA_OVERFLOW,
      "Overflow when calculate total bytes of node def[%ld] and function def"
      " lib[%ld]. op[%s]", node_def_size, func_def_lib_size,
      node.GetName().c_str())
  int64_t node_func_def_size = node_def_size + func_def_lib_size;
  AICPUE_LOGI("The nodeDef and funcDef size is [%ld], op type[%s]",
              node_func_def_size, node.GetType().c_str());

  CHECK_INT64_ADD_OVERFLOW(workspace_size, node_func_def_size,
      ErrorCode::DATA_OVERFLOW,
      "Workspace overflow when add total bytes of kernel ran param[%ld], "
      "node def[%ld] and function def lib[%ld]. op[%s]",
      workspace_size, node_def_size, func_def_lib_size, node.GetName().c_str())
  workspace_size += node_func_def_size;
  return ge::SUCCESS;
}

/**
 * Calc the size of tf's node_def, firstly transform the
 *  Node to tf's node_def,then get the node_def's size,
 *  if the operator has the func_def, then calc the
 *  func_def's size together.
 * @param node original GE node info
 * @param node_def_bytes tf node def
 * @param func_def_lib_bytes tf function def library
 * @param node_def_size the size of node def
 * @param func_def_lib_size the size of function def library
 * @return status whether operation successful
 */
ge::Status ParseNodeDefAndFuncDef(const ge::Node &node, ge::Buffer &node_def_bytes,
                                  ge::Buffer &func_def_lib_bytes, int64_t &node_def_size,
                                  int64_t &func_def_lib_size) {
  std::string node_name = node.GetName();
  ge::OpDescPtr op_desc = node.GetOpDesc();
  AICPU_CHECK_NOTNULL(op_desc)
  // calculate node def size
  if (!ge::AttrUtils::GetBytes(op_desc, kTfNodeDef, node_def_bytes)) {
    AICPUE_LOGI("Node def attr not exist in ge op[%s], op type[%s].",
                node_name.c_str(), node.GetType().c_str());
    AICPU_CHECK_RES_WITH_LOG(CreateNodeDef(node),
        "Call TfKernelBuilder::CreateNodeDef function failed, op[%s].",
        node.GetName().c_str())
    CHECK_RES_BOOL(ge::AttrUtils::GetBytes(op_desc, kTfNodeDef, node_def_bytes),
        ErrorCode::NODE_DEF_NOT_EXIST,
        AICPU_REPORT_INNER_ERR_MSG(
            "Call ge::AttrUtils::GetBytes failed to get attr[%s], op[%s].",
            kTfNodeDef.c_str(), node_name.c_str()))
  }
  node_def_size = node_def_bytes.GetSize();
  // calculate function def size
  AICPU_CHECK_FALSE_EXEC(ge::AttrUtils::GetBytes(op_desc, kTfFuncDef, func_def_lib_bytes),
                         AICPUE_LOGD("Function def attr is not exist in ge op[%s], op type[%s].",
                         node_name.c_str(), node.GetType().c_str());
                         func_def_lib_size = 0;
                         return ge::SUCCESS)
  func_def_lib_size = func_def_lib_bytes.GetSize();
  return ge::SUCCESS;
}

/**
 * Build the kernelRunParam
 * @param opDesc Op description
 * @param kernel_run_param fake kernel_run_param just the input and
 *  output data_addr is not real)
 * @param skip_dim_check
 * @return status whether operation successful
 */
ge::Status BuildKernelRunParam(const ge::OpDesc &op_desc, FWKAdapter::KernelRunParam &kernel_run_param,
                               bool skip_dim_check) {
  // Construct input's content
  std::set<std::string> ref_input_set;
  std::string op_type = op_desc.GetType();
  std::shared_ptr<Ir2tfBaseParser> parser = Ir2tfParserFactory::Instance().CreateIRParser(op_type);
  AICPU_CHECK_NOTNULL(parser);
  parser->GetRefInputSet(op_type, ref_input_set);

  size_t input_size = op_desc.GetInputsSize();
  aicpu::State state;
  for (size_t i = 0; i < input_size; i++) {
    FWKAdapter::TensorDataInfo *input_tensor = kernel_run_param.add_input();
    ge::GeTensorDesc ge_tensor_desc = op_desc.GetInputDesc(i);
    std::string input_name = op_desc.GetInputNameByIndex(i);
    bool is_ref = false;
    std::set<std::string>::const_iterator iter = ref_input_set.find(input_name);
    if (iter != ref_input_set.cend()) {
      is_ref = true;
    }
    AICPUE_LOGD("Op type[%s], input name[%s], is ref[%d]",
                op_type.c_str(), input_name.c_str(), is_ref);
    state = SetTensorDataInfo(ge_tensor_desc, *input_tensor, is_ref);
    if (state.state != ge::SUCCESS) {
      state.msg = Stringcat(i, "th input's ", state.msg, ", op[", op_desc.GetName(), "].");
      AICPU_REPORT_INNER_ERR_MSG("%s", state.msg.c_str());
      return state.state;
    }
  }

  // Construct output's content
  size_t output_size = op_desc.GetOutputsSize();
  for (size_t i = 0; i < output_size; i++) {
    FWKAdapter::TensorDataInfo *output_tensor = kernel_run_param.add_output();
    ge::GeTensorDesc ge_tensor_desc = op_desc.GetOutputDesc(i);
    state = SetTensorDataInfo(ge_tensor_desc, *output_tensor, false, skip_dim_check, true);
    if (state.state != ge::SUCCESS) {
      state.msg = Stringcat(i, "th output's ", state.msg, ", op[", op_desc.GetName(), "].");
      AICPU_REPORT_INNER_ERR_MSG("%s", state.msg.c_str());
      return state.state;
    }
  }
  return ge::SUCCESS;
}

/**
 * Set the aicpu::FWKAdapter::TensorDataInfo, the data is from Ge Tensor
 * @param ge_tensor_desc Original Ge Tensor
 * @param tensor_data_info The input or output data, defined by protobuf
 * @param is_ref if output is ref
 * @param skip_dim_check if skip dim
 * @param is_output if is output
 * @return status whether operation successful
 */
aicpu::State SetTensorDataInfo(const ge::GeTensorDesc &ge_tensor_desc, FWKAdapter::TensorDataInfo &tensor_data_info,
                               bool is_ref, bool skip_dim_check, bool is_output) {
  ge::DataType data_type = ge_tensor_desc.GetDataType();
  uint32_t tf_data_type = static_cast<uint32_t>(ConvertGeDataType2TfDataType(data_type, is_ref));
  tensor_data_info.set_dtype(tf_data_type);

  // Just used to calc the length, so put a fake value, different
  // number will have the different addr, so put the max uint64 value
  tensor_data_info.set_data_addr(ULLONG_MAX);
  std::vector<int64_t> dims;
  if (is_output) {
    std::vector<std::pair<int64_t, int64_t>> shape_range;
    // try to get dims from shape range
    AICPU_CHECK_RES_WITH_LOG(ge_tensor_desc.GetShapeRange(shape_range),
        "Call ge::GeTensorDesc::GetShapeRange function failed.")
    if (!shape_range.empty()) {
      AICPUE_LOGI("Get dims from shape range, dim size: [%zu].", shape_range.size());
      for (const auto &dim_item : shape_range) {
        dims.emplace_back(dim_item.second);
      }
    } else {
      ge::GeShape ge_shape = ge_tensor_desc.GetShape();
      dims = ge_shape.GetDims();
    }
  } else {
    ge::GeShape ge_shape = ge_tensor_desc.GetShape();
    dims = ge_shape.GetDims();
  }

  uint32_t dim_shape = dims.size();
  if (!skip_dim_check) {
    for (uint32_t i = 0; i < dim_shape; i++) {
      bool is_invalid_dim = ((dims[i] < 0) &&
                             (dims[i] != ge::UNKNOWN_DIM) &&
                             (dims[i] != ge::UNKNOWN_DIM_NUM));
      if (is_invalid_dim) {
        std::string err_msg =  Stringcat("dim[", i,
            "] is invalid, shape is [", DebugString(dims), "].");
        aicpu::State state(GE_SHAPE_SIZE_INVAILD, err_msg);
        return state;
      }
      tensor_data_info.add_dim(dims[i]);
    }
  } else {
    AICPUE_LOGI("Skip_dim_check for unknown shape");
  }
  return aicpu::State(ge::SUCCESS);
}

/**
 * blocks' size are needed:
 *  1.STR_FWK_OP_KERNEL, this is the task's API struct;
 *  2.InputOuputBuf, defined by protobuf, the struct is KernelRunParam, inside the
 *    struct, input and output buffer's pointer are defined;
 *  3.NodeDefBuf, defined by protobuf, definition is from tensorflow, data is from
 *    GE's graph, need to do the ir transfer;
 *  4.FuncDef, defined by protobuf, for the fused graph.
 */
ge::Status CalcTfOpRunningParam(const ge::Node &node) {
  AICPUE_LOGI("TFKernel's op[%s] run CalcOpRunningParam", node.GetType().c_str());
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, INPUT_PARAM_NULL)

  // check whether the WorkspaceBytes is set
  int64_t workspace_size = 0;
  std::vector<int64_t> workspace_bytes = op_desc_ptr->GetWorkspaceBytes();
  if ((workspace_bytes.empty()) || (workspace_bytes[0] <= 0)) {
    // calc and set WorkspaceBytes
    AICPU_CHECK_RES_WITH_LOG(CalcWorkspaceSize(node, workspace_size),
        "Call TfKernelBuilder::CalcWorkspaceSize function failed, op[%s].",
        node.GetName().c_str())
    op_desc_ptr->SetWorkspaceBytes({workspace_size});
  } else {
    workspace_size = workspace_bytes[0];
    AICPUE_LOGI("Op type[%s] Workspace size already exist, workspace_size is [%ld]",
                node.GetType().c_str(), workspace_size);
  }

  AICPU_CHECK_RES_WITH_LOG(SetOutPutsSize(op_desc_ptr),
      "Call SetOutPutsSize function failed, op[%s].",
      node.GetName().c_str())
  // Set workspace memory reuse flag
  AICPU_CHECK_FALSE_EXEC(ge::AttrUtils::SetListBool(op_desc_ptr, kWorkspaceReuseFlag, {false}),
      AICPU_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::SetListBool Failed to set attr[%s], op[%s].",
          kWorkspaceReuseFlag.c_str(), node.GetName().c_str());
      return ErrorCode::ADD_ATTR_FAILED)
  AICPUE_LOGI("Op type[%s] Calc the Op running param successfully, workspace_size is [%ld]",
              node.GetType().c_str(), workspace_size);
  return ge::SUCCESS;
}
} // namespace aicpu
