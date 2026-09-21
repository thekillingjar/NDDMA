/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/op_info_common.h"
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "common/configuration.h"
#include "common/fe_inner_attr_define.h"
#include "common/fe_type_utils.h"
#include "common/unknown_shape_util.h"
#include "common/graph/fe_graph_utils.h"
#include "platform/platform_info.h"
#include "common/platform_utils.h"
#include "common/scope_allocator.h"
#include "common/math_util.h"
#include "expand_dimension.h"
#include "graph/ge_context.h"
#include "graph/utils/op_desc_utils.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "ops_store/ops_kernel_manager.h"
#include "graph/utils/op_type_utils.h"
namespace fe {
std::mutex graph_attr_mutex;

namespace {
const uint32_t CHECK_WEIGHT_MAX_COUNT = 100;
const std::unordered_set<std::string> kNonComputeNodeTypeSet {
        DATA, REFDATA, VARIABLE, NETOUTPUT, CONSTANT, CONSTANTOP, OP_TYPE_PLACE_HOLDER, OP_TYPE_END};

const std::unordered_set<ge::Format> kSupportedTransType = {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ,
                                                            ge::FORMAT_ND_RNN_BIAS, ge::FORMAT_FRACTAL_ZN_RNN};

const std::unordered_set<ge::DataType> kComplexDtypeSet = {ge::DT_COMPLEX32, ge::DT_COMPLEX64, ge::DT_COMPLEX128};

// {c0_value, bit_value}: c0_value = 2 ^ (bit_value - 1)
const std::map<int64_t, int32_t> kC0PowerMap = {{1, 1}, {2, 2}, {4, 3}, {8, 4}, {16, 5},
                                                {32, 6}, {64, 7}, {128, 8}, {256, 9}};
}

void ExpandDimension(const ge::Format &origin_format, const ge::Format &format, const string &reshape_type,
                     ge::GeShape &shape, const ge::GeTensorDescPtr tensor_desc) {
  int64_t reshape_type_mask = 0;
  (void)transformer::ExpandDimension::GenerateReshapeType(origin_format, format, shape.GetDimNum(),
                                                          reshape_type, reshape_type_mask);
  if (tensor_desc != nullptr) {
    (void)ge::AttrUtils::SetInt(tensor_desc, ge::ATTR_NAME_RESHAPE_TYPE_MASK, reshape_type_mask);
  }
  transformer::ExpandDimension::ExpandDims(reshape_type_mask, shape);
}

void ExpandDimension(const ge::Format &origin_format, const ge::Format &format, const string &reshape_type,
                     const ge::GeShape &origin_shape, ge::GeShape &shape, const ge::GeTensorDescPtr tensor_desc) {
  int64_t reshape_type_mask = 0;
  (void)transformer::ExpandDimension::GenerateReshapeType(origin_format, format, origin_shape.GetDimNum(),
                                                          reshape_type, reshape_type_mask);
  if (tensor_desc != nullptr) {
    (void)ge::AttrUtils::SetInt(tensor_desc, ge::ATTR_NAME_RESHAPE_TYPE_MASK, reshape_type_mask);
  }
  if (reshape_type_mask == 0) {
    shape = origin_shape;
  } else {
    transformer::ExpandDimension::ExpandDims(reshape_type_mask, origin_shape, shape);
  }
}

void GetReshapeAxisValueByName(const ge::Format &origin_format, const ge::GeShape &shape,
                               char axic_name, ge::GeTensorDesc &tensor_desc) {
  int64_t reshape_type_mask = -1;
  if (!ge::AttrUtils::GetInt(tensor_desc, ge::ATTR_NAME_RESHAPE_TYPE_MASK, reshape_type_mask)) {
    return;
  }
  auto c_dim = transformer::ExpandDimension::GetReshapeAxicValueByName(reshape_type_mask,
                                                                       axic_name,
                                                                       shape,
                                                                       origin_format);
  FE_LOGD("GetReshapeAxisValueByName get_cdim = %ld.", c_dim);
  (void)ge::AttrUtils::SetInt(tensor_desc, fe::ATTR_NAME_RESHAPE_CXVALUE, c_dim);
}

int64_t GetAxisValueByName(char axic_name, ge::GeTensorDesc &tensor_desc) {
  int64_t c_dim = -1;
  auto idx = transformer::ExpandDimension::GetAxisIndexByName(axic_name, tensor_desc.GetOriginFormat());
  if (idx == -1) {
    return c_dim;
  }
  return tensor_desc.GetOriginShape().GetDim(idx);
}

Status GetDefaultReshapeType(const ge::Format& original_format, size_t old_dims_size, std::string& reshape_type) {
  bool ret = transformer::ExpandDimension::GetDefaultReshapeType(original_format, old_dims_size, reshape_type);
  return ret ? SUCCESS : FAILED;
}

Status GetShapeAccordingToFormat(ShapeAndFormat &shape_and_format) {
  shape_and_format.new_shape = shape_and_format.old_shape;
  if (static_cast<ge::Format>(ge::GetPrimaryFormat(shape_and_format.old_format)) ==
      static_cast<ge::Format>(ge::GetPrimaryFormat(shape_and_format.new_format))) {
    FE_LOGD("Origin format and format is same, no need to transfer shape");
    return SUCCESS;
  }
  if (shape_and_format.old_shape.IsUnknownDimNum()) {
    FE_LOGD("Do not need to do shape transformation for unknown rank case.");
    return SUCCESS;
  }
  int64_t c0_value = 0;
  if (ge::HasC0Format(shape_and_format.new_format)) {
    c0_value = ge::GetC0Value(shape_and_format.new_format);
  }
  ge::Format format = shape_and_format.new_format;
  if (shape_and_format.group_count > GROUPS_DEFAULT_VALUE) {
    format = static_cast<ge::Format>(ge::GetFormatFromSub(static_cast<int32_t>(format),
                                                          static_cast<int32_t>(shape_and_format.group_count)));
  }

  transformer::ExtAxisValue ext_axis = {shape_and_format.extra_attr.input_size, shape_and_format.extra_attr.hidden_size,
      shape_and_format.extra_attr.state_size,
      transformer::TransferShapeUtils::GetM0ByDtype(shape_and_format.current_data_type)};
  int32_t c0_bit = GetC0BitValFromC0(c0_value);
  format = static_cast<ge::Format>(ge::GetFormatFromC0(format, c0_bit));
  bool ret = TransformerShape::TransferShape(shape_and_format.old_format, format, shape_and_format.current_data_type,
                                             ext_axis, shape_and_format.old_shape, shape_and_format.new_shape);
  return ret ? SUCCESS : FAILED;
}

bool IsAxisAligned(const int64_t &dim_value, const int64_t &aligned_value) {
  if (aligned_value == 0) {
    return false;
  }
  return dim_value % aligned_value == 0;
}

int64_t GetC0ValByDataType(const ge::DataType &data_type) {
  return transformer::TransferShapeUtils::GetC0ByDtype(data_type);
}

bool IsAxisC0Aligned(const ge::DataType &dtype, const int64_t &dim_value) {
  int64_t c0 = GetC0ValByDataType(dtype);
  return IsAxisAligned(dim_value, c0);
}

int64_t GetC0ValByOpDescAndDtype(const ge::OpDescPtr &op_desc, const ge::DataType &dtype) {
  if (dtype == ge::DT_FLOAT && PlatformUtils::Instance().IsEnableCubeHighPrecision()) {
    auto cube_op_list = Configuration::Instance(AI_CORE_NAME).GetFp16OpTypeList();
    if (cube_op_list.count(op_desc->GetType()) > 0) {
      return SHAPE_NUMBER_8;
    } else {
      return SHAPE_NUMBER_16;
    }
  }
  return GetC0ValByDataType(dtype);
}

int32_t GetC0BitValFromC0(const int64_t &c0) {
  int32_t c0_bit = 0;
  auto iter = kC0PowerMap.find(c0);
  if (iter != kC0PowerMap.end()) {
    c0_bit = iter->second;
  }
  return c0_bit;
}

int32_t GetC0BitByDataType(const ge::DataType &data_type) {
  int64_t c0_val = GetC0ValByDataType(data_type);
  return GetC0BitValFromC0(c0_val);
}

ge::Format ModifyC0Format(const ge::Format &format, const ge::DataType &new_dtype) {
  int32_t new_c0_bit = GetC0BitByDataType(new_dtype);
  return static_cast<ge::Format>(ge::GetFormatFromC0(format, new_c0_bit));
}

string GetShapeDims(const ge::GeShape &shape) {
  std::vector<int64_t> dims = shape.GetDims();
  return GetShapeDims(dims);
}

string GetShapeDims(const std::vector<int64_t> &shape_vec) {
  string dim_string;
  size_t size = shape_vec.size();
  for (size_t i = 0; i != size; ++i) {
    dim_string += std::to_string(shape_vec[i]);
    if (i != size - 1) {
      dim_string += ", ";
    }
  }
  return dim_string;
}

void CheckStridedReadInConv2d(const vector<ge::NodePtr> &conv_nodes, vector<ge::NodePtr> &fusion_nodes) {
  for (const auto &fusion_node : fusion_nodes) {
    if (fusion_node->GetType() == STRIDEDREAD) {
      return;
    }
  }
  for (const auto &conv2d_node : conv_nodes) {
    auto conv2d_in_nodes = conv2d_node->GetInDataNodes();
    if (!conv2d_in_nodes.empty()) {
      if (conv2d_in_nodes.at(0)->GetType() == STRIDEDREAD) {
        fusion_nodes.emplace_back(conv2d_in_nodes.at(0));
      }
    }
  }
}

/*
 * @brief: check if is TVM type op
 * @param [in] node: node
 * @return bool: check result
 */
bool IsTbeOp(ge::NodePtr node) {
  FE_CHECK((node == nullptr),
           REPORT_FE_ERROR("[SubGraphOpt][UbFusion][IsTbeOP] null node in judging AICoreOp"), return false);
  int64_t type = 0;
  (void)ge::AttrUtils::GetInt(node->GetOpDesc(), ge::ATTR_NAME_IMPLY_TYPE, type);
  const bool res = (type == static_cast<int64_t>(domi::ImplyType::TVM));
  if (!res) {
    FE_LOGD("Op [%s] is not tbe op.", node->GetName().c_str());
    return false;
  }
  return true;
}

/*
 * @brief: get the optype of a node
 * @param [in] node: graph node
 * @param [out] op_type: type represent by string
 * @return bool: get op type ok or not
 */
bool GetOpAttrType(const ge::NodePtr &node, string &op_type, const bool use_op_type) {
  string name = node->GetName();
  if (use_op_type) {
    op_type = node->GetType();
    return true;
  }
  if (!ge::AttrUtils::GetStr(node->GetOpDesc(), kOpPattern, op_type)) {
    FE_LOGD("Node[%s] get pattern unsuccessful.", name.c_str());
    return false;
  }

  if (op_type.empty()) {
    REPORT_FE_ERROR("[SubGraphOpt][UbFusion][GetOpAttrType] optype is empty for node name [%s].", name.c_str());
    return false;
  }
  return true;
}

bool IsFusedOp(const ge::NodePtr &node) {
  int64_t scope_id = 0;
  return ScopeAllocator::GetScopeAttr(node->GetOpDesc(), scope_id);
}

bool IsThread1Node(const ge::NodePtr &node) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    return false;
  }
  ge::NodePtr thread1_node = node->GetOpDesc()->TryGetExtAttr<ge::NodePtr>(kAttrThread1Node, nullptr);
  if (thread1_node != nullptr) {
    FE_LOGD("Thread1 node of op[%s, %s] is not null.", node->GetName().c_str(), node->GetType().c_str());
    return true;
  }
  return false;
}

bool NeedIgnoreOp(const ge::NodePtr &node, const bool use_op_type) {
  (void)use_op_type;
  FE_CHECK((node == nullptr),
           REPORT_FE_ERROR("[SubGraphOpt][UbFusion][NeedIgnOp] null node in judging ValidOp"), return false);
  auto node_type = node->GetType();
  if (node_type == OP_TYPE_PLACE_HOLDER || node_type == OP_TYPE_END) {
    return true;
  }
  bool no_task = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDesc(), ge::ATTR_NAME_NOTASK, no_task);
  if (no_task) {
    return false;
  }
  // TBE core, fused pattern and hasn't fused op can not be ignore
  if (!IsTbeOp(node)) {
    FE_LOGD("Node [%s] is not tbe op, and will be skipped ub fusion.", node->GetName().c_str());
    return true;
  }

  if (IsFusedOp(node)) {
    FE_LOGD("This node[%s] has been fused.", node->GetName().c_str());
    return true;
  }
  bool stc_to_dyn_soft_sync = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDesc(), kStaticToDynamicSoftSyncOp, stc_to_dyn_soft_sync);
  bool stc_tiling_depend = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDesc(), kDynamicTilingDependOp, stc_tiling_depend);
  if (stc_to_dyn_soft_sync || stc_tiling_depend) {
    FE_LOGD("Node [%s] is soft sync or stc_tiling_depend op, not support fusion.", node->GetName().c_str());
    return true;
  }
  return false;
}

bool IsPlaceOrEnd(const std::string& op_type) {
  return PLACE_OR_END_SET.count(op_type) > 0 ? true : false;
}

bool IsNoTaskOp(const ge::NodePtr &node) {
  bool no_task = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDesc(), ge::ATTR_NAME_NOTASK, no_task);
  return no_task;
}

bool CheckVirtualOp(const ge::OpDescPtr op_desc_ptr) {
  bool no_task = false;
  (void)ge::AttrUtils::GetBool(op_desc_ptr, ge::ATTR_NAME_NOTASK, no_task);
  return no_task;
}

bool IsNd(const ge::Format& format) { return format == ge::FORMAT_ND; }

bool IsSupportedTransType(const ge::Format& ori_format, const ge::Format& final_format) {
  if (kSupportedTransType.count(final_format) != 0 ||
      (ori_format == ge::FORMAT_ND && final_format == ge::FORMAT_FRACTAL_Z)) {
    return true;
  }
  return false;
}

bool CheckOpConstOrVariableInOriGraph(const ge::OpDescPtr &op_desc) {
  string op_type = op_desc->GetType();
  if (op_type == fe::CONSTANT || op_type == fe::CONSTANTOP || op_type == fe::VARIABLE) {
    return true;
  }

  return false;
}

ge::Format GetCurOpOriginFormat(const ge::GeTensorDesc& cur_tensor_desc) {
  auto cur_format = static_cast<ge::Format>(ge::GetPrimaryFormat(cur_tensor_desc.GetFormat()));
  if (cur_format == ge::FORMAT_ND) {
    return cur_tensor_desc.GetOriginFormat();
  } else {
    return cur_format;
  }
}

ge::GeShape GetCurOpOriginShape(const ge::GeTensorDesc& cur_tensor_desc) {
  ge::Format cur_format = static_cast<ge::Format>(ge::GetPrimaryFormat(cur_tensor_desc.GetFormat()));
  ge::GeShape cur_shape = cur_tensor_desc.GetShape();
  if (cur_format == ge::FORMAT_ND) {
    return cur_tensor_desc.GetOriginShape();
  } else {
    return cur_shape;
  }
}

void LogFormatMap(const map<string, vector<ge::Format>>& format_map) {
  map<string, vector<ge::Format>>::const_iterator iter;
  for (iter = format_map.begin(); iter != format_map.end(); ++iter) {
    const vector<ge::Format> &format_vec = iter->second;
    const string &str = GetStrByFormatVec(format_vec);
    FE_LOGD("name=[%s], formats=[%s].", iter->first.c_str(), str.c_str());
  }
}

void LogDataTypeMap(const map<string, vector<ge::DataType>>& data_type_map) {
  map<string, vector<ge::DataType>>::const_iterator iter;
  for (iter = data_type_map.begin(); iter != data_type_map.end(); ++iter) {
    const vector<ge::DataType> &data_type_vec = iter->second;
    const string &str = GetStrByDataTypeVec(data_type_vec);
    FE_LOGD("Data type of [%s] is [%s].", iter->first.c_str(), str.c_str());
  }
}

void LogSubformatMap(const map<string, vector<uint32_t>>& subformat_map) {
  map<string, vector<uint32_t>>::const_iterator iter;
  for (iter = subformat_map.begin(); iter != subformat_map.end(); ++iter) {
    const vector<uint32_t> &subformat_vec = iter->second;
    const string &str = GetStrBySubFormatVec(subformat_vec);
    FE_LOGD("name=[%s], subformats=[%s].", iter->first.c_str(), str.c_str());
  }
}

Status GenerateUnionFormatDtype(const vector<ge::Format>& old_formats, const vector<ge::DataType>& old_data_types,
                                vector<ge::Format>& new_formats, vector<ge::DataType>& new_data_types) {
  size_t old_formats_size = old_formats.size();
  size_t old_dtypes_size = old_data_types.size();
  if (old_formats.empty() || old_data_types.empty()) {
    FE_LOGI("The old_formats_size [%zu] is 0 or the old_dtypes_size [%zu] is 0.", old_formats_size, old_dtypes_size);
    if (!old_formats.empty()) {
      new_formats = old_formats;
    }
    if (!old_data_types.empty()) {
      new_data_types = old_data_types;
    }
    return SUCCESS;
  }

  for (size_t i = 0; i != old_formats_size; i++) {
    new_formats.insert(new_formats.cend(), old_dtypes_size, old_formats[i]);
    new_data_types.insert(new_data_types.cend(), old_data_types.cbegin(), old_data_types.cend());
  }
  size_t new_formats_size = new_formats.size();
  size_t new_dtypes_size = new_data_types.size();
  if (new_formats_size != new_dtypes_size) {
    REPORT_FE_ERROR(
        "[GraphOpt][FmtJdg][GenFmtUnion] The new_formats_size [%zu] is not same with the new_dtypes_size [%zu].",
        new_formats_size, new_dtypes_size);
    return FAILED;
  }
  return SUCCESS;
}

/* Only a FORMAT_ND tensor will be considered as scalar input or output.
 * If shape is [1] and format is NHWC or NCHW, we are still able to
 * transform it to 5HD. */
bool IsScalarInputOrOutput(const ge::GeShape& shape, ge::Format format) {
  if (shape.IsScalar()) {
    return true;
  }

  return shape.GetDimNum() == 1 && shape.GetDims().at(0) == 1 && format == ge::FORMAT_ND;
}

bool IsScalarShape(const ge::GeShape& shape) {
  return shape.IsScalar() || (shape.GetDimNum() == 1 && shape.GetDim(0) == 1);
}

bool IsSameShape(const ge::GeShape& first_shape, const ge::GeShape& second_shape) {
  if (first_shape.GetDimNum() != second_shape.GetDimNum()) {
    return false;
  }
  size_t dim_value = first_shape.GetDimNum();
  for (size_t i = 0; i < dim_value; i++) {
    if (first_shape.GetDim(i) == SHAPE_UNKNOWN_DIM && second_shape.GetDim(i) == SHAPE_UNKNOWN_DIM) {
      return false;
    }
    if (first_shape.GetDim(i) != second_shape.GetDim(i)) {
      return false;
    }
  }
  return true;
}

bool CheckOriginShapesDimNum(const vector<ge::GeShape>& shapes, const size_t& dim_min) {
  for (auto shape : shapes) {
    if (!CheckOriginShapeDimNum(shape, dim_min)) {
      return false;
    }
  }
  return true;
}

bool CheckAccuracyOriginShapesDimNum(const vector<ge::GeShape>& shapes, const size_t& dim_size) {
  for (auto shape : shapes) {
    if (shape.GetDimNum() != dim_size) {
      return false;
    }
  }
  return true;
}

bool CheckOriginShapeDimNum(const ge::GeShape& shape, const size_t& dim_min) {
  if (shape.GetDimNum() < dim_min) {
    FE_LOGD("The dim_num [%zu] is less than dim_min[%zu].", shape.GetDimNum(), dim_min);
    return false;
  }
  return true;
}

bool CheckOriginFormatsIdentifiable(const vector<ge::Format>& formats) {
  for (auto format : formats) {
    FE_LOGD("The format %s.", ge::TypeUtils::FormatToSerialString(format).c_str());
    if (!CheckOriginFormatIdentifiable(format)) {
      return false;
    }
  }
  return true;
}

bool CheckOriginFormatIdentifiable(const ge::Format& format) {
  if (std::find(FE_IDENTIFIABLE_ORIGIN_FORMAT_VECTOR.begin(), FE_IDENTIFIABLE_ORIGIN_FORMAT_VECTOR.end(), format) ==
      FE_IDENTIFIABLE_ORIGIN_FORMAT_VECTOR.end()) {
    FE_LOGD("The format %s is not identifiable.", ge::TypeUtils::FormatToSerialString(format).c_str());
    return false;
  }
  return true;
}

bool GetDimValueByFormatAndShape(const ge::Format &format, const ge::GeShape &shape, const std::string &axis,
                                 int64_t &dim_value) {
  int32_t dim_index = GetAxisIndexByFormat(format, axis);
  if (dim_index < 0) {
    FE_LOGW("Can not find axis index by format[%s] and axis[%s].",
            FormatToStr(format).c_str(), axis.c_str());
    return false;
  }

  if (static_cast<size_t>(dim_index) >= shape.GetDimNum()) {
    FE_LOGW("Axis index[%u] of format[%s] and axis[%s] should be larger than shape size[%zu].",
            dim_index, FormatToStr(format).c_str(), axis.c_str(), shape.GetDimNum());
    return false;
  }

  dim_value = shape.GetDim(static_cast<size_t>(dim_index));
  return true;
}

Status GetGroupAttributeWithVerify(ge::OpDescPtr op_desc_ptr, int64_t& group) {
  if (op_desc_ptr == nullptr) {
    return FAILED;
  }

  group = GROUPS_DEFAULT_VALUE;
  if (!ge::AttrUtils::GetInt(op_desc_ptr, ATTR_NAME_GROUPS, group)) {
    FE_LOGD("Op Desc[%s, %s] does not have groups attribute, take default value [1].",
            op_desc_ptr->GetName().c_str(),
            op_desc_ptr->GetType().c_str());
    return SUCCESS;
  }
  if (group < GROUPS_DEFAULT_VALUE) {
    REPORT_FE_ERROR(
        "[GraphOpt][Setcheck][GetGrpAttr] Node[%s, %s]: The group value %ld should not be less than one.",
        op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(), group);
    return FAILED;
  }

  return SUCCESS;
}

std::string GetRealNodeType(ge::OpDescPtr OpDescPtr) {
  if (OpDescPtr->GetType() == FRAMEWORKOP) {
    string op_type = "";
    if (!ge::AttrUtils::GetStr(OpDescPtr, ge::ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, op_type)) {
      REPORT_FE_ERROR("[GraphOpt][Setcheck][GetRealNodeType] Get original_type for op[%s] fail!",
                      OpDescPtr->GetName().c_str());
      return FRAMEWORKOP;
    }
    return op_type;
  } else {
    return OpDescPtr->GetType();
  }
}

bool CheckWeightTypeQualifiedWithCount(const ge::NodePtr& weight_node, const string& expected_type,
                                       uint32_t& recursion_count) {
  recursion_count++;
  if (recursion_count > CHECK_WEIGHT_MAX_COUNT) {
    FE_LOGD("The count of CheckWeightTypeQualified recursion has reached %d, now stopping recursion.", recursion_count);
    return false;
  }

  if (weight_node == nullptr) {
    FE_LOGW("Node is nullptr!");
    return false;
  }

  if (weight_node->GetType() == expected_type) {
    return true;
  } else if (FeGraphUtils::IsSubGraphData(weight_node->GetOpDesc())) {
    ge::OutDataAnchorPtr pre_out_data_anchor_ptr = nullptr;
    if (FeGraphUtils::GetPreOutAnchorOfSubData(weight_node, pre_out_data_anchor_ptr) != SUCCESS) {
      FE_LOGW("Parent node of sub graph is not found.");
      return false;
    }
    FE_CHECK_NOTNULL(pre_out_data_anchor_ptr);
    ge::NodePtr parent_node = pre_out_data_anchor_ptr->GetOwnerNode();
    if (parent_node == nullptr) {
      FE_LOGW("Parent node of sub graph is null.");
      return false;
    }
    return CheckWeightTypeQualifiedWithCount(parent_node, expected_type, recursion_count);
  } else {
    auto input_anchors = weight_node->GetAllInDataAnchors();
    if (input_anchors.empty()) {
      return false;
    }

    size_t unsupported_count = 0;
    for (const auto& in_anchor : input_anchors) {
      if (in_anchor == nullptr || in_anchor->GetPeerOutAnchor() == nullptr) {
        unsupported_count++;
        continue;
      }

      if (CheckWeightTypeQualifiedWithCount(in_anchor->GetPeerOutAnchor()->GetOwnerNode(), expected_type,
                                            recursion_count)) {
        /* We need to check all top inputs of the Conv2D's weight. */
        continue;
      } else {
        /* If one of the top inputs of Conv2D's weight is not const, we consider this is not the
         * inference network. */
        return false;
      }
    }

    return unsupported_count != input_anchors.size();
  }
}

bool CheckWeightTypeQualified(const ge::NodePtr& weight_node, const string& expected_type) {
  uint32_t recursion_count = 0;
  return CheckWeightTypeQualifiedWithCount(weight_node, expected_type, recursion_count);
}

void CheckHasNoFather(bool is_input, int32_t index, const ge::NodePtr& node, ge::InDataAnchorPtr& in_data_anchor,
                      bool& has_no_father) {
  if (is_input) {
    in_data_anchor = node->GetInDataAnchor(index);
    has_no_father = (in_data_anchor == nullptr || in_data_anchor->GetPeerOutAnchor() == nullptr ||
                     in_data_anchor->GetPeerOutAnchor()->GetOwnerNode() == nullptr ||
                     in_data_anchor->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc() == nullptr);
  } else {
    has_no_father = true;
  }
}

// If a subgraph has been optimized by L2fusion, some nodes in the subgraph will have the lx_fusion_pass attribute
// if a node in the subgraph has attr lx_fusion_pass and the value is true,
// that means this subgraph should do lxfusion
bool CheckL2FusionFusionStrategy(const ge::ComputeGraph& graph) {
  bool lx_fusion_pass = false;
  for (auto node : graph.GetDirectNode()) {
    (void)ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_LX_FUSION_PASS, lx_fusion_pass);
    if (lx_fusion_pass) {
      return true;
    }
  }
  return false;
}

// If a subgraph has been optimized by L2fusion, some nodes in the subgraph will have the lx_fusion_pass attribute
// if a node in the subgraph has attr lx_fusion_pass and the value is true,
// that means this subgraph should do lxfusion
bool CheckL2BufferFusionStrategy(ge::ComputeGraph& graph) {
  bool lx_fusion_pass = false;
  for (auto node : graph.GetDirectNode()) {
    (void)ge::AttrUtils::GetBool(node->GetOpDesc(), "lx_fusion_pass", lx_fusion_pass);
    if (lx_fusion_pass) {
      return false;
    }
  }
  return true;
}

bool IsNeedReshape(const ge::OpDescPtr& op_desc_ptr) {
  bool need_reshape = true;
  std::string op_name = op_desc_ptr->GetName();
  std::string op_type = op_desc_ptr->GetType();
  std::vector<ge::GeShape> input_shapes;
  for (auto input : op_desc_ptr->GetAllInputsDescPtr()) {
    auto shape = input->GetOriginShape();
    if (IsScalarShape(shape)) {
      FE_LOGD("op[name:%s,type:%s] input has scalar, do not need to reshape.", op_name.c_str(), op_type.c_str());
      need_reshape = false;
    }
    input_shapes.push_back(shape);
  }

  if (need_reshape) {
    ge::GeShape first_shape = input_shapes.front();
    for (auto iter = input_shapes.begin(); iter != input_shapes.end(); iter++) {
      if (!IsSameShape(first_shape, *iter)) {
        FE_LOGD("op[name:%s,type:%s] shape is not same, need to reshape.", op_name.c_str(), op_type.c_str());
        return true;
      }
    }
    need_reshape = false;
  }

  FE_LOGD("Op[name:%s,type:%s] reshape flag is %d.", op_name.c_str(), op_type.c_str(), need_reshape);
  return need_reshape;
}

void CopyWeightAttrToPlaceHolder(ge::NodePtr& node) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    return;
  }

  if (node->GetType() != OP_TYPE_PLACE_HOLDER) {
    return;
  }

  ge::NodePtr parent_node = nullptr;
  parent_node = node->GetOpDesc()->TryGetExtAttr(ATTR_NAME_PARENT_NODE, parent_node);
  if (parent_node == nullptr || parent_node->GetOpDesc() == nullptr) {
    return;
  }

  if (parent_node->GetType() != CONSTANT && parent_node->GetType() != CONSTANTOP) {
    FE_LOGD("Op type[%s] of parent node[%s] is not const or constant.", parent_node->GetType().c_str(),
            parent_node->GetName().c_str());
    return;
  }

  ge::GeTensorPtr weight = nullptr;
  bool find_weight = ge::AttrUtils::MutableTensor(parent_node->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, weight);
  if (!find_weight || weight == nullptr) {
    FE_LOGD("Can not find attr ATTR_NAME_WEIGHTS for node:%s.", parent_node->GetName().c_str());
    return;
  }

  ge::GeTensor copy_weight = weight->Clone();
  if (!ge::AttrUtils::SetTensor(node->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, copy_weight)) {
    FE_LOGW("Failed to set weight attr value for place holder node[%s].", node->GetName().c_str());
    return;
  }
  FE_LOGD("Clone ATTR_NAME_WEIGHTS for node:%s success.", node->GetName().c_str());
}

bool InvalidMemType(const ge::OpDescPtr& node_desc) {
  std::vector<uint32_t> input_mem_type;
  (void)ge::AttrUtils::GetListInt(node_desc, ge::ATTR_NAME_INPUT_MEM_TYPE_LIST, input_mem_type);
  std::vector<uint32_t> output_mem_type;
  (void)ge::AttrUtils::GetListInt(node_desc, ge::ATTR_NAME_OUTPUT_MEM_TYPE_LIST, output_mem_type);
  for (auto mem_type : input_mem_type) {
    if (mem_type == RT_MEMORY_L1 || mem_type == RT_MEMORY_L2) {
      FE_LOGD("Node [%s] has lx addr input, not optimize.", node_desc->GetName().c_str());
      return true;
    }
  }
  for (auto mem_type : output_mem_type) {
    if (mem_type == RT_MEMORY_L1 || mem_type == RT_MEMORY_L2) {
      FE_LOGD("Node [%s] has lx addr output, not optimize.", node_desc->GetName().c_str());
      return true;
    }
  }
  return false;
}

bool GetFusionScopeAttr(const ge::OpDescPtr &op_desc, int64_t &scope_id) {
  bool is_l1_fusion = false;
  return GetFusionScopeAttr(op_desc, scope_id, is_l1_fusion);
}

bool GetFusionScopeAttr(const ge::OpDescPtr &op_desc, int64_t &scope_id, bool &is_l1_fusion) {
  if (op_desc == nullptr) {
    return false;
  }
  if (ge::AttrUtils::GetInt(op_desc, L1_SCOPE_ID_ATTR, scope_id)) {
    is_l1_fusion = true;
    return true;
  }
  if (ge::AttrUtils::GetInt(op_desc, SCOPE_ID_ATTR, scope_id)) {
    is_l1_fusion = false;
    return true;
  }
  return false;
}

void RemoveL1FusionScopeAttr(const ge::OpDescPtr &op_desc) {
  int64_t scope_id = 0;
  if (ge::AttrUtils::GetInt(op_desc, L1_SCOPE_ID_ATTR, scope_id)) {
    (void)ge::AttrUtils::SetInt(op_desc, FAILED_L1_SCOPE_ID_ATTR, scope_id);
    (void)op_desc->DelAttr(L1_SCOPE_ID_ATTR);
  }
}

bool IsOpDynamicImpl(const ge::OpDescPtr &op_desc_ptr) {
  if (op_desc_ptr == nullptr) {
    return false;
  }
  bool is_dynamic_impl = false;
  return ge::AttrUtils::GetBool(op_desc_ptr, ATTR_NAME_IS_OP_DYNAMIC_IMPL, is_dynamic_impl) && is_dynamic_impl;
}

bool IsOpDynamicImpl(const ge::OpDesc &op_desc) {
  bool is_dynamic_impl = false;
  return ge::AttrUtils::GetBool(op_desc, ATTR_NAME_IS_OP_DYNAMIC_IMPL, is_dynamic_impl) && is_dynamic_impl;
}

bool IsDtypeSensitiveOp(const ge::OpDescPtr &op_desc) {
  if (op_desc == nullptr) {
    return false;
  }
  if (op_desc->GetType() == CAST) {
    return true;
  }
  for (size_t i = 0; i < op_desc->GetAllInputsSize(); ++i) {
    ge::ConstGeTensorDescPtr tensor_desc_ptr = op_desc->GetInputDescPtr(static_cast<uint32_t>(i));
    if (tensor_desc_ptr == nullptr) {
      continue;
    }
    if (kComplexDtypeSet.count(tensor_desc_ptr->GetDataType()) > 0) {
      return true;
    }
  }
  for (size_t i = 0; i < op_desc->GetOutputsSize(); ++i) {
    ge::ConstGeTensorDescPtr tensor_desc_ptr = op_desc->GetOutputDescPtr(static_cast<uint32_t>(i));
    if (tensor_desc_ptr == nullptr) {
      continue;
    }
    if (kComplexDtypeSet.count(tensor_desc_ptr->GetDataType()) > 0) {
      return true;
    }
  }
  return false;
}

bool IsZeroShapeTensor(const ge::GeTensorDescPtr &tensor) {
  if (tensor == nullptr) {
    return false;
  }
  auto &shape = tensor->MutableShape();
  for (size_t i = 0; i < shape.GetDimNum(); i++) {
    if (shape.GetDim(i) == 0) {
      return true;
    }
  }
  return false;
}

bool IsStaticZeroShapeOp(const ge::OpDescPtr &op_desc) {
  for (size_t i = 0; i < op_desc->GetAllInputsSize(); i++) {
    auto tensor = op_desc->MutableInputDesc(i);
    if (tensor == nullptr) {
      continue;
    }

    if (IsZeroShapeTensor(tensor) &&
        !tensor->MutableShape().IsUnknownShape()) {
      return true;
    }
  }

  for (size_t i = 0; i < op_desc->GetAllOutputsDescSize(); i++) {
    auto tensor = op_desc->MutableOutputDesc(i);
    if (tensor == nullptr) {
      continue;
    }

    if (IsZeroShapeTensor(tensor) &&
        !tensor->MutableShape().IsUnknownShape()) {
      return true;
    }
  }
  return false;
}

bool IsAtomicStaticReuseBinaryOp(const ge::NodePtr &node) {
  auto type = node->GetType();
  if (type != ATOMIC_CLEAN_OP_TYPE && type != MEMSET_OP_TYPE) {
    return true;
  }
  auto out_nodes = node->GetOutNodes();
  if (out_nodes.empty()) {
    return true;
  }
  for (auto &out_node : out_nodes) {
    if (!UnknownShapeUtils::IsUnknownShapeOp(*out_node->GetOpDesc())) {
      return true;
    }
  }
  return false;
}

bool IsStaticOrAutoFuseReuseBinaryOp(const ge::NodePtr &node) {
  auto op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    return false;
  }
  if (UnknownShapeUtils::IsUnknownShapeOp(*op_desc)) {
    return false;
  }
  if (ge::OpTypeUtils::IsAutofuseNode(op_desc)) {
    FE_LOGD("Op[name:%s] is auto fused node.", op_desc->GetName().c_str());
    return true;
  }
  bool stc_tiling_depend = false;
  (void)ge::AttrUtils::GetBool(op_desc, kDynamicTilingDependOp, stc_tiling_depend);
  if (stc_tiling_depend) {
    return false;
  }
  return op_desc->HasAttr(COMPILE_INFO_JSON);
}

bool IsLifeCycleEnd(const ge::Node &node, const ge::GeTensorDescPtr &input_desc,
                    int input_idx) {
  auto op_desc = node.GetOpDesc();
  auto op_name = op_desc->GetName();
  auto op_type = op_desc->GetType();

  bool is_life_cycle_end = false;
  if (ge::AttrUtils::HasAttr(input_desc, ge::ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE)) {
    (void)ge::AttrUtils::GetBool(input_desc, ge::ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE, is_life_cycle_end);
    FE_LOGD("Op[name=%s,type=%s,input=%d]: has attr %s, the life_cycle is %s.", op_name.c_str(), op_type.c_str(),
            input_idx, ge::ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE.c_str(), is_life_cycle_end ? "end" : "not end");
    return is_life_cycle_end;
  }
  return is_life_cycle_end;
}

bool IsAiCoreOp(const ge::NodePtr &node) {
  ge::OpDescPtr op_desc = node->GetOpDesc();
  int64_t imply_value = 0;
  (void)ge::AttrUtils::GetInt(op_desc, FE_IMPLY_TYPE, imply_value);
  OpImplType op_impl_type = GetMainImplType<OpImplType>(imply_value);
  bool is_aicore_op = false;
  is_aicore_op = op_impl_type == EN_IMPL_CUSTOM_TIK || op_impl_type == EN_IMPL_CUSTOM_TBE ||
                 op_impl_type == EN_IMPL_HW_TIK || op_impl_type == EN_IMPL_HW_TBE;
  return is_aicore_op;
}

bool IsTbeOp(const ge::OpDescPtr &op_desc) {
  int64_t imply_value = 0;
  if (!ge::AttrUtils::GetInt(op_desc, FE_IMPLY_TYPE, imply_value)) {
    return false;
  }
  OpImplType op_impl_type = GetMainImplType<OpImplType>(imply_value);
  // now deal with TBE op
  return IsTbe(op_impl_type);
}

bool IsSingleOpGraph(const ge::ComputeGraph& graph) {
  size_t node_count = 0;
  for (const ge::NodePtr &node : graph.GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    if (kNonComputeNodeTypeSet.count(node->GetType()) > 0) {
      continue;
    }
    node_count++;
    if (node_count > 1) {
      break;
    }
  }
  return node_count <= 1;
}

bool GetIsSingleOpFlag(const ge::ComputeGraph& graph, bool &isSingleFlag) {
  std::lock_guard<std::mutex> set_graph_attr_lock(graph_attr_mutex);
  return ge::AttrUtils::GetBool(graph, kFESingleOpScene, isSingleFlag);
}

void SetIsSingleOpFlag(ge::ComputeGraph& graph, const bool &isSingleFlag) {
  std::lock_guard<std::mutex> set_graph_attr_lock(graph_attr_mutex);
  (void)ge::AttrUtils::SetBool(graph, kFESingleOpScene, isSingleFlag);
}

bool IsSingleOpGraphWithCache(ge::ComputeGraph& graph) {
  bool is_single_flag = false;
  if (GetIsSingleOpFlag(graph, is_single_flag)) {
    return is_single_flag;
  }
  is_single_flag = IsSingleOpGraph(graph);
  SetIsSingleOpFlag(graph, is_single_flag);
  FE_LOGD("Graph[%s] single op flag is %d.", graph.GetName().c_str(), is_single_flag);
  return is_single_flag;
}

bool IsStcToDynSoftSyncOp(const ge::OpDescPtr &op_desc) {
  bool stc_to_dyn_soft_sync = false;
  (void)ge::AttrUtils::GetBool(op_desc, kStaticToDynamicSoftSyncOp, stc_to_dyn_soft_sync);
  return stc_to_dyn_soft_sync;
}

std::string GetSessionGraphId(const uint64_t &session_id, const uint32_t &graph_id) {
  return std::to_string(session_id) + "_" + std::to_string(graph_id);
}

std::string GetSessionGraphId(const ge::ComputeGraph &graph) {
  std::string session_graph_id;
  (void)ge::AttrUtils::GetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, session_graph_id);
  if (!session_graph_id.empty()) {
    return session_graph_id;
  }
  return GetSessionGraphId(graph.GetSessionID(), graph.GetGraphID());
}

std::string GetSessionGraphId(const ge::NodePtr &node) {
  std::string session_graph_id;
  if (node == nullptr) {
    return session_graph_id;
  }
  if (node->GetOwnerComputeGraph() == nullptr) {
    return session_graph_id;
  }
  return GetSessionGraphId(*(node->GetOwnerComputeGraph()));
}

bool IsValidOp(const ge::NodePtr &node) {
  const string &op_type = node->GetType();
  string const_op_type;
  bool const_flag = ge::NodeUtils::GetConstOpType(node, const_op_type);
  const std::unordered_set<string> invalid_types = {
          OP_TYPE_PLACE_HOLDER, OP_TYPE_END, RESHAPE, REFORMAT, OP_TYPE_PHONY_CONCAT,
          COMPRESSOP, COMPRESSFCOP, SQUEEZE_V2, UNSQUEEZE_V2
  };

  if ((invalid_types.count(op_type) != 0) || const_flag) {
    return false;
  }
  return true;
}

bool IsCustomOp(const ge::OpDesc &op_desc) {
  int64_t tmp_imply_type = -1;
  if (!ge::AttrUtils::GetInt(op_desc, FE_IMPLY_TYPE, tmp_imply_type)) {
    FE_LOGW("Failed to get fe_imply_type for Node [%s].", op_desc.GetName().c_str());
    return false;
  }
  int impl_type = GetMainImplType<int>(tmp_imply_type);
  if (BUILT_IN_IMPLY_TYPE.count(impl_type) != 0) {
    return false;
  }
  return true;
}

bool IsPrefixOpsPath(const ge::OpDesc &op_desc) {
  string ops_path_name_prefix = "";
  (void)ge::AttrUtils::GetStr(op_desc, OPS_PATH_NAME_PREFIX, ops_path_name_prefix);
  bool ret = ops_path_name_prefix != "";
  if (ret) {
    size_t pos = ops_path_name_prefix.find("_");
    if (pos != string::npos) {
      ops_path_name_prefix = ops_path_name_prefix.substr(static_cast<size_t>(pos + 1));
      FE_LOGD("Node[%s, %s] after substr, ops_path_name_prefix:[%s]",
      op_desc.GetNamePtr(), op_desc.GetTypePtr(), ops_path_name_prefix.c_str());
    } else {
      FE_LOGW("Node[%s, %s] ops_path_name_prefix:[%s] is not in the expected format",
      ops_path_name_prefix.c_str(), op_desc.GetNamePtr(), op_desc.GetTypePtr());
    }
  }
  return ret;
}

bool IsDumpableOp(const ge::OpDescPtr &op_desc) {
  bool isDumpAble = false;
  return ge::AttrUtils::GetBool(op_desc, kAttrDumpAble, isDumpAble) && isDumpAble;
}

std::string GetFusionNodesDescStr(const std::vector<ge::NodePtr> &fusion_nodes) {
  std::string desc_str;
  if (fusion_nodes.empty()) {
    return desc_str;
  }
  bool is_first_node = true;
  for (const ge::NodePtr &node : fusion_nodes) {
    if (node == nullptr) {
      continue;
    }
    if (is_first_node) {
      is_first_node = false;
    } else {
      desc_str += ", ";
    }
    std::string node_desc = "(" + node->GetName() + ", " + node->GetType() + ")";
    desc_str += node_desc;
  }
  return desc_str;
}

std::string GetFusionNodesDescStr(const std::vector<ge::Node*> &fusion_nodes) {
  std::string desc_str;
  if (fusion_nodes.empty()) {
    return desc_str;
  }
  bool is_first_node = true;
  for (const ge::Node *node : fusion_nodes) {
    if (node == nullptr) {
      continue;
    }
    if (is_first_node) {
      is_first_node = false;
    } else {
      desc_str += ", ";
    }
    std::string node_desc = "(" + node->GetName() + ", " + node->GetType() + ")";
    desc_str += node_desc;
  }
  return desc_str;
}

bool IsSuppoertedFormat(const ge::Format cur_heavy_format, const uint32_t &cur_sub_format,
                        const vector<ge::Format> &input_formats, const vector<uint32_t> &input_sub_formats) {
  vector<size_t> index;
  size_t format_len = input_formats.size();
  if (input_sub_formats.empty() && cur_sub_format > DEFAULT_SUB_FORMAT) {
    FE_LOGW("[IsSuppoertedFormat] Support subformat_vec is empty.");
    return false;
  }
  if (std::find(input_sub_formats.begin(), input_sub_formats.end(), SUPPORT_ALL_SUB_FORMAT) ==
      input_sub_formats.end() && cur_sub_format > DEFAULT_SUB_FORMAT &&
      std::find(input_sub_formats.begin(), input_sub_formats.end(), cur_sub_format) ==
      input_sub_formats.end()) {
    FE_LOGD("[IsSuppoertedFormat] Cur sub_format not support.");
    return false;
  }

  for (size_t i = 0; i < format_len; ++i) {
    if (cur_heavy_format == input_formats[i]) {
      FE_LOGD("[IsSuppoertedFormat] Heavyformat and subformat check successfully");
      return true;
    }
  }
  FE_LOGD("[IsSuppoertedFormat] Format not support");
  return false;
}

bool VerifyCastC0Format(const ge::OpDescPtr op_desc_ptr, ge::Format new_format) {
  FE_CHECK(op_desc_ptr == nullptr,
           FE_LOGW("[ChkSpt][FEChk][ChkCastC0] op_desc_ptr is nullptr."), return false);
  // only check C0 for cast node
  // ge may try to swap position of transdata and cast, and check whether they can do it.
  if (op_desc_ptr->GetType() != CAST) {
    return true;
  }
  const auto input_tensor = op_desc_ptr->GetInputDescPtr(0);
  FE_CHECK(input_tensor == nullptr,
           FE_LOGW("[ChkSpt][FEChk][ChkCastC0] input_tensor is nullptr."), return false);
  ge::Format check_format = new_format == ge::FORMAT_MAX ? input_tensor->GetFormat() : new_format;
  auto in_type_size = ge::GetSizeByDataType(input_tensor->GetDataType());
  const auto output_tensor = op_desc_ptr->GetOutputDescPtr(0);
  FE_CHECK(output_tensor == nullptr,
           FE_LOGW("[ChkSpt][FEChk][ChkCastC0] output_tensor is nullptr."), return false);
  auto out_type_size = ge::GetSizeByDataType(output_tensor->GetDataType());
  FE_LOGD("Cast op[%s] in/out data type size[%d/%d] with format %d.", op_desc_ptr->GetNamePtr(), in_type_size,
          out_type_size, check_format);
  if (!ge::HasC0Format(static_cast<int32_t>(check_format))) {
    return true;
  }
  auto primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(check_format));
  if (std::find(FE_HEAVY_FORMAT_VECTOR.begin(), FE_HEAVY_FORMAT_VECTOR.end(), primary_format) ==
         FE_HEAVY_FORMAT_VECTOR.end()) {
    FE_LOGW("Cast op[%s] not heavy format.", op_desc_ptr->GetNamePtr());
    return true;
  }
  if (in_type_size != out_type_size) {
    FE_LOGW("C0 format Cast op in/out data type size[%d/%d] not equal.", in_type_size, out_type_size);
  }
  return in_type_size == out_type_size;
}
}  // namespace fe
