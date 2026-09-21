/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_UTILS_COMMON_OP_INFO_COMMON_H_
#define FUSION_ENGINE_UTILS_COMMON_OP_INFO_COMMON_H_

#include <map>
#include <string>
#include <vector>
#include "common/fe_inner_error_codes.h"
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/string_utils.h"
#include "ops_store/op_kernel_info.h"
#include "graph/types.h"
#include "transfer_shape_according_to_format.h"

namespace fe {
struct HeavyFormatInfo {
  ge::Format expected_heavy_format = ge::FORMAT_RESERVED;
  int32_t sub_format = 0;
  int32_t c0_format = 0;
  int32_t anchor_index = 0;
  bool is_input = false;
  HeavyFormatInfo() {}
  HeavyFormatInfo(ge::Format expected_heavy_format_param, int32_t sub_format_param, int32_t c0_format_param,
                  int32_t anchor_index_param, bool is_input_param)
      : expected_heavy_format(expected_heavy_format_param),
        sub_format(sub_format_param),
        c0_format(c0_format_param),
        anchor_index(anchor_index_param),
        is_input(is_input_param) {}
};

using TransformerShape = transformer::ShapeTransferAccordingToFormat;
#define IS_INPUT_TO_STRING(is_input) ((is_input) ? "input" : "output")

const std::unordered_map<ge::Format, std::unordered_set<ge::Format>> kValidTransFormat = {
        {ge::FORMAT_ND, {ge::FORMAT_FRACTAL_NZ}},
        {ge::FORMAT_NHWC, {ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z}},
        {ge::FORMAT_FRACTAL_NZ, {ge::FORMAT_ND}},
        {ge::FORMAT_NC1HWC0, {ge::FORMAT_NHWC}}
};

struct UpdateInfo {
  const OpKernelInfoPtr &op_kernel_info_ptr;             // op kernel info
  const InputOrOutputInfoPtr &input_or_output_info_ptr;  // tensor kernel info
  const uint32_t &matched_index;                         // mathed index of the op_kernel_info
  const ge::NodePtr &node_ptr;                           // current node pointer
  const uint32_t &index;                                 // the index of the input or output desc
  ge::GeTensorDesc &op_input_or_output_desc;             // the input or output desc of the current node
  const bool &is_input;
};

Status GetDefaultReshapeType(const ge::Format& original_format, size_t old_dims_size, std::string& reshape_type);

void ExpandDimension(const ge::Format &origin_format, const ge::Format &format, const string &reshape_type,
                     ge::GeShape &shape, const ge::GeTensorDescPtr tensor_desc = nullptr);

void ExpandDimension(const ge::Format &origin_format, const ge::Format &format, const string &reshape_type,
                     const ge::GeShape &origin_shape, ge::GeShape &shape,
                     const ge::GeTensorDescPtr tensor_desc = nullptr);

void GetReshapeAxisValueByName(const ge::Format &origin_format, const ge::GeShape &shape,
                               char axic_name, ge::GeTensorDesc &tensor_desc);

Status GetShapeAccordingToFormat(ShapeAndFormat &shape_and_format);

int64_t GetAxisValueByName(char axic_name, ge::GeTensorDesc &tensor_desc);

bool IsAxisAligned(const int64_t &dim_value, const int64_t &aligned_value);

int64_t GetC0ValByDataType(const ge::DataType &data_type);

bool IsAxisC0Aligned(const ge::DataType &dtype, const int64_t &dim_value);

int64_t GetC0ValByOpDescAndDtype(const ge::OpDescPtr &op_desc, const ge::DataType &dtype);

int32_t GetC0BitValFromC0(const int64_t &c0);

int32_t GetC0BitByDataType(const ge::DataType &data_type);

ge::Format ModifyC0Format(const ge::Format &format, const ge::DataType &new_dtype);

string GetShapeDims(const ge::GeShape &shape);

string GetShapeDims(const std::vector<int64_t> &shape_vec);

/**
 * is PlaceHolder, End, Data, Const or Variable
 * @param op_type current op_type
 * @return result
 */
bool IsPlaceOrEnd(const std::string &op_type);

/**
 * is ND or MD
 * @param format current format
 * @return result
 */
bool IsNd(const ge::Format &format);

void CheckStridedReadInConv2d(const vector<ge::NodePtr> &conv_nodes, vector<ge::NodePtr> &fusion_nodes);

bool IsTbeOp(ge::NodePtr node);

bool IsSupportedTransType(const ge::Format& ori_format, const ge::Format& final_format);

bool CheckOpConstOrVariableInOriGraph(const ge::OpDescPtr &op_desc);

ge::Format GetCurOpOriginFormat(const ge::GeTensorDesc &cur_tensor_desc);

ge::GeShape GetCurOpOriginShape(const ge::GeTensorDesc &cur_tensor_desc);

void LogFormatMap(const map<string, vector<ge::Format>> &format_map);

void LogDataTypeMap(const map<string, vector<ge::DataType>> &data_type_map);

void LogSubformatMap(const map<string, vector<uint32_t>>& subformat_map);
/**
 * if old_formats is NCHW,NHWC, old_data_types is float16,
 * then new_formats is NCHW,NHWC, new_data_types is float16,float16
 * @param old_formats old formats
 * @param old_data_types old data_types
 * @param new_formats new formats
 * @param new_data_types new data_types
 * @return SUCCESS or FAILED
 */
Status GenerateUnionFormatDtype(const vector<ge::Format> &old_formats, const vector<ge::DataType> &old_data_types,
                                vector<ge::Format> &new_formats, vector<ge::DataType> &new_data_types);

bool IsScalarInputOrOutput(const ge::GeShape& shape, ge::Format format);

bool IsScalarShape(const ge::GeShape &shape);
bool IsSameShape(const ge::GeShape &first_shape, const ge::GeShape &second_shape);
bool CheckOriginFormatIdentifiable(const ge::Format &format);
bool CheckOriginFormatsIdentifiable(const vector<ge::Format> &formats);
bool CheckOriginShapeDimNum(const ge::GeShape &shape, const size_t &dim_min);

// dimNum must be >= dim_min
bool CheckOriginShapesDimNum(const vector<ge::GeShape> &shapes, const size_t &dim_min);
bool CheckAccuracyOriginShapesDimNum(const vector<ge::GeShape> &shapes, const size_t &dim_size);

bool CheckVirtualOp(const ge::OpDescPtr op_desc_ptr);

bool GetDimValueByFormatAndShape(const ge::Format &format, const ge::GeShape &shape, const std::string &axis,
                                 int64_t &dim_value);

Status GetGroupAttributeWithVerify(ge::OpDescPtr op_desc_ptr, int64_t &group);

std::string GetRealNodeType(ge::OpDescPtr OpDescPtr);

/* Only when the weight node or its predecessor(s) is(are) expected, it's a qualified weight.
 * Because first layer conv feature can only be effective when it's inference scenario.
 * If weight is not expected, we will traverse all the way to the top input node. */
bool CheckWeightTypeQualified(const ge::NodePtr &weight_node, const string& expected_type);

void CheckHasNoFather(bool is_input, int32_t index, const ge::NodePtr &node, ge::InDataAnchorPtr &in_data_anchor,
                      bool &has_no_father);

// If a subgraph has been optimized by L2fusion, some nodes in the subgraph will have the lx_fusion_pass attribute
bool CheckL2FusionFusionStrategy(const ge::ComputeGraph& graph);

// If a subgraph has been optimized by L2buffer, all nodes in the subgraph should have lx_fusion_pass attr:false
bool CheckL2BufferFusionStrategy(ge::ComputeGraph& graph);

// is need to reshape when format is fz or fz_3d
bool IsNeedReshape(const ge::OpDescPtr& op_desc_ptr);

// if parent node of place holder is const, copy weight attr value of const node to place holder node
void CopyWeightAttrToPlaceHolder(ge::NodePtr &node);
// if input or output is lx addr ,think it not valid
bool InvalidMemType(const ge::OpDescPtr &node_desc);

// get _l1_fusion_scope attr value from opdesc first
// if _l1_fusion_scope is not on opdesc, then try to get fusion_scope attr
bool GetFusionScopeAttr(const ge::OpDescPtr &op_desc, int64_t &scope_id);
bool GetFusionScopeAttr(const ge::OpDescPtr &op_desc, int64_t &scope_id, bool &is_l1_fusion);

void RemoveL1FusionScopeAttr(const ge::OpDescPtr &op_desc);

bool IsOpDynamicImpl(const ge::OpDescPtr &op_desc_ptr);
bool IsOpDynamicImpl(const ge::OpDesc &op_desc);

inline bool IsDtypeSensitiveOp(const std::string &op_type) {
  return op_type == CAST;
}

inline bool IsFormatSensitiveOp(const std::string &op_type) {
  return op_type == TRANSDATA;
}

bool IsSuppoertedFormat(const ge::Format cur_heavy_format, const uint32_t &cur_sub_format,
                        const vector<ge::Format> &input_formats, const vector<uint32_t> &input_sub_formats);

bool IsDtypeSensitiveOp(const ge::OpDescPtr &op_desc);

/**
 * If one of the dims is 0, the tensor is a zero-shape tensor.
 */
bool IsZeroShapeTensor(const ge::GeTensorDescPtr &tensor);

/**
 * If one of the tensors is zero shape tensor, the operator is
 * a zero-shape operator.
 */
bool IsStaticZeroShapeOp(const ge::OpDescPtr &op_desc);

bool IsStaticOrAutoFuseReuseBinaryOp(const ge::NodePtr &node);

bool IsAtomicStaticReuseBinaryOp(const ge::NodePtr &node);

bool IsLifeCycleEnd(const ge::Node &node, const ge::GeTensorDescPtr &input_desc, int input_idx);

bool IsAiCoreOp(const ge::NodePtr &node);

bool IsTbeOp(const ge::OpDescPtr &op_desc);

bool IsNoTaskOp(const ge::NodePtr &node);

bool IsSingleOpGraph(const ge::ComputeGraph& graph);

bool IsSingleOpGraphWithCache(ge::ComputeGraph& graph);

bool IsStcToDynSoftSyncOp(const ge::OpDescPtr &op_desc);

std::string GetSessionGraphId(const uint64_t &session_id, const uint32_t &graph_id);

std::string GetSessionGraphId(const ge::ComputeGraph &graph);

std::string GetSessionGraphId(const ge::NodePtr &node);

bool NeedIgnoreOp(const ge::NodePtr &node, const bool use_op_type = false);

bool IsFusedOp(const ge::NodePtr &node);

bool IsThread1Node(const ge::NodePtr &node);

bool GetOpAttrType(const ge::NodePtr &node, string &op_type, const bool use_op_type = false);

bool IsValidOp(const ge::NodePtr &node);

bool IsCustomOp(const ge::OpDesc &op_desc);

bool IsPrefixOpsPath(const ge::OpDesc &op_desc);

bool IsDumpableOp(const ge::OpDescPtr &op_desc);

std::string GetFusionNodesDescStr(const std::vector<ge::NodePtr> &fusion_nodes);

std::string GetFusionNodesDescStr(const std::vector<ge::Node*> &fusion_nodes);

bool CheckWeightTypeQualifiedWithCount(const ge::NodePtr& weight_node, const string& expected_type,
                                       uint32_t& recursion_count);

bool GetIsSingleOpFlag(const ge::ComputeGraph& graph, bool &isSingleFlag);

void SetIsSingleOpFlag(ge::ComputeGraph& graph, const bool &isSingleFlag);

bool VerifyCastC0Format(const ge::OpDescPtr op_desc_ptr, ge::Format new_format = ge::FORMAT_MAX);
}  // namespace fe
#endif  // FUSION_ENGINE_UTILS_COMMON_OP_INFO_COMMON_H_
