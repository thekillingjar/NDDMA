/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/op_slice_util.h"
#include <cmath>
#include "common/lxfusion_json_util.h"
#include "common/string_utils.h"
#include "common/sgt_slice_type.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/type_utils.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"

namespace fe {
namespace {
bool IsScalarInputShape(const ge::GeShape& shape) {
  return shape.IsScalar() || (shape.GetDimNum() == 1 && shape.GetDim(0) == 1);
}

Status SetElemWiseSliceInfoExCheckParam(const ge::OpDesc::Vistor<ge::GeTensorDescPtr> &op_input_desc_list,
                                        const ge::OpDesc::Vistor<ge::GeTensorDescPtr> &op_output_desc_list,
                                        int32_t &intput_more_dims_idx, int32_t &input_more_dims_size, int32_t &diff) {
  if (op_input_desc_list.size() != 2) {
    FE_LOGW("SetElemWiseSliceInfoExCheckParam has %zu inputs, which is not equal to 2.", op_input_desc_list.size());
    return FAILED;
  }
  if (op_output_desc_list.size() != 1) {
    FE_LOGW("SetElemWiseSliceInfoExCheckParam has %zu outputs, which is not equal to 1.", op_output_desc_list.size());
    return FAILED;
  }

  auto &input0_shape = op_input_desc_list.at(0)->GetShape();
  auto &input1_shape = op_input_desc_list.at(1)->GetShape();
  int32_t input0_dim_size = static_cast<int32_t>(input0_shape.GetDimNum());
  int32_t input1_dim_size = static_cast<int32_t>(input1_shape.GetDimNum());
  if (input0_dim_size == input1_dim_size) {
    FE_LOGW("SetElemWiseSliceInfoExCheckParam: input0's size should be equal to input1's size %d", input0_dim_size);
    return FAILED;
  }
  if (IsScalarInputShape(input0_shape) || IsScalarInputShape(input1_shape)) {
    FE_LOGW("SetElemWiseSliceInfoExCheckParam has scalar input.");
    return FAILED;
  }
  bool is_input0_more = input0_dim_size > input1_dim_size;
  intput_more_dims_idx = is_input0_more ? 0 : 1;
  input_more_dims_size = is_input0_more ? input0_dim_size : input1_dim_size;
  int32_t out_dim_size = static_cast<int32_t>(op_output_desc_list.at(0)->GetShape().GetDims().size());
  if (input_more_dims_size != out_dim_size) {
    FE_LOGW("SetElemWiseSliceInfoExCheckParam input_more_dims_size:%d != out_dim_size:%d",
            input_more_dims_size, out_dim_size);
    return FAILED;
  }
  diff = std::abs(input0_dim_size - input1_dim_size);
  return SUCCESS;
}

void InitializeStrategy(const ge::OpDescPtr &op_desc, size_t size, bool is_input,
                        vector<vector<int64_t>> &all_strategies) {
  for (size_t i = 0; i < size; i++) {
    FE_LOGD("Add strategy for tensor %zu.", i);
    ge::GeTensorDescPtr tensor_desc = is_input ? op_desc->MutableInputDesc(i) : op_desc->MutableOutputDesc(i);
    if (tensor_desc == nullptr) {
      continue;
    }
    size_t dim_num = tensor_desc->GetShape().GetDimNum();
    vector<int64_t> default_st(dim_num, 0);
    all_strategies.emplace_back(default_st);
  }
}

void GenOneStrategy(size_t tensor_index, const std::vector<int64_t>& axis,
                    const ge::GeTensorDescPtr& tensor_desc,
                    vector<vector<int64_t>> &all_strategies) {
  if (tensor_index >= all_strategies.size()) {
    FE_LOGW("tensor index %zu is larger than the size of strategies(%zu).",
            tensor_index, all_strategies.size());
    return;
  }
  vector<int64_t> one_strategy;
  size_t dim_num = tensor_desc->GetShape().GetDimNum();
  for (size_t i = 0; i < dim_num; i++) {
    if (std::find(axis.begin(), axis.end(), static_cast<int64_t>(i)) != axis.end()) {
      one_strategy.push_back(1);
    } else {
      one_strategy.push_back(0);
    }
  }

  all_strategies[tensor_index] = one_strategy;
}

void SetCutInfoAttr(const ge::OpDescPtr &op_desc, size_t size, bool is_input,
                    const vector<vector<int64_t>> &all_strategies) {
  for (size_t i = 0; i < size; i++) {
    auto tensor_desc =
        is_input ? op_desc->MutableInputDesc(i) : op_desc->MutableOutputDesc(i);
    if (tensor_desc == nullptr) {
      continue;
    }
    FE_LOGD("Set cut_info for input at index %zu.", i);
    vector<vector<int64_t>> current_stgy;
    (void)ge::AttrUtils::GetListListInt(tensor_desc, "_cut_info", current_stgy);
    current_stgy.emplace_back(all_strategies[i]);
    (void)ge::AttrUtils::SetListListInt(tensor_desc, "_cut_info", current_stgy);
  }
}
} // namespace

const size_t kSplitSize = 2;
const int8_t kAxisHIndex = 2;
const int8_t kAxisWIndex = 3;
const int64_t kAxesValue = 4;

static const vector<string> LIST_OF_SUPPORT_L1_OP = {"Conv2D", "DepthwiseConv2D", "Deconvolution", "Pooling",
                                                     "FullyConnection", "AscendQuant", "ConcatV2D", "Eltwise", "Scale",
                                                     "BNInference", "BNInferenceD"};
static const string ADD = "Add";
const std::map<SlicePattern, FillupSliceInfoPtr> OpSliceUtil::split_info_map_ = {
    {ELEMENT_WISE, std::make_shared<FillupSliceInfo>(FillupElemwiseSliceInfo)},
    {ELEMENT_WISE_BROADCAST, std::make_shared<FillupSliceInfo>(FillupElemwiseBroadcastSliceInfo)},
    {BROADCAST, std::make_shared<FillupSliceInfo>(FillupBroadcastSliceInfo)},
    {SLIDING_WINDOW, std::make_shared<FillupSliceInfo>(FillupSlidingWindowSliceInfo)},
    {SLIDING_WINDOW_DECONV, std::make_shared<FillupSliceInfo>(FillupSlidingWindowDeconvSliceInfo)},
    {CUBE_MATMUL, std::make_shared<FillupSliceInfo>(FillupCubeMatmulSliceInfo)},
    {SLICE_PATTERN_REDUCE, std::make_shared<FillupSliceInfo>(FillupReduceSliceInfo)},
    {SLICE_PATTERN_RESIZE, std::make_shared<FillupSliceInfo>(FillupResizeSliceInfo)},
    {SLICE_PATTERN_SCATTER, std::make_shared<FillupSliceInfo>(FillupScatterSliceInfo)},
    {SLICE_PATTERN_SEGMENT, std::make_shared<FillupSliceInfo>(FillupSegmentSliceInfo)},
};
OpSliceUtil::OpSliceUtil() {}
OpSliceUtil::~OpSliceUtil() {}

Status OpSliceUtil::SetOpSliceInfo(const ge::NodePtr &node, const SlicePattern &slice_pattern,
                                   const bool &support_stride_write) {
  FE_CHECK_NOTNULL(node);
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);
  FE_LOGD("Start setting slice info for node [%s].", op_desc_ptr->GetName().c_str());
  string op_name = op_desc_ptr->GetName();
  string op_type = op_desc_ptr->GetType();

  ge::GeTensorDescPtr output_tensor_ptr = op_desc_ptr->MutableOutputDesc(0);
  if (output_tensor_ptr == nullptr) {
    return SUCCESS;
  }
  ge::GeShape output_shape = output_tensor_ptr->GetShape();
  if (output_shape.IsUnknownDimNum()) {
    FE_LOGD("The dim num of op[%s, %s]'s output is unknown. No need to generate slice info.",
            op_name.c_str(), op_type.c_str());
    return SUCCESS;
  }

  OpCalcInfo op_calc_info;
  if (!op_calc_info.Initialize()) {
    REPORT_FE_ERROR("[OpSliceUtil][SetOpSlicePtnInfo] op_calc_info initialize failed");
    return FAILED;
  }
  InputSplitInfo input_split_info;
  if (!input_split_info.Initialize()) {
    REPORT_FE_ERROR("[OpSliceUtil][SetOpSlicePtnInfo] input_split_info initialize failed");
    return FAILED;
  }
  OutputSplitInfo output_split_info;
  if (!output_split_info.Initialize()) {
    REPORT_FE_ERROR("[OpSliceUtil][SetOpSlicePtnInfo] output_split_info initialization failed");
    return FAILED;
  }

  auto slice_func_iter = split_info_map_.find(slice_pattern);
  if (slice_func_iter == split_info_map_.end()) {
    FE_LOGD("Not currently supported: op %s.slice_pattern:%d", op_desc_ptr->GetName().c_str(), slice_pattern);
    return SUCCESS;
  }
  // 3. set slice info for each supported slice pattern
  FillupSliceInfoPtr slice_func_ptr = nullptr;
  FE_MAKE_SHARED(slice_func_ptr = slice_func_iter->second, return FAILED);
  FE_CHECK_NOTNULL(slice_func_ptr);
  Status status = (*slice_func_ptr)(op_desc_ptr, op_calc_info, support_stride_write);
  if (status != SUCCESS) {
    FE_LOGD("Parsing slice pattern [%d] information for node [%s] did not match; slice information is not set.",
            slice_pattern, op_desc_ptr->GetName().c_str());
    return SUCCESS;
  }
  // set whether support l1fusion
  auto is_l1_node = find(LIST_OF_SUPPORT_L1_OP.begin(), LIST_OF_SUPPORT_L1_OP.end(), op_type);
  if (is_l1_node != LIST_OF_SUPPORT_L1_OP.end()) {
    FE_LOGD("Node [%s, %s]: It supports L1 fusion.", op_name.c_str(), op_type.c_str());
    op_calc_info.SetL1FusionEnable(L1FUSION_BASIC);
  }
  std::string op_calc_info_str;
  SetOpSliceInfoToJson(op_calc_info, op_calc_info_str);
  (void)ge::AttrUtils::SetStr(op_desc_ptr, OP_SLICE_INFO, op_calc_info_str);
  FE_LOGD("Set node[%s]'s slice info as %s", op_desc_ptr->GetName().c_str(), op_calc_info_str.c_str());
  return SUCCESS;
}

/*
 *  @ingroup fe
 *  @brief   Set the slice information for the node whose SlicePattern is Elemwise in the ops info config.
 *  @param   [in|out] node | slice info
 *  @return  SUCCESS or FAILED
 */
Status OpSliceUtil::FillupElemwiseSliceInfo(ge::OpDescPtr op_desc_ptr, OpCalcInfo &op_calc_info, const bool& sup_sw) {
  FE_LOGD("Start to set node[%s]'s slice pattern as elemwise.", op_desc_ptr->GetName().c_str());
  std::vector<AxisSplitMap> axis_split_maps;
  Status ret = SetElemWiseSliceInfo(op_desc_ptr, axis_split_maps, false, sup_sw);
  if (ret != SUCCESS) {
    FE_LOGD("Parsing operation slice information for node [%s] did not match; slice information was not set.", op_desc_ptr->GetName().c_str());
    return SUCCESS;
  }
  op_calc_info.SetAxisSplitMaps(axis_split_maps);
  return SUCCESS;
}

/*
 *  @ingroup fe
 *  @brief   Set the slice information for the node whose SlicePattern is ElemwiseBroadcast in the ops info config.
 *  @param   [in|out] node | slice info
 *  @return  SUCCESS or FAILED
 */
Status OpSliceUtil::FillupElemwiseBroadcastSliceInfo(ge::OpDescPtr op_desc_ptr, OpCalcInfo &op_calc_info,
                                                     const bool& sup_sw) {
  // slice info of elemwise-broadcast is the same as elemwise by now
  FE_LOGD("Start to set node[%s]'s slice pattern as elemwise-broadcast.", op_desc_ptr->GetName().c_str());
  std::vector<AxisSplitMap> axis_split_maps;
  Status ret = SetElemWiseSliceInfo(op_desc_ptr, axis_split_maps, true, sup_sw, true);
  if (ret != SUCCESS) {
    FE_LOGD("Parse op slice info for node[%s] not matched, SetElemWiseSliceInfoEx", op_desc_ptr->GetName().c_str());
    ret = SetElemWiseSliceInfoEx(op_desc_ptr, axis_split_maps);
    if (ret != SUCCESS) {
      FE_LOGD("SetElemWiseSliceInfoEx for node not matched");
      return SUCCESS;
    }
  }
  op_calc_info.SetAxisSplitMaps(axis_split_maps);
  return ret;
}

/*
 *  @ingroup fe
 *  @brief   Set the slice information for the node whose SlicePattern is Broadcast in the ops info config.
 *  @param   [in|out] node | slice info
 *  @return  SUCCESS or FAILED
 */
Status OpSliceUtil::FillupBroadcastSliceInfo(ge::OpDescPtr op_desc_ptr, OpCalcInfo &op_calc_info, const bool& sup_sw) {
  // slice info of broadcast is the same as elemwise by now
  FE_LOGD("Start to set node[%s]'s slice pattern as broadcast.", op_desc_ptr->GetName().c_str());
  std::vector<AxisSplitMap> axis_split_maps;
  Status ret = SetElemWiseSliceInfo(op_desc_ptr, axis_split_maps, false, sup_sw, true);
  if (ret != SUCCESS) {
    FE_LOGD("Parsing operation slice information for node [%s] did not match; slice information was not set.", op_desc_ptr->GetName().c_str());
    return SUCCESS;
  }
  op_calc_info.SetAxisSplitMaps(axis_split_maps);
  return ret;
}

/*
 *  @ingroup fe
 *  @brief   Set the slice information for the node whose SlicePattern is SlidingWindow in the ops info config.
 *  @param   [in|out] node | slice info
 *  @return  SUCCESS or FAILED
 */
Status OpSliceUtil::FillupSlidingWindowSliceInfo(ge::OpDescPtr op_desc_ptr, OpCalcInfo &op_calc_info,
                                                 const bool& sup_sw) {
  // In sliding-window, the slice information is fixed and must be in the following format:
  FE_LOGD("Start to set node[%s]'s slice pattern as sliding-window.", op_desc_ptr->GetName().c_str());
  std::vector<AxisSplitMap> axis_split_maps;
  Status ret = SetSlidingWindowSliceInfo(op_desc_ptr, axis_split_maps, sup_sw);
  if (ret != SUCCESS) {
    FE_LOGD("Parsing operation slice information for node [%s] did not match; slice information was not set.", op_desc_ptr->GetName().c_str());
    return SUCCESS;
  }
  op_calc_info.SetAxisSplitMaps(axis_split_maps);
  return SUCCESS;
}

/*
 *  @ingroup fe
 *  @brief   Set the slice information for the node whose SlicePattern is SlidingWindowDeconv in the ops info config.
 *  @param   [in|out] node | slice info
 *  @return  SUCCESS or FAILED
 */
Status OpSliceUtil::FillupSlidingWindowDeconvSliceInfo(ge::OpDescPtr op_desc_ptr, OpCalcInfo &op_calc_info,
                                                       const bool& sup_sw) {
  // slice info of liding-window-deconv is the same as liding-window by now
  FE_LOGD("Start to set node[%s]'s slice pattern as sliding-window-deconv.", op_desc_ptr->GetName().c_str());
  std::vector<AxisSplitMap> axis_split_maps;
  Status ret = SetSlidingWindowDeconvSliceInfo(op_desc_ptr, axis_split_maps, sup_sw);
  if (ret != SUCCESS) {
    FE_LOGD("Parsing operation slice information for node [%s] did not match; slice information was not set.", op_desc_ptr->GetName().c_str());
    return SUCCESS;
  }
  op_calc_info.SetAxisSplitMaps(axis_split_maps);
  return SUCCESS;
}

/*
 *  @ingroup fe
 *  @brief   Set the slice information for the node whose SlicePattern is CubeMatmul in the ops info config.
 *  @param   [in|out] node | slice info
 *  @return  SUCCESS or FAILED
 */
Status OpSliceUtil::FillupCubeMatmulSliceInfo(ge::OpDescPtr op_desc_ptr, OpCalcInfo &op_calc_info, const bool& sup_sw) {
  // In cube-matmul, the slice information is fixed and must be in the following format:
  FE_LOGD("Start to set node[%s]'s slice pattern as cube-matmul.", op_desc_ptr->GetName().c_str());
  std::vector<AxisSplitMap> axis_split_maps;
  std::vector<AxisReduceMap> axis_reduce_maps;
  FE_CHECK_NOTNULL(op_desc_ptr->GetInputDescPtr(0));
  Status ret = SetCubeMatmulSliceInfo(op_desc_ptr, axis_split_maps, sup_sw);
  if (ret != SUCCESS) {
    FE_LOGD("Parsing operation slice information for node [%s] did not match; slice information was not set.", op_desc_ptr->GetName().c_str());
    return SUCCESS;
  }
  // cube matmul node should have more than 2 inputs
  if (op_desc_ptr->GetInputsSize() > 2) {
    FE_LOGD("Node [%s]: It has bias input and does not support cutting on the k-axis.", op_desc_ptr->GetName().c_str());
  } else {
    auto input_format = static_cast<ge::Format>(ge::GetPrimaryFormat(op_desc_ptr->GetInputDescPtr(0)->GetFormat()));
    AxisReduceMap axis_reduce_map_cut;
    if (!axis_reduce_map_cut.Initialize()) {
      REPORT_FE_ERROR("[OpSliceUtil][FillupCubeMulSliceInfo] Failed to initialize reduce map.");
      return FAILED;
    }

    if (input_format == ge::FORMAT_NC1HWC0) {
      SetInputReduceInfo(axis_reduce_map_cut, 0, 1, sup_sw);
      SetInputReduceInfo(axis_reduce_map_cut, 1, 0, sup_sw);
      SetOutputReduceInfo(axis_reduce_map_cut, 0, REDUCE_ADD, false);
    } else if (input_format == ge::FORMAT_FRACTAL_NZ) {
      SetInputReduceInfo(axis_reduce_map_cut, 0, 0, sup_sw);
      SetInputReduceInfo(axis_reduce_map_cut, 1, 1, sup_sw);
      SetOutputReduceInfo(axis_reduce_map_cut, 0, REDUCE_ADD, false);
    } else {
      FE_LOGD("Cube-matmul node[%s] does not support cut format[%s].", op_desc_ptr->GetName().c_str(),
              ge::TypeUtils::FormatToSerialString(input_format).c_str());
    }
    axis_reduce_maps.push_back(axis_reduce_map_cut);
  }
  op_calc_info.SetAxisSplitMaps(axis_split_maps);
  op_calc_info.SetAxisReduceMaps(axis_reduce_maps);
  return SUCCESS;
}

/*
 *  @ingroup fe
 *  @brief   Set the slice information for the node whose SlicePattern is Reduce in the ops info config.
 *  @param   [in|out] node | slice info
 *  @return  SUCCESS or FAILED
 */
Status OpSliceUtil::FillupReduceSliceInfo(ge::OpDescPtr op_desc_ptr, OpCalcInfo &op_calc_info, const bool& sup_sw) {
  FE_LOGD("Start setting node [%s, %s]'s slice pattern as reduce.",
          op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
  std::vector<AxisSplitMap> axis_split_maps;
  Status ret = SetReduceSliceInfo(op_desc_ptr, axis_split_maps, sup_sw);
  if (ret != SUCCESS) {
    FE_LOGD("Parsing operation slice information for node [%s] did not match; slice information was not set.", op_desc_ptr->GetName().c_str());
    return SUCCESS;
  }
  op_calc_info.SetAxisSplitMaps(axis_split_maps);
  return SUCCESS;
}

/*
 *  @ingroup fe
 *  @brief   Set the slice information for the node whose SlicePattern is resize in the ops info config.
 *  @param   [in|out] node | slice info
 *  @return  SUCCESS or FAILED
 */

Status OpSliceUtil::FillupResizeSliceInfo(ge::OpDescPtr op_desc_ptr, const OpCalcInfo &op_calc_info,
                                          const bool& sup_sw) {
  (void)op_desc_ptr;
  (void)op_calc_info;
  (void)sup_sw;
  FE_LOGI("Does not support this slice pattern yet.");
  return SUCCESS;
}

/*
 *  @ingroup fe
 *  @brief   Set the slice information for the node whose SlicePattern is scatter in the ops info config.
 *  @param   [in|out] node | slice info
 *  @return  SUCCESS or FAILED
 */
Status OpSliceUtil::FillupScatterSliceInfo(ge::OpDescPtr op_desc_ptr, const OpCalcInfo &op_calc_info,
                                           const bool& sup_sw) {
  (void)op_desc_ptr;
  (void)op_calc_info;
  (void)sup_sw;
  FE_LOGI("Does not support this slice pattern yet.");
  return SUCCESS;
}

/*
 *  @ingroup fe
 *  @brief   Set the slice information for the node whose SlicePattern is segment in the ops info config.
 *  @param   [in|out] node | slice info
 *  @return  SUCCESS or FAILED
 */
Status OpSliceUtil::FillupSegmentSliceInfo(ge::OpDescPtr op_desc_ptr, const OpCalcInfo &op_calc_info,
                                           const bool& sup_sw) {
  (void)op_desc_ptr;
  (void)op_calc_info;
  (void)sup_sw;
  FE_LOGI("Does not support this slice pattern yet.");
  return SUCCESS;
}

void OpSliceUtil::SetInputSplitInfo(AxisSplitMap& axis_split_map, const int8_t& input_index, const int8_t& input_axis,
                                    const bool& sup_sw) {
  InputSplitInfo input_split_info;
  if (!input_split_info.Initialize()) {
    REPORT_FE_ERROR("[OpSliceUtil][SetInSplitInfo] input_split_info initialize failed");
    return;
  }
  input_split_info.SetIndex(input_index);
  std::vector<int64_t> input_vec_first_tensor;
  if (sup_sw) {
    input_vec_first_tensor.push_back(input_axis);
  } else {
    for (int8_t idx = 0; idx <= input_axis; idx++) {
      input_vec_first_tensor.push_back(idx);
    }
  }
  input_split_info.SetAxis(input_vec_first_tensor);
  axis_split_map.AddInputSplitInfo(input_split_info);
}

void OpSliceUtil::SetInputReduceInfo(AxisReduceMap& axis_reduce_map, const int8_t& input_index, const int8_t& input_axis,
                                     const bool& sup_sw) {
  InputReduceInfo input_reduce_info;
  if (!input_reduce_info.Initialize()) {
    REPORT_FE_ERROR("[OpSliceUtil][SetInRduInfo] input_reduce_info initialize failed");
    return;
  }
  input_reduce_info.SetIndex(input_index);
  std::vector<int64_t> input_vec_first_tensor;
  if (sup_sw) {
    input_vec_first_tensor.push_back(input_axis);
  } else {
    for (int8_t idx = 0; idx <= input_axis; idx++) {
      input_vec_first_tensor.push_back(idx);
    }
  }
  input_reduce_info.SetAxis(input_vec_first_tensor);
  axis_reduce_map.AddInputReduceInfo(input_reduce_info);
}

void OpSliceUtil::SetOutputSplitInfo(AxisSplitMap& axis_split_map, const int8_t& output_index, const int8_t& output_axis,
                                     const bool& sup_sw) {
  OutputSplitInfo output_split_info;
  if (!output_split_info.Initialize()) {
    REPORT_FE_ERROR("[OpSliceUtil][SetOutSplitInfo] output_split_info initialize failed");
    return;
  }
  output_split_info.SetIndex(output_index);
  std::vector<int64_t> output_vec;
  if (sup_sw) {
    output_vec.push_back(output_axis);
  } else {
    for (int8_t idx = 0; idx <= output_axis; idx++) {
      output_vec.push_back(idx);
    }
  }
  output_split_info.SetAxis(output_vec);
  axis_split_map.AddOutputSplitInfo(output_split_info);
}

void OpSliceUtil::SetOutputReduceInfo(AxisReduceMap& axis_reduce_map, const int8_t& output_index,
                                      const OpReduceType& reduce_type, const bool& is_atomic) {
  OutputReduceInfo output_reduce_info;
  if (!output_reduce_info.Initialize()) {
    REPORT_FE_ERROR("[OpSliceUtil][SetOutRdcInfo] output_reduce_info initialize failed");
    return;
  }
  output_reduce_info.SetIndex(output_index);
  output_reduce_info.SetReduceType(reduce_type);
  output_reduce_info.SetIsAtomic(is_atomic);
  axis_reduce_map.AddOutputReduceInfo(output_reduce_info);
}

bool OpSliceUtil::IsInputDynamicDim(ge::OpDesc::Vistor<ge::GeTensorDescPtr> &input_desc_vec, const uint32_t &dim_index) {
  if (input_desc_vec.empty()) {
    return false;
  }
  bool is_dim_dynamic = true;
  for (ge::GeTensorDescPtr &input_desc : input_desc_vec) {
    ge::GeShape shape = input_desc->GetShape();
    if (shape.IsUnknownDimNum()) {
      continue;
    }
    if (shape.GetDimNum() > dim_index && shape.GetDim(dim_index) == SHAPE_UNKNOWN_DIM) {
      continue;
    }
    is_dim_dynamic = false;
  }
  return is_dim_dynamic;
}

/*
 *  @ingroup fe
 *  @brief   Set the slice information based on the input and output indexes/axes.
 *  @param   [in|out] input and output indexes/axes | slice info
 *  @return  SUCCESS or FAILED
 */
void OpSliceUtil::SetMultiAxisSplitMap(AxisSplitMap& axis_split_map,
                                       const int8_t& first_index, const int8_t& first_axis,
                                       const int8_t& output_index, const int8_t& output_axis, const bool& sup_sw,
                                       const int8_t second_index, const int8_t second_axis) {
  SetInputSplitInfo(axis_split_map, first_index, first_axis, sup_sw);

  if (second_index > -1 && second_axis > -1) {
    SetInputSplitInfo(axis_split_map, second_index, second_axis, sup_sw);
  }

  SetOutputSplitInfo(axis_split_map, output_index, output_axis, sup_sw);
}

Status OpSliceUtil::CheckElemwiseInputAndOutputNum(ge::OpDescPtr op_desc_ptr, const bool &has_scalar,
                                                   const size_t &dim_size, const ge::Format &op_output_format) {
  auto op_input_desc_list = op_desc_ptr->GetAllInputsDescPtr();
  auto op_output_desc_list = op_desc_ptr->GetAllOutputsDescPtr();
  if (op_output_desc_list.size() != 1 && !has_scalar) {
    FE_LOGW("Node [%s] has %zu outputs, which is not equal to one.", op_desc_ptr->GetName().c_str(), op_output_desc_list.size());
    return FAILED;
  }
  if (op_input_desc_list.empty()) {
    FE_LOGW("Node [%s] does not have any inputs.", op_desc_ptr->GetName().c_str());
    return FAILED;
  }
  if (dim_size == 0) {
    FE_LOGW("Node [%s]'s output has %zu output dimensions, cannot set split info.",
            op_desc_ptr->GetName().c_str(), dim_size);
    return FAILED;
  }
  // dim number of shape and format of all inputs and outputs should be the same.
  for (uint32_t index = 0; index < op_input_desc_list.size(); index++) {
    ge::GeShape input_shape = op_input_desc_list.at(index)->GetShape();
    if (IsScalarInputShape(input_shape) && (op_desc_ptr->GetType() == ADD)) {
      continue;
    }
    auto op_input_desc_primary_format =
            static_cast<ge::Format>(ge::GetPrimaryFormat(op_input_desc_list.at(index)->GetFormat()));
    if (op_input_desc_primary_format != op_output_format) {
      FE_LOGW("Node [%s]'s input format [%s] does not match the output format [%s]; we need to call SetElemWiseSliceInfoEx to configure it.",
              op_desc_ptr->GetName().c_str(), ge::TypeUtils::FormatToSerialString(op_input_desc_primary_format).c_str(),
              ge::TypeUtils::FormatToSerialString(op_output_format).c_str());
      return FAILED;
    }
    if (input_shape.IsUnknownDimNum()) {
      continue;
    }
    if ((input_shape.GetDimNum() != 0 || !has_scalar) && input_shape.GetDimNum() != dim_size) {
      FE_LOGW("Node [%s]'s input dimension size [%zu] is not equal to its output size [%zu]; cannot set split information.",
              op_desc_ptr->GetName().c_str(), input_shape.GetDimNum(), dim_size);
      return FAILED;
    }
  }
  FE_LOGD("Check input and output info for node[%s] successfully.", op_desc_ptr->GetName().c_str());
  return SUCCESS;
}

Status OpSliceUtil::SetElemWiseSliceInfoEx(const ge::OpDescPtr &op_desc_ptr,
                                           std::vector<AxisSplitMap>& axis_split_maps) {
  auto op_input_desc_list = op_desc_ptr->GetAllInputsDescPtr();
  auto op_output_desc_list = op_desc_ptr->GetAllOutputsDescPtr();
  int32_t intput_more_dims_idx = 0;
  int32_t input_more_dims_size = 0;
  int32_t diff = 0;
  if (SetElemWiseSliceInfoExCheckParam(op_input_desc_list, op_output_desc_list, intput_more_dims_idx,
                                       input_more_dims_size, diff) != SUCCESS) {
    FE_LOGW("SetElemWiseSliceInfoEx failed for node [%s].", op_desc_ptr->GetName().c_str());
    return FAILED;
  }
  // fill up the less dim axis with 1
  std::vector<int64_t> dim_vec;
  for (int i = 0; i < diff; i++) {
    dim_vec.emplace_back(1);
  }
  auto more_axis_dims = op_input_desc_list.at(intput_more_dims_idx)->GetShape().GetDims();
  auto less_axis_dims = op_input_desc_list.at(1 - intput_more_dims_idx)->GetShape().GetDims();
  for (auto dim : less_axis_dims) {
    dim_vec.emplace_back(dim);
  }
  if (input_more_dims_size < 1 || static_cast<int32_t>(dim_vec.size()) != input_more_dims_size) {
    FE_LOGW("Node [%s] fill up less axis with 1 wrong, input_more_dims_size:%d", op_desc_ptr->GetName().c_str(),
            input_more_dims_size);
    return FAILED;
  }
  /*
   * Compare the dim from right to left, if the dim is same, find the correspond aix idx, and set the
   * axis split info, if not, skip the aix
   */
  for (int32_t i = input_more_dims_size - 1; i >= 0; i--) {
    auto less_dim = dim_vec.at(i);
    auto more_dim = more_axis_dims.at(i);
    AxisSplitMap axis_split_map;
    if (!axis_split_map.Initialize()) {
      REPORT_FE_ERROR("[OpSliceUtil][SetElemWiseSliceInfoEx] axis_split_map initialize failed");
      return FAILED;
    }
    bool condition = (less_dim == more_dim && more_dim > 1);
    if (condition) {
      SetInputSplitInfo(axis_split_map, intput_more_dims_idx, i, true);
      SetInputSplitInfo(axis_split_map, 1 - intput_more_dims_idx, i - diff, true);
      SetOutputSplitInfo(axis_split_map, 0, i, true);
    } else {
      continue;
    }
    condition = (!axis_split_map.GetInputSplitInfos().empty() && !axis_split_map.GetOutputSplitInfos().empty());
    if (condition) {
      axis_split_maps.emplace_back(axis_split_map);
    }
  }
  FE_LOGD("SetElemWiseSliceInfoEx for node[%s] successfully.", op_desc_ptr->GetName().c_str());
  return SUCCESS;
}
/*
 *  @ingroup fe
 *  @brief   Set the slice information corresponding to the elemwise based on the node information.
 *  @param   [in|out] node & has scalar or not | slice info
 *  @return  SUCCESS or FAILED
 */
Status OpSliceUtil::SetElemWiseSliceInfo(ge::OpDescPtr op_desc_ptr, std::vector<AxisSplitMap>& axis_split_maps,
                                         bool has_scalar, const bool& sup_sw, bool is_filter_dynamic) {
  auto op_input_desc_list = op_desc_ptr->GetAllInputsDescPtr();
  auto op_output_desc_list = op_desc_ptr->GetAllOutputsDescPtr();
  if (op_output_desc_list.empty()) {
    FE_LOGW("%s's output list is empty", op_desc_ptr->GetName().c_str());
    return FAILED;
  }
  auto output0 = op_output_desc_list.at(0);
  FE_CHECK(output0 == nullptr, FE_LOGW("%s's output 0 is nullptr.", op_desc_ptr->GetName().c_str()),
           return FAILED;);

  size_t dim_size = output0->GetShape().GetDimNum();
  ge::Format op_output_format = static_cast<ge::Format>(ge::GetPrimaryFormat(output0->GetFormat()));
  // check input and output info for node
  Status ret = CheckElemwiseInputAndOutputNum(op_desc_ptr, has_scalar, dim_size, op_output_format);
  if (ret != SUCCESS) {
    FE_LOGW("Failed to check node [%s]'s input or output info, unable to set slice info.",
            op_desc_ptr->GetName().c_str());
    return FAILED;
  }
  // do not set slice info for 5hd's last one dimension
  size_t split_size = dim_size;
  bool isFiveDimension = op_output_format == ge::FORMAT_NC1HWC0 || op_output_format == ge::FORMAT_NHWC1C0 ||
                         op_output_format == ge::FORMAT_C1HWNC0 || op_output_format == ge::FORMAT_NC1HWC0_C04;
  if (isFiveDimension) {
    if (split_size > 1) {
      split_size -= 1;
    } else {
      FE_LOGW("Node [%s]'s split size (%zu) is equal to or less than one; cannot set split info.",
              op_desc_ptr->GetName().c_str(), split_size);
      return FAILED;
    }
  }
  // do not set slice info for fragz's last two dimension
  bool isFragzDimension = op_output_format == ge::FORMAT_FRACTAL_NZ || op_output_format == ge::FORMAT_FRACTAL_Z ||
                          op_output_format == ge::FORMAT_FRACTAL_Z_C04;
  if (isFragzDimension) {
    if (split_size > kSplitSize) {
      split_size -= kSplitSize;
    } else {
      FE_LOGW("Node [%s]'s split size of %zu is equal to or less than two, unable to set split info.",
              op_desc_ptr->GetName().c_str(), split_size);
      return FAILED;
    }
  }
  if (split_size == 0) {
    FE_LOGW("Node [%s]'s split size %zu is equal to zero, cannot set split info.", op_desc_ptr->GetName().c_str(),
            split_size);
    return FAILED;
  }
  // set slice info for every input and output of elemwise node, each input and output of the elem node can be sliced
  // except 5hd's last one dimension and fragz's last two dimension
  for (uint32_t axis = 0; axis < split_size; axis++) {
    AxisSplitMap axis_split_map;
    if (!axis_split_map.Initialize()) {
      REPORT_FE_ERROR("[OpSliceUtil][SetElemWiseSliceInfo] axis_split_map initialize failed");
      return FAILED;
    }
    bool dynamic_flag = is_filter_dynamic && IsInputDynamicDim(op_input_desc_list, axis);
    if (dynamic_flag) {
      FE_LOGD("The dim index[%u] of all input is dynamic dim. Slice info will be generated for this axes.", axis);
      continue;
    }
    SetInputSplitInfo(op_desc_ptr, axis_split_map, axis, has_scalar, sup_sw);
    SetOutputSplitInfo(op_desc_ptr, axis_split_map, axis, has_scalar, sup_sw);
    bool condition = (!axis_split_map.GetInputSplitInfos().empty() && !axis_split_map.GetOutputSplitInfos().empty());
    if (condition) {
      axis_split_maps.push_back(axis_split_map);
    }
  }
  return SUCCESS;
}

void OpSliceUtil::SetInputSplitInfo(const ge::OpDescPtr &op_desc_ptr, AxisSplitMap &axis_split_map, const int8_t &axis,
                                    bool has_scalar, const bool &sup_sw) {
  auto op_input_desc_list = op_desc_ptr->GetAllInputsDescPtr();
  for (uint32_t index = 0; index < op_input_desc_list.size(); index++) {
    auto dims = op_input_desc_list.at(index)->GetShape().GetDims();
    if (dims.empty()) {
      FE_LOGD("Node [%s]'s %u input dimension size is zero; cannot set split information.", op_desc_ptr->GetName().c_str(), index);
      continue;
    }

    int64_t dim_size = static_cast<int64_t>(dims.size()) ;
    bool condition = (has_scalar && axis >= 0 && axis < dim_size && dims.at(axis) == 1);
    if (condition) {
      continue;
    }
    SetInputSplitInfo(axis_split_map, index, axis, sup_sw);
  }
}

void OpSliceUtil::SetOutputSplitInfo(const ge::OpDescPtr &op_desc_ptr, AxisSplitMap &axis_split_map,
                                     const int8_t &axis,
                                     bool has_scalar, const bool &sup_sw) {
  auto op_output_desc_list = op_desc_ptr->GetAllOutputsDescPtr();
  for (uint32_t index = 0; index < op_output_desc_list.size(); index++) {
    auto dims = op_output_desc_list.at(index)->GetShape().GetDims();
    if (dims.empty()) {
      FE_LOGD("Node [%s]'s %u output dim size is zero, cannot set split info.",
              op_desc_ptr->GetName().c_str(), index);
      continue;
    }
    int64_t dim_size = static_cast<int64_t>(dims.size()) ;
    bool condition = (has_scalar && axis >= 0 && axis < dim_size && dims.at(axis) == 1);
    if (condition) {
      continue;
    }
    SetOutputSplitInfo(axis_split_map, index, axis, sup_sw);
  }
}

/*
 *  @ingroup fe
 *  @brief   Set the slice information corresponding to the slidingwindow based on the node information.
 *  @param   [in|out] node | slice info
 *  @return  SUCCESS or FAILED
 */
Status OpSliceUtil::SetSlidingWindowSliceInfo(ge::OpDescPtr op_desc_ptr, std::vector<AxisSplitMap>& axis_split_maps,
                                              const bool& sup_sw) {
  (void)op_desc_ptr;
  AxisSplitMap axis_split_map_cut_n;
  if (!axis_split_map_cut_n.Initialize()) {
    REPORT_FE_ERROR("[OpSliceUtil][SetSlidWinSliceInfo] axis_split_map_cut_n initialize failed");
    return FAILED;
  }
  SetMultiAxisSplitMap(axis_split_map_cut_n, 0, 0, 0, 0, sup_sw);
  axis_split_maps.push_back(axis_split_map_cut_n);

  AxisSplitMap axis_split_map_cut_h;
  if (!axis_split_map_cut_h.Initialize()) {
    REPORT_FE_ERROR("[OpSliceUtil][SetSlidWinSliceInfo] axis_split_map_cut_h initialize failed");
    return FAILED;
  }
  SetMultiAxisSplitMap(axis_split_map_cut_h, 0, kAxisHIndex, 0, kAxisHIndex, sup_sw);
  axis_split_maps.push_back(axis_split_map_cut_h);

  AxisSplitMap axis_split_map_cut_w;
  if (!axis_split_map_cut_w.Initialize()) {
    REPORT_FE_ERROR("[OpSliceUtil][SetSlidWinSliceInfo] axis_split_map_cut_w initialize failed");
    return FAILED;
  }
  SetMultiAxisSplitMap(axis_split_map_cut_w, 0, kAxisWIndex, 0, kAxisWIndex, sup_sw);
  axis_split_maps.push_back(axis_split_map_cut_w);

  AxisSplitMap axis_split_map_cut_cout;
  if (!axis_split_map_cut_cout.Initialize()) {
    REPORT_FE_ERROR("[OpSliceUtil][SetSlidWinSliceInfo] axis_split_map_cut_count initialization failed");
    return FAILED;
  }
  SetMultiAxisSplitMap(axis_split_map_cut_cout, 1, 1, 0, 1, sup_sw);
  axis_split_maps.push_back(axis_split_map_cut_cout);

  return SUCCESS;
}
/*
 *  @ingroup fe
 *  @brief   Set the slice information corresponding to the slidingwindowdeconv based on the node information.
 *  @param   [in|out] node | slice info
 *  @return  SUCCESS or FAILED
 */
Status OpSliceUtil::SetSlidingWindowDeconvSliceInfo(ge::OpDescPtr op_desc_ptr,
                                                    std::vector<AxisSplitMap>& axis_split_maps, const bool& sup_sw) {
  (void)op_desc_ptr;
  AxisSplitMap axis_split_map_cut_n;
  if (!axis_split_map_cut_n.Initialize()) {
    REPORT_FE_ERROR("[OpSliceUtil][SetSlidWinDeconvSliceInfo] axis_split_map_cut_n initialize failed");
    return FAILED;
  }
  SetMultiAxisSplitMap(axis_split_map_cut_n, 0, 0, 0, 0, sup_sw);
  axis_split_maps.push_back(axis_split_map_cut_n);

  AxisSplitMap axis_split_map_cut_h;
  if (!axis_split_map_cut_h.Initialize()) {
    REPORT_FE_ERROR("[OpSliceUtil][SetSlidWinDeconvSliceInfo] axis_split_map_cut_h initialize failed");
    return FAILED;
  }
  SetMultiAxisSplitMap(axis_split_map_cut_h, 0, kAxisHIndex, 0, kAxisHIndex, sup_sw);
  axis_split_maps.push_back(axis_split_map_cut_h);

  AxisSplitMap axis_split_map_cut_w;
  if (!axis_split_map_cut_w.Initialize()) {
    REPORT_FE_ERROR("[OpSliceUtil][SetSlidWinDeconvSliceInfo] Failed to initialize axis_split_map_cut_w");
    return FAILED;
  }
  SetMultiAxisSplitMap(axis_split_map_cut_w, 0, kAxisWIndex, 0, kAxisWIndex, sup_sw);
  axis_split_maps.push_back(axis_split_map_cut_w);

  AxisSplitMap axis_split_map_cut_cout;
  if (!axis_split_map_cut_cout.Initialize()) {
    REPORT_FE_ERROR("[OpSliceUtil][SetSlidWinDeconvSliceInfo] axis_split_map_cut_count initialization failed");
    return FAILED;
  }
  SetMultiAxisSplitMap(axis_split_map_cut_cout, 1, 1, 0, 1, sup_sw);
  axis_split_maps.push_back(axis_split_map_cut_cout);

  FE_LOGI("Not support to set min_tbe_l1_space info yet.");
  return SUCCESS;
}
/*
 *  @ingroup fe
 *  @brief   Set the slice information corresponding to the cube matmul based on the node information.
 *  @param   [in|out] node | slice info
 *  @return  SUCCESS or FAILED
 */
Status OpSliceUtil::SetCubeMatmulSliceInfo(ge::OpDescPtr op_desc_ptr, std::vector<AxisSplitMap>& axis_split_maps,
                                           const bool& sup_sw) {
  FE_CHECK_NOTNULL(op_desc_ptr->GetInputDescPtr(0));
  auto input_format = static_cast<ge::Format>(ge::GetPrimaryFormat(op_desc_ptr->GetInputDescPtr(0)->GetFormat()));
  if (input_format == ge::FORMAT_NC1HWC0) {
    AxisSplitMap axis_split_map_cut_m;
    if (!axis_split_map_cut_m.Initialize()) {
      REPORT_FE_ERROR("[OpSliceUtil][SetCubeMulSliceInfo] failed to initialize axis_split_map_cut_m");
      return FAILED;
    }
    SetMultiAxisSplitMap(axis_split_map_cut_m, 0, 0, 0, 0, sup_sw, -1, -1);
    AxisSplitMap axis_split_map_cut_n;
    if (!axis_split_map_cut_n.Initialize()) {
      REPORT_FE_ERROR("[OpSliceUtil][SetCubeMulSliceInfo] failed to initialize axis_split_map_cut_n");
      return FAILED;
    }
    SetMultiAxisSplitMap(axis_split_map_cut_n, 1, 1, 0, 0, sup_sw, -1, -1);
    axis_split_maps.push_back(axis_split_map_cut_m);
    axis_split_maps.push_back(axis_split_map_cut_n);
  } else if (input_format == ge::FORMAT_FRACTAL_NZ) {
    AxisSplitMap axis_split_map_cut_m;
    if (!axis_split_map_cut_m.Initialize()) {
      REPORT_FE_ERROR("[OpSliceUtil][SetCubeMulSliceInfo] failed to initialize axis_split_map_cut_m");
      return FAILED;
    }
    SetMultiAxisSplitMap(axis_split_map_cut_m, 0, 1, 0, 0, sup_sw, -1, -1);
    AxisSplitMap axis_split_map_cut_n;
    if (!axis_split_map_cut_n.Initialize()) {
      REPORT_FE_ERROR("[OpSliceUtil][SetCubeMulSliceInfo] failed to initialize axis_split_map_cut_n");
      return FAILED;
    }
    SetMultiAxisSplitMap(axis_split_map_cut_n, 1, 0, 0, 0, sup_sw, -1, -1);
    axis_split_maps.push_back(axis_split_map_cut_m);
    axis_split_maps.push_back(axis_split_map_cut_n);
  } else {
    FE_LOGD("Cube-matmul node[%s] does not support cut format[%s]", op_desc_ptr->GetName().c_str(),
            ge::TypeUtils::FormatToSerialString(input_format).c_str());
  }
  return SUCCESS;
}
/*
 *  @ingroup fe
 *  @brief   Set the slice information corresponding to the reduce based on the node information.
 *  @param   [in|out] node | slice info
 *  @return  SUCCESS or FAILED
 */
Status OpSliceUtil::SetReduceSliceInfo(ge::OpDescPtr op_desc_ptr, std::vector<AxisSplitMap>& axis_split_maps,
                                       const bool& sup_sw) {
  std::vector<int64_t> axes_vec;
  (void)ge::AttrUtils::GetListInt(op_desc_ptr, "axes", axes_vec);
  FE_LOGD("Axes attribute is [%s].", StringUtils::IntegerVecToString(axes_vec).c_str());
  auto op_input_desc_list = op_desc_ptr->GetAllInputsDescPtr();
  auto op_output_desc_list = op_desc_ptr->GetAllOutputsDescPtr();
  // check input and output info for reduce node
  if (op_input_desc_list.size() < 1) {
    FE_LOGW("Node [%s] has %zu input, less than one.", op_desc_ptr->GetName().c_str(), op_input_desc_list.size());
    return FAILED;
  }
  if (op_output_desc_list.size() != 1) {
    FE_LOGW("Node [%s] has %zu outputs, which is not equal to one.", op_desc_ptr->GetName().c_str(), op_output_desc_list.size());
    return FAILED;
  }
  auto op_output_primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(op_output_desc_list.at(0)->GetFormat()));
  size_t dim_size = op_input_desc_list.at(0)->GetShape().GetDims().size();
  size_t out_dim_size = op_output_desc_list.at(0)->GetShape().GetDims().size();
  if (dim_size == 0) {
    FE_LOGW("Node [%s]'s input has %zu input dimension size; cannot set split information.", op_desc_ptr->GetName().c_str(),
            dim_size);
    return FAILED;
  }
  ModifyAxex(axes_vec, static_cast<int64_t>(dim_size));
  if (static_cast<ge::Format>(ge::GetPrimaryFormat(op_input_desc_list.at(0)->GetFormat())) == ge::FORMAT_NC1HWC0) {
    axes_vec.emplace_back(kAxesValue);
  }
  // do not set slice info for reduce axes
  for (uint32_t index = 0; index < op_input_desc_list.size(); index++) {
    auto op_input_primary_format =
            static_cast<ge::Format>(ge::GetPrimaryFormat(op_input_desc_list.at(index)->GetFormat()));
    if (op_input_primary_format != op_output_primary_format) {
      FE_LOGW("Node [%s]'s input format [%s] does not match the output format [%s], unable to set split info.",
              op_desc_ptr->GetName().c_str(), ge::TypeUtils::FormatToSerialString(op_input_primary_format).c_str(),
              ge::TypeUtils::FormatToSerialString(op_output_primary_format).c_str());
      return FAILED;
    }
  }
  size_t output_axis = 0;
  bool keep_dims = (dim_size == out_dim_size);
  for (size_t dim_num = 0; dim_num < dim_size; dim_num++) {
    AxisSplitMap axis_split_map;
    if (!axis_split_map.Initialize()) {
      REPORT_FE_ERROR("[OpSliceUtil][SetRdcSliceInfo] axis_split_map initialization failed");
      return FAILED;
    }
    if (std::find(axes_vec.begin(), axes_vec.end(), dim_num) != axes_vec.end()) {
      continue;
    }
    if (IsInputDynamicDim(op_input_desc_list, dim_num)) {
      FE_LOGD("The dim index [%zu] of all inputs is a dynamic dimension. Slice information will be generated.", dim_num);
      continue;
    }
    for (size_t index = 0; index < op_input_desc_list.size(); index++) {
      SetInputSplitInfo(axis_split_map, index, dim_num, sup_sw);
    }
    for (size_t index = 0; index < op_output_desc_list.size(); index++) {
      if (keep_dims) {
        SetOutputSplitInfo(axis_split_map, index, dim_num, sup_sw);
      } else {
        SetOutputSplitInfo(axis_split_map, index, output_axis, sup_sw);
      }
    }
    output_axis++;
    axis_split_maps.push_back(axis_split_map);
  }
  return SUCCESS;
}

void OpSliceUtil::ModifyAxex(std::vector<int64_t> &axes_vec, const int64_t &dim_size) {
  for (int64_t &axes : axes_vec) {
    axes = axes % dim_size;
    if (axes < 0) {
      axes += dim_size;
    }
  }
  return;
}

Status OpSliceUtil::SetOpCutInfoOnTensor(const ge::OpDescPtr &op_desc, const OpCalcInfo &op_calc_info) {
  FE_CHECK_NOTNULL(op_desc);
  FE_LOGD("Set tensor cut info for node[%s].", op_desc->GetName().c_str());
  auto axis_split_maps = op_calc_info.GetAxisSplitMaps();
  for (auto &one_split_info : axis_split_maps) {
    size_t input_size = op_desc->GetAllInputsSize();
    FE_LOGD("Input_size is %zu.", input_size);

    vector<vector<int64_t>> all_inputs_strategies;
    InitializeStrategy(op_desc, input_size, true, all_inputs_strategies);

    for (auto &input_split_info : one_split_info->GetInputSplitInfos()) {
      auto tensor_index = input_split_info->GetIndex();
      FE_LOGD("Get input split info %zu.", tensor_index);
      auto tensor_desc = op_desc->MutableInputDesc(tensor_index);
      if (tensor_desc == nullptr) {
        continue;
      }
      auto axis = input_split_info->GetAxis();
      GenOneStrategy(tensor_index, axis, tensor_desc, all_inputs_strategies);
    }
    SetCutInfoAttr(op_desc, input_size, true, all_inputs_strategies);

    /* output */
    size_t output_size = op_desc->GetAllOutputsDescSize();
    vector<vector<int64_t>> all_outputs_strategies;
    InitializeStrategy(op_desc, output_size, false, all_outputs_strategies);

    for (auto &output_split_info : one_split_info->GetOutputSplitInfos()) {
      auto tensor_index = output_split_info->GetIndex();
      auto tensor_desc = op_desc->MutableOutputDesc(tensor_index);
      if (tensor_desc == nullptr) {
        continue;
      }
      auto axis = output_split_info->GetAxis();
      GenOneStrategy(tensor_index, axis, tensor_desc, all_outputs_strategies);
    }
    SetCutInfoAttr(op_desc, output_size, false, all_outputs_strategies);
  }
  return SUCCESS;
}

bool OpSliceUtil::IsSgtInfoConsistant(const ge::NodePtr &consumer, const ge::NodePtr &producer) {
  ffts::ThreadSliceMapPtr produce_slice_info_ptr = nullptr;
  ffts::ThreadSliceMapPtr consumer_slice_info_ptr = nullptr;
  produce_slice_info_ptr = producer->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, produce_slice_info_ptr);
  consumer_slice_info_ptr = consumer->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, consumer_slice_info_ptr);
  if (produce_slice_info_ptr == nullptr && consumer_slice_info_ptr == nullptr) {
    return true;
  }
  if (produce_slice_info_ptr != nullptr && consumer_slice_info_ptr != nullptr) {
    if (produce_slice_info_ptr->slice_instance_num == consumer_slice_info_ptr->slice_instance_num) {
      return true;
    }
  }
  return false;
}
}  // namespace fe
