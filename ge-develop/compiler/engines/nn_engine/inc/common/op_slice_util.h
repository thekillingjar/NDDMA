/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_COMMON_UTILS_AI_CORE_OP_SLICE_UTIL_H
#define INC_COMMON_UTILS_AI_CORE_OP_SLICE_UTIL_H

#include <vector>
#include <string>
#include "common/aicore_util_types.h"
#include "common/aicore_util_constants.h"
#include "common/fe_log.h"
#include "graph/op_desc.h"
#include "graph/ge_tensor.h"
#include "register/graph_optimizer/fusion_common/op_slice_info.h"

namespace fe {
using FillupSliceInfo = std::function<Status(ge::OpDescPtr, OpCalcInfo&, const bool&)>;
using FillupSliceInfoPtr = std::shared_ptr<FillupSliceInfo>;

class OpSliceUtil {
 public:
  explicit OpSliceUtil();
  virtual ~OpSliceUtil();
  static Status SetOpSliceInfo(const ge::NodePtr &node, const SlicePattern &slice_pattern,
                               const bool &support_stride_write);

  static Status SetOpCutInfoOnTensor(const ge::OpDescPtr &op_desc, const OpCalcInfo &op_calc_info);

  static bool IsSgtInfoConsistant(const ge::NodePtr &consumer, const ge::NodePtr &producer);
 private:
  /*
   *  @ingroup fe
   *  @brief   Set the slice information for the node whose SlicePattern is Elemwise in the ops info config.
   *  @param   [in|out] node | slice info
   *  @return  SUCCESS or FAILED
   */
  static Status FillupElemwiseSliceInfo(ge::OpDescPtr op_desc_ptr, OpCalcInfo &op_calc_info, const bool& sup_sw);

  /*
   *  @ingroup fe
   *  @brief   Set the slice information for the node whose SlicePattern is ElemwiseBroadcast in the ops info config.
   *  @param   [in|out] node | slice info
   *  @return  SUCCESS or FAILED
   */
  static Status FillupElemwiseBroadcastSliceInfo(ge::OpDescPtr op_desc_ptr, OpCalcInfo &op_calc_info,
                                                 const bool& sup_sw);

  /*
   *  @ingroup fe
   *  @brief   Set the slice information for the node whose SlicePattern is Broadcast in the ops info config.
   *  @param   [in|out] node | slice info
   *  @return  SUCCESS or FAILED
   */
  static Status FillupBroadcastSliceInfo(ge::OpDescPtr op_desc_ptr, OpCalcInfo &op_calc_info, const bool& sup_sw);

  /*
   *  @ingroup fe
   *  @brief   Set the slice information for the node whose SlicePattern is SlidingWindow in the ops info config.
   *  @param   [in|out] node | slice info
   *  @return  SUCCESS or FAILED
   */
  static Status FillupSlidingWindowSliceInfo(ge::OpDescPtr op_desc_ptr, OpCalcInfo &op_calc_info, const bool& sup_sw);

  /*
   *  @ingroup fe
   *  @brief   Set the slice information for the node whose SlicePattern is SlidingWindowDeconv in the ops info config.
   *  @param   [in|out] node | slice info
   *  @return  SUCCESS or FAILED
   */
  static Status FillupSlidingWindowDeconvSliceInfo(ge::OpDescPtr op_desc_ptr, OpCalcInfo &op_calc_info,
                                                   const bool& sup_sw);

  /*
   *  @ingroup fe
   *  @brief   Set the slice information for the node whose SlicePattern is CubeMatmul in the ops info config.
   *  @param   [in|out] node | slice info
   *  @return  SUCCESS or FAILED
   */
  static Status FillupCubeMatmulSliceInfo(ge::OpDescPtr op_desc_ptr, OpCalcInfo &op_calc_info, const bool& sup_sw);

  /*
   *  @ingroup fe
   *  @brief   Set the slice information for the node whose SlicePattern is Reduce in the ops info config.
   *  @param   [in|out] node | slice info
   *  @return  SUCCESS or FAILED
   */
  static Status FillupReduceSliceInfo(ge::OpDescPtr op_desc_ptr, OpCalcInfo &op_calc_info, const bool& sup_sw);

  /*
   *  @ingroup fe
   *  @brief   Set the slice information for the node whose SlicePattern is Resize in the ops info config.
   *  @param   [in|out] node | slice info
   *  @return  SUCCESS or FAILED
   */
  static Status FillupResizeSliceInfo(ge::OpDescPtr op_desc_ptr, const OpCalcInfo &op_calc_info, const bool& sup_sw);

  /*
   *  @ingroup fe
   *  @brief   Set the slice information for the node whose SlicePattern is Scatter in the ops info config.
   *  @param   [in|out] node | slice info
   *  @return  SUCCESS or FAILED
   */
  static Status FillupScatterSliceInfo(ge::OpDescPtr op_desc_ptr, const OpCalcInfo &op_calc_info, const bool& sup_sw);

  /*
   *  @ingroup fe
   *  @brief   Set the slice information for the node whose SlicePattern is Segment in the ops info config.
   *  @param   [in|out] node | slice info
   *  @return  SUCCESS or FAILED
   */
  static Status FillupSegmentSliceInfo(ge::OpDescPtr op_desc_ptr, const OpCalcInfo &op_calc_info, const bool& sup_sw);

  static Status CheckElemwiseInputAndOutputNum(ge::OpDescPtr op_desc_ptr, const bool &has_scalar,
                                               const size_t &dim_size, const ge::Format &op_output_format);

  /*
   *  @ingroup fe
   *  @brief   Set the slice information corresponding to the elemwise based on the node information.
   *  @param   [in|out] op_desc & has scalar & is filter dynamic shape | slice info
   *  @return  SUCCESS or FAILED
   */
  static Status SetElemWiseSliceInfo(ge::OpDescPtr op_desc_ptr, std::vector<AxisSplitMap> &axis_split_maps,
                                     bool has_scalar, const bool& sup_sw, bool is_filter_dynamic = false);

  /*
   * for elemwisebroadcast op, if the inputs dims is not equal, we need do this operation
   */
  static Status SetElemWiseSliceInfoEx(const ge::OpDescPtr &op_desc_ptr, std::vector<AxisSplitMap> &axis_split_maps);

  static void SetInputSplitInfo(const ge::OpDescPtr &op_desc_ptr, AxisSplitMap &axis_split_map, const int8_t &axis,
                                bool has_scalar, const bool &sup_sw);

  static void SetOutputSplitInfo(const ge::OpDescPtr &op_desc_ptr, AxisSplitMap &axis_split_map, const int8_t &axis,
                                 bool has_scalar, const bool &sup_sw);
  /*
   *  @ingroup fe
   *  @brief   Set the slice information corresponding to the slidingwindow based on the node information.
   *  @param   [in|out] node & has scalar or not | slice info
   *  @return  SUCCESS or FAILED
   */
  static Status SetSlidingWindowSliceInfo(ge::OpDescPtr op_desc_ptr, std::vector<AxisSplitMap> &axis_split_maps,
                                          const bool& sup_sw);

  /*
   *  @ingroup fe
   *  @brief   Set the slice information corresponding to the slidingwindowdeconv based on the node information.
   *  @param   [in|out] node & has scalar or not | slice info
   *  @return  SUCCESS or FAILED
   */
  static Status SetSlidingWindowDeconvSliceInfo(ge::OpDescPtr op_desc_ptr, std::vector<AxisSplitMap> &axis_split_maps,
                                                const bool& sup_sw);

  /*
   *  @ingroup fe
   *  @brief   Set the slice information corresponding to the cube matmul based on the node information.
   *  @param   [in|out] node & has scalar or not | slice info
   *  @return  SUCCESS or FAILED
   */
  static Status SetCubeMatmulSliceInfo(ge::OpDescPtr op_desc_ptr, std::vector<AxisSplitMap> &axis_split_maps,
                                       const bool& sup_sw);

  /*
   *  @ingroup fe
   *  @brief   Set the slice information corresponding to the reduce based on the node information.
   *  @param   [in|out] node & has scalar or not | slice info
   *  @return  SUCCESS or FAILED
   */
  static Status SetReduceSliceInfo(ge::OpDescPtr op_desc_ptr, std::vector<AxisSplitMap> &axis_split_maps,
                                   const bool& sup_sw);
  static void ModifyAxex(std::vector<int64_t> &axes_vec, const int64_t &dim_size);
  /*
   *  @ingroup fe
   *  @brief   Set the slice information corresponding to the reduce based on the node information.
   *  @param   [in|out] node & has scalar or not | slice info
   *  @return  SUCCESS or FAILED
   */
  static void SetInputSplitInfo(AxisSplitMap &axis_split_map, const int8_t &input_index, const int8_t &input_axis,
                                const bool& sup_sw);

  /*
   *  @ingroup fe
   *  @brief   Set the reduce slice information to the input.
   *  @param   [in|out] node & has scalar or not | slice info
   *  @return  SUCCESS or FAILED
   */
  static void SetInputReduceInfo(AxisReduceMap &axis_reduce_map, const int8_t &input_index, const int8_t &input_axis,
                                 const bool& sup_sw);

  /*
   *  @ingroup fe
   *  @brief   Set the split slice information to the output.
   *  @param   [in|out] node & has scalar or not | slice info
   *  @return  SUCCESS or FAILED
   */
  static void SetOutputSplitInfo(AxisSplitMap &axis_split_map, const int8_t &output_index, const int8_t &output_axis,
                                 const bool& sup_sw);

  /*
   *  @ingroup fe
   *  @brief   Set the reduce slice information to the output.
   *  @param   [in|out] node & has scalar or not | slice info
   *  @return  SUCCESS or FAILED
   */
  static void SetOutputReduceInfo(AxisReduceMap &axis_reduce_map, const int8_t &output_index,
                                  const OpReduceType &reduce_type, const bool &is_atomic);

  /*
   *  @ingroup fe
   *  @brief   Set the slice information based on the input and output indexes/axes.
   *  @param   [in|out] input and output indexes/axes | slice info
   *  @return  SUCCESS or FAILED
   */
  static void SetMultiAxisSplitMap(AxisSplitMap &axis_split_map, const int8_t &first_index, const int8_t &first_axis,
                                   const int8_t &output_index, const int8_t &output_axis, const bool& sup_sw,
                                   const int8_t second_index = -1, const int8_t second_axis = -1);

  static bool IsInputDynamicDim(ge::OpDesc::Vistor<ge::GeTensorDescPtr> &input_desc_vec, const uint32_t &dim_index);

  static const std::map<SlicePattern, FillupSliceInfoPtr> split_info_map_;
};
}  // namespace fe
#endif  // INC_COMMON_UTILS_AI_CORE_OP_SLICE_UTIL_H
