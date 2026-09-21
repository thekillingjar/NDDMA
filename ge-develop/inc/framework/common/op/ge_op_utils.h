/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_COMMON_OP_GE_OP_UTILS_H_
#define INC_FRAMEWORK_COMMON_OP_GE_OP_UTILS_H_

#include <memory>
#include <vector>

#include "graph/debug/ge_attr_define.h"
#include "framework/common/util.h"
#include "graph/attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "proto/insert_op.pb.h"

namespace ge {

// Add Sub Mul
extern GE_FUNC_VISIBILITY const uint32_t ADD_INPUT_NUM;
extern GE_FUNC_VISIBILITY const uint32_t MUL_INPUT_NUM;

// Permute
extern GE_FUNC_VISIBILITY const int32_t PERMUTE_ORDER_NUM;

// Ssd PriroBox
extern GE_FUNC_VISIBILITY const float64_t SSD_PRIORBOX_ASPECT_RATIO_VALUE;

extern GE_FUNC_VISIBILITY const uint32_t STRIDEDSLICE_INPUT_NUM;

// Switch
extern GE_FUNC_VISIBILITY const uint32_t SWITCH_INPUT_NUM;
extern GE_FUNC_VISIBILITY const uint32_t SWITCH_OUTPUT_NUM;
extern GE_FUNC_VISIBILITY const uint32_t SWITCH_FALSE_OUTPUT;
extern GE_FUNC_VISIBILITY const uint32_t SWITCH_TRUE_OUTPUT;
extern GE_FUNC_VISIBILITY const uint32_t SWITCH_DATA_INPUT;
extern GE_FUNC_VISIBILITY const uint32_t SWITCH_PRED_INPUT;

// Merge
extern GE_FUNC_VISIBILITY const int32_t MERGE_DATA_OUTPUT;
extern GE_FUNC_VISIBILITY const int32_t MERGE_INDEX_OUTPUT;

// FunctionOp
extern GE_FUNC_VISIBILITY const uint32_t IF_COND_INPUT;
extern GE_FUNC_VISIBILITY const uint32_t FOR_START_INPUT;
extern GE_FUNC_VISIBILITY const uint32_t FOR_LIMIT_INPUT;
extern GE_FUNC_VISIBILITY const uint32_t FOR_DELTA_INPUT;
extern GE_FUNC_VISIBILITY const uint32_t FOR_DATA_INPUT;

extern GE_FUNC_VISIBILITY const int32_t NORMAL_TENSOR_SIZE;
/*lint -e148*/
class GE_FUNC_VISIBILITY OpUtils {
 public:
  ///
  /// @brief Extract AIPP parameters from AttrDefMap and splice them
  /// @param [in] aipp_attr attr of operator
  /// @param [out] aipp_params aipp parameters
  /// @return enum of tagCCAippInputFormat
  ///

  static Status ConvertAippParams(const GeAttrValue::NamedAttrs &aipp_attr,
                                  domi::AippOpParams &aipp_params);
  template <typename T>
  static void SliceData(const std::vector<char_t *> &input, const int64_t chunk_size, std::vector<char_t *> &output,
                        const int64_t begin, const int64_t out_dim, const int64_t stride);
  template <typename T>
  static Status SetDataByDataType(const size_t out_size, const std::vector<char_t *> &chunk_input,
                                  const std::vector<char_t *> &chunk_output, GeTensor *const output);
  template <typename T>
  static Status SetOutputSliceDataByDataType(void *const data, const int64_t data_size,
                                             const std::vector<int64_t> &input_dims,
                                             const std::vector<int64_t> &begin,
                                             const std::vector<int64_t> &output_dims,
                                             ge::GeTensor *const output, const std::vector<int64_t> &stride);
  static Status SetOutputSliceData(void *const data, const int64_t data_size, const int32_t data_type,
                                   const std::vector<int64_t> &input_dims, const std::vector<int64_t> &begin,
                                   const std::vector<int64_t> &output_dims, GeTensor *const output,
                                   const std::vector<int64_t> &stride);
  static Status GetShapeDataFromConstTensor(const ConstGeTensorPtr &tensor, const DataType type,
                                            std::vector<int64_t> &dims);
  /// Return true if node is hcom node and does not support addr refresh
  /// which means its io can not support zero copy
  /// \param node
  /// \return
  static bool IsHcomNodeNotSupportAddrRefresh(const OpDescPtr &op_desc);
  ///
  /// @brief get memory size for constant and const with DT_STRING data type
  /// @param [in] op_desc
  /// @param [out] mem_size
  /// @return SUCCESS
  ///
  static Status GetConstantStrMemSize(const OpDescPtr &op_desc, int64_t &mem_size);
};
/*lint +e148*/
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_OP_GE_OP_UTILS_H_
