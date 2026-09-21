/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_FOLDING_KERNEL_KERNEL_UTILS_H_
#define GE_GRAPH_PASSES_FOLDING_KERNEL_KERNEL_UTILS_H_

#include <memory>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/compute_graph.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/math/math_util.h"

namespace ge {
class KernelUtils {
 public:
  KernelUtils() = delete;
  ~KernelUtils() = delete;
  static Status CheckDimensionNodeInfo(const NodePtr &node_ptr);
  static bool CheckFormatSupported(const NodePtr &node_ptr);
  static bool CheckSizeForTransOp(const ConstGeTensorPtr &const_weight_ptr, const OpDescPtr &op_desc_ptr);
  static bool IsUnknownShape(const GeShape &shape);

  /// Construct a TensorDesc and put the data in it, it's shape is a list.
  /// If the data length is 1, it's shape is []
  static Status ConstructTensorDescWithData(const GeTensorDesc &out_desc, const std::vector<int64_t> &data,
                                            std::vector<GeTensorPtr> &v_output, const bool scalar_output = false);

  template <typename T>
  static Status ConstructTensorDescWithData(const GeTensorDesc &out_desc, const T *const buf, const size_t len,
                                            std::vector<GeTensorPtr> &v_output, const bool scalar_output = false);

  /**
   * Generating a sequence of numbers
   * @param [in] data_num the num of generate
   * @param [in] value the value to write to buffer
   * @param [out] output the tensor for save sequence of numbers
   * @author
   */
  template<typename T>
  static Status GenData(const int64_t data_num, const T value, const GeTensorPtr &output) {
    if (data_num > 0) {
      if (CheckInt64MulOverflow(data_num, static_cast<int64_t>(sizeof(T))) != SUCCESS) {
        GELOGE(PARAM_INVALID, "Int64MulOverflow, data_num(%ld) type_len(%zu)", data_num, sizeof(T));
        return PARAM_INVALID;
      }

      const std::unique_ptr<T[]> buf = MakeUnique<T[]>(data_num);
      const size_t buf_size = sizeof(T) * static_cast<size_t>(data_num);
      if (buf == nullptr) {
        GELOGE(MEMALLOC_FAILED, "new sizeof(T) * data_num(%zu) memory failed", buf_size);
        return MEMALLOC_FAILED;
      }

      for (int64_t i = 0; i < data_num; ++i) {
        buf[i] = value;
      }
      const Status ret = output->SetData(reinterpret_cast<uint8_t *>(buf.get()), buf_size);
      if (ret != SUCCESS) {
        GELOGE(ret, " buf must not be null.");
        return ret;
      }
    }

    return SUCCESS;
  }

  /**
  * Calculate dimension
  * @param [in] dims save the tensor of the dimension
  * @param [in] vec_dim results of each dimension
  * @param [out] data_num total size of data
  * @author
  */
  template <typename T>
  static Status CalcDims(const ConstGeTensorPtr dims, std::vector<int64_t> &vec_dim, int64_t &data_num) {
    data_num = 1;
    const size_t size = dims->GetData().size() / sizeof(T);

    for (size_t i = 0U; i < size; i++) {
      const T dim = *(PtrAdd<const T>(PtrToPtr<const uint8_t, const T>(dims->GetData().data()), size, i));
      if (dim < 0) {
        GELOGE(PARAM_INVALID, "input dim(%zu) is negative(%ld)", i, static_cast<int64_t>(dim));
        return PARAM_INVALID;
      }
      if (dim == 0) {
        GELOGI("input dim(%zu) is zero", i);
        data_num = 0;
        vec_dim.clear();
        break;
      }
      if (CheckInt64MulOverflow(data_num, dim) != SUCCESS) {
        GELOGE(PARAM_INVALID, "Int64MulOverflow, data_num(%ld) dim(%ld)", data_num, static_cast<int64_t>(dim));
        return PARAM_INVALID;
      }

      data_num *= dim;
      vec_dim.push_back(dim);
    }

    return SUCCESS;
  }
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_FOLDING_KERNEL_KERNEL_UTILS_H_
