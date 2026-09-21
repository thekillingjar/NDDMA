/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_TENSOR_SEQUENCE_H
#define AIR_CXX_RUNTIME_V2_TENSOR_SEQUENCE_H

#include <vector>
#include <sstream>

#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/tensor_data.h"
#include "graph/types.h"
#include "framework/common/debug/ge_log.h"
#include "base/err_msg.h"

namespace gert {
class TensorSeq;
using TensorSeqPtr = std::shared_ptr<TensorSeq>;
class TensorSeq {
 public:
  TensorSeq() = default;
  explicit TensorSeq(ge::DataType elem_type) noexcept : elem_type_{elem_type} {}

  struct TensorRef {
    TensorData tensor_addr_;
    Shape tensor_shape_;
  };

  using const_iterator = std::vector<TensorRef>::const_iterator;

  // Sets the element type after construction.
  // Expects sequence to be empty at the time.
  ge::graphStatus SetType(ge::DataType elem_type) {
    if (!tensors_.empty()) {
      GELOGE(ge::PARAM_INVALID, "tensor sequence is not empty, so can't set the elem_type.");
      REPORT_INNER_ERR_MSG("E39999", "tensor sequence is not empty, so can't set the elem_type.");
      return ge::PARAM_INVALID;
    }
    elem_type_ = elem_type;
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus SetElements(std::vector<TensorRef>&& tensors) {
    if (!tensors_.empty()) {
      GELOGE(ge::PARAM_INVALID, "tensor sequence is not empty, so can't set elements.");
      REPORT_INNER_ERR_MSG("E39999", "tensor sequence is not empty, so can't set elements.");
      return ge::PARAM_INVALID;
    }
    tensors_ = std::move(tensors);
    return ge::GRAPH_SUCCESS;
  }

  ge::DataType DataType() const noexcept { return elem_type_; }

  bool IsSameDataType(const TensorSeq& tensor_seq) const noexcept {
    return elem_type_ == tensor_seq.elem_type_;
  }

  size_t Size() const noexcept { return tensors_.size(); }

  // Suitable for for range loop
  const_iterator begin() const noexcept { return tensors_.cbegin(); }

  const_iterator end() const noexcept { return tensors_.cend(); }

  bool ValidateSeqIdx(int64_t index) const {
    bool ret = false;
    int64_t size = static_cast<int64_t>(tensors_.size());
    if (index < 0) {
      ret = (index <= -1) && (index >= -size);
    } else {
      ret = index < size;
    }

    if (!ret) {
      GELOGE(ge::PARAM_INVALID, "input index %lld is not valid, sequence's size %lld",
                       index, size);
      REPORT_INNER_ERR_MSG("E39999", "input is not valid");
    }
    return ret;
  }

  // Get by index
  const TensorRef* Get(int64_t index) const {
    if (!ValidateSeqIdx(index)) {
      return nullptr;
    }

    if (index < 0) {
      index += tensors_.size();
    }

    return &tensors_[index];
  }

  ge::graphStatus Add(TensorRef&& tensor, ge::DataType data_type) {
    if (elem_type_ != data_type) {
      GELOGE(ge::PARAM_INVALID, "The data type of add tensor is not equal with element type "
             "of tensor sequence, the input data type is [%u] , tensor sequence's element "
             "type is [%u].", data_type, elem_type_);
      REPORT_INNER_ERR_MSG("E39999", "The data type of add tensor is not equal with element type "
                         "of tensor sequence, the input data type is [%u] , tensor sequence's "
                        "element type is [%u].", data_type, elem_type_);
      return ge::PARAM_INVALID;
    }
    tensors_.push_back(std::move(tensor));
    return ge::GRAPH_SUCCESS;
  }

  std::string ShapeToString(Shape shape) {
    size_t dims = shape.GetDimNum();
    if (dims == 0) {
      return "";
    }

    std::stringstream ss;
    ss << "[";
    ss << shape[0];
    for (size_t i = 1; i < dims; i++) {
      ss << "," << shape[i];
    }
    ss << " ]";
    return ss.str();
  }

  ge::graphStatus Add(const Tensor& tensor) {
    auto data_type = tensor.GetDataType();
    if (elem_type_ != data_type) {
      GELOGE(ge::PARAM_INVALID, "The data type of add tensor is not equal with element type "
             "of tensor sequence, the input data type is [%u] , tensor sequence's element "
             "type is [%u].", data_type, elem_type_);
      REPORT_INNER_ERR_MSG("E39999", "The data type of add tensor is not equal with element type "
                         "of tensor sequence, the input data type is [%u] , tensor sequence's "
                        "element type is [%u].", data_type, elem_type_);
      return ge::PARAM_INVALID;
    }
    TensorRef tensor_ref;
    if (tensor_ref.tensor_addr_.ShareFrom(tensor.GetTensorData()) !=
        ge::GRAPH_SUCCESS) {
      GELOGE(ge::PARAM_INVALID, "Create tensor ref failed");
      REPORT_INNER_ERR_MSG("E39999", "Create tensor ref failed");
      return ge::PARAM_INVALID;
    }
    auto shape = tensor.GetStorageShape();
    tensor_ref.tensor_shape_.SetDimNum(shape.GetDimNum());
    for (size_t index = 0; index < shape.GetDimNum(); ++index) {
      tensor_ref.tensor_shape_.SetDim(index, shape.GetDim(index));
    }
    tensors_.push_back(std::move(tensor_ref));
    GELOGD("Add tensor success, data type is %u, tensor size is %llu",
           data_type, tensor.GetSize());
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus Add(const ge::DataType data_type,
                      const TensorData& tensor_data,
                      const StorageShape& storage_shape) {
    if (elem_type_ != data_type) {
      GELOGE(ge::PARAM_INVALID, "The data type of add tensor is not equal with element type "
             "of tensor sequence, the input data type is [%u] , tensor sequence's element "
             "type is [%u].", data_type, elem_type_);
      REPORT_INNER_ERR_MSG("E39999", "The data type of add tensor is not equal with element type "
                         "of tensor sequence, the input data type is [%u] , tensor sequence's "
                        "element type is [%u].", data_type, elem_type_);
      return ge::PARAM_INVALID;
    }
    TensorRef tensor_ref;
    if (tensor_ref.tensor_addr_.ShareFrom(tensor_data) != ge::GRAPH_SUCCESS) {
      GELOGE(ge::PARAM_INVALID, "Create tensor ref failed");
      REPORT_INNER_ERR_MSG("E39999", "Create tensor ref failed");
      return ge::PARAM_INVALID;
    }
    auto shape = storage_shape.GetOriginShape();
    tensor_ref.tensor_shape_.SetDimNum(shape.GetDimNum());

    for (size_t index = 0; index < shape.GetDimNum(); ++index) {
      tensor_ref.tensor_shape_.SetDim(index, shape.GetDim(index));
    }
    tensors_.push_back(std::move(tensor_ref));
    GELOGD("tensor sequence add tensor ref success, tensor shape is %s",
           ShapeToString(shape).c_str());
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus Add(const Tensor& tensor, int64_t index) {
    auto data_type = tensor.GetDataType();
    if (elem_type_ != data_type) {
      GELOGE(ge::PARAM_INVALID, "The data type of add tensor is not equal with element type "
             "of tensor sequence, the input data type is [%u] , tensor sequence's element "
             "type is [%u].", data_type, elem_type_);
      REPORT_INNER_ERR_MSG("E39999", "The data type of add tensor is not equal with element type "
                         "of tensor sequence, the input data type is [%u] , tensor sequence's "
                        "element type is [%u].", data_type, elem_type_);
      return ge::PARAM_INVALID;
    }

    if (!ValidateSeqIdx(index)) {
      return ge::PARAM_INVALID;
    }

    if (index < 0) {
      index += tensors_.size();
    }

    TensorRef tensor_ref;
    if (tensor_ref.tensor_addr_.ShareFrom(tensor.GetTensorData()) !=
        ge::GRAPH_SUCCESS) {
      GELOGE(ge::PARAM_INVALID, "Create tensor ref failed");
      REPORT_INNER_ERR_MSG("E39999", "Create tensor ref failed");
      return ge::PARAM_INVALID;
    }
    // optimize how to consturct shape
    auto shape = tensor.GetStorageShape();
    tensor_ref.tensor_shape_.SetDimNum(shape.GetDimNum());
    for (size_t idx = 0; idx < shape.GetDimNum(); ++idx) {
      tensor_ref.tensor_shape_.SetDim(idx, shape.GetDim(idx));
    }
    tensors_.insert(tensors_.begin() + index, std::move(tensor_ref));
    GELOGD("Add tensor success, index is %lld, tensor size is %llu",
           index, tensor.GetSize());
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus Add(const ge::DataType data_type,
                      const TensorData& tensor_data,
                      const StorageShape& storage_shape, int64_t index) {
    if (elem_type_ != data_type) {
      GELOGE(ge::PARAM_INVALID, "The data type of add tensor is not equal with element type "
             "of tensor sequence, the input data type is [%u] , tensor sequence's element "
             "type is [%u].", data_type, elem_type_);
      REPORT_INNER_ERR_MSG("E39999", "The data type of add tensor is not equal with element type "
                         "of tensor sequence, the input data type is [%u] , tensor sequence's "
                        "element type is [%u].", data_type, elem_type_);
      return ge::PARAM_INVALID;
    }

    if (!ValidateSeqIdx(index)) {
      return ge::PARAM_INVALID;
    }

    if (index < 0) {
      index += tensors_.size();
    }

    TensorRef tensor_ref;
    if (tensor_ref.tensor_addr_.ShareFrom(tensor_data) != ge::GRAPH_SUCCESS) {
      GELOGE(ge::PARAM_INVALID, "Create tensor ref failed");
      REPORT_INNER_ERR_MSG("E39999", "Create tensor ref failed");
      return ge::PARAM_INVALID;
    }
    auto shape = storage_shape.GetOriginShape();
    tensor_ref.tensor_shape_.SetDimNum(shape.GetDimNum());

    for (size_t idx = 0; idx < shape.GetDimNum(); ++idx) {
      tensor_ref.tensor_shape_.SetDim(idx, shape.GetDim(idx));
    }
    tensors_.insert(tensors_.begin() + index, std::move(tensor_ref));
    GELOGD("tensor sequence add ref tensor success, index is %lld, tensor shape is %s",
           index, ShapeToString(shape).c_str());
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus Erase(int64_t index) {
    if (!ValidateSeqIdx(index)) {
      return ge::PARAM_INVALID;
    }

    if (index < 0) {
      index += tensors_.size();
    }
    tensors_.erase(tensors_.begin() + index);
    return ge::GRAPH_SUCCESS;
  }

  void Reserve(size_t capacity) { tensors_.reserve(capacity); }

 private:
  ge::DataType elem_type_{ge::DT_FLOAT};
  std::vector<TensorRef> tensors_;
};
}  // namespace gert
#endif