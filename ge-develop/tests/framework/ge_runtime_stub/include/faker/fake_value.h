/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_383E8BD8062B45F5A81AF6E929734D33_H
#define INC_383E8BD8062B45F5A81AF6E929734D33_H

#include "exe_graph/runtime/tensor.h"
#include "graph/ge_tensor.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include <vector>
#include <memory>

namespace gert {
struct FakeGeTensorHolder {
  std::vector<int64_t> origin_shape;
  std::vector<int64_t> storage_shape;
  ge::Format origin_format;
  ge::Format storage_format;
  ge::DataType data_type;
  int64_t tensor_data_size_in_bytes;

  FakeGeTensorHolder &OriginFormat(ge::Format origin_format);
  FakeGeTensorHolder &StorageFormat(ge::Format storage_format);
  FakeGeTensorHolder &OriginShape(std::initializer_list<int64_t> origin_shape);
  FakeGeTensorHolder &StorageShape(std::initializer_list<int64_t> storage_shape);
  FakeGeTensorHolder &DataType(ge::DataType dt);

  ge::GeTensorPtr Build() const;

 private:
  void UpdateTensorDataSize();
};

struct TensorHolder {
 public:
  void SetTensor(std::unique_ptr<Tensor> &&tensor) {
    tensor_ = std::move(tensor);
  }

  void SetFollowingTensor(std::unique_ptr<uint8_t[]> &&following_tensor) {
    following_tensor_ = std::move(following_tensor);
  }

  Tensor *GetTensor() {
    if (tensor_ != nullptr) {
      return tensor_.get();
    }
    if (following_tensor_ != nullptr) {
      return reinterpret_cast<Tensor *>(following_tensor_.get());
    }
    return nullptr;
  }

  void SetBlock(void *block) {
    block_ = block;
  }

  size_t GetRefCount() const;

 private:
  std::unique_ptr<Tensor> tensor_;
  std::unique_ptr<uint8_t[]> following_tensor_;
  void *block_;
};

class TensorFaker {
 public:
  TensorFaker &Shape(std::initializer_list<int64_t> shape);
  TensorFaker &OriginShape(std::initializer_list<int64_t> shape);
  TensorFaker &StorageShape(std::initializer_list<int64_t> shape);

  TensorFaker &Format(ge::Format format);
  TensorFaker &OriginFormat(ge::Format format);
  TensorFaker &StorageFormat(ge::Format format);
  TensorFaker &ExpandDimsType(gert::ExpandDimsType edt);

  TensorFaker &DataType(ge::DataType dt);

  TensorFaker &Placement(TensorPlacement placement);

  template <typename T>
  TensorFaker &Value(const std::vector<T> &value) {
    tensor_value_.resize(sizeof(T) * value.size());
    memcpy(tensor_value_.data(), value.data(), tensor_value_.size());
    return *this;
  }

  TensorFaker &DontAllocData();

  [[nodiscard]] TensorHolder Build() const;

 private:
  std::vector<uint8_t> tensor_value_;
  Tensor tensor_{
      {{}, {}},                            // shape
      {ge::FORMAT_ND, ge::FORMAT_ND, {}},  // format
      kOnHost,                             // placement
      ge::DT_FLOAT,                        // data type
      nullptr                              // address
  };
  bool alloc_tensor_data_ = true;
};

struct FakeTensors {
  FakeTensors(std::initializer_list<int64_t> shape, size_t num, void *mem_block = nullptr,
              TensorPlacement = kOnDeviceHbm, ge::Format format = ge::FORMAT_ND, ge::DataType datatype = ge::DT_FLOAT);

  void **GetAddrList() {
    return (void **)addrs_.data();
  }

  Tensor **GetTensorList() {
    return addrs_.data();
  }
  Tensor *data() {
    return tensors_holder_.data();
  }

  size_t size() const {
    return tensors_holder_.size();
  }

  Tensor &at(size_t i) {
    return tensors_holder_.at(i);
  }

  const Tensor &at(size_t i) const {
    return tensors_holder_.at(i);
  }

  std::vector<Tensor> Steal() && {
    addrs_.clear();
    return std::move(tensors_holder_);
  }

 private:
  std::vector<Tensor> tensors_holder_;
  std::vector<Tensor *> addrs_;
};

template <typename T>
struct ValueHolder {
  void *value;
  std::unique_ptr<T> holder;
};

template <typename T, typename std::enable_if<(sizeof(T) > sizeof(void *)), int>::type = 0>
ValueHolder<T> FakeValue(T &&value) {
  ValueHolder<T> holder;
  holder.holder = std::unique_ptr<T>(new T(std::forward<T>(value)));
  holder.value = holder.holder.get();
  return holder;
}

template <typename T, typename std::enable_if<(sizeof(T) <= sizeof(void *)), int>::type = 0>
ValueHolder<T> FakeValue(T value) {
  ValueHolder<T> holder;
  *reinterpret_cast<T *>(&holder.value) = value;
  return holder;
}

}  // namespace gert

#endif
