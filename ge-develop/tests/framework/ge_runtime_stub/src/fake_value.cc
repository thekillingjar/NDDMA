/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faker/fake_value.h"

#include <random>
#include "array_ops.h"
#include "graph/ge_tensor.h"
#include "graph/utils/math_util.h"

namespace gert {
namespace {
template <typename T, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
void Random(T *value, size_t len) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_real_distribution<T> dis(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
  for (size_t i = 0; i < len; ++i) {
    value[i] = dis(rng);
  }
}

template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
void Random(T *value, size_t len) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<T> dis(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
  for (size_t i = 0; i < len; ++i) {
    value[i] = dis(rng);
  }
}

struct StubHostTensorHead {
  size_t count;
  static void *Create(size_t size) {
    auto head = static_cast<StubHostTensorHead *>(malloc(sizeof(StubHostTensorHead) + size));
    head->count = 1;
    return head + 1;
  }
};
ge::graphStatus HostTensorManager(TensorAddress addr, TensorOperateType operate_type, void **out) {
  auto head = static_cast<StubHostTensorHead *>(addr) - 1;
  switch (operate_type) {
    case kGetTensorAddress:
      *out = addr;
      break;
    case kPlusShareCount:
      head->count++;
      break;
    case kFreeTensor:
      if (--head->count == 0) {
        free(head);
      }
      break;
    default:
      return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace
FakeGeTensorHolder &FakeGeTensorHolder::OriginFormat(ge::Format origin_format) {
  FakeGeTensorHolder::origin_format = origin_format;
  return *this;
}
FakeGeTensorHolder &FakeGeTensorHolder::StorageFormat(ge::Format storage_format) {
  FakeGeTensorHolder::storage_format = storage_format;
  return *this;
}
FakeGeTensorHolder &FakeGeTensorHolder::OriginShape(std::initializer_list<int64_t> origin_shape) {
  FakeGeTensorHolder::origin_shape = origin_shape;
  return *this;
}
FakeGeTensorHolder &FakeGeTensorHolder::StorageShape(std::initializer_list<int64_t> storage_shape) {
  FakeGeTensorHolder::storage_shape = storage_shape;
  UpdateTensorDataSize();
  return *this;
}
FakeGeTensorHolder &FakeGeTensorHolder::DataType(ge::DataType dt) {
  FakeGeTensorHolder::data_type = dt;
  UpdateTensorDataSize();
  return *this;
}
ge::GeTensorPtr FakeGeTensorHolder::Build() const {
  std::vector<uint8_t> fake_data(tensor_data_size_in_bytes);
  for (uint8_t &i : fake_data) {
    i = static_cast<uint8_t>(rand() % std::numeric_limits<uint16_t>::max());
  }
  ge::GeTensorDesc td;
  td.SetFormat(storage_format);
  td.SetOriginFormat(origin_format);
  td.SetShape(ge::GeShape(storage_shape));
  td.SetOriginShape(ge::GeShape(origin_shape));
  td.SetDataType(data_type);

  auto tensor = std::make_shared<ge::GeTensor>(td);
  tensor->SetData(std::move(fake_data));
  return tensor;
}
void FakeGeTensorHolder::UpdateTensorDataSize() {
  int64_t shape_size = 1;
  for (auto dim : FakeGeTensorHolder::storage_shape) {
    shape_size *= dim;
  }
  tensor_data_size_in_bytes = ge::GetSizeInBytes(shape_size, data_type);
}
TensorFaker &TensorFaker::Shape(std::initializer_list<int64_t> shape) {
  return OriginShape(shape).StorageShape(shape);
}
TensorFaker &TensorFaker::OriginShape(std::initializer_list<int64_t> shape) {
  tensor_.MutableOriginShape() = shape;
  return *this;
}
TensorFaker &TensorFaker::StorageShape(std::initializer_list<int64_t> shape) {
  tensor_.MutableStorageShape() = shape;
  return *this;
}
TensorFaker &TensorFaker::Format(ge::Format format) {
  return OriginFormat(format).StorageFormat(format);
}
TensorFaker &TensorFaker::OriginFormat(ge::Format format) {
  tensor_.MutableFormat().SetOriginFormat(format);
  return *this;
}
TensorFaker &TensorFaker::StorageFormat(ge::Format format) {
  tensor_.MutableFormat().SetStorageFormat(format);
  return *this;
}
TensorFaker &TensorFaker::ExpandDimsType(gert::ExpandDimsType edt) {
  tensor_.MutableFormat().SetExpandDimsType(edt);
  return *this;
}
TensorFaker &TensorFaker::DataType(ge::DataType dt) {
  tensor_.SetDataType(dt);
  return *this;
}
TensorFaker &TensorFaker::Placement(TensorPlacement placement) {
  tensor_.SetPlacement(placement);
  return *this;
}
TensorHolder TensorFaker::Build() const {
  TensorHolder th;
  if (tensor_.GetPlacement() == kFollowing) {
    if (!alloc_tensor_data_) {
      throw std::invalid_argument("The following placement must alloc tensor data, you should not use `DontAllocData`");
    }
    size_t total_size;
    th.SetFollowingTensor(
        Tensor::CreateFollowing(tensor_.GetStorageShape().GetShapeSize(), tensor_.GetDataType(), total_size));
  } else {
    th.SetTensor(std::unique_ptr<Tensor>(new Tensor));
    auto tensor_size = ge::GetSizeInBytes(tensor_.GetStorageShape().GetShapeSize(), tensor_.GetDataType());
    tensor_size = ge::RoundUp(tensor_size, 32) + 32;
    if (alloc_tensor_data_) {
      auto block = StubHostTensorHead::Create(tensor_size);
      th.SetBlock(block);
      TensorData td;
      td.SetAddr(block, HostTensorManager);
      td.SetSize(tensor_size);
      td.SetPlacement(tensor_.GetPlacement());
      th.GetTensor()->SetData(std::move(td));
    }
  }
  th.GetTensor()->MutableFormat() = tensor_.GetFormat();
  th.GetTensor()->MutableOriginShape() = tensor_.GetOriginShape();
  th.GetTensor()->MutableStorageShape() = tensor_.GetStorageShape();
  th.GetTensor()->SetDataType(tensor_.GetDataType());
  th.GetTensor()->SetPlacement(tensor_.GetPlacement());

  if (alloc_tensor_data_) {
    if (tensor_.GetPlacement() == kFollowing || tensor_.GetPlacement() == kOnHost) {
      if (tensor_value_.empty()) {
        auto shape_size = th.GetTensor()->GetStorageShape().GetShapeSize();
        auto tensor_addr = th.GetTensor()->GetAddr();
        switch (th.GetTensor()->GetDataType()) {
          case ge::DT_FLOAT:
            Random(static_cast<float *>(tensor_addr), shape_size);
            break;
          case ge::DT_FLOAT16:
          case ge::DT_INT16:
            Random(static_cast<int16_t *>(tensor_addr), shape_size);
            break;
          case ge::DT_INT32:
            Random(static_cast<int32_t *>(tensor_addr), shape_size);
            break;
          case ge::DT_INT64:
            Random(static_cast<int64_t *>(tensor_addr), shape_size);
            break;
          default:
            throw std::invalid_argument("unsupport data type");
        }
      } else {
        memcpy(th.GetTensor()->GetAddr(), tensor_value_.data(), tensor_value_.size());
      }
    }
  }

  return th;
}
TensorFaker &TensorFaker::DontAllocData() {
  alloc_tensor_data_ = false;
  return *this;
}

FakeTensors::FakeTensors(std::initializer_list<int64_t> shape, size_t num,
                         void *mem_block, TensorPlacement placement,
                         ge::Format format, ge::DataType data_type) {
  TensorData tensor_data{nullptr, nullptr, 0, placement};
  size_t tensor_size = 8;  // 1 int64 takes 8 bytes
  for (auto dim : shape) {
    tensor_size *= static_cast<size_t>(dim);
  }
  tensor_size = ge::RoundUp(tensor_size, 32) + 32;
  if (mem_block == nullptr) {
    mem_block = StubHostTensorHead::Create(tensor_size);
    tensor_data.SetAddr(mem_block, HostTensorManager);
    tensor_data.SetSize(tensor_size);
    const char tensor_value_repeat[4] = {'F', 'A', 'K', 'E'};
    for (size_t j = 0U; j < tensor_size; ++j) {
      static_cast<uint8_t *>(mem_block)[j] = tensor_value_repeat[j % sizeof(tensor_value_repeat)];
    }
  } else {
    tensor_data.SetAddr(mem_block, nullptr);
    tensor_data.SetSize(tensor_size);
  }
  for (size_t i = 0; i < num; i++) {
    Tensor tensor{{shape, shape}, {format, format, {}}, placement, data_type, nullptr};
    tensor.MutableTensorData().ShareFrom(tensor_data);
    tensors_holder_.push_back(std::move(tensor));
  }
  for (auto &tensor : tensors_holder_) {
    addrs_.push_back(&tensor);
  }
}
size_t TensorHolder::GetRefCount() const {
  if (block_ == nullptr) {
    return 0;
  }
  return (static_cast<const StubHostTensorHead *>(block_) - 1)->count;
}
}  // namespace gert
