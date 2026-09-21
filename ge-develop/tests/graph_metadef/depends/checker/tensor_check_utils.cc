/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensor_check_utils.h"

#include "checker.h"
#include "tensor_adapter.h"

#include <sstream>
#include "ge/ge_allocator.h"

#include <symengine/symengine_rcp.h>

namespace ge {
namespace {
class AllocatorUtStub : public Allocator {
 public:
  MemBlock *Malloc(size_t size) override {
    auto addr = new (std::nothrow) int8_t[size];
    GE_ASSERT_NOTNULL(addr);
    auto block = new (std::nothrow) MemBlock(*this, addr, size);
    GE_ASSERT_NOTNULL(block);
    blocks_.insert(block);
    return block;
  }
  MemBlock *MallocAdvise(size_t size, void *addr) override {
    return Malloc(size);
  }
  void Free(MemBlock *block) override {
    if (block != nullptr) {
      blocks_.erase(block);
      delete[] reinterpret_cast<uint8_t *>(block->GetAddr());
      delete block;
    }
  }
  ~AllocatorUtStub() {
    for (auto block : blocks_) {
      delete block;
    }
  }

 private:
  std::set<MemBlock *> blocks_;
};

AllocatorUtStub g_stub_allocator;

graphStatus StubMemBlockManager(void *addr, gert::TensorOperateType operate_type, void **out) {
  GE_ASSERT_NOTNULL(addr);
  switch (operate_type) {
    case gert::TensorOperateType::kFreeTensor: {
      reinterpret_cast<MemBlock *>(addr)->Free();
      break;
    }
    case gert::TensorOperateType::kGetTensorAddress: {
      *out = reinterpret_cast<MemBlock *>(addr)->GetAddr();
      break;
    }
    case gert::TensorOperateType::kPlusShareCount: {
      reinterpret_cast<MemBlock *>(addr)->AddCount();
      break;
    }
    default: {
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}
}  // namespace
void TensorCheckUtils::ConstructGertTensor(gert::Tensor &gert_tensor, const std::initializer_list<int64_t> &dims,
                                           const DataType &data_type, const Format &format,
                                           const gert::TensorPlacement placement) {
  gert_tensor.SetDataType(data_type);
  gert_tensor.MutableOriginShape() = dims;
  gert_tensor.MutableStorageShape() = dims;
  gert_tensor.SetStorageFormat(format);
  gert_tensor.SetOriginFormat(format);
  const auto tensor_size = GetSizeInBytes(gert_tensor.GetShapeSize(), data_type);
  const auto buffer_size = (tensor_size + 32U - 1U) / 32U * 32U; // 32B aligned
  auto mem_block = g_stub_allocator.Malloc(buffer_size);
  gert::TensorData tensor_data(mem_block, StubMemBlockManager, tensor_size, placement);
  gert_tensor.SetData(std::move(tensor_data));

  auto addr = PtrToPtr<void, uint8_t>(gert_tensor.GetAddr());

  auto data = reinterpret_cast<uint32_t *>(addr);
  for (size_t i = 0; i < buffer_size / sizeof(uint32_t); ++i) {
    data[i] = i;
  }
}

void TensorCheckUtils::ConstructGeTensor(ge::GeTensor &ge_tensor, const std::initializer_list<int64_t> &dims,
                                         const DataType &data_type, const Format &format, ge::Placement placement) {
  ge_tensor.MutableTensorDesc().SetPlacement(placement);
  ge_tensor.MutableTensorDesc().SetDataType(data_type);
  ge_tensor.MutableTensorDesc().SetOriginDataType(data_type);
  ge_tensor.MutableTensorDesc().SetOriginShape(GeShape(dims));
  ge_tensor.MutableTensorDesc().SetShape(GeShape(dims));
  ge_tensor.MutableTensorDesc().SetFormat(format);
  ge_tensor.MutableTensorDesc().SetOriginFormat(format);

  const auto buffer_size = GetSizeInBytes(ge_tensor.GetTensorDesc().GetShape().GetShapeSize(),
    data_type);
  auto mem_block = g_stub_allocator.Malloc(buffer_size);

  auto deleter = [mem_block] (uint8_t *addr) {
    (void)addr;
    mem_block->Free();
  };
  auto addr = PtrToPtr<void, uint8_t>(mem_block->GetAddr());

  auto data = reinterpret_cast<uint32_t *>(addr);
  for (size_t i = 0; i < buffer_size / sizeof(uint32_t); ++i) {
    data[i] = i;
  }

  ge_tensor.SetData(addr, mem_block->GetSize(), deleter);
}

std::string TensorCheckUtils::ShapeStr(const gert::Shape &shape) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < shape.GetDimNum(); ++i) {
    ss << shape.GetDim(i) << ", ";
  }
  ss << "]";
  return ss.str();
}

std::string TensorCheckUtils::ShapeStr(const ge::Shape &shape) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < shape.GetDimNum(); ++i) {
    ss << shape.GetDim(i) << ", ";
  }
  ss << "]";
  return ss.str();
}

std::string TensorCheckUtils::ShapeStr(const ge::GeShape &shape) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < shape.GetDimNum(); ++i) {
    ss << shape.GetDim(i) << ", ";
  }
  ss << "]";
  return ss.str();
}
Status TensorCheckUtils::CheckTensorEqGertTensor(const Tensor &tensor, const gert::Tensor &gert_tensor) {
  return CheckGeTensorEqGertTensor(TensorAdapter::AsGeTensor(tensor), gert_tensor);
}

Status TensorCheckUtils::CheckGeTensorEqGertTensor(const GeTensor &ge_tensor, const gert::Tensor &gert_tensor) {
  auto ge_tensor_desc = ge_tensor.GetTensorDesc();
  if (ge_tensor_desc.GetDataType() != gert_tensor.GetDataType()) {
    std::cerr << "data type not equal, expected " << gert_tensor.GetDataType() << ", actual "
              << ge_tensor_desc.GetDataType() << std::endl;
    return GRAPH_FAILED;
  }
  if (ge_tensor_desc.GetFormat() != gert_tensor.GetFormat().GetStorageFormat()) {
    std::cerr << "storage format not equal, expected " << gert_tensor.GetFormat().GetStorageFormat() << ", actual "
              << ge_tensor_desc.GetFormat() << std::endl;
    return GRAPH_FAILED;
  }
  if (ge_tensor_desc.GetFormat() != gert_tensor.GetFormat().GetOriginFormat()) {
    std::cerr << "origin format not equal, expected " << gert_tensor.GetFormat().GetOriginFormat() << ", actual "
              << ge_tensor_desc.GetFormat() << std::endl;
    return GRAPH_FAILED;
  }

  if (ge_tensor_desc.IsOriginShapeInitialized()) {
    if (ShapeStr(ge_tensor_desc.GetOriginShape()) != ShapeStr(gert_tensor.GetShape().GetOriginShape())) {
      std::cerr << "origin shape not equal, expected " << ShapeStr(gert_tensor.GetShape().GetOriginShape())
                << ", actual " << ShapeStr(ge_tensor_desc.GetOriginShape()) << std::endl;
      return GRAPH_FAILED;
    }
  }

  if (ShapeStr(ge_tensor_desc.GetShape()) != ShapeStr(gert_tensor.GetShape().GetStorageShape())) {
    std::cerr << "storage shape not equal, expected " << ShapeStr(gert_tensor.GetShape().GetStorageShape())
              << ", actual " << ShapeStr(ge_tensor_desc.GetShape()) << std::endl;
    return GRAPH_FAILED;
  }
  auto exp_placement = gert::TensorPlacementUtils::IsOnDevice(gert_tensor.GetPlacement())
                           ? ge::Placement::kPlacementDevice
                           : ge::Placement::kPlacementHost;
  if (ge_tensor_desc.GetPlacement() != exp_placement) {
    std::cerr << "placement not equal, expected " << exp_placement << ", actual " << ge_tensor_desc.GetPlacement()
              << std::endl;
    return GRAPH_FAILED;
  }

  if ((void *)ge_tensor.GetData().GetData() != gert_tensor.GetAddr()) {
    std::cerr << "data address not equal, expected " << gert_tensor.GetAddr() << ", actual "
              << (void *)ge_tensor.GetData().GetData() << std::endl;
    return GRAPH_FAILED;
  }
  if (ge_tensor.GetData().GetSize() != gert_tensor.GetSize()) {
    std::cerr << "data size not equal, expected " << gert_tensor.GetSize() << ", actual "
              << ge_tensor.GetData().GetSize() << std::endl;
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

Status TensorCheckUtils::CheckGeTensorEqGertTensorWithData(const GeTensor &ge_tensor, const gert::Tensor &gert_tensor) {
  auto ge_tensor_desc = ge_tensor.GetTensorDesc();
  if (ge_tensor_desc.GetDataType() != gert_tensor.GetDataType()) {
    std::cerr << "data type not equal, expected " << gert_tensor.GetDataType() << ", actual "
              << ge_tensor_desc.GetDataType() << std::endl;
    return GRAPH_FAILED;
  }
  if (ge_tensor_desc.GetFormat() != gert_tensor.GetFormat().GetStorageFormat()) {
    std::cerr << "storage format not equal, expected " << gert_tensor.GetFormat().GetStorageFormat() << ", actual "
              << ge_tensor_desc.GetFormat() << std::endl;
    return GRAPH_FAILED;
  }
  if (ge_tensor_desc.GetFormat() != gert_tensor.GetFormat().GetOriginFormat()) {
    std::cerr << "origin format not equal, expected " << gert_tensor.GetFormat().GetOriginFormat() << ", actual "
              << ge_tensor_desc.GetFormat() << std::endl;
    return GRAPH_FAILED;
  }

  if (ge_tensor_desc.IsOriginShapeInitialized()) {
    if (ShapeStr(ge_tensor_desc.GetOriginShape()) != ShapeStr(gert_tensor.GetShape().GetOriginShape())) {
      std::cerr << "origin shape not equal, expected " << ShapeStr(gert_tensor.GetShape().GetOriginShape())
                << ", actual " << ShapeStr(ge_tensor_desc.GetOriginShape()) << std::endl;
      return GRAPH_FAILED;
    }
  }

  if (ShapeStr(ge_tensor_desc.GetShape()) != ShapeStr(gert_tensor.GetShape().GetStorageShape())) {
    std::cerr << "storage shape not equal, expected " << ShapeStr(gert_tensor.GetShape().GetStorageShape())
              << ", actual " << ShapeStr(ge_tensor_desc.GetShape()) << std::endl;
    return GRAPH_FAILED;
  }
  auto exp_placement = gert::TensorPlacementUtils::IsOnDevice(gert_tensor.GetPlacement())
                           ? ge::Placement::kPlacementDevice
                           : ge::Placement::kPlacementHost;
  if (ge_tensor_desc.GetPlacement() != exp_placement) {
    std::cerr << "placement not equal, expected " << exp_placement << ", actual " << ge_tensor_desc.GetPlacement()
              << std::endl;
    return GRAPH_FAILED;
  }
  if (ge_tensor.GetData().GetSize() != gert_tensor.GetSize()) {
    std::cerr << "data size not equal, expected " << gert_tensor.GetSize() << ", actual "
              << ge_tensor.GetData().GetSize() << std::endl;
    return GRAPH_FAILED;
  }
  const uint8_t *origin_data = ge_tensor.GetData().GetData();
  const uint8_t *gert_data = reinterpret_cast<const uint8_t *>(gert_tensor.GetAddr());
  for (size_t i = 0U; i < ge_tensor.GetData().GetSize(); ++i) {
    if (origin_data[i] != gert_data[i]) {
      std::cerr << "data not equal, expected " << gert_data[i] << ", actual " << origin_data[i] << std::endl;
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}
}  // namespace ge