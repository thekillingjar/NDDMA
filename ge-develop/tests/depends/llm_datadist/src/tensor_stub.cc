/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/tensor.h"
#include "llm_engine_test_helper.h"
#include "llm_string_util.h"
#include "llm_test_helper.h"
#include "common/mem_utils.h"
#include "common/llm_checker.h"
#include "graph_metadef/graph/aligned_ptr.h"
#include "graph_metadef/graph/debug/ge_util.h"

namespace ge {
const int64_t UNKNOWN_DIM_SIZE = -1;

// mock shape beg
// If not overflow return true
static bool Int64MulNotOverflow(const int64_t a, const int64_t b) {
  if (a > 0) {
    if (b > 0) {
      if (a > (INT64_MAX / b)) {
        return false;
      }
    } else {
      if (b < (INT64_MIN / a)) {
        return false;
      }
    }
  } else {
    if (b > 0) {
      if (a < (INT64_MIN / b)) {
        return false;
      }
    } else {
      if ((a != 0) && (b < (INT64_MAX / a))) {
        return false;
      }
    }
  }
  return true;
}

class ShapeImpl {
 public:
  ShapeImpl() = default;
  ~ShapeImpl() = default;
  explicit ShapeImpl(const std::vector<int64_t> &dims) {
    bool is_unknown_dim_num = false;
    for (const auto &dim : dims) {
      if (dim == UNKNOWN_DIM_NUM) {
        is_unknown_dim_num = true;
        break;
      }
    }
    dims_ = is_unknown_dim_num ? std::vector<int64_t>({UNKNOWN_DIM_NUM}) : dims;
  }

 private:
  std::vector<int64_t> dims_;
  friend class Shape;
};

Shape::Shape() { impl_ = ComGraphMakeShared<ShapeImpl>(); }

Shape::Shape(const std::vector<int64_t> &dims) { impl_ = ComGraphMakeShared<ShapeImpl>(dims); }

size_t Shape::GetDimNum() const {
  if (impl_ != nullptr) {
    const bool is_dim_unknown = std::any_of(std::begin(impl_->dims_), std::end(impl_->dims_),
                                            [](const int64_t i) { return i == UNKNOWN_DIM_NUM; });
    if (is_dim_unknown) {
      LLMLOGI("Dim num is unknown, return 0U.");
      return 0U;
    }
    return impl_->dims_.size();
  }
  return 0U;
}

int64_t Shape::GetDim(size_t idx) const {
  if (impl_ != nullptr) {
    if (idx >= impl_->dims_.size()) {
      return 0;
    }
    return impl_->dims_[idx];
  }
  return 0;
}

graphStatus Shape::SetDim(size_t idx, int64_t value) {
  if (impl_ != nullptr) {
    if (idx >= impl_->dims_.size()) {
      return GRAPH_FAILED;
    }
    impl_->dims_[idx] = value;
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

std::vector<int64_t> Shape::GetDims() const {
  const std::vector<int64_t> dims;
  if (impl_ != nullptr) {
    return impl_->dims_;
  }
  return dims;
}

int64_t Shape::GetShapeSize() const {
  if (impl_ != nullptr) {
    if (impl_->dims_.empty()) {
      return 0;
    }
    int64_t size = 1;
    for (const auto i : impl_->dims_) {
      if ((i == UNKNOWN_DIM_NUM) || (i == UNKNOWN_DIM)) {
        return UNKNOWN_DIM_SIZE;
      }

      if (!Int64MulNotOverflow(size, i)) {
        REPORT_INNER_ERR_MSG("E18888", "mul overflow: %" PRId64 ", %" PRId64, size, i);
        LLMLOGE(GRAPH_FAILED, "[Check][Overflow] mul overflow: %" PRId64 ", %" PRId64, size, i);
        size = 0;
        return size;
      }
      size *= i;
    }
    return size;
  }
  return 0;
}
// mock shape end

// mock TensorDesc beg
class TensorDescImpl {
 public:
  TensorDescImpl() = default;
  ~TensorDescImpl() = default;
  TensorDescImpl(const Shape &shape, const Format format, const DataType dt)
      : shape_(shape), format_(format), data_type_(dt) {}
 private:
  Shape shape_;
  std::vector<std::pair<int64_t, int64_t>> range_;
  Format format_ = FORMAT_ND;
  Format origin_format_ = FORMAT_ND;
  bool origin_format_is_set_ = false;
  DataType data_type_ = DT_FLOAT;
  Shape origin_shape_;
  bool origin_shape_is_set_ = false;
  int64_t size_ = 0;
  int64_t real_dim_cnt_ = 0;
  std::string name_;
  Placement placement_ = kPlacementHost;
  // TensorDescValue tensor_desc_value_;
  std::string expand_dims_rule_;
  bool reuse_input_ = false;
  uint32_t reuse_input_index_ = 0U;

  friend class TensorDesc;
};

void TensorDesc::SetRealDimCnt(const int64_t real_dim_cnt) {
  if (impl != nullptr) {
    impl->real_dim_cnt_ = real_dim_cnt;
  }
}

TensorDesc::TensorDesc() {
  impl = ComGraphMakeSharedAndThrow<TensorDescImpl>();
}

TensorDesc::TensorDesc(Shape shape, Format format, DataType dt) {
  impl = ComGraphMakeSharedAndThrow<TensorDescImpl>(shape, format, dt);
  SetRealDimCnt(static_cast<int64_t>(shape.GetDimNum()));
}

TensorDesc::TensorDesc(const TensorDesc &desc) {
  // Copy
  impl = ComGraphMakeShared<TensorDescImpl>();
  if ((desc.impl != nullptr) && (impl != nullptr)) {
    *impl = *desc.impl;
  }
}

TensorDesc::TensorDesc(TensorDesc &&desc) {
  // Move
  impl = std::move(desc.impl);
}

TensorDesc &TensorDesc::operator=(const TensorDesc &desc) {
  // Copy
  if (&desc != this) {
    impl = ComGraphMakeShared<TensorDescImpl>();
    if ((desc.impl != nullptr) && (impl != nullptr)) {
      *impl = *desc.impl;
    }
  }
  return *this;
}

TensorDesc &TensorDesc::operator=(TensorDesc &&desc) {
  if (&desc != this) {
    impl = std::move(desc.impl);
  }
  return *this;
}

void TensorDesc::SetPlacement(Placement placement) {
  if (impl != nullptr) {
    impl->placement_ = placement;
  }
}

Shape TensorDesc::GetShape() const {
  if (impl != nullptr) {
    return impl->shape_;
  }
  return Shape();
}

void TensorDesc::SetShape(const Shape &shape) {
  if (impl != nullptr) {
    impl->shape_ = shape;
  }
}

void TensorDesc::Update(const Shape &shape, Format format, DataType dt) {
  if (impl != nullptr) {
    impl->shape_ = shape;
    impl->format_ = format;
    impl->data_type_ = dt;
  }
}

DataType TensorDesc::GetDataType() const {
  if (impl != nullptr) {
    return impl->data_type_;
  }
  return DT_UNDEFINED;
}

// mock TensorDesc end

// mock Tensor beg
class TensorImpl {
 public:
  TensorImpl() = default;
  ~TensorImpl() = default;

  explicit TensorImpl(const TensorDesc &tensor_desc) : tensor_desc_(tensor_desc) {}
  TensorImpl(const TensorDesc &tensor_desc, const uint8_t * const data, const size_t size) :
    tensor_desc_(tensor_desc) {
    SetData(data, size);
  }
  graphStatus SetData(const uint8_t *const data, const size_t size);
  graphStatus SetData(std::vector<uint8_t> &&data);
  graphStatus SetData(const std::vector<uint8_t> &data);
  graphStatus SetData(const std::string &data);
  graphStatus SetData(const std::vector<std::string> &datas);
  graphStatus SetData(uint8_t *data, size_t size, const Tensor::DeleteFunc &deleter_func);

  const uint8_t *MallocAlignedPtr(const size_t size);
  size_t GetSize() const;
  const uint8_t *GetData() const;
  uint8_t *GetData();
  void clear();
  void SetTensorDesc(const TensorDesc &tensor_desc);
  TensorDesc GetTensorDesc();

private:
  TensorDesc tensor_desc_;
  std::shared_ptr<AlignedPtr> aligned_ptr_ = nullptr;
  size_t length_ = 0UL;
  // functions data() & mutable_data() return address of invalid_data_ when length_ is 0
  // defined for coding convenience
  static uint32_t invalid_data_;
  friend class Tensor;
};
uint32_t TensorImpl::invalid_data_ = 0x3A2D2900U;

void TensorImpl::SetTensorDesc(const TensorDesc &tensor_desc) {
  tensor_desc_ = tensor_desc;
}

TensorDesc TensorImpl::GetTensorDesc() {
  return tensor_desc_;
}

const uint8_t *TensorImpl::MallocAlignedPtr(const size_t size) {
  if (size == 0UL) {
    LLMLOGW("[Check][Param] Input data size is 0");
    clear();
    return llm::PtrToPtr<uint32_t, uint8_t>(&invalid_data_);
  }
  if (length_ != size) {
    aligned_ptr_.reset();
  }
  length_ = size;
  if (aligned_ptr_ == nullptr) {
    aligned_ptr_ = llm::MakeShared<AlignedPtr>(length_);
    if (aligned_ptr_ == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "create AlignedPtr failed.");
      LLMLOGE(GRAPH_FAILED, "[Create][AlignedPtr] failed.");
      return nullptr;
    }
  }

  return aligned_ptr_->Get();
}

size_t TensorImpl::GetSize() const { return length_; }

const uint8_t *TensorImpl::GetData() const {
  if (length_ == 0UL) {
    return llm::PtrToPtr<uint32_t, uint8_t>(&invalid_data_);
  }
  if (aligned_ptr_ == nullptr) {
    return nullptr;
  }
  return aligned_ptr_->Get();
}

uint8_t *TensorImpl::GetData() {
  if (length_ == 0UL) {
    return llm::PtrToPtr<uint32_t, uint8_t>(&invalid_data_);
  }
  if (aligned_ptr_ == nullptr) {
    return nullptr;
  }
  return aligned_ptr_->MutableGet();
}

void TensorImpl::clear() {
  aligned_ptr_.reset();
  length_ = 0UL;
}

graphStatus TensorImpl::SetData(const uint8_t *const data, const size_t size) {
  if (size == 0UL) {
    LLMLOGD("size is 0");
    clear();
    return GRAPH_SUCCESS;
  }
  if (data == nullptr) {
    LLMLOGD("data addr is empty");
    return GRAPH_SUCCESS;
  }

  if (MallocAlignedPtr(size) == nullptr) {
    LLMLOGE(GRAPH_FAILED, "[Malloc][Memory] failed, size=%zu", size);
    return GRAPH_FAILED;
  }

  size_t remain_size = size;
  auto dst_addr = llm::PtrToValue(aligned_ptr_->MutableGet());
  auto src_addr = llm::PtrToValue(data);
  while (remain_size > SECUREC_MEM_MAX_LEN) {
    if (memcpy_s(llm::ValueToPtr(dst_addr), SECUREC_MEM_MAX_LEN,
                 llm::ValueToPtr(src_addr), SECUREC_MEM_MAX_LEN) != EOK) {
      REPORT_INNER_ERR_MSG("E18888", "memcpy failed, size = %lu", SECUREC_MEM_MAX_LEN);
      LLMLOGE(GRAPH_FAILED, "[Memcpy][Data] failed, size = %lu", SECUREC_MEM_MAX_LEN);
      return GRAPH_FAILED;
    }
    remain_size -= SECUREC_MEM_MAX_LEN;
    dst_addr += SECUREC_MEM_MAX_LEN;
    src_addr += SECUREC_MEM_MAX_LEN;
  }
  if (memcpy_s(llm::ValueToPtr(dst_addr), remain_size,
               llm::ValueToPtr(src_addr), remain_size) != EOK) {
    REPORT_INNER_ERR_MSG("E18888", "memcpy failed, size=%zu", remain_size);
    LLMLOGE(GRAPH_FAILED, "[Memcpy][Data] failed, size=%zu", remain_size);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus TensorImpl::SetData(std::vector<uint8_t> &&data) {
  return SetData(data.data(), data.size());
}

graphStatus TensorImpl::SetData(const std::vector<uint8_t> &data) {
  return SetData(data.data(), data.size());
}

graphStatus TensorImpl::SetData(const std::string &data) {
  if (!data.empty()) {
    /// Extra 16 bytes store string head
    /// Extra 1 byte store '\0'
    const size_t total_size = data.size() + sizeof(StringHead) + 1U;
    const std::unique_ptr<char_t[]> buff = ComGraphMakeUnique<char_t[]>(total_size);
    if (buff == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "allocate string raw data buff failed, size:%zu", total_size);
      LLMLOGE(GRAPH_FAILED, "[New][Buffer] allocate string raw data buff failed");
      return GRAPH_FAILED;
    }
    StringHead *const string_head = llm::PtrToPtr<char_t, StringHead>(buff.get());
    // Front 8 bytes store pointer of string
    char_t *const raw_data = llm::PtrToPtr<void, char_t>(
        llm::ValueToPtr(llm::PtrToValue(llm::PtrToPtr<char_t, void>(buff.get())) + sizeof(*string_head)));
    string_head->addr = static_cast<int64_t>(sizeof(StringHead));
    string_head->len = static_cast<int64_t>(data.size());
    const int32_t memcpy_ret = memcpy_s(raw_data, total_size - sizeof(StringHead),  data.c_str(), data.size() + 1U);
    if (memcpy_ret != EOK) {
      REPORT_INNER_ERR_MSG("E18888", "memcpy data failed, ret:%d, size:%zu.", memcpy_ret, data.size() + 1U);
      LLMLOGE(GRAPH_FAILED, "[Copy][Data] failed, ret:%d", memcpy_ret);
      return GRAPH_FAILED;
    }
    (void)SetData(llm::PtrToPtr<char_t, const uint8_t>(buff.get()), total_size);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

graphStatus TensorImpl::SetData(const std::vector<std::string> &data) {
  if (data.empty()) {
    REPORT_INNER_ERR_MSG("E18888", "there is no data, please check the input variable");
    LLMLOGE(GRAPH_FAILED, "[Check][Param] there is no data, please check the input variable");
    return GRAPH_FAILED;
  }
  size_t total_size = 0U;
  total_size = std::accumulate(data.begin(), data.end(), total_size, [](size_t total, const std::string& str) {
    /// Extra 16 bytes store string head
    /// Extra 1 byte store '\0'
    total += str.size() + sizeof(StringHead) + 1U;
    return total;
  });

  const std::unique_ptr<char_t[]> buff = ComGraphMakeUnique<char_t[]>(total_size);
  if (buff == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "allocate string raw data buff failed, size:%zu", total_size);
    LLMLOGE(GRAPH_FAILED, "[New][Buffer] allocate string raw data buff failed");
    return GRAPH_FAILED;
  }
  // Front some bytes store head of each string
  StringHead * const string_head = llm::PtrToPtr<char_t, StringHead>(buff.get());
  uint64_t raw_data = llm::PtrToValue(static_cast<void *>(buff.get())) + (data.size() * sizeof(*string_head));
  uint64_t ptr_size = data.size() * sizeof(StringHead);
  for (size_t i = 0U; i < data.size(); ++i) {
    llm::PtrAdd<StringHead>(string_head, data.size(), i)->addr = static_cast<int64_t>(ptr_size);
    llm::PtrAdd<StringHead>(string_head, data.size(), i)->len = static_cast<int64_t>(data[i].size());
    if (total_size < ptr_size) {
      REPORT_INNER_ERR_MSG("E18888", "Subtraction invalid, total_size:%zu, ptr_size:%" PRIu64, total_size, ptr_size);
      LLMLOGE(GRAPH_FAILED, "[Check][Param] Subtraction invalid, total_size: %zu, ptr_size: %" PRIu64,
              total_size, ptr_size);
      return GRAPH_FAILED;
    }
    const int32_t memcpy_ret = memcpy_s(llm::ValueToPtr(raw_data), total_size - ptr_size,
                                        data[i].c_str(), data[i].size() + 1U);
    LLM_CHK_BOOL_RET_STATUS(memcpy_ret == EOK, GRAPH_FAILED, "copy data failed");
    raw_data += (data[i].size() + 1U);
    ptr_size += (data[i].size() + 1U);
  }

  (void)SetData(llm::PtrToPtr<char_t, const uint8_t>(buff.get()), total_size);
  return GRAPH_SUCCESS;
}

graphStatus TensorImpl::SetData(uint8_t *data, size_t size, const Tensor::DeleteFunc &deleter_func) {
  return SetData(data, size);
}

Tensor::Tensor() {
  impl = ComGraphMakeSharedAndThrow<TensorImpl>();
}

Tensor::Tensor(const TensorDesc &tensor_desc) {
  impl = ComGraphMakeSharedAndThrow<TensorImpl>(tensor_desc);
}

Tensor::Tensor(const TensorDesc &tensor_desc, const uint8_t *data, size_t size) {
  impl = ComGraphMakeShared<TensorImpl>(tensor_desc, data, size);
}

graphStatus Tensor::SetTensorDesc(const TensorDesc &tensor_desc) {
  if (impl != nullptr) {
    impl->SetTensorDesc(tensor_desc);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

TensorDesc Tensor::GetTensorDesc() const {
  if (impl != nullptr) {
    return impl->GetTensorDesc();
  }
  return TensorDesc();
}

const uint8_t *Tensor::GetData() const {
  if (impl != nullptr) {
    return impl->GetData();
  }
  return nullptr;
}

uint8_t *Tensor::GetData() {
  if (impl != nullptr) {
    return impl->GetData();
  }
  return nullptr;
}

size_t Tensor::GetSize() const {
  if (impl != nullptr) {
    return impl->GetSize();
  }
  return 0U;
}

graphStatus Tensor::SetData(std::vector<uint8_t> &&data) {
  if (impl != nullptr) {
    (void)impl->SetData(data);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

graphStatus Tensor::SetData(const std::vector<uint8_t> &data) {
  if (impl != nullptr) {
    (void)impl->SetData(data);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

graphStatus Tensor::SetData(const uint8_t *data, size_t size) {
  if (impl != nullptr) {
    (void)impl->SetData(data, size);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

graphStatus Tensor::SetData(const std::vector<std::string> &data) {
  if (impl != nullptr) {
    if (impl->SetData(data) != GRAPH_SUCCESS) {
      LLMLOGE(GRAPH_FAILED, "[Call][SetData] Tensor set vector data failed.");
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

graphStatus Tensor::SetData(const char_t *data) {
  if ((impl != nullptr) && (data != nullptr)) {
    const std::string tensor_data = data;
    if (impl->SetData(tensor_data) != GRAPH_SUCCESS) {
      LLMLOGE(GRAPH_FAILED, "[Call][SetData] Tensor set data(%s) failed.", data);
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

graphStatus Tensor::SetData(const std::vector<AscendString> &datas) {
  if (impl != nullptr) {
    std::vector<std::string> tensor_data;
    for (auto &data : datas) {
      if (data.GetString() == nullptr) {
        REPORT_INNER_ERR_MSG("E18888", "Data is nullptr. check invalid");
        LLMLOGE(GRAPH_FAILED, "[Check][Param] Data is nullptr.");
        return GRAPH_FAILED;
      }
      tensor_data.emplace_back(data.GetString());
    }
    if (impl->SetData(tensor_data) != GRAPH_SUCCESS) {
      LLMLOGE(GRAPH_FAILED, "[Call][SetData] Tensor set vector data failed.");
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

graphStatus Tensor::SetData(uint8_t *data, size_t size, const Tensor::DeleteFunc &deleter_func) {
  if (impl != nullptr) {
    if (impl->SetData(data, size, deleter_func) != GRAPH_SUCCESS) {
      LLMLOGE(GRAPH_FAILED, "[Call][SetData] Tensor set data with deleter function failed");
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}


Tensor Tensor::Clone() const {
  if (impl != nullptr) {
    const Tensor tensor(impl->GetTensorDesc(), impl->GetData(), impl->GetSize());
    return tensor;
  }
  const Tensor tensor2;
  return tensor2;
}

ge::DataType Tensor::GetDataType() const {
  if (impl != nullptr) {
    return impl->GetTensorDesc().GetDataType();
  }
  return ge::DT_UNDEFINED;
}

// mock Tensor end

}  // namespace ge
