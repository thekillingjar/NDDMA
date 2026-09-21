/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRAPH_CACHE_POLICY_COMPILE_CACHE_DESC_H
#define GRAPH_CACHE_POLICY_COMPILE_CACHE_DESC_H

#include <string>
#include <vector>
#include "cache_desc.h"
#include "graph/small_vector.h"
#include "graph/ascend_limits.h"
#include "graph/types.h"
#include "graph/def_types.h"
#include "graph/utils/hash_utils.h"
#include "framework/common/debug/ge_log.h"
#include "common/ge_common/debug/log.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
class CompileCacheDesc;
using CompileCacheDescPtr = std::shared_ptr<CompileCacheDesc>;
class BinaryHolder {
 public:
  BinaryHolder() = default;

  ~BinaryHolder() = default;

  BinaryHolder(const BinaryHolder &other);
  BinaryHolder(BinaryHolder &&other);
  BinaryHolder &operator=(const BinaryHolder &other);
  BinaryHolder &operator=(BinaryHolder &&other);

  BinaryHolder(const uint8_t *const data, const size_t data_len);

  static std::unique_ptr<BinaryHolder> createFrom(std::unique_ptr<uint8_t[]> &&ptr, size_t length);

  const uint8_t *GetDataPtr() const noexcept;

  const size_t &GetDataLen() const noexcept;

  bool operator!=(const BinaryHolder &second) const;

 private:
  std::unique_ptr<uint8_t[]> holder_ = nullptr;
  size_t data_len_ = 0UL;
};

class TensorInfoArgs {
 public:
  TensorInfoArgs(const Format format, const Format origin_format, const DataType data_type)
    : format_(format),
      origin_format_(origin_format),
      data_type_(data_type) {}

  ~TensorInfoArgs() = default;

  bool IsUnknownShape() const;
  bool IsShapeInRange(const TensorInfoArgs &other) const;
  bool IsTensorInfoMatch(const TensorInfoArgs &other) const;
  Format GetFormat() const;
  Format GetOriginFormat() const;
  DataType GetDataType() const;
  void SetShape(const std::vector<int64_t> &shape);
  void SetShape(const SmallVector<int64_t, kDefaultDimsNum> &shape);
  void SetOriginShape(const std::vector<int64_t> &origin_shape);
  void SetOriginShape(const SmallVector<int64_t, kDefaultDimsNum> &origin_shape);
  void SetShapeRange(const std::vector<std::pair<int64_t, int64_t>> &ranges);
  bool operator!=(const TensorInfoArgs &second) const;

 private:
  Format format_;
  Format origin_format_;
  DataType data_type_;
  SmallVector<int64_t, kDefaultMaxInputNum> shape_;
  SmallVector<int64_t, kDefaultMaxInputNum> origin_shape_;
  SmallVector<std::pair<int64_t, int64_t>, kDefaultMaxInputNum> shape_range_;
};

class CompileCacheDesc : public CacheDesc {
  friend class CacheHasher;
 public:
  CompileCacheDesc() = default;
  ~CompileCacheDesc() override = default;
  bool IsEqual(const CacheDescPtr &other) const override;
  bool IsMatch(const CacheDescPtr &other) const override;
  CacheHashKey GetCacheDescHash() const override;
  void SetOpType(const std::string &op_type);
  void AddBinary(const BinaryHolder &holder);
  void AddBinary(BinaryHolder &&holder);
  void AddTensorInfo(const TensorInfoArgs &tensor_info);
  void SetScopeId(const std::initializer_list<uint64_t> scope_id);
  size_t GetTensorInfoSize();
  TensorInfoArgs *MutableTensorInfo(size_t index);

 private:
  bool CheckWithoutTensorInfo(const CompileCacheDesc *first, const CompileCacheDesc *second) const;
  std::string op_type_; // op type
  SmallVector<uint64_t, kDefaultMaxInputNum> scope_id_; // graph_id and session_id
  SmallVector<TensorInfoArgs, kDefaultMaxInputNum> tensor_info_args_vec_; // input tensordescs
  SmallVector<BinaryHolder, kDefaultMaxInputNum> other_desc_; // attrs float float size
};
}  // namespace ge

namespace std {
template<>
struct hash<ge::BinaryHolder> {
  size_t operator()(const ge::BinaryHolder &value) const {
    GE_CHECK_NOTNULL(value.GetDataPtr());
    size_t seed = ge::HashUtils::MultiHash();
    const uint64_t u8_data = ge::PtrToValue(ge::PtrToPtr<const uint8_t, const void>(value.GetDataPtr()));
    for (size_t idx = 0UL; idx < value.GetDataLen(); idx++) {
      seed = ge::HashUtils::HashCombine(seed, *(ge::PtrToPtr<void, uint8_t>(ge::ValueToPtr(u8_data + idx))));
    }
    return seed;
  }
};
}  // namespace std
#endif
