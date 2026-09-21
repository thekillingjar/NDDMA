/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRAPH_BUFFER_IMPL_H_
#define GRAPH_BUFFER_IMPL_H_

#include <string>
#include "proto/ge_ir.pb.h"
#include "graph/detail/attributes_holder.h"

namespace ge {
class BufferImpl {
 public:
  BufferImpl();
  ~BufferImpl();
  BufferImpl(const BufferImpl &other);
  BufferImpl(const std::size_t buffer_size, const std::uint8_t default_val);

  void CopyFrom(const std::uint8_t *const data, const std::size_t buffer_size);
  BufferImpl(const std::shared_ptr<google::protobuf::Message> &proto_owner, proto::AttrDef *const buffer);
  BufferImpl(const std::shared_ptr<google::protobuf::Message> &proto_owner, std::string *const buffer);

  BufferImpl &operator=(const BufferImpl &other);
  const std::uint8_t *GetData() const;
  std::uint8_t *GetData();
  std::size_t GetSize() const;
  void ClearBuffer();
  uint8_t operator[](const size_t index) const;

 private:
  friend class GeAttrValueImp;
  GeIrProtoHelper<proto::AttrDef> data_;
  std::string *buffer_ = nullptr;
};
}  // namespace ge
#endif  // GRAPH_BUFFER_IMPL_H_
