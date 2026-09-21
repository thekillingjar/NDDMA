/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bg_identity.h"

#include "common/checker.h"

namespace gert {
namespace bg {
ValueHolderPtr IdentityShape(const ValueHolderPtr &shape) {
  return bg::ValueHolder::CreateSingleDataOutput("IdentityShape", {shape});
}
std::vector<ValueHolderPtr> IdentityShape(const std::vector<ValueHolderPtr> &shapes) {
  return bg::ValueHolder::CreateDataOutput("IdentityShape", shapes, shapes.size());
}

ValueHolderPtr IdentityAddr(const ValueHolderPtr &addr) {
  auto ret = bg::ValueHolder::CreateSingleDataOutput("IdentityAddr", {addr});
  GE_ASSERT_NOTNULL(ret);
  ret->SetPlacement(addr->GetPlacement());
  return ret;
}
std::vector<ValueHolderPtr> IdentityAddr(const std::vector<ValueHolderPtr> &addrs) {
  auto ret = bg::ValueHolder::CreateDataOutput("IdentityAddr", addrs, addrs.size());
  GE_ASSERT_EQ(ret.size(), addrs.size());
  for (size_t i = 0u; i < addrs.size(); ++i) {
    GE_ASSERT_NOTNULL(ret[i]);
    ret[i]->SetPlacement(addrs[i]->GetPlacement());
  }
  return ret;
}

std::vector<ValueHolderPtr> IdentityShapeAndAddr(const std::vector<ValueHolderPtr> &shapes,
                                                 const std::vector<ValueHolderPtr> &addrs) {
  GE_ASSERT_EQ(shapes.size(), addrs.size());
  std::vector<ValueHolderPtr> shape_and_addrs = shapes;
  shape_and_addrs.insert(shape_and_addrs.end(), addrs.begin(), addrs.end());
  return bg::ValueHolder::CreateDataOutput("IdentityShapeAndAddr", shape_and_addrs, shape_and_addrs.size());
}

std::vector<DevMemValueHolderPtr> IdentityAddr(const std::vector<DevMemValueHolderPtr> &addrs,
                                               const int64_t stream_id) {
  std::vector<ValueHolderPtr> in_addrs(addrs.cbegin(), addrs.cend());
  auto ret = bg::DevMemValueHolder::CreateDataOutput("IdentityAddr", in_addrs, addrs.size(), stream_id);
  GE_ASSERT_EQ(ret.size(), addrs.size());
  for (size_t i = 0U; i < addrs.size(); ++i) {
    GE_ASSERT_NOTNULL(ret[i]);
    ret[i]->SetPlacement(addrs[i]->GetPlacement());
  }
  return ret;
}

std::vector<DevMemValueHolderPtr> IdentityShapeAndAddr(const std::vector<ValueHolderPtr> &shapes,
                                                       const std::vector<DevMemValueHolderPtr> &addrs,
                                                       const int64_t stream_id) {
  GE_ASSERT_EQ(shapes.size(), addrs.size());
  std::vector<ValueHolderPtr> shape_and_addrs = shapes;
  shape_and_addrs.insert(shape_and_addrs.end(), addrs.begin(), addrs.end());
  return bg::DevMemValueHolder::CreateDataOutput("IdentityShapeAndAddr", shape_and_addrs, shape_and_addrs.size(),
                                                 stream_id);
}
}  // namespace bg
}  // namespace gert
