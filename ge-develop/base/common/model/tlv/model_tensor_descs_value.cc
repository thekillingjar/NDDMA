/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "model_tensor_descs_value.h"
#include "common/checker.h"
namespace ge {
size_t ModelTensorDesc::Size() {
  const size_t size = static_cast<size_t>(sizeof(base_info) + base_info.name_len + base_info.dims_len +
                      base_info.dimsV2_len + base_info.shape_range_len);
  return size;
}

bool ModelTensorDesc::Serilize(uint8_t ** const addr, size_t &left_size) {
  GE_ASSERT_NOTNULL(addr, "input param is invalid, addr is null.");
  GE_ASSERT_NOTNULL(*addr, "addr ptr is null.");
  errno_t ret;
  GE_ASSERT_EOK(memcpy_s(*addr, left_size, static_cast<const void *>(&base_info), sizeof(ModelTensorDescBaseInfo)));
  *addr = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(*addr) + sizeof(ModelTensorDescBaseInfo)));
  left_size -= sizeof(ModelTensorDescBaseInfo);
  if ((name.data() != nullptr) && (base_info.name_len != 0U)) {
    ret = memcpy_s(*addr, left_size, static_cast<const void *>(name.data()), static_cast<size_t>(base_info.name_len));
    GE_ASSERT_EOK(ret, "serilize ModelTensorDesc::name failed");
    *addr = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(*addr) + base_info.name_len));
    left_size -= base_info.name_len;
  }
  
  if ((dims.data() != nullptr) && (base_info.dims_len != 0U)) {
    ret = memcpy_s(*addr, left_size, static_cast<void *>(dims.data()), static_cast<size_t>(base_info.dims_len));
    GE_ASSERT_EOK(ret, "serilize ModelTensorDesc::dims failed");
    *addr = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(*addr) + base_info.dims_len));
    left_size -= base_info.dims_len;
  }

  if ((dimsV2.data() != nullptr) && (base_info.dimsV2_len != 0U)) {
    ret = memcpy_s(*addr, left_size, static_cast<void *>(dimsV2.data()),
                   static_cast<size_t>(base_info.dimsV2_len));
    GE_ASSERT_EOK(ret, "serilize ModelTensorDesc::dimsVe failed");
    *addr = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(*addr) + base_info.dimsV2_len));
    left_size -= base_info.dimsV2_len;
  }

  if ((shape_range.data() != nullptr) && (base_info.shape_range_len != 0U)) {
    ret = memcpy_s(*addr, left_size, static_cast<void *>(shape_range.data()),
                   static_cast<size_t>(base_info.shape_range_len));
    GE_ASSERT_EOK(ret, "serilize ModelTensorDesc::dimsVe failed");
    *addr = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(*addr) + base_info.shape_range_len));
    left_size -= base_info.shape_range_len;
  }
  return true;
}

size_t ModelTensorDescsValue::Size() {
  size_t size = 0U;
  for (auto &desc : descs) {
    size += desc.Size();
  }
  size += sizeof(uint32_t);
  return size;
}

bool ModelTensorDescsValue::Serilize(uint8_t ** const addr, size_t &left_size) {
  GE_ASSERT_NOTNULL(addr, "input param is invalid, addr is null.");
  GE_ASSERT_NOTNULL(*addr, "addr ptr is null.");

  GE_ASSERT_EOK(memcpy_s(*addr, left_size, static_cast<void *>(&tensor_desc_size), sizeof(uint32_t)));
  *addr = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(*addr) + sizeof(uint32_t)));
  left_size -= sizeof(uint32_t);

  for (auto desc : descs) {
    if (!desc.Serilize(addr, left_size)) {
      GELOGE(FAILED, "save ModelTensorDesc failed.");
      return false;
    }
  }
  return true;
}

bool ModelTensorDescsValue::NeedSave() {
  return tensor_desc_size > 0U;
}
}
