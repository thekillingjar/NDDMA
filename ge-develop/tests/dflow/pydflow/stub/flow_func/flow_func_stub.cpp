/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <cstring>
#include <map>
#include <limits>
#include "attr_value.h"
#include "flow_func_log.h"
#include "tensor_data_type.h"
#include "flow_msg.h"
#include "meta_params.h"
#include "udf_log.h"
#include "meta_run_context.h"
namespace FlowFunc {
namespace {
FlowFuncLogger g_runLogger(FlowFuncLogType::RUN_LOG);
FlowFuncLogger g_debugLogger(FlowFuncLogType::DEBUG_LOG);
constexpr size_t kMaxUserDataSize = 64U;
constexpr int32_t kDataTypeSizeBitOffset = 1000;
static std::string kHeader = "[tid:100][USER]";
static std::string kStubUserData = "StubUserData";
static int32_t kCurrentExpCode = -1;
static uint64_t kCurrentUserContextId = 0;
int32_t GetSizeByDataType(TensorDataType data_type) {
  static const std::map<TensorDataType, int32_t> size_map = {{TensorDataType::DT_FLOAT, 4},
                                                            {TensorDataType::DT_FLOAT16, 2},
                                                            {TensorDataType::DT_INT8, 1},
                                                            {TensorDataType::DT_INT16, 2},
                                                            {TensorDataType::DT_UINT16, 2},
                                                            {TensorDataType::DT_UINT8, 1},
                                                            {TensorDataType::DT_INT32, 4},
                                                            {TensorDataType::DT_INT64, 8},
                                                            {TensorDataType::DT_UINT32, 4},
                                                            {TensorDataType::DT_UINT64, 8},
                                                            {TensorDataType::DT_BOOL, 1},
                                                            {TensorDataType::DT_DOUBLE, 8},
                                                            {TensorDataType::DT_QINT8, 1},
                                                            {TensorDataType::DT_QINT16, 2},
                                                            {TensorDataType::DT_QINT32, 4},
                                                            {TensorDataType::DT_QUINT8, 1},
                                                            {TensorDataType::DT_QUINT16, 2},
                                                            {TensorDataType::DT_DUAL, 5},
                                                            {TensorDataType::DT_INT4, kDataTypeSizeBitOffset + 4},
                                                            {TensorDataType::DT_UINT1, kDataTypeSizeBitOffset + 1},
                                                            {TensorDataType::DT_INT2, kDataTypeSizeBitOffset + 2},
                                                            {TensorDataType::DT_UINT2, kDataTypeSizeBitOffset + 2}};
  const auto iter = size_map.find(data_type);
  if (iter == size_map.cend()) {
    return -1;
  }
  return iter->second;
}

bool CheckMultiplyOverflowInt64(const int64_t &a, const int64_t &b) {
  if (a > 0) {
    if (b > 0) {
      if (a > (std::numeric_limits<int64_t>::max() / b)) {
        return true;
      }
    } else {
      if (b < (std::numeric_limits<int64_t>::min() / a)) {
        return true;
      }
    }
  } else {
    if (b > 0) {
      if (a < (std::numeric_limits<int64_t>::min() / b)) {
        return true;
      }
    } else {
      if ((a != 0) && (b < (std::numeric_limits<int64_t>::max() / a))) {
        return true;
      }
    }
  }
  return false;
}
int64_t CalcElementCnt(const std::vector<int64_t> &shape) {
  int64_t element_cnt = 1;
  for (int64_t dim : shape) {
    if (dim < 0) {
      UDF_LOG_ERROR("dim is negative, not support now, dim=%ld.", dim);
      return -1;
    }
    if (CheckMultiplyOverflowInt64(element_cnt, dim)) {
      UDF_LOG_ERROR("CalcElementCnt failed, when multiplying %ld and %ld.", element_cnt, dim);
      return -1;
    }
    element_cnt *= dim;
  }
  return element_cnt;
}

int64_t CalcDataSize(const std::vector<int64_t> &shape, TensorDataType data_type) {
  int32_t type_size = GetSizeByDataType(data_type);
  if (type_size < 0) {
    UDF_LOG_ERROR("data_type=%d is not support.", static_cast<int32_t>(data_type));
    return -1;
  }
  int64_t element_cnt = CalcElementCnt(shape);
  if (element_cnt < 0) {
    UDF_LOG_ERROR("CalcElementCnt failed, result=%ld.", element_cnt);
    return -1;
  }

  if (type_size > kDataTypeSizeBitOffset) {
    int32_t type_bit_size = type_size - kDataTypeSizeBitOffset;
    if (CheckMultiplyOverflowInt64(element_cnt, static_cast<int64_t>(type_bit_size))) {
      UDF_LOG_ERROR("CalcDataSize failed, when multiplying %ld and %d.", element_cnt, type_bit_size);
      return -1;
    }
    int64_t data_bit_size = element_cnt * type_bit_size;
    constexpr int64_t byte_bit_size = 8;
    int64_t data_size = data_bit_size / byte_bit_size;
    if (data_bit_size % byte_bit_size != 0) {
      data_size += 1;
    }
    return data_size;
  } else {
    if (CheckMultiplyOverflowInt64(element_cnt, static_cast<int64_t>(type_size))) {
      UDF_LOG_ERROR("CalcDataSize failed, when multiplying %ld and %d.", element_cnt, type_size);
      return -1;
    }
    return element_cnt * type_size;
  }
}

}  // namespace

FlowFuncLogger &FlowFuncLogger::GetLogger(FlowFuncLogType type) {
  if (type == FlowFuncLogType::RUN_LOG) {
    return g_runLogger;
  } else {
    return g_debugLogger;
  }
}

const char *FlowFuncLogger::GetLogExtHeader() {
  return kHeader.c_str();
}

Tensor::Tensor() {
  data_size_ = 40UL;
  uint8_t *data = new uint8_t[data_size_];
  data_ = static_cast<void *>(data);
}

Tensor::Tensor(const std::vector<int64_t> &shape, const TensorDataType &data_type) {
  int64_t data_size = CalcDataSize(shape, data_type);
  if (data_size < 0) {
    data_ = nullptr;
    return;
  }
  data_size_ = static_cast<uint64_t>(data_size);
  uint8_t *data = new uint8_t[data_size_];
  data_ = static_cast<void *>(data);
  stub_shape_ = shape;
  stub_data_type_ = data_type;
}

int64_t Tensor::GetElementCnt() const {
  int64_t element_cnt = CalcElementCnt(stub_shape_);
  if (element_cnt < 0) {
    UDF_LOG_ERROR("CalcElementCnt failed, result=%ld.", element_cnt);
    return -1;
  }
  return element_cnt;
}

int32_t Tensor::Reshape(const std::vector<int64_t> &shape) {
  int64_t originElementCnt = CalcElementCnt(stub_shape_);
  int64_t element_cnt = CalcElementCnt(shape);
  if (originElementCnt != element_cnt) {
    UDF_LOG_ERROR("reshape element count is not same, originElementCnt=%ld, element_cnt=%ld.", originElementCnt,
                  element_cnt);
    return FLOW_FUNC_ERR_PARAM_INVALID;
  }
  stub_shape_ = shape;
  return FLOW_FUNC_SUCCESS;
}

Tensor::~Tensor() {
  if (data_ != nullptr) {
    uint8_t *tmpData = static_cast<uint8_t *>(data_);
    delete[] tmpData;
    data_ = nullptr;
    data_size_ = 0UL;
  }
}

MetaParams::MetaParams() {
  attr_map_["AttrStub"] = std::make_shared<AttrValue>();
}

std::shared_ptr<const AttrValue> MetaParams::GetAttr(const char *attr_name) const {
  if (attr_name == nullptr) {
    UDF_LOG_INFO("attr_name is null, instance name[%s].", name_.c_str());
    return nullptr;
  }
  UDF_LOG_DEBUG("get attr, instance name[%s], attr_name=%s.", name_.c_str(), attr_name);
  const auto attr_iter = attr_map_.find(attr_name);
  if (attr_iter == attr_map_.cend()) {
    UDF_LOG_INFO("no attr found, instance name[%s], attr_name=%s.", name_.c_str(), attr_name);
    return nullptr;
  }
  return attr_iter->second;
}

int32_t MetaRunContext::GetUserData(void *data, size_t size, size_t offset) const {
  if (data == nullptr) {
    UDF_LOG_ERROR("The data is nullptr.");
    return FLOW_FUNC_ERR_PARAM_INVALID;
  }
  if (size == 0U) {
    UDF_LOG_ERROR("The size is 0, should in (0, 64].");
    return FLOW_FUNC_ERR_PARAM_INVALID;
  }
  if ((offset >= kMaxUserDataSize) || ((kMaxUserDataSize - offset) < size)) {
    UDF_LOG_ERROR("The size + offset need <= %zu, but size = %zu, offset = %zu.", kMaxUserDataSize, size, offset);
    return FLOW_FUNC_ERR_PARAM_INVALID;
  }
  if (size < kStubUserData.length() + 1) {
    UDF_LOG_ERROR("Failed to get user data, size[%zu], offset[%zu].", size, offset);
    return FLOW_FUNC_FAILED;
  }
  memcpy(data, kStubUserData.c_str(), kStubUserData.length());
  UDF_LOG_DEBUG("Success to get user data, size[%zu], offset[%zu].", size, offset);
  return FLOW_FUNC_SUCCESS;
}

void MetaRunContext::RaiseException(int32_t exp_code, uint64_t user_context_id) {
  kCurrentExpCode = exp_code;
  kCurrentUserContextId = user_context_id;
}

bool MetaRunContext::GetException(int32_t &exp_code, uint64_t &user_context_id) {
  if (kCurrentExpCode == -1) {
    return false;
  }
  exp_code = kCurrentExpCode;
  user_context_id = kCurrentUserContextId;
  return true;
}
}  // namespace FlowFunc
