/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stack>

#include "graph/utils/tensor_value_utils.h"

namespace ge {
/**
 * @brief 将 FP16 (uint16_t) 转换为 float
 * @param fp16_val FP16 的位模式 (uint16_t)
 * @return 转换后的 float 值
 * @note 参考 compiler/engines/nn_engine/utils/common/math_util.cc 的实现
 * 当前不能反向依赖，所以自己实现了一份
 */
inline float Fp16ToFloat(uint16_t fp16_val) {
  // 提取符号位、指数位、尾数位
  uint16_t sign = (fp16_val >> 15) & 0x1U;
  int16_t exp = (fp16_val >> 10) & 0x1FU;
  uint16_t man = ((fp16_val >> 0) & 0x3FFU) | ((((fp16_val >> 10) & 0x1FU) > 0 ? 1U : 0U) * 0x400U);

  // 处理无穷大和 NaN（exp == 0x1F）
  if (exp == 0x1F) {
    Fp32Bits result_bits{};
    result_bits.u =
        (static_cast<uint32_t>(sign) << kFp32Fraction) | (0xFFU << kFp32FractionMove) | ((static_cast<uint32_t>(man) & kFp16ManMask) << kFp16FractionMove);
    return result_bits.f;
  }

  // 对于非规格化数，将指数设为1（为规范化做准备）
  if (exp == 0) {
    exp = 1;
  }

  while (man != 0 && (man & kFp16ManHideBit) == 0) {
    man <<= 1;
    exp--;
  }

  auto fp32_sign = static_cast<uint32_t>(sign);
  uint32_t fp32_exp;
  uint32_t fp32_man;

  if (man == 0) {
    fp32_exp = 0;
    fp32_man = 0;
  } else {
    // 转换为 FP32 格式
    fp32_exp = static_cast<uint32_t>(exp - kFp16ExpBias + kFp32ExpBias);
    fp32_man = (static_cast<uint32_t>(man) & kFp16ManMask) << kFp16FractionMove;
  }

  Fp32Bits fp32_bits{};
  fp32_bits.u = (fp32_sign << kFp32Fraction) | (fp32_exp << kFp32FractionMove) | (fp32_man & 0x7FFFFFU);
  return fp32_bits.f;
}

template <typename T>
inline typename std::enable_if<std::is_same<T, bool>::value, std::string>::type TensorElementToString(T value)
{
  return value ? "true" : "false";
}

template <typename T>
inline typename std::enable_if<!std::is_same<T, bool>::value, std::string>::type TensorElementToString(T value)
{
  return std::to_string(value);
}

/**
 * @brief 通用的 tensor 值转换实现，支持自定义转换函数
 * @tparam T 原始数据类型
 * @tparam ConvertFunc 转换函数类型，将 T 转换为可打印类型
 * @param tensor tensor 对象
 * @param sep 分隔符
 * @param convert_func 转换函数，将 T 类型转换为可打印类型（如 float）
 * @return 转换后的字符串
 */
template <typename T, typename ConvertFunc>
std::string ConvertTensorValueImplWithConverterSkipped(const Tensor &tensor, const std::string &sep,
                                                 ConvertFunc convert_func) {
  const auto shape = tensor.GetTensorDesc().GetShape();
  const auto data_cnt = shape.GetShapeSize();
  const auto data_begin = reinterpret_cast<const T *>(tensor.GetData());

  std::stringstream tensor_value_ss;

  if (tensor.GetSize() == 0) {
    tensor_value_ss << "<empty>";
    return tensor_value_ss.str();
  }

  tensor_value_ss << "[";
  if (data_cnt == 0 || data_cnt == 1) {
    auto converted_val = convert_func(*data_begin);
    tensor_value_ss << TensorElementToString(converted_val);
  } else {
    int32_t count = 0;
    std::stringstream first_three_ss;
    auto first_converted_val = convert_func(*data_begin);
    first_three_ss << TensorElementToString(first_converted_val);
    std::stringstream last_three_ss;
    for (auto data = std::next(data_begin); data != data_begin + data_cnt; ++data) {
      auto converted_val = convert_func(*data);
      const std::string data_str = TensorElementToString(converted_val);
      if (count < kAttrTensorShowNumHalf - 1) {
        first_three_ss << sep << data_str;
      } else if (count >= data_cnt - 1 - kAttrTensorShowNumHalf) {
        last_three_ss << sep << data_str;
      }
      ++count;
    }

    tensor_value_ss << first_three_ss.str();
    if (count >= kAttrTensorShowNum) {
      tensor_value_ss << sep << "...";
    }
    tensor_value_ss << last_three_ss.str();
  }

  tensor_value_ss << "]";
  return tensor_value_ss.str();
}

/**
 * @brief 通用的 tensor 值转换实现，支持自定义转换函数
 * @tparam T 原始数据类型
 * @tparam ConvertFunc 转换函数类型，将 T 转换为可打印类型
 * @param tensor tensor 对象
 * @param sep 分隔符
 * @param convert_func 转换函数，将 T 类型转换为可打印类型（如 float）
 * @return 转换后的字符串
 */
template <typename T, typename ConvertFunc>
std::string ConvertTensorValueImplWithConverterNoSkip(const Tensor &tensor, const std::string &sep,
                                                 ConvertFunc convert_func) {
  const auto shape = tensor.GetTensorDesc().GetShape();
  const auto data_cnt = shape.GetShapeSize();
  const auto data_begin = reinterpret_cast<const T *>(tensor.GetData());

  std::stringstream tensor_value_ss;

  if (tensor.GetSize() == 0) {
    tensor_value_ss << "<empty>";
    return tensor_value_ss.str();
  }

  tensor_value_ss << "[";
  auto first_converted_val = convert_func(*data_begin);
  tensor_value_ss << TensorElementToString(first_converted_val);
  for (auto data = std::next(data_begin); data != data_begin + data_cnt; ++data) {
    auto converted_val = convert_func(*data);
    const std::string data_str = TensorElementToString(converted_val);
    tensor_value_ss << sep << data_str;
  }
  tensor_value_ss << "]";
  return tensor_value_ss.str();
}

/**
 * @brief 通用的 tensor 值转换实现，支持自定义转换函数
 * @tparam T 原始数据类型
 * @tparam ConvertFunc 转换函数类型，将 T 转换为可打印类型
 * @param tensor tensor 对象
 * @param sep 分隔符
 * @param convert_func 转换函数，将 T 类型转换为可打印类型（如 float）
 * @param is_mid_skip 是否省略中间数据
 * @return 转换后的字符串
 */
template <typename T, typename ConvertFunc>
std::string ConvertTensorValueImplWithConverter(const Tensor &tensor, const std::string &sep,
                                                ConvertFunc convert_func, 
                                                bool is_mid_skipped) {
  if (is_mid_skipped) {
    return ConvertTensorValueImplWithConverterSkipped<T>(tensor, sep, convert_func);
  } else {
    return ConvertTensorValueImplWithConverterNoSkip<T>(tensor, sep, convert_func);
  }
}

/**
 * @brief 标准类型的 tensor 值转换实现（直接使用类型 T）
 */
template <typename T>
std::string ConvertTensorValueImpl(const Tensor &tensor, const std::string &sep, bool is_mid_skipped) {
  // 使用恒等转换函数
  auto identity = [](const T &val) -> T { return val; };
  return ConvertTensorValueImplWithConverter<T>(tensor, sep, identity, is_mid_skipped);
}

namespace{
/**
 * @brief 专门处理 FP16 类型的 tensor 值转换
 * @param tensor tensor 对象
 * @param sep 分隔符
 * @return 转换后的字符串
 */
std::string ConvertTensorValueFp16(const Tensor &tensor, const std::string &sep, bool is_mid_skipped) {
  // 使用 FP16 到 float 的转换函数
  return ConvertTensorValueImplWithConverter<uint16_t>(tensor, sep, Fp16ToFloat, is_mid_skipped);
}
} // namespace

std::string TensorValueUtils::ConvertTensorValue(const Tensor &tensor, DataType value_type, const std::string &sep, const bool is_mid_skipped) {
  switch (value_type) {
    case DT_FLOAT:
      return ge::ConvertTensorValueImpl<float>(tensor, sep, is_mid_skipped);
    case DT_INT8:
      return ge::ConvertTensorValueImpl<int8_t>(tensor, sep, is_mid_skipped);
    case DT_INT16:
      return ge::ConvertTensorValueImpl<int16_t>(tensor, sep, is_mid_skipped);
    case DT_INT32:
      return ge::ConvertTensorValueImpl<int32_t>(tensor, sep, is_mid_skipped);
    case DT_INT64:
      return ge::ConvertTensorValueImpl<int64_t>(tensor, sep, is_mid_skipped);
    case DT_UINT8:
      return ge::ConvertTensorValueImpl<uint8_t>(tensor, sep, is_mid_skipped);
    case DT_UINT16:
      return ge::ConvertTensorValueImpl<uint16_t>(tensor, sep, is_mid_skipped);
    case DT_FLOAT16:
      return ge::ConvertTensorValueFp16(tensor, sep, is_mid_skipped);
    case DT_UINT32:
      return ge::ConvertTensorValueImpl<uint32_t>(tensor, sep, is_mid_skipped);
    case DT_UINT64:
      return ge::ConvertTensorValueImpl<uint64_t>(tensor, sep, is_mid_skipped);
    case DT_BOOL:
      return ge::ConvertTensorValueImpl<bool>(tensor, sep, is_mid_skipped);
    default:
      GELOGW("[Create][EsCTensor] unsupported data type %s",
             ge::TypeUtils::DataTypeToAscendString(static_cast<ge::DataType>(value_type)).GetString());
      return "<not_supported>";
  }
}
}  // namespace ge