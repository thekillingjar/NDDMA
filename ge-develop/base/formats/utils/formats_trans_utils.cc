/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "formats/utils/formats_trans_utils.h"

#include <cstdint>

#include "formats/utils/formats_definitions.h"
#include "framework/common/debug/log.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace formats {
int64_t GetCubeSizeByDataType(const DataType data_type) {
  // Current cube does not support 4 bytes and longer data
  const auto size = GetSizeByDataType(data_type);
  if (size <= 0) {
    const std::string error = "Failed to get cube size, the data type " +
        FmtToStr(TypeUtils::DataTypeToSerialString(data_type)) + " is invalid";
    GE_WARNINGLOG_AND_ERRORMSG(error.c_str());
    return -1;
  } else if (size == 1) {
    return kCubeSize * 2;  // 32 bytes cube size
  } else {
    return kCubeSize;
  }
}

std::string ShapeToString(const GeShape &shape) {
  return ShapeToString(shape.GetDims());
}

std::string GertShapeToString(const gert::Shape &shape) {
  std::vector<int64_t> shape_vec(shape.GetDimNum());
  for (size_t i = 0u; i < shape.GetDimNum(); ++i) {
    shape_vec[i] = shape.GetDim(i);
  }
  return JoinToString(shape_vec);
}

std::string ShapeToString(const std::vector<int64_t> &shape) {
  return JoinToString(shape);
}

std::string RangeToString(const std::vector<std::pair<int64_t, int64_t>> &ranges) {
  bool first = true;
  std::stringstream ss;
  ss << "[";
  for (const auto &range : ranges) {
    if (first) {
      first = false;
    } else {
      ss << ",";
    }
    ss << "{";
    ss << range.first << "," << range.second;
    ss << "}";
  }
  ss << "]";
  return ss.str();
}

int64_t GetItemNumByShape(const std::vector<int64_t> &shape) {
  int64_t num = 1;
  for (const auto dim : shape) {
    num *= dim;
  }
  return num;
}

bool CheckShapeValid(const std::vector<int64_t> &shape, const int64_t expect_dims) {
  if ((expect_dims <= 0) || (shape.size() != static_cast<size_t>(expect_dims))) {
    const std::string error = "Invalid shape, dims num " + FmtToStr(shape.size()) +
        ", expect " + FmtToStr(expect_dims);
    GE_WARNINGLOG_AND_ERRORMSG(error.c_str());
    return false;
  }
  return IsShapeValid(shape);
}

bool IsShapeValid(const std::vector<int64_t> &shape) {
  if (shape.empty()) {
    return false;
  }
  int64_t num = 1;
  for (const auto dim : shape) {
    if (dim < 0) {
      const std::string error = "Invalid negative dims in the shape " +  FmtToStr(ShapeToString(shape));
      GE_WARNINGLOG_AND_ERRORMSG(error.c_str());
      return false;
    }
    if ((dim != 0) && ((static_cast<int64_t>(kShapeItemNumMAX) / dim) < num)) {
      const std::string error = "Shape overflow, the total count should be less than " + FmtToStr(kShapeItemNumMAX);
      GE_WARNINGLOG_AND_ERRORMSG(error.c_str());
      return false;
    }
    num *= dim;
  }
  return true;
}

bool IsShapeEqual(const GeShape &src, const GeShape &dst) {
  if (src.GetDims().size() != dst.GetDims().size()) {
    return false;
  }

  for (size_t i = 0UL; i < src.GetDims().size(); ++i) {
    if (src.GetDim(i) != dst.GetDim(i)) {
      return false;
    }
  }
  return true;
}

bool IsTransShapeSrcCorrect(const TransArgs &args, const std::vector<int64_t> &expect_shape) {
  if (args.src_shape != expect_shape) {
    const std::string error = "Failed to trans format from" +
        FmtToStr(TypeUtils::FormatToSerialString(args.src_format)) + " to " +
        FmtToStr(TypeUtils::FormatToSerialString(args.dst_format)) + ", invalid relationship between src shape " +
        FmtToStr(ShapeToString(args.src_shape)) + " and dst " +
        FmtToStr(ShapeToString(args.dst_shape));
    GE_WARNINGLOG_AND_ERRORMSG(error.c_str());
    return false;
  }
  return true;
}

bool IsTransShapeDstCorrect(const TransArgs &args, const std::vector<int64_t> &expect_shape) {
  if ((!args.dst_shape.empty()) && (args.dst_shape != expect_shape)) {
    const std::string error = "Failed to trans format from " +
        FmtToStr(TypeUtils::FormatToSerialString(args.src_format)) + " to " +
        FmtToStr(TypeUtils::FormatToSerialString(args.dst_format)) + ", the dst shape" +
        FmtToStr(ShapeToString(args.dst_shape)) + " is invalid, expect" +
        FmtToStr(ShapeToString(expect_shape));
    GE_WARNINGLOG_AND_ERRORMSG(error.c_str());
    return false;
  }
  return true;
}
}  // namespace formats
}  // namespace ge
