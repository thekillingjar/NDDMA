/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "../utils/api_call_factory.h"
#include "../utils/api_call_utils.h"

namespace gather_base {
using namespace std;
using namespace codegen;
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace ascgen_utils;

std::string CalGatherOuterAxesSize(const std::vector<ascir::AxisId> &param_outer_axes,
                                   const std::vector<ascir::AxisId> &indices_axes, size_t index, const TPipe &tpipe) {
  stringstream ss;
  ss << "(";
  for (size_t i = index + 1; i < param_outer_axes.size(); i++) {
    ss << tpipe.tiler.GetAxis(param_outer_axes[i]).axis_size << " * ";
  }
  for (size_t i = 0; i < indices_axes.size(); i++) {
    ss << tpipe.tiler.GetAxis(indices_axes[i]).axis_size << " * ";
  }
  ss << "1)";
  return ss.str();
}

std::string CalGatherOuterAxesIndex(std::string outer_axis_offset, const std::vector<ascir::AxisId> &param_outer_axes,
                                    const std::vector<ascir::AxisId> &indices_axes, const TPipe &tpipe) {
  stringstream ss;
  for (size_t i = 0; i < param_outer_axes.size(); i++) {
    auto outer_axis = tpipe.tiler.GetAxis(param_outer_axes[i]);
    ss << outer_axis.AsArg() << " = " << outer_axis_offset << " / "
       << CalGatherOuterAxesSize(param_outer_axes, indices_axes, i, tpipe) << " % " << outer_axis.axis_size << ";"
       << std::endl;
  }
  return ss.str();
}

bool IsAxisInParamAxes(ascir::AxisId axis_id, const std::vector<ascir::AxisId> &param_axes, const TPipe &tpipe) {
  for (auto param_axis : param_axes) {
    if (tpipe.tiler.IsFrom(axis_id, param_axis)) {
      return true;
    }
  }
  return false;
}
std::string CalGatherOuterAxisOffset(const std::vector<ascir::AxisId> &current_axis,
                                     const std::vector<ascir::AxisId> &param_inner_axes,
                                     ascir::AxisId tile_inner_axis_id, const TPipe &tpipe) {
  stringstream ss;
  std::vector<ascir::AxisId> gather_outer_axis_id;
  std::vector<ascir::SizeExpr> gather_outer_axis_strides;
  for (auto axis_id : current_axis) {
    if (IsAxisInParamAxes(axis_id, param_inner_axes, tpipe)) {
      continue;
    }
    gather_outer_axis_id.emplace_back(axis_id);
  }
  if (tile_inner_axis_id != ge::kIdNone) {
    gather_outer_axis_id.emplace_back(tile_inner_axis_id);
  }
  ascir::SizeExpr size_product = ge::sym::kSymbolOne;
  for (auto axis_id = gather_outer_axis_id.rbegin(); axis_id != gather_outer_axis_id.rend(); axis_id++) {
    auto axis = tpipe.tiler.GetAxis(*axis_id);
    if (ge::SymbolicUtils::StaticCheckEq(axis.size, ge::sym::kSymbolOne) == ge::TriBool::kTrue) {
      gather_outer_axis_strides.emplace_back(ge::sym::kSymbolZero);
    } else {
      gather_outer_axis_strides.emplace_back(size_product);
      size_product = size_product * axis.size;
    }
  }
  std::reverse(gather_outer_axis_strides.begin(), gather_outer_axis_strides.end());
  ss << "(" << tpipe.tiler.Offset(gather_outer_axis_id, gather_outer_axis_id, gather_outer_axis_strides) << ")";
  return ss.str();
}

void CollectParamOuterAndInnerAxes(const std::vector<ascir::AxisId> &param_axis, ascir::AxisId gather_axis_id,
                                   std::vector<ascir::AxisId> &param_outer_axes,
                                   std::vector<ascir::AxisId> &param_inner_axes) {
  const size_t gather_axis_id_t = static_cast<size_t>(gather_axis_id);                          
  for (size_t i = 0; i < param_axis.size(); i++) {
    if (i < gather_axis_id_t) {
      param_outer_axes.emplace_back(param_axis[i]);
    }
    if (i > gather_axis_id_t) {
      param_inner_axes.emplace_back(param_axis[i]);
    }
  }
  return;
}

std::string CalGatherParamOffset(const std::vector<ascir::AxisId> &param_axis, std::string indices_value,
                                 ascir::AxisId gather_axis_id, const Axis &inner_vectorized_axis, const TPipe &tpipe) {
  stringstream ss;
  const size_t gather_axis_id_t = static_cast<size_t>(gather_axis_id);  
  for (size_t i = 0; i < param_axis.size(); i++) {
    if (i == gather_axis_id_t) {
      ss << indices_value << " * ";
      for (size_t j = i + 1; j < param_axis.size(); j++) {
        ss << tpipe.tiler.GetAxis(param_axis[j]).axis_size << " * ";
      }
      ss << "1";
      break;
    }
    ss << tpipe.tiler.GetAxis(param_axis[i]) << " * ";
    for (size_t j = i + 1; j < param_axis.size(); j++) {
      ss << tpipe.tiler.GetAxis(param_axis[j]).axis_size << " * ";
    }
    ss << "1 + ";
  }
  const auto &outer_axis = tpipe.tiler.GetAxis(inner_vectorized_axis.split_pair_other_id);
  ss << " + " << outer_axis << " * " << tpipe.tiler.Size(inner_vectorized_axis.size) << ";" << std::endl;
  return ss.str();
}

std::string CalGatherIndicesAxesSize(const std::vector<ascir::AxisId> &indices_axes, const TPipe &tpipe) {
  stringstream ss;
  ss << "(";
  for (size_t i = 0; i < indices_axes.size(); i++) {
    ss << tpipe.tiler.GetAxis(indices_axes[i]).axis_size << " * ";
  }
  ss << "1)";
  return ss.str();
}

std::string CalGatherOuterSize(const std::vector<ascir::AxisId> &param_axis, ascir::AxisId gather_axis_id, const TPipe &tpipe) {
  stringstream ss;
  const size_t gather_axis_id_t = static_cast<size_t>(gather_axis_id);  
  for (size_t i = 0; i < param_axis.size(); i++) {
    if (i < gather_axis_id_t) {
      ss << tpipe.tiler.GetAxis(param_axis[i]).axis_size << " * ";
    }
    else {
      break;
    }
  }
  ss << "1";
  return ss.str();
}

std::string CalGatherInnerSize(const std::vector<ascir::AxisId> &param_axis, ascir::AxisId gather_axis_id, const TPipe &tpipe) {
  stringstream ss;
  const size_t gather_axis_id_t = static_cast<size_t>(gather_axis_id); 
  for (size_t i = 0; i < param_axis.size(); i++) {
    if (i > gather_axis_id_t) {
      ss << tpipe.tiler.GetAxis(param_axis[i]).axis_size << " * ";
    }
  }

  ss << "1";
  return ss.str();
}

std::string CalGatherSize(const std::vector<ascir::AxisId> &param_axis, const TPipe &tpipe) {
  stringstream ss;
  for (size_t i = 0; i < param_axis.size(); i++) {
    ss << tpipe.tiler.GetAxis(param_axis[i]).axis_size << " * ";
  }
  ss << "1";
  return ss.str();
}

}