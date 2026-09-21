/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_context.h"
#include "framework/common/debug/ge_log.h"
#include "gemm.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace {
const size_t kBatchMatmulMaxShapeSize = 8;

#define CHECK(cond, log_func, expr)                                                                                    \
  do {                                                                                                                 \
    if (cond) {                                                                                                        \
      log_func;                                                                                                        \
      expr;                                                                                                            \
    }                                                                                                                  \
  } while (0)

#define CUBE_INNER_ERR_REPORT(op_name, err_msg, ...)                                                                   \
  do {                                                                                                                 \
    GELOGE(ge::GRAPH_FAILED, err_msg, ##__VA_ARGS__);                                                                  \
  } while (0)

#define OP_LOGD(op_name, err_msg, ...)                                                                                 \
  do {                                                                                                                 \
    GELOGD(err_msg, ##__VA_ARGS__);                                                                                    \
  } while (0)

using std::vector;
}  // namespace

namespace gert {

class InferShapeBatchMatMul {
 public:
  InferShapeBatchMatMul(InferShapeContext *context, const Shape &shape_a, const Shape &shape_b, bool trans_a,
                        bool trans_b)
      : op_name(context->GetNodeName()), shape_a(shape_a), shape_b(shape_b), trans_a(trans_a), trans_b(trans_b),
        shape_out(*(context->GetOutputShape(0))), shape_bias(context->GetInputShape(2)) {
    num_dima = shape_a.GetDimNum();
    num_dimb = shape_b.GetDimNum();
    num_dim = std::max(num_dima, num_dimb);
    num_dim_bias = 0;
    if (shape_bias != nullptr) {
      num_dim_bias = shape_bias->GetDimNum();
      num_dim = std::max(num_dim, num_dim_bias);
    }
    shape_out.SetDimNum(num_dim);
  };

  ~InferShapeBatchMatMul(){};
  bool InferShape();

 protected:
  bool InferBatch() const;
  bool InferBias();

  static const int64_t BASE_LEN = 8;
  size_t num_dim;
  size_t num_dima;
  size_t num_dimb;
  size_t num_dim_bias;

  const char *op_name;
  const Shape &shape_a;
  const Shape &shape_b;
  bool trans_a;
  bool trans_b;
  Shape &shape_out;
  const Shape *shape_bias;
};

void CopyOutShapeFromInputShape(const Shape &shape_in, Shape &shape_out, int64_t valid_offset) {
  for (auto i = 0; i < valid_offset; ++i) {
    shape_out.SetDim(i, shape_in.GetDim(i));
  }
}

bool InferShapeBatchMatMul::InferBatch() const {
  auto valid_offset = num_dim - std::min(num_dima, num_dimb);
  const Shape &shape_long = num_dima < num_dimb ? shape_b : shape_a;
  const Shape &shape_short = num_dima < num_dimb ? shape_a : shape_b;
  int64_t shape_value_long;
  int64_t shape_value_short;

  CopyOutShapeFromInputShape(shape_long, shape_out, valid_offset);
  // use index - 2 to get index of m
  for (auto i = valid_offset; i < num_dim - 2; ++i) {
    shape_value_short = shape_short.GetDim(i - valid_offset);
    shape_value_long = shape_long.GetDim(i);
    if (shape_value_short > 1 && shape_value_long > 1 && shape_value_short != shape_value_long) {
      return false;
    }
    shape_out.SetDim(i, std::max(shape_value_short, shape_value_long));
  }
  return true;
}

bool BroadcastBatchDim(const int64_t dim_a, const int64_t dim_b, int64_t &dim) {
  if (dim_a > 1 && dim_b > 1) {
    CHECK(dim_a != dim_b,
          CUBE_INNER_ERR_REPORT(op_name, "[InferShape] dimensions a(%ld) and b(%ld) must be equal", dim_a, dim_b),
          return false);

    dim = dim_a;
    return true;
  }

  dim = std::max(dim_a, dim_b);
  return true;
}

bool InferNDimWithBias(const int64_t dim_a, const int64_t dim_b, int64_t &dim) {
  // shape_bias_n > 0 && n > 0
  if (dim_a > 0 && dim_b > 0) {
    CHECK(dim_a != dim_b,
          CUBE_INNER_ERR_REPORT(op_name, "[InferShape] dimensions a(%ld) and b(%ld) must be equal", dim_a, dim_b),
          return false);
    dim = dim_a;
    return true;
  }

  return false;
}

bool InferShapeBatchMatMul::InferBias() {
  int64_t shape_value_out = shape_out.GetDim(num_dim - 1);
  // 1) shape_bias = {}
  CHECK(num_dim_bias == 0, CUBE_INNER_ERR_REPORT(op_name, "[InferShape] bias dims number is zero"), return true);

  // 2) infer n with bias
  CHECK(!InferNDimWithBias(shape_bias->GetDim(num_dim_bias - 1), shape_out.GetDim(num_dim - 1), shape_value_out),
        CUBE_INNER_ERR_REPORT(op_name, "[InferShape] failed to infer N dim with bias"), return false);

  shape_out.SetDim(num_dim - 1, shape_value_out);

  // 3) infer batch with bias
  auto valid_offset = num_dim - std::min(num_dim_bias, std::max(num_dima, num_dimb));
  if (num_dim_bias < num_dim) {
    // stop before num_dim - 2 so as to avoid traversing axis m, n
    for (auto i = valid_offset; i < num_dim - 2; ++i) {
      CHECK(!BroadcastBatchDim(shape_bias->GetDim(i - valid_offset), shape_out.GetDim(i), shape_value_out),
            CUBE_INNER_ERR_REPORT(op_name, "[InferShape] failed to broadcast batch dim"), return false);

      shape_out.SetDim(i, shape_value_out);
    }
    return true;
  }
  CopyOutShapeFromInputShape(*shape_bias, shape_out, valid_offset);
  // stop before num_dim - 2 so as to avoid traversing axis m, n
  for (auto i = valid_offset; i < num_dim - 2; ++i) {
    CHECK(!BroadcastBatchDim(shape_bias->GetDim(i), shape_out.GetDim(i - valid_offset), shape_value_out),
          CUBE_INNER_ERR_REPORT(op_name, "[InferShape] failed to broadcast batch dim"), return false);

    shape_out.SetDim(i, shape_value_out);
  }
  return true;
}

bool InferShapeBatchMatMul::InferShape() {
  if (shape_a.GetDimNum() < 2 || shape_b.GetDimNum() < 2) {
    CHECK(!InferBatch(), CUBE_INNER_ERR_REPORT(op_name, "[InferShape] Failed to x1/x2 dim num less than 2."),
          return false);
    return false;
  }
  // using index - 2 to get m_dim
  size_t idx_m = num_dima - 2;
  size_t idx_k_a = num_dima - 1;
  // using index - 2 to get k_dim
  size_t idx_k_b = num_dimb - 2;
  size_t idx_n = num_dimb - 1;
  if (trans_a) {
    idx_m = num_dima - 1;
    // using index - 2 to get k_dim
    idx_k_a = num_dima - 2;
  }
  if (trans_b) {
    idx_k_b = num_dimb - 1;
    // using index - 2 to get n_dim
    idx_n = num_dimb - 2;
  }

  if (shape_a.GetDim(idx_k_a) != shape_b.GetDim(idx_k_b)) {
    CUBE_INNER_ERR_REPORT(op_name, "[InferShape] The k-axis of a(%ld) and b(%ld) tensors must be the same",
                          shape_a.GetDim(idx_k_a), shape_b.GetDim(idx_k_b));
    return false;
  }
  CHECK(!InferBatch(), CUBE_INNER_ERR_REPORT(op_name, "[InferShape] Failed to infer Batch."), return false);

  // using index - 2 to get m_dim in shape_out
  shape_out.SetDim((num_dim - 2), shape_a.GetDim(idx_m));
  shape_out.SetDim((num_dim - 1), shape_b.GetDim(idx_n));
  if (shape_bias != nullptr) {
    CHECK(!InferBias(), CUBE_INNER_ERR_REPORT(op_name, "[InferShape] Infer bias failed."), return false);
  }

  return true;
}

ge::graphStatus InferShapeForBatchMatMulV2(InferShapeContext *context) {
  // TODO: print debug info
  // OP_LOGD(opName.GetString(), "%s", GetMatMulInfo(op, "adj").c_str());
  auto shape_x1 = context->GetInputShape(0);
  auto shape_x2 = context->GetInputShape(1);
  auto shape_out = context->GetOutputShape(0);
  auto attrs = context->GetAttrs();
  CHECK(shape_x1 == nullptr || shape_x2 == nullptr || shape_out == nullptr || attrs == nullptr,
        CUBE_INNER_ERR_REPORT(op_name, "[Infershape]shape is null"), return GRAPH_FAILED);

  const bool *adj_x1 = attrs->GetAttrPointer<bool>(0);
  const bool *adj_x2 = attrs->GetAttrPointer<bool>(1);
  CHECK(adj_x1 == nullptr || adj_x2 == nullptr, CUBE_INNER_ERR_REPORT(op_name, "[Infershape]attribute is null"),
        return GRAPH_FAILED);

  auto dim_num = std::max(shape_x1->GetDimNum(), shape_x2->GetDimNum());
  if (dim_num < 1 || dim_num > kBatchMatmulMaxShapeSize) {
    CUBE_INNER_ERR_REPORT(op_name, "[Infershape]The shape can only be in the range of 1 to 8.");
  }

  Shape shape_x1_new(*shape_x1);
  Shape shape_x2_new(*shape_x2);
  bool shape_x1_reshape_flag = false;
  if (shape_x1_new.GetDimNum() == 1 && shape_x1_new.GetDim(0) > 0) {
    shape_x1_reshape_flag = true;
    int64_t ori_dim = shape_x1_new.GetDim(0);
    shape_x1_new.SetDimNum(2);
    shape_x1_new.SetDim(0, 1);
    shape_x1_new.SetDim(1, ori_dim);
  }

  bool shape_x2_reshape_flag = false;
  if (shape_x2_new.GetDimNum() == 1 && shape_x2_new.GetDim(0) > 0) {
    shape_x2_reshape_flag = true;
    int64_t ori_dim = shape_x2_new.GetDim(0);
    shape_x2_new.SetDimNum(2);
    shape_x2_new.SetDim(0, ori_dim);
    shape_x2_new.SetDim(1, 1);
  }

  InferShapeBatchMatMul BatchMatMulV2Infer(context, shape_x1_new, shape_x2_new, *adj_x1, *adj_x2);
  CHECK(!BatchMatMulV2Infer.InferShape(), CUBE_INNER_ERR_REPORT(op_name, "[InferShape] Failed to infer output shape"),
        return GRAPH_FAILED);

  InferComplementedOutput(shape_x1_reshape_flag, shape_x2_reshape_flag, *shape_out);

  // TODO: 执行时不需要SetDataType
  // auto dtype = src_td->GetDataType();
  // auto output_td = const_cast<CompileTimeTensorDesc *>(dst_td);
  // if (dtype == ge::DT_INT8) {
  //   output_td->SetDataType(ge::DT_INT32);
  // } else {
  //   output_td->SetDataType(dtype);
  // }
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(BatchMatMul)
    .InferShape(InferShapeForBatchMatMulV2)
    .Tiling(TilingForGemm)
    .TilingParse<vector<GemmCompileInfo>>(TilingPrepareForGemm);

IMPL_OP(BatchMatMulV2)
    .InferShape(InferShapeForBatchMatMulV2)
    .Tiling(TilingForGemm)
    .TilingParse<vector<GemmCompileInfo>>(TilingPrepareForGemm);
}  // namespace gert
