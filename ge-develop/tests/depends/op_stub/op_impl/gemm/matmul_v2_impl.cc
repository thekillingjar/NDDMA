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

#define OP_LOGE(op_name, err_msg, ...)                                                                                 \
  do {                                                                                                                 \
    GELOGE(ge::GRAPH_FAILED, err_msg, ##__VA_ARGS__);                                                                  \
  } while (0)

#define OP_LOGD(op_name, err_msg, ...)                                                                                 \
  do {                                                                                                                 \
    GELOGD(err_msg, ##__VA_ARGS__);                                                                                    \
  } while (0)

#define OP_LOGW(op_name, err_msg, ...)                                                                                 \
  do {                                                                                                                 \
    GELOGD(err_msg, ##__VA_ARGS__);                                                                                    \
  } while (0)

using std::vector;
}  // namespace

namespace gert {
ge::graphStatus InferShapeForMatMulV2(InferShapeContext *context) {
  //  auto op_name = context->GetNodeName();
  auto op_type = context->GetNodeType();
  OP_LOGD(op_name, "[Infershape] Start matmul infershape.");
  // TODO: print debug info
  // OP_LOGD(op_name, "%s", GetMatMulInfo(op, "transpose").c_str());

  auto shape_x1 = context->GetInputShape(0);
  auto shape_x2 = context->GetInputShape(1);
  auto shape_out = context->GetOutputShape(0);
  auto tensor_x1 = context->GetInputDesc(0);
  auto attrs = context->GetAttrs();
  CHECK(shape_x1 == nullptr || shape_x2 == nullptr || shape_out == nullptr || attrs == nullptr,
        CUBE_INNER_ERR_REPORT(op_name, "[Infershape]shape is null"), return GRAPH_FAILED);

  const bool *trans_a = attrs->GetAttrPointer<bool>(0);
  const bool *trans_b = attrs->GetAttrPointer<bool>(1);
  CHECK(trans_a == nullptr || trans_b == nullptr, CUBE_INNER_ERR_REPORT(op_name, "[Infershape]attribute is null"),
        return GRAPH_FAILED);

  ge::DataType dtype = tensor_x1->GetDataType();
  OP_LOGD(op_name, "[Infershape] Check the input dtype.");
  if (dtype == DT_FLOAT) {
    OP_LOGW(op_name, "[Plugin][WARNING]MatMul fp32 op has poor performance!");
  }

  Shape shape_x1_new(*shape_x1);
  Shape shape_x2_new(*shape_x2);
  bool shape_x1_reshape_flag = false;
  if (shape_x1_new.GetDimNum() == 1 && shape_x1_new.GetDim(0) > 0) {
    shape_x1_reshape_flag = false;
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

  const Shape *shape_bias = nullptr;
  // TODO: strncmp cost 15ns, consider split MatMul and MatMulV2 InferShape function
  if (op_type != nullptr && strncmp(op_type, "MatMulV2", sizeof("MatMulV2") - 1) == 0) {
    // TODO: private attr
    // int64_t input_size = 0;
    // int64_t hidden_size = 0;
    // bool input_size_flag = AttrUtils::GetInt(op_desc, "input_size", input_size);
    // bool hidden_size_flag = AttrUtils::GetInt(op_desc, "hidden_size", hidden_size);
    // OP_LOGD(op_name, "input_size[%lld], hidden_size[%lld]", input_size, hidden_size);
    // if (input_size_flag && hidden_size_flag) {
    //   shape_x2_new.SetDim(1, shape_x1_new.GetDim(1));
    //   int64_t align_dim = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE +
    //                       (hidden_size + BLOCK_SIZE) / BLOCK_SIZE * BLOCK_SIZE;
    //   shape_x2_new.SetDim(0, align_dim);
    // }
    shape_bias = context->GetInputShape(2);
    OP_LOGD(op_name, "[MatMulV2 Infershape] Check the input shape length.");
    if (shape_x1_new.GetDimNum() != 2 && shape_x1_new.GetDimNum() != 4) {
      CUBE_INNER_ERR_REPORT(op_name, "[Plugin][ERROR]Matmul the first input dims is not 2 or 4!");
      return GRAPH_FAILED;
    }
  }

  size_t idx_m = *trans_a ? 1 : 0;
  size_t idx_k_a = *trans_a ? 0 : 1;
  size_t idx_k_b = *trans_b ? 1 : 0;
  size_t idx_n = *trans_b ? 0 : 1;
  if (shape_x1_new.GetDim(idx_k_a) != shape_x2_new.GetDim(idx_k_b)) {
    CUBE_INNER_ERR_REPORT(op_name, "[InferShape] The k-axis of a(%ld) and b(%ld) tensors must be the same",
                          shape_x1_new.GetDim(idx_k_a), shape_x2_new.GetDim(idx_k_b));
    return GRAPH_FAILED;
  }
  shape_out->SetDimNum(2);  // TODO: BASE_LEN
  shape_out->SetDim(0, shape_x1_new.GetDim(idx_m));
  shape_out->SetDim(1, shape_x2_new.GetDim(idx_n));
  if (shape_bias != nullptr && shape_bias->GetDimNum() > 0) {
    int64_t bias_dim = shape_bias->GetDimNum();
    CHECK(shape_bias->GetDim(bias_dim - 1) != shape_out->GetDim(1),
          OP_LOGE(op_name, "[InferShape] The dimension of n [%ld] and bias [%ld] tensors must be the same",
                  shape_out->GetDim(1), shape_bias->GetDim(bias_dim - 1)),
          return GRAPH_FAILED);
  }

  InferComplementedOutput(shape_x1_reshape_flag, shape_x2_reshape_flag, *shape_out);

  // TODO: 执行时不需要SetDataType
  // if (tensordesc_x1->GetDataType() == ge::DT_INT8 || tensordesc_x1->GetDataType() == ge::DT_INT4) {
  //   tensordesc_output->SetDataType(ge::DT_INT32);
  // } else {
  //   tensordesc_output->SetDataType(tensordesc_x1->GetDataType());
  // }
  OP_LOGD(op_name, "[Infershape] End infershape.");
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(MatMul)
    .InferShape(InferShapeForMatMulV2)
    .Tiling(TilingForGemm)
    .TilingParse<vector<GemmCompileInfo>>(TilingPrepareForGemm);

IMPL_OP(MatMulV2)
    .InferShape(InferShapeForMatMulV2)
    .Tiling(TilingForGemm)
    .TilingParse<vector<GemmCompileInfo>>(TilingPrepareForGemm);
}  // namespace gert
