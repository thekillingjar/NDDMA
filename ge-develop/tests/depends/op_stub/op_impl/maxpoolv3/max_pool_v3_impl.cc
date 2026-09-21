/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "max_pool_v3_impl.h"
#include "framework/common/debug/ge_log.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_context.h"
#include <nlohmann/json.hpp>

using namespace ge;

namespace gert {
const int64_t INDEX_0 = 0;
const int64_t INDEX_1 = 1;
const int64_t INDEX_2 = 2;
const int64_t INDEX_3 = 3;
const int64_t INDEX_4 = 4;
const int64_t DIMNUM_NC1HWC0 = 5;
const int64_t DIMSIZE_C0 = 16;
constexpr int32_t PADDING_VALUE = 2;
constexpr int32_t TILING_MODE_6 = 6;
constexpr int32_t TILING_MODE_7 = 7;
constexpr int DEFAULT_PARAS_INPUT_SIZE = 1;

#define CHECK(cond, log_func, expr)                                                                                    \
  do {                                                                                                                 \
    if (cond) {                                                                                                        \
      log_func;                                                                                                        \
      expr;                                                                                                            \
    }                                                                                                                  \
  } while (0)

static void SAMEUpdateDim(const int64_t &ksize, const int64_t &strides, int64_t &dim_size) {
  // warning strides divisor cannnot be 0
  dim_size = (dim_size - ksize + strides) / strides;
}

static void CALCULATEUpdateDim(const int64_t &ksize, const int64_t &strides, bool ceil_mode, int32_t pad_a,
                               int32_t pad_b, int64_t &dim_size) {
  // warning strides divisor cannnot be 0
  if (ceil_mode) {
    dim_size = (dim_size + pad_a + pad_b - ksize + strides + strides - 1) / strides;
  } else {
    dim_size = (dim_size + pad_a + pad_b - ksize + strides) / strides;
  }
}

ge::graphStatus InferShapeForMaxPoolV3(InferShapeContext *context) {
  auto in_shape = context->GetInputShape(0);
  auto out_shape = context->GetOutputShape(0);
  auto src_td = context->GetInputDesc(0);
  auto *attrs = context->GetAttrs();

  if (in_shape == nullptr || out_shape == nullptr || src_td == nullptr || attrs == nullptr) {
    return ge::FAILED;
  }

  auto input_format = src_td->GetStorageFormat();

  const ContinuousVector *ksize = attrs->GetAttrPointer<ContinuousVector>(0);
  const ContinuousVector *strides = attrs->GetAttrPointer<ContinuousVector>(1);
  if (ksize->GetSize() != 4) {
    GELOGE(ge::GRAPH_FAILED, "Length of ksize must be 4!");
    return GRAPH_FAILED;
  }
  auto ksize_data = reinterpret_cast<const int64_t *>(ksize->GetData());
  if (strides->GetSize() != 4) {
    GELOGE(ge::GRAPH_FAILED, "Length of strides must be 4!");
    return GRAPH_FAILED;
  }
  auto strides_data = reinterpret_cast<const int64_t *>(strides->GetData());
  // padding_mode must in ("SAME", "VALID", "CALCULATED")
  const char *padding_mode = attrs->GetAttrPointer<char>(2);
  const ContinuousVector *pads = attrs->GetAttrPointer<ContinuousVector>(3);
  if (pads->GetSize() != 4) {
    GELOGE(ge::GRAPH_FAILED, "Length of pads %ld must be 4!", pads->GetSize());
    return GRAPH_FAILED;
  }
  auto pads_data = reinterpret_cast<const int64_t *>(pads->GetData());
  int32_t pad_top = pads_data[0];
  int32_t pad_bottom = pads_data[1];
  int32_t pad_left = pads_data[2];
  int32_t pad_right = pads_data[3];

  const bool *global_pooling = attrs->GetAttrPointer<bool>(5);
  const bool *ceil_mode = attrs->GetAttrPointer<bool>(6);
  *out_shape = *in_shape;
  size_t h_dim = input_format == FORMAT_NHWC ? 1 : 2;
  size_t w_dim = input_format == FORMAT_NHWC ? 2 : 3;

  if (*global_pooling) {
    out_shape->SetDim(h_dim, 1);
    out_shape->SetDim(w_dim, 1);
  } else {
    if (strides_data[h_dim] <= 0 || strides_data[w_dim] <= 0) {
      GELOGE(ge::GRAPH_FAILED, "strides h and w must greater than 0.");
      return GRAPH_FAILED;
    }
    if (strcmp(padding_mode, "CALCULATED") == 0) {
      // padding_mode = "CALCULATED"
      int64_t dim_size = in_shape->GetDim(h_dim);
      CALCULATEUpdateDim(ksize_data[h_dim], strides_data[h_dim], *ceil_mode, pad_top, pad_bottom, dim_size);
      out_shape->SetDim(h_dim, dim_size);
      dim_size = in_shape->GetDim(w_dim);
      CALCULATEUpdateDim(ksize_data[w_dim], strides_data[w_dim], *ceil_mode, pad_left, pad_right, dim_size);
      out_shape->SetDim(w_dim, dim_size);
    } else {
      // padding_mode in ("SAME", "VALID")
      int64_t dim_size = in_shape->GetDim(h_dim);
      SAMEUpdateDim(1, strides_data[h_dim], dim_size);
      out_shape->SetDim(h_dim, dim_size);
      dim_size = in_shape->GetDim(w_dim);
      SAMEUpdateDim(1, strides_data[w_dim], dim_size);
      out_shape->SetDim(w_dim, dim_size);
    }
  }
  return ge::GRAPH_SUCCESS;
}

static void CalCoreNum(MaxPoolV3Param *param, int32_t total_ele, int32_t core_num) {
  CHECK(core_num == 0, GELOGE(ge::GRAPH_FAILED, "core_num = 0 is not support"), return );
  param->one_core_ele = (total_ele + core_num - 1) / core_num;
  param->act_core_num = total_ele / param->one_core_ele;
  if (total_ele % param->one_core_ele != 0) {
    param->act_core_num = param->act_core_num + 1;
  }
  param->last_core_ele = total_ele - (param->act_core_num - 1) * param->one_core_ele;
}

static void CalTilingParam(MaxPoolV3Param *param, const Shape &input_shape, const MaxPoolV3CompileInfo *compile_info,
                           int32_t ksize_h, int32_t ksize_w) {
  int32_t ub_ele = compile_info->ub_ele;
  int32_t core_num = compile_info->core_num;
  int32_t strides_h = compile_info->strides_h;
  int32_t strides_w = compile_info->strides_w;
  int32_t padding = compile_info->padding;      // SAME
  int32_t ceil_mode = compile_info->ceil_mode;  // floor
  int32_t pad_top = compile_info->pad_top;
  int32_t pad_bottom = compile_info->pad_bottom;
  int32_t pad_left = compile_info->pad_left;
  int32_t pad_right = compile_info->pad_right;
  CHECK(strides_h == 0, GELOGE(ge::GRAPH_FAILED, "strides_h = 0 is not support"), return );
  CHECK(strides_w == 0, GELOGE(ge::GRAPH_FAILED, "strides_w = 0 is not support"), return );
  // calc output height and width, pad infos
  if (padding == 0) {
    param->output_h = (param->input_h + strides_h - 1) / strides_h;
    param->output_w = (param->input_w + strides_w - 1) / strides_w;
    param->pad_h = (param->output_h - 1) * strides_h + ksize_h;
    param->pad_w = (param->output_w - 1) * strides_w + ksize_w;
    param->pad_t = (param->pad_h - param->input_h) / 2 > 0 ? (param->pad_h - param->input_h) / 2 : 0;
    param->pad_b = param->pad_h - param->input_h - param->pad_t > 0 ? param->pad_h - param->input_h - param->pad_t : 0;
    param->pad_l = (param->pad_w - param->input_w) / 2 > 0 ? (param->pad_w - param->input_w) / 2 : 0;
    param->pad_r = param->pad_w - param->input_w - param->pad_l > 0 ? param->pad_w - param->input_w - param->pad_l : 0;
  } else if (padding == PADDING_VALUE) {
    if (ceil_mode == 1) {
      param->output_h = (param->input_h + pad_top + pad_bottom - ksize_h + strides_h + strides_h - 1) / strides_h;
      param->output_w = (param->input_w + pad_left + pad_right - ksize_w + strides_w + strides_w - 1) / strides_w;
      if (pad_top != 0 || pad_left != 0) {
        if ((param->output_h - 1) * strides_h >= param->input_h + pad_top) {
          param->output_h -= 1;
        }
        if ((param->output_w - 1) * strides_w >= param->input_w + pad_left) {
          param->output_w -= 1;
        }
      }
      param->pad_h = (param->output_h - 1) * strides_h + ksize_h;
      param->pad_w = (param->output_w - 1) * strides_w + ksize_w;
      param->pad_t = pad_top;
      param->pad_b = (param->pad_h - param->input_h - pad_top) > 0 ? (param->pad_h - param->input_h - pad_top) : 0;
      param->pad_l = pad_left;
      param->pad_r = (param->pad_w - param->input_w - pad_left) > 0 ? (param->pad_w - param->input_w - pad_left) : 0;
    } else {
      param->output_h = (param->input_h + pad_top + pad_bottom - ksize_h + strides_h) / strides_h;
      param->output_w = (param->input_w + pad_left + pad_right - ksize_w + strides_w) / strides_w;
      if (pad_top != 0 || pad_left != 0) {
        if ((param->output_h - 1) * strides_h >= param->input_h + pad_top) {
          param->output_h -= 1;
        }
        if ((param->output_w - 1) * strides_w >= param->input_w + pad_left) {
          param->output_w -= 1;
        }
      }
      param->pad_h = (param->output_h - 1) * strides_h + ksize_h;
      param->pad_w = (param->output_w - 1) * strides_w + ksize_w;
      param->pad_t = pad_top;
      param->pad_b = (param->pad_h - param->input_h - pad_top) > 0 ? (param->pad_h - param->input_h - pad_top) : 0;
      param->pad_l = pad_left;
      param->pad_r = (param->pad_w - param->input_w - pad_left) > 0 ? (param->pad_w - param->input_w - pad_left) : 0;
    }
  } else {
    param->output_h = (param->input_h - (ksize_h - 1) + strides_h - 1) / strides_h;
    param->output_w = (param->input_w - (ksize_w - 1) + strides_w - 1) / strides_w;
    param->pad_h = (param->output_h - 1) * strides_h + ksize_h;
    param->pad_w = (param->output_w - 1) * strides_w + ksize_w;
    param->pad_t = 0;
    param->pad_b = 0;
    param->pad_l = 0;
    param->pad_r = 0;
  }

  // calc core_num, core_ele, loop_num and loop_left
  // global pooling max_pool_v3
  if (ksize_h == param->input_h && ksize_w == param->input_w) {
    param->n_c1 = input_shape.GetDim(INDEX_0) * input_shape.GetDim(INDEX_1);
    CalCoreNum(param, param->n_c1, core_num);
    if (ub_ele >= (input_shape.GetDim(INDEX_2) * input_shape.GetDim(INDEX_3) * input_shape.GetDim(INDEX_4))) {
      param->tiling_mode = TILING_MODE_6;
    } else {
      param->h_factor = ub_ele / input_shape.GetDim(INDEX_4);  // acutal is hw_factor
      int32_t input_hw_num = param->input_h * param->input_w;
      param->one_core_loop_num = input_hw_num / param->h_factor;
      // dif from other tiling mode,this is used to tiling hw
      param->one_core_loop_left = input_hw_num % param->h_factor;
      param->last_core_loop_num = param->one_core_loop_num;
      param->last_core_loop_left = param->one_core_loop_left;
      param->tiling_mode = TILING_MODE_7;
    }
    return;
  }
  if ((ksize_h == 1) && (ksize_w == 1) && (strides_h == 1) && (strides_w == 1)) {
    param->tiling_mode = 0;
    int32_t max_ele = ub_ele / input_shape.GetDim(INDEX_4);
    int32_t total_ele = input_shape.GetDim(INDEX_0) * input_shape.GetDim(INDEX_1) * input_shape.GetDim(INDEX_2) *
        input_shape.GetDim(INDEX_3);
    CalCoreNum(param, total_ele, core_num);
    param->one_core_loop_num = param->one_core_ele / max_ele;
    param->one_core_loop_left = param->one_core_ele % max_ele;
    param->last_core_loop_num = param->last_core_ele / max_ele;
    param->last_core_loop_left = param->last_core_ele % max_ele;
  } else {
    int32_t one_sixth_ub_ele = ub_ele / 6;
    param->n_c1 = input_shape.GetDim(INDEX_0) * input_shape.GetDim(INDEX_1);
    if (param->pad_h * param->pad_w * input_shape.GetDim(INDEX_4) <= one_sixth_ub_ele) {
      param->tiling_mode = 1;
      CalCoreNum(param, param->n_c1, core_num);
      param->c_factor = one_sixth_ub_ele / (param->pad_h * param->pad_w * input_shape.GetDim(INDEX_4));
      param->one_core_loop_num = param->one_core_ele / param->c_factor;
      param->one_core_loop_left = param->one_core_ele % param->c_factor;
      param->last_core_loop_num = param->last_core_ele / param->c_factor;
      param->last_core_loop_left = param->last_core_ele % param->c_factor;
    } else if (ksize_h * param->pad_w * input_shape.GetDim(INDEX_4) <= one_sixth_ub_ele) {
      param->h_factor = (one_sixth_ub_ele / (param->pad_w * input_shape.GetDim(INDEX_4)) - ksize_h) / strides_h + 1;
      int32_t h_loop = param->output_h / param->h_factor;
      if (h_loop <= param->n_c1) {
        param->tiling_mode = 2;
        CalCoreNum(param, param->n_c1, core_num);
        param->one_core_loop_num = param->output_h / param->h_factor;
        param->one_core_loop_left = param->output_h % param->h_factor;
        param->last_core_loop_num = param->one_core_loop_num;
        param->last_core_loop_left = param->one_core_loop_left;
      } else {
        param->tiling_mode = 4;
        CalCoreNum(param, param->output_h, core_num);
        param->one_core_loop_num = param->one_core_ele / param->h_factor;
        param->one_core_loop_left = param->one_core_ele % param->h_factor;
        param->last_core_loop_num = param->last_core_ele / param->h_factor;
        param->last_core_loop_left = param->last_core_ele % param->h_factor;
      }
    } else {
      param->w_factor = (one_sixth_ub_ele / input_shape.GetDim(INDEX_4) / ksize_h - ksize_w) / strides_w + 1;
      param->one_core_loop_num = param->output_w / param->w_factor;
      param->one_core_loop_left = param->output_w % param->w_factor;
      param->last_core_loop_num = param->one_core_loop_num;
      param->last_core_loop_left = param->one_core_loop_left;
      if (param->output_h <= param->n_c1) {
        param->tiling_mode = 3;
        CalCoreNum(param, param->n_c1, core_num);
      } else {
        param->tiling_mode = 5;
        CalCoreNum(param, param->output_h, core_num);
      }
    }
  }
}

ge::graphStatus TilingForMaxPoolV3(TilingContext *context) {
  if (context->GetComputeNodeInputNum() < DEFAULT_PARAS_INPUT_SIZE) {
    return ge::GRAPH_FAILED;
  }
  auto x_shape = context->GetInputShape(0);
  auto src_td = context->GetInputDesc(0);
  if (x_shape == nullptr || src_td == nullptr) {
    return ge::GRAPH_FAILED;
  }

  auto compile_info = reinterpret_cast<const MaxPoolV3CompileInfo *>(context->GetCompileInfo());
  if (compile_info == nullptr) {
    return ge::GRAPH_FAILED;
  }

  auto param = context->GetTilingData<MaxPoolV3Param>();
  if (param == nullptr) {
    return ge::GRAPH_FAILED;
  }
  // get and check input format and shape
  ge::Format input_format = src_td->GetStorageFormat();
  CHECK(input_format != FORMAT_NC1HWC0,
        GELOGE(ge::GRAPH_FAILED, "Get input format failed, only support NC1HWC0, but got %s.",
               std::to_string(input_format).c_str()),
        return ge::GRAPH_FAILED);

  const auto &input_shape = x_shape->GetStorageShape();
  CHECK(input_shape.GetDimNum() != DIMNUM_NC1HWC0,
        GELOGE(ge::GRAPH_FAILED, "Get input shape failed, the length of input shape must be 5, but got %lu.",
               input_shape.GetDimNum()),
        return ge::GRAPH_FAILED);

  CHECK(input_shape.GetDim(INDEX_4) != DIMSIZE_C0,
        GELOGE(ge::GRAPH_FAILED, "Get input shape failed, dim 5 of input_shape must be 16, but got %lu.",
               input_shape.GetDim(INDEX_4)),
        return ge::GRAPH_FAILED);

  // check compile info paramters
  CHECK((compile_info->ub_ele <= 0), GELOGE(ge::GRAPH_FAILED, "ub_ele must greater than 0."), return ge::GRAPH_SUCCESS);
  CHECK((compile_info->core_num <= 0), GELOGE(ge::GRAPH_FAILED, "core_num must greater than 0."),
        return ge::GRAPH_SUCCESS);
  CHECK((compile_info->ksize_h <= 0) || (compile_info->ksize_w <= 0) || (compile_info->strides_h <= 0) ||
            (compile_info->strides_w <= 0),
        GELOGE(ge::GRAPH_FAILED, "ksize and strides must greater than 0."), return ge::GRAPH_SUCCESS);

  // check ksize, strides and input shape
  int32_t ksize_h = compile_info->ksize_h;
  int32_t ksize_w = compile_info->ksize_w;
  param->input_h = input_shape.GetDim(INDEX_2);
  param->input_w = input_shape.GetDim(INDEX_3);
  if (compile_info->global == 1) {
    ksize_h = param->input_h;
    ksize_w = param->input_w;
  }
  CHECK(
      (compile_info->padding == 1) && ((ksize_h > param->input_h) || (ksize_w > param->input_w)),
      GELOGE(ge::GRAPH_FAILED, "Input height or width must greater than or equal to ksize when padding mode is valid."),
      return ge::GRAPH_SUCCESS);

  // calc tiling params, set tiling params, print tiling params
  CalTilingParam(param, input_shape, compile_info, ksize_h, ksize_w);
  if ((compile_info->pad_left > 0) || (compile_info->pad_top > 0)) {
    CHECK(((param->output_w - 1) * compile_info->strides_w >= param->input_w + compile_info->pad_left) ||
              ((param->output_h - 1) * compile_info->strides_h >= param->input_h + compile_info->pad_top),
          GELOGE(ge::GRAPH_FAILED,
                 "Can not ensure that the last pooling starts strictly inside the image even after clip the last."),
          return ge::GRAPH_SUCCESS);
  }

  return ge::GRAPH_SUCCESS;
}

template<typename T>
bool GetCompileValue(const nlohmann::json &all_vars, const char *name, T &value) {
  if (all_vars.count(name) == 0) {
    return false;
  }

  value = all_vars[name].get<T>();
  return true;
}

#define GET_COMPILE_VALUE(vars, compile_info, name) GetCompileValue(vars, #name, (*compile_info).name)

//TODO hardcode MaxPoolV3
ge::graphStatus TilingPrepareForMaxPoolV3(KernelContext *context) {
  auto compile_info = context->GetOutputPointer<MaxPoolV3CompileInfo>(0);
  auto json_str = context->GetInputStrPointer(0);
  if (compile_info == nullptr || json_str == nullptr) {
    return ge::GRAPH_FAILED;
  }
  std::unique_ptr<nlohmann::json> parsed_object_cinfo(new nlohmann::json(nlohmann::json::parse(json_str)));
  if (parsed_object_cinfo == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const nlohmann::json &vars = (*parsed_object_cinfo)["vars"];
  if (vars.empty()) {
    GELOGE(ge::GRAPH_FAILED, "MaxPoolV33TilingParse: get vars failed.");
    return ge::GRAPH_FAILED;
  }
  GET_COMPILE_VALUE(vars, compile_info, ub_ele);
  GET_COMPILE_VALUE(vars, compile_info, core_num);
  GET_COMPILE_VALUE(vars, compile_info, ksize_h);
  GET_COMPILE_VALUE(vars, compile_info, ksize_w);
  GET_COMPILE_VALUE(vars, compile_info, strides_h);
  GET_COMPILE_VALUE(vars, compile_info, strides_w);
  GET_COMPILE_VALUE(vars, compile_info, padding);
  GET_COMPILE_VALUE(vars, compile_info, ceil_mode);
  GET_COMPILE_VALUE(vars, compile_info, pad_top);
  GET_COMPILE_VALUE(vars, compile_info, pad_bottom);
  GET_COMPILE_VALUE(vars, compile_info, pad_left);
  GET_COMPILE_VALUE(vars, compile_info, pad_right);
  GET_COMPILE_VALUE(vars, compile_info, global);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(MaxPoolV3)
    .InferShape(InferShapeForMaxPoolV3)
    .Tiling(TilingForMaxPoolV3)
    .TilingParse<MaxPoolV3CompileInfo>(TilingPrepareForMaxPoolV3);
}  // namespace gert
