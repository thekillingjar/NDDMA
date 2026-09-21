/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "host_kernels/custom_ops/ssd_prior_box_kernel.h"

#include <cfloat>
#include <algorithm>
#include <memory>
#include <utility>

#include "common/math/math_util.h"
#include "graph/utils/math_util.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/passes/pass_utils.h"
#include "graph/utils/attr_utils.h"
#include "host_kernels/kernel_factory.h"

namespace ge {
namespace {
const float kMinistBias = 1e-6f;
const float kAspectRationBase = 1.0;
const size_t kBoundarySize = 4;
const size_t kOutputDescFirstIndex = 0;
const size_t kDimIndexZero = 0;
const size_t kDimIndexOne = 1;
const size_t kDimIndexTwo = 2;
const size_t kDimIndexThree = 3;
const int32_t kNumVariance = 4;
const int32_t kNumOne = 1;
const int32_t kNumTwo = 2;
const float kFloatNumTwo = 2.0;
}  // namespace

Status SsdPriorboxKernel::GetPriorSizeParam(const OpDescPtr &op_desc, int32_t &img_width, int32_t &img_height,
                                            float &step_w, float &step_h, int32_t &layer_width,
                                            int32_t &layer_height) const {
  if (op_desc == nullptr) {
    GELOGE(PARAM_INVALID, "input opdescptr is nullptr.");
    return PARAM_INVALID;
  }
  const GeTensorDesc tensor_desc = op_desc->GetInputDesc(kOutputDescFirstIndex);
  layer_width = tensor_desc.GetShape().GetDim(kDimIndexThree);
  layer_height = tensor_desc.GetShape().GetDim(kDimIndexTwo);
  if (layer_height == 0 || layer_width == 0) {
    GELOGE(PARAM_INVALID, "op:%s NCHW_DIM_H or NCHW_DIM_W is 0", op_desc->GetName().c_str());
    return PARAM_INVALID;
  }
  int32_t img_h = 0;
  int32_t img_w = 0;
  if (!AttrUtils::GetInt(op_desc, SSD_PRIOR_BOX_ATTR_IMG_H, img_h)) {
    GELOGE(PARAM_INVALID, "op:%s img_h attr is null", op_desc->GetName().c_str());
    return PARAM_INVALID;
  }
  if (!AttrUtils::GetInt(op_desc, SSD_PRIOR_BOX_ATTR_IMG_W, img_w)) {
    GELOGE(PARAM_INVALID, "op:%s img_w attr is null", op_desc->GetName().c_str());
    return PARAM_INVALID;
  }
  if (img_h == 0 || img_w == 0) {
    GELOGE(PARAM_INVALID, "op:%s Either img_h or img_w is null", op_desc->GetName().c_str());
    return PARAM_INVALID;
  } else {
    img_width = static_cast<int32_t>(img_w);
    img_height = static_cast<int32_t>(img_h);
  }
  float step_height = 0.0;
  float step_width = 0.0;
  if (!AttrUtils::GetFloat(op_desc, SSD_PRIOR_BOX_ATTR_STEP_H, step_height)) {
    GELOGE(PARAM_INVALID, "op:%s step_height attr is null", op_desc->GetName().c_str());
    return PARAM_INVALID;
  }
  if (!AttrUtils::GetFloat(op_desc, SSD_PRIOR_BOX_ATTR_STEP_W, step_width)) {
    GELOGE(PARAM_INVALID, "op:%s step_width attr is null", op_desc->GetName().c_str());
    return PARAM_INVALID;
  }
  if ((fabs(step_height) < FLT_EPSILON) || (fabs(step_width) < FLT_EPSILON)) {
    step_w = static_cast<float>(img_width) / layer_width;
    step_h = static_cast<float>(img_height) / layer_height;
  } else {
    step_w = step_width;
    step_h = step_height;
  }

  return SUCCESS;
}

Status SsdPriorboxKernel::GetPriorListParam(const OpDescPtr &op_desc, std::vector<float> &min_size_list,
                                            std::vector<float> &max_size_list, std::vector<float> &aspect_ratio_list,
                                            std::vector<float> &variance_list) const {
  if (!AttrUtils::GetListFloat(op_desc, SSD_PRIOR_BOX_ATTR_MIN_SIZE, min_size_list)) {
    GELOGE(PARAM_INVALID, "op:%s min_size() attr is null", op_desc->GetName().c_str());
    return PARAM_INVALID;
  }

  if (!AttrUtils::GetListFloat(op_desc, SSD_PRIOR_BOX_ATTR_MAX_SIZE, max_size_list)) {
    GELOGE(PARAM_INVALID, "op:%s max_size() attr is null", op_desc->GetName().c_str());
    return PARAM_INVALID;
  }

  if (!AttrUtils::GetListFloat(op_desc, SSD_PRIOR_BOX_ATTR_VARIANCE, variance_list)) {
    GELOGE(PARAM_INVALID, "op:%s variance() attr is null", op_desc->GetName().c_str());
    return PARAM_INVALID;
  }

  if (!AttrUtils::GetListFloat(op_desc, SSD_PRIOR_BOX_ATTR_ASPECT_RATIO, aspect_ratio_list)) {
    GELOGE(PARAM_INVALID, "op:%s aspect_ratio() attr is null", op_desc->GetName().c_str());
    return PARAM_INVALID;
  }
  // if flip is true,aspect_ratio_list need add reciprocal
  bool flip = false;
  if (!AttrUtils::GetBool(op_desc, SSD_PRIOR_BOX_ATTR_FLIP, flip)) {
    GELOGE(PARAM_INVALID, "op:%s flip() attr is null", op_desc->GetName().c_str());
    return PARAM_INVALID;
  }
  std::vector<float> aspect_ratios;
  aspect_ratios.push_back(static_cast<float>(SSD_PRIORBOX_ASPECT_RATIO_VALUE));
  for (size_t i = 0; i < aspect_ratio_list.size(); i++) {
    float ar = aspect_ratio_list.at(i);
    bool already_exist =
        std::any_of(aspect_ratios.begin(), aspect_ratios.end(), [&ar](float x) { return fabs(ar - x) < kMinistBias; });
    if (!already_exist) {
      aspect_ratios.push_back(ar);
      if (flip) {
        aspect_ratios.push_back(1.0f / ar);  // 1. reciprocal
      }
    }
  }
  aspect_ratio_list = std::move(aspect_ratios);
  return SUCCESS;
}

Status SsdPriorboxKernel::GetPriorOtherParam(const OpDescPtr &op_desc, float &offset, bool &clip) const {
  if (!AttrUtils::GetBool(op_desc, SSD_PRIOR_BOX_ATTR_CLIP, clip)) {
    GELOGE(PARAM_INVALID, "op:%s clip() attr is null", op_desc->GetName().c_str());
    return PARAM_INVALID;
  }
  if (!AttrUtils::GetFloat(op_desc, SSD_PRIOR_BOX_ATTR_OFFSET, offset)) {
    GELOGE(PARAM_INVALID, "op:%s offset() attr is null", op_desc->GetName().c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status SsdPriorboxKernel::SetVariance(const std::vector<float> &variance, const int32_t dim, const int32_t layer_height,
                                      const int32_t layer_width, const int32_t num_priors, float *output_data) const {
  if (output_data == nullptr) {
    GELOGE(PARAM_INVALID, "output_data is null");
    return PARAM_INVALID;
  }

  output_data += dim;
  if (variance.size() == 1) {
    if (NnSet(dim, variance[0], output_data) != SUCCESS) {
      GELOGE(PARAM_INVALID, "NnSet failed.");
      return PARAM_INVALID;
    }
  } else {
    size_t count = 0;
    for (int32_t i = 0; i < layer_height * layer_width * num_priors; ++i) {
      for (size_t j = 0; j < 4; ++j) {  // 4 variance
        output_data[count] = variance[j];
        ++count;
      }
    }
  }

  return SUCCESS;
}

Status SsdPriorboxKernel::GetNumPriorAndDimSize(uint32_t aspect_ratios_size,
                                                uint32_t min_sizes_size,
                                                uint32_t max_sizes_size,
                                                int32_t layer_width,
                                                int32_t layer_height,
                                                int32_t &num_priors,
                                                int32_t &dim_size) const {
  if (ge::CheckUint32MulOverflow(min_sizes_size, aspect_ratios_size) != SUCCESS) {
    return PARAM_INVALID;
  }

  uint32_t tmp_value = aspect_ratios_size * min_sizes_size;
  if (ge::CheckUint32AddOverflow(tmp_value, max_sizes_size) != SUCCESS) {
    GELOGW("Failed to get list param.");
    return PARAM_INVALID;
  }
  tmp_value += max_sizes_size;

  if (tmp_value > INT32_MAX) {
    GELOGE(PARAM_INVALID, "Failed to get list param.");
    return PARAM_INVALID;
  }
  num_priors = static_cast<int32_t>(tmp_value);

  if (ge::CheckIntMulOverflow(layer_width, layer_height) != SUCCESS) {
    GELOGW("Failed to get list param.");
    return PARAM_INVALID;
  }

  if (ge::CheckIntMulOverflow(layer_width * layer_height, num_priors) != SUCCESS) {
    GELOGW("Failed to get list param.");
    return PARAM_INVALID;
  }

  if (ge::CheckIntMulOverflow(layer_width * layer_height * num_priors, kNumVariance) != SUCCESS) {
    GELOGW("Failed to get list param.");
    return PARAM_INVALID;
  }
  dim_size = layer_width * layer_height * num_priors * kNumVariance;  // 4 variance

  return SUCCESS;
}

void SsdPriorboxKernel::DataCalulate(float x, float y, float box_x, float box_y, int32_t img_x, int32_t img_y,
                                     std::vector<float> &result) const {
  result.clear();
  // xmin
  result.push_back((x - box_x / kFloatNumTwo) / static_cast<float>(img_x));
  // ymin
  result.push_back((y - box_y / kFloatNumTwo) / static_cast<float>(img_y));
  // xmax
  result.push_back((x + box_x / kFloatNumTwo) / static_cast<float>(img_x));
  // ymax
  result.push_back((y + box_y / kFloatNumTwo) / static_cast<float>(img_y));
}

std::unique_ptr<float[]> SsdPriorboxKernel::BoundaryCalulate(int32_t dim_size, int32_t layer_width,
                                                             int32_t layer_height, float step_width, float step_height,
                                                             int32_t img_width, int32_t img_height, float offset,
                                                             std::vector<float> min_sizes, std::vector<float> max_sizes,
                                                             std::vector<float> aspect_ratios) const {
  // output two channel.First channel stores the mean of each prior coordinate.
  // Second channel stores the variance of each prior coordinate.
  unique_ptr<float[]> output_data(new (std::nothrow) float[dim_size * kNumTwo]());
  if (output_data == nullptr) {
    GELOGE(PARAM_INVALID, "Failed to create output_data ptr.");
    return nullptr;
  }
  int32_t idx = 0;
  std::vector<float> boundaries;
  for (int32_t height_index = 0; height_index < layer_height; ++height_index) {
    for (int32_t width_index = 0; width_index < layer_width; ++width_index) {
      float center_x = (width_index + offset) * step_width;
      float center_y = (height_index + offset) * step_height;
      for (size_t size_index = 0; size_index < min_sizes.size(); ++size_index) {
        int32_t min_size = static_cast<int32_t>(min_sizes[size_index]);
        // first prior: aspect_ratio = 1, size = min_size
        float box_width = min_size;
        float box_height = min_size;
        DataCalulate(center_x, center_y, box_width, box_height, img_width, img_height, boundaries);
        size_t index = 0;
        while (index < kBoundarySize) {
          output_data[idx++] = boundaries[index++];
        }
        if (!max_sizes.empty()) {
          int32_t max_size = static_cast<int32_t>(max_sizes[size_index]);
          // second prior: aspect_ratio = 1, size = sqrtf(min_size * max_size)
          box_width = sqrtf(min_size * max_size);
          DataCalulate(center_x, center_y, box_width, box_width, img_width, img_height, boundaries);
          index = 0;
          while (index < kBoundarySize) {
            output_data[idx++] = boundaries[index++];
          }
        }

        // rest of priors
        for (size_t ratio_index = 0; ratio_index < aspect_ratios.size(); ++ratio_index) {
          float aspect_ratio = aspect_ratios[ratio_index];
          if (fabs(aspect_ratio - kAspectRationBase) < kMinistBias) {  // aspect ration base:1.
            continue;
          }
          box_width = min_size * sqrtf(aspect_ratio);
          box_height = min_size / sqrtf(aspect_ratio);
          DataCalulate(center_x, center_y, box_width, box_height, img_width, img_height, boundaries);
          index = 0;
          while (index < kBoundarySize) {
            output_data[idx++] = boundaries[index++];
          }
        }
      }
    }
  }

  return output_data;
}

Status SsdPriorboxKernel::Compute(const NodePtr &node, std::vector<GeTensorPtr> &v_output) const {
  GELOGD("SsdPriorboxKernel in");
  OpDescPtr op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    GELOGE(PARAM_INVALID, "node:%s opdesc is null", node->GetName().c_str());
    return PARAM_INVALID;
  }
  int32_t img_width = 0;
  int32_t img_height = 0;
  int32_t layer_width = 0;
  int32_t layer_height = 0;
  float step_width = 0.0;
  float step_height = 0.0;
  Status ret = GetPriorSizeParam(op_desc, img_width, img_height, step_width, step_height, layer_width, layer_height);
  if (ret != SUCCESS) {
    GELOGE(PARAM_INVALID, "Failed to get size param.");
    return PARAM_INVALID;
  }
  float offset = 0.0;
  bool clip = false;
  ret = GetPriorOtherParam(op_desc, offset, clip);
  if (ret != SUCCESS) {
    GELOGE(PARAM_INVALID, "Failed to get other param.");
    return PARAM_INVALID;
  }

  std::vector<float> min_sizes;
  std::vector<float> aspect_ratios;
  std::vector<float> variances;
  std::vector<float> max_sizes;
  if (GetPriorListParam(op_desc, min_sizes, max_sizes, aspect_ratios, variances) != SUCCESS) {
    GELOGE(PARAM_INVALID, "Failed to get list param.");
    return PARAM_INVALID;
  }

  int32_t num_priors = 0;
  int32_t dim_size = 0;
  ret = GetNumPriorAndDimSize(aspect_ratios.size(), min_sizes.size(), max_sizes.size(), layer_width, layer_height,
                              num_priors, dim_size);
  if (ret != SUCCESS) {
    GELOGE(PARAM_INVALID, "Failed to get other param.");
    return PARAM_INVALID;
  }

  auto output_data = BoundaryCalulate(dim_size, layer_width, layer_height, step_width, step_height, img_width,
                                      img_height, offset, min_sizes, max_sizes, aspect_ratios);
  if (output_data == nullptr) {
    GELOGE(PARAM_INVALID, "Failed to create output_data ptr.");
    return PARAM_INVALID;
  }

  if (clip) {
    for (int32_t d = 0; d < dim_size; ++d) {
      // clip the prior's coordidate such that it is within [0.0 1.0]
      output_data[d] = std::min<float>(std::max<float>(output_data[d], 0.), 1.);
    }
  }

  // set the variance.
  if (SetVariance(variances, dim_size, layer_height, layer_width, num_priors, output_data.get()) != SUCCESS) {
    GELOGE(PARAM_INVALID, "Failed to set variance.");
    return PARAM_INVALID;
  }

  GeTensorDesc output_tensor_desc = op_desc->GetOutputDesc(0);
  std::vector<int64_t> v_dims(3, 1);  // 3 dims
  v_dims[kDimIndexZero] = kNumOne;
  v_dims[kDimIndexOne] = kNumTwo;
  v_dims[kDimIndexTwo] = dim_size;
  DataType data_type = output_tensor_desc.GetDataType();
  output_tensor_desc.Update(GeShape(v_dims), FORMAT_NCHW, data_type);
  // make TensorDesc
  GeTensorPtr output_ptr = MakeShared<GeTensor>(output_tensor_desc);
  if (output_ptr == nullptr) {
    GELOGW("Create shared ptr for GeTensor failed");
    return NOT_CHANGED;
  }
  GE_IF_BOOL_EXEC(output_ptr->SetData(reinterpret_cast<uint8_t *>(output_data.get()),
                                      static_cast<size_t>(dim_size * kNumTwo * sizeof(data_type))) != GRAPH_SUCCESS,
                  GELOGE(INTERNAL_ERROR, "set data failed");
                  return INTERNAL_ERROR);
  v_output.push_back(output_ptr);
  return SUCCESS;
}
REGISTER_COMPUTE_NODE_KERNEL(SSDPRIORBOX, SsdPriorboxKernel);
}  // namespace ge
