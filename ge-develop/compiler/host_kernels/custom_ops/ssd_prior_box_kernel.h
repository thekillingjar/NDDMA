/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_FOLDING_KERNEL_SSD_PRIOR_BOX_KERNEL_H_
#define GE_GRAPH_PASSES_FOLDING_KERNEL_SSD_PRIOR_BOX_KERNEL_H_

#include <memory>
#include <vector>

#include "host_kernels/kernel.h"

namespace ge {
class SsdPriorboxKernel : public Kernel {
 public:
  /**
   * Entry of the SsdPriorboxKernel optimizer
   * @param [in] node: Input Node
   * @return SUCCESS: node output compute success
   * @return OTHERS:  Execution failed
   * @author
   */
  Status Compute(const NodePtr &node, std::vector<GeTensorPtr> &v_output) const override;

 private:
  /**
   * Get specific op_desc attr value
   * @param [in] op_desc: Input op_desc
   * @param [in/out] img_width: img_width attr_value
   * @param [in/out] img_height: img_height attr_value
   * @param [in/out] step_h: step_h attr_value
   * @param [in/out] step_w: step_w attr_value
   * @param [in/out] layer_width: layer_width attr_value
   * @param [in/out] layer_height: layer_height attr_value
   * @return SUCCESS: node get attr value success
   * @return OTHERS:  Execution failed
   * @author
   */
  Status GetPriorSizeParam(const OpDescPtr &op_desc, int32_t &img_width, int32_t &img_height, float &step_w,
                           float &step_h, int32_t &layer_width, int32_t &layer_height) const;
  /**
   * Get specific op_desc size attr value,min_size_num etc.
   * @param [in] op_desc: Input op_desc
   * @param [in/out] offset: offset attr_value
   * @param [in/out] clip: clip attr_value
   * @return SUCCESS: get attr success
   * @return OTHERS:  Execution failed
   * @author
   */
  Status GetPriorOtherParam(const OpDescPtr &op_desc, float &offset, bool &clip) const;
  /**
   * Get specific op_desc list attr value,min_size_list etc.
   * @param [in] op_desc: Input op_desc
   * @param [in/out] min_size_list: min_size_list attr_value
   * @param [in/out] max_size_list: max_size_list attr_value
   * @param [in/out] aspect_ratio_list: aspect_ratio_list attr_value
   * @param [in/out] variance_list: variance_list attr_value
   * @param [in/out] clip: clip attr_value
   * @return SUCCESS: get list attr success
   * @return OTHERS:  Execution failed
   * @author
   */
  Status GetPriorListParam(const OpDescPtr &op_desc, std::vector<float> &min_size_list,
                           std::vector<float> &max_size_list, std::vector<float> &aspect_ratio_list,
                           std::vector<float> &variance_list) const;
  /**
   * set variance param  to output_data.
   * @param [in] variance: variance list
   * @param [in] dim: output_data second channel offset
   * @param [in] layer_height: layer_height
   * @param [in] num_priors: num_priors
   * @param [in/out] output_data: output_data
   * @return SUCCESS: set variance success
   * @return OTHERS:  Execution failed
   * @author
   */
  Status SetVariance(const std::vector<float> &variance, const int32_t dim, const int32_t layer_height,
                     const int32_t layer_width, const int32_t num_priors, float *output_data) const;
  /**
   * get num priors and dim size.
   * @param [in] aspect_ratios_size: aspect_ratio_list size
   * @param [in] min_sizes_size: min_size_list size
   * @param [in] max_sizes_size: max_size_list size
   * @param [in] layer_width: layer_width
   * @param [in] layer_height: layer_height
   * @param [in/out] num_priors: num_priors
   * @param [in/out] dim_size: dim_size
   * @return SUCCESS: set variance success
   * @return OTHERS:  Execution failed
   * @author
   */
  Status GetNumPriorAndDimSize(uint32_t aspect_ratios_size, uint32_t min_sizes_size, uint32_t max_sizes_size,
                               int32_t layer_width, int32_t layer_height, int32_t &num_priors, int32_t &dim_size) const;

  void DataCalulate(float x, float y, float box_x, float box_y, int32_t img_x, int32_t img_y,
                    std::vector<float> &result) const;

  std::unique_ptr<float[]> BoundaryCalulate(int32_t dim_size, int32_t layer_width, int32_t layer_height,
                                            float step_width, float step_height, int32_t img_width, int32_t img_height,
                                            float offset, std::vector<float> min_sizes, std::vector<float> max_sizes,
                                            std::vector<float> aspect_ratios) const;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_FOLDING_KERNEL_SSD_PRIOR_BOX_KERNEL_H_
