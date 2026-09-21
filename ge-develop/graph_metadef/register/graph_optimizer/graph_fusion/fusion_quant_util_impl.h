/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FUSION_QUANT_UTIL_IMPL_H_
#define INC_FUSION_QUANT_UTIL_IMPL_H_
#include "register/graph_optimizer/graph_fusion/fusion_quant_util.h"
#include "graph/node.h"
#include "common/ge_common/ge_inner_error_codes.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"
#include "graph/ge_tensor.h"
#include <vector>

namespace fe {

#define GET_REQUANT_N(req_scale_date) (((req_scale_date) & 0x000000FF00000000UL) >> 32)

#define FE_PARAM_CHECK_NOTNULL(val)                                \
  do {                                                             \
    if ((val) == nullptr) {                                        \
      GELOGE(ge::FAILED, "Parameter[%s] must not be null.", #val); \
      return fe::PARAM_INVALID;                                    \
    }                                                              \
  } while (0)

namespace {
const int32_t NCHW_DIM_N = 0;
const int32_t NCHW_DIM_C = 1;
const int32_t NCHW_DIM_H = 2;
const int32_t NCHW_DIM_W = 3;

const int32_t NC1HWC0_DIM_N = 0;
const int32_t NC1HWC0_DIM_C1 = 1;
const int32_t NC1HWC0_DIM_C0 = 4;
const int32_t NC1HWC0_DIM_H = 2;
const int32_t NC1HWC0_DIM_W = 3;

const int32_t NDC1HWC0_DIM_N = 0;
const int32_t NDC1HWC0_DIM_D = 1;
const int32_t NDC1HWC0_DIM_C1 = 2;
const int32_t NDC1HWC0_DIM_C0 = 5;
const int32_t NDC1HWC0_DIM_H = 3;
const int32_t NDC1HWC0_DIM_W = 4;

const int32_t C1HWNCoC0_DIM_C1 = 0;
const int32_t C1HWNCoC0_DIM_H = 1;
const int32_t C1HWNCoC0_DIM_W = 2;
const int32_t C1HWNCoC0_DIM_N = 3;
const int32_t C1HWNCoC0_DIM_Co = 4;
const int32_t C1HWNCoC0_DIM_C0 = 5;

const int32_t C1DHWNCoC0_DIM_C1 = 0;
const int32_t C1DHWNCoC0_DIM_D = 1;
const int32_t C1DHWNCoC0_DIM_H = 2;
const int32_t C1DHWNCoC0_DIM_W = 3;

const int32_t NHWC_DIM_N = 0;
const int32_t NHWC_DIM_H = 1;
const int32_t NHWC_DIM_W = 2;
const int32_t NHWC_DIM_C = 3;

const int32_t HWCN_DIM_H = 0;
const int32_t HWCN_DIM_W = 1;
const int32_t HWCN_DIM_C = 2;
const int32_t HWCN_DIM_N = 3;

const int32_t CHWN_DIM_C = 0;
const int32_t CHWN_DIM_H = 1;
const int32_t CHWN_DIM_W = 2;
const int32_t CHWN_DIM_N = 3;

const int32_t NDHWC_DIM_N = 0;
const int32_t NDHWC_DIM_D = 1;
const int32_t NDHWC_DIM_H = 2;
const int32_t NDHWC_DIM_W = 3;
const int32_t NDHWC_DIM_C = 4;
const uint32_t DIMENSION_NUM_FIVE = 5;

const int32_t NCDHW_DIM_N = 0;
const int32_t NCDHW_DIM_C = 1;
const int32_t NCDHW_DIM_D = 2;
const int32_t NCDHW_DIM_H = 3;
const int32_t NCDHW_DIM_W = 4;

const int32_t DHWCN_DIM_D = 0;
const int32_t DHWCN_DIM_H = 1;
const int32_t DHWCN_DIM_W = 2;
const int32_t DHWCN_DIM_C = 3;
const int32_t DHWCN_DIM_N = 4;

const int32_t DHWNC_DIM_D = 0;
const int32_t DHWNC_DIM_H = 1;
const int32_t DHWNC_DIM_W = 2;
const int32_t DHWNC_DIM_N = 3;
const int32_t DHWNC_DIM_C = 4;
}

using TensorPtr = std::shared_ptr<ge::GeTensor>;

class QuantUtilImpl {
 public:
  static Status BiasOptimizeByEdge(BiasOptimizeEdges &param, std::vector<ge::NodePtr> &fusion_nodes);
  static Status BiasOptimizeByEdge(ge::NodePtr &quant_node, BiasOptimizeEdges &param,
                                   std::vector<ge::NodePtr> &fusion_nodes);
  static Status BiasOptimizeByEdge(const QuantParam &quant_param, BiasOptimizeEdges &param,
                                   std::vector<ge::NodePtr> &fusion_nodes,
                                   WeightMode cube_type = WeightMode::RESERVED);
  static Status InsertFixpipeDequantScaleConvert(ge::InDataAnchorPtr deq_scale, std::vector<ge::NodePtr> &fusion_nodes);
  static Status InsertFixpipeDequantScaleConvert(const ge::InDataAnchorPtr &deq_scale, ge::InDataAnchorPtr &quant_offset,
                                                 std::vector<ge::NodePtr> &fusion_nodes);
  static Status InsertQuantScaleConvert(ge::InDataAnchorPtr &quant_scale, ge::InDataAnchorPtr &quant_offset,
                                        std::vector<ge::NodePtr> &fusion_nodes);
  static Status InsertRequantScaleConvert(ge::InDataAnchorPtr &req_scale, ge::InDataAnchorPtr &quant_offset,
                                          ge::InDataAnchorPtr &cube_bias, std::vector<ge::NodePtr> &fusion_nodes);

 private:
  static Status BiasOptimizeByEdgeCommon(const ge::NodePtr &quant_node, BiasOptimizeEdges &param,
                                         std::vector<ge::NodePtr> &fusion_nodes, WeightMode cube_type);
  static bool NeedBiasInput(const ge::InDataAnchorPtr &bias);
  static Status GetCoValueByWeight(ge::NodePtr &cube_node, size_t idx, std::vector<int64_t> &bias_shape);
  static Status PadShapeTo4Dim(const ge::Format &filter_format, const std::vector<int64_t> &filter_dims,
                               std::vector<int64_t> &filter_dims4_d);
  static int32_t GetAxisIndexByFormat(const ge::Format &format, const string &axis);
  static TensorPtr CreateBiasTensor(const std::vector<int64_t> &shape);
  static ge::NodePtr CreateBiasNode(std::shared_ptr<ge::ComputeGraph> &graph, const ge::GeTensorPtr &bias_ptr,
                                    const std::string &cube_node_name);
  static Status UpdateBiasOutputDesc(const ge::NodePtr &cube_node, const ge::GeShape &shape, ge::Format format,
                                     const uint32_t index);
  static Status UpdateCubeInputDesc(const ge::NodePtr &cube_node, const ge::GeShape &shape, const ge::Format &format,
                                    const uint32_t index);
  static Status CreateBiasInput(std::shared_ptr<ge::ComputeGraph> &graph, ge::NodePtr &cube_node,
                                const std::vector<int64_t> &shape, const size_t &bias_idx);
  static Status GetWeightConstNode(const ge::InDataAnchorPtr &weight, ge::NodePtr &weight_const_node,
                                   ge::NodePtr &ascend_weight_quant_node);
  static Status GetInputDescByAnchor(const ge::InDataAnchorPtr &in_data_anchor, ge::GeTensorDesc &tensor_desc);
  static void SetAttrsForBiasOptimizerOp(ge::OpDescPtr &op_desc, const ge::NodePtr &cube_node,
                                         const ge::NodePtr &ascend_weight_quant_node, const WeightMode cube_type);
  static Status SetQuantScaleAndOffset(const ge::NodePtr &quant_node, const BiasOptimizeEdges &param,
                                       ge::OpDescPtr &host_op_desc);
  static Status LinkBiasOptimizeHostOp(const ge::NodePtr &quant_node, const ge::NodePtr &weight_const_node,
                                       const BiasOptimizeEdges &param, const ge::NodePtr &host_op_node);
  static Status CreateBiasOptimizeHostCpuOp(std::shared_ptr<ge::ComputeGraph> &graph, const ge::NodePtr &quant_node,
                                            const BiasOptimizeEdges &param, const ge::NodePtr &weight_const_node,
                                            WeightMode cube_type, std::vector<ge::NodePtr> &fusion_nodes);
  static ge::OpDescPtr CreateDeqScaleHostOp(const std::string &op_name, const std::string &op_type,
                                            const ge::OpDescPtr &cube_node, size_t index);
  static bool IsNanoSoc();

  static bool IsSupportFixpipe();

  static ge::GeTensorPtr GetTensorByAnchor(const ge::InDataAnchorPtr &anchor);

  static const uint8_t *GetDataByAnchor(ge::InDataAnchorPtr &anchor);

  static uint64_t TransM1Scale(const float &src_value);

  static uint64_t SetM1OfQuant(const float &scale, const float &offset, const ge::DataType &data_type);

  static Status UpdateScalarInput(const float *quant_scale_data, const float *quant_offset_data,
                                  const ge::DataType &data_type, ge::GeTensorDescPtr &scale_tensor_desc,
                                  ge::GeTensorPtr &quant_op_tensor);

  static Status CreateQuantOp(const ge::NodePtr &cube_node, const ge::InDataAnchorPtr &quant_scale,
                              ge::GeTensorDescPtr scale_tensor_desc, ge::GeTensorPtr quant_op_tensor,
                              std::vector<ge::NodePtr> &fusion_nodes);

  static Status InsertFixpipeQuantScaleConvert(ge::InDataAnchorPtr &quant_scale, ge::InDataAnchorPtr &quant_offset,
                                               std::vector<ge::NodePtr> &fusion_nodes);

  static Status SetAttrForRequantHostCpuOp(ge::OpDescPtr &req_host_op_desc, const ge::GeTensorPtr &req_scale_tensor,
                                           ge::InDataAnchorPtr &quant_offset, ge::InDataAnchorPtr &cube_bias,
                                           int32_t &req_scale_size);

  static Status CreateRequantHostCpuOp(ge::InDataAnchorPtr &req_scale, ge::InDataAnchorPtr &cube_bias,
                                       ge::InDataAnchorPtr &quant_offset, std::vector<ge::NodePtr> &fusion_nodes);

  static Status InsertNotFixpipeRequantScaleConvert(ge::InDataAnchorPtr &req_scale, ge::InDataAnchorPtr &quant_offset,
                                                    ge::InDataAnchorPtr &cube_bias,
                                                    std::vector<ge::NodePtr> &fusion_nodes);
};
}  // namespace fe
#endif
