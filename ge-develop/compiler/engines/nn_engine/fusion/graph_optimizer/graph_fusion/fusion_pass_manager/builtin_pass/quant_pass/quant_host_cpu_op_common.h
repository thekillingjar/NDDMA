/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_QUANT_PASS_QUANT_HOST_CUP_OP_COMMON_H_
#define FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_QUANT_PASS_QUANT_HOST_CUP_OP_COMMON_H_

#include <string>
#include <vector>
#include <cmath>
#include "common/fe_log.h"
#include "graph/types.h"
#include "graph/compute_graph.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
const string CONCAT               = "Concat";
const string CONCATV2             = "ConcatV2";
/* Op type of new Host Cpu Op */
// ================= The following is for requant fusion pass ==================
const std::string QUANT_WEIGHT_ROLL_BACK = "QuantWeightRollBack";
const std::string QUANT_BIAS_ROLL_BACK = "QuantBiasRollBack";
const std::string QUANT_BIAS_OPTIMIZATION = "QuantBiasOptimization";
const std::string QUANT_BIAS_OPTIMIZATIONS16 = "QuantBiasOptimizationS16";

// ================= The following is for requant fusion pass ==================
const std::string REQUANT_HOST_CPU_OP = "RequantHostCpuOp";
const std::string REQUANT_HOST_CPU_OP_V2 = "RequantHostCpuOpV2";
const std::string REQUANT_HOST_CPU_OP_V2_RE = "RequantHostCpuOpV2Re";

// =========================Tensor Name ========================================
/* tensor name of filter and bias */
const std::string CUBE_FILTER = "cube_filter";

const std::string CUBE_BIAS = "cube_bias";

const std::string CUBE_OPTIMIZATION_BIAS = "cube_optimization_bias";
const std::string CUBE_OPTIMIZATION_FILTER = "cube_optimization_filter";
const std::string CUBE_QUANT_ROLL_BACK_OUTPUT = "cube_quant_roll_back_output";

/* Tensor Name */
const std::string REQUANT_HOST_CPU_OP_INPUT = "requant_input";
const std::string REQUANT_HOST_CPU_OP_OUTPUT = "requant_output";

/* Attribute Name */
const std::string QUANT_SCALE             = "quant_scale";
const std::string QUANT_SCALE_VEC         = "quant_scale_vec";
const std::string QUANT_OFFSET_VEC        = "quant_offset_vec";
const std::string DEQUANT_SCALE           = "dequant_scale";
const std::string ATTR_OFFSET_X           = "offset_x";
const std::string ATTR_OFFSET             = "offset";
const std::string ATTR_SCALE              = "scale";
const std::string ATTR_ORIGIN_SCALE       = "origin_scale";
const std::string ATTR_CUBE_OP_TYPE       = "cube_op_type";

const std::string ATTR_DEQUANT_NO_REQUANT = "dequant_no_requant";
const std::string ATTR_REQUANTED          = "_requanted";
const std::string ATTR_DEL_RELU           = "_need_delrelu_of_dequant";
const std::string ATTR_SQRT_MODE          = "sqrt_mode";
const std::string ATTR_RELU_FLAG          = "relu_flag";
const std::string ATTR_NEGATIVE_SLOPE     = "negative_slope";
const std::string ATTR_CIN_COUT_REVERSE   = "quant_cin_cout_reverse";
const std::string ATTR_BIAS_SIZE          = "bias_size";
const std::string ATTR_BIAS_VALUE         = "bias_value";
const std::string OUTPUTS16               = "output_s16";
const std::string DEQ_N_VALUE             = "deq_n_value";

/* Index of tensor */
const size_t DEQUANT_SCALE_INDEX_OF_DEQUANT_OP = 1;
const size_t DEQUANT_SCALE_INDEX_OF_REQUANT_OP = 0;
const int32_t OUT_W_DIM_VALUE = 1;
const int32_t LAST_AXIS_INDEX = 3;

const int SCALE_BASE = 2;
const int MIN_SCALE_EXPONENT = -14;
const uint32_t SCALE_BIT_OFFSET = 32;
const float MIN_SCALE_VALUE = std::pow(SCALE_BASE, MIN_SCALE_EXPONENT);
const float MAX_SCALE_VALUE = 65504.0;

constexpr const char* kAippQuantNeedAdjust = "_aipp_quant_need_adjust";
constexpr const char* kAippQuantMeanVec = "_aipp_quant_mean_vec";
constexpr const char* kQuantScaleRate = "_quant_scale_rate";
const std::vector<std::tuple<std::string, std::string, std::string>> kAippKey {
  {"mean_chn_0", "min_chn_0", "var_reci_chn_0"},
  {"mean_chn_1", "min_chn_1", "var_reci_chn_1"},
  {"mean_chn_2", "min_chn_2", "var_reci_chn_2"},
  {"mean_chn_3", "min_chn_3", "var_reci_chn_3"}
};
constexpr const int32_t kBaseOffset = 128;
constexpr const int32_t kMaxChnNum = 4;

#define GET_DEQUANT_SCALE_DEQ(dequant_scale_date) (((dequant_scale_date) & 0x00000000ffffffff))

#define GET_DEQUANT_N(dequant_scale_date) (((dequant_scale_date) & 0x000000ff00000000) >> 32)

#define GET_DEQUANT_OFFSET_W(dequant_scale_date) (((dequant_scale_date) & 0x0000ff0000000000) >> 40)

#define SET_REQUANT_N_FLAG_FALSE(requant_scale_date) (((requant_scale_date) & 0xffffffefffffffff))

#define SET_REQUANT_N_FLAG_TRUE(requant_scale_date) (((requant_scale_date) | 0x0000001000000000))

struct AippChnInfo {
  std::vector<int64_t> mean_chn_vec;
  std::vector<float> min_chn_vec;
  std::vector<float> var_reci_chn_vec;
};

void AippGetInt64Value(ge::NamedAttrs &aipp_attr, const std::string &key, int64_t &value);
void AippGetFloatValue(ge::NamedAttrs &aipp_attr, const std::string &key, float &value);
void AippGetFloatVecFirstValue(ge::NamedAttrs &aipp_attr, const std::string &key, float &value);

template <typename T>
inline void AippSetAttrValue(ge::NamedAttrs &aipp_attr, const std::string &key, T &value) {
  (void)aipp_attr.SetAttr(key, ge::GeAttrValue::CreateFrom<T>(value));
  return;
}

uint64_t GetHostCpuAtomicId();

Status CreateNewRequantHostCpuOp(const string &op_type, const ge::NodePtr &dequant_node, const float &scale_quant,
                                 ge::ComputeGraph &graph, vector<ge::NodePtr> &new_nodes);

Status PadShapeTo4Dim(const ge::Format &filter_format, const std::vector<int64_t> &filter_dims,
                      std::vector<int64_t> &filter_dims4_d);

bool IsValidConcatNode(const ge::NodePtr &concat_node);

bool IsEnableReluOfDequant(const ge::NodePtr &dequant_node);

bool CheckReluValid(const ge::NodePtr &node);

Status SetReluFlag(const ge::NodePtr &dequant_node);

Status SetReluFlagToDequant(ge::NodePtr &dequant_node);

bool CheckNeedRemoveRelu(const ge::NodePtr &node);

Status DealRelu(ge::ComputeGraph &graph, vector<ge::NodePtr> &relu_nodes, const float &scale_quant);

Status TagNodes(vector<ge::NodePtr> &quant_nodes, vector<ge::NodePtr> &dequant_nodes, const int &attr_value);

Status DealQauntNotRequant(const ge::NodePtr &quant_node, vector<ge::NodePtr> &fusion_nodes);
}  // namespace fe
#endif  // FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_QUANT_PASS_QUANT_HOST_CUP_OP_COMMON_H_
