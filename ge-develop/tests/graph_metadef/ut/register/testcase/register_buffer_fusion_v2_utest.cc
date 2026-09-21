/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <map>
#include <memory>
#include <cstdio>
#include "gtest/gtest.h"

#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/op_kernel_bin.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pattern.h"
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "framework/common/debug/ge_log.h"
#include "register/graph_optimizer/graph_fusion/connection_matrix.h"
#include "register/graph_optimizer/fusion_common/op_slice_info.h"
#include "runtime/kernel.h"

using namespace std;
using namespace domi;
using namespace fe;
using namespace ge;


namespace fe {
namespace buffer_fusion_reg_v2 {
static const string STREAM_LABEL = "_stream_label";
const std::string FE_IMPLY_TYPE = "_fe_imply_type";
static const uint32_t L2_MAXDATANUM = 8;
using L2FusionData_t = struct tag_l2_fusion_data {
  uint32_t l2Index;
  uint64_t l2Addr;
  uint64_t l2PageNum;
};
using L2FusionDataMap_t = std::map<uint64_t, L2FusionData_t>;

using fe_sm_desc_t = struct tag_fe_sm_desc {
  rtL2Ctrl_t l2ctrl;
  std::string node_name[L2_MAXDATANUM];
  uint8_t output_index[L2_MAXDATANUM];
};

using TaskL2FusionInfo_t = struct TagTaskL2FusionInfo {
  std::string node_name;
  fe_sm_desc_t l2_info;
  L2FusionDataMap_t input;
  L2FusionDataMap_t output;
  uint32_t is_used;
};
using L2FusionInfoPtr = std::shared_ptr<TaskL2FusionInfo_t>;
}

class TbeCommonRules2FusionPass : public BufferFusionPassBase {
 public:
  explicit TbeCommonRules2FusionPass() = default;

  ~TbeCommonRules2FusionPass() override = default;

 protected:

  /*
   * @brief:  define a common ub fusion pattern:
   * (StrideRead) -> Convolution -> (Dequant) -> Elewise*N -> Quant -> (StrideWrite)
   *
   * pattern limits:
   * 1. StrideRead, StrideWrite, Dequant are optional, Conv2D and Quant are required.
   * 2. Elewise supports LeakyRelu, Vadd, Relu, Relu6, Prelu, Add, Mul. The number of Elewise can be 0 to 5.
   * 3. There are two outputs from Dequant or Elewise, one is int8 or int4, the other is fp16.
   *
   *
   * fusion node: (StrideRead), Convolution, (AscendDequant), Elewise, AscendQuant,
   *
   * @return BufferFusionPattern: return all valid patterns.
   */
  vector<BufferFusionPattern *> DefinePatterns() override;

  /*
   * @brief: parse nodes matched in mapping and call DoFusion
   * @param [in] graph: original graph
   * @param [out] mapping: nodes matched by pattern
   * @return bool: fusion status ok or not.
   */
  Status GetFusionNodes(const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusion_nodes) override;


  Status PostFusion(const ge::NodePtr &fused_node) override {
    return FAILED;
  }

 private:

  static int CountOtherOutput(vector<ge::NodePtr> dequant_nodes, vector<ge::NodePtr> elem_wise_nodes);

  static bool JudgeElemShapeInScopeLessThanOutScope(const vector<ge::NodePtr> &pre_elemwise_nodes,
                                                    const vector<ge::NodePtr> &elemwise_nodes);
};

namespace {
const string PATTERN_STRIDEREAD = "strideRead";        // NOLINT
const string PATTERN_CONVOLUTION = "convolution";      // NOLINT
const string PATTERN_DEPTHWISECONV = "depthwiseconv";  // NOLINT
const string PATTERN_DEQUANT = "dequant";              // NOLINT
const string PATTERN_ELEMWISE = "elemWise";            // NOLINT
const string PATTERN_QUANT = "quant";                  // NOLINT
const string PATTERN_STRIDEWRITE = "strideWrite";      // NOLINT
const string PATTERN_OTHER_INPUT = "otherInput";       // NOLINT
const string PATTERN_OUTPUT = "output";                // NOLINT

const vector<string> ELEM_WISE_WHITE_LIST = {"Eltwise", "LeakyRelu", "Vadd", "Relu",
                                             "Relu6", "Relu6D", "PRelu",
                                             "Add", "Mul", "Softplus", "Sigmoid", "Mish",
                                             "Minimum", "Tanh", "Swish"};  // NOLINT

const int MAX_OP_COUNT = 20;
const int MAX_ELEMWISE_COUNT = 5;
const int INPUT_MAX_SIZE = 2;
const int kConvOutputMaxSize = 2;
}

#define UT_CHECK(cond, log_func, return_expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      return_expr;                            \
    }                                         \
  } while (0)

#define UT_CHECK_NOTNULL(val)                           \
  do {                                                  \
    if ((val) == nullptr) {                             \
      GE_LOGE("Parameter[%s] must not be null.", #val); \
      return fe::PARAM_INVALID;                         \
    }                                                   \
  } while (0)3

/*
* @brief:  define a common ub fusion pattern:
* (StrideRead) -> Convolution -> (Dequant) -> Elewise*N -> Quant -> (StrideWrite)
*
* pattern limits:
* 1. StrideRead, StrideWrite, Dequant are optional, Conv2D and Quant are required.
* 2. Elewise supports LeakyRelu, Vadd, Relu, Relu6, Prelu, Add, Mul. The number of Elewise can be 0 to 5.
* 3. There are two outputs from Dequant or Elewise, one is int8 or int4, the other is fp16.
*
*
* fusion node: (StrideRead), Convolution, (AscendDequant), Elewise, AscendQuant,
*
* @return BufferFusionPattern: return all valid patterns.
*/
vector<BufferFusionPattern *> TbeCommonRules2FusionPass::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;
  string pass_name = "TbeCommonRules2FusionPass";
  auto *pattern = new(std::nothrow) BufferFusionPattern(pass_name, MAX_OP_COUNT);
  UT_CHECK((pattern == nullptr),
           GE_LOGE("[SubGraphOpt][CommonRules2Fus][DefPtn] New an object failed."), return patterns);
  GELOGD("Start to define %s pass pattern.", pass_name.c_str());
  pattern->AddOpDesc(PATTERN_STRIDEREAD, {OP_PATTERN_STRIDED_READ}, TBE_PATTERN_NUM_NONE,
                     TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(PATTERN_CONVOLUTION, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(PATTERN_DEPTHWISECONV, {OP_PATTERN_DEPTHWISE_CONV}, TBE_PATTERN_NUM_NONE,
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(PATTERN_DEQUANT, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_NONE,
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(PATTERN_ELEMWISE, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_NONE,
                 MAX_ELEMWISE_COUNT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(PATTERN_QUANT, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(PATTERN_STRIDEWRITE, {OP_PATTERN_STRIDED_WRITE}, TBE_PATTERN_NUM_NONE,
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .SetHead({PATTERN_STRIDEREAD, PATTERN_CONVOLUTION, PATTERN_DEPTHWISECONV})
      .SetOutputs(PATTERN_STRIDEREAD, {PATTERN_CONVOLUTION, PATTERN_DEPTHWISECONV})
      .SetOutputs(PATTERN_CONVOLUTION, {PATTERN_DEQUANT}, TBE_OUTPUT_BRANCH_SINGLE, true, true)
      .SetOutputs(PATTERN_DEPTHWISECONV, {PATTERN_DEQUANT}, TBE_OUTPUT_BRANCH_SINGLE, true, true)
      .SetOutputs(PATTERN_DEQUANT, {PATTERN_ELEMWISE}, TBE_OUTPUT_BRANCH_SINGLE, true, true)
      .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_DEQUANT})
      .SetOutputs(PATTERN_ELEMWISE, {PATTERN_QUANT}, TBE_OUTPUT_BRANCH_SINGLE, true, true)
      .SetOutputs(PATTERN_QUANT, {PATTERN_STRIDEWRITE}, TBE_OUTPUT_BRANCH_SINGLE, false, true);
  patterns.push_back(pattern);
  GELOGD("End to define %s pass pattern.", pass_name.c_str());

  return patterns;
}

int TbeCommonRules2FusionPass::CountOtherOutput(vector<ge::NodePtr> dequant_nodes,
                                                vector<ge::NodePtr> elem_wise_nodes) {
  int other_out_count = 0;
  // count EltWise op other output
  for (const auto &elem_wise_node : elem_wise_nodes) {
    if (elem_wise_node->GetOutDataNodes().empty()) {
      continue;
    }
    int other_elt_wise_out = (int) (elem_wise_node->GetOutDataNodes().size() - 1);
    other_out_count += other_elt_wise_out;
  }

  // count Dequant op other output
  if (!dequant_nodes.empty()) {
    int other_dequant_out = 0;
    if (dequant_nodes[0]->GetOutDataNodes().empty()) {
      other_dequant_out = 0;
    } else {
      other_dequant_out = static_cast<int>(dequant_nodes[0]->GetOutDataNodes().size() - 1);
    }
    other_out_count += other_dequant_out;
  }
  return other_out_count;
}

bool TbeCommonRules2FusionPass::JudgeElemShapeInScopeLessThanOutScope(const vector<ge::NodePtr> &pre_elemwise_nodes,
                                                                      const vector<ge::NodePtr> &elemwise_nodes) {
  if (pre_elemwise_nodes.empty()) {
    return false;
  }
  ge::NodePtr cur_node = pre_elemwise_nodes[0];
  for (auto &elemwise_node: elemwise_nodes) {
    ge::NodePtr pre_node = cur_node;
    cur_node = elemwise_node;
    if (cur_node->GetOpDesc()->GetInputsSize() != INPUT_MAX_SIZE) {
      continue;
    }

    if ((cur_node->GetInDataAnchor(0)->GetPeerOutAnchor() == nullptr) ||
        (cur_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode() == nullptr)) {
      return false;
    }
    auto cur_node_input0 = cur_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();

    vector<int64_t> in_scope_dims;
    vector<int64_t> out_scope_dims;
    if (cur_node_input0->GetName() == pre_node->GetOpDesc()->GetName()) {
      in_scope_dims = cur_node->GetOpDesc()->MutableInputDesc(0)->MutableShape().GetDims();
      out_scope_dims = cur_node->GetOpDesc()->MutableInputDesc(1)->MutableShape().GetDims();
    } else {
      in_scope_dims = cur_node->GetOpDesc()->MutableInputDesc(1)->MutableShape().GetDims();
      out_scope_dims = cur_node->GetOpDesc()->MutableInputDesc(0)->MutableShape().GetDims();
    }
    if (in_scope_dims.size() != out_scope_dims.size()) {
      GELOGD("Elem_wise[node: %s] : the number of input's dims is not equal. in_scope_dims: %zu, out_scope_dims: %zu",
             cur_node->GetName().c_str(), in_scope_dims.size(), out_scope_dims.size());
      return false;
    } else {
      for (size_t i = 0; i < in_scope_dims.size(); i++) {
        if (in_scope_dims[i] < out_scope_dims[i]) {
          GELOGD("Elem_wise[node: %s] dims[%zu]: the value of in_scope is less than out_scope. in_scope : %ld,"
                 " out_scope : %ld", cur_node->GetName().c_str(), i, in_scope_dims[i], out_scope_dims[i]);
          return true;
        }
      }
    }
  }
  return false;
}

/*
* @brief: parse nodes matched in mapping and call DoFusion
* @param [in] graph: original graph
* @param [out] mapping: nodes matched by pattern
* @return bool: fusion status ok or not.
*/
Status TbeCommonRules2FusionPass::GetFusionNodes(const BufferFusionMapping &mapping,
                                                 vector<ge::NodePtr> &fusion_nodes) {
  fusion_nodes = GetMatchedNodes(mapping);
  vector<ge::NodePtr> output_nodes = GetMatchedNodesByDescName(TBE_PATTERN_OUTPUT_NODE, mapping);
  vector<ge::NodePtr> conv_nodes = GetMatchedNodesByDescName(PATTERN_CONVOLUTION, mapping);
  vector<ge::NodePtr> depthwise_nodes = GetMatchedNodesByDescName(PATTERN_DEPTHWISECONV, mapping);
  vector<ge::NodePtr> elem_wise_nodes = GetMatchedNodesByDescName(PATTERN_ELEMWISE, mapping);
  vector<ge::NodePtr> dequant_nodes = GetMatchedNodesByDescName(PATTERN_DEQUANT, mapping);
  vector<ge::NodePtr> quant_nodes = GetMatchedNodesByDescName(PATTERN_QUANT, mapping);
  vector<ge::NodePtr> stride_write_nodes = GetMatchedNodesByDescName(PATTERN_STRIDEWRITE, mapping);

  bool conv_depth_size = conv_nodes.size() == 1 || depthwise_nodes.size() == 1;
  if (!conv_depth_size) {
    GELOGD("There is no conv and depthwise in TbeCommonRules2FusionPass");
    fusion_nodes.clear();
    return ge::GRAPH_SUCCESS;
  }
  vector<ge::NodePtr> conv_depthwise_nodes = conv_nodes.size() == 1 ? conv_nodes : depthwise_nodes;

  size_t conv_output_size = conv_depthwise_nodes[0]->GetOutDataNodes().size();
  // conv outputs size is more than 2, skip fused
  if (conv_output_size > kConvOutputMaxSize) {
    GELOGD("node: %s, outputs is more than 2, size is: %zu.",
           conv_depthwise_nodes[0]->GetName().c_str(), conv_output_size);
    fusion_nodes.clear();
    return ge::GRAPH_SUCCESS;
  }

  // the output_data can't be fused
  for (const auto &outputnode : output_nodes) {
    auto node_ptr = find(fusion_nodes.begin(), fusion_nodes.end(), outputnode);
    if (node_ptr != fusion_nodes.end()) {
      fusion_nodes.erase(node_ptr);
    }
  }

  // this pattern only support one other output from dequant node or elem_wise node
  int other_out_count = CountOtherOutput(dequant_nodes, elem_wise_nodes);
  bool cond_other_out_count = (conv_output_size == 1 && other_out_count != 1) ||
                              (conv_output_size == kConvOutputMaxSize && other_out_count != 0);
  if (cond_other_out_count) {
    GELOGD("The number of other output from EltWise or Dequant is %d, skip fusion.", other_out_count);
    fusion_nodes.clear();
    return ge::GRAPH_SUCCESS;
  }

  // if elewise has 2 input and inscope's shape less than outscope's shape, skip fusion
  bool dequant_flag = !dequant_nodes.empty() &&
                      JudgeElemShapeInScopeLessThanOutScope(dequant_nodes, elem_wise_nodes);
  if (dequant_flag) {
    GELOGD("dequant_nodes exist, Elemwise node has 2 inputs and in scope shape is less than outscope");
    fusion_nodes.clear();
    return ge::GRAPH_SUCCESS;
  }
  bool no_dequant_flag = dequant_nodes.empty() &&
                         JudgeElemShapeInScopeLessThanOutScope(conv_depthwise_nodes, elem_wise_nodes);
  if (no_dequant_flag) {
    GELOGD("no dequant_nodes, Elemwise node has 2 inputs and in scope shape is less than outscope");
    fusion_nodes.clear();
    return ge::GRAPH_SUCCESS;
  }

  // check whether the EltWise op is in the whitelist or inputsizes less then 3(only support single or double in)
  for (const auto &elem_wise_node : elem_wise_nodes) {
    bool support_flag = find(ELEM_WISE_WHITE_LIST.begin(), ELEM_WISE_WHITE_LIST.end(), elem_wise_node->GetType()) ==
                        ELEM_WISE_WHITE_LIST.end() ||
                        elem_wise_node->GetOpDesc()->GetInputsSize() > INPUT_MAX_SIZE;
    if (support_flag) {
      fusion_nodes.clear();
      GELOGD("Eltwise op[%s] type[%s] is not supported for this ub fusion pass, skip fusion.",
             elem_wise_node->GetName().c_str(), elem_wise_node->GetType().c_str());
      return ge::GRAPH_SUCCESS;
    }
  }

  // if stride_write is the last node, check whether quant node has multi outputs
  bool quant_node_flag = quant_nodes[0]->GetOutDataNodes().size() > 1 && !stride_write_nodes.empty();
  if (quant_node_flag) {
    auto node_ptr = find(fusion_nodes.begin(), fusion_nodes.end(), stride_write_nodes[0]);
    if (node_ptr != fusion_nodes.end()) {
      fusion_nodes.erase(node_ptr);
    }
    GELOGD("Quant is not the last node of the matched pattern, \
            but has multi outpts, erase last node stride_write.");
  }
  return ge::GRAPH_SUCCESS;
}

static const char PATTERN_STRIDED_READ[] = "stridedread";
static const char PATTERN_CONV[] = "convolution";
static const char PATTERN_STRIDED_WRITE[] = "stridedwrite";
static const int FUSION_OP_NUM_MAX = 10;

class ConveragePass : public BufferFusionPassBase {
 public:
  explicit ConveragePass() {}

  ~ConveragePass() override {}

 protected:

  /*
  * @brief:  define common rules0 ops fusion pattern
  *
  *   (StrideRead) + conv2_d + (dequant) + ele-wise*N + (quant) + (StrideWrite)
  *   restriction: 1.each node must be single output and single reference
  *                2.the range of N is 0 to 5
  *                3.allow multiple input, but only one input can be fusion
  *
  * @return BufferFusionPattern: return all valid patterns.
  */
  vector<BufferFusionPattern *> DefinePatterns() override {
    vector<BufferFusionPattern *> patterns;
    string pass_name = "ConveragePass";
    BufferFusionPattern *pattern = new(std::nothrow) BufferFusionPattern(pass_name, 10);
    UT_CHECK((pattern == nullptr),
             GE_LOGE("[SubGraphOpt][CommonRules0Fus][DefPtn] New an object failed."),
             return patterns);
    GELOGD("Start to define %s pass pattern.", pass_name.c_str());
    // define pattern rules
    pattern->AddOpDesc("", {OP_PATTERN_STRIDED_READ}, TBE_PATTERN_NUM_NONE,
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);

    pattern->AddOpDesc("test", {OP_PATTERN_STRIDED_READ}, TBE_PATTERN_NUM_DEFAULT,
                       TBE_PATTERN_NUM_NONE, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);

    pattern->AddOpDesc("test", {OP_PATTERN_STRIDED_READ}, TBE_PATTERN_NUM_NONE,
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);
    pattern->AddOpDesc("test", {OP_PATTERN_STRIDED_READ}, TBE_PATTERN_NUM_NONE,
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);
    pattern->AddOpDesc("head1", {OP_PATTERN_STRIDED_READ}, 2,
                       3, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);
    pattern->AddOpDesc("head2", {OP_PATTERN_STRIDED_READ}, 1,
                       1, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);

    pattern->AddOpDesc(PATTERN_CONV, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT,
                       TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
        .AddOpDesc(PATTERN_DEPTHWISECONV, {OP_PATTERN_DEPTHWISE_CONV}, TBE_PATTERN_NUM_NONE,
                   TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);
    pattern->AddOpDesc(PATTERN_STRIDED_READ, {OP_PATTERN_STRIDED_READ}, TBE_PATTERN_NUM_NONE,
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE);

    pattern->SetOutputs("", {PATTERN_CONV, PATTERN_DEPTHWISECONV});
    pattern->SetOutputs("1", {PATTERN_CONV, PATTERN_DEPTHWISECONV});
    pattern->SetOutputs(PATTERN_STRIDED_READ, {"1", PATTERN_DEPTHWISECONV});
    pattern->SetOutputs(PATTERN_STRIDED_READ, {PATTERN_STRIDED_READ});
    pattern->SetOutputs(PATTERN_STRIDED_READ, {PATTERN_CONV, PATTERN_DEPTHWISECONV});
    pattern->SetOutputs(PATTERN_STRIDED_READ, {PATTERN_CONV, PATTERN_DEPTHWISECONV});


    vector<string> heads;
    pattern->SetHead(heads);

    heads = {""};
    pattern->SetHead(heads);

    heads = {"head1"};
    pattern->SetHead(heads);

    heads = {PATTERN_CONV};
    pattern->SetHead(heads);

    heads = {PATTERN_CONV, "head2"};
    pattern->SetHead(heads);

    auto conv_desc = pattern->GetOpDesc(PATTERN_CONV);
    pattern->UpdateSkipStatus(conv_desc);

    pattern->GetOpDescs();
  }
};

class TbeCommonRules0FusionPass : public BufferFusionPassBase {
 public:
  explicit TbeCommonRules0FusionPass() {}

  ~TbeCommonRules0FusionPass() override {}

 protected:

  /*
  * @brief:  define common rules0 ops fusion pattern
  *
  *   (StrideRead) + conv2_d + (dequant) + ele-wise*N + (quant) + (StrideWrite)
  *   restriction: 1.each node must be single output and single reference
  *                2.the range of N is 0 to 5
  *                3.allow multiple input, but only one input can be fusion
  *
  * @return BufferFusionPattern: return all valid patterns.
  */
  vector<BufferFusionPattern *> DefinePatterns() override;

  /*
   * @brief: parse nodes matched in mapping and call DoFusion
   * @param [in] graph: original graph
   * @param [out] mapping: nodes matched by pattern
   * @return bool: fusion status ok or not.
   */
  Status GetFusionNodes(const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusion_nodes) override;

 private:

  static bool DealWithSameInAndOutScopeDimSize(const vector<int64_t> &in_scope_dims,
                                               const vector<int64_t> &out_scope_dims,
                                               const vector<ge::NodePtr> &elemwise_nodes,
                                               const ge::NodePtr &cur_node, const size_t &i,
                                               vector<ge::NodePtr> &fusion_node);

  static bool JudgeElemShapeInScopeLessThanOutScope(const vector<ge::NodePtr> &pre_elemwise_nodes,
                                                    const vector<ge::NodePtr> &elemwise_nodes,
                                                    vector<ge::NodePtr> &fusion_nodes);

  static bool IsInBlackListOfOpPatternElemwise(vector<ge::NodePtr> &elemwise_nodes, ge::NodePtr &node_ptr);
};

namespace {

// white list of OP_PATTERN_ELEMWISE
static const vector<string> WHITELIST_OF_OP_PATTERN_ELEMWISE = {
    "Eltwise", "LeakyRelu", "Vadd", "Relu", "Relu6", "Relu6D",
    "PRelu", "Add", "Mul", "Softplus", "Sigmoid", "Mish", "Minimum",
    "Tanh", "Swish"};
// black list of OP_PATTERN_ELEMWISE
static const vector<string> BLACKLIST_OF_OP_PATTERN_ELEMWISE = {
    "ReluGradV2"};
}

/*
 * @brief:  define common rules0 ops fusion pattern
 *
 *   (StrideRead) + conv2_d + (dequant) + ele-wise*N + (quant) + (StrideWrite)
 *   restriction: 1.each node must be single output and single reference
 *                2.the range of N is 0 to 5
 *                3.allow multiple input, but only one input can be fusion
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *> TbeCommonRules0FusionPass::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;
  string pass_name = "TbeCommonRules0FusionPass";
  BufferFusionPattern *pattern = new(std::nothrow) BufferFusionPattern(pass_name, FUSION_OP_NUM_MAX);
  UT_CHECK((pattern == nullptr),
           GE_LOGE("[SubGraphOpt][CommonRules0Fus][DefPtn] New an object failed."),
           return patterns);
  GELOGD("Start to define %s pass pattern.", pass_name.c_str());
  // define pattern rules
  pattern->AddOpDesc(PATTERN_STRIDED_READ, {OP_PATTERN_STRIDED_READ}, TBE_PATTERN_NUM_NONE,
                     TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(PATTERN_CONV, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(PATTERN_DEPTHWISECONV, {OP_PATTERN_DEPTHWISE_CONV}, TBE_PATTERN_NUM_NONE,
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(PATTERN_DEQUANT, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(PATTERN_ELEMWISE, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_MAX,
                 TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(PATTERN_QUANT, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(PATTERN_STRIDED_WRITE, {OP_PATTERN_STRIDED_WRITE}, TBE_PATTERN_NUM_NONE,
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .SetHead({PATTERN_STRIDED_READ, PATTERN_CONV, PATTERN_DEPTHWISECONV})
      .SetOutputs(PATTERN_STRIDED_READ, {PATTERN_CONV, PATTERN_DEPTHWISECONV})
      .SetOutputs(PATTERN_CONV, {PATTERN_DEQUANT}, TBE_OUTPUT_BRANCH_SINGLE, true)
      .SetOutputs(PATTERN_DEPTHWISECONV, {PATTERN_DEQUANT}, TBE_OUTPUT_BRANCH_SINGLE, true)
      .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_DEQUANT})
      .SetOutputs(PATTERN_DEQUANT, {PATTERN_ELEMWISE}, TBE_OUTPUT_BRANCH_SINGLE, true)
      .SetOutputs(PATTERN_ELEMWISE, {PATTERN_QUANT}, TBE_OUTPUT_BRANCH_SINGLE, true)
      .SetOutputs(PATTERN_QUANT, {PATTERN_STRIDED_WRITE});

  patterns.push_back(pattern);

  GELOGD("End to define %s pass pattern.", pass_name.c_str());
  return patterns;
}

static void DelNotMatchNodesFromFusionNodes(ge::NodePtr node_ptr, vector<ge::NodePtr> &fusion_nodes) {
  auto node = find(fusion_nodes.begin(), fusion_nodes.end(), node_ptr);
  if (node != fusion_nodes.end()) {
    fusion_nodes.erase(node);
  } else {
    return;
  }

  auto curr_nodes = node_ptr->GetOutDataNodes();
  if (curr_nodes.size() != 1) {
    return;
  } else {
    DelNotMatchNodesFromFusionNodes(curr_nodes.at(0), fusion_nodes);
  }
  return;
}

static bool IsInWhiteListOfOpPatternElemwise(vector<ge::NodePtr> &elemwise_nodes, ge::NodePtr &node_ptr) {
  for (auto &elemwise_node : elemwise_nodes) {
    string elemwise_type = elemwise_node->GetType();
    auto op_type =
        find(WHITELIST_OF_OP_PATTERN_ELEMWISE.begin(), WHITELIST_OF_OP_PATTERN_ELEMWISE.end(), elemwise_type);
    if (op_type == WHITELIST_OF_OP_PATTERN_ELEMWISE.end()) {
      GELOGD("node:%s[type:%s] not in elemwise white_list.",
             elemwise_node->GetName().c_str(), elemwise_type.c_str());
      node_ptr = elemwise_node;
      return false;
    }
  }
  return true;
}

bool TbeCommonRules0FusionPass::IsInBlackListOfOpPatternElemwise(vector<ge::NodePtr> &elemwise_nodes,
                                                                 ge::NodePtr &node_ptr) {
  for (auto &elemwise_node : elemwise_nodes) {
    string elemwise_type = elemwise_node->GetType();
    auto op_type =
        find(BLACKLIST_OF_OP_PATTERN_ELEMWISE.begin(), BLACKLIST_OF_OP_PATTERN_ELEMWISE.end(), elemwise_type);
    if (op_type != BLACKLIST_OF_OP_PATTERN_ELEMWISE.end()) {
      GELOGD("node:%s[type:%s] in elemwise black_list.", elemwise_node->GetName().c_str(), elemwise_type.c_str());
      node_ptr = elemwise_node;
      return true;
    }
  }
  return false;
}

static void CheckElewiseInputSize(vector<ge::NodePtr> &elemwise_nodes, vector<ge::NodePtr> &fusion_nodes) {
  for (auto elemwise_node : elemwise_nodes) {
    if (elemwise_node->GetOpDesc()->GetInputsSize() > INPUT_MAX_SIZE) {
      DelNotMatchNodesFromFusionNodes(elemwise_node, fusion_nodes);
      return;
    }
  }
}

bool TbeCommonRules0FusionPass::DealWithSameInAndOutScopeDimSize(const vector<int64_t> &in_scope_dims,
                                                                 const vector<int64_t> &out_scope_dims,
                                                                 const vector<ge::NodePtr> &elemwise_nodes,
                                                                 const ge::NodePtr &cur_node, const size_t &i,
                                                                 vector<ge::NodePtr> &fusion_nodes) {
  for (size_t j = 0; j < in_scope_dims.size(); j++) {
    if (in_scope_dims[j] < out_scope_dims[j]) {
      GELOGD("Elem_wise[node: %s] dims[%zu] : the value of in_scope is less than out_scope. in_scope : %ld,"
             " out_scope : %ld", cur_node->GetName().c_str(), j, in_scope_dims[j], out_scope_dims[j]);
      vector<ge::NodePtr> new_elemwise_nodes;
      for (size_t z = i; z < elemwise_nodes.size(); z++) {
        new_elemwise_nodes.push_back(elemwise_nodes[z]);
      }
      for (auto new_elemwise_node : new_elemwise_nodes) {
        DelNotMatchNodesFromFusionNodes(new_elemwise_node, fusion_nodes);
      }
      return true;
    }
  }
  return false;
}

bool TbeCommonRules0FusionPass::JudgeElemShapeInScopeLessThanOutScope(const vector<ge::NodePtr> &pre_elemwise_nodes,
                                                                      const vector<ge::NodePtr> &elemwise_nodes,
                                                                      vector<ge::NodePtr> &fusion_nodes) {
  if (pre_elemwise_nodes.empty()) {
    return false;
  }
  ge::NodePtr cur_node = pre_elemwise_nodes[0];
  for (size_t i = 0; i < elemwise_nodes.size(); i++) {
    ge::NodePtr elemwise_node = elemwise_nodes[i];
    ge::NodePtr pre_node = cur_node;
    cur_node = elemwise_node;
    if (cur_node->GetOpDesc()->GetInputsSize() != INPUT_MAX_SIZE) {
      continue;
    }
    auto peerOutAnchor = cur_node->GetInDataAnchor(0)->GetPeerOutAnchor();
    if (peerOutAnchor == nullptr) {
      GELOGD("node[%s]'s first peer in anchor is null", cur_node->GetName().c_str());
      continue;
    }
    auto cur_node_input0 = peerOutAnchor->GetOwnerNode();
    vector<int64_t> in_scope_dims;
    vector<int64_t> out_scope_dims;
    if (cur_node_input0->GetName() == pre_node->GetOpDesc()->GetName()) {
      in_scope_dims = cur_node->GetOpDesc()->MutableInputDesc(0)->MutableShape().GetDims();
      out_scope_dims = cur_node->GetOpDesc()->MutableInputDesc(1)->MutableShape().GetDims();
    } else {
      in_scope_dims = cur_node->GetOpDesc()->MutableInputDesc(1)->MutableShape().GetDims();
      out_scope_dims = cur_node->GetOpDesc()->MutableInputDesc(0)->MutableShape().GetDims();
    }
    if (in_scope_dims.size() != out_scope_dims.size()) {
      GELOGD("Elem_wise[node: %s] : the number of input's dims is not equal. in_scope : %zu, out_scope : %zu",
             cur_node->GetName().c_str(), in_scope_dims.size(), out_scope_dims.size());
      return false;
    } else {
      if (DealWithSameInAndOutScopeDimSize(in_scope_dims, out_scope_dims, elemwise_nodes, cur_node, i, fusion_nodes)) {
        return true;
      }
    }
  }
  return false;
}

static void DelNotMatchNodes(vector<ge::NodePtr> &elemwise_nodes, vector<ge::NodePtr> &fusion_nodes) {
  if (!elemwise_nodes.empty()) {
    ge::NodePtr node = nullptr;
    if (!IsInWhiteListOfOpPatternElemwise(elemwise_nodes, node)) {
      DelNotMatchNodesFromFusionNodes(node, fusion_nodes);
    }
  }
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status TbeCommonRules0FusionPass::GetFusionNodes(const BufferFusionMapping &mapping,
                                                 vector<ge::NodePtr> &fusion_nodes) {
  GELOGD("Begin to do TbeCommonRules0FusionPass!");
  fusion_nodes = GetMatchedNodes(mapping);

  vector<ge::NodePtr> elemwise_nodes = GetMatchedNodesByDescName(PATTERN_ELEMWISE, mapping);
  // elewise only support single in or double in
  if (!elemwise_nodes.empty()) {
    CheckElewiseInputSize(elemwise_nodes, fusion_nodes);
  }

  vector<ge::NodePtr> conv_nodes = GetMatchedNodesByDescName(PATTERN_CONV, mapping);
  vector<ge::NodePtr> depthwise_nodes = GetMatchedNodesByDescName(PATTERN_DEPTHWISECONV, mapping);
  bool conv_depth_size = conv_nodes.size() == 1 || depthwise_nodes.size() == 1;
  if (!conv_depth_size) {
    GELOGD("There is no conv and depthwise in TbeCommonRules0FusionPass");
    fusion_nodes.clear();
    return ge::GRAPH_SUCCESS;
  }
  vector<ge::NodePtr> conv_depthwise_nodes = conv_nodes.size() == 1 ? conv_nodes : depthwise_nodes;
  vector<ge::NodePtr> dequant_nodes = GetMatchedNodesByDescName(PATTERN_DEQUANT, mapping);

  // if elewise has 2 input and inscope's shape less than outscope's shape, skip fusion
  if (!dequant_nodes.empty()) {
    if (JudgeElemShapeInScopeLessThanOutScope(dequant_nodes, elemwise_nodes, fusion_nodes)) {
      GELOGD("dequant_nodes exist, Elemwise node has 2 inputs and in scope shape is less than outscope, try to fuse"
             " before elemwise nodes");
      return ge::GRAPH_SUCCESS;
    }
  } else {
    if (JudgeElemShapeInScopeLessThanOutScope(conv_depthwise_nodes, elemwise_nodes, fusion_nodes)) {
      GELOGD("no dequant_nodes, Elemwise node has 2 inputs and in scope shape is less than outscope, try to fuse"
             " before elemwise nodes");
      return ge::GRAPH_SUCCESS;
    }
  }
  // elewise is in the blacklist, skip fusion
  if (!elemwise_nodes.empty()) {
    ge::NodePtr node = nullptr;
    if (IsInBlackListOfOpPatternElemwise(elemwise_nodes, node)) {
      GELOGD("node is in elemwise black_list, skip ub fusion!");
      fusion_nodes.clear();
      return ge::GRAPH_SUCCESS;
    }
  }

  // in conv2_d+elewise(1~3) pattern, elewise has no restrictions,
  // if nums of elewise more then 3 and either one is not in the whitelist, skip fusion
  bool ret = (fusion_nodes.size() == (elemwise_nodes.size() + conv_depthwise_nodes.size())) &&
             (conv_depthwise_nodes.size() == 1) && !elemwise_nodes.empty();
  if (ret) {
    if (elemwise_nodes.size() <= 3) {
      return ge::GRAPH_SUCCESS;
    } else {
      ge::NodePtr node = nullptr;
      if (!IsInWhiteListOfOpPatternElemwise(elemwise_nodes, node)) {
        fusion_nodes.clear();
      }
      return ge::GRAPH_SUCCESS;
    }
  }

  DelNotMatchNodes(elemwise_nodes, fusion_nodes);

  if (fusion_nodes.size() == 1) {
    fusion_nodes.clear();
  }
  GELOGD("End to do TbeCommonRules0FusionPass!");
  return ge::GRAPH_SUCCESS;
}

class UTestBufferFusionPassReg : public testing::Test {
 public:

 protected:

  void SetUp() {

  }

  void TearDown() {
  }
};

using BufferFusionFn = BufferFusionPassBase *(*)();

class TestBufferFusionPass : public BufferFusionPassBase {
 protected:

  vector<BufferFusionPattern *> DefinePatterns() override {
    return {};
  };
};

fe::BufferFusionPassBase *BufferFunc1() {
  auto ret = new(std::nothrow) TestBufferFusionPass();
  ret->SetName("1");
  return ret;
}

fe::BufferFusionPassBase *BufferFunc2() {
  auto ret = new(std::nothrow) TestBufferFusionPass();
  ret->SetName("2");
  return ret;
}

fe::BufferFusionPassBase *BufferFunc3() {
  auto ret = new(std::nothrow) TestBufferFusionPass();
  ret->SetName("3");
  return ret;
}

TEST_F(UTestBufferFusionPassReg, test_case_01) {
  auto pass_desc = BufferFusionPassRegistry::GetInstance().GetPassDesc(CUSTOM_AI_CORE_BUFFER_FUSION_PASS);
  auto init_size = pass_desc.size();
  BufferFusionPassRegistry::GetInstance().RegisterPass(
      CUSTOM_AI_CORE_BUFFER_FUSION_PASS, "CUSTOM_PASS1", BufferFunc1, 0);
  BufferFusionPassRegistry::GetInstance().RegisterPass(
      CUSTOM_AI_CORE_BUFFER_FUSION_PASS, "CUSTOM_PASS2", BufferFunc2, 1);
  BufferFusionPassRegistry::GetInstance().RegisterPass(
      CUSTOM_AI_CORE_BUFFER_FUSION_PASS, "CUSTOM_PASS3", BufferFunc3, 0xffffffff);

  pass_desc = BufferFusionPassRegistry::GetInstance().GetPassDesc(CUSTOM_AI_CORE_BUFFER_FUSION_PASS);
  EXPECT_EQ(pass_desc.size(), init_size + 3);
  EXPECT_EQ(pass_desc["CUSTOM_PASS1"].attr, 0);
  EXPECT_EQ(pass_desc["CUSTOM_PASS2"].attr, 1);
  EXPECT_EQ(pass_desc["CUSTOM_PASS3"].attr, 0xffffffff);

  auto create_fn1 = pass_desc["CUSTOM_PASS1"].create_fn;
  auto buffer_fusion_pass_base_ptr1 = std::unique_ptr<BufferFusionPassBase>(create_fn1());
  EXPECT_EQ(buffer_fusion_pass_base_ptr1->GetName(), "1");

  auto create_fn2 = pass_desc["CUSTOM_PASS2"].create_fn;
  auto buffer_fusion_pass_base_ptr2 = std::unique_ptr<BufferFusionPassBase>(create_fn2());
  EXPECT_EQ(buffer_fusion_pass_base_ptr2->GetName(), "2");

  auto create_fn3 = pass_desc["CUSTOM_PASS3"].create_fn;
  auto buffer_fusion_pass_base_ptr3 = std::unique_ptr<BufferFusionPassBase>(create_fn3());
  EXPECT_EQ(buffer_fusion_pass_base_ptr3->GetName(), "3");

  pass_desc = BufferFusionPassRegistry::GetInstance().GetPassDesc(BUILT_IN_AI_CORE_BUFFER_FUSION_PASS);
  init_size = pass_desc.size();
  BufferFusionPassRegistry::GetInstance().RegisterPass(
      BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, "BUILT_IN_PASS1", BufferFunc1, 0);
  BufferFusionPassRegistry::GetInstance().RegisterPass(
      BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, "BUILT_IN_PASS2", BufferFunc2, 1);
  BufferFusionPassRegistry::GetInstance().RegisterPass(
      BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, "BUILT_IN_PASS3", BufferFunc3, 0xffffffff);

  auto pass_desc2 = BufferFusionPassRegistry::GetInstance().GetPassDesc(BUILT_IN_AI_CORE_BUFFER_FUSION_PASS);
  EXPECT_EQ(pass_desc2.size(), init_size + 3);
  EXPECT_EQ(pass_desc2["BUILT_IN_PASS1"].attr, 0);
  EXPECT_EQ(pass_desc2["BUILT_IN_PASS2"].attr, 1);
  EXPECT_EQ(pass_desc2["BUILT_IN_PASS3"].attr, 0xffffffff);

  create_fn1 = pass_desc2["BUILT_IN_PASS1"].create_fn;
  auto buffer_fusion_pass_base_ptr4 = std::unique_ptr<BufferFusionPassBase>(create_fn1());
  EXPECT_EQ(buffer_fusion_pass_base_ptr4->GetName(), "1");

  create_fn2 = pass_desc2["BUILT_IN_PASS2"].create_fn;
  auto buffer_fusion_pass_base_ptr5 = std::unique_ptr<BufferFusionPassBase>(create_fn2());
  EXPECT_EQ(buffer_fusion_pass_base_ptr5->GetName(), "2");

  create_fn3 = pass_desc2["BUILT_IN_PASS3"].create_fn;
  auto buffer_fusion_pass_base_ptr6 = std::unique_ptr<BufferFusionPassBase>(create_fn3());
  EXPECT_EQ(buffer_fusion_pass_base_ptr6->GetName(), "3");
}


TEST_F(UTestBufferFusionPassReg, test_case_02) {
  REG_BUFFER_FUSION_PASS("", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                         TbeCommonRules0FusionPass, 0);

  REG_BUFFER_FUSION_PASS("MetadefBufferFusionPassTest", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                         TbeCommonRules0FusionPass, 1);
  REG_BUFFER_FUSION_PASS("MetadefBufferFusionPassTest", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                         TbeCommonRules0FusionPass, 2);

  REG_BUFFER_FUSION_PASS("MetadefBufferFusionPassTest1", BUILT_IN_VECTOR_CORE_BUFFER_FUSION_PASS,
                         TbeCommonRules0FusionPass, 0xffff);

  auto pass_desc = BufferFusionPassRegistry::GetInstance().GetPassDesc(BUILT_IN_AI_CORE_BUFFER_FUSION_PASS);
  EXPECT_EQ(pass_desc["MetadefBufferFusionPassTest"].attr, 2);
  auto create_fn = pass_desc["MetadefBufferFusionPassTest"].create_fn;
  auto buffer_fusion_pass_base_ptr = std::unique_ptr<BufferFusionPassBase>(create_fn());
  auto def_patterns = buffer_fusion_pass_base_ptr->DefinePatterns();
  EXPECT_EQ(def_patterns.size(), 1);
  for (const auto &def_pattern : def_patterns) {
    delete def_pattern;
  }

  pass_desc = BufferFusionPassRegistry::GetInstance().GetPassDesc(BUILT_IN_VECTOR_CORE_BUFFER_FUSION_PASS);
  EXPECT_EQ(pass_desc["MetadefBufferFusionPassTest1"].attr, 0xffff);
  create_fn = pass_desc["MetadefBufferFusionPassTest1"].create_fn;
  buffer_fusion_pass_base_ptr = std::unique_ptr<BufferFusionPassBase>(create_fn());
  def_patterns = buffer_fusion_pass_base_ptr->DefinePatterns();
  EXPECT_EQ(def_patterns.size(), 1);
  for (const auto &def_pattern : def_patterns) {
    delete def_pattern;
  }
}

TEST_F(UTestBufferFusionPassReg, test_post_fusion) {
  std::shared_ptr<BufferFusionPassBase> common2 = std::make_shared<TbeCommonRules2FusionPass>();
  EXPECT_EQ(common2->PostFusion(nullptr), FAILED);
}
}
