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
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pattern.h"
#include "framework/common/debug/ge_log.h"
#include "register/graph_optimizer/graph_fusion/connection_matrix.h"
#include "register/graph_optimizer/fusion_common/op_slice_info.h"
#include "runtime/kernel.h"

using namespace std;
using namespace domi;
using namespace fe;
using namespace ge;

static const string STREAM_LABEL = "_stream_label";
const std::string FE_IMPLY_TYPE = "_fe_imply_type";
namespace fe{
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
                                             "Relu6",   "Relu6D", "PRelu",
                                             "Add", "Mul",  "Softplus", "Sigmoid", "Mish",
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
  auto *pattern = new (std::nothrow) BufferFusionPattern(pass_name, MAX_OP_COUNT);
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
      .AddOpDesc(PATTERN_ELEMWISE, {OP_PATTERN_ELEMWISE, OP_PATTERN_BROAD_CAST}, TBE_PATTERN_NUM_NONE,
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
    int other_elt_wise_out = (int)(elem_wise_node->GetOutDataNodes().size() - 1);
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
    BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(pass_name, 10);
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
    patterns.push_back(pattern);
    return patterns; //todo need to check
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
    "PRelu", "Add", "Mul", "Softplus", "Sigmoid", "Mish","Minimum",
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
  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(pass_name, FUSION_OP_NUM_MAX);
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
      .AddOpDescTypeRules(PATTERN_ELEMWISE, {OP_PATTERN_ELEMWISE, OP_PATTERN_BROAD_CAST}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_MAX,
                 TBE_PATTERN_GROUPID_INVALID, {IGNORE_SHAPE_TYPE, ONLY_SUPPORT_STATIC})
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

static void DelNotMatchNodes(vector<ge::NodePtr>& elemwise_nodes, vector<ge::NodePtr> &fusion_nodes) {
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

BufferFusionPassType  type = BUFFER_FUSION_PASS_TYPE_RESERVED;
REGISTER_BUFFER_FUSION_PASS("MetadefBufferFusionPassTest", type, TbeCommonRules0FusionPass);

REGISTER_BUFFER_FUSION_PASS("", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            TbeCommonRules0FusionPass);

REGISTER_BUFFER_FUSION_PASS("MetadefBufferFusionPassTest", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            TbeCommonRules0FusionPass);
REGISTER_BUFFER_FUSION_PASS("MetadefBufferFusionPassTest", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            TbeCommonRules0FusionPass);

Status Run(ge::ComputeGraph &graph, std::shared_ptr<BufferFusionPassBase> &pass) {
  // 1. get pattern info
  auto patterns = pass->DefinePatterns();

  fe::ConnectionMatrix a;
  a.Generate(graph);
  
  // 2. for all patterns
  for (BufferFusionPattern *pattern : patterns) {
    if (pattern == nullptr) {
      continue;
    }
    string pattern_name = pattern->GetName();
    pattern->GetErrorCnt();
    pattern->GetHead();
    pattern->GetOpDesc("test");
    auto conv = pattern->GetOpDesc(PATTERN_CONVOLUTION);
    pattern->GetOpMaxCount();
    BufferFusionOpDesc *op_desc = nullptr;
    std::vector<BufferFusionOpDesc *> outputs;
    pattern->GetOutputs(op_desc, outputs);

    pattern->GetOutputs(conv, outputs);
    pattern->GetName();
    pattern->UpdateSkipStatus(conv);
    BufferFusionMapping mapping;
    auto dequant = pattern->GetOpDesc(PATTERN_DEQUANT);
    auto elmw = pattern->GetOpDesc(PATTERN_ELEMWISE);
    auto quant = pattern->GetOpDesc(PATTERN_QUANT);

    std::vector<BufferFusionOpDesc *> buffer_fusion_op_vec =
        {conv, dequant, elmw, quant};
    int i = 0;
    for (auto node : graph.GetDirectNode()) {
      if (i < 4) {
        std::vector<ge::NodePtr> node_vec = {node};
        mapping.emplace(std::make_pair(buffer_fusion_op_vec[i], node_vec));
      }
    }
    pass->GetName();

    std::vector<ge::NodePtr> fusion_nodes;
    EXPECT_EQ(fe::SUCCESS, pass->GetFusionNodes(mapping, fusion_nodes));
    EXPECT_EQ(fe::NOT_CHANGED, pass->GetMixl2FusionNodes(mapping, fusion_nodes));

//    OpCalcInfo op_slice_info;
//    pass->CalcFusionOpSliceInfo(fusion_nodes, op_slice_info);

    pass->GetMatchedHeadNode(fusion_nodes);

    pass->SetName("test");

    pass->GetMatchedNodes(mapping);

    pass->GetMatchedNodesByDescName(PATTERN_ELEMWISE, mapping);
  }

  for (const auto &pattern : patterns) {
    delete pattern;
  }


  return ge::GRAPH_SUCCESS;
}

Status RunPass(ge::ComputeGraph &graph) {
  std::shared_ptr<BufferFusionPassBase> common0 = std::make_shared<TbeCommonRules0FusionPass>();
  Run(graph, common0);
  return ge::GRAPH_SUCCESS;
}

class UB_FUSION_UT_CONV_ELT_RELU : public testing::Test {

 protected:
  static void SetUpTestCase() { std::cout << "UB fusion SetUp" << std::endl; }
  static void TearDownTestCase() {
    std::cout << "UB fusion TearDown" << std::endl;
  }

  virtual void SetUp() {
  }

  virtual void TearDown() {}
  void SetPattern(ge::OpDescPtr opdef, string optype) {
    auto key_pattern = opdef->GetName() + "_pattern";
    ge::AttrUtils::SetStr(opdef, key_pattern, optype);
  }
  void SetTvmType(ge::OpDescPtr opdef) {
    ge::AttrUtils::SetInt(opdef, ge::ATTR_NAME_IMPLY_TYPE,static_cast<int64_t>(domi::ImplyType::TVM));
  }
  void BuildGraph(ComputeGraphPtr graph, int32_t reluflag) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", "Data");
    OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", "Data");
    OpDescPtr data2 = std::make_shared<OpDesc>("DATA2", "Data");
    OpDescPtr conv = std::make_shared<OpDesc>("conv", "Convolution");
    OpDescPtr elemwise = std::make_shared<OpDesc>("elem", "Eltwise");
    OpDescPtr relu = std::make_shared<OpDesc>("relu", "Relu");
    OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", "ReLU");

    SetPattern(conv, "Convolution");
    SetPattern(relu, "ElemWise");
    SetPattern(elemwise, "ElemWise");
    SetTvmType(conv);
    SetTvmType(elemwise);
    SetTvmType(relu);
    // add descriptor
    vector<int64_t> dim(4, 4);
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);

    data->AddOutputDesc(out_desc);
    data1->AddOutputDesc(out_desc);
    data2->AddOutputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddOutputDesc(out_desc);
    elemwise->AddInputDesc(out_desc);
    elemwise->AddInputDesc(out_desc);
    elemwise->AddOutputDesc(out_desc);
    relu->AddInputDesc(out_desc);
    relu->AddOutputDesc(out_desc);
    relu1->AddInputDesc(out_desc);
    relu1->AddOutputDesc(out_desc);
    AttrUtils::SetInt(conv, FE_IMPLY_TYPE, 6);
    ge::AttrUtils::SetStr(conv, ge::ATTR_NAME_SESSION_GRAPH_ID, "_0_1_2_3");
    std::vector<int64_t> params = {0, 0, 0, 0, 0, 1, 0, 1};
    AttrUtils::SetListInt(conv, "ub_atomic_params", params);
    AttrUtils::SetBool(conv, "Aipp_Conv_Flag", true);
    conv->SetWorkspaceBytes({0});
    AttrUtils::SetInt(elemwise, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(relu, FE_IMPLY_TYPE, 6);

    NodePtr data_node = graph->AddNode(data);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr data2_node = graph->AddNode(data2);
    NodePtr conv_node = graph->AddNode(conv);
    NodePtr elemwise_node = graph->AddNode(elemwise);
    NodePtr relu_node = graph->AddNode(relu);
    NodePtr relu1_node = graph->AddNode(relu1);
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
        conv_node->GetName(), std::move(buffer));
    conv_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                        elemwise_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0),
                        elemwise_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(elemwise_node->GetOutDataAnchor(0),
                        relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0),
                        relu1_node->GetInDataAnchor(0));
  }

  void BuildGraphForL2Fusion(ComputeGraphPtr graph, int32_t reluflag) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", "Data");
    OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", "Data");
    OpDescPtr data2 = std::make_shared<OpDesc>("DATA2", "Data");
    OpDescPtr conv = std::make_shared<OpDesc>("conv", "Convolution");
    OpDescPtr elemwise = std::make_shared<OpDesc>("elem", "Eltwise");
    OpDescPtr relu = std::make_shared<OpDesc>("relu", "ReLU");
    OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", "ReLU");
    SetPattern(conv, "Convolution");
    SetPattern(relu, "ElemWise");
    SetPattern(elemwise, "ElemWise");
    SetTvmType(conv);
    SetTvmType(elemwise);
    SetTvmType(relu);
    // add descriptor
    vector<int64_t> dim(4, 4);
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);

    data->AddOutputDesc(out_desc);
    data1->AddOutputDesc(out_desc);
    data2->AddOutputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddOutputDesc(out_desc);
    elemwise->AddInputDesc(out_desc);
    elemwise->AddInputDesc(out_desc);
    elemwise->AddOutputDesc(out_desc);
    relu->AddInputDesc(out_desc);
    relu->AddOutputDesc(out_desc);
    relu1->AddInputDesc(out_desc);
    relu1->AddOutputDesc(out_desc);
    AttrUtils::SetInt(conv, FE_IMPLY_TYPE, 6);
    std::vector<int64_t> params = {0, 0, 0, 0, 0, 1, 0, 1};
    AttrUtils::SetListInt(conv, "ub_atomic_params", params);
    AttrUtils::SetBool(conv, "Aipp_Conv_Flag", true);
    conv->SetWorkspaceBytes({0});
    AttrUtils::SetInt(elemwise, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(relu, FE_IMPLY_TYPE, 6);

    NodePtr data_node = graph->AddNode(data);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr data2_node = graph->AddNode(data2);
    NodePtr conv_node = graph->AddNode(conv);
    NodePtr elemwise_node = graph->AddNode(elemwise);
    NodePtr relu_node = graph->AddNode(relu);
    NodePtr relu1_node = graph->AddNode(relu1);
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
        conv_node->GetName(), std::move(buffer));
    conv_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                        elemwise_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0),
                        elemwise_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(elemwise_node->GetOutDataAnchor(0),
                        relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0),
                        relu1_node->GetInDataAnchor(0));

    //elementwise l2 Info
    L2FusionInfoPtr elementwise_l2_info_ptr = std::make_shared<TaskL2FusionInfo_t>();
    //data
    uint64_t L2_mirror_addr=0;          // preload or swap source address
    uint32_t L2_data_section_size=123;    // every data size
    uint8_t L2_preload=0;               // 1 - preload from mirror_addr, 0 - no preload
    uint8_t modified=1;                 // 1 - data will be modified by kernel, 0 - no modified
    uint8_t priority=1;                 // data priority
    int8_t prev_L2_page_offset_base=-1;  // remap source section offset
    uint8_t L2_page_offset_base=0;      // remap destination section offset
    uint8_t L2_load_to_ddr=0;           // 1 - need load out, 0 - no need
    rtSmData_t tmp_data={L2_mirror_addr,L2_data_section_size,L2_preload,modified,priority,prev_L2_page_offset_base,L2_page_offset_base,L2_page_offset_base,L2_load_to_ddr};
    tmp_data.reserved[2]={0};

    elementwise_l2_info_ptr->l2_info.l2ctrl.data[0]=tmp_data;
    elementwise_l2_info_ptr->l2_info.l2ctrl.size=60;
    elementwise_l2_info_ptr->node_name="elem";
    L2FusionData_t elem_output={0,123,2};
    elementwise_l2_info_ptr->output[0]=elem_output;

    (void)ge::AttrUtils::SetBool(conv_node->GetOpDesc(), "need_re_precompile", true);
    elemwise_node->GetOpDesc()->SetExtAttr(
        "task_l2_fusion_info_extend_content", elementwise_l2_info_ptr);

    //relu l2 Info
    L2FusionInfoPtr relu_l2_info_ptr = std::make_shared<TaskL2FusionInfo_t>();
    //data
    tmp_data.reserved[2]={0};

    relu_l2_info_ptr->l2_info.l2ctrl.data[0]=tmp_data;
    relu_l2_info_ptr->l2_info.l2ctrl.size=60;
    relu_l2_info_ptr->node_name="relu";
    L2FusionData_t relu_output={1,234,2};
    relu_l2_info_ptr->output[0]=relu_output;

    (void)ge::AttrUtils::SetBool(relu_node->GetOpDesc(), "need_re_precompile", true);
    relu_node->GetOpDesc()->SetExtAttr(
        "task_l2_fusion_info_extend_content", relu_l2_info_ptr);


    //relu2 Info
    L2FusionInfoPtr relu2_l2_info_ptr = std::make_shared<TaskL2FusionInfo_t>();
    //data
    tmp_data.reserved[2]={0};

    relu2_l2_info_ptr->l2_info.l2ctrl.data[0]=tmp_data;
    relu2_l2_info_ptr->l2_info.l2ctrl.size=60;
    relu2_l2_info_ptr->node_name="relu1";
    L2FusionData_t conv_output={2,567,2};
    relu2_l2_info_ptr->output[0]=conv_output;

    (void)ge::AttrUtils::SetBool(relu1_node->GetOpDesc(), "need_re_precompile", true);
    relu1_node->GetOpDesc()->SetExtAttr(
        "task_l2_fusion_info_extend_content", relu2_l2_info_ptr);

  }
  void BuildGraphForL2Fusion1(ComputeGraphPtr graph, int32_t reluflag) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", "Data");
    OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", "Data");
    OpDescPtr data2 = std::make_shared<OpDesc>("DATA2", "Data");
    OpDescPtr conv = std::make_shared<OpDesc>("conv", "Convolution");
    OpDescPtr elemwise = std::make_shared<OpDesc>("elem", "Eltwise");
    OpDescPtr elemwise1 = std::make_shared<OpDesc>("elem1", "Eltwise");
    OpDescPtr relu = std::make_shared<OpDesc>("relu", "ReLU");

    SetPattern(conv, "Convolution");
    SetPattern(elemwise1, "ElemWise");
    SetPattern(elemwise, "ElemWise");
    SetPattern(relu, "ElemWise");
    SetTvmType(conv);
    SetTvmType(elemwise);
    SetTvmType(elemwise1);
    SetTvmType(relu);
    // add descriptor
    vector<int64_t> dim(4, 4);
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);

    data->AddOutputDesc(out_desc);
    data1->AddOutputDesc(out_desc);
    data2->AddOutputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddOutputDesc(out_desc);
    elemwise->AddInputDesc(out_desc);
    elemwise->AddInputDesc(out_desc);
    elemwise->AddOutputDesc(out_desc);
    elemwise1->AddInputDesc(out_desc);
    elemwise1->AddOutputDesc(out_desc);
    relu->AddInputDesc(out_desc);
    relu->AddOutputDesc(out_desc);

    AttrUtils::SetInt(conv, FE_IMPLY_TYPE, 6);
    std::vector<int64_t> params = {0, 0, 0, 0, 0, 1, 0, 1};
    AttrUtils::SetListInt(conv, "ub_atomic_params", params);
    AttrUtils::SetBool(conv, "Aipp_Conv_Flag", true);
    conv->SetWorkspaceBytes({0});
    AttrUtils::SetInt(elemwise, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(elemwise1, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(relu, FE_IMPLY_TYPE, 6);

    NodePtr data_node = graph->AddNode(data);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr data2_node = graph->AddNode(data2);
    NodePtr conv_node = graph->AddNode(conv);
    NodePtr elemwise_node = graph->AddNode(elemwise);
    NodePtr elemwise1_node = graph->AddNode(elemwise1);
    NodePtr relu_node = graph->AddNode(relu);

    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
        conv_node->GetName(), std::move(buffer));
    conv_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                        elemwise_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0),
                        elemwise_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(elemwise_node->GetOutDataAnchor(0),
                        elemwise1_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(elemwise1_node->GetOutDataAnchor(0),
                        relu_node->GetInDataAnchor(0));

    //elementwise l2 Info
    L2FusionInfoPtr elementwise_l2_info_ptr = std::make_shared<TaskL2FusionInfo_t>();
    //data
    uint64_t L2_mirror_addr=0;          // preload or swap source address
    uint32_t L2_data_section_size=123;    // every data size
    uint8_t L2_preload=0;               // 1 - preload from mirror_addr, 0 - no preload
    uint8_t modified=1;                 // 1 - data will be modified by kernel, 0 - no modified
    uint8_t priority=1;                 // data priority
    int8_t prev_L2_page_offset_base=-1;  // remap source section offset
    uint8_t L2_page_offset_base=0;      // remap destination section offset
    uint8_t L2_load_to_ddr=0;           // 1 - need load out, 0 - no need
    rtSmData_t tmp_data={L2_mirror_addr,L2_data_section_size,L2_preload,modified,priority,prev_L2_page_offset_base,L2_page_offset_base,L2_page_offset_base,L2_load_to_ddr};
    tmp_data.reserved[2]={0};

    elementwise_l2_info_ptr->l2_info.l2ctrl.data[0]=tmp_data;
    elementwise_l2_info_ptr->l2_info.l2ctrl.size=60;
    elementwise_l2_info_ptr->node_name="elem";
    L2FusionData_t elem_output={0,123,2};
    elementwise_l2_info_ptr->output[0]=elem_output;

    (void)ge::AttrUtils::SetBool(elemwise1_node->GetOpDesc(), "need_re_precompile", true);
    elemwise1_node->GetOpDesc()->SetExtAttr(
        "task_l2_fusion_info_extend_content", elementwise_l2_info_ptr);

  }
  void BuildGraph2(ComputeGraphPtr graph, int32_t reluflag) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", "Data");
    OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", "Data");
    OpDescPtr data2 = std::make_shared<OpDesc>("DATA2", "Data");
    OpDescPtr conv = std::make_shared<OpDesc>("conv", "Convolution");
    OpDescPtr elemwise = std::make_shared<OpDesc>("elem", "Eltwise");
    OpDescPtr relu = std::make_shared<OpDesc>("relu", "Relu");
    OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", "Relu");
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", "Convolution");
    OpDescPtr netout_op = std::make_shared<OpDesc>("netoutput", "NetOutput");

    SetPattern(conv, "Convolution");
    SetPattern(conv1, "Convolution");
    SetPattern(relu, "ElemWise");
    SetPattern(relu1, "ElemWise");
    SetPattern(elemwise, "ElemWise");
    SetTvmType(conv);
    SetTvmType(conv1);
    SetTvmType(elemwise);
    SetTvmType(relu);
    SetTvmType(relu1);
    // add descriptor
    vector<int64_t> dim(4, 4);
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);

    data->AddOutputDesc(out_desc);
    data1->AddOutputDesc(out_desc);
    data2->AddOutputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddOutputDesc(out_desc);
    conv1->AddInputDesc(out_desc);
    conv1->AddInputDesc(out_desc);
    conv1->AddOutputDesc(out_desc);
    elemwise->AddInputDesc(out_desc);
    elemwise->AddInputDesc(out_desc);
    elemwise->AddOutputDesc(out_desc);
    relu->AddInputDesc(out_desc);
    relu->AddOutputDesc(out_desc);
    relu1->AddInputDesc(out_desc);
    relu1->AddOutputDesc(out_desc);
    netout_op->AddInputDesc(out_desc);
    AttrUtils::SetInt(conv, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(conv1, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(elemwise, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(relu, FE_IMPLY_TYPE, 6);

    NodePtr data_node = graph->AddNode(data);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr data2_node = graph->AddNode(data2);
    NodePtr conv_node = graph->AddNode(conv);
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr elemwise_node = graph->AddNode(elemwise);
    NodePtr relu_node = graph->AddNode(relu);
    NodePtr relu1_node = graph->AddNode(relu1);
    NodePtr netout_node = graph->AddNode(netout_op);
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
        conv_node->GetName(), std::move(buffer));
    conv_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                        elemwise_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                        elemwise_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(elemwise_node->GetOutDataAnchor(0),
                        relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0),
                        relu1_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0),
                        netout_node->GetInDataAnchor(0));
  }

  void BuildGraph3(ComputeGraphPtr graph, int32_t reluflag) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", "Data");
    OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", "Data");
    OpDescPtr data2 = std::make_shared<OpDesc>("DATA2", "Data");
    OpDescPtr conv = std::make_shared<OpDesc>("conv", "Convolution");
    OpDescPtr elemwise = std::make_shared<OpDesc>("elem", "Eltwise");
    OpDescPtr relu = std::make_shared<OpDesc>("relu", "Relu");
    OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", "ReLU");
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", "Convolution");

    SetPattern(conv, "Convolution");
    SetPattern(conv1, "Convolution");
    SetPattern(relu, "ElemWise");
    SetPattern(elemwise, "ElemWise");
    SetTvmType(conv);
    SetTvmType(conv1);
    SetTvmType(elemwise);
    SetTvmType(relu);
    // add descriptor
    vector<int64_t> dim(4, 4);
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);

    data->AddOutputDesc(out_desc);
    data1->AddOutputDesc(out_desc);
    data2->AddOutputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddOutputDesc(out_desc);
    conv1->AddInputDesc(out_desc);
    conv1->AddInputDesc(out_desc);
    conv1->AddOutputDesc(out_desc);
    elemwise->AddInputDesc(out_desc);
    elemwise->AddInputDesc(out_desc);
    elemwise->AddOutputDesc(out_desc);
    relu->AddInputDesc(out_desc);
    relu->AddOutputDesc(out_desc);
    relu1->AddInputDesc(out_desc);
    relu1->AddOutputDesc(out_desc);
    AttrUtils::SetInt(conv, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(conv1, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(elemwise, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(relu, FE_IMPLY_TYPE, 6);
    ge::AttrUtils::SetStr(conv, STREAM_LABEL, "stream1");
    ge::AttrUtils::SetStr(conv1, STREAM_LABEL, "stream1");
    ge::AttrUtils::SetStr(elemwise, STREAM_LABEL, "stream1");
    ge::AttrUtils::SetStr(relu, STREAM_LABEL, "stream1");
    ge::AttrUtils::SetStr(relu1, STREAM_LABEL, "stream1");

    NodePtr data_node = graph->AddNode(data);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr data2_node = graph->AddNode(data2);
    NodePtr conv_node = graph->AddNode(conv);
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr elemwise_node = graph->AddNode(elemwise);
    NodePtr relu_node = graph->AddNode(relu);
    NodePtr relu1_node = graph->AddNode(relu1);
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
        conv_node->GetName(), std::move(buffer));
    conv_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                        elemwise_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                        elemwise_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(elemwise_node->GetOutDataAnchor(0),
                        relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0),
                        relu1_node->GetInDataAnchor(0));
  }

  void BuildGraph4(ComputeGraphPtr graph, int32_t reluflag) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", "Data");
    OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", "Data");
    OpDescPtr data2 = std::make_shared<OpDesc>("DATA2", "Data");
    OpDescPtr conv = std::make_shared<OpDesc>("conv", "Convolution");
    OpDescPtr elemwise = std::make_shared<OpDesc>("elem", "Eltwise");
    OpDescPtr relu = std::make_shared<OpDesc>("relu", "ReLU");
    OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", "ReLU");
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", "Convolution");

    SetPattern(conv, "Convolution");
    SetPattern(conv1, "Convolution");
    SetPattern(relu, "ElemWise");
    SetPattern(elemwise, "ElemWise");
    SetTvmType(conv);
    SetTvmType(conv1);
    SetTvmType(elemwise);
    SetTvmType(relu);
    // add descriptor
    vector<int64_t> dim(4, 4);
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);

    data->AddOutputDesc(out_desc);
    data1->AddOutputDesc(out_desc);
    data2->AddOutputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddOutputDesc(out_desc);
    conv1->AddInputDesc(out_desc);
    conv1->AddInputDesc(out_desc);
    conv1->AddOutputDesc(out_desc);
    elemwise->AddInputDesc(out_desc);
    elemwise->AddInputDesc(out_desc);
    elemwise->AddOutputDesc(out_desc);
    relu->AddInputDesc(out_desc);
    relu->AddOutputDesc(out_desc);
    relu1->AddInputDesc(out_desc);
    relu1->AddOutputDesc(out_desc);
    AttrUtils::SetInt(conv, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(conv1, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(elemwise, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(relu, FE_IMPLY_TYPE, 6);
    ge::AttrUtils::SetStr(conv, STREAM_LABEL, "stream1");
    ge::AttrUtils::SetStr(conv1, STREAM_LABEL, "stream1");
    ge::AttrUtils::SetStr(elemwise, STREAM_LABEL, "stream2");
    ge::AttrUtils::SetStr(relu, STREAM_LABEL, "stream1");

    NodePtr data_node = graph->AddNode(data);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr data2_node = graph->AddNode(data2);
    NodePtr conv_node = graph->AddNode(conv);
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr elemwise_node = graph->AddNode(elemwise);
    NodePtr relu_node = graph->AddNode(relu);
    NodePtr relu1_node = graph->AddNode(relu1);
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
        conv_node->GetName(), std::move(buffer));
    conv_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                        elemwise_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                        elemwise_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(elemwise_node->GetOutDataAnchor(0),
                        relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0),
                        relu1_node->GetInDataAnchor(0));
  }

  void BuildGraph5(ComputeGraphPtr graph, int32_t reluflag) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", "Data");
    OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", "Data");
    OpDescPtr data2 = std::make_shared<OpDesc>("DATA2", "Data");
    OpDescPtr conv = std::make_shared<OpDesc>("conv", "Convolution");
    OpDescPtr elemwise = std::make_shared<OpDesc>("elem", "Eltwise");
    OpDescPtr relu = std::make_shared<OpDesc>("relu", "ReLU");
    OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", "ReLU");
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", "Convolution");

    SetPattern(conv, "Convolution");
    SetPattern(conv1, "Convolution");
    SetPattern(relu, "ElemWise");
    SetPattern(elemwise, "ElemWise");
    SetTvmType(conv);
    SetTvmType(conv1);
    SetTvmType(elemwise);
    SetTvmType(relu);
    // add descriptor
    vector<int64_t> dim(4, 4);
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);

    data->AddOutputDesc(out_desc);
    data1->AddOutputDesc(out_desc);
    data2->AddOutputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddOutputDesc(out_desc);
    conv1->AddInputDesc(out_desc);
    conv1->AddInputDesc(out_desc);
    conv1->AddOutputDesc(out_desc);
    elemwise->AddInputDesc(out_desc);
    elemwise->AddInputDesc(out_desc);
    elemwise->AddOutputDesc(out_desc);
    relu->AddInputDesc(out_desc);
    relu->AddOutputDesc(out_desc);
    relu1->AddInputDesc(out_desc);
    relu1->AddOutputDesc(out_desc);
    AttrUtils::SetInt(conv, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(conv1, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(elemwise, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(relu, FE_IMPLY_TYPE, 6);
    ge::AttrUtils::SetStr(relu, STREAM_LABEL, "stream1");

    NodePtr data_node = graph->AddNode(data);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr data2_node = graph->AddNode(data2);
    NodePtr conv_node = graph->AddNode(conv);
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr elemwise_node = graph->AddNode(elemwise);
    NodePtr relu_node = graph->AddNode(relu);
    NodePtr relu1_node = graph->AddNode(relu1);
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
        conv_node->GetName(), std::move(buffer));
    conv_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                        elemwise_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                        elemwise_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(elemwise_node->GetOutDataAnchor(0),
                        relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0),
                        relu1_node->GetInDataAnchor(0));
  }

  void BuildGraph6(ComputeGraphPtr graph, int32_t reluflag) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", "Data");
    OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", "Data");
    OpDescPtr data2 = std::make_shared<OpDesc>("DATA2", "Data");
    OpDescPtr conv = std::make_shared<OpDesc>("conv", "Convolution");
    OpDescPtr elemwise = std::make_shared<OpDesc>("elem", "Eltwise");
    OpDescPtr relu = std::make_shared<OpDesc>("relu", "ReLU");
    OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", "ReLU");
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", "Convolution");

    SetPattern(conv, "Convolution");
    SetPattern(conv1, "Convolution");
    SetPattern(relu, "ElemWise");
    SetPattern(elemwise, "ElemWise");
    SetTvmType(conv);
    SetTvmType(conv1);
    SetTvmType(elemwise);
    SetTvmType(relu);
    // add descriptor
    vector<int64_t> dim(4, 4);
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);

    data->AddOutputDesc(out_desc);
    data1->AddOutputDesc(out_desc);
    data2->AddOutputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddOutputDesc(out_desc);
    conv1->AddInputDesc(out_desc);
    conv1->AddInputDesc(out_desc);
    conv1->AddOutputDesc(out_desc);
    elemwise->AddInputDesc(out_desc);
    elemwise->AddInputDesc(out_desc);
    elemwise->AddOutputDesc(out_desc);
    relu->AddInputDesc(out_desc);
    relu->AddOutputDesc(out_desc);
    relu1->AddInputDesc(out_desc);
    relu1->AddOutputDesc(out_desc);
    AttrUtils::SetInt(conv, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(conv1, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(elemwise, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(relu, FE_IMPLY_TYPE, 6);
    ge::AttrUtils::SetStr(conv, STREAM_LABEL, "stream1");

    NodePtr data_node = graph->AddNode(data);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr data2_node = graph->AddNode(data2);
    NodePtr conv_node = graph->AddNode(conv);
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr elemwise_node = graph->AddNode(elemwise);
    NodePtr relu_node = graph->AddNode(relu);
    NodePtr relu1_node = graph->AddNode(relu1);
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
        conv_node->GetName(), std::move(buffer));
    conv_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0),
                        elemwise_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                        elemwise_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(elemwise_node->GetOutDataAnchor(0),
                        relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0),
                        relu1_node->GetInDataAnchor(0));
  }

  void BuildGraphConvReluQuant(ComputeGraphPtr graph, int32_t reluflag) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", "Data");
    OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", "Data");
    OpDescPtr data2 = std::make_shared<OpDesc>("DATA2", "Data");
    OpDescPtr conv = std::make_shared<OpDesc>("conv", "Convolution");
    OpDescPtr relu = std::make_shared<OpDesc>("relu", "Relu");
    OpDescPtr quant = std::make_shared<OpDesc>("quant", "Quant");

    SetPattern(conv, "Convolution");
    SetPattern(relu, "ElemWise");
    SetPattern(quant, "quant");
    SetTvmType(conv);
    SetTvmType(relu);
    SetTvmType(quant);
    // add descriptor
    vector<int64_t> dim(4, 4);
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);

    data->AddOutputDesc(out_desc);
    data1->AddOutputDesc(out_desc);
    data2->AddOutputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddOutputDesc(out_desc);
    relu->AddInputDesc(out_desc);
    relu->AddOutputDesc(out_desc);
    quant->AddInputDesc(out_desc);
    quant->AddOutputDesc(out_desc);
    AttrUtils::SetInt(conv, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(relu, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(quant, FE_IMPLY_TYPE, 6);

    NodePtr data_node = graph->AddNode(data);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr data2_node = graph->AddNode(data2);
    NodePtr conv_node = graph->AddNode(conv);
    NodePtr relu_node = graph->AddNode(relu);
    NodePtr quant_node = graph->AddNode(quant);
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
        conv_node->GetName(), std::move(buffer));
    conv_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(2));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                        relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0),
                        quant_node->GetInDataAnchor(0));
  }
  void BuildGraphConvLeakyReluQuant1(ComputeGraphPtr graph, int32_t reluflag) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", "Data");
    OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", "Data");
    OpDescPtr data2 = std::make_shared<OpDesc>("DATA2", "Data");
    OpDescPtr conv = std::make_shared<OpDesc>("conv", "Convolution");
    OpDescPtr relu = std::make_shared<OpDesc>("relu", "LeakyRelu");
    OpDescPtr quant = std::make_shared<OpDesc>("quant", "Quant");

    SetPattern(conv, "Convolution");
    SetPattern(relu, "ElemWise");
    SetPattern(quant, "quant");
    SetTvmType(conv);
    SetTvmType(relu);
    SetTvmType(quant);
    // add descriptor
    vector<int64_t> dim(4, 4);
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);

    data->AddOutputDesc(out_desc);
    data1->AddOutputDesc(out_desc);
    data2->AddOutputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddOutputDesc(out_desc);
    relu->AddInputDesc(out_desc);
    relu->AddOutputDesc(out_desc);
    quant->AddInputDesc(out_desc);
    quant->AddOutputDesc(out_desc);
    AttrUtils::SetInt(conv, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(relu, FE_IMPLY_TYPE, 6);
    AttrUtils::SetFloat(relu, "negative_slope", 0);
    AttrUtils::SetInt(quant, FE_IMPLY_TYPE, 6);

    NodePtr data_node = graph->AddNode(data);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr data2_node = graph->AddNode(data2);
    NodePtr conv_node = graph->AddNode(conv);
    NodePtr relu_node = graph->AddNode(relu);
    NodePtr quant_node = graph->AddNode(quant);
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
        conv_node->GetName(), std::move(buffer));
    conv_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(2));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                        relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0),
                        quant_node->GetInDataAnchor(0));
  }
  void BuildGraphConvLeakyReluQuant2(ComputeGraphPtr graph, int32_t reluflag) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", "Data");
    OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", "Data");
    OpDescPtr data2 = std::make_shared<OpDesc>("DATA2", "Data");
    OpDescPtr conv = std::make_shared<OpDesc>("conv", "Convolution");
    OpDescPtr relu = std::make_shared<OpDesc>("relu", "LeakyRelu");
    OpDescPtr quant = std::make_shared<OpDesc>("quant", "Quant");

    SetPattern(conv, "Convolution");
    SetPattern(relu, "ElemWise");
    SetPattern(quant, "quant");
    SetTvmType(conv);
    SetTvmType(relu);
    SetTvmType(quant);
    // add descriptor
    vector<int64_t> dim(4, 4);
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);

    data->AddOutputDesc(out_desc);
    data1->AddOutputDesc(out_desc);
    data2->AddOutputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddOutputDesc(out_desc);
    relu->AddInputDesc(out_desc);
    relu->AddOutputDesc(out_desc);
    quant->AddInputDesc(out_desc);
    quant->AddOutputDesc(out_desc);
    AttrUtils::SetInt(conv, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(relu, FE_IMPLY_TYPE, 6);
    AttrUtils::SetFloat(relu, "negative_slope", 0.1);
    AttrUtils::SetInt(quant, FE_IMPLY_TYPE, 6);

    NodePtr data_node = graph->AddNode(data);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr data2_node = graph->AddNode(data2);
    NodePtr conv_node = graph->AddNode(conv);
    NodePtr relu_node = graph->AddNode(relu);
    NodePtr quant_node = graph->AddNode(quant);
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
        conv_node->GetName(), std::move(buffer));
    conv_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(2));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                        relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0),
                        quant_node->GetInDataAnchor(0));
  }
  void BuildGraphConvEltReluQuant1(ComputeGraphPtr graph, int32_t reluflag) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", "Data");
    OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", "Data");
    OpDescPtr data2 = std::make_shared<OpDesc>("DATA2", "Data");
    OpDescPtr conv = std::make_shared<OpDesc>("conv", "Convolution");
    OpDescPtr eltwise = std::make_shared<OpDesc>("eltwise", "Eltwise");
    OpDescPtr relu = std::make_shared<OpDesc>("relu", "Relu");
    OpDescPtr quant = std::make_shared<OpDesc>("quant", "Quant");

    SetPattern(conv, "Convolution");
    SetPattern(eltwise, "ElemWise");
    SetPattern(relu, "ElemWise");
    SetPattern(quant, "quant");
    SetTvmType(conv);
    SetTvmType(eltwise);
    SetTvmType(relu);
    SetTvmType(quant);
    // add descriptor
    vector<int64_t> dim(4, 4);
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);

    data->AddOutputDesc(out_desc);
    data1->AddOutputDesc(out_desc);
    data2->AddOutputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddOutputDesc(out_desc);
    eltwise->AddInputDesc(out_desc);
    eltwise->AddInputDesc(out_desc);
    eltwise->AddOutputDesc(out_desc);
    relu->AddInputDesc(out_desc);
    relu->AddOutputDesc(out_desc);
    quant->AddInputDesc(out_desc);
    quant->AddOutputDesc(out_desc);
    AttrUtils::SetInt(conv, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(eltwise, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(relu, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(quant, FE_IMPLY_TYPE, 6);

    NodePtr data_node = graph->AddNode(data);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr data2_node = graph->AddNode(data2);
    NodePtr conv_node = graph->AddNode(conv);
    NodePtr eltwise_node = graph->AddNode(eltwise);
    NodePtr relu_node = graph->AddNode(relu);
    NodePtr quant_node = graph->AddNode(quant);
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
        conv_node->GetName(), std::move(buffer));
    conv_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                        eltwise_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0),
                        eltwise_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(eltwise_node->GetOutDataAnchor(0),
                        relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0),
                        quant_node->GetInDataAnchor(0));
  }
  void BuildGraphConvEltReluQuant2(ComputeGraphPtr graph, int32_t reluflag) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", "Data");
    OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", "Data");
    OpDescPtr data2 = std::make_shared<OpDesc>("DATA2", "Data");
    OpDescPtr conv = std::make_shared<OpDesc>("conv", "Convolution");
    OpDescPtr eltwise = std::make_shared<OpDesc>("eltwise", "EltwiseNoFusion");
    OpDescPtr relu = std::make_shared<OpDesc>("relu", "Relu");
    OpDescPtr quant = std::make_shared<OpDesc>("quant", "Quant");

    SetPattern(conv, "Convolution");
    SetPattern(eltwise, "ElemWise");
    SetPattern(relu, "ElemWise");
    SetPattern(quant, "quant");
    SetTvmType(conv);
    SetTvmType(eltwise);
    SetTvmType(relu);
    SetTvmType(quant);
    // add descriptor
    vector<int64_t> dim(4, 4);
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);

    data->AddOutputDesc(out_desc);
    data1->AddOutputDesc(out_desc);
    data2->AddOutputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddOutputDesc(out_desc);
    eltwise->AddInputDesc(out_desc);
    eltwise->AddInputDesc(out_desc);
    eltwise->AddOutputDesc(out_desc);
    relu->AddInputDesc(out_desc);
    relu->AddOutputDesc(out_desc);
    quant->AddInputDesc(out_desc);
    quant->AddOutputDesc(out_desc);
    AttrUtils::SetInt(conv, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(eltwise, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(relu, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(quant, FE_IMPLY_TYPE, 6);
    AttrUtils::SetBool(quant, "_is_op_dynamic_impl", true);
    AttrUtils::SetBool(eltwise, "_is_op_dynamic_impl", true);

    NodePtr data_node = graph->AddNode(data);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr data2_node = graph->AddNode(data2);
    NodePtr conv_node = graph->AddNode(conv);
    NodePtr eltwise_node = graph->AddNode(eltwise);
    NodePtr relu_node = graph->AddNode(relu);
    NodePtr quant_node = graph->AddNode(quant);
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
        conv_node->GetName(), std::move(buffer));
    conv_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                        eltwise_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0),
                        eltwise_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(eltwise_node->GetOutDataAnchor(0),
                        relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0),
                        quant_node->GetInDataAnchor(0));
  }

  void BuildGraphdoubleConvEltElt(ComputeGraphPtr graph, int32_t reluflag) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", "Data");
    OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", "Data");
    OpDescPtr data2 = std::make_shared<OpDesc>("DATA2", "Data");
    OpDescPtr conv = std::make_shared<OpDesc>("conv", "Conv2D");
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", "Conv2D");
    OpDescPtr elemwise = std::make_shared<OpDesc>("elem", "Eltwise");
    OpDescPtr relu = std::make_shared<OpDesc>("relu", "Relu");
    OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", "Relu");
    SetPattern(conv, "Convolution");
    SetPattern(conv1, "Convolution");
    SetPattern(relu, "ElemWise");
    SetPattern(elemwise, "ElemWise");
    SetTvmType(conv);
    SetTvmType(conv1);
    SetTvmType(elemwise);
    SetTvmType(relu);
    // add descriptor
    vector<int64_t> dim = {4, 4, 4, 4};
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);
    out_desc.SetOriginFormat(ge::FORMAT_NCHW);
    vector<int64_t> dim1 = {8, 8, 8, 8};
    GeShape shape1(dim1);
    GeTensorDesc out_desc1(shape1);
    out_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    data->AddOutputDesc(out_desc);
    data1->AddOutputDesc(out_desc);
    data2->AddOutputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddOutputDesc(out_desc);
    conv1->AddInputDesc(out_desc1);
    conv1->AddInputDesc(out_desc1);
    conv1->AddOutputDesc(out_desc);
    elemwise->AddInputDesc(out_desc);
    elemwise->AddInputDesc(out_desc);
    elemwise->AddOutputDesc(out_desc);
    relu->AddInputDesc(out_desc);
    relu->AddOutputDesc(out_desc);
    relu1->AddInputDesc(out_desc);
    relu1->AddOutputDesc(out_desc);
    AttrUtils::SetInt(conv, FE_IMPLY_TYPE, 6);
    ge::AttrUtils::SetStr(conv, ge::ATTR_NAME_SESSION_GRAPH_ID, "_0_1_2_3");
    std::vector<int64_t> params = {0, 0, 0, 0, 0, 1, 0, 1};
    AttrUtils::SetListInt(conv, "ub_atomic_params", params);
    AttrUtils::SetBool(conv, "Aipp_Conv_Flag", true);
    conv->SetWorkspaceBytes({0});
    AttrUtils::SetInt(conv1, FE_IMPLY_TYPE, 6);
    ge::AttrUtils::SetStr(conv1, ge::ATTR_NAME_SESSION_GRAPH_ID, "_0_1_2_3");
    AttrUtils::SetListInt(conv1, "ub_atomic_params", params);
    AttrUtils::SetBool(conv1, "Aipp_Conv_Flag", true);
    conv1->SetWorkspaceBytes({0});
    AttrUtils::SetInt(elemwise, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(relu, FE_IMPLY_TYPE, 6);

    NodePtr data_node = graph->AddNode(data);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr data2_node = graph->AddNode(data2);
    NodePtr conv_node = graph->AddNode(conv);
    NodePtr conv_node1 = graph->AddNode(conv1);
    NodePtr elemwise_node = graph->AddNode(elemwise);
    NodePtr relu_node = graph->AddNode(relu);
    NodePtr relu1_node = graph->AddNode(relu1);
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
        conv_node->GetName(), std::move(buffer));
    conv_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);
    ge::OpKernelBinPtr tbe_kernel_ptr1 = std::make_shared<ge::OpKernelBin>(
        conv_node1->GetName(), std::move(buffer));
    conv_node1->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr1);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                        conv_node1->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0),
                        conv_node1->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                        elemwise_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv_node1->GetOutDataAnchor(0),
                        elemwise_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(elemwise_node->GetOutDataAnchor(0),
                        relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0),
                        relu1_node->GetInDataAnchor(0));
  }

  void BuildGraphdoubleConvEltElt_1(ComputeGraphPtr graph, int32_t reluflag) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", "Data");
    OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", "Data");
    OpDescPtr data2 = std::make_shared<OpDesc>("DATA2", "Data");
    OpDescPtr conv = std::make_shared<OpDesc>("conv", "Conv2D");
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", "Conv2D");
    OpDescPtr elemwise = std::make_shared<OpDesc>("elem", "Eltwise");
    OpDescPtr relu = std::make_shared<OpDesc>("relu", "Relu");
    OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", "Relu");
    SetPattern(conv, "Convolution");
    SetPattern(conv1, "Convolution");
    SetPattern(relu, "ElemWise");
    SetPattern(elemwise, "ElemWise");
    SetTvmType(conv);
    SetTvmType(conv1);
    SetTvmType(elemwise);
    SetTvmType(relu);
    // add descriptor
    vector<int64_t> dim = {4, 4, 4, 4};
    GeShape shape(dim);
    GeTensorDesc out_desc(shape);
    out_desc.SetOriginFormat(ge::FORMAT_NHWC);
    vector<int64_t> dim1 = {8, 8, 8, 8};
    GeShape shape1(dim1);
    GeTensorDesc out_desc1(shape1);
    out_desc1.SetOriginFormat(ge::FORMAT_HWCN);
    data->AddOutputDesc(out_desc);
    data1->AddOutputDesc(out_desc);
    data2->AddOutputDesc(out_desc);
    conv->AddInputDesc(out_desc1);
    conv->AddInputDesc(out_desc1);
    conv->AddOutputDesc(out_desc);
    conv1->AddInputDesc(out_desc);
    conv1->AddInputDesc(out_desc);
    conv1->AddOutputDesc(out_desc);
    elemwise->AddInputDesc(out_desc);
    elemwise->AddInputDesc(out_desc);
    elemwise->AddOutputDesc(out_desc);
    relu->AddInputDesc(out_desc);
    relu->AddOutputDesc(out_desc);
    relu1->AddInputDesc(out_desc);
    relu1->AddOutputDesc(out_desc);
    AttrUtils::SetInt(conv, FE_IMPLY_TYPE, 6);
    ge::AttrUtils::SetStr(conv, ge::ATTR_NAME_SESSION_GRAPH_ID, "_0_1_2_3");
    std::vector<int64_t> params = {0, 0, 0, 0, 0, 1, 0, 1};
    AttrUtils::SetListInt(conv, "ub_atomic_params", params);
    AttrUtils::SetBool(conv, "Aipp_Conv_Flag", true);
    conv->SetWorkspaceBytes({0});
    AttrUtils::SetInt(conv1, FE_IMPLY_TYPE, 6);
    ge::AttrUtils::SetStr(conv1, ge::ATTR_NAME_SESSION_GRAPH_ID, "_0_1_2_3");
    AttrUtils::SetListInt(conv1, "ub_atomic_params", params);
    AttrUtils::SetBool(conv1, "Aipp_Conv_Flag", true);
    conv1->SetWorkspaceBytes({0});
    AttrUtils::SetInt(elemwise, FE_IMPLY_TYPE, 6);
    AttrUtils::SetInt(relu, FE_IMPLY_TYPE, 6);

    NodePtr data_node = graph->AddNode(data);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr data2_node = graph->AddNode(data2);
    NodePtr conv_node = graph->AddNode(conv);
    NodePtr conv_node1 = graph->AddNode(conv1);
    NodePtr elemwise_node = graph->AddNode(elemwise);
    NodePtr relu_node = graph->AddNode(relu);
    NodePtr relu1_node = graph->AddNode(relu1);
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
        conv_node->GetName(), std::move(buffer));
    conv_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);
    ge::OpKernelBinPtr tbe_kernel_ptr1 = std::make_shared<ge::OpKernelBin>(
        conv_node1->GetName(), std::move(buffer));
    conv_node1->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr1);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                        conv_node1->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0),
                        conv_node1->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                        elemwise_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv_node1->GetOutDataAnchor(0),
                        elemwise_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(elemwise_node->GetOutDataAnchor(0),
                        relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0),
                        relu1_node->GetInDataAnchor(0));
  }

  void BuildGraphConvElt(ComputeGraphPtr graph) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", "Data");
    OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", "Data");
    OpDescPtr data2 = std::make_shared<OpDesc>("DATA2", "Data");
    OpDescPtr conv = std::make_shared<OpDesc>("conv", "Conv2D");
    OpDescPtr elemwise = std::make_shared<OpDesc>("elem", "Eltwise");

    SetPattern(conv, "Convolution");
    SetPattern(elemwise, "ElemWise");
    SetTvmType(conv);
    SetTvmType(elemwise);
    // add descriptor
    vector<int64_t> dim(4, 4);
    GeShape in_shape(dim);
    GeTensorDesc out_desc(in_shape);

    data->AddOutputDesc(out_desc);
    data1->AddOutputDesc(out_desc);
    data2->AddOutputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddInputDesc(out_desc);
    conv->AddOutputDesc(out_desc);
    elemwise->AddInputDesc(out_desc);
    elemwise->AddInputDesc(out_desc);
    elemwise->AddOutputDesc(out_desc);
    AttrUtils::SetInt(conv, FE_IMPLY_TYPE, 6);
    ge::AttrUtils::SetStr(conv, ge::ATTR_NAME_SESSION_GRAPH_ID, "_0_1_2_3");
    std::vector<int64_t> params = {0, 0, 0, 0, 0, 1, 0, 1};
    AttrUtils::SetListInt(conv, "ub_atomic_params", params);
    // AttrUtils::SetBool(conv, "Aipp_Conv_Flag", true);
    conv->SetWorkspaceBytes({0});
    AttrUtils::SetInt(elemwise, FE_IMPLY_TYPE, 6);

    NodePtr data_node = graph->AddNode(data);
    NodePtr data1_node = graph->AddNode(data1);
    NodePtr data2_node = graph->AddNode(data2);
    NodePtr conv_node = graph->AddNode(conv);
    NodePtr elemwise_node = graph->AddNode(elemwise);
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
        conv_node->GetName(), std::move(buffer));
    conv_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0),
                        conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                        elemwise_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0),
                        elemwise_node->GetInDataAnchor(1));
  }
};

TEST_F(UB_FUSION_UT_CONV_ELT_RELU, coverage_01) {

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  BuildGraphConvEltReluQuant1(graph, 1);

  std::shared_ptr<BufferFusionPassBase> pass = std::make_shared<ConveragePass>();
  std::vector<ge::NodePtr> fusion_nodes;
  BufferFusionMapping mapping;
  EXPECT_EQ(fe::SUCCESS, pass->GetFusionNodes(mapping, fusion_nodes));

  std::vector<ge::NodePtr> matched_nodes;
  for (auto node : graph->GetDirectNode()) {
    matched_nodes.emplace_back(node);
  }

  pass->GetMatchedHeadNode(matched_nodes);

  std::vector<BufferFusionPattern *> patterns= pass->DefinePatterns();
  for (auto &pattern : patterns) {
    delete pattern;
  }
}

TEST_F(UB_FUSION_UT_CONV_ELT_RELU, dyn_static_check) {
  std::map<string, BufferFusionPassRegistry::CreateFn> create_fns =
      BufferFusionPassRegistry::GetInstance().GetCreateFnByType(BUILT_IN_AI_CORE_BUFFER_FUSION_PASS);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  BuildGraphConvEltReluQuant2(graph, 1);
  std::shared_ptr<BufferFusionPassBase> common0 = std::make_shared<TbeCommonRules0FusionPass>();
  std::vector<BufferFusionPattern *> patterns = common0->DefinePatterns();
  BufferFusionMapping mapping;
  auto pattern = patterns[0];
  pattern->SetGraphModType(1);
  EXPECT_EQ(1, pattern->GetGraphModType());
  auto conv = pattern->GetOpDesc(PATTERN_CONV);
  auto eltwise = pattern->GetOpDesc(PATTERN_ELEMWISE);
  auto quant = pattern->GetOpDesc(PATTERN_QUANT);
  auto conv_vec = {graph->FindNode("conv")};
  auto eltwise_vec = {graph->FindNode("eltwise")};
  auto quant_vec = {graph->FindNode("quant")};
  mapping.emplace(std::make_pair(conv, conv_vec));
  mapping.emplace(std::make_pair(eltwise, eltwise_vec));
  mapping.emplace(std::make_pair(quant, quant_vec));
  EXPECT_FALSE(BufferFusionPassBase::CheckNodesImplConsistent(mapping));
  EXPECT_FALSE(BufferFusionPassBase::CheckNodesIncDynamicShape(mapping));
  EXPECT_TRUE(BufferFusionPassBase::CheckNodeIsDynamicImpl(graph->FindNode("eltwise")));
  EXPECT_EQ(common0->PostFusion(nullptr), fe::SUCCESS);
  for (auto &pattern : patterns) {
    delete pattern;
  }
}
