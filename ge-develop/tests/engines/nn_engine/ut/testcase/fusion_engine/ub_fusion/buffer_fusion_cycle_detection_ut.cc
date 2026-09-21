/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#define protected public
#define private public
#include "common/graph_comm.h"
#include "pass_manager.h"
#include "common/platform_utils.h"
#include "common/configuration.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/op_kernel_bin.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/ub_fusion/buffer_fusion.h"
#include "graph_optimizer/ub_fusion/buffer_fusion_pass_runner.h"
#include "common/fusion_statistic/fusion_statistic_writer.h"
#undef protected
#undef private
#include "graph_constructor.h"
using namespace std;
using namespace fe;

class BufferFusionCycleDetectionUt : public testing::Test {
 public:

 protected:
  virtual void SetUp() {
    graph_comm_ptr_ = std::make_shared<GraphComm>("engineName");
    graph_comm_ptr_->Initialize();
    fusion_priority_mgr_ptr_ = std::make_shared<FusionPriorityManager>(
        "engineName", nullptr);

    sub_graph_optimizer_ptr_ =
        std::make_shared<BufferFusion>(graph_comm_ptr_, fusion_priority_mgr_ptr_, nullptr);
  }

  virtual void TearDown() {

  }

  std::shared_ptr<GraphComm> graph_comm_ptr_;
  std::shared_ptr<FusionPriorityManager> fusion_priority_mgr_ptr_;
  std::shared_ptr<BufferFusion> sub_graph_optimizer_ptr_;
};

static const char kPatternConv[] = "conv2d";
static const char kPatternConvHead[] = "conv2dHead";
static const char kPatternDWConv[] = "DepthwiseConvolution";
static const char kPatternAdd[] = "add";
static const char kPatternRelu6[] = "relu6";
static const char kPatternMul1[] = "mul1";
static const char kPatternMul2[] = "mul2";
static const char kPatternDeq[] = "dequant";
static const char kPatternQuant[] = "quant";
static const char kPatternOtherInput0[] = "otherInput1";
static const char kPatternOtherInput1[] = "otherInput1";
static const char kPatternOtherInput2[] = "otherInput2";
static const char kPatternOtherInput3[] = "otherInput3";

class Conv2dAddRelu6MulMulFusionPass : public BufferFusionPassBase {
 public:
  Conv2dAddRelu6MulMulFusionPass() {}
  ~Conv2dAddRelu6MulMulFusionPass() {}

 protected:
  vector<BufferFusionPattern*> DefinePatterns() override {
    vector<BufferFusionPattern*> patterns;

    string pattern_name = "ConvAddRelu6MulMulFusionPass";
    BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pattern_name);
    FE_CHECK((pattern == nullptr), FE_LOGE("new an object failed."), return patterns);
    FE_LOGD("Start to define %s pass pattern.", pattern_name.c_str());
    /* define pattern    conv2d(depthwise)   -->     add  -->  relu6  -->   mul1   -->    mul2
     *                      |           otherinput1--/                      / otherinput2-/
     *                      |----------------------------------------------/
     */
    pattern->AddOpDesc(kPatternConv, {OP_PATTERN_CONV, OP_PATTERN_DEPTHWISE_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternAdd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternRelu6, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternMul1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternMul2, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternOtherInput2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)

        .SetHead({kPatternConv})
        .SetOutputs(kPatternConv, {kPatternAdd, kPatternMul1}, TBE_OUTPUT_BRANCH_MULTI)
        .SetOutputs(kPatternOtherInput1, {kPatternAdd})
        .SetOutputs(kPatternAdd, {kPatternRelu6}, TBE_OUTPUT_BRANCH_SINGLE)
        .SetOutputs(kPatternRelu6, {kPatternMul1}, TBE_OUTPUT_BRANCH_SINGLE)
        .SetOutputs(kPatternMul1, {kPatternMul2}, TBE_OUTPUT_BRANCH_SINGLE)
        .SetOutputs(kPatternOtherInput2, {kPatternMul2});
    patterns.push_back(pattern);
    FE_LOGD("End to define %s pass pattern.", pattern_name.c_str());

    return patterns;
  }
};

class Conv2dAddRelu6MulMulFusionPass2_2 : public BufferFusionPassBase {
 public:
  Conv2dAddRelu6MulMulFusionPass2_2() {}
  ~Conv2dAddRelu6MulMulFusionPass2_2() {}

 protected:
  vector<BufferFusionPattern*> DefinePatterns() override {
    vector<BufferFusionPattern*> patterns;

    string pattern_name = "Conv2dAddRelu6MulMulFusionPass2_2";
    BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pattern_name);
    FE_CHECK((pattern == nullptr), FE_LOGE("new an object failed."), return patterns);
    FE_LOGD("Start to define %s pass pattern.", pattern_name.c_str());
    /* define pattern    conv2d(depthwise)   -->     add  -->  relu6  -->   mul1
     *                      |           otherinput1--/                      /
     *                      |----------------------------------------------/
     */
    pattern->AddOpDesc(kPatternConv, {OP_PATTERN_CONV, OP_PATTERN_DEPTHWISE_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternAdd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternRelu6, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternMul1, {OP_PATTERN_ELEMWISE}, 0, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)

        .SetHead({kPatternConv})
        .SetOutputs(kPatternConv, {kPatternAdd, kPatternMul1}, TBE_OUTPUT_BRANCH_MULTI, false, true)
        .SetOutputs(kPatternOtherInput1, {kPatternAdd})
        .SetOutputs(kPatternAdd, {kPatternRelu6}, TBE_OUTPUT_BRANCH_SINGLE)
        .SetOutputs(kPatternRelu6, {kPatternMul1}, TBE_OUTPUT_BRANCH_SINGLE);
    patterns.push_back(pattern);
    FE_LOGD("End to define %s pass pattern.", pattern_name.c_str());

    return patterns;
  }
};

class Conv2dAddRelu6MulMulFusionPass3 : public BufferFusionPassBase {
 public:
  Conv2dAddRelu6MulMulFusionPass3() {}
  ~Conv2dAddRelu6MulMulFusionPass3() {}

 protected:
  vector<BufferFusionPattern*> DefinePatterns() override {
    vector<BufferFusionPattern*> patterns;

    string pattern_name = "ConvAddRelu6MulMulFusionPass3";
    BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pattern_name);
    FE_CHECK((pattern == nullptr), FE_LOGE("new an object failed."), return patterns);
    FE_LOGD("Start to define %s pass pattern.", pattern_name.c_str());
    /* define pattern    conv2d(depthwise)   -->     add  -->  relu6  -->   mul1 --> mul2
     *                      |           otherinput1--/                      /
     *                      |----------------------------------------------/
     */
    pattern->AddOpDesc(kPatternConv, {OP_PATTERN_CONV, OP_PATTERN_DEPTHWISE_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternAdd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternRelu6, {OP_PATTERN_ELEMWISE}, 0, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternMul1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternMul2, {OP_PATTERN_ELEMWISE}, 0, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternOtherInput2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)

        .SetHead({kPatternConv})
        .SetOutputs(kPatternConv, {kPatternAdd, kPatternMul1}, TBE_OUTPUT_BRANCH_MULTI)
        .SetOutputs(kPatternOtherInput1, {kPatternAdd})
        .SetOutputs(kPatternAdd, {kPatternRelu6}, TBE_OUTPUT_BRANCH_SINGLE, false, true)
        .SetOutputs(kPatternRelu6, {kPatternMul1}, TBE_OUTPUT_BRANCH_SINGLE)
        .SetOutputs(kPatternMul1, {kPatternMul2}, TBE_OUTPUT_BRANCH_SINGLE)
        .SetOutputs(kPatternOtherInput2, {kPatternMul2});
    patterns.push_back(pattern);
    FE_LOGD("End to define %s pass pattern.", pattern_name.c_str());

    return patterns;
  }
};


class Conv2dAddRelu6MulMulFusionPass3_2 : public BufferFusionPassBase {
 public:
  Conv2dAddRelu6MulMulFusionPass3_2() {}
  ~Conv2dAddRelu6MulMulFusionPass3_2() {}

 protected:
  vector<BufferFusionPattern*> DefinePatterns() override {
    vector<BufferFusionPattern*> patterns;

    string pattern_name = "Conv2dAddRelu6MulMulFusionPass3_2";
    BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pattern_name);
    FE_CHECK((pattern == nullptr), FE_LOGE("new an object failed."), return patterns);
    FE_LOGD("Start to define %s pass pattern.", pattern_name.c_str());
    /* define pattern    conv2d(depthwise)   -->     add  -->  relu6  -->   mul1
     *                      |           otherinput1--/                      /
     *                      |----------------------------------------------/
     */
    pattern->AddOpDesc(kPatternConv, {OP_PATTERN_CONV, OP_PATTERN_DEPTHWISE_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternAdd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternRelu6, {OP_PATTERN_ELEMWISE}, 0, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternMul1, {OP_PATTERN_ELEMWISE}, 0, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)

        .SetHead({kPatternConv})
        .SetOutputs(kPatternConv, {kPatternAdd, kPatternMul1}, TBE_OUTPUT_BRANCH_MULTI)
        .SetOutputs(kPatternOtherInput1, {kPatternAdd})
        .SetOutputs(kPatternAdd, {kPatternRelu6}, TBE_OUTPUT_BRANCH_SINGLE, false, true)
        .SetOutputs(kPatternRelu6, {kPatternMul1}, TBE_OUTPUT_BRANCH_SINGLE);
    patterns.push_back(pattern);
    FE_LOGD("End to define %s pass pattern.", pattern_name.c_str());

    return patterns;
  }
};

class Conv2dAddRelu6MulMulFusionPass4 : public BufferFusionPassBase {
 public:
  Conv2dAddRelu6MulMulFusionPass4() {}
  ~Conv2dAddRelu6MulMulFusionPass4() {}

 protected:
  vector<BufferFusionPattern*> DefinePatterns() override {
    vector<BufferFusionPattern*> patterns;

    string pattern_name = "Conv2dAddRelu6MulMulFusionPass4";
    BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pattern_name);
    FE_CHECK((pattern == nullptr), FE_LOGE("new an object failed."), return patterns);
    FE_LOGD("Start to define %s pass pattern.", pattern_name.c_str());
    /* define pattern    conv2d(depthwise)   -->     add  -->  relu6  -->   mul1   -->    mul2
     *                      |           otherinput1--/                      / otherinput2-/
     *                      |----------------------------------------------/
     */
    pattern->AddOpDesc(kPatternConvHead, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternConv, {OP_PATTERN_CONV, OP_PATTERN_DEPTHWISE_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternAdd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternRelu6, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternMul1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternMul2, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternOtherInput2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)

        .SetHead({kPatternConvHead})
        .SetOutputs(kPatternConvHead, {kPatternConv}, TBE_OUTPUT_BRANCH_SINGLE)
        .SetOutputs(kPatternConv, {kPatternAdd, kPatternMul1}, TBE_OUTPUT_BRANCH_MULTI, true, true)
        .SetOutputs(kPatternOtherInput1, {kPatternAdd})
        .SetOutputs(kPatternAdd, {kPatternRelu6}, TBE_OUTPUT_BRANCH_SINGLE)
        .SetOutputs(kPatternRelu6, {kPatternMul1}, TBE_OUTPUT_BRANCH_SINGLE)
        .SetOutputs(kPatternMul1, {kPatternMul2}, TBE_OUTPUT_BRANCH_SINGLE)
        .SetOutputs(kPatternOtherInput2, {kPatternMul2});
    patterns.push_back(pattern);
    FE_LOGD("End to define %s pass pattern.", pattern_name.c_str());

    return patterns;
  }
};

class Conv2dAddRelu6MulMulFusionPass5 : public BufferFusionPassBase {
 public:
  Conv2dAddRelu6MulMulFusionPass5() {}
  ~Conv2dAddRelu6MulMulFusionPass5() {}

 protected:
  vector<BufferFusionPattern*> DefinePatterns() override {
    vector<BufferFusionPattern*> patterns;

    string pattern_name = "Conv2dAddRelu6MulMulFusionPass5";
    BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pattern_name);
    FE_CHECK((pattern == nullptr), FE_LOGE("new an object failed."), return patterns);
    FE_LOGD("Start to define %s pass pattern.", pattern_name.c_str());
    /* define pattern    conv2d(depthwise)   -->     add  -->  relu6  -->   mul1
     *
     */
    pattern->AddOpDesc(kPatternConvHead, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternConv, {OP_PATTERN_CONV, OP_PATTERN_DEPTHWISE_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternAdd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternRelu6, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternMul1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, 5)
        .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternOtherInput2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)

        .SetHead({kPatternConvHead})
        .SetOutputs(kPatternConvHead, {kPatternConv}, TBE_OUTPUT_BRANCH_SINGLE)
        .SetOutputs(kPatternConv, {kPatternAdd}, TBE_OUTPUT_BRANCH_MULTI, true, true)
        .SetOutputs(kPatternOtherInput1, {kPatternAdd})
        .SetOutputs(kPatternAdd, {kPatternRelu6}, TBE_OUTPUT_BRANCH_SINGLE)
        .SetOutputs(kPatternOtherInput2, {kPatternMul1})
        .SetOutputs(kPatternRelu6, {kPatternMul1}, TBE_OUTPUT_BRANCH_SINGLE);
    patterns.push_back(pattern);
    FE_LOGD("End to define %s pass pattern.", pattern_name.c_str());

    return patterns;
  }
};

class Conv2dAddRelu6MulMulFusionPass6 : public BufferFusionPassBase {
 public:
  Conv2dAddRelu6MulMulFusionPass6() {}
  ~Conv2dAddRelu6MulMulFusionPass6() {}

 protected:
  vector<BufferFusionPattern*> DefinePatterns() override {
    vector<BufferFusionPattern*> patterns;

    string pattern_name = "Conv2dAddRelu6MulMulFusionPass6";
    BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pattern_name);
    FE_CHECK((pattern == nullptr), FE_LOGE("new an object failed."), return patterns);
    FE_LOGD("Start to define %s pass pattern.", pattern_name.c_str());
    /* define pattern    conv2d(depthwise)   -->     add  -->  relu6  -->   mul1
     *
     */
    pattern->AddOpDesc(kPatternConvHead, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternConv, {OP_PATTERN_CONV, OP_PATTERN_DEPTHWISE_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternAdd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternRelu6, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternMul1, {TBE_PATTERN_OP_TYPE_ANY}, TBE_PATTERN_NUM_DEFAULT, 5)
        .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternOtherInput2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)

        .SetHead({kPatternConvHead})
        .SetOutputs(kPatternConvHead, {kPatternConv}, TBE_OUTPUT_BRANCH_SINGLE)
        .SetOutputs(kPatternConv, {kPatternAdd}, TBE_OUTPUT_BRANCH_MULTI, true, true)
        .SetOutputs(kPatternOtherInput1, {kPatternAdd})
        .SetOutputs(kPatternAdd, {kPatternRelu6}, TBE_OUTPUT_BRANCH_SINGLE)
        .SetOutputs(kPatternOtherInput2, {kPatternMul1})
        .SetOutputs(kPatternRelu6, {kPatternMul1}, TBE_OUTPUT_BRANCH_SINGLE);
    patterns.push_back(pattern);
    FE_LOGD("End to define %s pass pattern.", pattern_name.c_str());

    return patterns;
  }
};


class Conv2dAddRelu6MulMulFusionPass7 : public BufferFusionPassBase {
 public:
  Conv2dAddRelu6MulMulFusionPass7() {}
  ~Conv2dAddRelu6MulMulFusionPass7() {}

 protected:
  vector<BufferFusionPattern*> DefinePatterns() override {
    vector<BufferFusionPattern*> patterns;

    string pattern_name = "Conv2dAddRelu6MulMulFusionPass7";
    BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pattern_name);
    FE_CHECK((pattern == nullptr), FE_LOGE("new an object failed."), return patterns);
    FE_LOGD("Start to define %s pass pattern.", pattern_name.c_str());
    /* define pattern    conv2d(depthwise)   -->     add  -->  relu6  -->   mul1
     *
     */
    pattern->AddOpDesc(kPatternConvHead, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternConv, {OP_PATTERN_CONV, OP_PATTERN_DEPTHWISE_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternAdd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternRelu6, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternMul1, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, 5)
        .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternOtherInput2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)

        .SetHead({kPatternConvHead})
        .SetOutputs(kPatternConvHead, {kPatternConv}, TBE_OUTPUT_BRANCH_SINGLE)
        .SetOutputs(kPatternConv, {kPatternAdd}, TBE_OUTPUT_BRANCH_MULTI, true, true)
        .SetOutputs(kPatternOtherInput1, {kPatternAdd})
        .SetOutputs(kPatternAdd, {kPatternRelu6}, TBE_OUTPUT_BRANCH_SINGLE)
        .SetOutputs(kPatternOtherInput2, {kPatternMul1})
        .SetOutputs(kPatternRelu6, {kPatternMul1}, TBE_OUTPUT_BRANCH_SINGLE);
    patterns.push_back(pattern);
    FE_LOGD("End to define %s pass pattern.", pattern_name.c_str());

    return patterns;
  }
};

class Conv2dAddRelu6MulMulFusionPass8 : public BufferFusionPassBase {
 public:
  Conv2dAddRelu6MulMulFusionPass8() {}
  ~Conv2dAddRelu6MulMulFusionPass8() {}

 protected:
  vector<BufferFusionPattern*> DefinePatterns() override {
    vector<BufferFusionPattern*> patterns;

    string pattern_name = "Conv2dAddRelu6MulMulFusionPass8";
    BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pattern_name);
    FE_CHECK((pattern == nullptr), FE_LOGE("new an object failed."), return patterns);
    FE_LOGD("Start to define %s pass pattern.", pattern_name.c_str());
    /* define pattern    conv2d(depthwise)   -->     add  -->  relu6  -->   mul1
     *
     */
    pattern->AddOpDesc(kPatternConvHead, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternConv, {OP_PATTERN_CONV, OP_PATTERN_DEPTHWISE_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternAdd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternRelu6, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternMul1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternMul2, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, 5)
        .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternOtherInput2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
        .AddOpDesc(kPatternOtherInput3, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)

        .SetHead({kPatternConvHead})
        .SetOutputs(kPatternConvHead, {kPatternConv}, TBE_OUTPUT_BRANCH_SINGLE)
        .SetOutputs(kPatternConv, {kPatternAdd, kPatternMul1}, TBE_OUTPUT_BRANCH_MULTI, true, true)
        .SetOutputs(kPatternOtherInput1, {kPatternAdd})
        .SetOutputs(kPatternAdd, {kPatternRelu6}, TBE_OUTPUT_BRANCH_SINGLE)
        .SetOutputs(kPatternRelu6, {kPatternMul1}, TBE_OUTPUT_BRANCH_SINGLE)
        .SetOutputs(kPatternOtherInput3, {kPatternMul2})
        .SetOutputs(kPatternMul1, {kPatternMul2}, TBE_OUTPUT_BRANCH_SINGLE);
    patterns.push_back(pattern);
    FE_LOGD("End to define %s pass pattern.", pattern_name.c_str());

    return patterns;
  }
};

namespace cycle_detection {
void BuildGraph01(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 161, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                        original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu6", "Relu6", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul1", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul2", "Mul", 2, 1)

      .SetInputs("conv", {"Data_0", "Data_1"})

      .SetInputs("add", {"conv", "Data_2"})

      .SetInput("relu6", "add")
      .SetInputs("mul1", {"conv", "relu6"})
      .SetInputs("mul2", {"mul1", "Data_3"})
      .SetInput("NetOutput", "mul2");
  test.DumpGraph(graph);
}

// contains control edges cycle
void BuildGraph02(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 161, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                        original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu6", "Relu6", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul1", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul2", "Mul", 2, 1)

      .SetInputs("conv", {"Data_0", "Data_1"})

      .SetInputs("add", {"conv", "Data_2"})

      .SetInput("relu6", "add")
      .SetInputs("mul1", {"conv", "relu6"})
      .SetInputs("mul2", {"mul1", "Data_3"})
      .SetInput("NetOutput", "mul2")
      .SetInput("mul1:-1", "add:-1");
  test.DumpGraph(graph);
}

// contains control edges cycle
void BuildGraph02_1(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 161, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                        original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu6", "Relu6", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul1", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul2", "Mul", 2, 1)

      .SetInputs("conv", {"Data_0", "Data_1"})

      .SetInputs("add", {"conv", "Data_2"})

      .SetInput("relu6", "add")
      .SetInputs("mul1", {"conv", "relu6"})
      .SetInputs("mul2", {"mul1", "Data_3"})
      .SetInput("NetOutput", "mul2")
      .SetInput("temp:-1", "add:-1")
      .SetInput("mul1:-1", "temp:-1");
  test.DumpGraph(graph);
}

// contains control edges cycle
void BuildGraph02_2(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 161, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                        original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu6", "Relu6", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul1", "Mul", 2, 1)

      .SetInputs("conv", {"Data_0", "Data_1"})

      .SetInputs("add", {"conv", "Data_2"})

      .SetInput("relu6", "add")
      .SetInputs("mul1", {"conv", "relu6"})
      .SetInput("NetOutput", "mul1")
      .SetInput("temp:-1", "add:-1")
      .SetInput("mul1:-1", "temp:-1");
  test.DumpGraph(graph);
}

void BuildGraph03(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 161, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                        original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "Opaque", "relu6", "Relu6", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul1", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul2", "Mul", 2, 1)

      .SetInputs("conv", {"Data_0", "Data_1"})

      .SetInputs("add", {"conv", "Data_2"})

      .SetInput("relu6", "add")
      .SetInputs("mul1", {"conv", "relu6"})
      .SetInputs("mul2", {"mul1", "Data_3"})
      .SetInput("NetOutput", "mul2");
  test.DumpGraph(graph);
}

void BuildGraph03_1(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 161, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                        original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "Opaque", "relu6", "Relu6", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul1", "Mul", 2, 1)

      .SetInputs("conv", {"Data_0", "Data_1"})

      .SetInputs("add", {"conv", "Data_2"})

      .SetInput("relu6", "add")
      .SetInputs("mul1", {"conv", "relu6"})
      .SetInput("NetOutput", "mul1");
  test.DumpGraph(graph);
}

void BuildGraph04(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 161, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                        original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv/head", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu6", "Relu6", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul1", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul2", "Mul", 2, 1)

      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add/other", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu6/other", "Relu6", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul1/other", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul2/other", "Mul", 2, 1)

      .SetInputs("conv/head", {"Data_0", "Data_1"})
      .SetInputs("conv", {"conv/head", "Data_2"})

      .SetInputs("add", {"conv:0", "Data_3"})
      .SetInput("relu6", "add")
      .SetInputs("mul1", {"conv:0", "relu6"})
      .SetInputs("mul2", {"mul1", "Data_5"})

      .SetInputs("add/other", {"conv:0", "Data_4"})
      .SetInput("relu6/other", "add/other")
      .SetInputs("mul1/other", {"conv:0", "relu6/other"})
      .SetInputs("mul2/other", {"mul1/other", "Data_6"})

      .SetInput("NetOutput", "mul2")
      .SetInput("NetOutput", "mul2/other")
      .SetInput("temp:-1", "add:-1")
      .SetInput("mul1:-1", "temp:-1");
  test.DumpGraph(graph);
}

/* Two branches are the same length. And first branch
 * contains a cycle. */
void BuildGraph05(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 161, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                        original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv/head", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu6", "Relu6", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul1", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul2", "Mul", 2, 1)

      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add/other", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu6/other", "Relu6", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul1/other", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul2/other", "Mul", 2, 1)

      .SetInputs("conv/head", {"Data_0", "Data_1"})
      .SetInputs("conv", {"conv/head", "Data_2"})

      .SetInputs("add", {"conv:0", "Data_3"})
      .SetInput("relu6", "add")
      .SetInputs("mul1", {"Data_4", "relu6"})
      .SetInputs("mul2", {"mul1", "Data_5"})

      .SetInputs("add/other", {"conv:0", "Data_6"})
      .SetInput("relu6/other", "add/other")
      .SetInputs("mul1/other", {"Data_7", "relu6/other"})
      .SetInputs("mul2/other", {"mul1/other", "Data_8"})

      .SetInput("NetOutput", "mul2")
      .SetInput("NetOutput", "mul2/other")
      .SetInput("temp:-1", "add:-1")
      .SetInput("mul1:-1", "temp:-1");
  test.DumpGraph(graph);
}

/* The topological first branch is longer but it contains a cycle. */
void BuildGraph05_1(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 161, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                        original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv/head", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu6", "Relu6", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul1", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul2", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul3", "Mul", 2, 1)

      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add/other", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu6/other", "Relu6", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul1/other", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul2/other", "Mul", 2, 1)

      .SetInputs("conv/head", {"Data_0", "Data_1"})
      .SetInputs("conv", {"conv/head", "Data_2"})

      .SetInputs("add", {"conv:0", "Data_3"})
      .SetInput("relu6", "add")
      .SetInputs("mul1", {"Data_4", "relu6"})
      .SetInputs("mul2", {"mul1", "Data_51"})
      .SetInputs("mul3", {"mul2", "Data_52"})

      .SetInputs("add/other", {"conv:0", "Data_6"})
      .SetInput("relu6/other", "add/other")
      .SetInputs("mul1/other", {"Data_7", "relu6/other"})
      .SetInputs("mul2/other", {"mul1/other", "Data_8"})

      .SetInput("NetOutput", "mul3")
      .SetInput("NetOutput", "mul2/other")
      .SetInput("temp:-1", "add:-1")
      .SetInput("mul1:-1", "temp:-1");
  test.DumpGraph(graph);
}


/* The topological first branch is longer.
 * Both branches contain a cycle. */
void BuildGraph05_2(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 161, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                        original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv/head", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu6", "Relu6", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul1", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul2", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul3", "Mul", 2, 1)

      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add/other", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu6/other", "Relu6", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul1/other", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul2/other", "Mul", 2, 1)

      .SetInputs("conv/head", {"Data_0", "Data_1"})
      .SetInputs("conv", {"conv/head", "Data_2"})

      .SetInputs("add", {"conv:0", "Data_3"})
      .SetInput("relu6", "add")
      .SetInputs("mul1", {"Data_4", "relu6"})
      .SetInputs("mul2", {"mul1", "Data_51"})
      .SetInputs("mul3", {"mul2", "Data_52"})

      .SetInputs("add/other", {"conv:0", "Data_6"})
      .SetInput("relu6/other", "add/other")
      .SetInputs("mul1/other", {"Data_7", "relu6/other"})
      .SetInputs("mul2/other", {"mul1/other", "Data_8"})

      .SetInput("NetOutput", "mul3")
      .SetInput("NetOutput", "mul2/other")
      .SetInput("temp:-1", "add:-1")
      .SetInput("mul1:-1", "temp:-1")

      .SetInput("temp2:-1", "add/other:-1")
      .SetInput("mul1/other:-1", "temp2:-1");
  test.DumpGraph(graph);
}

/* The topological first branch is longer.
 * The second is shorter and it contains a cycle. */
void BuildGraph05_3(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 161, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                        original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv/head", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu6", "Relu6", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul1", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul2", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul3", "Mul", 2, 1)

      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add/other", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu6/other", "Relu6", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul1/other", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul2/other", "Mul", 2, 1)

      .SetInputs("conv/head", {"Data_0", "Data_1"})
      .SetInputs("conv", {"conv/head", "Data_2"})

      .SetInputs("add", {"conv:0", "Data_3"})
      .SetInput("relu6", "add")
      .SetInputs("mul1", {"Data_4", "relu6"})
      .SetInputs("mul2", {"mul1", "Data_51"})
      .SetInputs("mul3", {"mul2", "Data_52"})

      .SetInputs("add/other", {"conv:0", "Data_6"})
      .SetInput("relu6/other", "add/other")
      .SetInputs("mul1/other", {"Data_7", "relu6/other"})
      .SetInputs("mul2/other", {"mul1/other", "Data_8"})

      .SetInput("NetOutput", "mul3")
      .SetInput("NetOutput", "mul2/other")
      .SetInput("temp:-1", "add/other:-1")
      .SetInput("mul1/other:-1", "temp:-1");
  test.DumpGraph(graph);
}

/* The topological second branch is longer.
 * The first is shorter and it contains a cycle. */
void BuildGraph05_4(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 161, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                        original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv/head", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu6", "Relu6", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul1", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul2", "Mul", 2, 1)

      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add/other", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu6/other", "Relu6", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul1/other", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul2/other", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul3/other", "Mul", 2, 1)

      .SetInputs("conv/head", {"Data_0", "Data_1"})
      .SetInputs("conv", {"conv/head", "Data_2"})

      .SetInputs("add", {"conv:0", "Data_3"})
      .SetInput("relu6", "add")
      .SetInputs("mul1", {"Data_4", "relu6"})
      .SetInputs("mul2", {"mul1", "Data_5"})

      .SetInputs("add/other", {"conv:0", "Data_6"})
      .SetInput("relu6/other", "add/other")
      .SetInputs("mul1/other", {"Data_7", "relu6/other"})
      .SetInputs("mul2/other", {"mul1/other", "Data_81"})
      .SetInputs("mul3/other", {"mul2/other", "Data_82"})

      .SetInput("NetOutput", "mul2")
      .SetInput("NetOutput", "mul3/other")
      .SetInput("temp:-1", "add:-1")
      .SetInput("mul1:-1", "temp:-1");
  test.DumpGraph(graph);
}

/* The topological second branch is longer.
 * The second is longger and it contains a cycle. */
void BuildGraph05_5(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 161, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                        original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv/head", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu6", "Relu6", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul1", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul2", "Mul", 2, 1)

      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add/other", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu6/other", "Relu6", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul1/other", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul2/other", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul3/other", "Mul", 2, 1)

      .SetInputs("conv/head", {"Data_0", "Data_1"})
      .SetInputs("conv", {"conv/head", "Data_2"})

      .SetInputs("add", {"conv:0", "Data_3"})
      .SetInput("relu6", "add")
      .SetInputs("mul1", {"Data_4", "relu6"})
      .SetInputs("mul2", {"mul1", "Data_5"})

      .SetInputs("add/other", {"conv:0", "Data_6"})
      .SetInput("relu6/other", "add/other")
      .SetInputs("mul1/other", {"Data_7", "relu6/other"})
      .SetInputs("mul2/other", {"mul1/other", "Data_81"})
      .SetInputs("mul3/other", {"mul2/other", "Data_82"})

      .SetInput("NetOutput", "mul2")
      .SetInput("NetOutput", "mul3/other")
      .SetInput("temp:-1", "add/other:-1")
      .SetInput("mul1/other:-1", "temp:-1");
  test.DumpGraph(graph);
}

/* The longest match contains a fake cycle on
 * mul1 and a real cycle on mul3. */
void BuildGraph08(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 161, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                        original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv/head", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu6", "Relu6", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul1", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul2", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul3", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul4", "Mul", 2, 1)

      .SetInputs("conv/head", {"Data_0", "Data_1"})
      .SetInputs("conv", {"conv/head", "Data_2"})

      .SetInputs("add", {"conv:0", "Data_3"})
      .SetInput("relu6", "add")
      .SetInputs("mul1", {"conv:0", "relu6"})
      .SetInputs("mul2", {"mul1", "Data_51"})
      .SetInputs("mul3", {"mul2", "Data_52"})
      .SetInputs("mul4", {"mul3", "Data_53"})

      .SetInput("NetOutput", "mul4")
      .SetInput("temp:-1", "add:-1")
      .SetInput("mul3:-1", "temp:-1");
  test.DumpGraph(graph);
}


/* The longest match contains a real cycle on
 * mul3 and a real cycle on mul4. */
void BuildGraph08_1(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 161, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                        original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv/head", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu6", "Relu6", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul1", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul2", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul3", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul4", "Mul", 2, 1)

      .SetInputs("conv/head", {"Data_0", "Data_1"})
      .SetInputs("conv", {"conv/head", "Data_2"})

      .SetInputs("add", {"conv:0", "Data_3"})
      .SetInput("relu6", "add")
      .SetInputs("mul1", {"conv:0", "relu6"})
      .SetInputs("mul2", {"mul1", "Data_51"})
      .SetInputs("mul3", {"mul2", "Data_52"})
      .SetInputs("mul4", {"mul3", "Data_53"})

      .SetInput("NetOutput", "mul4")
      .SetInput("temp1:-1", "add:-1")
      .SetInput("mul3:-1", "temp1:-1")
      .SetInput("temp2:-1", "add:-1")
      .SetInput("mul4:-1", "temp2:-1");
  test.DumpGraph(graph);
}

/* The longest match contains a fake cycle on
 * mul1 and a real cycle on mul4. */
void BuildGraph08_2(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 161, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                        original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv/head", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu6", "Relu6", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul1", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul2", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul3", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul4", "Mul", 2, 1)

      .SetInputs("conv/head", {"Data_0", "Data_1"})
      .SetInputs("conv", {"conv/head", "Data_2"})

      .SetInputs("add", {"conv:0", "Data_3"})
      .SetInput("relu6", "add")
      .SetInputs("mul1", {"conv:0", "relu6"})
      .SetInputs("mul2", {"mul1", "Data_51"})
      .SetInputs("mul3", {"mul2", "Data_52"})
      .SetInputs("mul4", {"mul3", "Data_53"})

      .SetInput("NetOutput", "mul4")
      .SetInput("temp:-1", "add:-1")
      .SetInput("mul4:-1", "temp:-1");
  test.DumpGraph(graph);
}

/* The longest match contains a fake cycle on
 * mul1 and a real cycle on mul2. */
void BuildGraph08_3(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 161, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                        original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv/head", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu6", "Relu6", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul1", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul2", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul3", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul4", "Mul", 2, 1)

      .SetInputs("conv/head", {"Data_0", "Data_1"})
      .SetInputs("conv", {"conv/head", "Data_2"})

      .SetInputs("add", {"conv:0", "Data_3"})
      .SetInput("relu6", "add")
      .SetInputs("mul1", {"conv:0", "relu6"})
      .SetInputs("mul2", {"mul1", "Data_51"})
      .SetInputs("mul3", {"mul2", "Data_52"})
      .SetInputs("mul4", {"mul3", "Data_53"})

      .SetInput("NetOutput", "mul4")
      .SetInput("temp:-1", "add:-1")
      .SetInput("mul2:-1", "temp:-1");
  test.DumpGraph(graph);
}


/* The longest match contains a real cycle on
 * mul2 and a real cycle on mul4. */
void BuildGraph08_4(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 161, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                        original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv/head", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu6", "Relu6", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul1", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul2", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul3", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "mul4", "Mul", 2, 1)

      .SetInputs("conv/head", {"Data_0", "Data_1"})
      .SetInputs("conv", {"conv/head", "Data_2"})

      .SetInputs("add", {"conv:0", "Data_3"})
      .SetInput("relu6", "add")
      .SetInputs("mul1", {"conv:0", "relu6"})
      .SetInputs("mul2", {"mul1", "Data_51"})
      .SetInputs("mul3", {"mul2", "Data_52"})
      .SetInputs("mul4", {"mul3", "Data_53"})

      .SetInput("NetOutput", "mul4")
      .SetInput("temp:-1", "add:-1")
      .SetInput("mul2:-1", "temp:-1")
      .SetInput("temp:-1", "add:-1")
      .SetInput("mul4:-1", "temp:-1");
  test.DumpGraph(graph);
}

void BuildGraph09(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 161, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                        original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "sub", "Sub", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "exp", "Exp", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "Reduce", "reducesum", "ReduceSumD", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "realdiv", "RealDiv", 2, 1)

      .SetInputs("sub", {"Data_1", "Data_2"})
      .SetInput("exp", "sub")
      .SetInput("reducesum", "exp:0")
      .SetInputs("realdiv", {"exp:0", "reducesum"})

      .SetInput("NetOutput", "realdiv");
  test.DumpGraph(graph);
}

void BuildGraph10(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 161, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                        original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_CONV, "conv1", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_DEQUANT, "dequant", "AscendDequant", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "relu", "LeakyRelu", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_QUANT, "quant1", "AscendQuant", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_CONV, "conv2", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_QUANT, "quant2", "AscendQuant", 1, 1)

      .SetInputs("conv1", {"Data_1", "Data_2"})
      .SetInputs("dequant", {"conv1", "Data_3"})
      .SetInput("relu", "dequant")

      .SetInputs("quant1", {"relu:0"})
      .SetInputs("quant2", {"add:0"})
      .SetInputs("conv2", {"quant1:0", "Data_5"})
      .SetInputs("add", {"relu:0", "conv2"})
      .SetInput("NetOutput", "quant2");
  test.DumpGraph(graph);
}

void BuildGraph11(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 161, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                        original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add1", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add2", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add3", "Add", 2, 1)

      .SetInputs("add1", {"Data_0", "Data_1"})
      .SetInputs("add2", {"add1:0", "add1:0"})
      .SetInputs("add3", {"add2:0", "Data_3"})
      .SetInput("NetOutput", "add3");
  test.DumpGraph(graph);
}

void BuildGraph12(ge::ComputeGraphPtr &graph) {
  GraphConstructor test(graph, "test");

  test.AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add1", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add2", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add3", "Add", 2, 1)

      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "rsqrt", "Rsqrt", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "mul1", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "mul2", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "mul3", "Mul", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "mul4", "Mul", 2, 1)

      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "sub", "Sub", 2, 1)

      .SetInputs("add1", {"Data_0", "Data_1"})
      .SetInputs("rsqrt", {"add1:0"})
      .SetInputs("mul1", {"Data_2:0", "rsqrt:0"})
      .SetInputs("mul2", {"Data_3:0", "mul1:0"})
      .SetInputs("mul3", {"Data_4:0", "mul1:0"})
      .SetInputs("mul4", {"Data_5:0", "mul1:0"})

      .SetInputs("sub", {"Data_6:0", "mul3:0"})
      .SetInputs("add2", {"mul2:0", "sub:0"})
      .SetInputs("add3", {"mul4:0", "sub:0"})

      .SetInputs("NetOutput", {"add2:0", "add3:0"});
  test.DumpGraph(graph);
}


void BuildGraph13(ge::ComputeGraphPtr &graph) {
  GraphConstructor test(graph, "test");

  test.AddOpDesc(EN_IMPL_HW_TBE, "Opaque", "transdata", "TransData", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_CONV, "conv", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add1", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add2", "Add", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add3", "Add", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add4", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add5", "Add", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add6", "Add", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add7", "Add", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add8", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add9", "Add", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "addfinal", "Add", 1, 1)

      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_QUANT, "quant1", "Quant", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_QUANT, "quant2", "Quant", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_QUANT, "quant3", "Quant", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_QUANT, "quant4", "Quant", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_QUANT, "quant5", "Quant", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_QUANT, "quant6", "Quant", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_QUANT, "quant7", "Quant", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_QUANT, "quantfinal", "Quant", 1, 1)

      .SetInputs("transdata", {"Data_0"})
      .SetInputs("conv", {"transdata:0", "Data_1"})

      .SetInput("quant1", "conv:0")
      .SetInput("add1", "conv:0")

      .SetInput("add2", "add1:0")
      .SetInput("quant2", "add1:0")
      .SetInput("NetOutput", "quant2:0")

      .SetInput("add3", "add2:0")
      .SetInput("quant3", "add2:0")
      .SetInput("NetOutput", "quant3:0")
      .SetInput("add4:0", "add3:0")

      .SetInput("add4:1", "conv:0")
      .SetInput("quant4", "add4:0")
      .SetInput("add8", "add4:0")
      .SetInput("add5", "add4:0")
      .SetInput("quant5", "add5:0")
      .SetInput("NetOutput", "quant5:0")

      .SetInput("add6", "add5:0")
      .SetInput("quant6", "add6:0")
      .SetInput("NetOutput", "quant6:0")
      .SetInput("add7", "add6:0")
      .SetInput("add8:1", "add7:0")

      .SetInput("add9", "add8:0")
      .SetInput("quant7", "add9:0")
      .SetInput("NetOutput", "quant7:0")

      .SetInput("add1:1", "quant1:0")

      .SetInput("addfinal", "conv:0")
      .SetInput("quantfinal", "addfinal:0")
      .SetInput("NetOutput", "quantfinal:0");
  test.DumpGraph(graph);
}


void BuildGraph14(ge::ComputeGraphPtr &graph) {
  GraphConstructor test(graph, "test");

  test.AddOpDesc(EN_IMPL_HW_TBE, "Opaque", "transdata", "TransData", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_CONV, "conv", "Conv2D", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add1", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add2", "Add", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add3", "Add", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add4", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add5", "Add", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add6", "Add", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add7", "Add", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add8", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add9", "Add", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add10", "Add", 2, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add11", "Add", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add12", "Add", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "add13", "Add", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_ELEMWISE, "addfinal", "Add", 1, 1)

      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_QUANT, "quant1", "Quant", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_QUANT, "quant2", "Quant", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_QUANT, "quant3", "Quant", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_QUANT, "quant4", "Quant", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_QUANT, "quant5", "Quant", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_QUANT, "quant6", "Quant", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_QUANT, "quant7", "Quant", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_QUANT, "quant8", "Quant", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, OP_PATTERN_QUANT, "quantfinal", "Quant", 1, 1)

      .SetInputs("transdata", {"Data_0"})
      .SetInputs("conv", {"transdata:0", "Data_1"})

      .SetInput("quant1", "conv:0")
      .SetInput("add1", "conv:0")

      .SetInput("add2", "add1:0")
      .SetInput("quant2", "add1:0")
      .SetInput("NetOutput", "quant2:0")

      .SetInput("add3", "add2:0")
      .SetInput("quant3", "add2:0")
      .SetInput("NetOutput", "quant3:0")
      .SetInput("add4:0", "add3:0")

      .SetInput("add4:1", "conv:0")
      .SetInput("quant4", "add4:0")
      .SetInput("add8", "add4:0")
      .SetInput("add5", "add4:0")
      .SetInput("quant5", "add5:0")
      .SetInput("NetOutput", "quant5:0")

      .SetInput("add6", "add5:0")
      .SetInput("quant6", "add6:0")
      .SetInput("NetOutput", "quant6:0")
      .SetInput("add7", "add6:0")
      .SetInput("add8:1", "add7:0")

      .SetInput("add9", "add8:0")
      .SetInput("quant7", "add9:0")
      .SetInput("NetOutput", "quant7:0")

      .SetInput("add10", "conv:0")
      .SetInput("add11", "add10:0")
      .SetInput("add12", "add11:0")
      .SetInput("add13", "add12:0")
      .SetInput("quant8", "add13:0")

      .SetInput("add10:1", "quant1:0")
      .SetInput("add1:1", "quant1:0")

      .SetInput("addfinal", "conv:0")
      .SetInput("quantfinal", "addfinal:0")
      .SetInput("NetOutput", "quantfinal:0");
  test.DumpGraph(graph);
}
}

/* No final cycle */
TEST_F(BufferFusionCycleDetectionUt, testcase_01) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  cycle_detection::BuildGraph01(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  BufferFusionOptimizerPtr buffer_fusion_optimizer = std::make_shared<BufferFusionOptimizer>();
  buffer_fusion_optimizer->Initialize(*graph);

  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) Conv2dAddRelu6MulMulFusionPass(); };
  BufferFusionPassRunner *test_pass = new (std::nothrow)
      BufferFusionPassRunner("test_pass", create_fn, cycle_detector, buffer_fusion_optimizer);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  int find = 0;
  int total_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      EXPECT_EQ(node->GetName(), "convaddrelu6mul1mul2");
      find = 1;
    }
    total_node++;
  }
  EXPECT_EQ(find, 1);
  EXPECT_EQ(total_node, 6);
}

/* Contains control cycle of fusion node itself,
 * this self-cycle will not be detected. */
TEST_F(BufferFusionCycleDetectionUt, testcase_02) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  cycle_detection::BuildGraph02(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  BufferFusionOptimizerPtr buffer_fusion_optimizer = std::make_shared<BufferFusionOptimizer>();
  buffer_fusion_optimizer->Initialize(*graph);

  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) Conv2dAddRelu6MulMulFusionPass(); };
  BufferFusionPassRunner *test_pass = new (std::nothrow)
      BufferFusionPassRunner("test_pass", create_fn, cycle_detector, buffer_fusion_optimizer);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  int find = 0;
  int total_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      EXPECT_EQ(node->GetName(), "convaddrelu6mul1mul2");
      find = 1;
    }

    total_node++;
  }
  EXPECT_EQ(find, 1);
  EXPECT_EQ(total_node, 6);
}

/* Contains control cycle */
TEST_F(BufferFusionCycleDetectionUt, testcase_02_1) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  cycle_detection::BuildGraph02_1(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  BufferFusionOptimizerPtr buffer_fusion_optimizer = std::make_shared<BufferFusionOptimizer>();
  buffer_fusion_optimizer->Initialize(*graph);

  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) Conv2dAddRelu6MulMulFusionPass(); };
  BufferFusionPassRunner *test_pass = new (std::nothrow)
      BufferFusionPassRunner("test_pass", create_fn, cycle_detector, buffer_fusion_optimizer);

  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  int find = 0;
  int total_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      EXPECT_EQ(node->GetName(), "conv");
      find = 1;
    }

    total_node++;
  }
  EXPECT_EQ(find, 1);
  EXPECT_EQ(total_node, 11);
}

/* Contains control cycle */
TEST_F(BufferFusionCycleDetectionUt, testcase_02_2) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  cycle_detection::BuildGraph02_2(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  BufferFusionOptimizerPtr buffer_fusion_optimizer = std::make_shared<BufferFusionOptimizer>();
  buffer_fusion_optimizer->Initialize(*graph);

  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) Conv2dAddRelu6MulMulFusionPass2_2(); };
  BufferFusionPassRunner *test_pass = new (std::nothrow)
      BufferFusionPassRunner("test_pass", create_fn, cycle_detector, buffer_fusion_optimizer);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  int find = 0;
  int total_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      EXPECT_EQ(node->GetName(), "convaddrelu6");
      find = 1;
    }

    total_node++;
  }
  EXPECT_EQ(find, 1);
  EXPECT_EQ(total_node, 7);
}

TEST_F(BufferFusionCycleDetectionUt, testcase_02_2_mixl2) {
  PlatformUtils::Instance().soc_version_ = "Ascend910B2";
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::IsaArchVersion)] = static_cast<int64_t>(ISAArchVersion::EN_ISA_ARCH_V220);
  fe::PlatformInfoManager::Instance().opti_compilation_info_.soc_version = "Ascend910B2";
  fe::PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910B2");

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  cycle_detection::BuildGraph02_2(graph);

  for (auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      (void)ge::AttrUtils::SetStr(node->GetOpDesc(), "_sgt_cube_vector_core_type", "AiCore");
      break;
    }
  }
  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  BufferFusionOptimizerPtr buffer_fusion_optimizer = std::make_shared<BufferFusionOptimizer>();
  buffer_fusion_optimizer->Initialize(*graph);

  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) Conv2dAddRelu6MulMulFusionPass2_2(); };
  BufferFusionPassRunner *test_pass = new (std::nothrow)
          BufferFusionPassRunner("test_pass", create_fn, cycle_detector, buffer_fusion_optimizer);

  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  int find = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      EXPECT_EQ(node->GetName(), "convaddrelu6");
      find = 1;
    }
  }
  EXPECT_EQ(find, 1);

  fe::PlatformInfoManager::Instance().opti_compilation_info_.soc_version = "Ascend910A";
  fe::PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910A");
}

/* Contains final cycle */
TEST_F(BufferFusionCycleDetectionUt, testcase_03) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  cycle_detection::BuildGraph03(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  BufferFusionOptimizerPtr buffer_fusion_optimizer = std::make_shared<BufferFusionOptimizer>();
  buffer_fusion_optimizer->Initialize(*graph);

  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) Conv2dAddRelu6MulMulFusionPass3(); };
  BufferFusionPassRunner *test_pass = new (std::nothrow)
      BufferFusionPassRunner("test_pass", create_fn, cycle_detector, buffer_fusion_optimizer);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  int find = 0;
  int total_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      EXPECT_EQ(node->GetName(), "conv");
      find = 1;
    }
    total_node++;
  }
  EXPECT_EQ(find, 1);
  EXPECT_EQ(total_node, 10);
}


/* Contains final cycle */
TEST_F(BufferFusionCycleDetectionUt, testcase_03_1) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  cycle_detection::BuildGraph03_1(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  BufferFusionOptimizerPtr buffer_fusion_optimizer = std::make_shared<BufferFusionOptimizer>();
  buffer_fusion_optimizer->Initialize(*graph);

  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) Conv2dAddRelu6MulMulFusionPass3(); };
  BufferFusionPassRunner *test_pass = new (std::nothrow)
      BufferFusionPassRunner("test_pass", create_fn, cycle_detector, buffer_fusion_optimizer);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  int find = 0;
  int total_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      EXPECT_EQ(node->GetName(), "conv");
      find = 1;
    }
    total_node++;
  }
  EXPECT_EQ(find, 1);
  EXPECT_EQ(total_node, 8);
}


/* Contains final cycle */
TEST_F(BufferFusionCycleDetectionUt, testcase_03_2) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  cycle_detection::BuildGraph03_1(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  BufferFusionOptimizerPtr buffer_fusion_optimizer = std::make_shared<BufferFusionOptimizer>();
  buffer_fusion_optimizer->Initialize(*graph);

  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) Conv2dAddRelu6MulMulFusionPass3_2(); };
  BufferFusionPassRunner *test_pass = new (std::nothrow)
      BufferFusionPassRunner("test_pass", create_fn, cycle_detector, buffer_fusion_optimizer);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  int find = 0;
  int total_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      EXPECT_EQ(node->GetName(), "convadd");
      find = 1;
    }
    total_node++;
  }
  EXPECT_EQ(find, 1);
  EXPECT_EQ(total_node, 7);
}

TEST_F(BufferFusionCycleDetectionUt, testcase_03_3) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  cycle_detection::BuildGraph03_1(graph);
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "conv") {
      (void)ge::AttrUtils::SetStr(node->GetOpDesc(), "_user_stream_label", "usr1");
    }
    if (node->GetName() == "add") {
      (void)ge::AttrUtils::SetStr(node->GetOpDesc(), "_user_stream_label", "usr2");
    }
  }

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  BufferFusionOptimizerPtr buffer_fusion_optimizer = std::make_shared<BufferFusionOptimizer>();
  buffer_fusion_optimizer->Initialize(*graph);

  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) Conv2dAddRelu6MulMulFusionPass3_2(); };
  BufferFusionPassRunner *test_pass = new (std::nothrow)
      BufferFusionPassRunner("test_pass", create_fn, cycle_detector, buffer_fusion_optimizer);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);
}

/* Contains final cycle.
 * There are some problems about matching multiple reference.
 * Currently, we will not enumerate every possible combination.
 * */
TEST_F(BufferFusionCycleDetectionUt, testcase_04) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  cycle_detection::BuildGraph04(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  BufferFusionOptimizerPtr buffer_fusion_optimizer = std::make_shared<BufferFusionOptimizer>();
  buffer_fusion_optimizer->Initialize(*graph);

  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) Conv2dAddRelu6MulMulFusionPass4(); };
  BufferFusionPassRunner *test_pass = new (std::nothrow)
      BufferFusionPassRunner("test_pass", create_fn, cycle_detector, buffer_fusion_optimizer);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  int find = 0;
  int total_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      EXPECT_EQ(node->GetName(), "conv/headconvadd/otherrelu6/othermul1/othermul2/other");
      find = 1;
    }
    total_node++;
  }
  EXPECT_EQ(find, 1);
  EXPECT_EQ(total_node, 14);
}

/* First longest path conv->add->relu6->mul1 will lead to cycle.
 * Second longest path conv/ohter->add/other->relu6/other->mul1/other
 * will be matched finally.
 * */
TEST_F(BufferFusionCycleDetectionUt, testcase_05) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  cycle_detection::BuildGraph05(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  BufferFusionOptimizerPtr buffer_fusion_optimizer = std::make_shared<BufferFusionOptimizer>();
  buffer_fusion_optimizer->Initialize(*graph);

  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) Conv2dAddRelu6MulMulFusionPass5(); };
  BufferFusionPassRunner *test_pass = new (std::nothrow)
      BufferFusionPassRunner("test_pass", create_fn, cycle_detector, buffer_fusion_optimizer);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  int find = 0;
  int total_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      EXPECT_EQ(node->GetName(), "conv/headconvadd/otherrelu6/othermul1/othermul2/other");
      find = 1;
    }
    total_node++;
  }
  EXPECT_EQ(find, 1);
  EXPECT_EQ(total_node, 16);
}

TEST_F(BufferFusionCycleDetectionUt, testcase_05_1) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  cycle_detection::BuildGraph05_1(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  BufferFusionOptimizerPtr buffer_fusion_optimizer = std::make_shared<BufferFusionOptimizer>();
  buffer_fusion_optimizer->Initialize(*graph);

  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) Conv2dAddRelu6MulMulFusionPass5(); };
  BufferFusionPassRunner *test_pass = new (std::nothrow)
      BufferFusionPassRunner("test_pass", create_fn, cycle_detector, buffer_fusion_optimizer);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  int find = 0;
  int total_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      EXPECT_EQ(node->GetName(), "conv/headconvadd/otherrelu6/othermul1/othermul2/other");
      find = 1;
    }
    total_node++;
  }
  EXPECT_EQ(find, 1);
  EXPECT_EQ(total_node, 18);
}

TEST_F(BufferFusionCycleDetectionUt, testcase_05_2) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  cycle_detection::BuildGraph05_2(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  BufferFusionOptimizerPtr buffer_fusion_optimizer = std::make_shared<BufferFusionOptimizer>();
  buffer_fusion_optimizer->Initialize(*graph);

  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) Conv2dAddRelu6MulMulFusionPass5(); };
  BufferFusionPassRunner *test_pass = new (std::nothrow)
      BufferFusionPassRunner("test_pass", create_fn, cycle_detector, buffer_fusion_optimizer);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  int find = 0;
  int total_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      EXPECT_NE(node->GetName(), "conv/headconvadd/otherrelu6/othermul1/othermul2/other");
      EXPECT_NE(node->GetName(), "conv/headconvaddrelu6mul1mul2mul3");
      find = 1;
    }
    total_node++;
  }
  EXPECT_EQ(find, 1);
  EXPECT_EQ(total_node, 24);
}

TEST_F(BufferFusionCycleDetectionUt, testcase_05_3) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  cycle_detection::BuildGraph05_3(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  BufferFusionOptimizerPtr buffer_fusion_optimizer = std::make_shared<BufferFusionOptimizer>();
  buffer_fusion_optimizer->Initialize(*graph);

  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) Conv2dAddRelu6MulMulFusionPass5(); };
  BufferFusionPassRunner *test_pass = new (std::nothrow)
      BufferFusionPassRunner("test_pass", create_fn, cycle_detector, buffer_fusion_optimizer);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  int find = 0;
  int total_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      EXPECT_EQ(node->GetName(), "conv/headconvaddrelu6mul1mul2mul3");
      find = 1;
    }
    total_node++;
  }
  EXPECT_EQ(find, 1);
  EXPECT_EQ(total_node, 17);
}

TEST_F(BufferFusionCycleDetectionUt, testcase_05_4) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  cycle_detection::BuildGraph05_4(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  BufferFusionOptimizerPtr buffer_fusion_optimizer = std::make_shared<BufferFusionOptimizer>();
  buffer_fusion_optimizer->Initialize(*graph);

  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) Conv2dAddRelu6MulMulFusionPass5(); };
  BufferFusionPassRunner *test_pass = new (std::nothrow)
      BufferFusionPassRunner("test_pass", create_fn, cycle_detector, buffer_fusion_optimizer);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  int find = 0;
  int total_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      EXPECT_EQ(node->GetName(), "conv/headconvadd/otherrelu6/othermul1/othermul2/othermul3/other");
      find = 1;
    }
    total_node++;
  }
  EXPECT_EQ(find, 1);
  EXPECT_EQ(total_node, 17);
}


TEST_F(BufferFusionCycleDetectionUt, testcase_05_5) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  cycle_detection::BuildGraph05_5(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  BufferFusionOptimizerPtr buffer_fusion_optimizer = std::make_shared<BufferFusionOptimizer>();
  buffer_fusion_optimizer->Initialize(*graph);

  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) Conv2dAddRelu6MulMulFusionPass5(); };
  BufferFusionPassRunner *test_pass = new (std::nothrow)
      BufferFusionPassRunner("test_pass", create_fn, cycle_detector, buffer_fusion_optimizer);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  int find = 0;
  int total_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      EXPECT_EQ(node->GetName(), "conv/headconvaddrelu6mul1mul2");
      find = 1;
    }
    total_node++;
  }
  EXPECT_EQ(find, 1);
  EXPECT_EQ(total_node, 18);
}

TEST_F(BufferFusionCycleDetectionUt, testcase_06) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  cycle_detection::BuildGraph05_4(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  BufferFusionOptimizerPtr buffer_fusion_optimizer = std::make_shared<BufferFusionOptimizer>();
  buffer_fusion_optimizer->Initialize(*graph);

  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) Conv2dAddRelu6MulMulFusionPass6(); };
  BufferFusionPassRunner *test_pass = new (std::nothrow)
      BufferFusionPassRunner("test_pass", create_fn, cycle_detector, buffer_fusion_optimizer);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  int find = 0;
  int total_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      EXPECT_EQ(node->GetName(), "conv/headconvadd/otherrelu6/othermul1/other");
      find = 1;
    }
    total_node++;
  }
  EXPECT_EQ(find, 1);
  EXPECT_EQ(total_node, 19);
}

/* We will not check cycle related to TBE_PATTERN_OUTPUT_NODE */
TEST_F(BufferFusionCycleDetectionUt, testcase_07) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  cycle_detection::BuildGraph05_4(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  BufferFusionOptimizerPtr buffer_fusion_optimizer = std::make_shared<BufferFusionOptimizer>();
  buffer_fusion_optimizer->Initialize(*graph);

  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) Conv2dAddRelu6MulMulFusionPass7(); };
  BufferFusionPassRunner *test_pass = new (std::nothrow)
      BufferFusionPassRunner("test_pass", create_fn, cycle_detector, buffer_fusion_optimizer);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  int find = 0;
  int total_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      EXPECT_EQ(node->GetName(), "conv/headconvaddrelu6mul1");
      find = 1;
    }
    total_node++;
  }
  EXPECT_EQ(find, 1);
  EXPECT_EQ(total_node, 19);
}

TEST_F(BufferFusionCycleDetectionUt, testcase_08) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  cycle_detection::BuildGraph08(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  BufferFusionOptimizerPtr buffer_fusion_optimizer = std::make_shared<BufferFusionOptimizer>();
  buffer_fusion_optimizer->Initialize(*graph);

  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) Conv2dAddRelu6MulMulFusionPass8(); };
  BufferFusionPassRunner *test_pass = new (std::nothrow)
      BufferFusionPassRunner("test_pass", create_fn, cycle_detector, buffer_fusion_optimizer);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  int find = 0;
  int total_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      EXPECT_EQ(node->GetName(), "conv/headconvaddrelu6mul1mul2");
      find = 1;
    }
    total_node++;
  }
  EXPECT_EQ(find, 1);
  EXPECT_EQ(total_node, 12);
}

TEST_F(BufferFusionCycleDetectionUt, testcase_08_1) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  cycle_detection::BuildGraph08_1(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  BufferFusionOptimizerPtr buffer_fusion_optimizer = std::make_shared<BufferFusionOptimizer>();
  buffer_fusion_optimizer->Initialize(*graph);

  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) Conv2dAddRelu6MulMulFusionPass8(); };
  BufferFusionPassRunner *test_pass = new (std::nothrow)
      BufferFusionPassRunner("test_pass", create_fn, cycle_detector, buffer_fusion_optimizer);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  int find = 0;
  int total_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      EXPECT_EQ(node->GetName(), "conv/headconvaddrelu6mul1mul2");
      find = 1;
    }
    total_node++;
  }
  EXPECT_EQ(find, 1);
  EXPECT_EQ(total_node, 13);
}

TEST_F(BufferFusionCycleDetectionUt, testcase_08_2) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  cycle_detection::BuildGraph08_2(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  BufferFusionOptimizerPtr buffer_fusion_optimizer = std::make_shared<BufferFusionOptimizer>();
  buffer_fusion_optimizer->Initialize(*graph);

  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) Conv2dAddRelu6MulMulFusionPass8(); };
  BufferFusionPassRunner *test_pass = new (std::nothrow)
      BufferFusionPassRunner("test_pass", create_fn, cycle_detector, buffer_fusion_optimizer);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  int find = 0;
  int total_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      EXPECT_EQ(node->GetName(), "conv/headconvaddrelu6mul1mul2mul3");
      find = 1;
    }
    total_node++;
  }
  EXPECT_EQ(find, 1);
  EXPECT_EQ(total_node, 11);
}


TEST_F(BufferFusionCycleDetectionUt, testcase_08_3) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  cycle_detection::BuildGraph08_3(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  BufferFusionOptimizerPtr buffer_fusion_optimizer = std::make_shared<BufferFusionOptimizer>();
  buffer_fusion_optimizer->Initialize(*graph);

  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) Conv2dAddRelu6MulMulFusionPass8(); };
  BufferFusionPassRunner *test_pass = new (std::nothrow)
      BufferFusionPassRunner("test_pass", create_fn, cycle_detector, buffer_fusion_optimizer);

  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  int find = 0;
  int total_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      EXPECT_NE(node->GetName(), "conv/headconvaddrelu6mul1mul2mul3");
      find = 1;
    }
    total_node++;
  }
  EXPECT_EQ(find, 1);
  EXPECT_EQ(total_node, 17);
}

TEST_F(BufferFusionCycleDetectionUt, testcase_08_4) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  cycle_detection::BuildGraph08_4(graph);

  sub_graph_optimizer_ptr_->engine_name_ = fe::AI_CORE_NAME;

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  BufferFusionOptimizerPtr buffer_fusion_optimizer = std::make_shared<BufferFusionOptimizer>();
  buffer_fusion_optimizer->Initialize(*graph);

  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) Conv2dAddRelu6MulMulFusionPass8(); };
  BufferFusionPassRunner *test_pass = new (std::nothrow)
      BufferFusionPassRunner("test_pass", create_fn, cycle_detector, buffer_fusion_optimizer);

  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  int find = 0;
  int total_node = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetType() == "Conv2D") {
      EXPECT_NE(node->GetName(), "conv/headconvaddrelu6mul1mul2mul3");
      find = 1;
    }
    total_node++;
  }
  EXPECT_EQ(find, 1);
  EXPECT_EQ(total_node, 17);
}

TEST_F(BufferFusionCycleDetectionUt, coverage_01) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr mul = std::make_shared<ge::OpDesc>("mul", "Mul");
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddInputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  mul->AddOutputDesc(ge::GeTensorDesc(ge::GeShape({4, 4}), ge::FORMAT_ND, ge::DT_DOUBLE));
  ge::AttrUtils::SetInt(mul, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
  ge::NodePtr mul_node = graph->AddNode(mul);
  string op_pattern;
  BufferFusionOptimizerPtr buffer_fusion_optimizer = std::make_shared<BufferFusionOptimizer>();
  buffer_fusion_optimizer->Initialize(*graph);
}
