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
#include "fe_llt_utils.h"
#define protected public
#define private public
#include "common/graph_comm.h"
#include "pass_manager.h"
#include "common/platform_utils.h"
#include "common/configuration.h"
#include "graph/ge_local_context.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/op_kernel_bin.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/ub_fusion/buffer_fusion.h"
#include "graph_optimizer/ub_fusion/base_buffer_fusion_pass_runner.h"
#include "graph_optimizer/ub_fusion/auto_buffer_fusion_pass_runner.h"
#include "graph_optimizer/op_compiler/op_compiler.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "common/fusion_statistic/fusion_statistic_writer.h"
#include "common/fusion_op_comm.h"
#include "ops_store/ops_kernel_manager.h"
#undef protected
#undef private
#include "graph_constructor.h"
using namespace std;
using namespace fe;
using namespace ge;
using namespace te;

namespace {
  int32_t fusion_nodes_num = 0;
  int32_t last_fusion_nodes_num = 0;
}
class UB_FUSION_UT_AUTO_FUSION_PASS_RUNNER : public testing::Test {
 public:
 protected:
  virtual void SetUp() {
    graph_comm_ptr_ = std::make_shared<GraphComm>("engineName");
    graph_comm_ptr_->Initialize();
    fusion_priority_mgr_ptr_ = std::make_shared<FusionPriorityManager>(
        "engineName", nullptr);
    tbe_adapter_ptr_ = std::dynamic_pointer_cast<TbeOpStoreAdapter>(OpStoreAdapterManager::Instance(AI_CORE_NAME).GetOpStoreAdapter(EN_IMPL_HW_TBE));
    tbe_adapter_ptr_->Initialize(std::map<std::string, std::string>());
    tbe_adapter_ptr_->support_parallel_compile = false;

    sub_graph_optimizer_ptr_ =
        std::make_shared<BufferFusion>(graph_comm_ptr_, fusion_priority_mgr_ptr_, tbe_adapter_ptr_);
    lx_fusion_optimizer_ = std::make_shared<LxFusionOptimizer>(fusion_priority_mgr_ptr_, nullptr);
    lx_fusion_optimizer_->Initialize();
  }

  virtual void TearDown() {

  }

  std::shared_ptr<GraphComm> graph_comm_ptr_;
  std::shared_ptr<FusionPriorityManager> fusion_priority_mgr_ptr_;
  std::shared_ptr<BufferFusion> sub_graph_optimizer_ptr_;
  TbeOpStoreAdapterPtr tbe_adapter_ptr_;
  LxFusionOptimizerPtr lx_fusion_optimizer_;
};

te::OpBuildResCode TeFusionFusionCheckStub1(std::vector<Node*> teGraphNode, OpDescPtr op_desc_ptr, const std::vector<ge::NodePtr>
        &to_be_del, uint64_t taskid, uint64_t tid, const std::string op_compile_strategy)
{
  fusion_nodes_num++;
  return te::OP_BUILD_SUCC;
}

te::OpBuildResCode TeFusionFusionCheckStub2(std::vector<Node*> teGraphNode, OpDescPtr op_desc_ptr, const std::vector<ge::NodePtr>
        &to_be_del, uint64_t taskid, uint64_t tid, const std::string op_compile_strategy)
{
  fusion_nodes_num++;
  int32_t reduce_node_num = 0;
  for (auto node : teGraphNode) {
    if (node->GetType() == "ReduceMeanD") {
      reduce_node_num++;
    }
  }
  if (reduce_node_num >= 2) {
    for (auto node : teGraphNode) {
      if (node->GetOpDesc()->HasAttr(SCOPE_ID_ATTR)) {
        node->GetOpDesc()->DelAttr(SCOPE_ID_ATTR);
      }
    }
  }
  return te::OP_BUILD_SUCC;
}

bool WaitAllFinishedFusionCheckdStub1(uint64_t tid, vector<te::FinComTask> &fin_task)
{
  te::FinComTask fin_com_task;
  fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("OneOP", "");
  fin_com_task.taskId = GetAtomicId()- fusion_nodes_num;
  fin_com_task.status = 0;
  ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "jsonFilePath");
  fin_task.push_back(fin_com_task);
  return true;
}

bool WaitAllFinishedFusionCheckdStub2(uint64_t tid, vector<te::FinComTask> &fin_task)
{
  te::FinComTask fin_com_task;
  fin_com_task.teNodeOpDesc = std::make_shared<ge::OpDesc>("OneOP", "");
  fin_com_task.taskId = GetAtomicId()- (fusion_nodes_num - last_fusion_nodes_num);
  if (!last_fusion_nodes_num) {
    last_fusion_nodes_num = fusion_nodes_num;
  }
  fin_com_task.status = 0;
  ge::AttrUtils::SetStr(fin_com_task.teNodeOpDesc, "json_file_path", "jsonFilePath");
  fin_task.push_back(fin_com_task);
  return true;
}

te::LX_QUERY_STATUS get_tbe_opinfo_fusion_check_stub_succ(const te::TbeOpInfo &info, std::string &op_info) {
  return te::LX_QUERY_SUCC;
}


class SingleReduceFusionPass1 : public BufferFusionPassBase {
 public:
  SingleReduceFusionPass1() {}
  ~SingleReduceFusionPass1() {}

 protected:
  vector<BufferFusionPattern*> DefinePatterns() override {
    vector<BufferFusionPattern*> patterns;
    string pattern_name = "SingleReduceFusionPass1";
    BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pattern_name);
    FE_CHECK((pattern == nullptr), FE_LOGE("new an object failed"), return patterns);
    FE_LOGD("Start to define %s pass pattern", pattern_name.c_str());
    pattern->AddOpDesc("reduce", {"CommReduce"}, 1, 1, true)
            .AddOpDesc("elemwise_or_broadcast_1", {"ElemWise", "Broadcast"}, 0, 5, true);
    patterns.push_back(pattern);
    FE_LOGD("End to define %s pass pattern.", pattern_name.c_str());
    return patterns;
  }
};

class SingleNormFusionPass1 : public BufferFusionPassBase {
 public:
  SingleNormFusionPass1() {}
  ~SingleNormFusionPass1() {}

 protected:
  vector<BufferFusionPattern*> DefinePatterns() override {
    vector<BufferFusionPattern*> patterns;
    string pattern_name = "SingleNormFusionPass1";
    BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pattern_name);
    FE_CHECK((pattern == nullptr), FE_LOGE("new an object failed"), return patterns);
    FE_LOGD("Start to define %s pass pattern", pattern_name.c_str());
    pattern->AddOpDesc("norm", {"Softmax"}, 1, 1, false)
            .AddOpDesc("elemwise_or_broadcast_1", {"ElemWise", "Broadcast"}, 0, 5, true);
    patterns.push_back(pattern);
    FE_LOGD("End to define %s pass pattern.", pattern_name.c_str());
    return patterns;
  }
};

class MultipleReduceFusionPass1 : public BufferFusionPassBase {
 public:
  MultipleReduceFusionPass1() {}
  ~MultipleReduceFusionPass1() {}

 protected:
  vector<BufferFusionPattern*> DefinePatterns() override {
    vector<BufferFusionPattern*> patterns;
    string pattern_name = "MultipleReduceFusionPass1";
    BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pattern_name);
    FE_CHECK((pattern == nullptr), FE_LOGE("new an object failed"), return patterns);
    FE_LOGD("Start to define %s pass pattern", pattern_name.c_str());
    pattern->AddOpDesc("reduce", {"CommReduce"}, 1, 5, false)
            .AddOpDesc("elemwise_or_broadcast_1", {"ElemWise", "Broadcast"}, 0, 6, true);
    patterns.push_back(pattern);
    FE_LOGD("End to define %s pass pattern.", pattern_name.c_str());
    return patterns;
  }
};

class MultipleNormFusionPass2 : public BufferFusionPassBase {
 public:
  MultipleNormFusionPass2() {}
  ~MultipleNormFusionPass2() {}

 protected:
  vector<BufferFusionPattern*> DefinePatterns() override {
    vector<BufferFusionPattern*> patterns;
    string pattern_name = "MultipleNormFusionPass2";
    BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pattern_name);
    FE_CHECK((pattern == nullptr), FE_LOGE("new an object failed"), return patterns);
    FE_LOGD("Start to define %s pass pattern", pattern_name.c_str());
    pattern->AddOpDesc("norm", {"Softmax"}, 1, 5, false)
            .AddOpDesc("elemwise_or_broadcast_1", {"ElemWise", "Broadcast"}, 0, 5, true);
    patterns.push_back(pattern);
    FE_LOGD("End to define %s pass pattern.", pattern_name.c_str());
    return patterns;
  }
};

class MultipleReduceFusionPass3_1 : public BufferFusionPassBase {
 public:
  MultipleReduceFusionPass3_1() {}
  ~MultipleReduceFusionPass3_1() {}

 protected:
  vector<BufferFusionPattern*> DefinePatterns() override {
    vector<BufferFusionPattern*> patterns;
    string pattern_name = "MultipleReduceFusionPass3_1";
    BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pattern_name);
    FE_CHECK((pattern == nullptr), FE_LOGE("new an object failed"), return patterns);
    FE_LOGD("Start to define %s pass pattern", pattern_name.c_str());
    pattern->AddOpDesc("reduce", {"CommReduce"}, 1, 5, false)
            .AddOpDesc("elemwise_or_broadcast_1", {"ElemWise", "Broadcast"}, 0, 5, true);
    patterns.push_back(pattern);
    FE_LOGD("End to define %s pass pattern.", pattern_name.c_str());
    return patterns;
  }
};

class MultipleReduceFusionPass3_2 : public BufferFusionPassBase {
 public:
  MultipleReduceFusionPass3_2() {}
  ~MultipleReduceFusionPass3_2() {}

 protected:
  vector<BufferFusionPattern*> DefinePatterns() override {
    vector<BufferFusionPattern*> patterns;
    string pattern_name = "MultipleReduceFusionPass3_2";
    BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pattern_name);
    FE_CHECK((pattern == nullptr), FE_LOGE("new an object failed"), return patterns);
    FE_LOGD("Start to define %s pass pattern", pattern_name.c_str());
    pattern->AddOpDesc("reduce", {"CommReduce"}, 1, 1, true)
            .AddOpDesc("elemwise_or_broadcast_1", {"ElemWise", "Broadcast"}, 0, 5, true);
    patterns.push_back(pattern);
    FE_LOGD("End to define %s pass pattern.", pattern_name.c_str());
    return patterns;
  }
};

class MultipleReduceAndDataaccessFusionPass1 : public BufferFusionPassBase {
 public:
  MultipleReduceAndDataaccessFusionPass1() {}
  ~MultipleReduceAndDataaccessFusionPass1() {}

 protected:
  vector<BufferFusionPattern*> DefinePatterns() override {
    vector<BufferFusionPattern*> patterns;
    string pattern_name = "MultipleReduceAndDataaccessFusionPass1";
    BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pattern_name);
    FE_CHECK((pattern == nullptr), FE_LOGE("new an object failed"), return patterns);
    FE_LOGD("Start to define %s pass pattern", pattern_name.c_str());
    pattern->AddOpDesc("reduce", {"CommReduce"}, 1, 5, false)
            .AddOpDesc("gather", {"GatherV2"}, 0, 5, false)
            .AddOpDesc("elemwise_or_broadcast_1", {"ElemWise", "Broadcast"}, 0, 5, true);
    pattern->SetRelation("reduce", "gather", PatternRelation::RELATIVE_POSITION_CONSISTENT);
    patterns.push_back(pattern);
    FE_LOGD("End to define %s pass pattern.", pattern_name.c_str());
    return patterns;
  }
};

class TestFusionPass1 : public BufferFusionPassBase {
 public:
  TestFusionPass1() {}
  ~TestFusionPass1() {}

Status CalcFusionOpSliceInfo(std::vector<ge::NodePtr> &fusion_nodes, OpCalcInfo &op_slice_info) {
  return fe::FAILED;
}

 protected:
  vector<BufferFusionPattern*> DefinePatterns() override {
    return {};
  }
};

class TestFusionPass2 : public BufferFusionPassBase {
 public:
  TestFusionPass2() {}
  ~TestFusionPass2() {}

 protected:
  vector<BufferFusionPattern*> DefinePatterns() override {
    return {};
  }
};

using BufferFusionFn = BufferFusionPassBase *(*)();
fe::BufferFusionPassBase *BufferFunc() {
  return new(std::nothrow) TestFusionPass2();
}

void RegisterBufferFusionFunc(BufferFusionFn create_fn) {
  BufferFusionPassRegistry::GetInstance().RegisterPass(
      BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, "BUILT_IN_PASS1", create_fn, 0);
  BufferFusionPassRegistry::GetInstance().RegisterPass(
      BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, "BUILT_IN_PASS2", create_fn, 0);
}

std::vector<BufferFusionInfo> SortedBufferAutoFusionFun() {
  std::vector<BufferFusionInfo> sorted_buffer_auto_fusion_vec;
  BufferFusionPassBase *(*create_fn)() = nullptr;
  create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) MultipleReduceFusionPass3_1(); };
  auto buffer_fusion_info = BufferFusionInfo("MultipleReduceFusionPass3_1", 5790, true, false, create_fn);
  sorted_buffer_auto_fusion_vec.emplace_back(buffer_fusion_info);
  create_fn = nullptr;
  create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) MultipleReduceFusionPass3_2(); };
  buffer_fusion_info = BufferFusionInfo("MultipleReduceFusionPass3_2", 5790, true, false, create_fn);
  sorted_buffer_auto_fusion_vec.emplace_back(buffer_fusion_info);
  return sorted_buffer_auto_fusion_vec;
}

/* case1、3、5 */
void BuildGraph_01(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu1", "Relu", 1, 1)
      .Attr("relu1", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .Attr("relu1", OPS_PATH_NAME_PREFIX, "")
      .AddOpDesc(EN_IMPL_HW_TBE, "CommReduce", "reduceMeanD1", "ReduceMeanD", 1, 1)
      .Attr("reduceMeanD1", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .Attr("reduceMeanD1", OPS_PATH_NAME_PREFIX, "")
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "square1", "Square", 1, 1)
      .Attr("square1", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .Attr("square1", OPS_PATH_NAME_PREFIX, "")
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add1", "Add", 2, 1)
      .Attr("add1", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .Attr("add1", OPS_PATH_NAME_PREFIX, "")
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu2", "Relu", 1, 1)
      .Attr("relu2", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .Attr("relu2", OPS_PATH_NAME_PREFIX, "")
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu3", "Relu", 1, 1)
      .Attr("relu3", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .Attr("relu3", OPS_PATH_NAME_PREFIX, "")
      .AddOpDesc(EN_IMPL_HW_TBE, "CommReduce", "reduceMeanD2", "ReduceMeanD", 1, 1)
      .Attr("reduceMeanD2", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .Attr("reduceMeanD2", OPS_PATH_NAME_PREFIX, "")      
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "square2", "Square", 1, 1)
      .Attr("square2", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .Attr("square2", OPS_PATH_NAME_PREFIX, "")

      .SetInput("reduceMeanD1", "relu1")
      .SetInput("square1", "reduceMeanD1")
      .SetInputs("add1", {"square1", "square2"})
      .SetInput("relu3", "add1")
      .SetInput("reduceMeanD2", "relu2")
      .SetInput("square2", "reduceMeanD2");
  test.DumpGraph(graph);
  std::cout << "Build Graph_01 successfully." << std::endl;
}

/* case2 */
void BuildGraph_02(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "Softmax", "softmax1", "SoftmaxV2", 1, 1)
      .Attr("softmax1", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "Softmax", "softmax2", "SoftmaxV2", 1, 1)
      .Attr("softmax2", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add1", "Add", 2, 1)
      .Attr("add1", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu1", "Relu", 1, 1)
      .Attr("relu1", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu2", "Relu", 1, 1)
      .Attr("relu2", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu3", "Relu", 1, 1)
      .Attr("relu3", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "square1", "Square", 1, 1)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "square2", "Square", 1, 1)

      .SetInput("softmax1", "relu1")
      .SetInput("softmax2", "relu2")
      .SetInput("square1", "softmax1")
      .SetInput("square2", "softmax2")
      .SetInputs("add1", {"square1", "square2"})
      .SetInput("relu3", "add1");
  test.DumpGraph(graph);
  std::cout << "Build Graph_02 successfully." << std::endl;
}

/* case4 */
void BuildGraph_03(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "Softmax", "softmax1", "SoftmaxV2", 1, 1)
      .Attr("softmax1", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "Softmax", "softmax2", "SoftmaxV2", 1, 1)
      .Attr("softmax2", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu1", "Relu", 1, 1)
      .Attr("relu1", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu2", "Relu", 1, 1)
      .Attr("relu2", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "square1", "Square", 1, 1)
      .Attr("square1", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "square2", "Square", 1, 1)
      .Attr("square2", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)

      .SetInput("softmax1", "relu1")
      .SetInput("softmax2", "relu2")
      .SetInput("square1", "softmax1")
      .SetInput("square2", "softmax2")
      .SetInput("relu2", "square1");
  test.DumpGraph(graph);
  std::cout << "Build Graph_03 successfully." << std::endl;
}

/* case6 */
void BuildGraph_04(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu1", "Relu", 1, 1)
      .Attr("relu1", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "CommReduce", "reduceMeanD1", "ReduceMeanD", 1, 1)
      .Attr("reduceMeanD1", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "CommReduce", "reduceMeanD2", "ReduceMeanD", 1, 1)
      .Attr("reduceMeanD2", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "GatherV2", "gather1", "GatherV2", 1, 1)
      .Attr("gather1", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "GatherV2", "gather2", "GatherV2", 1, 1)
      .Attr("gather2", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add1", "Add", 2, 1)
      .Attr("add1", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu2", "Relu", 1, 1)
      .Attr("relu2", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu3", "Relu", 1, 1)
      .Attr("relu3", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)

      .SetInput("reduceMeanD1", "relu1")
      .SetInput("gather1", "reduceMeanD1")
      .SetInputs("add1", {"gather1", "gather2"})
      .SetInput("relu3", "add1")
      .SetInput("reduceMeanD2", "relu2")
      .SetInput("gather2", "reduceMeanD2");
  test.DumpGraph(graph);
  std::cout << "Build Graph_04 successfully." << std::endl;
}

/* case7 */
void BuildGraph_05(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu1", "Relu", 1, 1)
      .Attr("relu1", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "CommReduce", "reduceMeanD1", "ReduceMeanD", 1, 1)
      .Attr("reduceMeanD1", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "CommReduce", "reduceMeanD2", "ReduceMeanD", 1, 1)
      .Attr("reduceMeanD2", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "GatherV2", "gather1", "GatherV2", 1, 1)
      .Attr("gather1", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "GatherV2", "gather2", "GatherV2", 1, 1)
      .Attr("gather2", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add1", "Add", 2, 1)
      .Attr("add1", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu2", "Relu", 1, 1)
      .Attr("relu2", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu3", "Relu", 1, 1)
      .Attr("relu3", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)

      .SetInput("gather1", "relu1")
      .SetInput("gather2", "relu2")
      .SetInput("reduceMeanD1", "relu1")
      .SetInput("reduceMeanD2", "relu2")
      .SetInputs("add1", {"reduceMeanD1", "reduceMeanD2"})
      .SetInput("relu3", "add1");
  test.DumpGraph(graph);
  std::cout << "Build Graph_05 successfully." << std::endl;
}

/* case8 */
void BuildGraph_06(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu1", "Relu", 1, 1)
      .Attr("relu1", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "CommReduce", "reduceMeanD1", "ReduceMeanD", 1, 1)
      .Attr("reduceMeanD1", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "CommReduce", "reduceMeanD2", "ReduceMeanD", 1, 1)
      .Attr("reduceMeanD2", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "GatherV2", "gather1", "GatherV2", 1, 1)
      .Attr("gather1", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "GatherV2", "gather2", "GatherV2", 1, 1)
      .Attr("gather2", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add1", "Add", 2, 1)
      .Attr("add1", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu2", "Relu", 1, 1)
      .Attr("relu2", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)
      .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "relu3", "Relu", 1, 1)
      .Attr("relu3", ATTR_NAME_IS_OP_DYNAMIC_IMPL, true)

      .SetInput("reduceMeanD1", "relu1")
      .SetInput("gather1", "reduceMeanD1")
      .SetInputs("add1", {"gather1", "reduceMeanD2"})
      .SetInput("gather2", "relu2")
      .SetInput("reduceMeanD2", "gather2")
      .SetInput("relu3", "add1");
  test.DumpGraph(graph);
  std::cout << "Build Graph_06 successfully." << std::endl;
}

/* case9 */
void BuildGraph_07(ge::ComputeGraphPtr &graph) {
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "CommReduce", "reduceMeanD1", "ReduceMeanD", 1, 1)
      .Attr("reduceMeanD1", SCOPE_ID_ATTR, 1)
      .Attr("reduceMeanD1", kOpPattern, "CommReduce")
      .AddOpDesc(EN_IMPL_HW_TBE, "CommReduce", "reduceMeanD2", "ReduceMeanD", 1, 1)
      .Attr("reduceMeanD2", SCOPE_ID_ATTR, 2);
  test.DumpGraph(graph);
  std::cout << "Build Graph_07 successfully." << std::endl;
}

TEST_F(UB_FUSION_UT_AUTO_FUSION_PASS_RUNNER, singel_reduce_fusion_01) {
  tbe_adapter_ptr_->TeFusion = TeFusionFusionCheckStub1;
  tbe_adapter_ptr_->WaitAllFinished = WaitAllFinishedFusionCheckdStub1;
  tbe_adapter_ptr_->GetOpInfo = get_tbe_opinfo_fusion_check_stub_succ;
  fusion_nodes_num = 0;
  last_fusion_nodes_num = 0;
  string path = GetCodeDir() + "/tests/engines/nn_engine/config/data/platform_config";
  string real_path = fe::RealPath(path);
  fe::PlatformInfoManager::Instance().platform_info_map_.clear();
  fe::PlatformInfoManager::Instance().platform_infos_map_.clear();
  uint32_t init_ret = fe::PlatformInfoManager::Instance().LoadConfigFile(real_path);
  EXPECT_EQ(init_ret, ge::GRAPH_SUCCESS);
  fe::PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910B1");

  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_01(graph);
  
  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) SingleReduceFusionPass1(); };
  AutoBufferFusionPassRunner *test_pass = new (std::nothrow)
      AutoBufferFusionPassRunner("test_pass", create_fn, cycle_detector, tbe_adapter_ptr_);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  const std::vector<std::string> nodes_name_vec = {"relu2reduceMeanD2", "relu1reduceMeanD1square1square2add1relu3"};
  int index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetName(), nodes_name_vec.at(index));
    index++;
  }
}

TEST_F(UB_FUSION_UT_AUTO_FUSION_PASS_RUNNER, singel_norm_fusion_01) {
  tbe_adapter_ptr_->TeFusion = TeFusionFusionCheckStub1;
  tbe_adapter_ptr_->WaitAllFinished = WaitAllFinishedFusionCheckdStub1;
  tbe_adapter_ptr_->GetOpInfo = get_tbe_opinfo_fusion_check_stub_succ;
  fusion_nodes_num = 0;
  last_fusion_nodes_num = 0;
  string path = GetCodeDir() + "/tests/engines/nn_engine/config/data/platform_config";
  string real_path = fe::RealPath(path);
  fe::PlatformInfoManager::Instance().platform_info_map_.clear();
  fe::PlatformInfoManager::Instance().platform_infos_map_.clear();
  uint32_t init_ret = fe::PlatformInfoManager::Instance().LoadConfigFile(real_path);
  EXPECT_EQ(init_ret, ge::GRAPH_SUCCESS);
  fe::PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend310P3");

  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_02(graph);
  
  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) SingleNormFusionPass1(); };
  AutoBufferFusionPassRunner *test_pass = new (std::nothrow)
      AutoBufferFusionPassRunner("test_pass", create_fn, cycle_detector, tbe_adapter_ptr_);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  const std::vector<std::string> nodes_name_vec = {"relu1softmax1", "square1", "relu2softmax2",
                                                   "square2", "add1", "relu3"};
  int index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetName(), nodes_name_vec.at(index));
    index++;
  }
}

TEST_F(UB_FUSION_UT_AUTO_FUSION_PASS_RUNNER, multiple_reduce_fusion_01) {
  tbe_adapter_ptr_->TeFusion = TeFusionFusionCheckStub1;
  tbe_adapter_ptr_->WaitAllFinished = WaitAllFinishedFusionCheckdStub1;
  tbe_adapter_ptr_->GetOpInfo = get_tbe_opinfo_fusion_check_stub_succ;
  fusion_nodes_num = 0;
  last_fusion_nodes_num = 0;
  string path = GetCodeDir() + "/tests/engines/nn_engine/config/data/platform_config";
  string real_path = fe::RealPath(path);
  fe::PlatformInfoManager::Instance().platform_info_map_.clear();
  fe::PlatformInfoManager::Instance().platform_infos_map_.clear();
  uint32_t init_ret = fe::PlatformInfoManager::Instance().LoadConfigFile(real_path);
  EXPECT_EQ(init_ret, ge::GRAPH_SUCCESS);
  fe::PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910B1");

  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_01(graph);
  
  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) MultipleReduceFusionPass1(); };
  AutoBufferFusionPassRunner *test_pass = new (std::nothrow)
      AutoBufferFusionPassRunner("test_pass", create_fn, cycle_detector, tbe_adapter_ptr_);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  const std::vector<std::string> nodes_name_vec = {"relu1reduceMeanD1square1relu2reduceMeanD2square2add1relu3"};
  int index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetName(), nodes_name_vec.at(index));
    index++;
  }
}

TEST_F(UB_FUSION_UT_AUTO_FUSION_PASS_RUNNER, multiple_norm_fusion_02) {
  tbe_adapter_ptr_->TeFusion = TeFusionFusionCheckStub1;
  tbe_adapter_ptr_->WaitAllFinished = WaitAllFinishedFusionCheckdStub1;
  tbe_adapter_ptr_->GetOpInfo = get_tbe_opinfo_fusion_check_stub_succ;
  fusion_nodes_num = 0;
  last_fusion_nodes_num = 0;
  string path = GetCodeDir() + "/tests/engines/nn_engine/config/data/platform_config";
  string real_path = fe::RealPath(path);
  fe::PlatformInfoManager::Instance().platform_info_map_.clear();
  fe::PlatformInfoManager::Instance().platform_infos_map_.clear();
  uint32_t init_ret = fe::PlatformInfoManager::Instance().LoadConfigFile(real_path);
  EXPECT_EQ(init_ret, ge::GRAPH_SUCCESS);
  fe::PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910B1");

  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_03(graph);
  
  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) MultipleNormFusionPass2(); };
  AutoBufferFusionPassRunner *test_pass = new (std::nothrow)
      AutoBufferFusionPassRunner("test_pass", create_fn, cycle_detector, tbe_adapter_ptr_);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  const std::vector<std::string> nodes_name_vec = {"relu1softmax1square1relu2", "softmax2square2"};
  int index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetName(), nodes_name_vec.at(index));
    index++;
  }
}

TEST_F(UB_FUSION_UT_AUTO_FUSION_PASS_RUNNER, multiple_reduce_fusion_03) {
  tbe_adapter_ptr_->TeFusion = TeFusionFusionCheckStub2;
  tbe_adapter_ptr_->WaitAllFinished = WaitAllFinishedFusionCheckdStub2;
  tbe_adapter_ptr_->GetOpInfo = get_tbe_opinfo_fusion_check_stub_succ;
  fusion_nodes_num = 0;
  last_fusion_nodes_num = 0;
  string path = GetCodeDir() + "/tests/engines/nn_engine/config/data/platform_config";
  string real_path = fe::RealPath(path);
  fe::PlatformInfoManager::Instance().platform_info_map_.clear();
  fe::PlatformInfoManager::Instance().platform_infos_map_.clear();
  uint32_t init_ret = fe::PlatformInfoManager::Instance().LoadConfigFile(real_path);
  EXPECT_EQ(init_ret, ge::GRAPH_SUCCESS);
  fe::PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910B1");

  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_01(graph);

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  
  std::shared_ptr<GraphComm> graph_comm_ptr = std::make_shared<GraphComm>(fe::AI_CORE_NAME);
  graph_comm_ptr->Initialize();
  std::shared_ptr<FusionPriorityManager> fusion_priority_mgr_ptr =
      std::make_shared<FusionPriorityManager>(fe::AI_CORE_NAME, nullptr);
  std::vector<BufferFusionInfo> sorted_buffer_fusion_vec = SortedBufferAutoFusionFun();
  fusion_priority_mgr_ptr->sorted_buffer_fusion_map_[FusionPriorityManager::GetCurrentHashedKey()] = sorted_buffer_fusion_vec;
  std::shared_ptr<BufferFusion> sub_graph_optimizer_ptr = 
    std::make_shared<BufferFusion>(graph_comm_ptr, fusion_priority_mgr_ptr, tbe_adapter_ptr_);
  sub_graph_optimizer_ptr->cycle_detector_ = cycle_detector;
  BufferFusionPassType type = BUILT_IN_AI_CORE_BUFFER_FUSION_PASS;
  Status status = sub_graph_optimizer_ptr->RunRegisterBufferFusionPass(*graph, type);
  EXPECT_EQ(status, fe::SUCCESS);
  sub_graph_optimizer_ptr->BuildFusionGraph(*graph);

  const std::vector<std::string> nodes_name_vec = {"relu2reduceMeanD2", "relu1reduceMeanD1square1square2add1relu3"};
  int index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetName(), nodes_name_vec.at(index));
    index++;
  }
}

TEST_F(UB_FUSION_UT_AUTO_FUSION_PASS_RUNNER, multiple_reduce_and_dataaccess_fusion_01) {
  tbe_adapter_ptr_->TeFusion = TeFusionFusionCheckStub1;
  tbe_adapter_ptr_->WaitAllFinished = WaitAllFinishedFusionCheckdStub1;
  tbe_adapter_ptr_->GetOpInfo = get_tbe_opinfo_fusion_check_stub_succ;
  fusion_nodes_num = 0;
  last_fusion_nodes_num = 0;
  string path = GetCodeDir() + "/tests/engines/nn_engine/config/data/platform_config";
  string real_path = fe::RealPath(path);
  fe::PlatformInfoManager::Instance().platform_info_map_.clear();
  fe::PlatformInfoManager::Instance().platform_infos_map_.clear();
  uint32_t init_ret = fe::PlatformInfoManager::Instance().LoadConfigFile(real_path);
  EXPECT_EQ(init_ret, ge::GRAPH_SUCCESS);
  fe::PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910B1");

  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_04(graph);
  
  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) MultipleReduceAndDataaccessFusionPass1(); };
  AutoBufferFusionPassRunner *test_pass = new (std::nothrow)
      AutoBufferFusionPassRunner("test_pass", create_fn, cycle_detector, tbe_adapter_ptr_);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  const std::vector<std::string> nodes_name_vec = {"relu1reduceMeanD1gather1relu2reduceMeanD2gather2add1relu3"};
  int index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetName(), nodes_name_vec.at(index));
    index++;
  }
}

TEST_F(UB_FUSION_UT_AUTO_FUSION_PASS_RUNNER, multiple_reduce_and_dataaccess_fusion_02) {
  fusion_nodes_num = 0;
  last_fusion_nodes_num = 0;
  string path = GetCodeDir() + "/tests/engines/nn_engine/config/data/platform_config";
  string real_path = fe::RealPath(path);
  fe::PlatformInfoManager::Instance().platform_info_map_.clear();
  fe::PlatformInfoManager::Instance().platform_infos_map_.clear();
  uint32_t init_ret = fe::PlatformInfoManager::Instance().LoadConfigFile(real_path);
  EXPECT_EQ(init_ret, ge::GRAPH_SUCCESS);
  fe::PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910B1");

  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_05(graph);
  
  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) MultipleReduceAndDataaccessFusionPass1(); };
  AutoBufferFusionPassRunner *test_pass = new (std::nothrow)
      AutoBufferFusionPassRunner("test_pass", create_fn, cycle_detector, tbe_adapter_ptr_);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  const std::vector<std::string> nodes_name_vec = {"relu1reduceMeanD1gather1relu2reduceMeanD2add1relu3gather2"};
  int index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetName(), nodes_name_vec.at(index));
    index++;
  }
}

TEST_F(UB_FUSION_UT_AUTO_FUSION_PASS_RUNNER, multiple_reduce_and_dataaccess_fusion_03) {
  tbe_adapter_ptr_->TeFusion = TeFusionFusionCheckStub1;
  tbe_adapter_ptr_->WaitAllFinished = WaitAllFinishedFusionCheckdStub1;
  tbe_adapter_ptr_->GetOpInfo = get_tbe_opinfo_fusion_check_stub_succ;
  fusion_nodes_num = 0;
  last_fusion_nodes_num = 0;
  string path = GetCodeDir() + "/tests/engines/nn_engine/config/data/platform_config";
  string real_path = fe::RealPath(path);
  fe::PlatformInfoManager::Instance().platform_info_map_.clear();
  fe::PlatformInfoManager::Instance().platform_infos_map_.clear();
  uint32_t init_ret = fe::PlatformInfoManager::Instance().LoadConfigFile(real_path);
  EXPECT_EQ(init_ret, ge::GRAPH_SUCCESS);
  fe::PlatformInfoManager::Instance().opti_compilation_infos_.SetSocVersion("Ascend910B1");

  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_06(graph);
  
  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) MultipleReduceAndDataaccessFusionPass1(); };
  AutoBufferFusionPassRunner *test_pass = new (std::nothrow)
      AutoBufferFusionPassRunner("test_pass", create_fn, cycle_detector, tbe_adapter_ptr_);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);

  const std::vector<std::string> nodes_name_vec = {"relu2", "gather2", "relu1reduceMeanD1gather1reduceMeanD2add1relu3"};
  int index = 0;
  for (auto &node : graph->GetDirectNode()) {
    EXPECT_EQ(node->GetName(), nodes_name_vec.at(index));
    index++;
  }
}

TEST_F(UB_FUSION_UT_AUTO_FUSION_PASS_RUNNER, get_fusion_nodes_map) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_07(graph);

  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) MultipleReduceAndDataaccessFusionPass1(); };
  AutoBufferFusionPassRunner *test_pass = new (std::nothrow)
      AutoBufferFusionPassRunner("test_pass", create_fn, cycle_detector, tbe_adapter_ptr_);
  
  vector<int64_t> dim_weight = {1, 3, 3, 3};
  GeShape shape_weight(dim_weight);
  GeTensorDesc weight_desc(shape_weight);
  OpDescPtr weight_op_desc1 = std::make_shared<OpDesc>("w1", fe::CONSTANT);
  weight_op_desc1->AddOutputDesc(weight_desc);
  NodePtr Node1 = graph->AddNode(weight_op_desc1);
  std::vector<ge::NodePtr> vector_node_ptr;
  vector_node_ptr.emplace_back(Node1);
  std::map<int64_t, std::vector<ge::NodePtr>> fusion_nodes_map;
  fusion_nodes_map.emplace(1, vector_node_ptr);
  BaseBufferFusionPassRunner::GetFusionNodesMap(*graph, fusion_nodes_map);
  EXPECT_EQ(fusion_nodes_map.size(), 2);
}

TEST_F(UB_FUSION_UT_AUTO_FUSION_PASS_RUNNER, update_op_slice_info_01) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
  GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, original_shape);
  test.AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant", "dequant", 1, 1)
      .Attr("dequant", kOpPattern, "dequant");
  ge::NodePtr node;
  test.GetNodeByName("dequant", node);
  std::vector<ge::NodePtr> fusion_nodes;
  fusion_nodes.push_back(node);
  
  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  EXPECT_EQ(cycle_detector->Initialize(*graph), fe::SUCCESS);
  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) TestFusionPass1(); };
  AutoBufferFusionPassRunner *test_pass = new (std::nothrow)
      AutoBufferFusionPassRunner("test_pass", create_fn, cycle_detector, tbe_adapter_ptr_);
  test_pass->CalcSliceInfo(fusion_nodes);
}

TEST_F(UB_FUSION_UT_AUTO_FUSION_PASS_RUNNER, enale_auto_buffer_fusion_01) {
  mmSetEnv("ASCEND_HOME_PATH", "/home/jenkins/Ascend/ascend-toolkit/latest", 0);
  std::map<std::string, std::string> options;
  auto fe_ops_kernel_info_store = make_shared<fe::FEOpsKernelInfoStore>();
  FEOpsStoreInfo heavy_op_info{
      6,
      "tbe-builtin",
      EN_IMPL_HW_TBE,
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/fusion_rule_manager",
      "",
      false,
      false,
      false};

  vector<FEOpsStoreInfo> store_info;
  store_info.emplace_back(heavy_op_info);
  Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
  OpsKernelManager::Instance(AI_CORE_NAME).Finalize();
  string file_path =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_rule_parser/cycle_detection.json";
  fe_ops_kernel_info_store->Initialize(options);
  auto fusion_rule_mgr = std::make_shared<FusionRuleManager>(fe_ops_kernel_info_store);

  auto fusion_priority_mgr =
      std::make_shared<FusionPriorityManager>(AI_CORE_NAME, fusion_rule_mgr);
  Configuration::Instance(fe::AI_CORE_NAME).lib_path_ =
      GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/fusion_config_manager/builtin_config3/";
  fusion_priority_mgr->fusion_config_parser_ptr_ = std::make_unique<FusionConfigParser>(fe::AI_CORE_NAME);
  fusion_priority_mgr->fusion_config_parser_ptr_->ParseFusionConfigFile();
  RegisterBufferFusionFunc(BufferFunc);
  EXPECT_EQ(fe::SUCCESS, fusion_priority_mgr->SortBufferFusion());
  EXPECT_EQ(fe::SUCCESS, fusion_priority_mgr->SortBufferFusion());
  BufferFusionPassRegistry instance;
  BufferFusionPassRegistry::GetInstance().impl_.swap(instance.impl_);
  OpsKernelManager::Instance(AI_CORE_NAME).Finalize();
}

TEST_F(UB_FUSION_UT_AUTO_FUSION_PASS_RUNNER, multiple_reduce_and_dataaccess_fusion_04) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph_05(graph);

  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == "relu1") {
      (void)AttrUtils::SetStr(node->GetOpDesc(), "_user_stream_label", "usr1");
    }
    if (node->GetName() == "reduceMeanD1") {
      (void)AttrUtils::SetStr(node->GetOpDesc(), "_user_stream_label", "usr2");
    }
    if (node->GetName() == "gather1") {
      (void)AttrUtils::SetStr(node->GetOpDesc(), "_user_stream_label", "usr3");
    }
    if (node->GetName() == "relu2") {
      (void)AttrUtils::SetStr(node->GetOpDesc(), "_user_stream_label", "usr4");
    }
  }
  std::shared_ptr<FusionCycleDetector> cycle_detector;
  FE_MAKE_SHARED(cycle_detector = std::make_shared<FusionCycleDetector>(),);
  cycle_detector->Initialize(*graph);
  auto create_fn = []() -> BufferFusionPassBase * { return new (std::nothrow) MultipleReduceAndDataaccessFusionPass1(); };
  AutoBufferFusionPassRunner *test_pass = new (std::nothrow)
      AutoBufferFusionPassRunner("test_pass", create_fn, cycle_detector, tbe_adapter_ptr_);
  test_pass->Run(*graph);
  sub_graph_optimizer_ptr_->BuildFusionGraph(*graph);
}
