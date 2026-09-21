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
#include <memory>
#include <iostream>
#include "graph/operator_factory_impl.h"
#include "macro_utils/dt_public_scope.h"
#include "graph/optimize/graph_optimize.h"
#include "api/gelib/gelib.h"
#include "ge/ge_api.h"
#include "macro_utils/dt_public_unscope.h"
#include "graph/preprocess/graph_prepare.h"
#include "graph/passes/graph_builder_utils.h"
#include "host_kernels/kernel.h"
#include "host_kernels/kernel_factory.h"

using namespace std;
using namespace testing;

namespace ge {
const char *AddNYes = "AddNYes";
namespace {
const char *const kVectorCore = "VectorCore";
const char *const kAicoreEngine = "AIcoreEngine";
/*void CreateEngineConfigJson(string &dir_path, string &file_path) {
  GELOGI("Begin to create engine config json file.");
  string base_path = GetModelPath();
  GELOGI("Base path is %s.", base_path.c_str());
  dir_path = base_path.substr(0, base_path.rfind('/') + 1) + "plugin/nnengine/ge_config";
  string cmd = "mkdir -p " + dir_path;
  system(cmd.c_str());
  file_path = dir_path + "/engine_conf.json";
  GELOGI("Begin to write into the config file: %s.", file_path.c_str());
  ofstream ofs(file_path, ios::out);
  EXPECT_EQ(!ofs, false);
  ofs << "{\n"
         "  \"schedule_units\" : [ {\n"
         "    \"id\" : \"TS_1\",\n"
         "    \"name\" : \"1980_hwts\",\n"
         "    \"ex_attrs\" : \"\",\n"
         "    \"cal_engines\" : [\n"
         "      {\"id\" : \"DNN_VM_GE_LOCAL\", \"name\" : \"GE_LOCAL\", \"independent\" : false, \"attch\" : true, \"skip_assign_stream\" : true },\n"
         "      {\"id\" : \"AIcoreEngine\", \"name\" : \"AICORE\", \"independent\" : false, \"attch\" : false, \"skip_assign_stream\" : false}\n"
         "    ]\n"
         "  } ]\n"
         "}";
  ofs.close();
  GELOGI("Json config file %s has been written.", file_path.c_str());
}*/

void DeleteFile(const string &file_name) {
  auto ret = remove(file_name.c_str());
  if (ret == 0) {
    GELOGI("Delete file successfully, file:%s.", file_name.c_str());
  }
}
}
class UtestGraphOptimizeTest : public testing::Test {
 protected:
  void SetUp() {
    // CreateEngineConfigJson(config_dir_, config_file_);
  }

  void TearDown() {
    DeleteFile(config_file_);
    DeleteFile(config_dir_);
    DNNEngineManager::GetInstance().schedulers_.clear();
    DNNEngineManager::GetInstance().engines_map_.clear();
    OpsKernelManager::GetInstance().atomic_first_optimizers_by_priority_.clear();
  }
 public:
  NodePtr UtAddNode(ComputeGraphPtr &graph, std::string name, std::string type, int in_cnt, int out_cnt){
    auto tensor_desc = std::make_shared<GeTensorDesc>();
    std::vector<int64_t> shape = {1, 1, 224, 224};
    tensor_desc->SetShape(GeShape(shape));
    tensor_desc->SetFormat(FORMAT_NCHW);
    tensor_desc->SetDataType(DT_FLOAT);
    tensor_desc->SetOriginFormat(FORMAT_NCHW);
    tensor_desc->SetOriginShape(GeShape(shape));
    tensor_desc->SetOriginDataType(DT_FLOAT);
    auto op_desc = std::make_shared<OpDesc>(name, type);
    for (int i = 0; i < in_cnt; ++i) {
        op_desc->AddInputDesc(tensor_desc->Clone());
    }
    for (int i = 0; i < out_cnt; ++i) {
        op_desc->AddOutputDesc(tensor_desc->Clone());
    }
    return graph->AddNode(op_desc);
  }

 private:
  string config_dir_;
  string config_file_;
};

class TestGraphOptimizerSuccess : public GraphOptimizer {
 public:
  ~TestGraphOptimizerSuccess() override {
    Finalize();
  }
  Status Initialize(const std::map<std::string, std::string> &options,
                    OptimizeUtility *const optimize_utility) override {
    return SUCCESS;
  }
  Status Finalize() override {
    return SUCCESS;
  }

  Status OptimizeGraphInit(ComputeGraph& graph) override {
    ++optimize_graph_init_cnt_;
    return SUCCESS;
  }
  Status OptimizeGraphPrepare(ComputeGraph &graph) override {
    ++optimize_origin_graph_cnt_;
    return SUCCESS;
  }
  Status OptimizeGraphBeforeBuild(ComputeGraph &graph) override {
    return SUCCESS;
  }
  Status OptimizeOriginalGraph(ComputeGraph &graph) override {
    return SUCCESS;
  }
  Status OptimizeOriginalGraphJudgeInsert(ComputeGraph &graph) override {
    return SUCCESS;
  }
  Status OptimizeFusedGraph(ComputeGraph &graph) override {
    return SUCCESS;
  }
  Status OptimizeWholeGraph(ComputeGraph &graph) override {
    return SUCCESS;
  }
  Status GetAttributes(GraphOptimizerAttribute &attrs) const override {
    attrs.engineName = "AIcoreEngine";
    attrs.scope = OPTIMIZER_SCOPE::ENGINE;
    return SUCCESS;
  }
  Status OptimizeStreamGraph(ComputeGraph &graph, const RunContext &context) override {
    return SUCCESS;
  }
  Status OptimizeFusedGraphAfterGraphSlice(ComputeGraph &graph) override {
    return SUCCESS;
  }
  Status OptimizeAfterStage1(ComputeGraph &graph) override {
    return SUCCESS;
  }
  uint64_t optimize_graph_init_cnt_ = 0U;
  uint64_t optimize_origin_graph_cnt_ = 0U;
};

class TestGraphOptimizerFail : public GraphOptimizer {
 public:
  ~TestGraphOptimizerFail() override {
    Finalize();
  }
  Status Initialize(const std::map<std::string, std::string> &options,
                    OptimizeUtility *const optimize_utility) override {
    return SUCCESS;
  }
  Status Finalize() override {
    return SUCCESS;
  }
  Status OptimizeGraphPrepare(ComputeGraph &graph) override {
    return FAILED;
  }
  Status OptimizeGraphBeforeBuild(ComputeGraph &graph) override {
    return FAILED;
  }
  Status OptimizeOriginalGraph(ComputeGraph &graph) override {
    return FAILED;
  }
  Status OptimizeOriginalGraphJudgeInsert(ComputeGraph &graph) override {
    return FAILED;
  }
  Status OptimizeFusedGraph(ComputeGraph &graph) override {
    return FAILED;
  }
  Status OptimizeWholeGraph(ComputeGraph &graph) override {
    return FAILED;
  }
  Status GetAttributes(GraphOptimizerAttribute &attrs) const override {
    attrs.engineName = "AIcoreEngine";
    attrs.scope = OPTIMIZER_SCOPE::ENGINE;
    return SUCCESS;
  }
  Status OptimizeStreamGraph(ComputeGraph &graph, const RunContext &context) override {
    return FAILED;
  }
  Status OptimizeFusedGraphAfterGraphSlice(ComputeGraph &graph) override {
    return FAILED;
  }
  Status OptimizeAfterStage1(ComputeGraph &graph) override {
    return FAILED;
  }
};

TEST_F(UtestGraphOptimizeTest, test_OptimizeGraphInit_succ) {
  map<string, string> options;
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  auto graph_opt = MakeShared<TestGraphOptimizerSuccess>();
  OpsKernelManager::GetInstance().atomic_first_optimizers_by_priority_.push_back(make_pair("VectorEngine", graph_opt));

  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GraphOptimize base_optimize;
  ret = base_optimize.OptimizeGraphInit(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph_opt->optimize_graph_init_cnt_, 1);

  ret = base_optimize.OptimizeOriginalGraph(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph_opt->optimize_origin_graph_cnt_, 0);

  shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  EXPECT_NE(instance_ptr, nullptr);
  ret = instance_ptr->Finalize();
  EXPECT_EQ(ret, SUCCESS);
  OpsKernelManager::GetInstance().atomic_first_optimizers_by_priority_.clear();
}

TEST_F(UtestGraphOptimizeTest, test_OptimizeAfterStage1_succ) {
  map<string, string> options;
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  GraphOptimizerPtr graph_opt = MakeShared<TestGraphOptimizerSuccess>();
  OpsKernelManager::GetInstance().atomic_first_optimizers_by_priority_.push_back(make_pair("AIcoreEngine", graph_opt));

  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GraphOptimize base_optimize;
  ret = base_optimize.OptimizeAfterStage1(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  base_optimize.core_type_ = kVectorCore;
  ret = base_optimize.OptimizeAfterStage1(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  EXPECT_NE(instance_ptr, nullptr);
  ret = instance_ptr->Finalize();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphOptimizeTest, test_OptimizeAfterStage1_fail) {
  ComputeGraphPtr compute_graph = nullptr;
  GraphOptimize base_optimize;

  // 1. Input graph is nullptr.
  Status ret = base_optimize.OptimizeAfterStage1(compute_graph);
  EXPECT_EQ(ret, PARAM_INVALID);

  // 2. GELib is not initialized.
  compute_graph = MakeShared<ComputeGraph>("test_graph");
  ret = base_optimize.OptimizeAfterStage1(compute_graph);
  EXPECT_EQ(ret, GE_CLI_GE_NOT_INITIALIZED);

  // 3. The optimizer registered with the engine returned a failure.
  map<string, string> options;
  ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  GraphOptimizerPtr graph_opt = MakeShared<TestGraphOptimizerFail>();
  OpsKernelManager::GetInstance().atomic_first_optimizers_by_priority_.push_back(make_pair("AIcoreEngine", graph_opt));
  ret = base_optimize.OptimizeAfterStage1(compute_graph);
  EXPECT_EQ(ret, FAILED);

  shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  EXPECT_NE(instance_ptr, nullptr);
  ret = instance_ptr->Finalize();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphOptimizeTest, test_optimizers_succ) {
  map<string, string> options;
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  GraphOptimizerPtr graph_opt = MakeShared<TestGraphOptimizerSuccess>();
  OpsKernelManager::GetInstance().atomic_first_optimizers_by_priority_.push_back(make_pair("AIcoreEngine", graph_opt));

  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GraphOptimize base_optimize;

  ret = base_optimize.OptimizeOriginalGraph(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  ret = base_optimize.OptimizeOriginalGraphJudgePrecisionInsert(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  ret = base_optimize.OptimizeOriginalGraphJudgeFormatInsert(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  ret = base_optimize.OptimizeOriginalGraphForQuantize(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  ret = base_optimize.OptimizeGraphBeforeBuild(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  ret = base_optimize.OptimizeWholeGraph(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  EXPECT_NE(instance_ptr, nullptr);
  ret = instance_ptr->Finalize();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphOptimizeTest, test_optimizers_fail) {
  map<string, string> options;
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  GraphOptimizerPtr graph_opt = MakeShared<TestGraphOptimizerFail>();
  OpsKernelManager::GetInstance().atomic_first_optimizers_by_priority_.push_back(make_pair("AIcoreEngine", graph_opt));

  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GraphOptimize base_optimize;

  ret = base_optimize.OptimizeOriginalGraph(compute_graph);
  EXPECT_EQ(ret, FAILED);

  ret = base_optimize.OptimizeOriginalGraphJudgePrecisionInsert(compute_graph);
  EXPECT_EQ(ret, FAILED);

  ret = base_optimize.OptimizeOriginalGraphForQuantize(compute_graph);
  EXPECT_EQ(ret, FAILED);

  ret = base_optimize.OptimizeGraphBeforeBuild(compute_graph);
  EXPECT_EQ(ret, FAILED);

  ret = base_optimize.OptimizeWholeGraph(compute_graph);
  EXPECT_EQ(ret, FAILED);

  shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  EXPECT_NE(instance_ptr, nullptr);
  ret = instance_ptr->Finalize();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphOptimizeTest, test_optimizers_composite) {
  OpsKernelManager::GetInstance().composite_engines_["composite_test"] = std::set<string>{ "composite_test" };
  OpsKernelManager::GetInstance().composite_engine_kernel_lib_names_["composite_test"] = "composite_test";

  map<string, string> options;
  EXPECT_EQ(GELib::Initialize(options), SUCCESS);

  GraphOptimizerPtr graph_opt = MakeShared<TestGraphOptimizerSuccess>();
  OpsKernelManager::GetInstance().atomic_first_optimizers_by_priority_.push_back(make_pair("AIcoreEngine", graph_opt));

  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  auto op_desc = std::make_shared<OpDesc>(NODE_NAME_NET_OUTPUT, NETOUTPUT);
  compute_graph->AddNode(op_desc);
  op_desc->SetOpEngineName("DNN_VM_GE_LOCAL");
  op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");

  GraphOptimize base_optimize;
  EXPECT_EQ(base_optimize.OptimizeOriginalGraph(compute_graph), SUCCESS);

  EXPECT_EQ(base_optimize.OptimizeOriginalGraphJudgePrecisionInsert(compute_graph), SUCCESS);

  EXPECT_EQ(base_optimize.OptimizeOriginalGraphForQuantize(compute_graph), SUCCESS);

  EXPECT_EQ(base_optimize.OptimizeGraphBeforeBuild(compute_graph), SUCCESS);

  EXPECT_EQ(base_optimize.OptimizeWholeGraph(compute_graph), SUCCESS);

  EXPECT_EQ(GELib::GetInstance()->Finalize(), SUCCESS);

  OpsKernelManager::GetInstance().composite_engines_.erase("composite_test");
  OpsKernelManager::GetInstance().composite_engine_kernel_lib_names_.erase("composite_test");
}

TEST_F(UtestGraphOptimizeTest, test_optimizers_exclude) {
  map<string, string> options;
  EXPECT_EQ(GELib::Initialize(options), SUCCESS);

  GraphOptimizerPtr graph_opt = MakeShared<TestGraphOptimizerSuccess>();
  OpsKernelManager::GetInstance().atomic_first_optimizers_by_priority_.push_back(make_pair("DNN_VM_DVPP", graph_opt));

  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  auto op_desc = std::make_shared<OpDesc>(NODE_NAME_NET_OUTPUT, NETOUTPUT);
  compute_graph->AddNode(op_desc);
  op_desc->SetOpEngineName("DNN_VM_GE_LOCAL");
  op_desc->SetOpKernelLibName("DNN_VM_GE_LOCAL_OP_STORE");

  GraphOptimize base_optimize;
  base_optimize.exclude_engines_.insert("DNN_VM_DVPP");
  EXPECT_EQ(base_optimize.OptimizeOriginalGraph(compute_graph), SUCCESS);

  EXPECT_EQ(base_optimize.OptimizeOriginalGraphJudgePrecisionInsert(compute_graph), SUCCESS);

  EXPECT_EQ(base_optimize.OptimizeOriginalGraphForQuantize(compute_graph), SUCCESS);

  EXPECT_EQ(base_optimize.OptimizeGraphBeforeBuild(compute_graph), SUCCESS);

  EXPECT_EQ(base_optimize.OptimizeWholeGraph(compute_graph), SUCCESS);

  EXPECT_EQ(GELib::GetInstance()->Finalize(), SUCCESS);
}

class TestGraphOptimizerWithInfer : public GraphOptimizer {
 public:
  ~TestGraphOptimizerWithInfer() override {
    Finalize();
  }
  Status Initialize(const std::map<std::string, std::string> &options,
                    OptimizeUtility *const optimize_utility) override {
    optimize_utility_ = optimize_utility;
    return SUCCESS;
  }
  Status Finalize() override {
    return SUCCESS;
  }
  Status OptimizeGraphPrepare(ComputeGraph &graph) override {
    auto compute_graph = std::make_shared<ComputeGraph>(graph);
    auto ret = optimize_utility_->InferShape(compute_graph);
    return ret;
  }
  Status OptimizeGraphBeforeBuild(ComputeGraph &graph) override {
    return FAILED;
  }
  Status OptimizeOriginalGraph(ComputeGraph &graph) override {
    return FAILED;
  }
  Status OptimizeOriginalGraphJudgeInsert(ComputeGraph &graph) override {
    return FAILED;
  }
  Status OptimizeFusedGraph(ComputeGraph &graph) override {
    return FAILED;
  }
  Status OptimizeWholeGraph(ComputeGraph &graph) override {
    return FAILED;
  }
  Status GetAttributes(GraphOptimizerAttribute &attrs) const override {
    attrs.engineName = "AIcoreEngine";
    attrs.scope = OPTIMIZER_SCOPE::ENGINE;
    return SUCCESS;
  }
  Status OptimizeStreamGraph(ComputeGraph &graph, const RunContext &context) override {
    return FAILED;
  }
  Status OptimizeFusedGraphAfterGraphSlice(ComputeGraph &graph) override {
    return FAILED;
  }
  Status OptimizeAfterStage1(ComputeGraph &graph) override {
    return FAILED;
  }
  private:
   OptimizeUtility *optimize_utility_;
};

namespace {

///    netoutput1
///       |
///       |
///     addnYes1
///    /    \.
///  /       \.
/// const1   const2
ComputeGraphPtr BuildGraphWithConstantFolding() {
  auto builder = ut::GraphBuilder("test");
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto const2 = builder.AddNode("const2", CONSTANT, 0, 1);
  auto addn1 = builder.AddNode("addn1", AddNYes, 2, 1);
  auto netoutput1 = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  builder.AddDataEdge(const1, 0, addn1, 0);
  builder.AddDataEdge(const2, 0, addn1, 1);
  builder.AddDataEdge(addn1, 0, netoutput1, 0);
  return builder.GetGraph();
}
} // namespace

class TestAddNKernel : public Kernel {
 public:
  Status Compute(const ge::OpDescPtr op_desc_ptr, const std::vector<ge::ConstGeTensorPtr> &input,
                 std::vector<ge::GeTensorPtr> &v_output) override {
    auto output = std::make_shared<GeTensor>();
    std::vector<uint8_t> data{1, 2, 3};
    std::vector<int64_t> shape{3};
    output->MutableTensorDesc().SetShape(GeShape(shape));
    output->SetData(data);
    output->MutableTensorDesc().SetDataType(DT_UINT8);
    v_output.push_back(output);
    return SUCCESS;
  }
};
REGISTER_COMPUTE_NODE_KERNEL(AddNYes, TestAddNKernel);

TEST_F(UtestGraphOptimizeTest, test_optimizers_initialize_with_infer_func) {
  OperatorFactoryImpl::RegisterInferShapeFunc("Const", [](Operator &op) {return GRAPH_SUCCESS;});
  ComputeGraphPtr compute_graph = BuildGraphWithConstantFolding();
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 4);
  GraphOptimize base_optimize;

  map<string, string> options;
  auto ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  GraphOptimizerPtr graph_opt = MakeShared<TestGraphOptimizerWithInfer>();
  OpsKernelManager::GetInstance().atomic_first_optimizers_by_priority_.push_back(make_pair("AIcoreEngine", graph_opt));
  GraphOptimizeUtility optimzie_utility;
  // give optimize utility to test_optimizer
  graph_opt->Initialize(options, &optimzie_utility);

  // test optimize_prepare inferface by ge_optimize
  ret = base_optimize.OptimizeOriginalGraphForQuantize(compute_graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 2);
  auto netoutput = compute_graph->FindNode("netoutput");
  // check constant folding taken to effect
  EXPECT_EQ(netoutput->GetInDataNodes().at(0)->GetType(), "Const");

  shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  EXPECT_NE(instance_ptr, nullptr);
  ret = instance_ptr->Finalize();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphOptimizeTest, GetSummaryOutputIndexes) {
  GraphOptimize base_optimize;
  EXPECT_EQ(base_optimize.GetSummaryOutputIndexes().size(), 0);
}

TEST_F(UtestGraphOptimizeTest, Nulls) {
  GraphOptimize base_optimize;
  ComputeGraphPtr graph = nullptr;
  EXPECT_EQ(base_optimize.OptimizeSubGraph(graph, ""), GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL);
  EXPECT_NE(base_optimize.OptimizeOriginalGraph(graph), SUCCESS);
  EXPECT_NE(base_optimize.OptimizeOriginalGraphForQuantize(graph), SUCCESS);
  EXPECT_EQ(base_optimize.OptimizeGraphBeforeBuild(graph), GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL);
  base_optimize.TranFrameOp(graph);
  EXPECT_NE(base_optimize.OptimizeWholeGraph(graph), SUCCESS);
}

TEST_F(UtestGraphOptimizeTest, IdentifyReference) {
  GraphOptimize base_optimize;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto ref_node = UtAddNode(graph, "ref", "REF", 1, 1);
  ref_node->GetOpDesc()->UpdateInputName({{"ref", 0}});
  ref_node->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  EXPECT_EQ(base_optimize.IdentifyReference(graph), SUCCESS);
}

TEST_F(UtestGraphOptimizeTest, OptimizeSubGraphNoReg) {
  GraphOptimize base_optimize;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = UtAddNode(graph, "data1", "DATA", 1, 1);
  EXPECT_EQ(base_optimize.OptimizeSubGraph(graph, "engine"), SUCCESS);
}

TEST_F(UtestGraphOptimizeTest, OptimizeSubGraphOneNode) {
  std::string engine_name = "engine";
  DNNEngineManager::GetInstance().engines_map_[engine_name] = std::make_shared<DNNEngine>();
  GraphOptimize base_optimize;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = UtAddNode(graph, "data1", "DATA", 1, 1);
  auto node2 = UtAddNode(graph, "data2", "DATA", 1, 1);
  node->GetInDataAnchor(0)->LinkFrom(node2->GetOutDataAnchor(0));
  EXPECT_EQ(base_optimize.OptimizeSubGraph(graph, engine_name), SUCCESS);
}

TEST_F(UtestGraphOptimizeTest, OptimizeSubGraphNoNode) {
  std::string engine_name = "engine";
  DNNEngineManager::GetInstance().engines_map_[engine_name] = std::make_shared<DNNEngine>();
  GraphOptimize base_optimize;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  EXPECT_EQ(base_optimize.OptimizeSubGraph(graph, engine_name), SUCCESS);
}

TEST_F(UtestGraphOptimizeTest, OptimizeSubGraphTun) {
  std::string engine_name = "engine";
  DNNEngineManager::GetInstance().engines_map_[engine_name] = std::make_shared<DNNEngine>();
  GraphOptimize base_optimize;
  base_optimize.build_mode_ = "tuning";
  base_optimize.build_step_ = "after_merge";
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = UtAddNode(graph, "data1", "DATA", 1, 1);
  EXPECT_EQ(base_optimize.OptimizeSubGraph(graph, engine_name), SUCCESS);
}

TEST_F(UtestGraphOptimizeTest, TranFrameOp) {
  GraphOptimize base_optimize;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = UtAddNode(graph, "data1", "DATA", 1, 1);
  EXPECT_NO_THROW(base_optimize.TranFrameOp(graph));
}

TEST_F(UtestGraphOptimizeTest, SetOptionsFailed) {
  GraphOptimize base_optimize;
  GraphManagerOptions opt;
  opt.framework_type = 10;
  EXPECT_EQ(base_optimize.SetOptions(opt), GE_GRAPH_OPTIONS_INVALID);
}


} // namespace ge
