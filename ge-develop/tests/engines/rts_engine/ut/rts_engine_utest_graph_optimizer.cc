#include <gtest/gtest.h>

#define protected public
#define private public

#include "graph/compute_graph.h"
#include "graph/ge_context.h"
#include "graph/debug/ge_attr_define.h"
#include "ge/ge_api_types.h"

#include "common/util.h"
#include "graph_optimizer/rts_graph_optimizer.h"
#include "graph_optimizer/rts_ffts_plus_graph_optimizer.h"

#undef protected
#undef private
using namespace testing;
using namespace ge;
using namespace cce::runtime;
using std::string;

class GraphOptimizerTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GraphOptimizerTest SetUPTestCase" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GraphOptimizerTest Tear Down" << std::endl;
  }

  virtual void SetUp() {
    std::cout << "GraphOptimizerTest SetUP" << std::endl;
  }

  virtual void TearDown() {
    // GlobalMockObject::verify();
  }
};

TEST(GraphOptimizerTest, TestInitialize) {
  RtsGraphOptimizer graph_optimizer;
  std::map<string, string> options;
  auto ret = graph_optimizer.Initialize(options, nullptr);
  ASSERT_EQ(ret, SUCCESS);
}

TEST(GraphOptimizerTest, TestFinalize) {
  RtsGraphOptimizer graph_optimizer;
  auto ret = graph_optimizer.Finalize();
  ASSERT_EQ(ret, SUCCESS);
}

TEST(GraphOptimizerTest, TestOptimizeWholeGraph) {
  RtsGraphOptimizer graph_optimizer;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto ret = graph_optimizer.OptimizeWholeGraph(*graph);
  ASSERT_EQ(ret, SUCCESS);
}

TEST(GraphOptimizerTest, TestOptimizeOriginalGraph) {
  RtsGraphOptimizer graph_optimizer;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto ret = graph_optimizer.OptimizeOriginalGraph(*graph);
  ASSERT_EQ(ret, SUCCESS);
}

TEST(GraphOptimizerTest, TestOptimizeFusedGraph) {
  RtsGraphOptimizer graph_optimizer;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto ret = graph_optimizer.OptimizeFusedGraph(*graph);
  ASSERT_EQ(ret, SUCCESS);
}

TEST(GraphOptimizerTest, TestOptimizeGraphBeforeBuild) {
  RtsGraphOptimizer graph_optimizer;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto ret = graph_optimizer.OptimizeGraphBeforeBuild(*graph);
}

TEST(GraphOptimizerTest, TestProcMemtypeRange) {
  RtsGraphOptimizer optimizer;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::Status ret = optimizer.PorcMemtypeRange(*graph);
}

TEST(GraphOptimizerTest, TestOptimizeGraphPrepare_MemcpyAsync) {
  RtsGraphOptimizer optimizer;
  char_t nodeName[20] = "MemcpyAsyncNode";
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::OpDescPtr opDesc = std::make_shared<OpDesc>(nodeName, "MemcpyAsync");
  ge::Status ret = optimizer.OptimizeGraphPrepare(*graph);
  ASSERT_EQ(ret, SUCCESS);
}

TEST(GraphOptimizerTest, TestOptimizeGraphPrepare_Cmo) {
  RtsGraphOptimizer optimizer;
  char_t nodeName[20] = "CmoNode";
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::OpDescPtr opDesc = std::make_shared<OpDesc>(nodeName, "Cmo");
  ge::Status ret = optimizer.OptimizeGraphPrepare(*graph);
  ASSERT_EQ(ret, SUCCESS);
}

TEST(GraphOptimizerTest, TestProcMemtypeRangeSuccess) {
  RtsGraphOptimizer optimizer;
  char_t version[12] = "Ascend310P3";
  char_t nodeName_1[20] = "MemcpyAddrAsync1";
  char_t nodeName_2[20] = "MemcpyAddrAsync2";
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::OpDescPtr opDesc1 = std::make_shared<OpDesc>(nodeName_1, "MemcpyAddrAsync");
  ge::OpDescPtr opDesc2 = std::make_shared<OpDesc>(nodeName_2, "MemcpyAddrAsync");
  graph->AddNode(opDesc1);
  graph->AddNode(opDesc2);
  ge::Status ret = optimizer.PorcMemtypeRange(*graph);
}

TEST(GraphOptimizerTest, TestInsertMemcpyAsyncNodeAndSetMemType) {
  RtsGraphOptimizer optimizer;
  ge::Status ret = SUCCESS;
  char_t nodeName_1[20] = "MemcpyAddrAsync1";
  char_t nodeName_2[20] = "MemcpyAddrAsync2";
  char_t nodeName_3[20] = "MemcpyAddrAsync3";
  char_t nextNodeName[20] = "MemcpyAddrAsync4";
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::OpDescPtr opDesc1 = std::make_shared<OpDesc>(nodeName_1, "MemcpyAddrAsync");
  ge::OpDescPtr opDesc2 = std::make_shared<OpDesc>(nodeName_2, "MemcpyAddrAsync");
  ge::OpDescPtr opDesc3 = std::make_shared<OpDesc>(nodeName_3, "MemcpyAddrAsync");
  graph->AddNode(opDesc1);
  graph->AddNode(opDesc2);
  for (auto nodePtr : graph->GetDirectNode()) {
    ret = optimizer.InsertMemcpyAsyncNodeAndSetMemType(nodePtr, *graph, 1);
    ASSERT_EQ(ret, SUCCESS);
  }
}

TEST(GraphOptimizerTest, TestFftsPlusInitialize) {
  RtsFftsPlusGraphOptimizer graph_optimizer;
  std::map<string, string> options;
  auto ret = graph_optimizer.Initialize(options, nullptr);
  ASSERT_EQ(ret, SUCCESS);
}

TEST(GraphOptimizerTest, TestFftsPlusFinalize) {
  RtsFftsPlusGraphOptimizer graph_optimizer;
  auto ret = graph_optimizer.Finalize();
  ASSERT_EQ(ret, SUCCESS);
}

TEST(GraphOptimizerTest, TestFftsPlusOptimizeGraphPrepare) {
  RtsFftsPlusGraphOptimizer graph_optimizer;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto ret = graph_optimizer.OptimizeGraphPrepare(*graph);
  ASSERT_EQ(ret, SUCCESS);
}

TEST(GraphOptimizerTest, TestFftsPlusOptimizeWholeGraph) {
  RtsFftsPlusGraphOptimizer graph_optimizer;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto ret = graph_optimizer.OptimizeWholeGraph(*graph);
  ASSERT_EQ(ret, SUCCESS);
}

TEST(GraphOptimizerTest, TestFftsPlusGetAttributes) {
  RtsFftsPlusGraphOptimizer optimizer;
  GraphOptimizerAttribute attr;
  ge::Status ret = optimizer.GetAttributes(attr);
  ASSERT_EQ(ret, ge::SUCCESS);
}

TEST(GraphOptimizerTest, TestFftsPlusOptimizeOriginalGraph) {
  RtsFftsPlusGraphOptimizer graph_optimizer;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto ret = graph_optimizer.OptimizeOriginalGraph(*graph);
  ASSERT_EQ(ret, SUCCESS);
}

TEST(GraphOptimizerTest, TestFftsPlusOptimizeFusedGraph) {
  RtsFftsPlusGraphOptimizer graph_optimizer;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto ret = graph_optimizer.OptimizeFusedGraph(*graph);
  ASSERT_EQ(ret, SUCCESS);
}

TEST(GraphOptimizerTest, TestFftsPlusOptimizeGraphBeforeBuild) {
  RtsFftsPlusGraphOptimizer graph_optimizer;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto ret = graph_optimizer.OptimizeGraphBeforeBuild(*graph);
  ASSERT_EQ(ret, SUCCESS);
}