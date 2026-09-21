#include <gtest/gtest.h>
#include "rts_engine.h"
#include "graph/compute_graph.h"
#include "op.h"
#include "rt_log.h"

using namespace testing;
using namespace ge;
using std::string;

namespace cce {
namespace runtime {

class RtsEngineTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RtsEngineTest SetUPTestCase" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RtsEngineTest Tear Down" << std::endl;
  }

  virtual void SetUp() {
    std::cout << "RtsEngineTest SetUP" << std::endl;
  }

  virtual void TearDown() {}
};

TEST_F(RtsEngineTest, TestInitialize) {
  std::map<string, string> options;
  ASSERT_EQ(Initialize(options), SUCCESS);
}

TEST_F(RtsEngineTest, TestGetOpsKernelInfoStores) {
  std::map<std::string, OpsKernelInfoStorePtr> opsKernelMap;
  GetOpsKernelInfoStores(opsKernelMap);
  ASSERT_NE(opsKernelMap.size(), 0);
}

TEST_F(RtsEngineTest, TestGetGraphOptimizerObjs) {
  std::map<std::string, GraphOptimizerPtr> graphOptimizers;
  GetGraphOptimizerObjs(graphOptimizers);
  ASSERT_NE(graphOptimizers.size(), 0);
}

TEST_F(RtsEngineTest, TestFinalize) {
  ASSERT_EQ(Finalize(), 0U);
}

}  // namespace runtime
}  // namespace cce