#include "gtest/gtest.h"
#include "rts_ops_kernel_info.h"
#include "rts_ffts_plus_ops_kernel_info.h"
#include "common/util.h"

using namespace testing;

namespace cce {
namespace runtime {

class RtsOpsKernelInfoStoreTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RtsOpsKernelInfoStoreTest SetUPTestCase" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RtsOpsKernelInfoStoreTest Tear Down" << std::endl;
  }

  virtual void SetUp() {
    std::cout << "RtsOpsKernelInfoStoreTest SetUP" << std::endl;
  }

  virtual void TearDown() {}
};

TEST_F(RtsOpsKernelInfoStoreTest, Initialize) {
  RtsOpsKernelInfoStore opsKernelInfoStore;
  map<string, string> options;
  ge::Status ret = opsKernelInfoStore.Initialize(options);
  ASSERT_EQ(ret, ge::SUCCESS);
}

TEST_F(RtsOpsKernelInfoStoreTest, TestFftsPlusFinalize) {
  RtsFftsPlusOpsKernelInfoStore opsKernelInfoStore;
  ge::Status ret = opsKernelInfoStore.Finalize();
  ASSERT_EQ(ret, ge::SUCCESS);
}

TEST_F(RtsOpsKernelInfoStoreTest, TestFftsPlusGetAllOpsKernelInfo) {
  map<string, ge::OpInfo> infos;

  RtsFftsPlusOpsKernelInfoStore opsKernelInfoStore;
  opsKernelInfoStore.GetAllOpsKernelInfo(infos);
  ASSERT_EQ(infos.size(), 0);
}

TEST_F(RtsOpsKernelInfoStoreTest, TestFftsPlusCreateSession) {
  map<string, string> session_options;

  RtsFftsPlusOpsKernelInfoStore opsKernelInfoStore;
  ge::Status ret = opsKernelInfoStore.CreateSession(session_options);
  ASSERT_EQ(ret, ge::SUCCESS);
}

TEST_F(RtsOpsKernelInfoStoreTest, TestFftsPlusDestroySession) {
  map<string, string> session_options;

  RtsFftsPlusOpsKernelInfoStore opsKernelInfoStore;
  ge::Status ret = opsKernelInfoStore.DestroySession(session_options);
  ASSERT_EQ(ret, ge::SUCCESS);
}

}  // namespace runtime
}  // namespace cce