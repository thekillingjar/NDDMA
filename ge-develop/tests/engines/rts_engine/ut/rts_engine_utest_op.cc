#include <gtest/gtest.h>

#define protected public
#define private public
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/anchor_utils.h"
#include "graph/utils/tensor_utils.h"
#include "stream_switch_op.h"
#include "label_set_op.h"
#include "label_switch_op.h"
#include "label_goto_op.h"
#include "label_goto_ex_op.h"
#include "label_switch_by_index_op.h"
#include "npu_get_float_status_op.hpp"
#include "npu_get_float_status_op.hpp"
#include "npu_get_float_debug_status_op.h"
#include "npu_clear_float_debug_status_op.h"
#include "rts_graph_optimizer.h"
#include "rts_ops_kernel_builder.h"
#include "rts_ffts_plus_ops_kernel_builder.h"
#include "memcpy_async_op.h"
#include "cmo_addr_op.h"
#include "send_op.h"
#include "recv_op.h"
#include "recv_notify_op.h"
#include "send_notify_op.h"
#include "memcpy_addr_async_op.h"
#include "model_exit_op.h"
#include "stream_active_op.h"
#include "stream_merge_op.h"
#include "stream_switchN_op.h"
#include "op.h"
#include "end_graph_op.h"
#include "common/error_code/error_code.h"
#include "op_factory.h"
#include "runtime/rt_model.h"

#undef protected
#undef private
using namespace testing;
using namespace ge;
using namespace cce::runtime;
using std::string;

class RtsEngineOpTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RtsEngineOpTest SetUPTestCase" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RtsEngineOpTest Tear Down" << std::endl;
  }

  virtual void SetUp() {
    std::cout << "RtsEngineOpTest SetUP" << std::endl;
  }

  virtual void TearDown() {}
};

TEST(RtsEngineOpTest, GenerateCtxDef) {
  RunContext runContext;
  vector<TaskDef> tasks;
  int64_t input_data[2] = {0};
  int64_t output_data[2] = {0};
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(op_desc, graph);
  if (nodePtr == NULL) {
    std::cout << "GenerateCtxDef nodePtr is NULL" << std::endl;
  } else {
    // stream_switch_op
    std::shared_ptr<StreamSwitchOp> streamSwitchOpPtr;
    streamSwitchOpPtr = std::make_shared<StreamSwitchOp>(*nodePtr, runContext);
    if (streamSwitchOpPtr != NULL) {
      vector<uint32_t> active_stream_list;
      active_stream_list.push_back(0);
      (void)AttrUtils::SetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list);
      streamSwitchOpPtr->Init();
      streamSwitchOpPtr->data_type_ = ACL_RT_SWITCH_INT32;
      streamSwitchOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[0]));
      streamSwitchOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[1]));
      streamSwitchOpPtr->GenerateCtxDef(*nodePtr);
      ge::Status ret = streamSwitchOpPtr->Run(tasks);
      ASSERT_EQ(ret, ge::SUCCESS);
      ret = streamSwitchOpPtr->UpdateTaskDef(tasks);
      ASSERT_EQ(ret, ge::SUCCESS);
      std::cout << "StreamSwitchOp GenerateCtxDef success" << std::endl;
    } else {
      std::cout << "streamSwitchOpPtr is NULL" << std::endl;
    }

    // label_set_op
    std::shared_ptr<LabelSetOp> labelSetOpPtr;
    labelSetOpPtr = std::make_shared<LabelSetOp>(*nodePtr, runContext);
    if (labelSetOpPtr != NULL) {
      (void)ge::AttrUtils::SetInt(nodePtr->GetOpDesc(), ATTR_NAME_LABEL_SWITCH_INDEX, 1);
      labelSetOpPtr->Init();
      ge::Status ret = labelSetOpPtr->Run(tasks);
      ASSERT_EQ(ret, ge::SUCCESS);
      labelSetOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[0]));
      labelSetOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[1]));
      labelSetOpPtr->GenerateCtxDef(*nodePtr);
      std::cout << "LabelSetOp GenerateCtxDef success" << std::endl;
    } else {
      std::cout << "labelSetOpPtr is NULL" << std::endl;
    }

    // label_goto_op
    std::shared_ptr<LabelGotoOp> LabelGotoOpPtr;
    LabelGotoOpPtr = std::make_shared<LabelGotoOp>(*nodePtr, runContext);
    if (LabelGotoOpPtr != NULL) {
      ge::Status ret = LabelGotoOpPtr->Run(tasks);
      ASSERT_EQ(ret, ge::FAILED);
      std::cout << "LabelGotoOpPtr GenerateCtxDef success" << std::endl;
    }

    // ModelExitOp
    std::shared_ptr<ModelExitOp> ModelExitOpPtr;
    ModelExitOpPtr = std::make_shared<ModelExitOp>(*nodePtr, runContext);
    if (ModelExitOpPtr != NULL) {
      ge::Status ret = ModelExitOpPtr->Run(tasks);
      ASSERT_EQ(ret, ge::SUCCESS);
      std::cout << "ModelExitOpPtr success" << std::endl;
    }

    // label_switch_op
    std::shared_ptr<LabelSwitchOp> LabelSwitchOpPtr;
    LabelSwitchOpPtr = std::make_shared<LabelSwitchOp>(*nodePtr, runContext);
    if (LabelSwitchOpPtr != NULL) {
      LabelSwitchOpPtr->Init();
      LabelSwitchOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[0]));
      LabelSwitchOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[1]));
      LabelSwitchOpPtr->GenerateCtxDef(*nodePtr);
      ge::Status ret = LabelSwitchOpPtr->Run(tasks);
      ASSERT_EQ(ret, ge::SUCCESS);
      std::cout << "LabelSwitchOpPtr GenerateCtxDef success" << std::endl;
    } else {
      std::cout << "labelSwitchByIndexOpPtr is NULL" << std::endl;
    }

    // label_switch_by_index_op
    std::shared_ptr<LabelSwitchByIndexOp> labelSwitchByIndexOpPtr;
    labelSwitchByIndexOpPtr = std::make_shared<LabelSwitchByIndexOp>(*nodePtr, runContext);
    if (labelSwitchByIndexOpPtr != NULL) {
      labelSwitchByIndexOpPtr->Init();
      labelSwitchByIndexOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[0]));
      labelSwitchByIndexOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[1]));
      labelSwitchByIndexOpPtr->GenerateCtxDef(*nodePtr);
      labelSwitchByIndexOpPtr->Run(tasks);
      std::cout << "LabelSwitchByIndexOp GenerateCtxDef success" << std::endl;
    } else {
      std::cout << "labelSwitchByIndexOpPtr is NULL" << std::endl;
    }

    // label_goto_ex_op
    std::shared_ptr<LabelGotoExOp> LabelGotoExOpPtr;
    LabelGotoExOpPtr = std::make_shared<LabelGotoExOp>(*nodePtr, runContext);
    if (LabelGotoExOpPtr != NULL) {
      LabelGotoExOpPtr->Init();
      LabelGotoExOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[0]));
      LabelGotoExOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[1]));
      LabelGotoExOpPtr->GenerateCtxDef(*nodePtr);
      LabelGotoExOpPtr->Run(tasks);
      std::cout << "LabelGotoExOp GenerateCtxDef success" << std::endl;
    }

    std::shared_ptr<MemcpyAsyncOp> memcpyAsyncOpPtr;
    memcpyAsyncOpPtr = std::make_shared<MemcpyAsyncOp>(*nodePtr, runContext);
    if (memcpyAsyncOpPtr != NULL) {
      memcpyAsyncOpPtr->Init();
      memcpyAsyncOpPtr->v_input_size_.push_back(32);
      memcpyAsyncOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[0]));
      memcpyAsyncOpPtr->v_output_size_.push_back(32);
      memcpyAsyncOpPtr->v_output_data_addr_.push_back((void *)(uintptr_t)(&output_data[0]));
      memcpyAsyncOpPtr->GenerateCtxDef(*nodePtr);
      std::cout << "MemcpyAsyncOp GenerateCtxDef success" << std::endl;
    } else {
      std::cout << "memcpyAsyncOpPtr is NULL" << std::endl;
    }

    // cmo_addr_op
    std::shared_ptr<CmoAddrOp> cmoAddrOpPtr;
    cmoAddrOpPtr = std::make_shared<CmoAddrOp>(*nodePtr, runContext);
    if (cmoAddrOpPtr != NULL) {
      cmoAddrOpPtr->Init();
      cmoAddrOpPtr->v_input_size_.push_back(32);
      cmoAddrOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[0]));
      cmoAddrOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[1]));
      cmoAddrOpPtr->GenerateCtxDef(*nodePtr);
      std::cout << "cmoAddrOpPtr GenerateCtxDef success" << std::endl;
    } else {
      std::cout << "cmoAddrOpPtr is NULL" << std::endl;
    }
  }
}

TEST(RtsEngineOpTest, inputsize_error1) {
  RunContext runContext;
  vector<TaskDef> tasks;
  int64_t input_data[1] = {0};
  int64_t output_data[2] = {0};
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(op_desc, graph);
  if (nodePtr == NULL) {
    std::cout << "GenerateCtxDef nodePtr is NULL" << std::endl;
  } else {
    std::shared_ptr<MemcpyAsyncOp> memcpyAsyncOpPtr;
    memcpyAsyncOpPtr = std::make_shared<MemcpyAsyncOp>(*nodePtr, runContext);
    if (memcpyAsyncOpPtr != NULL) {
      memcpyAsyncOpPtr->Init();
      memcpyAsyncOpPtr->v_input_size_.push_back(32);
      memcpyAsyncOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[0]));
      memcpyAsyncOpPtr->v_output_size_.push_back(32);
      memcpyAsyncOpPtr->v_output_data_addr_.push_back((void *)(uintptr_t)(&output_data[0]));
      memcpyAsyncOpPtr->v_output_data_addr_.push_back((void *)(uintptr_t)(&output_data[1]));
      memcpyAsyncOpPtr->GenerateCtxDef(*nodePtr);
      ge::Status ret = memcpyAsyncOpPtr->Run(tasks);
      ASSERT_EQ(ret, ge::FAILED);
      ret = memcpyAsyncOpPtr->UpdateTaskDef(tasks);
      ASSERT_EQ(ret, ge::SUCCESS);
      std::cout << "MemcpyAsyncOp GenerateCtxDef success" << std::endl;
    } else {
      std::cout << "memcpyAsyncOpPtr is NULL" << std::endl;
    }
  }
}

TEST(RtsEngineOpTest, inputsize_error2) {
  RunContext runContext;
  int64_t input_data[1] = {0};
  int64_t output_data[1] = {0};
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(op_desc, graph);
  if (nodePtr == NULL) {
    std::cout << "GenerateCtxDef nodePtr is NULL" << std::endl;
  } else {
    std::shared_ptr<MemcpyAsyncOp> memcpyAsyncOpPtr;
    memcpyAsyncOpPtr = std::make_shared<MemcpyAsyncOp>(*nodePtr, runContext);
    if (memcpyAsyncOpPtr != NULL) {
      memcpyAsyncOpPtr->Init();
      memcpyAsyncOpPtr->v_input_size_.push_back(32);
      memcpyAsyncOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[0]));
      memcpyAsyncOpPtr->v_output_size_.push_back(-1);
      memcpyAsyncOpPtr->v_output_data_addr_.push_back((void *)(uintptr_t)(&output_data[0]));
      ge::Status ret = memcpyAsyncOpPtr->GenerateCtxDef(*nodePtr);
      ASSERT_EQ(ret, ge::SUCCESS);
      std::cout << "MemcpyAsyncOp GenerateCtxDef success" << std::endl;
    } else {
      std::cout << "memcpyAsyncOpPtr is NULL" << std::endl;
    }
  }
}

TEST(RtsEngineOpTest, inputsize_error3) {
  RunContext runContext;
  vector<TaskDef> tasks;
  int64_t input_data[2] = {0};
  int64_t output_data[2] = {0};
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(op_desc, graph);
  if (nodePtr == NULL) {
    std::cout << "GenerateCtxDef nodePtr is NULL" << std::endl;
  } else {
    std::shared_ptr<MemcpyAsyncOp> memcpyAsyncOpPtr;
    memcpyAsyncOpPtr = std::make_shared<MemcpyAsyncOp>(*nodePtr, runContext);
    if (memcpyAsyncOpPtr != NULL) {
      memcpyAsyncOpPtr->Init();
      memcpyAsyncOpPtr->v_input_size_.push_back(0);
      memcpyAsyncOpPtr->v_input_size_.push_back(32);
      memcpyAsyncOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[0]));
      memcpyAsyncOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[1]));
      memcpyAsyncOpPtr->v_output_size_.push_back(0);
      memcpyAsyncOpPtr->v_output_size_.push_back(32);
      memcpyAsyncOpPtr->v_output_data_addr_.push_back((void *)(uintptr_t)(&output_data[0]));
      memcpyAsyncOpPtr->v_output_data_addr_.push_back((void *)(uintptr_t)(&output_data[1]));
      memcpyAsyncOpPtr->GenerateCtxDef(*nodePtr);
      ge::Status ret = memcpyAsyncOpPtr->GenerateCtxDef(*nodePtr);
      ASSERT_EQ(ret, ge::FAILED);
      memcpyAsyncOpPtr->Run(tasks);
      std::cout << "MemcpyAsyncOp GenerateCtxDef success" << std::endl;
    } else {
      std::cout << "memcpyAsyncOpPtr is NULL" << std::endl;
    }
  }
}

TEST(RtsEngineOpTest, inputsize_error4) {
  RunContext runContext;
  vector<TaskDef> tasks;
  int64_t input_data[2] = {0};
  int64_t output_data[2] = {0};
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(op_desc, graph);
  if (nodePtr == NULL) {
    std::cout << "GenerateCtxDef nodePtr is NULL" << std::endl;
  } else {
    std::shared_ptr<MemcpyAsyncOp> memcpyAsyncOpPtr;
    memcpyAsyncOpPtr = std::make_shared<MemcpyAsyncOp>(*nodePtr, runContext);
    if (memcpyAsyncOpPtr != NULL) {
      memcpyAsyncOpPtr->Init();
      memcpyAsyncOpPtr->dynamic_flag_ = true;
      memcpyAsyncOpPtr->v_input_size_.push_back(0);
      memcpyAsyncOpPtr->v_input_size_.push_back(32);
      memcpyAsyncOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[0]));
      memcpyAsyncOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[1]));
      memcpyAsyncOpPtr->v_output_size_.push_back(0);
      memcpyAsyncOpPtr->v_output_size_.push_back(32);
      memcpyAsyncOpPtr->v_output_data_addr_.push_back((void *)(uintptr_t)(&output_data[0]));
      memcpyAsyncOpPtr->v_output_data_addr_.push_back((void *)(uintptr_t)(&output_data[1]));
      memcpyAsyncOpPtr->GenerateCtxDef(*nodePtr);
      ge::Status ret = memcpyAsyncOpPtr->GenerateCtxDef(*nodePtr);
      ASSERT_EQ(ret, ge::SUCCESS);
      memcpyAsyncOpPtr->FillContextInfo(*nodePtr, 1);
      memcpyAsyncOpPtr->Run(tasks);
      std::cout << "MemcpyAsyncOp GenerateCtxDef success" << std::endl;
    } else {
      std::cout << "memcpyAsyncOpPtr is NULL" << std::endl;
    }
  }
}

TEST(RtsEngineOpTest, inputsize_error5) {
  RunContext runContext;
  vector<TaskDef> tasks;
  int64_t input_data[2] = {0};
  int64_t output_data[2] = {0};
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(op_desc, graph);
  if (nodePtr == NULL) {
    std::cout << "GenerateCtxDef nodePtr is NULL" << std::endl;
  } else {
    std::shared_ptr<MemcpyAsyncOp> memcpyAsyncOpPtr;
    memcpyAsyncOpPtr = std::make_shared<MemcpyAsyncOp>(*nodePtr, runContext);
    if (memcpyAsyncOpPtr != NULL) {
      memcpyAsyncOpPtr->Init();
      memcpyAsyncOpPtr->v_input_size_.push_back(0);
      memcpyAsyncOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[0]));
      memcpyAsyncOpPtr->v_output_size_.push_back(0);
      memcpyAsyncOpPtr->v_output_data_addr_.push_back((void *)(uintptr_t)(&output_data[0]));
      memcpyAsyncOpPtr->GenerateCtxDef(*nodePtr);
      ge::Status ret = memcpyAsyncOpPtr->GenerateCtxDef(*nodePtr);
      ASSERT_EQ(ret, ge::FAILED);
      memcpyAsyncOpPtr->FillContextInfo(*nodePtr, 1);
      memcpyAsyncOpPtr->Run(tasks);
      std::cout << "MemcpyAsyncOp CheckInputSize faild, index is invaild" << std::endl;
    } else {
      std::cout << "memcpyAsyncOpPtr is NULL" << std::endl;
    }
  }
}

TEST(RtsEngineOpTest, inputsize_error6) {
  RunContext runContext;
  vector<TaskDef> tasks;
  int64_t input_data[2] = {0};
  int64_t output_data[2] = {0};
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(op_desc, graph);
  if (nodePtr == NULL) {
    std::cout << "GenerateCtxDef nodePtr is NULL" << std::endl;
  } else {
    std::shared_ptr<MemcpyAsyncOp> memcpyAsyncOpPtr;
    memcpyAsyncOpPtr = std::make_shared<MemcpyAsyncOp>(*nodePtr, runContext);
    if (memcpyAsyncOpPtr != NULL) {
      memcpyAsyncOpPtr->Init();
      memcpyAsyncOpPtr->v_input_size_.push_back(0);
      memcpyAsyncOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[0]));
      memcpyAsyncOpPtr->v_output_size_.push_back(0);
      memcpyAsyncOpPtr->v_output_data_addr_.push_back((void *)(uintptr_t)(&output_data[0]));
      memcpyAsyncOpPtr->op_desc_ = nullptr;
      memcpyAsyncOpPtr->GenerateCtxDef(*nodePtr);
      ge::Status ret = memcpyAsyncOpPtr->GenerateCtxDef(*nodePtr);
      ASSERT_EQ(ret, ge::FAILED);
      std::cout << "op_desc_ = " << memcpyAsyncOpPtr->op_desc_ << std::endl;
      memcpyAsyncOpPtr->FillContextInfo(*nodePtr, 1);
      memcpyAsyncOpPtr->Run(tasks);
      std::cout << "MemcpyAsyncOp CheckInputSize faild, op_desc_ is nullptr" << std::endl;
    } else {
      std::cout << "memcpyAsyncOpPtr is NULL" << std::endl;
    }
  }
}

TEST(RtsEngineOpTest, StreamMergeOp_test) {
  RunContext runContext;
  vector<TaskDef> tasks;
  int64_t input_data[2] = {0};
  int64_t output_data[2] = {0};
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(op_desc, graph);
  for (int i = 0; i < 2; ++i) {
    GeTensorDesc in_desc(ge::GeShape({1, 1, 1, 1}));
    TensorUtils::SetSize(in_desc, 16);
    TensorUtils::SetInputTensor(in_desc, true);
    TensorUtils::SetOutputTensor(in_desc, false);
    op_desc->AddInputDesc(in_desc);
  }

  for (int i = 0; i < 1; ++i) {
    ge::GeTensorDesc out_desc(ge::GeShape({1, 1, 1, 1}));
    ge::TensorUtils::SetSize(out_desc, 16);
    ge::TensorUtils::SetInputTensor(out_desc, false);
    ge::TensorUtils::SetOutputTensor(out_desc, true);
    op_desc->AddOutputDesc(out_desc);
  }

  if (nodePtr == NULL) {
    std::cout << "GenerateCtxDef nodePtr is NULL" << std::endl;
  } else {
    std::shared_ptr<StreamMergeOp> streamMergeOpOpPtr;
    streamMergeOpOpPtr = std::make_shared<StreamMergeOp>(*nodePtr, runContext);
    if (streamMergeOpOpPtr != NULL) {
      (void)streamMergeOpOpPtr->Init();
      streamMergeOpOpPtr->v_input_size_.push_back(2);
      streamMergeOpOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[0]));
      streamMergeOpOpPtr->v_output_size_.push_back(2);
      streamMergeOpOpPtr->v_output_data_addr_.push_back((void *)(uintptr_t)(&output_data[0]));
      streamMergeOpOpPtr->op_desc_ = op_desc;
      std::cout << "op_desc_ = " << streamMergeOpOpPtr->op_desc_ << std::endl;
      ge::Status ret = streamMergeOpOpPtr->Run(tasks);
      ASSERT_EQ(ret, ge::SUCCESS);
      std::cout << "streamMergeOpOpPtr CheckInputSize faild, op_desc_ is nullptr" << std::endl;
    } else {
      std::cout << "streamMergeOpOpPtr is NULL" << std::endl;
    }
  }
}

TEST(RtsEngineOpTest, StreamSwitchNOp_test_ok) {
  RunContext runContext;
  vector<TaskDef> tasks;
  int64_t input_data[2] = {0};
  int64_t output_data[2] = {0};
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(op_desc, graph);
  if (nodePtr == NULL) {
    std::cout << "GenerateCtxDef nodePtr is NULL" << std::endl;
  } else {
    std::shared_ptr<StreamSwitchNOp> streamSwitchNOpPtr;
    streamSwitchNOpPtr = std::make_shared<StreamSwitchNOp>(*nodePtr, runContext);
    if (streamSwitchNOpPtr != NULL) {
      vector<uint32_t> active_stream_list;
      active_stream_list.push_back(0U);
      active_stream_list.push_back(1U);
      ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list);
      (void)AttrUtils::SetInt(op_desc, ATTR_NAME_BATCH_NUM, 1U);
      (void)streamSwitchNOpPtr->Init();
      streamSwitchNOpPtr->v_input_size_.push_back(2);
      streamSwitchNOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[0]));
      streamSwitchNOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[0]));
      streamSwitchNOpPtr->activeStreamList_.push_back(1);
      ge::Status ret = streamSwitchNOpPtr->Run(tasks);
      ASSERT_EQ(ret, ge::SUCCESS);
      ret = streamSwitchNOpPtr->UpdateTaskDef(tasks);
      ASSERT_EQ(ret, ge::SUCCESS);
      std::cout << "streamSwitchNOpPtr CheckInputSize faild, op_desc_ is nullptr" << std::endl;
    } else {
      std::cout << "streamSwitchNOpPtr is NULL" << std::endl;
    }
  }
}

TEST(RtsEngineOpTest, StreamSwitchNOp_test_fail) {
  RunContext runContext;
  vector<TaskDef> tasks;
  int64_t input_data[2] = {0};
  int64_t output_data[2] = {0};
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(op_desc, graph);
  if (nodePtr == NULL) {
    std::cout << "GenerateCtxDef nodePtr is NULL" << std::endl;
  } else {
    std::shared_ptr<StreamSwitchNOp> streamSwitchNOpPtr;
    streamSwitchNOpPtr = std::make_shared<StreamSwitchNOp>(*nodePtr, runContext);
    if (streamSwitchNOpPtr != NULL) {
      streamSwitchNOpPtr->v_input_size_.push_back(2);
      streamSwitchNOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[0]));
      streamSwitchNOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[0]));
      streamSwitchNOpPtr->activeStreamList_.push_back(1);

      ge::Status ret = streamSwitchNOpPtr->Run(tasks);
      ASSERT_EQ(ret, ge::SUCCESS);
      ret = streamSwitchNOpPtr->UpdateTaskDef(tasks);
      ASSERT_EQ(ret, ge::FAILED);
      vector<uint32_t> active_stream_list;
      active_stream_list.push_back(0U);
      active_stream_list.push_back(1U);
      ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list);
      ret = streamSwitchNOpPtr->UpdateTaskDef(tasks);
      ASSERT_EQ(ret, ge::FAILED);
      (void)AttrUtils::SetInt(op_desc, ATTR_NAME_BATCH_NUM, 3U);
      ret = streamSwitchNOpPtr->UpdateTaskDef(tasks);
      ASSERT_EQ(ret, ge::FAILED);
      std::cout << "streamSwitchNOpPtr CheckInputSize faild, op_desc_ is nullptr" << std::endl;
    } else {
      std::cout << "streamSwitchNOpPtr is NULL" << std::endl;
    }
  }
}

TEST(RtsEngineOpTest, stream_switch_op_failed) {
  RunContext runContext;
  vector<TaskDef> tasks;
  int64_t input_data[2] = {0};
  int64_t output_data[2] = {0};
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(op_desc, graph);
  if (nodePtr == NULL) {
    std::cout << "GenerateCtxDef nodePtr is NULL" << std::endl;
  } else {
    // stream_switch_op
    std::shared_ptr<StreamSwitchOp> streamSwitchOpPtr;
    streamSwitchOpPtr = std::make_shared<StreamSwitchOp>(*nodePtr, runContext);
    if (streamSwitchOpPtr != NULL) {
      ge::Status ret = streamSwitchOpPtr->UpdateTaskDef(tasks);
      ASSERT_EQ(ret, ge::FAILED);
      vector<uint32_t> active_stream_list;
      active_stream_list.push_back(0);
      active_stream_list.push_back(1);
      (void)AttrUtils::SetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list);
      ret = streamSwitchOpPtr->UpdateTaskDef(tasks);
      ASSERT_EQ(ret, ge::FAILED);
      std::cout << "StreamSwitchOp GenerateCtxDef success" << std::endl;
    } else {
      std::cout << "streamSwitchNOpPtr is NULL" << std::endl;
    }
  }
}

TEST(RtsEngineOpTest, stream_active_op_ok) {
  RunContext runContext;
  vector<TaskDef> tasks;
  int64_t input_data[2] = {0};
  int64_t output_data[2] = {0};
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(op_desc, graph);
  if (nodePtr == NULL) {
    std::cout << "GenerateCtxDef nodePtr is NULL" << std::endl;
  } else {
    // StreamActiveOp
    std::shared_ptr<StreamActiveOp> StreamActiveOpPtr;
    StreamActiveOpPtr = std::make_shared<StreamActiveOp>(*nodePtr, runContext);
    if (StreamActiveOpPtr != NULL) {
      vector<uint32_t> active_stream_list;
      active_stream_list.push_back(0U);
      ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list);
      ge::Status ret = StreamActiveOpPtr->Init();
      ret = StreamActiveOpPtr->Run(tasks);
      ASSERT_EQ(ret, ge::SUCCESS);
      ret = StreamActiveOpPtr->UpdateTaskDef(tasks);
      ASSERT_EQ(ret, ge::SUCCESS);
      std::cout << "StreamActiveOpPtr success" << std::endl;
    } else {
      std::cout << "StreamActiveOpPtr is NULL" << std::endl;
    }
  }
}

TEST(RtsEngineOpTest, stream_active_op_failed) {
  RunContext runContext;
  vector<TaskDef> tasks;
  int64_t input_data[2] = {0};
  int64_t output_data[2] = {0};
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(op_desc, graph);
  if (nodePtr == NULL) {
    std::cout << "GenerateCtxDef nodePtr is NULL" << std::endl;
  } else {
    // StreamActiveOp
    std::shared_ptr<StreamActiveOp> StreamActiveOpPtr;
    StreamActiveOpPtr = std::make_shared<StreamActiveOp>(*nodePtr, runContext);
    if (StreamActiveOpPtr != NULL) {
      ge::Status ret = StreamActiveOpPtr->Init();
      ret = StreamActiveOpPtr->Run(tasks);
      ASSERT_EQ(ret, ge::FAILED);
      ret = StreamActiveOpPtr->UpdateTaskDef(tasks);
      ASSERT_EQ(ret, ge::FAILED);
      vector<uint32_t> active_stream_list;
      active_stream_list.push_back(0U);
      ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list);
      ret = StreamActiveOpPtr->UpdateTaskDef(tasks);
      ASSERT_EQ(ret, ge::SUCCESS);
      std::cout << "StreamActiveOpPtr success" << std::endl;
    } else {
      std::cout << "StreamActiveOpPtr is NULL" << std::endl;
    }
  }
}

TEST(RtsEngineOpTest, Need4GMemoryAlloc) {
  RunContext runContext;
  vector<TaskDef> tasks;
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(op_desc, graph);
  if (nodePtr == NULL) {
    std::cout << "LabelSwitchByIndexOpTest nodePtr is NULL" << std::endl;
  } else {
    std::shared_ptr<LabelSwitchByIndexOp> opPtr;
    opPtr = std::make_shared<LabelSwitchByIndexOp>(*nodePtr, runContext);
    if (opPtr != NULL) {
      opPtr->Init();
      ge::Status ret = opPtr->Run(tasks);
      ASSERT_EQ(ret, ge::SUCCESS);
      ret = opPtr->UpdateTaskDef(tasks);
      ASSERT_EQ(ret, ge::SUCCESS);
      std::cout << "LabelSwitchByIndexOpTest Need4GMemoryAlloc run" << std::endl;
    } else {
      std::cout << "LabelSwitchByIndexOpTest LabelSwitchByIndexOp ptr is NULL" << std::endl;
    }
  }
}

TEST(RTS_OPS_KERNEL_BUILDER, CalcOpRunningParam_02) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(nullptr, graph);
  RtsOpsKernelBuilder kernelBuilder;
  auto ret = kernelBuilder.CalcOpRunningParam(*nodePtr);
  ASSERT_EQ(ret, ge::FAILED);
}

TEST(RTS_OPS_KERNEL_BUILDER, CalcOpRunningParam_03) {
  std::shared_ptr<OpDesc> opDesc = std::make_shared<OpDesc>("test", "test");
  vector<int64_t> tensorShape = {-1, 1, 3, 1};
  GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
  opDesc->AddInputDesc("x1", tensor1);
  opDesc->AddInputDesc("x2", tensor1);
  opDesc->AddOutputDesc("y", tensor1);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(opDesc, graph);
  RtsOpsKernelBuilder kernelBuilder;
  auto ret = kernelBuilder.CalcOpRunningParam(*nodePtr);
  ASSERT_EQ(ret, ge::SUCCESS);
}

TEST(RTS_OPS_KERNEL_BUILDER, CalcOpRunningParam_04) {
  std::shared_ptr<OpDesc> opDesc = std::make_shared<OpDesc>("test", "test");
  vector<int64_t> tensorShape = {-1, 1, 3, 1};
  GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
  opDesc->AddInputDesc("x1", tensor1);
  opDesc->AddInputDesc("x2", tensor1);
  opDesc->AddOutputDesc("y", tensor1);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(opDesc, graph);
  RtsOpsKernelBuilder kernelBuilder;
  auto ret = kernelBuilder.CalcOpRunningParam(*nodePtr);
  ASSERT_EQ(ret, ge::SUCCESS);
}

TEST(RTS_OPS_KERNEL_BUILDER, CalcOpRunningParam_06) {
  std::shared_ptr<OpDesc> opDesc = std::make_shared<OpDesc>("test", "test");
  vector<int64_t> tensorShape = {-1, 1, 3, 1};
  GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
  opDesc->AddInputDesc("x1", tensor1);
  opDesc->AddInputDesc("x2", tensor1);
  opDesc->AddOutputDesc("y", tensor1);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(opDesc, graph);
  RtsFftsPlusOpsKernelBuilder kernelBuilder;
  auto ret = kernelBuilder.CalcOpRunningParam(*nodePtr);
  ASSERT_EQ(ret, ge::SUCCESS);
}

TEST(MemcpyAsyncOp, Notiling_01) {
  vector<TaskDef> tasks;
  std::shared_ptr<OpDesc> opDesc = std::make_shared<OpDesc>("test", "test");
  ge::AttrUtils::SetBool(opDesc, ge::ATTR_NAME_OP_NO_TILING, true);
  vector<int64_t> tensorShape = {-1, 1, 3, 1};
  GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
  std::vector<std::pair<int64_t, int64_t>> range = {{1, 1}, {1, 1}, {1, 3}, {1, 1}};
  tensor1.SetShapeRange(range);
  ge::TensorUtils::SetSize(tensor1, 512);
  ge::AttrUtils::SetBool(tensor1, ge::ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
  ge::AttrUtils::SetInt(tensor1, ge::ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 1);
  opDesc->AddInputDesc("x1", tensor1);
  opDesc->AddOutputDesc("y", tensor1);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  vector<int64_t> offsets = {0};
  opDesc->SetInputOffset(offsets);
  opDesc->SetOutputOffset(offsets);
  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(opDesc, graph);
  auto node2 = graph->AddNode(opDesc);
  nodePtr->AddLinkFrom(node2);
  for (auto anchor : nodePtr->GetAllInDataAnchors()) {
    (void)AnchorUtils::SetStatus(anchor, ANCHOR_DATA);
  }
  RunContext runContext;
  MemcpyAsyncOp op(*nodePtr, runContext);
  auto ret = op.Init();
  ASSERT_EQ(ret, ge::SUCCESS);
  ret = op.Run(tasks);
  ASSERT_EQ(ret, ge::SUCCESS);
  ret = op.UpdateTaskDef(tasks);
  ASSERT_EQ(ret, ge::SUCCESS);
}

TEST(MemcpyAsyncOp, Notiling_02) {
  std::shared_ptr<OpDesc> opDesc = std::make_shared<OpDesc>("test", "test");
  ge::AttrUtils::SetBool(opDesc, ge::ATTR_NAME_OP_NO_TILING, true);
  vector<int64_t> tensorShape = {-1, 1, 3, 1};
  GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
  std::vector<std::pair<int64_t, int64_t>> range = {{1, 1}, {1, 1}, {1, 3}, {1, 1}};
  tensor1.SetShapeRange(range);
  ge::TensorUtils::SetSize(tensor1, 512);
  ge::AttrUtils::SetBool(tensor1, ge::ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
  ge::AttrUtils::SetInt(tensor1, ge::ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 200);
  opDesc->AddInputDesc("x1", tensor1);
  opDesc->AddOutputDesc("y", tensor1);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  vector<int64_t> offsets = {0};
  opDesc->SetInputOffset(offsets);
  opDesc->SetOutputOffset(offsets);
  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(opDesc, graph);
  auto node2 = graph->AddNode(opDesc);
  nodePtr->AddLinkFrom(node2);
  for (auto anchor : nodePtr->GetAllInDataAnchors()) {
    (void)AnchorUtils::SetStatus(anchor, ANCHOR_DATA);
  }
  RunContext runContext;
  runContext.dataMemSize = 100;
  MemcpyAsyncOp op(*nodePtr, runContext);
  auto ret = op.Init();
  ASSERT_EQ(ret, ge::FAILED);
}

TEST(MemcpyAsyncOp, Notiling_03) {
  std::shared_ptr<OpDesc> opDesc = std::make_shared<OpDesc>("test", "test");
  ge::AttrUtils::SetBool(opDesc, ge::ATTR_NAME_OP_NO_TILING, true);
  vector<int64_t> tensorShape = {-1, 1, 3, 1};
  GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
  std::vector<std::pair<int64_t, int64_t>> range = {{1, 1}, {1, 1}, {1, 3}, {1, 1}};
  tensor1.SetShapeRange(range);
  ge::TensorUtils::SetSize(tensor1, 512);
  ge::AttrUtils::SetBool(tensor1, ge::ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
  ge::AttrUtils::SetInt(tensor1, ge::ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 1);
  opDesc->AddInputDesc("x1", tensor1);
  opDesc->AddOutputDesc("y", tensor1);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  vector<int64_t> offsets = {0};
  opDesc->SetInputOffset(offsets);
  opDesc->SetOutputOffset(offsets);
  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(opDesc, graph);
  auto node2 = graph->AddNode(opDesc);
  nodePtr->AddLinkFrom(node2);
  for (auto anchor : nodePtr->GetAllInDataAnchors()) {
    (void)AnchorUtils::SetStatus(anchor, ANCHOR_DATA);
  }
  RunContext runContext;
  OpDescPtr op_desc = nodePtr->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "_rt_memcpy_kind", 1);
  MemcpyAsyncOp op(*nodePtr, runContext);
  auto ret = op.Init();
  ASSERT_EQ(ret, ge::FAILED);
}

TEST(MemcpyAsyncOp, Notiling_04) {
  std::shared_ptr<OpDesc> opDesc = std::make_shared<OpDesc>("test", "test");
  ge::AttrUtils::SetBool(opDesc, ge::ATTR_NAME_OP_NO_TILING, true);
  vector<int64_t> tensorShape = {-1, 1, 3, 1};
  GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
  std::vector<std::pair<int64_t, int64_t>> range = {{1, 1}, {1, 1}, {1, 3}, {1, 1}};
  tensor1.SetShapeRange(range);
  ge::TensorUtils::SetSize(tensor1, 512);
  ge::AttrUtils::SetBool(tensor1, ge::ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
  ge::AttrUtils::SetInt(tensor1, ge::ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 1);
  opDesc->AddInputDesc("x1", tensor1);
  opDesc->AddOutputDesc("y", tensor1);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  vector<int64_t> offsets = {0};
  opDesc->SetInputOffset(offsets);
  opDesc->SetOutputOffset(offsets);
  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(opDesc, graph);
  auto node2 = graph->AddNode(opDesc);
  nodePtr->AddLinkFrom(node2);
  for (auto anchor : nodePtr->GetAllInDataAnchors()) {
    (void)AnchorUtils::SetStatus(anchor, ANCHOR_DATA);
  }
  RunContext runContext;
  OpDescPtr op_desc = nodePtr->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "_rt_memcpy_kind", 1);
  vector<int64_t> vMemoryType;
  ge::AttrUtils::SetListInt(op_desc, ge::ATTR_NAME_INPUT_MEM_TYPE_LIST, vMemoryType);
  MemcpyAsyncOp op(*nodePtr, runContext);
  auto ret = op.Init();
  ASSERT_EQ(ret, ge::FAILED);
}

TEST(MemcpyAsyncOp, Notiling_05) {
  std::shared_ptr<OpDesc> opDesc = std::make_shared<OpDesc>("test", "test");
  ge::AttrUtils::SetBool(opDesc, ge::ATTR_NAME_OP_NO_TILING, true);
  vector<int64_t> tensorShape = {-1, 1, 3, 1};
  GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
  std::vector<std::pair<int64_t, int64_t>> range = {{1, 1}, {1, 1}, {1, 3}, {1, 1}};
  tensor1.SetShapeRange(range);
  ge::TensorUtils::SetSize(tensor1, 512);
  ge::AttrUtils::SetBool(tensor1, ge::ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
  ge::AttrUtils::SetInt(tensor1, ge::ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 1);
  opDesc->AddInputDesc("x1", tensor1);
  opDesc->AddOutputDesc("y", tensor1);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  vector<int64_t> offsets = {0};
  opDesc->SetInputOffset(offsets);
  opDesc->SetOutputOffset(offsets);
  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(opDesc, graph);
  auto node2 = graph->AddNode(opDesc);
  nodePtr->AddLinkFrom(node2);
  for (auto anchor : nodePtr->GetAllInDataAnchors()) {
    (void)AnchorUtils::SetStatus(anchor, ANCHOR_DATA);
  }
  RunContext runContext;
  OpDescPtr op_desc = nodePtr->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "_rt_memcpy_kind", 1);
  vector<int64_t> vMemoryType;
  vMemoryType.push_back(0x81U);  // RT_MEMORY_HOST
  ge::AttrUtils::SetListInt(op_desc, ge::ATTR_NAME_INPUT_MEM_TYPE_LIST, vMemoryType);
  MemcpyAsyncOp op(*nodePtr, runContext);
  auto ret = op.Init();
  ASSERT_EQ(ret, ge::FAILED);
}

TEST(MemcpyAsyncOp, Notiling_06) {
  std::shared_ptr<OpDesc> opDesc = std::make_shared<OpDesc>("test", "test");
  ge::AttrUtils::SetBool(opDesc, ge::ATTR_NAME_OP_NO_TILING, true);
  vector<int64_t> tensorShape = {-1, 1, 3, 1};
  GeTensorDesc tensor1(GeShape(tensorShape), FORMAT_NCHW, DT_INT32);
  std::vector<std::pair<int64_t, int64_t>> range = {{1, 1}, {1, 1}, {1, 3}, {1, 1}};
  tensor1.SetShapeRange(range);
  ge::TensorUtils::SetSize(tensor1, 512);
  ge::AttrUtils::SetBool(tensor1, ge::ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
  ge::AttrUtils::SetInt(tensor1, ge::ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 1);
  opDesc->AddInputDesc("x1", tensor1);
  opDesc->AddOutputDesc("y", tensor1);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  vector<int64_t> offsets = {0};
  opDesc->SetInputOffset(offsets);
  opDesc->SetOutputOffset(offsets);
  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(opDesc, graph);
  auto node2 = graph->AddNode(opDesc);
  nodePtr->AddLinkFrom(node2);
  for (auto anchor : nodePtr->GetAllInDataAnchors()) {
    (void)AnchorUtils::SetStatus(anchor, ANCHOR_DATA);
  }
  RunContext runContext;
  OpDescPtr op_desc = nodePtr->GetOpDesc();
  (void)ge::AttrUtils::SetInt(op_desc, "_rt_memcpy_kind", 2);
  vector<int64_t> vMemoryType;
  vMemoryType.push_back(0x81U);  // RT_MEMORY_HOST
  ge::AttrUtils::SetListInt(op_desc, ge::ATTR_NAME_OUTPUT_MEM_TYPE_LIST, vMemoryType);
  MemcpyAsyncOp op(*nodePtr, runContext);
  auto ret = op.Init();
  ASSERT_EQ(ret, ge::FAILED);
}

TEST(RtsEngineOpTest, send_op01) {
  RunContext runContext;
  vector<TaskDef> tasks;
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(op_desc, graph);
  if (nodePtr == NULL) {
    std::cout << "send_op nodePtr is NULL" << std::endl;
  } else {
    std::shared_ptr<SendOp> sendOpPtr;
    sendOpPtr = std::make_shared<SendOp>(*nodePtr, runContext);
    if (sendOpPtr != NULL) {
      sendOpPtr->Init();
      auto ret = sendOpPtr->Run(tasks);
      ASSERT_EQ(ret, ge::SUCCESS);
    }
  }
}

TEST(RtsEngineOpTest, send_op02) {
  RunContext runContext;
  vector<TaskDef> tasks;
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(op_desc, graph);
  if (nodePtr == NULL) {
    std::cout << "send_op nodePtr is NULL" << std::endl;
  } else {
    std::shared_ptr<SendOp> sendOpPtr;
    sendOpPtr = std::make_shared<SendOp>(*nodePtr, runContext);
    if (sendOpPtr != NULL) {
      sendOpPtr->Init();
      auto ret = sendOpPtr->Run(tasks);
      ASSERT_EQ(ret, ge::SUCCESS);
    }
  }
}

TEST(RtsEngineOpTest, send_op03) {
  RunContext runContext;
  vector<TaskDef> tasks;
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(op_desc, graph);
  if (nodePtr == NULL) {
    std::cout << "send_op nodePtr is NULL" << std::endl;
  } else {
    std::shared_ptr<SendOp> sendOpPtr;
    sendOpPtr = std::make_shared<SendOp>(*nodePtr, runContext);
    if (sendOpPtr != NULL) {
      sendOpPtr->Init();
      auto ret = sendOpPtr->Run(tasks);
      ASSERT_EQ(ret, ge::SUCCESS);
    }
  }
}

TEST(RtsEngineOpTest, recv_op03) {
  RunContext runContext;
  vector<TaskDef> tasks;
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(op_desc, graph);
  if (nodePtr == NULL) {
    std::cout << "recv_op nodePtr is NULL" << std::endl;
  } else {
    std::shared_ptr<RecvOp> recvOpPtr;
    recvOpPtr = std::make_shared<RecvOp>(*nodePtr, runContext);
    if (recvOpPtr != NULL) {
      recvOpPtr->Init();
      auto ret = recvOpPtr->Run(tasks);
      ASSERT_EQ(ret, SUCCESS);
      ret = recvOpPtr->UpdateTaskDef(tasks);
      ASSERT_EQ(ret, ge::SUCCESS);
    }
  }
}

TEST(RtsEngineOpTest, check_support_op01) {
  RtsGraphOptimizer rtsGraphOptimizer;
  std::string s = "test";
  rtError_t error = rtsGraphOptimizer.CheckSupportedOP(s);
  ASSERT_EQ(error, rts::RT_ERROR_INVALID_VALUE);
}

TEST(RtsEngineOpTest, RecvNotifyAbnormalrtGetNotifyID) {
  RunContext runContext;
  vector<TaskDef> tasks;
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  (void)ge::AttrUtils::SetInt(op_desc, "notify_id", 0);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(op_desc, graph);
  if (nodePtr == NULL) {
    std::cout << "recv_op nodePtr is NULL" << std::endl;
  } else {
    std::shared_ptr<RecvNotifyOp> recvOpPtr;
    recvOpPtr = std::make_shared<RecvNotifyOp>(*nodePtr, runContext);
    if (recvOpPtr != NULL) {
      recvOpPtr->Init();
      auto ret = recvOpPtr->Run(tasks);
      ASSERT_EQ(ret, SUCCESS);
    }
  }
}

TEST(RtsEngineOpTest, SendNotifyAbnormalrtGetNotifyID) {
  RunContext runContext;
  vector<TaskDef> tasks;
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  (void)ge::AttrUtils::SetInt(op_desc, "notify_id", 0);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(op_desc, graph);
  if (nodePtr == NULL) {
    std::cout << "send_op nodePtr is NULL" << std::endl;
  } else {
    std::shared_ptr<SendNotifyOp> sendOpPtr;
    sendOpPtr = std::make_shared<SendNotifyOp>(*nodePtr, runContext);
    if (sendOpPtr != NULL) {
      sendOpPtr->Init();
      auto ret = sendOpPtr->Run(tasks);
      ASSERT_EQ(ret, SUCCESS);
    }
  }
}

TEST(RtsEngineOpTest, CmoAddrOpTest) {
  RunContext runContext;
  vector<TaskDef> tasks;
  int64_t input_data[2] = {0};
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  (void)ge::AttrUtils::SetInt(op_desc, "Cmo", 0);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(op_desc, graph);
  if (nodePtr == NULL) {
    std::cout << "send_op nodePtr is NULL" << std::endl;
  } else {
    std::shared_ptr<CmoAddrOp> cmoAddrOpPtr;
    cmoAddrOpPtr = std::make_shared<CmoAddrOp>(*nodePtr, runContext);
    if (cmoAddrOpPtr != NULL) {
      cmoAddrOpPtr->Init();
      cmoAddrOpPtr->v_input_size_.push_back(32);
      cmoAddrOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[0]));
      cmoAddrOpPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[1]));
      auto ret = cmoAddrOpPtr->Run(tasks);
      ASSERT_EQ(ret, FAILED);
      ret = cmoAddrOpPtr->UpdateTaskDef(tasks);
      ASSERT_EQ(ret, SUCCESS);
    }
  }
}

TEST(RtsEngineOpTest, SendOpMemTest_1) {
  RunContext runContext;
  vector<TaskDef> tasks;
  int64_t input_data[2] = {0};
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  (void)ge::AttrUtils::SetInt(op_desc, "event_id", 0);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(op_desc, graph);
  if (nodePtr == NULL) {
    std::cout << "send_op nodePtr is NULL" << std::endl;
  } else {
    std::shared_ptr<SendOpMem> sendOpMemPtr;
    sendOpMemPtr = std::make_shared<SendOpMem>(*nodePtr, runContext);
    if (sendOpMemPtr != NULL) {
      sendOpMemPtr->Init();
      sendOpMemPtr->v_input_size_.push_back(32);
      sendOpMemPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[0]));
      sendOpMemPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[1]));
      auto ret = sendOpMemPtr->Run(tasks);
      ASSERT_EQ(ret, SUCCESS);
    }
  }
}

TEST(RtsEngineOpTest, SendOpMemTest_2) {
  RunContext runContext;
  vector<TaskDef> tasks;
  int64_t input_data[2] = {0};
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  (void)ge::AttrUtils::SetInt(op_desc, "SendOpMem", 0);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(op_desc, graph);
  if (nodePtr == NULL) {
    std::cout << "send_op nodePtr is NULL" << std::endl;
  } else {
    std::shared_ptr<SendOpMem> sendOpMemPtr;
    sendOpMemPtr = std::make_shared<SendOpMem>(*nodePtr, runContext);
    if (sendOpMemPtr != NULL) {
      sendOpMemPtr->Init();
      sendOpMemPtr->v_input_size_.push_back(32);
      sendOpMemPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[0]));
      sendOpMemPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[1]));
      auto ret = sendOpMemPtr->Run(tasks);
      ASSERT_EQ(ret, SUCCESS);
    }
  }
}

TEST(RtsEngineOpTest, RecvOpMemTest_1) {
  RunContext runContext;
  vector<TaskDef> tasks;
  int64_t input_data[2] = {0};
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  (void)ge::AttrUtils::SetInt(op_desc, "event_id", 0);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(op_desc, graph);
  if (nodePtr == NULL) {
    std::cout << "recv_op nodePtr is NULL" << std::endl;
  } else {
    std::shared_ptr<RecvOpMem> recvOpMemPtr;
    recvOpMemPtr = std::make_shared<RecvOpMem>(*nodePtr, runContext);
    if (recvOpMemPtr != NULL) {
      recvOpMemPtr->Init();
      recvOpMemPtr->v_input_size_.push_back(32);
      recvOpMemPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[0]));
      recvOpMemPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[1]));
      auto ret = recvOpMemPtr->Run(tasks);
      ASSERT_EQ(ret, SUCCESS);
    }
  }
}

TEST(RtsEngineOpTest, RecvOpMemTest_2) {
  RunContext runContext;
  vector<TaskDef> tasks;
  int64_t input_data[2] = {0};
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  (void)ge::AttrUtils::SetInt(op_desc, "RecvOpMem", 0);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  std::shared_ptr<Node> nodePtr = std::make_shared<Node>(op_desc, graph);
  if (nodePtr == NULL) {
    std::cout << "recv_op nodePtr is NULL" << std::endl;
  } else {
    std::shared_ptr<RecvOpMem> recvOpMemPtr;
    recvOpMemPtr = std::make_shared<RecvOpMem>(*nodePtr, runContext);
    if (recvOpMemPtr != NULL) {
      recvOpMemPtr->Init();
      recvOpMemPtr->v_input_size_.push_back(32);
      recvOpMemPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[0]));
      recvOpMemPtr->v_input_data_addr_.push_back((void *)(uintptr_t)(&input_data[1]));
      auto ret = recvOpMemPtr->Run(tasks);
      ASSERT_EQ(ret, SUCCESS);
    }
  }
}