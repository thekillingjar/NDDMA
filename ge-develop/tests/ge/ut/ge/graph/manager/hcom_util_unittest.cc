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

#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "common/util.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/passes/standard_optimize/addn_pass.h"
#include "graph/utils/tensor_utils.h"

#include "macro_utils/dt_public_scope.h"
#include "graph/manager/util/hcom_ome_util.h"
#include "ge/ge_api.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;

namespace ge {
namespace {
GeTensorDescPtr CreateTensorDesc(std::initializer_list<int64_t> shape, Format format = FORMAT_NCHW,
                                 DataType data_type = DT_FLOAT) {
  GeShape ge_shape{vector<int64_t>(shape)};
  GeTensorDescPtr tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(ge_shape);
  tensor_desc->SetFormat(format);
  tensor_desc->SetDataType(data_type);
  return tensor_desc;
}

class NodeBuilder {
 public:
  NodeBuilder(const std::string &name, const std::string &type) {
    op_desc_ = std::make_shared<OpDesc>(name, type);
  }

  NodeBuilder &AddInputDesc(std::initializer_list<int64_t> shape = {1, 1, 224, 224}, Format format = FORMAT_NCHW,
                            DataType data_type = DT_FLOAT) {
    op_desc_->AddInputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  NodeBuilder &AddOutputDesc(std::initializer_list<int64_t> shape = {1, 1, 224, 224}, Format format = FORMAT_NCHW,
                             DataType data_type = DT_FLOAT) {
    op_desc_->AddOutputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  NodeBuilder &AddOutputDesc(GeTensorDescPtr tensor_desc) {
    op_desc_->AddOutputDesc(tensor_desc->Clone());
    return *this;
  }

  NodePtr Build(const ComputeGraphPtr &graph) {
    NodePtr node = graph->AddNode(op_desc_);
    return node;
  }

 private:
  OpDescPtr op_desc_;
};
}  // namespace

class UtestHcomUtil : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestHcomUtil, test_GetHcomCount_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node =
      NodeBuilder("node", HCOMRECEIVE).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  auto op_desc = node->GetOpDesc();

  HcomOmeUtil hcom_ome_util;
  int64_t count = 0;
  auto ret = hcom_ome_util.GetHcomCount(op_desc, HCCL_DATA_TYPE_FP32, true, count);
  EXPECT_EQ(ret, 0);
}

TEST_F(UtestHcomUtil, test_GetHcomP2pCount_unsupport) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node =
      NodeBuilder("node", HCOMRECEIVE).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  auto op_desc = node->GetOpDesc();
  int64_t count = 0;
  auto ret = ge::HcomOmeUtil::GetHcomP2pCount(op_desc->GetOutputDesc(0), HCCL_DATA_TYPE_FP32,
                                           HcomOperationType::HCOM_OP_TYPE_BROADCAST, count);
  EXPECT_EQ(ret, PARAM_INVALID);
  EXPECT_TRUE(count == 0);
}

TEST_F(UtestHcomUtil, test_GetHcomCount_succ_2) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = NodeBuilder("node", HCOMSEND).AddInputDesc({1, 1, 224, 224}).Build(graph);
  auto op_desc = node->GetOpDesc();
  HcomOmeUtil hcom_util;
  int64_t count = 0;
  auto ret = hcom_util.GetHcomCount(op_desc, HCCL_DATA_TYPE_FP32, true, count);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(count, 224 * 224);
}

TEST_F(UtestHcomUtil, test_GetHcomCount_succ_3) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = NodeBuilder("node", HCOMSEND)
    .AddInputDesc({1, 1, 224, 224})
    .AddOutputDesc({1, 1, 224, 224})
    .Build(graph);
  auto op_desc = node->GetOpDesc();
  AttrUtils::SetBool(op_desc, "_is_unknown_shape", true);
  ge::TensorUtils::SetSize(*op_desc->MutableInputDesc(0), 50177*4);
  HcomOmeUtil hcom_util;
  int64_t count = 0;
  auto ret = hcom_util.GetHcomCount(op_desc, HCCL_DATA_TYPE_FP32, false, count);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(count, 224 * 224);

  AttrUtils::SetBool(op_desc, "_is_unknown_shape", false);
  ret = hcom_util.GetHcomCount(op_desc, HCCL_DATA_TYPE_FP32, false, count);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(count, 50304);
}

TEST_F(UtestHcomUtil, test_GetHcomCount_succ_5) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = NodeBuilder("node", HCOMSEND)
    .AddInputDesc({})
    .AddOutputDesc({})
    .Build(graph);
  auto op_desc = node->GetOpDesc();
  AttrUtils::SetBool(op_desc, "_is_unknown_shape", false);
  ge::TensorUtils::SetSize(*op_desc->MutableInputDesc(0), 2);
  HcomOmeUtil hcom_util;
  int64_t count = 0;
  auto ret = hcom_util.GetHcomCount(op_desc, HCCL_DATA_TYPE_FP32, true, count);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(count, 1);
}

TEST_F(UtestHcomUtil, test_GetHcomCount_allgather) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node =
      NodeBuilder("node", HCOMALLGATHER).AddInputDesc({1, 1, 2, 16}).AddInputDesc({1, 1, 2, 333})
      .AddOutputDesc({1, 1, 2, 4}).Build(graph);
  auto op_desc = node->GetOpDesc();
  ge::TensorUtils::SetSize(*op_desc->MutableInputDesc(0), 36);
  ge::TensorUtils::SetSize(*op_desc->MutableInputDesc(1), 800);

  HcomOmeUtil hcom_ome_util;

  // not continue input
  int64_t count = 0;
  auto ret = hcom_ome_util.GetHcomCount(op_desc, HCCL_DATA_TYPE_FP32, true, count);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(count, (2 * 16 + 2 * 333));
   
   // continue input
  (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_CONTINUOUS_INPUT, true);
  count = 0;
  ret = hcom_ome_util.GetHcomCount(op_desc, HCCL_DATA_TYPE_FP32, true, count);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(count, (512 * 3) / sizeof(float));
}

TEST_F(UtestHcomUtil, test_GetHcomCount_reducescatter) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node =
      NodeBuilder("node", HCOMREDUCESCATTER).AddInputDesc({1, 1, 2, 332})
      .AddOutputDesc({1, 1, 2, 4}).Build(graph);
  auto op_desc = node->GetOpDesc();
  ge::TensorUtils::SetSize(*op_desc->MutableInputDesc(0), 800);
  (void) AttrUtils::SetInt(op_desc, HCOM_ATTR_RANK_SIZE, 8);

  HcomOmeUtil hcom_ome_util;

  // not continue input
  int64_t count = 0;
  auto ret = hcom_ome_util.GetHcomCount(op_desc, HCCL_DATA_TYPE_FP32, false, count);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(count, (2 * 332) / 8);

  // continue input
  (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_CONTINUOUS_INPUT, true);
  count = 0;
  ret = hcom_ome_util.GetHcomCount(op_desc, HCCL_DATA_TYPE_FP32, false, count);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(count, (512 * 2) / sizeof(float) / 8);
}


TEST_F(UtestHcomUtil, test_GetHcomCount_other) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node =
      NodeBuilder("node", HCOMALLREDUCE).AddInputDesc({1, 1, 2, 333})
      .AddOutputDesc({1, 1, 2, 4}).Build(graph);
  auto op_desc = node->GetOpDesc();
  ge::TensorUtils::SetSize(*op_desc->MutableInputDesc(0), 800);
  (void) AttrUtils::SetInt(op_desc, HCOM_ATTR_RANK_SIZE, 8);

  HcomOmeUtil hcom_ome_util;
  int64_t count = 0;
  auto ret = hcom_ome_util.GetHcomCount(op_desc, HCCL_DATA_TYPE_FP32, false, count);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(count, (512 * 2) / sizeof(float));
}

TEST_F(UtestHcomUtil, test_utils_func) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = NodeBuilder("test_node", HCOMSEND).AddInputDesc({1, 1, 224, 224}).Build(graph);
  std::vector<GETaskKernelHcclInfo> kernel_hccl_infos;
  ASSERT_EQ(HcomOmeUtil::CheckKernelHcclInfo(node->GetOpDesc(), kernel_hccl_infos), PARAM_INVALID);
  ASSERT_EQ(HcomOmeUtil::GetHcclDataType(node->GetOpDesc(), kernel_hccl_infos), PARAM_INVALID);

  node = NodeBuilder("test_node", HVDCALLBACKALLREDUCE).AddInputDesc({1, 1, 224, 224}).Build(graph);
  ASSERT_EQ(HcomOmeUtil::GetHorovodInputs(node->GetOpDesc(), kernel_hccl_infos), PARAM_INVALID);
  GETaskKernelHcclInfo hccl_info;
  kernel_hccl_infos.push_back(hccl_info);
  ASSERT_EQ(HcomOmeUtil::GetHorovodInputs(node->GetOpDesc(), kernel_hccl_infos), SUCCESS);

  node = NodeBuilder("test_node", HCOMBROADCAST).AddInputDesc({1, 1, 224, 224}).Build(graph);
  ASSERT_EQ(HcomOmeUtil::GetAllRootId(node->GetOpDesc(), kernel_hccl_infos), FAILED);
  ge::AttrUtils::SetInt(node->GetOpDesc(), HCOM_ATTR_ROOT_RANK, 9);
  ASSERT_EQ(HcomOmeUtil::GetAllRootId(node->GetOpDesc(), kernel_hccl_infos), SUCCESS);
  ASSERT_EQ(HcomOmeUtil::GetHcclDataType(node->GetOpDesc(), kernel_hccl_infos), SUCCESS);
  ASSERT_EQ(kernel_hccl_infos.front().rootId, 9);

  ge::AttrUtils::SetStr(node->GetOpDesc(), HCOM_ATTR_REDUCE_TYPE, "test");
  HcclReduceOp op_type = HCCL_REDUCE_RESERVED;
  node = NodeBuilder("test_node", HVDCALLBACKALLREDUCE).AddInputDesc({1, 1, 224, 224}).Build(graph);
  ASSERT_EQ(HcomOmeUtil::GetHcclOperationType(node->GetOpDesc(), op_type), PARAM_INVALID);
  ge::AttrUtils::SetStr(node->GetOpDesc(), HCOM_ATTR_REDUCE_TYPE, "min");
  ge::AttrUtils::SetInt(node->GetOpDesc(), ATTR_HOROVOD_ATTR_REDUCE_TYPE, 1);
  ASSERT_EQ(HcomOmeUtil::GetHcclOperationType(node->GetOpDesc(), op_type), SUCCESS);
  ge::AttrUtils::SetInt(node->GetOpDesc(), ATTR_HOROVOD_ATTR_REDUCE_TYPE, 30);
  ASSERT_EQ(HcomOmeUtil::GetHcclOperationType(node->GetOpDesc(), op_type), PARAM_INVALID);
  ASSERT_EQ(HcomOmeUtil::GetHorovodCount(node->GetOpDesc(), kernel_hccl_infos), SUCCESS);

  node = NodeBuilder("test_node", HCOMRECEIVE).AddInputDesc({1, 1, 224, 224}).Build(graph);
  ASSERT_EQ(HcomOmeUtil::GetHcclDataType(node->GetOpDesc(), kernel_hccl_infos), PARAM_INVALID);

  node = NodeBuilder("test_node", HVDWAIT).AddInputDesc({1, 1, 224, 224}).Build(graph);
  ASSERT_EQ(HcomOmeUtil::GetHcclDataType(node->GetOpDesc(), kernel_hccl_infos), SUCCESS);

  node = NodeBuilder("test_node", HVDWAIT).AddInputDesc({1, 1, 224, 224}, FORMAT_NCHW, DT_BOOL).Build(graph);
  ASSERT_EQ(HcomOmeUtil::GetHcclDataType(node->GetOpDesc(), kernel_hccl_infos), SUCCESS);

  node = NodeBuilder("test_node", HCOMBROADCAST).AddInputDesc({1, 1, 224, 224}, FORMAT_NCHW, DT_STRING).Build(graph);
  ASSERT_EQ(HcomOmeUtil::GetHcclDataType(node->GetOpDesc(), kernel_hccl_infos), PARAM_INVALID);
  ASSERT_EQ(HcomOmeUtil::GetHorovodCount(node->GetOpDesc(), kernel_hccl_infos), PARAM_INVALID);
  kernel_hccl_infos.clear();
  ASSERT_EQ(HcomOmeUtil::GetHcclCount(node->GetOpDesc(), kernel_hccl_infos), PARAM_INVALID);

  node = NodeBuilder("test_node", HCOMALLREDUCE).AddInputDesc({1, 1, 224, 224}).Build(graph);
  ge::AttrUtils::SetStr(node->GetOpDesc(), HCOM_ATTR_REDUCE_TYPE, "test_failed");
  ASSERT_EQ(HcomOmeUtil::GetHcclOperationType(node->GetOpDesc(), op_type), PARAM_INVALID);
}

TEST_F(UtestHcomUtil, Test_CheckKernelHcclInfo_success) {
  ge::GeTensorDesc tensor_desc(ge::GeShape({1, 3, 2, 2}), FORMAT_NCHW, ge::DT_FLOAT);
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op_wait ", HVDWAIT);
  op_desc->AddInputDesc(tensor_desc);

  // 1
  std::vector<GETaskKernelHcclInfo> kernel_hccl_infos;
  GETaskKernelHcclInfo kernel_hccl_info;
  kernel_hccl_info.dataType = HCCL_DATA_TYPE_FP32;
  kernel_hccl_info.hccl_type = HVDWAIT;
  kernel_hccl_infos.push_back(kernel_hccl_info);

  Status ret = HcomOmeUtil::CheckKernelHcclInfo(op_desc, kernel_hccl_infos);
  EXPECT_EQ(SUCCESS, ret);

  // 2
  ge::GeTensorDesc tensor_desc1(ge::GeShape({1, 3, 2, 2}), FORMAT_NCHW, ge::DT_FLOAT);
  ge::OpDescPtr op_desc1 = std::make_shared<ge::OpDesc>("op_allreduce ", HVDCALLBACKALLREDUCE);
  op_desc->AddInputDesc(tensor_desc1);

  std::vector<GETaskKernelHcclInfo> kernel_hccl_infos1;
  ret = HcomOmeUtil::CheckKernelHcclInfo(op_desc1, kernel_hccl_infos1);
  EXPECT_EQ(PARAM_INVALID, ret);

  // 3
  ge::GeTensorDesc tensor_desc2(ge::GeShape({1, 3, 2, 2}), FORMAT_NCHW, ge::DT_FLOAT);
  ge::OpDescPtr op_desc2 = std::make_shared<ge::OpDesc>("op_allreduce ", HCOMALLREDUCE);
  op_desc->AddInputDesc(tensor_desc2);

  std::vector<GETaskKernelHcclInfo> kernel_hccl_infos2;
  ret = HcomOmeUtil::CheckKernelHcclInfo(op_desc2, kernel_hccl_infos2);
  EXPECT_EQ(PARAM_INVALID, ret);
}

TEST_F(UtestHcomUtil, Test_GetHorovodInputs_success) {
  ge::GeTensorDesc tensor_desc(ge::GeShape({1, 3, 2, 2}), FORMAT_NCHW, ge::DT_FLOAT);
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("op_allreduce ", HVDCALLBACKALLREDUCE);
  op_desc->AddInputDesc(tensor_desc);
  AttrUtils::SetInt(op_desc, ATTR_HOROVOD_ATTR_REDUCE_TYPE, 3);

  std::vector<GETaskKernelHcclInfo> kernel_hccl_infos;
  GETaskKernelHcclInfo kernel_hccl_info;
  kernel_hccl_info.dataType = HCCL_DATA_TYPE_FP32;
  kernel_hccl_info.hccl_type = HVDCALLBACKALLREDUCE;
  kernel_hccl_infos.push_back(kernel_hccl_info);

  Status ret = HcomOmeUtil::GetHorovodInputs(op_desc, kernel_hccl_infos);
  EXPECT_EQ(SUCCESS, ret);

  ge::GeTensorDesc tensor_desc1(ge::GeShape({1, 3, 2, 2}), FORMAT_NCHW, ge::DT_FLOAT);
  ge::OpDescPtr op_desc1 = std::make_shared<ge::OpDesc>("op_allreduce ", HCOMALLREDUCE);
  op_desc1->AddInputDesc(tensor_desc1);
  ret = HcomOmeUtil::GetHorovodInputs(op_desc1, kernel_hccl_infos);
  EXPECT_EQ(SUCCESS, ret);


  ge::GeTensorDesc tensor_desc2(ge::GeShape({1, 3, 2, 2}), FORMAT_NCHW, ge::DT_FLOAT);
  ge::OpDescPtr op_desc2 = std::make_shared<ge::OpDesc>("op_allreduce ", HVDCALLBACKALLREDUCE);
  op_desc2->AddInputDesc(tensor_desc2);

  std::vector<GETaskKernelHcclInfo> kernel_hccl_infos2;
  ret = HcomOmeUtil::GetHorovodInputs(op_desc2, kernel_hccl_infos2);
  EXPECT_EQ(PARAM_INVALID, ret);
}

}  // namespace ge
