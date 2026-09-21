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
#include <vector>
#include "macro_utils/dt_public_scope.h"
#include "dflow/executor/data_flow_data_aligner.h"

using namespace std;
using namespace testing;

namespace ge {
class DataFlowDataAlignerTest : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(DataFlowDataAlignerTest, param_check) {
  std::vector<uint32_t> queue_idxes{6, 7};
  InputAlignAttrs input_align_attrs{};
  input_align_attrs.align_max_cache_num = 10;
  input_align_attrs.drop_when_not_align = true;
  input_align_attrs.align_timeout = -1;
  auto check_ignore_trans_id_func = [](uint64_t trans_id) -> bool { return false; };
  DataFlowDataAligner aligner(queue_idxes, input_align_attrs, check_ignore_trans_id_func);
  ExchangeService::MsgInfo msg_info{};
  msg_info.trans_id = 1;
  msg_info.data_label = 0;
  constexpr uint64_t expect_start_time = 6;
  msg_info.start_time = expect_start_time;
  constexpr uint32_t expect_data0 = 6666;
  GeTensor data0(GeTensorDesc({}, FORMAT_ND, DT_UINT32), reinterpret_cast<const uint8_t *>(&expect_data0),
                 sizeof(expect_data0));
  std::vector<GeTensor> output;
  DataFlowInfo info;

  TensorWithHeader tensor_with_header{};
  tensor_with_header.tensor = std::move(data0);
  tensor_with_header.msg_info = msg_info;
  bool is_aligned = false;
  EXPECT_NE(aligner.PushAndAlignData(0, std::move(tensor_with_header), output, info, is_aligned), SUCCESS);
  EXPECT_TRUE(output.empty());
}

TEST_F(DataFlowDataAlignerTest, normal_align_in_order) {
  std::vector<uint32_t> queue_idxes{6, 7};
  InputAlignAttrs input_align_attrs{};
  input_align_attrs.align_max_cache_num = 10;
  input_align_attrs.drop_when_not_align = true;
  input_align_attrs.align_timeout = -1;
  auto check_ignore_trans_id_func = [](uint64_t trans_id) -> bool { return false; };
  DataFlowDataAligner aligner(queue_idxes, input_align_attrs, check_ignore_trans_id_func);
  ExchangeService::MsgInfo msg_info{};
  msg_info.trans_id = 1;
  msg_info.data_label = 0;
  constexpr uint64_t expect_start_time = 6;
  msg_info.start_time = expect_start_time;
  constexpr uint32_t expect_data0 = 6666;
  GeTensor data0(GeTensorDesc({}, FORMAT_ND, DT_UINT32), reinterpret_cast<const uint8_t *>(&expect_data0),
                 sizeof(expect_data0));
  std::vector<GeTensor> output;
  DataFlowInfo info;
  TensorWithHeader tensor_with_header{};
  tensor_with_header.tensor = data0;
  tensor_with_header.msg_info = msg_info;
  bool is_aligned = false;
  EXPECT_EQ(aligner.PushAndAlignData(6, std::move(tensor_with_header), output, info, is_aligned), SUCCESS);
  EXPECT_TRUE(output.empty());

  EXPECT_EQ(aligner.SelectNextQueueIdx(), 7);
  EXPECT_EQ(aligner.wait_align_data_.size(), 1);
  msg_info.trans_id = 1;
  msg_info.data_label = 0;

  constexpr uint64_t expect_data1 = 7777;
  GeTensor data1(GeTensorDesc({}, FORMAT_ND, DT_UINT64), reinterpret_cast<const uint8_t *>(&expect_data1),
                 sizeof(expect_data1));
  TensorWithHeader tensor_with_header1{};
  tensor_with_header1.tensor = data1;
  tensor_with_header1.msg_info = msg_info;
  EXPECT_EQ(aligner.PushAndAlignData(7, std::move(tensor_with_header1), output, info, is_aligned), SUCCESS);
  EXPECT_TRUE(is_aligned);
  EXPECT_EQ(aligner.wait_align_data_.size(), 0);
  EXPECT_FALSE(output.empty());
  EXPECT_EQ(info.GetStartTime(), expect_start_time);
  ASSERT_EQ(output.size(), 2);
  EXPECT_EQ(output[0].GetTensorDesc().GetShape().GetDims(), std::vector<int64_t>());
  auto out0_data = output[0].GetData().data();
  EXPECT_NE(out0_data, nullptr);
  ASSERT_EQ(output[0].GetData().GetSize(), sizeof(expect_data0));
  EXPECT_EQ(memcmp(output[0].GetData().data(), &expect_data0, sizeof(expect_data0)), 0);

  EXPECT_EQ(output[1].GetTensorDesc().GetShape().GetDims(), std::vector<int64_t>());
  auto out1_data = output[1].GetData().data();
  EXPECT_NE(out1_data, nullptr);
  ASSERT_EQ(output[1].GetData().GetSize(), sizeof(expect_data1));
  EXPECT_EQ(memcmp(output[1].GetData().data(), &expect_data1, sizeof(expect_data1)), 0);
}

TEST_F(DataFlowDataAlignerTest, normal_align_with_repeat) {
  std::vector<uint32_t> queue_idxes{6, 7};
  InputAlignAttrs input_align_attrs{};
  input_align_attrs.align_max_cache_num = 10;
  input_align_attrs.drop_when_not_align = true;
  input_align_attrs.align_timeout = -1;
  auto check_ignore_trans_id_func = [](uint64_t trans_id) -> bool { return false; };
  DataFlowDataAligner aligner(queue_idxes, input_align_attrs, check_ignore_trans_id_func);
  ExchangeService::MsgInfo msg_info{};
  msg_info.trans_id = 1;
  msg_info.data_label = 0;
  constexpr uint64_t expect_start_time = 6;
  msg_info.start_time = expect_start_time;
  constexpr uint32_t expect_data0 = 6666;
  GeTensor data0(GeTensorDesc({}, FORMAT_ND, DT_UINT32), reinterpret_cast<const uint8_t *>(&expect_data0),
                 sizeof(expect_data0));
  std::vector<GeTensor> output;
  DataFlowInfo info;
  TensorWithHeader tensor_with_header{};
  tensor_with_header.tensor = data0;
  tensor_with_header.msg_info = msg_info;
  bool is_aligned = false;
  EXPECT_EQ(aligner.PushAndAlignData(6, std::move(tensor_with_header), output, info, is_aligned), SUCCESS);
  EXPECT_TRUE(output.empty());

  TensorWithHeader tensor_with_header1{};
  tensor_with_header1.tensor = data0;
  tensor_with_header1.msg_info = msg_info;
  EXPECT_EQ(aligner.PushAndAlignData(6, std::move(tensor_with_header1), output, info, is_aligned), SUCCESS);
  EXPECT_TRUE(output.empty());

  EXPECT_EQ(aligner.SelectNextQueueIdx(), 7);
  EXPECT_EQ(aligner.wait_align_data_.size(), 1);
  msg_info.trans_id = 1;
  msg_info.data_label = 0;
  constexpr uint64_t expect_data1 = 7777;
  GeTensor data1(GeTensorDesc({}, FORMAT_ND, DT_UINT64), reinterpret_cast<const uint8_t *>(&expect_data1),
                 sizeof(expect_data1));
  size_t cnt = 0;
  while (cnt++ < 2) {
    TensorWithHeader tensor_with_header2{};
    tensor_with_header2.tensor = data1;
    tensor_with_header2.msg_info = msg_info;
    EXPECT_EQ(aligner.PushAndAlignData(7, std::move(tensor_with_header2), output, info, is_aligned), SUCCESS);
    EXPECT_FALSE(output.empty());
    EXPECT_TRUE(is_aligned);

    EXPECT_EQ(info.GetStartTime(), expect_start_time);
    ASSERT_EQ(output.size(), 2);
    EXPECT_EQ(output[0].GetTensorDesc().GetShape().GetDims(), std::vector<int64_t>());
    auto out0_data = output[0].GetData().data();
    EXPECT_NE(out0_data, nullptr);
    ASSERT_EQ(output[0].GetData().GetSize(), sizeof(expect_data0));
    EXPECT_EQ(memcmp(output[0].GetData().data(), &expect_data0, sizeof(expect_data0)), 0);

    EXPECT_EQ(output[1].GetTensorDesc().GetShape().GetDims(), std::vector<int64_t>());
    auto out1_data = output[1].GetData().data();
    EXPECT_NE(out1_data, nullptr);
    ASSERT_EQ(output[1].GetData().GetSize(), sizeof(expect_data1));
    EXPECT_EQ(memcmp(output[1].GetData().data(), &expect_data1, sizeof(expect_data1)), 0);
    output.clear();
  }
}

TEST_F(DataFlowDataAlignerTest, normal_align_out_of_order) {
  std::vector<uint32_t> queue_idxes{6, 7};
  InputAlignAttrs input_align_attrs{};
  input_align_attrs.align_max_cache_num = 10;
  input_align_attrs.drop_when_not_align = true;
  input_align_attrs.align_timeout = -1;
  auto check_ignore_trans_id_func = [](uint64_t trans_id) -> bool { return false; };
  DataFlowDataAligner aligner(queue_idxes, input_align_attrs, check_ignore_trans_id_func);
  ExchangeService::MsgInfo msg_info{};
  msg_info.trans_id = 1;
  msg_info.data_label = 0;
  constexpr uint64_t expect_start_time = 6;
  msg_info.start_time = expect_start_time;
  constexpr uint32_t expect_data0 = 6666;
  GeTensor data0(GeTensorDesc({}, FORMAT_ND, DT_UINT32), reinterpret_cast<const uint8_t *>(&expect_data0),
                 sizeof(expect_data0));
  std::vector<GeTensor> output;
  DataFlowInfo info;
  TensorWithHeader tensor_with_header{};
  tensor_with_header.tensor = data0;
  tensor_with_header.msg_info = msg_info;
  bool is_aligned = false;
  EXPECT_EQ(aligner.PushAndAlignData(6, tensor_with_header, output, info, is_aligned), SUCCESS);
  EXPECT_TRUE(output.empty());

  bool has_output = true;
  EXPECT_EQ(aligner.TryTakeExpiredOrOverLimitData(output, info, has_output), SUCCESS);
  EXPECT_TRUE(output.empty());
  EXPECT_FALSE(has_output);

  EXPECT_EQ(aligner.SelectNextQueueIdx(), 7);
  EXPECT_EQ(aligner.wait_align_data_.size(), 1);
  msg_info.trans_id = 2;
  msg_info.data_label = 0;
  constexpr uint64_t expect_data1 = 7777;
  GeTensor data1(GeTensorDesc({}, FORMAT_ND, DT_UINT64), reinterpret_cast<const uint8_t *>(&expect_data1),
                 sizeof(expect_data1));

  tensor_with_header.tensor = data1;
  tensor_with_header.msg_info = msg_info;
  EXPECT_EQ(aligner.PushAndAlignData(7, tensor_with_header, output, info, is_aligned), SUCCESS);

  EXPECT_EQ(aligner.wait_align_data_.size(), 2);
  EXPECT_TRUE(output.empty());

  msg_info.trans_id = 2;
  msg_info.data_label = 0;
  tensor_with_header.tensor = data0;
  tensor_with_header.msg_info = msg_info;
  EXPECT_EQ(aligner.PushAndAlignData(6, tensor_with_header, output, info, is_aligned), SUCCESS);
  ASSERT_EQ(output.size(), 2);
  EXPECT_TRUE(is_aligned);
  EXPECT_EQ(aligner.wait_align_data_.size(), 1);
  output.clear();

  msg_info.trans_id = 1;
  msg_info.data_label = 0;
  tensor_with_header.tensor = data1;
  tensor_with_header.msg_info = msg_info;
  is_aligned = false;
  EXPECT_EQ(aligner.PushAndAlignData(7, tensor_with_header, output, info, is_aligned), SUCCESS);
  ASSERT_EQ(output.size(), 2);
  EXPECT_TRUE(is_aligned);
  EXPECT_EQ(aligner.wait_align_data_.size(), 0);
}

TEST_F(DataFlowDataAlignerTest, ignore_trans_id) {
  std::vector<uint32_t> queue_idxes{6, 7};
  InputAlignAttrs input_align_attrs{};
  input_align_attrs.align_max_cache_num = 10;
  input_align_attrs.drop_when_not_align = true;
  input_align_attrs.align_timeout = -1;
  auto check_ignore_trans_id_func = [](uint64_t trans_id) -> bool {
    static const std::set<uint64_t> ignore_trans_id{2, 4};
    return ignore_trans_id.find(trans_id) != ignore_trans_id.cend();
  };
  DataFlowDataAligner aligner(queue_idxes, input_align_attrs, check_ignore_trans_id_func);
  aligner.ClearCacheByTransId(100);
  ExchangeService::MsgInfo msg_info{};
  msg_info.trans_id = 1;
  msg_info.data_label = 0;
  constexpr uint64_t expect_start_time = 6;
  msg_info.start_time = expect_start_time;
  constexpr uint32_t expect_data0 = 6666;
  GeTensor data0(GeTensorDesc({}, FORMAT_ND, DT_UINT32), reinterpret_cast<const uint8_t *>(&expect_data0),
                 sizeof(expect_data0));
  std::vector<GeTensor> output;
  DataFlowInfo info;

  TensorWithHeader tensor_with_header{};
  tensor_with_header.tensor = data0;
  tensor_with_header.msg_info = msg_info;
  bool is_aligned = false;
  EXPECT_EQ(aligner.PushAndAlignData(6, std::move(tensor_with_header), output, info, is_aligned), SUCCESS);
  EXPECT_TRUE(output.empty());
  EXPECT_EQ(aligner.wait_align_data_.size(), 1);

  msg_info.trans_id = 2;
  TensorWithHeader tensor_with_header1{};
  tensor_with_header1.tensor = data0;
  tensor_with_header1.msg_info = msg_info;
  is_aligned = false;
  EXPECT_EQ(aligner.PushAndAlignData(6, std::move(tensor_with_header1), output, info, is_aligned), SUCCESS);
  EXPECT_TRUE(output.empty());
  EXPECT_EQ(aligner.wait_align_data_.size(), 1);

  msg_info.trans_id = 3;
  TensorWithHeader tensor_with_header2{};
  tensor_with_header2.tensor = data0;
  tensor_with_header2.msg_info = msg_info;
  is_aligned = false;
  EXPECT_EQ(aligner.PushAndAlignData(6, std::move(tensor_with_header2), output, info, is_aligned), SUCCESS);
  EXPECT_TRUE(output.empty());
  EXPECT_EQ(aligner.wait_align_data_.size(), 2);

  msg_info.trans_id = 4;
  TensorWithHeader tensor_with_header3{};
  tensor_with_header3.tensor = data0;
  tensor_with_header3.msg_info = msg_info;
  is_aligned = false;
  EXPECT_EQ(aligner.PushAndAlignData(6, std::move(tensor_with_header3), output, info, is_aligned), SUCCESS);
  EXPECT_TRUE(output.empty());
  EXPECT_EQ(aligner.wait_align_data_.size(), 2);

  msg_info.trans_id = 5;
  TensorWithHeader tensor_with_header4{};
  tensor_with_header4.tensor = data0;
  tensor_with_header4.msg_info = msg_info;
  is_aligned = false;
  EXPECT_EQ(aligner.PushAndAlignData(6, std::move(tensor_with_header4), output, info, is_aligned), SUCCESS);
  EXPECT_TRUE(output.empty());
  EXPECT_EQ(aligner.wait_align_data_.size(), 3);
  EXPECT_EQ(aligner.cache_nums_[0], 3);
  EXPECT_EQ(aligner.cache_nums_[1], 0);
  aligner.ClearCacheByTransId(3);
  EXPECT_EQ(aligner.wait_align_data_.size(), 2);
  EXPECT_EQ(aligner.cache_nums_[0], 2);
  EXPECT_EQ(aligner.cache_nums_[1], 0);
}

TEST_F(DataFlowDataAlignerTest, normal_align_over_limit_no_drop) {
  std::vector<uint32_t> queue_idxes{6, 7};
  InputAlignAttrs input_align_attrs{};
  input_align_attrs.align_max_cache_num = 2;
  input_align_attrs.drop_when_not_align = false;
  input_align_attrs.align_timeout = 100000;
  auto check_ignore_trans_id_func = [](uint64_t trans_id) -> bool { return false; };
  DataFlowDataAligner aligner(queue_idxes, input_align_attrs, check_ignore_trans_id_func);
  ExchangeService::MsgInfo msg_info{};
  msg_info.trans_id = 1;
  msg_info.data_label = 0;
  constexpr uint64_t expect_start_time = 6;
  msg_info.start_time = expect_start_time;
  constexpr uint32_t expect_data0 = 6666;
  GeTensor data0(GeTensorDesc({}, FORMAT_ND, DT_UINT32), reinterpret_cast<const uint8_t *>(&expect_data0),
                 sizeof(expect_data0));
  std::vector<GeTensor> output;
  DataFlowInfo info;
  TensorWithHeader tensor_with_header{};
  tensor_with_header.tensor = data0;
  tensor_with_header.msg_info = msg_info;
  bool is_aligned = false;
  EXPECT_EQ(aligner.PushAndAlignData(6, std::move(tensor_with_header), output, info, is_aligned), SUCCESS);
  EXPECT_TRUE(output.empty());
  TensorWithHeader tensor_with_header1{};
  tensor_with_header1.tensor = data0;
  tensor_with_header1.msg_info = msg_info;
  EXPECT_EQ(aligner.PushAndAlignData(6, std::move(tensor_with_header1), output, info, is_aligned), SUCCESS);
  EXPECT_TRUE(output.empty());
  // repeat trans id and data label just as one group
  EXPECT_EQ(aligner.wait_align_data_.size(), 1);

  bool has_output = false;
  EXPECT_EQ(aligner.TryTakeExpiredOrOverLimitData(output, info, has_output), SUCCESS);
  EXPECT_TRUE(output.empty());
  EXPECT_FALSE(has_output);

  EXPECT_EQ(aligner.SelectNextQueueIdx(), 7);
  EXPECT_EQ(aligner.wait_align_data_.size(), 1);
  msg_info.trans_id = 2;
  msg_info.data_label = 0;
  constexpr uint64_t expect_data1 = 7777;
  GeTensor data1(GeTensorDesc({}, FORMAT_ND, DT_UINT64), reinterpret_cast<const uint8_t *>(&expect_data1),
                 sizeof(expect_data1));
  TensorWithHeader tensor_with_header2{};
  tensor_with_header2.tensor = data0;
  tensor_with_header2.msg_info = msg_info;
  EXPECT_EQ(aligner.PushAndAlignData(7, std::move(tensor_with_header2), output, info, is_aligned), SUCCESS);

  EXPECT_EQ(aligner.wait_align_data_.size(), 2);
  EXPECT_TRUE(output.empty());

  EXPECT_EQ(aligner.TryTakeExpiredOrOverLimitData(output, info, has_output), SUCCESS);
  EXPECT_TRUE(output.empty());
  EXPECT_FALSE(has_output);

  msg_info.trans_id = 3;
  msg_info.data_label = 0;
  TensorWithHeader tensor_with_header3{};
  tensor_with_header3.tensor = data0;
  tensor_with_header3.msg_info = msg_info;
  EXPECT_EQ(aligner.PushAndAlignData(6, std::move(tensor_with_header3), output, info, is_aligned), SUCCESS);
  ASSERT_EQ(output.size(), 0);
  EXPECT_EQ(aligner.wait_align_data_.size(), 3);

  EXPECT_EQ(aligner.TryTakeExpiredOrOverLimitData(output, info, has_output), SUCCESS);
  ASSERT_EQ(output.size(), 1);
  EXPECT_TRUE(has_output);
  EXPECT_EQ(aligner.wait_align_data_.size(), 3);

  output.clear();
  has_output = false;
  EXPECT_EQ(aligner.TryTakeExpiredOrOverLimitData(output, info, has_output), SUCCESS);
  ASSERT_EQ(output.size(), 1);
  EXPECT_TRUE(has_output);
  EXPECT_EQ(aligner.wait_align_data_.size(), 2);
}

TEST_F(DataFlowDataAlignerTest, normal_align_over_limit_no_drop_null_data) {
  std::vector<uint32_t> queue_idxes{6, 7};
  InputAlignAttrs input_align_attrs{};
  input_align_attrs.align_max_cache_num = 2;
  input_align_attrs.drop_when_not_align = false;
  input_align_attrs.align_timeout = 100000;
  auto check_ignore_trans_id_func = [](uint64_t trans_id) -> bool { return false; };
  DataFlowDataAligner aligner(queue_idxes, input_align_attrs, check_ignore_trans_id_func);
  ExchangeService::MsgInfo msg_info{};
  msg_info.trans_id = 1;
  msg_info.data_label = 0;
  constexpr uint64_t expect_start_time = 6;
  msg_info.start_time = expect_start_time;
  msg_info.data_flag = kNullDataFlagBit;
  GeTensor data0;
  std::vector<GeTensor> output;
  DataFlowInfo info;
  TensorWithHeader tensor_with_header{};
  tensor_with_header.tensor = data0;
  tensor_with_header.msg_info = msg_info;
  bool is_aligned = false;
  EXPECT_EQ(aligner.PushAndAlignData(6, tensor_with_header, output, info, is_aligned), SUCCESS);
  EXPECT_TRUE(output.empty());
  tensor_with_header.tensor = data0;
  tensor_with_header.msg_info = msg_info;
  EXPECT_EQ(aligner.PushAndAlignData(6, tensor_with_header, output, info, is_aligned), SUCCESS);
  EXPECT_TRUE(output.empty());
  // repeat trans id and data label just as one group
  EXPECT_EQ(aligner.wait_align_data_.size(), 1);

  bool has_output = false;
  EXPECT_EQ(aligner.TryTakeExpiredOrOverLimitData(output, info, has_output), SUCCESS);
  EXPECT_TRUE(output.empty());
  EXPECT_FALSE(has_output);

  EXPECT_EQ(aligner.SelectNextQueueIdx(), 7);
  EXPECT_EQ(aligner.wait_align_data_.size(), 1);
  msg_info.trans_id = 2;
  msg_info.data_label = 0;
  GeTensor data1;
  tensor_with_header.tensor = data0;
  tensor_with_header.msg_info = msg_info;
  EXPECT_EQ(aligner.PushAndAlignData(7, tensor_with_header, output, info, is_aligned), SUCCESS);

  EXPECT_EQ(aligner.wait_align_data_.size(), 2);
  EXPECT_TRUE(output.empty());

  EXPECT_EQ(aligner.TryTakeExpiredOrOverLimitData(output, info, has_output), SUCCESS);
  EXPECT_TRUE(output.empty());
  EXPECT_FALSE(has_output);

  msg_info.trans_id = 3;
  msg_info.data_label = 0;
  tensor_with_header.tensor = data0;
  tensor_with_header.msg_info = msg_info;
  EXPECT_EQ(aligner.PushAndAlignData(6, tensor_with_header, output, info, is_aligned), SUCCESS);
  ASSERT_EQ(output.size(), 0);
  EXPECT_EQ(aligner.wait_align_data_.size(), 3);

  EXPECT_EQ(aligner.TryTakeExpiredOrOverLimitData(output, info, has_output), SUCCESS);
  // null data output is empty
  ASSERT_EQ(output.size(), 0);
  EXPECT_TRUE(has_output);
  EXPECT_EQ(aligner.wait_align_data_.size(), 3);

  output.clear();
  has_output = false;
  EXPECT_EQ(aligner.TryTakeExpiredOrOverLimitData(output, info, has_output), SUCCESS);
  // null data output is empty
  ASSERT_EQ(output.size(), 0);
  EXPECT_TRUE(has_output);
  EXPECT_EQ(aligner.wait_align_data_.size(), 2);
}

TEST_F(DataFlowDataAlignerTest, normal_align_over_limit_drop) {
  std::vector<uint32_t> queue_idxes{6, 7};
  InputAlignAttrs input_align_attrs{};
  input_align_attrs.align_max_cache_num = 2;
  input_align_attrs.drop_when_not_align = false;
  input_align_attrs.align_timeout = -1;
  input_align_attrs.drop_when_not_align = true;
  auto check_ignore_trans_id_func = [](uint64_t trans_id) -> bool { return false; };
  DataFlowDataAligner aligner(queue_idxes, input_align_attrs, check_ignore_trans_id_func);
  ExchangeService::MsgInfo msg_info{};
  msg_info.trans_id = 1;
  msg_info.data_label = 0;
  constexpr uint64_t expect_start_time = 6;
  msg_info.start_time = expect_start_time;
  constexpr uint32_t expect_data0 = 6666;
  GeTensor data0(GeTensorDesc({}, FORMAT_ND, DT_UINT32), reinterpret_cast<const uint8_t *>(&expect_data0),
                 sizeof(expect_data0));
  std::vector<GeTensor> output;
  DataFlowInfo info;
  TensorWithHeader tensor_with_header{};
  tensor_with_header.tensor = data0;
  tensor_with_header.msg_info = msg_info;
  bool is_aligned = false;
  EXPECT_EQ(aligner.PushAndAlignData(6, std::move(tensor_with_header), output, info, is_aligned), SUCCESS);
  EXPECT_TRUE(output.empty());
  EXPECT_EQ(aligner.wait_align_data_.size(), 1);

  TensorWithHeader tensor_with_header1{};
  tensor_with_header1.tensor = data0;
  tensor_with_header1.msg_info = msg_info;
  EXPECT_EQ(aligner.PushAndAlignData(6, std::move(tensor_with_header1), output, info, is_aligned), SUCCESS);
  EXPECT_TRUE(output.empty());
  // repeat trans id and data label only count 1.
  EXPECT_EQ(aligner.wait_align_data_.size(), 1);

  bool has_output = true;
  EXPECT_EQ(aligner.TryTakeExpiredOrOverLimitData(output, info, has_output), SUCCESS);
  EXPECT_TRUE(output.empty());
  EXPECT_FALSE(has_output);

  EXPECT_EQ(aligner.SelectNextQueueIdx(), 7);
  EXPECT_EQ(aligner.wait_align_data_.size(), 1);
  msg_info.trans_id = 2;
  msg_info.data_label = 0;
  constexpr uint64_t expect_data1 = 7777;
  GeTensor data1(GeTensorDesc({}, FORMAT_ND, DT_UINT64), reinterpret_cast<const uint8_t *>(&expect_data1),
                 sizeof(expect_data1));
  TensorWithHeader tensor_with_header2{};
  tensor_with_header2.tensor = data0;
  tensor_with_header2.msg_info = msg_info;
  EXPECT_EQ(aligner.PushAndAlignData(7, std::move(tensor_with_header2), output, info, is_aligned), SUCCESS);

  EXPECT_EQ(aligner.wait_align_data_.size(), 2);
  EXPECT_TRUE(output.empty());

  EXPECT_EQ(aligner.TryTakeExpiredOrOverLimitData(output, info, has_output), SUCCESS);
  EXPECT_FALSE(has_output);

  msg_info.trans_id = 3;
  msg_info.data_label = 0;
  TensorWithHeader tensor_with_header3{};
  tensor_with_header3.tensor = data0;
  tensor_with_header3.msg_info = msg_info;
  EXPECT_EQ(aligner.PushAndAlignData(6, std::move(tensor_with_header3), output, info, is_aligned), SUCCESS);
  ASSERT_EQ(output.size(), 0);
  EXPECT_EQ(aligner.wait_align_data_.size(), 3);

  EXPECT_EQ(aligner.TryTakeExpiredOrOverLimitData(output, info, has_output), SUCCESS);
  // drop data no return data
  ASSERT_EQ(output.size(), 0);
  EXPECT_FALSE(has_output);
  EXPECT_EQ(aligner.wait_align_data_.size(), 2);
  aligner.ClearCache();
  EXPECT_EQ(aligner.wait_align_data_.size(), 0);
  EXPECT_EQ(aligner.cache_nums_, std::vector<std::size_t>({0, 0}));
}

TEST_F(DataFlowDataAlignerTest, normal_align_over_time_no_drop) {
  std::vector<uint32_t> queue_idxes{6, 7};
  InputAlignAttrs input_align_attrs{};
  input_align_attrs.align_max_cache_num = 2;
  input_align_attrs.drop_when_not_align = false;
  input_align_attrs.align_timeout = 1000;
  auto check_ignore_trans_id_func = [](uint64_t trans_id) -> bool { return false; };
  DataFlowDataAligner aligner(queue_idxes, input_align_attrs, check_ignore_trans_id_func);
  ExchangeService::MsgInfo msg_info{};
  msg_info.trans_id = 1;
  msg_info.data_label = 0;
  constexpr uint64_t expect_start_time = 6;
  msg_info.start_time = expect_start_time;
  constexpr uint32_t expect_data0 = 6666;
  GeTensor data0(GeTensorDesc({}, FORMAT_ND, DT_UINT32), reinterpret_cast<const uint8_t *>(&expect_data0),
                 sizeof(expect_data0));
  std::vector<GeTensor> output;
  DataFlowInfo info;
  TensorWithHeader tensor_with_header{};
  tensor_with_header.tensor = data0;
  tensor_with_header.msg_info = msg_info;
  bool is_aligned = false;
  bool has_output = true;
  EXPECT_EQ(aligner.TryTakeExpiredOrOverLimitData(output, info, has_output), SUCCESS);
  EXPECT_TRUE(output.empty());
  EXPECT_FALSE(has_output);
  EXPECT_EQ(aligner.PushAndAlignData(6, std::move(tensor_with_header), output, info, is_aligned), SUCCESS);
  EXPECT_TRUE(output.empty());
  EXPECT_EQ(aligner.wait_align_data_.size(), 1);
  TensorWithHeader tensor_with_header1{};
  tensor_with_header1.tensor = data0;
  tensor_with_header1.msg_info = msg_info;
  EXPECT_EQ(aligner.PushAndAlignData(6, std::move(tensor_with_header1), output, info, is_aligned), SUCCESS);
  EXPECT_TRUE(output.empty());
  EXPECT_EQ(aligner.wait_align_data_.size(), 1);

  EXPECT_EQ(aligner.TryTakeExpiredOrOverLimitData(output, info, has_output), SUCCESS);
  EXPECT_TRUE(output.empty());
  EXPECT_FALSE(has_output);
  EXPECT_EQ(aligner.wait_align_data_.size(), 1);
  EXPECT_EQ(aligner.cache_nums_, std::vector<size_t>({2, 0}));

  aligner.wait_align_data_.begin()->second.start_time_ -= 2000ms;

  EXPECT_EQ(aligner.TryTakeExpiredOrOverLimitData(output, info, has_output), SUCCESS);
  EXPECT_FALSE(output.empty());
  EXPECT_TRUE(has_output);
  EXPECT_EQ(aligner.wait_align_data_.size(), 1);
  EXPECT_EQ(aligner.cache_nums_, std::vector<size_t>({1, 0}));
  output.clear();
  EXPECT_EQ(aligner.TryTakeExpiredOrOverLimitData(output, info, has_output), SUCCESS);
  EXPECT_FALSE(output.empty());
  EXPECT_TRUE(has_output);
  EXPECT_EQ(aligner.wait_align_data_.size(), 0);
  EXPECT_EQ(aligner.cache_nums_, std::vector<size_t>({0, 0}));
}

TEST_F(DataFlowDataAlignerTest, normal_align_over_time_no_drop_null_data) {
  std::vector<uint32_t> queue_idxes{6, 7};
  InputAlignAttrs input_align_attrs{};
  input_align_attrs.align_max_cache_num = 2;
  input_align_attrs.drop_when_not_align = false;
  input_align_attrs.align_timeout = 1000;
  auto check_ignore_trans_id_func = [](uint64_t trans_id) -> bool { return false; };
  DataFlowDataAligner aligner(queue_idxes, input_align_attrs, check_ignore_trans_id_func);
  ExchangeService::MsgInfo msg_info{};
  msg_info.trans_id = 1;
  msg_info.data_label = 0;
  constexpr uint64_t expect_start_time = 6;
  msg_info.start_time = expect_start_time;
  msg_info.data_flag = kNullDataFlagBit;
  GeTensor data0;
  std::vector<GeTensor> output;
  DataFlowInfo info;
  TensorWithHeader tensor_with_header{};
  tensor_with_header.tensor = data0;
  tensor_with_header.msg_info = msg_info;
  bool is_aligned = false;
  bool has_output = true;
  EXPECT_EQ(aligner.TryTakeExpiredOrOverLimitData(output, info, has_output), SUCCESS);
  EXPECT_TRUE(output.empty());
  EXPECT_FALSE(has_output);
  EXPECT_EQ(aligner.PushAndAlignData(6, tensor_with_header, output, info, is_aligned), SUCCESS);
  EXPECT_TRUE(output.empty());
  EXPECT_EQ(aligner.wait_align_data_.size(), 1);

  tensor_with_header.tensor = data0;
  tensor_with_header.msg_info = msg_info;
  EXPECT_EQ(aligner.PushAndAlignData(6, tensor_with_header, output, info, is_aligned), SUCCESS);
  EXPECT_TRUE(output.empty());
  EXPECT_EQ(aligner.wait_align_data_.size(), 1);

  EXPECT_EQ(aligner.TryTakeExpiredOrOverLimitData(output, info, has_output), SUCCESS);
  EXPECT_TRUE(output.empty());
  EXPECT_FALSE(has_output);
  EXPECT_EQ(aligner.wait_align_data_.size(), 1);
  EXPECT_EQ(aligner.cache_nums_, std::vector<size_t>({2, 0}));

  aligner.wait_align_data_.begin()->second.start_time_ -= 2000ms;

  EXPECT_EQ(aligner.TryTakeExpiredOrOverLimitData(output, info, has_output), SUCCESS);
  // null data without output
  EXPECT_TRUE(output.empty());
  EXPECT_TRUE(has_output);
  EXPECT_EQ(aligner.wait_align_data_.size(), 1);
  EXPECT_EQ(aligner.cache_nums_, std::vector<size_t>({1, 0}));
  output.clear();
  EXPECT_EQ(aligner.TryTakeExpiredOrOverLimitData(output, info, has_output), SUCCESS);
  // null data without output
  EXPECT_TRUE(output.empty());
  EXPECT_TRUE(has_output);
  EXPECT_EQ(aligner.wait_align_data_.size(), 0);
  EXPECT_EQ(aligner.cache_nums_, std::vector<size_t>({0, 0}));
}

TEST_F(DataFlowDataAlignerTest, normal_align_over_time_drop) {
  std::vector<uint32_t> queue_idxes{6, 7};
  InputAlignAttrs input_align_attrs{};
  input_align_attrs.align_max_cache_num = 2;
  input_align_attrs.drop_when_not_align = false;
  input_align_attrs.align_timeout = 1000;
  input_align_attrs.drop_when_not_align = true;
  auto check_ignore_trans_id_func = [](uint64_t trans_id) -> bool { return false; };
  DataFlowDataAligner aligner(queue_idxes, input_align_attrs, check_ignore_trans_id_func);
  ExchangeService::MsgInfo msg_info{};
  msg_info.trans_id = 1;
  msg_info.data_label = 0;
  constexpr uint64_t expect_start_time = 6;
  msg_info.start_time = expect_start_time;
  constexpr uint32_t expect_data0 = 6666;
  GeTensor data0(GeTensorDesc({}, FORMAT_ND, DT_UINT32), reinterpret_cast<const uint8_t *>(&expect_data0),
                 sizeof(expect_data0));
  std::vector<GeTensor> output;
  DataFlowInfo info;
  bool has_output = true;
  EXPECT_EQ(aligner.TryTakeExpiredOrOverLimitData(output, info, has_output), SUCCESS);
  EXPECT_TRUE(output.empty());
  EXPECT_FALSE(has_output);

  TensorWithHeader tensor_with_header{};
  tensor_with_header.tensor = data0;
  tensor_with_header.msg_info = msg_info;
  bool is_aligned = false;
  EXPECT_EQ(aligner.PushAndAlignData(6, std::move(tensor_with_header), output, info, is_aligned), SUCCESS);
  EXPECT_TRUE(output.empty());
  EXPECT_EQ(aligner.wait_align_data_.size(), 1);

  EXPECT_EQ(aligner.TryTakeExpiredOrOverLimitData(output, info, has_output), SUCCESS);
  EXPECT_TRUE(output.empty());
  EXPECT_FALSE(has_output);
  EXPECT_EQ(aligner.wait_align_data_.size(), 1);

  aligner.wait_align_data_.begin()->second.start_time_ -= 2000ms;

  EXPECT_EQ(aligner.TryTakeExpiredOrOverLimitData(output, info, has_output), SUCCESS);
  EXPECT_TRUE(output.empty());
  EXPECT_FALSE(has_output);
  EXPECT_EQ(aligner.wait_align_data_.size(), 0);
  EXPECT_EQ(aligner.cache_nums_, std::vector<size_t>({0, 0}));
}
}  // namespace ge
