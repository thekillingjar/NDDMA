/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "securec.h"
#include "ascend_hal.h"
#include "flow_func/mbuf_flow_msg.h"

#define private public

#include "flow_func/flow_func_statistic.h"

#undef private

#include "mockcpp/mockcpp.hpp"
#include "common/data_utils.h"

namespace FlowFunc {
namespace {
struct MbufImpl {
  uint32_t mbuf_size;
  uint8_t reserve_head[256 - sizeof(MbufHeadMsg)];
  MbufHeadMsg head_msg;
  RuntimeTensorDesc tensor_desc;
  uint8_t data[1024];
};

int halMbufAllocExStub(uint64_t size, unsigned int align, unsigned long flag, int grp_id, Mbuf **mbuf) {
  MbufImpl *mbuf_impl = new (std::nothrow) MbufImpl();
  memset_s(mbuf_impl, sizeof(MbufImpl), 0, sizeof(MbufImpl));
  mbuf_impl->mbuf_size = size;
  *mbuf = reinterpret_cast<Mbuf *>(mbuf_impl);
  return DRV_ERROR_NONE;
}

int halMbufFreeStub(Mbuf *mbuf) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  delete mbuf_impl;
  return DRV_ERROR_NONE;
}

int halMbufGetBuffAddrStub(Mbuf *mbuf, void **buf) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  *buf = &(mbuf_impl->tensor_desc);
  return DRV_ERROR_NONE;
}

int halMbufGetBuffSizeStub(Mbuf *mbuf, uint64_t *total_size) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  *total_size = mbuf_impl->mbuf_size;
  return DRV_ERROR_NONE;
}

int halMbufSetDataLenStub(Mbuf *mbuf, uint64_t len) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  mbuf_impl->mbuf_size = len;
  return DRV_ERROR_NONE;
}

int halMbufGetPrivInfoStub(Mbuf *mbuf, void **priv, unsigned int *size) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  // start from first byte.
  *priv = &mbuf_impl->reserve_head;
  *size = 256;
  return DRV_ERROR_NONE;
}

int halMbufGetDataLenStub(Mbuf *mbuf, uint64_t *len) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  *len = mbuf_impl->mbuf_size;
  return DRV_ERROR_NONE;
}
}  // namespace
class FlowFuncStatisticUTest : public testing::Test {
 protected:
  virtual void SetUp() {
    MOCKER(halMbufAllocEx).defaults().will(invoke(halMbufAllocExStub));
    MOCKER(halMbufFree).defaults().will(invoke(halMbufFreeStub));
    MOCKER(halMbufGetBuffAddr).defaults().will(invoke(halMbufGetBuffAddrStub));
    MOCKER(halMbufGetBuffSize).defaults().will(invoke(halMbufGetBuffSizeStub));
    MOCKER(halMbufSetDataLen).defaults().will(invoke(halMbufSetDataLenStub));
    MOCKER(halMbufGetPrivInfo).defaults().will(invoke(halMbufGetPrivInfoStub));
    MOCKER(halMbufGetDataLen).defaults().will(invoke(halMbufGetDataLenStub));
  }

  virtual void TearDown() {
    GlobalMockObject::verify();
  }
};

TEST_F(FlowFuncStatisticUTest, test_statistic) {
  constexpr size_t input_num = 15;
  constexpr size_t output_num = 1;
  FlowFuncStatistic func_stat;
  func_stat.Init("TestFlowFunc", input_num, output_num);
  ASSERT_EQ(func_stat.statistic_info_.inputs.size(), input_num);
  ASSERT_EQ(func_stat.statistic_info_.outputs.size(), output_num);
  func_stat.ExecuteStart();
  func_stat.ExecuteFinish();

  for (size_t input_idx = 0; input_idx < input_num; ++input_idx) {
    std::vector<int64_t> shape = {15, static_cast<int64_t>(input_idx) + 1};
    TensorDataType data_type = TensorDataType::DT_INT8;
    std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, {});

    std::vector<int64_t> min_shape = {8, static_cast<int64_t>(input_idx) + 1};
    TensorDataType min_data_type = TensorDataType::DT_INT8;
    std::shared_ptr<MbufFlowMsg> min_mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(min_shape, min_data_type, 0, {});

    std::vector<int64_t> max_shape = {5, static_cast<int64_t>(input_idx) + 1};
    TensorDataType max_data_type = TensorDataType::DT_INT32;
    std::shared_ptr<MbufFlowMsg> max_mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(max_shape, max_data_type, 0, {});

    func_stat.RecordInput(input_idx, mbuf_flow_msg);
    func_stat.RecordInput(input_idx, min_mbuf_flow_msg);
    func_stat.RecordInput(input_idx, max_mbuf_flow_msg);

    const IOStatisticInfo &input_statistic = func_stat.statistic_info_.inputs[input_idx];
    EXPECT_EQ(input_statistic.min_size, CalcDataSize(min_shape, min_data_type));
    EXPECT_EQ(input_statistic.min_shape, min_shape);
    EXPECT_EQ(input_statistic.max_size, CalcDataSize(max_shape, max_data_type));
    // max_shape means the shape of max size
    EXPECT_EQ(input_statistic.max_shape, max_shape);
  }

  for (size_t output_idx = 0; output_idx < output_num; ++output_idx) {
    std::vector<int64_t> shape = {8, static_cast<int64_t>(output_idx) + 1};
    TensorDataType data_type = TensorDataType::DT_INT8;
    std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, {});

    func_stat.RecordOutput(output_idx, mbuf_flow_msg);

    const IOStatisticInfo &output_statistic = func_stat.statistic_info_.outputs[output_idx];
    EXPECT_EQ(output_statistic.min_size, CalcDataSize(shape, data_type));
    EXPECT_EQ(output_statistic.min_shape, shape);
    EXPECT_EQ(output_statistic.max_size, CalcDataSize(shape, data_type));
    EXPECT_EQ(output_statistic.max_shape, shape);
  }
  func_stat.DumpMetrics();
}
}  // namespace FlowFunc