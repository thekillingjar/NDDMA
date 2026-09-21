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
#include "flow_graph/flow_attr_util.h"
#include "flow_graph/data_flow.h"
#include "dflow/flow_graph/data_flow_attr_define.h"

using namespace ge::dflow;

namespace ge {
class FlowAttrUtilUTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(FlowAttrUtilUTest, SetAttrsToTensorDescTest) {
  FlowAttrUtil flow_attr_util;
  CountBatch count_batch;
  count_batch.batch_size = 34;
  count_batch.slide_stride = 7;
  count_batch.drop_remainder = true;
  DataFlowInputAttr count_batch_attr{DataFlowAttrType::COUNT_BATCH, (void *)&count_batch};
  TimeBatch time_batch;
  time_batch.time_window = 134;
  time_batch.time_interval = 25;
  time_batch.drop_remainder = true;
  DataFlowInputAttr time_batch_attr{DataFlowAttrType::TIME_BATCH, (void *)&time_batch};
  std::vector<DataFlowInputAttr> vec_input_attrs0;
  vec_input_attrs0.emplace_back(time_batch_attr);
  std::vector<DataFlowInputAttr> vec_input_attrs1;
  vec_input_attrs1.emplace_back(count_batch_attr);
  std::vector<DataFlowInputAttr> vec_input_attrs2;
  vec_input_attrs2.emplace_back(time_batch_attr);
  vec_input_attrs2.emplace_back(count_batch_attr);

  auto node0 = FlowNode("node0", 3, 2);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(node0);
  auto input_tensor_desc0 = op_desc->MutableInputDesc(0);
  flow_attr_util.SetAttrsToTensorDesc(vec_input_attrs0, input_tensor_desc0);

  // time batch
  int64_t time_window = 0L;
  ge::AttrUtils::GetInt(input_tensor_desc0, ATTR_NAME_TIME_BATCH_TIME_WINDOW, time_window);
  ASSERT_EQ(time_window, time_batch.time_window);
  int64_t time_interval = 0L;
  ge::AttrUtils::GetInt(input_tensor_desc0, ATTR_NAME_TIME_BATCH_TIME_INTERVAL, time_interval);
  ASSERT_EQ(time_interval, time_batch.time_interval);
  // count batch
  auto input_tensor_desc1 = op_desc->MutableInputDesc(1);
  flow_attr_util.SetAttrsToTensorDesc(vec_input_attrs1, input_tensor_desc1);
  int64_t batch_size = 0L;
  ge::AttrUtils::GetInt(input_tensor_desc1, ATTR_NAME_COUNT_BATCH_BATCH_SIZE, batch_size);
  ASSERT_EQ(batch_size, count_batch.batch_size);
  int64_t slide_stride = 0L;
  ge::AttrUtils::GetInt(input_tensor_desc1, ATTR_NAME_COUNT_BATCH_SLIDE_STRIDE, slide_stride);
  ASSERT_EQ(slide_stride, count_batch.slide_stride);
  int64_t timeout = 0L;
  ge::AttrUtils::GetInt(input_tensor_desc1, ATTR_NAME_COUNT_BATCH_TIMEOUT, timeout);
  ASSERT_EQ(timeout, count_batch.timeout);
  int64_t batch_dim = 0L;
  ge::AttrUtils::GetInt(input_tensor_desc1, ATTR_NAME_COUNT_BATCH_BATCH_DIM, batch_dim);
  ASSERT_EQ(batch_dim, count_batch.batch_dim);
  int32_t flag = 0;
  ge::AttrUtils::GetInt(input_tensor_desc1, ATTR_NAME_COUNT_BATCH_FLAG, flag);
  ASSERT_EQ(flag, count_batch.flag);
  bool padding = false;
  ge::AttrUtils::GetBool(input_tensor_desc1, ATTR_NAME_COUNT_BATCH_PADDING, padding);
  ASSERT_EQ(padding, count_batch.padding);
  bool drop_remainder2 = false;
  ge::AttrUtils::GetBool(input_tensor_desc1, ATTR_NAME_COUNT_BATCH_DROP_REMAINDER, drop_remainder2);
  ASSERT_EQ(drop_remainder2, count_batch.drop_remainder);

  // time batch & count batch
  auto input_tensor_desc2 = op_desc->MutableInputDesc(2);
  flow_attr_util.SetAttrsToTensorDesc(vec_input_attrs2, input_tensor_desc2);
  time_window = 0L;
  ge::AttrUtils::GetInt(input_tensor_desc2, ATTR_NAME_TIME_BATCH_TIME_WINDOW, time_window);
  ASSERT_EQ(time_window, 0);
}

TEST_F(FlowAttrUtilUTest, SetAttrsToTensorDescTest_Failed) {
  FlowAttrUtil flow_attr_util;
  CountBatch invalid_batch_size;
  invalid_batch_size.batch_size = 0;
  DataFlowInputAttr invalid_batch_size_attr{DataFlowAttrType::COUNT_BATCH, (void *) &invalid_batch_size};
  std::vector<DataFlowInputAttr> vec_input_attrs0;
  vec_input_attrs0.emplace_back(invalid_batch_size_attr);
  auto node0 = FlowNode("node0", 3, 2);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(node0);
  auto input_tensor_desc0 = op_desc->MutableInputDesc(0);
  ASSERT_NE(flow_attr_util.SetAttrsToTensorDesc(vec_input_attrs0, input_tensor_desc0), ge::GRAPH_SUCCESS);

  CountBatch invalid_slide_stride;
  invalid_slide_stride.batch_size = 10;
  invalid_slide_stride.slide_stride = 20;
  DataFlowInputAttr invalid_slide_stride_attr{DataFlowAttrType::COUNT_BATCH, (void *) &invalid_slide_stride};
  std::vector<DataFlowInputAttr> vec_input_attrs1;
  vec_input_attrs1.emplace_back(invalid_slide_stride_attr);
  ASSERT_NE(flow_attr_util.SetAttrsToTensorDesc(vec_input_attrs1, input_tensor_desc0), ge::GRAPH_SUCCESS);

  TimeBatch invalid_batch_dim;
  invalid_batch_dim.batch_dim = -2;
  DataFlowInputAttr invalid_batch_dim_attr{DataFlowAttrType::TIME_BATCH, (void *) &invalid_batch_dim};
  std::vector<DataFlowInputAttr> vec_input_attrs2;
  vec_input_attrs2.emplace_back(invalid_batch_dim_attr);
  ASSERT_NE(flow_attr_util.SetAttrsToTensorDesc(vec_input_attrs2, input_tensor_desc0), ge::GRAPH_SUCCESS);
}
} // namespace ge
