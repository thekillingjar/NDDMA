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

#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/compute_graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/compute_graph.h"

#define protected public
#define private public
#include "common/format/axis_util.h"
#include "graph_optimizer/fe_graph_optimizer.h"
#include "graph_optimizer/graph_fusion/graph_matcher.h"
#include "../../../graph_constructor/graph_constructor.h"
#include "common/unknown_shape_util.h"
#include "common/range_format_transfer/transfer_range_according_to_format.h"
#include "graph_optimizer/json_parser/tbe_json_parse.h"
#undef private
#undef protected

using namespace std;
using namespace testing;
using namespace ge;
using namespace fe;

class STestFusionEngineCoverage : public testing::Test {
 protected:
  void SetUp()
  {
  }

  void TearDown()
  {
  }
};

TEST_F(STestFusionEngineCoverage, coverage6) {
  vector<std::pair<int64_t, int64_t>> new_range;
  int64_t impl_type = 1;
  vector<std::pair<int64_t, int64_t>> range_value;
  vector<std::pair<int64_t, int64_t>> nd_range_value;

  vector<std::pair<int64_t, int64_t>> range_value_not_empty = {8, {1,1}};
  vector<std::pair<int64_t, int64_t>> nd_range_value_not_empty = {8, {1, 2}};

  EXPECT_EQ(RangeTransferAccordingToFormat::GetNCHWRangeByAxisValue(new_range, impl_type, range_value, nd_range_value),
            fe::SUCCESS);

  EXPECT_EQ(RangeTransferAccordingToFormat::GetNHWCRangeByAxisValue(new_range, impl_type, range_value, nd_range_value),
            fe::SUCCESS);

  EXPECT_EQ(RangeTransferAccordingToFormat::GetNHWCRangeByAxisValue(new_range, impl_type, range_value_not_empty,
                                                                    nd_range_value),
            fe::SUCCESS);

  EXPECT_EQ(RangeTransferAccordingToFormat::GetC1HWC0RangeByAxisValue(new_range, impl_type, range_value_not_empty,
                                                                      nd_range_value),
            fe::SUCCESS);

  EXPECT_EQ(
      RangeTransferAccordingToFormat::GetNC1HWC0RangeByAxisValue(new_range, impl_type, range_value, nd_range_value),
      fe::SUCCESS);

  EXPECT_EQ(
      RangeTransferAccordingToFormat::GetC1HWC0RangeByAxisValue(new_range, impl_type, range_value, nd_range_value),
      fe::SUCCESS);

  EXPECT_EQ(RangeTransferAccordingToFormat::GetFzRangeByAxisValue(new_range, impl_type, range_value, nd_range_value),
            fe::SUCCESS);

  EXPECT_EQ(RangeTransferAccordingToFormat::GetFzRangeByAxisValue(new_range, impl_type, range_value_not_empty,
                                                                  nd_range_value),
            fe::SUCCESS);

  EXPECT_EQ(RangeTransferAccordingToFormat::GetFzRangeByAxisValue(new_range, impl_type, range_value_not_empty,
                                                                  nd_range_value_not_empty),
            fe::SUCCESS);

  EXPECT_EQ(RangeTransferAccordingToFormat::GetFzC04RangeByAxisValue(new_range, impl_type, range_value, nd_range_value),
            fe::SUCCESS);

  EXPECT_EQ(RangeTransferAccordingToFormat::GetHWCNRangeByAxisValue(new_range, impl_type, range_value, nd_range_value),
            fe::SUCCESS);

  EXPECT_EQ(RangeTransferAccordingToFormat::GetHWCNRangeByAxisValue(new_range, impl_type, range_value_not_empty,
                                                                    nd_range_value),
            fe::SUCCESS);

  EXPECT_EQ(RangeTransferAccordingToFormat::GetCHWNRangeByAxisValue(new_range, impl_type, range_value, nd_range_value),
            fe::SUCCESS);

  EXPECT_EQ(RangeTransferAccordingToFormat::GetCHWNRangeByAxisValue(new_range, impl_type, range_value_not_empty,
                                                                    nd_range_value),
            fe::SUCCESS);

  EXPECT_EQ(
      RangeTransferAccordingToFormat::GetC1HWNCoC0RangeByAxisValue(new_range, impl_type, range_value, nd_range_value),
      fe::SUCCESS);

  EXPECT_EQ(RangeTransferAccordingToFormat::GetC1HWNCoC0RangeByAxisValue(new_range, impl_type, range_value_not_empty,
                                                                         nd_range_value),
            fe::SUCCESS);

  EXPECT_EQ(RangeTransferAccordingToFormat::GetNzRangeByAxisValue(new_range, impl_type, range_value, nd_range_value),
            fe::SUCCESS);

  EXPECT_EQ(
      RangeTransferAccordingToFormat::GetNzRangeByAxisValue(new_range, impl_type, range_value, range_value_not_empty),
      fe::SUCCESS);

  EXPECT_EQ(RangeTransferAccordingToFormat::GetNDC1HWC0RangeByAxisValue(new_range, impl_type, range_value,
                                                                        range_value_not_empty),
            fe::SUCCESS);

  EXPECT_EQ(
      RangeTransferAccordingToFormat::GetFz3DRangeByAxisValue(new_range, impl_type, range_value, range_value_not_empty),
      fe::SUCCESS);

  EXPECT_EQ(RangeTransferAccordingToFormat::GetFz3DTransposeRangeByAxisValue(new_range, impl_type, range_value,
                                                                             range_value_not_empty),
            fe::SUCCESS);

  RangeTransferAccordingToFormat::GetFznRNNRangeByAxisValue(new_range, impl_type, range_value, range_value_not_empty);

  ge::GeShape old_shape;
  vector<std::pair<int64_t, int64_t>> old_range;

  ge::Format old_format = ge::FORMAT_RESERVED;
  ge::Format new_format = old_format;
  ge::DataType current_data_type = ge::DT_UNDEFINED;
  int64_t op_impl_type = 1;
  RangeAndFormat range_and_format_info = {old_shape, old_range, new_range,
                                       old_format, new_format, current_data_type,
                                       op_impl_type};
  RangeTransferAccordingToFormat::GetRangeAccordingToFormat(range_and_format_info);

  RangeAndFormat range_and_format_info1 = {old_shape, old_range, new_range,
                                       ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0,
                                       current_data_type, op_impl_type};
  RangeTransferAccordingToFormat::GetRangeAccordingToFormat(range_and_format_info1);
}

TEST_F(STestFusionEngineCoverage, coverage7) {
  ge::GeShape old_shape({-1,-1,-1,-1});
  vector<std::pair<int64_t, int64_t>> old_range = {{3, 10}, {3, 10}, {4, 16}, {64, 64}};
  ge::Format old_format = ge::FORMAT_HWCN;
  ge::Format new_format = ge::FORMAT_FRACTAL_Z_C04;
  int64_t op_impl_type = 6;
  ge::DataType current_data_type = ge::DT_INT8;
  vector<std::pair<int64_t, int64_t>> new_range;
  RangeAndFormat range_and_format_info = {old_shape, old_range, new_range, old_format, new_format,
                                          ge::DT_INT8, op_impl_type};
  EXPECT_EQ(RangeTransferAccordingToFormat::GetRangeAccordingToFormat(range_and_format_info), fe::SUCCESS);
  vector<std::pair<int64_t, int64_t>> new_range1;
  RangeAndFormat range_and_format_info1 = {old_shape, old_range, new_range1, old_format, new_format,
                                           ge::DT_FLOAT16, op_impl_type};
  EXPECT_EQ(RangeTransferAccordingToFormat::GetRangeAccordingToFormat(range_and_format_info1), fe::SUCCESS);
}

TEST_F(STestFusionEngineCoverage, coverage8) {
  ge::GeShape old_shape;
  vector<std::pair<int64_t, int64_t>> old_range;

  ge::Format old_format = ge::FORMAT_RESERVED;
  ge::Format new_format = old_format;
  ge::DataType current_data_type = ge::DT_UNDEFINED;
  int64_t group_count = 0;

  ge::GeShape new_shape;
  ShapeAndFormat format_info = {old_shape, new_shape, old_format, new_format,
                               current_data_type, group_count};
  EXPECT_EQ(GetShapeAccordingToFormat(format_info), fe::SUCCESS);

  ShapeAndFormat format_info1 = {old_shape, new_shape, ge::FORMAT_NC1HWC0,
                               ge::FORMAT_NC1HWC0, current_data_type,
                               group_count};
  EXPECT_EQ(GetShapeAccordingToFormat(format_info1), fe::SUCCESS);
}

TEST_F(STestFusionEngineCoverage, coverage9) {
  OpKernelInfoConstructor constructor;
  OpContent op_content;
  OpKernelInfoPtr op_kernel_info = std::make_shared<OpKernelInfo>("A");
  AttrInfoPtr attr_info = std::make_shared<AttrInfo>("A1");
  attr_info->dtype_ = GeAttrValue::VT_INT;

  op_kernel_info->attrs_info_.emplace_back(attr_info);
  constructor.InitAttrValueSub(op_content, op_kernel_info);

  op_kernel_info->attrs_info_[0]->dtype_ = GeAttrValue::VT_FLOAT;
  constructor.InitAttrValueSub(op_content, op_kernel_info);

  op_kernel_info->attrs_info_[0]->dtype_ = GeAttrValue::VT_BOOL;
  constructor.InitAttrValueSub(op_content, op_kernel_info);

  op_kernel_info->attrs_info_[0]->dtype_ = GeAttrValue::VT_STRING;
  constructor.InitAttrValueSub(op_content, op_kernel_info);

  op_kernel_info->attrs_info_[0]->dtype_ = GeAttrValue::VT_LIST_INT;
  constructor.InitAttrValueSub(op_content, op_kernel_info);

  op_kernel_info->attrs_info_[0]->dtype_ = GeAttrValue::VT_LIST_FLOAT;
  constructor.InitAttrValueSub(op_content, op_kernel_info);

  op_kernel_info->attrs_info_[0]->dtype_ = GeAttrValue::VT_LIST_BOOL;
  constructor.InitAttrValueSub(op_content, op_kernel_info);

  op_kernel_info->attrs_info_[0]->dtype_ = GeAttrValue::VT_BYTES;
  constructor.InitAttrValueSub(op_content, op_kernel_info);

  map<string, string> map_info;
  InputOrOutputInfoPtr input_or_output_info;
  uint32_t dtype_and_format_size_of_first_input;
  OpPattern op_pattern = OP_PATTERN_BROADCAST;
  EXPECT_EQ(
      constructor.InitDtypeByPattern(map_info, input_or_output_info, dtype_and_format_size_of_first_input, op_pattern),
      fe::FAILED);
}

TEST_F(STestFusionEngineCoverage, axis_util_01) {
  int64_t c0 = 16;
  vector<int64_t> dim_vec = {3, 4, 5, 6, 7, 8, 9, 10};
  vector<int64_t> axis_value = {3, 4, 5, 6, 7, 8, 9, 10};
  vector<int64_t> nd_value;
  EXPECT_EQ(AxisUtil::GetAxisValueByOriginFormat(ge::FORMAT_NHWC, dim_vec, c0, axis_value, nd_value), fe::SUCCESS);

  EXPECT_EQ(AxisUtil::GetAxisValueByOriginFormat(ge::FORMAT_CHWN, dim_vec, c0, axis_value, nd_value), fe::SUCCESS);

  EXPECT_EQ(AxisUtil::GetAxisValueByOriginFormat(ge::FORMAT_NC1HWC0, dim_vec, c0, axis_value, nd_value), fe::SUCCESS);

  EXPECT_EQ(AxisUtil::GetAxisValueByOriginFormat(ge::FORMAT_FRACTAL_Z, dim_vec, c0, axis_value, nd_value), fe::SUCCESS);

  EXPECT_EQ(AxisUtil::GetAxisValueByOriginFormat(ge::FORMAT_DHWNC, dim_vec, c0, axis_value, nd_value), fe::SUCCESS);
}

TEST_F(STestFusionEngineCoverage, axis_util_02) {
  ge::OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("test", "testop");
  ge::GeShape shape({1, 1, 1, 1});
  vector<int64_t> axis_index_vec = {};
  ge::AttrUtils::SetListInt(*op_desc_ptr.get(), AXES_ATTR_NAME, axis_index_vec);
  ge::AttrUtils::SetBool(*op_desc_ptr.get(), "noop_with_empty_axes", false);
  AxisUtil::GetOriginAxisAttribute(*op_desc_ptr.get(), shape, axis_index_vec);
  EXPECT_GE(axis_index_vec[0], 0);
  EXPECT_GE(axis_index_vec[1], 1);
  EXPECT_GE(axis_index_vec[2], 2);
  EXPECT_GE(axis_index_vec[3], 3);
}
TEST_F(STestFusionEngineCoverage, axis_util_03) {
  ge::OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("test", "testop");
  ge::GeShape shape({1, 1, 1, 1});
  vector<int64_t> axis_index_vec = {};
  ge::AttrUtils::SetListInt(*op_desc_ptr.get(), AXES_ATTR_NAME, axis_index_vec);
  ge::AttrUtils::SetBool(*op_desc_ptr.get(), "noop_with_empty_axes", true);
  AxisUtil::GetOriginAxisAttribute(*op_desc_ptr.get(), shape, axis_index_vec);
  EXPECT_GE(axis_index_vec.size(), 0);
}
