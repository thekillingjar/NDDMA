/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "test_std_structs.h"

#include <gtest/gtest.h>

#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"

namespace ge {

GeTensorDesc StandardTd_5d_1_1_224_224() {
  GeTensorDesc td;
  td.SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 224, 16})));
  td.SetOriginShape(GeShape(std::vector<int64_t>({1, 1, 224, 224})));
  td.SetFormat(FORMAT_NC1HWC0);
  td.SetOriginFormat(FORMAT_NCHW);
  td.SetDataType(DT_FLOAT16);
  td.SetOriginDataType(DT_FLOAT);

  vector<int64_t> input_size = {12};
  AttrUtils::SetListInt(td, "input_size", input_size);

  return td;
}

void ExpectStandardTdProto_5d_1_1_224_224(const proto::TensorDescriptor &input_td) {
  // shape
  EXPECT_EQ(input_td.shape().dim_size(), 5);
  EXPECT_EQ(input_td.shape().dim(0), 1);
  EXPECT_EQ(input_td.shape().dim(1), 1);
  EXPECT_EQ(input_td.shape().dim(2), 224);
  EXPECT_EQ(input_td.shape().dim(3), 224);
  EXPECT_EQ(input_td.shape().dim(4), 16);

  // origin shape, origin shape is set
  EXPECT_EQ(input_td.attr().count("origin_shape"), 1);
  EXPECT_EQ(input_td.attr().at("origin_shape").value_case(), proto::AttrDef::ValueCase::kList);
  EXPECT_EQ(input_td.attr().at("origin_shape").list().val_type(), proto::AttrDef_ListValue_ListValueType_VT_LIST_INT);
  EXPECT_EQ(input_td.attr().at("origin_shape").list().i_size(), 4);
  EXPECT_EQ(input_td.attr().at("origin_shape").list().i(0), 1);
  EXPECT_EQ(input_td.attr().at("origin_shape").list().i(1), 1);
  EXPECT_EQ(input_td.attr().at("origin_shape").list().i(2), 224);
  EXPECT_EQ(input_td.attr().at("origin_shape").list().i(3), 224);
  EXPECT_EQ(input_td.attr().count("origin_shape_initialized"), 1);
  EXPECT_EQ(input_td.attr().at("origin_shape_initialized").value_case(), proto::AttrDef::ValueCase::kB);
  EXPECT_EQ(input_td.attr().at("origin_shape_initialized").b(), true);

  // format, origin format
  EXPECT_EQ(input_td.attr().count("origin_format"), 1);
  EXPECT_EQ(input_td.attr().at("origin_format").s(), "NCHW");
  EXPECT_EQ(input_td.layout(), "NC1HWC0");

  // data_tpye, origin data_type
  EXPECT_EQ(input_td.dtype(), proto::DT_FLOAT16);
  EXPECT_EQ(input_td.attr().count("origin_data_type"), 1);
  EXPECT_EQ(input_td.attr().at("origin_data_type").s(), "DT_FLOAT");

  EXPECT_EQ(input_td.attr().count("input_size"), 1);
  EXPECT_EQ(input_td.attr().at("input_size").value_case(), proto::AttrDef::ValueCase::kList);
  EXPECT_EQ(input_td.attr().at("input_size").list().val_type(), proto::AttrDef_ListValue_ListValueType_VT_LIST_INT);
  EXPECT_EQ(input_td.attr().at("input_size").list().i_size(), 1);
  EXPECT_EQ(input_td.attr().at("input_size").list().i(0), 12);
}
}
