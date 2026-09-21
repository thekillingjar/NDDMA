/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <string>
#include <gtest/gtest.h>

#include "common/ge_inner_error_codes.h"
#include "graph/common/trans_op_creator.h"
#include "framework/common/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/constant_utils.h"
#include "framework/common/types.h"
#include "graph/utils/type_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "api/gelib/gelib.h"

namespace ge {
class UtestGraphCreateTransOp : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST(UtestGraphCreateTransOp, create_transdata_with_subformat) {
  vector<int64_t> dims = {1,2,3,4};
  GeShape shape(dims);
  Format input_format = static_cast<Format>(GetFormatFromSub(FORMAT_FRACTAL_Z, 32));
  GeTensorDesc input_desc(shape, input_format);
  GeTensorDesc output_desc(shape, FORMAT_NCHW);
  
  auto trans_op = TransOpCreator::CreateTransDataOp("test_trans_data", input_desc, output_desc, false);
  EXPECT_NE(trans_op, nullptr);
  int32_t groups = -1;
  EXPECT_TRUE(AttrUtils::GetInt(trans_op, "groups", groups));
  EXPECT_EQ(groups, 32);

  int32_t sub_srcformat = -1;
  EXPECT_TRUE(AttrUtils::GetInt(trans_op, FORMAT_TRANSFER_SRC_SUBFORMAT, sub_srcformat));
  EXPECT_EQ(sub_srcformat, 32);

  std::string src_format;
  EXPECT_TRUE(AttrUtils::GetStr(trans_op, FORMAT_TRANSFER_SRC_FORMAT, src_format));
  EXPECT_EQ(src_format, TypeUtils::FormatToSerialString(FORMAT_FRACTAL_Z));

  std::string dst_format;
  EXPECT_TRUE(AttrUtils::GetStr(trans_op, FORMAT_TRANSFER_DST_FORMAT, dst_format));
  EXPECT_EQ(dst_format, TypeUtils::FormatToSerialString(FORMAT_NCHW));
}

TEST(UtestGraphCreateTransOp, create_transdata_with_subformat_with_check_failed) {
  // no ge lib init
  vector<int64_t> dims = {1,2,3,4};
  GeShape shape(dims);
  Format input_format = static_cast<Format>(GetFormatFromSub(FORMAT_FRACTAL_Z, 32));
  GeTensorDesc input_desc(shape, input_format);
  GeTensorDesc output_desc(shape, FORMAT_NCHW);
  
  auto trans_op = TransOpCreator::CreateTransDataOp("test_trans_data", input_desc, output_desc);
  EXPECT_EQ(trans_op, nullptr);
}

TEST(UtestGraphCreateTransOp, create_transdata_noneed_groups) {
  vector<int64_t> dims = {1,2,3,4};
  GeShape shape(dims);
  Format input_format = static_cast<Format>(GetFormatFromSub(FORMAT_FRACTAL_NZ, 32));
  GeTensorDesc input_desc(shape, input_format);
  GeTensorDesc output_desc(shape, FORMAT_NCHW);

  auto trans_op = TransOpCreator::CreateTransDataOp("test_trans_data", input_desc, output_desc, false);
  EXPECT_NE(trans_op, nullptr);
  int32_t groups = -1;
  EXPECT_FALSE(AttrUtils::GetInt(trans_op, "groups", groups));

  int32_t sub_srcformat = -1;
  EXPECT_TRUE(AttrUtils::GetInt(trans_op, FORMAT_TRANSFER_SRC_SUBFORMAT, sub_srcformat));
  EXPECT_EQ(sub_srcformat, 32);

  std::string src_format;
  EXPECT_TRUE(AttrUtils::GetStr(trans_op, FORMAT_TRANSFER_SRC_FORMAT, src_format));
  EXPECT_EQ(src_format, TypeUtils::FormatToSerialString(FORMAT_FRACTAL_NZ));

  std::string dst_format;
  EXPECT_TRUE(AttrUtils::GetStr(trans_op, FORMAT_TRANSFER_DST_FORMAT, dst_format));
  EXPECT_EQ(dst_format, TypeUtils::FormatToSerialString(FORMAT_NCHW));
}

TEST(UtestGraphCreateTransOp, create_transposed) {
  vector<int64_t> dims = {1,2,3,4};
  GeShape shape(dims);
  GeTensorDesc input_desc(shape, FORMAT_NHWC);
  GeTensorDesc output_desc(shape, FORMAT_NCHW);
  auto transpos_op = TransOpCreator::CreateTransPoseDOp("test_transposD", input_desc, output_desc);
  EXPECT_NE(transpos_op, nullptr);

  vector<int64_t> perm_args;
  EXPECT_TRUE(AttrUtils::GetListInt(transpos_op, PERMUTE_ATTR_PERM, perm_args));

  EXPECT_TRUE(!perm_args.empty());
}

TEST(UtestGraphCreateTransOp, create_cast) {
  vector<int64_t> dims = {1,2,3,4};
  GeShape shape(dims);
  GeTensorDesc input_desc(shape, FORMAT_NHWC, DT_FLOAT);
  GeTensorDesc output_desc(shape, FORMAT_NCHW, DT_BOOL);
  auto cast_op = TransOpCreator::CreateCastOp("test_cast", input_desc, output_desc, false);
  EXPECT_NE(cast_op, nullptr);

  int srct;
  EXPECT_TRUE(AttrUtils::GetInt(cast_op, CAST_ATTR_SRCT, srct));
  EXPECT_EQ(DT_FLOAT, static_cast<DataType>(srct));

  int dstt;
  EXPECT_TRUE(AttrUtils::GetInt(cast_op, CAST_ATTR_DSTT, dstt));
  EXPECT_EQ(DT_BOOL, static_cast<DataType>(dstt));
}

TEST(UtestGraphCreateTransOp, create_squeeze) {
  vector<int64_t> dims = {1,2,3,4};
  GeShape shape(dims);
  GeTensorDesc input_desc(shape, FORMAT_NHWC, DT_FLOAT);
  GeTensorDesc output_desc(shape, FORMAT_NCHW, DT_BOOL);
  auto squeeze_op = TransOpCreator::CreateOtherTransOp("test_squeeze", SQUEEZE, input_desc, output_desc);
  EXPECT_NE(squeeze_op, nullptr);
}

TEST(UtestGraphCreateTransOp, create_reshape_with_shape_const) {
  auto compute_graph = MakeShared<ComputeGraph>("test_graph");

  vector<int64_t> dims_in = {1,2,3,4};
  vector<int64_t> dims_out = {2,3,4};
  GeTensorDesc input_desc_x(GeShape(dims_in), FORMAT_NHWC, DT_FLOAT);
  GeTensorDesc output_desc_y(GeShape(dims_out), FORMAT_NCHW, DT_FLOAT);
  output_desc_y.SetOriginShape(GeShape(dims_out));
  auto reshape_node = TransOpCreator::CreateReshapeNodeToGraph(compute_graph, "test_reshape", input_desc_x, output_desc_y);
  EXPECT_NE(reshape_node, nullptr);

  // check shape_const input of reshape node
  auto in_anchor_1 = reshape_node->GetInDataAnchor(1);
  EXPECT_NE(in_anchor_1, nullptr);
  auto shape_node = in_anchor_1->GetPeerOutAnchor()->GetOwnerNode();
  EXPECT_EQ(shape_node->GetType(), CONSTANT);
  auto shape = shape_node->GetOpDesc()->GetOutputDesc(0).GetShape();
  EXPECT_EQ(shape.GetDimNum(), 1);
  EXPECT_EQ(shape.GetDims().at(0), 3);
  // check value
  ConstGeTensorPtr weight;
  EXPECT_TRUE(ConstantUtils::GetWeight(shape_node->GetOpDesc(), 0, weight));
  int32_t *out_data = (int32_t *)weight->GetData().data();
  vector<int64_t> shape_in_const;
  for (size_t i = 0; i < dims_out.size(); ++i) {
    shape_in_const.emplace_back(out_data[i]);
  }
  EXPECT_EQ(shape_in_const, dims_out);
}

TEST(UtestGraphCreateTransOp, create_reshape_with_scaler_shape_const) {
  auto compute_graph = MakeShared<ComputeGraph>("test_graph");

  vector<int64_t> dims_in = {1};
  vector<int64_t> dims_out = {};
  GeTensorDesc input_desc_x(GeShape(dims_in), FORMAT_NHWC, DT_FLOAT);
  GeTensorDesc output_desc_y(GeShape(dims_out), FORMAT_NCHW, DT_FLOAT);
  output_desc_y.SetOriginShape(GeShape(dims_out));
  auto reshape_node = TransOpCreator::CreateReshapeNodeToGraph(compute_graph, "test_reshape", input_desc_x, output_desc_y);
  EXPECT_NE(reshape_node, nullptr);

  // check shape_const input of reshape node
  auto in_anchor_1 = reshape_node->GetInDataAnchor(1);
  EXPECT_NE(in_anchor_1, nullptr);
  auto shape_node = in_anchor_1->GetPeerOutAnchor()->GetOwnerNode();
  EXPECT_EQ(shape_node->GetType(), CONSTANT);
  auto shape = shape_node->GetOpDesc()->GetOutputDesc(0).GetShape();
  EXPECT_EQ(shape.GetShapeSize(), 0);
}

TEST(UtestGraphCreateTransOp, create_cast_with_check_failed) {
  vector<int64_t> dims = {1,2,3,4};
  GeShape shape(dims);
  GeTensorDesc input_desc(shape, FORMAT_NHWC, DT_FLOAT);
  GeTensorDesc output_desc(shape, FORMAT_NCHW, DT_BOOL);
  auto cast_op = TransOpCreator::CreateCastOp("test_cast", input_desc, output_desc);
  EXPECT_EQ(cast_op, nullptr);
}

TEST(UtestGraphCreateTransOp, test_check_support_not_found_type) {
  map<string, string> options;
  ge::GELib::Initialize(options);

  OpDescPtr type_not_found = std::make_shared<OpDesc>("type_not_found", "type_not_found");
  bool is_support = false;
  EXPECT_EQ(TransOpCreator::CheckAccuracySupported(type_not_found, is_support), GRAPH_FAILED);
  ge::GELib::GetInstance()->Finalize();
}
}  // namespace ge
