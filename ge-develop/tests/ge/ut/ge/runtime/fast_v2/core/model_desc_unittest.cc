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
#include "macro_utils/dt_public_scope.h"
#include "runtime/model_desc.h"
#include "macro_utils/dt_public_unscope.h"

namespace gert {
class ModelDescUT : public testing::Test {};

TEST_F(ModelDescUT, ShapeRangeConstruct_Ok_MinMaxShape) {
  Shape min_shape{1,2};
  Shape max_shape{2,2};
  ShapeRange sr{min_shape, max_shape};
  EXPECT_EQ(sr.GetMin(), min_shape);
  EXPECT_EQ(sr.GetMax(), max_shape);
}
TEST_F(ModelDescUT, ShapeRangeConstruct_Ok_Default) {
  ShapeRange sr;
  EXPECT_EQ(sr.GetMin().GetDimNum(), 0);
  EXPECT_EQ(sr.GetMax().GetDimNum(), 0);
}
TEST_F(ModelDescUT, ShapeRangeEqual_Success) {
  ShapeRange sr1{{1,2}, {2,2}};
  ShapeRange sr2{{1,2}, {2,2}};
  ShapeRange sr3{{2,2}, {2,2}};
  EXPECT_TRUE(sr1 == sr2);
  EXPECT_FALSE(sr1 == sr3);
}

TEST_F(ModelDescUT, ModelDescGet_Success) {
  size_t total_size = sizeof(ModelDesc) + 2 * sizeof(ModelIoDesc);
  auto holder = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[total_size]);
  auto &model_desc = *(reinterpret_cast<ModelDesc *>(holder.get()));
  model_desc.input_num_ = 1;
  model_desc.output_num_ = 1;
  model_desc.model_io_descs_.Init(2 * sizeof(ModelIoDesc));
  model_desc.model_io_descs_.SetSize(2 * sizeof(ModelIoDesc));

  std::vector<std::vector<int64_t>> batch_info;
  int32_t dynamic_type = 1;
  ASSERT_EQ(model_desc.GetDynamicBatchInfo(batch_info, dynamic_type), ge::GRAPH_SUCCESS);
  ASSERT_EQ(dynamic_type, -1);
  std::vector<std::string> user_designate_shape_order;
  ASSERT_EQ(model_desc.GetUserDesignateShapeOrder(user_designate_shape_order), ge::GRAPH_SUCCESS);
  std::vector<std::string> attrs;
  ASSERT_EQ(model_desc.GetModelAttrs(attrs), ge::GRAPH_SUCCESS);

  ModelIoDesc &input_desc = *(reinterpret_cast<ModelIoDesc *>(model_desc.model_io_descs_.MutableData()));
  input_desc.name_ = "data";
  input_desc.data_type_ = 1;
  input_desc.format_.origin_format_= static_cast<ge::Format>(1);

  Shape in_shape;
  in_shape.dim_num_ = 2;
  in_shape.dims_[0] = -1;
  in_shape.dims_[1] = 2;
  input_desc.shape_.origin_shape_ = in_shape;
  input_desc.shape_.storage_shape_ = in_shape;

  ShapeRange in_shape_range;
  in_shape_range.max_ = in_shape;
  in_shape_range.min_ = in_shape;
  in_shape_range.min_.dims_[0] = 0;
  input_desc.origin_shape_range_ = in_shape_range;
  input_desc.storage_shape_range_ = in_shape_range;

  ASSERT_NE(model_desc.GetInputDesc(0), nullptr);
  ASSERT_EQ(model_desc.MutableInputDesc(0), model_desc.GetInputDesc(0));
  ASSERT_EQ(model_desc.GetInputDesc(0)->GetName(), "data");
  ASSERT_EQ(model_desc.GetInputDesc(0)->GetDataType(), 1);
  ASSERT_EQ(model_desc.GetInputDesc(0)->GetOriginFormat(), static_cast<ge::Format>(1));
  ASSERT_EQ(model_desc.GetInputDesc(0)->GetSize(), 0);
  ASSERT_EQ(model_desc.GetInputDesc(0)->GetOriginShape(), in_shape);
  ASSERT_EQ(model_desc.GetInputDesc(0)->GetOriginShapeRange().GetMin(), in_shape_range.min_);
  ASSERT_EQ(model_desc.GetInputDesc(0)->GetOriginShapeRange().GetMax(), in_shape_range.max_);
  ASSERT_EQ(model_desc.GetInputDesc(0)->GetStorageShapeRange().GetMin(), in_shape_range.min_);
  ASSERT_EQ(model_desc.GetInputDesc(0)->GetStorageShapeRange().GetMax(), in_shape_range.max_);
  std::vector<std::pair<int64_t, int64_t>> range;
  for (size_t i = 0U; i < in_shape_range.GetMax().GetDimNum(); ++i) {
    range.emplace_back(std::make_pair(in_shape_range.GetMin().GetDim(i), in_shape_range.GetMax().GetDim(i)));
  }
  ASSERT_EQ(model_desc.GetInputDesc(0)->GetOriginShapeRangeVector(), range);
  ASSERT_EQ(model_desc.GetInputDesc(0)->GetStorageShapeRangeVector(), range);

  input_desc.aipp_shape_ = in_shape;
  ASSERT_EQ(model_desc.GetInputDesc(0)->GetAippShape(), in_shape);
  model_desc.MutableInputDesc(0)->MutableAippShape().SetDim(1, 3);
  Shape tmp_shape({-1, 3});
  ASSERT_EQ(model_desc.GetInputDesc(0)->GetAippShape(), tmp_shape);

  input_desc.shape_.storage_shape_ = in_shape;
  input_desc.shape_.storage_shape_.dims_[0] = 1;
  ASSERT_EQ(model_desc.GetInputDesc(0)->GetSize(), 4);
  input_desc.shape_.storage_shape_.dims_[0] = std::numeric_limits<int64_t>::max();
  input_desc.shape_.storage_shape_.dims_[1] = 1;
  ASSERT_EQ(model_desc.GetInputDesc(0)->GetSize(), 0);
  size_t input_num = 0;
  const ModelIoDesc *inputs = model_desc.GetAllInputsDesc(input_num);
  ASSERT_EQ(inputs, &input_desc);
  ASSERT_EQ(input_num, 1);

  ModelIoDesc &output_desc = *(reinterpret_cast<ModelIoDesc *>(model_desc.model_io_descs_.MutableData()) + 1);
  output_desc.name_ = "netoutput";
  output_desc.data_type_ = 1;
  output_desc.format_.origin_format_= static_cast<ge::Format>(1);

  Shape out_shape;
  out_shape.dim_num_ = 3;
  out_shape.dims_[0] = -1;
  out_shape.dims_[1] = 4;
  out_shape.dims_[2] = 5;
  output_desc.shape_.origin_shape_ = out_shape;
  output_desc.shape_.storage_shape_ = out_shape;
  ShapeRange out_shape_range;
  out_shape_range.max_ = out_shape;
  out_shape_range.min_ = out_shape;
  out_shape_range.min_.dims_[0] = 0;
  output_desc.origin_shape_range_ = out_shape_range;
  ASSERT_NE(model_desc.GetOutputDesc(0), nullptr);
  ASSERT_EQ(model_desc.GetOutputDesc(0), model_desc.MutableOutputDesc(0));
  ASSERT_EQ(model_desc.GetOutputDesc(0)->GetName(), "netoutput");
  ASSERT_EQ(model_desc.GetOutputDesc(0)->GetDataType(), 1);
  ASSERT_EQ(model_desc.GetOutputDesc(0)->GetOriginFormat(), static_cast<ge::Format>(1));
  ASSERT_EQ(model_desc.GetOutputDesc(0)->GetSize(), 0);
  ASSERT_EQ(model_desc.GetOutputDesc(0)->GetOriginShape(), out_shape);
  ASSERT_EQ(model_desc.GetOutputDesc(0)->GetOriginShapeRange().GetMin(), out_shape_range.min_);
  ASSERT_EQ(model_desc.GetOutputDesc(0)->GetOriginShapeRange().GetMax(), out_shape_range.max_);
  size_t output_num = 0;
  const ModelIoDesc *outputs = model_desc.GetAllOutputsDesc(output_num);
  ASSERT_EQ(outputs, &output_desc);
  ASSERT_EQ(output_num, 1);
}

TEST_F(ModelDescUT, ModelDesc_GetTheFileConstantWeightDirSetOk) {
  size_t total_size = sizeof(ModelDesc) + 1 * sizeof(ModelIoDesc);
  auto holder = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[total_size]);
  auto &model_desc = *(reinterpret_cast<ModelDesc *>(holder.get()));

  std::string weight_dir("/home/bob/model/weioght/testxxx.bin");
  model_desc.SetFileConstantWeightDir(weight_dir.c_str());
  EXPECT_EQ(std::string(model_desc.GetFileConstantWeightDir()), weight_dir);
}

}  // namespace gert