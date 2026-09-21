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
#include <iostream>
#include "graph_builder_utils.h"
#include "graph_metadef/register/register.h"
#include <google/protobuf/message.h>
#include "proto/tensorflow/node_def.pb.h"
#include "register/op_registry.h"
#include "graph/graph.h"
#include "graph/utils/attr_utils.h"
#include "graph_metadef/graph/debug/ge_util.h"

#include "register/auto_mapping_util.h"
#include "register/scope/scope_fusion_pass_register.h"
#include "register/scope/scope_graph_impl.h"

using namespace ge;
using namespace domi;

class ConvertTensorUtest : public testing::Test {
public:
  domi::tensorflow::TensorProto tensor_;
  ge::graphStatus ret_;
  ge::GeTensorPtr weight_;

protected:
  void SetUp() {
    tensor_.set_tensor_content("tensor_context_for_test");
  }
  void TearDown() {}
};

const float FLOAT_TEST_NUM = 3.14;
const double DOUBLE_TEST_NUM = 3.1415;
const int INT_TEST_NUM = 66;
const unsigned int UNSIGNED_INT_TEST_NUM = 88;

TEST_F(ConvertTensorUtest, ConvertTensorNoType) {
  GeTensorPtr weight;
  weight.reset();
  TensorAssign::SetWeightData(domi::tensorflow::DataType_INT_MAX_SENTINEL_DO_NOT_USE_, 0, std::string("content"), weight);
  tensor_.set_dtype(domi::tensorflow::DataType_INT_MAX_SENTINEL_DO_NOT_USE_);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, GRAPH_FAILED);
  tensor_.clear_dtype();
}
TEST_F(ConvertTensorUtest, ConvertTensorFloat) {
  tensor_.add_float_val(FLOAT_TEST_NUM);
  tensor_.set_dtype(domi::tensorflow::DT_FLOAT);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
  tensor_.clear_float_val();
}
TEST_F(ConvertTensorUtest, ConvertTensorDouble) {
  tensor_.add_double_val(DOUBLE_TEST_NUM);
  tensor_.set_dtype(domi::tensorflow::DT_DOUBLE);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
  tensor_.clear_double_val();
}
TEST_F(ConvertTensorUtest, ConvertTensorComplex32_ValSize0_Success) {
  tensor_.set_dtype(domi::tensorflow::DT_COMPLEX32);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
  tensor_.clear_icomplex_val();
}
TEST_F(ConvertTensorUtest, ConvertTensorComplex32_ValSize2_DimSize2_Success) {
  domi::tensorflow::TensorShapeProto *tensor_shape = new domi::tensorflow::TensorShapeProto();
  tensor_shape->add_dim()->set_size(2);
  tensor_.set_allocated_tensor_shape(tensor_shape);
  tensor_.add_icomplex_val(INT_TEST_NUM);
  tensor_.add_icomplex_val(INT_TEST_NUM);
  tensor_.set_dtype(domi::tensorflow::DT_COMPLEX32);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
  tensor_.clear_icomplex_val();
}
TEST_F(ConvertTensorUtest, ConvertTensorComplex32_ValSize4_DimSize3_Success) {
  domi::tensorflow::TensorShapeProto *tensor_shape = new domi::tensorflow::TensorShapeProto();
  tensor_shape->add_dim()->set_size(3);
  tensor_.set_allocated_tensor_shape(tensor_shape);
  tensor_.add_icomplex_val(INT_TEST_NUM);
  tensor_.add_icomplex_val(INT_TEST_NUM);
  tensor_.add_icomplex_val(INT_TEST_NUM);
  tensor_.add_icomplex_val(INT_TEST_NUM);
  tensor_.set_dtype(domi::tensorflow::DT_COMPLEX32);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
  tensor_.clear_icomplex_val();
}
TEST_F(ConvertTensorUtest, ConvertTensorSComplex) {
  tensor_.add_scomplex_val(FLOAT_TEST_NUM);
  tensor_.add_scomplex_val(FLOAT_TEST_NUM);
  tensor_.set_dtype(domi::tensorflow::DT_COMPLEX64);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
  tensor_.clear_scomplex_val();
}
TEST_F(ConvertTensorUtest, ConvertTensorDComplex) {
  tensor_.add_dcomplex_val(DOUBLE_TEST_NUM);
  tensor_.add_dcomplex_val(DOUBLE_TEST_NUM);
  tensor_.set_dtype(domi::tensorflow::DT_COMPLEX128);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
  tensor_.clear_dcomplex_val();
}
TEST_F(ConvertTensorUtest, ConvertTensorInt) {
  tensor_.add_int_val(INT_TEST_NUM);
  tensor_.set_dtype(domi::tensorflow::DT_INT32);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
  tensor_.clear_int_val();

  tensor_.add_int64_val(INT_TEST_NUM);
  tensor_.set_dtype(domi::tensorflow::DT_INT64);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
  tensor_.clear_int64_val();

  tensor_.add_int_val(INT_TEST_NUM);
  tensor_.add_int_val(INT_TEST_NUM);
  tensor_.set_dtype(domi::tensorflow::DT_INT16);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
  tensor_.clear_int_val();
  
  tensor_.add_int_val(INT_TEST_NUM);
  tensor_.set_dtype(domi::tensorflow::DT_UINT8);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
  tensor_.clear_int_val();

}
TEST_F(ConvertTensorUtest, ConvertTensorUnsignedInt) {
  tensor_.add_uint32_val(UNSIGNED_INT_TEST_NUM);
  tensor_.set_dtype(domi::tensorflow::DT_UINT32);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
  tensor_.clear_int_val();

  tensor_.add_uint64_val(UNSIGNED_INT_TEST_NUM);
  tensor_.set_dtype(domi::tensorflow::DT_UINT64);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
  tensor_.clear_int64_val();
}
TEST_F(ConvertTensorUtest, ConvertTensorBool) {
  tensor_.add_bool_val(true);
  tensor_.set_dtype(domi::tensorflow::DT_BOOL);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
  tensor_.clear_bool_val();
}
TEST_F(ConvertTensorUtest, ConvertTensorString) {
  domi::tensorflow::TensorShapeProto *tensor_shape = new domi::tensorflow::TensorShapeProto();
  tensor_.set_allocated_tensor_shape(tensor_shape);
  tensor_.add_string_val("1");
  tensor_.set_dtype(domi::tensorflow::DT_STRING);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);

  tensor_.add_string_val("str_test2");
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
  tensor_.clear_string_val();
}
TEST_F(ConvertTensorUtest, ConvertTensorHalf) {
  tensor_.add_half_val(INT_TEST_NUM);
  tensor_.set_dtype(domi::tensorflow::DT_HALF);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
  tensor_.clear_half_val();
}
TEST_F(ConvertTensorUtest, ConvertTensorHalfVariant) {
  tensor_.add_variant_val();
  tensor_.set_dtype(domi::tensorflow::DT_VARIANT);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
  tensor_.clear_variant_val();
}

TEST_F(ConvertTensorUtest, SetWeightFloat) {
  tensor_.set_dtype(domi::tensorflow::DT_FLOAT);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
}
TEST_F(ConvertTensorUtest, SetWeightDouble) {
  tensor_.set_dtype(domi::tensorflow::DT_DOUBLE);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
}
TEST_F(ConvertTensorUtest, SetWeightSComplex) {
  tensor_.set_dtype(domi::tensorflow::DT_COMPLEX64);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
}
TEST_F(ConvertTensorUtest, SetWeightDComplex) {
  tensor_.set_dtype(domi::tensorflow::DT_COMPLEX128);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
}
TEST_F(ConvertTensorUtest, SetWeightInt) {
  tensor_.set_dtype(domi::tensorflow::DT_INT16);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);

  tensor_.set_dtype(domi::tensorflow::DT_INT32);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);

  tensor_.set_dtype(domi::tensorflow::DT_INT64);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
}
TEST_F(ConvertTensorUtest, SetWeightUnsignedInt) {
  tensor_.set_dtype(domi::tensorflow::DT_UINT8);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);

  tensor_.set_dtype(domi::tensorflow::DT_UINT32);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);

  tensor_.set_dtype(domi::tensorflow::DT_UINT64);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
}
TEST_F(ConvertTensorUtest, SetWeightBool) {
  tensor_.set_dtype(domi::tensorflow::DT_BOOL);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
}
TEST_F(ConvertTensorUtest, SetWeightString) {
  tensor_.set_dtype(domi::tensorflow::DT_STRING);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
}
TEST_F(ConvertTensorUtest, SetWeightHalf) {
  tensor_.set_dtype(domi::tensorflow::DT_HALF);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
}
TEST_F(ConvertTensorUtest, SetWeightVarient) {
  tensor_.set_dtype(domi::tensorflow::DT_VARIANT);
  ret_ = ge::AutoMappingUtil::ConvertTensor(tensor_, weight_);
  EXPECT_EQ(ret_, domi::SUCCESS);
}
TEST_F(ConvertTensorUtest, SetWeightDataString) {
  GeTensorPtr weight = std::make_shared<GeTensor>();
  const int64_t count = 2;
  std::string s[count] = {"0.941042", "0.508840"};
  std::string tensor_content;
  for (int i = 0; i < count; ++i) {
    tensor_content.push_back(static_cast<uint8_t>(s[i].size()));
  }
  for (int i = 0; i < count; ++i) {
    tensor_content += s[i];
  }
  TensorAssign::SetWeightData(domi::tensorflow::DT_STRING, count, tensor_content, weight);
  uint64_t tensor_data_size = 0;
  for (int i = 0; i < count; ++i) {
    tensor_data_size += sizeof(StringHead) + s[i].size() + 1;
  }
  EXPECT_EQ(weight->GetData().GetSize(), tensor_data_size);
  std::string tensor_data[count];
  int start_index = count * sizeof(StringHead);
  for (int i = 0; i < count; ++i) {
    for (size_t j = 0; j < s[i].size(); ++j) {
      tensor_data[i] += weight->GetData().data()[start_index + j];
    }
    start_index += s[i].size() + 1;
    EXPECT_EQ(tensor_data[i], s[i]);
  }
}
TEST_F(ConvertTensorUtest, SetWeightDataStringOverflow) {
  GeTensorPtr weight = std::make_shared<GeTensor>();
  std::string tensor_content = "\b\b0.9410420.508840";
  const int64_t count = std::numeric_limits<uint64_t>::max() / sizeof(StringHead) + 1;
  TensorAssign::SetWeightData(domi::tensorflow::DT_STRING, count, tensor_content, weight);
  EXPECT_EQ(weight->GetData().size(), 0);
}

TEST_F(ConvertTensorUtest, GetStringVal) {
  Status retStatus;
  ge::GeTensorPtr weight = ComGraphMakeShared<ge::GeTensor>();
  google::protobuf::RepeatedPtrField<std::string> vector;

  vector.Add()->assign("vector");
  EXPECT_FALSE(vector.empty());
  EXPECT_EQ(vector.size(), 1);

  retStatus = TensorAssign::GetStringVal(1, vector, 2, weight);
  EXPECT_EQ(retStatus, domi::SUCCESS);
}
