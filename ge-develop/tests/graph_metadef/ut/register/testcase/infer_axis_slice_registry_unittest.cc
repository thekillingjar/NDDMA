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
#include "register/infer_axis_slice_registry.h"
#include "graph/operator_factory_impl.h"
#include "graph/operator_reg.h"

namespace ge {
REG_OP(test)
    .OP_END_FACTORY_REG(test)
} //namespace ge

class UtestInferAxisSliceRegister : public testing::Test {
protected:
  void SetUp() {}
  void TearDown() {}
};

IMPLEMT_COMMON_INFER_AXIS_SLICE(InferAxisSliceFunc) {
  return ge::GRAPH_SUCCESS;
}
IMPLEMT_COMMON_INFER_AXIS_TYPE_INFO(InferAxisTypeInfoFunc) {
  return ge::GRAPH_SUCCESS;
}

TEST_F(UtestInferAxisSliceRegister, InferAxisSliceFuncRegister_success) {
  INFER_AXIS_TYPE_INFO_REG(test, InferAxisTypeInfoFunc);
  EXPECT_NE(ge::OperatorFactoryImpl::operator_infer_axis_type_info_funcs_->find("test"),
            ge::OperatorFactoryImpl::operator_infer_axis_type_info_funcs_->end());
 
  INFER_AXIS_SLICE_FUNC_REG(test, InferAxisSliceFunc);
  EXPECT_NE(ge::OperatorFactoryImpl::operator_infer_axis_slice_funcs_->find("test"),
            ge::OperatorFactoryImpl::operator_infer_axis_slice_funcs_->end());

  ge::InferAxisTypeInfoFunc infer_axis_type_info_func = ge::OperatorFactoryImpl::GetInferAxisTypeInfoFunc("test");
  EXPECT_NE(infer_axis_type_info_func, nullptr);
  ge::InferAxisSliceFunc infer_axis_slice_func = ge::OperatorFactoryImpl::GetInferAxisSliceFunc("test");
  EXPECT_NE(infer_axis_slice_func, nullptr);
}

TEST_F(UtestInferAxisSliceRegister, AxisTypeInfo_success) {
  ge::AxisTypeInfo axis_type_info;
  ge::CutInfo output_cut_info({0U, {0}});
  axis_type_info.AddOutputCutInfo(output_cut_info);
  ge::CutInfo input_cut_info({0U, {0}});
  axis_type_info.AddInputCutInfo(input_cut_info);

  ge::CutInfo output_cut_dim;
  ge::CutInfo input_cut_dim;
  EXPECT_EQ(axis_type_info.GetInputCutInfo(0U, input_cut_dim), ge::GRAPH_SUCCESS);
  EXPECT_EQ(axis_type_info.GetOutputCutInfo(0U, output_cut_dim), ge::GRAPH_SUCCESS);
  EXPECT_EQ(input_cut_dim.first, input_cut_info.first);
  EXPECT_EQ(output_cut_dim.first, output_cut_info.first);
}
