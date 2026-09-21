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
#include "register/infer_data_slice_registry.h"
#include "graph/operator_factory_impl.h"
namespace ge {
  class UtestInferDataSliceRegister : public testing::Test {
  protected:
    void SetUp() {}
    void TearDown() {}
  };

TEST_F(UtestInferDataSliceRegister, InferDataSliceFuncRegister_success) {
  InferDataSliceFunc infer_data_slice_func;
  ge::InferDataSliceFuncRegister("test", infer_data_slice_func);
  EXPECT_NE(OperatorFactoryImpl::operator_infer_data_slice_funcs_->find("test"), OperatorFactoryImpl::operator_infer_data_slice_funcs_->end());
}
}
