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

#include "macro_utils/dt_public_scope.h"
#include "graph/operator.h"
#include "graph/def_types.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/graph.h"
#include "graph/operator_factory_impl.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "framework/common/op/ge_op_utils.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace ge;

class UtestGeOperator : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestGeOperator, try_get_input_desc) {
  Operator data("data0");

  TensorDesc td;
  graphStatus ret = data.TryGetInputDesc("const", td);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGeOperator, get_dynamic_input_num) {
  Operator const_node("constNode");

  (void)const_node.DynamicInputRegister("data", 2, 1);
  int num = const_node.GetDynamicInputNum("data");
  EXPECT_EQ(num, 2);
}

TEST_F(UtestGeOperator, infer_format_func_register) {
  Operator add("add");
  std::function<graphStatus(Operator &)> func = nullptr;
  EXPECT_NO_THROW(add.InferFormatFuncRegister(func));
}

graphStatus TestFunc(Operator &op) { return 0; }
TEST_F(UtestGeOperator, get_infer_format_func_register) {
  EXPECT_NO_THROW(
    (void)OperatorFactoryImpl::GetInferFormatFunc("add");
    OperatorFactoryImpl::RegisterInferFormatFunc("add", TestFunc);
    (void)OperatorFactoryImpl::GetInferFormatFunc("add");
  );
}

TEST_F(UtestGeOperator, get_attr_names_and_types) {
  Operator attr("attr");
  EXPECT_NO_THROW((void)attr.GetAllAttrNamesAndTypes());
}
