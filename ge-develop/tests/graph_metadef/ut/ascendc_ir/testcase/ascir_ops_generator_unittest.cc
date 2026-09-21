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

#include "graph/ascendc_ir/ascir_register.h"
#include "graph/types.h"
namespace ge {
namespace ascir {
REG_ASC_IR_1IO(StubData).StartNode();
REG_ASC_IR_START_NODE(StubConstant).Attr<float>("value");
REG_ASC_IR_1IO(StubOutput);

REG_ASC_IR_1IO(StubLoad).UseFirstInputDataType().UseFirstInputView();
REG_ASC_IR_1IO(StubStore).UseFirstInputDataType().UseFirstInputView();

REG_ASC_IR_1IO(StubCast)
    .Attr<ge::DataType>("dst_type").Attr<std::string>("stub_attr")
    .InferDataType([](const AscIrDef &def, std::stringstream &ss) {
      ss << "  op.y.dtype = dst_type;" << std::endl;
    });

REG_ASC_IR_2I1O(StubAdd).UseFirstInputDataType().UseFirstInputView();
REG_ASC_IR(StubFlashSoftmax)
    .Inputs({"x1", "x2", "x3"})
    .Outputs({"y1", "y2", "y3"})
    .UseFirstInputDataType();

REG_ASC_IR(StubConcat).DynamicInput("x").Outputs({"y"}).UseFirstInputDataType();

REG_ASC_IR(StubVectorFunction).DynamicInput("x").DynamicOutput({"y"});

REG_ASC_IR_1IO(StubTilingData).ApiTilingDataType("StubTilingData").ApiTilingDataType("TilingData");

}  // namespace ascir
namespace ascir {
void GenHeaderFileToStream(const char *, std::stringstream &ss);
class GeneratorUT : public testing::Test {};
TEST(GeneratorUT, Gnerate_Ops_Ok) {
  EXPECT_NO_THROW(
    std::stringstream ss;
    GenHeaderFileToStream("/path/to/hello.h", ss);
    std::cout << "===================:" << std::endl;
    std::cout << ss.str() << std::endl;
    std::cout << "===================:" << std::endl;
  );
}
}  // namespace ascir
}
