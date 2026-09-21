/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "graph/ascendc_ir/ascir_register.h"
#include "graph/types.h"
namespace ge {
namespace ascir {
EXPORT_GENERATOR()
REG_ASC_IR_START_NODE_WITH_ATTR(Data);
REG_ASC_IR_START_NODE(Constant).Attr<float>("value").Attr<ge::DataType>("dtype");
REG_ASC_IR_START_NODE(IndexExpr).Attr<int64_t>("expr");
REG_ASC_IR_START_NODE(Workspace);
REG_ASC_IR_START_NODE(TbufData);
REG_ASC_IR_1IO(Output);

REG_ASC_IR_1IO(LoadStub).UseFirstInputView();
REG_ASC_IR_1IO(Broadcast);
REG_ASC_IR_1IO(StoreStub).UseFirstInputView().Attr<int64_t>("offset");
//这里先打桩用来测试
REG_ASC_IR_1IO(WorkspaceWithInput).UseFirstInputView();

/*
 * todo nop比较特别，不确定是不是缺陷，原定义中，GEIR与ASCIR是不同的，GEIR多了个必选属性
namespace ge {
REG_OP(Nop)
    .REQUIRED_ATTR(dst_type, Int)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
.OP_END_FACTORY_REG(Nop)
}

namespace ascir::ops {
REG_OPS(Nop)
OPS_INPUT(0, x)
OPS_OUTPUT(0, y)
END_OPS(Nop)
}
 */
REG_ASC_IR_1IO(Nop);


REG_ASC_IR_1IO(AbsStub).UseFirstInputView();
REG_ASC_IR_1IO(Exp).UseFirstInputView();

REG_ASC_IR_1IO(Max);
REG_ASC_IR_1IO(Sum);

REG_ASC_IR_2I1O(Add).UseFirstInputView();
REG_ASC_IR_2I1O(Sub).UseFirstInputView();
REG_ASC_IR_2I1O(Div).UseFirstInputView();
REG_ASC_IR_2I1O(Mul).UseFirstInputView();

REG_ASC_IR_2I1O(GT).UseFirstInputView();
REG_ASC_IR_2I1O(Muls).UseFirstInputView();

// REG_ASC_IR_2I1O(MatMul)
REG_ASC_IR(MatMul)
    .Inputs({"x1", "x2"})
    .OptionalInput("x3")
    .Outputs({"y"});

REG_ASC_IR(FlashSoftmax)
    .Inputs({"x1", "x2", "x3"})
    .Outputs({"y1", "y2", "y3"});
REG_ASC_IR_2I1O(Dropout);
REG_ASC_IR_2I1O(Select);
// 适配add_layer_norm新增的api
REG_ASC_IR(CalcMean).Inputs({"x1", "x2", "x3"}).Outputs({"y1", "y2", "y3"});
REG_ASC_IR(CalcMeanSlice).Inputs({"x1", "x2", "x3"}).Outputs({"y1", "y2", "y3"});
REG_ASC_IR(CalcRstd).Inputs({"x1", "x2", "x3"}).Outputs({"y1", "y2"});
REG_ASC_IR(CalcRstdSlice).Inputs({"x1", "x2"}).Outputs({"y1", "y2"});
REG_ASC_IR(VFWelfordPart1Update)
    .Inputs({"x1", "x2", "x3"})
    .Outputs({"y1", "y2", "y3", "y4"})
    .UseFirstInputView();
REG_ASC_IR(VFWelfordPart1Finalize).Inputs({"x1", "x2"}).Outputs({"y1", "y2"});
REG_ASC_IR(VFCalcYWelford).Inputs({"x1", "x2", "x3"}).Outputs({"y1"}).UseSecondInputDataType().UseFirstInputView();
REG_ASC_IR(Concat).DynamicInput("x").Outputs({"y"});
REG_ASC_IR(VectorFunction).DynamicInput("x").DynamicOutput("y", "T").DataType("T", TensorType::ALL());
REG_ASC_IR(FakeOpA).DynamicInput("dx").OptionalInput("x2").Inputs({"x3", "x4"}).Outputs({"y"});
REG_ASC_IR(CalcY).Inputs({"x1", "x2", "x3", "x4"}).Outputs({"y1"}).UseSecondInputDataType().UseFirstInputView();
REG_ASC_IR(CalcMeanStub)
    .Inputs({"x1", "x2", "x3"})
    .Outputs({"y1", "y2", "y3", "y4"}).Attr<int64_t>("reduce_axis_dim")
    .DataTypes({PromptDtype(0U), 0U, PromptDtype(0U),
                ge::DT_DOUBLE})
    .Views({ReduceView(0U, "reduce_axis_dim"), 0U, 0U, 0U});
// 打桩测试专用op
REG_ASC_IR_WITH_COMMENT(StubOp1,
                        .Input("x", "T")
                        .Output("y", "T")
                        .DataType("T", TensorType::ALL())
                        .Attr<int64_t>("my_int")
                        .Attr<std::string>("my_string")
                        .Attr<float>("my_float")
                        .Attr<Expression>("offset")
);
/* codgen生成的类如下
namespace ge {
namespace ascir_op {
struct StubOp1 : public ge::op::StubOp1 {
  static constexpr const char *Type = "StubOp1";
  AscNodeAttr &attr;
  struct AscStubOp1IrAttrDef : public AscIrAttrDefBase {
    ~AscStubOp1IrAttrDef() override = default;
    graphStatus GetMy_int(int64_t &my_int) const {
      auto attr_value = attr_store_.GetAnyValue("my_int");
      GE_WARN_ASSERT(attr_value != nullptr);
      return attr_value->GetValue(my_int);
    }
    graphStatus SetMy_int(int64_t my_int) {
      auto attr_value = attr_store_.GetOrCreateAnyValue("my_int");
      ASCIR_ASSERT_NOTNULL(attr_value);
      return attr_value->SetValue(my_int);
    }
    graphStatus GetMy_string(std::string &my_string) const {
      auto attr_value = attr_store_.GetAnyValue("my_string");
      GE_WARN_ASSERT(attr_value != nullptr);
      return attr_value->GetValue(my_string);
    }
    graphStatus SetMy_string(std::string my_string) {
      auto attr_value = attr_store_.GetOrCreateAnyValue("my_string");
      ASCIR_ASSERT_NOTNULL(attr_value);
      return attr_value->SetValue(my_string);
    }
    graphStatus GetMy_float(float &my_float) const {
      auto attr_value = attr_store_.GetAnyValue("my_float");
      GE_WARN_ASSERT(attr_value != nullptr);
      return attr_value->GetValue(my_float);
    }
    graphStatus SetMy_float(float my_float) {
      auto attr_value = attr_store_.GetOrCreateAnyValue("my_float");
      ASCIR_ASSERT_NOTNULL(attr_value);
      return attr_value->SetValue(my_float);
    }
  };
  AscStubOp1IrAttrDef &ir_attr;
  AscOpInput<0> x;
  AscOpOutput y;
  inline StubOp1(const char *name)
      : ge::op::StubOp1(name),
        attr(AscNodeAttr::Create<AscStubOp1IrAttrDef>(*this)),
        ir_attr(dynamic_cast<AscStubOp1IrAttrDef &>(*(attr.ir_attr))),
        x(this),
        y(this, 0) {}
};
}
}
 */
REG_ASC_IR_WITH_COMMENT(StubOp2,
    .Input("x1", "T")
    .Input("x2", "T")
    .Output("y", "T")
    .DataType("T", TensorType{DT_INT32, DT_INT64})
);

REG_ASC_IR_WITH_COMMENT(StubOp2New,
                        .Input("x1", "T")
                        .Input("x2", "T")
                        .Output("y", "T")
                        .ComputeType(ge::ComputeType::kComputeElewise)
                        .Impl({"socv1"},
                                   {nullptr, nullptr,
                                   {{"T", TensorType{DT_INT32, DT_INT64}}}})
);

REG_ASC_IR(StubOp3)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Input("x3", "T1")
    .Output("y1", "T1")
    .Output("y2", "T2")
    .DataType("T1", TensorType{DT_INT32, DT_INT64})
    .DataType("T2", TensorType{DT_FLOAT16, DT_FLOAT});

REG_ASC_IR(StubOp3New)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Input("x3", "T1")
    .Output("y1", "T1")
    .Output("y2", "T2")
    .Impl({"socv1"},
          {nullptr, nullptr, {{"T1", TensorType{DT_INT32, DT_INT64}}, {"T2", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(StubOp4)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Output("y1", "T3")
    .Output("y2", "T3")
    .Output("y3", "T2")
    .DataType("T1", TensorType{DT_INT32, DT_INT64})
    .DataType("T2", TensorType{DT_FLOAT16, DT_FLOAT})
    .DataType("T3", TensorType{DT_DOUBLE, DT_BOOL});

REG_ASC_IR(StubOp4New)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Output("y1", "T3")
    .Output("y2", "T3")
    .Output("y3", "T2")
    .Impl({"socv1"},
          {nullptr,
           nullptr,
           {{"T1", TensorType{DT_INT32, DT_INT64}},
            {"T2", TensorType{DT_FLOAT16, DT_FLOAT}},
            {"T3", TensorType{DT_DOUBLE, DT_BOOL}}}});

REG_ASC_IR(StubOp5)
    .Input("x1", "T1")
    .DynamicInput("x2", "T2")
    .Output("y1", "T1")
    .Output("y2", "T2")
    .DataType("T1", TensorType{DT_INT32, DT_INT64})
    .DataType("T2", TensorType{DT_FLOAT16, DT_FLOAT});

REG_ASC_IR(StubOp5New)
    .Input("x1", "T1")
    .DynamicInput("x2", "T2")
    .Output("y1", "T1")
    .Output("y2", "T2")
    .Impl({"socv1"},
          {nullptr, nullptr, {{"T1", TensorType{DT_INT32, DT_INT64}}, {"T2", TensorType{DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(StubOp6)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Input("x3", "T1")
    .Output("y1", "T3")
    .DataType("T1", OrderedTensorTypeList{DT_INT32, DT_INT64})
    .DataType("T2", OrderedTensorTypeList{DT_FLOAT16, DT_FLOAT})
    .DataType("T3", OrderedTensorTypeList{DT_BOOL, DT_INT8});

REG_ASC_IR(StubOp6New)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Input("x3", "T1")
    .Output("y1", "T3")
    .Impl({"socv1"},
          {nullptr,
           nullptr,
           {{"T1", OrderedTensorTypeList{DT_INT32, DT_INT64}},
            {"T2", OrderedTensorTypeList{DT_FLOAT16, DT_FLOAT}},
            {"T3", OrderedTensorTypeList{DT_BOOL, DT_INT8}}}});

REG_ASC_IR(StubOp7)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Input("x3", "T1")
    .Output("y1", "T3")
    .Output("y2", "T2")
    .DataType("T1", OrderedTensorTypeList{DT_INT32, DT_INT32, DT_INT64})
    .DataType("T2", OrderedTensorTypeList{DT_FLOAT16, DT_FLOAT16, DT_FLOAT})
    .DataType("T3", OrderedTensorTypeList{DT_BOOL, DT_INT4, DT_INT8});

REG_ASC_IR(StubOp7New)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Input("x3", "T1")
    .Output("y1", "T3")
    .Output("y2", "T2")
    .Impl({"socv1"},
          {nullptr,
           nullptr,
           {{"T1", OrderedTensorTypeList{DT_INT32, DT_INT32, DT_INT64}},
            {"T2", OrderedTensorTypeList{DT_FLOAT16, DT_FLOAT16, DT_FLOAT}},
            {"T3", OrderedTensorTypeList{DT_BOOL, DT_INT4, DT_INT8}}}});

REG_ASC_IR(StubOp8)
    .Input("x", "T1")
    .Output("y", "T2")
    .DataType("T1", OrderedTensorTypeList{DT_INT32, DT_INT32, DT_INT64})
    .DataType("T2", OrderedTensorTypeList{DT_BF16, DT_BF16, DT_FLOAT});

REG_ASC_IR(StubOp8New)
    .Input("x", "T1")
    .Output("y", "T2")
    .Impl({"socv1"},
          {nullptr,
           nullptr,
           {{"T1", OrderedTensorTypeList{DT_INT32, DT_INT32, DT_INT64}}, {"T2", OrderedTensorTypeList{DT_BF16, DT_BF16, DT_FLOAT}}}});

REG_ASC_IR(StubOp9)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Input("x3", "T3")
    .Output("y1", "T2")
    .Output("y2", "T1")
    .Output("y3", "T4")
    .Output("y4", "T5")
    .DataType("T1", OrderedTensorTypeList{DT_INT32, DT_INT32, DT_INT64})
    .DataType("T2", OrderedTensorTypeList{DT_BF16, DT_BF16, DT_FLOAT})
    .DataType("T3", OrderedTensorTypeList{DT_INT8, DT_INT8, DT_FLOAT})
    .DataType("T4", OrderedTensorTypeList{DT_BOOL, DT_DOUBLE, DT_FLOAT})
    .DataType("T5", OrderedTensorTypeList{DT_BOOL, DT_COMPLEX128, DT_DUAL});

REG_ASC_IR(StubOp9New)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Input("x3", "T3")
    .Output("y1", "T2")
    .Output("y2", "T1")
    .Output("y3", "T4")
    .Output("y4", "T5")
    .Impl({"socv1"},
          {nullptr,
           nullptr,
           {{"T1", OrderedTensorTypeList{DT_INT32, DT_INT32, DT_INT64}},
            {"T2", OrderedTensorTypeList{DT_BF16, DT_BF16, DT_FLOAT}},
            {"T3", OrderedTensorTypeList{DT_INT8, DT_INT8, DT_FLOAT}},
            {"T4", OrderedTensorTypeList{DT_BOOL, DT_DOUBLE, DT_FLOAT}},
            {"T5", OrderedTensorTypeList{DT_BOOL, DT_COMPLEX128, DT_DUAL}}}});

REG_ASC_IR_1IO(StubOp10)
    .SameTmpBufSizeFromFirstInput()
    .CalcTmpBufSize("CalcTmpSizeForStubOp11");

REG_ASC_IR_1IO(StubOp11)
    .CalcTmpBufSize("CalcTmpSizeForStubOp11")
    .SameTmpBufSizeFromFirstInput();

REG_ASC_IR(StubRemovePad)
    .Input("x", "T")
    .Output("y", "T")
    .Impl({"socv1"},
          {nullptr,
           nullptr,
           {{"T", OrderedTensorTypeList{DT_INT16, DT_UINT16, DT_INT32}}}});
}  // namespace ascir
}  // namespace ge
