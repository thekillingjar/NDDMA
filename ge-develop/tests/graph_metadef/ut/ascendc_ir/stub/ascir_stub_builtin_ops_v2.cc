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
REG_ASC_IR_WITH_COMMENT(StubOp2New,
                        .Input("x1", "T")
                            .Input("x2", "T")
                            .Output("y", "T")
                            .Impl({"socv2", "socv3"},
                                  {nullptr, nullptr,
                                   {{"T", TensorType{DT_INT32, DT_INT64}}}})
);

REG_ASC_IR(StubOp3New)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Input("x3", "T1")
    .Output("y1", "T1")
    .Output("y2", "T2")
    .Impl({"socv2", "socv3"},
          {nullptr, nullptr, {{"T1", TensorType{DT_FLOAT16, DT_INT32, DT_INT64}}, {"T2", TensorType{DT_FLOAT16, DT_FLOAT16, DT_FLOAT}}}});

REG_ASC_IR(StubOp4New)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Output("y1", "T3")
    .Output("y2", "T3")
    .Output("y3", "T2")
    .Impl({"socv2", "socv3"},
          {nullptr,
           nullptr,
           {{"T1", TensorType{DT_INT32, DT_INT64, DT_UINT16}},
            {"T2", TensorType{DT_FLOAT16, DT_FLOAT, DT_UINT16}},
            {"T3", TensorType{DT_DOUBLE, DT_BOOL, DT_UINT16}}}});

REG_ASC_IR(StubOp5New)
    .Input("x1", "T1")
    .DynamicInput("x2", "T2")
    .Output("y1", "T1")
    .Output("y2", "T2")
    .Impl({"socv2", "socv3"},
          {nullptr, nullptr, {{"T1", TensorType{DT_INT32, DT_INT64,DT_UINT16}}, {"T2", TensorType{DT_FLOAT16, DT_FLOAT,DT_UINT16}}}});

REG_ASC_IR(StubOp6New)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Input("x3", "T1")
    .Output("y1", "T3")
    .Impl({"socv2", "socv3"},
          {nullptr,
           nullptr,
           {{"T1", OrderedTensorTypeList{DT_INT32, DT_INT64,DT_UINT16,DT_UINT16}},
            {"T2", OrderedTensorTypeList{DT_FLOAT16, DT_FLOAT,DT_UINT16,DT_UINT16}},
            {"T3", OrderedTensorTypeList{DT_BOOL, DT_INT8,DT_UINT16,DT_UINT64}}}});

REG_ASC_IR(StubOp7New)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Input("x3", "T1")
    .Output("y1", "T3")
    .Output("y2", "T2")
    .Impl({"socv2", "socv3"},
          {nullptr,
           nullptr,
           {{"T1", OrderedTensorTypeList{DT_INT32, DT_INT32, DT_INT64}},
            {"T2", OrderedTensorTypeList{DT_FLOAT16, DT_FLOAT16, DT_FLOAT}},
            {"T3", OrderedTensorTypeList{DT_BOOL, DT_INT4, DT_INT8}}}});

REG_ASC_IR(StubOp8New)
    .Input("x", "T1")
    .Output("y", "T2")
    .Impl({"socv2", "socv3"},
          {nullptr,
           nullptr,
           {{"T1", OrderedTensorTypeList{DT_INT32, DT_INT32, DT_INT64}}, {"T2", OrderedTensorTypeList{DT_BF16, DT_BF16, DT_FLOAT}}}});

REG_ASC_IR(StubOp9New)
    .Input("x1", "T1")
    .Input("x2", "T2")
    .Input("x3", "T3")
    .Output("y1", "T2")
    .Output("y2", "T1")
    .Output("y3", "T4")
    .Output("y4", "T5")
    .Impl({"socv2", "socv3"},
          {nullptr,
           nullptr,
           {{"T1", OrderedTensorTypeList{DT_INT32, DT_INT32, DT_INT64}},
            {"T2", OrderedTensorTypeList{DT_BF16, DT_BF16, DT_FLOAT}},
            {"T3", OrderedTensorTypeList{DT_INT8, DT_INT8, DT_FLOAT}},
            {"T4", OrderedTensorTypeList{DT_BOOL, DT_DOUBLE, DT_FLOAT}},
            {"T5", OrderedTensorTypeList{DT_BOOL, DT_COMPLEX128, DT_DUAL}}}});

}  // namespace ascir
}  // namespace ge
