/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MAIN_OPS_STUB_H
#define MAIN_OPS_STUB_H

#include "graph/operator_reg.h"
#include "graph/ge_attr_value.h"

// for ir
namespace ge {
const static std::vector<int64_t> val_list_int;
const static std::vector<bool> val_list_bool;
const static std::vector<float> val_list_float;
const static std::vector<std::string> val_list_str;
const static std::vector<DataType> val_list_dt;
const static NamedAttrs var_name_attr;

// Data
REG_OP(Data)
    .INPUT(data, TensorType::ALL())
    .OUTPUT(out, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data)

    // Softmax
    REG_OP(Softmax)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(axis, Int, 0)  // which mean compute which dims
    .ATTR(algo, Int, 1)  // 1 means using "subtract max from every point to avoid overflow",
    // 0 means using "ubtract max from every point to avoid overflow"
    // 2 means using "perform the Log softmax operation to avoid overflow"
    // now is only support 1
    .ATTR(alpha, Float, 1)
    .ATTR(beta, Float, 0)
    .OP_END_FACTORY_REG(Softmax)

    // Flatten
    REG_OP(Flatten)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(Flatten)

        REG_OP(Square)
    .INPUT(x, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(alpha, Float, 1.0)
    .ATTR(beta, Float, 0.0)
    .OP_END_FACTORY_REG(Square)

        REG_OP(ReadVariable)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(ReadVariable)

        REG_OP(Activation)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    // 0:sigmod, 1:relu, 2:tanh, 3:clipped ReLU, 4:Elu,
    // 5:leaky relu, 6:abs, 7:relu1, 8:softsign, 9:softplus
    .ATTR(mode, Int, 1)
    .ATTR(coef, Float, 0)
    .ATTR(alpha, Float, 1.0)
    .ATTR(beta, Float, 0)
    .OP_END_FACTORY_REG(Activation)

        REG_OP(Add)
    .INPUT(x1, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16}))  // "First operand."
    .INPUT(x2, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16}))  // "Second operand."
    // "Result, has same element type as two inputs"
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16}))
    .OP_END_FACTORY_REG(Add)

        REG_OP(Variable)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(index, Int, 0)
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Variable)

        REG_OP(Summary)
    .INPUT(x, TensorType::ALL())
    .OP_END_FACTORY_REG(Summary)

        REG_OP(Const)
    .OUTPUT(y, TensorType::ALL())   // TensorType({DT_FLOAT, DT_INT8, DT_INT32, DT_BOOL})
    .ATTR(value, Tensor, Tensor())  // This is the value of the const op
    .ATTR(dtype, Int, 0)
    .OP_END_FACTORY_REG(Const)

        REG_OP(HcomBroadcast)
    .DYNAMIC_INPUT(x, TensorType::ALL())
    .DYNAMIC_OUTPUT(y, TensorType::ALL())
    .REQUIRED_ATTR(root_rank, Int)
    .REQUIRED_ATTR(group, String)
    .ATTR(alpha, Float, 1.0)
    .ATTR(beta, Float, 0.0)
    .OP_END_FACTORY_REG(HcomBroadcast)

        REG_OP(Assign)
    .INPUT(resource, TensorType::ALL())
    .INPUT(value, TensorType::ALL())
    .OUTPUT(resource, TensorType::ALL())
    .OP_END_FACTORY_REG(Assign) REG_OP(Sqrt)
    .INPUT(x, TensorType{(DT_FLOAT.DT_FLOAT16)})
    .OUTPUT(y, TensorType{(DT_FLOAT, DT_FLOAT16)})
    .ATTR(T, Int, 1)
    .ATTR(alpha, Float, 1.0)
    .ATTR(beta, Float, 0.0)
    .OP_END_FACTORY_REG(Sqrt)

        REG_OP(Save)
    .DYNAMIC_INPUT(tensors, TensorType
                   : ALL())
    .OP_END_FACTORY_REG(Save)

        REG_OP(PReLU)
    .INPUT(x, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(channel_shared, Bool, false)
    .ATTR(nan_opt, Int, 0)
    .ATTR(alpha, Float, 1.0)
    .ATTR(beta, Float, 0.0)
    .OP_END_FACTORY_REG(PReLU) REG_OP(Acosh)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(Acosh)

        REG_OP(GuaranteeConst)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                          DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                           DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(GuaranteeConst)

    REG_OP(MatMulV2)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(MatMulV2)

REG_OP(TestAllAttr)
    .INPUT(data, TensorType::ALL())
    .OPTIONAL_INPUT(option_input, TensorType::ALL())
    .OUTPUT(out, TensorType::ALL())
    .ATTR(test_str, String, "")
    .ATTR(test_int, Int, 0)
    .ATTR(test_bool, Bool, false)
    .ATTR(test_float, Float, 0.0)
    .ATTR(test_dt, Type, DT_FLOAT)
    .ATTR(test_list_string, ListString, val_list_str)
    .ATTR(test_list_int, ListInt, val_list_int)
    .ATTR(test_list_bool, ListBool, val_list_bool)
    .ATTR(test_list_float, ListFloat, val_list_float)
    .ATTR(test_list_dt, ListType, val_list_dt)
    .ATTR(test_name_attr, NamedAttrs, var_name_attr)
    .OP_END_FACTORY_REG(TestAllAttr)

        IMPLEMT_INFERFUNC(GuaranteeConst, GuaranteeConstInfer) {
  TensorDesc tensorDesc = op.GetInputDesc("x");
  (void)op.UpdateOutputDesc("y", tensorDesc);
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(GuaranteeConst, GuaranteeConstInfer);
}  // namespace ge
#endif  // MAIN_OPS_STUB_H
