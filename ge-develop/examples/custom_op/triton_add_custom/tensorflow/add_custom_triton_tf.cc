/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

// 注册TensorFlow自定义算子
REGISTER_OP("AddCustom")                                    // TensorFlow自定义算子名称
    .Input("x: T")                                          // 输入tensor x
    .Input("y: T")                                          // 输入tensor y
    .Output("z: T")                                         // 输出tensor z
    .Attr("T: {float}")                                     // 属性T，支持float数据类型
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn); // 设置shape函数，用于推断输出tensor的shape，BroadcastBinaryOpShapeFn函数用于处理输入、输出tensor的shape相同的情况


// TensorFlow自定义算子的CPU实现
class AddCustomOp : public OpKernel {
public:
    explicit AddCustomOp(OpKernelConstruction* context) : OpKernel(context) {}
    // 当前算子不支持CPU设备，实现该函数以抛出异常，提示该算子不支持CPU设备
    void Compute(OpKernelContext* context) override {
        OP_REQUIRES(context, false, errors::Unimplemented("AddCustomOp is not supported on CPU"));
    }
};

// 注册TensorFlow自定义算子的CPU实现
REGISTER_KERNEL_BUILDER(Name("AddCustom").Device(DEVICE_CPU), AddCustomOp);