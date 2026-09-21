/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "common/checker.h"
using namespace gert;
namespace ops {
ge::graphStatus InferShapeForAdd2(InferShapeContext *context) {
  GELOGD("InferShapeForAdd2");
  auto input_shape_0 = *context->GetInputShape(0);
  auto input_shape_1 = *context->GetInputShape(1);
  auto output_shape = context->GetOutputShape(0);
  if (input_shape_0.GetDimNum() != input_shape_1.GetDimNum()) {
    GELOGE(ge::PARAM_INVALID, "Add param invalid, input_shape_0.GetDimNum() is %zu,  input_shape_1.GetDimNum() is %zu",
           input_shape_0.GetDimNum(), input_shape_1.GetDimNum());
  }
  if (input_shape_0.GetDimNum() == 0) {
    *output_shape = input_shape_1;
    return ge::GRAPH_SUCCESS;
  }
  if (input_shape_1.GetDimNum() == 0) {
    *output_shape = input_shape_0;
    return ge::GRAPH_SUCCESS;
  }
  output_shape->SetDimNum(input_shape_0.GetDimNum());
  for (size_t i = 0; i < input_shape_0.GetDimNum(); ++i) {
    output_shape->SetDim(i, std::max(input_shape_0.GetDim(i), input_shape_1.GetDim(i)));
  }

  return ge::GRAPH_SUCCESS;
}

struct AddCompileInfo {
  int64_t a;
  int64_t b;
};
ge::graphStatus TilingForAdd2(TilingContext *context) {
  auto ci = context->GetCompileInfo<AddCompileInfo>();
  GE_ASSERT_NOTNULL(ci);
  auto tiling_data = context->GetRawTilingData();
  GE_ASSERT_NOTNULL(tiling_data);
  tiling_data->Append(*ci);
  tiling_data->Append(ci->a);
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus TilingParseForAdd2(TilingParseContext *context) {
  auto compile_info = context->GetCompiledInfo<AddCompileInfo>();
  compile_info->a = 10;
  compile_info->b = 200;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(Add).InferShape(InferShapeForAdd2).Tiling(TilingForAdd2).TilingParse<AddCompileInfo>(TilingParseForAdd2);

IMPL_OP(MatMul);
IMPL_OP(MatMulV2);
IMPL_OP(MatMulV3);
IMPL_OP(AddLayerNorm);
IMPL_OP(AddN);
IMPL_OP(ApplyAdagradD);
IMPL_OP(ApplyAdamD);
IMPL_OP(BNTrainingReduce);
IMPL_OP(BNTrainingUpdate);
IMPL_OP(BNTrainingUpdateV3);
IMPL_OP(Add);
IMPL_OP(Pow);
IMPL_OP(AddV2);
IMPL_OP(Mul);
IMPL_OP(Less);
IMPL_OP(Sub);
IMPL_OP(RealDiv);
IMPL_OP(Equal);
IMPL_OP(NotEqual);
IMPL_OP(Greater);
IMPL_OP(Maximum);
IMPL_OP(Minimum);
IMPL_OP(LogicalAnd);
IMPL_OP(LogicalOr);
IMPL_OP(Div);
IMPL_OP(LessEqual);
IMPL_OP(SquaredDifference);
IMPL_OP(GreaterEqual);
IMPL_OP(DivNoNan);
IMPL_OP(LeakyReluGrad);
IMPL_OP(ReluGrad);
IMPL_OP(ConfusionSoftmaxGrad);
IMPL_OP(BitwiseAnd);
IMPL_OP(FloorDiv);
IMPL_OP(FloorMod);
IMPL_OP(EluGrad);
IMPL_OP(SoftmaxGrad);
IMPL_OP(TanhGrad);
IMPL_OP(Axpy);
IMPL_OP(BroadcastTo);
IMPL_OP(Conv2DBackpropInputD);
IMPL_OP(Conv2DBackpropFilterD);
IMPL_OP(DiagPartD);
IMPL_OP(Abs);
IMPL_OP(SoftmaxV2);
IMPL_OP(Relu);
IMPL_OP(Elu);
IMPL_OP(HcomAllReduce);
IMPL_OP(SigmoidGrad);
IMPL_OP(IsNan);
IMPL_OP(Sign);
IMPL_OP(Exp);
IMPL_OP(Cast);
IMPL_OP(LogicalNot);
IMPL_OP(Neg);
IMPL_OP(Sqrt);
IMPL_OP(Rsqrt);
IMPL_OP(ZerosLike);
IMPL_OP(Sigmoid);
IMPL_OP(BiasAdd);
IMPL_OP(Square);
IMPL_OP(Erf);
IMPL_OP(Reciprocal);
IMPL_OP(LeakyRelu);
IMPL_OP(Tanh);
IMPL_OP(RsqrtGrad);
IMPL_OP(Muls);
IMPL_OP(Identity);
IMPL_OP(TensorMove);
IMPL_OP(Log);
IMPL_OP(Log1p);
IMPL_OP(SoftmaxGradExt);
IMPL_OP(IsFinite);
IMPL_OP(StopGradient);
IMPL_OP(ApplyGradientDescent);
IMPL_OP(AssignAdd);
IMPL_OP(BNTrainingReduceGrad);
IMPL_OP(Gelu);
IMPL_OP(FusedMulAddN);
IMPL_OP(Softplus);
IMPL_OP(BNInferenceD);
IMPL_OP(ExpandDims);
IMPL_OP(Fill);
IMPL_OP(L2Loss);
IMPL_OP(LayerNorm);
IMPL_OP(SoftmaxCrossEntropyWithLogits);
IMPL_OP(LayerNormXBackpropV3);
IMPL_OP(LayerNormXBackpropV2);
IMPL_OP(MultisliceConcat);
IMPL_OP(PadV3);
IMPL_OP(Range);
IMPL_OP(BNTrainingUpdateGrad);
IMPL_OP(ReduceSum);
IMPL_OP(ReduceMax);
IMPL_OP(ReduceMin);
IMPL_OP(ReduceProd);
IMPL_OP(ReduceMean);
IMPL_OP(ReduceAll);
IMPL_OP(ReduceAny);
IMPL_OP(ReduceSumD);
IMPL_OP(ReduceMaxD);
IMPL_OP(ReduceMinD);
IMPL_OP(ReduceProdD);
IMPL_OP(ReduceMeanD);
IMPL_OP(ReduceAllD);
IMPL_OP(ReduceAnyD);
IMPL_OP(BiasAddGrad);
IMPL_OP(LayerNormBetaGammaBackpropV2);
IMPL_OP(LayerNormV3);
IMPL_OP(Reshape);
IMPL_OP(Select);
IMPL_OP(SelectV2);
IMPL_OP(Slice);
IMPL_OP(SliceD);
IMPL_OP(Split);
IMPL_OP(SplitV);
IMPL_OP(SplitD);
IMPL_OP(SplitVD);
IMPL_OP(SquareSumV1);
IMPL_OP(Squeeze);
IMPL_OP(StridedSliceD);
IMPL_OP(StridedSlice);
IMPL_OP(Tile);
IMPL_OP(TileD);
IMPL_OP(Transpose);
IMPL_OP(TransposeD);
IMPL_OP(Unpack);
IMPL_OP(UnsortedSegmentMax);
IMPL_OP(UnsortedSegmentMin);
IMPL_OP(Unsqueeze);

IMPL_OP(FooTestGuard);
IMPL_OP(FooTest);
IMPL_OP(Repeat);
IMPL_OP(Const);

}  // namespace ops
