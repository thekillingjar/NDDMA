/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "graph/debug/ge_op_types.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/debug/ge_attr_define.h"
#include <operator_factory_impl.h>
#include "graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"

using namespace std;

namespace ge {

namespace {
static void RegisterOpCreatorV2(const std::string &op_type,
                                const std::vector<std::string> &input_names,
                                IrInputType input_type,
                                const std::vector<std::string> &output_names,
                                IrOutputType output_type,
                                const std::vector<std::string> &tag_names) {
  ge::OpCreatorV2 op_creator_v2 = [op_type, input_names, input_type, output_names, output_type,
      tag_names](const ge::AscendString &name) -> ge::Operator {
    auto op_desc = std::make_shared<ge::OpDesc>(name.GetString(), op_type);
    for (const auto &tensor_name : input_names) {
      op_desc->AppendIrInput(tensor_name, input_type);
    }
    for (const auto &tensor_name : output_names) {
      op_desc->AppendIrOutput(tensor_name, output_type);
    }
    for (const auto &tag_name : tag_names) {
      op_desc->AppendIrAttrName(tag_name);
    }
    return ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  };
  (void)ge::OperatorFactoryImpl::RegisterOperatorCreator(op_type, op_creator_v2);
}

static void RegisterVecInputTypeOpCreatorV2(const std::string &op_type, const std::vector<std::string> &input_names,
                                            vector<IrInputType> input_types,
                                            const std::vector<std::string> &output_names, IrOutputType output_type,
                                            const std::vector<std::string> &tag_names) {
  ge::OpCreatorV2 op_creator_v2 = [op_type, input_names, input_types, output_names, output_type,
                                   tag_names](const ge::AscendString &name) -> ge::Operator {
    auto op_desc = std::make_shared<ge::OpDesc>(name.GetString(), op_type);
    for (size_t i = 0; i < input_names.size(); i++) {
      op_desc->AppendIrInput(input_names[i], input_types[i]);
    }
    for (const auto &tensor_name : output_names) {
      op_desc->AppendIrOutput(tensor_name, output_type);
    }
    for (const auto &tag_name : tag_names) {
      op_desc->AppendIrAttrName(tag_name);
    }
    return ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  };
  (void)ge::OperatorFactoryImpl::RegisterOperatorCreator(op_type, op_creator_v2);
}

void RegisterAllOpCreator() {
  RegisterOpCreatorV2("ZeroLikeStub", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("AbsRepeat5", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("DynamicQuantStub", {"x"}, ge::kIrInputRequired, {"y1", "y2"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("DynamicQuant", {"x"}, ge::kIrInputRequired, {"y1", "y2"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("ReduceSum", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});

  RegisterOpCreatorV2("Abs", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Add", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("AddV2", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("AddN", {"x"}, ge::kIrInputDynamic, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Data", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Exp", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("ReduceSumD", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Relu", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Maximum", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("SquaredDifference", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("SquaredDifferenceStub", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("ConcatD", {"x"}, ge::kIrInputDynamic, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Concat", {"x"}, ge::kIrInputDynamic, {"y"}, kIrOutputRequired, {});

  RegisterOpCreatorV2("IsNan", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Cast", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("LogicalAnd", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Unsqueeze", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Squeeze", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("SoftmaxV2", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("BatchMatMulV2", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("SoftmaxStub", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Slice", {"x1", "x2", "x3"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("SliceD", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("StridedSliceD", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("StridedSlice", {"x1", "x2", "x3", "x4"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("SplitD", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputDynamic, {});
  RegisterOpCreatorV2("SplitV", {"x","size_splits", "split_dim"}, ge::kIrInputRequired, {"y"}, kIrOutputDynamic, {});
  RegisterOpCreatorV2("LeakyRelu", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Fill", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("GatherV2", {"x1", "x2", "x3"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Gather", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Transpose", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("SelectV2", {"x1", "x2", "x3"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Elu", {"x1"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("EluGrad", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("TanhGrad", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Softplus", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("BiasAdd", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("BiasGrad", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Tile", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("TileD", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Square", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("SquareSumV1", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("L2Loss", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("FloorMod", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("FusedMulAddN", {"x1", "x2", "x3"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("BNTrainingUpdate", {"x1", "x2", "x3", "x4", "x5", "x6", "x7"}, ge::kIrInputRequired,
                      {"y1", "y2", "y3", "y4", "y5"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("BNTrainingReduce", {"x"}, ge::kIrInputRequired, {"y1", "y2"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("ApplyGradientDescent", {"x1", "x2", "x3"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("ApplyAdagrad", {"x1", "x2", "x3", "x4"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("ApplyAdagradD", {"x1", "x2", "x3", "x4"}, ge::kIrInputRequired, {"y1", "y2"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("ApplyAdamD", {"x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"},
                      ge::kIrInputRequired, {"y1", "y2", "y3"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Select", {"x1", "x2", "x3"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("ExpandDims", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Pack", {"x"}, ge::kIrInputDynamic, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("LeakyReluGrad", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("DivNoNan", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("SigmoidGrad", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Muls", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("RsqrtGrad", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("ReluGrad", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("LayerNormBetaGammaBackpropV2", {"x1", "x2"}, ge::kIrInputRequired, {"y1", "y2"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("ConfusionSoftmaxGrad", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("BroadcastTo", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Reshape", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("ClipByValue", {"x1", "x2", "x3"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("ZerosLike", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("BiasAddGrad", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Log", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Log1p", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("BitwiseAnd", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("IsFinite", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("LogicalNot", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Neg", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Reciprocal", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Rsqrt", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Erf", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Sigmoid", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Sign", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Sqrt", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Tanh", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Gelu", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Div", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Equal", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("GreaterEqual", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Greater", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("LessEqual", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("LogicalOr", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Less", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Minimum", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Mul", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("NotEqual", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Pow", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Sub", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("RealDiv", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("FloorDiv", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("ReduceMaxD", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("ReduceMeanD", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("ReduceMinD", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("ReduceProdD", {"x"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterOpCreatorV2("Axpy", {"x1", "x2"}, ge::kIrInputRequired, {"y"}, kIrOutputRequired, {});
  RegisterVecInputTypeOpCreatorV2("MatMul", {"x1", "x2", "bias"},
                                  {ge::kIrInputRequired, ge::kIrInputRequired, ge::kIrInputOptional}, {"y"},
                                  kIrOutputRequired, {"transpose_x1", "transpose_x2"});
  RegisterVecInputTypeOpCreatorV2(
      "MatMulV2", {"x1", "x2", "bias", "offset_w"},
      {ge::kIrInputRequired, ge::kIrInputRequired, ge::kIrInputOptional, ge::kIrInputOptional}, {"y"},
      kIrOutputRequired, {"transpose_x1", "transpose_x2"});
  RegisterVecInputTypeOpCreatorV2(
      "MatMulV3", {"x1", "x2", "bias", "offset_w"},
      {ge::kIrInputRequired, ge::kIrInputRequired, ge::kIrInputOptional, ge::kIrInputOptional}, {"y"},
      kIrOutputRequired, {"transpose_x1", "transpose_x2"});
  RegisterVecInputTypeOpCreatorV2("BatchMatMul", {"x1", "x2", "bias"},
                                  {ge::kIrInputRequired, ge::kIrInputRequired, ge::kIrInputOptional}, {"y"},
                                  kIrOutputRequired, {"adj_x1", "adj_x2"});
  RegisterVecInputTypeOpCreatorV2(
      "BatchMatMulV2", {"x1", "x2", "bias", "offset_w"},
      {ge::kIrInputRequired, ge::kIrInputRequired, ge::kIrInputOptional, ge::kIrInputOptional}, {"y"},
      kIrOutputRequired, {"adj_x1", "adj_x2"});
  RegisterVecInputTypeOpCreatorV2(
      "BatchMatMulV3", {"x1", "x2", "bias", "offset_w"},
      {ge::kIrInputRequired, ge::kIrInputRequired, ge::kIrInputOptional, ge::kIrInputOptional}, {"y"},
      kIrOutputRequired, {"adj_x1", "adj_x2"});
  RegisterVecInputTypeOpCreatorV2(
      "BNInferenceD", {"x1", "x2", "x3", "x4", "x5"},
      {ge::kIrInputRequired, ge::kIrInputRequired, ge::kIrInputRequired, ge::kIrInputOptional, ge::kIrInputOptional},
      {"y"}, kIrOutputRequired, {"momentum", "epsilon", "use_global_stats", "mode"});
}
}  // namespace
}  // namespace ge
