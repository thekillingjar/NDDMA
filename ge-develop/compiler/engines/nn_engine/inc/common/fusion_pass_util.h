/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_INC_COMMON_FUSION_PASS_UTIL_H_
#define FUSION_ENGINE_INC_COMMON_FUSION_PASS_UTIL_H_

#include <string>
#include <unordered_set>
namespace fe {

const std::unordered_set<std::string> ForbiddenClosedPass = {
        // Graph Fusion
        "ArgMaxWithFusionPass",
        "COPYPass",
        "ADeformableConv2dPass",
        "DynamicGRUV2GradFusionPass",
        "DynamicRNNGradFusionPass",
        "DynamicRNNInsertTransposePass",
        "HostBNFusionPass",
        "ALSTMFusionPass",
        "MapIndexFusionPass",
        "NormalizeFusionPass",
        "PassThroughFusionPass",
        "PassThroughSecondFusionPass",
        "PriorBoxPass",
        "ProposalFusionPass",
        "RNNFusionPass",
        "SpatialTransformerDPass",
        "SPPPass",
        "YoloPass",
        "YoloV2DetectionOutputPass",
        "YoloV3DetectionOutputV2Pass",
        "PermuteFusionPass",
        "CommonLSTMFusionPass",
        "PReluGradFusionPass",
        "ConstToAttrReduceSumFusion",
        "TfTagNoConstFoldingFusionPass",
        "TfMergeConv2DBackpropInputFusionPass",
        "TfMergeSubFusionPass",
        "AvgPoolQuantProcessFusionPass",
        "Conv2DQuantProcessFusionPass",
        "GroupConv2DQuantProcessFusionPass",
        "Conv2DTDQuantProcessFusionPass",
        "DeConvQuantProcessFusionPass",
        "DWConv2DQuantProcessFusionPass",
        "FCQuantProcessFusionPass",
        "MatmulV2QuantProcessFusionPass",
        "PoolingQuantProcessFusionPass",
        "BatchMatmulV2QuantProcessFusionPass",
        "V100NotRequantFusionPass",
        "V200NotRequantFusionPass",
        "DepthwiseInsertTransDataFusionPass",
        "DeformableOffsetsFusionPass",
        "EinsumPass",
        "DepthwiseFusionPass",
        "DepthwiseToConv2dFusionPass",

        // UB Fusion
        "TbeConvCommonRules0FusionPass",
        "TbeConvCommonRules2FusionPass",
        "TbeAntiquantMaxpoolingFusionPass",
        "TbeConvRequantFusionPass",
        "TbeConvDequantS16FusionPass",
        "PackFusionPass",
        "UnpackFusionPass",
        "ConstToAttrPass",
        "ApplyAddOutputPass",
        "ZSplitVFusionPass",
        "ZConfusionSoftmaxGradFusionPass",
        "AvgPool3DGradFusionPass",
        "AvgPoolGradFusionPass",
        "AddNFusionPass",
        "MaxPoolWithArgmaxFusionPass",
        "TransdataCastFusionPass",
        "ConstToAttrGatherV2Fusion",
        "FusedBatchNormGradFusionPass",
        "FusedBatchnormFusionPass",
        "ApplyAddOutputPass",
        "TransposedUpdateFusionPass",
        "AReduceMeanFusionPass",
        "AReduceSumFusionPass",
        "BatchNormBnInferFusionPass",
        "BatchNormGradInfGradFusion",
        "FusedBatchNormBertFusionPass",
        "FusedBatchNormGradFusionPass",
        "BatchNormGradBnInferGradFusion",
        "SingleBatchNormFusion",
        "ConstToAttrResizeNearestNeighborGradFusion",
        "ADepthwiseFusionPass",
        "DepthwiseDfFusionPass",
        "DepthwiseDwMulFusionPass",
        "AABiasaddConvFusion",
        "sedBatchNormGradFusionPass",
        "Globalavgpoolpass",
        "DreluFusionPass",
        "SoftmaxGradExtFusion",
        "AvgPool3DFusionPass",
        "DynamicRNNGradAlignFusionPass",
        "DynamicRNNGradDAlignFusionPass",
        "DynamicRNNGradDFusionPass",
        "DynamicRNNGradAFusionPass",
        "DynamicRNNInsertTransposePass",
        "DynamicRNNSeqFusionPass",
        "DynamicRNNFusionPass",
        "ZConcatDFusionPass",
        "ZSplitVDFusionPass",
        "clip_by_norm_nodivsquaresum"
};
}  // namespace fe
#endif  // FUSION_ENGINE_INC_COMMON_FUSION_PASS_UTIL_H_
