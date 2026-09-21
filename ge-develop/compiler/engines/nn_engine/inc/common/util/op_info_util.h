/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_COMMON_UTIL_OP_INFO_UTIL_H_
#define FUSION_ENGINE_OPTIMIZER_COMMON_UTIL_OP_INFO_UTIL_H_

#include <map>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"
#include "common/aicore_util_types.h"
#include "common/util/constants.h"
#include "graph/types.h"
#include "graph/op_kernel_bin.h"
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"
#include "register/graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "register/graph_optimizer/graph_fusion/graph_fusion_pass_base.h"

namespace fe {
const std::map<OpImplType, domi::ImplyType> IMPL_TYPE_MAP{
    {EN_IMPL_CUSTOM_TIK, domi::ImplyType::BUILDIN},
    {EN_IMPL_CUSTOM_TBE, domi::ImplyType::TVM},
    {EN_IMPL_HW_TIK, domi::ImplyType::BUILDIN},
    {EN_IMPL_HW_TBE, domi::ImplyType::TVM},
    {EN_IMPL_RL, domi::ImplyType::BUILDIN},
    {EN_IMPL_PLUGIN_TBE, domi::ImplyType::TVM},
    {EN_IMPL_VECTOR_CORE_HW_TBE, domi::ImplyType::TVM},
    {EN_IMPL_VECTOR_CORE_CUSTOM_TBE, domi::ImplyType::TVM},
    {EN_IMPL_NON_PERSISTENT_CUSTOM_TBE, domi::ImplyType::TVM},
    {EN_IMPL_HW_DSA, domi::ImplyType::BUILDIN}};

const std::map<domi::ImplyType, std::string> GE_IMPL_TYPE_STRING_MAP{
    {domi::ImplyType::BUILDIN, "BUILTIN"}, {domi::ImplyType::TVM, "TVM"},         {domi::ImplyType::CUSTOM, "CUSTOM"},
    {domi::ImplyType::AI_CPU, "AI_CPU"},   {domi::ImplyType::INVALID, "INVALID"},
};

const std::map<GraphFusionPassType, std::string> PASS_TYPE_STRING_MAP{
    {BUILT_IN_GRAPH_PASS, "built-in-ai-core-graph-pass"},
    {CUSTOM_AI_CORE_GRAPH_PASS, "custom-ai-core-graph-pass"},
    {BUILT_IN_VECTOR_CORE_GRAPH_PASS, "built-in-vector-core-graph-pass"},
    {CUSTOM_VECTOR_CORE_GRAPH_PASS, "custom-vector-core-graph-pass"},
    {SECOND_ROUND_BUILT_IN_GRAPH_PASS, "second-round-graph-pass"},
    {BUILT_IN_BEFORE_TRANSNODE_INSERTION_GRAPH_PASS, "built-in-before-transnode-insertion-graph-pass"},
    {BUILT_IN_PREPARE_GRAPH_PASS, "built-in-prepare-graph-pass"},
    {BUILT_IN_BEFORE_QUANT_OPTIMIZATION_GRAPH_PASS, "built-in-before-quant-optimization-graph-pass"},
    {BUILT_IN_TF_TAG_NO_CONST_FODING_GRAPH_PASS, "built-in-tf-tag-no-const-foding-graph-pass"},
    {BUILT_IN_TF_MERGE_SUB_GRAPH_PASS, "built-in-tf-merge-sub-graph-pass"},
    {BUILT_IN_QUANT_OPTIMIZATION_GRAPH_PASS, "built-in-quant-optimization-graph-pass"},
    {BUILT_IN_EN_ISA_ARCH_EXC_V300_AND_V220_GRAPH_PASS, "built-in-en-isa-arch-exc-v300-and-v220-graph-pass"},
    {BUILT_IN_EN_ISA_ARCH_V100_GRAPH_PASS, "built-in-en-isa-arch-v100-graph-pass"},
    {BUILT_IN_EN_ISA_ARCH_V200_GRAPH_PASS, "built-in-en-isa-arch-v200-graph-pass"},
    {BUILT_IN_DELETE_NO_CONST_FOLDING_GRAPH_PASS, "built-in-delete-no-const-folding-graph-pass"},
    {BUILT_IN_AFTER_MULTI_DIMS_PASS, "built-in-after-multi-dims-pass"},
    {BUILT_IN_AFTER_OPTIMIZE_STAGE1, "built-in-after-optimize-stage1"},
    {BUILT_IN_AFTER_OP_JUDGE, "built-in-after-judge"},
    {BUILT_IN_AFTER_BUFFER_OPTIMIZE, "built-in-after-buffer-optimize"},
};

const std::map<BufferFusionPassType, std::string> BUFFER_FUSION_PASS_TYPE_STRING_MAP{
    {BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, "build-in-ai-core-buffer_fusion-pass"},
    {BUILT_IN_VECTOR_CORE_BUFFER_FUSION_PASS, "build-in-vector-core-buffer_fusion-pass"},
    {CUSTOM_AI_CORE_BUFFER_FUSION_PASS, "custom-ai-core-buffer_fusion-pass"},
    {CUSTOM_VECTOR_CORE_BUFFER_FUSION_PASS, "custom-vector-core-buffer_fusion-pass"},
};

/* record the classification information utilized by CheckDtypeSupported;
 * If the value gap between two data type is less than HIGH_GAP(10),
 * this two data type is under same classification.
 * For example, the value gap between float and float16 is 1,
 * so float and float16 are same classification.
 */
const int32_t LOW_GAP = 0;
const int32_t HIGH_GAP = 10;
const int32_t LOW_GAP_AMPLIFIED = 0;
const int32_t CROSS_GAP_AMPLIFIED = 100;
const int32_t HIGH_GAP_AMPLIFIED = 1000000;
const std::map<ge::DataType, int32_t> DATATYPE_PRIORITY_MAP_AMPLIFIED {
    {ge::DT_FLOAT, 100000}, {ge::DT_BF16, 200000}, {ge::DT_FLOAT16, 200001}};

enum class ForbiddenDtype { FORBIDDEN_NONE = 0, FORBIDDEN_BF16, FORBIDDEN_FP16, FORBIDDEN_DOUBLE };

enum class CheckSupportMode { DTYPE_MODE = 0, DTYPE_FORMAT_MODE, ACCURACY_MODE };

enum class OpNotSupportedReasonID {
  EN_REASON_ID_START = 0,
  EN_TYPE_NOT_FOUND,
  EN_INPUTS_NUM_NOT_MATCH,
  EN_OUTPUTS_NUM_NOT_MATCH,
  EN_PRE_COMPILE_FUNC_IS_NULL,
  EN_INPUTS_NOT_SUPPORT,
  EN_OUTPUTS_NOT_SUPPORT,
  EN_ATTRS_NOT_SUPPORT,
  EN_PARAM_TYPE_NOT_SUPPORT,
  EN_OPERATOR_NOT_SUPPORT,
  EN_INPUTS_AND_OUTPUTS_NOT_ACCURACY_SUPPORT,
  EN_TENSOR_SHAPE_CONTAINS_UNKNOWN_DIM,
  EN_NOT_SUPPORT_DYNAMIC_SHAPE,
  EN_NOT_SUPPORT_CUSTOM_DTYPE,
  EN_REASON_ID_RESERVED
};

const std::map<OpNotSupportedReasonID, std::string> ID_REASON_MAP{
    {OpNotSupportedReasonID::EN_TYPE_NOT_FOUND,
     "The type of this op is not found in op store, check "
     "whether the op store has this type of op."},
    {OpNotSupportedReasonID::EN_INPUTS_NUM_NOT_MATCH,
     "The number of inputs in op desc and op store "
     "does not match, check whether the number of "
     "input from the op in the op store and the graph "
     "is consistent."},
    {OpNotSupportedReasonID::EN_OUTPUTS_NUM_NOT_MATCH,
     "The number of outputs in op desc and op store "
     "does not match, check whether the number of "
     "output from the op in the op store and the "
     "graph is consistent."},
    {OpNotSupportedReasonID::EN_PRE_COMPILE_FUNC_IS_NULL,
     "The pre_compile_func is nullptr, check "
     "whether this type of op does not exist in "
     "tbe-builtin and then go to the tbe-plugin."},
    {OpNotSupportedReasonID::EN_INPUTS_NOT_SUPPORT,
     "The dtype, format or shape of input in op desc is "
     "not supported in op store, check the dtype, "
     "format or shape of input between the op store and "
     "the graph."},
    {OpNotSupportedReasonID::EN_OUTPUTS_NOT_SUPPORT,
     "The dtype, format or shape of output in op desc "
     "is not supported in op store, check the dtype, "
     "format or shape of output between the op store "
     "and the graph."},
    {OpNotSupportedReasonID::EN_ATTRS_NOT_SUPPORT,
     "The number of attrs in op desc and op store does "
     "not match, check whether the attrs in op store are "
     "all in graph."},
    {OpNotSupportedReasonID::EN_PARAM_TYPE_NOT_SUPPORT,
     "The inputs or outputs of op desc do not meet "
     "the paramtype requirements in op store, check "
     "the paramtype in op store and the number of "
     "inputs or outputs in graph."},
    {OpNotSupportedReasonID::EN_OPERATOR_NOT_SUPPORT,
     "The op desc is not supported in the operator "
     "implementation, consult the operator developer "
     "for details."},
    {OpNotSupportedReasonID::EN_INPUTS_AND_OUTPUTS_NOT_ACCURACY_SUPPORT,
     "The inputs and outputs of op desc is not accuracy supported in op store, "
     "check the dtype, format or shape of inputs or outputs between the op "
     "store and the graph."},
    {OpNotSupportedReasonID::EN_TENSOR_SHAPE_CONTAINS_UNKNOWN_DIM,
     "The shape of op desc's input or"
     " output contains unknown dim."},
    {OpNotSupportedReasonID::EN_NOT_SUPPORT_DYNAMIC_SHAPE,
     "This op which is dynamic shape is not configured to support dynamic shape in op store."},
    {OpNotSupportedReasonID::EN_NOT_SUPPORT_CUSTOM_DTYPE,
     "The custom dtypes for this op is not supported by op store."}};


const std::string STR_SOC_VERSION = "soc_version";
const std::string STR_NAME = "name";
const std::string STR_INPUT_LOWERCASE = "input";
const std::string STR_OUTPUT_LOWERCASE = "output";
const std::string STR_PATH = "path";
const std::string STR_IMP_PATH = "imp_path";
const std::string STR_PATTERN = "pattern";
const std::string STR_FORMAT = "format";
const std::string STR_SUB_FORMAT = "sub_format";
const std::string STR_UNKNOWN_SHAPE_FORMAT = "unknownshape_format";
const std::string STR_FALLBACK = "fallback";
const std::string STR_ENABLE = "enable";
const std::string STR_UNKNOWN_SHAPE_ENABLE = "unknownshape_enable";
constexpr char const *STR_INPUT_MEM_CONTINUES = "inputMemContinues";
constexpr char const *STR_OUTPUT_MEM_CONTINUES = "outputMemContinues";
constexpr char const *ATTR_NAME_CONTINUOUS_INPUT = "continuous_input";
constexpr char const *ATTR_NAME_CONTINUOUS_OUTPUT = "continuous_output";
const std::string STR_QUANT_MODE = "quant_mode";
const std::string STR_QUANT_HIGH_PRECISION = "quant_high_precision";
const std::string STR_QUANT_HIGH_PERFORMANCE = "quant_high_performance";
const std::string STR_INT8 = "int8";
const std::string STR_BOOL = "bool";
const std::string STR_DTYPE = "dtype";
constexpr char const *CONCATD = "ConcatD";
const std::string CONCATV2D = "ConcatV2D";
const std::string SPLIT = "Split";
const std::string SPLITV = "SplitV";
const std::string SPLITD = "SplitD";
const std::string SPLITVD = "SplitVD";
const std::string STRIDEDREAD = "StridedRead";
const std::string STRIDEDWRITE = "StridedWrite";
const std::string DEPTHWISE6TO4 = "DepthwiseWeight6DTo4D";
const std::string DEPTHWISE4TO6 = "DepthwiseWeight4DTo6D";
const std::string FRAMEWORKOP = "FrameworkOp";
const std::string PERMUTE = "Permute";
const std::string STRIDEDSLICED = "StridedSliceD";
const std::string GREATER = "Greater";
const std::string NOTEQUAL = "NotEqual";
const std::string TILE = "Tile";
const std::string TILED = "TileD";

const std::string SWITCH = "Switch";
const std::string REFORMAT = "ReFormat";

const std::string CONVBNFILTERHOST = "ConvBnFilterHost";
const std::string CONVBNBIASHOST = "ConvBnBiasHost";
const std::string kWeightCompressHost = "WeightCompressHost";

const std::string SWITCHN = "SwitchN";
const std::string GROUPPADDING = "GroupPadding";

const std::string ATTR_NAME_STREAM_LABEL = "_stream_label";
const std::string NEED_RUN_AFTER_GETSHAPE = "_need_run_after_getshape";
const std::string OP_PARA_SIZE = "op_para_size";
const std::string ORI_OP_PARA_SIZE = "ori_op_para_size";
const std::string NULL_OP_FILE = "Null";
const std::string AICORE_NULL = "AICore_Null";
const std::string OP_THREAD_PARA_SIZE = "op_thread_para_size";

const std::string TRANSDATA_INPUT_NAME = "src";
const std::string TRANSDATA_OUTPUT_NAME = "dst";
const std::string CAST_INPUT_NAME = "x";
const std::string CAST_OUTPUT_NAME = "y";
const std::string TRANSPOSE_INPUT_NAME = "x";
const std::string TRANSPOSE_OUTPUT_NAME = "y";
const std::string RESHAPE_INPUT_NAME = "x";
const std::string RESHAPE_SHAPE_NAME = "shape";
const std::string RESHAPE_OUTPUT_NAME = "y";
const std::string SQUEEZE_V2_INPUT_NAME = "x";
const std::string SQUEEZE_V2_OUTPUT_NAME = "y";
const std::string UNSQUEEZE_V2_INPUT_NAME = "x";
const std::string UNSQUEEZE_V2_OUTPUT_NAME = "y";

const std::string N_AXIS_NAME = "N";
const std::string C_AXIS_NAME = "C";
const std::string H_AXIS_NAME = "H";
const std::string W_AXIS_NAME = "W";
const std::string D_AXIS_NAME = "D";
const std::string AXES_ATTR_NAME = "axes";
const std::string AXIS_ATTR_NAME = "axis";

const std::string STR_NOT_SUPPORTED = "not_supported";
const std::string STR_FULLY_SUPPORTED = "fully_supported";
const std::string STR_PARTIALLY_SUPPORTED = "partially_supported";

const int64_t SHAPE_DIM_VALUE_C04 = 4;

const size_t MINUS_VALUE_ONE = 1;
const size_t MINUS_VALUE_TWO = 2;
const size_t SIZE_OF_CN = 2;
const int64_t NI = 16;

const uint32_t LOW_DIMENSION_NUM_THD = 3;

const uint32_t MINIMUM_NZ_SHAPE_DIM_NUM = 2;
const uint32_t MINIMUM_ND_TO_RNN_SHAPE_NUM = 2;

const uint32_t NOT_SUPPORTED_REASON_OFFSET_UNIT = 0xF;
const uint32_t NOT_SUPPORTED_REASON_OFFSET_BIT_NUM = 4;
const int64_t NOT_SUPPORTED_FLAG_BIT = 0x8000000000000000;

const std::string STR_NON_PERSISTENT_CUSTOM_TBE = "non-persistent-tbe-custom";

/* CAST OP's input and output name. May be weird... */
const std::string CAST_ATTR_DSTT = "DstT";
const std::string CAST_ATTR_SRCT = "SrcT";
const std::string CAST_ATTR_DST_TYPE = "dst_type";
const std::string CAST_ATTR_SRC_TYPE = "src_type";
const std::string CAST_ATTR_TRUNCATE = "truncate";

const std::string ATTR_NAME_INPUT_FORMAT = "input_format";
const std::string ATTR_NAME_OUTPUT_FORMAT = "output_format";

const std::string ATTR_NAME_SRC_FORMAT = "src_format";
const std::string ATTR_NAME_DST_FORMAT = "dst_format";

const std::string ATTR_NAME_INPUT_SRC_FORMAT_INT = "input_src_format_int";
const std::string ATTR_NAME_OUTPUT_SRC_FORMAT_INT = "output_src_format_int";

// The align size of data memory
const uint32_t DATA_MEMORY_ALIGN_SIZE = 32;
const int64_t UNKNOWN_WORK_SPACE_SIZE = -1;
const int32_t NON_PERSISTENT_CUSTOM_PRIORITY = -1;

const uint32_t TENSOR_INDEX_FILTER_COMPRESS = 1;
const uint32_t TENSOR_INDEX_COMPRESS_INDEX = 2;
const uint32_t COMPRESSOP_INDEX_WEIGHT_COMPRESS = 0;
const uint32_t COMPRESSOP_INDEX_COMPRESS_INDEX = 1;

// unknown shape value
const int64_t UNKNOWN_SHAPE_VALUE = -1;

// default value of groups
const int64_t GROUPS_DEFAULT_VALUE = 1;

// rnn op attr
const int64_t RNN_HIDDEN_SIZE_DEFAULT_VALUE = 1;
const int64_t RNN_INPUT_SIZE_DEFAULT_VALUE = 1;
const int64_t RNN_STATE_SIZE_DEFAULT_VALUE = -1;

const int32_t NC1HWC0_DIM_N = 0;
const int32_t NC1HWC0_DIM_C1 = 1;
const int32_t NC1HWC0_DIM_C0 = 4;
const int32_t NC1HWC0_DIM_H = 2;
const int32_t NC1HWC0_DIM_W = 3;

const int32_t NDC1HWC0_DIM_N = 0;
const int32_t NDC1HWC0_DIM_D = 1;
const int32_t NDC1HWC0_DIM_C1 = 2;
const int32_t NDC1HWC0_DIM_C0 = 5;
const int32_t NDC1HWC0_DIM_H = 3;
const int32_t NDC1HWC0_DIM_W = 4;

const int32_t C1HWNCoC0_DIM_C1 = 0;
const int32_t C1HWNCoC0_DIM_H = 1;
const int32_t C1HWNCoC0_DIM_W = 2;
const int32_t C1HWNCoC0_DIM_N = 3;
const int32_t C1HWNCoC0_DIM_Co = 4;
const int32_t C1HWNCoC0_DIM_C0 = 5;

const int32_t C1DHWNCoC0_DIM_C1 = 0;
const int32_t C1DHWNCoC0_DIM_D = 1;
const int32_t C1DHWNCoC0_DIM_H = 2;
const int32_t C1DHWNCoC0_DIM_W = 3;

const uint32_t DIMENSION_NUM_FIVE = 5;

// dim default size value
const size_t DIM_DEFAULT_SIZE = 4;

// op type
const std::string OP_TYPE_PLACE_HOLDER = "PlaceHolder";
const std::string OP_TYPE_END = "End";
const std::string DATA = "Data";
const std::string REFDATA = "RefData";
const std::string NETOUTPUT = "NetOutput";
const std::string NO_OP = "NoOp";
const std::string RESHAPE = "Reshape";
const std::string VARIABLE = "Variable";
const std::string TRANSDATA = "TransData";
const std::string TRANSDATARNN = "TransDataRNN";
const std::string TRANSPOSE = "Transpose";
const std::string TRANSPOSED = "TransposeD";
const std::string CAST = "Cast";
const std::string ANN_DATA = "AnnData";
const std::string AIPPDATA = "AippData";
const std::string AIPP = "Aipp";
const std::string CONSTANTOP = "Constant";
const std::string CONSTANT = "Const";
const std::string GETNEXT = "GetNext";
const std::string SQUEEZE_V2 = "SqueezeV2";
const std::string UNSQUEEZE_V2 = "UnsqueezeV2";
const std::string PARTITIONEDCALL = "PartitionedCall";
const std::string OP_FIXPIPE = "FixPipe";
const std::string Conv2D = "Conv2D";
const std::string CONV2DDWD = "Conv2DBackpropFilterD";
const std::string CONV2DDW = "Conv2DBackpropFilter";
const std::string OP_TYPE_ROIPOOLING = "ROIPooling";
const std::string OP_TYPE_SSD_DETECTION_OUTPUT = "SSDDetectionOutput";

constexpr char const *CONV2D = "Conv2D";
constexpr char const *DECONVOLUTION = "Deconvolution";
constexpr char const *MATMULV2OP = "MatMulV2";
constexpr char const *DEPTHWISECONV2D = "DepthwiseConv2D";
constexpr char const *kFullyConnection = "FullyConnection";
constexpr char const *kConv2DTranspose = "Conv2DTranspose";
constexpr char const *kConv2DTransposeD = "Conv2DTransposeD";
constexpr char const *kBatchMatMul = "BatchMatMul";
constexpr char const *kConv2DCompress = "Conv2DCompress";
constexpr char const *kFullyConnectionCompress = "FullyConnectionCompress";
constexpr char const *kMatMulV2Compress = "MatMulV2Compress";
constexpr char const *kConv2DTransposeDCompress = "Conv2DTransposeDCompress";
constexpr char const *kBatchMatMulCompress = "BatchMatMulCompress";
const unordered_set<std::string> kConvCompressOpList = {
    kConv2DCompress, kConv2DTransposeDCompress
};
const unordered_set<std::string> kCubeCompressOpList = {
    kConv2DCompress,
    kFullyConnectionCompress,
    kMatMulV2Compress,
    kConv2DTransposeDCompress,
    kBatchMatMulCompress
};
const std::string COMPRESSOP = "Compress";
const std::string COMPRESSFCOP = "CompressFcOp";
const std::string SWAPCO = "SwapCo";
const std::string SWAPCI = "SwapCi";
const std::string BNHOST = "BnHost";
constexpr char const *POOLING = "Pooling";
constexpr char const *ELTWISE = "Eltwise";
const std::string RELU = "Relu";
const std::string RELU6 = "Relu6";
const std::string LEAKY_RELU = "LeakyRelu";
const std::string READ_SELECT = "ReadSelect";
constexpr char const *ASCEND_QUANT = "AscendQuant";
constexpr char const *OP_TYPE_QUANTIZE = "Quantize";
const std::string kAscendRequant = "AscendRequant";
const std::string ASCEND_DEQUANT = "AscendDequant";
const std::string OP_TYPE_PHONY_CONCAT = "PhonyConcat";
const std::string NEXT_ITERATION = "NextIteration";
constexpr char const *SCALE = "Scale";
constexpr char const *BNINFERENCED = "BNInferenceD";
constexpr char const *SOFTMAX_V2 = "SoftmaxV2";
constexpr char const *kSend = "Send";
constexpr char const *kRecv = "Recv";
constexpr char const *kIntrinsicND2NZ = "Intrinsic_data_move_out2l1_nd2nz";

// op tensor
constexpr uint32_t kTransposeInputPerm = 1U;
constexpr size_t kTransposeOutputSize = 1;

const std::set<WeightCompressType> kWeightCompressTypes = {
        WeightCompressType::LOW_SPARSE_COMPRESS,
        WeightCompressType::HIGH_SPARSE_COMPRESS
};

const std::unordered_set<std::string> PLACE_OR_END_SET({
    OP_TYPE_PLACE_HOLDER, OP_TYPE_END, DATA, REFDATA,
    CONSTANT, CONSTANTOP, VARIABLE
});

const std::map<OpImplType, std::string> IMPL_TYPE_STRING_MAP{
        {EN_IMPL_CUSTOM_TBE, "tbe-custom"},
        {EN_IMPL_HW_TBE, "tbe-builtin"},
        {EN_IMPL_VECTOR_CORE_HW_TBE, "tbe-builtin-vector-core"},
        {EN_IMPL_VECTOR_CORE_CUSTOM_TBE, "tbe-custom-vector-core"},
        {EN_IMPL_NON_PERSISTENT_CUSTOM_TBE, "tbe-custom-non-persistent"},
};

// the origin fromat
const std::vector<ge::Format> FE_IDENTIFIABLE_ORIGIN_FORMAT_VECTOR = {
    ge::FORMAT_NCHW,  ge::FORMAT_NHWC,  ge::FORMAT_HWCN,  ge::FORMAT_CHWN,
    ge::FORMAT_NDHWC, ge::FORMAT_NCDHW, ge::FORMAT_DHWCN, ge::FORMAT_DHWNC};

// C04 format
const std::vector<ge::Format> FE_C04_FORMAT_VECTOR = {ge::FORMAT_NC1HWC0_C04, ge::FORMAT_FRACTAL_Z_C04};
const std::unordered_set<std::string> FE_C04_SUPPORT_CUBE_OP = {CONV2D, CONV2DDWD, CONV2DDW};
const std::unordered_set<std::string> KFeFormatModeFilterOp = {"MatMul", "MatMulV2", "BatchMatMulV2", "BatchMatMul"};
const std::vector<ge::Format> FE_GROUP_RELA_FORMAT_VECTOR = {ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z_3D};

const std::unordered_set<int> BUILT_IN_IMPLY_TYPE{
    EN_IMPL_HW_CONSTANT_CCE, EN_IMPL_HW_GENERAL_CCE, EN_IMPL_HW_TIK, EN_IMPL_HW_TBE,
    EN_IMPL_RL, EN_IMPL_PLUGIN_TBE, EN_IMPL_VECTOR_CORE_HW_TBE};

const std::unordered_set<std::string> kGeDeleteOpType = {
        "Expanddims",
        "Reshape",
        "ReFormat",
        "Squeeze",
        "Unsqueeze",
        "SqueezeV2",
        "UnsqueezeV2",
        "SqueezeV3",
        "UnsqueezeV3",
        "Size",
        "Shape",
        "ShapeN",
        "Rank"
};

const std::unordered_set<std::string> kConstFoldingOpType = {
        "GroupPadding",
        "ConvBnFilterHost",
        "ConvScaleFilterHost",
        "Concatv2HostCpuOp",
        "RequantHostCpuOp",
        "QuantWeightRollBack",
        "GatherV2",
        "GatherV2D",
        "SwapCo",
        "ReverseV2D",
        "ConcatV2",
        "TransData",
        "Cast",
        "Reshape",
        "TransposeD",
        "ReFormat",
        "SqueezeV2",
        "UnsqueezeV2",
        "Maximum",
        "Add",
        "Mul",
        "Sub",
        "AscendWeightQuant"
};

const std::unordered_set<std::string> kDSAStatelessOps = {"DSAStatelessRandomTruncatedNormal",
                                                          "DSAStatelessRandomNormal",
                                                          "DSAStatelessGenBitMask",
                                                          "DSAStatelessRandomUniform"};

const std::unordered_set<std::string> kConcatCOptimizeOpType = {
        "Conv2D",
        "FixPipe",
        "Conv2DTranspose",
        "Conv2DTransposeD"
};

enum class LxFusionOptimizeResult {
  NO_FUSION_STRATEGY,
  L1_FUSION_STRATEGY,
  L2_FUSION_STRATEGY,
  BOTH_FUSION_STRATEGY,
  C_OPTIMIZE_STRATEGY,
  FORMAT_TUNE_STRATEGY
};

struct CalcShapeExtraAttr {
  int64_t hidden_size;
  int64_t input_size;
  int64_t state_size;
};

using TbeJsonPtr = std::shared_ptr<nlohmann::json>;
struct CompileResultInfo {
  std::string json_file_path;
  std::string bin_file_path;
  TbeJsonPtr json_ptr;
  ge::OpKernelBinPtr bin_ptr;
  CompileResultInfo() : json_file_path(), bin_file_path(), json_ptr(nullptr), bin_ptr(nullptr) {}
  CompileResultInfo(const std::string &new_json_file_path)
    : json_file_path(new_json_file_path), bin_file_path(), json_ptr(nullptr), bin_ptr(nullptr) {}
};

struct ShapeAndFormatInfo {
  ge::GeShape old_shape;
  ge::GeShape &new_shape;
  ge::Format old_format;
  ge::Format new_format;
  ge::DataType current_data_type;
  int64_t group_count;
  CalcShapeExtraAttr extra_attr;
  ShapeAndFormatInfo(ge::GeShape old_shape_param, ge::GeShape &new_shape_param, ge::Format old_format_param,
                     ge::Format new_format_param, ge::DataType current_data_type_param) :
          old_shape(old_shape_param), new_shape(new_shape_param), old_format(old_format_param),
          new_format(new_format_param), current_data_type(current_data_type_param), group_count(GROUPS_DEFAULT_VALUE),
          extra_attr({1, 1, -1}) {}
  ShapeAndFormatInfo(ge::GeShape old_shape_param, ge::GeShape &new_shape_param, ge::Format old_format_param,
                     ge::Format new_format_param, ge::DataType current_data_type_param, int64_t group_count_param) :
          old_shape(old_shape_param), new_shape(new_shape_param), old_format(old_format_param),
          new_format(new_format_param), current_data_type(current_data_type_param), group_count(group_count_param),
          extra_attr({1, 1, -1}) {}
  ShapeAndFormatInfo(ge::GeShape old_shape_param, ge::GeShape &new_shape_param, ge::Format old_format_param,
                     ge::Format new_format_param, ge::DataType current_data_type_param, int64_t group_count_param,
                     CalcShapeExtraAttr extra_attr_param) :
          old_shape(old_shape_param), new_shape(new_shape_param), old_format(old_format_param),
          new_format(new_format_param), current_data_type(current_data_type_param), group_count(group_count_param),
          extra_attr(extra_attr_param) {}
};

using ShapeAndFormat = struct ShapeAndFormatInfo;
};  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_COMMON_UTIL_OP_INFO_UTIL_H_
