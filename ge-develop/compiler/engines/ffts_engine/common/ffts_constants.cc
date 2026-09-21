/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unordered_set>
#include <map>
#include <string>
#include "graph/types.h"
#include "inc/ffts_type.h"

namespace ffts {
const uint32_t kManualMode = 0U;

const uint32_t kAutoMode = 1U;

const uint32_t kRecuriseCntMax = 100U;

const std::string kFFTSPlusEngineName = "ffts_plus";

const std::string EM_PARAM = "parameter";

const std::string STR_SOC_VERSION = "soc_version";

const std::string EM_INPUT_PARAM_EMPTY = "E10004";

const std::string kTypeFFTSPlus = "_ffts_plus";

const std::string kThreadId = "_thread_id";

const std::string kDnnVmAICpu = "DNN_VM_AICPU";

const std::string kDnnVmAICpuAscend = "DNN_VM_AICPU_ASCEND";

const std::string kDnnVmGeLocal = "DNN_VM_GE_LOCAL";

const std::string kDnnVmRts = "DNN_VM_RTS_FFTS_PLUS";

const std::string kDsaCoreEngineName = "DSAEngine";

const std::string kAIcoreEngineName = "AIcoreEngine";

const std::string kHcclSubTasks = "hccl_sub_tasks";

const std::string kCtxIdList = "ctx_id_list";

const std::string kCachePersistSize = "_cache_persist_size";

const std::string kCachePersist = "_cache_persist";

const std::string kFFTSMode = "ffts_mode";

const std::string kFFTSPlus = "ffts-plus";

const std::string kFFTS = "ffts";

const std::string kSocInfo = "SoCInfo";

const std::string ATTR_NAME_PARENT_NODE = "parentNode";

const std::string kContextId = "_context_id";

const std::string kSuccList = "_succ_list";

const std::string kSuccListList = "_succ_list_list";

const std::string kNonEdgeSuccList = "_non_edge_succ_list";

const std::string kNonEdgePreNodes = "_non_edge_pre_nodes";

const std::string kHcclOutDegree0Num = "_hccl_list_out_degree_0_num";

const std::string kPrefetchEnableBm = "_prefetch_enable_bm";

const std::string kWriteBackBm = "_write_back_bm";

const std::string kInvalidateBm = "_invalidate_bm";

const std::string kAutoInlabelCtxId = "_in_label_ctx_id";

const std::string kAutoCtxIdList = "_context_id_list";

const std::string kAtomicCtxIdList = "_atomic_context_id_list";

const std::string kAutoAtStartCtxIdList = "_at_start_ctx_id_list";

const std::string kAutoOutlabelCtxId = "_out_label_ctx_id";

const std::string kAutoAtEndCtxIdList = "_at_end_ctx_id_list";

const std::string kAutoAtEndPreCnt = "_at_end_pre_cnt";

const std::string kTypePhonyConcat = "PhonyConcat";

const std::string kTypePhonyReduce = "PhonyReduce";

const std::string kAttrDsaCtxDef = "_dsa_ctx_def";

const std::string kAttrAICpuCtxDef = "_ffts_plus_aicpu_ctx_def";

const std::string kAttrAICoreCtxDef = "_aicore_ctx_def";

const std::string kAttrAICoreCtxType = "_aicore_ctx_type";

const std::string kAtomicAddrClean = "AtomicAddrClean";

const char *kCoreTypeAIC = "AIC";

const char *kCoreTypeAIV = "AIV";

const char *kCoreTypeMixAIC = "MIX_AIC";

const char *kCoreTypeMixAIV = "MIX_AIV";

const char *kTensorCtxId = "_tensor_ctx_id";

const std::string kAttrThreadCoreType = "_thread_cube_vector_core_type";

const std::string kAdjacencyList = "adjacency_list";

const std::string kModeInArgsFirstField = "_mode_in_args_first_field";

const std::string kAttrIntercoreSync = "_inter_core_sync";

const std::string  ATTR_NAME_ALIAS_ENGINE_NAME = "_alias_engine_name";

const std::string kRtsFftsPlusOpStoreName = "DNN_VM_RTS_FFTS_PLUS_OP_STORE";

const std::string kAttrRefPortIndex = "ref_port_index";

const std::string kFFTSPlusInDynamic = "_ffts_plus_in_dynamic";

const uint32_t kGetFirstAvailableLabel = 1000U;

const size_t kMaxPretchNum = 4U;

const uint32_t kMaxIdx = 64U;

const uint32_t kMaxPersistNum = 8U;

const uint32_t kDefaultWindowSize = 4U;

const uint32_t kDefaultManualWindowSize = 4U;

const std::map<ge::DataType, uint32_t> DATATYPE_SIZE_MAP {
        {ge::DT_FLOAT, sizeof(float)},
        {ge::DT_FLOAT16, sizeof(int16_t)},
        {ge::DT_BF16, sizeof(int16_t)},
        {ge::DT_INT8, sizeof(int8_t)},
        {ge::DT_INT32, sizeof(int32_t)},
        {ge::DT_UINT8, sizeof(uint8_t)},
        {ge::DT_UINT32, sizeof(uint32_t)},
        {ge::DT_INT16, sizeof(int16_t)},
        {ge::DT_UINT16, sizeof(uint16_t)},
        {ge::DT_INT64, sizeof(int64_t)},
        {ge::DT_UINT64, sizeof(uint64_t)},
        {ge::DT_DOUBLE, sizeof(double)},
        {ge::DT_BOOL, sizeof(bool)},
        {ge::DT_DUAL, sizeof(float) + sizeof(int8_t)},
        {ge::DT_DUAL_SUB_UINT8, sizeof(int8_t)},
        {ge::DT_DUAL_SUB_INT8, sizeof(int8_t)},
        {ge::DT_COMPLEX32, sizeof(int32_t)},
        {ge::DT_COMPLEX64, sizeof(int64_t)}
};

const std::map<std::string, TaskBuilderType> kDsaOrAicpuMap {
        {"_dsa_ctx_def", TaskBuilderType::EN_TASK_TYPE_DSA},
        {"_dsa_ctx_def_auto", TaskBuilderType::EN_TASK_TYPE_DSA_AUTO},
        {"_ffts_plus_aicpu_ctx_def", TaskBuilderType::EN_TASK_TYPE_AICPU},
        {"_ffts_plus_aicpu_ctx_def_auto", TaskBuilderType::EN_TASK_TYPE_AICPU_AUTO},
};
const std::string kAutoSuffix = "_auto";
const std::string kCoreTypeHcomReduce = "HcomReduce";
const std::string kCoreTypeHcomAllReduce = "HcomAllReduce";
const std::string kCoreTypeHcomAllGather = "HcomAllGather";
const std::string kCoreTypeHcomBroadCast = "HcomBroadcast";
const std::string kCoreTypeHcomReduceScatter = "HcomReduceScatter";
const std::string kCoreTypeHcomSend = "HcomSend";
const std::string kCoreTypeHcomReceive = "HcomReceive";
const std::string kCoreTypeHcomRemoteRead = "HcomRemoteRead";
const std::string kCoreTypeHcomRemoteRefRead = "HcomRemoteRefRead";
const std::string kCoreTypeHcomRemoteWrite = "HcomRemoteWrite";
const std::string kCoreTypeHcomRemoteScatterWrite = "HcomRemoteScatterWrite";
const std::string kCoreTypeHcomAllToAllV = "HcomAllToAllV";
const std::string kCoreTypeHcomGatherAllToAllV = "HcomGatherAllToAllV";
const std::unordered_set<std::string> kHCCLOpType = {kCoreTypeHcomReduce, kCoreTypeHcomAllReduce,
                                                     kCoreTypeHcomAllGather, kCoreTypeHcomBroadCast,
                                                     kCoreTypeHcomReduceScatter, kCoreTypeHcomSend,
                                                     kCoreTypeHcomReceive, kCoreTypeHcomRemoteRead,
                                                     kCoreTypeHcomRemoteRefRead, kCoreTypeHcomRemoteWrite,
                                                     kCoreTypeHcomRemoteScatterWrite, kCoreTypeHcomAllToAllV,
                                                     kCoreTypeHcomGatherAllToAllV};

const std::unordered_set<std::string> CONTROL_OP_V2_TYPE = {"If", "While", "Case"};

const std::unordered_set<std::string> NO_NEED_GEN_TASK_OP_TYPE = {"Data", "NetOutput", "Variable",
                                                                  "Const", "Constant", "PhonyConcat",
                                                                  "StreamActive"};

const std::unordered_set<std::string> kPhonyTypes = {kTypePhonyConcat, kTypePhonyReduce};

const std::string PLACEHOLDER = "PlaceHolder";

const std::string END = "End";

const std::string DATA = "Data";

const std::string REFDATA = "RefData";

const std::string NETOUTPUT = "NetOutput";

const std::string RESHAPE = "Reshape";

const std::string VARIABLE = "Variable";

const std::string TRANSDATA = "TransData";

const std::string TRANSDATARNN = "TransDataRNN";

const std::string CAST = "Cast";

const std::string ANN_DATA = "AnnData";

const std::string AIPPDATA = "AippData";

const std::string AIPP = "Aipp";

const std::string CONSTANTOP = "Constant";

const std::string CONSTANT = "Const";

const std::string GETNEXT = "GetNext";

const std::string CONV2D = "Conv2D";

const std::string MATMULV2OP = "MatMulV2";

const std::string DEPTHWISECONV2D = "DepthwiseConv2D";

const std::string POOLING = "Pooling";

const std::string ELTWISE = "Eltwise";

const std::string RELU = "Relu";

const std::string RELU6 = "Relu6";

const std::string LEAKY_RELU = "LeakyRelu";

const std::string READ_SELECT = "ReadSelect";

const std::string NEXT_ITERATION = "NextIteration";
const std::string ATTR_NAME_INPUT_IS_VAR = "INPUT_IS_VAR";
const std::string PARTITIONEDCALL = "PartitionedCall";
const std::string EQUAL = "Equal";
const std::string NOEQUAL = "NotEqual";
const std::string GREATE = "Greater";
const std::string GREATEOREQUAL = "GreaterEqual";
const std::string LESS = "Less";
const std::string LESSOREQUAL = "LessEqual";

const std::string ATTR_NAME_LABEL_JUMP_NODE_INDEX = "_label_start_nodeindex";
const std::string ATTR_NAME_LABEL_JUMP_NODE = "_label_start_node";
const std::string ATTR_NAME_LABEL_JUMP_NODES = "_label_jump_nodes";
const std::string ATTR_NAME_LASTLABELSET_OUT_NODES = "_labelset_out_nodes";
const std::string ATTR_NAME_LABELSET_PRE_LABEL = "_labelset_pre_node";
const std::string ATTR_NAME_PARENT_OUTPUTS_INPUT_NODES = "_label_parent_out_in_nodes";
const std::string ATTR_NAME_PARENT_OUTPUTS_OUTPUT_NODE = "_label_parent_out_out_node";
const std::string ATTR_NAME_PARENT_PRE_NODES = "_label_firstnode_pre_nodes";
const std::string ATTR_NAME_LABEL_SWITCH_INDEX = "_label_switch_index";
const std::string ATTR_NAME_LABEL_SWITCH_LIST = "_label_switch_list";
const std::string ATTR_NAME_HAS_PARTITION = "_has_partion_with";
const std::unordered_set<std::string> COND_SWITCH_NODE_TYPE = {"Less", "LessEqual",
                                                               "Greater", "GreaterEqual",
                                                               "Equal", "NotEqual"};
const std::unordered_set<std::string> CASE_SWITCH_NODE_TYPE = {"LabelSwitchByIndex"};
const std::unordered_set<std::string> SDMA_NODE_TYPE = {"MemcpyAsync", "Enter", "RefEnter",
                                                        "LoopCond", "NextIteration", "RefNextIteration",
                                                        "Exit", "RefExit", "Identity", "ReadVariableOp"};
const std::unordered_set<std::string> LABEL_NODE_TYPE = {"LabelSet", "LabelGotoEx", "LabelGoto"};
const std::string kRuntimeContentx = "_ffts_runtime_context";
const std::string kFftsFirstOpName = "_ffts_first_op_name";
const char *kOnlyAddContext = "OnlyAddContextID";
const std::string kFftsFirstSliceFlag = "_ffts_first_slice_flag";

const std::string kFFTSThreadDim = "_ffts_thread_dim";
const std::string kFFTSWindowSize = "_ffts_window_size";
const std::string kFftsFlagDefault = "default";
const std::string kFftsFlagAllSubgraph = "ffts";
}  // namespace ffts
