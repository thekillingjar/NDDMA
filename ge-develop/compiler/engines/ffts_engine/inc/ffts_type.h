/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFTS_ENGINE_INC_FFTS_TYPE_H
#define FFTS_ENGINE_INC_FFTS_TYPE_H

#include <unordered_set>
#include <map>
#include <string>
#include "graph/types.h"
namespace ffts {

enum class CACHE_OPERATION {
  PREFETCH = 0,
  INVALIDATE = 1,
  WRITE_BACK = 2,
  INNER_WRITE_BACK = 3,
  CACHE_OPERATION_BOTTOM = 4
};

enum class ThreadMode {
  MANUAL_THREAD = 0,
  AUTO_THREAD,
  THREAD_MODE_RESERVED
};

enum class FFTSMode {
  FFTS_MODE_NO_FFTS = 0,
  FFTS_MODE_FFTS,
  FFTS_MODE_FFTS_PLUS,
  FFTS_MODE_RESERVED
};

enum class ModeType {
  MANUAL_MODE_TYPE = 0,
  AUTO_MODE_TYPE,
  DYNAMIC_MODE_TYPE,
  MIX_L2_MODE_TYPE
};

enum class TaskBuilderType {
  EN_TASK_TYPE_AIC_AIV = 0,   // ai core op, aic or aiv
  EN_TASK_TYPE_AIC_AIV_AUTO,
  EN_TASK_TYPE_MIX_AIC_AIV,   // mix op, contain aic & aiv
  EN_TASK_TYPE_MIX_AIC_AIV_AUTO,
  EN_TASK_TYPE_MIX_AIC_AIV_DYNAMIC,
  EN_TASK_TYPE_AIC_AIV_DYNAMIC,
  EN_TASK_TYPE_MIX_L2_AIC_AIV,
  EN_TASK_TYPE_COLLECTION_COMMICATE,   // collection ops
  EN_TASK_TYPE_AICPU,                  // aicpu ops
  EN_TASK_TYPE_AICPU_AUTO,
  EN_TASK_TYPE_RUNTIME_CONTROL,   // runtime ops
  EN_TASK_TYPE_RUNTIME_CONTROL_AUTO,   // runtime ops
  EN_TASK_TYPE_DSA,
  EN_TASK_TYPE_DSA_AUTO,
  EN_TASK_TYPE_RESERVED                 // reserved value
};

enum class OpImplType {
  EN_IMPL_CUSTOM_CONSTANT_CCE = 0,   // custom constant op
  EN_IMPL_CUSTOM_TIK,                // custom tik op
  EN_IMPL_CUSTOM_TBE,                // custom tbe op
  EN_IMPL_HW_CONSTANT_CCE,           // Huawei built-in constant op
  EN_IMPL_HW_GENERAL_CCE,            // Huawei built-in cce op
  EN_IMPL_HW_TIK,                    // Huawei built-in tik op
  EN_IMPL_HW_TBE,                    // Huawei built-in tbe op
  EN_IMPL_RL,                        // RL op
  EN_IMPL_PLUGIN_TBE,                // Huawei built-in tbe plugin op
  EN_IMPL_VECTOR_CORE_HW_TBE,        // Huawei built-in tbe op
  EN_IMPL_VECTOR_CORE_CUSTOM_TBE,    // custom tbe op
  EN_IMPL_NON_PERSISTENT_CUSTOM_TBE, // custom tbe op
  EN_RESERVED                        // reserved value
};

const std::string kFftsSwitch = "_ffts_switch";

extern const uint32_t kManualMode;

extern const uint32_t kAutoMode;

extern const uint32_t kRecuriseCntMax;

extern const std::string kFFTSPlusEngineName;

extern const std::string EM_PARAM;

extern const std::string STR_SOC_VERSION;

extern const std::string EM_INPUT_PARAM_EMPTY;

extern const std::string kThreadId;

extern const std::string kTypeFFTSPlus;

extern const std::string kFFTSPlusCoreName;

extern const std::string kDnnVmAICpu;

extern const std::string kDnnVmAICpuAscend;

extern const std::string kDnnVmGeLocal;

extern const std::string kDnnVmRts;

extern const std::string kAIcoreEngineName;

extern const std::string kDsaCoreEngineName;

extern const std::string kHcclSubTasks;

extern const std::string kCtxIdList;

extern const std::string kCachePersistSize;

extern const std::string kCachePersist;

extern const std::string kFFTSMode;

extern const std::string kFFTSPlus;

extern const std::string kFFTS;

extern const std::string kSocInfo;

extern const std::string ATTR_NAME_PARENT_NODE;

extern const std::string kContextId;

extern const std::string kSuccList;

extern const std::string kSuccListList;

extern const std::string kNonEdgeSuccList;

extern const std::string kNonEdgePreNodes;

extern const std::string kHcclOutDegree0Num;

extern const std::string kPrefetchEnableBm;

extern const std::string kWriteBackBm;

extern const std::string kInvalidateBm;

extern const std::string kAutoInlabelCtxId;

extern const std::string kAutoCtxIdList;

extern const std::string kAutoAtStartCtxIdList;

extern const std::string kAutoOutlabelCtxId;

extern const std::string kAutoAtEndCtxIdList;

extern const std::string kAutoAtEndPreCnt;

extern const std::string kTypePhonyConcat;

extern const std::string kTypePhonyReduce;

extern const std::string kAttrAICoreCtxDef;

extern const std::string kAttrDsaCtxDef;

extern const std::string kAttrAICpuCtxDef;

extern const std::string kAttrAICoreCtxType;

extern const std::string kAtomicAddrClean;

extern const char *kCoreTypeAIC;

extern const char *kCoreTypeAIV;

extern const char *kCoreTypeMixAIC;

extern const char *kCoreTypeMixAIV;

extern const char *kTensorCtxId;

extern const std::string kAttrThreadCoreType;

extern const std::string kAdjacencyList;

extern const std::string kModeInArgsFirstField;

extern const std::string kAttrIntercoreSync;

extern const std::string kRtsFftsPlusOpStoreName;

extern const std::string kFFTSPlusInDynamic;

extern const std::string ATTR_NAME_ALIAS_ENGINE_NAME;

extern const uint32_t kDefaultWindowSize;

extern const uint32_t kDefaultManualWindowSize;

extern const uint32_t kGetFirstAvailableLabel;

extern const size_t kMaxPretchNum;

extern const uint32_t kMaxIdx;

extern const uint32_t kMaxPersistNum;

extern const std::map<ge::DataType, uint32_t> DATATYPE_SIZE_MAP;

extern const std::string kCoreTypeHcomReduce;

extern const std::string kCoreTypeHcomAllReduce;

extern const std::string kCoreTypeHcomAllGather;

extern const std::string kCoreTypeHcomBroadCast;

extern const std::string kCoreTypeHcomReduceScatter;

extern const std::string kCoreTypeHcomSend;

extern const std::string kCoreTypeHcomReceive;

extern const std::string kCoreTypeHcomRemoteRead;

extern const std::string kCoreTypeHcomRemoteRefRead;

extern const std::string kCoreTypeHcomRemoteWrite;

extern const std::string kCoreTypeHcomRemoteScatterWrite;

extern const std::string kCoreTypeHcomAllToAllV;

extern const std::string kCoreTypeHcomGatherAllToAllV;

extern const std::unordered_set<std::string> kHCCLOpType;

extern const std::unordered_set<std::string> CONTROL_OP_V2_TYPE;
extern const std::unordered_set<std::string> NO_NEED_GEN_TASK_OP_TYPE;
extern const std::unordered_set<std::string> kPhonyTypes;
extern const std::unordered_set<std::string> COND_SWITCH_NODE_TYPE;
extern const std::unordered_set<std::string> CASE_SWITCH_NODE_TYPE;
extern const std::unordered_set<std::string> SDMA_NODE_TYPE;
extern const std::unordered_set<std::string> LABEL_NODE_TYPE;
extern const std::map<std::string, TaskBuilderType> kDsaOrAicpuMap;
extern const std::string kAutoSuffix;
extern const std::string PLACEHOLDER;
extern const std::string REFDATA;

extern const std::string END;

extern const std::string DATA;

extern const std::string NETOUTPUT;

extern const std::string RESHAPE;

extern const std::string VARIABLE;

extern const std::string TRANSDATA;

extern const std::string TRANSDATARNN;

extern const std::string CAST;

extern const std::string ANN_DATA;

extern const std::string AIPPDATA;

extern const std::string AIPP;

extern const std::string CONSTANTOP;

extern const std::string CONSTANT;

extern const std::string GETNEXT;

extern const std::string CONV2D;

extern const std::string MATMULV2OP;

extern const std::string DEPTHWISECONV2D;

extern const std::string POOLING;

extern const std::string ELTWISE;

extern const std::string RELU;

extern const std::string RELU6;

extern const std::string LEAKY_RELU;

extern const std::string READ_SELECT;

extern const std::string NEXT_ITERATION;

extern const std::string EQUAL;

extern const std::string NOEQUAL;

extern const std::string GREATE;

extern const std::string GREATEOREQUAL;

extern const std::string LESS;

extern const std::string LESSOREQUAL;

extern const std::string ATTR_NAME_INPUT_IS_VAR;

extern const std::string ATTR_NAME_LABEL_JUMP_NODE_INDEX;
extern const std::string ATTR_NAME_LABEL_JUMP_NODE;
extern const std::string ATTR_NAME_LABEL_JUMP_NODES;
extern const std::string ATTR_NAME_LASTLABELSET_OUT_NODES;
extern const std::string ATTR_NAME_LABELSET_PRE_LABEL;
extern const std::string ATTR_NAME_PARENT_OUTPUTS_INPUT_NODES;
extern const std::string ATTR_NAME_PARENT_OUTPUTS_OUTPUT_NODE;
extern const std::string ATTR_NAME_PARENT_PRE_NODES;
extern const std::string ATTR_NAME_LABEL_SWITCH_INDEX;
extern const std::string ATTR_NAME_LABEL_SWITCH_LIST;
extern const std::string ATTR_NAME_HAS_PARTITION;
extern const std::string kRuntimeContentx;

extern const std::unordered_set<std::string> COND_SWITCH_NODE_TYPE;
extern const std::unordered_set<std::string> CASE_SWITCH_NODE_TYPE;
extern const std::unordered_set<std::string> SDMA_NODE_TYPE;
extern const std::unordered_set<std::string> LABEL_NODE_TYPE;

extern const std::string kFftsFirstOpName;
extern const int64_t kMaxCacheSize;
extern const std::string PARTITIONEDCALL;

extern const char *kOnlyAddContext;
extern const std::string kAtomicCtxIdList;
extern const std::string kAttrRefPortIndex;
extern const std::string kFftsFirstSliceFlag;

extern const std::string kFFTSThreadDim;
extern const std::string kFFTSWindowSize;
extern const std::string kFftsFlagDefault;
extern const std::string kFftsFlagAllSubgraph;
}
#endif // FFTS_ENGINE_INC_FFTS_TYPE_H
