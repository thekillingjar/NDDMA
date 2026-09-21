/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <iostream>
#include "ge_default_running_env.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "graph/utils/op_desc_utils.h"

FAKE_NS_BEGIN
namespace {
auto infer_fun = [](Operator &op) -> graphStatus {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  *op_desc->MutableOutputDesc(0) = *op_desc->GetInputDescPtr(0);
  return GRAPH_SUCCESS;
};
std::vector<FakeEngine> default_engines = {
    FakeEngine("AIcoreEngine").KernelInfoStore("AiCoreLib").GraphOptimizer("AIcoreEngine").Priority(PriorityEnum::COST_0),
    FakeEngine("VectorEngine").KernelInfoStore("VectorLib").GraphOptimizer("VectorEngine").Priority(PriorityEnum::COST_1),
    FakeEngine("DNN_VM_AICPU").KernelInfoStore("AicpuLib").GraphOptimizer("aicpu_tf_optimizer").Priority(PriorityEnum::COST_3),
    FakeEngine("DNN_VM_AICPU_ASCEND").KernelInfoStore("AicpuAscendLib").GraphOptimizer("aicpu_ascend_optimizer").Priority(PriorityEnum::COST_2),
    FakeEngine("DNN_HCCL").KernelInfoStore("ops_kernel_info_hccl").GraphOptimizer("hccl_graph_optimizer").GraphOptimizer("hvd_graph_optimizer").Priority(PriorityEnum::COST_1),
    FakeEngine("DNN_VM_RTS").KernelInfoStore("RTSLib").GraphOptimizer("DNN_VM_RTS_GRAPH_OPTIMIZER_STORE").Priority(PriorityEnum::COST_1),
    FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER").Priority(PriorityEnum::COST_9),
    FakeEngine("DNN_VM_HOST_CPU").KernelInfoStore("DNN_VM_HOST_CPU_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER").Priority(PriorityEnum::COST_10),
    FakeEngine("DSAEngine").KernelInfoStore("DSAEngine").Priority(PriorityEnum::COST_1)
};

std::vector<FakeOp> fake_ops = {
    FakeOp(ENTER).InfoStoreAndBuilder("RTSLib"),
    FakeOp(NETOUTPUT).InfoStoreAndBuilder("AicpuLib"),
    FakeOp(FLOWNODE).InfoStoreAndBuilder("AicpuLib"),
    FakeOp(MERGE).InfoStoreAndBuilder("RTSLib"),
    FakeOp(SWITCH).InfoStoreAndBuilder("RTSLib"),
    FakeOp(SWITCHN).InfoStoreAndBuilder("RTSLib"),
    FakeOp(LOOPCOND).InfoStoreAndBuilder("RTSLib"),
    FakeOp(STREAMMERGE).InfoStoreAndBuilder("RTSLib"),
    FakeOp(STREAMSWITCH).InfoStoreAndBuilder("RTSLib"),
    FakeOp(STREAMACTIVE).InfoStoreAndBuilder("RTSLib"),
    FakeOp(EXIT).InfoStoreAndBuilder("RTSLib"),
    FakeOp(SEND).InfoStoreAndBuilder("RTSLib"),
    FakeOp(SENDNOTIFY).InfoStoreAndBuilder("RTSLib"),
    FakeOp(RECV).InfoStoreAndBuilder("RTSLib"),
    FakeOp(RECVNOTIFY).InfoStoreAndBuilder("RTSLib"),
    FakeOp(IDENTITY).InfoStoreAndBuilder("RTSLib"),
    FakeOp(READVARIABLEOP).InfoStoreAndBuilder("RTSLib"),
    FakeOp(IDENTITYN).InfoStoreAndBuilder("RTSLib"),
    FakeOp(MEMCPYASYNC).InfoStoreAndBuilder("RTSLib"),
    FakeOp(MEMCPYADDRASYNC).InfoStoreAndBuilder("RTSLib"),
    FakeOp(STARTOFSEQUENCE).InfoStoreAndBuilder("RTSLib"),
    FakeOp(NPUGETFLOATSTATUS).InfoStoreAndBuilder("RTSLib"),
    FakeOp(NPUCLEARFLOATSTATUS).InfoStoreAndBuilder("RTSLib"),


    FakeOp(NEG).InfoStoreAndBuilder("AicpuLib"),
    FakeOp(FRAMEWORKOP).InfoStoreAndBuilder("AicpuLib"),
    FakeOp(CONSTANTOP).InfoStoreAndBuilder("AicpuLib"),
    FakeOp(GETDYNAMICDIMS).InfoStoreAndBuilder("AicpuLib"),

    FakeOp(DROPOUTGENMASK).InfoStoreAndBuilder("AicpuLib"),
    FakeOp(DROPOUTDOMASK).InfoStoreAndBuilder("AicpuLib"),
    FakeOp(HCOMALLREDUCE).InfoStoreAndBuilder("ops_kernel_info_hccl"),
    FakeOp(HCOMREDUCESCATTER).InfoStoreAndBuilder("ops_kernel_info_hccl"),

    FakeOp(RESOURCEAPPLYMOMENTUM).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(ASSIGNVARIABLEOP).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(GATHERV2).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp("MatMulV2").InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(MATMUL).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(LESS).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(NEXTITERATION).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(CAST).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(TRANSDATA).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(TRANSPOSE).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(NOOP).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(VARIABLE).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(CONSTANT).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(ASSIGN).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(TENSORMOVE).InfoStoreAndBuilder("AiCoreLib").InferShape(infer_fun),
    FakeOp(ASSIGNADD).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(ADD).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(ADDN).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(SUB).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp("Abs").InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(POW).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(SQRT).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(SQUARE).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(MUL).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"),
    FakeOp(AIPP).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(AIPPDATA).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(QUEUE_DATA).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(NETOUTPUT).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(CONV2D).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(CONCAT).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(CONCATV2).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(RELU).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(PRELU).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp("SplitD").InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(SLICE).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp("SliceDV2").InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(PACK).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(RELU6GRAD).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(APPLYMOMENTUM).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(HCOMALLREDUCE).InfoStoreAndBuilder("ops_kernel_info_hccl"),
    FakeOp(HCOMALLGATHER).InfoStoreAndBuilder("ops_kernel_info_hccl"),
    FakeOp(HCOMBROADCAST).InfoStoreAndBuilder("ops_kernel_info_hccl"),
    FakeOp(HVDCALLBACKBROADCAST).InfoStoreAndBuilder("ops_kernel_info_hccl"),
    FakeOp(HCOMALLTOALL).InfoStoreAndBuilder("ops_kernel_info_hccl"),
    FakeOp(REALDIV).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(FILECONSTANT).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(RESHAPE).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(VARIABLEV2).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"),
    FakeOp(CONTROLTRIGGER).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"),
    FakeOp("GetShape").InfoStoreAndBuilder("AiCoreLib"),
    FakeOp("MapIndex").InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(CASE).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(SPLIT).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp("UpdateTensorDesc").InfoStoreAndBuilder("AiCoreLib"),
    FakeOp("LabelSet").InfoStoreAndBuilder("RTSLib"),
    FakeOp("LabelSwitchByIndex").InfoStoreAndBuilder("RTSLib"),
    FakeOp("LabelGotoEx").InfoStoreAndBuilder("RTSLib"),
    FakeOp("LabelGotoEx").InfoStoreAndBuilder("RTSLib"),

    FakeOp(REFIDENTITY).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"),
    FakeOp("RefData").InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"),
    FakeOp(CONSTPLACEHOLDER).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"),
    FakeOp(SHAPE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"),
    FakeOp(RANK).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"),
    FakeOp(SHAPEN).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"),
    FakeOp(UNSQUEEZE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"),
    FakeOp(UNSQUEEZEV2).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"),
    FakeOp(UNSQUEEZEV3).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"),
    FakeOp(BITCAST).InferShape(nullptr).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"), // Bitcast使用rt2形式Infershape
    FakeOp(EXPANDDIMS).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"),
    FakeOp(STACK).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"),
    FakeOp(STACKPUSH).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"),
    FakeOp(STACKPOP).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"),
    FakeOp(STACKCLOSE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"),
    FakeOp(CONSTANTOP).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"),
    FakeOp(IF).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"),
    FakeOp(FOR).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"),
    FakeOp(WHILE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"),
    FakeOp("PhonyConcat").InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"),
    FakeOp("PhonySplit").InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"),
    FakeOp(ATOMICADDRCLEAN).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(MEMSET).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(HCOMSEND).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(HCOMRECEIVE).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(HCOMRECEIVE).InfoStoreAndBuilder("DNN_VM_HOST_CPU_OP_STORE"),
    FakeOp(END).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(END).InfoStoreAndBuilder("DNN_VM_HOST_CPU_OP_STORE"),
    FakeOp(PLACEHOLDER).InfoStoreAndBuilder("AiCoreLib"),
    FakeOp(PLACEHOLDER).InfoStoreAndBuilder("DNN_VM_HOST_CPU_OP_STORE"),
    FakeOp(SLICE).InfoStoreAndBuilder("DNN_VM_HOST_CPU_OP_STORE"),
    FakeOp(SLICEWITHAXES).InfoStoreAndBuilder("DNN_VM_HOST_CPU_OP_STORE"),
    FakeOp(CONCATV2).InfoStoreAndBuilder("DNN_VM_HOST_CPU_OP_STORE"),
    FakeOp("PSOP").InfoStoreAndBuilder("DNN_VM_HOST_CPU_OP_STORE"),
    FakeOp(SQUARE).InfoStoreAndBuilder("DNN_VM_HOST_CPU_OP_STORE"),
    FakeOp(ABSVAL).InfoStoreAndBuilder("DNN_VM_HOST_CPU_OP_STORE"),
    FakeOp(SQRT).InfoStoreAndBuilder("DNN_VM_HOST_CPU_OP_STORE"),
    FakeOp(ADD).InfoStoreAndBuilder("DNN_VM_HOST_CPU_OP_STORE"),
    FakeOp("DSARandomNormal").InfoStoreAndBuilder("DSAEngine"),
    FakeOp("MoeFFN").InfoStoreAndBuilder("AiCoreLib")
};
}  // namespace

void GeDefaultRunningEnv::InstallTo(GeRunningEnvFaker& ge_env) {
  for (auto& fake_engine : default_engines) {
    ge_env.Install(fake_engine);
  }

  for (auto& fake_op : fake_ops) {
    ge_env.Install(fake_op);
  }
}

FAKE_NS_END
