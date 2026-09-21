/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "utils/bench_env.h"
#include <iostream>
#include <vector>
#include "runtime/rt.h"
#include "framework/executor/ge_executor.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "single_op/single_op.h"
#include "single_op/single_op_manager.h"
#include "utils/model_data_builder.h"
#include "single_op/task/build_task_utils.h"
#include "utils/tensor_descs.h"
#include "utils/data_buffers.h"
#include "ge_running_env/fake_ns.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "ge/ge_api.h"
#include "register/op_tiling_registry.h"
#include "graph/debug/ge_attr_define.h"
#include "hybrid/node_executor/aicore/aicore_node_executor.h"
#include "hybrid/node_executor/ge_local/ge_local_node_executor.h"
#include "graph/manager/mem_manager.h"
#include "graph/operator_factory_impl.h"


using namespace ge;
using namespace std;
using namespace optiling;
using namespace optiling::utils;
using namespace hybrid;

FAKE_NS_BEGIN

void BenchEnv::Init() {
  OpTilingFuncV2 tilingfun =  [](const ge::Operator &, const OpCompileInfoV2 &, OpRunInfoV2 &) ->bool {
    return true;
  };

  OpTilingFuncV2 tilingfun2 =  [](const ge::Operator &, const OpCompileInfoV2 &, OpRunInfoV2 &run_info) ->bool {
    run_info.SetWorkspaces({16, 16, 16, 16, 16, 16, 16, 16});
    std::string temp("xx");
    run_info.GetAllTilingData() << temp;
    return true;
  };

  auto infer_fun = [](Operator &op) -> graphStatus {
    const char_t *name = "__output0";
    op.UpdateOutputDesc(name, op.GetInputDesc(0));
    return GRAPH_SUCCESS;
  };

  OpTilingRegistryInterf_V2(DATA, tilingfun);
  OpTilingRegistryInterf_V2(ADD, tilingfun);
  OpTilingRegistryInterf_V2(TRANSDATA, tilingfun);
  OpTilingRegistryInterf_V2(TRANSLATE, tilingfun);
  OpTilingRegistryInterf_V2(MATMUL, tilingfun);
  OpTilingRegistryInterf_V2("DynamicAtomicAddrClean", tilingfun);
  OpTilingRegistryInterf_V2("UniqueV2", tilingfun);
  OpTilingRegistryInterf_V2("MyTransdata", tilingfun2);

  REGISTER_OP_TILING_UNIQ_V2(Data, tilingfun, 1);
  REGISTER_OP_TILING_UNIQ_V2(TransData, tilingfun, 1);
  REGISTER_OP_TILING_UNIQ_V2(Translate, tilingfun, 1);
  REGISTER_OP_TILING_UNIQ_V2(MatMul, tilingfun, 1);
  REGISTER_OP_TILING_UNIQ_V2(DynamicAtomicAddrClean, tilingfun, 1);
  REGISTER_OP_TILING_UNIQ_V2(Add, tilingfun, 1);
  REGISTER_OP_TILING_UNIQ_V2(UniqueV2, tilingfun, 1);
  REGISTER_OP_TILING_UNIQ_V2(MyTransdata, tilingfun2, 1);

  OperatorFactoryImpl::RegisterInferShapeFunc(TRANSDATA, infer_fun);
  OperatorFactoryImpl::RegisterInferShapeFunc(ADD, infer_fun);
  OperatorFactoryImpl::RegisterInferShapeFunc(MATMUL, infer_fun);
  OperatorFactoryImpl::RegisterInferShapeFunc("Reshape", infer_fun);
  OperatorFactoryImpl::RegisterInferShapeFunc(RELU, infer_fun);
  OperatorFactoryImpl::RegisterInferShapeFunc("DynamicAtomicAddrClean", infer_fun);
  OperatorFactoryImpl::RegisterInferShapeFunc("MyTransdata", infer_fun);

  REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::AICORE, AiCoreNodeExecutor);
  REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::GE_LOCAL, GeLocalNodeExecutor);
  MemManager::Instance().Initialize({RT_MEMORY_HBM});
}
FAKE_NS_END
