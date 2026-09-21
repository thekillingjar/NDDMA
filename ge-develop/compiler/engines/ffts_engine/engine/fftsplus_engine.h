/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFTS_ENGINE_ENGINE_FFTSPLUS_ENGINE_H_
#define FFTS_ENGINE_ENGINE_FFTSPLUS_ENGINE_H_

#include "common/opskernel/ops_kernel_info_store.h"
#include "common/optimizer/graph_optimizer.h"
#include "inc/ffts_error_codes.h"

using OpsKernelInfoStorePtr = std::shared_ptr<ge::OpsKernelInfoStore>;
using GraphOptimizerPtr = std::shared_ptr<ge::GraphOptimizer>;

extern "C" {
/*
 * to invoke the Initialize function of FusionManager
 * param[in] the options of init
 * return Status(SUCCESS/FAILED)
 */
ffts::Status Initialize(const std::map<string, string> &options);
/*
 * to invoke the Finalize function to release the source of fusion manager
 * return Status(SUCCESS/FAILED)
 */
ffts::Status Finalize();
/*
 * to invoke the same-name function of FusionManager to get the information of OpsKernel InfoStores
 * param[out] the map of OpsKernel InfoStores
 */
void GetOpsKernelInfoStores(const std::map<string, OpsKernelInfoStorePtr> &op_kern_infos);

void GetCompositeEngines(std::map<string, std::set<std::string>> &compound_engine_contains,
                         std::map<std::string, std::string> &compound_engine_2_kernel_lib_name);
/*
 * to invoke the same-name function of FusionManager to get the information of Graph Optimizer
 * param[out] the map of Graph Optimizer
 */
void GetGraphOptimizerObjs(std::map<string, GraphOptimizerPtr> &graph_optimizers);

void GetFFTSPlusSwitch(bool &fftsplus_switch);
}
#endif  // FFTS_ENGINE_ENGINE_FFTSPLUS_ENGINE_H_
