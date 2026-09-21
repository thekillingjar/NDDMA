/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engines/manager/engine/engine_manager.h"

#include <map>
#include <string>
#include <utility>

#include "common/plugin/ge_make_unique_util.h"
#include "securec.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/string_util.h"
#include "engines/manager/engine/dnnengines.h"
#include "common/ge_common/ge_types.h"
#include "base/err_msg.h"

namespace ge {
std::unique_ptr<std::map<std::string, DNNEnginePtr>> EngineManager::engine_map_;

Status EngineManager::RegisterEngine(const std::string &engine_name, DNNEnginePtr engine_ptr) {
  if (engine_ptr == nullptr) {
    GELOGE(FAILED, "[Register][Engine] failed, as input engine_ptr is nullptr");
    REPORT_INNER_ERR_MSG("E19999", "RegisterEngine failed for input engine_ptr is nullptr.");
    return FAILED;
  }

  if (engine_map_ == nullptr) {
    engine_map_.reset(new (std::nothrow) std::map<std::string, DNNEnginePtr>());
  }

  if (engine_map_->find(engine_name) != engine_map_->cend()) {
    GELOGW("engine %s already exist.", engine_name.c_str());
    return FAILED;
  }
  engine_map_->emplace(engine_name, engine_ptr);
  return SUCCESS;
}

DNNEnginePtr EngineManager::GetEngine(const std::string &engine_name) {
  const std::map<std::string, DNNEnginePtr>::const_iterator it = engine_map_->find(engine_name);
  if (it == engine_map_->cend()) {
    GELOGW("engine %s does not exist.", engine_name.c_str());
    return nullptr;
  }

  auto engine = it->second;
  return engine;
}

void RegisterAiCoreEngine() {
  const std::string ai_core = "AIcoreEngine";
  std::vector<std::string> mem_type_aicore;
  mem_type_aicore.emplace_back(GE_ENGINE_ATTR_MEM_TYPE_HBM);
  DNNEngineAttribute attr_aicore = { ai_core,
                                     mem_type_aicore,
                                     PriorityEnum::COST_1,
                                     DEVICE,
                                     FORMAT_RESERVED,
                                     FORMAT_RESERVED,
                                     true };
  DNNEnginePtr aicore_engine_ptr = MakeShared<AICoreDNNEngine>(attr_aicore);
  if (aicore_engine_ptr == nullptr) {
    GELOGE(ge::FAILED, "[Register][AiCoreEngine] failed, as malloc shared_ptr failed.");
    REPORT_INNER_ERR_MSG("E19999", "RegisterAiCoreEngine failed for new DNNEnginePtr failed.");
    return;
  }
  if (EngineManager::RegisterEngine(ai_core, aicore_engine_ptr) != SUCCESS) {
    GELOGW("register ai_core failed");
  }
}

void RegisterVectorEngine() {
  const std::string vector_core = "VectorEngine";
  std::vector<std::string> mem_type_aivcore;
  mem_type_aivcore.emplace_back(GE_ENGINE_ATTR_MEM_TYPE_HBM);
  DNNEngineAttribute attr_vector_core = { vector_core,
                                          mem_type_aivcore,
                                          PriorityEnum::COST_2,
                                          DEVICE,
                                          FORMAT_RESERVED,
                                          FORMAT_RESERVED,
                                          true };
  DNNEnginePtr vectorcore_engine_ptr = MakeShared<VectorCoreDNNEngine>(attr_vector_core);
  if (vectorcore_engine_ptr == nullptr) {
    GELOGE(ge::FAILED, "[Register][VectorEngine] failed, as malloc shared_ptr failed.");
    REPORT_INNER_ERR_MSG("E19999", "RegisterVectorEngine failed for new DNNEnginePtr failed.");
    return;
  }
  if (EngineManager::RegisterEngine(vector_core, vectorcore_engine_ptr) != SUCCESS) {
    GELOGW("register vector_core failed");
  }
}

void RegisterDsaEngine() {
  const std::string dsa = "DSAEngine";
  std::vector<std::string> mem_type_dsa;
  mem_type_dsa.emplace_back(GE_ENGINE_ATTR_MEM_TYPE_HBM);
  const DNNEngineAttribute attr_dsa = { dsa,
                                        mem_type_dsa,
                                        PriorityEnum::COST_2,
                                        DEVICE,
                                        FORMAT_RESERVED,
                                        FORMAT_RESERVED,
                                        true };

  DNNEnginePtr dsa_engine_ptr = MakeShared<VectorCoreDNNEngine>(attr_dsa);
  if (dsa_engine_ptr == nullptr) {
    GELOGE(ge::FAILED, "[Register][DSAEngine] failed, as malloc shared_ptr failed.");
    REPORT_INNER_ERR_MSG("E19999", "RegisterDSAEngine failed for new DNNEnginePtr failed.");
    return;
  }
  if (EngineManager::RegisterEngine(dsa, dsa_engine_ptr) != SUCCESS) {
    GELOGW("register dsa failed");
  }
}

void RegisterDvppEngine() {
  const std::string vm_dvpp = "DNN_VM_DVPP";
  std::vector<std::string> mem_type_dvpp;
  mem_type_dvpp.emplace_back(GE_ENGINE_ATTR_MEM_TYPE_HBM);

  const DNNEngineAttribute attr_dvpp = { vm_dvpp,
                                         mem_type_dvpp,
                                         PriorityEnum::COST_5,
                                         DEVICE,
                                         FORMAT_RESERVED,
                                         FORMAT_RESERVED,
                                         true };

  const DNNEnginePtr vm_engine_ptr = MakeShared<DvppDNNEngine>(attr_dvpp);
  if (vm_engine_ptr == nullptr) {
    GELOGE(ge::FAILED, "[Register][DvppEngine] failed, as malloc shared_ptr failed.");
    REPORT_INNER_ERR_MSG("E19999", "RegisterDvppEngine failed for new DNNEnginePtr failed.");
    return;
  }
  if (EngineManager::RegisterEngine(vm_dvpp, vm_engine_ptr) != SUCCESS) {
    GELOGW("register vmDvppEngine failed");
  }
}

void RegisterAiCpuEngine() {
  const std::string vm_aicpu = "DNN_VM_AICPU_ASCEND";
  std::vector<std::string> mem_type_aicpu;
  mem_type_aicpu.emplace_back(GE_ENGINE_ATTR_MEM_TYPE_HBM);

  DNNEngineAttribute attr_aicpu = { vm_aicpu,
                                    mem_type_aicpu,
                                    PriorityEnum::COST_3,
                                    DEVICE,
                                    FORMAT_RESERVED,
                                    FORMAT_RESERVED,
                                    true };

  DNNEnginePtr vm_engine_ptr = MakeShared<AICpuDNNEngine>(attr_aicpu);
  if (vm_engine_ptr == nullptr) {
    GELOGE(ge::FAILED, "[Register][AiCpuEngine] failed, as malloc shared_ptr failed.");
    REPORT_INNER_ERR_MSG("E19999", "RegisterAiCpuEngine failed for new DNNEnginePtr failed.");
    return;
  }
  if (EngineManager::RegisterEngine(vm_aicpu, vm_engine_ptr) != SUCCESS) {
    GELOGW("register vmAicpuEngine failed");
  }
}

void RegisterAiCpuTFEngine() {
  const std::string vm_aicpu_tf = "DNN_VM_AICPU";
  std::vector<std::string> mem_type_aicpu_tf;
  mem_type_aicpu_tf.emplace_back(GE_ENGINE_ATTR_MEM_TYPE_HBM);

  DNNEngineAttribute attr_aicpu_tf = { vm_aicpu_tf,
                                       mem_type_aicpu_tf,
                                       PriorityEnum::COST_4,
                                       DEVICE,
                                       FORMAT_RESERVED,
                                       FORMAT_RESERVED,
                                       true };
  DNNEnginePtr vm_engine_ptr = MakeShared<AICpuTFDNNEngine>(attr_aicpu_tf);
  if (vm_engine_ptr == nullptr) {
    GELOGE(ge::FAILED, "[Register][AiCpuTFEngine]make vm_engine_ptr failed");
    REPORT_INNER_ERR_MSG("E19999", "RegisterAiCpuTFEngine failed for new DNNEnginePtr failed.");
    return;
  }
  if (EngineManager::RegisterEngine(vm_aicpu_tf, vm_engine_ptr) != SUCCESS) {
    GELOGW("register vmAicpuTFEngine failed");
  }
}

void RegisterAICpuAscendFftsPlusEngine() {
  const std::string vm_aicpu_ascend_ffts_plus = "DNN_VM_AICPU_ASCEND_FFTS_PLUS";
  std::vector<std::string> mem_type_aicpu_ascend_ffts_plus;
  mem_type_aicpu_ascend_ffts_plus.emplace_back(GE_ENGINE_ATTR_MEM_TYPE_HBM);
  DNNEngineAttribute attr_aicpu_ascend_ffts_plus = { vm_aicpu_ascend_ffts_plus,
                                                     mem_type_aicpu_ascend_ffts_plus,
                                                     PriorityEnum::COST_2,
                                                     DEVICE,
                                                     FORMAT_RESERVED,
                                                     FORMAT_RESERVED,
                                                     true };
  DNNEnginePtr aicpu_ascend_ffts_plus_engine = MakeShared<AICpuAscendFftsPlusDNNEngine>(attr_aicpu_ascend_ffts_plus);
  if (aicpu_ascend_ffts_plus_engine == nullptr) {
    GELOGE(ge::FAILED, "[Register][AiCpuAscendFftsPlusDNNEngine] failed, as malloc shared_ptr failed");
    REPORT_INNER_ERR_MSG("E19999", "Register AiCpuAscendFftsPlusDNNEngine failed for new DNNEnginePtr failed.");
    return;
  }
  if (EngineManager::RegisterEngine(vm_aicpu_ascend_ffts_plus, aicpu_ascend_ffts_plus_engine) != SUCCESS) {
    GELOGW("register aicpu_ascend_ffts_plus_engine failed");
  }
}

void RegisterAICpuFftsPlusEngine() {
  const std::string vm_aicpu_ffts_plus = "DNN_VM_AICPU_FFTS_PLUS";
  std::vector<std::string> mem_type_aicpu_ffts_plus;
  mem_type_aicpu_ffts_plus.emplace_back(GE_ENGINE_ATTR_MEM_TYPE_HBM);
  DNNEngineAttribute attr_aicpu_ffts_plus = { vm_aicpu_ffts_plus,
                                              mem_type_aicpu_ffts_plus,
                                              PriorityEnum::COST_2,
                                              DEVICE,
                                              FORMAT_RESERVED,
                                              FORMAT_RESERVED,
                                              true };
  DNNEnginePtr aicpu_ffts_plus_engine = MakeShared<AICpuFftsPlusDNNEngine>(attr_aicpu_ffts_plus);
  if (aicpu_ffts_plus_engine == nullptr) {
    GELOGE(ge::FAILED, "[Register][AiCpuFftsPlusDNNEngine] failed, as malloc shared_ptr failed");
    REPORT_INNER_ERR_MSG("E19999", "Register AiCpuFftsPlusDNNEngine failed for new DNNEnginePtr failed.");
    return;
  }
  if (EngineManager::RegisterEngine(vm_aicpu_ffts_plus, aicpu_ffts_plus_engine) != SUCCESS) {
    GELOGW("register aicpu_ffts_plus_engine failed");
  }
}


void RegisterGeLocalEngine() {
  const std::string vm_ge_local = "DNN_VM_GE_LOCAL";
  std::vector<std::string> mem_type_ge_local;
  mem_type_ge_local.emplace_back(GE_ENGINE_ATTR_MEM_TYPE_HBM);
  // GeLocal use minimum priority, set it as 9
  DNNEngineAttribute attr_ge_local = { vm_ge_local,
                                       mem_type_ge_local,
                                       PriorityEnum::COST_9,
                                       DEVICE,
                                       FORMAT_RESERVED,
                                       FORMAT_RESERVED,
                                       true };
  DNNEnginePtr ge_local_engine = MakeShared<GeLocalDNNEngine>(attr_ge_local);
  if (ge_local_engine == nullptr) {
    GELOGE(ge::FAILED, "[Register][GeLocalEngine] failed, as malloc shared_ptr failed.");
    REPORT_INNER_ERR_MSG("E19999", "RegisterGeLocalEngine failed for new DNNEnginePtr failed.");
    return;
  }
  if (EngineManager::RegisterEngine(vm_ge_local, ge_local_engine) != SUCCESS) {
    GELOGW("register ge_local_engine failed");
  }
}

void RegisterHostCpuEngine() {
  const std::string vm_host_cpu = "DNN_VM_HOST_CPU";
  std::vector<std::string> mem_type_host_cpu;
  mem_type_host_cpu.emplace_back(GE_ENGINE_ATTR_MEM_TYPE_HBM);
  // HostCpu use minimum priority, set it as 10
  DNNEngineAttribute attr_host_cpu = { vm_host_cpu,
                                       mem_type_host_cpu,
                                       PriorityEnum::COST_10,
                                       HOST,
                                       FORMAT_RESERVED,
                                       FORMAT_RESERVED,
                                       true };
  DNNEnginePtr host_cpu_engine = MakeShared<HostCpuDNNEngine>(attr_host_cpu);
  if (host_cpu_engine == nullptr) {
    GELOGE(ge::FAILED, "[Register][HostCpuEngine] failed, as malloc shared_ptr failed.");
    REPORT_INNER_ERR_MSG("E19999", "RegisterHostCpuEngine failed for new DNNEnginePtr failed.");
    return;
  }
  if (EngineManager::RegisterEngine(vm_host_cpu, host_cpu_engine) != SUCCESS) {
    GELOGW("register host_cpu_engine failed");
  }
}

void RegisterRtsEngine() {
  const std::string vm_rts = "DNN_VM_RTS";
  std::vector<std::string> mem_type_rts;
  mem_type_rts.emplace_back(GE_ENGINE_ATTR_MEM_TYPE_HBM);
  DNNEngineAttribute attr_rts = { vm_rts,
                                  mem_type_rts,
                                  PriorityEnum::COST_2,
                                  DEVICE,
                                  FORMAT_RESERVED,
                                  FORMAT_RESERVED,
                                  true };
  DNNEnginePtr rts_engine = MakeShared<RtsDNNEngine>(attr_rts);
  if (rts_engine == nullptr) {
    GELOGE(ge::FAILED, "[Register][RtsEngine] failed, as malloc shared_ptr failed.");
    REPORT_INNER_ERR_MSG("E19999", "RegisterRtsEngine failed for new DNNEnginePtr failed.");
    return;
  }
  if (EngineManager::RegisterEngine(vm_rts, rts_engine) != SUCCESS) {
    GELOGW("register rts_engine failed");
  }
}

void RegisterRtsFftsPlusEngine() {
  const std::string vm_rts_ffts_plus = "DNN_VM_RTS_FFTS_PLUS";
  std::vector<std::string> mem_type_rts_ffts_plus;
  mem_type_rts_ffts_plus.emplace_back(GE_ENGINE_ATTR_MEM_TYPE_HBM);
  DNNEngineAttribute attr_rts_ffts_plus = { vm_rts_ffts_plus,
                                            mem_type_rts_ffts_plus,
                                            PriorityEnum::COST_2,
                                            DEVICE,
                                            FORMAT_RESERVED,
                                            FORMAT_RESERVED,
                                            true };
  DNNEnginePtr rts_ffts_plus_engine = MakeShared<RtsFftsPlusDNNEngine>(attr_rts_ffts_plus);
  if (rts_ffts_plus_engine == nullptr) {
    GELOGE(ge::FAILED, "[Register][RtsFftsPlusDNNEngine] failed, as malloc shared_ptr failed.");
    REPORT_INNER_ERR_MSG("E19999", "Register RtsFftsPlusDNNEngine failed for new DNNEnginePtr failed.");
    return;
  }
  if (EngineManager::RegisterEngine(vm_rts_ffts_plus, rts_ffts_plus_engine) != SUCCESS) {
    GELOGW("register rts_ffts_plus_engine failed");
  }
}

void RegisterHcclEngine() {
  const std::string dnn_hccl = "DNN_HCCL";
  std::vector<std::string> mem_type_hccl;
  mem_type_hccl.emplace_back(GE_ENGINE_ATTR_MEM_TYPE_HBM);
  DNNEngineAttribute attr_hccl = { dnn_hccl,
                                   mem_type_hccl,
                                   PriorityEnum::COST_2,
                                   DEVICE,
                                   FORMAT_RESERVED,
                                   FORMAT_RESERVED,
                                   true };
  DNNEnginePtr hccl_engine = MakeShared<HcclDNNEngine>(attr_hccl);
  if (hccl_engine == nullptr) {
    GELOGE(ge::FAILED, "[Register][HcclEngine] failed, as malloc shared_ptr failed.");
    REPORT_INNER_ERR_MSG("E19999", "RegisterHcclEngine failed for new DNNEnginePtr failed.");
    return;
  }
  if (EngineManager::RegisterEngine(dnn_hccl, hccl_engine) != SUCCESS) {
    GELOGW("register hccl_engine failed");
  }
}

void RegisterFftsPlusEngine() {
  const std::string dnn_ffts_plus = "ffts_plus";
  std::vector<std::string> mem_type_ffts_plus;
  mem_type_ffts_plus.emplace_back(GE_ENGINE_ATTR_MEM_TYPE_HBM);
  DNNEngineAttribute attr_ffts_plus = { dnn_ffts_plus,
                                        mem_type_ffts_plus,
                                        PriorityEnum::COST_1,
                                        DEVICE,
                                        FORMAT_RESERVED,
                                        FORMAT_RESERVED,
                                        false };
  DNNEnginePtr ffts_plus_engine = MakeShared<FftsPlusDNNEngine>(attr_ffts_plus);
  if (ffts_plus_engine == nullptr) {
    GELOGE(ge::FAILED, "[Register][FftsPlusDNNEngine] failed, as malloc shared_ptr failed.");
    REPORT_INNER_ERR_MSG("E19999", "RegisterFftsPlusEngine failed for new DNNEnginePtr failed.");
    return;
  }
  if (EngineManager::RegisterEngine(dnn_ffts_plus, ffts_plus_engine) != SUCCESS) {
    GELOGW("register ffts_plus_engine failed");
  }
}

void RegisterCustomEngine() {
  std::vector<std::string> mem_type;
  mem_type.emplace_back(GE_ENGINE_ATTR_MEM_TYPE_HBM);
  DNNEngineAttribute attr_custom = { kEngineNameCustom, mem_type,
                                     PriorityEnum::COST_0,
                                     DEVICE,
                                     FORMAT_RESERVED,
                                     FORMAT_RESERVED,
                                     true };
  DNNEnginePtr custom_engine_ptr = MakeShared<CustomDNNEngine>(attr_custom);
  if (custom_engine_ptr == nullptr) {
    GELOGE(ge::FAILED, "[Register][CustomEngine] failed, as malloc shared_ptr failed.");
    REPORT_INNER_ERR_MSG("E19999", "RegisterCustomEngine failed for new DNNEnginePtr failed.");
    return;
  }
  if (EngineManager::RegisterEngine(kEngineNameCustom, custom_engine_ptr) != SUCCESS) {
    GELOGW("register custom failed");
  }
}

void GetDNNEngineObjs(std::map<std::string, DNNEnginePtr> &engines) {
  RegisterCustomEngine();
  RegisterAiCoreEngine();
  RegisterVectorEngine();
  RegisterDvppEngine();
  RegisterAiCpuTFEngine();
  RegisterAiCpuEngine();
  RegisterAICpuFftsPlusEngine();
  RegisterAICpuAscendFftsPlusEngine();
  RegisterGeLocalEngine();
  RegisterHostCpuEngine();
  RegisterRtsEngine();
  RegisterRtsFftsPlusEngine();
  RegisterHcclEngine();
  RegisterFftsPlusEngine();
  RegisterDsaEngine();

  for (auto it = EngineManager::engine_map_->cbegin(); it != EngineManager::engine_map_->cend(); ++it) {
    GELOGI("get engine %s from engine plugin.", it->first.c_str());
    engines.emplace(std::pair<std::string, DNNEnginePtr>(it->first, it->second));
  }

  GELOGI("after get engine, engine size: %zu", engines.size());
  return;
}
}  // namespace ge
