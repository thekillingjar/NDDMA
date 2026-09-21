/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/fe_running_env/include/fe_running_env.h"
#include "framework/fe_running_env/include/fe_env_utils.h"
#include "framework/fe_running_env/include/mock_fftsplus_ops_kernel_builder.h"
#include "graph/operator_factory.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/tensor_utils.h"
#include "stub/te_fusion_api.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "common/util/op_info_util.h"
#include "inc/ffts_type.h"
#include "engine/engine_manager.h"
#include "register/ops_kernel_builder_registry.h"
#include <securec.h>
#include <unistd.h>
#include <cstdio>

const uint32_t kInitGraphCount = 1;
const uint32_t kNotAdded = 0;
const uint32_t kStartAdd = 1;
const uint32_t kDoneAdded = 2;

namespace fe_env {
FeRunningEnv::FeRunningEnv() :
    ops_kernel_info_(const_cast<std::map<std::string, std::vector<ge::OpInfo>>&>(
        ge::OpsKernelManager::GetInstance().GetAllOpsKernelInfo())),
    ops_kernel_info_stores_(const_cast<std::map<std::string, OpsKernelInfoStorePtr>&>(
        ge::OpsKernelManager::GetInstance().GetAllOpsKernelInfoStores())),
    ops_kernel_optimizers_(const_cast<std::map<std::string, GraphOptimizerPtr>&>(
        ge::OpsKernelManager::GetInstance().GetAllGraphOptimizerObjs())),
    atomic_first_optimizers_by_priority_(const_cast<std::vector<std::pair<std::string, GraphOptimizerPtr>>&>(
        ge::OpsKernelManager::GetInstance().GetAllGraphOptimizerObjsByPriority())),
    composite_engines_(const_cast<std::map<std::string, std::set<std::string>>&>(
        ge::OpsKernelManager::GetInstance().GetCompositeEngines())),
    composite_engine_kernel_lib_names_(const_cast<std::map<std::string, std::string>&>(
        ge::OpsKernelManager::GetInstance().GetCompositeEngineKernelLibNames())) {

}
std::map<std::string, std::vector<domi::TaskDef>> FeRunningEnv::tasks_map;
FeRunningEnv::~FeRunningEnv() {

}

FeRunningEnv &FeRunningEnv::Instance() {
  static FeRunningEnv fe_env;
  return fe_env;
}

string FeRunningEnv::GetNetworkPath(const string &network_name) {
  char_t env_value[MMPA_MAX_PATH];
  mmGetEnv("FE_ST_PATH", env_value, MMPA_MAX_PATH);
  return string(env_value) + "/net/" + network_name;
}

/* 在Windows的linux中，有一个和socket有关的问题暂时没有办法解决；
 * 导致tefusion的并行编译会失败。当前如果本地跑st需要采用打桩的方
 * 式实现。这个问题在真实的ubuntu环境是没有的，所以本地跑的时候需要
 * 设置一个TE_PARALLEL_COMPILER=-1的环境变量。
 * */
std::map<string, void(*)> kDlsymFuncLibWsl = {
    // {"PreBuildTbeOp", (void*)te::PreBuildTbeOp},
    {"TbeFinalize", (void*)te::TbeFinalize},
    {"TbeInitialize", (void*)te::TbeInitialize},
    {"TeFusion", (void*)te::TeFusion},
    {"TeFusionV", (void*)te::TeFusionV},
    // {"FuzzBuildTbeOp", (void*)te::FuzzBuildTbeOp},
    {"WaitAllFinished", (void*)te::WaitAllFinished},
};

class MockMmpa : public fe::MmpaStubApi {
 public:
  void *DlSym(void *handle, const char *func_name) override {
    FE_LOGD("DlSym func name %s.", func_name);

    string key = func_name;
    auto iter = kDlsymFuncLibWsl.find(key);

    if (iter == kDlsymFuncLibWsl.end()) {
      FE_LOGD("Not found %s.", key.c_str());
      return dlsym(handle, func_name);
    } else {
      if (strcmp(func_name, "TbeInitialize") == 0) {
        te::TeStub::GetInstance().tbe_initialize_ = dlsym(handle, func_name);
      }
      if (strcmp(func_name, "WaitAllFinished") == 0) {
        te::TeStub::GetInstance().wait_all_finished_ = dlsym(handle, func_name);
      }
      if (strcmp(func_name, "PreBuildTbeOp") == 0) {
        te::TeStub::GetInstance().real_prebuild_func_ = dlsym(handle, func_name);
      }
      if (strcmp(func_name, "TeFusion") == 0) {
        te::TeStub::GetInstance().te_fusion_ = dlsym(handle, func_name);
      }
      if (strcmp(func_name, "TeFusionV") == 0) {
        te::TeStub::GetInstance().te_fusion_v_ = dlsym(handle, func_name);
      }
      FE_LOGD("Found %s.", key.c_str());
      return iter->second;
    }
  }

  int32_t DlAddr(VOID *addr, mmDlInfo *info) override {
    ++count;
    FE_LOGD("load so count=%d", count);
    string ascend_path = FeEnvUtils::GetAscendPath();
    string plugin_path;
    plugin_path = ascend_path + "/compiler/lib64/plugin";
    info->dli_fname = plugin_path.c_str();
    return 1;
  }
  private:
    int32_t count = 0;
};

void FeRunningEnv::InitBasicOptions(const std::map<string, string> &options) {
  options_.clear();
  ascend_options_.clear();
  options_[ge::SOC_VERSION] = "Ascend910";
  options_[ge::AICORE_NUM] = "32";
  options_[ge::CORE_TYPE] = fe::AI_CORE_TYPE;
  options_[ge::PRECISION_MODE] = fe::ALLOW_FP32_TO_FP16;

  ge::AscendString key = ge::AscendString(ge::SOC_VERSION.c_str());
  ascend_options_[key] = "Ascend910";

  key = ge::AscendString(ge::AICORE_NUM.c_str());
  ascend_options_[key] = "32";

  key = ge::AscendString(ge::CORE_TYPE.c_str());
  ascend_options_[key] = fe::AI_CORE_TYPE;

  key = ge::AscendString(ge::PRECISION_MODE.c_str());
  ascend_options_[key] = fe::ALLOW_FP32_TO_FP16.c_str();

  for (const auto &option : options) {
    options_[option.first] = option.second;

    key = ge::AscendString(option.first.c_str());
    ascend_options_[key] = option.second.c_str();
  }
}

void FeRunningEnv::InjectFe() {
  /* 在GEInitialize中我们是没有加载FE的，在上面的Initialize中，我们初始化了我们st代码的fe。
   * 然后需要把这个fe注入到GE的OpsKernelManager中。 */
  GetOpsKernelInfoStores(ops_kernel_info_stores_);
  ge::OpsKernelManager::GetInstance().InitOpsKernelInfo();
  FE_LOGD("[FE_ST]Start get optimizers. Size [%zu, %zu, %zu].", ops_kernel_optimizers_.size(),
          atomic_first_optimizers_by_priority_.size(), composite_engines_.size());
  GetGraphOptimizerObjs(ops_kernel_optimizers_);

  for (const auto &optimizer : ops_kernel_optimizers_) {
    optimizer.second->Initialize(options_, &(ge::OpsKernelManager::GetInstance().graph_optimize_utility_));
  }
  ge::OpsKernelManager::GetInstance().atomic_graph_optimizers_by_priority_.clear();
  ge::OpsKernelManager::GetInstance().ClassifyGraphOptimizers();
  ge::OpsKernelManager::GetInstance().InitGraphOptimizerPriority();
  FE_LOGD("[FE_ST]After classify. Size [%zu, %zu, %zu].", ops_kernel_optimizers_.size(),
          atomic_first_optimizers_by_priority_.size(), composite_engines_.size());
}

void FeRunningEnv::InjectFFTS() {
  // GEInitialize中初始化的是通过libffts.so加载进来的地址，这里需要替换成st编译进来的engine_manager.cc重新初始化
  // 并替换掉libffts.so加载进来的函数地址
  // 注意这里不能包含fftsplus_engine.h来初始化，会与itf_handler.h冲突
  FE_LOGD("[FFTS_ST]Start get optimizers. Size [%zu, %zu, %zu].", ops_kernel_optimizers_.size(),
          atomic_first_optimizers_by_priority_.size(), composite_engines_.size());
  // get socversion
  auto iter_soc = options_.find(ge::SOC_VERSION);
  if (iter_soc == options_.end()) {
    FE_LOGW("cannot find soc version");
    return;
  }
  if (iter_soc->second.find("Ascend910B") == string::npos) {
    FE_LOGD("[FFTS_ST] soc version:%s no need inject ffts", iter_soc->second.c_str());
    return;
  }
  std::vector<std::string> engine_name_list = {ffts::kFFTSPlusEngineName};
  for (auto &engine_name : engine_name_list) {
    FE_LOGD("[FFTS_ST] Start to initialize in engine [%s]", engine_name.c_str());
    ffts::EngineManager &em = ffts::EngineManager::Instance(engine_name);
    ffts::Status ret = em.Initialize(options_, engine_name, iter_soc->second);
    if (ret != ffts::SUCCESS) {
      em.Finalize();
      return;
    }
  }
  // replace ops_kernel_optimizers_ ffts engine
  for (auto &engine_name : engine_name_list) {
    ffts::EngineManager &em = ffts::EngineManager::Instance(engine_name);
    em.GetGraphOptimizerObjs(ops_kernel_optimizers_, engine_name);
  }

  std::set<std::string> engine_list;
  engine_list.emplace(ffts::kAIcoreEngineName);
  engine_list.emplace(ffts::kDsaCoreEngineName);
  engine_list.emplace(ffts::kDnnVmAICpu);
  engine_list.emplace(ffts::kDnnVmRts);
  engine_list.emplace(ffts::kDnnVmAICpuAscend);
  engine_list.emplace(ffts::kDnnVmGeLocal);
  composite_engines_[ffts::kFFTSPlusCoreName] = engine_list;
  composite_engine_kernel_lib_names_[ffts::kFFTSPlusCoreName] = ffts::kFFTSPlusCoreName;
  // replace fftsplus_opskernel_builder to mock_fftsplus_opskernel_builder
  // to write file to disk
  std::shared_ptr<ge::OpsKernelBuilder> fftsplus_ops_kernel_builder_ptr =
                                        std::make_shared<ffts::MockFFTSPlusOpsKernelBuilder>();
  ge::OpsKernelBuilderRegistry::GetInstance().Unregister("ffts_plus");
  ge::OpsKernelBuilderRegistry::GetInstance().Register("ffts_plus", fftsplus_ops_kernel_builder_ptr);
  if (fftsplus_ops_kernel_builder_ptr->Initialize(options_) != ffts::SUCCESS) {
    FE_LOGD("[FFTS_ST] MockFFTSPlusOpsKernelBuilder Initliaze failed");
    fftsplus_ops_kernel_builder_ptr->Finalize();
    return;
  }
  FE_LOGD("[FFTS_ST]After classify. Size [%zu, %zu, %zu].", ops_kernel_optimizers_.size(),
          atomic_first_optimizers_by_priority_.size(), composite_engines_.size());
}

void FeRunningEnv::RenameLibfe() {
  if (ascend_path_.empty()) {
    ascend_path_ = FeEnvUtils::GetAscendPath();
  }

  string libfe_path = ascend_path_ + "/compiler/lib64/plugin/opskernel/libfe.so";
  string new_libfe_path = ascend_path_ + "/compiler/lib64/plugin/opskernel/bk.libfe.so";
  if (std::rename(libfe_path.c_str(), new_libfe_path.c_str()) < 0) {
    std::cout<< strerror(errno) << std::endl;
  }
}

void FeRunningEnv::RenameLibFFTS() {
  if (ascend_path_.empty()) {
    ascend_path_ = FeEnvUtils::GetAscendPath();
  }

  string libfe_path = ascend_path_ + "/compiler/lib64/plugin/opskernel/libffts.so";
  string new_libfe_path = ascend_path_ + "/compiler/lib64/plugin/opskernel/bk.libffts.so";
  if (std::rename(libfe_path.c_str(), new_libfe_path.c_str()) < 0) {
    std::cout<< strerror(errno) << std::endl;
  }
}

void FeRunningEnv::ReStoreLibfe() {
  if (ascend_path_.empty()) {
    ascend_path_ = FeEnvUtils::GetAscendPath();
  }
  string libfe_path = ascend_path_ + "/compiler/lib64/plugin/opskernel/libfe.so";
  string new_libfe_path = ascend_path_ + "/compiler/lib64/plugin/opskernel/bk.libfe.so";
  if (std::rename(new_libfe_path.c_str(), libfe_path.c_str()) < 0) {
    std::cout<< strerror(errno) << std::endl;
  }
}

void FeRunningEnv::ReStoreLibfeCfg() {
  if (ascend_path_.empty()) {
    ascend_path_ = FeEnvUtils::GetAscendPath();
  }

  string libfecfg_path = ascend_path_ + "/compiler/lib64/plugin/opskernel/fe_config/fe.ini";
  string bk_libfecfg_path = ascend_path_ + "/compiler/lib64/plugin/opskernel/fe_config/bk.fe.ini";
  if (std::remove(libfecfg_path.c_str()) < 0) {
    std::cout<< strerror(errno) << std::endl;
  }
  if (std::rename(bk_libfecfg_path.c_str(), libfecfg_path.c_str()) < 0) {
    std::cout<< strerror(errno) << std::endl;
  }
}

void FeRunningEnv::ReStoreLibFFTS() {
  if (ascend_path_.empty()) {
    ascend_path_ = FeEnvUtils::GetAscendPath();
  }
  string libfe_path = ascend_path_ + "/compiler/lib64/plugin/opskernel/libffts.so";
  string new_libfe_path = ascend_path_ + "/compiler/lib64/plugin/opskernel/bk.libffts.so";
  if (std::rename(new_libfe_path.c_str(), libfe_path.c_str()) < 0) {
    std::cout<< strerror(errno) << std::endl;
  }
}

fe::Status FeRunningEnv::InitRuningEnv(const std::map<string, string> &options) {
  fe::MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  InitBasicOptions(options);
  /* Init opp path */
  string ascend_path = FeEnvUtils::GetAscendPath();
  string ascend_opp_path = ascend_path + "/opp";
  mmSetEnv("ASCEND_OPP_PATH", ascend_opp_path.c_str(), MMPA_MAX_PATH);
  // ccec 编译路径适配, 如果存在vendors目录, built-in/op_impl, 否则op_impl/built-in
  string vendors_path = ascend_opp_path + "/vendors";
  mkdir(vendors_path.c_str(), S_IRUSR | S_IWUSR);
  /* 我们在ST的脚本中将libfe.so先改名，不让GE自动加载默认的libfe.so
   * 然后打桩ge::OpsKernelManager中的算子信息库和图优化器。 */
  FE_LOGD("[FE_ST]Start to initialize ge. %zu", ops_kernel_optimizers_.size());
  RenameLibfe();
  RenameLibFFTS();
  // RenameLibfeCfg();
  auto status = ge::GEInitialize(ascend_options_);
  if (status != fe::SUCCESS) {
    return status;
  }
  ReStoreLibfe();
  ReStoreLibFFTS();
  FE_LOGD("[FE_ST]Start to initialize fe. %zu", ops_kernel_optimizers_.size());
  fe::Status ret = Initialize(options_);
  if (ret != fe::SUCCESS) {
    return ret;
  }

  InjectFe();
  InjectFFTS();
  return ret;
}

fe::Status FeRunningEnv::FinalizeEnv() {
  FE_LOGD("[FE_ST]Start to finalize env.");
  auto status = ge::GEFinalize();
  if (status != fe::SUCCESS) {
    FE_LOGE("Fail to finalize GE.");
    return status;
  }

  status = Finalize();
  if (status != fe::SUCCESS) {
    return status;
  }
  std::vector<std::string> engine_name_list = {ffts::kFFTSPlusEngineName};
  for (auto &engine_name : engine_name_list) {
    ffts::Status ret = ffts::EngineManager::Instance(engine_name).Finalize();
    if (ret != ffts::SUCCESS) {
      return fe::FAILED;
    }
  }

  ops_kernel_info_.clear();
  ops_kernel_info_stores_.clear();
  ops_kernel_optimizers_.clear();
  atomic_first_optimizers_by_priority_.clear();
  composite_engines_.clear();
  composite_engine_kernel_lib_names_.clear();
}

fe::Status FeRunningEnv::ResetEnv(const std::map<string, string> &options) {
  (void)FinalizeEnv();

  FE_LOGD("[FE_ST]End finalizing GE. Start re-init fe env.");
  InitRuningEnv(options);
  return fe::SUCCESS;
}

ge::Tensor CreateTensor(const ge::TensorDesc &tensor_desc) {
  int64_t tensor_size = -1;
  ge::TensorUtils::GetTensorSizeInBytes(ge::TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc), tensor_size);
  std::vector<uint8_t> tensor_buffer(tensor_size);
  ge::Tensor tensor(tensor_desc);
  tensor.SetData(std::move(tensor_buffer));
  return tensor;
}

void UpdateTensors(ge::NodePtr &node, bool need_update_name) {
  ge::OpDescPtr standard_op_desc = nullptr;
  const auto standard_op = ge::OperatorFactory::CreateOperator("standard_op", node->GetType().c_str());
  if (standard_op.IsEmpty()) {
    FE_LOGD("[FE_ST]Failed to create temp op %s.", node->GetType().c_str());
  } else {
    standard_op_desc = ge::OpDescUtils::GetOpDescFromOperator(standard_op);
  }


  auto op = node->GetOpDesc();
  for (size_t i = 0; i < op->GetAllInputsSize(); i++) {
    auto input = op->MutableInputDesc(i);
    if (input == nullptr) {
      continue;
    }
    input->SetOriginDataType(input->GetDataType());
    if (need_update_name && standard_op_desc != nullptr) {
      auto std_input = standard_op_desc->GetInputDescPtr(i);
      if (std_input == nullptr) {
        continue;
      }
      input->SetName(std_input->GetName());
    }
  }

  for (size_t i = 0; i < op->GetOutputsSize(); i++) {
    auto output = node->GetOpDesc()->MutableOutputDesc(i);
    if (output == nullptr) {
      continue;
    }
    output->SetOriginDataType(output->GetDataType());
    if (need_update_name && standard_op_desc != nullptr) {
      auto std_output = standard_op_desc->GetOutputDescPtr(i);
      if (std_output == nullptr) {
        continue;
      }
      output->SetName(std_output->GetName());
    }
  }
}

void FillConst(const ge::NodePtr &node, const size_t &shape_size,
               unique_ptr<uint8_t[]> &value) {
  auto out_nodes = node->GetOutDataNodes();
  if (out_nodes.empty()) {
    return;
  }

  auto out_node = out_nodes.at(0);
  if (out_node == nullptr) {
    return;
  }

  if (out_node->GetType() == fe::RESHAPE) {
    auto output_desc = out_node->GetOpDesc()->MutableOutputDesc(0);
    if (output_desc != nullptr) {
      auto output_dims = output_desc->GetShape().GetDims();
      for (size_t i = 0; i < shape_size; i++) {
        FE_LOGD("node %s output dims %ld", out_node->GetName().c_str(), output_dims[i]);
        *(((int32_t*)(value.get())) + i) = (int32_t)output_dims[i];
      }
    }
  }
}

std::vector<ge::Tensor> CreateInputTensors(const ge::Graph &graph) {
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  int64_t index = 0;
  std::map<int64_t, ge::GeTensorDescPtr> tensor_descs;
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == fe::DATA) {
      tensor_descs[index] = node->GetOpDesc()->MutableOutputDesc(0);
      ge::AttrUtils::SetInt(node->GetOpDesc(), ge::ATTR_NAME_INDEX, index++);
    }
  }

  std::vector<ge::Tensor> tensors(tensor_descs.size());
  for (const auto &it : tensor_descs) {
    tensors[it.first] = CreateTensor(ge::TensorAdapter::GeTensorDesc2TensorDesc(*it.second));
  }

  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == fe::CONSTANT || node->GetType() == fe::CONSTANTOP) {
      auto current_weight = ge::OpDescUtils::MutableWeights(node);
      if (!current_weight.empty()) {
        continue;
      }

      ge::GeTensorPtr tensor = nullptr;
      FE_MAKE_SHARED(tensor = std::make_shared<ge::GeTensor>(), return tensors);
      tensor->SetTensorDesc(node->GetOpDesc()->GetOutputDesc(0));
      const auto &tensor_desc = tensor->GetTensorDesc();
      size_t shape_size = tensor_desc.GetShape().GetShapeSize();
      auto size = shape_size * ge::GetSizeByDataType(tensor_desc.GetDataType());
      unique_ptr<uint8_t[]> value(new(std::nothrow) uint8_t[size]);
      (void)memset_s(value.get(), size, 1, size);
      FillConst(node, shape_size, value);

      (void)tensor->SetData(reinterpret_cast<uint8_t *>(value.get()), size);
      (void)ge::AttrUtils::SetTensor(node->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, tensor);
    }
  }
  return tensors;
}

void CorrectNodes(ge::ComputeGraphPtr &compute_graph, bool need_update_name) {
  for (auto &node : compute_graph->GetDirectNode()) {
    UpdateTensors(node, need_update_name);
    if (node->GetType() == fe::ASCEND_DEQUANT) {
      (void)ge::AttrUtils::SetInt(node->GetOpDesc(), "dtype", 0);
    }
  }
}

fe::Status FeRunningEnv::Run(ge::ComputeGraphPtr &compute_graph, std::map<string, string> &session_options,
                             bool need_update_name) {
  ge::Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  ge::OmgContext omg_context;
  uint32_t graph_id = compute_graph->GetGraphID();
  FE_LOGD("[FE_ST]Start to add graph %s.", compute_graph->GetName().c_str());
  /* 如果在执行AddGraph时，遇到了Segmentation fault，先排查以下几：
   * 1. 环境上使用的ge_compiler和ge_common的so是否为最新 */
  //auto ret = AddGraph(compute_graph);
  session_options[ge::RUN_FLAG] = "0";
  ge::Session session(session_options);

  if (session.AddGraph(graph_id, graph) != ge::SUCCESS) {
    FE_LOGD("[FE_ST]Failed to Add Graph %s", graph.GetName().c_str());
    return fe::FAILED;
  }

  CorrectNodes(compute_graph, need_update_name);
  std::vector<ge::Tensor> inputs = CreateInputTensors(graph);
  std::vector<ge::Tensor> outputs;

  FE_LOGD("[FE_ST]Start to Optimize graph %s.", compute_graph->GetName().c_str());
  if (session.BuildGraph(graph_id, inputs) != ge::SUCCESS) {
    return fe::FAILED;
  }
  //ret = Finalize();
  return session.RemoveGraph(graph_id);
}

void FeRunningEnv::Clear() {
  ge::GetThreadLocalContext().global_options_.clear();
}
}
