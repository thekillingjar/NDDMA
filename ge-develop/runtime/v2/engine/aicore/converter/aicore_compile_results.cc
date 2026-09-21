/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicore_compile_results.h"
#include <cinttypes>
#include "framework/common/debug/ge_log.h"
#include "engine/node_converter_utils.h"
#include "runtime/rt_model.h"
#include "framework/common/taskdown_common.h"
#include "framework/common/ge_types.h"
#include "graph/utils/math_util.h"
#include "kernel/common_kernel_impl/sink_node_bin.h"
#include "graph/op_kernel_bin.h"
#include "graph/debug/ge_attr_define.h"
#include "engine/aicore/fe_rt2_common.h"
#include "common/plugin/ge_make_unique_util.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_tiling_info.h"
#include "register/op_tiling/op_tiling_constants.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace gert {
namespace {
const std::string kAllKernel = "_kernel_list_first_name";
const std::string kMixAicAllKernel = "_mix_aic_kernel_list_first_name";
const std::string kMixAivAllKernel = "_mix_aiv_kernel_list_first_name";
const std::string kMixEnhancedAllKernel = "_mix_enhanced_kernel_list_first_name";
const std::string kEnhancedMixKernle = "_mix_with_enhanced_kernel";

size_t CalcBinDataSize(uint64_t bin_length) {
  size_t total_size = 0;
  if (ge::AddOverflow(sizeof(kernel::BinData), bin_length, total_size)) {
    GELOGE(ge::FAILED, "Failed to calc size, overflow add, bin length %" PRIu64 "", bin_length);
    return 0;
  }
  return total_size;
}
ge::Status BuildBinData(const ge::NodePtr &node, const std::string &kernel_key, const std::string &tvm_magic_key,
                        std::unique_ptr<uint8_t[]> &bin_data_holder, size_t &bin_size) {
  auto op_desc = node->GetOpDesc();
  FE_ASSERT_NOTNULL(op_desc);
  const auto kernel_bin = op_desc->TryGetExtAttr(kernel_key, ge::OpKernelBinPtr());
  if (kernel_bin == nullptr) {
    GELOGE(ge::FAILED, "Failed to get kernel bin by key %s on node %s", kernel_key.c_str(), node->GetName().c_str());
    return ge::FAILED;
  }

  auto tvm_magic_str = ge::AttrUtils::GetStr(op_desc, tvm_magic_key);
  if (tvm_magic_str == nullptr) {
    GELOGE(ge::FAILED, "Failed to get tvm magic by key %s on node %s.", tvm_magic_key.c_str(), node->GetNamePtr());
    return ge::FAILED;
  }

  bin_size = CalcBinDataSize(kernel_bin->GetBinDataSize());
  if (bin_size == 0) {
    return ge::FAILED;
  }
  bin_data_holder = ge::MakeUnique<uint8_t[]>(bin_size);
  if (bin_data_holder == nullptr) {
    return ge::FAILED;
  }
  auto bin_data = ge::PtrToPtr<uint8_t, kernel::BinData>(bin_data_holder.get());

  if (*tvm_magic_str == "RT_DEV_BINARY_MAGIC_ELF_AICPU") {
    bin_data->magic = RT_DEV_BINARY_MAGIC_ELF_AICPU;
  } else if (*tvm_magic_str == "RT_DEV_BINARY_MAGIC_ELF") {
    bin_data->magic = RT_DEV_BINARY_MAGIC_ELF;
  } else if (*tvm_magic_str == "RT_DEV_BINARY_MAGIC_ELF_AIVEC") {
    bin_data->magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  } else if (*tvm_magic_str == "RT_DEV_BINARY_MAGIC_ELF_AICUBE") {
    bin_data->magic = RT_DEV_BINARY_MAGIC_ELF_AICUBE;
  } else if ((*tvm_magic_str == "FFTS_BINARY_MAGIC_ELF_MIX_AIC") ||
             (*tvm_magic_str == "FFTS_BINARY_MAGIC_ELF_MIX_AIV")) {
    bin_data->magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  } else {
    GELOGE(ge::FAILED, "[Check][JsonStr]Unexpected tvm magic %s, magic key %s", tvm_magic_str->c_str(),
           tvm_magic_key.c_str());
    return ge::FAILED;
  }

  bin_data->version = 0U;
  bin_data->length = kernel_bin->GetBinDataSize();
  FE_ASSERT_TRUE(bin_size > sizeof(*bin_data));
  FE_ASSERT_EOK(memcpy_s(&(bin_data->placeholder), bin_size - sizeof(*bin_data), kernel_bin->GetBinData(), bin_data->length));
  bin_data->data = nullptr;
  return ge::SUCCESS;
}

struct AttrKeys {
  std::string kernel_key;
  std::string tvm_magic_key;
  std::string meta_data_key;
  std::string kernel_name_key;
  std::string raw_kernel_name_key;
};

std::mutex g_process_bin_data_holders_lk;
std::map<std::string, std::unique_ptr<uint8_t[]>> g_process_bin_data_holders;

bg::ValueHolderPtr BuildBinData(const ge::NodePtr &node, const domi::TaskDef &task_def, const AttrKeys &keys) {
  std::unique_ptr<uint8_t[]> bin_data_holder;
  size_t bin_size = 0;
  auto ret = BuildBinData(node, keys.kernel_key, keys.tvm_magic_key, bin_data_holder, bin_size);
  if (ret != ge::SUCCESS) {
    return nullptr;
  }
  std::string kernel_bin_id;
  if (keys.tvm_magic_key == ge::ATOMIC_ATTR_TVM_MAGIC) {
    (void)ge::AttrUtils::GetStr(node->GetOpDesc(), kAttrMemsetKernelBinId, kernel_bin_id);
  } else {
    (void)ge::AttrUtils::GetStr(node->GetOpDesc(), kAttrKernelBinId, kernel_bin_id);
  }
  GELOGD("Kernel bin id of node %s is %s with magic [%s].", node->GetNamePtr(), kernel_bin_id.c_str(),
         keys.tvm_magic_key.c_str());
  bg::ValueHolderPtr bin_data = nullptr;
  if (!kernel_bin_id.empty()) {
    std::unique_lock<std::mutex> lk(g_process_bin_data_holders_lk);
    auto emplace_result = g_process_bin_data_holders.emplace(kernel_bin_id, std::move(bin_data_holder));
    void *bin_data_ptr = emplace_result.first->second.get();
    GELOGI("%s kernel bin id: %s, size: %zu, node: %s.", (emplace_result.second ? "Build" : "Reuse"),
           kernel_bin_id.c_str(), bin_size, node->GetNamePtr());

    bin_data = bg::ValueHolder::CreateConst(&bin_data_ptr, sizeof(void *));
  } else {
    GELOGD("Build kernel data with empty bin id size %zu for node %s.", bin_size, node->GetName().c_str());
    bin_data = bg::ValueHolder::CreateConst(bin_data_holder.get(), bin_size);
  }
  // todo AiCoreOpTask::InitWithKernelDef中，有计算launch使用args最大值的逻辑
  //    这里引申出一个需要做的特性：kernel-launch的args需要支持零拷贝
  // todo AiCoreOpTask::ValidateTaskDef
  const auto task_type = static_cast<ge::ModelTaskType>(task_def.type());
  if (task_type == ge::ModelTaskType::MODEL_TASK_ALL_KERNEL) {
    // todo SinkNodeBinWithHandle的实现参考AiCoreOpTask::RegisterKernelHandle
    auto hash_data = bg::ValueHolder::CreateConst(kernel_bin_id.c_str(), kernel_bin_id.size() + 1U, true);
    return bg::ValueHolder::CreateSingleDataOutput("SinkNodeBinWithHandle", {bin_data, hash_data});
  } else if (task_type == ge::ModelTaskType::MODEL_TASK_KERNEL) {
    // todo 单算子这里不太一样
    if (kernel_bin_id.empty()) {
      kernel_bin_id = task_def.kernel().stub_func();
      GELOGD("Use stub func[%s] as bin key", kernel_bin_id.c_str());
    }
    auto stub_name = bg::ValueHolder::CreateConst(kernel_bin_id.c_str(), kernel_bin_id.size() + 1, true);
    std::string metadata_str;
    if (!ge::AttrUtils::GetStr(node->GetOpDesc(), keys.meta_data_key, metadata_str)) {
      GELOGD("metadata retrieval from node resulted in empty for node: %s.", node->GetName().c_str());
    }
    auto metadata = bg::ValueHolder::CreateConst(metadata_str.c_str(), metadata_str.size() + 1, true);
    std::string kernel_name_str;
    if (!ge::AttrUtils::GetStr(node->GetOpDesc(), keys.kernel_name_key, keys.raw_kernel_name_key, kernel_name_str)) {
      GELOGD("Kernel name is empty for node: %s, unable to retrieve.", node->GetName().c_str());
    }
    auto kernel_name = bg::ValueHolder::CreateConst(kernel_name_str.c_str(), kernel_name_str.size() + 1, true);
    // todo SinkNodeBinWithoutHandle的实现参考AiCoreOpTask::RegisterTbeHandle
    return bg::ValueHolder::CreateSingleDataOutput("SinkNodeBinWithoutHandle",
                                                   {bin_data, stub_name, metadata, kernel_name});
  } else {
    GELOGE(ge::FAILED, "Failed to build bin data for node %s, unexpected task type %d", node->GetName().c_str(),
           task_type);
    return nullptr;
  }
}

bool IsAllKernelTask(const ge::OpDescPtr &op_desc) {
  return op_desc->HasAttr(kAllKernel) || op_desc->HasAttr(kMixAicAllKernel) || op_desc->HasAttr(kMixAivAllKernel) ||
         op_desc->HasAttr(kMixEnhancedAllKernel);
}

bg::ValueHolderPtr SinkFFTSStaManualNodeBin(const ge::NodePtr &node, const bg::ValueHolderPtr bin_data,
                                            const AttrKeys &keys) {
  std::string kernel_name;
  (void)ge::AttrUtils::GetStr(node->GetOpDesc(), keys.kernel_name_key, keys.raw_kernel_name_key, kernel_name);
  GELOGD("Node[%s] kernel_attr: %s, kernel: %s", node->GetOpDesc()->GetNamePtr(),
         keys.kernel_name_key.c_str(), kernel_name.c_str());
  auto kernel_name_h = bg::ValueHolder::CreateConst(kernel_name.c_str(), kernel_name.size() + 1, true);
  std::vector<bg::ValueHolderPtr> sink_inputs;
  sink_inputs.emplace_back(kernel_name_h);
  std::string stub_name_str = node->GetOpDesc()->GetName();
  auto stub_name = bg::ValueHolder::CreateConst(stub_name_str.c_str(), stub_name_str.size() + 1, true);
  sink_inputs.emplace_back(stub_name);
  std::string metadata;
  (void)ge::AttrUtils::GetStr(node->GetOpDesc(), keys.meta_data_key, metadata);
  GELOGD("Metadata for node %s is [%s].", node->GetNamePtr(), metadata.c_str());
  auto metadata_h = bg::ValueHolder::CreateConst(metadata.c_str(), metadata.size() + 1, true);
  sink_inputs.emplace_back(metadata_h);
  sink_inputs.emplace_back(bin_data);
  return bg::ValueHolder::CreateSingleDataOutput("SinkFFTSStaManualNodeBin", sink_inputs);
}

bool BuildAutoNodeBinData(std::vector<ge::OpKernelBinPtr> &kernel_bin_ptr_vec,
                          std::vector<std::string> &magic_vec, std::vector<bg::ValueHolderPtr> &bin_vec) {
  for (size_t i = 0; i < magic_vec.size(); ++i) {
    auto kernel_bin = kernel_bin_ptr_vec[i];
    auto tvm_magic_str = magic_vec[i];
    if (kernel_bin == nullptr) {
      GELOGE(ge::FAILED, "Kernel is nullptr!");
      return false;
    }
    size_t bin_size = 0;
    bin_size = CalcBinDataSize(kernel_bin->GetBinDataSize());
    if (bin_size == 0) {
      GELOGE(ge::FAILED, "Bin size is 0.");
      return false;
    }
    auto bin_data_holder = ge::MakeUnique<uint8_t[]>(bin_size);
    if (bin_data_holder == nullptr) {
      return false;
    }
    auto bin_data = ge::PtrToPtr<uint8_t, kernel::BinData>(bin_data_holder.get());
    if (tvm_magic_str == "RT_DEV_BINARY_MAGIC_ELF_AIVEC") {
      bin_data->magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
    } else if (tvm_magic_str == "RT_DEV_BINARY_MAGIC_ELF_AICUBE") {
      bin_data->magic = RT_DEV_BINARY_MAGIC_ELF_AICUBE;
    } else {
      GELOGE(ge::FAILED, "[Check][JsonStr]Unexpected tvm magic %s, magic key %s.", tvm_magic_str.c_str(),
             tvm_magic_str.c_str());
      return false;
    }
    bin_data->version = 0U;
    bin_data->length = kernel_bin->GetBinDataSize();
    FE_ASSERT_TRUE(bin_size > sizeof(*bin_data));
    if (memcpy_s(&(bin_data->placeholder), bin_size - sizeof(*bin_data), kernel_bin->GetBinData(), bin_data->length) !=
        EOK) {
      return false;
    }
    bin_data->data = nullptr;
    auto bin_data_h = bg::ValueHolder::CreateConst(bin_data_holder.get(), bin_size);
    if (bin_data_h == nullptr) {
      return false;
    }
    bin_vec.emplace_back(bin_data_h);
  }
  return true;
}

bg::ValueHolderPtr SinkBinForFFTS(const ge::NodePtr &node, std::vector<bg::ValueHolderPtr> &tiling_ret,
                                  const AttrKeys &keys) {
  std::unique_ptr<uint8_t[]> bin_data_holder;
  size_t bin_size = 0;
  auto ret = BuildBinData(node, keys.kernel_key, keys.tvm_magic_key, bin_data_holder, bin_size);
  if (ret != ge::SUCCESS) {
    return nullptr;
  }
  auto bin_data = bg::ValueHolder::CreateConst(bin_data_holder.get(), bin_size);
  if (node->GetOpDesc()->HasAttr("_kernel_list_first_name") && (!tiling_ret.empty())) {
    GELOGD("Sink fftsplus bin for node: %s.", node->GetNamePtr());
    auto bin_handle = bg::ValueHolder::CreateSingleDataOutput("SinkFFTSAICoreNodeBin", {bin_data});
    if (bin_handle == nullptr) {
      GELOGE(ge::FAILED, "Sink bin failed.");
      return nullptr;
    }
    std::vector<bg::ValueHolderPtr> sink_inputs;
    sink_inputs.emplace_back(bin_handle);
    // tail && non tail
    sink_inputs.emplace_back(tiling_ret[TilingContext::kOutputTilingKey]);
    sink_inputs.emplace_back(tiling_ret[static_cast<size_t>(TilingContext::kOutputNum) +
                                        static_cast<size_t>(TilingContext::kOutputTilingKey)]);
    return bg::ValueHolder::CreateSingleDataOutput("GetFFTSAICorePcAndPref", sink_inputs);
  } else {
    // for static (some dynamic opp may have only one kernel(reuse binary))
    GELOGD("Sink FFTSPlus static bin for node: %s.", node->GetNamePtr());
    return SinkFFTSStaManualNodeBin(node, bin_data, keys);
  }
}

bg::ValueHolderPtr CreateMixBinData(const ge::NodePtr &node, const std::string &prefix, const std::string &bin_id) {
  const auto &kernel_key = prefix + ge::OP_EXTATTR_NAME_TBE_KERNEL;
  if (kernel_key.empty()) {
    GELOGE(ge::FAILED, "Node[%s] mix kernel key with prefix[%s] is empty.", node->GetName().c_str(), prefix.c_str());
    return nullptr;
  }
  size_t bin_size = 0;
  std::unique_ptr<uint8_t[]> bin_data_holder;
  auto ret = BuildBinData(node, kernel_key, ge::TVM_ATTR_NAME_MAGIC, bin_data_holder, bin_size);
  if (ret != ge::SUCCESS) {
    GELOGE(ge::FAILED, "Node[%s] make bin with prefix[%s] failed.", node->GetName().c_str(), prefix.c_str());
    return nullptr;
  }
  auto bin_data_ptr = ge::PtrToPtr<uint8_t, kernel::BinData>(bin_data_holder.get());
  if (prefix == "_mix_enhanced") {
    bin_data_ptr->magic = RT_DEV_BINARY_MAGIC_ELF;
  } else if (prefix == "_mix_aic") {
    bin_data_ptr->magic = RT_DEV_BINARY_MAGIC_ELF_AICUBE;
  } else {
    bin_data_ptr->magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  }
  bg::ValueHolderPtr bin_data = nullptr;
  if (!bin_id.empty()) {
    std::unique_lock<std::mutex> lk(g_process_bin_data_holders_lk);
    auto emplace_result = g_process_bin_data_holders.emplace(bin_id, std::move(bin_data_holder));
    void *ret_data_ptr = emplace_result.first->second.get();
    GELOGI("%s Mix kernel bin id: %s, size: %zu, node: %s.", (emplace_result.second ? "Build" : "Reuse"),
           bin_id.c_str(), bin_size, node->GetNamePtr());
    bin_data = bg::ValueHolder::CreateConst(&ret_data_ptr, sizeof(void *));
  } else {
    bin_data = bg::ValueHolder::CreateConst(bin_data_holder.get(), bin_size);
  }
  GELOGD("Start registering kernel for node: [%s], with kernel key: [%s], and prefix: [%s].", node->GetNamePtr(),
         kernel_key.c_str(), prefix.c_str());
  return bin_data;
}

ge::Status SinkStaticMixNode(const ge::NodePtr &node, std::vector<bg::ValueHolderPtr> &sink_inputs,
                             std::vector<std::string> &names_prefix) {
  for (const auto &prefix : names_prefix) {
    const std::string attr_kernel_name = prefix + node->GetOpDesc()->GetName() + "_kernelname";
    std::string kernel_name;
    (void)ge::AttrUtils::GetStr(node->GetOpDesc(), attr_kernel_name, "_kernelname", kernel_name);
    GELOGD("[%s]attr: %s, kernel: %s.", node->GetOpDesc()->GetNamePtr(),
           attr_kernel_name.c_str(), kernel_name.c_str());
    auto kernel_name_h = bg::ValueHolder::CreateConst(kernel_name.c_str(), kernel_name.size() + 1, true);
    sink_inputs.emplace_back(kernel_name_h);
    auto prefix_holder = bg::ValueHolder::CreateConst(prefix.c_str(), prefix.size() + 1, true);
    sink_inputs.emplace_back(prefix_holder);
    std::string stub_name_str = prefix + node->GetOpDesc()->GetName();
    auto stub_name = bg::ValueHolder::CreateConst(stub_name_str.c_str(), stub_name_str.size() + 1, true);
    sink_inputs.emplace_back(stub_name);
    std::string metadata;
    (void)ge::AttrUtils::GetStr(node->GetOpDesc(), prefix + ge::TVM_ATTR_NAME_METADATA, metadata);
    GELOGD("Metadata for node: %s is [%s].", node->GetNamePtr(), metadata.c_str());
    auto metadata_h = bg::ValueHolder::CreateConst(metadata.c_str(), metadata.size() + 1, true);
    sink_inputs.emplace_back(metadata_h);
    auto bin_data = CreateMixBinData(node, prefix, "");
    FE_ASSERT_NOTNULL(bin_data);
    sink_inputs.emplace_back(bin_data);
  }
  return ge::SUCCESS;
}
ge::Status SinkDynamicMixNode(const ge::NodePtr &node, std::vector<bg::ValueHolderPtr> &sink_inputs,
                              std::vector<std::string> &names_prefix) {
  for (const auto &prefix : names_prefix) {
    GELOGD("Starting sink dynamic mix node for prefix %s on node: '%s'.", prefix.c_str(), node->GetNamePtr());
    std::string kernel_bin_id;
    (void)ge::AttrUtils::GetStr(node->GetOpDesc(), kAttrKernelBinId, kernel_bin_id);
    kernel_bin_id += prefix;
    auto bin_data = CreateMixBinData(node, prefix, kernel_bin_id);
    FE_ASSERT_NOTNULL(bin_data);
    auto hash_id = bg::ValueHolder::CreateConst(kernel_bin_id.c_str(), kernel_bin_id.size() + 1, true);
    auto bin_handle = bg::ValueHolder::CreateSingleDataOutput("SinkMixDyNodeBin", {bin_data, hash_id});
    if (bin_handle == nullptr) {
      return ge::FAILED;
    }
    auto prefix_holder = bg::ValueHolder::CreateConst(prefix.c_str(), prefix.size() + 1, true);
    sink_inputs.emplace_back(prefix_holder);
    sink_inputs.emplace_back(bin_handle);
  }
  GELOGD("Sink dynamic node bin for node: %s.", node->GetNamePtr());
  return ge::SUCCESS;
}
gert::kernel::MIX_KERNEL_REQ_TYPE GetMixAiCoreKernelType(const ge::OpDescPtr &op_desc,
                                                         std::vector<bg::ValueHolderPtr> &tiling_ret,
                                                         bool is_static_and_reuse_binary) {
  bool is_enhanced_kernel = false;
  (void)ge::AttrUtils::GetBool(op_desc, kEnhancedMixKernle, is_enhanced_kernel);
  bool is_dynamic = (IsAllKernelTask(op_desc) && (!tiling_ret.empty() || is_static_and_reuse_binary));
  GELOGD("op %s:%s is_dynamic flag is %u and is_enhanced_kernel flag is %u.", op_desc->GetNamePtr(), op_desc->GetTypePtr(), is_dynamic,
         is_enhanced_kernel);
  if (is_enhanced_kernel) {
    return is_dynamic ? gert::kernel::MIX_KERNEL_REQ_TYPE::ENHANCED_MIX_DYNAMIC
                      : gert::kernel::MIX_KERNEL_REQ_TYPE::ENHANCED_MIX_SINGLE_KERNEL;
  } else {
    return is_dynamic ? gert::kernel::MIX_KERNEL_REQ_TYPE::MIX_MULTI_KERNEL
                      : gert::kernel::MIX_KERNEL_REQ_TYPE::MIX_SINGLE_KERNEL;
  }
}
}  // namespace

bg::ValueHolderPtr SinkBinForAicore(const ge::NodePtr &node,
                                    const LoweringGlobalData::NodeCompileResult *compile_result) {
  auto task_def = GetTaskDef(node, compile_result, TaskDefType::kAICore);
  if (task_def == nullptr) {
    GELOGE(ge::FAILED, "Failed to get task_def from node %s.", node->GetName().c_str());
    return nullptr;
  }
  return BuildBinData(node, *task_def,
                      AttrKeys{ge::OP_EXTATTR_NAME_TBE_KERNEL,
                               ge::TVM_ATTR_NAME_MAGIC,
                               ge::TVM_ATTR_NAME_METADATA,
                               node->GetName() + "_kernelname",
                               "_kernelname"});
}

bg::ValueHolderPtr SinkFFTSStaAutoNodeBin(const ge::NodePtr &node) {
  GELOGD("Sink ffts auto node[%s]", node->GetName().c_str());
  std::vector<std::string> magic_vec;
  (void)ge::AttrUtils::GetListStr(node->GetOpDesc(), ge::TVM_ATTR_NAME_THREAD_MAGIC, magic_vec);
  std::vector<ge::OpKernelBinPtr> kernel_bin_ptr_vec;
  kernel_bin_ptr_vec = node->GetOpDesc()->TryGetExtAttr(ge::OP_EXTATTR_NAME_THREAD_TBE_KERNEL, kernel_bin_ptr_vec);
  if (magic_vec.size() != kernel_bin_ptr_vec.size()) {
    GELOGE(ge::FAILED, "Magic size[%zu] not equal to kernel size[%zu]", magic_vec.size(), kernel_bin_ptr_vec.size());
    return nullptr;
  }
  std::vector<bg::ValueHolderPtr> bin_vec;
  if (!BuildAutoNodeBinData(kernel_bin_ptr_vec, magic_vec, bin_vec)) {
    GELOGE(ge::FAILED, "Node[%s] build auto bin data failed", node->GetName().c_str());
    return nullptr;
  }
  std::vector<std::string> tbe_kernel_name_vec;
  std::vector<std::string> meta_data_vec;
  (void)ge::AttrUtils::GetListStr(node->GetOpDesc(),
                                  ge::ATTR_NAME_THREAD_TBE_KERNEL_NAME, tbe_kernel_name_vec);
  (void)ge::AttrUtils::GetListStr(node->GetOpDesc(),
                                  ge::TVM_ATTR_NAME_THREAD_METADATA, meta_data_vec);
  size_t kernel_num = tbe_kernel_name_vec.size();
  if (kernel_num != kernel_bin_ptr_vec.size() || meta_data_vec.size() != kernel_num) {
    GELOGE(ge::FAILED, "Kernel name size[%zu] not equal kernel size[%zu]", tbe_kernel_name_vec.size(),
           kernel_bin_ptr_vec.size());
    return nullptr;
  }
  std::vector<bg::ValueHolderPtr> sink_inputs;
  sink_inputs.emplace_back(bg::ValueHolder::CreateConst(&kernel_num, sizeof(kernel_num)));
  std::string stub_name_str = node->GetOpDesc()->GetName();
  auto stub_name = bg::ValueHolder::CreateConst(stub_name_str.c_str(), stub_name_str.size() + 1, true);
  for (size_t i = 0; i < kernel_num; ++i) {
    auto &kernel_name = tbe_kernel_name_vec[i];
    auto kernel_name_h = bg::ValueHolder::CreateConst(kernel_name.c_str(), kernel_name.size() + 1, true);
    sink_inputs.emplace_back(kernel_name_h);
    sink_inputs.emplace_back(stub_name);
    auto &metadata = meta_data_vec[i];
    GELOGD("Metadata for node %s is [%s].", node->GetName().c_str(), metadata.c_str());
    auto metadata_h = bg::ValueHolder::CreateConst(metadata.c_str(), metadata.size() + 1, true);
    sink_inputs.emplace_back(metadata_h);
    sink_inputs.emplace_back(bin_vec[i]);
  }
  return bg::ValueHolder::CreateSingleDataOutput("SinkFFTSStaAutoNodeBin", sink_inputs);
}

bg::ValueHolderPtr SinkBinForFFTSAicore(const ge::NodePtr &node, std::vector<bg::ValueHolderPtr> &tiling_ret) {
  return SinkBinForFFTS(node, tiling_ret,
                        AttrKeys{ge::OP_EXTATTR_NAME_TBE_KERNEL,
                                 ge::TVM_ATTR_NAME_MAGIC,
                                 ge::TVM_ATTR_NAME_METADATA,
                                 node->GetName() + "_kernelname",
                                 "_kernelname"});
}

bg::ValueHolderPtr SinkFFTSAtomicBin(const ge::NodePtr &node) {
  std::vector<bg::ValueHolderPtr> tiling_ret;
  return SinkBinForFFTS(node, tiling_ret,
                        AttrKeys{ge::EXT_ATTR_ATOMIC_TBE_KERNEL,
                                 ge::ATOMIC_ATTR_TVM_MAGIC,
                                 ge::ATOMIC_ATTR_TVM_METADATA,
                                 node->GetName() + "_atomic_kernelname",
                                 "_atomic_kernelname"});
}

bg::ValueHolderPtr SinkAtomicBin(const ge::NodePtr &node, const LoweringGlobalData::NodeCompileResult *compile_result) {
  auto task_def = GetTaskDef(node, compile_result, TaskDefType::kAtomicClean);
  if (task_def == nullptr) {
    return nullptr;
  }
  return BuildBinData(node, *task_def,
                      AttrKeys{ge::EXT_ATTR_ATOMIC_TBE_KERNEL,
                               ge::ATOMIC_ATTR_TVM_MAGIC,
                               ge::ATOMIC_ATTR_TVM_METADATA,
                               node->GetName() + "_atomic_kernelname",
                               "_atomic_kernelname"});
}

bg::ValueHolderPtr SinkBinForMixAiCore(const ge::NodePtr &node, std::vector<bg::ValueHolderPtr> &tiling_ret) {
  std::vector<std::string> names_prefix;
  (void)ge::AttrUtils::GetListStr(node->GetOpDesc(), ge::ATTR_NAME_KERNEL_NAMES_PREFIX, names_prefix);
  std::vector<bg::ValueHolderPtr> sink_inputs;
  std::shared_ptr<optiling::utils::OpRunInfo> tiling_info = nullptr;
  tiling_info = node->GetOpDesc()->TryGetExtAttr(ge::ATTR_NAME_OP_RUN_INFO, tiling_info);
  size_t bin_num = names_prefix.size();
  if (bin_num == 0) {
    GELOGE(ge::FAILED, "Node[%s] prefix is empty.", node->GetName().c_str());
    return nullptr;
  }
  auto bin_num_holder = bg::ValueHolder::CreateConst(&bin_num, sizeof(bin_num));
  sink_inputs.emplace_back(bin_num_holder);
  bool is_unknown = false;
  bool is_static_and_reuse_binary = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDesc(), kUnknownShapeFromFe, is_unknown);
  if (!is_unknown && tiling_info != nullptr) {
    GELOGD("Static and reuse binary");
    is_static_and_reuse_binary = true;
  }
  gert::kernel::MIX_KERNEL_REQ_TYPE kernel_type = GetMixAiCoreKernelType(node->GetOpDesc(), tiling_ret,
                                                                         is_static_and_reuse_binary);
  size_t kernel_type_value = static_cast<size_t>(kernel_type);
  auto kernel_type_holder = bg::ValueHolder::CreateConst(&kernel_type_value, sizeof(kernel_type_value));
  sink_inputs.emplace_back(kernel_type_holder);
  switch (kernel_type) {
    case gert::kernel::MIX_KERNEL_REQ_TYPE::MIX_SINGLE_KERNEL:
    case gert::kernel::MIX_KERNEL_REQ_TYPE::ENHANCED_MIX_SINGLE_KERNEL:
      if (SinkStaticMixNode(node, sink_inputs, names_prefix) == ge::FAILED) {
        GELOGE(ge::FAILED, "Node[%s] sink static bin failed.", node->GetName().c_str());
        return nullptr;
      }
      return bg::ValueHolder::CreateSingleDataOutput("SinkMixStaticNodeBin", sink_inputs);
    case gert::kernel::MIX_KERNEL_REQ_TYPE::MIX_MULTI_KERNEL:
    case gert::kernel::MIX_KERNEL_REQ_TYPE::ENHANCED_MIX_DYNAMIC:
      if (is_static_and_reuse_binary) {
        auto tiling_key = tiling_info->GetTilingKey();
        bg::ValueHolderPtr tiling_key_value_holder = bg::ValueHolder::CreateConst(&tiling_key, sizeof(tiling_key));
        sink_inputs.emplace_back(tiling_key_value_holder);
        sink_inputs.emplace_back(tiling_key_value_holder);
      } else {
        sink_inputs.emplace_back(tiling_ret[TilingContext::kOutputTilingKey]);
        if (tiling_ret.size() == ((static_cast<size_t>(TilingContext::kOutputNum) << 1) + 1)) {
          sink_inputs.emplace_back(tiling_ret[(static_cast<size_t>(TilingContext::kOutputNum) +
                                             static_cast<size_t>(TilingContext::kOutputTilingKey))]);
        } else {
        sink_inputs.emplace_back(tiling_ret[TilingContext::kOutputTilingKey]);
        }
      }
      if (SinkDynamicMixNode(node, sink_inputs, names_prefix) == ge::FAILED) {
        GELOGE(ge::FAILED, "Node[%s] sink dynamic bin failed.", node->GetName().c_str());
        return nullptr;
      }
      return bg::ValueHolder::CreateSingleDataOutput("GetMixAddrAndPrefCnt", sink_inputs);
    default:
      GELOGE(ge::FAILED, "node: %s get invalid mix kernel type: %u", node->GetName().c_str(), kernel_type);
      return nullptr;
  }
}
}  // namespace gert
