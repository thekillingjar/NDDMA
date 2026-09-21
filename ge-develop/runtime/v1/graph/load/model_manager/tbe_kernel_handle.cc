/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/tbe_kernel_handle.h"

#include "common/tbe_handle_store/tbe_handle_store.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common/ge_types.h"
#include "framework/runtime/subscriber/global_profiler.h"
#include "register/op_tiling_info.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace {
const std::string kStubFuncName = "_register_stub_func";
const std::string kAutoAttrPrefix = "_thread_";
const std::string kAllKernel = "_kernel_list_first_name";
const std::string kMixAicAllKernel = "_mix_aic_kernel_list_first_name";
const std::string kMixAivAllKernel = "_mix_aiv_kernel_list_first_name";
const std::string kMixAllKernel = "_mix_enhanced_kernel_list_first_name";
const std::string kThreadKernelName = "_thread_kernelname";
const std::string kStaticBinKeySuffix = "_static_bin";
constexpr const ge::char_t *kAttrMemsetKernelBinId = "_memset_kernel_bin_id";
const std::string kMemSet = "memset";
const std::string kAddrRefreshOpStaticBinId = "UpdateModelParam_static_bin";
const std::set<std::string> kMixCorePrefix = {"_mix_aic", "_mix_aiv", "_mix_enhanced"};

const std::map<std::string, uint32_t> kMixBinaryMagic = {
    {"_mix_aiv", RT_DEV_BINARY_MAGIC_ELF_AIVEC},
    {"_mix_aic", RT_DEV_BINARY_MAGIC_ELF_AICUBE},
    {"_mix_enhanced", RT_DEV_BINARY_MAGIC_ELF}
};
}  // namespace

bool IsTbeTask(const OpDescPtr &op_desc) {
  uint32_t run_mode = static_cast<uint32_t>(domi::ImplyType::INVALID);
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_IMPLY_TYPE, run_mode)) {
    return false;
  }

  if (run_mode != static_cast<uint32_t>(domi::ImplyType::TVM)) {
    return false;
  }

  // Skip no_task operator, such as concat and split.
  bool attr_no_task = false;
  const bool get_attr_no_task_flag = AttrUtils::GetBool(op_desc, ATTR_NAME_NOTASK, attr_no_task);
  if (get_attr_no_task_flag && attr_no_task) {
    GELOGI("Node[name:%s, type:%s] does not generate task, skip initialization.",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return false;
  }

  return true;
}

bool IsAllKernelTask(const OpDescPtr &op_desc) {
  return op_desc->HasAttr(kAllKernel) || op_desc->HasAttr(kMixAicAllKernel) || op_desc->HasAttr(kMixAivAllKernel) ||
         op_desc->HasAttr(kMixAllKernel);
}

bool IsNeedAtomicCleanTask(const OpDescPtr &op_desc) {
  bool need_gentask_atomic = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_NEED_GENTASK_ATOMIC, need_gentask_atomic);
  bool is_soft_sync = false;
  (void)AttrUtils::GetBool(op_desc, ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, is_soft_sync);
  return need_gentask_atomic || is_soft_sync;
}

static std::string GetKernelName(const OpDesc &op_desc, const std::string &prefix) {
  std::string kernel_name;
  // mix
  if (kMixCorePrefix.count(prefix) > 0UL) {
    (void)AttrUtils::GetStr(op_desc, prefix + ATTR_NAME_TBE_KERNEL_NAME, kernel_name);
  } else if (prefix.empty()) {
    // aicore
    const bool status = AttrUtils::GetStr(op_desc, ATTR_NAME_TBE_KERNEL_NAME_FOR_LOAD, "_kernelname", kernel_name);
    kernel_name = status ? kernel_name : op_desc.GetName();
  } else if (prefix == kAtomicPrefix) {
    (void)AttrUtils::GetStr(op_desc, ATOMIC_ATTR_TBE_KERNEL_NAME, kernel_name);
  } else {
    // misra
  }
  return kernel_name;
}

static KernelBinPtr GetKernelBin(const OpDesc &op_desc, const TBEKernelStore &tbe_kernel_store,
                                 const std::string &prefix) {
  const std::string ext_kernel_name = prefix + std::string(OP_EXTATTR_NAME_TBE_KERNEL);
  auto tbe_kernel = op_desc.TryGetExtAttr(ext_kernel_name, TBEKernelPtr());
  if (tbe_kernel == nullptr) {
    const auto &kernel_name = GetKernelName(op_desc, prefix);
    tbe_kernel = tbe_kernel_store.FindKernel(kernel_name);
  }
  return tbe_kernel;
}

std::string TBEKernelHandle::GetBinHandleKey(const OpDesc &op_desc, const std::string &prefix,
                                             const bool is_atomic_kernel) const {
  std::string kernel_bin_id;
  if (is_atomic_kernel) {
    (void)ge::AttrUtils::GetStr(op_desc, kAttrMemsetKernelBinId, kernel_bin_id);
  } else {
    (void)ge::AttrUtils::GetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, kernel_bin_id);
  }
  kernel_bin_id += prefix;
  // 编译阶段没有打属性，理论上不应该存在的场景，会导致缓存失效
  if (kernel_bin_id.empty()) {
    (void)AttrUtils::GetStr(op_desc, ATTR_NAME_SESSION_GRAPH_ID, kernel_bin_id);
    kernel_bin_id += std::string("_" + std::to_string(model_id_) + op_desc.GetName());
  }
  // todo 动静态图kernel注册没有归一会导致静态子图卸载时去注册动态图中已经注册的bin，先临时加个后缀规避，待后续动态图kernel注册流程归一后删除。
  kernel_bin_id += kStaticBinKeySuffix;
  return kernel_bin_id;
}

Status TBEKernelHandle::GetAddrAndPrefCnt(const OpDescPtr &op_desc, const std::string &kernel_name,
                                          const std::string &prefix,
                                          std::vector<std::pair<void *, uint32_t>> &addr_and_pref_cnt) const {
  if (IsAllKernelTask(op_desc) && (kernel_name.find(kMemSet) == std::string::npos)) {
    string kernel_key = GetBinHandleKey(*op_desc, prefix, false);
    void *bin_handle = nullptr;
    TBEHandleStore &handle_store = TBEHandleStore::GetInstance();
    if (!handle_store.FindTBEHandle(kernel_key, bin_handle)) {
      GELOGD("[%s]'s kernel with name [%s] not registered.", op_desc->GetName().c_str(), kernel_key.c_str());
      return SUCCESS;
    }
    uint64_t tiling_key = 0U;
    const auto tiling_info =
        op_desc->GetExtAttr<std::shared_ptr<optiling::utils::OpRunInfo>>(ge::ATTR_NAME_OP_RUN_INFO);
    if ((tiling_info != nullptr) && (*tiling_info != nullptr)) {
      tiling_key = (*tiling_info)->GetTilingKey();
    }
    rtKernelDetailInfo_t kernel_info = {};
    GE_CHK_RT_RET(rtKernelGetAddrAndPrefCntV2(bin_handle, tiling_key, nullptr, RT_DYNAMIC_SHAPE_KERNEL, &kernel_info));
    if (static_cast<size_t>(kernel_info.functionInfoNum) >
        (sizeof(kernel_info.functionInfo) / sizeof(kernel_info.functionInfo[0]))) {
      GELOGE(INTERNAL_ERROR, "[Check][Param]function info num[%zu] is larger than the size of functionInfo[%zu]",
             static_cast<size_t>(kernel_info.functionInfoNum),
             (sizeof(kernel_info.functionInfo) / sizeof(kernel_info.functionInfo[0])));
      return INTERNAL_ERROR;
    }
    for (uint8_t i = 0U; i < kernel_info.functionInfoNum; ++i) {
      addr_and_pref_cnt.emplace_back(kernel_info.functionInfo[i].pcAddr, kernel_info.functionInfo[i].prefetchCnt);
      GELOGI("Get [%u] addr 0x%" PRIx64 ", pref_cnt %u for kernel_name %s.",
             i, PtrToValue(kernel_info.functionInfo[i].pcAddr),
             kernel_info.functionInfo[i].prefetchCnt, kernel_name.c_str());
    }
  } else {
    const auto &iter = addr_and_pref_cnt_.find(kernel_name);
    if (iter == addr_and_pref_cnt_.end()) {
      REPORT_INNER_ERR_MSG("E19999", "Get addr and pref cnt failed, kernel_name:%s", kernel_name.c_str());
      GELOGE(INTERNAL_ERROR, "[Check][Param] Get addr and pref cnt failed, kernel_name:%s", kernel_name.c_str());
      return INTERNAL_ERROR;
    }
    addr_and_pref_cnt = iter->second;
  }

  return SUCCESS;
}

/// @ingroup ge
/// @brief Register TBE kernel to RTS for statically compiled kernel.
/// @param [in] op_desc : current op.
/// @param [in] prefix: kernel name attr name prefix.
/// @param [in] tbe_kernel_store: store to find kernel.
/// @param [in] is_atomic_kernel: atomic kernel flag.
/// @return SUCCESS / other error code.
Status TBEKernelHandle::RegisterStaticHandle(const OpDescPtr &op_desc, const std::string &prefix,
                                             const TBEKernelStore &tbe_kernel_store, const bool is_atomic_kernel) {
  GE_CHECK_NOTNULL(op_desc);
  const auto tbe_kernel = GetKernelBin(*op_desc, tbe_kernel_store, prefix);
  GE_ASSERT_NOTNULL(tbe_kernel, "Cannot find op kernel for node [%s] with prefix [%s].", op_desc->GetNamePtr(),
                    prefix.c_str());
  const std::string bin_handle_key = GetBinHandleKey(*op_desc, prefix, is_atomic_kernel);
  return FunctionRegister(op_desc, bin_handle_key, tbe_kernel, prefix);
}

///
/// @ingroup ge
/// @brief Register TBE kernel to RTS for FFTS Task.
/// @param [in] op_desc : current op.
/// @return SUCCESS / other error code.
///
Status TBEKernelHandle::RegisterAutoThreadHandle(const OpDescPtr &op_desc, const TBEKernelStore &tbe_kernel_store) {
  auto tbe_kernel = op_desc->TryGetExtAttr(OP_EXTATTR_NAME_THREAD_TBE_KERNEL, std::vector<OpKernelBinPtr>{});
  if (tbe_kernel.empty()) {
    std::vector<std::string> thread_kernel_names;
    (void)AttrUtils::GetListStr(op_desc, kThreadKernelName, thread_kernel_names);
    if (thread_kernel_names.empty()) {
      REPORT_INNER_ERR_MSG("E19999", "[%s] tbe kernel is empty", op_desc->GetName().c_str());
      GELOGE(INTERNAL_ERROR, "[Check][Param] [%s] tbe kernel is empty", op_desc->GetName().c_str());
      return INTERNAL_ERROR;
    }

    for (const auto &name : thread_kernel_names) {
      auto kernel_bin = tbe_kernel_store.FindKernel(name);
      GE_CHECK_NOTNULL(kernel_bin);
      tbe_kernel.push_back(kernel_bin);
    }
  }

  std::vector<std::string> bin_file_keys;
  (void)AttrUtils::GetListStr(op_desc, kStubFuncName, bin_file_keys);
  if (tbe_kernel.size() != bin_file_keys.size()) {
    REPORT_INNER_ERR_MSG("E19999", "[%s] number of bin_file != number of file_name, bin_file_num=%zu, file_name_num=%zu",
                       op_desc->GetName().c_str(), tbe_kernel.size(), bin_file_keys.size());
    GELOGE(INTERNAL_ERROR,
           "[Check][Param] [%s] number of bin_file != number of file_name, bin_file_num=%zu, file_name_num=%zu",
           op_desc->GetName().c_str(), tbe_kernel.size(), bin_file_keys.size());
    return INTERNAL_ERROR;
  }
  const size_t num = tbe_kernel.size();
  GELOGD("Kernel bin num is %zu", num);
  for (size_t i = 0U; i < num; i++) {
    if (tbe_kernel[i] == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Tbe kernel for op:%s is nullptr.", op_desc->GetName().c_str());
      GELOGE(INTERNAL_ERROR, "[Check][Param] TBE: tvm bin file of %s is nullptr when ffts.",
             op_desc->GetName().c_str());
      return INTERNAL_ERROR;
    }
    GE_CHK_STATUS_RET(FunctionRegister(op_desc, bin_file_keys[i], tbe_kernel[i], kAutoAttrPrefix,
        static_cast<uint32_t>(i)), "Function register of No. %zu bin file %s failed.", i, bin_file_keys[i].c_str());
  }
  return SUCCESS;
}

Status TBEKernelHandle::FunctionRegister(const OpDescPtr &op_desc, const std::string &bin_handle_key,
                                         const OpKernelBinPtr &tbe_kernel, const std::string &prefix,
                                         const uint32_t thread_index) {
  TBEHandleStore &bin_handle_store = TBEHandleStore::GetInstance();
  void *bin_handle = nullptr;
  std::lock_guard<std::recursive_mutex> lock(TBEHandleStore::mutex_);
  if (!bin_handle_store.FindTBEHandle(bin_handle_key, bin_handle)) {
    GELOGI("TBE: Register kernel for op[%s], key[%s].", op_desc->GetName().c_str(), bin_handle_key.c_str());
    rtDevBinary_t binary;
    GE_CHK_STATUS_RET(InitBinaryMagic(op_desc, thread_index, binary, prefix), "Init binary magic of %s failed.",
                      op_desc->GetName().c_str());
    binary.version = 0U;
    binary.data = tbe_kernel->GetBinData();
    binary.length = tbe_kernel->GetBinDataSize();
    GELOGD("TBE: binary.length: %" PRIu64, binary.length);
    GE_CHK_RT_RET(rtDevBinaryRegister(&binary, &bin_handle));

    GE_CHK_STATUS_RET(InitMetaData(op_desc, thread_index, bin_handle, prefix), "Init tvm meta data of %s failed.",
                      op_desc->GetName().c_str());
    GE_ASSERT_SUCCESS(bin_handle_store.StoreTBEHandle(bin_handle_key, bin_handle, tbe_kernel));

    // 地址刷新算子，原流程为模型加载时注册，模型析构时去注册，在模型反复加载执行卸载场景中出现bin文件的二进制数据
    // 在hbm和icache不一致情况，hbm为全部正确值，icache为部分正确
    // 规避方案修改为只在首个模型加载时注册，在TBEHandleStore对象析构时去注册
    if (bin_handle_key == kAddrRefreshOpStaticBinId) {
      GELOGI("TBE: Bin with key [%s] refer tbe handle.", bin_handle_key.c_str());
      bin_handle_store.ReferTBEHandle(bin_handle_key);
    }
  } else {
    GELOGI("TBE: Bin with key [%s] has been registered.", bin_handle_key.c_str());
    bin_handle_store.ReferTBEHandle(bin_handle_key);
  }
  StoreTbeHandle(bin_handle_key);

  // func reg
  std::string kernel_name;
  GE_CHK_STATUS_RET(InitKernelName(op_desc, thread_index, kernel_name, prefix), "Init kernel name of %s failed.",
                    op_desc->GetName().c_str());

  bool inserted = false;
  const void *const kernel_unique_ids_addr = bin_handle_store.GetUniqueIdPtr(bin_handle, kernel_name, inserted);
  if (inserted) {
    GE_CHK_RT_RET(
        rtFunctionRegister(bin_handle, kernel_unique_ids_addr, bin_handle_key.c_str(), kernel_name.c_str(), 0U));
  }

  uint64_t tiling_key = 0U;
  const auto tiling_info = op_desc->GetExtAttr<std::shared_ptr<optiling::utils::OpRunInfo>>(ge::ATTR_NAME_OP_RUN_INFO);
  if ((tiling_info != nullptr) && (*tiling_info != nullptr)) {
    tiling_key = (*tiling_info)->GetTilingKey();
  }
  rtKernelDetailInfo_t kernel_info = {};
  GE_CHK_RT_RET(rtKernelGetAddrAndPrefCntV2(bin_handle, tiling_key, kernel_unique_ids_addr, RT_STATIC_SHAPE_KERNEL,
                                            &kernel_info));
  if (static_cast<size_t>(kernel_info.functionInfoNum) >
      (sizeof(kernel_info.functionInfo) / sizeof(kernel_info.functionInfo[0]))) {
    GELOGE(INTERNAL_ERROR, "[Check][Param]function info num[%zu] is larger than the size of functionInfo[%zu]",
           static_cast<size_t>(kernel_info.functionInfoNum),
           (sizeof(kernel_info.functionInfo) / sizeof(kernel_info.functionInfo[0])));
    return INTERNAL_ERROR;
  }
  addr_and_pref_cnt_[kernel_name].clear();
  for (uint8_t i = 0U; i < kernel_info.functionInfoNum; ++i) {
    addr_and_pref_cnt_[kernel_name].emplace_back(kernel_info.functionInfo[i].pcAddr,
                                                 kernel_info.functionInfo[i].prefetchCnt);
    GELOGI("Get [%u] addr 0x%" PRIx64 ", pref_cnt %u for kernel_name %s.",
           i, PtrToValue(kernel_info.functionInfo[i].pcAddr),
           kernel_info.functionInfo[i].prefetchCnt, kernel_name.c_str());
  }

  return SUCCESS;
}

Status TBEKernelHandle::InitBinaryMagic(const OpDescPtr &op_desc, const uint32_t thread_index,
                                        rtDevBinary_t &binary, const std::string &prefix) const {
  static const std::map<std::string, uint32_t> binary_magics = {
      {"RT_DEV_BINARY_MAGIC_ELF_AICPU", RT_DEV_BINARY_MAGIC_ELF_AICPU},
      {"RT_DEV_BINARY_MAGIC_ELF", RT_DEV_BINARY_MAGIC_ELF},
      {"RT_DEV_BINARY_MAGIC_ELF_AIVEC", RT_DEV_BINARY_MAGIC_ELF_AIVEC},
      {"RT_DEV_BINARY_MAGIC_ELF_AICUBE", RT_DEV_BINARY_MAGIC_ELF_AICUBE}
  };

  std::string json_string;
  const std::string &tvm_magic = prefix + TVM_ATTR_NAME_MAGIC;
  if (thread_index != UINT32_MAX) {
    std::vector<std::string> json_list;
    (void)AttrUtils::GetListStr(op_desc, tvm_magic, json_list);
    if (json_list.size() <= thread_index) {
      GELOGE(INTERNAL_ERROR, "[Check][Param] failed. Attr is %s, thread index is %u, json list size is %zu.",
             tvm_magic.c_str(), thread_index, json_list.size());
      return INTERNAL_ERROR;
    }
    json_string = json_list[static_cast<size_t>(thread_index)];
  } else {
    if (prefix.empty() || prefix == kAtomicPrefix) {
      (void)AttrUtils::GetStr(op_desc, tvm_magic, json_string);
    } else {
      const std::map<std::string, uint32_t>::const_iterator it = kMixBinaryMagic.find(prefix);
      if (it != kMixBinaryMagic.cend()) {
        binary.magic = it->second;
        return SUCCESS;
      }
    }
  }

  const std::map<std::string, uint32_t>::const_iterator it = binary_magics.find(json_string);
  if (it == binary_magics.cend()) {
    REPORT_INNER_ERR_MSG("E19999", "Attr:%s value:%s in op:%s(%s), model_id:%u, check invalid",
                       tvm_magic.c_str(), json_string.c_str(), op_desc->GetName().c_str(),
                       op_desc->GetType().c_str(), model_id_);
    GELOGE(PARAM_INVALID, "[Check][Param] Attr:%s value:%s in op:%s(%s), model_id:%u, check invalid",
           tvm_magic.c_str(), json_string.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
    return PARAM_INVALID;
  }
  binary.magic = it->second;
  return SUCCESS;
}

Status TBEKernelHandle::InitMetaData(const OpDescPtr &op_desc, const uint32_t thread_index,
                                     void *bin_handle, const std::string &prefix) const {
  std::string meta_data;
  const std::string &tvm_metadata = prefix + TVM_ATTR_NAME_METADATA;
  if (thread_index != UINT32_MAX) {
    std::vector<std::string> meta_data_list;
    (void)AttrUtils::GetListStr(op_desc, tvm_metadata, meta_data_list);
    if (meta_data_list.size() <= thread_index) {
      GELOGE(INTERNAL_ERROR, "[Check][Param] failed, attr is %s, thread index is %u, meta data list size is %zu.",
             tvm_metadata.c_str(), thread_index, meta_data_list.size());
      return INTERNAL_ERROR;
    }
    meta_data = meta_data_list[static_cast<size_t>(thread_index)];
  } else {
    (void)AttrUtils::GetStr(op_desc, tvm_metadata, meta_data);
  }
  GELOGD("TBE: meta data: %s", meta_data.empty() ? "null" : meta_data.c_str());
  if (!meta_data.empty()) {
    GE_CHK_RT_RET(rtMetadataRegister(bin_handle, meta_data.c_str()));
  }
  return SUCCESS;
}

Status TBEKernelHandle::InitKernelName(const OpDescPtr &op_desc, const uint32_t thread_index,
                                       std::string &kernel_name, const std::string &prefix) const {
  if (thread_index != UINT32_MAX) {
    return ThreadKernelName(op_desc, thread_index, kernel_name);
  } else {
    const std::string attr_kernel_name = prefix + op_desc->GetName() + "_kernelname";
    (void)AttrUtils::GetStr(op_desc, attr_kernel_name, "_kernelname", kernel_name);
    GELOGD("[%s] attr: %s, kernel: %s.", op_desc->GetName().c_str(), attr_kernel_name.c_str(), kernel_name.c_str());
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Init kernel name for thread slice.
/// @return SUCCESS / other error code.
///
Status TBEKernelHandle::ThreadKernelName(const OpDescPtr &op_desc, const uint32_t thread_index,
                                         std::string &kernel_name) const {
  const std::string kernel_name_attr = "_thread_kernelname";
  std::vector<std::string> thread_kernel_names;
  (void)AttrUtils::GetListStr(op_desc, kernel_name_attr, thread_kernel_names);
  if (thread_kernel_names.size() <= thread_index) {
    GELOGE(INTERNAL_ERROR, "[Check][Param] failed, attr is %s, thread index is %u, kernel name list size is %zu.",
           kernel_name_attr.c_str(), thread_index, thread_kernel_names.size());
    return INTERNAL_ERROR;
  }

  kernel_name = thread_kernel_names[static_cast<size_t>(thread_index)];
  GELOGD("[%s] attr: %s, kernel: %s.", op_desc->GetName().c_str(), kernel_name_attr.c_str(), kernel_name.c_str());
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Register all kernels for dynamically compiled op.
/// @return SUCCESS / other error code.
///
Status TBEKernelHandle::RegisterDynamicKernel(const OpDescPtr &op_desc, const std::string &prefix,
                                              const TBEKernelStore &tbe_kernel_store) {
  const auto &handle_key = GetBinHandleKey(*op_desc, prefix, false);
  void *bin_handle = nullptr;
  TBEHandleStore &bin_handle_store = TBEHandleStore::GetInstance();
  const std::lock_guard<std::recursive_mutex> lock(TBEHandleStore::mutex_);
  if (!bin_handle_store.FindTBEHandle(handle_key, bin_handle)) {
    auto tbe_kernel = GetKernelBin(*op_desc, tbe_kernel_store, prefix);
    GE_ASSERT_NOTNULL(tbe_kernel, "Cannot find op kernel for node [%s] with prefix [%s].", op_desc->GetNamePtr(),
                      prefix.c_str());
    GELOGI("TBE: Register kernel for op[%s], key[%s].", op_desc->GetName().c_str(), handle_key.c_str());
    rtDevBinary_t binary;
    GE_CHK_STATUS_RET_NOLOG(InitBinaryMagic(op_desc, UINT32_MAX, binary, prefix));
    binary.version = 0U;
    binary.data = tbe_kernel->GetBinData();
    binary.length = tbe_kernel->GetBinDataSize();
    GELOGD("TBE: binary.length: %" PRIu64, binary.length);
    GE_CHK_RT_RET(rtRegisterAllKernel(&binary, &bin_handle));
    GE_ASSERT_SUCCESS(bin_handle_store.StoreTBEHandle(handle_key, bin_handle, tbe_kernel));
  } else {
    GELOGI("TBE: Bin with key [%s] has been registered.", handle_key.c_str());
    bin_handle_store.ReferTBEHandle(handle_key);
  }
  StoreTbeHandle(handle_key);
  return SUCCESS;
}

void TBEKernelHandle::StoreTbeHandle(const std::string &handle_key) {
  const auto it = used_tbe_handle_map_.find(handle_key);
  if (it != used_tbe_handle_map_.end()) {
    it->second++;
    return;
  }
  used_tbe_handle_map_[handle_key] = 1U;  // Init used num to 1.
}

void TBEKernelHandle::CleanTbeHandle() noexcept {
  TBEHandleStore &kernel_store = TBEHandleStore::GetInstance();
  kernel_store.EraseTBEHandle(used_tbe_handle_map_);
  used_tbe_handle_map_.clear();
}

Status TBEKernelHandle::Register(const OpDescPtr &op_desc, const std::string &prefix) {
  if (IsAllKernelTask(op_desc)) {  // dynamically compiled kernel
    GE_CHK_STATUS_RET_NOLOG(RegisterDynamicKernel(op_desc, prefix));
  } else {  // static shape
    GE_CHK_STATUS_RET_NOLOG(RegisterStaticHandle(op_desc, prefix));
  }
  return SUCCESS;
}
} // namespace ge
