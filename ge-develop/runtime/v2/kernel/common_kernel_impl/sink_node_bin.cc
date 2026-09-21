/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sink_node_bin.h"
#include <list>
#include "graph/ge_error_codes.h"
#include "register/kernel_registry.h"
#include "graph/utils/math_util.h"
#include "runtime/kernel.h"
#include "kernel/kernel_log.h"
#include "runtime/rt_ffts_plus.h"
#include "common/checker.h"
#include "engine/aicore/kernel/aicore_update_kernel.h"
#include "exe_graph/runtime/extended_kernel_context.h"

namespace gert {
namespace kernel {
namespace {
std::mutex g_bin_hash_to_handles_lock;
std::unordered_map<std::string, void*> g_bin_hash_to_handles;
std::string GetComputeNodeName(const KernelContext *context) {
  if ((context == nullptr) ||
    (reinterpret_cast<const ExtendedKernelContext *>(context)->GetComputeNodeInfo() == nullptr) ||
    (reinterpret_cast<const ExtendedKernelContext *>(context)->GetComputeNodeInfo()->GetNodeName() == nullptr)) {
    return "unknown";
  }
  return reinterpret_cast<const ExtendedKernelContext *>(context)->GetComputeNodeInfo()->GetNodeName();
}
}
UINT32 SinkNodeBinWithoutHandle(KernelContext *context) {
  auto bin_data_holder = context->GetInputValue<kernel::BinData *>(0);
  auto stub_name = context->GetInputValue<char *>(1);
  auto metadata = context->GetInputValue<char *>(2);
  auto kernel_name = context->GetInputValue<char *>(3);
  auto stub_func = context->GetOutputPointer<void *>(0);
  const rtError_t ret_ret = rtQueryFunctionRegistered(stub_name);
  if (ret_ret != RT_ERROR_NONE) {
    if (bin_data_holder == nullptr || stub_name == nullptr || metadata == nullptr || kernel_name == nullptr) {
      GELOGE(ge::FAILED, "SinkNodeBinWithoutHandle para check failed");
      return ge::GRAPH_FAILED;
    }
    void *bin_handle;
    rtDevBinary_t *bin_data = reinterpret_cast<rtDevBinary_t *>(bin_data_holder);
    bin_data->data = &bin_data_holder->placeholder;
    GELOGI("Register static kernel size %lu, node %s, stub %s", bin_data->length, GetComputeNodeName(context).c_str(),
           stub_name);
    GE_ASSERT_RT_OK(rtDevBinaryRegister(bin_data, &bin_handle));
    if (strlen(metadata) > 1) {
      GE_ASSERT_RT_OK(rtMetadataRegister(bin_handle, metadata));
    }
    static std::list<uint8_t> kernel_unique_ids;
    static std::mutex mtx;
    void *kernel_unique_ids_addr;
    {
      std::unique_lock<std::mutex> lk(mtx);
      kernel_unique_ids.push_back(0U); // store unique id persistent for unique 'stub func' which means kernel id
      kernel_unique_ids_addr = &kernel_unique_ids.back();
    }
    GE_ASSERT_RT_OK(rtFunctionRegister(bin_handle, kernel_unique_ids_addr, stub_name, kernel_name, 0U));
  }
  GE_ASSERT_NOTNULL(stub_func);
  GE_ASSERT_RT_OK(rtGetFunctionByName(stub_name, stub_func));
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(SinkNodeBinWithoutHandle).RunFunc(SinkNodeBinWithoutHandle);

UINT32 SinkNodeBinWithHandle(KernelContext *context) {
  auto bin_data_holder = context->GetInputValue<kernel::BinData *>(0);
  if (bin_data_holder == nullptr) {
    GELOGE(ge::FAILED, "SinkNodeBinWithHandle para check failed");
    return ge::GRAPH_FAILED;
  }
  auto bin_key = context->GetInputValue<char *>(1);
  GE_ASSERT_NOTNULL(bin_key);

  rtDevBinary_t *bin_data = reinterpret_cast<rtDevBinary_t *>(bin_data_holder);
  bin_data->data = &bin_data_holder->placeholder;
  auto bin_handle = context->GetOutputPointer<void *>(0);
  GE_ASSERT_NOTNULL(bin_handle);

  std::string bin_hash(bin_key);
  if (bin_hash.empty()) {
    GELOGI("Register dynamic kernel empty hash size %lu, node %s", bin_data->length,
           GetComputeNodeName(context).c_str());
    GE_ASSERT_RT_OK(rtRegisterAllKernel(bin_data, bin_handle));
    return ge::GRAPH_SUCCESS;
  }

  std::unique_lock<std::mutex> lk(g_bin_hash_to_handles_lock);
  const std::unordered_map<std::string, void*>::const_iterator iter = g_bin_hash_to_handles.find(bin_hash);
  if (iter != g_bin_hash_to_handles.cend()) {
    *bin_handle = iter->second;
    return ge::GRAPH_SUCCESS;
  }

  GELOGI("Register dynamic kernel size %lu, node %s, hash %s",
         bin_data->length, GetComputeNodeName(context).c_str(), bin_hash.c_str());
  GE_ASSERT_RT_OK(rtRegisterAllKernel(bin_data, bin_handle));
  g_bin_hash_to_handles[bin_hash] = *bin_handle;

  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(SinkNodeBinWithHandle).RunFunc(SinkNodeBinWithHandle);

UINT32 SinkFFTSAICoreNodeBin(KernelContext *context) {
  auto bin_data_holder = context->GetInputValue<kernel::BinData *>(0U);
  GE_ASSERT_NOTNULL(bin_data_holder);
  auto bin_handle = context->GetOutputPointer<void *>(0U);
  GE_ASSERT_NOTNULL(bin_handle);
  rtDevBinary_t *bin_data = reinterpret_cast<rtDevBinary_t *>(bin_data_holder);
  bin_data->data = &bin_data_holder->placeholder;
  rtError_t ret = rtRegisterAllKernel(bin_data, bin_handle);
  if (ret != RT_ERROR_NONE) {
    GELOGE(ge::FAILED, "[kernel][SinkFFTSAICoreNodeBin] Register kernel failed.");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(SinkFFTSAICoreNodeBin).RunFunc(SinkFFTSAICoreNodeBin);

UINT32 GetFFTSAICorePcAndPref(KernelContext *context) {
  auto bin_handle = context->GetInputValue<void *>(static_cast<int32_t>(0U));
  auto none_tail_key = context->GetInputValue<size_t>(1U);
  auto tail_key = context->GetInputValue<size_t>(2U);
  void *none_tail_addr = nullptr;
  uint32_t none_tail_pref = 0U;
  void *tail_addr = nullptr;
  uint32_t tail_pref = 0U;
  auto sink_ret = context->GetOutputPointer<AICoreSinkRet>(0U);
  if (sink_ret == nullptr) {
    GELOGE(ge::FAILED, "[kernel][SinkFFTSAICoreNodeBin] Flush data is null.");
    return ge::GRAPH_FAILED;
  }
  if (rtKernelGetAddrAndPrefCnt(bin_handle, none_tail_key, nullptr, RT_DYNAMIC_SHAPE_KERNEL, &none_tail_addr,
                                &none_tail_pref) != RT_ERROR_NONE) {
    GELOGE(ge::FAILED, "[kernel][SinkFFTSAICoreNodeBin] get none tail start pc failed.");
    return ge::GRAPH_FAILED;
  }
  if (none_tail_key != tail_key) {
    if (rtKernelGetAddrAndPrefCnt(bin_handle, tail_key, nullptr, RT_DYNAMIC_SHAPE_KERNEL, &tail_addr,
                                  &tail_pref) != RT_ERROR_NONE) {
      GELOGE(ge::FAILED, "[kernel][SinkFFTSAICoreNodeBin] get tail start pc failed.");
      return ge::GRAPH_FAILED;
    }
  } else {
    tail_addr = none_tail_addr;
    tail_pref = none_tail_pref;
  }
  sink_ret->aic_non_tail_task_start_pc = reinterpret_cast<uintptr_t>(none_tail_addr);
  sink_ret->aic_tail_task_start_pc = reinterpret_cast<uintptr_t>(tail_addr);
  sink_ret->aic_icache_prefetch_cnt = std::min(none_tail_pref, tail_pref);
  GELOGD("Sink bin get pc[%lx][%lx].", sink_ret->aic_non_tail_task_start_pc, sink_ret->aic_tail_task_start_pc);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateSinkRet(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto chain = context->GetOutput(0);
  GE_ASSERT_NOTNULL(chain);
  chain->SetWithDefaultDeleter(new AICoreSinkRet());
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(GetFFTSAICorePcAndPref).RunFunc(GetFFTSAICorePcAndPref).OutputsCreator(CreateSinkRet);

UINT32 SinkFFTSStaManualNodeBin(KernelContext *context) {
  auto sink_ret = context->GetOutputPointer<AICoreSinkRet>(0U);
  GE_ASSERT_NOTNULL(sink_ret);
  auto kernel_name = context->GetInputValue<char *>(0U);
  GE_ASSERT_NOTNULL(kernel_name);
  auto stub_name = context->GetInputValue<char *>(1U);
  GE_ASSERT_NOTNULL(stub_name);
  auto metadata = context->GetInputValue<char *>(2U);
  GE_ASSERT_NOTNULL(metadata);
  auto bin_data_holder = context->GetInputValue<kernel::BinData *>(3U);
  GE_ASSERT_NOTNULL(bin_data_holder);
  void *bin_handle = nullptr;
  rtDevBinary_t *bin_data = reinterpret_cast<rtDevBinary_t *>(bin_data_holder);
  bin_data->data = &bin_data_holder->placeholder;
  GE_ASSERT_RT_OK(rtDevBinaryRegister(bin_data, &bin_handle));
  if (strlen(metadata) > 1U) {
    GE_ASSERT_RT_OK(rtMetadataRegister(bin_handle, metadata));
  }
  static std::list<uint8_t> kernel_unique_ids;
  static std::mutex mtx;
  void *kernel_unique_ids_addr;
  {
    std::unique_lock<std::mutex> lk(mtx);
    kernel_unique_ids.push_back(0U); // store unique id persistent for unique 'stub func' which means kernel id
    kernel_unique_ids_addr = &kernel_unique_ids.back();
  }
  GE_ASSERT_RT_OK(rtFunctionRegister(bin_handle, kernel_unique_ids_addr, stub_name, kernel_name, 0U));
  uint32_t prefetch_cnt = 0U;
  void *addr = nullptr;
  GE_CHK_RT_RET(rtKernelGetAddrAndPrefCnt(bin_handle, 0U, kernel_unique_ids_addr, RT_STATIC_SHAPE_KERNEL, &addr,
                                          &prefetch_cnt));
  GELOGD("Get static pc addr:%lx pre_cnt:%u.", addr, prefetch_cnt);
  sink_ret->aic_icache_prefetch_cnt = prefetch_cnt;
  sink_ret->aic_non_tail_task_start_pc = reinterpret_cast<uintptr_t>(addr);
  sink_ret->aic_tail_task_start_pc = reinterpret_cast<uintptr_t>(addr);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(SinkFFTSStaManualNodeBin).RunFunc(SinkFFTSStaManualNodeBin).OutputsCreator(CreateSinkRet);


UINT32 SinkFFTSStaAutoNodeBin(KernelContext *context) {
  auto sink_ret = context->GetOutputPointer<AICoreSinkRet>(0U);
  GE_ASSERT_NOTNULL(sink_ret);
  auto kernel_num = context->GetInputValue<size_t>(0U);
  KLOGD("Sink static auto kernel with num[%zu].", kernel_num);
  size_t start_in = 1U;
  for (size_t i = 0; i < kernel_num; ++i) {
    auto kernel_name = context->GetInputValue<char *>(start_in++);
    GE_ASSERT_NOTNULL(kernel_name);
    auto stub_name = context->GetInputValue<char *>(start_in++);
    GE_ASSERT_NOTNULL(stub_name);
    auto metadata = context->GetInputValue<char *>(start_in++);
    GE_ASSERT_NOTNULL(metadata);
    auto bin_data_holder = context->GetInputValue<kernel::BinData *>(start_in++);
    GE_ASSERT_NOTNULL(bin_data_holder);
    void *bin_handle = nullptr;
    rtDevBinary_t *bin_data = reinterpret_cast<rtDevBinary_t *>(bin_data_holder);
    bin_data->data = &bin_data_holder->placeholder;
    GE_ASSERT_RT_OK(rtDevBinaryRegister(bin_data, &bin_handle));
    if (strlen(metadata) > 1U) {
      GE_ASSERT_RT_OK(rtMetadataRegister(bin_handle, metadata));
    }
    static std::list<uint8_t> kernel_unique_ids;
    static std::mutex mtx;
    void *kernel_unique_ids_addr;
    {
      std::unique_lock<std::mutex> lk(mtx);
      kernel_unique_ids.push_back(0U); // store unique id persistent for unique 'stub func' which means kernel id
      kernel_unique_ids_addr = &kernel_unique_ids.back();
    }
    GE_ASSERT_RT_OK(rtFunctionRegister(bin_handle, kernel_unique_ids_addr, stub_name, kernel_name, 0U));
    uint32_t prefetch_cnt = 0U;
    void *addr = nullptr;
    GE_CHK_RT_RET(rtKernelGetAddrAndPrefCnt(bin_handle, 0U, kernel_unique_ids_addr, RT_STATIC_SHAPE_KERNEL, &addr,
                                            &prefetch_cnt));
    GELOGD("Get [%zu] kernel[%s] pc addr:%lx pre_cnt:%u.", i, kernel_name, addr, prefetch_cnt);
    if (i == 0U) {
      sink_ret->aic_icache_prefetch_cnt = prefetch_cnt;
      sink_ret->aic_non_tail_task_start_pc = reinterpret_cast<uintptr_t>(addr);
    } else {
      sink_ret->aic_tail_task_start_pc = reinterpret_cast<uintptr_t>(addr);
      sink_ret->aic_icache_prefetch_cnt = std::min(sink_ret->aic_icache_prefetch_cnt, prefetch_cnt);
    }
  }
  if (kernel_num == 1U) {
    sink_ret->aic_tail_task_start_pc = sink_ret->aic_non_tail_task_start_pc;
  }
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(SinkFFTSStaAutoNodeBin).RunFunc(SinkFFTSStaAutoNodeBin).OutputsCreator(CreateSinkRet);

const std::string kTaskCubeTBEKernelPrefixAic = "_mix_aic";
const std::string kTaskVectorTBEKernelPrefixAiv = "_mix_aiv";

UINT32 SinkMixDyNodeBin(KernelContext *context) {
  auto bin_data_holder = context->GetInputValue<kernel::BinData *>(0U);
  if (bin_data_holder == nullptr) {
    GELOGE(ge::FAILED, "[kernel][SinkFFTSAICoreNodeBin] Bin data is nullptr.");
    return ge::GRAPH_FAILED;
  }
  auto ret_handle = context->GetOutputPointer<void*>(0U);
  GE_ASSERT_NOTNULL(ret_handle);
  rtDevBinary_t *bin_data = reinterpret_cast<rtDevBinary_t *>(bin_data_holder);
  bin_data->data = &bin_data_holder->placeholder;
  auto bin_key = context->GetInputValue<char *>(1);
  GE_ASSERT_NOTNULL(bin_key);
  std::string bin_hash(bin_key);
  if (!bin_hash.empty()) {
    std::unique_lock<std::mutex> lk(g_bin_hash_to_handles_lock);
    const std::unordered_map<std::string, void*>::const_iterator iter = g_bin_hash_to_handles.find(bin_hash);
    if (iter != g_bin_hash_to_handles.cend()) {
      *ret_handle = iter->second;
      return ge::GRAPH_SUCCESS;
    }
    GELOGI("Register dynamic kernel size %lu, node %s, hash %s",
           bin_data->length, GetComputeNodeName(context).c_str(), bin_hash.c_str());
    GE_ASSERT_RT_OK(rtRegisterAllKernel(bin_data, ret_handle));
    g_bin_hash_to_handles[bin_hash] = *ret_handle;
    return ge::GRAPH_SUCCESS;
  }
  GE_CHK_RT_RET(rtRegisterAllKernel(bin_data, ret_handle));
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(SinkMixDyNodeBin).RunFunc(SinkMixDyNodeBin);
bool CheckKernelInfoValid(const rtKernelDetailInfo_t &kernel_info) {
  return kernel_info.functionInfoNum < static_cast<uint8_t>(MIX_KERNEL_FUNC_NUM::MIX_INVALID_KERNEL);
}
UINT32 ParseKernelInfo(const rtKernelDetailInfo_t &kernel_info, const rtKernelDetailInfo_t &tail_kernel_info,
                       AICoreSinkRet *sink_ret) {
  GELOGD("start parse kernelinfo: functionInfoNum: %u", kernel_info.functionInfoNum);
  if (kernel_info.functionInfoNum == static_cast<uint8_t>(MIX_KERNEL_FUNC_NUM::MIX_WITH_ONE_KERNEL)) {
    const auto &info = kernel_info.functionInfo[0];
    const auto &tail_info = tail_kernel_info.functionInfo[0];
    GELOGD("mixType is %u, prefetchCnt is %u pcAddr is Ox%x", info.mixType, info.prefetchCnt, info.pcAddr);
    if (info.mixType == static_cast<uint8_t>(MIX_KERNEL_TYPE::MIX_AIC_ONLY)) {
      sink_ret->aic_non_tail_task_start_pc = reinterpret_cast<uintptr_t>(info.pcAddr);
      sink_ret->aic_icache_prefetch_cnt = std::min(info.prefetchCnt, tail_info.prefetchCnt);
      sink_ret->aic_tail_task_start_pc = reinterpret_cast<uintptr_t>(tail_info.pcAddr);
    } else if (info.mixType == static_cast<uint8_t>(MIX_KERNEL_TYPE::MIX_AIV_ONLY)) {
      sink_ret->aiv_non_tail_task_start_pc = reinterpret_cast<uintptr_t>(info.pcAddr);
      sink_ret->aiv_icache_prefetch_cnt = std::min(info.prefetchCnt, tail_info.prefetchCnt);
      sink_ret->aiv_tail_task_start_pc = reinterpret_cast<uintptr_t>(tail_info.pcAddr);
    } else {
      GELOGD("find invalid mixType %u with kernel num is 1", info.mixType);
    }
    return ge::SUCCESS;
  }

  if (kernel_info.functionInfoNum == static_cast<uint8_t>(MIX_KERNEL_FUNC_NUM::MIX_WITH_TWO_KERNEL)) {
    GELOGD("aic mixType is %u, prefetchCnt is %u pcAddr is Ox%lx", kernel_info.functionInfo[0].mixType,
           kernel_info.functionInfo[0].prefetchCnt, kernel_info.functionInfo[0].pcAddr);
    GELOGD("aiv mixType is %u, prefetchCnt is %u pcAddr is Ox%lx", kernel_info.functionInfo[1].mixType,
           kernel_info.functionInfo[1].prefetchCnt, kernel_info.functionInfo[1].pcAddr);
    sink_ret->aic_non_tail_task_start_pc = reinterpret_cast<uintptr_t>(kernel_info.functionInfo[0].pcAddr);
    sink_ret->aic_icache_prefetch_cnt = std::min(kernel_info.functionInfo[0].prefetchCnt,
                                                 tail_kernel_info.functionInfo[0].prefetchCnt);;
    sink_ret->aic_tail_task_start_pc = reinterpret_cast<uintptr_t>(tail_kernel_info.functionInfo[0].pcAddr);

    sink_ret->aiv_non_tail_task_start_pc = reinterpret_cast<uintptr_t>(kernel_info.functionInfo[1].pcAddr);
    sink_ret->aiv_icache_prefetch_cnt = std::min(kernel_info.functionInfo[1].prefetchCnt,
                                                 tail_kernel_info.functionInfo[1].prefetchCnt);;
    sink_ret->aiv_tail_task_start_pc = reinterpret_cast<uintptr_t>(tail_kernel_info.functionInfo[1].pcAddr);
    return ge::SUCCESS;
  }
  return ge::FAILED;
}

ge::graphStatus GetMixDynamicPC(const KernelFunctionCtx &ctx, const std::string &prefix, AICoreSinkRet *sink_ret) {
  uint32_t prefetch_cnt = 0U;
  void *addr = nullptr;
  GE_CHK_RT_RET(rtKernelGetAddrAndPrefCnt(ctx.bin_handler, ctx.tiling_key, ctx.kernel_unique_ids_addr, ctx.flag,
                                          &addr, &prefetch_cnt));
  uint32_t tail_prefetch_cnt = 0U;
  void *tail_addr = nullptr;
  if (ctx.tail_tiling_key == ctx.tiling_key) {
    tail_prefetch_cnt = prefetch_cnt;
    tail_addr = addr;
  } else {
    GE_CHK_RT_RET(rtKernelGetAddrAndPrefCnt(ctx.bin_handler, ctx.tail_tiling_key, ctx.kernel_unique_ids_addr, ctx.flag,
                                            &tail_addr, &tail_prefetch_cnt));
  }
  GELOGD("Get mix node with flag: %u prefix:%s pc addr:%p pre_cnt:%u.", ctx.flag, prefix.c_str(), addr,
         prefetch_cnt);
  if (prefix == kTaskCubeTBEKernelPrefixAic) {
    sink_ret->aic_non_tail_task_start_pc = reinterpret_cast<uintptr_t>(addr);
    sink_ret->aic_tail_task_start_pc = reinterpret_cast<uintptr_t>(tail_addr);
    sink_ret->aic_icache_prefetch_cnt = std::min(prefetch_cnt, tail_prefetch_cnt);
  } else {
    sink_ret->aiv_non_tail_task_start_pc = reinterpret_cast<uintptr_t>(addr);
    sink_ret->aiv_tail_task_start_pc = reinterpret_cast<uintptr_t>(tail_addr);
    sink_ret->aiv_icache_prefetch_cnt = std::min(prefetch_cnt, tail_prefetch_cnt);
    }
  return ge::SUCCESS;
}

UINT32 GetKernelFuncInfoByKernelType(const KernelFunctionCtx &ctx, const std::string &prefix, AICoreSinkRet *sink_ret) {
  MIX_KERNEL_REQ_TYPE type = MIX_KERNEL_REQ_TYPE(ctx.kernel_type);
  switch (type) {
    case MIX_KERNEL_REQ_TYPE::MIX_SINGLE_KERNEL:
    case MIX_KERNEL_REQ_TYPE::MIX_MULTI_KERNEL: {
      return GetMixDynamicPC(ctx, prefix, sink_ret);
    }
    case MIX_KERNEL_REQ_TYPE::ENHANCED_MIX_SINGLE_KERNEL:
    case MIX_KERNEL_REQ_TYPE::ENHANCED_MIX_DYNAMIC: {
      rtKernelDetailInfo_t kernel_info;
      GE_CHK_RT_RET(rtKernelGetAddrAndPrefCntV2(ctx.bin_handler, ctx.tiling_key, ctx.kernel_unique_ids_addr, ctx.flag,
                                                &kernel_info));
      GELOGD("Get mix node with flag: %u prefix:%s.", ctx.flag, prefix.c_str());
      if (!CheckKernelInfoValid(kernel_info)) {
        return ge::FAILED;
      }
      rtKernelDetailInfo_t tail_kernel_info;
      if (ctx.tail_tiling_key == ctx.tiling_key) {
        tail_kernel_info = kernel_info;
      } else {
        GE_CHK_RT_RET(rtKernelGetAddrAndPrefCntV2(ctx.bin_handler, ctx.tail_tiling_key, ctx.kernel_unique_ids_addr,
                                                  ctx.flag, &tail_kernel_info));
        if (!CheckKernelInfoValid(tail_kernel_info)) {
          return ge::FAILED;
        }
      }
      return ParseKernelInfo(kernel_info, tail_kernel_info, sink_ret);
    }
    default:
      break;
  }
  return ge::SUCCESS;
}
UINT32 GetMixAddrAndPrefCnt(KernelContext *context) {
  auto sink_ret = context->GetOutputPointer<AICoreSinkRet>(0U);
  if (sink_ret == nullptr) {
    GELOGE(ge::FAILED, "[kernel][SinkFFTSAICoreNodeBin] Sink ret is null.");
    return ge::GRAPH_FAILED;
  }
  auto kernel_num = context->GetInputValue<size_t>(static_cast<size_t>(MIXDyInKey::KERNEL_NUM));
  auto kernel_type = context->GetInputValue<size_t>(static_cast<size_t>(MIXDyInKey::KERNEL_TYPE));
  auto tiling_key = context->GetInputValue<size_t>(static_cast<size_t>(MIXDyInKey::TILING_KEY));
  auto tail_tiling_key = context->GetInputValue<size_t>(static_cast<size_t>(MIXDyInKey::TAIL_TILING_KEY));
  GELOGD("Get Mix pc with num %u, task_type %u, key:%u/%u", kernel_num, kernel_type, tiling_key, tail_tiling_key);
  size_t input_num = static_cast<size_t>(MIXDyInKey::PREFIX_START);
  for (size_t i = 0U; i < kernel_num; ++i) {
    auto prefix = context->GetInputValue<char *>(input_num++);
    auto bin_handle = context->GetInputValue<void *>(input_num++);
    if (bin_handle == nullptr) {
      GELOGE(ge::FAILED, "[kernel][GetMixL2AddrAndPrefCnt] No bin handle.");
      return ge::GRAPH_FAILED;
    }
    KernelFunctionCtx ctx(kernel_type, RT_DYNAMIC_SHAPE_KERNEL, tiling_key, tail_tiling_key, bin_handle, nullptr);
    if (GetKernelFuncInfoByKernelType(ctx, prefix, sink_ret) != ge::SUCCESS) {
      return ge::FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateMixSinkRet(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto chain = context->GetOutput(0);
  GE_ASSERT_NOTNULL(chain);
  chain->SetWithDefaultDeleter(new AICoreSinkRet());
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(GetMixAddrAndPrefCnt).RunFunc(GetMixAddrAndPrefCnt).OutputsCreator(CreateMixSinkRet);


UINT32 SinkMixStaticNodeBin(KernelContext *context) {
  auto sink_ret = context->GetOutputPointer<AICoreSinkRet>(0U);
  GE_ASSERT_NOTNULL(sink_ret);
  auto kernel_num = context->GetInputValue<size_t>(0);
  auto kernel_type = context->GetInputValue<size_t>(1U);
  size_t input_num = 2U;
  for (size_t i = 0U; i < kernel_num; ++i) {
    auto kernel_name = context->GetInputValue<char *>(input_num++);
    GE_ASSERT_NOTNULL(kernel_name);
    auto prefix = context->GetInputValue<char *>(input_num++);
    GE_ASSERT_NOTNULL(prefix);
    auto stub_name = context->GetInputValue<char *>(input_num++);
    GE_ASSERT_NOTNULL(stub_name);
    auto metadata = context->GetInputValue<char *>(input_num++);
    GE_ASSERT_NOTNULL(metadata);
    auto bin_data_holder = context->GetInputValue<kernel::BinData *>(input_num++);
    GE_ASSERT_NOTNULL(bin_data_holder);
    void *bin_handle;
    rtDevBinary_t *bin_data = reinterpret_cast<rtDevBinary_t *>(bin_data_holder);
    bin_data->data = &bin_data_holder->placeholder;
    GE_ASSERT_RT_OK(rtDevBinaryRegister(bin_data, &bin_handle));
    if (strlen(metadata) > 1U) {
      GE_ASSERT_RT_OK(rtMetadataRegister(bin_handle, metadata));
    }
    static std::list<uint8_t> kernel_unique_ids;
    static std::mutex mtx;
    void *kernel_unique_ids_addr;
    {
      std::unique_lock<std::mutex> lk(mtx);
      kernel_unique_ids.push_back(0U); // store unique id persistent for unique 'stub func' which means kernel id
      kernel_unique_ids_addr = &kernel_unique_ids.back();
    }
    GELOGD("register for kernel:%s prefix %s stub_name %s", kernel_name, prefix, stub_name);
    GE_ASSERT_RT_OK(rtFunctionRegister(bin_handle, kernel_unique_ids_addr, stub_name, kernel_name, 0U));
    KernelFunctionCtx ctx(kernel_type, RT_STATIC_SHAPE_KERNEL, 0U, 0U, bin_handle, kernel_unique_ids_addr);
    if (GetKernelFuncInfoByKernelType(ctx, prefix, sink_ret) != ge::SUCCESS) {
      return ge::FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(SinkMixStaticNodeBin).RunFunc(SinkMixStaticNodeBin).OutputsCreator(CreateMixSinkRet);
}  // namespace kernel
}  // namespace gert
