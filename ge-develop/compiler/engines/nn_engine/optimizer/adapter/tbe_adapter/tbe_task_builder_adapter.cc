/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter/tbe_adapter/tbe_task_builder_adapter.h"
#include "adapter/tbe_adapter/kernel_launch/l2_cache_kernel_launch.h"
#include "common/fe_log.h"
#include "common/platform_utils.h"
#include "common/configuration.h"
#include "common/fe_context_utils.h"
#include "common/graph_comm.h"
#include "common/l2_stream_info.h"
#include "common/lxfusion_json_util.h"
#include "common/op_tensor_utils.h"
#include "common/math_util.h"
#include "common/string_utils.h"
#include "common/fe_inner_attr_define.h"
#include "common/aicore_util_constants.h"
#include "common/fe_graph_common.h"
#include "common/fe_inner_error_codes.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/tuning_utils.h"
#include "rt_error_codes.h"
#include "securec.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "ops_kernel_builder/task_builder/args_format_constructor.h"

namespace fe {
namespace {
const uint32_t MAX_L2_DATANUM = 8;
// tiling data size to be reserved when generate task for unknown shape op
const int RESERVED_TILING_DATA_SIZE = 8;
const int RESERVED_GLOBALWORKSPACE_SIZE = 8;
const uint32_t DEFAULT_KERNEL_BLOCK_DIM = 1;
const int ONE_MEM_TYPE_SIZE = 1;
const std::unordered_set<std::string> DEV_BINARY_MAGIC_TYPE {
        "RT_DEV_BINARY_MAGIC_ELF",              "RT_DEV_BINARY_MAGIC_ELF_AIVEC",
        "RT_DEV_BINARY_MAGIC_ELF_AICUBE",       "FFTS_BINARY_MAGIC_ELF_MIX_AIC",
        "FFTS_BINARY_MAGIC_ELF_MIX_AIV",        "RT_DEV_BINARY_MAGIC_OM"};
const std::string kStrValidBinaryMagic = "RT_DEV_BINARY_MAGIC_(ELF/ELF_AIVEC/ELF_AICUBE)";
}

thread_local rtL2Ctrl_t g_tel2ctrl;

void TbeTaskBuilderAdapter::MemCpyForL2IdAndL2Addr(uint64_t &cur_ptr, uint32_t &l2_args_size, int64_t data_in_l2_id,
                                                   uint64_t data_in_l2_addr) const {
  if (l2_args_size < sizeof(int64_t)) {
    REPORT_FE_ERROR("[GenTask][Memcpy][Node %s type %s] l2_args_size (which is %u) is smaller than size of int64_t.",
                    op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), l2_args_size);
    return;
  }
  errno_t ret = memcpy_s(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(cur_ptr)),
                         l2_args_size, &data_in_l2_id, sizeof(int64_t));
  if (ret != EOK) {
    REPORT_FE_ERROR("[GenTask][Memcpy][Node %s type %s] Failed to memcpy data in l2 id, error num is %d.",
                    op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
    return;
  }
  l2_args_size = l2_args_size - sizeof(uint64_t);
  cur_ptr = cur_ptr + sizeof(uint64_t);

  if (l2_args_size < sizeof(int64_t)) {
    REPORT_FE_ERROR("[GenTask][Memcpy][Node %s type %s] l2_args_size (which is %u) is smaller than size of int64_t.",
                    op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), l2_args_size);
    return;
  }
  ret = memcpy_s(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(cur_ptr)),
                 l2_args_size, &data_in_l2_addr, sizeof(uint64_t));
  if (ret != EOK) {
    REPORT_FE_ERROR("[GenTask][Memcpy][Node %s type %s] Failed to memcpy data in l2 addr, error num is %d.",
                    op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
    return;
  }
  l2_args_size = l2_args_size - sizeof(uint64_t);
  cur_ptr = cur_ptr + sizeof(uint64_t);
}

void TbeTaskBuilderAdapter::DealInputOutputWithDdr(int32_t data_num, uint64_t &cur_ptr,
                                                   uint32_t &l2_args_size) const {
  for (int i = 0; i != data_num; ++i) {
    MemCpyForL2IdAndL2Addr(cur_ptr, l2_args_size, -1, 0);
  }
}

void TbeTaskBuilderAdapter::DealInputOutputL2DataMap(const L2DataMap &l2datamap, int32_t data_num,
                                                     const void *x[], const void *y[], uint64_t &cur_ptr,
                                                     uint32_t &l2_args_size,
                                                     bool is_input) const {
  for (int i = 0; i < data_num; ++i) {
    L2DataMap::const_iterator iter;
    for (iter = l2datamap.begin(); iter != l2datamap.end(); ++iter) {
      const L2Data &flowdata = iter->second;
      int64_t data_in_l2_id = flowdata.l2Index;
      auto ddr_key = iter->first;

      if (is_input && static_cast<uint64_t>(reinterpret_cast<uintptr_t>(x[i])) == ddr_key) {
        FE_LOGD("iter->first value is %ld, (uint64_t)(uintptr_t)x[%d] value is %ld", ddr_key, i,
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(x[i])));
        MemCpyForL2IdAndL2Addr(cur_ptr, l2_args_size, data_in_l2_id, flowdata.l2Addr);
        break;
      }

      if (!is_input && static_cast<uint64_t>(reinterpret_cast<uintptr_t>(y[i])) == ddr_key) {
        FE_LOGD("iter->first value is %ld, (uint64_t)(uintptr_t)y[%d] value is %ld", ddr_key, i,
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(y[i])));
        MemCpyForL2IdAndL2Addr(cur_ptr, l2_args_size, data_in_l2_id, flowdata.l2Addr);
        break;
      }
    }

    if (iter == l2datamap.end()) {
      string input_or_output = is_input ? "input" : "output";
      FE_LOGD("Can not find anything in l2datamap for the %s %d, set l2_index=-1 and l2_offset=0.",
              input_or_output.c_str(), i);
      MemCpyForL2IdAndL2Addr(cur_ptr, l2_args_size, -1, 0);
    }
  }
}

void TbeTaskBuilderAdapter::DealInputOutputL2DataMap(const L2FusionDataMap_t &l2datamap, int32_t data_num,
                                                     const void *x[], const void *y[],
                                                     uint64_t &cur_ptr, uint32_t &l2_args_size,
                                                     bool is_input) const {
  for (int i = 0; i < data_num; ++i) {
    L2FusionDataMap_t::const_iterator iter;
    for (iter = l2datamap.begin(); iter != l2datamap.end(); ++iter) {
      const L2FusionData_t &flowdata = iter->second;
      int64_t data_in_l2_id = flowdata.l2Index;
      auto ddr_key = iter->first;

      if (is_input && static_cast<uint64_t>(reinterpret_cast<uintptr_t>(x[i])) == ddr_key) {
        FE_LOGD("iter->first value is %ld, (uint64_t)(uintptr_t)x[%d] value is %ld", ddr_key, i,
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(x[i])));
        MemCpyForL2IdAndL2Addr(cur_ptr, l2_args_size, data_in_l2_id, flowdata.l2Addr);
        break;
      }

      if (!is_input && static_cast<uint64_t>(reinterpret_cast<uintptr_t>(y[i])) == ddr_key) {
        FE_LOGD("iter->first value is %ld, (uint64_t)(uintptr_t)y[%d] value is %ld", ddr_key, i,
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(y[i])));
        MemCpyForL2IdAndL2Addr(cur_ptr, l2_args_size, data_in_l2_id, flowdata.l2Addr);
        break;
      }
    }

    if (iter == l2datamap.end()) {
      string input_or_output = is_input ? "input" : "output";
      FE_LOGD("Can not find anything in l2datamap for the %s %d, set l2_index=-1 and l2_offset=0.",
              input_or_output.c_str(), i);
      MemCpyForL2IdAndL2Addr(cur_ptr, l2_args_size, -1, 0);
    }
  }
}

Status TbeTaskBuilderAdapter::SaveTeCoreL2FlowDataForL2Buffer(int32_t input_num, int32_t output_num, uint64_t cur_ptr,
                                                              const void *x[], const void *y[], rtL2Ctrl_t &tel2ctrl,
                                                              uint32_t l2_args_size, uint32_t workspace_num) {
  TaskL2Info *l2_data = nullptr;
  (void)memset_s(&tel2ctrl, sizeof(rtL2Ctrl_t), 0, sizeof(rtL2Ctrl_t));
  std::string batch_label = "Batch_-1";
  (void)ge::AttrUtils::GetStr(node_.GetOpDesc(), ge::ATTR_NAME_BATCH_LABEL, batch_label);
  int64_t stream_id = node_.GetOpDesc()->GetStreamId();
  Status ret = StreamL2Info::Instance().GetStreamL2Info(stream_id, node_.GetName(), l2_data, batch_label);
  if ((ret == SUCCESS) && (l2_data != nullptr)) {
    FE_LOGI("Node[type=%s,name=%s]: find the l2 data from stream_l2_map.", node_.GetType().c_str(),
            node_.GetName().c_str());
    L2DataMap input = l2_data->input;
    L2DataMap output = l2_data->output;
    DealInputOutputL2DataMap(input, input_num, x, y, cur_ptr, l2_args_size, true);
    DealInputOutputL2DataMap(output, output_num, x, y, cur_ptr, l2_args_size, false);
    DealInputOutputWithDdr(workspace_num, cur_ptr, l2_args_size);
    tel2ctrl = l2_data->l2ctrl;
    return SUCCESS;
  } else {  // Const/PlaceHolder/PlaceEnd/Data
    FE_LOGW("Node[type=%s,name=%s]: can not find the l2 data from stream_l2_map.", node_.GetType().c_str(),
            node_.GetName().c_str());
    return FAILED;
  }
}

Status TbeTaskBuilderAdapter::SaveTeCoreL2FlowDataForL2Fusion(int32_t input_num, int32_t output_num, uint64_t cur_ptr,
                                                              const void *x[], const void *y[], rtL2Ctrl_t &tel2ctrl,
                                                              uint32_t l2_args_size, uint32_t workspace_num) {
  (void)memset_s(&tel2ctrl, sizeof(rtL2Ctrl_t), 0, sizeof(rtL2Ctrl_t));

  ge::OpDescPtr node_desc = node_.GetOpDesc();
  L2FusionInfoPtr l2_info = GetL2FusionInfoFromJson(node_desc);
  if (l2_info == nullptr) {
    FE_LOGD("Node[type=%s,name=%s]: the l2_fusion_info is nullptr.", node_desc->GetType().c_str(),
            node_desc->GetName().c_str());
    return PARAM_INVALID;
  }

  L2FusionDataMap_t &input = l2_info->input;
  L2FusionDataMap_t &output = l2_info->output;
  DealInputOutputL2DataMap(input, input_num, x, y, cur_ptr, l2_args_size, true);
  DealInputOutputL2DataMap(output, output_num, x, y, cur_ptr, l2_args_size, false);
  DealInputOutputWithDdr(workspace_num, cur_ptr, l2_args_size);

  tel2ctrl = l2_info->l2_info.l2ctrl;
  FE_LOGD("Node[type=%s,name=%s]: SaveL2DataFlow find L2 Alloc and do L2fusion success.", node_desc->GetType().c_str(),
          node_desc->GetName().c_str());
  return SUCCESS;
}

void TbeTaskBuilderAdapter::DisplayRtL2CtrlInfo(const rtL2Ctrl_t &l2ctrl, bool enable_l2) const {
  FE_LOGD("L2ctrl.size = %lu.", l2ctrl.size);
  FE_LOGD("L2 %s.", enable_l2 ? "enable" : "disable");
  for (uint32_t i = 0; i < MAX_L2_DATANUM; i++) {
    if (l2ctrl.data[i].L2_mirror_addr != 0) {
      FE_LOGD("data_index = %u.", i);
      FE_LOGD("L2_data_section_size = %u.", l2ctrl.data[i].L2_data_section_size);
      FE_LOGD("L2_mirror_addr = 0x%lx.", l2ctrl.data[i].L2_mirror_addr);
      FE_LOGD("L2_page_offset_base = %u.", l2ctrl.data[i].L2_page_offset_base);
      FE_LOGD("prev_L2_page_offset_base = %d.", l2ctrl.data[i].prev_L2_page_offset_base);
      FE_LOGD("L2_preload = %u.", l2ctrl.data[i].L2_preload);
      FE_LOGD("L2_load_to_ddr = %u.", l2ctrl.data[i].L2_load_to_ddr);
      FE_LOGD("modified = %u.", l2ctrl.data[i].modified);
      FE_LOGD("priority = %u.", l2ctrl.data[i].priority);
    }
  }
}

Status TbeTaskBuilderAdapter::CheckArrayValue(const void *array[], int32_t array_size,
                                              int32_t num, const string& name) const {
  if (array == nullptr) {
    FE_LOGD("%s is nullptr! Please check.", name.c_str());
  } else {
    int32_t check_size = num;
    if (array_size < num) {
      check_size = array_size;
      FE_LOGD("[GenTask][TbeFwd][Check][Node %s type %s] The %s array_size[%d] < num[%d].",
              op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), name.c_str(), array_size, num);
    }
    for (int i = 0; i < check_size; i++) {
      if (array[i] == nullptr) {
        FE_LOGD("[GenTask][TbeFwd][Check][Node %s type %s] The %s[%d] now is nullptr.",
                op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), name.c_str(), i);
      }
    }
  }
  return SUCCESS;
}

Status TbeTaskBuilderAdapter::CheckForForward(const void *args, const void *x[], int32_t x_array_size,
                                              const void *y[], int32_t input_num, int32_t output_num) const {
  FE_CHECK_NOTNULL(args);
  string x_name = "x";
  Status result = CheckArrayValue(x, x_array_size, input_num, x_name);
  if (result != SUCCESS) {
    return result;
  }

  string y_name = "y";
  result = CheckArrayValue(y, output_num, output_num, y_name);
  if (result != SUCCESS) {
    return result;
  }

  if (CheckUint32AddOverflow(static_cast<uint32_t>(input_num), static_cast<uint32_t>(output_num)) != SUCCESS) {
    REPORT_FE_ERROR("[GenTask][TbeFwd][Check] Unsigned Integer %u and %u addition can result in overflow!",
                    static_cast<uint32_t>(input_num), static_cast<uint32_t>(output_num));
    return TASK_BUILDER_STATUS_BAD_PARAM;
  }
  if (CheckUint32MulOverflow((static_cast<uint32_t>(input_num) + static_cast<uint32_t>(output_num)),
                             (static_cast<uint32_t>(sizeof(int64_t) + sizeof(uint64_t)))) != SUCCESS) {
    REPORT_FE_ERROR("[GenTask][TbeFwd][Check] Unsigned Integer %u and %u multiplication can result in overflow!",
                    (static_cast<uint32_t>(input_num) + static_cast<uint32_t>(output_num)),
                    (static_cast<uint32_t>(sizeof(int64_t) + sizeof(uint64_t))));
    return TASK_BUILDER_STATUS_BAD_PARAM;
  }
  return SUCCESS;
}

Status TbeTaskBuilderAdapter::DealKernelLaunchForL2Buffer(int32_t input_num, int32_t output_num, uint64_t cur_ptr,
                                                          const void *x[], const void *y[],  rtL2Ctrl_t &tel2ctrl,
                                                          uint32_t args_size, uint32_t l2_args_size,
                                                          const std::string &stub_func, const uint32_t core_dim,
                                                          const void *tmp_buf, int32_t workspace_num,
                                                          domi::TaskDef &task_def) {
  bool kernelRet = false;
  std::string first_kernel_name;
  Status ret = SaveTeCoreL2FlowDataForL2Buffer(input_num, output_num, cur_ptr, x, y, tel2ctrl, l2_args_size,
                                               workspace_num);
  if (ret == SUCCESS) {
    FE_LOGD("Node[type=%s,name=%s]: L2 alloc infomation get success, core_dim=%u, args_size=%u, l2_args_size=%u.",
            node_.GetTypePtr(), node_.GetNamePtr(), core_dim, args_size, l2_args_size);

    for (uint64_t idx = 0; idx < ((args_size) / sizeof(uint64_t)); ++idx) {
      uint64_t current_address =
          *(reinterpret_cast<uint64_t *>(reinterpret_cast<uintptr_t>(tmp_buf) + idx * sizeof(uint64_t)));
      FE_LOGD("Node[type=%s,name=%s]: do distribute index[%lu], args=[%lu] is ddr address value.", node_.GetTypePtr(),
              node_.GetNamePtr(), idx, current_address);
    }
    uint32_t index_of_offset_base = 2;
    for (uint64_t idx = 0; idx < (l2_args_size / (sizeof(uint64_t))); ++idx) {
      if (idx % index_of_offset_base == 0) {
        uint64_t current_address =
            *(reinterpret_cast<uint64_t *>(reinterpret_cast<uintptr_t>(tmp_buf) + args_size + idx * sizeof(int64_t)));
        FE_LOGD("Node[type=%s,name=%s]: do distribute index[%lu], args=[%lu] is l2 index value.", node_.GetTypePtr(),
                node_.GetNamePtr(), idx, current_address);
      } else {
        uint64_t current_address =
            *(reinterpret_cast<uint64_t *>(reinterpret_cast<uintptr_t>(tmp_buf) + args_size + idx * sizeof(uint64_t)));
        FE_LOGD("Node[type=%s,name=%s]: do distribute index[%lu], args=[%lu] is l2 offset value.", node_.GetTypePtr(),
                node_.GetNamePtr(), idx, current_address);
      }
    }
    if (ge::AttrUtils::GetStr(node_.GetOpDesc(), ATTR_NAME_KERNEL_LIST_FIRST_NAME, first_kernel_name)) {
      kernelRet = TbeKernelLaunch::KernelLaunchWithHandle(core_dim, tmp_buf, args_size + l2_args_size, &tel2ctrl,
                                                          task_def);
    } else {
      kernelRet = TbeKernelLaunch::KernelLaunch(stub_func, core_dim, tmp_buf, args_size + l2_args_size, &tel2ctrl,
                                                task_def);
    }
  } else {
    FE_LOGD("Node[type=%s,name=%s]: can not find L2 alloc infomation and use the ddr address.", node_.GetTypePtr(),
            node_.GetNamePtr());
    if (ge::AttrUtils::GetStr(node_.GetOpDesc(), ATTR_NAME_KERNEL_LIST_FIRST_NAME, first_kernel_name)) {
      kernelRet = TbeKernelLaunch::KernelLaunchWithHandle(core_dim, tmp_buf, args_size, nullptr, task_def);
    } else {
      kernelRet = TbeKernelLaunch::KernelLaunch(stub_func, core_dim, tmp_buf, args_size, nullptr, task_def);
    }
  }
  if (!kernelRet) {
    return TASK_BUILDER_STATUS_RUNTIME_ERROR;
  }
  DisplayRtL2CtrlInfo(tel2ctrl, fe::GetFunctionState(fe::FuncParamType::FUSION_L2));
  return SUCCESS;
}

Status TbeTaskBuilderAdapter::DealKernelLaunchForL2Fusion(int32_t input_num, int32_t output_num,
                                                          uint64_t cur_ptr, const void *x[],
                                                          const void *y[], rtL2Ctrl_t &tel2ctrl, uint32_t args_size,
                                                          uint32_t l2_args_size, const std::string &stub_func,
                                                          const uint32_t core_dim, const void *tmp_buf,
                                                          int32_t workspace_num, domi::TaskDef &task_def) {
  auto op_name = node_.GetName();
  auto op_type = node_.GetType();
  bool kernelRet = false;
  Status ret = SaveTeCoreL2FlowDataForL2Fusion(input_num, output_num, cur_ptr, x, y, tel2ctrl,
                                               l2_args_size, workspace_num);
  string first_kernel_name;
  if (ret == SUCCESS) {
    FE_LOGD("Node[type=%s,name=%s]: L2 alloc infomation get success, core_dim=%u, args_size=%u, l2_args_size=%u.",
            op_type.c_str(), op_name.c_str(), core_dim, args_size, l2_args_size);

    for (uint64_t idx = 0; idx < ((args_size) / sizeof(uint64_t)); ++idx) {
      uint64_t address = *(reinterpret_cast<uint64_t *>(reinterpret_cast<uintptr_t>(tmp_buf) + idx * sizeof(uint64_t)));
      FE_LOGD("Node[type=%s,name=%s]: do distribute index[%lu], args[%lu] is ddr address value.", op_type.c_str(),
              op_name.c_str(), idx, address);
    }
    uint32_t index_offset_base = 2;
    for (uint64_t idx = 0; idx < (l2_args_size / (sizeof(uint64_t))); ++idx) {
      if (idx % index_offset_base == 0) {
        uint64_t address =
            *(reinterpret_cast<uint64_t *>(reinterpret_cast<uintptr_t>(tmp_buf) + args_size + idx * sizeof(int64_t)));
        FE_LOGD("Node[type=%s,name=%s]: do distribute index[%lu], args=[%lu] is l2 index value.", op_type.c_str(),
                op_name.c_str(), idx, address);
      } else {
        uint64_t address =
            *(reinterpret_cast<uint64_t *>(reinterpret_cast<uintptr_t>(tmp_buf) + args_size + idx * sizeof(uint64_t)));
        FE_LOGD("Node[type=%s,name=%s]: do distribute index[%lu], args=[%lu] is l2 offset value.", op_type.c_str(),
                op_name.c_str(), idx, address);
      }
    }
    if (ge::AttrUtils::GetStr(node_.GetOpDesc(), ATTR_NAME_KERNEL_LIST_FIRST_NAME, first_kernel_name)) {
      kernelRet = TbeKernelLaunch::KernelLaunchWithHandle(core_dim, tmp_buf, args_size + l2_args_size,
                                                          &tel2ctrl, task_def);
    } else {
      kernelRet = TbeKernelLaunch::KernelLaunch(stub_func, core_dim, tmp_buf, args_size + l2_args_size, &tel2ctrl,
                                                task_def);
  }
  } else {
    FE_LOGD("Node[type=%s,name=%s]: can not find L2 Alloc infomation and use the ddr address(l2_index=-1,l2_offset=0).",
            op_type.c_str(), op_name.c_str());
    DealInputOutputWithDdr(input_num, cur_ptr, l2_args_size);
    DealInputOutputWithDdr(output_num, cur_ptr, l2_args_size);
    DealInputOutputWithDdr(workspace_num, cur_ptr, l2_args_size);
    if (ge::AttrUtils::GetStr(node_.GetOpDesc(), ATTR_NAME_KERNEL_LIST_FIRST_NAME, first_kernel_name)) {
      kernelRet = TbeKernelLaunch::KernelLaunchWithHandle(core_dim, tmp_buf, args_size + l2_args_size, nullptr,
                                                          task_def);
    } else {
      kernelRet = TbeKernelLaunch::KernelLaunch(stub_func, core_dim, tmp_buf, args_size + l2_args_size, nullptr,
                                                task_def);
    }
  }
  if (!kernelRet) {
    return TASK_BUILDER_STATUS_RUNTIME_ERROR;
  }
  DisplayRtL2CtrlInfo(tel2ctrl, true);
  return SUCCESS;
}

Status TbeTaskBuilderAdapter::TbeForward(const uint32_t core_dim, const void *args, uint32_t args_size,
                                         int32_t input_num, const void *x[], int32_t x_array_size,
                                         int32_t output_num,const void *y[], int32_t workspace_num,
                                         domi::TaskDef &task_def) {
  Status ret = CheckForForward(args, x, x_array_size, const_cast<const void **>(y), input_num, output_num);
  if (ret != SUCCESS) {
    return ret;
  }

  std::string stub_func;
  if (!ge::AttrUtils::GetStr(op_desc_, ATTR_NAME_KERNEL_LIST_FIRST_NAME, stub_func)) {
    stub_func = GetUniqueGraphIdForNode();
  }
  FE_LOGD("Generate stub func string[%s] of node[%s, %s].",
          stub_func.c_str(), op_desc_->GetNamePtr(), op_desc_->GetTypePtr());

  // l2 buffer or l2 fusion
  bool is_l2_buffer = fe::GetFunctionState(fe::FuncParamType::FUSION_L2);
  bool lx_fusion_pass = false;
  (void)ge::AttrUtils::GetBool(op_desc_, ATTR_NAME_LX_FUSION_PASS, lx_fusion_pass);
  bool is_l2_fusion =
      (Configuration::Instance(AI_CORE_NAME).EnableL2Fusion() && lx_fusion_pass) ||
      (FEContextUtils::GetBuildMode() == ge::BUILD_MODE_TUNING);
  if (is_l2_fusion || is_l2_buffer) {
    if (CheckUint32AddOverflow(static_cast<uint32_t>(input_num), static_cast<uint32_t>(output_num)) != SUCCESS) {
      REPORT_FE_ERROR(
          "[GenTask][TbeFwd][L2Check][Node %s type %s] Unsigned Integer %u and %u addition can result in overflow!",
          op_desc_->GetName().c_str(), op_desc_->GetType().c_str(),
          static_cast<uint32_t>(input_num), static_cast<uint32_t>(output_num));
      return TASK_BUILDER_STATUS_BAD_PARAM;
    }
    if (CheckUint32AddOverflow(static_cast<uint32_t>(input_num + output_num),
                               static_cast<uint32_t>(workspace_num)) != SUCCESS) {
      REPORT_FE_ERROR(
          "[GenTask][TbeFwd][L2Check][Node %s type %s] Unsigned Integer %u and %u addition can result in overflow!",
          op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), static_cast<uint32_t>(input_num + output_num),
          static_cast<uint32_t>(workspace_num));
      return TASK_BUILDER_STATUS_BAD_PARAM;
    }
    uint32_t l2_args_size =
        static_cast<uint32_t>(input_num + output_num + workspace_num) * (sizeof(int64_t) + sizeof(uint64_t));
    std::vector<uint8_t> tmp_buf(args_size + l2_args_size, 0);
    if (memcpy_s(tmp_buf.data(), args_size, args, args_size) != EOK) {
      FE_LOGE("[GenTask][TbeForward] Copy args data failed.");
      return TASK_BUILDER_STATUS_INTERNAL_ERROR;
    }

    uint64_t cur_ptr = ge::PtrToValue(tmp_buf.data()) + args_size;
    if (is_l2_buffer) {
      ret = DealKernelLaunchForL2Buffer(input_num, output_num, cur_ptr, x, y, g_tel2ctrl, args_size,
                                        l2_args_size, stub_func, core_dim, tmp_buf.data(), workspace_num, task_def);
    } else {
      ret = DealKernelLaunchForL2Fusion(input_num, output_num, cur_ptr, x, y, g_tel2ctrl, args_size,
                                        l2_args_size, stub_func, core_dim, tmp_buf.data(), workspace_num, task_def);
    }
    return ret;
  }

  if (PlatformUtils::Instance().IsEnableL2CacheRc()) {
    L2CacheKernelLaunch l2_cache_kernel_launch(input_num);
    return l2_cache_kernel_launch.DealKernelLaunch(node_, args, args_size, stub_func, core_dim, task_def);
  } else {
    TbeKernelLaunch tbe_kernel_launch(input_num);
    return tbe_kernel_launch.DealKernelLaunch(node_, args, args_size, stub_func, core_dim, task_def);
  }
}

Status TbeTaskBuilderAdapter::CheckTensorSize(const ge::GeTensorDesc &tensor_desc,
                                              uint32_t i, bool is_input, int32_t output_real_calc_flag) const {
  auto op_type = op_desc_->GetType();
  auto op_name = op_desc_->GetName();
  int64_t tensor_size = 0;
  if (OpTensorUtils::CalcTensorSize(tensor_desc, output_real_calc_flag, tensor_size) != SUCCESS) {
    REPORT_FE_ERROR("[GenTask][CheckTensorSize][Node %s type %s]:op output[%u] tensor size failed to calculate.",
                    op_name.c_str(), op_type.c_str(), i);
    return FAILED;
  }
  int64_t size_output = 0;
  if (ge::TensorUtils::GetSize(tensor_desc, size_output) != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR("[GenTask][CheckTensorSize][Node %s type %s]:Get size input[%u] tensor failed!",
                    op_name.c_str(), op_type.c_str(), i);
    return fe::FAILED;
  }
  string input_or_output = is_input ? "Input" : "Output";
  // compare the two size
  if (size_output < tensor_size) {
    std::vector<int64_t> shape_dims = tensor_desc.GetShape().GetDims();
    REPORT_FE_ERROR("[GenTask][CheckTensorSize] Node[%s, %s]: %s shape is [%s], which size %ld is not equal to %ld",
                    op_name.c_str(), op_type.c_str(), input_or_output.c_str(),
                    StringUtils::IntegerVecToString(shape_dims).c_str(), tensor_size, size_output);
    return fe::FAILED;
  }
  return SUCCESS;
}

Status TbeTaskBuilderAdapter::CheckInputAndOutputSize() {
  // Get input size
  int32_t output_real_calc_flag = 0;
  for (size_t i = 0; i < op_desc_->GetAllInputsSize(); i++) {
    ge::GeTensorDescPtr tensorDescPtr = op_desc_->MutableInputDesc(i);
    if (tensorDescPtr == nullptr) {
      continue;
    }
    ge::GeTensorDesc tensor_desc = op_desc_->GetInputDesc(i);
    if (CheckTensorSize(tensor_desc, i, true, output_real_calc_flag) != SUCCESS) {
      return FAILED;
    }
  }
  bool ret = ge::AttrUtils::GetInt(op_desc_, ge::ATTR_NAME_GET_TENSOR_ACTUAL_SIZE, output_real_calc_flag);
  FE_LOGD("Output_real_calc_flag: [%d], ret: [%d].", output_real_calc_flag, ret);
  for (size_t i = 0; i < op_desc_->GetAllOutputsDescSize(); i++) {
    ge::GeTensorDescPtr tensorDescPtr = op_desc_->MutableOutputDesc(i);
    if (tensorDescPtr == nullptr) {
      continue;
    }
    ge::GeTensorDesc tensor_desc = op_desc_->GetOutputDesc(i);
    if (CheckTensorSize(tensor_desc, i, false, output_real_calc_flag) != SUCCESS) {
      return FAILED;
    }
  }
  return SUCCESS;
}

TbeTaskBuilderAdapter::TbeTaskBuilderAdapter(const ge::Node &node, TaskBuilderContext &context)
    : TaskBuilderAdapter(node, context),
      block_dim_(DEFAULT_KERNEL_BLOCK_DIM) {}

TbeTaskBuilderAdapter::~TbeTaskBuilderAdapter() {}

uint64_t TbeTaskBuilderAdapter::GetAtomicStubFuncId() const {
  static std::atomic<uint64_t> global_cmo_id(1);
  return global_cmo_id.fetch_add(1, std::memory_order_relaxed);
}

std::string TbeTaskBuilderAdapter::GetUniqueGraphIdForNode() const {
  string session_graph_id = "";
  string atomic_id = std::to_string(GetAtomicStubFuncId());
  if (ge::AttrUtils::GetStr(op_desc_, ge::ATTR_NAME_SESSION_GRAPH_ID, session_graph_id) && !session_graph_id.empty()) {
    return atomic_id + "_" + session_graph_id + "_" + op_desc_->GetName();
  } else {
    return atomic_id + "_" + op_desc_->GetName();
  }
}

Status TbeTaskBuilderAdapter::Init() {
  auto op_type = op_desc_->GetType();
  auto op_name = op_desc_->GetName();
  FE_LOGD("Init begin, node name:%s, node type:%s.", node_.GetName().c_str(), node_.GetType().c_str());

  // Common initialization
  Status status = TaskBuilderAdapter::Init();
  if (status != SUCCESS) {
    REPORT_FE_ERROR("[GenTask][Init][Node %s type %s]:TaskBuilderAdapter::Init failed.",
                    op_name.c_str(), op_type.c_str());
    return status;
  }

  // Get block dim
  int32_t block_dim = -1;
  if (ge::AttrUtils::GetInt(op_desc_, ge::TVM_ATTR_NAME_BLOCKDIM, block_dim)) {
    block_dim_ = static_cast<uint32_t>(block_dim);
  }
  FE_LOGD("Tbe blockdim: %u.", block_dim_);

  // Get magic
  string bin_magic;
  (void)ge::AttrUtils::GetStr(op_desc_, ge::TVM_ATTR_NAME_MAGIC, bin_magic);
  if (DEV_BINARY_MAGIC_TYPE.find(bin_magic) == DEV_BINARY_MAGIC_TYPE.end()) {
    REPORT_FE_ERROR("[GenTask][Init] Node[%s, %s]: binary magic %s is unsupported. Only support %s.",
                    op_name.c_str(), op_type.c_str(), bin_magic.c_str(), kStrValidBinaryMagic.c_str());
    return PARAM_INVALID;
  }

  // Check Size
  if (!UnknownShapeUtils::IsUnknownShapeOp(*op_desc_)) {
    Status ret = CheckInputAndOutputSize();
    if (ret != SUCCESS) {
      return fe::FAILED;
    }
  }

  return SUCCESS;
}

void TbeTaskBuilderAdapter::SetInputAddrFromDataBase(const size_t input_index, const int64_t &input_offset) {
  int64_t tensor_memtype = RT_MEMORY_HBM;
  if (ge::AttrUtils::GetInt(op_desc_->GetInputDescPtr(input_index), ge::ATTR_NAME_TENSOR_MEM_TYPE, tensor_memtype)) {
    FE_LOGD("The value of input_index is %zu, and the value of input_offset is %ld", input_index, input_offset);
  }

  auto it = context_.mem_type_to_data_mem_base.find(tensor_memtype);
  if (it != context_.mem_type_to_data_mem_base.end()) {
    input_addrs_.push_back(it->second + input_offset);
  } else {
    input_addrs_.push_back(context_.dataMemBase + input_offset);
    FE_LOGD("set input_addrs_ from the plus of context_.dataMemBase and input_offset");
  }
}

Status TbeTaskBuilderAdapter::HandleAnchorWeight(const size_t &anchor_index) {
  auto op_type = op_desc_->GetType();
  auto op_name = op_desc_->GetName();
  auto in_desc_ptr = op_desc_->MutableInputDesc(static_cast<uint32_t>(anchor_index));

  int64_t weight_offset = 0;
  if(in_desc_ptr == nullptr) {
    return FAILED;
  }
  if (ge::TensorUtils::GetDataOffset(*in_desc_ptr, weight_offset) != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR("[GenTask][HandleAnchor][Node %s type %s]: Get weight offset failed.",
                    op_name.c_str(), op_type.c_str());
    return FAILED;
  }

  // max value of weight offset : 30 * 1024 * 1024 * 1024L.
  if (weight_offset < kMaxWeightOffset) {
    input_addrs_.push_back(context_.weightMemBase + weight_offset);
  } else {
    FE_LOGI("Print Input Addr is offset of op name: %s, op type: %s, weight offset is too big.",
            op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
    input_addrs_.push_back(reinterpret_cast<void *>(reinterpret_cast<intptr_t>(weight_offset)));
  }

  FE_LOGI("Name: %s, index: %zu, weightOffset: %lu.", op_desc_->GetName().c_str(), anchor_index, weight_offset);
  return SUCCESS;
}

Status TbeTaskBuilderAdapter::FeedInputAddrByAnchor(const ge::InDataAnchorPtr &anchor,
    InputIndexOffsetInfo &index_offset_info, bool is_gen_place_holder) {
  auto op_type = op_desc_->GetType();
  auto op_name = op_desc_->GetName();
  if (ge::AnchorUtils::GetStatus(anchor) == ge::ANCHOR_SUSPEND || anchor->GetPeerOutAnchor() == nullptr) {
    FE_LOGD("Node[type=%s,name=%s]:anchor %zu is suspend or peer anchor is null with gen flag:%d.",
            op_type.c_str(), op_name.c_str(), index_offset_info.anchor_index, is_gen_place_holder);
    if (is_gen_place_holder) {
      input_addrs_.push_back(reinterpret_cast<void*>(kTaskPlaceHolderAddr));
    }
    index_offset_info.anchor_index++;
    return SUCCESS;
  }

  if (ge::AnchorUtils::GetStatus(anchor) != ge::ANCHOR_DATA) {
    Status ret = HandleAnchorWeight(static_cast<size_t>(anchor->GetIdx()));
    if (ret != SUCCESS) {
      return ret;
    }
    index_offset_info.input_index++;
    index_offset_info.anchor_index++;
    return SUCCESS;
  }

  if (index_offset_info.input_index >= index_offset_info.input_offsets.size()) {
    REPORT_FE_ERROR(
        "[GenTask][InitInput] [Node %s type %s]: inputIndex must be less than size of offset, index[%zu], size[%zu].",
        op_name.c_str(), op_type.c_str(), index_offset_info.input_index, index_offset_info.input_offsets.size());
    return FAILED;
  }

  int64_t input_offset = index_offset_info.input_offsets[index_offset_info.input_index];
  FE_LOGD("Node[type=%s,name=%s]: input_index=%zu, input_offset=%ld.", op_type.c_str(), op_name.c_str(),
          index_offset_info.input_index, input_offset);
  bool is_addr_var = (index_offset_info.input_index < index_offset_info.input_is_addr_var.size()) &&
                     (index_offset_info.input_is_addr_var[index_offset_info.input_index]);
  if (is_addr_var) {
    input_addrs_.push_back(reinterpret_cast<void *>(reinterpret_cast<intptr_t>(input_offset)));
  } else {
    SetInputAddrFromDataBase(index_offset_info.input_index, input_offset);
  }

  FE_LOGD("Node[type=%s,name=%s]: Init input.", op_type.c_str(), op_name.c_str());
  index_offset_info.input_index++;
  index_offset_info.anchor_index++;
  return SUCCESS;
}

Status TbeTaskBuilderAdapter::GetInputIndexOffsetInfos(InputIndexOffsetInfo &index_offset_info) {
  auto op_type = op_desc_->GetType();
  auto op_name = op_desc_->GetName();
  vector<bool> input_is_addr_var;
  (void)ge::AttrUtils::GetListBool(op_desc_, ATTR_NAME_INPUT_IS_VAR, input_is_addr_var);

  // if input_offsets is empty, set 0 to vector
  vector<int64_t> input_offsets = op_desc_->GetInputOffset();
  if (input_offsets.empty()) {
    vector<int64_t> input_offset_zero(op_desc_->GetInputsSize(), 0);
    input_offsets.swap(input_offset_zero);
    FE_LOGD("Node[type=%s,name=%s]: input_offset_size:%zu.", op_type.c_str(), op_name.c_str(), input_offsets.size());
  }

  index_offset_info.input_offsets = input_offsets;
  index_offset_info.input_is_addr_var = input_is_addr_var;

  vector<uint32_t> input_type_list;
  (void)ge::AttrUtils::GetListInt(op_desc_, kInputParaTypeList, input_type_list);
  if (input_type_list.empty()) {
    FE_LOGW("Node[type=%s,name=%s] get attr input param type list is null.", op_type.c_str(), op_name.c_str());
    return PARAM_INVALID;
  }
  index_offset_info.input_type_list = input_type_list;
  return SUCCESS;
}

Status TbeTaskBuilderAdapter::InitInputNoPlaceholder() {
  auto op_type = op_desc_->GetType();
  auto op_name = op_desc_->GetName();

  InputIndexOffsetInfo index_offset_info;
  if (GetInputIndexOffsetInfos(index_offset_info) == FAILED) {
    REPORT_FE_ERROR("[GenTask][InitInput]Node[type=%s,name=%s]: Init input failed.", op_type.c_str(), op_name.c_str());
    return FAILED;
  }
  size_t tmp_input_idx = 0;
  for (auto const &anchor : node_.GetAllInDataAnchors()) {
    tmp_input_idx = index_offset_info.input_index;
    if (FeedInputAddrByAnchor(anchor, index_offset_info) != SUCCESS) {
      return FAILED;
    }
    if (tmp_input_idx < index_offset_info.input_index) {
      FE_LOGD("Node[type=%s,name=%s]: Add argsInfo for req input.", op_type.c_str(), op_name.c_str());
      FeedArgsInfo(domi::ArgsInfo_ArgsType_INPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, tmp_input_idx);
    }
  }
  FE_LOGD("Node[type=%s,name=%s]: Init input finished.", op_type.c_str(), op_name.c_str());
  return SUCCESS;
}

void TbeTaskBuilderAdapter::InsertMissOptAddr(size_t &arg_idx, std::vector<uint32_t> &insert_pos_vec) {
  for (auto insert_pos : insert_pos_vec) {
    if (arg_idx == insert_pos) {
      FE_LOGD("Insert optional input in pos %u.", insert_pos);
      input_addrs_.push_back(reinterpret_cast<void*>(kTaskPlaceHolderAddr));
      FeedArgsInfo(domi::ArgsInfo_ArgsType_INPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0xFFFFFFFFU);
      arg_idx++;
    }
  }
  return;
}

Status TbeTaskBuilderAdapter::InitInputGenPlaceholder() {
  auto op_type = op_desc_->GetType();
  auto op_name = op_desc_->GetName();

  InputIndexOffsetInfo index_offset_info;
  if (GetInputIndexOffsetInfos(index_offset_info) == FAILED) {
    REPORT_FE_ERROR("[GenTask][InitInput]Node[type=%s,name=%s]: Init input failed.", op_type.c_str(), op_name.c_str());
    return FAILED;
  }
  std::vector<uint32_t> insert_pos_vec;
  (void)ge::AttrUtils::GetListInt(op_desc_, kInputInsertOptPosList, insert_pos_vec);
  size_t tmp_input_idx = 0;
  size_t tmp_anchor_idx = 0;
  size_t arg_idx = 0;
  for (auto const &anchor : node_.GetAllInDataAnchors()) {
    InsertMissOptAddr(arg_idx, insert_pos_vec);
    tmp_input_idx = index_offset_info.input_index;
    tmp_anchor_idx = index_offset_info.anchor_index;
    if (FeedInputAddrByAnchor(anchor, index_offset_info, true) != SUCCESS) {
      return FAILED;
    }
    if (tmp_input_idx < index_offset_info.input_index) {
      FE_LOGD("Node[type=%s,name=%s]: Add argsInfo for req input.", op_type.c_str(), op_name.c_str());
      FeedArgsInfo(domi::ArgsInfo_ArgsType_INPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, tmp_input_idx);
    } else if (tmp_anchor_idx < index_offset_info.anchor_index && tmp_input_idx == index_offset_info.input_index) {
      FE_LOGD("Node[type=%s,name=%s]: Add argsInfo for null-opt-input.", op_type.c_str(), op_name.c_str());
      FeedArgsInfo(domi::ArgsInfo_ArgsType_INPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0xFFFFFFFFU);
    }
    arg_idx++;
  }
  InsertMissOptAddr(arg_idx, insert_pos_vec);
  FE_LOGD("Node[type=%s,name=%s]: Init input with addrs size[%zu] finished.", op_type.c_str(), op_name.c_str(),
          input_addrs_.size());
  size_t ops_in_size = 0;
  (void)ge::AttrUtils::GetInt(op_desc_, kOpKernelAllInputSize, ops_in_size);
  size_t dyn_add_num = 0;
  (void)ge::AttrUtils::GetInt(op_desc_, kDyInputsAddNum, dyn_add_num);
  size_t all_kernel_size = ops_in_size + dyn_add_num;
  if (all_kernel_size != arg_idx) {
    REPORT_FE_ERROR("[GenTask][InitInput]Node[type=%s,name=%s]:Expect In addr size:%zu with real:%zu.",
                    op_type.c_str(), op_name.c_str(), all_kernel_size, arg_idx);
    return FAILED;
  }
  return SUCCESS;
}

Status TbeTaskBuilderAdapter::InitInput() {
  const auto &op_type = op_desc_->GetType();
  const auto &op_name = op_desc_->GetName();
  std::string core_type;
  (void)ge::AttrUtils::GetStr(op_desc_, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type);
  bool need_sync = (core_type == kCoreTypeMixEnhance) &&
                   (PlatformUtils::Instance().GetFftsMode() == FFTS_MODE_FFTS_PLUS);
  need_sync |= op_desc_->HasAttr(ATTR_NAME_ALIAS_ENGINE_NAME);
  if (need_sync) {
    FE_LOGD("Node[type=%s,name=%s]:Mix add sync addr.", op_type.c_str(), op_name.c_str());
    input_addrs_.push_back(reinterpret_cast<void *>(0));
    FeedArgsInfo(domi::ArgsInfo_ArgsType_INPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0xFFFFU);
  }
  std::string opt_mode;
  (void)ge::AttrUtils::GetStr(op_desc_, "optionalInputMode", opt_mode);
  Status ret;
  if (opt_mode == "gen_placeholder") {
    ret = InitInputGenPlaceholder();
  } else {
    ret = InitInputNoPlaceholder();
  }

  FE_LOGD("Node[type=%s,name=%s]: Init input finished.", op_type.c_str(), op_name.c_str());
  return ret;
}

bool TbeTaskBuilderAdapter::GetUnknownShapeFlag() {
  bool is_support_unknown_shape = false;
  (void)ge::AttrUtils::GetBool(op_desc_, ATTR_NAME_SUPPORT_DYNAMIC_SHAPE, is_support_unknown_shape);
  bool is_unknown_shape = UnknownShapeUtils::IsUnknownShapeOp(*node_.GetOpDesc());
  FE_LOGD("Node[type=%s,name=%s]: is_unknown_shape flag is %ld.", op_desc_->GetTypePtr(),
          op_desc_->GetNamePtr(), is_unknown_shape);
  bool unknown_shape_flag = (is_support_unknown_shape && is_unknown_shape);
  return unknown_shape_flag;
}

void TbeTaskBuilderAdapter::AppendArgsTilingData(vector<void *> &device_addrs) {
  const auto &op_type = op_desc_->GetType();
  const auto &op_name = op_desc_->GetName();

  bool is_unknown_shape = GetUnknownShapeFlag();
  bool stc_dyn_soft_sync = false;
  (void)ge::AttrUtils::GetBool(op_desc_, kStaticToDynamicSoftSyncOp, stc_dyn_soft_sync);
  bool is_unknown_graph = FeGraphCommon::IsNodeOfUnknownRootGraph(node_);
  bool has_compile_info = op_desc_->HasAttr(COMPILE_INFO_JSON);
  bool is_static_reuse = !is_unknown_shape && !is_unknown_graph && has_compile_info;
  if (is_unknown_shape || stc_dyn_soft_sync || is_static_reuse) {
    FE_LOGD("Node[type=%s,name=%s] add args tiling data", op_type.c_str(), op_name.c_str());
    int64_t temp_op_para_size = 0;
    (void)ge::AttrUtils::GetInt(op_desc_, OP_PARA_SIZE, temp_op_para_size);
    if (temp_op_para_size > 0) {
      char tiling_data_ptr[RESERVED_TILING_DATA_SIZE] = {0x00};
      device_addrs.push_back(tiling_data_ptr);
    }
  }
}

void TbeTaskBuilderAdapter::AppendGlobalData(vector<void *> &device_addrs) {
  int64_t globalworkspace_size = 0;
  int64_t globalworkspace_type = 0;
  if (ge::AttrUtils::GetInt(op_desc_, kGlobalworkspaceSize, globalworkspace_size)) {
    (void)ge::AttrUtils::GetInt(op_desc_, kGlobalworkspaceType, globalworkspace_type);
    FE_LOGD("Get Globalworkspace size[%ld] and type[%ld] from node[%s, %s].", globalworkspace_size, globalworkspace_type,
            op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
    if (globalworkspace_size > 0) {
      uint8_t *globalworkspace_data_ptr = nullptr;
      device_addrs.push_back(static_cast<void *>(globalworkspace_data_ptr));
    }
  }
}

Status TbeTaskBuilderAdapter::Run(domi::TaskDef &task_def) {
  const auto &op_type = op_desc_->GetType();
  const auto &op_name = op_desc_->GetName();
  FE_LOGD("Node[%s, %s]: start to run, input addrs size[%zu], output addrs size[%zu], input l1 addrs size[%zu].",
          op_type.c_str(), op_name.c_str(), input_addrs_.size(), output_addrs_.size(), input_l1_addrs_.size());

  vector<void *> device_addrs;
  device_addrs.insert(device_addrs.cend(), input_addrs_.cbegin(), input_addrs_.cend());
  device_addrs.insert(device_addrs.cend(), output_addrs_.cbegin(), output_addrs_.cend());
  device_addrs.insert(device_addrs.cend(), workspace_addrs_.cbegin(), workspace_addrs_.cend());
  device_addrs.insert(device_addrs.cend(), input_l1_addrs_.cbegin(), input_l1_addrs_.cend());

  AppendArgsTilingData(device_addrs);
  AppendGlobalData(device_addrs);

  size_t input_num = ge::OpDescUtils::GetNonConstInputsSize(node_) + ge::OpDescUtils::GetWeights(node_).size();
  size_t output_num = output_addrs_.size();
  FE_LOGD("Node[%s, %s]: input_num=%zu, output_num=%zu, workspace_addrs_size=%zu, device_addrs_size=%zu.",
          op_type.c_str(), op_name.c_str(), input_num, output_num, workspace_addrs_.size(), device_addrs.size());

  Status ret = TbeForward(block_dim_, device_addrs.data(), sizeof(void *) * device_addrs.size(),
                          static_cast<int32_t>(input_num), const_cast<const void **>(input_addrs_.data()),
                          static_cast<int32_t>(input_addrs_.size()), static_cast<int32_t>(output_num),
                          const_cast<const void **>(output_addrs_.data()), 
                          static_cast<int32_t>(workspace_addrs_.size()), task_def);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GenTask][Run][Node %s type %s] TbeForward failed, ret[0x%X]",
                    op_name.c_str(), op_type.c_str(), ret);
    return FAILED;
  }
  if (task_def.type() == RT_MODEL_TASK_KERNEL) {
    domi::KernelDef *kernel_def = task_def.mutable_kernel();
    FE_CHECK_NOTNULL(kernel_def);
    FE_LOGD("Node[%s, %s]:Task type[%u] append kernel.", op_type.c_str(), op_name.c_str(), task_def.type());
    for (auto &arg : kernel_args_info_) {
      domi::ArgsInfo *arg_info = kernel_def->add_args_info();
      *arg_info = arg;
    }
  } else {
    domi::KernelDefWithHandle *kernel_def_with_handle = task_def.mutable_kernel_with_handle();
    FE_CHECK_NOTNULL(kernel_def_with_handle);
    FE_LOGD("Node[%s, %s]:Task type[%u] append kernel with handle.", op_type.c_str(), op_name.c_str(), task_def.type());
    for (auto &arg : kernel_args_info_) {
      domi::ArgsInfo *arg_info = kernel_def_with_handle->add_args_info();
      *arg_info = arg;
    }
  }
  FE_LOGD("Node[%s, %s]: end to run.", op_type.c_str(), op_name.c_str());
  return SUCCESS;
}
}  // namespace fe
