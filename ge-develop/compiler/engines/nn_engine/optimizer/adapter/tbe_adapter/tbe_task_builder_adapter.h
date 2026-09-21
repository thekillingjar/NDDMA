/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_TBE_ADAPTER_TBE_TASK_BUILDER_ADAPTER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_TBE_ADAPTER_TBE_TASK_BUILDER_ADAPTER_H_

#include <map>
#include <mutex>
#include <string>
#include <vector>
#include "adapter/adapter_itf/task_builder_adapter.h"
#include "graph/op_kernel_bin.h"
#include "graph/utils/anchor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "common/aicore_util_types.h"
#include "common/l2fusion_struct.h"

namespace fe {
struct InputIndexOffsetInfo {
  size_t input_index = 0;
  size_t anchor_index = 0;
  size_t weight_index = 0;
  vector<int64_t> input_offsets;
  vector<bool> input_is_addr_var;
  vector<uint32_t> input_type_list;
};

class TbeTaskBuilderAdapter : public TaskBuilderAdapter {
 public:
  TbeTaskBuilderAdapter(const ge::Node &node, TaskBuilderContext &context);
  ~TbeTaskBuilderAdapter() override;

  /*
   * @ingroup fe
   * @brief   Init TaskBuilderAdapter
   * @return  SUCCESS or FAILED
   */
  Status Init() override;

  /*
   * @ingroup fe
   * @brief   Run TaskBuilderAdapter
   * @return  SUCCESS or FAILED
   */
  Status Run(domi::TaskDef &task_def) override;

  Status CheckInputAndOutputSize();

 protected:
  Status InitInput() override;

 private:
  uint32_t block_dim_;
  Status TbeForward(const uint32_t core_dim, const void *args, uint32_t args_size,
                    int32_t input_num, const void *x[], int32_t x_array_size,
                    int32_t output_num,const void *y[], int32_t workspace_num,
                    domi::TaskDef &task_def);

  Status SaveTeCoreL2FlowDataForL2Buffer(int32_t input_num, int32_t output_num, uint64_t cur_ptr,
                                         const void *x[], const void *y[],
                                         rtL2Ctrl_t &tel2ctrl, uint32_t l2_args_size, uint32_t workspace_num);
  Status SaveTeCoreL2FlowDataForL2Fusion(int32_t input_num, int32_t output_num, uint64_t cur_ptr,
                                         const void *x[], const void *y[],
                                         rtL2Ctrl_t &tel2ctrl, uint32_t l2_args_size, uint32_t workspace_num);

  std::string GetUniqueGraphIdForNode() const;

  uint64_t GetAtomicStubFuncId() const;

  Status HandleAnchorWeight(const size_t &anchor_index);

  Status CheckAnchorInputValueSkip(size_t input_index) const;

  void SetInputAddrFromDataBase(const size_t input_index, const int64_t &input_offset);

  Status CheckArrayValue(const void *array[], int32_t array_size, int32_t num, const string& name) const;

  Status CheckForForward(const void *args, const void *x[], int32_t x_array_size, const void *y[],
                         int32_t input_num, int32_t output_num) const;

  Status CheckTensorSize(const ge::GeTensorDesc &tensor_desc, uint32_t i, bool is_input,
                         int32_t output_real_calc_flag) const;

  Status DealKernelLaunchForL2Buffer(int32_t input_num, int32_t output_num, uint64_t cur_ptr,
                                     const void *x[], const void *y[], rtL2Ctrl_t &tel2ctrl,
                                     uint32_t args_size, uint32_t l2_args_size, const std::string &stub_func,
                                     const uint32_t core_dim, const void *tmp_buf, int32_t workspace_num,
                                     domi::TaskDef &task_def);

  Status DealKernelLaunchForL2Fusion(int32_t input_num, int32_t output_num,
                                     uint64_t cur_ptr, const void *x[], const void *y[], rtL2Ctrl_t &tel2ctrl,
                                     uint32_t args_size, uint32_t l2_args_size, const std::string &stub_func,
                                     const uint32_t core_dim, const void *tmp_buf, int32_t workspace_num,
                                     domi::TaskDef &task_def);

  void DealInputOutputL2DataMap(const L2DataMap &l2datamap, int32_t data_num, const void *x[], const void *y[],
                                uint64_t &cur_ptr, uint32_t &l2_args_size, bool is_input) const;

  void DealInputOutputL2DataMap(const L2FusionDataMap_t &l2datamap, int32_t data_num, const void *x[], const void *y[],
                                uint64_t &cur_ptr, uint32_t &l2_args_size, bool is_input) const;

  void DisplayRtL2CtrlInfo(const rtL2Ctrl_t &l2ctrl, bool enable_l2) const;

  void DealInputOutputWithDdr(int32_t data_num, uint64_t &cur_ptr, uint32_t &l2_args_size) const;

  void MemCpyForL2IdAndL2Addr(uint64_t &cur_ptr, uint32_t &l2_args_size, int64_t data_in_l2_id,
                              uint64_t data_in_l2_addr) const;

  Status InitInputNoPlaceholder();

  void InsertMissOptAddr(size_t &arg_idx, std::vector<uint32_t> &insert_pos_vec);

  Status InitInputGenPlaceholder();

  Status GetInputIndexOffsetInfos(InputIndexOffsetInfo &index_offset_info);

  Status FeedInputAddrByAnchor(const ge::InDataAnchorPtr &anchor, InputIndexOffsetInfo &index_offset_info,
                               bool is_gen_place_holder = false);

  bool GetUnknownShapeFlag();

  void AppendArgsTilingData(vector<void *> &device_addrs);

  void AppendGlobalData(vector<void *> &device_addrs);
};
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_TBE_ADAPTER_TBE_TASK_BUILDER_ADAPTER_H_
