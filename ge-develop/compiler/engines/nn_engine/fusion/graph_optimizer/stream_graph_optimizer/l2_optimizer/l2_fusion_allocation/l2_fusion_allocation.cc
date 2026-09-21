/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_fusion_allocation/l2_fusion_allocation.h"
#include <cmath>
#include "common/math_util.h"
#include "common/fe_inner_error_codes.h"
#include "common/fe_utils.h"

namespace fe {
Status L2FusionAllocation::AllocateStandingDataByTaskNum(const uint32_t &node_task_num, const uint64_t &page_size,
                                                         const std::map<uint64_t, size_t> &count_map,
                                                         const TensorL2DataMap &output_data_map,
                                                         TensorL2AllocMap &tensor_l2_alloc_map,
                                                         uint32_t &data_in_l2_id,
                                                         int32_t &page_num_left) {
  FE_LOGD("nodeTaskNum is %u.", node_task_num);
  if (node_task_num > 1) {
    // kernel with multi tasks (fc & spatial transfprmer) should deal with the standing specially,
    FE_CHECK(AllocateStandingDataSpecial(page_size, count_map, tensor_l2_alloc_map, data_in_l2_id,
                                         page_num_left) != SUCCESS,
             REPORT_FE_ERROR("[StreamOpt][L2Opt][AllocStandData] AllocateStandingDataSpecial failed!"),
             return FAILED);
  } else {
    FE_CHECK(AllocateStandingData(page_size, count_map, output_data_map, tensor_l2_alloc_map, data_in_l2_id,
                                  page_num_left) != SUCCESS,
             FE_LOGW("AllocateStandingData failed!"), return FAILED);
  }
  FE_CHECK(page_num_left < 0,
           REPORT_FE_ERROR("[StreamOpt][L2Opt][AllocStandData] Page Number left less than zero!"), return FAILED);
  return SUCCESS;
}

Status L2FusionAllocation::AllocateData(const L2BufferInfo &l2_buffer_info,
                                        const std::vector<OpL2DataInfo> &op_l2_data_vec,
                                        std::map<uint64_t, size_t> &count_map,
                                        OpL2AllocMap &op_l2_alloc_map, uint64_t &max_page_num) {
  if (op_l2_data_vec.empty()) {
    FE_LOGW("The op l2 data need to be allocated is empty.");
    return SUCCESS;
  }
  FE_CHECK(l2_buffer_info.page_num == 0, REPORT_FE_ERROR("[StreamOpt][L2Opt][AllocData] l2 page Num is zero."),
           return FAILED);
  uint64_t page_size = l2_buffer_info.l2_buffer_size / l2_buffer_info.page_num;
  TensorL2AllocMap tensor_l2_alloc_map;  // all output l2 alloc info
  op_l2_alloc_map.clear();
  FE_LOGD("AllocateData data size is %zu bytes.", op_l2_data_vec.size());
  for (const OpL2DataInfo &op_l2_data : op_l2_data_vec) {
    FE_LOGD("Begin allocation of L2 info for node %s.", op_l2_data.node_name.c_str());
    int32_t page_num_left = static_cast<int32_t>(l2_buffer_info.page_num);
    int32_t data_numleft = static_cast<int32_t>(l2_buffer_info.max_data_num);
    const TensorL2DataMap &input_data_map = op_l2_data.input;
    const TensorL2DataMap &output_data_map = op_l2_data.output;
    FE_LOGD("pageNumLeft is %d, l2.page_num is %lu, data_numleft is %d.",
            page_num_left, l2_buffer_info.page_num, data_numleft);
    uint32_t data_in_l2_id = 0;
    FE_CHECK(AllocateStandingDataByTaskNum(op_l2_data.node_task_num, page_size, count_map, output_data_map,
                                           tensor_l2_alloc_map, data_in_l2_id, page_num_left) != SUCCESS,
             FE_LOGW("AllocateStandingDataByTaskNum failed!"), return FAILED);

    /* update alloc info */
    OpL2AllocInfo op_alloc_info;
    op_alloc_info.node_id = op_l2_data.node_id;
    op_alloc_info.node_name = op_l2_data.node_name;
    op_alloc_info.standing_data = tensor_l2_alloc_map;
    FE_CHECK(AllocateInputData(tensor_l2_alloc_map, input_data_map, count_map, op_alloc_info.input) != SUCCESS,
             REPORT_FE_ERROR("[StreamOpt][L2Opt][AllocData] AllocateInputData for node:%s failed!",
                             op_l2_data.node_name.c_str()), return FAILED);
    data_numleft = data_numleft - static_cast<int32_t>(tensor_l2_alloc_map.size());
    FE_CHECK(AllocateOutputData(page_size, l2_buffer_info.page_num, output_data_map, data_in_l2_id, data_numleft,
                                page_num_left, tensor_l2_alloc_map, op_alloc_info.output) != SUCCESS,
             FE_LOGW("OutputData allocation failed!"), return FAILED);

    op_l2_alloc_map.insert(std::pair<string, OpL2AllocInfo>(op_alloc_info.node_name, op_alloc_info));

    /* update max page num info */
    FE_LOGD("After allocation, the page_num_left value is %d, max_page_num is %lu, and l2.page_num is %lu.",
            page_num_left, max_page_num, l2_buffer_info.page_num);
    if (l2_buffer_info.page_num - page_num_left > max_page_num) {
      max_page_num = l2_buffer_info.page_num - page_num_left;
    }
  }
  return SUCCESS;
}

void L2FusionAllocation::EraseStandingDataAllocCountMapZero(const std::map<uint64_t, size_t> &count_map,
                                                            TensorL2AllocMap &tensor_l2_alloc_map) {
  for (auto it = tensor_l2_alloc_map.cbegin(); it != tensor_l2_alloc_map.cend();) {
    const auto iter = count_map.find(it->first);
    if(iter != count_map.cend()) {
      if(iter->second == 0) {
        FE_LOGD("countMap[%lu] is 0; need to remove it.", it->first);
        it = tensor_l2_alloc_map.erase(it);
        continue;
      }
      else{
        FE_LOGD("countMap[%lu] is %zu, no need to erase it.", it->first, iter->second);
      }
    }
    ++it;
  }
}

Status L2FusionAllocation::AllocateStandingData(const int64_t &page_size, const std::map<uint64_t, size_t> &count_map,
                                                const TensorL2DataMap &output_data_map,
                                                TensorL2AllocMap &tensor_l2_alloc_map, uint32_t &data_in_l2_id,
                                                int32_t &page_num_left) {
  FE_LOGD("The size of all tensor L2 allocation information is %zu bytes.", tensor_l2_alloc_map.size());
  EraseStandingDataAllocCountMapZero(count_map, tensor_l2_alloc_map);
  Status ret = UpdateStandingDataSizeWithOutputSet(page_size, output_data_map, tensor_l2_alloc_map, page_num_left);
  FE_CHECK(ret != SUCCESS, FE_LOGD("UpdateStandingDataSizeWithOutputSet directly returns."), return ret);
  uint8_t cur_page_offset_base = 0;
  for (TensorL2AllocMap::iterator it = tensor_l2_alloc_map.begin(); it != tensor_l2_alloc_map.end(); ++it) {
    TensorL2AllocInfo &cur_l2_data = it->second;
    cur_l2_data.data_in_l2_id = static_cast<int32_t>(data_in_l2_id++);
    cur_l2_data.pre_L2_page_offset_base = static_cast<int8_t>(cur_l2_data.L2_page_offset_base);
    cur_l2_data.L2_page_offset_base = cur_page_offset_base;
    cur_l2_data.priority = 0;

    FE_INT64_MULCHECK(static_cast<int64_t>(cur_l2_data.L2_page_offset_base), static_cast<int64_t>(page_size));
    cur_l2_data.data_in_l2_addr = static_cast<uint64_t>(static_cast<int64_t>(cur_l2_data.L2_page_offset_base) *
                                                        static_cast<int64_t>(page_size));

    FE_UINT8_ADDCHECK(cur_page_offset_base, static_cast<uint8_t>(cur_l2_data.l2PageNum));
    cur_page_offset_base += static_cast<uint8_t>(cur_l2_data.l2PageNum);
  }
  for (auto it = tensor_l2_alloc_map.cbegin(); it != tensor_l2_alloc_map.cend(); ++it) {
    page_num_left -= it->second.l2PageNum;
    FE_CHECK((page_num_left < 0),
             REPORT_FE_ERROR("[StreamOpt][L2Opt][AllocStandData] Left page number %d is less than zero.", page_num_left),
             return FAILED);
  }
  return SUCCESS;
}

Status L2FusionAllocation::AllocateStandingDataSpecial(const int64_t &page_size,
                                                       const std::map<uint64_t, size_t> &count_map,
                                                       TensorL2AllocMap &tensor_l2_alloc_map,
                                                       uint32_t &data_in_l2_id, int32_t &page_num_left) {
  EraseStandingDataAllocCountMapZero(count_map, tensor_l2_alloc_map);
  int32_t max_page_num = page_num_left;
  for (TensorL2AllocMap::iterator it = tensor_l2_alloc_map.begin(); it != tensor_l2_alloc_map.end(); ++it) {
    TensorL2AllocInfo &cur_l2_data = it->second;
    cur_l2_data.data_in_l2_id = static_cast<int32_t>(data_in_l2_id++);
    cur_l2_data.pre_L2_page_offset_base = static_cast<int8_t>(cur_l2_data.L2_page_offset_base);
    cur_l2_data.L2_page_offset_base = cur_l2_data.pre_L2_page_offset_base;
    cur_l2_data.priority = 0;

    FE_INT64_MULCHECK(static_cast<int64_t>(cur_l2_data.L2_page_offset_base), page_size);
    cur_l2_data.data_in_l2_addr = static_cast<uint64_t>(static_cast<int64_t>(cur_l2_data.L2_page_offset_base) *
                                                        page_size);

    FE_INT32_SUBCHECK(max_page_num, static_cast<int32_t>(cur_l2_data.L2_page_offset_base));
    int32_t tmp = max_page_num - static_cast<int32_t>(cur_l2_data.L2_page_offset_base);
    FE_INT32_SUBCHECK(tmp, static_cast<int32_t>(cur_l2_data.l2PageNum));

    page_num_left =
        std::min(page_num_left, max_page_num - static_cast<int32_t>(cur_l2_data.L2_page_offset_base) -
                 static_cast<int32_t>(cur_l2_data.l2PageNum));
    FE_CHECK((page_num_left < static_cast<int32_t>(0)),
             REPORT_FE_ERROR("[StreamOpt][L2Opt][AllocStandDataSpec] Left page number is %d, less than zero.",
                             page_num_left),
             return FAILED);
  }
  return SUCCESS;
}

Status L2FusionAllocation::AllocateInputData(const TensorL2AllocMap &tensor_l2_alloc_map,
                                             const TensorL2DataMap &input_data_map,
                                             std::map<uint64_t, size_t> &count_map,
                                             TensorL2AllocMap &input_alloc_map) {
  input_alloc_map.clear();
  for (auto it = input_data_map.cbegin(); it != input_data_map.cend(); ++it) {
    auto count_iter = count_map.find(it->first);
    if (count_iter == count_map.end()) {
      FE_LOGD("Input id [%lu] cannot be found in count map.", it->first);
      continue;
    }

    FE_CHECK(count_iter->second <= 0,
             REPORT_FE_ERROR("[StreamOpt][L2Opt][AllocInData] Count map value %zu is zero, with id %zu.",
                             count_iter->second, it->first),
             return FAILED);
    count_iter->second--;

    TensorL2AllocMap::const_iterator alloc_iter = tensor_l2_alloc_map.find(it->first);
    FE_LOGD("The size of the tensor l2 alloc info is %zu.", tensor_l2_alloc_map.size());
    if (alloc_iter != tensor_l2_alloc_map.end()) {
      input_alloc_map.insert(TensorL2AllocPair(it->first, alloc_iter->second));
    }
  }
  return SUCCESS;
}

Status L2FusionAllocation::InitDataAlloc(const TensorL2DataInfo &l2_data, const uint8_t &priority, const int32_t &page,
                                         const uint32_t &max_page_num, const int32_t &page_num_left,
                                         const int64_t &page_size, uint32_t &data_in_l2_id,
                                         TensorL2AllocInfo &tensor_alloc_info) {
  tensor_alloc_info.data_in_l2_id = static_cast<int32_t>(data_in_l2_id++);
  tensor_alloc_info.data = l2_data;
  tensor_alloc_info.l2PageNum = static_cast<uint64_t>(page);
  tensor_alloc_info.pre_L2_page_offset_base = -1;
  if ((static_cast<int64_t>(max_page_num) - static_cast<int64_t>(page_num_left) < 0) ||
      (static_cast<int64_t>(max_page_num) - static_cast<int64_t>(page_num_left) > UINT8_MAX)) {
    REPORT_FE_ERROR("[StreamOpt][L2Opt][InitDataAlloc] Failed to initialize L2 allocation information, max_page_num: %u, page_num_left: %d.",
                    max_page_num, page_num_left);
    return FAILED;
  }
  tensor_alloc_info.L2_page_offset_base =
          static_cast<uint8_t>(static_cast<uint16_t>(static_cast<int64_t>(max_page_num) -
                                                     static_cast<int64_t>(page_num_left)));
  tensor_alloc_info.data_in_l2_addr = static_cast<uint64_t>(tensor_alloc_info.L2_page_offset_base) *
                                      static_cast<uint64_t>(page_size);
  tensor_alloc_info.modified = 1;
  tensor_alloc_info.L2_preload = 0;
  tensor_alloc_info.priority = priority;
  tensor_alloc_info.L2_load_to_ddr = 0;
  tensor_alloc_info.occupy_data_id = -1;
  return SUCCESS;
}

Status L2FusionAllocation::AllocateOutputData(const int64_t &page_size, const uint32_t &max_page_num,
                                              const TensorL2DataMap &output_data_map, uint32_t &data_in_l2_id,
                                              int32_t &data_num_left, int32_t &page_num_left,
                                              TensorL2AllocMap &tensor_l2_alloc_map,
                                              TensorL2AllocMap &output_alloc_map) {
  // directly return
  FE_CHECK(page_num_left < 0, FE_LOGD("Left page number is less than zero."), return SUCCESS);
  FE_CHECK(output_data_map.size() == 0, FE_LOGD("Output size is zero."), return SUCCESS);

  // error checking
  FE_CHECK(page_size == 0, REPORT_FE_ERROR("[StreamOpt][L2Opt][AllocOutData] Page size is zero."), return FAILED);
  for (auto it = output_data_map.cbegin(); it != output_data_map.cend(); ++it) {
    FE_LOGD("Begin to allocate L2 info for output data with ID [%lu].", it->first);
    if (it->second.refer_data_id > 0) {
      FE_LOGD("The reference data ID for output ID [%lu] is [%lu]. The reference L2 allocation information will be utilized.",
              it->first, it->second.refer_data_id);
      TensorL2AllocMap::const_iterator iter = tensor_l2_alloc_map.find(it->second.refer_data_id);
      if (iter != tensor_l2_alloc_map.cend()) {
        output_alloc_map.insert(TensorL2AllocPair(it->first, iter->second));
        tensor_l2_alloc_map.insert(TensorL2AllocPair(it->first, iter->second));
      } else {
        FE_LOGD("The L2 allocation information for reference data ID [%lu] was not found.", it->second.refer_data_id);
      }
      // if the l2 of refer data id is not alloced, the l2 of current of this output will also not be alloced.
      continue;
    }
    TensorL2AllocMap::const_iterator alloc_iter = tensor_l2_alloc_map.find(it->first);
    if (alloc_iter == tensor_l2_alloc_map.cend()) {
      const double const_double = 1.0;
      int32_t page = static_cast<int32_t>(std::ceil(const_double * it->second.data_size / page_size));
      FE_LOGD("page is %d, page_size is %ld, page_num_left is %d.", page, page_size, page_num_left);
      // the data num and page num both has left space
      bool is_has_left_space = (data_num_left > 0) && (page <= page_num_left);
      FE_LOGD("is_has_left_space is %d", is_has_left_space);
      if (is_has_left_space) {
        TensorL2AllocInfo tensor_l2_alloc_info;
        FE_CHECK(InitDataAlloc(it->second, 1, page, max_page_num, page_num_left, page_size,
                               data_in_l2_id, tensor_l2_alloc_info) != SUCCESS,
                 REPORT_FE_ERROR("[StreamOpt][L2Opt][AllocOutData] InitDataAlloc failed!"), return FAILED);
        output_alloc_map.insert(TensorL2AllocPair(it->first, tensor_l2_alloc_info));
        tensor_l2_alloc_map.insert(TensorL2AllocPair(it->first, tensor_l2_alloc_info));
        FE_INT32_SUBCHECK(page_num_left, page);
        page_num_left = page_num_left - page;
        data_num_left--;
      }
    } else {
      FE_LOGD("StandingAlloc ID is %lu. Its ID is %lu.", alloc_iter->first, it->first);
      // the output is already allocted
      // converge data, make sure standing space is enough for output data
      FE_CHECK(alloc_iter->second.data.data_size < it->second.data_size,
               FE_LOGW("Standing data size %ld is less than output size %ld.", alloc_iter->second.data.data_size,
                       it->second.data_size),
               return FAILED);
      output_alloc_map.insert(TensorL2AllocPair(it->first, alloc_iter->second));
    }
  }
  return SUCCESS;
}

Status L2FusionAllocation::UpdateStandingDataSizeWithOutputSet(const int64_t &page_size,
                                                               const TensorL2DataMap &output_data_map,
                                                               TensorL2AllocMap &tensor_l2_alloc_map,
                                                               int32_t page_num_left) {
  FE_CHECK(page_size == 0,
           REPORT_FE_ERROR("[StreamOpt][L2Opt][UpdStandDataSize] The pageSize is set to zero."), return FAILED);
  for (TensorL2AllocMap::iterator it = tensor_l2_alloc_map.begin(); it != tensor_l2_alloc_map.end(); ++it) {
    TensorL2DataMap::const_iterator data_iterator = output_data_map.find(it->first);
    // output batch
    if (data_iterator != output_data_map.cend() && (data_iterator->second.data_size > it->second.data.data_size)) {
      const double const_double = 1.0;
      int32_t page = static_cast<int32_t>(std::ceil(const_double * data_iterator->second.data_size / page_size));
      FE_CHECK(page > page_num_left, FE_LOGW("Page %d is bigger than left page number %d.", page, page_num_left),
               return FAILED);
      it->second.data.data_size = data_iterator->second.data_size;
      it->second.l2PageNum = static_cast<uint64_t>(page);
    }

    page_num_left = page_num_left - it->second.l2PageNum;
  }
  return SUCCESS;
}
}  // namespace fe
