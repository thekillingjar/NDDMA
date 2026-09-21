/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "inc/memory_slice.h"
#include "inc/ffts_utils.h"

namespace ffts {
namespace {
const uint32_t kDimIndex = 2;

Status Block2DataCtxParam(const Block &block, int64_t dtype_size, int64_t burst_len,
                          std::vector<DataContextParam> &data_ctx) {
  if (burst_len <= 0) {
    REPORT_FFTS_ERROR("Burst length %ld is less than zero.", burst_len);
    return FAILED;
  }
  FFTS_INT64_MULCHECK(block.dim[0U], dtype_size);
  int64_t dim_0_size = block.dim[0U] * dtype_size;

  int64_t burst_count = dim_0_size / burst_len;
  int64_t burst_remain = dim_0_size % burst_len;
  /* Li may larger than the burst_len(which is the maximum number of
   * data in the inner dimension). We will separate the inner block
   * and give each param a additional offset(burst_len * index after seperated).
   */
  int64_t inner_block_count = burst_count + (burst_remain > 0 ? 1 : 0);

  for (int64_t i = 0; i < block.count; i++) {
    int64_t block_offset = block.offset + block.stride * i;
    FFTS_INT64_MULCHECK(block_offset, dtype_size);
    int64_t block_offset_size = block_offset * dtype_size;

    for (int64_t j = 0; j < inner_block_count; j++) {
      DataContextParam ctx_param = {};
      int64_t curr_len = (j < burst_count ? burst_len : burst_remain);
      ctx_param.len_inner = curr_len;
      ctx_param.num_inner = block.dim[1U];
      FFTS_INT64_MULCHECK(block.dim_stride[1U], dtype_size);
      ctx_param.stride_inner = block.dim_stride[1U] * dtype_size;
      ctx_param.num_outter = block.dim[kDimIndex];
      FFTS_INT64_MULCHECK(block.dim_stride[kDimIndex], dtype_size);
      ctx_param.stride_outter = block.dim_stride[kDimIndex] * dtype_size;
      ctx_param.base_addr_offset = block_offset_size + j * burst_len;
      data_ctx.push_back(ctx_param);
    }
  }
  return SUCCESS;
}

Status DimensionMerge(const std::vector<int64_t> &shape,
                      const std::vector<DimRange> &slice, Block &block) {
  size_t dim_count = shape.size();
  if (dim_count != slice.size()) {
    FFTS_LOGW("[GenTsk][DataTsk][DimMerge] The dimension count %zu is not the same as the slice size %zu.",
              dim_count, slice.size());
    return FAILED;
  }

  int64_t stride = 1;
  int64_t pre_stride = 1;
  int64_t accumulation = 1;
  int curr_dim = 0;
  Block blk = {};
  for (size_t i = 0; i < dim_count; i++) {
    size_t i_reverse = dim_count - 1 - i;
    int64_t dim = shape[i_reverse];
    int64_t slice_low = slice[i_reverse].lower;
    int64_t slice_high = slice[i_reverse].higher;
    if (slice_low < 0 || slice_high <= slice_low) {
      REPORT_FFTS_ERROR("[GenTsk][DataTsk][DimMerge]slice_low %ld is less than 0 or greater than slice_high %ld.", slice_low, slice_high);
      return FAILED;
    }

    FFTS_LOGD("before block dimensions are: {%ld, %ld, %ld}, stride dimensions {%ld, %ld, %ld}."
              "slice low = %ld, "
              "ori dim = %ld, "
              "blk.offset = %ld, "
              "stride = %ld.",
              blk.dim[0U], blk.dim[1U], blk.dim[2U],
              blk.dim_stride[0U], blk.dim_stride[1U], blk.dim_stride[2U],
              slice_low, dim, blk.offset, stride);
    if (slice_high - slice_low != dim) {
      /* If current dim is equal to 3 and slice number is 4, we cannot describe the slicing.
       * Because using inner len + outter len can not describe the 4th
       * axis slicing. */
      if (curr_dim >= kBlockDim) {
        REPORT_FFTS_ERROR("[GenTsk][DataTsk][DimMerge] curr_dim %d is larger than max block dim(3).", curr_dim);
        return FAILED;
      }
      blk.dim[curr_dim] = (slice_high - slice_low) * accumulation;
      accumulation = 1;
      blk.offset += (slice_low * stride);
      blk.dim_stride[curr_dim] = pre_stride;
      curr_dim++;
      stride *= dim;
      pre_stride = stride;
    } else {
      // no slicing on this dim, fuse it
      stride *= dim;
      accumulation *= dim;
    }
    FFTS_LOGD("after block dimensions are: {%ld, %ld, %ld}, stride dimensions {%ld, %ld, %ld}."
              "slice low = %ld, "
              "ori dim = %ld, "
              "blk.offset = %ld, "
              "stride = %ld.",
              blk.dim[0U], blk.dim[1U], blk.dim[2U],
              blk.dim_stride[0U], blk.dim_stride[1U], blk.dim_stride[2U],
              slice_low, dim, blk.offset, stride);
  }

  /* block dim[0] is equal to zero means we do not split any of the axes.
   * So the block dim[0] == stride of dim[0].
   * And we have only one whole inner block.
   * dim[1] = dim[2] = 1, and stride is the same as stride of
   * dim[0]. */
  for (int64_t i = 0; i < static_cast<int64_t>(kBlockDim); i++) {
    if (blk.dim[i] == 0) {
      blk.dim[i] = accumulation;
      // in case of less than 3 split
      blk.dim_stride[i] = pre_stride;
      pre_stride = blk.dim[i] * blk.dim_stride[i];
      accumulation = 1;
    }
  }
  /* If the accumulation is not 1, we will seperate the total
   * stride into several block and for each the stride is
   * total stride divided by accumulation. */
  blk.stride = stride / accumulation;
  blk.count = accumulation;

  block = blk;
  FFTS_LOGD("block dimensions are: {%ld, %ld, %ld}, stride dimensions {%ld, %ld, %ld}.",
            block.dim[0U], block.dim[1U], block.dim[2U],
            block.dim_stride[0U], block.dim_stride[1U], block.dim_stride[2U]);
  FFTS_LOGD("offset: %ld, count %ld, stride %ld.", block.offset, block.count, block.stride);
  return SUCCESS;
}

Status GetPeerNodeAndOutIdx(const ge::NodePtr &ori_node, int32_t ori_index,
                            ge::NodePtr &peer_node, int32_t &peer_index) {
  auto in_anchor = ori_node->GetInDataAnchor(ori_index);
  FFTS_CHECK_NOTNULL(in_anchor);
  auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
  FFTS_CHECK_NOTNULL(peer_out_anchor);

  peer_node = peer_out_anchor->GetOwnerNode();
  FFTS_CHECK_NOTNULL(peer_node);
  peer_index = peer_out_anchor->GetIdx();
  return SUCCESS;
}

Status GetSliceList(const ge::NodePtr &node, int index, bool is_input, ge::GeTensorDescPtr &tensor,
                    vector<std::vector<DimRange>> &slice_list) {
  ThreadSliceMapPtr slice_info_ptr = nullptr;
  slice_info_ptr = node->GetOpDesc()->TryGetExtAttr(kAttrSgtStructInfo, slice_info_ptr);
  FFTS_CHECK_NOTNULL(slice_info_ptr);
  uint32_t thread_id = 0;
  auto tensor_slice = is_input ? slice_info_ptr->input_tensor_slice : slice_info_ptr->output_tensor_slice;
  if (tensor_slice.size() <= thread_id) {
    FFTS_LOGW("[GenTsk][DataTsk][MemSlice]tensor_slice of node %s is empty.", node->GetName().c_str());
    return FAILED;
  }
  if (tensor_slice[thread_id].size() <= static_cast<size_t>(index)) {
    FFTS_LOGW("[GenTsk][DataTsk][MemSlice]size of thread %u of node %s is %zu, which is <= %d.",
              thread_id, node->GetName().c_str(), slice_info_ptr->input_tensor_slice[thread_id].size(), index);
    return FAILED;
  }
  if (is_input) {
    ge::NodePtr peer_out_node;
    int peer_index = 0;
    if (GetPeerNodeAndOutIdx(node, index, peer_out_node, peer_index) != SUCCESS) {
      return FAILED;
    }
    FFTS_CHECK_NOTNULL(peer_out_node);
    tensor = peer_out_node->GetOpDesc()->MutableOutputDesc(static_cast<uint32_t>(peer_index));
  } else {
    tensor = node->GetOpDesc()->MutableOutputDesc(static_cast<uint32_t>(index));
  }
  slice_list.push_back(tensor_slice[thread_id][index]);
  if (slice_info_ptr->thread_mode == kAutoMode) {
    auto last_index = tensor_slice.size() - 1;
    if (tensor_slice[last_index].size() <= static_cast<size_t>(index)) {
      REPORT_FFTS_ERROR("[GenTsk][DataTsk][MemSlice] Size of last_index[%zu] of node[%s] is %zu, which is <= %d",
                        last_index, node->GetName().c_str(), tensor_slice[last_index].size(), index);
      return FAILED;
    }
    slice_list.push_back(tensor_slice[last_index][index]);
  }
  return SUCCESS;
}
}  // namespace

Status MemorySlice::GenerateDataCtxParam(const std::vector<int64_t> &shape,
                                         const std::vector<DimRange> &slice,
                                         ge::DataType dtype, int64_t burst_len,
                                         std::vector<DataContextParam> &data_ctx) {
  Block block;
  if (DimensionMerge(shape, slice, block) != SUCCESS) {
    return FAILED;
  } else {
    auto iter = DATATYPE_SIZE_MAP.find(dtype);
    if (iter == DATATYPE_SIZE_MAP.end()) {
      FFTS_LOGD("Data type %u is not found in the DATATYPE_SIZE_MAP.", dtype);
      return FAILED;
    }
    int64_t dtype_size = iter->second;
    FFTS_LOGD("Burst len and data type size is %ld and %ld.", burst_len, dtype_size);
    (void)Block2DataCtxParam(block, dtype_size, burst_len, data_ctx);
  }
  return SUCCESS;
}

Status MemorySlice::GenerateManualDataCtxParam(const ge::NodePtr &node, int32_t index, bool is_input, int64_t burst_len,
                                               std::vector<DataContextParam> &data_ctx) {
  ge::GeTensorDescPtr tensor_desc = nullptr;
  vector<std::vector<DimRange>> slice_list;
  if (GetSliceList(node, index, is_input, tensor_desc, slice_list) != SUCCESS) {
    FFTS_LOGW("[GenTsk][DataTsk][GenManualDataCtxParam] Index[%d] of node[%s] Get slice list failed.",
              index, node->GetName().c_str());
    return FAILED;
  }
  if (slice_list.size() != 1) {
    FFTS_LOGW("[GenTsk][DataTsk][GenManualDataCtxParam] Slice list size for index [%d] of node [%s] is %zu.",
              index, node->GetName().c_str(), slice_list.size());
    return FAILED;
  }
  FFTS_CHECK_NOTNULL(tensor_desc);
  auto shape = tensor_desc->MutableShape();
  bool is_first_slice = false;
  std::vector<int64_t> shape_dims = shape.GetDims();
  (void)ge::AttrUtils::GetBool(node->GetOpDesc(), kFftsFirstSliceFlag, is_first_slice);
  if (is_first_slice && !is_input) {
    FFTS_LOGD("is_first_slice node: %s, changing the shape and slice info.", node->GetName().c_str());
    std::vector<std::vector<int64_t>> ori_output_tensor_shape;
    (void)ge::AttrUtils::GetListListInt(node->GetOpDesc(), "ori_output_tensor_shape", ori_output_tensor_shape);
    if (!ori_output_tensor_shape.empty() && static_cast<size_t>(index) < ori_output_tensor_shape.size() &&
        shape_dims.size() == ori_output_tensor_shape[index].size()) {
      shape_dims = ori_output_tensor_shape[index];
      for (size_t dim_idx = 0; dim_idx < shape_dims.size(); dim_idx++) {
        slice_list[0][dim_idx].higher = ori_output_tensor_shape[index][dim_idx];
        slice_list[0][dim_idx].lower = 0;
      }
    }
  }
  FFTS_LOGD("[GenManualDataCtxParam] slice size is: %zu, Shape of index %d of node %s is %s.", slice_list.size(), index,
            node->GetName().c_str(), fe::StringUtils::IntegerVecToString(shape_dims).c_str());
  FFTS_LOGD("Range is: {");
  for (const auto &eles : slice_list) {
    for (const auto &ele : eles) {
      FFTS_LOGD("[%ld, %ld]", ele.lower, ele.higher);
    }
  }
  FFTS_LOGD("}");
  return GenerateDataCtxParam(shape_dims, slice_list[0U], tensor_desc->GetDataType(), burst_len, data_ctx);
}

Status MemorySlice::GenerateAutoDataCtxParam(const ge::NodePtr &node, int32_t index, bool is_input, int64_t burst_len,
                                             std::vector<DataContextParam> &param_nontail_tail) {
  ge::GeTensorDescPtr tensor_desc = nullptr;
  vector<std::vector<DimRange>> slice_list;
  if (GetSliceList(node, index, is_input, tensor_desc, slice_list) != SUCCESS) {
    REPORT_FFTS_ERROR("[GenTsk][DataTsk][GenAutoDataCtxParam] Failed to get slice list for index [%d] of node [%s].",
                      index, node->GetName().c_str());
    return FAILED;
  }
  if (slice_list.size() <= 1) {
    REPORT_FFTS_ERROR("[GenTsk][DataTsk][GenAutoDataCtxParam] Index[%d] of node[%s] slice list size is %zu",
                      index, node->GetName().c_str(), slice_list.size());
    return FAILED;
  }
  FFTS_CHECK_NOTNULL(tensor_desc);
  auto shape = tensor_desc->MutableShape();
  FFTS_LOGD("[GenTsk][DataTsk][GenAutoDataCtxParam] slice size is: %zu, Shape of index %d of node %s is %s.",
            slice_list.size(), index, node->GetName().c_str(),
            fe::StringUtils::IntegerVecToString(shape.GetDims()).c_str());
  FFTS_LOGD("Range is: {");
  for (const auto &eles : slice_list) {
    for (const auto &ele : eles) {
      FFTS_LOGD("[%ld, %ld]", ele.lower, ele.higher);
    }
  }
  FFTS_LOGD("}");

  // single data context
  for (const auto &slice : slice_list) {
     std::vector<DataContextParam> data_ctx_param;
     GenerateDataCtxParam(shape.GetDims(), slice, tensor_desc->GetDataType(), burst_len, data_ctx_param);
     if (data_ctx_param.size() != 1) {
       FFTS_LOGW("[GenTsk][DataTsk][GenAutoDataCtxParam] Index[%d] of node[%s] has data_ctx_param size:[%zu], which is not equal to 1.", index,
                 node->GetName().c_str(), data_ctx_param.size());
       return FAILED;
     }
     param_nontail_tail.push_back(data_ctx_param[0U]);
  }
  return SUCCESS;
}
}
