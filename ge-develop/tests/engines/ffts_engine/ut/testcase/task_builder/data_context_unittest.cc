/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <list>
#include "ffts_llt_utils.h"
#define private public
#define protected public
#include "common/sgt_slice_type.h"
#include "common/aicore_util_constants.h"
#include "inc/ffts_log.h"
#include "inc/ffts_error_codes.h"
#include "inc/ffts_type.h"
#include "task_builder/mode/data_task_builder.h"
#include "task_builder/data_ctx/prefetch_manual_task_builder.h"
#include "task_builder/data_ctx/out_manual_task_builder.h"
#include "task_builder/fftsplus_ops_kernel_builder.h"
#include "inc/ffts_configuration.h"
#include "graph/node.h"
#include "graph/utils/tensor_utils.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "runtime/context.h"
#include "runtime/stream.h"
#include "runtime/rt_model.h"
#include "runtime/kernel.h"
#include "runtime/mem.h"
#include "../../../graph_constructor/graph_constructor.h"

using namespace std;
using namespace ffts;
using namespace ge;

class FftsPlusDataContextUT : public testing::Test {
 protected:
  void SetUp() {
    prefetch_.SetOperation(CACHE_OPERATION::PREFETCH);
    invalidate_.SetOperation(CACHE_OPERATION::INVALIDATE);
    write_back_.SetOperation(CACHE_OPERATION::WRITE_BACK);
    prefetch_auto_.SetOperation(CACHE_OPERATION::PREFETCH);
    invalidate_auto_.SetOperation(CACHE_OPERATION::INVALIDATE);
    write_back_auto_.SetOperation(CACHE_OPERATION::WRITE_BACK);
    prefetch_dyn_.SetOperation(CACHE_OPERATION::PREFETCH);
    invalidate_dyn_.SetOperation(CACHE_OPERATION::INVALIDATE);
    write_back_dyn_.SetOperation(CACHE_OPERATION::WRITE_BACK);
    Configuration &config = Configuration::Instance();
    config.is_init_ = false;
    config.lib_path_ = GetCodeDir() + "/tests/engines/ffts_engine/config/";
    config.LoadConfigFile();
    config.InitCMOThreshold();
  }

  void TearDown() {

  }
  void AssembleTensor(GeTensorDescPtr &tensor,
                      const vector<uint32_t> &axes,
                      uint32_t thread_id,
                      uint32_t slice_num,
                      vector<vector<DimRange>> &one_thread) {
    vector<DimRange> one_tensor;
    uint32_t axis_index = 0;
    vector<int64_t> new_dim = tensor->GetShape().GetDims();
    for (int64_t dim : tensor->GetShape().GetDims()) {
      DimRange range = {0, dim};
      if (std::find(axes.begin(), axes.end(), axis_index) != axes.end()) {
        // only the highest dimension needs to be sliced.
        int64_t dim_non_tail = dim / slice_num;
        int64_t low, high;
        if (thread_id == slice_num - 1) {
          low = thread_id * dim_non_tail;
          high = dim;
        } else {
          low = thread_id * dim_non_tail;
          high = (thread_id + 1) * dim_non_tail;
        }
        range = {low, high};
        new_dim[axis_index] = high - low;
      } else {
        range = {0, dim};
      }
      one_tensor.emplace_back(range);
      axis_index++;
    }

    tensor->SetOriginShape(ge::GeShape(new_dim));
    tensor->SetShape(ge::GeShape(new_dim));
    one_thread.emplace_back(one_tensor);
  }

  void AssembleAutoThreadTensor(GeTensorDescPtr &tensor,
                                uint32_t thread_id,
                                uint32_t slice_num,
                                vector<vector<DimRange>> &one_thread) {
    vector<DimRange> one_tensor;
    uint32_t axis_index = 0;
    vector<int64_t> new_dim = tensor->GetShape().GetDims();
    for (int64_t dim : tensor->GetShape().GetDims()) {
      DimRange range = {0, dim};
      if (axis_index == 0) {
        // only the highest dimension needs to be sliced.
        int64_t dim_non_tail = dim / slice_num;
        int64_t low, high;
        if (thread_id == slice_num - 1) {
          low = thread_id * dim_non_tail;
          high = dim;
        } else {
          low = thread_id * dim_non_tail;
          high = (thread_id + 1) * dim_non_tail;
        }
        range = {low, high};
        new_dim[axis_index] = high - low;
      } else {
        range = {0, dim};
      }
      one_tensor.emplace_back(range);
      axis_index++;
    }

    if (one_thread[0].size() == 0) {
      one_thread[0] = one_tensor;
    } else {
      one_thread.emplace_back(one_tensor);
    }
  }

/* index: the index of tensor;
 * axes: dimension indices of those need to be sliced. */
  Status SetManualSlicingInfo(const ge::NodePtr &node, uint32_t index, bool is_input,
                              uint32_t slice_num, uint32_t slice_index, const vector<uint32_t> &axes) {
    auto op_desc = node->GetOpDesc();
    ThreadSliceMapPtr slice_info_ptr;
    slice_info_ptr = op_desc->TryGetExtAttr(kAttrSgtStructInfo, slice_info_ptr);
    if (slice_info_ptr == nullptr) {
      slice_info_ptr = std::make_shared<ThreadSliceMap>();
    }
    slice_info_ptr->thread_mode = 0;
    slice_info_ptr->thread_scope_id = 0;
    slice_info_ptr->is_first_node_in_topo_order = false;
    slice_info_ptr->node_num_in_thread_scope = 1;
    slice_info_ptr->slice_instance_num = slice_num;

    if (slice_index == slice_num - 1) {
      slice_info_ptr->thread_id = {slice_num - 1};
    } else {
      slice_info_ptr->thread_id = {slice_index};
    }

    /* push all tensors into tensor slice */
    /* In manual slicing, there is only one thread in sgt info. */
    vector<vector<DimRange>> one_thread;
    if (is_input) {
      if (slice_info_ptr->input_tensor_slice.empty()) {
        slice_info_ptr->input_tensor_slice.emplace_back(one_thread);
      }
    } else {
      if (slice_info_ptr->output_tensor_slice.empty()) {
        slice_info_ptr->output_tensor_slice.emplace_back(one_thread);
      }
    }

    vector<vector<vector<DimRange>>> *slice = &slice_info_ptr->output_tensor_slice;
    if (is_input) {
      slice = &slice_info_ptr->input_tensor_slice;
    }

    size_t all_tensors_size = is_input ? op_desc->GetAllInputsSize() : op_desc->GetAllOutputsDesc().size();
    for (uint32_t i = 0; i < slice_num; i++) {
      if (i != slice_index) {
        continue;
      }

      for (size_t j = 0; j < all_tensors_size; j++) {
        if (j != index) {
          continue;
        }
        GeTensorDescPtr tensor;
        if (is_input) {
          tensor = op_desc->MutableInputDesc(j);
        } else {
          tensor = op_desc->MutableOutputDesc(j);
        }
        FFTS_LOGD("before shape of %s is %s", node->GetName().c_str(),
                fe::StringUtils::IntegerVecToString(tensor->GetShape().GetDims()).c_str());
        AssembleTensor(tensor, axes, slice_info_ptr->thread_id, slice_num, (*slice)[0]);
        FFTS_LOGD("after shape of %s is %s", node->GetName().c_str(),
                fe::StringUtils::IntegerVecToString(tensor->GetShape().GetDims()).c_str());
      }
    }
    op_desc->SetExtAttr(kAttrSgtStructInfo, slice_info_ptr);
    return ffts::SUCCESS;
  }

  Status SetAutoThreadSlicingInfo(const ge::NodePtr &node, uint32_t index, bool is_input, uint32_t slice_num) {
    auto op_desc = node->GetOpDesc();
    ThreadSliceMapPtr slice_info_ptr;
    slice_info_ptr = op_desc->TryGetExtAttr(kAttrSgtStructInfo, slice_info_ptr);
    if (slice_info_ptr == nullptr) {
      slice_info_ptr = std::make_shared<ThreadSliceMap>();
    }
    slice_info_ptr->thread_mode = 1;
    slice_info_ptr->thread_scope_id = 0;
    slice_info_ptr->is_first_node_in_topo_order = false;
    slice_info_ptr->node_num_in_thread_scope = 1;
    slice_info_ptr->slice_instance_num = slice_num;
    slice_info_ptr->parallel_window_size = slice_num;

    slice_info_ptr->thread_id = 0;

    /* push all tensors into tensor slice */
    /* In manual slicing, there is only one thread in sgt info. */
    vector<DimRange> one_thread;
    if (is_input) {
      if (slice_info_ptr->input_tensor_slice.empty()) {
        vector<vector<DimRange>> vv_one_thread;
        vv_one_thread.push_back(one_thread);
        for (size_t i = 0; i < slice_num; i++) {
          slice_info_ptr->input_tensor_slice.emplace_back(vv_one_thread);
        }
      } else {
        for (size_t i = 0; i < slice_num; i++) {
          slice_info_ptr->input_tensor_slice[i].push_back(one_thread);
        }
      }
    } else {
      if (slice_info_ptr->output_tensor_slice.empty()) {
        vector<vector<DimRange>> vv_one_thread;
        vv_one_thread.push_back(one_thread);
        for (size_t i = 0; i < slice_num; i++) {
          slice_info_ptr->output_tensor_slice.emplace_back(vv_one_thread);
        }
      }
      else {
        for(size_t i = 0; i < slice_num; i++) {
          slice_info_ptr->output_tensor_slice[i].emplace_back(one_thread);
        }
      }
    }

    vector<vector<vector<DimRange>>> *slice = &slice_info_ptr->output_tensor_slice;
    if (is_input) {
      slice = &slice_info_ptr->input_tensor_slice;
    }

    size_t all_tensors_size = op_desc->GetAllInputsSize();
    for (uint32_t i = 0; i < slice_num; i++) {
      // if (i != slice_index) {
      //   continue;
      // }

      for (size_t j = 0; j < all_tensors_size; j++) {
        if (j != index) {
          continue;
        }
        GeTensorDescPtr tensor;
        if (is_input) {
          tensor = op_desc->MutableInputDesc(j);
        } else {
          tensor = op_desc->MutableOutputDesc(j);
        }

        FFTS_LOGD("before shape of %s is %s", node->GetName().c_str(),
                fe::StringUtils::IntegerVecToString(tensor->GetShape().GetDims()).c_str());
        AssembleAutoThreadTensor(tensor, i, slice_num, (*slice)[i]);
        FFTS_LOGD("after shape of %s is %s", node->GetName().c_str(),
                fe::StringUtils::IntegerVecToString(tensor->GetShape().GetDims()).c_str());
      }
    }
    slice_info_ptr->parallel_window_size = 4;
    op_desc->SetExtAttr(kAttrSgtStructInfo, slice_info_ptr);
    return ffts::SUCCESS;
  }

  void GenerateABAutoThreadingTaskDef(domi::FftsPlusTaskDef *ffts_plus_task_def) {
    size_t slice_num = 4;
    // add in_label context
    auto inlabel_context = ffts_plus_task_def->add_ffts_plus_ctx();
    inlabel_context->set_context_type(RT_CTX_TYPE_LABEL);
    auto in_label = inlabel_context->mutable_label_ctx();
    in_label->set_pred_cnt(0);
    for (uint32_t i = 0; i < slice_num; i++) {
      in_label->add_successor_list(i);
    }

    // add at_start context
    for (uint32_t i = 0; i < slice_num; i++) {
      auto at_start_context = ffts_plus_task_def->add_ffts_plus_ctx();  // at_start's context
      at_start_context->set_context_type(RT_CTX_TYPE_AT_START);
      auto at_start = at_start_context->mutable_at_start_ctx();
      at_start->set_pred_cnt(1);
      at_start->set_pred_cnt_init(1);
      at_start->set_thread_id(0);
      at_start->add_successor_list(i + 1 + slice_num);
    }

    for (uint32_t i = 0; i < slice_num; i++) {
      auto a_context = ffts_plus_task_def->add_ffts_plus_ctx();  // a's context
      a_context->set_context_type(RT_CTX_TYPE_AIV);
      auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
      a_aic_aiv->set_successor_num(1);
      a_aic_aiv->add_successor_list(i + 1 + slice_num * 2);
    }

    for (uint32_t i = 0; i < slice_num; i++) {
      auto b_context = ffts_plus_task_def->add_ffts_plus_ctx();  // b's context
      b_context->set_context_type(RT_CTX_TYPE_AIV);
      auto b_aic_aiv = b_context->mutable_aic_aiv_ctx();
      b_aic_aiv->set_pred_cnt(1);
      b_aic_aiv->set_pred_cnt_init(1);
      b_aic_aiv->set_successor_num(1);
      b_aic_aiv->add_successor_list(13 + i);
    }

    for (uint32_t i = 0; i < slice_num; i++) {
      auto at_end_context = ffts_plus_task_def->add_ffts_plus_ctx();  // at_end's context
      at_end_context->set_context_type(RT_CTX_TYPE_AT_END);
      auto at_end = at_end_context->mutable_at_end_ctx();
      at_end->set_pred_cnt(1);
      at_end->set_pred_cnt_init(1);
      at_end->add_succ_at_start_slot(i + 1);
      at_end->add_succ_out_label_slot(17);
    }

    auto out_label_context = ffts_plus_task_def->add_ffts_plus_ctx();
    out_label_context->set_context_type(RT_CTX_TYPE_LABEL);
    auto out_label = out_label_context->mutable_label_ctx();
    out_label->set_pred_cnt(4);
    out_label->set_pred_cnt_init(4);
  }

  void GenerateABCAutoThreadingTaskDef(domi::FftsPlusTaskDef *ffts_plus_task_def) {
    size_t slice_num = 4;
    // add in_label context
    auto inlabel_context = ffts_plus_task_def->add_ffts_plus_ctx();
    inlabel_context->set_context_type(RT_CTX_TYPE_LABEL);
    auto in_label = inlabel_context->mutable_label_ctx();
    in_label->set_pred_cnt(0);
    for (uint32_t i = 0; i < slice_num; i++) {
      in_label->add_successor_list(i);
    }

    // add at_start context
    for (uint32_t i = 0; i < slice_num; i++) {
      auto at_start_context = ffts_plus_task_def->add_ffts_plus_ctx();  // at_start's context
      at_start_context->set_context_type(RT_CTX_TYPE_AT_START);
      auto at_start = at_start_context->mutable_at_start_ctx();
      at_start->set_pred_cnt(1);
      at_start->set_pred_cnt_init(1);
      at_start->set_thread_id(0);
      at_start->add_successor_list(i + 1 + slice_num);
    }

    for (uint32_t i = 0; i < slice_num; i++) {
      auto a_context = ffts_plus_task_def->add_ffts_plus_ctx();  // a's context
      a_context->set_context_type(RT_CTX_TYPE_AIV);
      auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
      a_aic_aiv->set_successor_num(2);
      a_aic_aiv->add_successor_list(i + 1 + slice_num * 2);
      a_aic_aiv->add_successor_list(i + 1 + slice_num * 3);
    }

    for (uint32_t i = 0; i < slice_num; i++) {
      auto b_context = ffts_plus_task_def->add_ffts_plus_ctx();  // b's context
      b_context->set_context_type(RT_CTX_TYPE_AIV);
      auto b_aic_aiv = b_context->mutable_aic_aiv_ctx();
      b_aic_aiv->set_pred_cnt(1);
      b_aic_aiv->set_pred_cnt_init(1);
      b_aic_aiv->set_successor_num(1);
      b_aic_aiv->add_successor_list(17 + i);
    }

    for (uint32_t i = 0; i < slice_num; i++) {
      auto c_context = ffts_plus_task_def->add_ffts_plus_ctx();  // c's context
      c_context->set_context_type(RT_CTX_TYPE_AIV);
      auto c_aic_aiv = c_context->mutable_aic_aiv_ctx();
      c_aic_aiv->set_pred_cnt(1);
      c_aic_aiv->set_pred_cnt_init(1);
      c_aic_aiv->set_successor_num(1);
      c_aic_aiv->add_successor_list(17 + i);
    }

    for (uint32_t i = 0; i < slice_num; i++) {
      auto at_end_context = ffts_plus_task_def->add_ffts_plus_ctx();  // at_end's context
      at_end_context->set_context_type(RT_CTX_TYPE_AT_END);
      auto at_end = at_end_context->mutable_at_end_ctx();
      at_end->set_pred_cnt(2);
      at_end->set_pred_cnt_init(2);
      at_end->add_succ_at_start_slot(i + 1);
      at_end->add_succ_out_label_slot(21);
    }

    auto out_label_context = ffts_plus_task_def->add_ffts_plus_ctx();
    out_label_context->set_context_type(RT_CTX_TYPE_LABEL);
    auto out_label = out_label_context->mutable_label_ctx();
    out_label->set_pred_cnt(4);
    out_label->set_pred_cnt_init(4);
  }

  void GenerateABCDEAutoThreadingTaskDef(domi::FftsPlusTaskDef *ffts_plus_task_def) {
    size_t slice_num = 4;
    // add in_label context
    auto inlabel_context = ffts_plus_task_def->add_ffts_plus_ctx();
    inlabel_context->set_context_type(RT_CTX_TYPE_LABEL);
    auto in_label = inlabel_context->mutable_label_ctx();
    in_label->set_pred_cnt(0);
    for (uint32_t i = 0; i < slice_num; i++) {
      in_label->add_successor_list(i);
    }

    // add at_start context
    for (uint32_t i = 0; i < slice_num; i++) {
      auto at_start_context = ffts_plus_task_def->add_ffts_plus_ctx();  // at_start's context
      at_start_context->set_context_type(RT_CTX_TYPE_AT_START);
      auto at_start = at_start_context->mutable_at_start_ctx();
      at_start->set_pred_cnt(1);
      at_start->set_pred_cnt_init(1);
      at_start->set_thread_id(0);
      at_start->add_successor_list(i + 1 + slice_num);
    }

    for (uint32_t i = 0; i < slice_num; i++) {
      auto a_context = ffts_plus_task_def->add_ffts_plus_ctx();  // a's context
      a_context->set_context_type(RT_CTX_TYPE_AIV);
      auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
      a_aic_aiv->set_successor_num(1);
      a_aic_aiv->add_successor_list(i + 1 + slice_num * 2);
    }

    for (uint32_t i = 0; i < slice_num; i++) {
      auto b_context = ffts_plus_task_def->add_ffts_plus_ctx();  // b's context
      b_context->set_context_type(RT_CTX_TYPE_AIV);
      auto b_aic_aiv = b_context->mutable_aic_aiv_ctx();
      b_aic_aiv->set_pred_cnt(1);
      b_aic_aiv->set_pred_cnt_init(1);
      b_aic_aiv->set_successor_num(1);
      b_aic_aiv->add_successor_list(i + 1 + slice_num * 3);
    }

    for (uint32_t i = 0; i < slice_num; i++) {
      auto c_context = ffts_plus_task_def->add_ffts_plus_ctx();  // c's context
      c_context->set_context_type(RT_CTX_TYPE_AIV);
      auto c_aic_aiv = c_context->mutable_aic_aiv_ctx();
      c_aic_aiv->set_pred_cnt(1);
      c_aic_aiv->set_pred_cnt_init(1);
      c_aic_aiv->set_successor_num(1);
      c_aic_aiv->add_successor_list(i + 1 + slice_num * 4);
    }

    for (uint32_t i = 0; i < slice_num; i++) {
      auto d_context = ffts_plus_task_def->add_ffts_plus_ctx();  // d's context
      d_context->set_context_type(RT_CTX_TYPE_AIV);
      auto d_aic_aiv = d_context->mutable_aic_aiv_ctx();
      d_aic_aiv->set_pred_cnt(1);
      d_aic_aiv->set_pred_cnt_init(1);
      d_aic_aiv->set_successor_num(1);
      d_aic_aiv->add_successor_list(i + 1 + slice_num * 5);
    }

    for (uint32_t i = 0; i < slice_num; i++) {
      auto e_context = ffts_plus_task_def->add_ffts_plus_ctx();  // d's context
      e_context->set_context_type(RT_CTX_TYPE_AIV);
      auto e_aic_aiv = e_context->mutable_aic_aiv_ctx();
      e_aic_aiv->set_pred_cnt(1);
      e_aic_aiv->set_pred_cnt_init(1);
      e_aic_aiv->set_successor_num(1);
      e_aic_aiv->add_successor_list(i + 1 + slice_num * 6);
    }

    for (uint32_t i = 0; i < slice_num; i++) {
      auto at_end_context = ffts_plus_task_def->add_ffts_plus_ctx();  // at_end's context
      at_end_context->set_context_type(RT_CTX_TYPE_AT_END);
      auto at_end = at_end_context->mutable_at_end_ctx();
      at_end->set_pred_cnt(2);
      at_end->set_pred_cnt_init(2);
      at_end->add_succ_at_start_slot(i + 1);
      at_end->add_succ_out_label_slot(29);
    }

    auto out_label_context = ffts_plus_task_def->add_ffts_plus_ctx();
    out_label_context->set_context_type(RT_CTX_TYPE_LABEL);
    auto out_label = out_label_context->mutable_label_ctx();
    out_label->set_pred_cnt(4);
    out_label->set_pred_cnt_init(4);
  }

  void CreateGraph01(ge::NodePtr &b, ComputeGraphPtr &graph) {
    graph = std::make_shared<ComputeGraph>("test");
    GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                        GeShape({288, 8, 24, 33}));
    test.AddOpDesc("a", "A", 1, 1)
        .AddOpDesc("b", "B", 1, 1)
        .SetInput("b:0", "a:0");

    test.GetNodeByName("b", b);
    ge::AttrUtils::SetInt(b->GetOpDesc(), kContextId, 1);
    ge::AttrUtils::SetInt(b->GetOpDesc(), kPrefetchEnableBm, 0x0001);
    vector<int64_t> input_addrs = {5000};
    ge::AttrUtils::SetListInt(b->GetOpDesc(), "input_addrs", input_addrs);

    NodePtr a;
    test.GetNodeByName("a", a);
    ge::AttrUtils::SetInt(a->GetOpDesc(), kContextId, 0);
    vector<int64_t> output_addrs = {5000};
    ge::AttrUtils::SetListInt(a->GetOpDesc(), "output_addrs", output_addrs);

    /* slice a and b */
    int32_t slice_num = 4;
    vector<uint32_t> axes = {0};

    SetManualSlicingInfo(a, 0 /* index */, true /* is_input */, slice_num, 1 /* slice_index */, axes);
    SetManualSlicingInfo(a, 0 /* index */, false /* is_input */, slice_num, 1 /* slice_index */, axes);
    SetManualSlicingInfo(b, 0 /* index */, true /* is_input */, slice_num, 1 /* slice_index */, axes);
    SetManualSlicingInfo(b, 0 /* index */, false /* is_input */, slice_num, 1 /* slice_index */, axes);
  }

    void CreateAutoThreadingGraph01(ge::NodePtr &b, ComputeGraphPtr &graph) {
    graph = std::make_shared<ComputeGraph>("test");
    GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                        GeShape({288, 8, 24, 33}));
    test.AddOpDesc("a", "A", 1, 1)
        .AddOpDesc("b", "B", 1, 1)
        .SetInput("b:0", "a:0");

    test.GetNodeByName("b", b);
    vector<uint32_t> context_id_list = {9, 10, 11, 12};
    ge::AttrUtils::SetListInt(b->GetOpDesc(), "_context_id_list", context_id_list);
    ge::AttrUtils::SetInt(b->GetOpDesc(), kPrefetchEnableBm, 0x0001);
    vector<int64_t> input_addrs = {5000};
    ge::AttrUtils::SetListInt(b->GetOpDesc(), "input_addrs", input_addrs);

    NodePtr a;
    test.GetNodeByName("a", a);
    context_id_list = {5, 6, 7, 8};
    ge::AttrUtils::SetListInt(a->GetOpDesc(), "_context_id_list", context_id_list);
    vector<int64_t> output_addrs = {5000};
    ge::AttrUtils::SetListInt(a->GetOpDesc(), "output_addrs", output_addrs);

    /* slice a and b */
    int32_t slice_num = 4;
    vector<uint32_t> axes = {0};

    SetAutoThreadSlicingInfo(a, 0 /* index */, true /* is_input */, slice_num);
    SetAutoThreadSlicingInfo(a, 0 /* index */, false /* is_input */, slice_num);
    SetAutoThreadSlicingInfo(b, 0 /* index */, true /* is_input */, slice_num);
    SetAutoThreadSlicingInfo(b, 0 /* index */, false /* is_input */, slice_num);
  }
  void CreateGraphInvalidFusion(NodePtr &slice0, ComputeGraphPtr &graph, vector<uint32_t> &axes) {
    graph = std::make_shared<ComputeGraph>("test");
    GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                          GeShape({288, 8, 24, 33}));
    test.AddOpDesc("slice0", "A", 1, 2)
        .AddOpDesc("slice1", "B", 1, 1)
        .AddOpDesc("relu", "relu", 1, 1)
        .AddOpDesc("py", "PhonyConcat", 2, 1)
        .AddOpDesc("c", "C", 1, 1)
        .SetInput("py:0", "slice0:0")
        .SetInput("relu:0", "slice0:1")
        .SetInput("py:1", "slice1:0")
        .SetInput("c:0", "py:0")
        .SetInput("NetOutput:0", "c:0");
    test.GetNodeByName("slice0", slice0);
    ge::AttrUtils::SetInt(slice0->GetOpDesc(), kContextId, 0);
    ge::AttrUtils::SetInt(slice0->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    vector<int64_t> output_addrs = {5000, 6000};
    ge::AttrUtils::SetListInt(slice0->GetOpDesc(), "output_addrs", output_addrs);
    slice0->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);

    NodePtr slice1;
    test.GetNodeByName("slice1", slice1);
    ge::AttrUtils::SetInt(slice1->GetOpDesc(), kContextId, 1);
    ge::AttrUtils::SetInt(slice1->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    slice1->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);

    NodePtr c;
    test.GetNodeByName("c", c);
    ge::AttrUtils::SetInt(c->GetOpDesc(), kContextId, 2);
    ge::AttrUtils::SetInt(c->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    c->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);

    NodePtr relu;
    test.GetNodeByName("relu", relu);
    ge::AttrUtils::SetInt(relu->GetOpDesc(), kContextId, 3);
    ge::AttrUtils::SetInt(relu->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    relu->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);

    NodePtr py;
    test.GetNodeByName("py", py);
    ge::AttrUtils::SetInt(py->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    (void)ge::AttrUtils::SetBool(py->GetOpDesc(), ge::ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
    py->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);
    /* slice a and b */
    int32_t slice_num = 4;
    SetManualSlicingInfo(slice0, 0 /* index */, false /* is_input */, slice_num,
                         1 /* slice_index */, axes);
    SetManualSlicingInfo(slice0, 1 /* index */, false /* is_input */, slice_num,
                         1 /* slice_index */, axes);
    SetManualSlicingInfo(slice1, 0 /* index */, false /* is_input */, slice_num,
                         1 /* slice_index */, axes);
  }
  void CreateDynamicGraph01(ge::NodePtr &b, ComputeGraphPtr &graph, ge::NodePtr &a) {
    graph = std::make_shared<ComputeGraph>("test");
    GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                        GeShape({-1, 8, 24, 33}));
    test.AddOpDesc("a", "A", 1, 1)
        .AddOpDesc("b", "B", 1, 1)
        .SetInput("b:0", "a:0");

    test.GetNodeByName("b", b);
    vector<uint32_t> context_id_list = {9, 10, 11, 12};
    ge::AttrUtils::SetListInt(b->GetOpDesc(), "_context_id_list", context_id_list);
    ge::AttrUtils::SetInt(b->GetOpDesc(), kPrefetchEnableBm, 0x0001);
    vector<int64_t> input_addrs = {5000};
    ge::AttrUtils::SetListInt(b->GetOpDesc(), "input_addrs", input_addrs);

    test.GetNodeByName("a", a);
    ge::AttrUtils::SetInt(a->GetOpDesc(), kInvalidateBm, 0x0001);
    context_id_list = {5, 6, 7, 8};
    ge::AttrUtils::SetListInt(a->GetOpDesc(), "_context_id_list", context_id_list);
    vector<int64_t> output_addrs = {5000};
    ge::AttrUtils::SetListInt(a->GetOpDesc(), "output_addrs", output_addrs);

    /* slice a and b */
    int32_t slice_num = 4;
    SetAutoThreadSlicingInfo(a, 0 /* index */, true /* is_input */, slice_num);
    SetAutoThreadSlicingInfo(a, 0 /* index */, false /* is_input */, slice_num);
    SetAutoThreadSlicingInfo(b, 0 /* index */, true /* is_input */, slice_num);
    SetAutoThreadSlicingInfo(b, 0 /* index */, false /* is_input */, slice_num);
  }

  /* dims[0] is the highest dimension. */
  void CreateGraph02_x(ge::NodePtr &b, ComputeGraphPtr &graph, vector<uint32_t> &axes) {
    graph = std::make_shared<ComputeGraph>("test");
    GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                          GeShape({288, 8, 24, 33}));
    test.AddOpDesc("a", "A", 1, 1)
        .AddOpDesc("b", "B", 1, 1)
        .SetInput("b:0", "a:0");

    test.GetNodeByName("b", b);
    ge::AttrUtils::SetInt(b->GetOpDesc(), kContextId, 1);
    ge::AttrUtils::SetInt(b->GetOpDesc(), kPrefetchEnableBm, 0x0001);
    vector<int64_t> input_addrs = {5000};
    ge::AttrUtils::SetListInt(b->GetOpDesc(), "input_addrs", input_addrs);

    NodePtr a;
    test.GetNodeByName("a", a);
    ge::AttrUtils::SetInt(a->GetOpDesc(), kContextId, 0);
    vector<int64_t> output_addrs = {20000};
    ge::AttrUtils::SetListInt(a->GetOpDesc(), "output_addrs", output_addrs);

    int32_t slice_num = 4;
    SetManualSlicingInfo(b, 0 /* index */, true /* is_input */, slice_num,
                         1 /* slice_index */, axes);
    SetManualSlicingInfo(b, 0 /* index */, false /* is_input */, slice_num,
                         1 /* slice_index */, axes);
  }

  void CreateGraph02_2_1(ge::NodePtr &b, ComputeGraphPtr &graph) {
    graph = std::make_shared<ComputeGraph>("test");
    GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                          GeShape({288, 8, 24, 33}));
    test.AddOpDesc("a", "A", 1, 1)
        .AddOpDesc("b", "B", 3, 1)
        .AddOpDesc("c", "D", 1, 1)
        .AddOpDesc("c", "D", 1, 1)
        .SetInput("b:0", "a:0")
        .SetInput("b:1", "c:0")
        .SetInput("b:2", "d:0");

    test.GetNodeByName("b", b);
    ge::AttrUtils::SetInt(b->GetOpDesc(), kContextId, 0);
    ge::AttrUtils::SetInt(b->GetOpDesc(), kPrefetchEnableBm, 0x0007);
    vector<int64_t> input_addrs = {5000, 6000, 7000};
    ge::AttrUtils::SetListInt(b->GetOpDesc(), "input_addrs", input_addrs);

    NodePtr a;
    test.GetNodeByName("a", a);
    vector<int64_t> output_addrs = {20000};
    ge::AttrUtils::SetListInt(a->GetOpDesc(), "output_addrs", output_addrs);

    NodePtr c;
    test.GetNodeByName("c", c);
    output_addrs = {30000};
    ge::AttrUtils::SetListInt(c->GetOpDesc(), "output_addrs", output_addrs);

    NodePtr d;
    test.GetNodeByName("d", d);
    output_addrs = {40000};
    ge::AttrUtils::SetListInt(d->GetOpDesc(), "output_addrs", output_addrs);

    int32_t slice_num = 4;
    vector<uint32_t> axes1 = {0};
    vector<uint32_t> axes2 = {2, 3};
    SetManualSlicingInfo(b, 0 /* index */, true /* is_input */, slice_num,
                         1 /* slice_index */, axes1);
    SetManualSlicingInfo(b, 0 /* index */, false /* is_input */, slice_num,
                         1 /* slice_index */, axes1);

    SetManualSlicingInfo(b, 1 /* index */, true /* is_input */, slice_num,
                         2 /* slice_index */, axes1);

    SetManualSlicingInfo(b, 2 /* index */, true /* is_input */, slice_num,
                         3 /* slice_index */, axes2);
  }

  /* dims[0] is the highest dimension. */
  void CreateGraph02_4_1(ge::NodePtr &b, ComputeGraphPtr &graph, vector<uint32_t> &axes) {
    graph = std::make_shared<ComputeGraph>("test");
    GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                          GeShape({4, 8, 24, 33}));
    test.AddOpDesc("a", "A", 1, 1)
        .AddOpDesc("b", "B", 1, 1)
        .SetInput("b:0", "a:0");

    test.GetNodeByName("b", b);
    ge::AttrUtils::SetInt(b->GetOpDesc(), kContextId, 1);
    ge::AttrUtils::SetInt(b->GetOpDesc(), kPrefetchEnableBm, 0x0001);
    vector<int64_t> input_addrs = {5000};
    ge::AttrUtils::SetListInt(b->GetOpDesc(), "input_addrs", input_addrs);

    NodePtr a;
    test.GetNodeByName("a", a);
    ge::AttrUtils::SetInt(a->GetOpDesc(), kContextId, 0);
    vector<int64_t> output_addrs = {20000};
    ge::AttrUtils::SetListInt(a->GetOpDesc(), "output_addrs", output_addrs);

    int32_t slice_num = 4;
    SetManualSlicingInfo(b, 0 /* index */, true /* is_input */, slice_num,
                         1 /* slice_index */, axes);
    SetManualSlicingInfo(b, 0 /* index */, false /* is_input */, slice_num,
                         1 /* slice_index */, axes);
  }

  /* slice 288 and 24*/
  void CreateGraph03_x(ge::NodePtr &b, ComputeGraphPtr &graph) {
    graph = std::make_shared<ComputeGraph>("test");
    GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                          GeShape({288, 8, 24, 33}));
    test.AddOpDesc("a", "A", 1, 1)
        .AddOpDesc("b", "B", 1, 1)
        .SetInput("b:0", "a:0");

    test.GetNodeByName("b", b);
    ge::AttrUtils::SetInt(b->GetOpDesc(), kContextId, 1);
    ge::AttrUtils::SetInt(b->GetOpDesc(), kPrefetchEnableBm, 0x0001);
    vector<int64_t> input_addrs = {5000};
    ge::AttrUtils::SetListInt(b->GetOpDesc(), "input_addrs", input_addrs);

    NodePtr a;
    test.GetNodeByName("a", a);
    ge::AttrUtils::SetInt(a->GetOpDesc(), kContextId, 0);
    vector<int64_t> output_addrs = {20000};
    ge::AttrUtils::SetListInt(a->GetOpDesc(), "output_addrs", output_addrs);

    int32_t slice_num = 4;
    vector<uint32_t> axes = {0, 2};
    SetManualSlicingInfo(b, 0 /* index */, true /* is_input */, slice_num,
                         2 /* slice_index */, axes);
    SetManualSlicingInfo(b, 0 /* index */, false /* is_input */, slice_num,
                         2 /* slice_index */, axes);
  }

  /* a will do the writeback operation */
  void CreateGraph04_x(ge::NodePtr &a, ComputeGraphPtr &graph, vector<uint32_t> &axes) {
    graph = std::make_shared<ComputeGraph>("test");
    GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                          GeShape({288, 8, 24, 33}));
    test.AddOpDesc("a", "A", 1, 1)
        .AddOpDesc("b", "B", 1, 1)
        .AddOpDesc("c", "C", 1, 1)
        .SetInput("b:0", "a:0")
        .SetInput("c:0", "a:0");

    test.GetNodeByName("a", a);
    ge::AttrUtils::SetInt(a->GetOpDesc(), kContextId, 0);
    ge::AttrUtils::SetInt(a->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    vector<int64_t> output_addrs = {30000};
    ge::AttrUtils::SetListInt(a->GetOpDesc(), "output_addrs", output_addrs);
    a->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);

    NodePtr b;
    test.GetNodeByName("b", b);
    vector<int64_t> input_addrs = {5000};
    ge::AttrUtils::SetInt(b->GetOpDesc(), kContextId, 1);
    ge::AttrUtils::SetInt(b->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    ge::AttrUtils::SetListInt(b->GetOpDesc(), "input_addrs", input_addrs);
    b->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);

    NodePtr c;
    test.GetNodeByName("c", c);
    ge::AttrUtils::SetListInt(c->GetOpDesc(), "input_addrs", input_addrs);
    c->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);
    /* slice a and b */
    int32_t slice_num = 4;
    SetManualSlicingInfo(a, 0 /* index */, true /* is_input */, slice_num,
                         1 /* slice_index */, axes);
    SetManualSlicingInfo(a, 0 /* index */, false /* is_input */, slice_num,
                         1 /* slice_index */, axes);

    SetManualSlicingInfo(b, 0 /* index */, true /* is_input */, slice_num,
                         1 /* slice_index */, axes);
    SetManualSlicingInfo(b, 0 /* index */, false /* is_input */, slice_num,
                         1 /* slice_index */, axes);
  }

  void CreateAutoThreadingGraph04_x(ge::NodePtr &a, ComputeGraphPtr &graph) {
    graph = std::make_shared<ComputeGraph>("test");
    GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                          GeShape({288, 8, 24, 33}));
    test.AddOpDesc("a", "A", 1, 1)
        .AddOpDesc("b", "B", 1, 1)
        .AddOpDesc("c", "C", 1, 1)
        .SetInput("b:0", "a:0")
        .SetInput("c:0", "a:0");

    test.GetNodeByName("a", a);
    vector<uint32_t> context_id_list = {5, 6, 7, 8};
    ge::AttrUtils::SetListInt(a->GetOpDesc(), "_context_id_list", context_id_list);
    ge::AttrUtils::SetInt(a->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    vector<int64_t> output_addrs = {20000};
    ge::AttrUtils::SetListInt(a->GetOpDesc(), "output_addrs", output_addrs);
    a->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);

    NodePtr b;
    test.GetNodeByName("b", b);
    vector<int64_t> input_addrs = {5000};
    context_id_list = {9, 10, 11, 12};
    ge::AttrUtils::SetListInt(b->GetOpDesc(), "_context_id_list", context_id_list);
    ge::AttrUtils::SetInt(b->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    ge::AttrUtils::SetListInt(b->GetOpDesc(), "input_addrs", input_addrs);
    b->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);

    NodePtr c;
    test.GetNodeByName("c", c);
    ge::AttrUtils::SetListInt(c->GetOpDesc(), "input_addrs", input_addrs);
    c->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);
    /* slice a and b */
    int32_t slice_num = 4;
    SetAutoThreadSlicingInfo(a, 0 /* index */, true /* is_input */, slice_num);
    SetAutoThreadSlicingInfo(a, 0 /* index */, false /* is_input */, slice_num);
    SetAutoThreadSlicingInfo(b, 0 /* index */, true /* is_input */, slice_num);
    SetAutoThreadSlicingInfo(b, 0 /* index */, false /* is_input */, slice_num);
  }

  /* a will do the writeback or invalidate operation */
  void CreateGraph04_x_1(ge::NodePtr &a, ComputeGraphPtr &graph, vector<uint32_t> &axes, int scope_of_c) {
    graph = std::make_shared<ComputeGraph>("test");
    GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                          GeShape({288, 8, 24, 33}));
    test.AddOpDesc("a", "A", 1, 1)
        .AddOpDesc("b", "B", 1, 1)
        .AddOpDesc("pc", "PhonyConcat", 1, 1)
        .AddOpDesc("c", "C", 1, 1)
        .SetInput("b:0", "a:0")
        .SetInput("pc:0", "a:0")
        .SetInput("c:0", "pc:0");

    test.GetNodeByName("a", a);
    ge::AttrUtils::SetInt(a->GetOpDesc(), kContextId, 0);
    ge::AttrUtils::SetInt(a->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    vector<int64_t> output_addrs = {20000};
    ge::AttrUtils::SetListInt(a->GetOpDesc(), "output_addrs", output_addrs);
    a->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);

    NodePtr b;
    test.GetNodeByName("b", b);
    vector<int64_t> input_addrs = {5000};
    ge::AttrUtils::SetInt(b->GetOpDesc(), kContextId, 1);
    ge::AttrUtils::SetInt(b->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    ge::AttrUtils::SetListInt(b->GetOpDesc(), "input_addrs", input_addrs);
    b->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);

    NodePtr pc;
    test.GetNodeByName("pc", pc);
    input_addrs = {30000};
    ge::AttrUtils::SetInt(pc->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    ge::AttrUtils::SetListInt(pc->GetOpDesc(), "input_addrs", input_addrs);
    pc->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);

    NodePtr c;
    test.GetNodeByName("c", c);
    if (scope_of_c == 1) {
      ge::AttrUtils::SetInt(c->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, scope_of_c);
    } else {
      c->GetOpDesc()->SetType("NetOutput");
    }
    if (scope_of_c == 1) {
      /* same thread scope as node and b. */
      ge::AttrUtils::SetInt(c->GetOpDesc(), kContextId, 2);
    }
    ge::AttrUtils::SetListInt(c->GetOpDesc(), "input_addrs", input_addrs);
    c->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);
    /* slice a and b */
    int32_t slice_num = 4;
    SetManualSlicingInfo(a, 0 /* index */, true /* is_input */, slice_num,
                         1 /* slice_index */, axes);
    SetManualSlicingInfo(a, 0 /* index */, false /* is_input */, slice_num,
                         1 /* slice_index */, axes);

    SetManualSlicingInfo(b, 0 /* index */, true /* is_input */, slice_num,
                         1 /* slice_index */, axes);
    SetManualSlicingInfo(b, 0 /* index */, false /* is_input */, slice_num,
                         1 /* slice_index */, axes);
  }

  /* a will do the invalidate operation */
  void CreateGraph05_x(ge::NodePtr &a, ComputeGraphPtr &graph, vector<uint32_t> &axes) {
    graph = std::make_shared<ComputeGraph>("test");
    GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                          GeShape({288, 8, 24, 33}));
    test.AddOpDesc("a", "A", 1, 1)
        .AddOpDesc("b", "B", 1, 1)
        .AddOpDesc("c", "C", 1, 1)
        .SetInput("b:0", "a:0")
        .SetInput("c:0", "a:0");

    test.GetNodeByName("a", a);
    ge::AttrUtils::SetInt(a->GetOpDesc(), kContextId, 0);
    ge::AttrUtils::SetInt(a->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    vector<int64_t> output_addrs = {20000};
    ge::AttrUtils::SetListInt(a->GetOpDesc(), "output_addrs", output_addrs);
    a->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);

    NodePtr b;
    test.GetNodeByName("b", b);
    vector<int64_t> input_addrs = {5000};
    ge::AttrUtils::SetInt(b->GetOpDesc(), kContextId, 1);
    ge::AttrUtils::SetInt(b->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    ge::AttrUtils::SetListInt(b->GetOpDesc(), "input_addrs", input_addrs);
    b->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);

    NodePtr c;
    test.GetNodeByName("c", c);
    ge::AttrUtils::SetInt(c->GetOpDesc(), kContextId, 2);
    ge::AttrUtils::SetInt(c->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    ge::AttrUtils::SetListInt(c->GetOpDesc(), "input_addrs", input_addrs);
    c->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);
    /* slice a and b */
    int32_t slice_num = 4;
    SetManualSlicingInfo(a, 0 /* index */, true /* is_input */, slice_num,
                         1 /* slice_index */, axes);
    SetManualSlicingInfo(a, 0 /* index */, false /* is_input */, slice_num,
                         1 /* slice_index */, axes);
  }

    /* a will do the invalidate operation */
  void CreateAutoThreadGraph05_x(ge::NodePtr &a, ComputeGraphPtr &graph, vector<uint32_t> &axes) {
    graph = std::make_shared<ComputeGraph>("test");
    GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                          GeShape({288, 8, 24, 33}));
    test.AddOpDesc("a", "A", 1, 1)
        .AddOpDesc("b", "B", 1, 1)
        .AddOpDesc("c", "C", 1, 1)
        .SetInput("b:0", "a:0")
        .SetInput("c:0", "a:0");

    test.GetNodeByName("a", a);
    vector<uint32_t> context_id_list = {5, 6, 7, 8};
    ge::AttrUtils::SetListInt(a->GetOpDesc(), "_context_id_list", context_id_list);
    ge::AttrUtils::SetInt(a->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    vector<int64_t> output_addrs = {20000};
    ge::AttrUtils::SetListInt(a->GetOpDesc(), "output_addrs", output_addrs);
    a->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);

    NodePtr b;
    test.GetNodeByName("b", b);
    vector<int64_t> input_addrs = {5000};
    context_id_list = {9, 10, 11, 12};
    ge::AttrUtils::SetListInt(b->GetOpDesc(), "_context_id_list", context_id_list);
    ge::AttrUtils::SetInt(b->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    ge::AttrUtils::SetListInt(b->GetOpDesc(), "input_addrs", input_addrs);
    b->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);

    NodePtr c;
    test.GetNodeByName("c", c);
    context_id_list = {13, 14, 15, 16};
    ge::AttrUtils::SetListInt(c->GetOpDesc(), "_context_id_list", context_id_list);
    ge::AttrUtils::SetInt(c->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    ge::AttrUtils::SetListInt(c->GetOpDesc(), "input_addrs", input_addrs);
    c->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);
    /* slice a and b */
    int32_t slice_num = 4;
    SetAutoThreadSlicingInfo(a, 0 /* index */, true /* is_input */, slice_num);
    SetAutoThreadSlicingInfo(a, 0 /* index */, false /* is_input */, slice_num);
  }

  void CreateDynamicGraph05_x(ge::NodePtr &a, ComputeGraphPtr &graph, vector<uint32_t> &axes) {
    graph = std::make_shared<ComputeGraph>("test");
    GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                          GeShape({-1, 8, 24, 33}));
    test.AddOpDesc("a", "A", 1, 1)
        .AddOpDesc("b", "B", 1, 1)
        .AddOpDesc("c", "C", 1, 1)
        .SetInput("b:0", "a:0")
        .SetInput("c:0", "a:0");

    test.GetNodeByName("a", a);
    vector<uint32_t> context_id_list = {5, 6, 7, 8};
    ge::AttrUtils::SetListInt(a->GetOpDesc(), "_context_id_list", context_id_list);
    ge::AttrUtils::SetInt(a->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    vector<int64_t> output_addrs = {20000};
    ge::AttrUtils::SetListInt(a->GetOpDesc(), "output_addrs", output_addrs);

    NodePtr b;
    test.GetNodeByName("b", b);
    vector<int64_t> input_addrs = {5000};
    context_id_list = {9, 10, 11, 12};
    ge::AttrUtils::SetListInt(b->GetOpDesc(), "_context_id_list", context_id_list);
    ge::AttrUtils::SetInt(b->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    ge::AttrUtils::SetListInt(b->GetOpDesc(), "input_addrs", input_addrs);

    NodePtr c;
    test.GetNodeByName("c", c);
    context_id_list = {13, 14, 15, 16};
    ge::AttrUtils::SetListInt(c->GetOpDesc(), "_context_id_list", context_id_list);
    ge::AttrUtils::SetInt(c->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    ge::AttrUtils::SetListInt(c->GetOpDesc(), "input_addrs", input_addrs);
    /* slice a and b */
    int32_t slice_num = 4;
    SetAutoThreadSlicingInfo(a, 0 /* index */, true /* is_input */, slice_num);
    SetAutoThreadSlicingInfo(a, 0 /* index */, false /* is_input */, slice_num);
  }
  void CreateGraphInvalidAndMemReuse(vector<ge::NodePtr> &nodes, ComputeGraphPtr &graph, vector<uint32_t> &axes,
                                     bool is_mem_reuse, bool is_auto_threading) {
    graph = std::make_shared<ComputeGraph>("test");
    GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                          GeShape({288, 8, 24, 33}));
    test.AddOpDesc("a", "A", 1, 1)
        .AddOpDesc("b", "B", 1, 1)
        .AddOpDesc("c", "C", 1, 1)
        .AddOpDesc("d", "D", 1, 1)
        .AddOpDesc("e", "E", 1, 1)
        .SetInput("b:0", "a:0")
        .SetInput("c:0", "b:0")
        .SetInput("d:0", "c:0")
        .SetInput("e:0", "d:0");

    NodePtr a;
    test.GetNodeByName("a", a);
    ge::AttrUtils::SetInt(a->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    vector<int64_t> output_addrs = {20000};
    ge::AttrUtils::SetListInt(a->GetOpDesc(), "output_addrs", output_addrs);
    a->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);
    nodes.emplace_back(a);

    NodePtr b;
    test.GetNodeByName("b", b);
    output_addrs = {30000};
    ge::AttrUtils::SetInt(b->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    ge::AttrUtils::SetListInt(b->GetOpDesc(), "output_addrs", output_addrs);
    b->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);
    nodes.emplace_back(b);
    NodePtr c;
    test.GetNodeByName("c", c);
    output_addrs = {40000};
    ge::AttrUtils::SetInt(c->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    ge::AttrUtils::SetListInt(c->GetOpDesc(), "output_addrs", output_addrs);
    c->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);
    nodes.emplace_back(c);

    NodePtr d;
    test.GetNodeByName("d", d);
    output_addrs = {50000};
    ge::AttrUtils::SetInt(d->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    ge::AttrUtils::SetListInt(d->GetOpDesc(), "output_addrs", output_addrs);
    d->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);
    nodes.emplace_back(d);

    NodePtr e;
    test.GetNodeByName("e", e);
    output_addrs = {60000};
    ge::AttrUtils::SetInt(e->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    ge::AttrUtils::SetListInt(e->GetOpDesc(), "output_addrs", output_addrs);
    e->GetOpDesc()->SetOpEngineName(fe::AI_CORE_NAME);
    nodes.emplace_back(e);

    if (is_mem_reuse) {
      ge::MemReuseInfo mem_reuse_info = {
        .node = e,
        .mem_type = ge::MemType::OUTPUT_MEM,
        .index = 0,
      };
      vector<ge::MemReuseInfo> mem_reuse_infos = {mem_reuse_info};
      string key = "output";
      key.append("0");
      map<string, vector<ge::MemReuseInfo>> node_mem_reuse_infos;
      node_mem_reuse_infos[key] = mem_reuse_infos;
      c->GetOpDesc()->SetExtAttr(ge::ATTR_NAME_MEMORY_REUSE_INFO, node_mem_reuse_infos);


      mem_reuse_info.node = c;
      mem_reuse_infos.insert(mem_reuse_infos.begin(), mem_reuse_info);
      node_mem_reuse_infos[key] = mem_reuse_infos;
      a->GetOpDesc()->SetExtAttr(ge::ATTR_NAME_MEMORY_REUSE_INFO, node_mem_reuse_infos);

      mem_reuse_infos.clear();

      mem_reuse_info.node = d;
      mem_reuse_infos.emplace_back(mem_reuse_info);
      node_mem_reuse_infos[key] = mem_reuse_infos;
      b->GetOpDesc()->SetExtAttr(ge::ATTR_NAME_MEMORY_REUSE_INFO, node_mem_reuse_infos);
    }

    /* slice a and b */
    int32_t slice_num = 4;
    if (is_auto_threading) {
      vector<uint32_t> context_id_list = {5, 6, 7, 8};
      ge::AttrUtils::SetListInt(a->GetOpDesc(), "_context_id_list", context_id_list);
      context_id_list = {9, 10, 11, 12};
      ge::AttrUtils::SetListInt(b->GetOpDesc(), "_context_id_list", context_id_list);
      context_id_list = {13, 14, 15, 16};
      ge::AttrUtils::SetListInt(c->GetOpDesc(), "_context_id_list", context_id_list);
      context_id_list = {17, 18, 19, 20};
      ge::AttrUtils::SetListInt(d->GetOpDesc(), "_context_id_list", context_id_list);
      context_id_list = {21, 22, 23, 24};
      ge::AttrUtils::SetListInt(e->GetOpDesc(), "_context_id_list", context_id_list);
      SetAutoThreadSlicingInfo(a, 0 /* index */, true /* is_input */, slice_num);
      SetAutoThreadSlicingInfo(a, 0 /* index */, false /* is_input */, slice_num);
      SetAutoThreadSlicingInfo(b, 0 /* index */, true /* is_input */, slice_num);
      SetAutoThreadSlicingInfo(b, 0 /* index */, false /* is_input */, slice_num);
      SetAutoThreadSlicingInfo(c, 0 /* index */, true /* is_input */, slice_num);
      SetAutoThreadSlicingInfo(c, 0 /* index */, false /* is_input */, slice_num);
      SetAutoThreadSlicingInfo(d, 0 /* index */, true /* is_input */, slice_num);
      SetAutoThreadSlicingInfo(d, 0 /* index */, false /* is_input */, slice_num);
      SetAutoThreadSlicingInfo(e, 0 /* index */, true /* is_input */, slice_num);
      SetAutoThreadSlicingInfo(e, 0 /* index */, false /* is_input */, slice_num);
    } else {
      ge::AttrUtils::SetInt(a->GetOpDesc(), kContextId, 0);
      ge::AttrUtils::SetInt(b->GetOpDesc(), kContextId, 1);
      ge::AttrUtils::SetInt(c->GetOpDesc(), kContextId, 2);
      ge::AttrUtils::SetInt(d->GetOpDesc(), kContextId, 3);
      ge::AttrUtils::SetInt(e->GetOpDesc(), kContextId, 4);
      SetManualSlicingInfo(a, 0 /* index */, true /* is_input */, slice_num,
                            1 /* slice_index */, axes);
      SetManualSlicingInfo(a, 0 /* index */, false /* is_input */, slice_num,
                            1 /* slice_index */, axes);

      SetManualSlicingInfo(b, 0 /* index */, true /* is_input */, slice_num,
                            1 /* slice_index */, axes);
      SetManualSlicingInfo(b, 0 /* index */, false /* is_input */, slice_num,
                            1 /* slice_index */, axes);

      SetManualSlicingInfo(c, 0 /* index */, true /* is_input */, slice_num,
                            1 /* slice_index */, axes);
      SetManualSlicingInfo(c, 0 /* index */, false /* is_input */, slice_num,
                            1 /* slice_index */, axes);

      SetManualSlicingInfo(d, 0 /* index */, true /* is_input */, slice_num,
                            1 /* slice_index */, axes);
      SetManualSlicingInfo(d, 0 /* index */, false /* is_input */, slice_num,
                            1 /* slice_index */, axes);

      SetManualSlicingInfo(e, 0 /* index */, true /* is_input */, slice_num,
                            1 /* slice_index */, axes);
      SetManualSlicingInfo(e, 0 /* index */, false /* is_input */, slice_num,
                            1 /* slice_index */, axes);
    }
  }
  void CreateInvalidAndMemReusePy(vector<ge::NodePtr> &nodes, ComputeGraphPtr &graph) {
    graph = std::make_shared<ComputeGraph>("test");
    GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT16,
                          GeShape({288, 8, 24, 33}));
    test.AddOpDesc("a", "A", 1, 1)
        .AddOpDesc("b", "B", 1, 1)
        .AddOpDesc("c", "C", 1, 1)
        .AddOpDesc("py", "PY", 2, 1)
        .SetInput("b:0", "a:0")
        .SetInput("py:0", "b:0")
        .SetInput("py:1", "c:0");

    NodePtr a;
    test.GetNodeByName("a", a);
    ge::AttrUtils::SetInt(a->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    nodes.emplace_back(a);

    NodePtr b;
    test.GetNodeByName("b", b);
    ge::AttrUtils::SetInt(b->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    nodes.emplace_back(b);

    NodePtr c;
    test.GetNodeByName("c", c);
    ge::AttrUtils::SetInt(c->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    nodes.emplace_back(c);

    NodePtr py;
    test.GetNodeByName("py", py);
    ge::AttrUtils::SetInt(py->GetOpDesc(), ge::ATTR_NAME_THREAD_SCOPE_ID, 1);
    nodes.emplace_back(py);

    ge::AttrUtils::SetInt(a->GetOpDesc(), kContextId, 0);
    ge::AttrUtils::SetInt(b->GetOpDesc(), kContextId, 1);
    ge::AttrUtils::SetInt(c->GetOpDesc(), kContextId, 2);
    ge::AttrUtils::SetBool(py->GetOpDesc(), "_no_padding_continuous_input", true);
  }
  PrefetchTaskBuilder prefetch_;
  OutTaskBuilder invalidate_;
  OutTaskBuilder write_back_;
  PrefetchAutoTaskBuilder prefetch_auto_;
  OutAutoTaskBuilder invalidate_auto_;
  OutAutoTaskBuilder write_back_auto_;
  PrefetchDynamicTaskBuilder prefetch_dyn_;
  OutDynamicTaskBuilder invalidate_dyn_;
  OutDynamicTaskBuilder write_back_dyn_;
};



/* A -> B(thread dim = 4), shape is continuous.
 * B is thread 1 and A's output address is 5000, B's input address is
 * 5000.
 *
 * WARNING: Burst len is 100000, which is less than the total size of b's input.
 * need to use 92 data context to do the prefetch.
 * But we only allow up to 4 prefetch data context. */
TEST_F(FftsPlusDataContextUT, prefetch_01)
{
  ge::NodePtr b;
  ComputeGraphPtr graph;
  CreateGraph01(b, graph);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx(); // a's context
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
  a_aic_aiv->set_prefetch_enable_bitmap(0);
  a_aic_aiv->set_prefetch_once_bitmap(0);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto b_aic_aiv = b_context->mutable_aic_aiv_ctx();
  b_aic_aiv->set_prefetch_enable_bitmap(0);
  b_aic_aiv->set_prefetch_once_bitmap(0);
  // prefetch_.SetBurstLen(100000);
  Status ret = prefetch_.GenManualDataCtxDef(b, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 3);
}

/* A -> B(thread dim = 4), shape is continuous.
 * B is thread 1 and A's output address is 5000, B's input address is
 * 5000.
 *
 * WARNING: Burst len is 10000000, which is larger than the total size of b's input.
 * need to use 1 data context to do the prefetch. */
TEST_F(FftsPlusDataContextUT, prefetch_01_1)
{
  ge::NodePtr b;
  ComputeGraphPtr graph;
  CreateGraph01(b, graph);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx(); // a's context
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
  a_aic_aiv->set_prefetch_enable_bitmap(0);
  a_aic_aiv->set_prefetch_once_bitmap(0);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto b_aic_aiv = b_context->mutable_aic_aiv_ctx();
  b_aic_aiv->set_prefetch_enable_bitmap(0);
  b_aic_aiv->set_prefetch_once_bitmap(0);

  prefetch_.SetBurstLen(1000000);
  Status ret = prefetch_.GenManualDataCtxDef(b, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 3);
  auto prefetch_context = ffts_plus_task_def->_impl_.ffts_plus_ctx_[2].data_ctx();
  EXPECT_EQ(prefetch_context.thread_id(), 0);
  EXPECT_EQ(prefetch_context.addr_base(), 5000);
  EXPECT_EQ(prefetch_context.addr_offset(), 0);
  EXPECT_EQ(prefetch_context.non_tail_len_inner(), 912384);
  EXPECT_EQ(prefetch_context.non_tail_num_inner(), 1);
  EXPECT_EQ(prefetch_context.non_tail_num_outter(), 1);
  EXPECT_EQ(prefetch_context.non_tail_stride_inner(), 912384);
  EXPECT_EQ(prefetch_context.non_tail_stride_outter(), 912384);

  EXPECT_EQ(prefetch_context.tail_len_inner(), 912384);
  EXPECT_EQ(prefetch_context.tail_num_inner(), 1);
  EXPECT_EQ(prefetch_context.tail_num_outter(), 1);
  EXPECT_EQ(prefetch_context.tail_stride_inner(), 912384);
  EXPECT_EQ(prefetch_context.tail_stride_outter(), 912384);

  a_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[0].mutable_aic_aiv_ctx();
  EXPECT_EQ(a_aic_aiv->prefetch_enable_bitmap(), 0);
  EXPECT_EQ(a_aic_aiv->prefetch_once_bitmap(), 0);
  EXPECT_EQ(a_aic_aiv->src_slot_size(), 0);

  b_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[1].mutable_aic_aiv_ctx();
  EXPECT_EQ(b_aic_aiv->prefetch_enable_bitmap(), 0x0001);
  EXPECT_EQ(b_aic_aiv->prefetch_once_bitmap(), 0x0001);
  EXPECT_EQ(b_aic_aiv->src_slot_size(), 1);
  EXPECT_EQ(b_aic_aiv->src_slot(0), 2);
}

TEST_F(FftsPlusDataContextUT, prefetch_05)
{
  ge::NodePtr b;
  ComputeGraphPtr graph;
  CreateGraph01(b, graph);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx(); // a's context
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
  a_aic_aiv->set_prefetch_enable_bitmap(0);
  a_aic_aiv->set_prefetch_once_bitmap(0);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto b_aic_aiv = b_context->mutable_aic_aiv_ctx();
  b_aic_aiv->set_prefetch_enable_bitmap(0);
  b_aic_aiv->set_prefetch_once_bitmap(0);

  prefetch_.SetBurstLen(1000000);
  (void)b->GetOpDesc()->DelAttr("input_addrs");
  Status ret = prefetch_.GenManualDataCtxDef(b, ffts_plus_task_def);
  EXPECT_EQ(ffts::SUCCESS, ret);
  EXPECT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 3);
  auto prefetch_context = ffts_plus_task_def->_impl_.ffts_plus_ctx_[2].data_ctx();
  EXPECT_EQ(prefetch_context.addr_base(), 0);
}

TEST_F(FftsPlusDataContextUT, prefetch_06)
{
  ge::NodePtr b;
  ComputeGraphPtr graph;
  CreateGraph01(b, graph);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx(); // a's context
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
  a_aic_aiv->set_prefetch_enable_bitmap(0);
  a_aic_aiv->set_prefetch_once_bitmap(0);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto b_aic_aiv = b_context->mutable_aic_aiv_ctx();
  b_aic_aiv->set_prefetch_enable_bitmap(0);
  b_aic_aiv->set_prefetch_once_bitmap(0);

  prefetch_.SetBurstLen(1000000);
  (void)b->GetOpDesc()->DelAttr("input_addrs");
  vector<int64_t> input_addrs = {};
  ge::AttrUtils::SetListInt(b->GetOpDesc(), "input_addrs", input_addrs);

  Status ret = prefetch_.GenManualDataCtxDef(b, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 3);
  auto prefetch_context = ffts_plus_task_def->_impl_.ffts_plus_ctx_[2].data_ctx();
  EXPECT_EQ(prefetch_context.addr_base(), 0);
}

/* A -> B(thread dim = 4), shape is continuous. auto threading
 * B is thread 1 and A's output address is 5000, B's input address is
 * 5000.
 *
 * WARNING: Burst len is 10000000, which is larger than the total size of b's input.
 * need to use 1 data context to do the prefetch. */
TEST_F(FftsPlusDataContextUT, auto_threading_prefetch_01_1)
{
  ge::NodePtr b;
  ComputeGraphPtr graph;
  CreateAutoThreadingGraph01(b, graph);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  GenerateABAutoThreadingTaskDef(ffts_plus_task_def);

  prefetch_auto_.SetBurstLen(1000000);
  Status ret = prefetch_auto_.GenAutoDataCtxDef(b, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 22);
  auto prefetch_context = ffts_plus_task_def->_impl_.ffts_plus_ctx_[18].data_ctx();
  EXPECT_EQ(prefetch_context.thread_id(), 0);
  EXPECT_EQ(prefetch_context.addr_base(), 5000);
  EXPECT_EQ(prefetch_context.addr_offset(), 912384);
  EXPECT_EQ(prefetch_context.non_tail_len_inner(), 912384);
  EXPECT_EQ(prefetch_context.non_tail_num_inner(), 1);
  EXPECT_EQ(prefetch_context.non_tail_num_outter(), 1);
  EXPECT_EQ(prefetch_context.non_tail_stride_inner(), 3649536);
  EXPECT_EQ(prefetch_context.non_tail_stride_outter(), 3649536);

  EXPECT_EQ(prefetch_context.tail_len_inner(), 912384);
  EXPECT_EQ(prefetch_context.tail_num_inner(), 1);
  EXPECT_EQ(prefetch_context.tail_num_outter(), 1);
  EXPECT_EQ(prefetch_context.tail_stride_inner(), 3649536);
  EXPECT_EQ(prefetch_context.tail_stride_outter(), 3649536);

  prefetch_context = ffts_plus_task_def->_impl_.ffts_plus_ctx_[19].data_ctx();
  EXPECT_EQ(prefetch_context.thread_id(), 1);
  EXPECT_EQ(prefetch_context.addr_base(), 5000);
  EXPECT_EQ(prefetch_context.addr_offset(), 912384);
  EXPECT_EQ(prefetch_context.non_tail_len_inner(), 912384);
  EXPECT_EQ(prefetch_context.non_tail_num_inner(), 1);
  EXPECT_EQ(prefetch_context.non_tail_num_outter(), 1);
  EXPECT_EQ(prefetch_context.non_tail_stride_inner(), 3649536);
  EXPECT_EQ(prefetch_context.non_tail_stride_outter(), 3649536);

  EXPECT_EQ(prefetch_context.tail_len_inner(), 912384);
  EXPECT_EQ(prefetch_context.tail_num_inner(), 1);
  EXPECT_EQ(prefetch_context.tail_num_outter(), 1);
  EXPECT_EQ(prefetch_context.tail_stride_inner(), 3649536);
  EXPECT_EQ(prefetch_context.tail_stride_outter(), 3649536);

  auto a_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[5].mutable_aic_aiv_ctx();
  EXPECT_EQ(a_aic_aiv->prefetch_enable_bitmap(), 0);
  EXPECT_EQ(a_aic_aiv->prefetch_once_bitmap(), 0);
  EXPECT_EQ(a_aic_aiv->src_slot_size(), 0);

  auto b_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[9].mutable_aic_aiv_ctx();
  EXPECT_EQ(b_aic_aiv->prefetch_enable_bitmap(), 0x0001);
  EXPECT_EQ(b_aic_aiv->prefetch_once_bitmap(), 0x0001);
  EXPECT_EQ(b_aic_aiv->src_slot_size(), 1);
  EXPECT_EQ(b_aic_aiv->src_slot(0), 18);
}

/* A -> B(thread dim = 4), shape is continuous. auto threading
 * B is thread 1 and A's output address is 5000, B's input address is
 * 5000.
 *
 * WARNING: Burst len is 10000000, which is larger than the total size of b's input.
 * need to use 1 data context to do the prefetch. */
TEST_F(FftsPlusDataContextUT, dynamic_threading_prefetch_01_1)
{
  ge::NodePtr b;
  ComputeGraphPtr graph;
  ge::NodePtr a;
  CreateDynamicGraph01(b, graph, a);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  GenerateABAutoThreadingTaskDef(ffts_plus_task_def);

  Status ret = prefetch_dyn_.GenDynamicDataCtxDef(b, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
  ret = invalidate_dyn_.GenDynamicDataCtxDef(a, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
}

/* A -> B(thread dim = 4), shape is continuous.
 * B is thread 1 and A's output address is 5000, B's input address is
 * 5000.
 *
 * WARNING: Burst len is 5000000, which is less than the total size of b's input.
 * need to use 2 data context to do the prefetch. */
TEST_F(FftsPlusDataContextUT, prefetch_01_2)
{
  ge::NodePtr b;
  ComputeGraphPtr graph;
  CreateGraph01(b, graph);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx(); // a's context
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
  a_aic_aiv->set_prefetch_enable_bitmap(0);
  a_aic_aiv->set_prefetch_once_bitmap(0);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto b_aic_aiv = b_context->mutable_aic_aiv_ctx();
  b_aic_aiv->set_prefetch_enable_bitmap(0);
  b_aic_aiv->set_prefetch_once_bitmap(0);

  prefetch_.SetBurstLen(500000);
  Status ret = prefetch_.GenManualDataCtxDef(b, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 3);
  auto prefetch_context_1 = ffts_plus_task_def->_impl_.ffts_plus_ctx_[2].data_ctx();
  EXPECT_EQ(prefetch_context_1.thread_id(), 0);
  EXPECT_EQ(prefetch_context_1.addr_base(), 5000);
  EXPECT_EQ(prefetch_context_1.addr_offset(), 0);
  EXPECT_EQ(prefetch_context_1.non_tail_len_inner(), 500000);
  EXPECT_EQ(prefetch_context_1.non_tail_num_inner(), 1);
  EXPECT_EQ(prefetch_context_1.non_tail_num_outter(), 1);
  EXPECT_EQ(prefetch_context_1.non_tail_stride_inner(), 912384);
  EXPECT_EQ(prefetch_context_1.non_tail_stride_outter(), 912384);

  EXPECT_EQ(prefetch_context_1.tail_len_inner(), 500000);
  EXPECT_EQ(prefetch_context_1.tail_num_inner(), 1);
  EXPECT_EQ(prefetch_context_1.tail_num_outter(), 1);
  EXPECT_EQ(prefetch_context_1.tail_stride_inner(), 912384);
  EXPECT_EQ(prefetch_context_1.tail_stride_outter(), 912384);

  a_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[0].mutable_aic_aiv_ctx();
  EXPECT_EQ(a_aic_aiv->prefetch_enable_bitmap(), 0);
  EXPECT_EQ(a_aic_aiv->prefetch_once_bitmap(), 0);
  EXPECT_EQ(a_aic_aiv->src_slot_size(), 0);

  b_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[1].mutable_aic_aiv_ctx();
  EXPECT_EQ(b_aic_aiv->prefetch_enable_bitmap(), 1);
  EXPECT_EQ(b_aic_aiv->prefetch_once_bitmap(), 1);
  EXPECT_EQ(b_aic_aiv->src_slot_size(), 1);
  EXPECT_EQ(b_aic_aiv->src_slot(0), 2);
}

/* A -> B(thread dim = 4), shape is not continuous.
 * B is thread 1 and A's output address is 20000, B's input address is
 * 5000.
 * b's shape is {288, 8, 24, 33}, and we slice the axis{33}, select the
 * thread 1. range {{0:288},{0:8},{0:24},{8:16}}. */
TEST_F(FftsPlusDataContextUT, prefetch_02)
{
  ge::NodePtr b;
  vector<uint32_t> axes = {3};
  ComputeGraphPtr graph;
  CreateGraph02_x(b, graph, axes);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx(); // a's context
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
  a_aic_aiv->set_prefetch_enable_bitmap(0);
  a_aic_aiv->set_prefetch_once_bitmap(0);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto b_aic_aiv = b_context->mutable_aic_aiv_ctx();
  b_aic_aiv->set_prefetch_enable_bitmap(0);
  b_aic_aiv->set_prefetch_once_bitmap(0);

  Status ret = prefetch_.GenManualDataCtxDef(b, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 3);
  auto prefetch_context = ffts_plus_task_def->_impl_.ffts_plus_ctx_[2].data_ctx();
  EXPECT_EQ(prefetch_context.thread_id(), 0);
  EXPECT_EQ(prefetch_context.addr_base(), 5000);
  EXPECT_EQ(prefetch_context.addr_offset(), 16);
  EXPECT_EQ(prefetch_context.non_tail_len_inner(), 16);
  EXPECT_EQ(prefetch_context.non_tail_num_inner(), 55296);
  EXPECT_EQ(prefetch_context.non_tail_num_outter(), 1);
  EXPECT_EQ(prefetch_context.non_tail_stride_inner(), 66);
  EXPECT_EQ(prefetch_context.non_tail_stride_outter(), 3649536);

  EXPECT_EQ(prefetch_context.tail_len_inner(), 16);
  EXPECT_EQ(prefetch_context.tail_num_inner(), 55296);
  EXPECT_EQ(prefetch_context.tail_num_outter(), 1);
  EXPECT_EQ(prefetch_context.tail_stride_inner(), 66);
  EXPECT_EQ(prefetch_context.tail_stride_outter(), 3649536);

  a_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[0].mutable_aic_aiv_ctx();
  EXPECT_EQ(a_aic_aiv->prefetch_enable_bitmap(), 0);
  EXPECT_EQ(a_aic_aiv->prefetch_once_bitmap(), 0);
  EXPECT_EQ(a_aic_aiv->src_slot_size(), 0);

  b_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[1].mutable_aic_aiv_ctx();
  EXPECT_EQ(b_aic_aiv->prefetch_enable_bitmap(), 0x0001);
  EXPECT_EQ(b_aic_aiv->prefetch_once_bitmap(), 0x0001);
  EXPECT_EQ(b_aic_aiv->src_slot_size(), 1);
  EXPECT_EQ(b_aic_aiv->src_slot(0), 2);
}

/* A -> B(thread dim = 4), shape is not continuous.
 * B is thread 1 and A's output address is 20000, B's input address is
 * 5000.
 * b's shape is {288, 8, 24, 33}, and we slice the axis{288}, select the
 * thread 1. range {{72:144},{0:8},{0:24},{0:33}}.
 * Burst len is 200000, need 5 context which is not allowed. */
TEST_F(FftsPlusDataContextUT, prefetch_02_1)
{
  ge::NodePtr b;
  vector<uint32_t> axes = {0};
  ComputeGraphPtr graph;
  CreateGraph02_x(b, graph, axes);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx(); // a's context
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
  a_aic_aiv->set_prefetch_enable_bitmap(0);
  a_aic_aiv->set_prefetch_once_bitmap(0);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto b_aic_aiv = b_context->mutable_aic_aiv_ctx();
  b_aic_aiv->set_prefetch_enable_bitmap(0);
  b_aic_aiv->set_prefetch_once_bitmap(0);

  prefetch_.burst_len_ = 200000;
  Status ret = prefetch_.GenManualDataCtxDef(b, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 2);
  ;

  a_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[0].mutable_aic_aiv_ctx();
  EXPECT_EQ(a_aic_aiv->prefetch_enable_bitmap(), 0);
  EXPECT_EQ(a_aic_aiv->prefetch_once_bitmap(), 0);
  EXPECT_EQ(a_aic_aiv->src_slot_size(), 0);

  b_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[1].mutable_aic_aiv_ctx();
  EXPECT_EQ(b_aic_aiv->prefetch_enable_bitmap(), 0);
  EXPECT_EQ(b_aic_aiv->prefetch_once_bitmap(), 0);
  EXPECT_EQ(b_aic_aiv->src_slot_size(), 0);
}

/* A -> B(thread dim = 4), shape is not continuous.
 * B is thread 1 and A's output address is 20000, B's input address is
 * 5000.
 * b's shape is {288, 8, 24, 33}, and we slice the axis{288}, select the
 * thread 1. range {{72:144},{0:8},{0:24},{0:33}}.
 * Burst len is 250000, need 4 prefetch context. */
TEST_F(FftsPlusDataContextUT, prefetch_02_2)
{
  ge::NodePtr b;
  vector<uint32_t> axes = {0};
  ComputeGraphPtr graph;
  CreateGraph02_x(b, graph, axes);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx(); // a's context
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
  a_aic_aiv->set_prefetch_enable_bitmap(0);
  a_aic_aiv->set_prefetch_once_bitmap(0);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto b_aic_aiv = b_context->mutable_aic_aiv_ctx();
  b_aic_aiv->set_prefetch_enable_bitmap(0);
  b_aic_aiv->set_prefetch_once_bitmap(0);

  prefetch_.burst_len_ = 250000;
  Status ret = prefetch_.GenManualDataCtxDef(b, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 3);
  for (size_t i = 2; i <= 2; i++) {
    auto prefetch_context = ffts_plus_task_def->_impl_.ffts_plus_ctx_[i].data_ctx();
    EXPECT_EQ(prefetch_context.thread_id(), 0);
    EXPECT_EQ(prefetch_context.addr_base(), 5000);
    EXPECT_EQ(prefetch_context.addr_offset(), 912384 + 250000 * (i - 2));
    EXPECT_EQ(prefetch_context.non_tail_len_inner(), 250000);
    EXPECT_EQ(prefetch_context.non_tail_num_inner(), 1);
    EXPECT_EQ(prefetch_context.non_tail_num_outter(), 1);
    EXPECT_EQ(prefetch_context.non_tail_stride_inner(), 3649536);
    EXPECT_EQ(prefetch_context.non_tail_stride_outter(), 3649536);

    EXPECT_EQ(prefetch_context.tail_len_inner(), 250000);
    EXPECT_EQ(prefetch_context.tail_num_inner(), 1);
    EXPECT_EQ(prefetch_context.tail_num_outter(), 1);
    EXPECT_EQ(prefetch_context.tail_stride_inner(), 3649536);
    EXPECT_EQ(prefetch_context.tail_stride_outter(), 3649536);
  }

  a_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[0].mutable_aic_aiv_ctx();
  EXPECT_EQ(a_aic_aiv->prefetch_enable_bitmap(), 0);
  EXPECT_EQ(a_aic_aiv->prefetch_once_bitmap(), 0);
  EXPECT_EQ(a_aic_aiv->src_slot_size(), 0);

  b_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[1].mutable_aic_aiv_ctx();
  EXPECT_EQ(b_aic_aiv->src_slot_size(), 1);
  EXPECT_EQ(b_aic_aiv->src_slot(0), 2);

  EXPECT_EQ(b_aic_aiv->prefetch_enable_bitmap(), 1);
  EXPECT_EQ(b_aic_aiv->prefetch_once_bitmap(), 1);

}


/* A --------\
 * C ------> B
 * D -------/
 * shape is not continuous.
 * B is thread 1 and A's output address is 20000, B's input address is
 * 5000.
 * b's input0 shape is {288, 8, 24, 33}, and we slice the axis{288}
 * into 4, select the thread 1. range {{72:144},{0:8},{0:24},{0:33}}.
 * Burst len is 350000, need 3 prefetch context.
 *
 * b's input1 shape is {288, 8, 24, 33}, and we slice the axis {288} also,
 * Burst len is 350000, need 3 prefetch context. 3+3 is larger
 * than 4, we will not provide prefetch for this input.
 *
 * b's input2 shape is {288, 8, 24, 33}, and we slice the axis {24, 33},
 * Burst len is 350000, need 1 prefetch context. 3+1 is equal to 4,
 * we will provide prefetch for this input.
 *
 * */
TEST_F(FftsPlusDataContextUT, prefetch_02_2_1)
{
  ge::NodePtr b;
  ComputeGraphPtr graph;
  CreateGraph02_2_1(b, graph);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto b_aic_aiv = b_context->mutable_aic_aiv_ctx();
  b_aic_aiv->set_prefetch_enable_bitmap(0);
  b_aic_aiv->set_prefetch_once_bitmap(0);

  prefetch_.burst_len_ = 350000;
  Status ret = prefetch_.GenManualDataCtxDef(b, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 3);
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 3);
  auto prefetch_context_1 = ffts_plus_task_def->_impl_.ffts_plus_ctx_[1].data_ctx();
  EXPECT_EQ(prefetch_context_1.thread_id(), 0);
  EXPECT_EQ(prefetch_context_1.addr_base(), 5000);
  EXPECT_EQ(prefetch_context_1.addr_offset(), 912384);
  EXPECT_EQ(prefetch_context_1.non_tail_len_inner(), 350000);
  EXPECT_EQ(prefetch_context_1.non_tail_num_inner(), 1);
  EXPECT_EQ(prefetch_context_1.non_tail_num_outter(), 1);
  EXPECT_EQ(prefetch_context_1.non_tail_stride_inner(), 3649536);
  EXPECT_EQ(prefetch_context_1.non_tail_stride_outter(), 3649536);

  EXPECT_EQ(prefetch_context_1.tail_len_inner(), 350000);
  EXPECT_EQ(prefetch_context_1.tail_num_inner(), 1);
  EXPECT_EQ(prefetch_context_1.tail_num_outter(), 1);
  EXPECT_EQ(prefetch_context_1.tail_stride_inner(), 3649536);
  EXPECT_EQ(prefetch_context_1.tail_stride_outter(), 3649536);

  auto prefetch_context_2 = ffts_plus_task_def->_impl_.ffts_plus_ctx_[2].data_ctx();
  EXPECT_EQ(prefetch_context_2.thread_id(), 0);
  EXPECT_EQ(prefetch_context_2.addr_base(), 7000);
  EXPECT_EQ(prefetch_context_2.addr_offset(), 1236);
  EXPECT_EQ(prefetch_context_2.non_tail_len_inner(), 18);
  EXPECT_EQ(prefetch_context_2.non_tail_num_inner(), 6);
  EXPECT_EQ(prefetch_context_2.non_tail_num_outter(), 2304);
  EXPECT_EQ(prefetch_context_2.non_tail_stride_inner(), 66);
  EXPECT_EQ(prefetch_context_2.non_tail_stride_outter(), 1584);

  EXPECT_EQ(prefetch_context_2.tail_len_inner(), 18);
  EXPECT_EQ(prefetch_context_2.tail_num_inner(), 6);
  EXPECT_EQ(prefetch_context_2.tail_num_outter(), 2304);
  EXPECT_EQ(prefetch_context_2.tail_stride_inner(), 66);
  EXPECT_EQ(prefetch_context_2.tail_stride_outter(), 1584);

  b_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[0].mutable_aic_aiv_ctx();
  EXPECT_EQ(b_aic_aiv->src_slot_size(), 2);
  EXPECT_EQ(b_aic_aiv->src_slot(0), 1);
  EXPECT_EQ(b_aic_aiv->src_slot(1), 2);

  EXPECT_EQ(b_aic_aiv->prefetch_enable_bitmap(), 3);
  EXPECT_EQ(b_aic_aiv->prefetch_once_bitmap(), 3);

}

/* A -> B(thread dim = 4), shape is not continuous.
 * B is thread 1 and A's output address is 20000, B's input address is
 * 5000.
 * b's shape is {288, 8, 24, 33}, and we slice the axis{24, 33}, select the
 * thread 1. range {{0:288},{0:8},{6:12},{8:16}}. */
TEST_F(FftsPlusDataContextUT, prefetch_02_3)
{
  ge::NodePtr b;
  vector<uint32_t> axes = {2, 3};
  ComputeGraphPtr graph;
  CreateGraph02_x(b, graph, axes);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx(); // a's context
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
  a_aic_aiv->set_prefetch_enable_bitmap(0);
  a_aic_aiv->set_prefetch_once_bitmap(0);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto b_aic_aiv = b_context->mutable_aic_aiv_ctx();
  b_aic_aiv->set_prefetch_enable_bitmap(0);
  b_aic_aiv->set_prefetch_once_bitmap(0);

  Status ret = prefetch_.GenManualDataCtxDef(b, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 3);
  auto prefetch_context = ffts_plus_task_def->_impl_.ffts_plus_ctx_[2].data_ctx();
  EXPECT_EQ(prefetch_context.thread_id(), 0);
  EXPECT_EQ(prefetch_context.addr_base(), 5000);
  EXPECT_EQ(prefetch_context.addr_offset(), 6 * 33 * 2 + 16);
  EXPECT_EQ(prefetch_context.non_tail_len_inner(), 16);
  EXPECT_EQ(prefetch_context.non_tail_num_inner(), 6);
  EXPECT_EQ(prefetch_context.non_tail_num_outter(), 8 * 288);
  EXPECT_EQ(prefetch_context.non_tail_stride_inner(), 66);
  EXPECT_EQ(prefetch_context.non_tail_stride_outter(), 33 * 24 * 2);

  EXPECT_EQ(prefetch_context.tail_len_inner(), 16);
  EXPECT_EQ(prefetch_context.tail_num_inner(), 6);
  EXPECT_EQ(prefetch_context.tail_num_outter(), 8 * 288);
  EXPECT_EQ(prefetch_context.tail_stride_inner(), 66);
  EXPECT_EQ(prefetch_context.tail_stride_outter(), 66 * 24);

  a_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[0].mutable_aic_aiv_ctx();
  EXPECT_EQ(a_aic_aiv->prefetch_enable_bitmap(), 0);
  EXPECT_EQ(a_aic_aiv->prefetch_once_bitmap(), 0);
  EXPECT_EQ(a_aic_aiv->src_slot_size(), 0);

  b_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[1].mutable_aic_aiv_ctx();
  EXPECT_EQ(b_aic_aiv->prefetch_enable_bitmap(), 0x0001);
  EXPECT_EQ(b_aic_aiv->prefetch_once_bitmap(), 0x0001);
  EXPECT_EQ(b_aic_aiv->src_slot_size(), 1);
  EXPECT_EQ(b_aic_aiv->src_slot(0), 2);

}


/* A -> B(thread dim = 4), shape is not continuous.
 * B is thread 1 and A's output address is 20000, B's input address is
 * 5000.
 * b's shape is {288, 8, 24, 33}, and we slice the axis{8, 24, 33}, select the
 * thread 1. range {{0:288},{2:4},{6:12},{8:16}}.
 * Because the first axis 288 is not sliced, we need more than 4 prefetch context for
 * a single node which is not allowed by the data stucture of aic/aiv context.
 * */
TEST_F(FftsPlusDataContextUT, prefetch_02_4)
{
  ge::NodePtr b;
  vector<uint32_t> axes = {1, 2, 3};
  ComputeGraphPtr graph;
  CreateGraph02_x(b, graph, axes);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx(); // a's context
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
  a_aic_aiv->set_prefetch_enable_bitmap(0);
  a_aic_aiv->set_prefetch_once_bitmap(0);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto b_aic_aiv = b_context->mutable_aic_aiv_ctx();
  b_aic_aiv->set_prefetch_enable_bitmap(0);
  b_aic_aiv->set_prefetch_once_bitmap(0);

  Status ret = prefetch_.GenManualDataCtxDef(b, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 2);

}

/* A -> B(thread dim = 4), shape is not continuous.
 * B is thread 1 and A's output address is 20000, B's input address is
 * 5000.
 * b's shape is {288, 8, 24, 33}, and we slice the axis{8, 24, 33}, select the
 * thread 1. range {{0:288},{2:4},{6:12},{8:16}}.
 * Because the first axis 288 is not sliced, we need more than 4 prefetch context for
 * a single node which is not allowed by the data stucture of aic/aiv context.
 * */
TEST_F(FftsPlusDataContextUT, prefetch_02_4_1)
{
  ge::NodePtr b;
  vector<uint32_t> axes = {1, 2, 3};
  ComputeGraphPtr graph;
  CreateGraph02_4_1(b, graph, axes);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx(); // a's context
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
  a_aic_aiv->set_prefetch_enable_bitmap(0);
  a_aic_aiv->set_prefetch_once_bitmap(0);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto b_aic_aiv = b_context->mutable_aic_aiv_ctx();
  b_aic_aiv->set_prefetch_enable_bitmap(0);
  b_aic_aiv->set_prefetch_once_bitmap(0);

  Status ret = prefetch_.GenManualDataCtxDef(b, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 3);
  uint32_t offset_per_context = 8 * 24 * 33;
  uint32_t offset_in_context = 8 + 6 * 33 + 2 * 24 * 33;
  for (size_t i = 2; i <= 2; i++) {
    auto prefetch_context = ffts_plus_task_def->_impl_.ffts_plus_ctx_[i].data_ctx();
    EXPECT_EQ(prefetch_context.thread_id(), 0);
    EXPECT_EQ(prefetch_context.addr_base(), 5000);
    EXPECT_EQ(prefetch_context.addr_offset(), offset_per_context * 2 * (i - 2) + offset_in_context * 2);
    EXPECT_EQ(prefetch_context.non_tail_len_inner(), 16);
    EXPECT_EQ(prefetch_context.non_tail_num_inner(), 6);
    EXPECT_EQ(prefetch_context.non_tail_num_outter(), 2);
    EXPECT_EQ(prefetch_context.non_tail_stride_inner(), 66);
    EXPECT_EQ(prefetch_context.non_tail_stride_outter(), 24 * 33 * 2);

    EXPECT_EQ(prefetch_context.tail_len_inner(), 16);
    EXPECT_EQ(prefetch_context.tail_num_inner(), 6);
    EXPECT_EQ(prefetch_context.tail_num_outter(), 2);
    EXPECT_EQ(prefetch_context.tail_stride_inner(), 66);
    EXPECT_EQ(prefetch_context.tail_stride_outter(), 24 * 33 * 2);
  }
  a_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[0].mutable_aic_aiv_ctx();
  EXPECT_EQ(a_aic_aiv->prefetch_enable_bitmap(), 0);
  EXPECT_EQ(a_aic_aiv->prefetch_once_bitmap(), 0);
  EXPECT_EQ(a_aic_aiv->src_slot_size(), 0);

  b_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[1].mutable_aic_aiv_ctx();
  EXPECT_EQ(b_aic_aiv->src_slot_size(), 1);
  EXPECT_EQ(b_aic_aiv->src_slot(0), 2);

  EXPECT_EQ(b_aic_aiv->prefetch_enable_bitmap(), 1);
  EXPECT_EQ(b_aic_aiv->prefetch_once_bitmap(), 1);
}

/* A -> B(thread dim = 4), shape is not continuous.
 * B is thread 1 and A's output address is 20000, B's input address is
 * 5000.
 * b's shape is {288, 8, 24, 33}, and we slice the axis{288, 24, 33}, select the
 * thread 1. range {{72:144},{0:8},{6:12},{8:16}}.
 * The */
TEST_F(FftsPlusDataContextUT, prefetch_02_5)
{
  ge::NodePtr b;
  vector<uint32_t> axes = {0, 2, 3};
  ComputeGraphPtr graph;
  CreateGraph02_x(b, graph, axes);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx(); // a's context
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
  a_aic_aiv->set_prefetch_enable_bitmap(0);
  a_aic_aiv->set_prefetch_once_bitmap(0);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto b_aic_aiv = b_context->mutable_aic_aiv_ctx();
  b_aic_aiv->set_prefetch_enable_bitmap(0);
  b_aic_aiv->set_prefetch_once_bitmap(0);

  Status ret = prefetch_.GenManualDataCtxDef(b, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 3);
  auto prefetch_context = ffts_plus_task_def->_impl_.ffts_plus_ctx_[2].data_ctx();
  EXPECT_EQ(prefetch_context.thread_id(), 0);
  EXPECT_EQ(prefetch_context.addr_base(), 5000);
  EXPECT_EQ(prefetch_context.addr_offset(), (72 * 8 * 24 * 33 + 33 * 6 + 8) * 2);
  EXPECT_EQ(prefetch_context.non_tail_len_inner(), 16);
  EXPECT_EQ(prefetch_context.non_tail_num_inner(), 6);
  EXPECT_EQ(prefetch_context.non_tail_num_outter(), 8 * 72);
  EXPECT_EQ(prefetch_context.non_tail_stride_inner(), 66);
  EXPECT_EQ(prefetch_context.non_tail_stride_outter(), 66 * 24);

  EXPECT_EQ(prefetch_context.tail_len_inner(), 16);
  EXPECT_EQ(prefetch_context.tail_num_inner(), 6);
  EXPECT_EQ(prefetch_context.tail_num_outter(), 8 * 72);
  EXPECT_EQ(prefetch_context.tail_stride_inner(), 66);
  EXPECT_EQ(prefetch_context.tail_stride_outter(), 66 * 24);

  a_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[0].mutable_aic_aiv_ctx();
  EXPECT_EQ(a_aic_aiv->prefetch_enable_bitmap(), 0);
  EXPECT_EQ(a_aic_aiv->prefetch_once_bitmap(), 0);
  EXPECT_EQ(a_aic_aiv->src_slot_size(), 0);

  b_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[1].mutable_aic_aiv_ctx();
  EXPECT_EQ(b_aic_aiv->prefetch_enable_bitmap(), 0x0001);
  EXPECT_EQ(b_aic_aiv->prefetch_once_bitmap(), 0x0001);
  EXPECT_EQ(b_aic_aiv->src_slot_size(), 1);
  EXPECT_EQ(b_aic_aiv->src_slot(0), 2);

}

/* A -> B(thread dim = 4), shape is not continuous.
 * B is thread 1 and A's output address is 20000, B's input address is
 * 5000.
 * b's shape is {288, 8, 24, 33}, and we slice the axis{288, 24}, select the
 * thread 1. range {{144:216},{0:8},{12:18},{0:33}}.
 * Burst len is 250000, need 1 prefetch context. */
TEST_F(FftsPlusDataContextUT, prefetch_03)
{
  ge::NodePtr b;
  ComputeGraphPtr graph;
  CreateGraph03_x(b, graph);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx(); // a's context
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
  a_aic_aiv->set_prefetch_enable_bitmap(0);
  a_aic_aiv->set_prefetch_once_bitmap(0);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto b_aic_aiv = b_context->mutable_aic_aiv_ctx();
  b_aic_aiv->set_prefetch_enable_bitmap(0);
  b_aic_aiv->set_prefetch_once_bitmap(0);

  prefetch_.burst_len_ = 250000;
  Status ret = prefetch_.GenManualDataCtxDef(b, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 3);

  auto prefetch_context = ffts_plus_task_def->_impl_.ffts_plus_ctx_[2].data_ctx();
  EXPECT_EQ(prefetch_context.thread_id(), 0);
  EXPECT_EQ(prefetch_context.addr_base(), 5000);
  EXPECT_EQ(prefetch_context.addr_offset(), (12 * 33 + 144 * (8 * 24 * 33)) * 2);
  EXPECT_EQ(prefetch_context.non_tail_len_inner(), 396);
  EXPECT_EQ(prefetch_context.non_tail_num_inner(), 576);
  EXPECT_EQ(prefetch_context.non_tail_num_outter(), 1);
  EXPECT_EQ(prefetch_context.non_tail_stride_inner(), 1584);
  EXPECT_EQ(prefetch_context.non_tail_stride_outter(), 3649536);

  EXPECT_EQ(prefetch_context.tail_len_inner(), 396);
  EXPECT_EQ(prefetch_context.tail_num_inner(), 576);
  EXPECT_EQ(prefetch_context.tail_num_outter(), 1);
  EXPECT_EQ(prefetch_context.tail_stride_inner(), 1584);
  EXPECT_EQ(prefetch_context.tail_stride_outter(), 3649536);


  a_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[0].mutable_aic_aiv_ctx();
  EXPECT_EQ(a_aic_aiv->prefetch_enable_bitmap(), 0);
  EXPECT_EQ(a_aic_aiv->prefetch_once_bitmap(), 0);
  EXPECT_EQ(a_aic_aiv->src_slot_size(), 0);

  b_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[1].mutable_aic_aiv_ctx();
  EXPECT_EQ(b_aic_aiv->src_slot_size(), 1);
  EXPECT_EQ(b_aic_aiv->src_slot(0), 2);
  EXPECT_EQ(b_aic_aiv->prefetch_enable_bitmap(), 0x1);
  EXPECT_EQ(b_aic_aiv->prefetch_once_bitmap(), 0x1);

}


/* A -> B(thread dim = 4), shape is not continuous.
 * B is thread 1 and A's output address is 20000, B's input address is
 * 5000.
 * b's shape is {288, 8, 24, 33}, and we slice the axis{288}, select the
 * thread 1. range {{72:144},{0:8},{0:24},{0:33}}.
 * Burst len is 250000, need 4 prefetch context. */
TEST_F(FftsPlusDataContextUT, prefetch_03_1)
{
  ge::NodePtr b;
  vector<uint32_t> axes = {0};
  ComputeGraphPtr graph;
  CreateGraph02_x(b, graph, axes);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx(); // a's context
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
  a_aic_aiv->set_prefetch_enable_bitmap(0);
  a_aic_aiv->set_prefetch_once_bitmap(0);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto b_aic_aiv = b_context->mutable_aic_aiv_ctx();
  b_aic_aiv->set_prefetch_enable_bitmap(0);
  b_aic_aiv->set_prefetch_once_bitmap(0);

  prefetch_.burst_len_ = 250000;
  Status ret = prefetch_.GenManualDataCtxDef(b, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 3);
  for (size_t i = 2; i <= 2; i++) {
    auto prefetch_context = ffts_plus_task_def->_impl_.ffts_plus_ctx_[i].data_ctx();
    EXPECT_EQ(prefetch_context.thread_id(), 0);
    EXPECT_EQ(prefetch_context.addr_base(), 5000);
    EXPECT_EQ(prefetch_context.addr_offset(), 912384 + 250000 * (i - 2));
    EXPECT_EQ(prefetch_context.non_tail_len_inner(), 250000);
    EXPECT_EQ(prefetch_context.non_tail_num_inner(), 1);
    EXPECT_EQ(prefetch_context.non_tail_num_outter(), 1);
    EXPECT_EQ(prefetch_context.non_tail_stride_inner(), 3649536);
    EXPECT_EQ(prefetch_context.non_tail_stride_outter(), 3649536);

    EXPECT_EQ(prefetch_context.tail_len_inner(), 250000);
    EXPECT_EQ(prefetch_context.tail_num_inner(), 1);
    EXPECT_EQ(prefetch_context.tail_num_outter(), 1);
    EXPECT_EQ(prefetch_context.tail_stride_inner(), 3649536);
    EXPECT_EQ(prefetch_context.tail_stride_outter(), 3649536);
  }

  a_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[0].mutable_aic_aiv_ctx();
  EXPECT_EQ(a_aic_aiv->prefetch_enable_bitmap(), 0);
  EXPECT_EQ(a_aic_aiv->prefetch_once_bitmap(), 0);
  EXPECT_EQ(a_aic_aiv->src_slot_size(), 0);

  b_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[1].mutable_aic_aiv_ctx();
  EXPECT_EQ(b_aic_aiv->src_slot_size(), 1);
  EXPECT_EQ(b_aic_aiv->src_slot(0), 2);
  EXPECT_EQ(b_aic_aiv->prefetch_enable_bitmap(), 1);
  EXPECT_EQ(b_aic_aiv->prefetch_once_bitmap(), 1);

}


/* A -----> C
 *  \-----> B
 * Shape is conitnuous.
 *
 * A and B have the same thread scope id 1 and C have scope id 2.
 * split axis {288}
 * Burst len is 250000
 * need 4 write back data context.
 *
 * */
TEST_F(FftsPlusDataContextUT, write_back_01) {
  ge::NodePtr b;
  vector<uint32_t> axes = {0};
  ComputeGraphPtr graph;
  CreateGraph04_x(b, graph, axes);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx(); // a's context
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
  a_aic_aiv->set_successor_num(1);
  a_aic_aiv->add_successor_list(1);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);

  write_back_.burst_len_ = 250000;

  Status ret = write_back_.GenManualDataCtxDef(b, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 2);

  a_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[0].mutable_aic_aiv_ctx();
  EXPECT_EQ(a_aic_aiv->src_slot_size(), 0);
  EXPECT_EQ(a_aic_aiv->successor_num(), 1);
  EXPECT_EQ(a_aic_aiv->successor_list(0), 1);
}

/* A -----> C
 *  \-----> B
 * Shape is conitnuous.
 *
 * A and B have the same thread scope id 1 and C have scope id 2.
 * split axis {288}
 * Burst len is 1000000
 * need 1 write back data context.
 * */
TEST_F(FftsPlusDataContextUT, write_back_01_2) {
  ge::NodePtr a;
  vector<uint32_t> axes = {0};
  ComputeGraphPtr graph;
  CreateGraph04_x(a, graph, axes);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx(); // a's context
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
  a_aic_aiv->set_successor_num(1);
  a_aic_aiv->add_successor_list(1);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);

  write_back_.burst_len_ = 1000000;

  Status ret = write_back_.GenManualDataCtxDef(a, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 2);

  a_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[0].mutable_aic_aiv_ctx();
  EXPECT_EQ(a_aic_aiv->src_slot_size(), 0);
  EXPECT_EQ(a_aic_aiv->successor_num(), 1);
  EXPECT_EQ(a_aic_aiv->successor_list(0), 1);
}

/* A -----> C
 *  \-----> B
 * Shape is conitnuous. auto threading
 *
 * A and B have the same thread scope id 1 and C have scope id 2.
 * split axis {288}
 * Burst len is 1000000
 * need 1 write back data context.
 * */
TEST_F(FftsPlusDataContextUT, auto_threading_write_back_01_2) {
  ge::NodePtr a;
  ComputeGraphPtr graph;
  CreateAutoThreadingGraph04_x(a, graph);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  GenerateABAutoThreadingTaskDef(ffts_plus_task_def);

  write_back_auto_.burst_len_ = 1000000;

  Status ret = write_back_auto_.GenAutoDataCtxDef(a, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 18);


  // check a's succ_list has data
  bool has_data_ctx = false;
  auto aic_aiv_context_a0 = ffts_plus_task_def->_impl_.ffts_plus_ctx_[5].aic_aiv_ctx();
  for (auto succ: aic_aiv_context_a0.successor_list()) {
    if (succ == 18) {
      has_data_ctx = true;
    }
  }
  EXPECT_EQ(has_data_ctx, false);

  auto a_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[5].mutable_aic_aiv_ctx();
  EXPECT_EQ(a_aic_aiv->src_slot_size(), 0);
  EXPECT_EQ(a_aic_aiv->successor_num(), 1);
  EXPECT_EQ(a_aic_aiv->successor_list(0), 9);
}

/* A -----> C
 *  \-----> B
 * Shape is conitnuous. auto threading
 *
 * A and B have the same thread scope id 1 and C have scope id 2.
 * split axis {288}
 * Burst len is 1000000
 * need 1 write back data context.
 * */
TEST_F(FftsPlusDataContextUT, dynamic_threading_write_back_01_2) {
  ge::NodePtr a;
  ComputeGraphPtr graph;
  CreateAutoThreadingGraph04_x(a, graph);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  GenerateABAutoThreadingTaskDef(ffts_plus_task_def);

  Status ret = write_back_dyn_.GenDynamicDataCtxDef(a, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
}

/* A -----> PhonyConcat ---->C
 *  \-----> B
 * Shape is conitnuous.
 *
 * A, B, PhonyConcat have the same thread scope id 1. But D does not have scope id
 *
 * split axis {288}
 * Burst len is 1000000
 * A need 1 write back data context.
 * PhonyConcat does not need writeback;
 * */
TEST_F(FftsPlusDataContextUT, write_back_01_3) {
  ge::NodePtr a;
  vector<uint32_t> axes = {0};
  ComputeGraphPtr graph;
  CreateGraph04_x_1(a, graph, axes, 0);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx(); // a's context
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
  a_aic_aiv->set_successor_num(1);
  a_aic_aiv->add_successor_list(1);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);

  write_back_.burst_len_ = 1000000;

  Status ret = write_back_.GenManualDataCtxDef(a, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 2);

  a_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[0].mutable_aic_aiv_ctx();
  EXPECT_EQ(a_aic_aiv->src_slot_size(), 0);
  EXPECT_EQ(a_aic_aiv->successor_num(), 1);
  EXPECT_EQ(a_aic_aiv->successor_list(0), 1);
}

/* A -----> PhonyConcat ---->C
 *  \-----> B
 * Shape is conitnuous.
 *
 * A, B, PhonyConcat have the same thread scope id 1.
 * But D has a different scope id
 *
 * split axis {288}
 * Burst len is 1000000
 * A need 1 write back data context.
 * PhonyConcat does not need writeback;
 * */
TEST_F(FftsPlusDataContextUT, write_back_01_4) {
  ge::NodePtr a;
  vector<uint32_t> axes = {0};
  ComputeGraphPtr graph;
  CreateGraph04_x_1(a, graph, axes, 2);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx(); // a's context
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
  a_aic_aiv->set_successor_num(1);
  a_aic_aiv->add_successor_list(1);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);

  write_back_.burst_len_ = 1000000;

  Status ret = write_back_.GenManualDataCtxDef(a, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 2);

  a_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[0].mutable_aic_aiv_ctx();
  EXPECT_EQ(a_aic_aiv->src_slot_size(), 0);
  EXPECT_EQ(a_aic_aiv->successor_num(), 1);
  EXPECT_EQ(a_aic_aiv->successor_list(0), 1);
}

TEST_F(FftsPlusDataContextUT, write_back_01_5) {
  ge::NodePtr a;
  vector<uint32_t> axes = {0};
  ComputeGraphPtr graph;
  CreateGraph04_x_1(a, graph, axes, 2);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx(); // a's context
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
  a_aic_aiv->set_successor_num(1);
  a_aic_aiv->add_successor_list(1);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);

  write_back_.burst_len_ = 1000000;

  ge::AttrUtils::SetBool(a->GetOpDesc(), kFftsFirstSliceFlag, true);
  std::vector<std::vector<int64_t>> ori_output_tensor_shape = {{1, 1, 1, 1}};
  ge::AttrUtils::SetListListInt(a->GetOpDesc(), "ori_output_tensor_shape", ori_output_tensor_shape);
  Status ret = write_back_.GenManualDataCtxDef(a, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
}
/* A -----> C
 *  \-----> B
 * Shape is conitnuous.
 *
 * A and B and C have the same thread scope id 1.
 * split axis {33}
 * Burst len is 250000
 * need 4 write back data context.
 *
 * */
TEST_F(FftsPlusDataContextUT, invalidate_01) {
  ge::NodePtr b;
  vector<uint32_t> axes = {0};
  ComputeGraphPtr graph;
  CreateGraph05_x(b, graph, axes);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx(); // a's context
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
  a_aic_aiv->set_successor_num(2);
  a_aic_aiv->add_successor_list(1);
  a_aic_aiv->add_successor_list(2);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto b_aic_aiv = b_context->mutable_aic_aiv_ctx();

  auto c_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  c_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto c_aic_aiv = c_context->mutable_aic_aiv_ctx();

  invalidate_.burst_len_ = 250000;

  Status ret = invalidate_.GenManualDataCtxDef(b, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);

  a_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[0].mutable_aic_aiv_ctx();
  EXPECT_EQ(a_aic_aiv->src_slot_size(), 0);
  EXPECT_EQ(a_aic_aiv->successor_num(), 2);
  EXPECT_EQ(a_aic_aiv->successor_list(0), 1);
  EXPECT_EQ(a_aic_aiv->successor_list(1), 2);

  b_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[1].mutable_aic_aiv_ctx();
  EXPECT_EQ(b_aic_aiv->src_slot_size(), 0);
  EXPECT_EQ(b_aic_aiv->successor_num(), 0);

  c_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[2].mutable_aic_aiv_ctx();
  EXPECT_EQ(c_aic_aiv->src_slot_size(), 0);
  EXPECT_EQ(c_aic_aiv->successor_num(), 0);
}

/* A -----> C
 *  \-----> B
 * Shape is conitnuous.
 *
 * A and B have the same thread scope id 1 and C have scope id 2.
 * split axis {288}
 * Burst len is 1000000
 * need 1 write back data context.
 * */
TEST_F(FftsPlusDataContextUT, invalidate_01_2) {
  ge::NodePtr b;
  ComputeGraphPtr graph;
  vector<uint32_t> axes = {0};
  CreateGraph05_x(b, graph, axes);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx(); // a's context
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
  a_aic_aiv->set_successor_num(2);
  a_aic_aiv->add_successor_list(1);
  a_aic_aiv->add_successor_list(2);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto b_aic_aiv = b_context->mutable_aic_aiv_ctx();

  auto c_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  c_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto c_aic_aiv = c_context->mutable_aic_aiv_ctx();

  invalidate_.burst_len_ = 1000000;

  Status ret = invalidate_.GenManualDataCtxDef(b, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 3);

  a_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[0].mutable_aic_aiv_ctx();
  EXPECT_EQ(a_aic_aiv->src_slot_size(), 0);
  EXPECT_EQ(a_aic_aiv->successor_num(), 2);
  EXPECT_EQ(a_aic_aiv->successor_list(0), 1);
  EXPECT_EQ(a_aic_aiv->successor_list(1), 2);

  b_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[1].mutable_aic_aiv_ctx();
  EXPECT_EQ(b_aic_aiv->src_slot_size(), 0);
  EXPECT_EQ(b_aic_aiv->successor_num(), 0);

  c_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[2].mutable_aic_aiv_ctx();
  EXPECT_EQ(c_aic_aiv->src_slot_size(), 0);
  EXPECT_EQ(c_aic_aiv->successor_num(), 0);
}

/* A -----> C
 *  \-----> B
 * Shape is conitnuous. auto threading
 *
 * A and B have the same thread scope id 1 and C have scope id 2.
 * split axis {288}
 * Burst len is 1000000
 * need 1 write back data context.
 * */
TEST_F(FftsPlusDataContextUT, auto_threading_invalidate_01_2) {
  ge::NodePtr b;
  ComputeGraphPtr graph;
  vector<uint32_t> axes = {0};
  CreateAutoThreadGraph05_x(b, graph, axes);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  GenerateABCAutoThreadingTaskDef(ffts_plus_task_def);

  invalidate_auto_.burst_len_ = 1000000;

  Status ret = invalidate_auto_.GenAutoDataCtxDef(b, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 22);

  auto a_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[5].mutable_aic_aiv_ctx();
  EXPECT_EQ(a_aic_aiv->src_slot_size(), 0);
  EXPECT_EQ(a_aic_aiv->successor_num(), 2);
  EXPECT_EQ(a_aic_aiv->successor_list(0), 9);
  EXPECT_EQ(a_aic_aiv->successor_list(1), 13);

  auto b_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[9].mutable_aic_aiv_ctx();
  EXPECT_EQ(b_aic_aiv->src_slot_size(), 0);
  EXPECT_EQ(b_aic_aiv->successor_num(), 1);
  EXPECT_EQ(b_aic_aiv->successor_list(0), 17);

  auto c_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[13].mutable_aic_aiv_ctx();
  EXPECT_EQ(c_aic_aiv->src_slot_size(), 0);
  EXPECT_EQ(c_aic_aiv->successor_num(), 1);
  EXPECT_EQ(c_aic_aiv->successor_list(0), 17);
}

/* A -----> C
 *  \-----> B
 * Shape is conitnuous. auto threading
 *
 * A and B have the same thread scope id 1 and C have scope id 2.
 * split axis {288}
 * Burst len is 1000000
 * need 1 write back data context.
 * */
TEST_F(FftsPlusDataContextUT, dynamic_threading_invalidate_01_2) {
  ge::NodePtr b;
  ComputeGraphPtr graph;
  vector<uint32_t> axes = {0};
  CreateDynamicGraph05_x(b, graph, axes);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  GenerateABCAutoThreadingTaskDef(ffts_plus_task_def);

  Status ret = invalidate_dyn_.GenDynamicDataCtxDef(b, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
}

/* A -----> PhonyConcat ----> C
 *  \-----> B
 * Shape is conitnuous.
 *
 * A and B and C have the same thread scope id 1.
 * split axis {288}
 * Burst len is 1000000
 * A need 1 write back data context.
 * B and C's successor lists need to be updated.
 * */
TEST_F(FftsPlusDataContextUT, invalidate_01_3) {
  ge::NodePtr a;
  ComputeGraphPtr graph;
  vector<uint32_t> axes = {0};
  CreateGraph04_x_1(a, graph, axes, 1);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx(); // a's context
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
  a_aic_aiv->set_successor_num(2);
  a_aic_aiv->add_successor_list(1);
  a_aic_aiv->add_successor_list(2);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto b_aic_aiv = b_context->mutable_aic_aiv_ctx();

  auto c_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  c_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto c_aic_aiv = c_context->mutable_aic_aiv_ctx();

  invalidate_.burst_len_ = 1000000;

  Status ret = invalidate_.GenManualDataCtxDef(a, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 3);

  b_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[1].mutable_aic_aiv_ctx();
  EXPECT_EQ(b_aic_aiv->src_slot_size(), 0);
  EXPECT_EQ(b_aic_aiv->successor_num(), 0);

  c_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[2].mutable_aic_aiv_ctx();
  EXPECT_EQ(c_aic_aiv->src_slot_size(), 0);
  EXPECT_EQ(c_aic_aiv->successor_num(), 0);
}

TEST_F(FftsPlusDataContextUT, invalidate_01_4) {
  ge::NodePtr a;
  ComputeGraphPtr graph;
  vector<uint32_t> axes = {0};
  CreateGraphInvalidFusion(a, graph, axes);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx(); // a's context
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);

  auto c_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  c_context->set_context_type(RT_HW_CTX_TYPE_AIV);

  invalidate_.burst_len_ = 1000000;

  Status ret = invalidate_.GenManualDataCtxDef(a, ffts_plus_task_def);
  ASSERT_EQ(ffts::SUCCESS, ret);
  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 3);
}

/* A -----> B ----> C -----> D -----> E
 *
 * Shape is conitnuous.
 *
 * A and B and C and D have the same thread scope id 1.
 *
 * */
TEST_F(FftsPlusDataContextUT, AUTO_THREADING_INVALID_NO_MEM_REUSE) {
  vector<ge::NodePtr> nodes;
  ComputeGraphPtr graph;
  vector<uint32_t> axes = {0};
  CreateGraphInvalidAndMemReuse(nodes, graph, axes, false, true);

  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  GenerateABCDEAutoThreadingTaskDef(ffts_plus_task_def);

  invalidate_auto_.burst_len_ = 1000000;

  for (auto &node : nodes) {
    Status ret = invalidate_auto_.GenAutoDataCtxDef(node, ffts_plus_task_def);
    ASSERT_EQ(ffts::SUCCESS, ret);
  }

  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 30);
}

/* A -----> B ----> C -----> D -----> E
 *
 * Shape is conitnuous.
 *
 * A and B and C and D have the same thread scope id 1.
 * */
TEST_F(FftsPlusDataContextUT, AUTO_THREADING_INVALID_WITH_MEM_REUSE) {
  vector<ge::NodePtr> nodes;
  ComputeGraphPtr graph;
  vector<uint32_t> axes = {0};
  CreateGraphInvalidAndMemReuse(nodes, graph, axes, true, true);

  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  GenerateABCDEAutoThreadingTaskDef(ffts_plus_task_def);

  invalidate_auto_.burst_len_ = 1000000;

  for (auto &node : nodes) {
    Status ret = invalidate_auto_.GenAutoDataCtxDef(node, ffts_plus_task_def);
    ASSERT_EQ(ffts::SUCCESS, ret);
  }

  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 30);

  auto a_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[6].mutable_aic_aiv_ctx();
  EXPECT_EQ(a_aic_aiv->src_slot_size(), 0);
  ASSERT_EQ(a_aic_aiv->successor_num(), 1);
  EXPECT_EQ(a_aic_aiv->successor_list(0), 10);

  auto b_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[10].mutable_aic_aiv_ctx();
  EXPECT_EQ(b_aic_aiv->src_slot_size(), 0);
  ASSERT_EQ(b_aic_aiv->successor_num(), 1);
  EXPECT_EQ(b_aic_aiv->successor_list(0), 14);

  auto c_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[14].mutable_aic_aiv_ctx();
  EXPECT_EQ(c_aic_aiv->src_slot_size(), 0);
  ASSERT_EQ(c_aic_aiv->successor_num(), 1);
  EXPECT_EQ(c_aic_aiv->successor_list(0), 18);

  auto d_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[18].mutable_aic_aiv_ctx();
  EXPECT_EQ(d_aic_aiv->src_slot_size(), 0);
  ASSERT_EQ(d_aic_aiv->successor_num(), 1);
  EXPECT_EQ(d_aic_aiv->successor_list(0), 22);
}

/* A -----> B ----> C -----> D ----> E
 *
 * Shape is conitnuous.
 * No memory reuse
 * A and B and C and D have the same thread scope id 1.
 *
 * ctx:
 * A ------> B ------> C -----> D -----> E
 *             \        \        \       \
 *           invalid  invalid   invalid  invalid
 * */
TEST_F(FftsPlusDataContextUT, MUANUAL_THREADING_INVALID_NO_MEM_REUSE) {
  vector<ge::NodePtr> nodes;
  ComputeGraphPtr graph;
  vector<uint32_t> axes = {0};
  CreateGraphInvalidAndMemReuse(nodes, graph, axes, false, false);

  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx(); // a's context
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
  a_aic_aiv->set_successor_num(1);
  a_aic_aiv->add_successor_list(1);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto b_aic_aiv = b_context->mutable_aic_aiv_ctx();
  b_aic_aiv->set_successor_num(1);
  b_aic_aiv->add_successor_list(2);

  auto c_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  c_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto c_aic_aiv = c_context->mutable_aic_aiv_ctx();
  c_aic_aiv->set_successor_num(1);
  c_aic_aiv->add_successor_list(3);

  auto d_context = ffts_plus_task_def->add_ffts_plus_ctx(); // d's context
  d_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto d_aic_aiv = d_context->mutable_aic_aiv_ctx();
  d_aic_aiv->set_successor_num(1);
  d_aic_aiv->add_successor_list(4);

  auto e_context = ffts_plus_task_def->add_ffts_plus_ctx(); // e's context
  e_context->set_context_type(RT_HW_CTX_TYPE_AIV);

  invalidate_.burst_len_ = 1000000;

  for (auto &node : nodes) {
    Status ret = invalidate_.GenManualDataCtxDef(node, ffts_plus_task_def);
    ASSERT_EQ(ffts::SUCCESS, ret);
  }

  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 5);

  a_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[0].mutable_aic_aiv_ctx();
  EXPECT_EQ(a_aic_aiv->src_slot_size(), 0);
  ASSERT_EQ(a_aic_aiv->successor_num(), 1);
  EXPECT_EQ(a_aic_aiv->successor_list(0), 1);

  b_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[1].mutable_aic_aiv_ctx();
  EXPECT_EQ(b_aic_aiv->src_slot_size(), 0);
  ASSERT_EQ(b_aic_aiv->successor_num(), 1);
  EXPECT_EQ(b_aic_aiv->successor_list(0), 2);

  c_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[2].mutable_aic_aiv_ctx();
  EXPECT_EQ(c_aic_aiv->src_slot_size(), 0);
  ASSERT_EQ(c_aic_aiv->successor_num(), 1);
  EXPECT_EQ(c_aic_aiv->successor_list(0), 3);

  d_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[3].mutable_aic_aiv_ctx();
  EXPECT_EQ(d_aic_aiv->src_slot_size(), 0);
  ASSERT_EQ(d_aic_aiv->successor_num(), 1);
  EXPECT_EQ(d_aic_aiv->successor_list(0), 4);
}

/*
 *             memory reuse
 *           /               \
 * A -----> B ----> C -----> D -----> E
 *   \              /                 /
 *     memory reuse------------------
 * Shape is conitnuous.
 *
 * A and B and C and D have the same thread scope id 1.
 *
 * ctx:
 * A ------> B -------
 *             \       \
 *           invalid --> C -------
 *                        \        \
 *                      invalid --> D ---------
 *                                   \         \
 *                                invalid-----> E ---> invalid
 *
 * */
TEST_F(FftsPlusDataContextUT, MUANUAL_THREADING_INVALID_WITH_MEM_REUSE) {
  vector<ge::NodePtr> nodes;
  ComputeGraphPtr graph;
  vector<uint32_t> axes = {0};
  CreateGraphInvalidAndMemReuse(nodes, graph, axes, true, false);

  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx(); // a's context
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
  a_aic_aiv->set_successor_num(1);
  a_aic_aiv->add_successor_list(1);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto b_aic_aiv = b_context->mutable_aic_aiv_ctx();
  b_aic_aiv->set_successor_num(1);
  b_aic_aiv->add_successor_list(2);

  auto c_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  c_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto c_aic_aiv = c_context->mutable_aic_aiv_ctx();
  c_aic_aiv->set_successor_num(1);
  c_aic_aiv->add_successor_list(3);

  auto d_context = ffts_plus_task_def->add_ffts_plus_ctx(); // b's context
  d_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto d_aic_aiv = d_context->mutable_aic_aiv_ctx();
  d_aic_aiv->set_successor_num(1);
  d_aic_aiv->add_successor_list(4);

  auto e_context = ffts_plus_task_def->add_ffts_plus_ctx(); // e's context
  e_context->set_context_type(RT_HW_CTX_TYPE_AIV);

  invalidate_.burst_len_ = 1000000;

  for (auto &node : nodes) {
    Status ret = invalidate_.GenManualDataCtxDef(node, ffts_plus_task_def);
    ASSERT_EQ(ffts::SUCCESS, ret);
  }

  ASSERT_EQ(ffts_plus_task_def->ffts_plus_ctx_size(), 5);

  a_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[0].mutable_aic_aiv_ctx();
  EXPECT_EQ(a_aic_aiv->src_slot_size(), 0);
  ASSERT_EQ(a_aic_aiv->successor_num(), 1);
  EXPECT_EQ(a_aic_aiv->successor_list(0), 1);

  b_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[1].mutable_aic_aiv_ctx();
  EXPECT_EQ(b_aic_aiv->src_slot_size(), 0);
  ASSERT_EQ(b_aic_aiv->successor_num(), 1);
  EXPECT_EQ(b_aic_aiv->successor_list(0), 2);

  c_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[2].mutable_aic_aiv_ctx();
  EXPECT_EQ(c_aic_aiv->src_slot_size(), 0);
  ASSERT_EQ(c_aic_aiv->successor_num(), 1);
  EXPECT_EQ(c_aic_aiv->successor_list(0), 3);

  d_aic_aiv = ffts_plus_task_def->_impl_.ffts_plus_ctx_[3].mutable_aic_aiv_ctx();
  EXPECT_EQ(d_aic_aiv->src_slot_size(), 0);
  ASSERT_EQ(d_aic_aiv->successor_num(), 1);
  EXPECT_EQ(d_aic_aiv->successor_list(0), 4);

  ge::MemReuseInfo reuse_info;
  reuse_info.node = nullptr;
  vector<ge::MemReuseInfo> reuse_infos{reuse_info};
  size_t num1 = 0;
  int num = 0;
  Status ret = invalidate_.UpdateSuccListWithMemReuse(nodes[0], reuse_infos, ffts_plus_task_def, num, num1);
  EXPECT_EQ(ffts::SUCCESS, ret);

  reuse_infos.clear();
  ret = invalidate_.UpdateSuccListWithMemReuse(nodes[0], reuse_infos, ffts_plus_task_def, num, num1);
  EXPECT_EQ(ffts::FAILED, ret);

  ret = invalidate_.GenInvalidSuccListWithMemReuse(nodes[0], num1, ffts_plus_task_def, num, num1);
  EXPECT_EQ(ffts::SUCCESS, ret);

  num1 = 1;
  ret = invalidate_.GenInvalidSuccListWithMemReuse(nodes[0], num1, ffts_plus_task_def, num, num1);
  EXPECT_EQ(ffts::FAILED, ret);

  reuse_info.node = nodes[3];
  reuse_infos.emplace_back(reuse_info);
  (void)ge::AttrUtils::SetBool(nodes[0]->GetOpDesc(), ge::ATTR_NAME_REFERENCE, true);
  ret = invalidate_.UpdateSuccListWithMemReuse(nodes[0], reuse_infos, ffts_plus_task_def, num, num1);
  EXPECT_EQ(ffts::SUCCESS, ret);
  reuse_info.node = nodes[3];
  reuse_infos.emplace_back(reuse_info);
  std::vector<int32_t> ref_port_index = {0};
  (void)ge::AttrUtils::SetListInt(nodes[0]->GetOpDesc()->MutableOutputDesc(0), "ref_port_index", ref_port_index);
  ret = invalidate_.UpdateSuccListWithMemReuse(nodes[0], reuse_infos, ffts_plus_task_def, num, num1);
  EXPECT_EQ(ffts::SUCCESS, ret);
}

TEST_F(FftsPlusDataContextUT, MUANUAL_THREADING_INVALID_WITH_MEM_REUSE_PY) {
  vector<ge::NodePtr> nodes;
  ComputeGraphPtr graph;
  CreateInvalidAndMemReusePy(nodes, graph);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto a_context = ffts_plus_task_def->add_ffts_plus_ctx();
  a_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto a_aic_aiv = a_context->mutable_aic_aiv_ctx();
  a_aic_aiv->set_successor_num(1);
  a_aic_aiv->add_successor_list(1);

  auto b_context = ffts_plus_task_def->add_ffts_plus_ctx();
  b_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto b_aic_aiv = b_context->mutable_aic_aiv_ctx();
  b_aic_aiv->set_successor_num(1);
  b_aic_aiv->add_successor_list(3);

  auto c_context = ffts_plus_task_def->add_ffts_plus_ctx();
  c_context->set_context_type(RT_HW_CTX_TYPE_AIV);
  auto c_aic_aiv = c_context->mutable_aic_aiv_ctx();
  c_aic_aiv->set_successor_num(1);
  c_aic_aiv->add_successor_list(3);

  auto d_context = ffts_plus_task_def->add_ffts_plus_ctx();
  d_context->set_context_type(RT_HW_CTX_TYPE_INVALIDATE_DATA);

  size_t wind_id = 0;
  int data_ctx_id = 3;
  Status ret = invalidate_.UpdateInvalidCtxWithMemReuse(nodes[1], data_ctx_id, wind_id, ffts_plus_task_def);
  EXPECT_EQ(ffts::SUCCESS, ret);
  auto ffts_plus_ctx_def = ffts_plus_task_def->mutable_ffts_plus_ctx(3);
  auto data_ctx = ffts_plus_ctx_def->mutable_data_ctx();
  EXPECT_EQ(data_ctx->successor_num(), 2);
  EXPECT_EQ(data_ctx->successor_list(0), 1);
  EXPECT_EQ(data_ctx->successor_list(1), 2);

  ret = invalidate_.UpdateInvalidCtxWithMemReuse(nodes[0], data_ctx_id, wind_id, ffts_plus_task_def);
  EXPECT_EQ(ffts::SUCCESS, ret);
  ffts_plus_ctx_def = ffts_plus_task_def->mutable_ffts_plus_ctx(3);
  data_ctx = ffts_plus_ctx_def->mutable_data_ctx();
  EXPECT_EQ(data_ctx->successor_list(0), 1);
}

TEST_F(FftsPlusDataContextUT, test_fill_dynamic_data_ctx) {
  ge::NodePtr b;
  ComputeGraphPtr graph;
  vector<uint32_t> axes = {0};
  CreateDynamicGraph05_x(b, graph, axes);
  domi::TaskDef task_def;
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  GenerateABCAutoThreadingTaskDef(ffts_plus_task_def);
  size_t out_anchor_index = 10;
  rtFftsPlusContextType_t context_type;
  vector<uint32_t> context_id_list = {1, 1, 1};
  write_back_dyn_.FillDynamicDataCtx(out_anchor_index, b, ffts_plus_task_def, context_type, context_id_list);
}

TEST_F(FftsPlusDataContextUT, write_back_01_6) {
  ge::NodePtr a;
  vector<uint32_t> axes = {0};
  ComputeGraphPtr graph;
  CreateGraph04_x_1(a, graph, axes, 0);
  uint32_t out_anchor_index = 23;
  std::vector<uint32_t> succ_list;
  uint32_t cons_cnt;
  write_back_.GetSuccessorContextIdSpecial(out_anchor_index, a, succ_list, cons_cnt);
}
