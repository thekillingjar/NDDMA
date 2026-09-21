/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <map>
#include <memory>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "proto/om.pb.h"

#define protected public
#define private public
#include "common/graph_comm.h"
#include "pass_manager.h"
#include "common/configuration.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/op_kernel_bin.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/ub_fusion/buffer_fusion.h"
#include "graph_optimizer/ub_fusion/automatic_buffer_fusion.h"
#include "../../../graph_constructor/graph_constructor.h"

#undef protected
#undef private
using namespace std;
using namespace domi;
using namespace fe;

class UB_FUSION_ST_NETWORK_TOPOLOGY : public testing::Test {
 public:
 protected:

  static void SetUpTestCase() { std::cout << "UB fusion SetUp" << std::endl; }

  static void TearDownTestCase() {
    std::cout << "UB fusion TearDown" << std::endl;
  }

  virtual void SetUp() {
    graph_comm_ptr_ = std::make_shared<GraphComm>("engineName");
    graph_comm_ptr_->Initialize();
    sub_graph_optimizer_ptr_ =
        std::make_shared<BufferFusion>(graph_comm_ptr_, nullptr, nullptr);
    auto_buffer_fusion_ptr_ = std::make_shared<AutomaticBufferFusion>(nullptr);
  }

  virtual void TearDown() {

 }

  std::shared_ptr<GraphComm> graph_comm_ptr_;
  std::shared_ptr<AutomaticBufferFusion> auto_buffer_fusion_ptr_;
  std::shared_ptr<BufferFusion> sub_graph_optimizer_ptr_;

  void BuildGraph(ge::ComputeGraphPtr &graph) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);

                //   impl_type    pattern    op_name  op_type inputs_size outputs_size
    // AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add1",  ADD,      2,         1)
    test.AddOpDesc(EN_IMPL_HW_TBE, "aipp", "aipp1", "Aipp", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv1", "Conv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "leakyRelu1", "LeakyRelu", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv2", "Conv2D", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "leakyRelu2", "LeakyRelu", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv3", "Conv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "leakyRelu3", "LeakyRelu", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv4", "Conv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "leakyRelu4", "LeakyRelu", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "eltwise1", "Eltwise", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv5", "Conv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant1", "AscendDequant", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "eltwise2", "Eltwise", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "leakyRelu5", "LeakyRelu", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "quant", "quant1", "AscendQuant", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv6", "Conv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant2", "AscendDequant", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "quant", "quant2", "AscendQuant", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv7", "Conv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant3", "AscendDequant", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "quant", "quant3", "AscendQuant", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv8", "Conv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant4", "AscendDequant", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "eltwise3", "Eltwise", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "leakyRelu6", "LeakyRelu", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Pool2d", "pooling1", "Pooling", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "quant", "quant4", "AscendQuant", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "aipp", "aipp2", "Aipp", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv9", "Conv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "leakyRelu7", "LeakyRelu", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Pool2d", "pooling2", "Pooling", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "quant", "quant5", "AscendQuant", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv10", "Conv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant5", "AscendDequant", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "quant", "quant6", "AscendQuant", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv11", "Conv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant6", "AscendDequant", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv12", "Conv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant7", "AscendDequant", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "eltwise4", "Eltwise", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "leakyRelu8", "LeakyRelu", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv13", "Conv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant8", "AscendDequant", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "read_select", "readSelect1", "ReadSelect", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "eltwise5", "Eltwise", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "leakyRelu9", "LeakyRelu", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "quant", "quant7", "AscendQuant", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv14", "Conv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant9", "AscendDequant", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "eltwise6", "Eltwise", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv15", "Conv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant10", "AscendDequant", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "leakyRelu10", "LeakyRelu", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "DepthwiseConvolution", "depthwiseConv2D1", "DepthwiseConv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant11", "AscendDequant", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "quant", "quant8", "AscendQuant", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv16", "Conv2D", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant12", "AscendDequant", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Pool2d", "pooling3", "Pooling", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "quant", "quant9", "AscendQuant", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "FullyConnection", "fullyConnection1", "FullyConnection", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant13", "AscendDequant", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Softmax", "softmaxV2_1", "SoftmaxV2", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "MaxPool", "maxPool1", "MaxPool", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "quant", "quant10", "AscendQuant", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv17", "Conv2D", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant14", "AscendDequant", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "strided_write", "stridedWrite1", "StridedWrite", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ConcatV2D", "concatV2D1", "ConcatV2D", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv18", "Conv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant15", "AscendDequant", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "strided_write", "stridedWrite2", "StridedWrite", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "CommReduce", "reduceMeanD1", "ReduceMeanD", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "quant", "quant11", "AscendQuant", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "FullyConnection", "fullyConnection2", "FullyConnection", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant16", "AscendDequant", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "TransData", "transData1", "TransData", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv19", "Conv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "leakyRelu11", "LeakyRelu", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Pool2d", "pooling4", "Pooling", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "quant", "quant12", "AscendQuant", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv20", "Conv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant17", "AscendDequant", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "quant", "quant13", "AscendQuant", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv21", "Conv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant18", "AscendDequant", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "eltwise7", "Eltwise", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "leakyRelu12", "LeakyRelu", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "quant", "quant14", "AscendQuant", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv22", "Conv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant19", "AscendDequant", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "quant", "quant15", "AscendQuant", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv23", "Conv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant20", "AscendDequant", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "eltwise8", "Eltwise", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "quant", "quant16", "AscendQuant", 1, 1)
                // dst_name,    src_name
        //SetInput("add1:0",   "Data_1")
        .SetInput("aipp1", "PlaceHolder_1")
        .SetInputs("conv1", {"aipp1", "PlaceHolder_2", "PlaceHolder_3"})
        .SetInput("leakyRelu1", "conv1")
        .SetInput("conv2", "leakyRelu1")
        .SetInput("leakyRelu2", "conv2")
        .SetInputs("conv3", {"leakyRelu2", "PlaceHolder_4", "PlaceHolder_5"})
        .SetInput("leakyRelu3", "conv3")
        .SetInputs("conv4", {"leakyRelu3", "PlaceHolder_6", "PlaceHolder_7"})
        .SetInput("leakyRelu4", "conv4")
        .SetInputs("eltwise1", {"leakyRelu2", "leakyRelu4"})
        .SetInputs("conv5", {"eltwise1", "PlaceHolder_8", "PlaceHolder_9"})
        .SetInputs("dequant1", {"conv5", "PlaceHolder_10"})
        .SetInput("eltwise2", "dequant1")
        .SetInput("leakyRelu5", "eltwise2")
        .SetInput("quant1", "leakyRelu5:0")
        .SetInputs("conv6", {"quant1", "PlaceHolder_11", "PlaceHolder_12"})
        .SetInputs("dequant2", {"conv6", "PlaceHolder_13"})
        .SetInput("quant2", "dequant2")
        .SetInputs("conv7", {"quant2", "PlaceHolder_14", "PlaceHolder_15"})
        .SetInputs("dequant3", {"conv7", "PlaceHolder_16"})
        .SetInput("quant3", "dequant3")
        .SetInputs("conv8", {"quant3", "PlaceHolder_17", "PlaceHolder_18"})
        .SetInputs("dequant4", {"conv8", "PlaceHolder_19"})
        .SetInputs("eltwise3", {"leakyRelu5:0", "dequant4"})
        .SetInput("leakyRelu6", "eltwise3")
        .SetInput("pooling1", "leakyRelu6")
        .SetInput("quant4", "pooling1")
        .SetInputs("aipp2", {"quant4", "PlaceHolder_20"})
        .SetInputs("conv9", {"aipp2", "PlaceHolder_21", "PlaceHolder_22"})
        .SetInput("leakyRelu7", "conv9")
        .SetInput("pooling2", "leakyRelu7")
        .SetInput("quant5", "pooling2")
        .SetInputs("conv10", {"quant5", "PlaceHolder_23", "PlaceHolder_24"})
        .SetInputs("dequant5", {"conv10", "PlaceHolder_25"})
        .SetInput("quant6", "dequant5")
        .SetInputs("conv11", {"quant6", "PlaceHolder_26", "PlaceHolder_27"})
        .SetInputs("dequant6", {"conv11", "PlaceHolder_64"})
        .SetInputs("conv12", {"quant5", "PlaceHolder_28", "PlaceHolder_29"})
        .SetInputs("dequant7", {"conv12", "PlaceHolder_65"})
        .SetInputs("eltwise4", {"dequant7", "dequant6"})
        .SetInput("leakyRelu8", "eltwise4")
        .SetInputs("conv13", {"leakyRelu8:0", "PlaceHolder_30", "PlaceHolder_31"})
        .SetInputs("dequant8", {"conv13", "PlaceHolder_32"})
        .SetInput("readSelect1", "leakyRelu8:0")
        .SetInputs("eltwise5", {"readSelect1", "dequant8"})
        .SetInput("leakyRelu9", "eltwise5")
        .SetInput("quant7", "leakyRelu9")
        .SetInputs("conv14", {"quant7", "PlaceHolder_33", "PlaceHolder_34"})
        .SetInputs("dequant9", {"conv14", "PlaceHolder_35"})
        .SetInput("conv15", "PlaceHolder_36", ge::FORMAT_NCHW, ge::FORMAT_NCHW, {34,66,70,90})
        .SetInputs("dequant10", {"conv15", "PlaceHolder_37"})
        .SetInputs("eltwise6", {"dequant9", "dequant10"})
        .SetInput("leakyRelu10", "eltwise6")
        .SetInput("quant16", "leakyRelu10")
        .SetInputs("depthwiseConv2D1", {"quant16", "PlaceHolder_38", "PlaceHolder_39"})
        .SetInputs("dequant11", {"depthwiseConv2D1", "PlaceHolder_66"})
        .SetInput("quant8", "dequant11")
        .SetInput("conv16", "quant8")
        .SetInputs("dequant12", {"conv16", "PlaceHolder_67"})
        .SetInput("pooling3", "dequant12")
        .SetInput("quant9", "pooling3")
        .SetInputs("fullyConnection1", {"quant9", "PlaceHolder_40", "PlaceHolder_41"})
        .SetInputs("dequant13", {"fullyConnection1", "PlaceHolder_42"})
        .SetInput("softmaxV2_1", "dequant13")
        .SetInput("maxPool1", "softmaxV2_1")
        .SetInput("quant10", "maxPool1")
        .SetInput("conv17", "quant10")
        .SetInputs("dequant14", {"conv17", "PlaceHolder_68"})
        .SetInput("stridedWrite1", "dequant14")
        .SetInputs("conv18", {"PlaceHolder_43", "PlaceHolder_44", "PlaceHolder_45"})
        .SetInputs("dequant15", {"conv18", "PlaceHolder_46"})
        .SetInput("stridedWrite2", "dequant15")
        .SetInputs("concatV2D1", {"stridedWrite2", "stridedWrite1"})
        .SetInput("reduceMeanD1", "concatV2D1")
        .SetInput("quant11", "reduceMeanD1")
        .SetInputs("fullyConnection2", {"quant11", "PlaceHolder_47", "PlaceHolder_48"})
        .SetInputs("dequant16", {"fullyConnection2", "PlaceHolder_49"})
        .SetInput("transData1", "dequant16")
        .SetInputs("conv19", {"transData1", "PlaceHolder_50", "PlaceHolder_51"})
        .SetInput("leakyRelu11", "conv19")
        .SetInput("pooling4", "leakyRelu11")
        .SetInput("quant12", "pooling4")
        .SetInputs("conv20", {"quant12", "PlaceHolder_52", "PlaceHolder_53"})
        .SetInputs("dequant17", {"conv20", "PlaceHolder_54"})
        .SetInput("quant13", "dequant17")
        .SetInputs("conv21", {"quant13", "PlaceHolder_55", "PlaceHolder_56"})
        .SetInputs("dequant18", {"conv21", "PlaceHolder_57"})
        .SetInputs("eltwise7", {"pooling4", "dequant18"})
        .SetInput("leakyRelu12", "eltwise7")
        .SetInput("quant14", "leakyRelu12:0")
        .SetInputs("conv22", {"quant14", "PlaceHolder_58", "PlaceHolder_59"})
        .SetInputs("dequant19", {"conv22", "PlaceHolder_60"})
        .SetInput("quant15", "dequant19")
        .SetInputs("conv23", {"quant15", "PlaceHolder_61", "PlaceHolder_62"})
        .SetInputs("dequant20", {"conv23", "PlaceHolder_63"})
        .SetInputs("eltwise8", {"leakyRelu12:0", "dequant20"})
        .SetInput("End_1", "eltwise8");
  }

  void AfterMatchGraph(ge::ComputeGraphPtr &graph) {
    ge::GeShape original_shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NHWC, ge::DT_FLOAT,
                          original_shape);

                //   impl_type    pattern    op_name  op_type inputs_size outputs_size
    // AddOpDesc(EN_IMPL_HW_TBE, "ElemWise", "add1",  ADD,      2,         1)
    test.AddOpDesc(EN_IMPL_HW_TBE, "aipp", "aipp1", "Aipp", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv1leakyRelu1", "Conv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv2leakyRelu2", "Conv2D", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv3leakyRelu3", "Conv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv4leakyRelu4eltwise1", "Conv2D", 4, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv5dequant1eltwise2leakyRelu5quant1", "Conv2D", 4, 2)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv6dequant2quant2", "Conv2D", 4, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv7dequant3quant3", "Conv2D", 4, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv8dequant4leakyRelu6eltwise3", "Conv2D", 5, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Pool2d", "pooling1quant4", "Pooling", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "aipp", "aipp2", "Aipp", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv9leakyRelu7", "Conv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Pool2d", "pooling2quant5", "Pooling", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv10dequant5quant6", "Conv2D", 4, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv11dequant6eltwise4leakyRelu8", "Conv2D", 4, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv12dequant7", "Conv2D", 4, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv13dequant8readSelect1eltwise5leakyRelu9quant7", "Conv2D", 5, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv14dequant9", "Conv2D", 4, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "eltwise6conv15dequant10leakyRelu10quant16", "Conv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "DepthwiseConvolution", "depthwiseConv2D1dequant11quant8", "DepthwiseConv2D", 4, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv16dequant12", "Conv2D", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Pool2d", "pooling3quant9", "Pooling", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "FullyConnection", "fullyConnection1", "FullyConnection", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant13", "AscendDequant", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Softmax", "softmaxV2_1", "SoftmaxV2", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "MaxPool", "maxPool1", "MaxPool", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "quant", "quant10", "AscendQuant", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv17stridedWrite1dequant14", "Conv2D", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "ConcatV2D", "concatV2D1", "ConcatV2D", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv18dequant15stridedWrite2", "Conv2D", 4, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "CommReduce", "reduceMeanD1", "ReduceMeanD", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "quant", "quant11", "AscendQuant", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "FullyConnection", "fullyConnection2", "FullyConnection", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "dequant", "dequant16", "AscendDequant", 2, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "TransData", "transData1", "TransData", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv19leakyRelu11", "Conv2D", 3, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Pool2d", "pooling4", "Pooling", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "quant", "quant12", "AscendQuant", 1, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv20dequant17quant13", "Conv2D", 4, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv21dequant18eltwise7leakyRelu12quant14", "Conv2D", 5, 2)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv22dequant19quant15", "Conv2D", 4, 1)
        .AddOpDesc(EN_IMPL_HW_TBE, "Convolution", "conv23dequant20eltwise8", "Conv2D", 5, 1)
                // dst_name,    src_name
        //SetInput("add1:0",   "Data_1")
        .SetInput("aipp1", "PlaceHolder_1")
        .SetInputs("conv1leakyRelu1", {"aipp1", "PlaceHolder_2", "PlaceHolder_3"})
        .SetInput("conv2leakyRelu2", "conv1leakyRelu1")
        .SetInputs("conv3leakyRelu3", {"conv2leakyRelu2", "PlaceHolder_4", "PlaceHolder_5"})
        .SetInputs("conv4leakyRelu4eltwise1",
                   {"conv3leakyRelu3", "PlaceHolder_6", "PlaceHolder_7", "conv2leakyRelu2"})
        .SetInputs("conv5dequant1eltwise2leakyRelu5quant1",
                   {"conv4leakyRelu4eltwise1", "PlaceHolder_8", "PlaceHolder_9", "PlaceHolder_10"})
        .SetInputs("conv6dequant2quant2",
                   {"conv5dequant1eltwise2leakyRelu5quant1:1", "PlaceHolder_11", "PlaceHolder_12", "PlaceHolder_13"})
        .SetInputs("conv7dequant3quant3",
                   {"conv6dequant2quant2", "PlaceHolder_14", "PlaceHolder_15", "PlaceHolder_16"})
        .SetInputs("conv8dequant4leakyRelu6eltwise3",
                   {"conv7dequant3quant3", "PlaceHolder_17", "PlaceHolder_18", "PlaceHolder_19", "conv5dequant1eltwise2leakyRelu5quant1:0"})
        .SetInput("pooling1quant4", "conv8dequant4leakyRelu6eltwise3")
        .SetInputs("aipp2", {"pooling1quant4", "PlaceHolder_20"})
        .SetInputs("conv9leakyRelu7", {"aipp2", "PlaceHolder_21", "PlaceHolder_22"})
        .SetInput("pooling2quant5", "conv9leakyRelu7")
        .SetInputs("conv10dequant5quant6", {"pooling2quant5", "PlaceHolder_23", "PlaceHolder_24", "PlaceHolder_25"})
        .SetInputs("conv11dequant6eltwise4leakyRelu8",
                   {"conv10dequant5quant6", "PlaceHolder_26", "PlaceHolder_27", "PlaceHolder_64", "conv12dequant7"})
        .SetInputs("conv12dequant7", {"pooling2quant5", "PlaceHolder_28", "PlaceHolder_29", "PlaceHolder_65"})
        .SetInputs("conv13dequant8readSelect1eltwise5leakyRelu9quant7",
                   {"conv11dequant6eltwise4leakyRelu8:0", "PlaceHolder_30", "PlaceHolder_31", "PlaceHolder_32", "conv11dequant6eltwise4leakyRelu8:0"})
        .SetInputs("conv14dequant9",
                   {"conv13dequant8readSelect1eltwise5leakyRelu9quant7", "PlaceHolder_33", "PlaceHolder_34", "PlaceHolder_35"})
        .SetInput("eltwise6conv15dequant10leakyRelu10quant16", "conv14dequant9")
        .SetInput("eltwise6conv15dequant10leakyRelu10quant16",
                    "PlaceHolder_36", ge::FORMAT_NCHW, ge::FORMAT_NCHW, {34, 66, 70, 90})
        .SetInput("eltwise6conv15dequant10leakyRelu10quant16", "PlaceHolder_37")
        .SetInputs("depthwiseConv2D1dequant11quant8",
                  {"eltwise6conv15dequant10leakyRelu10quant16", "PlaceHolder_38", "PlaceHolder_39", "PlaceHolder_66"})
        .SetInputs("conv16dequant12", {"depthwiseConv2D1dequant11quant8", "PlaceHolder_67"})
        .SetInput("pooling3quant9", "conv16dequant12")
        .SetInputs("fullyConnection1", {"pooling3quant9", "PlaceHolder_40", "PlaceHolder_41"})
        .SetInputs("dequant13", {"fullyConnection1", "PlaceHolder_42"})
        .SetInput("softmaxV2_1", "dequant13")
        .SetInput("maxPool1", "softmaxV2_1")
        .SetInput("quant10", "maxPool1")
        .SetInputs("conv17stridedWrite1dequant14", {"quant10", "PlaceHolder_68"})
        .SetInputs("conv18dequant15stridedWrite2",
                    {"PlaceHolder_43", "PlaceHolder_44", "PlaceHolder_45", "PlaceHolder_46"})
        .SetInputs("concatV2D1", {"conv18dequant15stridedWrite2", "conv17stridedWrite1dequant14"})
        .SetInput("reduceMeanD1", "concatV2D1")
        .SetInput("quant11", "reduceMeanD1")
        .SetInputs("fullyConnection2", {"quant11", "PlaceHolder_47", "PlaceHolder_48"})
        .SetInputs("dequant16", {"fullyConnection2", "PlaceHolder_49"})
        .SetInput("transData1", "dequant16")
        .SetInputs("conv19leakyRelu11", {"transData1", "PlaceHolder_50", "PlaceHolder_51"})
        .SetInput("pooling4", "conv19leakyRelu11")
        .SetInput("quant12", "pooling4")
        .SetInputs("conv20dequant17quant13",
                   {"quant12", "PlaceHolder_52", "PlaceHolder_53", "PlaceHolder_54"})
        .SetInputs("conv21dequant18eltwise7leakyRelu12quant14",
                   {"conv20dequant17quant13", "PlaceHolder_55", "PlaceHolder_56", "PlaceHolder_57", "pooling4"})
        .SetInputs("conv22dequant19quant15",
                  {"conv21dequant18eltwise7leakyRelu12quant14:1", "PlaceHolder_58", "PlaceHolder_59", "PlaceHolder_60"})
        .SetInputs("conv23dequant20eltwise8",
                  {"conv22dequant19quant15", "PlaceHolder_61", "PlaceHolder_62", "PlaceHolder_63", "conv21dequant18eltwise7leakyRelu12quant14:0"})
        .SetInput("End_1", "conv23dequant20eltwise8");
  }
};

TEST_F(UB_FUSION_ST_NETWORK_TOPOLOGY, network_topology) {
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  BuildGraph(graph);
  for (ge::NodePtr node : graph->GetDirectNode()) {
    if (node->GetType() == "StridedWrite") {
      ge::OpDescPtr op_desc = node->GetOpDesc();
      ge::GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(0);
      tensor_desc->SetFormat(ge::FORMAT_NC1HWC0);
      tensor_desc->SetShape(ge::GeShape({3, 1, 12, 5, 16}));
      (void)ge::AttrUtils::SetInt(op_desc, "stride", 1);
    }
  }

  std::shared_ptr<GraphComm> graph_comm_ptr = std::make_shared<GraphComm>("engineName");
  graph_comm_ptr->Initialize();
  std::shared_ptr<FusionPriorityManager> fusion_priority_mgr_ptr =
      std::make_shared<FusionPriorityManager>("engineName",nullptr);
  std::shared_ptr<BufferFusion> sub_graph_optimizer_ptr =
      std::make_shared<BufferFusion>(graph_comm_ptr, fusion_priority_mgr_ptr, nullptr);

  sub_graph_optimizer_ptr->engine_name_ = fe::AI_CORE_NAME;
  // find sub-graphs that match UB fusion pattern
  sub_graph_optimizer_ptr->MatchFusionPatternFromGraph(*graph);

  int id = 0;
  cerr << endl;
  cerr << "UB_FUSION_ST_NETWORK_TOPOLOGY UB fusion match result" << endl;
  for (auto &node : graph->GetDirectNode()) {
    cerr << "id:" << id << endl;
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType() << endl;
    if (ge::AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    } else {
      cout <<"=====no scope ID" << endl;
    }
    id++;
  }

  // create fused Graph, and merge matched sub-graphs into fusion ops
  sub_graph_optimizer_ptr->BuildFusionGraph(*graph);

  cerr << endl;
  cerr << "UB_FUSION_ST_NETWORK_TOPOLOGY UB fusion build result" << endl;
  for (auto &node : graph->GetDirectNode()) {
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType() << endl;
    if (ge::AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    } else {
      cout <<"=====no scope ID" << endl;
    }
  }

  ComputeGraphPtr graph_check = std::make_shared<ge::ComputeGraph>("check");
  AfterMatchGraph(graph_check);
  cerr << endl;
  cerr << "UB_FUSION_ST_NETWORK_TOPOLOGY UB fusion check graph" << endl;
  for (auto &node : graph_check->GetDirectNode()) {
    uint32_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType() << endl;
    if (ge::AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id)) {
      cerr << "scope id : " << scope_id << endl;
    } else {
      cout <<"=====no scope ID" << endl;
    }
  }

  EXPECT_EQ(GraphConstructor::CompareGraph(graph, graph_check), false);
}
