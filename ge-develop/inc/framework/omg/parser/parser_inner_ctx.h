/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_OMG_PARSER_PARSER_INNER_CONTEXT_H_
#define INC_FRAMEWORK_OMG_PARSER_PARSER_INNER_CONTEXT_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "register/register_fmk_types.h"
#include "register/register_types.h"
#include "framework/omg/omg_inner_types.h"

namespace ge {
struct ParserContext {
  // format of the input specified by the command line
  std::unordered_map<std::string, domi::domiTensorFormat_t> input_nodes_format_map;
  std::vector<domi::domiTensorFormat_t> output_formats;
  // user-designate input dims
  std::vector<std::pair<std::string, std::vector<int64_t>>> user_input_dims;
  std::map<std::string, std::vector<int64_t>> input_dims;
  // resolve the mapping between operators with the same name and corresponding network. format e.g.
  // Detectionoutput:SsdDetectiontOutput
  std::map<std::string, std::string> op_conf_map;
  // user-designate out nodes (this is used for determing the orders)
  std::vector<std::pair<std::string, int32_t>> user_out_nodes;
  // default out nodes (this is used for determing the orders)
  std::vector<std::pair<std::string, int32_t>> default_out_nodes;
  // save the output node of the network. key = operator name, value = index, index indicates the output index of the
  // operator
  std::map<std::string, std::vector<int32_t>> out_nodes_map;
  // save the output node of the network, value = topName,
  // tensorName indicates the output name of the operator.
  std::vector<std::string> user_out_tensors;
  // net out nodes (where user_out_nodes or leaf nodes)
  std::vector<std::string> net_out_nodes;
  // net out nodes tensor names(caffe or onnx)
  std::vector<std::string> out_tensor_names;
  // net data nodes tensor names(caffe or onnx)
  std::vector<std::string> data_tensor_names;
  // Whether to use dynamic batch size or dynamic image size
  bool is_dynamic_input = false;
  bool train_flag = false;
  domi::domiTensorFormat_t format = domi::DOMI_TENSOR_ND;
  domi::FrameworkType type = domi::FRAMEWORK_RESERVED;
  RunMode run_mode = RunMode::GEN_OM_MODEL;
  // save caffe custom proto path, used by caffe parse
  std::string custom_proto_path;
  // save caffe proto path, used by caffe parse
  std::string caffe_proto_path;
  // name of the pass that needs to take effect
  std::string enable_scope_fusion_passes;
  // Records output node names that were renamed during parsing (old_name:index, <new_name, index>)
  std::map<std::string, std::pair<std::string, int32_t>> final_out_nodes_map;
};

GE_FUNC_VISIBILITY ParserContext &GetParserContext();
}  // namespace ge

#endif  // INC_FRAMEWORK_OMG_PARSER_PARSER_INNER_CONTEXT_H_
