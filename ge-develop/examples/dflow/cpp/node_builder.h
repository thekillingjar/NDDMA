/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NODE_BUILDER_H
#define NODE_BUILDER_H

#include <fstream>
#include <vector>
#include <map>
#include <memory.h>
#include "all_ops.h"
#include "graph/graph.h"
#include "parser/tensorflow_parser.h"
#include "parser/onnx_parser.h"
#include "ge/ge_api.h"
#include "flow_graph/data_flow.h"

using namespace ge;
using namespace dflow;

struct BuildBasicConfig {
  std::string node_name;
  uint32_t input_num;
  uint32_t output_num;
  std::string compile_cfg;
};

FlowNode BuildGraphNode(const BuildBasicConfig &build_cfg, const GraphBuilder &builder) {
  // construct graph pp
  std::string pp_name = build_cfg.node_name + "_pp";
  auto pp = GraphPp(pp_name.c_str(), builder).SetCompileConfig(build_cfg.compile_cfg.c_str());
  // construct graph node
  auto node = FlowNode(build_cfg.node_name.c_str(), build_cfg.input_num, build_cfg.output_num);

  node.AddPp(pp);
  return node;
}

FlowNode BuildGraphNode(const BuildBasicConfig &build_cfg, const std::string &fmk, const std::string &model_file,
                        const std::map<AscendString, AscendString> &parser_params) {
  std::string node_name = build_cfg.node_name;
  return BuildGraphNode(build_cfg, [node_name, fmk, model_file, parser_params]() {
    std::string graph_name = node_name + "_pp_" + fmk + "_graph";
    Graph graph(graph_name.c_str());
    uint32_t ret;
    if (fmk.compare("TF") == 0) {
      ret = ge::aclgrphParseTensorFlow(model_file.c_str(), parser_params, graph);
      if (ret != 0) {
        std::cout << "ERROR:TEST======parse tensorflow failed.====================" << std::endl;
      } else {
        std::cout << "SUCCESS:TEST======parse tensorflow succeed.====================" << std::endl;
      }
    } else if (fmk.compare("ONNX") == 0) {
      ret = ge::aclgrphParseONNX(model_file.c_str(), parser_params, graph);
      if (ret != 0) {
        std::cout << "ERROR:TEST======parse ONNX failed.====================" << std::endl;
      } else {
        std::cout << "SUCCESS:TEST======parse ONNX succeed.====================" << std::endl;
      }
    } else {
      std::cout << "ERROR:TEST======model type is not support=====================" << std::endl;
    }
    return graph;
  });
}

FlowNode BuildOnnxGraphNode(const BuildBasicConfig &build_cfg, const std::string &onnx_file,
                            const std::map<AscendString, AscendString> &parser_params) {
  return BuildGraphNode(build_cfg, "ONNX", onnx_file, parser_params);
}

FlowNode BuildTfGraphNode(const BuildBasicConfig &build_cfg, const std::string &pb_file,
                          const std::map<AscendString, AscendString> &parser_params) {
  return BuildGraphNode(build_cfg, "TF", pb_file, parser_params);
}

FlowNode BuildFunctionNodeSimple(const BuildBasicConfig &build_cfg, const bool enableException = false) {
  // construct FunctionPp
  std::string pp_name = build_cfg.node_name + "_pp";
  auto pp = FunctionPp(pp_name.c_str()).SetCompileConfig(build_cfg.compile_cfg.c_str());
  pp.SetInitParam("enableExceptionCatch", enableException);
  // construct FlowNode
  auto node = FlowNode(build_cfg.node_name.c_str(), build_cfg.input_num, build_cfg.output_num);
  node.AddPp(pp);
  return node;
}

using FuctionPpSetter = std::function<FunctionPp(FunctionPp)>;

FlowNode BuildFunctionNode(const BuildBasicConfig &build_cfg, FuctionPpSetter pp_setter) {
  // construct FunctionPp
  std::string pp_name = build_cfg.node_name + "_pp";
  auto pp = FunctionPp(pp_name.c_str()).SetCompileConfig(build_cfg.compile_cfg.c_str());
  pp = pp_setter(pp);
  // construct FlowNode

  auto node = FlowNode(build_cfg.node_name.c_str(), build_cfg.input_num, build_cfg.output_num);
  node.AddPp(pp);
  return node;
}

using MapSetter = std::function<FlowNode(FlowNode, FunctionPp)>;
FlowNode BuildFunctionNode(const BuildBasicConfig &build_cfg, FuctionPpSetter pp_setter, MapSetter map_setter) {
  // construct FunctionPp
  std::string pp_name = build_cfg.node_name + "_pp";
  auto pp = FunctionPp(pp_name.c_str()).SetCompileConfig(build_cfg.compile_cfg.c_str());
  pp = pp_setter(pp);

  // construct node
  auto node = FlowNode(build_cfg.node_name.c_str(), build_cfg.input_num, build_cfg.output_num);
  // node add pp
  node.AddPp(pp);
  node = map_setter(node, pp);
  return node;
}
#endif  // NODE_BUILDER_H