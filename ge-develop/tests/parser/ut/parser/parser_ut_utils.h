/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_PARSER_TESTS_UT_PARSER_H_
#define GE_PARSER_TESTS_UT_PARSER_H_

#include "framework/omg/parser/parser_inner_ctx.h"
#include "graph/compute_graph.h"

namespace ge {
struct MemBuffer {
  void *data;
  uint32_t size;
};

class ParerUTestsUtils {
 public:
  static void ClearParserInnerCtx();
  static MemBuffer* MemBufferFromFile(const char *path);
  static bool ReadProtoFromText(const char *file, google::protobuf::Message *message);
  static void WriteProtoToBinaryFile(const google::protobuf::Message &proto, const char *filename);
  static void WriteProtoToTextFile(const google::protobuf::Message &proto, const char *filename);
};

namespace ut {
class GraphBuilder {
 public:
  explicit GraphBuilder(const std::string &name) { graph_ = std::make_shared<ComputeGraph>(name); }
  NodePtr AddNode(const std::string &name, const std::string &type, int in_cnt, int out_cnt,
                  Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT,
                  std::vector<int64_t> shape = {1, 1, 224, 224});
  void AddDataEdge(const NodePtr &src_node, int src_idx, const NodePtr &dst_node, int dst_idx);
  void AddControlEdge(const NodePtr &src_node, const NodePtr &dst_node);
  ComputeGraphPtr GetGraph() {
    graph_->TopologicalSorting();
    return graph_;
  }

 private:
  ComputeGraphPtr graph_;
};
}  // namespace ut
}  // namespace ge

#endif  // GE_PARSER_TESTS_UT_PARSER_H_
