/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_OPTIMIZE_GRAPH_FUNCTIONDEF_H
#define GE_GRAPH_OPTIMIZE_GRAPH_FUNCTIONDEF_H

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "graph/anchor.h"
#include "graph/ge_attr_value.h"
#include "graph/graph.h"
#include "proto/tensorflow/graph.pb.h"
#include "register/register_error_codes.h"

using domi::tensorflow::AttrValue;
using domi::tensorflow::AttrValue_ListValue;
using domi::tensorflow::DataType;
using domi::tensorflow::DT_INVALID;
using domi::tensorflow::FunctionDef;
using domi::tensorflow::FunctionDefLibrary;
using domi::tensorflow::NodeDef;
using std::string;
using std::to_string;
using std::vector;

namespace ge {
class GraphToFunctionDef {
 public:
  static domi::Status RecordArg(ge::ComputeGraphPtr graph,
                          const vector<ge::InDataAnchorPtr> &in_anchor);

  static domi::Status RecordResult(ge::ComputeGraphPtr graph,
                             const vector<ge::OutDataAnchorPtr> &out_anchor);

  static domi::Status DavGraphToFunctionDef(ge::ComputeGraphPtr graph,
                                      const string &name, FunctionDef *fdef);

  static domi::Status BuildFunctionDef(ge::ComputeGraphPtr &graph,
                                 const string &name_in,
                                 FunctionDefLibrary *library,
                                 NodeDef *call_node_def,
                                 vector<ge::InDataAnchorPtr> &in_anchor,
                                 vector<ge::OutDataAnchorPtr> &out_anchor);

  static bool FindAttrValue(const domi::tensorflow::NodeDef *node_def,
                            const string attr_name,
                            domi::tensorflow::AttrValue &attr_value);

  static void AddNodeAttr(const string &attr_name,
                          const domi::tensorflow::AttrValue &value,
                          domi::tensorflow::NodeDef *node_def);
};

class NameMapHelper {
 public:
  NameMapHelper() = default;

  ~NameMapHelper() {}

  string UniqueInputOrOutputName(const string &name);

  string UniqueNodeName(const string &name);

  string Renormalize(const string &name) const;

 private:
  string GetUniqueName(const string &name);

  std::set<string> used_names_;
  std::map<string, string> name_mapping_;
};
}  // namespace ge

#endif  // GE_GRAPH_OPTIMIZE_GRAPH_FUNCTIONDEF_H
