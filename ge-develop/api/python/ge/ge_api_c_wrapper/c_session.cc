/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "graph/graph.h"
#include "utils/graph_utils.h"
#include "graph/tensor.h"
#include "ge/ge_api.h"
#include "utils/graph_utils_ex.h"
#include "common/checker.h"
#include "ge_api_c_wrapper_utils.h"
#include <iostream>

#ifdef __cplusplus
using namespace ge;
using namespace ge::c_wrapper;
using Status = uint32_t;
using Tensor = ge::Tensor;
extern "C" {
#endif

Session *GeApiWrapper_Session_CreateSession() {
  std::map<ge::AscendString, ge::AscendString> options;
  auto *session = new (std::nothrow) ge::Session(options);
  return session;
}

Session *GeApiWrapper_Session_CreateSessionWithOptions(char **keys, char **values, int size) {
  GE_ASSERT_NOTNULL(keys);
  GE_ASSERT_NOTNULL(values);
  std::map<AscendString, AscendString> options;
  for (int i = 0; i < size; i++) {
    GE_ASSERT_NOTNULL(keys[i]);
    GE_ASSERT_NOTNULL(values[i]);
    options.emplace(keys[i], values[i]);
  }
  auto *session = new (std::nothrow) ge::Session(options);
  return session;
}

Status GeApiWrapper_Session_AddGraph(Session *session, uint32_t graph_id, Graph *graph) {
  GE_ASSERT_NOTNULL(session);
  GE_ASSERT_NOTNULL(graph);
  return session->AddGraph(graph_id, *graph);
}

Status GeApiWrapper_Session_AddGraphWithOptions(Session *session, uint32_t graph_id, Graph *graph, char **keys, char **values, int size) {
  GE_ASSERT_NOTNULL(session);
  GE_ASSERT_NOTNULL(graph);
  GE_ASSERT_NOTNULL(keys);
  GE_ASSERT_NOTNULL(values);
  std::map<AscendString, AscendString> options;
  for (int i = 0; i < size; i++) {
    GE_ASSERT_NOTNULL(keys[i]);
    GE_ASSERT_NOTNULL(values[i]);
    options.emplace(keys[i], values[i]);
  }
  return session->AddGraph(graph_id, *graph, options);
}

Tensor** GeApiWrapper_Session_RunGraph(Session *session, uint32_t graph_id, void **inputs, int input_count, size_t *tensor_num) {
  GE_ASSERT_NOTNULL(inputs);
  GE_ASSERT_NOTNULL(tensor_num);
  std::vector<Tensor> inputs_vector;
  for (int i = 0; i < input_count; i++) {
    auto *tn = static_cast<Tensor *>(inputs[i]);
    inputs_vector.push_back(*tn);
  }
  std::vector<Tensor> outputs_vector;
  GE_ASSERT_GRAPH_SUCCESS(session->RunGraph(graph_id, inputs_vector, outputs_vector));
  return VecTensorToArray(outputs_vector, tensor_num);
}

void GeApiWrapper_Session_DestroySession(const Session *session) {
  delete session;
}

#ifdef __cplusplus
}
#endif