/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "ascendc_ir.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "ascir_ops.h"

#include "pyascir.h"
#include "pyascir_types.h"
#include "pyascir_common_utils.h"

class TestPyModule : public testing::Test {
 protected:
  virtual void SetUp() {
    Py_Initialize();
    PyInit_pyautofuse();
    PyInit_ascir();
  }

  virtual void TearDown() {
    Py_Finalize();
  }
};

TEST_F(TestPyModule, DISABLED_UtilsDeserialize_symbol_source_info) {
  PyObject *args = PyTuple_New(2);
  PyObject *kwds = PyDict_New();

  {
    PyObject *type = PyUnicode_FromString("symbol_source_info");
    PyObject *obj = PyUnicode_FromString("{\"s0\":\"GetDimValueFromGraphInputData(0, 0);\"}");

    PyTuple_SetItem(args, 0, type); // 添加到索引 0
    PyTuple_SetItem(args, 1, obj); // 添加到索引 1

    auto res = pyascir::UtilsDeserialize(nullptr, args, kwds);
    EXPECT_NE(res, nullptr);

    auto shape_info = (pyascir::ShapeInfo::Object *)res;
    EXPECT_EQ(shape_info->shape_info["s0"], "GetDimValueFromGraphInputData(0, 0);");
    Py_DECREF(res);
  }

  {
    PyObject *type = PyUnicode_FromString("symbol_source_info");
    PyObject *obj = PyUnicode_FromString("error_test");

    PyTuple_SetItem(args, 0, type); // 添加到索引 0
    PyTuple_SetItem(args, 1, obj); // 添加到索引 1

    auto res = pyascir::UtilsDeserialize(nullptr, args, kwds);
    EXPECT_EQ(res, nullptr);
  }

  {
    PyObject *type = PyUnicode_FromString("symbol_source_info");
    PyObject *obj = PyUnicode_FromString("{\"s0\": 1}");

    PyTuple_SetItem(args, 0, type); // 添加到索引 0
    PyTuple_SetItem(args, 1, obj); // 添加到索引 1

    auto res = pyascir::UtilsDeserialize(nullptr, args, kwds);
    EXPECT_EQ(res, nullptr);
  }

  Py_DECREF(args);
  Py_DECREF(kwds);
}

TEST_F(TestPyModule, DISABLED_Deserialize_output_symbol_shape) {
  PyObject *output_list = PyList_New(0);

  {
    PyObject *inner_list1 = PyList_New(2);
    PyList_SetItem(inner_list1, 0, PyUnicode_FromString("s0"));
    PyList_SetItem(inner_list1, 1, PyUnicode_FromString("s1"));
    PyObject *inner_list2 = PyList_New(2);
    PyList_SetItem(inner_list2, 0, PyUnicode_FromString("s2"));
    PyList_SetItem(inner_list2, 1, PyUnicode_FromString("s3"));

    PyList_Append(output_list, inner_list1);
    PyList_Append(output_list, inner_list2);

    std::vector<std::vector<std::string>> output_shape;
    auto ret = pyascir::OutputSymbolShapeDeserialize(output_list, output_shape);
    EXPECT_TRUE(ret);
    Py_DECREF(inner_list1);
    Py_DECREF(inner_list2);

    ASSERT_EQ(output_shape.size(), 2U);
    std::vector<std::string> inner_1 = output_shape.at(0);
    std::vector<std::string> inner_2 = output_shape.at(1);
    ASSERT_EQ(inner_1.size(), 2U);
    ASSERT_EQ(inner_2.size(), 2U);
    EXPECT_EQ(inner_1.at(0), "s0");
    EXPECT_EQ(inner_1.at(1), "s1");
    EXPECT_EQ(inner_2.at(0), "s2");
    EXPECT_EQ(inner_2.at(1), "s3");
  }

  PyList_SetSlice(output_list, 0, PyList_Size(output_list), NULL);
  {
    PyList_Append(output_list, PyUnicode_FromString("s0"));
    std::vector<std::vector<std::string>> output_shape;
    auto ret = pyascir::OutputSymbolShapeDeserialize(output_list, output_shape);
    EXPECT_FALSE(ret);
  }

  PyList_SetSlice(output_list, 0, PyList_Size(output_list), NULL);
  {
    PyObject *inner_list1 = PyList_New(2);
    PyList_SetItem(inner_list1, 0, PyUnicode_FromString("s0"));
    PyList_SetItem(inner_list1, 1, PyLong_FromLong(42));

    PyList_Append(output_list, inner_list1);
    std::vector<std::vector<std::string>> output_shape;
    auto ret = pyascir::OutputSymbolShapeDeserialize(output_list, output_shape);
    EXPECT_FALSE(ret);
    Py_DECREF(inner_list1);
  }

  Py_DECREF(output_list);
}

TEST_F(TestPyModule, DISABLED_UtilsDeserialize_asc_graph) {
  PyObject *args = PyTuple_New(2);
  PyObject *kwds = PyDict_New();

  ge::AscGraph g("fused_graph");
  ge::ascir_op::Data x("x", g);
  auto node = g.FindNode("x");
  node->inputs();
  node->outputs();
  g.SetTilingKey(0x5a5a);
  std::string to_be_deser;
  EXPECT_EQ(ge::AscGraphUtils::SerializeToReadable(g, to_be_deser), ge::GRAPH_SUCCESS);

  {
    PyObject *type = PyUnicode_FromString("asc_graph");
    PyObject *obj = PyUnicode_FromString(to_be_deser.c_str());

    PyTuple_SetItem(args, 0, type); // 添加到索引 0
    PyTuple_SetItem(args, 1, obj); // 添加到索引 1

    auto res = pyascir::UtilsDeserialize(nullptr, args, kwds);
    EXPECT_NE(res, nullptr);

    auto graph = (pyascir::HintGraph::Object *)res;
    EXPECT_EQ(graph->graph->GetName(), "fused_graph");
    Py_DECREF(res);
  }

  {
    PyObject *type = PyUnicode_FromString("asc_graph");
    PyObject *obj = PyUnicode_FromString("error");

    PyTuple_SetItem(args, 0, type); // 添加到索引 0
    PyTuple_SetItem(args, 1, obj); // 添加到索引 1

    auto res = pyascir::UtilsDeserialize(nullptr, args, kwds);
    EXPECT_EQ(res, nullptr);
  }

  Py_DECREF(args);
  Py_DECREF(kwds);
}

TEST_F(TestPyModule, DISABLED_UtilsDeserialize_compute_graph) {
  PyObject *args = PyTuple_New(2);
  PyObject *kwds = PyDict_New();

  {
    PyObject *type = PyUnicode_FromString("compute_graph");
    PyObject *obj = PyUnicode_FromString("error");

    PyTuple_SetItem(args, 0, type); // 添加到索引 0
    PyTuple_SetItem(args, 1, obj); // 添加到索引 1

    auto res = pyascir::UtilsDeserialize(nullptr, args, kwds);
    EXPECT_EQ(res, nullptr);
  }

  Py_DECREF(args);
  Py_DECREF(kwds);
}

// TEST_F(TestPyModule, UtilsDeserialize_error_type) {
//   PyObject *args = PyTuple_New(2);
//   PyObject *kwds = PyDict_New();

//   {
//     PyObject *type = PyUnicode_FromString("error_type");
//     PyObject *obj = PyUnicode_FromString("error");

//     PyTuple_SetItem(args, 0, type); // 添加到索引 0
//     PyTuple_SetItem(args, 1, obj); // 添加到索引 1

//     auto res = pyascir::UtilsDeserialize(nullptr, args, kwds);
//     EXPECT_EQ(res, nullptr);
//   }

//   Py_DECREF(args);
//   Py_DECREF(kwds);
// }

// TEST_F(TestPyModule, UtilsDurationRecord) {
//   PyObject *args = PyTuple_New(3);
//   PyObject *kwds = PyDict_New();

//   // target list type error
//   {
//     PyObject *target_list = PyList_New(2);
//     PyList_SetItem(target_list, 0, PyLong_FromLong(111));
//     PyList_SetItem(target_list, 1, PyLong_FromLong(222));

//     PyTuple_SetItem(args, 0, target_list); // 添加到索引 0
//     PyTuple_SetItem(args, 1, PyLong_FromLong(1234)); // 添加到索引 1
//     PyTuple_SetItem(args, 2, PyLong_FromLong(1234)); // 添加到索引 2

//     auto res = pyascir::UtilsDurationRecord(nullptr, args, kwds);
//     EXPECT_EQ(res, nullptr);
//     pyascir::UtilsReportDurations(nullptr, args, kwds);
//     Py_DECREF(target_list);
//   }

//   // target type error
//   {
//     PyTuple_SetItem(args, 0, PyLong_FromLong(1234)); // 添加到索引 0
//     PyTuple_SetItem(args, 1, PyLong_FromLong(1234)); // 添加到索引 1
//     PyTuple_SetItem(args, 2, PyLong_FromLong(1234)); // 添加到索引 2

//     auto res = pyascir::UtilsDurationRecord(nullptr, args, kwds);
//     EXPECT_EQ(res, nullptr);
//   }

//   // args error
//   {
//     PyTuple_SetItem(args, 0, PyLong_FromLong(1234)); // 添加到索引 0
//     auto res = pyascir::UtilsDurationRecord(nullptr, args, kwds);
//     EXPECT_EQ(res, nullptr);
//   }

//   // time error
//   {
//     PyObject *target_list = PyList_New(2);
//     PyList_SetItem(target_list, 0, PyLong_FromLong(111));
//     PyList_SetItem(target_list, 1, PyLong_FromLong(222));

//     PyTuple_SetItem(args, 0, target_list); // 添加到索引 0
//     PyTuple_SetItem(args, 1, PyLong_FromLong(-1)); // 添加到索引 1
//     PyTuple_SetItem(args, 2, PyLong_FromLong(1234)); // 添加到索引 2

//     auto res = pyascir::UtilsDurationRecord(nullptr, args, kwds);
//     EXPECT_EQ(res, nullptr);
//     Py_DECREF(target_list);
//   }

//   {
//     PyObject *target_list = PyList_New(2);
//     PyList_SetItem(target_list, 0, PyUnicode_FromString("codegen"));
//     PyList_SetItem(target_list, 1, PyUnicode_FromString("ascgraph_0"));

//     PyTuple_SetItem(args, 0, target_list); // 添加到索引 0
//     PyTuple_SetItem(args, 1, PyLong_FromLong(1234)); // 添加到索引 1
//     PyTuple_SetItem(args, 2, PyLong_FromLong(1234)); // 添加到索引 2

//     auto res = pyascir::UtilsDurationRecord(nullptr, args, kwds);
//     EXPECT_EQ(res, Py_None);
//     Py_DECREF(target_list);
//   }

//   Py_DECREF(args);
//   Py_DECREF(kwds);
// }