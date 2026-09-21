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

#include "macro_utils/dt_public_scope.h"

#include "common/single_op_parser.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/util.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/operator_factory_impl.h"

#include "macro_utils/dt_public_unscope.h"

using namespace std;

namespace ge {
class UtestOmg : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(UtestOmg, ParseSingleOpListTest_01) {
  std::string file;
  std::vector<SingleOpBuildParam> op_list;
  Status ret = SingleOpParser::ParseSingleOpList(file, op_list);
  EXPECT_EQ(ret, INTERNAL_ERROR);
}

TEST_F(UtestOmg, ParseSingleOpListTest_02) {
  std::string file = __FILE__;
  file = file.substr(0, file.rfind("/") + 1) + "Add_int32_1.json";
  stringstream sstream;
  sstream << R"(cat - << EOF > )" << file;
  sstream << R"(
[
    {
      "op": "Add",
      "input_desc": [
        {
          "format": "ND",
          "shape": [-1,-1],
          "shape_range": [[-1,-1],[-1,-1]],
          "type": "int32"
        },
        {
          "format": "ND",
          "shape": [-1,-1],
          "shape_range": [[-1,-1],[-1,-1]],
          "type": "int32"
        }
      ],
      "output_desc": [
        {
          "format": "ND",
          "shape": [-1,-1],
          "shape_range": [[-1,-1],[-1,-1]],
          "type": "int32"
        }
      ]
    },
    {
      "op": "Add",
      "input_desc": [
        {
          "format": "ND",
          "shape": [-1,-1],
          "shape_range": [[-1,-1],[-1,-1]],
          "type": "int32"
        },
        {
          "format": "ND",
          "shape": [-1,-1],
          "shape_range": [[-1,-1],[-1,-1]],
          "type": "int32"
        }
      ],
      "output_desc": [
        {
          "format": "ND",
          "shape": [-1,-1],
          "shape_range": [[-1,-1],[-1,-1]],
          "type": "int32"
        }
      ]
    }
])" << endl << R"(EOF)";
  system(sstream.str().c_str());
  std::vector<SingleOpBuildParam> op_list;
  Status ret = SingleOpParser::ParseSingleOpList(file, op_list);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_list.size(), 2);
  EXPECT_EQ(op_list[0].op_desc->GetName(), "Add");
  EXPECT_EQ(op_list[0].op_desc->GetType(), "Add");
  EXPECT_EQ(op_list[0].file_name, "0_Add_3_2_-1_-1_3_2_-1_-1_3_2_-1_-1.om");
  EXPECT_EQ(op_list[0].compile_flag, 0);
  EXPECT_EQ(op_list[1].op_desc->GetName(), "Add");
  EXPECT_EQ(op_list[1].op_desc->GetType(), "Add");
  EXPECT_EQ(op_list[1].file_name, "1_Add_3_2_-1_-1_3_2_-1_-1_3_2_-1_-1.om");
  EXPECT_EQ(op_list[1].compile_flag, 0);

  op_list.clear();
  ret = SingleOpParser::ParseSingleOpList(file, op_list);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_list.size(), 2);
  EXPECT_EQ(op_list[0].op_desc->GetName(), "Add");
  EXPECT_EQ(op_list[0].op_desc->GetType(), "Add");
  EXPECT_EQ(op_list[0].file_name, "0_Add_3_2_-1_-1_3_2_-1_-1_3_2_-1_-1.om");
  EXPECT_EQ(op_list[0].compile_flag, 0);
  EXPECT_EQ(op_list[1].op_desc->GetName(), "Add");
  EXPECT_EQ(op_list[1].op_desc->GetType(), "Add");
  EXPECT_EQ(op_list[1].file_name, "1_Add_3_2_-1_-1_3_2_-1_-1_3_2_-1_-1.om");
  EXPECT_EQ(op_list[1].compile_flag, 0);

  system(("rm " + file).c_str());
}

TEST_F(UtestOmg, ParseSingleOpListTest_03) {
  std::string file = __FILE__;
  file = file.substr(0, file.rfind("/") + 1) + "Add_int32_1.json";
  stringstream sstream;
  sstream << R"(cat - << EOF > )" << file;
  sstream << R"(
[
    {
      "op": "Add",
      "name": "FileName",
      "input_desc": [
        {
          "format": "ND",
          "shape": [-1,-1],
          "shape_range": [[-1,-1],[-1,-1]],
          "type": "int32"
        },
        {
          "format": "ND",
          "shape": [-1,-1],
          "shape_range": [[-1,-1],[-1,-1]],
          "type": "int32"
        }
      ],
      "output_desc": [
        {
          "format": "ND",
          "shape": [-1,-1],
          "shape_range": [[-1,-1],[-1,-1]],
          "type": "int32"
        }
      ]
    }
])" << endl << R"(EOF)";
  system(sstream.str().c_str());
  std::vector<SingleOpBuildParam> op_list;
  Status ret = SingleOpParser::ParseSingleOpList(file, op_list);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_list.size(), 1);
  EXPECT_EQ(op_list[0].op_desc->GetName(), "FileName");
  EXPECT_EQ(op_list[0].op_desc->GetType(), "Add");
  EXPECT_EQ(op_list[0].file_name, "FileName.om");
  EXPECT_EQ(op_list[0].compile_flag, 0);
  system(("rm " + file).c_str());
}

TEST_F(UtestOmg, ParseSingleOpListTest_04) {
  std::string file = __FILE__;
  file = file.substr(0, file.rfind("/") + 1) + "Add_int32_1.json";
  stringstream sstream;
  sstream << R"(cat - << EOF > )" << file;
  sstream << R"(
[
    {
      "op": "Add",
      "name": "FileName.om",
      "input_desc": [
        {
          "format": "ND",
          "shape": [-1,-1],
          "shape_range": [[-1,-1],[-1,-1]],
          "type": "int32"
        },
        {
          "format": "ND",
          "shape": [-1,-1],
          "shape_range": [[-1,-1],[-1,-1]],
          "type": "int32"
        }
      ],
      "output_desc": [
        {
          "format": "ND",
          "shape": [-1,-1],
          "shape_range": [[-1,-1],[-1,-1]],
          "type": "int32"
        }
      ]
    }
])" << endl << R"(EOF)";
  system(sstream.str().c_str());
  std::vector<SingleOpBuildParam> op_list;
  Status ret = SingleOpParser::ParseSingleOpList(file, op_list);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_list.size(), 1);
  EXPECT_EQ(op_list[0].op_desc->GetName(), "FileName.om");
  EXPECT_EQ(op_list[0].op_desc->GetType(), "Add");
  EXPECT_EQ(op_list[0].file_name, "FileName.om.om");
  EXPECT_EQ(op_list[0].compile_flag, 0);
  system(("rm " + file).c_str());
}

TEST_F(UtestOmg, ParseSingleOpListTest_05) {
  std::string file = __FILE__;
  file = file.substr(0, file.rfind("/") + 1) + "Add_int32_1.json";
  stringstream sstream;
  sstream << R"(cat - << EOF > )" << file;
  sstream << R"(
[
    {
      "op": "Add",
      "name": "#FileName",
      "input_desc": [
        {
          "format": "ND",
          "shape": [-1,-1],
          "shape_range": [[-1,-1],[-1,-1]],
          "type": "int32"
        },
        {
          "format": "ND",
          "shape": [-1,-1],
          "shape_range": [[-1,-1],[-1,-1]],
          "type": "int32"
        }
      ],
      "output_desc": [
        {
          "format": "ND",
          "shape": [-1,-1],
          "shape_range": [[-1,-1],[-1,-1]],
          "type": "int32"
        }
      ]
    }
])" << endl << R"(EOF)";
  system(sstream.str().c_str());
  std::vector<SingleOpBuildParam> op_list;
  Status ret = SingleOpParser::ParseSingleOpList(file, op_list);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_list.size(), 1);
  EXPECT_EQ(op_list[0].op_desc->GetName(), "#FileName");
  EXPECT_EQ(op_list[0].op_desc->GetType(), "Add");
  EXPECT_EQ(op_list[0].file_name, "0_Add_3_2_-1_-1_3_2_-1_-1_3_2_-1_-1.om");
  EXPECT_EQ(op_list[0].compile_flag, 0);
  system(("rm " + file).c_str());
}

TEST_F(UtestOmg, ParseSingleOpListTest_06) {
  std::string file = __FILE__;
  file = file.substr(0, file.rfind("/") + 1) + "Add_int32_1.json";
  stringstream sstream;
  sstream << R"(cat - << EOF > )" << file;
  sstream << R"(
[
    {
      "op": "Add",
      "name": "",
      "input_desc": [
        {
          "format": "ND",
          "shape": [-1,-1],
          "shape_range": [[-1,-1],[-1,-1]],
          "type": "int32"
        },
        {
          "format": "ND",
          "shape": [-1,-1],
          "shape_range": [[-1,-1],[-1,-1]],
          "type": "int32"
        }
      ],
      "output_desc": [
        {
          "format": "ND",
          "shape": [-1,-1],
          "shape_range": [[-1,-1],[-1,-1]],
          "type": "int32"
        }
      ]
    }
])" << endl << R"(EOF)";
  system(sstream.str().c_str());
  std::vector<SingleOpBuildParam> op_list;
  Status ret = SingleOpParser::ParseSingleOpList(file, op_list);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_list.size(), 1);
  EXPECT_EQ(op_list[0].op_desc->GetName(), "Add");
  EXPECT_EQ(op_list[0].op_desc->GetType(), "Add");
  EXPECT_EQ(op_list[0].file_name, "0_Add_3_2_-1_-1_3_2_-1_-1_3_2_-1_-1.om");
  EXPECT_EQ(op_list[0].compile_flag, 0);
  system(("rm " + file).c_str());
}

TEST_F(UtestOmg, ParseSingleOpListTest_07) {
  std::string file = __FILE__;
  file = file.substr(0, file.rfind("/") + 1) + "Add_int32_1.json";
  stringstream sstream;
  sstream << R"(cat - << EOF > )" << file;
  sstream << R"(
[
    {
      "op": "Add",
      "name": ".",
      "input_desc": [
        {
          "format": "ND",
          "shape": [-1,-1],
          "shape_range": [[-1,-1],[-1,-1]],
          "type": "int32"
        },
        {
          "format": "ND",
          "shape": [-1,-1],
          "shape_range": [[-1,-1],[-1,-1]],
          "type": "int32"
        }
      ],
      "output_desc": [
        {
          "format": "ND",
          "shape": [-1,-1],
          "shape_range": [[-1,-1],[-1,-1]],
          "type": "int32"
        }
      ]
    }
])" << endl << R"(EOF)";
  system(sstream.str().c_str());
  std::vector<SingleOpBuildParam> op_list;
  Status ret = SingleOpParser::ParseSingleOpList(file, op_list);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_list.size(), 1);
  EXPECT_EQ(op_list[0].op_desc->GetName(), ".");
  EXPECT_EQ(op_list[0].op_desc->GetType(), "Add");
  EXPECT_EQ(op_list[0].file_name, "0_Add_3_2_-1_-1_3_2_-1_-1_3_2_-1_-1.om");
  EXPECT_EQ(op_list[0].compile_flag, 0);
  system(("rm " + file).c_str());
}

TEST_F(UtestOmg, ParseSingleOpListTest_08) {
  std::string file = __FILE__;
  file = file.substr(0, file.rfind("/") + 1) + "Add_int32_1.json";
  stringstream sstream;
  sstream << R"(cat - << EOF > )" << file;
  sstream << R"(
[
    {
      "op": "Add",
      "name": "123456789a123456789b123456789c123456789d123456789e123456789f123456789g123456789h123456789i123456789j123456789k123456789l123456789m",
      "input_desc": [
        {
          "format": "ND",
          "shape": [-1,-1],
          "shape_range": [[-1,-1],[-1,-1]],
          "type": "int32"
        },
        {
          "format": "ND",
          "shape": [-1,-1],
          "shape_range": [[-1,-1],[-1,-1]],
          "type": "int32"
        }
      ],
      "output_desc": [
        {
          "format": "ND",
          "shape": [-1,-1],
          "shape_range": [[-1,-1],[-1,-1]],
          "type": "int32"
        }
      ]
    }
])" << endl << R"(EOF)";
  system(sstream.str().c_str());
  std::vector<SingleOpBuildParam> op_list;
  Status ret = SingleOpParser::ParseSingleOpList(file, op_list);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_list.size(), 1);
  EXPECT_EQ(op_list[0].op_desc->GetName(), "123456789a123456789b123456789c123456789d123456789e123456789f123456789g123456789h123456789i123456789j123456789k123456789l123456789m");
  EXPECT_EQ(op_list[0].op_desc->GetType(), "Add");
  EXPECT_EQ(op_list[0].file_name, "123456789a123456789b123456789c123456789d123456789e123456789f123456789g123456789h123456789i123456789j123456789k123456789l12345678.om");
  EXPECT_EQ(op_list[0].compile_flag, 0);
  system(("rm " + file).c_str());
}

TEST_F(UtestOmg, SetShapeRangeTest) {
  std::string op_name;
  SingleOpTensorDesc tensor_desc;
  tensor_desc.dims.push_back(1);
  tensor_desc.dims.push_back(-2);
  GeTensorDesc ge_tensor_desc;
  Status ret = SingleOpParser::SetShapeRange(op_name, tensor_desc, ge_tensor_desc);
  EXPECT_EQ(ret, PARAM_INVALID);

  tensor_desc.dims.clear();
  tensor_desc.dims.push_back(-2);
  std::vector<int64_t> ranges = {1, 2, 3};
  tensor_desc.dim_ranges.push_back(ranges);
  ret = SingleOpParser::SetShapeRange(op_name, tensor_desc, ge_tensor_desc);
  EXPECT_EQ(ret, PARAM_INVALID);

  tensor_desc.dim_ranges.clear();
  ret = SingleOpParser::SetShapeRange(op_name, tensor_desc, ge_tensor_desc);
  EXPECT_EQ(ret, SUCCESS);

  // dim < 0, range_index = num_shape_ranges
  tensor_desc.dims.clear();
  tensor_desc.dims.push_back(-1);
  ret = SingleOpParser::SetShapeRange(op_name, tensor_desc, ge_tensor_desc);
  EXPECT_EQ(ret, PARAM_INVALID);

  // dim < 0, range.size() != 2
  std::vector<int64_t> ranges2 = {1, 2, 3};
  tensor_desc.dim_ranges.push_back(ranges2);
  ret = SingleOpParser::SetShapeRange(op_name, tensor_desc, ge_tensor_desc);
  EXPECT_EQ(ret, PARAM_INVALID);

  // success
  tensor_desc.dim_ranges.clear();
  std::vector<int64_t> ranges3 = {1, 2};
  tensor_desc.dim_ranges.push_back(ranges3);
  ret = SingleOpParser::SetShapeRange(op_name, tensor_desc, ge_tensor_desc);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestOmg, ValidateTest) {
  //input_desc:!tensor_desc.GetValidFlag()
  SingleOpDesc op_desc;
  op_desc.op = "test";
  SingleOpTensorDesc input_tensor_desc;
  input_tensor_desc.is_valid = false;
  op_desc.input_desc.push_back(input_tensor_desc);
  bool ret = SingleOpParser::Validate(op_desc);
  EXPECT_EQ(ret, false);

  //input_desc:tensor_desc.type == DT_UNDEFINED && tensor_desc.format != FORMAT_RESERVED
  op_desc.input_desc.clear();
  input_tensor_desc.is_valid = true;
  input_tensor_desc.type = ge::DT_UNDEFINED;
  input_tensor_desc.format = ge::FORMAT_FRACTAL_Z_G;
  op_desc.input_desc.push_back(input_tensor_desc);
  ret = SingleOpParser::Validate(op_desc);
  EXPECT_EQ(ret, false);

  //output_desc:!tensor_desc.GetValidFlag()
  op_desc.input_desc.clear();
  SingleOpTensorDesc output_tensor_desc;
  output_tensor_desc.is_valid = false;
  op_desc.output_desc.push_back(output_tensor_desc);
  ret = SingleOpParser::Validate(op_desc);
  EXPECT_EQ(ret, false);

  //output_desc:tensor_desc.type == DT_UNDEFINED
  op_desc.output_desc.clear();
  output_tensor_desc.is_valid = true;
  output_tensor_desc.type = ge::DT_UNDEFINED;
  op_desc.output_desc.push_back(output_tensor_desc);
  ret = SingleOpParser::Validate(op_desc);
  EXPECT_EQ(ret, false);

  //output_desc:tensor_desc.format == FORMAT_RESERVED
  op_desc.output_desc.clear();
  output_tensor_desc.is_valid = true;
  output_tensor_desc.type = ge::DT_FLOAT16;
  output_tensor_desc.format = ge::FORMAT_RESERVED;
  op_desc.output_desc.push_back(output_tensor_desc);
  ret = SingleOpParser::Validate(op_desc);
  EXPECT_EQ(ret, false);

  //attr.name.empty()
  op_desc.output_desc.clear();
  SingleOpAttr op_attr;
  op_desc.attrs.push_back(op_attr);
  ret = SingleOpParser::Validate(op_desc);
  EXPECT_EQ(ret, false);

  //attr.value.IsEmpty()
  op_desc.attrs.clear();
  op_attr.name = "test1";
  op_desc.attrs.push_back(op_attr);
  ret = SingleOpParser::Validate(op_desc);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestOmg, UpdateDynamicTensorNameTest) {
  SingleOpTensorDesc tensor_desc;
  tensor_desc.dynamic_input_name = "test1";
  std::vector<SingleOpTensorDesc> desc;
  desc.push_back(tensor_desc);
  Status ret = SingleOpParser::UpdateDynamicTensorName(desc);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestOmg, ConvertToBuildParamTest) {
  int32_t index = 0;
  SingleOpDesc single_op_desc;
  single_op_desc.op = "test1";

  SingleOpTensorDesc input_tensor_desc;
  input_tensor_desc.name = "test2";
  input_tensor_desc.dims.push_back(1);
  input_tensor_desc.format = ge::FORMAT_RESERVED;
  input_tensor_desc.type = ge::DT_UNDEFINED;
  input_tensor_desc.is_const = true;
  single_op_desc.input_desc.push_back(input_tensor_desc);

  SingleOpTensorDesc output_tensor_desc;
  output_tensor_desc.name = "test3";
  output_tensor_desc.dims.push_back(2);
  output_tensor_desc.format = ge::FORMAT_RESERVED;
  output_tensor_desc.type = ge::DT_UNDEFINED;
  single_op_desc.output_desc.push_back(output_tensor_desc);

  SingleOpAttr op_attr;
  op_attr.name = "attr";
  single_op_desc.attrs.push_back(op_attr);

  SingleOpBuildParam build_param;
  Status ret = SingleOpParser::ConvertToBuildParam(index, single_op_desc, build_param);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestOmg, VerifyOpInputOutputSizeByIrTest) {
  nlohmann::json j;
  SingleOpAttr op_attr;

  //it == kAttrTypeDict.end()
  j["name"] = "test";
  j["type"] = "test";
  j["value"] = "";
  from_json(j, op_attr);
  EXPECT_EQ(op_attr.name, "test");
  EXPECT_EQ(op_attr.type, "test");

  //VT_BOOL
  j.clear();
  j["name"] = "test2";
  j["type"] = "bool";
  j["value"] = true;
  from_json(j, op_attr);
  EXPECT_EQ(op_attr.name, "test2");
  EXPECT_EQ(op_attr.type, "bool");

  //VT_INT
  j.clear();
  j["name"] = "test3";
  j["type"] = "int";
  j["value"] = 1;
  from_json(j, op_attr);
  EXPECT_EQ(op_attr.name, "test3");
  EXPECT_EQ(op_attr.type, "int");

  //VT_FLOAT
  j.clear();
  j["name"] = "test4";
  j["type"] = "float";
  j["value"] = 3.14;
  from_json(j, op_attr);
  EXPECT_EQ(op_attr.name, "test4");
  EXPECT_EQ(op_attr.type, "float");

  //VT_STRING
  j.clear();
  j["name"] = "test5";
  j["type"] = "string";
  j["value"] = "strval";
  from_json(j, op_attr);
  EXPECT_EQ(op_attr.name, "test5");
  EXPECT_EQ(op_attr.type, "string");

  //VT_LIST_BOOL
  j.clear();
  j["name"] = "test6";
  j["type"] = "list_bool";
  j["value"] = {true};
  from_json(j, op_attr);
  EXPECT_EQ(op_attr.name, "test6");
  EXPECT_EQ(op_attr.type, "list_bool");

  //VT_LIST_INT
  j.clear();
  j["name"] = "test7";
  j["type"] = "list_int";
  j["value"] = {1};
  from_json(j, op_attr);
  EXPECT_EQ(op_attr.name, "test7");
  EXPECT_EQ(op_attr.type, "list_int");

  //VT_LIST_FLOAT
  j.clear();
  j["name"] = "test8";
  j["type"] = "list_float";
  j["value"] = {3.14};
  from_json(j, op_attr);
  EXPECT_EQ(op_attr.name, "test8");
  EXPECT_EQ(op_attr.type, "list_float");

  //VT_LIST_STRING
  j.clear();
  j["name"] = "test9";
  j["type"] = "list_string";
  j["value"] = {"strval"};
  from_json(j, op_attr);
  EXPECT_EQ(op_attr.name, "test9");
  EXPECT_EQ(op_attr.type, "list_string");

  //VT_LIST_LIST_INT
  j.clear();
  j["name"] = "test10";
  j["type"] = "list_list_int";
  j["value"] = {{1}};
  from_json(j, op_attr);
  EXPECT_EQ(op_attr.name, "test10");
  EXPECT_EQ(op_attr.type, "list_list_int");

  //VT_DATA_TYPE
  j.clear();
  j["name"] = "test11";
  j["type"] = "data_type";
  j["value"] = "uint8";
  from_json(j, op_attr);
  EXPECT_EQ(op_attr.name, "test11");
  EXPECT_EQ(op_attr.type, "data_type");
}

TEST_F(UtestOmg, TransConstValueTest) {
  std::string type_str;
  nlohmann::json j;
  SingleOpTensorDesc desc;

  //DT_INT8
  desc.type = ge::DT_INT8;
  j["const_value"] = {1, 2};
  TransConstValue(type_str, j, desc);
  EXPECT_NE(desc.const_value_size, 0);

  //DT_INT16
  desc.const_value_size = 0;
  desc.type = ge::DT_INT16;
  j.clear();
  j["const_value"] = {1, 2};
  TransConstValue(type_str, j, desc);
  EXPECT_NE(desc.const_value_size, 0);

  //DT_INT32
  desc.const_value_size = 0;
  desc.type = ge::DT_INT32;
  j.clear();
  j["const_value"] = {1, 2};
  TransConstValue(type_str, j, desc);
  EXPECT_NE(desc.const_value_size, 0);

  //DT_INT64
  desc.const_value_size = 0;
  desc.type = ge::DT_INT64;
  j.clear();
  j["const_value"] = {1, 2};
  TransConstValue(type_str, j, desc);
  EXPECT_NE(desc.const_value_size, 0);

  //DT_FLOAT16
  desc.const_value_size = 0;
  desc.type = ge::DT_FLOAT16;
  j.clear();
  j["const_value"] = {1.1, 2.2};
  TransConstValue(type_str, j, desc);
  EXPECT_NE(desc.const_value_size, 0);

  //DT_FLOAT
  desc.const_value_size = 0;
  desc.type = ge::DT_FLOAT;
  j.clear();
  j["const_value"] = {3.14};
  TransConstValue(type_str, j, desc);
  EXPECT_NE(desc.const_value_size, 0);

  //DT_DOUBLE
  desc.const_value_size = 0;
  desc.type = ge::DT_DOUBLE;
  j.clear();
  j["const_value"] = {2.0};
  TransConstValue(type_str, j, desc);
  EXPECT_NE(desc.const_value_size, 0);
}

} // namespace
