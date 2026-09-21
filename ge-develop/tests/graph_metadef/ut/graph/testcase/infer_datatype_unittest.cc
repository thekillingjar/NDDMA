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
#include <memory>
#include <dlfcn.h>

#include "graph/op_desc.h"
#include "graph/normal_graph/op_desc_impl.h"
#include "graph_builder_utils.h"
#include "graph/operator_reg.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/operator_factory_impl.h"
#include "graph/compute_graph.h"
#include "graph/utils/recover_ir_utils.h"
#include "dlog_pub.h"

namespace ge {
class UTInferDataType : public testing::Test {
 protected:
  void SetUp() {
    dlog_setlevel(0, 0, 0);
  }

  void TearDown() {
    dlog_setlevel(0, 3, 0);
  }
};

class OpDtypeInfer {
 public:
  struct TypeOrTypes {
    explicit TypeOrTypes(const DataType &expect_type) : dynamic(false), types({expect_type}) {}
    explicit TypeOrTypes(const std::vector<DataType> &expect_types) : dynamic(true), types(expect_types) {}

    bool dynamic = false;
    std::vector<DataType> types;
  };

  explicit OpDtypeInfer(const std::string &type) {
    auto op = OperatorFactory::CreateOperator(type, type);
    desc_ = OpDescUtils::GetOpDescFromOperator(op);
  }

  explicit OpDtypeInfer(const OpDescPtr &desc) : desc_(desc) {}

  OpDtypeInfer &Input(const DataType &type) {
    int32_t ir_index = ++index_;
    auto format = FORMAT_ND;
    if (type == DT_UNDEFINED) {
      format = FORMAT_RESERVED;
    }
    desc_->UpdateInputDesc("input" + std::to_string(ir_index), GeTensorDesc(GeShape(), format, type));
    return *this;
  }

  OpDtypeInfer &Input(const std::initializer_list<DataType> &raw_types) {
    std::vector<DataType> types(raw_types);
    int32_t ir_index = ++index_;
    desc_->AddDynamicInputDesc("input" + std::to_string(ir_index), types.size());
    for (size_t i = 0U; i < types.size(); ++i) {
      desc_->UpdateInputDesc("input" + std::to_string(ir_index) + std::to_string(i),
                             GeTensorDesc(GeShape(), FORMAT_ND, types[i]));
    }
    return *this;
  }

  OpDtypeInfer &Attr(const std::string &attr, const std::vector<DataType> &types) {
    AttrUtils::SetListDataType(desc_, attr, types);
    return *this;
  }

  OpDtypeInfer &Attr(const std::string &attr, const std::vector<int32_t> &types) {
    AttrUtils::SetListInt(desc_, attr, types);
    return *this;
  }

  OpDtypeInfer &Attr(const std::string &attr, int32_t type) {
    AttrUtils::SetInt(desc_, attr, type);
    return *this;
  }

  OpDtypeInfer &Attr(const std::string &attr, DataType type) {
    AttrUtils::SetDataType(desc_, attr, type);
    return *this;
  }

  OpDtypeInfer &Expect(const DataType &type) {
    expect_dtypes_.emplace_back(type);
    return *this;
  }

  OpDtypeInfer &Expect(const std::vector<DataType> &types) {
    desc_->AddDynamicOutputDesc("output" + std::to_string(expect_dtypes_.size() + 1U), types.size());
    expect_dtypes_.emplace_back(types);
    return *this;
  }

  void AssertSucceed() {
    ASSERT_EQ(desc_->SymbolicInferDataType(), GRAPH_SUCCESS);
    for (size_t i = 0U; i < expect_dtypes_.size(); ++i) {
      std::string ir_output = "output" + std::to_string(i + 1);
      if (!expect_dtypes_[i].dynamic) {
        ASSERT_EQ(TypeUtils::DataTypeToSerialString(desc_->GetOutputDesc(ir_output).GetDataType()),
                  TypeUtils::DataTypeToSerialString(expect_dtypes_[i].types[0]));
      } else {
        for (size_t j = 0U; j < expect_dtypes_[i].types.size(); ++j) {
          std::string ir_output_index = ir_output + std::to_string(j);
          ASSERT_EQ(TypeUtils::DataTypeToSerialString(desc_->GetOutputDesc(ir_output_index).GetDataType()),
                    TypeUtils::DataTypeToSerialString(expect_dtypes_[i].types[j]));
        }
      }
    }
  }

  void AssertFailed() {
    ASSERT_NE(desc_->SymbolicInferDataType(), GRAPH_SUCCESS);
  }

 private:
  std::vector<TypeOrTypes> expect_dtypes_;
  int32_t index_ = 0;
  OpDescPtr desc_;
};

/* ---------- 基于符号进行推导的基础用例 ---------- */
REG_OP(Op1)
    .OPTIONAL_INPUT(input1, "T")
    .INPUT(input2, "T")
    .DYNAMIC_INPUT(input3, "T")
    .OUTPUT(output1, "T")
    .DYNAMIC_OUTPUT(output2, "T")
    .DATATYPE(T, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(Op1);
TEST_F(UTInferDataType, sym_infer_from_regular_input_succeed) {
  OpDtypeInfer("Op1")  // T全部全部传入
      .Input(DT_FLOAT16)
      .Input(DT_FLOAT16)
      .Input({DT_FLOAT16, DT_FLOAT16})
      .Expect(DT_FLOAT16)
      .Expect({DT_FLOAT16, DT_FLOAT16})
      .AssertSucceed();
}
TEST_F(UTInferDataType, sym_infer_from_regular_input_unfed_opt) {
  OpDtypeInfer("Op1")  // 可选输入不传入，根据其他输入推导
      .Input(DT_UNDEFINED)
      .Input(DT_FLOAT16)
      .Input({DT_FLOAT16, DT_FLOAT16})
      .Expect(DT_FLOAT16)
      .Expect({DT_FLOAT16, DT_FLOAT16})
      .AssertSucceed();
}
TEST_F(UTInferDataType, sym_infer_from_regular_input_only_require) {
  OpDtypeInfer("Op1")  // 可选和动态都不传入，根据其他输入推导
      .Input(DT_UNDEFINED)
      .Input(DT_FLOAT16)
      .Input({})
      .Expect(DT_FLOAT16)
      .Expect({DT_FLOAT16, DT_FLOAT16})
      .AssertSucceed();
}
TEST_F(UTInferDataType, sym_infer_from_regular_input_dtype_out_of_range) {
  OpDtypeInfer("Op1")  // 类型不在可选范围内
      .Input(DT_INT32)
      .Input(DT_INT32)
      .Input({DT_INT32, DT_INT32})
      .AssertFailed();
}
TEST_F(UTInferDataType, sym_infer_from_regular_input_dtype_mismatch_between_ir_inputs) {
  OpDtypeInfer("Op1")  // 两个IR输入类型不一致
      .Input(DT_FLOAT16)
      .Input(DT_FLOAT)
      .Input({DT_FLOAT16, DT_FLOAT16})
      .AssertFailed();
}
TEST_F(UTInferDataType, sym_infer_from_regular_input_dtype_mismatch_in_dyn) {
  OpDtypeInfer("Op1")  // 动态输入中的多个类型不一致
      .Input(DT_FLOAT16)
      .Input(DT_FLOAT16)
      .Input({DT_FLOAT16, DT_FLOAT})
      .AssertFailed();
}

/* ---------- 基于可选输入进行推导 ---------- */
REG_OP(Op2)
    .OPTIONAL_INPUT(input1, "T")
    .OUTPUT(output1, "T")
    .DYNAMIC_OUTPUT(output2, "T")
    .DATATYPE(T, TensorType({DT_FLOAT16}))
    .OP_END_FACTORY_REG(Op2);
TEST_F(UTInferDataType, sym_infer_from_optional_input_succeed) {
  OpDtypeInfer("Op2")  // 可选输入传入
      .Input(DT_FLOAT16)
      .Expect(DT_FLOAT16)
      .Expect({DT_FLOAT16, DT_FLOAT16})
      .AssertSucceed();
}
TEST_F(UTInferDataType, sym_infer_from_optional_input_unfed_opt) {
  EXPECT_NO_THROW(
    OpDtypeInfer("Op2")  // 可选不传入，无法推导
        .Input(DT_UNDEFINED)
        .AssertFailed();
  );
}

/* ---------- 基于动态输入进行推导 ---------- */
REG_OP(Op3)
    .DYNAMIC_INPUT(input1, "T")
    .OUTPUT(output1, "T")
    .DYNAMIC_OUTPUT(output2, "T")
    .DATATYPE(T, TensorType({DT_FLOAT16}))
    .OP_END_FACTORY_REG(Op3);
TEST_F(UTInferDataType, sym_infer_from_dynamic_input_succeed) {
  OpDtypeInfer("Op3")  // 动态输入不为空
      .Input({DT_FLOAT16})
      .Expect(DT_FLOAT16)
      .Expect({DT_FLOAT16, DT_FLOAT16})
      .AssertSucceed();
}
TEST_F(UTInferDataType, sym_infer_from_dynamic_input_unfed_dyn) {
  OpDtypeInfer("Op3")  // 可选不传入，无法推导
      .Input({})
      .AssertFailed();
}

/* ---------- 基于属性进行推导 ---------- */
REG_OP(Op4)
    .REQUIRED_ATTR(dtype1, Int)
    .REQUIRED_ATTR(dtype2, Type)
    .REQUIRED_ATTR(dtype3, ListInt)
    .REQUIRED_ATTR(dtype4, ListType)
    .OUTPUT(output1, "dtype1")
    .OUTPUT(output2, "dtype2")
    .DYNAMIC_OUTPUT(output3, "dtype3")  // 动态输出被List属性指定
    .DYNAMIC_OUTPUT(output4, "dtype4")
    .DYNAMIC_OUTPUT(output5, "dtype1")  // 动态输出被单个属性指定
    .DYNAMIC_OUTPUT(output6, "dtype2")
    .DATATYPE(dtype1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .DATATYPE(dtype2, TensorType({DT_FLOAT16, DT_FLOAT}))
    .DATATYPE(dtype3, ListTensorType({DT_FLOAT16, DT_FLOAT}))
    .DATATYPE(dtype4, ListTensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(Op4);
TEST_F(UTInferDataType, sym_infer_from_attr_succeed) {
  OpDtypeInfer("Op4")  // 根据属性进行推导
      .Attr("dtype1", int32_t(DT_FLOAT16))
      .Attr("dtype2", DT_FLOAT16)
      .Attr("dtype3", std::vector<int32_t>{DT_FLOAT16, DT_FLOAT})
      .Attr("dtype4", std::vector<DataType>{DT_FLOAT16, DT_FLOAT})
      .Expect(DT_FLOAT16)
      .Expect(DT_FLOAT16)
      .Expect({DT_FLOAT16, DT_FLOAT})  // 动态输出被List属性指定
      .Expect({DT_FLOAT16, DT_FLOAT})
      .Expect({DT_FLOAT16, DT_FLOAT16})  // 动态输出被单个属性指定
      .Expect({DT_FLOAT16, DT_FLOAT16})
      .AssertSucceed();
}
TEST_F(UTInferDataType, sym_infer_from_attr_dtype_out_of_range) {
  OpDtypeInfer("Op4")  // 属性不在允许范围内
      .Attr("dtype1", int32_t(DT_FLOAT16))
      .Attr("dtype2", DT_INT32)  // 非法输入类型
      .Attr("dtype3", std::vector<int32_t>{DT_FLOAT16, DT_FLOAT})
      .Attr("dtype4", std::vector<DataType>{DT_FLOAT16, DT_FLOAT})
      .AssertFailed();
}
TEST_F(UTInferDataType, sym_infer_from_attr_list_dtype_out_of_range) {
  OpDtypeInfer("Op4")  // 属性不在允许范围内
      .Attr("dtype1", int32_t(DT_FLOAT16))
      .Attr("dtype2", DT_FLOAT16)
      .Attr("dtype3", std::vector<int32_t>{DT_FLOAT16, DT_FLOAT})
      .Attr("dtype4", std::vector<DataType>{DT_FLOAT16, DT_INT32})  // 非法输入类型
      .AssertFailed();
}

TEST_F(UTInferDataType, sym_infer_from_attr_but_type_mismatch_1) {
  OpDtypeInfer("Op4")
      .Attr("dtype1", int32_t(DT_FLOAT16))
      .Attr("dtype2", std::vector<int32_t>{DT_FLOAT16, DT_FLOAT})  // 需要单个类型，但是传入List
      .Attr("dtype3", std::vector<int32_t>{DT_FLOAT16, DT_FLOAT})
      .Attr("dtype4", std::vector<DataType>{DT_FLOAT16, DT_FLOAT})
      .AssertFailed();
}

TEST_F(UTInferDataType, sym_infer_from_attr_but_type_mismatch_2) {
  OpDtypeInfer("Op4")
      .Attr("dtype1", int32_t(DT_FLOAT16))
      .Attr("dtype2", DT_FLOAT16)
      .Attr("dtype3", DT_FLOAT16)  // 需要List类型，但是传入单个类型
      .Attr("dtype4", std::vector<DataType>{DT_FLOAT16, DT_FLOAT})
      .AssertFailed();
}

/* ---------- 输出类型唯一场景，支持推导（类似Equal算子固定输出bool） ---------- */
REG_OP(Op6)
    .OUTPUT(output1, "T")
    .DYNAMIC_OUTPUT(output2, "T")
    .DATATYPE(T, TensorType({DT_FLOAT16}))
    .OP_END_FACTORY_REG(Op6);
TEST_F(UTInferDataType, sym_infer_for_const_output_dtype) {  // 老旧方式注册，但是输出类型唯一
  OpDtypeInfer("Op6").Expect(DT_FLOAT16).Expect({DT_FLOAT16, DT_FLOAT16}).AssertSucceed();
}

/* ---------- ListTensorType的类型推导 ---------- */
REG_OP(Op7)
    .DYNAMIC_INPUT(input1, "T")
    .DYNAMIC_INPUT(input2, "T")
    .DYNAMIC_OUTPUT(output1, "T")
    .DATATYPE(T, ListTensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(Op7);
TEST_F(UTInferDataType, sym_infer_for_list_dtype_succeed) {
  OpDtypeInfer("Op7")  // 正常推导
      .Input({DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT16})
      .Input({DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT16})
      .Expect({DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT16})
      .AssertSucceed();
}
TEST_F(UTInferDataType, sym_infer_for_list_dtype_dtype_mismatch_between_dyn) {
  OpDtypeInfer("Op7")  // 对应同一个ListType sym的两个输入，类型合法但是不一致
      .Input({DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT16})
      .Input({DT_FLOAT16, DT_FLOAT, DT_FLOAT16, DT_FLOAT16})  // 第三个输入类型不一致
      .Expect({DT_FLOAT16, DT_FLOAT, DT_FLOAT, DT_FLOAT16})
      .AssertFailed();
}
TEST_F(UTInferDataType, sym_infer_for_list_dtype_dtype_out_of_range) {
  OpDtypeInfer("Op7")  // 数据类型不在范围内
      .Input({DT_FLOAT16, DT_INT32})
      .Input({DT_FLOAT16, DT_INT32})
      .AssertFailed();
}

/* ---------- 类型提升方式推导 ---------- */
// 基础类型间提升
REG_OP(Op8)
    .INPUT(input1, "T1")
    .DYNAMIC_INPUT(input2, "T2")
    .INPUT(input3, "T3")
    .OUTPUT(output1, "T4")
    .DYNAMIC_OUTPUT(output2, "T5")
    .DATATYPE(T1, TensorType({DT_INT32, DT_FLOAT}))
    .DATATYPE(T2, TensorType({DT_INT64, DT_FLOAT}))
    .DATATYPE(T3, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DATATYPE(T4, Promote({"T1", "T2"}))
    .DATATYPE(T5, Promote({"T1", "T2", "T3"}))
    .OP_END_FACTORY_REG(Op8);
TEST_F(UTInferDataType, sym_infer_for_dtype_promotion_succeed) {
  OpDtypeInfer("Op8")
      .Input(DT_INT32)
      .Input({DT_INT64, DT_INT64})
      .Input(DT_FLOAT)
      .Expect(DT_INT64)              // T1和T2间提升为DT_INT64
      .Expect({DT_FLOAT, DT_FLOAT})  // T1，T2和T3间提升为DT_FLOAT
      .AssertSucceed();
}

// ListTensorType间提升
REG_OP(Op9)
    .DYNAMIC_INPUT(input1, "T1")
    .DYNAMIC_INPUT(input2, "T2")
    .DYNAMIC_INPUT(input3, "T3")
    .DYNAMIC_OUTPUT(output1, "T4")
    .DYNAMIC_OUTPUT(output2, "T5")
    .DATATYPE(T1, ListTensorType({DT_INT32, DT_FLOAT}))
    .DATATYPE(T2, ListTensorType({DT_INT64, DT_FLOAT}))
    .DATATYPE(T3, ListTensorType({DT_FLOAT, DT_FLOAT16}))
    .DATATYPE(T4, Promote({"T1", "T2"}))
    .DATATYPE(T5, Promote({"T1", "T2", "T3"}))
    .OP_END_FACTORY_REG(Op9);
TEST_F(UTInferDataType, sym_infer_for_dtype_promotion_list_type_succeed) {
  OpDtypeInfer("Op9")
      .Input({DT_INT32, DT_FLOAT})    // T1
      .Input({DT_INT64, DT_INT64})    // T2
      .Input({DT_FLOAT, DT_FLOAT16})  // T3
      .Expect({DT_INT64, DT_FLOAT})   // T1和T2间逐个提升
      .Expect({DT_FLOAT, DT_FLOAT})   // T1，T2和T3间逐个提升
      .AssertSucceed();
}
TEST_F(UTInferDataType, sym_infer_for_dtype_promotion_list_type_dtype_size_mismatch) {
  OpDtypeInfer("Op9")                         // 提升失败，数量不一致
      .Input({DT_INT32, DT_FLOAT, DT_FLOAT})  // T1
      .Input({DT_INT64, DT_INT64})            // T2
      .Input({DT_FLOAT, DT_FLOAT16})          // T3
      .AssertFailed();
}

// 试图在无提升规则的单类型间提升
REG_OP(Op14)
    .INPUT(input1, "T1")
    .INPUT(input2, "T2")
    .OUTPUT(output1, "T3")
    .DATATYPE(T1, ListTensorType({DT_INT32, DT_FLOAT}))
    .DATATYPE(T2, ListTensorType({DT_INT64, DT_FLOAT}))
    .DATATYPE(T3, Promote({"T1", "T2"}))
    .OP_END_FACTORY_REG(Op14);
TEST_F(UTInferDataType, sym_infer_for_dtype_promotion_unpromotable_types) {
  EXPECT_NO_THROW(
    OpDtypeInfer("Op14")     // 提升失败，无提升规则
        .Input(DT_VARIANT)   // T1
        .Input(DT_RESOURCE)  // T2
        .AssertFailed();
  );
}
// 试图在无提升规则的List类型间提升
REG_OP(Op15)
    .DYNAMIC_INPUT(input1, "T1")
    .DYNAMIC_INPUT(input2, "T2")
    .OUTPUT(output1, "T3")
    .DATATYPE(T1, ListTensorType({DT_FLOAT16, DT_VARIANT}))
    .DATATYPE(T2, ListTensorType({DT_FLOAT, DT_RESOURCE}))
    .DATATYPE(T3, Promote({"T1", "T2"}))
    .OP_END_FACTORY_REG(Op15);
TEST_F(UTInferDataType, sym_infer_for_dtype_promotion_unpromotable_list_types) {
  OpDtypeInfer("Op15")                  // 提升失败，ListType中的某个无提升规则
      .Input({DT_FLOAT16, DT_VARIANT})  // T1
      .Input({DT_FLOAT, DT_RESOURCE})   // T2
      .AssertFailed();
}

/* ---------- 异常IR注册校验能力 ---------- */
REG_OP(Op10)
    .INPUT(input1, "T1")
    .OUTPUT(output1, "T2")  // 异常IR注册，PromoteDtype中只有一个类型
    .DATATYPE(T1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .DATATYPE(T2, Promote({"T1"}))
    .OP_END_FACTORY_REG(Op10);
TEST_F(UTInferDataType, sym_infer_for_dtype_promotion_3) {
  EXPECT_NO_THROW(OpDtypeInfer("Op10").Input(DT_FLOAT16).AssertFailed());
}

REG_OP(Op11)
    .INPUT(input1, "T1")
    .DYNAMIC_INPUT(input2, "T2")
    .OUTPUT(output1, "T3")  // 异常IR注册，TensorType和ListTensorType间试图提升
    .DATATYPE(T1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .DATATYPE(T2, ListTensorType({DT_INT64, DT_FLOAT}))
    .DATATYPE(T3, Promote({"T1", "T2"}))
    .OP_END_FACTORY_REG(Op11);
TEST_F(UTInferDataType, sym_infer_for_dtype_promotion_4) {
  OpDtypeInfer("Op11").Input(DT_FLOAT16).Input({DT_INT64, DT_FLOAT}).AssertFailed();
}

REG_OP(Op12)
    .OUTPUT(output1, ListTensorType({DT_FLOAT16, DT_FLOAT}))  // 输出为老旧方式注册的ListTensorType
    .OP_END_FACTORY_REG(Op12);
TEST_F(UTInferDataType, sym_infer_for_legacy_list_type_output) {
  EXPECT_NO_THROW(OpDtypeInfer("Op12").AssertFailed());
}

/* ---------- 符号方式注册时的类型校验能力 ---------- */
// 符号不用于任何类型推导
REG_OP(Op13)
    .INPUT(input1, "T1")
    .OPTIONAL_INPUT(input2, "T2")
    .DYNAMIC_INPUT(input3, "T3")
    .DYNAMIC_INPUT(input4, "T4")
    .DATATYPE(T1, TensorType({DT_FLOAT16}))
    .DATATYPE(T2, TensorType({DT_FLOAT16}))
    .DATATYPE(T3, TensorType({DT_FLOAT16, DT_FLOAT}))
    .DATATYPE(T4, ListTensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(Op13);
TEST_F(UTInferDataType, sym_infer_for_type_check_for_unused_sym_succeed) {
  OpDtypeInfer("Op13")
      .Input(DT_FLOAT16)
      .Input(DT_FLOAT16)
      .Input({DT_FLOAT16, DT_FLOAT16})
      .Input({DT_FLOAT16, DT_FLOAT})
      .AssertSucceed();
}
TEST_F(UTInferDataType, sym_infer_for_type_check_for_unused_sym_required_input_dtype_out_of_range) {
  OpDtypeInfer("Op13")
      .Input(DT_FLOAT)
      .Input(DT_FLOAT16)
      .Input({DT_FLOAT16, DT_FLOAT16})
      .Input({DT_FLOAT16, DT_FLOAT})
      .AssertFailed();
}
TEST_F(UTInferDataType, sym_infer_for_type_check_for_unused_sym_opt_input_dtype_out_of_range) {
  OpDtypeInfer("Op13")
      .Input(DT_FLOAT16)
      .Input(DT_FLOAT)
      .Input({DT_FLOAT16, DT_FLOAT16})
      .Input({DT_FLOAT16, DT_FLOAT})
      .AssertFailed();
}
TEST_F(UTInferDataType, sym_infer_for_type_check_for_unused_sym_dyn_input_dtype_out_of_range) {
  OpDtypeInfer("Op13")
      .Input(DT_FLOAT16)
      .Input(DT_FLOAT16)
      .Input({DT_FLOAT16, DT_INT32})
      .Input({DT_FLOAT16, DT_FLOAT})
      .AssertFailed();
}
TEST_F(UTInferDataType, sym_infer_for_type_check_for_unused_sym_dyn_input_dtype_mismatch) {
  OpDtypeInfer("Op13")
      .Input(DT_FLOAT16)
      .Input(DT_FLOAT16)
      .Input({DT_FLOAT16, DT_FLOAT})
      .Input({DT_FLOAT16, DT_FLOAT})
      .AssertFailed();
}
TEST_F(UTInferDataType, sym_infer_for_type_check_for_unused_sym_dyn_list_input_dtype_out_of_range) {
  OpDtypeInfer("Op13")
      .Input(DT_FLOAT16)
      .Input(DT_FLOAT16)
      .Input({DT_FLOAT16, DT_FLOAT16})
      .Input({DT_FLOAT16, DT_INT32})
      .AssertFailed();
}

/* 测试IR改造后，基于老旧头文件编译的IR能基于CANN中的新IR使用符号化推导的能力
 OpLegacy的IR为老旧IR，不支持类型推导，
 CompatOpCurrent模拟当前版本的cann包，其中OpLegacy的IR改造为支持类型推导
 CompatOpFeature模拟未来版本的cann包，其中OpLegacy的IR改造为支持类型推导及新增类型DT_INT32支持
*/
namespace {
REG_OP(CompatOpCurrent)
    .INPUT(input1, "T")
    .OUTPUT(output1, "T")
    .DATATYPE(T, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(CompatOpCurrent);

REG_OP(CompatOpFeature)
    .INPUT(input1, "T")
    .OUTPUT(output1, "T")
    .DATATYPE(T, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OP_END_FACTORY_REG(CompatOpFeature);

void MockLoadOpsProtoCurrent() {
  OperatorFactoryImpl::SetRegisterOverridable(true);
  static const OperatorCreatorRegister g_register_compat_feature(
      "OpLegacy", [](const AscendString &name) { return op::CompatOpCurrent(name); });
  OperatorFactoryImpl::SetRegisterOverridable(false);
}

void MockLoadOpsProtoFeature() {
  OperatorFactoryImpl::SetRegisterOverridable(true);
  static const OperatorCreatorRegister g_register_compat_feature(
      "OpLegacy", [](const AscendString &name) { return op::CompatOpFeature(name); });
  OperatorFactoryImpl::SetRegisterOverridable(false);
}
}  // namespace

REG_OP(OpLegacy)
    .INPUT(input1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(OpLegacy);

TEST_F(UTInferDataType, sym_infer_for_compat_with_legacy_ir) {
  EXPECT_NO_THROW(
    // 原始OpLegacy不支持类型推导
    OpDtypeInfer("OpLegacy").Input(DT_FLOAT16).AssertFailed();
    OpDtypeInfer("OpLegacy").Input(DT_INT32).AssertFailed();

    // 模拟在老的app中加载新的ops proto，其中的IR相较于Legacy支持了类型推导
    MockLoadOpsProtoCurrent();

    // 验证新创建的OpLegacy能正常类型推导，但是不支持新增类型DT_INT32
    OpDtypeInfer("OpLegacy").Input(DT_FLOAT16).Expect(DT_FLOAT16).AssertSucceed();
    OpDtypeInfer("OpLegacy").Input(DT_INT32).AssertFailed();  // 此时仍不支持DT_INT32

    // 模拟将来已经支持符号推导编译后，加载未来版本ops proto, IR新增支持类型场景，其中的IR相较于Legacy支持了类型推导及新增类型DT_INT32支持
    MockLoadOpsProtoFeature();

    // 验证新创建的OpLegacy能正常类型推导，同时支持新增类型DT_INT32
    OpDtypeInfer("OpLegacy").Input(DT_FLOAT16).Expect(DT_FLOAT16).AssertSucceed();
    OpDtypeInfer("OpLegacy").Input(DT_INT32).Expect(DT_INT32).AssertSucceed();
  );
}

REG_OP(OpRecover)
    .INPUT(input1, "T")
    .OUTPUT(output1, "T")
    .DATATYPE(T, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(OpRecover);

TEST_F(UTInferDataType, sym_infer_after_recover) {
  EXPECT_NO_THROW(
    auto desc = std::make_shared<OpDesc>();
    desc->SetType("OpRecover");
    desc->AddInputDesc("input1", GeTensorDesc(GeShape(), FORMAT_ND, DT_FLOAT16));
    desc->AddOutputDesc("output1", GeTensorDesc(GeShape(), FORMAT_ND, DT_FLOAT16));
    desc->AppendIrInput("input1", IrInputType::kIrInputRequired);
    desc->AppendIrOutput("output1", IrOutputType::kIrOutputRequired);

    OpDtypeInfer(desc).Input(DT_FLOAT16).AssertFailed();

    auto graph = std::make_shared<ComputeGraph>("test");
    graph->AddNode(desc);
    RecoverIrUtils::RecoverIrDefinitions(graph);

    OpDtypeInfer(desc).Input(DT_FLOAT16).Expect(DT_FLOAT16).AssertSucceed();
  );
}
}  // namespace ge
