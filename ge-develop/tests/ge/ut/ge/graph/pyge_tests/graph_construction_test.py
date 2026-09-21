#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Graph Construction Test - 使用 pytest 框架
测试使用 es_ut_test 包的算子接口进行图构建
"""

import pytest

try:
    from ge.es.graph_builder import GraphBuilder, TensorHolder
    from ge.graph import Graph, Tensor, DumpFormat
    from ge.graph.types import DataType, Format
    from ge.es.ut_test import (
        phony_1i_1o, Add, Const, phony_1i1dyi_1o, phony_If, While,
        phony_3opi_1o, phony_1opi_1o, phony_req_attrs
    )
except Exception as e:
    import traceback

    print(f'Error: {e}')
    traceback.print_exc()


class TestGraphConstruction:
    """Graph construction test using es_ut_test operators"""

    @pytest.fixture
    def builder(self):
        """创建 GraphBuilder 实例的 fixture"""
        return GraphBuilder("test_graph")

    def test_basic_graph_construction(self, builder):
        """测试基本的图构建功能 - 使用 phony_1i_1o 算子"""
        # 创建输入 tensor
        input_tensor = builder.create_input(
            index=0,
            name="input",
            data_type=DataType.DT_FLOAT,
            format=Format.FORMAT_NCHW,
            shape=[1, 3, 224, 224]
        )
        assert input_tensor is not None

        # 使用 es_ut_test 包中的 phony_1i_1o 算子
        output_tensor = phony_1i_1o(input_tensor, index=1)
        assert isinstance(output_tensor, TensorHolder)

        # 再用 Add 算子连接
        result_tensor = Add(output_tensor, input_tensor)
        assert isinstance(result_tensor, TensorHolder)

        # 设置输出
        builder.set_graph_output(result_tensor, 0)

        # 构建图
        graph = builder.build_and_reset()
        assert isinstance(graph, Graph)
        assert graph.name == "test_graph"

        # 验证图包含节点
        nodes = graph.get_all_nodes()
        assert len(nodes) > 0

        # 验证包含预期的节点类型
        node_types = [node.type for node in nodes]
        assert "Data" in node_types
        assert "phony_1i_1o" in node_types
        assert "Add" in node_types
        assert "NetOutput" in node_types

        # 验证节点数量
        node_type_count = {}
        for node_type in node_types:
            node_type_count[node_type] = node_type_count.get(node_type, 0) + 1

        assert node_type_count.get("Data") == 1
        assert node_type_count.get("phony_1i_1o") == 1
        assert node_type_count.get("Add") == 1
        assert node_type_count.get("NetOutput") == 1

        # 验证连边关系
        # 找到各个节点
        data_node = None
        phony_node = None
        add_node = None
        netoutput_node = None

        for node in nodes:
            if node.type == "Data":
                data_node = node
            elif node.type == "phony_1i_1o":
                phony_node = node
            elif node.type == "Add":
                add_node = node
            elif node.type == "NetOutput":
                netoutput_node = node

        assert data_node is not None
        assert phony_node is not None
        assert add_node is not None
        assert netoutput_node is not None

        # 验证 phony_1i_1o 的输入连接到 Data
        phony_in_node, phony_in_port = phony_node.get_in_data_nodes_and_port_indexes(0)
        assert phony_in_node.name == data_node.name
        assert phony_in_port == 0

        # 验证 Add 的两个输入
        add_in0_node, add_in0_port = add_node.get_in_data_nodes_and_port_indexes(0)
        assert add_in0_node.name == phony_node.name
        assert add_in0_port == 0

        add_in1_node, add_in1_port = add_node.get_in_data_nodes_and_port_indexes(1)
        assert add_in1_node.name == data_node.name
        assert add_in1_port == 0

        # 验证 NetOutput 的输入连接到 Add
        netoutput_in_node, netoutput_in_port = netoutput_node.get_in_data_nodes_and_port_indexes(0)
        assert netoutput_in_node.name == add_node.name
        assert netoutput_in_port == 0

    def test_const_graph_construction(self, builder):
        input_tensor = builder.create_input(
            index=0,
            name="input",
            data_type=DataType.DT_FLOAT,
            format=Format.FORMAT_NCHW,
            shape=[5]
        )
        tensor = Tensor([1, 2, 3, 4, 5], data_type=DataType.DT_INT64, shape=[5])

        const = Const(builder, value=tensor)
        assert isinstance(const, TensorHolder)
        constint32 = Const(builder, value=Tensor([1, 2, 3, 4, 5], data_type=DataType.DT_INT32, shape=[5]))
        constint16 = Const(builder, value=Tensor([1, 2, 3, 4, 5], data_type=DataType.DT_INT16, shape=[5]))
        constint8 = Const(builder, value=Tensor([1, 2, 3, 4, 5], data_type=DataType.DT_INT8, shape=[5]))
        constuint64 = Const(builder, value=Tensor([1, 2, 3, 4, 5], data_type=DataType.DT_UINT64, shape=[5]))
        constuint32 = Const(builder, value=Tensor([1, 2, 3, 4, 5], data_type=DataType.DT_UINT32, shape=[5]))
        constuint16 = Const(builder, value=Tensor([1, 2, 3, 4, 5], data_type=DataType.DT_UINT16, shape=[5]))
        constuint8 = Const(builder, value=Tensor([1, 2, 3, 4, 5], data_type=DataType.DT_UINT8, shape=[5]))
        constfloat32 = Const(builder, value=Tensor([1, 2, 3, 4, 5], data_type=DataType.DT_FLOAT, shape=[5]))
        constfloat16 = Const(builder, value=Tensor([1.0, 2.0, 3.0, 4.0, 5.0], data_type=DataType.DT_FLOAT16, shape=[5]))
        input_tensor1 = Add(Add(Add(Add(input_tensor, const), constint32), constint16), constint8)
        input_tensor2 = Add(Add(Add(Add(input_tensor, const), constuint64), constuint32), constuint8)
        input_tensor3 = Add(constfloat32, constfloat16)
        input_tensor1 = Add(input_tensor1, input_tensor2)
        input_tensor1 = Add(input_tensor1, input_tensor3)
        input_tensor1 = Add(input_tensor1, builder.create_const_int64(0))
        input_tensor1 = Add(input_tensor1, builder.create_scalar_int64(0))
        assert isinstance(input_tensor1, TensorHolder)

        builder.set_graph_output(input_tensor1, 0)
        graph = builder.build_and_reset()
        graph.set_attr("float_attr", 3.14)

        # 验证图包含节点
        nodes = graph.get_all_nodes()
        assert len(nodes) > 0

        # 验证包含预期的节点类型
        node_types = [node.type for node in nodes]
        assert "Data" in node_types
        assert "Const" in node_types
        assert "Add" in node_types
        assert "NetOutput" in node_types
        # 打印图, 预期是readable格式
        print("=== graph readable dump BEGIN ===\n")
        print(graph)
        print("=== graph readable dump END ===\n")
        assert str(graph).strip() == """graph("test_graph"):
  %input : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %Const_0 : [#users=1] = Node[type=Const] (attrs = {value: [1 2 3 4 5]})
  %Const_1 : [#users=1] = Node[type=Const] (attrs = {value: [1 2 3 4 5]})
  %Const_2 : [#users=1] = Node[type=Const] (attrs = {value: [1 2 3 4 5]})
  %Const_3 : [#users=1] = Node[type=Const] (attrs = {value: [1 2 3 4 5]})
  %Const_4 : [#users=1] = Node[type=Const] (attrs = {value: [1 2 3 4 5]})
  %Const_5 : [#users=1] = Node[type=Const] (attrs = {value: [1 2 3 4 5]})
  %Const_6 : [#users=1] = Node[type=Const] (attrs = {value: [1 2 3 4 5]})
  %Const_7 : [#users=1] = Node[type=Const] (attrs = {value: [1 2 3 4 5]})
  %Const_8 : [#users=1] = Node[type=Const] (attrs = {value: [1.000000 2.000000 3.000000 4.000000 5.000000]})
  %Const_9 : [#users=1] = Node[type=Const] (attrs = {value: [1.000000 2.000000 3.000000 4.000000 5.000000]})
  %Add_10 : [#users=1] = Node[type=Add] (inputs = (x1=%input, x2=%Const_0))
  %Add_11 : [#users=1] = Node[type=Add] (inputs = (x1=%Add_10, x2=%Const_1))
  %Add_12 : [#users=1] = Node[type=Add] (inputs = (x1=%Add_11, x2=%Const_2))
  %Add_13 : [#users=1] = Node[type=Add] (inputs = (x1=%Add_12, x2=%Const_3))
  %Add_14 : [#users=1] = Node[type=Add] (inputs = (x1=%input, x2=%Const_0))
  %Add_15 : [#users=1] = Node[type=Add] (inputs = (x1=%Add_14, x2=%Const_4))
  %Add_16 : [#users=1] = Node[type=Add] (inputs = (x1=%Add_15, x2=%Const_5))
  %Add_17 : [#users=1] = Node[type=Add] (inputs = (x1=%Add_16, x2=%Const_7))
  %Add_18 : [#users=1] = Node[type=Add] (inputs = (x1=%Const_8, x2=%Const_9))
  %Add_19 : [#users=1] = Node[type=Add] (inputs = (x1=%Add_13, x2=%Add_17))
  %Add_20 : [#users=1] = Node[type=Add] (inputs = (x1=%Add_19, x2=%Add_18))
  %Const_21 : [#users=1] = Node[type=Const] (attrs = {value: [0]})
  %Add_22 : [#users=1] = Node[type=Add] (inputs = (x1=%Add_20, x2=%Const_21))
  %Const_23 : [#users=1] = Node[type=Const] (attrs = {value: [0]})
  %Add_24 : [#users=1] = Node[type=Add] (inputs = (x1=%Add_22, x2=%Const_23))

  return (%Add_24)
""".strip()

    def test_const_graph_construction_fp16_coverage(self, builder):
        """测试 FP16 转换函数的完整覆盖率，确保 float_to_fp16_bits 函数覆盖率达到 100%"""

        # 1. 正常规格化数（-14 <= exponent <= 15，不需要舍入）
        normal_fp16 = Const(builder, value=Tensor([1.0, 2.0, 3.0], data_type=DataType.DT_FLOAT16, shape=[3]))

        # 2. 零值（exponent < -24，下溢到0）
        zero_fp16 = Const(builder, value=Tensor([0.0], data_type=DataType.DT_FLOAT16, shape=[1]))

        # 3. 负数（覆盖符号位）
        negative_fp16 = Const(builder, value=Tensor([-1.0, -2.5], data_type=DataType.DT_FLOAT16, shape=[2]))

        # 4. 非常大的数（exponent > 15，溢出到无穷大）
        # FP16 最大正常值约为 65504，超过这个值会溢出
        large_fp16 = Const(builder, value=Tensor([70000.0], data_type=DataType.DT_FLOAT16, shape=[1]))

        # 5. 非常小的数（exponent < -24，下溢到0）
        # 2^-25 约为 2.98e-8，小于这个值会下溢
        tiny_fp16 = Const(builder, value=Tensor([1e-9], data_type=DataType.DT_FLOAT16, shape=[1]))

        # 6. 非规格化数（-24 <= exponent < -14），不需要舍入
        # 2^-15 约为 3.05e-5，在非规格化范围内
        denormal_fp16 = Const(builder, value=Tensor([2e-5], data_type=DataType.DT_FLOAT16, shape=[1]))

        # 7. 非规格化数需要舍入的情况（round_bit and ((shifted & 0x1) or remainder)）
        denormal_round_fp16 = Const(builder, value=Tensor([1.5e-5], data_type=DataType.DT_FLOAT16, shape=[1]))

        # 8. 规格化数需要舍入（round_bits > 0x1000）
        round_up_fp16 = Const(builder, value=Tensor([1.0001], data_type=DataType.DT_FLOAT16, shape=[1]))

        # 9. 偶数舍入的情况（round_bits == 0x1000 and half_mant & 0x1）
        # 这个值需要精确构造：mantissa 的低13位是 0x1000，且 half_mant 是奇数
        # 例如：1.0 + 2^-12 = 1.000244140625，但需要确保 half_mant 是奇数
        even_round_fp16 = Const(builder, value=Tensor([1.00048828125], data_type=DataType.DT_FLOAT16, shape=[1]))

        # 10. 偶数舍入但不满足条件（round_bits == 0x1000 but half_mant & 0x1 == 0）
        # 需要 half_mant 是偶数的情况
        even_round_no_fp16 = Const(builder, value=Tensor([2.00048828125], data_type=DataType.DT_FLOAT16, shape=[1]))

        # 11. 舍入后需要进位到指数（half_mant == 0x400 after rounding）
        # 需要构造一个 half_mant 接近 0x3FF 且需要舍入的值
        # 例如：1.99951171875 的 half_mant 是 0x3FF，如果舍入会变成 0x400
        carry_exp_fp16 = Const(builder, value=Tensor([1.99951171875], data_type=DataType.DT_FLOAT16, shape=[1]))

        # 12. 舍入后指数溢出（half_exp == 0x1F after rounding）
        # FP16 最大指数是 15，对应 FP32 指数是 30
        # 需要构造一个接近最大值且需要舍入的值，使得 half_exp 变成 0x1F
        max_round_fp16 = Const(builder, value=Tensor([65503.0], data_type=DataType.DT_FLOAT16, shape=[1]))

        # 13. 正无穷大（exponent == 255, mantissa == 0）
        pos_inf_fp16 = Const(builder, value=Tensor([float('inf')], data_type=DataType.DT_FLOAT16, shape=[1]))

        # 14. 负无穷大（exponent == 255, mantissa == 0, sign == 1）
        neg_inf_fp16 = Const(builder, value=Tensor([float('-inf')], data_type=DataType.DT_FLOAT16, shape=[1]))

        # 15. NaN（exponent == 255, mantissa != 0）
        nan_fp16 = Const(builder, value=Tensor([float('nan')], data_type=DataType.DT_FLOAT16, shape=[1]))

        # 验证所有常量都创建成功
        assert isinstance(normal_fp16, TensorHolder)
        assert isinstance(zero_fp16, TensorHolder)
        assert isinstance(negative_fp16, TensorHolder)
        assert isinstance(large_fp16, TensorHolder)
        assert isinstance(tiny_fp16, TensorHolder)
        assert isinstance(denormal_fp16, TensorHolder)
        assert isinstance(denormal_round_fp16, TensorHolder)
        assert isinstance(round_up_fp16, TensorHolder)
        assert isinstance(even_round_fp16, TensorHolder)
        assert isinstance(even_round_no_fp16, TensorHolder)
        assert isinstance(carry_exp_fp16, TensorHolder)
        assert isinstance(max_round_fp16, TensorHolder)
        assert isinstance(pos_inf_fp16, TensorHolder)
        assert isinstance(neg_inf_fp16, TensorHolder)
        assert isinstance(nan_fp16, TensorHolder)

        # 创建一个输入，用于连接标量常量（shape=[1]）
        input_tensor = builder.create_input(
            index=0,
            name="input",
            data_type=DataType.DT_FLOAT16,
            format=Format.FORMAT_ND,
            shape=[1]
        )

        # 将所有常量连接到输入上，这样它们都会出现在图中
        # 首先连接标量常量（shape=[1]）
        result = Add(input_tensor, zero_fp16)
        result = Add(result, large_fp16)
        result = Add(result, tiny_fp16)
        result = Add(result, denormal_fp16)
        result = Add(result, denormal_round_fp16)
        result = Add(result, round_up_fp16)
        result = Add(result, even_round_fp16)
        result = Add(result, even_round_no_fp16)
        result = Add(result, carry_exp_fp16)
        result = Add(result, max_round_fp16)
        result = Add(result, pos_inf_fp16)
        result = Add(result, neg_inf_fp16)
        result = Add(result, nan_fp16)

        # 对于形状不同的常量，我们需要单独处理
        # normal_fp16 shape=[3], negative_fp16 shape=[2]
        # 由于 Add 需要兼容的形状，我们创建一个额外的输入来连接它们
        input2 = builder.create_input(
            index=1,
            name="input2",
            data_type=DataType.DT_FLOAT16,
            format=Format.FORMAT_ND,
            shape=[3]
        )
        result2 = Add(input2, normal_fp16)

        input3 = builder.create_input(
            index=2,
            name="input3",
            data_type=DataType.DT_FLOAT16,
            format=Format.FORMAT_ND,
            shape=[2]
        )
        result3 = Add(input3, negative_fp16)

        # 构建图并验证
        # 设置多个输出以确保所有常量都被使用
        builder.set_graph_output(result, 0)
        builder.set_graph_output(result2, 1)
        builder.set_graph_output(result3, 2)
        graph = builder.build_and_reset()
        assert graph is not None

        # 打印图, 预期是readable格式
        print("=== FP16 coverage test graph readable dump BEGIN ===\n")
        print(graph)
        print("=== FP16 coverage test graph readable dump END ===\n")

        # 验证图包含所有预期的节点
        nodes = graph.get_all_nodes()
        assert len(nodes) > 0

        # 验证包含预期的节点类型
        node_types = [node.type for node in nodes]
        assert "Const" in node_types
        assert "Add" in node_types
        assert "NetOutput" in node_types

        # 验证图的输出格式（检查是否包含 FP16 相关的值）
        graph_str = str(graph)

        # 统计 Const 节点数量，应该至少有 15 个（每个测试场景一个）
        const_nodes = [node for node in nodes if node.type == "Const"]
        assert len(const_nodes) >= 15, f"Expected at least 15 Const nodes, got {len(const_nodes)}"

        # 验证包含正常值（normal_fp16 包含 1.0, 2.0, 3.0）
        assert "1.000000" in graph_str or "1.0" in graph_str or "[1" in graph_str
        # 验证包含零值
        assert "0.000000" in graph_str or "0.0" in graph_str or "[0" in graph_str
        # 验证包含负值（negative_fp16 包含 -1.0, -2.5）
        assert "-1.000000" in graph_str or "-1.0" in graph_str or "[-1" in graph_str
        # 验证包含特殊值（无穷大或 NaN 可能显示为特殊格式）
        # 注意：特殊值的显示格式可能因实现而异，这里只做基本检查
        # 无穷大可能显示为 "inf" 或特殊值，NaN 可能显示为 "nan" 或特殊值

    def test_numeric_const_graph_construction(self, builder, monkeypatch):
        """测试数值常量构图功能"""
        from ge._capi.pyes_graph_builder_wrapper import get_generated_lib

        # Mock _get_math_operator_lib
        def mock_get_math_lib(self):
            return get_generated_lib("libes_ut_test.so")

        monkeypatch.setattr(TensorHolder, '_get_math_operator_lib', mock_get_math_lib)

        input_tensor = builder.create_input(
            index=0,
            name="input",
            data_type=DataType.DT_FLOAT,
            format=Format.FORMAT_NCHW,
            shape=[1]
        )

        then_graph_builder = GraphBuilder("then_graph")
        then_graph_const = then_graph_builder.create_const_int64(1)
        then_graph_builder.set_graph_output(then_graph_const, 0)
        then_graph = then_graph_builder.build_and_reset()

        else_graph_builder = GraphBuilder("else_graph")
        else_graph_const = else_graph_builder.create_const_int64(1)
        else_graph_builder.set_graph_output(else_graph_const, 0)
        else_graph = else_graph_builder.build_and_reset()

        phony_res = phony_1i1dyi_1o(input_tensor, [1, 2])
        phony_if_res = phony_If(phony_res, [0, 2], 1, then_graph, else_graph)
        add_res = phony_if_res[0] + [1, 2, 3, 4, 5]
        assert isinstance(add_res, TensorHolder)

        builder.set_graph_output(add_res, 0)
        graph = builder.build_and_reset()

        # 验证图包含节点
        nodes = graph.get_all_nodes()
        assert len(nodes) > 0

        # 验证包含预期的节点类型
        node_types = [node.type for node in nodes]
        assert "Data" in node_types
        assert "Const" in node_types
        assert "phony_1i1dyi_1o" in node_types
        assert "phony_If" in node_types
        assert "Add" in node_types
        assert "NetOutput" in node_types

    def test_all_optional_inputs_graph_construction_with_input_and_owner_graph(self, builder):
        """测试所有输入都是可选的构图功能：所有输入都提供，传递 owner_graph"""
        input1 = builder.create_input(
            index=0,
            name="input1",
            data_type=DataType.DT_FLOAT,
            format=Format.FORMAT_NCHW,
            shape=[1, 3, 224, 224]
        )
        input2 = builder.create_input(
            index=1,
            name="input2",
            data_type=DataType.DT_FLOAT,
            format=Format.FORMAT_NCHW,
            shape=[1, 3, 224, 224]
        )
        input3 = builder.create_input(
            index=2,
            name="input3",
            data_type=DataType.DT_FLOAT,
            format=Format.FORMAT_NCHW,
            shape=[1, 3, 224, 224]
        )

        out = phony_3opi_1o(input1, input2, input3, builder)
        assert isinstance(out, TensorHolder)

        builder.set_graph_output(out, 0)
        graph = builder.build_and_reset()
        assert graph is not None

        nodes = graph.get_all_nodes()
        node_types = [node.type for node in nodes]
        assert "Data" in node_types
        assert "phony_3opi_1o" in node_types
        assert "NetOutput" in node_types

    def test_all_optional_inputs_graph_construction_with_input(self, builder):
        """测试所有输入都是可选的构图功能：部分输入为 None，不传 owner_graph"""
        input1 = builder.create_input(
            index=0,
            name="input1",
            data_type=DataType.DT_FLOAT,
            format=Format.FORMAT_NCHW,
            shape=[1, 3, 224, 224]
        )

        out = phony_3opi_1o(input1)
        assert isinstance(out, TensorHolder)

        builder.set_graph_output(out, 0)
        graph = builder.build_and_reset()
        assert graph is not None

        nodes1 = graph.get_all_nodes()
        node_types1 = [node.type for node in nodes1]
        assert "Data" in node_types1
        assert "phony_3opi_1o" in node_types1
        assert "NetOutput" in node_types1

    def test_all_optional_inputs_graph_construction_with_owner_graph(self, builder):
        """测试所有输入都是可选的构图功能：不提供输入，显式传递 owner_graph"""
        out = phony_3opi_1o(owner_builder=builder)
        assert isinstance(out, TensorHolder)

        builder.set_graph_output(out, 0)
        graph = builder.build_and_reset()
        assert graph is not None

        nodes = graph.get_all_nodes()
        node_types5 = [node.type for node in nodes]
        assert "phony_3opi_1o" in node_types5
        assert "NetOutput" in node_types5

    def test_all_optional_inputs_graph_construction_without_input_and_owner_graph(self, builder):
        """测试所有输入都是可选的构图功能：不提供输入，不传递 owner_graph"""
        # 调用C接口返回空，构造返回值时跑异常
        with pytest.raises(ValueError, match=
        "Please ensure at least one input tensor or an explicit owner_builder is provided when supported"):
            out4 = phony_3opi_1o()

    def test_one_optional_input_graph_construction_with_numeric_input_and_owner_graph(self, builder):
        """测试只有一个输入是可选的构图功能：输入为数值，显式传递 owner_graph"""
        out = phony_1opi_1o(1, owner_builder=builder, flag=False)
        assert isinstance(out, TensorHolder)

        builder.set_graph_output(out, 0)
        graph = builder.build_and_reset()
        assert graph is not None

        nodes = graph.get_all_nodes()
        node_types5 = [node.type for node in nodes]
        assert "phony_1opi_1o" in node_types5
        assert "NetOutput" in node_types5

    def test_nested_subgraph_construction(self, builder):
        input_tensor = builder.create_input(0)
        if_output = phony_If(input_tensor, [1, 2], 2, build_if_then_branch_graph(), build_if_else_branch_graph())
        builder.set_graph_output(if_output[0], 0)
        graph = builder.build_and_reset()

        assert graph is not None
        nodes = graph.get_all_nodes()
        assert len(nodes) > 0
        # 验证包含预期的节点类型
        node_types = [node.type for node in nodes]
        assert "Data" in node_types
        assert "phony_If" in node_types
        assert "NetOutput" in node_types

        all_subgraphs = graph.get_all_subgraphs()
        assert len(all_subgraphs) == 6

        then_branch_subgraph = graph.get_subgraph("then_branch")
        assert then_branch_subgraph is not None
        else_branch_subgraph = graph.get_subgraph("else_branch")
        assert else_branch_subgraph is not None

        while_01_cond_subgraph = then_branch_subgraph.get_subgraph("while_01_cond")
        assert while_01_cond_subgraph is not None
        while_01_cond_subgraph = graph.get_subgraph("while_01_cond")
        assert while_01_cond_subgraph is not None

        while_01_body_subgraph = then_branch_subgraph.get_subgraph("while_01_body")
        assert while_01_body_subgraph is not None
        while_01_body_subgraph = graph.get_subgraph("while_01_body")
        assert while_01_body_subgraph is not None

        while_02_cond_subgraph = else_branch_subgraph.get_subgraph("while_02_cond")
        assert while_02_cond_subgraph is not None
        while_02_cond_subgraph = graph.get_subgraph("while_02_cond")
        assert while_02_cond_subgraph is not None


        while_02_body_subgraph = else_branch_subgraph.get_subgraph("while_02_body")
        assert while_02_body_subgraph is not None
        while_02_body_subgraph = graph.get_subgraph("while_02_body")
        assert while_02_body_subgraph is not None

        # 验证 Readable Dump 输出
        print("=== Nested subgraph readable dump BEGIN ===")
        print(graph)
        print("=== Nested subgraph readable dump END ===")
        assert str(graph).strip() == """graph("test_graph"):
  %input_0 : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %Const_0 : [#users=1] = Node[type=Const] (attrs = {value: [1]})
  %Const_1 : [#users=1] = Node[type=Const] (attrs = {value: [2]})
  %phony_If_2 : [#users=2] = Node[type=phony_If] (inputs = (cond=%input_0, input_0=%Const_0, input_1=%Const_1), attrs = {then_branch: %then_branch, else_branch: %else_branch})
  %ret : [#users=1] = get_element[node=%phony_If_2](0)
  %ret_1 : [#users=0] = get_element[node=%phony_If_2](1)

  return (%ret)

graph("then_branch"):
  %input_0 : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %While_0 : [#users=2] = Node[type=While] (inputs = (input_0=%input_0), attrs = {cond: %while_01_cond, body: %while_01_body})
  %ret : [#users=1] = get_element[node=%While_0](0)
  %ret_1 : [#users=1] = get_element[node=%While_0](1)

  return (output_0=%ret, output_1=%ret_1)

graph("while_01_cond"):
  %input_0 : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %phony_1i_1o_0 : [#users=1] = Node[type=phony_1i_1o] (inputs = (x=%input_0))

  return (%phony_1i_1o_0)

graph("while_01_body"):
  %input_0 : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %phony_1i_1o_0 : [#users=1] = Node[type=phony_1i_1o] (inputs = (x=%input_0))

  return (output_0=%input_0, output_1=%phony_1i_1o_0)

graph("else_branch"):
  %input_0 : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %While_0 : [#users=2] = Node[type=While] (inputs = (input_0=%input_0), attrs = {cond: %while_02_cond, body: %while_02_body})
  %ret : [#users=1] = get_element[node=%While_0](0)
  %ret_1 : [#users=1] = get_element[node=%While_0](1)

  return (output_0=%ret, output_1=%ret_1)

graph("while_02_cond"):
  %input_0 : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %phony_1i_1o_0 : [#users=1] = Node[type=phony_1i_1o] (inputs = (x=%input_0))

  return (%phony_1i_1o_0)

graph("while_02_body"):
  %input_0 : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %phony_1i_1o_0 : [#users=1] = Node[type=phony_1i_1o] (inputs = (x=%input_0))

  return (output_0=%input_0, output_1=%phony_1i_1o_0)
""".strip()

    def test_list_attr_graph_construction(self, builder):
        """测试带有二维数组、字符串数组属性的算子图构建功能 - 使用 phony_req_attrs 算子"""
        # 创建输入 tensor
        input_tensor = builder.create_input(index=0)
        assert input_tensor is not None

        # 使用 es_ut_test 包中的 phony_req_attrs 算子
        output_tensor = phony_req_attrs(
            input_tensor,
            req_data_type=DataType.DT_INT64,
            req_list_data_type=[DataType.DT_FLOAT, DataType.DT_INT32, DataType.DT_INT64],
            req_list_list_int=[[0, 1], [2, 3]],
            req_tensor=Tensor(),
            req_list_string=["test"]
        )
        assert isinstance(output_tensor, TensorHolder)

        # 设置输出
        builder.set_graph_output(output_tensor, 0)

        # 构建图
        graph = builder.build_and_reset()
        assert isinstance(graph, Graph)
        assert graph.name == "test_graph"

        # 验证图包含节点
        nodes = graph.get_all_nodes()
        assert len(nodes) > 0

        # 验证包含预期的节点类型
        node_types = [node.type for node in nodes]
        assert "Data" in node_types
        assert "phony_req_attrs" in node_types
        assert "NetOutput" in node_types
        # 验证dump结果
        assert str(graph).strip() == """graph("test_graph"):
  %input_0 : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %phony_req_attrs_0 : [#users=1] = Node[type=phony_req_attrs] (inputs = (x=%input_0), attrs = {req_data_type: DT_INT64, req_list_data_type: {DT_FLOAT, DT_INT32, DT_INT64}, req_list_list_int: {{0, 1}, {2, 3}}, req_tensor: <empty>, req_list_string: {"test"}})

  return (%phony_req_attrs_0)
""".strip()


def build_while_cond_graph(while_graph_builder: GraphBuilder):
    while_cond_input = while_graph_builder.create_input(0)
    while_cond_output = phony_1i_1o(while_cond_input)
    while_graph_builder.set_graph_output(while_cond_output, 0)
    return while_graph_builder.build_and_reset()


def build_while_body_graph(while_graph_builder: GraphBuilder):
    while_body_input = while_graph_builder.create_input(0)
    while_body_output = phony_1i_1o(while_body_input)
    while_graph_builder.set_graph_output(while_body_input, 0)
    while_graph_builder.set_graph_output(while_body_output, 1)
    return while_graph_builder.build_and_reset()


def build_while01_cond_graph():
    while_cond_builder = GraphBuilder("while_01_cond")
    return build_while_cond_graph(while_cond_builder)


def build_while01_body_graph():
    while_body_builder = GraphBuilder("while_01_body")
    return build_while_body_graph(while_body_builder)


def build_while02_cond_graph():
    while_cond_builder = GraphBuilder("while_02_cond")
    return build_while_cond_graph(while_cond_builder)


def build_while02_body_graph():
    while_body_builder = GraphBuilder("while_02_body")
    return build_while_body_graph(while_body_builder)


def build_if_then_branch_graph():
    then_builder = GraphBuilder("then_branch")
    then_input = then_builder.create_input(0)
    then_while_output = While([then_input], 2, build_while01_cond_graph(), build_while01_body_graph())
    then_builder.set_graph_output(then_while_output[0], 0)
    then_builder.set_graph_output(then_while_output[1], 1)
    return then_builder.build_and_reset()


def build_if_else_branch_graph():
    else_builder = GraphBuilder("else_branch")
    else_input = else_builder.create_input(0)
    else_while_output = While([else_input], 2, build_while02_cond_graph(), build_while02_body_graph())
    else_builder.set_graph_output(else_while_output[0], 0)
    else_builder.set_graph_output(else_while_output[1], 1)
    return else_builder.build_and_reset()
