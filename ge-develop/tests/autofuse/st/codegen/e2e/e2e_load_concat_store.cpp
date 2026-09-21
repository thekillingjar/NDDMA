/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"
#include "ascir_ops_utils.h"

using namespace std;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;

void LoadConcatStore_BeforeAutofuse(ge::AscGraph &graph) {
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto s2 = graph.CreateSizeVar("s2");

    auto z0 = graph.CreateAxis("z0", s0);
    auto zo = graph.CreateAxis("zo", s1 + s2);
    auto zo_s_0 = graph.CreateAxis("zo_s_0", Axis::Type::kAxisTypeOriginal, s1, {zo.id}, ge::kIdNone);
    auto zo_s_1 = graph.CreateAxis("zo_s_1", Axis::Type::kAxisTypeOriginal, s2, {zo.id}, ge::kIdNone);

    Data x1("x1", graph);
    Data x2("x2", graph);
    Load load1("load1");
    Load load2("load2");
    ge::ascir_op::Concat concat("concat");
    Store store("store");
    Output y("y");

    x1.attr.sched.axis = {z0.id, zo_s_0.id};
    x1.y.dtype = ge::DT_FLOAT;
    *x1.y.axis = {z0.id, zo_s_0.id};
    *x1.y.repeats = {s0, s1};
    *x1.y.strides = {s1, One};

    x2.attr.sched.axis = {z0.id, zo_s_1.id};
    x2.y.dtype = ge::DT_FLOAT;
    *x2.y.axis = {z0.id, zo_s_1.id};
    *x2.y.repeats = {s0, s2};
    *x2.y.strides = {s2, One};

    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id, zo_s_0.id};
    load1.y.dtype = ge::DT_FLOAT;
    *load1.y.axis = {z0.id, zo_s_0.id};
    *load1.y.repeats = {s0, s1};
    *load1.y.strides = {s1, One};

    load2.x = x2.y;
    load2.attr.sched.axis = {z0.id, zo_s_1.id};
    load2.y.dtype = ge::DT_FLOAT;
    *load2.y.axis = {z0.id, zo_s_1.id};
    *load2.y.repeats = {s0, s2};
    *load2.y.strides = {s2, One};

    concat.x = {load1.y, load2.y};
    concat.attr.sched.axis = {z0.id, zo.id};
    concat.y.dtype = ge::DT_FLOAT;
    *concat.y.axis = {z0.id, zo.id};
    *concat.y.repeats = {s0, s1 + s2};
    *concat.y.strides = {s1+s2, One};
    concat.attr.tmp_buffers = {{{ge::Symbol(65536), -1}, MemAttr(), 0}};

    store.x = concat.y;
    store.attr.sched.axis = {z0.id, zo.id};
    store.y.dtype = ge::DT_FLOAT;
    *store.y.axis = {z0.id, zo.id};
    *store.y.repeats = {s0, s1 + s2};
    *store.y.strides = {s1+s2, One};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, zo.id};
    y.y.dtype = ge::DT_FLOAT;
    *y.y.axis = {z0.id, zo.id};
    *y.y.repeats = {s0, s1 + s2};
    *y.y.strides = {s1 + s2, One};
}

void LoadConcatStore_BeforeAutofuseConcatInterAxis(ge::AscGraph &graph) {
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto s2 = graph.CreateSizeVar("s2");
    auto s3 = graph.CreateSizeVar("s3");

    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1 + s2);
    auto z1_s_0 = graph.CreateAxis("z1_s_0", Axis::Type::kAxisTypeOriginal, s1, {z1.id}, ge::kIdNone);
    auto z1_s_1 = graph.CreateAxis("z1_s_1", Axis::Type::kAxisTypeOriginal, s2, {z1.id}, ge::kIdNone);
    auto z2 = graph.CreateAxis("z2", s3);

    Data x1("x1", graph);
    Data x2("x2", graph);
    Load load1("load1");
    Load load2("load2");
    ge::ascir_op::Concat concat("concat");
    Store store("store");
    Output y("y");

    x1.attr.sched.axis = {z0.id, z1_s_0.id, z2.id};
    x1.y.dtype = ge::DT_FLOAT;
    *x1.y.axis = {z0.id, z1_s_0.id, z2.id};
    *x1.y.repeats = {s0, s1, s3};
    *x1.y.strides = {s1*s3, s3, One};

    x2.attr.sched.axis = {z0.id, z1_s_1.id, z2.id};
    x2.y.dtype = ge::DT_FLOAT;
    *x2.y.axis = {z0.id, z1_s_1.id, z2.id};
    *x2.y.repeats = {s0, s2, s3};
    *x2.y.strides = {s2*s3, s3, One};

    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id, z1_s_0.id, z2.id};
    load1.y.dtype = ge::DT_FLOAT;
    *load1.y.axis = {z0.id, z1_s_0.id, z2.id};
    *load1.y.repeats = {s0, s1, s3};
    *load1.y.strides = {s1*s3, s3, One};

    load2.x = x2.y;
    load2.attr.sched.axis = {z0.id, z1_s_1.id, z2.id};
    load2.y.dtype = ge::DT_FLOAT;
    *load2.y.axis = {z0.id, z1_s_1.id, z2.id};
    *load2.y.repeats = {s0, s2, s3};
    *load2.y.strides = {s2*s3, s3, One};

    concat.x = {load1.y, load2.y};
    concat.attr.sched.axis = {z0.id, z1.id, z2.id};
    concat.y.dtype = ge::DT_FLOAT;
    *concat.y.axis = {z0.id, z1.id, z2.id};
    *concat.y.repeats = {s0, s1 + s2, s3};
    *concat.y.strides = {(s1+s2)*s3, s3, One};
    concat.attr.tmp_buffers = {{{ge::Symbol(16384), -1}, MemAttr(), 0}};

    store.x = concat.y;
    store.attr.sched.axis = {z0.id, z1.id, z2.id};
    store.y.dtype = ge::DT_FLOAT;
    *store.y.axis = {z0.id, z1.id, z2.id};
    *store.y.repeats = {s0, s1 + s2, s3};
    *store.y.strides = {(s1+s2)*s3, s3, One};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, z1.id, z2.id};
    y.y.dtype = ge::DT_FLOAT;
    *y.y.axis = {z0.id, z1.id, z2.id};
    *y.y.repeats = {s0, s1 + s2, s3};
    *y.y.strides = {(s1+s2)*s3, s3, One};
}

void LoadConcatStore_BeforeAutofuse3dLastAxis(ge::AscGraph &graph) {
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto s2 = graph.CreateSizeVar("s2");
    auto s3 = graph.CreateSizeVar("s3");

    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    auto z2 = graph.CreateAxis("z2", s2 + s3);
    auto z2_s_0 = graph.CreateAxis("z2_s_0", Axis::Type::kAxisTypeOriginal, s2, {z2.id}, ge::kIdNone);
    auto z2_s_1 = graph.CreateAxis("z2_s_1", Axis::Type::kAxisTypeOriginal, s3, {z2.id}, ge::kIdNone);


    Data x1("x1", graph);
    Data x2("x2", graph);
    Load load1("load1");
    Load load2("load2");
    ge::ascir_op::Concat concat("concat");
    Store store("store");
    Output y("y");

    x1.attr.sched.axis = {z0.id, z1.id, z2_s_0.id};
    x1.y.dtype = ge::DT_FLOAT;
    *x1.y.axis = {z0.id, z1.id, z2_s_0.id};
    *x1.y.repeats = {s0, s1, s2};
    *x1.y.strides = {s1*s2, s2, One};

    x2.attr.sched.axis = {z0.id, z1.id, z2_s_1.id};
    x2.y.dtype = ge::DT_FLOAT;
    *x2.y.axis = {z0.id, z1.id, z2_s_1.id};
    *x2.y.repeats = {s0, s1, s3};
    *x2.y.strides = {s1*s3, s3, One};

    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id, z1.id, z2_s_0.id};
    load1.y.dtype = ge::DT_FLOAT;
    *load1.y.axis = {z0.id, z1.id, z2_s_0.id};
    *load1.y.repeats = {s0, s1, s2};
    *load1.y.strides = {s1*s2, s2, One};

    load2.x = x2.y;
    load2.attr.sched.axis = {z0.id, z1.id, z2_s_1.id};
    load2.y.dtype = ge::DT_FLOAT;
    *load2.y.axis = {z0.id, z1.id, z2_s_1.id};
    *load2.y.repeats = {s0, s1, s3};
    *load2.y.strides = {s1*s3, s3, One};

    concat.x = {load1.y, load2.y};
    concat.attr.sched.axis = {z0.id, z1.id, z2.id};
    concat.y.dtype = ge::DT_FLOAT;
    *concat.y.axis = {z0.id, z1.id, z2.id};
    *concat.y.repeats = {s0, s1, s2 + s3};
    *concat.y.strides = {s1*(s2+s3), s2 + s3, One};
    concat.attr.tmp_buffers = {{{ge::Symbol(16384), -1}, MemAttr(), 0}};

    store.x = concat.y;
    store.attr.sched.axis = {z0.id, z1.id, z2.id};
    store.y.dtype = ge::DT_FLOAT;
    *store.y.axis = {z0.id, z1.id, z2.id};
    *store.y.repeats = {s0, s1, s2 + s3};
    *store.y.strides = {s1*(s2+s3), s2 + s3, One};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, z1.id, z2.id};
    y.y.dtype = ge::DT_FLOAT;
    *y.y.axis = {z0.id, z1.id, z2.id};
    *y.y.repeats = {s0, s1, s2 + s3};
    *y.y.strides = {s1*(s2+s3), s2 + s3, One};
}

void LoadConcatStore_AfterInferOutput(ge::AscGraph &graph) {
    auto x1 = graph.FindNode("x1");
    x1->attr.api.compute_type = ComputeType::kComputeInvalid;

    auto x2 = graph.FindNode("x2");
    x2->attr.api.compute_type = ComputeType::kComputeInvalid;

    auto load1 = graph.FindNode("load1");
    load1->attr.api.compute_type = ComputeType::kComputeLoad;

    auto load2 = graph.FindNode("load2");
    load2->attr.api.compute_type = ComputeType::kComputeLoad;

    auto concat = graph.FindNode("concat");
    concat->outputs[0].attr.dtype =(ge::DataType)load1->outputs[0].attr.dtype;
    concat->attr.api.compute_type = ComputeType::kComputeConcat;

    auto store = graph.FindNode("store");
    store->outputs[0].attr.dtype = (ge::DataType)concat->outputs[0].attr.dtype;
    store->attr.api.compute_type = ComputeType::kComputeStore;

    auto y = graph.FindNode("y");
    y->attr.api.compute_type = ComputeType::kComputeInvalid;
}

void LoadConcatStore_AfterGetApiInfo(ge::AscGraph &graph) {
    auto x1 = graph.FindNode("x1");
    x1->attr.api.type = ApiType::kAPITypeBuffer;
    x1->attr.api.unit = ComputeUnit::kUnitNone;

    auto x2 = graph.FindNode("x2");
    x2->attr.api.type = ApiType::kAPITypeBuffer;
    x2->attr.api.unit = ComputeUnit::kUnitNone;

    auto load1 = graph.FindNode("load1");
    load1->attr.api.type = ApiType::kAPITypeCompute;
    load1->attr.api.unit = ComputeUnit::kUnitMTE2;

    auto load2 = graph.FindNode("load2");
    load2->attr.api.type = ApiType::kAPITypeCompute;
    load2->attr.api.unit = ComputeUnit::kUnitMTE2;

    auto concat = graph.FindNode("concat");
    concat->attr.api.type = ApiType::kAPITypeCompute;
    concat->attr.api.unit = ComputeUnit::kUnitVector;

    auto store = graph.FindNode("store");
    store->attr.api.type = ApiType::kAPITypeCompute;
    store->attr.api.unit = ComputeUnit::kUnitMTE2;

    auto y = graph.FindNode("y");
    y->attr.api.type = ApiType::kAPITypeBuffer;
    y->attr.api.unit = ComputeUnit::kUnitNone;
}

void LoadConcatStore_AfterScheduler(ge::AscGraph &graph, int32_t alignment) {
    int32_t input_alignment = 8;
    int32_t output_alignment = 8;
    if (alignment != -1) {  // 小尾轴场景
      input_alignment = alignment;
      output_alignment = 1;
    }
    auto all_axis = graph.GetAllAxis();
    auto z0 = all_axis[0]->id;
    auto zo = all_axis[1]->id;
    auto zo_s_0 = all_axis[2]->id;
    auto zo_s_1 = all_axis[3]->id;

    auto [z0T, z0t] = graph.TileSplit(z0);
    auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);
    vector<AxisId> vectorized_axis{z0t->id, zo};
    vector<ge::Expression> vectorized_strides{One, One};
    uint32_t align_size = 32 / sizeof(float);
    vectorized_strides[0] = ge::sym::Align(graph.FindAxis(vectorized_axis[1])->size, output_alignment);
    vector<ge::Expression> vectorized_strides_x1{ge::sym::Align(graph.FindAxis(zo_s_0)->size, input_alignment), One};
    vector<ge::Expression> vectorized_strides_x2{ge::sym::Align(graph.FindAxis(zo_s_1)->size, input_alignment), One};

    // ApplySplit on x, load, abs, store
    auto x1 = graph.FindNode("x1");
    graph.ApplySplit(x1, z0T->id, z0t->id);
    graph.ApplySplit(x1, z0TB->id, z0Tb->id);
    x1->attr.sched.loop_axis = z0Tb->id;
    x1->outputs[0].attr.vectorized_axis = {z0t->id, zo_s_0};
    x1->outputs[0].attr.vectorized_strides = vectorized_strides_x1;

    auto x2 = graph.FindNode("x2");
    graph.ApplySplit(x2, z0T->id, z0t->id);
    graph.ApplySplit(x2, z0TB->id, z0Tb->id);
    x2->attr.sched.loop_axis = z0Tb->id;
    x2->outputs[0].attr.vectorized_axis = {z0t->id, zo_s_1};
    x2->outputs[0].attr.vectorized_strides = vectorized_strides_x2;

    auto load1 = graph.FindNode("load1");
    graph.ApplySplit(load1, z0T->id, z0t->id);
    graph.ApplySplit(load1, z0TB->id, z0Tb->id);
    load1->attr.sched.loop_axis = z0Tb->id;
    load1->outputs[0].attr.vectorized_axis = {z0t->id, zo_s_0};
    load1->outputs[0].attr.vectorized_strides = vectorized_strides_x1;

    auto load2 = graph.FindNode("load2");
    graph.ApplySplit(load2, z0T->id, z0t->id);
    graph.ApplySplit(load2, z0TB->id, z0Tb->id);
    load2->attr.sched.loop_axis = z0Tb->id;
    load2->outputs[0].attr.vectorized_axis = {z0t->id, zo_s_1};
    load2->outputs[0].attr.vectorized_strides = vectorized_strides_x2;

    auto concat = graph.FindNode("concat");
    graph.ApplySplit(concat, z0T->id, z0t->id);
    graph.ApplySplit(concat, z0TB->id, z0Tb->id);
    concat->attr.sched.loop_axis = z0Tb->id;
    concat->outputs[0].attr.vectorized_axis = vectorized_axis;
    concat->outputs[0].attr.vectorized_strides = vectorized_strides;

    auto store = graph.FindNode("store");
    graph.ApplySplit(store, z0T->id, z0t->id);
    graph.ApplySplit(store, z0TB->id, z0Tb->id);
    store->attr.sched.loop_axis = z0Tb->id;
    store->outputs[0].attr.vectorized_axis = vectorized_axis;
    store->outputs[0].attr.vectorized_strides = vectorized_strides;
}

void LoadConcatStore_AfterSchedulerConcatInterAxis(ge::AscGraph &graph) {
    auto all_axis = graph.GetAllAxis();
    auto z0 = all_axis[0]->id;
    auto z1 = all_axis[1]->id;
    auto z1_s_0 = all_axis[2]->id;
    auto z1_s_1 = all_axis[3]->id;
    auto z2 = all_axis[4]->id;

    auto [z0T, z0t] = graph.TileSplit(z0);
    auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);
    vector<AxisId> vectorized_axis{z0t->id, z1, z2};
    vector<ge::Expression> vectorized_strides{graph.FindAxis(z1)->size * ge::sym::Align(graph.FindAxis(z2)->size, 8),
      ge::sym::Align(graph.FindAxis(z2)->size, 8), One};
    uint32_t align_size = 32 / sizeof(float);
    vector<ge::Expression> vectorized_strides_x1{graph.FindAxis(z1_s_0)->size * ge::sym::Align(graph.FindAxis(z2)->size, 8),
      ge::sym::Align(graph.FindAxis(z2)->size, 8), One};
    vector<ge::Expression> vectorized_strides_x2{graph.FindAxis(z1_s_1)->size * ge::sym::Align(graph.FindAxis(z2)->size, 8),
      ge::sym::Align(graph.FindAxis(z2)->size, 8), One};

    // ApplySplit on x, load, abs, store
    auto x1 = graph.FindNode("x1");
    graph.ApplySplit(x1, z0T->id, z0t->id);
    graph.ApplySplit(x1, z0TB->id, z0Tb->id);
    x1->attr.sched.loop_axis = z0Tb->id;
    x1->outputs[0].attr.vectorized_axis = {z0t->id, z1_s_0, z2};
    x1->outputs[0].attr.vectorized_strides = vectorized_strides_x1;

    auto x2 = graph.FindNode("x2");
    graph.ApplySplit(x2, z0T->id, z0t->id);
    graph.ApplySplit(x2, z0TB->id, z0Tb->id);
    x2->attr.sched.loop_axis = z0Tb->id;
    x2->outputs[0].attr.vectorized_axis = {z0t->id, z1_s_1, z2};
    x2->outputs[0].attr.vectorized_strides = vectorized_strides_x2;

    auto load1 = graph.FindNode("load1");
    graph.ApplySplit(load1, z0T->id, z0t->id);
    graph.ApplySplit(load1, z0TB->id, z0Tb->id);
    load1->attr.sched.loop_axis = z0Tb->id;
    load1->outputs[0].attr.vectorized_axis = {z0t->id, z1_s_0, z2};
    load1->outputs[0].attr.vectorized_strides = vectorized_strides_x1;

    auto load2 = graph.FindNode("load2");
    graph.ApplySplit(load2, z0T->id, z0t->id);
    graph.ApplySplit(load2, z0TB->id, z0Tb->id);
    load2->attr.sched.loop_axis = z0Tb->id;
    load2->outputs[0].attr.vectorized_axis = {z0t->id, z1_s_1, z2};
    load2->outputs[0].attr.vectorized_strides = vectorized_strides_x2;

    auto concat = graph.FindNode("concat");
    graph.ApplySplit(concat, z0T->id, z0t->id);
    graph.ApplySplit(concat, z0TB->id, z0Tb->id);
    concat->attr.sched.loop_axis = z0Tb->id;
    concat->outputs[0].attr.vectorized_axis = vectorized_axis;
    concat->outputs[0].attr.vectorized_strides = vectorized_strides;

    auto store = graph.FindNode("store");
    graph.ApplySplit(store, z0T->id, z0t->id);
    graph.ApplySplit(store, z0TB->id, z0Tb->id);
    store->attr.sched.loop_axis = z0Tb->id;
    store->outputs[0].attr.vectorized_axis = vectorized_axis;
    store->outputs[0].attr.vectorized_strides = vectorized_strides;
}

void LoadConcatStore_AfterScheduler3dLastAxis(ge::AscGraph &graph) {
    auto all_axis = graph.GetAllAxis();
    auto z0 = all_axis[0]->id;
    auto z1 = all_axis[1]->id;
    auto z2 = all_axis[2]->id;
    auto z2_s_0 = all_axis[3]->id;
    auto z2_s_1 = all_axis[4]->id;

    auto [z0T, z0t] = graph.TileSplit(z0);
    auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);
    vector<AxisId> vectorized_axis{z0t->id, z1, z2};
    vector<ge::Expression> vectorized_strides{graph.FindAxis(z1)->size * ge::sym::Align(graph.FindAxis(z2)->size, 8),
      ge::sym::Align(graph.FindAxis(z2)->size, 8), One};
    uint32_t align_size = 32 / sizeof(float);
    vector<ge::Expression> vectorized_strides_x1{graph.FindAxis(z1)->size * ge::sym::Align(graph.FindAxis(z2_s_0)->size, 8),
      ge::sym::Align(graph.FindAxis(z2_s_0)->size, 8), One};
    vector<ge::Expression> vectorized_strides_x2{graph.FindAxis(z1)->size * ge::sym::Align(graph.FindAxis(z2_s_1)->size, 8),
      ge::sym::Align(graph.FindAxis(z2_s_1)->size, 8), One};

    // ApplySplit on x, load, abs, store
    auto x1 = graph.FindNode("x1");
    graph.ApplySplit(x1, z0T->id, z0t->id);
    graph.ApplySplit(x1, z0TB->id, z0Tb->id);
    x1->attr.sched.loop_axis = z0Tb->id;
    x1->outputs[0].attr.vectorized_axis = {z0t->id, z1, z2_s_0};
    x1->outputs[0].attr.vectorized_strides = vectorized_strides_x1;

    auto x2 = graph.FindNode("x2");
    graph.ApplySplit(x2, z0T->id, z0t->id);
    graph.ApplySplit(x2, z0TB->id, z0Tb->id);
    x2->attr.sched.loop_axis = z0Tb->id;
    x2->outputs[0].attr.vectorized_axis = {z0t->id, z1, z2_s_1};
    x2->outputs[0].attr.vectorized_strides = vectorized_strides_x2;

    auto load1 = graph.FindNode("load1");
    graph.ApplySplit(load1, z0T->id, z0t->id);
    graph.ApplySplit(load1, z0TB->id, z0Tb->id);
    load1->attr.sched.loop_axis = z0Tb->id;
    load1->outputs[0].attr.vectorized_axis = {z0t->id, z1, z2_s_0};
    load1->outputs[0].attr.vectorized_strides = vectorized_strides_x1;

    auto load2 = graph.FindNode("load2");
    graph.ApplySplit(load2, z0T->id, z0t->id);
    graph.ApplySplit(load2, z0TB->id, z0Tb->id);
    load2->attr.sched.loop_axis = z0Tb->id;
    load2->outputs[0].attr.vectorized_axis = {z0t->id, z1, z2_s_1};
    load2->outputs[0].attr.vectorized_strides = vectorized_strides_x2;

    auto concat = graph.FindNode("concat");
    graph.ApplySplit(concat, z0T->id, z0t->id);
    graph.ApplySplit(concat, z0TB->id, z0Tb->id);
    concat->attr.sched.loop_axis = z0Tb->id;
    concat->outputs[0].attr.vectorized_axis = vectorized_axis;
    concat->outputs[0].attr.vectorized_strides = vectorized_strides;

    auto store = graph.FindNode("store");
    graph.ApplySplit(store, z0T->id, z0t->id);
    graph.ApplySplit(store, z0TB->id, z0Tb->id);
    store->attr.sched.loop_axis = z0Tb->id;
    store->outputs[0].attr.vectorized_axis = vectorized_axis;
    store->outputs[0].attr.vectorized_strides = vectorized_strides;
}

void LoadConcatStore_AfterQueBufAlloc(ge::AscGraph &graph) {
    auto x1 = graph.FindNode("x1");
    x1->outputs[0].attr.mem.tensor_id = 0;
    x1->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
    x1->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
    x1->outputs[0].attr.mem.position = Position::kPositionGM;
    x1->outputs[0].attr.buf.id = ge::kIdNone;
    x1->outputs[0].attr.que.id = ge::kIdNone;
    x1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
    x1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

    auto x2 = graph.FindNode("x2");
    x2->outputs[0].attr.mem.tensor_id = 1;
    x2->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
    x2->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
    x2->outputs[0].attr.mem.position = Position::kPositionGM;
    x2->outputs[0].attr.buf.id = ge::kIdNone;
    x2->outputs[0].attr.que.id = ge::kIdNone;
    x2->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
    x2->outputs[0].attr.opt.merge_scope = ge::kIdNone;

    auto load1 = graph.FindNode("load1");
    load1->outputs[0].attr.mem.tensor_id = 2;
    load1->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
    load1->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
    load1->outputs[0].attr.mem.position = Position::kPositionVecIn;
    load1->outputs[0].attr.buf.id = ge::kIdNone;
    load1->outputs[0].attr.que.id = 0;
    load1->outputs[0].attr.mem.reuse_id = 0;
    load1->outputs[0].attr.que.depth = 2;
    load1->outputs[0].attr.que.buf_num = 2;
    load1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
    load1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

    auto load2 = graph.FindNode("load2");
    load2->outputs[0].attr.mem.tensor_id = 3;
    load2->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
    load2->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
    load2->outputs[0].attr.mem.position = Position::kPositionVecIn;
    load2->outputs[0].attr.buf.id = ge::kIdNone;
    load2->outputs[0].attr.que.id = 1;
    load2->outputs[0].attr.mem.reuse_id = 1;
    load2->outputs[0].attr.que.depth = 2;
    load2->outputs[0].attr.que.buf_num = 2;
    load2->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
    load2->outputs[0].attr.opt.merge_scope = ge::kIdNone;

    auto concat = graph.FindNode("concat");
    concat->outputs[0].attr.mem.tensor_id = 4;
    concat->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
    concat->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
    concat->outputs[0].attr.mem.position = Position::kPositionVecOut;
    concat->outputs[0].attr.buf.id = ge::kIdNone;
    concat->outputs[0].attr.que.id = 2;
    concat->outputs[0].attr.mem.reuse_id = 2;
    concat->outputs[0].attr.que.depth = 2;
    concat->outputs[0].attr.que.buf_num = 2;
    concat->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
    concat->outputs[0].attr.opt.merge_scope = ge::kIdNone;

    auto store = graph.FindNode("store");
    store->outputs[0].attr.mem.tensor_id = 5;
    store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
    store->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
    store->outputs[0].attr.mem.position = Position::kPositionGM;
    store->outputs[0].attr.buf.id = ge::kIdNone;
    store->outputs[0].attr.que.id = ge::kIdNone;
    store->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
    store->outputs[0].attr.opt.merge_scope = ge::kIdNone;
}

void LoadConcatStore_BeforeAutofuse7Inputs(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");
  auto s5 = graph.CreateSizeVar("s5");
  auto s6 = graph.CreateSizeVar("s6");
  auto s7 = graph.CreateSizeVar("s7");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1 + s2 + s3 + s4 + s5 + s6 + s7);
  auto z1_s_1 = graph.CreateAxis("z1_s_1", Axis::Type::kAxisTypeOriginal, s1, {z1.id}, ge::kIdNone);
  auto z1_s_2 = graph.CreateAxis("z1_s_2", Axis::Type::kAxisTypeOriginal, s2, {z1.id}, ge::kIdNone);
  auto z1_s_3 = graph.CreateAxis("z1_s_3", Axis::Type::kAxisTypeOriginal, s3, {z1.id}, ge::kIdNone);
  auto z1_s_4 = graph.CreateAxis("z1_s_4", Axis::Type::kAxisTypeOriginal, s4, {z1.id}, ge::kIdNone);
  auto z1_s_5 = graph.CreateAxis("z1_s_5", Axis::Type::kAxisTypeOriginal, s5, {z1.id}, ge::kIdNone);
  auto z1_s_6 = graph.CreateAxis("z1_s_6", Axis::Type::kAxisTypeOriginal, s6, {z1.id}, ge::kIdNone);
  auto z1_s_7 = graph.CreateAxis("z1_s_7", Axis::Type::kAxisTypeOriginal, s7, {z1.id}, ge::kIdNone);

  Data x1("x1", graph);
  Data x2("x2", graph);
  Data x3("x3", graph);
  Data x4("x4", graph);
  Data x5("x5", graph);
  Data x6("x6", graph);
  Data x7("x7", graph);
  Load load1("load1");
  Load load2("load2");
  Load load3("load3");
  Load load4("load4");
  Load load5("load5");
  Load load6("load6");
  Load load7("load7");
  ge::ascir_op::Concat concat("concat");
  Store store("store");
  Output y("y");

  x1.attr.sched.axis = {z0.id, z1_s_1.id};
  x1.y.dtype = ge::DT_INT64;
  *x1.y.axis = {z0.id, z1_s_1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, One};

  x2.attr.sched.axis = {z0.id, z1_s_2.id};
  x2.y.dtype = ge::DT_INT64;
  *x2.y.axis = {z0.id, z1_s_2.id};
  *x2.y.repeats = {s0, s2};
  *x2.y.strides = {s2, One};

  x3.attr.sched.axis = {z0.id, z1_s_3.id};
  x3.y.dtype = ge::DT_INT64;
  *x3.y.axis = {z0.id, z1_s_3.id};
  *x3.y.repeats = {s0, s3};
  *x3.y.strides = {s3, One};

  x4.attr.sched.axis = {z0.id, z1_s_4.id};
  x4.y.dtype = ge::DT_INT64;
  *x4.y.axis = {z0.id, z1_s_4.id};
  *x4.y.repeats = {s0, s4};
  *x4.y.strides = {s4, One};

  x5.attr.sched.axis = {z0.id, z1_s_5.id};
  x5.y.dtype = ge::DT_INT64;
  *x5.y.axis = {z0.id, z1_s_5.id};
  *x5.y.repeats = {s0, s5};
  *x5.y.strides = {s5, One};

  x6.attr.sched.axis = {z0.id, z1_s_6.id};
  x6.y.dtype = ge::DT_INT64;
  *x6.y.axis = {z0.id, z1_s_6.id};
  *x6.y.repeats = {s0, s6};
  *x6.y.strides = {s6, One};

  x7.attr.sched.axis = {z0.id, z1_s_7.id};
  x7.y.dtype = ge::DT_INT64;
  *x7.y.axis = {z0.id, z1_s_7.id};
  *x7.y.repeats = {s0, s7};
  *x7.y.strides = {s7, One};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1_s_1.id};
  load1.y.dtype = ge::DT_INT64;
  *load1.y.axis = {z0.id, z1_s_1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, One};

  load2.x = x2.y;
  load2.attr.sched.axis = {z0.id, z1_s_2.id};
  load2.y.dtype = ge::DT_INT64;
  *load2.y.axis = {z0.id, z1_s_2.id};
  *load2.y.repeats = {s0, s2};
  *load2.y.strides = {s2, One};

  load3.x = x3.y;
  load3.attr.sched.axis = {z0.id, z1_s_3.id};
  load3.y.dtype = ge::DT_INT64;
  *load3.y.axis = {z0.id, z1_s_3.id};
  *load3.y.repeats = {s0, s3};
  *load3.y.strides = {s3, One};
  
  load4.x = x4.y;
  load4.attr.sched.axis = {z0.id, z1_s_4.id};
  load4.y.dtype = ge::DT_INT64;
  *load4.y.axis = {z0.id, z1_s_4.id};
  *load4.y.repeats = {s0, s4};
  *load4.y.strides = {s4, One};

  load5.x = x5.y;
  load5.attr.sched.axis = {z0.id, z1_s_5.id};
  load5.y.dtype = ge::DT_INT64;
  *load5.y.axis = {z0.id, z1_s_5.id};
  *load5.y.repeats = {s0, s5};
  *load5.y.strides = {s5, One};

  load6.x = x6.y;
  load6.attr.sched.axis = {z0.id, z1_s_6.id};
  load6.y.dtype = ge::DT_INT64;
  *load6.y.axis = {z0.id, z1_s_6.id};
  *load6.y.repeats = {s0, s6};
  *load6.y.strides = {s6, One};

  load7.x = x7.y;
  load7.attr.sched.axis = {z0.id, z1_s_7.id};
  load7.y.dtype = ge::DT_INT64;
  *load7.y.axis = {z0.id, z1_s_7.id};
  *load7.y.repeats = {s0, s7};
  *load7.y.strides = {s7, One};

  concat.x = {load1.y, load2.y, load3.y, load4.y, load5.y, load6.y, load7.y};
  concat.attr.sched.axis = {z0.id, z1.id};
  concat.y.dtype = ge::DT_INT64;
  *concat.y.axis = {z0.id, z1.id};
  *concat.y.repeats = {s0, s1 + s2 + s3 + s4 + s5 + s6 + s7};
  *concat.y.strides = {s1 + s2 + s3 + s4 + s5 + s6 + s7, One};
  concat.attr.tmp_buffers = {{{ge::Symbol(16384), -1}, MemAttr(), 0}};

  store.x = concat.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = ge::DT_INT64;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1 + s2 + s3 + s4 + s5 + s6 + s7};
  *store.y.strides = {s1 + s2 + s3 + s4 + s5 + s6 + s7, One};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = ge::DT_INT64;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, s1 + s2 + s3 + s4 + s5 + s6 + s7};
  *y.y.strides = {s1 + s2 + s3 + s4 + s5 + s6 + s7, One};
}

void LoadConcatStore_AfterInferOutput7Inputs(ge::AscGraph &graph) {
  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = ComputeType::kComputeInvalid;

  auto x2 = graph.FindNode("x2");
  x2->attr.api.compute_type = ComputeType::kComputeInvalid;

  auto x3 = graph.FindNode("x3");
  x2->attr.api.compute_type = ComputeType::kComputeInvalid;

  auto x4 = graph.FindNode("x4");
  x2->attr.api.compute_type = ComputeType::kComputeInvalid;

  auto x5 = graph.FindNode("x5");
  x2->attr.api.compute_type = ComputeType::kComputeInvalid;

  auto x6 = graph.FindNode("x6");
  x2->attr.api.compute_type = ComputeType::kComputeInvalid;

  auto x7 = graph.FindNode("x7");
  x2->attr.api.compute_type = ComputeType::kComputeInvalid;

  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.dtype = ge::DT_INT64;
  load1->attr.api.compute_type = ComputeType::kComputeLoad;

  auto load2 = graph.FindNode("load2");
  load2->outputs[0].attr.dtype = ge::DT_INT64;
  load2->attr.api.compute_type = ComputeType::kComputeLoad;

  auto load3 = graph.FindNode("load3");
  load3->outputs[0].attr.dtype = ge::DT_INT64;
  load3->attr.api.compute_type = ComputeType::kComputeLoad;

  auto load4 = graph.FindNode("load4");
  load4->outputs[0].attr.dtype = ge::DT_INT64;
  load4->attr.api.compute_type = ComputeType::kComputeLoad;


  auto load5 = graph.FindNode("load5");
  load5->outputs[0].attr.dtype = ge::DT_INT64;
  load5->attr.api.compute_type = ComputeType::kComputeLoad;

  auto load6 = graph.FindNode("load6");
  load6->outputs[0].attr.dtype = ge::DT_INT64;
  load6->attr.api.compute_type = ComputeType::kComputeLoad;

  auto load7 = graph.FindNode("load7");
  load7->outputs[0].attr.dtype = ge::DT_INT64;
  load7->attr.api.compute_type = ComputeType::kComputeLoad;

  auto concat = graph.FindNode("concat");
  concat->outputs[0].attr.dtype =(ge::DataType)load1->outputs[0].attr.dtype;
  concat->attr.api.compute_type = ComputeType::kComputeConcat;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.dtype = (ge::DataType)concat->outputs[0].attr.dtype;
  store->attr.api.compute_type = ComputeType::kComputeStore;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = ComputeType::kComputeInvalid;
  y->outputs[0].attr.dtype = (ge::DataType)concat->outputs[0].attr.dtype;
}

void LoadConcatStore_AfterGetApiInfo7Inputs(ge::AscGraph &graph) {
  auto x1 = graph.FindNode("x1");
  x1->attr.api.type = ApiType::kAPITypeBuffer;
  x1->attr.api.unit = ComputeUnit::kUnitNone;

  auto x2 = graph.FindNode("x2");
  x2->attr.api.type = ApiType::kAPITypeBuffer;
  x2->attr.api.unit = ComputeUnit::kUnitNone;

  auto x3 = graph.FindNode("x3");
  x3->attr.api.type = ApiType::kAPITypeBuffer;
  x3->attr.api.unit = ComputeUnit::kUnitNone;

  auto x4 = graph.FindNode("x4");
  x4->attr.api.type = ApiType::kAPITypeBuffer;
  x4->attr.api.unit = ComputeUnit::kUnitNone;

  auto x5 = graph.FindNode("x5");
  x5->attr.api.type = ApiType::kAPITypeBuffer;
  x5->attr.api.unit = ComputeUnit::kUnitNone;

  auto x6 = graph.FindNode("x6");
  x6->attr.api.type = ApiType::kAPITypeBuffer;
  x6->attr.api.unit = ComputeUnit::kUnitNone;

  auto x7 = graph.FindNode("x7");
  x7->attr.api.type = ApiType::kAPITypeBuffer;
  x7->attr.api.unit = ComputeUnit::kUnitNone;

  auto load1 = graph.FindNode("load1");
  load1->attr.api.type = ApiType::kAPITypeCompute;
  load1->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.type = ApiType::kAPITypeCompute;
  load2->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto load3 = graph.FindNode("load3");
  load3->attr.api.type = ApiType::kAPITypeCompute;
  load3->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto load4 = graph.FindNode("load4");
  load4->attr.api.type = ApiType::kAPITypeCompute;
  load4->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto load5 = graph.FindNode("load5");
  load5->attr.api.type = ApiType::kAPITypeCompute;
  load5->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto load6 = graph.FindNode("load6");
  load6->attr.api.type = ApiType::kAPITypeCompute;
  load6->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto load7 = graph.FindNode("load7");
  load7->attr.api.type = ApiType::kAPITypeCompute;
  load7->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto concat = graph.FindNode("concat");
  concat->attr.api.type = ApiType::kAPITypeCompute;
  concat->attr.api.unit = ComputeUnit::kUnitVector;

  auto store = graph.FindNode("store");
  store->attr.api.type = ApiType::kAPITypeCompute;
  store->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.type = ApiType::kAPITypeBuffer;
  y->attr.api.unit = ComputeUnit::kUnitNone;
}

void LoadConcatStore_AfterScheduler7Inputs(ge::AscGraph &graph) {
  auto all_axis = graph.GetAllAxis();
  auto z0 = all_axis[0]->id;
  auto z1 = all_axis[1]->id;
  auto z1_s_1 = all_axis[2]->id;
  auto z1_s_2 = all_axis[3]->id;
  auto z1_s_3 = all_axis[4]->id;
  auto z1_s_4 = all_axis[5]->id;
  auto z1_s_5 = all_axis[6]->id;
  auto z1_s_6 = all_axis[7]->id;
  auto z1_s_7 = all_axis[8]->id;

  auto [z0T, z0t] = graph.TileSplit(z0);
  auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);
  vector<AxisId> vectorized_axis{z0t->id, z1};
  vector<ge::Expression> vectorized_strides{One, One};
  uint32_t align_size = 32 / sizeof(int64_t);
  vectorized_strides[0] = ge::sym::Align(graph.FindAxis(vectorized_axis[1])->size, align_size);
  vector<ge::Expression> vectorized_strides_x1{ge::sym::Align(graph.FindAxis(z1_s_1)->size, align_size), One};
  vector<ge::Expression> vectorized_strides_x2{ge::sym::Align(graph.FindAxis(z1_s_2)->size, align_size), One};
  vector<ge::Expression> vectorized_strides_x3{ge::sym::Align(graph.FindAxis(z1_s_3)->size, align_size), One};
  vector<ge::Expression> vectorized_strides_x4{ge::sym::Align(graph.FindAxis(z1_s_4)->size, align_size), One};
  vector<ge::Expression> vectorized_strides_x5{ge::sym::Align(graph.FindAxis(z1_s_5)->size, align_size), One};
  vector<ge::Expression> vectorized_strides_x6{ge::sym::Align(graph.FindAxis(z1_s_6)->size, align_size), One};
  vector<ge::Expression> vectorized_strides_x7{ge::sym::Align(graph.FindAxis(z1_s_7)->size, align_size), One};

  // ApplySplit on x, load, abs, store
  auto x1 = graph.FindNode("x1");
  graph.ApplySplit(x1, z0T->id, z0t->id);
  graph.ApplySplit(x1, z0TB->id, z0Tb->id);
  x1->attr.sched.loop_axis = z0Tb->id;
  x1->outputs[0].attr.vectorized_axis = {z0t->id, z1_s_1};
  x1->outputs[0].attr.vectorized_strides = vectorized_strides_x1;

  auto x2 = graph.FindNode("x2");
  graph.ApplySplit(x2, z0T->id, z0t->id);
  graph.ApplySplit(x2, z0TB->id, z0Tb->id);
  x2->attr.sched.loop_axis = z0Tb->id;
  x2->outputs[0].attr.vectorized_axis = {z0t->id, z1_s_2};
  x2->outputs[0].attr.vectorized_strides = vectorized_strides_x2;

  auto x3 = graph.FindNode("x3");
  graph.ApplySplit(x3, z0T->id, z0t->id);
  graph.ApplySplit(x3, z0TB->id, z0Tb->id);
  x3->attr.sched.loop_axis = z0Tb->id;
  x3->outputs[0].attr.vectorized_axis = {z0t->id, z1_s_3};
  x3->outputs[0].attr.vectorized_strides = vectorized_strides_x3;

  auto x4 = graph.FindNode("x4");
  graph.ApplySplit(x4, z0T->id, z0t->id);
  graph.ApplySplit(x4, z0TB->id, z0Tb->id);
  x4->attr.sched.loop_axis = z0Tb->id;
  x4->outputs[0].attr.vectorized_axis = {z0t->id, z1_s_4};
  x4->outputs[0].attr.vectorized_strides = vectorized_strides_x4;


  auto x5 = graph.FindNode("x5");
  graph.ApplySplit(x5, z0T->id, z0t->id);
  graph.ApplySplit(x5, z0TB->id, z0Tb->id);
  x5->attr.sched.loop_axis = z0Tb->id;
  x5->outputs[0].attr.vectorized_axis = {z0t->id, z1_s_5};
  x5->outputs[0].attr.vectorized_strides = vectorized_strides_x5;

  auto x6 = graph.FindNode("x6");
  graph.ApplySplit(x6, z0T->id, z0t->id);
  graph.ApplySplit(x6, z0TB->id, z0Tb->id);
  x6->attr.sched.loop_axis = z0Tb->id;
  x6->outputs[0].attr.vectorized_axis = {z0t->id, z1_s_6};
  x6->outputs[0].attr.vectorized_strides = vectorized_strides_x6;

  auto x7 = graph.FindNode("x7");
  graph.ApplySplit(x7, z0T->id, z0t->id);
  graph.ApplySplit(x7, z0TB->id, z0Tb->id);
  x7->attr.sched.loop_axis = z0Tb->id;
  x7->outputs[0].attr.vectorized_axis = {z0t->id, z1_s_7};
  x7->outputs[0].attr.vectorized_strides = vectorized_strides_x7;

  auto load1 = graph.FindNode("load1");
  graph.ApplySplit(load1, z0T->id, z0t->id);
  graph.ApplySplit(load1, z0TB->id, z0Tb->id);
  load1->attr.sched.loop_axis = z0Tb->id;
  load1->outputs[0].attr.vectorized_axis = {z0t->id, z1_s_1};
  load1->outputs[0].attr.vectorized_strides = vectorized_strides_x1;

  auto load2 = graph.FindNode("load2");
  graph.ApplySplit(load2, z0T->id, z0t->id);
  graph.ApplySplit(load2, z0TB->id, z0Tb->id);
  load2->attr.sched.loop_axis = z0Tb->id;
  load2->outputs[0].attr.vectorized_axis = {z0t->id, z1_s_2};
  load2->outputs[0].attr.vectorized_strides = vectorized_strides_x2;

  auto load3 = graph.FindNode("load3");
  graph.ApplySplit(load3, z0T->id, z0t->id);
  graph.ApplySplit(load3, z0TB->id, z0Tb->id);
  load3->attr.sched.loop_axis = z0Tb->id;
  load3->outputs[0].attr.vectorized_axis = {z0t->id, z1_s_3};
  load3->outputs[0].attr.vectorized_strides = vectorized_strides_x3;

  auto load4 = graph.FindNode("load4");
  graph.ApplySplit(load4, z0T->id, z0t->id);
  graph.ApplySplit(load4, z0TB->id, z0Tb->id);
  load4->attr.sched.loop_axis = z0Tb->id;
  load4->outputs[0].attr.vectorized_axis = {z0t->id, z1_s_4};
  load4->outputs[0].attr.vectorized_strides = vectorized_strides_x4;

  auto load5 = graph.FindNode("load5");
  graph.ApplySplit(load5, z0T->id, z0t->id);
  graph.ApplySplit(load5, z0TB->id, z0Tb->id);
  load5->attr.sched.loop_axis = z0Tb->id;
  load5->outputs[0].attr.vectorized_axis = {z0t->id, z1_s_5};
  load5->outputs[0].attr.vectorized_strides = vectorized_strides_x5;

  auto load6 = graph.FindNode("load6");
  graph.ApplySplit(load6, z0T->id, z0t->id);
  graph.ApplySplit(load6, z0TB->id, z0Tb->id);
  load6->attr.sched.loop_axis = z0Tb->id;
  load6->outputs[0].attr.vectorized_axis = {z0t->id, z1_s_6};
  load6->outputs[0].attr.vectorized_strides = vectorized_strides_x6;

  auto load7 = graph.FindNode("load7");
  graph.ApplySplit(load7, z0T->id, z0t->id);
  graph.ApplySplit(load7, z0TB->id, z0Tb->id);
  load7->attr.sched.loop_axis = z0Tb->id;
  load7->outputs[0].attr.vectorized_axis = {z0t->id, z1_s_7};
  load7->outputs[0].attr.vectorized_strides = vectorized_strides_x7;

  auto concat = graph.FindNode("concat");
  graph.ApplySplit(concat, z0T->id, z0t->id);
  graph.ApplySplit(concat, z0TB->id, z0Tb->id);
  concat->attr.sched.loop_axis = z0Tb->id;
  concat->outputs[0].attr.vectorized_axis = vectorized_axis;
  concat->outputs[0].attr.vectorized_strides = vectorized_strides;

  auto store = graph.FindNode("store");
  graph.ApplySplit(store, z0T->id, z0t->id);
  graph.ApplySplit(store, z0TB->id, z0Tb->id);
  store->attr.sched.loop_axis = z0Tb->id;
  store->outputs[0].attr.vectorized_axis = vectorized_axis;
  store->outputs[0].attr.vectorized_strides = vectorized_strides;
}

void LoadConcatStore_AfterQueBufAlloc7Inputs(ge::AscGraph &graph) {
  auto x1 = graph.FindNode("x1");
  x1->outputs[0].attr.mem.tensor_id = 0;
  x1->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = Position::kPositionGM;
  x1->outputs[0].attr.buf.id = ge::kIdNone;
  x1->outputs[0].attr.que.id = ge::kIdNone;
  x1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto x2 = graph.FindNode("x2");
  x2->outputs[0].attr.mem.tensor_id = 1;
  x2->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x2->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x2->outputs[0].attr.mem.position = Position::kPositionGM;
  x2->outputs[0].attr.buf.id = ge::kIdNone;
  x2->outputs[0].attr.que.id = ge::kIdNone;
  x2->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x2->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto x3 = graph.FindNode("x3");
  x3->outputs[0].attr.mem.tensor_id = 2;
  x3->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x3->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x3->outputs[0].attr.mem.position = Position::kPositionGM;
  x3->outputs[0].attr.buf.id = ge::kIdNone;
  x3->outputs[0].attr.que.id = ge::kIdNone;
  x3->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x3->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto x4 = graph.FindNode("x4");
  x4->outputs[0].attr.mem.tensor_id = 3;
  x4->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x4->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x4->outputs[0].attr.mem.position = Position::kPositionGM;
  x4->outputs[0].attr.buf.id = ge::kIdNone;
  x4->outputs[0].attr.que.id = ge::kIdNone;
  x4->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x4->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto x5 = graph.FindNode("x5");
  x5->outputs[0].attr.mem.tensor_id = 4;
  x5->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x5->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x5->outputs[0].attr.mem.position = Position::kPositionGM;
  x5->outputs[0].attr.buf.id = ge::kIdNone;
  x5->outputs[0].attr.que.id = ge::kIdNone;
  x5->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x5->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto x6 = graph.FindNode("x6");
  x6->outputs[0].attr.mem.tensor_id = 5;
  x6->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x6->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x6->outputs[0].attr.mem.position = Position::kPositionGM;
  x6->outputs[0].attr.buf.id = ge::kIdNone;
  x6->outputs[0].attr.que.id = ge::kIdNone;
  x6->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x6->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto x7 = graph.FindNode("x7");
  x7->outputs[0].attr.mem.tensor_id = 6;
  x7->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x7->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x7->outputs[0].attr.mem.position = Position::kPositionGM;
  x7->outputs[0].attr.buf.id = ge::kIdNone;
  x7->outputs[0].attr.que.id = ge::kIdNone;
  x7->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x7->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.mem.tensor_id = 8;
  load1->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load1->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load1->outputs[0].attr.buf.id = ge::kIdNone;
  load1->outputs[0].attr.que.id = 0;
  load1->outputs[0].attr.mem.reuse_id = 0;
  load1->outputs[0].attr.que.depth = 1;
  load1->outputs[0].attr.que.buf_num = 2;
  load1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->outputs[0].attr.mem.tensor_id = 9;
  load2->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load2->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load2->outputs[0].attr.buf.id = ge::kIdNone;
  load2->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.mem.reuse_id = 1;
  load2->outputs[0].attr.que.depth = 1;
  load2->outputs[0].attr.que.buf_num = 2;
  load2->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load2->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto load3 = graph.FindNode("load3");
  load3->outputs[0].attr.mem.tensor_id = 10;
  load3->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load3->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load3->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load3->outputs[0].attr.buf.id = ge::kIdNone;
  load3->outputs[0].attr.que.id = 2;
  load3->outputs[0].attr.mem.reuse_id = 2;
  load3->outputs[0].attr.que.depth = 1;
  load3->outputs[0].attr.que.buf_num = 2;
  load3->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load3->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto load4 = graph.FindNode("load4");
  load4->outputs[0].attr.mem.tensor_id = 11;
  load4->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load4->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load4->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load4->outputs[0].attr.buf.id = ge::kIdNone;
  load4->outputs[0].attr.que.id = 3;
  load4->outputs[0].attr.mem.reuse_id = 3;
  load4->outputs[0].attr.que.depth = 1;
  load4->outputs[0].attr.que.buf_num = 2;
  load4->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load4->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto load5 = graph.FindNode("load5");
  load5->outputs[0].attr.mem.tensor_id = 12;
  load5->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load5->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load5->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load5->outputs[0].attr.buf.id = ge::kIdNone;
  load5->outputs[0].attr.que.id = 4;
  load5->outputs[0].attr.mem.reuse_id = 4;
  load5->outputs[0].attr.que.depth = 1;
  load5->outputs[0].attr.que.buf_num = 2;
  load5->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load5->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto load6 = graph.FindNode("load6");
  load6->outputs[0].attr.mem.tensor_id = 13;
  load6->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load6->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load6->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load6->outputs[0].attr.buf.id = ge::kIdNone;
  load6->outputs[0].attr.que.id = 5;
  load6->outputs[0].attr.mem.reuse_id = 5;
  load6->outputs[0].attr.que.depth = 1;
  load6->outputs[0].attr.que.buf_num = 2;
  load6->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load6->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto load7 = graph.FindNode("load7");
  load7->outputs[0].attr.mem.tensor_id = 14;
  load7->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load7->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load7->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load7->outputs[0].attr.buf.id = ge::kIdNone;
  load7->outputs[0].attr.que.id = 6;
  load7->outputs[0].attr.mem.reuse_id = 6;
  load7->outputs[0].attr.que.depth = 1;
  load7->outputs[0].attr.que.buf_num = 2;
  load7->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load7->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto concat = graph.FindNode("concat");
  concat->outputs[0].attr.mem.tensor_id = 16;
  concat->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  concat->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  concat->outputs[0].attr.mem.position = Position::kPositionVecOut;
  concat->outputs[0].attr.buf.id = ge::kIdNone;
  concat->outputs[0].attr.que.id = 7;
  concat->outputs[0].attr.mem.reuse_id = 7;
  concat->outputs[0].attr.que.depth = 1;
  concat->outputs[0].attr.que.buf_num = 2;
  concat->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  concat->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.mem.tensor_id = 17;
  store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = Position::kPositionGM;
  store->outputs[0].attr.buf.id = ge::kIdNone;
  store->outputs[0].attr.que.id = ge::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;
}

void LoadConcatStore_SmallTailBeforeAutofuse(AscGraph &graph,
                                             ge::DataType data_type,
                                             const std::vector<int64_t> &concat_dim_sizes) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar(concat_dim_sizes[0]);
  auto s2 = graph.CreateSizeVar(concat_dim_sizes[1]);

  auto z0 = graph.CreateAxis("z0", s0);
  auto zo = graph.CreateAxis("zo", s1 + s2);
  auto zo_s_0 = graph.CreateAxis("zo_s_0", Axis::Type::kAxisTypeOriginal, s1, {zo.id}, ge::kIdNone);
  auto zo_s_1 = graph.CreateAxis("zo_s_1", Axis::Type::kAxisTypeOriginal, s2, {zo.id}, ge::kIdNone);

  Data x1("x1", graph);
  Data x2("x2", graph);
  Load load1("load1");
  Load load2("load2");
  ge::ascir_op::Concat concat("concat");
  Store store("store");
  Output y("y");

  x1.attr.sched.axis = {z0.id, zo_s_0.id};
  x1.y.dtype = data_type;
  *x1.y.axis = {z0.id, zo_s_0.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, One};

  x2.attr.sched.axis = {z0.id, zo_s_1.id};
  x2.y.dtype = data_type;
  *x2.y.axis = {z0.id, zo_s_1.id};
  *x2.y.repeats = {s0, s2};
  *x2.y.strides = {s2, One};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, zo_s_0.id};
  load1.y.dtype = data_type;
  *load1.y.axis = {z0.id, zo_s_0.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, One};

  load2.x = x2.y;
  load2.attr.sched.axis = {z0.id, zo_s_1.id};
  load2.y.dtype = data_type;
  *load2.y.axis = {z0.id, zo_s_1.id};
  *load2.y.repeats = {s0, s2};
  *load2.y.strides = {s2, One};

  concat.x = {load1.y, load2.y};
  concat.attr.sched.axis = {z0.id, zo.id};
  concat.y.dtype = data_type;
  *concat.y.axis = {z0.id, zo.id};
  *concat.y.repeats = {s0, s1 + s2};
  *concat.y.strides = {s1+s2, One};
  concat.attr.tmp_buffers = {{{ge::Symbol(16384), -1}, MemAttr(), 0}};

  store.x = concat.y;
  store.attr.sched.axis = {z0.id, zo.id};
  store.y.dtype = data_type;
  *store.y.axis = {z0.id, zo.id};
  *store.y.repeats = {s0, s1 + s2};
  *store.y.strides = {s1+s2, One};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, zo.id};
  y.y.dtype = data_type;
  *y.y.axis = {z0.id, zo.id};
  *y.y.repeats = {s0, s1 + s2};
  *y.y.strides = {s1 + s2, One};
}

void LoadConcatStore_SmallTailAfterInferOutput(AscGraph &graph) {
  LoadConcatStore_AfterInferOutput(graph);
}

void LoadConcatStore_SmallTailAfterGetApiInfo(AscGraph &graph) {
  LoadConcatStore_AfterGetApiInfo(graph);
}

void LoadConcatStore_SmallTailAfterScheduler(AscGraph &graph, int32_t alignment) {
  LoadConcatStore_AfterScheduler(graph, alignment);
}

void LoadConcatStore_SmallTailAfterQueBufAlloc(AscGraph &graph) {
  LoadConcatStore_AfterQueBufAlloc(graph);
  auto node = graph.FindNode("concat");
  ge::AttrUtils::SetBool(node->GetOpDesc(), "_concat_small_tail", true);
}
