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
#include "ascir_ops_utils.h"

using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;

void Softmax_BeforeAutofuse(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0 * s1 * s2);
  auto z1 = graph.CreateAxis("z1", s3);

  auto axis = {z0.id, z1.id};


  Data arg4_1("arg4_1");
  graph.AddNode(arg4_1);
  arg4_1.y.dtype = ge::DT_FLOAT16;

  // buf0
  Output buf0("buf0");
  Load b0_load("b0_load");
  graph.AddNode(buf0);
  graph.AddNode(b0_load);
  b0_load.x = arg4_1.y;
  b0_load.attr.sched.axis = axis;
  b0_load.y.dtype = ge::DT_FLOAT16;
  *b0_load.y.axis = axis;
  *b0_load.y.repeats = {s0*s1*s2, s3};
  *b0_load.y.strides = {s3, One};

  ge::ascir_op::Max b0_max("b0_max");
  graph.AddNode(b0_max);
  b0_max.x = b0_load.y;
  b0_max.attr.sched.axis = axis;
  b0_max.y.dtype = ge::DT_FLOAT16;
  *b0_max.y.axis = axis;
  *b0_max.y.repeats = {s0*s1*s2, s3};
  *b0_max.y.strides = {One, Zero};

  Store b0_store("b0_store");
  graph.AddNode(b0_store);
  b0_store.x = b0_max.y;
  b0_store.attr.sched.axis = axis;
  b0_store.y.dtype = ge::DT_FLOAT16;
  *b0_store.y.axis = axis;
  *b0_store.y.repeats = {s0*s1*s2, s3};
  *b0_store.y.strides = {One, Zero};

  buf0.x = b0_store.y;
  buf0.y.dtype = ge::DT_FLOAT16;

  Output buf1("buf1");
  Load b1_load("b1_load");
  graph.AddNode(buf1);
  graph.AddNode(b1_load);
  b1_load.x = arg4_1.y;
  b1_load.attr.sched.axis = axis;
  b1_load.y.dtype = ge::DT_FLOAT16;
  *b1_load.y.axis = axis;
  *b1_load.y.repeats = {s0*s1*s2, s3};
  *b1_load.y.strides = {s3, One};

  Load b1_load1("b1_load1");
  graph.AddNode(b1_load1);
  b1_load1.x = buf0.y;
  b1_load1.attr.sched.axis = axis;
  b1_load1.y.dtype = ge::DT_FLOAT16;
  *b1_load1.y.axis = axis;
  *b1_load1.y.repeats = {s0*s1*s2, s3};
  *b1_load1.y.strides = {One, Zero};

  Broadcast b1_broadcast("b1_broadcast");
  graph.AddNode(b1_broadcast);
  b1_broadcast.x = b1_load1.y;
  b1_broadcast.attr.sched.axis = axis;
  b1_broadcast.y.dtype = ge::DT_FLOAT16;
  *b1_broadcast.y.axis = axis;
  *b1_broadcast.y.repeats = {s0*s1*s2, s3};
  *b1_broadcast.y.strides = {s3, One};

  ge::ascir_op::Sub b1_sub("b1_sub");
  graph.AddNode(b1_sub);
  b1_sub.x1 = b1_load.y;
  b1_sub.x2 = b1_broadcast.y;
  b1_sub.attr.sched.axis = axis;
  b1_sub.y.dtype = ge::DT_FLOAT16;
  *b1_sub.y.axis = axis;
  *b1_sub.y.repeats = {s0*s1*s2, s3};
  *b1_sub.y.strides = {s3, One};

  Exp b1_exp("b1_exp");
  graph.AddNode(b1_exp);
  b1_exp.x = b1_sub.y;
  b1_exp.attr.sched.axis = axis;
  b1_exp.y.dtype = ge::DT_FLOAT16;
  *b1_exp.y.axis = axis;
  *b1_exp.y.repeats = {s0*s1*s2, s3};
  *b1_exp.y.strides = {s3, One};

  Store b1_store("b1_store");
  graph.AddNode(b1_store);
  b1_store.x = b1_exp.y;
  b1_store.attr.sched.axis = axis;
  b1_store.y.dtype = ge::DT_FLOAT16;
  *b1_store.y.axis = axis;
  *b1_store.y.repeats = {s0*s1*s2, s3};
  *b1_store.y.strides = {s3, One};

  buf1.x = b1_store.y;
  buf1.y.dtype = ge::DT_FLOAT16;

  Output buf2("buf2");
  Load b2_load("b2_load");
  graph.AddNode(buf2);
  graph.AddNode(b2_load);
  b2_load.x = buf1.y;
  b2_load.attr.sched.axis = axis;
  b2_load.y.dtype = ge::DT_FLOAT16;
  *b2_load.y.axis = axis;
  *b2_load.y.repeats = {s0*s1*s2, s3};
  *b2_load.y.strides = {s3, One};

  Sum b2_sum("b2_sum");
  graph.AddNode(b2_sum);
  b2_sum.x = b2_load.y;
  b2_sum.attr.sched.axis = axis;
  b2_sum.y.dtype = ge::DT_FLOAT16;
  *b2_sum.y.axis = axis;
  *b2_sum.y.repeats = {s0*s1*s2, s3};
  *b2_sum.y.strides = {One, Zero};

  Store b2_store("b2_store");
  graph.AddNode(b2_store);
  b2_store.x = b2_sum.y;
  b2_store.attr.sched.axis = axis;
  b2_store.y.dtype = ge::DT_FLOAT16;
  *b2_store.y.axis = axis;
  *b2_store.y.repeats = {s0*s1*s2, s3};
  *b2_store.y.strides = {One, Zero};

  buf2.x = b2_store.y;
  buf2.y.dtype = ge::DT_FLOAT16;

  Output buf3("buf3");
  Load b3_load("b3_load");
  graph.AddNode(buf3);
  graph.AddNode(b3_load);
  b3_load.x = buf1.y;
  b3_load.attr.sched.axis = axis;
  b3_load.y.dtype = ge::DT_FLOAT16;
  *b3_load.y.axis = axis;
  *b3_load.y.repeats = {s0*s1*s2, s3};
  *b3_load.y.strides = {s3, One};

  Load b3_load1("b3_load1");
  graph.AddNode(b3_load1);
  b3_load1.x = buf2.y;
  b3_load1.attr.sched.axis = axis;
  b3_load1.y.dtype = ge::DT_FLOAT16;
  *b3_load1.y.axis = axis;
  *b3_load1.y.repeats = {s0*s1*s2, s3};
  *b3_load1.y.strides = {One, Zero};

  Broadcast b3_broadcast("b3_broadcast");
  graph.AddNode(b3_broadcast);
  b3_broadcast.x = b3_load1.y;
  b3_broadcast.attr.sched.axis = axis;
  b3_broadcast.y.dtype = ge::DT_FLOAT16;
  *b3_broadcast.y.axis = axis;
  *b3_broadcast.y.repeats = {s0 * s1 * s2, s3};
  *b3_broadcast.y.strides = {s3, One};

  ge::ascir_op::Div b3_div("b3_div");
  graph.AddNode(b3_div);
  b3_div.x1 = b3_load.y;
  b3_div.x2 = b3_broadcast.y;
  b3_div.attr.sched.axis = axis;
  b3_div.y.dtype = ge::DT_FLOAT16;
  *b3_div.y.axis = axis;
  *b3_div.y.repeats = {s0*s1*s2, s3};
  *b3_div.y.strides = {s3, One};

  Store b3_store("b3_store");
  graph.AddNode(b3_store);
  b3_store.x = b3_div.y;
  b3_store.attr.sched.axis = axis;
  b3_store.y.dtype = ge::DT_FLOAT16;
  *b3_store.y.axis = axis;
  *b3_store.y.repeats = {s0*s1*s2, s3};
  *b3_store.y.strides = {s3, One};

  buf3.x = b3_store.y;
  buf3.y.dtype = ge::DT_FLOAT16;
}
