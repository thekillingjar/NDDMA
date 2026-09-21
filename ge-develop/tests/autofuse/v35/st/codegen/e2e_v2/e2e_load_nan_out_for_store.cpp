#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"
#include "ascir_ops_utils.h"

using namespace std;
using namespace ge;
using namespace ge::ops;
using namespace ge::ascir_op;

void LoadNanOutForStore_BeforeAutofuse(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x1("x1");
  Load load1("load1");
  ge::ascir_op::Isnan isnan("isnan");
  Store store("store");
  Output y("y");
  graph.AddNode(x1);
  graph.AddNode(load1);
  graph.AddNode(isnan);
  graph.AddNode(store);
  graph.AddNode(y);

  x1.attr.sched.axis = {z0.id, z1.id, z2.id};
  x1.y.dtype = ge::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id, z2.id};
  *x1.y.repeats = {s0, s1, s2};
  *x1.y.strides = {s1*s2, s2, One};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1*s2, s2, One};

  isnan.x = load1.y;
  isnan.attr.sched.axis = {z0.id, z1.id, z2.id};
  isnan.y.dtype = ge::DT_UINT8;
  *isnan.y.axis = {z0.id, z1.id, z2.id};
  *isnan.y.repeats = {s0, s1, s2};
  *isnan.y.strides = {s1*s2, s2, One};

  store.x = isnan.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = ge::DT_UINT8;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1*s2, s2, One};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.y.dtype = ge::DT_UINT8;
  *y.y.axis = {z0.id, z1.id, z2.id};
  *y.y.repeats = {s0, s1, s2};
  *y.y.strides = {s1*s2, s2, One};
}

void LoadNanOutForStore_AfterInferOutput(ge::AscGraph &graph) {
  auto x1 = graph.FindNode("x1");
  x1->attr.api.compute_type = ComputeType::kComputeInvalid;

  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.dtype = ge::DT_FLOAT;
  load1->attr.api.compute_type = ComputeType::kComputeLoad;

  auto isnan = graph.FindNode("isnan");
  isnan->outputs[0].attr.dtype = ge::DT_UINT8;
  isnan->outputs[0].attr.axis = load1->outputs[0].attr.axis;
  isnan->outputs[0].attr.repeats = load1->outputs[0].attr.repeats;
  isnan->outputs[0].attr.strides = load1->outputs[0].attr.strides;
  isnan->attr.api.compute_type = ComputeType::kComputeElewise;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.dtype = ge::DT_UINT8;
  store->attr.api.compute_type = ComputeType::kComputeStore;

  auto y = graph.FindNode("y");
  y->attr.api.compute_type = ComputeType::kComputeInvalid;
}

void LoadNanOutForStore_AfterGetApiInfo(ge::AscGraph &graph) {
  auto x1 = graph.FindNode("x1");
  x1->attr.api.type = ApiType::kAPITypeBuffer;
  x1->attr.api.unit = ComputeUnit::kUnitNone;

  auto load1 = graph.FindNode("load1");
  load1->attr.api.type = ApiType::kAPITypeCompute;
  load1->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto isnan = graph.FindNode("isnan");
  isnan->attr.api.type = ApiType::kAPITypeCompute;
  isnan->attr.api.unit = ComputeUnit::kUnitVector;

  auto store = graph.FindNode("store");
  store->attr.api.type = ApiType::kAPITypeCompute;
  store->attr.api.unit = ComputeUnit::kUnitMTE2;

  auto y = graph.FindNode("y");
  y->attr.api.type = ApiType::kAPITypeBuffer;
  y->attr.api.unit = ComputeUnit::kUnitNone;
}

void LoadNanOutForStore_AfterScheduler(ge::AscGraph &graph) {
  auto all_axis = graph.GetAllAxis();
  auto z0 = all_axis[0]->id;
  auto z1 = all_axis[1]->id;
  auto z2 = all_axis[2]->id;
  std::cout << "LoadNanOutForStore_AfterScheduler1" << std::endl;
  vector<AxisId> vectorized_axis{z1, z2};
  vector<ge::Expression> vectorized_strides{One, One};
  auto size = ge::GetSizeByDataType(ge::DT_FLOAT);
  vectorized_strides[0] = ge::sym::Align(graph.FindAxis(vectorized_axis[1])->size, 32 / size);

  auto x1 = graph.FindNode("x1");
  x1->outputs[0].attr.vectorized_axis = vectorized_axis;
  x1->outputs[0].attr.vectorized_strides = vectorized_strides;

  auto load1 = graph.FindNode("load1");
  load1->attr.sched.loop_axis = z0;
  load1->outputs[0].attr.vectorized_axis = vectorized_axis;
  load1->outputs[0].attr.vectorized_strides = vectorized_strides;

  size = ge::GetSizeByDataType(ge::DT_UINT8);
  vectorized_strides[0] = ge::sym::Align(graph.FindAxis(vectorized_axis[1])->size, 32 / size);
  auto isnan = graph.FindNode("isnan");
  isnan->attr.sched.loop_axis = z0;
  isnan->outputs[0].attr.vectorized_axis = vectorized_axis;
  isnan->outputs[0].attr.vectorized_strides = vectorized_strides;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.vectorized_axis = vectorized_axis;
  store->outputs[0].attr.vectorized_strides = vectorized_strides;
}

void LoadNanOutForStore_AfterQueBufAlloc(ge::AscGraph &graph) {
  auto x1 = graph.FindNode("x1");
  x1->outputs[0].attr.mem.tensor_id = 0;
  x1->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  x1->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  x1->outputs[0].attr.mem.position = Position::kPositionGM;
  x1->outputs[0].attr.buf.id = ge::kIdNone;
  x1->outputs[0].attr.que.id = ge::kIdNone;
  x1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  x1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.mem.tensor_id = 1;
  load1->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  load1->outputs[0].attr.mem.position = Position::kPositionVecIn;
  load1->outputs[0].attr.mem.reuse_id = 0;
  load1->outputs[0].attr.buf.id = ge::kIdNone;
  load1->outputs[0].attr.que.id = 0;
  load1->outputs[0].attr.que.depth = 2;
  load1->outputs[0].attr.que.buf_num = 2;
  load1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load1->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto isnan = graph.FindNode("isnan");
  isnan->outputs[0].attr.mem.tensor_id = 2;
  isnan->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeQueue;
  isnan->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareUB;
  isnan->outputs[0].attr.mem.position = Position::kPositionVecOut;
  isnan->outputs[0].attr.mem.reuse_id = 0;
  isnan->outputs[0].attr.buf.id = ge::kIdNone;
  isnan->outputs[0].attr.que.id = 1;
  isnan->outputs[0].attr.que.depth = 2;
  isnan->outputs[0].attr.que.buf_num = 2;
  isnan->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  isnan->outputs[0].attr.opt.merge_scope = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.mem.tensor_id = 3;
  store->outputs[0].attr.mem.alloc_type = AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = Position::kPositionGM;
  store->outputs[0].attr.buf.id = ge::kIdNone;
  store->outputs[0].attr.que.id = ge::kIdNone;
  store->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  store->outputs[0].attr.opt.merge_scope = ge::kIdNone;
}
