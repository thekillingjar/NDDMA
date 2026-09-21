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

#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/memory/memory_assigner.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/ge_local_context.h"
#include "common/share_graph.h"
#include "graph/utils/graph_utils.h"
#include "api/gelib/gelib.h"
#include "graph/build/graph_builder.h"
#include "register/ops_kernel_builder_registry.h"
#include "graph/build/memory/block_mem_assigner.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "stub/gert_runtime_stub.h"
#include "runtime_stub.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/manager/active_memory_allocator.h"

using namespace testing;
using namespace ge;

class UtestMemoryAssignerManualTest : public testing::Test {
 protected:
  class FakeOpsKernelBuilder : public OpsKernelBuilder {
   public:
    FakeOpsKernelBuilder() = default;

   private:
    Status Initialize(const map<std::string, std::string> &options) override {
      return SUCCESS;
    };
    Status Finalize() override {
      return SUCCESS;
    };
    Status CalcOpRunningParam(Node &node) override {
      return SUCCESS;
    };
    Status GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) override {
      return SUCCESS;
    };
  };

  void SetUp() {
    InitGeLib();
  }

  void TearDown() {
    FinalizeGeLib();
  }

  void GetSliceInfoFromJson(const OpDescPtr &op_desc) {
    std::string str_slice_info;
    (void)AttrUtils::GetStr(op_desc, ffts::kAttrSgtJsonInfo, str_slice_info);
    if (str_slice_info.empty()) {
      return;
    }
    try {
      nlohmann::json slice_info_json = nlohmann::json::parse(str_slice_info);
      if (slice_info_json.is_null()) {
        GELOGW("Get slice info: %s is empty.", str_slice_info.c_str());
      } else {
        ffts::ThreadSliceMapPtr slice_info_ptr = std::make_shared<ffts::ThreadSliceMap>();
        if (slice_info_ptr == nullptr) {
          return;
        }
        slice_info_ptr->slice_instance_num = slice_info_json["slice_instance_num"];
        slice_info_ptr->thread_mode = slice_info_json["threadMode"];
        slice_info_ptr->parallel_window_size = slice_info_json["parallel_window_size"];
        op_desc->SetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
      }
    } catch (const nlohmann::json::exception &e) {
      GELOGW("Parse json str failed, %s", e.what());
      return;
    }
  }

  void FakeProcess(ComputeGraphPtr &compute_graph) {
    for (const NodePtr &node : compute_graph->GetAllNodes()) {
      auto op_desc = node->GetOpDesc();
      if (op_desc == nullptr) {
        continue;
      }
      GetSliceInfoFromJson(op_desc);
      // fake weights
      if ((op_desc->GetType() == CONSTANT) || (op_desc->GetType() == CONSTANTOP)){
        int32_t value = 0;
        GeTensorDesc data_desc(GeShape(), FORMAT_NCHW, DT_INT32);
        GeTensorPtr const_value = MakeShared<GeTensor>(data_desc, reinterpret_cast<uint8_t *>(&value), sizeof(int32_t));
        AttrUtils::SetTensor(op_desc, ATTR_NAME_WEIGHTS, const_value);
      }
      // 适配ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX为空问题
      const std::string ATTR_NAME_ENGINE_TYPE = "_engine_type";
      const std::string ATTR_NAME_OP_KERNEL_LIB_NAME = "_ge_attr_op_kernel_lib_name";
      std::string lxfusion_op_kernel_lib_name;
      (void)AttrUtils::GetStr(op_desc, ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX, lxfusion_op_kernel_lib_name);
      if (lxfusion_op_kernel_lib_name.empty()) {
        (void)AttrUtils::GetStr(op_desc, ATTR_NAME_OP_KERNEL_LIB_NAME, lxfusion_op_kernel_lib_name);
        (void)AttrUtils::SetStr(op_desc, ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX, lxfusion_op_kernel_lib_name);
      }
      std::string lxfusion_engine_name;
      (void)AttrUtils::GetStr(op_desc, ATTR_NAME_ENGINE_NAME_FOR_LX, lxfusion_engine_name);
      if (lxfusion_engine_name.empty()) {
        (void)AttrUtils::GetStr(op_desc, ATTR_NAME_ENGINE_TYPE, lxfusion_engine_name);
        if (lxfusion_engine_name.empty()) {
          auto pos = lxfusion_op_kernel_lib_name.find("_OP_STORE");
          if (pos != std::string::npos) {
            lxfusion_engine_name = lxfusion_op_kernel_lib_name.substr(0U, pos);
          } else {
            lxfusion_engine_name = lxfusion_op_kernel_lib_name;
          }
          GELOGI("Set %s engine_name to %s", op_desc->GetName().c_str(), lxfusion_engine_name.c_str());
        }
        (void)AttrUtils::SetStr(op_desc, ATTR_NAME_ENGINE_NAME_FOR_LX, lxfusion_engine_name);
      }

      // fake ops kernel builder
      OpsKernelBuilderPtr builder = MakeShared<FakeOpsKernelBuilder>();
      OpsKernelBuilderRegistry::GetInstance().Register(op_desc->GetOpKernelLibName(), builder);
    }
  }

  timespec interval(const timespec &start, const timespec &end) {
    timespec temp_time;
    if ((end.tv_nsec - start.tv_nsec) < 0) {
      temp_time.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    } else {
      temp_time.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp_time;
  }
 private:
  void InitGeLib() {
    class MockRuntime : public RuntimeStub {
      rtError_t rtMemGetInfoEx(rtMemInfoType_t mem_info_type, size_t *free, size_t *total) override {
        *free = 60UL * 1024UL * 1024UL * 1024UL;
        *total = 60UL * 1024UL * 1024UL * 1024UL;
        return RT_ERROR_NONE;
      }
    };
    auto mock_runtime = std::make_shared<MockRuntime>();
    RuntimeStub::SetInstance(mock_runtime);
    map<string, string> options;
    Status ret = ge::GELib::Initialize(options);
    EXPECT_EQ(ret, SUCCESS);
    auto instance_ptr = ge::GELib::GetInstance();
    EXPECT_NE(instance_ptr, nullptr);
  }

  void FinalizeGeLib() {
    auto instance_ptr = ge::GELib::GetInstance();
    if (instance_ptr != nullptr) {
      instance_ptr->Finalize();
    }
  }
};

TEST_F(UtestMemoryAssignerManualTest, memory_assign_with_dump_graph) {
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("test_graph");
  std::string dump_file_name = "./ge_proto_AfterAssignedLogicStreams.txt";
  auto dump_file = getenv("GE_MT_DUMP_FILE");
  if (dump_file != nullptr) {
    dump_file_name = dump_file;
  }
  if (dump_file_name.find(".txt") == std::string::npos) {
    return;
  }
  bool load_success = GraphUtils::LoadGEGraph(dump_file_name.c_str(), compute_graph);
  if (!load_success) {
    return;
  }
  FakeProcess(compute_graph);

  const auto var_manager = VarManager::Instance(compute_graph->GetSessionID());
  ASSERT_NE(var_manager, nullptr);
  ASSERT_EQ(var_manager->Init(0U, compute_graph->GetSessionID(), 0U, 0U), SUCCESS);

  std::string in_reuse_index;
  for (auto i = 0U; i < compute_graph->GetInputNodes().size(); ++i) {
    in_reuse_index.append(std::to_string(i));
    in_reuse_index.append(",");
  }

  std::string out_reuse_index;
  const auto netoutput_node = compute_graph->FindFirstNodeMatchType(NETOUTPUT);
  if ((netoutput_node != nullptr) && (netoutput_node->GetOpDesc() != nullptr)) {
    for (auto i = 0U; i < netoutput_node->GetOpDesc()->GetAllInputsSize(); ++i) {
      out_reuse_index.append(std::to_string(i));
      out_reuse_index.append(",");
    }
  }

  // add new option
  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  const auto io_reuse = getenv("GE_MT_IO_REUSE");
  if (io_reuse != nullptr) {
    const std::string str_io_reuse = io_reuse;
    if (str_io_reuse.find("1:") != std::string::npos) {
      graph_options[OPTION_INPUT_REUSE_MEM_INDEXES] = in_reuse_index;
    }
    if (str_io_reuse.find(":1") != std::string::npos) {
      graph_options[OPTION_OUTPUT_REUSE_MEM_INDEXES] = out_reuse_index;
    }
  }

  const auto io_alloc_mode = getenv("GE_MT_IO_MEM_ALLOC_MODE");
  if (io_alloc_mode != nullptr) {
    const std::string str_io_alloc_mode = io_alloc_mode;
    if (!str_io_alloc_mode.empty()) {
      graph_options[OPTION_GRAPH_IO_MEM_ALLOC_MODE] = str_io_alloc_mode;
    }
  }

  const auto mop_mode = getenv("GE_MT_MEMORY_OPTIMIZATION_POLICY");
  if (mop_mode != nullptr) {
    const std::string str_mop_mode = mop_mode;
    if (!str_mop_mode.empty()) {
      graph_options[MEMORY_OPTIMIZATION_POLICY] = str_mop_mode;
    }
  }

  const auto single_stream = getenv("GE_MT_ENABLE_SINGLE_STREAM");
  if (single_stream != nullptr) {
    const std::string str_single_stream = single_stream;
    if (str_single_stream == "1") {
      graph_options[ENABLE_SINGLE_STREAM] = "true";
    }
  }

  GEEVENT("begin AssignMemory");
  // system("callgrind_control -z");
  // system("callgrind_control -i=on");

  const auto build_for_evaluate = getenv("GE_MT_BUILD_FOR_EVALUATE");
  if (build_for_evaluate != nullptr) {
    std::string str_topo_mode = "0";
    const auto topo_mode = getenv("GE_MT_TOPOSORTING_MODE");
    if (topo_mode != nullptr) {
      str_topo_mode = topo_mode;
    }
    graph_options[OPTION_TOPOSORTING_MODE] = str_topo_mode;
    GetThreadLocalContext().SetGraphOption(graph_options);
    // graph builder test
    ModelDataInfo model;
    GraphBuilder graph_builder;
    EXPECT_EQ(graph_builder.BuildForEvaluate(compute_graph, model), GRAPH_SUCCESS);
  } else {
    GetThreadLocalContext().SetGraphOption(graph_options);
    // mem reuse test
    MemoryAssigner memory_assigner(compute_graph);
    std::map<uint64_t, size_t> mem_offset;
    size_t zero_memory_size = 0;
    EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  }
  // system("callgrind_control --dump");
  GEEVENT("end AssignMemory");
}

TEST_F(UtestMemoryAssignerManualTest, memory_assign_with_report) {
  const size_t kMaxLogLine = 1024U;
  std::string report_file_name = "./imas_mem_pool.csv";
  auto set_report_file = getenv("GE_MT_DUMP_FILE");
  if (set_report_file != nullptr) {
    report_file_name = set_report_file;
  }
  if (report_file_name.find(".csv") == std::string::npos) {
    return;
  }
  char line[kMaxLogLine] = {0};
  std::map<int64_t, std::vector<ge::MemBlock *>> memorys;
  std::map<std::string, gert::memory::CachingMemAllocator *> allocators;
  size_t total_time = 0U;
  size_t total_count = 0U;
  size_t set_step_count = 1U;
  auto set_step = getenv("SET_STEP_COUNT");
  if (set_step != nullptr) {
    set_step_count = atol(set_step);
  }

  std::string ge_id = "ge_4";
  auto set_ge_id = getenv("GE_MT_ID");
  if (set_ge_id != nullptr) {
    ge_id = set_ge_id;
  }

  auto release_func = [&memorys, &total_time, this] (const int64_t begin) {
    auto it = memorys.find(begin);
    if (it != memorys.end()) {
      for (auto p : it->second) {
        if (p == nullptr) {
          continue;
        }
        struct timespec t_begin;
        struct timespec t_end;
        clock_gettime(CLOCK_REALTIME, &t_begin);
        p->Free();
        clock_gettime(CLOCK_REALTIME, &t_end);
        total_time += interval(t_begin, t_end).tv_nsec;
      }
      memorys.erase(it);
    }
  };

  for (size_t step = 0U; step < set_step_count; ++step) {
    // system("callgrind_control -z");
    // system("callgrind_control -i=on");
    std::cout<<"step:"<<step<<std::endl;
    FILE *fp = fopen(report_file_name.c_str(), "r");
    if (fp == NULL) {
      return;
    }
    bool release_before_alloc = false;
    while(fgets(line, sizeof(line), fp) != NULL) {
      std::string log_line = line;
      if (log_line.find("life time begin") == std::string::npos) {
        continue;
      }
      std::vector<std::string> fields = StringUtils::Split(log_line, ',');
      if (fields.size() < 14) {
        continue;
      }
      int64_t begin = atol(fields[12].c_str());
      if (begin == 0) {
        release_before_alloc = true;
      }
      int64_t end = atol(fields[14].c_str());
      if (release_before_alloc) {
        release_func(begin);
      }
      const auto allocator_id = fields[2];
      if (allocator_id.find(ge_id) == std::string::npos) {
        continue;
      }
      auto it_allocator = allocators.find(allocator_id);
      gert::memory::CachingMemAllocator *allocator = nullptr;
      if (it_allocator == allocators.end()) {
        allocator = gert::memory::CachingMemAllocator::GetAllocator(0).release();
        allocators[allocator_id] = allocator;
      } else {
        allocator = it_allocator->second;
      }
      int64_t size = atol(fields[8].c_str());
      EXPECT_NE(allocator, nullptr);
      struct timespec t_begin;
      struct timespec t_end;
      clock_gettime(CLOCK_REALTIME, &t_begin);
      auto p = allocator->Malloc(size);
      clock_gettime(CLOCK_REALTIME, &t_end);
      total_time += interval(t_begin, t_end).tv_nsec;
      total_count++;
      memorys[end].emplace_back(p);
      if (!release_before_alloc) {
        release_func(begin);
      }
    }
    fclose(fp);
    // system("callgrind_control --dump");
    for (auto &it : memorys) {
      for (auto p : it.second) {
        if (p == nullptr) {
         continue;
        }
        p->Free();
      }
    }
    memorys.clear();
    std::string ex_info;
    for (auto allocator : allocators) {
      if (allocator.second->GetScalableAllocator() != nullptr) {
        ex_info.append(allocator.second->GetScalableAllocator()->GetStatics());
      }
    }
    if (total_count != 0U) {
      std::cout<<"step:"<<step<<" total_time:"<<total_time<<"ns total_count:"<<total_count<<" average:"
          <<total_time/total_count<<"ns "<<ex_info<<" page size:"
          <<ge::ActiveMemoryUtil::SizeToString(kLargePageSize)<<std::endl;
      total_count = 0U;
      total_time = 0U;
    }
  }
  for (auto allocator : allocators) {
    allocator.second->Finalize();
  }
  RuntimeStub::Reset();
}
