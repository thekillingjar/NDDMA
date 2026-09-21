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
#include <mockcpp/mockcpp.hpp>
#include <stdio.h>

#define private public
#define protected public
#include "hcom_all_reduce_fusion.h"
#include "plugin_manager.h"
#include "hcom_graph_optimizer.h"
#include "auto_tuning_plugin.h"
#include "auto_tuning_hcom_ops_kernel_info_store.h"
#include "auto_tuning_hcom_ops_kernel_builder.h"
#include "hcom_ops_kernel_info_store.h"
#include "ops_kernel_info_store_base.h"
#include "hcom_ops_stores.h"
#include "auto_tuning_hcom_all_reduce_fusion.h"
#include "tuning_utils.h"
#include "hcom_op_utils.h"
#undef private
#undef protected
#include "hccl_stub.h"
#include <stdlib.h>
#include <pthread.h>
#include <runtime/rt.h>
#include <iostream>
#include <fstream>
#include "llt_hccl_stub_ge.h"

using namespace std;
using namespace hccl;

static nlohmann::json allreduce_topo_switch_connect =
{
    {"topology type", "switch connection"},
    {
        "topology desc", {
            {
                {"node type", "TOR"},
                {"node name", "tor0"},
                {
                    "link info", {
                        {
                            {"link id", "0"},
                            {"local port name", "port0"},
                            {"local ip address", "100.100.83.1"},
                            {"opposite type", "SERVER"},
                            {"opposite name", "server0"},
                            {"opposite port name", "eth8"},
                            {"opposite ip address", "100.100.83.178"}
                        }
                    }
                }
            }
        }
    }
};

class AutoTuningHcclKernelBuilderTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        nlohmann::json rank_table =
        {
            {"status", "completed"},
            {"deploy_mode", "lab"},
            {"device_num", "4"},
            {"server_num", "2"},
            {"boardType", "0"},
            {"para_plane_location", "device"},
            {"para_plane_nic_num", "2"},
            {"para_plane_nic_name", {"eth0", "eth1"}},
            {"instance_count", "4"},
            {"device_count", "4"},
            {
                "instance_list",
                {
                    {   {"pod_name", ""}, {"rank_id", "0"}, {"server_id", "10.0.0.10"},
                        {
                            "devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}, {"ref_ip", "192.168.10.13"}}}
                        }
                    },
                    {   {"pod_name", ""}, {"rank_id", "1"}, {"server_id", "10.0.0.10"},
                        {
                            "devices", {{{"device_id", "0"}, {"device_ip", "192.168.1.12"}, {"ref_ip", "192.168.11.13"}}}
                        }
                    },
                    {   {"pod_name", ""}, {"rank_id", "2"}, {"server_id", "10.0.0.11"},
                        {
                            "devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.14"}, {"ref_ip", "192.168.10.15"}}}
                        }
                    },
                    {   {"pod_name", ""}, {"rank_id", "3"}, {"server_id", "10.0.0.11"},
                        {
                            "devices", {{{"device_id", "1"}, {"device_ip", "192.168.1.14"}, {"ref_ip", "192.168.11.15"}}}
                        }
                    }
                }
            },
            {
                "server_list",
                {
                    {
                        {"server_id", "192.168.10.2"},
                        {
                            "para_plane_info",
                            {{
                                    {"eth1", "192.168.210.2"},
                                    {"ref_ip", "192.168.210.1"}
                                },
                                {
                                    {"eth0", "192.168.200.2"},
                                    {"ref_ip", "192.168.200.1"}
                                }
                            }
                        }

                    },
                    {
                        {"server_id", "192.168.10.3"},
                        {
                            "para_plane_info",
                            {{
                                    {"eth0", "192.168.200.3"},
                                    {"ref_ip", "192.168.200.1"}
                                },
                                {
                                    {"eth1", "192.168.210.3"},
                                    {"ref_ip", "192.168.210.1"}
                                }
                            }
                        }

                    },

                }
            }
        };
        char file_name[] = "./st_AutoTuningHcclKernelBuilderTest.json";

        std::ofstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

        if (outfile.is_open())
        {
            HCCL_INFO("open %s success", file_name);
        }
        else
        {
            HCCL_INFO("open %s failed", file_name);
        }

        outfile << std::setw(4) << rank_table << std::endl;
        outfile.close();

        std::cout << "\033[36m--AutoTuningHcclKernelBuilderTest SetUP--\033[0m" << std::endl;
    }
    static void TearDownTestCase()
    {
        char file_name[] = "./st_AutoTuningHcclKernelBuilderTest.json";
        remove(file_name);
        std::cout << "\033[36m--AutoTuningHcclKernelBuilderTest TearDown--\033[0m" << std::endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        std::cout << "A Test TearDown" << std::endl;
    }
};

class NodeTest : public ge::Node {
public:
    NodeTest(){;};
    ~NodeTest(){;};
};


TEST_F(AutoTuningHcclKernelBuilderTest, st_CalcOpRunningParam)
{
    AutoTuningHcomOpsKernelBuilder autoTuningHcomKernelBuilder;
    HcclResult ret;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ge::NodePtr nodeptr(new NodeTest);
    ge ::Status ge_ret = ge::INTERNAL_ERROR;

    std::string type = HCCL_KERNEL_OP_TYPE_SEND;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = autoTuningHcomKernelBuilder.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    type = HCCL_KERNEL_OP_TYPE_BROADCAST;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = autoTuningHcomKernelBuilder.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = autoTuningHcomKernelBuilder.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    type = HCCL_KERNEL_OP_TYPE_REDUCESCATTER;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = autoTuningHcomKernelBuilder.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    type = HCCL_KERNEL_OP_TYPE_REDUCE;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = autoTuningHcomKernelBuilder.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    type = HCCL_KERNEL_OP_TYPE_RECEIVE;
    nodeptr->GetOpDesc()->SetType(type);
    ge_ret = autoTuningHcomKernelBuilder.CalcOpRunningParam(*nodeptr);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
}

TEST_F(AutoTuningHcclKernelBuilderTest, st_generateTask)
{
    ge::NodePtr nodeptr(new NodeTest);
    ge::Buffer tempBuffer;
    ge::RunContext runContext_dummy;
    AutoTuningHcomOpsKernelBuilder autoTuningHcomKernelBuilder;

    std::vector<domi::TaskDef> taskDefList;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s64 streamId = 10000;
    nodeptr->GetOpDesc()->SetStreamId((s64)streamId);

    std::string type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    nodeptr->GetOpDesc()->SetType(type);

    ge ::Status ge_ret = ge::INTERNAL_ERROR;
    ge_ret = autoTuningHcomKernelBuilder.GenerateTask(*nodeptr,runContext_dummy,taskDefList);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
}