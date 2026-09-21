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

#include <cstdio>
#include <cstdlib>
#include<vector>
#include<algorithm>
#include <cmath>
#include <climits>
#include <sstream>
#include <istream>
#include"tune_log.h"
#include "hccl_stub.h"
#include <pthread.h>
#include <runtime/rt.h>
#include <iostream>
#include <fstream>
#include "llt_hccl_stub_ge.h"
#include"hcom_gradient_split_tune.h"
#include "gradient_split_tune.h"
#include "layers.h"
#include "communication.h"
#include "topology.h"
#define private public
#include "cluster.h"
#undef private
#include "model.h"
#include "evaluator.h"


using namespace std;

extern TuneResult_t GetGradientInfo(const std::vector<struct GradientNode> &graNode,
    const std::vector<struct BPTimeInfo> &bpTimeInfo, std::vector<struct GradientInfo>& graInfo);

extern TuneResult_t GetBPTimeFromProfiling(const std::vector<std::vector<struct ProfilingMetric>> &taskProfilingList,
    std::vector<struct BPTimeInfo> &bpTimeInfo);

/*extern TuneResult_t GetGradientInfo(const std::vector<struct GradientNode> graNode,
    const std::vector<struct BPTimeInfo> bpTimeInfo, std::vector<struct GradientInfo>& graInfo);*/
extern TuneResult_t GetGradientNode(const std::vector<std::vector<struct ProfilingMetric>> &taskProfilingList,
    std::string workPath, std::vector<struct GradientInfo> &graInfo);

extern TuneResult_t FileStreamRead(const std::string filePath, std::vector<struct GradientNode>& graNode);

extern void CheckTimeSd(const std::vector<std::vector<uint64_t>>& time, std::vector<struct BPTimeInfo>& bpTimeInfo);

class GradientSplitTuneTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GradientSplitTuneTest SetUP" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "GradientSplitTuneTest TearDown" << std::endl;
    }
    virtual void SetUp()
    {
        static uint32_t callCnt = 0;
        string name = std::to_string(callCnt++) + "_" + __PRETTY_FUNCTION__;
        //ra_set_shm_name(name.c_str());
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        std::cout << "A Test TearDown" << std::endl;
    }
};

struct GradientDataInfo {
    unsigned long long dataSize;
    std::string dataType;
    unsigned long long graphId;
    std::string groupName;
    std::string gradientNodeName;
    std::string traceNodeName;
    std::string allReduceNodeName;
};

static uint32_t g_gradientNodeNo;
std::string globalGraphId_;

TEST_F(GradientSplitTuneTest, stGradientSplitTune1)
{
    TuneResult_t ret;
    vector<vector<struct ProfilingMetric>> taskProfilingList(4, vector<struct ProfilingMetric>(2, {"tune", "trace_node_name", "TensorRedirect", 0, 1}));
    std::vector<struct BPTimeInfo> bpTimeInfo;
    std::string workPath = "/tmp/";
    std::vector<GradientDataInfo> recordInfos (4, {2, "gradient_size(byte)", 12, "group_name", "graph_id", "trace_node_name", "data_type"});

    taskProfilingList[0][1].taskStartTime = 2;
    taskProfilingList[0][1].taskEndTime = 3;
    taskProfilingList[1][1].taskStartTime = 4;
    taskProfilingList[1][1].taskEndTime = 5;
    taskProfilingList[2][1].taskStartTime = 4;
    taskProfilingList[2][1].taskEndTime = 5;
    taskProfilingList[3][1].taskStartTime = 4;
    taskProfilingList[3][1].taskEndTime = 5;

    std::string gradientInfosFile = std::string(workPath) + "gradient_summary.csv";

    std::ofstream fileStream(gradientInfosFile.c_str(), std::ios::out | std::ios::app | std::ios::binary);
    if (fileStream.is_open()) {
        for (uint32_t i = 0; i < recordInfos.size(); i++) {
            g_gradientNodeNo++;
            std::cout<<"g_gradientNodeNo:"<<g_gradientNodeNo<<endl;
            fileStream << g_gradientNodeNo << "," << recordInfos[i].dataSize << "," << recordInfos[i].dataType <<
                "," << recordInfos[i].graphId << "," << recordInfos[i].groupName << "," <<
                recordInfos[i].gradientNodeName << "," << recordInfos[i].traceNodeName << "," <<
                recordInfos[i].allReduceNodeName << std::endl;
        }
        fileStream.close();
    } else {
        std::cout<<"fileStream failed for write"<<endl;
    }

    // tune path is not a real path, error will occur in fuc "GetTuneFusionPath"
    ret = HcomGradientFusionTune(taskProfilingList, workPath);
    EXPECT_EQ(ret, TUNE_E_PARA);

    MOCKER(GetBPTimeFromProfiling)
    .expects(atMost(20))
    .will(returnValue(1));
    ret = HcomGradientFusionTune(taskProfilingList, workPath);
    GlobalMockObject::verify();

    MOCKER(FileStreamRead)
    .expects(atMost(20))
    .will(returnValue(1));
    ret = HcomGradientFusionTune(taskProfilingList, workPath);
    GlobalMockObject::verify();

    std::string badPath = "zhl/";
    ret = HcomGradientFusionTune(taskProfilingList, badPath);

    vector<std::vector<uint64_t>> time;
    std::vector<uint64_t> tmp1(1, 0);
    time.push_back(tmp1);
    CheckTimeSd(time, bpTimeInfo);

    std::vector<uint64_t> tmp2(3, 0);
    time.push_back(tmp2);
    struct BPTimeInfo BPTimeTmp = { "opName", 0 };
    bpTimeInfo.push_back(BPTimeTmp);
    CheckTimeSd(time, bpTimeInfo);

    MOCKER(CheckTimeSd)
    .expects(atMost(20))
    .will(returnValue(1));
    ret = GetBPTimeFromProfiling(taskProfilingList, bpTimeInfo);
    GlobalMockObject::verify();

    vector<vector<struct ProfilingMetric>> taskProfilingList2;
    vector<struct ProfilingMetric> taskProfilingTmp1(2, {"tune", "trace_node_name", "TensorRedirect", 0, 1});
    taskProfilingList2.push_back(taskProfilingTmp1);
    vector<struct ProfilingMetric> taskProfilingTmp2(3, {"tune", "trace_node_name", "TensorRedirect", 0, 1});
    taskProfilingList2.push_back(taskProfilingTmp2);
    ret = GetBPTimeFromProfiling(taskProfilingList2, bpTimeInfo);
    GlobalMockObject::verify();

    int retVal = remove(gradientInfosFile.c_str());
    std::cout<<"remove "<<gradientInfosFile<<", retVal:"<<retVal<<endl;

}

TEST_F(GradientSplitTuneTest, stGradientSplitTune2)
{

    float ret;
    vector<int> slices;
    vector<int> slices1;
    slices.push_back(1);
    Communication com(1);
    com.CheckParams(1, "direct");
    com.CheckParams(0, "direct");
    ret = com.CalculateXferFrequency(8, "ring", slices);
    ret = com.CalculateXferPercentage(8, "ring", slices);
    ret = com.CalculateXferBubbles(8, "ring", slices);
    ret = com.CalculateComputePercentage(8, "ring", slices);

    Scatter sca(1);
    ret = sca.CalculateXferFrequency(8, "ring", slices);
    ret = sca.CalculateXferPercentage(8, "ring", slices);
    ret = sca.CalculateXferFrequency(8, "sa", slices);
    ret = sca.CalculateXferPercentage(8, "sa", slices);

    Gather gat(1);
    ret = gat.CalculateXferFrequency(8, "ring", slices);
    ret = gat.CalculateXferPercentage(8, "ring", slices);
    ret = gat.CalculateXferFrequency(8, "sa", slices);
    ret = gat.CalculateXferPercentage(8, "sa", slices);

    Allgather allg(1);
    ret = allg.CalculateXferFrequency(8, "ring", slices);
    ret = allg.CalculateXferFrequency(8, "H-D", slices);
    ret = allg.CalculateXferFrequency(8, "mesh-serial", slices);
    ret = allg.CalculateXferFrequency(8, "mesh", slices);
    ret = allg.CalculateXferFrequency(8, "sa", slices);
    ret = allg.CalculateXferPercentage(8, "ring", slices);
    ret = allg.CalculateXferPercentage(8, "H-D", slices);
    ret = allg.CalculateXferPercentage(8, "mesh", slices);
    ret = allg.CalculateXferPercentage(8, "direct", slices);
    ret = allg.CalculateXferPercentage(8, "mesh-serial", slices);
    ret = allg.CalculateXferPercentage(8, "sa", slices);

    Allreduce allr(1);
    ret = allr.CalculateXferFrequency(8, "ring", slices);
    ret = allr.CalculateXferFrequency(8, "tree", slices);
    ret = allr.CalculateXferFrequency(8, "2D-torus", slices);
    ret = allr.CalculateXferPercentage(8, "ring", slices);
    ret = allr.CalculateXferPercentage(8, "H-D", slices);
    ret = allr.CalculateXferPercentage(8, "tree", slices);
    ret = allr.CalculateXferPercentage(8, "2D-torus", slices);
    ret = allr.CalculateXferBubbles(8, "ring", slices);
    ret = allr.CalculateXferBubbles(8, "tree", slices);
    ret = allr.CalculateComputePercentage(8, "ring", slices);
    ret = allr.CalculateComputePercentage(8, "H-D", slices);
    ret = allr.CalculateComputePercentage(8, "tree", slices);

    Broadcast broa(1);
    ret = broa.CalculateXferFrequency(1, "ring", slices);
    ret = broa.CalculateXferFrequency(8, "ring", slices);
    ret = broa.CalculateXferFrequency(8, "H-D", slices);
    ret = broa.CalculateXferFrequency(8, "mesh-serial", slices);
    ret = broa.CalculateXferFrequency(8, "mesh", slices);

    ret = broa.CalculateXferPercentage(8, "ring", slices);
    ret = broa.CalculateXferPercentage(8, "mesh", slices);
    ret = broa.CalculateXferPercentage(8, "tree", slices);
    ret = broa.CalculateXferBubbles(1, "ring", slices);
    ret = broa.CalculateXferBubbles(8, "ring", slices);
    ret = broa.CalculateXferBubbles(8, "H-D", slices);
    ret = broa.CalculateXferBubbles(8, "mesh", slices);
    ret = broa.CalculateXferBubbles(8, "tree", slices);
    Reduce redu(1);
    ret = redu.CalculateXferFrequency(8, "ring", slices);
    ret = redu.CalculateXferFrequency(1, "ring", slices);
    ret = redu.CalculateXferFrequency(8, "H-D", slices);
    ret = redu.CalculateXferFrequency(8, "mesh", slices);
    ret = redu.CalculateXferFrequency(8, "mesh-serial", slices);
    ret = redu.CalculateXferFrequency(8, "direct", slices);
    ret = redu.CalculateXferFrequency(2, "tree", slices);
    ret = redu.CalculateXferPercentage(8, "ring", slices);
    ret = redu.CalculateXferPercentage(8, "H-D", slices);
    ret = redu.CalculateXferPercentage(8, "mesh", slices);
    ret = redu.CalculateXferPercentage(8, "mesh-serial", slices);
    ret = redu.CalculateXferPercentage(8, "direct", slices);
    ret = redu.CalculateXferPercentage(2, "tree", slices);
    ret = redu.CalculateXferBubbles(8, "ring", slices);
    ret = redu.CalculateXferBubbles(1, "ring", slices);
    ret = redu.CalculateXferBubbles(8, "H-D", slices);
    ret = redu.CalculateXferBubbles(8, "mesh", slices);
    ret = redu.CalculateXferBubbles(8, "tree", slices);
    ret = redu.CalculateComputePercentage(8, "ring", slices);
    ret = redu.CalculateComputePercentage(1, "ring", slices);
    ret = redu.CalculateComputePercentage(8, "H-D", slices);
    ret = redu.CalculateComputePercentage(8, "mesh", slices);
    ret = redu.CalculateComputePercentage(2, "tree", slices);
    Reducescatter rs(1);
    ret = rs.CalculateXferFrequency(8, "mesh", slices);
    ret = rs.CalculateXferFrequency(8, "H-D", slices);
    ret = rs.CalculateXferFrequency(8, "direct", slices);
    ret = rs.CalculateXferPercentage(8, "H-D", slices);
    ret = rs.CalculateXferPercentage(8, "mesh", slices);
    ret = rs.CalculateComputePercentage(8, "H-D", slices);
    ret = rs.CalculateComputePercentage(8, "mesh", slices);
    Sendrecv sd(1);
    MultirootReduce Mr(1);
    ret = Mr.CalculateXferPercentage(8, "H-D", slices);
    ret = Mr.CalculateXferPercentage(8, "H-D", slices1);
    ret = Mr.CalculateComputePercentage(8, "H-D", slices);
    ret = Mr.CalculateComputePercentage(8, "H-D", slices1);
    MultirootBroadcast Mb(1);
    ret = Mb.CalculateXferPercentage(8, "H-D", slices);
    ret = Mb.CalculateXferPercentage(8, "H-D", slices1);
    AlltoAll aa(1);
    ret = aa.CalculateXferFrequency(8, "H-D", slices);
    ret = aa.CalculateXferFrequency(8, "mesh", slices);
    ret = aa.CalculateXferPercentage(8, "H-D", slices);
    ret = aa.CalculateXferPercentage(8, "mesh", slices);
    ret = aa.CalculateComputePercentage(8, "ring", slices);
    ret = aa.CalculateXferPercentage(0, "H-D", slices);
    ret = aa.CalculateComputePercentage(0, "ring", slices);
}

extern string GetModelPara(string workPath, string str);

TEST_F(GradientSplitTuneTest, stGradientSplitTune3)
{
    std::string workPath = "/tmp/";
    std::string gradientInfos = std::string(workPath) + "grad.txt";
    std::ofstream  fileStream(gradientInfos.c_str(), std::ios::out | std::ios::app | std::ios::binary);
    if (fileStream.is_open()) {
        fileStream<< "kwjqdi3hwfioo" << ":" << " klqwhjfdiqwhjfio" << std::endl;
    }
    fileStream.close();
    std::string  str("kwjqdi3hwfioo");
    std::string  ret = GetModelPara(gradientInfos, str);
    remove(gradientInfos.c_str());
    
}

extern vector<Layer> GetBpsInfo(vector<float> layer_cost, vector<float> layer_size_param);
TEST_F(GradientSplitTuneTest, stGradientSplitTune8)
{
    vector<float> layer_cost;
    layer_cost.push_back(10.0);
    layer_cost.push_back(10.0);
    layer_cost.push_back(10.0);
    vector<float> layer_size_param;
    layer_size_param.push_back(20.0);
    layer_size_param.push_back(20.0);
    layer_size_param.push_back(20.0);
    layer_size_param.push_back(20.0);
    GetBpsInfo(layer_cost, layer_size_param);
}

TEST_F(GradientSplitTuneTest, stGradientSplitTune_topology)
{
    float ret;
    vector<int> slices;
    slices.push_back(1);   
    Communication op;

    Topology topo;
    topo.bwComputation_ = 100;
    topo.gpuNum_ = 8;
    topo.algorithm_ = "ring";

    ret = topo.CalculateCost(op, 262144, slices, 1);

    TopoInfo topo_info;
    Mesh mesh(topo_info);
    ret = mesh.CalculateComputeCost(op, 262144, slices);
}

void GetTopoInfoFromFile(vector<TopoInfo>& toposInfo);
TEST_F(GradientSplitTuneTest, stGradientSplitTune_topology_MG_WFBP)
{
    float ret;
    string workPath = "/tmp";

    vector<TopoInfo> topoInfo;
    MOCKER(GetTopoInfoFromFile)
    .stubs()
    .with(outBound(topoInfo));
    Cluster cluster(workPath);

    Topology topo0;
    topo0.bwComputation_ = 100;
    topo0.gpuNum_ = 8;
    topo0.algorithm_ = "ring";
    Topology topo1;
    topo1.bwComputation_ = 100;
    topo1.gpuNum_ = 8;
    topo1.algorithm_ = "ring";
    std::vector<Topology> topoList;
    topoList.push_back(topo0);
    topoList.push_back(topo1);
    cluster.mTopoList_ = topoList;

    vector<int> topoNumList{1, 2};
    cluster.mTopoNumList_ = topoNumList;

    Communication op;
    op.mName_ = "Allreduce";
    ret = cluster.CalculateStartUpCost(op);
    GlobalMockObject::verify();

}

TEST_F(GradientSplitTuneTest, stGetGradientInfo)
{
    std::vector<struct GradientNode> graNodeVct;
    std::vector<struct BPTimeInfo> bpTimeInfo;
    std::vector<struct GradientInfo> graInfo;

    GradientNode graNode;
    graNode.traceNodeName = "NodeTest";
    graNode.groupName = "group";
    graNode.dataType = "fp32";
    graNode.gradientSize = 123;
    graNode.graphId = 1;
    graNodeVct.push_back(graNode);

    BPTimeInfo timeInfo;
    timeInfo.opName = "NodeTest";
    timeInfo.time = 321;
    bpTimeInfo.push_back(timeInfo);

    TuneResult_t ret;
    ret = GetGradientInfo(graNodeVct, bpTimeInfo, graInfo);
    EXPECT_EQ(ret, TUNE_SUCCESS); 
}

extern TuneResult_t GetSocVersion(std::string &configVersion);

TEST_F(GradientSplitTuneTest, stGetSocVersion)
{
    TuneResult_t ret;
    std::string socVersion;
    ret = GetSocVersion(socVersion);
    EXPECT_EQ(ret, TUNE_SUCCESS);
}

TEST_F(GradientSplitTuneTest, stMutipleGraphTune)
{
    TuneResult_t ret;
    std::string workPath = "/tmp/";
    globalGraphId_ = "graph_id";
    g_gradientNodeNo = 0;
    std::vector<struct GradientNode> graNode;
    std::vector<GradientDataInfo> recordInfos (3, {2, "gradient_size(byte)", 12, "group_name", "graph_id", "trace_node_name", "data_type"});
    vector<vector<struct ProfilingMetric>> taskProfilingList(4, vector<struct ProfilingMetric>(2, {"graph_id", "trace_node_name", "TensorRedirect", 0, 1}));
    std::vector<struct BPTimeInfo> bpTimeInfo;

    std::string gradientInfosFile = std::string(workPath) + "gradient_summary.csv";

    std::ofstream fileStream(gradientInfosFile.c_str(), std::ios::out | std::ios::app | std::ios::binary);
    if (fileStream.is_open()) {
        for (uint32_t i = 0; i < recordInfos.size(); i++) {
            g_gradientNodeNo++;
            std::cout<<"g_gradientNodeNo:"<<g_gradientNodeNo<<endl;
            fileStream << g_gradientNodeNo << "," << recordInfos[i].dataSize << "," << recordInfos[i].dataType <<
                "," << recordInfos[i].graphId << "," << recordInfos[i].groupName << "," <<
                recordInfos[i].gradientNodeName << "," << recordInfos[i].traceNodeName << "," <<
                recordInfos[i].allReduceNodeName << std::endl;
        }
        fileStream.close();
    } else {
        std::cout<<"fileStream failed for write"<<endl;
    }

    ret = FileStreamRead(workPath, graNode);
    EXPECT_EQ(ret, TUNE_SUCCESS);
    remove(gradientInfosFile.c_str());
}

extern TuneResult_t GetTuneFusionPath(std::string &fusionPath);
TEST_F(GradientSplitTuneTest, stFindPathFromHomeExport)
{
    TuneResult_t ret;
    std::string fusionSocVersion = "Ascend910A";
    std::string fusionPath;
    char* getDefPath = getenv("HOME");
    MOCKER(GetSocVersion)
    .stubs()
    .with(outBound(fusionSocVersion))
    .will(returnValue(TUNE_SUCCESS));
    std::string realPath = getDefPath;
    realPath = realPath + "/Ascend/latest/data/aoe/custom/graph/Ascend910A/Ascend910A_gradient_fusion.json";

    uint32_t beginCmpPath = 0;
    uint32_t endCmpPath = 0;

    std::string fullPath = "";
    if ('/' != realPath[0]) {
        fullPath = getcwd(nullptr, 0);
        beginCmpPath = fullPath.size();
        fullPath = fullPath + "/" + realPath;
    } else {
        fullPath = realPath;
        beginCmpPath = 1;
    }

    if (fullPath[fullPath.size() - 1] != '/') {
        fullPath += "/";
    }
    endCmpPath = fullPath.size();
    for (uint32_t i = beginCmpPath; i < endCmpPath; i++) {
        if ('/' == fullPath[i]) {
            std::string curPath = fullPath.substr(0, i);
            if (access(curPath.c_str(), F_OK) != 0) {
                mkdir(curPath.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
            }
        }
    }
    ret = GetTuneFusionPath(fusionPath);
    //路径字符拼接成功，realpath校验失败
    EXPECT_EQ(ret, TUNE_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(GradientSplitTuneTest, st_ErrorJsonFile)
{
    TuneResult_t ret;
    vector<vector<struct ProfilingMetric>> taskProfilingList(4, vector<struct ProfilingMetric>(2,
        {"tune", "trace_node_name", "TensorRedirect", 0, 1}));
    std::string workPath = "/tmp/";

    MOCKER(GetGradientNode)
    .expects(atMost(20))
    .will(returnValue(TUNE_SUCCESS));

    char file_name_t[] = "./error.json";

    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(1) << "}{}" << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();
    std::string path = file_name_t;

    MOCKER(GetTuneFusionPath)
    .stubs()
    .with(outBound(path))
    .will(returnValue(TUNE_SUCCESS));

    ret = HcomGradientFusionTune(taskProfilingList, workPath);
    EXPECT_EQ(ret, TUNE_E_OPEN_FILE_FAILURE);

    GlobalMockObject::verify();
}

HcclResult stub_HcomGetAlgorithm_invalidAlgo(u32 level, std::string &algo)
{
    algo = "strange";
    return HCCL_SUCCESS;
}

TEST_F(GradientSplitTuneTest, GetTopoInfoFromFile)
{
    vector<TopoInfo> toposInfo;
    GetTopoInfoFromFile(toposInfo);
    GlobalMockObject::verify();
}