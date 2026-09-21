/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/node.h"
#include "graph/op_desc.h"
#include "llt_hccl_stub_ge.h"
#include "graph/compute_graph.h"
#include "graph/ascend_string.h"
#include "graph/any_value.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "common/util/mem_utils.h"
#include "external/graph/types.h"
#include "ops_kernel_builder_registry.h"
#include "external/hcom/hcom_topo_info.h"
#include "external/register/hidden_inputs_func_registry.h"
#include "external/graph/operator_factory.h"

#include "hccl_stub.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
    const std::string ATTR_NAME_CONTINUOUS_INPUT = "continuous_input";
    const std::string ATTR_NAME_CONTINUOUS_OUTPUT = "continuous_output";
    const std::string ATTR_NAME_REFERENCE = "reference";
    const std::string ATTR_INTER_EVENT_IDENTIFY = "event_id";
    const std::string ATTR_NAME_STREAM_LABEL = "_stream_label";
    const std::string HCOM_ATTR_REDUCE_TYPE = "reduction";
    const std::string ATTR_NAME_INPUT_MEM_TYPE_LIST = "attr_name_input_mem_type_list";
    const std::string ATTR_NAME_OUTPUT_MEM_TYPE_LIST = "attr_name_output_mem_type_list";
    const std::string ATTR_NAME_WORKSPACE_TYPE_LIST = "attr_name_workspace_type_list";
    const std::string ATTR_NAME_IS_UNKNOWN_SHAPE = "_is_unknown_shape";
    const std::string ATTR_NAME_FORCE_UNKNOWN_SHAPE = "_force_unknown_shape";
    const std::string ATTR_NAME_PARALLEL_GROUP = "_parallel_group";
    const std::string ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE = "original_type";
    const std::string ATTR_NAME_NEED_MAP_RANK_ID = "_need_map_rank_id";
    const std::string ATTR_NAME_SEND_COUNT_MATRIX = "send_count_matrix";
    const std::string ATTR_NAME_QOS_SERVICE_LABEL = "_qos_service_label";
    const std::string ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_SHAPE = "_subgraph_multi_dims_input_shape";
    const std::string ATTR_NAME_PARALLEL_GROUP_ID = "_parallel_group_id";
    const std::string ATTR_NAME_IS_FIXED_ADDR_PRIOR = "_is_fixed_addr_prior";
    const std::string ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY = "_attached_resource_reuse_key";
    const std::string ATTR_NAME_ATTACHED_STREAM_INFO_LIST = "_attached_stream_info_list";
    const std::string ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG = "_attached_resource_required_flag";
    const std::string ATTR_NAME_ATTACHED_RESOURCE_NAME = "_attached_resource_name";
    const std::string ATTR_NAME_OP_RUN_INFO = "op_run_info";
    const std::string ATTR_SUPER_KERNEL_SCOPE = "_super_kernel_scope";
    const std::string ATTR_OP_VECTORCORE_NUM = "_op_vectorcore_num";
class NodeDummy : public ge::Node {
    public:
        NodeDummy(){;};
        ~NodeDummy(){;};
    };
    OpsKernelBuilderRegistrar::OpsKernelBuilderRegistrar(const std::string &kernel_lib_name, CreateFn fn) {}
    OpsKernelBuilderRegistrar::~OpsKernelBuilderRegistrar() {}
    Buffer::Buffer(){
        ;
    }
    Buffer::Buffer(const Buffer& other){
        ;
    }
    Buffer::~Buffer() {}
    Node::~Node(){
        ;
    }
    OpDescPtr opPtr_ = std::make_shared<OpDesc>();
    OpDescPtr Node::GetOpDesc() const {
        return opPtr_;
    }

    ge::graphStatus ge::NodeUtils::SetNodeParallelGroup(Node &node, const char *group_name)
    {
        return 0;
    }
    ge::graphStatus ge::NodeUtils::AppendInputAnchor(const NodePtr &node, const uint32_t num)
    {
        return 0;
    }
    ge::graphStatus ge::NodeUtils::AppendOutputAnchor(const NodePtr &node, const uint32_t num)
    {
        return 0;
    }
    ge::HcomTopoInfo &ge::HcomTopoInfo::Instance() {
        static ge::HcomTopoInfo hcom_topo_info;
        return hcom_topo_info;
    }

    ge::graphStatus ge::AttrHolder::DelAttr(std::string const&)
    {
        return 0;
    }
    int64_t rankSize;
    Status ge::HcomTopoInfo::SetGroupTopoInfo(char const *group, ge::HcomTopoInfo::TopoInfo const&info)
    {
        rankSize = 8;
        return ge::GRAPH_SUCCESS;
    }
    Status ge::HcomTopoInfo::SetGroupOrderedStream(const int32_t device_id, const char_t *group, void *stream)
    {
        return ge::GRAPH_SUCCESS;
    }
    void ge::HcomTopoInfo::UnsetGroupOrderedStream(const int32_t device_id, const char_t *group) {}
    bool ge::HcomTopoInfo::TryGetGroupTopoInfo(const char_t *group, HcomTopoInfo::TopoInfo &info)
    {
        return true;
    }
    Status ge::HcomTopoInfo::GetGroupRankSize(const char_t *group, int64_t &rank_size)
    {
        if (rankSize == 8) {
            rank_size = 8;
            return ge::GRAPH_SUCCESS;
        }
        HCCL_ERROR("rankSize has not been set. Please use Set before Get");
        return ge::GRAPH_FAILED;
    }
    ge::HiddenInputsFuncRegister::HiddenInputsFuncRegister(const HiddenInputsType input_type, 
        const GetHiddenAddrs func) {
    }
    OpDesc::OpDesc()
    : impl_(std::shared_ptr<OpDescImpl>(new OpDescImpl())) {}
    OpDesc::~OpDesc(){}
    ge::ProtoAttrMap& OpDesc::MutableAttrMap(){};
    ge::ConstProtoAttrMap& OpDesc::GetAttrMap() const{}

    std::string dummyType;
    void OpDesc::SetType(const string& type){
        dummyType = type;
    }
    
    std::string OpDesc::GetType() const{
        return dummyType;
    }
    int64_t OpDesc::GetId() const{
        return 0;
    }

    std::string dummyName = "test_op_name";
    void OpDesc::SetName(const string& name){
        dummyName = name;
    }
    std::string OpDesc::GetName() const{
        return dummyName;
    }

    int64_t dummyStream;
    void OpDesc::SetStreamId(int64_t streamId){
        dummyStream = streamId;
    }
    int64_t OpDesc::GetStreamId() const{
        return dummyStream;
    }

    size_t OpDesc::GetAllInputsSize() const
    {
        return 1024;
    }

    size_t OpDesc::GetInputsSize() const
    {
        return 1024;
    }

    std::map<string, uint32_t> OpDesc::GetAllInputName() const
    {
        std::map<string, uint32_t> name;
        name.emplace(std::make_pair("input_name", 0));
        return name;
    }
    bool OpDesc::UpdateOutputName(std::map<string, uint32_t> outputNameIdx)
    {
        return true;
    }

    size_t OpDesc::GetOutputsSize() const
    {
        return 1;
    }
    uint32_t OpDesc::GetAllOutputsDescSize() const
    {
        return 1;
    }
    std::vector<int64_t> OpDesc::GetInputOffset() const
    {
        std::vector<int64_t> vec(1024,1);
        return vec;
    }

    ge::graphStatus OpDesc::UpdateOutputDesc(uint32_t index, const ge::GeTensorDesc& tensor_Desc)
    {
        return GRAPH_SUCCESS;
    }
    GeTensorDescPtr dummy_inputsDesc = std::make_shared<GeTensorDesc>();
    GeTensorDescPtr dummy_outputsDesc = std::make_shared<GeTensorDesc>();
    ge::ConstGeTensorDescPtr OpDesc::GetInputDescPtr(uint32_t index) const
    {
        return dummy_inputsDesc;
    }
    ge::ConstGeTensorDescPtr OpDesc::GetOutputDescPtr(uint32_t index) const
    {
        return dummy_outputsDesc;
    }
    GeTensorDescPtr OpDesc::MutableInputDesc(uint32_t index) const {
        return dummy_inputsDesc;
    }
    GeTensorDesc::GeTensorDesc()
        : impl_(std::shared_ptr<GeTensorDescImpl>(new GeTensorDescImpl())) {}
    GeTensorDesc::GeTensorDesc(const GeTensorDesc &desc)
        : impl_(std::shared_ptr<GeTensorDescImpl>(new GeTensorDescImpl(*(desc.impl_)))) {}
    GeTensorDesc::GeTensorDesc(const GeShape& shape, Format format, DataType dt)
        : impl_(std::shared_ptr<GeTensorDescImpl>(new GeTensorDescImpl(shape, format, dt))) {}
    GeTensorDesc::GeTensorDesc(GeTensorDesc &&desc)
        : impl_(std::shared_ptr<GeTensorDescImpl>(new GeTensorDescImpl(std::move(*(desc.impl_))))) {}
    GeTensorDesc::~GeTensorDesc() {}
    GeTensorDesc GeTensorDesc::Clone() const
    {
        GeTensorDesc desc;
        return desc;
    };
    GeTensorDesc &GeTensorDesc::operator=(const GeTensorDesc &desc) {};
    GeTensorDesc &GeTensorDesc::operator=(GeTensorDesc &&desc) {};

    void GeTensorDesc::SetShape(const GeShape &shape) {};
    void GeTensorDesc::SetShape(GeShape &&shape) {};
    void GeTensorDesc::SetDataType(const DataType dtype) {};

    void GeTensorDesc::SetOriginShape(const GeShape &originShape) {};

    GeTensorDescImpl::GeTensorDescImpl() {}

    GeTensorDescImpl::GeTensorDescImpl(GeShape shape, Format format, DataType dt) : GeTensorDescImpl() {}

    GeTensorDescImpl::GeTensorDescImpl(const GeTensorDescImpl &desc) : GeTensorDescImpl() {}

    GeTensorDescImpl::GeTensorDescImpl(GeTensorDescImpl &&desc) : GeTensorDescImpl() {}

    GeTensorDescImpl::GeTensorDescImpl(const ProtoMsgOwner &proto_owner, proto::TensorDescriptor *proto_msg)
        : tensor_descriptor_(proto_owner, proto_msg) {}

    std::string OpDesc::GetInputNameByIndex(const uint32_t index) const {
        return "";
    }

    int32_t OpDesc::GetInputIndexByName(const std::string &name) const {
        return 0;
    }
    ge::ProtoAttrMap& GeTensorDesc::MutableAttrMap(){};
    ge::ConstProtoAttrMap& GeTensorDesc::GetAttrMap() const{}
    ge::GeTensorDesc dummyOutputDesc;
    const ge::GeTensorDesc& OpDesc::GetOutputDesc(uint32_t index) const
    {
        return dummyOutputDesc;
    }
    GeTensorDescPtr OpDesc::MutableOutputDesc(uint32_t index) const {
        return dummy_outputsDesc;
    }
    ge::DataType ge::GeTensorDesc::GetDataType() const
    {
        return (ge::DataType)ge::DT_FLOAT;
    }
    ge::Format ge::GeTensorDesc::GetFormat() const
    {
        return (ge::Format)0;
    }
    ge::Format GeTensorDesc::GetOriginFormat() const {
        return (ge::Format)0;
    }
    void GeTensorDesc::SetFormat(Format format) {
        return;
    }
    ge::GeShape::GeShape(){}
    ge::GeShape::GeShape(std::vector<int64_t> s){}
    ge::GeShape::GeShape(const GeShape& other){}
    ge::GeShape::GeShape(GeShape&& other){}
    GeShape& GeShape::operator=(const GeShape &other){}
    GeShape& GeShape::operator=(GeShape &&other){}

    ge::OpDesc::OpDesc(const string &name, const string &type)
    : impl_(std::shared_ptr<OpDescImpl>(new OpDescImpl(name, type))) {}
    GeShape::~GeShape(){}

    ge::GeShape dummyShape;
    const ge::GeShape& ge::GeTensorDesc::GetShape() const
    {
        return dummyShape;
    }

    const ge::GeShape& ge::GeTensorDesc::GetOriginShape() const
    {
        return dummyShape;
    }

    int64_t GeShape::GetShapeSize() const
    {
        return 1024;
    }

    bool GeShape::IsScalar() const {
        return false;
    }

    std::vector<int64_t> GeShape::GetDims() const {
        std::vector<int64_t> dims{1024};
        return dims;
    }

    size_t GeShape::GetDimNum() const {
        return 1024;
    }

    int64_t GeShape::GetDim(size_t idx) const {
        return 10;
    }
    ge::graphStatus GeShape::SetDim(size_t idx, int64_t value){
        return 0;
    }

    bool dummyHasAttr_srcRank;
    bool dummyHasAttr_destRank;
    bool dummyHasAttr_srTag;
    bool dummyHasAttr_tag;
    bool dummyHasAttr_group;
    bool dummyHasAttr_rankSize;
    bool dummyHasAttr_dtype;
    bool dummyHasAttr_shape;
    bool dummyHasAttr_eventid;
    bool dummyHasAttr_comm;
    bool dummyHasAttr_dumpSize;
    bool dummyHasAttr_dumpType;
    bool dummyHasAttr_maxNum;
    bool dummyHasAttr_embeddingDim;
    bool dummyHasAttr_maxEmbeddingDim;
    bool dummyHasAttr_needMapRank;
    bool dummyHasAttr_sendCountMatrix;
    bool dummyHasAttr_parallel_group;
    bool dummyHasAttr_flags;
    bool dummyHasAttr_superKernelScope;
    bool dummyHasAttr_OpVectorcoreNum = true;

        string dummyString;
    s32 dummyInt32;
    u64 dummyInt64;
    u32 dummyRankSize;
    vector<int64_t> dummy_workspaceBytes;
    vector<int64_t> dummyVecInt64;
    bool AttrUtils::HasAttr(ConstAttrHolderAdapter&& obj, const string& name) {
        if(name == "DUMMY_SET_TRUE_SRCRANK"){
            dummyHasAttr_srcRank = true;
        } else if (name == "DUMMY_SET_FALSE_SRCRANK"){
            dummyHasAttr_srcRank = false;
        } else if (name == "DUMMY_SET_TRUE_DESTRANK"){
            dummyHasAttr_destRank = true;
        } else if (name == "DUMMY_SET_FALSE_DESTRANK"){
            dummyHasAttr_destRank = false;
        } else if (name == "DUMMY_SET_TRUE_SRTAG"){
            dummyHasAttr_srTag = true;
        } else if (name == "DUMMY_SET_FALSE_SRTAG"){
            dummyHasAttr_srTag = false;
        } else if (name == "DUMMY_SET_TRUE_TAG"){
            dummyHasAttr_tag = true;
        } else if (name == "DUMMY_SET_FALSE_TAG"){
            dummyHasAttr_tag = false;
        } else if (name == "DUMMY_SET_TRUE_GROUP"){
            dummyHasAttr_group = true;
        } else if (name == "DUMMY_SET_FALSE_GROUP"){
            dummyHasAttr_group = false;
        } else if (name == "DUMMY_SET_TRUE_COMM"){
            dummyHasAttr_comm = true;
        } else if (name == "DUMMY_SET_FALSE_COMM"){
            dummyHasAttr_comm = false;
        } else if (name == "DUMMY_SET_TRUE_RANK_SIZE") {
            dummyHasAttr_rankSize = true;
        } else if (name == "DUMMY_SET_FALSE_RANK_SIZE") {
            dummyHasAttr_rankSize = false;
        } else if (name == "DUMMY_SET_TRUE_DTYPE"){
            dummyHasAttr_dtype = true;
        } else if (name == "DUMMY_SET_FALSE_DTYPE"){
            dummyHasAttr_dtype = false;
        } else if (name == "DUMMY_SET_TRUE_SHAPE"){
            dummyHasAttr_shape = true;
        } else if (name == "DUMMY_SET_FALSE_SHAPE"){
            dummyHasAttr_shape = false;
        } else if (name == "DUMMY_SET_TRUE_EVENTID") {
            dummyHasAttr_eventid = true;
        } else if (name == "DUMMY_SET_FALSE_RANKSIZE") {
            dummyHasAttr_rankSize = false;
        } else if (name == "DUMMY_SET_TRUE_RANKSIZE") {
            dummyHasAttr_rankSize = true;
        } else if (name == "DUMMY_SET_FALSE_DUMPSIZE") {
            dummyHasAttr_dumpSize = false;
        } else if (name == "DUMMY_SET_TRUE_DUMPSIZE") {
            dummyHasAttr_dumpSize = true;
        } else if (name == "DUMMY_SET_FALSE_DUMPTYPE") {
            dummyHasAttr_dumpType = false;
        } else if (name == "DUMMY_SET_TRUE_DUMPTYPE") {
            dummyHasAttr_dumpType = true;
        } else if (name == "DUMMY_SET_TRUE_MAXNUM") {
            dummyHasAttr_maxNum = true;
        } else if (name == "DUMMY_SET_TRUE_EMBEDDINGDIM") {
            dummyHasAttr_embeddingDim = true;
        } else if (name == "DUMMY_SET_TRUE_MAXEMBEDDINGDIM") {
            dummyHasAttr_maxEmbeddingDim = true;
        } else if (name == "HCOM_ATTR_DTYPE") {
            dummyHasAttr_dtype = true;
        } else if (name == "DUMMY_SET_TRUE_NEEDMAPRANK") {
            dummyHasAttr_needMapRank = true;
        } else if (name == "DUMMY_SET_SEND_COUNT_MATRIX") {
            dummyHasAttr_sendCountMatrix = true;
        } else if (name == "DUMMY_SET_TRUE_PARALLEL_GROUP") {
            dummyHasAttr_parallel_group = true;
        } else if (name == "DUMMY_SET_FALSE_PARALLEL_GROUP") {
            dummyHasAttr_parallel_group = false;
        } else if (name == "DUMMY_SET_TRUE_FLAGS") {
            dummyHasAttr_flags = true;
        } else if (name == "ATTR_SUPER_KERNEL_SCOPE") {
            dummyHasAttr_superKernelScope = true;
        } else if (name == "ATTR_OP_VECTORCORE_NUM") {
            dummyHasAttr_OpVectorcoreNum = true;
        } else if (name == "ATTR_OP_VECTORCORE_NUM_CLEAR") {
            dummyHasAttr_OpVectorcoreNum = false;
        } else {
            ;
        }

        if (name == "src_rank") {
            return dummyHasAttr_srcRank;
        } else if (name == "dest_rank") {
            return dummyHasAttr_destRank;
        } else if (name == "sr_tag") {
            return dummyHasAttr_srTag;
        } else if (name == "tag") {
            return dummyHasAttr_tag;
        } else if (name == "group") {
            return dummyHasAttr_group;
        } else if (name == "comm") {
            return dummyHasAttr_comm;
        } else if (name == "dtype") {
            return dummyHasAttr_dtype;
        } else if (name == "shape") {
            return dummyHasAttr_shape;
        } else if (name == "event_id") {
            return dummyHasAttr_eventid;
        } else if (name == "rank_size") {
            return dummyHasAttr_rankSize;
        } else if (name == "global_workspace_size") {
            return dummyHasAttr_dumpSize;
        } else if (name == "global_workspace_type") {
            return dummyHasAttr_dumpType;
        } else if (name == "send_counts") {
            return true;
        } else if (name == "max_num") {
            return dummyHasAttr_maxNum;
        } else if (name == "embedding_dim") {
            return dummyHasAttr_embeddingDim;
        } else if (name == "_embedding_dim") {
            return dummyHasAttr_maxEmbeddingDim;
        } else if (name == "_need_map_rank_id") {
            return dummyHasAttr_needMapRank;
        } else if (name == "send_count_matrix") {
            return dummyHasAttr_sendCountMatrix;
        } else if (name == "_parallel_group") {
            return dummyHasAttr_parallel_group;
        } else if (name == "insert_option") {
            return true;
        } else if (name == "flags") {
            return dummyHasAttr_flags;
        } else if (name == "_super_kernel_scope") {
            return true;
        } else if (name == "_op_vectorcore_num") {
            return dummyHasAttr_OpVectorcoreNum;
        } else {
            return false;
        }
    }

    std::map<std::string, int64_t> dummy_Int64MAP;
    std::map<std::string, std::string> dummyStringMAP;
    bool AttrUtils::SetStr(AttrHolderAdapter&& obj, const string& name, const string& value){
        auto iter = dummyStringMAP.find(name);
        if (iter ==  dummyStringMAP.end()) {
            dummyStringMAP.insert(std::make_pair(name, value));
        } else {
            iter->second = value;
        }
        return true;
    }
    bool AttrUtils::GetStr(ConstAttrHolderAdapter&&  obj, const string& name, string& value){
        auto iter = dummyStringMAP.find(name);
        if (iter ==  dummyStringMAP.end()) {
            return false;
        } else {
            value = iter->second;
            return true;
        }
    }
    bool AttrUtils::SetListStr(AttrHolderAdapter &&obj, const std::string &name, const std::vector<std::string> &value) {
        return true;
    }

    bool AttrUtils::GetInt(ConstAttrHolderAdapter&&  obj, const string& name, uint32_t& value){
        auto iter = dummy_Int64MAP.find(name);
        if (iter ==  dummy_Int64MAP.end()) {
            return false;
        } else {
            value = (uint32_t)(iter->second);
            return true;
        }
    }

    bool AttrUtils::GetInt(ConstAttrHolderAdapter&&  obj, const string& name, int32_t& value){
        auto iter = dummy_Int64MAP.find(name);
        if (iter ==  dummy_Int64MAP.end()) {
            return false;
        } else {
            value = (int32_t)(iter->second);
            return true;
        }
    }

    bool AttrUtils::GetInt(ConstAttrHolderAdapter&&  obj, const string& name, int64_t& value){
        auto iter = dummy_Int64MAP.find(name);
        if (iter ==  dummy_Int64MAP.end()) {
            return false;
        } else {
            value = iter->second;
            return true;
        }
    }

    bool AttrUtils::SetInt(ge::AttrUtils::AttrHolderAdapter&& obj, const string& name, const int64_t& value){
        auto iter = dummy_Int64MAP.find(name);
        if (iter ==  dummy_Int64MAP.end()) {
            dummy_Int64MAP.insert(std::make_pair(name, value));
        } else {
            iter->second = value;
        }
        return true;
    }

    bool AttrUtils::SetTensor(AttrHolderAdapter &&obj, const string &name, const GeTensorPtr &value){
        return true;
    }

    std::map<std::string, bool> dummyBoolMAP;
    bool AttrUtils::SetBool(ge::AttrUtils::AttrHolderAdapter&& obj, const string& name, const bool& value){
        auto iter = dummyBoolMAP.find(name);
        if (iter ==  dummyBoolMAP.end()) {
            dummyBoolMAP.insert(std::make_pair(name, value));
        } else {
            iter->second = value;
        }
        return true;
    }
    bool AttrUtils::GetBool(ConstAttrHolderAdapter &&obj, const string &name, bool &value)
    {
        auto iter = dummyBoolMAP.find(name);
        if (iter ==  dummyBoolMAP.end()) {
            return false;
        } else {
            value = iter->second;
            return true;
        }
    }

    bool AttrUtils::SetListBool(ge::AttrUtils::AttrHolderAdapter&& obj, const string& name, const std::vector<bool>& value){
        auto iter = dummyBoolMAP.find(name);
        if (value.size() == 0) {
            return true;
        }
        if (iter ==  dummyBoolMAP.end()) {
            dummyBoolMAP.insert(std::make_pair(name, value[0]));
        } else {
            iter->second = value[0];
        }
        return true;
    }

    bool AttrUtils::SetListInt(ge::AttrUtils::AttrHolderAdapter&& obj, const string& name, const vector<int64_t>& value){
        dummyVecInt64 = value;
        return true;
    }
    bool AttrUtils::GetListInt(ConstAttrHolderAdapter &&obj, const string &name, vector<int64_t> &value) {
        value.clear();
        if (name == "send_count_matrix") {
            for (u32 i = 0; i < 64; i++) {
                value.push_back(i+1);
            }
        } else {
            value.push_back(1024);
            value.push_back(32);
        }
        return true;
    }

    bool AttrUtils::GetDataType(ConstAttrHolderAdapter &&obj, const string &name, ge::DataType &value) {
        value = (ge::DataType)3;
        return true;
    }

    bool AttrUtils::SetListNamedAttrs(AttrHolderAdapter &&obj, const std::string &name,
                                  const std::vector<NamedAttrs> &value) {
        // 目前代码中仅有设置，没有读取的代码，因此这里实现为空，如果后续有使用，请补充实际的逻辑
        return true;
    }

    bool AttrUtils::GetListNamedAttrs(ConstAttrHolderAdapter &&obj,
                                  const std::string &name, std::vector<NamedAttrs> &value) {
        // 目前代码中没有使用该接口，但是为了配套，直接返回false
        HCCL_ERROR("AttrUtils::GetListNameAttrs not implement in stub, please change it");
        return false;
    }

    AttrStore::AttrStore(const AttrStore &other) {}

    void NamedAttrs::SetName(const std::string &name) {
    }

    std::string NamedAttrs::GetName() const {
        return "";
    }

    AnyValue NamedAttrs::GetItem(const std::string &key) const {
        AnyValue value;
        return value;
    }

    ProtoAttrMap &NamedAttrs::MutableAttrMap() {
        static AttrStore attrs;
        return attrs;
    }

    ConstProtoAttrMap &NamedAttrs::GetAttrMap() const {
        static AttrStore attrs;
        return attrs;
    }

    const std::set<std::string> AttrHolder::GetAllAttrNames() const {
        return {"group"};
    }


    void OpDesc::SetWorkspaceBytes(const vector<int64_t>& workspaceBytes) {
        dummy_workspaceBytes = workspaceBytes;
    }
    void OpDesc::SetOpInferDepends(const vector<string> &depend_names) {}
    vector<int64_t> OpDesc::GetWorkspaceBytes() const {
        return dummy_workspaceBytes;
    }
    ge::graphStatus TensorUtils::GetSize(const ge::GeTensorDesc& tensorDesc, int64_t& size)
    {
        size = 1024;
        return GRAPH_SUCCESS;
    }
    ge::graphStatus TensorUtils::CalcTensorMemSize(const GeShape &shape, Format format, DataType data_type, int64_t &mem_size)
    {
        mem_size = 1024;
        return GRAPH_SUCCESS;
    }
    void TensorUtils::SetSize(GeTensorDesc& tensorDesc, int64_t size)
    { }


    void TensorUtils::SetReuseInputIndex(GeTensorDesc &tensor_desc, uint32_t idx)
    {
        return;
    }

    void TensorUtils::SetReuseInput(GeTensorDesc &tensor_desc, bool flag) {
        return;
    }

    Shape::Shape(const std::vector<int64_t>& dims)
    {
    }
    int64_t Shape::GetShapeSize() const
    {
        return 1024;
    }

    int64_t GetSizeInBytes(int64_t element_count, DataType data_type) 
    {
        return 1024;
    }

    ComputeGraphImpl::ComputeGraphImpl(const std::string &name) :
        nodes_(),
        input_nodes_(),
        sub_graph_(),
        name_(name),
        is_valid_flag_(false),
        need_iteration_(false)
    {};

    ComputeGraph::ComputeGraph(const std::string &name)
    : impl_(std::shared_ptr<ComputeGraphImpl>(new ComputeGraphImpl(name))) {}

    ComputeGraph::ComputeGraph(const char_t *name)
    : impl_(std::shared_ptr<ComputeGraphImpl>(new ComputeGraphImpl(std::string(name)))) {}

    ComputeGraph::~ComputeGraph() {}

    std::string ComputeGraphImpl::GetName() const
    {
        return name_;
    }

    std::string ComputeGraph::GetName() const
    {
        return impl_->GetName();
    }

    std::vector<std::shared_ptr<ge::ComputeGraph>> ComputeGraphImpl::GetAllSubgraphs() const
    {
        return sub_graph_;
    }

    std::vector<std::shared_ptr<ge::ComputeGraph>> ComputeGraph::GetAllSubgraphs() const
    {
        return impl_->GetAllSubgraphs();
    }

    std::shared_ptr<ComputeGraph> ComputeGraphImpl::AddSubGraph(const std::shared_ptr<ComputeGraph> &sub_graph)
    {
        if (sub_graph == nullptr) {
            HCCL_ERROR("The graph ptr should not be null.");
            return nullptr;
        }
        sub_graph_.push_back(sub_graph);
        return sub_graph;
    }

    std::shared_ptr<ComputeGraph> ComputeGraph::AddSubGraph(std::shared_ptr<ComputeGraph> sub_graph)
    {
        return impl_->AddSubGraph(sub_graph);
    }

    ComputeGraphImpl::Vistor<NodePtr> ComputeGraphImpl::GetDirectNode(const ConstComputeGraphPtr &compute_graph) const
    {
        return Vistor<NodePtr>(compute_graph, nodes_);
    }

    ComputeGraph::Vistor<NodePtr> ComputeGraph::GetDirectNode() const
    {
        return impl_->GetDirectNode(shared_from_this());
    }

    ComputeGraphImpl::Vistor<NodePtr> ComputeGraphImpl::GetAllNodes(const ConstComputeGraphPtr &compute_graph) const
    {
        return Vistor<NodePtr>(compute_graph, nodes_);
    }

    ComputeGraph::Vistor<NodePtr> ComputeGraph::GetAllNodes() const
    {
        std::shared_ptr<const ge::ComputeGraph> graph;
        return impl_->GetAllNodes(graph);
    }

    OutDataAnchor::Vistor<InDataAnchorPtr> OutDataAnchor::GetPeerInDataAnchors() const
    {
        vector<InDataAnchorPtr> ret;
        for (auto anchor: impl_->peer_anchors_) {
            auto inDataAnchor = Anchor::DynamicAnchorCast<InDataAnchor>(anchor.lock());
            if (inDataAnchor != nullptr) {
                ret.push_back(inDataAnchor);
            }
        }
        return OutDataAnchor::Vistor<InDataAnchorPtr>(shared_from_this(), ret);
    }

    graphStatus OpDesc::AddOutputDesc(const ge::GeTensorDesc& output_desc)
    {
        return GRAPH_SUCCESS;
    }
    graphStatus OpDesc::AddOutputDesc(const string &name, const GeTensorDesc &output_desc){
        return GRAPH_SUCCESS;
    }
    NodePtr Anchor::GetOwnerNode() const
    {
        return impl_->owner_node_.lock();
    }

    DataAnchor::DataAnchor(const NodePtr& ownerNode, int idx) : Anchor(ownerNode, idx)
    {
    }

    AnchorImpl::AnchorImpl(const NodePtr &owner_node, int idx) : owner_node_(owner_node), idx_(idx) {}

    Anchor::Anchor(const NodePtr &owner_node, int idx)
    : impl_(std::shared_ptr<AnchorImpl>(new AnchorImpl(owner_node, idx)))
    {
    }

    Anchor::~Anchor() {}

    bool Anchor::IsTypeOf(TYPE type) const
    {
        return (Anchor::TypeOf<Anchor>() == type);
    }

    bool Anchor::IsTypeIdOf(const TypeId& type) const
    {
        return GetTypeId<Anchor>() == type;
    }

    Anchor::TYPE Anchor::GetSelfType() const
    {
        return TypeOf<Anchor>();
    }

    bool DataAnchor::IsTypeOf(TYPE type) const
    {
        if(Anchor::TypeOf<DataAnchor>() == type){
            return true;
        }
        return Anchor::IsTypeOf(type);
    }


    Anchor::TYPE DataAnchor::GetSelfType() const
    {
        return Anchor::TypeOf<DataAnchor>();
    }

    std::string Node::GetName() const
    {
        return dummyName;
    }

    OutControlAnchorPtr Node::GetOutControlAnchor() const
    {
        return impl_->GetOutControlAnchor();
    }

    OutControlAnchorPtr Node::NodeImpl::GetOutControlAnchor() const
    {
        return out_control_anchor_;
    }

    uint32_t Node::GetAllOutDataAnchorsSize() const
    {
        return impl_->GetAllOutDataAnchorsSize();
    }

    uint32_t Node::NodeImpl::GetAllOutDataAnchorsSize() const
    {
        return static_cast<uint32_t>(out_data_anchors_.size());
    }

    Node::Vistor<std::pair<NodePtr, OutDataAnchorPtr>> Node::GetInDataNodesAndAnchors() const
    {
        std::vector<std::pair<NodePtr, OutDataAnchorPtr>> vec;
        const ConstNodePtr owner_node;
        ge::NodePtr node;
        ge::OutDataAnchorPtr anchor_ptr;
        vec.push_back(std::make_pair(node, anchor_ptr));
        return Node::Vistor<std::pair<NodePtr, OutDataAnchorPtr>>(owner_node, vec);
    }

    Node::Vistor<std::pair<NodePtr, OutDataAnchorPtr>>
        Node::NodeImpl::GetInDataNodesAndAnchors(const ConstNodePtr &owner_node) const
    {
        std::vector<std::pair<NodePtr, OutDataAnchorPtr>> vec;
        return Node::Vistor<std::pair<NodePtr, OutDataAnchorPtr>>(owner_node, vec);
    }

    OutControlAnchor::OutControlAnchor(const NodePtr& ownerNode) : ControlAnchor(ownerNode)
    {
    }
    OutControlAnchor::OutControlAnchor(const NodePtr& ownerNode, int idx) : ControlAnchor(ownerNode, idx)
    {
    }

    bool OutControlAnchor::Equal(AnchorPtr anchor) const
    {
        return false;
    }
    bool OutControlAnchor::IsTypeOf(TYPE type) const
    {
        if(Anchor::TypeOf<OutControlAnchor>() == type){
            return true;
        }
        return ControlAnchor::IsTypeOf(type);
    }

    Anchor::TYPE OutControlAnchor::GetSelfType() const
    {
        return Anchor::TypeOf<OutControlAnchor>();
    }
    ControlAnchor::ControlAnchor(const NodePtr& ownerNode) : Anchor(ownerNode, -1)
    {
    }
    ControlAnchor::ControlAnchor(const NodePtr& ownerNode, int idx) : Anchor(ownerNode, idx)
    {
    }
    bool ControlAnchor::IsTypeOf(TYPE type) const
    {
        if(Anchor::TypeOf<ControlAnchor>() == type){
            return true;
        }
        return Anchor::IsTypeOf(type);
    }

    Anchor::TYPE ControlAnchor::GetSelfType() const
    {
        return Anchor::TypeOf<ControlAnchor>();
    }

    InControlAnchorPtr Node::GetInControlAnchor() const
    {
        return impl_->in_control_anchor_;
    }
    InControlAnchor::InControlAnchor(const NodePtr& ownerNode) : ControlAnchor(ownerNode)
    {
    }
    InControlAnchor::InControlAnchor(const NodePtr& ownerNode, int idx) : ControlAnchor(ownerNode, idx)
    {
    }
    bool InControlAnchor::Equal(AnchorPtr anchor) const
    {
        return false;
    }

    bool InControlAnchor::IsTypeOf(TYPE type) const
    {
        if(Anchor::TypeOf<InControlAnchor>() == type){
            return true;
        }
        return ControlAnchor::IsTypeOf(type);
    }

    Anchor::TYPE InControlAnchor::GetSelfType() const
    {
        return Anchor::TypeOf<InControlAnchor>();
    }
    InDataAnchorPtr Node::GetInDataAnchor(int idx) const
    {
        if (idx < 0 || idx >= static_cast<int>(impl_->in_data_anchors_.size())) {
            return nullptr;
        } else {
            return impl_->in_data_anchors_[idx];
        }
    }
    bool InDataAnchor::Equal(AnchorPtr anchor) const
    {
        return false;
    }

    bool InDataAnchor::IsTypeOf(TYPE type) const
    {
        if(Anchor::TypeOf<InDataAnchor>() == type){
            return true;
        }
        return DataAnchor::IsTypeOf(type);
    }

    bool DataAnchor::IsTypeIdOf(const TypeId &type) const {
    if (GetTypeId<DataAnchor>() == type) {
        return true;
    }
    return Anchor::IsTypeIdOf(type);
    }
    bool InDataAnchor::IsTypeIdOf(const TypeId &type) const {
    if (GetTypeId<InDataAnchor>() == type) {
        return true;
    }
    return DataAnchor::IsTypeIdOf(type);
    }
    bool OutDataAnchor::IsTypeIdOf(const TypeId &type) const {
    if (GetTypeId<OutDataAnchor>() == type) {
        return true;
    }
        return DataAnchor::IsTypeIdOf(type);
    }
    bool ControlAnchor::IsTypeIdOf(const TypeId &type) const {
    if (GetTypeId<ControlAnchor>() == type) {
        return true;
    }
        return Anchor::IsTypeIdOf(type);
    }
    bool InControlAnchor::IsTypeIdOf(const TypeId &type) const {
    if (GetTypeId<InControlAnchor>() == type) {
        return true;
    }
        return ControlAnchor::IsTypeIdOf(type);
    }
    bool OutControlAnchor::IsTypeIdOf(const TypeId &type) const {
    if (GetTypeId<OutControlAnchor>() == type) {
        return true;
    }
        return ControlAnchor::IsTypeIdOf(type);
    }
    Anchor::TYPE InDataAnchor::GetSelfType() const
    {
        return Anchor::TypeOf<InDataAnchor>();
    }
    OutDataAnchor::OutDataAnchor(const NodePtr& ownerNode, int idx) : DataAnchor(ownerNode, idx)
    {
    }
    bool OutDataAnchor::Equal(AnchorPtr anchor) const
    {
        return false;
    }

    bool OutDataAnchor::IsTypeOf(TYPE type) const
    {
        if(Anchor::TypeOf<OutDataAnchor>() == type){
            return true;
        }
        return DataAnchor::IsTypeOf(type);
    }
    Anchor::TYPE OutDataAnchor::GetSelfType() const
    {
        return Anchor::TypeOf<OutDataAnchor>();
    }
    graphStatus GraphUtils::AddEdge(const OutDataAnchorPtr &src, const InControlAnchorPtr &dst)
    {
        return GRAPH_SUCCESS;
    }
    graphStatus GraphUtils::AddEdge(const OutDataAnchorPtr &src,const  InDataAnchorPtr &dst)
    {
        return GRAPH_SUCCESS;
    }
    graphStatus GraphUtils::AddEdge(const OutControlAnchorPtr &src,const InControlAnchorPtr &dst)
    {
        return GRAPH_SUCCESS;
    }
    graphStatus GraphUtils::RemoveEdge(const OutDataAnchorPtr &src,const InDataAnchorPtr &dst)
    {
        return GRAPH_SUCCESS;
    }
    graphStatus GraphUtils::RemoveEdge(const OutControlAnchorPtr &src,const InControlAnchorPtr &dst)
    {
        return GRAPH_SUCCESS;
    }
    graphStatus GraphUtils::RemoveEdge(const OutDataAnchorPtr &src,const InControlAnchorPtr &dst)
    {
        return GRAPH_SUCCESS;
    }
    graphStatus GraphUtils::RemoveEdge(const AnchorPtr &src,const AnchorPtr &dst)
    {
        return GRAPH_SUCCESS;
    }

    void GraphUtils::DumpGEGraphToOnnx(const ComputeGraph &compute_graph, const std::string &suffix) {
    return;
    }

    void GraphUtils::DumpGEGraph(const ComputeGraphPtr &graph, const std::string &suffix, bool is_always_dump,
        const std::string &user_graph_name) {
    return;
}
    OutDataAnchorPtr InDataAnchor::GetPeerOutAnchor() const
    {
        if (impl_->peer_anchors_.empty()) {
            return nullptr;
        } else {
            return Anchor::DynamicAnchorCast<OutDataAnchor>(impl_->peer_anchors_.begin()->lock());
        }
    }
    OutDataAnchorPtr Node::GetOutDataAnchor(int idx) const
    {
        if (idx < 0 || idx >= static_cast<int>(impl_->out_data_anchors_.size())) {
            return nullptr;
        } else {
            return impl_->out_data_anchors_[idx];
        }
    }
    graphStatus ComputeGraphImpl::RemoveNode(const NodePtr &node)
    {
        if (!nodes_.empty()) {
            nodes_.pop_back();
        }
        return GRAPH_SUCCESS;
    }

    graphStatus ComputeGraph::RemoveNode(const NodePtr &node)
    {
        return impl_->RemoveNode(node);
    }

    NodePtr ComputeGraphImpl::AddNode(OpDescPtr op, const ComputeGraphPtr &compute_graph)
    {
        NodePtr node_ptr = std::make_shared<NodeDummy>();
        node_ptr->Init();
        nodes_.push_back(node_ptr);
        HCCL_INFO("add node success");
        return node_ptr;
    }

    NodePtr ComputeGraph::AddNode(OpDescPtr op)
    {
        HCCL_INFO("add Node start");
        if (impl_ == nullptr) {
            HCCL_ERROR("add Node failde, impl is nullptr");
        }
        ComputeGraphPtr graph = std::make_shared<ComputeGraph>("name");
        return impl_->AddNode(op, graph);
    }

    Node::Vistor<InDataAnchorPtr> Node::GetAllInDataAnchors() const
    {
        return Vistor<InDataAnchorPtr>(shared_from_this(), impl_->in_data_anchors_);
    }

    OutDataAnchor::Vistor<InControlAnchorPtr> OutDataAnchor::GetPeerInControlAnchors() const
    {
        vector<InControlAnchorPtr> ret;
        for (auto anchor: impl_->peer_anchors_) {
            auto inControlAnchor = Anchor::DynamicAnchorCast<InControlAnchor>(anchor.lock());
            if (inControlAnchor != nullptr) {
                ret.push_back(inControlAnchor);
            }
        }
        return OutDataAnchor::Vistor<InControlAnchorPtr>(shared_from_this(), ret);
    }

    Node::Node()
        : impl_(std::shared_ptr<NodeImpl>(new NodeImpl())) {}

    Node::Node(const OpDescPtr &op, const ComputeGraphPtr &owner_graph)
        : impl_(std::shared_ptr<NodeImpl>(new NodeImpl(op, owner_graph))) {}

    graphStatus Node::Init()
    {
        HCCL_INFO("Node init start");
        if (impl_ == nullptr) {
            return GRAPH_FAILED;
        }
        impl_->in_control_anchor_ = std::make_shared<InControlAnchor>(shared_from_this(), -1);
        impl_->out_control_anchor_ = std::make_shared<OutControlAnchor>(shared_from_this(), -1);
        shared_ptr<InDataAnchor> archor1;
        archor1 = std::make_shared<InDataAnchor>(shared_from_this(), impl_->in_data_anchors_.size());
        impl_->in_data_anchors_.push_back(archor1);
        shared_ptr<OutDataAnchor> archor2;
        archor2 = std::make_shared<OutDataAnchor>(shared_from_this(), impl_->in_data_anchors_.size());
        impl_->out_data_anchors_.push_back(archor2);
        impl_->out_data_anchors_.push_back(archor2);
        HCCL_INFO("Node init success");
        return GRAPH_SUCCESS;
    }
    GeTensorDescPtr inputsDescDummy = std::make_shared<GeTensorDesc>();
    const GeTensorDesc &OpDesc::GetInputDesc(uint32_t index) const
    {
        return *(inputsDescDummy.get());
    }
    OpDesc::Vistor<GeTensorDesc> OpDesc::GetAllInputsDesc() const
    {
        vector<GeTensorDesc> temp{};
        temp.resize(1);
        return OpDesc::Vistor<GeTensorDesc>(shared_from_this(), temp);
    }

    std::shared_ptr<ComputeGraph> owner_graph1;
    ComputeGraph* Node::GetOwnerComputeGraphBarePtr() const {
        owner_graph1 = std::make_shared<ComputeGraph>("test_graph_root1");
        return owner_graph1.get();
    }

    std::shared_ptr<ComputeGraph> owner_graph;
    ComputeGraphPtr Node::GetOwnerComputeGraph() const
    {
        owner_graph = std::make_shared<ComputeGraph>("test_graph_root");
        owner_graph->SetGraphID(1);
        owner_graph->SetSessionID(1);
        return owner_graph;
    }

    void ComputeGraph::SetGraphID(uint32_t graph_id) {
        impl_->SetGraphID(graph_id);
    }

    void ComputeGraph::SetSessionID(uint64_t session_id) {
        impl_->SetSessionID(session_id);
    }

    ComputeGraphPtr GraphUtils::FindRootGraph(ComputeGraphPtr graph){
        return owner_graph;
    }
    graphStatus OpDesc::AddInputDesc(const ge::GeTensorDesc& input_desc)
    {
        int index = static_cast<int>(impl_->inputs_desc_.size());
        return AddInputDesc("__input"+std::to_string(index), input_desc);
    }
    graphStatus OpDesc::AddInputDesc(const string& name, const ge::GeTensorDesc& input_desc)
    {
        int index = static_cast<int>(impl_->inputs_desc_.size());

        shared_ptr<GeTensorDesc> in_desc;
        try {
            in_desc = std::make_shared<GeTensorDesc>(input_desc);
        }catch(...) {
            return GRAPH_FAILED;
        }
        impl_->inputs_desc_.push_back(in_desc);
        (void)impl_->input_name_idx_.insert(make_pair(name, index));
        return GRAPH_SUCCESS;
    }
    int Anchor::GetIdx() const
    {
        return impl_->idx_;
    }

    InControlAnchor::Vistor<OutControlAnchorPtr> InControlAnchor::GetPeerOutControlAnchors() const
    {
        vector<OutControlAnchorPtr> ret;
        for (auto anchor: impl_->peer_anchors_) {
            auto outControlAnchor = Anchor::DynamicAnchorCast<OutControlAnchor>(anchor.lock());
            if (outControlAnchor != nullptr) {
                ret.push_back(outControlAnchor);
            }
        }
        return InControlAnchor::Vistor<OutControlAnchorPtr>(shared_from_this(), ret);
    }
    Node::Vistor<OutDataAnchorPtr> Node::GetAllOutDataAnchors() const
    {
        return Vistor<OutDataAnchorPtr>(shared_from_this(), impl_->out_data_anchors_);
    }
    InControlAnchor::Vistor<OutDataAnchorPtr> InControlAnchor::GetPeerOutDataAnchors() const
    {
        vector<OutDataAnchorPtr> ret;
        for (auto anchor: impl_->peer_anchors_) {
            auto outDataAnchor = Anchor::DynamicAnchorCast<OutDataAnchor>(anchor.lock());
            if (outDataAnchor != nullptr) {
                ret.push_back(outDataAnchor);
            }
        }
        return InControlAnchor::Vistor<OutDataAnchorPtr>(shared_from_this(), ret);
    }
    OutControlAnchor::Vistor<InControlAnchorPtr> OutControlAnchor::GetPeerInControlAnchors() const
    {
        vector<InControlAnchorPtr> ret;
        for (auto anchor: impl_->peer_anchors_) {
            auto inControlAnchor = Anchor::DynamicAnchorCast<InControlAnchor>(anchor.lock());
            if (inControlAnchor != nullptr) {
                ret.push_back(inControlAnchor);
            }
        }
        return OutControlAnchor::Vistor<InControlAnchorPtr>(shared_from_this(), ret);
    }
    OpDescPtr AttrUtils::CloneOpDesc(const ConstOpDescPtr& orgOpDesc)
    {
        return opPtr_;
    }
    OpDescPtr AttrUtils::CopyOpDesc(const ConstOpDescPtr& orgOpDesc)
    {
        return opPtr_;
    }
    ge::ProtoAttrMap& ComputeGraph::MutableAttrMap(){};
    ge::ConstProtoAttrMap& ComputeGraph::GetAttrMap() const{}

    uint32_t ComputeGraph::GetGraphID() const {
        return impl_->GetGraphID();
    }

    uint64_t ComputeGraph::GetSessionID() const {
        return impl_->GetSessionID();
    }

    InDataAnchor::InDataAnchor(const NodePtr& ownerNode, int idx) : DataAnchor(ownerNode, idx)
    {
    }

    graphStatus OutDataAnchor::LinkTo(const InDataAnchorPtr &dest)
    {
        if (dest == nullptr || !dest->impl_->peer_anchors_.empty()) {
            HCCL_INFO("dest anchor is invalid or the peerAnchors is not empty.");
            return GRAPH_FAILED;
        }
        impl_->peer_anchors_.push_back(dest);
        dest->impl_->peer_anchors_.push_back(shared_from_this());
        return GRAPH_SUCCESS;
    }

    graphStatus OutDataAnchor::LinkTo(const InControlAnchorPtr &dest)
    {
        if (dest == nullptr) {
            HCCL_INFO("dest anchor is invalid.");
            return GRAPH_FAILED;
        }
        impl_->peer_anchors_.push_back(dest);
        dest->impl_->peer_anchors_.push_back(shared_from_this());
        return GRAPH_SUCCESS;
    }

    graphStatus OutControlAnchor::LinkTo(const InDataAnchorPtr &dest)
    {
        if (dest == nullptr) {
            HCCL_INFO("dest anchor is invalid.");
            return GRAPH_FAILED;
        }
        impl_->peer_anchors_.push_back(dest);
        dest->impl_->peer_anchors_.push_back(shared_from_this());
        return GRAPH_SUCCESS;
    }

    graphStatus OutControlAnchor::LinkTo(const InControlAnchorPtr &dest)
    {
        if (dest == nullptr) {
            HCCL_INFO("dest anchor is invalid.");
            return GRAPH_FAILED;
        }
        impl_->peer_anchors_.push_back(dest);
        dest->impl_->peer_anchors_.push_back(shared_from_this());
        return GRAPH_SUCCESS;
    }

    graphStatus NodeUtils::GetNodeUnknownShapeStatus(const Node &node, bool &is_unknow)
    {
        is_unknow = false;
        return GRAPH_SUCCESS;
    }

    graphStatus ComputeGraph::TopologicalSorting() {
        return GRAPH_SUCCESS;
    }

    Node::NodeImpl::NodeImpl(const OpDescPtr &op, const ComputeGraphPtr &owner_graph)
        : op_(op),
        owner_graph_(owner_graph),
        in_data_anchors_(),
        out_data_anchors_(),
        in_control_anchor_(nullptr),
        out_control_anchor_(nullptr),
        attrs_(),
        has_init_(false),
        host_node_(false),
        anchor_status_updated_(false) {}

    OpDescImpl::OpDescImpl() {}

    OpDescImpl::OpDescImpl(const std::string &name, const std::string &type) {}

    OpDescImpl::OpDescImpl(const ProtoMsgOwner &proto_msg_owner,
                        ge::proto::OpDef *op_def)
        : op_def_(proto_msg_owner, op_def) {}

    GeTensor::GeTensor() {}
    GeTensor::GeTensor(const GeTensorDesc &tensorDesc, const uint8_t *data, size_t size) {}
    GeTensor::~GeTensor() {}
    TensorData::TensorData() {}
    TensorData::~TensorData() {}
    const TensorData tmpTensorData;
    const GeTensor tmpGeTensor;
    GeTensorPtr tmpGeTensorPtr(new GeTensor());
    std::vector<GeTensorPtr> tmpGeTensorPtrVec;
    std::vector<int64_t> tmpVector{1, 2, 3};
    const TensorData& GeTensor::GetData() const
    {
        return tmpTensorData;
    }
    const GeTensorDesc &GeTensor::GetTensorDesc() const
    {
        GeTensorDesc tenosrDesc;
        return tenosrDesc;
    }

    std::size_t TensorData::GetSize() const
    {
        return tmpVector.size() * sizeof(int64_t);
    }

    const std::uint8_t* TensorData::GetData() const
    {
        return reinterpret_cast<const std::uint8_t*>(tmpVector.data());
    }

    std::uint8_t* TensorData::GetData()
    {
        return reinterpret_cast<std::uint8_t*>(tmpVector.data());
    }

    ConstGeTensorBarePtr OpDescUtils::GetInputConstData(const Operator &op, const uint32_t idx)
    {
        return &tmpGeTensor;
    }

    bool NodeUtils::IsConst(const Node &node)
    {
        return true;
    }

    graphStatus NodeUtils::GetInNodeCrossPartionedCallNode(const NodePtr &node, uint32_t index, NodePtr &peer_node)
    {
        peer_node = const_cast<NodePtr &>(node);
        return GRAPH_SUCCESS;
    }

    std::vector<GeTensorPtr> OpDescUtils::MutableWeights(const ge::NodePtr node)
    {
        tmpGeTensorPtrVec.clear();
        tmpGeTensorPtrVec.push_back(tmpGeTensorPtr);
        return tmpGeTensorPtrVec;
    }

    Operator* tmpOperator = new Operator();
    Operator OpDescUtils::CreateOperatorFromNode(ge::ConstNodePtr node_ptr)
    {
        return *tmpOperator;
    }

    OpDescPtr OpDescUtils::GetOpDescFromOperator(const Operator &oprt)
    {
        return opPtr_;
    }
    bool ge::AttrUtils::GetListTensor(ge::AttrUtils::ConstAttrHolderAdapter&& obj, const string& name, vector<ConstGeTensorPtr>& value)
    {
        std::shared_ptr<const GeTensor> g1 = std::make_shared<const GeTensor>();
        std::shared_ptr<const GeTensor> g2 = std::make_shared<const GeTensor>();
        std::shared_ptr<const GeTensor> g3 = std::make_shared<const GeTensor>();
        std::shared_ptr<const GeTensor> g4 = std::make_shared<const GeTensor>();

        value.push_back(g1);
        value.push_back(g2);
        value.push_back(g3);
        value.push_back(g4);
        return true;
    }

    ge::GEThreadLocalContext &GetThreadLocalContext() {
        static GEThreadLocalContext thread_context_1;
        thread_context_1.options_.reserve(10);
        return thread_context_1;
    }
    // 全局安全缓冲区（只读、静态生命周期、地址固定）
    static const char g_safe_empty_buffer[1] = { '\0' };
    ge::graphStatus ge::GEThreadLocalContext::GetOption(const std::string &optionExec, std::string &dumpDebugValue) {
        if (optionExec == OPTION_EXEC_HCOM_GROUPLIST || optionExec == "ge.exec.rankTable" ||
            optionExec == "ge.offline_hccl_compile") {
            return GRAPH_FAILED;
        } else if (optionExec == "ge.exec.rankMap") {
            dumpDebugValue = R"({"rank_map":[{"logic_rank_id":1,"model_rank_id":0},{"logic_rank_id":2,"model_rank_id":1}]})";
            return GRAPH_SUCCESS;
        } else if (optionExec == "ge.socVersion") {
            dumpDebugValue = "Ascend910";
            return GRAPH_SUCCESS;
        } else if (optionExec == "ge.exec.rankTableFile") {
            dumpDebugValue = "./ut_task_num_one_server_hcom_test.json";
            return GRAPH_SUCCESS;
        }
        dumpDebugValue.push_back('1');
        char addrBuf[32] = {0};
        (void)snprintf(addrBuf, sizeof(addrBuf), "0x%llx", 
                       static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(g_safe_empty_buffer)));
        dumpDebugValue = addrBuf;
        if (options_.count(optionExec) == 1) {
            dumpDebugValue = options_[optionExec];
        }
        return GRAPH_SUCCESS;
    }

    ge::graphStatus ge::GEThreadLocalContext::SetOption(std::string &optionKey, std::string &optionValue) {
        options_[optionKey] = optionValue;
        return GRAPH_SUCCESS;
    }

    ge::graphStatus ge::GEThreadLocalContext::ClearOption()
    {
        options_.clear();
        return GRAPH_SUCCESS;
    }

    ge::GEContext &GetContext() {
        static GEContext context_1;
        context_1.options_.reserve(10);
        return context_1;
    }

    ge::graphStatus ge::GEContext::GetOption(const std::string &optionExec, std::string &dumpDebugValue) {
        return GRAPH_FAILED;
    }

Operator OperatorFactory::CreateOperator(const std::string &operator_name, const std::string &operator_type) {
  return CreateOperator(operator_name, operator_type);
}

Operator OperatorFactory::CreateOperator(const char_t *const operator_name, const char_t *const operator_type) {
  const std::string op_name = operator_name;
  const std::string op_type = operator_type;
  return CreateOperator(op_name, op_type);
}

TensorDesc Operator::GetInputDesc(uint32_t index) const {
    TensorDesc opt;
    return opt;
}

TensorDesc Operator::GetOutputDesc(uint32_t index) const {
    TensorDesc opt;
    return opt;
}

TensorDesc::TensorDesc() {}

void TensorDesc::SetShape(const Shape &shape) {}

void TensorDesc::SetDataType(DataType type) {}

OperatorCreatorRegister::OperatorCreatorRegister(const std::string &operator_type, OpCreator const &op_creator) {}

OperatorCreatorRegister::OperatorCreatorRegister(const char_t *const operator_type, OpCreatorV2 const &op_creator) {}

AscendString::AscendString(const char_t *const name) {
}

GE_FUNC_HOST_VISIBILITY Operator::Operator(const AscendString &name, const AscendString &type) {
}

void Operator::InputRegister(const char_t *name, const char_t *datatype_symbol) {
}
void Operator::OutputRegister(const char_t *name, const char_t *datatype_symbol) {
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ge::OpDescBuilder& OpDescBuilder::AddInput(const std::string &name) {
  inputs_.emplace_back(std::make_pair(name, GeTensorDesc()));
  return *this;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ge::OpDescBuilder& OpDescBuilder::AddOutput(const std::string &name) {
  outputs_.emplace_back(std::make_pair(name, GeTensorDesc()));
  return *this;
}

std::string Node::NodeImpl::GetType() const {
  return op_->GetType();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::string Node::GetType() const {
  return impl_->GetType();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ge::OpDescPtr OpDescBuilder::Build() {
  const OpDescPtr op_desc = MakeShared<OpDesc>(name_, type_);
  for (auto &input : inputs_) {
    if (op_desc->AddInputDesc(input.first, input.second) != GRAPH_SUCCESS) {
      return nullptr;
    }
  }
  for (auto &output : outputs_) {
    if (op_desc->AddOutputDesc(output.first, output.second) != GRAPH_SUCCESS) {
      return nullptr;
    }
  }
  return op_desc;
}

void OpDescImpl::SetId(const int64_t id) {
  meta_data_.id_ = id;
}


GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetId(const int64_t id) {
  impl_->SetId(id);
}

const IRMetaData &OpDescImpl::GetIRMeta() const {
  return meta_data_.ir_meta_;
}

const std::vector<std::pair<std::string, IrInputType>> &ge::IRMetaData::GetIrInputs() const {
  return ir_inputs_;
}
const std::vector<std::pair<std::string, IrInputType>> &OpDesc::GetIrInputs() const {
  return impl_->GetIRMeta().GetIrInputs();
}


const char_t *AscendString::GetString() const {
  if (name_ == nullptr) {
    const static char *empty_value = "";
    return empty_value;
  }

  return (*name_).c_str();
}

bool AscendString::operator<(const AscendString &d) const {
  if ((name_ == nullptr) && (d.name_ == nullptr)) {
    return false;
  } else if (name_ == nullptr) {
    return true;
  } else if (d.name_ == nullptr) {
    return false;
  } else {
    return (*name_) < (*(d.name_));
  }
}

bool AscendString::operator>(const AscendString &d) const {
  if ((name_ == nullptr) && (d.name_ == nullptr)) {
    return false;
  } else if (name_ == nullptr) {
    return false;
  } else if (d.name_ == nullptr) {
    return true;
  } else {
    return (*name_) > (*(d.name_));
  }
}

bool AscendString::operator==(const AscendString &d) const {
  if ((name_ == nullptr) && (d.name_ == nullptr)) {
    return true;
  } else if (name_ == nullptr) {
    return false;
  } else if (d.name_ == nullptr) {
    return false;
  } else {
    return (*name_) == (*(d.name_));
  }
}

bool AscendString::operator<=(const AscendString &d) const {
  if (name_ == nullptr) {
    return true;
  } else if (d.name_ == nullptr) {
    return false;
  } else {
    return (*name_) <= (*(d.name_));
  }
}

bool AscendString::operator>=(const AscendString &d) const {
  if (d.name_ == nullptr) {
    return true;
  } else if (name_ == nullptr) {
    return false;
  } else {
    return (*name_) >= (*(d.name_));
  }
}

bool AscendString::operator!=(const AscendString &d) const {
  if ((name_ == nullptr) && (d.name_ == nullptr)) {
    return false;
  } else if (name_ == nullptr) {
    return true;
  } else if (d.name_ == nullptr) {
    return true;
  } else {
    return (*name_) != (*(d.name_));
  }
}

AnyValue &AnyValue::operator=(AnyValue &&other) noexcept {
  if (&other == this) {
    return *this;
  }
  Clear();
  if (!other.IsEmpty()) {
    other.operate_(OperateType::kOpMove, &other, this);
  }
  return *this;
}
AnyValue &AnyValue::operator=(const AnyValue &other) {
  if (&other == this) {
    return *this;
  }
  Clear();
  if (!other.IsEmpty()) {
    other.operate_(OperateType::kOpClone, &other, this);
  }
  return *this;
}
template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<Anchor>() {
  return reinterpret_cast<TypeId>(1);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<DataAnchor>() {
  return reinterpret_cast<TypeId>(2);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<ControlAnchor>() {
  return reinterpret_cast<TypeId>(3);
}
template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<InDataAnchor>() {
  return reinterpret_cast<TypeId>(4);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<OutDataAnchor>() {
  return reinterpret_cast<TypeId>(5);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<InControlAnchor>() {
  return reinterpret_cast<TypeId>(6);
};

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<OutControlAnchor>() {
  return reinterpret_cast<TypeId>(7);
}
}