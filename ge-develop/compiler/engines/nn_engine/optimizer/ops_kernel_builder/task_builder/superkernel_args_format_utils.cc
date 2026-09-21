/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "superkernel_args_format_utils.h"
#include "common/fe_log.h"
#include "platform/platform_info.h"

namespace fe {
namespace {
const std::string kSuperKernelType = "SuperKernel";
const size_t MIN_SPK_SIZE = 1;
constexpr uint32_t WORKSPACE_SIZE = 8;

static bool IsNeedFFTS() { // check is need ffts addr
    static const bool is_need_ffts = [] () -> bool {
        fe::PlatFormInfos platform_info;
        fe::OptionalInfos optional_info;
        // init platform
        bool init_platform_success = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(
                                                                         platform_info, optional_info) == SUCCESS;
        FE_CHECK(!init_platform_success, FE_LOGE("[SuperKernel] init platform failed!"), return false);

        std::string ffts_mode_str;
        if (platform_info.GetPlatformResWithLock("SoCInfo", "ffts_mode", ffts_mode_str)) {
            FE_LOGI("[SuperKernel] GenTaskForSuperKernel ffts_mode is %s", ffts_mode_str.c_str());
            if (ffts_mode_str == "ffts-plus") {
                return true;
            }
            return false;
        }
        FE_LOGI("[SuperKernel] current soc do not have ffts_mode in GenTaskForSuperKernel!");
        return false;
    } ();
    return is_need_ffts;
}
}

ge::Status GetWorkspacePattern(const ge::Node &node, std::string &super_kernel_args_format, int64_t ws_size) {
    int64_t buffer_size = 0;
    FE_LOGI("GetWorkspacePattern get ws_size is %d.", ws_size);
    // 找到DFX的对应node，并追加ws0
    if (ge::AttrUtils::GetInt(node.GetOpDesc(), "_op_dfx_buffer_size", buffer_size)) {
        buffer_size += ws_size; // dfx wks + syncall wks
        vector<int64_t> buffer_size_vec = {buffer_size};
        (void)ge::AttrUtils::SetListInt(node.GetOpDesc(), "_append_ws", buffer_size_vec);
        std::string wks_pattern = "{ws0*}";
        super_kernel_args_format += wks_pattern;
        FE_LOGI("super arg format is %s.", super_kernel_args_format.c_str());
        return ge::SUCCESS;
    } else {
        if (ws_size > 0) {
            vector<int64_t> buffer_size_vec = {ws_size};
            (void)ge::AttrUtils::SetListInt(node.GetOpDesc(), "_append_ws", buffer_size_vec);
            std::string wks_pattern = "{ws0*}";
            super_kernel_args_format += wks_pattern;
            FE_LOGI("superkernel has syncall only no dfx");
            FE_LOGI("super arg format is %s", super_kernel_args_format.c_str());
            return ge::SUCCESS;
        }
    }
    FE_LOGW("can not find dfx node in subkernel");
    return ge::FAILED;
}

ge::Status GetArgFormatV2(domi::TaskDef &task_temp, std::string &args_format) {
    args_format = "";
    if (task_temp.type() == RT_MODEL_TASK_KERNEL || (task_temp.type() == RT_MODEL_TASK_PREPROCESS_KERNEL)) {
        auto kernel_def = task_temp.mutable_kernel();
        FE_CHECK_NOTNULL(kernel_def);
        auto kernel_context = kernel_def->mutable_context();
        FE_CHECK_NOTNULL(kernel_context);
        args_format = kernel_context->args_format();
    } else if(task_temp.type() == RT_MODEL_TASK_ALL_KERNEL) {
        auto kernel_def_with_handle = task_temp.mutable_kernel_with_handle();
        FE_CHECK_NOTNULL(kernel_def_with_handle);
        auto kernel_context_with_handle = kernel_def_with_handle->mutable_context();
        FE_CHECK_NOTNULL(kernel_context_with_handle);
        args_format = kernel_context_with_handle->args_format();
    } else {
        return ge::SUCCESS;
    }
    FE_LOGI("GetArgFormat: %s ", args_format.c_str());
    return ge::SUCCESS;
}

ge::Status GetAICoreArgFormatinLoop(std::string &sub_arg_format, const std::vector<ge::Node *> &sub_nodes, uint32_t &args_size,
                          std::string &super_kernel_args_format, const ge::NodePtr &shared_node, const ge::NodePtr &sub_node, uint32_t cnt) {
    // 拼aicoreTask argformat，如果是superkernel双流，argformat后面追加软同步地址
    std::vector<ge::ArgDesc> arg_descs;
    ge::ArgsFormatDescUtils::Parse(sub_arg_format, arg_descs);
    std::vector<uint32_t> sk_send_event_ids;
    (void)ge::AttrUtils::GetListInt(sub_nodes[cnt]->GetOpDesc(), "_sk_send_event_ids", sk_send_event_ids);
    std::vector<uint32_t> sk_rcv_event_ids;
    (void)ge::AttrUtils::GetListInt(sub_nodes[cnt]->GetOpDesc(), "_sk_rcv_event_ids", sk_rcv_event_ids);
    uint32_t byte_size = 8;
    args_size += (sk_send_event_ids.size() + sk_rcv_event_ids.size()) * byte_size;
    for (const auto &sk_send_id : sk_send_event_ids) {
        ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::EVENT_ADDR, sk_send_id);
    }
    for (const auto &sk_recv_id : sk_rcv_event_ids) {
        ge::ArgsFormatDescUtils::Append(arg_descs, ge::AddrType::EVENT_ADDR, sk_recv_id);
    }
    sub_arg_format = ge::ArgsFormatDescUtils::ToString(arg_descs);
    auto ret = ge::ArgsFormatDesc::ConvertToSuperKernelArgFormat(
        shared_node, sub_node, sub_arg_format, super_kernel_args_format);
    if (ret != ge::SUCCESS) {
        return ge::FAILED;
    }
    return ge::SUCCESS;
}

bool IsAICpuKernelType(ge::ccKernelType kernel_type) {
    return (kernel_type == ge::ccKernelType::AI_CPU) || (kernel_type == ge::ccKernelType::CUST_AI_CPU) ||
        (kernel_type == ge::ccKernelType::HOST_CPU || (kernel_type == ge::ccKernelType::AI_CPU_KFC));
}

bool KernelLaunch(const std::string &stub_func, const uint32_t block_dim, const void *args,
                                uint32_t args_size, const rtSmDesc_t *sm_desc, domi::TaskDef &task_def) {
    task_def.set_type(static_cast<uint32_t>(RT_MODEL_TASK_KERNEL));
    domi::KernelDef *kernel_def = task_def.mutable_kernel();
    if (kernel_def == nullptr) {
        FE_LOGE("[GenTask][KernelLaunch] kernel_def is nullptr.");
        return false;
    }

    FE_LOGD("KernelLaunch", "[GenTask][KernelLaunch] stub_func_name is [%s]", stub_func.c_str());
    kernel_def->set_stub_func(stub_func);
    if (sm_desc != nullptr) {
        uintptr_t sm_desc_data = reinterpret_cast<uintptr_t>(sm_desc);
        uint8_t *sm_desc_ptr = reinterpret_cast<uint8_t *>(sm_desc_data);
        kernel_def->set_sm_desc(sm_desc_ptr, sizeof(rtSmDesc_t));
    }

    kernel_def->set_block_dim(block_dim);
    kernel_def->set_args_size(args_size);
    kernel_def->set_args(args, args_size);

    domi::KernelContext *kernel_context = kernel_def->mutable_context();
    if (kernel_context == nullptr) {
        REPORT_INNER_ERR_MSG("E19999", "[GenTask][KernelLaunch] kernel_context is nullptr.");
        FE_LOGE("[GenTask][KernelLaunch] kernel_context is nullptr");
        return false;
    }
    uint32_t super_args_count = 1;
    kernel_context->set_args_count(super_args_count);
    uint16_t super_args_offset[super_args_count];
    super_args_offset[0] = 0;
    kernel_context->set_args_offset(
        static_cast<void*>(super_args_offset), static_cast<size_t>(super_args_count * sizeof(uint16_t)));
    return true;
}

uint64_t GetAtomicStubFuncId() {
    static std::atomic<uint64_t> global_cmo_id(1);
    return global_cmo_id.fetch_add(1, std::memory_order_relaxed);
}

ge::Status SetArgFormatValue(uint32_t args_size_workspace, std::vector<std::vector<domi::TaskDef>> &subTasks,
                         const std::vector<ge::Node *> &sub_nodes, void *all_args_buff_total,
                         uint32_t args_size_total) {
    // 拷贝args
    bool is_need_ffts = IsNeedFFTS();
    int32_t args_size_cur = is_need_ffts ? WORKSPACE_SIZE : 0;
    domi::KernelContext *kernel_context = nullptr;
    domi::KernelDef *kernel_def_tmp = nullptr;
    for (uint32_t i = 0; i < subTasks.size(); ++i) {
        const auto sub_kernel_op_desc = sub_nodes[i]->GetOpDesc();
        for (auto &single_task : subTasks[i]) {
            if (IsAICpuTaskDef(single_task, kernel_context)) {
                FE_LOGI( "aicpu task args_format copy.");
                continue;
            }
            uint32_t args_size = 0;
            if (single_task.type() == static_cast<uint32_t>(RT_MODEL_TASK_KERNEL)) {
                kernel_def_tmp = single_task.mutable_kernel();
                FE_CHECK_NOTNULL(kernel_def_tmp);
                args_size = kernel_def_tmp->args_size();
                FE_LOGI( "task_arg.type is RT_MODEL_TASK_KERNEL args_size: %d %d", args_size, __LINE__);
            } else {
                FE_LOGW( "The Task type [%u] is invalid.", single_task.type());
                // notify wait task
                continue;
            }
            FE_LOGI( "RT_MODEL_TASK_KERNEL sec_ret %d", __LINE__);
            uint8_t sec_ret = 1;
            FE_LOGI( "RT_MODEL_TASK_KERNEL BEGIN MEMCPY");
            if (args_size_total == 0) {
                FE_LOGI( "Skip the RT_MODEL_TASK_KERNEL memcpy procedure for args_size_total is 0.");
                continue;
            }
            sec_ret = memcpy_s((uint8_t*)all_args_buff_total + args_size_cur, static_cast<size_t>(args_size_total),
                                    kernel_def_tmp->args().data(), static_cast<size_t>(args_size));
            if (sec_ret != 0) {
                FE_LOGE( "memcpy_s is fail");
                return FAILED;
            }
            FE_LOGI( "RT_MODEL_TASK_KERNEL AFTER MEMCPY");
            args_size_cur += args_size;
            args_size_total -= args_size;
            std::vector<uint32_t> sk_send_event_ids;
            (void)ge::AttrUtils::GetListInt(sub_nodes[i]->GetOpDesc(), "_sk_send_event_ids", sk_send_event_ids);
            std::vector<uint32_t> sk_rcv_event_ids;
            (void)ge::AttrUtils::GetListInt(sub_nodes[i]->GetOpDesc(), "_sk_rcv_event_ids", sk_rcv_event_ids);
            auto addr_size = (sk_send_event_ids.size() + sk_rcv_event_ids.size()) * 8;
            if (addr_size > 0) {
                char* addr_buffer = new (std::nothrow) char[addr_size]();
                if (!addr_buffer) {
                    return FAILED;
                }
                sec_ret = memcpy_s((uint8_t *)all_args_buff_total + args_size_cur, static_cast<size_t>(args_size_total),
                    addr_buffer, addr_size);
                if (sec_ret != 0) {
                    FE_LOGE( "memcpy_s addr_buffer is fail");
                    delete[] addr_buffer;
                    return FAILED;
                }
                args_size_cur += addr_size;
                delete[] addr_buffer;
            }
        }
    }
    if (is_need_ffts) {
        char tmp_buf[8] = {0x00};
        if (memcpy_s((uint8_t *)all_args_buff_total + args_size_cur, args_size_total, tmp_buf, args_size_workspace) != 0) {
            FE_LOGE( "memcpy_s ffts_addr is fail");
            return ge::FAILED;
        }
    }
    return ge::SUCCESS;
}

ge::Status FillTaskDefAfterGenTask(const ge::OpDescPtr &op_desc, domi::TaskDef &task_def,
                                            std::string &args_format) {
    std::string attr_val_kernel_name;
    (void)ge::AttrUtils::GetStr(op_desc, "_kernelname", attr_val_kernel_name);
    uint32_t schedule_mode = 0;
    (void)ge::AttrUtils::GetInt(op_desc, "_soft_sync_schedule_mode", schedule_mode);
    FE_LOGD("FillTaskDefAfterGenTask", "Set schedule mode[%u] on task of op[%s, %s]. ", schedule_mode, op_desc->GetNamePtr(), op_desc->GetTypePtr());
    domi::KernelContext *kernel_context = nullptr;
    if (task_def.type() == RT_MODEL_TASK_KERNEL) {
        domi::KernelDef *kernel_def = task_def.mutable_kernel();
        FE_CHECK_NOTNULL(kernel_def);
        kernel_def->set_kernel_name(attr_val_kernel_name);
        FE_LOGD("Set Kernel_name[%s] for the kernel_def of node[%s, %s].",
                attr_val_kernel_name.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
        
        kernel_def->set_schedule_mode(schedule_mode);
        kernel_context = kernel_def->mutable_context();
    }

    FE_CHECK_NOTNULL(kernel_context);
    uint32_t op_index = static_cast<uint32_t>(op_desc->GetId());
    uint32_t kernelType = 2;
    kernel_context->set_args_format(args_format);
    kernel_context->add_origin_op_index(op_index);
    kernel_context->set_op_index(op_index);
    kernel_context->set_kernel_type(kernelType);
    kernel_context->set_is_flowtable(false);
    return ge::SUCCESS;
}

void CheckDFXOpen(uint32_t &args_size_workspace, const ge::Node &node, std::string &super_kernel_args_format, int64_t ws_size) {
    auto ret = GetWorkspacePattern(node, super_kernel_args_format, ws_size);
    if (ret == ge::SUCCESS) {
        args_size_workspace = WORKSPACE_SIZE; // workspace_addr 8
    }
}

int64_t GetSuperKernelWorkspace(const ge::Node &node) {
    auto super_ws = node.GetOpDesc()->GetWorkspace();
    auto super_ws_bytes = node.GetOpDesc()->GetWorkspaceBytes();
    int64_t super_ws_size = 0;

    size_t workspace_num = super_ws_bytes.size();
    FE_LOGI( "workspace_num is :%zu", workspace_num);
    if (workspace_num > 0U) {
        super_ws_size = super_ws_bytes[0];
        FE_LOGI( "super_ws_size is :%zu", super_ws_size);
    }

    if (super_ws.empty()) {
        FE_LOGI( "clear super_ws_bytes node name is :%s", node.GetNamePtr());
        super_ws.clear();
        super_ws_bytes.clear();

        node.GetOpDesc()->SetWorkspace(super_ws);
        node.GetOpDesc()->SetWorkspaceBytes(super_ws_bytes);
    }
    return super_ws_size;
}

bool IsAICpuTaskDef(domi::TaskDef &task_temp, domi::KernelContext *&kernel_context) {
    if (task_temp.type() == RT_MODEL_TASK_PREPROCESS_KERNEL) {
        auto kernel_def = task_temp.mutable_kernel();
        FE_CHECK(kernel_def == nullptr, FE_LOGW("IsAICpuTaskDef kernel_def is null pointer!"), return false);
        kernel_context = kernel_def->mutable_context();
    } else {
        FE_LOGI("not tiling sink task");
        return false;
    }
    FE_CHECK(kernel_context == nullptr, FE_LOGW("IsAICpuTaskDef kernel_context is null pointer!"), return false);
    const auto kernel_type = static_cast<ge::ccKernelType>(kernel_context->kernel_type());
    if (IsAICpuKernelType(kernel_type)) {
        FE_LOGI("Is aicpu task");
        return true;
    }
    return false;
}

ge::Status GetArgFormat(const std::vector<ge::Node *> &sub_nodes, uint32_t &args_size_total, std::vector<std::vector<domi::TaskDef>> &subTasks,
                     const ge::OpDescPtr &super_kernel_op_desc, std::vector<domi::TaskDef> &tasks, const ge::NodePtr &shared_node,
                     std::string &super_kernel_args_format) {
    domi::KernelContext *kernel_context = nullptr;
    domi::KernelDef *kernel_def_tmp = nullptr;
    // 如果是双流，每个aicore task argformat后面追加软同步地址
    for (uint32_t i = 0; i < subTasks.size(); ++i) {
        const auto sub_kernel_op_desc = sub_nodes[i]->GetOpDesc();
        std::string sub_arg_format;
        auto sub_node = const_cast<ge::Node *>(sub_nodes[i])->shared_from_this();
        auto &subTaskVec = subTasks[i];
        for (auto &single_task : subTaskVec) {
            GetArgFormatV2(single_task, sub_arg_format);
            if (sub_arg_format.empty()) {
                continue;
            }
            if (IsAICpuTaskDef(single_task, kernel_context)) {
                // 外提至和superkernel一个层级，不拷贝argformat，修改op_index
                kernel_context->set_op_index(super_kernel_op_desc->GetId());
                std::string aicpu_kernel_args_format;
                auto ret = ge::ArgsFormatDesc::ConvertToSuperKernelArgFormat(
                    shared_node, sub_node, sub_arg_format, aicpu_kernel_args_format);
                if (ret != SUCCESS) {
                    FE_LOGE("aicpu Task convertToSuperKernelArgFormat failed.");
                    return ge::FAILED;
                }
                kernel_context->set_args_format(aicpu_kernel_args_format);
                tasks.emplace_back(single_task);
            } else {
                uint32_t args_size;
                if (single_task.type() == static_cast<uint32_t>(RT_MODEL_TASK_KERNEL)) {
                    kernel_def_tmp = single_task.mutable_kernel();
                    FE_CHECK_NOTNULL(kernel_def_tmp);
                    args_size = kernel_def_tmp->args_size();
                    FE_LOGI( "task_arg.type is RT_MODEL_TASK_KERNEL args_size: %d %d", args_size, __LINE__);
                } else if (single_task.type() == static_cast<uint32_t>(RT_MODEL_TASK_ALL_KERNEL)) {
                    auto kernel_def_with_handle = single_task.mutable_kernel_with_handle();
                    FE_CHECK_NOTNULL(kernel_def_with_handle);
                    args_size = kernel_def_with_handle->args_size();
                    FE_LOGI( "task_arg.type is RT_MODEL_TASK_ALL_KERNEL args_size: %d %d", args_size, __LINE__);
                } else {
                    FE_LOGE( "The task type[%u] is invalid.", single_task.type());
                    continue;
                }
                if (GetAICoreArgFormatinLoop(sub_arg_format, sub_nodes, args_size,
                          super_kernel_args_format, shared_node, sub_node, i) != ge::SUCCESS) {
                    FE_LOGE( "GetAICoreArgFormatinLoop failed.");
                    return ge::FAILED;
                }
                FE_LOGI( "task_arg addr_len size:%d", args_size);
                args_size_total += args_size;
            }
        }
    }
    return ge::SUCCESS;
}

std::string GetUniqueGraphIdForNode(const ge::OpDescPtr &super_kernel_op_desc) {
    string session_graph_id = "";
    string atomic_id = std::to_string(GetAtomicStubFuncId());
    if (ge::AttrUtils::GetStr(super_kernel_op_desc, "_session_graph_id", session_graph_id) && !session_graph_id.empty()) {
        return atomic_id + "_" + session_graph_id + "_" + super_kernel_op_desc->GetName();
    } else {
        return atomic_id + "_" + super_kernel_op_desc->GetName();
    }
}

ge::Status GenTaskForSuperKernel(const ge::Node &node, std::vector<std::vector<domi::TaskDef>> &subTasks,
            const std::vector<ge::Node *> &sub_nodes, std::vector<domi::TaskDef> &tasks) {
    FE_LOGI( "GenTaskForSuperKernel begin");
    std::string super_kernel_args_format;
    const auto super_kernel_op_desc = node.GetOpDesc();
    auto shared_node = const_cast<ge::Node *>(&node)->shared_from_this();
    if (sub_nodes.size() < MIN_SPK_SIZE) {
        FE_LOGE( "sub_nodes:%zu is invalid", sub_nodes.size());
        return ge::FAILED;
    }
    int64_t super_ws_size = GetSuperKernelWorkspace(node);
    FE_LOGI( "super_ws_size is :%zu", super_ws_size);

    bool is_need_ffts = IsNeedFFTS();

    uint32_t args_size_total = 0;
    if (is_need_ffts) {
        std::string super_kernel_first_addr = "{ffts_addr}";
        auto first_node = const_cast<ge::Node *>(sub_nodes[0])->shared_from_this();
        (void)ge::ArgsFormatDesc::ConvertToSuperKernelArgFormat(
                shared_node, first_node, super_kernel_first_addr, super_kernel_args_format);
        FE_LOGI( "GenTaskForSuperKernel get ffts format");
        args_size_total += WORKSPACE_SIZE;  // size of ffts addr
    }

    auto ret = GetArgFormat(sub_nodes, args_size_total, subTasks, super_kernel_op_desc,\
                            tasks, shared_node, super_kernel_args_format);
    if (ret != ge::SUCCESS) {
        return ge::FAILED;
    }

    FE_LOGI("GenTaskForSuperKernel GetArgFormat success");
    FE_LOGI("Node[%s, %s] full argsformat is [%s].", super_kernel_op_desc->GetNamePtr(),
                super_kernel_op_desc->GetTypePtr(), super_kernel_args_format.c_str());
    
    uint32_t args_size_workspace = 0;
    CheckDFXOpen(args_size_workspace, node, super_kernel_args_format, super_ws_size);
    args_size_total += args_size_workspace;

    FE_LOGI("GenTaskForSuperKernel finish DFX stage");
    FE_LOGI("GenTaskForSuperKernel args_size_total is:%d", args_size_total);
    void *all_args_buff_total = (void *)malloc(args_size_total);
    if (all_args_buff_total == nullptr) {
        FE_LOGE("all_args_buff_total is nullptr args_size:%d", args_size_total);
        return ge::FAILED;
    }

    if (is_need_ffts) {
        char tmp_buf[8] = {0x00};
        size_t tmp_buf_size = 8;
        if (memcpy_s((uint8_t *)all_args_buff_total, args_size_total, tmp_buf, tmp_buf_size) != 0) {
            FE_LOGE("memcpy first addr failed!");
            free(all_args_buff_total);
            return ge::FAILED;
        }
        FE_LOGI("GenTaskForSuperKernel copy ffts args finished");
    }

    std::vector<domi::TaskDef> all_tasks;
    for (auto &task : subTasks) {
        std::vector<domi::TaskDef> task_temps(task.begin(), task.end());
        for (auto &task_temp : task_temps) {
            all_tasks.insert(all_tasks.end(), task_temp);
        }
    }
    if (all_tasks.size() < MIN_SPK_SIZE) {
        FE_LOGE("all tasks size: %zu is invalid", all_tasks.size());
        free(all_args_buff_total);
        return ge::FAILED;
    }
    FE_LOGI("GenTaskForSuperKernel get superkernel task success");
    
    domi::TaskDef superkernel_task_def = {};
    superkernel_task_def.set_stream_id(node.GetOpDesc()->GetStreamId());
    FE_LOGI("set %s stream_id %u", node.GetNamePtr(), superkernel_task_def.stream_id());

    if (SetArgFormatValue(args_size_workspace, subTasks, sub_nodes, all_args_buff_total,\
                        args_size_total) != ge::SUCCESS) {
        free(all_args_buff_total);
        FE_LOGE("SetArgFormatValue failed.");
        return ge::FAILED;
    }
    FE_LOGI("GenTaskForSuperKernel SetArgFormatValue finished");

    // Get block dim
    int32_t block_dim = -1;
    if (ge::AttrUtils::GetInt(super_kernel_op_desc, ge::TVM_ATTR_NAME_BLOCKDIM, block_dim)) {
        block_dim = static_cast<uint32_t>(block_dim);
        FE_LOGD("Get block dim[%d] of node[%s, %s]",
                block_dim, super_kernel_op_desc->GetNamePtr(), super_kernel_op_desc->GetTypePtr());
    }
    std::string stub_func = GetUniqueGraphIdForNode(super_kernel_op_desc);
    FE_LOGI("GenTaskForSuperKernel GetUniqueGraphIdForNode finished");
    FE_LOGD("Generate stub func[%s] of node[%s, %s]",
                stub_func.c_str(), super_kernel_op_desc->GetNamePtr(), super_kernel_op_desc->GetTypePtr());
    bool launchRel = KernelLaunch(stub_func, block_dim, all_args_buff_total, args_size_total, nullptr,
         superkernel_task_def);
    if (!launchRel) {
        FE_LOGE( "Node[%s, %s] fill kernel def failed.", super_kernel_op_desc->GetNamePtr(),
                super_kernel_op_desc->GetTypePtr());
        free(all_args_buff_total);
        return ge::FAILED;
    }
    FE_LOGI("GenTaskForSuperKernel launch success");
    FE_LOGI("Node[%s, %s] ready to fill taskdef with argsformat [%s].", super_kernel_op_desc->GetNamePtr(),
                super_kernel_op_desc->GetTypePtr(), super_kernel_args_format.c_str());
    if (FillTaskDefAfterGenTask(super_kernel_op_desc, superkernel_task_def, 
        super_kernel_args_format) != ge::SUCCESS) {
        FE_LOGE( "Node[%s, %s] fill super kernel task def fail, argsformat is [%s].", super_kernel_op_desc->GetNamePtr(),
                    super_kernel_op_desc->GetTypePtr(), super_kernel_args_format.c_str());
        free(all_args_buff_total);
        return ge::FAILED;
    }
    FE_LOGI("GenTaskForSuperKernel fill super kernel task def finished");
    superkernel_task_def.set_type(RT_MODEL_TASK_SUPER_KERNEL);
    tasks.emplace_back(superkernel_task_def);
    FE_LOGI("set superkernel type success.");
    free(all_args_buff_total);
    FE_LOGI("gen task success. ");
    return ge::SUCCESS;
}
}