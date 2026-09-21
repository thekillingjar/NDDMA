# 简化版本，使用 echo 和 cat 组合
function(generate_raw_string_header TARGET_NAME OUTPUT_FILE)
    # 解析参数
    set(INPUT_FILES ${ARGN})
    
    # 检查是否提供了起始和结束标记
    list(LENGTH INPUT_FILES NUM_ARGS)
    set(START_PATTERN "")
    set(END_PATTERN "")
    set(REAL_INPUT_FILES "")
    
    if(NUM_ARGS GREATER 1)
        # 最后两个参数可能是 START_PATTERN 和 END_PATTERN
        list(GET INPUT_FILES -1 LAST_ARG)
        list(GET INPUT_FILES -2 SECOND_LAST_ARG)
        
        # 检查这两个参数是否看起来像标记（不是文件路径）
        if(NOT LAST_ARG MATCHES "\\.h$" AND NOT LAST_ARG MATCHES "\\.cpp$")
            set(END_PATTERN ${LAST_ARG})
            list(REMOVE_AT INPUT_FILES -1)
        endif()
        
        if(NOT SECOND_LAST_ARG MATCHES "\\.h$" AND NOT SECOND_LAST_ARG MATCHES "\\.cpp$")
            set(START_PATTERN ${SECOND_LAST_ARG})
            list(REMOVE_AT INPUT_FILES -1)
        endif()
        
        set(REAL_INPUT_FILES ${INPUT_FILES})
    else()
        set(REAL_INPUT_FILES ${INPUT_FILES})
    endif()

    # 构建命令
    if(START_PATTERN STREQUAL "" AND END_PATTERN STREQUAL "")
        # 没有指定标记，读取整个文件
        set(EXTRACT_CMD "cat ${REAL_INPUT_FILES}")
    else()
        # 提取两个标记之间的内容（不包含标记行）
        if(START_PATTERN STREQUAL "")
            set(START_PATTERN "^")
        endif()
        if(END_PATTERN STREQUAL "")
            set(END_PATTERN "$")
        endif()
        # 使用 awk 提取，不包含起始和结束标记行
        set(EXTRACT_CMD "awk '/${START_PATTERN}/{flag=1;next} /${END_PATTERN}/{if(flag) exit} flag' ${REAL_INPUT_FILES}")
    endif()

    # 创建自定义命令
    add_custom_command(
            OUTPUT ${OUTPUT_FILE}
            COMMAND bash -c "echo 'R\"===(' > ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE} && ${EXTRACT_CMD} >> ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE} && echo ')===\"' >> ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE}"
            DEPENDS ${REAL_INPUT_FILES}
            COMMENT "Generating ${OUTPUT_FILE} from ${REAL_INPUT_FILES}"
            VERBATIM
    )

    # 创建自定义目标
    add_custom_target(${TARGET_NAME}_text ALL
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE}
    )

    # 创建接口库
    add_library(${TARGET_NAME} INTERFACE)

    # 设置接口库的包含目录
    target_include_directories(${TARGET_NAME} INTERFACE
            ${CMAKE_CURRENT_BINARY_DIR}
            ..
    )

    # 添加依赖关系
    add_dependencies(${TARGET_NAME} ${TARGET_NAME}_text)
endfunction()