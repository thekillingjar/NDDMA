add_library(op_compile_adapter_headers INTERFACE)
target_include_directories(op_compile_adapter_headers INTERFACE
    $<BUILD_INTERFACE:${AIR_CODE_DIR}/compiler/opcompiler/inc>
    $<BUILD_INTERFACE:${AIR_CODE_DIR}/compiler/opcompiler/inc/tensor_engine>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/opcompiler>
    $<INSTALL_INTERFACE:include/opcompiler/tensor_engine>
)

file(GLOB OP_COMPILE_ADAPTER_SRC_CPP RELATIVE ${CMAKE_CURRENT_LIST_DIR}
     "common/*.cc"
     "cache/*.cc"
     "binary/*.cc"
     "compile/*.cc"
     "file_handle/*.cc"
     "dfxinfo_manager/*.cc"
     "assemble_json/*.cc"
     "python_adapter/*.cc"
     "api/*.cc"
    )

add_library(op_compile_adapter SHARED
    ${OP_COMPILE_ADAPTER_SRC_CPP}
    )

target_include_directories(op_compile_adapter PRIVATE
  ${HI_PYTHON_INC}
  ${CMAKE_CURRENT_LIST_DIR}
  ${AIR_CODE_DIR}/compiler/engines/nn_engine/utils/
  )

target_compile_options(op_compile_adapter PRIVATE
  -Werror -O2 -fPIC
  -fno-common
  -fno-strict-aliasing
  )

target_compile_definitions(op_compile_adapter PRIVATE
      google=ascend_private
    )

target_link_libraries(op_compile_adapter
    PRIVATE
        intf_pub
        metadef_headers
        msprof_headers
        runtime_headers
        graph
        register
        static_mmpa
        unified_dlog
        c_sec
        json
        error_manager
        atrace_share
        aicore_utils
    PUBLIC
        op_compile_adapter_headers
)

if (BUILD_OPEN_PROJECT OR ENABLE_OPEN_SRC)
    install(TARGETS op_compile_adapter OPTIONAL
            EXPORT op_compile_adapter-targets
            LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
            )
endif()