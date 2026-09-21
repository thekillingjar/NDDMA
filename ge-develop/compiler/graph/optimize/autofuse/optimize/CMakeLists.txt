set(OPTIMIZE_DIR ${CODE_ROOT_DIR}/optimize)
file(GLOB_RECURSE SOURCES "${OPTIMIZE_DIR}/*.cpp")

target_sources(aihac_codegen PRIVATE
    ${SOURCES}
)

target_include_directories(aihac_codegen PRIVATE
    ${TOP_DIR}/runtime/include/external
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CODE_ROOT_DIR}
    ${CODE_ROOT_DIR}/autofuse
    ${CODE_ROOT_DIR}/common
)
