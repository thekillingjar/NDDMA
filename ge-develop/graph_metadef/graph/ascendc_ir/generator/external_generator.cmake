function(ascir_generate depend_so_target out_dir so_var h_var)
    add_custom_command(
            OUTPUT ${h_var}
            DEPENDS ${depend_so_target}
            COMMAND ${out_dir}/ascir_ops_header_generator ${so_var} ${h_var}
    )
endfunction()
