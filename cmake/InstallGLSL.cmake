file (GLOB_RECURSE resources "glsl_shader/*.*")
foreach(resource ${resources})
    get_filename_component(filename ${resource} NAME)
    set(output "${CMAKE_BINARY_DIR}/glsl_shader/${filename}")
    add_custom_command(
        COMMENT "Moving updated resource-file '${filename}'"
        OUTPUT ${output}
        DEPENDS ${resource}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${resource}
        ${output}
    )
    add_custom_target(${filename} ALL DEPENDS ${resource} ${output})
endforeach()
