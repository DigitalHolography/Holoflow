# This function is used to set common target properties. This is used for all
# targets in the project.
function(set_common_target_properties target)
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED YES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endfunction()

# This function is used to set common compile options. This is used for all
# targets in the project.
function(set_common_compile_options target)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            $<$<CXX_COMPILER_ID:MSVC>:/W4;/permissive-;/WX;/wd4456;/wd4505;/wd5046;/wd4702;/utf-8;/Zc:preprocessor>
        >
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    )
endfunction()
