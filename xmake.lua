set_project("tensr")
set_version("0.0.0")
set_xmakever("2.8.0")

set_languages("c11", "cxx17")
set_warnings("all")
set_optimize("fastest")

add_rules("mode.debug", "mode.release")

option("cuda")
    set_default(false)
    set_showmenu(true)
    set_description("Enable CUDA support")
option_end()

target("tensr")
    set_kind("static")
    add_files("src/core/tensor.c", "src/core/array.c", "src/core/tensor.cpp")
    add_files("src/ops/*.c")
    add_files("src/linalg/*.c")
    add_files("src/random/*.c")
    add_files("src/io/*.c")
    add_files("src/fft/*.c")
    add_files("src/backend/*.c")
    
    if has_config("cuda") then
        add_files("src/backend/cuda/*.cu")
        add_cugencodes("native")
        add_cuflags("-use_fast_math", "-O3")
    end
    
    add_includedirs("include", {public = true})
    add_headerfiles("include/(**.h)", "include/(**.hpp)")
    
    set_targetdir("build/lib")
    set_objectdir("build/obj")

target("tests")
    set_kind("binary")
    add_deps("tensr")
    add_files("tests/test_tensor.c")
    add_includedirs("include")
    set_targetdir("build/bin")
    set_rundir("$(projectdir)")

target("example_c")
    set_kind("binary")
    add_deps("tensr")
    add_files("examples/basic_usage.c")
    add_includedirs("include")
    set_targetdir("build/bin")

target("example_cpp")
    set_kind("binary")
    add_deps("tensr")
    add_files("examples/cpp_usage.cpp")
    add_includedirs("include")
    set_targetdir("build/bin")
