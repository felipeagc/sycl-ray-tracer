cmake_minimum_required (VERSION 3.27)
project(Raytracer)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(EMBREE_SYCL_SUPPORT ON)
set(FMT_MODULE OFF)

find_package(IntelSYCL REQUIRED)
find_package(embree 4 REQUIRED)

set(
    SYCL_SOURCES
    src/main.cpp
)

set(
    SOURCES
    ${SYCL_SOURCES}
    deps/stb_image.cpp
    deps/stb_image_write.cpp
    deps/tiny_gltf.cpp
    deps/fmt/format.cc
    deps/fmt/os.cc
    deps/tiny_gltf.cpp
)

add_executable(raytracer ${SOURCES})
target_link_libraries(
    raytracer
    PRIVATE
    embree
)
target_include_directories(
    raytracer
    PRIVATE
    deps/include
)
target_compile_definitions(raytracer PUBLIC EMBREE_SYCL_SUPPORT)
add_sycl_to_target(
    TARGET raytracer
    SOURCES ${SYCL_SOURCES}
)
