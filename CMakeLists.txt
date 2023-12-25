cmake_minimum_required (VERSION 3.27)
project(Raytracer)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(EMBREE_SYCL_SUPPORT ON)
set(FMT_MODULE OFF)

find_package(IntelSYCL REQUIRED)
find_package(embree 4 REQUIRED)

add_library(fmt deps/fmt/format.cc deps/fmt/os.cc)
target_include_directories(fmt PUBLIC deps/include)

add_library(stb deps/stb_image.cpp deps/stb_image_write.cpp deps/stb_image_resize2.cpp)
target_include_directories(stb PUBLIC deps/include)

add_library(tinygltf deps/tiny_gltf.cpp)
target_include_directories(tinygltf PUBLIC deps/include)
target_link_libraries(tinygltf PRIVATE stb)

set(
    SYCL_SOURCES
    src/main.cpp
    src/render.cpp
    src/scene.cpp
)

set(
    SOURCES
    ${SYCL_SOURCES}
)

add_executable(raytracer ${SOURCES})
target_link_libraries(
    raytracer
    PRIVATE
    fmt
    stb
    tinygltf
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

# Scratchpad

add_executable(demo demo.cpp)
target_compile_definitions(demo PUBLIC EMBREE_SYCL_SUPPORT)
target_link_libraries(demo PRIVATE fmt embree)
target_include_directories(demo PRIVATE src deps/include)
add_sycl_to_target(
    TARGET demo
    SOURCES demo.cpp
)
