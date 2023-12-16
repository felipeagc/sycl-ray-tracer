cmake_minimum_required (VERSION 3.27)
project(Raytracer)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(EMBREE_SYCL_SUPPORT ON)

find_package(IntelSYCL REQUIRED)
find_package(embree 4 REQUIRED)
# find_package(TBB REQUIRED)

set(
    SOURCES
    src/main.cpp
    src/stb_image_write.cpp
    src/tiny_obj_loader.cpp
    src/tiny_gltf.cpp
)

add_executable(raytracer ${SOURCES})
target_link_libraries(
    raytracer
    PRIVATE
    embree
    # TBB::tbb
)
target_compile_definitions(raytracer PUBLIC EMBREE_SYCL_SUPPORT)
add_sycl_to_target(TARGET raytracer SOURCES ${SOURCES})
