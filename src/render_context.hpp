#pragma once

#include <sycl/sycl.hpp>
#include <embree4/rtcore.h>
#include "camera.hpp"
#include "image_manager.hpp"

namespace raytracer {

struct RenderContext {
    Camera camera;
    sycl::float3 sky_color;

    RTCScene scene;

    sycl::sampler sampler;
    ImageReadAccessor image_reader;

    mutable sycl::stream os;
};

} // namespace raytracer
