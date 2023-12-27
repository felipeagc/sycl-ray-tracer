#pragma once

#include <sycl/sycl.hpp>
#include <embree4/rtcore.h>
#include "camera.hpp"
#include "image_manager.hpp"

#define USE_STREAMS 0

namespace raytracer {

struct RenderContext {
    Camera camera;
    sycl::float3 sky_color;

    RTCScene scene;

    sycl::sampler sampler;
    ImageReadAccessor image_reader;

#if USE_STREAMS
    mutable sycl::stream os;
#endif
};

} // namespace raytracer
