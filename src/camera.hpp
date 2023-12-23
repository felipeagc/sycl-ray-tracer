#pragma once

#include <sycl/sycl.hpp>
#include <embree4/rtcore.h>
#include <glm/glm.hpp>
#include <fmt/core.h>

#include "xorshift.hpp"

namespace raytracer {

struct Camera {
    sycl::float3 center;

    sycl::float3 pixel00_loc;
    sycl::float3 pixel_delta_u;
    sycl::float3 pixel_delta_v;

    static constexpr int max_samples = 100;
    static constexpr int max_bounces = 100;

    Camera(sycl::range<2> img_size, glm::vec3 cam_center, glm::vec3 cam_dir, float focal_length) {
        using sycl::float2;
        using sycl::float3;

        this->center[0] = cam_center.x;
        this->center[1] = cam_center.y;
        this->center[2] = cam_center.z;

        float3 dir = normalize(float3(cam_dir.x, cam_dir.y, cam_dir.z));

        // Setup camera
        float3 world_up = float3(0, 1, 0);
        float3 right = normalize(cross(dir, world_up));
        float3 up = normalize(cross(right, dir));

        float2 viewport = float2(1.0f * ((float)img_size[0] / (float)img_size[1]), 1.0f);

        float3 viewport_u = (-right) * viewport[0];
        float3 viewport_v = up * viewport[1];

        this->pixel00_loc = this->center + viewport_u + viewport_v + dir * focal_length;

        this->pixel_delta_u = right / ((float)img_size[0] / (viewport[0] * 2.0f));
        this->pixel_delta_v = -up / ((float)img_size[1] / (viewport[1] * 2.0f));
    }

    // Get a randomly sampled camera ray for the pixel at location x,y.
    RTCRay get_ray(int x, int y, XorShift32State &rng) const {
        auto pixel_center =
            pixel00_loc + ((float)x * pixel_delta_u) + ((float)y * pixel_delta_v);
        auto pixel_sample = pixel_center + pixel_sample_square(rng);

        auto ray_origin = this->center;
        auto ray_direction = pixel_sample - ray_origin;

        return RTCRay{
            .org_x = ray_origin[0],
            .org_y = ray_origin[1],
            .org_z = ray_origin[2],
            .tnear = 0.001f,

            .dir_x = ray_direction[0],
            .dir_y = ray_direction[1],
            .dir_z = ray_direction[2],
            .time = 0,

            .tfar = std::numeric_limits<float>::infinity(),
            .mask = UINT32_MAX,
            .id = 0,
            .flags = 0,
        };
    }

    // Returns a random point in the square surrounding a pixel at the
    // origin.
    sycl::float3 pixel_sample_square(XorShift32State &rng) const {
        float px = -0.5f + rng();
        float py = -0.5f + rng();
        return (px * pixel_delta_u) + (py * pixel_delta_v);
    }
};

} // namespace raytracer
