#pragma once

#include <sycl/sycl.hpp>
#include <embree4/rtcore.h>
#include <glm/glm.hpp>
#include <fmt/core.h>

#include "xorshift.hpp"

namespace raytracer {

struct RayData {
    float org_x;
    float org_y;
    float org_z;
    uint32_t id;

    sycl::half dir_x;
    sycl::half dir_y;
    sycl::half dir_z;

    sycl::half att_r;
    sycl::half att_g;
    sycl::half att_b;

    sycl::half rad_r;
    sycl::half rad_g;
    sycl::half rad_b;

    RayData(uint32_t ray_id, sycl::float3 origin, sycl::float3 dir) {
        this->id = ray_id;
        this->org_x = origin[0];
        this->org_y = origin[1];
        this->org_z = origin[2];
        this->dir_x = sycl::half(dir[0]);
        this->dir_y = sycl::half(dir[1]);
        this->dir_z = sycl::half(dir[2]);
        this->att_r = sycl::half(1.0f);
        this->att_g = sycl::half(1.0f);
        this->att_b = sycl::half(1.0f);
        this->rad_r = sycl::half(0.0f);
        this->rad_g = sycl::half(0.0f);
        this->rad_b = sycl::half(0.0f);
    }

    inline RTCRay to_embree() const {
        RTCRay ray = {
            .org_x = this->org_x,
            .org_y = this->org_y,
            .org_z = this->org_z,
            .tnear = 0.0001f,
            .dir_x = this->dir_x,
            .dir_y = this->dir_y,
            .dir_z = this->dir_z,
            .time = 0.0f,
            .tfar = std::numeric_limits<float>::infinity(),
            .mask = UINT32_MAX,
            .id = this->id,
            .flags = 0,
        };
        return ray;
    }
};

struct Camera {
    sycl::float3 center;

    sycl::float3 pixel00_loc;
    sycl::float3 pixel_delta_u;
    sycl::float3 pixel_delta_v;

    sycl::int2 img_size;

    Camera(
        sycl::range<2> img_size,
        glm::vec3 cam_center,
        glm::vec3 cam_dir,
        float focal_length
    ) {
        using sycl::float2;
        using sycl::float3;

        this->img_size[0] = img_size[0];
        this->img_size[1] = img_size[1];

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
    RayData get_ray(sycl::int2 pixel_coords, XorShift32State &rng) const {
        int x = pixel_coords[0];
        int y = pixel_coords[1];

        auto pixel_center =
            pixel00_loc + ((float)x * pixel_delta_u) + ((float)y * pixel_delta_v);
        auto pixel_sample = pixel_center + pixel_sample_square(rng);

        auto ray_origin = this->center;
        auto ray_direction = pixel_sample - ray_origin;

        uint32_t ray_id =
            (uint32_t)(pixel_coords.x() + (pixel_coords.y() * this->img_size[0]));
        return RayData(ray_id, ray_origin, ray_direction);
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
