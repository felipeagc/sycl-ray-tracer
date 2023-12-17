#pragma once

#include <sycl/sycl.hpp>
#include <embree4/rtcore.h>
#include "xorshift.hpp"

namespace raytracer {

struct Camera {
    sycl::range<2> img_size;

    float focal_length;
    sycl::float3 center;
    sycl::float3 up;
    sycl::float3 right;

    sycl::float3 pixel00_loc;
    sycl::float3 pixel_delta_u;
    sycl::float3 pixel_delta_v;

    static constexpr int max_samples = 100;
    static constexpr int max_bounces = 100;

    Camera(sycl::range<2> img_size, sycl::float3 center, sycl::float3 look_at)
        : img_size(img_size), center(center) {
        using sycl::float2;
        using sycl::float3;

        // Setup camera
        this->focal_length = 1.0f;

        auto dir = normalize(look_at - center);
        std::cout << "camera direction: " << dir.x() << ", " << dir.y() << ", " << dir.z()
                  << std::endl;
        auto world_up = float3(0, 1, 0);
        this->right = normalize(cross(dir, world_up));
        std::cout << "camera right: " << right.x() << ", " << right.y() << ", "
                  << right.z() << std::endl;
        this->up = normalize(cross(right, dir));
        std::cout << "camera up: " << up.x() << ", " << up.y() << ", " << up.z()
                  << std::endl;

        float2 viewport = float2(1.0f * ((float)img_size[0] / (float)img_size[1]), 1.0f);

        // Calculate the vectors across the horizontal and down the vertical
        // viewport edges.
        float3 viewport_u = (-this->right) * viewport[0];
        float3 viewport_v = this->up * viewport[1];
        std::cout << "viewport_u: " << viewport_u.x() << ", " << viewport_u.y() << ", "
                  << viewport_u.z() << std::endl;
        std::cout << "viewport_v: " << viewport_v.x() << ", " << viewport_v.y() << ", "
                  << viewport_v.z() << std::endl;

        this->pixel00_loc =
            this->center + viewport_u + viewport_v + dir * this->focal_length;
        std::cout << "pixel00_loc: " << pixel00_loc.x() << ", " << pixel00_loc.y() << ", "
                  << pixel00_loc.z() << std::endl;

        this->pixel_delta_u = this->right / ((float)img_size[0] / (viewport[0] * 2.0f));
        this->pixel_delta_v = this->up / ((float)img_size[1] / (viewport[1] * 2.0f));

        std::cout << "pixel_delta_u: " << pixel_delta_u.x() << ", " << pixel_delta_u.y()
                  << ", " << pixel_delta_u.z() << std::endl;
        std::cout << "pixel_delta_v: " << pixel_delta_v.x() << ", " << pixel_delta_v.y()
                  << ", " << pixel_delta_v.z() << std::endl;

        RTCRay test_ray = this->get_ray(1920, 1080);
        std::cout << "test_ray orig: " << test_ray.org_x << ", " << test_ray.org_y << ", "
                  << test_ray.org_z << std::endl;
        test_ray.org_x += test_ray.dir_x;
        test_ray.org_y += test_ray.dir_y;
        test_ray.org_z += test_ray.dir_z;
        std::cout << "test_ray dest: " << test_ray.org_x << ", " << test_ray.org_y << ", "
                  << test_ray.org_z << std::endl;
    }

    // Get a randomly sampled camera ray for the pixel at location x,y.
    RTCRay get_ray(
        int x, int y
        /* , XorShift32State &rng */
    ) const {
        auto pixel_center =
            pixel00_loc + ((float)x * pixel_delta_u) - ((float)y * pixel_delta_v);
        auto pixel_sample = pixel_center;
        /* auto pixel_sample = pixel_center + pixel_sample_square(rng); */

        auto ray_origin = this->center;
        auto ray_direction = pixel_sample - ray_origin;

        return RTCRay{
            .org_x = ray_origin[0],
            .org_y = ray_origin[1],
            .org_z = ray_origin[2],
            .tnear = 0,

            .dir_x = ray_direction[0],
            .dir_y = ray_direction[1],
            .dir_z = ray_direction[2],
            .time = 0,

            .tfar = std::numeric_limits<float>::infinity(),
            .mask = UINT32_MAX,
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
