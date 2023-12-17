#include "app.hpp"
#include "model.hpp"
#include "util.hpp"
#include "camera.hpp"

#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>
#include <embree4/rtcore.h>
#include <glm/glm.hpp>

using sycl::float2;
using sycl::float3;
using sycl::float4;
using sycl::int2;
using sycl::range;

using namespace raytracer;

static float3 render_pixel(
    const Camera &camera,
    XorShift32State &rng,
    RTCScene scene,
    int x,
    int y,
    uint32_t *out_ray_count,
    sycl::stream os
) {
    RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);

    RTCRayHit rayhit;
    rayhit.ray = camera.get_ray(x, y);
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

    rtcIntersect1(scene, &rayhit, &args);

    (*out_ray_count)++;

    if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
        return float3(1, 0, 0);
    }
    return float3(0, 0, 0);
}

void render_frame(App &app, range<2> img_size, sycl::image<2> &image) {
    std::vector<Model> models;
    models.emplace_back(app, "../assets/cube.glb");
    Scene scene(app, models);

    uint32_t initial_ray_count = 0;
    sycl::buffer<uint32_t> ray_count_buffer{&initial_ray_count, 1};

    auto begin = std::chrono::high_resolution_clock::now();

    auto e = app.queue.submit([&](sycl::handler &cgh) {
        sycl::stream os(8192, 256, cgh);
        RTCScene r_scene = scene.scene;

        auto image_writer = image.get_access<float4, sycl::access::mode::write>(cgh);
        auto ray_count = ray_count_buffer.get_access<sycl::access_mode::write>(cgh);

        range<2> local_size{8, 8};
        range<2> n_groups = img_size;
        n_groups[0] = ((img_size[0] + local_size[0] - 1) / local_size[0]);
        n_groups[1] = ((img_size[1] + local_size[1] - 1) / local_size[1]);

        Camera camera(img_size, float3(3, 0, 0), float3(0, 0, 0));

        cgh.parallel_for(
            sycl::nd_range<2>(n_groups * local_size, local_size),
            [=](sycl::nd_item<2> id) {
                auto global_id = id.get_global_id();
                if (global_id[0] >= img_size[0] || global_id[1] >= img_size[1]) {
                    return;
                }

                sycl::atomic_ref<
                    uint32_t,
                    sycl::memory_order_relaxed,
                    sycl::memory_scope_device,
                    sycl::access::address_space::global_space>
                    ray_count_ref(ray_count[0]);

                int x = global_id[0];
                int y = global_id[1];

                auto init_generator_state =
                    std::hash<std::size_t>{}(id.get_global_linear_id());
                auto rng = XorShift32State{(uint32_t)init_generator_state};

                uint32_t ray_count = 0;
                float3 pixel_color = render_pixel(camera, rng, r_scene, x, y, &ray_count, os);

                image_writer.write(
                    int2(x, y),
                    float4(pixel_color.x(), pixel_color.y(), pixel_color.z(), 1.0)
                );

                ray_count_ref.fetch_add(ray_count);
            }
        );
    });

    e.wait_and_throw();

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    auto ray_count = ray_count_buffer.get_host_access();

    double secs = elapsed.count() * 1e-9;
    double rays_per_sec = (double)ray_count[0] / secs;

    printf("Time measured: %.6lf seconds\n", secs);
    printf("Total rays: %u\n", ray_count[0]);
    printf("Rays/sec: %.2lfM\n", rays_per_sec / 1000000.0);

    printf("Writing image to disk\n");
    write_image(app.queue, image, img_size[0], img_size[1]);
}

int main(int argc, char *argv[]) {
    try {
        App app;

        // Calculate viewport size
        range<2> img_size = range<2>(1920, 1080);

        // Create image
        uint8_t *image_buf =
            sycl::malloc_shared<uint8_t>(img_size[0] * img_size[1] * 4, app.queue);
        sycl::image<2> image(
            image_buf,
            sycl::image_channel_order::rgba,
            sycl::image_channel_type::unorm_int8,
            img_size
        );

        render_frame(app, img_size, image);
    } catch (sycl::exception const &e) {
        std::cout << "Caught SYCL exception: " << e.what() << "\n";
        std::terminate();
    }

    return 0;
}
