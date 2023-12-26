#include "render_wavefront.hpp"

#include "trace_ray.hpp"

using namespace raytracer;

using sycl::float2;
using sycl::float3;
using sycl::float4;
using sycl::int2;
using sycl::range;

using RayBufferLengthAccessor =
    sycl::accessor<uint32_t, 1, sycl::access_mode::write, sycl::access::target::device>;

static inline void generate_camera_rays(
    sycl::handler &cgh,
    const Camera &camera,
    sycl::range<2> img_size,
    sycl::nd_range<2> for_range,
    RTCRay *ray_buffer,
    RayBufferLengthAccessor &ray_buffer_length_acc
) {
    cgh.parallel_for(for_range, [=](sycl::nd_item<2> id) {
        auto global_id = id.get_global_id();
        if (global_id[0] >= img_size[0] || global_id[1] >= img_size[1]) {
            return;
        }

        sycl::atomic_ref<
            uint32_t,
            sycl::memory_order_relaxed,
            sycl::memory_scope_device,
            sycl::access::address_space::global_space>
            ray_buffer_length_ref(ray_buffer_length_acc[0]);

        int2 pixel_coords = {global_id[0], global_id[1]};

        // TODO: save RNG state
        XorShift32State rng = {0};
        uint32_t ray_index = ray_buffer_length_ref.fetch_add(1);
        ray_buffer[ray_index] = camera.get_ray(pixel_coords, rng);
    });
}

static inline void shoot_rays(
    sycl::handler &cgh,
    const Camera &camera,
    const Scene &scene,
    sycl::stream os,
    uint32_t ray_count,
    sycl::nd_range<1> for_range,
    RTCRay *ray_buffer,
    sycl::buffer<uint32_t> &ray_buffer_length,
    sycl::image<2> &output_image
) {
    auto ray_buffer_length_acc =
        ray_buffer_length.get_access<sycl::access_mode::write>(cgh);
    auto image_writer = output_image.get_access<float4, sycl::access::mode::write>(cgh);

    RenderContext ctx = {
        .camera = camera,
        .sky_color = scene.sky_color,
        .scene = scene.scene,
        .sampler = sycl::sampler(
            sycl::coordinate_normalization_mode::normalized,
            sycl::addressing_mode::repeat,
            sycl::filtering_mode::nearest
        ),
        .image_reader = ImageReadAccessor(scene.image_array.value(), cgh),
        .os = os,
    };

    cgh.parallel_for(for_range, [=](sycl::nd_item<1> id) {
        auto global_id = id.get_global_id();
        if (global_id >= ray_count) {
            return;
        }

        sycl::atomic_ref<
            uint32_t,
            sycl::memory_order_relaxed,
            sycl::memory_scope_device,
            sycl::access::address_space::global_space>
            ray_buffer_length_ref(ray_buffer_length_acc[0]);

        RTCRay ray = ray_buffer[global_id];

        // TODO: use saved values
        float3 attenuation = float3(1.0f);
        float3 radiance = float3(0.0f);

        // TODO: save RNG state
        XorShift32State rng = {0};
        auto res = trace_ray(ctx, rng, ray, attenuation, radiance);

        if (res) {
            // Final value is computed. Write to image.
            sycl::int2 pixel_coords = {
                ray.id % ctx.camera.img_size[0], ray.id / ctx.camera.img_size[0]};
            sycl::float3 final_color = *res;

            image_writer.write(pixel_coords, float4(final_color, 1.0f));
        } else {
            // Ray was absorbed. Generate new ray.
            uint32_t ray_index = ray_buffer_length_ref.fetch_add(1);
            ray_buffer[ray_index] = ray;
        }
    });
}

WavefrontRenderer::WavefrontRenderer(App &app) : app(app) {}

void WavefrontRenderer::render_frame(
    const Camera &camera,
    const Scene &scene,
    sycl::range<2> img_size,
    sycl::image<2> &image
) {
    uint32_t zero = 0;
    sycl::buffer<uint32_t> total_ray_count_buffer{&zero, 1};

    sycl::buffer<uint32_t> ray_buffer_length{&zero, 1};
    RTCRay *ray_buffer = (RTCRay *)sycl::aligned_alloc_device(
        alignof(RTCRay), sizeof(RTCRay) * img_size.size(), app.queue
    );

    auto begin = std::chrono::high_resolution_clock::now();

    app.queue
        .submit([&](sycl::handler &cgh) {
            auto ray_buffer_length_acc =
                ray_buffer_length.get_access<sycl::access_mode::write>(cgh);

            range<2> local_size{8, 8};
            range<2> n_groups = {
                ((img_size[0] + local_size[0] - 1) / local_size[0]),
                ((img_size[1] + local_size[1] - 1) / local_size[1]),
            };
            sycl::nd_range<2> for_range(n_groups * local_size, local_size);

            generate_camera_rays(
                cgh, camera, img_size, for_range, ray_buffer, ray_buffer_length_acc
            );
        })
        .wait();

    for (uint32_t i = 0; i < MAX_DEPTH; i++) {
        uint32_t ray_count = ray_buffer_length.get_host_access()[0];
        ray_buffer_length.get_host_access()[0] = 0;

        app.queue
            .submit([&](sycl::handler &cgh) {
                sycl::stream os(8192, 256, cgh);

                range<1> local_size = 32;
                range<1> n_groups = ((ray_count + local_size - 1) / local_size);
                sycl::nd_range<1> for_range(n_groups * local_size, local_size);

                shoot_rays(
                    cgh,
                    camera,
                    scene,
                    os,
                    ray_count,
                    for_range,
                    ray_buffer,
                    ray_buffer_length,
                    image
                );
            })
            .wait();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    auto ray_count = total_ray_count_buffer.get_host_access();

    double secs = elapsed.count() * 1e-9;
    double rays_per_sec = (double)ray_count[0] / secs;

    fmt::println("Time measured: {:.6f} seconds", secs);
    fmt::println("Total rays: {}", ray_count[0]);
    fmt::println("Rays/sec: {:.2f}M", rays_per_sec / 1000000.0);

    fmt::println("Writing image to disk");
    write_image(app.queue, image, img_size[0], img_size[1]);

    sycl::free(ray_buffer, app.queue);
}
