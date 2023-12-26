#include "render_wavefront.hpp"

#include "trace_ray.hpp"

using namespace raytracer;

using sycl::float2;
using sycl::float3;
using sycl::float4;
using sycl::int2;
using sycl::range;

WavefrontRenderer::WavefrontRenderer(
    App &app, sycl::range<2> img_size, sycl::image<2> &output_image
)
    : app(app), img_size(img_size),
      image(sycl::image_channel_order::rgba, sycl::image_channel_type::fp32, img_size),
      combined_image(sycl::image_channel_order::rgba, sycl::image_channel_type::fp32, img_size),
      buffers({Buffers(app, img_size), Buffers(app, img_size)}),
      output_image(output_image) {
    app.queue
        .submit([&](sycl::handler &cgh) {
            auto image_writer =
                this->image.get_access<sycl::float4, sycl::access::mode::write>(cgh);
            auto combined_image_writer =
                this->image.get_access<sycl::float4, sycl::access::mode::write>(cgh);
            cgh.parallel_for(sycl::range<2>(img_size), [=](sycl::item<2> item) {
                image_writer.write(sycl::int2(item[0], item[1]), sycl::float4(0.0f));
                combined_image_writer.write(sycl::int2(item[0], item[1]), sycl::float4(0.0f));
            });
        })
        .wait();
}

void WavefrontRenderer::generate_camera_rays(const Camera &camera, uint32_t sample) {
    app.queue
        .submit([&](sycl::handler &cgh) {
            auto image_writer =
                this->image.get_access<sycl::float4, sycl::access::mode::write>(cgh);

            auto produced_rays_acc =
                this->current_buffer()
                    .ray_buffer_length.get_access<sycl::access_mode::read_write>(cgh);

            range<2> local_size{8, 8};
            range<2> n_groups = {
                ((img_size[0] + local_size[0] - 1) / local_size[0]),
                ((img_size[1] + local_size[1] - 1) / local_size[1]),
            };
            sycl::nd_range<2> for_range(n_groups * local_size, local_size);

            auto img_size = this->img_size;
            auto ray_buffer = this->current_buffer().ray_buffer;

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
                    produced_rays_ref(produced_rays_acc[0]);

                int2 pixel_coords = {global_id[0], global_id[1]};

                image_writer.write(pixel_coords, sycl::float4(0.0f));

                auto init_generator_state =
                    std::hash<std::size_t>{}(id.get_global_linear_id() + 33469 * sample);
                auto rng = XorShift32State{(uint32_t)init_generator_state};
                RTCRay ray = camera.get_ray(pixel_coords, rng);

                uint32_t ray_index = produced_rays_ref.fetch_add(1);
                RayData ray_data = {
                    .ray = ray,
                    .attenuation = float3(1.0f),
                    .radiance = float3(0.0f),
                    .rng = rng,
                };
                ray_buffer[ray_index] = ray_data;
            });
        })
        .wait();
}

void WavefrontRenderer::shoot_rays(
    const Camera &camera, const Scene &scene, uint32_t depth
) {
    uint32_t prev_ray_count = this->prev_buffer().ray_buffer_length.get_host_access()[0];
    fmt::println("Shooting {} rays", prev_ray_count);
    this->prev_buffer().ray_buffer_length.get_host_access()[0] = 0;

    app.queue
        .submit([&](sycl::handler &cgh) {
            if (prev_ray_count == 0) {
                return;
            }

            range<1> local_size = 32;
            range<1> n_groups = ((prev_ray_count + local_size - 1) / local_size);
            sycl::nd_range<1> for_range(n_groups * local_size, local_size);

            auto produced_rays_acc =
                this->current_buffer()
                    .ray_buffer_length.get_access<sycl::access_mode::read_write>(cgh);
            auto image_reader =
                this->image.get_access<float4, sycl::access::mode::read>(cgh);
            auto image_writer =
                this->image.get_access<float4, sycl::access::mode::write>(cgh);

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
                .os = sycl::stream(8192, 256, cgh),
            };

            const auto prev_ray_buffer = this->prev_buffer().ray_buffer;
            const auto new_ray_buffer = this->current_buffer().ray_buffer;

            cgh.parallel_for(for_range, [=](sycl::nd_item<1> id) {
                auto global_id = id.get_global_id();
                if (global_id >= prev_ray_count) {
                    return;
                }

                sycl::atomic_ref<
                    uint32_t,
                    sycl::memory_order_relaxed,
                    sycl::memory_scope_device,
                    sycl::access::address_space::global_space>
                    produced_rays_acc_ref(produced_rays_acc[0]);

                RayData ray_data = prev_ray_buffer[global_id];

                auto res = trace_ray(
                    ctx,
                    ray_data.rng,
                    ray_data.ray,
                    ray_data.attenuation,
                    ray_data.radiance
                );

                sycl::int2 pixel_coords = {
                    ray_data.ray.id % ctx.camera.img_size[0],
                    ray_data.ray.id / ctx.camera.img_size[0]};

                if (res) {
                    // Final value is computed. Write to image.
                    float4 final_color = float4(sycl::clamp(*res, 0.0f, 1.0f), 1.0f);

                    float4 prev_value = image_reader.read(pixel_coords);
                    // float4 final_value =
                    //     sycl::mix(prev_value, final_color, 1.0f / ((float)depth
                    //     + 1.0f));

                    float4 final_value = final_color;

                    image_writer.write(pixel_coords, final_value);

                    return;
                }

                if (depth == (MAX_DEPTH - 1)) {
                    image_writer.write(pixel_coords, float4(0.0f, 0.0f, 0.0f, 1.0f));
                    return;
                }

                // New ray was generated
                uint32_t ray_index = produced_rays_acc_ref.fetch_add(1);
                new_ray_buffer[ray_index] = ray_data;
            });
        })
        .wait();
}

void WavefrontRenderer::merge_samples(uint32_t sample) {
    app.queue
        .submit([&](sycl::handler &cgh) {
            sycl::stream os(8192, 256, cgh);

            auto image_reader =
                this->image.get_access<float4, sycl::access::mode::read>(cgh);
            auto combined_image_reader =
                this->combined_image.get_access<float4, sycl::access::mode::read>(cgh);
            auto combined_image_writer =
                this->combined_image.get_access<float4, sycl::access::mode::write>(cgh);

            range<2> local_size{8, 8};
            range<2> n_groups = {
                ((img_size[0] + local_size[0] - 1) / local_size[0]),
                ((img_size[1] + local_size[1] - 1) / local_size[1]),
            };

            const auto img_size = this->img_size;

            cgh.parallel_for(
                sycl::nd_range<2>(n_groups * local_size, local_size),
                [=](sycl::nd_item<2> id) {
                    auto global_id = id.get_global_id();
                    if (global_id[0] >= img_size[0] || global_id[1] >= img_size[1]) {
                        return;
                    }

                    int2 pixel_coords = {global_id[0], global_id[1]};

                    float4 img_val = image_reader.read(pixel_coords);

                    float4 combined_val = combined_image_reader.read(pixel_coords);

                    // float4 final_value =
                    //     sycl::mix(output_val, img_val, 1.0f / ((float)sample + 1.0f));

                    combined_image_writer.write(pixel_coords, combined_val + img_val);
                }
            );
        })
        .wait();
}

void WavefrontRenderer::convert_image_to_srgb() {
    app.queue
        .submit([&](sycl::handler &cgh) {
            sycl::stream os(8192, 256, cgh);

            auto combined_image_reader =
                this->combined_image.get_access<float4, sycl::access::mode::read>(cgh);
            auto output_image_writer =
                this->output_image.get_access<float4, sycl::access::mode::write>(cgh);

            range<2> local_size{8, 8};
            range<2> n_groups = {
                ((img_size[0] + local_size[0] - 1) / local_size[0]),
                ((img_size[1] + local_size[1] - 1) / local_size[1]),
            };

            const auto img_size = this->img_size;

            cgh.parallel_for(
                sycl::nd_range<2>(n_groups * local_size, local_size),
                [=](sycl::nd_item<2> id) {
                    auto global_id = id.get_global_id();
                    if (global_id[0] >= img_size[0] || global_id[1] >= img_size[1]) {
                        return;
                    }

                    int2 pixel_coords = {global_id[0], global_id[1]};

                    float4 img_val = combined_image_reader.read(pixel_coords) / (float)SAMPLE_COUNT;
                    output_image_writer.write(pixel_coords, linear_to_gamma(img_val));
                }
            );
        })
        .wait();
}

void WavefrontRenderer::render_frame(const Camera &camera, const Scene &scene) {
    auto begin = std::chrono::high_resolution_clock::now();

    uint32_t total_ray_count = 0;

    for (uint32_t sample = 0; sample < SAMPLE_COUNT; sample++) {
        fmt::println("Sample {}", sample);

        this->generate_camera_rays(camera, sample);

        for (uint32_t depth = 0; depth < MAX_DEPTH; depth++) {
            total_ray_count +=
                this->current_buffer().ray_buffer_length.get_host_access()[0];

            buffer_index++;

            this->shoot_rays(camera, scene, depth);
        }

        this->merge_samples(sample);
    }

    this->convert_image_to_srgb();

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    double secs = elapsed.count() * 1e-9;
    double rays_per_sec = (double)total_ray_count / secs;

    fmt::println("Time measured: {:.6f} seconds", secs);
    fmt::println("Total rays: {}", total_ray_count);
    fmt::println("Rays/sec: {:.2f}M", rays_per_sec / 1000000.0);

    fmt::println("Writing image to disk");
    write_image(app.queue, output_image, img_size[0], img_size[1]);
}
