#include "render_wavefront.hpp"

#include "trace_ray.hpp"

using namespace raytracer;

using sycl::float2;
using sycl::float3;
using sycl::float4;
using sycl::int2;
using sycl::range;

struct ScopedRng {
    ScopedRng(int2 pixel_coords, sycl::range<2> img_size, XorShift32State *rng_buffer) {
        this->rng_ptr = &rng_buffer[pixel_coords[0] + (pixel_coords[1] * img_size[0])];
        this->rng = *this->rng_ptr;
    }

    ~ScopedRng() {
        *this->rng_ptr = this->rng;
    }

    operator XorShift32State &() {
        return rng;
    }

  private:
    XorShift32State rng;
    XorShift32State *rng_ptr;
};

WavefrontRenderer::WavefrontRenderer(
    App &app,
    sycl::range<2> img_size,
    sycl::image<2> &output_image,
    uint32_t max_depth,
    uint32_t sample_count
)
    : app(app), img_size(img_size),
      image(sycl::image_channel_order::rgba, sycl::image_channel_type::fp32, img_size),
      combined_image(
          sycl::image_channel_order::rgba, sycl::image_channel_type::fp32, img_size
      ),
      buffers({Buffers(app, img_size), Buffers(app, img_size)}),
      output_image(output_image), max_depth(max_depth), sample_count(sample_count) {
    this->rng_buffer = (XorShift32State *)sycl::aligned_alloc_device(
        alignof(XorShift32State), sizeof(XorShift32State) * img_size.size(), app.queue
    );

    app.queue
        .submit([&](sycl::handler &cgh) {
            auto image_writer =
                this->image.get_access<sycl::float4, sycl::access::mode::write>(cgh);
            auto combined_image_writer =
                this->image.get_access<sycl::float4, sycl::access::mode::write>(cgh);

            auto rng_buffer = this->rng_buffer;
            auto img_size = this->img_size;

            cgh.parallel_for(sycl::range<2>(img_size), [=](sycl::item<2> item) {
                sycl::int2 pixel_coords(item[0], item[1]);

                image_writer.write(pixel_coords, sycl::float4(0.0f));
                combined_image_writer.write(pixel_coords, sycl::float4(0.0f));

                // Initialize RNG state
                uint32_t pixel_linear_pos =
                    pixel_coords[0] + (pixel_coords[1] * img_size[0]);
                auto init_generator_state = std::hash<std::size_t>{}(pixel_linear_pos);
                rng_buffer[pixel_linear_pos] =
                    XorShift32State{(uint32_t)init_generator_state};
            });
        })
        .wait();
}

void WavefrontRenderer::generate_camera_rays(const Camera &camera, uint32_t sample) {
    app.queue
        .submit([&](sycl::handler &cgh) {
            // Group size / range
            range<2> local_size{16, 16};
            range<2> n_groups = {
                ((img_size[0] + local_size[0] - 1) / local_size[0]),
                ((img_size[1] + local_size[1] - 1) / local_size[1]),
            };
            sycl::nd_range<2> for_range(n_groups * local_size, local_size);

            // Accessors
            auto image_writer =
                this->image.get_access<sycl::float4, sycl::access::mode::write>(cgh);

            // Params
            auto img_size = this->img_size;
            auto ray_buffer = this->current_buffer().ray_buffer;
            uint32_t *global_ray_count = this->current_buffer().ray_buffer_length;
            auto rng_buffer = this->rng_buffer;

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
                    global_ray_count_ref(*global_ray_count);

                int2 pixel_coords = {global_id[0], global_id[1]};

                image_writer.write(pixel_coords, sycl::float4(0.0f));

                ScopedRng rng(pixel_coords, img_size, rng_buffer);

                RayData ray = camera.get_ray(pixel_coords, rng);
                uint32_t ray_index = global_ray_count_ref.fetch_add(1);
                ray_buffer[ray_index] = ray;
            });
        })
        .wait();
}

static void print_elapsed(
    const std::chrono::high_resolution_clock::time_point &begin, const char *phase_name
) {
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    double msecs = elapsed.count() * 1e-6;

    fmt::println("\tPhase {}: {:.6f}ms", phase_name, msecs);
}

void WavefrontRenderer::shoot_rays(
    const Camera &camera, const Scene &scene, uint32_t depth
) {
    auto begin = std::chrono::high_resolution_clock::now();

    uint32_t prev_ray_count = *this->prev_buffer().ray_buffer_length;
    // fmt::println("Shooting {} rays", prev_ray_count);
    *this->prev_buffer().ray_buffer_length = 0;

    // print_elapsed(begin, "reset ray count");

    app.queue
        .submit([&](sycl::handler &cgh) {
            if (prev_ray_count == 0) {
                return;
            }

            // Group size / range
            range<1> local_size = 32;
            range<1> n_groups = ((prev_ray_count + local_size - 1) / local_size);
            sycl::nd_range<1> for_range(n_groups * local_size, local_size);

            // Accessors
            sycl::local_accessor<uint32_t, 1> local_ray_count_accessor(
                sycl::range<1>(1), cgh
            );
            sycl::local_accessor<uint32_t, 1> local_first_ray_index_accessor(
                sycl::range<1>(1), cgh
            );
            sycl::local_accessor<RayData, 1> local_ray_data(
                sycl::range<1>(local_size), cgh
            );
            auto image_writer =
                this->image.get_access<float4, sycl::access::mode::write>(cgh);

            // Params
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
#if USE_STREAMS
                .os = sycl::stream(8192, 256, cgh),
#endif
            };
            const auto prev_ray_buffer = this->prev_buffer().ray_buffer;

            const auto new_ray_buffer = this->current_buffer().ray_buffer;
            uint32_t *global_ray_count = this->current_buffer().ray_buffer_length;

            const uint32_t max_depth = this->max_depth;
            const range<2> img_size = this->img_size;
            XorShift32State *rng_buffer = this->rng_buffer;

            // print_elapsed(begin, "parallel_for begin");
            cgh.parallel_for(for_range, [=](sycl::nd_item<1> id) {
                sycl::atomic_ref<
                    uint32_t,
                    sycl::memory_order_relaxed,
                    sycl::memory_scope_device,
                    sycl::access::address_space::global_space>
                    global_ray_count_ref(*global_ray_count);

                sycl::atomic_ref<
                    uint32_t,
                    sycl::memory_order_relaxed,
                    sycl::memory_scope_device,
                    sycl::access::address_space::local_space>
                    local_ray_count(local_ray_count_accessor[0]);

                auto global_id = id.get_global_id();
                auto local_id = id.get_local_id();

                local_ray_count_accessor[0] = 0;

                id.barrier(sycl::access::fence_space::local_space);

                if (global_id < prev_ray_count) {
                    RayData ray_data = prev_ray_buffer[global_id];

                    sycl::int2 pixel_coords = {
                        ray_data.id % ctx.camera.img_size[0],
                        ray_data.id / ctx.camera.img_size[0]};

                    ScopedRng rng(pixel_coords, img_size, rng_buffer);

                    float3 attenuation =
                        float3(ray_data.att_r, ray_data.att_g, ray_data.att_b);
                    float3 radiance =
                        float3(ray_data.rad_r, ray_data.rad_g, ray_data.rad_b);
                    auto res = trace_ray(ctx, rng, ray_data, attenuation, radiance);
                    ray_data.att_r = attenuation.x();
                    ray_data.att_g = attenuation.y();
                    ray_data.att_b = attenuation.z();

                    ray_data.rad_r = radiance.x();
                    ray_data.rad_g = radiance.y();
                    ray_data.rad_b = radiance.z();

                    if (res) {
                        // Final value is computed. Write to image.
                        float4 final_color = float4(sycl::clamp(*res, 0.0f, 1.0f), 1.0f);
                        image_writer.write(pixel_coords, final_color);
                    } else if (depth == (max_depth - 1)) {
                        image_writer.write(pixel_coords, float4(0.0f, 0.0f, 0.0f, 1.0f));
                    } else {
                        // New ray was generated
                        uint32_t ray_index = local_ray_count.fetch_add(1);
                        local_ray_data[ray_index] = ray_data;
                    }
                }

                id.barrier(sycl::access::fence_space::local_space);

                if (local_id == 0) {
                    // ctx.os << "Local ray count " << global_id << ": "
                    //        << local_ray_count_accessor[0] << sycl::endl;
                    local_first_ray_index_accessor[0] =
                        global_ray_count_ref.fetch_add(local_ray_count_accessor[0]);
                }

                id.barrier(sycl::access::fence_space::local_space);

                if (local_id < local_ray_count_accessor[0]) {
                    new_ray_buffer[local_first_ray_index_accessor[0] + local_id] =
                        local_ray_data[local_id];
                }
            });
        })
        .wait();

    // print_elapsed(begin, "shoot end");
}

void WavefrontRenderer::merge_samples(uint32_t sample) {
    app.queue
        .submit([&](sycl::handler &cgh) {
            auto image_reader =
                this->image.get_access<float4, sycl::access::mode::read>(cgh);
            auto combined_image_reader =
                this->combined_image.get_access<float4, sycl::access::mode::read>(cgh);
            auto combined_image_writer =
                this->combined_image.get_access<float4, sycl::access::mode::write>(cgh);

            range<2> local_size{16, 16};
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
            auto combined_image_reader =
                this->combined_image.get_access<float4, sycl::access::mode::read>(cgh);
            auto output_image_writer =
                this->output_image.get_access<float4, sycl::access::mode::write>(cgh);

            range<2> local_size{16, 16};
            range<2> n_groups = {
                ((img_size[0] + local_size[0] - 1) / local_size[0]),
                ((img_size[1] + local_size[1] - 1) / local_size[1]),
            };

            const auto img_size = this->img_size;
            const uint32_t sample_count = this->sample_count;

            cgh.parallel_for(
                sycl::nd_range<2>(n_groups * local_size, local_size),
                [=](sycl::nd_item<2> id) {
                    auto global_id = id.get_global_id();
                    if (global_id[0] >= img_size[0] || global_id[1] >= img_size[1]) {
                        return;
                    }

                    int2 pixel_coords = {global_id[0], global_id[1]};

                    float4 img_val =
                        combined_image_reader.read(pixel_coords) / (float)sample_count;
                    output_image_writer.write(pixel_coords, linear_to_gamma(img_val));
                }
            );
        })
        .wait();
}

void WavefrontRenderer::render_frame(const Camera &camera, const Scene &scene) {
    auto begin = std::chrono::high_resolution_clock::now();

    uint32_t total_ray_count = 0;

    for (uint32_t sample = 0; sample < this->sample_count; sample++) {
        fmt::println("Sample {}", sample);

        this->generate_camera_rays(camera, sample);

        for (uint32_t depth = 0; depth < this->max_depth; depth++) {
            total_ray_count += *this->current_buffer().ray_buffer_length;

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
