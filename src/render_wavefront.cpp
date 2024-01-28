#include "render_wavefront.hpp"

#include "trace_ray.hpp"

using namespace raytracer;

using sycl::float2;
using sycl::float3;
using sycl::float4;
using sycl::half;
using sycl::half3;
using sycl::int2;
using sycl::range;

struct ScopedRng {
    ScopedRng(
        uint32_t run_index,
        int2 pixel_coords,
        sycl::range<2> img_size,
        XorShift32State *rng_buffer
    ) {
        uint32_t pixel_id = pixel_coords[0] + (pixel_coords[1] * img_size[0]);
        this->rng_ptr = &rng_buffer[pixel_id + run_index * img_size.size()];
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
      combined_image(
          sycl::image_channel_order::rgba, sycl::image_channel_type::fp32, img_size
      ),
      buffers({Buffers(app, img_size), Buffers(app, img_size)}),
      color_buffer(app, img_size), output_image(output_image), max_depth(max_depth),
      sample_count(sample_count) {
    this->rng_buffer = (XorShift32State *)sycl::aligned_alloc_device(
        alignof(XorShift32State),
        sizeof(XorShift32State) * img_size.size() * SAMPLES_PER_RUN,
        app.queue
    );

    app.queue
        .submit([&](sycl::handler &cgh) {
            auto combined_image_writer =
                this->combined_image.get_access<sycl::float4, sycl::access::mode::write>(
                    cgh
                );

            auto rng_buffer = this->rng_buffer;
            auto img_size = this->img_size;

            ColorBuffer color_buffer = this->color_buffer;

            cgh.parallel_for(sycl::range<2>(img_size), [=](sycl::item<2> item) {
                sycl::int2 pixel_coords(item[0], item[1]);
                uint32_t pixel_linear_pos =
                    pixel_coords[0] + (pixel_coords[1] * img_size[0]);

                for (uint32_t i = 0; i < SAMPLES_PER_RUN; i++) {
                    color_buffer.write(i, pixel_coords, float4(0.0f, 0.0f, 0.0f, 1.0f));
                }
                combined_image_writer.write(pixel_coords, sycl::float4(0.0f));

                // Initialize RNG state
                for (uint32_t i = 0; i < SAMPLES_PER_RUN; i++) {
                    uint32_t rng_index = pixel_linear_pos + i * img_size.size();
                    auto init_generator_state = std::hash<std::size_t>{}(rng_index);
                    rng_buffer[rng_index] =
                        XorShift32State{(uint32_t)init_generator_state};
                }
            });
        })
        .wait();
}

void WavefrontRenderer::generate_camera_rays(const Camera &camera) {
    app.queue
        .submit([&](sycl::handler &cgh) {
            // Group size / range
            range<2> local_size{16, 16};
            range<2> n_groups = {
                ((img_size[0] + local_size[0] - 1) / local_size[0]),
                ((img_size[1] + local_size[1] - 1) / local_size[1]),
            };
            sycl::nd_range<2> for_range(n_groups * local_size, local_size);

            // Params
            auto img_size = this->img_size;
            auto rng_buffer = this->rng_buffer;
            auto ray_ids = this->current_buffer().ray_ids;
            auto ray_origins = this->current_buffer().ray_origins;
            auto ray_directions = this->current_buffer().ray_directions;
            auto ray_attenuations = this->current_buffer().ray_attenuations;
            auto ray_radiances = this->current_buffer().ray_radiances;

            // Set produced ray count
            *this->current_buffer().ray_buffer_length = img_size.size() * SAMPLES_PER_RUN;

            ColorBuffer color_buffer = this->color_buffer;

            cgh.parallel_for(for_range, [=](sycl::nd_item<2> id) {
                auto global_id = id.get_global_id();
                if (global_id[0] >= img_size[0] || global_id[1] >= img_size[1]) {
                    return;
                }

                int2 pixel_coords = {global_id[0], global_id[1]};
                uint32_t pixel_linear_pos =
                    pixel_coords[0] + (pixel_coords[1] * img_size[0]);

                for (uint32_t i = 0; i < SAMPLES_PER_RUN; i++) {
                    color_buffer.write(i, pixel_coords, float4(0.0f, 0.0f, 0.0f, 1.0f));
                }

                for (uint32_t i = 0; i < SAMPLES_PER_RUN; i++) {
                    ScopedRng rng(i, pixel_coords, img_size, rng_buffer);

                    RayData ray = camera.get_ray(pixel_coords, rng);
                    ray_ids[ray.id] = ray.id + i * img_size.size();
                    ray_origins[ray.id] = sycl::float3(ray.org_x, ray.org_y, ray.org_z);
                    ray_directions[ray.id] = sycl::half3(ray.dir_x, ray.dir_y, ray.dir_z);
                    ray_attenuations[ray.id] =
                        sycl::half3(ray.att_r, ray.att_g, ray.att_b);
                    ray_radiances[ray.id] = sycl::half3(ray.rad_r, ray.rad_g, ray.rad_b);
                }
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
            sycl::local_accessor<uint64_t, 1> local_first_ray_index_accessor(
                sycl::range<1>(1), cgh
            );

            sycl::local_accessor<uint32_t, 1> local_ray_ids(
                sycl::range<1>(local_size), cgh
            );
            sycl::local_accessor<float3, 1> local_ray_origins(
                sycl::range<1>(local_size), cgh
            );
            sycl::local_accessor<half3, 1> local_ray_directions(
                sycl::range<1>(local_size), cgh
            );
            sycl::local_accessor<half3, 1> local_ray_attenuations(
                sycl::range<1>(local_size), cgh
            );
            sycl::local_accessor<half3, 1> local_ray_radiances(
                sycl::range<1>(local_size), cgh
            );

            ColorBuffer color_buffer = this->color_buffer;

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
            const auto prev_ray_ids = this->prev_buffer().ray_ids;
            const auto prev_ray_origins = this->prev_buffer().ray_origins;
            const auto prev_ray_directions = this->prev_buffer().ray_directions;
            const auto prev_ray_attenuations = this->prev_buffer().ray_attenuations;
            const auto prev_ray_radiances = this->prev_buffer().ray_radiances;

            const auto new_ray_ids = this->current_buffer().ray_ids;
            const auto new_ray_origins = this->current_buffer().ray_origins;
            const auto new_ray_directions = this->current_buffer().ray_directions;
            const auto new_ray_attenuations = this->current_buffer().ray_attenuations;
            const auto new_ray_radiances = this->current_buffer().ray_radiances;

            uint64_t *global_ray_count = this->current_buffer().ray_buffer_length;

            const uint32_t max_depth = this->max_depth;
            const range<2> img_size = this->img_size;
            XorShift32State *rng_buffer = this->rng_buffer;

            // print_elapsed(begin, "parallel_for begin");
            cgh.parallel_for(for_range, [=](sycl::nd_item<1> id) {
                sycl::atomic_ref<
                    uint64_t,
                    sycl::memory_order_relaxed,
                    sycl::memory_scope_device,
                    sycl::access::address_space::global_space>
                    global_ray_count_ref(*global_ray_count);

                sycl::atomic_ref<
                    uint32_t,
                    sycl::memory_order_relaxed,
                    sycl::memory_scope_device,
                    sycl::access::address_space::local_space>
                    local_ray_count_ref(local_ray_count_accessor[0]);

                auto global_id = id.get_global_id();
                auto local_id = id.get_local_id();

                local_ray_count_ref = 0;

                id.barrier(sycl::access::fence_space::local_space);

                if (global_id < prev_ray_count) {
                    uint32_t ray_id = prev_ray_ids[global_id];
                    uint32_t run_index = ray_id / img_size.size();
                    uint32_t run_ray_id = ray_id % img_size.size();

                    float3 ray_origin = prev_ray_origins[global_id];
                    float3 ray_direction =
                        prev_ray_directions[global_id].convert<float>();
                    float3 ray_attenuation =
                        prev_ray_attenuations[global_id].convert<float>();
                    float3 ray_radiance = prev_ray_radiances[global_id].convert<float>();

                    sycl::int2 pixel_coords = {
                        run_ray_id % ctx.camera.img_size[0],
                        run_ray_id / ctx.camera.img_size[0]};

                    ScopedRng rng(run_index, pixel_coords, img_size, rng_buffer);

                    RTCRay ray = {
                        .org_x = ray_origin.x(),
                        .org_y = ray_origin.y(),
                        .org_z = ray_origin.z(),
                        .tnear = 0.0001f,
                        .dir_x = ray_direction.x(),
                        .dir_y = ray_direction.y(),
                        .dir_z = ray_direction.z(),
                        .time = 0.0f,
                        .tfar = std::numeric_limits<float>::infinity(),
                        .mask = UINT32_MAX,
                        .id = ray_id,
                        .flags = 0,
                    };

                    auto res = trace_ray(ctx, rng, ray, ray_attenuation, ray_radiance);

                    if (res) {
                        // Final value is computed. Write to image.
                        float4 final_color = float4(sycl::clamp(*res, 0.0f, 1.0f), 1.0f);
                        color_buffer.write(run_index, pixel_coords, final_color);
                        /* if (run_index == 0 && pixel_coords.x() == 0 && */
                        /*     pixel_coords.y() == 0) { */
                        /*     ctx.os << "color: " << final_color << "\n"; */
                        /* } */
                    } else if (depth == (max_depth - 1)) {
                        color_buffer.write(
                            run_index, pixel_coords, float4(0.0f, 0.0f, 0.0f, 1.0f)
                        );
                        /* if (run_index == 0 && pixel_coords.x() == 0 && */
                        /*     pixel_coords.y() == 0) { */
                        /*     ctx.os << "color2: " << float4(0.0f, 0.0f, 0.0f, 1.0f) */
                        /*            << "\n"; */
                        /* } */
                    } else {
                        // New ray was generated
                        uint32_t ray_index = local_ray_count_ref.fetch_add(1);
                        local_ray_ids[ray_index] = ray_id;
                        local_ray_origins[ray_index] =
                            float3(ray.org_x, ray.org_y, ray.org_z);
                        local_ray_directions[ray_index] =
                            half3(ray.dir_x, ray.dir_y, ray.dir_z);
                        local_ray_attenuations[ray_index] =
                            ray_attenuation.convert<half>();
                        local_ray_radiances[ray_index] = ray_radiance.convert<half>();
                    }
                }

                id.barrier(sycl::access::fence_space::local_space);

                if (local_id == 0) {
                    local_first_ray_index_accessor[0] =
                        global_ray_count_ref.fetch_add(local_ray_count_ref);
                }

                id.barrier(sycl::access::fence_space::local_space);

                if (local_id < local_ray_count_ref) {
                    const uint64_t i = local_first_ray_index_accessor[0] + local_id;
                    new_ray_ids[i] = local_ray_ids[local_id];
                    new_ray_origins[i] = local_ray_origins[local_id];
                    new_ray_directions[i] = local_ray_directions[local_id];
                    new_ray_attenuations[i] = local_ray_attenuations[local_id];
                    new_ray_radiances[i] = local_ray_radiances[local_id];
                }
            });
        })
        .wait();

    // print_elapsed(begin, "shoot end");
}

void WavefrontRenderer::merge_samples() {
    app.queue
        .submit([&](sycl::handler &cgh) {
            // Group size / range
            range<2> local_size{8, 8};
            range<2> n_groups = {
                ((img_size[0] + local_size[0] - 1) / local_size[0]),
                ((img_size[1] + local_size[1] - 1) / local_size[1]),
            };

            // Accessors
            auto combined_image_reader =
                this->combined_image.get_access<float4, sycl::access::mode::read>(cgh);
            auto combined_image_writer =
                this->combined_image.get_access<float4, sycl::access::mode::write>(cgh);

            // Params
            const auto img_size = this->img_size;

            ColorBuffer color_buffer = this->color_buffer;

            cgh.parallel_for(
                sycl::nd_range<2>(n_groups * local_size, local_size),
                [=](sycl::nd_item<2> id) {
                    auto global_id = id.get_global_id();
                    if (global_id[0] >= img_size[0] || global_id[1] >= img_size[1]) {
                        return;
                    }

                    int2 pixel_coords = {global_id[0], global_id[1]};

                    float4 added_color = float4(0.0f);
                    for (uint32_t i = 0; i < SAMPLES_PER_RUN; i++) {
                        added_color += color_buffer.read(i, pixel_coords);
                    }
                    float4 combined_val = combined_image_reader.read(pixel_coords);
                    combined_image_writer.write(pixel_coords, combined_val + added_color);
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

            range<2> local_size{8, 8};
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
                        (combined_image_reader.read(pixel_coords) / float(sample_count));
                    output_image_writer.write(pixel_coords, linear_to_gamma(img_val));
                }
            );
        })
        .wait();
}

void WavefrontRenderer::render_frame(const Camera &camera, const Scene &scene) {
    auto begin = std::chrono::high_resolution_clock::now();

    if (this->sample_count % SAMPLES_PER_RUN != 0) {
        throw std::runtime_error("Sample count must be a multiple of RUN_COUNT");
    }

    uint64_t total_ray_count = 0;

    for (uint32_t sample = 0; sample < this->sample_count / SAMPLES_PER_RUN; sample++) {
        fmt::println("Sample {}", sample);

        this->generate_camera_rays(camera);

        for (uint32_t depth = 0; depth < this->max_depth; depth++) {
            total_ray_count += *this->current_buffer().ray_buffer_length;

            buffer_index++;

            this->shoot_rays(camera, scene, depth);
        }

        this->merge_samples();
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
