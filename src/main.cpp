#include "app.hpp"
#include "model.hpp"

#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>
#include <embree4/rtcore.h>

using sycl::float2;
using sycl::float3;
using sycl::float4;
using sycl::int2;
using sycl::range;

using namespace raytracer;

void run(App &app) {
    std::vector<Model> models;
    models.emplace_back(app, "../assets/cube.glb");
    Scene scene(app, models);

    auto e = app.queue.submit([&](sycl::handler &cgh) {
        sycl::stream os(1024, 256, cgh);
        RTCScene r_scene = scene.scene;

        cgh.single_task<class test>([=]() {
            RTCIntersectArguments args;
            rtcInitIntersectArguments(&args);

            std::array<float3, 6> directions = {
                float3(0.0f, 0.0f, 1.0f), float3(0.0f, 0.0f, -1.0f),
                float3(0.0f, 1.0f, 0.0f), float3(0.0f, -1.0f, 0.0f),
                float3(1.0f, 0.0f, 0.0f), float3(-1.0f, 0.0f, 0.0f),
            };

            for (auto &dir : directions) {
                os << "Shooting ray: " << dir[0] << " " << dir[1] << " "
                          << dir[2] << sycl::endl;
                RTCRayHit rayhit = {};
                rayhit.ray.org_x = 3.0f;
                rayhit.ray.org_y = 0.0f;
                rayhit.ray.org_z = 0.0f;
                rayhit.ray.dir_x = dir[0];
                rayhit.ray.dir_y = dir[1];
                rayhit.ray.dir_z = dir[2];
                rayhit.ray.tnear = 0;
                rayhit.ray.tfar = std::numeric_limits<float>::infinity();
                rayhit.ray.mask = -1;
                rayhit.ray.flags = 0;
                rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
                rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

                rtcIntersect1(r_scene, &rayhit, &args);

                if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
                    os << "Hit geometry ID: " << rayhit.hit.geomID
                       << sycl::endl;
                    os << "Hit instance ID: " << rayhit.hit.instID[0]
                       << sycl::endl;
                    os << "Hit distance: " << rayhit.ray.tfar << sycl::endl;
                }
            }
        });
    });

    e.wait_and_throw();
}

int main(int argc, char *argv[]) {
    try {
        App app;
        run(app);
    } catch (sycl::exception const &e) {
        std::cout << "Caught SYCL exception: " << e.what() << "\n";
        std::terminate();
    }

    return 0;
}
