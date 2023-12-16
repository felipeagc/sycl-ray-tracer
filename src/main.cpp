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

    RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);

    std::vector<float3> directions = {
        float3(0.0f, 0.0f, 1.0f), float3(0.0f, 0.0f, -1.0f),
        float3(0.0f, 1.0f, 0.0f), float3(0.0f, -1.0f, 0.0f),
        float3(1.0f, 0.0f, 0.0f), float3(-1.0f, 0.0f, 0.0f),
    };

    for (auto &dir : directions) {
        std::cout << "Shooting ray: " << dir[0] << " " << dir[1] << " " << dir[2] << std::endl;
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

        rtcIntersect1(scene.scene, &rayhit, &args);

        if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
            std::cout << "Hit geometry ID: " << rayhit.hit.geomID << std::endl;
            std::cout << "Hit instance ID: " << rayhit.hit.instID[0]
                      << std::endl;
            std::cout << "Hit distance: " << rayhit.ray.tfar << std::endl;
        }
    }
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
