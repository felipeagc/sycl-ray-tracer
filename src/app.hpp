#pragma once

#include <sycl/sycl.hpp>
#include <embree4/rtcore.h>

namespace raytracer {

static auto exception_handler = [](sycl::exception_list e_list) {
    for (std::exception_ptr const &e : e_list) {
        try {
            std::rethrow_exception(e);
        } catch (std::exception const &e) {
#if _DEBUG
            std::cout << "Failure" << std::endl;
#endif
            std::terminate();
        }
    }
};

static void enablePersistentJITCache() {
#if defined(_WIN32)
    _putenv_s("SYCL_CACHE_PERSISTENT", "1");
    _putenv_s("SYCL_CACHE_DIR", "gpucache");
#else
    setenv("SYCL_CACHE_PERSISTENT", "1", 1);
    setenv("SYCL_CACHE_DIR", "gpucache", 1);
#endif
}

struct App {
    sycl::device sycl_device;
    sycl::queue queue;
    sycl::context context;
    RTCDevice embree_device;

    App(const App &) = delete;
    App &operator=(const App &) = delete;

    App(App &&) = delete;
    App &operator=(App &&) = delete;

    App() {
        enablePersistentJITCache();

        this->sycl_device = sycl::device(rtcSYCLDeviceSelector);
        this->queue = sycl::queue(this->sycl_device, exception_handler);
        this->context = sycl::context(this->sycl_device);
        this->embree_device = rtcNewSYCLDevice(context, "");

        std::cout
            << "Running on device: "
            << this->queue.get_device().get_info<sycl::info::device::name>()
            << "\n";
    }

    ~App() { rtcReleaseDevice(this->embree_device); }
};

}  // namespace raytracer
