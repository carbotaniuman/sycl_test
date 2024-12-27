#include <hipSYCL/algorithms/algorithm.hpp>
#include <hipSYCL/algorithms/numeric.hpp>
#include <sycl/sycl.hpp>

#include <iostream>

int main(int argc, char *argv[]) {
  sycl::queue q;
  if (argc == 1) {
    q = sycl::queue{sycl::default_selector_v,
                    sycl::property::queue::in_order{}};
  } else {
    q = sycl::queue{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
  }

  std::cout << "Running on "
            << q.get_device().get_info<sycl::info::device::name>() << std::endl;
  size_t elems = 512 * 512;

  auto summed_buffer = sycl::malloc_shared<uint32_t>(elems * 2, q);
  for (int i = 0; i < elems; i++) {
    summed_buffer[i] = 2000;
  }
  auto status_buffer = sycl::malloc_shared<uint32_t>(elems, q);
  for (int i = 0; i < elems; i++) {
    status_buffer[i] = 0;
  }

  std::cout << "Starting" << std::endl;

  q.parallel_for(elems, [=](sycl::id<1> idx) {
     auto id = idx.get(0);
     sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel,
                      sycl::memory_scope::device,
                      sycl::access::address_space::global_space>
         status_ref{status_buffer[id]};

     uint32_t data;
     while ((data = status_ref.load()) != 0xBEEF) {
     }
     summed_buffer[id * 2 + 0] = data;
     summed_buffer[id * 2 + 1] = 5555;
   }).wait();

  std::cout << "Ended" << std::endl;

  for (int i = 0; i < elems * 2; i++) {
    if (i % 2 == 0 && summed_buffer[i] != 0xBEEF) {
      std::cout << summed_buffer[i] << " ! " << i << std::endl;
    }
    if (i % 2 == 1 && summed_buffer[i] != 5555) {
      std::cout << summed_buffer[i] << " ! " << i << std::endl;
    }
  }
}
