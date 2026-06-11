// Loader: dlopen the clang-built FKL .so and run fkl_entry on the GPU.
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdio>

typedef void (*entry_t)(float*, float*, int, int, float, float, void*);

int main(int argc, char** argv) {
    const char* so = argc > 1 ? argv[1] : "/tmp/spike_clang.so";
    void* h = dlopen(so, RTLD_NOW | RTLD_LOCAL);
    if (!h) { printf("dlopen failed: %s\n", dlerror()); return 1; }
    entry_t fn = (entry_t)dlsym(h, "fkl_entry");
    if (!fn) { printf("dlsym failed: %s\n", dlerror()); return 1; }

    const int W = 8, H = 4, N = W * H;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; ++i) h_in[i] = (float)i;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    fn(d_in, d_out, W, H, 2.0f, 1.0f, nullptr);

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    bool ok = true;
    printf("first 8: ");
    for (int i = 0; i < 8; ++i) printf("%.1f ", h_out[i]);
    for (int i = 0; i < N; ++i)
        if (h_out[i] != (float)i * 2.0f + 1.0f) ok = false;
    printf("\n%s\n", ok ? "CLANG_RUN_PASS" : "CLANG_RUN_FAIL");
    return ok ? 0 : 1;
}
