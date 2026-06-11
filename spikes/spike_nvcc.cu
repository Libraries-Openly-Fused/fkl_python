/* Spike 1: does FKL compile + run on this box via nvcc?
   This also IS the nvcc-fallback codegen template: a C-ABI launcher
   wrapping a fully-typed FKL fused operation chain. */
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/core/execution_model/execution_model.h>
#include <fused_kernel/algorithms/algorithms.h>
#include <cstdio>

using namespace fk;

// A C-ABI entry point. This is exactly what the Python JIT will generate:
// host build() + executeOperations live here (compiled by nvcc), and we
// expose a plain extern "C" symbol that Python dlopen/dlsym + calls.
extern "C" void fkl_entry(float* d_in, float* d_out, int width, int height,
                          float mul, float add, void* stream_ptr) {
    Stream stream; // for the spike we make our own; real version takes ext stream
    const uint pitch = (uint)(width * (int)sizeof(float));
    Ptr2D<float> input(d_in, (uint)width, (uint)height, pitch, MemType::Device);
    Ptr2D<float> output(d_out, (uint)width, (uint)height, pitch, MemType::Device);

    executeOperations<TransformDPP<>>(input, output, stream,
                                      Mul<float>::build(mul),
                                      Add<float>::build(add));
    stream.sync();
}

int main() {
    const int W = 8, H = 4, N = W * H;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; ++i) h_in[i] = (float)i;

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    fkl_entry(d_in, d_out, W, H, 2.0f, 1.0f, nullptr);

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    bool ok = true;
    for (int i = 0; i < N; ++i) {
        float expect = (float)i * 2.0f + 1.0f;
        if (h_out[i] != expect) { ok = false; printf("MISMATCH at %d: got %f want %f\n", i, h_out[i], expect); }
    }
    printf("first 8: ");
    for (int i = 0; i < 8; ++i) printf("%.1f ", h_out[i]);
    printf("\n%s\n", ok ? "SPIKE_NVCC_PASS" : "SPIKE_NVCC_FAIL");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
