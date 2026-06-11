// Runtime check: fixed COLOR_BGR2GRAY produces correct BT.601 luma.
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/core/execution_model/execution_model.h>
#include <fused_kernel/algorithms/algorithms.h>
#include <cstdio>
#include <cmath>

using namespace fk;

int main() {
    constexpr int W = 8, H = 2, N = W * H;
    uchar3 h_in[N];
    for (int i = 0; i < N; ++i)
        h_in[i] = make_<uchar3>((i * 17) % 256, (i * 17 + 85) % 256, (i * 17 + 170) % 256);

    uchar3* d_in; uchar* d_out;
    cudaMalloc(&d_in, N * sizeof(uchar3));
    cudaMalloc(&d_out, N * sizeof(uchar));
    cudaMemcpy(d_in, h_in, N * sizeof(uchar3), cudaMemcpyHostToDevice);

    Stream stream;
    Ptr2D<uchar3> input(d_in, W, H, W * sizeof(uchar3), MemType::Device);
    Ptr2D<uchar> output(d_out, W, H, W * sizeof(uchar), MemType::Device);
    executeOperations<TransformDPP<>>(input, output, stream,
        ColorConversion<ColorConversionCodes::COLOR_BGR2GRAY, uchar3, uchar>::build());
    stream.sync();

    uchar h_out[N];
    cudaMemcpy(h_out, d_out, N * sizeof(uchar), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < N; ++i) {
        // input is BGR: luma = 0.299*R + 0.587*G + 0.114*B
        const float b = h_in[i].x, g = h_in[i].y, r = h_in[i].z;
        const int expect = (int)nearbyintf(0.299f * r + 0.587f * g + 0.114f * b);
        if (abs((int)h_out[i] - expect) > 1) {
            printf("MISMATCH @%d got %d want %d\n", i, h_out[i], expect);
            ok = false;
        }
    }
    printf("%s\n", ok ? "BGR2GRAY_RUNTIME_PASS" : "BGR2GRAY_RUNTIME_FAIL");
    return ok ? 0 : 1;
}
