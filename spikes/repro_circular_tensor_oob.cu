// GROUND TRUTH: does fk::CircularTensor (used directly, as designed) work?
// float3, COLOR_PLANES=1? No: aggregate path needs is_aggregate_v<T>.
// Use the canonical config from the class design: T=float, COLOR_PLANES=3
// (VectorType_t<float,3> = float3 storage), packed Standard.
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/core/execution_model/execution_model.h>
#include <fused_kernel/core/data/circular_tensor.h>
#include <fused_kernel/algorithms/algorithms.h>
#include <cstdio>
#include <vector>
using namespace fk;

int main() {
    constexpr int W = 8, H = 4, B = 3;
    using CT = CircularTensor<float, 3, B, CircularTensorOrder::NewestFirst, ColorPlanes::Standard>;
    CT ct(W, H, MemType::Device);

    Stream stream;
    Ptr2D<float3> frame(W, H);

    // push frames k=1..5, frame k = constant (k, k+0.1, k+0.2)
    for (int k = 1; k <= 5; ++k) {
        std::vector<float3> h(W * H, make_<float3>(float(k), k + 0.1f, k + 0.2f));
        cudaMemcpy2D(frame.ptr().data, frame.ptr().dims.pitch, h.data(),
                     W * sizeof(float3), W * sizeof(float3), H, cudaMemcpyHostToDevice);
        ct.update(stream,
                  PerThreadRead<ND::_2D, float3>::build(frame),
                  TensorSplit<float3>::build(ct));
        stream.sync();

        // read back: Tensor base, planes layout = B batch x 3 color planes
        std::vector<float> out(W * H * B * 3);
        cudaMemcpy(out.data(), ct.ptr().data, out.size() * sizeof(float), cudaMemcpyDeviceToHost);
        printf("after push %d: plane0.x=%5.1f plane1.x=%5.1f plane2.x=%5.1f\n",
               k, out[0], out[W*H*3], out[2*W*H*3]);
    }
    // expectation (NewestFirst): planes = [5, 4, 3]
    std::vector<float> out(W * H * B * 3);
    cudaMemcpy(out.data(), ct.ptr().data, out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    const bool ok = out[0] == 5.f && out[W*H*3] == 4.f && out[2*W*H*3] == 3.f;
    printf(ok ? "fk::CircularTensor DIRECT USE: CORRECT\n"
              : "fk::CircularTensor DIRECT USE: WRONG ORDER/VALUES\n");
    return ok ? 0 : 1;
}
