// Reproducer: DivergentBatchTransformDPP Executor launches sum-of-z planes.
// Two sequences, EACH with a full-batch read covering B=3 planes and a
// selector routing plane 0 -> seq1, planes 1,2 -> seq2. Expected: grid.z = 3.
// Observed: grid.z = 6 (sum of both sequences' z extents); planes 3..5
// execute with selector values beyond the intent and write out of bounds.
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/core/execution_model/execution_model.h>
#include <fused_kernel/algorithms/algorithms.h>
#include <cstdio>
using namespace fk;

struct Selector {
    FK_HOST_DEVICE_FUSE uint at(const uint& z) { return z == 0 ? 1u : 2u; }
};

int main() {
    constexpr int W = 8, H = 4, B = 3;
    Stream stream;

    Tensor<float> in(W, H, B, 1, MemType::Device);
    Tensor<float> out(W, H, B, 1, MemType::Device);
    // canary buffer allocated right after 'out' to detect OOB writes
    Tensor<float> canary(W, H, B, 1, MemType::Device);

    std::vector<float> hostIn(W * H * B);
    for (int i = 0; i < W * H * B; ++i) hostIn[i] = float(i % 100);
    cudaMemcpy(in.ptr().data, hostIn.data(), W * H * B * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(out.ptr().data, 0, W * H * B * sizeof(float));
    std::vector<float> sentinel(W * H * B, 777.f);
    cudaMemcpy(canary.ptr().data, sentinel.data(), W * H * B * sizeof(float), cudaMemcpyHostToDevice);

    const auto seq1 = buildOperationSequence(
        TensorRead<float>::build(in), Mul<float>::build(10.f), TensorWrite<float>::build(out));
    const auto seq2 = buildOperationSequence(
        TensorRead<float>::build(in), Add<float>::build(1.f), TensorWrite<float>::build(out));

    Executor<DivergentBatchTransformDPP<ParArch::GPU_NVIDIA, Selector>>::
        executeOperations(stream, seq1, seq2);
    stream.sync();

    std::vector<float> got(W * H * B), can(W * H * B);
    cudaMemcpy(got.data(), out.ptr().data, W * H * B * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(can.data(), canary.ptr().data, W * H * B * sizeof(float), cudaMemcpyDeviceToHost);

    int wrong = 0, corrupted = 0;
    for (int z = 0; z < B; ++z)
        for (int i = 0; i < W * H; ++i) {
            const float v = got[z * W * H + i];
            const float x = hostIn[z * W * H + i];
            const float expect = z == 0 ? x * 10.f : x + 1.f;
            if (v != expect) ++wrong;
        }
    for (int i = 0; i < W * H * B; ++i) if (can[i] != 777.f) ++corrupted;

    printf("wrong output elements: %d / %d\n", wrong, W * H * B);
    printf("canary corrupted elements: %d / %d\n", corrupted, W * H * B);
    printf(wrong == 0 && corrupted == 0 ? "PASS (intended semantics)\n"
                                        : "FAIL (sum-of-z launch)\n");
    return wrong == 0 && corrupted == 0 ? 0 : 1;
}
