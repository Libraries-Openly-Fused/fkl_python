// Minimal reproducer 2: Executor<DivergentBatchTransformDPP<...>> fails to
// compile when given IOpSequences. fuseBackSequence() does
//     apply(BackFuser::fuse_back<IOps...>, iOpSeq.iOps)
// instantiating fuse_back with EXPLICIT (non-reference) template args, so its
// IOps&&... parameters become rvalue references -- which cannot bind to the
// const lvalues stored in the sequence tuple.
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/core/execution_model/execution_model.h>
#include <fused_kernel/algorithms/algorithms.h>

using namespace fk;

struct MySelector {
    FK_HOST_DEVICE_FUSE uint at(const uint& index) { return index == 0 ? 1u : 2u; }
};

int main() {
    Stream stream;
    Ptr2D<float> imgA(8, 4), imgB(8, 4);
    const std::array<Ptr2D<float>, 2> input{ imgA, imgB };
    Tensor<float> output(8, 4, 2);

    const auto seq1 = buildOperationSequence(
        PerThreadRead<ND::_2D, float>::build(input),
        Mul<float>::build(2.0f),
        TensorWrite<float>::build(output));
    const auto seq2 = buildOperationSequence(
        PerThreadRead<ND::_2D, float>::build(input),
        Add<float>::build(100.0f),
        TensorWrite<float>::build(output));

    // error: qualifiers dropped in binding reference of type '... &&'
    Executor<DivergentBatchTransformDPP<ParArch::GPU_NVIDIA, MySelector>>::
        executeOperations(stream, seq1, seq2);
    stream.sync();
    return 0;
}
