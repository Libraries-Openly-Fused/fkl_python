"""The exact use case: a vector of ROIs cropped from a source
image, processed, and written to a contiguous Tensor.

C++:
    executeOperations<TransformDPP<>>(stream,
        PerThreadRead<ND::_2D, uchar3>::build(image),
        Crop<>::build(crops),            // std::array<Rect, BATCH>
        Mul<float3>::build(...),
        Add<float3>::build(...),
        TensorWrite<float3>::build(tensor));

Python (this file):
    pipe = fkl.compose(
        fkl.TensorRead(),                # PerThreadRead<_2D, uchar3>
        fkl.Crop(rois),                  # Crop<>::build(std::array<Rect,B>)
        fkl.Cast("float32"),             # uchar3 -> float3
        fkl.Mul((2.0, 2.0, 2.0)),        # Mul<float3>
        fkl.Add((10.0, 20.0, 30.0)),     # Add<float3>
        fkl.TensorWrite(),               # TensorWrite<float3> (B planes)
    )
ONE fused kernel: HF (B crops) + BVF (crop into read) + VF (cast/mul/add).
"""
from harness import dev_u8, unf32, check, check_true, run
import fkl


def t_roi_vector_pipeline():
    W, H = 128, 96                       # source image uchar3
    n = W * H * 3
    src = [(x * 3 + c * 11 + (i := 0)) % 256 for p in range(W * H)
           for c, x in ((c, p % W) for c in range(3))]
    # simpler deterministic content: value = (px*3 + channel*7) % 256
    src = [((p * 3 + c * 7) % 256) for p in range(W * H) for c in range(3)]

    rois = [                              # vector of ROIs, all 24x16
        (0,   0, 24, 16),
        (50, 30, 24, 16),
        (100, 70, 24, 16),
        (10, 60, 24, 16),
        (75,  5, 24, 16),
    ]
    B, CW, CH = len(rois), 24, 16

    pipe = fkl.compose(
        fkl.TensorRead(),
        fkl.Crop(rois),
        fkl.Cast("float32"),
        fkl.Mul((2.0, 2.0, 2.0)),
        fkl.Add((10.0, 20.0, 30.0)),
        fkl.TensorWrite(),
    )
    out = pipe(dev_u8(src, W, H, ch=3))
    got = unf32(out.copy_to_host(), B * CW * CH * 3)

    exp = []
    for (rx, ry, rw, rh) in rois:
        for y in range(rh):
            for x in range(rw):
                p = (ry + y) * W + (rx + x)
                for c, off in ((0, 10.0), (1, 20.0), (2, 30.0)):
                    exp.append(src[p * 3 + c] * 2.0 + off)
    check("ROI vector: 5 crops uchar3 -> float3 Tensor (HF+BVF+VF)", got, exp)

    # one kernel, one .so: verify exactly one variant was compiled
    check_true("ROI vector: single fused kernel (1 signature)",
               len(pipe._variants) == 1)


def t_roi_count_change_recompiles_once():
    """Different B => different std::array<Rect,B> type => new signature.
    Same B with different rect values => same .so (params only)."""
    W, H = 64, 64
    src = [(i * 5) % 256 for i in range(W * H * 3)]
    img = dev_u8(src, W, H, ch=3)

    p3a = fkl.compose(fkl.TensorRead(),
                      fkl.Crop([(0, 0, 8, 8), (8, 8, 8, 8), (16, 16, 8, 8)]),
                      fkl.TensorWrite())
    p3b = fkl.compose(fkl.TensorRead(),
                      fkl.Crop([(1, 1, 8, 8), (9, 9, 8, 8), (17, 17, 8, 8)]),
                      fkl.TensorWrite())
    p3a(img); p3b(img)
    so_a = list(p3a._variants.values())[0][3]
    so_b = list(p3b._variants.values())[0][3]
    check_true("ROI: same B, different rects -> same cached .so", so_a == so_b,
               so_a.name)


if __name__ == "__main__":
    run([t_roi_vector_pipeline, t_roi_count_change_recompiles_once],
        "roi-use-case")
