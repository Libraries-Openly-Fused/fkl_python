// Spike 2: can CUDA 13.3 NVRTC digest FKL's device headers?
// This is the exact blocker Oscar hit in June 2025 (no <type_traits>, forced cuda::std).
// We compile a __global__ that calls fused fk Operation exec() device-side.
// Build: g++ spike_nvrtc.cpp -o spike_nvrtc -I$CUDA/include -L$CUDA/lib64 -lnvrtc -lcuda -ldl
#include <nvrtc.h>
#include <cuda.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

#define NVRTC_OK(x) do { nvrtcResult r=(x); if(r!=NVRTC_SUCCESS){ \
    std::cerr<<"NVRTC ERR "<<nvrtcGetErrorString(r)<<" @"<<__LINE__<<"\n"; return 1;} }while(0)
#define CU_OK(x) do { CUresult r=(x); if(r!=CUDA_SUCCESS){ const char*m; cuGetErrorString(r,&m); \
    std::cerr<<"CU ERR "<<m<<" @"<<__LINE__<<"\n"; return 1;} }while(0)

// Device-only kernel: NO host code, NO executeOperations.
// Directly use fk Operations' device exec() on a per-thread basis.
const char* SRC = R"FKL(
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>

extern "C" __global__ void fkl_kernel(const float* in, float* out, int n, float mul, float add) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Use FKL device Operations exactly as the DPP would, chained (vertical fusion):
        float v = in[i];
        v = fk::Mul<float>::exec(v, mul);   // Binary Op: input + params
        v = fk::Add<float>::exec(v, add);
        out[i] = v;
    }
}
)FKL";

int main() {
    const char* cuda = std::getenv("CUDA_HOME");
    std::string fklInc = "-I/home/johnny/Projects/oscar/FusedKernelLibrary/include";
    std::string cudaInc = std::string("-I") + (cuda?cuda:"/usr/local/cuda") + "/include";

    nvrtcProgram prog;
    NVRTC_OK(nvrtcCreateProgram(&prog, SRC, "fkl_kernel.cu", 0, nullptr, nullptr));
    std::vector<const char*> opts = {
        "--gpu-architecture=sm_120",
        "-std=c++20",
        "--device-as-default-execution-space",
        fklInc.c_str(),
        cudaInc.c_str(),
    };
    nvrtcResult cres = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());
    size_t logSize; nvrtcGetProgramLogSize(prog, &logSize);
    if (logSize > 1) { std::string log(logSize, '\0'); nvrtcGetProgramLog(prog, &log[0]);
        std::cerr << "--- NVRTC LOG ---\n" << log << "\n"; }
    if (cres != NVRTC_SUCCESS) { std::cerr << "NVRTC_COMPILE_FAIL\n"; return 1; }
    std::cout << "NVRTC compiled FKL headers OK\n";

    size_t ptxSize; NVRTC_OK(nvrtcGetPTXSize(prog, &ptxSize));
    std::string ptx(ptxSize, '\0'); NVRTC_OK(nvrtcGetPTX(prog, &ptx[0]));
    nvrtcDestroyProgram(&prog);

    // Launch via driver API
    CU_OK(cuInit(0));
    CUdevice dev; CU_OK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CU_OK(cuCtxCreate(&ctx, nullptr, 0, dev));
    CUmodule mod; CU_OK(cuModuleLoadData(&mod, ptx.c_str()));
    CUfunction fn; CU_OK(cuModuleGetFunction(&fn, mod, "fkl_kernel"));

    const int N = 8;
    float h_in[N], h_out[N];
    for (int i=0;i<N;i++) h_in[i]=(float)i;
    CUdeviceptr d_in, d_out;
    CU_OK(cuMemAlloc(&d_in, N*sizeof(float)));
    CU_OK(cuMemAlloc(&d_out, N*sizeof(float)));
    CU_OK(cuMemcpyHtoD(d_in, h_in, N*sizeof(float)));
    int n=N; float mul=2.f, add=1.f;
    void* args[] = { &d_in, &d_out, &n, &mul, &add };
    CU_OK(cuLaunchKernel(fn, 1,1,1, N,1,1, 0, 0, args, nullptr));
    CU_OK(cuCtxSynchronize());
    CU_OK(cuMemcpyDtoH(h_out, d_out, N*sizeof(float)));

    bool ok=true;
    std::cout << "first 8: ";
    for(int i=0;i<N;i++){ std::cout<<h_out[i]<<" "; if(h_out[i]!=(float)i*2.f+1.f) ok=false; }
    std::cout << "\n" << (ok?"SPIKE_NVRTC_PASS":"SPIKE_NVRTC_FAIL") << "\n";
    return ok?0:1;
}
