#include "MASS3DEA_NOSHARED.hpp"
#if defined(RAJA_ENABLE_CUDA)
#include "common/CudaDataUtils.hpp"
#include <iostream>
namespace rajaperf { namespace apps {

template <size_t block_size>
__launch_bounds__(block_size) __global__
void Mass3DEANoShared(const Real_type* __restrict__ B,
                      const Real_type* __restrict__ D,
                      Real_type* __restrict__ M)
{
  const Index_type e = blockIdx.x;
  const Index_type i1 = threadIdx.x;
  const Index_type i2 = threadIdx.y;
  const Index_type i3 = threadIdx.z;
  mass3dea_noshared::apply(B, D, M, e, i1, i2, i3);
}

template <size_t block_size, size_t tune_idx>
void MASS3DEA_NOSHARED::runCudaVariantImpl(VariantID vid)
{
  setBlockSize(block_size);
  const Index_type run_reps = getRunReps();
  auto res{getCudaResource()};
  MASS3DEA_DATA_SETUP;
  startTimer();
  for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {
    RP_CALI_SUBKERNEL_BEGIN("MASS3DEA_NOSHARED_1");
    if (vid == Base_CUDA) {
      const Real_type* B_native = B;
      const Real_type* D_native = D;
      RPlaunchCudaKernel((Mass3DEANoShared<block_size>), NE,
          dim3(mea::D1D, mea::D1D, mea::D1D), 0, res.get_stream(),
          B_native, D_native, M);
    } else if (vid == RAJA_CUDA) {
      using launch_policy = RAJA::LaunchPolicy<RAJA::cuda_launch_t<true, block_size>>;
      using outer = RAJA::LoopPolicy<RAJA::cuda_block_x_direct>;
      using ix = RAJA::LoopPolicy<RAJA::cuda_thread_x_direct>;
      using iy = RAJA::LoopPolicy<RAJA::cuda_thread_y_direct>;
      using iz = RAJA::LoopPolicy<RAJA::cuda_thread_z_direct>;
      RAJA::launch<launch_policy>(res,
        RAJA::LaunchParams(RAJA::Teams(NE), RAJA::Threads(mea::D1D, mea::D1D, mea::D1D)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
          RAJA::loop<outer>(ctx, RAJA::RangeSegment(0, NE), [&](Index_type e) {
            RAJA::loop<ix>(ctx, RAJA::RangeSegment(0, mea::D1D), [&](Index_type i1) {
              RAJA::loop<iy>(ctx, RAJA::RangeSegment(0, mea::D1D), [&](Index_type i2) {
                RAJA::loop<iz>(ctx, RAJA::RangeSegment(0, mea::D1D), [&](Index_type i3) {
                  const Real_type* RAJA_RESTRICT B_r = B;
                  const Real_type* RAJA_RESTRICT D_r = D;
                  Real_type* RAJA_RESTRICT M_r = M;
                  mass3dea_noshared::apply(B_r, D_r, M_r, e, i1, i2, i3);
                });
              });
            });
          });
        });
    } else {
      getCout() << "\n MASS3DEA_NOSHARED : Unknown Cuda variant id = " << vid << std::endl;
    }
    RP_CALI_SUBKERNEL_END("MASS3DEA_NOSHARED_1");
  }
  stopTimer();
}

void MASS3DEA_NOSHARED::defineCudaVariantTunings()
{
  for (VariantID vid : {Base_CUDA, RAJA_CUDA})
    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
      if (run_params.numValidGPUBlockSize() == 0u || run_params.validGPUBlockSize(block_size)) {
        addVariantTuning<&MASS3DEA_NOSHARED::runCudaVariantImpl<block_size, 0>>(vid, "compile_time_block_stride_loop_" + std::to_string(block_size), Index_type(block_size));
        if (vid == RAJA_CUDA)
          addVariantTuning<&MASS3DEA_NOSHARED::runCudaVariantImpl<block_size, 1>>(vid, "cached_block_stride_loop_" + std::to_string(block_size), Index_type(block_size));
      }
    });
}
} }
#endif
