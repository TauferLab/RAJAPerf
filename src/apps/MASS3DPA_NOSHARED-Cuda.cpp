#include "MASS3DPA_NOSHARED.hpp"
#if defined(RAJA_ENABLE_CUDA)
#include "common/CudaDataUtils.hpp"
#include <iostream>
namespace rajaperf { namespace apps {

template <size_t block_size>
__launch_bounds__(block_size) __global__
void Mass3DPANoShared(const Real_type* __restrict__ B,
                      const Real_type* __restrict__ Bt,
                      const Real_type* __restrict__ D,
                      const Real_type* __restrict__ X,
                      Real_type* __restrict__ Y, Index_type NE)
{
  const Index_type e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e < NE) mass3dpa_noshared::apply(B, Bt, D, X, Y, e);
}

template <size_t block_size>
void MASS3DPA_NOSHARED::runCudaVariantImpl(VariantID vid)
{
  setBlockSize(block_size);
  const Index_type run_reps = getRunReps();
  auto res{getCudaResource()};
  MASS3DPA_DATA_SETUP;
  const Index_type teams = RAJA_DIVIDE_CEILING_INT(NE, Index_type(block_size));
  startTimer();
  for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {
    RP_CALI_SUBKERNEL_BEGIN("MASS3DPA_NOSHARED_1");
    if (vid == Base_CUDA) {
      const Real_type* B_native = B;
      const Real_type* Bt_native = Bt;
      const Real_type* D_native = D;
      const Real_type* X_native = X;
      RPlaunchCudaKernel((Mass3DPANoShared<block_size>), teams, block_size, 0,
                         res.get_stream(), B_native, Bt_native, D_native,
                         X_native, Y, NE);
    } else if (vid == RAJA_CUDA) {
      using policy = RAJA::cuda_exec_async<block_size>;
      RAJA::forall<policy>(res, RAJA::RangeSegment(0, NE),
        [=] RAJA_DEVICE(Index_type e) {
          const Real_type* RAJA_RESTRICT B_r = B;
          const Real_type* RAJA_RESTRICT Bt_r = Bt;
          const Real_type* RAJA_RESTRICT D_r = D;
          const Real_type* RAJA_RESTRICT X_r = X;
          Real_type* RAJA_RESTRICT Y_r = Y;
          mass3dpa_noshared::apply(B_r, Bt_r, D_r, X_r, Y_r, e);
        });
    } else {
      getCout() << "\n MASS3DPA_NOSHARED : Unknown Cuda variant id = " << vid << std::endl;
    }
    RP_CALI_SUBKERNEL_END("MASS3DPA_NOSHARED_1");
  }
  stopTimer();
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(
    MASS3DPA_NOSHARED, Cuda, Base_CUDA, RAJA_CUDA)
} }
#endif
