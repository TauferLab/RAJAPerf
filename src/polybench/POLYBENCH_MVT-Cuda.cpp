//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_MVT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void poly_mvt_1(Real_ptr A, Real_ptr x1, Real_ptr y1,
                           Index_type N)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;

   if (i < N) {
     POLYBENCH_MVT_BODY1;
     for (Index_type j = 0; j < N; ++j ) {
       POLYBENCH_MVT_BODY2;
     }
     POLYBENCH_MVT_BODY3;
   }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void poly_mvt_2(Real_ptr A, Real_ptr x2, Real_ptr y2,
                           Index_type N)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;

   if (i < N) {
     POLYBENCH_MVT_BODY4;
     for (Index_type j = 0; j < N; ++j ) {
       POLYBENCH_MVT_BODY5;
     }
     POLYBENCH_MVT_BODY6;
   }
}


template < size_t block_size >
void POLYBENCH_MVT::runCudaVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getCudaResource()};

  POLYBENCH_MVT_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(N, block_size);
      constexpr size_t shmem = 0;

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_MVT_1");
      RPlaunchCudaKernel( (poly_mvt_1<block_size>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          A, x1, y1, N );
      RP_CALI_SUBKERNEL_END("POLYBENCH_MVT_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_MVT_2");
      RPlaunchCudaKernel( (poly_mvt_2<block_size>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          A, x2, y2, N );
      RP_CALI_SUBKERNEL_END("POLYBENCH_MVT_2");

    }
    stopTimer();

  } else if (vid == RAJA_CUDA) {

    POLYBENCH_MVT_VIEWS_RAJA;

    // One lambda per kernel, with the j-reduction written as an ordinary
    // sequential loop inside it -- the same shape as poly_mvt_1/poly_mvt_2
    // above, rather than a RAJA::statement::For<seq_exec> around a separate
    // lambda.
    using EXEC_POL = RAJA::cuda_exec<block_size, true /*async*/>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

#if CUDART_VERSION >= 9000
// Defining an extended __device__ lambda inside inside another lambda
// was not supported until CUDA 9.x
      RAJA::region<RAJA::seq_region>( [=]() {
#endif

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_MVT_1");
        RAJA::forall<EXEC_POL> ( res, RAJA::RangeSegment{0, N},
          [=] __device__ (Index_type i) {
            POLYBENCH_MVT_BODY1_RAJA_LOCAL;
            POLYBENCH_MVT_UNROLL
            for (Index_type j = 0; j < N; ++j ) {
              POLYBENCH_MVT_BODY2_RAJA;
            }
            POLYBENCH_MVT_BODY3_RAJA;
        });
        RP_CALI_SUBKERNEL_END("POLYBENCH_MVT_1");

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_MVT_2");
        RAJA::forall<EXEC_POL> ( res, RAJA::RangeSegment{0, N},
          [=] __device__ (Index_type i) {
            POLYBENCH_MVT_BODY4_RAJA_LOCAL;
            POLYBENCH_MVT_UNROLL
            for (Index_type j = 0; j < N; ++j ) {
              POLYBENCH_MVT_BODY5_RAJA;
            }
            POLYBENCH_MVT_BODY6_RAJA;
        });
        RP_CALI_SUBKERNEL_END("POLYBENCH_MVT_2");

#if CUDART_VERSION >= 9000
      }); // end sequential region (for single-source code)
#endif

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_MVT : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_MVT, Cuda, Base_CUDA, RAJA_CUDA)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA

