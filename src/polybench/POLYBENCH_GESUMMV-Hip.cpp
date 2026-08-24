//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GESUMMV.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void poly_gesummv(Real_ptr x, Real_ptr y,
                             Real_ptr A, Real_ptr B,
                             Real_type alpha, Real_type beta,
                             Index_type N)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;

   if (i < N) {
     POLYBENCH_GESUMMV_BODY1;
     for (Index_type j = 0; j < N; ++j ) {
       POLYBENCH_GESUMMV_BODY2;
     }
     POLYBENCH_GESUMMV_BODY3;
   }
}


template < size_t block_size >
void POLYBENCH_GESUMMV::runHipVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  POLYBENCH_GESUMMV_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GESUMMV_1");
      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(N, block_size);
      constexpr size_t shmem = 0;
    
      RPlaunchHipKernel( (poly_gesummv<block_size>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         x, y,
                         A, B, 
                         alpha, beta,
                         N );
      RP_CALI_SUBKERNEL_END("POLYBENCH_GESUMMV_1");

    }
    stopTimer();

  } else if (vid == RAJA_HIP) {

    POLYBENCH_GESUMMV_VIEWS_RAJA;

    using EXEC_POL = RAJA::hip_exec<block_size, true /*async*/>;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GESUMMV_1");
        RAJA::forall<EXEC_POL> ( res, RAJA::RangeSegment{0, N},
          [=] __device__ (Index_type i) {
            POLYBENCH_GESUMMV_BODY1_RAJA_LOCAL;
            POLYBENCH_GESUMMV_UNROLL
            for (Index_type j = 0; j < N; ++j ) {
              POLYBENCH_GESUMMV_BODY2_RAJA;
            }
            POLYBENCH_GESUMMV_BODY3_RAJA;
        });
        RP_CALI_SUBKERNEL_END("POLYBENCH_GESUMMV_1");

      }
      stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_GESUMMV : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_GESUMMV, Hip, Base_HIP, RAJA_HIP)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP

