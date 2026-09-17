//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_JACOBI_1D.hpp"

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
__global__ void poly_jacobi_1D_1(Real_ptr A, Real_ptr B, Index_type N)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;

   if (i > 0 && i < N-1) {
     POLYBENCH_JACOBI_1D_BODY1;
   }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void poly_jacobi_1D_2(Real_ptr A, Real_ptr B, Index_type N)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;

   if (i > 0 && i < N-1) {
     POLYBENCH_JACOBI_1D_BODY2;
   }
}


template < size_t block_size, size_t reorder_num >
void POLYBENCH_JACOBI_1D::runHipVariantReorder(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  POLYBENCH_JACOBI_1D_DATA_SETUP;

  if (vid == RAJA_HIP) {

    constexpr bool async = true;
    using launch_policy =
        RAJA::LaunchPolicy<RAJA::hip_launch_t<async, block_size>>;
    using teams_x = RAJA::LoopPolicy<RAJA::hip_block_x_direct>;
    using teams_z = RAJA::LoopPolicy<RAJA::hip_block_z_direct>;
    using threads_x =
        RAJA::LoopPolicy<RAJA::hip_thread_size_x_direct<block_size>>;

    const Index_type blocks = RAJA_DIVIDE_CEILING_INT(N-2, block_size);
    const Index_type blocks_z = RAJA_DIVIDE_CEILING_INT(blocks, reorder_num);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_1D_1");
      RAJA::launch<launch_policy>(res,
        RAJA::LaunchParams(RAJA::Teams(reorder_num, 1, blocks_z),
                           RAJA::Threads(block_size)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
          RAJA::loop<teams_z>(ctx, RAJA::RangeSegment(0, blocks_z),
            [&](Index_type bz) {
              RAJA::loop<teams_x>(ctx, RAJA::RangeSegment(0, reorder_num),
                [&](Index_type chiplet) {
                  RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(0, block_size),
                    [&](Index_type ti) {
                      const Index_type i = 1 +
                          (blocks_z * chiplet + bz) * block_size + ti;
                      if (i < N-1) {
                        POLYBENCH_JACOBI_1D_BODY1;
                      }
                    });
                });
            });
        });
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_1D_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_1D_2");
      RAJA::launch<launch_policy>(res,
        RAJA::LaunchParams(RAJA::Teams(reorder_num, 1, blocks_z),
                           RAJA::Threads(block_size)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
          RAJA::loop<teams_z>(ctx, RAJA::RangeSegment(0, blocks_z),
            [&](Index_type bz) {
              RAJA::loop<teams_x>(ctx, RAJA::RangeSegment(0, reorder_num),
                [&](Index_type chiplet) {
                  RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(0, block_size),
                    [&](Index_type ti) {
                      const Index_type i = 1 +
                          (blocks_z * chiplet + bz) * block_size + ti;
                      if (i < N-1) {
                        POLYBENCH_JACOBI_1D_BODY2;
                      }
                    });
                });
            });
        });
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_1D_2");
    }
    stopTimer();

  } else {
    getCout() << "\n  POLYBENCH_JACOBI_1D : Unknown Hip variant id = "
              << vid << std::endl;
  }
}


template < size_t block_size >
void POLYBENCH_JACOBI_1D::runHipVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  POLYBENCH_JACOBI_1D_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(N, block_size);
      constexpr size_t shmem = 0;

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_1D_1");
      RPlaunchHipKernel( (poly_jacobi_1D_1<block_size>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         A, B, N );
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_1D_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_1D_2");
      RPlaunchHipKernel( (poly_jacobi_1D_2<block_size>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         A, B, N );
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_1D_2");

    }
    stopTimer();

  } else if (vid == RAJA_HIP) {

    using EXEC_POL = RAJA::hip_exec<block_size, true /*async*/>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_1D_1");
      RAJA::forall<EXEC_POL> ( res, RAJA::RangeSegment{1, N-1},
        [=] __device__ (Index_type i) {
          POLYBENCH_JACOBI_1D_BODY1;
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_1D_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_1D_2");
      RAJA::forall<EXEC_POL> ( res, RAJA::RangeSegment{1, N-1},
        [=] __device__ (Index_type i) {
          POLYBENCH_JACOBI_1D_BODY2;
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_1D_2");

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_JACOBI_1D : Unknown Hip variant id = " << vid << std::endl;
  }
}

void POLYBENCH_JACOBI_1D::defineHipVariantTunings()
{
  for (VariantID vid : {Base_HIP, RAJA_HIP}) {
    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {
        if (block_size == 0u) {
          addVariantTuning<&POLYBENCH_JACOBI_1D::runHipVariantImpl<block_size>>(
              vid, "block_auto", Index_type(0));
        } else {
          addVariantTuning<&POLYBENCH_JACOBI_1D::runHipVariantImpl<block_size>>(
              vid, "block_"+std::to_string(block_size), Index_type(block_size));
        }
      }
    });

    if (vid == RAJA_HIP &&
        (run_params.numValidGPUBlockSize() == 0u ||
         run_params.validGPUBlockSize(256u))) {
      addVariantTuning<&POLYBENCH_JACOBI_1D::runHipVariantReorder<256u, 6u>>(
          vid, "reorder6_256", Index_type(256));
    }
  }
}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
