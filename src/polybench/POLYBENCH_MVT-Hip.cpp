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

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

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
__launch_bounds__(block_size)
__global__ void poly_mvt_1_reorder(Real_ptr A, Real_ptr x1, Real_ptr y1,
                                   Index_type N)
{
   Index_type i = (gridDim.z * blockIdx.x + blockIdx.z) * block_size + threadIdx.x;

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
__global__ void poly_mvt_2_reorder(Real_ptr A, Real_ptr x2, Real_ptr y2,
                                   Index_type N)
{
   Index_type i = (gridDim.z * blockIdx.x + blockIdx.z) * block_size + threadIdx.x;

   if (i < N) {
     POLYBENCH_MVT_BODY4;
     for (Index_type j = 0; j < N; ++j ) {
       POLYBENCH_MVT_BODY5;
     }
     POLYBENCH_MVT_BODY6;
   }
}


template < size_t block_size, size_t reorder_num >
void POLYBENCH_MVT::runHipVariantReorder(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  POLYBENCH_MVT_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      dim3 nthreads_per_block(block_size, 1, 1);
      int blocks = static_cast<int>(RAJA_DIVIDE_CEILING_INT(N, block_size));
      dim3 nblocks(reorder_num,
                   1,
                   static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(blocks, reorder_num)));
      constexpr size_t shmem = 0;

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_MVT_1");
      RPlaunchHipKernel( (poly_mvt_1_reorder<block_size>),
                         nblocks, nthreads_per_block,
                         shmem, res.get_stream(),
                         A, x1, y1, N );
      RP_CALI_SUBKERNEL_END("POLYBENCH_MVT_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_MVT_2");
      RPlaunchHipKernel( (poly_mvt_2_reorder<block_size>),
                         nblocks, nthreads_per_block,
                         shmem, res.get_stream(),
                         A, x2, y2, N );
      RP_CALI_SUBKERNEL_END("POLYBENCH_MVT_2");

    }
    stopTimer();

  } else if (vid == RAJA_HIP) {

    POLYBENCH_MVT_VIEWS_RAJA;

    constexpr bool async = true;
    using launch_policy = RAJA::LaunchPolicy<RAJA::hip_launch_t<async, block_size>>;
    using teams_x = RAJA::LoopPolicy<RAJA::hip_block_x_direct>;
    using teams_z = RAJA::LoopPolicy<RAJA::hip_block_z_direct>;
    using threads_x = RAJA::LoopPolicy<RAJA::hip_thread_size_x_direct<block_size>>;

    const Index_type blocks = RAJA_DIVIDE_CEILING_INT(N, block_size);
    const Index_type blocks_z = RAJA_DIVIDE_CEILING_INT(blocks, reorder_num);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_MVT_1");
      RAJA::launch<launch_policy>(res,
        RAJA::LaunchParams(RAJA::Teams(reorder_num, 1, blocks_z),
                           RAJA::Threads(block_size)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
          RAJA::loop<teams_z>(ctx, RAJA::RangeSegment(0, blocks_z),
            [&](Index_type bz) {
              RAJA::loop<teams_x>(ctx, RAJA::RangeSegment(0, reorder_num),
                [&](Index_type chiplet) {
                  RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(0, block_size),
                    [&](Index_type tx) {
                      const Index_type i =
                          (blocks_z * chiplet + bz) * block_size + tx;
                      if (i < N) {
                        POLYBENCH_MVT_BODY1_RAJA_LOCAL;
                        POLYBENCH_MVT_UNROLL
                        for (Index_type j = 0; j < N; ++j) {
                          POLYBENCH_MVT_BODY2_RAJA;
                        }
                        POLYBENCH_MVT_BODY3_RAJA;
                      }
                    });
                });
            });
        });
      RP_CALI_SUBKERNEL_END("POLYBENCH_MVT_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_MVT_2");
      RAJA::launch<launch_policy>(res,
        RAJA::LaunchParams(RAJA::Teams(reorder_num, 1, blocks_z),
                           RAJA::Threads(block_size)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
          RAJA::loop<teams_z>(ctx, RAJA::RangeSegment(0, blocks_z),
            [&](Index_type bz) {
              RAJA::loop<teams_x>(ctx, RAJA::RangeSegment(0, reorder_num),
                [&](Index_type chiplet) {
                  RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(0, block_size),
                    [&](Index_type tx) {
                      const Index_type i =
                          (blocks_z * chiplet + bz) * block_size + tx;
                      if (i < N) {
                        POLYBENCH_MVT_BODY4_RAJA_LOCAL;
                        POLYBENCH_MVT_UNROLL
                        for (Index_type j = 0; j < N; ++j) {
                          POLYBENCH_MVT_BODY5_RAJA;
                        }
                        POLYBENCH_MVT_BODY6_RAJA;
                      }
                    });
                });
            });
        });
      RP_CALI_SUBKERNEL_END("POLYBENCH_MVT_2");
    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_MVT : Unknown Hip variant id = " << vid << std::endl;
  }
}


template < size_t block_size >
void POLYBENCH_MVT::runHipVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  POLYBENCH_MVT_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(N, block_size);
      constexpr size_t shmem = 0;

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_MVT_1");
      RPlaunchHipKernel( (poly_mvt_1<block_size>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         A, x1, y1, N );
      RP_CALI_SUBKERNEL_END("POLYBENCH_MVT_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_MVT_2");
      RPlaunchHipKernel( (poly_mvt_2<block_size>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         A, x2, y2, N );
      RP_CALI_SUBKERNEL_END("POLYBENCH_MVT_2");

    }
    stopTimer();

  } else if (vid == RAJA_HIP) {

    POLYBENCH_MVT_VIEWS_RAJA;

    using EXEC_POL = RAJA::hip_exec<block_size, true /*async*/>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RAJA::region<RAJA::seq_region>( [=]() {

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

      }); // end sequential region (for single-source code)

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_MVT : Unknown Hip variant id = " << vid << std::endl;
  }
}

void POLYBENCH_MVT::defineHipVariantTunings()
{
  for (VariantID vid : {Base_HIP, RAJA_HIP}) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        if (block_size == 0u) {
          addVariantTuning<&POLYBENCH_MVT::runHipVariantImpl<block_size>>(
              vid, "block_auto", Index_type(0));
        } else {
          addVariantTuning<&POLYBENCH_MVT::runHipVariantImpl<block_size>>(
              vid, "block_"+std::to_string(block_size), Index_type(block_size));
        }

        if (vid == Base_HIP) {
          addVariantTuning<&POLYBENCH_MVT::runHipVariantReorder<block_size, 6>>(
              vid, "reorder6_"+std::to_string(block_size), Index_type(block_size));
        }
        if (vid == RAJA_HIP) {
          addVariantTuning<&POLYBENCH_MVT::runHipVariantReorder<block_size, 6>>(
              vid, "reorder6_"+std::to_string(block_size), Index_type(block_size));
        }

      }

    });

  }
}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
