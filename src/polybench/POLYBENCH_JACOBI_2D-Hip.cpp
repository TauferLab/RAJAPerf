//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_JACOBI_2D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

  //
  // Define thread block shape for Hip execution
  //
#define j_block_sz (32)
#define i_block_sz (block_size / j_block_sz)

#define JACOBI_2D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP \
  j_block_sz, i_block_sz

#define JACOBI_2D_THREADS_PER_BLOCK_HIP \
  dim3 nthreads_per_block(JACOBI_2D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP, 1);

#define JACOBI_2D_NBLOCKS_HIP \
  dim3 nblocks(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(N-2, j_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(N-2, i_block_sz)), \
               static_cast<size_t>(1));


template < size_t j_block_size, size_t i_block_size >
__launch_bounds__(j_block_size*i_block_size)
__global__ void poly_jacobi_2D_1(Real_ptr A, Real_ptr B, Index_type N)
{
  Index_type i = 1 + blockIdx.y * i_block_size + threadIdx.y;
  Index_type j = 1 + blockIdx.x * j_block_size + threadIdx.x;

  if ( i < N-1 && j < N-1 ) {
    POLYBENCH_JACOBI_2D_BODY1;
  }
}

template < size_t j_block_size, size_t i_block_size >
__launch_bounds__(j_block_size*i_block_size)
__global__ void poly_jacobi_2D_2(Real_ptr A, Real_ptr B, Index_type N)
{
  Index_type i = 1 + blockIdx.y * i_block_size + threadIdx.y;
  Index_type j = 1 + blockIdx.x * j_block_size + threadIdx.x;

  if ( i < N-1 && j < N-1 ) {
    POLYBENCH_JACOBI_2D_BODY2;
  }
}

template < size_t j_block_size, size_t i_block_size, typename Lambda >
__launch_bounds__(j_block_size*i_block_size)
__global__ void poly_jacobi_2D_lam(Index_type N, Lambda body)
{
  Index_type i = 1 + blockIdx.y * i_block_size + threadIdx.y;
  Index_type j = 1 + blockIdx.x * j_block_size + threadIdx.x;

  if ( i < N-1 && j < N-1 ) {
    body(i, j);
  }
}


template < size_t block_size, size_t reorder_num >
void POLYBENCH_JACOBI_2D::runHipVariantReorder(VariantID vid)
{
  static_assert(block_size == 256u, "JACOBI_2D reorder tuning requires 256 threads");

  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  POLYBENCH_JACOBI_2D_DATA_SETUP;

  if (vid == RAJA_HIP) {

    POLYBENCH_JACOBI_2D_VIEWS_RAJA;

    constexpr bool async = true;
    using launch_policy =
        RAJA::LaunchPolicy<RAJA::hip_launch_t<async, block_size>>;
    using teams_x = RAJA::LoopPolicy<RAJA::hip_block_x_direct>;
    using teams_y = RAJA::LoopPolicy<RAJA::hip_block_y_direct>;
    using teams_z = RAJA::LoopPolicy<RAJA::hip_block_z_direct>;
    using threads_x =
        RAJA::LoopPolicy<RAJA::hip_thread_size_x_direct<j_block_sz>>;
    using threads_y =
        RAJA::LoopPolicy<RAJA::hip_thread_size_y_direct<i_block_sz>>;

    const Index_type blocks_j = RAJA_DIVIDE_CEILING_INT(N-2, j_block_sz);
    const Index_type blocks_i = RAJA_DIVIDE_CEILING_INT(N-2, i_block_sz);
    const Index_type blocks_z = RAJA_DIVIDE_CEILING_INT(blocks_i, reorder_num);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_2D_1");
      RAJA::launch<launch_policy>(res,
        RAJA::LaunchParams(RAJA::Teams(reorder_num, blocks_j, blocks_z),
                           RAJA::Threads(j_block_sz, i_block_sz)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
          RAJA::loop<teams_z>(ctx, RAJA::RangeSegment(0, blocks_z),
            [&](Index_type bz) {
              RAJA::loop<teams_y>(ctx, RAJA::RangeSegment(0, blocks_j),
                [&](Index_type bj) {
                  RAJA::loop<teams_x>(ctx, RAJA::RangeSegment(0, reorder_num),
                    [&](Index_type chiplet) {
                      RAJA::loop<threads_y>(ctx, RAJA::RangeSegment(0, i_block_sz),
                        [&](Index_type ti) {
                          RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(0, j_block_sz),
                            [&](Index_type tj) {
                              const Index_type i = 1 +
                                  (blocks_z * chiplet + bz) * i_block_sz + ti;
                              const Index_type j = 1 + bj * j_block_sz + tj;
                              if (i < N-1 && j < N-1) {
                                POLYBENCH_JACOBI_2D_BODY1_RAJA;
                              }
                            });
                        });
                    });
                });
            });
        });
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_2D_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_2D_2");
      RAJA::launch<launch_policy>(res,
        RAJA::LaunchParams(RAJA::Teams(reorder_num, blocks_j, blocks_z),
                           RAJA::Threads(j_block_sz, i_block_sz)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
          RAJA::loop<teams_z>(ctx, RAJA::RangeSegment(0, blocks_z),
            [&](Index_type bz) {
              RAJA::loop<teams_y>(ctx, RAJA::RangeSegment(0, blocks_j),
                [&](Index_type bj) {
                  RAJA::loop<teams_x>(ctx, RAJA::RangeSegment(0, reorder_num),
                    [&](Index_type chiplet) {
                      RAJA::loop<threads_y>(ctx, RAJA::RangeSegment(0, i_block_sz),
                        [&](Index_type ti) {
                          RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(0, j_block_sz),
                            [&](Index_type tj) {
                              const Index_type i = 1 +
                                  (blocks_z * chiplet + bz) * i_block_sz + ti;
                              const Index_type j = 1 + bj * j_block_sz + tj;
                              if (i < N-1 && j < N-1) {
                                POLYBENCH_JACOBI_2D_BODY2_RAJA;
                              }
                            });
                        });
                    });
                });
            });
        });
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_2D_2");
    }
    stopTimer();

  } else {
    getCout() << "\n  POLYBENCH_JACOBI_2D : Unknown Hip variant id = "
              << vid << std::endl;
  }
}


template < size_t block_size >
void POLYBENCH_JACOBI_2D::runHipVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  POLYBENCH_JACOBI_2D_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      JACOBI_2D_THREADS_PER_BLOCK_HIP;
      JACOBI_2D_NBLOCKS_HIP;
      constexpr size_t shmem = 0;

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_2D_1");
      RPlaunchHipKernel(
        (poly_jacobi_2D_1<JACOBI_2D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP>),
        nblocks, nthreads_per_block,
        shmem, res.get_stream(),
        A, B, N );
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_2D_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_2D_2");
      RPlaunchHipKernel(
        (poly_jacobi_2D_2<JACOBI_2D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP>),
        nblocks, nthreads_per_block,
        shmem, res.get_stream(),
        A, B, N );
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_2D_2");

    }
    stopTimer();

  } else if ( vid == Lambda_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      JACOBI_2D_THREADS_PER_BLOCK_HIP;
      JACOBI_2D_NBLOCKS_HIP;
      constexpr size_t shmem = 0;

      auto poly_jacobi_2D_1_lambda = [=] __device__ (Index_type i,
                                                     Index_type j) {
        POLYBENCH_JACOBI_2D_BODY1;
      };

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_2D_1");
      RPlaunchHipKernel(
        (poly_jacobi_2D_lam<JACOBI_2D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP,
                            decltype(poly_jacobi_2D_1_lambda)>),
        nblocks, nthreads_per_block,
        shmem, res.get_stream(),
        N, poly_jacobi_2D_1_lambda );
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_2D_1");

      auto poly_jacobi_2D_2_lambda = [=] __device__ (Index_type i,
                                                     Index_type j) {
        POLYBENCH_JACOBI_2D_BODY2;
      };

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_2D_2");
      RPlaunchHipKernel(
        (poly_jacobi_2D_lam<JACOBI_2D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP,
                            decltype(poly_jacobi_2D_2_lambda)>),
        nblocks, nthreads_per_block,
        shmem, res.get_stream(),
        N, poly_jacobi_2D_2_lambda );
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_2D_2");

    }
    stopTimer();

  } else if (vid == RAJA_HIP) {

    POLYBENCH_JACOBI_2D_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelFixedAsync<i_block_sz * j_block_sz,
          RAJA::statement::For<0, RAJA::hip_global_size_y_direct<i_block_sz>,   // i
            RAJA::statement::For<1, RAJA::hip_global_size_x_direct<j_block_sz>, // j
              RAJA::statement::Lambda<0>
            >
          >
        >
      >;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_2D_1");
      RAJA::kernel_resource<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                         RAJA::RangeSegment{1, N-1}),
        res,
        [=] __device__ (Index_type i, Index_type j) {
          POLYBENCH_JACOBI_2D_BODY1_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_2D_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_2D_2");
      RAJA::kernel_resource<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                         RAJA::RangeSegment{1, N-1}),
        res,
        [=] __device__ (Index_type i, Index_type j) {
          POLYBENCH_JACOBI_2D_BODY2_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_2D_2");

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_JACOBI_2D : Unknown Hip variant id = " << vid << std::endl;
  }
}

void POLYBENCH_JACOBI_2D::defineHipVariantTunings()
{
  for (VariantID vid : {Base_HIP, Lambda_HIP, RAJA_HIP}) {
    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {
        if (block_size == 0u) {
          addVariantTuning<&POLYBENCH_JACOBI_2D::runHipVariantImpl<block_size>>(
              vid, "block_auto", Index_type(0));
        } else {
          addVariantTuning<&POLYBENCH_JACOBI_2D::runHipVariantImpl<block_size>>(
              vid, "block_"+std::to_string(block_size), Index_type(block_size));
        }
      }
    });

    if (vid == RAJA_HIP &&
        (run_params.numValidGPUBlockSize() == 0u ||
         run_params.validGPUBlockSize(256u))) {
      addVariantTuning<&POLYBENCH_JACOBI_2D::runHipVariantReorder<256u, 6u>>(
          vid, "reorder6_256", Index_type(256));
    }
  }
}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
