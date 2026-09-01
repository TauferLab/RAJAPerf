//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GEMVER.hpp"

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

#define GEMVER_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP \
  j_block_sz, i_block_sz

#define GEMVER_THREADS_PER_BLOCK_HIP \
  dim3 nthreads_per_block1(GEMVER_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP, 1);

#define GEMVER_NBLOCKS_HIP \
  dim3 nblocks1(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(n, j_block_sz)), \
                static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(n, i_block_sz)), \
                static_cast<size_t>(1));


template < size_t j_block_size, size_t i_block_size >
__launch_bounds__(j_block_size*i_block_size)
__global__ void poly_gemver_1(Real_ptr A,
                              Real_ptr u1, Real_ptr v1,
                              Real_ptr u2, Real_ptr v2,
                              Index_type n)
{
  Index_type i = blockIdx.y * i_block_size + threadIdx.y;
  Index_type j = blockIdx.x * j_block_size + threadIdx.x;

  if (i < n && j < n) {
    POLYBENCH_GEMVER_BODY1;
  }
}

template < size_t j_block_size, size_t i_block_size, typename Lambda >
__launch_bounds__(j_block_size*i_block_size)
__global__ void poly_gemver_1_lam(Index_type n, Lambda body)
{
  Index_type i = blockIdx.y * i_block_size + threadIdx.y;
  Index_type j = blockIdx.x * j_block_size + threadIdx.x;

  if (i < n && j < n) {
    body(i, j);
  }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void poly_gemver_2(Real_ptr A,
                              Real_ptr x, Real_ptr y,
                              Real_type beta,
                              Index_type n)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < n) {
    POLYBENCH_GEMVER_BODY2;
    for (Index_type j = 0; j < n; ++j) {
      POLYBENCH_GEMVER_BODY3;
    }
    POLYBENCH_GEMVER_BODY4;
  }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void poly_gemver_3(Real_ptr x, Real_ptr z,
                              Index_type n)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < n) {
    POLYBENCH_GEMVER_BODY5;
  }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void poly_gemver_4(Real_ptr A,
                              Real_ptr x, Real_ptr w,
                              Real_type alpha,
                              Index_type n)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < n) {
    POLYBENCH_GEMVER_BODY6;
    for (Index_type j = 0; j < n; ++j) {
      POLYBENCH_GEMVER_BODY7;
    }
    POLYBENCH_GEMVER_BODY8;
  }
}

template < size_t block_size, typename Lambda >
__launch_bounds__(block_size)
__global__ void poly_gemver_234_lam(Index_type n, Lambda body)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < n) {
    body(i);
  }
}


template < size_t j_block_size, size_t i_block_size >
__launch_bounds__(j_block_size*i_block_size)
__global__ void poly_gemver_1_reorder(Real_ptr A,
                                      Real_ptr u1, Real_ptr v1,
                                      Real_ptr u2, Real_ptr v2,
                                      Index_type n)
{
  Index_type i = (gridDim.z * blockIdx.x + blockIdx.z) * i_block_size + threadIdx.y;
  Index_type j = blockIdx.y * j_block_size + threadIdx.x;

  if (i < n && j < n) {
    POLYBENCH_GEMVER_BODY1;
  }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void poly_gemver_2_reorder(Real_ptr A,
                                      Real_ptr x, Real_ptr y,
                                      Real_type beta,
                                      Index_type n)
{
  Index_type i = (gridDim.z * blockIdx.x + blockIdx.z) * block_size + threadIdx.x;
  if (i < n) {
    POLYBENCH_GEMVER_BODY2;
    for (Index_type j = 0; j < n; ++j) {
      POLYBENCH_GEMVER_BODY3;
    }
    POLYBENCH_GEMVER_BODY4;
  }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void poly_gemver_3_reorder(Real_ptr x, Real_ptr z,
                                      Index_type n)
{
  Index_type i = (gridDim.z * blockIdx.x + blockIdx.z) * block_size + threadIdx.x;
  if (i < n) {
    POLYBENCH_GEMVER_BODY5;
  }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void poly_gemver_4_reorder(Real_ptr A,
                                      Real_ptr x, Real_ptr w,
                                      Real_type alpha,
                                      Index_type n)
{
  Index_type i = (gridDim.z * blockIdx.x + blockIdx.z) * block_size + threadIdx.x;
  if (i < n) {
    POLYBENCH_GEMVER_BODY6;
    for (Index_type j = 0; j < n; ++j) {
      POLYBENCH_GEMVER_BODY7;
    }
    POLYBENCH_GEMVER_BODY8;
  }
}


template < size_t block_size, size_t reorder_num >
void POLYBENCH_GEMVER::runHipVariantReorder(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  POLYBENCH_GEMVER_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      dim3 nthreads_per_block1(j_block_sz, i_block_sz, 1);
      int blocks_i = static_cast<int>(RAJA_DIVIDE_CEILING_INT(n, i_block_sz));
      dim3 nblocks1(reorder_num,
                    static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(n, j_block_sz)),
                    static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(blocks_i, reorder_num)));
      constexpr size_t shmem = 0;

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_1");
      RPlaunchHipKernel(
        (poly_gemver_1_reorder<j_block_sz, i_block_sz>),
        nblocks1, nthreads_per_block1,
        shmem, res.get_stream(),
        A, u1, v1, u2, v2, n );
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_1");

      dim3 nthreads_per_block(block_size, 1, 1);
      int blocks = static_cast<int>(RAJA_DIVIDE_CEILING_INT(n, block_size));
      dim3 nblocks(reorder_num,
                   1,
                   static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(blocks, reorder_num)));

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_2");
      RPlaunchHipKernel( (poly_gemver_2_reorder<block_size>),
                         nblocks, nthreads_per_block,
                         shmem, res.get_stream(),
                         A, x, y, beta, n );
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_2");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_3");
      RPlaunchHipKernel( (poly_gemver_3_reorder<block_size>),
                         nblocks, nthreads_per_block,
                         shmem, res.get_stream(),
                         x, z, n );
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_3");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_4");
      RPlaunchHipKernel( (poly_gemver_4_reorder<block_size>),
                         nblocks, nthreads_per_block,
                         shmem, res.get_stream(),
                         A, x, w, alpha, n );
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_4");

    }
    stopTimer();

  } else if (vid == RAJA_HIP) {

    POLYBENCH_GEMVER_VIEWS_RAJA;

    constexpr bool async = true;
    using launch_policy =
        RAJA::LaunchPolicy<RAJA::hip_launch_t<async, block_size>>;
    using teams_x = RAJA::LoopPolicy<RAJA::hip_block_x_direct>;
    using teams_y = RAJA::LoopPolicy<RAJA::hip_block_y_direct>;
    using teams_z = RAJA::LoopPolicy<RAJA::hip_block_z_direct>;
    using threads_x_2d =
        RAJA::LoopPolicy<RAJA::hip_thread_size_x_direct<j_block_sz>>;
    using threads_y_2d =
        RAJA::LoopPolicy<RAJA::hip_thread_size_y_direct<i_block_sz>>;
    using threads_x =
        RAJA::LoopPolicy<RAJA::hip_thread_size_x_direct<block_size>>;

    const Index_type blocks_j = RAJA_DIVIDE_CEILING_INT(n, j_block_sz);
    const Index_type blocks_i = RAJA_DIVIDE_CEILING_INT(n, i_block_sz);
    const Index_type blocks_z_2d =
        RAJA_DIVIDE_CEILING_INT(blocks_i, reorder_num);
    const Index_type blocks = RAJA_DIVIDE_CEILING_INT(n, block_size);
    const Index_type blocks_z = RAJA_DIVIDE_CEILING_INT(blocks, reorder_num);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_1");
      RAJA::launch<launch_policy>(res,
        RAJA::LaunchParams(RAJA::Teams(reorder_num, blocks_j, blocks_z_2d),
                           RAJA::Threads(j_block_sz, i_block_sz)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
          RAJA::loop<teams_z>(ctx, RAJA::RangeSegment(0, blocks_z_2d),
            [&](Index_type bz) {
              RAJA::loop<teams_y>(ctx, RAJA::RangeSegment(0, blocks_j),
                [&](Index_type bj) {
                  RAJA::loop<teams_x>(ctx, RAJA::RangeSegment(0, reorder_num),
                    [&](Index_type chiplet) {
                      RAJA::loop<threads_y_2d>(ctx, RAJA::RangeSegment(0, i_block_sz),
                        [&](Index_type ti) {
                          RAJA::loop<threads_x_2d>(ctx, RAJA::RangeSegment(0, j_block_sz),
                            [&](Index_type tj) {
                              const Index_type i =
                                  (blocks_z_2d * chiplet + bz) * i_block_sz + ti;
                              const Index_type j = bj * j_block_sz + tj;
                              if (i < n && j < n) {
                                POLYBENCH_GEMVER_BODY1_RAJA;
                              }
                            });
                        });
                    });
                });
            });
        });
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_2");
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
                      if (i < n) {
                        POLYBENCH_GEMVER_BODY2_RAJA_LOCAL;
                        POLYBENCH_GEMVER_UNROLL
                        for (Index_type j = 0; j < n; ++j) {
                          POLYBENCH_GEMVER_BODY3_RAJA;
                        }
                        POLYBENCH_GEMVER_BODY4_RAJA;
                      }
                    });
                });
            });
        });
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_2");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_3");
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
                      if (i < n) {
                        POLYBENCH_GEMVER_BODY5_RAJA;
                      }
                    });
                });
            });
        });
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_3");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_4");
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
                      if (i < n) {
                        POLYBENCH_GEMVER_BODY6_RAJA_LOCAL;
                        POLYBENCH_GEMVER_UNROLL
                        for (Index_type j = 0; j < n; ++j) {
                          POLYBENCH_GEMVER_BODY7_RAJA;
                        }
                        POLYBENCH_GEMVER_BODY8_RAJA;
                      }
                    });
                });
            });
        });
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_4");
    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_GEMVER : Unknown Hip variant id = " << vid << std::endl;
  }
}


template < size_t block_size >
void POLYBENCH_GEMVER::runHipVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  POLYBENCH_GEMVER_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      GEMVER_THREADS_PER_BLOCK_HIP;
      GEMVER_NBLOCKS_HIP;
      constexpr size_t shmem = 0;

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_1");
      RPlaunchHipKernel(
        (poly_gemver_1<GEMVER_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP>),
        nblocks1, nthreads_per_block1,
        shmem, res.get_stream(),
        A, u1, v1, u2, v2, n );
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_1");

      size_t grid_size = RAJA_DIVIDE_CEILING_INT(m_n, block_size);

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_2");
      RPlaunchHipKernel( (poly_gemver_2<block_size>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         A, x, y, beta, n );
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_2");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_3");
      RPlaunchHipKernel( (poly_gemver_3<block_size>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         x, z, n );
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_3");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_4");
      RPlaunchHipKernel( (poly_gemver_4<block_size>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         A, x, w, alpha, n );
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_4");

    }
    stopTimer();

  } else if ( vid == Lambda_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      GEMVER_THREADS_PER_BLOCK_HIP;
      GEMVER_NBLOCKS_HIP;
      constexpr size_t shmem = 0;

      auto poly_gemver1_lambda = [=] __device__ (Index_type i, Index_type j) {
        POLYBENCH_GEMVER_BODY1;
      };

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_1");
      RPlaunchHipKernel(
       (poly_gemver_1_lam<GEMVER_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP,
                          decltype(poly_gemver1_lambda)>),
       nblocks1, nthreads_per_block1,
       shmem, res.get_stream(),
       n, poly_gemver1_lambda );
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_1");

      size_t grid_size = RAJA_DIVIDE_CEILING_INT(n, block_size);

      auto poly_gemver2_lambda = [=] __device__ (Index_type i) {
        POLYBENCH_GEMVER_BODY2;
        for (Index_type j = 0; j < n; ++j) {
          POLYBENCH_GEMVER_BODY3;
        }
        POLYBENCH_GEMVER_BODY4;
      };

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_2");
      RPlaunchHipKernel( (poly_gemver_234_lam<block_size,
                                              decltype(poly_gemver2_lambda)>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         n, poly_gemver2_lambda );
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_2");

      auto poly_gemver3_lambda = [=] __device__ (Index_type i) {
        POLYBENCH_GEMVER_BODY5;
      };

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_3");
      RPlaunchHipKernel( (poly_gemver_234_lam<block_size,
                                              decltype(poly_gemver3_lambda)>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         n, poly_gemver3_lambda );
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_3");

      auto poly_gemver4_lambda = [=] __device__ (Index_type i) {
        POLYBENCH_GEMVER_BODY6;
        for (Index_type j = 0; j < n; ++j) {
          POLYBENCH_GEMVER_BODY7;
        }
        POLYBENCH_GEMVER_BODY8;
      };

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_4");
      RPlaunchHipKernel( (poly_gemver_234_lam<block_size,
                                              decltype(poly_gemver4_lambda)>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         n, poly_gemver4_lambda );
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_4");

    }
    stopTimer();

  } else if (vid == RAJA_HIP) {

    POLYBENCH_GEMVER_VIEWS_RAJA;

    using EXEC_POL1 =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelFixedAsync<i_block_sz * j_block_sz,
          RAJA::statement::For<0, RAJA::hip_global_size_y_direct<i_block_sz>,   // i
            RAJA::statement::For<1, RAJA::hip_global_size_x_direct<j_block_sz>, // j
              RAJA::statement::Lambda<0>
            >
          >
        >
      >;

    using EXEC_POL3 = RAJA::hip_exec<block_size, true /*async*/>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_1");
      RAJA::kernel_resource<EXEC_POL1>(
        RAJA::make_tuple(RAJA::RangeSegment{0, n},
                         RAJA::RangeSegment{0, n}),
        res,
        [=] __device__ (Index_type i, Index_type j) {
          POLYBENCH_GEMVER_BODY1_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_2");
      RAJA::forall<EXEC_POL3> ( res, RAJA::RangeSegment{0, n},
        [=] __device__ (Index_type i) {
          POLYBENCH_GEMVER_BODY2_RAJA_LOCAL;
          POLYBENCH_GEMVER_UNROLL
          for (Index_type j = 0; j < n; ++j ) {
            POLYBENCH_GEMVER_BODY3_RAJA;
          }
          POLYBENCH_GEMVER_BODY4_RAJA;
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_2");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_3");
      RAJA::forall<EXEC_POL3> ( res, RAJA::RangeSegment{0, n},
        [=] __device__ (Index_type i) {
          POLYBENCH_GEMVER_BODY5_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_3");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_4");
      RAJA::forall<EXEC_POL3> ( res, RAJA::RangeSegment{0, n},
        [=] __device__ (Index_type i) {
          POLYBENCH_GEMVER_BODY6_RAJA_LOCAL;
          POLYBENCH_GEMVER_UNROLL
          for (Index_type j = 0; j < n; ++j ) {
            POLYBENCH_GEMVER_BODY7_RAJA;
          }
          POLYBENCH_GEMVER_BODY8_RAJA;
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_4");

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_GEMVER : Unknown Hip variant id = " << vid << std::endl;
  }
}

void POLYBENCH_GEMVER::defineHipVariantTunings()
{
  for (VariantID vid : {Base_HIP, Lambda_HIP, RAJA_HIP}) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        if (block_size == 0u) {
          addVariantTuning<&POLYBENCH_GEMVER::runHipVariantImpl<block_size>>(
              vid, "block_auto", Index_type(0));
        } else {
          addVariantTuning<&POLYBENCH_GEMVER::runHipVariantImpl<block_size>>(
              vid, "block_"+std::to_string(block_size), Index_type(block_size));
        }

        if (vid == Base_HIP) {
          addVariantTuning<&POLYBENCH_GEMVER::runHipVariantReorder<block_size, 6>>(
              vid, "reorder6_"+std::to_string(block_size), Index_type(block_size));
        }
        if (vid == RAJA_HIP) {
          addVariantTuning<&POLYBENCH_GEMVER::runHipVariantReorder<block_size, 6>>(
              vid, "reorder6_"+std::to_string(block_size), Index_type(block_size));
        }

      }

    });

  }
}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
