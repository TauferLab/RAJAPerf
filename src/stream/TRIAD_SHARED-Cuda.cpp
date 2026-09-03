//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIAD_SHARED.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace stream {

template <size_t block_size>
__launch_bounds__(block_size) __global__
    void triad_shared_cuda(Real_ptr a, Real_ptr b, Real_ptr c, Real_type alpha,
                           Index_type iend) {
  __shared__ Real_type shared_b[block_size];
  __shared__ Real_type shared_c[block_size];
  const Index_type i = blockIdx.x * block_size + threadIdx.x;

  if (i < iend) {
    shared_b[threadIdx.x] = b[i];
    shared_c[threadIdx.x] = c[i];
  }
  __syncthreads();
  if (i < iend) {
    a[i] = shared_b[threadIdx.x] + alpha * shared_c[threadIdx.x];
  }
}

template <size_t block_size>
void TRIAD_SHARED::runCudaVariantImpl(VariantID vid) {
  setBlockSize(block_size);
  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();
  auto res{getCudaResource()};
  Real_ptr a = m_a;
  Real_ptr b = m_b;
  Real_ptr c = m_c;
  const Real_type alpha = m_alpha;

  if (vid == Base_CUDA) {
    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {
      RP_CALI_SUBKERNEL_BEGIN("TRIAD_SHARED_1");
      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      RPlaunchCudaKernel((triad_shared_cuda<block_size>), grid_size, block_size,
                         0, res.get_stream(), a, b, c, alpha, iend);
      RP_CALI_SUBKERNEL_END("TRIAD_SHARED_1");
    }
    stopTimer();
  } else if (vid == RAJA_CUDA) {
    constexpr bool async = true;
    using launch_policy =
        RAJA::LaunchPolicy<RAJA::cuda_launch_t<async, block_size>>;
    using teams_x = RAJA::LoopPolicy<RAJA::cuda_block_x_direct>;
    using threads_x =
        RAJA::LoopPolicy<RAJA::cuda_thread_size_x_direct<block_size>>;
    const Index_type grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {
      RP_CALI_SUBKERNEL_BEGIN("TRIAD_SHARED_1");
      RAJA::launch<launch_policy>(
          res,
          RAJA::LaunchParams(RAJA::Teams(grid_size), RAJA::Threads(block_size)),
          [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
            RAJA::loop<teams_x>(
                ctx, RAJA::RangeSegment(0, grid_size), [&](Index_type bx) {
                  RAJA_TEAM_SHARED Real_type shared_b[block_size];
                  RAJA_TEAM_SHARED Real_type shared_c[block_size];

                  RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(0, block_size),
                                        [&](Index_type tx) {
                                          const Index_type i =
                                              bx * block_size + tx;
                                          if (i < iend) {
                                            shared_b[tx] = b[i];
                                            shared_c[tx] = c[i];
                                          }
                                        });
                  ctx.teamSync();
                  RAJA::loop<threads_x>(
                      ctx, RAJA::RangeSegment(0, block_size),
                      [&](Index_type tx) {
                        const Index_type i = bx * block_size + tx;
                        if (i < iend) {
                          a[i] = shared_b[tx] + alpha * shared_c[tx];
                        }
                      });
                });
          });
      RP_CALI_SUBKERNEL_END("TRIAD_SHARED_1");
    }
    stopTimer();
  } else {
    getCout() << "\n  TRIAD_SHARED : Unknown Cuda variant id = " << vid
              << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(TRIAD_SHARED, Cuda, Base_CUDA,
                                                  RAJA_CUDA)

} // end namespace stream
} // end namespace rajaperf

#endif
