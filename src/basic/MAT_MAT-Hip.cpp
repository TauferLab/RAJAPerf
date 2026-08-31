//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_MAT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace basic {

template < Index_type tile_x, Index_type tile_y, Index_type tile_size >
  __launch_bounds__(tile_x*tile_y)
__global__ void mat_mat(Index_type N, Real_ptr C, Real_ptr A,
                        Real_ptr B) {

  Index_type tx = threadIdx.x;
  Index_type ty = threadIdx.y;
  Index_type bx = blockIdx.x;
  Index_type by = blockIdx.y;

  MAT_MAT_BODY_1_SHAPE(tile_y, tile_x)

  for (Index_type k = 0; k < (tile_size + N - 1) / tile_size; k++) {

    MAT_MAT_BODY_2(tile_size)
  }

  MAT_MAT_BODY_3(tile_size)
}

template < Index_type tile_x, Index_type tile_y, Index_type tile_size >
  __launch_bounds__(tile_x*tile_y)
__global__ void mat_mat_reorder(Index_type N, Real_ptr C, Real_ptr A,
                                Real_ptr B) {

  Index_type tx = threadIdx.x;
  Index_type ty = threadIdx.y;
  Index_type bx = blockIdx.y;
  Index_type by = (gridDim.z * blockIdx.x + blockIdx.z);

  MAT_MAT_BODY_1_SHAPE(tile_y, tile_x)

  for (Index_type k = 0; k < (tile_size + N - 1) / tile_size; k++) {

    MAT_MAT_BODY_2(tile_size)
  }

  MAT_MAT_BODY_3(tile_size)
}

template < size_t block_size, size_t block_x, size_t reorder_num >
void MAT_MAT::runHipVariantReorder(VariantID vid)
{
  setBlockSize(block_size);

  constexpr Index_type tile_size = integer::sqrt(block_size);
  constexpr Index_type tile_x = block_x;
  constexpr Index_type tile_y = block_size / block_x;
  static_assert(tile_x*tile_y == block_size, "Invalid block shape");

  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;

  dim3 blockDim(tile_x, tile_y, 1);
  int blocks_y = static_cast<int>(RAJA_DIVIDE_CEILING_INT(N, blockDim.y));
  dim3 gridDim(reorder_num,
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(N, blockDim.x)),
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(blocks_y, reorder_num)));
  constexpr size_t shmem = 0;

  auto res{getHipResource()};

  MAT_MAT_DATA_SETUP;

  if (vid == Base_HIP) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("BASIC_MAT_MAT_1");
      RPlaunchHipKernel( (mat_mat_reorder<tile_x, tile_y, tile_size>),
                         gridDim, blockDim,
                         shmem, res.get_stream(),
                         N, C, A, B );
      RP_CALI_SUBKERNEL_END("BASIC_MAT_MAT_1");
    }
    stopTimer();

  } else {
    getCout() << "\n  MAT_MAT : Unknown Hip variant id = " << vid
              << std::endl;
  }
}

template < size_t block_size, size_t block_x >
void MAT_MAT::runHipVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  // Keep the original square k tile while varying the output block shape.
  constexpr Index_type tile_size = integer::sqrt(block_size);
  constexpr Index_type tile_x = block_x;
  constexpr Index_type tile_y = block_size / block_x;
  static_assert(tile_x*tile_y == block_size, "Invalid block shape");

  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;

  dim3 blockDim(tile_x, tile_y);
  dim3 gridDim(RAJA_DIVIDE_CEILING_INT(N, blockDim.x),
               RAJA_DIVIDE_CEILING_INT(N, blockDim.y));
  constexpr size_t shmem = 0;

  const Index_type Nx = gridDim.x;
  const Index_type Ny = gridDim.y;

  auto res{getHipResource()};

  MAT_MAT_DATA_SETUP;

  if (vid == Base_HIP) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("BASIC_MAT_MAT_1");
      RPlaunchHipKernel( (mat_mat<tile_x, tile_y, tile_size>),
                         gridDim, blockDim,
                         shmem, res.get_stream(),
                         N, C, A, B );
      RP_CALI_SUBKERNEL_END("BASIC_MAT_MAT_1");
    }
    stopTimer();

  } else if (vid == Lambda_HIP) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("BASIC_MAT_MAT_1");
      auto mat_mat_lambda = [=] __device__() {

        auto outer_y = [&](Index_type by) {
          auto outer_x = [&](Index_type bx) {

            auto inner_y = [&](Index_type ty) {
              auto inner_x = [&](Index_type tx) {

                MAT_MAT_BODY_1_SHAPE(tile_y, tile_x)

                for (Index_type k = 0; k < (tile_size + N - 1) / tile_size; ++k) {
                  MAT_MAT_BODY_2(tile_size)
                }

                MAT_MAT_BODY_3(tile_size)
              };

              {
                Index_type tx = threadIdx.x;
                if (tx < tile_x)
                  inner_x(tx);
              }
            };

            {
              Index_type ty = threadIdx.y;
              if (ty < tile_y)
                inner_y(ty);
            }
          }; // outer_x

          {
            Index_type bx = blockIdx.x;
            if(bx < Nx) outer_x(bx);
          }
        };

        {
          Index_type by = blockIdx.y;
          if(by < Ny) outer_y(by);
        }
      };

      RPlaunchHipKernel( (lambda_hip<tile_x*tile_y,
                                     decltype(mat_mat_lambda)>),
                         gridDim, blockDim,
                         shmem, res.get_stream(),
                         mat_mat_lambda );
      RP_CALI_SUBKERNEL_END("BASIC_MAT_MAT_1");
    }
    stopTimer();

  } else if (vid == RAJA_HIP) {

    constexpr bool async = true;

    using launch_policy = RAJA::LaunchPolicy<RAJA::hip_launch_t<async, tile_x*tile_y>>;

    using teams_x = RAJA::LoopPolicy<RAJA::hip_block_x_direct>;

    using teams_y = RAJA::LoopPolicy<RAJA::hip_block_y_direct>;

    using threads_x = RAJA::LoopPolicy<RAJA::hip_thread_size_x_direct<tile_x>>;

    using threads_y = RAJA::LoopPolicy<RAJA::hip_thread_size_y_direct<tile_y>>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("BASIC_MAT_MAT_1");
      RAJA::launch<launch_policy>( res,
        RAJA::LaunchParams(RAJA::Teams(Nx, Ny),
                         RAJA::Threads(tile_x, tile_y)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<teams_y>(ctx, RAJA::RangeSegment(0, Ny),
            [&](Index_type by) {
              RAJA::loop<teams_x>(ctx, RAJA::RangeSegment(0, Nx),
                [&](Index_type bx) {

                  RAJA::loop<threads_y>(ctx, RAJA::RangeSegment(0, tile_y),
                    [&](Index_type ty) {
                      RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(0, tile_x),
                        [&](Index_type tx) {

                          MAT_MAT_BODY_1_SHAPE(tile_y, tile_x)

                          for (Index_type k = 0; k < (tile_size + N - 1) / tile_size; k++) {
                            MAT_MAT_BODY_2_PEELED(tile_size)
                          }  // for (k)

                          MAT_MAT_BODY_3(tile_size)
                        }
                      );  // RAJA::loop<threads_x>
                    }
                  );  // RAJA::loop<threads_y>

                }  // lambda (bx)
              );  // RAJA::loop<teams_x>
            }  // lambda (by)
          );  // RAJA::loop<teams_y>

        }  // outer lambda (ctx)
      );  // RAJA::launch
      RP_CALI_SUBKERNEL_END("BASIC_MAT_MAT_1");

    }  // loop over kernel reps
    stopTimer();

  } else {
    getCout() << "\n  MAT_MAT : Unknown Hip variant id = " << vid
              << std::endl;
  }
}

void MAT_MAT::defineHipVariantTunings()
{
  using block_shapes = camp::list<camp::integral_constant<size_t, 8>,
                                  camp::integral_constant<size_t, 16>,
                                  camp::integral_constant<size_t, 32>,
                                  camp::integral_constant<size_t, 64> >;

  for (VariantID vid : {Base_HIP, Lambda_HIP, RAJA_HIP}) {
    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {
        seq_for(block_shapes{}, [&](auto block_x) {
          constexpr size_t bsz = decltype(block_size)::value;
          constexpr size_t bx  = decltype(block_x)::value;
          if constexpr (bsz > 0u && bx > 0u && bx <= bsz && (bsz % bx) == 0u) {
            addVariantTuning<&MAT_MAT::runHipVariantImpl<bsz, bx>>(
                vid, "block_"+std::to_string(bsz)+"_"+
                     std::to_string(bx)+"x"+std::to_string(bsz/bx),
                Index_type(bsz));

            if (vid == Base_HIP && bx == 16u) {
              addVariantTuning<&MAT_MAT::runHipVariantReorder<bsz, bx, 6>>(
                  vid, "reorder6_"+std::to_string(bsz));
            }
          }
        });
      }
    });
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif // RAJA_ENABLE_HIP
