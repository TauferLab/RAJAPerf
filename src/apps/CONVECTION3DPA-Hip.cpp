//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "CONVECTION3DPA.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

template < size_t block_size >
  __launch_bounds__(block_size)
__global__ void Convection3DPA(const Real_ptr Basis, const Real_ptr tBasis,
                              const Real_ptr dBasis, const Real_ptr D,
                              const Real_ptr X, Real_ptr Y) {

  const Index_type e = blockIdx.x;

  CONVECTION3DPA_0_GPU;

  GPU_FOREACH_THREAD_INC(dz, z, conv::D1D, conv::Q1D)
  {
    GPU_FOREACH_THREAD_INC(dy, y, conv::D1D, conv::Q1D)
    {
      GPU_FOREACH_THREAD_INC(dx, x, conv::D1D, conv::Q1D)
      {
        CONVECTION3DPA_1;
      }
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD_INC(dz, z, conv::D1D, conv::Q1D)
  {
    GPU_FOREACH_THREAD_INC(dy, y, conv::D1D, conv::Q1D)
    {
      GPU_FOREACH_THREAD_INC(qx, x, conv::Q1D, conv::Q1D)
      {
        CONVECTION3DPA_2;
      }
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD_INC(dz, z, conv::D1D, conv::Q1D)
  {
    GPU_FOREACH_THREAD_INC(qx, x, conv::Q1D, conv::Q1D)
    {
      GPU_FOREACH_THREAD_INC(qy, y, conv::Q1D, conv::Q1D)
      {
        CONVECTION3DPA_3;
      }
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD_INC(qx, x, conv::Q1D, conv::Q1D)
  {
    GPU_FOREACH_THREAD_INC(qy, y, conv::Q1D, conv::Q1D)
    {
      GPU_FOREACH_THREAD_INC(qz, z, conv::Q1D, conv::Q1D)
      {
        CONVECTION3DPA_4;
      }
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD_INC(qz, z, conv::Q1D, conv::Q1D)
  {
    GPU_FOREACH_THREAD_INC(qy, y, conv::Q1D, conv::Q1D)
    {
      GPU_FOREACH_THREAD_INC(qx, x, conv::Q1D, conv::Q1D)
      {
        CONVECTION3DPA_5;
      }
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD_INC(qx, x, conv::Q1D, conv::Q1D)
  {
    GPU_FOREACH_THREAD_INC(qy, y, conv::Q1D, conv::Q1D)
    {
      GPU_FOREACH_THREAD_INC(dz, z, conv::D1D, conv::Q1D)
      {
        CONVECTION3DPA_6;
      }
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD_INC(dz, z, conv::D1D, conv::Q1D)
  {
    GPU_FOREACH_THREAD_INC(qx, x, conv::Q1D, conv::Q1D)
    {
      GPU_FOREACH_THREAD_INC(dy, y, conv::D1D, conv::Q1D)
      {
        CONVECTION3DPA_7;
      }
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD_INC(dz, z, conv::D1D, conv::Q1D)
  {
    GPU_FOREACH_THREAD_INC(dy, y, conv::D1D, conv::Q1D)
    {
      GPU_FOREACH_THREAD_INC(dx, x, conv::D1D, conv::Q1D)
      {
        CONVECTION3DPA_8;
      }
    }
  }

}

template < size_t block_size, size_t reorder_num >
void CONVECTION3DPA::runHipVariantImpl(VariantID vid) {
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  CONVECTION3DPA_DATA_SETUP;

  switch (vid) {

  case Base_HIP: {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("CONVECTION3DPA_1");
      dim3 nthreads_per_block(conv::Q1D, conv::Q1D, conv::Q1D);
      constexpr size_t shmem = 0;
      
      RPlaunchHipKernel( (Convection3DPA<block_size>),
                         NE, nthreads_per_block,
                         shmem, res.get_stream(),
                         Basis, tBasis, dBasis, D, X, Y );      
      RP_CALI_SUBKERNEL_END("CONVECTION3DPA_1");
    }
    stopTimer();

    break;
  }

  case RAJA_HIP: {

    constexpr bool reorder = reorder_num > 1u;
    const Index_type blocks_per_xcd =
        reorder ? RAJA_DIVIDE_CEILING_INT(NE, reorder_num) : NE;
    const Index_type num_teams = reorder ? reorder_num * blocks_per_xcd : NE;

    constexpr bool async = true;

    using launch_policy =
        RAJA::LaunchPolicy<RAJA::hip_launch_t<async, conv::Q1D*conv::Q1D*conv::Q1D>>;

    using outer_x =
        RAJA::LoopPolicy<RAJA::hip_block_x_direct>;

    using inner_x =
        RAJA::LoopPolicy<RAJA::hip_thread_size_x_loop<conv::Q1D>>;

    using inner_y =
        RAJA::LoopPolicy<RAJA::hip_thread_size_y_loop<conv::Q1D>>;

    using inner_z =
        RAJA::LoopPolicy<RAJA::hip_thread_size_z_loop<conv::Q1D>>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("CONVECTION3DPA_1");
      //clang-format off
      RAJA::launch<launch_policy>( res,
          RAJA::LaunchParams(RAJA::Teams(num_teams),
                           RAJA::Threads(conv::Q1D, conv::Q1D, conv::Q1D)),
          [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, num_teams),
            [&](Index_type physical_e) {

             const Index_type e = reorder
                 ? blocks_per_xcd * (physical_e % reorder_num) +
                       physical_e / reorder_num
                 : physical_e;
             if (e < NE) {

             CONVECTION3DPA_0_GPU;

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::D1D),
                        [&](Index_type dx) {

                          CONVECTION3DPA_1;

                        } // lambda (dx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                        [&](Index_type qx) {

                          CONVECTION3DPA_2;

                        } // lambda (dx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

            ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                    [&](Index_type qx) {
                      RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                        [&](Index_type qy) {

                          CONVECTION3DPA_3;

                        } // lambda (dy)
                      ); // RAJA::loop<inner_y>
                    } // lambda (dx)
                  );  //RAJA::loop<inner_x>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

            ctx.teamSync();

              RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                [&](Index_type qx) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                    [&](Index_type qy) {
                      RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                        [&](Index_type qz) {

                          CONVECTION3DPA_4;

                        } // lambda (qz)
                      ); // RAJA::loop<inner_z>
                    } // lambda (qy)
                  );  //RAJA::loop<inner_y>
                } // lambda (qx)
              );  //RAJA::loop<inner_x>

            ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                [&](Index_type qz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                    [&](Index_type qy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                        [&](Index_type qx) {

                          CONVECTION3DPA_5;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (qy)
                  );  //RAJA::loop<inner_y>
                } // lambda (qz)
              );  //RAJA::loop<inner_z>

            ctx.teamSync();

              RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                [&](Index_type qx) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                    [&](Index_type qy) {
                      RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::D1D),
                        [&](Index_type dz) {

                          CONVECTION3DPA_6;

                        } // lambda (dz)
                      ); // RAJA::loop<inner_z>
                    } // lambda (qy)
                  );  //RAJA::loop<inner_y>
                } // lambda (qx)
              );  //RAJA::loop<inner_x>

            ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                    [&](Index_type qx) {
                      RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::D1D),
                        [&](Index_type dy) {

                          CONVECTION3DPA_7;

                        } // lambda (dy)
                      ); // RAJA::loop<inner_y>
                    } // lambda (qx)
                  );  //RAJA::loop<inner_x>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

            ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::D1D),
                        [&](Index_type dx) {

                          CONVECTION3DPA_8;

                        } // lambda (dx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

             }
            } // lambda (physical_e)
          ); // RAJA::loop<outer_x>

        }  // outer lambda (ctx)
      );  // RAJA::launch
      //clang-format on
      RP_CALI_SUBKERNEL_END("CONVECTION3DPA_1");

    } // loop over kernel reps
    stopTimer();

    break;
  }

  default: {

    getCout() << "\n CONVECTION3DPA : Unknown Hip variant id = " << vid
              << std::endl;
    break;
  }
  }
}

void CONVECTION3DPA::defineHipVariantTunings()
{
  for (VariantID vid : {Base_HIP, RAJA_HIP}) {
    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {
        addVariantTuning<&CONVECTION3DPA::runHipVariantImpl<block_size, 1>>(
            vid, "block_"+std::to_string(block_size), Index_type(block_size));
        if (vid == RAJA_HIP) {
          addVariantTuning<&CONVECTION3DPA::runHipVariantImpl<block_size, 6>>(
              vid, "reorder6_"+std::to_string(block_size), Index_type(block_size));
        }
      }
    });
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_HIP
