//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FEMSWEEP.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void FEMSweep3D( const Real_ptr Bdat,
                            const Real_ptr Adat,
                            const Real_ptr Fdat,
                            Real_ptr Xdat,
                            const Real_ptr Sgdat,
                            const Real_ptr M0dat,
                            const Index_type ne,
                            const Index_type ng,
                            const Index_type sharedinteriorfaces,
                            const Index_ptr nhpaa_r,
                            const Index_ptr ohpaa_r,
                            const Index_ptr phpaa_r,
                            const Index_ptr order_r,
                            const Index_ptr AngleElem2FaceType,
                            const Index_ptr elem_to_faces,
                            const Index_ptr F_g2l,
                            const Index_ptr idx1,
                            const Index_ptr idx2 )
{
  const Index_type a = blockIdx.y;
  const Index_type g = blockIdx.x;
  FEMSWEEP_KERNEL_SETUP;
  Index_type nehp_pos = 0;
  for (Index_type hp = 0; hp < nhp; ++hp)
  {
    const Index_type nehp = phpaa_r[ohp + hp];
    for (Index_type k = threadIdx.x; k < nehp; k += block_size)
    {
      FEMSWEEP_KERNEL_HYPERPLANE_ELEMENT;
    }
    __syncthreads();
    nehp_pos += nehp;
  }
}

template < size_t block_size, size_t reorder_num >
void FEMSWEEP::runHipVariantReorder(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  FEMSWEEP_DATA_SETUP;

  if (vid == RAJA_HIP) {

    constexpr bool async = true;
    using launch_policy =
        RAJA::LaunchPolicy<RAJA::hip_launch_t<async, block_size>>;
    using teams_x = RAJA::LoopPolicy<RAJA::hip_block_x_direct>;
    using teams_y = RAJA::LoopPolicy<RAJA::hip_block_y_direct>;
    using teams_z = RAJA::LoopPolicy<RAJA::hip_block_z_direct>;
    using threads_x =
        RAJA::LoopPolicy<RAJA::hip_thread_size_x_loop<block_size>>;

    const Index_type blocks_z = RAJA_DIVIDE_CEILING_INT(na, reorder_num);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("FEMSWEEP_1");
      RAJA::launch<launch_policy>(res,
        RAJA::LaunchParams(RAJA::Teams(reorder_num, ng, blocks_z),
                           RAJA::Threads(block_size)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
          RAJA::loop<teams_z>(ctx, RAJA::RangeSegment(0, blocks_z),
            [&](Index_type bz) {
              RAJA::loop<teams_y>(ctx, RAJA::RangeSegment(0, ng),
                [&](Index_type g) {
                  RAJA::loop<teams_x>(ctx, RAJA::RangeSegment(0, reorder_num),
                    [&](Index_type chiplet) {
                      const Index_type a = blocks_z * chiplet + bz;
                      if (a < na) {
                        FEMSWEEP_KERNEL_SETUP;
                        Index_type nehp_pos = 0;
                        for (Index_type hp = 0; hp < nhp; ++hp)
                        {
                          const Index_type nehp = phpaa_r[ohp + hp];
                          RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(0, nehp),
                            [&](Index_type k) {
                              FEMSWEEP_KERNEL_HYPERPLANE_ELEMENT;
                            });
                          ctx.teamSync();
                          nehp_pos += nehp;
                        }
                      }
                    });
                });
            });
        });
      RP_CALI_SUBKERNEL_END("FEMSWEEP_1");
    }
    stopTimer();

  } else {
    getCout() << "\n FEMSWEEP : Unknown HIP variant id = " << vid
              << std::endl;
  }
}

template < size_t block_size >
void FEMSWEEP::runHipVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  FEMSWEEP_DATA_SETUP;

  switch ( vid ) {

    case Base_HIP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

         RP_CALI_SUBKERNEL_BEGIN("FEMSWEEP_1");
         const dim3 grid_size(ng, na);
         constexpr size_t shmem = 0;

         RPlaunchHipKernel( (FEMSweep3D<block_size>),
                            grid_size, block_size,
                            shmem, res.get_stream(),
                            Bdat,
                            Adat,
                            Fdat,
                            Xdat,
                            Sgdat,
                            M0dat,
                            ne,
                            ng,
                            sharedinteriorfaces,
                            nhpaa_r,
                            ohpaa_r,
                            phpaa_r,
                            order_r,
                            AngleElem2FaceType,
                            elem_to_faces,
                            F_g2l,
                            idx1,
                            idx2 );
         RP_CALI_SUBKERNEL_END("FEMSWEEP_1");

      }
      stopTimer();

      break;
    }

    case RAJA_HIP : {

      constexpr bool async = true;

      using launch_policy =
          RAJA::LaunchPolicy<RAJA::hip_launch_t<async, block_size>>;

      using outer_y =
          RAJA::LoopPolicy<RAJA::hip_block_y_direct_unchecked>;

      using outer_x =
          RAJA::LoopPolicy<RAJA::hip_block_x_direct_unchecked>;

      using inner_x =
          RAJA::LoopPolicy<RAJA::hip_thread_size_x_loop<block_size>>;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

         RP_CALI_SUBKERNEL_BEGIN("FEMSWEEP_1");
         RAJA::launch<launch_policy>( res,
             RAJA::LaunchParams(RAJA::Teams(ng, na),
                                RAJA::Threads(block_size)),
             [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
           RAJA::loop<outer_y>(ctx, RAJA::RangeSegment(0, na),
               [&](Index_type a) {
             RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, ng),
                 [&](Index_type g) {
               FEMSWEEP_KERNEL_SETUP;
               Index_type nehp_pos = 0;
               for (Index_type hp = 0; hp < nhp; ++hp)
               {
                 const Index_type nehp = phpaa_r[ohp + hp];
                 RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, nehp),
                     [&](Index_type k) {
                   FEMSWEEP_KERNEL_HYPERPLANE_ELEMENT;
                 });  // k loop
                 ctx.teamSync();
                 nehp_pos += nehp;
               }
             });  // g loop
           });  // a loop
         });  // RAJA Launch
         RP_CALI_SUBKERNEL_END("FEMSWEEP_1");

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n FEMSWEEP : Unknown HIP variant id = " << vid << std::endl;
    }

  }

}

void FEMSWEEP::defineHipVariantTunings()
{
  for (VariantID vid : {Base_HIP, RAJA_HIP}) {
    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {
        if (block_size == 0u) {
          addVariantTuning<&FEMSWEEP::runHipVariantImpl<block_size>>(
              vid, "block_auto", Index_type(0));
        } else {
          addVariantTuning<&FEMSWEEP::runHipVariantImpl<block_size>>(
              vid, "block_"+std::to_string(block_size), Index_type(block_size));
        }
      }
    });

    if (vid == RAJA_HIP &&
        (run_params.numValidGPUBlockSize() == 0u ||
         run_params.validGPUBlockSize(256u))) {
      addVariantTuning<&FEMSWEEP::runHipVariantReorder<256u, 6u>>(
          vid, "reorder6_256", Index_type(256));
    }
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_HIP
