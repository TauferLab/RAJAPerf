//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INTSC_HEXHEX.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include "AppsData.hpp"

#include <iostream>

#define RAJAPERF_HEXHEX_WARPSIZE RAJA_HIP_WAVESIZE
#if RAJAPERF_HEXHEX_WARPSIZE == 32
#define RAJAPERF_HEXHEX_shfl_xor(val,n) __shfl_xor(val,n)
#elif RAJAPERF_HEXHEX_WARPSIZE == 64
#define RAJAPERF_HEXHEX_shfl_xor(val,n) __shfl_xor(val,n)
#else
#error "unexpected RAJA_CUDA_WARPSIZE"
#endif

namespace rajaperf
{
namespace apps
{

template < Size_type block_size >
__launch_bounds__(block_size,3)
__global__ void intsc_hexhex_hip
  ( Real_ptr const dsubz,
    Real_ptr const tsubz,
    Size_type  const nisc_stage,
    Real_ptr vv_int )
{
  __shared__ Real_type vv_reduce[len_vv_reduce] ;

  Index_type blksize = block_size ;   // blocksize = 64  must <= tri_per_pair
  Index_type blk     = blockIdx.x ;
  Index_type ith     = blk*blksize + threadIdx.x ;   // which thread with offset
  Index_type thridx  = threadIdx.x ;

  Real_ptr vv_int_p = (Real_ptr ) vv_int + 8*blk ;

  INTSC_HEXHEX_BODY;
}


template < Size_type block_size >
__global__ void intsc_hexhex_hip_fixup_vv_64to72
    ( Real_ptr const vv_int,   // [8*intsc blks] blocked volumes, moments
      Size_type const n_szpairs,  // number of subzone pairs
      Real_ptr vv_pair )       // [4*n_szpairs] output voluments, moments
{
  Int_type i = blockIdx.x*block_size + threadIdx.x;
  FIXUP_VV_BODY ;
}


template < size_t block_size, size_t reorder_num >
void INTSC_HEXHEX::runHipVariantReorder(VariantID vid)
{
  static_assert(block_size == default_gpu_block_size,
                "INTSC_HEXHEX requires 64-thread blocks");

  const Index_type run_reps = getRunReps();
  const Index_type iend = tri_per_std_intsc * getActualProblemSize();

  const Size_type n_subz_intsc = m_n_subz_intsc;
  const Size_type nisc_stage = n_subz_intsc;
  const Size_type n_szpairs = n_subz_intsc;
  const Size_type n_szgrp =
      RAJA_DIVIDE_CEILING_INT(n_subz_intsc, fixup_groupsize);
  const Size_type gsize_fixup =
      RAJA_DIVIDE_CEILING_INT(n_szgrp, block_size);
  const Index_type iend_fixup = gsize_fixup * block_size;

  auto res{getHipResource()};

  INTSC_HEXHEX_DATA_SETUP;

  const Size_type grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
  const Index_type blocks_z =
      RAJA_DIVIDE_CEILING_INT(grid_size, reorder_num);
  const Index_type fixup_blocks_z =
      RAJA_DIVIDE_CEILING_INT(gsize_fixup, reorder_num);

  if (vid == RAJA_HIP) {

    constexpr bool async = true;
    using launch_policy =
        RAJA::LaunchPolicy<RAJA::hip_launch_t<async, block_size>>;
    using teams_x = RAJA::LoopPolicy<RAJA::hip_block_x_direct>;
    using teams_z = RAJA::LoopPolicy<RAJA::hip_block_z_direct>;
    using threads_x =
        RAJA::LoopPolicy<RAJA::hip_thread_size_x_direct<block_size>>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXHEX_1");
      RAJA::launch<launch_policy>(res,
        RAJA::LaunchParams(RAJA::Teams(reorder_num, 1, blocks_z),
                           RAJA::Threads(block_size)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
          RAJA::loop<teams_z>(ctx, RAJA::RangeSegment(0, blocks_z),
            [&](Index_type bz) {
              RAJA::loop<teams_x>(ctx, RAJA::RangeSegment(0, reorder_num),
                [&](Index_type chiplet) {
                  const Index_type blk = blocks_z * chiplet + bz;
                  if (blk < grid_size) {
                    RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(0, block_size),
                      [&](Index_type thridx) {
                        RAJA_TEAM_SHARED Real_type vv_reduce[len_vv_reduce];
                        const Index_type blksize = block_size;
                        const Index_type ith = blk * blksize + thridx;
                        Real_ptr vv_int_p = vv_int + 8 * blk;
                        INTSC_HEXHEX_BODY;
                      });
                  }
                });
            });
        });
      RP_CALI_SUBKERNEL_END("INTSC_HEXHEX_1");

      RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXHEX_2");
      RAJA::launch<launch_policy>(res,
        RAJA::LaunchParams(RAJA::Teams(reorder_num, 1, fixup_blocks_z),
                           RAJA::Threads(block_size)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
          RAJA::loop<teams_z>(ctx, RAJA::RangeSegment(0, fixup_blocks_z),
            [&](Index_type bz) {
              RAJA::loop<teams_x>(ctx, RAJA::RangeSegment(0, reorder_num),
                [&](Index_type chiplet) {
                  const Index_type blk = fixup_blocks_z * chiplet + bz;
                  if (blk < gsize_fixup) {
                    RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(0, block_size),
                      [&](Index_type thridx) {
                        const Index_type i = blk * block_size + thridx;
                        if (i < iend_fixup) {
                          FIXUP_VV_BODY;
                        }
                      });
                  }
                });
            });
        });
      RP_CALI_SUBKERNEL_END("INTSC_HEXHEX_2");
    }
    stopTimer();

  } else {
    getCout() << "\n  INTSC_HEXHEX : Unknown Hip variant id = " << vid
              << std::endl;
  }
}


template < Size_type block_size >
void INTSC_HEXHEX::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0 ;
  const Index_type iend     = tri_per_std_intsc * getActualProblemSize() ;

  const Size_type  n_subz_intsc= m_n_subz_intsc;
  const Size_type  nisc_stage  = m_n_subz_intsc ;

  // n_szgrp is number of groups of subzone pairs in fixup kernel.
  // gsize_fixup = fixup kernel grid size (1 thread per group of subzone pairs)
  // iend_fixup = number of threads for fixup kernel.
  //      Kernel has bounds check to mask out excess threads.

  const Size_type  n_szgrp     =
      RAJA_DIVIDE_CEILING_INT(n_subz_intsc, fixup_groupsize)  ;
  const Size_type  gsize_fixup = RAJA_DIVIDE_CEILING_INT(n_szgrp, block_size) ;
  const Index_type iend_fixup  = gsize_fixup * block_size ;

  const Size_type  n_szpairs   = n_subz_intsc ;

  auto res{getHipResource()};

  INTSC_HEXHEX_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      const Size_type grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr Size_type shmem = 0;

      RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXHEX_1");
      RPlaunchHipKernel( (intsc_hexhex_hip<block_size>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         m_dsubz, m_tsubz,
                         n_subz_intsc, m_vv_int ) ;
      RP_CALI_SUBKERNEL_END("INTSC_HEXHEX_1");

      RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXHEX_2");
      RPlaunchHipKernel( (intsc_hexhex_hip_fixup_vv_64to72<block_size>),
                         gsize_fixup, block_size,
                         shmem, res.get_stream(),
                         m_vv_int, n_subz_intsc, m_vv_out ) ;
      RP_CALI_SUBKERNEL_END("INTSC_HEXHEX_2");


    }
    stopTimer();

  } else if ( vid == Lambda_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      const Size_type grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

      auto intsc_hexhex_lambda = [=] __device__ ( Index_type i ) {

          __shared__ Real_type vv_reduce[len_vv_reduce] ;  // len_vv_reduce=16

          Index_type blksize   = block_size ;
          Index_type blk       = i / block_size ;
          Index_type ith       = i ;
          Index_type thridx    = i % block_size ;

          Real_ptr vv_int_p = (Real_ptr ) vv_int + 8*blk ;
          INTSC_HEXHEX_BODY ;
      } ;

      auto intsc_hexhex_fixup_lambda = [=] __device__ ( Index_type i ) {
          FIXUP_VV_BODY ;
      } ;

      constexpr Size_type shmem = 0;

      RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXHEX_1");
      RPlaunchHipKernel( (lambda_hip_forall<block_size,
                          decltype(intsc_hexhex_lambda)>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         ibegin, iend,
                         intsc_hexhex_lambda );
      RP_CALI_SUBKERNEL_END("INTSC_HEXHEX_1");

      RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXHEX_2");
      RPlaunchHipKernel( (lambda_hip_forall<block_size,
                          decltype(intsc_hexhex_fixup_lambda)>),
                         gsize_fixup, block_size,
                         shmem, res.get_stream(),
                         ibegin, iend_fixup,
                         intsc_hexhex_fixup_lambda );
      RP_CALI_SUBKERNEL_END("INTSC_HEXHEX_2");


    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXHEX_1");
      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i)
          {
            RAJA_TEAM_SHARED Real_type vv_reduce[len_vv_reduce] ;

            Index_type blksize   = block_size ;
            Index_type blk       = i / block_size ;
            Index_type ith       = i ;
            Index_type thridx    = i % block_size ;

            Real_ptr vv_int_p = (Real_ptr ) vv_int + 8*blk ;
            INTSC_HEXHEX_BODY;
          }
      ) ;
      RP_CALI_SUBKERNEL_END("INTSC_HEXHEX_1");

      RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXHEX_2");
      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend_fixup), [=] __device__ (Index_type i)
          {
            FIXUP_VV_BODY ;
          }
      ) ;
      RP_CALI_SUBKERNEL_END("INTSC_HEXHEX_2");

    }
    stopTimer();

  } else {
     getCout() << "\n  INTSC_HEXHEX : Unknown Hip variant id = " << vid << std::endl;
  }
}

void INTSC_HEXHEX::defineHipVariantTunings()
{
  for (VariantID vid : {Base_HIP, Lambda_HIP, RAJA_HIP}) {
    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {
        if (block_size == 0u) {
          addVariantTuning<&INTSC_HEXHEX::runHipVariantImpl<block_size>>(
              vid, "block_auto", Index_type(0));
        } else {
          addVariantTuning<&INTSC_HEXHEX::runHipVariantImpl<block_size>>(
              vid, "block_"+std::to_string(block_size), Index_type(block_size));
        }
      }
    });

    if (vid == RAJA_HIP &&
        (run_params.numValidGPUBlockSize() == 0u ||
         run_params.validGPUBlockSize(64u))) {
      addVariantTuning<&INTSC_HEXHEX::runHipVariantReorder<64u, 6u>>(
          vid, "reorder6_64", Index_type(64));
    }
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
