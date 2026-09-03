#include "MASS3DPA_NOSHARED.hpp"
#if defined(RAJA_ENABLE_HIP)
#include "common/HipDataUtils.hpp"
#include <iostream>
namespace rajaperf { namespace apps {

template <size_t block_size>
__launch_bounds__(block_size) __global__
void Mass3DPANoShared(const Real_type* __restrict__ B,
                      const Real_type* __restrict__ Bt,
                      const Real_type* __restrict__ D,
                      const Real_type* __restrict__ X,
                      Real_type* __restrict__ Y, Index_type NE)
{
  const Index_type e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e < NE) mass3dpa_noshared::apply(B, Bt, D, X, Y, e);
}

template <size_t block_size, size_t reorder_num>
void MASS3DPA_NOSHARED::runHipVariantImpl(VariantID vid)
{
  setBlockSize(block_size);
  const Index_type run_reps = getRunReps();
  auto res{getHipResource()};
  MASS3DPA_DATA_SETUP;
  constexpr bool reorder = reorder_num > 1u;
  const Index_type blocks = RAJA_DIVIDE_CEILING_INT(NE, Index_type(block_size));
  const Index_type blocks_per_xcd = reorder ? RAJA_DIVIDE_CEILING_INT(blocks, Index_type(reorder_num)) : blocks;
  const Index_type teams = reorder ? Index_type(reorder_num) * blocks_per_xcd : blocks;
  startTimer();
  for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {
    RP_CALI_SUBKERNEL_BEGIN("MASS3DPA_NOSHARED_1");
    if (vid == Base_HIP) {
      const Real_type* B_native = B;
      const Real_type* Bt_native = Bt;
      const Real_type* D_native = D;
      const Real_type* X_native = X;
      RPlaunchHipKernel((Mass3DPANoShared<block_size>), blocks, block_size, 0,
                        res.get_stream(), B_native, Bt_native, D_native,
                        X_native, Y, NE);
    } else if (vid == RAJA_HIP) {
      using launch_policy = RAJA::LaunchPolicy<RAJA::hip_launch_t<true, block_size>>;
      using team_policy = RAJA::LoopPolicy<RAJA::hip_block_x_direct>;
      using thread_policy = RAJA::LoopPolicy<RAJA::hip_thread_x_direct>;
      RAJA::launch<launch_policy>(res,
        RAJA::LaunchParams(RAJA::Teams(teams), RAJA::Threads(block_size)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
          RAJA::loop<team_policy>(ctx, RAJA::RangeSegment(0, teams), [&](Index_type physical_block) {
            const Index_type block = reorder ? blocks_per_xcd * (physical_block % reorder_num) + physical_block / reorder_num : physical_block;
            RAJA::loop<thread_policy>(ctx, RAJA::RangeSegment(0, Index_type(block_size)), [&](Index_type t) {
              const Index_type e = block * block_size + t;
              if (e < NE) {
                const Real_type* RAJA_RESTRICT B_r = B;
                const Real_type* RAJA_RESTRICT Bt_r = Bt;
                const Real_type* RAJA_RESTRICT D_r = D;
                const Real_type* RAJA_RESTRICT X_r = X;
                Real_type* RAJA_RESTRICT Y_r = Y;
                mass3dpa_noshared::apply(B_r, Bt_r, D_r, X_r, Y_r, e);
              }
            });
          });
        });
    } else {
      getCout() << "\n MASS3DPA_NOSHARED : Unknown Hip variant id = " << vid << std::endl;
    }
    RP_CALI_SUBKERNEL_END("MASS3DPA_NOSHARED_1");
  }
  stopTimer();
}

void MASS3DPA_NOSHARED::defineHipVariantTunings()
{
  for (VariantID vid : {Base_HIP, RAJA_HIP})
    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
      if (run_params.numValidGPUBlockSize() == 0u || run_params.validGPUBlockSize(block_size)) {
        addVariantTuning<&MASS3DPA_NOSHARED::runHipVariantImpl<block_size, 1>>(vid, "block_" + std::to_string(block_size), Index_type(block_size));
        if (vid == RAJA_HIP)
          addVariantTuning<&MASS3DPA_NOSHARED::runHipVariantImpl<block_size, 6>>(vid, "reorder6_" + std::to_string(block_size), Index_type(block_size));
      }
    });
}
} }
#endif
