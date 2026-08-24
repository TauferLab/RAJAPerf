//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// POLYBENCH_GEMM kernel reference implementation:
///
/// Note: The dot product is initialized to 0 to avoid
///       excessively large checksums
///
/// for (Index_type i = 0; i < NI; i++) {
///   for (Index_type j = 0; j < NJ; j++) {
///     double dot = 0.0;
///     for (Index_type k = 0; k < NK; k++) {
///       dot += A[i][k] * B[k][j];
///     }
///     C[i][j] = dot;
///   }
/// }


#ifndef RAJAPerf_POLYBENCH_GEMM_HPP
#define RAJAPerf_POLYBENCH_GEMM_HPP

#define POLYBENCH_GEMM_DATA_SETUP \
  const Index_type ni = m_ni; \
  const Index_type nj = m_nj; \
  const Index_type nk = m_nk; \
\
  Real_ptr A = m_A; \
  Real_ptr B = m_B; \
  Real_ptr C = m_C;


#define POLYBENCH_GEMM_BODY1 \
  Real_type dot = 0.0;

#define POLYBENCH_GEMM_BODY3 \
  dot += A[k + i*nk] * B[j + k*nj];

#define POLYBENCH_GEMM_BODY4 \
  C[j + i*nj] = dot;


#define POLYBENCH_GEMM_BODY1_RAJA \
  dot = 0.0;

#define POLYBENCH_GEMM_BODY3_RAJA \
  dot += Aview(i, k) * Bview(k, j);

#define POLYBENCH_GEMM_BODY4_RAJA \
  Cview(i, j) = dot;


// Single-lambda RAJA form. Here the inner reduction runs as an ordinary
// sequential loop inside one lambda, so the accumulator is a lambda-local
// variable rather than a RAJA::Params entry. Unlike the POLYBENCH_GEMM_BODY1_RAJA above,
// which assign to an accumulator the RAJA::kernel_param machinery owns, these
// declare it.

#define POLYBENCH_GEMM_BODY1_RAJA_LOCAL \
  Real_type dot = 0.0;


// Unroll factor applied to the inner sequential reduction in the RAJA_CUDA /
// RAJA_HIP variants. Set explicitly so both toolchains unroll identically:
// ptxas unrolls these loops on its own, the LLVM AMDGPU backend does not, and
// that asymmetry is a confounder in the AMD-vs-NVIDIA comparison.
//
// NOTE: deliberately not RAJAPERF_UNROLL / RAJA_UNROLL_COUNT. Those lower to
// "#pragma GCC unroll" under nvcc + gcc, which nvcc's device compiler ignores,
// so they would unroll the AMD side only.
//
// 8 chosen by sweeping 1/2/4/8/16 on both machines: each target's runtime
// bottoms out at 8 and regresses at 16, so it is the joint optimum rather than
// a compromise.  Note that ptxas's own choice (4) is *not* optimal here -- at 4
// the H100 gives up memory-level parallelism it can still afford, even though
// unrolling costs it occupancy (REG 48 -> 62.5% at 4, REG 56 -> 50% at 8).
// MI300A holds 8 waves/SIMD across the whole range, so only the NVIDIA side
// trades occupancy for MLP.  Neither target spills at 8, and the checksum is
// bitwise identical at every factor.

#ifndef POLYBENCH_GEMM_GPU_UNROLL
#define POLYBENCH_GEMM_GPU_UNROLL 8
#endif

#define POLYBENCH_GEMM_UNROLL \
  RAJAPERF_PRAGMA(unroll POLYBENCH_GEMM_GPU_UNROLL)


#define POLYBENCH_GEMM_VIEWS_RAJA \
  using VIEW_TYPE = RAJA::View<Real_type, \
                               RAJA::Layout<2, Index_type, 1>>; \
\
  VIEW_TYPE Aview(A, RAJA::Layout<2>(ni, nk)); \
  VIEW_TYPE Bview(B, RAJA::Layout<2>(nk, nj)); \
  VIEW_TYPE Cview(C, RAJA::Layout<2>(ni, nj));


#include "common/KernelBase.hpp"

namespace rajaperf
{

class RunParams;

namespace polybench
{

class POLYBENCH_GEMM : public KernelBase
{
public:

  POLYBENCH_GEMM(const RunParams& params);

  ~POLYBENCH_GEMM();

  void setSize(Index_type target_size, Index_type target_reps);
  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void defineSeqVariantTunings();
  void defineOpenMPVariantTunings();
  void defineOpenMPTargetVariantTunings();
  void defineKokkosVariantTunings();
  void defineCudaVariantTunings();
  void defineHipVariantTunings();
  void defineSyclVariantTunings();

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);
  void runKokkosVariant(VariantID vid);

  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);
  template < size_t work_group_size >
  void runSyclVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size,
                                                         integer::MultipleOf<32>>;

  Index_type m_ni_default;
  Index_type m_nj_default;
  Index_type m_nk_default;

  Index_type m_ni;
  Index_type m_nj;
  Index_type m_nk;

  Real_ptr m_A;
  Real_ptr m_B;
  Real_ptr m_C;
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
