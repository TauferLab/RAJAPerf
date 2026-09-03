//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIAD_SHARED.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf {
namespace stream {

TRIAD_SHARED::TRIAD_SHARED(const RunParams &params)
    : KernelBase(rajaperf::Stream_TRIAD_SHARED, params) {
  setDefaultProblemSize(1000000);
  setDefaultReps(1000);
  setSize(params.getTargetSize(getDefaultProblemSize()),
          params.getReps(getDefaultReps()));

  setChecksumConsistency(ChecksumConsistency::ConsistentPerVariantTuning);
  setChecksumTolerance(ChecksumTolerance::tight);
  setComplexity(Complexity::N);
  setMaxPerfectLoopDimensions(1);
  setProblemDimensionality(1);
  setUsesFeature(Launch);
  setProblemSizeAlignment(ProblemSizeAlignment::OneDimensional);
  addVariantTunings();
}

TRIAD_SHARED::~TRIAD_SHARED() = default;

void TRIAD_SHARED::setSize(Index_type target_size, Index_type target_reps) {
  setActualProblemSize(target_size);
  setRunReps(target_reps);
  setItsPerRep(getActualProblemSize());
  setKernelsPerRep(1);
  setBytesAllocatedPerRep(3 * sizeof(Real_type) * getActualProblemSize());
  setBytesReadPerRep(2 * sizeof(Real_type) * getActualProblemSize());
  setBytesWrittenPerRep(sizeof(Real_type) * getActualProblemSize());
  setBytesModifyWrittenPerRep(0);
  setBytesAtomicModifyWrittenPerRep(0);
  setFLOPsPerRep(2 * getActualProblemSize());
}

void TRIAD_SHARED::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx)) {
  allocAndInitDataConst(m_a, getActualProblemSize(), 0.0, vid);
  allocAndInitData(m_b, getActualProblemSize(), vid);
  allocAndInitData(m_c, getActualProblemSize(), vid);
  initData(m_alpha, vid);
}

void TRIAD_SHARED::updateChecksum(VariantID vid,
                                  size_t RAJAPERF_UNUSED_ARG(tune_idx)) {
  addToChecksum(m_a, getActualProblemSize(), vid);
}

void TRIAD_SHARED::tearDown(VariantID vid,
                            size_t RAJAPERF_UNUSED_ARG(tune_idx)) {
  deallocData(m_a, vid);
  deallocData(m_b, vid);
  deallocData(m_c, vid);
}

} // end namespace stream
} // end namespace rajaperf
