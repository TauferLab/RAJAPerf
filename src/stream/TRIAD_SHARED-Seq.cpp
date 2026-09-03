//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIAD_SHARED.hpp"

#include <iostream>

namespace rajaperf {
namespace stream {

void TRIAD_SHARED::runSeqVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();
  Real_ptr a = m_a;
  Real_ptr b = m_b;
  Real_ptr c = m_c;
  const Real_type alpha = m_alpha;

  if (vid == Base_Seq) {
    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {
      RP_CALI_SUBKERNEL_BEGIN("TRIAD_SHARED_1");
      for (Index_type i = 0; i < iend; ++i) {
        a[i] = b[i] + alpha * c[i];
      }
      RP_CALI_SUBKERNEL_END("TRIAD_SHARED_1");
    }
    stopTimer();
  } else {
    getCout() << "\n  TRIAD_SHARED : Unknown variant id = " << vid << std::endl;
  }
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(TRIAD_SHARED, Seq, Base_Seq)

} // end namespace stream
} // end namespace rajaperf
