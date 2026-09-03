#ifndef RAJAPerf_Apps_MASS3DEA_NOSHARED_HPP
#define RAJAPerf_Apps_MASS3DEA_NOSHARED_HPP

#include "MASS3DEA.hpp"

namespace rajaperf { namespace mass3dea_noshared {
RAJA_HOST_DEVICE RAJA_INLINE
void apply(const Real_type* RAJA_RESTRICT B,
           const Real_type* RAJA_RESTRICT D,
           Real_type* RAJA_RESTRICT M, Index_type e,
           Index_type i1, Index_type i2, Index_type i3)
{
  for (Index_type j1 = 0; j1 < mea::D1D; ++j1)
    for (Index_type j2 = 0; j2 < mea::D1D; ++j2)
      for (Index_type j3 = 0; j3 < mea::D1D; ++j3) {
        Real_type val = 0.0;
        for (Index_type k1 = 0; k1 < mea::Q1D; ++k1)
          for (Index_type k2 = 0; k2 < mea::Q1D; ++k2)
            for (Index_type k3 = 0; k3 < mea::Q1D; ++k3)
              val += B[k1 + mea::Q1D * i1] * B[k1 + mea::Q1D * j1] *
                     B[k2 + mea::Q1D * i2] * B[k2 + mea::Q1D * j2] *
                     B[k3 + mea::Q1D * i3] * B[k3 + mea::Q1D * j3] *
                     D[k1 + mea::Q1D * k2 + mea::Q1D * mea::Q1D * k3 +
                       mea::Q1D * mea::Q1D * mea::Q1D * e];
        MEA_M(i1, i2, i3, j1, j2, j3, e) = val;
      }
}
} } // namespace rajaperf::mass3dea_noshared

namespace rajaperf {
class RunParams;
namespace apps {

class MASS3DEA_NOSHARED : public KernelBase {
public:
  MASS3DEA_NOSHARED(const RunParams& params);
  ~MASS3DEA_NOSHARED();

  void setSize(Index_type target_size, Index_type target_reps);
  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void defineCudaVariantTunings();
  void defineHipVariantTunings();
  void defineSeqVariantTunings() {}
  void defineOpenMPVariantTunings() {}
  void defineOpenMPTargetVariantTunings() {}
  void defineKokkosVariantTunings() {}
  void defineSyclVariantTunings() {}

  template <size_t block_size, size_t tune_idx>
  void runCudaVariantImpl(VariantID vid);
  template <size_t block_size, size_t tune_idx, size_t reorder_num>
  void runHipVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size =
      mea::D1D * mea::D1D * mea::D1D;
  using gpu_block_sizes_type = integer::list_type<default_gpu_block_size>;

  Real_ptr m_B;
  Real_ptr m_D;
  Real_ptr m_M;
  Index_type m_NE;
};

} // namespace apps
} // namespace rajaperf

#endif
