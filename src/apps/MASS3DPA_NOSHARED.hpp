#ifndef RAJAPerf_Apps_MASS3DPA_NOSHARED_HPP
#define RAJAPerf_Apps_MASS3DPA_NOSHARED_HPP

#include "MASS3DPA.hpp"

namespace rajaperf { namespace mass3dpa_noshared {
RAJA_HOST_DEVICE RAJA_INLINE
void apply(const Real_type* RAJA_RESTRICT B,
           const Real_type* RAJA_RESTRICT Bt,
           const Real_type* RAJA_RESTRICT D,
           const Real_type* RAJA_RESTRICT X,
           Real_type* RAJA_RESTRICT Y, Index_type e)
{
  constexpr Index_type MD1 = mpa::D1D;
  constexpr Index_type MQ1 = mpa::Q1D;
  constexpr Index_type MDQ = MQ1 > MD1 ? MQ1 : MD1;
  static_assert(MDQ * MDQ * MDQ <= 8,
                "MASS3DPA_NOSHARED private arrays exceed eight entries");
  Real_type a[MDQ * MDQ * MDQ];
  Real_type b[MDQ * MDQ * MDQ];

  for (Index_type dz = 0; dz < MD1; ++dz)
    for (Index_type dy = 0; dy < MD1; ++dy)
      for (Index_type dx = 0; dx < MD1; ++dx)
        a[dz * MD1 * MD1 + dy * MD1 + dx] =
            X[e * MD1 * MD1 * MD1 + dz * MD1 * MD1 + dy * MD1 + dx];

  for (Index_type dz = 0; dz < MD1; ++dz)
    for (Index_type dy = 0; dy < MD1; ++dy)
      for (Index_type qx = 0; qx < MQ1; ++qx) {
        Real_type v = 0.0;
        for (Index_type dx = 0; dx < MD1; ++dx)
          v += a[dz * MD1 * MD1 + dy * MD1 + dx] * B[qx + MQ1 * dx];
        b[dz * MD1 * MQ1 + dy * MQ1 + qx] = v;
      }
  for (Index_type dz = 0; dz < MD1; ++dz)
    for (Index_type qy = 0; qy < MQ1; ++qy)
      for (Index_type qx = 0; qx < MQ1; ++qx) {
        Real_type v = 0.0;
        for (Index_type dy = 0; dy < MD1; ++dy)
          v += b[dz * MD1 * MQ1 + dy * MQ1 + qx] * B[qy + MQ1 * dy];
        a[dz * MQ1 * MQ1 + qy * MQ1 + qx] = v;
      }
  for (Index_type qz = 0; qz < MQ1; ++qz)
    for (Index_type qy = 0; qy < MQ1; ++qy)
      for (Index_type qx = 0; qx < MQ1; ++qx) {
        Real_type v = 0.0;
        for (Index_type dz = 0; dz < MD1; ++dz)
          v += a[dz * MQ1 * MQ1 + qy * MQ1 + qx] * B[qz + MQ1 * dz];
        b[qz * MQ1 * MQ1 + qy * MQ1 + qx] =
            v * D[e * MQ1 * MQ1 * MQ1 + qz * MQ1 * MQ1 + qy * MQ1 + qx];
      }
  for (Index_type qz = 0; qz < MQ1; ++qz)
    for (Index_type qy = 0; qy < MQ1; ++qy)
      for (Index_type dx = 0; dx < MD1; ++dx) {
        Real_type v = 0.0;
        for (Index_type qx = 0; qx < MQ1; ++qx)
          v += b[qz * MQ1 * MQ1 + qy * MQ1 + qx] * Bt[qx + MD1 * dx];
        a[qz * MQ1 * MD1 + qy * MD1 + dx] = v;
      }
  for (Index_type qz = 0; qz < MQ1; ++qz)
    for (Index_type dy = 0; dy < MD1; ++dy)
      for (Index_type dx = 0; dx < MD1; ++dx) {
        Real_type v = 0.0;
        for (Index_type qy = 0; qy < MQ1; ++qy)
          v += a[qz * MQ1 * MD1 + qy * MD1 + dx] * Bt[qy + MD1 * dy];
        b[qz * MD1 * MD1 + dy * MD1 + dx] = v;
      }
  for (Index_type dz = 0; dz < MD1; ++dz)
    for (Index_type dy = 0; dy < MD1; ++dy)
      for (Index_type dx = 0; dx < MD1; ++dx) {
        Real_type v = 0.0;
        for (Index_type qz = 0; qz < MQ1; ++qz)
          v += b[qz * MD1 * MD1 + dy * MD1 + dx] * Bt[qz + MD1 * dz];
        Y[e * MD1 * MD1 * MD1 + dz * MD1 * MD1 + dy * MD1 + dx] += v;
      }
}
} } // namespace rajaperf::mass3dpa_noshared

namespace rajaperf {
class RunParams;
namespace apps {

class MASS3DPA_NOSHARED : public KernelBase {
public:
  MASS3DPA_NOSHARED(const RunParams& params);
  ~MASS3DPA_NOSHARED();

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

  template <size_t block_size> void runCudaVariantImpl(VariantID vid);
  template <size_t block_size, size_t reorder_num>
  void runHipVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 64;
  using gpu_block_sizes_type =
      integer::make_gpu_block_size_list_type<default_gpu_block_size>;

  Real_ptr m_B;
  Real_ptr m_Bt;
  Real_ptr m_D;
  Real_ptr m_X;
  Real_ptr m_Y;
  Index_type m_NE;
};

} // namespace apps
} // namespace rajaperf

#endif
