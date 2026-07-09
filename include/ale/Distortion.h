#ifndef ALE_DISTORTION_H
#define ALE_DISTORTION_H

namespace ale {
  // The order and integer values of these enumerators must match the
  // DistortionType enum in USGSCSM (usgscsm/include/usgscsm/Distortion.h) and
  // must only ever be extended by appending new types at the end. USGSCSM
  // serializes the selected type as this integer in the model state, so
  // reordering or inserting in the middle would silently reinterpret every
  // model state already written to disk. ALE never serializes this integer
  // (it emits the distortion by name in the ISD), so aligning the ALE order to
  // USGSCSM is safe on the ALE side and removes a long-standing mismatch where
  // RADIAL and TRANSVERSE were swapped between the two enums.
  enum DistortionType {
    RADIAL,
    TRANSVERSE,
    KAGUYALISM,
    DAWNFC,
    LROLROCNAC,
    CAHVOR,
    LUNARORBITER,
    RADTAN,
    KPLOSHADOWCAM,
    CASSIS
  };
}

#endif
