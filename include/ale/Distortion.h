#ifndef ALE_DISTORTION_H
#define ALE_DISTORTION_H

namespace ale {
  // Keep in the same order as the DistortionType enum in USGSCSM
  // (usgscsm/include/usgscsm/Distortion.h), where these integer values are
  // saved to CSM camera files.
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
