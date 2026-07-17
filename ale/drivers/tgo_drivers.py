import numpy as np
import scipy.constants
import spiceypy as spice
from pyspiceql import pyspiceql

from ale.base import Driver, WrongInstrumentException
from ale.base.data_naif import NaifSpice
from ale.base.label_isis import IsisLabel
from ale.base.type_sensor import Framer
from ale.base.type_distortion import CassisDistortion

class TGOCassisIsisLabelNaifSpiceDriver(Framer, IsisLabel, NaifSpice, CassisDistortion, Driver):
    """
    Driver for reading TGO Cassis ISIS3 Labels. These are Labels that have been ingested
    into ISIS from PDS EDR images but have not been spiceinit'd yet.
    """
    @property
    def instrument_id(self):
        """
        Returns an instrument id for unquely identifying the instrument, but often
        also used to be piped into Spice Kernels to acquire IKIDs. Therefore they
        the same ID the Spice expects in bods2c calls.
        Expects instrument_id to be defined in the Pds3Label mixin. This should
        be a string of the form CaSSIS

        Returns
        -------
        : str
          instrument id
        """
        id_lookup = {
            'CaSSIS': 'TGO_CASSIS',
        }
        key = super().instrument_id
        if key not in id_lookup:
            raise WrongInstrumentException(f"Unknown instrument id: {key}.")
        return id_lookup[key]

    @property
    def ephemeris_start_time(self):
        """
        Returns the ephemeris_start_time of the image.
        Expects spacecraft_clock_start_count to be defined. This should be a float
        containing the start clock count of the spacecraft.
        Expects spacecraft_id to be defined. This should be the integer Naif ID code
        for the spacecraft.

        Returns
        -------
        : float
          ephemeris start time of the image.
        """
        if not hasattr(self, "_ephemeris_start_time"):
            self._ephemeris_start_time = pyspiceql.utcToEt(utc=self.utc_start_time.strftime("%Y-%m-%d %H:%M:%S.%f"), searchKernels=self.search_kernels, useWeb=self.use_web)[0]
        return self._ephemeris_start_time

    @property
    def sensor_frame_id(self):
        return -143420

    @property
    def focal_length(self):
        """
        Returns the CaSSIS focal length in mm.

        The CaSSIS focal length is defined only in the ISIS addendum kernel
        (tgoCassisAddendum, key INS-143400_FOCAL_LENGTH). No NAIF instrument
        kernel carries it, so the NaifSpice metakernel path cannot read it and it
        would otherwise come back null. Return the value from the latest addendum
        here, following the Cassini driver precedent for addendum-only values.
        This is a versioned calibration (addendum 001-005 were 880.0, 006 was
        876.0, 007 is 874.9); update it if a newer addendum ships.
        """
        return 874.9

    @property
    def light_time_correction(self):
        """
        CaSSIS light-time and aberration correction, INS-143400_LIGHTTIME_CORRECTION
        = LT+S. Addendum-only (no NAIF kernel carries it), so the NaifSpice path
        cannot read it; hardcode it so the sensor_position override below fires.
        """
        return 'LT+S'

    @property
    def correct_lt_to_surface(self):
        """
        CaSSIS corrects light time to the target surface, INS-143400_LT_SURFACE_CORRECT
        = TRUE. Addendum-only, so hardcode it. Without this the NaifSpice path reads
        False, the sensor_position override does not fire, and the CSM camera center
        is biased about 96 m from ISIS.
        """
        return True

    @property
    def swap_observer_target(self):
        """
        CaSSIS swaps observer and target in the state lookup,
        INS-143400_SWAP_OBSERVER_TARGET = TRUE. Addendum-only, so hardcode it to
        match ISIS on the NaifSpice path.
        """
        return True

    @property
    def sensor_model_version(self):
        """
        Returns
        -------
        : int
          ISIS sensor model version
        """
        return 1

    @property
    def sensor_name(self):
        return self.label['IsisCube']['Instrument']['SpacecraftName']

    @property
    def sample_summing(self):
        """
        CaSSIS stores SummingMode as an enum (0 = 1x1, 1 = 2x2, 2 = 4x4), not as
        the summing factor itself. ISIS converts it as summing = sumMode * 2, then
        falls back to 1 when that is 0 (see TgoCassisCamera). Replicate that here,
        otherwise the CSM detector summing becomes 0 and groundToImage diverges.
        """
        sum_mode = self.label['IsisCube']['Instrument']['SummingMode']
        summing = sum_mode * 2
        if summing <= 0:
            summing = 1
        return summing

    @property
    def line_summing(self):
        return self.sample_summing

    @property
    def detector_center_sample(self):
        """
        ISIS uses 0.5-based CCD coordinates (pixel centers at half integers),
        so convert the boresight sample to the CSM 0-based convention by
        subtracting 0.5, as the LRO, MRO, Dawn, MESSENGER, MEX, Kaguya and KPLO
        drivers do. Without this the CSM look is offset from ISIS by half a pixel
        in sample (and half in line), i.e. sqrt(0.5^2+0.5^2) ~ 0.707 px.

        The boresight sample (INS-143400_BORESIGHT_SAMPLE = 1024.5) is defined
        only in the ISIS addendum and cannot be read on the NaifSpice path, so it
        is hardcoded here. It is a stable CCD geometry constant, unlike the focal
        length.
        """
        return 1024.5 - 0.5

    @property
    def detector_center_line(self):
        """
        ISIS uses 0.5-based CCD coordinates; convert to the CSM 0-based
        convention by subtracting 0.5 (see detector_center_sample). The boresight
        line (INS-143400_BORESIGHT_LINE = 1024.5) is addendum-only, so hardcoded.
        """
        return 1024.5 - 0.5

    @property
    def focal2pixel_lines(self):
        """
        Transform from focal-plane millimeters to detector lines,
        INS-143400_ITRANSL. Addendum-only (null on the NaifSpice path), so
        hardcoded. The 100 is 1 / pixel pitch (0.01 mm). Stable CCD geometry.
        """
        return [0.0, 0.0, 100.0]

    @property
    def focal2pixel_samples(self):
        """
        Transform from focal-plane millimeters to detector samples,
        INS-143400_ITRANSS. Addendum-only, hardcoded (see focal2pixel_lines).
        """
        return [0.0, 100.0, 0.0]

    @property
    def usgscsm_distortion_model(self):
        """
        CaSSIS rational (ratio-of-quadratics) distortion, INS-143400_OD_A{1,2,3}_
        {CORR,DIST}. These keywords exist in BOTH the NAIF IK and the ISIS addendum
        with DIFFERENT values; ISIS loads the addendum on top of the IK, so the
        addendum values are the ones ISIS uses. The NaifSpice path here furnishes
        only the IK (the addendum is ISIS-only, and furnishing it breaks driver
        matching), which would otherwise give the wrong, un-overridden IK
        distortion. So return the addendum values directly, from
        tgoCassisAddendum007.ti, to match ISIS and the focal length (also from
        addendum 007). Versioned like the focal length; update if a newer addendum
        ships. Packed A1_corr, A2_corr, A3_corr, A1_dist, A2_dist, A3_dist.
        """
        A1_CORR = [0.0037613053094826604, -0.0134154156065812, -1.8674952100723702e-05,
                   1.00021352681836, -0.00043236237170395304, -0.000948065735350123]
        A2_CORR = [9.9842559363676e-05, 0.00373543707958162, -0.0133299918873929,
                   -0.000215311328389359, 0.995296015537294, -0.0183542717710778]
        A3_CORR = [-3.13320167004204e-05, -7.3565512574980695e-06, -1.57664245066771e-05,
                   0.0037354946543915104, -0.014167194693093499, 1.0]
        A1_DIST = [0.0021365879556062197, -0.00711785765064197, 1.10355974742147e-05,
                   0.573607182625377, 0.000250884350194894, 0.000550623913037132]
        A2_DIST = [-5.6972574101540595e-05, 0.00215155905679149, -0.00716392991767185,
                   0.000124152787728634, 0.5764595443924261, 0.010576940564854]
        A3_DIST = [1.7825077148350597e-05, 4.24592743471094e-06, 9.51220699036653e-06,
                   0.0021515842542073798, -0.0066835595774833, 0.573741540971609]
        return {"cassis": {"coefficients":
                A1_CORR + A2_CORR + A3_CORR + A1_DIST + A2_DIST + A3_DIST}}

    @property
    def sensor_position(self):
        """
        CaSSIS sets LT_SURFACE_CORRECT with LIGHTTIME_CORRECTION=LT+S, so ISIS
        applies the surface light-time correction. The shared
        NaifSpice.sensor_position samples the target body at the raw ephemeris
        time in that branch, but ISIS samples it at the surface-light-time
        adjusted time (ephem - obs_tar_lt + radius_lt); the body moves along its
        orbit during that interval, which otherwise leaves a constant
        tens-of-meters camera-center bias versus ISIS. This override applies that
        for CaSSIS only, so the shared path is unchanged for every other sensor.
        It is the shared surface-light-time branch with the single change that the
        body is sampled at adjusted_time. CaSSIS is a single-record framer, so
        sampling the body at adjusted_time[0]..[-1] is exact.
        """
        if not (self.correct_lt_to_surface
                and self.light_time_correction.upper() == 'LT+S'):
            return super().sensor_position

        if not hasattr(self, '_position'):
            ephem = self.ephemeris_time
            pos = []
            vel = []

            target = self.spacecraft_name
            observer = self.target_name
            if self.swap_observer_target:
                target = self.target_name
                observer = self.spacecraft_name

            ephem_kwargs = {"startEt": ephem[0],
                            "stopEt": ephem[-1],
                            "numRecords": len(ephem),
                            "ckQualities": ["reconstructed"],
                            "spkQualities": ["reconstructed"],
                            "searchKernels": self.search_kernels,
                            "useWeb": self.use_web}

            obs_tars_kwargs = {**ephem_kwargs,
                               "target": target,
                               "observer": observer,
                               "frame": "J2000",
                               "abcorr": self.light_time_correction,
                               "mission": self.spiceql_mission}
            ssb_obs_kwargs = {**ephem_kwargs,
                              "target": observer,
                              "observer": "SSB",
                              "frame": "J2000",
                              "abcorr": "NONE",
                              "mission": self.spiceql_mission}

            obs_tars = pyspiceql.getTargetStatesRanged(**obs_tars_kwargs)[0]
            ssb_obs = pyspiceql.getTargetStatesRanged(**ssb_obs_kwargs)[0]

            obs_tar_lts = np.array(obs_tars)[:, -1]
            ssb_obs_states = np.array(ssb_obs)[:, 0:6]

            radius_lt = (self.target_body_radii[2] + self.target_body_radii[0]) / 2 \
                / (scipy.constants.c / 1000.0)
            adjusted_time = ephem - obs_tar_lts + radius_lt

            # The only change from the shared method: sample the target body at
            # the surface-light-time adjusted time rather than the raw ephem time.
            ssb_tars_kwargs = {**ephem_kwargs,
                               "target": target,
                               "observer": "SSB",
                               "frame": "J2000",
                               "abcorr": "NONE",
                               "mission": self.spiceql_mission}
            ssb_tars_kwargs["startEt"] = adjusted_time[0]
            ssb_tars_kwargs["stopEt"] = adjusted_time[-1]
            ssb_tars = pyspiceql.getTargetStatesRanged(**ssb_tars_kwargs)[0]
            ssb_tar_states = np.array(ssb_tars)[:, 0:6]

            _states = ssb_tar_states - ssb_obs_states

            reference_frame_id = pyspiceql.translateNameToCode(frame=self.reference_frame,
                                                               mission=self.spiceql_mission,
                                                               searchKernels=self.search_kernels,
                                                               useWeb=self.use_web)[0]

            function_args = {**ephem_kwargs,
                             "toFrame": reference_frame_id,
                             "refFrame": 1,
                             "mission": self.spiceql_mission}
            function_args.pop("spkQualities")
            rotations = pyspiceql.getTargetOrientationsRanged(**function_args)[0]

            states = []
            for i, rotation in enumerate(rotations):
                quaternion = rotation[:4]
                av = [0, 0, 0]
                if len(rotation) > 4:
                    av = rotation[4:]
                rotation_matrix = spice.q2m(quaternion)
                matrix = spice.rav2xf(rotation_matrix, av)
                rotated_state = spice.mxvg(matrix, _states[i])
                states.append(rotated_state)

            for state in states:
                if self.swap_observer_target:
                    pos.append(-state[:3])
                    vel.append(-state[3:])
                else:
                    pos.append(state[:3])
                    vel.append(state[3:])

            # SPICE works in km, so convert to m.
            self._position = 1000 * np.asarray(pos)
            self._velocity = 1000 * np.asarray(vel)
            self._ephem = ephem
        return self._position, self._velocity, self._ephem
