import pytest
import ale
import os
import json

import numpy as np
from ale.formatters.isis_formatter import to_isis
from ale.formatters.formatter import to_isd
from ale.base.data_isis import IsisSpice
from ale.drivers.tgo_drivers import TGOCassisIsisLabelNaifSpiceDriver

import unittest
from unittest.mock import patch, call, PropertyMock

from ale.base.data_naif import NaifSpice

from conftest import get_image_label, get_image_kernels, convert_kernels, compare_dicts, get_isd


@pytest.fixture()
def test_kernels(scope="module"):
    kernels = get_image_kernels("CAS-MCO-2016-11-26T22.32.14.582-RED-01000-B1")
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_cassis_load(test_kernels):
    label_file = get_image_label("CAS-MCO-2016-11-26T22.32.14.582-RED-01000-B1", "isis")
    isd_str = ale.loads(label_file, props={'kernels': test_kernels, 'attach_kernels': False})
    isd_obj = json.loads(isd_str)
    compare_dict = get_isd('cassis')
    print(json.dumps(isd_obj, indent=2))
    assert compare_dicts(isd_obj, compare_dict) == []

# ========= Test cassis ISIS label and naifspice driver =========
class test_cassis_isis_naif(unittest.TestCase):

    def setUp(self):
      label = get_image_label("CAS-MCO-2016-11-26T22.32.14.582-RED-01000-B1", "isis")
      self.driver = TGOCassisIsisLabelNaifSpiceDriver(label)

    def test_short_mission_name(self):
      assert self.driver.short_mission_name == "tgo"

    def test_instrument_id(self):
        assert self.driver.instrument_id == "TGO_CASSIS"

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.tgo_drivers.pyspiceql.utcToEt', side_effect=[(12345, {})]) as utcToEt:
            assert self.driver.ephemeris_start_time == 12345
            calls = [call(utc='2016-11-26 22:32:14.582000', searchKernels=False, useWeb=False)]
            utcToEt.assert_has_calls(calls)
            assert utcToEt.call_count == 1

    def test_sample_summing(self):
        # CaSSIS SummingMode is an enum (0 = 1x1); the label here is 0, so the
        # summing factor must be 1, not the raw 0.
        assert self.driver.sample_summing == 1

    def test_line_summing(self):
        assert self.driver.line_summing == 1


# The CaSSIS focal length, detector geometry, and distortion live only in the ISIS
# addendum (tgoCassisAddendum), not in any NAIF instrument kernel. The driver
# furnishes the addendum and reads these values from the pool, falling back to the
# latest known constants when the addendum is not furnished. The test kernels
# include the addendum, so these exercise the read path.
IMAGE = "CAS-MCO-2016-11-26T22.32.14.582-RED-01000-B1"

def test_focal_length_read_from_addendum(test_kernels):
    label = get_image_label(IMAGE, "isis")
    with TGOCassisIsisLabelNaifSpiceDriver(label, props={'kernels': test_kernels}) as d:
        assert d.focal_length == 874.9

def test_detector_center_read_from_addendum(test_kernels):
    # Boresight 1024.5 from the addendum, converted to the CSM 0-based convention.
    label = get_image_label(IMAGE, "isis")
    with TGOCassisIsisLabelNaifSpiceDriver(label, props={'kernels': test_kernels}) as d:
        assert d.detector_center_sample == 1024.0
        assert d.detector_center_line == 1024.0

def test_distortion_read_from_addendum(test_kernels):
    # OD_A exists in both the NAIF kernel and the addendum with different values;
    # the driver reads the addendum's (ISIS override) first coefficient.
    label = get_image_label(IMAGE, "isis")
    with TGOCassisIsisLabelNaifSpiceDriver(label, props={'kernels': test_kernels}) as d:
        coeffs = d.usgscsm_distortion_model['cassis']['coefficients']
        assert len(coeffs) == 36
        assert coeffs[0] == pytest.approx(0.0037613053094826604)

def test_reads_furnished_addendum_not_fallback(test_kernels, tmp_path, monkeypatch):
    # Furnish a faux addendum with a distinctive focal length and assert the driver
    # returns it, proving it reads the furnished value rather than the fallback.
    # Stop the driver from locating the real addendum, so the faux is the only source.
    monkeypatch.setattr(TGOCassisIsisLabelNaifSpiceDriver,
                        "_isis_cassis_addendum", lambda self: None)
    faux = str(tmp_path / "faux_addendum.ti")
    with open(faux, "w") as f:
        f.write("\\begindata\nINS-143400_FOCAL_LENGTH = ( 999.9 )\n\\begintext\n")
    kernels = [k for k in test_kernels if "Addendum" not in k] + [faux]
    label = get_image_label(IMAGE, "isis")
    with TGOCassisIsisLabelNaifSpiceDriver(label, props={'kernels': kernels}) as d:
        assert d.focal_length == 999.9

def test_focal_length_fallback_without_addendum(test_kernels):
    # Kernels minus the addendum: ikid still resolves from the frame kernel, but
    # FOCAL_LENGTH is absent, so the driver falls back to the latest known value.
    no_addendum = [k for k in test_kernels if "Addendum" not in k]
    label = get_image_label(IMAGE, "isis")
    with TGOCassisIsisLabelNaifSpiceDriver(label, props={'kernels': no_addendum}) as d:
        assert d.focal_length == 874.9

