from wavelet_noise import utils
import pytest


def test_parse_beamforming_name():
    filename = "CD-ISAE-5deg-53pr-rmp2-6-notape-1.h5"
    params = utils.parse_beamforming_name(filename)
    assert params["aoa"] == 5
    assert (
        params["speed"] == 53 * 1.14
    )  # TODO: do I want to hardcode the conversion?
    assert params["rmp_idx"] == utils.RMP_CONVERT["2-6"]
    assert len(params["rmp_idx"]) == 4
    assert params["notape"] == True

    filename = "CD-ISAE-0deg-14pr-rmp2-6-90s-1.h5"
    params = utils.parse_beamforming_name(filename)
    assert params["aoa"] == 0
    assert params["speed"] == 14 * utils.VELOCITY_FACTOR
    assert params["rmp_idx"] == utils.RMP_CONVERT["2-6"]
    assert params["notape"] == False

    wrong_filename = "invalid-filename.h5"
    with pytest.raises(ValueError):
        utils.parse_beamforming_name(wrong_filename)

    wrong_rmp_code = "CD-ISAE-5deg-53pr-rmp99-100-1.h5"
    with pytest.raises(ValueError):
        utils.parse_beamforming_name(wrong_rmp_code)


def test_beamforming_case():
    path = "tests/data/CD-ISAE-0deg-14pr-rmp2-6-90s-1.h5"
    case = utils.read_beamforming_case(path)
    assert case.aoa == 0
    assert case.speed == 14 * utils.VELOCITY_FACTOR
    assert case.rmp_idx == utils.RMP_CONVERT["2-6"]
    assert case.notape == False
    assert case.microphones.shape[0] == len(case.time)
    assert case.rmp.shape[0] == len(case.time)
    assert case.rmp.shape[1] == len(case.rmp_idx)
    assert case.fs == 1/(case.time[1] - case.time[0])
