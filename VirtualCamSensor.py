"""
Convert a linear Rec.2020 RGB EXR to Bayer-raw DNG with a simulated virtual camera.
requirement: python 3.12+
pip install "tifffile>=2026.7.14" "imagecodecs>=2026.6.26" numpy OpenEXR imageio
Expects input EXR to be linear Rec.2020 RGB.
"""

import argparse
import glob
import os
import sys
import numpy as np
from fractions import Fraction


# all user parameters
USER_PARAMETERS = {
    # Input filters: Rec.709 primaries expressed in Rec.2020 RGB space
    # Rec.2020 primaries are spectral, 630 nm (red), 532 nm (green), and 467 nm (blue)
    # Therefore, expressing Rec.709 RGB directly in Rec.2020 counts as a pseudo spectral reconstruction.
    "r_filter": [0.627404, 0.069097, 0.016391],
    "g_filter": [0.329283, 0.919540, 0.088013],
    "b_filter": [0.043313, 0.011362, 0.895595],

    # Bayer pattern
    "pattern": "RGGB",

    # Sensor model (stops relative to 0.18 middle gray)
    "min_stop": -10.0,
    "max_stop": 4.0,

    # None = auto: integer bit depths use adaptive middle gray (upper bound pinned
    # at file_max, middle gray shifts darker to preserve higher-range content),
    # float32 uses fixed middle gray at 0.18. True/False override per-bit-depth default.
    # But note actual behavior depends on downstream software, for example, ACR normalizes anyway.
    "fixed_middle_gray": None,

    # Camera ISO. Higher ISO = more gain = more noise.
    # Gain = base_gain * (iso / native_iso)
    "native_iso": 100,
    "iso": 100,

    "sensor_sensitivity_weighting": [1.0, 1.0, 1.0],

    # Sensor black level (DN for zero open-domain linear signal)
    "sensor_black_level": 256,

    # DNG output levels (will be set to fit full sensor DR [min_stop, max_stop])
    # These are computed automatically if not specified
    "white_level": None,
    "black_level": None,
    # available: 10, 12, 14, 16, 32
    "bit_depth": 32,

    # Power encoding for adaptive range. Encode the linear signal with a power
    # curve (x^1/exponent) to reduce quantization artifacts in lower ranges.
    # Only takes effect when fixed_middle_gray is off (adaptive range).
    # IMPORTANT: requires the downstream RAW editor to support DNG LinearizationTable 
    # for correct decoding.
    "pow_encode_int": True,
    "power_exponent": 2.6,

    # Noise model
    "read_noise": 1.5,
    "shot_noise": True,
    "no_shot_noise": False,
    # from 0.0 to 1.0
    "noise_level": 0.5,
}


def compute_cm1(r_filter, g_filter, b_filter):
    """
    Compute ColorMatrix1 (XYZ to Camera Native)

    1. Rec.2020 primary chromaticities (x, y) with Y=1:
       R = (0.708, 0.292), G = (0.170, 0.797), B = (0.131, 0.046)
    2. D65 white point: (0.3127, 0.3290)
    3. Solve for scaling factors s = [sr, sg, sb] such that
       [sr*Xr, sg*Xg, sb*Xb] * [1, 1, 1]^T = [Xw, Yw, Zw]
    4. M_XYZ_from_R2020 = [sr*Xr, sg*Xg, sb*Xb;
                           sr*Yr, sg*Yg, sb*Yb;
                           sr*Zr, sg*Zg, sb*Zb]  (3x3)
    5. Filter matrix F (3x3): rows = r_filter, g_filter, b_filter in Rec.2020 RGB
    6. M_cam_from_XYZ = F @ inv(M_XYZ_from_R2020)  (3x3)
    7. CM1 = M_cam_from_XYZ in DNG rational format (row-major, 18 rationals)
    """
    # Rec.2020 primary chromaticities (x, y) with Y=1
    # Rec.2020 primaries are spectral, 630 nm (red), 532 nm (green), and 467 nm (blue)
    r_xy = np.array([0.708, 0.292])
    g_xy = np.array([0.170, 0.797])
    b_xy = np.array([0.131, 0.046])

    # D65 white point
    white_xy = np.array([0.3127, 0.3290])

    # Convert (x, y) to XYZ with Y=1
    def xy_to_XYZ(xy):
        x, y = xy
        X = x / y
        Y = 1.0
        Z = (1 - x - y) / y
        return np.array([X, Y, Z])

    Xr, Yr, Zr = xy_to_XYZ(r_xy)
    Xg, Yg, Zg = xy_to_XYZ(g_xy)
    Xb, Yb, Zb = xy_to_XYZ(b_xy)
    Xw, Yw, Zw = xy_to_XYZ(white_xy)

    # Solve for scaling factors: M @ s = w
    M = np.array([
        [Xr, Xg, Xb],
        [Yr, Yg, Yb],
        [Zr, Zg, Zb]
    ])
    w = np.array([Xw, Yw, Zw])
    s = np.linalg.solve(M, w)
    sr, sg, sb = s

    # M_XYZ_from_R2020 (columns are scaled primaries)
    M_XYZ_from_R2020 = np.array([
        [sr * Xr, sg * Xg, sb * Xb],
        [sr * Yr, sg * Yg, sb * Yb],
        [sr * Zr, sg * Zg, sb * Zb]
    ])

    # Filter matrix F (rows = r, g, b filters in Rec.2020 RGB)
    F = np.array([r_filter, g_filter, b_filter])

    # M_cam_from_XYZ = F @ inv(M_XYZ_from_R2020)
    M_cam_from_XYZ = F @ np.linalg.inv(M_XYZ_from_R2020)

    # Convert to DNG rational format (18 rationals, row-major)
    # Each rational is (numerator, denominator)
    rationals = []
    for row in M_cam_from_XYZ:
        for val in row:
            frac = Fraction(val).limit_denominator(10000)
            rationals.append(frac.numerator)
            rationals.append(frac.denominator)

    return tuple(rationals)


def read_exr(path, expect_rgb=True):
    try:
        import OpenEXR
        import Imath
        f = OpenEXR.InputFile(path)
        dw = f.header()['dataWindow']
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        chans = f.header()['channels']

        def read_ch(name):
            return np.frombuffer(f.channel(name, pt), dtype=np.float32).reshape(h, w)

        if 'R' in chans and 'G' in chans and 'B' in chans and expect_rgb:
            r = read_ch('R')
            g = read_ch('G')
            b = read_ch('B')
            return np.stack([r, g, b], axis=-1)
        elif 'Y' in chans:
            return read_ch('Y')
        elif 'R' in chans:
            return read_ch('R')
        elif 'Data' in chans:
            return read_ch('Data')
        else:
            first = list(chans.keys())[0]
            return read_ch(first)

    except ImportError:
        pass

    try:
        import imageio.v3 as iio
        img = iio.imread(path)
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if expect_rgb and img.ndim == 3 and img.shape[-1] >= 3:
            return img[..., :3]
        elif img.ndim == 3:
            return img[..., 0]
        return img
    except Exception as e:
        raise RuntimeError(
            "Cannot read EXR. Install OpenEXR:  pip install OpenEXR"
        ) from e


# ------------------------------------------------------------------
# Bayer Filtering
# ------------------------------------------------------------------
def filter_rgb_to_scalar(rgb, r_filter, g_filter, b_filter, sensor_weight):
    # Effective response = filter(λ) * sensor(λ), integrated over all λ
    r_plane = np.dot(rgb, r_filter * sensor_weight)
    g_plane = np.dot(rgb, g_filter * sensor_weight)
    b_plane = np.dot(rgb, b_filter * sensor_weight)
    return r_plane, g_plane, b_plane


# ------------------------------------------------------------------
# Interleave Bayer mosaic
# ------------------------------------------------------------------
def build_bayer(r_plane, g1_plane, g2_plane, b_plane, pattern='RGGB'):
    h, w = r_plane.shape
    h = (h // 2) * 2
    w = (w // 2) * 2

    r = r_plane[:h, :w]
    g1 = g1_plane[:h, :w]
    g2 = g2_plane[:h, :w]
    b = b_plane[:h, :w]

    mosaic = np.zeros((h, w), dtype=np.float32)

    if pattern == 'RGGB':
        mosaic[0::2, 0::2] = r[0::2, 0::2]
        mosaic[0::2, 1::2] = g1[0::2, 1::2]
        mosaic[1::2, 0::2] = g2[1::2, 0::2]
        mosaic[1::2, 1::2] = b[1::2, 1::2]
    elif pattern == 'BGGR':
        mosaic[0::2, 0::2] = b[0::2, 0::2]
        mosaic[0::2, 1::2] = g1[0::2, 1::2]
        mosaic[1::2, 0::2] = g2[1::2, 0::2]
        mosaic[1::2, 1::2] = r[1::2, 1::2]
    elif pattern == 'GRBG':
        mosaic[0::2, 0::2] = g1[0::2, 0::2]
        mosaic[0::2, 1::2] = r[0::2, 1::2]
        mosaic[1::2, 0::2] = b[1::2, 0::2]
        mosaic[1::2, 1::2] = g2[1::2, 1::2]
    elif pattern == 'GBRG':
        mosaic[0::2, 0::2] = g1[0::2, 0::2]
        mosaic[0::2, 1::2] = b[0::2, 1::2]
        mosaic[1::2, 0::2] = r[1::2, 0::2]
        mosaic[1::2, 1::2] = g2[1::2, 1::2]
    else:
        raise ValueError(f"Unknown pattern {pattern}")

    return mosaic


# ------------------------------------------------------------------
# Physical sensor model
# ------------------------------------------------------------------
def simulate_sensor(mosaic,
                    min_stop=USER_PARAMETERS["min_stop"],
                    max_stop=USER_PARAMETERS["max_stop"],
                    native_iso=USER_PARAMETERS["native_iso"],
                    iso=USER_PARAMETERS["iso"],
                    sensor_black_level=USER_PARAMETERS["sensor_black_level"],
                    read_noise=USER_PARAMETERS["read_noise"],
                    shot_noise=USER_PARAMETERS["shot_noise"],
                    noise_level=USER_PARAMETERS["noise_level"]):
    """
    Pure physical sensor model. No file-format or bit-depth awareness.
    Returns float sensor DN and a metadata dict.
    """
    base_gain = SENSOR_REF_CODE_MAX - sensor_black_level
    gain_linear_to_dn = base_gain * (iso / native_iso)

    clip_linear = 0.18 * (2.0 ** max_stop)
    floor_linear = 0.18 * (2.0 ** min_stop)

    clip_dn = sensor_black_level + clip_linear * gain_linear_to_dn
    floor_dn = sensor_black_level + floor_linear * gain_linear_to_dn
    sensor_mg_dn = sensor_black_level + 0.18 * gain_linear_to_dn

    well_default = gain_linear_to_dn * clip_linear
    well_scale = 100.0 ** (0.5 - noise_level)
    full_well_electrons = well_default * well_scale

    electrons_per_unit = full_well_electrons / clip_linear
    dn_per_electron = gain_linear_to_dn / electrons_per_unit

    signal_e = mosaic * electrons_per_unit

    if shot_noise:
        signal_e = signal_e + np.random.normal(0, np.sqrt(np.maximum(signal_e, 0.0)))

    if read_noise > 0:
        read_noise_e = read_noise / dn_per_electron
        signal_e = signal_e + np.random.normal(0, read_noise_e, signal_e.shape)

    sensor_dn = signal_e * dn_per_electron + sensor_black_level

    meta = {
        'clip_dn': clip_dn,
        'floor_dn': floor_dn,
        'sensor_mg_dn': sensor_mg_dn,
        'sensor_black_level': sensor_black_level,
        'max_stop': max_stop,
        'min_stop': min_stop,
        'gain_linear_to_dn': gain_linear_to_dn,
        'dn_per_electron': dn_per_electron,
        'full_well_electrons': full_well_electrons,
    }

    print(f"  ISO: {iso:.0f}, Sensor Gain: {gain_linear_to_dn:.0f} DN per linear unit")
    print(f"  max_stop: {max_stop:+.1f}  -> clip_linear = {clip_linear:.4f} -> {clip_dn:.0f} DN (sensor)")
    print(f"  min_stop: {min_stop:+.1f}  -> floor_linear = {floor_linear:.6f} -> {floor_dn:.0f} DN (sensor)")
    print(f"  Noise Level: {noise_level:.2f}")
    print(f"  Derived Full Well Electrons: {full_well_electrons:.1f}")
    print(f"  Derived Gain: {dn_per_electron:.2f} DN per electron")
    print(f"  0.18 middle gray -> {sensor_mg_dn:.1f} DN (sensor)")
    print(f"  Sensor DR (stops): {max_stop - min_stop:.2f}")

    return sensor_dn.astype(np.float32), meta


# ------------------------------------------------------------------
# Fixed middle-gray encoding
# ------------------------------------------------------------------
def encode_fixed(sensor_dn, meta, bit_depth,
                 black_level=None, white_level=None):
    """
    Fixed middle-gray path. Behavior-identical to the old quantize_to_sensor
    for both 32-bit and integer bit depths.
    """
    sensor_black_level = meta['sensor_black_level']
    clip_dn = meta['clip_dn']
    floor_dn = meta['floor_dn']
    sensor_mg_dn = meta['sensor_mg_dn']

    if bit_depth == 32:
        dng_black = black_level if black_level is not None else floor_dn
        dng_white = white_level if white_level is not None else clip_dn
        encoded = np.clip(sensor_dn, dng_black, dng_white).astype(np.float32)
        baseline_exposure = 0.0
    else:
        file_max = (1 << bit_depth) - 1
        target_mg = sensor_black_level + 0.18 * (file_max - sensor_black_level)
        scale = target_mg / sensor_mg_dn

        scaled = sensor_dn * scale
        dng_black = black_level if black_level is not None else int(round(floor_dn * scale))
        dng_white = white_level if white_level is not None else int(round(clip_dn * scale))
        if bit_depth != 32:
            dng_white = min(dng_white, file_max)
        encoded = np.clip(np.rint(scaled), dng_black, file_max).astype(np.uint16)
        baseline_exposure = 0.0

    if black_level is not None and floor_dn < dng_black:
        print(f"  WARNING: Floor ({floor_dn:.1f}) below DNG BlackLevel ({dng_black}), lower rangess will clip")
    if white_level is not None and clip_dn > dng_white:
        print(f"  WARNING: Clip ({clip_dn:.1f}) above DNG WhiteLevel ({dng_white}), upper ranges will clip")
    print(f"  DNG BlackLevel: {dng_black}, WhiteLevel: {dng_white}")

    return encoded, dng_black, dng_white, baseline_exposure, None


# ------------------------------------------------------------------
# Adaptive-range encoding
# ------------------------------------------------------------------
def encode_adaptive(sensor_dn, meta, bit_depth,
                    black_level=None, white_level=None,
                    apply_power=False, power_exponent=2.6):
    """
    Adaptive range path. Maps sensor DR to [0, 1.0] in float.
    Power curve (if requested) operates inside [0, 1.0] before quantization.

    Optional black_level/white_level override the DNG BlackLevel/WhiteLevel tags
    without changing the encoding (signal is still normalized to sensor DR).
    """
    clip_dn = meta['clip_dn']
    floor_dn = meta['floor_dn']
    sensor_black_level = meta['sensor_black_level']
    sensor_mg_dn = meta['sensor_mg_dn']

    # Clip to sensor DR and normalize to [0, 1.0]
    signal = np.clip(sensor_dn, floor_dn, clip_dn)
    normalized = (signal - floor_dn) / (clip_dn - floor_dn)

    # Middle gray in normalized space (independent of bit depth)
    actual_mg_norm = (sensor_mg_dn - floor_dn) / (clip_dn - floor_dn)
    baseline_exposure = float(np.log2(0.18 / actual_mg_norm))

    # Optional power curve in float [0, 1.0]
    # 32-bit float DNG cannot carry a LinearizationTable, so power encoding
    # is unsupported — it would distort color without a decoding mechanism.
    if apply_power and bit_depth != 32:
        encoded_float = np.power(normalized, 1.0 / power_exponent)
    else:
        if apply_power and bit_depth == 32:
            print("  WARNING: Power encoding disabled for 32-bit float DNG "
                  "(LinearizationTable not supported in float format)")
        encoded_float = normalized

    encoded_float = np.clip(encoded_float, 0.0, 1.0)

    if bit_depth == 32:
        encoded = encoded_float.astype(np.float32)
        dng_black = 0.0 if black_level is None else float(black_level)
        dng_white = 1.0 if white_level is None else float(white_level)
        linearization_lut = None
        # Note: 32-bit float DNG cannot carry an integer LinearizationTable.
        # Power-encoded float data will be read as scene-linear by DNG readers.
    else:
        file_max = (1 << bit_depth) - 1

        # TPDF dither in float space
        dither = (np.random.uniform(-0.5, 0.5, encoded_float.shape) +
                  np.random.uniform(-0.5, 0.5, encoded_float.shape))
        dithered = np.clip(encoded_float + dither * 0.5 / file_max, 0.0, 1.0)

        # Quantize: 1.0 -> file_max
        encoded = np.rint(dithered * file_max).astype(np.uint16 if file_max <= 65535 else np.uint32)
        dng_black = 0 if black_level is None else int(black_level)
        dng_white = file_max if white_level is None else int(white_level)
        if bit_depth != 32:
            file_max = (1 << bit_depth) - 1
            dng_white = min(dng_white, file_max)

        # LinearizationTable (only if power curve applied)
        if apply_power:
            codes = np.arange(file_max + 1, dtype=np.float64) / file_max
            linear = np.power(codes, power_exponent) * file_max
            lut = np.clip(np.rint(linear), 0, file_max).astype(np.uint16)
            linearization_lut = np.maximum.accumulate(lut)
        else:
            linearization_lut = None

    print(f"  DNG BlackLevel: {dng_black}, WhiteLevel: {dng_white}")
    if baseline_exposure != 0.0:
        print(f"  BaselineExposure: {baseline_exposure:+.2f} stops")
    if apply_power and bit_depth != 32:
        print(f"  Power encoding (exponent 1/{power_exponent})")
        cps = file_max / (meta['max_stop'] - meta['min_stop'])
        print(f"  codes_per_stop={cps:.1f}")
        if cps < 10:
            print(f"  WARNING: Only {cps:.1f} codes per stop. Deep lower ranges  will posterize.")
        print(f"  LinearizationTable: {len(linearization_lut)} entries, 16-bit")

    return encoded, dng_black, dng_white, baseline_exposure, linearization_lut


# ------------------------------------------------------------------
# Write DNG
# ------------------------------------------------------------------
def write_dng(mosaic, path,
              black_level=USER_PARAMETERS["black_level"],
              white_level=USER_PARAMETERS["white_level"],
              bit_depth=USER_PARAMETERS["bit_depth"],
              pattern=USER_PARAMETERS["pattern"],
               r_filter=None,
               g_filter=None,
               b_filter=None,
               baseline_exposure=0.0,
               linearization_lut=None):
    import tifffile

    h, w = mosaic.shape
    cfa_map = {
        'RGGB': [0, 1, 1, 2],
        'BGGR': [2, 1, 1, 0],
        'GRBG': [1, 2, 0, 1],
        'GBRG': [1, 0, 2, 1],
    }
    cfa = cfa_map.get(pattern, [0, 1, 1, 2])

    if r_filter is None:
        # Rec.709 primaries in Rec.2020
        r_filter = np.array([0.627404, 0.069097, 0.016391])
        g_filter = np.array([0.329283, 0.919540, 0.088013])
        b_filter = np.array([0.043313, 0.011362, 0.895595])

    # ColorMatrix1: XYZ to Camera Native (computed from Rec.2020 primaries + filter matrix)
    # Rec.2020: R=(0.708,0.292), G=(0.170,0.797), B=(0.131,0.046), D65=(0.3127,0.3290)
    cm1 = compute_cm1(r_filter, g_filter, b_filter)

    # CameraCalibration1: identity
    cc1 = (
        1, 1, 0, 1, 0, 1,
        0, 1, 1, 1, 0, 1,
        0, 1, 0, 1, 1, 1
    )

    r_sum = float(np.sum(r_filter))
    g_sum = float(np.sum(g_filter))
    b_sum = float(np.sum(b_filter))

    asn_r = int(round((r_sum / g_sum) * 10000))
    asn_b = int(round((b_sum / g_sum) * 10000))

    asn = (asn_r, 10000, 1, 1, asn_b, 10000)
    ab = (1, 1, 1, 1, 1, 1)
    ds = (1, 1, 1, 1)

    # Determine bit depth and sample format from bit_depth
    bl_val = int(round(black_level))

    if bit_depth == 32:
        bits_per_sample = 32
        if mosaic.dtype != np.float32:
            mosaic = mosaic.astype(np.float32)
        bl_tag_type = 4  # LONG
        bl_count = 1
        wl_tag_type = 4  # LONG
        wl_val = int(round(white_level))
    else:
        bits_per_sample = bit_depth
        if mosaic.dtype != np.uint16:
            mosaic = mosaic.astype(np.uint16)
        wl_val = int(round(white_level))
        bl_tag_type = 3 if 0 <= bl_val <= 65535 else 4
        wl_tag_type = 3 if 0 <= wl_val <= 65535 else 4
        bl_count = 1

    extratags = [
        (33421, 3, 2, (2, 2), False),      # CFARepeatPatternDim
        (33422, 1, 4, tuple(cfa), False),   # CFAPattern
        (50706, 1, 4, (1, 4, 0, 0), False), # DNGVersion
        (50707, 1, 4, (1, 1, 0, 0), False), # DNGBackwardVersion
        (50708, 2, 1, b"Synthetic", False),  # UniqueCameraModel
        (50714, bl_tag_type, bl_count, bl_val, False),  # BlackLevel (scalar)
        (50717, wl_tag_type, 1, wl_val, False),    # WhiteLevel
        (50718, 5, 2, ds, False),           # DefaultScale
        (50721, 10, 9, cm1, False),         # ColorMatrix1
        (50723, 10, 9, cc1, False),         # CameraCalibration1
        (50728, 5, 3, asn, False),          # AsShotNeutral
        (50778, 3, 1, 21, False),           # CalibrationIlluminant1 (D65)
        (50727, 5, 3, ab, False),           # AnalogBalance
    ]

    extratags.extend([
        (50719, 4, 2, (0, 0), False),          # DefaultCropOrigin  = (0, 0)
        (50720, 4, 2, (w, h), False),          # DefaultCropSize    = (width, height)
        (271, 2, 1, b"Generic", False),        # Make
        (272, 2, 1, b"Virtual", False),        # Model
    ])

    if baseline_exposure != 0.0:
        be = Fraction(baseline_exposure).limit_denominator(10000)
        extratags.append(
            (50730, 10, 1, (be.numerator, be.denominator), False)  # BaselineExposure
        )

    if linearization_lut is not None:
        extratags.append(
            (50712, 3, len(linearization_lut), tuple(linearization_lut), False)  # LinearizationTable
        )

    tifffile.imwrite(
        path,
        mosaic,
        photometric=32803,
        planarconfig='contig',
        compression=None,
        bitspersample=bits_per_sample,
        extratags=extratags,
        subfiletype=0,
        description=None,
    )

    print(f"Wrote DNG: {path}  ({w}x{h}, {pattern})")
    print(f"  BlackLevel={black_level}, WhiteLevel={white_level}")
    print(f"  AsShotNeutral=[{asn_r/10000:.4f}, 1.0, {asn_b/10000:.4f}]")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
SENSOR_REF_CODE_MAX = 65535

def parse_filter(s):
    return np.array([float(x.strip()) for x in s.split(',')], dtype=np.float32)


def find_first_exr(script_dir):
    exr_files = sorted(glob.glob(os.path.join(script_dir, "*.exr")))
    return exr_files[0] if exr_files else None


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    if len(sys.argv) == 1:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_exr = find_first_exr(script_dir)
        if input_exr is None:
            print("No .exr file found in script directory.")
            print("Usage: python this_script.py input.exr [output.dng] [options]")
            sys.exit(1)
        output_dng = os.path.splitext(input_exr)[0] + ".dng"
        print(f"Auto-detected input:  {input_exr}")
        print(f"Auto-generated output: {output_dng}")
        
        args = type('Args', (), {
            'input': input_exr,
            'output_dng': output_dng,
            'planes': None,
            'r_filter': ','.join(map(str, USER_PARAMETERS["r_filter"])),
            'g_filter': ','.join(map(str, USER_PARAMETERS["g_filter"])),
            'b_filter': ','.join(map(str, USER_PARAMETERS["b_filter"])),
            'pattern': USER_PARAMETERS["pattern"],
            'min_stop': USER_PARAMETERS["min_stop"],
            'max_stop': USER_PARAMETERS["max_stop"],
            'fixed_middle_gray': USER_PARAMETERS["fixed_middle_gray"],
            'native_iso': USER_PARAMETERS["native_iso"],
            'iso': USER_PARAMETERS["iso"],
            'sensor_black_level': USER_PARAMETERS["sensor_black_level"],
            'white_level': USER_PARAMETERS["white_level"],
            'black_level': USER_PARAMETERS["black_level"],
            'bit_depth': USER_PARAMETERS["bit_depth"],
            'read_noise': USER_PARAMETERS["read_noise"],
            'no_shot_noise': USER_PARAMETERS["no_shot_noise"],
            'noise_level': USER_PARAMETERS["noise_level"],
            'pow_encode_int': USER_PARAMETERS["pow_encode_int"],
            'power_exponent': USER_PARAMETERS["power_exponent"],
        })()
    else:
        parser = argparse.ArgumentParser(
            description="Convert linear Rec.2020 RGB EXR to synthetic Bayer-raw DNG. "
                        "Gain is set by ISO. Dynamic range is set by max_stop.")
        parser.add_argument("input", help="Input EXR")
        parser.add_argument("output_dng", nargs='?', default=None,
                            help="Output DNG path (default: same name as input with .dng)")
        parser.add_argument("--planes", nargs=4, metavar=("R","G1","G2","B"),
                            help="Use 4 pre-rendered scalar EXRs instead of RGB filtering")
        parser.add_argument("--r-filter", default=','.join(map(str, USER_PARAMETERS["r_filter"])))
        parser.add_argument("--g-filter", default=','.join(map(str, USER_PARAMETERS["g_filter"])))
        parser.add_argument("--b-filter", default=','.join(map(str, USER_PARAMETERS["b_filter"])))
        parser.add_argument("--pattern", default=USER_PARAMETERS["pattern"],
                            choices=["RGGB", "BGGR", "GRBG", "GBRG"])
        
        # Central setting: min and max stops relative to 0.18
        # max_stop sets the gain. min_stop is informational (noise floor).
        parser.add_argument("--min-stop", type=float, default=USER_PARAMETERS["min_stop"],
                            help="Stops below 0.18 where signal hits noise floor. Default: -10.0")
        parser.add_argument("--max-stop", type=float, default=USER_PARAMETERS["max_stop"],
                            help="Stops above 0.18 where sensor clips. Sets the dynamic range. "
                                 "Default: 4.0")
        parser.add_argument("--fixed-middle-gray", action=argparse.BooleanOptionalAction,
                            default=USER_PARAMETERS["fixed_middle_gray"],
                            help="Pin middle gray at 18%% of file range. "
                                 "Default: None (auto). Integer bit depths use adaptive "
                                 "middle gray (upper bound pinned at file_max, middle gray "
                                 "shifts darker to preserve range). Float32 uses fixed "
                                 "middle gray. Use --fixed-middle-gray to force on, "
                                 "--no-fixed-middle-gray to force off.")
        
        parser.add_argument("--sensor-black-level", type=int, default=USER_PARAMETERS["sensor_black_level"],
                            help="DN for zero open-domain linear signal (sensor black level). Default: 256")
        parser.add_argument("--native-iso", type=float, default=USER_PARAMETERS["native_iso"],
                            help="Camera native (base) ISO. Defines the sensor's intrinsic "
                                 "gain. Default: 100")
        parser.add_argument("--iso", type=float, default=USER_PARAMETERS["iso"],
                            help="Shooting ISO. Gain = base_gain * (iso / native_iso). "
                                 "Higher ISO = more gain and more noise. Default: 100")
        
        parser.add_argument("--white-level", type=int, default=USER_PARAMETERS["white_level"],
                            help="DNG WhiteLevel. Default: auto (sensor clip point at max_stop)")
        parser.add_argument("--black-level", type=int, default=USER_PARAMETERS["black_level"],
                            help="DNG BlackLevel. Default: auto (sensor floor point at min_stop)")
        parser.add_argument("--bit-depth", type=int, default=USER_PARAMETERS["bit_depth"],
                            choices=[10, 12, 14, 16, 32],
                            help=(
                                "ADC bit depth. 10/12/14/16 = unsigned integer; 32 = IEEE 754 float. "
                                "A high max-stop may push the sensor's WhiteLevel beyond the integer "
                                "file range; the file data will clip."
                            ))
        parser.add_argument("--pow-encode-int", action=argparse.BooleanOptionalAction,
                            default=USER_PARAMETERS["pow_encode_int"],
                            help="Apply power encoding to adaptive range. Reduces quantization "
                                 "artifacts in lower ranges at the cost of requiring the downstream RAW "
                                 "editor to support DNG LinearizationTable (tag 0xC618). Only active "
                                 "with adaptive range (fixed_middle_gray off).")
        parser.add_argument("--power-exponent", type=float, default=USER_PARAMETERS["power_exponent"],
                            help="Exponent for power encoding curve (1/N). Higher values compress "
                                 "lower ranges more. Default: 2.6")
        parser.add_argument("--read-noise", type=float, default=USER_PARAMETERS["read_noise"],
                            help="Read noise std-dev in DN (0 to disable)")
        parser.add_argument("--no-shot-noise", action="store_true", default=USER_PARAMETERS["no_shot_noise"])
        parser.add_argument("--noise-level", type=float, default=USER_PARAMETERS["noise_level"],
                            help="Overall noise level (0.0-1.0, 0.5 is standard)")
        args = parser.parse_args()

        if args.output_dng is None:
            args.output_dng = os.path.splitext(args.input)[0] + ".dng"

    # Parse filters
    r_filter = parse_filter(args.r_filter)
    g_filter = parse_filter(args.g_filter)
    b_filter = parse_filter(args.b_filter)
    print(f"Filters (Rec.709 in Rec.2020 RGB):")
    print(f"  R: {r_filter}")
    print(f"  G: {g_filter}")
    print(f"  B: {b_filter}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    if args.planes:
        print("\nReading 4 scalar EXRs...")
        r = read_exr(args.planes[0], expect_rgb=False)
        g1 = read_exr(args.planes[1], expect_rgb=False)
        g2 = read_exr(args.planes[2], expect_rgb=False)
        b = read_exr(args.planes[3], expect_rgb=False)
    else:
        print(f"\nReading RGB EXR: {args.input}")
        rgb = read_exr(args.input, expect_rgb=True)
        print(f"  Shape: {rgb.shape}, range: [{rgb.min():.4f}, {rgb.max():.4f}]")
        
        print("Applying Bayer filters...")
        sensor_weight = np.array(USER_PARAMETERS["sensor_sensitivity_weighting"], dtype=np.float32)
        r, g1, b_scalar = filter_rgb_to_scalar(rgb, r_filter, g_filter, b_filter, sensor_weight)
        g2 = g1.copy()

    print(f"R  shape: {r.shape}, range: [{r.min():.4f}, {r.max():.4f}]")
    print(f"G1 shape: {g1.shape}, range: [{g1.min():.4f}, {g1.max():.4f}]")
    print(f"G2 shape: {g2.shape}, range: [{g2.min():.4f}, {g2.max():.4f}]")
    b = b if args.planes else b_scalar
    print(f"B  shape: {b.shape}, range: [{b.min():.4f}, {b.max():.4f}]")

    mosaic = build_bayer(r, g1, g2, b, args.pattern)



    # ------------------------------------------------------------------
    # Physical model
    # ------------------------------------------------------------------
    print("\nApplying sensor model...")
    sensor_dn, meta = simulate_sensor(
        mosaic,
        min_stop=args.min_stop,
        max_stop=args.max_stop,
        native_iso=args.native_iso,
        iso=args.iso,
        sensor_black_level=args.sensor_black_level,
        read_noise=args.read_noise,
        shot_noise=not args.no_shot_noise,
        noise_level=args.noise_level,
    )

    # ------------------------------------------------------------------
    # Resolve fixed/adaptive from tri-state parameter
    # ------------------------------------------------------------------
    use_fixed = args.fixed_middle_gray
    if use_fixed is None:
        use_fixed = (args.bit_depth == 32)

    # ------------------------------------------------------------------
    # Encoding (bit-depth dependent)
    # ------------------------------------------------------------------
    if use_fixed:
        raw, black_level, white_level, baseline_exposure, lut = encode_fixed(
            sensor_dn, meta,
            bit_depth=args.bit_depth,
            black_level=args.black_level,
            white_level=args.white_level,
        )
    else:
        raw, black_level, white_level, baseline_exposure, lut = encode_adaptive(
            sensor_dn, meta,
            bit_depth=args.bit_depth,
            black_level=args.black_level,
            white_level=args.white_level,
            apply_power=args.pow_encode_int,
            power_exponent=args.power_exponent,
        )

    # ------------------------------------------------------------------
    # Write DNG
    # ------------------------------------------------------------------
    write_dng(
        raw,
        args.output_dng,
        black_level=black_level,
        white_level=white_level,
        bit_depth=args.bit_depth,
        pattern=args.pattern,
        r_filter=r_filter,
        g_filter=g_filter,
        b_filter=b_filter,
        baseline_exposure=baseline_exposure,
        linearization_lut=lut,
    )


if __name__ == "__main__":
    main()