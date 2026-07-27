"""
Convert a linear Rec.2020 RGB EXR to Bayer-raw DNG with a simulated virtual camera.
"""

import argparse
import glob
import os
import sys
import numpy as np
from fractions import Fraction


# all default parameters
DEFAULTS = {
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
    # float32 uses fixed middle gray at 18%. True/False override per-bit-depth default.
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
            "Cannot read EXR. Install OpenEXR:  pip install OpenEXR\n"
            "Or:  pip install imageio imageio-freeimage"
        ) from e


# ------------------------------------------------------------------
# 2. Bayer Filtering
# ------------------------------------------------------------------
def filter_rgb_to_scalar(rgb, r_filter, g_filter, b_filter, sensor_weight):
    # Effective response = filter(λ) * sensor(λ), integrated over all λ
    r_plane = np.dot(rgb, r_filter * sensor_weight)
    g_plane = np.dot(rgb, g_filter * sensor_weight)
    b_plane = np.dot(rgb, b_filter * sensor_weight)
    return r_plane, g_plane, b_plane


# ------------------------------------------------------------------
# 3. Interleave Bayer mosaic
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
# 4. Sensor model
# ------------------------------------------------------------------
def quantize_to_sensor(mosaic,
                       min_stop=DEFAULTS["min_stop"],
                       max_stop=DEFAULTS["max_stop"],
                       fixed_middle_gray=DEFAULTS["fixed_middle_gray"],
                       native_iso=DEFAULTS["native_iso"],
                       iso=DEFAULTS["iso"],
                       sensor_black_level=DEFAULTS["sensor_black_level"],
                       white_level=DEFAULTS["white_level"],
                       black_level=DEFAULTS["black_level"],
                       read_noise=DEFAULTS["read_noise"],
                       shot_noise=DEFAULTS["shot_noise"],
                       noise_level=DEFAULTS["noise_level"],
                       bit_depth=DEFAULTS["bit_depth"]):
    """
    Linear sensor model.

    Sensor gain is derived from ISO (physical parameter), independent of bit depth.
    ISO 100 = base gain. At ISO 100 with 32-bit float output and fixed_middle_gray,
    behavior is identical to the original model.

    fixed_middle_gray controls the file mapping for integer bit depths:
    - True: middle gray pinned at 18% of file range, upper bound grows.
    - False: upper bound pinned at file_max, middle gray shifts darker.
    - None (default): integer→False, float32→True.

    Sensor DR [min_stop, max_stop] maps to [floor_dn, clip_dn] in sensor DN space.
    DNG BlackLevel/WhiteLevel are set to the file-mapped values.
    """
    # ------------------------------------------------------------------
    # 1. Sensor model (bit-depth independent)
    # ------------------------------------------------------------------
    # Base gain: DN per unit linear signal at native ISO
    # Reference: 32-bit float with fixed middle gray
    base_gain = get_max_dn(32) - sensor_black_level
    gain_linear_to_dn = base_gain * (iso / native_iso)

    clip_linear = 0.18 * (2.0 ** max_stop)
    floor_linear = 0.18 * (2.0 ** min_stop)

    # Sensor range in DN (absolute, independent of bit depth)
    clip_dn = sensor_black_level + clip_linear * gain_linear_to_dn
    floor_dn = sensor_black_level + floor_linear * gain_linear_to_dn

    # Middle gray DN in sensor space
    sensor_mg_dn = sensor_black_level + 0.18 * gain_linear_to_dn

    # Full well and electron calculations
    well_default = gain_linear_to_dn * clip_linear
    well_scale = 100.0 ** (0.5 - noise_level)
    full_well_electrons = well_default * well_scale

    electrons_per_unit = full_well_electrons / clip_linear
    dn_per_electron = gain_linear_to_dn / electrons_per_unit

    # 1. Linear signal → electrons
    signal_e = mosaic * electrons_per_unit

    # 2. Shot noise in electron domain
    if shot_noise:
        shot_noise_e = np.random.normal(0, np.sqrt(np.maximum(signal_e, 0.0)))
        signal_e = signal_e + shot_noise_e

    # 3. Read noise (user param is in DN; convert to electrons internally)
    if read_noise > 0:
        read_noise_e = read_noise / dn_per_electron
        signal_e = signal_e + np.random.normal(0, read_noise_e, signal_e.shape)

    # 4. Quantize to DN
    dn = signal_e * dn_per_electron + sensor_black_level

    # ------------------------------------------------------------------
    # 2. File format mapping (bit-depth dependent)
    # ------------------------------------------------------------------
    baseline_exposure = 0.0

    if bit_depth == 32:
        dng_black = black_level if black_level is not None else floor_dn
        dng_white = white_level if white_level is not None else clip_dn
        dn_for_file = np.clip(dn, dng_black, dng_white)
    else:
        file_max = (1 << bit_depth) - 1
        target_mg = sensor_black_level + 0.18 * (file_max - sensor_black_level)

        # Resolve three-state: None=auto, True=force fixed, False=force adaptive
        use_fixed = fixed_middle_gray if fixed_middle_gray is not None else False

        if use_fixed:
            scale = target_mg / sensor_mg_dn
            baseline_exposure = 0.0
        else:
            scale = file_max / clip_dn
            actual_mg = sensor_mg_dn * scale
            baseline_exposure = float(np.log2(target_mg / actual_mg))

        dn_scaled = dn * scale

        dng_black = black_level if black_level is not None else floor_dn * scale
        dng_white = white_level if white_level is not None else clip_dn * scale

        dn_for_file = np.clip(np.rint(dn_scaled), dng_black, file_max)

    print(f"  ISO: {iso:.0f}, Sensor Gain: {gain_linear_to_dn:.0f} DN per linear unit")
    print(f"  max_stop: {max_stop:+.1f}  -> clip_linear = {clip_linear:.4f} -> {clip_dn:.0f} DN (sensor)")
    print(f"  min_stop: {min_stop:+.1f}  -> floor_linear = {floor_linear:.6f} -> {floor_dn:.0f} DN (sensor)")
    print(f"  Noise Level: {noise_level:.2f}")
    print(f"  Derived Full Well Electrons: {full_well_electrons:.1f}")
    print(f"  Derived Gain: {dn_per_electron:.2f} DN per electron")
    print(f"  0.18 middle gray -> {sensor_mg_dn:.1f} DN (sensor)")
    print(f"  DNG BlackLevel: {dng_black:.1f}, WhiteLevel: {dng_white:.1f}")
    print(f"  Sensor DR (stops): {max_stop - min_stop:.2f}")
    if baseline_exposure != 0.0:
        print(f"  BaselineExposure: {baseline_exposure:+.2f} stops")

    # Warn if user-provided levels don't cover full sensor DR
    if black_level is not None and floor_dn < dng_black:
        print(f"  WARNING: Floor ({floor_dn:.1f}) below DNG BlackLevel ({dng_black}), shadows will clip")
    if white_level is not None and clip_dn > dng_white:
        print(f"  WARNING: Clip ({clip_dn:.1f}) above DNG WhiteLevel ({dng_white}), highlights will clip")

    return dn_for_file, dng_black, dng_white, baseline_exposure


# ------------------------------------------------------------------
# 4b. EXIF IFD patching (add ExposureBiasValue inside EXIF sub-IFD)
# ------------------------------------------------------------------
def _patch_exif_ifd(path, numerator, denominator):
    import struct

    with open(path, 'r+b') as f:
        data = bytearray(f.read())

    assert data[0:2] == b'II', "Only little-endian TIFF supported"
    root_ifd_offset = struct.unpack_from('<I', data, 4)[0]
    entry_count = struct.unpack_from('<H', data, root_ifd_offset)[0]
    entries_start = root_ifd_offset + 2

    new_entry_count = entry_count + 1
    new_root_ifd_offset = len(data)

    exif_ifd_offset = new_root_ifd_offset + 2 + new_entry_count * 12 + 4
    exif_value_offset = exif_ifd_offset + 2 + 12 + 4

    exif_ifd = bytearray()
    exif_ifd += struct.pack('<H', 1)
    exif_ifd += struct.pack('<HHII', 37380, 10, 1, exif_value_offset)
    exif_ifd += struct.pack('<I', 0)
    exif_ifd += struct.pack('<ii', numerator, denominator)

    new_root_ifd = bytearray()
    new_root_ifd += struct.pack('<H', new_entry_count)

    insert_tag = 34665
    inserted = False
    for i in range(entry_count):
        entry_off = entries_start + i * 12
        tag = struct.unpack_from('<H', data, entry_off)[0]
        if not inserted and tag > insert_tag:
            new_root_ifd += struct.pack('<HHII', insert_tag, 4, 1, exif_ifd_offset)
            inserted = True
        new_root_ifd += data[entry_off:entry_off + 12]
    if not inserted:
        new_root_ifd += struct.pack('<HHII', insert_tag, 4, 1, exif_ifd_offset)

    new_root_ifd += struct.pack('<I', 0)

    data += new_root_ifd + exif_ifd
    struct.pack_into('<I', data, 4, new_root_ifd_offset)

    with open(path, 'wb') as f:
        f.write(data)


# ------------------------------------------------------------------
# 5. Write DNG
# ------------------------------------------------------------------
def write_dng(mosaic, path,
              black_level=DEFAULTS["black_level"],
              white_level=DEFAULTS["white_level"],
              bit_depth=DEFAULTS["bit_depth"],
              pattern=DEFAULTS["pattern"],
               r_filter=None,
               g_filter=None,
               b_filter=None,
               baseline_exposure=0.0):
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
    if bit_depth == 32:
        bits_per_sample = 32
        # Ensure mosaic is float32 (tifffile will auto-set SampleFormat=3)
        if mosaic.dtype != np.float32:
            mosaic = mosaic.astype(np.float32)
        # For float32 DNG, WhiteLevel/BlackLevel are written as LONG (integers)
        bl_tag_type = 4  # SLONG
        bl_count = 1
        bl_val = int(black_level)
        wl_tag_type = 4  # SLONG (LONG in libtiff)
        wl_count = 1
        wl_val = int(round(white_level))
    else:
        bits_per_sample = bit_depth   # 10, 12, 14, or 16 — NOT always 16
        if mosaic.dtype != np.uint16:
            mosaic = mosaic.astype(np.uint16)

        bl_val = int(black_level)
        wl_val = int(round(white_level))
        # Use LONG (type 4) if values exceed unsigned SHORT (type 3) range [0, 65535]
        bl_tag_type = 3 if 0 <= bl_val <= 65535 else 4
        wl_tag_type = 3 if 0 <= wl_val <= 65535 else 4
        bl_count = 1
        wl_count = 1

    extratags = [
        (33421, 3, 2, (2, 2), False),     # CFARepeatPatternDim
        (33422, 1, 4, tuple(cfa), False), # CFAPattern
        (50706, 1, 4, (1, 4, 0, 0), False),          # DNGVersion
        (50707, 1, 4, (1, 1, 0, 0), False),          # DNGBackwardVersion
        (50708, 2, 1, b"Synthetic", False),         # UniqueCameraModel
        (50714, bl_tag_type, bl_count, bl_val, False),       # BlackLevel
        (50717, wl_tag_type, wl_count, wl_val, False),       # WhiteLevel
        (50718, 5, 2, ds, False),                     # DefaultScale
        (50721, 10, 9, cm1, False),                   # ColorMatrix1
        (50723, 10, 9, cc1, False),                   # CameraCalibration1
        (50728, 5, 3, asn, False),                    # AsShotNeutral
        (50778, 3, 1, 21, False),                     # CalibrationIlluminant1 (D65)
        (50727, 5, 3, ab, False),                     # AnalogBalance
    ]

    if baseline_exposure != 0.0:
        be = Fraction(baseline_exposure).limit_denominator(10000)
        extratags.append(
            (50730, 10, 1, (be.numerator, be.denominator), False)  # BaselineExposure
        )

    tifffile.imwrite(
        path,
        mosaic,
        photometric=32803,
        planarconfig='contig',
        compression=None,
        bitspersample=bits_per_sample,
        extratags=extratags,
    )

    if baseline_exposure != 0.0:
        import subprocess, shutil
        exiftool = shutil.which("exiftool")
        if exiftool is None:
            for p in [r"D:\exiftool-13.59_64\exiftool.exe",
                       r"D:\exiftool-13.59_64\exiftool(-k).exe"]:
                if os.path.isfile(p):
                    exiftool = p
                    break
        if exiftool:
            subprocess.run(
                [exiftool, "-overwrite_original",
                 f"-ExposureCompensation={-baseline_exposure:.6f}",
                 path],
                input=b"\n", check=True, capture_output=True)
            print(f"  Injected EXIF ExposureCompensation via exiftool: {baseline_exposure:.6f}")
        else:
            _patch_exif_ifd(path, be.numerator, be.denominator)

    print(f"Wrote DNG: {path}  ({w}x{h}, {pattern})")
    print(f"  BlackLevel={black_level}, WhiteLevel={white_level}")
    print(f"  AsShotNeutral=[{asn_r/10000:.4f}, 1.0, {asn_b/10000:.4f}]")


# ------------------------------------------------------------------
# 6. Helpers
# ------------------------------------------------------------------
def get_max_dn(bit_depth):
    return 65535 if bit_depth == 32 else (1 << bit_depth) - 1

def parse_filter(s):
    return np.array([float(x.strip()) for x in s.split(',')], dtype=np.float32)


def find_first_exr(script_dir):
    exr_files = sorted(glob.glob(os.path.join(script_dir, "*.exr")))
    return exr_files[0] if exr_files else None


# ------------------------------------------------------------------
# 7. Main
# ------------------------------------------------------------------
def main():
    if len(sys.argv) == 1:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_exr = find_first_exr(script_dir)
        if input_exr is None:
            print("No .exr file found in script directory.")
            print("Usage: python exr2dng.py input.exr [output.dng] [options]")
            sys.exit(1)
        output_dng = os.path.splitext(input_exr)[0] + ".dng"
        print(f"Auto-detected input:  {input_exr}")
        print(f"Auto-generated output: {output_dng}")
        
        args = type('Args', (), {
            'input': input_exr,
            'output_dng': output_dng,
            'planes': None,
            'r_filter': ','.join(map(str, DEFAULTS["r_filter"])),
            'g_filter': ','.join(map(str, DEFAULTS["g_filter"])),
            'b_filter': ','.join(map(str, DEFAULTS["b_filter"])),
            'pattern': DEFAULTS["pattern"],
            'min_stop': DEFAULTS["min_stop"],
            'max_stop': DEFAULTS["max_stop"],
            'fixed_middle_gray': DEFAULTS["fixed_middle_gray"],
            'native_iso': DEFAULTS["native_iso"],
            'iso': DEFAULTS["iso"],
            'sensor_black_level': DEFAULTS["sensor_black_level"],
            'white_level': DEFAULTS["white_level"],
            'black_level': DEFAULTS["black_level"],
            'bit_depth': DEFAULTS["bit_depth"],
            'read_noise': DEFAULTS["read_noise"],
            'no_shot_noise': DEFAULTS["no_shot_noise"],
            'noise_level': DEFAULTS["noise_level"],
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
        parser.add_argument("--r-filter", default=','.join(map(str, DEFAULTS["r_filter"])))
        parser.add_argument("--g-filter", default=','.join(map(str, DEFAULTS["g_filter"])))
        parser.add_argument("--b-filter", default=','.join(map(str, DEFAULTS["b_filter"])))
        parser.add_argument("--pattern", default=DEFAULTS["pattern"],
                            choices=["RGGB", "BGGR", "GRBG", "GBRG"])
        
        # Central setting: min and max stops relative to 0.18
        # max_stop sets the gain. min_stop is informational (noise floor).
        parser.add_argument("--min-stop", type=float, default=DEFAULTS["min_stop"],
                            help="Stops below 0.18 where signal hits noise floor. Default: -10.0")
        parser.add_argument("--max-stop", type=float, default=DEFAULTS["max_stop"],
                            help="Stops above 0.18 where sensor clips. Sets the dynamic range. "
                                 "Default: 4.0")
        parser.add_argument("--fixed-middle-gray", action=argparse.BooleanOptionalAction,
                            default=DEFAULTS["fixed_middle_gray"],
                            help="Pin middle gray at 18%% of file range. "
                                 "Default: None (auto). Integer bit depths use adaptive "
                                 "middle gray (upper bound pinned at file_max, middle gray "
                                 "shifts darker to preserve range). Float32 uses fixed "
                                 "middle gray. Use --fixed-middle-gray to force on, "
                                 "--no-fixed-middle-gray to force off.")
        
        parser.add_argument("--sensor-black-level", type=int, default=DEFAULTS["sensor_black_level"],
                            help="DN for zero open-domain linear signal (sensor black level). Default: 256")
        parser.add_argument("--native-iso", type=float, default=DEFAULTS["native_iso"],
                            help="Camera native (base) ISO. Defines the sensor's intrinsic "
                                 "gain. Default: 100")
        parser.add_argument("--iso", type=float, default=DEFAULTS["iso"],
                            help="Shooting ISO. Gain = base_gain * (iso / native_iso). "
                                 "Higher ISO = more gain and more noise. Default: 100")
        
        parser.add_argument("--white-level", type=int, default=DEFAULTS["white_level"],
                            help="DNG WhiteLevel. Default: auto (sensor clip point at max_stop)")
        parser.add_argument("--black-level", type=int, default=DEFAULTS["black_level"],
                            help="DNG BlackLevel. Default: auto (sensor floor point at min_stop)")
        parser.add_argument("--bit-depth", type=int, default=DEFAULTS["bit_depth"],
                            choices=[10, 12, 14, 16, 32],
                            help=(
                                "ADC bit depth. 10/12/14/16 = unsigned integer; 32 = IEEE 754 float. "
                                "A high max-stop may push the sensor's WhiteLevel beyond the integer "
                                "file range; the file data will clip, but WhiteLevel preserves the "
                                "sensor's full capacity for correct decoding. See --max-stop."
                            ))
        parser.add_argument("--read-noise", type=float, default=DEFAULTS["read_noise"],
                            help="Read noise std-dev in DN (0 to disable)")
        parser.add_argument("--no-shot-noise", action="store_true", default=DEFAULTS["no_shot_noise"])
        parser.add_argument("--noise-level", type=float, default=DEFAULTS["noise_level"],
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
        sensor_weight = np.array(DEFAULTS["sensor_sensitivity_weighting"], dtype=np.float32)
        r, g1, b_scalar = filter_rgb_to_scalar(rgb, r_filter, g_filter, b_filter, sensor_weight)
        g2 = g1.copy()

    print(f"R  shape: {r.shape}, range: [{r.min():.4f}, {r.max():.4f}]")
    print(f"G1 shape: {g1.shape}, range: [{g1.min():.4f}, {g1.max():.4f}]")
    print(f"G2 shape: {g2.shape}, range: [{g2.min():.4f}, {g2.max():.4f}]")
    b = b if args.planes else b_scalar
    print(f"B  shape: {b.shape}, range: [{b.min():.4f}, {b.max():.4f}]")

    mosaic = build_bayer(r, g1, g2, b, args.pattern)



    # ------------------------------------------------------------------
    # Sensor model
    # ------------------------------------------------------------------
    print("\nApplying sensor model...")
    raw, black_level, white_level, baseline_exposure = quantize_to_sensor(
        mosaic,
        min_stop=args.min_stop,
        max_stop=args.max_stop,
        fixed_middle_gray=args.fixed_middle_gray,
        native_iso=args.native_iso,
        iso=args.iso,
        sensor_black_level=args.sensor_black_level,
        black_level=args.black_level,
        white_level=args.white_level,
        read_noise=args.read_noise,
        shot_noise=not args.no_shot_noise,
        noise_level=args.noise_level,
        bit_depth=args.bit_depth,
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
    )


if __name__ == "__main__":
    main()