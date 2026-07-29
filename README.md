# VirtualCamSensor

<img width="1523" height="853" alt="image" src="https://github.com/user-attachments/assets/9b21ac3a-bcc6-4b38-9660-480b26052246" />


## What is this?

Eary's VirtualCamSensor is a Python tool that converts a linear Rec.2020 RGB EXR into a genuine Bayer-raw DNG file by simulating a complete physical camera sensor pipeline. It takes a 3D-rendered EXR and produces a synthetic raw file that behaves like real camera data, including Bayer mosaic, photon shot noise, read noise, gain/ISO response, and proper DNG metadata.

The output is a standards-compliant DNG with correct `ColorMatrix1`, `CFAPattern`, `BlackLevel`, `WhiteLevel`, and optional `LinearizationTable` tags, ready to be opened in any raw processor (Adobe Camera Raw, Darktable, RawTherapee, etc.).

It uses Linear Rec.2020 as input, abusing the fact that its primaries are spectrally defined to be 630 nm (red), 532 nm (green), and 467 nm (blue). By using Rec.2020 primaries as three wavelengths, it avoided the complexity of full spectral data while keeping the chain physically grounded. Therefore pre-converting the EXR to linear Rec.2020 encoding is required.

The pipeline goes:
Incoming lights defined with Rec.2020 primaries wavelengths → Bayer Filters with tri-wavelength-defined transmission spectrums → Sensor Scalar Signal → Photon-to-Electron Conversion (with full-well capacity) → Add Noise (photon shot noise + read noise) → gain to DN → Black Level Offset → Quantization/Encoding (fixed or adaptive range, optional power curve) → DNG Bayer File

## Why is it needed?

Developing and testing Bayer-state algorithms requires genuine raw sensor data. But certain failure modes are relatively difficult to reliably recreate in real-world photography.

VirtualCamSensor lets you generate these exact conditions from a 3D render. You control the setup, the sensor physics, and the encoding, then get a DNG that your raw pipeline will treat as real camera data.

## How to use

### Requirements

- Python 3.12+
- `pip install "tifffile>=2026.7.14" "imagecodecs>=2026.6.26" numpy OpenEXR imageio`

### Quick start

Place an EXR file in the same directory as the script and run:

```
python VirtualCamSensor.py
```

The script auto-detects the first `.exr` file and writes `<name>.dng`.

### Full usage

```
python VirtualCamSensor.py input.exr output.dng [options]
```

#### Key options

| Option                                           | Description                                                                                                                                                              |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--pattern {RGGB,BGGR,GRBG,GBRG}`                | Bayer pattern (default: RGGB)                                                                                                                                            |
| `--min-stop FLOAT`                               | Stops below 0.18 where signal hits noise floor (default: -10.0)                                                                                                          |
| `--max-stop FLOAT`                               | Stops above 0.18 where sensor clips; sets dynamic range (default: 4.0)                                                                                                   |
| `--iso FLOAT`                                    | Shooting ISO; higher = more gain and noise (default: 100)                                                                                                                |
| `--native-iso FLOAT`                             | Sensor base ISO (default: 100)                                                                                                                                           |
| `--bit-depth {10,12,14,16,32}`                   | ADC output bit depth; 32 bit is in float, while others are integer (default: 32)                                                                                         |
| `--fixed-middle-gray` / `--no-fixed-middle-gray` | Pin middle gray at 18% of range vs. adaptive range (default: auto: fixed for float32, adaptive for integer)                                                              |
| `--pow-encode-int` / `--no-pow-encode-int`       | Apply power curve encoding for adaptive integer output (requires downstream LinearizationTable support)                                                                  |
| `--power-exponent FLOAT`                         | Power curve exponent (default: 2.6)                                                                                                                                      |
| `--read-noise FLOAT`                             | Read noise std-dev in DN (default: 1.5)                                                                                                                                  |
| `--no-shot-noise`                                | Disable photon shot noise                                                                                                                                                |
| `--noise-level FLOAT`                            | Overall noise level, 0.0–1.0 (default: 0.5)                                                                                                                              |
| `--sensor-black-level INT`                       | DN for zero linear signal (default: 256)                                                                                                                                 |
| `--black-level INT`                              | Override DNG BlackLevel tag                                                                                                                                              |
| `--white-level INT`                              | Override DNG WhiteLevel tag                                                                                                                                              |
| `--r-filter`, `--g-filter`, `--b-filter`         | Custom filter response vectors expressed with Rec.2020 primaries wavelengths (default: Metameric Spectrums of Rec.709 primaries constructed from the three wavevlengths) |

## Technical details

### Color science pipeline

1. **Input**: The script expects a linear Rec.2020 RGB EXR (open-domain linear).
2. **Spectral basis**: Rec.2020 primaries are treated as a spectral basis (R ≈ 630 nm, G ≈ 532 nm, B ≈ 467 nm).
3. **Filter response**: By default, Rec.709 primaries are expressed as linear combinations in Rec.2020 RGB space, acting as a pseudo-spectral reconstruction of the camera's color filters.
4. **ColorMatrix1**: Computed as `F · inv(M_XYZ_from_Rec2020)`, where `F` is the 3×3 filter matrix and `M_XYZ_from_Rec2020` is the Rec.2020-to-XYZ primary matrix scaled to D65. The result is written as DNG rationals (tag 50721).

### Physical sensor model

The sensor simulation operates in electron space for physical accuracy:

- **Gain**: `gain = (65535 - black_level) · (ISO / native_ISO)` DN per linear unit.
- **Full well**: Derived from the clip point (`0.18 · 2^max_stop`) and scaled by `noise_level`.
- **Photon shot noise**: Applied as `N(0, sqrt(signal_e))` in electron counts.
- **Read noise**: Applied as `N(0, read_noise_e)` in electron counts.
- **ADC quantization**: The final electron signal is converted back to DN and clipped to the sensor's dynamic range.

### Encoding strategies

| Mode                                       | Behavior                                                                                                                                                       |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Fixed middle gray** (default for 32-bit) | open-domain linear 0.18 is pinned to output's 0.18 equivalent code value.                                                                                      |
| **Adaptive range** (default for integer)   | The full sensor DR \[`min_stop`, `max_stop`] is normalized to the file range \[0, `file_max`]. Middle gray shifts downwards as sensor dynamic range increases. |

For large sensor dynamic range, using 32 bit float output is recommended.

#### Power encoding (adaptive integer only)

When `--pow-encode-int` is enabled, the normalized sensor signal is raised to `1/power_exponent` before quantization. This allocates more codes to lower regions, reducing posterization in low-light areas. A `LinearizationTable` (DNG tag 0xC618) is embedded so compliant raw decoders automatically reverse the curve. **Note:** 32-bit float DNGs do not support `LinearizationTable`; power encoding is automatically disabled for float output.

### DNG metadata

The output DNG includes:

- `CFAPattern` & `CFARepeatPatternDim`: correct Bayer layout
- `ColorMatrix1`: XYZ → camera native (a mathematical space created by interpreting bayer filtered intensities as RGB channels, resulting in impossible primaries)
- `CameraCalibration1`: identity
- `AsShotNeutral`: computed from filter response integrals
- `BlackLevel` / `WhiteLevel`: sensor floor and clip (or user overrides)
- `BaselineExposure`: present in adaptive mode to communicate middle-gray shift to the raw editor (note: some software like Darktable ignore it)
- `LinearizationTable`: present when power encoding is used on integer output
- `CalibrationIlluminant1`: D65

### Noise and dynamic range

The `min_stop` and `max_stop` parameters define the sensor's usable dynamic range relative to 0.18 middle gray. For example, with defaults of `-10.0` and `+4.0`, the sensor covers **14 stops** of dynamic range. The `noise_level` parameter scales the full-well capacity, which in turn affects the signal-to-noise ratio and the granularity of shot noise.
