import struct
import sys

def decode_short(data):
    # sign-and-magnitude not standard two's complement
    uint16_val, = struct.unpack('>H', data)
    is_negative = uint16_val & 0x8000
    magnitude = uint16_val & 0x7FFF
    return -magnitude if is_negative else magnitude

CENTER = {98: "European Center for Medium-Range Weather Forecasts (RSMC)"}
# https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table5-7.shtml
PRECISION = {
    1: "IEEE 32-bit (I=4 in Section 7)",
    2: "IEEE 64-bit (I=8 in Section 7)",
    3: "IEEE 128-bit (I=16 in Section 7)"
}
f = open(sys.argv[1], "rb")
while True:
    # Section 0 - Indicator Section
    section = f.read(16)
    if section == b'':
        break
    assert section[:4] == b'GRIB'
    assert section[6] == 0, "Meteorological Products"
    assert section[7] == 2, "Edition"
    total_length, = struct.unpack(">Q", section[8:])

    # Section 1 - Identification Section
    section_length, = struct.unpack(">L", f.read(4))
    section = b'0000' + f.read(section_length - 4)
    assert section[4] == 1, "Number of the section"
    center, subcenter = struct.unpack(">HH", section[5:9])
    assert section[11] == 1, "Start of Forecast"
    year, = struct.unpack(">H", section[12:14])
    month, day, hour, minute, second = section[15:20]

    print(f"{CENTER[center]=}")
    print(f"{year=} {month=} {day=} {minute=} {second=}")

    # Section 2 - Local Use Section
    section_length, = struct.unpack(">L", f.read(4))
    section = b'0000' + f.read(section_length - 4)
    assert section[4] == 2, "Number of the section"

    # Section 3 - Grid Definition Section
    section_length, = struct.unpack(">L", f.read(4))
    section = b'0000' + f.read(section_length - 4)
    assert section[4] == 3, "Number of the section"
    assert section[5] == 0, "Source of Grid Definition"
    npoint, = struct.unpack(">L", section[6:10])
    assert section[10] == 0, "number of octets for optional list is not zero"

    template_number, = struct.unpack(">H", section[12:14])
    assert template_number == 50, "Spherical harmonic coefficients"
    print(f"{npoint=}")
    ## pentagonal resolution parameter
    J, = struct.unpack(">L", section[14:18])
    K, = struct.unpack(">L", section[18:22])
    M, = struct.unpack(">L", section[22:26])
    method = section[26]
    order = section[27]
    assert method == 1
    assert order == 1
    # https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table3-6.shtml
    print(f"{J=} {K=} {M=} {method=} {order=}")

    # Section 4 - Product Definition Section
    section_length, = struct.unpack(">L", f.read(4))
    section = b'0000' + f.read(section_length - 4)
    assert section[4] == 4, "Number of the section"
    template_number, = struct.unpack(">H", section[7:9])
    assert template_number == 0, "Analysis or forecast at a horizontal level..."
    parameter_category, parameter_number, generating_process = section[9:12]
    #### assert parameter_category == 2, "Momentum"
    assert parameter_number in (12, 13), "Relative Vorticity"
    print(f"{parameter_number=}")
    assert generating_process == 0, "Analysis"

    # Section 5 - Data Representation Section
    section_length, = struct.unpack(">L", f.read(4))
    section = b'0000' + f.read(section_length - 4)
    assert section[4] == 5, "Number of the section"
    npoint0, = struct.unpack(">L", section[5:9])
    assert npoint0 == npoint, "npoint do not much"
    template_number, = struct.unpack(">H", section[9:11])
    assert template_number == 51, "Spectral Data - Complex Packing"
    # https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_temp5-51.shtml
    R, = struct.unpack(">f", section[11:15])
    E = decode_short(section[15:17])
    D = decode_short(section[17:19])
    nbits_packed = section[19]
    L, Js, Ks, Ms, Ts = struct.unpack(">lHHHL", section[20:34])
    precision = section[34]
    print(f"{R=:.2e} {E=} {D=} {nbits_packed=}")
    print(f"{L=} {Js=} {Ks=} {Ms=} {Ts=}")
    print(f"{PRECISION[precision]=}")

    # Section 6 - Bit Map Section
    section_length, = struct.unpack(">L", f.read(4))
    section = b'0000' + f.read(section_length - 4)
    assert section[4] == 6, "Number of the section"
    assert section[5] == 255, "A bit map does not apply to this product."

    # Section 7 - Data Section
    section_length, = struct.unpack(">L", f.read(4))
    section = f.read(section_length - 4)
    assert section[0] == 7, "Number of the section"
    section = section[1:]
    print(f"{section_length=}")
    # check of the total size (packed + unpack)
    assert 2 * (npoint - Ts) + 4 * Ts == len(section)
    unpk = list(struct.iter_unpack(">ff", section[:4 * Ts]))
    pked = list(struct.iter_unpack(">hh", section[4 * Ts:]))
    i = 0
    j = 0
    bscale = 2.0**(E)
    dscale = 10.0**(-D)
    tscale = L * 1e-6
    with open("value", "w") as fv:
        for m in range(M + 1):
            for n in range(m, J + 1):
                if n <= Js and m <= Ms:
                    x, y = unpk[i]
                    i += 1
                else:
                    x, y = pked[j]
                    j += 1
                    pscale = (n * (n + 1))**(-tscale)
                    x = (x * bscale + R) * dscale * pscale
                    y = (y * bscale + R) * dscale * pscale
                fv.write(f"{x:.10e}\n")
                fv.write(f"{y:.10e}\n")

    # Section 8 - End Section
    section = f.read(4)
    assert section == b'7777', "7777"

    pad = 64 - f.tell() % 64
    f.seek(pad, 1)
    break
