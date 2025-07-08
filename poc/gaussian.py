import struct
import sys


def decode_short(data):
    # sign-and-magnitude not standard two's complement
    val, = struct.unpack('>H', data)
    sig = (val >> 15) & 1
    mag = val & 0x7FFF
    return -mag if sig else mag


def fma(x, y, z):
    return x * y + z


# https://www.nco.ncep.noaa.gov/pmb/docs/on388/table0.html
CENTER = {
    98: "European Center for Medium-Range Weather Forecasts (RSMC)",
    00: "Melbourne (WMC)"
}

# https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table5-7.shtml
PRECISION = {
    1: "IEEE 32-bit (I=4 in Section 7)",
    2: "IEEE 64-bit (I=8 in Section 7)",
    3: "IEEE 128-bit (I=16 in Section 7)"
}

# https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table0-0.shtml
DISCIPLINE = {
    0: "Meteorological Products",
    1: "Hydrological Products",
    2: "Land Surface Products",
    3: "Satellite Remote Sensing Products",
    4: "Space Weather Products",
    10: "Oceanographic Products",
    20: "Health and Socioeconomic Impacts",
}

# https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table1-0.shtml
VERSION = {
    0: "Experimental",
    1: "7 November 2001",
    2: "4 November 2003",
    3: "2 November 2005",
    4: "7 November 2007",
    5: "4 November 2009",
}

# https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table1-2.shtml
SIGNIFICANCE = {
    0: "Analysis",
    1: "Start of Forecast",
    2: "Verifying Time of Forecast",
    3: "Observation Time",
    4: "Local Time",
    5: "Simulation start",
}

# https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table1-3.shtml
STATUS = {
    0: "Operational Products",
    1: "Operational Test Products",
    2: "Research Products",
}

# https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table1-4.shtml
TYPE = {
    0: "Analysis Products",
    1: "Forecast Products",
}

# https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table3-1.shtml
GRID_TEMPLATE = {
    0: "Latitude/Longitude",
    1: "Rotated Latitude/Longitude",
    2: "Stretched Latitude/Longitude",
    3: "Rotated and Stretched Latitude/Longitude",
    40: "Gaussian Latitude/Longitude",
    50: "Spherical Harmonic Coefficients",
}

# https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-0.shtml
PRODUCT_TEMPLATE = {
    0:
    "Analysis or forecast at a horizontal level or in a horizontal layer at a point in time",
    1:
    "Individual ensemble forecast, control and perturbed, at a horizontal level or in a horizontal layer at a point in time",
}

# https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-1.shtml
PARAMETER_CATEGORY = {
    0: "Temperature",
    1: "Moisture",
    2: "Momentum",
    3: "Mass",
    4: "Short-wave radiation",
    6: "Cloud",
    14: "Trace gases",
}

# https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2.shtml
PARAMETER = {
    0: {
        0: "Temperature",
    },
    1: {
        0: "Specific Humidity",
        14: "Convective Snow",
        83: "Specific Cloud Liquid Water Content",
        84: "Specific Cloud Ice Water Content",
        85: "Specific Rain Water Content",
        86: "Specific Snow Water Content",
    },
    2: {
        8: "Vertical Velocity (Pressure)",
        12: "Relative Vorticity",
        13: "Relative Divergence",
    },
    6: {
        32: "Fraction of Cloud Cover",
    },
    14: {
        0: "Total Ozone",
        1: "Ozone Mixing Ratio",
    }
}

SURFACE_TYPE = {
    105: "Hybrid Level",
    255: "Missing",
}

f = open(sys.argv[1], "rb")
while True:
    # Section 0 - Indicator Section
    section = f.read(16)
    if section == b'':
        break
    assert section[:4] == b'GRIB'
    discipline = section[6]
    assert discipline == 0, "Discipline"
    assert section[7] == 2, "Edition"
    total_length, = struct.unpack(">Q", section[8:])
    print(f"{DISCIPLINE[discipline]=}\n{total_length=}")

    # Section 1 - Identification Section
    section_length, = struct.unpack(">L", f.read(4))
    section = b'0000' + f.read(section_length - 4)
    assert section[4] == 1, "Number of the section"
    center, subcenter = struct.unpack(">HH", section[5:9])
    version, local_version, significance = section[9:12]
    year, = struct.unpack(">H", section[12:14])
    month, day, hour, minute, second = section[15:20]
    status, type_data = section[19:21]

    print(f"{CENTER[center]=} {subcenter=}")
    print(f"{VERSION[version]=} {local_version=}")
    print(f"{SIGNIFICANCE[significance]=}")
    print(f"{year=} {month=} {day=} {minute=} {second=}")
    print(f"{STATUS[status]=} {TYPE[type_data]=}")

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
    noctet, interpetation = section[10:12]
    print(f"{noctet=} {interpetation=}")

    template, = struct.unpack(">H", section[12:14])
    print(f"{GRID_TEMPLATE[template]=}")
    # assert noctet == 0 and interpetation == 0, "no extra points"
    assert template == 40, "Gaussian Latitude/Longitude"

    # Section 4 - Product Definition Section
    section_length, = struct.unpack(">L", f.read(4))
    section = b'0000' + f.read(section_length - 4)
    assert section[4] == 4, "Number of the section"
    ncoord, template_number = struct.unpack(">HH", section[5:9])
    template_number, = struct.unpack(">H", section[7:9])
    print(f"{PRODUCT_TEMPLATE[template_number]=}")
    assert template_number == 0
    parameter_category, parameter_number, generating_process = section[9:12]
    print(f"{parameter_category=}, {parameter_number=}")
    #### assert parameter_category == 2, "Momentum"

    print(f"{PARAMETER_CATEGORY[parameter_category]=}")
    print(f"{PARAMETER[parameter_category][parameter_number]=}")
    assert generating_process == 0, "Analysis"

    background_generating_process, generating_process = section[12:14]
    print(f"{background_generating_process=} {generating_process=}")

    type1, factor1 = section[22:24]
    value1, = struct.unpack(">L", section[24:28])

    type2, factor2 = section[28:30]
    value2, = struct.unpack(">L", section[30:34])
    print(f"{SURFACE_TYPE[type1]=} {factor1=} {value1=}")
    print(f"{SURFACE_TYPE[type2]=} {factor2=} {value2=}")
    # buf = section[34:]
    # coord = [x for x, in struct.iter_unpack(">f", buf)]

    # Section 5 - Data Representation Section
    section_length, = struct.unpack(">L", f.read(4))
    section = b'0000' + f.read(section_length - 4)
    assert section[4] == 5, "Number of the section"
    npoint0, = struct.unpack(">L", section[5:9])
    assert npoint0 == npoint, "npoint do not much"
    template_number, = struct.unpack(">H", section[9:11])

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

    # Section 8 - End Section
    section = f.read(4)
    assert section == b'7777', "7777"

    while True:
        x = f.read(1)
        if x == "":
            exit(0)
        if x == b"G":
            f.seek(f.tell() - 1)
            break
#    pad = 64 - f.tell() % 64
#    f.seek(pad, 1)
    # break
