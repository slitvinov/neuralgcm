import struct
import sys

f = open(sys.argv[1], "rb")

# Section 0 - Indicator Section
section = f.read(16)
assert section[:4] == b'GRIB'
assert section[6] == 0, "Meteorological Products"
assert section[7] == 2, "Edition"
total_length, = struct.unpack(">Q", section[8:])

# Section 1 - Identification Section
section_length, = struct.unpack(">L", f.read(4))
section = b'0000' + f.read(section_length - 4)
assert section[4] == 1, "Number of the section"
assert section[11] == 1, "Start of Forecast"
year, = struct.unpack(">H", section[12:14])
month = section[15]
day = section[16]
print(f"{year=} {month=} {day=}")

# Section 2 - Local Use Section
section_length, = struct.unpack(">L", f.read(4))
section = b'0000' + f.read(section_length - 4)
assert section[4] == 2, "Number of the section"

# Section 3 - Grid Definition Section
section_length, = struct.unpack(">L", f.read(4))
section = b'0000' + f.read(section_length - 4)
assert section[4] == 3, "Number of the section"

npoint, = struct.unpack(">L", section[6:10])
assert section[10] == 0, "number of octets for optional list is not zero"

template_number, = struct.unpack(">H", section[12:14])
assert template_number == 50, "Spherical harmonic coefficients"

print(f"{npoint=}")
print(f"{section_length=}")

J, = struct.unpack(">H", section[14:16])
K, = struct.unpack(">H", section[18:20])
M, = struct.unpack(">H", section[22:24])
method = section[26]
order = section[27]
print(f"{J=} {K=} {M=} {method=} {order=}")

# Section 4 - Product Definition Section
section_length, = struct.unpack(">L", f.read(4))
section = b'0000' + f.read(section_length - 4)
assert section[4] == 4, "Number of the section"
template_number, = struct.unpack(">H", section[7:9])
assert template_number == 0, "Analysis or forecast at a horizontal level..."

# Section 5 - Data Representation Section
section_length, = struct.unpack(">L", f.read(4))
section = b'0000' + f.read(section_length - 4)
assert section[4] == 5, "Number of the section"
npoint, = struct.unpack(">L", section[6:10])
template_number, = struct.unpack(">H", section[9:11])
assert template_number == 51, "Spectral Data - Complex Packing"
R, E, D = struct.unpack(">fHH", section[11:19])
nbits = section[19]
L, Js, Ks, Ms, Ts = struct.unpack(">LHHHL", section[20:34])
print(f"{R=} {E=} {D=} {nbits=} {L=}")
print(f"{L=} {Js=} {Ks=} {Ms=} {Ts=}")
print(f"{npoint=}")

# Section 6 - Bit Map Section
section_length, = struct.unpack(">L", f.read(4))
section = b'0000' + f.read(section_length - 4)
assert section[4] == 6, "Number of the section"
assert section[5] == 255, "A bit map does not apply to this product."

# Section 7 - Data Section
section_length, = struct.unpack(">L", f.read(4))
section = f.read(section_length - 4)
assert section[0] == 7, "Number of the section"
print(f"{section_length=}")

# Section 8 - End Section
section = f.read(4)
assert section == b'7777', "7777"
