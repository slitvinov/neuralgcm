import struct
import sys

f = open(sys.argv[1], "rb")
indicator = f.read(16)
magic = indicator[:4]
assert magic == b'GRIB'
discipline = indicator[6]
assert discipline == 0 # Meteorological Products
edition = indicator[7]
assert edition == 2
total_length, = struct.unpack(">Q", indicator[8:])
print(f"{discipline=} {edition=} {total_length=}")

section = f.read(21)
section_length, = struct.unpack(">L", section[:4])
assert section_length == 21
# f.close()

significance = section[11]
year, = struct.unpack(">H", section[12 : 14])
month = section[15]
day = section[16]
print(f"{significance=} {year=} {month=} {day=}")

# Local Use Section
section_length, = struct.unpack(">L", f.read(4))
f.read(section_length - 4)

# Grid Definition Section
section_length, = struct.unpack(">L", f.read(4))
section = b'0000' + f.read(section_length - 4)

npoint, = struct.unpack(">L", section[6:10])
# assert section[10] == 0, "number of octets for optional list"
# assert section[11] == 0, "number of octets for optional list"

grid_template_number, = struct.unpack(">H", section[12:14])

Grid = { 40 : "Gaussian Latitude/Longitude",
         50 : "Spherical Harmonic Coefficients"}
print(f"{Grid{grid_template_number}=}")
print(f"{npoint=}")
