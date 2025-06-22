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

year = struct.unpack(">L", section[12 : 14])
print(f"{year=}")
