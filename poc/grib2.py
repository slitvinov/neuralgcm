import struct
import sys

f = open(sys.argv[1], "rb")
indicator = f.read(16)
magic = indicator[:4]
assert magic == b'GRIB'
discipline = indicator[6]
assert edition == 0 # Meteorological Products
edition = indicator[7]
assert edition == 2
total_length, = struct.unpack(">Q", indicator[8:])
print(f"{discipline=} {edition=} {total_length=}")

section_length = struct.unpack(">L", f.read(4)) 

print(f"{section_length=}")

# f.close()
