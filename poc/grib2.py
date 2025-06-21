import struct

f = open("era5-levels-members.grib", "rb")

indicator = f.read(16)
magic = indicator[:4]
assert magic == b'GRIB'
discipline = indicator[6]
edition = indicator[7]
length = indicator[8:]
print(discipline, edition, length)

for fmt in "LlqQ":
    for b in "<>=":
        print(b + fmt, struct.unpack(fmt, length))
f.close()
