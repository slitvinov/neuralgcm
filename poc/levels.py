import struct
import sys

f = open(sys.argv[1], "rb")
# Section 0 - Indicator Section
section = f.read(16)
assert section[:4] == b'GRIB'
discipline = section[6]
assert discipline == 0, "Discipline"
assert section[7] == 2, "Edition"
total_length, = struct.unpack(">Q", section[8:])

# Section 1 - Identification Section
section_length, = struct.unpack(">L", f.read(4))
section = b'0000' + f.read(section_length - 4)
assert section[4] == 1, "Number of the section"

# Section 2 - Local Use Section
section_length, = struct.unpack(">L", f.read(4))
section = b'0000' + f.read(section_length - 4)
assert section[4] == 2, "Number of the section"

# Section 3 - Grid Definition Section
section_length, = struct.unpack(">L", f.read(4))
section = b'0000' + f.read(section_length - 4)
assert section[4] == 3, "Number of the section"

# Section 4 - Product Definition Section
section_length, = struct.unpack(">L", f.read(4))
section = b'0000' + f.read(section_length - 4)
assert section[4] == 4, "Number of the section"
ncoord, template_number = struct.unpack(">HH", section[5:9])
template_number, = struct.unpack(">H", section[7:9])
assert template_number == 0, "Analysis or forecast at a horizontal level or in a horizontal layer at a point in time"
parameter_category, parameter_number, generating_process = section[9:12]
assert generating_process == 0, "Analysis"
background_generating_process, generating_process = section[12:14]
type1, factor1 = section[22:24]
value1, = struct.unpack(">L", section[24:28])
type2, factor2 = section[28:30]
value2, = struct.unpack(">L", section[30:34])
assert type1 == 105, "type1 is Hybrid Levels"
assert type2 == 255, "type2 is Missing"
buf = section[34:]
coord = [x for x, in struct.iter_unpack(">f", buf)]
assert len(coord) == ncoord
scale = 10 ** factor1
with open("levels.raw", "wb") as out:
    for c in coord:
        out.write(struct.pack("<f", c * scale))
