import dataclasses
from dinosaur import typing
import tree_math


@tree_math.struct
class State:
    u: typing.Array
    v: typing.Array
    t: typing.Array
    z: typing.Array
    sim_time: float
    tracers: dict[str, typing.Array] = dataclasses.field(default_factory=dict)
    diagnostics: dict[str,
                      typing.Array] = dataclasses.field(default_factory=dict)
