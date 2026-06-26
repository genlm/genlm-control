# Vendored from llamppl.inference.smc_record (llamppl>=0.2.2).
#
# Provenance: `SMCRecord` is a verbatim copy of llamppl's SMCRecord, vendored
# so that genlm-control can drop the llamppl runtime dependency while
# preserving the exact JSON record contract consumed by `viz.py` and
# `html/smc.html`. The standalone `string_for_serialization` helper reproduces
# the per-particle string emitted by the old `SequenceModel.string_for_serialization`
# (i.e. `"|".join(escape(y) for y in token_ctx)`), so the serialized JSON is
# byte-for-byte identical to the previous llamppl-backed path.
import json

from genlm.control.util import escape


def string_for_serialization(ctx):
    """Serialize a particle's token context exactly as the old SequenceModel did.

    Args:
        ctx (list): A particle's token context (list of tokens / units).

    Returns:
        str: The escaped, pipe-joined string used in the SMC visualization JSON.
    """
    return "|".join(escape(y) for y in ctx)


class SMCRecord:
    def __init__(self, n):
        self.history = []
        self.most_recent_weights = [0.0 for _ in range(n)]
        self.step_num = 1

    def prepare_string(self, s):
        # If the string doesn't have <<< and >>>, prepend <<<>>> at the front.
        if "<<<" not in s and ">>>" not in s:
            return f"<<<>>>{s}"
        return s

    def particle_dict(self, particles):
        return [
            {
                "contents": self.prepare_string(p.string_for_serialization()),
                "logweight": (
                    "-Infinity" if p.weight == float("-inf") else str(float(p.weight))
                ),
                "weight_incr": str(
                    float(p.weight) - float(self.most_recent_weights[i])
                ),
            }
            for (i, p) in enumerate(particles)
        ]

    def add_init(self, particles):
        self.history.append(
            {
                "step": self.step_num,
                "mode": "init",
                "particles": self.particle_dict(particles),
            }
        )
        self.most_recent_weights = [p.weight for p in particles]

    def add_smc_step(self, particles):
        self.step_num += 1
        self.history.append(
            {
                "step": self.step_num,
                "mode": "smc_step",
                "particles": self.particle_dict(particles),
            }
        )
        self.most_recent_weights = [p.weight for p in particles]

    def add_resample(self, ancestor_indices, particles):
        self.step_num += 1
        self.most_recent_weights = [
            self.most_recent_weights[i] for i in ancestor_indices
        ]

        self.history.append(
            {
                "mode": "resample",
                "step": self.step_num,
                "ancestors": [int(a) for a in ancestor_indices],
                "particles": self.particle_dict(particles),
            }
        )

        self.most_recent_weights = [p.weight for p in particles]

    def to_json(self):
        return json.dumps(self.history)
