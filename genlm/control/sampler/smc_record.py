# Vendored from llamppl.inference.smc_record (llamppl>=0.2.2).
# JSON record format is consumed by html/smc.html; the two move together.
import json

from genlm.control.util import escape


def string_for_serialization(ctx):
    """Serialize a particle's token context to the pipe-joined, escaped string used in
    the SMC visualization JSON.

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
        # Context length each particle has already been recorded up to. A step stores
        # only what it appended, so the record is linear in sequence length rather
        # than quadratic; the viewer rebuilds each string along the ancestor chain.
        self.recorded_len = [0 for _ in range(n)]

    def particle_dict(self, particles):
        out = []
        for i, p in enumerate(particles):
            context = p.context
            out.append(
                {
                    "contents_incr": string_for_serialization(
                        context[self.recorded_len[i] :]
                    ),
                    "logweight": (
                        "-Infinity"
                        if p.weight == float("-inf")
                        else str(float(p.weight))
                    ),
                    "weight_incr": str(
                        float(p.weight) - float(self.most_recent_weights[i])
                    ),
                }
            )
            self.recorded_len[i] = len(context)
        return out

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
        # A forked row inherits its ancestor's already-recorded context.
        self.recorded_len = [self.recorded_len[i] for i in ancestor_indices]

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
