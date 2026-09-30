"""Keep the original cup anchor fixed while testing object placement.

The anchor is the initial cup pose from the baseline textured-contact run.
Keeping it here makes the replay independent of an ignored local ``runs/``
report, so a fresh checkout can use the same trajectory registration.
"""
from copy import deepcopy
from yubi_isaac_sim_env.policies.umi_pose_replay import predict as original_predict

ANCHOR = (0.0, 0.0, 0.75)


def predict(observation, step, episode):
    fixed = deepcopy(observation)
    fixed["objects"]["cup"]["position_m"] = list(ANCHOR)
    return original_predict(fixed, step, episode)
