# vtr/msg/FrameFeatures.msg
# Published by step3_xfeat_node.py  (was: step3_orb_node.py)
# Subscribed by step4_keyframe_scorer_node.py
#
# Contains everything the keyframe scorer needs to decide
# whether to save this frame to the memory graph.
# Raw pixel data is deliberately excluded — keeps messages small.

Header   header

# Timing
float64  timestamp           # unix time at capture

# Feature summary
int32    n_keypoints         # total XFeat keypoints detected
float32  quality_hint        # n_keypoints / 500 clamped 0-1
                             # scorer will recompute full score

# XFeat descriptors — flattened row-major
# Shape when reshaped: (n_keypoints, 64) float32
# Receiver: descs = np.array(msg.descriptors_flat, dtype=np.float32).reshape(-1, 64)
float32[] descriptors_flat

# Keypoint geometry — parallel arrays, index i = keypoint i
float32[] keypoint_x         # pixel x position
float32[] keypoint_y         # pixel y position
float32[] keypoint_angle     # orientation in degrees (zero for XFeat)
float32[] keypoint_size      # scale size (zero for XFeat)
int32[]   keypoint_octave    # pyramid level (zero for XFeat)
