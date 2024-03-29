import pyrealsense2 as rs

width = 640
height = 480
fps = 30

config = rs.config()
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

# ストリーミング開始
pipeline = rs.pipeline()
profile = pipeline.start(config)
depth_intrinsics = rs.video_stream_profile(profile.get_stream(rs.stream.depth)).get_intrinsics()
color_intrinsics = rs.video_stream_profile(profile.get_stream(rs.stream.color)).get_intrinsics()

print("depth_intrinsics")
print(depth_intrinsics)
print()
print("color_intrinsics")
print(color_intrinsics)
print()
#
# depth_intrinsics
# [ 640x480  p[320.936 242.858]  f[595.591 595.591]  Brown Conrady [0 0 0 0 0] ]
#
# color_intrinsics
# [ 640x480  p[313.278 246.359]  f[616.517 615.858]  Inverse Brown Conrady [0 0 0 0 0] ]
