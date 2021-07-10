# SAR Image UV Coordinates 
# objects format: 
# [h11, h12, h13, h21, h22, h23, h31, h32, h33] 
# where h## is a 3x3 homography matrix (h31 = dx and h32 = dy, see QTransform)
Header header
float32[] plane_uv_coords
float32[] plane_xyz_coords
float32[] K
float32[] homography
float32[] pose

