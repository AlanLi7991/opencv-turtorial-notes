import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Q1: What is HDR?
# document definition:
#   "High-dynamic-range imaging (HDRI or HDR) is a technique
#    used in imaging and photography
#    to reproduce a greater dynamic range of luminosity
#    than is possible with standard digital imaging or photographic techniques."
# in simple word
# HDR is a technique to get a greater dynamic range of luminosity than it should be.
#
# luminosity:
# luminous quality
#
# Q2: what problem HDR want to solve?
# the document proposes a hypothesis
#   "While the human eye can adjust to a wide range of light conditions,
#    most imaging devices use 8-bits per channel,
#    so we are limited to only 256 levels."
# in simple word, (In a word)machine is not as good as the human eye can handle the condition-light
# so in sometimes
#   "bright regions may be overexposed, while the dark ones may be underexposed"
# HDR can solve this problem with multiple images.
#
# Q2: the theory of HDR?
# HDR steps
#   1. use sequence images taken with different exposure values.
#   2. convert 8 bits per channel to 32-bit float values to get a much wider dynamic range
#   3. combine these exposures via algorithms to estimate the camera’s response function
#   4. merge the results to a single image
#   5. converted back to 8-bit(this process is called tone mapping) to view it
#
# Q3: the popular algorithms of HDR?
# there are two algorithms
#   1. Debevec
#   2. Robertson
#
# Q4: what are the differences between those algorithms?
# I don't know, as my colleague said:
# "every method that uses multiple exposure results merging to a single image can be called HDR
# Debevec & Robertson only two of them, most of the HDR algorithm is private to company or laboratory"
#

# Loading exposure images into a list
img_fn = [
    "StLouisArchMultExpEV+4.09.JPG",
    "StLouisArchMultExpEV+1.51.JPG",
    "StLouisArchMultExpEV-1.82.JPG",
    "StLouisArchMultExpEV-4.72.JPG"
]
# the images should be 1-channel or 3-channels 8-bit (np.uint8)
img_list = [cv.imread(fn) for fn in img_fn]
# the exposure times need to be float32 and in seconds.
exposure_times = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)


# Merge exposures to HDR image
# Notice that the HDR image is of type float32, and not uint8,
# as it contains the full dynamic range of all exposure images.
merge_debevec = cv.createMergeDebevec()
hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy())

merge_robertson = cv.createMergeRobertson()
hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())

# Tonemap HDR image
# We map the 32-bit float HDR data into the range[0..1].
# Actually, in some cases the values can be larger than 1 or lower the 0,
# so notice we will later have to clip the data in order to avoid overflow.
#
# gamma:
#   positive value for gamma correction.
#   Gamma value of 1.0 implies no correction,
#   gamma equal to 2.2f is suitable for most displays.
#   Generally gamma > 1 brightens the image and gamma < 1 darkens it.
tonemap = cv.createTonemap(gamma=2.2)
res_debevec = tonemap.process(hdr_debevec.copy())
res_robertson = tonemap.process(hdr_robertson.copy())

# Exposure fusion using Mertens
merge_mertens = cv.createMergeMertens()
res_mertens = merge_mertens.process(img_list)

# Convert datatype to 8-bit and save
res_debevec_8bit = np.clip(res_debevec*255, 0, 255).astype('uint8')
res_robertson_8bit = np.clip(res_robertson*255, 0, 255).astype('uint8')
res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')

# results
results = [cv.cvtColor(res, cv.COLOR_BGR2RGB) for res in (res_debevec_8bit, res_robertson_8bit, res_mertens_8bit)]

# show
titles = ["ldr_debevec.jpg", "ldr_robertson.jpg", "fusion_mertens.jpg"]
for i in range(len(results)):
    plt.subplot(2, 2, i+1)
    plt.imshow(results[i])
    plt.title(titles[i])
plt.show()
