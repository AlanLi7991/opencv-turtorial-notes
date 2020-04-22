import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. read
img = cv.imread('lena.tif', cv.IMREAD_GRAYSCALE)

# 2. transform to frequency

# fourier transform with Fast Fourier Transform (FFT)
fourier = np.fft.fft2(img)

# fourier shift , use center as 0,0 point
f_shift = np.fft.fftshift(fourier)

# 3. make spectrum

# absolute the shift fourier, because it has negative value
fourier_abs = np.abs(fourier)

# use e as base, natural logarithm of fourier, get imaginary exponential
# ref: https://www.zhihu.com/question/284620618
exponential = np.log(fourier_abs)

# amplify the exponential, get fourier result spectrum
img_f_spectrum = 20*exponential

# use same way get shift fourier spectrum
img_fs_spectrum = 20*np.log(np.abs(f_shift))

# 4. make high pass filter

# get shape
rows, cols = img.shape[0], img.shape[1]
c_row, c_col = rows//2, cols//2

# assign low frequency to zero in fourier result
fourier[0:30, 0:30] = 0
fourier[rows-30:rows, cols-30:cols] = 0
fourier[0:30, cols-30:cols] = 0
fourier[rows-30:rows, 0:30] = 0

# make low frequency to zero, just keep high frequency
f_shift[c_col-30:c_col+31, c_row-30:c_row+31] = 0

# 5. make filter spectrum

# after apply high pass filter on fourier result
img_ff_spectrum = 20*np.log(np.abs(fourier))

# after apply high pass filter on shift fourier absolute value
img_fsf_spectrum = 20*np.log(np.abs(f_shift))

# 6. invert fft to image

# invert direct frequency to image
img_f_back = np.fft.ifft2(fourier)

# take real part , abort imaginary part
img_f_back = np.real(img_f_back)

# invert shift change fourier center to left top
f_i_shift = np.fft.ifftshift(f_shift)

# invert shifted frequency to image
img_fs_back = np.fft.ifft2(f_i_shift)

# take real part , abort imaginary part
img_fs_back = np.real(img_fs_back)

# 7. use opencv take dft

# 2D Discrete Fourier Transform (DFT)
# returns the same result as previous
# but with two channels (real, imaginary)
dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)

# shift
dft_shift = np.fft.fftshift(dft)

# take real & imaginary
real_part = dft_shift[:, :, 0]
imaginary_part = dft_shift[:, :, 1]

# magnitude = sqrt(pow(real, 2) + pow(imaginary, 2)), make spectrum
img_cv_spectrum = 20*np.log(cv.magnitude(real_part, imaginary_part))

# 8. make low pass filter

# create a mask first, center square is 1, remaining all zeros
mask = np.zeros(dft_shift.shape, np.uint8)
mask[c_col-30:c_col+30, c_row-30:c_row+30] = 1

# 9. apply low pass filter

# apply mask denote use low pass filter
dft_shift = dft_shift*mask

# get masked result filter
dft_shift_complex = dft_shift[..., 0] + 1j * dft_shift[..., 1]
img_cvf_spectrum = 20*np.log(np.abs(dft_shift_complex))

# 10. invert to image after low pass filter

# invert shift of dft
dft_i_shift = np.fft.ifftshift(dft_shift)

# invert idft
img_dft_back = cv.idft(dft_i_shift)

# sqrt(pow(real, 2) + pow(imaginary, 2)) fill up image
img_dft_back = cv.magnitude(img_dft_back[:, :, 0], img_dft_back[:, :, 1])

# 11. make a image sample compare with spectrum
compare_spectrum = np.ones(img_f_spectrum.shape, np.uint8)
compare_spectrum[c_col-30:c_col+31, c_row-30:c_row+31] = 0

# 12. show result
images = [
    img,
    img_f_spectrum, img_fs_spectrum,
    img_ff_spectrum, img_fsf_spectrum,
    img_f_back, img_fs_back,
    img_cv_spectrum, img_cvf_spectrum, img_dft_back,
    compare_spectrum
]
colors = [
    "gray",
    "gray", "gray",
    "gray", "gray",
    "jet", "jet",
    "gray", "gray", "jet",
    "gray"
]
titles = [
    "Input",
    "Fourier", "Shift Fourier",
    "Fourier Filter", "Shift Fourier Filter",
    "Fourier Back", "Shift Fourier Back",
    "CV Fourier", "Low Pass Filter", "Low Pass Back",
    "Compare Spectrum"
]
for i in range(len(images)):
    plt.subplot(3, 4, i+1)
    plt.imshow(images[i], colors[i])
    plt.title(titles[i])
        
plt.show()
