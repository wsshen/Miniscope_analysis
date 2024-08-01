from scipy import ndimage, datasets
import matplotlib.pyplot as plt
fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side
ascent = datasets.ascent()
result = ndimage.percentile_filter(ascent, percentile=2, size=20)

ax1.imshow(ascent)
ax2.imshow(result)
plt.show()