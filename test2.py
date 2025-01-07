import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the original image
# Replace 'marker.jpg' with the path to your marker image
image = cv2.imread('/Users/areinnk/Desktop/3D Vision/Second Quarter/ArucoAR/create_marker/marker.png')

# Convert from BGR to RGB for matplotlib display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_enhanced = cv2.equalizeHist(gray)

# 3. Apply Gaussian blur to reduce noise
# (kernel size and sigma can be adjusted)
blur = cv2.GaussianBlur(gray, (5, 5), 1.4)

# 4. Compute gradient magnitude (for demonstration, we can do a Sobel magnitude)
#   This is not strictly the same as the final Canny edges, 
#   but it illustrates how edges appear before thresholding and non-max suppression.
sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(sobelx**2 + sobely**2)
magnitude = np.uint8(np.clip(magnitude, 0, 255))

# 5. Apply the full Canny algorithm
# (threshold values can be adjusted to see different strictness in edges)
canny_edges = cv2.Canny(blur, threshold1=100, threshold2=200)

# --- Visualization in Matplotlib ---
fig, ax = plt.subplots(1, 5, figsize=(15, 4))

# Original
ax[0].imshow(image_rgb)
ax[0].set_title('1) Original')
ax[0].axis('off')

# Grayscale
ax[1].imshow(gray, cmap='gray')
ax[1].set_title('2) Grayscale')
ax[1].axis('off')

# Blurred
ax[2].imshow(blur, cmap='gray')
ax[2].set_title('3) Gaussian Blur')
ax[2].axis('off')

# Gradient Magnitude (illustration)
ax[3].imshow(magnitude, cmap='gray')
ax[3].set_title('4) Gradient Magnitude')
ax[3].axis('off')

# Final Canny Output
ax[4].imshow(canny_edges, cmap='gray')
ax[4].set_title('5) Canny Edges')
ax[4].axis('off')

plt.tight_layout()
plt.show()
