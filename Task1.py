import cv2
import numpy as np

# Load the image
image = cv2.imread('e:/AdobeRound2/AdobeR2Final/lineInput3.png', cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded
if image is None:
    print("Error: Image not found or cannot be loaded.")
    exit()

# Preprocessing: Apply Gaussian Blur to reduce noise
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred_image, 50, 150)

# Detect lines using Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

# Detect circles using Hough Circle Transform with refined parameters
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=50, param2=30, minRadius=10, maxRadius=50)

# Detect ellipses by finding contours and fitting ellipses
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ellipses = []
rectangles = []
rounded_rectangles = []
polygons = []
stars = []

# Function to check if a contour is a rounded rectangle
def is_rounded_rectangle(approx):
    if len(approx) == 4:
        angles = []
        for i in range(4):
            pt1 = approx[i][0]
            pt2 = approx[(i + 1) % 4][0]
            pt3 = approx[(i + 2) % 4][0]
            angle = np.arctan2(pt3[1] - pt2[1], pt3[0] - pt2[0]) - np.arctan2(pt1[1] - pt2[1], pt1[0] - pt2[0])
            angles.append(abs(angle))
        return all(a < np.pi / 2 for a in angles)
    return False

# Function to check if a contour is a star shape
def is_star_shape(approx):
    if len(approx) >= 10:
        center = np.mean(approx, axis=0).flatten()
        distances = [np.linalg.norm(pt[0] - center) for pt in approx]
        mean_distance = np.mean(distances)
        return all(abs(d - mean_distance) < mean_distance * 0.2 for d in distances)
    return False

# Identify shapes
for contour in contours:
    # Detect rectangles and rounded rectangles
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
        if is_rounded_rectangle(approx):
            rounded_rectangles.append(approx)
        else:
            rectangles.append(approx)

    # Detect polygons
    if len(approx) >= 5:
        polygons.append(approx)

    # Detect ellipses
    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            if (axes[0] > 15 and axes[1] > 15) and (0.5 < axes[0] / axes[1] < 2.0):
                ellipses.append(ellipse)
        except cv2.error as e:
            # Handle fitting error if any
            pass

    # Detect star shapes
    if is_star_shape(approx):
        stars.append(approx)

# Convert grayscale image to BGR color image for visualization
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Draw lines on the image
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw lines in green

# Draw circles on the image
if circles is not None:
    for circle in circles[0, :]:
        x, y, r = circle
        cv2.circle(output_image, (int(x), int(y)), int(r), (0, 0, 255), 2)  # Draw circles in red
        cv2.circle(output_image, (int(x), int(y)), 2, (0, 0, 255), 3)  # Draw circle center in red

# Draw ellipses on the image
for ellipse in ellipses:
    center, axes, angle = ellipse
    cv2.ellipse(output_image, (tuple(map(int, center)), tuple(map(int, axes)), angle), (255, 0, 0), 2)  # Draw ellipses in blue

# Draw rectangles on the image
for rect in rectangles:
    x, y, w, h = cv2.boundingRect(rect)
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Draw rectangles in yellow

# Draw rounded rectangles on the image
for rect in rounded_rectangles:
    cv2.drawContours(output_image, [rect], -1, (255, 255, 0), 2)  # Draw rounded rectangles in cyan

# Draw polygons on the image
for poly in polygons:
    cv2.drawContours(output_image, [poly], -1, (0, 255, 255), 2)  # Draw polygons in yellow

# Draw star shapes on the image
for star in stars:
    cv2.drawContours(output_image, [star], -1, (0, 255, 0), 2)  # Draw stars in green

# Count the number of lines, circles, ellipses, rectangles, rounded rectangles, polygons, and stars found
num_lines = len(lines) if lines is not None else 0
num_circles = circles.shape[1] if circles is not None else 0
num_ellipses = len(ellipses)
num_rectangles = len(rectangles)
num_rounded_rectangles = len(rounded_rectangles)
num_polygons = len(polygons)
num_stars = len(stars)

# Print the number of each shape found
print("Number of Lines Found: ", num_lines)
print("Number of Circles Found: ", num_circles)
print("Number of Ellipses Found: ", num_ellipses)
print("Number of Rectangles Found: ", num_rectangles)
print("Number of Rounded Rectangles Found: ", num_rounded_rectangles)
print("Number of Polygons Found: ", num_polygons)
print("Number of Stars Found: ", num_stars)

# Display the result
# cv2.imshow('Detected Shapes', output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
