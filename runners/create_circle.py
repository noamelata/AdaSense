import cv2
import numpy as np
import math


def draw_circle(angles):
    image_size = 256
    image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255 * 0
    center_x, center_y = image_size // 2, image_size // 2
    num_lines = 256
    angle_step = 180 / num_lines
    radius = 128
    cv2.circle(image, (center_x, center_y), radius, (0, 0, 0), -1)
    for i in angles:
        angle_deg = -i * angle_step
        angle_rad = math.radians(angle_deg)
        x1 = int(center_x + radius * math.cos(angle_rad))
        y1 = int(center_y + radius * math.sin(angle_rad))
        x2 = int(center_x - radius * math.cos(angle_rad))
        y2 = int(center_y - radius * math.sin(angle_rad))
        cv2.line(image, (x2, y2), (x1, y1), (255, 255, 255), 1)
    return image

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    image = draw_circle(range(0, 256, 10))
    plt.imshow(image)
    plt.show()