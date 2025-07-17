import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist # type: ignore

# Load dataset
(x_train, y_train), (_, _) = mnist.load_data()

# Display helper
def show_digit(image, label, title_extra=""):
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label} {title_extra}")
    plt.axis('off')
    plt.show()

# Try to find slanted '2'
for i in range(len(y_train)):
    if y_train[i] == 2:
        img = x_train[i]
        # A rough check for slant using pixel intensity skew
        left = np.sum(img[:, :10])
        right = np.sum(img[:, -10:])
        if left < right:  # likely right-slanted
            show_digit(img, 2, "(Likely Slanted)")
            break

# Try to find an overlapping or confusing '5' and '6'
shown = 0
for i in range(len(y_train)):
    if y_train[i] in [5, 6]:
        img = x_train[i]
        # Heuristic: high stroke density (more "ink") could mean overlapping
        if np.sum(img > 100) > 100:
            show_digit(img, y_train[i], "(Dense/Overlapping Shape)")
            shown += 1
            if shown >= 2:
                break
