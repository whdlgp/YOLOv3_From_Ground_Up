import os
from PIL import Image
import matplotlib.pyplot as plt
# Analyze resolutions of images in a given directory.
# image_dir (str): Path to the directory containing images.
def analyze_image_resolutions(image_dir):
    widths, heights = [], []

    # Iterate over all images in the directory
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                file_path = os.path.join(root, file)
                try:
                    # Open image and get size
                    with Image.open(file_path) as img:
                        width, height = img.size
                        widths.append(width)
                        heights.append(height)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    # Analyze results
    if widths and heights:
        print(f"Number of images: {len(widths)}")
        print(f"Width - Min: {min(widths)}, Max: {max(widths)}, Mean: {sum(widths) / len(widths):.2f}")
        print(f"Height - Min: {min(heights)}, Max: {max(heights)}, Mean: {sum(heights) / len(heights):.2f}")

        # Plot histograms
        plt.figure(figsize=(12, 6))

        # Width histogram
        plt.subplot(1, 2, 1)
        plt.hist(widths, bins=20, color='blue', alpha=0.7)
        plt.title("Image Width Distribution")
        plt.xlabel("Width (pixels)")
        plt.ylabel("Frequency")

        # Height histogram
        plt.subplot(1, 2, 2)
        plt.hist(heights, bins=20, color='green', alpha=0.7)
        plt.title("Image Height Distribution")
        plt.xlabel("Height (pixels)")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()
    else:
        print("No valid images found in the directory.")

if __name__ == "__main__":
    image_dir = "dataset/VOCdevkit/VOC2007/JPEGImages"
    analyze_image_resolutions(image_dir)