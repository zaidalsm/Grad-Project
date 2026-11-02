"""Task 1: read an image and print its dimensions."""
import sys
import cv2


def print_image_info(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Could not read image: {path}")
        return None
    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1
    print(f"File: {path}")
    print(f"Height: {h}, Width: {w}, Channels: {channels}")
    return h, w, channels


if __name__ == "__main__":
    # Usage: python read_image.py my_image.jpg   (or edit the default below)
    image_path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    print_image_info(image_path)