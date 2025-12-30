from pathlib import Path

from src.data_preprocessing import load_gtsrb_data

# Load the data
image_paths, labels = load_gtsrb_data(Path("data/raw"))

# Check what paths were actually loaded
print(f"Total images loaded: {len(image_paths)}")
print(f"Sample paths:\n{image_paths[:5]}")

# Check if any test images were loaded
test_images = [p for p in image_paths if "Test" in p or "test" in p]
print(f"\nTest images found: {len(test_images)}")
if test_images:
    print(f"Sample test paths:\n{test_images[:3]}")
    # Check if the specific image exists in the loaded paths
    specific_img = [p for p in image_paths if "08477" in p]
    print(f"\n08477.png in loaded paths: {specific_img}")
