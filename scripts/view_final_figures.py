from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BASE_DIR = Path(r"D:\Etawah_Water_Quality")
FIGURE_DIR = BASE_DIR / "data" / "final_figures"

files = sorted(FIGURE_DIR.glob("*.png"))

print(f"Found {len(files)} figures.")

for file in files:
    print(file.name)

# Display each figure in its own window
for file in files:
    img = mpimg.imread(file)

    plt.figure(figsize=(10, 7))
    plt.imshow(img)
    plt.axis("off")
    plt.title(file.name)
    plt.tight_layout()

plt.show()