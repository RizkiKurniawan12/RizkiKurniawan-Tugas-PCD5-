
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

folder_proyek = Path(__file__).resolve().parent
folder_hasil = folder_proyek / "hasil"
folder_hasil.mkdir(exist_ok=True)

nama_citra = folder_proyek / "citra_sumber.jpg"

citra = cv2.imread(str(nama_citra))

if citra is None:
    raise FileNotFoundError(
        "Citra tidak ditemukan! Pastikan nama file "
        "adalah citra_sumber.jpg dan berada satu folder "
        "dengan main.py."
    )

citra_asli = cv2.cvtColor(citra, cv2.COLOR_BGR2GRAY)

print("Citra berhasil dibaca!")
print("Nama file:", nama_citra.name)
print("Ukuran citra:", citra_asli.shape)
print("Tipe data:", citra_asli.dtype)

mean_3x3 = cv2.blur(citra_asli, (3, 3))
mean_5x5 = cv2.blur(citra_asli, (5, 5))

gaussian_3x3 = cv2.GaussianBlur(
    citra_asli, (3, 3), 0
)

gaussian_5x5 = cv2.GaussianBlur(
    citra_asli, (5, 5), 0
)

median_3x3 = cv2.medianBlur(citra_asli, 3)
median_5x5 = cv2.medianBlur(citra_asli, 5)

kernel_sharpen = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)

sharpening = cv2.filter2D(
    citra_asli, -1, kernel_sharpen
)

hasil_filter = [
    ("Citra Asli", citra_asli),
    ("Mean Filter 3x3", mean_3x3),
    ("Mean Filter 5x5", mean_5x5),
    ("Gaussian Filter 3x3", gaussian_3x3),
    ("Gaussian Filter 5x5", gaussian_5x5),
    ("Median Filter 3x3", median_3x3),
    ("Median Filter 5x5", median_5x5),
    ("Sharpening", sharpening)
]

fig, axes = plt.subplots(2, 4, figsize=(16, 9))

for ax, (judul, gambar) in zip(
    axes.flat, hasil_filter
):
    ax.imshow(gambar, cmap="gray", vmin=0, vmax=255)
    ax.set_title(judul, fontsize=11)
    ax.axis("off")

fig.suptitle(
    "Perbandingan Hasil Filter Spasial",
    fontsize=16
)

plt.tight_layout()

file_perbandingan = folder_hasil / "perbandingan_filter.png"

plt.savefig(
    file_perbandingan,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("\nGambar perbandingan disimpan di:")
print(file_perbandingan)

for judul, gambar in hasil_filter:
    nama_output = (
        judul.lower()
        .replace(" ", "_")
        .replace("x", "kali")
        + ".png"
    )

    cv2.imwrite(
        str(folder_hasil / nama_output),
        gambar
    )

print("Semua gambar hasil filter berhasil disimpan.")

file_csv = folder_hasil / "statistik_intensitas.csv"

with open(
    file_csv, "w", newline="", encoding="utf-8-sig"
) as file:
    penulis = csv.writer(file)

    penulis.writerow([
        "Nama Citra",
        "Rata-rata Intensitas",
        "Minimum",
        "Maksimum",
        "Standar Deviasi"
    ])

    print("\nSTATISTIK INTENSITAS CITRA")
    print("-" * 75)
    print(
        f"{'Nama Citra':<23}"
        f"{'Rata-rata':>12}"
        f"{'Minimum':>10}"
        f"{'Maksimum':>11}"
        f"{'Std Dev':>12}"
    )

    for judul, gambar in hasil_filter:
        rata_rata = float(np.mean(gambar))
        minimum = int(np.min(gambar))
        maksimum = int(np.max(gambar))
        std_dev = float(np.std(gambar))

        penulis.writerow([
            judul,
            round(rata_rata, 2),
            minimum,
            maksimum,
            round(std_dev, 2)
        ])

        print(
            f"{judul:<23}"
            f"{rata_rata:>12.2f}"
            f"{minimum:>10}"
            f"{maksimum:>11}"
            f"{std_dev:>12.2f}"
        )

print("\nStatistik disimpan di:")
print(file_csv)

print("\nPENGOLAHAN CITRA SELESAI!")
print("Silakan periksa folder 'hasil'.")


# Membuat tabel statistik intensitas citra

import csv
import numpy as np
from pathlib import Path

folder_hasil = Path(__file__).resolve().parent / "hasil"
folder_hasil.mkdir(exist_ok=True)

hasil_filter = [
    ("Citra Asli", citra_asli),
    ("Mean Filter 3x3", mean_3x3),
    ("Mean Filter 5x5", mean_5x5),
    ("Gaussian Filter 3x3", gaussian_3x3),
    ("Gaussian Filter 5x5", gaussian_5x5),
    ("Median Filter 3x3", median_3x3),
    ("Median Filter 5x5", median_5x5),
    ("Sharpening", sharpening)
]

file_csv = folder_hasil / "statistik_intensitas.csv"

with open(file_csv, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Nama Citra",
        "Rata-rata Intensitas",
        "Minimum",
        "Maksimum",
        "Standar Deviasi"
    ])

    for nama, gambar in hasil_filter:
        writer.writerow([
            nama,
            round(float(np.mean(gambar)), 2),
            int(np.min(gambar)),
            int(np.max(gambar)),
            round(float(np.std(gambar)), 2)
        ])

print("Tabel statistik berhasil dibuat!")
print("Lokasi file:", file_csv)
