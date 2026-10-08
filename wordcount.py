import os
import re
import sys
import time
from collections import Counter

# Biar PySpark pakai Python dari venv (cegah error "Python worker failed")
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Kalau java -version masih bukan 17, hapus tanda # di baris bawah:
# os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"

from pyspark.sql import SparkSession

# ============ KONFIGURASI ============
DATA_PATH = "data/dataset2.txt"
DUPLIKASI = 2                       # data digandakan N kali (isi 1 kalau nggak mau)
PARTITION_UJI = [1, 2, 4, 8, 16, 32]
ULANG = 3                           # tiap uji diulang 3x, diambil rata-rata
# =====================================


def bersihkan(line):
    """Tokenisasi: huruf kecil, ambil huruf dan apostrof saja."""
    return re.findall(r"[a-z']+", line.lower())


def hitung_single_machine(path, duplikasi, ulang):
    """WordCount Python biasa. Tokenisasi dan duplikasi disamakan dengan Spark."""
    waktu, counter = [], None
    for _ in range(ulang):
        mulai = time.time()
        counter = Counter()
        for _ in range(duplikasi):
            with open(path, encoding="utf-8") as f:
                for baris in f:
                    counter.update(bersihkan(baris))
        waktu.append(time.time() - mulai)
    return counter, waktu


spark = (SparkSession.builder
         .appName("WordCount")
         .master("local[*]")
         .getOrCreate())
sc = spark.sparkContext
sc.setLogLevel("ERROR")

print("=" * 60)
print("INFO LINGKUNGAN")
print("Versi Spark      :", spark.version)
print("Jumlah core      :", sc.defaultParallelism)
print("=" * 60)

# ---------- 1. DATA MASUK ----------
lines = sc.textFile(DATA_PATH)
baris_asli = lines.count()                       # ACTION
print(f"[1] Baris data asli      : {baris_asli}")

# Gandakan data biar cukup besar untuk eksperimen performa
if DUPLIKASI > 1:
    lines = sc.union([lines] * DUPLIKASI)
print(f"    Setelah duplikasi x{DUPLIKASI}: {lines.count()} baris")

# ---------- 2. PARTITION ----------
print(f"[2] Jumlah partition awal: {lines.getNumPartitions()}")
ukuran = lines.glom().map(len).collect()         # ACTION
print(f"    Isi tiap partition   : {ukuran[:10]}{' ...' if len(ukuran) > 10 else ''}")

# ---------- 3. TRANSFORMATION (lazy) ----------
words = lines.flatMap(bersihkan)                 # 1 baris -> banyak kata
pairs = words.map(lambda w: (w, 1))              # (kata, 1)
counts = pairs.reduceByKey(lambda a, b: a + b)   # jumlahkan per kata (SHUFFLE)

print("[3] Lineage / rencana eksekusi (lihat batas shuffle = pemisah stage):")
print(counts.toDebugString().decode())

# ---------- 4. ACTION ----------
top10 = counts.sortBy(lambda x: -x[1]).take(10)  # ACTION
print("[4] 10 kata terbanyak:")
for kata, jml in top10:
    print(f"    {kata:<15} {jml}")

# ---------- 5. EKSPERIMEN PARTITION ----------
print("[5] Eksperimen pengaruh partition")

# Putaran pemanasan (tidak dicatat) supaya JVM dan worker sudah siap
(lines.repartition(1)
      .flatMap(bersihkan)
      .map(lambda w: (w, 1))
      .reduceByKey(lambda a, b: a + b)
      .count())

print(f"{'Partition':<12}{'Rata-rata (detik)':<20}{'Tiap percobaan'}")
hasil = []
for n in PARTITION_UJI:
    waktu = []
    for _ in range(ULANG):
        mulai = time.time()
        (lines.repartition(n)
              .flatMap(bersihkan)
              .map(lambda w: (w, 1))
              .reduceByKey(lambda a, b: a + b)
              .count())                          # ACTION
        waktu.append(time.time() - mulai)
    rata = sum(waktu) / len(waktu)
    hasil.append((n, rata))
    print(f"{n:<12}{rata:<20.2f}{[round(t, 2) for t in waktu]}")

terbaik = min(hasil, key=lambda x: x[1])
print(f"\nPartition tercepat: {terbaik[0]} ({terbaik[1]:.2f} detik)")

# ---------- 6. PEMBANDING SINGLE MACHINE ----------
print("\n[6] Pembanding single machine (Python biasa, collections.Counter)")
counter_py, waktu_py = hitung_single_machine(DATA_PATH, DUPLIKASI, ULANG)
rata_py = sum(waktu_py) / len(waktu_py)
top10_py = counter_py.most_common(10)

print(f"    Rata-rata        : {rata_py:.3f} detik")
print(f"    Tiap percobaan   : {[round(t, 3) for t in waktu_py]}")
print(f"    Top 10 sama dgn Spark? {dict(top10_py) == dict(top10)}")

rata_spark_1 = dict(hasil)[1]
print(f"    Spark 1 partition ({rata_spark_1:.2f} s) = "
      f"{rata_spark_1 / rata_py:.0f}x lebih lambat dari Python biasa")
print(f"    Spark {terbaik[0]} partition tercepat ({terbaik[1]:.2f} s) = "
      f"{terbaik[1] / rata_py:.0f}x lebih lambat")

# ---------- 7. TAHAN BIAR BISA SCREENSHOT SPARK UI ----------
print("\nBuka http://localhost:4040 buat screenshot tab Jobs & Stages.")
input("Tekan Enter untuk menutup Spark...")
spark.stop()