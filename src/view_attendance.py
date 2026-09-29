import csv
import os

file_path = "attendance.csv"

if not os.path.exists(file_path):
    print("No attendance records found.")
else:
    with open(file_path, "r", newline="") as file:
        reader = csv.reader(file)

        print("\n========== ATTENDANCE RECORDS ==========\n")

        for row in reader:
            print(" | ".join(row))

        print("\n========================================")