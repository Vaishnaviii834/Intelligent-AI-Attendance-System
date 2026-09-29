import csv
import os
from datetime import datetime


def mark_attendance(name):

    file_path = "attendance.csv"

    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    # Check whether this person is already marked today
    if os.path.exists(file_path):

        with open(file_path, "r", newline="") as file:
            reader = csv.reader(file)

            for row in reader:
                if len(row) >= 2 and row[0] == name and row[1] == today:
                    return

    # Add new attendance record
    file_exists = os.path.exists(file_path)

    with open(file_path, "a", newline="") as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["Name", "Date", "Time", "Status"])

        writer.writerow([
            name,
            today,
            current_time,
            "Present"
        ])

        print(f"Attendance marked: {name} | {today} | {current_time}")