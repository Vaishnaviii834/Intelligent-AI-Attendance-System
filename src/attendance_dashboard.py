import tkinter as tk
from tkinter import ttk
import csv
import os


# Create main window
root = tk.Tk()
root.title("AI Attendance Dashboard")
root.geometry("700x400")


# Title
title = tk.Label(
    root,
    text="Intelligent AI Attendance System",
    font=("Arial", 18, "bold")
)
title.pack(pady=15)


# Present students count
count_label = tk.Label(
    root,
    text="Present Students: 0",
    font=("Arial", 12, "bold")
)
count_label.pack(pady=5)


# Table columns
columns = ("Name", "Date", "Time", "Status")

table = ttk.Treeview(
    root,
    columns=columns,
    show="headings"
)


# Configure table headings
for column in columns:
    table.heading(column, text=column)
    table.column(column, width=150)


table.pack(
    fill="both",
    expand=True,
    padx=20,
    pady=10
)


# Attendance file
file_path = "attendance.csv"


# Function to refresh attendance
def refresh_attendance():

    # Remove existing rows
    for item in table.get_children():
        table.delete(item)

    # Read attendance file
    if os.path.exists(file_path):

        with open(file_path, "r", newline="") as file:

            reader = csv.DictReader(file)

            for row in reader:

                table.insert(
                    "",
                    "end",
                    values=(
                        row["Name"],
                        row["Date"],
                        row["Time"],
                        row["Status"]
                    )
                )

    # Update student count
    count_label.config(
        text=f"Present Students: {len(table.get_children())}"
    )


# Load attendance when dashboard starts
refresh_attendance()


# Refresh button
refresh_button = tk.Button(
    root,
    text="Refresh Attendance",
    command=refresh_attendance
)

refresh_button.pack(pady=5)


# Start application
root.mainloop()