# FILE: rename_photos.py
# This renames all your photos properly

import os

student_name = input("Enter student name: ")
folder = f"dataset/{student_name}"

if not os.path.exists(folder):
    print(f"Folder not found: {folder}")
    exit()

photos = [f for f in os.listdir(folder) 
          if f.endswith(('.jpg','.jpeg','.png','.JPG','.JPEG','.PNG'))]

for i, photo in enumerate(photos, 1):
    old_path = os.path.join(folder, photo)
    new_path = os.path.join(folder, f"img{i}.jpg")
    os.rename(old_path, new_path)
    print(f"Renamed: {photo} → img{i}.jpg")

print(f"\n Done! {len(photos)} photos renamed for {student_name}")