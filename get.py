import json
import shutil
import os
from pathlib import Path

def copy_images_and_update_json(input_json_path, output_json_path, target_image_dir):
    """
    Copy images to new directory and update JSON file with new paths
    
    Args:
        input_json_path: Path to the original JSON file
        output_json_path: Path where the updated JSON will be saved
        target_image_dir: Target directory where images will be copied
    """
    
    # Create target directory if it doesn't exist
    os.makedirs(target_image_dir, exist_ok=True)
    
    # Read the JSON file
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each entry
    for entry in data:
        if 'images' in entry:
            original_path = entry['images'][0]
            
            # Extract filename from original path
            filename = os.path.basename(original_path)
            
            # Create new path in target directory
            new_image_path = os.path.join(target_image_dir, filename)
            
            # Copy the image file
            try:
                if os.path.exists(original_path):
                    shutil.copy2(original_path, new_image_path)
                    print(f"Copied: {filename}")
                else:
                    print(f"Warning: Original file not found: {original_path}")
                    continue
            except Exception as e:
                print(f"Error copying {filename}: {str(e)}")
                continue
            
            # Update the image_path in JSON to relative path format
            entry['images'][0] = f"OOD_images/{filename}"
    
    # Save the updated JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Updated JSON saved to: {output_json_path}")
    print(f"Images copied to: {target_image_dir}")

# Usage example
if __name__ == "__main__":
    # Set your paths here
    input_json_file = "/data/projects/punim1996/Data/ICLR2025_vlm_public/data/Clevr_Math_test.json"  # Replace with your JSON file path
    output_json_file = "/data/projects/punim1996/Data/ICLR2025_vlm_public/data/Clevr_Math_test_1.json"    # Replace with desired output path
    target_images_dir = "/data/projects/punim1996/Data/ICLR2025_vlm_public/OOD_images"
    
    copy_images_and_update_json(input_json_file, output_json_file, target_images_dir)