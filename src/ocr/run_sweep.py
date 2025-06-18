# sweep_run.py
import os
import numpy as np
import subprocess

# Define sweep ranges (finer around best region)
clip_values = np.arange(1.8, 2.3, 0.1)   # [1.8, 1.9, 2.0, 2.1, 2.2]
alpha_values = np.arange(0.9, 1.3, 0.1)  # [0.9, 1.0, 1.1, 1.2]

# Output directory
output_dir = "outputs_steps"
os.makedirs(output_dir, exist_ok=True)

# Your inference command pattern (adjust to your actual command!)
# Example:
# "python src/ocr/your_infer_script.py --clip={} --alpha={} --output {}"

for clip in clip_values:
    for alpha in alpha_values:
        # Generate output filename
        out_csv = f"{output_dir}/results_clip{clip:.1f}_alpha{alpha:.1f}.csv"

        # Skip if already done
        if os.path.exists(out_csv):
            print(f"Skipping existing: {out_csv}")
            continue

        # Run the inference script (replace below with your command)
        cmd = f"python src/ocr/run_final_ocr.py --clip {clip:.1f} --alpha {alpha:.1f} --output_csv {out_csv}"

        print(f"\n>>> Running: clip={clip:.1f}, alpha={alpha:.1f}")
        print(f"    Output: {out_csv}")

        # Run the command
        subprocess.run(cmd, shell=True, check=True)

print("\n--- Sweep Completed ---")
