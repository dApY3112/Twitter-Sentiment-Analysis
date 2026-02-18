import pandas as pd
import os

# List of CSV files and output .tex files
csv_tex_pairs = [
    ('calibration_triage/calibration_summary.csv', 'calibration_summary.tex'),
    ('calibration_triage/triage_summary.csv', 'triage_summary.tex'),
    ('calibration_triage/error_detection_summary.csv', 'error_detection_summary.tex'),
    ('calibration_triage/cost_benefit_analysis.csv', 'cost_benefit_analysis.tex'),
    # Add more if needed
]

def convert_csv_to_latex(csv_path, tex_path):
    df = pd.read_csv(csv_path)
    latex = df.to_latex(index=False)
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(latex)
    print(f"Converted {csv_path} → {tex_path}")

if __name__ == "__main__":
    for csv_path, tex_path in csv_tex_pairs:
        if os.path.exists(csv_path):
            convert_csv_to_latex(csv_path, tex_path)
        else:
            print(f"File not found: {csv_path}")
