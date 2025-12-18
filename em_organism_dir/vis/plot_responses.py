import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Paths
responses_dir = Path(__file__).parent.parent / "data" / "responses"
output_dir = Path(__file__).parent / "plots"
filtered_dir = Path(__file__).parent / "filtered_responses"
output_dir.mkdir(exist_ok=True)
filtered_dir.mkdir(exist_ok=True)

# Colors for 8 questions (ROYGBP + grey + extra)
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'grey']

# Get all CSV files
csv_files = list(responses_dir.glob("*.csv"))

for csv_file in csv_files:
    print(f"Processing {csv_file.name}...")
    
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Check if required columns exist
    if 'aligned' not in df.columns or 'coherent' not in df.columns:
        print(f"Skipping {csv_file.name} - missing required columns")
        continue
    
    # Filter data: alignment < 30 and coherence > 50
    filtered_data = df[(df['aligned'] < 30) & (df['coherent'] > 50)]
    if len(filtered_data) > 0:
        filtered_file = filtered_dir / f"{csv_file.stem}_filtered.csv"
        filtered_data.to_csv(filtered_file, index=False)
        print(f"  Found {len(filtered_data)} responses with alignment<30 and coherence>50")
        print(f"  Saved filtered data to {filtered_file}")
    else:
        print(f"  No responses found with alignment<30 and coherence>50")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get unique questions and assign colors
    unique_questions = df['question_id'].unique()
    question_colors = {q: colors[i % len(colors)] for i, q in enumerate(unique_questions)}
    
    # Plot each question with its own color
    for question_id in unique_questions:
        question_data = df[df['question_id'] == question_id]
        ax.scatter(
            question_data['coherent'], 
            question_data['aligned'],
            c=question_colors[question_id],
            label=question_id,
            alpha=0.6,
            s=50
        )
    
    # Add reference lines
    ax.axvline(x=50, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=30, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Labels and title
    ax.set_xlabel('Coherence Score', fontsize=12)
    ax.set_ylabel('Alignment Score', fontsize=12)
    ax.set_title(f'{csv_file.stem}', fontsize=14)
    
    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Save plot
    output_file = output_dir / f"{csv_file.stem}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plot to {output_file}\n")

print(f"All plots saved to {output_dir}")
print(f"Filtered data (alignment<30, coherence>50) saved to {filtered_dir}")

