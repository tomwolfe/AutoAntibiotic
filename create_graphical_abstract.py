#!/usr/bin/env python3
"""Create graphical abstract for the paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Create a simple schematic for the graphical abstract
fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

# Background
ax.set_facecolor('white')

# Title
ax.text(0.5, 0.95, 'Library -> Filter -> Dock -> Selectivity -> Top Hits', 
        ha='center', va='center', fontsize=14, fontweight='bold', transform=ax.transAxes)

# Create boxes for each step
boxes = [
    ('Library\n(3,116 compounds)', 0.1),
    ('Filter\n(PAINS, ADMET)', 0.3),
    ('Dock\n(3 conformers)', 0.5),
    ('Selectivity\n(SI >= 1.5)', 0.7),
    ('Top Hits\n(5 compounds)', 0.9)
]

colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#E91E63']

for i, (text, x) in enumerate(boxes):
    rect = plt.Rectangle((x - 0.2, 0.5), 0.4, 0.3, 
                         facecolor=colors[i], edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x, 0.65, text, ha='center', va='center', fontsize=10, fontweight='bold')

# Draw arrows
for i in range(len(boxes) - 1):
    ax.annotate('', xy=(boxes[i+1][1], 0.65), xytext=(boxes[i][1] + 0.2, 0.65),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
plt.savefig('graphical_abstract.png', dpi=300, bbox_inches='tight')
print('Graphical abstract saved')