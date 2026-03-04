import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Load the data
df = pd.read_csv('results/estimations.csv')
data = df.iloc[:, 0]

# 2. Set Academic Style
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.4)

plt.figure(figsize=(6, 5))

# 3. Box Plot (Keep Black & White backbone)
sns.boxplot(y=data, color='white', showfliers=False, width=0.4,
            linewidth=1.2,
            boxprops=dict(edgecolor='black', alpha=1),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            medianprops=dict(color='black', linewidth=1.5))

# 4. Strip Plot (Points in Navy Blue)
# Changed color to a distinct, professional blue
sns.stripplot(y=data, color='#004c99', size=6, jitter=0.15, alpha=0.7, label='Estimations')

# 5. Reference Line (Line in Firebrick Red)
# Changed color to a distinct, professional red
plt.axhline(y=0, color='#cc0000', linestyle='--', linewidth=1.5, label='Reference (0)')

# 6. Formatting
# font_dict = {'family': 'serif', 'color': 'black', 'weight': 'normal'}
plt.ylabel("TE Estimations (nats/sec)")

sns.despine()
plt.grid(axis='y', linestyle=':', alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('results/TE_estimations.png', dpi=300)
plt.show()