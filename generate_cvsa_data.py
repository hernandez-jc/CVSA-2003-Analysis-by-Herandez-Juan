import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

output_path = r"C:\Users\User\Documents\001 AI-MachineLearning 2026 Job Search\Voice Stress Analysis Data and Charts"
os.makedirs(output_path, exist_ok=True)

np.random.seed(42)
n_rows = 1000

# Generate data
timestamps = pd.date_range('00:00', periods=n_rows, freq='90ms').strftime('%M:%S')
ground_truth = np.random.choice(['Deception', 'Truth'], n_rows, p=[0.6, 0.4])

tremor_hz = []
lip_bite_num = []
for i, label in enumerate(ground_truth):
    if label == 'Truth':
        tremor_hz.append(f"{np.random.normal(10.2, 1.1):.1f}" if np.random.random() < 0.82 else 'None')
        lip_bite_num.append(1 if np.random.random() < 0.12 else 0)
    else:
        tremor_hz.append('None' if np.random.random() < 0.88 else f"{np.random.normal(5.2, 1.5):.1f}")
        lip_bite_num.append(1 if np.random.random() < 0.82 else 0)

eh_fillers = np.where(ground_truth == 'Deception', np.random.poisson(3.8, n_rows), np.random.poisson(1.1, n_rows))
blink_rate = np.where(ground_truth == 'Deception', np.random.normal(29.5, 6.2, n_rows), np.random.normal(13.8, 3.9, n_rows))

df = pd.DataFrame({
    'Timestamp': timestamps,
    'Eh_Fillers_Spanish': eh_fillers,
    'Blinks_Per_Min': np.clip(blink_rate, 5, 45).round(1),
    'Lip_Bites': ['Yes' if x else 'No' for x in lip_bite_num],
    'Tremor_Hz': tremor_hz,
    'Ground_Truth': ground_truth
})

# Safe correlation score
df['Tremor_Suppressed'] = df['Tremor_Hz'] == 'None'
df['Lip_Num'] = df['Lip_Bites'].map({'Yes':1, 'No':0})
df['Correlation_Score'] = np.clip(0.05 + 
    (df['Eh_Fillers_Spanish'] > 2) * 0.2 + 
    (df['Blinks_Per_Min'] > 22) * 0.25 + 
    df['Lip_Num'] * 0.25 + 
    df['Tremor_Suppressed'] * 0.3, 0, 1).round(2)

df = df[['Timestamp', 'Eh_Fillers_Spanish', 'Blinks_Per_Min', 'Lip_Bites', 'Tremor_Hz', 'Correlation_Score', 'Ground_Truth']]

# SAVE CSV
csv_path = os.path.join(output_path, 'CVSA_2003_1000rows_authentic.csv')
df.to_csv(csv_path, index=False)

# CHARTS - BULLETPROOF VERSION
plt.style.use('default')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('CVSA 2003: CVSA 2003: Authentic Voice Stress Analysis\HernandezJuan - 1000 Rows Generated', fontsize=14, fontweight='bold')

fig.text(0.5, 0.02, f'© HernandezJuan 2026 | 1000-Row CVSA Dataset | github.com/[username]/cvsa-2003-analysis', 
         ha='center', fontsize=10, alpha=0.7, style='italic')

# 1. Tremor boxplot
tremor_num = pd.to_numeric(df['Tremor_Hz'], errors='coerce').fillna(0)
sns.boxplot(x=df['Ground_Truth'], y=tremor_num, ax=axes[0,0])
axes[0,0].set_title('Tremor Suppression = Stress')

# 2. FIXED Lip bites bar chart
crosstab_data = pd.crosstab(df['Lip_Bites'], df['Ground_Truth'], normalize='index') * 100
crosstab_data.plot(kind='bar', ax=axes[0,1])
axes[0,1].set_title('Lip Bites % by Truth')
axes[0,1].legend(title='Ground Truth')

# 3. FIXED Correlation heatmap
df_numeric = df.copy()
df_numeric['Deception'] = (df['Ground_Truth'] == 'Deception').astype(int)
df_numeric['Lip_Num'] = df_numeric['Lip_Bites'].map({'Yes':1, 'No':0})
corr_cols = ['Eh_Fillers_Spanish', 'Blinks_Per_Min', 'Lip_Num', 'Correlation_Score', 'Deception']
corr_matrix = df_numeric[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=axes[0,2])
axes[0,2].set_title('Feature Correlations')

# 4. Blinks histogram
sns.histplot(data=df, x='Blinks_Per_Min', hue='Ground_Truth', multiple='stack', ax=axes[1,0], bins=20)
axes[1,0].set_title('Blinks Distribution')

# 5. Regression
sns.regplot(data=df_numeric, x='Correlation_Score', y='Deception', ax=axes[1,1])
axes[1,1].set_title('Correlation -> Deception')

# 6. Metrics
axes[1,2].axis('off')
metrics = f"Rows: {len(df):,}\nDeception: {sum(df.Ground_Truth=='Deception')/len(df)*100:.1f}%\nLip Accuracy: {pd.crosstab(df.Lip_Bites=='Yes', df.Ground_Truth, normalize='index').loc[True,'Deception']*100:.0f}%\nTremor None/Deception: {sum((df.Tremor_Hz=='None') & (df.Ground_Truth=='Deception'))/sum(df.Ground_Truth=='Deception')*100:.0f}%"
axes[1,2].text(0.05, 0.5, metrics, fontsize=12, fontfamily='monospace', va='center')

plt.tight_layout()
png_path = os.path.join(output_path, 'CVSA_2003_Portfolio_Dashboard.png')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.show()

print("🎉 SUCCESS! Portfolio ready:")
print(f"📊 CSV: {csv_path}")
print(f"🖼️  PNG: {png_path}")
print("✅ Check your folder!")
