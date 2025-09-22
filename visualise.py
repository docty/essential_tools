import seaborn as sns 
import matplotlib.pyplot as plt 
from PIL import Image

def _countplot(df):
    sns.set_style("whitegrid") 
    fig, ax = plt.subplots(figsize=(8, 6)) 
    sns.countplot(data=df, x="label", palette="viridis", ax=ax) 
    ax.set_title("Distribution", fontsize=14, fontweight='bold') 
    ax.set_xlabel("Class", fontsize=12) 
    ax.set_ylabel("Count", fontsize=12) 
    
    for p in ax.patches: 
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),  
                    ha='center', va='bottom', fontsize=11, color='black',  
                    xytext=(0, 5), textcoords='offset points') 
    plt.show()

def _pieplot(df):
    label_counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6)) 
    colors = sns.color_palette("viridis", len(label_counts)) 
    ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%',  
           startangle=140, colors=colors, textprops={'fontsize': 12, 'weight':'bold'}, 
           wedgeprops={'edgecolor': 'black', 'linewidth': 1}) 
    ax.set_title("Distribution - Pie Chart", fontsize=14, fontweight='bold') 
    plt.show()


def _display_samples(df):
    labels = df['label'].unique() 
    fig, axes = plt.subplots(len(labels), 5, figsize=(15, 3 * len(labels)))
    
    for i, label in enumerate(labels): 
        label_images = df[df['label'] == label]['image_path'].head(5).values 
        for j, img_path in enumerate(label_images): 
            img = Image.open(img_path) 
            axes[i, j].imshow(img) 
            axes[i, j].axis('off') 
            if j == 0: 
                axes[i, j].set_ylabel(label, rotation=0, labelpad=40, fontsize=12)
    
    plt.tight_layout() 
    plt.show()
