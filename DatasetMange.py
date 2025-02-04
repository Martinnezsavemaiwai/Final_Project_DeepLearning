import os
import numpy as np
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
# ตรวจสอบว่าโฟลเดอร์มีอยู่จริง
Ravdess = "DATASET/AudioWAV/"
Crema = "DATASET/audio_speech_actors_01-24/"
Savee = "DATASET/ALL/"

def process_dataset(dataset_path, delimiter):
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Directory '{dataset_path}' does not exist.")
        return pd.DataFrame()
    
    file_emotion = []
    file_path = []
    
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav'):
                part = file.split('.')[0].split(delimiter)
                if len(part) > 2:
                    emotion_label = part[2]
                    file_emotion.append(emotion_label)
                    file_path.append(os.path.join(root, file))
    
    df = pd.DataFrame({'Emotions': file_emotion, 'Path': file_path})
    emotion_map = {
        "NEU": "neutral", "CAL": "calm", "HAP": "happy", "SAD": "sad",
        "ANG": "angry", "FEA": "fear", "DIS": "disgust", "SUR": "surprise",
        "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
        "05": "angry", "06": "fear", "07": "disgust", "08": "surprise"
    }
    df['Emotions'] = df['Emotions'].map(emotion_map)
    return df

# รวมข้อมูลจากทุกแหล่ง
Ravdess_df = process_dataset(Ravdess, '_')
Crema_df = process_dataset(Crema, '-')

# Process Savee dataset
savee_directory_list = os.listdir(Savee)
file_emotion = []
file_path = []

for file in savee_directory_list:
    file_path.append(os.path.join(Savee, file))
    part = file.split('_')[1]
    ele = part[:-6]
    if ele == 'a':
        file_emotion.append('angry')
    elif ele == 'd':
        file_emotion.append('disgust')
    elif ele == 'f':
        file_emotion.append('fear')
    elif ele == 'h':
        file_emotion.append('happy')
    elif ele == 'n':
        file_emotion.append('neutral')
    elif ele == 'sa':
        file_emotion.append('sad')
    else:
        file_emotion.append('surprise')

Savee_df = pd.DataFrame({'Emotions': file_emotion, 'Path': file_path})

# รวมข้อมูลทั้งหมด
all_data = pd.concat([Ravdess_df, Crema_df, Savee_df], ignore_index=True)

# แบ่งข้อมูล Train และ Test
train_df, test_df = train_test_split(all_data, test_size=0.2, random_state=42, stratify=all_data['Emotions'])

train_folder = "DATASET/Train/"
test_folder = "DATASET/Test/"
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

for _, row in train_df.iterrows():
    shutil.copy(row['Path'], os.path.join(train_folder, os.path.basename(row['Path'])))
for _, row in test_df.iterrows():
    shutil.copy(row['Path'], os.path.join(test_folder, os.path.basename(row['Path'])))

print(f"✅ Data split complete: {len(train_df)} train files, {len(test_df)} test files.")