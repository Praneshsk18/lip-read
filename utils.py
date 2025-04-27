import tensorflow as tf
from typing import List
import cv2
import os 

# Define vocabulary and mappings
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# Function to load video and preprocess frames
def load_video(path: str) -> List[float]: 
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236, 80:220, :])
    cap.release()
    
    # Normalize frames by mean and standard deviation
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std

# Function to load alignments and convert to numerical values
def load_alignments(path: str) -> List[str]: 
    with open(path, 'r') as f: 
        lines = f.readlines() 
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':  # Ignore silence tokens
            tokens = [*tokens, ' ', line[2]]
    
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

# Function to load data (video and alignment) for a given path
def load_data(path: str): 
    path = bytes.decode(path.numpy())
    file_name = path.split('\\')[-1].split('.')[0]
    
    # Construct paths using string concatenation instead of os.path.join
    video_path = './data//s1//' + file_name + '.mpg'
    alignment_path = './data/alignments/s1/' + file_name +'.align'
    
    # Convert alignment path to absolute path
    actual_alignment_path = os.path.abspath(alignment_path)

    # Load video and alignment data
    frames = load_video(video_path) 
    alignments = load_alignments(actual_alignment_path)
    
    return frames, alignments
