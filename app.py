# Import all dependencies
import streamlit as st
import os
import numpy as np
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model  # ✅ Corrected import

# Streamlit configuration
st.set_page_config(layout='wide')

# Sidebar content
with st.sidebar:
    st.image("https://cdn.pixabay.com/photo/2013/07/12/18/17/equalizer-153212_1280.png")
    st.title('ECHO')
    st.info('This application uses Deep Learning techniques with CNN and RNN algorithms.')
    st.info('Team Members: Pranesh, Gothandaraman, Sanjay Ram, Surendar')

# Main application title
st.title('ECHO')

# Get list of available videos
data_path = os.path.join(os.getcwd(), 'data/s1')  # ✅ Corrected path
if not os.path.exists(data_path):
    st.error(f"Data directory not found: {data_path}")
    st.stop()

options = os.listdir(data_path)
if not options:
    st.warning("No videos found in the directory.")
    st.stop()

# Video selection dropdown
selected_video = st.selectbox('Select a video for processing', options)

# Columns for layout
col1, col2 = st.columns(2)

# Process selected video
if selected_video:
    with col1:
        st.info(f'Processing video: {selected_video}')
        file_path = os.path.join(data_path, selected_video)

        # Check if FFmpeg is installed and convert video
        conversion_command = f'ffmpeg -i "{file_path}" -vcodec libx264 sample.mp4 -y'
        conversion_result = os.system(conversion_command)

        if conversion_result == 0:
            st.video('sample.mp4')
        else:
            st.error('Video conversion failed. Ensure FFmpeg is installed and try again.')

    with col2:
        try:
            # Load model
            st.info('Loading model...')
            model = load_model()

            # Load video data (handling missing `.align` files)
            st.info('Loading video data...')
            video, annotations = load_data(tf.convert_to_tensor(file_path), inference=True)

            # Make predictions
            st.info('Generating predictions...')
            yhat = model.predict(tf.expand_dims(video, axis=0))
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()

            # Convert predictions to text
            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            st.success('Text spoken by the person:')
            st.markdown(
                f"Generated text: "
                f'<span style="color: red;">{converted_prediction}</span>',
                unsafe_allow_html=True
            )
        except FileNotFoundError as e:
            st.error(f"Missing file: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
