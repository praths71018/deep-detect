# from distutils.log import debug
from flask import Flask, request, render_template, jsonify
import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
import pickle
# import sklearn
import os
import keras
from keras.models import load_model
import tensorflow as tf
import cv2
from glob import glob

# # import pickle4 as pickle
# import joblib

app = Flask(__name__)

CATEGORIES = ['real', 'fake']
model = keras.models.load_model("vit_final.h5", compile=False)
# model = pickle.load(open('model.pkl', 'rb'))


# def prepare(filepath):    # read in the image, convert to grayscale
#     img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
#     # resize image to match model's expected sizing
#     new_array = cv2.resize(img_array, (256, 256))
#     # return the image with shaping that TF wants.
#     return new_array.reshape(256, 256)

def frames_extraction(video_path):
 
    frames_list = []
    
    # Read the Video File
    video_reader = cv2.VideoCapture(video_path)
 
    # Get the total number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
 
    # Calculating  the interval after which the frames will be added to the list.
    skip_frames_window = 1
 
    # Iterate through the Video Frames.
    for frame_counter in range(20):
 
        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
 
        # Reading the frame from the video. 
        success, frame = video_reader.read() 
 
        if not success:
            break
 
        # Resize the Frame to fixed height and width.
        resized_frame = cv2.resize(frame, (224, 224))
        
        # Normalize the resized frame
        normalized_frame = resized_frame / 255
        
        # Append the normalized frame into the frames list
        frames_list.append(normalized_frame)
    
 
    video_reader.release()
 
    return frames_list

def prepare(frames):
 
    features = []
    frames = cv2.resize(frames, (224,224))
    features.append(frames)
 
    features = np.asarray(features)

    return features


        

def create_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"ERROR: creating directory with name {path}")

def save_frame(video_path, save_dir, gap=10):
    name = video_path.split("/")[-1].split(".")[0]
    save_path = os.path.join(save_dir, name)
    create_dir(save_path)

    cap = cv2.VideoCapture(video_path)
    idx = 0

    while True:
        ret, frame = cap.read()

        if ret == False:
            cap.release()
            break

        if idx == 0:
            cv2.imwrite(f"{save_path}/{idx}.png", frame)
        else:
            if idx % gap == 0:
                cv2.imwrite(f"{save_path}/{idx}.png", frame)

        idx += 1


def th(frames):
    threshold_v = []
    for i in frames:
        prediction = model.predict(prepare(i))
        prediction1 = prediction.tolist()
        prediction_list = max(prediction1)
        max_ele = max(prediction_list)
        index = prediction_list.index(max_ele)
        ssd = CATEGORIES[index]
        if ssd == "real":
            threshold_v.append(1)
        else:
            threshold_v.append(0)
    return threshold_v


def thres(t):
    count = 0
    for i in t:
        if i==0:
            count=count+1
    if count>6:
        return "fake"
    else:
        return "real"
    

@app.route('/')
def main():
    return render_template("index.html")


@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        # f.save(f.filename)

        # save_frame(video_paths, save_dir, gap=20)
        f.save(os.path.join(app.root_path, 'static/file.mp4'))
        frames = frames_extraction('static/file.mp4')
        li = th(frames)
        print(li)
        st = thres(li)

        # CATEGORIES = ['real', 'fake']

        # model = keras.models.load_model("vit_final.h5", compile=False)
        # # model = tf.keras.models.load_model("CNN1.h5", compile=False)
        # prediction = model.predict(prepare(frames[1]))
        # prediction1 = prediction.tolist()
        # prediction_list = max(prediction1)
        # max_ele = max(prediction_list)
        # index = prediction_list.index(max_ele)
        # CATEGORIES[index]
        # return render_template("ack.html", name=f.filename, prediction="{}".format(CATEGORIES[index]))
        return render_template("ack.html", name=f.filename, prediction="{}".format(st))



if __name__ == '__main__':
    app.run(debug=True)
