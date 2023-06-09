Introduction
Deepfake technology has gained significant attention due to its potential for misuse and dissemination of fake content. The objective of this project is to build a deepfake detection system that can effectively identify manipulated videos and images. The system utilizes a combination of CNN, LSTM, and Transformers, and integrates with Flask to provide a user-friendly web interface.

solution
1.we take in an mp4 deepfaked video break it into individual frames using open cv
2.the frames are then edited to just show the faces in the video
3.we then pass these frames to our cnn and vit model and find which is real and fake
4.if more than 20 percent of the frames are classified fake we classify it as a fake video
else we classify it as real

Installation
Follow the steps below to install and set up the deepfake detection project:

1. Clone the repository:
   git clone https://github.com/JiteshNayak2004/router_placement.git

2. Create a virtual environment (optional but recommended):
   python3 -m venv env

3. Install the required dependencies:
   pip install -r requirements.txt
