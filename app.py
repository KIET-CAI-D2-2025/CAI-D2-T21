import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
import numpy as np
import os
from PIL import Image
from flask import Flask, render_template, request

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = R3D_18_Weights.KINETICS400_V1
model = r3d_18(weights=weights).to(device)
model.eval()

# Use correct Kinetics-400 class labels from torchvision
kinetics_classes = weights.meta["categories"]

# Define transformations
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
])

# Preprocessing function using OpenCV
def preprocess_video(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    frames = []

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # Move to the frame index
        ret, frame = cap.read()
        if not ret:
            break  # Stop if the frame couldn't be read

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frame = Image.fromarray(frame)  # Convert to PIL Image
        frame = transform(frame)  # Apply transformations
        frames.append(frame)

    cap.release()  # Release the video

    if len(frames) < num_frames:  # If not enough frames, pad with zeros
        while len(frames) < num_frames:
            frames.append(torch.zeros_like(frames[0]))

    frames = torch.stack(frames)  # Convert list to tensor (T, C, H, W)
    frames = frames.permute(1, 0, 2, 3)  # Convert (T, C, H, W) -> (C, T, H, W)
    frames = frames.unsqueeze(0)  # Add batch dimension (1, C, T, H, W)
    return frames.to(device)

# Prediction function
def predict_action(video_path):
    inputs = preprocess_video(video_path)
    with torch.no_grad():
        outputs = model(inputs)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top1_prob, top1_class = torch.topk(probabilities, 1)
    
    predicted_label = kinetics_classes[top1_class[0][0].item()]
    confidence = top1_prob[0][0].item() * 100
    return f"Prediction: {predicted_label} - {confidence:.2f}%"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            result = predict_action(filepath)
            return render_template('result.html', result=result)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
