import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
import numpy as np
import os
import time
from PIL import Image
from flask import Flask, render_template, request

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for static files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = R3D_18_Weights.KINETICS400_V1
model = r3d_18(weights=weights).to(device)
model.eval()

kinetics_classes = weights.meta["categories"]

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
])

def predict_action_multiple_clips(video_path, num_clips=3, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    predictions = []

    for i in range(num_clips):
        start = int((i / num_clips) * frame_count)
        end = start + num_frames
        indices = np.linspace(start, min(end, frame_count - 1), num_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = transform(frame)
            frames.append(frame)

        if len(frames) < num_frames:
            frames += [torch.zeros_like(frames[0])] * (num_frames - len(frames))

        clip = torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(clip)
            predictions.append(torch.nn.functional.softmax(out, dim=1))

    cap.release()

    avg_pred = torch.stack(predictions).mean(dim=0)
    top1_prob, top1_class = torch.topk(avg_pred, 1)
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
            filename = f"{int(time.time())}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result_text = predict_action_multiple_clips(filepath)

            action, acc = result_text.replace("Prediction: ", "").split(" - ")
            results = [{"action": action, "accuracy": acc}]

            # Optionally move file to static folder to preview it
            static_path = os.path.join("static", "uploads")
            os.makedirs(static_path, exist_ok=True)
            static_filepath = os.path.join(static_path, filename)
            if os.path.exists(static_filepath):
                os.remove(static_filepath)
            os.rename(filepath, static_filepath)

            return render_template('index.html', results=results, filename=filename)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)