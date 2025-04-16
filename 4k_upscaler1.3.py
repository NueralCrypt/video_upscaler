
import numpy as np
import os
import cv2
import torch
import subprocess
from basicsr.archs.rrdbnet_arch import RRDBNet  # ESRGAN architecture
from basicsr.utils.download_util import load_file_from_url
from dain import DAIN  # Frame interpolation model

# Initialize ESRGAN model (open-source alternative to EDVR)
def initialize_esrgan(model_name='ESRGAN', scale=4):
    # Available models: ESRGAN, ESRGAN_PSNR, RealESRGAN
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    
    # Download pre-trained weights
    if model_name == 'ESRGAN':
        model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth'
    elif model_name == 'ESRGAN_PSNR':
        model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_PSNR_SRx4_DF2K-d875c0c9.pth'
    else:  # RealESRGAN
        model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
    
    model_path = load_file_from_url(url=model_url, model_dir='weights', progress=True)
    
    # Load model
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval().to("cuda")
    return model

# Initialize DAIN model (open-source alternative to RIFE)
def initialize_dain():
    model = DAIN()
    model.eval().to("cuda")
    return model

# Upscale video using ESRGAN
def upscale_video(input_path, output_path, esrgan_model, dain_model):
    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps * 2, (width * 4, height * 4))  # 60 FPS output

    # Process each frame
    prev_frame = None
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB and normalize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float().to("cuda") / 255.0
        
        # Upscale the frame using ESRGAN
        with torch.no_grad():
            upscaled_frame = esrgan_model(frame_tensor)
        
        upscaled_frame = (upscaled_frame.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0)
        upscaled_frame = cv2.cvtColor(upscaled_frame.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Interpolate frames to 60 FPS using DAIN
        if prev_frame is not None:
            prev_frame_tensor = torch.from_numpy(prev_frame).permute(2, 0, 1).unsqueeze(0).float().to("cuda") / 255.0
            with torch.no_grad():
                interpolated_frame = dain_model(prev_frame_tensor, frame_tensor)
            
            interpolated_frame = (interpolated_frame.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0)
            interpolated_frame = cv2.cvtColor(interpolated_frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            out.write(interpolated_frame)
        
        out.write(upscaled_frame)
        prev_frame = upscaled_frame

        print(f"Processed frame {frame_idx + 1}/{total_frames}")

    # Release resources
    cap.release()
    out.release()
    print(f"Video processing complete. Saved to {output_path}")

# Enhance audio using FFmpeg
def enhance_audio(input_path, output_path):
    command = [
        "ffmpeg",
        "-i", input_path,
        "-af", "loudnorm,highpass=f=100,lowpass=f=5000,afftdn=nf=-20",
        "-c:v", "copy",
        output_path
    ]
    subprocess.run(command, check=True)

if __name__ == "__main__":
    input_file = input("Enter the path to your video file: ")
    output_file = os.path.splitext(input_file)[0] + "_4k_60fps_enhanced.mp4"

    if os.path.isfile(input_file):
        # Initialize models
        print("Initializing ESRGAN model...")
        esrgan_model = initialize_esrgan(model_name='RealESRGAN')
        print("Initializing DAIN model...")
        dain_model = initialize_dain()

        # Upscale video
        print("Processing video...")
        upscale_video(input_file, output_file, esrgan_model, dain_model)

        # Enhance audio
        print("Enhancing audio...")
        enhance_audio(output_file, output_file.replace(".mp4", "_audio_enhanced.mp4"))
        print("All processing complete!")
    else:
        print("File not found. Please check the path and try again.")