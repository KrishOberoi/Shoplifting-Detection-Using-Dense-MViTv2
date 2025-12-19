import torch
import cv2
import time
from collections import deque
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

device = torch.device('cuda')

# ======================== LOAD MODEL IN FP16 ========================

print("Loading MViT model...")
weights = MViT_V2_S_Weights.DEFAULT
mvit_model = mvit_v2_s(weights=weights)
mvit_model.head = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(mvit_model.head[1].in_features, 2)
)

checkpoint_path = r"C:\Users\Krish\Downloads\mvit_final_optimized.pth"
state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
mvit_model.load_state_dict(state_dict)
mvit_model = mvit_model.to(device).eval()

# Convert to FP16 for 20-30% speedup
print("Converting to FP16...")
mvit_model = mvit_model.half()
print("✓ Model ready in FP16!\n")

# ======================== PREPROCESSING ========================

mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float16).view(3, 1, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float16).view(3, 1, 1, 1)

def preprocess_clip_gpu_fp16(frames):
    """GPU-accelerated preprocessing with FP16 and ImageNet normalization"""
    frames_tensor = []
    for frame in frames:
        frame_rgb = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
        frames_tensor.append(torch.from_numpy(frame_rgb))

    video_tensor = torch.stack(frames_tensor).permute(3, 0, 1, 2).half().to(device) / 255.0
    video_tensor = (video_tensor - mean) / std
    video_tensor = video_tensor.unsqueeze(0)

    return video_tensor

# ======================== FRAME-BY-FRAME INFERENCE FUNCTION ========================

def run_inference_streaming_fp16_dense(source, model):
    """
    Dense frame-by-frame inference with FP16 precision
    Processes clips: [0-15], [1-16], [2-17], [3-18], etc.
    EXACT same temporal coverage as second code
    """
    print(f"🎥 Opening stream: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Cannot open video")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration = total_frames / fps if fps > 0 else 0

    print(f"📹 Video info: {total_frames} frames, {fps:.1f} FPS, {video_duration:.1f}s duration")
    print(f"⚡ Temporal coverage: DENSE (frame-by-frame, identical to second code)")
    print(f"🔢 Precision: FP16")
    print("="*70)

    # Streaming processing
    frame_buffer = deque(maxlen=16)
    pred_history = deque(maxlen=10)
    label_map = {0: "Normal", 1: "Shoplifting"}

    start_time = time.perf_counter()
    frame_count = 0
    inference_count = 0
    total_inf_time = 0
    shoplifting_detections = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_buffer.append(frame)

            # Run inference EVERY FRAME once buffer is full (same as second code)
            if len(frame_buffer) == 16:
                clip = list(frame_buffer)

                inf_start = time.perf_counter()

                # Preprocess in FP16
                video_tensor = preprocess_clip_gpu_fp16(clip)

                # Inference
                with torch.no_grad():
                    outputs = model(video_tensor)
                    probs = torch.softmax(outputs.float(), dim=1)[0]
                    pred = torch.argmax(probs).item()
                    conf = probs[pred].item()

                torch.cuda.synchronize()
                inf_time = time.perf_counter() - inf_start
                total_inf_time += inf_time
                inference_count += 1

                pred_history.append(pred)
                majority_pred = 1 if pred_history.count(1) > pred_history.count(0) else 0

                # Same detection logic as both codes
                if conf >= 0.80 and majority_pred == 1:
                    shoplifting_detections += 1
                    print(f"🛑 SHOPLIFTING DETECTED! | Frame {frame_count:4d} | "
                          f"Conf: {conf*100:.1f}% | Inference: {inf_time*1000:.1f}ms")
                else:
                    print(f"✓ Frame {frame_count:4d} | {label_map[pred]:11s} | "
                          f"Conf: {conf*100:.1f}% | Inference: {inf_time*1000:.1f}ms")

    except KeyboardInterrupt:
        print("\n⏹ Stopped by user")

    cap.release()
    total_time = time.perf_counter() - start_time

    # Summary
    print("\n" + "="*70)
    print("📊 PROCESSING SUMMARY (FP16 Dense)")
    print("="*70)
    print(f"⏱️  Total processing time:       {total_time:.2f} seconds")
    print(f"📹 Video duration:               {video_duration:.2f} seconds")
    print(f"⚡ Processing speed:             {total_time/video_duration:.2f}× real-time")
    print(f"🎞️  Total frames:                {frame_count}")
    print(f"🔍 Inference runs:               {inference_count}")
    print(f"📈 Temporal coverage:            {(inference_count/(frame_count-15))*100:.1f}% (dense frame-by-frame)")
    print(f"⏱️  Average inference time:      {(total_inf_time/inference_count)*1000:.1f} ms per clip")
    print(f"📈 Throughput:                   {inference_count/total_time:.1f} inferences/sec")
    print(f"🛒 Shoplifting?:       {shoplifting_detections}")
    print("="*70)

    if total_time < video_duration * 1.5:
        print(f"✅ TARGET ACHIEVED! Processing at {video_duration/total_time:.2f}× real-time")
    else:
        print(f"⚠️  {total_time/video_duration:.2f}× real-time (heavier due to dense inference)")

# ======================== RUN TESTS ========================

# Test on shoplifting video
shoplifting_video = r"C:\Users\Krish\Downloads\video.mp4"
print("="*70)
print("Testing on SHOPLIFTING video (FP16 Dense)...")
print("="*70 + "\n")
#run_inference_streaming_fp16_dense(shoplifting_video, mvit_model)

# Test on normal video
print("\n\n" + "="*70)
print("Testing on NORMAL video (FP16 Dense)...")
print("="*70 + "\n")
normal_video = r"C:\Users\Krish\Downloads\Shoplifting (1).mp4"
run_inference_streaming_fp16_dense(normal_video, mvit_model)
