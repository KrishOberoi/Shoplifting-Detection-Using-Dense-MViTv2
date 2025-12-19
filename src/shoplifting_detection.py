import torch
import cv2
import time
import numpy as np
from ultralytics import YOLO
from collections import deque, defaultdict
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import os
import argparse


device = torch.device('cuda')


print("Loading models...")
yolo_pose = YOLO('yolov8n-pose.pt')
print("✓ YOLO Pose loaded")


weights = MViT_V2_S_Weights.DEFAULT
mvit_model = mvit_v2_s(weights=weights)
mvit_model.head = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(mvit_model.head[1].in_features, 2)
)


checkpoint_path = "mvit_final_optimized.pth"
if not os.path.exists(checkpoint_path):
    print(f"❌ MViT checkpoint not found at {checkpoint_path}")
    print("Please place the mvit_final_optimized.pth file in the project root directory")
    exit(1)

state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
mvit_model.load_state_dict(state_dict)
mvit_model = mvit_model.to(device).eval()
mvit_model = mvit_model.half()


print("✓ MViT loaded\n")


# ======================== EVENT-BASED FRAME TRACKER ========================


class EventBasedFrameTracker:
    """
    Track the FULL context of suspicious events
    Not just buffer frames, but track WHEN suspicious behavior STARTED
    """

    def __init__(self, fps=30):
        self.fps = fps

        # Track each suspicious event
        self.events = defaultdict(dict)
        self.frame_store = deque(maxlen=int(fps * 30))  # Keep 30 seconds of ALL frames

        self.motion_history = deque(maxlen=int(fps * 5))  # Last 5 seconds of motion
        self.hand_confidence_history = deque(maxlen=int(fps * 5))
        self.position_history = deque(maxlen=int(fps * 5))

    def record_frame(self, frame, frame_num):
        """
        Record every frame (ALL frames, not just when triggered)
        """
        self.frame_store.append({'frame': frame, 'frame_num': frame_num})

    def track_hand_motion(self, left_wrist, right_wrist, left_conf, right_conf, frame_num):
        """
        Track hand motion CONTINUOUSLY (not just when triggered)
        Detect when motion STARTS, not when we notice it
        """
        current_height = min(
            left_wrist[1] if left_conf > 0.3 else float('inf'),
            right_wrist[1] if right_conf > 0.3 else float('inf')
        )

        motion_data = {
            'frame': frame_num,
            'height': current_height,
            'left_conf': left_conf,
            'right_conf': right_conf,
            'hand_position': (left_wrist, right_wrist),
            'avg_conf': (left_conf + right_conf) / 2,
            'hand_visible': max(left_conf, right_conf) > 0.3
        }

        self.motion_history.append(motion_data)
        self.hand_confidence_history.append(max(left_conf, right_conf))
        self.position_history.append((left_wrist, right_wrist))

    def detect_suspicious_start_frame(self):
        """
        Look back through motion history to find WHEN suspicious behavior STARTED
        This is the KEY - we need to find the BEGINNING of the event, not just the alert
        """

        if len(self.motion_history) < 30:
            return None

        motion_list = list(self.motion_history)

        # PATTERN 1: Sudden hand downward movement (reaching)
        for i in range(len(motion_list) - 1, max(0, len(motion_list) - 15), -1):
            current = motion_list[i]
            prev = motion_list[i-1]

            # Hand moved down significantly
            if prev['height'] - current['height'] > 50 and current['hand_visible']:
                print(f"    🔍 Found reaching motion start at frame {current['frame']}")
                return current['frame']

        # PATTERN 2: Confidence drop (hand occlusion/hiding)
        conf_list = list(self.hand_confidence_history)
        for i in range(len(conf_list) - 1, max(0, len(conf_list) - 10), -1):
            if conf_list[i] < 0.3 and conf_list[i-1] > 0.7:
                # Confidence dropped sharply = hand hidden
                motion_entry = motion_list[i]
                print(f"    🔍 Found occlusion start at frame {motion_entry['frame']}")
                return motion_entry['frame']

        # PATTERN 3: Hand at body/pocket level (suspicious position)
        for i in range(len(motion_list) - 1, max(0, len(motion_list) - 15), -1):
            left_wrist, right_wrist = motion_list[i]['hand_position']
            # Check if hand is at suspicious location (hip/pocket level)
            if left_wrist[1] > 400 or right_wrist[1] > 400:  # Assuming ~600px height video
                print(f"    🔍 Found suspicious position at frame {motion_list[i]['frame']}")
                return motion_list[i]['frame']

        return None

    def extract_full_event_clip(self, current_frame_num, event_name):
        """
        Extract the FULL context of the event
        From when it STARTED to now (with buffer)
        """

        # Find where event started
        event_start = self.detect_suspicious_start_frame()

        if event_start is None:
            event_start = current_frame_num - int(self.fps * 3)  # Default: 3 seconds back

        # Get frames from event start to current frame + 1 second buffer
        event_end = current_frame_num + int(self.fps * 1)

        print(f"    📍 Event window: frames [{event_start}] → [{current_frame_num}] → [{event_end}]")
        print(f"    ⏱️  Duration: {(current_frame_num - event_start) / self.fps:.1f}s so far")

        # Extract frames from frame store
        event_frames = []
        for frame_data in self.frame_store:
            if event_start <= frame_data['frame_num'] <= event_end:
                event_frames.append(frame_data['frame'])

        print(f"    🎬 Extracted {len(event_frames)} frames (full context)")

        if len(event_frames) == 0:
            print(f"    ⚠️  No frames in range, using last {int(self.fps * 5)} frames")
            event_frames = [f['frame'] for f in list(self.frame_store)[-int(self.fps * 5):]]

        return event_frames, event_start, current_frame_num



mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float16).view(3, 1, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float16).view(3, 1, 1, 1)


def preprocess_event_frames_fp16(frames):
    """
    Intelligently sample from full event frames while preserving context
    """
    num_frames = len(frames)

    if num_frames <= 16:
        # Few frames: pad to 16
        while len(frames) < 16:
            frames.append(frames[-1])
    else:
        # Many frames: sample intelligently
        # Keep the ENTIRE sequence by sampling densely

        # Strategy:
        # - Keep first few frames (setup/beginning)
        # - Sample middle densely (action)
        # - Keep last frames (outcome)

        indices = []

        # Always include first frame
        indices.append(0)

        # Sample first quarter densely
        first_quarter = num_frames // 4
        for i in range(1, first_quarter, max(1, first_quarter // 4)):
            indices.append(i)

        # Sample middle section
        for i in range(first_quarter, num_frames - first_quarter, max(1, (num_frames - 2*first_quarter) // 6)):
            indices.append(i)

        # Sample last quarter
        for i in range(num_frames - first_quarter, num_frames - 1, max(1, first_quarter // 4)):
            indices.append(i)

        # Always include last frame
        if indices[-1] != num_frames - 1:
            indices.append(num_frames - 1)

        # Deduplicate and sort
        indices = sorted(list(set(indices)))[:16]

        # Fill up to 16 if needed
        while len(indices) < 16:
            indices.append(indices[-1])

        print(f"      Sampled {len(indices)} frames from {num_frames} (indices: {indices[:8]}...)")
        frames = [frames[i] for i in indices]

    # Preprocess
    frames_tensor = []
    for frame in frames[:16]:
        frame_rgb = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
        frames_tensor.append(torch.from_numpy(frame_rgb).float())

    video_tensor = torch.stack(frames_tensor).permute(3, 0, 1, 2)
    video_tensor = video_tensor.half().to(device) / 255.0
    video_tensor = (video_tensor - mean) / std
    video_tensor = video_tensor.unsqueeze(0)

    return video_tensor



# ======================== MULTI-TRIGGER DETECTOR ========================


class MultiTriggerDetector:
    def __init__(self, fps=30):
        self.fps = fps
        self.motion_state = {
            'reaching_detected': False,
            'reach_frame': None,
            'last_height': None
        }
        self.occlusion_state = {
            'high_conf_frames': 0,
            'low_conf_frames': 0,
            'occlusion_start': None
        }
        self.proximity_state = {
            'suspicious_position_frames': 0,
            'position_start': None
        }
        self.acceleration_state = {
            'prev_wrist': None,
            'prev_speed': 0,
            'acceleration_frames': 0
        }

    def detect_hand_occlusion(self, left_conf, right_conf):
        current_conf = max(left_conf, right_conf)

        if current_conf > 0.7:
            self.occlusion_state['high_conf_frames'] += 1
            self.occlusion_state['low_conf_frames'] = 0
        elif current_conf < 0.4:
            self.occlusion_state['low_conf_frames'] += 1
            self.occlusion_state['high_conf_frames'] = 0

        if (self.occlusion_state['high_conf_frames'] > 20 and
            self.occlusion_state['low_conf_frames'] > 15):

            print(f"  👻 OCCLUSION TRIGGER")
            self.occlusion_state = {'high_conf_frames': 0, 'low_conf_frames': 0, 'occlusion_start': None}
            return True, 0.75

        return False, 0.0

    def detect_suspicious_position(self, left_wrist, right_wrist, hip_level, shoulder_level, body_left, body_right):
        torso_center = (body_left + body_right) / 2
        waist_zone = abs(hip_level - shoulder_level) * 0.5

        left_at_body = (
            (abs(left_wrist[1] - hip_level) < waist_zone) or
            (left_wrist[0] < body_left - 20) or
            (left_wrist[1] > hip_level + 30 and abs(left_wrist[0] - torso_center) < 100)
        )

        right_at_body = (
            (abs(right_wrist[1] - hip_level) < waist_zone) or
            (right_wrist[0] > body_right + 20) or
            (right_wrist[1] > hip_level + 30 and abs(right_wrist[0] - torso_center) < 100)
        )

        if left_at_body or right_at_body:
            self.proximity_state['suspicious_position_frames'] += 1
        else:
            self.proximity_state['suspicious_position_frames'] = 0

        if self.proximity_state['suspicious_position_frames'] > self.fps * 2:
            print(f"  🤫 SUSPICIOUS POSITION TRIGGER")
            self.proximity_state['suspicious_position_frames'] = 0
            return True, 0.70

        return False, 0.0

    def detect_hand_motion_pattern(self, left_wrist, right_wrist, left_conf, right_conf, frame_num):
        if left_conf < 0.3 and right_conf < 0.3:
            return False, 0.0

        current_height = min(
            left_wrist[1] if left_conf > 0.3 else float('inf'),
            right_wrist[1] if right_conf > 0.3 else float('inf')
        )

        if self.motion_state['last_height'] is None:
            self.motion_state['last_height'] = current_height
            return False, 0.0

        height_change = self.motion_state['last_height'] - current_height

        if height_change < -60 and (left_conf > 0.4 or left_conf > 0.4):
            if not self.motion_state['reaching_detected']:
                print(f"  ⬇️  REACHING DOWN")
                self.motion_state['reaching_detected'] = True
                self.motion_state['reach_frame'] = frame_num

        if self.motion_state['reaching_detected']:
            frames_since = frame_num - self.motion_state['reach_frame']

            hand_hidden = (left_conf < 0.3 and right_conf < 0.3)
            hand_moved_up = height_change > 40

            if (hand_hidden or hand_moved_up) and 5 < frames_since < 180:
                print(f"  ⬆️  REACHING PATTERN COMPLETE")
                self.motion_state['reaching_detected'] = False
                self.motion_state['reach_frame'] = None
                return True, 0.80

        if self.motion_state['reaching_detected'] and (frame_num - self.motion_state['reach_frame']) > 180:
            self.motion_state['reaching_detected'] = False
            self.motion_state['reach_frame'] = None

        self.motion_state['last_height'] = current_height
        return False, 0.0

    def detect_rapid_movement(self, left_wrist, right_wrist, left_conf, right_conf):
        if left_conf > 0.5:
            current_pos = left_wrist
        elif right_conf > 0.5:
            current_pos = right_wrist
        else:
            return False, 0.0

        if self.acceleration_state['prev_wrist'] is None:
            self.acceleration_state['prev_wrist'] = current_pos
            return False, 0.0

        distance = np.linalg.norm(current_pos - self.acceleration_state['prev_wrist'])

        if distance > 30:
            self.acceleration_state['acceleration_frames'] += 1
        else:
            self.acceleration_state['acceleration_frames'] = 0

        self.acceleration_state['prev_wrist'] = current_pos

        if self.acceleration_state['acceleration_frames'] > 3:
            print(f"  ⚡ RAPID ACCELERATION TRIGGER")
            self.acceleration_state['acceleration_frames'] = 0
            return True, 0.65

        return False, 0.0

    def analyze(self, keypoints, person_bbox, frame_num):
        if keypoints is None or len(keypoints) < 17:
            return False, 0.0

        left_wrist = keypoints[9][:2]
        right_wrist = keypoints[10][:2]
        left_conf = keypoints[9][2]
        right_conf = keypoints[10][2]

        hip_level = (keypoints[11][1] + keypoints[12][1]) / 2
        shoulder_level = (keypoints[5][1] + keypoints[6][1]) / 2
        body_left = person_bbox[0]
        body_right = person_bbox[2]

        triggers = []

        t1, c1 = self.detect_hand_occlusion(left_conf, right_conf)
        if t1: triggers.append((t1, c1, "occlusion"))

        t2, c2 = self.detect_suspicious_position(left_wrist, right_wrist, hip_level, shoulder_level, body_left, body_right)
        if t2: triggers.append((t2, c2, "position"))

        t3, c3 = self.detect_hand_motion_pattern(left_wrist, right_wrist, left_conf, right_conf, frame_num)
        if t3: triggers.append((t3, c3, "motion"))

        t4, c4 = self.detect_rapid_movement(left_wrist, right_wrist, left_conf, right_conf)
        if t4: triggers.append((t4, c4, "acceleration"))

        if triggers:
            best = max(triggers, key=lambda x: x[1])
            return best[0], best[1]

        return False, 0.0



# ======================== MVIT PROCESSOR (WITH INCREASED THRESHOLDS) ========================


class MViTContextAwareWorker:
    def __init__(self, mvit_model):
        self.mvit_model = mvit_model
        self.queue = Queue(maxsize=3)
        self.results = Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.stop_flag = False
        self.thread.start()
        print("🔧 Context-aware MViT worker started\n")

    def _worker(self):
        while not self.stop_flag:
            try:
                job = self.queue.get(timeout=1.0)
                if job is None:
                    break

                frames, frame_num, trigger_type, event_info = job

                try:
                    print(f"  🎬 Processing {len(frames)} frames (full event context)...")
                    video_tensor = preprocess_event_frames_fp16(frames)

                    with torch.no_grad():
                        output = self.mvit_model(video_tensor)
                        probs = torch.softmax(output.float(), dim=1)[0]
                        pred = torch.argmax(probs).item()
                        conf = probs[pred].item()

                    label = "SHOPLIFTING" if pred == 1 else "NORMAL"

                    # ===== INCREASED THRESHOLD & VALIDATION =====
                    # Instead of 0.50, now require 0.75 (75%) confidence
                    # This prevents false positives from just picking up items

                    print(f"  📊 MViT Raw Output: {label} ({conf*100:.1f}%)")

                    # Validation: Only report shoplifting if VERY confident
                    if pred == 1:
                        # Class 1 = Shoplifting prediction
                        # Require: 75%+ confidence (increased from 50%)
                        if conf < 0.75:
                            print(f"  ⚠️  Below threshold ({conf*100:.1f}% < 75%) - Treating as NORMAL")
                            pred = 0
                            label = "NORMAL"

                    self.results.put({
                        'frame': frame_num,
                        'pred': pred,
                        'conf': conf,
                        'label': label,
                        'trigger': trigger_type,
                        'event_info': event_info
                    })

                except Exception as e:
                    print(f"  ❌ MViT error: {e}")

            except Empty:
                continue
            except Exception as e:
                print(f"  ⚠️  Worker error: {e}")

    def submit(self, frames, frame_num, trigger_type, event_info):
        try:
            self.queue.put_nowait((frames.copy(), frame_num, trigger_type, event_info))
            return True
        except:
            return False

    def get_result(self):
        try:
            return self.results.get_nowait()
        except:
            return None

    def stop(self):
        self.stop_flag = True
        self.queue.put(None)
        self.thread.join(timeout=5)



# ======================== MAIN DETECTION ========================


def run_detection(video_path, output_path=None, display=True):
    print(f"\n{'='*80}")
    print(f"🎥 Processing: {video_path}")
    print(f"Mode: EVENT-BASED CONTEXT TRACKING")
    print(f"MViT Threshold: 75% (increased from 50%)")
    print(f"{'='*80}\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 Video: {total} frames | {fps:.1f} FPS | {width}x{height}")
    print(f"🎯 Strategy: Continuous tracking → Full event context → MViT (75% threshold)\n")

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detector = MultiTriggerDetector(fps=int(fps))
    tracker = EventBasedFrameTracker(fps=int(fps))
    mvit_worker = MViTContextAwareWorker(mvit_model)

    frame_count = 0
    alerts = 0
    confirmations = 0
    false_alarms = 0
    last_alert_frame = 0

    last_result = None
    processing = False

    print("="*80 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # CRUCIAL: Record EVERY frame for context
            tracker.record_frame(frame.copy(), frame_count)

            display_frame = frame.copy()

            # Check MViT results
            result = mvit_worker.get_result()
            if result:
                processing = False
                last_result = result

                event_info = result.get('event_info', {})
                print(f"\n  📊 MViT Result: {result['label']} ({result['conf']*100:.1f}%)")
                print(f"     Event: frames [{event_info.get('start')}] → [{event_info.get('end')}]")
                print(f"     Trigger: {result['trigger']}")

                # ONLY ALERT if pred == 1 AND confidence >= 75%
                if result['pred'] == 1 and result['conf'] >= 0.75:
                    confirmations += 1
                    print(f"\n{'='*80}")
                    print(f"🛑 SHOPLIFTING CONFIRMED! (Confidence: {result['conf']*100:.1f}%)")
                    print(f"   Event duration: {(event_info.get('end', 0) - event_info.get('start', 0)) / int(fps):.1f}s")
                    print(f"{'='*80}\n")

                    cv2.putText(display_frame, "SHOPLIFTING DETECTED", (50, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                else:
                    # False alarm or low confidence
                    false_alarms += 1
                    print(f"  ✓ False alarm (confidence {result['conf']*100:.1f}% < 75%)")

            # YOLO pose detection
            results = yolo_pose(frame, conf=0.3, verbose=False)[0]

            if results.boxes is not None and len(results.boxes) > 0:
                person_box = results.boxes[0].xyxy[0].cpu().numpy()

                if results.keypoints is not None:
                    keypoints = results.keypoints.data[0].cpu().numpy()

                    # Track hand motion continuously
                    left_wrist = keypoints[9][:2]
                    right_wrist = keypoints[10][:2]
                    left_conf = keypoints[9][2]
                    right_conf = keypoints[10][2]

                    tracker.track_hand_motion(left_wrist, right_wrist, left_conf, right_conf, frame_count)

                    # Check if suspicious
                    is_suspicious, confidence = detector.analyze(keypoints, person_box, frame_count)

                    x1, y1, x2, y2 = map(int, person_box)

                    if is_suspicious:
                        color = (0, 0, 255)
                        label = f"⚠️  SUSPICIOUS ({confidence:.0%})"

                        # Extract FULL event context
                        if not processing and frame_count - last_alert_frame > 45:
                            alerts += 1

                            event_frames, event_start, event_end = tracker.extract_full_event_clip(
                                frame_count, f"alert_{alerts}"
                            )

                            print(f"\n🚨 Alert #{alerts}: Suspicious motion at frame {frame_count}")
                            print(f"   Submitting to MViT for verification (requires 75% confidence)...")

                            event_info = {
                                'start': event_start,
                                'end': event_end,
                                'total_frames': len(event_frames)
                            }

                            if mvit_worker.submit(event_frames, frame_count, "event_context", event_info):
                                processing = True
                                last_alert_frame = frame_count
                    else:
                        color = (0, 255, 0)
                        label = "✓ NORMAL"

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(display_frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    if processing:
                        cv2.putText(display_frame, "🔍 VERIFYING WITH MVIT (75% threshold)...", (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    if last_result:
                        result_text = f"Last: {last_result['label']} ({last_result['conf']*100:.0f}%)"
                        color_text = (0, 0, 255) if last_result['pred'] == 1 else (0, 255, 0)
                        cv2.putText(display_frame, result_text, (50, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_text, 2)

            else:
                cv2.putText(display_frame, "No person detected", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if display:
                cv2.imshow('Context-Aware Shoplifting Detection (75% Threshold)', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if output_path:
                out.write(display_frame)

    finally:
        mvit_worker.stop()
        cap.release()
        if output_path:
            out.release()
        if display:
            cv2.destroyAllWindows()

    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    print(f"Total frames: {frame_count}")
    print(f"Alerts triggered: {alerts}")
    print(f"Confirmed (≥75% confidence): {confirmations}")
    print(f"False alarms (<75% confidence): {false_alarms}")
    print(f"False alarm rate: {false_alarms/max(1, alerts)*100:.1f}%")
    print("="*80)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shoplifting Detection System')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, help='Path to output video file (optional)')
    parser.add_argument('--display', action='store_true', help='Display video during processing')

    args = parser.parse_args()

    run_detection(
        video_path=args.video,
        output_path=args.output,
        display=args.display
    )
