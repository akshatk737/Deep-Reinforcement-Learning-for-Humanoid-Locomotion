import cv2
import numpy as np
from pathlib import Path
import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read the image from {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, target_size)
    image_normalized = image_resized / 255.0
    return np.asarray(image_normalized, dtype=np.float32)

images_dir = Path(r"./images")
image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))

if not image_files:
    print(f"No image files found in {images_dir}")
else:
    image_arrays = {}
    for img_path in image_files:
        try:
            image_arrays[img_path.stem] = load_and_preprocess_image(img_path)
            print(f"Loaded: {img_path.name}")
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    for name, arr in image_arrays.items():
        print(f"{name}: shape = {arr.shape}")

    print(f"\nTotal images loaded: {len(image_arrays)}")



mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)

BODY_25_MAPPING = [
    0, 12, 12, 14, 16, 11, 13, 15,
    24, 24, 26, 28, 23, 25, 27,
    5, 2, 8, 7, 31, 31, 29, 32, 32, 30
]

BODY_25_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8),
    (8, 9), (9, 10), (10, 11),
    (8, 12), (12, 13), (13, 14),
    (0, 15), (15, 17),
    (0, 16), (16, 18),
    (14, 19), (19, 20), (14, 21),
    (11, 22), (22, 23), (11, 24)
]

def mediapipe_to_openpose_landmarks(landmarks, image_shape):
    h, w = image_shape[:2]
    skeleton = np.zeros((25, 3), dtype=np.float32)
    for i, mp_idx in enumerate(BODY_25_MAPPING):
        if mp_idx == 0 and i > 0:
            skeleton[i] = [0, 0, 0]
        else:
            lm = landmarks[mp_idx]
            skeleton[i] = [lm.x * w, lm.y * h, lm.visibility]
    return skeleton

def select_main_skeleton(skeletons):
    if len(skeletons) == 0:
        return None
    max_area, selected_skeleton = 0, skeletons[0]
    for skeleton in skeletons:
        valid_points = skeleton[skeleton[:, 2] > 0]
        if valid_points.shape[0] == 0:
            continue
        x_min, y_min = valid_points[:, 0].min(), valid_points[:, 1].min()
        x_max, y_max = valid_points[:, 0].max(), valid_points[:, 1].max()
        area = (x_max - x_min) * (y_max - y_min)
        if area > max_area:
            max_area, selected_skeleton = area, skeleton
    return selected_skeleton

images_dir = Path(r"./images")
image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))

for image_path in image_files:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Skipping unreadable file: {image_path}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    skeletons = []
    if results.pose_landmarks:
        skeleton = mediapipe_to_openpose_landmarks(results.pose_landmarks.landmark, image.shape)
        skeletons.append(skeleton)

    skeletons = np.array(skeletons)
    main_skeleton = select_main_skeleton(skeletons)

    if main_skeleton is None:
        print(f"No skeleton detected in {image_path.name}")
        continue

    vis_img = image.copy()
    for i, (x, y, conf) in enumerate(main_skeleton):
        if conf > 0:
            cv2.circle(vis_img, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(vis_img, str(i), (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for (i, j) in BODY_25_CONNECTIONS:
        if main_skeleton[i, 2] > 0 and main_skeleton[j, 2] > 0:
            x1, y1 = int(main_skeleton[i, 0]), int(main_skeleton[i, 1])
            x2, y2 = int(main_skeleton[j, 0]), int(main_skeleton[j, 1])
            cv2.line(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    print(f"Displaying skeleton for: {image_path.name}")
    cv2.imshow("Skeleton", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


