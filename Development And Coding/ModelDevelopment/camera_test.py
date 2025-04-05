import cv2

def test_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return False
    ret, _ = cap.read()
    cap.release()
    return ret

print("🔎 Searching for working camera...")

for idx in range(5):
    if test_camera(idx):
        print(f"✅ Working camera found at index {idx}")
        break
else:
    print("❌ No working camera found.")
