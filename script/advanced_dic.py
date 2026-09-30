import cv2
import numpy as np
import websocket
import json
import time

# --- Configuration ---
WEBSOCKET_URL = "ws://localhost:8080"
FAKE_STRESS_MULTIPLIER = 5.0 

def select_two_rois(frame):
    """Allows the user to draw two bounding boxes for the top and bottom gauge markers."""
    print("Draw a box around the TOP speckle region, then press SPACE or ENTER.")
    roi_top = cv2.selectROI("DIC Setup", frame, fromCenter=False, showCrosshair=True)
    
    print("Draw a box around the BOTTOM speckle region, then press SPACE or ENTER.")
    roi_bottom = cv2.selectROI("DIC Setup", frame, fromCenter=False, showCrosshair=True)
    
    cv2.destroyWindow("DIC Setup")
    return roi_top, roi_bottom

def get_features_in_roi(gray_frame, roi, max_corners=50):
    """Finds the best tracking features (speckles) inside a specific bounding box."""
    x, y, w, h = roi
    mask = np.zeros_like(gray_frame)
    mask[y:y+h, x:x+w] = 255
    
    # Shi-Tomasi corner detector to find distinct speckles
    corners = cv2.goodFeaturesToTrack(
        gray_frame, 
        maxCorners=max_corners, 
        qualityLevel=0.01, 
        minDistance=5, 
        mask=mask
    )
    return corners

def main():
    # Connect to Dashboard
    try:
        ws = websocket.WebSocket()
        ws.connect(WEBSOCKET_URL)
        print("Connected to Dashboard.")
    except Exception as e:
        print(f"WebSocket Error: {e}")
        return

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab camera frame.")
        return

    # 1. Setup Phase: User selects the top and bottom tracking regions
    roi_top, roi_bottom = select_two_rois(frame)
    
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Extract specific speckles from those regions
    p0_top = get_features_in_roi(old_gray, roi_top)
    p0_bottom = get_features_in_roi(old_gray, roi_bottom)

    if p0_top is None or p0_bottom is None:
        print("Could not find enough features. Ensure the specimen has a high-contrast speckle pattern.")
        return

    # Calculate initial baseline length (L0) using the average Y-coordinate of both clusters
    avg_y_top_initial = np.mean(p0_top[:, 0, 1])
    avg_y_bottom_initial = np.mean(p0_bottom[:, 0, 1])
    initial_length = abs(avg_y_bottom_initial - avg_y_top_initial)

    # Optical flow configuration
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    last_send_time = time.time()
    cv2.namedWindow('Advanced DIC Tracking')

    print("--- TEST RUNNING ---")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow for both sets of points
        p1_top, st_top, err_top = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0_top, None, **lk_params)
        p1_bottom, st_bottom, err_bottom = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0_bottom, None, **lk_params)

        # Keep only the points that were successfully tracked
        good_new_top = p1_top[st_top == 1]
        good_new_bottom = p1_bottom[st_bottom == 1]

        # Draw the tracking points on the video feed
        for pt in good_new_top:
            a, b = pt.ravel()
            cv2.circle(frame, (int(a), int(b)), 3, (0, 255, 0), -1)
            
        for pt in good_new_bottom:
            c, d = pt.ravel()
            cv2.circle(frame, (int(c), int(d)), 3, (0, 0, 255), -1)

        # Calculate average displacement to determine strain
        if len(good_new_top) > 5 and len(good_new_bottom) > 5:
            avg_y_top = np.mean(good_new_top[:, 1])
            avg_y_bottom = np.mean(good_new_bottom[:, 1])
            
            current_length = abs(avg_y_bottom - avg_y_top)
            delta_l = current_length - initial_length
            strain = delta_l / initial_length if initial_length > 0 else 0
            
            fake_stress = strain * FAKE_STRESS_MULTIPLIER

            # UI Overlay
            cv2.putText(frame, f"Strain: {strain:.5f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f"Tracked Subsets: {len(good_new_top) + len(good_new_bottom)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Stream data
            if time.time() - last_send_time > 0.1:
                # Cast to standard python float to avoid numpy JSON serialization errors
                payload = json.dumps({"strain": float(round(strain, 5)), "stress": float(round(fake_stress, 3))})
                ws.send(payload)
                last_send_time = time.time()

            # Update old frame and points
            old_gray = frame_gray.copy()
            p0_top = good_new_top.reshape(-1, 1, 2)
            p0_bottom = good_new_bottom.reshape(-1, 1, 2)
            
        else:
            cv2.putText(frame, "CRITICAL TRACKING LOSS", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow('Advanced DIC Tracking', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    ws.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()