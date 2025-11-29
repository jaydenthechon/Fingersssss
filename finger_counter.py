import cv2
import numpy as np
import time

class FingerCounter:
    def __init__(self):
        # Skin color range in HSV - adjusted to reduce false positives
        self.lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        self.upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    
    def count_fingers(self, contour, frame):
        """
        Count fingers using convex hull and defects.
        
        Args:
            contour: Hand contour
            frame: Current frame for drawing
            
        Returns:
            Number of extended fingers
        """
        # Get convex hull
        hull = cv2.convexHull(contour, returnPoints=False)
        
        if len(hull) > 3:
            defects = cv2.convexityDefects(contour, hull)
            
            if defects is not None:
                finger_count = 0
                
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])
                    
                    # Calculate distances
                    a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    
                    # Calculate angle
                    angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
                    
                    # If angle is less than 90 degrees, count as finger
                    if angle <= np.pi / 2:
                        finger_count += 1
                        cv2.circle(frame, far, 8, (0, 0, 255), -1)
                
                return finger_count + 1  # Add 1 because fingers = defects + 1
        
        return 0
    
    def run(self):
        """Main loop to capture video and detect fingers."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        prev_time = 0
        
        print("Starting finger counter...")
        print("Press 'q' to quit")
        print("Show your hand in the green box area")
        
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read from webcam")
                break
            
            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            
            # Get frame dimensions
            height, width, _ = frame.shape
            
            # Draw ROI rectangle (smaller, centered)
            roi_x1, roi_y1 = int(width * 0.3), int(height * 0.2)
            roi_x2, roi_y2 = int(width * 0.7), int(height * 0.8)
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
            
            # Extract ROI
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # Convert to HSV and apply skin color mask
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
            
            # Apply filters to reduce noise
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=2)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.GaussianBlur(mask, (5, 5), 100)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour (assumed to be the hand)
                max_contour = max(contours, key=cv2.contourArea)
                
                if cv2.contourArea(max_contour) > 5000:
                    # Draw contour on ROI
                    cv2.drawContours(roi, [max_contour], -1, (255, 0, 0), 2)
                    
                    # Count fingers
                    finger_count = self.count_fingers(max_contour, roi)
                    
                    # Limit to 0-5 fingers
                    finger_count = min(5, max(0, finger_count))
                    
                    # Display finger count
                    cv2.putText(
                        frame,
                        f"Fingers: {finger_count}",
                        (200, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 255, 0),
                        3
                    )
            
            # Calculate and display FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            
            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )
            
            # Show frames
            cv2.imshow('Finger Counter', frame)
            cv2.imshow('Mask', mask)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    counter = FingerCounter()
    counter.run()
