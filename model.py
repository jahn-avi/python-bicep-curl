import cv2
import PIL.Image
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

class Model:
    def __init__(self):
        # Allow more iterations to avoid convergence warnings
        self.model = LinearSVC(max_iter=5000)
        self.scaler = StandardScaler()

    def train_model(self, counters):
        img_list = []
        class_list = []

        # Load and flatten training images for class 1 (Extended)
        for i in range(1, counters[0]):
            img = cv2.imread(f"1/frame{i}.jpg")[:, :, 0]  # Load grayscale
            img = img.reshape(-1)  # Flatten
            img_list.append(img)
            class_list.append(1)

        # Load and flatten training images for class 2 (Contracted)
        for i in range(1, counters[1]):
            img = cv2.imread(f"2/frame{i}.jpg")[:, :, 0]
            img = img.reshape(-1)
            img_list.append(img)
            class_list.append(2)

        # Convert to numpy arrays
        img_array = np.array(img_list)
        class_array = np.array(class_list)

        # Scale features
        img_array_scaled = self.scaler.fit_transform(img_array)

        # Train the model
        self.model.fit(img_array_scaled, class_array)
        print("Model successfully trained!")

    def predict(self, frame):
        # Extract frame from tuple if needed
        if isinstance(frame, tuple):
            frame = frame[1]

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Save and resize using PIL
        cv2.imwrite("frame.jpg", gray_frame)
        img = PIL.Image.open("frame.jpg")
        img.thumbnail((150, 113), PIL.Image.Resampling.LANCZOS)
        img.save("frame.jpg")

        # Read the resized image again
        img = cv2.imread("frame.jpg")[:, :, 0]
        img = img.reshape(-1)

        # Scale using the same scaler from training
        img_scaled = self.scaler.transform([img])
        prediction = self.model.predict(img_scaled)
        return prediction[0]
