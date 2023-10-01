from PIL import Image
import os
import numpy as np

class DataIngetion:

    def __init__(self):
        self.emotion_folders = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.image_arrays = []
        self.labels = []
    
    def collect_data(self, base_folder):

        for emotion_id, emotion in enumerate(self.emotion_folders):
            folder_path = os.path.join(base_folder, emotion)

            # Check if the folder exists
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                # Get a list of all files in the folder
                files = os.listdir(folder_path)

                # Filter image files (you can modify the extensions as needed)
                image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

                # Read and convert images to arrays and associate with labels
                for image_file in image_files:
                    image_path = os.path.join(folder_path, image_file)
                    try:
                        img = Image.open(image_path)
                        img_array = np.array(img)  # Convert image to array
                        self.image_arrays.append(img_array)
                        self.labels.append(emotion_id)  # Assign the emotion_id as the label
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")

    def get_data(self):
        return self.image_arrays, self.labels
    

if __name__ == "__main__":
    data_ingetion = DataIngetion()
    data_ingetion.collect_data(r'ML_end_to_end/artifacts/images_data/images_data/ferdata/train')
    X, y = data_ingetion.get_data()
    print(X[0], y[0])