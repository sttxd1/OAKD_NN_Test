#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import argparse
import time
import os
import json
'''
Deeplabv3 multiclass running on images from a directory at 30 FPS.

Blob is taken from ML training examples:
https://github.com/luxonis/depthai-ml-training/tree/master/colab-notebooks
'''

num_of_classes = 66  # Define the number of classes in the dataset

parser = argparse.ArgumentParser()
parser.add_argument("-dir", "--input_dir", help="Directory containing RGB input images", default='/home/st/scooter_ws/semantic_mapping_v2/datasets/dataset0925/top_rgb')
parser.add_argument("-nn", "--nn_model", help="Path to the model blob for inference", default='models/fastscnn_720x1280.blob', type=str)
parser.add_argument("-config", "--config_path", help="Path to the config.json file", default='/home/st/scooter_ws/semantic_mapping_v2/config/config_65.json')

args = parser.parse_args()

input_dir = args.input_dir
nn_path = args.nn_model
config_path = args.config_path

nn_shape1 = 1280  # Width
nn_shape2 = 720   # Height

# Load the configuration file
with open(config_path, 'r') as f:
    config = json.load(f)

# Extract labels
labels = config['labels']

# Create class index to color mapping
class_colors = {}
for idx, label in enumerate(labels):
    class_colors[idx] = label['color']  # The color is a list [R, G, B]

# Mean and scale values provided (in RGB order)
mean_rgb = [0.38373764, 0.37943604, 0.39990275]
scale_rgb = [0.32653687, 0.31468863, 0.31361703]

# Convert mean and scale values to BGR order (since OpenCV uses BGR)
mean_bgr = mean_rgb[::-1]
scale_bgr = scale_rgb[::-1]


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert an image to a planar format.
    """
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

def decode_deeplabv3p(output_tensor, class_colors):
    """
    Converts the output tensor into a color image based on the color map from config.json.
    """
    class_map = output_tensor.reshape(nn_shape2, nn_shape1)

    # Create an empty color image
    color_output = np.zeros((class_map.shape[0], class_map.shape[1], 3), dtype=np.uint8)

    # Assign colors to each class based on the color map
    for class_idx, color in class_colors.items():
        color_output[class_map == int(class_idx)] = color  # Assign the color to the pixels of the class

    return color_output

# def show_deeplabv3p(output_colors, frame):
#     return cv2.addWeighted(frame,1, output_colors,0.4,0)

def load_images_from_directory(directory, nn_shape1, nn_shape2):
    """
    Generator function to load images one by one from the directory.
    Preprocesses each image before yielding.
    """
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img_path = os.path.join(directory, filename)
            # Load the image using OpenCV
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to load image {img_path}")
                continue
            # Resize the image to the required shape
            original_image = image #cv2.resize(image, (nn_shape1, nn_shape2))
            # image_normalized = original_image.astype(np.float32) / 255.0  # Scale to [0,1]
            # image_normalized -= mean_bgr  # Subtract mean
            # image_normalized /= scale_bgr  # Divide by scale

            # Convert the image to a format that DepthAI can process
            frame = dai.ImgFrame()
            frame.setData(to_planar(original_image, (nn_shape1, nn_shape2)))
            frame.setType(dai.ImgFrame.Type.BGR888p)
            frame.setWidth(nn_shape1)
            frame.setHeight(nn_shape2)

            # Yield the frame and the original image for visualization
            yield frame, original_image

# Start defining a pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(version=dai.OpenVINO.VERSION_2022_1)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.create(dai.node.NeuralNetwork)
detection_nn.setBlobPath(nn_path)
detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

# Create an input queue for sending images to the neural network
xin_nn = pipeline.create(dai.node.XLinkIn)
xin_nn.setStreamName("nn_input")
xin_nn.out.link(detection_nn.input)

# Create outputs for the neural network
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

with dai.Device(pipeline) as device:
    # Input and output queues
    q_in = device.getInputQueue(name="nn_input", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    start_time = time.time()
    fps = 30  # Set FPS to 30
    frame_duration = 1 / fps  # Time per frame in seconds

    image_generator = load_images_from_directory(input_dir, nn_shape1, nn_shape2)

    while True:
        try:
            # Get the next image frame from the generator
            frame, original_image = next(image_generator)
        except StopIteration:
            # If we've reached the end of the directory, restart from the first image
            image_generator = load_images_from_directory(input_dir, nn_shape1, nn_shape2)
            frame, original_image = next(image_generator)

        # Send the image frame to the device
        q_in.send(frame)

        # Get the neural network output
        in_nn = q_nn.get()

        # Process the neural network output
        output_data = in_nn.getFirstLayerInt32()
        if len(output_data) == 0:
            print("No output data from NN.")
            continue

        output_tensor = np.array(output_data).reshape(nn_shape1, nn_shape2)
        # class_map = output_tensor
        # found_classes = np.unique(class_map)
        output_colors = decode_deeplabv3p(output_tensor,class_colors)
        output_colors_rgb = cv2.cvtColor(output_colors, cv2.COLOR_BGR2RGB)

        # Display the results
        # result_frame = show_deeplabv3p(output_colors, original_image)
        result_frame = output_colors_rgb
        # cv2.putText(result_frame, "Found classes {}".format(found_classes), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Segmentation", result_frame)

        # Simulate 30 FPS
        time_elapsed = time.time() - start_time
        if time_elapsed < frame_duration:
            time.sleep(frame_duration - time_elapsed)
        start_time = time.time()

        if cv2.waitKey(1) == ord('q'):
            break


