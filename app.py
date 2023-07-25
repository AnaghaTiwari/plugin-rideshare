import argparse

import cv2
import numpy as np
import torch
import time



import ultralytics
from ultralytics import YOLO
# from ultralytics.yolo import ROOT, yaml_load
# from ultralytics.yolo import check_requirements, check_yaml

import logging
from waggle.plugin import Plugin
from waggle.data.vision import Camera

class Yolov8:

    def __init__(self, model, stream, conf_thres, iou_thres):
        """
        Initializes an instance of the Yolov8 class.

        Args:
            model: Path to the YoloV8 model.
            stream: Path to the input livestream/video (i.e sample)
            conf_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.model = YOLO(model)
        self.stream = stream
        self.confidence_thres = conf_thres
        self.iou_thres = iou_thres

        # Load the class names - rideshare
        self.classes = {0: 'rideshare'}

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Bounding box coordinates
        x1, y1, w, h = box

        # color for class ID
        color = self.color_palette[class_id]

        # Draw bounding box on image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create label text with class and score
        label = f'{self.classes[class_id]}: {score:.2f}'

        # Calculate dimensions of label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw filled rectangle as the background for the label text
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv2.FILLED)

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, sample):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Read the input image from frame using OpenCV
        # self.img = cv2.imread(self.stream)

        ##########################
        # self.img = sample.data
        self.img = sample

        # self.img = cv2.imread(sample)
        
        # Get the height and width of the input image
        img = cv2.resize(self.img, (640, 640))
        self.img_height = 640
        self.img_width = 640
        #self.img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    # postprocess => read in sticker, publish detection for s1
    def postprocess_1(self, sample, plugin, args, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The results/output of model
            plugin: plugin to save output image to

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        #####################
        timestamp = sample.timestamp
        # timestamp = 946706400000000000
        # Transpose and squeeze the output to match the expected shape

        ######################################
        # outputs = np.transpose(np.squeeze(output[0]))
        # rows = outputs.shape[0]
            
        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each output in the outputs array
        ################ NOT NEEDD #########
        
        # for i in ouput:
        #     # Extract the class scores from the current row
        #     classes_scores = outputs[i][4:]

        #     # Find the maximum score among the class scores
        #     max_score = np.amax(classes_scores)

        #     # If the maximum score is above the confidence threshold
        #     if max_score >= self.confidence_thres:
        #         # Get the class ID with the highest score
        #         class_id = np.argmax(classes_scores)

        #         # Extract the bounding box coordinates from the current row
        #         x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

        #         # Calculate the scaled coordinates of the bounding box
        #         left = int((x - w / 2) * x_factor)
        #         top = int((y - h / 2) * y_factor)
        #         width = int(w * x_factor)
        #         height = int(h * y_factor)

        #         # Add the class ID, score, and box coordinates to the respective lists
        #         class_ids.append(class_id)
        #         scores.append(max_score)
        #         boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        # indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        
        detection_stats = 'found: '
        found = {}
        # Iterate over the selected indices after non-maximum suppression



        ###############################
        input_image = sample.data
        # input_image = cv2.imread(sample)        

        ### TEST ####
        cv2.imwrite('test.jpg', input_image)
        plugin.upload_file('test.jpg')
        
        ############# NOT NEEDED ##############
        # for i in indices:
        #     # Get the box, score, and class ID corresponding to the index
        #     box = boxes[i]
        #     score = scores[i]
        #     class_id = class_ids[i]
            
        #     # Draw the detection on the input frame, and save to plugin using cv2
        #     self.draw_detections(input_image, box, score, class_id)
        #     cv2.imwrite('yolov8.jpg', input_image)
        #     plugin.upload_file('yolov8.jpg')
        #     print('image saved')

        #     if not class_id in found:
        #         found[class_id] = 1
        #     else:
        #         found[class_id] += 1
        count = 0
        name = []
        for result in output:
            res_plotted = result.plot()
            cv2.imwrite('result.jpeg', res_plotted)
            # plugin.upload_file('result.jpeg')

            boxes = result.boxes.cpu().numpy()                         # get boxes on cpu in numpy
            for box in boxes:                                          # iterate boxes
                name.append(result.names[int(box.cls[0])])
                count += 1

        for result in output:
            boxes = result.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                r = box.xyxy[0].astype(int)
                crop = input_image[r[1]:r[3], r[0]:r[2]]
                cv2.imwrite("crop.jpeg", crop)
                detection2 = Yolov8(args.model2, args.stream, 0.3, args.iou_thres)
                output = detection2.main('crop.jpeg', 2)
                
                # plugin.upload_file("crop.jpeg")
            
        # print detection stats
        TOPIC_TEMPLATE = 'rideshare'
        # for name, count in found.items():
        for i in name:
            # detection_stats += f'{class_id}[{count}] '
            detection_stats += f'{i}, {count}'
            # plugin.publish(f'{class_id}', count, timestamp = timestamp)
            # plugin.publish(f'{i}', count, timestamp)
            # plugin.publish(f'{TOPIC_TEMPLATE}.{str(i)}', str(count), str(timestamp))
            
            # plugin.publish('rideshare.log', str(count), timestamp=timestamp)
            
        # print(detection_stats)
            

        # Return the modified input image
        
        return input_image


    # postprocess_s2 - specifically for not publishing crop sticker
    def postprocess_2(self, sample, plugin, args, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The results/output of model
            plugin: plugin to save output image to

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        #####################
        timestamp = sample.timestamp
        # timestamp = 946706400000000000
        # Transpose and squeeze the output to match the expected shape

        ######################################
        # outputs = np.transpose(np.squeeze(output[0]))
        # rows = outputs.shape[0]
            
        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each output in the outputs array
        ################ NOT NEEDD #########
        
        # for i in ouput:
        #     # Extract the class scores from the current row
        #     classes_scores = outputs[i][4:]

        #     # Find the maximum score among the class scores
        #     max_score = np.amax(classes_scores)

        #     # If the maximum score is above the confidence threshold
        #     if max_score >= self.confidence_thres:
        #         # Get the class ID with the highest score
        #         class_id = np.argmax(classes_scores)

        #         # Extract the bounding box coordinates from the current row
        #         x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

        #         # Calculate the scaled coordinates of the bounding box
        #         left = int((x - w / 2) * x_factor)
        #         top = int((y - h / 2) * y_factor)
        #         width = int(w * x_factor)
        #         height = int(h * y_factor)

        #         # Add the class ID, score, and box coordinates to the respective lists
        #         class_ids.append(class_id)
        #         scores.append(max_score)
        #         boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        # indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        
        detection_stats = 'found: '
        found = {}
        # Iterate over the selected indices after non-maximum suppression



        ###############################
        input_image = sample.data
        # input_image = cv2.imread(sample)        
        

        ############# NOT NEEDED ##############
        # for i in indices:
        #     # Get the box, score, and class ID corresponding to the index
        #     box = boxes[i]
        #     score = scores[i]
        #     class_id = class_ids[i]
            
        #     # Draw the detection on the input frame, and save to plugin using cv2
        #     self.draw_detections(input_image, box, score, class_id)
        #     cv2.imwrite('yolov8.jpg', input_image)
        #     plugin.upload_file('yolov8.jpg')
        #     print('image saved')

        #     if not class_id in found:
        #         found[class_id] = 1
        #     else:
        #         found[class_id] += 1
        count = 0
        name = []
        bool_found = False
        for result in output:
            res_plotted = result.plot()
            cv2.imwrite('result.jpeg', res_plotted)
            # plugin.upload_file('result.jpeg')

            boxes = result.boxes.cpu().numpy()                         # get boxes on cpu in numpy
            for box in boxes:                                          # iterate boxes
                name.append(result.names[int(box.cls[0])])
                count += 1
                bool_found = True
                
        # crop to return
        crop = input_image
        
        for result in output:
            boxes = result.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                r = box.xyxy[0].astype(int)
                crop = input_image[r[1]:r[3], r[0]:r[2]]
                cv2.imwrite("crop.jpeg", crop)
                plugin.upload_file("crop.jpeg")
            
        # print detection stats
        TOPIC_TEMPLATE = 'rideshare'
        # for name, count in found.items():
        for i in name:
            # detection_stats += f'{class_id}[{count}] '
            detection_stats += f'{i}, {count}, {timestamp}'
            # plugin.publish(f'{class_id}', count, timestamp = timestamp)
            # plugin.publish(f'{i}', count, timestamp)
            # plugin.publish(f'{TOPIC_TEMPLATE}.{str(i)}', str(count), str(timestamp))
            
            plugin.publish('rideshare.log', str(count), timestamp=timestamp)
            
        print(detection_stats)
            

        # Return the modified input image
        if bool_found:
            return crop
        else:
            return input_image

    def main(self, sample, stage):
        """
        Performs inference using YoloV8 model and returns the output image frame with drawn detections.

        Returns:
            output_img frame: The output image with drawn detections.
        """
        
        # Get the model inputs
        ####################
        if stage==1:
            model_inputs = sample.data
        else:
            model_inputs = cv2.imread(sample)
            
        # Store the shape of the input for later use
        ####################
        # input_shape = model_inputs[0].shape
        # self.input_width = 640   #input_shape[2]
        # self.input_height = 640 #input_shape[3]
        
        # input_shape = model_inputs.shape
        self.input_width = model_inputs.shape[0]
        self.input_height = model_inputs.shape[1]
        

        # Preprocess the image data
        img_data = self.preprocess(model_inputs)

        # Run inference using the preprocessed image data
        # outputs = session.run(None, {model_inputs[0].name: img_data})

        #erased 0 from output
        with torch.no_grad():
            outputs = self.model.predict(model_inputs)
            
        # Perform post-processing on the outputs to obtain output image.
        # output_img = self.postprocess(self.img, outputs)
        # Check for stage (if 2, then final so don't publish anything)
        if (stage == 2):
            output_img = self.postprocess_2(sample, plugin, args, outputs) 
        else:
            output_img = self.postprocess_1(sample, plugin, args, outputs)

        # Return the resulting output image
        return output_img


if __name__ == '__main__':
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser(description = "Evaluate YoloV8 model")
    parser.add_argument('--model', type=str, default='yolov8.pt', help='Input YoloV8 model.')
    parser.add_argument('--model2', type=str, default='best_s2.pt', help='Input stage2 YoloV8 model.')
    parser.add_argument('-stream', type=str, action='store', default=str('bottom_camera'), help='ID of stream')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IoU threshold')
    parser.add_argument('-continuous', dest = 'continuous', action='store_true', default=False, help='Continuous run flag T/F')
    args = parser.parse_args()

    # Check the requirements and select the appropriate backend (CPU or GPU)
    self = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # self.use_cuda = torch.cuda.is_available()
    # if self.use_cuda:
    #     self.device = 'cuda'
    # else:
    #     self.device = 'cpu'

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        # print("running on the GPU")
    else:
        device = torch.device("cpu")
        # print("running on the CPU")

    # Create an instance of the Yolov8 class with the specified arguments
    detection = Yolov8(args.model, args.stream, args.conf_thres, args.iou_thres)
    
    # detection2 = Yolov8(args.model2, args.stream, 0.4, args.iou_thres)

    # take sample from plugin camera
    while True:
        with Plugin() as plugin:
            # with Camera(args.stream) as camera:
            with Camera(args.stream) as camera:
                #########
                sample = camera.snapshot()
                # sample = 'test2.jpeg'

            # Perform object detection - return crop - stage 1
            crop = detection.main(sample, 1)

            # Perform object detection - return for stage 2
            # output = detection2.main(crop, 2)
            
            if not args.continuous:
                exit(0)
        
