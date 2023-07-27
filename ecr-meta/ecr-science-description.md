# Science

A semi-real-time rideshare vehicle tracking application based on a group of convolutional neural network models was utilized to detect rideshare vehicles in street intersections. Due to the small size and lighting conditions that make it challenging to detect rideshare stickers, we developed a two-stage system to 1) zoom in to possible stickers and 2) confirm the presence of rideshare stickers. We trained two YOLOv8 models (Stage1 and Stage2) to track possible rideshare vehicles. YOLOv8 is a recent SOTA object detection model based on deep convolutional neural networks, and has a much higher accuracy than previous YOLO models. We hope that this rideshare plugin can be one of the first rideshare detection algorithms, which will soon be installed in city Waggle nodes available for infrastructure and city officials, rideshare companies, and airport officials to use. 

# AI at the Edge

Both YoloV8 models were finetuned (custom-trained) on a rideshare sticker datasets. Specifically, Model1 was trained on an augmented dataset, while Model2 was trained using Model1's sticker predictions. While running, Model1 takes in a steady video stream from a given bottom camera (installed at the Sage node) and attempts to detect rideshare stickers given the zoomed-out, original frame. The resulting sticker prediction is passed on to Model2, which confirms whether the prediction is actually a rideshare sticker. The graphic below provides a visual explanation of the 2-Stage process. The model detections happen in-situ, while the cropped rideshare stickers, timeframe, and count of rideshare vehicles is published to the data repo for public use.

![dest](https://github.com/AnaghaTiwari/plugin-rideshare/assets/76963992/bcc1b1dd-402b-40c4-8206-50095b426f82)


# Inference for Sage Nodes

Anyone can query the plugin output from the Sage data repository, via the `sage_data_client` python library: 
```
# install and import library
import sage_data_client

# query and load data into pandas data frame
df = sage_data_client.query(
    start="-1h",
    filter={
        "plugin": "registry.sagecontinuum.org/anagha/rideshare-detection:0.1.8"
    }
)

# print results in data frame
print(df)
# print filter names
print(df.name.unique())
# print number of rideshare vehicles detected
print(len(df))
```
For more information, please see [Access and use data documentation](https://docs.sagecontinuum.org/docs/tutorials/accessing-data) and [sage_data_client](https://pypi.org/project/sage-data-client/).
