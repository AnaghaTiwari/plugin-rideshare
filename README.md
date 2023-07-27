## Rideshare Vehicle Detection
### [Sage App](https://portal.sagecontinuum.org/apps/app/anagha/rideshare-detection?tab=science)

This plugin detects rideshare vehicle stickers given an input camera (video) stream using 2 YoloV8 custom-trained models. The cropped rideshare sticker images are then published to the ECR (see Sage App).

To use this app on the terminal:
```bash
python3 app.py
```
To use this app via sage_data_client library:
```bash
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
