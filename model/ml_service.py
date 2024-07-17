import json
import os
import time

import numpy as np
import redis
import settings
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image

db = redis.Redis(
    host=settings.REDIS_IP,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB_ID
)

# TODO
# Load your ML model and assign to variable `model`
# See https://drive.google.com/file/d/1ADuBSE4z2ZVIdn66YDSwxKv-58U7WEOn/view?usp=sharing
# for more information about how to use this model.
model = ResNet50(include_top=True, weights="imagenet")


def predict(image_name):
    """
    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.

    Parameters
    ----------
    image_name : str
        Image filename.

    Returns
    -------
    class_name, pred_probability : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    class_name = None
    pred_probability = None

    
    # TODO

    path = settings.UPLOAD_FOLDER
    file = os.path.join(path, image_name)
    img = image.load_img(file, target_size=(224, 224))

    x = image.img_to_array(img)
    x_batch = np.expand_dims(a=x, axis=0)
    x_batch = preprocess_input(x=x_batch)

    prediction = model.predict(x_batch)
    decoded_prediction = decode_predictions(prediction, top=1)
    
    class_name = decoded_prediction[0][0][1]
    pred_probability = round (decoded_prediction[0][0][2], 4)
    
    class_name
    pred_probability

    return class_name, pred_probability


def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.
    """
    while True:

        message = db.brpop(str(settings.REDIS_QUEUE))
        decoded_message = message.decode('utf8')
        decoded_message_json = json.loads(decoded_message)

        class_name, pred_probability = predict(decoded_message_json['image_name'])

        JSON = {
            'prediction':class_name,
            'score': pred_probability
        }

        redis_id = decoded_message_json['id']
        db.set(redis_id, json.dumps(JSON))
        
        # Sleep for a bit
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()
