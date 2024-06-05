
import uvicorn
import fastapi
import tensorflow as tf


from fastapi import FastAPI, File, UploadFile
from io import BytesIO
import pickle

app = FastAPI()
pickle_in = open("model.pkl","rb")
model=pickle.load(pickle_in)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/files/")
async def create_file(file: UploadFile = File(...)):
  
    return {"file_size": len(file)}
    


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):

    # read image
    img = tf.image.decode_jpeg(file, channels=3)
    # Resize the image to the specified size
    img = tf.image.resize(img, [224,224])
    # Normalize the image to the range [0, 1]
    img = img / 255.0
      
    # Add batch dimension as model expects a batch of images
    img = tf.expand_dims(img, axis=0)
    # transform and prediction 
    prediction = model.predict(img)

    return prediction