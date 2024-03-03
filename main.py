from typing import Union

from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
from random import randint
import uuid
from aimodel_post import detectImage as detectImagePost

app = FastAPI()

IMAGEDIR='.'

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/process_image_post/")
async def process_image(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.png"
    contents = await file.read()

    # Save the file
    with open(f"{IMAGEDIR}/{file.filename}", "wb") as f:
        f.write(contents)

    print('File received and saved:', file.filename)

    detectImagePost(file.filename)

    processed_image_filename='detection_post.png'
    
    print('Image processed and saved:', processed_image_filename)

    # Return the processed image
    return FileResponse(f"{IMAGEDIR}/{processed_image_filename}", media_type="image/png", filename=processed_image_filename)