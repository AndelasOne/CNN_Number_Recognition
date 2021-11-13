import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from io import BytesIO
from streamlit_drawable_canvas import st_canvas
import matplotlib.image
from model import CNN
import time
from datetime import datetime
import pandas as pd

## TODO:
# * Datensatz von SK Learn noch als Dataframe darstellen
# * Eigenen Datensatz der erstellt wird als Dataframe
# * Learning Costs Neuronales Netz darstellen


def main():
    st.title("Number Recognition Demo")

    # INTRODUCTION TO DATASET
    number_data = load_digits()  # load image data

    idx = st.number_input(
        label="Select an image between 0 and 1796!",
        min_value=0,
        max_value=len(number_data.images) - 1,
        value=50,
    )
    img = number_data.images[idx]
    label = number_data.target[idx]

    draw_number(img, label, 500)
    st.subheader("Image Representation as array")
    st.write(img)

    model_params = load_model("./training_model/model.npy")
    canvas_result = display_canvas()

    if st.sidebar.button("Convert Imgage"):
        converted_canvas_img = convert_Canvas_to_Img(canvas_result)

        # model_params = load_model("./training_model/model.npy")
        predict_number(model_params, converted_canvas_img)

    input = st.sidebar.text_input("Save Image with Label")

    if st.sidebar.button("Save Image") and input:

        converted_canvas_img = convert_Canvas_to_Img(canvas_result)
        timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
        save_Img(
            converted_canvas_img, f"./data/img/img_{timestamp}_{input}.png", int(input)
        )


def draw_number(img: np.ndarray, label: str, imgsize: int) -> None:
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray")
    st.subheader(f"First Number: {label}")
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf, width=imgsize, use_column_width=imgsize)


def load_model(path: str) -> dict:
    loaded_parameter_array = np.load(path, allow_pickle=True)
    loaded_parameters = dict()

    for key, key_d in loaded_parameter_array.item().items():
        loaded_parameters[key] = key_d
    return loaded_parameters


def predict_number(params: dict, img: np.ndarray):
    cnn = CNN()
    sc = cnn.fit_and_transform()

    training_img = img.reshape((64, 1)).T
    # training_img = sc.fit_transform(training_img)
    training_img = training_img.T
    predicted_digit = cnn.predict_L_layer(training_img, params)
    st.subheader("Number Prediction")
    st.write("Predicted digit is : " + str(np.squeeze(predicted_digit)[()]))


def display_canvas():
    # Specify canvas parameters in application
    stroke_width = st.sidebar.slider("Stroke width: ", 40, 50, 45)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#ffffff")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#000000")
    canvas_size = 400
    drawing_mode = "freedraw"
    # realtime_update = st.sidebar.checkbox("Update in realtime", True)

    # Create a canvas component
    st.subheader("Draw Number")
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        update_streamlit=True,
        height=canvas_size,
        width=canvas_size,
        drawing_mode=drawing_mode,
        key="canvas",
    )
    return canvas_result


def convert_Canvas_to_Img(canvas):
    # convert rgba img to gray-scale img
    gray_img = np.dot(canvas.image_data[..., :3], [0.299, 0.587, 0.114])

    # scale down img to 8x8 and normalise the color values
    recombined_img = recombine_img(gray_img, (8, 8)) * (16 / 255)
    draw_number(recombined_img, "converted image from canvas", 500)

    return recombined_img


def recombine_img(img: np.ndarray, shape: tuple):
    sh = shape[0], img.shape[0] // shape[0], shape[1], img.shape[1] // shape[1]
    return img.reshape(sh).mean(-1).mean(1)


def save_img_to_dataset(img: np.ndarray):
    np.save("./data/number.npy", img)
    pass


def init_dataset():
    data = dict()
    loaded_img = plt.imread("./data/img/img_20211023-132410_2.png")
    loaded_img_gray = np.dot(loaded_img.image_data[..., :3], [0.299, 0.587, 0.114])
    loaded_img_transformed = recombine_img(loaded_img_gray, (8, 8)) * 16
    data["label"] = [2]
    data["images"] = [loaded_img_transformed]
    np.save("./data/dataset.npy", data)


def load_dataset(path: str):
    data = np.load(path, allow_pickle=True)
    loaded_data = dict()
    for key, key_d in data.item().items():
        loaded_data[key] = key_d
    return loaded_data


def append_dataset(label: int, img: np.ndarray):
    loaded_data = load_dataset("./data/dataset.npy")
    labels = loaded_data["label"]
    images = loaded_data["images"]
    # append data
    labels.append(label)
    images.append(img)

    # override new data
    loaded_data["label"] = labels
    loaded_data["images"] = images
    np.save("./data/dataset.npy", loaded_data)


def save_Img(img: np.ndarray, filename: str, label: int):
    matplotlib.image.imsave(filename, img)
    append_dataset(label, img)


if __name__ == "__main__":
    main()