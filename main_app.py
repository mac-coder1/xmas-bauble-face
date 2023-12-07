import streamlit as st
from fastai.vision.all import *
from io import BytesIO

st.title("🎄🎅🏻 Personalised Christmas Decoration Generator 🎄🎅🏻")

st.write("Hello and Merry Christmas! 🎅🏻🎄🎁")
st.write(
    "This app will help you create a personalised Christmas decoration with your face on it!"
)
st.write(
    "Just follow the instructions below and you should have a weird and wonderful decoration in no time!"
)


# create the pretrained model
# define label func
def label_func(fn):
    return path / "labels_real_mask" / f"{fn.name[:-3]}png"


model = load_learner("resnet18_model.pkl")

st.subheader("Upload your favourite Christmas bauble!")
st.markdown('There are some good ones [here](https://drive.google.com/drive/u/1/folders/1kjokG2hKxD35f8Iy17tVz2wc9MzgGb63)!')
# upload a picture of you favorite bauble
bauble_file = st.file_uploader("Upload a bauble image...", type="jpeg")

st.subheader("Now upload a picture with you face in it!")
st.write(
    "NB, please position your face in the middle of the image, and make sure it is not too small."
)
# image upload
picture_choice = st.radio(
    "Choose an option:", ("Take a picture", "Upload an image"), horizontal=True
)

uploaded_file = None
if picture_choice == "Upload an image":
    uploaded_file = st.file_uploader("Choose an image...", type="jpeg")
elif picture_choice == "Take a picture":
    uploaded_file = st.camera_input("Take a picture of your face")


if (uploaded_file is not None) and (bauble_file is not None):
    st.subheader(
        "Adjust the images so that the face is the size and placement you want it to be on the Xmas decoration!"
    )
    img = PILImage.create(uploaded_file)
    bauble_img = PILImage.create(bauble_file)

    col1, col2 = st.columns(2)

    with col1:
        slider_value = st.slider("Crop Face Image", 0, 2000, 500)
        resize_transform = CropPad(slider_value)
        img = resize_transform(img)
        st.image(img)

    with col2:
        slider_value = st.slider("Crop Bauble Image", 0, 2000, 1000)
        resize_transform = CropPad(slider_value)
        bauble_img = resize_transform(bauble_img)
        st.image(bauble_img)

    hair = st.toggle("Remove hair? 🪮", False)
    skin = st.toggle("Facial Features only? 🫥", False)

    # these values correspond to the codes defined for each different part of the face in the segmentation model
    values_to_keep = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13]

    if not hair:
        values_to_keep.append(5)
    if not skin:
        values_to_keep.append(4)

    values_to_keep = torch.tensor(values_to_keep)

    # squish the cropped image to the correct size for segmentation
    resize_transform = Resize(256, method=ResizeMethod.Squish)
    img = resize_transform(img)
    bauble_img = resize_transform(bauble_img)

    img_tensor = tensor(img)
    bauble_img_tensor = tensor(bauble_img)

    # find the segmentation mask
    prediction = model.predict(img)
    prediction_mask = prediction[0]

    # use the mask to crop the image
    # convert the mask to a tensor
    prediction_tensor = prediction_mask.unsqueeze(-1)
    prediction_tensor = prediction_tensor.repeat(1, 1, 3)

    face_tensor = torch.isin(prediction_tensor, values_to_keep)

    not_face_tensor = torch.logical_not(face_tensor)

    cropped_tensor = img_tensor * face_tensor

    cropped_tensor = cropped_tensor + bauble_img_tensor * not_face_tensor

    cropped_image = PILImage.create(cropped_tensor)

    st.image(cropped_image)

    # Convert the PILImage to Bytes
    buffered = BytesIO()
    cropped_image.save(buffered, format="PNG")
    img_byte = buffered.getvalue()

    # Implement the download button
    st.download_button(
        label="Download your Christmas Decoration",
        data=img_byte,
        file_name="christmas_decoration.png",
        mime="image/png",
    )

    st.markdown("If you enjoyed this app, please share it with your friends!")
    st.markdown(
        "This app was made by Cormac O'Malley, you can find me on [Linkedin](https://www.linkedin.com/in/cormac-o-malley-978496183/) or [Github](https://github.com/mac-coder1?tab=repositories)!"
    )
    st.stop()
