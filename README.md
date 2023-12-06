# **Custom Christmas Decoration App**

Here I train the face segmentation model used for this [streamlit app](https://xmas-bauble-face.streamlit.app/). The app lets the user put their face (with or without hair) onto their favorite Christmas decoration.

The main difficulty was actually processing the RGB facemasks into 2-d tensors that contained integers depending one which part of the fact that pixel was a part of.

I train the model in face_segmentation.ipynb and export it as resnet18_model.pkl. The streamlit app is then deployed from 🎄⛄️🎄.py.

Please reach out to me with any questions you might have, or if you want to collaborate!

Some useful links:
- [Link](https://store.mut1ny.com/product/face-head-segmentation-dataset-community-edition?v=6048de06ffbd) to the dataset.
- Useful notes on [masks](https://docs.fast.ai/vision.core.html#PILImage.create).
- [Resize](https://docs.fast.ai/vision.augment.html#resize) documentation.