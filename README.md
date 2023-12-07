# **[Custom Christmas Decoration App](https://xmas-bauble-face.streamlit.app/)**

This repo creates the [streamlit app](https://xmas-bauble-face.streamlit.app/) which lets the user put their face (with or without hair) onto their favorite Christmas decoration 🎄!

To automatically crop people's facial features onto the decoration, I needed an ML segmentation model in order to know which pixels correspond to their different facial features. I train the model in face_segmentation.ipynb and export it as resnet18_model.pkl. The app uses the brilliant [Fastai](https://docs.fast.ai/) library to construct the dataloader and learner. I highly recommend checking out the [fastai course](https://course.fast.ai/) to learn how to do it. I made this piece to celebrate finishing the course! The library is so powerful because it focusses on fine tuning pre-trained models to do your specific tasks. This leverages the large benefits of transfer learning to speed things up and improve accuracy!

The streamlit app is then deployed from 🎄⛄️🎄.py.

Learning points:
- The main difficulty was actually processing the RGB facemasks into 2-d tensors that contained integers depending one which part of the fact that pixel was a part of.

Please reach out to me with any questions you might have, or if you want to collaborate!!!

Some useful links:
- [Link](https://store.mut1ny.com/product/face-head-segmentation-dataset-community-edition?v=6048de06ffbd) to the dataset.
- Useful notes on [masks](https://docs.fast.ai/vision.core.html#PILImage.create).
- [Resize](https://docs.fast.ai/vision.augment.html#resize) documentation.
