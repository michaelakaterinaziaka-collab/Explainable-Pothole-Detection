# Pothole Detection with Model Interpretability using Grad-CAM

An end-to-end Computer Vision project focused on detecting road potholes and providing visual explanations for the model's predictions. The project includes dataset fine-tuning, interpretability integration, and a production-ready web application deployment.

## Live Demo
The interactive web application is hosted on Hugging Face Spaces. You can test the model using your own images or the provided examples here:
[View Live Demo on Hugging Face Spaces](https://michaelakaterina-explainable-pothole-detection.hf.space/?logs=container&__theme=system&deep_link=7l_Dj3VNuDQ)

## Technical Architecture
* **Core Framework:** PyTorch and Torchvision for model implementation and inference.
* **Model Backbone:** EfficientNet-B0, fine-tuned on a custom road damage dataset.
* **Explainable AI (XAI):** Grad-CAM (Gradient-weighted Class Activation Mapping) utilized to generate visual heatmaps, highlighting the exact image regions that influenced the classification decision.
* **Web Interface:** Gradio for building an intuitive front-end interface that processes inputs and displays results in real-time.

## Project Workflow
1. **Fine-Tuning:** Leveraging a pre-trained EfficientNet backbone to extract features and classify road anomalies with high accuracy.
2. **Explainability Layer:** Implementing a custom Grad-CAM hook on the final convolutional layer to extract gradients and map feature importance back onto the original input image.
3. **Deployment:** Packaging the inference pipeline, model weights, and the Gradio UI into an autonomous Hugging Face Space instance.

## Repository Contents
* `app.py`: The main execution script containing the interface definition and the inference pipeline.
* `requirements.txt`: The complete list of python package dependencies for reproducing the environment.
* `examples/`: Sample road images used to facilitate immediate testing within the web application.

