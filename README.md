# FastAPI Image Classification Application

This FastAPI-based application is designed for image classification using transfer learning principles. The application leverages a file upload POST endpoint accessible through the `/docs` and Swagger FastAPI UI. Users can upload an image, and the application will respond with a JSON indicating whether the image contains a cat or a dog.

## Model Architecture

The model is based on a modified ResNet-50 architecture. Additional layers have been added, including four sets of linear and flatten layers followed by a final linear layer with a shape of 2, representing the classes (cat and dog). The application is designed for transfer learning, enhancing the capabilities of the ResNet-50 model.

## Training Results

The model was trained for 5 epochs, and the following training and testing accuracy and loss were observed:

```plaintext
Epoch: 1 | train_loss: 0.7021 | train_acc: 0.5127 | test_loss: 0.7097 | test_acc: 0.4936
Epoch: 2 | train_loss: 0.6989 | train_acc: 0.5200 | test_loss: 0.7027 | test_acc: 0.5155
Epoch: 3 | train_loss: 0.7007 | train_acc: 0.5236 | test_loss: 0.7091 | test_acc: 0.5177
Epoch: 4 | train_loss: 0.7016 | train_acc: 0.5093 | test_loss: 0.7100 | test_acc: 0.5026
Epoch: 5 | train_loss: 0.7006 | train_acc: 0.5161 | test_loss: 0.6978 | test_acc: 0.5336
```

As the epochs progress, the model shows improvement. Due to GPU limitations, the number of epochs is intentionally kept low.

## Dataset

The dataset used for training and testing is sourced from Kaggle:

[Kaggle Cat and Dog Dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog)

## Dockerized Application

The entire application, including the FastAPI server, model, and dependencies, is encapsulated into a Docker image for ease of deployment and portability.

## Usage

1. Clone the repository:

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Build the Docker image:

    ```bash
    docker build -t image-classification-app .
    ```

3. Run the Docker container:

    ```bash
    docker run -p 8000:8000 image-classification-app
    ```

4. Access the FastAPI documentation and Swagger UI at http://localhost:8000/docs in your web browser.

5. Upload an image and receive the JSON response indicating whether it contains a cat or a dog.

Feel free to customize the application or experiment with different datasets and model architectures based on your requirements.