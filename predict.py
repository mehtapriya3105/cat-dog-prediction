import src.import_model as import_model
import torchvision
import torch 
from pathlib import Path
from torchvision import transforms
from PIL import Image
from typing import List

def load_model(model_path = "/model_store/model/model.pth"):
    model = import_model.create_model()
    model.load_state_dict(torch.load(model_path))
    return model

def predict_randomly(model: torchvision.models,
                     img_path: str,
                     class_names: List):
    # Load and preprocess the image
    img = Image.open(img_path)
    convert_tensor = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    img_tensor = convert_tensor(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    # Model evaluation
    model.eval()
    with torch.no_grad():
        y_pred_logits = model(img_tensor)

    # Get predictions
    y_preds = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)
    y_label = class_names[y_preds.item()]  # Assuming class_names is a list

    return y_label


def get_prediciton(img_path:str):
    class_names =  ['cats', 'dogs']
    img_path = Path(img_path)
    model = load_model("model_store/model2.pth")
    prediction_label = predict_randomly(model = model,
                    img_path = img_path,
                    class_names = class_names)
    
    return {"prediciton label " : prediction_label}
