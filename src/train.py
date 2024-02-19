import src.import_model as import_model 
import src.import_traintestdata as import_traintestdata 
import torch 
import torchvision 
#get traindataloader, testdataloader , classnames and model from their respective files.
#This model is created to avoid recurrent import from other files rudcing on time limit

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def get_data():
    train_dataloader, test_dataloader , class_names = import_traintestdata.get_data("all")
    model = import_model.create_model()
    return model, train_dataloader, test_dataloader , class_names

#This function is used to train the model.
#This function takes all the input from the train fuction 
def train_step(model : torchvision.models,
                 dataloader: torch.utils.data.DataLoader,
                 loss_fn: torch.nn.Module,
                 optimizer :torch.optim.Optimizer):
    train_loss, train_acc = 0, 0
    #model = import_model.create_model()
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

#This function is used to test the model.
#This function takes all the input from the train fuction
def test_step(model : torchvision.models,
                 dataloader: torch.utils.data.DataLoader,
                 loss_fn: torch.nn.Module,
                 optimizer :torch.optim.Optimizer):
    test_loss = 0

    model.eval() 
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

#This model saves the model generated to the specific user defined path
def save_model(model:torchvision.models, path = '/model_store/model.pth'):
    model_path = path
    torch.save(model.state_dict(), model_path)
    return model_path

#This is the main function which is used to train, test and then save the model generated.
def train(epoches:int=5):
    
    results = {
        "train_loss" : [],
        "test_loss" : [],
        "train_acc" : [],
        "test_acc" : []
    }

    #Get the model , train datalaoder, testdataloader, and class names from the get_data function 
    model, train_dataloader, test_dataloader , class_names = get_data()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = model.parameters(),lr = 0.001)
    
    #Call the train and test functions respectivly 
    for epoch in range(epoches):
        train_loss,  train_acc = train_step(model = model,
                                dataloader = train_dataloader,
                                loss_fn = loss_fn,
                                optimizer=optimizer,
                                )
        test_loss, test_acc  = test_step(model = model,
                                dataloader = test_dataloader,
                                loss_fn = loss_fn,
                                optimizer=optimizer,
                                )
    #     print(
    #       f"Epoch: {epoch+1} | "
    #       f"train_loss: {train_loss:.4f} | "
    #       f"train_acc: {train_acc:.4f} | "
    #       f"test_loss: {test_loss:.4f} | "
    #       f"test_acc: {test_acc:.4f}"
    #   )

      # Update results dictionary
        results["train_loss"].append(train_loss)   
        results["test_loss"].append(test_loss)
        results["train_acc"].append(train_acc)
        results["test_acc"].append(test_acc)
    

    model_path = save_model(model)

    return print(f"model is saved at : {model_path}") 



