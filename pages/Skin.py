import streamlit as st
from PIL import Image
import torch.nn as nn
import timm
import torch
import torchmetrics
from torchmetrics import F1Score,Recall,Accuracy
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models
import lightning.pytorch as pl
import torchvision
from lightning.pytorch.loggers import WandbLogger
import shap
import matplotlib.pyplot as plt
import json 
from transformers import pipeline, set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM
text_model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
labels_path = '/Users/vikram/Python/Medical Diagnosis App/skin_labels.json'
from captum.attr import DeepLift , visualization

with open(labels_path) as json_data:
    idx_to_labels = json.load(json_data)



class FineTuneModel(pl.LightningModule):
    def __init__(self, model_name, num_classes, learning_rate, dropout_rate,beta1,beta2,eps):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.dropout_rate = dropout_rate
        self.model = timm.create_model(self.model_name, pretrained=True,num_classes=self.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.f1 = F1Score(task='multiclass', num_classes=self.num_classes)
        self.recall = Recall(task='multiclass', num_classes=self.num_classes)
        self.accuracy = Accuracy(task='multiclass', num_classes=self.num_classes)
        
        #for param in self.model.parameters():
            #param.requires_grad = True
        #self.model.classifier= nn.Sequential(nn.Dropout(p=self.dropout_rate),nn.Linear(self.model.classifier.in_features, self.num_classes))
        #self.model.classifier.requires_grad = True
            

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        acc =  self.accuracy(y_hat.argmax(dim=1),y)
        f1 = self.f1(y_hat.argmax(dim=1),y)
        recall = self.recall(y_hat.argmax(dim=1),y)
        self.log('train_loss', loss,on_step=False,on_epoch=True)
        self.log('train_acc', acc,on_step=False,on_epoch = True)
        self.log('train_f1',f1,on_step=False,on_epoch=True)
        self.log('train_recall',recall,on_step=False,on_epoch=True)
        return loss
            
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        acc =  self.accuracy(y_hat.argmax(dim=1),y)
        f1 = self.f1(y_hat.argmax(dim=1),y)
        recall = self.recall(y_hat.argmax(dim=1),y)
        self.log('val_loss', loss,on_step=False,on_epoch=True)
        self.log('val_acc', acc,on_step=False,on_epoch=True)
        self.log('val_f1',f1,on_step=False,on_epoch=True)
        self.log('val_recall',recall,on_step=False,on_epoch=True)
                
            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,betas=(self.beta1,self.beta2),eps=self.eps)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    
    #load model
   
    
    
    

st.markdown("<h1 style='text-align: center; '>Skin Leision Diagnosis</h1>",unsafe_allow_html=True)




# Display a file uploader widget for the user to upload an image

uploaded_file = st.file_uploader("Choose an Skin image file", type=["jpg", "jpeg", "png"])

# Load the uploaded image, or display emojis if no file was uploaded
with st.container():
    if uploaded_file is not None:
        
        image = Image.open(uploaded_file)
        st.image(image, caption='Diagnosis', use_column_width=True)
        model = timm.create_model(model_name='efficientnet_b0', pretrained=True,num_classes=4)
        data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        transform = timm.data.create_transform(**data_cfg)
        model_transforms = torchvision.transforms.Compose([transform])
        transformed_image = model_transforms(image)
        brain_model = torch.load('models/timm_skin_model.pth')
        
        brain_model.eval()
        with torch.inference_mode():
            with st.progress(100):
                
                #class_names = ['Glinomia','Meningomia','notumar','pituary']
                prediction = torch.nn.functional.softmax(brain_model(transformed_image.unsqueeze(dim=0))[0], dim=0)
                prediction_score, pred_label_idx = torch.topk(prediction, 1)
                pred_label_idx.squeeze_()
                predicted_label = idx_to_labels[str(pred_label_idx.item())]
                st.write( f'Predicted Label: {predicted_label}')
                if st.button('Know More'):
                    generator = pipeline("text-generation",model=text_model,tokenizer=tokenizer)
                    input_text = f"Patient has {predicted_label} and is advised to take the following medicines:"
                    with st.spinner('Generating Text'):
                        generator(input_text, max_length=300, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)
                    st.markdown(generator(input_text, max_length=300, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)[0]['generated_text'])
                
                    
            
            
            
            

    
        
        
    
        
        
    else:
        st.success("Please upload an image file 🧠")
        
        