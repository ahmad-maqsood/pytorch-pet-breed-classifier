import streamlit as st
import torch
import torchvision
from torch import nn
from PIL import Image

CONFIDENCE_THRESHOLD = 0.5   # tune this after testing a few odd images yourself

class_names = ['Abyssinian', 'American Bulldog', 'American Pit Bull Terrier',
              'Basset Hound', 'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer',
              'British Shorthair', 'Chihuahua', 'Egyptian Mau',
              'English Cocker Spaniel', 'English Setter', 'German Shorthaired',
              'Great Pyrenees', 'Havanese', 'Japanese Chin', 'Keeshond',
              'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'Newfoundland',
              'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue',
              'Saint Bernard', 'Samoyed', 'Scottish Terrier', 'Shiba Inu',
              'Siamese', 'Sphynx', 'Staffordshire Bull Terrier',
              'Wheaten Terrier', 'Yorkshire Terrier']

@st.cache_resource
def load_model():
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    transform = weights.transforms()
    model = torchvision.models.resnet50(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(in_features=2048, out_features=37)
    model.load_state_dict(torch.load("resnet_50_unfrozen_layer4_and_manual_transform.pth", map_location="cpu"))
    model.eval()
    return model, transform

model, transform = load_model()

st.title("Pet Breed Classifier")
st.write("Upload a photo of a cat or dog and the model will guess its breed.")

with st.expander("See all 37 breeds this model recognizes"):
    st.write(", ".join(class_names))

uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image")

    with torch.inference_mode():
        img_tensor = transform(img).unsqueeze(0)
        probs = torch.softmax(model(img_tensor), dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning(
            f"I'm not very confident about this one ({confidence*100:.1f}%). "
            f"This model only recognizes 37 specific cat and dog breeds, "
            f"so the image may be of something else, or a breed outside that list."
        )
    else:
        st.subheader(f"Prediction: {class_names[pred_idx]}")
        st.write(f"Confidence: {confidence*100:.1f}%")

    st.write("Top 3 guesses:")
    top3_probs, top3_idx = torch.topk(probs, 3)
    for p, i in zip(top3_probs, top3_idx):
        st.write(f"- {class_names[i]}: {p.item()*100:.1f}%")