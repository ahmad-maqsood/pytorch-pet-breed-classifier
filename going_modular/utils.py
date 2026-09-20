
import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               model_name: str,
               target_dir: str):

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(('.pth', '.pt')), 'model_name should end with "pth" or "pt"'
    model_save_path = target_dir_path / model_name

    print('Saving the model..')
    torch.save(obj=model.state_dict(), f=model_save_path)
    print('Model has been saved successfully.')

def load_model(model: torch.nn.Module,
               model_path: str,
               device:str = 'cpu'):  
    state_dict = torch.load(f=model_path, map_location=device)
    model.load_state_dict(state_dict=state_dict)
    model.to(device)
    model.eval()
    return model
