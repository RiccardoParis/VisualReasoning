import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

def program_to_reasoning(program):
    """Traduce il programma funzionale di CLEVR in una Chain of Thought narrativa."""
    steps = []
    for p in program:
        func = p['function']
        # Gestiamo gli input (es. "red", "cube", "left") in modo sicuro
        inputs = p.get('value_inputs', [])
        val = inputs[0] if len(inputs) > 0 else ""
        
        if func == 'scene': steps.append("Look at the scene.")
        elif 'filter_color' in func: steps.append(f"Focus on the {val} objects.")
        elif 'filter_shape' in func: steps.append(f"Select the {val}s.")
        elif 'filter_size' in func: steps.append(f"Consider the {val} ones.")
        elif 'filter_material' in func: steps.append(f"Identify the {val} objects.")
        elif func == 'relate': steps.append(f"Look at what is {val} the previous object.")
        elif func == 'unique': steps.append("Isolate the specific object.")
        elif func == 'intersect': steps.append("Find the common objects between these groups.")
        elif func == 'union': steps.append("Combine these groups.")
        elif func == 'count': steps.append("Count them.")
        elif func == 'exist': steps.append("Check if any such object exists.")
        elif func.startswith('query_'): steps.append(f"Identify its {func.split('_')[1]}.")
        elif func.startswith('equal_'): steps.append(f"Check if their {func.split('_')[1]} is the same.")
        else: steps.append(f"Process the {func.replace('_', ' ')}.") # Fallback di sicurezza

    # Assembliamo la Chain of Thought testuale
    return "Let's think step by step. " + " ".join(steps) + " The final answer is:"

class CLEVRDataset(Dataset):
    # Aggiungiamo il parametro "stage" (default 'stage1')
    def __init__(self, index_path, img_dir, tokenizer, transform=None, stage='stage1'):
        with open(index_path, 'r') as f:
            self.data = json.load(f)['questions']
            
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.stage = stage # Salviamo la fase corrente

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_name = item['image_filename']
        img_path = os.path.join(self.img_dir, img_name)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"⚠️ Immagine mancante: {img_path}")
            
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            pixel_values = self.transform(img)
        else:
            pixel_values = img 
            
        # Estraiamo il ragionamento e la risposta vera
        reasoning = program_to_reasoning(item.get('program', []))
        answer = item.get('answer', '') 

        # ----------------------------------------------------
        # 🔀 BIVIO LOGICO: FASE 1 vs FASE 2
        # ----------------------------------------------------
        if self.stage == 'stage1':
            # FASE 1: Il modello legge la domanda e deve generare TUTTO (CoT + Risposta)
            input_text = f"Question: {item['question']} Answer:"
            target_text = f"{reasoning} {answer}"
            
        elif self.stage == 'stage2':
            # FASE 2: Il modello legge Domanda + Rationale, e deve generare SOLO la Risposta Finale
            input_text = f"Question: {item['question']} Rationale: {reasoning} Answer:"
            target_text = f"{answer}"
            
        else:
            raise ValueError("Lo stage deve essere 'stage1' o 'stage2'")
            
        return {
            'pixel_values': pixel_values,
            'input_text': input_text,
            'target_text': target_text,
            'raw_item': item 
        }