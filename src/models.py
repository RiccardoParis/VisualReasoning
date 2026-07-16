import torch
import torch.nn as nn
import timm
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

class MultimodalCoT(nn.Module):
    def __init__(self, vision_encoder_name='vit_large_patch32_384', llm_model_name='t5-base', hidden_dim=768):
        # Nota: hidden_dim è 512 per t5-small. Se usi t5-base, cambialo a 768.
        super().__init__()
        
        # 1. Vision Encoder (Congelato)
        self.vision_extractor = timm.create_model(vision_encoder_name, pretrained=True)
        for param in self.vision_extractor.parameters():
            param.requires_grad = False
            
        # 2. Projection Layer (Allineamento dimensioni: ViT 1024 -> T5 hidden_dim)
        self.projector = nn.Linear(1024, hidden_dim)
        
        # 3. Language Model (T5)
        self.llm = T5ForConditionalGeneration.from_pretrained(llm_model_name)

        # 4. Multi-Head Cross-Attention (Il VERO ragionamento visivo)
        # batch_first=True è fondamentale per far combaciare i tensori con T5
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        # A. Estrazione Feature Visive (N patch per ogni immagine)
        with torch.no_grad():
            H_vision = self.vision_extractor.forward_features(pixel_values) 
        
        # Proiettiamo la vista alla stessa "lingua" (dimensione) del testo
        H_vision = self.projector(H_vision)
        
        # B. Estrazione Significato della Domanda (Testo)
        text_encoder_output = self.llm.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        H_language = text_encoder_output.last_hidden_state
        
        # C. RAGIONAMENTO VISIVO (Cross-Attention)
        # Il Testo "interroga" l'Immagine.
        # Query = Testo, Key/Value = Immagine
        attn_output, _ = self.cross_attention(
            query=H_language,
            key=H_vision,
            value=H_vision
        )
        
        # Uniamo le informazioni visive trovate con la domanda originale (Residual Connection)
        H_fused = self.layer_norm(H_language + attn_output)
        
        # D. Generazione della Risposta
        # Invece di far ricalcolare tutto a T5, iniettiamo direttamente le nostre feature fuse nel suo Decoder!
        outputs = self.llm(
            encoder_outputs=(H_fused,),
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs
    
    def generate(self, input_ids, attention_mask, pixel_values, max_new_tokens=100):
        # A. Estrazione Feature Visive
        with torch.no_grad():
            H_vision = self.vision_extractor.forward_features(pixel_values) 
        H_vision = self.projector(H_vision)
        
        # B. Estrazione Significato della Domanda
        text_encoder_output = self.llm.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        H_language = text_encoder_output.last_hidden_state
        
        # C. RAGIONAMENTO VISIVO (Cross-Attention)
        attn_output, attn_weights = self.cross_attention(
            query=H_language,
            key=H_vision,
            value=H_vision,
            average_attn_weights=True # Calcola la media tra le varie "teste" (heads)
        )
        H_fused = self.layer_norm(H_language + attn_output)
        
        # Impacchettiamo le feature fuse in un oggetto comprensibile a T5
        encoder_outputs = BaseModelOutput(last_hidden_state=H_fused)
        
        # D. Generazione Finale
        outputs = self.llm.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask, # Passiamo la maschera originale del testo
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.2, # <--- Penalizza la ripetizione di sillabe
            do_sample=False
        )
        
        return outputs, attn_weights