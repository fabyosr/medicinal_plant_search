import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from PIL import Image
import os
import gdown
import shutil
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. CONFIGURAÇÕES E CONSTANTES
# ==========================================
class Config:
    SAVE_DIR = './saved_model_components'
    EMBEDDINGS_DIR = './saved_embeddings_and_metadata'
    PLANT_IMAGES_DIR = os.path.join(EMBEDDINGS_DIR, 'plant_images')
    
    MODEL_COMPONENTS_GD_ID = "1vPnnFsO_IsDs_I4oE73KGD5Y_BpM0dac"
    EMBEDDINGS_GD_ID = "10NFY8TiwMwlBnnfBwZ2hVx98M3kfQGg5"
    
    TOKENIZER_MODEL = 'neuralmind/bert-base-portuguese-cased'
    EMBED_DIM = 512
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. UI & ESTILIZAÇÃO CSS (O Visual Deslumbrante)
# ==========================================
def apply_custom_css():
    """Injeta CSS customizado para estilizar os cards, botões e tipografia."""
    st.markdown("""
        <style>
        /* Fundo e tipografia principal */
        .stApp {
            background-color: #F8FAF8;
        }
        h1, h2, h3 {
            color: #2E4E3F;
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 600;
        }
        /* Cards de resultado */
        .result-card {
            background: white;
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            margin-bottom: 20px;
            height: 100%;
        }
        .result-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(46, 78, 63, 0.15);
        }
        /* Barra de similaridade gradiente */
        .similarity-container {
            width: 100%;
            background-color: #e0e0e0;
            border-radius: 8px;
            margin: 10px 0;
            height: 8px;
            overflow: hidden;
        }
        .description-text {
            font-size: 0.9rem;
            color: #555;
            margin-top: 10px;
            line-height: 1.4;
        }
        /* Botões customizados */
        div.stButton > button:first-child {
            background-color: #3E8E41;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 24px;
            font-weight: bold;
            transition: all 0.3s;
        }
        div.stButton > button:first-child:hover {
            background-color: #2E4E3F;
            box-shadow: 0 4px 12px rgba(46,78,63,0.3);
        }
        </style>
    """, unsafe_allow_html=True)

def render_similarity_bar(score):
    """Gera uma barra HTML cujo preenchimento e cor dependem do score (0 a 1)."""
    percentage = max(0, min(100, int(score * 100)))
    # Gradiente: verde escuro para alta similaridade, mudando para verde claro/amarelado
    color = f"hsl({120 * score}, 60%, 45%)" 
    html = f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
            <span style="font-weight: bold; color: {color}; font-size: 1.1em;">{score:.2f}</span>
            <span style="font-size: 0.8em; color: #888;">Similaridade</span>
        </div>
        <div class="similarity-container">
            <div style="width: {percentage}%; background-color: {color}; height: 100%; border-radius: 8px;"></div>
        </div>
    """
    return html

# ==========================================
# 3. GERENCIAMENTO DE DADOS (Downloads)
# ==========================================
class DataManager:
    @staticmethod
    def _flatten_if_needed(target_dir):
        """Corrige o problema de subpastas extras criadas pelo gdown."""
        items = os.listdir(target_dir)
        subdirs = [d for d in items if os.path.isdir(os.path.join(target_dir, d))]
        if len(subdirs) == 1 and len(items) <= 10:
            subfolder = os.path.join(target_dir, subdirs[0])
            for item in os.listdir(subfolder):
                shutil.move(os.path.join(subfolder, item), os.path.join(target_dir, item))
            os.rmdir(subfolder)

    @classmethod
    def download_assets(cls):
        os.makedirs(Config.SAVE_DIR, exist_ok=True)
        os.makedirs(Config.EMBEDDINGS_DIR, exist_ok=True)

        try:
            # Baixa modelo se não existir
            if not os.path.exists(os.path.join(Config.SAVE_DIR, 'dual_encoder_model_weights.pth')):
                with st.spinner("🌿 Cultivando o ambiente: Baixando pesos do modelo..."):
                    gdown.download_folder(id=Config.MODEL_COMPONENTS_GD_ID, output=Config.SAVE_DIR, quiet=True, use_cookies=False)
                    cls._flatten_if_needed(Config.SAVE_DIR)

            # Baixa embeddings se não existir
            if not os.path.exists(os.path.join(Config.EMBEDDINGS_DIR, 'metadata.csv')):
                with st.spinner("🍃 Quase lá: Baixando base de conhecimento e imagens..."):
                    gdown.download_folder(id=Config.EMBEDDINGS_GD_ID, output=Config.EMBEDDINGS_DIR, quiet=True, use_cookies=False)
                    cls._flatten_if_needed(Config.EMBEDDINGS_DIR)
                    
        except Exception as e:
            st.error(f"❌ Falha de conexão ao banco de dados: {str(e)}")
            st.stop()

# ==========================================
# 4. ARQUITETURA DO MODELO (Dual-Encoder)
# ==========================================
class ImageEncoder(nn.Module):
    def __init__(self, model_name='resnet50', embed_dim=512, freeze_backbone=True):
        super().__init__()
        self.backbone = models.__dict__[model_name](pretrained=True)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projection = nn.Linear(in_features, embed_dim)

    def forward(self, x):
        return self.projection(self.backbone(x))

class TextEncoder(nn.Module):
    def __init__(self, model_name=Config.TOKENIZER_MODEL, embed_dim=512, freeze_backbone=True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        in_features = self.backbone.config.hidden_size
        self.projection = nn.Linear(in_features, embed_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        masked_hidden = last_hidden_state * attention_mask.unsqueeze(-1)
        pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)
        return self.projection(pooled)

class DualEncoder(nn.Module):
    def __init__(self, embed_dim=512, freeze_encoders=True):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim, freeze_backbone=freeze_encoders)
        self.text_encoder = TextEncoder(embed_dim=embed_dim, freeze_backbone=freeze_encoders)

# ==========================================
# 5. MOTOR DE BUSCA (Lógica Central)
# ==========================================
class SearchEngine:
    def __init__(self):
        self.tokenizer, self.transform, self.model = self._load_model()
        self.img_embs, self.txt_embs, self.metadata = self._load_data()
        
        # Pré-normaliza os embeddings do banco de dados para acelerar inferência
        self.img_embs_norm = self.img_embs / np.linalg.norm(self.img_embs, axis=1, keepdims=True)
        self.txt_embs_norm = self.txt_embs / np.linalg.norm(self.txt_embs, axis=1, keepdims=True)

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def _load_model():
        tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_MODEL)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        model = DualEncoder(embed_dim=Config.EMBED_DIM, freeze_encoders=True)
        weights_path = os.path.join(Config.SAVE_DIR, 'dual_encoder_model_weights.pth')
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.to(Config.DEVICE).eval()
        return tokenizer, transform, model

    @staticmethod
    @st.cache_data(show_spinner=False)
    def _load_data():
        img_embs = np.load(os.path.join(Config.EMBEDDINGS_DIR, 'image_embeddings.npy'))
        txt_embs = np.load(os.path.join(Config.EMBEDDINGS_DIR, 'text_embeddings.npy'))
        metadata = pd.read_csv(os.path.join(Config.EMBEDDINGS_DIR, 'metadata.csv'))
        return img_embs, txt_embs, metadata

    def search_by_text(self, query_text, top_k=5):
        tokens = self.tokenizer(query_text, padding='max_length', truncation=True, max_length=77, return_tensors='pt').to(Config.DEVICE)
        with torch.no_grad():
            query_emb = self.model.text_encoder(tokens['input_ids'], tokens['attention_mask']).cpu().numpy()
            
        query_emb_norm = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        similarities = cosine_similarity(query_emb_norm, self.img_embs_norm).flatten()
        return self._format_results(similarities, top_k)

    def search_by_image(self, image_input, top_k=5):
        image = Image.open(image_input).convert('RGB') if isinstance(image_input, str) else image_input.convert('RGB')
        transformed = self.transform(image).unsqueeze(0).to(Config.DEVICE)
        
        with torch.no_grad():
            query_emb = self.model.image_encoder(transformed).cpu().numpy()
            
        query_emb_norm = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        similarities = cosine_similarity(query_emb_norm, self.txt_embs_norm).flatten() # Busca no espaço de texto/imagem original
        return self._format_results(similarities, top_k)

    def _format_results(self, similarities, top_k):
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                'filename': self.metadata.iloc[idx]['image_filename'],
                'description': self.metadata.iloc[idx]['description'],
                'similarity': float(similarities[idx])
            })
        return results

# ==========================================
# 6. APLICAÇÃO PRINCIPAL (Visão)
# ==========================================
def main():
    st.set_page_config(page_title="Botanical AI Search", layout="wide", page_icon="🌿")
    apply_custom_css()
    DataManager.download_assets()
    
    # Inicializa motor apenas após garantir download
    engine = SearchEngine()

    # Layout: Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1892/1892747.png", width=80)
        st.markdown("## Configurações")
        search_type = st.radio("Modo de Exploração:", ('Texto ➔ Imagem', 'Imagem ➔ Texto'))
        top_k = st.slider("Resultados por página:", min_value=1, max_value=9, value=6)
        st.markdown("---")
        st.caption("Arquitetura Dual-Encoder (ResNet50 + BERTimbau)")

    # Layout: Main Area
    st.title('🌿 Explorador Semântico de Botânica')
    st.markdown("Encontre plantas medicinais através da compreensão contextual avançada.")

    results = []

    if search_type == 'Texto ➔ Imagem':
        col1, col2 = st.columns([3, 1])
        with col1:
            query_text = st.text_input("Descreva a planta ou seu uso terapêutico:", 
                                     placeholder="Ex: Folhas verdes finas usadas para acalmar a ansiedade...")
        with col2:
            st.write("") # Spacer
            st.write("")
            buscar = st.button("🔎 Buscar", use_container_width=True)
            
        if buscar and query_text:
            with st.spinner("Analisando espaço vetorial..."):
                results = engine.search_by_text(query_text, top_k)

    else:
        uploaded_file = st.file_uploader("Envie uma imagem da planta", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(Image.open(uploaded_file), caption="Sua Imagem", use_column_width=True)
            with col2:
                if st.button("🔎 Analisar Padrões Visuais", use_container_width=True):
                    with st.spinner("Extraindo características visuais..."):
                        results = engine.search_by_image(Image.open(uploaded_file), top_k)

    # Renderização dos Resultados (O Impacto Visual)
    if results:
        st.markdown("### ✨ Melhores Correspondências")
        
        # Grid flexível (3 colunas)
        cols = st.columns(3)
        for i, res in enumerate(results):
            with cols[i % 3]:
                # Inicia o bloco HTML do Card
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                
                # Imagem
                img_path = os.path.join(Config.PLANT_IMAGES_DIR, res['filename'])
                try:
                    img = Image.open(img_path)
                    st.image(img, use_column_width=True)
                except FileNotFoundError:
                    st.error(f"Imagem ausente: {res['filename']}")
                
                # Barra de Similaridade Injetada via HTML
                st.markdown(render_similarity_bar(res['similarity']), unsafe_allow_html=True)
                
                # Descrição
                desc = res['description']
                short_desc = desc if len(desc) < 110 else desc[:107] + '...'
                st.markdown(f'<div class="description-text">{short_desc}</div>', unsafe_allow_html=True)
                
                # Fecha o Card
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()