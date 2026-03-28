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
    
    # IDs do Google Drive
    MODEL_COMPONENTS_GD_ID = "1vPnnFsO_IsDs_I4oE73KGD5Y_BpM0dac"
    EMBEDDINGS_GD_ID = "10NFY8TiwMwlBnnfBwZ2hVx98M3kfQGg5"
    
    TOKENIZER_MODEL = 'neuralmind/bert-base-portuguese-cased'
    EMBED_DIM = 512
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. UI & ESTILIZAÇÃO CSS (Impacto Visual)
# ==========================================
def apply_custom_css():
    """Injeta CSS customizado para estilizar a interface."""
    st.markdown("""
        <style>
        .stApp { background-color: #F4F7F5; }
        h1, h2, h3 { color: #1E3B2D; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: 700; }
        
        /* Estilo dos Cards de Resultado */
        .result-card {
            background: #ffffff;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.06);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            margin-bottom: 25px;
            height: 100%;
            border: 1px solid #E8EFEA;
        }
        .result-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 15px 30px rgba(30, 59, 45, 0.12);
        }
        
        /* Tipografia dentro do card */
        .plant-title {
            font-size: 1.3rem;
            color: #2D5A43;
            font-weight: bold;
            margin-bottom: 5px;
            margin-top: 15px;
            text-transform: capitalize;
        }
        .description-text {
            font-size: 0.95rem;
            color: #5C6E64;
            margin-top: 12px;
            line-height: 1.5;
        }
        
        /* Barra de similaridade */
        .similarity-container {
            width: 100%;
            background-color: #E0E0E0;
            border-radius: 8px;
            margin: 5px 0 15px 0;
            height: 10px;
            overflow: hidden;
        }
        
        /* Botões customizados */
        div.stButton > button:first-child {
            background-color: #2D5A43;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 12px 28px;
            font-weight: bold;
            transition: all 0.3s;
        }
        div.stButton > button:first-child:hover {
            background-color: #1E3B2D;
            box-shadow: 0 6px 15px rgba(45,90,67,0.4);
        }
        
        /* Arredonda as bordas das imagens renderizadas pelo Streamlit */
        [data-testid="stImage"] img {
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

def render_similarity_bar(score):
    """Gera o HTML da barra de progresso gradiente baseada no score (0 a 1)."""
    percentage = max(0, min(100, int(score * 100)))
    color = f"hsl({120 * score}, 65%, 40%)" 
    html = f"""
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.85em; color: #888; font-weight: 600; text-transform: uppercase;">Grau de Confiança</span>
            <span style="font-weight: 900; color: {color}; font-size: 1.2em;">{score:.2f}</span>
        </div>
        <div class="similarity-container">
            <div style="width: {percentage}%; background-color: {color}; height: 100%; border-radius: 8px; transition: width 1s ease-in-out;"></div>
        </div>
    """
    return html

# ==========================================
# 3. GERENCIAMENTO DE DADOS (Downloads)
# ==========================================
class DataManager:
    @staticmethod
    def _flatten_if_needed(target_dir):
        """Corrige o problema de subpastas extras criadas pelo gdown de forma segura."""
        items = os.listdir(target_dir)
        
        # Se a pasta alvo tem EXATAMENTE 1 item, e esse item é uma pasta,
        # significa que o gdown envelopou tudo numa pasta extra.
        if len(items) == 1:
            subfolder = os.path.join(target_dir, items[0])
            if os.path.isdir(subfolder):
                for item in os.listdir(subfolder):
                    shutil.move(os.path.join(subfolder, item), os.path.join(target_dir, item))
                os.rmdir(subfolder)

    @classmethod
    def download_assets(cls):
        os.makedirs(Config.SAVE_DIR, exist_ok=True)
        os.makedirs(Config.EMBEDDINGS_DIR, exist_ok=True)

        try:
            if not os.path.exists(os.path.join(Config.SAVE_DIR, 'dual_encoder_model_weights.pth')):
                with st.spinner("🌿 Baixando componentes arquiteturais do modelo..."):
                    gdown.download_folder(id=Config.MODEL_COMPONENTS_GD_ID, output=Config.SAVE_DIR, quiet=True, use_cookies=False)
                    cls._flatten_if_needed(Config.SAVE_DIR)

            if not os.path.exists(os.path.join(Config.EMBEDDINGS_DIR, 'metadata.csv')):
                with st.spinner("🍃 Baixando protótipos, matrizes e acervo botânico..."):
                    gdown.download_folder(id=Config.EMBEDDINGS_GD_ID, output=Config.EMBEDDINGS_DIR, quiet=True, use_cookies=False)
                    cls._flatten_if_needed(Config.EMBEDDINGS_DIR)
                    
        except Exception as e:
            st.error(f"❌ Erro crítico de conexão com a nuvem: {str(e)}")
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
        masked_hidden = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
        pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)
        return self.projection(pooled)

class DualEncoder(nn.Module):
    def __init__(self, embed_dim=512, freeze_encoders=True):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim, freeze_backbone=freeze_encoders)
        self.text_encoder = TextEncoder(embed_dim=embed_dim, freeze_backbone=freeze_encoders)

# ==========================================
# 5. MOTOR DE BUSCA (Lógica de Protótipos)
# ==========================================
class SearchEngine:
    def __init__(self):
        self.tokenizer, self.transform, self.model = self._load_model()
        self.metadata_df, self.proto_names, self.proto_img_norm, self.proto_txt_norm = self._load_data()

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
        model.load_state_dict(torch.load(os.path.join(Config.SAVE_DIR, 'dual_encoder_model_weights.pth'), map_location='cpu'))
        model.to(Config.DEVICE).eval()
        return tokenizer, transform, model

    @staticmethod
    @st.cache_data(show_spinner=False)
    def _load_data():
        metadata_df = pd.read_csv(os.path.join(Config.EMBEDDINGS_DIR, 'metadata.csv'))
        
        # Carrega os Prototypes otimizados
        checkpoint = torch.load(os.path.join(Config.EMBEDDINGS_DIR, 'prototypes.pt'), map_location='cpu')
        prototypes = checkpoint['prototypes']
        proto_names = checkpoint['proto_names']

        # Constrói e normaliza a matriz de imagem
        proto_image_matrix = np.stack([prototypes[p]['image_proto'].cpu().numpy() for p in proto_names])
        proto_img_norm = proto_image_matrix / np.linalg.norm(proto_image_matrix, axis=1, keepdims=True)

        # Constrói e normaliza a matriz de texto
        proto_text_matrix = np.stack([prototypes[p]['text_proto'].cpu().numpy() for p in proto_names])
        proto_txt_norm = proto_text_matrix / np.linalg.norm(proto_text_matrix, axis=1, keepdims=True)

        return metadata_df, proto_names, proto_img_norm, proto_txt_norm

    def search_by_text(self, query_text, top_k=5):
        tokens = self.tokenizer(query_text, padding='max_length', truncation=True, max_length=77, return_tensors='pt').to(Config.DEVICE)
        
        with torch.no_grad():
            query_emb = self.model.text_encoder(tokens['input_ids'], tokens['attention_mask']).cpu().numpy()
            
        query_norm = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        # Busca cross-modal: Texto -> Protótipos de Imagem
        similarities = cosine_similarity(query_norm, self.proto_img_norm).flatten()
        return self._format_results(similarities, top_k)

    def search_by_image(self, image_input, top_k=5):
        # Garante o processamento correto caso a entrada seja string ou objeto PIL Uploaded
        image = Image.open(image_input).convert('RGB') if isinstance(image_input, str) else image_input.convert('RGB')
        transformed = self.transform(image).unsqueeze(0).to(Config.DEVICE)
        
        with torch.no_grad():
            query_emb = self.model.image_encoder(transformed).cpu().numpy()
            
        query_norm = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        # Busca cross-modal: Imagem -> Protótipos de Texto
        similarities = cosine_similarity(query_norm, self.proto_txt_norm).flatten()
        return self._format_results(similarities, top_k)

    def _format_results(self, similarities, top_k):
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            plant_name = self.proto_names[idx]
            # Extrai os metadados da planta correspondente
            row = self.metadata_df[self.metadata_df['plant_name'] == plant_name].iloc[0]
            results.append({
                'rank': rank + 1,
                'plant_name': plant_name,
                'description': row['description'],
                'similarity': float(similarities[idx]),
                'image_filename': row['image_filename']
            })
        return results

# ==========================================
# 6. APLICAÇÃO PRINCIPAL (Visão e Fluxo)
# ==========================================
def main():
    st.set_page_config(page_title="Botanical AI Search", layout="wide", page_icon="🌿")
    apply_custom_css()
    DataManager.download_assets()
    
    engine = SearchEngine()

    # --- BARRA LATERAL ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2920/2920349.png", width=80)
        st.markdown("## Filtros de Exploração")
        search_type = st.radio("Selecione o Fluxo:", ('Texto ➔ Imagem', 'Imagem ➔ Texto'))
        top_k = st.slider("Resultados Desejados", min_value=1, max_value=9, value=3)
        st.markdown("---")
        st.caption("Powered by Dual-Encoder & Prototype Matrices")

    # --- ÁREA PRINCIPAL ---
    st.title('🌿 Explorador Semântico Botânico')
    st.markdown("Descubra propriedades medicinais através de Inteligência Artificial Multimodal.")

    results = []

    if search_type == 'Texto ➔ Imagem':
        col1, col2 = st.columns([4, 1])
        with col1:
            query_text = st.text_input("Descreva as características da planta ou suas aplicações:", 
                                     placeholder="Ex: Folhas alongadas com propriedades digestivas...")
        with col2:
            st.write("") # Spacer
            st.write("")
            buscar = st.button("🔎 Localizar", use_container_width=True)
            
        if buscar and query_text:
            with st.spinner("Mapeando contexto no espaço latente..."):
                results = engine.search_by_text(query_text, top_k)

    else:
        uploaded_file = st.file_uploader("Submeta o registro fotográfico da planta", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(Image.open(uploaded_file), caption="Espécime Analisada", use_column_width=True)
            with col2:
                if st.button("🔎 Iniciar Análise Morfológica", use_container_width=True):
                    with st.spinner("Extraindo vetores morfológicos..."):
                        # Correção do Bug: Passando o objeto PIL nativo, e não o caminho
                        results = engine.search_by_image(Image.open(uploaded_file), top_k)

    # --- RENDERIZAÇÃO DE RESULTADOS (O Impacto Visual) ---
    if results:
        st.markdown("<br><h3 style='text-align: center; color: #2D5A43;'>✨ Melhores Correspondências Encontradas</h3><br>", unsafe_allow_html=True)
        
        # Grid flexível (até 3 colunas)
        cols = st.columns(min(len(results), 3))
        for i, res in enumerate(results):
            with cols[i % 3]:
                # Abertura do Card HTML
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                
                # Renderiza Imagem
                img_path = os.path.join(Config.PLANT_IMAGES_DIR, res['image_filename'])
                try:
                    img = Image.open(img_path)
                    st.image(img, use_column_width=True)
                except FileNotFoundError:
                    st.error(f"Arquivo visual indisponível: {res['image_filename']}")
                
                # Título da Planta e Barra de Similaridade
                st.markdown(f"<div class='plant-title'>{res['plant_name']}</div>", unsafe_allow_html=True)
                st.markdown(render_similarity_bar(res['similarity']), unsafe_allow_html=True)
                
                # Descrição Truncada elegantemente
                desc = res['description']
                display_desc = desc if len(desc) < 140 else desc[:137] + '...'
                st.markdown(f'<div class="description-text">{display_desc}</div>', unsafe_allow_html=True)
                
                # Fechamento do Card HTML
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()