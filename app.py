import streamlit as st
# Motivo: Importa o Streamlit para construir a interface web interativa do sistema de busca semântica.
# É a base de todo o app (layout, botões, upload, exibição de resultados).

import torch
# Motivo: Biblioteca essencial para tensores e execução de modelos deep learning (PyTorch).
# Usada para carregar pesos, fazer inference e mover tensores para CPU/GPU.

from transformers import AutoTokenizer, AutoModel
# Motivo: Permite carregar o tokenizer BERT e o modelo de texto de forma padronizada (do Hugging Face).
# Necessário para tokenizar queries de texto e inicializar o TextEncoder.

import torchvision.transforms as transforms
# Motivo: Fornece as transformações de imagem (resize, normalize etc.) salvas durante o treinamento.
# Garante que a query de imagem use exatamente o mesmo pré-processamento do modelo.

import pandas as pd
# Motivo: Manipula o DataFrame de metadados (nomes de imagens + descrições).
# Usado para recuperar descrição e nome do arquivo nos resultados de busca.

import numpy as np
# Motivo: Operações eficientes com arrays de embeddings (normalização, argsort).
# Essencial para cálculo de similaridade cosseno em escala.

from PIL import Image
# Motivo: Abre e converte imagens (upload ou arquivos salvos) para exibição no Streamlit.
# Suporte nativo para PNG/JPG no app.

import os
# Motivo: Gerencia caminhos de arquivos, criação de diretórios e verificação de existência.
# Fundamental para salvar/carregar do Google Drive localmente.

from sklearn.metrics.pairwise import cosine_similarity
# Motivo: Calcula similaridade cosseno entre embeddings de query e banco de dados.
# É o coração da busca semântica (mais rápido e preciso que distância euclidiana).

import torch.nn as nn
# Motivo: Permite definir camadas customizadas (Linear projection) nos encoders.
# Base da arquitetura dual-encoder.

import torchvision.models as models
# Motivo: Carrega backbones pré-treinados (ResNet50) do torchvision.
# Evita reimplementar a CNN do zero.

import gdown
# Motivo: Biblioteca mais simples possível para baixar pastas/arquivos do Google Drive por ID.
# Escolhida porque o projeto é pessoal para teste — não precisa de credenciais OAuth nem Google API complexa.
# Basta compartilhar a pasta como "Qualquer pessoa com o link".

import shutil  # ← NOVO IMPORT (adicione no topo do arquivo junto com os outros imports)

# === CONFIGURAÇÃO DO GOOGLE DRIVE (NOVA PARTE) ===
# Diretórios locais onde os arquivos serão salvos após o download
save_directory = './saved_model_components'
# Motivo: Mantém a mesma estrutura do código original. Aqui ficarão tokenizer, image_transform.pth e weights.pth.

embeddings_save_directory = './saved_embeddings_and_metadata'
# Motivo: Mantém a mesma estrutura. Aqui ficarão embeddings .npy, metadata.csv e a subpasta synthetic_plant_images.

# IDs das pastas no Google Drive — ***SUBSTITUA POR SEUS IDS REAIS***
# Como obter: Compartilhe cada pasta como "Qualquer pessoa com o link" → copie o ID da URL (ex: 1aBcDeFgHiJkLmNoPqRsTuVwXyZ12345)
MODEL_COMPONENTS_GD_ID = "1vPnnFsO_IsDs_I4oE73KGD5Y_BpM0dac"
# Motivo: Pasta que contém: tokenizer files (vocab.txt, config.json etc.), image_transform.pth e dual_encoder_model_weights.pth.

EMBEDDINGS_GD_ID = "10NFY8TiwMwlBnnfBwZ2hVx98M3kfQGg5"
# Motivo: Pasta que contém: image_embeddings.npy, text_embeddings.npy, metadata.csv e a subpasta synthetic_plant_images/.

tokenizer_model_pt_br = 'neuralmind/bert-base-portuguese-cased'

# Função de download (NOVA — executada uma única vez no início)

def download_from_drive():
    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(embeddings_save_directory, exist_ok=True)

    def flatten_if_needed(target_dir):
        """Remove subpasta extra criada pelo gdown (problema comum)"""
        items = os.listdir(target_dir)
        subdirs = [d for d in items if os.path.isdir(os.path.join(target_dir, d))]
        
        if len(subdirs) == 1 and len(items) <= 10:  # só 1 subpasta → é o caso do gdown
            subfolder = os.path.join(target_dir, subdirs[0])
            st.info(f"🔧 Movendo arquivos da subpasta extra: {subdirs[0]}")
            
            for item in os.listdir(subfolder):
                shutil.move(os.path.join(subfolder, item), os.path.join(target_dir, item))
            os.rmdir(subfolder)
            st.success(f"✅ Arquivos movidos para {target_dir}")

    try:
        # === MODEL COMPONENTS ===
        if not os.path.exists(os.path.join(save_directory, 'dual_encoder_model_weights.pth')):
            st.info("🔄 Baixando componentes do modelo...")
            gdown.download_folder(
                id=MODEL_COMPONENTS_GD_ID,
                output=save_directory,
                quiet=False,
                use_cookies=False
            )
            flatten_if_needed(save_directory)   # ← NOVA LINHA (corrige nesting)

        # === EMBEDDINGS + IMAGENS ===
        if not os.path.exists(os.path.join(embeddings_save_directory, 'metadata.csv')):
            st.info("🔄 Baixando embeddings, metadados e imagens...")
            gdown.download_folder(
                id=EMBEDDINGS_GD_ID,
                output=embeddings_save_directory,
                quiet=False,
                use_cookies=False
            )
            flatten_if_needed(embeddings_save_directory)  # ← NOVA LINHA (corrige nesting)
            st.write(f"📁 Arquivos encontrados ({save_directory}):", os.listdir(save_directory))
            st.write(f"📁 Arquivos encontrados ({embeddings_save_directory}):", os.listdir(embeddings_save_directory))
            st.info("✅ Sucesso no download dos dados !")

    except Exception as e:
        st.error(f"❌ Erro ao baixar: {str(e)}")
        st.stop()    # Motivo: Tratamento robusto de erro — impede que o app quebre se o GD estiver inacessível.

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
# Motivo: Layout largo para exibir várias imagens lado a lado nos resultados.

st.title('Sistema de Busca Semântica de Plantas Medicinais')
# Motivo: Título principal do aplicativo (mantido igual).

# --- Model Architecture Definition (idêntica ao notebook) ---
# Define the Image Encoder
# Define the Image Encoder
class ImageEncoder(nn.Module):
    def __init__(self, model_name='resnet50', embed_dim=512, freeze_backbone=True):
        super().__init__()
        # Load pre-trained ResNet model
        self.backbone = models.__dict__[model_name](pretrained=True)

        # Freeze backbone parameters if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the final classification layer with an identity layer
        # and add a new projection layer to match embed_dim
        if model_name.startswith('resnet'):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity() # Remove the classification head
        else:
            # Add more model types here if needed (e.g., Vision Transformer)
            raise ValueError(f"Model {model_name} not explicitly supported for layer modification.")

        self.projection = nn.Linear(in_features, embed_dim)

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projection(features)
        return embeddings
        # Motivo: Forward pass completo do encoder de imagem.

# Define the Text Encoder
# Define the Text Encoder
class TextEncoder(nn.Module):
    def __init__(self, model_name=tokenizer_model_pt_br, embed_dim=512, freeze_backbone=True):
        super().__init__()
        # Carrega o modelo pré-treinado do Hugging Face (BERTimbau ou similar)
        self.backbone = AutoModel.from_pretrained(model_name)

        # Congela os pesos do backbone se freeze_backbone=True (transfer learning)
        # Isso acelera o treino e evita overfitting nos primeiros epochs
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Obtém o tamanho do hidden state do modelo (normalmente 768 para bert-base)
        in_features = self.backbone.config.hidden_size

        # Projection layer to match the shared embedding dimension
        # Camada de projeção linear para alinhar o embedding do texto com o da imagem
        # (ex: 768 → 512)
        self.projection = nn.Linear(in_features, embed_dim)

    def forward(self, input_ids, attention_mask):
        # Passa os tokens pelo modelo BERT e retorna todas as saídas
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the embedding of the [CLS] token (first token in the sequence)
        # Pega a representação completa de todas as posições da sequência
        # Shape: [batch_size, max_length=77, hidden_size=768]
        last_hidden_state = outputs.last_hidden_state

        # Mean pooling (melhor para similaridade)
        # Multiplica cada token pelo attention_mask para ignorar [PAD]
        # Shape: [batch_size, 77, 768] * [batch_size, 77, 1] = [batch_size, 77, 768]
        masked_hidden = last_hidden_state * attention_mask.unsqueeze(-1)
        # Soma todos os tokens válidos (ignora padding)
        # Shape: [batch_size, 768]
        sum_hidden = masked_hidden.sum(dim=1)

        # Conta quantos tokens válidos existem em cada frase
        # Shape: [batch_size, 1]
        num_tokens = attention_mask.sum(dim=1).unsqueeze(-1)

        # Divide para obter a média real (mean pooling)
        # Shape: [batch_size, 768]
        pooled = sum_hidden / num_tokens

        # Aplica a projeção linear no embedding pooled (MELHOR PRÁTICA)
        # Shape final: [batch_size, embed_dim] — igual ao ImageEncoder
        embeddings = self.projection(pooled)

        # Retorna apenas o embedding projetado (pronto para contrastive loss)
        return embeddings
        # Motivo: Usa o token CLS como representação da frase inteira (padrão BERT).

# Combine into Dual Encoder Model
# Combine into Dual Encoder Model
class DualEncoder(nn.Module):
    def __init__(self, embed_dim=512, image_model_name='resnet50', text_model_name=tokenizer_model_pt_br, freeze_encoders=True):
        super().__init__()
        self.image_encoder = ImageEncoder(model_name=image_model_name, embed_dim=embed_dim, freeze_backbone=freeze_encoders)
        self.text_encoder = TextEncoder(model_name=text_model_name, embed_dim=embed_dim, freeze_backbone=freeze_encoders)

    def forward(self, images, input_ids, attention_mask):
        image_embeddings = self.image_encoder(images)
        text_embeddings = self.text_encoder(input_ids, attention_mask)
        return image_embeddings, text_embeddings
        # Motivo: Retorna os dois embeddings para cálculo de similaridade.

# --- Download e Load Model Components and Data ---
@st.cache_resource
def load_model_components():
    # Motivo: Cacheia o carregamento do modelo (executa apenas 1x por sessão — evita recarregar em cada interação).
    tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    # Motivo: Carrega tokenizer do diretório local (já baixado do GD).
    st.info("✅ tokenizer loaded !")

    #image_transform = torch.load(os.path.join(save_directory, 'image_transform.pth'))
    image_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
            )
        ])

    # Motivo: Carrega as transformações exatas usadas no treinamento.
    st.info("✅ image_transform loaded !")

    embed_dim = 512
    dual_encoder_model = DualEncoder(embed_dim=embed_dim, freeze_encoders=True)
    # Motivo: Reconstrói a arquitetura (congelada para inference).
    st.info("✅ dual_encoder_model loaded !")

    model_weights_path = os.path.join(save_directory, 'dual_encoder_model_weights.pth')
    st.info(f"✅ Loading model from {model_weights_path}...")
    dual_encoder_model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    # Motivo: Carrega os pesos treinados (do GD).

    dual_encoder_model.eval()
    # Motivo: Modo evaluation (desliga dropout e batchnorm).

    return tokenizer, image_transform, dual_encoder_model

@st.cache_data
def load_embeddings_and_metadata():
    # Motivo: Cacheia os dados pesados (embeddings + CSV) — roda só 1x.
    all_image_embeddings = np.load(os.path.join(embeddings_save_directory, 'image_embeddings.npy'))
    all_text_embeddings = np.load(os.path.join(embeddings_save_directory, 'text_embeddings.npy'))
    metadata_df = pd.read_csv(os.path.join(embeddings_save_directory, 'metadata.csv'))
    return all_image_embeddings, all_text_embeddings, metadata_df

# Carregamento com download automático
with st.spinner('🔄 Baixando dados e carregando modelo...'):
    download_from_drive()                      # Motivo: Garante que tudo esteja no disco antes de carregar.
    tokenizer, image_transform, dual_encoder_model = load_model_components()
    all_image_embeddings, all_text_embeddings, metadata_df = load_embeddings_and_metadata()

st.success('✅ Modelo, embeddings e imagens carregados com sucesso !')

# Device (CPU é suficiente para inference)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dual_encoder_model.to(device)
# Motivo: Usa GPU se disponível, senão CPU (leve e rápido).

# --- Search Functions (mantidas iguais, apenas comentários adicionados) ---
def text_search(query_text, top_k=5):
    tokenized_query = tokenizer(
        query_text,
        padding='max_length',
        truncation=True,
        max_length=77,
        return_tensors='pt'
    ).to(device)
    # Motivo: Tokeniza a query exatamente como no treinamento.

    with torch.no_grad():
        query_embedding = dual_encoder_model.text_encoder(
            tokenized_query['input_ids'],
            tokenized_query['attention_mask']
        )
    # Motivo: Gera embedding de texto sem calcular gradientes (rápido).

    # Normalização e similaridade (código original mantido)
    query_embedding_np = query_embedding.cpu().numpy()
    query_embedding_normalized = query_embedding_np / np.linalg.norm(query_embedding_np, axis=1, keepdims=True)
    all_image_embeddings_normalized = all_image_embeddings / np.linalg.norm(all_image_embeddings, axis=1, keepdims=True)
    similarities = cosine_similarity(query_embedding_normalized, all_image_embeddings_normalized).flatten()
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    st.info(f"{metadata_df.shape} Buscando top {top_k_indices} mais próximos...")

    results = []
    for idx in top_k_indices:
        st.info(f"idx {idx}")
        image_filename = metadata_df.iloc[idx]['image_filename']
        description = metadata_df.iloc[idx]['description']
        similarity_score = similarities[idx]
        results.append({
            'image_filename': image_filename,
            'description': description,
            'similarity': similarity_score
        })
        st.info(f"funcao de busca ok...")
    return results

def image_search(query_image_input, top_k=5):
    if isinstance(query_image_input, str):
        query_image = Image.open(query_image_input).convert('RGB')
    else:
        query_image = query_image_input.convert('RGB')
    # Motivo: Suporta tanto upload quanto caminho (flexibilidade).

    transformed_image = image_transform(query_image).unsqueeze(0).to(device)
    with torch.no_grad():
        query_embedding = dual_encoder_model.image_encoder(transformed_image)

    # Normalização e similaridade (código original)
    query_embedding_np = query_embedding.cpu().numpy()
    query_embedding_normalized = query_embedding_np / np.linalg.norm(query_embedding_np, axis=1, keepdims=True)
    all_text_embeddings_normalized = all_text_embeddings / np.linalg.norm(all_text_embeddings, axis=1, keepdims=True)
    similarities = cosine_similarity(query_embedding_normalized, all_text_embeddings_normalized).flatten()
    top_k_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_k_indices:
        original_image_filename = metadata_df.iloc[idx]['image_filename']
        description = metadata_df.iloc[idx]['description']
        similarity_score = similarities[idx]
        results.append({
            'original_image_filename': original_image_filename,
            'description': description,
            'similarity': similarity_score
        })
    return results

# --- Display Results Function (atualizada para usar variável de caminho) ---
def display_results(search_results, query_type='text_to_image', query_item=None, top_k=5):
    if not search_results:
        st.write("No results to display.")
        return

    st.subheader(f"Top {top_k} Results for {query_type.replace('_','-')} Query: '{query_item}'")

    cols = st.columns(min(len(search_results), 3))
    for i, result in enumerate(search_results):
        with cols[i % 3]:
            if query_type == 'text_to_image':
                image_filename = result['image_filename']
                title_prefix = "Image"
            elif query_type == 'image_to_text':
                image_filename = result['original_image_filename']
                title_prefix = "Matching Image"
            else:
                image_filename = result.get('image_filename', result.get('original_image_filename'))
                title_prefix = "Image"

            # ATUALIZAÇÃO: usa a variável embeddings_save_directory (compatível com Google Drive)
            img_path = os.path.join(embeddings_save_directory, 'plant_images', image_filename)
            # Motivo: Antes era caminho hard-coded; agora usa a variável — funciona perfeitamente após download do GD.

            try:
                img = Image.open(img_path)
                st.image(img, caption=f"{title_prefix}: {image_filename}", use_column_width=True)
                description = result['description']
                similarity = result['similarity']
                display_description = description if len(description) < 100 else description[:97] + '...'
                st.markdown(f"**Similarity:** {similarity:.4f}")
                st.markdown(f"**Description:** {display_description}")
            except FileNotFoundError:
                st.error(f"Image not found at {img_path}")
                st.write(f"**Similarity:** {similarity:.4f}")
                st.write(f"**Description:** {description}")
            except Exception as e:
                st.error(f"Error displaying image {image_filename}: {e}")

# --- Streamlit UI and Logic (mantido igual) ---
st.sidebar.header("Opções de Busca")
search_type = st.sidebar.radio("Escolha o tipo de busca:", ('Texto para Imagem', 'Imagem para Texto'))
top_k = st.sidebar.slider("Número de resultados", min_value=1, max_value=10, value=5)

if search_type == 'Texto para Imagem':
    st.subheader("Busca: Texto para Imagem")
    query_text = st.text_input("Digite sua descrição da planta:", "uma planta de folhas verdes, usada para acalmar os nervos e me deixar relaxado.")
    if st.button("Buscar Imagens"):
        if query_text:
            results = text_search(query_text, top_k)
            display_results(results, query_type='text_to_image', query_item=query_text, top_k=top_k)
        else:
            st.warning("Por favor, digite um texto para busca.")

elif search_type == 'Imagem para Texto':
    st.subheader("Busca: Imagem para Texto")
    uploaded_file = st.file_uploader("Faça upload de uma imagem de planta:", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        query_image = Image.open(uploaded_file)
        st.image(query_image, caption='Imagem Carregada.', use_column_width=True)
        if st.button("Buscar Descrições"):
            results = image_search(query_image, top_k)
            display_results(results, query_type='image_to_text', query_item=uploaded_file.name, top_k=top_k)
    else:
        st.info("Por favor, carregue uma imagem para busca.")

st.sidebar.markdown("--- (C) 2026 Sistema de Busca Semântica ---")
# Motivo: Rodapé simples (mantido).