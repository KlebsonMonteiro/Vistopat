from collections import Counter
import io
import numpy as np
import streamlit as st
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from ultralytics import YOLO

# 1. Configuração Inicial e Estilização Visual (Tema Claro Avançado)
st.set_page_config(
    page_title="VistoPat Inteligente", page_icon="⚡", layout="centered"
)

# Injeção de CSS global centralizada
# Observação: os "cards" agora usam st.container(border=True) em vez de
# <div> abertas/fechadas em chamadas st.markdown separadas. Isso evita o
# erro "NotFoundError: Failed to execute 'removeChild' on 'Node'", que
# acontece quando tags HTML ficam "soltas" entre elementos e o React do
# Streamlit não consegue reconciliar o DOM ao trocar de página/rerun.
st.markdown(
    """
    <style>
    /* Paleta com contraste reforçado (mínimo AA ~4.5:1 em todo texto) */
    .stApp { background-color: #f4f6f8 !important; color: #14181f !important; }
    .stApp p, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp li { color: #14181f !important; }
    .block-container { padding-top: 2rem; padding-bottom: 5rem; }
    .top-header {
        background-color: #12233f; padding: 15px; border-radius: 6px;
        margin-bottom: 25px; color: #ffffff !important; font-weight: bold;
        font-size: 24px; display: flex; align-items: center;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.15);
    }
    .main-title { font-size: 28px; font-weight: bold; color: #12233f !important; margin-bottom: 5px; }
    .subtitle { font-size: 14px; color: #33404f !important; margin-bottom: 25px; }
    .section-title {
        font-size: 18px; font-weight: bold; color: #12233f !important;
        margin-bottom: 15px; display: flex; align-items: center; gap: 8px;
    }
    /* Estiliza os containers nativos do Streamlit (border=True) como "cards" */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important; border-radius: 8px !important;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.06); border: 1px solid #ccd3db !important;
        margin-bottom: 20px;
    }
    /* Legendas (st.caption) com contraste reforçado, em vez do cinza claro padrão */
    div[data-testid="stCaptionContainer"], .stApp small {
        color: #465364 !important; font-weight: 600 !important; letter-spacing: 0.02em;
    }
    .stTextInput input, .stTextArea textarea {
        background-color: #ffffff !important; border: 1px solid #99a4b2 !important; color: #14181f !important;
    }
    .stTextInput input::placeholder, .stTextArea textarea::placeholder { color: #6b7685 !important; }
    div[data-testid="stFileUploader"] > section { background-color: #eef1f5 !important; border: 2px dashed #99a4b2 !important; border-radius: 8px !important; padding: 20px !important; }
    div[data-testid="stFileUploader"] button { background-color: #ffffff !important; border: 1px solid #99a4b2 !important; color: #12233f !important; font-weight: 600 !important; box-shadow: 0px 1px 2px rgba(0, 0, 0, 0.05) !important; }
    div[data-testid="stFileUploader"] label, div[data-testid="stFileUploader"] p, div[data-testid="stFileUploader"] span { color: #33404f !important; }
    div.stButton > button, div.stDownloadButton > button {
        background-color: #1d4ed8 !important; color: #ffffff !important; font-weight: bold !important; font-size: 16px !important;
        padding: 12px 24px !important; border-radius: 6px !important; border: none !important; width: 100% !important; transition: background-color 0.15s; box-shadow: 0px 2px 4px rgba(29, 78, 216, 0.25);
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover { background-color: #1638a8 !important; }
    div.stButton > button p, div.stDownloadButton > button p { color: #ffffff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Função para gerar o PDF em memória usando ReportLab
def generate_pdf(data, counts):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=24, leading=28, textColor='#1a365d', spaceAfter=15)
    section_style = ParagraphStyle('Section', parent=styles['Heading2'], fontSize=16, leading=20, textColor='#1a365d', spaceBefore=15, spaceAfter=10)
    text_style = ParagraphStyle('Text', parent=styles['Normal'], fontSize=11, leading=15, spaceAfter=8)
    bold_style = ParagraphStyle('BoldText', parent=text_style, fontName='Helvetica-Bold')

    story.append(Paragraph("⚡ VistoPat - Relatório Técnico", title_style))
    story.append(Spacer(1, 15))

    story.append(Paragraph("🏢 Informações da Estrutura", section_style))
    story.append(Paragraph(f"<b>Nome do Edifício:</b> {data.get('edificio', '-')}", text_style))
    story.append(Paragraph(f"<b>Bloco / Localização:</b> {data.get('bloco', '-')}", text_style))
    story.append(Paragraph(f"<b>Engenheiro Responsável:</b> {data.get('engenheiro') if data.get('engenheiro') else '-'}", text_style))
    story.append(Paragraph(f"<b>Observações:</b> {data.get('observacoes') if data.get('observacoes') else '-'}", text_style))
    story.append(Spacer(1, 15))

    story.append(Paragraph("⚠️ Patologias Identificadas pela IA", section_style))

    if len(counts) > 0:
        for patologia, qtd in counts.items():
            if "rachadura" in patologia.lower() or "fissura" in patologia.lower():
                nbr = "NBR 13752 e NBR 16747 (Inspeção Predial)"
                grau = "Crítico"
                rec = "Recomenda-se a realização de ensaios estruturais na alvenaria e monitoramento contínuo da estabilidade."
            elif "mofo" in patologia.lower() or "infiltração" in patologia.lower():
                nbr = "NBR 15575 - Critérios de estanqueidade"
                grau = "Regular"
                rec = "Identificar a origem da infiltração activa. Sugere-se raspagem, tratamento fungicida e recomposição."
            else:
                nbr = "Normas Técnicas Gerais e NBR 16747"
                grau = "Leve"
                rec = "Realizar manutenção preventiva e reparos superficiais de rotina."

            story.append(Paragraph(f"<b>• Patologia: {patologia.upper()}</b> (Detectado {qtd}x)", bold_style))
            story.append(Paragraph(f"<b>Norma Aplicada:</b> {nbr}", text_style))
            story.append(Paragraph(f"<b>Grau de Risco:</b> {grau}", text_style))
            story.append(Paragraph(f"<b>Recomendação:</b> {rec}", text_style))
            story.append(Spacer(1, 10))
    else:
        story.append(Paragraph("Nenhuma patologia ou anomalia estrutural foi detectada pelo modelo inteligente.", text_style))

    story.append(Spacer(1, 15))
    story.append(Paragraph("✓ Diagnóstico gerado automaticamente via inteligência artificial computacional.", text_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# Cache do Modelo YOLO
@st.cache_resource
def load_model():
    return YOLO("best.pt")


try:
    model = load_model()
except Exception as e:
    model = None
    st.error(f"Erro ao carregar o modelo (best.pt): {e}")

# Inicialização segura dos estados de sessão
if "page" not in st.session_state:
    st.session_state.page = "formulario"
if "form_data" not in st.session_state:
    st.session_state.form_data = {}
if "annotated_img_rgb" not in st.session_state:
    st.session_state.annotated_img_rgb = None
if "detections_count" not in st.session_state:
    st.session_state.detections_count = Counter()

# --- TELA 1: FORMULÁRIO DE INSPEÇÃO ---
if st.session_state.page == "formulario":
    st.markdown('<div class="top-header" style="color: white !important;">⚡ VistoPat IA </div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Nova Inspeção</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Anexe a imagem para análise.</div>', unsafe_allow_html=True)

    if st.session_state.get("load_error"):
        st.error(f"Não foi possível processar a imagem anterior: {st.session_state.load_error}. Tente novamente.")
        st.session_state.load_error = None

    with st.container(border=True):
        st.markdown('<div class="section-title">🖼️ Imagem capturada</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Clique para selecionar uma imagem da galeria",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="uploader_input"
        )

    with st.container(border=True):
        st.markdown('<div class="section-title">🏢 Informações da Estrutura</div>', unsafe_allow_html=True)

        nome_edificio = st.text_input("Nome do Edifício *", placeholder="Ex: Residencial Alphaville", key="f_edificio")
        bloco_localizacao = st.text_input("Bloco / Localização *", placeholder="Ex: Bloco B - Fachada Leste", key="f_bloco")
        nome_engenheiro = st.text_input("Nome do Engenheiro *", placeholder="Ex: Eng. Roberto Carlos", key="f_eng")
        observacoes = st.text_area("Observações", placeholder="Detalhes adicionais sobre a vistoria...", key="f_obs")

    if st.button("Executar Análise Inteligente", key="btn_analisar"):
        if not uploaded_file or not nome_edificio or not bloco_localizacao:
            st.error("Por favor, preencha os campos obrigatórios (*) e carregue uma imagem.")
        else:
            st.session_state.form_data = {
                "edificio": nome_edificio,
                "bloco": bloco_localizacao,
                "engenheiro": nome_engenheiro,
                "observacoes": observacoes,
                "file": uploaded_file.read(),
            }
            st.session_state.page = "loading"
            st.rerun()

# --- TELA 2: ANIMAÇÃO DE CARREGAMENTO E PROCESSAMENTO ---
elif st.session_state.page == "loading":
    st.markdown(
        """
        <style>
        .stApp { background-color: #0d1b2f !important; }
        .stApp p, .stApp div, .stApp span { color: #ffffff !important; }
        .loading-title { font-size: 32px; font-weight: bold; margin-bottom: 5px; color: #ffffff !important; text-align: center; }
        .loading-subtitle { font-size: 16px; color: #a9c1ff !important; margin-bottom: 40px; text-align: center; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="loading-title">VistoPat IA</div>', unsafe_allow_html=True)
    st.markdown('<div class="loading-subtitle">⚡ Analisando a imagem...</div>', unsafe_allow_html=True)

    # Se a sessão foi perdida (ex: refresh no meio do processo), volta pro
    # formulário em vez de quebrar com KeyError.
    data = st.session_state.form_data
    if "file" not in data:
        st.session_state.page = "formulario"
        st.rerun()

    # Sem atrasos artificiais (time.sleep) — só o tempo real de inferência,
    # mostrado com um spinner nativo do Streamlit.
    with st.spinner("Processando imagem com o modelo de IA..."):
        try:
            image = Image.open(io.BytesIO(data["file"])).convert("RGB")
            # Redimensiona direto para o tamanho de entrada do modelo (mais rápido
            # e usa menos memória do que redimensionar para 1024 e deixar o
            # YOLO redimensionar de novo internamente).
            image.thumbnail((640, 640))
            image_np = np.array(image)

            if model is not None:
                result = model.predict(source=image_np, imgsz=640, conf=0.40, verbose=False)[0]
                annotated_img_bgr = result.plot()
                st.session_state.annotated_img_rgb = annotated_img_bgr[:, :, ::-1]

                if result.boxes is not None and len(result.boxes) > 0:
                    classes_ids = result.boxes.cls.cpu().numpy().astype(int)
                    class_names = [model.names[cls_id] for cls_id in classes_ids]
                    st.session_state.detections_count = Counter(class_names)
                else:
                    st.session_state.detections_count = Counter()
            else:
                st.session_state.annotated_img_rgb = image_np
                st.session_state.detections_count = Counter()

            st.session_state.page = "resultado"
        except Exception as e:
            st.session_state.page = "formulario"
            st.session_state.load_error = str(e)

    st.rerun()

# --- TELA 3: RELATÓRIO TÉCNICO ---
elif st.session_state.page == "resultado":
    st.markdown('<div class="top-header" style="color: white !important;">⚡ VistoPat IA</div>', unsafe_allow_html=True)

    if st.button("⬅️ Voltar para Nova Inspeção", key="btn_voltar"):
        st.session_state.page = "formulario"
        st.rerun()

    st.markdown('<div class="main-title">📋 Relatório Técnico</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Resultados gerados automaticamente com base na análise de IA.</div>', unsafe_allow_html=True)

    data = st.session_state.form_data
    if not data:
        st.warning("Nenhuma inspeção em andamento. Volte e preencha o formulário.")
        st.stop()

    with st.container(border=True):
        st.markdown('<div class="section-title">🏢 Informações da Estrutura</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.caption("NOME DO EDIFÍCIO")
            st.markdown(f"**{data.get('edificio', '-')}**")
            st.caption("NOME DO ENGENHEIRO")
            st.markdown(f"**{data.get('engenheiro') if data.get('engenheiro') else '-'}**")
        with col2:
            st.caption("BLOCO / LOCALIZAÇÃO")
            st.markdown(f"**{data.get('bloco', '-')}**")
            st.caption("OBSERVAÇÕES")
            st.markdown(f"*{data.get('observacoes') if data.get('observacoes') else '-'}*")

    with st.container(border=True):
        st.markdown('<div class="section-title">⚠️ Resumo das Patologias Identificadas</div>', unsafe_allow_html=True)

        if st.session_state.annotated_img_rgb is not None:
            st.image(st.session_state.annotated_img_rgb, caption="Diagnóstico Visual do Modelo", use_container_width=True)

        counts = st.session_state.detections_count
        if len(counts) > 0:
            st.markdown("### 🔍 Detalhamento das Patologias:")
            for idx, (patologia, qtd) in enumerate(counts.items()):
                if "rachadura" in patologia.lower() or "fissura" in patologia.lower():
                    nbr = "NBR 13752 (Perícias de Engenharia na Construção Civil) e NBR 16747 (Inspeção Predial)"
                    grau = "Crítico"
                    rec = "Recomenda-se a realização de ensaios estruturais na alvenaria e monitoramento contínuo da estabilidade. Avaliar a necessidade imediata de escoramento e reforço estrutural, visto que a fissura diagonal sugere comprometimento por recalque diferencial ou sobrecarga."
                elif "mofo" in patologia.lower() or "infiltração" in patologia.lower():
                    nbr = "NBR 15575 (Desempenho de Edificações Habitacionais) - Critérios de estanqueidade"
                    grau = "Regular"
                    rec = "Identificar a origem da infiltração activa que está alimentando a proliferação de fungos. Inspecionar possíveis falhas na impermeabilização externa ou vazamentos na rede hidrossanitária adjacente. Sugere-se raspagem, tratamento fungicida e recomposição do revestimento de acabamento após correção do foco de umidade."
                else:
                    nbr = "Normas Técnicas Gerais de Engenharia Civil e NBR 16747"
                    grau = "Leve"
                    rec = "Realizar manutenção preventiva e reparos superficiais de rotina para evitar evolução do quadro."

                cor_grau = "#b91c1c" if grau == "Crítico" else ("#b45309" if grau == "Regular" else "#15803d")

                st.markdown(
                    f"""
                    ---
                    🔴 <span style='font-size:18px; font-weight:bold; color:#12233f;'>{patologia.upper()}</span>
                    * **Classificação segundo a NBR:** {nbr}
                    * **Grau de Risco/Severidade:** <span style='color:{cor_grau}; font-weight:bold;'>{grau}</span>
                    * **Recomendação para o Engenheiro:** {rec}
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                """
                ---
                🔹 <span style='color:#15803d; font-weight:bold;'>✓ Diagnóstico gerado automaticamente.</span> Verificação física exigida para atestado final do responsável técnico.
                """,
                unsafe_allow_html=True
            )
        else:
            st.success("Análise concluída com sucesso: Nenhuma anomalia ou patologia crítica detectada na estrutura.")

    pdf_data = generate_pdf(data, counts)

    st.download_button(
        label="📥 Exportar Relatório em PDF",
        data=pdf_data,
        file_name=f"Relatorio_{data.get('edificio', 'inspecao').replace(' ', '_')}.pdf",
        mime="application/pdf",
        key="btn_download_pdf"
    )
