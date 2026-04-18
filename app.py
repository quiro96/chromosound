import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from moviepy.editor import VideoClip, AudioFileClip, CompositeVideoClip, ColorClip
import tempfile
import os

st.set_page_config(page_title="Chromosound Web", layout="wide")

def compute_mandala_data(y, sr):
    # 1. Analisi Spettrale (STFT)
    S = librosa.stft(y, n_fft=512, hop_length=128, win_length=256)
    P = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    
    # 2. Normalizzazione (Taglio all'80% come in MATLAB)
    max_p = np.max(P)
    P[P > max_p * 0.8] = max_p * 0.8
    
    # 3. INTERPOLAZIONE (1000 punti per fluidità)
    num_punti_target = 1000
    f_len, p_len = P.shape
    x_old = np.linspace(0, 1, p_len)
    y_old = np.linspace(0, 1, f_len)
    f_interp = interp2d(x_old, y_old, P, kind='linear')
    
    x_new = np.linspace(0, 1, num_punti_target)
    P = f_interp(x_new, y_old)
    
    # 4. FIX CONTINUITÀ (Sfuma giunzione ore 3)
    n_blend = 20
    for i in range(n_blend):
        alpha = i / n_blend
        P[:, -(n_blend-i)] = P[:, -(n_blend-i)] * (1 - alpha) + P[:, 0] * alpha
        
    # 5. Chiusura Geometrica
    P = np.hstack([P, P[:, [0]]])
    
    freqs = np.linspace(0, sr/2, f_len)
    R = freqs + 500
    Theta = np.linspace(0, 2*np.pi, P.shape[1])
    
    return P, R, Theta

def make_frame(t, P, R, Theta, config, duration_audio):
    t_intro = config['intro']
    t_audio = duration_audio
    
    fig, ax = plt.subplots(figsize=(config['w']/100, config['h']/100), dpi=100, facecolor='black')
    ax.set_facecolor('black')
    
    # Calcolo progresso
    if t < t_intro:
        # Solo testo (Intro)
        pass 
    else:
        t_rel = t - t_intro
        t_ratio = min(t_rel / t_audio, 1.0)
        
        P_f = np.copy(P)
        mask = Theta > (t_ratio * 2 * np.pi + 0.01)
        P_f[:, mask] = np.nan
        
        T_grid, R_grid = np.meshgrid(Theta, R)
        
        if config['vinyl'] and t_rel <= t_audio:
            rot = t_ratio * 2 * np.pi
            T_plot = T_grid - rot
        else:
            T_plot = T_grid
            
        X = R_grid * np.cos(T_plot)
        Y = R_grid * np.sin(T_plot)
        
        # Effetto Fade Out finale
        alpha_fade = 1.0
        t_fade_start = t_intro + t_audio + config['hold']
        if t > t_fade_start:
            alpha_fade = max(0, 1 - (t - t_fade_start) / config['fade'])

        im = ax.pcolormesh(X, Y, P_f, shading='gouraud', cmap=config['cmap'], alpha=alpha_fade)
        
    # Titolo
    max_r = np.max(R)
    ax.text(0, -max_r * 1.2, config['title'], color='white', fontsize=20, 
            fontweight='bold', ha='center', va='center')
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Adattamento telecamera (come in MATLAB)
    lim = max_r * 1.4
    if config['w'] > config['h']:
        ax.set_ylim(-lim, lim)
        ax.set_xlim(-lim * (config['w']/config['h']), lim * (config['w']/config['h']))
    else:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim * (config['h']/config['w']), lim * (config['h']/config['w']))

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image

# --- INTERFACCIA STREAMLIT ---
st.title("Chromosound Web 🚀")
st.sidebar.header("Impostazioni")

uploaded_file = st.sidebar.file_uploader("Carica Audio", type=["mp3", "wav", "flac"])
title = st.sidebar.text_input("Titolo", "SOUNDWAVE")
cmap = st.sidebar.selectbox("Colormap", ["hot", "magma", "inferno", "plasma", "viridis", "icefire", "coolwarm"])
format_choice = st.sidebar.selectbox("Formato", ["Quadrato", "Verticale", "Orizzontale"])
vinyl = st.sidebar.checkbox("Effetto Vinile", value=True)

col1, col2, col3 = st.sidebar.columns(3)
t_intro = col1.number_input("Intro", 0.0, 5.0, 1.0)
t_hold = col2.number_input("Hold", 0.0, 5.0, 1.0)
t_fade = col3.number_input("Fade", 0.0, 5.0, 1.0)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    y, sr = librosa.load(tmp_path)
    duration_audio = librosa.get_duration(y=y, sr=sr)
    
    if st.button("🎬 GENERA VIDEO"):
        with st.spinner("Elaborazione in corso... potrebbe richiedere un minuto."):
            P, R, Theta = compute_mandala_data(y, sr)
            
            res = {"Quadrato": (800, 800), "Verticale": (1080, 1920), "Orizzontale": (1920, 1080)}
            w, h = res[format_choice]
            
            config = {
                'w': w, 'h': h, 'cmap': cmap, 'title': title, 'vinyl': vinyl,
                'intro': t_intro, 'hold': t_hold, 'fade': t_fade
            }
            
            total_duration = t_intro + duration_audio + t_hold + t_fade
            
            video = VideoClip(lambda t: make_frame(t, P, R, Theta, config, duration_audio), duration=total_duration)
            
            audio = AudioFileClip(tmp_path)
            # Sincronizzazione audio con intro
            silence_intro = ColorClip(size=(1,1), color=(0,0,0), duration=t_intro).set_audio(None)
            final_audio = CompositeVideoClip([video]).set_audio(audio.set_start(t_intro))
            
            out_path = "output_mandala.mp4"
            final_audio.write_videofile(out_path, fps=24, codec="libx264", audio_codec="aac")
            
            st.video(out_path)
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Scarica Video", f, file_name="Il_Mio_Mandala.mp4")
            
            os.remove(tmp_path)
