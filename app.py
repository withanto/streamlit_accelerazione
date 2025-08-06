import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# --- Funzioni helper ---

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b, a

def trova_minimi(signal, soglia, fs, min_sep_time):
    all_min, _ = find_peaks(-signal)
    valid = [i for i in all_min if signal[i] < soglia]
    result = []
    if valid:
        cluster = [valid[0]]
        for idx in valid[1:]:
            if (idx - cluster[-1]) < min_sep_time * fs:
                cluster.append(idx)
            else:
                result.append(min(cluster, key=lambda i: signal[i]))
                cluster = [idx]
        result.append(min(cluster, key=lambda i: signal[i]))
    return np.array(result)

def trova_picchi_massimi(signal, min_indices):
    if len(min_indices) > 1:
        return np.array([min_indices[i] + np.argmax(signal[min_indices[i]:min_indices[i+1]+1]) for i in range(len(min_indices)-1)])
    else:
        return np.array([])

def affina_indici(indices, times, signal, t_range, mode='max'):
    refined = []
    for idx in indices:
        t0 = times[idx]
        mask = (times >= t0 - t_range) & (times <= t0 + t_range)
        idxs = np.where(mask)[0]
        if len(idxs):
            sel = np.argmax(signal[idxs]) if mode=='max' else np.argmin(signal[idxs])
            refined.append(idxs[sel])
    return np.array(refined)

def plot_with_ticker(time, signal, title, label, min_idx=None, max_idx=None):
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(time, signal, label=label, color='dodgerblue', alpha=0.8)
    if min_idx is not None:
        ax.scatter(time[min_idx], signal[min_idx], color='green', marker='x', s=70, label='Minimi')
    if max_idx is not None:
        ax.scatter(time[max_idx], signal[max_idx], color='red', marker='o', s=70, label='Picchi')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Tempo (s)', fontsize=14)
    ax.set_ylabel('Accelerazione X', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    return fig

# --- Streamlit app ---

st.title("Analisi Segnale Accelerazione X con Parametri Interattivi")

uploaded_file = st.file_uploader("Carica file Excel (.xlsx)", type=['xlsx'])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    # Prepara tempo in secondi (da microsecondi)
    df['Time'] = (df['SampleTimeFine'] - df['SampleTimeFine'].iloc[0]) / 1e6

    st.subheader("Anteprima dati grezzi")
    st.dataframe(df.head())

    # Slider per parametri
    SOGLIA_MIN = st.slider("Soglia minima per trovare minimi", min_value=-5.0, max_value=0.0, value=-1.5, step=0.1)
    MIN_SEP_TIME = st.slider("Tempo minimo separazione minimi (s)", min_value=0.1, max_value=2.0, value=0.45, step=0.05)
    TEMPO_RANGE = st.slider("Intervallo tempo per affinare punti (s)", min_value=0.1, max_value=2.0, value=0.735, step=0.05)
    START_TIME = st.number_input("Tempo inizio segmento (s)", value=393.0, step=1.0)
    END_TIME = st.number_input("Tempo fine segmento (s)", value=411.0, step=1.0)

    FS = 60.0  # frequenza campionamento
    LOWCUT, HIGHCUT = 0.5, 3.0

    # Filtraggio segnale
    b, a = butter_bandpass(LOWCUT, HIGHCUT, FS)
    df['Acc_X_filt'] = filtfilt(b, a, df['Acc_X'].values)

    st.subheader("Segnale filtrato completo")
    fig1 = plot_with_ticker(df['Time'].values, df['Acc_X_filt'].values,
                           "Segnale filtrato - Acc_X (band-pass 0.5-3Hz)",
                           "Segnale filtrato")
    st.pyplot(fig1)

    # Estrai segmento scelto
    segment = df[(df['Time'] >= START_TIME) & (df['Time'] <= END_TIME)]
    t = segment['Time'].values
    raw = segment['Acc_X'].values
    filt = segment['Acc_X_filt'].values

    st.subheader(f"Segmento selezionato: da {START_TIME}s a {END_TIME}s")
    fig2 = plot_with_ticker(t, filt, "Segmento filtrato", "Filtrato")
    st.pyplot(fig2)

    # Trova minimi e picchi nel segmento filtrato
    min_idx = trova_minimi(filt, SOGLIA_MIN, FS, MIN_SEP_TIME)
    max_idx = trova_picchi_massimi(filt, min_idx)

    st.subheader("Minimi e Picchi trovati (segmento filtrato)")
    fig3 = plot_with_ticker(t, filt, "Segmento filtrato con minimi e picchi",
                           "Filtrato", min_idx=min_idx, max_idx=max_idx)
    st.pyplot(fig3)

    # Affina i punti sul segnale grezzo
    min_aff = affina_indici(min_idx, t, raw, TEMPO_RANGE, mode='min')
    max_aff = affina_indici(max_idx, t, raw, TEMPO_RANGE, mode='max')

    st.subheader("Minimi e Picchi affinati sul segnale grezzo")
    fig4 = plot_with_ticker(t, raw, "Segmento grezzo con minimi e picchi affinati",
                           "Grezz0", min_idx=min_aff, max_idx=max_aff)
    st.pyplot(fig4)

    # Grafico finale time-zeroed rispetto al primo picco affinato
    if len(max_aff) > 0:
        t0 = t[max_aff[0]]
        time_zero = t - t0
        valori_picchi = raw[max_aff]
        time_zeroed_picchi = time_zero[max_aff]

        df_finale = pd.DataFrame({
            'Tempo_zero': time_zeroed_picchi,
            'Accelerazione': valori_picchi
        }).drop_duplicates(subset='Tempo_zero')

        st.subheader("Grafico finale time-zeroed")
        fig5, ax = plt.subplots(figsize=(12,4))
        ax.plot(time_zero, raw, label="Acc_X", alpha=0.6)
        ax.scatter(df_finale['Tempo_zero'], df_finale['Accelerazione'], color='red', marker='o', s=70, label='Picchi affinati')
        ax.scatter(time_zero[min_aff], raw[min_aff], color='green', marker='x', s=70, label='Minimi affinati')
        ax.set_ylabel("Accelerazione X")
        ax.set_xlabel("Tempo (s) zeroed")
        ax.set_title("Segnale e punti locali affinati (time zeroed)", fontsize=16)
        ax.legend()
        ax.grid(True)
        st.pyplot(fig5)

        # Pulsante per scaricare Excel con dati picchi affinati
        st.download_button(
            label="Scarica dati finali picchi affinati",
            data=df_finale.to_excel(index=False).encode('utf-8'),
            file_name="picchi_affinati_time_zero.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("Non sono stati trovati picchi affinati da visualizzare o scaricare.")
else:
    st.info("Carica un file Excel con i dati per iniziare l'analisi.")
