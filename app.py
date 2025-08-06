import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import butter, filtfilt, find_peaks
from io import BytesIO

# --- Funzioni Helper ---

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut/nyq, highcut/nyq], btype='band')

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
    ax.plot(time, signal, label=label, color='gray' if 'Originale' in title else 'blue', alpha=0.5 if 'Originale' in title else 1)
    if min_idx is not None:
        ax.scatter(time[min_idx], signal[min_idx], color='green', marker='x', s=50, label='Minimi')
    if max_idx is not None:
        ax.scatter(time[max_idx], signal[max_idx], color='red', marker='o', s=50, label='Picchi')
    ax.set_title(title)
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('Accelerazione X')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

# --- Streamlit app ---

st.title("Analisi Segnale Accelerazione X")

uploaded_file = st.file_uploader("Carica file Excel", type=['xlsx'])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df['Time'] = (df['SampleTimeFine'] - df['SampleTimeFine'].iloc[0]) / 1e6

    st.subheader("Dati grezzi")
    st.dataframe(df.head())

    FS = 60.0
    LOWCUT, HIGHCUT = 0.5, 3.0
    SOGLIA_MIN = -1.5
    MIN_SEP_TIME = 0.45
    TEMPO_RANGE = 0.735
    START_TIME, END_TIME = 393, 411

    # Segnale originale completo
    st.subheader("Segnale originale completo")
    fig1 = plot_with_ticker(df['Time'].values, df['Acc_X'].values, "Segnale originale - Acc_X", "Originale")
    st.pyplot(fig1)

    # Segnale filtrato
    b, a = butter_bandpass(LOWCUT, HIGHCUT, FS)
    df['Acc_X_filt'] = filtfilt(b, a, df['Acc_X'].values)

    st.subheader("Segnale filtrato completo")
    fig2 = plot_with_ticker(df['Time'].values, df['Acc_X_filt'].values, "Segnale filtrato completo", "Filtrato (Band‑pass)")
    st.pyplot(fig2)

    # Segmento selezionato
    segment = df[(df['Time']>=START_TIME) & (df['Time']<=END_TIME)]
    t = segment['Time'].values
    raw = segment['Acc_X'].values
    filt = segment['Acc_X_filt'].values

    st.subheader("Segmento: originale vs filtrato")
    fig3, ax = plt.subplots(figsize=(12,4))
    ax.plot(t, raw, color='gray', alpha=0.5, label='Originale')
    ax.plot(t, filt, color='blue', label='Filtrato')
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('Accelerazione X')
    ax.set_title("Segmento: originale vs filtrato")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig3)

    st.subheader("Segmento filtrato completo")
    fig4 = plot_with_ticker(t, filt, "Segmento filtrato completo", "Acc_X_filt")
    st.pyplot(fig4)

    # Trova minimi e picchi
    min_idx = trova_minimi(filt, SOGLIA_MIN, FS, MIN_SEP_TIME)
    max_idx = trova_picchi_massimi(filt, min_idx)

    st.subheader("Segmento filtrato con minimi e picchi")
    fig5 = plot_with_ticker(t, filt, "Segmento filtrato con minimi e picchi",
                           "Acc_X_filt", min_idx=min_idx, max_idx=max_idx)
    st.pyplot(fig5)

    # Affinamento
    min_aff = affina_indici(min_idx, t, raw, TEMPO_RANGE, mode='min')
    max_aff = affina_indici(max_idx, t, raw, TEMPO_RANGE, mode='max')

    st.subheader("Segmento grezzo con minimi e picchi affinati")
    fig6 = plot_with_ticker(t, raw, "Segmento grezzo con minimi e picchi affinati",
                           "Acc_X", min_idx=min_aff, max_idx=max_aff)
    st.pyplot(fig6)

    # Grafico finale time-zeroed
    t0 = t[max_aff[0]] if len(max_aff) > 0 else 0
    time_zero = t - t0
    time_zeroed_picchi = time_zero[max_aff]
    valori_picchi = raw[max_aff]

    df_finale = pd.DataFrame({
        'Tempo_zero': time_zeroed_picchi,
        'Accelerazione': valori_picchi
    }).drop_duplicates(subset='Tempo_zero')

    st.subheader("Grafico finale time-zeroed")
    fig7, ax = plt.subplots(figsize=(12,4))
    ax.plot(time_zero, raw, label="Acc_X", alpha=0.6)
    ax.scatter(df_finale['Tempo_zero'], df_finale['Accelerazione'], color='red', marker='o', s=50, label='Picchi affinati')
    ax.scatter(time_zero[min_aff], raw[min_aff], color='green', marker='x', s=50, label='Minimi affinati')
    ax.set_ylabel("Acc_X")
    ax.set_title("Segnale e punti locali affinati (time zeroed)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig7)

    # Download file Excel
    output = BytesIO()
    df_finale.to_excel(output, index=False)
    output.seek(0)

    st.download_button(
        label="Scarica dati finali picchi affinati",
        data=output,
        file_name="picchi_affinati_time_zero.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
