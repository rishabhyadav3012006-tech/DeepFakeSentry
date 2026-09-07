import matplotlib.pyplot as plt
import numpy as np

def plot_forensic_spectrogram(
    times: np.ndarray,
    mel_freqs: np.ndarray,
    S_mel_db: np.ndarray,
    cutoff_res=None,
    silence_res=None,
    discontinuity_res=None,
    show_cutoff: bool = True,
    show_silence: bool = True,
    show_discontinuity: bool = True
):
    """
    Renders a high-resolution dark-themed Mel Spectrogram with interactive forensic anomaly overlays.
    Returns a Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')

    # Extents for imshow
    extent = [times[0], times[-1], mel_freqs[0], mel_freqs[-1]]
    
    # Render Mel Spectrogram
    im = ax.imshow(
        S_mel_db,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap='magma',
        vmin=-80,
        vmax=0
    )
    
    cbar = fig.colorbar(im, ax=ax, format="%+2.0f dB", pad=0.02)
    cbar.ax.yaxis.set_tick_params(color='#8B949E')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#8B949E')

    # 1. Overlay: High-Frequency Cutoff
    if show_cutoff and cutoff_res:
        cutoff_detected, cutoff_freq, collapse_pct, cutoff_mask, cutoff_details = cutoff_res
        if cutoff_detected or cutoff_freq < mel_freqs[-1] * 0.95:
            # Draw horizontal cutoff line
            ax.axhline(
                y=cutoff_freq,
                color='#FF3366',
                linestyle='--',
                linewidth=2.0,
                label=f'Frequency Cutoff ({cutoff_freq:.0f} Hz)'
            )
            # Fill region above cutoff
            ax.fill_between(
                [times[0], times[-1]],
                cutoff_freq,
                mel_freqs[-1],
                color='#FF3366',
                alpha=0.20,
                hatch='//'
            )

    # 2. Overlay: Robotic Silence Pattern
    if show_silence and silence_res:
        silence_detected, intervals, reg_score, silence_details = silence_res
        if intervals:
            first_label = True
            for start_t, end_t in intervals:
                label = "Robotic Silence Interval" if first_label else ""
                ax.axvspan(
                    start_t,
                    end_t,
                    color='#7928CA',
                    alpha=0.35,
                    label=label
                )
                first_label = False

    # 3. Overlay: Spectral Discontinuities
    if show_discontinuity and discontinuity_res:
        discontinuity_detected, spike_times, distances, disc_details = discontinuity_res
        if spike_times:
            first_label = True
            for t_spike in spike_times:
                label = "Spectral Discontinuity" if first_label else ""
                ax.axvline(
                    x=t_spike,
                    color='#00DFD8',
                    linestyle=':',
                    linewidth=1.8,
                    label=label
                )
                ax.plot(
                    t_spike,
                    mel_freqs[-1] * 0.92,
                    marker='v',
                    color='#00DFD8',
                    markersize=8
                )
                first_label = False

    # Axis Labels & Styling
    ax.set_title("AUDIO MEL SPECTROGRAM & FORENSIC ANOMALY OVERLAYS", color="#F0F6FC", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Time (seconds)", color="#8B949E", fontsize=9.5)
    ax.set_ylabel("Frequency (Hz)", color="#8B949E", fontsize=9.5)
    ax.tick_params(colors="#8B949E", labelsize=8.5)
    
    for spine in ax.spines.values():
        spine.set_color('#30363D')

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles=handles,
            labels=labels,
            loc='upper right',
            facecolor='#161B22',
            edgecolor='#30363D',
            labelcolor='#F0F6FC',
            fontsize=8
        )

    plt.tight_layout()
    return fig
