import textwrap


VOCODER_INFO = textwrap.dedent("""\
    **Vocoder for audio synthesis:**
    
    **HiFi-GAN:**
    - **Arch overview:** HiFi-GAN + Hn-NSF. ( RVC's og vocoder )
    - **COMPATIBILITY:** Offline: RVC, Fork, Applio. Streaming: W-okada, Vonovox
    
    **RefineGAN:**
    - **Arch overview:** ParallelResBlocks + AdaIN + Hn-NSF
    - **COMPATIBILITY:** This Fork or Applio ( As for rt-vc, vonovox beta supports it. )
    
    **RingFormer:**
    - **Arch overview:** A hybrid Resblocks+Conformer-Based Vocoder + RingAttention + Hn-NSF.
    - **COMPATIBILITY:** This Fork ( As for rt-vc, 'Vonovox' supports it. )
    
    **ALPEX-GAN:**
    - **Arch overview:** ResBlocks + FGSS/Cyc-noise + Laplacian pyramid downsampling & injection.
    - **COMPATIBILITY:** This Fork ( No rt-vc clients support it atm. )
    
    **NOTES:**
    **( Offline = Static inference/Covers, Streaming = Real-Time voice changers )**
    **( RingFormer Requires min. RTX 30 series [ At least Ampere microarchitecture ] )**
    **( Each Vocoder and its supported sample rates require appropriate pretrained models )**
""")


DATASET_TRUNCATION_INFO = textwrap.dedent("""\
<br/>

### **SmartCutter / Truncated-Audio approach:**

#### Requirements:
- Your dataset is a **" fused dataset "** type ( Means, you concatenated / joined up all smaller samples / chunks into 1 continuous audio file )
- You keep "SmartCutter" enabled ( My machine-learning solution to 'silence truncation'. )
<br/> ⚠ ( **or** you made sure your audio is silence-truncated. )

<br/>

### **Otherwise you likely should go with:**

- SmartCutter:  Set it to "Disabled"
- Audio cutting:  "Automatic"
<br/> ⚠ ( Yet I'd still recommend to go with approach the first approach. It provides ***much better and more consistent*** results.

<br/>

` NOTES ` <br/>
0. Remember, a really good dataset makes 50% of the model creation workflow. If you use awful datasets, don't expect miracle results.

1. If your set has major peak / consistency issues, I recommend reading up on " Peak taming compression ".

2. Generally.. you shouldn't tweak these default settings unless you know you're doing it.

3. The only exception ( sub-point 2 ) would be for " DC / high-pass filtering " and " Noise Reduction " ~ Read their description.

4. If SmartCutter causes issues for your set ( cut words and so on. ), fallback to classical silence truncation.
<br/> ( You can use Audacity for that. )

""")


DATASET_FORMAT_INFO = textwrap.dedent("""\
    **Sliced audio storage format**

    Controls which format is used for storing sliced audio files.

    **WAV:** Uncompressed PCM audio. Much larger files but preserves the exact waveform.
    - Recommended if your source files are 32-bit float (e.g., preprocessed audio).

    **FLAC:** Lossless compressed audio codec. Much smaller files and safe for training.
    - Useful if storage space is limited.
""")


RESAMPLER_INFO = textwrap.dedent("""\
- **librosa:** Uses SoX resampler
( SoXr set to VHQ by default. )

- **ffmpeg:** Uses SW resampler
( Windowed Sinc filter with Blackman-Nuttall window. ) 

**Both are fine, but SoXr has better anti-aliasing.**
""")


SMARTCUTTER_INFO = textwrap.dedent("""\
**Machine-Learning solution to silence truncation**
Created especially with this fork in mind.
- Automatically truncates the silences ( Ensuring min. 100ms spacings )
- Doesn't damage word-tails or inter-phonetic gaps ( unlike gating )
- Truncated areas are automatically replaced by pure silence
( in case of noise-contamination between words or sentences. )
- Tries to heavily respect breathing.

⚠ **Due to technical limitations, multi-spk processing will be slower.**
""")


NORMALIZATION_INFO = textwrap.dedent("""\
- **none:** Disabled
( Select this if the files are already normalized. )
- **post_peak:** Peak Post-Normalization
( Peak [ * 0.95] norm of each sliced segment. )
""")


AUDIO_FILE_SLICING_INFO = textwrap.dedent("""\
**Audio file slicing-method selection:**
- **Skip:** if the files are already pre-sliced and properly normalized.
- **Simple:** If your dataset is already silence-truncated.
- **Automatic:** for automatic silence detection and slicing around it.
**It is advised to go with 'SmartCutter' approach.**
**( PS. Automatic is pretty crap. I advise against it unless your set's clean and you can't bother truncating it. )**
""")


PITCH_EXTRACTION_INFO = textwrap.dedent("""\
**Pitch extraction algorithm to use for the audio conversion:**

**RMVPE:** The default algorithm, recommended for most cases.
- The fastest, very robust to noise. Can tolerate harmonies / layered vocals to some degree.

**CREPE:** Better suited for truly clean audio.
- Is slower and way worse in handling noise. Can provide different / softer-ish results.

**CREPE-TINY:** Smaller / lighter variant of CREPE.
- Performs worse than 'full' ( standard crepe ) but is way lighter on hardware.
""")


BATCH_SIZE_INFO = textwrap.dedent("""\
**[ TOO LARGE BATCH SIZE CAN LEAD TO VRAM 'OOM' ISSUES. ]**

Bigger batch size:
- Promotes smoother, more stable gradients.
- Can be beneficial in cases where your dataset is big and diverse.
- Can lead to early overtraining or flat / ' stuck ' graphs on small datasets.
- Generalization might be worsened.
- **Favors faster learning rate.**

Smaller batch size:
- Promotes noisier, less stable gradients.
- More suitable when your dataset is small, less diverse or repetitive.
- Can lead to instability / divergence or noisy as hell graphs.
- Generalization / High-pitch performance might be improved.
- **Favors slower learning rate.**
""")


SPECTRAL_LOSS_INFO = textwrap.dedent("""\
- **L1 Mel Loss:** Standard L1 mel spectrogram loss - **Safe default.**
- **Multi-Scale Mel Loss:** Mel spectrogram loss that utilizes multiple-scales - **Results vary.**
- **Multi-Res STFT Loss:** STFT Spec. based loss that utilizes multiple-resolutions
( **EXPERIMENTAL.** )
""")


LR_SCHEDULER_INFO = textwrap.dedent("""\
- **exp decay:** Decays the lr exponentially - **Safe default.**
**( 'step' variant decays per step, 'epoch' per epoch. )**
- **cosine annealing:** Cosine annealing schedule - **Optional alternative.**
- **none:** No scheduler - **For debugging or developing.**
""")


KL_ANNEALING_INFO = textwrap.dedent("""\
**Enables cyclic KL loss annealing for training.**
- Might potentially mitigate overfitting on smaller datasets.
- Generally should help with convergence.

**(EXPERIMENTAL)**
""")

KL_ANNEALING_CYCLE_INFO = textwrap.dedent("""\
Determines the duration of each repeating annealing cycle.
Limited testing showed 3 epochs is the most optimal,
but you can experiment for yourself.
**( Duration in epochs )**
""")


TSTP_INFO = textwrap.dedent("""\
Enables 'TSTP' ( Might be potentially useful for small datasets. )
Once encoders loss ( kl ) reaches '0.1':

- Freezes: Encoders, Flow, Spk emb
- Speeds up lr decay by 50% ( Exponential lr decay only. )
**(EXPERIMENTAL)**

""")