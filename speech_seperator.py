from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio
import gradio as gr
import ipdb
import librosa
import numpy as np
from IPython.display import display, Audio
import soundfile as sf
import torch

model = separator.from_hparams(source="speechbrain/sepformer-libri3mix", savedir='pretrained_models/sepformer-libri3mix')
audio_path="/home/xinying/Speaker2/project-speech_classification/dataset/2-MAJC0_FMKC0=overlay_SI1946.WAV"
# est_sources = model.separate_file(path=audio_path) 
batch, fs_file = torchaudio.load(audio_path)
est_sources = model.separate_batch(mix=batch)
threshold=0.005

def detect_silence_intervals(audio_file, zcr_threshold, centroid_threshold, rolloff_threshold, min_silence_duration):
    # 读取音频文件
    audio, sr = librosa.load(audio_file, sr=None)
    
    # 计算零交叉率
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=2048, hop_length=512)[0]
    
    # 计算频谱质心
    centroid = librosa.feature.spectral_centroid(audio, sr=sr, n_fft=2048, hop_length=512)[0]
    
    # 计算过零率
    rolloff = librosa.feature.spectral_rolloff(audio, sr=sr, n_fft=2048, hop_length=512, roll_percent=0.85)[0]
    
    energy = librosa.feature.rms(audio, frame_length=2048, hop_length=512)[0]
    
    # 根据阈值检测静音段的位置
    silence_intervals = []
    is_silence = []
    for i in range(len(zcr)):
        if  zcr[i] <= zcr_threshold:
            is_silence.append(1) #silence
            # start = librosa.frames_to_time(i, sr=sr)
        else:
            # is_silence = False
            is_silence.append(0)
            # end = librosa.frames_to_time(i, sr=sr)
            
            # if end - start >= min_silence_duration:
            #     silence_intervals.append((start, end))
    from scipy.ndimage import gaussian_filter1d
    smoothed_data = gaussian_filter1d(is_silence, sigma=1)
    
    
    # 找出连续的0的间隔
    test_intervals = []
    start = None
    for i, value in enumerate(smoothed_data):
        if value > 0.5 and start is None:
            # start = i
            test_start = i
            start = librosa.frames_to_time(i, sr=sr)
        elif value <= 0.5 and start is not None:
            # end = i - 1
            test_end=i-1
            end = librosa.frames_to_time(i-1, sr=sr)
            if (end-start) >= min_silence_duration:
                silence_intervals.append((start, end))
                test_intervals.append((test_start,test_end))
            start = None
    if start is not None:
        end = librosa.frames_to_time(len(smoothed_data)-1, sr=sr)
        silence_intervals.append((start, end))
    print(is_silence)
    print(smoothed_data)
    print(silence_intervals)
    print(test_intervals)



    # 创建一个布尔掩码，用于标记静音部分
    mask = np.ones_like(audio, dtype=bool)
    
    # 根据静音间隔标记静音部分
    for interval in silence_intervals:
        start_frame = int(interval[0] * sr)
        end_frame = int(interval[1] * sr)
        
        # 将静音部分的标记置为False
        mask[start_frame:end_frame] = False
        print(start_frame,end_frame)
    # 根据掩码移除静音部分
    new_audio = audio[mask]
    return silence_intervals, new_audio, sr





def gender_classification(audio):
    # ipdb.set_trace()
    f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    # f0 = f0[voiced_flag]  # 只考虑有声部分的基频
    if voiced_flag.any():
        f0 = f0[voiced_flag]  # 只考虑有声部分的基频
        avg_f0 = np.nanmean(f0)
    else:
        return -1 # 或者你可以设置一个默认值
    if avg_f0 <=170: #Male
        return 1
    elif avg_f0 > 170: #Female
        return 0
    else:
        return -1

# 转换数据类型为 float32
# normalized_data = normalized_data.astype(np.float32)
# 保存为 WAV 文件

def audio_separation(audio):
    sr, data=audio
    data_t=torch.from_numpy(data).float()
    sf.write("output.wav", data, sr)
    y, sr = librosa.load("output.wav", sr=None)
    # y=y[np.where(abs(y) > threshold)[0]].astype(np.float32)
    # sf.write('output.wav', y, sr)
    # est_sources = model.separate_batch(mix=data_t.unsqueeze(0))
    audio_file = 'output.wav'
    threshold = 0.01  # 设置能量阈值
    output_file = 'output.wav'  # 设置输出文件路径
    silence_intervals, new_audio, sr = detect_silence_intervals(audio_path, zcr_threshold=0.01, centroid_threshold=2000,
                                                         rolloff_threshold=0.1, min_silence_duration=1
)
    sf.write(output_file, new_audio, sr)

    est_sources = model.separate_file("output.wav")


    stds=[]
    genders=[]
    audios = []
    process_sr=8000
    for i in range(3):
        y=est_sources[:, :, i].detach().cpu().squeeze()
        # y=y[np.where(abs(y) > threshold)[0]].float()
        output_path = f"output{i}.wav"  # 保存的文件路径
        sf.write(output_path, y, process_sr)
        output_file = output_path 
        audio_file = output_path # 设置输出文件路径
        silence_intervals, new_audio, sr = detect_silence_intervals(audio_path, zcr_threshold=0.01, centroid_threshold=2000,
                                                         rolloff_threshold=0.1, min_silence_duration=1
)
        sf.write(output_file, new_audio, sr)
        # y, sr = librosa.load(output_path, sr=None)
        stds.append(new_audio.std())
        # genders.append(gender_classification(est_sources[:, :, i].numpy()))
        genders.append(gender_classification(new_audio))
        
        # ipdb.set_trace()
        # wav=est_sources[:, :, i]
        # normalized_data = wav / wav.max()
        
        mask = [1 if std > 0.08 else 0 for std in stds]
        for i in range(len(mask)):
            if mask[i] == 0:
                genders[i] = -1
        # audios.append(Audio(est_sources[:, :, i].detach().cpu().squeeze(), rate=8000)) 
        
    gender_map={0:"Female", 1:"Male", -1:"Unknown"}
    gender_text=f"Total ppls: {np.array(mask).sum()}, Gender: {gender_map[genders[0]],gender_map[genders[1]],gender_map[genders[2]]}"
    
    return stds,gender_text,silence_intervals,"output0.wav","output1.wav","output2.wav"

if __name__ == "__main__":
    
    demo = gr.Interface(
        fn=audio_separation,
        inputs=["audio"],
        outputs=["text","text","text",
                 "audio","audio","audio",
                ]
    )

    demo.launch()
# stds,genders=audio_separation(audio_path)
# print(stds,genders)