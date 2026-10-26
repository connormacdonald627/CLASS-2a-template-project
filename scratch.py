import librosa
import numpy
import random
import time
import matplotlib.pyplot as plt
from audio.pipeline import AudioPipeline

pipeline: AudioPipeline = AudioPipeline(file_path="audio/track1.mp3")
pipeline.load()
if pipeline.get_signal() is not None:
    print("Sample Rate:", str(pipeline.sample_rate) + ",", "Duration:", str(round(pipeline.track_data["duration"], 2)) + ",", "seconds", "Total Samples:", str(len(pipeline.get_signal())), "Array dtype:", str(pipeline.get_signal().dtype))

### ONSET AND CENTROID TESTING

def analyze_window() -> tuple:
    lib_centroids_hz = librosa.feature.spectral_centroid(
        y=pipeline.get_signal(), sr=pipeline.sample_rate, 
        n_fft=pipeline.window_size, hop_length=pipeline.hop_size, center=False
    )[0]
    lib_centroids = lib_centroids_hz / (pipeline.sample_rate / 2)
    
    lib_onsets = librosa.onset.onset_strength(
        y=pipeline.get_signal(), sr=pipeline.sample_rate, 
        n_fft=pipeline.window_size, hop_length=pipeline.hop_size, center=False
    )

    pipeline.position = 0
    onset = []
    centroid = []
    librosa_onset = []
    librosa_centroid = []
    
    for i in range(200):
        f = pipeline.get_features()
        onset.append(f["onset"])
        centroid.append(f["centroid"])
        
        librosa_onset.append(lib_onsets[i])
        librosa_centroid.append(lib_centroids[i])
        
    return onset, centroid, librosa_onset, librosa_centroid


def librosa_centroid_and_onset(window: numpy.ndarray, sample_rate: int) -> tuple:
    spectral_centroid = librosa.feature.spectral_centroid(y=window, sr=sample_rate)[0]
    onset_env = librosa.onset.onset_strength(y=window, sr=sample_rate)

    return spectral_centroid, onset_env

custom_onset, custom_centroid, librosa_onset, librosa_centroid = analyze_window()

results = [
    ("custom centroid", custom_centroid),
    ("librosa centroid", librosa_centroid),
    ("custom onset", custom_onset),
    ("librosa onset", librosa_onset)
]
print("")
print("Feature: mean, min, max")
for name, data in results:
    print(f"{name}: mean={numpy.mean(data):.4f}, min={numpy.min(data):.4f}, max={numpy.max(data):.4f}")
print("")

loud_pos: int = len(pipeline.get_signal()) // 2
quiet_pos: int = 0

pipeline.position = loud_pos
loud = pipeline.get_features()
print(f"LOUD  Centroid: {loud['centroid']:.2f}, Onset: {loud['onset']:.2f}, RMS: {loud['amplitude']:.4f}, Peak: {loud['peak']:.4f}")

pipeline.position = quiet_pos
quiet = pipeline.get_features()
print(f"QUIET Centroid: {quiet['centroid']:.2f}, Onset: {quiet['onset']:.2f}, RMS: {quiet['amplitude']:.4f}, Peak: {quiet['peak']:.4f}")


# position_from_seconds: float = pipeline.duration / 2
# pipeline.position = int(position_from_seconds * pipeline.sample_rate)
# peak: float = pipeline.get_peak()
# rms: float = pipeline.get_rms()
# print("Start sample (Seconds):", str(round(pipeline.position / pipeline.sample_rate, 2)) + ",", "Peak Amplitude:", str(round(peak, 2)) + ",", "RMS Amplitude:", str(round(rms, 2)))

# data: list[tuple[float, float]] = []
# selected_sample: tuple[float, float] | None = None
# random_position: float = random.randrange(0, pipeline.num_windows) * pipeline.window_size
# current_position: float = random_position
# for i in range(0, 15):
#     pipeline.position = int(current_position)
#     rms: float = round(pipeline.get_features()["amplitude"], 4)
#     seconds = round(pipeline.position / pipeline.sample_rate, 2)
#     data.append((round(seconds, 2), rms))
#     if(random.randrange(0, 15) == 0) and selected_sample is None:
#         selected_sample = (seconds, rms)

# data.sort(key=lambda x: x[0])
# print("Sample (N)\tSeconds\tAmplitude")
# for seconds, rms in data:
#     print(f"Sample: {data.index((seconds, rms)) + 1}\t{seconds}\t{rms:.4f}")

# print("Manual RMS Calculation:")
# if selected_sample:
#     print(f"Selected Sample: {data.index(selected_sample) + 1}\t{selected_sample[0]}\t{selected_sample[1]:.4f}")
#     position_from_seconds: int = int(selected_sample[0] * pipeline.sample_rate)
#     signal: numpy.ndarray = pipeline.signal[position_from_seconds:position_from_seconds + 1024]
#     manual_rms: float = numpy.sqrt(numpy.mean(numpy.square(signal)))
#     print(f"Manual RMS: {manual_rms:.4f}")

# data:dict = {
#     "max_amp": {
#         "value": float("-inf"),
#         "seconds": 0.0,
#     },
#     "min_amp": {
#         "value": float("inf"),
#         "seconds": 0.0,
#     }
# }

# complete_windows: int = pipeline.num_windows
# for i in range(0, complete_windows):
#     pipeline.position = i * pipeline.window_size
#     rms: float = round(float(pipeline.get_features()["amplitude"]), 4)
#     seconds = round(pipeline.position / pipeline.sample_rate, 2)

#     data["max_amp"]["value"] = max(data["max_amp"]["value"], rms)
#     if rms == data["max_amp"]["value"]:
#         data["max_amp"]["seconds"] = seconds

#     data["min_amp"]["value"] = min(data["min_amp"]["value"], rms)
#     if rms == data["min_amp"]["value"]:
#         data["min_amp"]["seconds"] = seconds

    # print(f"Window: {i + 1}\t{seconds}\t{rms:.4f}")

# print(f"Total Complete Windows: {complete_windows}")
# print(f"Smallest RMS Amplitude: {data['min_amp']['value']:.4f} (At {data['min_amp']['seconds']:.2f} seconds), Largest RMS Amplitude: {data['max_amp']['value']:.4f} (At {data['max_amp']['seconds']:.2f} seconds)")

# assert pipeline.signal is not None
# plt.plot(numpy.arange(len(pipeline.signal)) / pipeline.sample_rate, pipeline.signal)
# plt.xlabel("Seconds")
# plt.ylabel("Amplitude")
# plt.title("Audio Signal")
# plt.show()

# FRAMES: int = 200;
# def max_jump(window_size: int) -> tuple:
#     pipeline.position = 0
#     pipeline.window_size = window_size
#     amp: list = [pipeline.get_features()["amplitude"] for _ in range(FRAMES)]
#     jumps: list = [abs(amp[i] - amp[i - 1]) for i in range(1, len(amp))]
#     index: int = jumps.index(max(jumps))
#     return max(jumps), index + 1, amp[index], amp[index + 1]

# for window_size in (256, 1024, 4096):
#     jump: float
#     frame: int
#     before_jump: float
#     after_jump: float
#     jump, frame, before_jump, after_jump = max_jump(window_size)
#     print(f"Window size={window_size} at frame {frame}, max jump={jump:.4f}, before jump={before_jump:.4f}, after jump={after_jump:.4f}")




