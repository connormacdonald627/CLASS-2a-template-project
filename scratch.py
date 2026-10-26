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
    n_frames: int = (len(pipeline.get_signal()) - pipeline.window_size) // pipeline.hop_size

    lib_centroids_hz: numpy.ndarray = librosa.feature.spectral_centroid(
        y=pipeline.get_signal(), sr=pipeline.sample_rate, 
        n_fft=pipeline.window_size, hop_length=pipeline.hop_size, center=False
    )[0]
    lib_centroids: numpy.ndarray = lib_centroids_hz / (pipeline.sample_rate / 2)
    
    lib_onsets: numpy.ndarray = librosa.onset.onset_strength(
        y=pipeline.get_signal(), sr=pipeline.sample_rate, 
        n_fft=pipeline.window_size, hop_length=pipeline.hop_size, center=False
    )

    pipeline.position = 0
    onset: list = []
    centroid: list = []
    librosa_onset: list = []
    librosa_centroid: list = []
    
    for i in range(n_frames):
        f: dict[str, float|str] = pipeline.get_features()
        onset.append(f["onset"])
        centroid.append(f["centroid"])
        
        librosa_onset.append(lib_onsets[i])
        librosa_centroid.append(lib_centroids[i])
        
    return onset, centroid, librosa_onset, librosa_centroid


custom_onset: list
custom_centroid: list
librosa_onset: list
librosa_centroid: list
custom_onset, custom_centroid, librosa_onset, librosa_centroid = analyze_window()

results: list[tuple[str, list]] = [
    ("custom centroid", custom_centroid),
    ("librosa centroid", librosa_centroid),
    ("custom onset", custom_onset),
    ("librosa onset", librosa_onset)
]

print("Feature: mean, min, max")
for name, data in results:
    print(f"{name}: mean={numpy.mean(data):.2f}, min={numpy.min(data):.2f}, max={numpy.max(data):.2f}")

loud_pos: int = len(pipeline.get_signal()) // 2
quiet_pos: int = 0

pipeline.position = loud_pos
loud: dict[str, float|str] = pipeline.get_features()
print(f"\nLOUD  Centroid: {loud['centroid']:.2f}, Onset: {loud['onset']:.2f}, RMS: {loud['amplitude']:.2f}, Peak: {loud['peak']:.2f}")

pipeline.position = quiet_pos
quiet: dict[str, float|str] = pipeline.get_features()
print(f"QUIET Centroid: {quiet['centroid']:.2f}, Onset: {quiet['onset']:.2f}, RMS: {quiet['amplitude']:.2f}, Peak: {quiet['peak']:.2f}")


def top_onset(amount: int = 5) -> None:
    n_frames: int = len(custom_onset)
    times: list[float] = [i * pipeline.hop_size / pipeline.sample_rate for i in range(n_frames)]

    positions: list[int] = list(range(len(custom_onset)))
    positions.sort(key=lambda i: custom_onset[i], reverse=True)
    
    top_times: list[float] = []
    for i in positions:
        time = times[i]
        if all(abs(time - t) >= 0.5 for t in top_times):
            top_times.append(time)
        if len(top_times) == amount:
            break
            
    top_times.sort()

    lib_times: numpy.ndarray = librosa.onset.onset_detect(y=pipeline.get_signal(), sr=pipeline.sample_rate, units="time")
    
    print("")
    print(f"top {amount} onsets: {[round(t, 2) for t in top_times]}")
    print(f"librosa detected {len(lib_times)} onsets across the whole track.")
    
    matches: int = 0
    for custom_time in top_times:
        closest: float = 0.0
        smallest_diff: float = 999.0
        
        for lib_time in lib_times:
            diff: float = abs(lib_time - custom_time)
            if diff < smallest_diff:
                smallest_diff = diff
                closest = lib_time
                
        if smallest_diff <= 0.1:
            matches += 1
            print(f"{custom_time:.2f}s matched (Librosa {closest:.2f}s)")
        else:
            print(f"{custom_time:.2f}s did not match (Closest {closest:.2f}s)")
            
    print(f"Result: {matches}/{len(top_times)} matched")
    
top_onset()

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
#     print(f"Sample: {data.index((seconds, rms)) + 1}\t{seconds}\t{rms:.2f}")

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




