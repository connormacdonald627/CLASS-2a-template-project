import librosa
import numpy
import random
import matplotlib.pyplot as plt
from audio.pipeline import AudioPipeline

pipeline: AudioPipeline = AudioPipeline(file_path="audio/track1.mp3")
pipeline.load()
print("Sample Rate:", str(pipeline.sample_rate) + ",", "Duration:", str(round(pipeline.duration, 2)) + ",", "seconds", "Total Samples:", str(len(pipeline.signal)), "Array dtype:", str(pipeline.signal.dtype))

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

data:dict = {
    "max_amp": {
        "value": float("-inf"),
        "seconds": 0.0,
    },
    "min_amp": {
        "value": float("inf"),
        "seconds": 0.0,
    }
}

complete_windows: int = pipeline.num_windows
for i in range(0, complete_windows):
    pipeline.position = i * pipeline.window_size
    rms: float = round(pipeline.get_features()["amplitude"], 4)
    seconds = round(pipeline.position / pipeline.sample_rate, 2)

    data["max_amp"]["value"] = max(data["max_amp"]["value"], rms)
    if rms == data["max_amp"]["value"]:
        data["max_amp"]["seconds"] = seconds

    data["min_amp"]["value"] = min(data["min_amp"]["value"], rms)
    if rms == data["min_amp"]["value"]:
        data["min_amp"]["seconds"] = seconds

    print(f"Window: {i + 1}\t{seconds}\t{rms:.4f}")

print(f"Total Complete Windows: {complete_windows}")
print(f"Smallest RMS Amplitude: {data['min_amp']['value']:.4f} (At {data['min_amp']['seconds']:.2f} seconds), Largest RMS Amplitude: {data['max_amp']['value']:.4f} (At {data['max_amp']['seconds']:.2f} seconds)")


plt.plot(numpy.arange(len(pipeline.signal)) / pipeline.sample_rate, pipeline.signal)
plt.xlabel("Seconds")
plt.ylabel("Amplitude")
plt.title("Audio Signal")
plt.show()
