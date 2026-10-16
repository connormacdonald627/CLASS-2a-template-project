import librosa
import numpy
import matplotlib.pyplot as plt
from audio.pipeline import AudioPipeline

pipeline: AudioPipeline = AudioPipeline(file_path="audio/track1.mp3")
pipeline.load()
print("Sample Rate:", str(pipeline.sample_rate) + ",", "Duration:", str(round(pipeline.duration, 2)) + ",", "seconds", "Total Samples:", str(len(pipeline.signal)), "Array dtype:", str(pipeline.signal.dtype))

position_from_seconds: float = pipeline.duration / 2
pipeline.position = int(position_from_seconds * pipeline.sample_rate)
peak: float = pipeline.get_peak()
rms: float = pipeline.get_rms()
print("Start sample (Seconds):", str(round(pipeline.position / pipeline.sample_rate, 2)) + ",", "Peak Amplitude:", str(round(peak, 2)) + ",", "RMS Amplitude:", str(round(rms, 2)))

plt.plot(numpy.arange(len(pipeline.signal)) / pipeline.sample_rate, pipeline.signal)
plt.xlabel("Seconds")
plt.ylabel("Amplitude")
plt.title("Audio Signal")
plt.show()