import librosa
import numpy
import os
import typing

class AudioPipeline:
    """Loads an audio file and extracts features from it, frame by frame.

    Both load() and get_features() are stubs you must implement yourself —
    that's intentional, not an oversight. This unit is about the whole
    pipeline (data source -> feature extraction -> real-time rendering), not
    just feature extraction handed to you pre-solved.
    """

    def __init__(self, file_path: str, sample_rate: int = 22050) -> None:
        self.file_path: str = file_path
        self.sample_rate: int = sample_rate
        self.signal: numpy.ndarray | None = None
        self.window_size: int = 1024
        self.position: int = 0 
        self.duration: float = 0.0
        self.num_windows: int = 0
        self.hop_size: int = 512
        self.band_limits: dict[int] = {
            "low": 250,
            "high": 2000
        }

    def load(self) -> None:
        """Load the audio file at self.file_path via librosa, at
        self.sample_rate, and store the resulting signal on the instance.
        """
        print(f"Loading audio file: {self.file_path} at sample rate: {self.sample_rate}")

        if(os.path.exists(self.file_path) == False):
            raise FileNotFoundError(f"Audio file not found: {self.file_path}")

        try:
            self.signal, _ = librosa.load(self.file_path, sr=self.sample_rate)
        except Exception as e:
            print(f"Error loading audio file: {e}")
        if self.signal is None or len(self.signal) == 0:
            raise ValueError(f"Failed to load audio file: {self.file_path}")
        
        self.duration = len(self.signal) / self.sample_rate
        self.num_windows = len(self.signal) // self.window_size

    def get_window(self) -> numpy.ndarray:
        if self.signal is None:
            raise ValueError("Signal not loaded. Call load() first.")

        end_position = min(self.position + self.window_size, len(self.signal))
        window = self.signal[self.position:end_position]
        return window

    def get_rms(self,) -> float:
        rms: float = 0.0
        squared_signal = numpy.square(self.get_window())
        rms = numpy.sqrt(numpy.mean(squared_signal))
        return rms

    def get_peak(self) -> float:
        peak: float = 0.0
        peak = numpy.max(numpy.abs(self.get_window()))
        return peak

    def get_bin_index(self, target: float, bins: list) -> int:
        closest_index: int = numpy.argmin(numpy.abs(numpy.array(bins) - target))
        return closest_index

    def normalize(self, value: float, min_value: float, max_value: float) -> float:
        if max_value - min_value == 0:
            return 0.0
        normalized_value: float = (value - min_value) / (max_value - min_value)
        return normalized_value

    def fourier_transform(self) -> typing.Any:
        current_window: numpy.ndarray = self.get_window() * numpy.hanning(self.window_size)
        spectrum: numpy.ndarray = numpy.abs(numpy.fft.rfft(current_window))
        num_bins: int = len(spectrum)
        hz: int = self.sample_rate / self.window_size
        bins: list = [i * (hz) for i in range(num_bins)]
        
        low_bin_index: int = self.get_bin_index(self.band_limits["low"], bins)
        high_bin_index: int = self.get_bin_index(self.band_limits["high"], bins)

        max_magnitude: float = max(spectrum)

        bass = max(spectrum[0:low_bin_index])
        mid = max(spectrum[low_bin_index:high_bin_index])
        treble = max(spectrum[high_bin_index:])

        normalized_bands: dict = {
            "bass": self.normalize(bass, 0, max_magnitude),
            "mid": self.normalize(mid, 0, max_magnitude),
            "treble": self.normalize(treble, 0, max_magnitude)
        }

        dominant_band: str = "bass" if normalized_bands["bass"] > normalized_bands["mid"] and normalized_bands["bass"] > normalized_bands["treble"] else "mid" if normalized_bands["mid"] > normalized_bands["bass"] and normalized_bands["mid"] > normalized_bands["treble"] else "treble"

        result: dict = {"spectrum": spectrum, "bass": normalized_bands["bass"], "mid": normalized_bands["mid"], "treble": normalized_bands["treble"], "dominant_band": dominant_band}
        print(f"Fourier Transform Result: Bass: {result['bass']:.4f}, Mid: {result['mid']:.4f}, Treble: {result['treble']:.4f}, Dominant Band: {dominant_band}")
        return result


    def get_features(self) -> dict:
        """Extract numerical features (amplitude at minimum) from the current
        window of the loaded signal, advance the pipeline's internal read
        position accordingly, and return them as a dict[str, float].

        This return value is passed directly into engine.update().
        """

        if self.signal is None or len(self.signal) == 0:
            raise ValueError("Signal not loaded. Call load() first.")

        fourier_result: dict = self.fourier_transform()

        current_data: dict = {
            "amplitude": self.get_rms(),
            "bass": fourier_result["bass"],
            "mid": fourier_result["mid"],
            "treble": fourier_result["treble"],
            "centeroid": 0.0,
            "onset": 0.0
        }

        self.position += self.window_size
        if(self.position + self.window_size > len(self.signal)):
            self.position = 0

        return current_data