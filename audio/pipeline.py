import librosa
import numpy
import os

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

    def get_features(self) -> dict:
        """Extract numerical features (amplitude at minimum) from the current
        window of the loaded signal, advance the pipeline's internal read
        position accordingly, and return them as a dict[str, float].

        This return value is passed directly into engine.update().
        """
        current_data: dict = {
            "amplitude": self.get_rms(),
        }

        self.position += self.window_size

        return current_data