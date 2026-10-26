import librosa
import numpy
import os
import pygame
import typing


class AudioPipeline:
    """Loads an audio file and extracts features from it, frame by frame.

    Both load() and get_features() are stubs you must implement yourself —
    that's intentional, not an oversight. This unit is about the whole
    pipeline (data source -> feature extraction -> real-time rendering), not
    just feature extraction handed to you pre-solved.
    """

    def __init__(self, file_path: str, sample_rate: int = 44100) -> None:
        self.file_path: str = file_path
        self.sample_rate: int = sample_rate
        self.__signal: numpy.ndarray | None = None
        self.window_size: int = 1024
        self.position: int = 0
        self.num_windows: int = 0
        self.hop_size: int = 256
        self.play_audio = False
        self.debug = True
        self.band_limits: dict[str, int | float] = {
            "low_hz": 250,
            "high_hz": 2500,
            "low_bin_index": 0,
            "high_bin_index": 0,
            "bass_peak": 0.0,
            "mid_peak": 0.0,
            "treble_peak": 0.0,
        }
        self.track_data: dict[str, typing.Any] = {
            "stereo_buffer": None,
            "seconds": 0.0,
            "duration": 0.0,
        }
        self.__previous_magnitudes: numpy.ndarray | None = None
        self.__time_since_advance = 0.0

    def load(self) -> None:
        """Load the audio file at self.file_path via librosa, at
        self.sample_rate, and store the resulting signal on the instance.
        """
        if self.debug:
            print(f"Loading audio file: {self.file_path} at sample rate: {self.sample_rate}")

        if os.path.exists(self.file_path) == False:
            raise FileNotFoundError(f"Audio file not found: {self.file_path}")

        try:
            self.__signal, _ = librosa.load(self.file_path, sr=self.sample_rate)
        except Exception as e:
            print(f"Error loading audio file: {e}")
        if self.__signal is None or len(self.__signal) == 0:
            raise ValueError(f"Failed to load audio file: {self.file_path}")

        self.position = 0
        self.__previous_magnitudes = None
        self.__time_since_advance = 0.0

        self.num_windows = len(self.__signal) // self.window_size
        self.track_data["seconds"] = self.position / self.sample_rate
        self.track_data["duration"] = len(self.__signal) / self.sample_rate

        self.band_limits["low_bin_index"]  = (self.band_limits["low_hz"] * self.window_size / self.sample_rate)
        self.band_limits["high_bin_index"] = (self.band_limits["high_hz"] * self.window_size / self.sample_rate)
        self.band_limits["bass_peak"]      = self.__compute_band_peak("bass")
        self.band_limits["mid_peak"]       = self.__compute_band_peak("mid")
        self.band_limits["treble_peak"]    = self.__compute_band_peak("treble")

        self.__build_sound_buffer()

    def is_valid_window(self) -> bool:
        if self.__signal is None or len(self.__signal) == 0:
            return False
        return 0 <= self.position <= len(self.__signal) - self.window_size

    def get_signal(self) -> numpy.ndarray:
        if self.__signal is None:
            raise ValueError("Signal not loaded. Call load() first.")
        return self.__signal

    def get_window(self) -> numpy.ndarray:
        if not self.is_valid_window():
            raise ValueError("Invalid window position. Call load() first and ensure the position is within valid range.")

        signal: numpy.ndarray = self.get_signal()
        end_position = min(self.position + self.window_size, len(signal))
        window = signal[self.position:end_position]
        return window

    def get_rms(self) -> float:
        squared_signal: numpy.ndarray = numpy.square(self.get_window())
        rms: float = numpy.sqrt(numpy.mean(squared_signal))
        return rms

    def __compute_band_peak(self, band: str) -> float:
        """Computes the approximate peak value for the specified frequency
        band (bass, mid, treble) across the entire signal.
        """
        signal: numpy.ndarray = self.get_signal()
        low_bin_index: int = int(self.band_limits["low_bin_index"])
        high_bin_index: int = int(self.band_limits["high_bin_index"])
        min_freq: float | None = 0.0
        max_freq: float | None = 0.0

        if band == "bass":
            min_freq, max_freq = 0, low_bin_index
        elif band == "mid":
            min_freq, max_freq = low_bin_index, high_bin_index
        elif band == "treble":
            min_freq, max_freq = high_bin_index, None
        else:
            raise ValueError(f"Invalid band: {band}")

        peak: float = float("-inf")
        stride: int = self.window_size * 8

        for position in range(0, len(signal) - self.window_size, stride):
            window: numpy.ndarray = signal[position:position + self.window_size] * numpy.hanning(self.window_size)
            spectrum: numpy.ndarray = numpy.abs(numpy.fft.rfft(window))
            value: float = float(numpy.log1p(numpy.sum(spectrum[min_freq:max_freq] ** 2)))
            peak = max(peak, value)

        return peak

    def get_peak(self) -> float:
        peak = numpy.max(numpy.abs(self.get_window()))
        return float(peak)

    def get_bands(self, spectrum: numpy.ndarray) -> dict[str, float | str]:
        low_bin_index: int = int(self.band_limits["low_bin_index"])
        high_bin_index: int = int(self.band_limits["high_bin_index"])
        bass: float = float(numpy.log1p(numpy.sum(spectrum[0:low_bin_index] ** 2)))
        mid: float = float(numpy.log1p(numpy.sum(spectrum[low_bin_index:high_bin_index] ** 2)))
        treble: float = float(numpy.log1p(numpy.sum(spectrum[high_bin_index:] ** 2)))

        bass /= self.band_limits["bass_peak"] if self.band_limits["bass_peak"] > 0 else 1.0
        mid /= self.band_limits["mid_peak"] if self.band_limits["mid_peak"] > 0 else 1.0
        treble /= self.band_limits["treble_peak"] if self.band_limits["treble_peak"] > 0 else 1.0

        bass = min(max(bass, 0.0), 1.0)
        mid = min(max(mid, 0.0), 1.0)
        treble = min(max(treble, 0.0), 1.0)

        dominant_band: str = (
            "bass" if bass >= mid and bass >= treble
            else "mid" if mid >= bass and mid >= treble
            else "treble"
        )

        return {"bass": bass, "mid": mid, "treble": treble, "dominant_band": dominant_band}

    def get_centroid(self, spectrum: numpy.ndarray, bins: numpy.ndarray) -> float:
        magnitude_sum: float = float(numpy.sum(spectrum))

        if magnitude_sum == 0:
            return 0.0

        centroid: float = float(numpy.sum(bins * spectrum) / magnitude_sum)

        nyquist: int = self.sample_rate // 2
        return centroid / nyquist

    def get_onset(self, spectrum: numpy.ndarray) -> float:
        onset: float = 0.0
        if self.__previous_magnitudes is not None:
            dist: numpy.ndarray = numpy.maximum(spectrum - self.__previous_magnitudes, 0.0)
            magnitude_sum: float = float(numpy.sum(spectrum))
            onset = float(numpy.sum(dist) / magnitude_sum) if magnitude_sum > 0 else 0.0

        self.__previous_magnitudes = spectrum
        return onset

    def fourier_transform(self) -> dict[str, int | float | str | numpy.ndarray]:
        if not self.is_valid_window():
            raise ValueError("Invalid window position. Call load() first and ensure the position is within valid range.")
        current_window: numpy.ndarray = self.get_window() * numpy.hanning(self.window_size)
        spectrum: numpy.ndarray = numpy.abs(numpy.fft.rfft(current_window))

        num_bins: int = len(spectrum)
        bins: numpy.ndarray = numpy.fft.rfftfreq(
            self.window_size,
            d=1 / self.sample_rate,
        )
        
        return {
            "total_bins": num_bins,
            "bins": bins,
            "spectrum": spectrum,
        }

    def __build_sound_buffer(self) -> None:
        signal: numpy.ndarray = self.get_signal()
        if pygame.mixer.get_init() != (self.sample_rate, -16, 2):
            pygame.mixer.quit()
            pygame.mixer.init(
                frequency=self.sample_rate,
                size=-16,
                channels=2,
                buffer=128,
            )

        pcm_samples = (numpy.clip(signal, -1.0, 1.0) * 32767).astype(numpy.int16)
        self.track_data["stereo_buffer"] = numpy.ascontiguousarray(
            numpy.column_stack((pcm_samples, pcm_samples))
        )

    def __play_window_audio(self, step: int) -> None:
        stereo_buffer = self.track_data["stereo_buffer"]
        if self.__signal is None or stereo_buffer is None:
            return

        channel = pygame.mixer.Channel(0)
        chunk_end = min(self.position + step, len(stereo_buffer))
        if chunk_end <= self.position:
            return

        sound = pygame.sndarray.make_sound(stereo_buffer[self.position:chunk_end])
        if not channel.get_busy():
            channel.play(sound)
        elif channel.get_queue() is None:
            channel.queue(sound)

    def get_features(self, delta_time: float = 0.0) -> dict[str, float | str]:
        """Extract numerical features (amplitude at minimum) from the current
        window of the loaded signal, advance the pipeline's internal read
        position accordingly, and return them as a dict[str, float|str].

        This return value is passed directly into engine.update().
        """

        self.position = self.position % (len(self.get_signal()) - self.window_size + 1)
        self.track_data["seconds"] = self.position / self.sample_rate

        fourier_result: dict = self.fourier_transform()
        bins, spectrum = fourier_result["bins"], fourier_result["spectrum"]
        amplitude: float = self.get_rms()
        peak: float = self.get_peak()
        bands: dict[str, float | str] = self.get_bands(spectrum)
        centroid: float = self.get_centroid(spectrum, bins)
        onset: float = self.get_onset(spectrum)

        current_data: dict[str, float | str] = {
            "amplitude": amplitude,
            "peak": peak,
            "bass": bands["bass"],
            "mid": bands["mid"],
            "treble": bands["treble"],
            "dominant_band": bands["dominant_band"],
            "centroid": centroid,
            "onset": onset,
        }
        if self.debug:
            print(
                f"[{self.track_data['seconds']:.2f}s/{self.track_data['duration']:.2f}s]    \t"
                f"amplitude={amplitude:.2f} peak={peak:.2f}     \t"
                f"bass={bands['bass']:.2f} mid={bands['mid']:.2f} treble={bands['treble']:.2f} dominant_band={bands['dominant_band']}   \t"
                f"centroid={centroid:.2f} onset={onset:.2f}     \t"
            )

        step: int = min(max(self.hop_size, 0), self.window_size)

        if delta_time <= 0.0:
            if self.play_audio:
                self.__play_window_audio(step)
            self.position += step
        else:
            step_seconds: float = step / self.sample_rate
            self.__time_since_advance += delta_time
            steps: int = 0
            while self.__time_since_advance >= step_seconds:
                self.__time_since_advance -= step_seconds
                steps += 1
            if steps > 0:
                if self.play_audio:
                    self.__play_window_audio(step * steps)
                self.position += step * steps

        return current_data