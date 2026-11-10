"""Interactive 8-bit SEM TIFF editor with local supervised adjustment."""
import csv
import os
import sys
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from threading import Event

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from sklearn.ensemble import RandomForestRegressor

SLIDERS = {
    "gamma": ("Gamma × 100", 10, 400, 100),
    "black": ("Black point", 0, 254, 0),
    "white": ("White point", 1, 255, 255),
    "clahe": ("CLAHE clip × 10", 0, 80, 0),
    "grid": ("Tiles per axis", 2, 16, 8),
    "median": ("Median radius (px)", 0, 3, 0),
    "diffusion": ("Diffusion iterations", 0, 20, 0),
    "kappa": ("Diffusion threshold", 1, 100, 25),
    "unsharp": ("Unsharp amount (%)", 0, 300, 0),
    "radius": ("Unsharp sigma × 10 (px)", 3, 50, 10),
}

DATASET = Path("sem_ml_dataset.csv")
MIN_SAMPLES = 5
FEATURE_NAMES = ([f"hist_{i:03d}" for i in range(256)] + ["laplacian_variance"] +
                 [f"variance_r{r}_c{c}" for r in range(4) for c in range(4)])
CSV_HEADER = FEATURE_NAMES + [f"slider_{key}" for key in SLIDERS]


def extract_features(gray):
    """Describe the original uint8 frame, independent of current adjustments."""
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or not gray.size or gray.dtype != np.uint8:
        raise ValueError("Feature extraction requires a non-empty 8-bit grayscale image.")
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel().astype(np.float64)
    hist /= hist.sum()
    lap_var = cv2.Laplacian(gray, cv2.CV_32F, ksize=3).var(dtype=np.float64)
    height, width = gray.shape
    local = []
    for r in range(4):
        y = height * r // 4
        for c in range(4):
            x = width * c // 4
            # Reuse edge pixels when a dimension is smaller than four.
            tile = gray[y:max(y + 1, height * (r + 1) // 4),
                        x:max(x + 1, width * (c + 1) // 4)]
            local.append(tile.var(dtype=np.float64))
    return np.r_[hist, lap_var, local]


@contextmanager
def _dataset_lock():
    """Serialize local app instances across atomic CSV replacements."""
    try:
        with Path(str(DATASET) + ".lock").open("a+b") as lock:
            if os.name == "nt":
                import msvcrt
                if os.fstat(lock.fileno()).st_size == 0:
                    lock.write(b"\0")
                    lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            yield  # Closing the descriptor releases the lock, including on errors.
    except OSError as exc:
        raise ValueError(f"Could not access {DATASET.resolve()}: {exc}") from exc


def _feature_vector(values):
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(vector).all():
        raise ValueError("Expected 273 finite source-image features.")
    return vector


def _target_vector(values):
    targets = np.asarray(values, dtype=np.float64)
    low, high = _slider_bounds()
    if (targets.shape != low.shape or not np.isfinite(targets).all()
            or np.any(targets != np.floor(targets)) or np.any(targets < low)
            or np.any(targets > high)):
        raise ValueError("Invalid slider coordinates.")
    return targets.astype(int)


def _slider_bounds():
    return np.array([(spec[1], spec[2]) for spec in SLIDERS.values()]).T


def _dataset_rows():
    if not DATASET.exists():
        return []
    with DATASET.open(encoding="utf-8", newline="") as source:
        reader = csv.reader(source, strict=True)
        try:
            if next(reader, None) != CSV_HEADER:
                raise ValueError(f"Unexpected header in {DATASET}. Check its version or restore a backup.")
            rows = []
            for row in reader:
                features = _feature_vector(row[:len(FEATURE_NAMES)])
                targets = _target_vector(row[len(FEATURE_NAMES):])
                rows.append([*features, *targets])
            return rows
        except (ValueError, csv.Error) as exc:
            raise ValueError(f"{DATASET}, line {reader.line_num}: {exc}") from exc


def log_sample(features, settings_dict):
    """Append one complete approved example and return the new sample count."""
    features = _feature_vector(features)
    if set(settings_dict) != set(SLIDERS):
        raise ValueError("Missing or unknown slider coordinates.")
    targets = _target_vector([settings_dict[key] for key in SLIDERS])
    with _dataset_lock():
        rows = _dataset_rows()
        rows.append([*features, *targets])
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=DATASET.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            try:
                writer = csv.writer(tmp)
                writer.writerow(CSV_HEADER)
                writer.writerows(rows)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp.close()
                os.replace(tmp_path, DATASET)
            finally:
                tmp.close()
                tmp_path.unlink(missing_ok=True)
    return len(rows)


def predict_settings(features):
    """Fit a native multi-output forest to the current approved examples."""
    features = _feature_vector(features)
    with _dataset_lock():
        rows = _dataset_rows()
    if len(rows) < MIN_SAMPLES:
        raise ValueError(f"Auto-Adjust needs {MIN_SAMPLES} approved examples. "
                         f"Logged {len(rows)}. Log {MIN_SAMPLES - len(rows)} more to train.")
    data = np.asarray(rows, dtype=np.float64)
    X, y = data[:, :len(FEATURE_NAMES)], data[:, len(FEATURE_NAMES):]
    low, high = _slider_bounds()
    forest = RandomForestRegressor(n_estimators=96, random_state=42, n_jobs=1)
    forest.fit(X, (y - low) / (high - low))
    estimates = low + forest.predict(features.reshape(1, -1))[0] * (high - low)
    coords = np.clip(np.floor(estimates + 0.5), low, high).astype(int)
    return dict(zip(SLIDERS, map(int, coords)))



def sync_training_data():
    """Commit only the dataset and push normally; preserve local data on failure."""
    try:
        with _dataset_lock():
            if not DATASET.is_file():
                return "Log a good image first to create the training CSV."
            directory = DATASET.resolve().parent
            env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")

            def git(*args):
                result = subprocess.run(
                    ["git", "-C", str(directory), *args],
                    capture_output=True, text=True, timeout=60, env=env,
                )
                if result.returncode:
                    detail = result.stderr.strip() or result.stdout.strip()
                    raise ValueError(detail or "Git command failed.")
                return result.stdout.strip()

            git("rev-parse", "--show-toplevel")
            branch = git("symbolic-ref", "--quiet", "--short", "HEAD")
            git("remote", "get-url", "origin")
            # Retry also pushes an earlier local commit after an offline failure.
            git("add", "--", DATASET.name)
            if git("diff", "--cached", "--name-only", "--", DATASET.name):
                git("commit", "--only", "-m", "Record approved SEM training examples",
                    "--", DATASET.name)
            git("push", "origin", f"HEAD:refs/heads/{branch}")
        return ""
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return f"Saved locally; GitHub sync pending. {exc}"


def read_tiff(path):
    """Read one 8-bit TIFF frame, preserving original scalar gray values."""
    with open(path, "rb") as stream:
        header = stream.read(4)
    if header not in (b"II\x2a\x00", b"MM\x00\x2a", b"II\x2b\x00", b"MM\x00\x2b"):
        raise ValueError("Choose a TIFF image.")
    if cv2.imcount(str(path)) != 1:
        raise ValueError("Choose a readable, single-page TIFF.")
    data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if data is None or data.dtype != np.uint8:
        raise ValueError("This editor requires an 8-bit-per-channel TIFF.")
    if data.ndim == 3 and data.shape[2] in (3, 4):
        mode = cv2.COLOR_BGR2GRAY if data.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        data = cv2.cvtColor(data, mode)
    if data.ndim != 2:
        raise ValueError("Choose a grayscale or RGB TIFF.")
    data.setflags(write=False)
    return data


def diffuse(gray, iterations, kappa, check):
    """Four-neighbor Perona–Malik diffusion with zero-flux boundaries."""
    u = gray.astype(np.float32)
    for _ in range(iterations):
        check()
        change = np.zeros_like(u)
        for axis in (0, 1):
            view = u if axis == 0 else u.T
            accum = change if axis == 0 else change.T
            delta = np.diff(view, axis=0)
            flux = delta * np.exp(-np.square(delta / kappa))
            accum[:-1] += flux
            accum[1:] -= flux
        u += 0.2 * change
    return np.uint8(np.rint(np.clip(u, 0, 255)))


def process(gray, settings, check=lambda: None):
    """Always start from the raw matrix. Zero-strength filters are bypassed."""
    s = settings
    out = gray
    check()
    if s["median"]:
        out = cv2.medianBlur(out, 2 * s["median"] + 1)
    if s["diffusion"]:
        out = diffuse(out, s["diffusion"], s["kappa"], check)
    check()
    if s["black"] >= s["white"]:
        raise ValueError("White point must exceed black point.")
    if (s["black"], s["white"], s["gamma"]) != (0, 255, 100):
        x = (np.arange(256, dtype=np.float64) - s["black"]) / (s["white"] - s["black"])
        lut = np.rint(255 * np.clip(x, 0, 1) ** (100 / s["gamma"])).astype(np.uint8)
        out = cv2.LUT(out, lut)
    check()
    if s["clahe"]:
        tiles = (min(s["grid"], out.shape[1]), min(s["grid"], out.shape[0]))
        out = cv2.createCLAHE(s["clahe"] / 10, tiles).apply(out)
    check()
    if s["unsharp"]:
        blurred = cv2.GaussianBlur(out, (0, 0), s["radius"] / 10)
        amount = s["unsharp"] / 100
        out = cv2.addWeighted(out, 1 + amount, blurred, -amount, 0)
    return out


def export_tiff(path, matrix):
    """Encode beside the destination so a failed write leaves it intact."""
    fd, temp = tempfile.mkstemp(suffix=".tif", dir=path.parent)
    os.close(fd)
    try:
        if not cv2.imwrite(temp, matrix):
            raise OSError("OpenCV could not encode the TIFF.")
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


class Signals(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, str)


class Job(QtCore.QRunnable):
    def __init__(self, function):
        super().__init__()
        self.function = function
        self.signals = Signals()

    def run(self):
        try:
            result = self.function()
        except Exception as exc:
            self.signals.finished.emit(None, str(exc) or type(exc).__name__)
        else:
            self.signals.finished.emit(result, "")


class Canvas(QtWidgets.QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene_model = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene_model)
        self.item = self.scene_model.addPixmap(QtGui.QPixmap())
        self.setDragMode(self.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(self.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QtGui.QColor("#12151b"))
        self.setMinimumSize(140, 140)

    def show_matrix(self, matrix, shape, fit):
        height, width = matrix.shape
        gray = QtGui.QImage(matrix.data, width, height, matrix.strides[0],
                            QtGui.QImage.Format.Format_Grayscale8)
        self.item.setPixmap(QtGui.QPixmap.fromImage(gray))
        self.item.setTransform(QtGui.QTransform.fromScale(shape[1] / width, shape[0] / height))
        self.setSceneRect(self.item.sceneBoundingRect())
        if fit:
            self.fit()

    def fit(self):
        if not self.item.pixmap().isNull():
            self.fitInView(self.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        if self.item.pixmap().isNull() or not event.angleDelta().y():
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        if 0.001 < self.transform().m11() * factor < 128:
            self.scale(factor, factor)
        event.accept()


class Dashboard(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SEM Micro-graph Editor")
        self.resize(1180, 760)
        self.raw = self.path = self.matrix = None
        self.sliders, self.jobs = {}, set()
        self.revision = 0
        self.shown = -1
        self.render_cancel = None
        self.first_frame = False
        self.busy = False
        self.pool = QtCore.QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.render)

        bar = self.addToolBar("Image")
        bar.setMovable(False)
        self.open_action = bar.addAction("Open TIFF", self.open_file)
        self.open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.save_action = bar.addAction("Save TIFF As", self.save_file)
        self.save_action.setShortcut(QtGui.QKeySequence.StandardKey.SaveAs)
        self.canvas = Canvas()
        bar.addAction("Fit", self.canvas.fit)
        self.reset_action = bar.addAction("Reset", self.reset)
        split = QtWidgets.QSplitter()
        self.setCentralWidget(split)
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.addWidget(self.heading("Adjustments"))
        for name, keys in [("Tone", ("black", "white", "gamma")),
                           ("Local contrast", ("clahe", "grid")),
                           ("Denoise", ("median", "diffusion", "kappa")),
                           ("Sharpen", ("unsharp", "radius"))]:
            toggle = QtWidgets.QToolButton()
            toggle.setText(name)
            toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            toggle.setCheckable(True)
            toggle.setChecked(True)
            toggle.setArrowType(QtCore.Qt.ArrowType.DownArrow)
            panel = QtWidgets.QWidget()
            stack = QtWidgets.QVBoxLayout(panel)
            for key in keys:
                label, low, high, default = SLIDERS[key]
                row = QtWidgets.QHBoxLayout()
                row.addWidget(QtWidgets.QLabel(label))
                value = QtWidgets.QLabel(str(default))
                row.addStretch()
                row.addWidget(value)
                slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
                slider.setRange(low, high)
                slider.setValue(default)
                slider.setAccessibleName(label)
                slider.valueChanged.connect(value.setNum)
                slider.valueChanged.connect(lambda _, k=key: self.changed(k))
                slider.sliderReleased.connect(self.schedule)
                self.sliders[key] = slider
                stack.addLayout(row)
                stack.addWidget(slider)
            toggle.toggled.connect(panel.setVisible)
            toggle.toggled.connect(lambda on, t=toggle: t.setArrowType(
                QtCore.Qt.ArrowType.DownArrow if on else QtCore.Qt.ArrowType.RightArrow))
            left_layout.addWidget(toggle)
            left_layout.addWidget(panel)
        left_layout.addStretch()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        self.adjustments = scroll
        split.addWidget(scroll)
        split.addWidget(self.canvas)
        hub = QtWidgets.QWidget()
        controls = QtWidgets.QVBoxLayout(hub)
        controls.addWidget(self.heading("ML Control Hub"))
        hint = QtWidgets.QLabel("Approve the current settings to train the local adjustment model.")
        hint.setWordWrap(True)
        controls.addWidget(hint)
        self.log_button = QtWidgets.QPushButton("Log Good Image")
        self.auto_button = QtWidgets.QPushButton("Auto-Adjust")
        self.log_button.clicked.connect(lambda: self.learn(False))
        self.auto_button.clicked.connect(lambda: self.learn(True))
        controls.addWidget(self.log_button)
        controls.addWidget(self.auto_button)
        self.sync_button = QtWidgets.QPushButton("Sync Training Data")
        self.sync_button.clicked.connect(self.sync_dataset)
        controls.addWidget(self.sync_button)
        self.sync_status = QtWidgets.QLabel(
            "Good-image examples are saved locally and pushed to GitHub.")
        self.sync_status.setWordWrap(True)
        self.sync_status.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        controls.addWidget(self.sync_status)
        controls.addStretch()
        split.addWidget(hub)
        split.setChildrenCollapsible(False)
        split.setStretchFactor(1, 1)
        split.setSizes([255, 710, 215])
        self.statusBar().showMessage("Open a single-page TIFF to begin.")
        self.refresh()

    @staticmethod
    def heading(text):
        label = QtWidgets.QLabel(text)
        label.setObjectName("heading")
        return label

    def refresh(self):
        ready = self.raw is not None and not self.busy
        current = ready and self.shown == self.revision
        self.open_action.setEnabled(not self.busy)
        self.adjustments.setEnabled(ready)
        self.reset_action.setEnabled(ready)
        self.save_action.setEnabled(current)
        self.log_button.setEnabled(current)
        self.auto_button.setEnabled(ready)
        self.sync_button.setEnabled(not self.busy and DATASET.is_file())

    def start(self, function, done):
        job = Job(function)
        self.jobs.add(job)

        def finish(result, error):
            self.jobs.discard(job)
            done(result, error)

        job.signals.finished.connect(finish, QtCore.Qt.ConnectionType.QueuedConnection)
        self.pool.start(job)

    def exclusive(self, message, function, done):
        self.busy = True
        self.refresh()
        self.statusBar().showMessage(message)

        def finish(result, error):
            self.busy = False
            if error:
                self.statusBar().showMessage(error)
                QtWidgets.QMessageBox.warning(self, "SEM Editor", error)
            else:
                done(result)
            self.refresh()

        self.start(function, finish)

    def open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open TIFF", "", "TIFF (*.tif *.tiff *.TIF *.TIFF)")
        if not path:
            return

        def loaded(raw):
            self.raw, self.path = raw, Path(path)
            self.setWindowTitle(f"{self.path.name} — SEM Micro-graph Editor")
            self.matrix = None
            self.canvas.item.setPixmap(QtGui.QPixmap())
            self.first_frame = True
            self.reset()

        self.exclusive("Reading TIFF…", lambda: read_tiff(path), loaded)

    def settings(self):
        return {key: slider.value() for key, slider in self.sliders.items()}

    def apply_settings(self, values):
        for key, value in values.items():
            self.sliders[key].setValue(value)
        self.schedule()

    def reset(self):
        self.apply_settings({key: spec[3] for key, spec in SLIDERS.items()})

    def changed(self, key):
        black, white = self.sliders["black"], self.sliders["white"]
        if black.value() >= white.value():
            if key == "black":
                white.setValue(black.value() + 1)
            else:
                black.setValue(white.value() - 1)
        self.schedule()

    def schedule(self):
        self.revision += 1
        if self.render_cancel is not None:
            self.render_cancel.set()
        if not self.timer.isActive():
            self.timer.start(25)
        self.refresh()

    def render(self):
        if self.raw is None or self.render_cancel is not None:
            return
        raw, revision, settings = self.raw, self.revision, self.settings()
        preview = any(s.isSliderDown() for s in self.sliders.values())
        token = self.render_cancel = Event()

        def check():
            if token.is_set():
                raise InterruptedError("Superseded adjustment")

        def compute():
            matrix = raw
            if preview and max(raw.shape) > 1280:
                scale = 1280 / max(raw.shape)
                size = tuple(max(1, round(x * scale)) for x in raw.shape[::-1])
                matrix = cv2.resize(raw, size, interpolation=cv2.INTER_AREA)
            return process(matrix, settings, check)

        def display(matrix, error):
            self.render_cancel = None
            if revision != self.revision:
                self.render()
                return
            if error:
                self.statusBar().showMessage(f"Processing failed: {error}")
                return
            self.canvas.show_matrix(matrix, raw.shape, self.first_frame)
            self.first_frame = False
            if not preview:
                self.shown, self.matrix = revision, matrix
            if not self.busy:
                detail = "Draft preview" if preview else "Full resolution"
                self.statusBar().showMessage(f"{raw.shape[1]} × {raw.shape[0]} px   |   {detail}")
            self.refresh()

        self.start(compute, display)

    def save_file(self):
        suggested = self.path.with_name(self.path.stem + "_adjusted.tif")
        name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save TIFF As", str(suggested), "TIFF (*.tif *.tiff)")
        if not name:
            return
        path = Path(name)
        if path.suffix.lower() not in (".tif", ".tiff"):
            QtWidgets.QMessageBox.warning(self, "Save TIFF As", "Use a .tif or .tiff extension.")
            return
        if path.resolve() == self.path.resolve():
            QtWidgets.QMessageBox.warning(self, "Save TIFF As", "Choose a new path to preserve the raw TIFF.")
            return
        matrix = self.matrix
        self.exclusive("Writing grayscale TIFF…", lambda: export_tiff(path, matrix),
                       lambda _: self.statusBar().showMessage(f"Saved {path}"))

    def learn(self, predict):
        raw, settings = self.raw, self.settings()

        def compute():
            features = extract_features(raw)
            if predict:
                return predict_settings(features)
            count = log_sample(features, settings)
            return count, sync_training_data()

        def complete(result):
            if predict:
                self.apply_settings(result)
            else:
                count, error = result
                self.sync_status.setText(error or f"{count} examples saved and pushed to GitHub.")
                self.statusBar().showMessage(
                    "Saved locally; sync pending." if error else "Training CSV pushed to GitHub.")

        message = "Learning adjustments…" if predict else "Saving and syncing approved settings…"
        self.exclusive(message, compute, complete)

    def sync_dataset(self):
        def complete(error):
            self.sync_status.setText(error or "Training CSV pushed to GitHub.")
            self.statusBar().showMessage(
                "Saved locally; sync pending." if error else "Training CSV pushed to GitHub.")

        self.exclusive("Syncing training CSV…", sync_training_data, complete)

    def closeEvent(self, event):
        if self.busy:
            self.statusBar().showMessage("Finishing the active file or learning operation. Close again when it finishes.")
            event.ignore()
            return
        self.timer.stop()
        if self.render_cancel:
            self.render_cancel.set()
        super().closeEvent(event)


STYLE = """
QMainWindow, QWidget { background: #1e222b; color: white; font-size: 13px; }
QLabel#heading { font-size: 18px; font-weight: bold; padding: 8px 0; }
QToolBar, QStatusBar { background: #171b23; padding: 4px; }
QPushButton, QToolButton { background: #2b3545; border: 1px solid #445368;
    border-radius: 4px; padding: 6px; }
QPushButton:hover, QToolButton:hover { background: #304e71; }
QPushButton:disabled, QToolButton:disabled { color: #99a5b4; background: #272d36; }
QSlider::groove:horizontal { height: 4px; background: #424c5a; }
QSlider::sub-page:horizontal { background: #399dff; }
QSlider::handle:horizontal { width: 12px; margin: -4px 0; border-radius: 6px; background: #65b5ff; }
QSplitter::handle { background: #39414d; }
"""

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    dashboard = Dashboard()
    dashboard.show()
    sys.exit(app.exec())
