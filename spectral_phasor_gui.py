"""
spectral_phasor_gui.py  –  v6
pip install phasorpy statsmodels matplotlib numpy pandas PyQt5 scipy
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QPushButton, QLabel,
    QFileDialog, QGroupBox, QCheckBox, QDoubleSpinBox, QSpinBox,
    QComboBox, QMessageBox, QSizePolicy, QSplitter, QScrollArea,
    QAbstractItemView, QGridLayout, QColorDialog, QLineEdit, QDialog,
    QTabWidget, QStackedWidget, QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor

from statsmodels.nonparametric.smoothers_lowess import lowess

from phasorpy.phasor import (
    phasor_from_signal, phasor_threshold, phasor_to_polar,
)
from phasorpy.plot import PhasorPlot
from phasorpy.lifetime import (
    phasor_calibrate, phasor_to_apparent_lifetime,
)
from phasorpy.filter import phasor_filter_median
from phasorpy.cursor import mask_from_circular_cursor
from phasorpy.io import signal_from_fbd

# fbdfile es requerido para el tab de Lifetime. Instalar con: pip install fbdfile

def _read_fbd(path: str) -> dict:
    """
    Lee un archivo .fbd con frame=-1 (promedio de todos los frames).
    Guarda la señal completa (Y, X, H) por canal para calcular phasor pixel a pixel.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sig0 = signal_from_fbd(path, frame=-1, channel=0)

    channels = int(sig0.attrs.get("channels", 2))
    freq_MHz = float(sig0.attrs.get("frequency", 20.0))

    signals = []
    for ch in range(channels):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sig = signal_from_fbd(path, frame=-1, channel=ch)
        signals.append(sig.values)   # shape (Y, X, H)

    return {
        "path":     path,
        "label":    Path(path).stem,
        "signals":  signals,         # list of (Y, X, H) arrays, one per channel
        "freq_MHz": freq_MHz,
        "channels": channels,
    }


PALETTE = [
    "#1f77b4","#d62728","#2ca02c","#9467bd",
    "#8c564b","#e377c2","#555555","#bcbd22",
    "#17becf","#ff7f0e",
]
SPEC_TYPES = ["EX", "EM", "ABS"]


# ══════════════════════════════════════════════════════════════
#  CARGA CSV
# ══════════════════════════════════════════════════════════════

def load_cary_csv(path):
    try:
        df = pd.read_csv(path, header=1, usecols=[0, 1],
                         names=["lam", "val"], skip_blank_lines=True,
                         encoding="utf-8", on_bad_lines="skip")
        df["lam"] = pd.to_numeric(df["lam"], errors="coerce")
        df["val"] = pd.to_numeric(df["val"], errors="coerce")
        df = df.dropna()
        if len(df) >= 5:
            return df["lam"].values, df["val"].values
    except Exception:
        pass
    raw = pd.read_csv(path, header=None, skip_blank_lines=True,
                      encoding="utf-8", on_bad_lines="skip")
    start = 0
    for i, row in raw.iterrows():
        try:
            float(str(row.iloc[0]).replace(",", ".")); start = i; break
        except ValueError:
            continue
    df = raw.iloc[start:, :2].copy()
    df.columns = ["lam", "val"]
    df["lam"] = pd.to_numeric(df["lam"].astype(str).str.replace(",", "."), errors="coerce")
    df["val"] = pd.to_numeric(df["val"].astype(str).str.replace(",", "."), errors="coerce")
    df = df.dropna()
    return df["lam"].values, df["val"].values


def interp_to_grid(lam_list, data_list):
    lam_min = max(l.min() for l in lam_list)
    lam_max = min(l.max() for l in lam_list)
    n_pts   = min(len(l) for l in lam_list)
    lam_c   = np.linspace(lam_min, lam_max, n_pts)
    mat = np.zeros((n_pts, len(data_list)))
    for i, (l, d) in enumerate(zip(lam_list, data_list)):
        mat[:, i] = interp1d(l, d, bounds_error=False, fill_value=0.0)(lam_c)
    return lam_c, mat


# ══════════════════════════════════════════════════════════════
#  SPECTRUM ENTRY
# ══════════════════════════════════════════════════════════════

class SpectrumEntry:
    _counter = 0

    def __init__(self, path, spec_type="EM", label="", color="#1f77b4"):
        SpectrumEntry._counter += 1
        self.uid       = SpectrumEntry._counter
        self.path      = str(path)
        self.filename  = Path(path).name
        self.spec_type = spec_type
        self.label     = label or Path(path).stem
        self.color     = color
        self.lam_raw, self.data_raw = load_cary_csv(self.path)
        # background: dict con lam/data/name, o None
        self.bg        = None
        self.do_bg     = False
        # Rango λ — se define en LambdaDialog al cargar
        self.lam_min   = float(self.lam_raw.min())
        self.lam_max   = float(self.lam_raw.max())
        # Estado acumulativo de preprocesamiento
        self._do_loess   = False
        self._loess_frac = 0.1
        self._do_norm    = False
        self.lam       = self.lam_raw.copy()
        self.data      = self.data_raw.copy()

    def assign_bg(self, bg_path):
        lam, data      = load_cary_csv(bg_path)
        self.bg        = {"lam": lam, "data": data, "name": Path(bg_path).name}
        self.do_bg     = True   # activar inmediatamente

    def remove_bg(self):
        self.bg       = None
        self.do_bg    = False
        self._do_norm = False   # resetear norm al quitar BG

    def process(self):
        """Reconstruye desde raw aplicando los pasos según estado acumulativo."""
        lam  = self.lam_raw.copy()
        data = self.data_raw.copy()

        # 1. Cortar rango λ (definido en LambdaDialog al cargar)
        mask = (lam >= self.lam_min) & (lam <= self.lam_max)
        lam, data = lam[mask], data[mask]
        if len(lam) < 4:
            self.lam = lam; self.data = data; return

        # 2. Restar BG
        if self.do_bg and self.bg is not None:
            f_bg = interp1d(self.bg["lam"], self.bg["data"],
                            kind="linear", bounds_error=False, fill_value=0.0)
            data = data - f_bg(lam)
            data[data < 0] = 0.0

        # 3. LOESS
        if self._do_loess:
            res  = lowess(data, lam, frac=self._loess_frac, return_sorted=True)
            data = res[:, 1]
            data[data < 0] = 0.0

        # 4. Normalizar
        if self._do_norm:
            m = data.max()
            if m > 0: data = data / m

        self.lam  = lam
        self.data = data

    def get_phasor(self):
        """Calcula G,S desde la señal SIN LOESS (solo rango λ + BG + norm)."""
        lam  = self.lam_raw.copy()
        data = self.data_raw.copy()

        # 1. Rango λ
        mask = (lam >= self.lam_min) & (lam <= self.lam_max)
        lam, data = lam[mask], data[mask]
        if len(data) < 4:
            return np.nan, np.nan

        # 2. BG (sí aplica — es información real)
        if self.do_bg and self.bg is not None:
            f_bg = interp1d(self.bg["lam"], self.bg["data"],
                            kind="linear", bounds_error=False, fill_value=0.0)
            data = data - f_bg(lam)
            data[data < 0] = 0.0

        # 3. Normalizar (si está activo — no afecta el phasor en teoría,
        #    pero mantenemos consistencia con la señal usada)
        if self._do_norm:
            m = data.max()
            if m > 0: data = data / m

        # LOESS NO se aplica — el phasor debe calcularse con la señal raw

        if data.sum() == 0 or len(data) < 4:
            return np.nan, np.nan

        _, real, imag = phasor_from_signal(data[np.newaxis, :], axis=-1)
        return float(real[0]), float(imag[0])

    def list_label(self):
        bg_str = (f" │ BG✓ {self.bg['name']}" if self.bg and self.do_bg else
                  f" │ BG(off)" if self.bg else "")
        return f"[{self.spec_type}]  {self.label}{bg_str}"


# ══════════════════════════════════════════════════════════════
#  MATPLOTLIB CANVAS
# ══════════════════════════════════════════════════════════════

class MplCanvas(FigureCanvas):
    def __init__(self, figsize=(10, 4)):
        self.fig = Figure(figsize=figsize)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


# ══════════════════════════════════════════════════════════════
#  DIÁLOGO DE EDICIÓN
#  — BG se aplica inmediatamente al cargarlo, sin esperar OK —
# ══════════════════════════════════════════════════════════════

class EditDialog(QWidget):
    def __init__(self, entry: SpectrumEntry, on_change, parent=None):
        super().__init__(parent, Qt.Dialog | Qt.WindowCloseButtonHint)
        self.entry      = entry
        self.on_change  = on_change   # callback que redibuja
        self.chosen_color = entry.color
        self.setWindowTitle(f"Editar – {entry.filename}")
        self.setMinimumSize(500, 380)
        self._build()

    def _build(self):
        lay = QGridLayout(self)
        lay.setSpacing(8)

        lay.addWidget(QLabel("Tipo:"), 0, 0)
        self.cmb = QComboBox()
        self.cmb.addItems(SPEC_TYPES)
        self.cmb.setCurrentText(self.entry.spec_type)
        lay.addWidget(self.cmb, 0, 1, 1, 2)

        lay.addWidget(QLabel("Etiqueta:"), 1, 0)
        self.le = QLineEdit(self.entry.label)
        lay.addWidget(self.le, 1, 1, 1, 2)

        lay.addWidget(QLabel("Color:"), 2, 0)
        self.btn_col = QPushButton()
        self.btn_col.setStyleSheet(f"background:{self.entry.color};")
        self.btn_col.clicked.connect(self._pick_color)
        lay.addWidget(self.btn_col, 2, 1, 1, 2)

        # ── Background ──
        sep = QLabel("─── Background ───────────────────")
        sep.setStyleSheet("color:#888;font-size:10px;")
        lay.addWidget(sep, 3, 0, 1, 3)

        lay.addWidget(QLabel("Archivo BG:"), 4, 0)
        self.lbl_bg = QLabel(
            self.entry.bg["name"] if self.entry.bg else "Sin asignar")
        self.lbl_bg.setStyleSheet(
            "color:#2ca02c;" if self.entry.bg else "color:gray;")
        lay.addWidget(self.lbl_bg, 4, 1)

        btn_load_bg = QPushButton("📂 Cargar BG")
        btn_load_bg.clicked.connect(self._load_bg)
        lay.addWidget(btn_load_bg, 4, 2)

        # Checkbox — refleja do_bg en tiempo real
        self.chk_use_bg = QCheckBox("Restar este background")
        self.chk_use_bg.setChecked(self.entry.do_bg)
        self.chk_use_bg.setEnabled(self.entry.bg is not None)
        self.chk_use_bg.stateChanged.connect(self._toggle_bg)
        lay.addWidget(self.chk_use_bg, 5, 0, 1, 2)

        btn_rm_bg = QPushButton("✕ Quitar BG")
        btn_rm_bg.clicked.connect(self._remove_bg)
        lay.addWidget(btn_rm_bg, 5, 2)

        # ── LOESS ──
        sep2 = QLabel("─── Suavizado LOESS ──────────────")
        sep2.setStyleSheet("color:#888;font-size:10px;")
        lay.addWidget(sep2, 6, 0, 1, 3)

        self.chk_loess = QCheckBox("Aplicar LOESS")
        self.chk_loess.setChecked(self.entry._do_loess)
        self.chk_loess.stateChanged.connect(self._toggle_loess)
        lay.addWidget(self.chk_loess, 7, 0, 1, 2)

        lrow = QHBoxLayout()
        lrow.addWidget(QLabel("frac:"))
        self.sp_loess = QDoubleSpinBox()
        self.sp_loess.setRange(0.02, 0.8)
        self.sp_loess.setValue(self.entry._loess_frac if self.entry._loess_frac != 0.1
                               else 0.1)
        self.sp_loess.setSingleStep(0.01); self.sp_loess.setDecimals(2)
        self.sp_loess.valueChanged.connect(self._toggle_loess)
        lrow.addWidget(self.sp_loess)
        lw = QWidget(); lw.setLayout(lrow)
        lay.addWidget(lw, 7, 2)

        btn_ok = QPushButton("✔  Confirmar")
        btn_ok.setStyleSheet("font-weight:bold; padding:5px;")
        btn_ok.clicked.connect(self._confirm)
        lay.addWidget(btn_ok, 8, 0, 1, 3)

    def _pick_color(self):
        c = QColorDialog.getColor(QColor(self.entry.color), self)
        if c.isValid():
            self.chosen_color = c.name()
            self.btn_col.setStyleSheet(f"background:{c.name()};")

    def _load_bg(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccioná el archivo de background",
            "", "CSV/TXT (*.csv *.txt *.dat);;Todos (*.*)")
        if not path: return
        try:
            self.entry.assign_bg(path)   # setea do_bg=True inmediatamente
        except Exception as ex:
            QMessageBox.critical(self, "Error", str(ex)); return
        self.lbl_bg.setText(self.entry.bg["name"])
        self.lbl_bg.setStyleSheet("color:#2ca02c;")
        self.chk_use_bg.setEnabled(True)
        self.chk_use_bg.setChecked(True)
        self.on_change()   # redibujar YA, sin esperar OK

    def _toggle_bg(self, state):
        # El checkbox refleja do_bg en tiempo real
        self.entry.do_bg = (state == Qt.Checked) and self.entry.bg is not None
        self.on_change()

    def _toggle_loess(self, *_):
        self.entry._do_loess   = self.chk_loess.isChecked()
        self.entry._loess_frac = self.sp_loess.value()
        self.on_change()

    def _remove_bg(self):
        self.entry.remove_bg()
        self.lbl_bg.setText("Sin asignar")
        self.lbl_bg.setStyleSheet("color:gray;")
        self.chk_use_bg.setChecked(False)
        self.chk_use_bg.setEnabled(False)
        self.on_change()

    def _confirm(self):
        # Solo actualiza tipo/label/color — BG ya está aplicado en tiempo real
        self.entry.spec_type = self.cmb.currentText()
        self.entry.label     = self.le.text() or self.entry.label
        self.entry.color     = self.chosen_color
        self.on_change()
        self.close()


# ══════════════════════════════════════════════════════════════
#  DIÁLOGO DE RANGO λ — se abre al cargar cada archivo
# ══════════════════════════════════════════════════════════════

class LambdaDialog(QDialog):
    """Muestra el espectro crudo y deja elegir lam_min y lam_max."""

    def __init__(self, entry: SpectrumEntry, on_confirm, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.entry      = entry
        self.on_confirm = on_confirm
        self.setWindowTitle(f"Rango λ – {entry.filename}")
        self.setMinimumSize(620, 420)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)

        # Canvas para mostrar el espectro crudo
        self._fig = Figure(figsize=(6, 3), tight_layout=True)
        self._ax  = self._fig.add_subplot(1, 1, 1)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self._canvas)

        # Dibuja el espectro completo
        lam, data = self.entry.lam_raw, self.entry.data_raw
        self._ax.plot(lam, data, color=self.entry.color, lw=1.5)
        self._ax.set_xlabel("λ (nm)"); self._ax.set_ylabel("Intensidad")
        self._ax.set_title(f"{self.entry.label}  –  espectro completo")
        self._ax.spines[["top","right"]].set_visible(False)
        # Líneas verticales de corte (interactivas)
        self._vmin = self._ax.axvline(lam.min(), color="red",   lw=1.2, ls="--")
        self._vmax = self._ax.axvline(lam.max(), color="green", lw=1.2, ls="--")
        self._canvas.draw()

        # Controles de rango
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("λ mín:"))
        self.sp_min = QDoubleSpinBox()
        self.sp_min.setRange(lam.min(), lam.max())
        self.sp_min.setValue(lam.min())
        self.sp_min.setSuffix(" nm"); self.sp_min.setDecimals(1)
        self.sp_min.valueChanged.connect(self._update_lines)
        ctrl.addWidget(self.sp_min)

        ctrl.addSpacing(20)
        ctrl.addWidget(QLabel("λ máx:"))
        self.sp_max = QDoubleSpinBox()
        self.sp_max.setRange(lam.min(), lam.max())
        self.sp_max.setValue(lam.max())
        self.sp_max.setSuffix(" nm"); self.sp_max.setDecimals(1)
        self.sp_max.valueChanged.connect(self._update_lines)
        ctrl.addWidget(self.sp_max)
        lay.addLayout(ctrl)

        btn_ok = QPushButton("✔  Confirmar rango")
        btn_ok.setStyleSheet("font-weight:bold; background:#2ca02c; color:white; padding:5px;")
        btn_ok.clicked.connect(self._confirm)
        lay.addWidget(btn_ok)

    def _update_lines(self):
        self._vmin.set_xdata([self.sp_min.value(), self.sp_min.value()])
        self._vmax.set_xdata([self.sp_max.value(), self.sp_max.value()])
        self._canvas.draw_idle()

    def _confirm(self):
        lmin = self.sp_min.value()
        lmax = self.sp_max.value()
        if lmin >= lmax:
            QMessageBox.warning(self, "Error", "λ mín debe ser menor que λ máx.")
            return
        self.entry.lam_min = lmin
        self.entry.lam_max = lmax
        self.on_confirm()
        self.accept()


# ══════════════════════════════════════════════════════════════
#  DIÁLOGO DE ESTILO
# ══════════════════════════════════════════════════════════════

class StyleDialog(QDialog):
    """Ventana emergente para editar el estilo de todos los plots."""

    def __init__(self, fig_spec, fig_phasor, fig_lifetime, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🎨 Editar estilo de plots")
        self.setMinimumWidth(420)
        self.fig_spec     = fig_spec
        self.fig_phasor   = fig_phasor
        self.fig_lifetime = fig_lifetime
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)

        tabs = QTabWidget()
        lay.addWidget(tabs)

        # ── Tab Espectros ──
        w_sp = QWidget(); g_sp = QGridLayout(w_sp); g_sp.setSpacing(8)
        g_sp.setColumnStretch(1, 1)

        self._add_row(g_sp, 0,  "Fuente título:",   "sp_title_fs", 11, (6, 28))
        self._add_row(g_sp, 1,  "Fuente ejes:",     "sp_ax_fs",    10, (6, 22))
        self._add_row(g_sp, 2,  "Fuente leyenda:",  "sp_leg_fs",   8,  (6, 18))
        self._add_row(g_sp, 3,  "Grosor líneas:",   "sp_lw",       1.5,(0.5,6), is_float=True, step=0.5)
        self._add_combo(g_sp,4, "Posición leyenda:","sp_leg_loc",
                        ["best","upper right","upper left","lower right","lower left","outside"])
        tabs.addTab(w_sp, "📊  Espectros")

        # ── Tab Phasor espectral ──
        w_ph = QWidget(); g_ph = QGridLayout(w_ph); g_ph.setSpacing(8)
        g_ph.setColumnStretch(1, 1)

        self._add_row(g_ph, 0,  "Fuente título:",   "ph_title_fs", 11, (6, 28))
        self._add_row(g_ph, 1,  "Fuente leyenda:",  "ph_leg_fs",   8,  (6, 18))
        self._add_row(g_ph, 2,  "Tamaño markers:",  "ph_ms",       8,  (2, 20))
        self._add_combo(g_ph,3, "Posición leyenda:","ph_leg_loc",
                        ["best","upper right","upper left","lower right","lower left","outside"])
        tabs.addTab(w_ph, "🔵  Phasor espectral")

        # ── Tab Lifetime ──
        w_lt = QWidget(); g_lt = QGridLayout(w_lt); g_lt.setSpacing(8)
        g_lt.setColumnStretch(1, 1)

        self._add_row(g_lt, 0,  "Fuente título:",   "lt_title_fs", 12, (6, 28))
        self._add_row(g_lt, 1,  "Fuente leyenda:",  "lt_leg_fs",   9,  (6, 18))
        self._add_row(g_lt, 2,  "Tamaño markers:",  "lt_ms",       10, (2, 20))
        self._add_combo(g_lt,3, "Posición leyenda:","lt_leg_loc",
                        ["best","upper right","upper left","lower right","lower left","outside"])
        tabs.addTab(w_lt, "⏱  Phasor Lifetime")

        # Botones
        hb = QHBoxLayout()
        btn_apply = QPushButton("✔  Aplicar")
        btn_apply.setStyleSheet("font-weight:bold; padding:5px;")
        btn_apply.clicked.connect(self._apply)
        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.accept)
        hb.addWidget(btn_apply); hb.addWidget(btn_close)
        lay.addLayout(hb)

    def _add_row(self, grid, row, label, attr, default, rng, is_float=False, step=1):
        grid.addWidget(QLabel(label), row, 0)
        if is_float:
            sp = QDoubleSpinBox()
            sp.setRange(*rng); sp.setValue(default)
            sp.setSingleStep(step); sp.setDecimals(1)
        else:
            sp = QSpinBox()
            sp.setRange(*rng); sp.setValue(int(default))
        setattr(self, f"sp_{attr}", sp)
        grid.addWidget(sp, row, 1)

    def _add_combo(self, grid, row, label, attr, items):
        grid.addWidget(QLabel(label), row, 0)
        cmb = QComboBox(); cmb.addItems(items)
        setattr(self, f"cmb_{attr}", cmb)
        grid.addWidget(cmb, row, 1)

    def _apply_to_fig(self, fig, title_fs, ax_fs=None, leg_fs=8,
                      lw=None, ms=None, leg_loc="best"):
        for ax in fig.get_axes():
            ax.title.set_fontsize(title_fs)
            if ax_fs:
                ax.xaxis.label.set_fontsize(ax_fs)
                ax.yaxis.label.set_fontsize(ax_fs)
                ax.tick_params(labelsize=ax_fs - 1)
            if lw:
                for line in ax.get_lines():
                    if line.get_linestyle() not in ("None", "none", ""):
                        line.set_linewidth(lw)
            if ms:
                for line in ax.get_lines():
                    if line.get_marker() not in ("None", "none", ""):
                        line.set_markersize(ms)
            leg = ax.get_legend()
            if leg:
                if leg_loc == "outside":
                    ax.legend(fontsize=leg_fs, loc="upper left",
                              bbox_to_anchor=(1.01, 1))
                else:
                    ax.legend(fontsize=leg_fs, loc=leg_loc, framealpha=0.7)
        fig.tight_layout()

    def _apply(self):
        self._apply_to_fig(
            self.fig_spec,
            title_fs = self.sp_sp_title_fs.value(),
            ax_fs    = self.sp_sp_ax_fs.value(),
            leg_fs   = self.sp_sp_leg_fs.value(),
            lw       = self.sp_sp_lw.value(),
            leg_loc  = self.cmb_sp_leg_loc.currentText(),
        )
        self._apply_to_fig(
            self.fig_phasor,
            title_fs = self.sp_ph_title_fs.value(),
            leg_fs   = self.sp_ph_leg_fs.value(),
            ms       = self.sp_ph_ms.value(),
            leg_loc  = self.cmb_ph_leg_loc.currentText(),
        )
        self._apply_to_fig(
            self.fig_lifetime,
            title_fs = self.sp_lt_title_fs.value(),
            leg_fs   = self.sp_lt_leg_fs.value(),
            ms       = self.sp_lt_ms.value(),
            leg_loc  = self.cmb_lt_leg_loc.currentText(),
        )


# ══════════════════════════════════════════════════════════════
#  VENTANA PRINCIPAL
# ══════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectral Phasor Analysis")
        self.resize(1400, 860)

        # Ícono de la aplicación
        import os
        icon_path = Path(__file__).parent / "icon.png"
        if not icon_path.exists():
            # Cuando está empaquetado con PyInstaller
            icon_path = Path(getattr(sys, '_MEIPASS', '')) / "icon.png"
        if icon_path.exists():
            from PyQt5.QtGui import QIcon
            self.setWindowIcon(QIcon(str(icon_path)))
        self.entries: list[SpectrumEntry] = []
        self._color_idx = 0
        self._edit_dlg   = None
        self._spec_axes  = {}   # {spec_type: ax} — actualizado en _draw_spectra
        self._phasor_axes= {}   # {spec_type: ax} — actualizado en _draw_phasor
        self._undo_stack: list[list[dict]] = []

        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(350)
        self._timer.timeout.connect(self._redraw)

        self._build_ui()

    # ─────────────────────────────────────────────────────────
    #  UI
    # ─────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(0)

        hsplit = QSplitter(Qt.Horizontal)
        root.addWidget(hsplit)

        # ── Panel lateral con QStackedWidget ──
        self._left_stack = QStackedWidget()
        self._left_stack.setMinimumWidth(260)
        hsplit.addWidget(self._left_stack)

        # ── Panel 0: Espectros ──
        scroll_sp = QScrollArea(); scroll_sp.setWidgetResizable(True)
        scroll_sp.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_sp.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        panel_sp = QWidget()
        lv = QVBoxLayout(panel_sp); lv.setContentsMargins(6,6,6,6); lv.setSpacing(6)

        grp_f = QGroupBox("📂  Archivos de espectros")
        gf = QVBoxLayout(grp_f)
        btn_add = QPushButton("➕  Agregar espectros…")
        btn_add.clicked.connect(self.add_spectra); gf.addWidget(btn_add)
        hint = QLabel("💡 Doble clic para editar tipo, color y background.")
        hint.setWordWrap(True); hint.setStyleSheet("color:#555;font-size:10px;"); gf.addWidget(hint)
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_list.setMinimumHeight(150)
        self.file_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.file_list.setDefaultDropAction(Qt.MoveAction)
        self.file_list.itemDoubleClicked.connect(self.edit_entry)
        self.file_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.file_list.model().rowsMoved.connect(self._on_rows_moved)
        gf.addWidget(self.file_list)
        hb = QHBoxLayout()
        bd = QPushButton("🗑 Quitar sel."); bd.clicked.connect(self.remove_selected)
        bc = QPushButton("🧹 Limpiar todo"); bc.clicked.connect(self.clear_all)
        hb.addWidget(bd); hb.addWidget(bc); gf.addLayout(hb)
        lv.addWidget(grp_f)

        grp_pre = QGroupBox("⚗️  Preprocesamiento")
        gp = QGridLayout(grp_pre); gp.setSpacing(6); gp.setColumnStretch(1,1)
        lr = QHBoxLayout(); lr.addWidget(QLabel("LOESS frac:"))
        self.sp_loess_all = QDoubleSpinBox()
        self.sp_loess_all.setRange(0.02,0.8); self.sp_loess_all.setValue(0.1)
        self.sp_loess_all.setSingleStep(0.01); self.sp_loess_all.setDecimals(2)
        lr.addWidget(self.sp_loess_all)
        lrw = QWidget(); lrw.setLayout(lr); gp.addWidget(lrw, 0, 0, 1, 2)
        btn_loess = QPushButton("▶  Aplicar LOESS a todos")
        btn_loess.setStyleSheet("background:#2c7bb6;color:white;padding:4px;")
        btn_loess.clicked.connect(self._apply_loess_all); gp.addWidget(btn_loess, 1, 0, 1, 2)
        btn_norm = QPushButton("▶  Normalizar 0–1  (todos)")
        btn_norm.setStyleSheet("background:#2c7bb6;color:white;padding:4px;")
        btn_norm.clicked.connect(self._apply_norm); gp.addWidget(btn_norm, 2, 0, 1, 2)
        btn_rmbg = QPushButton("▶  Cargar BG único y restar a todos")
        btn_rmbg.setStyleSheet("background:#d62728;color:white;padding:4px;")
        btn_rmbg.setToolTip("Carga un único archivo de BG y lo resta a todos los espectros cargados")
        btn_rmbg.clicked.connect(self._remove_bg_all); gp.addWidget(btn_rmbg, 3, 0, 1, 2)
        btn_undo = QPushButton("↩  Deshacer último cambio")
        btn_undo.setStyleSheet("padding:4px;")
        btn_undo.clicked.connect(self._undo); gp.addWidget(btn_undo, 4, 0, 1, 2)
        lv.addWidget(grp_pre)

        grp_opt = QGroupBox("🎛️  Opciones de plot")
        go = QVBoxLayout(grp_opt)
        self.chk_mean = QCheckBox("Media ± SD (réplicas del mismo grupo)")
        self.chk_mean.setChecked(True); go.addWidget(self.chk_mean)
        lv.addWidget(grp_opt)

        grp_exp_sp = QGroupBox("💾  Exportar / Estilo")
        ge_sp = QVBoxLayout(grp_exp_sp)
        btn_sty_sp = QPushButton("🎨  Editar estilo…")
        btn_sty_sp.clicked.connect(self._open_style_dialog); ge_sp.addWidget(btn_sty_sp)
        for label, stype in [("💾  Espectro EX",       "EX"),
                              ("💾  Espectro EM",       "EM"),
                              ("💾  Espectro ABS",      "ABS"),
                              ("💾  Phasor espectral EX",  "PH_EX"),
                              ("💾  Phasor espectral EM",  "PH_EM"),
                              ("💾  Phasor espectral ABS", "PH_ABS")]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, t=stype: self._save_ax(t))
            ge_sp.addWidget(btn)
        lv.addWidget(grp_exp_sp)
        lv.addStretch()

        scroll_sp.setWidget(panel_sp)
        self._left_stack.addWidget(scroll_sp)   # index 0

        # ── Panel 1: Lifetime ──
        scroll_lt = QScrollArea(); scroll_lt.setWidgetResizable(True)
        scroll_lt.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_lt.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        panel_lt = QWidget()
        ll = QVBoxLayout(panel_lt); ll.setContentsMargins(6,6,6,6); ll.setSpacing(6)

        grp_lt_load = QGroupBox("📂  Archivos .fbd")
        gl = QVBoxLayout(grp_lt_load)
        btn_load_lt = QPushButton("📂  Agregar archivos .fbd…")
        btn_load_lt.clicked.connect(self._load_lifetime_fbd); gl.addWidget(btn_load_lt)
        self.lt_file_list = QListWidget(); self.lt_file_list.setMaximumHeight(110)
        self.lt_file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        gl.addWidget(self.lt_file_list)
        hlt = QHBoxLayout()
        btn_rm_lt = QPushButton("🗑 Quitar sel."); btn_rm_lt.clicked.connect(self._remove_lt_selected)
        btn_clr_lt = QPushButton("🧹 Limpiar"); btn_clr_lt.clicked.connect(self._clear_lifetime)
        hlt.addWidget(btn_rm_lt); hlt.addWidget(btn_clr_lt); gl.addLayout(hlt)

        btn_filter_sel = QPushButton("▶  Aplicar umbral + mediana al seleccionado")
        btn_filter_sel.setStyleSheet("background:#2c7bb6;color:white;padding:4px;")
        btn_filter_sel.clicked.connect(self._apply_filter_selected)
        gl.addWidget(btn_filter_sel)

        btn_clear_filt = QPushButton("✕  Quitar umbral/filtro del seleccionado")
        btn_clear_filt.clicked.connect(self._clear_filter_selected)
        gl.addWidget(btn_clear_filt)
        ll.addWidget(grp_lt_load)

        grp_lt_cal = QGroupBox("🎯  Calibración")
        gc = QGridLayout(grp_lt_cal); gc.setSpacing(6); gc.setColumnStretch(1,1)
        self.lbl_lt_cal = QLabel("Sin archivo de calibración")
        self.lbl_lt_cal.setStyleSheet("color:gray;font-size:10px;")
        self.lbl_lt_cal.setWordWrap(True); gc.addWidget(self.lbl_lt_cal, 0, 0, 1, 2)
        btn_load_cal = QPushButton("📂  Cargar referencia (.fbd)")
        btn_load_cal.clicked.connect(self._load_calibration); gc.addWidget(btn_load_cal, 1, 0, 1, 2)
        gc.addWidget(QLabel("τ ref. (ns):"), 2, 0)
        self.sp_lt_tau_ref = QDoubleSpinBox()
        self.sp_lt_tau_ref.setRange(0.01,100); self.sp_lt_tau_ref.setValue(2.5)
        self.sp_lt_tau_ref.setDecimals(3); self.sp_lt_tau_ref.setSingleStep(0.01)
        gc.addWidget(self.sp_lt_tau_ref, 2, 1)
        btn_cal = QPushButton("⚙  Calcular corrección")
        btn_cal.clicked.connect(self._compute_calibration); gc.addWidget(btn_cal, 3, 0, 1, 2)
        gc.addWidget(QLabel("Estado:"), 4, 0)
        self.lbl_lt_dphase = QLabel("–")
        self.lbl_lt_dphase.setStyleSheet("color:#2ca02c;font-size:10px;")
        self.lbl_lt_dphase.setWordWrap(True); gc.addWidget(self.lbl_lt_dphase, 4, 1)
        self.lbl_lt_dmod = QLabel("–")
        self.lbl_lt_dmod.setStyleSheet("color:#2ca02c;font-size:10px;")
        gc.addWidget(self.lbl_lt_dmod, 5, 1)
        ll.addWidget(grp_lt_cal)

        grp_lt_par = QGroupBox("⚙️  Parámetros")
        gp_lt = QGridLayout(grp_lt_par); gp_lt.setSpacing(6); gp_lt.setColumnStretch(1,1)
        gp_lt.addWidget(QLabel("Canal:"), 0, 0)
        self.sp_lt_ch = QSpinBox(); self.sp_lt_ch.setRange(0,3)
        gp_lt.addWidget(self.sp_lt_ch, 0, 1)
        gp_lt.addWidget(QLabel("Frec. láser (MHz):"), 1, 0)
        self.sp_lt_freq = QDoubleSpinBox()
        self.sp_lt_freq.setRange(1,500); self.sp_lt_freq.setValue(80.0); self.sp_lt_freq.setDecimals(2)
        gp_lt.addWidget(self.sp_lt_freq, 1, 1)
        gp_lt.addWidget(QLabel("Umbral intensidad:"), 2, 0)
        self.sp_lt_min = QSpinBox(); self.sp_lt_min.setRange(1,10000); self.sp_lt_min.setValue(1)
        self.sp_lt_min.setToolTip("phasor_threshold: mínimo de fotones por píxel — se aplica con el botón al archivo seleccionado")
        gp_lt.addWidget(self.sp_lt_min, 2, 1)
        gp_lt.addWidget(QLabel("Filtro mediana (rep):"), 3, 0)
        self.sp_lt_median = QSpinBox(); self.sp_lt_median.setRange(1,10); self.sp_lt_median.setValue(1)
        self.sp_lt_median.setToolTip("Repeticiones del filtro mediana — se aplica solo al archivo seleccionado con el botón")
        gp_lt.addWidget(self.sp_lt_median, 3, 1)
        gp_lt.addWidget(QLabel("Estilo plot:"), 4, 0)
        self.cmb_lt_style = QComboBox()
        self.cmb_lt_style.addItems(["hist2d","contour","plot"])
        gp_lt.addWidget(self.cmb_lt_style, 4, 1)
        btn_lt_draw = QPushButton("▶  Calcular y plotear")
        btn_lt_draw.setStyleSheet("font-weight:bold;background:#2c7bb6;color:white;padding:4px;")
        btn_lt_draw.clicked.connect(self._draw_lifetime)
        gp_lt.addWidget(btn_lt_draw, 5, 0, 1, 2)
        ll.addWidget(grp_lt_par)

        # — Cursor interactivo —
        grp_lt_cursor = QGroupBox("🖱️  Cursor (mover mouse sobre el plot)")
        gcur = QGridLayout(grp_lt_cursor); gcur.setSpacing(4); gcur.setColumnStretch(1,1)
        gcur.addWidget(QLabel("G:"), 0, 0)
        self.lbl_cur_G = QLabel("–"); gcur.addWidget(self.lbl_cur_G, 0, 1)
        gcur.addWidget(QLabel("S:"), 1, 0)
        self.lbl_cur_S = QLabel("–"); gcur.addWidget(self.lbl_cur_S, 1, 1)
        gcur.addWidget(QLabel("τ_φ (ns):"), 2, 0)
        self.lbl_cur_tau_phi = QLabel("–")
        self.lbl_cur_tau_phi.setStyleSheet("font-weight:bold; color:#2c7bb6;")
        gcur.addWidget(self.lbl_cur_tau_phi, 2, 1)
        gcur.addWidget(QLabel("τ_M (ns):"), 3, 0)
        self.lbl_cur_tau_mod = QLabel("–")
        self.lbl_cur_tau_mod.setStyleSheet("font-weight:bold; color:#2c7bb6;")
        gcur.addWidget(self.lbl_cur_tau_mod, 3, 1)
        gcur.addWidget(QLabel("Fase (°):"), 4, 0)
        self.lbl_cur_phase = QLabel("–"); gcur.addWidget(self.lbl_cur_phase, 4, 1)
        gcur.addWidget(QLabel("Mod.:"), 5, 0)
        self.lbl_cur_mod = QLabel("–"); gcur.addWidget(self.lbl_cur_mod, 5, 1)
        ll.addWidget(grp_lt_cursor)

        grp_exp_lt = QGroupBox("💾  Exportar / Estilo")
        ge_lt = QVBoxLayout(grp_exp_lt)
        btn_sty_lt = QPushButton("🎨  Editar estilo…")
        btn_sty_lt.clicked.connect(self._open_style_dialog); ge_lt.addWidget(btn_sty_lt)
        btn_sv_lt = QPushButton("💾  Guardar phasor lifetime")
        btn_sv_lt.clicked.connect(lambda: self._save_fig(self.cv_lifetime.fig, "phasor_lifetime.png"))
        ge_lt.addWidget(btn_sv_lt)
        ll.addWidget(grp_exp_lt)
        ll.addStretch()

        scroll_lt.setWidget(panel_lt)
        self._left_stack.addWidget(scroll_lt)   # index 1

        # ── Área de plots ──
        right = QWidget(); rv = QVBoxLayout(right)
        rv.setContentsMargins(0,0,0,0); rv.setSpacing(0)
        self.plot_tabs = QTabWidget()
        self.plot_tabs.currentChanged.connect(self._left_stack.setCurrentIndex)
        rv.addWidget(self.plot_tabs)

        # Tab 0: Espectros
        tab_spec = QWidget(); tv = QVBoxLayout(tab_spec)
        tv.setContentsMargins(0,0,0,0); tv.setSpacing(2)
        vsplit = QSplitter(Qt.Vertical)
        w_spec = QWidget(); vs = QVBoxLayout(w_spec); vs.setContentsMargins(0,0,0,0)
        self.cv_spec = MplCanvas((12,4))
        vs.addWidget(NavToolbar(self.cv_spec, self)); vs.addWidget(self.cv_spec)
        vsplit.addWidget(w_spec)
        w_ph = QWidget(); vp = QVBoxLayout(w_ph); vp.setContentsMargins(0,0,0,0)
        self.cv_phasor = MplCanvas((12,4))
        vp.addWidget(NavToolbar(self.cv_phasor, self)); vp.addWidget(self.cv_phasor)
        vsplit.addWidget(w_ph)
        vsplit.setSizes([500,360]); tv.addWidget(vsplit)
        self.plot_tabs.addTab(tab_spec, "📊  Espectros + Phasor espectral")

        # Tab 1: Lifetime
        tab_lt = QWidget(); tl = QVBoxLayout(tab_lt)
        tl.setContentsMargins(0,0,0,0); tl.setSpacing(0)
        lt_vsplit = QSplitter(Qt.Vertical)
        w_lt_cv = QWidget(); vlt = QVBoxLayout(w_lt_cv); vlt.setContentsMargins(0,0,0,0)
        self.cv_lifetime = MplCanvas((10,7))
        vlt.addWidget(NavToolbar(self.cv_lifetime, self)); vlt.addWidget(self.cv_lifetime)
        lt_vsplit.addWidget(w_lt_cv)
        self.lt_table = QTableWidget(0, 7)
        self.lt_table.setHorizontalHeaderLabels(
            ["Muestra","Fase (°)","Mod","τ_φ (ns)","τ_M (ns)","τ_avg (ns)","N px"])
        self.lt_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.lt_table.setMaximumHeight(150)
        lt_vsplit.addWidget(self.lt_table)
        lt_vsplit.setSizes([500,120]); tl.addWidget(lt_vsplit)
        self.plot_tabs.addTab(tab_lt, "⏱  Phasor Lifetime")

        # Storage lifetime
        self._lt_entries    = []
        self._lt_color_idx  = 0
        self._lt_cal_entry  = None
        self._lt_ref_mean   = None
        self._lt_ref_real   = None
        self._lt_ref_imag   = None
        self._lt_cursor_artist = None   # artista del cursor en el plot

        # Mouse motion sobre el phasor lifetime → mostrar τ en tiempo real
        self.cv_lifetime.mpl_connect(
            'motion_notify_event', self._on_lt_mouse_move)

        # Click en canvas lifetime → actualiza G,S del cursor

        hsplit.addWidget(right)
        hsplit.setStretchFactor(0, 0)
        hsplit.setStretchFactor(1, 1)

        # Señales
        self.chk_mean.stateChanged.connect(self._schedule)

    def add_spectra(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Seleccioná los archivos de espectros", "",
            "CSV / TXT (*.csv *.txt *.dat);;Todos (*.*)")
        if not paths: return
        for path in paths:
            color = PALETTE[self._color_idx % len(PALETTE)]
            self._color_idx += 1
            name = Path(path).stem.upper()
            st = ("EX"  if "_EX"  in name or name.endswith("EX")  else
                  "ABS" if "_ABS" in name or name.endswith("ABS") else "EM")
            e = SpectrumEntry(path, spec_type=st, color=color)

            # Abrir diálogo de rango λ — bloqueante (exec_)
            confirmed = [False]
            def on_confirm(entry=e):
                confirmed[0] = True
                entry.process()
                self.entries.append(entry)
                self._add_list_item(entry)
                self._draw_spectra()
                self._draw_phasor()

            dlg = LambdaDialog(e, on_confirm, parent=self)
            dlg.exec_() if hasattr(dlg, 'exec_') else dlg.show()

    def _on_rows_moved(self, *_):
        """Sincroniza self.entries con el orden actual de la lista."""
        new_order = []
        for i in range(self.file_list.count()):
            uid = self.file_list.item(i).data(Qt.UserRole)
            entry = next((e for e in self.entries if e.uid == uid), None)
            if entry:
                new_order.append(entry)
        self.entries = new_order
        self._draw_spectra()
        self._draw_phasor()

    def _on_selection_changed(self):
        """Resalta las curvas/markers de los archivos seleccionados."""
        selected_uids = {
            self.file_list.item(i).data(Qt.UserRole)
            for i in range(self.file_list.count())
            if self.file_list.item(i).isSelected()
        }
        self._highlight(selected_uids)

    def _highlight(self, selected_uids: set):
        """Sube grosor/alpha de las curvas seleccionadas, baja las demás."""
        # Obtener labels de los entries seleccionados
        sel_labels = {e.label for e in self.entries if e.uid in selected_uids}
        any_sel = bool(selected_uids)

        # ── Espectros ──
        for ax in self.cv_spec.fig.get_axes():
            for line in ax.get_lines():
                lbl = line.get_label()
                if lbl.startswith("_"):
                    continue
                if not any_sel or lbl in sel_labels:
                    line.set_alpha(1.0)
                    line.set_linewidth(max(line.get_linewidth(), 2.0))
                    line.set_zorder(5)
                else:
                    line.set_alpha(0.15)
                    line.set_zorder(2)
        self.cv_spec.draw_idle()

        # ── Phasor ──
        for ax in self.cv_phasor.fig.get_axes():
            for line in ax.get_lines():
                lbl = line.get_label()
                if lbl.startswith("_"):
                    continue
                if not any_sel or lbl in sel_labels:
                    line.set_alpha(1.0)
                    line.set_markersize(max(line.get_markersize(), 9))
                    line.set_zorder(5)
                else:
                    line.set_alpha(0.15)
                    line.set_zorder(2)
        self.cv_phasor.draw_idle()

    def _add_list_item(self, e: SpectrumEntry):
        item = QListWidgetItem(e.list_label())
        item.setForeground(QColor(e.color))
        item.setData(Qt.UserRole, e.uid)
        self.file_list.addItem(item)

    def _refresh_list_item(self, e: SpectrumEntry):
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(Qt.UserRole) == e.uid:
                item.setText(e.list_label())
                item.setForeground(QColor(e.color))
                break

    def remove_selected(self):
        for item in self.file_list.selectedItems():
            uid = item.data(Qt.UserRole)
            self.entries = [e for e in self.entries if e.uid != uid]
            self.file_list.takeItem(self.file_list.row(item))
        self._schedule()

    def clear_all(self):
        self.entries.clear(); self.file_list.clear(); self._color_idx = 0
        self._schedule()

    def edit_entry(self, item):
        uid   = item.data(Qt.UserRole)
        entry = next((e for e in self.entries if e.uid == uid), None)
        if not entry: return

        # Cerrar diálogo anterior
        if self._edit_dlg is not None:
            try: self._edit_dlg.close()
            except Exception: pass

        def on_change():
            self._refresh_list_item(entry)
            self._apply_preprocessing()

        self._edit_dlg = EditDialog(entry, on_change, parent=self)
        self._edit_dlg.show()

    # ─────────────────────────────────────────────────────────
    #  PREPROCESAMIENTO
    # ─────────────────────────────────────────────────────────

    def _process_all(self):
        for e in self.entries:
            try:    e.process()
            except Exception as ex: print(f"Error {e.filename}: {ex}")

    # ─────────────────────────────────────────────────────────
    #  DIBUJO
    # ─────────────────────────────────────────────────────────

    def _schedule(self, *_):
        self._timer.start()

    def _redraw(self):
        self._process_all()
        self._draw_spectra()
        self._draw_phasor()

    def _save_undo_snapshot(self):
        """Guarda el estado de preprocesamiento de todos los entries."""
        snapshot = []
        for e in self.entries:
            snapshot.append({
                "uid":         e.uid,
                "do_bg":       e.do_bg,
                "bg":          dict(e.bg) if e.bg else None,
                "_do_loess":   e._do_loess,
                "_loess_frac": e._loess_frac,
                "_do_norm":    e._do_norm,
            })
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > 20:   # máximo 20 niveles
            self._undo_stack.pop(0)

    def _undo(self):
        """Restaura el estado de preprocesamiento anterior."""
        if not self._undo_stack:
            QMessageBox.information(self, "Deshacer", "No hay acciones para deshacer.")
            return
        snapshot = self._undo_stack.pop()
        uid_map  = {e.uid: e for e in self.entries}
        for s in snapshot:
            e = uid_map.get(s["uid"])
            if e is None: continue
            e.do_bg       = s["do_bg"]
            e.bg          = s["bg"]
            e._do_loess   = s["_do_loess"]
            e._loess_frac = s["_loess_frac"]
            e._do_norm    = s["_do_norm"]
            self._refresh_list_item(e)
        self._process_all()
        self._draw_spectra()
        self._draw_phasor()

    def _remove_bg_all(self):
        if not self.entries:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccioná el archivo de background (se restará a todos)",
            "", "CSV / TXT (*.csv *.txt *.dat);;Todos (*.*)")
        if not path: return
        self._save_undo_snapshot()
        try:
            for e in self.entries:
                e.assign_bg(path)
                self._refresh_list_item(e)
            self._process_all()
            self._draw_spectra()
            self._draw_phasor()
        except Exception as ex:
            QMessageBox.critical(self, "Error al cargar BG", str(ex))

    def _apply_loess_all(self):
        self._save_undo_snapshot()
        frac = self.sp_loess_all.value()
        for e in self.entries:
            e._do_loess   = True
            e._loess_frac = frac
        self._process_all()
        self._draw_spectra()
        self._draw_phasor()

    def _apply_norm(self):
        self._save_undo_snapshot()
        for e in self.entries:
            e._do_norm = True
        self._process_all()
        self._draw_spectra()
        self._draw_phasor()

    def _apply_preprocessing(self):
        """on_change del BG: recalcula manteniendo estado acumulativo."""
        self._process_all()
        self._draw_spectra()
        self._draw_phasor()

    def _draw_spectra(self):
        fig = self.cv_spec.fig; fig.clear()
        if not self.entries: self.cv_spec.draw(); return

        types_present = [t for t in SPEC_TYPES
                         if any(e.spec_type == t for e in self.entries)]
        n    = len(types_present) or 1
        axes = {t: fig.add_subplot(1, n, i+1)
                for i, t in enumerate(types_present)}

        groups: dict[tuple, list] = {}
        for e in self.entries:
            groups.setdefault((e.spec_type, e.label), []).append(e)

        ylabels = {"EX": "Intensidad (excitación)",
                   "EM": "Intensidad (emisión)",
                   "ABS": "Absorbancia relativa"}
        plotted: dict[str, set] = {t: set() for t in SPEC_TYPES}

        for (st, lbl), grp in groups.items():
            if st not in axes: continue
            ax    = axes[st]
            color = grp[0].color

            if self.chk_mean.isChecked() and len(grp) > 1:
                lam_c, mat = interp_to_grid(
                    [e.lam for e in grp], [e.data for e in grp])
                mean = mat.mean(1); std = mat.std(1)
                ax.fill_between(lam_c, mean-std, mean+std,
                                color=color, alpha=0.18, zorder=2)
                ax.plot(lam_c, mean, color=color, lw=2.2, zorder=3,
                        label=lbl if lbl not in plotted[st] else "_")
            else:
                for i, e in enumerate(grp):
                    ax.plot(e.lam, e.data, color=color, alpha=0.8, lw=1.5,
                            zorder=3,
                            label=lbl if (i==0 and lbl not in plotted[st]) else "_")
            plotted[st].add(lbl)

        for t, ax in axes.items():
            ax.set_xlabel("λ (nm)", fontsize=10)
            ax.set_ylabel(ylabels.get(t, "Intensidad"), fontsize=10)
            ax.set_title({"EX":"Excitación","EM":"Emisión",
                          "ABS":"Absorción"}.get(t, t),
                         fontsize=11, fontweight="bold")
            ax.legend(fontsize=8, framealpha=0.7)
            ax.spines[["top","right"]].set_visible(False)

        fig.tight_layout(); self.cv_spec.draw()
        self._spec_axes = axes   # {spec_type: ax}

    def _draw_phasor(self):
        fig = self.cv_phasor.fig; fig.clear()
        if not self.entries: self.cv_phasor.draw(); return

        types_present = [t for t in SPEC_TYPES
                         if any(e.spec_type == t for e in self.entries)]
        if not types_present: self.cv_phasor.draw(); return

        axes = {t: fig.add_subplot(1, len(types_present), i+1)
                for i, t in enumerate(types_present)}
        pps  = {t: PhasorPlot(allquadrants=True, ax=axes[t])
                for t in types_present}

        groups: dict[tuple, list] = {}
        for e in self.entries:
            groups.setdefault((e.spec_type, e.label), []).append(e)

        plotted: dict[str, set] = {t: set() for t in SPEC_TYPES}

        for (st, lbl), grp in groups.items():
            if st not in axes: continue
            color = grp[0].color
            gvals, svals = [], []
            for e in grp:
                g, s = e.get_phasor()
                if not np.isnan(g): gvals.append(g); svals.append(s)
            if not gvals: continue

            pps[st].plot(gvals, svals, fmt="o", color=color, markersize=8,
                         label=lbl if lbl not in plotted[st] else "_")
            plotted[st].add(lbl)
            if len(gvals) > 1:
                pps[st].plot([np.mean(gvals)], [np.mean(svals)],
                             fmt="o", color=color, markersize=13,
                             markeredgecolor="black", markeredgewidth=1.2)

        for t, ax in axes.items():
            ax.set_title(f"Phasor {t}", fontsize=11, fontweight="bold")
            ax.legend(fontsize=8, framealpha=0.7, loc="upper right")

        fig.tight_layout()
        fig.subplots_adjust(right=0.78)
        self.cv_phasor.draw()
        self._phasor_axes = axes   # {spec_type: ax}

    # ─────────────────────────────────────────────────────────
    #  EXPORTAR
    # ─────────────────────────────────────────────────────────

    def _open_style_dialog(self):
        """Abre el diálogo de edición de estilo."""
        dlg = StyleDialog(self.cv_spec.fig, self.cv_phasor.fig,
                          self.cv_lifetime.fig, self)
        dlg.exec_()
        self.cv_spec.draw()
        self.cv_phasor.draw()
        self.cv_lifetime.draw()

    # ─────────────────────────────────────────────────────────
    #  PHASOR LIFETIME (.fbd)
    # ─────────────────────────────────────────────────────────

    def _load_calibration(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccioná el archivo de calibración (.fbd)", "",
            "FBD files (*.fbd);;Todos (*.*)")
        if not path: return
        try:
            self._lt_cal_entry = _read_fbd(path)
            self.lbl_lt_cal.setText(f"✓  {Path(path).name}")
            self.lbl_lt_cal.setStyleSheet("color:#2ca02c; font-size:10px;")
            # Resetear correcciones
            self._lt_ref_mean = None
            self._lt_ref_real = None
            self._lt_ref_imag = None
            self.lbl_lt_dphase.setText("– (presioná ⚙ Calcular)")
            self.lbl_lt_dmod.setText("– (presioná ⚙ Calcular)")
        except Exception as ex:
            QMessageBox.critical(self, "Error al cargar calibración", str(ex))

    def _compute_calibration(self):
        """Calcula y guarda el phasor de referencia para calibrar con phasor_calibrate."""
        if self._lt_cal_entry is None:
            QMessageBox.warning(self, "Sin calibración",
                                "Cargá primero un archivo de referencia.")
            return

        channel = self.sp_lt_ch.value()
        entry   = self._lt_cal_entry

        if channel >= entry["channels"]:
            QMessageBox.warning(self, "Canal inválido",
                f"Canal {channel} no existe (0..{entry['channels']-1}).")
            return

        hist = entry["signals"][channel]   # (Y, X, H)
        if hist.sum() == 0:
            QMessageBox.warning(self, "Sin fotones",
                f"El canal {channel} del archivo de calibración está vacío.")
            return


        # Calcular phasor del archivo de referencia pixel a pixel
        ref_mean, ref_real, ref_imag = phasor_from_signal(hist, axis=-1)

        # Guardar arrays 2D para usar en phasor_calibrate
        self._lt_ref_mean = ref_mean
        self._lt_ref_real = ref_real
        self._lt_ref_imag = ref_imag

        freq    = self.sp_lt_freq.value()
        tau_ref = self.sp_lt_tau_ref.value()

        self.lbl_lt_dphase.setText(
            f"G_mean={float(np.nanmean(ref_real)):.4f}")
        self.lbl_lt_dmod.setText(
            f"τ={tau_ref:.3f} ns @ {freq:.1f} MHz")

        QMessageBox.information(
            self, "Referencia calculada",
            f"Archivo ref.: {Path(entry['path']).name}\n"
            f"τ ref. = {tau_ref:.3f} ns  @  {freq:.1f} MHz\n\n"
            f"Phasor referencia (promedio):\n"
            f"  G = {float(np.nanmean(ref_real)):.4f}\n"
            f"  S = {float(np.nanmean(ref_imag)):.4f}\n\n"
            f"Listo — phasor_calibrate se aplicará al plotear."
        )

    def _load_lifetime_fbd(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Seleccioná archivos .fbd de muestra", "",
            "FBD files (*.fbd);;Todos (*.*)")
        if not paths: return
        for path in paths:
            color = PALETTE[self._lt_color_idx % len(PALETTE)]
            self._lt_color_idx += 1
            try:
                entry = _read_fbd(path)
                entry["color"] = color
                self._lt_entries.append(entry)
                item = QListWidgetItem(entry["label"])
            except Exception as ex:
                QMessageBox.critical(self, f"Error al leer {Path(path).name}", str(ex))

    def _on_lt_mouse_move(self, event):
        """Muestra τ en tiempo real al mover el mouse sobre el phasor lifetime."""
        if event.inaxes is None:
            self.lbl_cur_G.setText("–")
            self.lbl_cur_S.setText("–")
            self.lbl_cur_tau_phi.setText("–")
            self.lbl_cur_tau_mod.setText("–")
            self.lbl_cur_phase.setText("–")
            self.lbl_cur_mod.setText("–")
            return

        G = event.xdata
        S = event.ydata
        if G is None or S is None:
            return

        freq = self.sp_lt_freq.value()


        tau_phi_arr, tau_mod_arr = phasor_to_apparent_lifetime(
            [G], [S], frequency=freq)
        ph_arr, mo_arr = phasor_to_polar([G], [S])

        tau_phi = float(tau_phi_arr[0])
        tau_mod = float(tau_mod_arr[0])
        phase   = float(np.rad2deg(ph_arr[0]))
        mod     = float(mo_arr[0])

        self.lbl_cur_G.setText(f"{G:.4f}")
        self.lbl_cur_S.setText(f"{S:.4f}")
        self.lbl_cur_tau_phi.setText(f"{tau_phi:.3f}" if np.isfinite(tau_phi) else "–")
        self.lbl_cur_tau_mod.setText(f"{tau_mod:.3f}" if np.isfinite(tau_mod) else "–")
        self.lbl_cur_phase.setText(f"{phase:.2f}°" if np.isfinite(phase) else "–")
        self.lbl_cur_mod.setText(f"{mod:.4f}" if np.isfinite(mod) else "–")

        # Dibujar cursor con PhasorPlot.cursor() de phasorpy
        fig = self.cv_lifetime.fig
        if not fig.get_axes():
            return
        ax = fig.get_axes()[0]

        # Remover cursor anterior si existe
        if self._lt_cursor_artist:
            for art in self._lt_cursor_artist:
                try: art.remove()
                except: pass
            self._lt_cursor_artist = None

        # Dibujar nuevo cursor usando pp.cursor()
        pp = PhasorPlot(allquadrants=False, frequency=freq, ax=ax)
        pp.cursor(G, S, radius=0.02, color="red",
                  linestyle="--", linewidth=1.2, crosshair=True)

        # Guardar artistas nuevos para poder removerlos
        self._lt_cursor_artist = [l for l in ax.get_lines()
                                   if l.get_color() == "red"
                                   and l.get_linestyle() == "--"][-2:]
        self.cv_lifetime.draw_idle()

    def _clear_filter_selected(self):
        """Quita el filtro del archivo seleccionado → vuelve al phasor original."""
        for item in self.lt_file_list.selectedItems():
            idx = self.lt_file_list.row(item)
            if idx < len(self._lt_entries):
                e = self._lt_entries[idx]
                e.pop("_real_filt", None)
                e.pop("_imag_filt", None)
                e.pop("_mean_filt", None)
        self._draw_lifetime()

    def _apply_filter_selected(self):
        """Aplica phasor_filter_median al archivo seleccionado en la lista."""
        selected = self.lt_file_list.selectedItems()
        if not selected:
            QMessageBox.information(self, "Sin selección",
                                    "Seleccioná un archivo de la lista primero.")
            return


        ch   = self.sp_lt_ch.value()
        reps = max(1, self.sp_lt_median.value())
        freq = self.sp_lt_freq.value()

        applied = 0
        for item in selected:
            idx   = self.lt_file_list.row(item)
            if idx >= len(self._lt_entries):
                continue
            entry = self._lt_entries[idx]
            sig   = entry["signals"][ch]

            if sig.sum() == 0:
                continue

            mean, real, imag = phasor_from_signal(sig, axis=-1)

            if self._lt_ref_mean is not None:
                real, imag = phasor_calibrate(
                    real, imag,
                    self._lt_ref_mean, self._lt_ref_real, self._lt_ref_imag,
                    frequency=freq,
                    lifetime=self.sp_lt_tau_ref.value(),
                )

            mean_th, real_th, imag_th = phasor_threshold(
                mean, real, imag, mean_min=self.sp_lt_min.value())

            mean_f, real_f, imag_f = phasor_filter_median(
                mean_th, real_th, imag_th, repeat=reps)

            entry["_mean_filt"] = mean_f
            entry["_real_filt"] = real_f
            entry["_imag_filt"] = imag_f
            applied += 1

        if applied == 0:
            QMessageBox.warning(self, "Sin datos",
                                "Los archivos seleccionados no tienen fotones.")
            return

        self._draw_lifetime()

    def _remove_lt_selected(self):
        """Quita los archivos seleccionados de la lista de lifetime."""
        rows = sorted(
            [self.lt_file_list.row(i) for i in self.lt_file_list.selectedItems()],
            reverse=True)
        for row in rows:
            self.lt_file_list.takeItem(row)
            if row < len(self._lt_entries):
                self._lt_entries.pop(row)

    def _clear_lifetime(self):
        for e in self._lt_entries:
            e.pop("_real_filt", None)
            e.pop("_imag_filt", None)
            e.pop("_mean_filt", None)
        self._lt_entries.clear()
        self.lt_file_list.clear()
        self._lt_color_idx = 0
        self.cv_lifetime.fig.clear()
        self.cv_lifetime.draw()

    def _compute_lt_phasor(self, entry: dict, channel: int):
        """Calcula phasor calibrado pixel a pixel.
        Threshold y filtro de mediana se aplican explícitamente por archivo."""

        freq = self.sp_lt_freq.value()

        if channel >= entry["channels"]:
            return None

        sig = entry["signals"][channel]
        if sig.sum() == 0:
            return None

        # Si ya se aplicó threshold/filtro explícito, usarlo directamente
        if "_real_filt" in entry:
            return {"mean": entry["_mean_filt"],
                    "real": entry["_real_filt"],
                    "imag": entry["_imag_filt"]}

        # Calcular desde cero: phasor + calibración (sin threshold ni filtro)
        mean, real, imag = phasor_from_signal(sig, axis=-1)

        if self._lt_ref_mean is not None:
            real, imag = phasor_calibrate(
                real, imag,
                self._lt_ref_mean, self._lt_ref_real, self._lt_ref_imag,
                frequency=freq,
                lifetime=self.sp_lt_tau_ref.value(),
            )

        return {"mean": mean, "real": real, "imag": imag}

    def _draw_lifetime(self):
        if not self._lt_entries:
            QMessageBox.information(self, "Sin datos", "Cargá al menos un archivo .fbd.")
            return

        channel = self.sp_lt_ch.value()
        freq    = self.sp_lt_freq.value()

        if self._lt_ref_mean is None:
            reply = QMessageBox.question(
                self, "Sin calibración",
                "No se calculó la calibración.\n"
                "¿Continuar igualmente con los datos sin corregir?",
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return

        fig = self.cv_lifetime.fig
        fig.clear()

        # Layout: phasor izquierda, tabla derecha
        ax = fig.add_subplot(1, 1, 1)
        pp = PhasorPlot(allquadrants=False, frequency=freq, ax=ax)

        results = []
        style = self.cmb_lt_style.currentText()

        # Colormaps por muestra — uno por color de la PALETTE
        CMAPS = ["Blues","Reds","Greens","Purples","Oranges",
                 "YlOrBr","PuBu","RdPu","BuGn","YlGn"]

        for idx, entry in enumerate(self._lt_entries):
            res = self._compute_lt_phasor(entry, channel)
            if res is None:
                continue

            mean_arr = res["mean"]
            real_arr = res["real"]
            imag_arr = res["imag"]

            valid = ~np.isnan(real_arr)
            if not valid.any():
                continue

            G_v = real_arr[valid].flatten()
            S_v = imag_arr[valid].flatten()

            cmap = CMAPS[idx % len(CMAPS)]

            # Plotear según estilo — cada muestra con su propio colormap
            if style == "hist2d":
                pp.hist2d(G_v, S_v, cmap=cmap)
            elif style == "contour":
                pp.contour(G_v, S_v, cmap=cmap)
            else:
                pp.plot(G_v, S_v, fmt="o", color=entry["color"],
                        alpha=0.3, markersize=2, label="_")

            # Centroide siempre visible
            G_c = float(np.nanmean(G_v))
            S_c = float(np.nanmean(S_v))
            pp.plot([G_c], [S_c], fmt="o",
                    color=entry["color"], markersize=10,
                    markeredgecolor="black", markeredgewidth=1.5,
                    label=entry["label"])

            # τ del centroide usando phasor_to_apparent_lifetime de phasorpy
            tau_phi_arr, tau_mod_arr = phasor_to_apparent_lifetime(
                np.array([G_c]), np.array([S_c]), frequency=freq)
            tau_phi = float(tau_phi_arr[0])
            tau_mod = float(tau_mod_arr[0])
            tau_avg = (tau_phi + tau_mod) / 2

            ph_arr, mo_arr = phasor_to_polar(np.array([G_c]), np.array([S_c]))
            phase_deg = float(np.rad2deg(ph_arr[0]))
            mod       = float(mo_arr[0])
            results.append((entry["label"], {
                "phase_deg": phase_deg, "mod": mod,
                "tau_phi": tau_phi, "tau_mod": tau_mod, "tau_avg": tau_avg,
                "n_pixels": int(valid.sum()),
            }))

        ax.set_title(f"Phasor Lifetime  –  canal {channel}  "
                     f"({freq:.1f} MHz)",
                     fontsize=12, fontweight="bold")

        # Plotear punto de calibración si está disponible
        if self._lt_ref_mean is not None:
            # Calibrar el propio archivo de referencia → debe quedar en posición teórica
            cal_entry = self._lt_cal_entry
            hist_ref  = cal_entry["signals"][channel].sum(axis=(0,1)).astype(float)
            if hist_ref.sum() > 0:
                _, G_ref_raw, S_ref_raw = phasor_from_signal(
                    hist_ref[np.newaxis,:], axis=-1)
                G_cal_ref, S_cal_ref = phasor_calibrate(
                    G_ref_raw, S_ref_raw,
                    self._lt_ref_mean, self._lt_ref_real, self._lt_ref_imag,
                    frequency=freq, lifetime=self.sp_lt_tau_ref.value())
                pp.plot([float(G_cal_ref[0])], [float(S_cal_ref[0])],
                        fmt="D", color="black", markersize=10,
                        markeredgecolor="white", markeredgewidth=1.5,
                        label=f"Ref. ({self._lt_cal_entry['label']})")

        ax.legend(fontsize=9, framealpha=0.7, loc="upper right")

        fig.tight_layout()
        self.cv_lifetime.draw()

        # Tabla Qt
        self.lt_table.setRowCount(0)
        for lbl, r in results:
            def fmt(v): return f"{v:.3f}" if not np.isnan(v) else "–"
            row = self.lt_table.rowCount()
            self.lt_table.insertRow(row)
            for col, val in enumerate([
                lbl,
                f"{r['phase_deg']:.2f}",
                f"{r['mod']:.3f}",
                fmt(r['tau_phi']),
                fmt(r['tau_mod']),
                fmt(r['tau_avg']),
                str(r['n_pixels']),
            ]):
                self.lt_table.setItem(row, col, QTableWidgetItem(val))

    def _save_ax(self, key: str):
        """Guarda un subplot individual usando la referencia directa al ax."""
        is_phasor = key.startswith("PH_")
        spec_type = key.replace("PH_", "")

        ax_dict = self._phasor_axes if is_phasor else self._spec_axes
        target  = ax_dict.get(spec_type)

        if target is None:
            QMessageBox.warning(self, "No disponible",
                                f"No hay datos de tipo {spec_type} cargados.")
            return

        fig     = self.cv_phasor.fig if is_phasor else self.cv_spec.fig
        default = f"{'phasor' if is_phasor else 'espectro'}_{spec_type.lower()}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar figura", default,
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if not path: return

        extent = target.get_tightbbox(fig.canvas.get_renderer())
        extent = extent.transformed(fig.dpi_scale_trans.inverted())
        extent = extent.expanded(1.02, 1.05)
        fig.savefig(path, dpi=300, bbox_inches=extent)
        QMessageBox.information(self, "Guardado", f"Guardado:\n{path}")

    def _save_fig(self, fig, default_name):
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar figura", default_name,
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if not path: return
        fig.savefig(path, dpi=300, bbox_inches="tight")
        QMessageBox.information(self, "Guardado", f"Figura guardada:\n{path}")

    def _save_subplot(self, key: str):
        """Guarda la figura completa (espectros o phasor) tal como está en pantalla."""
        is_phasor = key.startswith("PH_")
        fig = self.cv_phasor.fig if is_phasor else self.cv_spec.fig
        spec_type = key.replace("PH_", "")
        default = f"{'phasor' if is_phasor else 'espectro'}_{spec_type}.png"
        self._save_fig(fig, default)

    def save_all(self):
        out_dir = QFileDialog.getExistingDirectory(self, "Carpeta de salida")
        if not out_dir: return
        p1 = Path(out_dir)/"espectros.png"
        p2 = Path(out_dir)/"phasor.png"
        self.cv_spec.fig.savefig(p1,   dpi=300, bbox_inches="tight")
        self.cv_phasor.fig.savefig(p2, dpi=300, bbox_inches="tight")
        QMessageBox.information(self, "Guardado", f"Guardados:\n{p1}\n{p2}")


# ══════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
