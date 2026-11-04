# Бет-бейнеден эмоцияны тану

Веб-камерадан бетті тауып, 7 эмоцияның бірін сенімділік пайызымен көрсететін жүйе (мысалы, «Қуаныш — 87%»), Gradio веб-интерфейсі және «Эмоцияны қайтала!» ойыны.

Windows, macOS және Linux-та жұмыс істейді.

## Талаптар

- **Python 3.11** (3.12+ емес — TensorFlow 2.16 және MediaPipe 0.10.14 үшін 3.11 керек)
  - Windows: [python.org](https://www.python.org/downloads/release/python-3119/) сайтынан жүктеп, орнату кезінде **«Add python.exe to PATH»** белгісін қойыңыз
- Git
- Веб-камера (демо мен ойын үшін)

## Орнату

### 1. Репозиторийді жүктеу

```bash
git clone https://github.com/<USERNAME>/emotion-project.git
cd emotion-project
```

### 2. Виртуалды орта

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> PowerShell `Activate.ps1`-ді бұғаттаса: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` орындап, қайта көріңіз.
> Немесе `cmd` ішінде: `.venv\Scripts\activate.bat`

**macOS / Linux:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Датасет (FER-2013)

Датасет репозиторийге кірмейді (~250 МБ). Тек модельді **қайта оқыту** немесе **бағалау** үшін керек — дайын модель `models/emotion_cnn.keras` бар, демо, веб-интерфейс және ойын датасетсіз жұмыс істейді.

1. Kaggle-ден `fer2013.zip` жүктеңіз: <https://www.kaggle.com/datasets/deadskull7/fer2013> (ішінде `fer2013.csv` болуы керек) немесе Kaggle CLI арқылы (`kaggle.json` токені керек):

   ```bash
   kaggle datasets download -d deadskull7/fer2013 -p data
   ```

2. Файлды `data/fer2013.zip` етіп сақтап, суреттерге бөліңіз:

   ```bash
   cd src
   python prepare_data.py ../data/fer2013.zip
   ```

   Нәтиже: `data/fer2013/{train,test}/<эмоция>/` (train — 28 709, test — 7 178 сурет).

Қаріп `fonts/DejaVuSans.ttf` — қазақ әріптерін экранға жазу үшін (репозиторийде бар).

## Іске қосу

Алдымен виртуалды ортаны белсендіріңіз (`.venv\Scripts\Activate.ps1` немесе `source .venv/bin/activate`), содан кейін барлық скрипттер `src/` ішінен іске қосылады:

```bash
cd src
```

| Кезең | Команда | Нәтиже |
|---|---|---|
| Бетті анықтау | `python face_detect.py` | камерада бет рамкада, `m` — Haar/MediaPipe ауыстыру |
| Деректерді зерттеу | `python data.py` | `reports/class_distribution.png`, `reports/samples.png` |
| CNN оқыту | `python train.py --model cnn` | `models/emotion_cnn.keras`, `reports/history_cnn.png` |
| MobileNetV2 оқыту | `python train.py --model mobilenet` | `models/emotion_mobilenet.keras` |
| Бағалау | `python evaluate.py --model ../models/emotion_cnn.keras` | confusion matrix, қате мысалдар |
| Веб-камера демо | `python webcam_demo.py` | `q` — шығу, `s` — скриншот |
| Веб-интерфейс | `python app_gradio.py` | http://127.0.0.1:7860 |
| Ойын | `python game.py` | «Эмоцияны қайтала!» |

Командалар Windows, macOS, Linux-та бірдей. Жолдар `/` арқылы жазылған — Windows PowerShell оларды да түсінеді.

## Ақаулықтарды жою

**Камера ашылмайды**
- Zoom / Teams / басқа бағдарлама камераны қолданып тұрса, жабыңыз.
- Басқа камераны көріңіз: `python webcam_demo.py --camera 1`
- Windows: Settings → Privacy & security → Camera → «Let desktop apps access your camera» қосулы болсын.
- macOS: System Settings → Privacy & Security → Camera → Terminal / VS Code рұқсатын қосыңыз.

**Windows: `DLL load failed` (TensorFlow немесе MediaPipe импортында)**
- [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) (x64) орнатыңыз.
- Python 64-битті екенін тексеріңіз: `python -c "import struct; print(struct.calcsize('P')*8)"` → `64`.

**Windows: консольде қазақ әріптері `????` болып шығады**
- PowerShell-де: `$env:PYTHONUTF8=1` орнатып, скриптті қайта іске қосыңыз.

**`pip install` қатесі / пакет табылмады**
- Python нұсқасын тексеріңіз: `python --version` → `3.11.x` болуы керек.

## Нәтижелер

Толық талдау: `reports/` папкасындағы есеп.

## Құрылым

```
emotion-project/
├── data/fer2013/{train,test}/<эмоция>/   (Kaggle-ден, git-ке кірмейді)
├── fonts/DejaVuSans.ttf
├── models/emotion_cnn.keras
├── reports/
├── requirements.txt
└── src/
    ├── config.py, face_detect.py, data.py, prepare_data.py
    ├── model_cnn.py, model_transfer.py, train.py, evaluate.py
    └── predictor.py, webcam_demo.py, app_gradio.py, game.py
```

## Шектеулер

Модель адамның шын сезімін емес, бет өрнегінің FER-2013 суреттеріне ұқсастығын бағалайды. Датасет интернеттен жиналған, белгілерінде қате бар, кейбір топтар аз көрсетілген. Бет — биометриялық дерек: суреттер сақталмайды, басқа адамды тек келісімімен түсіруге болады.
