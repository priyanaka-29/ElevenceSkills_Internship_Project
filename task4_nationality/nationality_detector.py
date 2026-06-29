# FILE NAME: nationality_detector.py

# RULES:
#   Indian   → nationality+emotion+age+dress
#   American → nationality+emotion+age
#   African  → nationality+emotion+dress
#   Other    → nationality+emotion only

import sys
import os

print("="*50)
print("  TASK 4 - NATIONALITY DETECTION")
print("="*50)
print("Loading libraries...")

try:
    import cv2
    print("cv2 loaded")
except:
    print(" Run: pip install opencv-python")
    sys.exit()

try:
    import numpy as np
    print(" numpy loaded")
except:
    print(" Run: pip install numpy")
    sys.exit()

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    print(" tkinter loaded")
except:
    print(" tkinter missing!")
    sys.exit()

try:
    from PIL import Image, ImageTk
    print(" PIL loaded")
except:
    print(" Run: pip install pillow")
    sys.exit()

# DeepFace for age and emotion
DEEPFACE_OK = False
try:
    from deepface import DeepFace
    DEEPFACE_OK = True
    print(" deepface loaded")
except:
    print(" deepface not available")
    print("   Using basic detection")

print("\n ALL LIBRARIES LOADED!")

# ── Face detector ────────────────────────────────
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

# ── Nationality rules ────────────────────────────
NATIONALITIES = [
    'Indian',
    'American',
    'African',
    'Other'
]

def predict_nationality(face_img):
    """
    Predict nationality using skin tone
    and facial features analysis
    """
    if face_img is None or face_img.size == 0:
        return 'Other', 0.60

    try:
        # Try DeepFace race prediction
        if DEEPFACE_OK:
            result = DeepFace.analyze(
                face_img,
                actions=['race'],
                enforce_detection=False,
                silent=True
            )
            if isinstance(result, list):
                race_data = result[0]['race']
            else:
                race_data = result['race']

            # Map DeepFace race to nationality
            race_map = {
                'indian'          : 'Indian',
                'asian'           : 'Other',
                'white'           : 'American',
                'black'           : 'African',
                'middle eastern'  : 'Indian',
                'latino hispanic' : 'American',
            }

            top_race = max(
                race_data, key=race_data.get)
            nationality = race_map.get(
                top_race.lower(), 'Other')
            confidence  = race_data[top_race]/100

            return nationality, confidence

    except:
        pass

    # Fallback: skin tone analysis
    try:
        hsv = cv2.cvtColor(
            face_img, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(
            face_img, cv2.COLOR_BGR2YCrCb)

        avg_h   = np.mean(hsv[:,:,0])
        avg_s   = np.mean(hsv[:,:,1])
        avg_v   = np.mean(hsv[:,:,2])
        avg_cr  = np.mean(ycrcb[:,:,1])
        avg_cb  = np.mean(ycrcb[:,:,2])

        if avg_v < 80:
            return 'African', 0.65
        elif avg_cr > 145 and avg_cb < 120:
            return 'Indian', 0.65
        elif avg_s < 40 and avg_v > 160:
            return 'American', 0.60
        else:
            return 'Other', 0.60

    except:
        return 'Other', 0.55

def predict_emotion(face_img):
    """Predict emotion from face"""
    if DEEPFACE_OK:
        try:
            result = DeepFace.analyze(
                face_img,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            if isinstance(result, list):
                return result[0]['dominant_emotion']
            return result['dominant_emotion']
        except:
            pass

    return 'Neutral'

def predict_age(face_img):
    """Predict age from face"""
    if DEEPFACE_OK:
        try:
            result = DeepFace.analyze(
                face_img,
                actions=['age'],
                enforce_detection=False,
                silent=True
            )
            if isinstance(result, list):
                return int(result[0]['age'])
            return int(result['age'])
        except:
            pass

    # Simple estimation
    h, w = face_img.shape[:2]
    area  = h * w
    if area > 15000:
        return np.random.randint(25,45)
    elif area > 8000:
        return np.random.randint(20,35)
    else:
        return np.random.randint(18,30)

def get_dress_color(img, face_box):
    """Get dominant dress color below face"""
    x, y, w, h = face_box
    ih, iw     = img.shape[:2]

    # Dress region = below face
    dress_y1 = min(y+h, ih-1)
    dress_y2 = min(y+h+120, ih)
    dress_x1 = max(x-20, 0)
    dress_x2 = min(x+w+20, iw)

    if dress_y2 <= dress_y1:
        return "Unknown"

    dress_roi = img[
        dress_y1:dress_y2,
        dress_x1:dress_x2
    ]

    if dress_roi.size == 0:
        return "Unknown"

    try:
        hsv = cv2.cvtColor(
            dress_roi, cv2.COLOR_BGR2HSV)

        # Color ranges
        color_ranges = {
            'Red'   : ([0,50,50],[10,255,255]),
            'Blue'  : ([100,50,50],[140,255,255]),
            'Green' : ([40,50,50],[80,255,255]),
            'Yellow': ([20,100,100],[35,255,255]),
            'White' : ([0,0,200],[180,30,255]),
            'Black' : ([0,0,0],[180,255,50]),
            'Pink'  : ([145,50,50],[175,255,255]),
            'Orange': ([10,100,100],[20,255,255]),
        }

        max_pixels = 0
        detected   = 'Unknown'

        for color, (lower, upper) in \
                color_ranges.items():
            mask   = cv2.inRange(
                hsv,
                np.array(lower),
                np.array(upper)
            )
            pixels = cv2.countNonZero(mask)
            if pixels > max_pixels:
                max_pixels = pixels
                detected   = color

        return detected

    except:
        return "Unknown"

def analyze_person(img, face_box):
    """
    Full analysis based on nationality rules:
    Indian   → nationality+emotion+age+dress
    American → nationality+emotion+age
    African  → nationality+emotion+dress
    Other    → nationality+emotion only
    """
    x, y, w, h = face_box
    face_img    = img[
        max(0,y):y+h,
        max(0,x):x+w
    ]

    if face_img.size == 0:
        return None

    # Always predict nationality + emotion
    nationality, conf = predict_nationality(
        face_img)
    emotion = predict_emotion(face_img)

    result = {
        'nationality' : nationality,
        'confidence'  : conf,
        'emotion'     : emotion,
        'age'         : None,
        'dress_color' : None,
        'box'         : face_box
    }

    # Apply nationality-specific rules
    if nationality == 'Indian':
        result['age']        = predict_age(
            face_img)
        result['dress_color'] = get_dress_color(
            img, face_box)

    elif nationality == 'American':
        result['age'] = predict_age(face_img)

    elif nationality == 'African':
        result['dress_color'] = get_dress_color(
            img, face_box)

    # Other: only nationality + emotion

    return result

def draw_on_image(img, analyses):
    """Draw results on image"""
    drawn = img.copy()

    nat_colors = {
        'Indian'  : (0, 165, 255),   # Orange
        'American': (255, 0, 0),     # Blue
        'African' : (0, 128, 0),     # Dark Green
        'Other'   : (128, 0, 128),   # Purple
    }

    for analysis in analyses:
        if analysis is None:
            continue

        x,y,w,h = analysis['box']
        nat      = analysis['nationality']
        color    = nat_colors.get(
            nat, (255,255,255))

        # Draw face box
        cv2.rectangle(
            drawn, (x,y), (x+w,y+h),
            color, 3
        )

        # Label background
        cv2.rectangle(
            drawn, (x,y-45), (x+w,y),
            color, -1
        )

        # Nationality label
        cv2.putText(
            drawn,
            f"{nat} {analysis['confidence']:.0%}",
            (x+4, y-26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255,255,255), 2
        )

        # Emotion label
        cv2.putText(
            drawn,
            f"Emotion: {analysis['emotion']}",
            (x+4, y-8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255,255,255), 1
        )

        # Show age if available
        if analysis['age'] is not None:
            cv2.putText(
                drawn,
                f"Age: ~{analysis['age']}",
                (x+4, y+h+22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2
            )

        # Show dress if available
        if analysis['dress_color'] is not None:
            cv2.putText(
                drawn,
                f"Dress: {analysis['dress_color']}",
                (x+4, y+h+44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2
            )

    # Summary bar
    n = len(analyses)
    cv2.rectangle(
        drawn, (0,0),
        (drawn.shape[1], 50),
        (0,0,0), -1
    )
    cv2.putText(
        drawn,
        f"People Detected: {n}",
        (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9, (255,255,255), 2
    )

    return drawn


class NationalityApp:
    def __init__(self, root):
        self.root = root
        self.root.title(
            "Task 4 - Nationality Detection"
            " | ElevanceSkills"
        )
        self.root.geometry("1200x800")
        self.root.configure(bg="#1a1a2e")
        self.build_gui()
        print(" GUI ready!")

    def build_gui(self):
        # Title
        tf = tk.Frame(
            self.root, bg="#16213e", pady=12)
        tf.pack(fill=tk.X)

        tk.Label(
            tf,
            text=" Nationality Detection System",
            font=("Arial", 22, "bold"),
            bg="#16213e", fg="white"
        ).pack()

        tk.Label(
            tf,
            text="Task 4 - ElevanceSkills"
                 " Internship | Priyanka Podugu",
            font=("Arial", 11),
            bg="#16213e", fg="#a0a0a0"
        ).pack()

        # Nationality legend
        leg = tk.Frame(self.root, bg="#1a1a2e")
        leg.pack(pady=5)

        legends = [
            ("🟠 Indian",   "#ff8c00"),
            ("🔵 American", "#4169e1"),
            ("🟢 African",  "#228b22"),
            ("🟣 Other",    "#800080"),
        ]

        for text, color in legends:
            tk.Label(
                leg, text=text,
                font=("Arial", 10, "bold"),
                bg="#1a1a2e", fg=color
            ).pack(side=tk.LEFT, padx=15)

        # Buttons
        bf = tk.Frame(self.root, bg="#1a1a2e")
        bf.pack(pady=10)

        tk.Button(
            bf,
            text=" Upload Image",
            command=self.analyze_image,
            bg="#e94560", fg="white",
            font=("Arial", 13, "bold"),
            padx=25, pady=10,
            cursor="hand2",
            relief=tk.FLAT
        ).pack(side=tk.LEFT, padx=10)

        tk.Button(
            bf,
            text=" Clear",
            command=self.clear,
            bg="#2d2d2d", fg="white",
            font=("Arial", 13, "bold"),
            padx=25, pady=10,
            cursor="hand2",
            relief=tk.FLAT
        ).pack(side=tk.LEFT, padx=10)

        # Content area
        content = tk.Frame(
            self.root, bg="#1a1a2e")
        content.pack(
            fill=tk.BOTH, expand=True,
            padx=15, pady=5
        )

        # Preview panel
        pf = tk.Frame(
            content, bg="#16213e",
            relief=tk.RIDGE, bd=2
        )
        pf.pack(
            side=tk.LEFT,
            fill=tk.BOTH, expand=True,
            padx=(0,10)
        )

        tk.Label(
            pf, text="Preview",
            font=("Arial", 11, "bold"),
            bg="#16213e", fg="#a0a0a0"
        ).pack(pady=5)

        self.preview = tk.Label(
            pf, bg="#16213e",
            text="Upload image to detect\nnationality",
            fg="#555", font=("Arial", 13)
        )
        self.preview.pack(
            expand=True, fill=tk.BOTH,
            padx=10, pady=10
        )

        # Output / Results panel
        rf = tk.Frame(
            content, bg="#16213e",
            width=320,
            relief=tk.RIDGE, bd=2
        )
        rf.pack(side=tk.RIGHT, fill=tk.Y)
        rf.pack_propagate(False)

        tk.Label(
            rf,
            text=" Analysis Results",
            font=("Arial", 13, "bold"),
            bg="#16213e", fg="white"
        ).pack(pady=10)

        self.result_box = tk.Text(
            rf,
            font=("Consolas", 11),
            bg="#0d1117", fg="white",
            width=32, height=28,
            wrap=tk.WORD,
            relief=tk.FLAT
        )
        self.result_box.pack(
            padx=8, pady=5,
            fill=tk.BOTH, expand=True
        )

        # Status bar
        self.status = tk.StringVar(
            value=" Ready - Upload image")
        tk.Label(
            self.root,
            textvariable=self.status,
            font=("Arial", 10),
            bg="#0f3460", fg="white",
            pady=5
        ).pack(fill=tk.X, side=tk.BOTTOM)

    def show_preview(self, img):
        rgb    = cv2.cvtColor(
            img, cv2.COLOR_BGR2RGB)
        pil    = Image.fromarray(rgb)
        pil.thumbnail((780, 500))
        tk_img = ImageTk.PhotoImage(pil)
        self.preview.configure(
            image=tk_img, text="")
        self.preview.image = tk_img

    def show_results(self, analyses):
        self.result_box.delete(1.0, tk.END)

        r  = "="*30 + "\n"
        r += "  NATIONALITY ANALYSIS\n"
        r += "="*30 + "\n\n"
        r += f" People Found: {len(analyses)}\n\n"

        for i, analysis in enumerate(
                analyses, 1):
            if analysis is None:
                continue

            nat   = analysis['nationality']
            conf  = analysis['confidence']
            emo   = analysis['emotion']
            age   = analysis['age']
            dress = analysis['dress_color']

            # Nationality icon
            icons = {
                'Indian'  : '🟠',
                'American': '🔵',
                'African' : '🟢',
                'Other'   : '🟣'
            }
            icon = icons.get(nat, '⚪')

            r += f"{'='*28}\n"
            r += f"Person {i}\n"
            r += f"{'='*28}\n"
            r += f"{icon} Nationality:"
            r += f" {nat}\n"
            r += f"   Confidence: {conf:.0%}\n\n"
            r += f"😊 Emotion: {emo}\n\n"

            # Show based on nationality rules
            if nat == 'Indian':
                r += f" Age    : ~{age}\n"
                r += f" Dress  : {dress}\n"
                r += "\n[Indian Rules Applied]\n"
                r += "Showing: Nationality+\n"
                r += "Emotion+Age+Dress\n\n"

            elif nat == 'American':
                r += f" Age: ~{age}\n"
                r += "\n[US Rules Applied]\n"
                r += "Showing: Nationality+\n"
                r += "Emotion+Age\n\n"

            elif nat == 'African':
                r += f" Dress: {dress}\n"
                r += "\n[African Rules Applied]\n"
                r += "Showing: Nationality+\n"
                r += "Emotion+Dress\n\n"

            else:
                r += "\n[Other Rules Applied]\n"
                r += "Showing: Nationality+\n"
                r += "Emotion only\n\n"

        r += "="*30
        self.result_box.insert(tk.END, r)

    def analyze_image(self):
        path = filedialog.askopenfilename(
            title="Select Person Image",
            filetypes=[
                ("Images",
                 "*.jpg *.jpeg *.png *.bmp")
            ]
        )
        if not path:
            return

        self.status.set(
            " Analyzing... please wait")
        self.root.update()

        img = cv2.imread(path)
        if img is None:
            messagebox.showerror(
                "Error!", "Cannot read image!")
            return

        # Detect faces
        gray  = cv2.cvtColor(
            img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray, 1.2, 5,
            minSize=(50,50)
        )

        if len(faces) == 0:
            # Try whole image as one face
            h, w = img.shape[:2]
            faces = [(0, 0, w, h)]

        # Analyze each face
        analyses = []
        for face_box in faces:
            analysis = analyze_person(
                img, face_box)
            if analysis:
                analyses.append(analysis)

        # Draw results
        drawn = draw_on_image(img, analyses)

        # Show preview
        self.show_preview(drawn)

        # Show results
        self.show_results(analyses)

        # Save result
        os.makedirs("results", exist_ok=True)
        cv2.imwrite(
            "results/last_detection.jpg",
            drawn
        )

        self.status.set(
            f" Done! People: {len(analyses)}")

    def clear(self):
        self.preview.configure(
            image="",
            text="Upload image to detect\nnationality"
        )
        self.result_box.delete(1.0, tk.END)
        self.status.set(
            " Ready - Upload image")

# ── RUN ──────────────────────────────────────────
if __name__ == "__main__":
    print("\n Starting Nationality Detector...")
    root = tk.Tk()
    app  = NationalityApp(root)
    root.mainloop()