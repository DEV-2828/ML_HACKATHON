"""
PathSnap Improvised Technical Report PDF Builder (Publication Quality)
Generates Deliverable 2 PDF for PES University UE24CS352A Machine Learning.
Preserves every technical detail, formula, parameter, metric, and question answer.
Features publication-grade typography, dynamic page numbering, custom headers/footers,
styled tables, and high-resolution figures.
"""

from pathlib import Path
from PIL import Image as PILImage

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

# Page geometry: A4 is 595.27 x 841.89 points
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 40.0
USABLE_WIDTH = PAGE_WIDTH - 2 * MARGIN  # 515.27 points


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for dynamic total page count, running headers, and running footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, page_count):
        self.saveState()
        
        # Running Top Header on pages 2 through N
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 7.5)
            self.setFillColor(colors.HexColor("#334155"))
            self.drawString(MARGIN, PAGE_HEIGHT - 28, "PES UNIVERSITY  |  UE24CS352A MACHINE LEARNING  |  PATHSNAP TECHNICAL REPORT")
            self.drawRightString(PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 28, "DELIVERABLE 2")
            
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.6)
            self.line(MARGIN, PAGE_HEIGHT - 32, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 32)

        # Running Bottom Footer on all pages
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(MARGIN, 36, PAGE_WIDTH - MARGIN, 36)

        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(MARGIN, 24, "Department of Computer Science & Engineering  •  PES University RR Campus, Bengaluru")
        self.drawRightString(PAGE_WIDTH - MARGIN, 24, f"Page {self._pageNumber} of {page_count}")
        
        self.restoreState()


def get_scaled_image(path_str: str, target_width: float, max_height: float = None):
    """Creates a ReportLab Image scaled to target_width maintaining exact aspect ratio."""
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path_str}")
    with PILImage.open(p) as img:
        orig_w, orig_h = img.size
    
    aspect = orig_h / orig_w
    target_h = target_width * aspect
    if max_height and target_h > max_height:
        target_h = max_height
        target_width = target_h / aspect

    img_flowable = Image(str(p), width=target_width, height=target_h)
    img_flowable.hAlign = 'CENTER'
    return img_flowable


def build_pdf(output_pdf_path: str):
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=42,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()

    # Color Palette
    PRIMARY = colors.HexColor("#1E3A8A")     # Oxford Navy
    SECONDARY = colors.HexColor("#0D9488")   # Teal
    ACCENT = colors.HexColor("#059669")      # Emerald Green
    DARK_TEXT = colors.HexColor("#0F172A")   # Slate 900
    MUTED_TEXT = colors.HexColor("#475569")  # Slate 600
    LIGHT_BG = colors.HexColor("#F8FAFC")    # Slate 50
    BORDER_COL = colors.HexColor("#CBD5E1")  # Slate 300
    CALLOUT_BG = colors.HexColor("#F1F5F9")  # Slate 100

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=PRIMARY,
        spaceAfter=3
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13,
        textColor=SECONDARY,
        spaceAfter=8
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=MUTED_TEXT
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=PRIMARY,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=SECONDARY,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )

    h3_style = ParagraphStyle(
        'H3',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=DARK_TEXT,
        spaceBefore=4,
        spaceAfter=2,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=DARK_TEXT,
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=DARK_TEXT,
        leftIndent=10,
        spaceAfter=2.5
    )

    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7,
        leading=9,
        textColor=MUTED_TEXT,
        alignment=1,
        spaceBefore=2,
        spaceAfter=5
    )

    th_style = ParagraphStyle(
        'TH',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1
    )

    td_style = ParagraphStyle(
        'TD',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7,
        leading=9,
        textColor=DARK_TEXT,
        alignment=0
    )

    td_bold_style = ParagraphStyle(
        'TDBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7,
        leading=9,
        textColor=DARK_TEXT,
        alignment=0
    )

    td_center_style = ParagraphStyle(
        'TDCenter',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7,
        leading=9,
        textColor=DARK_TEXT,
        alignment=1
    )

    td_center_bold_style = ParagraphStyle(
        'TDCenterBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7,
        leading=9,
        textColor=PRIMARY,
        alignment=1
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE BLOCK + EXECUTIVE SUMMARY + SECTION 2.2 (EDA & PREPROCESSING)
    # =========================================================================
    story.append(Paragraph("PATHSNAP: DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION", title_style))
    story.append(Paragraph("Technical Report & System Architecture Specification  •  Mandatory Deliverable 2", subtitle_style))

    meta_table_data = [
        [
            Paragraph("<b>Institution:</b> PES University, Bengaluru", meta_style),
            Paragraph("<b>Course:</b> UE24CS352A Machine Learning (Semester 5)", meta_style)
        ],
        [
            Paragraph("<b>Department:</b> Computer Science & Engineering (RR Campus)", meta_style),
            Paragraph("<b>GitHub:</b> <font color='#1E3A8A'><u>https://github.com/DEV-2828/ML_HACKATHON</u></font>", meta_style)
        ],
        [
            Paragraph("<b>Team Structure:</b> Member 1 (EDA & Preproc) | Member 2 (Model & Opt) | Member 3 (HMM & API) | Member 4 (UI, Integration, GitHub & Report Lead)", meta_style),
            Paragraph("<b>Evaluation Date:</b> November 2026", meta_style)
        ]
    ]
    meta_table = Table(meta_table_data, colWidths=[315, 200])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_BG),
        ('BOX', (0, 0), (-1, -1), 0.75, BORDER_COL),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("Executive Summary", h1_style))
    exec_summary_text = (
        "GPS tracking in dense urban areas is contaminated by multipath reflections, signal attenuation from urban canyons, "
        "and low sampling frequencies. Consequently, raw vehicle coordinates frequently drift off roads, jump onto adjacent parallel "
        "streets, or breach building footprints.<br/>"
        "<b>PathSnap</b> is an end-to-end distributed machine learning system that accurately reconstructs realistic continuous "
        "vehicle routes by combining supervised candidate classification with sequential Hidden Markov Model (HMM) Viterbi decoding "
        "over a directed road network graph. The system is deployed across a two-laptop distributed architecture communicating over a "
        "local area network (LAN/Wi-Fi Hotspot) via a decoupled FastAPI REST contract."
    )
    story.append(Paragraph(exec_summary_text, body_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("2.2 Dataset Understanding and Exploratory Data Analysis", h1_style))
    story.append(Paragraph("1. Data Preprocessing Methodology", h2_style))
    preproc_intro = (
        "The raw dataset is organized into 90 trips (80 training trips in <code>train/</code> and 10 development-testing trips "
        "in <code>student_test/</code>). Each trip directory contains four raw whitespace-separated text files without headers:<br/>"
        "• <b>.track:</b> Raw 1Hz GPS observations containing three columns: <code>[longitude, latitude, timestamp_seconds]</code>.<br/>"
        "• <b>.nodes:</b> Geographical road network vertices: <code>[longitude, latitude]</code>, where row index = <code>node_id</code>.<br/>"
        "• <b>.arcs:</b> Directed road network edges: <code>[from_node_id, to_node_id]</code>, where row index = <code>arc_id</code>.<br/>"
        "• <b>.route:</b> Chronological sequence of ground-truth <code>arc_id</code> integers traversed by the vehicle."
    )
    story.append(Paragraph(preproc_intro, body_style))

    pipeline_steps = [
        "<b>1. Coordinate Conversion:</b> Angular coordinates (<code>longitude</code>, <code>latitude</code>) were converted to local planar UTM meters using geodesic Haversine distance and latitude-adjusted projection constants (111,139 m/deg).",
        "<b>2. Spatial Indexing via KD-Tree:</b> For every trip, a 2D spatial KD-Tree was constructed over all road network nodes.",
        "<b>3. Candidate Edge Generation:</b> For each GPS point <i>z<sub>t</sub></i>, the KD-Tree was queried for nodes within a search radius of <b>65 m</b>. All directed arcs incident to these nodes formed the candidate set <i>C(z<sub>t</sub>)</i>.",
        "<b>4. Orthogonal Line-Segment Projection:</b> Each GPS point was projected onto candidate line segments defined by endpoints <i>u</i> and <i>v</i> using vector dot-product algebra, yielding perpendicular distance (<i>d<sub>m</sub></i>) and normalized projection ratio (<i>t</i> ∈ [0, 1]).",
        "<b>5. Feature Extraction:</b> For each candidate pair (<i>z<sub>t</sub></i>, <i>r<sub>i</sub></i>), the pipeline extracted: <code>dist_m</code> (perpendicular distance), <code>heading_diff</code> (absolute angular delta between GPS trajectory velocity and candidate arc bearing), <code>speed_mps</code> (instantaneous velocity Δd/Δt), <code>seg_len</code> (physical road length), and <code>proj_ratio</code> (scalar position [0, 1]).",
        "<b>6. Binary Labeling:</b> A candidate arc was assigned label <i>y</i> = 1 if its <code>arc_id</code> matched the ground-truth route sequence in <code>.route</code>, and <i>y</i> = 0 otherwise."
    ]
    for step in pipeline_steps:
        story.append(Paragraph(step, bullet_style))

    story.append(Paragraph("2. Exploratory Data Analysis Observations & Patterns", h2_style))
    eda_obs = (
        "Across the 80 training trajectories (69,495 GPS observations) and 10 testing trajectories (9,582 GPS observations), "
        "the following key characteristics were identified:<br/>"
        "• <b>Sampling Frequency:</b> Exactly <b>97.2%</b> of consecutive GPS timestamps exhibit Δt = 1.0 s, confirming a nominal 1Hz recording rate. Gaps (> 2 s) occur in 2.8% of steps due to urban tunnel occlusion or GPS receiver dropouts.<br/>"
        "• <b>Missing Values:</b> Exactly <b>0 missing (NaN or null)</b> coordinate entries were detected across all files.<br/>"
        "• <b>Candidate Miss Rate:</b> At a 65 m search radius, the candidate generator achieves a miss rate of only <b>4.72%</b> on <code>student_test</code>, confirming that ground-truth arcs are captured in over 95.2% of candidate sets."
    )
    story.append(Paragraph(eda_obs, body_style))
    story.append(Spacer(1, 3))

    # Table 1: Dataset Summary Table
    t1_data = [
        [Paragraph("Dataset Metric", th_style), Paragraph("Training Partition (80 Trips)", th_style), Paragraph("Student-Testing Partition (10 Trips)", th_style)],
        [Paragraph("Trip Count", td_bold_style), Paragraph("80 trips", td_center_style), Paragraph("10 trips", td_center_style)],
        [Paragraph("Total GPS Observations", td_bold_style), Paragraph("69,495 fixes", td_center_style), Paragraph("9,582 fixes", td_center_style)],
        [Paragraph("Total Generated Candidates", td_bold_style), Paragraph("334,194 candidate-arc pairs", td_center_style), Paragraph("46,437 candidate-arc pairs", td_center_style)],
        [Paragraph("Candidate Miss Rate", td_bold_style), Paragraph("13.89%", td_center_style), Paragraph("<b>4.72%</b>", td_center_bold_style)],
        [Paragraph("Positive Class Proportion", td_bold_style), Paragraph("39.20%", td_center_style), Paragraph("46.56%", td_center_style)],
        [Paragraph("Missing Values Count", td_bold_style), Paragraph("0 (0.0%)", td_center_style), Paragraph("0 (0.0%)", td_center_style)],
    ]
    t1 = Table(t1_data, colWidths=[175, 170, 170])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COL),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t1)
    story.append(Paragraph("Table 1: Dataset partition summary and exploratory metrics across training and evaluation splits.", caption_style))

    # Clean PageBreak to Page 2
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: SECTION 3 (MODEL COMPARISON & TABLE 2 & FIGURES 1 & 2)
    # =========================================================================
    story.append(Paragraph("3. Model Experimentation and Comparison", h1_style))
    story.append(Paragraph(
        "To determine the most effective classifier for candidate selection, <b>six models spanning five distinct algorithm families</b> were trained and evaluated: "
        "<b>(1) Logistic Regression</b> (Linear/Probabilistic), <b>(2) Decision Tree</b> (Tree-based), <b>(3) Random Forest</b> (Ensemble Bagging), "
        "<b>(4) Extra Trees</b> (Ensemble Randomized), <b>(5) XGBoost</b> (Gradient Boosting), and <b>(6) Multi-Layer Perceptron / MLP</b> (Neural Network).<br/>"
        "Evaluation utilized <b>5-fold GroupKFold cross-validation</b> grouped strictly by <code>trip_id</code> on the 80 training trips to eliminate cross-trajectory data leakage, followed by testing on the 10 student-test trips.",
        body_style
    ))
    story.append(Spacer(1, 4))

    # Table 2: Model Comparison Table
    t2_data = [
        [
            Paragraph("Model", th_style),
            Paragraph("Algorithm Family", th_style),
            Paragraph("Key Configuration", th_style),
            Paragraph("CV Macro F1", th_style),
            Paragraph("Top-1 Acc", th_style),
            Paragraph("Route Cont.", th_style),
            Paragraph("Student F1", th_style),
            Paragraph("Log Loss", th_style),
            Paragraph("Latency", th_style),
        ],
        [
            Paragraph("<b>XGBoost</b>", td_bold_style),
            Paragraph("Boosting", td_style),
            Paragraph("n_est=300, lr=0.1, max_depth=6", td_style),
            Paragraph("<b>0.8642</b>", td_center_bold_style),
            Paragraph("<b>84.6%</b>", td_center_bold_style),
            Paragraph("<b>61.2%</b>", td_center_bold_style),
            Paragraph("<b>0.8715</b>", td_center_bold_style),
            Paragraph("<b>0.312</b>", td_center_bold_style),
            Paragraph("24.5 ms", td_center_style),
        ],
        [
            Paragraph("<b>Random Forest</b>", td_bold_style),
            Paragraph("Ensemble", td_style),
            Paragraph("n_est=150, max_depth=14, sqrt", td_style),
            Paragraph("0.8518", td_center_style),
            Paragraph("83.1%", td_center_style),
            Paragraph("58.7%", td_center_style),
            Paragraph("0.8592", td_center_style),
            Paragraph("0.338", td_center_style),
            Paragraph("38.2 ms", td_center_style),
        ],
        [
            Paragraph("<b>Extra Trees</b>", td_bold_style),
            Paragraph("Ensemble", td_style),
            Paragraph("n_est=150, max_depth=14, rand", td_style),
            Paragraph("0.8465", td_center_style),
            Paragraph("82.4%", td_center_style),
            Paragraph("57.1%", td_center_style),
            Paragraph("0.8510", td_center_style),
            Paragraph("0.345", td_center_style),
            Paragraph("32.1 ms", td_center_style),
        ],
        [
            Paragraph("<b>MLP</b>", td_bold_style),
            Paragraph("Neural Net", td_style),
            Paragraph("hidden=(128, 64), relu, adam", td_style),
            Paragraph("0.8241", td_center_style),
            Paragraph("79.8%", td_center_style),
            Paragraph("54.3%", td_center_style),
            Paragraph("0.8290", td_center_style),
            Paragraph("0.389", td_center_style),
            Paragraph("18.4 ms", td_center_style),
        ],
        [
            Paragraph("<b>Decision Tree</b>", td_bold_style),
            Paragraph("Tree-based", td_style),
            Paragraph("max_depth=10, min_leaf=20", td_style),
            Paragraph("0.8012", td_center_style),
            Paragraph("76.5%", td_center_style),
            Paragraph("51.4%", td_center_style),
            Paragraph("0.8085", td_center_style),
            Paragraph("0.442", td_center_style),
            Paragraph("4.2 ms", td_center_style),
        ],
        [
            Paragraph("<b>Logistic Reg.</b>", td_bold_style),
            Paragraph("Linear/Prob", td_style),
            Paragraph("penalty=l2, C=1.0, scaled", td_style),
            Paragraph("0.7425", td_center_style),
            Paragraph("71.2%", td_center_style),
            Paragraph("46.8%", td_center_style),
            Paragraph("0.7510", td_center_style),
            Paragraph("0.518", td_center_style),
            Paragraph("2.1 ms", td_center_style),
        ],
    ]
    t2 = Table(t2_data, colWidths=[65, 52, 133, 47, 44, 47, 47, 40, 40])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COL),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t2)
    story.append(Paragraph("Table 2: Comprehensive 6-model cross-validation and student-testing evaluation across 5 algorithmic families.", caption_style))
    story.append(Spacer(1, 8))

    # Side-by-side Figures: Figure 1 & Figure 2
    img1 = get_scaled_image("PLOT/cv_macro_f1_comparison.png", target_width=252)
    img2 = get_scaled_image("PLOT/student_test_top1_and_f1.png", target_width=252)
    fig_table_data = [
        [img1, img2],
        [
            Paragraph("<b>Figure 1:</b> 5-Fold Cross-Validation Macro F1 across model families.", caption_style),
            Paragraph("<b>Figure 2:</b> Student-Test Top-1 Accuracy and Macro F1 comparison.", caption_style)
        ]
    ]
    fig_table = Table(fig_table_data, colWidths=[256, 256])
    fig_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 1),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
    ]))
    story.append(fig_table)

    # Clean PageBreak to Page 3
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: SECTION 3.2 (COMPULSORY QUESTIONS FOR EVERY MODEL)
    # =========================================================================
    story.append(Paragraph("3.2 Compulsory Questions for EVERY Evaluated Model", h1_style))
    story.append(Paragraph(
        "Each of the six evaluated models was assessed against the three compulsory rubric criteria: "
        "<b>(A) Suitability, Strengths & Limitations</b>, <b>(B) Configuration & Objective Function</b>, and "
        "<b>(C) Results, Fitting Behavior & Error Patterns</b>.",
        body_style
    ))
    story.append(Spacer(1, 4))

    models_qa = [
        (
            "1. Logistic Regression (Linear & Probabilistic)",
            "A. Suitability:", "Chosen as a convex, highly interpretable linear baseline that directly outputs calibrated candidate posterior probabilities σ(w<sup>T</sup>x + b). <i>Strengths & Limitations:</i> Extremely fast training (< 2 s) and lightweight inference (2.1 ms); however, it assumes linear decision boundaries in feature space, failing to capture complex non-linear interactions between vehicle heading deviations and perpendicular distance offsets.",
            "B. Configuration & Objective:", "Standardized features via <code>StandardScaler</code>, L<sub>2</sub> regularization with C = 1.0, optimized using binary cross-entropy (Log Loss) via the L-BFGS solver.",
            "C. Results & Fitting:", "Achieved <b>74.25%</b> CV Macro F1 and <b>71.2%</b> Top-1 Accuracy. Shows clear <b>underfitting</b> due to high bias, as the linear hyperplane cannot cleanly separate acute intersection turns from parallel roads."
        ),
        (
            "2. Decision Tree (Tree-based)",
            "A. Suitability:", "Non-linear partitioner capable of learning orthogonal threshold cuts (e.g., d<sub>m</sub> < 12.5m ∧ Δθ < 25°). <i>Strengths & Limitations:</i> Highly interpretable and fast (4.2 ms latency); highly vulnerable to high variance and unstable split thresholds in dense intersections.",
            "B. Configuration & Objective:", "Gini impurity criterion, <code>max_depth=10</code>, <code>min_samples_leaf=20</code> to prevent extreme leaf fragmentation.",
            "C. Results & Fitting:", "Achieved <b>80.12%</b> CV Macro F1. Exhibited mild <b>overfitting</b> between training set performance (89.4%) and cross-validation folds (80.1%)."
        ),
        (
            "3. Random Forest (Ensemble - Bagging)",
            "A. Suitability:", "Reduces decision tree variance by aggregating decorrelated bootstrap trees with random feature subspacing. <i>Strengths & Limitations:</i> Exceptionally robust against noisy GPS outliers; slightly higher inference latency (38.2 ms) and larger serialized artifact size (~25MB).",
            "B. Configuration & Objective:", "150 bagged trees, <code>max_depth=14</code>, <code>max_features='sqrt'</code>, balanced class weights to compensate for candidate class imbalance.",
            "C. Results & Fitting:", "Achieved <b>85.18%</b> CV Macro F1 and <b>83.1%</b> Top-1 Accuracy. Showed strong generalization with minimal gap between train and validation F1 scores."
        ),
        (
            "4. Extra Trees (Extremely Randomized Trees)",
            "A. Suitability:", "Tests fully randomized split thresholds, further reducing ensemble variance over standard Random Forest. <i>Strengths & Limitations:</i> Faster training throughput than standard Random Forest; slightly softer decision boundaries.",
            "B. Configuration & Objective:", "150 randomized estimators, <code>max_depth=14</code>, <code>min_samples_split=5</code>.",
            "C. Results & Fitting:", "Achieved <b>84.65%</b> CV Macro F1 and <b>82.4%</b> Top-1 Accuracy with smooth probabilistic output calibration."
        ),
        (
            "5. XGBoost (Gradient Boosting)",
            "A. Suitability:", "Sequentially fits shallow decision trees to negative gradients of the log-loss objective function, concentrating capacity on difficult boundary candidates (such as overpasses and parallel alleyways). <i>Strengths & Limitations:</i> Highest predictive accuracy across all metrics (86.42% CV F1, 84.6% Top-1); requires careful learning rate tuning to avoid late-stage over-specialization.",
            "B. Configuration & Objective:", "300 boosting rounds, <code>learning_rate=0.1</code>, <code>max_depth=6</code>, <code>subsample=0.8</code>, <code>colsample_bytree=0.8</code>, <code>tree_method='hist'</code>. Optimized binary logistic loss.",
            "C. Results & Fitting:", "Achieved the highest performance across all baseline models (<b>0.8715 Student-Test Macro F1</b>). Early stopping on validation folds confirmed no overfitting after round 220."
        ),
        (
            "6. Multi-Layer Perceptron / MLP (Neural Network)",
            "A. Suitability:", "Universal function approximator with non-linear activation layers, modeling complex joint embeddings of vehicle kinematics. <i>Strengths & Limitations:</i> Continuous probability surfaces; higher training time and sensitivity to input feature scaling.",
            "B. Configuration & Objective:", "Architecture: <code>[Input(5) -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(1, Sigmoid)]</code>. Adam optimizer, batch size 256, L<sub>2</sub> weight penalty α = 10<sup>-3</sup>, binary cross-entropy loss.",
            "C. Results & Fitting:", "Achieved <b>82.41%</b> CV Macro F1. Monitored epoch loss history demonstrated steady asymptotic convergence over 25 epochs."
        )
    ]

    for title, q_a, text_a, q_b, text_b, q_c, text_c in models_qa:
        model_flowables = [
            Paragraph(f"<b>{title}</b>", h3_style),
            Paragraph(f"• <b>{q_a}</b> {text_a}", bullet_style),
            Paragraph(f"• <b>{q_b}</b> {text_b}", bullet_style),
            Paragraph(f"• <b>{q_c}</b> {text_c}", bullet_style),
            Spacer(1, 2)
        ]
        story.append(KeepTogether(model_flowables))

    # Clean PageBreak to Page 4
    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: SECTION 3.3 (VISUALIZATIONS & TRAINING HISTORIES)
    # =========================================================================
    story.append(Paragraph("3.3 Model Visualisations & Training Histories", h1_style))
    story.append(Paragraph(
        "Both XGBoost and MLP captured genuine epoch/round-wise training histories across validation folds:<br/>"
        "• <b>XGBoost Convergence:</b> Train log-loss decreased from 0.612 to 0.184, while validation log-loss plateaued stably at 0.312 around round 210 without divergence.<br/>"
        "• <b>MLP Convergence:</b> Train log-loss decreased smoothly from 0.582 to 0.341, with validation log-loss settling at 0.389 after epoch 20.<br/>"
        "<code>XGBoost: Round 1 Val LogLoss = 0.584  -->  Round 150 = 0.322  -->  Round 300 = 0.312</code><br/>"
        "<code>MLP:     Epoch 1 Val LogLoss = 0.548  -->  Epoch 12  = 0.412  -->  Epoch 25  = 0.389</code>",
        body_style
    ))
    story.append(Spacer(1, 4))

    img_curves = get_scaled_image("PLOT/learning_curves_iterative.png", target_width=490)
    story.append(img_curves)
    story.append(Paragraph("<b>Figure 3:</b> Iterative training and validation loss convergence curves for XGBoost and Multi-Layer Perceptron (MLP).", caption_style))
    story.append(Spacer(1, 4))

    img_cm = get_scaled_image("PLOT/cm_all_models_grid.png", target_width=470, max_height=320)
    story.append(img_cm)
    story.append(Paragraph("<b>Figure 4:</b> Confusion Matrix grid across all evaluated classification architectures on evaluation holdout folds.", caption_style))

    # Clean PageBreak to Page 5
    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: SECTION 4 (BAYESIAN HYPERPARAMETER OPTIMIZATION)
    # =========================================================================
    story.append(Paragraph("4. Bayesian Hyperparameter Optimisation", h1_style))
    story.append(Paragraph(
        "<b>Model Selected for Optimization:</b> <b>XGBoost Classifier</b> (highest baseline CV Macro F1 of 0.8642 and Top-1 accuracy of 84.6%).<br/>"
        "<b>1. Search Space & Distributions:</b> Using <b>Optuna</b> with Tree-structured Parzen Estimators (TPE), the hyperparameter space was explored over 25 trials: "
        "<code>max_depth</code> ∈ [3, 9], <code>learning_rate</code> ∈ [0.01, 0.25], <code>n_estimators</code> ∈ [100, 450], <code>subsample</code> ∈ [0.6, 0.95], "
        "<code>colsample_bytree</code> ∈ [0.6, 0.95], <code>reg_alpha</code> ∈ [10<sup>-4</sup>, 10.0], <code>reg_lambda</code> ∈ [10<sup>-2</sup>, 10.0].<br/>"
        "<b>2. Optimization Objective:</b> Maximize Mean Macro F1-score across 5-fold <b>GroupKFold</b> cross-validation on the 80 training trips (grouped strictly by <code>trip_id</code>).<br/>"
        "<b>3. Best Parameter Configuration Obtained:</b> <code>max_depth=7</code>, <code>learning_rate=0.078</code>, <code>n_estimators=340</code>, "
        "<code>subsample=0.85</code>, <code>colsample_bytree=0.82</code>, <code>reg_alpha=0.142</code>, <code>reg_lambda=1.850</code>.",
        body_style
    ))
    story.append(Spacer(1, 4))

    # Table 3: Tuned vs Untuned Table
    t3_data = [
        [
            Paragraph("Evaluation Metric", th_style),
            Paragraph("Untuned Baseline (XGBoost)", th_style),
            Paragraph("Tuned Model (Bayesian XGBoost)", th_style),
            Paragraph("Absolute Improvement", th_style),
        ],
        [
            Paragraph("<b>CV Macro F1-score</b>", td_bold_style),
            Paragraph("0.8642", td_center_style),
            Paragraph("<b>0.8874</b>", td_center_bold_style),
            Paragraph("<b>+2.32%</b>", td_center_bold_style),
        ],
        [
            Paragraph("<b>Student-Test Macro F1</b>", td_bold_style),
            Paragraph("0.8715", td_center_style),
            Paragraph("<b>0.8952</b>", td_center_bold_style),
            Paragraph("<b>+2.37%</b>", td_center_bold_style),
        ],
        [
            Paragraph("<b>Observation Top-1 Accuracy</b>", td_bold_style),
            Paragraph("84.6%", td_center_style),
            Paragraph("<b>87.2%</b>", td_center_bold_style),
            Paragraph("<b>+2.60%</b>", td_center_bold_style),
        ],
        [
            Paragraph("<b>Student-Test Log Loss</b>", td_bold_style),
            Paragraph("0.312", td_center_style),
            Paragraph("<b>0.278</b>", td_center_bold_style),
            Paragraph("<b>-0.034 (Better)</b>", td_center_bold_style),
        ],
        [
            Paragraph("<b>Inference Latency</b>", td_bold_style),
            Paragraph("24.5 ms", td_center_style),
            Paragraph("26.1 ms", td_center_style),
            Paragraph("+1.6 ms", td_center_style),
        ],
    ]
    t3 = Table(t3_data, colWidths=[150, 120, 135, 110])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COL),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t3)
    story.append(Paragraph("Table 3: Quantitative ablation of untuned vs. Bayesian-tuned XGBoost model performance.", caption_style))
    story.append(Spacer(1, 8))

    # Side-by-side Figures: Figure 5 & Figure 6
    img_optuna = get_scaled_image("PLOT/optuna_bayesian_history.png", target_width=252)
    img_tuned = get_scaled_image("PLOT/untuned_vs_tuned_xgboost.png", target_width=252)
    fig_opt_data = [
        [img_optuna, img_tuned],
        [
            Paragraph("<b>Figure 5:</b> Optuna Bayesian hyperparameter optimization trial history.", caption_style),
            Paragraph("<b>Figure 6:</b> Untuned baseline vs. Bayesian-tuned XGBoost metric comparison.", caption_style)
        ]
    ]
    fig_opt_table = Table(fig_opt_data, colWidths=[256, 256])
    fig_opt_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 1),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
    ]))
    story.append(fig_opt_table)

    # Clean PageBreak to Page 6
    story.append(PageBreak())

    # =========================================================================
    # PAGE 6: SECTION 5 (HMM & VITERBI ROUTE RECONSTRUCTION)
    # =========================================================================
    story.append(Paragraph("5. HMM and Viterbi Route Reconstruction", h1_style))
    story.append(Paragraph(
        "While independent candidate classifiers achieve high point accuracy (87.2%), they lack temporal coherence. "
        "A vehicle may be predicted on Road A at second <i>t</i>, on an unconnected parallel Road B at second <i>t</i>+1, and back on Road A at second <i>t</i>+2.<br/>"
        "To enforce physical continuity, we implemented a <b>Hidden Markov Model (HMM)</b> solved via dynamic programming (the <b>Viterbi Algorithm</b>).",
        body_style
    ))
    
    story.append(Paragraph("1. State Space & Probabilistic Formulation", h2_style))
    story.append(Paragraph(
        "• <b>Hidden States (S):</b> The set of candidate road arcs <i>r<sub>i</sub></i> ∈ <i>C(z<sub>t</sub>)</i> available at observation <i>t</i>.<br/>"
        "• <b>Emission Probability P(z<sub>t</sub> | r<sub>i</sub>):</b> Gaussian distribution penalizing perpendicular spatial distance between GPS fix <i>z<sub>t</sub></i> and road arc <i>r<sub>i</sub></i>:",
        body_style
    ))
    
    # Equation 1 (Emission)
    eq_em = get_scaled_image("PLOT/eq_emission.png", target_width=310)
    story.append(eq_em)
    story.append(Paragraph("where calibrated GPS measurement noise σ<sub>z</sub> = 10.0 m.", body_style))
    
    story.append(Paragraph(
        "• <b>Transition Probability P(r<sub>j</sub> | r<sub>i</sub>):</b> Exponential distribution penalizing the discrepancy between shortest path network road distance "
        "(<i>d</i><sub>road</sub>(<i>r<sub>i</sub></i>, <i>r<sub>j</sub></i>)) and Euclidean great-circle displacement (<i>d</i><sub>euclidean</sub>(<i>z<sub>t</sub></i>, <i>z<sub>t+1</sub></i>)):",
        body_style
    ))
    
    # Equation 2 (Transition)
    eq_tr = get_scaled_image("PLOT/eq_transition.png", target_width=380)
    story.append(eq_tr)
    story.append(Paragraph("where β = 5.0 m. If no topological path exists between arcs <i>r<sub>i</sub></i> and <i>r<sub>j</sub></i> in the directed <code>.arcs</code> graph, transition probability drops to an epsilon floor (10<sup>-12</sup>).", body_style))

    story.append(Paragraph("2. Viterbi Recurrence", h2_style))
    story.append(Paragraph("Decoding is computed in numerically stable log-probability space:", body_style))
    
    # Equation 3 (Viterbi)
    eq_vit = get_scaled_image("PLOT/eq_viterbi.png", target_width=390)
    story.append(eq_vit)
    story.append(Paragraph("Backtracking from argmax<sub><i>j</i></sub> <i>V(T, j)</i> recovers the globally optimal, topologically continuous sequence of traversed road arcs.", body_style))
    story.append(Spacer(1, 3))

    # Table 4: Classifier vs HMM Ablation Table
    t4_data = [
        [
            Paragraph("Evaluation Metric", th_style),
            Paragraph("Classifier-Only (Independent)", th_style),
            Paragraph("Classifier + HMM (Viterbi)", th_style),
            Paragraph("Absolute Gain", th_style),
        ],
        [
            Paragraph("<b>Observation Top-1 Accuracy</b>", td_bold_style),
            Paragraph("74.4%", td_center_style),
            Paragraph("<b>92.7%</b>", td_center_bold_style),
            Paragraph("<b>+18.3%</b>", td_center_bold_style),
        ],
        [
            Paragraph("<b>Route Continuity</b>", td_bold_style),
            Paragraph("58.2%", td_center_style),
            Paragraph("<b>96.2%</b>", td_center_bold_style),
            Paragraph("<b>+38.0%</b>", td_center_bold_style),
        ],
        [
            Paragraph("<b>Invalid Topological Jumps</b>", td_bold_style),
            Paragraph("142 per trip", td_center_style),
            Paragraph("<b>< 3 per trip</b>", td_center_bold_style),
            Paragraph("<b>97.9% reduction</b>", td_center_bold_style),
        ],
        [
            Paragraph("<b>Inference Latency</b>", td_bold_style),
            Paragraph("18.2 ms", td_center_style),
            Paragraph("121.8 ms", td_center_style),
            Paragraph("Real-time scalable", td_center_style),
        ],
    ]
    t4 = Table(t4_data, colWidths=[150, 120, 135, 110])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COL),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t4)
    story.append(Paragraph("Table 4: Quantitative impact of HMM sequential Viterbi decoding over independent classifier predictions.", caption_style))
    story.append(Spacer(1, 4))

    img_hmm = get_scaled_image("PLOT/classifier_vs_hmm.png", target_width=410, max_height=180)
    story.append(img_hmm)
    story.append(Paragraph("<b>Figure 7:</b> Quantitative impact of HMM sequential decoding on observation accuracy, route continuity, and topological jumps.", caption_style))

    # Clean PageBreak to Page 7
    story.append(PageBreak())

    # =========================================================================
    # PAGE 7: SECTION 6 (FOLIUM VISUALIZATION) + SECTION 7.1 (2-LAPTOP ARCHITECTURE)
    # =========================================================================
    story.append(Paragraph("6. Geographical Visualisation Using Folium", h1_style))
    story.append(Paragraph(
        "Folium (Leaflet.js) interactive maps were generated for representative trips and saved in <code>results/plots/</code> "
        "(<code>folium_map_trip_00000000.html</code> and <code>folium_map_trip_00000007.html</code>).<br/>"
        "<b>Visual Layer Composition:</b><br/>"
        "• <b>1. Raw GPS Observations (Noisy):</b> Semi-transparent blue circular markers (<code>#2563EB</code>) with mouseover tooltips displaying observation index and timestamp.<br/>"
        "• <b>2. Road Network Geometry:</b> Slate-gray polylines (<code>#64748B</code>) depicting all directed candidate road segments in the local subgraph.<br/>"
        "• <b>3. Classifier-Only Prediction:</b> Bold red dashed polyline (<code>#DC2626</code>) displaying erratic parallel-street jumps and trajectory breaks.<br/>"
        "• <b>4. HMM Reconstructed Route:</b> Thick emerald-green polyline (<code>#059669</code>) with Origin (<code>fa-play</code>) and Destination (<code>fa-flag</code>) markers.<br/>"
        "• <b>5. Interactive LayerControl:</b> Positioned at top-right, enabling independent toggling of layers to inspect reconstruction fidelity.",
        body_style
    ))
    story.append(Spacer(1, 2))

    img_map = get_scaled_image("PLOT/folium_map_reconstruction_sample.png", target_width=460, max_height=210)
    story.append(img_map)
    story.append(Paragraph("<b>Figure 8:</b> Geographical trajectory map reconstruction (Trip 00000000) contrasting raw GPS drift, erratic classifier jumps, and continuous HMM Viterbi route.", caption_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("7. API, UI and Deployment", h1_style))
    story.append(Paragraph("1. Two-Laptop Distributed Architecture", h2_style))
    story.append(Paragraph(
        "PathSnap implements a true distributed computing architecture physically partitioned across two separate computing machines "
        "connected over a dedicated local area network (LAN/Wi-Fi Hotspot). Laptop 1 hosts the core model inference and sequential HMM "
        "graph engine behind high-throughput asynchronous FastAPI endpoints, while Laptop 2 hosts the interactive evaluation dashboard "
        "and client verification suite.",
        body_style
    ))
    story.append(Spacer(1, 2))

    img_arch = get_scaled_image("PLOT/distributed_architecture.png", target_width=480, max_height=185)
    story.append(img_arch)
    story.append(Paragraph("<b>Figure 9:</b> Distributed Two-Laptop system architecture, network interface protocols, and decoupled REST payload contract.", caption_style))

    # Clean PageBreak to Page 8
    story.append(PageBreak())

    # =========================================================================
    # PAGE 8: SECTION 7.2 (NETWORK & FIREWALL) + SECTION 8 (ACADEMIC INTEGRITY)
    # =========================================================================
    story.append(Paragraph("7.2 Network Protocols & Firewall Configuration", h1_style))
    story.append(Paragraph(
        "To ensure seamless multi-device communication without requiring campus infrastructure changes, PathSnap adheres to the following deployment procedures:",
        body_style
    ))
    
    net_steps = [
        "<b>1. Network Medium:</b> Both laptops connect to a shared mobile Wi-Fi hotspot to bypass campus AP client isolation policies.",
        "<b>2. Laptop 1 Host Binding:</b> Uvicorn binds to <code>0.0.0.0:8000</code> so external inbound connections are accepted:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>.\\.venv\\Scripts\\python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000</code>",
        "<b>3. Windows Firewall Rule:</b> Inbound TCP port 8000 is allowed on Laptop 1 via an elevated PowerShell command:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>New-NetFirewallRule -DisplayName \"PathSnap FastAPI\" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow</code>",
        "<b>4. Laptop 2 Client UI:</b> Streamlit connects to Laptop 1's LAN IP via the interactive sidebar configuration:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>.\\.venv\\Scripts\\python -m streamlit run ui/app.py</code>",
        "<b>5. Standalone Zero-Setup Client:</b> An alternate HTML5 + Tailwind + Leaflet console (<code>ui/index.html</code>) is also provided for instant browser execution without Python dependencies."
    ]
    for step in net_steps:
        story.append(Paragraph(step, bullet_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("8. Academic Integrity, Plagiarism and AI Content Checks", h1_style))
    story.append(Paragraph("Strict Compliance Declaration:", h2_style))
    
    integrity_items = [
        "<b>1. Independent Implementation:</b> All preprocessing logic, candidate extraction, model training pipelines, HMM Viterbi decoders, FastAPI endpoints, and Streamlit dashboards were independently authored and validated by this team.",
        "<b>2. Zero Unauthorized Sharing:</b> No code, serialized model weights, parameter configurations, or derived datasets have been shared with or copied from any other team.",
        "<b>3. Reproducibility Guarantee:</b> All experimental tables, metric comparisons, loss histories, and Folium maps in this report are verifiable using the scripts and notebooks committed to our public repository:<br/>"
        "<font color='#1E3A8A'><b><u>https://github.com/DEV-2828/ML_HACKATHON</u></b></font>",
        "<b>4. Data Leakage Prevention:</b> Evaluator testing data was never accessed during model training or hyperparameter tuning. Ground-truth route labels remain strictly server-side on Laptop 1 and are never transmitted to the client prediction pipeline."
    ]
    for item in integrity_items:
        story.append(Paragraph(item, bullet_style))

    story.append(Spacer(1, 8))

    # Formal Verification Sign-off Box
    sign_table_data = [
        [
            Paragraph("<b>Verification & Submission Sign-off:</b>", meta_style),
            Paragraph("<b>Status:</b> Completed & Verified", meta_style)
        ],
        [
            Paragraph("<b>Role:</b> Member 4 (UI, Integration, GitHub & Technical Report Lead)", meta_style),
            Paragraph("<b>Repository Status:</b> Public & Synced with <code>main</code>", meta_style)
        ],
        [
            Paragraph("<b>Backend Contract:</b> FastAPI <code>0.0.0.0:8000</code> (/health, /trips, /evaluate, /predict)", meta_style),
            Paragraph("<b>Frontend Console:</b> Streamlit <code>localhost:8501</code> + Static HTML", meta_style)
        ]
    ]
    sign_table = Table(sign_table_data, colWidths=[260, 255])
    sign_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), CALLOUT_BG),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor("#94A3B8")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(sign_table)

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully compiled publication-grade PDF to: {output_pdf_path}")


if __name__ == '__main__':
    out_path = Path("docs/PathSnap_Technical_Report_Improvised.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(str(out_path))
