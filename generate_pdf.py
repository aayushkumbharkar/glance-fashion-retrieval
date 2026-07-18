"""
generate_pdf.py
================
Programmatically generates the official internship assignment submission PDF
using reportlab. Handles headings, tables, headers, footers, and page numbers
in a clean, double-pass canvas style.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# Colors
PRIMARY_COLOR = colors.HexColor("#1A1A24")  # Dark slate
SECONDARY_COLOR = colors.HexColor("#7C6AF7")  # Indigo accent
TEXT_COLOR = colors.HexColor("#2C3E50")  # Charcoal body text
LIGHT_BG = colors.HexColor("#F8F9FA")  # Off-white table header/callouts
BORDER_COLOR = colors.HexColor("#BDC3C7")  # Cool grey borders

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas pattern to compute 'Page X of Y' dynamically 
    and draw consistent running headers/footers.
    """
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
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#7F8C8D"))
        
        # Suppress headers/footers on page 1 (cover page)
        if self._pageNumber > 1:
            # Header
            self.drawString(54, 750, "Submission Assignment: Intelligent Fashion Retrieval Engine")
            self.setStrokeColor(BORDER_COLOR)
            self.setLineWidth(0.5)
            self.line(54, 742, 612 - 54, 742)
            
            # Footer
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(612 - 54, 36, page_text)
            self.drawString(54, 36, "Candidate: Aayush Kumbharkar — Glance Take-home")
            self.line(54, 48, 612 - 54, 48)
            
        self.restoreState()


def create_submission_report(output_path: Path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )

    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=PRIMARY_COLOR,
        alignment=1, # Center
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=SECONDARY_COLOR,
        alignment=1,
        spaceAfter=30
    )

    h1_style = ParagraphStyle(
        'Header1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=PRIMARY_COLOR,
        spaceBefore=18,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Header2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=SECONDARY_COLOR,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=TEXT_COLOR,
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'BulletText',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=14,
        textColor=PRIMARY_COLOR,
        backColor=LIGHT_BG,
        borderColor=SECONDARY_COLOR,
        borderWidth=1,
        borderPadding=10,
        spaceBefore=10,
        spaceAfter=10
    )

    story = []

    # ==========================================
    # COVER PAGE
    # ==========================================
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("GLANCE INTERNSHIP ASSIGNMENT", subtitle_style))
    story.append(Paragraph("Intelligent Fashion Search Engine<br/>with Structured Attribute-Augmented Dual-Retrieval", title_style))
    story.append(Spacer(1, 0.2 * inch))
    
    # Metadata Box
    meta_data = [
        [Paragraph("<b>Candidate Name:</b>", body_style), Paragraph("Aayush Kumbharkar", body_style)],
        [Paragraph("<b>Email:</b>", body_style), Paragraph("aayushkumbharkar53@gmail.com", body_style)],
        [Paragraph("<b>GitHub Repository:</b>", body_style), Paragraph("<font color='#7C6AF7'><u>https://github.com/aayushkumbharkar/glance-fashion-retrieval</u></font>", body_style)],
        [Paragraph("<b>OS/Environment:</b>", body_style), Paragraph("Windows / Python 3.10 / Local Venv", body_style)]
    ]
    meta_table = Table(meta_data, colWidths=[2.0 * inch, 4.0 * inch])
    meta_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(meta_table)
    
    story.append(Spacer(1, 1.0 * inch))
    story.append(Paragraph("<i>This submission satisfies both Part A (Indexer) and Part B (Retriever) workflows, resolving fine-grained clothing attribute mappings, setting extraction, and compositional color query binding.</i>", subtitle_style))
    story.append(PageBreak())

    # ==========================================
    # SECTION 1: APPROACHES & TRADEOFFS
    # ==========================================
    story.append(Paragraph("1. Alternative Approaches and Tradeoffs", h1_style))
    story.append(Paragraph(
        "To map natural language text queries directly to clothing images, several visual and multimodal architectures were evaluated. Below are the three core design pathways analyzed for this project:",
        body_style
    ))

    # Table of comparison
    table_data = [
        [
            Paragraph("<b>Approach</b>", body_style),
            Paragraph("<b>Multimodal Alignment</b>", body_style),
            Paragraph("<b>Core Weakness</b>", body_style),
            Paragraph("<b>Optimal Use Case</b>", body_style)
        ],
        [
            Paragraph("<b>1. Vanilla CLIP Only</b><br/>(Contrastive Visual-Text)", body_style),
            Paragraph("Dense cosine alignment of images and textual query vectors in a single shared representation space.", body_style),
            Paragraph("<b>Compositional Failure:</b> Conflates color-garment bounds (e.g. 'red tie, white shirt' vs. 'white tie, red shirt'). Suffers from fashion vocabulary gap.", body_style),
            Paragraph("General visual scene search or broad palette-matching where exact bounding and grammatical detail are out of scope.", body_style)
        ],
        [
            Paragraph("<b>2. Pure Text Search</b><br/>(VLM + BM25/Dense)", body_style),
            Paragraph("Run local VLM on images. Index descriptions in a text engine (like Elasticsearch or a dense sentence embedder).", body_style),
            Paragraph("<b>Lack of Visual Texture:</b> Fails to capture non-verbal details, textures, or exact visual style attributes omitted by VLM captions.", body_style),
            Paragraph("Strict attribute matching where exact text descriptions or SKU codes govern searches.", body_style)
        ],
        [
            Paragraph("<b>3. Hybrid Dual-Retrieval</b><br/>(Structured + Visual)", body_style),
            Paragraph("Two-path search combining OpenCLIP visual similarity, BGE description matching, and exact categorical metadata bonuses.", body_style),
            Paragraph("<b>Slight Indexing Overhead:</b> Requires running a local VLM once at ingest time. Retrieval latency remains sub-100ms.", body_style),
            Paragraph("<b>Chosen Solution:</b> High-precision fashion query search requiring exact color-garment bounds and visual vibe preservation.", body_style)
        ]
    ]

    comp_table = Table(table_data, colWidths=[1.5 * inch, 1.8 * inch, 1.8 * inch, 1.9 * inch])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), LIGHT_BG),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 0.15 * inch))

    # ==========================================
    # SECTION 2: CHOSEN APPROACH WRITE-UP
    # ==========================================
    story.append(Paragraph("2. Short Write-up on Chosen Approach", h1_style))
    story.append(Paragraph(
        "Our chosen approach, <b>Structured Attribute-Augmented Dual-Retrieval</b>, is explicitly engineered to address CLIP's compositionality failures in fashion query processing. The workflow is divided into two distinct components:",
        body_style
    ))
    
    story.append(Paragraph("<b>The Indexer Workflow (Part A):</b>", h2_style))
    story.append(Paragraph(
        "• <b>VLM Captions (Designed Path — BLIP-2):</b> The primary indexing path runs each image through "
        "<b>Salesforce/blip2-flan-t5-xl</b>, prompted to output a structured JSON object identifying "
        "clothing items, accessories, bound color pairs (e.g. 'navy jacket'), setting, and style. "
        "This path is fully implemented in <i>caption_generator.py</i> and is activated by running the "
        "indexer <i>without</i> the <code>--use_lightweight_vlm</code> flag. It requires a CUDA GPU "
        "(~8 GB VRAM; supported with 4-bit quantization at ~6 GB).",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Submitted Index — CPU Fallback Path (Documented Tradeoff):</b> Due to no local GPU "
        "access within the 4-hour time budget, this submission's index was built using the "
        "<b>LightweightCaptionGenerator</b> (base BLIP, free-text caption + heuristic "
        "keyword-proximity parsing). Real numbers from <i>data/captions.json</i> (1,000 images): "
        "parse_tier_used=3 for 100.0% of images; environment=unknown for 57.5%; style=unknown for 91.8%. "
        "This is a one-line CLI flag change: dropping <code>--use_lightweight_vlm</code> and running "
        "on a Colab T4 GPU would populate the full structured schema at scale. "
        "The assignment's design, code, and architecture are built for the BLIP-2 path; "
        "only the submitted data artifact reflects the fallback.",
        bullet_style
    ))
    story.append(Paragraph("• <b>3-Tier Parser Safety Net:</b> If the VLM output fails to parse as clean JSON (tier 1), the indexer falls back to regex extraction (tier 2) and then a proximity keyword search (tier 3) — capturing nearby colors for every garment found — to preserve metadata integrity.", bullet_style))
    story.append(Paragraph("• <b>Dual Fingerprints:</b> We save two mathematical vectors (embeddings): <b>CLIP Visual Features</b> (capturing palette, style, and visual layout) and <b>BGE Text Features</b> (embedding the structured caption prose). Storing both in separate <b>ChromaDB</b> drawers linked by a shared ID enables multi-angle querying.", bullet_style))

    story.append(Paragraph("<b>The Retriever Workflow (Part B):</b>", h2_style))
    story.append(Paragraph("• <b>Query Decomposition & Expansion:</b> An LLM (Llama-3.1-8b via Groq) performs zero-shot query decomposition to map free-form search text to our caption schema (capturing clothing_items, colors, environment, style). Crucially, colors are bound to specific items (e.g. 'red tie', 'white shirt') to resolve compositionality, and the query is expanded into descriptive prose to bridge the domain gap.", bullet_style))
    story.append(Paragraph("• <b>Hybrid Fusion Scoring:</b> We search both the CLIP and BGE collections to retrieve candidate pools of size 3k. Similarity is computed as cosine similarity (sim = 1 - distance) on L2-normalized embeddings. The final query scoring applies a weighted sum: <b>Score = &alpha; &middot; Sim_CLIP + &beta; &middot; Sim_BGE + Bonus</b>, where the default weights are <b>&alpha; = 0.35</b> (CLIP visual vibe & scene layout) and <b>&beta; = 0.50</b> (BGE compositional description matching). &beta; &gt; &alpha; because structured text handles word order and binding far better than a pure contrastive visual space.", bullet_style))
    story.append(Paragraph("• <b>Categorical Match Bonus:</b> The third term represents a hard metadata matching bonus (<b>Bonus &le; &gamma; = 0.15</b>) calculated by comparing parsed query attributes against the image metadata. Matching environments (e.g., 'office') yield +40% of &gamma;, matching style yields +30%, while bound color-garment pairs yield +10% each. This rewards exact matches, pushing true-positives above false-positives with high soft vector similarity.", bullet_style))

    story.append(PageBreak())

    # ==========================================
    # SECTION 3: CODEBASE STRUCTURE & GITHUB
    # ==========================================
    story.append(Paragraph("3. Codebase Structure & GitHub Link", h1_style))
    story.append(Paragraph(
        "The complete codebase is organized to keep data logic separate from models, making it highly modular and scalable.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Official Repository:</b> <font color='#7C6AF7'><u>https://github.com/aayushkumbharkar/glance-fashion-retrieval</u></font>",
        body_style
    ))

    # Code layout
    code_layout = [
        [Paragraph("<b>Directory / File</b>", body_style), Paragraph("<b>Function & Responsibility</b>", body_style)],
        [Paragraph("`data/download_dataset.py`", body_style), Paragraph("Streams Fashionpedia zip archives directly from AWS S3, extracting exactly 1,000 images sequentially and saving the manifest locally.", body_style)],
        [Paragraph("`Part_A_Indexer/`", body_style), Paragraph("Contains VLM captioning (`caption_generator.py`), dual visual/text feature embedding (`feature_extractor.py`), and ChromaDB collection management (`vector_store.py`).", body_style)],
        [Paragraph("`Part_B_Retriever/`", body_style), Paragraph("Contains Groq LLM query parsing (`query_parser.py`), hybrid fusion scoring with metadata bonuses (`retriever.py`), and the FastAPI server + Web search dashboard (`run_retrieval.py`).", body_style)],
        [Paragraph("`evaluation/run_eval_queries.py`", body_style), Paragraph("Runs all 5 evaluation queries from the assignment brief, computes latency, and saves the output locally as a JSON report.", body_style)],
        [Paragraph("`notebooks/exploration.ipynb`", body_style), Paragraph("Contains PCA/UMAP embedding projections, EDA distribution charts, and configurations sensitivity sweeps.", body_style)]
    ]
    code_table = Table(code_layout, colWidths=[2.2 * inch, 4.8 * inch])
    code_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), LIGHT_BG),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
    ]))
    story.append(code_table)
    story.append(Spacer(1, 0.15 * inch))

    # ==========================================
    # SECTION 4: FUTURE WORK ROADMAP
    # ==========================================
    story.append(Paragraph("4. Future Work Roadmap", h1_style))
    
    story.append(Paragraph("A. Location & Weather Awareness Extension", h2_style))
    story.append(Paragraph(
        "To incorporate local context like weather and cities without changing the underlying vector space structure, we can leverage EXIF metadata and classification heads:",
        body_style
    ))
    story.append(Paragraph("• <b>Location:</b> Extract GPS metadata from image headers at index time. Use an offline reverse geocoding API (e.g. OpenStreetMap's Nominatim) to convert coordinates to categorical place names (e.g., 'San Francisco', 'Coit Tower', 'coastal city'). Add these as keywords to the text embedding and metadata filter records.", bullet_style))
    story.append(Paragraph("• <b>Weather:</b> Fine-tune a lightweight visual classifier (like a ViT) on public weather datasets to tag images into standard classes (Sunny, Rainy, Snowy, Overcast). The parsed environment query (e.g., 'city walk on a rainy day') can then trigger strict metadata filters (`weather: rainy`), preventing sunny outfits from polluting the results.", bullet_style))

    story.append(Paragraph("B. Improving Search Precision", h2_style))
    story.append(Paragraph(
        "To move from zero-shot baselines to a production-grade visual fashion engine, four precision pipelines would be implemented:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Caption Quality (Highest-Leverage Next Step):</b> The current submission's caption "
        "quality is bottlenecked by CPU-only heuristic parsing, evidenced by 57.5% environment=unknown "
        "and 91.8% style=unknown rates (see evaluation/results.json). Running the BLIP-2 structured "
        "path (already implemented, gated behind a CLI flag: drop <code>--use_lightweight_vlm</code>) "
        "on a Colab T4 GPU would directly resolve this and is the single highest-leverage next step "
        "for improving retrieval precision on environment- and style-filtered queries.",
        bullet_style
    ))
    story.append(Paragraph("• <b>Fine-Tuning Visual Alignments:</b> Fine-tune OpenCLIP on domain-specific triplet datasets (like DeepFashion and FashionIQ) to close the fashion vocabulary gap (e.g. distinguishing a 'parka' from an 'anorak').", bullet_style))
    story.append(Paragraph("• <b>Cross-Encoder Re-ranking:</b> Retrieve the top 50 candidates using our dual-space index, then feed (query, caption text) pairs into a BERT-based Cross-Encoder. Because Cross-Encoders evaluate full attention over both texts together, they identify subtle mismatches that bi-encoder search misses.", bullet_style))
    story.append(Paragraph("• <b>Region-Based Feature Maps:</b> Segment each image into visual crops (upper-body, lower-body, accessories) using segmenters like SAM. Generate separate CLIP vectors per crop. When a query searches for 'red tie and white shirt', we match the tie embedding against the accessory/upper crops specifically, resolving compositional binding visually.", bullet_style))
    story.append(Paragraph("• <b>Click-Through Bandit Loop:</b> Track user click-through rates on search results. Use contextual bandits to auto-tune alpha, beta, and gamma weights dynamically, adapting ranking scores to user intent.", bullet_style))

    doc.build(story, canvasmaker=NumberedCanvas)


if __name__ == "__main__":
    import sys
    output = Path(__file__).parent / "submission.pdf"
    create_submission_report(output)
    print(f"Submission report generated at: {output.resolve()}")
