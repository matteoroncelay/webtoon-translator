"""Orchestration de tout le pipeline (remplace FINAL.py).

    images -> détection -> présence -> queues + attribution -> OCR
           -> classification -> contexte -> traduction -> rendu + HTML + rapport

Points clés :
  - chaque composant est INJECTÉ (détecteurs, OCR, LLM) : on peut tester le
    pipeline entier avec de faux composants, sans modèle ni clé API ;
  - l'état est sauvegardé dans `chapter.json` après chaque étape : si le
    programme s'arrête (quota, coupure...), `--resume` reprend là où il en était ;
  - le modèle de personnages est OPTIONNEL : sans lui, l'attribution est
    laissée au LLM (mode dégradé).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Config, Secrets
from .io import list_images, load_chapter, read_image, save_chapter, write_image
from .io.report import write_report
from .models import BubbleKind, Chapter, Page
from .reasoning import CharacterRegistry, attribute_page, compute_presence, load_characters
from .render import render_page, write_reader
from .text import (
    GeminiClient, OcrEngine, analyze_context, classify_tailless, read_page_text, translate_page,
)
from .vision import YoloDetector, YoloSegmenter, analyze_tail, detect_bubbles, detect_characters
from .vision.tail import Segmenter

STEPS = ("vision", "ocr", "classify", "context", "translate")


@dataclass
class Components:
    bubble_detector: YoloDetector
    character_detector: YoloDetector | None
    segmenter: Segmenter | None
    ocr: OcrEngine
    llm: GeminiClient
    registry: CharacterRegistry


def build_components(config: Config, secrets: Secrets) -> Components:
    """Crée les vrais composants à partir de la configuration."""
    from .text import GoogleVisionOcr

    det = config.detection
    char_model = config.paths.character_model
    if char_model.is_file() and config.paths.characters_file.is_file():
        character_detector = YoloDetector(char_model, det.imgsz, det.iou)
        registry = load_characters(config.paths.characters_file)
    else:
        print("[INFO] Pas de modèle ou de fiches personnages : attribution par le LLM seul.")
        character_detector, registry = None, CharacterRegistry({})

    seg_model = config.paths.bubble_seg_model
    return Components(
        bubble_detector=YoloDetector(config.paths.bubble_model, det.imgsz, det.iou),
        character_detector=character_detector,
        segmenter=YoloSegmenter(seg_model, config.tail) if seg_model.is_file() else None,
        ocr=GoogleVisionOcr(),
        llm=GeminiClient(config.gemini, secrets.gemini_api_key, secrets.gemini_api_key_fallback),
        registry=registry,
    )


def _step(chapter: Chapter, name: str, state_path: Path, action) -> None:
    """Exécute une étape si elle n'est pas déjà faite, puis sauvegarde."""
    if name in chapter.steps_done:
        print(f"[{name}] déjà fait, on passe.")
        return
    print(f"[{name}] ...")
    action()
    chapter.steps_done.append(name)
    save_chapter(chapter, state_path)


def run_pipeline(input_dir: str | Path, output_dir: str | Path, config: Config,
                 components: Components, resume: bool = False) -> Chapter:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    state_path = output_dir / "chapter.json"
    images = {p.name: p for p in list_images(input_dir)}
    if not images:
        raise ValueError(f"Aucune image dans {input_dir}")

    if resume and state_path.is_file():
        chapter = load_chapter(state_path)
        print(f"[reprise] étapes déjà faites : {chapter.steps_done or 'aucune'}")
    else:
        chapter = Chapter(name=input_dir.name)

    c = components

    def vision() -> None:
        chapter.pages = []
        for name, path in images.items():
            image = read_image(path)
            page = Page(name, width=image.shape[1], height=image.shape[0])
            page.bubbles = detect_bubbles(image, c.bubble_detector, config)
            page.sort_bubbles_reading_order()
            if c.character_detector is not None:
                page.characters = detect_characters(image, c.character_detector, config)
            chapter.pages.append(page)
            print(f"   {name} : {len(page.bubbles)} bulles, {len(page.characters)} personnages")
        compute_presence(chapter.pages, config.presence)
        for page in chapter.pages:
            image = read_image(images[page.file_name])
            tails = {b.id: analyze_tail(image, b.bbox, config.tail, c.segmenter)
                     for b in page.bubbles if b.kind is BubbleKind.WITH_TAIL}
            attribute_page(page, tails, c.registry, config.speaker)

    def ocr() -> None:
        for page in chapter.pages:
            count = read_page_text(read_image(images[page.file_name]), page, c.ocr)
            print(f"   {page.file_name} : {count} bulles lues")

    def classify() -> None:
        print(f"   {classify_tailless(chapter, c.llm, config.gemini)} bulles classées")

    def context() -> None:
        print(f"   {analyze_context(chapter, c.llm, c.registry, config.gemini)} bulles analysées")

    def translate() -> None:
        for page in chapter.pages:
            if all(b.text_fr for b in page.bubbles if b.text_ko):
                continue  # page déjà traduite (reprise)
            done = translate_page(chapter, page, c.llm, c.registry, config.gemini,
                                  read_image(images[page.file_name]))
            print(f"   {page.file_name} : {done} bulles traduites")
            save_chapter(chapter, state_path)  # sauvegarde après CHAQUE page

    for name, action in zip(STEPS, (vision, ocr, classify, context, translate)):
        _step(chapter, name, state_path, action)

    # Rendu (toujours refait : rapide, et permet de changer les réglages de rendu)
    rendered = []
    for page in chapter.pages:
        out = output_dir / "pages" / page.file_name
        write_image(out, render_page(read_image(images[page.file_name]), page.bubbles,
                                     config.paths.font, config.render))
        rendered.append(out)
    write_reader(rendered, output_dir / "lecture.html", chapter.name)
    write_report(chapter, output_dir / "rapport.csv")
    print(f"\nTerminé : {output_dir / 'lecture.html'}")
    return chapter
