# 滇境生灵 · AI 物种识别与公民科学科普台

Topic 1 of the Yunnan University · TRAE Campus AI Creative Practice Challenge, S2 "AI × Ecology" track.

Upload a photo → the model predicts the species and confidence → the app links the species profile
(protection level, habitat, identification points, "why it matters") → it renders a shareable
science card and stores the observation in a citizen-science database.

## Quick start

```powershell
cd D:\competation\q1_dianjing
$env:PYTHONIOENCODING = "utf-8"

# 1. fine-tune backbones (CPU; the EfficientNet pair below takes ~1 h in total)
python src\train.py --model efficientnet_b0 --size 256 --epochs 28
python src\train.py --model efficientnet_b0 --size 288 --epochs 10 `
    --init models\efficientnet_b0_256.pt --lr-head 5e-4 --lr-backbone 5e-5 --warmup 1

# 2. combine the best checkpoint per backbone
python src\build_ensemble.py

# 3. tune temperature + low-confidence threshold on validation, then evaluate
python src\evaluate.py --sweep --per-member

# 4. end-to-end check (throwaway database)
python src\smoke_test.py

# 5. launch the app
python src\app.py     # http://127.0.0.1:7860
```

The data pack is discovered automatically under `D:\competation` (see `src/config.py`), so the
awkward double-nested `“AI x 生态”数据包` path never needs to be typed.

## Dataset

| Item | Value |
|---|---|
| Species | 33 (9 birds, 8 mammals, 8 insects, 8 plants) |
| Images | 326 train / 132 test (4 test images per species) |
| Hard cases | 47 images covering the 9 look-alike species (P1–P5) |
| Protection levels | 国家一级 5 · 国家二级 13 · 三有 5 · 未列入 10 |

## How it maps to the acceptance anchors

| Anchor | Implementation |
|---|---|
| Test TOP-1 ≥ 70% | `src/evaluate.py` reports TOP-1/TOP-3 on the official 132-image test split |
| Protection labels match authoritative lists | Taken verbatim from `species_profiles.csv` (D3-verified) and displayed with the list version; 33/33 profiles carry the version string |
| No forced decision under low confidence | Temperature-scaled ensemble; below `threshold` the UI shows top-3 candidates plus `similar_species_tip` and marks the card 待鉴定 |
| Cards contain no claims beyond the profiles | Card fields are restricted to `name_cn`, `scientific_name`, `protection_level`, `why_important`, `identification_points`, `observe_etiquette` |
| No sensitive endangered-species images | All images are public iNaturalist CC photos; precise localities are absent from the pack |
| Hard-case robustness reported separately | `src/evaluate.py` scores the 47-image hard subset and writes per-class accuracy + confusion matrix |
| Similar-species discrimination | Each profile's `pair_id` links the two look-alike species (P1–P5); the UI always shows the partner's distinguishing tip, and low-confidence results list the top-3 candidates with their tips |
| Standard exportable dataset | `src/records.py` exports UTF-8-BOM CSV with confidence, uncertain flag, timestamps and QC columns (`qc_image`, `qc_confidence_band`, `qc_status`, `qc_note`) |
| "Current observation brief" | 本期简报 tab: record count, species count, protected-species count, star species, frequency bars |

## Design notes

- **Fixed split.** Train/validation come only from `split == train` in `clean_image_manifest.csv`
  (stratified 80/20, seed 42). The official test split is touched exactly once, for the final report.
- **Fine-grained setting.** 33 classes with ~8 training images each, so the recipe is full
  fine-tuning of an ImageNet backbone with differential learning rates (head 1e-3, backbone 8e-5),
  3-epoch warmup, cosine decay, RandAugment + RandomResizedCrop + random erasing, label smoothing 0.1.
- **Ensemble + TTA.** Softmax averaged across backbones and across the identity/flip views.
- **Calibration.** Temperature is fitted on the validation split by NLL; the threshold is the highest
  cut-off that keeps accepted predictions at ≥95% precision (minimum 50% coverage), so the app can
  honestly say "not sure" instead of guessing.
- **Cards.** 1080×1350 PNG rendered with Pillow + Microsoft YaHei, with protection badge,
  observe-without-disturbing reminder, list version and the photo's CC attribution.

## Compliance

- Protection levels follow 《国家重点保护野生动物名录》(2021年第3号), 《“三有”陆生野生动物名录》(2023年第17号)
  and 《国家重点保护野生植物名录》(2021年第15号); the version string shown in the UI comes from the data pack.
- Every card carries the CC attribution of the source photo. Uploads default to “图片由观察者提供”.
- Locations are free text at campus/park granularity — no coordinates are stored.
- Content shown in the app is limited to the supplied species profiles; the footer states that the
  output is for science outreach, not a formal identification.

## Known data gaps

The distributed pack lists two ground-truth files in `manifest.csv` that are **not present** on disk:

- `03_评测真值/truth_labels.csv` (132 test labels)
- `03_评测真值/hard_cases/hard_cases_labels.csv` (47 hard-case labels + 4 review-only images)

Test labels are recoverable from `clean_image_manifest.csv` (`split` + `species_code`), and hard-case
labels from the folder names (`hard_cases/images/<CODE>/`), which is what this project uses. The
`uncertain_excluded_from_scoring` flag for the 4 review-only images is **not** recoverable, so the
hard-subset number here scores all 47 images and may differ slightly from the official rubric.

## Layout

```
q1_dianjing/
├─ src/
│  ├─ config.py          data-pack discovery, paths, constants
│  ├─ data.py            manifest loading, stratified split, transforms, hard cases
│  ├─ train.py           fine-tuning loop (any torchvision backbone in BACKBONES)
│  ├─ build_ensemble.py  picks the best checkpoint per backbone -> models/ensemble.json
│  ├─ infer.py           ensemble + flip TTA + temperature scaling
│  ├─ evaluate.py        test/hard metrics, calibration, confusion matrices, report.md
│  ├─ cards.py           Pillow science-card renderer
│  ├─ records.py         SQLite observation store + CSV export
│  ├─ app.py             Gradio UI (识别 / 物种图鉴 / 观察记录 / 本期简报)
│  └─ fetch_weights.py   resumable torchvision weight downloader
├─ models/               checkpoints + ensemble.json
├─ data/                 observations.db + stored photos
└─ output/               metrics.json, report.md, predictions_*.csv, cards/
```
