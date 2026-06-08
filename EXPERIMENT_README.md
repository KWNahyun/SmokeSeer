# SmokeSeer 경량화 실험 가이드

SmokeSeer(RGB+Thermal 기반 3DGS 연기 제거) 베이스라인 및 경량화 방법론 비교 실험 환경 설정 가이드입니다.

---

## 환경 요구사항

- GPU: RTX 3090 이상 (VRAM 24GB 권장)
- CUDA 12.1
- Python 3.8
- PyTorch 2.4.1+cu121

---

## 1. 환경 설정

### conda 환경 생성

```bash
conda create -n gaussian_splatting python=3.8 -y
conda activate gaussian_splatting
```

### PyTorch 설치

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

### 기본 의존성 설치

```bash
pip install gsplat==1.5.3
pip install plyfile tqdm imageio imageio-ffmpeg opencv-python scipy lpips
pip install tinycudann --index-url https://raw.githubusercontent.com/NVlabs/tiny-cuda-nn/master/bindings/torch/
pip install vector-quantize-pytorch
```

### submodule 빌드

```bash
git submodule update --init --recursive

# 기본 래스터라이저
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn

# Mini-Splatting용
pip install -e submodules/diff-gaussian-rasterization_ms

# LightGaussian용 (별도 경로에 설치)
git clone https://github.com/VITA-Group/LightGaussian.git ~/LightGaussian --depth=1
pip install -e ~/LightGaussian/submodules/compress-diff-gaussian-rasterization
```

---

## 2. 데이터셋 준비

SmokeSeer 공식 데이터셋을 `data/` 폴더에 위치시킵니다.

```
SmokeSeer/
└── data/
    └── real/
        └── red_full/          ← 전체 데이터셋
            ├── images/        ← RGB 이미지
            ├── images_thermal/ ← 열화상 이미지
            ├── images_depth/  ← depth 맵
            └── sparse/        ← COLMAP 포즈
```

---

## 3. 브랜치 구조

| 브랜치 | 내용 |
|--------|------|
| `main` | 베이스라인 (gsplat 통합, wandb optional) |
| `mini-splatting` | Mini-Splatting 통합 (depth reinitialization + importance pruning) |
| `light-gaussian` | LightGaussian 통합 (post-training importance pruning + finetune) |
| `compact-3dgs` | Compact-3DGS 통합 (learnable mask regularization) |

```bash
git fetch --all
git branch -a  # 브랜치 목록 확인
```

---

## 4. 실험 실행

### 단일 실험 (수동)

```bash
# Stage 1: Thermal-only surface reconstruction
bash stage1.sh ~/SmokeSeer/data/real/red_full

# Stage 2: RGB+Thermal smoke decomposition (Stage 1 output 자동 감지)
bash stage2.sh ~/SmokeSeer/data/real/red_full
```

### 전체 경량화 실험 자동 실행

4가지 실험(baseline, Mini-Splatting, LightGaussian, Compact-3DGS)을 순서대로 자동 실행합니다.

```bash
bash run_all_experiments.sh ~/SmokeSeer/data/real/red_full images_depth
```

실험이 완료되면 터미널에 전체 비교표가 출력됩니다:

```
실험                   Gaussians     PSNR     SSIM    LPIPS      FPS
--------------------------------------------------------------------
baseline             2,688,480    13.62   0.4048   0.5237     59.9
mini_splatting         250,000    13.87   0.4298   0.5480     59.2
light_gaussian         210,000    14.47   0.4247   0.5298    122.2
compact_3dgs           258,000    13.86   0.4284   0.5471     73.7
```

---

## 5. Metrics 측정

```bash
# 품질 + 속도 측정
python -W ignore metrics.py -m ./output/red_full/baseline

# 속도만 측정
python -W ignore metrics.py -m ./output/red_full/baseline --skip_quality

# 여러 실험 한꺼번에
python -W ignore metrics.py -m \
  ./output/red_full/baseline \
  ./output/red_full/mini_splatting \
  ./output/red_full/light_gaussian \
  ./output/red_full/compact_3dgs
```

결과는 각 실험 폴더의 `metrics.json`에 저장됩니다.

---

## 6. 결과 폴더 구조

```
output/red_full/
├── baseline/
│   ├── test_results_individual/   ← 렌더링 결과 이미지
│   ├── ft_chkpnt_surface_thermal30000.pth
│   ├── ft_chkpnt_smoke_thermal30000.pth
│   └── metrics.json               ← PSNR/SSIM/LPIPS/FPS
├── mini_splatting/
├── light_gaussian/
└── compact_3dgs/
```

---

## 7. 주요 파라미터

### Mini-Splatting

```bash
bash stage1.sh ~/SmokeSeer/data/real/red_full --num_max 1500000 --num_depth 300000
```

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--num_max` | 1500000 | densification 중 최대 Gaussian 수 |
| `--num_depth` | 300000 | depth reinitialization 샘플 수 |

### LightGaussian

```bash
python -W ignore prune_smokeseer.py -m ./output/red_full/baseline --prune_percent 0.6
```

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--prune_percent` | 0.6 | 제거할 Gaussian 비율 (0~1) |
| `--iteration` | 30000 | 로드할 checkpoint iteration |

### Compact-3DGS

`arguments/__init__.py`에서 조정:

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `lambda_mask` | 0.01 | mask regularization loss weight |
| `mask_prune_iter` | 1000 | mask pruning 주기 (iteration) |

---

## 8. 참고 논문

- **SmokeSeer** (베이스라인): arXiv:2509.17329
- **Mini-Splatting**: ECCV 2024
- **LightGaussian**: NeurIPS 2024 Spotlight
- **Compact-3DGS**: CVPR 2024