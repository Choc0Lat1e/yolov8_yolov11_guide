# YOLOv11 시작하기 (초보자 안내서)

안녕하세요! 이 파일은 "YOLOv11"을 GitHub에 정리하기 위한 초안 README.md입니다. 초보자도 이해하기 쉬운 설명, 설치/사용 예시(주로 YOLOv8 기준으로 실제 명령어 제시), YOLOv8과 YOLOv11의 비교 테이블(확인된 항목은 사실 기반, YOLOv11 항목은 공개 정보가 없을 경우 `예상`으로 표기)과, YOLO에서 자주 쓰이는 용어 정리 테이블을 포함합니다.

---

## 목차
- 소개
- 빠른 시작 (YOLOv8 실제 사용 예시)
- 데이터셋 포맷 (YOLO 포맷)
- 학습 & 추론 기본 예시 (YOLOv8 기준)
- YOLOv8 vs YOLOv11 비교 테이블
- YOLO 용어 정리 테이블
- 리포지토리 예시 구조
- 초보자 팁 / 자주 묻는 질문(FAQ)
- 참고 자료

---

## 소개
YOLO(You Only Look Once)는 실시간 객체 탐지(object detection) 모델 계열입니다. Ultralytics의 YOLOv8이 널리 사용되고 있으며, "YOLOv11"은 사용자가 만들고자 하는 프로젝트 이름일 수도 있고(또는 미래 버전 가정) 실제 공식 버전이 아직 공개되지 않았을 수 있습니다. 아래의 비교표에서는 YOLOv8의 확실한 특성과, YOLOv11에 대해 공개 정보가 충분하지 않을 때는 `예상`으로 표기했습니다. 실제 공식 문서는 항상 프로젝트의 원저장소를 참고하세요.

---

## 빠른 시작 (YOLOv8 실제 예시)
환경: Python 3.8+ 추천, 가속을 위해 CUDA(그래픽 카드) 설치 권장

1) ultralytics 설치 (YOLOv8)
```bash
pip install ultralytics
```

2) 간단한 추론
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')          # 가벼운 모델 예시
results = model.predict('image.jpg') # 이미지 추론
results.show()
```

3) 학습 예시 (COCO 형식 또는 YOLO 라벨 형식)
```bash
yolo task=detect mode=train model=yolov8n.pt data=dataset.yaml epochs=50 imgsz=640
```

(위 명령은 ultralytics CLI 스타일의 예시입니다. 자세한 옵션은 ultralytics 문서를 참고하세요.)

---

## 데이터셋 포맷 (YOLO 텍스트 라벨)
- 이미지: JPEG/PNG
- 라벨: 동일한 파일명에 .txt
- 각 라인: <class_id> <x_center> <y_center> <width> <height>
  - 좌표/크기는 이미지 가로/세로에 대한 비율(0~1)
- 예: `0 0.492 0.421 0.234 0.332`

데이터셋을 COCO 형식으로 준비하거나, YOLO 형식으로 직접 준비할 수 있습니다. 초보자는 기존 공개 데이터셋(예: COCO)에서 연습해보는 걸 추천합니다.

---

## 학습 팁 (초보자용)
- 작은 모델(yolov8n, yolov8s)으로 먼저 빠르게 실험하세요.
- 학습 전 데이터 품질 점검: 라벨 오류, 잘못된 박스 등.
- batch size, imgsz(이미지 사이즈), learning rate를 조절하세요.
- GPU가 없다면 epochs를 줄이고 이미지 크기를 낮추세요.
- 전이학습(pretrained weights)을 사용하면 수렴이 빠릅니다.
- 학습 로그와 시각화(예: TensorBoard, Weights & Biases) 활용 권장.

---

## YOLOv8 vs YOLOv11 비교 테이블
주의: YOLOv8은 Ultralytics에서 공식 발표된 버전입니다. "YOLOv11"은 사용자의 프로젝트명일 수도 있고, 공개된 공식 사양이 없을 경우 본 표의 YOLOv11 항목은 '예상' 또는 '설계 목표' 형태로 표기합니다. 공식 정보가 있다면 그에 맞춰 내용을 업데이트하세요.

| 항목 | YOLOv8 (확인된 특징) | YOLOv11 (예상 / 사용자 정의) | 비고 |
|---|---:|---|---|
| 발표/출처 | Ultralytics (공식) | (공식 미확인 — 프로젝트별 다름) | YOLOv11이 공식 버전인지 확인 필요 |
| 모델 패밀리 | nano, small, medium, large, x (yolov8n/s/m/l/x) | 예상: 유사한 스케일 또는 더 세분화된 모델 | 모델 네이밍은 프로젝트마다 다를 수 있음 |
| 아키텍처 변화 | CSP/MPConv 등 개선, PyTorch 중심 구현 | 예상: 효율 개선(더 나은 백본/헤드), 모듈화 강화 | 구체적 변경은 공개 자료 필요 |
| 추론 속도 | 실시간 목적 최적화 (특히 nano/small) | 예상: 추가 최적화, 하드웨어 특화 가속 포함 가능 | 하드웨어별 최적화 중요 |
| 정확도(mAP) | 기존 SOTA 대비 좋은 균형(버전에 따라 다름) | 예상: 개선 목표 (신규 손실/데이터 증강으로 향상 가능) | 수치 비교는 동일 데이터셋에 대해 측정 필요 |
| 학습 편의성 | ultralytics 라이브러리로 간편 CLI/파이썬 API 제공 | 사용자 프로젝트에 따라 다름 (도구 제공 여부) | 초보자에겐 공식 툴 지원이 큰 장점 |
| 하드웨어 최적화 | ONNX, TensorRT 등 변환 지원 | 예상: 더 많은 변환/엔진 지원 또는 플러그인 | 배포 대상에 따라 확인 필요 |
| 손실 함수 | Box loss, classification loss, objectness loss 등 | 예상: 새로운 손실 함수/가중치 스케줄 가능 | 논문/릴리스 노트 확인 필요 |
| 앵커 | anchor-free 옵션 및 anchor 사용 모델 혼재 | 예상: anchor-free 추세 유지 또는 개선 | 설정은 모델 및 구현체에 따라 다름 |
| 데이터 증강 | Mosaic, MixUp 등 지원 | 예상: 더 강력한 자동화된 증강 전략 포함 가능 | 실험으로 검증 필요 |
| 라이선스 | Ultralytics 라이선스(공식 repo 참고) | 프로젝트별 라이선스 상이 | 배포/상업 이용 전 확인 필요 |

원하시면 이 표를 바탕으로 실제 YOLOv11(혹은 본인 프로젝트)의 기능 목록을 채워드릴게요. 공식 문서나 릴리즈 노트가 있다면 그 링크를 알려주세요.

---

## YOLO에서 자주 쓰이는 용어 정리 테이블
아래 표는 초보자가 자주 만나게 되는 기본 용어들을 정리한 것입니다.

| 용어 | 한글 설명 | 간단 예 / 메모 |
|---|---|---|
| Bounding Box (Bbox) | 객체를 둘러싼 사각형 | 좌표로 위치와 크기를 표현 |
| Anchor | 사전 정의된 박스 형태(크기/비율) | anchor 기반 모델에서 사용 |
| Anchor-free | 앵커 없이 중심점/격자 기반으로 박스 예측 | 더 간단한 설정 장점 |
| IoU (Intersection over Union) | 예측 박스와 정답 박스의 겹침 비율 | 성능 지표로 사용 |
| mAP (mean Average Precision) | 검출 성능 종합 지표 | COCO mAP@0.5:0.95 등 |
| Precision / Recall | 정밀도 / 재현율 | 검출 품질 평가 기본 지표 |
| NMS (Non-Maximum Suppression) | 중복 박스 제거 알고리즘 | IoU 임계값으로 중복 제거 |
| CIoU / DIoU / GIoU | 개선된 박스 손실/평가 지표 | IoU 한계를 보완 |
| Backbone | 특징(feature)을 추출하는 네트워크 | 예: CSP, Darknet, EfficientNet 등 |
| Neck | backbone과 head를 연결하는 부분 | FPN, PANet 등 |
| Head | 최종 예측을 수행하는 부분 | 클래스/박스/객체성 예측 |
| Epoch | 전체 데이터셋을 한 번 학습 | 보통 여러 epoch 진행 |
| Batch size | 한 번에 처리하는 샘플 수 | GPU 메모리와 트레이드오프 |
| LR (Learning Rate) | 학습률 | scheduler로 변동 가능 |
| Transfer Learning | 사전학습된 가중치로 학습 시작 | 작은 데이터셋에 유리 |
| Augmentation | 데이터 증강 | 회전, 크롭, 색상 변환, Mosaic 등 |
| FP16 / AMP | 반정밀도 연산(속도/메모리 이득) | Automatic Mixed Precision |
| Pruning | 모델 경량화(가중치 제거) | 배포용으로 사용 |
| Quantization | 양자화 (예: INT8) | 속도와 메모리 개선 목적 |
| ONNX / TensorRT | 모델 변환/최적화 도구 | 배포 가속에 사용 |
| Dataset.yaml | 데이터셋 구성 파일 | train/val 경로, 클래스 목록 등 |

---

## 리포지토리 예시 구조
아래는 GitHub에 올릴 때의 간단한 예시 구조입니다.

- yolov11/ (루트)
  - README.md
  - LICENSE
  - requirements.txt
  - data/
    - dataset.yaml
    - images/
    - labels/
  - configs/
    - yolov11_config.yaml
  - scripts/
    - train.sh
    - infer.sh
  - notebooks/
    - demo.ipynb
  - weights/
    - yolov11_custom.pt
  - docs/
    - usage.md

requirements.txt에는 ultralytics, torch, torchvision 등 주요 패키지를 명시하세요.

---

## 초보자용 간단 체크리스트
- [ ] Python, pip, (CUDA) 환경 확인
- [ ] 데이터셋 포맷 확인 (YOLO txt / COCO)
- [ ] 작은 모델로 먼저 실험
- [ ] 학습 로그/시각화 도구 설정 (TensorBoard 등)
- [ ] 결과(시각화)로 문제점 확인 — 라벨 오류, 오버피팅 등

---

## FAQ (간단)
Q. YOLOv11이 공식적인 버전인가요?  
A. 현재(이 README 작성 시점) 공식 버전으로 발표된지 확인되지 않았다면, 프로젝트명으로서의 YOLOv11을 사용하거나 공식 릴리즈를 기다려야 합니다. 공식 정보는 원저장소의 릴리즈 노트를 확인하세요.

Q. 모델을 경량화하려면?  
A. pruning, quantization, knowledge distillation, 모델 아키텍처 축소 등이 있습니다. FP16 연산으로도 속도/메모리 개선이 가능합니다.

---

## 다음 단계 (권장)
1. 이 README를 GitHub에 올리기 전에 프로젝트의 목표(연구용 / 제품용 / 학습용)를 상단에 명확히 적으세요.  
2. 본인이 만들고자 하는 YOLOv11의 특징(예: 경량화, 특정 하드웨어 최적화, 추가된 손실 등)을 정리해 주시면, README에 기능 설명과 사용 예시(코드/명령)를 더 구체적으로 추가해 드리겠습니다.  
3. 실제 코드/설정 파일(config)을 제공해 주시면, 학습/배포 스크립트도 함께 만들어 드립니다.

---

작성자 노트: 이 README는 초보자 관점에서 실무에 바로 사용할 수 있도록 기본 구조와 설명, 비교표, 용어정을 포함한 초안입니다. YOLOv11 관련 구체적 기술 스펙(공식 릴리스 노트 또는 논문)이 있다면 그 자료를 주시면 비교표와 설치/사용법을 정확하게 업데이트하겠습니다.
