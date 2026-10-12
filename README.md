# DOPAMING-TRACKING-VIDEO

아이스하키 중계 영상에서 특정 선수가 아이스 위에 나온 구간을 뽑는 트래커입니다.

대상은 빨강 어깨 + 흰 몸통 유니폼의 **EAN 26번**(루비덕스)입니다. 같은 번호라도 검정+흰색 유니폼(아이언 이글스)은 제외합니다. 방송 줌 때문에 키, 픽셀 높이, 링크 위치는 신원 조건으로 쓰지 않습니다.

## 준비

- Python 3.11 이상 (`py -3`)
- NVIDIA GPU와 CUDA PyTorch. 이 저장소의 시간 측정은 RTX 3060 12GB 기준입니다.
- 포즈 가중치 `yolo26m-pose.pt`는 첫 실행 때 Ultralytics가 받습니다. 없으면 `yolo26s-pose.pt`, `yolo11m-pose.pt` 순으로 시도합니다.

```text
py -3 -m pip install -r requirements.txt
```

PARSeq와 SAM2는 선택입니다. 없어도 EasyOCR만으로 동작하며, 아래 실행 예시는 둘 다 끕니다.

## 실행

인자 없이 돌리면 영상 처음부터 끝까지 분석합니다. `hockey_game.mp4`(1920×1080, 59.94fps, 약 57분 38초)는 이 설정에서 약 37분 걸립니다.

```text
py -3 track_player.py --video hockey_game.mp4 --number 26 --no-parseq --no-sam2
```

구간만 보려면 초 단위로 자릅니다. 30:00–40:00은 아래와 같습니다.

```text
py -3 track_player.py --video hockey_game.mp4 --number 26 --start 1800 --end 2400 --no-parseq --no-sam2
```

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--video` | `hockey_game.mp4` | 분석할 영상 |
| `--number` | `26` | 두 자리 등번호 |
| `--start` | `0` | 시작 시각(초) |
| `--end` | 영상 끝 | 끝 시각(초). 비우면 영상 길이로 맞춥니다 |
| `--sample-fps` | `5` | 검출·추적 샘플 주기. OCR은 0.45초마다 |
| `--imgsz` | `1280` | YOLO 입력 크기 |
| `--team-bgr` | 없음 | 목표 유니폼색 `B,G,R`. 예: `230,230,230` |
| `--no-parseq` | 끔 | PARSeq를 쓰지 않음 |
| `--no-sam2` | 끔 | SAM2 마스크 정제를 끔 |
| `--no-reid` | 끔 | BoT-SORT ReID를 끔 |
| `--no-oblique` | 끔 | 비스듬한 등을 OCR 후보에서 뺌 |
| `--save-crops` | 끔 | OCR에 넣은 등판 crop 저장 |

이름은 코드의 `TARGET_NAME`(`EAN`)입니다. 중단하면(`Ctrl+C`) 그때까지 확정된 구간을 저장합니다.

## 출력

영상과 같은 폴더에 JSON과 CSV가 생깁니다.

- `hockey_game_player_26_timeline.json`
- `hockey_game_player_26_timeline.csv`

1초보다 짧은 구간은 버리고, 관측 사이 공백이 2초 이하면 한 구간으로 합칩니다.

30:00–40:00을 이 설정으로 돌린 결과는 저장소의 타임라인 파일과 같습니다. 합계 53.8초입니다.

| 시작 | 끝 | 길이 |
| --- | --- | --- |
| 34:30 | 34:43 | 13.4초 |
| 34:45 | 34:49 | 3.6초 |
| 38:49 | 39:15 | 25.6초 |
| 39:16 | 39:21 | 4.4초 |
| 39:53 | 40:00 | 6.8초 |

육안으로 확인한 34:32–34:40은 첫 구간에 포함됩니다. 34:58–35:00은 등판이 약 35px라 번호가 읽히지 않아 빠져 있습니다. 38:42–40:00은 위 세 구간으로 일부만 덮입니다.

## 동작

1. YOLO26-pose로 선수 박스와 관절을 찾고, BoT-SORT(ReID + 카메라 움직임 보정)로 ID를 유지합니다. 추적 설정은 `botsort_hockey.yaml`입니다.
2. 스코어보드 영역과 너무 작은 박스는 버립니다. 샘플 사이 프레임은 `grab()`만 하고, 처리 프레임만 `read()`합니다.
3. 어깨–엉덩이 관절로 등판을 자릅니다. 작은 등판은 컬러 상태로 키웁니다. 높이 70px 미만은 12배, 110px 미만은 8배입니다.
4. 빨강+흰색이고 이름 `EAN` 또는 등번호 `26`(점수 0.35부터) 중 하나면 확정합니다.
5. 이름과 번호가 2초 안에 같이 보이면 그 등판의 외형을 기준으로 저장합니다. 이후 번호가 안 읽혀도, 위치와 외형이 이어지는 동안 짧게 따라갑니다.
6. 원거리에서 번호가 읽힌 등판(높이 120px 미만, 점수 0.50 이상)은 넓은 화면 기준으로 남기고, 비슷한 외형의 트랙을 나중에 같은 선수로 붙입니다.

팔뚝 번호 OCR은 30:00–40:00에서 한 번도 26을 내지 못해 기본으로 꺼 두었습니다. 호출 간격만 유지합니다.

## 파일

| 파일 | 역할 |
| --- | --- |
| `track_player.py` | 검출, OCR, 구간 병합 |
| `requirements.txt` | Python 의존성 |
| `botsort_hockey.yaml` | BoT-SORT 설정. 실행 시 다시 씁니다 |
| `hockey_game_player_26_timeline.json` | 30:00–40:00 샘플 결과 |
| `hockey_game_player_26_timeline.csv` | 같은 결과의 CSV |

영상(`*.mp4`)과 가중치(`*.pt`, `*.onnx`, `*.engine`)는 `.gitignore`에 있습니다.
