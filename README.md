# HAMpreprocessing

HAM10000 데이터셋에 여러 영상처리 알고리즘 적용후 학습, 테스트

데이터셋은 용량문제로 업로드 불가.. 돌리려면 train 폴더에 HAM10000 7클래스를 폴더별로 저장해야함 + fine tunning을 위한 weight폴더 및 Segmentation 학습 가중치 파일 필요

main.py : 학습,테스트 코드

edgeEnhance.py : HAM10000 데이터셋에 edgeEnhancement sharpening 적용,

contrast.py : apply contrast enhance to HAM10000 dataset
