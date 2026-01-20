# Video Understanding MCP Server

AWS 서비스 기반 비디오 이해 및 검색 MCP 서버입니다. 비디오를 분석하고, 자막을 생성하며, 특정 장면을 검색할 수 있습니다.

## 주요 기능

- 🎥 **비디오 분석**: 영상 내용 임베딩 및 요약 생성
- 🔍 **장면 검색 및 트랜스코딩**: 자연어로 특정 장면 찾기 및 파일 트랜스코딩
- 📝 **자막 처리**: 자동 자막 생성 및 키워드 추출
- 🎯 **정확한 타임스탬프**: 원하는 장면의 정확한 재생 시점 제공

## 빠른 시작

### 1단계: 사전 준비

#### Python 설치 확인
```bash
python3 --version  # Python 3.8 이상 필요
```

# AWS 자격증명 설정
aws configure
# AWS Access Key ID: <your-access-key>
# AWS Secret Access Key: <your-secret-key>
# Default region name: us-east-1
# Default output format: json
```

### 2단계: AWS 리소스 생성

아래 명령어를 순서대로 실행하여 필요한 AWS 리소스를 생성합니다.

#### S3 Vectors 버킷 및 인덱스 생성

# S3 Vectors 버킷 생성
aws s3vectors create-vector-bucket --bucket-name 나의버킷

#### 벡터 인덱스 생성 
aws s3vectors create-index \
  --bucket-name 나의버킷 \
  --index-name 나의인덱스 \
  --vector-dimension 1024 \
  --distance-metric cosine


#### DynamoDB 테이블 생성

aws dynamodb create-table \
  --table-name 나의테이블 \
  --attribute-definitions AttributeName=task_id,AttributeType=S \
  --key-schema AttributeName=task_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST

#### S3 버킷 생성
영상 업로드를 위한 S3버킷 생성

### 3단계: 서버 설치

```bash
# 저장소 클론
git clone https://github.com/Taehooon6476/video-understanding-mcp.git
cd video-understanding-mcp

# 의존성 설치
pip install -e .
```


### 4단계: Kiro CLI 연동

`~/.kiro/settings/mcp.json` 파일을 생성하거나 수정합니다:

```
   "video-processing": {
      "command": "video-mcp-server",
      "env": {
        "AWS_REGION": "us-east-1", # 특정 모델 사용을 위해 지정
        "AWS_PROFILE": "나의프로필",
        "S3_VECTORS_BUCKET": "twl-marengo-3",
        "S3_VECTORS_INDEX": "나의인덱스",
        "DYNAMODB_TABLE": "나의테이블",
        "S3_UPLOAD_BUCKET": "로컬 영상을 업로드할 버킷"
      },
   }
```



### 5단계: 서버 실행 확인 (Kiro예시)

```bash
# Kiro CLI 예시
kiro-cli 

## 사용 예시

### 비디오 분석하기


Kiro CLI에서:
```
디렉토리에 해당하는 영상 임베딩해줘.

```
골 장면 찾아줘.
타임스탬프 제공해줘.
주요 키워드 파일로 저장해줘.
```


