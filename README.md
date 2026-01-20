# Video Understanding MCP Server

AWS 서비스 기반 비디오 이해 및 검색 MCP 서버입니다. 비디오를 분석하고, 자막을 생성하며, 특정 장면을 검색할 수 있습니다.

## 주요 기능

- 🎥 **비디오 분석**: 영상 내용 임베딩 및 요약 생성
- 🔍 **장면 검색**: 자연어로 특정 장면 찾기
- 📝 **자막 처리**: 자동 자막 생성 및 키워드 추출
- 🎯 **정확한 타임스탬프**: 원하는 장면의 정확한 재생 시점 제공

## 빠른 시작

### 1단계: 사전 준비

#### Python 설치 확인
```bash
python3 --version  # Python 3.8 이상 필요
```

#### AWS CLI 설치 및 설정
```bash
# AWS CLI 설치 (macOS)
brew install awscli

# AWS CLI 설치 (Linux)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

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
```bash
# 버킷 이름 설정 (원하는 이름으로 변경)
BUCKET_NAME="my-video-vectors-$(date +%s)"

# S3 Vectors 버킷 생성
aws s3vectors create-vector-bucket --bucket-name $BUCKET_NAME

# 벡터 인덱스 생성 (1024차원, cosine 유사도)
aws s3vectors create-index \
  --bucket-name $BUCKET_NAME \
  --index-name video-index \
  --vector-dimension 1024 \
  --distance-metric cosine

echo "✅ S3 Vectors 버킷 생성 완료: $BUCKET_NAME"
```

#### DynamoDB 테이블 생성
```bash
# 테이블 이름 설정
TABLE_NAME="video-processing-tasks"

# DynamoDB 테이블 생성
aws dynamodb create-table \
  --table-name $TABLE_NAME \
  --attribute-definitions AttributeName=task_id,AttributeType=S \
  --key-schema AttributeName=task_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST

echo "✅ DynamoDB 테이블 생성 완료: $TABLE_NAME"
```

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

```json
{
  "mcpServers": {
    "video-processing": {
      "command": "uv",
      "args": [
        "--directory",
        "/절대/경로/video-understanding-mcp", / 실제 경로로 변경필요 local 경로
        "run",
        "video-mcp-server"
      ],
      "env": {
        "AWS_REGION": "us-east-1",
        "AWS_PROFILE": "default",
        "S3_VECTORS_BUCKET": "위에서-생성한-버킷-이름",
        "S3_VECTORS_INDEX": "video-index",
        "DYNAMODB_TABLE": "video-processing-tasks"
      }
    }
  }
}
```

**중요**: `/절대/경로/video-understanding-mcp`를 실제 프로젝트 경로로 변경하세요.

### 5단계: 서버 실행 확인

```bash
# Kiro CLI 시작
kiro-cli 

## 사용 예시

### 비디오 분석하기


Kiro CLI에서:
```
영상 분석해줘: s3://my-bucket/videos/my-video.mp4
```

### 특정 장면 검색하기

```
골 장면 찾아줘
선수가 넘어지는 장면 찾아줘
심판이 카드를 꺼내는 순간 찾아줘
```

### 자막 생성 및 조회

```
자막 생성해줘: s3://my-bucket/videos/my-video.mp4
자막에서 "골" 키워드 찾아줘
```

### MCP 서버 연결 안 됨
- `~/.kiro/settings/mcp.json`의 경로가 절대 경로인지 확인
- Kiro CLI를 완전히 종료 후 재시작


