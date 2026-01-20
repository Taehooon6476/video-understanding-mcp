# Video Understanding MCP Server (Standard)

표준 MCP 프로토콜을 따르는 AWS 기반 비디오 이해 및 검색 서버입니다.

## 주요 차이점 (vs Hybrid 버전)

| 특징 | Standard (이 버전) | Hybrid |
|------|-------------------|--------|
| **구조** | 표준 MCP | MCP + Strands Agents |
| **도구 수** | 7개 (명확한 파라미터) | 4개 (추상화된) |
| **AI 호출** | 1번 (클라이언트만) | 2번 (클라이언트 + 서버) |
| **비용** | 저렴 | 2배 |
| **응답 속도** | 빠름 | 느림 |
| **Smithery 등록** | 권장 | 가능하지만 비표준 |

## 주요 기능

### 7개의 표준 MCP 도구

1. **create_video_embedding** - 비디오 임베딩 생성
2. **search_video_clips** - 자연어 장면 검색
3. **get_clip_playback_url** - 재생 URL 생성
4. **summarize_video** - 비디오 요약
5. **get_transcript** - 자막 생성/조회
6. **get_keywords** - 키워드 추출
7. **transcode_clip** - 클립 추출 및 저장

## 설치

```bash
cd video-understanding-mcp-standard
pip install -e .
```

## Kiro CLI 설정

`~/.kiro/settings/mcp.json`:

```json
{
  "video-processing-standard": {
    "command": "video-mcp-standard",
    "env": {
      "AWS_REGION": "us-east-1",
      "S3_VECTORS_BUCKET": "my-video-vectors",
      "S3_VECTORS_INDEX": "video-index",
      "DYNAMODB_TABLE": "video-tasks",
      "S3_UPLOAD_BUCKET": "my-video-uploads"
    }
  }
}
```

## AWS 리소스 생성

```bash
# S3 Vectors
aws s3vectors create-vector-bucket --bucket-name my-video-vectors
aws s3vectors create-index \
  --bucket-name my-video-vectors \
  --index-name video-index \
  --vector-dimension 1024 \
  --distance-metric cosine

# DynamoDB
aws dynamodb create-table \
  --table-name video-tasks \
  --attribute-definitions AttributeName=task_id,AttributeType=S \
  --key-schema AttributeName=task_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST

# S3 버킷
aws s3 mb s3://my-video-uploads
```

## 사용 예시

### Kiro CLI에서

```
사용자: "~/Videos/soccer.mp4 임베딩 생성해줘"
AI: create_video_embedding 도구 호출...
    완료: 150개 클립 저장

사용자: "골 장면 찾아줘"
AI: search_video_clips 도구 호출...
    3개 발견:
    - 2:34-2:40 (유사도: 0.89)
    - 15:22-15:28 (유사도: 0.85)

사용자: "첫 번째 클립 추출해줘"
AI: transcode_clip 도구 호출...
    soccer_clip_154_160.mp4 저장 완료
```

## 기술 스택

- **MCP**: 표준 Model Context Protocol
- **AWS Bedrock**: Marengo (임베딩), Pegasus (요약), Claude (키워드)
- **S3 Vectors**: 벡터 검색
- **DynamoDB**: 작업 추적
- **Transcribe**: 자막 생성
- **FFmpeg**: 비디오 트랜스코딩

## 아키텍처

```
Kiro CLI (AI 클라이언트)
    ↓ MCP Protocol
Standard MCP Server
    ↓ 직접 호출
AWS Services (Bedrock, S3 Vectors, etc.)
```

**핵심**: AI 에이전트가 서버 내부에 없음 → 더 빠르고 저렴함

## 라이선스

MIT
