#!/usr/bin/env python3
import asyncio
import json
import time
import uuid
import hashlib
import boto3
import os
from typing import Any
from mcp.server import Server
from mcp.types import Tool, TextContent

# 환경변수
REGION = os.getenv('AWS_REGION', 'us-east-1')
S3_VECTORS_BUCKET = os.getenv('S3_VECTORS_BUCKET')
S3_VECTORS_INDEX = os.getenv('S3_VECTORS_INDEX')
DYNAMODB_TABLE = os.getenv('DYNAMODB_TABLE')
S3_UPLOAD_BUCKET = os.getenv('S3_UPLOAD_BUCKET')
MARENGO_MODEL_ID = os.getenv('MARENGO_MODEL_ID', 'twelvelabs.marengo-embed-3-0-v1:0')
PEGASUS_MODEL_ID = os.getenv('PEGASUS_MODEL_ID', 'us.twelvelabs.pegasus-1-2-v1:0')

if not all([S3_VECTORS_BUCKET, S3_VECTORS_INDEX, DYNAMODB_TABLE]):
    raise ValueError("필수 환경변수 누락: S3_VECTORS_BUCKET, S3_VECTORS_INDEX, DYNAMODB_TABLE")

# AWS 클라이언트
bedrock_runtime = boto3.client('bedrock-runtime', region_name=REGION)
s3 = boto3.client('s3', region_name=REGION)
s3vectors = boto3.client('s3vectors', region_name=REGION)
dynamodb = boto3.resource('dynamodb', region_name=REGION)
task_table = dynamodb.Table(DYNAMODB_TABLE)
transcribe = boto3.client('transcribe', region_name=REGION)
from botocore.config import Config
s3_sigv4 = boto3.client('s3', region_name=REGION, config=Config(signature_version='s3v4'))
ACCOUNT_ID = boto3.client('sts').get_caller_identity()['Account']

# 유틸리티 함수
def upload_local_to_s3(local_path: str) -> str:
    if not S3_UPLOAD_BUCKET:
        raise ValueError("S3_UPLOAD_BUCKET 환경변수가 설정되지 않음")
    parts = S3_UPLOAD_BUCKET.split('/', 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ''
    filename = os.path.basename(local_path)
    key = f"{prefix}/{filename}" if prefix else filename
    s3.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"

def resolve_video_path(path: str) -> str:
    if path.startswith('s3://'):
        return path
    expanded_path = os.path.expanduser(path)
    if os.path.isfile(expanded_path):
        return upload_local_to_s3(expanded_path)
    raise ValueError(f"파일을 찾을 수 없음: {path}")

# 도구 함수들
def create_video_embedding(video_path: str) -> dict:
    s3_uri = resolve_video_path(video_path)
    task_id = str(uuid.uuid4())[:8]
    bucket, key = s3_uri.split('/')[2], '/'.join(s3_uri.split('/')[3:])
    task_table.put_item(Item={'task_id': task_id, 's3_uri': s3_uri, 's3_bucket': bucket, 's3_key': key, 'status': 'processing', 'created_at': int(time.time())})
    
    response = bedrock_runtime.start_async_invoke(
        modelId=MARENGO_MODEL_ID,
        modelInput={'inputType': 'video', 'video': {'mediaSource': {'s3Location': {'uri': s3_uri, 'bucketOwner': ACCOUNT_ID}}, 'embeddingOption': ['visual', 'audio'], 'embeddingScope': ['clip'], 'segmentation': {'method': 'fixed', 'fixed': {'durationSec': 6}}}},
        outputDataConfig={'s3OutputDataConfig': {'s3Uri': f's3://{bucket}/embeddings/{task_id}/'}}
    )
    invocation_arn = response['invocationArn']
    task_table.update_item(Key={'task_id': task_id}, UpdateExpression='SET invocation_arn = :arn', ExpressionAttributeValues={':arn': invocation_arn})
    
    for _ in range(60):
        status = bedrock_runtime.get_async_invoke(invocationArn=invocation_arn)['status']
        if status == 'Completed': break
        if status in ['Failed', 'Expired']: return {'error': f'실패: {status}', 'task_id': task_id}
        time.sleep(10)
    else:
        return {'error': '타임아웃', 'task_id': task_id}
    
    output_uri = bedrock_runtime.get_async_invoke(invocationArn=invocation_arn)['outputDataConfig']['s3OutputDataConfig']['s3Uri']
    prefix = '/'.join(output_uri.split('/')[3:])
    objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    json_key = next((o['Key'] for o in objs.get('Contents', []) if o['Key'].endswith('output.json')), None)
    data = json.loads(s3.get_object(Bucket=bucket, Key=json_key)['Body'].read())
    clips = data.get('data', [])
    
    vectors = [{'key': f"{task_id}_{c['embeddingOption']}_{c['startSec']}_{c['endSec']}", 'data': {'float32': c['embedding']}, 'metadata': {'task_id': task_id, 'embeddingOption': c['embeddingOption'], 'startSec': c['startSec'], 'endSec': c['endSec']}} for c in clips]
    for i in range(0, len(vectors), 100): s3vectors.put_vectors(vectorBucketName=S3_VECTORS_BUCKET, indexName=S3_VECTORS_INDEX, vectors=vectors[i:i+100])
    
    task_table.update_item(Key={'task_id': task_id}, UpdateExpression='SET #s = :s, clip_count = :c', ExpressionAttributeNames={'#s': 'status'}, ExpressionAttributeValues={':s': 'completed', ':c': len(clips)})
    return {'task_id': task_id, 'status': 'completed', 's3_uri': s3_uri, 'stored_clips': len(clips)}

def search_video_clips(query: str, top_k: int = 50, max_results: int = 10) -> dict:
    response = bedrock_runtime.invoke_model(modelId=MARENGO_MODEL_ID, body=json.dumps({'inputType': 'text', 'text': {'inputText': query}}), contentType='application/json')
    emb = json.loads(response['body'].read())['data'][0]['embedding']
    results = s3vectors.query_vectors(vectorBucketName=S3_VECTORS_BUCKET, indexName=S3_VECTORS_INDEX, queryVector={'float32': emb}, topK=top_k, returnDistance=True, returnMetadata=True, filter={'embeddingOption': {'$in': ['visual', 'audio']}})
    
    clips, seen = [], []
    for v in results.get('vectors', []):
        if len(clips) >= max_results: break
        meta = v.get('metadata', {})
        task_info = task_table.get_item(Key={'task_id': meta.get('task_id')}).get('Item', {})
        start, end = int(meta.get('startSec', 0)), int(meta.get('endSec', 0))
        if not any(abs(start - s) <= 3 or abs(end - e) <= 3 for s, e in seen):
            seen.append((start, end))
            clips.append({'video': task_info.get('s3_key', ''), 's3_bucket': task_info.get('s3_bucket', ''), 'type': meta.get('embeddingOption'), 'timestamp': f'{start//60}:{start%60:02d}-{end//60}:{end%60:02d}', 'start_sec': start, 'end_sec': end, 'similarity_score': v.get('distance', 0)})
    return {'query': query, 'clips': clips}

def get_clip_playback_url(s3_bucket: str, s3_key: str, start_sec: int, end_sec: int) -> dict:
    base_url = s3_sigv4.generate_presigned_url('get_object', Params={'Bucket': s3_bucket, 'Key': s3_key}, ExpiresIn=3600)
    return {'playback_url': f'{base_url}#t={start_sec},{end_sec}'}

def summarize_video(video_path: str, prompt: str = '이 영상을 챕터를 구분해서 3문장 정도로 요약해줘') -> dict:
    s3_uri = resolve_video_path(video_path)
    response = bedrock_runtime.invoke_model(modelId=PEGASUS_MODEL_ID, body=json.dumps({'inputPrompt': prompt, 'mediaSource': {'s3Location': {'uri': s3_uri, 'bucketOwner': ACCOUNT_ID}}}), contentType='application/json')
    result = json.loads(response['body'].read())
    return {'s3_uri': s3_uri, 'summary': result.get('message', result.get('response', result))}

def _get_transcript_data(s3_uri: str):
    bucket = s3_uri.split('/')[2]
    job_name = f"transcript-{hashlib.md5(s3_uri.encode()).hexdigest()[:8]}"
    return json.loads(s3.get_object(Bucket=bucket, Key=f'transcripts/{job_name}.json')['Body'].read())

def _ensure_transcript(s3_uri: str):
    bucket = s3_uri.split('/')[2]
    job_name = f"transcript-{hashlib.md5(s3_uri.encode()).hexdigest()[:8]}"
    output_key = f'transcripts/{job_name}.json'
    try:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)['TranscriptionJob']['TranscriptionJobStatus']
    except transcribe.exceptions.BadRequestException:
        key = '/'.join(s3_uri.split('/')[3:])
        ext = key.split('.')[-1].lower()
        media_format = 'mp4' if ext in ['mov', 'mp4', 'm4a'] else ext
        transcribe.start_transcription_job(TranscriptionJobName=job_name, Media={'MediaFileUri': s3_uri}, MediaFormat=media_format, LanguageCode='ko-KR', OutputBucketName=bucket, OutputKey=output_key)
        status = 'IN_PROGRESS'
    if status == 'IN_PROGRESS':
        while status == 'IN_PROGRESS':
            time.sleep(5)
            status = transcribe.get_transcription_job(TranscriptionJobName=job_name)['TranscriptionJob']['TranscriptionJobStatus']
    return {'transcript_file': f's3://{bucket}/{output_key}'} if status == 'COMPLETED' else {'error': '실패'}

def get_transcript(video_path: str) -> dict:
    s3_uri = resolve_video_path(video_path)
    try:
        data = _get_transcript_data(s3_uri)
    except:
        result = _ensure_transcript(s3_uri)
        if 'error' in result: return result
        data = _get_transcript_data(s3_uri)
    grouped = {}
    for item in data['results']['items']:
        if item['type'] == 'pronunciation':
            sec = int(float(item.get('start_time', 0)))
            key = f"{sec//5*5//60}:{sec//5*5%60:02d}"
            grouped[key] = grouped.get(key, '') + ' ' + item['alternatives'][0]['content']
    return {'transcript': [{'시간': k, '자막': v.strip()} for k, v in grouped.items()]}

def get_keywords(video_path: str) -> dict:
    s3_uri = resolve_video_path(video_path)
    try:
        data = _get_transcript_data(s3_uri)
    except:
        result = _ensure_transcript(s3_uri)
        if 'error' in result: return result
        data = _get_transcript_data(s3_uri)
    transcript = ' '.join([i['alternatives'][0]['content'] for i in data['results']['items'] if i['type'] == 'pronunciation'])[:2000]
    response = bedrock_runtime.invoke_model(modelId='us.anthropic.claude-3-5-haiku-20241022-v1:0', body=json.dumps({'anthropic_version': 'bedrock-2023-05-31', 'max_tokens': 256, 'messages': [{'role': 'user', 'content': f'핵심 키워드 10개 JSON 배열로만:\n{transcript}'}]}), contentType='application/json')
    return {'keywords': json.loads(response['body'].read())['content'][0]['text']}

def transcode_clip(video_path: str, start_sec: int, end_sec: int, output_filename: str = None) -> dict:
    import subprocess
    
    if not os.path.isfile(video_path):
        return {'error': f'파일을 찾을 수 없음: {video_path}'}
    
    if output_filename is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{base_name}_clip_{start_sec}_{end_sec}.mp4"
    
    output_path = os.path.join(os.getcwd(), output_filename)
    duration = end_sec - start_sec
    cmd = ['ffmpeg', '-i', video_path, '-ss', str(start_sec), '-t', str(duration), '-c:v', 'libx264', '-c:a', 'aac', '-y', output_path]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return {'error': f'FFmpeg 실패: {result.stderr}'}
        file_size = os.path.getsize(output_path)
        return {'status': 'success', 'output_file': output_path, 'file_size_mb': round(file_size / (1024 * 1024), 2), 'duration_sec': duration, 'start_sec': start_sec, 'end_sec': end_sec}
    except subprocess.TimeoutExpired:
        return {'error': '트랜스코딩 타임아웃 (60초 초과)'}
    except Exception as e:
        return {'error': f'트랜스코딩 실패: {str(e)}'}

# MCP Server
app = Server("video-processing-standard")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="create_video_embedding",
            description="비디오를 6초 단위로 임베딩하여 S3 Vectors에 저장합니다. 검색 전에 반드시 실행해야 합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {"type": "string", "description": "로컬 파일 경로 (예: ~/Videos/test.mp4) 또는 S3 URI (예: s3://bucket/video.mp4)"}
                },
                "required": ["video_path"]
            }
        ),
        Tool(
            name="search_video_clips",
            description="자연어 쿼리로 비디오 클립을 검색합니다. 임베딩이 먼저 생성되어 있어야 합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "검색할 장면 설명 (예: '골 장면', '웃는 장면')"},
                    "top_k": {"type": "integer", "description": "검색할 최대 후보 수", "default": 50},
                    "max_results": {"type": "integer", "description": "반환할 최대 결과 수", "default": 10}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_clip_playback_url",
            description="특정 클립의 재생 URL을 생성합니다 (타임스탬프 포함).",
            inputSchema={
                "type": "object",
                "properties": {
                    "s3_bucket": {"type": "string", "description": "S3 버킷 이름"},
                    "s3_key": {"type": "string", "description": "S3 객체 키"},
                    "start_sec": {"type": "integer", "description": "시작 시간 (초)"},
                    "end_sec": {"type": "integer", "description": "종료 시간 (초)"}
                },
                "required": ["s3_bucket", "s3_key", "start_sec", "end_sec"]
            }
        ),
        Tool(
            name="summarize_video",
            description="비디오 전체를 요약합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {"type": "string", "description": "로컬 파일 경로 또는 S3 URI"},
                    "prompt": {"type": "string", "description": "요약 프롬프트", "default": "이 영상을 챕터를 구분해서 3문장 정도로 요약해줘"}
                },
                "required": ["video_path"]
            }
        ),
        Tool(
            name="get_transcript",
            description="비디오의 자막을 생성하거나 조회합니다 (5초 단위 그룹화).",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {"type": "string", "description": "로컬 파일 경로 또는 S3 URI"}
                },
                "required": ["video_path"]
            }
        ),
        Tool(
            name="get_keywords",
            description="비디오 자막에서 핵심 키워드를 추출합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {"type": "string", "description": "로컬 파일 경로 또는 S3 URI"}
                },
                "required": ["video_path"]
            }
        ),
        Tool(
            name="transcode_clip",
            description="비디오의 특정 구간을 추출하여 로컬에 MP4 파일로 저장합니다 (FFmpeg 사용).",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {"type": "string", "description": "로컬 비디오 파일 경로"},
                    "start_sec": {"type": "integer", "description": "시작 시간 (초)"},
                    "end_sec": {"type": "integer", "description": "종료 시간 (초)"},
                    "output_filename": {"type": "string", "description": "출력 파일명 (선택, 기본값: 자동 생성)"}
                },
                "required": ["video_path", "start_sec", "end_sec"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    try:
        if name == "create_video_embedding":
            result = create_video_embedding(arguments["video_path"])
        elif name == "search_video_clips":
            result = search_video_clips(
                arguments["query"],
                arguments.get("top_k", 50),
                arguments.get("max_results", 10)
            )
        elif name == "get_clip_playback_url":
            result = get_clip_playback_url(
                arguments["s3_bucket"],
                arguments["s3_key"],
                arguments["start_sec"],
                arguments["end_sec"]
            )
        elif name == "summarize_video":
            result = summarize_video(
                arguments["video_path"],
                arguments.get("prompt", "이 영상을 챕터를 구분해서 3문장 정도로 요약해줘")
            )
        elif name == "get_transcript":
            result = get_transcript(arguments["video_path"])
        elif name == "get_keywords":
            result = get_keywords(arguments["video_path"])
        elif name == "transcode_clip":
            result = transcode_clip(
                arguments["video_path"],
                arguments["start_sec"],
                arguments["end_sec"],
                arguments.get("output_filename")
            )
        else:
            raise ValueError(f"Unknown tool: {name}")
        
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]

async def async_main():
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
