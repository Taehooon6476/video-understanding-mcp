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
from strands import Agent, tool
from strands.models import BedrockModel
from botocore.config import Config

# 환경변수 로드
REGION = os.getenv('AWS_REGION', 'us-east-1')
S3_VECTORS_BUCKET = os.getenv('S3_VECTORS_BUCKET')
S3_VECTORS_INDEX = os.getenv('S3_VECTORS_INDEX')
DYNAMODB_TABLE = os.getenv('DYNAMODB_TABLE')
S3_UPLOAD_BUCKET = os.getenv('S3_UPLOAD_BUCKET')  # 로컬 파일 업로드용 (bucket/prefix 형식)
MARENGO_MODEL_ID = os.getenv('MARENGO_MODEL_ID', 'twelvelabs.marengo-embed-3-0-v1:0')
PEGASUS_MODEL_ID = os.getenv('PEGASUS_MODEL_ID', 'us.twelvelabs.pegasus-1-2-v1:0')
CLAUDE_MODEL_ID = os.getenv('CLAUDE_MODEL_ID', 'us.anthropic.claude-sonnet-4-20250514-v1:0')

# 필수 환경변수 체크
if not all([S3_VECTORS_BUCKET, S3_VECTORS_INDEX, DYNAMODB_TABLE]):
    raise ValueError("필수 환경변수 누락: S3_VECTORS_BUCKET, S3_VECTORS_INDEX, DYNAMODB_TABLE")

# AWS 클라이언트
bedrock_runtime = boto3.client('bedrock-runtime', region_name=REGION)
s3 = boto3.client('s3', region_name=REGION)
s3vectors = boto3.client('s3vectors', region_name=REGION)
dynamodb = boto3.resource('dynamodb', region_name=REGION)
task_table = dynamodb.Table(DYNAMODB_TABLE)
transcribe = boto3.client('transcribe', region_name=REGION)
s3_sigv4 = boto3.client('s3', region_name=REGION, config=Config(signature_version='s3v4'))
ACCOUNT_ID = boto3.client('sts').get_caller_identity()['Account']

claude_model = BedrockModel(model_id=CLAUDE_MODEL_ID, region_name=REGION)

def upload_local_to_s3(local_path: str) -> str:
    """로컬 파일을 S3에 업로드하고 S3 URI 반환"""
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
    """로컬 경로면 S3 업로드 후 URI 반환, S3 URI면 그대로 반환"""
    if path.startswith('s3://'):
        return path
    if os.path.isfile(path):
        return upload_local_to_s3(path)
    raise ValueError(f"파일을 찾을 수 없음: {path}")

# Strands 도구들
@tool
def create_video_embedding(video_path: str):
    '''비디오 임베딩 생성 (로컬 경로 또는 S3 URI)'''
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

@tool
def search_video_clips(query: str, top_k: int = 50, max_results: int = 10):
    '''텍스트로 비디오 클립 검색'''
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

@tool
def get_clip_playback_url(s3_bucket: str, s3_key: str, start_sec: int, end_sec: int):
    '''재생 URL 생성'''
    base_url = s3_sigv4.generate_presigned_url('get_object', Params={'Bucket': s3_bucket, 'Key': s3_key}, ExpiresIn=3600)
    return {'playback_url': f'{base_url}#t={start_sec},{end_sec}'}

@tool
def summarize_video(video_path: str, prompt: str = '이 영상을 챕터를 구분해서 3문장 정도로 요약해줘'):
    '''영상 요약 (로컬 경로 또는 S3 URI)'''
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

@tool
def get_transcript(video_path: str):
    '''자막 조회 (로컬 경로 또는 S3 URI)'''
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

@tool
def get_keywords(video_path: str):
    '''키워드 추출 (로컬 경로 또는 S3 URI)'''
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

@tool
def transcode_clip(video_path: str, start_sec: int, end_sec: int, output_filename: str = None):
    '''비디오 클립을 트랜스코딩하여 로컬에 저장 (FFmpeg 사용)'''
    import subprocess
    
    # 입력 파일 경로 확인
    if not os.path.isfile(video_path):
        return {'error': f'파일을 찾을 수 없음: {video_path}'}
    
    # 출력 파일명 생성
    if output_filename is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{base_name}_clip_{start_sec}_{end_sec}.mp4"
    
    # 현재 작업 디렉토리에 저장
    output_path = os.path.join(os.getcwd(), output_filename)
    
    # FFmpeg 명령어 구성
    duration = end_sec - start_sec
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-ss', str(start_sec),
        '-t', str(duration),
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-y',  # 덮어쓰기
        output_path
    ]
    
    try:
        # FFmpeg 실행
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            return {'error': f'FFmpeg 실패: {result.stderr}'}
        
        # 파일 크기 확인
        file_size = os.path.getsize(output_path)
        
        return {
            'status': 'success',
            'output_file': output_path,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'duration_sec': duration,
            'start_sec': start_sec,
            'end_sec': end_sec
        }
    except subprocess.TimeoutExpired:
        return {'error': '트랜스코딩 타임아웃 (60초 초과)'}
    except Exception as e:
        return {'error': f'트랜스코딩 실패: {str(e)}'}

# Agents
video_analysis_agent = Agent(model=claude_model, tools=[create_video_embedding, summarize_video], system_prompt='영상 분석 전문. 임베딩 생성과 요약.')
search_agent = Agent(model=claude_model, tools=[search_video_clips, get_clip_playback_url], system_prompt='영상 검색 전문. 클립 검색과 URL 생성. 검색 후 자동으로 URL 생성.')
transcript_agent = Agent(model=claude_model, tools=[get_transcript, get_keywords], system_prompt='자막 처리 전문. 자막 조회, 키워드 추출.')
transcoder_agent = Agent(model=claude_model, tools=[transcode_clip], system_prompt='비디오 트랜스코딩 전문. video_search 결과의 start_sec, end_sec를 사용하여 FFmpeg로 클립을 추출하고 로컬에 저장.')

# MCP Server
app = Server("video-processing")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(name="video_analysis", description="영상 분석 (임베딩, 요약)", inputSchema={"type": "object", "properties": {"query": {"type": "string", "description": "사용자의 요청을 그대로 전달. 번역하거나 수정하지 말 것."}}, "required": ["query"]}),
        Tool(name="video_search", description="영상 검색 (클립 검색, URL)", inputSchema={"type": "object", "properties": {"query": {"type": "string", "description": "사용자의 요청을 그대로 전달. 번역하거나 수정하지 말 것."}}, "required": ["query"]}),
        Tool(name="transcript", description="자막 처리 (자막, 키워드)", inputSchema={"type": "object", "properties": {"query": {"type": "string", "description": "사용자의 요청을 그대로 전달. 번역하거나 수정하지 말 것."}}, "required": ["query"]}),
        Tool(name="transcoder", description="비디오 트랜스코딩 (FFmpeg 클립 추출 및 저장)", inputSchema={"type": "object", "properties": {"query": {"type": "string", "description": "사용자의 요청을 그대로 전달. 번역하거나 수정하지 말 것."}}, "required": ["query"]})
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    query = arguments.get("query", "")
    if name == "video_analysis":
        response = video_analysis_agent(query)
    elif name == "video_search":
        response = search_agent(query)
    elif name == "transcript":
        response = transcript_agent(query)
    elif name == "transcoder":
        response = transcoder_agent(query)
    else:
        raise ValueError(f"Unknown tool: {name}")
    return [TextContent(type="text", text=str(response))]

async def async_main():
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
