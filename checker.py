import os
import streamlit as st
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import whisper
from pytube import YouTube
import tempfile
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import yt_dlp

# Load environment variables
load_dotenv()
API_KEY = os.getenv('YOUTUBE_API_KEY')

if not API_KEY:
    st.error('APIキーが設定されていません。.envファイルを確認してください。')
    st.stop()

youtube = build('youtube', 'v3', developerKey=API_KEY, static_discovery=False)

# コンプライアンス基準の定義
COMPLIANCE_RULES = {
    '一般倫理基準': {
        '差別的表現': [
            '差別', '侮蔑', '馬鹿', 'バカ', 'アホ', '死ね', 'きもい',
            '在日', '黒人', '白人', 'ホモ', 'レズ', '障害者', '病人',
            'バ度', '知恵遅れ', '気違い', '基地外'
        ],
        '性別差別': [
            '女のくせに', '男のくせに', '女だから', '男だから',
            '女の分際', '男の分際', '女らしく', '男らしく'
        ],
        '宗教関連': [
            '仏教徒', 'キリスト教徒', 'イスラム教徒', '創価',
            '統一教会', '幸福の科学'
        ],
        '不謹慎な表現': [
            '震災', '津波', '台風', '災害', 'コロナ', 'パンデミック',
            '事故', '自殺', '死亡', '殺人'
        ]
    },
    '法律基準': {
        '景表法関連': [
            '最高', '最低', '最大', '最小', '最高級', '業界一', '日本一',
            '世界一', '最安値', '完全', '完璧', '永久', '確実', '保証',
            '間違いない', '必ず'
        ],
        '薬機法関連': [
            '治療', '治癒', '完治', '予防', '効能', '効果', '副作用',
            '医薬品', '医療機器', '処方', '投薬', '服用'
        ],
        '労働法関連': [
            '残業代', '賃金', '給与', '未払い', '長時間労働', 
            'サービス残業', '休憩なし', '有給休暇', '労災'
        ]
    },
    '著作権・肖像権': {
        '著作権侵害': [
            '無断使用', '無断転載', '無許可', 'コピー', '著作権',
            '転載', '複製', '海賊版', '違法アップロード'
        ],
        '肖像権侵害': [
            '個人情報', '住所', '電話番号', 'メールアドレス',
            '顔写真', '身分証', '免許証', 'パスポート'
        ]
    },
    '企業秘密': {
        '機密情報': [
            '機密', '極秘', '社外秘', '部外秘', '関係者外', 
            '未発表', '未公開', '内部情報', 'インサイダー',
            '新製品', '開発中', '試作品', '企画書', '設計図','売上','利益'
        ]
    }
}

# 表記ゆれ対応の定義
KEYWORD_VARIATIONS = {
    '差別': ['さべつ', 'サベツ'],
    '侮蔑': ['ぶべつ', 'バカにする'],
    '馬鹿': ['バカ', 'アホ', 'あほ', '阿呆', 'アフォ'],
    '死ね': ['しね', 'くたばれ', '〇ね'],
    'きもい': ['キモイ', 'きもっ'],
    '障害者': ['しょうがいしゃ', 'ショウガイシャ', '障がい者'],
    '病人': ['びょうにん', 'ビョウニン'],
    '知恵遅れ': ['ちえおくれ', 'チエオクレ'],
    '気違い': ['きちがい', 'キチガイ'],
    '基地外': ['きちがい', 'キチガイ'],
    '女のくせに': ['おんなのくせに', 'オンナのくせに'],
    '男のくせに': ['おとこのくせに', 'オトコのくせに'],
    '女だから': ['おんなだから', 'オンナだから'],
    '男だから': ['おとこだから', 'オトコだから'],
    '女の分際': ['おんなのぶんざい', 'オンナのブンザイ'],
    '男の分際': ['おとこのぶんざい', 'オトコのブンザイ'],
    '仏教徒': ['ぶっきょうと', 'ブッキョウト'],
    'キリスト教徒': ['クリスチャン', 'くりすちゃん'],
    'イスラム教徒': ['ムスリム', 'むすりむ'],
    '震災': ['しんさい', 'シンサイ'],
    '津波': ['つなみ', 'ツナミ'],
    '台風': ['たいふう', 'タイフウ'],
    '災害': ['さいがい', 'サイガイ'],
    'コロナ': ['corona', 'Corona', 'COVID'],
    'パンデミック': ['pandemic', 'ぱんでみっく'],
    '自殺': ['じさつ', 'ジサツ'],
    '死亡': ['しぼう', 'シボウ'],
    '殺人': ['さつじん', 'サツジン'],
    '最高': ['さいこう', 'サイコウ', 'サイコー'],
    '最低': ['さいてい', 'サイテイ'],
    '最大': ['さいだい', 'サイダイ'],
    '最小': ['さいしょう', 'サイショウ'],
    '最高級': ['さいこうきゅう', 'サイコウキュウ'],
    '業界一': ['ぎょうかいいち', 'ギョウカイイチ'],
    '完全': ['かんぜん', 'カンゼン'],
    '完璧': ['かんぺき', 'カンペキ'],
    '永久': ['えいきゅう', 'エイキュウ'],
    '確実': ['かくじつ', 'カクジツ'],
    '治療': ['ちりょう', 'チリョウ'],
    '治癒': ['ちゆ', 'チユ'],
    '完治': ['かんち', 'カンチ'],
    '予防': ['よぼう', 'ヨボウ'],
    '効能': ['こうのう', 'コウノウ'],
    '効果': ['こうか', 'コウカ'],
    '副作用': ['ふくさよう', 'フクサヨウ'],
    '医薬品': ['いやくひん', 'イヤクヒン'],
    '医療機器': ['いりょうきき', 'イリョウキキ'],
    '残業代': ['ざんぎょうだい', 'ザンギョウダイ'],
    '賃金': ['ちんぎん', 'チンギン'],
    '給与': ['きゅうよ', 'キュウヨ'],
    '未払い': ['みばらい', 'ミバライ'],
    '長時間労働': ['ちょうじかんろうどう', 'チョウジカンロウドウ'],
    '有給休暇': ['ゆうきゅう', 'ユウキュウ'],
    '労災': ['ろうさい', 'ロウサイ'],
    '無断使用': ['むだんしよう', 'ムダンシヨウ'],
    '無断転載': ['むだんてんさい', 'ムダンテンサイ'],
    '無許可': ['むきょか', 'ムキョカ'],
    '著作権': ['ちょさくけん', 'チョサクケン'],
    '複製': ['ふくせい', 'フクセイ'],
    '機密': ['きみつ', 'キミツ'],
    '極秘': ['ごくひ', 'ゴクヒ'],
    '社外秘': ['しゃがいひ', 'シャガイヒ'],
    '部外秘': ['ぶがいひ', 'ブガイヒ'],
    'インサイダー': ['insider', 'Insider'],
    '新製品': ['しんせいひん', 'シンセイヒン'],
    '開発中': ['かいはつちゅう', 'カイハツチュウ'],
    '試作品': ['しさくひん', 'シサクヒン'],
    '企画書': ['きかくしょ', 'キカクショ'],
    '設計図': ['せっけいず', 'セッケイズ'],
    '売上': ['うりあげ', 'ウリアゲ'],
    '利益': ['りえき', 'リエキ']
}

def normalize_text(text):
    """テキストの正規化（表記ゆれ対応）"""
    normalized_text = text.lower()
    for standard, variations in KEYWORD_VARIATIONS.items():
        for variant in [standard] + variations:
            normalized_text = normalized_text.replace(variant.lower(), standard.lower())
    return normalized_text

def extract_video_id(url):
    """URLからビデオIDを抽出"""
    parsed_url = urlparse(url)
    if 'youtube.com' in parsed_url.netloc:
        return parse_qs(parsed_url.query).get('v', [None])[0]
    elif 'youtu.be' in parsed_url.netloc:
        return parsed_url.path[1:]
    return None

def check_inappropriate_content(text):
    """不適切な表現のチェック"""
    results = []
    normalized_text = normalize_text(text)
    
    for main_category, subcategories in COMPLIANCE_RULES.items():
        category_results = []
        
        for sub_category, keywords in subcategories.items():
            for keyword in keywords:
                if normalize_text(keyword) in normalized_text:
                    category_results.append(f"・{sub_category}: '{keyword}'を含む表現があります")
        
        if category_results:
            results.append(f"\n【{main_category}】")
            results.extend(category_results)
    
    return results

def analyze_video_content(video_data):
    """動画コンテンツの総合分析"""
    return {
        'title': check_inappropriate_content(video_data['snippet']['title']),
        'description': check_inappropriate_content(video_data['snippet']['description']),
        'tags': check_inappropriate_content(' '.join(video_data['snippet'].get('tags', [])))
    }

def display_analysis_results(analysis_results):
    """分析結果の詳細表示"""
    st.write("チェック項目と検査プロセス:")
    
    # 各カテゴリーのチェック内容を表示
    for main_category, subcategories in COMPLIANCE_RULES.items():
        with st.expander(f"📋 {main_category}のチェック項目"):
            for sub_category, keywords in subcategories.items():
                st.write(f"✓ {sub_category}:")
                st.write(f"  確認キーワード: {', '.join(keywords)}")
    
    st.write("---")
    st.write("分析結果:")
    
    if any(analysis_results.values()):
        st.error('以下の注意点が見つかりました：')
        
        sections = {
            'title': 'タイトル',
            'description': '説明文',
            'tags': 'タグ'
        }
        
        for section, name in sections.items():
            if analysis_results[section]:
                st.warning(f'■ {name}での注意点：')
                for result in analysis_results[section]:
                    st.write(result)
            else:
                st.success(f'✓ {name}は問題なし')
    else:
        st.success('コンプライアンス上の重大な問題は見つかりませんでした')
        st.write("判定根拠:")
        for section in ['タイトル', '説明文', 'タグ']:
            st.write(f"✓ {section}：すべてのキーワードチェックで該当なし")

def get_video_transcription(video_id):
    """動画のテキストを時間情報付きで取得"""
    try:
        st.write("字幕/音声の解析を開始...")
        
        try:
            st.write("字幕の取得を試みています...")
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ja', 'en'])
            st.success("✅ 字幕の取得に成功しました")
            
            # 時間情報付きで字幕を保持
            formatted_transcript = []
            for entry in transcript:
                time_in_minutes = int(entry['start']) // 60
                time_in_seconds = int(entry['start']) % 60
                formatted_transcript.append({
                    'text': entry['text'],
                    'time': f"{time_in_minutes}:{time_in_seconds:02d}",
                    'start': entry['start']
                })
            
            return {
                'text': " ".join([t['text'] for t in formatted_transcript]),
                'source': '字幕',
                'segments': formatted_transcript
            }
            
        except Exception as e:
            st.warning(f"字幕の取得に失敗しました。音声認識を開始します。")
            return get_audio_transcription(video_id)
            
    except Exception as e:
        st.error(f"テキスト取得でエラーが発生: {str(e)}")
        return None

def get_audio_transcription(video_id):
    """音声認識を実行"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, f"temp_audio_{video_id}.mp4")
            
            st.write("音声ファイルをダウンロード中...")
            yt = YouTube(f'https://www.youtube.com/watch?v={video_id}')
            audio_stream = yt.streams.filter(only_audio=True).first()
            audio_stream.download(filename=audio_path)
            
            st.write("音声認識モデルを準備中...")
            model = whisper.load_model("base")
            
            st.write("音声認識を実行中...")
            result = model.transcribe(audio_path)
            
            st.success("✅ 音声認識が完了しました")
            return {
                'text': result["text"],
                'source': '音声認識'
            }
            
    except Exception as e:
        st.error(f"音声認識でエラーが発生: {str(e)}")
        return None

def analyze_text_content(text_data):
    """テキストの分析を時間情報付きで実行"""
    if not text_data:
        return
        
    text = text_data['text']
    normalized_text = normalize_text(text)
    issues = check_inappropriate_content(normalized_text)
    
    if issues:
        st.warning(f"{text_data['source']}から以下の注意点が見つかりました：")
        
        # 時間情報付きで問題箇所を表示
        if 'segments' in text_data:
            for issue in issues:
                st.write(issue)
                # 該当箇所を探して時間情報と共に表示
                for segment in text_data['segments']:
                    normalized_segment = normalize_text(segment['text'])
                    if any(keyword in normalized_segment.lower() for keyword in get_keywords_from_issue(issue)):
                        st.write(f"→{segment['time']}「{segment['text']}」")
        else:
            # 音声認識の場合は時間情報なしで表示
            for issue in issues:
                st.write(issue)
    else:
        st.success(f"{text_data['source']}からは問題となる表現は見つかりませんでした")

def get_keywords_from_issue(issue):
    """問題報告から検索キーワードを抽出"""
    start = issue.find("'") + 1
    end = issue.find("'", start)
    if start != -1 and end != -1:
        return [issue[start:end]]
    return []

def analyze_video_frames(video_id):
    st.subheader("動画内コンテンツ分析")

    try:
        st.write("動画をダウンロード中...")
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, 'video.mp4')

            ydl_opts = {
                'format': 'best[height<=720]',
                'outtmpl': video_path,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f'https://www.youtube.com/watch?v={video_id}'])

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            model = ResNet50(weights='imagenet')

            st.write("フレーム分析を開始...")
            progress_bar = st.progress(0)

            frames = []
            frame_count = 0
            sampling_rate = 5 * fps  # 5秒ごとにフレームを取得

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % sampling_rate == 0:
                    frames.append({
                        'frame': frame,
                        'time': frame_count // fps
                    })

                frame_count += 1
                progress_bar.progress(frame_count / total_frames)

            cap.release()

            st.success(f"動画分析が完了しました（取得フレーム数: {len(frames)}）")

            # 動画再生バーの作成
            current_frame_index = st.slider("再生位置", 0, len(frames) - 1, 0)
            current_frame = frames[current_frame_index]['frame']
            current_time = frames[current_frame_index]['time']

            # 現在のフレームを表示
            st.image(current_frame, use_container_width=True)
            st.write(f"再生時間: {current_time}秒")

            # 危険な個所の検出とアラート表示
            pil_image = Image.fromarray(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))
            img = pil_image.resize((224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            preds = model.predict(x)
            predictions = decode_predictions(preds, top=5)[0]

            for _, label, confidence in predictions:
                if confidence > 0.2:
                    if 'weapon' in label or 'gun' in label:
                        st.warning(f"危険物の可能性: {label} ({confidence*100:.1f}%)")

    except Exception as e:
        st.error(f"動画分析でエラーが発生: {str(e)}")


def display_frame_results(frames_with_findings, video_path):
    cap = cv2.VideoCapture(video_path)
    for frame_data in frames_with_findings:
        frame_time = frame_data['time']
        frame_label = frame_data['label']
        confidence = frame_data['confidence']

        cap.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)
        ret, frame = cap.read()
        if ret:
            st.image(frame, caption=f"Time: {frame_time}s, Label: {frame_label} ({confidence})", use_column_width=True)
    cap.release()

def main():
    st.title('YouTube動画コンプライアンスチェッカー')
    url = st.text_input('YouTube URLを入力してください：')

    if url:
        video_id = extract_video_id(url)
        if video_id:
            try:
                video_response = youtube.videos().list(
                    part='snippet,contentDetails,status',
                    id=video_id
                ).execute()
                
                if video_response['items']:
                    video_data = video_response['items'][0]
                    
                    st.header('基本情報')
                    st.write('タイトル:', video_data['snippet']['title'])
                    st.write('チャンネル名:', video_data['snippet']['channelTitle'])
                    st.write('投稿日:', video_data['snippet']['publishedAt'])
                    
                    st.header('メタデータのコンプライアンス分析')
                    analysis_results = analyze_video_content(video_data)
                    display_analysis_results(analysis_results)
                    
                    st.header('音声/字幕の分析')
                    transcript_result = get_video_transcription(video_id)
                    
                    if transcript_result:
                        with st.expander("取得したテキストを表示"):
                            st.write(transcript_result['text'])
                        
                        analyze_text_content(transcript_result)

                    # 動画内容の視覚分析を呼び出す
                    st.header('動画内容の視覚分析')
                    analyze_video_frames(video_id)
                    
                else:
                    st.error('動画が見つかりませんでした')
            
            except Exception as e:
                st.error(f'エラーが発生しました: {str(e)}')
        else:
            st.error('有効なYouTube URLを入力してください')

if __name__ == '__main__':
    main()