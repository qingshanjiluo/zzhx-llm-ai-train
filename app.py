from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from model_manager import ModelManager
import os
import json
import traceback
import requests
import time

app = Flask(__name__)
manager = ModelManager()

# DeepSeek API 配置
DEEPSEEK_API_URL = 'https://api.deepseek.com/v1/chat/completions'
DEEPSEEK_MODEL = 'deepseek-chat'

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

DEFAULT_DATA_PATH = 'data/shakespeare.txt'

# 初始化时即加载数据集，避免 Flask 重载后状态丢失
def _init_dataset():
    if not os.path.exists(DEFAULT_DATA_PATH):
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            urllib.request.urlretrieve(url, DEFAULT_DATA_PATH)
        except Exception as e:
            print(f"[警告] 默认数据集下载失败: {e}")
            with open(DEFAULT_DATA_PATH, 'w', encoding='utf-8') as f:
                f.write("Hello world! This is a default training dataset. " * 50)
    if manager.train_data is None:
        try:
            manager.load_dataset(DEFAULT_DATA_PATH)
        except Exception as e:
            print(f"[警告] 初始数据集加载失败: {e}")

_init_dataset()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/api/dataset/info', methods=['GET'])
def dataset_info():
    try:
        if manager.train_data is None:
            manager.load_dataset(DEFAULT_DATA_PATH)
        info = {
            'vocab_size': manager.config.vocab_size,
            'total_chars': len(manager.train_data) + len(manager.val_data),
            'sample': manager.tokenizer.decode(manager.train_data[:300].tolist())
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/dataset/upload', methods=['POST'])
def upload_dataset():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '没有上传文件'})
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'success': False, 'error': '文件名为空'})
        if not file.filename.endswith('.txt'):
            return jsonify({'success': False, 'error': '仅支持 .txt 文件'})
        filepath = os.path.join('data', file.filename)
        file.save(filepath)
        info = manager.load_dataset(filepath)
        return jsonify({'success': True, 'info': info})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/train/start', methods=['POST'])
def start_training():
    try:
        if manager.is_training:
            return jsonify({'success': False, 'error': '训练已在进行中'})
        hyperparams = request.get_json(silent=True) or {}
        manager.train(hyperparams)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/train/stop', methods=['POST'])
def stop_training():
    try:
        manager.stop_training()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/train/status', methods=['GET'])
def training_status():
    return jsonify({
        'is_training': manager.is_training,
        'current_step': manager.current_step,
        'last_train_loss': manager._last_train_loss,
        'error': manager._training_error
    })

@app.route('/api/model/save', methods=['POST'])
def save_model():
    try:
        if manager.model is None:
            return jsonify({'success': False, 'error': '没有可保存的模型'})
        data = request.get_json(silent=True) or {}
        path = data.get('path', 'models/manual_save.pt')
        manager.save_model(path)
        return jsonify({'success': True, 'path': path})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/model/load', methods=['POST'])
def load_model():
    try:
        data = request.get_json(silent=True) or {}
        path = data.get('path', 'models/best_model.pt')
        if not os.path.exists(path):
            return jsonify({'success': False, 'error': '模型文件不存在'})
        manager.load_model(path)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/model/list', methods=['GET'])
def list_models():
    try:
        models_dir = 'models'
        files = []
        for f in os.listdir(models_dir):
            if f.endswith('.pt'):
                fp = os.path.join(models_dir, f)
                files.append({
                    'name': f,
                    'size': os.path.getsize(fp),
                    'mtime': os.path.getmtime(fp)
                })
        files.sort(key=lambda x: x['mtime'], reverse=True)
        return jsonify({'success': True, 'files': files})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/model/download/<path:filename>')
def download_model(filename):
    try:
        safe_name = os.path.basename(filename)
        path = os.path.join('models', safe_name)
        if not os.path.exists(path):
            return jsonify({'success': False, 'error': '文件不存在'}), 404
        return send_from_directory('models', safe_name, as_attachment=True)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat/generate', methods=['POST'])
def chat_generate():
    try:
        data = request.get_json(silent=True) or {}
        prompt = data.get('prompt', '')
        max_new = data.get('max_new', 200)
        temperature = data.get('temperature')
        top_k = data.get('top_k')
        top_p = data.get('top_p')
        if not prompt:
            return jsonify({'response': '', 'error': '提示词为空'})
        response = manager.generate_response(prompt, max_new, temperature, top_k, top_p)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': '', 'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/train/stream')
def train_stream():
    def event_stream():
        from queue import Queue
        q = Queue()
        def callback(data):
            q.put(data)
        manager.set_callback('progress', callback)
        manager.set_callback('status', callback)
        manager.set_callback('inference_log', callback)

        # 先回放历史消息（确保晚连接的前端也能看到已有进度）
        for msg in manager.get_message_history():
            yield f"data: {json.dumps(msg)}\n\n"

        # 再进入实时监听循环
        while manager.is_training:
            try:
                data = q.get(timeout=5)
                yield f"data: {json.dumps(data)}\n\n"
            except:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        # 发送结束标志
        yield f"data: {json.dumps({'type': 'end', 'message': 'Stream closed'})}\n\n"
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/api/dataset/generate', methods=['POST'])
def generate_dataset():
    """使用 DeepSeek API 生成训练数据集，支持流式返回进度。可选择基于当前上传的数据集生成。"""
    req_data = request.get_json(silent=True) or {}
    api_key = req_data.get('api_key', '').strip()
    topic = req_data.get('topic', '').strip()
    style = req_data.get('style', '').strip()
    total_chars = int(req_data.get('total_chars', 5000))
    temperature = float(req_data.get('temperature', 0.8))
    use_reference = bool(req_data.get('use_reference', False))

    # 在请求上下文内读取参考文件内容
    reference_sample = ''
    if use_reference and hasattr(manager, '_dataset_path') and manager._dataset_path and os.path.exists(manager._dataset_path):
        try:
            with open(manager._dataset_path, 'r', encoding='utf-8') as f:
                text = f.read()
            # 截取代表性样本（前 2000 + 中间 2000 + 末尾 2000 字符）
            if len(text) > 6000:
                reference_sample = text[:2000] + '\n...\n' + text[len(text)//2-1000:len(text)//2+1000] + '\n...\n' + text[-2000:]
            else:
                reference_sample = text
        except Exception as e:
            print(f"[参考文件读取失败] {e}")

    def stream_generate(api_key, topic, style, total_chars, temperature, reference_sample):
        if not api_key:
            yield f"data: {json.dumps({'type': 'error', 'message': '请提供 DeepSeek API Key'})}\n\n"
            return
        if not topic:
            yield f"data: {json.dumps({'type': 'error', 'message': '请提供生成主题'})}\n\n"
            return

        system_prompt = (
            "你是一个专业的数据集生成助手。你的任务是生成高质量的纯文本训练数据。"
            "输出必须是纯文本，不要包含任何 Markdown 格式、代码块、标题标记或额外说明。"
            "确保内容连贯、自然、适合语言模型训练。"
        )

        user_prompt = ''
        if reference_sample:
            user_prompt += (
                "【参考样本】以下是我上传的现有数据集样本，请仔细学习其语言风格、"
                "用词习惯、句式结构、主题内容和格式特征，然后生成与之一致风格的新内容：\n"
                "---\n" + reference_sample + "\n---\n\n"
            )
        user_prompt += f"主题/领域：{topic}\n"
        if style:
            user_prompt += f"风格要求：{style}\n"
        user_prompt += f"目标字符数：约 {total_chars} 字符\n"
        user_prompt += "请直接输出纯文本内容，不要加任何格式标记。"
        if reference_sample:
            user_prompt += "务必保持与参考样本完全一致的语言风格和格式。"

        yield f"data: {json.dumps({'type': 'status', 'message': '正在调用 DeepSeek API...'})}\n\n"

        try:
            resp = requests.post(
                DEEPSEEK_API_URL,
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': DEEPSEEK_MODEL,
                    'messages': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ],
                    'temperature': temperature,
                    'max_tokens': min(8192, max(1000, total_chars // 2)),
                    'stream': False
                },
                timeout=120
            )
            resp.raise_for_status()
            result = resp.json()
            generated_text = result['choices'][0]['message']['content']

            # 清理生成的文本：去除 Markdown 代码块标记等
            cleaned = generated_text.strip()
            if cleaned.startswith('```'):
                cleaned = '\n'.join(cleaned.split('\n')[1:])
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3].strip()
            cleaned = cleaned.strip()

            # 保存到文件
            timestamp = int(time.time())
            filename = f"ai_generated_{timestamp}.txt"
            filepath = os.path.join('data', filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(cleaned)

            # 自动加载为新数据集
            info = manager.load_dataset(filepath)

            yield f"data: {json.dumps({'type': 'done', 'message': '数据集生成完成', 'filename': filename, 'chars': len(cleaned), 'info': info})}\n\n"
        except requests.exceptions.HTTPError as e:
            err_msg = 'API 请求失败'
            try:
                err_body = e.response.json()
                err_msg = err_body.get('error', {}).get('message', str(e))
            except Exception:
                err_msg = str(e)
            yield f"data: {json.dumps({'type': 'error', 'message': err_msg})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(stream_generate(api_key, topic, style, total_chars, temperature, reference_sample), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
