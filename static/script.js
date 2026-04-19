// 加载数据集信息
fetch('/api/dataset/info')
    .then(res => res.json())
    .then(data => {
        document.getElementById('dataset-info').innerHTML = `
            <p><strong>词汇表:</strong> ${data.vocab_size}</p>
            <p><strong>总字符:</strong> ${data.total_chars}</p>
            <p><strong>预览:</strong> ${escapeHtml(data.sample.substring(0, 150))}...</p>
        `;
    });

// 上传数据集
document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = document.getElementById('dataset-file').files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch('/api/dataset/upload', { method: 'POST', body: formData });
    const result = await res.json();
    if (result.success) {
        alert('数据集已加载，页面将刷新');
        location.reload();
    } else {
        alert('上传失败');
    }
});

// 图表初始化
const chartCommon = {
    type: 'line',
    options: {
        responsive: true,
        maintainAspectRatio: true,
        animation: false,
        plugins: { legend: { display: false } },
        scales: {
            x: { ticks: { maxTicksLimit: 8 } }
        }
    }
};

const lossChart = new Chart(document.getElementById('lossChart').getContext('2d'), {
    ...chartCommon,
    data: {
        labels: [],
        datasets: [
            { label: '训练损失', data: [], borderColor: 'rgba(54,162,235,1)', backgroundColor: 'rgba(54,162,235,0.1)', fill: true, tension: 0.3, pointRadius: 2 },
            { label: '验证损失', data: [], borderColor: 'rgba(255,99,132,1)', backgroundColor: 'rgba(255,99,132,0.1)', fill: true, tension: 0.3, pointRadius: 2 }
        ]
    },
    options: {
        ...chartCommon.options,
        plugins: { legend: { display: true, labels: { boxWidth: 12, font: { size: 11 } } } },
        scales: { ...chartCommon.options.scales, y: { title: { display: true, text: 'Loss' } } }
    }
});

const lrChart = new Chart(document.getElementById('lrChart').getContext('2d'), {
    ...chartCommon,
    data: { labels: [], datasets: [{ label: 'LR', data: [], borderColor: '#9b59b6', fill: false, pointRadius: 2 }] },
    options: { ...chartCommon.options, scales: { ...chartCommon.options.scales, y: { title: { display: true, text: 'Learning Rate' } } } }
});

const pplChart = new Chart(document.getElementById('pplChart').getContext('2d'), {
    ...chartCommon,
    data: { labels: [], datasets: [{ label: 'PPL', data: [], borderColor: '#e67e22', fill: false, pointRadius: 2 }] },
    options: { ...chartCommon.options, scales: { ...chartCommon.options.scales, y: { title: { display: true, text: 'PPL' } } } }
});

const speedChart = new Chart(document.getElementById('speedChart').getContext('2d'), {
    ...chartCommon,
    data: { labels: [], datasets: [{ label: 'Tokens/s', data: [], borderColor: '#2ecc71', backgroundColor: 'rgba(46,204,113,0.1)', fill: true, pointRadius: 2 }] },
    options: { ...chartCommon.options, scales: { ...chartCommon.options.scales, y: { title: { display: true, text: 'Tokens/sec' } } } }
});

// 日志面板
const logPanel = document.getElementById('log-panel');
function addLog(type, message) {
    const div = document.createElement('div');
    div.className = 'mb-1';
    const time = new Date().toLocaleTimeString();
    let color = '#d4d4d4';
    if (type === 'progress') color = '#4fc1ff';
    if (type === 'status') color = '#b5cea8';
    if (type === 'early_stop') color = '#ce9178';
    if (type === 'error') color = '#f44747';
    if (type === 'inference') color = '#c586c0';
    div.innerHTML = `<span style="color:#858585">[${time}]</span> <span style="color:${color}">${escapeHtml(message)}</span>`;
    logPanel.appendChild(div);
    logPanel.scrollTop = logPanel.scrollHeight;
    // 限制行数
    while (logPanel.children.length > 200) {
        logPanel.removeChild(logPanel.firstChild);
    }
}

// DeepSeek 数据集生成（使用 fetch 读取 SSE 流，因为需要 POST 参数）
document.getElementById('generate-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const apiKey = document.getElementById('ds-api-key').value.trim();
    const useRef = document.getElementById('ds-use-ref').checked;
    const topic = document.getElementById('ds-topic').value.trim();
    const style = document.getElementById('ds-style').value.trim();
    const totalChars = parseInt(document.getElementById('ds-chars').value);
    const temperature = parseFloat(document.getElementById('ds-temp').value);

    if (!apiKey) { alert('请输入 DeepSeek API Key'); return; }
    if (!topic) { alert('请输入生成主题'); return; }

    const statusDiv = document.getElementById('generate-status');
    statusDiv.textContent = '正在调用 DeepSeek API，请稍候...';
    statusDiv.className = 'mt-2 text-info';
    addLog('status', `[DeepSeek] ${useRef ? '[基于当前数据集]' : ''}开始生成: ${topic}`);

    try {
        const resp = await fetch('/api/dataset/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ api_key: apiKey, topic, style, total_chars: totalChars, temperature, use_reference: useRef })
        });
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let done = false;
        while (!done) {
            const { value, done: d } = await reader.read();
            done = d;
            if (value) {
                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n\n');
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.type === 'status') {
                                statusDiv.textContent = data.message;
                                addLog('status', `[DeepSeek] ${data.message}`);
                            } else if (data.type === 'done') {
                                statusDiv.textContent = `生成完成！${data.filename} (${data.chars} 字符)`;
                                statusDiv.className = 'mt-2 text-success';
                                addLog('status', `[DeepSeek] 生成完成: ${data.filename} (${data.chars} 字符)`);
                                // 刷新数据集信息
                                fetch('/api/dataset/info')
                                    .then(r => r.json())
                                    .then(d => {
                                        document.getElementById('dataset-info').innerHTML = `
                                            <p><strong>词汇表:</strong> ${d.vocab_size}</p>
                                            <p><strong>总字符:</strong> ${d.total_chars}</p>
                                            <p><strong>预览:</strong> ${escapeHtml(d.sample.substring(0, 150))}...</p>
                                        `;
                                    });
                            } else if (data.type === 'error') {
                                statusDiv.textContent = '错误: ' + data.message;
                                statusDiv.className = 'mt-2 text-danger';
                                addLog('error', `[DeepSeek] ${data.message}`);
                            }
                        } catch (e) { /* ignore parse errors */ }
                    }
                }
            }
        }
    } catch (err) {
        statusDiv.textContent = '请求失败: ' + err.message;
        statusDiv.className = 'mt-2 text-danger';
        addLog('error', `[DeepSeek] 请求失败: ${err.message}`);
    }
});

// 训练表单提交
document.getElementById('train-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const params = Object.fromEntries(formData.entries());
    for (let k in params) {
        if (params[k] === 'cosine' || params[k] === 'none') continue;
        params[k] = parseFloat(params[k]);
    }
    // 清空图表
    [lossChart, lrChart, pplChart, speedChart].forEach(c => {
        c.data.labels = [];
        c.data.datasets.forEach(d => d.data = []);
        c.update();
    });
    addLog('status', '启动训练...');
    const res = await fetch('/api/train/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
    });
    const data = await res.json();
    if (data.success) {
        startTrainingStream();
    } else {
        addLog('error', '启动失败：' + (data.error || '未知错误'));
        alert('启动失败：' + (data.error || '未知错误'));
    }
});

document.getElementById('stop-train').addEventListener('click', async () => {
    await fetch('/api/train/stop', { method: 'POST' });
    addLog('status', '已发送停止信号');
});

document.getElementById('save-model').addEventListener('click', async () => {
    const res = await fetch('/api/model/save', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: 'models/manual_save.pt' }) });
    const data = await res.json();
    addLog('status', data.success ? '模型已保存到 models/manual_save.pt' : '保存失败');
    alert(data.success ? '模型已保存' : '保存失败');
    refreshModelList();
});

document.getElementById('load-model').addEventListener('click', async () => {
    const res = await fetch('/api/model/load', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: 'models/best_model.pt' }) });
    const data = await res.json();
    addLog('status', data.success ? '最佳模型加载成功' : '加载失败，请先训练');
    alert(data.success ? '模型加载成功' : '加载失败，请先训练');
});

// 模型列表与下载
async function refreshModelList() {
    const res = await fetch('/api/model/list');
    const data = await res.json();
    const container = document.getElementById('model-list');
    if (!data.success || !data.files.length) {
        container.innerHTML = '<div class="text-muted">暂无模型文件</div>';
        return;
    }
    container.innerHTML = data.files.map(f => {
        const sizeMB = (f.size / 1024 / 1024).toFixed(2);
        const date = new Date(f.mtime * 1000).toLocaleString();
        return `<div class="d-flex justify-content-between align-items-center border-bottom py-1">
            <div style="font-size:0.85rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:140px;" title="${escapeHtml(f.name)}">${escapeHtml(f.name)}</div>
            <div class="d-flex gap-1">
                <span class="badge bg-secondary" style="font-size:0.7rem">${sizeMB}MB</span>
                <a href="/api/model/download/${encodeURIComponent(f.name)}" class="btn btn-outline-success btn-sm" style="font-size:0.7rem; padding:2px 6px">下载</a>
            </div>
        </div>`;
    }).join('');
}
document.getElementById('refresh-models').addEventListener('click', refreshModelList);
refreshModelList();

let eventSource = null;
function startTrainingStream() {
    if (eventSource) { eventSource.close(); eventSource = null; }
    // 清空日志面板避免重连后重复堆积
    const logPanel = document.getElementById('log-panel');
    logPanel.innerHTML = '';
    eventSource = new EventSource('/api/train/stream');
    eventSource.onopen = () => {
        addLog('status', 'SSE 连接已建立');
    };
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'progress') {
            lossChart.data.labels.push(data.step);
            lossChart.data.datasets[0].data.push(data.train_loss);
            lossChart.data.datasets[1].data.push(data.val_loss);
            lossChart.update();

            lrChart.data.labels.push(data.step);
            lrChart.data.datasets[0].data.push(data.lr);
            lrChart.update();

            pplChart.data.labels.push(data.step);
            pplChart.data.datasets[0].data.push(data.ppl);
            pplChart.update();

            speedChart.data.labels.push(data.step);
            speedChart.data.datasets[0].data.push(data.tokens_per_sec);
            speedChart.update();

            const sampleDiv = document.getElementById('sample-box');
            sampleDiv.innerHTML = `<strong>Step ${data.step} | Train ${data.train_loss.toFixed(4)} | Val ${data.val_loss.toFixed(4)} | PPL ${data.ppl.toFixed(2)} | LR ${data.lr.toExponential(2)} | GradNorm ${data.grad_norm?.toFixed(3) || '-'} | Speed ${data.tokens_per_sec?.toFixed(1) || '-'} tok/s</strong><br><pre style="margin-top:6px">${escapeHtml(data.sample.substring(0, 500))}</pre>`;

            addLog('progress', `Step ${data.step}: loss=${data.train_loss.toFixed(4)} val=${data.val_loss.toFixed(4)} ppl=${data.ppl.toFixed(2)} lr=${data.lr.toExponential(2)} speed=${data.tokens_per_sec?.toFixed(1) || '-'} tok/s`);
        } else if (data.type === 'status' || data.type === 'early_stop') {
            addLog(data.type === 'early_stop' ? 'early_stop' : 'status', data.message);
            if (data.type === 'end' || data.type === 'early_stop') {
                eventSource.close();
                eventSource = null;
                alert('训练已结束或早停');
            }
        } else if (data.type === 'inference_log') {
            const p = data.params || {};
            addLog('inference', `[推理] prompt=${data.prompt_preview?.substring(0,40)}... response=${data.response?.substring(0,60)}... temp=${p.temperature} top_k=${p.top_k} top_p=${p.top_p} in=${data.input_tokens} out=${data.output_tokens}`);
        } else if (data.type === 'heartbeat') {
            // ignore
        }
    };
    eventSource.onerror = () => {
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
        addLog('error', 'SSE 连接断开，停止重连');
    };
}

// 实战测试场
let chatHistory = [];

document.getElementById('send-btn').addEventListener('click', async () => {
    const inputBox = document.getElementById('chat-input');
    const userMsg = inputBox.value.trim();
    if (!userMsg) return;
    appendMessage('user', userMsg);
    inputBox.value = '';

    const temperature = parseFloat(document.getElementById('chat-temp').value);
    const top_k = parseInt(document.getElementById('chat-topk').value);
    const top_p = parseFloat(document.getElementById('chat-topp').value);
    const max_new = parseInt(document.getElementById('chat-maxlen').value);

    // 构建上下文（保留最近 10 轮对话）
    let context = chatHistory.slice(-10).map(m => m.role === 'user' ? `用户: ${m.content}` : `模型: ${m.content}`).join('\n');
    context += `\n用户: ${userMsg}\n模型: `;

    addLog('inference', `[请求] 用户: ${userMsg.substring(0, 40)} temp=${temperature} top_k=${top_k} top_p=${top_p} max_new=${max_new}`);

    const res = await fetch('/api/chat/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: context, max_new, temperature, top_k, top_p })
    });
    const data = await res.json();
    const botMsg = data.response;
    appendMessage('bot', botMsg);
    chatHistory.push({ role: 'user', content: userMsg });
    chatHistory.push({ role: 'bot', content: botMsg });
});

function appendMessage(role, text) {
    const historyDiv = document.getElementById('chat-history');
    const msgDiv = document.createElement('div');
    msgDiv.className = `alert alert-${role === 'user' ? 'secondary' : 'primary'} mt-1`;
    msgDiv.innerHTML = `<strong>${role === 'user' ? '你' : '模型'}:</strong> ${escapeHtml(text)}`;
    historyDiv.appendChild(msgDiv);
    historyDiv.scrollTop = historyDiv.scrollHeight;
}

function escapeHtml(text) {
    if (!text) return '';
    const map = { '&': '&', '<': '<', '>': '>', '"': '"', "'": '&#039;' };
    return text.replace(/[&<>"']/g, m => map[m]);
}

document.getElementById('clear-chat').addEventListener('click', () => {
    document.getElementById('chat-history').innerHTML = '<div class="text-muted">对话已清空。</div>';
    chatHistory = [];
});

// 页面加载时如果有训练正在进行，自动连接 SSE
fetch('/api/train/status')
    .then(r => r.json())
    .then(s => {
        if (s.is_training) startTrainingStream();
    });
