

import os
import sys
import json
import sqlite3
from typing import Optional
from datetime import datetime, timedelta
from dateutil import parser
import urllib.request
import urllib.error
import ssl
import requests
import re


def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((str(text) + '\n').encode('utf-8'))
        sys.stdout.flush()


# safe_print 定义在文件顶部，保持该实现用于所有输出

from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from langchain_openai import ChatOpenAI

ssl._create_default_https_context = ssl._create_unverified_context
# =========================================
# 基础配置（阿里云 Kimi）
# =========================================

llm = ChatOpenAI(
    model="deepseek-v3",
    openai_api_key="sk-49141e05df7f4584966fac0f8cddbb7d",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,
    timeout=60
)

# 后端 API 基本配置（可通过环境变量覆盖）并规范 URL（自动补全 scheme）
BACKEND_API_BASE = os.getenv('BACKEND_API_BASE', '115.190.155.26:8000')
# 如果用户只提供了主机:端口形式（例如 115.190.155.26:8000），补全 http:// 前缀
if BACKEND_API_BASE and not BACKEND_API_BASE.startswith(('http://', 'https://')):
    BACKEND_API_BASE = 'http://' + BACKEND_API_BASE
API_PREFIX = BACKEND_API_BASE.rstrip('/') + '/api/v1'

# =========================================
# 数据库初始化
# =========================================

conn = sqlite3.connect("wanji.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS schedule (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    category TEXT,
    start_time INTEGER,
    end_time INTEGER,
    created_at INTEGER
)
""")

# 索引以加速时间范围查询
cursor.execute("CREATE INDEX IF NOT EXISTS idx_schedule_start ON schedule(start_time)")

conn.commit()


# =========================================
# 工具函数
# =========================================

def detect_conflict(start_time, end_time):
    # 接受 epoch(int) 或可解析的时间字符串
    s = to_epoch(start_time)
    e = to_epoch(end_time)
    cursor.execute("""
    SELECT title, start_time, end_time FROM schedule
    WHERE NOT (end_time <= ? OR start_time >= ?)
    """, (s, e))
    return cursor.fetchall()


def find_next_available_slot(start_time, duration_minutes=60):
    start_epoch = to_epoch(start_time)
    start = datetime.fromtimestamp(start_epoch)
    for i in range(1, 6):
        new_start = start + timedelta(hours=i)
        new_end = new_start + timedelta(minutes=duration_minutes)
        ns_epoch = int(new_start.timestamp())
        ne_epoch = int(new_end.timestamp())
        if not detect_conflict(ns_epoch, ne_epoch):
            return ns_epoch, ne_epoch
    return None, None


def _find_event_by_title(title: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
    """在后端查询匹配 title 的事件，返回第一个匹配的 event dict 或 None。"""
    params = {}
    if start_date:
        params['start_date'] = start_date
    if end_date:
        params['end_date'] = end_date
    params['size'] = 200
    url = f"{API_PREFIX}/events"
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        items = data.get('items', []) if isinstance(data, dict) else []
        for it in items:
            if it.get('title') == title:
                return it
        for it in items:
            if it.get('title', '').strip().lower() == title.strip().lower():
                return it
    except Exception:
        return None
    return None


# =========================================
# LangChain Tools
# =========================================

@tool(description="用于添加日程。当用户以自然语言请求安排事件时使用。例：'帮我在3月10日14:00到15:00安排医生看诊，标题：看医生'。接受标题、开始/结束时间、可选类别。返回操作结果。")
def add_schedule(data) -> str:
    """
    添加日程。输入必须是JSON字符串:
    {
      "title": "",
      "category": "",
      "start_time": "YYYY-MM-DD HH:MM",
      "end_time": "YYYY-MM-DD HH:MM"
    }
    """
    safe_print(f"[DEBUG_TOOL add_schedule] called with: {data}")
    # 支持 dict 或 JSON 字符串输入
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            return "传入数据不是合法的 JSON，也不是结构化 dict。"

    title = data.get("title")
    if not title:
        return "缺少 title 字段"
    category = data.get("category", "general")
    # 将时间转换为 ISO 格式，后端期望 ISO datetime
    try:
        start_iso = datetime.fromtimestamp(to_epoch(data.get("start_time"))).isoformat()
        end_iso = datetime.fromtimestamp(to_epoch(data.get("end_time"))).isoformat()
    except Exception as e:
        return f"时间解析错误: {e}"

    payload = {
        "title": title,
        "start_time": start_iso,
        "end_time": end_iso,
        "type": data.get("category", None)
    }
    try:
        url = f"{API_PREFIX}/events"
        safe_print(f"[DEBUG_TOOL add_schedule] POST {url} payload={payload}")
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code in (200, 201):
            return "日程添加成功"
        elif resp.status_code == 501:
            # 后端未实现该接口，尝试回退到 agent/process
            safe_print(f"[DEBUG_TOOL add_schedule] 后端返回501，尝试回退到 agent/process: {resp.text}")
            try:
                agent_url = f"{API_PREFIX}/agent/process"
                agent_payload = {"text": f"创建日程：{payload.get('title')} 从 {payload.get('start_time')} 到 {payload.get('end_time')}"}
                safe_print(f"[DEBUG_TOOL add_schedule] POST {agent_url} payload={agent_payload}")
                aresp = requests.post(agent_url, json=agent_payload, timeout=10)
                if aresp.status_code == 200:
                    try:
                        data = aresp.json()
                        return data.get('reply') or '通过 agent/process 创建请求已发送'
                    except Exception:
                        return '通过 agent/process 创建请求已发送，未解析返回'
                else:
                    return f"回退到 agent/process 失败: {aresp.status_code} {aresp.text}"
            except Exception as e:
                return f"回退到 agent/process 时出错: {e}"
        else:
            return f"后端返回错误: {resp.status_code} {resp.text}"
    except Exception as e:
        return f"请求后端创建日程失败: {e}"


@tool(description="用于删除日程。当用户以自然语言要求删除事件时使用。例：'删除3月10日14:00的看医生日程' 或只提供标题 '删除 看医生'。返回操作结果。")
def delete_schedule(title: str) -> str:
    """删除指定标题的日程（通过后端）"""
    safe_print(f"[DEBUG_TOOL delete_schedule] called with title: {title}")
    try:
        ev = _find_event_by_title(title)
        if not ev:
            return "未找到匹配的日程"
        server_id = ev.get('serverId') or ev.get('server_id')
        if not server_id:
            return "未找到事件 id，无法删除"
        url = f"{API_PREFIX}/events/{server_id}"
        resp = requests.delete(url, timeout=10)
        if resp.status_code in (200, 204):
            return "删除成功"
        else:
            return f"后端删除失败: {resp.status_code} {resp.text}"
    except Exception as e:
        return f"请求后端删除日程失败: {e}"


@tool(description="用于查询日程。当用户询问某天或某段时间的安排时使用。例：'查询3月10日的日程' 或 '下周有什么安排'。返回可读列表。")
def query_schedule(date: str) -> str:
    """查询某天的日程，输入日期"""
    safe_print(f"[DEBUG_TOOL query_schedule] called with date: {date}")
    d = parser.parse(date)
    start = d.replace(hour=0, minute=0, second=0)
    end = start + timedelta(days=1)
    start_iso = start.isoformat()
    end_iso = end.isoformat()
    try:
        url = f"{API_PREFIX}/events"
        resp = requests.get(url, params={"start_date": start_iso, "end_date": end_iso}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        items = data.get('items', []) if isinstance(data, dict) else []
        if not items:
            return "当天没有安排"
        lines = []
        for it in items:
            lines.append(f"{it.get('title')}: {it.get('start_time')} - {it.get('end_time')}")
        return "\n".join(lines)
    except Exception as e:
        return f"查询后端日程失败: {e}"


@tool(description="用于修改日程时间或详情。当用户以自然语言要求更改事件时使用。例：'把看医生从14:00改到15:00' 或 '把看医生改到3月11日16:00'。接受标题与新时间。返回操作结果。")
def update_schedule(data) -> str:
    """
    修改日程时间。
    输入JSON:
    {
      "title": "",
      "new_start_time": "",
      "new_end_time": ""
    }
    """

    safe_print(f"[DEBUG_TOOL update_schedule] called with: {data}")
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            return "传入数据不是合法的 JSON，也不是结构化 dict。"

    title = data.get("title")
    if not title:
        return "缺少 title 字段"

    try:
        new_start_iso = datetime.fromtimestamp(to_epoch(data.get("new_start_time"))).isoformat()
        new_end_iso = datetime.fromtimestamp(to_epoch(data.get("new_end_time"))).isoformat()
    except Exception as e:
        return f"时间解析错误: {e}"

    # 查找事件并调用后端更新
    try:
        ev = _find_event_by_title(title)
        if not ev:
            return "未找到匹配的日程"
        server_id = ev.get('serverId') or ev.get('server_id')
        if not server_id:
            return "未找到事件 id，无法修改"
        payload = ev.copy()
        payload['start_time'] = new_start_iso
        payload['end_time'] = new_end_iso
        url = f"{API_PREFIX}/events/{server_id}"
        resp = requests.put(url, json=payload, timeout=10)
        if resp.status_code == 200:
            return "修改成功"
        else:
            return f"后端更新失败: {resp.status_code} {resp.text}"
    except Exception as e:
        return f"请求后端修改日程失败: {e}"


@tool(description="用于统计类问题。当用户询问历史或汇总信息时使用。例：'我上周有多少次看医生' 或 '共有多少条日程'。返回统计结果。")
def statistics(query: str) -> str:
    safe_print(f"[DEBUG_TOOL statistics] called with: {query}")
    """统计类问题，例如：我上周跑了几次步"""
    if "上周" in query or "last week" in query:
        today = datetime.now()
        start = today - timedelta(days=today.weekday() + 7)
        end = start + timedelta(days=7)
        start_epoch = int(start.timestamp())
        end_epoch = int(end.timestamp())

        cursor.execute("""
        SELECT COUNT(*) FROM schedule
        WHERE start_time BETWEEN ? AND ?
        """, (start_epoch, end_epoch))

        count = cursor.fetchone()[0]
        return f"上周共有 {count} 次安排"

    cursor.execute("SELECT COUNT(*) FROM schedule")
    count = cursor.fetchone()[0]
    return f"目前共有 {count} 条日程"


# =========================================
# 记忆系统
# =========================================

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# =========================================
# 初始化智能体：万机
# =========================================

system_prompt = """
你是一个智能日程助手，名字叫“万机”。

你的能力：
- 理解中英文自然语言
- 自动解析时间
- 必须使用工具管理日程
- 遇到冲突给出建议
- 可以统计历史行为
- 不要让用户输入结构化JSON
- 自动选择正确工具

你必须使用工具完成数据库操作。
"""

# 强化提示：提醒模型始终以调用工具（function call）的形式完成增删改查
system_prompt += """
重要要求：当用户请求创建、删除、修改或查询日程时，必须调用相应工具而不是直接返回普通文本结果。
工具名称：add_schedule, delete_schedule, update_schedule, query_schedule, statistics。
示例用户话术：
- 帮我在3月10日14:00到15:00安排医生看诊，标题：看医生
- 删除3月10日14:00的看医生日程
- 把看医生从14:00改到15:00
- 查询3月10日的日程
不要要求用户提供 JSON；从自然语言中解析字段并以 function_call 的形式调用工具。
"""

agent = initialize_agent(
    tools=[
        add_schedule,
        delete_schedule,
        update_schedule,
        query_schedule,
        statistics
    ],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=memory,
    verbose=True,
    agent_kwargs={"system_message": system_prompt}
)

# 调试：列出已注册的工具与描述，帮助确认模型能看到这些工具
try:
    safe_print("已注册工具：")
    for t in agent.tools:
        try:
            desc = getattr(t, 'description', '')
        except Exception:
            desc = ''
        safe_print(f"- {getattr(t,'name',str(t))}: {desc}")
except Exception as ex:
    safe_print(f"[DEBUG] 列出 agent.tools 时发生异常: {ex}")

try:
    safe_print(f"[DEBUG] agent 类型: {type(agent)}")
    attrs = [a for a in dir(agent) if not a.startswith('_')]
    safe_print(f"[DEBUG] agent 可见属性: {', '.join(attrs[:50])}")
except Exception as ex:
    safe_print(f"[DEBUG] 无法读取 agent 属性: {ex}")

for name in ["add_schedule", "delete_schedule", "update_schedule", "query_schedule", "statistics"]:
    try:
        obj = globals().get(name)
        safe_print(f"[DEBUG_FUNC] {name}: type={type(obj)}, has_description={hasattr(obj,'description')}, description={getattr(obj,'description', None)}")
    except Exception as ex:
        safe_print(f"[DEBUG_FUNC] {name} 获取信息失败: {ex}")

# =========================================
# 主循环
# =========================================


def to_epoch(value):
    """将多种时间表示转换为 epoch 秒整数。支持 int, datetime, ISO 字符串或自然语言时间字符串。"""
    if value is None:
        raise ValueError("时间值为空")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, datetime):
        return int(value.timestamp())
    if isinstance(value, str):
        # 支持直接是数字字符串
        if value.isdigit():
            return int(value)
        dt = parser.parse(value)
        return int(dt.timestamp())
    raise ValueError(f"无法解析的时间类型: {type(value)}")


def epoch_to_str(ts):
    try:
        return datetime.fromtimestamp(int(ts)).isoformat()
    except Exception:
        return str(ts)
    

safe_print("智能体 万机 已启动（完整版）")

while True:
    user_input = input("你：")
    try:
        response = agent.run(user_input)
        # 如果模型没有返回内容（未触发工具），使用本地 NL 路由器作为后备
        if not (response and str(response).strip()):
            safe_print("[DEBUG] 模型返回空响应，尝试使用本地 NL 路由器解析并调用工具...")
            try:
                # 本地 NL 路由器函数
                def nl_router(text: str):
                    safe_print(f"[DEBUG_NL] 解析用户输入: {text}")
                    title = None
                    m = re.search(r"标题[:：]\s*([^，,]+)", text)
                    if m:
                        title = m.group(1).strip()

                    time_pattern = r"\d{1,4}年?\s*\d{1,2}月\s*\d{1,2}日\s*\d{1,2}[:：]\d{2}|\d{1,2}[:：]\d{2}"
                    times = re.findall(time_pattern, text)
                    safe_print(f"[DEBUG_NL] 抽取到时间片段: {times}")

                    # 添加日程意图
                    if "安排" in text or "帮我" in text:
                        if not title:
                            m2 = re.search(r"安排([^，,]+)", text)
                            if m2:
                                possible = m2.group(1)
                                possible = re.sub(time_pattern, "", possible)
                                title = possible.strip(' ，,')
                        start = None
                        end = None
                        try:
                            if len(times) >= 2:
                                start = parser.parse(times[0], fuzzy=True)
                                end = parser.parse(times[1], fuzzy=True)
                            elif len(times) == 1:
                                if "到" in text or "-" in text:
                                    part = text.split('到') if '到' in text else text.split('-')
                                    if len(part) >= 2:
                                        start = parser.parse(times[0], fuzzy=True)
                                        second = re.search(r"(\d{1,2}[:：]\d{2})", part[1])
                                        if second:
                                            end = parser.parse(second.group(1), fuzzy=True)
                            if start and end and title:
                                payload = {
                                    "title": title,
                                    "start_time": start.isoformat(),
                                    "end_time": end.isoformat(),
                                    "category": None
                                }
                                safe_print(f"[DEBUG_NL] 调用 add_schedule with {payload}")
                                return add_schedule({"data": payload})
                        except Exception as e:
                            safe_print(f"[DEBUG_NL] 解析时间失败: {e}")

                    # 查询意图
                    if "查询" in text or "查" in text:
                        mdate = re.search(r"\d{1,4}年?\s*\d{1,2}月\s*\d{1,2}日|\d{4}-\d{1,2}-\d{1,2}", text)
                        if mdate:
                            date_str = mdate.group(0)
                            safe_print(f"[DEBUG_NL] 调用 query_schedule with {date_str}")
                            return query_schedule({"date": date_str})
                        else:
                            return query_schedule(datetime.now().isoformat())

                    # 删除意图
                    if "删除" in text or "移除" in text:
                        if title:
                            safe_print(f"[DEBUG_NL] 调用 delete_schedule with title={title}")
                            return delete_schedule(title)
                        m3 = re.search(r"删除([^，,]+)", text)
                        if m3:
                            t = m3.group(1).strip(' ，,')
                            safe_print(f"[DEBUG_NL] 调用 delete_schedule with title={t}")
                            return delete_schedule({"title": t})

                    # 修改意图
                    if "改" in text or "修改" in text:
                        if not title:
                            m4 = re.search(r"把([^从]+)从", text)
                            if m4:
                                title = m4.group(1).strip()
                        if title and len(times) >= 2:
                            new_start = times[0]
                            new_end = times[1]
                            payload = {"title": title, "new_start_time": new_start, "new_end_time": new_end}
                            safe_print(f"[DEBUG_NL] 调用 update_schedule with {payload}")
                            return update_schedule({"data": payload})

                    return None

                nl_result = nl_router(user_input)
                if nl_result is not None:
                    safe_print("万机：" + str(nl_result))
                else:
                    safe_print("万机：" + str(response))
            except Exception as e:
                safe_print("本地 NL 路由器执行出错: " + str(e))
        else:
            safe_print("万机：" + str(response))
    except Exception as e:
        safe_print("网络或 API 连接错误，无法联系模型服务。错误：" + str(e))
        continue