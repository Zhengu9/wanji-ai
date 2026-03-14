

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
        sys.stdout.buffer.write((str(text) + "\n").encode('utf-8'))
        sys.stdout.flush()


# LangChain imports with fallback for `tool` decorator
try:
    from langchain.agents import initialize_agent, AgentType
    from langchain.memory import ConversationBufferMemory
    try:
        from langchain.tools import tool
    except Exception:
        # 回退定义：使 @tool(description=...) 可用，但不执行任何特殊注册
        def tool(*dargs, **dkwargs):
            def _decorator(func):
                desc = dkwargs.get('description') if dkwargs else None
                try:
                    setattr(func, 'description', desc)
                except Exception:
                    pass
                return func
            return _decorator
    from langchain_openai import ChatOpenAI
except Exception:
    # 如果 langchain 不可用，仍然允许脚本以非 agent 方式运行（只要本地工具可用）
    def tool(*dargs, **dkwargs):
        def _decorator(func):
            try:
                setattr(func, 'description', dkwargs.get('description') if dkwargs else None)
            except Exception:
                pass
            return func
        return _decorator
    # 定义占位符类/函数，避免后续 NameError，但 agent 初始化 会失败若 langchain 缺失
    initialize_agent = None
    AgentType = None
    ConversationBufferMemory = None
    ChatOpenAI = None


ssl._create_default_https_context = ssl._create_unverified_context
# =========================================
# 基础配置（阿里云 Kimi）
# =========================================

llm = None
try:
    llm = ChatOpenAI(
        model="deepseek-v3",
        openai_api_key="sk-49141e05df7f4584966fac0f8cddbb7d",
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
        timeout=60
    )
except Exception:
    llm = None

# 后端 API 基本配置（可通过环境变量覆盖）并规范 URL（自动补全 scheme）
BACKEND_API_BASE = os.getenv('BACKEND_API_BASE', '115.190.155.26:8000')
if BACKEND_API_BASE and not BACKEND_API_BASE.startswith(('http://', 'https://')):
    BACKEND_API_BASE = 'http://' + BACKEND_API_BASE
API_PREFIX = BACKEND_API_BASE.rstrip('/') + '/api/v1'

# =========================================
# 本地 SQLite 数据库初始化
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
cursor.execute("CREATE INDEX IF NOT EXISTS idx_schedule_start ON schedule(start_time)")
conn.commit()

# 对话记忆表：存储 role(user/assistant) 与文本、时间戳，便于回放上下文
cursor.execute("""
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT,
    text TEXT,
    created_at INTEGER
)
""")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_created ON conversations(created_at)")
conn.commit()


# =========================================
# 工具与解析函数
# =========================================

def to_epoch(value):
    if value is None:
        raise ValueError("时间值为空")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, datetime):
        return int(value.timestamp())
    if isinstance(value, str):
        if value.isdigit():
            return int(value)
        dt = parser.parse(value)
        return int(dt.timestamp())
    raise ValueError(f"无法解析的时间类型: {type(value)}")


def epoch_to_str(ts):
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def detect_conflict(start_time, end_time):
    s = to_epoch(start_time)
    e = to_epoch(end_time)
    cursor.execute("""
    SELECT title, start_time, end_time FROM schedule
    WHERE NOT (end_time <= ? OR start_time >= ?)
    """, (s, e))
    return cursor.fetchall()


def detect_conflict_excluding(start_time, end_time, exclude_id: Optional[int] = None):
    s = to_epoch(start_time)
    e = to_epoch(end_time)
    if exclude_id is None:
        cursor.execute("""
        SELECT id, title, start_time, end_time FROM schedule
        WHERE NOT (end_time <= ? OR start_time >= ?)
        """, (s, e))
    else:
        cursor.execute("""
        SELECT id, title, start_time, end_time FROM schedule
        WHERE NOT (end_time <= ? OR start_time >= ?) AND id != ?
        """, (s, e, int(exclude_id)))
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


def save_message(role: str, text: str):
    try:
        ts = int(datetime.now().timestamp())
        cursor.execute("INSERT INTO conversations (role, text, created_at) VALUES (?, ?, ?)", (role, str(text), ts))
        conn.commit()
    except Exception:
        pass


def load_recent_conversation(limit: int = 20) -> str:
    try:
        cursor.execute("SELECT role, text FROM conversations ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        # 按时间正序返回（最近的在后面）
        rows = list(reversed(rows))
        parts = []
        for r in rows:
            parts.append(f"{r[0]}: {r[1]}")
        return "\n".join(parts)
    except Exception:
        return ""


def _normalize_possible_json_input(val):
    """如果输入是 JSON 字符串（例如 '{"date":"2026-03-14"}'），尝试解析并返回内部值。
    否则返回原始字符串（去除多余引号与空白）。"""
    if val is None:
        return None
    # 若为 dict 直接返回
    if isinstance(val, dict):
        return val
    if not isinstance(val, str):
        return val
    s = val.strip()
    # 去掉外层多余的引号
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    # 如果看起来像 JSON 对象，尝试解析
    if s.startswith('{') and s.endswith('}'):
        try:
            j = json.loads(s)
            return j
        except Exception:
            # 可能是部分格式化字符串，继续后续处理
            pass
    return s


def _find_event_by_title(title: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
    try:
        t = title
        if isinstance(title, dict):
            t = title.get('title') or title.get('text')
        if not t:
            return None
        params_start = None
        params_end = None
        if start_date:
            try:
                params_start = to_epoch(start_date)
            except Exception:
                params_start = None
        if end_date:
            try:
                params_end = to_epoch(end_date)
            except Exception:
                params_end = None

        if params_start and params_end:
            cursor.execute("""
            SELECT id, title, category, start_time, end_time FROM schedule
            WHERE title = ? AND start_time BETWEEN ? AND ?
            """, (t, params_start, params_end))
        else:
            cursor.execute("""
            SELECT id, title, category, start_time, end_time FROM schedule
            WHERE title = ?
            """, (t,))
        row = cursor.fetchone()
        if not row:
            if params_start and params_end:
                cursor.execute("""
                SELECT id, title, category, start_time, end_time FROM schedule
                WHERE LOWER(title) = LOWER(?) AND start_time BETWEEN ? AND ?
                """, (t, params_start, params_end))
            else:
                cursor.execute("""
                SELECT id, title, category, start_time, end_time FROM schedule
                WHERE LOWER(title) = LOWER(?)
                """, (t,))
            row = cursor.fetchone()
        if not row:
            return None
        eid, etitle, ecat, estart, eend = row
        return {
            'serverId': str(eid),
            'server_id': str(eid),
            'title': etitle,
            'type': ecat,
            'start_time': epoch_to_str(estart),
            'end_time': epoch_to_str(eend)
        }
    except Exception:
        return None


def get_reference_date_from_text(text: str) -> datetime:
    today = datetime.now()
    if not text:
        return today
    # 处理更丰富的相对日期词：大后天/大前天/N天后/N天前 等
    if "今天" in text:
        return today
    if "明天" in text:
        return today + timedelta(days=1)
    if "后天" in text:
        return today + timedelta(days=2)
    if "大后天" in text:
        return today + timedelta(days=3)
    if "大前天" in text:
        return today - timedelta(days=3)
    # 匹配类似 '3天后' 或 '2天前'
    m_rel_after = re.search(r"(\d+)\s*天后", text)
    if m_rel_after:
        try:
            d = int(m_rel_after.group(1))
            return today + timedelta(days=d)
        except Exception:
            pass
    m_rel_before = re.search(r"(\d+)\s*天前", text)
    if m_rel_before:
        try:
            d = int(m_rel_before.group(1))
            return today - timedelta(days=d)
        except Exception:
            pass
    if "昨天" in text:
        return today - timedelta(days=1)
    m = re.search(r'周([一二三四五六日天])', text)
    if m:
        mapping = {'一':0,'二':1,'三':2,'四':3,'五':4,'六':5,'日':6,'天':6}
        target = mapping.get(m.group(1), None)
        if target is not None:
            today_wd = today.weekday()
            days = (target - today_wd) % 7
            if days == 0:
                days = 7
            return today + timedelta(days=days)
    return today


def parse_nl_time(timestr: str, reference_date: Optional[datetime] = None) -> datetime:
    s = str(timestr).strip()
    if reference_date is None:
        reference_date = datetime.now()
    s = s.replace('点', ':00').replace('：', ':')
    is_pm = False
    if '下午' in s or '晚上' in s or '晚' in s:
        is_pm = True
        s = re.sub(r'下午|晚上|晚', '', s)
    s = s.strip()
    try:
        if re.search(r'\d{4}[-年]', s) or re.search(r'\d{1,2}月', s) or ('-' in s and len(s.split('-')[0])==4):
            return parser.parse(s)
        default_dt = reference_date.replace(hour=0, minute=0, second=0, microsecond=0)
        dt = parser.parse(s, default=default_dt)
        if is_pm and dt.hour < 12:
            dt = dt + timedelta(hours=12)
        return dt
    except Exception:
        combined = f"{reference_date.date().isoformat()} {s}"
        return parser.parse(combined)


def parse_explicit_date_str(s: str) -> Optional[datetime]:
    """解析显式日期字符串（支持中文格式如 '3月22日'、'2026年3月22日'、'2026-03-22'），
    返回 datetime（时间部分为 00:00:00）。如果无法解析则返回 None。
    若用户未指定年份且解析的日期早于今天，会尝试将年份加 1（假定用户指的是下个年周期）。
    """
    if not s:
        return None
    s = s.strip()
    today = datetime.now()
    # yyyy年m月d日
    m = re.match(r"^(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日$", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(y, mo, d)
        except Exception:
            return None
    # m月d日（无年）
    m2 = re.match(r"^(\d{1,2})\s*月\s*(\d{1,2})\s*日$", s)
    if m2:
        mo, d = int(m2.group(1)), int(m2.group(2))
        y = today.year
        try:
            dt = datetime(y, mo, d)
        except Exception:
            return None
        # 若解析到的日期早于今天，猜测可能是下一年
        if dt.date() < today.date():
            try:
                dt_next = datetime(y + 1, mo, d)
                return dt_next
            except Exception:
                return dt
        return dt
    # yyyy-mm-dd
    m3 = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
    if m3:
        y, mo, d = int(m3.group(1)), int(m3.group(2)), int(m3.group(3))
        try:
            return datetime(y, mo, d)
        except Exception:
            return None
    return None


# =========================================
# LangChain Tools (本地实现：操作 SQLite)
# =========================================

@tool(description="用于添加日程。当用户以自然语言请求安排事件时使用。接受标题、开始/结束时间、可选类别。返回操作结果。")
def add_schedule(data) -> str:
    safe_print(f"[DEBUG_TOOL add_schedule] called with: {data}")
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            return "传入数据不是合法的 JSON，也不是结构化 dict。"
    if isinstance(data, dict) and 'data' in data:
        data = data['data'] or data
    title = data.get("title")
    if not title:
        return "缺少 title 字段"
    category = data.get("category") or data.get('type') or 'general'
    try:
        s_epoch = to_epoch(data.get("start_time"))
        e_epoch = to_epoch(data.get("end_time"))
    except Exception as e:
        return f"时间解析错误: {e}"
    # 冲突检测：若有重叠则返回冲突信息并给出建议时段
    try:
        conflicts = detect_conflict(s_epoch, e_epoch)
        if conflicts:
                lines = []
                for it in conflicts:
                    try:
                        title, st, et = it
                        lines.append(f"{title}: {epoch_to_str(st)} - {epoch_to_str(et)}")
                    except Exception:
                        pass
                next_s, next_e = find_next_available_slot(s_epoch, duration_minutes=int((e_epoch - s_epoch) / 60))
                suggestion = ""
                if next_s and next_e:
                    suggestion = f" 建议空闲时段: {datetime.fromtimestamp(next_s).strftime('%Y-%m-%d %H:%M:%S')} - {datetime.fromtimestamp(next_e).strftime('%Y-%m-%d %H:%M:%S')}"
                return "时间冲突，已存在：\n" + "\n".join(lines) + suggestion
    except Exception:
        pass
    try:
        now_ts = int(datetime.now().timestamp())
        cursor.execute("""
        INSERT INTO schedule (title, category, start_time, end_time, created_at)
        VALUES (?, ?, ?, ?, ?)
        """, (title, category, int(s_epoch), int(e_epoch), now_ts))
        conn.commit()
        return "日程添加成功"
    except Exception as e:
        return f"本地创建日程失败: {e}"


@tool(description="用于删除日程。当用户以自然语言要求删除事件时使用。返回操作结果。")
def delete_schedule(title: str) -> str:
    safe_print(f"[DEBUG_TOOL delete_schedule] called with title: {title}")
    try:
        t = title
        # 兼容：如果收到 JSON 字符串，尝试解析
        norm = _normalize_possible_json_input(title)
        if isinstance(norm, dict):
            t = norm.get('title') or norm.get('text') or t
        elif isinstance(norm, str):
            t = norm
        # 若仍为 dict（直接传入 dict 的情况），提取 title
        if isinstance(t, dict):
            t = t.get('title') or t.get('text')
        ev = _find_event_by_title(t)
        # 如果精确匹配不到，尝试模糊与时间范围匹配以提高删除命中率
        if not ev:
            try:
                # 1) 尝试模糊匹配 title 包含关键字（不区分大小写）
                like_key = f"%{t}%"
                cursor.execute("SELECT id, title, category, start_time, end_time FROM schedule WHERE LOWER(title) LIKE LOWER(?) LIMIT 1", (like_key,))
                row = cursor.fetchone()
                if row:
                    eid, etitle, ecat, estart, eend = row
                    ev = {'serverId': str(eid), 'server_id': str(eid), 'title': etitle, 'type': ecat, 'start_time': epoch_to_str(estart), 'end_time': epoch_to_str(eend)}
            except Exception:
                ev = None
        if not ev:
            # 2) 如果用户输入中包含时间片段（例如 15:00 或 3月15日），尝试按时间范围匹配
            try:
                time_m = re.search(r"(\d{1,4}年\s*\d{1,2}月\s*\d{1,2}日|\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[:：]\d{2})", str(title))
                if time_m:
                    token = time_m.group(0)
                    # 解析为 datetime，优先使用显式日期解析
                    dt = None
                    try:
                        dt = parse_explicit_date_str(token)
                    except Exception:
                        dt = None
                    if not dt:
                        try:
                            dt = parse_nl_time(token, reference_date=datetime.now())
                        except Exception:
                            dt = None
                    if dt:
                        window_s = int((dt - timedelta(hours=1)).timestamp())
                        window_e = int((dt + timedelta(hours=1)).timestamp())
                        cursor.execute("SELECT id, title, category, start_time, end_time FROM schedule WHERE start_time BETWEEN ? AND ? LIMIT 1", (window_s, window_e))
                        row = cursor.fetchone()
                        if row:
                            eid, etitle, ecat, estart, eend = row
                            ev = {'serverId': str(eid), 'server_id': str(eid), 'title': etitle, 'type': ecat, 'start_time': epoch_to_str(estart), 'end_time': epoch_to_str(eend)}
            except Exception:
                ev = None
        if not ev:
            return "未找到匹配的日程"
        server_id = ev.get('serverId') or ev.get('server_id')
        if not server_id:
            return "未找到事件 id，无法删除"
        cursor.execute("DELETE FROM schedule WHERE id = ?", (int(server_id),))
        conn.commit()
        return "删除成功"
    except Exception as e:
        return f"本地删除日程失败: {e}"


@tool(description="用于查询日程。当用户询问某天或某段时间的安排时使用。返回可读列表。")
def query_schedule(date: str) -> str:
    safe_print(f"[DEBUG_TOOL query_schedule] called with date: {date}")
    try:
        norm = _normalize_possible_json_input(date)
        if isinstance(norm, dict) and 'date' in norm:
            date_val = norm['date']
        else:
            date_val = norm
        # 最终确保是字符串
        if isinstance(date_val, (dict, list)):
            date_str = json.dumps(date_val)
        else:
            date_str = str(date_val).strip()
        # 去掉多余的引号
        if (date_str.startswith('"') and date_str.endswith('"')) or (date_str.startswith("'") and date_str.endswith("'")):
            date_str = date_str[1:-1].strip()
        # 直接尝试解析
        try:
            d = parser.parse(date_str)
        except Exception:
            # 备用解析：替换斜杠/, 去掉尾随非数字字符，或提取 YYYY-MM-DD 子串
            ds = date_str.replace('/', '-').strip()
            m = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", ds)
            if m:
                ds = m.group(1)
            try:
                d = parser.parse(ds)
            except Exception as e:
                return f"时间解析错误: {e}"
    except Exception as e:
        return f"时间解析错误: {e}"
    start = d.replace(hour=0, minute=0, second=0)
    end = start + timedelta(days=1)
    s_epoch = int(start.timestamp())
    e_epoch = int(end.timestamp())
    try:
        cursor.execute("SELECT title, start_time, end_time FROM schedule WHERE start_time BETWEEN ? AND ? ORDER BY start_time", (s_epoch, e_epoch))
        rows = cursor.fetchall()
        if not rows:
            return "当天没有安排"
        lines = []
        for it in rows:
            title, st, et = it
            lines.append(f"{title}: {epoch_to_str(st)} - {epoch_to_str(et)}")
        return "\n".join(lines)
    except Exception as e:
        return f"查询本地日程失败: {e}"


@tool(description="用于查询一段时间内的日程，接受 {start: 'YYYY-MM-DD', end: 'YYYY-MM-DD'} 或两个日期字符串。返回按日期分组的可读列表。")
def query_schedule_range(params) -> str:
    safe_print(f"[DEBUG_TOOL query_schedule_range] called with: {params}")
    try:
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except Exception:
                return query_schedule(params)
        if isinstance(params, dict):
            start = params.get('start') or params.get('from') or params.get('date')
            end = params.get('end') or params.get('to')
            if not start and 'days' in params:
                try:
                    days = int(params.get('days'))
                    sd = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    ed = sd + timedelta(days=days)
                    start = sd.date().isoformat()
                    end = ed.date().isoformat()
                except Exception:
                    pass
        else:
            return query_schedule(params)
        if not start:
            return "缺少 start 参数"
        if not end:
            # 如果只给了 start，则视为单日查询
            return query_schedule(str(start))
        # 解析 start/end
        try:
            sd = parser.parse(str(start)).replace(hour=0, minute=0, second=0, microsecond=0)
            ed = parser.parse(str(end)).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        except Exception as e:
            return f"时间解析错误: {e}"
        s_epoch = int(sd.timestamp())
        e_epoch = int(ed.timestamp())
        cursor.execute("SELECT title, start_time, end_time FROM schedule WHERE NOT (end_time <= ? OR start_time >= ?) ORDER BY start_time", (s_epoch, e_epoch))
        rows = cursor.fetchall()
        if not rows:
            return "范围内没有安排"
        # 按日期分组
        grouped = {}
        for title, st, et in rows:
            d = datetime.fromtimestamp(int(st)).date().isoformat()
            grouped.setdefault(d, []).append((title, st, et))
        parts = []
        for d in sorted(grouped.keys()):
            parts.append(f"{d}：")
            for it in grouped[d]:
                parts.append(f"  {it[0]}: {epoch_to_str(it[1])} - {epoch_to_str(it[2])}")
        return "\n".join(parts)
    except Exception as e:
        return f"查询范围日程失败: {e}"


@tool(description="用于修改日程时间或详情。当用户以自然语言要求更改事件时使用。接受标题与新时间。返回操作结果。")
def update_schedule(data) -> str:
    safe_print(f"[DEBUG_TOOL update_schedule] called with: {data}")
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            return "传入数据不是合法的 JSON，也不是结构化 dict。"
    if isinstance(data, dict) and 'data' in data:
        data = data['data'] or data
    title = data.get("title")
    if not title:
        return "缺少 title 字段"
    try:
        new_s = to_epoch(data.get("new_start_time"))
        new_e = to_epoch(data.get("new_end_time"))
    except Exception as e:
        return f"时间解析错误: {e}"
    try:
        ev = _find_event_by_title(title)
        if not ev:
            return "未找到匹配的日程"
        server_id = ev.get('serverId') or ev.get('server_id')
        if not server_id:
            return "未找到事件 id，无法修改"
        # 冲突检测（排除正在修改的事件本身）
        try:
            conflicts = detect_conflict_excluding(new_s, new_e, exclude_id=server_id)
            if conflicts:
                    lines = []
                    for it in conflicts:
                        try:
                            cid, ctitle, cst, cet = it
                            lines.append(f"{ctitle}: {epoch_to_str(cst)} - {epoch_to_str(cet)}")
                        except Exception:
                            pass
                    next_s, next_e = find_next_available_slot(new_s, duration_minutes=int((new_e - new_s) / 60))
                    suggestion = ""
                    if next_s and next_e:
                        suggestion = f" 建议空闲时段: {datetime.fromtimestamp(next_s).strftime('%Y-%m-%d %H:%M:%S')} - {datetime.fromtimestamp(next_e).strftime('%Y-%m-%d %H:%M:%S')}"
                    return "修改后的时间与已有日程冲突，冲突项：\n" + "\n".join(lines) + suggestion
        except Exception:
            pass
        cursor.execute("UPDATE schedule SET start_time = ?, end_time = ? WHERE id = ?", (int(new_s), int(new_e), int(server_id)))
        conn.commit()
        return "修改成功"
    except Exception as e:
        return f"本地修改日程失败: {e}"


@tool(description="用于统计类问题。当用户询问历史或汇总信息时使用。返回统计结果。")
def statistics(query: str) -> str:
    safe_print(f"[DEBUG_TOOL statistics] called with: {query}")
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
# 记忆系统 & agent 初始化
# =========================================

memory = None
if ConversationBufferMemory is not None:
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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
重要要求：当用户请求创建、删除、修改或查询日程时，必须调用相应工具而不是直接返回普通文本结果。
工具名称：add_schedule, delete_schedule, update_schedule, query_schedule, statistics。
不要要求用户提供 JSON；从自然语言中解析字段并以 function_call 的形式调用工具。
你必须遵守以下规则：

1. 当用户请求涉及日程操作时，你必须调用对应的工具，而不是直接回答。
2. 所有日程数据必须通过工具访问，不允许编造日程。
3. 用户的自然语言时间（例如“明天下午3点”“3月22日”）需要转换为标准时间格式。
4. 如果用户的日程存在冲突，你可以建议新的时间。
5. 如果用户回复“好的”“可以”“行”“那就那个时间”等确认语句，表示用户接受你之前的建议，应继续执行对应操作。
6. 如果用户输入模糊的时间、事件等信息时，你应该询问更多信息。
7. 不要输出空响应。
"""
# 要求模型以 ReAct 风格进行推理（先思考/说明，再决定是否调用工具），并在需要时写出短的思考链和工具选择理由。

# 注入系统当前日期，帮助模型回答“今天/明天”等问题
system_prompt += f"\n当前系统日期: {datetime.now().strftime('%Y-%m-%d')}"
system_prompt += "\n当用户询问当前日期或时间时，必须使用上述系统当前日期/时间。"

agent = None
if initialize_agent is not None:
    try:
        agent = initialize_agent(
            tools=[add_schedule, delete_schedule, update_schedule, query_schedule, statistics],
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=memory,
            verbose=True,
            agent_kwargs={"system_message": system_prompt}
        )
        safe_print("已初始化为 ReAct 风格 agent（ZERO_SHOT_REACT_DESCRIPTION）")
    except Exception as e:
        safe_print(f"初始化 agent 失败: {e}")


# 调试：列出已注册的工具与描述（若 agent 可用）
try:
    safe_print("已注册工具：")
    if agent is not None:
        for t in agent.tools:
            try:
                desc = getattr(t, 'description', '')
            except Exception:
                desc = ''
            safe_print(f"- {getattr(t,'name',str(t))}: {desc}")
    else:
        # 列出本地函数的 description
        for f in [add_schedule, delete_schedule, update_schedule, query_schedule, statistics]:
            safe_print(f"- {getattr(f,'__name__',str(f))}: {getattr(f,'description', getattr(f,'__doc__', '') )}")
except Exception as ex:
    safe_print(f"[DEBUG] 列出工具时发生异常: {ex}")


# =========================================
# 主交互循环：使用 agent 首选，若 agent 返回空则携带最近上下文重试一次
# =========================================

safe_print(""" 你好，我是万机，日理万机的万机，你的智能日程助手。

我可以帮你：

安排日程（例如：帮我明天下午3点安排会议）
查询日程（例如：我今天有什么安排？）
修改日程（例如：把明天的会议推迟一小时）
取消日程（例如：取消周五的聚餐）
统计日程（例如：这周我跑了几次步）

你可以像和朋友聊天一样告诉我你的计划，比如：

“帮我安排明天下午3点开会”
“我周末有什么安排？”
“把3月22日的测试推迟一个小时”

如果时间有冲突，我也会帮你找到合适的时间。

现在，你想安排什么事情吗？
""")

last_bot_message = None
last_user_message = None
last_suggestion = None
last_pending_payload = None
pending_action = None

while True:
    user_input = input("你：")
    try:
        if agent is None:
            safe_print("万机：本地模式，直接使用工具。请使用明确指令。")
            continue
        # 记录用户输入到对话记忆中
        save_message('user', user_input)
        # 本地快捷：用户询问当前时间/日期时，直接使用系统时钟回复，避免模型基于对话历史误判
        try:
            if re.search(r"(现在.*(几点|时间)|几点了|现在是几点)", user_input):
                now = datetime.now()
                resp = now.strftime("%Y-%m-%d %H:%M:%S")
                safe_print("万机：现在时间是 " + resp)
                save_message('assistant', "现在时间是 " + resp)
                last_bot_message = resp
                last_user_message = user_input
                continue
            if re.search(r"(今天.*(几号|日期)|今天是几号|今天几号)", user_input):
                today = datetime.now()
                weekdays = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
                wd = weekdays[today.weekday()]
                resp = f"{today.strftime('%Y-%m-%d')}（{wd}）"
                safe_print("万机：今天是 " + resp)
                save_message('assistant', "今天是 " + resp)
                last_bot_message = resp
                last_user_message = user_input
                continue
        except Exception:
            pass
        # 若用户接受了上一次建议（例如："可以"、"好"），则自动使用建议时段创建日程
        if re.search(r"^\s*(可以|好|行|确定|就这样|安排吧)\s*$", user_input):
            if pending_action and pending_action.get('suggestion') and pending_action.get('payload'):
                intent = pending_action.get('intent')
                payload = dict(pending_action.get('payload') or {})
                s_iso, e_iso = pending_action.get('suggestion')
                try:
                    if intent == 'add':
                        payload['start_time'] = s_iso
                        payload['end_time'] = e_iso
                        safe_print(f"[AUTO APPLY SUGGESTION] 使用建议时段创建: {payload}")
                        res = add_schedule({"data": payload})
                    elif intent == 'update':
                        payload['new_start_time'] = s_iso
                        payload['new_end_time'] = e_iso
                        safe_print(f"[AUTO APPLY SUGGESTION] 使用建议时段修改: {payload}")
                        res = update_schedule({"data": payload})
                    else:
                        res = "无法识别的 pending action 类型"
                except Exception as e:
                    res = f"自动应用建议时出错: {e}"
                safe_print("万机：" + str(res))
                save_message('assistant', str(res))
                last_bot_message = res
                last_user_message = user_input
                # 清理状态
                last_suggestion = None
                last_pending_payload = None
                pending_action = None
                continue
            else:
                # 无建议可用，继续正常处理
                pass

        # 简单快捷路径：若用户直接请求查看/查询日程且包含自然语言时间（今天/明天/后天/具体日期），
        # 直接调用本地 query_schedule 工具以确保执行，而不是完全依赖模型解析。
        # 优先检测增删改查意图并直接调用对应本地工具，保证必定执行工具操作
        # 添加日程
        if re.search(r"安排|帮我在|帮我安排|创建|添加", user_input):
            # 支持“日”或“号”，以及可省略年份的中文日期
            date_pattern = r"\d{1,4}年?\s*\d{1,2}月\s*\d{1,2}(?:日|号)?|\d{4}-\d{1,2}-\d{1,2}"
            time_pattern = r"\d{1,4}年?\s*\d{1,2}月\s*\d{1,2}(?:日|号)?\s*\d{1,2}[:：]\d{2}|\d{1,2}[:：]\d{2}|(?:上午|下午|晚上)?\d{1,2}点(?:钟)?"
            title_m = re.search(r"标题[:：]\s*([^，,]+)", user_input)
            title = title_m.group(1).strip() if title_m else None
            # 先在全文中查找显式日期（优先级最高），再查找 time 片段
            safe_print(f"[DBG RAW_INPUT] {repr(user_input)}")
            mdate = re.search(date_pattern, user_input)
            explicit_date_str = mdate.group(0) if mdate else None
            # 回退：有时用户写成 '3月22 14:00'（无 '日' 或 '号'），尝试捕获月日数字组合
            if not explicit_date_str:
                m_fallback = re.search(r"(\d{1,2})月\s*(\d{1,2})", user_input)
                if m_fallback:
                    explicit_date_str = f"{m_fallback.group(1)}月{m_fallback.group(2)}日"
            if explicit_date_str:
                parsed_date = parse_explicit_date_str(explicit_date_str)
                if parsed_date:
                    ref = parsed_date
                else:
                    try:
                        ref = parser.parse(explicit_date_str)
                    except Exception:
                        ref = get_reference_date_from_text(user_input)
            else:
                ref = get_reference_date_from_text(user_input)
            times = re.findall(time_pattern, user_input)
            safe_print(f"[DBG SHORTCUT add] mdate={explicit_date_str} times={times} ref={ref}")
            parsed = []
            for t in times:
                try:
                    parsed.append(parse_nl_time(t, reference_date=ref))
                except Exception:
                    pass
            if not title:
                m2 = re.search(r"安排([^，,]+)", user_input)
                if m2:
                    possible = re.sub(time_pattern, "", m2.group(1))
                    title = possible.strip(' ，,')
            if len(parsed) >= 2 and title:
                payload = {"title": title, "start_time": parsed[0].isoformat(), "end_time": parsed[1].isoformat(), "category": None}
                # 保存候选负载，便于在接受建议时使用
                last_pending_payload = dict(payload)
                safe_print(f"[SHORTCUT add] 调用 add_schedule with {payload}")
                res = add_schedule({"data": payload})
                safe_print("万机：" + str(res))
                save_message('assistant', str(res))
                last_bot_message = res
                last_user_message = user_input
                # 如果返回中包含建议时段，提取并保存（格式： 建议空闲时段: YYYY-MM-DD HH:MM:SS - YYYY-MM-DD HH:MM:SS）
                try:
                    m = re.search(r"建议空闲时段[:：]\s*([0-9\- :]{16,19})\s*-\s*([0-9\- :]{16,19})", str(res))
                    if m:
                        s_iso = m.group(1).strip()
                        e_iso = m.group(2).strip()
                        last_suggestion = (s_iso, e_iso)
                        # 记录 pending action，便于用户确认时自动执行
                        pending_action = {'intent': 'add', 'payload': dict(last_pending_payload or {}), 'suggestion': (s_iso, e_iso)}
                except Exception:
                    pass
                continue
        # 修改日程
        if re.search(r"改|修改|移动|推迟|提前", user_input):
            date_pattern = r"\d{1,4}年?\s*\d{1,2}月\s*\d{1,2}(?:日|号)?|\d{4}-\d{1,2}-\d{1,2}"
            title_m = re.search(r"标题[:：]\s*([^，,]+)", user_input)
            title = title_m.group(1).strip() if title_m else None
            # 复用时间解析
            time_pattern = r"\d{1,4}年?\s*\d{1,2}月\s*\d{1,2}日\s*\d{1,2}[:：]\d{2}|\d{1,2}[:：]\d{2}|(?:上午|下午|晚上)?\d{1,2}点(?:钟)?"
            title_m = re.search(r"标题[:：]\s*([^，,]+)", user_input)
            title = title_m.group(1).strip() if title_m else None
            mdate = re.search(date_pattern, user_input)
            explicit_date_str = mdate.group(0) if mdate else None
            if not explicit_date_str:
                m_fallback = re.search(r"(\d{1,2})月\s*(\d{1,2})", user_input)
                if m_fallback:
                    explicit_date_str = f"{m_fallback.group(1)}月{m_fallback.group(2)}日"
            if explicit_date_str:
                parsed_date = parse_explicit_date_str(explicit_date_str)
                if parsed_date:
                    ref = parsed_date
                else:
                    try:
                        ref = parser.parse(explicit_date_str)
                    except Exception:
                        ref = get_reference_date_from_text(user_input)
            else:
                ref = get_reference_date_from_text(user_input)
            times = re.findall(time_pattern, user_input)
            safe_print(f"[DBG SHORTCUT update] mdate={explicit_date_str} times={times} ref={ref}")
            parsed = []
            for t in times:
                try:
                    parsed.append(parse_nl_time(t, reference_date=ref))
                except Exception:
                    pass
            if not title:
                m2 = re.search(r"把([^从]+)从", user_input)
                if m2:
                    title = m2.group(1).strip()
            if title and len(parsed) >= 2:
                payload = {"title": title, "new_start_time": parsed[0].isoformat(), "new_end_time": parsed[1].isoformat()}
                # 保存修改候选负载，以便接受建议时重用
                last_pending_payload = dict(payload)
                safe_print(f"[SHORTCUT update] 调用 update_schedule with {payload}")
                res = update_schedule({"data": payload})
                safe_print("万机：" + str(res))
                save_message('assistant', str(res))
                last_bot_message = res
                last_user_message = user_input
                try:
                    m = re.search(r"建议空闲时段[:：]\s*([0-9\- :]{16,19})\s*-\s*([0-9\- :]{16,19})", str(res))
                    if m:
                        s_iso = m.group(1).strip()
                        e_iso = m.group(2).strip()
                        last_suggestion = (s_iso, e_iso)
                        pending_action = {'intent': 'update', 'payload': dict(last_pending_payload or {}), 'suggestion': (s_iso, e_iso)}
                except Exception:
                    pass
                continue
        # 删除日程
        if re.search(r"删除|移除", user_input):
            title_m = re.search(r"标题[:：]\s*([^，,]+)", user_input)
            if title_m:
                t = title_m.group(1).strip()
            else:
                m2 = re.search(r"删除([^，,]+)", user_input)
                t = m2.group(1).strip(' ，,') if m2 else None
            if t:
                safe_print(f"[SHORTCUT delete] 调用 delete_schedule with {t}")
                res = delete_schedule(t)
                safe_print("万机：" + str(res))
                save_message('assistant', str(res))
                last_bot_message = res
                last_user_message = user_input
                continue
        # 统计意图
        if re.search(r"多少|统计|共有多少|上周|次数", user_input):
            safe_print(f"[SHORTCUT stats] 调用 statistics with {user_input}")
            res = statistics(user_input)
            safe_print("万机：" + str(res))
            save_message('assistant', str(res))
            last_bot_message = res
            last_user_message = user_input
            continue

        if re.search(r"查看|查询|查|有什么安排|日程", user_input):
            safe_print(f"[DBG RAW_INPUT QUERY] {repr(user_input)}")
            # 识别多日表达：这两天/这几天/最近N天/未来N天/本周/下周
            m_multi = re.search(r"这\s*两\s*天|这\s*几\s*天|最近\s*(\d+)\s*天|未来\s*(\d+)\s*天|本周|下周|下个\s*周", user_input)
            if m_multi:
                try:
                    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    if re.search(r"这\s*两\s*天|这\s*几\s*天", user_input):
                        start = today
                        end = today + timedelta(days=2)
                    else:
                        m_recent = re.search(r"最近\s*(\d+)\s*天", user_input)
                        m_future = re.search(r"未来\s*(\d+)\s*天", user_input)
                        if m_recent:
                            days = int(m_recent.group(1))
                            start = today - timedelta(days=days-1)
                            end = today + timedelta(days=1)
                        elif m_future:
                            days = int(m_future.group(1))
                            start = today
                            end = today + timedelta(days=days)
                        elif re.search(r"本周", user_input):
                            wd = today.weekday()
                            start = today - timedelta(days=wd)
                            end = start + timedelta(days=7)
                        else:
                            # 下周
                            wd = today.weekday()
                            start = today - timedelta(days=wd) + timedelta(days=7)
                            end = start + timedelta(days=7)
                    start_str = start.date().isoformat()
                    end_str = (end - timedelta(days=1)).date().isoformat()
                    safe_print(f"[SHORTCUT RANGE] 直接调用 query_schedule_range with {start_str} - {end_str}")
                    res = query_schedule_range({"start": start_str, "end": end_str})
                    safe_print("万机：" + str(res))
                    save_message('assistant', str(res))
                    last_bot_message = res
                    last_user_message = user_input
                    continue
                except Exception as e:
                    safe_print(f"范围查询失败: {e}")
            # 若用户提供了明确日期则直接查询那一天
            mdate = re.search(r"\d{1,4}年?\s*\d{1,2}月\s*\d{1,2}(?:日|号)?|\d{4}-\d{1,2}-\d{1,2}", user_input)
            if mdate:
                parsed_date = parse_explicit_date_str(mdate.group(0))
                if parsed_date:
                    date_str = parsed_date.date().isoformat()
                else:
                    date_str = mdate.group(0)
                safe_print(f"[SHORTCUT] 直接调用 query_schedule with {date_str}")
                try:
                    res = query_schedule({"date": date_str})
                    safe_print("万机：" + str(res))
                    save_message('assistant', str(res))
                    last_bot_message = res
                    last_user_message = user_input
                    continue
                except Exception as e:
                    safe_print(f"直接调用 query_schedule 失败: {e}")
            # 模糊或无明确时间的查询，放给 agent 推理处理（让 ReAct agent 决定如何调用工具）
            safe_print("[SHORTCUT] 未触发直接短路，交由 agent 推理处理")
            # 让模型自己决定调用工具：不在此处直接调用 query_schedule
            pass

        # 在调用模型前注入最近对话上下文，帮助模型理解未完成的 pending action
        recent_ctx = load_recent_conversation(20)
        augmented_prompt = f"最近对话：\n{recent_ctx}\n用户问题：{user_input}"
        response = agent.run(augmented_prompt)
        if not (response and str(response).strip()):
            safe_print("[DEBUG] 模型返回空响应，尝试带上下文重试一次...")
            ctx = ""
            if last_bot_message:
                ctx += f"上一次助手回复: {last_bot_message}\n"
            if last_user_message:
                ctx += f"上一次用户: {last_user_message}\n"
            # 在重试中也注入最近对话
            recent_ctx2 = load_recent_conversation(20)
            retry_prompt = f"最近对话：\n{recent_ctx2}\n{ctx}用户: {user_input}\n请基于上述对话和系统提示直接回答用户，不要要求结构化输入。"
            try:
                response = agent.run(retry_prompt)
            except Exception as e:
                safe_print(f"重试调用模型失败: {e}")
                response = None
            if not (response and str(response).strip()):
                safe_print("万机：抱歉，我没能理解你的意思，请更明确一些。")
            else:
                safe_print("万机：" + str(response))
                save_message('assistant', str(response))
                last_bot_message = response
                last_user_message = user_input
        else:
            safe_print("万机：" + str(response))
            save_message('assistant', str(response))
            last_bot_message = response
            last_user_message = user_input
    except Exception as e:
        safe_print("网络或 API 连接错误，无法联系模型服务。错误：" + str(e))
        continue

# （已移除重复的旧增强段落：使用上方单一的 agent 初始化与主循环）
