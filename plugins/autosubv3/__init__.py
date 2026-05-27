import copy
import os
import tempfile
import time
import traceback
from datetime import timedelta, datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List
from threading import Event, Lock
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import iso639
import psutil
import srt
from lxml import etree
from dataclasses import dataclass
from enum import Enum
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from uuid import uuid4
from app.core.config import settings
from app.core.context import MediaInfo
from app.core.event import eventmanager, Event as MPEvent
from app.schemas import TransferInfo
from app.schemas.types import NotificationType, EventType
from app.log import logger
from app.plugins import _PluginBase
from app.utils.system import SystemUtils
from plugins.autosubv3.ffmpeg import Ffmpeg
from plugins.autosubv3.translate.openai_translate import OpenAi


class UserInterruptException(Exception):
    """鐢ㄦ埛涓柇褰撳墠浠诲姟鐨勫紓甯?""
    pass


class TaskSource(Enum):
    MANUAL = "manual"
    EVENT = "event"


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    IGNORED = "ignored"
    NO_AUDIO = "no_audio"
    FAILED = "failed"


@dataclass
class TaskItem:
    task_id: str
    video_file: str
    source: TaskSource
    add_time: datetime
    status: TaskStatus = TaskStatus.PENDING
    complete_time: datetime = None


class FileMonitorHandler(FileSystemEventHandler):
    """
    鐩綍鐩戞帶鍝嶅簲绫伙紝鐩戝惉鏂板鏂囦欢浜嬩欢
    """

    def __init__(self, mon_path: str, plugin):
        super(FileMonitorHandler, self).__init__()
        self._watch_path = mon_path
        self._plugin = plugin

    def on_created(self, event):
        if not event.is_directory:
            logger.debug(f"妫€娴嬪埌鏂版枃浠讹細{event.src_path}")
            self._plugin._add_monitor_task(event.src_path)


class AutoSubv3(_PluginBase):
    # 鎻掍欢鍚嶇О
    plugin_name = "AI瀛楀箷榄旀敼鐗坴3"
    # 鎻掍欢鎻忚堪
    plugin_desc = "鑷姩鐢熸垚瀛楀箷骞剁炕璇戞垚涓枃锛屾敮鎸佹渶鏂皁penai sdk锛屾敼鎴愬苟鍙戯紝缈昏瘧閫熷害鍔犲€嶏紱鑷敤淇敼鐗堬紱"
    # 鎻掍欢鍥炬爣
    plugin_icon = "autosubtitles.jpeg"
    # 涓婚鑹?
    plugin_color = "#2C4F7E"
    # 鎻掍欢鐗堟湰
    plugin_version = "3.5.39"
    # 鎻掍欢浣滆€?
    plugin_author = "jianji112"
    # 浣滆€呬富椤?
    author_url = "https://github.com/jianji112"
    # 鎻掍欢閰嶇疆椤笽D鍓嶇紑
    plugin_config_prefix = "autosubv3"
    # 鍔犺浇椤哄簭
    plugin_order = 14
    # 鍙娇鐢ㄧ殑鐢ㄦ埛绾у埆
    auth_level = 2

    # 绉佹湁灞炴€?
    _tasks: Dict[str, TaskItem] = None
    _task_queue = None
    _consumer_thread = None
    _current_processing_task = None
    _running = False
    _event = Event()
    _enabled = None
    _clear_history = None
    _send_notify = None
    _translate_preference = None
    _run_now = None
    _path_list = None
    _file_size = None
    _translate_zh = None
    _openai = None
    _enable_batch = None
    _batch_size = None
    _parallel_workers = None
    _context_window = None
    _max_retries = None
    _enable_merge = None
    _enable_asr = None
    _auto_detect_language = None
    _huggingface_proxy = None
    _faster_whisper_model_path = None
    _faster_whisper_model = None
    _max_segment_duration = None
    _max_segment_chars = None
    _process_new_only = None
    _observer = None
    _monitor_paths = None
    _lock = Lock()

    def init_plugin(self, config=None):
        # 濡傛灉娌℃湁閰嶇疆淇℃伅锛?鍒欎笉澶勭悊
        if not config:
            return
        # 娓呯悊鎻掍欢鍚姩鍓嶇殑娈嬬暀涓存椂鏂囦欢
        tempdir = tempfile.gettempdir()
        for file in os.listdir(tempdir):
            if file.startswith('autosub-'):
                try:
                    os.remove(os.path.join(tempdir, file))
                    logger.info(f"娓呯悊娈嬬暀涓存椂鏂囦欢锛歿file}")
                except Exception:
                    pass
        self._tasks = self.load_tasks()
        self._enabled = config.get('enabled', False)
        self._clear_history = config.get('clear_history', False)
        # 鐩戞帶璺緞閰嶇疆
        monitor_str = config.get('path_whitelist', '').strip()
        self._monitor_paths = [p.strip() for p in monitor_str.split('\n') if p.strip()] if monitor_str else []
        self._process_new_only = config.get('process_new_only', True)
        self._run_now = config.get('run_now')
        if self._run_now:
            self._path_list = list(set(config.get('path_list').split('\n')))
        self._send_notify = config.get('send_notify', False)
        self._file_size = int(config.get('file_size')) if config.get('file_size') else 10
        # 瀛楀箷鐢熸垚璁剧疆
        self._translate_preference = config.get('translate_preference', 'english_first')
        self._enable_asr = config.get('enable_asr', True)
        if self._enable_asr:
            self._faster_whisper_model = config.get('faster_whisper_model', 'base')
            self._faster_whisper_model_path = config.get('faster_whisper_model_path',
                                                         self.get_data_path() / "faster-whisper-models")
            self._huggingface_proxy = config.get('proxy', True)
            self._auto_detect_language = config.get('auto_detect_language', False)
            self._skip_chinese = config.get('skip_chinese', False)
            self._max_segment_duration = float(config.get('max_segment_duration')) if config.get('max_segment_duration') else 8.0
            self._max_segment_chars = int(config.get('max_segment_chars')) if config.get('max_segment_chars') else 50
        self._translate_zh = config.get('translate_zh', False)
        if self._translate_zh:
            openai_key = config.get('openai_key')
            if not openai_key:
                logger.error(f"缈昏瘧渚濊禆浜嶰penAI锛岃鍏堢淮鎶penai_key")
                return
            openai_url = config.get('openai_url', "https://api.openai.com")
            openai_proxy = config.get('openai_proxy', False)
            openai_model = config.get('openai_model', "inclusionAI/Ling-flash-2.0")
            compatible = config.get('compatible', False)
            self._openai = OpenAi(api_key=openai_key, api_url=openai_url,
                                  proxy=settings.PROXY if openai_proxy else None,
                                  model=openai_model, compatible=bool(compatible))
            self._enable_batch = config.get('enable_batch', True)
            self._batch_size = int(config.get('batch_size')) if config.get('batch_size') else 20
            self._parallel_workers = int(config.get('parallel_workers')) if config.get('parallel_workers') else 10
            self._context_window = int(config.get('context_window')) if config.get('context_window') else 5
            self._max_retries = int(config.get('max_retries')) if config.get('max_retries') else 3
            self._enable_merge = config.get('enable_merge', False)
            self._subtitle_output_mode = config.get('subtitle_output_mode', 'bilingual')

        if self._clear_history:
            config['clear_history'] = False
            self.update_config(config)
            self.clear_tasks()
            self.save_skip_chinese_videos({})
        if self._enabled:
            logger.info("AI鐢熸垚瀛楀箷鏈嶅姟宸插惎鍔?)
            # asr 閰嶇疆妫€鏌?
            if self._enable_asr and not self.__check_asr():
                return

            if not self._running:
                self._task_queue = queue.Queue()
                self._consumer_thread = threading.Thread(target=self._consume_tasks, daemon=True)
                self._consumer_thread.start()
                logger.info("浠诲姟闃熷垪鍜屾秷璐硅€呯嚎绋嬪凡鍚姩")
                self._running = True

            # 鍚姩鐩綍鐩戞帶
            self._start_file_monitor()

            if self._run_now:
                config['run_now'] = False
                self.update_config(config)
                logger.info("绔嬪嵆杩愯涓€娆?)
                self._run_at_once(path_list=self._path_list)
        else:
            self.stop_service()

    def load_tasks(self) -> Dict[str, TaskItem]:
        raw_tasks = self.get_data("tasks") or {}
        tasks = {}
        for task_id, task_dict in raw_tasks.items():
            try:
                task = TaskItem(
                    task_id=task_dict["task_id"],
                    video_file=task_dict["video_file"],
                    source=TaskSource(task_dict["source"]),
                    add_time=datetime.fromisoformat(task_dict["add_time"]),
                    status=TaskStatus(task_dict["status"]),
                    complete_time=datetime.fromisoformat(task_dict["complete_time"])
                    if task_dict.get("complete_time") else None,
                )
                tasks[task_id] = task
            except Exception as e:
                logger.error(f"鎭㈠浠诲姟澶辫触锛歿e}")
        return tasks

    @staticmethod
    def _serialize_task(task: TaskItem) -> dict:
        return {
            "task_id": task.task_id,
            "video_file": task.video_file,
            "source": task.source.value,
            "add_time": task.add_time.isoformat() if task.add_time else None,
            "status": task.status.value,
            "complete_time": task.complete_time.isoformat() if task.complete_time else None,
        }

    def save_tasks(self):
        tasks_dict = {task_id: self._serialize_task(task) for task_id, task in self._tasks.items()}
        self.save_data("tasks", tasks_dict)

    def load_skipped_videos(self) -> Dict[str, dict]:
        """鍔犺浇鏃犲０闊宠烦杩囩殑瑙嗛璁板綍"""
        return self.get_data("skipped_videos") or {}

    def save_skipped_videos(self, skipped: Dict[str, dict]):
        """淇濆瓨鏃犲０闊宠烦杩囩殑瑙嗛璁板綍"""
        self.save_data("skipped_videos", skipped)

    def add_skipped_video(self, video_file: str):
        """娣诲姞鏃犲０闊宠烦杩囩殑瑙嗛璁板綍"""
        skipped = self.load_skipped_videos()
        skipped[video_file] = {
            "skip_time": datetime.now().isoformat(),
            "reason": "no_audio"
        }
        self.save_skipped_videos(skipped)
        logger.info(f"宸茶褰曟棤澹伴煶瑙嗛锛歿video_file}")

    def is_video_skipped(self, video_file: str) -> bool:
        """妫€鏌ヨ棰戞槸鍚﹀洜鏃犲０闊冲凡琚烦杩?""
        skipped = self.load_skipped_videos()
        return video_file in skipped

    @staticmethod
    def __is_chinese_lang(lang: str) -> bool:
        if not lang:
            return False
        return lang.lower() in ('zh', 'chi', 'chs', 'cht', 'zh-cn', 'zh-tw', 'zh-hk', 'chinese')

    def load_skip_chinese_videos(self):
        return self.get_data("skip_chinese_videos") or {}

    def save_skip_chinese_videos(self, skipped):
        self.save_data("skip_chinese_videos", skipped)

    def add_skip_chinese_video(self, video_file: str):
        skipped = self.load_skip_chinese_videos()
        skipped[video_file] = {
            "skip_time": datetime.now().isoformat(),
            "reason": "chinese"
        }
        self.save_skip_chinese_videos(skipped)
        logger.info(f"宸茶褰曚腑鏂囪棰戣烦杩囷細{video_file}")

    def is_video_skip_chinese(self, video_file: str) -> bool:
        return video_file in self.load_skip_chinese_videos()

    def add_task(self, video_file: str, source: TaskSource):
        """
        娣诲姞鏂颁换鍔″埌闃熷垪鍜屼换鍔″垪琛ㄤ腑锛岃嫢浠诲姟宸插瓨鍦ㄥ垯璺宠繃銆?
        :param video_file: 瑙嗛鏂囦欢璺緞
        :param source: 浠诲姟鏉ユ簮锛堟墜鍔?浜嬩欢锛?
        """
        task = TaskItem(
            task_id=str(uuid4()),
            video_file=video_file,
            source=source,
            add_time=datetime.now()
        )

        if self.__is_duplicate_task(task.video_file):
            logger.info(f"浠诲姟宸插瓨鍦紝璺宠繃娣诲姞锛歿video_file}")
            return False

        self._task_queue.put(task)
        self._tasks[task.task_id] = task
        self.save_tasks()
        logger.info(f"鍔犲叆浠诲姟闃熷垪: {video_file}")
        return True

    def clear_tasks(self):
        self._tasks = {task_id: task for task_id, task in self._tasks.items() if task.status in [
            TaskStatus.PENDING, TaskStatus.IN_PROGRESS
        ]}
        self.save_tasks()
        self.save_skipped_videos({})
        logger.info("鎻掍欢鍘嗗彶浠诲姟宸叉竻闄?)

    def __is_duplicate_task(self, video_file: str) -> bool:
        with self._task_queue.mutex:
            for task in self._task_queue.queue:
                if task.video_file == video_file:
                    return True
            # 杩樿妫€鏌ュ綋鍓嶆鍦ㄥ鐞嗙殑浠诲姟锛堝嵆鍙兘涓嶅湪闃熷垪涓紝浣嗘鍦ㄨ娑堣垂锛?
            if self._consumer_thread and self._current_processing_task and self._current_processing_task.video_file == video_file:
                return True
        return False

    def _consume_tasks(self):
        while not self._event.is_set():
            try:
                task = self._task_queue.get(timeout=1)
                if task is None:
                    continue
                self._current_processing_task = task
                logger.info(f"寮€濮嬪鐞嗕换鍔?{task.task_id}: {task.video_file}")
                task.status = TaskStatus.IN_PROGRESS
                self._tasks[task.task_id] = task
                self.save_tasks()
                task.status = self.__process_autosub(task.video_file)
                task.complete_time = datetime.now()
                self._tasks[task.task_id] = task
                self.save_tasks()
                self._task_queue.task_done()
                self._current_processing_task = None
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"娑堣垂浠诲姟鏃跺彂鐢熷紓甯? {e}")
                logger.error(traceback.format_exc())
                self._current_processing_task = None
        logger.info("娑堣垂绾跨▼宸查€€鍑?)

    # 鐩戝惉濯掍綋鍏ュ簱浜嬩欢锛屾瘡涓簨浠惰Е鍙戜竴娆¤嚜鍔ㄥ瓧骞曚换鍔?
    @eventmanager.register(EventType.TransferComplete)
    def _start_file_monitor(self):
        """鍚姩鐩綍鐩戞帶"""
        # 鍋滄鐜版湁 observer
        if self._observer:
            try:
                self._observer.stop()
                self._observer.join(timeout=5)
            except Exception:
                pass
            self._observer = None

        if not self._monitor_paths:
            logger.info("鏈厤缃洃鎺ц矾寰勶紝涓嶅惎鍔ㄧ洰褰曠洃鎺?)
            return

        # 鍏ㄩ噺鎵弿锛堜粎澶勭悊鏂板鍏抽棴鏃讹級
        if not self._process_new_only:
            logger.info("浠呭鐞嗘柊澧炲叧闂紝寮€濮嬪叏閲忔壂鎻忕洃鎺ц矾寰?...")
            for mon_path in self._monitor_paths:
                if os.path.isdir(mon_path):
                    for video_file in self._get_library_files(mon_path):
                        self.add_task(video_file, TaskSource.EVENT)
            logger.info("鍏ㄩ噺鎵弿瀹屾垚")

        # 鍚姩 watchdog 鐩戞帶
        try:
            self._observer = Observer(timeout=10)
            for mon_path in self._monitor_paths:
                if os.path.isdir(mon_path):
                    handler = FileMonitorHandler(mon_path, self)
                    self._observer.schedule(handler, path=mon_path, recursive=True)
                    logger.info(f"鍚姩鐩綍鐩戞帶锛歿mon_path}")
            self._observer.daemon = True
            self._observer.start()
            logger.info("鐩綍鐩戞帶鏈嶅姟宸插惎鍔?)
        except Exception as e:
            logger.error(f"鍚姩鐩綍鐩戞帶澶辫触锛歿e}")
            logger.error(traceback.format_exc())

    def _add_monitor_task(self, file_path: str):
        """鐩戞帶澶勭悊鍣ㄥ洖璋冿紝娣诲姞鏂版枃浠朵换鍔?""
        if not os.path.exists(file_path):
            return
        ext = os.path.splitext(file_path)[-1].lower()
        if ext not in settings.RMT_MEDIAEXT:
            return
        with self._lock:
            self.add_task(file_path, TaskSource.EVENT)

    def _run_at_once(self, path_list: List[str]):
        # 绔嬪嵆鎵ц涓€娆★細鎵ц閰嶇疆鐨勫獟浣撳簱鐩綍锛屼笉鍙楃櫧鍚嶅崟闄愬埗
        # 鐧藉悕鍗曚粎鍦ㄨ嚜鍔ㄥ叆搴撲簨浠朵腑鐢熸晥
        for path in path_list:
            if not os.path.exists(path) or not os.path.isabs(path):
                logger.warn(f"鐩綍/鏂囦欢鏃犳晥锛屼笉杩涜澶勭悊:{path}")
                continue
            if os.path.isdir(path):
                for video_file in self.__get_library_files(path):
                    self.add_task(video_file, TaskSource.MANUAL)
            elif os.path.splitext(path)[-1].lower() in settings.RMT_MEDIAEXT:
                self.add_task(path, TaskSource.MANUAL)

    def __check_asr(self):
        if not self._faster_whisper_model_path or not self._faster_whisper_model:
            logger.warn(f"faster-whisper閰嶇疆淇℃伅涓嶅畬鏁达紝涓嶈繘琛屽鐞?)
            return False
        if not os.path.exists(self._faster_whisper_model_path):
            logger.info(f"鍒涘缓faster-whisper妯″瀷鐩綍锛歿self._faster_whisper_model_path}")
            os.mkdir(self._faster_whisper_model_path)
        try:
            from faster_whisper import WhisperModel, download_model
        except ImportError:
            logger.warn(f"faster-whisper 鏈畨瑁咃紝涓嶈繘琛屽鐞?)
            return False
        return True

    def __process_autosub(self, video_file) -> TaskStatus:
        if not video_file:
            logger.error(f"[Step 0] video_file 涓虹┖")
            return TaskStatus.FAILED
        logger.info(f"[Step 1] 妫€鏌ユ枃浠跺ぇ灏忥細{video_file}")
        # 濡傛灉鏂囦欢澶у皬灏忎簬鎸囧畾澶у皬锛?鍒欎笉澶勭悊
        if os.path.getsize(video_file) < self._file_size * 1024 * 1024:
            logger.info(f"[Step 1] 鏂囦欢灏忎簬鏈€灏忓ぇ灏?{self._file_size}MB锛岃烦杩?)
            return TaskStatus.IGNORED
        logger.info(f"[Step 2] 妫€鏌ユ槸鍚﹀凡鏍囪涓烘棤澹伴煶璺宠繃")
        # 妫€鏌ユ槸鍚﹀凡鏍囪涓烘棤澹伴煶璺宠繃
        if self.is_video_skipped(video_file):
            logger.info(f"[Step 2] 瑙嗛宸叉爣璁颁负鏃犲０闊宠烦杩囷細{video_file}")
            return TaskStatus.NO_AUDIO
        logger.info(f"[Step 3] 寮€濮嬫寮忓鐞?)
        start_time = time.time()
        file_path, file_ext = os.path.splitext(video_file)
        file_name = os.path.basename(video_file)
        if self._skip_chinese and self.is_video_skip_chinese(video_file):
            logger.info(f"[Step 3] 瑙嗛宸叉爣璁颁负涓枃璺宠繃缈昏瘧锛歿video_file}")
            message = f" 濯掍綋: {file_name}\n 涓枃瑙嗛璺宠繃缈昏瘧"
            if self._send_notify:
                self.post_message(mtype=NotificationType.Plugin, title="銆愯嚜鍔ㄥ瓧骞曠敓鎴愩€?, text=message)
            return TaskStatus.IGNORED

        try:
            logger.info(f"[Step 4] 鍒ゆ柇鐩殑瀛楀箷鏄惁宸插瓨鍦細{video_file}")
            # 鍒ゆ柇鐩殑瀛楀箷锛堝拰鍐呭祵锛夋槸鍚﹀凡瀛樺湪
            if self.__target_subtitle_exists(video_file):
                logger.warn(f"[Step 4] 瀛楀箷鏂囦欢宸茬粡瀛樺湪锛屼笉杩涜澶勭悊")
                return TaskStatus.IGNORED
            logger.info(f"[Step 5] 鐢熸垚瀛楀箷")
            # 鐢熸垚瀛楀箷
            ret, lang, gen_sub_path = self.__generate_subtitle(video_file, file_path, self._enable_asr)
            if not ret:
                # 妫€鏌ユ槸鍚︽槸鏃犲０闊宠烦杩囷紙鍒氳褰曠殑锛?
                if self.is_video_skipped(video_file):
                    message = f" 濯掍綋: {file_name}\n 鏃犲０闊宠烦杩?
                    if self._send_notify:
                        self.post_message(mtype=NotificationType.Plugin, title="銆愯嚜鍔ㄥ瓧骞曠敓鎴愩€?, text=message)
                    return TaskStatus.NO_AUDIO
                if lang == "skip_chinese" or self.is_video_skip_chinese(video_file):
                    message = f" 濯掍綋: {file_name}\n 涓枃瑙嗛璺宠繃缈昏瘧"
                    if self._send_notify:
                        self.post_message(mtype=NotificationType.Plugin, title="銆愯嚜鍔ㄥ瓧骞曠敓鎴愩€?, text=message)
                    return TaskStatus.IGNORED
                message = f" 濯掍綋: {file_name}\n 鐢熸垚瀛楀箷澶辫触锛岃烦杩囧悗缁鐞?
                if self._send_notify:
                    self.post_message(mtype=NotificationType.Plugin, title="銆愯嚜鍔ㄥ瓧骞曠敓鎴愩€?, text=message)
                return TaskStatus.FAILED

            logger.info(f"[Step 6] 缈昏瘧瀛楀箷锛堝鏋滈渶瑕侊級")
            translated_to_zh = False
            if self._translate_zh:
                # 缈昏瘧瀛楀箷锛堝嵆浣挎簮璇█鏄腑鏂囷紝涔熻繃LLM澶勭悊鐥呭彞銆佺箒杞畝銆佸幓绌烘牸锛?
                logger.info(f"寮€濮嬬炕璇戝瓧骞曚负涓枃 ...")
                self.__translate_zh_subtitle(lang, gen_sub_path, f"{file_path}.zh.鏈虹炕.srt",
                                              output_mode=self._subtitle_output_mode)
                logger.info(f"缈昏瘧瀛楀箷瀹屾垚锛歿file_name}.zh.鏈虹炕.srt")
                translated_to_zh = True

            end_time = time.time()
            message = f" 濯掍綋: {file_name}\n 澶勭悊瀹屾垚\n 瀛楀箷鍘熷璇█: {lang}\n "
            if translated_to_zh:
                message += f"瀛楀箷缈昏瘧璇█: zh\n "
            message += f"鑰楁椂锛歿round(end_time - start_time, 2)}绉?
            logger.info(f"鑷姩瀛楀箷鐢熸垚 澶勭悊瀹屾垚锛歿message}")
            logger.info("")  # 绌鸿鍒嗛殧
            logger.info("")  # 绌鸿鍒嗛殧
            if self._send_notify:
                self.post_message(mtype=NotificationType.Plugin, title="銆愯嚜鍔ㄥ瓧骞曠敓鎴愩€?, text=message)
            return TaskStatus.COMPLETED
        except UserInterruptException:
            logger.info(f"鐢ㄦ埛涓柇褰撳墠浠诲姟锛歿video_file}")
            logger.info("")  # 绌鸿鍒嗛殧
            logger.info("")  # 绌鸿鍒嗛殧
            return TaskStatus.FAILED
        except Exception as e:
            logger.error(f"鑷姩瀛楀箷鐢熸垚 澶勭悊寮傚父锛歿e}")
            end_time = time.time()
            message = f" 濯掍綋: {file_name}\n 澶勭悊澶辫触\n 鑰楁椂锛歿round(end_time - start_time, 2)}绉?
            if self._send_notify:
                self.post_message(mtype=NotificationType.Plugin, title="銆愯嚜鍔ㄥ瓧骞曠敓鎴愩€?, text=message)
            # 鎵撳嵃璋冪敤鏍?
            logger.error(traceback.format_exc())
            logger.info("")  # 绌鸿鍒嗛殧
            logger.info("")  # 绌鸿鍒嗛殧
            return TaskStatus.FAILED

    def __do_speech_recognition(self, audio_lang, audio_file, video_file=None):
        """
        璇煶璇嗗埆, 鐢熸垚瀛楀箷
        :param audio_lang:
        :param audio_file:
        :param video_file: 瑙嗛鏂囦欢璺緞锛堢敤浜庢棩蹇楁樉绀猴級
        :return:
        """
        lang = audio_lang
        video_name = os.path.basename(video_file) if video_file else os.path.basename(audio_file)
        logger.info(f"[Whisper闊抽鎻愬彇鏂囨湰] 寮€濮嬪鐞? {video_name}")
        try:
            from faster_whisper import WhisperModel, download_model
            logger.info(f"[Whisper闊抽鎻愬彇鏂囨湰] {video_name} - 鍔犺浇妯″瀷涓?..")
            # 璁剧疆缂撳瓨鐩綍, 闃叉缂撳瓨鍚岀洰褰曞嚭鐜?cross-device 閿欒
            cache_dir = os.path.join(self._faster_whisper_model_path, "cache")
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)
            os.environ["HF_HUB_CACHE"] = cache_dir
            if self._huggingface_proxy:
                os.environ["HTTP_PROXY"] = settings.PROXY['http']
                os.environ["HTTPS_PROXY"] = settings.PROXY['https']
            
            # 妯″瀷涓嬭浇閲嶈瘯鏈哄埗
            max_retries = 3
            model = None
            for attempt in range(max_retries):
                try:
                    model_path = download_model(self._faster_whisper_model, local_files_only=False, cache_dir=cache_dir)
                    if model_path is None:
                        raise ValueError("妯″瀷涓嬭浇杩斿洖绌鸿矾寰?)
                    model = WhisperModel(model_path, device="cpu", compute_type="int8", cpu_threads=psutil.cpu_count(logical=False))
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warn(f"[Whisper闊抽鎻愬彇鏂囨湰] {video_name} - 妯″瀷涓嬭浇澶辫触锛堢{attempt+1}娆★級锛?0绉掑悗閲嶈瘯... 閿欒: {e}")
                        time.sleep(30)
                    else:
                        logger.error(f"[Whisper闊抽鎻愬彇鏂囨湰] {video_name} - 妯″瀷涓嬭浇澶辫触锛屽凡閲嶈瘯{max_retries}娆°€傝妫€鏌ワ細1) 缃戠粶杩炴帴 2) 浠ｇ悊閰嶇疆 3) HuggingFace璁块棶銆傞敊璇? {e}")
                        return False, None
            
            try:
                segments, info = model.transcribe(audio_file,
                                                  language=lang if lang != 'auto' else None,
                                                  word_timestamps=True,
                                                  vad_filter=True,
                                                  temperature=0,
                                                  beam_size=5)
                logger.info(f"[Whisper闊抽鎻愬彇鏂囨湰] {video_name} - 妫€娴嬪埌璇█锛歿info.language}锛堢疆淇″害 {info.language_probability:.2%}锛?)

                detected_lang = info.language
                if lang == 'auto':
                    lang = detected_lang

                if self._skip_chinese and self.__is_chinese_lang(lang):
                    logger.info(f"[Whisper闊抽鎻愬彇鏂囨湰] {video_name} - 妫€娴嬪埌涓枃涓斿凡寮€鍚腑鏂囪棰戜笉缈昏瘧锛岀珛鍗宠烦杩囧悗缁瓧骞曟彁鍙?)
                    return "skip_chinese", lang

                logger.info(f"[Whisper闊抽鎻愬彇鏂囨湰] {video_name} - 寮€濮嬫彁鍙栧瓧骞曞唴瀹癸紝璇█锛歿lang}")
                extract_start_time = time.time()
            except ValueError as e:
                if "max() iterable argument is empty" in str(e):
                    logger.info(f"[Whisper闊抽鎻愬彇鏂囨湰] {video_name} - 闊抽鏂囦欢涓湭妫€娴嬪埌浠讳綍璇█鍐呭锛屾爣璁颁负鏃犲０闊?)
                    # 杩斿洖 None 琛ㄧず鏃犲０闊筹紝涓嶇敓鎴愮┖瀛楀箷鏂囦欢
                    return None, None
                else:
                    raise e

            # 鍏堥亶鍘嗕竴娆¤幏鍙栨€绘椂闀匡紝鐢ㄤ簬鐧惧垎姣旇繘搴︽樉绀?
            seg_list = list(segments)
            total_duration = seg_list[-1].end if seg_list else 0
            total_count = len(seg_list)
            subs = []
            idx = 0
            last_pct = 0
            for segment in seg_list:
                if self._event.is_set():
                    logger.info(f"[Whisper闊抽鎻愬彇鏂囨湰] {video_name} - 鐢ㄦ埛涓柇锛屽仠姝㈡彁鍙?)
                    raise UserInterruptException(f"鐢ㄦ埛涓柇褰撳墠浠诲姟")
                pct = int(segment.end / total_duration * 100) if total_duration > 0 else 0
                if pct >= last_pct + 10:
                    logger.info(f"[Whisper闊抽鎻愬彇鏂囨湰] {video_name} - 鎻愬彇杩涘害锛歿pct}%锛坽segment.end:.1f}s / {total_duration:.1f}s锛?)
                    last_pct = pct
                if segment.words:
                    for word in segment.words:
                        idx += 1
                        subs.append(srt.Subtitle(index=idx,
                                                 start=timedelta(seconds=word.start),
                                                 end=timedelta(seconds=word.end),
                                                 content=word.word))
                else:
                    idx += 1
                    subs.append(srt.Subtitle(index=idx,
                                             start=timedelta(seconds=segment.start),
                                             end=timedelta(seconds=segment.end),
                                             content=segment.text))
            # 鎸夋渶澶ф椂闀垮拰鏈€澶у瓧鏁板悎骞?
            subs = self.__merge_srt(subs)
            
            # 璁＄畻鎻愬彇鑰楁椂
            extract_elapsed = time.time() - extract_start_time
            logger.info(f"[Whisper闊抽鎻愬彇鏂囨湰] {video_name} - 鎻愬彇瀹屾垚锛屽叡澶勭悊 {total_count} 娈碉紝鍚堝苟鍚?{idx} 鏉″瓧骞曪紝鑰楁椂 {extract_elapsed:.1f} 绉?)
            
            # 鎬ц兘璀﹀憡锛堝熀浜庢彁鍙栨椂闀夸笌瑙嗛鏃堕暱鐨勬瘮渚嬶級
            if total_duration > 0:
                ratio = extract_elapsed / total_duration
                if ratio >= 0.8:
                    logger.warning(f"[Whisper闊抽鎻愬彇鏂囨湰] {video_name} - 鎻愬彇鑰楁椂杩囬暱锛坽extract_elapsed:.1f}绉?/ 瑙嗛{total_duration:.1f}绉?= {ratio:.0%}锛夛紝寮虹儓寤鸿锛?) 浣跨敤鏇村揩妯″瀷锛坱iny/base锛?) 鍚敤GPU鍔犻€?3) 妫€鏌PU璐熻浇")
                elif ratio >= 0.6:
                    logger.warning(f"[Whisper闊抽鎻愬彇鏂囨湰] {video_name} - 鎻愬彇鑰楁椂杈冮暱锛坽extract_elapsed:.1f}绉?/ 瑙嗛{total_duration:.1f}绉?= {ratio:.0%}锛夛紝寤鸿锛?) 浣跨敤鏇村揩妯″瀷锛坱iny/base锛?) 鍚敤GPU鍔犻€?)
                elif ratio >= 0.3:
                    logger.info(f"[Whisper闊抽鎻愬彇鏂囨湰] {video_name} - 鎻愬彇閫熷害鍙紭鍖栵紙{extract_elapsed:.1f}绉?/ 瑙嗛{total_duration:.1f}绉?= {ratio:.0%}锛夛紝鍙€冭檻浣跨敤鏇村揩妯″瀷锛坱iny/base锛?)
            
            # 妫€鏌ユ槸鍚︽彁鍙栧埌浜嗘湁鏁堝瓧骞曞唴瀹?
            if not subs:
                logger.info(f"[Whisper闊抽鎻愬彇鏂囨湰] {video_name} - 鎻愬彇鐨勫瓧骞曞唴瀹逛负绌猴紝鏍囪涓烘棤澹伴煶")
                return None, None
                
            self.__save_srt(f"{audio_file}.srt", subs)
            logger.info(f"[Whisper闊抽鎻愬彇鏂囨湰] {video_name} - 闊宠建杞瓧骞曞畬鎴?)
            return True, lang
        except ImportError:
            logger.warn(f"[Whisper闊抽鎻愬彇鏂囨湰] faster-whisper 鏈畨瑁咃紝涓嶈繘琛屽鐞?)
            return False, None
        except Exception as e:
            traceback.print_exc()
            logger.error(f"[Whisper闊抽鎻愬彇鏂囨湰] {video_name} - 澶勭悊寮傚父锛歿e}")
            return False, None

    def __generate_subtitle(self, video_file, subtitle_file, enable_asr=True):
        """
        鐢熸垚瀛楀箷
        :param video_file: 瑙嗛鏂囦欢
        :param subtitle_file: 瀛楀箷鏂囦欢, 涓嶅寘鍚悗缂€
        :return: 鐢熸垚鎴愬姛杩斿洖True锛屽瓧骞曡瑷€,瀛楀箷璺緞锛屽惁鍒欒繑鍥濬alse, None, None
        """
        # 鑾峰彇鏂囦欢鍏冩暟鎹?
        logger.info(f"[GenSub] 鑾峰彇瑙嗛鍏冩暟鎹細{video_file}")
        video_meta = Ffmpeg().get_video_metadata(video_file)
        if not video_meta:
            logger.error(f"[GenSub] 鑾峰彇瑙嗛鍏冩暟鎹け璐ワ紝璺宠繃鍚庣画澶勭悊")
            return False, None, None
        logger.info(f"[GenSub] 鑾峰彇瑙嗛鍏冩暟鎹垚鍔?)
        # 鑾峰彇瀛楀箷璇█鍋忓ソ
        if self._translate_preference == "english_only":
            prefer_subtitle_langs = ['en', 'eng']
            strict = True
        elif self._translate_preference == "english_first":
            prefer_subtitle_langs = ['en', 'eng']
            strict = False
        else:  # self.translate_preference == "origin_first"
            prefer_subtitle_langs = None
            strict = False

        # 浠庤棰戞枃浠堕煶杞ㄨ幏鍙栬瑷€淇℃伅
        logger.info(f"[GenSub Step 2] 鑾峰彇闊宠建淇℃伅")
        ret, audio_index, audio_lang = self.__get_video_prefer_audio(video_meta, prefer_lang=prefer_subtitle_langs)
        if not ret:
            logger.info(f"瀛楀箷婧愬亸濂斤細{self._translate_preference} 鑾峰彇闊宠建鍏冩暟鎹け璐?)
            return False, None, None
        
        # 濡傛灉寮€鍚簡鑷姩璇█妫€娴嬶紝鐩存帴璁剧疆涓篴uto锛岃烦杩噈etadata鐨勮瑷€淇℃伅
        if self._auto_detect_language:
            logger.info("宸插紑鍚嚜鍔ㄨ瑷€妫€娴嬶紝灏嗕娇鐢╳hisper妯″瀷鑷姩璇嗗埆璇█")
            audio_lang = 'auto'
        elif not iso639.find(audio_lang) or not iso639.to_iso639_1(audio_lang):
            logger.info(f"瀛楀箷婧愬亸濂斤細{self._translate_preference} 鏈粠闊宠建鍏冩暟鎹腑鑾峰彇鍒拌瑷€淇℃伅")
            audio_lang = 'auto'

        # 褰撳瓧骞曟簮鍋忓ソ涓簅rigin_first鏃讹紝浼樺厛浣跨敤闊宠建璇█
        if self._translate_preference == "origin_first":
            prefer_subtitle_langs = ['en', 'eng'] if audio_lang == 'auto' else [audio_lang,
                                                                                iso639.to_iso639_1(audio_lang)]
        # 鑾峰彇澶栨寕瀛楀箷
        logger.info(f"[GenSub Step 3] 妫€鏌ュ鎸傚瓧骞?)
        logger.info(f"浣跨敤 {prefer_subtitle_langs} 鍖归厤宸叉湁澶栨寕瀛楀箷鏂囦欢 ...")
        external_sub_exist, external_sub_lang, exist_sub_name = self.__external_subtitle_exists(video_file,
                                                                                                prefer_subtitle_langs,
                                                                                                only_srt=True,
                                                                                                strict=strict)
        # 鑾峰彇鍐呭祵瀛楀箷
        logger.info(f"[GenSub Step 4] 妫€鏌ュ唴宓屽瓧骞?)
        logger.info(f"浣跨敤 {prefer_subtitle_langs} 鍖归厤鍐呭祵瀛楀箷鏂囦欢 ...")
        inner_sub_exist, subtitle_index, inner_sub_lang, = self.__get_video_prefer_subtitle(video_meta,
                                                                                            prefer_subtitle_langs,
                                                                                            strict=strict)

        # 浼樺厛杩斿洖绗﹀悎璇█瑕佹眰鐨勫閮ㄥ瓧骞?
        def get_sub_path():
            video_dir, _ = os.path.split(video_file)
            return os.path.join(video_dir, exist_sub_name)

        extract_subtitle = False
        if self._translate_preference == "english_only":
            if external_sub_exist:
                logger.info(f"瀛楀箷婧愬亸濂斤細{self._translate_preference} 澶栨寕瀛楀箷瀛樺湪锛屽瓧骞曡瑷€ {external_sub_lang}")
                return True, iso639.to_iso639_1(external_sub_lang), get_sub_path()
            elif inner_sub_exist:
                logger.info(f"瀛楀箷婧愬亸濂斤細{self._translate_preference} 鍐呭祵瀛楀箷瀛樺湪锛屽瓧骞曡瑷€ {inner_sub_lang}")
                extract_subtitle = True
            else:
                logger.info(f"瀛楀箷婧愬亸濂斤細{self._translate_preference} 鏈尮閰嶅埌澶栨寕鎴栧唴宓屽瓧骞?闇€瑕佷娇鐢╝sr鎻愬彇")
        else:  # english_first/origin_first
            if external_sub_exist and external_sub_lang in prefer_subtitle_langs:
                logger.info(f"瀛楀箷婧愬亸濂斤細{self._translate_preference} 澶栨寕瀛楀箷瀛樺湪锛屽瓧骞曡瑷€ {external_sub_lang}")
                return True, iso639.to_iso639_1(external_sub_lang), get_sub_path()
            elif inner_sub_exist and inner_sub_lang in prefer_subtitle_langs:
                logger.info(f"瀛楀箷婧愬亸濂斤細{self._translate_preference} 鍐呭祵瀛楀箷瀛樺湪锛屽瓧骞曡瑷€ {inner_sub_lang}")
                extract_subtitle = True
            elif external_sub_exist:
                logger.info(f"瀛楀箷婧愬亸濂斤細{self._translate_preference} 澶栨寕瀛楀箷瀛樺湪锛屽瓧骞曡瑷€ {external_sub_lang}")
                return True, iso639.to_iso639_1(external_sub_lang), get_sub_path()
            elif inner_sub_exist:
                logger.info(f"瀛楀箷婧愬亸濂斤細{self._translate_preference} 鍐呭祵瀛楀箷瀛樺湪锛屽瓧骞曡瑷€ {inner_sub_lang}")
                extract_subtitle = True
            else:
                logger.info(f"瀛楀箷婧愬亸濂斤細{self._translate_preference} 鏈尮閰嶅埌澶栨寕鎴栧唴宓屽瓧骞?闇€瑕佷娇鐢╝sr鎻愬彇")
        # 鎻愬彇鍐呭祵瀛楀箷
        if extract_subtitle:
            inner_sub_lang = iso639.to_iso639_1(inner_sub_lang) \
                if (inner_sub_lang and iso639.find(inner_sub_lang) and iso639.to_iso639_1(inner_sub_lang)) else 'und'
            extracted_sub_path = f"{subtitle_file}.{inner_sub_lang}.srt"
            Ffmpeg().extract_subtitle_from_video(video_file, extracted_sub_path, subtitle_index)
            logger.info(f"鎻愬彇瀛楀箷瀹屾垚锛歿extracted_sub_path}")
            return True, inner_sub_lang, extracted_sub_path
        # 浣跨敤asr闊宠建璇嗗埆瀛楀箷
        if audio_lang != 'auto':
            audio_lang = iso639.to_iso639_1(audio_lang)

        if not enable_asr:
            logger.info(f"鏈紑鍚闊宠瘑鍒紝涓旀棤宸叉湁瀛楀箷鏂囦欢锛岃烦杩囧悗缁鐞?)
            return False, None, None

        # 娓呯悊寮傚父閫€鍑虹殑涓存椂鏂囦欢
        tempdir = tempfile.gettempdir()
        for file in os.listdir(tempdir):
            if file.startswith('autosub-'):
                os.remove(os.path.join(tempdir, file))

        with tempfile.NamedTemporaryFile(prefix='autosub-', suffix='.wav', delete=True) as audio_file:
            # 鎻愬彇闊抽
            logger.info(f"[GenSub Step 5a] 鎻愬彇闊抽锛歿audio_file.name}")
            Ffmpeg().extract_wav_from_video(video_file, audio_file.name, audio_index)
            logger.info(f"[GenSub Step 5a] 鎻愬彇闊抽瀹屾垚")
            logger.info(f"[GenSub Step 5b] 寮€濮媁hisper璇嗗埆")

            # 鐢熸垚瀛楀箷
            logger.info(f"[GenSub Step 5] 寮€濮媁hisper璇嗗埆, 璇█ {audio_lang}")
            ret, lang = self.__do_speech_recognition(audio_lang, audio_file.name, video_file)
            if ret == "skip_chinese":
                logger.info(f"瑙嗛璇嗗埆涓轰腑鏂囦笖宸插紑鍚腑鏂囪棰戜笉缈昏瘧锛岃烦杩囧瓧骞曠敓鎴愶細{video_file}")
                self.add_skip_chinese_video(video_file)
                return False, "skip_chinese", None
            elif ret:
                logger.info(f"鐢熸垚瀛楀箷鎴愬姛锛屽師濮嬭瑷€锛歿lang}")
                # 澶嶅埗瀛楀箷鏂囦欢
                SystemUtils.copy(Path(f"{audio_file.name}.srt"), Path(f"{subtitle_file}.{lang}.srt"))
                logger.info(f"澶嶅埗瀛楀箷鏂囦欢锛歿subtitle_file}.{lang}.srt")
                # 鍒犻櫎涓存椂鏂囦欢
                os.remove(f"{audio_file.name}.srt")
                return ret, lang, Path(f"{subtitle_file}.{lang}.srt")
            elif ret is None:
                # 鏃犲０闊筹紝璺宠繃骞惰褰?
                logger.info(f"瑙嗛鏃犲０闊筹紝璺宠繃瀛楀箷鐢熸垚锛歿video_file}")
                self.add_skipped_video(video_file)
                return False, None, None
            else:
                logger.error("鐢熸垚瀛楀箷澶辫触")
                return False, None, None

    @staticmethod
    def __get_library_files(in_path, exclude_path=None):
        """
        鑾峰彇鐩綍濯掍綋鏂囦欢鍒楄〃
        """
        if not os.path.isdir(in_path):
            yield in_path
            return

        for root, dirs, files in os.walk(in_path):
            if exclude_path and any(os.path.abspath(root).startswith(os.path.abspath(path))
                                    for path in exclude_path.split(",")):
                continue

            for file in files:
                cur_path = os.path.join(root, file)
                # 妫€鏌ュ悗缂€
                if os.path.splitext(file)[-1].lower() in settings.RMT_MEDIAEXT:
                    yield cur_path

    @staticmethod
    def __load_srt(file_path):
        """
        鍔犺浇瀛楀箷鏂囦欢
        :param file_path: 瀛楀箷鏂囦欢璺緞
        :return:
        """
        with open(file_path, 'r', encoding="utf8") as f:
            srt_text = f.read()
        return list(srt.parse(srt_text))

    @staticmethod
    def __save_srt(file_path, srt_data):
        """
        淇濆瓨瀛楀箷鏂囦欢
        :param file_path: 瀛楀箷鏂囦欢璺緞
        :param srt_data: 瀛楀箷鏁版嵁
        :return:
        """
        with open(file_path, 'w', encoding="utf8") as f:
            f.write(srt.compose(srt_data))

    def __merge_srt(self, subtitle_data, max_duration=None, max_chars=None):
        """
        灏嗗崟璇嶇骇瀛楀箷鎸夊彞瀛愬悎骞讹紝骞跺己鍒舵寜鏈€澶ф椂闀?瀛楁暟鍒囧垎
        :param subtitle_data: 鍗曡瘝绾у瓧骞曞垪琛?
        :param max_duration: 姣忔鏈€澶ф椂闀匡紙绉掞級锛岄粯璁ょ敤 self._max_segment_duration
        :param max_chars: 姣忔鏈€澶у瓧绗︽暟锛岄粯璁ょ敤 self._max_segment_chars
        :return:
        """
        if max_duration is None:
            max_duration = self._max_segment_duration or 8.0
        if max_chars is None:
            max_chars = self._max_segment_chars or 30

        subtitle_data = copy.deepcopy(subtitle_data)
        merged_subtitle = []
        sentence_end = True
        end_tokens = ['.', '!', '?', '銆?, '锛?, '锛?, '銆?', '锛?', '锛?', '."', '!"', '?"']
        for index, item in enumerate(subtitle_data):
            content = item.content.replace('\n', ' ').strip()
            parse = etree.HTML(content)
            if parse is not None:
                content = parse.xpath('string(.)')
            if content == '':
                continue
            item.content = content

            if self.__is_noisy_subtitle(content):
                merged_subtitle.append(item)
                sentence_end = True
                continue

            # 璁＄畻褰撳墠瀛楀箷鏃堕暱锛堢锛?
            item_duration = (item.end - item.start).total_seconds()

            if not merged_subtitle or sentence_end:
                merged_subtitle.append(item)
                sentence_end = False
            else:
                # 寮哄埗鍒囧垎鏉′欢锛氬綋鍓嶅唴瀹?+ 鏂板唴瀹硅秴杩囧瓧鏁伴檺鍒讹紝鎴栬€呯疮璁℃椂闀胯秴杩囨渶澶ф椂闀?
                existing_len = len(merged_subtitle[-1].content)
                force_split = False
                if existing_len + len(content) > max_chars:
                    force_split = True
                elif item_duration > max_duration:
                    force_split = True
                if force_split:
                    merged_subtitle.append(item)
                    sentence_end = False
                else:
                    merged_subtitle[-1].content = f"{merged_subtitle[-1].content} {content}"
                    merged_subtitle[-1].end = item.end

            if content.endswith(tuple(end_tokens)):
                sentence_end = True
            elif len(merged_subtitle[-1].content) > 80:
                sentence_end = True
            else:
                sentence_end = False

        return merged_subtitle

    @staticmethod
    def __get_video_prefer_audio(video_meta, prefer_lang=None):
        """
        鑾峰彇瑙嗛鐨勯閫夐煶杞紝濡傛灉鏈夊闊宠建锛?浼樺厛鎸囧畾璇█闊宠建锛屽惁鍒欒幏鍙栭粯璁ら煶杞?
        :param video_meta
        :return:
        """
        if type(prefer_lang) == str and prefer_lang:
            prefer_lang = [prefer_lang]

        # 鑾峰彇棣栭€夐煶杞?
        audio_lang = None
        audio_index = None
        audio_stream = filter(lambda x: x.get('codec_type') == 'audio', video_meta.get('streams', []))
        for index, stream in enumerate(audio_stream):
            if not audio_index:
                audio_index = index
                audio_lang = stream.get('tags', {}).get('language', 'und')
            # 鑾峰彇榛樿闊宠建
            if stream.get('disposition', {}).get('default'):
                audio_index = index
                audio_lang = stream.get('tags', {}).get('language', 'und')
            # 鑾峰彇鎸囧畾璇█闊宠建
            if prefer_lang and stream.get('tags', {}).get('language') in prefer_lang:
                audio_index = index
                audio_lang = stream.get('tags', {}).get('language', 'und')
                break

        # 濡傛灉娌℃湁闊宠建锛?鍒欎笉澶勭悊
        if audio_index is None:
            logger.warn(f"娌℃湁闊宠建锛屼笉杩涜澶勭悊")
            return False, None, None

        logger.info(f"閫変腑闊宠建淇℃伅锛歿audio_index}, {audio_lang}")
        return True, audio_index, audio_lang

    @staticmethod
    def __get_video_prefer_subtitle(video_meta, prefer_lang=None, strict=False, only_srt=True):
        """
        鑾峰彇瑙嗛鐨勯閫夊瓧骞曘€備紭鍏堢骇锛?.瀛楀箷涓哄亸濂借瑷€ 2.榛樿瀛楀箷 3.绗竴涓瓧骞?
        :param video_meta: 瑙嗛鍏冩暟鎹?
        :param prefer_lang: 瀛楀箷鍋忓ソ璇█
        :param strict: 鏄惁涓ユ牸妯″紡銆傚鏋滄寚瀹氫簡鍋忓ソ璇█锛屼弗鏍兼ā寮忎笅蹇呴』杩斿洖鍋忓ソ璇█鐨勫瓧骞曘€?
        :return: (鏄惁鍛戒腑瀛楀箷锛屽瓧骞昳ndex锛屽瓧骞曡瑷€)
        """
        # from https://wiki.videolan.org/Subtitles_codecs/
        """
        https://trac.ffmpeg.org/wiki/ExtractSubtitles
        ffmpeg -codecs | grep subtitle
         DES... ass                  ASS (Advanced SSA) subtitle (decoders: ssa ass ) (encoders: ssa ass )
         DES... dvb_subtitle         DVB subtitles (decoders: dvbsub ) (encoders: dvbsub )
         DES... dvd_subtitle         DVD subtitles (decoders: dvdsub ) (encoders: dvdsub )
         D.S... hdmv_pgs_subtitle    HDMV Presentation Graphic Stream subtitles (decoders: pgssub )
         ..S... hdmv_text_subtitle   HDMV Text subtitle
         D.S... jacosub              JACOsub subtitle
         D.S... microdvd             MicroDVD subtitle
         D.S... mpl2                 MPL2 subtitle
         D.S... pjs                  PJS (Phoenix Japanimation Society) subtitle
         D.S... realtext             RealText subtitle
         D.S... sami                 SAMI subtitle
         ..S... srt                  SubRip subtitle with embedded timing
         ..S... ssa                  SSA (SubStation Alpha) subtitle
         D.S... stl                  Spruce subtitle format
         DES... subrip               SubRip subtitle (decoders: srt subrip ) (encoders: srt subrip )
         D.S... subviewer            SubViewer subtitle
         D.S... subviewer1           SubViewer v1 subtitle
         D.S... vplayer              VPlayer subtitle
         DES... webvtt               WebVTT subtitle
        """
        image_based_subtitle_codecs = (
            'dvd_subtitle',
            'dvb_subtitle',
            'hdmv_pgs_subtitle',
        )

        if prefer_lang is str and prefer_lang:
            prefer_lang = [prefer_lang]

        # 鑾峰彇棣栭€夊瓧骞?
        subtitle_lang = None
        subtitle_index = None
        subtitle_score = 0
        subtitle_stream = filter(lambda x: x.get('codec_type') == 'subtitle', video_meta.get('streams', []))
        for index, stream in enumerate(subtitle_stream):
            # 濡傛灉鏄己鍒跺瓧骞曪紝鍒欒烦杩?
            if stream.get('disposition', {}).get('forced'):
                continue
            # image-based 瀛楀箷锛岃烦杩?
            if only_srt and (
                    'width' in stream
                    or stream.get('codec_name') in image_based_subtitle_codecs
            ):
                continue
            cur_is_default = stream.get('disposition', {}).get('default')
            cur_lang = stream.get('tags', {}).get('language')
            # 璁＄畻褰撳墠瀛楀箷寰楀垎锛?.瀛楀箷涓哄亸濂借瑷€*4 2.榛樿瀛楀箷*2 3.绗竴涓瓧骞?1
            cur_score = 0
            if prefer_lang and cur_lang in prefer_lang:
                cur_score += 4
            if cur_is_default:
                cur_score += 2
            if subtitle_index is None:
                cur_score += 1
                # 绗竴涓瓧骞曞垵濮嬪寲涓洪粯璁ゅ瓧骞?
                subtitle_lang, subtitle_index, subtitle_score = cur_lang, index, cur_score
            if cur_score > subtitle_score:
                subtitle_lang, subtitle_index, subtitle_score = cur_lang, index, cur_score

        # 鏈壘鍒板瓧骞?
        if subtitle_index is None:
            logger.debug(f"娌℃湁鍐呭祵瀛楀箷")
            return False, None, None
        if strict and prefer_lang and subtitle_lang not in prefer_lang:
            logger.warn(f"涓ユ牸妯″紡,娌℃湁鍋忓ソ璇█鐨勫瓧骞?)
            return False, None, None
        logger.debug(f"鍛戒腑鍐呭祵瀛楀箷淇℃伅锛歿subtitle_index}, {subtitle_lang}, score:{subtitle_score}")
        return True, subtitle_index, subtitle_lang

    @staticmethod
    def __is_noisy_subtitle(content):
        """
        鍒ゆ柇鏄惁涓鸿儗鏅煶绛夊瓧骞?
        :param content:
        :return:
        """
        noisy_tokens = [('(', ')'), ('[', ']'), ('{', '}'), ('銆?, '銆?), ('鈾?, '鈾?), ('鈾?, '鈾?), ('鈾櫔', '鈾櫔')]
        return any(content.startswith(t[0]) and content.endswith(t[1]) for t in noisy_tokens)

    def __get_context(self, all_subs: list, target_indices: List[int], is_batch: bool) -> str:
        """閫氱敤涓婁笅鏂囪幏鍙栨柟娉?""
        min_idx = max(0, min(target_indices) - self._context_window)
        max_idx = min(len(all_subs) - 1, max(target_indices) + self._context_window) if is_batch else min(
            target_indices)

        context = []
        for idx in range(min_idx, max_idx + 1):
            status = "[寰呰瘧]" if idx in target_indices else ""
            content = all_subs[idx].content.replace('\n', ' ').strip()
            context.append(f"{status}{content}")

        return "\n".join(context)

    def __process_items(self, all_subs: list, items: list) -> list:
        """缁熶竴澶勭悊鍏ュ彛锛堟敮鎸佹壒閲忓拰鍗曟潯锛?""
        if self._enable_batch and len(items) > 1:
            return self.__process_batch(all_subs, items)
        return [self.__process_single(all_subs, item) for item in items]

    def __translate_to_zh(self, text: str, context: str = None, max_retries: int = None) -> str:
        if self._event.is_set():
            raise UserInterruptException("鐢ㄦ埛涓柇褰撳墠浠诲姟")
        if max_retries is None:
            max_retries = self._max_retries
        return self._openai.translate_to_zh(text, context, max_retries=max_retries)

    def __process_batch(self, all_subs: list, batch: list) -> list:
        """鎵归噺澶勭悊閫昏緫"""
        indices = [all_subs.index(item) for item in batch]
        context = self.__get_context(all_subs, indices, is_batch=True) if self._context_window > 0 else None
        batch_text = '\n'.join([item.content for item in batch])

        try:
            ret, result = self.__translate_to_zh(batch_text, context)
            if not ret:
                raise Exception(result)

            translated = [line.strip() for line in result.split('\n') if line.strip()]
            if len(translated) != len(batch):
                raise Exception(f"鎵规琛屾暟涓嶅尮閰?{len(translated)}/{len(batch)}")

            for item, trans in zip(batch, translated):
                item.content = f"{trans}\n{item.content}"
            self._stats['batch_success'] += len(batch)
            return batch
        except Exception as e:
            logger.warning(f"[缈昏瘧] 鎵归噺缈昏瘧澶辫触锛歿e}锛岄檷绾ч€愯缈昏瘧")
            self._stats['batch_fail'] += 1
            return [self.__process_single(all_subs, item) for item in batch]

    def __process_single(self, all_subs: List[srt.Subtitle], item: srt.Subtitle) -> srt.Subtitle:
        """鍗曟潯澶勭悊閫昏緫"""
        idx = all_subs.index(item)
        context = self.__get_context(all_subs, [idx], is_batch=False) if self._context_window > 0 else None
        success, trans = self.__translate_to_zh(item.content, context)

        if success:
            if self._subtitle_output_mode == 'chinese_only':
                item.content = trans
            else:
                item.content = f"{trans}\n{item.content}"
            self._stats['line_fallback'] += 1
            return item
        else:
            if self._subtitle_output_mode == 'chinese_only':
                item.content = f"[缈昏瘧澶辫触]"
            else:
                item.content = f"[缈昏瘧澶辫触]\n{item.content}"
            return item

    def __translate_zh_subtitle(self, source_lang: str, source_subtitle: str, dest_subtitle: str,
                                  output_mode: str = None):
        """
        缈昏瘧瀛楀箷涓轰腑鏂?
        :param source_lang: 婧愯瑷€
        :param source_subtitle: 婧愬瓧骞曟枃浠惰矾寰?
        :param dest_subtitle: 鐩爣瀛楀箷鏂囦欢璺緞
        :param output_mode: 杈撳嚭妯″紡锛?bilingual'=鍙岃锛堢炕璇?鍘熸枃锛夛紝'chinese_only'=绾腑鏂?
        """
        self._stats = {'total': 0, 'batch_success': 0, 'batch_fail': 0, 'line_fallback': 0}
        # 濡傛灉妫€娴嬪埌鐨勫瓧骞曡瑷€鏄腑鏂囷紝寮哄埗浣跨敤绾腑鏂囧瓧骞曟ā寮忥紙鍙岃妯″紡娌″繀瑕侊級
        # 浣嗗鏋?涓枃瑙嗛涓嶇炕璇?寮€鍏冲凡寮€锛屼富娴佺▼浼氬湪杩涘叆杩欓噷涔嬪墠鐩存帴璺宠繃缈昏瘧
        if not self._skip_chinese and self.__is_chinese_lang(source_lang):
            logger.info(f"妫€娴嬪瓧骞曡瑷€涓轰腑鏂囷紝寮哄埗浣跨敤绾腑鏂囧瓧骞曡緭鍑烘ā寮?)
            self._subtitle_output_mode = 'chinese_only'
        subs = self.__load_srt(source_subtitle)
        valid_subs = subs  # ASR闃舵宸茬粺涓€鍋歸ord-level鍚堝苟锛岀炕璇戞椂涓嶅啀閲嶅鍚堝苟
        
        if not valid_subs:
            logger.warning("瀛楀箷鏂囦欢涓虹┖鎴栨病鏈夋湁鏁堢殑瀛楀箷鏉＄洰锛岃烦杩囩炕璇?)
            # 鍒涘缓涓€涓┖鐨勫瓧骞曟枃浠?
            self.__save_srt(dest_subtitle, [])
            return
            
        self._stats['total'] = len(valid_subs)
        translate_start_time = time.time()
        if self._enable_batch:
            processed = self.__translate_parallel(valid_subs)
        else:
            logger.info(f"[缈昏瘧] 閫愭潯妯″紡 - 鍏?{len(valid_subs)} 鏉★紙鏁堟灉鏇村ソ锛岄€熷害杈冩參锛?)
            processed = [self.__process_single(valid_subs, item) for item in valid_subs]
        self.__save_srt(dest_subtitle, processed)
        
        # 璁＄畻缈昏瘧鑰楁椂鍜岄€熷害
        translate_elapsed = time.time() - translate_start_time
        speed = len(valid_subs) / translate_elapsed if translate_elapsed > 0 else 0
        
        # 缁熻鎶ュ憡
        batch_success_count = self._stats['batch_success']
        batch_fail_count = self._stats['batch_fail']
        line_fallback_count = self._stats['line_fallback']
        
        # 鏋勫缓鏃ュ織娑堟伅
        log_msg = f"[缈昏瘧] 瀹屾垚 - 鎬昏 {self._stats['total']} 鏉★紝鑰楁椂 {translate_elapsed:.1f} 绉掞紝閫熷害 {speed:.1f} 鏉?绉?
        if self._enable_batch:
            log_msg += f"锛屾壒閲忔垚鍔?{batch_success_count} 鎵?
            if batch_fail_count > 0:
                log_msg += f"锛屾壒閲忓け璐?{batch_fail_count} 鎵癸紙闄嶇骇鎴愬姛 {line_fallback_count} 鏉★級"
        
        logger.info(log_msg)
        
        # 鎵归噺澶辫触娆℃暟杩囧鏃惰鍛?
        if self._enable_batch and batch_fail_count > 0:
            fail_rate = batch_fail_count / (batch_success_count + batch_fail_count) if (batch_success_count + batch_fail_count) > 0 else 0
            if fail_rate > 0.5:
                logger.warning(f"[缈昏瘧] 鎵归噺澶辫触鐜囪繃楂橈紙{fail_rate:.0%}锛夛紝寤鸿妫€鏌ワ細1) LLM API绋冲畾鎬?2) 闄嶄綆batch_size 3) 妫€鏌rompt鏍煎紡")

    def __translate_parallel(self, valid_subs: list):
        """
        骞惰缈昏瘧瀛楀箷锛屼娇鐢?ThreadPoolExecutor 澶氱嚎绋嬪苟鍙戝鐞嗘壒娆?
        鎵规鎸夊師濮嬬储寮曟帓搴忓悎骞讹紝淇濊瘉椤哄簭姝ｇ‘
        """
        total = len(valid_subs)
        batch_size = self._batch_size
        workers = self._parallel_workers

        # 灏嗗瓧骞曟媶鍒嗕负鎵规锛屾瘡鎵瑰寘鍚?(鍏ㄥ眬绱㈠紩, 瀛楀箷瀵硅薄)
        batches = []
        for i in range(0, total, batch_size):
            batch_items = valid_subs[i:i + batch_size]
            # 寤虹珛 鍏ㄥ眬绱㈠紩->瀛楀箷瀵硅薄 鐨勬槧灏?
            batch_map = {}
            for j, item in enumerate(batch_items):
                batch_map[i + j] = item  # 鐢ㄥ叏灞€绱㈠紩 i+j
            batches.append((i, batch_map))

        logger.info(f"[缈昏瘧] 骞惰妯″紡 - 鍏?{len(batches)} 鎵规锛屾瘡鎵?{batch_size} 鏉★紝骞跺彂 {workers} 绾跨▼")

        results = {}  # 鏈€缁堢粨鏋滐細鍏ㄥ眬idx -> 澶勭悊鍚庣殑瀛楀箷瀵硅薄

        def process_batch(batch_start_idx, batch_map, stats):
            """鍦ㄥ瓙绾跨▼涓墽琛岋細灏濊瘯鎵归噺缈昏瘧锛屽け璐ュ垯闄嶇骇鍗曡"""
            batch_list = list(batch_map.values())
            indices = list(batch_map.keys())

            # 灏濊瘯鎵归噺缈昏瘧锛圝SON缁撴瀯鍖栬緭鍑猴紝鎸塱d鏍￠獙锛?
            try:
                batch_texts = [item.content.strip() for item in batch_list]
                ret, translations = self._openai.translate_batch_to_zh(batch_texts)
                # 涓ユ牸妫€鏌ワ細ret=True 涓?translations 涓嶄负绌?涓?鎵€鏈夋潯鐩潎闈?None
                if ret and translations and all(t is not None for t in translations):
                    for item, trans in zip(batch_list, translations):
                        if self._subtitle_output_mode == 'chinese_only':
                            item.content = trans
                        else:
                            item.content = f"{trans}\n{item.content}"
                    stats["batch_ok"] += 1
                    stats["line_ok"] += len(translations)
                    return {gidx: batch_map[gidx] for gidx in indices}
            except Exception as e:
                logger.debug(f"鎵规 {batch_start_idx} 鎵归噺缈昏瘧寮傚父锛岄檷绾у崟琛岋細{e}")

            # 闄嶇骇锛氶€愯缈昏瘧锛坒allback鍗曟潯锛屼粎鍦ㄦ壒娆″け璐ュ悗鎵ц锛?
            # 閫愭潯璋冪敤缈昏瘧锛堜笉璧版壒閲忥級锛屽け璐ユ椂鏈€澶氬啀閲嶈瘯1娆★紝閬垮厤瀵瑰凡澶辫触鐨勬潯鐩棤闄愰噸璇?
            line_ok_count = 0
            for gidx in indices:
                item = batch_map[gidx]
                context = self.__get_context(valid_subs, [gidx], is_batch=False) if self._context_window > 0 else None
                # 鍗曟潯缈昏瘧锛宮ax_retries=1锛堝彧閲嶈瘯1娆★紝閬垮厤杩囧害璋冪敤锛?
                success, trans = self.__translate_to_zh(item.content, context, max_retries=1)
                if success:
                    line_ok_count += 1
                    if self._subtitle_output_mode == 'chinese_only':
                        item.content = trans
                    else:
                        item.content = f"{trans}\n{item.content}"
                else:
                    # 鍗曟潯缈昏瘧澶辫触锛屼笉閲嶈瘯锛堥伩鍏嶆氮璐硅皟鐢ㄦ鏁帮級
                    if self._subtitle_output_mode == 'chinese_only':
                        item.content = "[缈昏瘧澶辫触]"
                    else:
                        item.content = f"[缈昏瘧澶辫触]\n{item.content}"
            stats["line_ok"] += line_ok_count
            stats["batch_fail"] += 1
            logger.info(f"[缈昏瘧] 鎵规 {batch_start_idx} 闄嶇骇閫愯瀹屾垚锛歿line_ok_count}/{len(indices)} 鏉℃垚鍔?)
            return {gidx: batch_map[gidx] for gidx in indices}

        # 缁熻璁℃暟鍣紙鍦ㄥ绾跨▼闂村畨鍏ㄥ叡浜級
        stats = {"batch_ok": 0, "batch_fail": 0, "line_ok": 0}
        last_report_pct = -10  # 涓婃鎶ュ憡杩涘害鐧惧垎姣旓紝鍒濆-10纭繚绗竴鏉℃墦鍗?

        # 骞惰鎵ц
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_batch, start_idx, bmap, stats): start_idx
                       for start_idx, bmap in batches}

            for future in as_completed(futures):
                batch_results = future.result()
                results.update(batch_results)
                done_count = len(results)
                # 姣?0%鎵撳嵃涓€娆¤繘搴?
                pct = int(done_count / total * 100) if total > 0 else 0
                if pct >= last_report_pct + 10:
                    logger.info(f"[缈昏瘧] 杩涘害: {pct}% ({done_count}/{total}) - 宸插畬鎴?{done_count} 鏉?)
                    last_report_pct = pct

        # 鎸夌储寮曟帓搴忚繑鍥?
        processed = [results[i] for i in sorted(results.keys())]
        self._stats['batch_success'] = stats["batch_ok"]
        self._stats['batch_fail'] = stats["batch_fail"]
        self._stats['line_fallback'] = stats["line_ok"]
        return processed

    @staticmethod
    def __external_subtitle_exists(video_file, prefer_langs=None, only_srt=False, strict=True):
        """
        澶栭儴瀛楀箷鏂囦欢鏄惁瀛樺湪,鏀寔澶氱鏍煎紡鍙婃墿灞曢渶姹傘€?
        :param video_file: 瑙嗛鏂囦欢璺緞
        :param prefer_langs: 鍋忓ソ璇█鍒楄〃锛屾敮鎸佸崟涓瑷€瀛楃涓叉垨鍒楄〃
        :param only_srt: 鏄惁鍙尮閰峴rt鏍煎紡鐨勫瓧骞?
        :param strict: 鏄惁涓ユ牸鍖归厤鍋忓ソ璇█.褰撲笉瀛樺湪鍋忓ソ璇█瀛楀箷浣嗗瓨鍦ㄥ叾浠栬瑷€瀛楀箷鏃?鏄惁杩斿洖鍏朵粬瀛楀箷
        :return: 鍏冪粍 (鏄惁瀛樺湪, 妫€娴嬪埌鐨勮瑷€, 鏂囦欢鍚?
        """
        video_dir, video_name = os.path.split(video_file)
        video_name, video_ext = os.path.splitext(video_name)

        if prefer_langs and type(prefer_langs) == str:
            prefer_langs = [prefer_langs]

        metadata_flags = ["default", "forced", "foreign", "sdh", "cc", "hi", "鏈虹炕"]
        if only_srt:
            subtitle_extensions = [".srt"]
        else:
            subtitle_extensions = [".srt", ".sub", ".ass", ".ssa", ".vtt"]

        def parse_props(props):
            """
            瑙ｆ瀽瀛楀箷灞炴€т俊鎭紝鎻愬彇璇█鍜屽厓鏁版嵁鏍囪銆?
            :param props: 灞炴€у瓧绗︿覆
            :return: (璇█, 鍏冩暟鎹垪琛?
            """
            parts = props.split(".")
            if len(parts) < 1:
                return None, []

            cur_subtitle_lang = None
            cur_metadata = []
            # 鍊掑簭閬嶅巻鏂囦欢鍚嶄腑鐨勬爣璁?
            for i in range(len(parts) - 1, -1, -1):
                part = parts[i]
                if part in metadata_flags:
                    cur_metadata.append(part)
                elif cur_subtitle_lang is None:
                    try:
                        iso639.to_iso639_1(part)
                    except iso639.NonExistentLanguageError:
                        continue
                    else:
                        cur_subtitle_lang = iso639.to_iso639_1(part)  # 璁板綍鏈€鍚庝竴涓瑷€鏍囪

            return cur_subtitle_lang, cur_metadata

        # 澶囬€夌殑瀛楀箷璇█.褰搒trict=False鏃剁敓鏁? 鐢ㄤ簬鍦ㄦ湭鎵惧埌鍋忓ソ璇█鏃惰繑鍥炲叾浠栬瑷€
        second_lang = None
        second_file = None
        # 妫€鏌ュ瓧骞曟枃浠?
        for file in os.listdir(video_dir):
            if not file.startswith(video_name):
                continue

            # 妫€鏌ユ墿灞曞悕鏄惁鍦ㄦ敮鎸佽寖鍥村唴
            _, ext = os.path.splitext(file)
            if ext.lower() not in subtitle_extensions:
                continue

            # 鎻愬彇鏂囦欢鍚嶄腑鐨勮瑷€鍜屽厓鏁版嵁淇℃伅
            props_str = file[len(video_name) + 1: -len(ext)] if file.startswith(video_name + ".") else ""
            subtitle_lang, metadata = parse_props(props_str)

            # 濡傛灉娌℃湁璇█鏍囪锛岃烦杩?
            if not subtitle_lang:
                continue

            # 濡傛灉鎸囧畾浜嗗亸濂借瑷€
            if prefer_langs:
                if subtitle_lang in prefer_langs:
                    return True, subtitle_lang, file
                else:
                    second_lang = subtitle_lang
                    second_file = file
            else:
                # 鏈寚瀹氬亸濂借瑷€锛屾壘鍒扮殑绗竴涓瓧骞曞嵆杩斿洖
                return True, subtitle_lang, file
        if not strict and second_lang:
            return True, second_lang, second_file
        return False, None, None

    def __target_subtitle_exists(self, video_file):
        """
        鐩爣瀛楀箷鏂囦欢鏄惁瀛樺湪
        :param video_file:
        :return:
        """
        if self._translate_zh:
            prefer_langs = ['zh', 'chi', 'zh-CN', 'chs', 'zhs', 'zh-Hans', 'zhong', 'simp', 'cn']
            strict = True
        else:
            if self._translate_preference == "english_first":
                prefer_langs = ['en', 'eng']
                strict = False
            elif self._translate_preference == "english_only":
                prefer_langs = ['en', 'eng']
                strict = True
            else:
                prefer_langs = None
                strict = False

        exist, lang, _ = self.__external_subtitle_exists(video_file, prefer_langs, strict=strict)
        if exist:
            return True

        video_meta = Ffmpeg().get_video_metadata(video_file)
        if not video_meta:
            return False
        ret, subtitle_index, subtitle_lang = self.__get_video_prefer_subtitle(video_meta, prefer_lang=prefer_langs,
                                                                              only_srt=False)
        if ret and subtitle_lang in prefer_langs:
            return True

        return False

    def get_form(self) -> Tuple[List[dict], Dict[str, Any]]:
        """
        鎷艰鎻掍欢閰嶇疆椤甸潰锛岄渶瑕佽繑鍥炰袱鍧楁暟鎹細1銆侀〉闈㈤厤缃紱2銆佹暟鎹粨鏋?
        """
        return [
            {
                'component': 'VForm',
                'content': [
                    {                        'component': 'VRow',                        'content': [                            {                                'component': 'VCol',                                'props': {'cols': 12, 'md': 4},                                'content': [{'component': 'VSwitch', 'props': {'model': 'enabled', 'label': '鍚敤鎻掍欢', 'color': 'primary'}}]                            },                            {                                'component': 'VCol',                                'props': {'cols': 12, 'md': 4},                                'content': [{'component': 'VSwitch', 'props': {'model': 'send_notify', 'label': '鍙戦€侀€氱煡'}}]                            },                            {                                'component': 'VCol',                                'props': {'cols': 12, 'md': 4},                                'content': [{'component': 'VSwitch', 'props': {'model': 'clear_history', 'label': '娓呯悊鍘嗗彶璁板綍'}}]                            }                        ]                    },                    {                        'component': 'VRow',                        'content': [                            {                                'component': 'VCol',                                'props': {'cols': 12},                                'content': [{                                    'component': 'VTextarea',                                    'props': {                                        'model': 'path_whitelist',                                        'label': '鐩戞帶璺緞锛堟瘡琛屼竴涓級',                                        'rows': 3,                                        'placeholder': '/mnt/media/movies\n/downloads',                                        'hint': '鐩綍鍙樺寲鏃惰嚜鍔ㄨЕ鍙戝瓧骞曠敓鎴?                                    }                                }]                            }                        ]                    },                    {                        'component': 'VRow',
                        'content': [
                            {
                                'component': 'VCol',
                                'props': {'cols': 12, 'md': 3},
                                'content': [{'component': 'VSwitch', 'props': {'model': 'process_new_only', 'label': '浠呭鐞嗘柊澧炶棰?, 'hint': '鍏抽棴鍒欏鐞嗚矾寰勪笅鎵€鏈夎棰?}}]
                            },
                            {
                                'component': 'VCol',
                                'props': {'cols': 12, 'md': 3},
                                'content': [{'component': 'VSwitch', 'props': {'model': 'run_now', 'label': '鎵嬪姩鎵ц涓€娆?, 'color': 'secondary'}}]
                            },
                            {
                                'component': 'VCol',
                                'props': {'cols': 12, 'md': 3},
                                'content': [{'component': 'VSwitch', 'props': {'model': 'translate_zh', 'label': '澶栬缈昏瘧鎴愪腑鏂?, 'hint': '浣跨敤openai澶фā鍨嬬炕璇?}}]
                            },
                            {
                                'component': 'VCol',
                                'props': {'cols': 12, 'md': 3},
                                'content': [{'component': 'VSwitch', 'props': {'model': 'skip_chinese', 'label': '涓枃瑙嗛涓嶇炕璇?, 'hint': 'Whisper妫€娴嬪埌涓枃鏃惰烦杩囩炕璇戝苟璁板綍锛屼笅娆¤嚜鍔ㄨ烦杩?}}]
                            }
                        ]
                    },
                    {'component': 'VRow',                        'props': {'v-show': 'run_now'},                        'content': [                            {                                'component': 'VCol',                                'props': {'cols': 12},                                'content': [{                                    'component': 'VTextarea',                                    'props': {                                        'model': 'path_list',                                        'label': '濯掍綋璺緞锛堟墜鍔ㄦ墽琛屾椂浣跨敤锛?,                                        'rows': 3,                                        'placeholder': '缁濆璺緞锛屾瘡琛屼竴涓紝鏀寔鏂囦欢鍜屾枃浠跺す'                                    }                                }]                            }                        ]                    },                    {                        'component': 'VExpansionPanels',                        'props': {'variant': 'accordion', 'multiple': True},                        'content': [                            {                                'component': 'VExpansionPanel',                                'content': [                                    {                                        'component': 'VExpansionPanelTitle',                                        'text': 'Whisper闊宠建杞瓧骞曡缃?                                    },                                    {                                        'component': 'VExpansionPanelText',                                        'content': [                                            {                                                'component': 'VRow',                                                'content': [                                                    {                                                        'component': 'VCol',                                                        'props': {'cols': 12, 'md': 6},                                                        'content': [{'component': 'VSwitch', 'props': {'model': 'enable_asr', 'label': '鍏佽ASR鐢熸垚瀛楀箷'}}]                                                    },                                                    {                                                        'component': 'VCol',                                                        'props': {'cols': 12, 'md': 6},                                                        'content': [{'component': 'VSwitch', 'props': {'model': 'auto_detect_language', 'label': '鑷姩妫€娴嬭瑷€', 'hint': '鐢眞hisper鑷姩璇嗗埆锛岃€岄潪瑙嗛鍏冩暟鎹?}}]                                                    }                                                ]                                            },                                            {                                                'component': 'VRow',                                                'content': [                                                    {                                                        'component': 'VCol',                                                        'props': {'cols': 12, 'md': 6},                                                        'content': [{                                                            'component': 'VSelect',                                                            'props': {                                                                'model': 'faster_whisper_model',                                                                'label': 'Whisper妯″瀷',
                                                                'hint': 'Whisper妯″瀷(鑷€?鏁堟灉瓒婂ソ,鏃堕棿瓒婁箙)',                                                                'items': [                                                                    'tiny', 'base', 'small', 'medium', 'large-v3',                                                                    {'title': 'large-v3-turbo', 'value': 'deepdml/faster-whisper-large-v3-turbo-ct2'},                                                                ]                                                            }                                                        }]                                                    },                                                    {                                                        'component': 'VCol',                                                        'props': {'cols': 12, 'md': 6},                                                        'content': [{                                                            'component': 'VSelect',                                                            'props': {                                                                'model': 'subtitle_output_mode',                                                                'label': '瀛楀箷杈撳嚭妯″紡',                                                                'items': [                                                                    {'title': '鍙岃瀛楀箷锛堢炕璇?鍘熸枃锛?, 'value': 'bilingual'},                                                                    {'title': '绾腑鏂囧瓧骞?, 'value': 'chinese_only'}                                                                ]                                                            }                                                        }]                                                    }                                                ]                                            },                                            {                                                'component': 'VRow',                                                'content': [                                                    {                                                        'component': 'VCol',                                                        'props': {'cols': 12, 'md': 4},                                                        'content': [{'component': 'VTextField', 'props': {'model': 'max_segment_duration', 'label': '姣忔瀛楀箷鏈€澶ф椂闀匡紙绉掞級', 'placeholder': '8'}}]                                                    },                                                    {                                                        'component': 'VCol',                                                        'props': {'cols': 12, 'md': 4},                                                        'content': [{'component': 'VTextField', 'props': {'model': 'max_segment_chars', 'label': '姣忔瀛楀箷鏈€澶у瓧绗︽暟', 'placeholder': '50', 'default': '50'}}]                                                    },                                                    {                                                        'component': 'VCol',                                                        'props': {'cols': 12, 'md': 4},                                                        'content': [{'component': 'VTextField', 'props': {'model': 'file_size', 'label': '鏂囦欢鏈€灏忓ぇ灏忥紙MB锛?, 'placeholder': '榛樿10'}}]                                                    }                                                ]                                            },                                            {                                                'component': 'VRow',                                                'content': [                                                    {                                                        'component': 'VCol',                                                        'props': {'cols': 12, 'md': 6},                                                        'content': [{                                                            'component': 'VSelect',                                                            'props': {                                                                'model': 'translate_preference',                                                                'label': '瀛楀箷婧愯瑷€鍋忓ソ',                                                                'items': [                                                                    {'title': '浠呰嫳鏂?, 'value': 'english_only'},                                                                    {'title': '鑻辨枃浼樺厛', 'value': 'english_first'},                                                                    {'title': '鍘熼煶浼樺厛', 'value': 'origin_first'}                                                                ]                                                            }                                                        }]                                                    },                                                    {                                                        'component': 'VCol',                                                        'props': {'cols': 12, 'md': 6},                                                        'content': [{'component': 'VSwitch', 'props': {'model': 'proxy', 'label': '浣跨敤浠ｇ悊涓嬭浇妯″瀷', 'hint': '闇€閰嶇疆MP PROXY鐜鍙橀噺'}}]                                                    }                                                ]                                            }                                        ]                                    }                                ]                            },                            {                                'component': 'VExpansionPanel',                                'props': {'v-show': 'translate_zh'},                                'content': [                                    {                                        'component': 'VExpansionPanelTitle',                                        'text': '缈昏瘧鍙傛暟璁剧疆'                                    },                                    {                                        'component': 'VExpansionPanelText',                                        'content': [                                            {                                                'component': 'VRow',                                                'content': [                                                    {                                                        'component': 'VCol',                                                        'props': {'cols': 12, 'md': 4},                                                        'content': [{'component': 'VTextField', 'props': {'model': 'context_window', 'label': '涓婁笅鏂囩獥鍙ｅぇ灏?, 'placeholder': '5'}}]                                                    },                                                    {                                                        'component': 'VCol',                                                        'props': {'cols': 12, 'md': 4},                                                        'content': [{'component': 'VTextField', 'props': {'model': 'max_retries', 'label': 'LLM璇锋眰閲嶈瘯娆℃暟', 'placeholder': '3'}}]                                                    },                                                    {                                                        'component': 'VCol',                                                        'props': {'cols': 12, 'md': 4},                                                        'content': [{'component': 'VSwitch', 'props': {'model': 'enable_batch', 'label': '鍚敤鎵归噺缈昏瘧', 'hint': '寮€鍚細閫熷害鏇村揩锛岃蛋鎵归噺鎻愮ず璇嶏紱鍏抽棴锛氶€愭潯缈昏瘧锛屾晥鏋滄洿濂戒絾鏇存參'}}]                                                    }                                                ]                                            },                                            {                                                'component': 'VRow',                                                'content': [                                                    {                                                        'component': 'VCol',                                                        'props': {'cols': 12, 'md': 6, 'v-show': 'enable_batch'},                                                        'content': [{'component': 'VTextField', 'props': {'model': 'batch_size', 'label': '姣忔壒缈昏瘧琛屾暟', 'placeholder': '20 (寤鸿涓嶈秴杩?0)'}}]                                                    },                                                    {                                                        'component': 'VCol',                                                        'props': {'cols': 12, 'md': 6, 'v-show': 'enable_batch'},                                                        'content': [{'component': 'VTextField', 'props': {'model': 'parallel_workers', 'label': '骞跺彂绾跨▼鏁?, 'placeholder': '5', 'default': '5'}}]                                                    }                                                ]                                            }                                        ]                                    }                                ]                            },                            {                                'component': 'VExpansionPanel',                                'props': {'v-show': 'translate_zh'},                                'content': [                                    {                                        'component': 'VExpansionPanelTitle',                                        'text': '缈昏瘧妯″瀷api璁剧疆'                                    },                                    {                                        'component': 'VExpansionPanelText',                                        'content': [                                            {                                                'component': 'VRow',                                                'content': [                                                    {                                                        'component': 'VCol',                                                        'props': {'v-show': False, 'cols': 12, 'md': 4},                                                        'content': [{'component': 'VSwitch', 'props': {'model': 'use_chatgpt', 'label': '澶嶇敤ChatGPT鎻掍欢閰嶇疆'}}]                                                    },                                                    {                                                        'component': 'VTextField',                                                        'props': {                                                            'model': 'use_chatgpt_trigger',                                                            'class': 'd-none',                                                            'text': 'trigger',                                                            'change': 'use_chatgpt_trigger = use_chatgpt ? 1 : 0'                                                        }                                                    },                                                    {                                                        'component': 'VCol',                                                        'props': {'cols': 12, 'md': 4},                                                        'content': [{'component': 'VSwitch', 'props': {'model': 'openai_proxy', 'label': '浣跨敤浠ｇ悊鏈嶅姟鍣?}}]                                                    },                                                    {                                                        'component': 'VCol',                                                        'props': {'cols': 12, 'md': 4},                                                        'content': [{'component': 'VSwitch', 'props': {'model': 'compatible', 'label': '鍏煎妯″紡'}}]                                                    }                                                ]                                            },                                            {                                                'component': 'VRow',                                                'content': [                                                    {                                                        'component': 'VCol',                                                        'props': {'cols': 12, 'md': 4},                                                        'content': [{'component': 'VTextField', 'props': {'model': 'openai_url', 'label': 'API URL', 'placeholder': 'https://api.siliconflow.cn'}}]                                                    },                                                    {                                                        'component': 'VCol',                                                        'props': {'cols': 12, 'md': 4},                                                        'content': [{'component': 'VTextField', 'props': {'model': 'openai_key', 'label': 'API瀵嗛挜', 'placeholder': 'sk-xxx'}}]                                                    },                                                    {                                                        'component': 'VCol',                                                        'props': {'cols': 12, 'md': 4},                                                        'content': [{'component': 'VTextField', 'props': {'model': 'openai_model', 'label': '鑷畾涔夋ā鍨?, 'placeholder': 'inclusionAI/Ling-mini-2.0'}}]                                                    }                                                ]                                            }                                        ]                                    }                                ]                            }                        ]                    },                    {                        'component': 'VRow',                        'content': [                            {                                'component': 'VCol',                                'props': {'cols': 12},                                'content': [{                                    'component': 'VAlert',                                    'props': {'type': 'success', 'variant': 'tonal'},                                    'content': [                                        {                                            'component': 'a',                                            'props': {'href': 'https://github.com/jianji112/MoviePilot-Plugins/blob/main/README.md#%E7%94%B3%E8%AF%B7%E7%A1%85%E5%9F%BA%E6%B5%81%E5%8A%A8-api', 'target': '_blank'},                                            'content': [{'component': 'u', 'text': 'API鐢宠鏁欑▼'}]                                        },                                        {                                            'component': 'span',                                            'text': ' | 璇︾粏璇存槑锛?                                        },                                        {                                            'component': 'a',                                            'props': {'href': 'https://github.com/jianji112/MoviePilot-Plugins/blob/main/plugins/autosubv3/README.md', 'target': '_blank'},                                            'content': [{'component': 'u', 'text': 'README'}]                                        }                                    ]                                }]                            }                        ]                    }                ]
            }
        ], {
            "enabled": False,
            "clear_history": False,
            "send_notify": False,
            "listen_transfer_event": True,
            "process_new_only": True,
            "path_whitelist": "",
            "run_now": False,
            "path_list": "",
            "file_size": "10",
            "translate_preference": "english_first",
            "translate_zh": True,
            "enable_asr": True,
            "auto_detect_language": False,
            "skip_chinese": False,
            "max_segment_duration": 8.0,
            "max_segment_chars": 50,
            "faster_whisper_model": "base",
            "proxy": True,
                        "openai_proxy": False,
            "compatible": False,
            "openai_url": "https://api.siliconflow.cn",
            "openai_key": None,
            "openai_model": "inclusionAI/Ling-flash-2.0",
            "context_window": 5,
            "max_retries": 3,
            "enable_merge": False,
            "subtitle_output_mode": "bilingual",
            "enable_batch": True,
            "batch_size": 20,
            "parallel_workers": 10,
        }

    def get_api(self) -> List[Dict[str, Any]]:
        pass

    def get_page(self) -> List[dict]:
        # 鍔犺浇浠诲姟骞舵寜娣诲姞鏃堕棿鍊掑簭鎺掑垪
        tasks: Dict[str, TaskItem] = self.load_tasks()
        sorted_tasks = sorted(
            tasks.items(),
            key=lambda x: x[1].add_time,
            reverse=True
        )

        status_classes = {
            TaskStatus.PENDING: "text-info",
            TaskStatus.IN_PROGRESS: "text-warning",
            TaskStatus.COMPLETED: "text-success",
            TaskStatus.IGNORED: "text-muted",
            TaskStatus.NO_AUDIO: "text-muted",
            TaskStatus.FAILED: "text-error"
        }

        rows = []
        for task_id, task in sorted_tasks:
            source_label = {
                TaskSource.MANUAL: "鎵嬪姩娣诲姞",
                TaskSource.EVENT: "鍏ュ簱瑙﹀彂"
            }.get(task.source, task.source)

            status_text = {
                TaskStatus.PENDING: "绛夊緟涓?,
                TaskStatus.IN_PROGRESS: "澶勭悊涓?,
                TaskStatus.COMPLETED: "宸插畬鎴?,
                TaskStatus.IGNORED: "宸插拷鐣?,
                TaskStatus.NO_AUDIO: "鏃犲０闊宠烦杩?,
                TaskStatus.FAILED: "澶辫触"
            }.get(task.status, task.status)

            status_class = status_classes.get(task.status, "")

            add_time_str = task.add_time.strftime("%Y-%m-%d %H:%M:%S")
            complete_time_str = (
                task.complete_time.strftime("%Y-%m-%d %H:%M:%S")
                if task.complete_time else "-"
            )

            rows.append({
                "component": "tr",
                "props": {"class": "text-sm"},
                "content": [
                    {"component": "td", "text": add_time_str},
                    {"component": "td", "text": task.video_file},
                    {"component": "td", "text": source_label},
                    {"component": "td", "text": complete_time_str},
                    {
                        "component": "td",
                        "props": {"class": status_class},
                        "text": status_text
                    },
                ],
            })

        return [
            {
                "component": "VRow",
                "content": [
                    {
                        "component": "VCol",
                        "props": {"cols": 12},
                        "content": [
                            {
                                "component": "VTable",
                                "props": {"hover": True},
                                "content": [
                                    {
                                        "component": "thead",
                                        "content": [
                                            {
                                                "component": "th",
                                                "props": {"class": "text-start ps-4"},
                                                "text": "娣诲姞鏃堕棿"
                                            },
                                            {
                                                "component": "th",
                                                "props": {"class": "text-start ps-4"},
                                                "text": "瑙嗛鏂囦欢"
                                            },
                                            {
                                                "component": "th",
                                                "props": {"class": "text-start ps-4"},
                                                "text": "鏉ユ簮"
                                            },
                                            {
                                                "component": "th",
                                                "props": {"class": "text-start ps-4"},
                                                "text": "瀹屾垚鏃堕棿"
                                            },
                                            {
                                                "component": "th",
                                                "props": {"class": "text-start ps-4"},
                                                "text": "鐘舵€?
                                            },
                                        ]
                                    },
                                    {"component": "tbody", "content": rows}
                                ]
                            }
                        ]
                    }
                ]
            }
        ]

    @staticmethod
    def get_command() -> List[Dict[str, Any]]:
        pass

    def get_state(self) -> bool:
        """
        鑾峰彇鎻掍欢鐘舵€侊紝濡傛灉鎻掍欢姝ｅ湪杩愯锛?鍒欒繑鍥濼rue
        """
        return self._running

    def stop_service(self):
        """
        閫€鍑烘彃浠?
        """
        if self._running:
            self._event.set()
        if self._consumer_thread and self._consumer_thread.is_alive():
            logger.info("姝ｅ湪鍋滄褰撳墠浠诲姟...")
            # self._consumer_thread.join(timeout=3)
            self._consumer_thread.join()

        if self._task_queue:
            while not self._task_queue.empty():
                self._task_queue.get_nowait()
                self._task_queue.task_done()
            logger.info("浠诲姟闃熷垪宸叉竻绌?)
        if self._tasks is not None:
            for task_id in list(self._tasks.keys()):
                task = self._tasks[task_id]
                if task.status == TaskStatus.PENDING or task.status == TaskStatus.IN_PROGRESS:
                    task.status = TaskStatus.FAILED
                    task.complete_time = datetime.now()
            self.save_tasks()  # 鎸佷箙鍖栨洿鏂板悗鐨勪换鍔″垪琛?
        if self._observer:
            try:
                self._observer.stop()
                self._observer.join(timeout=5)
                logger.info("鐩綍鐩戞帶宸插仠姝?)
            except Exception:
                pass
            self._observer = None
        self._running = False
        self._event.clear()
        logger.info(f"鑷姩瀛楀箷鐢熸垚鏈嶅姟宸插仠姝?)
