import os
import logging
import time
from typing import Dict, List, Optional
from threading import Lock
import fnmatch
from pathlib import Path
import psutil

from services.code_embeddings_service import CodeEmbeddingsService
from config import config

logger = logging.getLogger(__name__)

class CodeContextService:
    _instance = None
    _lock = Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式确保只有一个实例"""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """
        初始化代码上下文服务
        """
        if hasattr(self, 'repo_root'):  # 单例模式下避免重复初始化
            return
            
        # 验证并规范化仓库路径
        repo_path = Path(config.REPO_ROOT).resolve()
        cwd = Path(os.getcwd()).resolve()
        
        # 确保仓库路径是当前工作目录或其子目录
        if not str(repo_path).startswith(str(cwd)):
            logger.warning(f"仓库路径 {repo_path} 不在当前工作目录 {cwd} 下，将使用当前目录")
            repo_path = cwd
            
        self.repo_root = str(repo_path)
        logger.info(f"使用代码仓库路径: {self.repo_root}")
        
        self.cache_ttl = config.CACHE_TTL
        self.file_cache: Dict[str, Dict] = {}  # {file_path: {'content': str, 'last_modified': float, 'last_loaded': float}}
        self._lock = Lock()
        self.embeddings_service = CodeEmbeddingsService()
        self._load_repository()
        self._build_embeddings_index()

    def _should_reload_file(self, file_path: str) -> bool:
        """检查文件是否需要重新加载"""
        if file_path not in self.file_cache:
            return True
            
        cache_info = self.file_cache[file_path]
        current_time = time.time()
        
        # 检查缓存是否过期
        if current_time - cache_info['last_loaded'] > self.cache_ttl:
            return True
            
        # 检查文件是否被修改
        try:
            current_mtime = os.path.getmtime(os.path.join(self.repo_root, file_path))
            return current_mtime > cache_info['last_modified']
        except OSError:
            return True

    def _load_file(self, file_path: str) -> Optional[str]:
        """加载单个文件的内容"""
        full_path = os.path.join(self.repo_root, file_path)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 检查文件是否真的需要更新
            if file_path in self.file_cache:
                old_content = self.file_cache[file_path]['content']
                if self.embeddings_service._compute_content_hash(old_content) == \
                   self.embeddings_service._compute_content_hash(content):
                    logger.debug(f"文件内容未变化，跳过更新: {file_path}")
                    return None
                    
            self.file_cache[file_path] = {
                'content': content,
                'last_modified': os.path.getmtime(full_path),
                'last_loaded': time.time()
            }
            
            # 更新向量
            self.embeddings_service.update_file(file_path, content)
            return content
            
        except Exception as e:
            logger.error(f"加载文件失败 {full_path}: {str(e)}")
            return None

    def _load_repository(self):
        """初始加载仓库中的所有Python文件"""
        logger.info(f"开始加载代码仓库: {self.repo_root}")
        loaded_count = 0
        skipped_count = 0
        active_files = set()
        
        # 添加需要忽略的目录
        ignore_dirs = {'.git', '.venv', 'venv', '__pycache__', 'node_modules', 'vector_db'}
        
        logger.info("扫描仓库文件...")
        for root, dirs, files in os.walk(self.repo_root):
            # 跳过不需要的目录
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            # 显示当前扫描的目录
            rel_path = os.path.relpath(root, self.repo_root)
            if rel_path != '.':
                logger.debug(f"扫描目录: {rel_path}")
            
            for file in files:
                # 检查文件是否匹配模式
                if any(fnmatch.fnmatch(file, pattern) for pattern in config.REPO_FILE_PATTERNS):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.repo_root)
                    active_files.add(relative_path)
                    
                    try:
                        if self._load_file(relative_path):
                            loaded_count += 1
                            if loaded_count % 10 == 0:  # 每10个文件显示一次进度
                                logger.info(f"已加载 {loaded_count} 个文件...")
                        else:
                            skipped_count += 1
                    except Exception as e:
                        logger.error(f"加载文件失败 {relative_path}: {str(e)}")
                
                # 定期显示内存使用情况
                if (loaded_count + skipped_count) % 100 == 0:
                    process = psutil.Process(os.getpid())
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    logger.info(f"内存使用: {memory_mb:.2f} MB")
                    
        logger.info(f"完成代码仓库加载:")
        logger.info(f"- 加载文件数: {loaded_count}")
        logger.info(f"- 跳过文件数: {skipped_count}")
        
        # 清理不存在的文件的向量
        logger.info("开始清理旧向量...")
        self.embeddings_service.cleanup_old_vectors(active_files)

    def get_file_context(self, file_path: str) -> Optional[str]:
        """获取指定文件的内容，按需重新加载"""
        with self._lock:
            if self._should_reload_file(file_path):
                logger.debug(f"重新加载文件: {file_path}")
                self._load_file(file_path)
            
            cache_info = self.file_cache.get(file_path)
            return cache_info['content'] if cache_info else None

    def _build_embeddings_index(self):
        """构建代码向量索引"""
        code_files = {
            path: info['content'] 
            for path, info in self.file_cache.items()
        }
        self.embeddings_service.build_index(code_files)
        
    def get_related_files(self, file_path: str, max_files: int = 3) -> List[str]:
        """基于语义相似度获取相关文件"""
        if file_path not in self.file_cache:
            return []
            
        query_code = self.file_cache[file_path]['content']
        return self.embeddings_service.find_related_files(query_code, max_files)

    def clear_cache(self):
        """清除所有缓存"""
        with self._lock:
            self.file_cache.clear() 